#!/usr/bin/env python
"""
Phase 1 gate — synthetic ground-truth validation of the context-split coupling model.

Builds a synthetic session with four KNOWN neuron types and asks whether the
shared- and split-weight coupling models recover the ground truth:

  (a) stationary   — same coupling weights in both contexts
  (b) sign-flipped — coupling inverted between contexts  (the REMAPPING case)
  (c) uncoupled    — behaviour only, no population drive (FALSE-POSITIVE check)
  (d) gated        — coupled in visual only, same sign in both (GATING, not remapping)
  (e) time-flipped — coupling inverted between the FIRST and SECOND half of the
                     session, ignoring context entirely (TIME CONFOUND check)

Required outcomes:
  shared: gain for (a), ~0 for (b), ~0 for (c)
  split : gain for (a) AND (b), ~0 for (c)   <- (b) is the whole point
  split must NOT manufacture gain for (c)    <- the one that matters most
  split must NOT credit (e) as context-specific coupling when the blocks are ABAB

Design note 1 (learned the hard way): sklearn PCA picks svd_solver='randomized'
with random_state=None, so the coupling PCs are only reproducible when the
population spectrum is WELL SEPARATED. The synthetic population is therefore
built from latents with decreasing, well-spaced variance — a flat spectrum makes
the top-5 PCs arbitrary and the whole test meaningless.

Design note 2 (learned the harder way): real sessions are context-BLOCKED, with
only 1-5 switches per session, in an ABAB pattern (e.g. A:88 V:57 A:79 V:47).
The first version of this script used the raw generate_synthetic_session trial
order, which emits ALL visual trials then ALL audio trials — context perfectly
confounded with time. That made even the pure-latent source neurons look
context-asymmetric (FDE 0.166 vis vs 0.073 aud) because each context saw a
different stretch of the latent realisation. The session is now laid out in ABAB
blocks mirroring the real structure, and type (e) exists to detect what remains:
if a purely TIME-varying neuron is credited with context-specific coupling, the
split model is measuring non-stationarity rather than context.
"""
import io, contextlib, os, sys, time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

GLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GLM_DIR)

from testing.synthetic_session import generate_synthetic_session
from core.session_pipeline import (
    generate_predictors, stack_design_matrices, prepare_features_for_glm,
    add_coupling_predictor, fit_glm_single_session, BASE_DATA_PATH,
)
from core.predictor_builder import create_cosine_bumps

N_PCS, N_LAG_BASES, LAG_FRAMES = 5, 4, 8
SEED = 11
MOUSE, DATE = 'simulated', 'split_coupling_validation'
TYPES = ['a_stationary', 'b_signflip', 'c_uncoupled', 'd_gated', 'e_timeflip']

# Mirrors IS-2-1L/2025-03-10 (A:88 V:57 A:79 V:47) at reduced scale.
# 2 blocks per context => a context recurs, so context is NOT collinear with time.
BLOCK_PLAN = [(2, 35), (1, 35), (2, 35), (1, 35)]


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def lay_out_in_blocks(gdd, block_plan):
    """
    Re-key a glm_data_dict into ABAB context blocks matching real session
    structure. generate_synthetic_session emits trials grouped by context
    (all ctx0, then all ctx1, then all ctx2), which would make context
    perfectly confounded with time — see module docstring.
    """
    by_ctx = {}
    for k in sorted(gdd):
        by_ctx.setdefault(gdd[k]['context'], []).append(gdd[k])
    need = {}
    for ctx, n in block_plan:
        need[ctx] = need.get(ctx, 0) + n
    for ctx, n in need.items():
        if len(by_ctx.get(ctx, [])) < n:
            raise ValueError(f'block plan needs {n} trials of context {ctx}, '
                             f'have {len(by_ctx.get(ctx, []))}')
    out, ptr, i = {}, {c: 0 for c in by_ctx}, 0
    for ctx, n in block_plan:
        for _ in range(n):
            out[i] = by_ctx[ctx][ptr[ctx]]
            ptr[ctx] += 1
            i += 1
    return out


def build_synthetic_Y(X, metadata, pred_info, rng,
                      n_source=250, n_per_type=14,
                      coupling_gain=0.6, beh_gain=0.5, base_rate=15.0,
                      lat_amp=0.30, src_beh=0.10, tgt_rate=None):
    """
    Y = [sources | targets].  Sources carry the network latents (they define the
    coupling PCs); targets are the five ground-truth types driven by behaviour
    plus a LAGGED copy of latent 1.

    tgt_rate : Poisson rate for the TARGETS only (defaults to base_rate). Source
        and target noise are INDEPENDENT knobs: the sources' SNR sets how well the
        coupling PCs recover the latents (must be high), while the targets' SNR
        sets their FDE (must be LOW to match real neurons, where FDE_base ~ 0.20
        and coupling gain ~ 0.086 -- not the ~0.90 / ~0.60 a high-SNR target gives).
        Separating them lets the gate run a POWER CHECK at the real effect size
        without degrading the PCA that the whole test depends on.

    Design note 3 (the subtlest one): the source drive amplitude must stay SMALL
    so that exp(drive) is near-linear. With drive sd ~1.3, exp() is a strongly
    skewed lognormal whose covariance is dominated by a few large excursions, and
    linear PCA then does NOT recover the linear latents — measured R^2(latent |
    top-5 PCs) was only 0.43, and adding sources made it WORSE, not better. At
    lat_amp=0.30 (drive sd ~0.41) recovery is R^2 = 0.98/0.98/0.96/0.87 with a
    well-separated spectrum. Without this the coupling predictors are noise and
    the entire gate is meaningless regardless of which checks happen to pass.
    """
    n_frames = X.shape[0]
    ctx = metadata['context']

    # ── Network latents: decreasing, well-separated variance ──────────────────
    lat_scales = lat_amp * np.array([1.0, 0.55, 0.32, 0.19, 0.11])
    latents = np.empty((n_frames, len(lat_scales)))
    for k, s in enumerate(lat_scales):
        z = gaussian_filter1d(rng.normal(size=n_frames), sigma=4.0)
        latents[:, k] = s * z / z.std()

    # ── Ground-truth coupling drive, UNIQUE PER TARGET ────────────────────────
    # Design note 4: every target must get its own latent mixture, lag kernel and
    # gain. An earlier version gave all 14 targets of a type the IDENTICAL signal
    # — 14 clones then formed their own principal component, so the coupling PCs
    # literally contained cpl*(vis-aud) (R^2 = 0.83 in full Y vs 0.01 in sources
    # alone). The shared model then "recovered" remapping at 0.49 by reading a PC
    # its own clone targets had created: group-level self-prediction leakage, and
    # an artifact of the synthetic design rather than anything about the model.
    # Only latents 1-2 rise above the Poisson floor (R^2 0.97/0.97; latents 3-5
    # are ~0), so targets couple to a mixture of those two.
    lag_axis = np.arange(LAG_FRAMES, dtype=float)
    centers  = np.linspace(0, LAG_FRAMES - 1, N_LAG_BASES)
    widths   = np.full(N_LAG_BASES, (centers[1] - centers[0]) * 2.0)
    bases    = create_cosine_bumps(lag_axis, centers, widths)

    def make_cpl_drive():
        """One target's private coupling drive: random latent mix x random kernel."""
        mix    = rng.normal(size=2)
        mix   /= np.linalg.norm(mix)
        sig    = latents[:, :2] @ mix
        beta   = rng.uniform(0.1, 1.0, size=N_LAG_BASES)
        kernel = bases @ (beta / beta.sum())
        d      = np.convolve(sig, kernel, mode='full')[:n_frames]
        return (d - d.mean()) / d.std()

    # ── Behaviour drive: context-neutral groups only, so the PCs stay latent-
    #    dominated rather than context-locked. ────────────────────────────────
    beh_cols = np.concatenate([
        np.arange(pred_info[g]['start'], pred_info[g]['end'])
        for g in ('position_only', 'velocity', 'choice_spatial') if g in pred_info])

    def beh_drive(scale):
        w = rng.normal(size=len(beh_cols))
        d = X[:, beh_cols] @ w
        return scale * (d - d.mean()) / d.std()

    # ── Sources: latents dominate; weak heterogeneous behaviour ──────────────
    src = np.empty((n_frames, n_source))
    for j in range(n_source):
        drive = latents @ rng.normal(size=len(lat_scales)) + beh_drive(src_beh)
        src[:, j] = rng.poisson(base_rate * np.exp(drive - drive.var() / 2))

    # ── Targets ───────────────────────────────────────────────────────────────
    is_vis = (ctx == 1).astype(float)
    is_aud = (ctx == 2).astype(float)
    # Time sign: +1 in the first half of the session, -1 in the second. With ABAB
    # blocks this is NOT collinear with context, so a neuron whose coupling flips
    # on time alone can be told apart from one that flips on context.
    is_early = (np.arange(n_frames) < n_frames / 2).astype(float)
    time_sign = 2.0 * is_early - 1.0

    t_rate = base_rate if tgt_rate is None else tgt_rate
    tgt, labels, probes = [], [], []
    for t in TYPES:
        for _ in range(n_per_type):
            cpl_drive = make_cpl_drive()
            g = coupling_gain * rng.uniform(0.8, 1.2)
            drive = beh_drive(beh_gain)
            if t == 'a_stationary':
                drive = drive + g * cpl_drive
            elif t == 'b_signflip':
                drive = drive + g * cpl_drive * (is_vis - is_aud)
                probes.append(cpl_drive * (is_vis - is_aud))
            elif t == 'd_gated':
                drive = drive + g * cpl_drive * is_vis
            elif t == 'e_timeflip':
                drive = drive + g * cpl_drive * time_sign
            # c_uncoupled: behaviour only
            tgt.append(rng.poisson(t_rate * np.exp(drive - drive.var() / 2)))
            labels.append(t)
    tgt = np.column_stack(tgt)

    Y = np.hstack([src, tgt]).astype(float)
    target_idx = np.arange(n_source, n_source + tgt.shape[1])
    return Y, target_idx, np.array(labels), latents, np.array(probes).T


def fit_variant(X, Y_pca, Y_fit, metadata, pred_info, tag, split=False):
    """
    Fit one model variant and return per-context + train FDE.

    Y_pca : population the coupling PCs are built from (sources only).
    Y_fit : neurons actually fit (sources + targets).

    Building the PCs from the sources alone emulates the real setting, where a
    target neuron is ~1/2115 of the population and its self-contribution to the
    coupling predictors is negligible. Here the targets would otherwise be ~4% of
    Y each and — because a whole type shares one modulation pattern — they form
    their own PC, handing the SHARED model the remapped signal it is supposed to
    be unable to represent (measured leakage R^2 = 0.52-0.83). This is the
    large-population limit, not a thumb on the scale: the real model's
    exclude_col=None leakage is ~1/n and is flagged separately in the report.
    """
    from core.glm_class import make_prediction, deviance
    pi = dict(pred_info)
    Xv = X
    if tag != 'base':
        Xv, pi, _ = quiet(add_coupling_predictor, X, Y_pca, pred_info,
                          n_pcs=N_PCS, n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES,
                          split_by_context=metadata['context'] if split else None)
    pis, fgs = quiet(prepare_features_for_glm, pi)
    t0 = time.time()
    _, res, _, _ = quiet(fit_glm_single_session, Xv, Y_fit, metadata, fgs, pis,
                         MOUSE, f'{DATE}_{tag}', model_dir='synthetic_validation',
                         model_prefix=f'syn_{tag}', contexts=[1, 2])
    # train FDE (nested-model sanity: split >= shared on TRAIN)
    tr = res['combined_train_mask']
    yp = make_prediction(Xv[tr], res['weights'], res['w0'].flatten(), activation='exp')
    fd_tr, _, _ = deviance(yp, Y_fit[tr])
    res['train_frac_dev'] = np.maximum(fd_tr, 0.0)
    print(f'    {tag:8s} fitted in {time.time()-t0:.0f}s  '
          f'({Xv.shape[1]} cols, {len(fgs)} groups)')
    return res, pis


def main(coupling_gain=0.6, tgt_rate=None, beh_gain=0.5, seed=SEED, tag=''):
    rng = np.random.default_rng(seed)
    print('=' * 78)
    print(f'PHASE 1 GATE — synthetic ground-truth validation{tag}')
    print(f'  coupling_gain={coupling_gain}  tgt_rate={tgt_rate}  beh_gain={beh_gain}  seed={seed}')
    print('=' * 78)

    gdd = quiet(generate_synthetic_session, n_trials_per_context=70, seed=SEED)
    gdd = lay_out_in_blocks(gdd, BLOCK_PLAN)
    dm, _, pir = quiet(generate_predictors, gdd, include_velocity=True)
    X, _, metadata, _ = quiet(stack_design_matrices, gdd, dm, pir, neural_metric='deconv')

    m = np.isin(metadata['context'], [1, 2])
    X = X[m]
    metadata = {k: v[m] for k, v in metadata.items()}
    print(f'\n  X {X.shape} | frames: vis={(metadata["context"]==1).sum():,} '
          f'aud={(metadata["context"]==2).sum():,}')
    print(f'  block plan: {" ".join(("V" if c==1 else "A")+f":{n}" for c, n in BLOCK_PLAN)}'
          f'  (ABAB — mirrors real sessions; context NOT collinear with time)')
    # Prove context is decorrelated from time in this layout.
    ctx_a = metadata['context']
    t_norm = np.arange(len(ctx_a)) / len(ctx_a)
    print(f'  corr(context, time) = {np.corrcoef((ctx_a == 1).astype(float), t_norm)[0,1]:+.4f}'
          f'  (was ~-0.87 with the all-vis-then-all-aud layout)')

    Y, tidx, labels, latents, probes = build_synthetic_Y(
        X, metadata, pir, rng, coupling_gain=coupling_gain, beh_gain=beh_gain,
        tgt_rate=tgt_rate)
    print(f'  Y {Y.shape} | {len(tidx)} targets ({len(TYPES)} types x {len(tidx)//len(TYPES)}) '
          f'| mean rate {Y.mean():.3f}')

    # ── PRECONDITION: the coupling PCs must actually recover the latents ──────
    # This is not decoration. Two earlier versions of this gate ran to a verdict
    # (one of them "PASSING") while the top-5 PCs had R^2 ~0.43 with the latents,
    # i.e. the coupling predictors were noise. Refuse to report a verdict unless
    # the PCs are a faithful proxy for the ground-truth latents.
    Y_pca = Y[:, :tidx[0]]          # sources only — see fit_variant docstring
    from sklearn.decomposition import PCA
    from numpy.linalg import lstsq
    pca = PCA(n_components=N_PCS, svd_solver='full')
    sc = pca.fit_transform(Y_pca)
    evr = pca.explained_variance_ratio_
    A = np.c_[sc, np.ones(len(sc))]
    r2 = []
    for k in range(latents.shape[1]):
        b, _, _, _ = lstsq(A, latents[:, k], rcond=None)
        resid = latents[:, k] - A @ b
        r2.append(1 - resid.var() / latents[:, k].var())
    r2 = np.array(r2)
    print(f'  population PC spectrum : {np.round(evr, 4)}   PC1/PC2 = {evr[0]/evr[1]:.2f}')
    print(f'  R2(latent_k | top-5 PCs): {np.round(r2, 3)}')
    if r2[0] < 0.9 or evr[0] / evr[1] < 1.5:
        raise SystemExit(
            f'\n  ✗ PRECONDITION FAILED: coupling PCs do not recover the latents '
            f'(R2(lat1)={r2[0]:.2f}, PC1/PC2={evr[0]/evr[1]:.2f}).\n'
            f'    The coupling predictors would be noise and any verdict meaningless.\n'
            f'    Retune the population (drive amplitude / base_rate / n_source).\n')

    # ── PRECONDITION 2: the PCs must NOT already contain the remapped signal ──
    # If the sign-flipped targets are numerous/identical enough to form their own
    # PC, the SHARED model can read cpl*(vis-aud) straight out of the coupling
    # predictors and "recover" remapping — group-level self-prediction leakage
    # that makes the (b) contrast meaningless. Measured, not assumed.
    leak = []
    for j in range(probes.shape[1]):
        b, _, _, _ = lstsq(A, probes[:, j], rcond=None)
        leak.append(1 - (probes[:, j] - A @ b).var() / probes[:, j].var())
    leak = np.array(leak)
    print(f'  R2(remap signal | PCs)  : mean {leak.mean():.3f}  max {leak.max():.3f}  '
          f'(leakage — must be LOW)')
    if leak.mean() > 0.15:
        raise SystemExit(
            f'\n  ✗ PRECONDITION FAILED: the coupling PCs already encode the remapped\n'
            f'    signal (mean R2={leak.mean():.2f}). The sign-flipped targets have\n'
            f'    formed their own PC, so the shared model can recover (b) by reading\n'
            f'    its own targets back. Diversify targets or shrink their share of Y.\n')
    print(f'  ✓ preconditions met: PCs recover the latents, spectrum separated, no remap leakage')

    print('\n  fitting three variants ...')
    base,  _   = fit_variant(X, Y_pca, Y, metadata, pir, 'base')
    shar,  _   = fit_variant(X, Y_pca, Y, metadata, pir, 'shared', split=False)
    spl,   pis = fit_variant(X, Y_pca, Y, metadata, pir, 'split',  split=True)

    # ── Assemble per-neuron gains, exactly as Fig. 4 does (cell 43) ───────────
    def gain(model):
        return ((model['test_frac_dev_per_ctx'][1] - base['test_frac_dev_per_ctx'][1]) +
                (model['test_frac_dev_per_ctx'][2] - base['test_frac_dev_per_ctx'][2])) / 2

    df = pd.DataFrame({
        'type': labels,
        'fde_base':   base['test_frac_dev'][tidx],
        'fde_shared': shar['test_frac_dev'][tidx],
        'fde_split':  spl['test_frac_dev'][tidx],
        'gain_shared': gain(shar)[tidx],
        'gain_split':  gain(spl)[tidx],
        'gain_shared_vis': (shar['test_frac_dev_per_ctx'][1] - base['test_frac_dev_per_ctx'][1])[tidx],
        'gain_shared_aud': (shar['test_frac_dev_per_ctx'][2] - base['test_frac_dev_per_ctx'][2])[tidx],
        'gain_split_vis':  (spl['test_frac_dev_per_ctx'][1] - base['test_frac_dev_per_ctx'][1])[tidx],
        'gain_split_aud':  (spl['test_frac_dev_per_ctx'][2] - base['test_frac_dev_per_ctx'][2])[tidx],
        'train_shared': shar['train_frac_dev'][tidx],
        'train_split':  spl['train_frac_dev'][tidx],
    })

    print('\n' + '=' * 78)
    print('RESULT — mean coupling gain by ground-truth type (targets only)')
    print('=' * 78)
    print(f'  {"type":14s} {"n":>3s} {"FDE_base":>9s} {"gain_shared":>12s} {"gain_split":>11s} '
          f'{"split-shared":>13s}')
    print('  ' + '-' * 72)
    for t in TYPES:
        s = df[df['type'] == t]
        print(f'  {t:14s} {len(s):>3d} {s["fde_base"].mean():>9.4f} '
              f'{s["gain_shared"].mean():>12.4f} {s["gain_split"].mean():>11.4f} '
              f'{(s["gain_split"]-s["gain_shared"]).mean():>13.4f}')

    print('\n  per-context gain (reveals gating vs remapping):')
    print(f'  {"type":14s} {"shared_vis":>11s} {"shared_aud":>11s} {"split_vis":>10s} {"split_aud":>10s}')
    print('  ' + '-' * 62)
    for t in TYPES:
        s = df[df['type'] == t]
        print(f'  {t:14s} {s["gain_shared_vis"].mean():>11.4f} {s["gain_shared_aud"].mean():>11.4f} '
              f'{s["gain_split_vis"].mean():>10.4f} {s["gain_split_aud"].mean():>10.4f}')

    # ── Verdict ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)
    g = {t: df[df['type'] == t] for t in TYPES}
    csc = lambda t: (g[t]['gain_split'] - g[t]['gain_shared']).mean()  # context_specific_coupling
    b_sh, b_sp = g['b_signflip']['gain_shared'].mean(), g['b_signflip']['gain_split'].mean()
    a_sh = g['a_stationary']['gain_shared'].mean()
    # Thresholds are RELATIVE: "shared fails on (b)" means shared recovers far less
    # of (b) than of (a), and far less of (b) than the split model does. An absolute
    # cutoff is meaningless here because the achievable gain depends on the injected
    # coupling strength.
    checks = [
        ('shared recovers (a) stationary',        a_sh > 0.02),
        ('shared FAILS (b) vs its own (a)  '
         '(<25% of a)',                           b_sh < 0.25 * a_sh),
        ('shared FAILS (b) vs SPLIT on (b) '
         '(<25% of split)',                       b_sh < 0.25 * b_sp),
        ('shared ~0 for (c) uncoupled',           abs(g['c_uncoupled']['gain_shared'].mean()) < 0.02),
        ('SPLIT recovers (a) stationary',         g['a_stationary']['gain_split'].mean() > 0.02),
        ('SPLIT RECOVERS (b) sign-flipped  <<<',  b_sp > 0.02),
        ('SPLIT ~0 for (c) uncoupled (FP check)', abs(g['c_uncoupled']['gain_split'].mean()) < 0.02),
        ('split > shared for (b)',                csc('b_signflip') > 0.02),
        ('(e) time-flip NOT read as context-'
         'specific (< 25% of (b))',               csc('e_timeflip') < 0.25 * csc('b_signflip')),
        ('train FDE: split >= shared (nested)',   (df['train_split'] >= df['train_shared'] - 1e-3).mean() > 0.9),
    ]
    # ── Degenerate / plumbing checks ─────────────────────────────────────────
    # The prompt asks what happens when both masks are all-ones (the two groups
    # would become perfectly collinear duplicates). The builder REFUSES that
    # input rather than silently producing duplicate groups — verify the guard
    # actually fires, since a silent no-op is the failure mode that matters.
    deg = []
    try:
        quiet(add_coupling_predictor, X, Y_pca, pir, n_pcs=N_PCS,
              n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES,
              split_by_context=np.ones(len(metadata['context']), dtype=int))
        deg.append(('all-ones mask rejected (would be collinear dupes)', False))
    except ValueError as e:
        deg.append((f'all-ones mask rejected ({str(e)[:38]}...)', True))
    try:
        quiet(add_coupling_predictor, X, Y_pca, pir, n_pcs=N_PCS,
              n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES,
              split_by_context=metadata['context'] * 0)   # context 0 present
        deg.append(('unexpected context code rejected', False))
    except ValueError:
        deg.append(('unexpected context code rejected', True))
    try:
        quiet(add_coupling_predictor, X, Y_pca, pir, n_pcs=N_PCS,
              n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES,
              split_by_context=metadata['context'][:100])  # wrong length
        deg.append(('frame-count mismatch rejected', False))
    except ValueError:
        deg.append(('frame-count mismatch rejected', True))

    ok = True
    for name, passed in checks + deg:
        print(f'  [{"PASS" if passed else "FAIL"}] {name}')
        ok &= bool(passed)
    frac = (df['train_split'] >= df['train_shared'] - 1e-3).mean()
    print(f'\n  train FDE split>=shared in {frac:.0%} of targets')
    print(f'  context_specific_coupling (split-shared): '
          f'(b) remap={csc("b_signflip"):+.4f}  (e) time-flip={csc("e_timeflip"):+.4f}  '
          f'(c) uncoupled={csc("c_uncoupled"):+.4f}')
    print(f'\n  ==> {"GATE PASSED" if ok else "GATE FAILED"}')

    # Realism / power readout: how does this synthetic compare to the real pilot?
    a, b = g['a_stationary'], g['b_signflip']
    print('\n  --- realism vs the real pilot (IS-2-1L/2025-03-10) ---')
    print(f'    {"":22s} {"synthetic":>10s} {"real pilot":>11s}')
    print(f'    {"FDE_base (coupled)":22s} {a["fde_base"].mean():>10.4f} {0.1973:>11.4f}')
    print(f'    {"gain_shared (a)":22s} {a["gain_shared"].mean():>10.4f} {0.0860:>11.4f}')
    print(f'    {"csc, no remap present":22s} {csc("a_stationary"):>+10.4f} {-0.0147:>+11.4f}')
    print(f'    {"csc, TRUE remap (b)":22s} {csc("b_signflip"):>+10.4f} {"?":>11s}')
    print(f'\n    POWER: with true remapping present, csc(b) = {csc("b_signflip"):+.4f}; '
          f'with none, csc(a) = {csc("a_stationary"):+.4f}.')
    print(f'    separation = {csc("b_signflip") - csc("a_stationary"):+.4f} '
          f'({"DETECTABLE" if csc("b_signflip") - csc("a_stationary") > 0.01 else "NOT DETECTABLE"} '
          f'at this effect size)')

    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'synthetic_validation{tag.replace(" ","_")}.csv'), index=False)
    return ok, df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--coupling-gain', type=float, default=0.6,
                    help='amplitude of the injected coupling drive (0.6 = the strong-effect gate)')
    ap.add_argument('--tgt-rate', type=float, default=None,
                    help='Poisson rate for TARGETS only (default: same as sources). Lower it to '
                         'match real-neuron SNR for a power check at the real effect size.')
    ap.add_argument('--beh-gain', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--tag', type=str, default='')
    a = ap.parse_args()
    ok, _ = main(coupling_gain=a.coupling_gain, tgt_rate=a.tgt_rate,
                 beh_gain=a.beh_gain, seed=a.seed, tag=a.tag)
    sys.exit(0 if ok else 1)
