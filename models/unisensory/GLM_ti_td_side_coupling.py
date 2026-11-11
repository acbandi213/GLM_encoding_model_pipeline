#!/usr/bin/env python
"""
GLM_ti_td_side_coupling.py
==========================
Side-matched single-neuron coupling model for the snCCGP classes (mouse PPC).

Question: is a TI neuron explained better by other TI neurons that prefer the SAME choice side than by (a) TI neurons
preferring the other side, (b) TD neurons preferring the same side, (c) equally well-driven neurons outside the Choice
class?  And does a TD neuron show any of that for TD sources?

Why side-matched.  Noise correlations showed the TI ensemble is side-specific: same-side TI–TI pairs +0.106, opposite-side
+0.005 (TI_TD_noise_correlations.ipynb §8).  A pooled TI source block averages the two.

Design (per session, contexts 1 + 2)
------------------------------------
Labels : Updated_Figure_4 snCCGP quartile classes (TI / Int / TD) within the Choice class.
Side   : sign of the neuron's choice d' (trial-mean activity, choice 1 vs 0), averaged over the two contexts.
Groups : TI_L, TI_R, TD_L, TD_R.  Each is split in half (fixed per-session seed):
             fold 1 : first halves are SOURCES, second halves are TARGETS;   fold 2 : swapped.
         No target is ever a source in its own fold (no self-prediction).
Blocks : five source blocks of the SAME size n = min(smallest source half, 10):
             TI_L, TI_R, TD_L, TD_R, and CTL = non-Choice-class responsive neurons FDE-matched to the TI sources.
Predictors: SINGLE NEURONS — each source neuron's activity through 4 causal raised-cosine lag bases (8 frames), z-scored,
         one group-lasso group per source neuron.  (With ~6 neurons per block, 5 PCs would be the same space rotated.)
Fits per fold, all targets in one batch, identical seeded train/test trials (asserted, and asserted equal to GLM_uni's):
             base : [behaviour]                    refitted on this batch, so every gain is within one fitting regime
             one fit per block : [behaviour | block]
         For a target preferring side s: same-side TI = TI_s, opposite-side TI = TI_(other), likewise TD.
Targets: held-out TI and TD halves, all Int neurons, and CTL targets (non-Choice neurons FDE-matched to the TI targets,
         disjoint from every CTL source).  Int and CTL targets are fitted in both folds (one row per fold).
No lesion fits.

--residual-sources : the same blocks built from the sources' RESIDUALS (activity − GLM_uni prediction), so the predictors
                     carry no task signal at all.  Results dir GLM_ti_td_side_resid_results (base fit reused).
--null             : TI/TD labels shuffled within each side before anything else.  Results dir GLM_ti_td_side_null_results.

Outputs  ({BASE_DATA_PATH}/{mouse}/{date}/{MODEL_DIR}/)
    side_fold{k}_{base,TI_L,TI_R,TD_L,TD_R,CTL}_full.pkl, side_fold{k}_design.pkl, side_session.pkl

Usage
-----
python GLM_ti_td_side_coupling.py --sessions IS-2-1L/2025-03-10
python GLM_ti_td_side_coupling.py --residual-sources --sessions ...
python GLM_ti_td_side_coupling.py --null
"""
import argparse
import contextlib
import gc
import io
import os
import pickle
import sys
import traceback
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if GLM_DIR not in sys.path:
    sys.path.insert(0, GLM_DIR)

from core.session_pipeline import (
    BASE_DATA_PATH, open_glm_stacked_data, prepare_features_for_glm, fit_glm_single_session,
)
from core.predictor_builder import create_cosine_bumps
from config.session_registry import PPC_AVC

def _cache_dir(name, base=BASE_DATA_PATH):
    """Analysis caches sit directly under ProcessedData (they were under ProcessedData/analysis until 2026-09-20)."""
    for d in (os.path.join(base, name), os.path.join(base, 'analysis', name)):
        if os.path.isdir(d):
            return d
    return os.path.join(base, name)


# ==============================================================================
MODEL_NAME = 'GLM_ti_td_side_coupling'
CONTEXTS   = [1, 2]
VARIANTS = {'main': 'GLM_ti_td_side_results', 'resid': 'GLM_ti_td_side_resid_results', 'null': 'GLM_ti_td_side_null_results'}
PREFIX = 'side'
BLOCKS = ['TI_L', 'TI_R', 'TD_L', 'TD_R', 'CTL']

N_LAG_BASES, LAG_FRAMES = 4, 8
LABEL_CACHE = os.path.join(_cache_dir('_ti_td_snccgp_dial_cache'), 'mouse_raw_index.csv')
PROP_TI, PROP_TD = 0.25, 0.25
BASE_MODEL_DIR, BASE_MODEL_PREFIX = 'GLM_uni_results', 'GLM_uni'
RESP_THRESHOLD = 0.1
MIN_GROUP, MAX_BLOCK = 6, 10               # every side x class group >= 6 neurons -> block size 3..10 (stored; filter later)
NEAR_PX = 20
SPLIT_SEED = 20260920
# ==============================================================================

_INDEX = None


def model_dir_for(variant, prop=PROP_TI):
    """Results dir: the quartile runs keep their names; another tail fraction gets its own dir (e.g. GLM_ti_td_side_tails20_results)."""
    d = VARIANTS[variant]
    return d if np.isclose(prop, PROP_TI) else d.replace('_results', f'_tails{int(round(100 * prop))}_results')


def load_index(prop=PROP_TI):
    """TI / TD = top / bottom `prop` of the pooled snCCGP (0.25 = quartiles, the original runs; 0.20 = the Figure 4 tails)."""
    global _INDEX
    if _INDEX is None or _INDEX[1] != prop:
        idx = pd.read_csv(LABEL_CACHE)
        idx = idx[(idx.half.astype(str) == 'all') & np.isfinite(idx.snCCGP)].copy()
        hi, lo = float(idx.snCCGP.quantile(1 - prop)), float(idx.snCCGP.quantile(prop))
        idx['cls'] = np.where(idx.snCCGP >= hi, 'TI', np.where(idx.snCCGP <= lo, 'TD', 'Int'))
        _INDEX = (idx, prop)
    return _INDEX[0]


def choice_side(Y, meta):
    """d' of the trial-mean activity between choice 1 and choice 0, averaged over the two contexts (one value per neuron)."""
    d = []
    for c in CONTEXTS:
        f = meta['context'] == c
        ym = pd.DataFrame(Y[f]).groupby(meta['trial_id'][f]).mean()
        ch = pd.Series(meta['choice'][f]).groupby(meta['trial_id'][f]).first().reindex(ym.index).values
        a, b = ym.values[ch == 1], ym.values[ch == 0]
        d.append((a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12))
    return np.mean(d, axis=0)


def lag_features(S):
    """Each column of S (frames x neurons) -> N_LAG_BASES causal raised-cosine lag features, z-scored: (frames, neurons*bases)."""
    centers = np.linspace(0, LAG_FRAMES - 1, N_LAG_BASES)
    bases = create_cosine_bumps(np.arange(LAG_FRAMES, dtype=float), centers, np.full(N_LAG_BASES, (centers[1] - centers[0]) * 2.0))
    out = np.zeros((S.shape[0], S.shape[1] * N_LAG_BASES), np.float32)
    for u in range(S.shape[1]):
        for b in range(N_LAG_BASES):
            out[:, u * N_LAG_BASES + b] = np.convolve(S[:, u], bases[:, b], mode='full')[:S.shape[0]]
    sd = out.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (out - out.mean(0, keepdims=True)) / sd


def fde_match(pool, fde, ref, n, rng, exclude=()):
    """n neurons from `pool` whose base FDE matches a random n of `ref` (greedy nearest neighbour, no replacement)."""
    pool = [p for p in pool if p not in set(exclude)]
    want = fde[rng.choice(ref, size=min(n, len(ref)), replace=False)] if len(ref) else []
    chosen = []
    for v in np.sort(want)[::-1]:
        j = min(pool, key=lambda i: abs(fde[i] - v))
        chosen.append(j)
        pool.remove(j)
    return np.sort(np.array(chosen, int))


def fit_model(X_aug, Y_t, metadata, pinfo, mouse_ID, date, model_dir, prefix, skip_existing, log, reuse_from=None):
    for d in ([model_dir] if reuse_from is None else [model_dir, reuse_from]):
        p = os.path.join(BASE_DATA_PATH, mouse_ID, date, d, f'{prefix}_full.pkl')
        if skip_existing and os.path.exists(p):
            with open(p, 'rb') as f:
                res = pickle.load(f)
            if np.asarray(res['test_frac_dev']).shape[0] == Y_t.shape[1]:
                log(f'    [{prefix}] loaded from cache ({d})')
                return res
    t0 = datetime.now()
    with contextlib.redirect_stdout(io.StringIO()):
        pinfo_split, fgs = prepare_features_for_glm(pinfo)
        assert int(fgs.sum()) == X_aug.shape[1], 'feature_group_size mismatch'
        _, res, _, _ = fit_glm_single_session(X_aug, Y_t, metadata, fgs, pinfo_split, mouse_ID, date,
                                              model_dir=model_dir, model_prefix=prefix, contexts=CONTEXTS)
    log(f'    [{prefix}] fitted in {(datetime.now() - t0).total_seconds()/60:.1f} min (median FDE {np.median(res["test_frac_dev"]):.3f})')
    gc.collect()
    if _HAS_TF:
        tf.keras.backend.clear_session()
    return res


def run_session(mouse_ID, date, variant='main', skip_existing=True, log=print, prop=PROP_TI):
    model_dir = model_dir_for(variant, prop)
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir)
    session_path = os.path.join(save_dir, f'{PREFIX}_session.pkl')
    if skip_existing and os.path.exists(session_path):
        log(f'[SKIP] {mouse_ID}/{date}: session file exists')
        return 'skipped'

    lab = load_index(prop)
    lab = lab[lab.sid == f'{mouse_ID}_{date}']
    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(mouse_ID, date, print_summary=False)
    m = np.isin(metadata['context'], CONTEXTS)
    X, Y, metadata = X[m], Y[m], {k: v[m] for k, v in metadata.items()}
    with open(os.path.join(BASE_DATA_PATH, mouse_ID, date, BASE_MODEL_DIR, f'{BASE_MODEL_PREFIX}_full.pkl'), 'rb') as f:
        uni = pickle.load(f)
    fde_uni = np.asarray(uni['test_frac_dev'], float)
    n_neurons = Y.shape[1]
    assert n_neurons == len(fde_uni) and (len(lab) == 0 or lab.col.max() < n_neurons), 'neuron indexing mismatch'

    seed = (SPLIT_SEED + zlib.crc32(f'{mouse_ID}/{date}'.encode())) % (2**32)
    rng = np.random.default_rng(seed)
    cls = np.full(n_neurons, 'NA', dtype=object)
    cls[lab.col.values] = lab.cls.values
    true_cls = cls.copy()
    dprime = choice_side(Y, metadata)
    side = np.where(dprime > 0, 'R', 'L')
    if variant == 'null':                                     # shuffle TI/TD labels within each side
        for s in 'LR':
            k = np.where(np.isin(cls, ['TI', 'TD']) & (side == s))[0]
            cls[k] = rng.permutation(cls[k])
    groups = {f'{c}_{s}': np.where((cls == c) & (side == s))[0] for c in ('TI', 'TD') for s in 'LR'}
    sizes = {k: len(v) for k, v in groups.items()}
    if min(sizes.values()) < MIN_GROUP:
        log(f'[EXCLUDE] {mouse_ID}/{date}: {sizes} (< {MIN_GROUP} in a group)')
        return 'excluded'
    os.makedirs(save_dir, exist_ok=True)
    halves = {}
    for k, v in groups.items():
        p = rng.permutation(v)
        halves[k] = (np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:]))
    n_block = min(min(len(h[0]), len(h[1])) for h in halves.values())
    n_block = min(n_block, MAX_BLOCK)

    # control pool: responsive neurons outside the Choice class (never labelled TI/Int/TD)
    ctl_pool = list(np.where((true_cls == 'NA') & (fde_uni > RESP_THRESHOLD))[0])
    ti_all = np.concatenate([groups['TI_L'], groups['TI_R']])
    ctl_src = {k: fde_match(ctl_pool, fde_uni, np.concatenate([halves['TI_L'][k - 1], halves['TI_R'][k - 1]]), n_block, rng)
               for k in (1, 2)}
    ctl_tgt = fde_match(ctl_pool, fde_uni, ti_all, len(ti_all) // 2, rng, exclude=np.concatenate(list(ctl_src.values())))
    int_idx = np.where(cls == 'Int')[0]

    S = Y.astype(np.float32)
    if variant == 'resid':                                    # sources carry no task signal: activity − GLM_uni prediction
        S = (Y - np.exp(X @ np.asarray(uni['weights']) + np.ravel(uni['w0'])[None, :])).astype(np.float32)
    try:
        from analysis.ti_td_noise_corr import _mouse_positions
        pos = _mouse_positions(mouse_ID, date, n_neurons)
    except Exception:
        pos = None
    log(f'[{MODEL_NAME} {variant}] {mouse_ID}/{date}: groups {sizes} | block size {n_block} | Int {len(int_idx)} | '
        f'CTL targets {len(ctl_tgt)} | seed {seed} | {X.shape[0]:,} frames')

    rows = []
    for k in (1, 2):
        src_h, tgt_h = (0, 1) if k == 1 else (1, 0)
        src = {g: np.sort(rng.choice(halves[g][src_h], n_block, replace=False)) for g in groups}
        src['CTL'] = ctl_src[k]
        tgt = {g: halves[g][tgt_h] for g in groups}
        targets = np.concatenate([tgt[g] for g in groups] + [int_idx, ctl_tgt])
        tcls = np.concatenate([np.full(len(tgt[g]), g[:2], dtype=object) for g in groups] +
                              [np.full(len(int_idx), 'Int', dtype=object), np.full(len(ctl_tgt), 'CTL', dtype=object)])
        all_src = np.concatenate(list(src.values()))
        assert not (set(targets) & set(all_src)), 'target inside a source block'
        assert len(np.unique(targets)) == len(targets), 'duplicate targets'
        log(f'  fold {k}: {n_block} sources per block | targets ' + ', '.join(f'{c} {int((tcls == c).sum())}' for c in ('TI', 'TD', 'Int', 'CTL')))
        Y_t = Y[:, targets]
        pfx = f'{PREFIX}_fold{k}'
        base = fit_model(X, Y_t, metadata, dict(pred_info), mouse_ID, date, model_dir, f'{pfx}_base', skip_existing, log,
                         reuse_from=model_dir_for('main', prop) if variant == 'resid' else None)
        same_split = bool(np.array_equal(base['combined_test_mask'], uni['combined_test_mask']))
        assert same_split, 'test trials differ from GLM_uni'
        r = dict(target_idx=targets, target_cls=tcls.astype(str), true_cls=true_cls[targets].astype(str), side=side[targets],
                 fold=np.full(len(targets), k), n_block=np.full(len(targets), n_block),
                 fde_base=np.asarray(base['test_frac_dev']), fde_base_vis=np.asarray(base['test_frac_dev_per_ctx'][1]),
                 fde_base_aud=np.asarray(base['test_frac_dev_per_ctx'][2]))
        for b in BLOCKS:
            pi = dict(pred_info)
            for j, s in enumerate(src[b]):
                pi[f'cpl_{b}_n{s}'] = {'start': X.shape[1] + j * N_LAG_BASES, 'end': X.shape[1] + (j + 1) * N_LAG_BASES,
                                       'names': [f'cpl_{b}_n{s}_lag{q}' for q in range(N_LAG_BASES)]}
            res = fit_model(np.hstack([X, lag_features(S[:, src[b]])]), Y_t, metadata, pi, mouse_ID, date, model_dir,
                            f'{pfx}_{b}', skip_existing, log)
            assert np.array_equal(res['combined_test_mask'], base['combined_test_mask']), 'blocks scored on different test trials'
            r[f'fde_{b}'] = np.asarray(res['test_frac_dev'])
            r[f'fde_{b}_vis'], r[f'fde_{b}_aud'] = np.asarray(res['test_frac_dev_per_ctx'][1]), np.asarray(res['test_frac_dev_per_ctx'][2])
            r[f'near_{b}'] = (np.full(len(targets), False) if pos is None else
                              np.array([np.min(np.linalg.norm(pos[src[b]] - pos[t], axis=1)) < NEAR_PX for t in targets]))
        with open(os.path.join(save_dir, f'{pfx}_design.pkl'), 'wb') as f:
            pickle.dump(dict(fold=k, targets=targets, target_cls=tcls, sources=src, n_block=n_block, variant=variant), f)
        rows.append(r)
        del Y_t
        gc.collect()

    T = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    t = T['target_idx']
    same = lambda c: np.where(T['side'] == 'L', T[f'fde_{c}_L'], T[f'fde_{c}_R'])
    opp = lambda c: np.where(T['side'] == 'L', T[f'fde_{c}_R'], T[f'fde_{c}_L'])
    T.update(fde_sTI=same('TI'), fde_oTI=opp('TI'), fde_sTD=same('TD'), fde_oTD=opp('TD'),
             dprime=dprime[t], fde_uni=fde_uni[t], mouse_ID=mouse_ID, date=date, variant=variant, split_seed=seed, group_sizes=sizes)
    sn = np.full(n_neurons, np.nan); snw = np.full(n_neurons, np.nan)
    sn[lab.col.values], snw[lab.col.values] = lab.snCCGP.values, lab.sn_within.values
    T.update(snCCGP=sn[t], sn_within=snw[t])
    with open(session_path, 'wb') as f:
        pickle.dump(T, f)
    for c in ('TI', 'TD', 'Int', 'CTL'):
        mk = T['target_cls'] == c
        log(f'    {c:3s} targets (n={mk.sum()}): gain over base — ' + '  '.join(
            f'{lab_} {np.mean(T[key][mk] - T["fde_base"][mk]):+.4f}' for lab_, key in
            (('same-side TI', 'fde_sTI'), ('opp-side TI', 'fde_oTI'), ('same-side TD', 'fde_sTD'), ('opp-side TD', 'fde_oTD'), ('control', 'fde_CTL'))))
    del X, Y, S, metadata
    gc.collect()
    return 'success'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=None, help='MOUSE/DATE list')
    ap.add_argument('--residual-sources', action='store_true')
    ap.add_argument('--null', action='store_true')
    ap.add_argument('--no-skip', action='store_true')
    ap.add_argument('--prop', type=float, default=PROP_TI, help='tail fraction of the TI / TD labels (0.25 = quartiles; 0.20 = the Figure 4 tails, results in *_tails20_results)')
    args = ap.parse_args()
    variant = 'resid' if args.residual_sources else ('null' if args.null else 'main')

    sessions = [(m, d) for m, ds in PPC_AVC.items() for d in ds]
    if args.sessions:
        want = {s for a in args.sessions for s in a.split(',') if s}
        sessions = [(m, d) for m, d in sessions if f'{m}/{d}' in want]

    log_dir = os.path.join(BASE_DATA_PATH, 'ti_td_split_logs')
    os.makedirs(log_dir, exist_ok=True)
    tag = '' if np.isclose(args.prop, PROP_TI) else f'_tails{int(round(100 * args.prop))}'
    log_path = os.path.join(log_dir, f'{MODEL_NAME}_{variant}{tag}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log')

    def log(line):
        stamp = f'[{datetime.now():%H:%M:%S}] {line}'
        print(stamp, flush=True)
        with open(log_path, 'a') as f:
            f.write(stamp + '\n')

    log(f'{MODEL_NAME}: {len(sessions)} sessions | variant {variant} -> {model_dir_for(variant, args.prop)} | labels: top / bottom {args.prop:.2f} of the index | single-neuron predictors, '
        f'{N_LAG_BASES} lag bases x {LAG_FRAMES} frames | blocks {BLOCKS} | min group {MIN_GROUP}, max block {MAX_BLOCK}')
    status = {}
    for m, d in sessions:
        t0 = datetime.now()
        try:
            status[(m, d)] = run_session(m, d, variant=variant, skip_existing=not args.no_skip, log=log, prop=args.prop)
        except Exception as e:
            traceback.print_exc()
            status[(m, d)] = f'error: {e}'
            log(f'[ERROR] {m}/{d}: {e}')
        log(f'  -> {m}/{d}: {status[(m, d)]}  ({(datetime.now() - t0).total_seconds()/60:.1f} min)')
        gc.collect()
        if _HAS_TF:
            tf.keras.backend.clear_session()
    log('SUMMARY: ' + ', '.join(f'{k[0]}/{k[1]}={v}' for k, v in status.items()))


if __name__ == '__main__':
    main()
