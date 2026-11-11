"""
Converter unit tests, pinned to the verified `kendra_scrappy_0136a` schema.

Run:  cd FEF_v2 && python -m pytest tests/ -q
(WP0 gates on ZERO skips — a skip means the discovery glob is pointing where the
data is not, which is how 23 data-dependent tests passed vacuously in v1.)

PORT NOTE (WP0). v1's file also carried 10 predictor-contract tests importing
`nhp.predictors`, which pulls in `core.predictor_builder` from the mouse repo.
That module is WP4's port ("Port and rebuild the design matrix"), and the design
QC it checks — full rank, condition number, `ctx_main` aliasing — is WP4's own
acceptance criterion, re-measured at 251 columns. Those tests are therefore NOT
ported here; they move to `tests/test_predictors.py` in WP4. The names dropped
are logged in `results/WP0_data_validation.md` (S8).
"""

import glob
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fef import config as C                                  # noqa: E402
from fef.io.convert import NHPSessionConverter, _hemifield   # noqa: E402
from fef.io.matio import open_session                        # noqa: E402

# ONE discovery source: the server. v1 globbed the repo as a fallback, so when
# the sessions moved the suite went green while exercising nothing.
MAT = next(iter(sorted(glob.glob(os.path.join(C.SESSIONS_DIR,
                                              'kendra_scrappy_0136a*.mat')))),
           None)
pytestmark = pytest.mark.skipif(
    MAT is None,
    reason=f'kendra_scrappy_0136a .mat not found in {C.SESSIONS_DIR}')


@pytest.fixture(scope='module')
def mat():
    s = open_session(MAT)
    yield s
    s.close()


@pytest.fixture(scope='module')
def converted():
    cv = NHPSessionConverter(MAT, verbose=False)
    glm = cv.build(0)
    yield cv, glm
    cv.close()


# ── metadata / schema ────────────────────────────────────────────────────────
def test_session_metadata(mat):
    assert mat.sess_name == 'kendra_scrappy_0136a'
    assert mat.monkey == 'scrappy'
    assert mat.probe_labels == ['lFEF', 'rFEF']
    assert np.allclose(mat.probe_depths, [7.2, 9.8])


def test_unit_counts(mat):
    """
    Bilateral FEF: 366 units on lFEF, 414 on rFEF.

    RE-DERIVED IN WP0 on export `044e61186379fd29`. v1 measured 374 / 430 on
    export `171af6ac94149c62` — the spike sorting changed between exports
    (-2.14% / -3.72% units, with mean rate UP ~12%; see
    results/WP0_data_validation.md). Four independent sources inside the file
    agree on the new counts: the mdir1 and purs1 block spike cells, the table's
    `spiketimes_N` cell width, and the unique ids in `sorting.spike_clusters`.
    The trial tables, event times and behaviour are unchanged.
    """
    assert mat.n_units(0) == 366
    assert mat.n_units(1) == 414


def test_trial_counts(mat):
    assert mat.n_trials('mdir1') == 432
    assert mat.n_trials('purs1') == 1467


def test_mcos_tables_decode(mat):
    """The MCOS tables must decode without MATLAB (see nhp/matio.py)."""
    md = mat.read_table('mdir1')
    pu = mat.read_table('purs1')
    assert len(md) == 44 and len(pu) == 41
    for need in ['angle', 'result', 'eyePos', 'TARG_ON', 'saccadeOnset',
                 'REWARD', 'END_TRIAL', 'spiketimes_1', 'spiketimes_2']:
        assert need in md, f'mdir1 missing {need}'
    for need in ['angle', 'speed', 'jump', 'PURSUIT_TARG_ON', 'pursuitOnset']:
        assert need in pu, f'purs1 missing {need}'


def test_result_categorical(mat):
    """`result` decodes to label strings; CORRECT counts are the known ones."""
    md, pu = mat.read_table('mdir1'), mat.read_table('purs1')
    assert sum(r == C.RESULT_CORRECT for r in md['result']) == 400
    assert sum(r == C.RESULT_CORRECT for r in pu['result']) == 1330


def test_directions_balanced(mat):
    """8 directions, 50 correct saccade trials each."""
    md = mat.read_table('mdir1')
    ok = np.array([r == C.RESULT_CORRECT for r in md['result']])
    ang = np.asarray(md['angle'], float)[ok]
    u, c = np.unique(ang, return_counts=True)
    assert list(u) == [0, 45, 90, 135, 180, 225, 270, 315]
    assert set(c) == {50}


def test_pursuit_jump_levels(mat):
    """`jump` has only -1/+1 here — no 0 level, contrary to the format notes."""
    pu = mat.read_table('purs1')
    assert set(np.unique(np.asarray(pu['jump'], float))) == {-1.0, 1.0}


# ── converted glm_data_dict ──────────────────────────────────────────────────
def test_trial_composition(converted):
    """
    Post-gate composition (DROP_DEAD_TRIALS is on by default): saccade is
    unaffected (~397); pursuit keeps BOTH step directions (1330 correct) and
    loses the dead-recording tail, landing near ~680.

    Was ~340 pursuit when PURSUIT_JUMP_KEEP dropped the forward-step half.
    """
    _, glm = converted
    ctx = np.array([t['context'] for t in glm.values()])
    n_sac = int((ctx == C.CTX_SACCADE).sum())
    n_pur = int((ctx == C.CTX_PURSUIT).sum())
    assert 390 <= n_sac <= 400            # saccade block is stable, not gated
    assert 600 <= n_pur <= 760            # ~677 after the dead-trial gate
    assert set(np.unique(ctx)) == {C.CTX_SACCADE, C.CTX_PURSUIT}


def test_both_step_directions_present_and_balanced(converted):
    """
    Both `jump` levels survive conversion, crossed with direction.

    Keeping backward steps only discarded 50.3% of the correct pursuit trials,
    in the task the model already under-fits.
    """
    _, glm = converted
    pur = [t for t in glm.values() if t['context'] == C.CTX_PURSUIT]
    jumps = np.array([t['jump'] for t in pur])
    assert set(np.unique(jumps)) == {-1.0, 1.0}
    # each step direction is represented in every direction
    for j in (-1.0, 1.0):
        dirs = {t['trial_type'] for t in pur if t['jump'] == j}
        assert len(dirs) == 8, f'jump={j:+.0f} covers only {len(dirs)} directions'
    frac = float((jumps > 0).mean())
    assert 0.35 < frac < 0.65, f'step directions unbalanced after gating: {frac:.2f}'
    # saccade trials carry no step direction
    assert all(not np.isfinite(t['jump'])
               for t in glm.values() if t['context'] == C.CTX_SACCADE)


def test_step_direction_flips_only_backward_pursuit(converted):
    """
    `step_direction` is where the target APPEARS; `angle` is where it then goes.

    They differ exactly on backward-step pursuit trials (the Rashbass step puts
    the target at angle+180 before it ramps through fixation) and nowhere else.
    """
    _, glm = converted
    for t in glm.values():
        flipped = abs((t['step_direction'] - t['angle']) % 360 - 180) < 1e-6
        backward = t['context'] == C.CTX_PURSUIT and t['jump'] == -1.0
        assert flipped == backward, (
            f'ctx={t["context"]} jump={t["jump"]} angle={t["angle"]} '
            f'step_direction={t["step_direction"]}')


def test_magnitude_code_standardised_within_task(converted):
    """
    Each task's own magnitude, z-scored within task, so the column is a
    magnitude contrast rather than a task contrast.
    """
    _, glm = converted
    for ctx in (C.CTX_SACCADE, C.CTX_PURSUIT):
        m = np.array([t['magnitude_code'] for t in glm.values()
                      if t['context'] == ctx])
        assert abs(m.mean()) < 0.05, f'ctx {ctx} magnitude_code mean {m.mean():.3f}'
        assert 0.9 < m.std() < 1.1, f'ctx {ctx} magnitude_code sd {m.std():.3f}'
        assert len(np.unique(np.round(m, 3))) == 2      # two balanced levels


def test_catchup_vectors_aligned(converted):
    """
    Catch-up onsets, directions and amplitudes stay index-aligned.

    A saccade with an unusable offset keeps its onset and gets a NaN direction,
    so the plain `catchup_saccade` group is unchanged while the direction-
    weighted group skips that event. Alignment is what must hold; finiteness of
    every direction is not.
    """
    _, glm = converted
    seen = n_ev = n_finite = 0
    for t in glm.values():
        ev = t['events']
        n = len(ev['catchup_saccades'])
        assert len(ev['catchup_dirs']) == n and len(ev['catchup_amps']) == n
        if n:
            seen += 1
            n_ev += n
            fin = np.isfinite(ev['catchup_dirs'])
            n_finite += int(fin.sum())
            amps = np.asarray(ev['catchup_amps'])
            assert np.all(amps[np.isfinite(amps)] >= 0)
            # direction and amplitude drop out together or not at all
            assert np.array_equal(fin, np.isfinite(amps))
    assert seen > 0.5 * len(glm), 'catch-up saccades vanished from most trials'
    assert n_finite > 0.95 * n_ev, (
        f'only {n_finite}/{n_ev} catch-up saccades have a usable direction')


def test_stim_offset_inside_window(converted):
    """
    Each task's stimulus-offset event lands inside the analysis window.

    Pursuit's PURSUIT_TARG_OFF is the one that matters: it sits ~1017 ms after
    target onset, mid-window, and was previously unmodelled.
    """
    _, glm = converted
    for ctx in (C.CTX_SACCADE, C.CTX_PURSUIT):
        offs = np.array([t['events']['stim_offset'] for t in glm.values()
                         if t['context'] == ctx], float)
        nb = next(iter(glm.values()))['events']['trial_length']
        inside = np.isfinite(offs) & (offs >= 0) & (offs < nb)
        assert inside.mean() > 0.9, (
            f'ctx {ctx}: only {inside.mean():.0%} of stim_offset events are in '
            f'the window')


def test_stability_gate_drops_dead_pursuit_trials():
    """The gate must remove the pursuit rundown tail and leave saccade intact.
    Guards the balancing precondition: without it, ~half the pursuit trials are
    near-dead recording and FDE_pursuit collapses."""
    from fef.io.convert import NHPSessionConverter
    mat = MAT      # resolved once at module level (server, then repo)

    orig = C.DROP_DEAD_TRIALS
    try:
        C.DROP_DEAD_TRIALS = False
        off = NHPSessionConverter(mat, bin_ms=25, verbose=False).build(0)
        C.DROP_DEAD_TRIALS = True
        on = NHPSessionConverter(mat, bin_ms=25, verbose=False).build(0)
    finally:
        C.DROP_DEAD_TRIALS = orig

    def counts(g):
        c = np.array([t['context'] for t in g.values()])
        return int((c == C.CTX_SACCADE).sum()), int((c == C.CTX_PURSUIT).sum())

    s_off, p_off = counts(off)
    s_on, p_on = counts(on)
    assert s_on == s_off, 'gate wrongly dropped saccade trials'
    assert p_on < 0.7 * p_off, f'gate should drop the dead pursuit tail ({p_off} -> {p_on})'


def test_trial_schema(converted):
    cv, glm = converted
    t = glm[0]
    assert t['neural']['spikes'].shape == (cv.n_bins, 366)   # 374 on v1's export
    assert np.issubdtype(t['neural']['spikes'].dtype, np.integer)
    assert t['neural']['spikes'].min() >= 0
    for k in ['eye_h', 'eye_v', 'eye_speed', 'eye_acc']:
        assert t['movement'][k].shape == (cv.n_bins,)
    for k in ['trial_length', 'target_on', 'move_onset', 'reward', 'go_cue']:
        assert k in t['events']


def test_window_anchored_on_target(converted):
    """Window is anchored to target onset, so target_on is a constant bin."""
    cv, glm = converted
    tb = np.array([t['events']['target_on'] for t in glm.values()])
    assert np.allclose(tb, -C.WIN_PRE_MS / cv.bin_ms)


def test_event_order_within_trial(converted):
    """target -> (go) -> movement -> reward, with movement inside the window."""
    cv, glm = converted
    for t in glm.values():
        e = t['events']
        assert 0 <= e['target_on'] < e['move_onset'] < cv.n_bins
        assert e['move_onset'] < e['reward']
        if t['context'] == C.CTX_SACCADE:
            assert np.isfinite(e['go_cue'])
            assert e['target_on'] < e['go_cue'] < e['move_onset']
        else:
            # pursuit FIX_OFF coincides with target onset, so no separate go cue
            assert not np.isfinite(e['go_cue'])


def test_reward_mostly_inside_window(converted):
    """
    Reward may fall past the window end on a few long saccade trials; the reward
    kernel is then legitimately all-zero for them. Measured: 4/397 saccade
    trials (1.0%) and 0 pursuit. Guard that it stays a rare edge case — if a
    large fraction fell outside, the reward group would be mostly unidentified.
    """
    cv, glm = converted
    for ctx, tol in ((C.CTX_SACCADE, 0.05), (C.CTX_PURSUIT, 0.01)):
        rw = np.array([t['events']['reward'] for t in glm.values()
                       if t['context'] == ctx])
        assert (rw >= cv.n_bins).mean() <= tol


def test_movement_latency_physiological(converted):
    """
    Saccade latency ~200 ms measured from the GO CUE (not target onset — the
    memory delay sits between them, making target->saccade ~1030 ms), and
    pursuit latency ~100 ms from target motion onset.
    """
    cv, glm = converted
    sac = np.array([(t['events']['move_onset'] - t['events']['go_cue']) * cv.bin_ms
                    for t in glm.values() if t['context'] == C.CTX_SACCADE])
    pur = np.array([(t['events']['move_onset'] - t['events']['target_on']) * cv.bin_ms
                    for t in glm.values() if t['context'] == C.CTX_PURSUIT])
    assert 120 <= np.median(sac) <= 320, f'saccade latency {np.median(sac)}'
    assert 50 <= np.median(pur) <= 220, f'pursuit latency {np.median(pur)}'


def test_direction_tuning_positive_control(converted):
    """
    FEF must show clear 8-way direction tuning. If this fails, binning or
    alignment is wrong and no CPI result should be believed.
    """
    cv, glm = converted
    _, si = cv.direction_tuning(glm, context=C.CTX_SACCADE)
    assert np.median(si) > 0.1
    assert (si > 0.2).mean() > 0.3


def test_direction_tuning_beats_shuffle(converted):
    """
    The tuning gate must be above its own null. Shuffling direction labels
    across trials destroys real tuning; if the shuffled selectivity index were
    comparable to the real one, the "positive control" would be certifying
    nothing.
    """
    cv, glm = converted
    _, si = cv.direction_tuning(glm, context=C.CTX_SACCADE, around='move_onset')
    rng = np.random.default_rng(0)
    keys = [k for k, t in glm.items() if t['context'] == C.CTX_SACCADE]
    perm = rng.permutation([glm[k]['trial_type'] for k in keys])
    shuffled = {}
    for i, k in enumerate(keys):
        t = dict(glm[k]); t['trial_type'] = int(perm[i]); shuffled[i] = t
    _, si_null = cv.direction_tuning(shuffled, context=C.CTX_SACCADE,
                                     around='move_onset')
    assert np.median(si) > np.median(si_null) * 1.3, (
        f'real SI {np.median(si):.3f} vs shuffled {np.median(si_null):.3f} — '
        f'the tuning gate is not above its null')


def test_movement_locked_tuning_detects_misalignment(converted):
    """
    ALIGNMENT sensitivity. A whole-trial selectivity index barely moves when the
    window is shifted, so on its own it cannot detect a misaligned anchor. The
    movement-locked index must, otherwise nothing in the suite would catch the
    binning/alignment errors the positive control is documented to catch.
    """
    cv, glm = converted
    _, si_true = cv.direction_tuning(glm, context=C.CTX_SACCADE,
                                     around='move_onset', halfwidth_bins=6)
    # Corrupt only the event times, leaving the spikes untouched.
    bad = {}
    for i, (k, t) in enumerate(glm.items()):
        t2 = dict(t)
        t2['events'] = dict(t['events'])
        t2['events']['move_onset'] = 5.0          # anchor in the pre-target baseline
        bad[i] = t2
    _, si_bad = cv.direction_tuning(bad, context=C.CTX_SACCADE,
                                    around='move_onset', halfwidth_bins=6)
    assert np.median(si_true) > np.median(si_bad), (
        f'movement-locked SI is insensitive to a wrecked anchor '
        f'({np.median(si_true):.3f} vs {np.median(si_bad):.3f})')


def test_hemifield_balanced():
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    h = [_hemifield(a) for a in angles]
    assert h.count(0) == 4 and h.count(1) == 4
