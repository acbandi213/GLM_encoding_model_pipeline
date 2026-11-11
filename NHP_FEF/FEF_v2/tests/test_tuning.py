"""
Tests for fef.readout.tuning.

The point of DGI is that it is scale-free, so the tests that matter are the
invariance tests: a per-task gain change or baseline shift must not move it,
because those are exactly the two confounds (2.1x rate difference, task-mean
offset) that broke the v1 FDE-based CPI.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fef.readout import tuning as T    # noqa: E402


THETA = T.DEFAULT_THETA


def vonmises(pd_deg, kappa=2.0, gain=1.0, base=0.0, theta=THETA):
    return base + gain * np.exp(kappa * np.cos(theta - np.deg2rad(pd_deg)))


# ---------------------------------------------------------------- correlation

def test_row_correlation_identity_and_reversal():
    a = vonmises(90)[None, :]
    assert T.row_correlation(a, a)[0] == pytest.approx(1.0)
    assert T.row_correlation(a, -a)[0] == pytest.approx(-1.0)


def test_flat_curve_is_nan_not_zero():
    """A flat neuron is unmeasured, not invariant. v1 filed such units as CI."""
    flat = np.ones((1, 8))
    assert np.isnan(T.row_correlation(flat, vonmises(0)[None, :])[0])


def test_row_correlation_is_vectorised():
    A = np.vstack([vonmises(0), vonmises(90), vonmises(180)])
    B = np.vstack([vonmises(0), vonmises(180), vonmises(180)])
    r = T.row_correlation(A, B)
    assert r.shape == (3,)
    assert r[0] == pytest.approx(1.0) and r[2] == pytest.approx(1.0)
    assert r[1] < 0


# ------------------------------------------------------- the invariance claim

@pytest.mark.parametrize("gain,base", [(0.3, 0.0), (1.0, 5.0), (0.05, 2.0)])
def test_dgi_invariant_to_per_task_gain_and_offset(gain, base):
    """
    The whole argument for DGI: scaling one task's tuning curve (the firing-rate
    difference) or adding a baseline (the ctx_main offset) must not move it.
    """
    sac = vonmises(45, gain=1.0, base=1.0)[None, :]
    pur = vonmises(45, gain=gain, base=base)[None, :]
    assert T.direction_generalization_index(sac, pur)[0] == pytest.approx(1.0)


def test_dgi_separates_the_three_mouse_motifs():
    """Mouse Fig. 2d: same-sign, one-context-only, and reversed neurons."""
    sac = np.vstack([vonmises(90), vonmises(90), vonmises(90)])
    pur = np.vstack([
        vonmises(90, gain=0.3),                       # invariant, weaker
        np.ones(8) + 1e-9 * np.arange(8),             # untuned in pursuit
        vonmises(270),                                # reversed preference
    ])
    d = T.direction_generalization_index(sac, pur)
    assert d[0] == pytest.approx(1.0)
    assert abs(d[1]) < 0.7
    assert d[2] < -0.5


# ------------------------------------------------------------------ direction

def test_preferred_direction_recovers_truth():
    for pd in (0, 45, 135, 270):
        got = T.preferred_direction(vonmises(pd)[None, :], harmonic=1)[0]
        assert min(abs(got - pd), 360 - abs(got - pd)) < 5.0


def test_pd_shift_is_unsigned_and_wrapped():
    a, b = vonmises(10)[None, :], vonmises(350)[None, :]
    assert T.pd_shift(a, b)[0] == pytest.approx(20.0, abs=3.0)
    assert 0.0 <= T.pd_shift(vonmises(0)[None, :], vonmises(180)[None, :])[0] <= 180.0


def test_selectivity_index_bounds():
    """Noneman eq. 2: 0 for uniform firing, near 1 for a single direction."""
    assert T.selectivity_index(np.ones((1, 8)))[0] == pytest.approx(0.0, abs=1e-9)
    one = np.zeros((1, 8))
    one[0, 2] = 1.0
    assert T.selectivity_index(one)[0] == pytest.approx(1.0)


def test_depth_ratio_reports_the_gain_dgi_ignores():
    sac = vonmises(45, gain=1.0)[None, :]
    pur = vonmises(45, gain=0.25)[None, :]
    assert T.direction_generalization_index(sac, pur)[0] == pytest.approx(1.0)
    assert T.tuning_depth_ratio(sac, pur)[0] == pytest.approx(0.25, rel=1e-6)


# ------------------------------------------------------------ fisher averaging

def test_fisher_mean_handles_unity_without_inf():
    assert T.fisher_mean(np.array([[1.0], [0.0]]), axis=0)[0] > 0.9
    assert np.isfinite(T.fisher_mean(np.array([[1.0], [1.0]]), axis=0)[0])


# ------------------------------------------------------------- noise ceiling

def _poisson_session(pd_sac, pd_pur, sac_gain=8.0, pur_gain=8.0,
                     n_trials=60, seed=0):
    """Simulate two blocks of Poisson trials with known per-task tuning."""
    rng = np.random.default_rng(seed)
    d = np.tile(np.arange(8), n_trials)
    task = np.repeat([0, 1], d.size)
    d = np.concatenate([d, d])
    lam = np.where(
        task == 0,
        vonmises(pd_sac, gain=sac_gain, base=1.0)[d],
        vonmises(pd_pur, gain=pur_gain, base=1.0)[d],
    )
    counts = rng.poisson(lam)[:, None].astype(float)
    return counts, d, task == 0, task == 1


def test_noise_ceiling_recovers_invariance_despite_a_rate_difference():
    """
    Same tuning in both tasks but 8x fewer spikes in 'pursuit' — the FEF
    situation. Raw DGI is attenuated by the pursuit noise; the corrected value
    must return to ~1.
    """
    counts, d, ma, mb = _poisson_session(90, 90, sac_gain=8.0, pur_gain=0.2,
                                         n_trials=20, seed=1)
    out = T.dgi_with_noise_ceiling(counts, d, ma, mb, n_splits=60,
                                   rng=np.random.default_rng(2))
    assert out["usable"][0]
    assert out["reliability_b"][0] < out["reliability_a"][0]   # pursuit noisier
    assert out["dgi_raw"][0] < 0.85              # attenuated by that noise
    assert out["dgi_corrected"][0] > 0.85        # ...and recovered
    assert out["dgi_corrected"][0] > out["dgi_raw"][0]


def test_noise_ceiling_detects_genuine_context_dependence():
    counts, d, ma, mb = _poisson_session(0, 180, n_trials=80, seed=3)
    out = T.dgi_with_noise_ceiling(counts, d, ma, mb, n_splits=60,
                                   rng=np.random.default_rng(4))
    assert out["usable"][0]
    assert out["dgi_corrected"][0] < -0.5


def test_unreliable_units_are_excluded_not_mislabelled():
    """
    An untuned, low-rate unit must come back NaN. If it came back near 0 it
    would be counted as context-DEPENDENT for being poorly measured — the
    mechanism that manufactured the v1 coupling slope.
    """
    rng = np.random.default_rng(5)
    d = np.tile(np.arange(8), 40)
    d = np.concatenate([d, d])
    task = np.repeat([0, 1], d.size // 2)
    counts = rng.poisson(0.05, size=(d.size, 1)).astype(float)
    out = T.dgi_with_noise_ceiling(counts, d, task == 0, task == 1,
                                  n_splits=40, reliability_floor=0.2,
                                  rng=np.random.default_rng(6))
    assert not out["usable"][0]
    assert np.isnan(out["dgi_corrected"][0])


def test_split_halves_are_balanced_and_disjoint():
    counts = np.arange(64, dtype=float)[:, None]
    d = np.tile(np.arange(8), 8)
    mask = np.ones(64, bool)
    a, b = T.tuning_curves_by_split(counts, d, mask,
                                    rng=np.random.default_rng(7))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    # every direction contributes to both halves, so the grand means agree
    assert a.mean() == pytest.approx(b.mean(), abs=8.0)


def test_direction_with_too_few_trials_yields_nan():
    counts = np.ones((9, 1))
    d = np.array([0, 0, 1, 2, 2, 3, 3, 4, 4])   # direction 1 has a single trial
    a, b = T.tuning_curves_by_split(counts, d, np.ones(9, bool),
                                    rng=np.random.default_rng(8))
    assert np.isnan(a[0, 1]) and np.isnan(b[0, 1])
    assert np.isnan(a[0, 5])                     # directions 5-7 are absent


# ------------------------------------------------- cross-EPOCH (same trials)

def test_split_indices_are_disjoint_and_within_direction():
    d = np.tile(np.arange(8), 10)
    pairs = T.split_indices(d, np.ones(d.size, bool), rng=np.random.default_rng(11))
    for k, (a, b) in enumerate(pairs):
        assert not set(a.tolist()) & set(b.tolist())      # disjoint
        assert (d[a] == k).all() and (d[b] == k).all()    # within direction


def test_dgi_between_epochs_uses_one_shared_split():
    """
    Two epochs of the SAME trials. Independent splits would let a trial land in
    both halves of a cross term, so shared trial-level noise would inflate the
    cross correlation toward "invariant". The shared-split version must not.
    """
    rng = np.random.default_rng(12)
    n_rep = 40
    d = np.tile(np.arange(8), n_rep)
    # a per-trial gain shared by both epochs: pure trial-level nuisance
    trial_gain = rng.gamma(4.0, 0.25, size=d.size)
    lam_vis = vonmises(0, gain=3.0, base=1.0)[d] * trial_gain
    lam_sac = vonmises(180, gain=3.0, base=1.0)[d] * trial_gain   # OPPOSITE tuning
    cv = rng.poisson(lam_vis)[:, None].astype(float)
    cs = rng.poisson(lam_sac)[:, None].astype(float)
    mask = np.ones(d.size, bool)

    shared = T.dgi_between_epochs(cv, cs, d, mask, n_splits=60,
                                  rng=np.random.default_rng(13))
    assert shared["usable"][0]
    # tuning really is opposite, so a correct estimator must report it
    assert shared["dgi_corrected"][0] < -0.5

    # the naive route (independent splits) is biased upward by the shared gain
    naive = T.dgi_with_noise_ceiling(np.vstack([cv, cs]),
                                     np.concatenate([d, d]),
                                     np.r_[np.ones(d.size, bool), np.zeros(d.size, bool)],
                                     np.r_[np.zeros(d.size, bool), np.ones(d.size, bool)],
                                     n_splits=60, rng=np.random.default_rng(13))
    assert shared["dgi_corrected"][0] <= naive["dgi_corrected"][0] + 1e-9


def test_dgi_between_epochs_recovers_identical_epoch_tuning():
    rng = np.random.default_rng(14)
    d = np.tile(np.arange(8), 40)
    lam = vonmises(90, gain=4.0, base=1.0)[d]
    a = rng.poisson(lam)[:, None].astype(float)
    b = rng.poisson(lam * 0.4)[:, None].astype(float)   # same tuning, weaker
    out = T.dgi_between_epochs(a, b, d, np.ones(d.size, bool), n_splits=60,
                               rng=np.random.default_rng(15))
    assert out["usable"][0]
    assert out["dgi_corrected"][0] > 0.8
