"""
WP4 design-matrix tests. Ported from v1 tests + the v2 additions
(prompts/WP4_IMPLEMENTATION_PLAN.md §3.4).

Data-dependent tests convert one Monkey S DEV probe (0136a lFEF) and one
Monkey Y probe (Ya_250619_s412) with skip_spikes=True — the design does not
depend on spikes, and the spike derefs are the expensive part over SMB.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fef import config as C                             # noqa: E402
from fef.design.predictors import FEFPredictorBuilder   # noqa: E402
from fef.design import qc, reparam                      # noqa: E402
from fef.fit import invariants                          # noqa: E402
from fef.io import NHPSessionConverter                  # noqa: E402

S_SESSION = "kendra_scrappy_0136a"
Y_SESSION = "Ya_250619_s412"


def _session_path(name):
    return os.path.join(C.SESSIONS_DIR, f"{name}_g0-{C.EXPORT_HASH}.mat")


def _zscore(X):
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 1e-12, sd, 1.0), sd


@pytest.fixture(scope="module")
def glm_S():
    with NHPSessionConverter(_session_path(S_SESSION), verbose=False) as cv:
        return cv.build(0, skip_spikes=True)


@pytest.fixture(scope="module")
def glm_Y():
    with NHPSessionConverter(_session_path(Y_SESSION), verbose=False) as cv:
        return cv.build(0, skip_spikes=True)


@pytest.fixture(scope="module")
def built_S(glm_S):
    b = FEFPredictorBuilder(verbose=False)
    dms, names, info = b.build(glm_S, monkey="S")
    return b, dms, names, info, glm_S


# ---------------------------------------------------------------- contract

def test_contiguity_and_names(built_S):
    _, dms, names, info, _ = built_S
    cur = 0
    for g, meta in info.items():
        assert meta["start"] == cur, f"{g} not contiguous"
        w = meta["end"] - meta["start"]
        assert len(meta["names"]) == w, f"{g} name/width mismatch"
        cur = meta["end"]
    assert cur == dms[0].shape[1] == len(names)


def test_default_column_count_228(built_S):
    _, dms, _, info, _ = built_S
    widths = {g: m["end"] - m["start"] for g, m in info.items()}
    expected = {
        "target": 6, "movement": 6, "go": 4, "reward": 4,
        "direction_shared": 48, "direction_diff": 48, "context": 13,
        "sustained": 8, "sustained_direction": 32,
        "catchup_saccade": 6, "catchup_direction": 12,
        "eye_position": 10, "eye_velocity": 20,
        "stim_offset": 4, "jump": 6, "delay": 1,
    }
    assert widths == expected, f"group widths deviate: {widths}"
    assert dms[0].shape[1] == 228


# ---------------------------------------------------------------- rank / QC

def _stacked_z(dms):
    X = np.vstack(dms)
    return _zscore(X)[0]


def test_full_rank_default(built_S):
    _, dms, _, _, _ = built_S
    Xz = _stacked_z(dms)
    G = Xz.T @ Xz
    ev = np.linalg.eigvalsh(G)
    assert ev.min() > 1e-8 * ev.max(), (
        f"design rank-deficient: min/max eig ratio {ev.min() / ev.max():.2e}")


def test_full_rank_all_optional_on(glm_S):
    b = FEFPredictorBuilder(verbose=False, include_magnitude=True,
                            include_catchup_amplitude=True)
    dms, _, info = b.build(glm_S, monkey="S")
    assert "magnitude" in info and "catchup_amplitude" in info
    assert dms[0].shape[1] == 228 + 12 + 6
    Xz = _stacked_z(dms)
    ev = np.linalg.eigvalsh(Xz.T @ Xz)
    assert ev.min() > 1e-8 * ev.max()


def test_no_predictor_aliases_context(built_S):
    _, dms, names, info, glm = built_S
    X = np.vstack(dms)
    ctx_col = info["context"]["start"]          # ctx_main is first in group
    ctx = X[:, ctx_col]
    Xz, _ = _zscore(X)
    cz = (ctx - ctx.mean()) / max(ctx.std(), 1e-12)
    r = (Xz * cz[:, None]).mean(0)
    outside = np.ones(X.shape[1], bool)
    outside[info["context"]["start"]:info["context"]["end"]] = False
    worst = np.nanmax(np.abs(r[outside]))
    assert worst < 0.5, (
        f"column aliases context: |r|={worst:.3f} at "
        f"{names[int(np.nanargmax(np.where(outside, np.abs(r), 0)))]}")


def test_groups_shared_or_cleanly_task_specific(built_S):
    """Every group is live in both tasks OR dead in exactly one, and the S
    task-specific budget is near-symmetric (|sac_only - pur_only| <= 4)."""
    _, dms, _, info, glm = built_S
    X = np.vstack(dms)
    ctx = np.concatenate([
        np.full(d.shape[0], t["context"])
        for d, t in zip(dms, [glm[k] for k in sorted(glm)])])
    sac, pur = ctx == C.CTX_SACCADE, ctx == C.CTX_PURSUIT
    n_sac_only = n_pur_only = 0
    for g, m in info.items():
        cols = X[:, m["start"]:m["end"]]
        live_s = np.abs(cols[sac]).max() > 1e-9
        live_p = np.abs(cols[pur]).max() > 1e-9
        assert live_s or live_p, f"{g} dead in BOTH tasks"
        w = m["end"] - m["start"]
        if live_s and not live_p:
            n_sac_only += w
        elif live_p and not live_s:
            n_pur_only += w
    assert abs(n_sac_only - n_pur_only) <= 4, (
        f"task budget asymmetric: sac-only {n_sac_only}, "
        f"pur-only {n_pur_only}")


def test_step_direction_changes_only_target_locked(glm_S):
    b_on = FEFPredictorBuilder(verbose=False, include_step_direction=True)
    b_off = FEFPredictorBuilder(verbose=False, include_step_direction=False)
    dms_on, names, info = b_on.build(glm_S, monkey="S")
    dms_off, _, _ = b_off.build(glm_S, monkey="S")
    X_on, X_off = np.vstack(dms_on), np.vstack(dms_off)
    diff = np.abs(X_on - X_off).max(0)
    changed = set(np.array(names)[diff > 1e-12])
    for nm in changed:
        assert ("_targ_" in nm or nm.startswith("jump_")), (
            f"step_direction changed non-target-locked column {nm}")


# ------------------------------------------------------------- Monkey Y

def test_monkey_Y_degeneracies(glm_Y):
    b = FEFPredictorBuilder(verbose=False, include_magnitude=True)
    dms, _, info = b.build(glm_Y, monkey="Y")
    assert "jump" in b.dropped_groups, "Y jump not dropped"
    assert "magnitude" in b.dropped_groups, "Y magnitude not dropped"
    assert "jump" not in info and "magnitude" not in info
    # remaining group order identical to the S order minus dropped groups
    bS = FEFPredictorBuilder(verbose=False, include_magnitude=True)
    order_S = [g for g in ("target", "movement", "go", "reward",
                           "direction_shared", "direction_diff", "context",
                           "sustained", "sustained_direction",
                           "catchup_saccade", "catchup_direction",
                           "eye_position", "eye_velocity", "stim_offset",
                           "delay")]
    assert [g for g in info] == order_S, f"Y group order shifted: {list(info)}"
    # full rank without the degenerate groups
    Xz = _stacked_z(dms)
    ev = np.linalg.eigvalsh(Xz.T @ Xz)
    assert ev.min() > 1e-8 * ev.max()


def test_monkey_Y_config_crosscheck_raises_on_mismatch(glm_S):
    """0136a has both jump levels; claiming it is monkey Y must raise."""
    b = FEFPredictorBuilder(verbose=False)
    with pytest.raises(RuntimeError, match="degeneracy mismatch"):
        b.build(glm_S, monkey="Y")


def test_monkey_Y_h1_stimulus_flip(glm_Y):
    """All-backward pursuit: stimulus direction = movement + 180 on every
    pursuit trial, so cos/sin h1 flip sign and h2 does not."""
    for t in glm_Y.values():
        if int(t["context"]) != C.CTX_PURSUIT:
            continue
        ang, sd = np.radians(t["angle"]), np.radians(t["step_direction"])
        assert np.isclose(np.cos(sd), -np.cos(ang), atol=1e-9)
        assert np.isclose(np.sin(sd), -np.sin(ang), atol=1e-9)
        assert np.isclose(np.cos(2 * sd), np.cos(2 * ang), atol=1e-9)


def test_monkey_Y_kernel_windows(glm_Y, glm_S):
    """WP4e decision (Akhil 2026-08-11): per-monkey kernel windows + lag-bank
    sizes restore Y identifiability (cond 45-93 in 10/10). S must be
    bit-untouched (exceptions-only dicts)."""
    bY = FEFPredictorBuilder(verbose=False)
    dms, _, info = bY.build(glm_Y, monkey="Y")
    assert bY.move_win == (-150.0, 200.0)
    assert bY.reward_win == (0.0, 80.0)
    assert bY.n_eye_lags == 3 and bY.n_pos_lags == 2
    assert info["eye_position"]["end"] - info["eye_position"]["start"] == 4
    assert info["eye_velocity"]["end"] - info["eye_velocity"]["start"] == 12
    assert dms[0].shape[1] == 208
    bS = FEFPredictorBuilder(verbose=False)
    dms_S, _, _ = bS.build(glm_S, monkey="S")
    assert bS.move_win == C.MOVE_WIN_MS and bS.n_eye_lags == 5
    assert dms_S[0].shape[1] == 228
    # explicit overrides still win over the per-monkey dicts
    bE = FEFPredictorBuilder(verbose=False, move_win=(-100.0, 300.0),
                             n_eye_lags=4)
    bE.build(glm_Y, monkey="Y")
    assert bE.move_win == (-100.0, 300.0) and bE.n_eye_lags == 4


# ------------------------------------------------------------- reparam

def test_per_task_equals_shared_diff_times_T(glm_S):
    b_sd = FEFPredictorBuilder(verbose=False, direction_param="shared_diff")
    b_pt = FEFPredictorBuilder(verbose=False, direction_param="per_task")
    dms_sd, _, info_sd = b_sd.build(glm_S, monkey="S")
    dms_pt, _, info_pt = b_pt.build(glm_S, monkey="S")
    X_sd = reparam.stack_direction_columns(
        dms_sd, info_sd, ("direction_shared", "direction_diff"))
    X_pt = reparam.stack_direction_columns(
        dms_pt, info_pt, ("direction_sac", "direction_pur"))
    T = reparam.shared_diff_to_per_task_transform(X_sd.shape[1] // 2)
    assert np.abs(X_pt - X_sd @ T).max() < 1e-9


def test_weight_recovery_round_trip(built_S):
    _, dms, _, info, _ = built_S
    rng = np.random.default_rng(0)
    n = dms[0].shape[1]
    w = rng.normal(size=n)
    rec = reparam.recover_per_task_direction_weights(w, info)
    assert np.allclose(rec["w_sac"] + rec["w_pur"], 2 * rec["w_shared"])
    assert np.allclose(rec["w_sac"] - rec["w_pur"], 2 * rec["w_diff"])
    sh = info["direction_shared"]
    assert np.allclose(rec["w_shared"], w[sh["start"]:sh["end"]])


# ------------------------------------------------------------- invariants

def test_production_kwargs_pinned():
    k = invariants.PRODUCTION_FIT_KWARGS
    assert k["balance_key"] == "trial_type"
    assert k["balance_levels"] is None
    assert k["equalize_contexts"] is True
    assert k["seed"] == 42
    assert k["split_random_state"] == 0
    assert k["contexts"] == [C.CTX_SACCADE, C.CTX_PURSUIT]


def test_balancing_violation_raises():
    bad = dict(invariants.PRODUCTION_FIT_KWARGS, balance_key="choice")
    with pytest.raises(invariants.BalancingViolation, match="balance_key"):
        invariants.assert_production_balancing(bad, where="test")
    bad2 = dict(invariants.PRODUCTION_FIT_KWARGS, equalize_contexts=False)
    with pytest.raises(invariants.BalancingViolation):
        invariants.assert_production_balancing(bad2)
    invariants.assert_production_balancing(
        dict(invariants.PRODUCTION_FIT_KWARGS))     # must not raise


def test_trial_subsetting_forbidden():
    with pytest.raises(invariants.BalancingViolation):
        invariants.assert_no_trial_subsetting(trial_step=2)
    invariants.assert_no_trial_subsetting(trial_step=None)


def test_pupil_flag_refuses():
    with pytest.raises(NotImplementedError, match="pupil"):
        FEFPredictorBuilder(verbose=False, include_pupil=True)


# ------------------------------------------------------------- qc module

@pytest.fixture(scope="module")
def staged_S(glm_S):
    b = FEFPredictorBuilder(verbose=False)
    return qc.stage_design(glm_S, b, monkey="S")


def test_design_qc_verdicts_S(staged_S):
    """0136a default design clears every hard QC bound through the REAL
    staging path (stack_design_matrices z-scoring, not a hand-rolled one)."""
    r = qc.design_qc(staged_S["X"], staged_S["names"], staged_S["pinfo"],
                     staged_S["ctx_frames"], X_raw=staged_S["X_raw"],
                     feature_group_size=staged_S["feature_group_size"])
    assert r["full_rank"] and r["rank"] == r["n_cols"] == 228
    assert np.isfinite(r["cond"])
    assert r["alias_hard_ok"], (
        f'{r["worst_alias_col"]} aliases ctx_main at |r|={r["max_alias_r"]:.3f}')
    assert r["liveness_ok"], r["liveness_failures"]
    assert r["feature_group_size_ok"]


def test_design_qc_liveness_classes_S(staged_S):
    r = qc.design_qc(staged_S["X"], staged_S["names"], staged_S["pinfo"],
                     staged_S["ctx_frames"], X_raw=staged_S["X_raw"])
    live = r["liveness"]
    assert live["go"] == "sac_only" and live["delay"] == "sac_only"
    assert live["jump"] == "pur_only"
    for g in ("target", "movement", "direction_shared", "direction_diff",
              "context", "eye_velocity"):
        assert live[g] == "shared", f"{g}: {live[g]}"


def test_degeneracy_report_S(glm_S):
    b = FEFPredictorBuilder(verbose=False)
    b.build(glm_S, monkey="S")
    d = qc.degeneracy_report("0136a", "S", glm_S, builder=b, win_post=1500)
    assert d["jump_levels"] == (-1.0, 1.0)
    assert 0.4 < d["jump_forward_frac"] < 0.6          # era 1 ~50% forward
    assert d["magnitude_levels_match_config"]
    assert d["groups_dropped"] == ""
    assert d["n_blocks"] == 2 and not d["is_repeated_block"]
    assert np.isfinite(d["r_time_ctx"]) and abs(d["r_time_ctx"]) > 0.5


def test_degeneracy_report_Y(glm_Y):
    b = FEFPredictorBuilder(verbose=False)
    b.build(glm_Y, monkey="Y")
    d = qc.degeneracy_report("Ya_250619_s412", "Y", glm_Y, builder=b,
                             win_post=1350)
    assert len(d["jump_levels"]) == 1                  # all-backward pursuit
    assert d["jump_forward_frac"] == 0.0
    assert "jump" in d["groups_dropped"]
    assert d["magnitude_levels_match_config"]
    assert np.isfinite(d["r_time_ctx"])
