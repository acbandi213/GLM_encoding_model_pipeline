"""
qc.py
=====
WP4e design-matrix QC as library functions (blueprint:
prompts/WP4_IMPLEMENTATION_PLAN.md §3.1).

Everything numeric is computed on the SAME X the GLM will see:
`stack_design_matrices(glm, dms, pinfo, neural_metric='spikes')`, which
z-scores — not a hand-rolled zscore — so the QC numbers are the fit's numbers.
The one exception is task-LIVENESS, which is meaningless after z-scoring
(centering makes a task-dead column nonzero on its dead task's frames), so it
is computed on the raw stacked design and the staging helper carries both.

Acceptance targets (docs/05_PREDICTORS.md §7; v1 baselines at 204 cols:
cond 77.4, max |r| with ctx_main 0.299 — do NOT carry over, re-measure):

    full rank                 all columns
    condition number          < 100   (fallback: reduce DIRECTION_HARMONICS
                                       before dropping groups, and report)
    max |r| with ctx_main     < 0.35 target, < 0.5 hard
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

# repo root (…/GLM_encoding_model, which holds core/) is four levels up:
# design -> fef -> FEF_v2 -> NHP_FEF -> GLM_encoding_model
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     *(['..'] * 4)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.session_pipeline import (stack_design_matrices,          # noqa: E402
                                   prepare_features_for_glm)

from .. import config as C                                         # noqa: E402

#: Full-rank criterion, identical to tests/test_design.py: the smallest
#: eigenvalue of X'X must exceed this fraction of the largest.
RANK_TOL = 1e-8

COND_TARGET = 100.0        # docs/05_PREDICTORS.md §7
ALIAS_TARGET = 0.35        # reported target
ALIAS_HARD = 0.50          # hard bound (tests assert this)
CROSS_GROUP_FLAG = 0.85    # v1's expected within-family ceiling 0.84-0.85


def stage_design(glm, builder, monkey=None):
    """
    convert-dict + builder -> everything design_qc needs, via the exact
    staging the GLM uses (`stack_design_matrices` -> `prepare_features_for_glm`)
    but WITHOUT any fit. Both helpers print; their output is swallowed so a
    26-session sweep log stays readable.

    Returns dict with X (z-scored), X_raw, names, pinfo, ctx_frames,
    feature_group_size, builder.
    """
    dms, names, pinfo = builder.build(glm, monkey=monkey)
    with redirect_stdout(io.StringIO()):
        X, _Y, meta, _gi = stack_design_matrices(glm, dms, pinfo,
                                                 neural_metric='spikes')
        _split, fgs = prepare_features_for_glm(pinfo)
    return dict(X=X, X_raw=np.vstack(dms), names=names, pinfo=pinfo,
                ctx_frames=meta['context'], feature_group_size=fgs,
                builder=builder)


def _group_liveness(X_raw, pinfo, ctx_frames):
    """
    Classify every group: 'shared' | 'sac_only' | 'pur_only' | 'PARTIAL' |
    'DEAD'. PARTIAL (some columns dead in a task, others not) and DEAD (all
    columns zero in both tasks) are failures — a group half-defined on a task
    is silently mis-penalised by the group lasso (v1
    test_shared_groups_live_in_both_tasks).
    """
    sac = ctx_frames == C.CTX_SACCADE
    pur = ctx_frames == C.CTX_PURSUIT
    out = {}
    for g, m in pinfo.items():
        cols = X_raw[:, m['start']:m['end']]
        w = m['end'] - m['start']
        dead_s = int((np.abs(cols[sac]).max(axis=0) <= 1e-12).sum())
        dead_p = int((np.abs(cols[pur]).max(axis=0) <= 1e-12).sum())
        if dead_s == 0 and dead_p == 0:
            out[g] = 'shared'
        elif dead_s == w and dead_p == w:
            out[g] = 'DEAD'
        elif dead_p == w and dead_s == 0:
            out[g] = 'sac_only'
        elif dead_s == w and dead_p == 0:
            out[g] = 'pur_only'
        else:
            out[g] = f'PARTIAL(sac_dead {dead_s}/{w}, pur_dead {dead_p}/{w})'
    return out


def design_qc(X, names, pinfo, ctx_frames, X_raw=None,
              feature_group_size=None, top_k=10):
    """
    The WP4 acceptance measurements for one staged design.

    Parameters
    ----------
    X : (n_frames, n_cols) z-scored stacked design (from stage_design)
    names, pinfo : the builder's outputs
    ctx_frames : (n_frames,) per-frame context code
    X_raw : raw stacked design, required for the liveness classes
    feature_group_size : from prepare_features_for_glm, for the sum check
    top_k : how many worst aliasing columns / cross-group pairs to name

    Returns a flat dict; keys ending in `_ok` are the acceptance verdicts.
    """
    n, p = X.shape

    # ---- rank + condition number, one Gram decomposition for both ----------
    G = X.T @ X
    ev = np.linalg.eigvalsh(G)
    ev_max = float(ev[-1])
    rank = int((ev > RANK_TOL * ev_max).sum())
    full_rank = bool(ev[0] > RANK_TOL * ev_max)
    # 2-norm condition number of X = sqrt(ev_max/ev_min) of X'X. Reported on
    # the positive spectrum so a rank-deficient design gets a finite "cond of
    # the live subspace" alongside full_rank=False rather than inf.
    ev_pos = ev[ev > RANK_TOL * ev_max]
    cond = float(np.sqrt(ev_max / ev_pos[0])) if ev_pos.size else np.inf

    # ---- context aliasing ---------------------------------------------------
    # X is z-scored with ddof=0, so R = G/n is the correlation matrix exactly
    # (constant columns were zeroed by nan_to_num and get r = 0).
    R = G / n
    ctx_idx = names.index('ctx_main')
    r_ctx = R[:, ctx_idx].copy()
    outside = np.ones(p, bool)
    ci = pinfo['context']
    outside[ci['start']:ci['end']] = False
    a = np.where(outside, np.abs(r_ctx), 0.0)
    order = np.argsort(-a)[:top_k]
    alias_top = [(names[i], float(r_ctx[i])) for i in order if a[i] > 0]
    max_alias = float(a.max())
    n_over_target = int((a > ALIAS_TARGET).sum())

    # ---- cross-group |r| summary -------------------------------------------
    gi = np.empty(p, dtype=object)
    for g, m in pinfo.items():
        gi[m['start']:m['end']] = g
    R_abs = np.abs(R)
    np.fill_diagonal(R_abs, 0.0)
    same = gi[:, None] == gi[None, :]
    cross = np.where(same, 0.0, R_abs)
    iu = np.triu_indices(p, k=1)
    vals = cross[iu]
    order = np.argsort(-vals)[:top_k]
    cross_top = [(names[iu[0][k]], names[iu[1][k]], float(R[iu[0][k], iu[1][k]]))
                 for k in order if vals[k] > 0]
    max_cross = float(vals.max()) if vals.size else 0.0
    n_cross_flagged = int((vals > CROSS_GROUP_FLAG).sum())

    # ---- liveness -----------------------------------------------------------
    liveness = (_group_liveness(X_raw, pinfo, ctx_frames)
                if X_raw is not None else {})
    liveness_failures = [g for g, v in liveness.items()
                         if v.startswith(('PARTIAL', 'DEAD'))]

    # ---- feature_group_size sum check (train.py::prepare port) -------------
    fgs_ok = (bool(int(np.sum(feature_group_size)) == p)
              if feature_group_size is not None else None)

    return dict(
        n_frames=n, n_cols=p,
        rank=rank, full_rank=full_rank,
        min_max_eig_ratio=float(ev[0] / ev_max),
        cond=cond,
        max_alias_r=max_alias, worst_alias_col=(alias_top[0][0] if alias_top
                                                else ''),
        alias_top=alias_top, n_alias_over_target=n_over_target,
        max_cross_group_r=max_cross, cross_group_top=cross_top,
        n_cross_group_flagged=n_cross_flagged,
        liveness=liveness, liveness_failures=liveness_failures,
        feature_group_size_ok=fgs_ok,
        rank_ok=full_rank,
        cond_ok=bool(cond < COND_TARGET),
        alias_ok=bool(max_alias < ALIAS_TARGET),
        alias_hard_ok=bool(max_alias < ALIAS_HARD),
        liveness_ok=not liveness_failures,
    )


def _expected_levels(session_key, monkey, which):
    """Config-expected condition levels ('eccentricity_deg'|'ramp_speed_dps'),
    with SESSION_CONDITION_EXCEPTIONS overriding MONKEY_CONDITIONS."""
    base = C.MONKEY_CONDITIONS[monkey][which]
    return tuple(C.SESSION_CONDITION_EXCEPTIONS
                 .get(session_key, {}).get(which, base))


def _levels(vals):
    """Sorted unique finite levels, rounded to kill float jitter."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return tuple(sorted(set(np.round(v, 3))))


def degeneracy_report(session_key, monkey, glm, builder=None, win_post=None):
    """
    Per-session degeneracy accounting (plan §3.1): blocks pooled with kept
    trial counts, jump levels + forward fraction, magnitude levels per task
    checked against config, groups dropped, and r(abs_time, ctx_main) — the
    time-vs-task decorrelation that was never measured for the repeated-block
    sessions 0145a and Ya_250522_s399 (A6-2; v1 band 0.72-0.87, interleaved
    sessions 0146a 0.512 / 0147a 0.773).

    `builder` should be the (already-built) default-config builder so its
    `dropped_groups` reflect the real degeneracy decisions.
    """
    trials = [glm[k] for k in sorted(glm)]
    ctx = np.array([int(t['context']) for t in trials])
    ctx_code = np.where(ctx == C.CTX_SACCADE, 1.0, -1.0)

    blocks = {}
    for t in trials:
        blocks[t['block']] = blocks.get(t['block'], 0) + 1

    pur = ctx == C.CTX_PURSUIT
    jump = np.array([float(t.get('jump', np.nan)) for t in trials])
    jump_levels = _levels(jump[pur])
    fwd_frac = (float((jump[pur] > 0).mean()) if pur.any() else np.nan)

    dist = np.array([float(t.get('distance', np.nan)) for t in trials])
    speed = np.array([float(t.get('speed', np.nan)) for t in trials])
    ecc_meas = _levels(dist[ctx == C.CTX_SACCADE])
    spd_meas = _levels(speed[pur])
    ecc_exp = _expected_levels(session_key, monkey, 'eccentricity_deg')
    spd_exp = _expected_levels(session_key, monkey, 'ramp_speed_dps')
    # Measured on kept correct trials vs config measured on ALL trials — a
    # level absent here is a stop-and-report (S10), not a silent absorb.
    mag_match = (ecc_meas == tuple(sorted(ecc_exp))
                 and spd_meas == tuple(sorted(spd_exp)))

    at = np.array([float(t['abs_time']) for t in trials])
    ok = np.isfinite(at)
    r_time_ctx = (float(np.corrcoef(at[ok], ctx_code[ok])[0, 1])
                  if ok.sum() > 2 and np.unique(ctx_code[ok]).size > 1
                  else np.nan)

    return dict(
        session=session_key, monkey=monkey,
        win_post_ms=win_post,
        n_trials=len(trials),
        n_sac=int((ctx == C.CTX_SACCADE).sum()),
        n_pur=int(pur.sum()),
        blocks=';'.join(f'{b}:{n}' for b, n in sorted(blocks.items())),
        n_blocks=len(blocks),
        is_repeated_block=session_key in C.REPEATED_BLOCK_SESSIONS,
        jump_levels=jump_levels, jump_forward_frac=fwd_frac,
        ecc_levels_measured=ecc_meas, ecc_levels_expected=tuple(sorted(ecc_exp)),
        speed_levels_measured=spd_meas,
        speed_levels_expected=tuple(sorted(spd_exp)),
        magnitude_levels_match_config=bool(mag_match),
        groups_dropped=';'.join(sorted(builder.dropped_groups))
        if builder is not None else '',
        drop_reasons=' | '.join(f'{g}: {why}' for g, why in
                                sorted(builder.dropped_groups.items()))
        if builder is not None else '',
        r_time_ctx=r_time_ctx,
    )
