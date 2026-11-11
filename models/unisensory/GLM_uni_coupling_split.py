#!/usr/bin/env python
"""
GLM_uni_coupling_split.py
=========================
Train and test the GLM on unisensory trials only (visual + audio; contexts 1 & 2),
with functional coupling predictors from the top 5 population PCs — fit with
SEPARATE COUPLING WEIGHTS PER CONTEXT.

Identical to GLM_uni_coupling in every respect (same pooled PCA on the
context-filtered Y, same n_pcs=5 / n_lag_bases=4 / lag_frames=8, same lambda
path, optimizer, CV procedure, bases and thresholds) except that each coupling
column is duplicated into a visual-masked and an audio-masked copy, yielding
10 predictor groups (network_pc{1..5}_vis, network_pc{1..5}_aud) of 4 columns
each in place of the 5 shared groups.

Why
---
A neuron that REMAPS across contexts (active in both, inverted relationship to
the population in each) has +w in visual and -w in audio. A single shared
coupling weight averages them toward zero, and group lasso — retain-or-discard
per group — then zeroes the group outright. Such a neuron is also CD by
construction, so "CD neurons couple less" may reduce to "neurons that remap are
poorly fit by stationary models, and we measured both axes with stationary
models". Shared coupling is the nested special case w_vis = w_aud, so a
held-out FDE comparison against GLM_uni_coupling is clean.

Three variants (--mask), all the SAME model with a different 1/2 labelling
-------------------------------------------------------------------------
The controls are not optional. The group-lasso penalty is per-group
(sum_g ||w_g||*sqrt|g|), so splitting 5 coupling groups into 10 makes this model
pay ~2x to express the IDENTICAL function as GLM_uni_coupling. Measured on
IS-2-1L/2025-03-10 it declines to pay: it shrinks each group's weights to ~0.7x,
paying 1.49x instead of 2.0x, which under-fits the coupling and lowers train AND
test FDE. So (split - shared) has a NEGATIVE null that scales with the
CV-selected lambda and cannot be assumed to be zero, nor calibrated from
synthetic data (where the clean signal selects a weak lambda and the handicap
vanishes entirely).

  --mask context   visual vs audio                 -> the real test
  --mask sham      random trials -> 1/2            -> measures the penalty handicap
  --mask time      early vs late WITHIN each ctx   -> measures session drift

  true context effect = csc_context - csc_sham
  true time effect    = csc_time    - csc_sham
  if csc_context ~ csc_time, "context-specific coupling" is really
  non-stationarity (these sessions are context-BLOCKED, ABAB, 1-5 switches).

Reuses glm_data_stacked.pkl built by the base pipeline (run_sessions.py).
Results are saved to:
    {BASE_DATA_PATH}/{mouse_ID}/{date}/GLM_uni_coupling_split_results/  [context]
                                       GLM_uni_coupling_sham_results/   [sham]
                                       GLM_uni_coupling_time_results/   [time]
        glm_uni_cpl_{split,sham,time}_full.pkl
        glm_uni_cpl_{...}_lesion_{group}.pkl  × 19  (9 base + 10 network_pc)  [--lesions]
        glm_uni_cpl_{...}_lesion_all.pkl                                      [--lesions]

Lesions are OFF by default: every downstream panel reads the nested per-context
FDE out of _full.pkl only, and the encoding-class breakdown uses the BASE
model's lesions (GLM_uni_results), which already exist. Lesions cost ~19x the
full fit and nothing currently consumes them. --lesions adds them later without
refitting the full model (_full.pkl stores the train/test masks needed to
resume).

Usage
-----
python GLM_uni_coupling_split.py
python GLM_uni_coupling_split.py --no-skip
python GLM_uni_coupling_split.py --lesions
python GLM_uni_coupling_split.py --sessions IS-2-1L/2025-03-10

--metric is kept for parity with the GLM_uni_coupling template but is VESTIGIAL
there and here: the neural metric is baked into glm_data_stacked.pkl at build
time (core/build_stacked_data.py), and open_glm_stacked_data just unpickles it.
Passing --metric dff yields byte-identical output to passing nothing. To change
metric, rebuild the stacked data.
"""

import argparse
import gc
import os
import pickle
import sys
import traceback
from datetime import datetime

import numpy as np

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GLM_DIR        = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (GLM_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.session_pipeline import (
    BASE_DATA_PATH,
    open_glm_stacked_data,
    add_coupling_predictor,
    prepare_features_for_glm,
    fit_glm_single_session,
    run_lesion_models,
)
from config.session_registry import PPC_AVC, BASE_LESION_GROUPS

# ==============================================================================
# Model identity
# ==============================================================================
MODEL_NAME       = 'GLM_uni_coupling_split'
MODEL_TAG        = 'uni_cpl_split'
MODEL_DIR        = 'GLM_uni_coupling_split_results'
MODEL_PREFIX     = 'glm_uni_cpl_split'
CONTEXTS         = [1, 2]             # visual (1) + audio (2) only
INCLUDE_COUPLING = True
SPLIT_COUPLING   = True               # separate coupling weights per split group
COUPLING_PARAMS  = dict(n_pcs=5, n_lag_bases=4, lag_frames=8)

# Which 1/2 labelling the coupling weights are split on. Set by configure_mask().
MASK_NAME        = 'context'
SPLIT_LABELS     = ('vis', 'aud')
SHAM_SEED        = 0

# Model this one is nested against — its stored train/test masks are the
# reference every variant must reproduce exactly.
REFERENCE_MODEL_DIR    = 'GLM_uni_results'
REFERENCE_MODEL_PREFIX = 'glm_uni'

COUPLING_GROUPS        = []
EXPECTED_LESION_GROUPS = []


def configure_mask(mask_name):
    """
    Point the module at one of the three split variants (see SPLIT_MASKS).

    Each variant is the SAME model with a different 1/2 labelling, written to its
    own results dir. The controls exist because the group-lasso penalty gives
    (split - shared) a negative, lambda-dependent null that must be measured, not
    assumed — see the SPLIT_MASKS block above.
    """
    global MASK_NAME, SPLIT_LABELS, MODEL_DIR, MODEL_PREFIX, MODEL_NAME, MODEL_TAG
    global COUPLING_GROUPS, EXPECTED_LESION_GROUPS
    if mask_name not in SPLIT_MASKS:
        raise SystemExit(f'  ✗ unknown mask {mask_name!r}; choose from {list(SPLIT_MASKS)}')
    _, labels, model_dir, prefix = SPLIT_MASKS[mask_name]
    MASK_NAME    = mask_name
    SPLIT_LABELS = labels
    MODEL_DIR    = model_dir
    MODEL_PREFIX = prefix
    MODEL_NAME   = f'GLM_uni_coupling_split[{mask_name}]'
    MODEL_TAG    = prefix.replace('glm_', '')
    COUPLING_GROUPS = ([f'network_pc{i + 1}_{labels[0]}' for i in range(COUPLING_PARAMS['n_pcs'])] +
                       [f'network_pc{i + 1}_{labels[1]}' for i in range(COUPLING_PARAMS['n_pcs'])]
                       ) if INCLUDE_COUPLING else []
    EXPECTED_LESION_GROUPS = BASE_LESION_GROUPS + COUPLING_GROUPS


def build_split_mask(metadata):
    """Build the per-frame 1/2 labelling for the configured mask."""
    fn = SPLIT_MASKS[MASK_NAME][0]
    return fn(metadata, seed=SHAM_SEED) if MASK_NAME == 'sham' else fn(metadata)
# ==============================================================================


def require_data_drive():
    """Fail loudly if the external results drive is not mounted."""
    if not os.path.isdir(BASE_DATA_PATH):
        raise SystemExit(
            f'\n  ✗ FATAL: results drive not mounted.\n'
            f'    Expected BASE_DATA_PATH at: {BASE_DATA_PATH}\n'
            f'    Mount the "Akhil Data" drive and re-run. '
            f'Results are never written anywhere else.\n')
    print(f'[drive] ✓ mounted: {BASE_DATA_PATH}')


def _filter_contexts(X, Y, metadata):
    """Restrict X, Y, metadata to frames whose context is in CONTEXTS."""
    mask = np.isin(metadata['context'], CONTEXTS)
    return X[mask], Y[mask], {k: v[mask] for k, v in metadata.items()}


# ── Split masks ───────────────────────────────────────────────────────────────
# All three produce a per-frame 1/2 array, so all three models have IDENTICAL
# structure: 10 coupling groups, 40 columns, the same penalty budget. Only the
# meaning of the labelling changes. That is what makes them comparable.
#
# Why the controls are not optional: the group-lasso penalty is per-group, so the
# split model pays ~2x to express the same function as the shared model. Measured
# on IS-2-1L/2025-03-10 it shrinks its weights to ~0.7x rather than pay, which
# lowers train AND test FDE — so (split - shared) has a NEGATIVE null that scales
# with the CV-selected lambda. csc_sham measures that null empirically.

def make_context_mask(metadata):
    """The real test: visual (1) vs audio (2)."""
    return metadata['context'].copy()


def make_sham_mask(metadata, seed=0):
    """
    PENALTY CALIBRATION. Within EACH context separately, that context's trials are
    assigned at RANDOM to 1/2. The labelling carries no information about context
    OR time, so a correctly-behaving model can express nothing with it that the
    shared model cannot — any (sham - shared) gain is pure penalty handicap.

    Deliberately the exact structural twin of make_time_mask: both split WITHIN
    context and 50/50, differing ONLY in the assignment rule (random vs temporal).
    That makes (time - sham) a clean read of temporal structure, with group count,
    column count, penalty budget and frame counts all held fixed.

    Balancing within context matters. An unrestricted random split over all trials
    correlates with context at ~0.05-0.13 by chance, so it would absorb a sliver of
    any real context effect and overstate the handicap. Splitting within context
    drives that correlation to ~0 by construction.

    Split by TRIAL, not by frame: frames within a trial are autocorrelated, so a
    frame-wise shuffle would let each masked copy see essentially the same signal
    and understate the handicap.
    """
    tid, ctx = metadata['trial_id'], metadata['context']
    rng = np.random.default_rng(seed)
    out = np.zeros(len(tid), dtype=int)
    for c in CONTEXTS:
        c_trials = np.unique(tid[ctx == c])
        perm = rng.permutation(c_trials)
        first = set(perm[:len(perm) // 2].tolist())
        for t in c_trials:
            out[tid == t] = 1 if t in first else 2
    if (out == 0).any():
        raise ValueError('sham mask left frames unassigned')
    return out


def make_time_mask(metadata):
    """
    NON-STATIONARITY CONTROL. Within EACH context separately, that context's
    trials are split at their temporal median: early -> 1, late -> 2.

    So code 1 = early-visual + early-audio and code 2 = late-visual + late-audio.
    Both codes contain both contexts in the same proportion as the real split, so
    this mask is orthogonal to context by construction and measures ONLY whether
    coupling drifts over the session.

    This matters because these sessions are context-BLOCKED (ABAB, 1-5 switches),
    so "context" and "time" are partly confounded and a positive csc_context is
    otherwise ambiguous between remapping and drift.

    Splitting WITHIN context (rather than first-half/second-half of the session)
    is what keeps it orthogonal: on a 3-block ABA session a naive session-half
    split would put the whole trailing A block in code 2, re-creating the context
    split under a different name. It also means the control is defined for ALL
    sessions, including 2-block ones, rather than only the 34 with >=3 blocks.
    """
    tid, ctx = metadata['trial_id'], metadata['context']
    out = np.zeros(len(tid), dtype=int)
    for c in CONTEXTS:
        c_trials = np.unique(tid[ctx == c])
        c_trials.sort()                                  # trial_id is temporal order
        cut = len(c_trials) // 2
        early = set(c_trials[:cut].tolist())
        for t in c_trials:
            out[tid == t] = 1 if t in early else 2
    if (out == 0).any():
        raise ValueError('time mask left frames unassigned')
    return out


SPLIT_MASKS = {
    'context': (make_context_mask, ('vis', 'aud'),
                'GLM_uni_coupling_split_results', 'glm_uni_cpl_split'),
    'sham':    (make_sham_mask, ('sham1', 'sham2'),
                'GLM_uni_coupling_sham_results', 'glm_uni_cpl_sham'),
    'time':    (make_time_mask, ('early', 'late'),
                'GLM_uni_coupling_time_results', 'glm_uni_cpl_time'),
}


def check_session_results(mouse_ID, date, with_lesions=False):
    """Return completion status for this session."""
    res_dir   = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR)
    full_done = os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_full.pkl'))
    all_done  = os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_lesion_all.pkl'))
    lesion_done = {
        g: os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_lesion_{g}.pkl'))
        for g in EXPECTED_LESION_GROUPS
    }
    missing = [g for g, done in lesion_done.items() if not done]
    if not with_lesions:
        # Full-fit-only mode: the full model IS the deliverable.
        return {
            'full_model':            full_done,
            'lesion_all':            all_done,
            'all_complete':          full_done,
            'missing_lesion_groups': [],
        }
    return {
        'full_model':            full_done,
        'lesion_all':            all_done,
        'all_complete':          full_done and all_done and not missing,
        'missing_lesion_groups': missing,
    }


# ── Integrity checks ──────────────────────────────────────────────────────────
def assert_split_is_live(X, pred_info_aug, metadata):
    """
    Prove the context split is real before spending hours fitting it.

    A knob that silently does nothing looks exactly like a knob that doesn't
    help, so verify the masked columns actually have the expected structure:
      1. every group is present with the expected width;
      2. each *_vis column is zero on exactly the non-visual frames (and *_aud
         on the non-audio frames) — i.e. the mask is applied at all;
      3. the vis and aud blocks have disjoint support, so they cannot be
         collinear duplicates of each other;
      4. each masked column is ~zero-mean, ~unit-variance WITHIN its support
         (constraint 5 — the scale/centering that keeps it orthogonal to that
         context's main-effect indicator).
    """
    ctx = build_split_mask(metadata)
    for tag, code in ((SPLIT_LABELS[0], 1), (SPLIT_LABELS[1], 2)):
        m = ctx == code
        for i in range(COUPLING_PARAMS['n_pcs']):
            g = f'network_pc{i + 1}_{tag}'
            if g not in pred_info_aug:
                raise AssertionError(f'split group {g!r} missing from predictor_info')
            info = pred_info_aug[g]
            w = info['end'] - info['start']
            if w != COUPLING_PARAMS['n_lag_bases']:
                raise AssertionError(
                    f'{g}: width {w} != n_lag_bases {COUPLING_PARAMS["n_lag_bases"]}')
            block = X[:, info['start']:info['end']]
            if np.any(block[~m] != 0):
                raise AssertionError(
                    f'{g}: {(block[~m] != 0).sum()} non-zero entries OUTSIDE its '
                    f'context mask — masking did not take effect')
            if np.all(block[m] == 0):
                raise AssertionError(f'{g}: identically zero INSIDE its support')
            mu, sd = block[m].mean(), block[m].std()
            if abs(mu) > 1e-4 or abs(sd - 1.0) > 1e-3:
                raise AssertionError(
                    f'{g}: within-support mean={mu:.2e} std={sd:.5f} — expected '
                    f'~0 / ~1 (constraint 5 scaling not applied correctly)')

    v0 = pred_info_aug[f'network_pc1_{SPLIT_LABELS[0]}']
    a0 = pred_info_aug[f'network_pc1_{SPLIT_LABELS[1]}']
    overlap = (X[:, v0['start']] != 0) & (X[:, a0['start']] != 0)
    if overlap.any():
        raise AssertionError(
            f'{SPLIT_LABELS[0]} and {SPLIT_LABELS[1]} coupling supports overlap on '
            f'{overlap.sum()} frames — they must partition the unisensory frames')
    n1 = int((ctx == 1).sum())
    print(f'  [check] ✓ split is live [{MASK_NAME}]: {len(COUPLING_GROUPS)} masked groups, '
          f'disjoint support, within-support mean≈0 std≈1 | '
          f'{SPLIT_LABELS[0]} {n1:,} frames / {SPLIT_LABELS[1]} {len(ctx)-n1:,}')


def assert_split_weights_differ(all_results, pred_info_aug):
    """
    After fitting: confirm the vis and aud coupling weights actually diverge.

    If they were identical for every neuron, the split would be a no-op and the
    whole test would be measuring nothing.
    """
    w = all_results['weights']
    n_pcs = COUPLING_PARAMS['n_pcs']
    v_cols = np.concatenate([
        np.arange(pred_info_aug[f'network_pc{i + 1}_{SPLIT_LABELS[0]}']['start'],
                  pred_info_aug[f'network_pc{i + 1}_{SPLIT_LABELS[0]}']['end']) for i in range(n_pcs)])
    a_cols = np.concatenate([
        np.arange(pred_info_aug[f'network_pc{i + 1}_{SPLIT_LABELS[1]}']['start'],
                  pred_info_aug[f'network_pc{i + 1}_{SPLIT_LABELS[1]}']['end']) for i in range(n_pcs)])
    w_vis, w_aud = w[v_cols], w[a_cols]

    diff      = np.abs(w_vis - w_aud).max(axis=0)
    n_differ  = int((diff > 1e-8).sum())
    n_neurons = w.shape[1]
    vis_live  = int((np.abs(w_vis).max(axis=0) > 1e-8).sum())
    aud_live  = int((np.abs(w_aud).max(axis=0) > 1e-8).sum())
    print(f'  [check] coupling weights: {SPLIT_LABELS[0]} non-zero in {vis_live}/{n_neurons} | '
          f'{SPLIT_LABELS[1]} non-zero in {aud_live}/{n_neurons} | '
          f'w_{SPLIT_LABELS[0]}≠w_{SPLIT_LABELS[1]} in {n_differ}/{n_neurons}')
    if n_differ == 0:
        raise AssertionError(
            f'w_{SPLIT_LABELS[0]} == w_{SPLIT_LABELS[1]} for EVERY neuron — the '
            f'{MASK_NAME} split is a no-op. Do not trust any downstream result.')
    return dict(n_differ=n_differ, n_neurons=n_neurons,
                vis_live=vis_live, aud_live=aud_live)


def _discard_unverified_full(mouse_ID, date, reason):
    """
    Delete a _full.pkl whose integrity checks did not pass.

    fit_glm_single_session writes _full.pkl itself, BEFORE the asserts below can
    run. check_session_results is existence-only, so a pkl left behind by a
    failed session would be indistinguishable from a good one on the next run —
    it would be reported '[SKIP] all results exist' and silently feed the
    analysis a fit whose train/test split was never verified against GLM_uni
    (constraint 4). A knob that silently does nothing looks exactly like a knob
    that doesn't help, so the unverified artifact has to go.
    """
    p = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR, f'{MODEL_PREFIX}_full.pkl')
    if os.path.exists(p):
        os.remove(p)
        print(f'  [check] ✗ removed unverified {os.path.basename(p)} — {reason}')


def assert_split_matches_reference(all_results, mouse_ID, date):
    """
    The entire analysis is a within-neuron difference of FDEs across models, so
    a mismatched train/test split would contaminate it with split noise. The
    split is a pure function of metadata + fixed seeds and never sees X, so it
    should reproduce the reference model's masks exactly — verify, don't assume.
    """
    ref_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                            REFERENCE_MODEL_DIR, f'{REFERENCE_MODEL_PREFIX}_full.pkl')
    if not os.path.exists(ref_path):
        print(f'  [check] ⚠ reference {REFERENCE_MODEL_DIR} not found — '
              f'cannot verify split identity for {mouse_ID}/{date}')
        return False
    with open(ref_path, 'rb') as f:
        ref = pickle.load(f)
    for key in ('combined_train_mask', 'combined_test_mask'):
        if not np.array_equal(all_results[key], ref[key]):
            raise AssertionError(
                f'{key} DIFFERS from {REFERENCE_MODEL_DIR} for {mouse_ID}/{date} — '
                f'FDE differences would be contaminated by split noise. Aborting.')
    for ctx in ref['test_masks']:
        if not np.array_equal(all_results['test_masks'][ctx], ref['test_masks'][ctx]):
            raise AssertionError(
                f'test_masks[{ctx}] DIFFERS from {REFERENCE_MODEL_DIR} for '
                f'{mouse_ID}/{date} — aborting.')
    print(f'  [check] ✓ train/test split identical to {REFERENCE_MODEL_DIR}')
    return True


# ── Master log ────────────────────────────────────────────────────────────────
class MasterLog:
    def __init__(self, log_path):
        self.path    = log_path
        self.entries = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.path, 'w') as f:
            f.write(f'{MODEL_NAME} — run started {datetime.now()}\n')
            f.write(f'Contexts: {CONTEXTS}  Coupling: {INCLUDE_COUPLING}  '
                    f'Split: {SPLIT_COUPLING}\n')
            f.write(f'Coupling params: {COUPLING_PARAMS}\n')
            f.write('=' * 70 + '\n\n')
        print(f'[log] {self.path}')

    def write(self, line, also_print=True):
        ts  = datetime.now().strftime('%H:%M:%S')
        msg = f'[{ts}] {line}'
        if also_print:
            print(msg)
        with open(self.path, 'a') as f:
            f.write(msg + '\n')

    def record(self, mouse_ID, date, status, detail=''):
        self.entries.append(dict(mouse_ID=mouse_ID, date=date,
                                 status=status, detail=detail,
                                 ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        n_ok   = sum(1 for e in self.entries if e['status'] == 'success')
        n_skip = sum(1 for e in self.entries if e['status'] == 'skipped')
        n_fail = sum(1 for e in self.entries if e['status'] == 'error')
        last   = self.entries[-1]
        icon   = {'success': '✓', 'skipped': '–', 'error': '✗'}.get(last['status'], '?')
        d      = f'  ({last["detail"]})' if last['detail'] else ''
        with open(self.path, 'a') as f:
            f.write(f'{icon} {last["mouse_ID"]} / {last["date"]}  '
                    f'[{last["status"]}]{d}  @ {last["ts"]}\n')
            f.write(f'\n--- ✓{n_ok}  –{n_skip}  ✗{n_fail}  '
                    f'(processed: {len(self.entries)}) ---\n\n')

    def finalize(self, total):
        n_ok   = sum(1 for e in self.entries if e['status'] == 'success')
        n_skip = sum(1 for e in self.entries if e['status'] == 'skipped')
        n_fail = sum(1 for e in self.entries if e['status'] == 'error')
        s = (f'\n{"=" * 70}\nFINISHED  {datetime.now()}\n'
             f'Total: {total}  ✓{n_ok}  –{n_skip}  ✗{n_fail}\n')
        if n_fail:
            for e in self.entries:
                if e['status'] == 'error':
                    s += f'  ✗  {e["mouse_ID"]} / {e["date"]}  — {e["detail"]}\n'
        print(s)
        with open(self.path, 'a') as f:
            f.write(s)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_all_sessions(sessions_dict, skip_existing=True,
                     neural_metric='deconv', log=None, with_lesions=False):
    """Run the full GLM pipeline for every (mouse_ID, date) in sessions_dict."""
    results_log = []
    total = sum(len(v) for v in sessions_dict.values())
    print(f'[{MODEL_NAME}] Starting: {len(sessions_dict)} mice | {total} sessions | '
          f'contexts={CONTEXTS} | coupling={INCLUDE_COUPLING} | split={SPLIT_COUPLING} | '
          f'lesions={with_lesions}')

    for mouse_ID, dates in sessions_dict.items():
        for date in dates:

            status = check_session_results(mouse_ID, date, with_lesions=with_lesions)

            if skip_existing and status['all_complete']:
                print(f'  [SKIP] {mouse_ID} / {date} — all results exist')
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'skipped'})
                if log:
                    log.record(mouse_ID, date, 'skipped')
                continue

            if skip_existing and with_lesions and status['full_model']:
                missing = status['missing_lesion_groups']
                print(f'\n  [RESUME] {mouse_ID} / {date} — '
                      f'missing lesions: {missing}')
                X = Y = metadata = pred_info = pred_info_split = all_results = None
                try:
                    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                                             MODEL_DIR, f'{MODEL_PREFIX}_full.pkl')
                    with open(full_path, 'rb') as f:
                        all_results = pickle.load(f)

                    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(
                        mouse_ID, date, print_summary=False)
                    X, Y, metadata = _filter_contexts(X, Y, metadata)

                    if INCLUDE_COUPLING:
                        X, pred_info, _ = add_coupling_predictor(
                            X, Y, pred_info, **COUPLING_PARAMS,
                            split_by_context=metadata['context'] if SPLIT_COUPLING else None)
                        assert_split_is_live(X, pred_info, metadata)

                    pred_info_split, _ = prepare_features_for_glm(pred_info)
                    run_lesion_models(
                        X, Y, metadata, all_results, pred_info_split,
                        mouse_ID=mouse_ID, date=date, skip_existing=True,
                        model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                    )
                    results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                        'status': 'success'})
                    if log:
                        log.record(mouse_ID, date, 'success', 'lesion-resume')

                except Exception as e:
                    traceback.print_exc()
                    results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                        'status': f'error: {e}'})
                    if log:
                        log.record(mouse_ID, date, 'error', str(e))

                finally:
                    del X, Y, metadata, pred_info, pred_info_split, all_results
                    gc.collect()
                    if _HAS_TF:
                        tf.keras.backend.clear_session()
                continue

            sep = '=' * 60
            print(f'\n{sep}\n  [{MODEL_NAME}] Running: {mouse_ID} / {date}  '
                  f'[{datetime.now().strftime("%H:%M:%S")}]\n{sep}')
            if log:
                log.write(f'START {mouse_ID} / {date}', also_print=False)

            t_start = datetime.now()
            X = Y = metadata = pred_info = pred_info_split = None
            feature_group_size = model_cv = all_results = lesion_results = None

            try:
                stacked_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                                            'glm_data_stacked.pkl')
                if not os.path.exists(stacked_path):
                    raise FileNotFoundError(
                        f'glm_data_stacked.pkl not found for {mouse_ID}/{date}. '
                        f'Run run_sessions.py first to build the stacked data.')

                X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(
                    mouse_ID, date, print_summary=False)
                X, Y, metadata = _filter_contexts(X, Y, metadata)

                if INCLUDE_COUPLING:
                    X, pred_info, _ = add_coupling_predictor(
                        X, Y, pred_info, **COUPLING_PARAMS,
                        split_by_context=build_split_mask(metadata) if SPLIT_COUPLING else None,
                        split_labels=SPLIT_LABELS)
                    assert_split_is_live(X, pred_info, metadata)

                pred_info_split, feature_group_size = \
                    prepare_features_for_glm(pred_info)

                model_cv, all_results, _, _ = fit_glm_single_session(
                    X, Y, metadata, feature_group_size, pred_info_split,
                    mouse_ID, date,
                    model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                    contexts=CONTEXTS,
                )
                del model_cv
                model_cv = None
                gc.collect()

                try:
                    assert_split_matches_reference(all_results, mouse_ID, date)
                    assert_split_weights_differ(all_results, pred_info_split)
                except AssertionError as e:
                    _discard_unverified_full(mouse_ID, date, str(e)[:60])
                    raise

                if with_lesions:
                    lesion_results = run_lesion_models(
                        X, Y, metadata, all_results, pred_info_split,
                        mouse_ID=mouse_ID, date=date, skip_existing=skip_existing,
                        model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                    )
                    del lesion_results
                    lesion_results = None

                del all_results
                all_results = None
                del X, Y
                X = Y = None
                gc.collect()

                mins = (datetime.now() - t_start).total_seconds() / 60
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'success'})
                msg = f'Completed: {mouse_ID} / {date}  ({mins:.1f} min)'
                print(f'\n  ✓ {msg}  [{datetime.now().strftime("%H:%M:%S")}]')
                if log:
                    log.write(f'✓ {msg}', also_print=False)
                    log.record(mouse_ID, date, 'success', f'{mins:.1f} min')

            except Exception as e:
                traceback.print_exc()
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': f'error: {e}'})
                msg = f'FAILED: {mouse_ID} / {date} — {e}'
                print(f'\n  ✗ {msg}')
                if log:
                    log.write(f'✗ {msg}', also_print=False)
                    log.record(mouse_ID, date, 'error', str(e))

            finally:
                del X, Y, metadata, pred_info, pred_info_split
                del feature_group_size, model_cv, all_results, lesion_results
                gc.collect()
                if _HAS_TF:
                    tf.keras.backend.clear_session()

    if log:
        log.finalize(total)
    return results_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f'Run {MODEL_NAME} GLM pipeline.')
    parser.add_argument('--no-skip', action='store_true',
                        help='Force re-run of all sessions')
    parser.add_argument('--metric', type=str, default='deconv',
                        help='VESTIGIAL (template parity): the metric is baked into '
                             'glm_data_stacked.pkl at build time, so this changes nothing')
    parser.add_argument('--lesions', action='store_true',
                        help='Also refit the 19 lesion models per session '
                             '(~19x the cost; nothing downstream needs them)')
    parser.add_argument('--sessions', type=str, nargs='+', default=None,
                        help='Limit to MOUSE/DATE pairs, e.g. IS-2-1L/2025-03-10')
    parser.add_argument('--mask', type=str, default='context',
                        choices=list(SPLIT_MASKS),
                        help="Which 1/2 labelling to split coupling weights on. "
                             "'context' = the real test (visual vs audio). "
                             "'sham' = random trials, calibrates the group-lasso penalty "
                             "handicap (the null for split-shared is NOT zero). "
                             "'time' = early vs late within each context, separates "
                             "remapping from session drift. Each writes its own results dir.")
    parser.add_argument('--sham-seed', type=int, default=0,
                        help='RNG seed for the sham trial assignment (--mask sham)')
    args = parser.parse_args()

    configure_mask(args.mask)
    SHAM_SEED = args.sham_seed
    require_data_drive()

    sessions = PPC_AVC
    if args.sessions:
        sessions = {}
        for s in args.sessions:
            m, d = s.split('/')
            if m not in PPC_AVC or d not in PPC_AVC[m]:
                raise SystemExit(f'  ✗ {s} is not in the session registry (PPC_AVC).')
            sessions.setdefault(m, []).append(d)

    log_path = os.path.join(
        GLM_DIR, 'logs',
        f'run_{MODEL_TAG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log = MasterLog(log_path)

    total = sum(len(v) for v in sessions.values())
    log.write(f'{MODEL_NAME} | contexts={CONTEXTS} | coupling={INCLUDE_COUPLING} | '
              f'split={SPLIT_COUPLING} | lesions={args.lesions} | '
              f'n_pcs={COUPLING_PARAMS["n_pcs"]} | '
              f'{len(sessions)} mice | {total} sessions')
    for m, dates in sessions.items():
        log.write(f'  {m}: {dates}')

    run_all_sessions(
        sessions,
        skip_existing=not args.no_skip,
        neural_metric=args.metric,
        log=log,
        with_lesions=args.lesions,
    )
