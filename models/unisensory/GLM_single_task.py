#!/usr/bin/env python
"""
GLM_single_task.py
==================
Train and test the GLM on ONE task only: visual, audio, or congruent (CG).

    --task visual   context 1   results in GLM_visual_results/  (glm_visual_*.pkl)
    --task audio    context 2   results in GLM_audio_results/   (glm_audio_*.pkl)
    --task cg       context 0   results in GLM_cg_single_results/ (glm_cg_single_*.pkl)

Add --coupling for the population-coupling variant
(GLM_visual_coupling_results/, glm_visual_coupling_*.pkl, ...).

What is different from GLM_uni / GLM_CG
---------------------------------------
Within a single unisensory task the predictors of the OTHER modality are
zero-filled on every trial, so after z-scoring they are constant columns. This
script removes those two groups before fitting and does not lesion them:

    visual : drops audio_context, audio_stim
    audio  : drops visual_context, visual_stim
    cg     : drops nothing (both cues are present)

The task's own *_context group is kept. In a one-task fit it is no longer a
"context" term: it is the mean cue-locked response shared by all trials of that
task (the *_stim group is the left/right cue difference).

Reuses glm_data_stacked.pkl built by the base pipeline. Results are saved to
    {BASE_DATA_PATH}/{mouse_ID}/{date}/{MODEL_DIR}/
        {prefix}_full.pkl
        {prefix}_lesion_{group}.pkl
        {prefix}_lesion_all.pkl

Usage
-----
python GLM_single_task.py --task visual
python GLM_single_task.py --task audio --coupling
python GLM_single_task.py --task cg --no-skip
python GLM_single_task.py --task visual --mouse KO-5-1L --date 2026-03-06
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_DIR     = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if GLM_DIR not in sys.path:
    sys.path.insert(0, GLM_DIR)

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
# Task table — the only place that knows about the three tasks.
#   context : context code in metadata['context']  (0 = CG, 1 = visual, 2 = audio)
#   drop    : predictor groups that carry nothing in this task (removed from X)
# ==============================================================================
TASKS = {
    'visual': dict(context=1, tag='visual',    drop=['audio_context', 'audio_stim']),
    'audio':  dict(context=2, tag='audio',     drop=['visual_context', 'visual_stim']),
    'cg':     dict(context=0, tag='cg_single', drop=[]),
}
COUPLING_PARAMS = dict(n_pcs=5, n_lag_bases=4, lag_frames=8)

# Model identity. Filled in by configure(); the defaults are the visual task.
TASK = MODEL_NAME = MODEL_TAG = MODEL_DIR = MODEL_PREFIX = None
CONTEXTS = DROP_GROUPS = EXPECTED_LESION_GROUPS = None
INCLUDE_COUPLING = False


def configure(task='visual', coupling=False):
    """Set the module-level model identity for one task. Call before running."""
    global TASK, MODEL_NAME, MODEL_TAG, MODEL_DIR, MODEL_PREFIX
    global CONTEXTS, DROP_GROUPS, EXPECTED_LESION_GROUPS, INCLUDE_COUPLING
    if task not in TASKS:
        raise ValueError(f'task must be one of {list(TASKS)}, got {task!r}')
    cfg = TASKS[task]
    tag = cfg['tag'] + ('_coupling' if coupling else '')
    TASK             = task
    MODEL_TAG        = tag
    MODEL_NAME       = f'GLM_{tag}'
    MODEL_DIR        = f'GLM_{tag}_results'
    MODEL_PREFIX     = f'glm_{tag}'
    CONTEXTS         = [cfg['context']]
    DROP_GROUPS      = list(cfg['drop'])
    INCLUDE_COUPLING = bool(coupling)
    EXPECTED_LESION_GROUPS = (
        [g for g in BASE_LESION_GROUPS if g not in DROP_GROUPS]
        + ([f'network_pc{i + 1}' for i in range(COUPLING_PARAMS['n_pcs'])]
           if coupling else [])
    )


configure()


# ── Data helpers ──────────────────────────────────────────────────────────────
def _filter_contexts(X, Y, metadata):
    """Restrict X, Y, metadata to frames whose context is in CONTEXTS."""
    mask = np.isin(metadata['context'], CONTEXTS)
    return X[mask], Y[mask], {k: v[mask] for k, v in metadata.items()}


def drop_predictor_groups(X, pred_info, drop_groups, tol=1e-8):
    """
    Remove whole predictor groups from X and re-index pred_info.

    Each dropped group must be constant over the frames in X (that is the reason
    it is dropped); any OTHER group that is entirely constant raises, because it
    would be an empty predictor the model cannot use.

    Returns
    -------
    X_new              : ndarray (n_frames, n_kept_features)
    pred_info_new      : dict, same group order, contiguous start / end
    feature_group_size : ndarray of int
    """
    col_sd = X.std(axis=0)
    keep_cols, new_info, pos = [], {}, 0
    for group, info in pred_info.items():
        s, e = info['start'], info['end']
        all_const = bool(np.all(col_sd[s:e] < tol))
        if group in drop_groups:
            if not all_const:
                raise ValueError(
                    f"group '{group}' was meant to be dropped as empty, but it "
                    f'varies in these frames. Check the context filter.')
            continue
        if all_const:
            raise ValueError(
                f"group '{group}' is constant in these frames but is not in the "
                f'drop list {drop_groups}.')
        n = e - s
        keep_cols.extend(range(s, e))
        new_info[group] = dict(info, start=pos, end=pos + n)
        pos += n
    missing = [g for g in drop_groups if g not in pred_info]
    if missing:
        raise KeyError(f'groups to drop not found in predictor_info: {missing}')
    sizes = np.array([i['end'] - i['start'] for i in new_info.values()])
    if drop_groups:
        print(f'  [{MODEL_NAME}] dropped empty groups {drop_groups}: '
              f'{X.shape[1]} -> {len(keep_cols)} features')
    return X[:, keep_cols], new_info, sizes


def load_design(mouse_ID, date):
    """Stacked data for this task: context-filtered, coupling added, empty groups removed."""
    stacked_path = os.path.join(BASE_DATA_PATH, mouse_ID, date, 'glm_data_stacked.pkl')
    if not os.path.exists(stacked_path):
        raise FileNotFoundError(
            f'glm_data_stacked.pkl not found for {mouse_ID}/{date}. '
            f'Build it first (runners/run_unisensory_models.py --build-stacked).')
    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(
        mouse_ID, date, print_summary=False)
    X, Y, metadata = _filter_contexts(X, Y, metadata)
    if X.shape[0] == 0:
        raise ValueError(f'no frames for task {TASK!r} (context {CONTEXTS}) '
                         f'in {mouse_ID}/{date}')
    if INCLUDE_COUPLING:
        X, pred_info, _ = add_coupling_predictor(X, Y, pred_info, **COUPLING_PARAMS)
    pred_info_split, _ = prepare_features_for_glm(pred_info)
    X, pred_info_split, feature_group_size = drop_predictor_groups(
        X, pred_info_split, DROP_GROUPS)
    return X, Y, metadata, pred_info_split, feature_group_size


def check_session_results(mouse_ID, date):
    """Return completion status for this session."""
    res_dir   = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR)
    full_done = os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_full.pkl'))
    all_done  = os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_lesion_all.pkl'))
    missing = [g for g in EXPECTED_LESION_GROUPS
               if not os.path.exists(
                   os.path.join(res_dir, f'{MODEL_PREFIX}_lesion_{g}.pkl'))]
    return {
        'full_model':            full_done,
        'lesion_all':            all_done,
        'all_complete':          full_done and all_done and not missing,
        'missing_lesion_groups': missing,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_session(mouse_ID, date, skip_existing=True, lesions=True):
    """Fit one session. Returns 'skipped' or 'success'; raises on failure."""
    status = check_session_results(mouse_ID, date)
    if skip_existing and status['all_complete']:
        print(f'  [SKIP] {mouse_ID} / {date} — all results exist')
        return 'skipped'

    print(f'\n{"=" * 60}\n  [{MODEL_NAME}] {mouse_ID} / {date}  '
          f'[{datetime.now().strftime("%H:%M:%S")}]\n{"=" * 60}')
    X, Y, metadata, pred_info, feature_group_size = load_design(mouse_ID, date)

    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR,
                             f'{MODEL_PREFIX}_full.pkl')
    if skip_existing and status['full_model']:
        print(f'  [RESUME] missing lesions: {status["missing_lesion_groups"]}')
        with open(full_path, 'rb') as f:
            all_results = pickle.load(f)
    else:
        model_cv, all_results, _, _ = fit_glm_single_session(
            X, Y, metadata, feature_group_size, pred_info, mouse_ID, date,
            model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX, contexts=CONTEXTS)
        del model_cv
        gc.collect()

    if lesions:
        run_lesion_models(
            X, Y, metadata, all_results, pred_info,
            mouse_ID=mouse_ID, date=date, skip_existing=skip_existing,
            model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX)
    return 'success'


def run_all_sessions(sessions_dict, skip_existing=True, neural_metric='deconv',
                     log_path=None, lesions=True):
    """Run every (mouse_ID, date) in sessions_dict. One failure does not stop the rest."""
    def _log(line):
        print(line)
        if log_path:
            with open(log_path, 'a') as f:
                f.write(f'[{datetime.now().strftime("%H:%M:%S")}] {line}\n')

    total = sum(len(v) for v in sessions_dict.values())
    _log(f'{MODEL_NAME} | task={TASK} contexts={CONTEXTS} coupling={INCLUDE_COUPLING} '
         f'| {len(sessions_dict)} mice | {total} sessions')
    results_log = []
    for mouse_ID, dates in sessions_dict.items():
        for date in dates:
            try:
                status = run_session(mouse_ID, date, skip_existing=skip_existing,
                                     lesions=lesions)
                _log(f'{"–" if status == "skipped" else "✓"} {mouse_ID} / {date}  [{status}]')
            except Exception as e:
                traceback.print_exc()
                status = f'error: {e}'
                _log(f'✗ {mouse_ID} / {date}  [{status}]')
            finally:
                gc.collect()
                if _HAS_TF:
                    tf.keras.backend.clear_session()
            results_log.append({'mouse_ID': mouse_ID, 'date': date, 'status': status})

    n_fail = sum(r['status'].startswith('error') for r in results_log)
    _log(f'FINISHED  total {total}  failed {n_fail}')
    return results_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit the GLM on one task only.')
    parser.add_argument('--task', required=True, choices=list(TASKS),
                        help='Which task to fit.')
    parser.add_argument('--coupling', action='store_true',
                        help='Add population-coupling predictors.')
    parser.add_argument('--no-skip', action='store_true',
                        help='Force re-run of all sessions')
    parser.add_argument('--no-lesions', action='store_true',
                        help='Fit the full model only, skip the lesion refits.')
    parser.add_argument('--mouse', type=str, default=None,
                        help='Run one mouse only (with --date: one session).')
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--metric', type=str, default='deconv',
                        help='Neural metric: deconv | dff | z_dff')
    args = parser.parse_args()

    configure(args.task, args.coupling)

    sessions = PPC_AVC
    if args.mouse:
        sessions = {args.mouse: [args.date] if args.date else PPC_AVC[args.mouse]}

    log_dir = os.path.join(GLM_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f'run_{MODEL_TAG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    run_all_sessions(sessions, skip_existing=not args.no_skip,
                     neural_metric=args.metric, log_path=log_path,
                     lesions=not args.no_lesions)
