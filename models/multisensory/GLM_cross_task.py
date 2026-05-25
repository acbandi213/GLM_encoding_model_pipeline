#!/usr/bin/env python
"""
GLM_cross_task.py
=================
Train GLM on priming (AVC) sessions and test on paired VACANT (multisensory) sessions.

This script implements cross-task generalization:
- Training: Priming sessions (unisensory, contexts 1 & 2)
- Testing: VACANT sessions (multisensory, contexts 0, 1 & 2)

Results are saved to:
    {BASE_DATA_PATH}/{mouse_ID}/{test_date}/GLM_cross_task_results/
        glm_cross_task_full.pkl              (trained on priming, tested on VACANT)
        glm_cross_task_lesion_{group}.pkl    × 9
        glm_cross_task_lesion_all.pkl
        training_info.txt                     (documents which priming session was used)

Usage
-----
python GLM_cross_task.py
python GLM_cross_task.py --no-skip
python GLM_cross_task.py --metric dff
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

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GLM_DIR        = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
ICLOUD_GLM_DIR = ('/Users/akhilbandi/Library/Mobile Documents/'
                  'com~apple~CloudDocs/Documents/Portfolio/code_projects/'
                  'Context Paper Code/GLM_encoding_model')
for _p in (GLM_DIR, ICLOUD_GLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.session_pipeline import (
    BASE_DATA_PATH,
    open_glm_stacked_data,
    add_coupling_predictor,
    prepare_features_for_glm,
    fit_glm_single_session,
    run_lesion_models,
    compute_best_trial_avg_r2,
)
from config.session_registry_multisensory import CROSS_TASK_PAIRS, BASE_LESION_GROUPS
from core.glm_class import make_prediction, deviance
import core.glm_class as glm_class

# ==============================================================================
# Model identity
# ==============================================================================
MODEL_NAME       = 'GLM_cross_task'
MODEL_TAG        = 'cross_task'
MODEL_DIR        = 'GLM_cross_task_results'
MODEL_PREFIX     = 'glm_cross_task'
TRAIN_CONTEXTS   = [1, 2]             # Train on unisensory only (priming)
TEST_CONTEXTS    = [0, 1, 2]          # Test on all contexts (VACANT)
INCLUDE_COUPLING = False
COUPLING_PARAMS  = dict(n_pcs=5, n_lag_bases=4, lag_frames=8)

EXPECTED_LESION_GROUPS = BASE_LESION_GROUPS + (
    [f'network_pc{i + 1}' for i in range(COUPLING_PARAMS['n_pcs'])]
    if INCLUDE_COUPLING else []
)
# ==============================================================================


def _filter_contexts(X, Y, metadata, contexts):
    """Restrict X, Y, metadata to frames whose context is in contexts list."""
    mask = np.isin(metadata['context'], contexts)
    return X[mask], Y[mask], {k: v[mask] for k, v in metadata.items()}


def check_session_results(mouse_ID, test_date):
    """Return completion status for this cross-task session pair."""
    res_dir   = os.path.join(BASE_DATA_PATH, mouse_ID, test_date, MODEL_DIR)
    full_done = os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_full.pkl'))
    all_done  = os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_lesion_all.pkl'))
    lesion_done = {
        g: os.path.exists(os.path.join(res_dir, f'{MODEL_PREFIX}_lesion_{g}.pkl'))
        for g in EXPECTED_LESION_GROUPS
    }
    missing = [g for g, done in lesion_done.items() if not done]
    return {
        'full_model':            full_done,
        'lesion_all':            all_done,
        'all_complete':          full_done and all_done and not missing,
        'missing_lesion_groups': missing,
    }


class MasterLog:
    def __init__(self, log_path):
        self.path    = log_path
        self.entries = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.path, 'w') as f:
            f.write(f'{MODEL_NAME} — run started {datetime.now()}\n')
            f.write(f'Train contexts: {TRAIN_CONTEXTS}  Test contexts: {TEST_CONTEXTS}  '
                    f'Coupling: {INCLUDE_COUPLING}\n')
            f.write('=' * 70 + '\n\n')
        print(f'[log] {self.path}')

    def write(self, line, also_print=True):
        ts  = datetime.now().strftime('%H:%M:%S')
        msg = f'[{ts}] {line}'
        if also_print:
            print(msg)
        with open(self.path, 'a') as f:
            f.write(msg + '\n')

    def record(self, mouse_ID, train_date, test_date, status, detail=''):
        label = f'{mouse_ID} {train_date}→{test_date}'
        self.entries.append(dict(
            mouse_ID=mouse_ID, train_date=train_date, test_date=test_date,
            status=status, detail=detail,
            ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        n_ok   = sum(1 for e in self.entries if e['status'] == 'success')
        n_skip = sum(1 for e in self.entries if e['status'] == 'skipped')
        n_fail = sum(1 for e in self.entries if e['status'] == 'error')
        last   = self.entries[-1]
        icon   = {'success': '✓', 'skipped': '–', 'error': '✗'}.get(last['status'], '?')
        d      = f'  ({last["detail"]})' if last['detail'] else ''
        with open(self.path, 'a') as f:
            f.write(f'{icon} {label}  [{last["status"]}]{d}  @ {last["ts"]}\n')
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
                    s += f'  ✗  {e["mouse_ID"]} {e["train_date"]}→{e["test_date"]}  — {e["detail"]}\n'
        print(s)
        with open(self.path, 'a') as f:
            f.write(s)


def run_cross_task_session(mouse_ID, train_date, test_date, skip_existing=True,
                           neural_metric='deconv', log=None):
    """
    Train on priming session, test on VACANT session.
    
    Parameters
    ----------
    mouse_ID : str
    train_date : str
        Priming session date (e.g., '2026-04-08')
    test_date : str
        VACANT session date (e.g., '2026-04-08_1')
    """
    status_check = check_session_results(mouse_ID, test_date)
    
    if skip_existing and status_check['all_complete']:
        print(f'  [SKIP] {mouse_ID} {train_date}→{test_date} — all results exist')
        if log:
            log.record(mouse_ID, train_date, test_date, 'skipped')
        return {'mouse_ID': mouse_ID, 'train_date': train_date, 
                'test_date': test_date, 'status': 'skipped'}
    
    sep = '=' * 60
    print(f'\n{sep}\n  [{MODEL_NAME}] Running: {mouse_ID} {train_date}→{test_date}  '
          f'[{datetime.now().strftime("%H:%M:%S")}]\n{sep}')
    if log:
        log.write(f'START {mouse_ID} {train_date}→{test_date}', also_print=False)
    
    # Initialize all variables
    X_train = Y_train = metadata_train = None
    X_test = Y_test = metadata_test = None
    pred_info_train = pred_info_test = None
    pred_info_split = feature_group_size = None
    model_cv = all_results = lesion_results = None
    
    try:
        # ── Load and prepare training data (priming session) ──────────────
        train_stacked_path = os.path.join(BASE_DATA_PATH, mouse_ID, train_date,
                                         'glm_data_stacked.pkl')
        if not os.path.exists(train_stacked_path):
            raise FileNotFoundError(
                f'glm_data_stacked.pkl not found for training session {mouse_ID}/{train_date}. '
                f'Run prepare_glm_data.py --task priming first.')
        
        print(f'  Loading training data: {mouse_ID}/{train_date}')
        X_train, Y_train, metadata_train, _, pred_info_train, _ = \
            open_glm_stacked_data(mouse_ID, train_date, print_summary=False)
        X_train, Y_train, metadata_train = _filter_contexts(
            X_train, Y_train, metadata_train, TRAIN_CONTEXTS)
        
        if INCLUDE_COUPLING:
            X_train, pred_info_train, _ = add_coupling_predictor(
                X_train, Y_train, pred_info_train, **COUPLING_PARAMS)
        
        pred_info_split, feature_group_size = \
            prepare_features_for_glm(pred_info_train)
        
        print(f'  Training data: {X_train.shape[0]} frames, {X_train.shape[1]} features, '
              f'{Y_train.shape[1]} neurons')
        
        # ── Load and prepare test data (VACANT session) ────────────────────
        test_stacked_path = os.path.join(BASE_DATA_PATH, mouse_ID, test_date,
                                        'glm_data_stacked.pkl')
        if not os.path.exists(test_stacked_path):
            raise FileNotFoundError(
                f'glm_data_stacked.pkl not found for test session {mouse_ID}/{test_date}. '
                f'Run prepare_glm_data.py --task vacant first.')
        
        print(f'  Loading test data: {mouse_ID}/{test_date}')
        X_test, Y_test, metadata_test, _, pred_info_test, _ = \
            open_glm_stacked_data(mouse_ID, test_date, print_summary=False)
        X_test, Y_test, metadata_test = _filter_contexts(
            X_test, Y_test, metadata_test, TEST_CONTEXTS)
        
        if INCLUDE_COUPLING:
            X_test, pred_info_test, _ = add_coupling_predictor(
                X_test, Y_test, pred_info_test, **COUPLING_PARAMS)
        
        print(f'  Test data: {X_test.shape[0]} frames, {X_test.shape[1]} features, '
              f'{Y_test.shape[1]} neurons')
        
        # ── Train model on priming data ─────────────────────────────────────
        print(f'  Training GLM on priming session...')
        # Note: This will save results to train session folder as well
        model_cv, all_results, _, _ = fit_glm_single_session(
            X_train, Y_train, metadata_train, feature_group_size, pred_info_split,
            mouse_ID, train_date,
            model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
            contexts=TRAIN_CONTEXTS,
        )
        
        # ── Evaluate on VACANT test data ────────────────────────────────────
        print(f'  Evaluating on VACANT session...')
        
        # Make predictions on test data
        Y_pred_test = make_prediction(
            X_test, model_cv.selected_w, model_cv.selected_w0.flatten(), activation='exp'
        )
        
        # Compute fraction deviance explained (FDE)
        test_fde_per_neuron, _, _ = deviance(Y_pred_test, Y_test)
        test_fde_per_neuron = np.maximum(test_fde_per_neuron, 0.0)
        test_fde_mean = np.mean(test_fde_per_neuron)
        test_fde_median = np.median(test_fde_per_neuron)
        
        # Compute trial-averaged R²
        test_r2_per_neuron, test_r2_by_split = compute_best_trial_avg_r2(
            Y_test, Y_pred_test, metadata_test
        )
        test_r2_mean = np.mean(test_r2_per_neuron)
        test_r2_median = np.median(test_r2_per_neuron)
        
        print(f'  Test FDE: mean={test_fde_mean:.4f}, median={test_fde_median:.4f}, '
              f'>0.05: {np.sum(test_fde_per_neuron > 0.05)}/{len(test_fde_per_neuron)}')
        print(f'  Test R²:  mean={test_r2_mean:.4f}, median={test_r2_median:.4f}, '
              f'>0.1: {np.sum(test_r2_per_neuron > 0.1)}/{len(test_r2_per_neuron)}')
        
        # Update results with test performance
        all_results['test_fde_per_neuron'] = test_fde_per_neuron
        all_results['test_fde_mean'] = test_fde_mean
        all_results['test_fde_median'] = test_fde_median
        all_results['test_r2_per_neuron'] = test_r2_per_neuron
        all_results['test_r2_mean'] = test_r2_mean
        all_results['test_r2_median'] = test_r2_median
        all_results['test_r2_by_split'] = test_r2_by_split
        all_results['train_session'] = f'{mouse_ID}/{train_date}'
        all_results['test_session'] = f'{mouse_ID}/{test_date}'
        all_results['train_contexts'] = TRAIN_CONTEXTS
        all_results['test_contexts'] = TEST_CONTEXTS
        
        # Save full model results to test session directory
        res_dir = os.path.join(BASE_DATA_PATH, mouse_ID, test_date, MODEL_DIR)
        os.makedirs(res_dir, exist_ok=True)
        
        full_path = os.path.join(res_dir, f'{MODEL_PREFIX}_full.pkl')
        with open(full_path, 'wb') as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  ✓ Saved: {MODEL_PREFIX}_full.pkl')
        
        # Save training info file
        info_path = os.path.join(res_dir, 'training_info.txt')
        with open(info_path, 'w') as f:
            f.write(f'Cross-task GLM Model\n')
            f.write(f'{"=" * 50}\n')
            f.write(f'Trained on: {mouse_ID}/{train_date} (priming, contexts {TRAIN_CONTEXTS})\n')
            f.write(f'Tested on:  {mouse_ID}/{test_date} (VACANT, contexts {TEST_CONTEXTS})\n')
            f.write(f'Coupling:   {INCLUDE_COUPLING}\n')
            f.write(f'Metric:     {neural_metric}\n')
            f.write(f'Timestamp:  {datetime.now()}\n\n')
            f.write(f'Test Performance:\n')
            f.write(f'  FDE mean:   {test_fde_mean:.4f}\n')
            f.write(f'  FDE median: {test_fde_median:.4f}\n')
            f.write(f'  R² mean:    {test_r2_mean:.4f}\n')
            f.write(f'  R² median:  {test_r2_median:.4f}\n')
        
        del model_cv
        model_cv = None
        gc.collect()
        
        # ── Run lesion models ───────────────────────────────────────────────
        print(f'  Running lesion analysis...')
        lesion_results = run_lesion_models(
            X_test, Y_test, metadata_test, all_results, pred_info_split,
            mouse_ID=mouse_ID, date=test_date, skip_existing=skip_existing,
            model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
        )
        
        del all_results, lesion_results
        all_results = lesion_results = None
        del X_train, Y_train, X_test, Y_test
        X_train = Y_train = X_test = Y_test = None
        gc.collect()
        
        msg = f'Completed: {mouse_ID} {train_date}→{test_date}'
        print(f'\n  ✓ {msg}  [{datetime.now().strftime("%H:%M:%S")}]')
        if log:
            log.write(f'✓ {msg}', also_print=False)
            log.record(mouse_ID, train_date, test_date, 'success')
        
        return {'mouse_ID': mouse_ID, 'train_date': train_date,
                'test_date': test_date, 'status': 'success'}
    
    except Exception as e:
        traceback.print_exc()
        msg = f'FAILED: {mouse_ID} {train_date}→{test_date} — {e}'
        print(f'\n  ✗ {msg}')
        if log:
            log.write(f'✗ {msg}', also_print=False)
            log.record(mouse_ID, train_date, test_date, 'error', str(e))
        
        return {'mouse_ID': mouse_ID, 'train_date': train_date,
                'test_date': test_date, 'status': f'error: {e}'}
    
    finally:
        del X_train, Y_train, metadata_train
        del X_test, Y_test, metadata_test
        del pred_info_train, pred_info_test, pred_info_split
        del feature_group_size, model_cv, all_results, lesion_results
        gc.collect()
        if _HAS_TF:
            tf.keras.backend.clear_session()


def run_all_cross_task_pairs(cross_task_pairs, skip_existing=True,
                             neural_metric='deconv', log=None):
    """Run cross-task GLM for all session pairs."""
    results_log = []
    total = sum(len(pairs) for pairs in cross_task_pairs.values())
    print(f'[{MODEL_NAME}] Starting: {total} cross-task pairs | '
          f'train_ctx={TRAIN_CONTEXTS} | test_ctx={TEST_CONTEXTS} | '
          f'coupling={INCLUDE_COUPLING}')
    
    for mouse_ID, pair_list in cross_task_pairs.items():
        for train_date, test_date in pair_list:
            result = run_cross_task_session(
                mouse_ID, train_date, test_date,
                skip_existing=skip_existing,
                neural_metric=neural_metric,
                log=log
            )
            results_log.append(result)
    
    if log:
        log.finalize(total)
    return results_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f'Run {MODEL_NAME} GLM pipeline (train on priming, test on VACANT).')
    parser.add_argument('--no-skip', action='store_true',
                        help='Force re-run of all session pairs')
    parser.add_argument('--metric', type=str, default='deconv',
                        help='Neural metric: deconv | dff | z_dff')
    args = parser.parse_args()
    
    log_path = os.path.join(
        GLM_DIR, 'logs',
        f'run_{MODEL_TAG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log = MasterLog(log_path)
    
    n_pairs = sum(len(pairs) for pairs in CROSS_TASK_PAIRS.values())
    log.write(f'{MODEL_NAME} | train_ctx={TRAIN_CONTEXTS} | test_ctx={TEST_CONTEXTS} | '
              f'coupling={INCLUDE_COUPLING} | {n_pairs} pairs')
    for m, pair_list in CROSS_TASK_PAIRS.items():
        for priming_date, vacant_date in pair_list:
            log.write(f'  {m}: {priming_date} → {vacant_date}')
    
    run_all_cross_task_pairs(
        CROSS_TASK_PAIRS,
        skip_existing=not args.no_skip,
        neural_metric=args.metric,
        log=log,
    )
