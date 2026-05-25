#!/usr/bin/env python
"""
GLM_CG_coupling.py
==================
Train and test the GLM on combined-guidance (CG) trials only (context 0),
with functional coupling predictors from the top 5 population PCs.

PCA is fit on the context-filtered Y (CG frames only), so the coupling
signal reflects population structure within the CG regime.

Reuses glm_data_stacked.pkl built by the base pipeline (run_sessions.py).
Results are saved to:
    {BASE_DATA_PATH}/{mouse_ID}/{date}/GLM_CG_coupling_results/
        glm_cg_cpl_full.pkl
        glm_cg_cpl_lesion_{group}.pkl  × 14  (9 base + 5 network_pc)
        glm_cg_cpl_lesion_all.pkl

Usage
-----
python GLM_CG_coupling.py
python GLM_CG_coupling.py --no-skip
python GLM_CG_coupling.py --metric dff
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
)
from config.session_registry import PPC_AVC, BASE_LESION_GROUPS

# ==============================================================================
# Model identity
# ==============================================================================
MODEL_NAME       = 'GLM_CG_coupling'
MODEL_TAG        = 'cg_cpl'
MODEL_DIR        = 'GLM_CG_coupling_results'
MODEL_PREFIX     = 'glm_cg_cpl'
CONTEXTS         = [0]                # CG only
INCLUDE_COUPLING = True
COUPLING_PARAMS  = dict(n_pcs=5, n_lag_bases=4, lag_frames=8)

EXPECTED_LESION_GROUPS = BASE_LESION_GROUPS + (
    [f'network_pc{i + 1}' for i in range(COUPLING_PARAMS['n_pcs'])]
    if INCLUDE_COUPLING else []
)
# ==============================================================================


def _filter_contexts(X, Y, metadata):
    """Restrict X, Y, metadata to frames whose context is in CONTEXTS."""
    mask = np.isin(metadata['context'], CONTEXTS)
    return X[mask], Y[mask], {k: v[mask] for k, v in metadata.items()}


def check_session_results(mouse_ID, date):
    """Return completion status for this session."""
    res_dir   = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR)
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


# ── Master log ────────────────────────────────────────────────────────────────
class MasterLog:
    def __init__(self, log_path):
        self.path    = log_path
        self.entries = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.path, 'w') as f:
            f.write(f'{MODEL_NAME} — run started {datetime.now()}\n')
            f.write(f'Contexts: {CONTEXTS}  Coupling: {INCLUDE_COUPLING}\n')
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
                     neural_metric='deconv', log=None):
    """Run the full GLM pipeline for every (mouse_ID, date) in sessions_dict."""
    results_log = []
    total = sum(len(v) for v in sessions_dict.values())
    print(f'[{MODEL_NAME}] Starting: {len(sessions_dict)} mice | {total} sessions | '
          f'contexts={CONTEXTS} | coupling={INCLUDE_COUPLING}')

    for mouse_ID, dates in sessions_dict.items():
        for date in dates:

            status = check_session_results(mouse_ID, date)

            if skip_existing and status['all_complete']:
                print(f'  [SKIP] {mouse_ID} / {date} — all results exist')
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'skipped'})
                if log:
                    log.record(mouse_ID, date, 'skipped')
                continue

            if skip_existing and status['full_model']:
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
                            X, Y, pred_info, **COUPLING_PARAMS)

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
                        X, Y, pred_info, **COUPLING_PARAMS)

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

                lesion_results = run_lesion_models(
                    X, Y, metadata, all_results, pred_info_split,
                    mouse_ID=mouse_ID, date=date, skip_existing=True,
                    model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                )

                del all_results, lesion_results
                all_results = lesion_results = None
                del X, Y
                X = Y = None
                gc.collect()

                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'success'})
                msg = f'Completed: {mouse_ID} / {date}'
                print(f'\n  ✓ {msg}  [{datetime.now().strftime("%H:%M:%S")}]')
                if log:
                    log.write(f'✓ {msg}', also_print=False)
                    log.record(mouse_ID, date, 'success')

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
                        help='Neural metric: deconv | dff | z_dff')
    args = parser.parse_args()

    log_path = os.path.join(
        GLM_DIR, 'logs',
        f'run_{MODEL_TAG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log = MasterLog(log_path)

    total = sum(len(v) for v in PPC_AVC.values())
    log.write(f'{MODEL_NAME} | contexts={CONTEXTS} | coupling={INCLUDE_COUPLING} | '
              f'n_pcs={COUPLING_PARAMS["n_pcs"]} | '
              f'{len(PPC_AVC)} mice | {total} sessions')
    for m, dates in PPC_AVC.items():
        log.write(f'  {m}: {dates}')

    run_all_sessions(
        PPC_AVC,
        skip_existing=not args.no_skip,
        neural_metric=args.metric,
        log=log,
    )
