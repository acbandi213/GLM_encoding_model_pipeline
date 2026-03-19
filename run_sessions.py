#!/usr/bin/env python
"""
run_sessions.py
===============
Command-line runner for the Model 1 GLM pipeline.
Runs all sessions sequentially in a single process.

Usage
-----
# Run all sessions (skip already-completed ones):
python run_sessions.py

# Force re-run of everything:
python run_sessions.py --no-skip

# Use a different neural metric:
python run_sessions.py --metric dff

A timestamped master log is written to:
    GLM_encoding_model/logs/run_YYYYMMDD_HHMMSS.log
It is updated after every session so you can tail -f it to watch progress.
"""

import argparse
import gc
import os
import pickle
import sys
import traceback
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────
GLM_DIR = os.path.dirname(os.path.abspath(__file__))
ICLOUD_GLM_DIR = '/Users/akhilbandi/Library/Mobile Documents/com~apple~CloudDocs/Documents/Portfolio/code_projects/Context Paper Code/GLM_encoding_model'
for _p in (GLM_DIR, ICLOUD_GLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Pipeline imports ──────────────────────────────────────────────────────
from session_pipeline import (
    BASE_DATA_PATH,
    MODEL_1_DIR,
    extract_data,
    generate_predictors,
    stack_design_matrices,
    save_glm_data,
    open_glm_stacked_data,
    prepare_features_for_glm,
    fit_glm_single_session,
    run_lesion_models,
)

# ── Full session registry ─────────────────────────────────────────────────
PPC_AVC = {
    'JP-2-1L': ['2025-09-17', '2025-09-19', '2025-09-22'],
    'KA-1-00': ['2025-12-16', '2026-01-08'],
    'KA-2-00': ['2026-01-08', '2026-01-13', '2026-01-14', '2026-01-28'],
    'KO-5-1L': ['2026-03-05', '2026-03-06', '2026-03-12', '2026-03-13'],
    'KO-6-00': ['2026-03-12']
}

EXPECTED_LESION_GROUPS = [
    'position_only', 'audio_context', 'audio_stim',
    'visual_context', 'visual_stim', 'velocity',
    'choice_spatial', 'turn_onset', 'outcome',
]


# ── Master log helper ─────────────────────────────────────────────────────
class MasterLog:
    """Writes a human-readable log to disk after every session."""

    def __init__(self, log_path):
        self.path    = log_path
        self.entries = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.path, 'w') as f:
            f.write(f'GLM Model 1 — run started {datetime.now()}\n')
            f.write('=' * 70 + '\n\n')
        print(f'[log] {self.path}')

    def write(self, line, also_print=True):
        ts = datetime.now().strftime('%H:%M:%S')
        msg = f'[{ts}] {line}'
        if also_print:
            print(msg)
        with open(self.path, 'a') as f:
            f.write(msg + '\n')

    def record(self, mouse_ID, date, status, detail=''):
        self.entries.append(dict(mouse_ID=mouse_ID, date=date,
                                 status=status, detail=detail,
                                 ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self._flush_summary()

    def _flush_summary(self):
        n_ok   = sum(1 for e in self.entries if e['status'] == 'success')
        n_skip = sum(1 for e in self.entries if e['status'] == 'skipped')
        n_fail = sum(1 for e in self.entries if e['status'] == 'error')
        with open(self.path, 'a') as f:
            last = self.entries[-1]
            icon = {'success': '✓', 'skipped': '–', 'error': '✗'}.get(last['status'], '?')
            detail = f'  ({last["detail"]})' if last['detail'] else ''
            f.write(f'{icon} {last["mouse_ID"]} / {last["date"]}  '
                    f'[{last["status"]}]{detail}  @ {last["ts"]}\n')
        # Rewrite running summary at end of file
        with open(self.path, 'a') as f:
            f.write(f'\n--- progress: ✓{n_ok}  –{n_skip}  ✗{n_fail} '
                    f'(total processed: {len(self.entries)}) ---\n\n')

    def finalize(self, total):
        n_ok   = sum(1 for e in self.entries if e['status'] == 'success')
        n_skip = sum(1 for e in self.entries if e['status'] == 'skipped')
        n_fail = sum(1 for e in self.entries if e['status'] == 'error')
        summary = (
            f'\n{"=" * 70}\n'
            f'FINISHED  {datetime.now()}\n'
            f'Total sessions: {total}  |  ✓ {n_ok} success  '
            f'|  – {n_skip} skipped  |  ✗ {n_fail} failed\n'
        )
        if n_fail:
            summary += '\nFailed sessions:\n'
            for e in self.entries:
                if e['status'] == 'error':
                    summary += f'  ✗  {e["mouse_ID"]} / {e["date"]}  — {e["detail"]}\n'
        print(summary)
        with open(self.path, 'a') as f:
            f.write(summary)


# ── Session status checker ────────────────────────────────────────────────
def check_session_results(mouse_ID, date):
    res_dir    = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_1_DIR)
    full_done  = os.path.exists(os.path.join(res_dir, 'glm_M1_full.pkl'))
    all_done   = os.path.exists(os.path.join(res_dir, 'glm_M1_lesion_all.pkl'))
    lesion_done = {
        g: os.path.exists(os.path.join(res_dir, f'glm_M1_lesion_{g}.pkl'))
        for g in EXPECTED_LESION_GROUPS
    }
    missing = [g for g, done in lesion_done.items() if not done]
    return {
        'full_model':            full_done,
        'lesion_groups':         lesion_done,
        'lesion_all':            all_done,
        'all_complete':          full_done and all_done and not missing,
        'missing_lesion_groups': missing,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────
def run_all_sessions(sessions_dict, skip_existing=True,
                     neural_metric='deconv', log=None):
    """
    Run the full GLM pipeline for every (mouse_ID, date) in sessions_dict.

    skip_existing behaviour (three cases):
      • full model + all lesion files present  → skip session entirely
      • full model done, some lesion pkls missing → load saved results,
        refit only the missing lesion groups
      • full model missing → run entire pipeline from scratch
    """
    results_log = []
    total = sum(len(v) for v in sessions_dict.values())

    msg = f'Starting pipeline: {len(sessions_dict)} mice | {total} sessions'
    print(msg)
    if log:
        log.write(msg, also_print=False)

    for mouse_ID, dates in sessions_dict.items():
        for date in dates:

            # ── Granular skip / resume logic ───────────────────────────────
            status = check_session_results(mouse_ID, date)

            if skip_existing and status['all_complete']:
                msg = f'[SKIP] {mouse_ID} / {date} — all results exist'
                print(f'  {msg}')
                if log:
                    log.write(msg, also_print=False)
                    log.record(mouse_ID, date, 'skipped')
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'skipped'})
                continue

            if skip_existing and status['full_model']:
                # Full model done, resume missing lesion groups only
                missing = status['missing_lesion_groups']
                msg = (f'[RESUME] {mouse_ID} / {date} — full model done, '
                       f'refitting {len(missing)} lesion group(s): {missing}')
                print(f'\n  {msg}')
                if log:
                    log.write(msg, also_print=False)

                X = Y = metadata = predictor_info = predictor_info_split = all_results = None
                try:
                    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                                             MODEL_1_DIR, 'glm_M1_full.pkl')
                    with open(full_path, 'rb') as f:
                        all_results = pickle.load(f)

                    X, Y, metadata, _, predictor_info, _ = \
                        open_glm_stacked_data(mouse_ID, date, print_summary=False)
                    predictor_info_split, _ = prepare_features_for_glm(predictor_info)

                    run_lesion_models(
                        X, Y, metadata, all_results, predictor_info_split,
                        mouse_ID=mouse_ID, date=date, skip_existing=True,
                    )

                    results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                        'status': 'success'})
                    msg = f'Completed: {mouse_ID} / {date}  [lesion resume]'
                    print(f'\n  ✓ {msg}')
                    if log:
                        log.write(f'✓ {msg}', also_print=False)
                        log.record(mouse_ID, date, 'success', 'lesion-resume')

                except Exception as e:
                    results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                        'status': f'error: {e}'})
                    msg = f'FAILED (lesion resume): {mouse_ID} / {date} — {e}'
                    print(f'\n  ✗ {msg}')
                    traceback.print_exc()
                    if log:
                        log.write(f'✗ {msg}', also_print=False)
                        log.record(mouse_ID, date, 'error', str(e))

                finally:
                    del X, Y, metadata, predictor_info, predictor_info_split, all_results
                    gc.collect()
                continue

            # ── Full pipeline (full model missing) ─────────────────────────
            sep = '=' * 60
            print(f'\n{sep}')
            print(f'  Running: {mouse_ID} / {date}  '
                  f'[{datetime.now().strftime("%H:%M:%S")}]')
            print(f'{sep}')
            if log:
                log.write(f'START {mouse_ID} / {date}', also_print=False)

            glm_data_dict = design_matrices = X = Y = metadata = None
            group_indices = predictor_names = predictor_info = None
            predictor_info_split = feature_group_size = model_cv = None
            all_results = lesion_results = None

            try:
                glm_data_dict = extract_data(mouse_ID, date)

                design_matrices, predictor_names, predictor_info = \
                    generate_predictors(glm_data_dict)

                X, Y, metadata, group_indices = stack_design_matrices(
                    glm_data_dict, design_matrices, predictor_info,
                    neural_metric=neural_metric
                )

                save_glm_data(X, Y, design_matrices, predictor_names,
                              predictor_info, metadata, group_indices,
                              mouse_ID, date)

                # Free raw objects before fitting
                del glm_data_dict, design_matrices, X, Y
                glm_data_dict = design_matrices = X = Y = None
                gc.collect()

                X, Y, metadata, predictor_names, predictor_info, group_indices = \
                    open_glm_stacked_data(mouse_ID, date, print_summary=False)
                predictor_info_split, feature_group_size = \
                    prepare_features_for_glm(predictor_info)

                model_cv, all_results, _, _ = fit_glm_single_session(
                    X, Y, metadata, feature_group_size,
                    predictor_info_split, mouse_ID, date
                )
                # Free all fold weights immediately — all_results has the summary
                del model_cv
                model_cv = None
                gc.collect()

                lesion_results = run_lesion_models(
                    X, Y, metadata, all_results, predictor_info_split,
                    mouse_ID=mouse_ID, date=date, skip_existing=True,
                )

                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'success'})
                msg = f'Completed: {mouse_ID} / {date}'
                print(f'\n  ✓ {msg}  [{datetime.now().strftime("%H:%M:%S")}]')
                if log:
                    log.write(f'✓ {msg}', also_print=False)
                    log.record(mouse_ID, date, 'success')

            except Exception as e:
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': f'error: {e}'})
                msg = f'FAILED: {mouse_ID} / {date} — {e}'
                print(f'\n  ✗ {msg}')
                traceback.print_exc()
                if log:
                    log.write(f'✗ {msg}', also_print=False)
                    log.record(mouse_ID, date, 'error', str(e))

            finally:
                # Always release all large objects before the next session
                del glm_data_dict, design_matrices, X, Y, metadata
                del group_indices, predictor_names, predictor_info
                del predictor_info_split, feature_group_size
                del model_cv, all_results, lesion_results
                gc.collect()

    # ── Final summary ──────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results_log if r['status'] == 'success')
    n_skip = sum(1 for r in results_log if r['status'] == 'skipped')
    n_fail = sum(1 for r in results_log if r['status'].startswith('error'))
    print(f'\n{"=" * 60}')
    print(f'Done — {total} sessions:  '
          f'✓ {n_ok} success  |  – {n_skip} skipped  |  ✗ {n_fail} failed')
    for r in results_log:
        icon = ('✓' if r['status'] == 'success'
                else ('–' if r['status'] == 'skipped' else '✗'))
        print(f'  {icon}  {r["mouse_ID"]} / {r["date"]}  [{r["status"]}]')

    if log:
        log.finalize(total)

    return results_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Model 1 GLM pipeline for all sessions.')
    parser.add_argument('--no-skip', action='store_true',
                        help='Force re-run of all sessions (ignore existing results)')
    parser.add_argument('--metric', type=str, default='deconv',
                        help='Neural metric: deconv | dff | z_dff  (default: deconv)')
    args = parser.parse_args()

    # ── Master log ─────────────────────────────────────────────────────────
    log_dir  = os.path.join(GLM_DIR, 'logs')
    log_path = os.path.join(log_dir,
                            f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log = MasterLog(log_path)

    # ── Print session manifest ─────────────────────────────────────────────
    total = sum(len(v) for v in PPC_AVC.values())
    log.write(f'Sessions to process: {len(PPC_AVC)} mice | {total} total',
              also_print=True)
    for m, dates in PPC_AVC.items():
        log.write(f'  {m}: {dates}', also_print=True)
    log.write(f'skip_existing={not args.no_skip}  metric={args.metric}\n',
              also_print=True)

    run_all_sessions(
        PPC_AVC,
        skip_existing=not args.no_skip,
        neural_metric=args.metric,
        log=log,
    )
