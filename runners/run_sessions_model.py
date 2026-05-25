#!/usr/bin/env python
"""
run_sessions_model.py
=====================
Generic GLM pipeline runner — same architecture as Model 1 but with
configurable basis function counts, extended predictor groups, and
optional functional coupling.

To create a new model variant, edit ONLY the "Model identity" and
"Builder params" blocks at the top of this file.  Every filename,
directory, log message, and pkl prefix is derived automatically from
MODEL_NAME / MODEL_TAG.

Differences from Model 1 (run_sessions.py)
-------------------------------------------
This file is currently configured as "Model 2 (Coupling)" — identical
group structure to Model 1, with the addition of functional coupling
predictors from the simultaneously recorded population.

  1. Functional coupling predictors  (COUPLING_PARAMS['include_coupling'])
       NEW groups: network_pc1 … network_pc{n_pcs}
       Top PCs of the simultaneously recorded population, convolved
       with raised-cosine lag bases.  Added post-stacking via
       add_coupling_predictor() in session_pipeline.py.

Group structure (lesion targets) — same as Model 1:
    position_only, audio_context, audio_stim,
    visual_context, visual_stim, velocity,
    choice_spatial, turn_onset, outcome,
    network_pc1 … network_pc{n_pcs}  (if coupling enabled)

Usage
-----
python run_sessions_model.py
python run_sessions_model.py --no-skip
python run_sessions_model.py --metric dff
"""

import argparse
import gc
import os
import pickle
import sys
import traceback
from datetime import datetime

import tensorflow as tf

# ── Path setup ────────────────────────────────────────────────────────────────
GLM_DIR        = os.path.dirname(os.path.abspath(__file__))
ICLOUD_GLM_DIR = ('/Users/akhilbandi/Library/Mobile Documents/'
                  'com~apple~CloudDocs/Documents/Portfolio/code_projects/'
                  'Context Paper Code/GLM_encoding_model')
for _p in (GLM_DIR, ICLOUD_GLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Pipeline imports ──────────────────────────────────────────────────────────
from core.session_pipeline import (
    BASE_DATA_PATH,
    extract_data,
    stack_design_matrices,
    add_coupling_predictor,
    prepare_features_for_glm,
    fit_glm_single_session,
    run_lesion_models,
)
from core.predictor_builder import GLMPredictorBuilder

# ==============================================================================
# Model identity — CHANGE ONLY THIS BLOCK to create a new model variant.
#
#   MODEL_NAME  : human-readable label used in directory names and log headers
#   MODEL_TAG   : short lowercase token used in pkl filenames
#                 → results saved as  glm_{MODEL_TAG}_full.pkl
#                                     glm_{MODEL_TAG}_lesion_{group}.pkl
#                                     glm_{MODEL_TAG}_lesion_all.pkl
# ==============================================================================
MODEL_NAME        = 'coupling'                                # e.g. 'Model_3', 'Coupling', 'HighRes'
MODEL_TAG         = 'coupling'                                # e.g. 'M3',      'coupling',  'highres'
MODEL_DIR         = f'{MODEL_NAME}_results'                   # subdirectory under mouse/date/
STACKED_DATA_FILE = f'glm_data_stacked_{MODEL_TAG}.pkl'       # stacked design matrix file
MODEL_PREFIX      = f'glm_{MODEL_TAG}'                        # pkl filename prefix

# ── Builder params — identical to Model 1 except coupling is added afterwards
BUILDER_PARAMS = dict(
    sampling_rate             = 15.6,
    stem_y_max                = 500.0,
    pos_only_y_max            = 350.0,
    audio_start_position      = 50.0,
    turn_y_thresh             = 505.0,
    turn_x_vel_thresh         = 20.0,
    visual_y_thresh           = 500.0,
    pos_bump_width_ratio      = 4.0,
    temporal_bump_width_ratio = 2.0,
    outcome_offset_from_iti   = 0,

    n_pos_bases    = 10,
    n_place_fields = 30,
    n_vel_bases    = 5,
    n_turn_bases   = 4,
    n_reward_bases = 6,
    turn_window    = 1.0,
    reward_window  = 3.0,

    include_position_only   = True,
    include_sound_position  = True,
    include_visual_position = True,
    include_velocity        = True,
    include_spatial_choice  = True,
    include_turn_onset      = True,
    include_reward          = True,

    # All extended / combined group flags off — same structure as Model 1
    include_combined_stimulus = False,
    include_combined_choice   = False,
    include_stim_temporal     = False,
    include_pre_turn          = False,
    include_context_choice    = False,
)

# ── Functional coupling configuration ────────────────────────────────────────
# Set include_coupling=False to run without coupling predictors.
COUPLING_PARAMS = dict(
    include_coupling = True,
    n_pcs            = 5,    # number of population PCs to use
    n_lag_bases      = 4,    # raised-cosine bases per PC
    lag_frames       = 8,    # temporal window (~500 ms at 15.6 Hz)
)

# ── Session registry ──────────────────────────────────────────────────────────
PPC_AVC = {
    'KO-5-1L': ['2026-03-24'],
    'KO-6-00': ['2026-03-23', '2026-03-25'],
}

_BASE_LESION_GROUPS = [
    # Same 9 groups as Model 1
    'position_only',
    'audio_context',
    'audio_stim',
    'visual_context',
    'visual_stim',
    'velocity',
    'choice_spatial',
    'turn_onset',
    'outcome',
]
# Append coupling lesion groups when coupling is enabled
_COUPLING_LESION_GROUPS = (
    [f'network_pc{i+1}' for i in range(COUPLING_PARAMS['n_pcs'])]
    if COUPLING_PARAMS['include_coupling'] else []
)
EXPECTED_LESION_GROUPS = _BASE_LESION_GROUPS + _COUPLING_LESION_GROUPS


# ── Predictor builder ─────────────────────────────────────────────────────────
def generate_predictors_model(glm_data_dict, plot_predictors=False):
    """Build design matrices using BUILDER_PARAMS."""
    builder = GLMPredictorBuilder(**BUILDER_PARAMS)
    design_matrices, predictor_names, predictor_info = \
        builder.build_design_matrices(glm_data_dict)

    print(f'[{MODEL_NAME}] Trials: {len(design_matrices)} | Predictors: {len(predictor_names)}')
    print(f'[{MODEL_NAME}] Predictor groups:')
    for group, info in predictor_info.items():
        n_pred = info['end'] - info['start']
        print(f'  {group:20s}: {n_pred:3d}  (cols {info["start"]}–{info["end"]-1})')

    if plot_predictors:
        builder.plot_predictors(glm_data_dict, design_matrices, predictor_info,
                                start_trial=30, n_trials=3)
    return design_matrices, predictor_names, predictor_info


def save_glm_data_model(X, Y, design_matrices, predictor_names,
                        predictor_info, metadata, group_indices,
                        mouse_ID, date):
    """Save stacked data for this model to disk.

    design_matrices are intentionally excluded from the saved pkl — they are
    only needed to build X/Y and would waste significant disk space and RAM
    on reload (each per-trial array is ~100s of MB for large sessions).
    """
    save_dir  = os.path.join(BASE_DATA_PATH, mouse_ID, date)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, STACKED_DATA_FILE)

    glm_data = {
        'X': X, 'Y': Y, 'metadata': metadata,
        'predictor_names': predictor_names,
        'predictor_info': predictor_info,
        'group_indices': group_indices,
        'builder_params': BUILDER_PARAMS,
        'model': MODEL_NAME,
        'mouse_ID': mouse_ID,
        'date': date,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(glm_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(save_path) / 1024**2
    print(f'  [{MODEL_NAME}] Saved stacked data → {save_path}  ({size_mb:.1f} MB)')


def open_glm_stacked_data_model(mouse_ID, date, print_summary=False):
    """Load stacked data for this model, applying Y scaling."""
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    from core.session_pipeline import Y_SCALE, Y_SMOOTH_SIGMA

    load_path = os.path.join(BASE_DATA_PATH, mouse_ID, date, STACKED_DATA_FILE)
    with open(load_path, 'rb') as f:
        glm_data = pickle.load(f)

    X               = glm_data['X']
    Y               = glm_data['Y'] * Y_SCALE
    if Y_SMOOTH_SIGMA > 0:
        Y = gaussian_filter1d(Y, sigma=Y_SMOOTH_SIGMA, axis=0)
        Y = np.clip(Y, 0.0, None)
    metadata        = glm_data['metadata']
    predictor_names = glm_data['predictor_names']
    predictor_info  = glm_data['predictor_info'].copy()
    group_indices   = glm_data['group_indices']
    print(f'  [{MODEL_NAME}] Loaded  X:{X.shape}  Y:{Y.shape}')
    return X, Y, metadata, predictor_names, predictor_info, group_indices


def check_session_results(mouse_ID, date):
    """Check which result files exist for this model."""
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
            f.write(f'Builder params: {BUILDER_PARAMS}\n')
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
        s = (f'\n{"="*70}\nFINISHED  {datetime.now()}\n'
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
    results_log = []
    total = sum(len(v) for v in sessions_dict.values())
    print(f'[{MODEL_NAME}] Starting: {len(sessions_dict)} mice | {total} sessions')

    for mouse_ID, dates in sessions_dict.items():
        for date in dates:

            status = check_session_results(mouse_ID, date)

            if skip_existing and status['all_complete']:
                print(f'  [SKIP] {mouse_ID} / {date}')
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'skipped'})
                if log:
                    log.record(mouse_ID, date, 'skipped')
                continue

            # ── Lesion-resume (full model done, some lesions missing) ─────────
            if skip_existing and status['full_model']:
                missing = status['missing_lesion_groups']
                print(f'\n  [RESUME] {mouse_ID} / {date} — missing lesions: {missing}')
                X = Y = metadata = predictor_info = predictor_info_split = all_results = None
                try:
                    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                                             MODEL_DIR, f'{MODEL_PREFIX}_full.pkl')
                    with open(full_path, 'rb') as f:
                        all_results = pickle.load(f)

                    X, Y, metadata, _, predictor_info, _ = \
                        open_glm_stacked_data_model(mouse_ID, date)

                    if COUPLING_PARAMS['include_coupling']:
                        X, predictor_info, _ = add_coupling_predictor(
                            X, Y, predictor_info,
                            n_pcs=COUPLING_PARAMS['n_pcs'],
                            n_lag_bases=COUPLING_PARAMS['n_lag_bases'],
                            lag_frames=COUPLING_PARAMS['lag_frames'],
                        )

                    predictor_info_split, _ = prepare_features_for_glm(predictor_info)
                    run_lesion_models(
                        X, Y, metadata, all_results, predictor_info_split,
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
                    del X, Y, metadata, predictor_info, predictor_info_split, all_results
                    gc.collect()
                    tf.keras.backend.clear_session()
                continue

            # ── Full pipeline ─────────────────────────────────────────────────
            sep = '=' * 60
            print(f'\n{sep}\n  [{MODEL_NAME}] Running: {mouse_ID} / {date}  '
                  f'[{datetime.now().strftime("%H:%M:%S")}]\n{sep}')
            if log:
                log.write(f'START {mouse_ID} / {date}', also_print=False)

            glm_data_dict = design_matrices = X = Y = metadata = None
            group_indices = predictor_names = predictor_info = None
            predictor_info_split = feature_group_size = model_cv = None
            all_results = lesion_results = None

            try:
                glm_data_dict = extract_data(mouse_ID, date)

                design_matrices, predictor_names, predictor_info = \
                    generate_predictors_model(glm_data_dict)

                X, Y, metadata, group_indices = stack_design_matrices(
                    glm_data_dict, design_matrices, predictor_info,
                    neural_metric=neural_metric
                )

                save_glm_data_model(X, Y, design_matrices, predictor_names,
                                    predictor_info, metadata, group_indices,
                                    mouse_ID, date)

                del glm_data_dict, design_matrices, X, Y
                glm_data_dict = design_matrices = X = Y = None
                gc.collect()

                X, Y, metadata, predictor_names, predictor_info, group_indices = \
                    open_glm_stacked_data_model(mouse_ID, date)

                if COUPLING_PARAMS['include_coupling']:
                    X, predictor_info, _ = add_coupling_predictor(
                        X, Y, predictor_info,
                        n_pcs=COUPLING_PARAMS['n_pcs'],
                        n_lag_bases=COUPLING_PARAMS['n_lag_bases'],
                        lag_frames=COUPLING_PARAMS['lag_frames'],
                    )

                predictor_info_split, feature_group_size = \
                    prepare_features_for_glm(predictor_info)

                model_cv, all_results, _, _ = fit_glm_single_session(
                    X, Y, metadata, feature_group_size, predictor_info_split,
                    mouse_ID, date,
                    model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                )
                del model_cv
                model_cv = None
                gc.collect()

                lesion_results = run_lesion_models(
                    X, Y, metadata, all_results, predictor_info_split,
                    mouse_ID=mouse_ID, date=date, skip_existing=True,
                    model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                )

                # Free the two largest objects immediately — both are already
                # saved to disk; no need to hold them in RAM until finally.
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
                del glm_data_dict, design_matrices, X, Y, metadata
                del group_indices, predictor_names, predictor_info
                del predictor_info_split, feature_group_size
                del model_cv, all_results, lesion_results
                gc.collect()
                tf.keras.backend.clear_session()

    if log:
        log.finalize(total)
    return results_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f'Run {MODEL_NAME} GLM pipeline.')
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--metric', type=str, default='deconv')
    args = parser.parse_args()

    log_path = os.path.join(
        GLM_DIR, 'logs',
        f'run_{MODEL_TAG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    log = MasterLog(log_path)

    total = sum(len(v) for v in PPC_AVC.values())
    log.write(f'{MODEL_NAME} | {len(PPC_AVC)} mice | {total} sessions')
    for m, dates in PPC_AVC.items():
        log.write(f'  {m}: {dates}')

    run_all_sessions(
        PPC_AVC,
        skip_existing=not args.no_skip,
        neural_metric=args.metric,
        log=log,
    )
