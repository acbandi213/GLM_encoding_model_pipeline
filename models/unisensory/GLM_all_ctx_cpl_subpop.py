#!/usr/bin/env python
"""
GLM_all_ctx_cpl_subpop.py
==========================
Train and test the GLM on all three contexts (CG + visual + audio) with
functional coupling predictors built separately from CI and CD neuron
subpopulations.

Architecture (Option B — combined model):
    Design matrix: [behavioral predictors | CI coupling | CD coupling]
    Coupling groups: ci_pc1 … ci_pc5  and  cd_pc1 … cd_pc5

CI/CD labels are read from the base unisensory model (GLM_uni_results) using
the same Context Preference Index (CPI) thresholds as the paper figures
notebook:
    CI : |vis_aud_ci| < 0.1   (context-invariant)
    CD : |vis_aud_ci| >= 0.8  (strongly context-dependent)

In addition to per-PC lesion models, two COMPOUND lesion models are fitted:
    ci_network : remove all CI coupling columns simultaneously
    cd_network : remove all CD coupling columns simultaneously

These compound lesions reveal which subpopulation uniquely drives each target
neuron, enabling the "CI backbone" structural claim test:
    - CI backbone → CI target neurons show higher ΔFDE from ci_network lesion
                    than from cd_network lesion
    - Non-specific  → both ΔFDE distributions are symmetric around zero

    ⚠️⚠️ SELF-PREDICTION LEAKAGE — READ BEFORE TRUSTING THE WITHIN-CLASS RESULT.
    This shared-design build puts every CI neuron in the CI PCA and every CD
    neuron in the CD PCA. So a CI target's `ci_network` ΔFDE (and a CD target's
    `cd_network` ΔFDE) is inflated because the neuron is ~1/n_class of its own
    class's coupling PCA — it partly predicts ITSELF. The within-class DIAGONAL
    of the backbone/periphery test is therefore NOT leakage-free; the cross terms
    (CI→CD, CD→CI) are. On a 128-unit RNN this exact leakage fabricated a 5×
    "double dissociation" that COLLAPSED (CI arm 5.4×→0.82×, CD arm 4.9×→1.36×)
    once each target was excluded from its own class PCA. The real Fig-4 backbone
    claim MUST be recomputed leave-one-out before it stands. Use
    `core.session_pipeline.subpop_coupling_gain_loo` (per-target, target excluded
    from its own class PCA) for the within-class gains; keep this file for the
    shared-design fit and the leakage-free cross terms only.

Results are saved to:
    {BASE_DATA_PATH}/{mouse_ID}/{date}/GLM_all_ctx_cpl_subpop_results/
        glm_all_ctx_cpl_sp_full.pkl
        glm_all_ctx_cpl_sp_lesion_{group}.pkl  × 19  (9 base + 5 ci_pc + 5 cd_pc)
        glm_all_ctx_cpl_sp_lesion_ci_network.pkl      (compound)
        glm_all_ctx_cpl_sp_lesion_cd_network.pkl      (compound)
        glm_all_ctx_cpl_sp_lesion_all.pkl             (all 21 groups merged)

Usage
-----
python GLM_all_ctx_cpl_subpop.py
python GLM_all_ctx_cpl_subpop.py --no-skip
python GLM_all_ctx_cpl_subpop.py --label-model GLM_uni_results GLM_uni
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
    get_ci_cd_masks,
    add_subpop_coupling_predictors,
    prepare_features_for_glm,
    fit_glm_single_session,
    run_lesion_models,
    compute_best_trial_avg_r2,
)
import core.glm_class as glm_class
from core.glm_class import make_prediction, deviance
from config.session_registry import PPC_AVC, BASE_LESION_GROUPS

# ==============================================================================
# Model identity
# ==============================================================================
MODEL_NAME   = 'GLM_all_ctx_cpl_subpop'
MODEL_TAG    = 'all_ctx_cpl_subpop'
MODEL_DIR    = 'GLM_all_ctx_cpl_subpop_results'
MODEL_PREFIX = 'glm_all_ctx_cpl_sp'
CONTEXTS     = [0, 1, 2]

# Coupling configuration
N_PCS       = 5
N_LAG_BASES = 4
LAG_FRAMES  = 8
COUPLING_PARAMS = dict(n_pcs=N_PCS, n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES)

# CI/CD labeling — uses the base unisensory model, same as paper figures notebook
LABEL_MODEL_DIR    = 'GLM_uni_results'
LABEL_MODEL_PREFIX = 'GLM_uni'
INVARIANT_THRESH   = 0.1   # |vis_aud_ci| < this  → CI
CD_THRESH          = 0.8   # |vis_aud_ci| >= this  → CD
RESP_THRESHOLD     = 0.1   # neuron must clear this FDE to be labeled

# Derived group lists
CI_PC_GROUPS    = [f'ci_pc{i+1}' for i in range(N_PCS)]
CD_PC_GROUPS    = [f'cd_pc{i+1}' for i in range(N_PCS)]
COMPOUND_GROUPS = ['ci_network', 'cd_network']

EXPECTED_LESION_GROUPS = BASE_LESION_GROUPS + CI_PC_GROUPS + CD_PC_GROUPS + COMPOUND_GROUPS
# = 9 + 5 + 5 + 2 = 21 total lesion files
# ==============================================================================


def _filter_contexts(X, Y, metadata):
    """Restrict X, Y, metadata to frames whose context is in CONTEXTS."""
    mask = np.isin(metadata['context'], CONTEXTS)
    return X[mask], Y[mask], {k: v[mask] for k, v in metadata.items()}


def check_session_results(mouse_ID, date):
    """Return completion status dict for this session."""
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
            f.write(f'Contexts: {CONTEXTS}\n')
            f.write(f'Coupling params: {COUPLING_PARAMS}\n')
            f.write(f'Label model: {LABEL_MODEL_DIR}/{LABEL_MODEL_PREFIX}\n')
            f.write(f'CI threshold: |CPI| < {INVARIANT_THRESH}  '
                    f'CD threshold: |CPI| >= {CD_THRESH}\n')
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


# ── Compound lesion helper ─────────────────────────────────────────────────────

def _run_compound_lesions(X, Y, metadata, all_results, pred_info_split,
                           ci_col_range, cd_col_range,
                           mouse_ID, date, skip_existing=True):
    """
    Fit and evaluate two compound lesion models:
        'ci_network' — remove ALL CI coupling columns simultaneously
        'cd_network' — remove ALL CD coupling columns simultaneously

    Unlike single-PC lesions (handled by run_lesion_models), these remove an
    entire subpopulation's coupling block at once, so feature_group_size must
    be rebuilt by excluding all individual PC entries for that subpopulation.

    Each compound result is saved as its own pkl and then merged into the
    existing _lesion_all.pkl so the analysis notebook sees all 21 groups in
    one file.
    """
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR)
    os.makedirs(save_dir, exist_ok=True)

    combined_train_mask = all_results['combined_train_mask']
    combined_test_mask  = all_results['combined_test_mask']
    full_frac_dev       = all_results['test_frac_dev']
    full_r2             = all_results.get('test_trial_avg_r2',
                                          np.zeros_like(full_frac_dev))

    X_train_full    = X[combined_train_mask]
    Y_train         = Y[combined_train_mask]
    X_test          = X[combined_test_mask]
    Y_test          = Y[combined_test_mask]
    meta_test       = {k: v[combined_test_mask] for k, v in metadata.items()}
    group_idx_train = metadata['trial_id'][combined_train_mask]

    group_names        = list(pred_info_split.keys())
    feature_group_size = all_results['feature_group_size']

    compound_defs = {
        'ci_network': {
            'col_range':      ci_col_range,
            'exclude_prefix': 'ci_pc',
        },
        'cd_network': {
            'col_range':      cd_col_range,
            'exclude_prefix': 'cd_pc',
        },
    }

    compound_delta_fd = {}
    compound_delta_r2 = {}
    compound_lesion_fd = {}
    compound_lesion_r2 = {}

    for compound_name, cdef in compound_defs.items():
        grp_path = os.path.join(save_dir,
                                f'{MODEL_PREFIX}_lesion_{compound_name}.pkl')

        if skip_existing and os.path.exists(grp_path):
            with open(grp_path, 'rb') as f:
                cached = pickle.load(f)
            compound_lesion_fd[compound_name] = cached['lesion_test_frac_dev']
            compound_lesion_r2[compound_name] = cached['lesion_test_trial_avg_r2']
            compound_delta_fd[compound_name]  = cached['delta_frac_dev']
            compound_delta_r2[compound_name]  = cached['delta_r2']
            print(f'  [Compound] "{compound_name}" — [LOADED from cache]')
            continue

        col_start, col_end    = cdef['col_range']
        exclude_prefix        = cdef['exclude_prefix']

        print(f'  [Compound] Lesion "{compound_name}" '
              f'(cols {col_start}–{col_end-1}) … ', end='', flush=True)

        # Remove the compound block from X
        col_keep              = np.ones(X.shape[1], dtype=bool)
        col_keep[col_start:col_end] = False
        X_lesion_train        = X_train_full[:, col_keep]
        X_lesion_test         = X_test[:, col_keep]

        # Rebuild feature_group_size: exclude all individual PC groups for this
        # subpopulation (identified by their group name prefix).
        exclude_indices = [
            i for i, name in enumerate(group_names)
            if name.startswith(exclude_prefix)
        ]
        feature_group_size_lesion = np.delete(feature_group_size, exclude_indices)

        # Refit with same hyperparameters as fit_glm_single_session
        model_lesion = glm_class.GLM_CV(
            n_folds=3,
            auto_split=True,
            split_by_group=True,
            skip_final_fold=True,
            activation='exp',
            loss_type='poisson',
            regularization='group_lasso',
            lambda_series=10.0 ** np.linspace(-1, -5, 5),
            l1_ratio=0.95,
            smooth_strength=0.1,
            optimizer='adam',
            learning_rate=1e-3,
            max_iter_per_lambda=5000,
            convergence_tol=1e-5,
        )
        model_lesion.fit(
            X_lesion_train, Y_train,
            feature_group_size=feature_group_size_lesion,
            group_idx=group_idx_train,
        )
        model_lesion.select_model(se_fraction=0., make_fig=False)

        Y_pred_lesion = make_prediction(
            X_lesion_test,
            model_lesion.selected_w,
            model_lesion.selected_w0.flatten(),
            activation='exp',
        )
        fd_lesion, _, _ = deviance(Y_pred_lesion, Y_test)
        fd_lesion       = np.maximum(fd_lesion, 0.0)

        r2_lesion_best, r2_lesion_splits = compute_best_trial_avg_r2(
            Y_test, Y_pred_lesion, meta_test
        )

        delta_fd = full_frac_dev - fd_lesion
        delta_r2 = full_r2 - r2_lesion_best

        compound_lesion_fd[compound_name] = fd_lesion
        compound_lesion_r2[compound_name] = r2_lesion_best
        compound_delta_fd[compound_name]  = delta_fd
        compound_delta_r2[compound_name]  = delta_r2

        med_fd  = np.median(fd_lesion)
        med_dfd = np.median(delta_fd)
        med_dr2 = np.median(delta_r2)
        print(f'✓  lesion frac_dev={med_fd:.3f}  '
              f'Δfrac_dev={med_dfd:+.3f}  Δr²={med_dr2:+.3f}')

        # Save individual compound file
        grp_results = {
            'group':                   compound_name,
            'full_test_frac_dev':      full_frac_dev,
            'full_test_trial_avg_r2':  full_r2,
            'lesion_test_frac_dev':    fd_lesion,
            'lesion_test_trial_avg_r2': r2_lesion_best,
            'lesion_r2_by_split':      r2_lesion_splits,
            'delta_frac_dev':          delta_fd,
            'delta_r2':                delta_r2,
            'ci_col_range':            ci_col_range,
            'cd_col_range':            cd_col_range,
            'mouse_ID':                mouse_ID,
            'date':                    date,
        }
        with open(grp_path, 'wb') as f:
            pickle.dump(grp_results, f)

        del model_lesion, X_lesion_train, X_lesion_test
        del Y_pred_lesion, fd_lesion, r2_lesion_best, r2_lesion_splits
        del col_keep, feature_group_size_lesion
        gc.collect()

    # ── Merge compound results into _lesion_all.pkl ────────────────────────────
    all_path = os.path.join(save_dir, f'{MODEL_PREFIX}_lesion_all.pkl')
    if os.path.exists(all_path):
        with open(all_path, 'rb') as f:
            lesion_results = pickle.load(f)

        for compound_name in compound_defs:
            if compound_name not in lesion_results['groups']:
                lesion_results['groups'].append(compound_name)
            lesion_results['lesion_test_frac_dev'][compound_name] = (
                compound_lesion_fd[compound_name])
            lesion_results['lesion_test_trial_avg_r2'][compound_name] = (
                compound_lesion_r2[compound_name])
            lesion_results['delta_frac_dev'][compound_name] = (
                compound_delta_fd[compound_name])
            lesion_results['delta_r2'][compound_name] = (
                compound_delta_r2[compound_name])

        with open(all_path, 'wb') as f:
            pickle.dump(lesion_results, f)

        mb = os.path.getsize(all_path) / 1e6
        print(f'  [Compound] Merged into _lesion_all.pkl  ({mb:.2f} MB, '
              f'{len(lesion_results["groups"])} groups total)')
    else:
        print(f'  [Compound] WARNING: _lesion_all.pkl not found at {all_path}; '
              f'compound results saved only as individual files.')

    return compound_delta_fd, compound_delta_r2


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_all_sessions(sessions_dict, skip_existing=True,
                     neural_metric='deconv', log=None,
                     label_model_dir=LABEL_MODEL_DIR,
                     label_model_prefix=LABEL_MODEL_PREFIX):
    """Run the full GLM pipeline for every (mouse_ID, date) in sessions_dict."""
    results_log = []
    total       = sum(len(v) for v in sessions_dict.values())
    print(f'[{MODEL_NAME}] Starting: {len(sessions_dict)} mice | {total} sessions\n'
          f'  contexts={CONTEXTS}  |  n_pcs={N_PCS}  '
          f'|  CI thresh={INVARIANT_THRESH}  |  CD thresh={CD_THRESH}\n'
          f'  Label model: {label_model_dir}/{label_model_prefix}')

    for mouse_ID, dates in sessions_dict.items():
        for date in dates:

            status = check_session_results(mouse_ID, date)

            # ── All results exist → skip ───────────────────────────────────
            if skip_existing and status['all_complete']:
                print(f'  [SKIP] {mouse_ID} / {date} — all results exist')
                results_log.append({'mouse_ID': mouse_ID, 'date': date,
                                    'status': 'skipped'})
                if log:
                    log.record(mouse_ID, date, 'skipped')
                continue

            # ── Full model exists but some lesions missing → resume ────────
            if skip_existing and status['full_model']:
                missing = status['missing_lesion_groups']
                print(f'\n  [RESUME] {mouse_ID} / {date} — '
                      f'missing lesions: {missing}')

                X = Y = metadata = pred_info = pred_info_split = None
                all_results = ci_col_range = cd_col_range = None
                try:
                    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                                             MODEL_DIR,
                                             f'{MODEL_PREFIX}_full.pkl')
                    with open(full_path, 'rb') as f:
                        all_results = pickle.load(f)

                    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(
                        mouse_ID, date, print_summary=False)
                    X, Y, metadata = _filter_contexts(X, Y, metadata)

                    ci_mask, cd_mask, _, label_stats = get_ci_cd_masks(
                        mouse_ID, date,
                        source_model_dir=label_model_dir,
                        source_prefix=label_model_prefix,
                        resp_threshold=RESP_THRESHOLD,
                        invariant_thresh=INVARIANT_THRESH,
                        cd_thresh=CD_THRESH,
                    )

                    X, pred_info, _, _, ci_col_range, cd_col_range = \
                        add_subpop_coupling_predictors(
                            X, Y, pred_info, ci_mask, cd_mask, **COUPLING_PARAMS)

                    pred_info_split, _ = prepare_features_for_glm(pred_info)

                    # Run missing individual-PC lesions
                    individual_missing = [g for g in missing
                                          if g not in COMPOUND_GROUPS]
                    if individual_missing:
                        run_lesion_models(
                            X, Y, metadata, all_results, pred_info_split,
                            mouse_ID=mouse_ID, date=date, skip_existing=True,
                            model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                        )

                    # Run missing compound lesions
                    compound_missing = [g for g in missing
                                        if g in COMPOUND_GROUPS]
                    if compound_missing:
                        _run_compound_lesions(
                            X, Y, metadata, all_results, pred_info_split,
                            ci_col_range, cd_col_range,
                            mouse_ID=mouse_ID, date=date, skip_existing=True,
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
                    del ci_col_range, cd_col_range
                    gc.collect()
                    if _HAS_TF:
                        tf.keras.backend.clear_session()
                continue

            # ── Full run ───────────────────────────────────────────────────
            sep = '=' * 60
            print(f'\n{sep}\n  [{MODEL_NAME}] Running: {mouse_ID} / {date}  '
                  f'[{datetime.now().strftime("%H:%M:%S")}]\n{sep}')
            if log:
                log.write(f'START {mouse_ID} / {date}', also_print=False)

            X = Y = metadata = pred_info = pred_info_split = None
            feature_group_size = model_cv = all_results = lesion_results = None
            ci_col_range = cd_col_range = None

            try:
                stacked_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                                            'glm_data_stacked.pkl')
                if not os.path.exists(stacked_path):
                    raise FileNotFoundError(
                        f'glm_data_stacked.pkl not found for {mouse_ID}/{date}. '
                        f'Run run_sessions.py first to build the stacked data.')

                # ── 1. Load data ───────────────────────────────────────────
                X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(
                    mouse_ID, date, print_summary=False)
                X, Y, metadata = _filter_contexts(X, Y, metadata)

                # ── 2. Get CI/CD neuron masks ──────────────────────────────
                ci_mask, cd_mask, vis_aud_ci, label_stats = get_ci_cd_masks(
                    mouse_ID, date,
                    source_model_dir=label_model_dir,
                    source_prefix=label_model_prefix,
                    resp_threshold=RESP_THRESHOLD,
                    invariant_thresh=INVARIANT_THRESH,
                    cd_thresh=CD_THRESH,
                )
                if log:
                    log.write(
                        f'  Labels: n_CI={label_stats["n_ci"]}  '
                        f'n_CD={label_stats["n_cd"]}  '
                        f'(n_resp={label_stats["n_responsive"]})',
                        also_print=False,
                    )

                # ── 3. Build subpop coupling predictors ────────────────────
                X, pred_info, ci_pca, cd_pca, ci_col_range, cd_col_range = \
                    add_subpop_coupling_predictors(
                        X, Y, pred_info, ci_mask, cd_mask, **COUPLING_PARAMS)
                del ci_pca, cd_pca

                # ── 4. Prepare features for GLM ────────────────────────────
                pred_info_split, feature_group_size = \
                    prepare_features_for_glm(pred_info)

                # ── 5. Fit full model ──────────────────────────────────────
                model_cv, all_results, _, _ = fit_glm_single_session(
                    X, Y, metadata, feature_group_size, pred_info_split,
                    mouse_ID, date,
                    model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                    contexts=CONTEXTS,
                )
                del model_cv
                model_cv = None
                gc.collect()

                # ── 6. Individual-PC lesion models ─────────────────────────
                lesion_results = run_lesion_models(
                    X, Y, metadata, all_results, pred_info_split,
                    mouse_ID=mouse_ID, date=date, skip_existing=True,
                    model_dir=MODEL_DIR, model_prefix=MODEL_PREFIX,
                )
                del lesion_results
                lesion_results = None
                gc.collect()

                # ── 7. Compound lesion models (ci_network, cd_network) ─────
                _run_compound_lesions(
                    X, Y, metadata, all_results, pred_info_split,
                    ci_col_range, cd_col_range,
                    mouse_ID=mouse_ID, date=date, skip_existing=True,
                )

                del all_results
                all_results = None
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
                del ci_col_range, cd_col_range
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
    parser.add_argument('--label-model', type=str, nargs=2,
                        default=[LABEL_MODEL_DIR, LABEL_MODEL_PREFIX],
                        metavar=('DIR', 'PREFIX'),
                        help=(f'Model directory and prefix to use for CI/CD '
                              f'labeling (default: {LABEL_MODEL_DIR} {LABEL_MODEL_PREFIX})'))
    args = parser.parse_args()

    label_dir, label_prefix = args.label_model

    log_path = os.path.join(
        GLM_DIR, 'logs',
        f'run_{MODEL_TAG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log = MasterLog(log_path)

    total = sum(len(v) for v in PPC_AVC.values())
    log.write(f'{MODEL_NAME} | contexts={CONTEXTS} | n_pcs={N_PCS} | '
              f'CI_thresh={INVARIANT_THRESH} | CD_thresh={CD_THRESH} | '
              f'label_model={label_dir}/{label_prefix} | '
              f'{len(PPC_AVC)} mice | {total} sessions')
    for m, dates in PPC_AVC.items():
        log.write(f'  {m}: {dates}')

    run_all_sessions(
        PPC_AVC,
        skip_existing=not args.no_skip,
        neural_metric=args.metric,
        log=log,
        label_model_dir=label_dir,
        label_model_prefix=label_prefix,
    )
