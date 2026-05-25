"""
session_pipeline.py
===================
Per-session pipeline functions for GLM fitting.

Typical call order
------------------
1.  glm_data_dict                             = extract_data(mouse_ID, date)
2.  design_matrices, pred_names, pred_info    = generate_predictors(glm_data_dict)
3.  X, Y, metadata, group_indices             = stack_design_matrices(glm_data_dict, design_matrices, pred_info)
4.                                              save_glm_data(X, Y, ...)
5.  X, Y, metadata, pred_names, pred_info, _  = open_glm_stacked_data(mouse_ID, date)
6.  pred_info_split, feature_group_size       = prepare_features_for_glm(pred_info)
7.  model_cv, results, _, _                   = fit_glm_single_session(X, Y, metadata, ...)
    → saves  {BASE_DATA_PATH}/{mouse_ID}/{date}/Model_1_results/glm_M1_full.pkl
8.  lesion_results                            = run_lesion_models(X, Y, metadata, results, ...)
    → saves  {BASE_DATA_PATH}/{mouse_ID}/{date}/Model_1_results/glm_M1_lesion_{group}.pkl  × n_groups
             {BASE_DATA_PATH}/{mouse_ID}/{date}/Model_1_results/glm_M1_lesion_all.pkl
"""

import gc
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split

# Ensure the GLM_encoding_model directory is on the path regardless of
# where the calling notebook's working directory is.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import glm_class
from glm_class import make_prediction, deviance
from predictor_builder import GLMPredictorBuilder
from scipy.ndimage import gaussian_filter1d

BASE_DATA_PATH = '/Volumes/Akhil Data/Akhil/ProcessedData'

# Y scaling applied when loading stacked data.
# Raw deconvolved values are tiny floats (~0.001–0.01 events/frame).
# Scaling to integer-like counts improves Poisson GLM numerical stability.
# Change this value to re-run with a different scale (requires re-fitting).
Y_SCALE = 10   # × raw deconv  →  ~0.1–1.0 events/frame

# Gaussian smoothing applied to Y after scaling, before model fitting.
# Deconvolved calcium traces are sparse (point-process-like); smoothing
# spreads single-frame events into ~2-frame bumps that basis functions
# can represent.  Set to 0 to disable.  Requires re-fitting if changed.
# After smoothing, Y is clipped to ≥ 0 to preserve Poisson validity.
Y_SMOOTH_SIGMA = 5  # frames  (~67 ms at 30 Hz)

# Subfolder within each session directory where Model 1 results are stored.
MODEL_1_DIR = 'Model_1_results'


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def extract_data(mouse_ID, date):
    """Load glm_data_dict (per-trial dict) from compressed pickle."""
    save_path = f'{BASE_DATA_PATH}/{mouse_ID}/{date}/glm_data_dict.pkl.gz'
    with gzip.open(save_path, 'rb') as f:
        glm_data_dict = pickle.load(f)
    return glm_data_dict


def save_glm_data(X, Y, design_matrices, predictor_names, predictor_info,
                  metadata, group_indices, mouse_ID, date):
    """Save stacked X/Y matrices and metadata to disk."""
    save_dir = f'{BASE_DATA_PATH}/{mouse_ID}/{date}/'
    os.makedirs(save_dir, exist_ok=True)

    glm_data = {
        'X': X,
        'Y': Y,
        'metadata': metadata,
        'predictor_names': predictor_names,
        'predictor_info': predictor_info,
        'group_indices': group_indices,
        'design_matrices': design_matrices,
        'builder_params': {
            'sampling_rate': 15.6,
            'n_pos_bases': 10,
            'stem_y_max': 500.0,
            'pos_bump_width_ratio': 4.0,
            'audio_start_position': 50.0,
            'n_vel_bases': 5,
            'n_place_fields': 30,
            'n_turn_bases': 4,
            'turn_window': 1.0,
            'n_reward_bases': 6,
            'reward_window': 3.0,
            'temporal_bump_width_ratio': 2.0,
        },
        'neural_metric': 'deconv',
        'mouse_ID': mouse_ID,
        'date': date,
    }

    save_path = os.path.join(save_dir, 'glm_data_stacked.pkl')
    print(f'Saving GLM data to: {save_path}')
    print(f'  X: {X.shape} | Y: {Y.shape} | '
          f'{len(np.unique(metadata["trial_id"]))} trials | '
          f'{len(predictor_names)} predictors')

    with open(save_path, 'wb') as f:
        pickle.dump(glm_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f'  ✓ Saved ({file_size_mb:.2f} MB)')


def open_glm_stacked_data(mouse_ID, date, print_summary=True):
    """
    Load previously saved stacked GLM data.

    Y is scaled by Y_SCALE (currently ×100) to improve Poisson GLM
    numerical stability, then optionally Gaussian-smoothed (σ=Y_SMOOTH_SIGMA
    frames) and clipped to ≥ 0 to preserve non-negativity for Poisson fitting.
    Change Y_SCALE or Y_SMOOTH_SIGMA at the top of this module; both require
    re-running fit_glm_single_session.
    """
    load_path = f'{BASE_DATA_PATH}/{mouse_ID}/{date}/glm_data_stacked.pkl'
    print(f'Loading: {load_path}')

    with open(load_path, 'rb') as f:
        glm_data = pickle.load(f)

    X              = glm_data['X']
    Y              = glm_data['Y'] * Y_SCALE

    if Y_SMOOTH_SIGMA > 0:
        Y = gaussian_filter1d(Y, sigma=Y_SMOOTH_SIGMA, axis=0)
        Y = np.clip(Y, 0.0, None)   # enforce non-negativity for Poisson GLM

    metadata       = glm_data['metadata']
    predictor_names = glm_data['predictor_names']
    predictor_info  = glm_data['predictor_info'].copy()
    group_indices   = glm_data['group_indices']

    smooth_str = f'smoothed σ={Y_SMOOTH_SIGMA}fr + clipped' if Y_SMOOTH_SIGMA > 0 else 'no smoothing'
    print(f'  ✓ Loaded  X: {X.shape}  Y: {Y.shape}  [{smooth_str}]')

    if print_summary:
        print('\nData Verification')
        print('=' * 60)
        print(f'\nX — mean: {np.mean(X):.6f}  std: {np.std(X):.3f}  '
              f'range: [{np.min(X):.2f}, {np.max(X):.2f}]  '
              f'NaN: {np.sum(np.isnan(X))}')
        print(f'Y — mean: {np.mean(Y):.3f}  std: {np.std(Y):.3f}  '
              f'range: [{np.min(Y):.2f}, {np.max(Y):.2f}]  '
              f'NaN: {np.sum(np.isnan(Y))}')
        print('\nContext distribution:')
        for ctx in [0, 1, 2]:
            n = np.sum(metadata['context'] == ctx)
            print(f'  Context {ctx}: {n:,} frames ({n / len(metadata["context"]) * 100:.1f}%)')
        print('\n✓ Verification complete')

    return X, Y, metadata, predictor_names, predictor_info, group_indices


# ---------------------------------------------------------------------------
# Predictor building
# ---------------------------------------------------------------------------

def generate_predictors(glm_data_dict, plot_predictors=False, include_velocity=True):
    """
    Build per-trial design matrices using GLMPredictorBuilder.

    Parameters
    ----------
    glm_data_dict    : dict | list
    plot_predictors  : bool
    include_velocity : bool  — include velocity × position predictors (default False).
                               Velocity is excluded from Model 1 to reduce collinearity
                               with position_only and choice_spatial predictors.
                               Pass True to re-enable for a velocity-specific analysis.

    Returns
    -------
    design_matrices  : list of ndarray
    predictor_names  : list of str
    predictor_info   : dict
    """
    builder = GLMPredictorBuilder(
        sampling_rate=15.6,
        n_pos_bases=10,
        stem_y_max=500.0,
        pos_bump_width_ratio=4.0,
        audio_start_position=50.0,
        # position_only bumps are restricted to the proximal zone (0–350 cm).
        # The distal maze (y > 350 cm) is covered exclusively by choice_spatial,
        # eliminating collinearity between position_only and choice_spatial.
        pos_only_y_max=350.0,
        n_vel_bases=5,
        n_place_fields=30,
        n_turn_bases=4,
        turn_window=1.0,
        n_reward_bases=6,
        reward_window=3.0,
        temporal_bump_width_ratio=2.0,
        include_position_only=True,
        include_sound_position=True,
        include_visual_position=True,
        include_velocity=include_velocity,
        include_spatial_choice=True,
        include_turn_onset=True,
        include_reward=True,
    )

    design_matrices, predictor_names, predictor_info = builder.build_design_matrices(glm_data_dict)

    print(f'Trials: {len(design_matrices)} | Predictors: {len(predictor_names)}')
    print('\nPredictor groups:')
    for group, info in predictor_info.items():
        n_pred = info['end'] - info['start']
        print(f'  {group:20s}: {n_pred:3d}  (cols {info["start"]}–{info["end"] - 1})')

    if plot_predictors:
        builder.plot_predictors(glm_data_dict, design_matrices, predictor_info,
                                start_trial=30, n_trials=3)

    return design_matrices, predictor_names, predictor_info


def stack_design_matrices(glm_data_dict, design_matrices, predictor_info,
                           neural_metric='deconv', print_summary=False):
    """
    Stack per-trial design matrices and neural data into full X / Y matrices.

    Follows Harvey Lab tutorial:
    - Concatenates all trials
    - Z-scores X along the sample dimension for even regularisation
    - Tracks frame-level metadata for balanced train/test splitting

    Parameters
    ----------
    glm_data_dict   : dict | list
    design_matrices : list of ndarray
    predictor_info  : dict
    neural_metric   : str  ('deconv', 'dff', 'z_dff')
    print_summary   : bool

    Returns
    -------
    X             : ndarray  (n_samples, n_predictors)  — z-scored
    Y             : ndarray  (n_samples, n_neurons)     — raw (×10 applied on load)
    metadata      : dict of ndarrays
    group_indices : list of str
    """
    if isinstance(glm_data_dict, dict):
        trial_indices = sorted(glm_data_dict.keys())
    else:
        trial_indices = range(len(glm_data_dict))

    X_raw = np.vstack(design_matrices)

    Y_list               = []
    trial_id_list        = []
    context_list         = []
    choice_list          = []
    outcome_list         = []
    audio_stim_list      = []
    visual_stim_list     = []
    trial_type_list      = []
    frame_in_trial_list  = []
    y_position_list      = []
    turn_frame_list      = []   # turn_frame index within each trial, broadcast to all frames

    # Thresholds matching GLMPredictorBuilder defaults
    _TURN_Y_THRESH   = 505.0
    _TURN_X_THRESH   = 20.0

    for trial_idx in trial_indices:
        trial_data  = glm_data_dict[trial_idx]
        neural_data = trial_data['neural'][neural_metric]
        n_frames    = neural_data.shape[0]
        movement    = trial_data.get('movement', {})
        events      = trial_data.get('events', {})
        choice      = trial_data.get('choice', 0)

        Y_list.append(neural_data)
        trial_id_list.append(np.full(n_frames, trial_idx, dtype=int))
        context_list.append(np.full(n_frames, trial_data.get('context', -1), dtype=int))
        choice_list.append(np.full(n_frames, trial_data.get('choice', -1), dtype=int))
        outcome_list.append(np.full(n_frames, trial_data.get('outcome', -1), dtype=int))
        audio_stim_list.append(np.full(n_frames, trial_data.get('Audio_Stim', np.nan), dtype=float))
        visual_stim_list.append(np.full(n_frames, trial_data.get('Visual_Stim', np.nan), dtype=float))
        trial_type_list.append(np.full(n_frames, trial_data.get('trial_type', -1), dtype=int))
        frame_in_trial_list.append(np.arange(n_frames, dtype=int))

        # y_position — pad / truncate to n_frames
        y_pos = np.asarray(movement.get('y_position', np.zeros(n_frames))).flatten()
        if y_pos.size < n_frames:
            y_pos = np.pad(y_pos, (0, n_frames - y_pos.size), constant_values=np.nan)
        y_position_list.append(y_pos[:n_frames].astype(float))

        # turn_frame — use stored event if available, otherwise derive from kinematics
        if 'turn_onset' in events:
            turn_f = int(events['turn_onset'])
        else:
            x_vel  = np.asarray(movement.get('x_velocity', [])).flatten()
            turn_f = n_frames - 1   # fallback
            for i in range(min(len(y_pos), len(x_vel))):
                if y_pos[i] > _TURN_Y_THRESH:
                    if (choice == 0 and x_vel[i] < -_TURN_X_THRESH) or \
                       (choice == 1 and x_vel[i] >  _TURN_X_THRESH):
                        turn_f = i
                        break
        turn_frame_list.append(np.full(n_frames, turn_f, dtype=int))

    Y_raw = np.vstack(Y_list)

    metadata = {
        'trial_id':           np.concatenate(trial_id_list),
        'context':            np.concatenate(context_list),
        'choice':             np.concatenate(choice_list),
        'outcome':            np.concatenate(outcome_list),
        'audio_stim':         np.concatenate(audio_stim_list),
        'visual_stim':        np.concatenate(visual_stim_list),
        'trial_type':         np.concatenate(trial_type_list),
        'frame_in_trial':     np.concatenate(frame_in_trial_list),
        'y_position':         np.concatenate(y_position_list),
        'turn_frame_in_trial': np.concatenate(turn_frame_list),
    }

    # Z-score X so regularisation has equal effect across features
    X = scipy.stats.zscore(X_raw, axis=0, nan_policy='omit')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Y = Y_raw.copy()  # ×10 scaling is applied in open_glm_stacked_data

    group_indices = []
    for group_name, info in predictor_info.items():
        n_pred = info['end'] - info['start']
        group_indices.extend([group_name] * n_pred)

    if print_summary:
        print('=' * 60)
        print('STACKED DATA SUMMARY')
        print('=' * 60)
        print(f'\nX: {X.shape}  range [{X.min():.3f}, {X.max():.3f}]  '
              f'mean {X.mean():.6f}  std {X.std():.6f}')
        print(f'Y: {Y_raw.shape}  range [{Y_raw.min():.3f}, {Y_raw.max():.3f}]')
        print(f'\nTrials: {len(np.unique(metadata["trial_id"]))} | '
              f'Frames/trial (mean): {X.shape[0] / len(np.unique(metadata["trial_id"])):.1f}')
        for ctx in [0, 1, 2]:
            n = np.sum(metadata['context'] == ctx)
            print(f'  Context {ctx}: {n:,} frames ({n/X.shape[0]*100:.1f}%)')
        print(f'\n  Left:    {np.sum(metadata["choice"]==0):,}  '
              f'Right: {np.sum(metadata["choice"]==1):,}')
        print(f'  Correct: {np.sum(metadata["outcome"]==1):,}  '
              f'Error:  {np.sum(metadata["outcome"]==0):,}')
        print('=' * 60)

    return X, Y, metadata, group_indices


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features_for_glm(predictor_info):
    """
    Split the compound 'sound' and 'visual' predictor groups into four
    interpretable sub-groups for group-lasso regularisation.

    Returns
    -------
    predictor_info_split : dict
    feature_group_size   : ndarray of int
    """
    new_predictor_info = {}
    for group_name, info in predictor_info.items():
        if group_name == 'sound':
            n_half = (info['end'] - info['start']) // 2
            new_predictor_info['audio_context'] = {
                'start': info['start'],
                'end':   info['start'] + n_half,
                'names': info.get('names', [])[:n_half],
            }
            new_predictor_info['audio_stim'] = {
                'start': info['start'] + n_half,
                'end':   info['end'],
                'names': info.get('names', [])[n_half:],
            }
        elif group_name == 'visual':
            n_half = (info['end'] - info['start']) // 2
            new_predictor_info['visual_context'] = {
                'start': info['start'],
                'end':   info['start'] + n_half,
                'names': info.get('names', [])[:n_half],
            }
            new_predictor_info['visual_stim'] = {
                'start': info['start'] + n_half,
                'end':   info['end'],
                'names': info.get('names', [])[n_half:],
            }
        else:
            new_predictor_info[group_name] = info

    feature_group_size = np.array(
        [info['end'] - info['start'] for info in new_predictor_info.values()]
    )

    print('Predictor groups:')
    print('=' * 60)
    for group_name, info in new_predictor_info.items():
        n = info['end'] - info['start']
        print(f'  {group_name:20s}: {n:2d} features  (cols {info["start"]}–{info["end"] - 1})')
    print(f'\nTotal: {np.sum(feature_group_size)} features')

    return new_predictor_info, feature_group_size


# ---------------------------------------------------------------------------
# Trial-averaged R² helper
# ---------------------------------------------------------------------------

def compute_trial_avg_r2(Y, Y_pred, metadata, split_by='choice', min_frames=3):
    """
    Compute trial-averaged variance explained (R²) per neuron on the test set.

    For each condition (context × split_value combination), the mean activity
    per frame-in-trial bin is computed for both data and model prediction.
    R² is then calculated across all (condition, bin) pairs — i.e. how well
    the model captures the shape of the tuning curves, not just frame-level
    statistics.

    Choosing the right split variable matters for different neuron types:
        split_by='choice'      — best for choice / position / turn neurons
        split_by='outcome'     — best for outcome neurons
        split_by='audio_stim'  — best for audio stimulus identity neurons
        split_by='visual_stim' — best for visual stimulus identity neurons

    Call compute_best_trial_avg_r2 to automatically take the max across all
    four split variables.

    This is a supplementary metric alongside frac_dev:
        frac_dev       — Poisson log-likelihood quality (frame-level)
        trial_avg_r2   — variance explained in condition-averaged traces

    Parameters
    ----------
    Y         : ndarray (n_frames, n_neurons)
    Y_pred    : ndarray (n_frames, n_neurons)
    metadata  : dict of ndarrays for the test frames
                must contain 'frame_in_trial', 'context', and the split_by key
    split_by  : str — metadata key to split conditions on alongside context.
                One of 'choice', 'outcome', 'audio_stim', 'visual_stim'.
    min_frames: int — minimum frames in a bin to include it

    Returns
    -------
    trial_avg_r2 : ndarray (n_neurons,)  — R² values, clipped to [-1, 1]
    """
    ctx_arr   = metadata['context']
    split_arr = metadata[split_by]
    frit_arr  = metadata['frame_in_trial']
    max_frame = int(frit_arr.max()) + 1

    split_vals = np.unique(split_arr)
    conditions = [(ctx, sv) for ctx in np.unique(ctx_arr) for sv in split_vals]

    Y_avg_rows  = []
    Yp_avg_rows = []

    for ctx, sv in conditions:
        cond_mask = (ctx_arr == ctx) & (split_arr == sv)
        if cond_mask.sum() == 0:
            continue
        for f in range(max_frame):
            bin_mask = cond_mask & (frit_arr == f)
            if bin_mask.sum() >= min_frames:
                Y_avg_rows.append(Y[bin_mask].mean(axis=0))
                Yp_avg_rows.append(Y_pred[bin_mask].mean(axis=0))

    if len(Y_avg_rows) == 0:
        return np.zeros(Y.shape[1])

    Y_avg  = np.array(Y_avg_rows)    # (n_bins_total, n_neurons)
    Yp_avg = np.array(Yp_avg_rows)

    grand_mean = Y_avg.mean(axis=0)
    SS_res = np.sum((Y_avg - Yp_avg) ** 2, axis=0)
    SS_tot = np.sum((Y_avg - grand_mean) ** 2, axis=0)

    r2 = np.where(SS_tot > 1e-12, 1.0 - SS_res / SS_tot, 0.0)
    return np.clip(r2, -1.0, 1.0)


def compute_best_trial_avg_r2(Y, Y_pred, metadata, min_frames=3):
    """
    Compute trial-averaged R² under four condition splits and return the
    per-neuron maximum.

    Returns
    -------
    r2_best   : ndarray (n_neurons,) — element-wise max across all splits
    r2_by_split : dict  split_key -> ndarray (n_neurons,)
        Individual R² arrays for each split variable, useful for post-hoc
        inspection of which split drove a neuron's best R².
    """
    split_keys = ['choice', 'outcome', 'audio_stim', 'visual_stim']
    r2_by_split = {}
    for key in split_keys:
        if key not in metadata:
            r2_by_split[key] = np.zeros(Y.shape[1])
        else:
            r2_by_split[key] = compute_trial_avg_r2(
                Y, Y_pred, metadata, split_by=key, min_frames=min_frames
            )
    r2_best = np.maximum.reduce(list(r2_by_split.values()))
    return r2_best, r2_by_split


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_glm_single_session(X, Y, metadata, feature_group_size, predictor_info,
                            mouse_ID, date):
    """
    Fit a single combined GLM on balanced data from all 3 contexts.

    Pipeline
    --------
    1. Per-context balanced 70/30 trial splits (stratified by choice)
    2. Combine splits across contexts → single train / test set
    3. Fit GLM_CV with group-lasso regularisation
    4. Select optimal lambda (max CV deviance)
    5. Evaluate on combined test set and per-context held-out sets
    6. Save results to disk

    Returns
    -------
    model_cv         : glm_class.GLM_CV
    all_results      : dict
    predictor_info   : dict  (unchanged, passed through)
    feature_group_size : ndarray  (unchanged, passed through)
    """
    context_names = {0: 'CG', 1: 'visual', 2: 'audio'}
    rng = np.random.default_rng(seed=42)

    # ── 1. Per-context balanced splits ────────────────────────────────────
    print('Per-context balanced splits:')
    train_masks = {}
    test_masks  = {}

    for ctx in [0, 1, 2]:
        ctx_frame_mask = metadata['context'] == ctx
        meta_ctx = {k: v[ctx_frame_mask] for k, v in metadata.items()}

        trial_df = (
            pd.DataFrame({'trial_id': meta_ctx['trial_id'],
                          'choice':   meta_ctx['choice'],
                          'outcome':  meta_ctx['outcome']})
            .drop_duplicates(subset='trial_id')
        )

        # Balance L/R choice (keep all trials, no hard outcome balance).
        # A combined choice×outcome stratum label is used only so that
        # train_test_split preserves the natural outcome ratio in both
        # halves — no trials are discarded for this purpose.
        left_trials  = trial_df[trial_df['choice'] == 0]['trial_id'].to_numpy()
        right_trials = trial_df[trial_df['choice'] == 1]['trial_id'].to_numpy()
        n_bal = min(len(left_trials), len(right_trials))

        if n_bal == 0:
            print(f'  Skipping context {ctx}: insufficient trials.')
            continue

        left_sel  = rng.choice(left_trials,  size=n_bal, replace=False)
        right_sel = rng.choice(right_trials, size=n_bal, replace=False)
        balanced_ids = np.concatenate([left_sel, right_sel])

        bal_df = trial_df[trial_df['trial_id'].isin(balanced_ids)].copy()

        # Combined stratum label: stratify 70/30 split by choice × outcome so
        # both train and test preserve the natural correct/error proportions.
        bal_df['strat'] = (bal_df['choice'].astype(str) + '_'
                           + bal_df['outcome'].astype(str))

        # sklearn requires each stratum to appear at least twice for
        # stratified splitting.  If any stratum has only one member, fall
        # back to stratifying on choice alone.
        strat_counts = bal_df['strat'].value_counts()
        if (strat_counts < 2).any():
            stratify_col = bal_df['choice']
        else:
            stratify_col = bal_df['strat']

        train_trials, test_trials = train_test_split(
            bal_df['trial_id'].to_numpy(),
            test_size=0.3,
            random_state=0,
            stratify=stratify_col,
        )

        ctx_global_idx = np.where(ctx_frame_mask)[0]
        tr_local = np.isin(meta_ctx['trial_id'], train_trials)
        te_local = np.isin(meta_ctx['trial_id'], test_trials)

        train_global = np.zeros(len(metadata['context']), dtype=bool)
        test_global  = np.zeros(len(metadata['context']), dtype=bool)
        train_global[ctx_global_idx[tr_local]] = True
        test_global[ctx_global_idx[te_local]]  = True

        train_masks[ctx] = train_global
        test_masks[ctx]  = test_global

        print(f'  ctx {ctx} ({context_names[ctx]:6s}): {len(balanced_ids)} balanced trials | '
              f'train {train_global.sum():,} frames | test {test_global.sum():,} frames')

    # ── 2. Combined train / test ───────────────────────────────────────────
    if not train_masks:
        raise ValueError('No contexts had sufficient balanced trials — cannot fit GLM.')

    n_frames = len(metadata['context'])
    combined_train_mask = np.zeros(n_frames, dtype=bool)
    combined_test_mask  = np.zeros(n_frames, dtype=bool)
    for ctx in train_masks:   # only iterate over contexts that were actually populated
        combined_train_mask |= train_masks[ctx]
        combined_test_mask  |= test_masks[ctx]

    skipped = [c for c in [0, 1, 2] if c not in train_masks]
    print(f'Contexts used: {sorted(train_masks.keys())}'
          + (f'  (skipped — insufficient trials: {skipped})' if skipped else ''))

    X_train = X[combined_train_mask]
    Y_train = Y[combined_train_mask]
    X_test  = X[combined_test_mask]
    Y_test  = Y[combined_test_mask]

    print(f'\nCombined  train: {X_train.shape[0]:,} frames | test: {X_test.shape[0]:,} frames')

    # ── 3. Fit GLM_CV ──────────────────────────────────────────────────────
    # n_folds=3 means exactly 3 CV folds (0, 1, 2) are used to select the best
    # lambda per neuron. GLM_CV then performs one mandatory final refit (fold 3)
    # on ALL training data with the selected lambda — this is hardcoded in
    # select_model() which reads weights from w_series_dict[n_folds] and cannot
    # be skipped. learning_rate=1e-3 (Harvey Lab default) prevents gradient
    # explosion on the larger full-data fold. max_iter_per_lambda=5000 caps any
    # runaway fold so the pipeline does not hang indefinitely.
    model_cv = glm_class.GLM_CV(
        n_folds=3,
        auto_split=True,
        split_by_group=True,
        skip_final_fold=True,        # fold n_folds diverges (warm-start mismatch on full data); use fold 0 weights instead
        activation='exp',
        loss_type='poisson',
        regularization='group_lasso',
        lambda_series=10.0 ** np.linspace(-1, -5, 5),
        l1_ratio=0.95,
        smooth_strength=0.1,
        optimizer='adam',
        learning_rate=1e-3,
        max_iter_per_lambda=5000,
        convergence_tol=1e-5,        # 1e-6 was too tight — lambda 0 cold start never converged in 5000 iters
    )

    print(f'\nFitting GLM: {Y.shape[1]} neurons | {X_train.shape[0]:,} train frames | '
          f'{len(feature_group_size)} groups …')

    group_idx_train = metadata['trial_id'][combined_train_mask]
    model_cv.fit(X_train, Y_train,
                 feature_group_size=feature_group_size,
                 group_idx=group_idx_train)

    # ── 4. Select optimal lambda ───────────────────────────────────────────
    model_cv.select_model(se_fraction=0., make_fig=False)
    cv_frac_dev = np.maximum(model_cv.selected_frac_dev_expl_cv, 0.0)
    print(f'\nCV frac_dev: median={np.median(cv_frac_dev):.3f} | '
          f'neurons >0.05: {np.sum(cv_frac_dev > 0.05)}/{len(cv_frac_dev)}')

    # ── 5. Evaluate on test sets ───────────────────────────────────────────
    Y_pred_test = make_prediction(
        X_test, model_cv.selected_w, model_cv.selected_w0.flatten(), activation='exp'
    )
    fd_all, _, _ = deviance(Y_pred_test, Y_test)
    test_frac_dev = np.maximum(fd_all, 0.0)
    print(f'Test frac_dev (combined): median={np.median(test_frac_dev):.3f} | '
          f'>0.05: {np.sum(test_frac_dev > 0.05)}/{len(test_frac_dev)}')

    test_frac_dev_per_ctx = {}
    for ctx in [0, 1, 2]:
        X_te_ctx = X[test_masks[ctx]]
        Y_te_ctx = Y[test_masks[ctx]]
        Y_pr_ctx = make_prediction(
            X_te_ctx, model_cv.selected_w, model_cv.selected_w0.flatten(), activation='exp'
        )
        fd_ctx, _, _ = deviance(Y_pr_ctx, Y_te_ctx)
        test_frac_dev_per_ctx[ctx] = np.maximum(fd_ctx, 0.0)
        print(f'  ctx {ctx} ({context_names[ctx]:6s}): '
              f'median={np.median(test_frac_dev_per_ctx[ctx]):.3f} | '
              f'>0.05: {np.sum(test_frac_dev_per_ctx[ctx] > 0.05)}/{len(test_frac_dev_per_ctx[ctx])}')

    # ── 5b. Trial-averaged R² (supplementary metric) ──────────────────────
    # R² is computed under four condition splits (choice / outcome /
    # audio_stim / visual_stim) and the per-neuron maximum is reported.
    # This ensures that e.g. outcome neurons are not penalised because their
    # signal is blurred when averaging within choice conditions.
    meta_test = {k: v[combined_test_mask] for k, v in metadata.items()}
    test_trial_avg_r2, test_r2_by_split = compute_best_trial_avg_r2(
        Y_test, Y_pred_test, meta_test
    )
    print(f'Trial-avg R² (best-split, combined):  '
          f'median={np.median(test_trial_avg_r2):.3f} | '
          f'>0.1: {np.sum(test_trial_avg_r2 > 0.1)}/{len(test_trial_avg_r2)}')
    for split_key, r2_arr in test_r2_by_split.items():
        print(f'  split={split_key:<12s}: median={np.median(r2_arr):.3f}')

    # Per-context best-split R²
    test_trial_avg_r2_per_ctx    = {}
    test_r2_by_split_per_ctx     = {}
    for ctx in [0, 1, 2]:
        te_mask  = test_masks[ctx]
        meta_ctx = {k: v[te_mask] for k, v in metadata.items()}
        Y_te_ctx = Y[te_mask]
        Y_pr_ctx = make_prediction(
            X[te_mask], model_cv.selected_w, model_cv.selected_w0.flatten(), activation='exp'
        )
        r2_best_ctx, r2_splits_ctx = compute_best_trial_avg_r2(Y_te_ctx, Y_pr_ctx, meta_ctx)
        test_trial_avg_r2_per_ctx[ctx]  = r2_best_ctx
        test_r2_by_split_per_ctx[ctx]   = r2_splits_ctx
        print(f'  ctx {ctx} ({context_names[ctx]:6s}): '
              f'median={np.median(r2_best_ctx):.3f} | '
              f'>0.1: {np.sum(r2_best_ctx > 0.1)}/{len(r2_best_ctx)}')

    # ── 6. Assemble & save results ─────────────────────────────────────────
    all_results = {
        'weights':                       model_cv.selected_w,
        'w0':                            model_cv.selected_w0,
        'cv_frac_dev':                   cv_frac_dev,
        'test_frac_dev':                 test_frac_dev,
        'test_frac_dev_per_ctx':         test_frac_dev_per_ctx,
        # R² — best split (per-neuron max across choice/outcome/stim splits)
        'test_trial_avg_r2':             test_trial_avg_r2,
        'test_r2_by_split':              test_r2_by_split,
        'test_trial_avg_r2_per_ctx':     test_trial_avg_r2_per_ctx,
        'test_r2_by_split_per_ctx':      test_r2_by_split_per_ctx,
        'predictor_info':                predictor_info,
        'train_masks':                   train_masks,
        'test_masks':                    test_masks,
        'combined_train_mask':           combined_train_mask,
        'combined_test_mask':            combined_test_mask,
        'feature_group_size':            feature_group_size,
        'context_names':                 context_names,
        'mouse_ID':                      mouse_ID,
        'date':                          date,
    }

    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_1_DIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'glm_M1_full.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)

    mb = os.path.getsize(save_path) / 1e6
    print(f'\n  ✓ Results saved → {save_path}  ({mb:.2f} MB)')

    return model_cv, all_results, predictor_info, feature_group_size


# ---------------------------------------------------------------------------
# Lesion analysis
# ---------------------------------------------------------------------------

def run_lesion_models(X, Y, metadata, res, pred_info, mouse_ID, date,
                      skip_existing=True):
    """
    True lesion analysis: for each predictor group, refit the GLM with that
    group's columns entirely removed from X, then evaluate on the same
    held-out test set as the full model.

    Delta metrics (full − lesion) capture each group's *unique* contribution
    to model performance — what the model loses when it cannot use that group
    at all.  Unlike weight ablation (zeroing post-hoc weights), refitting lets
    the remaining groups compensate as much as they can, so the delta reflects
    variance that is truly unique to the lesioned group.

    The same train/test masks from `res` are reused, ensuring a fair comparison
    against the full model.

    Parameters
    ----------
    X             : ndarray (n_frames, n_features) — z-scored design matrix
    Y             : ndarray (n_frames, n_neurons)  — neural data (Y_SCALE-scaled)
    metadata      : dict of ndarrays — frame-level metadata
    res           : dict — saved results from fit_glm_single_session
                    must contain: combined_train_mask, combined_test_mask,
                                  test_masks, test_frac_dev, test_trial_avg_r2,
                                  feature_group_size
    pred_info     : dict — split predictor info from prepare_features_for_glm
                    (same dict used during fitting)
    mouse_ID      : str
    date          : str
    skip_existing : bool
                    If True (default), load results from any already-saved
                    glm_M1_lesion_{group}.pkl files instead of refitting those
                    groups.  Useful for resuming an interrupted run.

    Returns
    -------
    lesion_results : dict
        'full_test_frac_dev'        : ndarray (n_neurons,)
        'full_test_trial_avg_r2'    : ndarray (n_neurons,)
        'groups'                    : list of str — group names
        'lesion_test_frac_dev'      : dict  group -> ndarray (n_neurons,)
        'lesion_test_trial_avg_r2'  : dict  group -> ndarray (n_neurons,)
        'delta_frac_dev'            : dict  group -> ndarray (n_neurons,)
                                      positive = group uniquely contributes
        'delta_r2'                  : dict  group -> ndarray (n_neurons,)
        'pred_info'                 : dict (passed through)
        'combined_test_mask'        : ndarray bool
        'mouse_ID'                  : str
        'date'                      : str
    """
    combined_train_mask = res['combined_train_mask']
    combined_test_mask  = res['combined_test_mask']
    test_masks          = res['test_masks']
    feature_group_size  = res['feature_group_size']

    full_frac_dev    = res['test_frac_dev']
    full_r2          = res.get('test_trial_avg_r2',
                               np.zeros_like(full_frac_dev))  # graceful fallback
    full_r2_by_split = res.get('test_r2_by_split', {})

    X_train_full    = X[combined_train_mask]
    X_test          = X[combined_test_mask]
    Y_train         = Y[combined_train_mask]
    Y_test          = Y[combined_test_mask]
    meta_test       = {k: v[combined_test_mask] for k, v in metadata.items()}
    group_idx_train = metadata['trial_id'][combined_train_mask]

    group_names = list(pred_info.keys())
    n_neurons   = Y.shape[1]

    lesion_frac_dev    = {}
    lesion_r2          = {}
    lesion_r2_by_split = {}
    delta_frac_dev     = {}
    delta_r2           = {}

    print(f'\nRunning lesion models: {mouse_ID} / {date}')
    print(f'  Full model — test frac_dev median={np.median(full_frac_dev):.3f} | '
          f'trial-avg R² median={np.median(full_r2):.3f}')
    print(f'  {len(group_names)} groups × refit  ({n_neurons} neurons)\n')

    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_1_DIR)
    os.makedirs(save_dir, exist_ok=True)

    newly_fitted = set()  # track which groups were actually refit (vs loaded)

    for lesion_group, lesion_info in pred_info.items():
        g_start = lesion_info['start']
        g_end   = lesion_info['end']
        g_width = g_end - g_start

        grp_path = os.path.join(save_dir, f'glm_M1_lesion_{lesion_group}.pkl')

        # ── Load from cache if already saved ───────────────────────────────
        if skip_existing and os.path.exists(grp_path):
            with open(grp_path, 'rb') as f:
                cached = pickle.load(f)
            lesion_frac_dev[lesion_group]    = cached['lesion_test_frac_dev']
            lesion_r2[lesion_group]          = cached['lesion_test_trial_avg_r2']
            lesion_r2_by_split[lesion_group] = cached.get('lesion_r2_by_split', {})
            delta_frac_dev[lesion_group]     = cached['delta_frac_dev']
            delta_r2[lesion_group]           = cached['delta_r2']
            print(f'  [{group_names.index(lesion_group) + 1}/{len(group_names)}] '
                  f'"{lesion_group}" — [LOADED from cache]')
            continue
        # ──────────────────────────────────────────────────────────────────

        print(f'  [{group_names.index(lesion_group) + 1}/{len(group_names)}] '
              f'Lesion "{lesion_group}" (cols {g_start}–{g_end - 1}, '
              f'{g_width} features) … ', end='', flush=True)

        newly_fitted.add(lesion_group)

        # ── Build X_lesion: delete the group's columns ─────────────────────
        col_keep           = np.ones(X.shape[1], dtype=bool)
        col_keep[g_start:g_end] = False
        X_lesion_train     = X_train_full[:, col_keep]
        X_lesion_test      = X_test[:, col_keep]

        # ── Rebuild feature_group_size (exclude lesioned group) ────────────
        lesion_group_idx          = group_names.index(lesion_group)
        feature_group_size_lesion = np.delete(feature_group_size, lesion_group_idx)

        # ── Refit GLM_CV with same hyperparameters as fit_glm_single_session ─
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

        # ── Evaluate on same combined test set ─────────────────────────────
        Y_pred_lesion = make_prediction(
            X_lesion_test,
            model_lesion.selected_w,
            model_lesion.selected_w0.flatten(),
            activation='exp',
        )

        fd_lesion, _, _ = deviance(Y_pred_lesion, Y_test)
        fd_lesion = np.maximum(fd_lesion, 0.0)

        r2_lesion_best, r2_lesion_splits = compute_best_trial_avg_r2(
            Y_test, Y_pred_lesion, meta_test
        )

        lesion_frac_dev[lesion_group]    = fd_lesion
        lesion_r2[lesion_group]          = r2_lesion_best
        lesion_r2_by_split[lesion_group] = r2_lesion_splits
        delta_frac_dev[lesion_group]     = full_frac_dev - fd_lesion
        delta_r2[lesion_group]           = full_r2 - r2_lesion_best

        med_fd  = np.median(fd_lesion)
        med_dfd = np.median(delta_frac_dev[lesion_group])
        med_dr2 = np.median(delta_r2[lesion_group])
        print(f'✓  lesion frac_dev={med_fd:.3f}  '
              f'Δfrac_dev={med_dfd:+.3f}  Δr²={med_dr2:+.3f}')

        # ── Free per-group temporaries before the next group is fit ────────
        del model_lesion, X_lesion_train, X_lesion_test
        del Y_pred_lesion, fd_lesion, r2_lesion_best, r2_lesion_splits
        del col_keep, feature_group_size_lesion
        gc.collect()

    # ── Assemble & save ────────────────────────────────────────────────────
    lesion_results = {
        'full_test_frac_dev':          full_frac_dev,
        'full_test_trial_avg_r2':      full_r2,
        'full_r2_by_split':            full_r2_by_split,
        'groups':                      group_names,
        'lesion_test_frac_dev':        lesion_frac_dev,
        'lesion_test_trial_avg_r2':    lesion_r2,
        'lesion_r2_by_split':          lesion_r2_by_split,
        'delta_frac_dev':              delta_frac_dev,
        'delta_r2':                    delta_r2,
        'pred_info':                   pred_info,
        'combined_test_mask':          combined_test_mask,
        'mouse_ID':                    mouse_ID,
        'date':                        date,
    }

    # ── Granular per-group files (only write newly fitted groups) ──────────
    for grp in newly_fitted:
        grp_results = {
            'group':                      grp,
            'full_test_frac_dev':         full_frac_dev,
            'full_test_trial_avg_r2':     full_r2,
            'full_r2_by_split':           full_r2_by_split,
            'lesion_test_frac_dev':       lesion_frac_dev[grp],
            'lesion_test_trial_avg_r2':   lesion_r2[grp],
            'lesion_r2_by_split':         lesion_r2_by_split[grp],
            'delta_frac_dev':             delta_frac_dev[grp],
            'delta_r2':                   delta_r2[grp],
            'mouse_ID':                   mouse_ID,
            'date':                       date,
        }
        grp_path = os.path.join(save_dir, f'glm_M1_lesion_{grp}.pkl')
        with open(grp_path, 'wb') as f:
            pickle.dump(grp_results, f)

    # ── Combined all-lesion file ────────────────────────────────────────────
    all_path = os.path.join(save_dir, 'glm_M1_lesion_all.pkl')
    with open(all_path, 'wb') as f:
        pickle.dump(lesion_results, f)

    mb = os.path.getsize(all_path) / 1e6
    print(f'\n  ✓ Lesion results saved → {save_dir}/')
    print(f'      glm_M1_lesion_{{group}}.pkl  × {len(group_names)}')
    print(f'      glm_M1_lesion_all.pkl        ({mb:.2f} MB)')

    return lesion_results


def run_all_lesion_models(sessions_dict, pred_info_split_cache=None):
    """
    Run true lesion models for all sessions in sessions_dict.

    Loads the saved glm_M1_full.pkl and glm_data_stacked.pkl for each
    session, then calls run_lesion_models.  Results are saved per-session to
    Model_1_results/glm_M1_lesion_{group}.pkl and glm_M1_lesion_all.pkl.

    Parameters
    ----------
    sessions_dict       : dict  mouse_ID -> list of dates
    pred_info_split_cache : dict or None
        Optional pre-computed {(mouse_ID, date): pred_info} mapping to avoid
        reloading and re-splitting pred_info from disk for every session.
        If None, pred_info is loaded and split from glm_data_stacked.pkl.

    Returns
    -------
    all_lesion_results : list of (mouse_ID, date, lesion_results)
        lesion_results is None for sessions that failed or were skipped.
    """
    import gc

    all_lesion_results = []

    for mouse_ID, dates in sessions_dict.items():
        for date in dates:
            res_path = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_1_DIR, 'glm_M1_full.pkl')
            if not os.path.exists(res_path):
                print(f'  [SKIP] {mouse_ID} / {date} — no glm_M1_full.pkl')
                all_lesion_results.append((mouse_ID, date, None))
                continue

            X = Y = None
            try:
                with open(res_path, 'rb') as f:
                    res = pickle.load(f)

                X, Y, metadata, _, pred_info_raw, _ = open_glm_stacked_data(
                    mouse_ID, date, print_summary=False
                )
                pred_info, _ = prepare_features_for_glm(pred_info_raw)

                lesion_results = run_lesion_models(
                    X, Y, metadata, res, pred_info,
                    mouse_ID=mouse_ID, date=date,
                )
                all_lesion_results.append((mouse_ID, date, lesion_results))

            except Exception:
                import traceback
                print(f'\n  [FAILED] {mouse_ID} / {date}')
                traceback.print_exc()
                all_lesion_results.append((mouse_ID, date, None))

            finally:
                del X, Y
                gc.collect()

    n_ok = sum(1 for _, _, r in all_lesion_results if r is not None)
    print(f'\nLesion analysis complete: {n_ok}/{len(all_lesion_results)} sessions succeeded.')
    return all_lesion_results
