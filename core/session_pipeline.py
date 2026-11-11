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
5b. X, pred_info = add_coupling_predictor(X, Y, pred_info, ...)  # optional
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

import core.glm_class as glm_class
from core.glm_class import make_prediction, deviance
from core.predictor_builder import GLMPredictorBuilder, create_cosine_bumps
from sklearn.decomposition import PCA as _SklearnPCA
from scipy.ndimage import gaussian_filter1d

from core.paths import BASE_DATA_PATH

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

def generate_predictors(glm_data_dict, plot_predictors=False, include_velocity=True,
                        audio_start_position=50.0):
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
    audio_start_position : float — corridor position (cm) where the audio predictor's
                               temporal support begins (real data: 50, audio plays past
                               that corridor point). Set to 0.0 to give the audio stimulus
                               predictor the SAME frame-0 support as visual — required for
                               abstract models (e.g. the RNN readout) where audio and visual
                               evidence are simultaneous, otherwise early audio-driven
                               activity is unreadable and no audio-preferring units appear.

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
        audio_start_position=audio_start_position,
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
# Functional coupling predictor
# ---------------------------------------------------------------------------

def add_coupling_predictor(X, Y, predictor_info,
                            n_pcs=5, n_lag_bases=4, lag_frames=8,
                            exclude_col=None, split_by_context=None,
                            split_labels=('vis', 'aud')):
    """
    Augment the design matrix X with functional-coupling predictors derived
    from the simultaneously recorded population activity (Y).

    exclude_col : int or None
        LEAVE-ONE-OUT. Column index into Y of a TARGET neuron to drop from the
        PCA input, so the coupling predictors never contain the target's own
        activity. Use in a per-target loop when the coupling gain of that target
        is being measured (see subpop_coupling_gain_loo). For the full population
        (n≈100s) the self-contribution is ~1/n and negligible, so None is the
        default; for a small SUBPOPULATION it is material and must be set.

    split_by_context : ndarray or None
        CONTEXT-SPLIT COUPLING WEIGHTS. None (default) → the standard shared
        model: one coupling weight vector per neuron across all frames, giving
        5 groups ``network_pc1 … network_pc5``.

        Pass the per-frame context array (``metadata['context']``, 1 = visual,
        2 = audio) to duplicate each coupling column into a visual-masked and an
        audio-masked copy, giving 10 groups ``network_pc{1..5}_vis`` and
        ``network_pc{1..5}_aud`` of ``n_lag_bases`` columns each. Each neuron
        then gets separate coupling weights per context, and the shared model is
        the nested special case w_vis = w_aud.

        This tests whether neurons that REMAP across contexts (active in both,
        inverted relationship to the population in each) are scored as
        "uncoupled" by a context-blind model: a single shared weight averages
        +w and -w toward zero, and group lasso — being retain-or-discard per
        group — then drops the group outright.

        Only the codes 1 and 2 are accepted; filter to unisensory frames first.

        The array need not be the true context. Passing a DIFFERENT 1/2 labelling
        with the same group structure gives a control model at an identical
        penalty budget — which is required to interpret this model at all, because
        the group-lasso penalty is per-group (sum_g ||w_g||*sqrt|g|), so doubling
        5 coupling groups into 10 makes the split model pay ~2x to express the
        IDENTICAL function. Measured on real data: it declines to pay, shrinking
        each context's weights to ~0.7x instead, which lowers train AND test FDE.
        The null for (split - shared) is therefore NOT zero, and the bias scales
        with the CV-selected lambda, so it must be calibrated empirically:
          * sham mask (random trials -> 1/2): meaningless labels, so any gain is
            pure penalty handicap.  true context effect = csc_context - csc_sham
          * time mask (early/late within each context): pure non-stationarity, so
            it separates "coupling remaps with context" from "coupling drifts over
            the session" — which matters because these sessions are context-BLOCKED.

    split_labels : tuple of two str
        Group-name suffixes for codes 1 and 2 — ('vis','aud') for the context
        split, ('early','late') for the time control, ('sham1','sham2') for the
        sham. Keeps group names honest about what the mask actually encodes.

    Notes on scaling of the masked columns (split_by_context only)
    -------------------------------------------------------------
    Each masked column is centered and scaled using ONLY the frames where its
    mask is active, so within its own support it is zero-mean and unit-variance.
    This is deliberate and NOT cosmetic. Masking AFTER a global z-score would
    leave each column with ~half zeros, reduced variance, and a non-zero mean
    that correlates with the context indicator — so the split groups would (a)
    sit on a different regularisation scale from each other and from the shared
    groups, and (b) steal variance from the sum-coded ``visual_context`` /
    ``audio_context`` groups, inflating apparent coupling gain. Within-support
    centering makes each masked column orthogonal to its own context's
    indicator (their inner product is the sum of a centered variable over
    exactly its support = 0).

    Caveat, documented deliberately: the existing sum-coded context groups are
    z-scored GLOBALLY by stack_design_matrices, so the split coupling groups sit
    on a different scale from those groups. That is the intended trade — the
    orthogonality above is worth more than scale-matching to a group whose own
    scaling is already support-dependent. Absolute FDE from the split model is
    therefore NOT comparable to the shared model's (40 coupling columns vs 20,
    different penalty budget); only the CI-vs-CD contrast within one fitting
    regime is interpretable.

    For each of the top ``n_pcs`` principal components of Y, a set of
    ``n_lag_bases`` causal raised-cosine filters is applied (convolution),
    producing one temporal-basis expansion per PC.  Each PC becomes its own
    named predictor group (``network_pc1`` … ``network_pc{n_pcs}``), so the
    group-lasso and lesion framework handle them automatically.

    Parameters
    ----------
    X             : ndarray (n_frames, n_features)
                    Z-scored design matrix from stack_design_matrices / open_glm_stacked_data.
    Y             : ndarray (n_frames, n_neurons)
                    Raw neural data (not z-scored).
    predictor_info : dict
                    Predictor group index dict from build_design_matrices.
    n_pcs         : int
                    Number of population PCs to include (default 5).
    n_lag_bases   : int
                    Number of raised-cosine lag bases per PC (default 4).
    lag_frames    : int
                    Temporal extent of the lag window in frames (default 8,
                    ≈ 500 ms at 15.6 Hz).  Bases span lags 0 … lag_frames-1.

    Returns
    -------
    X_aug          : ndarray (n_frames, n_features + n_pcs * n_lag_bases)
    predictor_info_aug : dict  (original entries + network_pc1 … network_pcN)
    pca            : fitted sklearn PCA object (for inspection / projection)

    Notes
    -----
    * PCA is fit on the full Y matrix (train + test).  A small amount of
      data leakage to the test set is accepted in exchange for simplicity;
      the coupling signal is still computed from *other* neurons' activity,
      not the target neuron.
    * Convolution is causal: each output frame depends only on the current
      and past frames of the PC signal.
    * Each new column is z-scored independently before concatenation so that
      regularisation treats coupling features on the same scale as the
      behavioural features.
    """
    n_frames, n_neurons = Y.shape

    # ── 1. PCA on neural population ───────────────────────────────────────────
    if exclude_col is not None:                # leave-one-out: drop target neuron
        keep = np.ones(n_neurons, dtype=bool)
        keep[exclude_col] = False
        Y = Y[:, keep]
        n_neurons = Y.shape[1]
    n_pcs = min(n_pcs, n_neurons)
    pca = _SklearnPCA(n_components=n_pcs, whiten=False)
    pc_scores = pca.fit_transform(Y)          # (n_frames, n_pcs)
    var_exp   = pca.explained_variance_ratio_
    print(f'\n  [Coupling] PCA on Y ({n_neurons} neurons):')
    for k in range(n_pcs):
        print(f'    PC{k+1}: {var_exp[k]*100:.1f}% variance explained')
    print(f'    Total: {var_exp.sum()*100:.1f}%')

    # ── 2. Raised-cosine lag bases ─────────────────────────────────────────────
    # Centers evenly spaced over [0, lag_frames-1]; widths chosen so adjacent
    # bumps overlap at their half-height points.
    lag_axis   = np.arange(lag_frames, dtype=float)
    centers    = np.linspace(0, lag_frames - 1, n_lag_bases)
    spacing    = centers[1] - centers[0] if n_lag_bases > 1 else lag_frames
    widths     = np.full(n_lag_bases, spacing * 2.0)
    lag_bases  = create_cosine_bumps(lag_axis, centers, widths)  # (lag_frames, n_lag_bases)

    # ── 3. Convolve each PC with each lag basis (causal) ──────────────────────
    coupling_cols = np.zeros((n_frames, n_pcs * n_lag_bases), dtype=np.float32)
    col_names     = []

    for pc_idx in range(n_pcs):
        signal = pc_scores[:, pc_idx]   # (n_frames,)
        for b_idx in range(n_lag_bases):
            basis   = lag_bases[:, b_idx]   # (lag_frames,)
            # Full convolution then keep causal (first n_frames elements)
            conv    = np.convolve(signal, basis, mode='full')[:n_frames]
            out_col = pc_idx * n_lag_bases + b_idx
            coupling_cols[:, out_col] = conv
            col_names.append(f'network_pc{pc_idx+1}_lag{b_idx}')

    # ── 4. Standardise coupling columns ───────────────────────────────────────
    if split_by_context is None:
        # Shared weights: z-score each column across all frames.
        col_std = coupling_cols.std(axis=0, keepdims=True)
        col_std[col_std == 0] = 1.0
        coupling_cols = (coupling_cols - coupling_cols.mean(axis=0, keepdims=True)) / col_std
        group_specs = [(f'network_pc{i + 1}',
                        col_names[i * n_lag_bases:(i + 1) * n_lag_bases])
                       for i in range(n_pcs)]
    else:
        # Split weights: mask each column per context, standardising WITHIN the
        # mask's support (see docstring — this is what keeps each masked column
        # orthogonal to its own context's indicator).
        ctx_arr = np.asarray(split_by_context)
        if ctx_arr.shape[0] != n_frames:
            raise ValueError(
                f'split_by_context has {ctx_arr.shape[0]} frames but Y has {n_frames}.')
        unexpected = set(np.unique(ctx_arr)) - {1, 2}
        if unexpected:
            raise ValueError(
                f'split_by_context must contain only the codes 1 and 2; '
                f'found {sorted(unexpected)}. Filter to unisensory frames first.')
        if len(split_labels) != 2:
            raise ValueError(f'split_labels must be a pair, got {split_labels!r}')

        split_cols  = []
        group_specs = []
        for ctx_code, tag in ((1, split_labels[0]), (2, split_labels[1])):
            ctx_mask = ctx_arr == ctx_code
            n_ctx    = int(ctx_mask.sum())
            if n_ctx == 0:
                raise ValueError(f'split_by_context selects 0 frames for context '
                                 f'{ctx_code} ({tag}) — cannot build split coupling.')
            for pc_idx in range(n_pcs):
                block  = np.zeros((n_frames, n_lag_bases), dtype=np.float32)
                b_names = []
                for b_idx in range(n_lag_bases):
                    raw = coupling_cols[ctx_mask, pc_idx * n_lag_bases + b_idx]
                    sd  = raw.std()
                    if sd == 0:
                        sd = 1.0
                    block[ctx_mask, b_idx] = (raw - raw.mean()) / sd
                    b_names.append(f'network_pc{pc_idx + 1}_{tag}_lag{b_idx}')
                split_cols.append(block)
                group_specs.append((f'network_pc{pc_idx + 1}_{tag}', b_names))
            print(f'  [Coupling] context {ctx_code} ({tag}): {n_ctx:,}/{n_frames:,} frames '
                  f'({100 * n_ctx / n_frames:.1f}%) — {n_pcs} masked groups')
        coupling_cols = np.hstack(split_cols)

    # ── 5. Append to X and update predictor_info ─────────────────────────────
    X_aug = np.hstack([X, coupling_cols])

    predictor_info_aug = dict(predictor_info)   # shallow copy — don't mutate original
    current_end = X.shape[1]
    for g_idx, (group_name, g_names) in enumerate(group_specs):
        start = current_end + g_idx * n_lag_bases
        predictor_info_aug[group_name] = {
            'start': start,
            'end':   start + n_lag_bases,
            'names': g_names,
        }

    mode = 'shared' if split_by_context is None else 'context-split'
    print(f'  [Coupling] Added {len(group_specs)} {mode} groups × {n_lag_bases} lag bases = '
          f'{coupling_cols.shape[1]} columns  (X: {X.shape[1]} → {X_aug.shape[1]})\n')

    return X_aug, predictor_info_aug, pca


# ---------------------------------------------------------------------------
# Subpopulation coupling predictors (CI / CD network)
# ---------------------------------------------------------------------------

def get_ci_cd_masks(mouse_ID, date,
                    source_model_dir='GLM_uni_results',
                    source_prefix='GLM_uni',
                    resp_threshold=0.1,
                    invariant_thresh=0.1,
                    cd_thresh=0.8):
    """
    Compute context-invariant (CI) and context-dependent (CD) neuron masks
    from a previously fitted base model's per-context fractional deviance.

    The Context Preference Index (CPI / vis_aud_ci) is defined as:
        (FDE_visual - FDE_audio) / (FDE_visual + FDE_audio + 1e-8)
    ranging from -1 (pure audio preference) to +1 (pure visual preference).

    CI neurons : |vis_aud_ci| <  invariant_thresh  AND  responsive
    CD neurons : |vis_aud_ci| >= cd_thresh          AND  responsive

    Parameters
    ----------
    mouse_ID           : str
    date               : str
    source_model_dir   : str  — sub-directory of the labeling model
                               (default 'GLM_uni_results')
    source_prefix      : str  — filename prefix of the labeling model
                               (default 'GLM_uni')
    resp_threshold     : float — minimum test_frac_dev to count as responsive
    invariant_thresh   : float — |CPI| must be below this to be CI (default 0.1)
    cd_thresh          : float — |CPI| must reach this to be CD (default 0.8)

    Returns
    -------
    ci_mask    : ndarray bool (n_neurons,)
    cd_mask    : ndarray bool (n_neurons,)
    vis_aud_ci : ndarray float (n_neurons,)  — raw CPI values
    stats      : dict  — counts and thresholds for logging
    """
    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date,
                             source_model_dir, f'{source_prefix}_full.pkl')
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f'Labeling model not found: {full_path}\n'
            f'Run {source_model_dir} first before fitting the subpop coupling model.')

    with open(full_path, 'rb') as f:
        full_res = pickle.load(f)

    fde_total  = full_res['test_frac_dev']
    n_neurons  = len(fde_total)

    ctx_fde    = full_res.get('test_frac_dev_per_ctx', {})
    fde_visual = ctx_fde.get(1, None)
    fde_audio  = ctx_fde.get(2, None)

    if fde_visual is None:
        print(f'  [CI/CD] WARNING: context 1 (visual) FDE not found for '
              f'{mouse_ID}/{date} — using zeros')
        fde_visual = np.zeros(n_neurons)
    if fde_audio is None:
        print(f'  [CI/CD] WARNING: context 2 (audio) FDE not found for '
              f'{mouse_ID}/{date} — using zeros')
        fde_audio = np.zeros(n_neurons)

    vis_aud_ci = ((fde_visual - fde_audio) /
                  (fde_visual + fde_audio + 1e-8))

    is_resp = fde_total > resp_threshold
    abs_cpi = np.abs(vis_aud_ci)

    ci_mask = is_resp & (abs_cpi <  invariant_thresh)
    cd_mask = is_resp & (abs_cpi >= cd_thresh)

    n_resp = is_resp.sum()
    n_ci   = ci_mask.sum()
    n_cd   = cd_mask.sum()
    print(f'  [CI/CD] {n_neurons} neurons  |  {n_resp} responsive  |  '
          f'CI={n_ci} (|CPI|<{invariant_thresh})  '
          f'CD={n_cd} (|CPI|≥{cd_thresh})')

    stats = dict(
        n_neurons=n_neurons, n_responsive=int(n_resp),
        n_ci=int(n_ci), n_cd=int(n_cd),
        invariant_thresh=invariant_thresh, cd_thresh=cd_thresh,
        resp_threshold=resp_threshold,
    )
    return ci_mask, cd_mask, vis_aud_ci, stats


def _make_pc_lag_features(Y_sub, group_prefix, n_pcs, n_lag_bases, lag_frames,
                           base_col_offset, exclude_col=None):
    """
    Fit PCA on Y_sub, convolve PCs with raised-cosine lag bases, z-score.

    exclude_col : int or None
        LEAVE-ONE-OUT. Column index into Y_sub of a TARGET neuron to drop from
        the PCA input before fitting, so the target never contributes to the
        coupling predictors used to predict it. Essential when Y_sub is a small
        subpopulation (a neuron is ~1/n of its own class's PCA — see
        subpop_coupling_gain_loo). None keeps all columns (self-inclusive).

    Returns
    -------
    coupling_cols  : ndarray float32 (n_frames, actual_pcs * n_lag_bases)
    pred_info_new  : dict  group_name -> {start, end, names}
                     column indices are absolute (offset from base_col_offset)
    pca            : fitted sklearn PCA object
    actual_pcs     : int  — min(n_pcs, source-neuron count after exclusion)
    """
    if exclude_col is not None:
        keep = np.ones(Y_sub.shape[1], dtype=bool)
        keep[exclude_col] = False
        Y_sub = Y_sub[:, keep]
    n_frames, n_sub = Y_sub.shape
    actual_pcs      = min(n_pcs, n_sub)

    pca       = _SklearnPCA(n_components=actual_pcs, whiten=False)
    pc_scores = pca.fit_transform(Y_sub)
    var_exp   = pca.explained_variance_ratio_

    print(f'  [{group_prefix}] PCA on {n_sub} neurons → {actual_pcs} PCs:')
    for k in range(actual_pcs):
        print(f'    PC{k+1}: {var_exp[k]*100:.1f}%')
    print(f'    Total: {var_exp.sum()*100:.1f}%')

    lag_axis  = np.arange(lag_frames, dtype=float)
    centers   = np.linspace(0, lag_frames - 1, n_lag_bases)
    spacing   = centers[1] - centers[0] if n_lag_bases > 1 else float(lag_frames)
    widths    = np.full(n_lag_bases, spacing * 2.0)
    lag_bases = create_cosine_bumps(lag_axis, centers, widths)

    n_cols        = actual_pcs * n_lag_bases
    coupling_cols = np.zeros((n_frames, n_cols), dtype=np.float32)
    col_names     = []

    for pc_idx in range(actual_pcs):
        signal = pc_scores[:, pc_idx]
        for b_idx in range(n_lag_bases):
            basis   = lag_bases[:, b_idx]
            conv    = np.convolve(signal, basis, mode='full')[:n_frames]
            out_col = pc_idx * n_lag_bases + b_idx
            coupling_cols[:, out_col] = conv
            col_names.append(f'{group_prefix}_pc{pc_idx+1}_lag{b_idx}')

    col_std = coupling_cols.std(axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    coupling_cols = (coupling_cols - coupling_cols.mean(axis=0, keepdims=True)) / col_std

    pred_info_new = {}
    for pc_idx in range(actual_pcs):
        grp   = f'{group_prefix}_pc{pc_idx + 1}'
        start = base_col_offset + pc_idx * n_lag_bases
        end   = start + n_lag_bases
        pred_info_new[grp] = {
            'start': start,
            'end':   end,
            'names': col_names[pc_idx * n_lag_bases: pc_idx * n_lag_bases + n_lag_bases],
        }

    return coupling_cols, pred_info_new, pca, actual_pcs


def add_subpop_coupling_predictors(X, Y, predictor_info,
                                    ci_mask, cd_mask,
                                    n_pcs=5, n_lag_bases=4, lag_frames=8):
    """
    Augment the design matrix X with coupling predictors from two subpopulations:
    CI neurons (context-invariant) and CD neurons (context-dependent).

    ⚠️ SELF-PREDICTION LEAKAGE — for the WITHIN-class (backbone/periphery) claim,
    use `subpop_coupling_gain_loo` instead. This function builds ONE shared design
    in which every CI neuron is in the CI PCA and every CD neuron is in the CD PCA,
    so when a lesion of the ci_pc / cd_pc groups is used to score a neuron's own
    coupling, the neuron predicts ITSELF (it is ~1/n_class of its class's PCA). On
    small classes this inflates the within-class diagonal and can fabricate a
    backbone/periphery double dissociation that vanishes under leave-one-out.
    Cross-class terms (CI→CD, CD→CI) are leakage-free here; only the diagonal is
    affected. Kept for the shared-design fit + cross terms; measure within-class
    gains per target with subpop_coupling_gain_loo.

    PCA is fit independently on Y[:, ci_mask] and Y[:, cd_mask].
    Lag-convolved features from both subpopulations are appended in two
    contiguous blocks:
        [X | CI coupling | CD coupling]

    Predictor groups are registered as:
        ci_pc1 … ci_pc{actual_ci_pcs}  (one per CI principal component)
        cd_pc1 … cd_pc{actual_cd_pcs}  (one per CD principal component)

    Parameters
    ----------
    X              : ndarray (n_frames, n_features)
    Y              : ndarray (n_frames, n_neurons)
    predictor_info : dict  — from build_design_matrices / open_glm_stacked_data
    ci_mask        : ndarray bool (n_neurons,)  — True for CI neurons
    cd_mask        : ndarray bool (n_neurons,)  — True for CD neurons
    n_pcs          : int   — PCs to extract from each subpopulation (default 5)
    n_lag_bases    : int   — raised-cosine lag bases per PC (default 4)
    lag_frames     : int   — temporal extent of lag window in frames (default 8)

    Returns
    -------
    X_aug          : ndarray (n_frames, n_features + n_ci_cols + n_cd_cols)
    pred_info_aug  : dict  — original entries + ci_pc1…N + cd_pc1…N
    ci_pca         : fitted sklearn PCA on CI sub-population
    cd_pca         : fitted sklearn PCA on CD sub-population
    ci_col_range   : (int, int)  — (start, end) column indices of ALL CI coupling cols
    cd_col_range   : (int, int)  — (start, end) column indices of ALL CD coupling cols

    Raises
    ------
    ValueError if ci_mask or cd_mask selects zero neurons.
    """
    n_ci = ci_mask.sum()
    n_cd = cd_mask.sum()
    if n_ci == 0:
        raise ValueError('ci_mask selects 0 neurons — cannot fit CI coupling PCA.')
    if n_cd == 0:
        raise ValueError('cd_mask selects 0 neurons — cannot fit CD coupling PCA.')

    if n_ci < n_pcs:
        print(f'  [Subpop] WARNING: only {n_ci} CI neurons < n_pcs={n_pcs}; '
              f'using {n_ci} CI PCs.')
    if n_cd < n_pcs:
        print(f'  [Subpop] WARNING: only {n_cd} CD neurons < n_pcs={n_pcs}; '
              f'using {n_cd} CD PCs.')

    base_end = X.shape[1]

    # ── CI subpopulation ──────────────────────────────────────────────────────
    Y_ci = Y[:, ci_mask]
    ci_cols, ci_pred_info, ci_pca, actual_ci = _make_pc_lag_features(
        Y_ci, group_prefix='ci', n_pcs=n_pcs,
        n_lag_bases=n_lag_bases, lag_frames=lag_frames,
        base_col_offset=base_end,
    )
    ci_col_range = (base_end, base_end + ci_cols.shape[1])

    # ── CD subpopulation ──────────────────────────────────────────────────────
    Y_cd = Y[:, cd_mask]
    cd_start = ci_col_range[1]
    cd_cols, cd_pred_info, cd_pca, actual_cd = _make_pc_lag_features(
        Y_cd, group_prefix='cd', n_pcs=n_pcs,
        n_lag_bases=n_lag_bases, lag_frames=lag_frames,
        base_col_offset=cd_start,
    )
    cd_col_range = (cd_start, cd_start + cd_cols.shape[1])

    # ── Concatenate ───────────────────────────────────────────────────────────
    X_aug = np.hstack([X, ci_cols, cd_cols])

    pred_info_aug = dict(predictor_info)
    pred_info_aug.update(ci_pred_info)
    pred_info_aug.update(cd_pred_info)

    print(f'  [Subpop] Added CI coupling: {actual_ci} PCs × {n_lag_bases} bases = '
          f'{ci_cols.shape[1]} cols  (cols {ci_col_range[0]}–{ci_col_range[1]-1})')
    print(f'  [Subpop] Added CD coupling: {actual_cd} PCs × {n_lag_bases} bases = '
          f'{cd_cols.shape[1]} cols  (cols {cd_col_range[0]}–{cd_col_range[1]-1})')
    print(f'  [Subpop] X: {X.shape[1]} → {X_aug.shape[1]} features\n')

    return X_aug, pred_info_aug, ci_pca, cd_pca, ci_col_range, cd_col_range


def subpop_coupling_gain_loo(X_base, Y, metadata, predictor_info,
                             source_mask, target_indices, base_fde,
                             mouse_ID, date, model_dir, model_prefix,
                             n_pcs=5, n_lag_bases=4, lag_frames=8,
                             contexts=(1, 2), skip_existing=True, verbose=False,
                             fit_kwargs=None):
    """
    LEAVE-ONE-OUT coupling gain from a SOURCE subpopulation to each TARGET neuron.

    This is the leakage-free way to measure subpopulation (backbone/periphery)
    coupling: the coupling predictors for a target neuron must NEVER include that
    neuron's own activity. `add_subpop_coupling_predictors` (shared design) puts
    every neuron in its own class's PCA, so the within-class diagonal (CI→CI,
    CD→CD) is inflated by units predicting themselves — an artifact that can
    manufacture a spurious backbone/periphery double dissociation (verified: on a
    128-unit RNN it fabricated a 5x effect that vanished under leave-one-out).

    For each target t in `target_indices`:
      * coupling PCs are fit on Y[:, source_mask] with t EXCLUDED iff t is in the
        source (off-source targets need no exclusion — they are already clean);
      * a GLM is fit for t alone on [base predictors | those coupling PCs];
      * gain[t] = FDE_t - base_fde[t].

    Per-target fits are cached at model_dir/{model_prefix}_u{t}_full.pkl.

    `fit_kwargs` is passed through to `fit_glm_single_session` (e.g.
    balance_key / balance_levels / equalize_contexts / context_names). It MUST
    match the regime `base_fde` was fitted under: gain is a difference of two
    held-out FDEs, so if the two fits balance trials differently they score
    different test sets and the difference is meaningless. Defaults to None,
    which preserves the original behaviour (balance_key='choice',
    equalize_contexts=False) for existing callers.

    Returns
    -------
    gain : ndarray (n_neurons,)   per-target coupling gain, NaN where not computed
    """
    fit_kwargs = dict(fit_kwargs or {})
    src_idx = np.where(source_mask)[0]
    n_src = len(src_idx)
    gain = np.full(Y.shape[1], np.nan)
    for t in target_indices:
        t = int(t)
        in_src = bool(source_mask[t])
        n_eff = n_src - 1 if in_src else n_src
        k_pcs = min(n_pcs, n_eff)
        if k_pcs < 1:
            gain[t] = 0.0
            continue
        # Source Y (the whole subpopulation), and the target's position within it.
        Y_src = Y[:, source_mask]
        excl = int(np.where(src_idx == t)[0][0]) if in_src else None
        mpre = f'{model_prefix}_u{t}'
        pkl = os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir, f'{mpre}_full.pkl')
        if skip_existing and os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                fde_t = float(np.asarray(pickle.load(f)['test_frac_dev'])[0])
        else:
            Xs, pinfo_s, _ = add_coupling_predictor(
                X_base, Y_src, dict(predictor_info), n_pcs=k_pcs,
                n_lag_bases=n_lag_bases, lag_frames=lag_frames, exclude_col=excl)
            pinfo_split, fgs = prepare_features_for_glm(pinfo_s)
            _, res, _, _ = fit_glm_single_session(
                Xs, Y[:, [t]], metadata, fgs, pinfo_split, mouse_ID, date,
                model_dir=model_dir, model_prefix=mpre, contexts=list(contexts),
                **fit_kwargs)
            fde_t = float(np.asarray(res['test_frac_dev'])[0])
        gain[t] = fde_t - base_fde[t]
        if verbose:
            print(f'    LOO target {t}: gain={gain[t]:+.3f} '
                  f'(source n={n_eff}{" excl-self" if in_src else ""})')
    return gain


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
                            mouse_ID, date,
                            model_dir=None, model_prefix='glm_M1',
                            contexts=None,
                            balance_key='choice', balance_levels=(0, 1),
                            context_names=None, seed=42, split_random_state=0,
                            equalize_contexts=False):
    """
    Fit a GLM on balanced data from the specified contexts.

    Pipeline
    --------
    1. Per-context balanced 70/30 trial splits (stratified by choice)
    2. Combine splits across contexts → single train / test set
    3. Fit GLM_CV with group-lasso regularisation
    4. Select optimal lambda (max CV deviance)
    5. Evaluate on combined test set and per-context held-out sets
    6. Save results to disk

    Parameters
    ----------
    contexts : list of int or None
        Contexts to include in train/test splits.  None → [0, 1, 2].
        e.g. [1, 2] for unisensory-only, [0] for CG-only.
    balance_key : str
        Metadata column whose levels are equalised within each context before
        the 70/30 split, and which (with outcome) forms the stratum label.
        Defaults to 'choice' — the mouse L/R behaviour, unchanged.
        The macaque FEF path passes 'trial_type' (8-way target direction),
        because those sessions have no binary choice: every trial would fall in
        a single 'choice' level, `n_bal` would be 0, and every context would be
        skipped as unbalanced.
    balance_levels : tuple or None
        Levels of `balance_key` to equalise. The default (0, 1) reproduces the
        original hard-coded left/right behaviour exactly, including ignoring the
        -1 "missing" sentinel. Pass None to use every non-negative level present
        (e.g. the 8 directions).
    context_names : dict or None
        Context code -> display name, for logging only. None keeps the mouse
        mapping {0: 'CG', 1: 'visual', 2: 'audio'}.

    Returns
    -------
    model_cv         : glm_class.GLM_CV
    all_results      : dict
    predictor_info   : dict  (unchanged, passed through)
    feature_group_size : ndarray  (unchanged, passed through)
    """
    if context_names is None:
        context_names = {0: 'CG', 1: 'visual', 2: 'audio'}
    contexts_to_use = [0, 1, 2] if contexts is None else list(contexts)
    # seed / split_random_state default to the historical fixed values (42, 0)
    # so existing callers are byte-for-byte unchanged; vary them for multi-seed
    # robustness (the balancing draw and the train/test split respectively).
    rng = np.random.default_rng(seed=seed)

    # ── 1. Per-context balanced splits ────────────────────────────────────
    print('Per-context balanced splits:')
    train_masks = {}
    test_masks  = {}

    def _levels_for(trial_df):
        if balance_levels is None:
            return [v for v in np.unique(trial_df[balance_key].to_numpy()) if v >= 0]
        return list(balance_levels)

    # Cross-context equalisation: cap every (context, level) cell at the GLOBAL
    # minimum count, so each context contributes the SAME number of balanced
    # trials per level. Without this each context is balanced only within itself
    # and the two can differ in total (e.g. 384 saccade vs 336 pursuit), biasing
    # the pooled fit toward the larger task — which matters when FDE is compared
    # ACROSS contexts to form CPI. Default off preserves the mouse behaviour.
    global_n_bal = None
    if equalize_contexts:
        cell_counts = []
        for ctx in contexts_to_use:
            m = metadata['context'] == ctx
            tdf = pd.DataFrame(
                {'trial_id': metadata['trial_id'][m],
                 balance_key: metadata[balance_key][m]}).drop_duplicates('trial_id')
            for lv in _levels_for(tdf):
                cell_counts.append(int((tdf[balance_key] == lv).sum()))
        global_n_bal = min(cell_counts) if cell_counts else 0

    for ctx in contexts_to_use:
        ctx_frame_mask = metadata['context'] == ctx
        meta_ctx = {k: v[ctx_frame_mask] for k, v in metadata.items()}

        trial_df = (
            pd.DataFrame({'trial_id': meta_ctx['trial_id'],
                          'choice':   meta_ctx['choice'],
                          'outcome':  meta_ctx['outcome'],
                          balance_key: meta_ctx[balance_key]})
            .drop_duplicates(subset='trial_id')
        )

        # Equalise the levels of balance_key (keep all trials, no hard outcome
        # balance).  A combined balance_key×outcome stratum label is used only
        # so that train_test_split preserves the natural outcome ratio in both
        # halves — no trials are discarded for this purpose.
        levels = _levels_for(trial_df)

        level_trials = [trial_df[trial_df[balance_key] == lv]['trial_id'].to_numpy()
                        for lv in levels]
        n_bal = min([len(t) for t in level_trials], default=0)
        if equalize_contexts:
            n_bal = min(n_bal, global_n_bal)

        if n_bal == 0:
            print(f'  Skipping context {ctx}: insufficient trials.')
            continue

        balanced_ids = np.concatenate(
            [rng.choice(t, size=n_bal, replace=False) for t in level_trials])

        bal_df = trial_df[trial_df['trial_id'].isin(balanced_ids)].copy()

        # Combined stratum label: stratify 70/30 split by balance_key × outcome
        # so both train and test preserve the natural correct/error proportions.
        bal_df['strat'] = (bal_df[balance_key].astype(str) + '_'
                           + bal_df['outcome'].astype(str))

        # sklearn requires each stratum to appear at least twice for
        # stratified splitting.  If any stratum has only one member, fall
        # back to stratifying on balance_key alone.
        strat_counts = bal_df['strat'].value_counts()
        if (strat_counts < 2).any():
            stratify_col = bal_df[balance_key]
        else:
            stratify_col = bal_df['strat']

        train_trials, test_trials = train_test_split(
            bal_df['trial_id'].to_numpy(),
            test_size=0.3,
            random_state=split_random_state,
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

    skipped = [c for c in contexts_to_use if c not in train_masks]
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
    for ctx in sorted(test_masks.keys()):
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
    for ctx in sorted(test_masks.keys()):
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

    out_dir = model_dir if model_dir is not None else MODEL_1_DIR
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, out_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_prefix}_full.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)

    mb = os.path.getsize(save_path) / 1e6
    print(f'\n  ✓ Results saved → {save_path}  ({mb:.2f} MB)')

    return model_cv, all_results, predictor_info, feature_group_size


# ---------------------------------------------------------------------------
# Lesion analysis
# ---------------------------------------------------------------------------

def run_lesion_models(X, Y, metadata, res, pred_info, mouse_ID, date,
                      skip_existing=True, model_dir=None, model_prefix='glm_M1'):
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
                    {model_prefix}_lesion_{group}.pkl files instead of refitting those
                    groups.  Useful for resuming an interrupted run.
    model_dir     : str or None
                    Sub-directory name under {BASE_DATA_PATH}/{mouse_ID}/{date}/ where
                    results are saved.  Defaults to MODEL_1_DIR ('Model_1_results').
    model_prefix  : str
                    Filename prefix for saved pickle files (e.g. 'glm_M1', 'glm_M2').
                    Defaults to 'glm_M1'.

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

    out_dir = model_dir if model_dir is not None else MODEL_1_DIR
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, out_dir)
    os.makedirs(save_dir, exist_ok=True)

    newly_fitted = set()  # track which groups were actually refit (vs loaded)

    for lesion_group, lesion_info in pred_info.items():
        g_start = lesion_info['start']
        g_end   = lesion_info['end']
        g_width = g_end - g_start

        grp_path = os.path.join(save_dir, f'{model_prefix}_lesion_{lesion_group}.pkl')

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
        grp_path = os.path.join(save_dir, f'{model_prefix}_lesion_{grp}.pkl')
        with open(grp_path, 'wb') as f:
            pickle.dump(grp_results, f)

    # ── Combined all-lesion file ────────────────────────────────────────────
    all_path = os.path.join(save_dir, f'{model_prefix}_lesion_all.pkl')
    with open(all_path, 'wb') as f:
        pickle.dump(lesion_results, f)

    mb = os.path.getsize(all_path) / 1e6
    print(f'\n  ✓ Lesion results saved → {save_dir}/')
    print(f'      {model_prefix}_lesion_{{group}}.pkl  × {len(group_names)}')
    print(f'      {model_prefix}_lesion_all.pkl        ({mb:.2f} MB)')

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
