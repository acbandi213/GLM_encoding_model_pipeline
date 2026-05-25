#!/usr/bin/env python
"""
run_model_comparison.py
=======================
Quick 3-way model comparison on a **single example session**.

Three configurations are fit back-to-back (full model only, no lesions):

  Config A  ``base``        — Model 1 behavioural groups
                               (position_only, audio_context, audio_stim,
                                visual_context, visual_stim, velocity,
                                choice_spatial, turn_onset, outcome)
  Config B  ``ctx_choice``  — Base + context × choice interaction
                               (+ visual_choice, audio_choice)
  Config C  ``coupling``    — Base + functional coupling PCs
                               (+ network_pc1 … network_pc{n_pcs})

Outputs
-------
Results are saved to:
    {data_dir}/comparison/glm_{config_name}_full.pkl

A 4-panel comparison figure is displayed showing:
    Panel 1: CDF of fractional deviance explained (all 3 configs)
    Panel 2: Violin / strip plot of FDE distributions
    Panel 3: Scatter — Base vs Ctx×Choice FDE
    Panel 4: Scatter — Base vs Coupling FDE

Usage
-----
Edit MOUSE_ID / DATE / COUPLING_PARAMS at the top, then:
    python run_model_comparison.py
"""

import gc
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
GLM_DIR        = os.path.dirname(os.path.abspath(__file__))
ICLOUD_GLM_DIR = ('/Users/akhilbandi/Library/Mobile Documents/'
                  'com~apple~CloudDocs/Documents/Portfolio/code_projects/'
                  'Context Paper Code/GLM_encoding_model')
for _p in (GLM_DIR, ICLOUD_GLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.ndimage import gaussian_filter1d
from core.session_pipeline import (
    BASE_DATA_PATH,
    Y_SCALE,
    Y_SMOOTH_SIGMA,
    extract_data,
    stack_design_matrices,
    add_coupling_predictor,
    prepare_features_for_glm,
    fit_glm_single_session,
)
from core.predictor_builder import GLMPredictorBuilder

# ==============================================================================
# ── SESSION TO COMPARE ────────────────────────────────────────────────────────
# ==============================================================================
MOUSE_ID = 'KO-6-00'
DATE     = '2026-03-12'

# ── Coupling config (shared by Config C) ──────────────────────────────────────
COUPLING_PARAMS = dict(
    n_pcs       = 5,
    n_lag_bases = 4,
    lag_frames  = 8,
)

# ── Neural metric ─────────────────────────────────────────────────────────────
NEURAL_METRIC = 'deconv'

# ── Minimum FDE threshold for "responsive" neurons ────────────────────────────
FDE_THRESHOLD = 0.1

# ── Output directory (under mouse/date/) ─────────────────────────────────────
COMPARISON_SUBDIR = 'comparison'

# ==============================================================================
# ── Shared builder parameters (basis counts, etc.) ───────────────────────────
# ==============================================================================
_SHARED_PARAMS = dict(
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

    # ── Model 1 group structure ───────────────────────────────────────────────
    n_pos_bases    = 10,
    n_place_fields = 30,
    n_vel_bases    = 3,
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

    # All extended / combined flags off — Config B overrides include_context_choice
    include_combined_stimulus = False,
    include_combined_choice   = False,
    include_stim_temporal     = False,
    include_pre_turn          = False,
    include_context_choice    = False,
)

# ==============================================================================
# ── Three configurations ──────────────────────────────────────────────────────
# ==============================================================================
CONFIGS = {
    'base': {
        'label':            'Model 1 (base)',
        'color':            '#2196F3',
        'builder_params':   {**_SHARED_PARAMS},
        'include_coupling': False,
    },
    'ctx_choice': {
        'label':            'M1 + Ctx×Choice',
        'color':            '#4CAF50',
        'builder_params':   {**_SHARED_PARAMS, 'include_context_choice': True},
        'include_coupling': False,
    },
    'coupling': {
        'label':            'M1 + Coupling',
        'color':            '#FF5722',
        'builder_params':   {**_SHARED_PARAMS},
        'include_coupling': True,
    },
}


# ==============================================================================
# ── Helpers ───────────────────────────────────────────────────────────────────
# ==============================================================================

def _save_dir(mouse_ID, date):
    d = os.path.join(BASE_DATA_PATH, mouse_ID, date, COMPARISON_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _result_path(mouse_ID, date, config_name):
    return os.path.join(_save_dir(mouse_ID, date),
                        f'glm_{config_name}_full.pkl')


def run_config(config_name, config, glm_data_dict, mouse_ID, date):
    """Fit one model configuration and return the results dict."""
    result_path = _result_path(mouse_ID, date, config_name)
    if os.path.exists(result_path):
        print(f'\n  [SKIP] {config_name} — result already exists')
        with open(result_path, 'rb') as f:
            return pickle.load(f)

    print(f'\n{"="*60}')
    print(f'  Running config: {config["label"]}')
    print(f'{"="*60}')

    # Build per-trial design matrices
    builder = GLMPredictorBuilder(**config['builder_params'])
    design_matrices, predictor_names, predictor_info = \
        builder.build_design_matrices(glm_data_dict)

    n_groups = len(predictor_info)
    n_cols   = sum(i['end'] - i['start'] for i in predictor_info.values())
    print(f'  Groups: {n_groups}  |  Predictor columns: {n_cols}')

    # Stack trials → X, Y
    X, Y, metadata, group_indices = stack_design_matrices(
        glm_data_dict, design_matrices, predictor_info,
        neural_metric=NEURAL_METRIC,
    )

    # Apply the same Y scaling + smoothing used in run_sessions_model.py
    Y = Y * Y_SCALE
    if Y_SMOOTH_SIGMA > 0:
        Y = gaussian_filter1d(Y, sigma=Y_SMOOTH_SIGMA, axis=0)
        Y = np.clip(Y, 0.0, None)

    print(f'  X: {X.shape}  Y: {Y.shape}  (Y_SCALE={Y_SCALE}, smooth_sigma={Y_SMOOTH_SIGMA})')

    # Add coupling if needed
    if config['include_coupling']:
        X, predictor_info, _ = add_coupling_predictor(
            X, Y, predictor_info,
            n_pcs=COUPLING_PARAMS['n_pcs'],
            n_lag_bases=COUPLING_PARAMS['n_lag_bases'],
            lag_frames=COUPLING_PARAMS['lag_frames'],
        )
        print(f'  After coupling:  X: {X.shape}  '
              f'groups: {len(predictor_info)}')

    # Prepare for GLM (split compound groups if any)
    predictor_info_split, feature_group_size = \
        prepare_features_for_glm(predictor_info)

    # Fit
    _, results, _, _ = fit_glm_single_session(
        X, Y, metadata, feature_group_size, predictor_info_split,
        mouse_ID=mouse_ID, date=date,
        model_dir=COMPARISON_SUBDIR,
        model_prefix=f'glm_{config_name}',
    )

    with open(result_path, 'rb') as f:
        saved = pickle.load(f)

    del X, Y, design_matrices, builder
    gc.collect()

    return saved


def extract_fde(results):
    """Return per-neuron fractional deviance explained from a full-model results dict."""
    return np.asarray(results['test_frac_dev'])


# ==============================================================================
# ── Main ──────────────────────────────────────────────────────────────────────
# ==============================================================================

def main():
    print(f'\nModel comparison: {MOUSE_ID} / {DATE}')
    print(f'Configs: {list(CONFIGS.keys())}')
    print(f'Output: {_save_dir(MOUSE_ID, DATE)}\n')

    # Load session data once
    print('Loading session data…')
    glm_data_dict = extract_data(MOUSE_ID, DATE)
    print(f'  Trials: {len(glm_data_dict)}')

    # Run all configs
    all_results = {}
    for name, cfg in CONFIGS.items():
        all_results[name] = run_config(name, cfg, glm_data_dict,
                                       MOUSE_ID, DATE)

    del glm_data_dict
    gc.collect()

    # ── Extract FDE vectors ───────────────────────────────────────────────────
    fde = {name: extract_fde(res) for name, res in all_results.items()}
    n_neurons = len(fde['base'])
    n_resp    = {name: int(np.sum(v >= FDE_THRESHOLD)) for name, v in fde.items()}

    # Responsive mask from base model
    resp_mask = fde['base'] >= FDE_THRESHOLD

    print('\n' + '─'*50)
    print(f'  Total neurons:      {n_neurons}')
    for name, cfg in CONFIGS.items():
        med = float(np.nanmedian(fde[name][resp_mask]))
        print(f'  {cfg["label"]:30s}  '
              f'n_resp={n_resp[name]:4d}  '
              f'median FDE={med:.4f}')
    print('─'*50 + '\n')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f'Model Comparison — {MOUSE_ID} / {DATE}\n'
        f'(responsive = FDE ≥ {FDE_THRESHOLD}, n={n_resp["base"]})',
        fontsize=13, fontweight='bold',
    )

    names  = list(CONFIGS.keys())
    labels = [CONFIGS[n]['label']  for n in names]
    colors = [CONFIGS[n]['color']  for n in names]

    # Panel 1: CDF
    ax = axes[0]
    for name, lbl, col in zip(names, labels, colors):
        x = np.sort(fde[name][resp_mask])
        y = np.linspace(0, 1, len(x))
        ax.plot(x, y, color=col, lw=2, label=lbl)
    ax.axvline(FDE_THRESHOLD, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Frac. deviance explained', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title('CDF', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, None)

    # Panel 2: Violin
    ax = axes[1]
    vparts = ax.violinplot(
        [fde[n][resp_mask] for n in names],
        positions=range(len(names)),
        showmedians=True, showextrema=False,
    )
    for pc, col in zip(vparts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    vparts['cmedians'].set_color('black')
    vparts['cmedians'].set_linewidth(2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([CONFIGS[n]['label'] for n in names],
                       rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Frac. deviance explained', fontsize=11)
    ax.set_title('Distribution', fontsize=12)

    # Panel 3: Scatter — Base vs Ctx×Choice
    ax = axes[2]
    x3 = fde['base'][resp_mask]
    y3 = fde['ctx_choice'][resp_mask]
    delta3 = y3 - x3
    ax.scatter(x3, y3, s=6, alpha=0.4,
               c=delta3, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    lim = max(x3.max(), y3.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Base FDE', fontsize=11)
    ax.set_ylabel('Ctx×Choice FDE', fontsize=11)
    gain3 = float(np.nanmedian(delta3))
    ax.set_title(f'Base → Ctx×Choice\nmedian gain={gain3:+.4f}', fontsize=11)

    # Panel 4: Scatter — Base vs Coupling
    ax = axes[3]
    x4 = fde['base'][resp_mask]
    y4 = fde['coupling'][resp_mask]
    delta4 = y4 - x4
    sc = ax.scatter(x4, y4, s=6, alpha=0.4,
                    c=delta4, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    lim = max(x4.max(), y4.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Base FDE', fontsize=11)
    ax.set_ylabel('Coupling FDE', fontsize=11)
    gain4 = float(np.nanmedian(delta4))
    ax.set_title(f'Base → Coupling\nmedian gain={gain4:+.4f}', fontsize=11)
    fig.colorbar(sc, ax=ax, label='ΔFDE', fraction=0.046, pad=0.04)

    plt.tight_layout()

    fig_path = os.path.join(_save_dir(MOUSE_ID, DATE), 'model_comparison.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {fig_path}')
    plt.show()


if __name__ == '__main__':
    main()
