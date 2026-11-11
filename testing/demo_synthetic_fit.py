#!/usr/bin/env python
"""
demo_synthetic_fit.py
=====================
End-to-end check that needs NO real data.

1. Makes a synthetic session (three contexts, balanced trials).
2. Builds the design matrix exactly as for a real session.
3. Simulates neurons that each respond to ONE predictor group.
4. Fits the GLM and prints how well each simulated neuron is predicted.

Results go to a temporary folder, never to your data path.

    python testing/demo_synthetic_fit.py
    python testing/demo_synthetic_fit.py --trials 60     # faster
"""
import argparse
import os
import sys
import tempfile

# The demo must not write into a real data folder: point the pipeline at a
# temporary one BEFORE core.session_pipeline is imported.
os.environ['GLM_BASE_DATA_PATH'] = tempfile.mkdtemp(prefix='glm_demo_')

GLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GLM_DIR)

import numpy as np                                                # noqa: E402
from testing.synthetic_session import generate_synthetic_session  # noqa: E402
from testing.simulated_neurons import SimulatedNeurons            # noqa: E402
from core.session_pipeline import (                               # noqa: E402
    BASE_DATA_PATH, generate_predictors, stack_design_matrices,
    prepare_features_for_glm, fit_glm_single_session,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=100,
                    help='trials per context (default 100)')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    glm_data_dict = generate_synthetic_session(
        n_trials_per_context=a.trials, seed=a.seed)
    design_matrices, _, pred_info_raw = generate_predictors(
        glm_data_dict, include_velocity=False)
    X, _, metadata, _ = stack_design_matrices(
        glm_data_dict, design_matrices, pred_info_raw, neural_metric='deconv')
    pred_info, feature_group_size = prepare_features_for_glm(pred_info_raw)

    sim = SimulatedNeurons(pred_info, metadata, X, baseline_rate=0.004,
                           seed=a.seed)
    sim_data = sim.build_all_neurons()
    Y = sim_data['Y_sim'] * 10

    _, res, _, _ = fit_glm_single_session(
        X, Y, metadata, feature_group_size, pred_info,
        mouse_ID='simulated', date='demo',
        model_dir='GLM_demo_results', model_prefix='glm_demo')

    fde = np.asarray(res['test_frac_dev'])
    print('\n' + '=' * 60)
    print(f'Demo finished. {Y.shape[1]} simulated neurons, {X.shape[1]} predictors.')
    print(f'Held-out fraction of deviance explained: '
          f'median {np.median(fde):.2f}, range {fde.min():.2f} to {fde.max():.2f}')
    print(f'Results folder: {os.path.join(BASE_DATA_PATH, "simulated", "demo")}')


if __name__ == '__main__':
    main()
