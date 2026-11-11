#!/usr/bin/env python
"""
GLM_subpop_coupling_loo.py
==========================
LEAKAGE-FREE backbone/periphery (CI vs CD) subpopulation coupling for REAL data.

Replaces the within-class part of GLM_all_ctx_cpl_subpop.py, whose shared-design
`ci_network` / `cd_network` compound lesions put every neuron in its own class's
coupling PCA — so a CI target's ci_network ΔFDE (and a CD target's cd_network
ΔFDE) is inflated by the neuron partly predicting ITSELF (~1/n_class of its class
PCA). On a 128-unit RNN that exact leakage fabricated a 5x backbone/periphery
"double dissociation" that COLLAPSED (CI 5.4x->0.82x, CD 4.9x->1.36x) once each
target was excluded from its own class PCA.

This driver computes, per target neuron, the LEAVE-ONE-OUT coupling gain from the
CI population and from the CD population (target always excluded from its own
class's PCA), using the shared, RNN-validated
`core.session_pipeline.subpop_coupling_gain_loo`. The 2x2 block matrix of mean
gains is the leakage-free backbone/periphery test.

Cost: one single-neuron GLM refit per (target, source) — O(n_responsive) fits per
session per source population. Fits cache under the model_dir (skip_existing).

Usage
-----
python GLM_subpop_coupling_loo.py --mouse MOUSE --date DATE
    [--label-model GLM_uni_results GLM_uni] [--n-pcs 5]

NOTE: requires the session's glm_data_stacked.pkl and a fitted label model
(GLM_uni) on BASE_DATA_PATH. Validated on the RNN readout; run on your data to
regenerate the corrected Fig-4 backbone/periphery numbers.
"""
import argparse
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.session_pipeline as sp
from core.session_pipeline import (
    open_glm_stacked_data, get_ci_cd_masks, subpop_coupling_gain_loo,
    BASE_DATA_PATH,
)

MODEL_DIR = 'GLM_subpop_coupling_loo_results'
CONTEXTS = (1, 2)
RESP_THRESHOLD, INVARIANT_THRESH, CD_THRESH = 0.1, 0.1, 0.8


def _filter_contexts(X, Y, metadata, contexts=CONTEXTS):
    m = np.isin(metadata['context'], list(contexts))
    return X[m], Y[m], {k: v[m] for k, v in metadata.items()}


def run_session(mouse_ID, date, label_model=('GLM_uni_results', 'GLM_uni'),
                n_pcs=5, n_lag_bases=4, lag_frames=8, skip_existing=True):
    X, Y, metadata, _, pinfo, _ = open_glm_stacked_data(mouse_ID, date, print_summary=False)
    X, Y, metadata = _filter_contexts(X, Y, metadata)

    ci_mask, cd_mask, cpi, stats = get_ci_cd_masks(
        mouse_ID, date, source_model_dir=label_model[0], source_prefix=label_model[1],
        resp_threshold=RESP_THRESHOLD, invariant_thresh=INVARIANT_THRESH, cd_thresh=CD_THRESH)

    with open(os.path.join(BASE_DATA_PATH, mouse_ID, date,
                           label_model[0], f'{label_model[1]}_full.pkl'), 'rb') as f:
        base_fde = np.asarray(pickle.load(f)['test_frac_dev'], float)

    resp = base_fde > RESP_THRESHOLD
    targets = np.where(resp)[0]
    if ci_mask.sum() < 2 or cd_mask.sum() < 2:
        raise ValueError(f'need >=2 CI and >=2 CD neurons (got {ci_mask.sum()}/{cd_mask.sum()})')

    # Per-target leave-one-out coupling gain FROM each source population.
    gain_from_ci = subpop_coupling_gain_loo(
        X, Y, metadata, pinfo, source_mask=ci_mask, target_indices=targets, base_fde=base_fde,
        mouse_ID=mouse_ID, date=date, model_dir=MODEL_DIR, model_prefix='loo_ci',
        n_pcs=n_pcs, n_lag_bases=n_lag_bases, lag_frames=lag_frames,
        contexts=CONTEXTS, skip_existing=skip_existing)
    gain_from_cd = subpop_coupling_gain_loo(
        X, Y, metadata, pinfo, source_mask=cd_mask, target_indices=targets, base_fde=base_fde,
        mouse_ID=mouse_ID, date=date, model_dir=MODEL_DIR, model_prefix='loo_cd',
        n_pcs=n_pcs, n_lag_bases=n_lag_bases, lag_frames=lag_frames,
        contexts=CONTEXTS, skip_existing=skip_existing)

    def _m(a, m):
        return float(np.nanmean(a[m])) if m.any() else float('nan')
    block = np.array([[_m(gain_from_ci, ci_mask), _m(gain_from_cd, ci_mask)],
                      [_m(gain_from_ci, cd_mask), _m(gain_from_cd, cd_mask)]])
    out = {
        'block_matrix': block,           # rows: [CI targets, CD targets]; cols: [CI src, CD src]
        'gain_from_ci': gain_from_ci, 'gain_from_cd': gain_from_cd,
        'ci_mask': ci_mask, 'cd_mask': cd_mask, 'cpi': cpi, 'stats': stats,
        'diag_ratio_ci': block[0, 0] / block[0, 1] if block[0, 1] > 0 else float('inf'),
        'diag_ratio_cd': block[1, 1] / block[1, 0] if block[1, 0] > 0 else float('inf'),
        'leave_target_out': True,
    }
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'block_coupling_loo.pkl'), 'wb') as f:
        pickle.dump(out, f)
    print(f'\n[LOO backbone/periphery] {mouse_ID}/{date}  (CI={ci_mask.sum()} CD={cd_mask.sum()})')
    print(f'  CI targets: CI-src {block[0,0]:.3f}  CD-src {block[0,1]:.3f}  (ratio {out["diag_ratio_ci"]:.2f}x)')
    print(f'  CD targets: CI-src {block[1,0]:.3f}  CD-src {block[1,1]:.3f}  (ratio {out["diag_ratio_cd"]:.2f}x)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mouse', required=True)
    ap.add_argument('--date', required=True)
    ap.add_argument('--label-model', nargs=2, default=['GLM_uni_results', 'GLM_uni'])
    ap.add_argument('--n-pcs', type=int, default=5)
    ap.add_argument('--no-skip', action='store_true')
    args = ap.parse_args()
    run_session(args.mouse, args.date, label_model=tuple(args.label_model),
                n_pcs=args.n_pcs, skip_existing=not args.no_skip)


if __name__ == '__main__':
    main()
