#!/usr/bin/env python
"""
GLM_base_coupling.py
====================
The FEF v2 coupling model (WP8) — the analog of the mouse
`models/unisensory/GLM_uni_coupling.py`: the base design PLUS functional
coupling predictors (population PCs x causal lag bases; v2 lag budget
12 frames x 40 ms ~ 480 ms, matching the mouse's ~0.5 s).

Coupling gain per neuron = FDE(this model) - FDE(GLM_base), computed
downstream on matched seeds/splits (both models assert the same production
balancing, so the comparison is valid by construction).

CAVEAT (config.COUPLING_EXCLUDE_SELF — read before quoting per-neuron gains):
this is the POPULATION model; the PCs include every neuron, so a
high-variance neuron partly predicts itself and its own gain is inflated —
exactly the v1 Figure-4d confound direction. Population-level readouts are
fine; PER-NEURON gain claims must come from the WP9 leave-one-out subpop
harness (exclude_col per target), not from this model.

Usage: identical to GLM_base.py (same driver, same caches).
"""

import os
import sys

_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

from fef import config as C        # noqa: E402
from fef.fit import model_runner   # noqa: E402

# ==============================================================================
# Model identity
# ==============================================================================
MODEL_CFG = dict(
    model_name='GLM_base_coupling',
    model_tag='base_cpl',
    model_dir='GLM_base_coupling_results',
    model_prefix='glm_base_cpl',
    contexts=[1, 2],
    include_coupling=True,
    coupling_params=dict(n_pcs=C.COUPLING_N_PCS,
                         n_lag_bases=C.COUPLING_N_LAG_BASES,
                         lag_frames=C.COUPLING_LAG_FRAMES),
)
# ==============================================================================

if __name__ == '__main__':
    model_runner.main(MODEL_CFG, os.path.abspath(__file__))
