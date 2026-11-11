#!/usr/bin/env python
"""
GLM_base.py
===========
The FEF v2 PRODUCTION base model (WP6) — the analog of the mouse
`models/unisensory/GLM_uni.py`: pooled Poisson GLM per neuron on both tasks
(saccade context 1 + pursuit context 2), no coupling predictors, followed by
leave-one-group-out lesion refits (the WP7 lesion profiles).

Reuses the per-probe stacked caches built by `scripts/wp6_build_stacked.py`
(the analog of `glm_data_stacked.pkl`). Results:

    {RESULTS_DIR}/fits/{session}_{probe}/GLM_base_results/
        glm_base_full.pkl
        glm_base_lesion_{group}.pkl   x 16 (S) / 15 (Y; no `jump`)

Balancing is the production invariant (trial_type, balance_levels=None,
equalize_contexts=True, seed 42, split 0) and is ASSERTED before every fit.
Y designs are 208 columns with per-monkey kernel windows — every readout must
key groups by NAME via predictor_info, never by position.

Usage (M2 Ultra)
----------------
python models/GLM_base.py --dev                    # S7 subset first (DEV4)
python models/GLM_base.py --workers 6              # full run, 6 x 4 threads
python models/GLM_base.py --monkey S               # primary arm only
python models/GLM_base.py --stages base            # base fits, no lesions
python models/GLM_base.py --probe-key 0136a_lFEF   # one probe, in-process
"""

import os
import sys

_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

from fef.fit import model_runner   # noqa: E402

# ==============================================================================
# Model identity
# ==============================================================================
MODEL_CFG = dict(
    model_name='GLM_base',
    model_tag='base',
    model_dir='GLM_base_results',
    model_prefix='glm_base',
    contexts=[1, 2],              # saccade (1) + pursuit (2), pooled fit
    include_coupling=False,
    coupling_params=dict(),
)
# ==============================================================================

if __name__ == '__main__':
    model_runner.main(MODEL_CFG, os.path.abspath(__file__))
