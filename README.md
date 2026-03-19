# GLM Encoding Model Pipeline

A Poisson generalized linear model (GLM) pipeline for fitting neural encoding models to calcium imaging data from head-fixed mice navigating a virtual reality T-maze. The model decomposes single-neuron activity into contributions from nine behavioural predictor groups using raised-cosine basis functions and elastic-net / group-lasso regularisation.

---

## Overview

Each neuron's activity is modelled as:

$$\hat{y} = \exp(X \mathbf{w} + w_0)$$

where $X$ is a design matrix of behavioural predictors (position, context, stimulus identity, choice, velocity, turn onset, and outcome), and $\mathbf{w}$ is a sparse weight vector fit by cross-validated penalised maximum likelihood.

The unique contribution of each predictor group is quantified via **lesion models**: the full model is refit with one group removed at a time, and the drop in fractional deviance explained ($\Delta \text{FDE}$) is recorded per neuron.

---

## Repository Structure

```
GLM_encoding_model/
├── glm_class.py           # Core GLM class (TensorFlow-based)
├── predictor_builder.py   # Raised-cosine basis functions & design matrix builder
├── session_pipeline.py    # Per-session pipeline: data I/O, fitting, lesion models
├── run_sessions.py        # Command-line runner for all sessions
├── simulated_neurons.py   # Synthetic neurons with known ground-truth weights
└── synthetic_session.py   # Fully synthetic trial generator (no real data needed)
```

---

## File Descriptions

### `glm_class.py`

Core `GLM` class implementing regularised GLM fitting via gradient descent (TensorFlow). The base code is from Tseng et al 2022 (Neuron). Only additions are for running this code efficiently on Apple Silicon (M2) 

**Key features:**
- Supports Poisson (`exp` + `poisson`), Gaussian (`linear` + `gaussian`), and logistic (`sigmoid` + `binomial`) models
- Regularisation: elastic-net (L1+L2) or group lasso
- Optional smoothness penalty (finite-difference prior on weight vectors)
- Warm-started lambda path: fits a series of regularisation strengths from strongest to weakest, using each solution to initialise the next
- Adam and SGD+momentum optimisers
- Model selection by validation-set fractional deviance explained
- `GLM_CV` subclass: full k-fold cross-validated fitting with balanced train/val/test splitting by trial context

**Main classes:**

| Class | Description |
|---|---|
| `GLM` | Single train/val split. `.fit()`, `.select_model()`, `.predict()`, `.evaluate()` |
| `GLM_CV` | Cross-validated wrapper. Handles k-fold splitting, aggregates test predictions, saves results |

**Standalone utilities:**

| Function | Description |
|---|---|
| `make_prediction(model, X)` | Generate predictions from a fitted model |
| `deviance(Y, Y_pred, loss_type)` | Compute null and model deviance |

---

### `predictor_builder.py`

Constructs the GLM design matrix from raw behavioural and movement data.

**Basis functions:**
- `create_cosine_bumps(x, centers, widths)` — raised cosine basis (Harvey Lab convention): overlapping bumps that tile a 1D variable (position or time)

**`GLMPredictorBuilder` class** builds nine predictor groups:

| Group | Type | Predictors | Description |
|---|---|---|---|
| `position_only` | Spatial | 10 | Position place fields, proximal zone only (0–350 cm). Restricted to avoid collinearity with `choice_spatial` in the distal maze. |
| `audio_context` | Spatial | 10 | Position fields active on all audio & congruent trials (sum coding) |
| `audio_stim` | Spatial | 10 | Contrast-coded position fields: +left tone, −right tone |
| `visual_context` | Spatial | 10 | Position fields active on all visual & congruent trials (sum coding) |
| `visual_stim` | Spatial | 10 | Contrast-coded position fields: +0°, −90° |
| `velocity` | Spatial | 20 | Velocity × position (4 directions × 5 bases) |
| `choice_spatial` | Spatial | 60 | Distal place fields split by L/R choice (30 bases × 2) |
| `turn_onset` | Temporal | 8 | Turn-aligned temporal bases (4 forward × L/R) |
| `outcome` | Temporal | 12 | Reward/ITI-aligned temporal bases (6 × correct/error) |

Total: **150 predictors** (with velocity) or **130** (without).

Congruent trials (context 0) activate both `audio_context` and `visual_context` simultaneously, allowing the model to separate shared context signals from context-specific tuning.

---

### `session_pipeline.py`

Orchestrates the full per-session pipeline. Functions are designed to be called in sequence:

```
1. extract_data()             — load glm_data_dict from compressed pickle
2. generate_predictors()      — build per-trial design matrices
3. stack_design_matrices()    — concatenate trials → X / Y matrices
4. save_glm_data()            — save stacked data to disk
5. open_glm_stacked_data()    — reload, apply Y scaling + smoothing
6. prepare_features_for_glm() — split predictor_info into train/test subsets
7. fit_glm_single_session()   — fit full GLM with cross-validation
8. run_lesion_models()        — refit with each predictor group removed
```

**Key constants (top of file):**

| Constant | Default | Description |
|---|---|---|
| `BASE_DATA_PATH` | `/Volumes/Akhil Data/...` | Root data directory |
| `Y_SCALE` | `10` | Multiplier applied to raw deconvolved traces |
| `Y_SMOOTH_SIGMA` | `5` frames | Gaussian smoothing σ applied to Y before fitting |
| `MODEL_1_DIR` | `Model_1_results` | Subdirectory for saving model outputs |

**Output files per session** (saved under `{BASE_DATA_PATH}/{mouse_ID}/{date}/`):

| File | Contents |
|---|---|
| `glm_data_stacked.pkl` | Stacked X, Y, metadata, predictor info |
| `Model_1_results/glm_M1_full.pkl` | Full model results (weights, test FDE, R², per-context R²) |
| `Model_1_results/glm_M1_lesion_{group}.pkl` | Per-group lesion model results (one file per group) |
| `Model_1_results/glm_M1_lesion_all.pkl` | Aggregated lesion results across all groups |

**Granular resume logic in `run_lesion_models`:** if `skip_existing=True`, each per-group `.pkl` is checked before fitting. Already-complete groups are loaded from cache; only missing groups are refit.

---

### `run_sessions.py`

Command-line runner that processes all sessions sequentially with robust memory management and a persistent master log.

**Usage:**

```bash
# Run all sessions (skip fully completed ones)
python run_sessions.py

# Force re-run of everything
python run_sessions.py --no-skip

# Use a different neural metric
python run_sessions.py --metric dff
```

**Three-way skip logic per session:**

| Condition | Action |
|---|---|
| Full model + all lesion files present | Skip entirely |
| Full model done, some lesion files missing | Load full results, refit only missing lesion groups |
| Full model missing | Run entire pipeline from scratch |

**Memory management:** all large arrays (`X`, `Y`, `model_cv`, design matrices) are explicitly deleted and `gc.collect()` is called between sessions and after each lesion group fit.

**Master log:** a timestamped `.log` file is written to `logs/run_YYYYMMDD_HHMMSS.log` and updated after every session:

```bash
tail -f logs/run_YYYYMMDD_HHMMSS.log
```

---

### `simulated_neurons.py`

`SimulatedNeurons` class — generates synthetic neural responses with known ground-truth weights for validating the GLM pipeline end-to-end.

Each simulated neuron has non-zero weights in **exactly one** predictor group, providing unambiguous ground truth for the lesion analysis.

**Neuron catalogue (14 neurons):**

| ID | Group | Description |
|---|---|---|
| N01 | `position_only` | Gaussian place field at y ≈ 175 cm, all trial types |
| N02 | `audio_context` | Spatial bump active on all audio + congruent trials |
| N03 | `audio_stim` | Contrast-coded: selective for left tone (Audio_Stim = 0°) |
| N04 | `audio_stim` | Contrast-coded: selective for right tone (Audio_Stim = 90°) |
| N05 | `visual_context` | Spatial bump active on all visual + congruent trials |
| N06 | `visual_stim` | Contrast-coded: selective for 0° grating |
| N07 | `visual_stim` | Contrast-coded: selective for 90° grating |
| N08 | `choice_spatial` | Distal place fields, left-choice trials |
| N09 | `choice_spatial` | Distal place fields, right-choice trials |
| N10 | `turn_onset` | Temporal: fires at left turn onset |
| N11 | `turn_onset` | Temporal: fires at right turn onset |
| N12 | `outcome` | Temporal: fires after reward (correct trials) |
| N13 | `outcome` | Temporal: fires after error (unrewarded trials) |
| N14 | null | No tuning; pure Poisson noise |

**Usage:**
```python
from simulated_neurons import SimulatedNeurons
from session_pipeline import prepare_features_for_glm, fit_glm_single_session

sim      = SimulatedNeurons(pred_info, metadata, X, baseline_rate=0.004, seed=42)
sim_data = sim.build_all_neurons()
Y_sim    = sim_data['Y_sim']   # (n_frames, 14) — drop into fit_glm_single_session

fit_glm_single_session(X, Y_sim * 10, metadata,
                       feature_group_size, pred_info,
                       mouse_ID='simulated', date='v1')
```

---

### `synthetic_session.py`

Generates a fully synthetic `glm_data_dict` with tightly controlled trial structure — no real behavioural data required.

**Why use this?** Real sessions have confounds (trial-length variability, kinematic asymmetries, L/R imbalances) that corrupt ground-truth validation. The synthetic session eliminates all of these:
- Every trial has identical kinematics (linear y-ramp, fixed turn at frame 125)
- Exact 50/50 L/R choice and stimulus balance
- Fixed turn onset, reward onset, and ITI timing across all trials

**Trial structure (200 frames per trial):**

| Phase | Frames | Description |
|---|---|---|
| Approach | 0–124 | Linear y-ramp 0 → 504 cm, constant forward velocity |
| Turn onset | 125 | y = 510 cm, lateral velocity ±30 cm/frame |
| Post-turn | 125–199 | Fixed y position, lateral movement |
| Reward/ITI | 160 | Outcome temporal bases start here |

**Usage:**
```python
from synthetic_session import generate_synthetic_session
from session_pipeline import generate_predictors, stack_design_matrices, prepare_features_for_glm

glm_data_dict = generate_synthetic_session(n_trials_per_context=100, seed=42)
design_matrices, pred_names, pred_info_raw = generate_predictors(
    glm_data_dict, include_velocity=False
)
X, Y_placeholder, metadata, group_indices = stack_design_matrices(
    glm_data_dict, design_matrices, pred_info_raw
)
pred_info, feature_group_size = prepare_features_for_glm(pred_info_raw)
```

---

## Quick Start

### Fit a single real session

```python
from session_pipeline import (
    extract_data, generate_predictors, stack_design_matrices,
    save_glm_data, open_glm_stacked_data, prepare_features_for_glm,
    fit_glm_single_session, run_lesion_models,
)

mouse_ID, date = 'KO-5-1L', '2026-03-05'

glm_data_dict                            = extract_data(mouse_ID, date)
design_matrices, pred_names, pred_info   = generate_predictors(glm_data_dict)
X, Y, metadata, group_indices            = stack_design_matrices(glm_data_dict, design_matrices, pred_info)
save_glm_data(X, Y, design_matrices, pred_names, pred_info, metadata, group_indices, mouse_ID, date)

X, Y, metadata, pred_names, pred_info, _ = open_glm_stacked_data(mouse_ID, date)
pred_info_split, feature_group_size      = prepare_features_for_glm(pred_info)

model_cv, results, _, _                  = fit_glm_single_session(
    X, Y, metadata, feature_group_size, pred_info_split, mouse_ID, date
)
lesion_results                           = run_lesion_models(
    X, Y, metadata, results, pred_info_split, mouse_ID=mouse_ID, date=date
)
```

### Validate with simulated data

```python
from synthetic_session import generate_synthetic_session
from simulated_neurons import SimulatedNeurons
from session_pipeline import (
    generate_predictors, stack_design_matrices, prepare_features_for_glm,
    fit_glm_single_session, run_lesion_models,
)

glm_data_dict                           = generate_synthetic_session(n_trials_per_context=100, seed=42)
design_matrices, pred_names, pred_info  = generate_predictors(glm_data_dict, include_velocity=False)
X, _, metadata, _                       = stack_design_matrices(glm_data_dict, design_matrices, pred_info)
pred_info_split, feature_group_size     = prepare_features_for_glm(pred_info)

sim      = SimulatedNeurons(pred_info_split, metadata, X, baseline_rate=0.004, seed=42)
sim_data = sim.build_all_neurons()

fit_glm_single_session(
    X, sim_data['Y_sim'] * 10, metadata,
    feature_group_size, pred_info_split,
    mouse_ID='simulated', date='v1'
)
```

### Run all sessions from the command line

```bash
# Run all sessions, skip completed ones
python run_sessions.py

# Force re-run
python run_sessions.py --no-skip

# Monitor progress live
tail -f logs/run_YYYYMMDD_HHMMSS.log
```

---

## Expected Data Format

The pipeline expects per-session data stored as a compressed pickle at:
```
{BASE_DATA_PATH}/{mouse_ID}/{date}/glm_data_dict.pkl.gz
```

Each file contains a list of per-trial dicts with the following keys:

| Key | Type | Description |
|---|---|---|
| `trial` | dict | `context` (0=CG, 1=visual, 2=audio), `choice` (0=L, 1=R), `outcome`, stimulus identity |
| `movement` | dict | `y_position`, `x_velocity`, `y_velocity` arrays (frames,) |
| `neural` | dict | `deconv`, `dff`, `z_dff` arrays (neurons × frames) |
| `events` | dict | Frame indices for `turn_onset`, `iti_start`, `sound_onset`, etc. |

---

## Dependencies

```
numpy
scipy
pandas
matplotlib
seaborn
tensorflow >= 2.x
scikit-learn
```

Install with:
```bash
pip install numpy scipy pandas matplotlib seaborn tensorflow scikit-learn
```
