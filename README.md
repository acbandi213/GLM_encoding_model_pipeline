# GLM Encoding Model Pipeline

A Poisson generalized linear model (GLM) pipeline for fitting neural encoding models to calcium imaging data from head-fixed mice navigating a virtual reality T-maze, with a matching pipeline for Neuropixels recordings from monkey frontal eye field (FEF). The model decomposes single-neuron activity into contributions from behavioural predictor groups using raised-cosine basis functions and elastic-net / group-lasso regularization.

**Latest update**: single-task fitting (`GLM_single_task.py`), monkey FEF fitting (`NHP_FEF/FEF_v2/`), one configurable data path, a pinned `requirements.txt`, and a demo that runs without any data.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Model Types](#model-types)
  - [Unisensory Models (AVC Task)](#unisensory-models-avc-task)
  - [Single-Task Models](#single-task-models)
  - [Sub-population and Split-Coupling Models](#sub-population-and-split-coupling-models)
  - [Multisensory Models](#multisensory-models)
  - [Monkey FEF Models](#monkey-fef-models)
- [Installation & Setup](#installation--setup)
- [Try It Without Data](#try-it-without-data)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Data Preparation](#data-preparation)
- [Understanding Model Outputs](#understanding-model-outputs)
- [File Descriptions](#file-descriptions)

---

## Overview

Each neuron's activity is modeled as:

$$\hat{y} = \exp(X \mathbf{w} + w_0)$$

where $X$ is a design matrix of behavioural predictors (position, context, stimulus identity, choice, velocity, turn onset, and outcome), and $\mathbf{w}$ is a sparse weight vector fit by cross-validated penalized maximum likelihood.

The unique contribution of each predictor group is quantified via **lesion models**: the full model is refit with one group removed at a time, and the drop in fractional deviance explained ($\Delta \text{FDE}$) is recorded per neuron.

### Key Features

- **9 Predictor Groups**: Position, context, visual stimulus, audio stimulus, choice, velocity, turn onset, outcome timing, outcome identity
- **Functional Coupling** (optional): Population activity predictors from top 5 PCs
- **Cross-Validated**: 3-fold cross-validation on the training trials (folds split by trial), scored on a held-out 30 % of trials
- **Lesion Analysis**: Per-group contribution quantification
- **Multi-Task Support**: Unisensory (AVC) and multisensory (VACANT) tasks
- **Single-Task Fits**: Visual only, audio only, or congruent only
- **Two Species**: Mouse PPC (calcium imaging) and monkey FEF (Neuropixels), sharing one fitting core
- **Cross-Task Generalization**: Train on one task, test on another

---

## Repository Structure

```
GLM_encoding_model/
├── core/                           # Core utilities & shared functions
│   ├── glm_class.py                   # TensorFlow-based GLM class
│   ├── predictor_builder.py           # Design matrix builder with cosine basis
│   ├── session_pipeline.py            # Data I/O, fitting, coupling, lesion pipeline
│   ├── prepare_glm_data.py            # Data preparation pipeline
│   ├── build_stacked_data.py          # Builds glm_data_stacked.pkl
│   ├── paths.py                       # BASE_DATA_PATH (set GLM_BASE_DATA_PATH)
│   └── datasets.py                    # Dataset utilities
│
├── config/                         # Configuration & session registries
│   ├── session_registry.py            # Unisensory (AVC) session lists
│   ├── session_registry_multisensory.py  # Priming, VACANT, cross-task pairs
│   └── check_model_status.py          # Which sessions have which results
│
├── models/                         # Model scripts
│   ├── unisensory/                    # AVC task models
│   │   ├── GLM_all_context.py            # All contexts [0,1,2]
│   │   ├── GLM_uni.py                    # Visual + Audio contexts [1,2]
│   │   ├── GLM_CG.py                     # Congruent context only
│   │   ├── GLM_*_coupling.py             # The three above + population coupling
│   │   ├── GLM_single_task.py            # ONE task: --task visual | audio | cg
│   │   ├── GLM_all_ctx_cpl_subpop.py     # Coupling split by neuron sub-population
│   │   ├── GLM_subpop_coupling_loo.py    # Leave-one-out sub-population coupling
│   │   ├── GLM_uni_coupling_split.py     # Coupling weights split by context
│   │   └── GLM_cd_split_* / GLM_ti_td_*  # Paper-specific coupling models
│   │
│   └── multisensory/                  # Priming, VACANT, cross-task models
│       ├── GLM_priming_uni.py            # Priming sessions (contexts 1,2)
│       ├── GLM_cross_task.py             # Train priming → Test VACANT
│       ├── GLM_vacant_all.py             # VACANT all contexts [0,1,2]
│       ├── GLM_vacant_conflict_only.py   # VACANT conflicts only [1,2]
│       └── GLM_*_coupling.py             # Each of the above + coupling
│
├── runners/                        # Master runner scripts
│   ├── run_unisensory_models.py       # Run unisensory + single-task models
│   ├── run_multisensory_models.py     # Run all multisensory analyses
│   ├── _run_single_session.py         # Helper: one model, one session, one process
│   ├── run_*.sh                       # Batch scripts for the paper-specific models
│   └── deprecated/                    # Old runners, kept for reference
│
├── NHP_FEF/FEF_v2/                 # Monkey FEF pipeline (see PIPELINE.md there)
│   ├── fef/                           # .mat reader, design matrix, stacking, runner
│   ├── models/                        # GLM_base.py, GLM_base_coupling.py
│   ├── scripts/wp6_build_stacked.py   # Builds the per-probe stacked data
│   └── tests/
│
├── analysis/                       # Analysis scripts & notebooks
│   ├── check_model_runs.py            # Verify model outputs
│   ├── run_model_comparison.py        # Compare model performance
│   └── notebooks/                     # Jupyter notebooks
│
├── testing/                        # Testing & simulation
│   ├── demo_synthetic_fit.py          # End-to-end fit with no real data
│   ├── simulated_neurons.py           # Synthetic neurons with ground truth
│   ├── synthetic_session.py           # Fully synthetic trial generator
│   └── validate_split_coupling.py     # Synthetic check of the split-coupling model
│
├── docs/                           # RUNNING.md, QUICK_REFERENCE.md, MULTISENSORY_README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Model Types

### Unisensory Models (AVC Task)

The original AVC (Audio-Visual-Context) task where mice navigate with unisensory or congruent cues.

#### **Task Structure**
- **Context 0 (CG)**: Congruent audio-visual trials (both cues present and aligned)
- **Context 1 (Visual)**: Visual-only trials (audio is `nan`)
- **Context 2 (Audio)**: Audio-only trials (visual is `nan`)

#### **Available Models**

| Model | Script | Contexts | Coupling | Description |
|-------|--------|----------|----------|-------------|
| `GLM_CG` | `models/unisensory/GLM_CG.py` | [0] | No | Congruent context only |
| `GLM_CG_coupling` | `models/unisensory/GLM_CG_coupling.py` | [0] | Yes | Congruent + coupling |
| `GLM_uni` | `models/unisensory/GLM_uni.py` | [1, 2] | No | Visual + Audio contexts only |
| `GLM_uni_coupling` | `models/unisensory/GLM_uni_coupling.py` | [1, 2] | Yes | Visual + Audio + coupling |
| `GLM_all_context` | `models/unisensory/GLM_all_context.py` | [0, 1, 2] | No | All contexts |
| `GLM_all_context_coupling` | `models/unisensory/GLM_all_context_coupling.py` | [0, 1, 2] | Yes | All contexts + coupling |

**Session Registry**: `config/session_registry.py` (`PPC_AVC`)

### Single-Task Models

`models/unisensory/GLM_single_task.py` fits the model on one task alone. One file covers all three tasks; pick the task with `--task`.

| Command | Context | Results folder | Groups removed |
|---------|---------|----------------|----------------|
| `--task visual` | [1] | `GLM_visual_results/` | `audio_context`, `audio_stim` |
| `--task audio` | [2] | `GLM_audio_results/` | `visual_context`, `visual_stim` |
| `--task cg` | [0] | `GLM_cg_single_results/` | none |

Add `--coupling` for the coupling variant (`GLM_visual_coupling_results/`, and so on).

```bash
python models/unisensory/GLM_single_task.py --task visual
python models/unisensory/GLM_single_task.py --task audio --coupling
python models/unisensory/GLM_single_task.py --task visual --mouse KO-5-1L --date 2026-03-06

# Through the runner (parallel, one process per session). Single-task models
# only run when you name them; a plain run does not include them.
python runners/run_unisensory_models.py --models GLM_visual GLM_audio
```

**Why two groups are removed.** In a visual-only trial the audio predictors are filled with zeros, so within the visual task those 20 columns are constant and carry nothing. The script removes them before fitting and does not run a lesion for them. It stops with an error if a group it is about to remove is not constant, or if any other group is.

**How to read the task's own `*_context` group.** With a single task there is no context to compare, so `visual_context` (in a visual fit) is the average cue-locked response on all trials, and `visual_stim` is the left/right cue difference.

To add another single-condition model, add one line to the `TASKS` table at the top of the file.

### Sub-population and Split-Coupling Models

These extend the coupling models. They need a finished base model for the same session, because they sort neurons using its results.

| Model | Script | What it adds |
|-------|--------|--------------|
| `GLM_all_ctx_cpl_subpop` | `models/unisensory/GLM_all_ctx_cpl_subpop.py` | Separate coupling predictors for two neuron sub-populations (context-invariant and context-dependent) |
| `GLM_subpop_coupling_loo` | `models/unisensory/GLM_subpop_coupling_loo.py` | The same, with each neuron left out of its own coupling predictor |
| `GLM_uni_coupling_split` | `models/unisensory/GLM_uni_coupling_split.py` | Separate coupling weights for the visual and audio contexts, with sham and time-split controls |
| `GLM_cd_split_coupling`, `GLM_ti_td_*` | `models/unisensory/` | Paper-specific versions of the above; run with the `runners/run_*.sh` scripts |

`testing/validate_split_coupling.py` checks the split-coupling model on synthetic data.

---

### Multisensory Models

Extended framework for the VACANT (multisensory conflict) task and cross-task generalization analyses.

#### **Task Comparison**

| Task | Context 0 | Context 1 | Context 2 |
|------|-----------|-----------|-----------|
| **AVC (Priming)** | Congruent (both aligned) | Visual only (audio `nan`) | Audio only (visual `nan`) |
| **VACANT** | Congruent (both aligned) | **Both present, conflict** (visual relevant) | **Both present, conflict** (audio relevant) |

**Key Difference**: In VACANT, contexts 1 and 2 have **both stimuli present but conflicting**, not unisensory. The model learns which stimulus is relevant from context.

---

#### **1. Priming Analysis Models**

Baseline performance on unisensory priming sessions (no congruent context, contexts [1,2] only).

| Model | Script | Description | Train/Test |
|-------|--------|-------------|------------|
| `GLM_priming_uni` | `models/multisensory/GLM_priming_uni.py` | Unisensory contexts [1,2] | Train & Test on priming |
| `GLM_priming_uni_coupling` | `models/multisensory/GLM_priming_uni_coupling.py` | Same + coupling | Train & Test on priming |

**Purpose**: Establish baseline encoding performance on unisensory-only trials.

**Session Registry**: `config/session_registry_multisensory.py` (`PPC_AVC_PRIMING`)

**Example Sessions**:
- IS-2-1L: 4 sessions (2025-04-14, 2025-04-15, 2025-04-22, 2025-12-12)
- KO-5-1L: 6 sessions (2026-04-01, 2026-04-02, 2026-04-08, 2026-04-28, 2026-04-29, 2026-04-30)

---

#### **2. Cross-Task Generalization Models**

Test whether representations learned in unisensory contexts generalize to multisensory conflict situations.

| Model | Script | Description | Train Context | Test Context |
|-------|--------|-------------|---------------|--------------|
| `GLM_cross_task` | `models/multisensory/GLM_cross_task.py` | Cross-task generalization | [1,2] priming | [0,1,2] VACANT |
| `GLM_cross_task_coupling` | `models/multisensory/GLM_cross_task_coupling.py` | Same + coupling | [1,2] priming | [0,1,2] VACANT |

**Training Phase**:
- Fit GLM on priming session (unisensory contexts [1,2])
- Learn weights $\mathbf{w}$ for each neuron
- No test data used during training

**Testing Phase**:
- Load paired VACANT session for same mouse
- Apply learned weights to VACANT data
- Evaluate: FDE (Fractional Deviance Explained) and R²
- **Lesion analysis**: Refit full model on VACANT, then lesion each group

**Key Insight**: Quantifies how well unisensory-trained models handle multisensory conflicts.

**Session Pairs**: Defined in `config/session_registry_multisensory.py` (`CROSS_TASK_PAIRS`)

**Results Saved In**: Test session directory (VACANT session folder)
- `GLM_cross_task_results/glm_cross_task_full.pkl`
- `GLM_cross_task_results/glm_cross_task_lesion_*.pkl`
- `GLM_cross_task_results/training_info.txt` (documents which priming session was used)

---

#### **3. VACANT Analysis Models**

Direct analysis of VACANT sessions (multisensory with conflict trials).

| Model | Script | Contexts | Coupling | Description |
|-------|--------|----------|----------|-------------|
| `GLM_vacant_all` | `models/multisensory/GLM_vacant_all.py` | [0,1,2] | No | All contexts (congruent + conflicts) |
| `GLM_vacant_all_coupling` | `models/multisensory/GLM_vacant_all_coupling.py` | [0,1,2] | Yes | All contexts + coupling |
| `GLM_vacant_conflict_only` | `models/multisensory/GLM_vacant_conflict_only.py` | [1,2] | No | Conflict trials only (no congruent) |
| `GLM_vacant_conflict_only_coupling` | `models/multisensory/GLM_vacant_conflict_only_coupling.py` | [1,2] | Yes | Conflict trials + coupling |

**Purpose**: 
- `vacant_all`: Full VACANT performance including congruent trials
- `vacant_conflict_only`: Isolate conflict-resolution encoding (analogous to priming's [1,2])

**Session Registry**: `config/session_registry_multisensory.py` (`PPC_VACANT`)

**Train/Test**: Standard within-session cross-validation (3-fold, split by trial)

---

### Monkey FEF Models

`NHP_FEF/FEF_v2/` fits the same Poisson GLM to Neuropixels recordings from monkey frontal eye field. The two contexts are memory-guided saccade (1) and step-ramp pursuit (2). It has its own data reader and design matrix, and reuses `core/` for stacking, fitting, coupling and lesions.

| Model | Script | Contexts | Coupling |
|-------|--------|----------|----------|
| `GLM_base` | `NHP_FEF/FEF_v2/models/GLM_base.py` | [1, 2] | No |
| `GLM_base_coupling` | `NHP_FEF/FEF_v2/models/GLM_base_coupling.py` | [1, 2] | Yes |

```bash
export NHP_SESSIONS=/path/to/session_mat_files
export NHP_RESULTS=/path/to/results
cd NHP_FEF/FEF_v2
python scripts/wp6_build_stacked.py          # build the per-probe stacked data
python models/GLM_base.py --workers 6        # fit every probe, then lesions
```

See [`NHP_FEF/FEF_v2/PIPELINE.md`](NHP_FEF/FEF_v2/PIPELINE.md) for the details.

---

## Installation & Setup

### Requirements

Python 3.11 and the packages in `requirements.txt` (TensorFlow 2.17, NumPy, SciPy, pandas, scikit-learn, matplotlib).

### Installation

```bash
git clone https://github.com/acbandi213/GLM_encoding_model_pipeline.git
cd GLM_encoding_model_pipeline
pip install -r requirements.txt
```

### Point the pipeline at your data

Every script reads and writes under one folder, `BASE_DATA_PATH`, laid out as `{mouse_ID}/{date}/`. Set it once with an environment variable:

```bash
export GLM_BASE_DATA_PATH=/path/to/ProcessedData
```

If the variable is not set, the default in `core/paths.py` is used. The monkey pipeline uses `NHP_SESSIONS` and `NHP_RESULTS` in the same way.

The mouse sessions are available as NWB files on DANDI ([dandiset 002050](https://dandiarchive.org/dandiset/002050)). The pipeline itself reads the lab's per-session pickle files described under [Data Preparation](#data-preparation), not NWB.

---

## Try It Without Data

```bash
python testing/demo_synthetic_fit.py
```

This makes a synthetic session, builds the design matrix, simulates 14 neurons that each respond to one predictor group, fits the model, and prints the held-out fit quality. It writes to a temporary folder. Use `--trials 60` for a faster run.

---

## Quick Start

### 1. Prepare Data

```bash
# Prepare GLM data for all sessions
python core/prepare_glm_data.py --task all

# Or prepare specific tasks
python core/prepare_glm_data.py --task priming
python core/prepare_glm_data.py --task vacant
```

This creates `glm_data_dict.pkl.gz` and `glm_data_stacked.pkl` for each session.

### 2. Run Models

#### **Unisensory Analysis**
```bash
# Run all unisensory models
python runners/run_unisensory_models.py --metric deconv

# Run specific model
python models/unisensory/GLM_uni.py --metric deconv
```

#### **Multisensory Analysis**
```bash
# Run all multisensory analyses (priming + cross-task + VACANT)
python runners/run_multisensory_models.py --analysis all --metric deconv

# Run specific analysis
python runners/run_multisensory_models.py --analysis priming --metric deconv
python runners/run_multisensory_models.py --analysis cross-task --metric deconv
python runners/run_multisensory_models.py --analysis vacant --metric deconv

# Parallel execution (faster)
python runners/run_multisensory_models.py --analysis all --max-workers 3 --metric deconv
```

### 3. Check Results

```bash
# Check which sessions completed
python analysis/check_model_runs.py

# Compare models on a single session
python analysis/run_model_comparison.py
```

---

## Detailed Usage

### Running Individual Models

All model scripts can be run directly:

```bash
# Unisensory models
python models/unisensory/GLM_uni.py [OPTIONS]
python models/unisensory/GLM_all_context.py [OPTIONS]

# Multisensory models
python models/multisensory/GLM_priming_uni.py [OPTIONS]
python models/multisensory/GLM_cross_task.py [OPTIONS]
python models/multisensory/GLM_vacant_all.py [OPTIONS]

# Common options:
#   --metric deconv|dff       Neural activity metric (default: deconv)
#   --no-skip                 Re-run even if results exist
```

### Using Master Runners

#### `run_unisensory_models.py` (Unisensory)

```bash
# Run all unisensory models
python runners/run_unisensory_models.py --metric deconv

# Run specific models
python runners/run_unisensory_models.py --models GLM_uni GLM_uni_coupling --metric deconv

# Skip coupling models (faster)
python runners/run_unisensory_models.py --no-coupling --metric deconv

# Model-first ordering (finish each model on all sessions before the next)
python runners/run_unisensory_models.py --model-first --metric deconv

# Single-task models (only run when named)
python runners/run_unisensory_models.py --models GLM_visual GLM_audio

# Build any missing glm_data_stacked.pkl first
python runners/run_unisensory_models.py --build-stacked
```

**Modes**:
- `session-first`: For each session → run all models → clear memory (default, safer)
- `model-first` (`--model-first`): For each model → run all sessions (faster if models differ greatly)

#### `run_multisensory_models.py` (Multisensory)

```bash
# Run everything
python runners/run_multisensory_models.py --analysis all --metric deconv

# Specific analysis types
python runners/run_multisensory_models.py --analysis priming --metric deconv
python runners/run_multisensory_models.py --analysis cross-task --metric deconv
python runners/run_multisensory_models.py --analysis vacant --metric deconv

# Select specific models
python runners/run_multisensory_models.py --analysis vacant \
    --models GLM_vacant_all GLM_vacant_all_coupling --metric deconv

# Parallel execution
python runners/run_multisensory_models.py --analysis all --max-workers 3 --metric deconv

# Skip coupling variants (runs twice as fast)
python runners/run_multisensory_models.py --analysis all --no-coupling --metric deconv

# List available models
python runners/run_multisensory_models.py --list
```

---

## Data Preparation

### Input Data Structure

Expected directory structure:
```
BASE_DATA_PATH/
├── {mouse_ID}/
│   └── {date}/
│       ├── F.npy                  # Neural activity (frames × neurons)
│       ├── behavior_data.pkl      # Trial structure & behavior
│       └── ... (other raw data)
```

### GLM Data Pipeline

The pipeline creates two files per session:

1. **`glm_data_dict.pkl.gz`** (trial-based)
   - Dictionary keyed by context → trial_idx
   - Contains: Y (neural), position, stimuli, choice, velocity, etc.

2. **`glm_data_stacked.pkl`** (stacked matrix)
   - X: Design matrix (frames × predictors)
   - Y: Neural activity (frames × neurons)
   - metadata: Context, trial, group labels for CV

### Running Data Preparation

```bash
# Prepare all datasets
python core/prepare_glm_data.py --task all

# Prepare specific tasks
python core/prepare_glm_data.py --task avc
python core/prepare_glm_data.py --task priming
python core/prepare_glm_data.py --task vacant

# Custom base path
python core/prepare_glm_data.py --task all \
    --base-path /path/to/data
```

---

## Understanding Model Outputs

### Output Directory Structure

For each session, model results are saved to:
```
{BASE_DATA_PATH}/{mouse_ID}/{date}/{MODEL_DIR}/
├── {MODEL_PREFIX}_full.pkl                # Full model weights & metrics
├── {MODEL_PREFIX}_lesion_{group}.pkl      # Individual lesion results (9 groups)
├── {MODEL_PREFIX}_lesion_all.pkl          # Summary of all lesions
└── training_info.txt                      # Metadata (cross-task only)
```

### Full Model Output (`*_full.pkl`)

```python
{
    'weights': ndarray (n_neurons, n_predictors),     # Learned GLM weights
    'intercepts': ndarray (n_neurons,),               # Per-neuron baseline
    'pred_names': list,                               # Predictor names
    'fde': ndarray (n_neurons,),                      # Fractional deviance explained
    'r2_by_trial': ndarray (n_neurons, n_trials),     # Per-trial R²
    'best_r2': ndarray (n_neurons,),                  # Best trial-averaged R²
    'predictions': ndarray (n_frames, n_neurons),     # Full predictions
    'metadata': dict,                                 # CV fold info, timing, etc.
}
```

### Lesion Output (`*_lesion_{group}.pkl`)

```python
{
    'group_name': str,                                # E.g., 'visual_all', 'choice'
    'fde': ndarray (n_neurons,),                      # FDE with this group removed
    'delta_fde': ndarray (n_neurons,),                # FDE_full - FDE_lesion
    'r2_by_trial': ndarray (n_neurons, n_trials),
    'best_r2': ndarray (n_neurons,),
    'delta_r2': ndarray (n_neurons,),                 # R²_full - R²_lesion
    # ... (weights, predictions, etc.)
}
```

### Lesion Summary (`*_lesion_all.pkl`)

Aggregates all lesion results for easy comparison:

```python
{
    'group_names': list,                              # All lesioned groups
    'fde_full': ndarray (n_neurons,),                 # Full model FDE
    'fde_lesion': ndarray (n_neurons, n_groups),      # FDE for each lesion
    'delta_fde': ndarray (n_neurons, n_groups),       # Drop in FDE per group
    'r2_full': ndarray (n_neurons,),
    'r2_lesion': ndarray (n_neurons, n_groups),
    'delta_r2': ndarray (n_neurons, n_groups),
    # ... (metadata, timing, etc.)
}
```

### Predictor Groups

Standard 9 groups (14 with coupling):

1. **position** - Maze position (5 bumps)
2. **context** - Task context (one-hot: CG, Visual, Audio)
3. **visual_all** - Visual stimulus identity
4. **audio_all** - Audio stimulus identity
5. **choice** - Left/right choice
6. **velocity** - Movement speed (5 bumps)
7. **turn_onset** - Turn initiation timing (5 bumps)
8. **outcome_time** - Outcome delivery timing (5 bumps)
9. **outcome** - Outcome identity (correct/error)

**With coupling** (+5 groups):
- **network_pc1** through **network_pc5** - Population coupling predictors

---

## File Descriptions

### Core Utilities (`core/`)

#### `glm_class.py`
Core `GLM` class implementing regularized Poisson GLM via TensorFlow.

**Features**:
- Elastic-net + group-lasso regularization
- Adam optimizer with early stopping
- Cross-validation support
- Handles NaN predictors (for unisensory trials)

**Key Methods**:
- `fit()`: Fit model weights
- `predict()`: Generate predictions
- `score()`: Compute deviance/log-likelihood

#### `predictor_builder.py`
Design matrix construction with raised-cosine basis functions.

**Key Class**: `GLMPredictorBuilder`
- Generates time-lagged predictors with smooth temporal kernels
- Handles categorical (one-hot) and continuous (basis) predictors
- Groups predictors for lesion analysis

#### `session_pipeline.py`
High-level pipeline functions for data I/O, fitting, and lesion analysis.

**Key Functions**:
- `open_glm_stacked_data()`: Load prepared data
- `fit_glm_single_session()`: Fit full model
- `run_lesion_models()`: Run all lesion analyses
- `add_coupling_predictor()`: Add population coupling PCs

### Configuration (`config/`)

#### `session_registry.py`
Unisensory (AVC) session definitions.

```python
PPC_AVC = {
    'IS-2-1L': ['2025-04-14', '2025-04-15', ...],
    'KO-5-1L': ['2026-04-01', '2026-04-02', ...],
    # ... more mice
}
```

#### `session_registry_multisensory.py`
Multisensory session definitions.

```python
# Priming sessions (unisensory only)
PPC_AVC_PRIMING = {
    'IS-2-1L': ['2025-04-14', '2025-04-15', ...],
    'KO-5-1L': ['2026-04-01', '2026-04-02', ...],
}

# VACANT sessions (multisensory)
PPC_VACANT = {
    'IS-2-1L': ['2025-04-16', '2025-04-17', ...],
    'KO-5-1L': ['2026-04-03', '2026-04-04', ...],
}

# Cross-task pairs (priming_date, vacant_date)
CROSS_TASK_PAIRS = {
    'IS-2-1L': [
        ('2025-04-14', '2025-04-16'),  # Priming → VACANT
        ('2025-04-15', '2025-04-17'),
    ],
    # ...
}
```

### Analysis Scripts (`analysis/`)

#### `core/prepare_glm_data.py`
Prepare `glm_data_dict.pkl.gz` and `glm_data_stacked.pkl` for all sessions.

#### `check_model_runs.py`
Verify which sessions have completed model outputs.

#### `run_model_comparison.py`
Quick 3-way model comparison on a single session.

### Notebooks (`analysis/notebooks/`)

9 analysis notebooks covering:
- Model fitting and validation
- Predictor contribution analysis
- Population structure (Q1)
- Context invariance (Q2)
- Paper figures

**Important**: Notebooks now use new import paths:
```python
from core.session_pipeline import BASE_DATA_PATH
from config.session_registry import PPC_AVC
```

---

## Tips & Best Practices

### Memory Management

- Use `--max-workers 2` or `3` for parallel execution
- Prefer `session-first` mode for large runs (clears memory between sessions)
- Monitor with `htop` or Activity Monitor

### Debugging

```bash
# Run single session
python models/multisensory/GLM_priming_uni.py --metric deconv

# Check what's completed
python analysis/check_model_runs.py
```

### Performance

- **Fastest**: No coupling, parallel workers, skip existing
  ```bash
  python runners/run_multisensory_models.py --analysis all \
      --no-coupling --max-workers 3 --metric deconv
  ```

- **Most thorough**: All models, session-first
  ```bash
  python runners/run_multisensory_models.py --analysis all \
      --metric deconv --no-skip
  ```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'session_pipeline'`:

Run scripts from the repository root, or by their full path. All imports should use new paths:
```python
from core.session_pipeline import ...
from config.session_registry import ...
```

### Missing Data

Ensure `glm_data_stacked.pkl` exists:

```bash
python core/prepare_glm_data.py --task all
```

### Model Not Running

Check logs in `logs/` directory:
```bash
ls -lt logs/
tail -100 logs/run_multisensory_all_*.log
```

---

## Citation

If you use this code, please cite:

See [`CITATION.cff`](CITATION.cff). A paper reference will be added when it is available.

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Contact

Open an issue on the [GitHub repository](https://github.com/acbandi213/GLM_encoding_model_pipeline/issues).

---

**Last Updated**: November 11, 2026  
**Version**: 2.1 (single-task fitting, monkey FEF pipeline, configurable data path)

