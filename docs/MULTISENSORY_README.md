# Multisensory GLM Analysis Framework

This document describes the GLM encoding model framework for analyzing multisensory (VACANT) task data and cross-task generalization between priming (AVC) and VACANT sessions.

## Overview

The multisensory analysis framework extends the original AVC GLM analysis to support:

1. **Priming Analysis**: GLM on unisensory priming sessions (contexts 1 & 2 only)
2. **Cross-Task Generalization**: Train on priming, test on paired VACANT sessions
3. **VACANT Analysis**: GLM on multisensory VACANT sessions (contexts 0, 1 & 2)

## Key Concepts

### Task Differences

**Priming (AVC unisensory)**:
- Context 0 (CG): Congruent audio-visual trials
- Context 1 (Visual): Visual-only trials (audio absent, `np.nan`)
- Context 2 (Audio): Audio-only trials (visual absent, `np.nan`)

**VACANT (multisensory)**:
- Context 0 (CG): Congruent audio-visual trials (both stimuli present, aligned)
- Context 1 (Visual relevant): Both stimuli present, but in **conflict** (visual is relevant)
- Context 2 (Audio relevant): Both stimuli present, but in **conflict** (audio is relevant)

### Trial Type Mappings

From `prepare_glm_data.py` (lines 70-118):

**AVC ('avc' tasktype)**:
```python
visual_map: {1: 0, 2: 90, 3: 0, 4: 90, 5: np.nan, 6: np.nan}
audio_map:  {1: 0, 2: 1, 3: np.nan, 4: np.nan, 5: 0, 6: 1}
context_map: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
```

**VACANT ('2loc' tasktype)**:
```python
visual_map: {1: 0, 2: 90, 3: 0, 4: 90, 5: 90, 6: 0}
audio_map:  {1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}
context_map: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
```

**Key difference**: In VACANT, both modalities are present (not `np.nan`) even when only one is behaviorally relevant, creating conflict trials.

## Data Preparation

### Step 1: Prepare GLM Data

Use `prepare_glm_data.py` to create `glm_data_dict.pkl.gz` files:

```bash
# Process priming sessions (AVC tasktype)
python prepare_glm_data.py --task priming --base-path /Volumes/Data/ProcessedData

# Process VACANT sessions (2loc tasktype)
python prepare_glm_data.py --task vacant --base-path /Volumes/Data/ProcessedData

# Process all
python prepare_glm_data.py --task all --base-path /Volumes/Data/ProcessedData
```

**Output files** (per session):
- `{mouse_ID}/{date}/glm_data_dict.pkl.gz` - Trial-by-trial data structure
- `{mouse_ID}/{date}/task_df_pruned.pkl` - Pruned behavioral dataframe
- `{mouse_ID}/{date}/neural_activity_windows_*.npz` - Task-period activity windows

### Step 2: Build Stacked Data (if needed)

The existing GLM scripts can build `glm_data_stacked.pkl` on-the-fly, or you can pre-build:

```bash
python run_multisensory_models.py --analysis priming --build-stacked
```

## Session Registries

Defined in `session_registry_multisensory.py`:

### PPC_AVC_PRIMING
```python
{
    'IS-2-1L': ['2025-04-14', '2025-04-15', '2025-04-22', '2025-12-12'],
    'KO-5-1L': ['2026-04-01', '2026-04-02', '2026-04-08', '2026-04-28', 
                '2026-04-29', '2026-04-30'],
    'KO-6-00': ['2026-04-02', '2026-04-07']
}
```

### PPC_VACANT
```python
{
    'IS-2-1L': ['2025-04-14_1', '2025-04-15_1', '2025-04-22_1', '2025-12-12_1'],
    'KO-5-1L': ['2026-04-01_1', '2026-04-02_1', '2026-04-08_1', '2026-04-28_1',
                '2026-04-29_1', '2026-04-30_1'],
    'KO-6-00': ['2026-04-02_1', '2026-04-07_1']
}
```

Note: VACANT sessions have `_1` suffix

### CROSS_TASK_PAIRS
Maps priming sessions to paired VACANT sessions for cross-task generalization.

```python
{
    'IS-2-1L': [
        ('2025-04-14', '2025-04-14_1'),
        ('2025-04-15', '2025-04-15_1'),
        ('2025-04-22', '2025-04-22_1'),
        ('2025-12-12', '2025-12-12_1'),
    ],
    'KO-5-1L': [
        ('2026-04-01', '2026-04-01_1'),
        ('2026-04-02', '2026-04-02_1'),
        ('2026-04-08', '2026-04-08_1'),
        ('2026-04-28', '2026-04-28_1'),
        ('2026-04-29', '2026-04-29_1'),
        ('2026-04-30', '2026-04-30_1'),
    ]
}
```

Structure: `{mouse_ID: [(priming_date, vacant_date), ...]}`

## Running Analyses

### Master Runner: run_multisensory_models.py

Central script to run all multisensory analyses.

#### List Available Models

```bash
python run_multisensory_models.py --list
```

#### Run All Analyses

```bash
# Run everything (priming, cross-task, and VACANT)
python run_multisensory_models.py --analysis all

# With parallel workers (recommended: 2-3 on M2 Ultra)
python run_multisensory_models.py --analysis all --max-workers 3
```

#### Run Specific Analyses

```bash
# Priming only (unisensory GLMs on priming sessions)
python run_multisensory_models.py --analysis priming

# Cross-task only (train priming → test VACANT)
python run_multisensory_models.py --analysis cross-task

# VACANT only (GLMs on VACANT sessions)
python run_multisensory_models.py --analysis vacant
```

#### Run Specific Models

```bash
# Run only cross-task models
python run_multisensory_models.py --models GLM_cross_task GLM_cross_task_coupling

# Run priming without coupling
python run_multisensory_models.py --analysis priming --no-coupling
```

#### Other Options

```bash
# Force re-run (skip existing results)
python run_multisensory_models.py --analysis all --no-skip

# Use different neural metric
python run_multisensory_models.py --analysis all --metric dff

# Model-first execution (complete each model before starting next)
python run_multisensory_models.py --analysis all --model-first
```

### Running Individual GLM Scripts

You can also run GLM scripts directly:

#### Priming Analysis

```bash
# Unisensory GLM on priming sessions
python GLM_priming_uni.py --metric deconv
python GLM_priming_uni_coupling.py --metric deconv
```

#### Cross-Task Analysis

```bash
# Train on priming, test on VACANT
python GLM_cross_task.py --metric deconv
python GLM_cross_task_coupling.py --metric deconv
```

#### VACANT Analysis

```bash
# GLMs on VACANT sessions (all contexts)
python GLM_vacant_all.py --metric deconv
python GLM_vacant_all_coupling.py --metric deconv

# GLMs on VACANT sessions (conflict-only contexts 1,2)
python GLM_vacant_conflict_only.py --metric deconv
python GLM_vacant_conflict_only_coupling.py --metric deconv
```

These dedicated VACANT scripts automatically use `PPC_VACANT` from `session_registry_multisensory.py`.

## Model Descriptions

### Priming Analysis Models

| Model | Contexts | Coupling | Description |
|-------|----------|----------|-------------|
| `GLM_priming_uni` | [1, 2] | No | Unisensory GLM (visual + audio) on priming sessions |
| `GLM_priming_uni_coupling` | [1, 2] | Yes | Same with functional coupling predictors |

### Cross-Task Models

| Model | Train Contexts | Test Contexts | Coupling | Description |
|-------|----------------|---------------|----------|-------------|
| `GLM_cross_task` | [1, 2] | [0, 1, 2] | No | Train on priming (unisensory), test on VACANT (multisensory) |
| `GLM_cross_task_coupling` | [1, 2] | [0, 1, 2] | Yes | Same with functional coupling predictors |

**Key insight**: These models test whether representations learned in unisensory contexts generalize to multisensory conflict situations.

### VACANT Analysis Models

| Model | Contexts | Coupling | Description |
|-------|----------|----------|-------------|
| `GLM_vacant_all` | [0, 1, 2] | No | All contexts (CG + visual + audio) on VACANT |
| `GLM_vacant_all_coupling` | [0, 1, 2] | Yes | Same with coupling |
| `GLM_vacant_conflict_only` | [1, 2] | No | Conflict-only contexts (visual + audio) on VACANT |
| `GLM_vacant_conflict_only_coupling` | [1, 2] | Yes | Same with coupling |

## Output Structure

### Priming Analysis Results

Saved in: `{BASE_DATA_PATH}/{mouse_ID}/{date}/GLM_priming_uni_results/`
- `glm_priming_uni_full.pkl` - Full model results
- `glm_priming_uni_lesion_{group}.pkl` - Individual lesion results (9 groups)
- `glm_priming_uni_lesion_all.pkl` - Summary of all lesions

### Cross-Task Results

Saved in: `{BASE_DATA_PATH}/{mouse_ID}/{test_date}/GLM_cross_task_results/`
- `glm_cross_task_full.pkl` - Full model results (trained on priming, tested on VACANT)
- `glm_cross_task_lesion_{group}.pkl` - Individual lesion results (9 groups)
- `glm_cross_task_lesion_all.pkl` - Summary of all lesions
- `training_info.txt` - Documents which priming session was used for training

**Important**: Results are saved in the **test** session folder (VACANT), but include information about the **train** session (priming).

### VACANT Analysis Results

Saved in: `{BASE_DATA_PATH}/{mouse_ID}/{date_1}/GLM_vacant_all_results/`
- `glm_vacant_all_full.pkl` - Full model results
- `glm_vacant_all_lesion_{group}.pkl` - Individual lesion results
- `glm_vacant_all_lesion_all.pkl` - Summary of all lesions

## Lesion Analysis

All models include full lesion analysis with the following predictor groups:

**Base groups** (9 total):
1. `position_only` - Position-only (trialPhase) predictors
2. `sound` - Audio context × position + audio stimulus × position
3. `visual` - Visual context × position + visual stimulus × position
4. `velocity` - Velocity × position (4 directions)
5. `choice_spatial` - Choice place fields (truncated at turn)
6. `turn_onset` - Turn onset temporal predictors
7. `pre_turn` - Pre-turn anticipatory predictors (if enabled)
8. `context_choice` - Context × choice interaction (if enabled)
9. `outcome` - Reward/ITI temporal predictors

**Coupling groups** (5 additional, if coupling enabled):
- `network_pc1` through `network_pc5` - Functional coupling predictors (PCA components)

## Predictor Builder Compatibility

The existing `predictor_builder.py` works correctly for VACANT data without modification. Here's why:

### Stimulus Presence Gating

The predictor builder uses `np.isnan()` to gate predictors:

```python
# Audio predictors (line 348)
if np.isnan(audio_stim):
    return pred, names  # Zero predictors if audio absent

# Visual predictors (line 377)  
if np.isnan(visual_stim):
    return pred, names  # Zero predictors if visual absent
```

### VACANT Behavior

In VACANT tasks:
- **Both audio and visual stimuli are present** (not `np.nan`) on ALL trials
- **Both predictor groups are active** on all trials
- **Context mapping determines behavioral relevance**:
  - Context 0: both relevant (congruent)
  - Context 1: visual relevant, audio distractor
  - Context 2: audio relevant, visual distractor

### GLM Learning

The GLM automatically learns:
- Which stimulus drives behavior in each context
- Cross-modal effects during conflict trials
- How conflicting information is resolved

This is the **intended behavior** - the GLM should see both stimuli to capture cross-modal interactions.

## Analysis Recommendations

### Recommended Execution Order

1. **Data Preparation**:
   ```bash
   python prepare_glm_data.py --task all --base-path /Volumes/Data/ProcessedData
   ```

2. **Priming Analysis** (baseline unisensory performance):
   ```bash
   python run_multisensory_models.py --analysis priming --max-workers 2
   ```

3. **VACANT Analysis** (multisensory within-task):
   ```bash
   python run_multisensory_models.py --analysis vacant --max-workers 2
   ```

4. **Cross-Task Analysis** (generalization):
   ```bash
   python run_multisensory_models.py --analysis cross-task --max-workers 2
   ```

### Performance Considerations

- Use `--max-workers 2-3` for optimal performance on M2 Ultra with Metal GPU
- Each session × model combination clears memory via subprocess
- Already-completed sessions are automatically skipped (use `--no-skip` to override)
- Expected runtime: ~2-4 hours for full analysis (all models, all sessions)

### Metrics to Compare

Key metrics saved in results files:

**Within-task metrics** (train and test on same task):
- `cv_frac_dev` - Cross-validated fraction deviance explained (training metric)
- `test_frac_dev` - Test set fraction deviance explained
- `test_trial_avg_r2` - Trial-averaged R² (best split per neuron)
- `test_frac_dev_per_ctx` - Per-context test performance

**Cross-task metrics**:
- `test_fde_per_neuron` - Test FDE on VACANT (model trained on priming)
- `test_r2_per_neuron` - Test R² on VACANT
- Compare to within-task VACANT performance to quantify generalization gap

**Lesion deltas**:
- `delta_fde` - Unique contribution of each predictor group
- `delta_r2` - Change in R² when group is lesioned

## Troubleshooting

### Common Issues

**Issue**: `glm_data_dict.pkl.gz not found`
- **Solution**: Run `prepare_glm_data.py` first for the relevant task type

**Issue**: `glm_data_stacked.pkl not found`
- **Solution**: Use `--build-stacked` flag or let GLM scripts build it automatically

**Issue**: Memory errors during parallel execution
- **Solution**: Reduce `--max-workers` to 1 or 2

**Issue**: Existing GLM scripts using wrong sessions
- **Solution**: Use the master runner (`run_multisensory_models.py`) which manages session selection

### Checking Results

To verify results were saved correctly:

```python
import pickle
import os

# Check cross-task results
mouse_ID = 'KO-5-1L'
test_date = '2026-04-08_1'
results_dir = f'/Volumes/Data/ProcessedData/{mouse_ID}/{test_date}/GLM_cross_task_results/'

# Load full model
with open(os.path.join(results_dir, 'glm_cross_task_full.pkl'), 'rb') as f:
    results = pickle.load(f)

print(f"Train session: {results['train_session']}")
print(f"Test session: {results['test_session']}")
print(f"Test FDE mean: {results['test_fde_mean']:.4f}")
print(f"Test R² mean: {results['test_r2_mean']:.4f}")
```

## Files Created

New files in this framework:

1. **Data Preparation**:
   - `prepare_glm_data.py` - Unified data preparation for AVC and VACANT

2. **Session Registries**:
   - `session_registry_multisensory.py` - Priming, VACANT, and cross-task pairs

3. **GLM Scripts**:
   - `GLM_cross_task.py` - Cross-task generalization (no coupling)
   - `GLM_cross_task_coupling.py` - Cross-task with coupling

4. **Master Runner**:
   - `run_multisensory_models.py` - Central runner for all analyses

5. **Documentation**:
   - `MULTISENSORY_README.md` - This document

## Next Steps

After running the analyses:

1. **Load and compare results** across analysis types
2. **Analyze lesion deltas** to identify key predictor groups
3. **Compare cross-task vs. within-task performance** to quantify generalization
4. **Per-neuron analysis** to identify neurons with stable vs. context-dependent encoding
5. **Visualization** of weights, performance metrics, and lesion effects

---

For questions or issues, refer to the main `run_all_models.py` documentation or contact the analysis lead.
