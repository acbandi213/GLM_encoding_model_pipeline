# Running the GLM Model Pipeline

## Overview

The pipeline trains 6 GLM variants per session, covering all combinations of
context subset and functional coupling:

| Model | Contexts | Coupling | Results directory |
|---|---|---|---|
| `GLM_all_context` | CG + visual + audio | No | `GLM_all_context_results/` |
| `GLM_all_context_coupling` | CG + visual + audio | Yes | `GLM_all_context_coupling_results/` |
| `GLM_uni` | visual + audio only | No | `GLM_uni_results/` |
| `GLM_uni_coupling` | visual + audio only | Yes | `GLM_uni_coupling_results/` |
| `GLM_CG` | CG only | No | `GLM_CG_results/` |
| `GLM_CG_coupling` | CG only | Yes | `GLM_CG_coupling_results/` |

Each model saves:
- `glm_{prefix}_full.pkl` — full model weights + test metrics
- `glm_{prefix}_lesion_{group}.pkl` × 9 (or 14 with coupling) — per-group lesion refits
- `glm_{prefix}_lesion_all.pkl` — combined lesion summary

---

## Prerequisites

### 1. Activate the conda environment

```bash
conda activate Bandi_Runyan_24
```

### 2. Navigate to the code directory

```bash
cd /path/to/GLM_encoding_model_pipeline
```

**Optional — add a shortcut alias** (run once, then available in every new terminal):

```bash
echo 'alias cdglm="cd /path/to/GLM_encoding_model_pipeline"' >> ~/.zshrc
source ~/.zshrc
```

Then just type `cdglm` instead of the full path.

### 3. Ensure base stacked data exists

The 6 GLM model scripts all reuse `glm_data_stacked.pkl`, which is built by the
base pipeline. If it does not yet exist for a session, run this first:

```bash
python run_sessions.py
```

---

## Registering Sessions

All sessions are defined in **`session_registry.py`** — this is the single file
to edit when adding or removing sessions:

```python
PPC_AVC = {
    'KO-5-1L': ['2026-03-24'],
    'KO-6-00': ['2026-03-23', '2026-03-25'],
}
```

Every runner script imports `PPC_AVC` from here, so one edit propagates
everywhere.

---

## Running the Models

### `runners/run_unisensory_models.py` — recommended entry point

All commands below assume you are already in the `GLM_encoding_model` directory
with `Bandi_Runyan_24` activated.

#### Run all 6 models (session-first, best for memory)

```bash
python runners/run_unisensory_models.py
```

The default **session-first** mode completes all 6 models for session 1, fully
clears memory, then moves to session 2. Each `(session × model)` runs as a
separate subprocess so the OS reclaims all RAM between combinations.

#### List available model names

```bash
python runners/run_unisensory_models.py --list
```

#### Run specific models only

```bash
python runners/run_unisensory_models.py --models GLM_uni GLM_CG
python runners/run_unisensory_models.py --models GLM_all_context GLM_uni GLM_CG
```

#### Skip all coupling variants (3 base models only — much faster)

```bash
python runners/run_unisensory_models.py --no-coupling
```

#### Force re-run (ignore existing results)

```bash
python runners/run_unisensory_models.py --no-skip
```

#### Use a different neural metric

```bash
python runners/run_unisensory_models.py --metric dff
python runners/run_unisensory_models.py --metric z_dff
```

#### Model-first ordering (one model across all sessions before the next)

```bash
python runners/run_unisensory_models.py --model-first
```

#### Combine flags

```bash
python runners/run_unisensory_models.py --no-coupling --no-skip --metric dff
python runners/run_unisensory_models.py --models GLM_uni GLM_CG --no-skip
```

---

## Testing with a Single Session

To test the pipeline before committing to a full run, temporarily edit
`session_registry.py` to contain just one session:

```python
PPC_AVC = {
    'KO-5-1L': ['2026-03-24'],   # test session
    # 'KO-6-00': ['2026-03-23', '2026-03-25'],
}
```

Run only the fast base models first:

```bash
python runners/run_unisensory_models.py --no-coupling
```

Once results look sensible, restore the full session list (or use `git stash`
if you staged the change).

---

## Recommended First-Run Sequence

```bash
# 1. Run and validate the 3 base models
python runners/run_unisensory_models.py --no-coupling

# 2. Inspect results in notebooks, check frac_dev and R² look reasonable

# 3. Add the 3 coupling models
python runners/run_unisensory_models.py --models GLM_all_context_coupling GLM_uni_coupling GLM_CG_coupling
```

---

## Crash Recovery and Resuming

Every model script has built-in skip/resume logic — no special flags needed:

| Situation | Behaviour |
|---|---|
| Full model + all lesions exist | Session skipped entirely |
| Full model done, some lesion pkls missing | Loads saved full model, refits only the missing lesion groups |
| Full model missing | Runs entire pipeline from scratch |

Simply re-run the same command after a crash:

```bash
python runners/run_unisensory_models.py        # picks up exactly where it left off
```

To force a full re-run of everything regardless:

```bash
python runners/run_unisensory_models.py --no-skip
```

---

## Running Individual Model Scripts

Each model script can be run standalone — useful for debugging or re-running a
single model:

```bash
python GLM_all_context.py
python GLM_uni.py
python GLM_CG.py
python GLM_uni_coupling.py
python GLM_CG_coupling.py
python GLM_all_context_coupling.py

# With flags:
python GLM_uni.py --no-skip
python GLM_CG.py --metric dff
```

---

## Logs

Each run writes a timestamped log to the `logs/` subdirectory:

- **Master log** (all models): `logs/run_all_models_YYYYMMDD_HHMMSS.log`
- **Per-model log**: `logs/run_{tag}_YYYYMMDD_HHMMSS.log`

To watch a run live in a second terminal tab:

```bash
tail -f logs/run_all_models_*.log
```

---

## File Reference

| File | Purpose |
|---|---|
| `session_registry.py` | **Edit this** to add/remove sessions |
| `runners/run_unisensory_models.py` | Master runner for the unisensory and single-task models |
| `_run_single_session.py` | Internal helper (do not call directly) |
| `GLM_all_context.py` | Runner: all contexts, no coupling |
| `GLM_all_context_coupling.py` | Runner: all contexts + coupling |
| `GLM_uni.py` | Runner: unisensory only, no coupling |
| `GLM_uni_coupling.py` | Runner: unisensory only + coupling |
| `GLM_CG.py` | Runner: CG only, no coupling |
| `GLM_CG_coupling.py` | Runner: CG only + coupling |
| `session_pipeline.py` | Core library (data loading, fitting, lesions) |
| `run_sessions.py` | Builds `glm_data_stacked.pkl` (run first) |
