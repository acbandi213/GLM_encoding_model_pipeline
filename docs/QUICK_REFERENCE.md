# GLM Models Quick Reference

## Model Types Summary

### Unisensory (AVC Task)
| Model | Contexts | Coupling | Use Case |
|-------|----------|----------|----------|
| `GLM_CG` | [0] CG only | No | Baseline congruent |
| `GLM_uni` | [1,2] Visual+Audio | No | **Standard unisensory** |
| `GLM_all_context` | [0,1,2] All | No | **Standard full model** |

### Multisensory Framework
| Analysis | Model | Train | Test | Use Case |
|----------|-------|-------|------|----------|
| **Priming** | `GLM_priming_uni` | Priming [1,2] | Priming [1,2] | **Baseline unisensory** |
| **Cross-Task** | `GLM_cross_task` | Priming [1,2] | VACANT [0,1,2] | **Generalization test** |
| **VACANT** | `GLM_vacant_all` | VACANT [0,1,2] | VACANT [0,1,2] | **Full multisensory** |
| **VACANT** | `GLM_vacant_conflict_only` | VACANT [1,2] | VACANT [1,2] | **Conflicts only** |

*Each has a `_coupling` variant that adds population coupling predictors*

---

## Quick Commands

### Prepare Data
```bash
python core/prepare_glm_data.py --task all
```

### Run Everything (Fastest)
```bash
# All multisensory analyses, no coupling, 3 workers
python runners/run_multisensory_models.py --analysis all \
    --no-coupling --max-workers 3 --metric deconv
```

### Run Specific Analysis
```bash
# Just priming
python runners/run_multisensory_models.py --analysis priming --metric deconv

# Just cross-task
python runners/run_multisensory_models.py --analysis cross-task --metric deconv

# Just VACANT
python runners/run_multisensory_models.py --analysis vacant --metric deconv
```

### Run Single Model
```bash
python models/multisensory/GLM_priming_uni.py --metric deconv
python models/multisensory/GLM_cross_task.py --metric deconv
python models/multisensory/GLM_vacant_all.py --metric deconv
```

### Check Results
```bash
python analysis/check_model_runs.py
```

---

## Task Differences at a Glance

| | AVC (Unisensory) | VACANT (Multisensory) |
|---|---|---|
| **Context 0** | Congruent (both aligned) | Congruent (both aligned) |
| **Context 1** | Visual only (audio absent) | **Both present, conflict** (visual relevant) |
| **Context 2** | Audio only (visual absent) | **Both present, conflict** (audio relevant) |

---

## Output Locations

### Within-Task Models
```
{BASE_DATA_PATH}/{mouse_ID}/{date}/{MODEL_DIR}/
└── glm_{MODEL_TAG}_*.pkl
```

### Cross-Task Models
```
{BASE_DATA_PATH}/{mouse_ID}/{test_date}/GLM_cross_task_results/
├── glm_cross_task_full.pkl
├── glm_cross_task_lesion_*.pkl
└── training_info.txt  # Documents which priming session was used
```

---

## Key Metrics

- **FDE** (Fractional Deviance Explained): Primary performance metric (0-1)
- **R²** (Trial-averaged): Per-trial correlation measure
- **ΔD**: Unique contribution from lesion analysis

---

## Predictor Groups (9 standard, 14 with coupling)

1. position
2. context  
3. visual_all
4. audio_all
5. choice
6. velocity
7. turn_onset
8. outcome_time
9. outcome

**With coupling** (+5):
- network_pc1 through network_pc5

---

## Import Paths (New Structure)

```python
# Core utilities
from core.session_pipeline import BASE_DATA_PATH, open_glm_stacked_data
from core.glm_class import GLM
from core.predictor_builder import GLMPredictorBuilder

# Configuration
from config.session_registry import PPC_AVC
from config.session_registry_multisensory import PPC_AVC_PRIMING, PPC_VACANT, CROSS_TASK_PAIRS
```

---

## Troubleshooting

**Import Error**: Run scripts from the repository root

**Missing Data**: Run `python core/prepare_glm_data.py --task all`

**Check Logs**: `ls -lt logs/` then `tail -f logs/run_*.log`
