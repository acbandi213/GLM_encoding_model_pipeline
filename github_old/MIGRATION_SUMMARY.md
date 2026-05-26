# GLM_encoding_model Reorganization Summary

**Date**: May 25, 2026  
**Status**: ✅ COMPLETE - New structure is ready to use!

## What Changed

The GLM_encoding_model directory has been reorganized into a cleaner, more maintainable structure:

### New Directory Structure

```
GLM_encoding_model/
├── core/                           # Core utilities & shared functions
│   ├── glm_class.py
│   ├── predictor_builder.py
│   ├── session_pipeline.py
│   └── datasets.py
│
├── config/                         # Configuration files
│   ├── session_registry.py
│   └── session_registry_multisensory.py
│
├── models/
│   ├── unisensory/                # Original AVC models
│   │   ├── GLM_CG.py
│   │   ├── GLM_CG_coupling.py
│   │   ├── GLM_all_context.py
│   │   ├── GLM_all_context_coupling.py
│   │   ├── GLM_uni.py
│   │   └── GLM_uni_coupling.py
│   │
│   └── multisensory/              # Priming, VACANT, cross-task models
│       ├── GLM_priming_uni.py
│       ├── GLM_priming_uni_coupling.py
│       ├── GLM_vacant_all.py
│       ├── GLM_vacant_all_coupling.py
│       ├── GLM_vacant_conflict_only.py
│       ├── GLM_vacant_conflict_only_coupling.py
│       ├── GLM_cross_task.py
│       └── GLM_cross_task_coupling.py
│
├── runners/                        # Master runner scripts
│   ├── run_all_models.py
│   ├── run_multisensory_models.py
│   ├── run_sessions.py
│   ├── run_sessions_model.py
│   └── _run_single_session.py
│
├── analysis/                       # Analysis code
│   ├── check_model_runs.py
│   ├── run_model_comparison.py
│   ├── prepare_glm_data.py
│   └── notebooks/                 # Jupyter notebooks
│       ├── GLM_analyses_paper_figures.ipynb
│       ├── Model_1_analysis.ipynb
│       ├── Model_1_fit_all_sessions.ipynb
│       ├── Model_1_predictor_contributions.ipynb
│       ├── Model_analysis_unisensory.ipynb
│       ├── PPC_encoding_analysis.ipynb
│       ├── Q1_Population_Structure.ipynb
│       ├── Q2_Context_Invariance.ipynb
│       └── Simulated_neurons_model_updated.ipynb
│
├── testing/                        # Testing & simulation
│   ├── simulated_neurons.py
│   └── synthetic_session.py
│
├── docs/                           # Documentation
│   ├── README.md
│   ├── RUNNING.md
│   └── MULTISENSORY_README.md
│
├── logs/                           # Logs (unchanged)
├── figures/                        # Figures (unchanged)
└── simulated_data/                 # Simulated data (unchanged)
```

## Migration Strategy: Copy-First Approach

**IMPORTANT**: The old files are still in place! This is a **zero-risk migration**:
- ✅ New structure has been created and updated
- ✅ Old files remain in the root directory (unchanged)
- ✅ Any currently running models will continue to work
- ✅ You can test the new structure before removing old files

## How to Use the New Structure

### For New Model Runs

Use the scripts in the **`runners/`** folder:

```bash
# Multisensory analyses (from anywhere)
cd "/Users/akhilbandi/Library/Mobile Documents/com~apple~CloudDocs/Documents/Portfolio/code_projects/Context Paper Code/GLM_encoding_model"
python runners/run_multisensory_models.py --analysis all

# Unisensory analyses
python runners/run_all_models.py --model GLM_uni --metric deconv

# Individual models (from anywhere, using full path)
python models/multisensory/GLM_priming_uni.py --metric deconv
python models/unisensory/GLM_uni.py --metric deconv
```

### For Notebooks

Notebooks are now in `analysis/notebooks/`. Add this at the top of each notebook to ensure imports work:

```python
import os
import sys

# Add GLM_encoding_model root to path (go up 2 levels from notebooks/)
GLM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if GLM_ROOT not in sys.path:
    sys.path.insert(0, GLM_ROOT)

# Now imports work
from core.glm_class import GLM
from config.session_registry import PPC_AVC
from core.session_pipeline import open_glm_stacked_data
```

### For Analysis Scripts

```python
from core.session_pipeline import BASE_DATA_PATH, open_glm_stacked_data
from config.session_registry import PPC_AVC
from core.glm_class import GLM
```

## Key Technical Changes

### 1. Path Resolution
All scripts now correctly resolve the GLM_encoding_model root:
- Model scripts: Go up 2 levels (`models/multisensory/` → `models/` → root)
- Runner scripts: Go up 1 level (`runners/` → root)
- Analysis scripts: Should be run from root or update path accordingly

### 2. Import Changes
Old imports:
```python
from session_pipeline import BASE_DATA_PATH
from session_registry import PPC_AVC
from glm_class import GLM
```

New imports:
```python
from core.session_pipeline import BASE_DATA_PATH
from config.session_registry import PPC_AVC
from core.glm_class import GLM
```

### 3. Model Script Paths
Runner scripts now reference models with their full subdirectory path:
```python
# In MULTISENSORY_MODELS:
script='models/multisensory/GLM_priming_uni.py'

# In ALL_MODELS:
script='models/unisensory/GLM_uni.py'
```

## Testing the New Structure

### Quick Test
```bash
cd "/Users/akhilbandi/Library/Mobile Documents/com~apple~CloudDocs/Documents/Portfolio/code_projects/Context Paper Code/GLM_encoding_model"

# List available models (should work without errors)
python runners/run_multisensory_models.py --list
python runners/run_all_models.py --list

# Run a single session to verify (replace with your data)
python models/multisensory/GLM_priming_uni.py --metric deconv
```

### Full Test
Pick one session from each model type and run:
```bash
# Test priming
python runners/run_multisensory_models.py --analysis priming --models GLM_priming_uni --metric deconv

# Test VACANT
python runners/run_multisensory_models.py --analysis vacant --models GLM_vacant_all --metric deconv

# Test cross-task
python runners/run_multisensory_models.py --analysis cross-task --metric deconv
```

## Cleanup Plan (When Ready)

Once you've tested thoroughly and are confident:

### Option A: Archive Old Files
```bash
cd GLM_encoding_model/
mkdir _old_structure_backup
mv GLM_*.py run_*.py session_*.py glm_class.py predictor_builder.py \
   datasets.py check_model_runs.py simulated_neurons.py synthetic_session.py \
   *.ipynb *.md _old_structure_backup/
```

### Option B: Delete Old Files
```bash
cd GLM_encoding_model/
rm -i GLM_*.py run_*.py session_*.py glm_class.py predictor_builder.py \
      datasets.py check_model_runs.py simulated_neurons.py synthetic_session.py \
      *.ipynb README.md RUNNING.md MULTISENSORY_README.md
```

**Note**: Don't delete logs/, figures/, or simulated_data/ folders!

## Benefits

✅ **Clarity**: Clear separation of models, utilities, config, and analysis  
✅ **Maintainability**: Easy to find and update specific components  
✅ **Scalability**: Easy to add new model types or analysis scripts  
✅ **Organization**: Related files are grouped together  
✅ **Zero Disruption**: Old structure still works during transition  

## Rollback Plan

If anything goes wrong:
1. The old files are still in the root directory
2. Simply use the old files and ignore the new folders
3. Or delete the new folders: `rm -rf core/ config/ models/ runners/ analysis/notebooks/ testing/ docs/`

## Questions or Issues?

If you encounter any problems:
1. Check that you're running scripts from the correct location
2. Verify imports use the new `core.` and `config.` prefixes
3. Ensure Python can find the GLM_encoding_model root in sys.path
4. Check this file for the correct usage examples

---

**Next Steps**:
1. Test the new structure with a few model runs
2. Update any custom scripts or notebooks you have
3. Once confident, archive or delete the old files
4. Update any external documentation that references the old structure
