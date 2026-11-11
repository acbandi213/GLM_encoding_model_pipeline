#!/bin/bash
# ==============================================================================
# Phase 3 — split coupling sweep, PARALLEL across 3 workers.
#
# Machine: Apple M2 Ultra, 24 cores (16 perf + 8 efficiency), 128 GB.
# A single fit uses ~13.6 cores and ~14 GB, so RAM is nowhere near the limit and
# ~10 cores sat idle in the sequential sweep. Three workers x 8 intra-op threads
# = 24 threads, matched to the core count, at ~42 GB total.
#
# Work is partitioned by SESSION (stride), not by mask: each worker runs ALL
# THREE masks over its own session subset. Partitioning by mask would be badly
# unbalanced (context has 24 sessions left, sham and time have 40 each), so the
# context worker would idle while the others finished.
#
# Workers write to disjoint (session, mask) result dirs, so there is no write
# race. skip_existing means the 16 context sessions already done are skipped.
#
# Usage:  nohup bash runners/run_phase3_parallel.sh > logs/phase3_parallel.log 2>&1 &
# Watch:  tail -f logs/phase3_w*.log
#         python testing/verify_split_results.py
# Stop:   pkill -f run_phase3_parallel.sh ; pkill -f GLM_uni_coupling_split.py
#         then ALWAYS: python testing/verify_split_results.py --delete-bad
#         (fit_glm_single_session writes _full.pkl before the runner's asserts
#          can run, so a SIGKILL in that window can leave an unverified pkl that
#          existence-only skip logic would honour forever.)
# ==============================================================================
set -u

GLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GLM_DIR" || exit 1
mkdir -p logs

DRIVE="${GLM_BASE_DATA_PATH:-/Volumes/Akhil Data/Akhil/ProcessedData}"
if [ ! -d "$DRIVE" ]; then
    echo "✗ FATAL: results drive not mounted at $DRIVE — aborting before any work."
    exit 1
fi

N_WORKERS=3
THREADS=8            # 3 x 8 = 24 = core count

# Keep each worker inside its thread budget; unset, TF grabs ~14 cores per proc
# and 3 procs would oversubscribe 24 cores by ~1.7x and thrash.
export TF_NUM_INTRAOP_THREADS=$THREADS
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

echo "=================================================================="
echo "PHASE 3 PARALLEL — started $(date)"
echo "  $N_WORKERS workers x $THREADS threads on 24 cores | masks: context, sham, time"
echo "=================================================================="

# Stride-partition the registry so each worker gets a mix of large and small
# sessions (they are listed roughly by mouse, and session size varies a lot).
ALL_SESSIONS=$(python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from config.session_registry import PPC_AVC
print(' '.join(f'{m}/{d}' for m, dates in PPC_AVC.items() for d in dates))
PY
)
read -r -a SESSIONS <<< "$ALL_SESSIONS"
echo "  registry: ${#SESSIONS[@]} sessions"

for w in $(seq 0 $((N_WORKERS - 1))); do
    subset=()
    for i in "${!SESSIONS[@]}"; do
        if [ $((i % N_WORKERS)) -eq "$w" ]; then subset+=("${SESSIONS[$i]}"); fi
    done
    echo "  worker $w: ${#subset[@]} sessions -> ${subset[*]}"
    (
        for MASK in context sham time; do
            echo "### worker $w  mask=$MASK  starting $(date)"
            python models/unisensory/GLM_uni_coupling_split.py \
                --mask "$MASK" --sessions "${subset[@]}"
            echo "### worker $w  mask=$MASK  finished $(date)  rc=$?"
        done
        echo "### worker $w  ALL MASKS DONE $(date)"
    ) > "logs/phase3_w${w}.log" 2>&1 &
done

echo "  launched $N_WORKERS workers; waiting ..."
wait

echo ""
echo "=================================================================="
echo "PHASE 3 PARALLEL COMPLETE  $(date)"
echo "  verifying every result before anyone reads it ..."
python testing/verify_split_results.py
echo "  next: python analysis/split_coupling_analysis.py"
echo "        analysis/notebooks/Figure_4_extra_analysis_plots.ipynb"
echo "=================================================================="
