#!/bin/bash
# run_ti_td_side_coupling.sh — mouse side-matched single-neuron coupling, one detached job.
#   0. wait for the FEF single-neuron run to release the CPU
#   1. main      : all sessions (every side x class group >= 6 neurons)       ~6.5 h
#   2. resid     : residual-source control, 10-session subset (5 mice)         ~2 h
#   3. null      : label-shuffle control, same 10-session subset               ~2.5 h
# 4 workers x 6 threads = 24 cores.  Every stage skips finished sessions, so re-running resumes.
#   nohup bash runners/run_ti_td_side_coupling.sh > logs/ti_td_side_master.log 2>&1 &
GLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GLM_DIR" || exit 1
mkdir -p logs
N_WORKERS=4; THREADS=6
export TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

echo "waiting for the FEF single-neuron run to finish ... $(date)"
while pgrep -f "GLM_single_neuron_coupling.py" > /dev/null; do sleep 60; done
echo "CPU free $(date)"

ALL=$(python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from config.session_registry import PPC_AVC
print(' '.join(f'{m}/{d}' for m, dates in PPC_AVC.items() for d in dates))
PY
)
SUBSET="IS-2-1L/2025-03-10 IS-2-1L/2025-04-08 IS-3-1R/2024-12-23 IS-3-1R/2024-12-24 JP-2-1L/2025-09-19 JP-2-1L/2025-08-26 KO-5-1L/2026-03-12 KO-5-1L/2026-03-24 KO-6-00/2026-03-25 KO-6-00/2026-03-30"

run_stage () {            # $1 = tag, $2 = session list, rest = CLI flags
    local tag=$1; local list=$2; shift 2
    read -r -a S <<< "$list"
    echo "### stage '$tag' ($*) — ${#S[@]} sessions — started $(date)"
    for w in $(seq 0 $((N_WORKERS - 1))); do
        sub=()
        for i in "${!S[@]}"; do if [ $((i % N_WORKERS)) -eq "$w" ]; then sub+=("${S[$i]}"); fi; done
        python models/unisensory/GLM_ti_td_side_coupling.py "$@" --sessions "${sub[@]}" > "logs/titd_side_${tag}_w${w}.log" 2>&1 &
    done
    wait
    echo "### stage '$tag' finished $(date)"
    grep -h "success\|EXCLUDE\|ERROR" logs/titd_side_${tag}_w*.log | grep -c "success" | xargs echo "    sessions fitted:"
}

run_stage main  "$ALL"
run_stage resid "$SUBSET" --residual-sources
run_stage null  "$SUBSET" --null
echo "TI/TD SIDE COUPLING COMPLETE $(date)"
