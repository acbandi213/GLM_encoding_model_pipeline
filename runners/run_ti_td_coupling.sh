#!/bin/bash
# run_ti_td_coupling.sh
# TI/TD subpopulation coupling, both species, one detached overnight job.
#
#   1. mouse  GLM_ti_td_split_coupling   split 0            (the headline fit)
#   2. mouse  GLM_ti_td_split_coupling   --null split 0     (label-shuffle control)
#   3. FEF    GLM_ti_td_loo_coupling     --all              (per-target leave-one-out)
#   4. mouse  GLM_ti_td_split_coupling   splits 1, 2        (independent neuron splits)
#
# Mouse stages: 4 workers x 6 threads = 24 cores, ~14 GB per worker.
# FEF stage   : 6 probe subprocesses x 4 threads (its own driver).
# Every stage skips finished sessions/probes, so re-running resumes.
#
# Launch detached:
#   nohup bash runners/run_ti_td_coupling.sh > logs/ti_td_coupling_master.log 2>&1 &
# Watch:
#   tail -f logs/ti_td_coupling_master.log
#   grep -h "success\|ERROR\|EXCLUDE" logs/titd_*_w*.log

GLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GLM_DIR" || exit 1
mkdir -p logs

N_WORKERS=4
THREADS=6            # 4 x 6 = 24 = core count

export TF_NUM_INTRAOP_THREADS=$THREADS
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

ALL_SESSIONS=$(python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from config.session_registry import PPC_AVC
print(' '.join(f'{m}/{d}' for m, dates in PPC_AVC.items() for d in dates))
PY
)
read -r -a SESSIONS <<< "$ALL_SESSIONS"

run_mouse_stage () {            # $1 = tag for the log files, rest = extra CLI flags
    local tag=$1; shift
    echo "### mouse stage '$tag' ($*) started $(date)"
    for w in $(seq 0 $((N_WORKERS - 1))); do
        subset=()
        for i in "${!SESSIONS[@]}"; do
            if [ $((i % N_WORKERS)) -eq "$w" ]; then subset+=("${SESSIONS[$i]}"); fi
        done
        python models/unisensory/GLM_ti_td_split_coupling.py "$@" --sessions "${subset[@]}" \
            > "logs/titd_${tag}_w${w}.log" 2>&1 &
    done
    wait
    echo "### mouse stage '$tag' finished $(date)"
}

echo "=================================================================="
echo "TI/TD COUPLING — started $(date) | ${#SESSIONS[@]} mouse sessions"
echo "=================================================================="

run_mouse_stage split0 --split 0
run_mouse_stage null0  --split 0 --null

echo "### FEF leave-one-out started $(date)"
python NHP_FEF/FEF_v2/models/GLM_ti_td_loo_coupling.py --all --min-per-class 3 --int-targets \
    --workers 6 --tf-threads 4
echo "### FEF leave-one-out finished $(date)"

run_mouse_stage split1 --split 1
run_mouse_stage split2 --split 2

echo "=================================================================="
echo "TI/TD COUPLING COMPLETE  $(date)"
echo "=================================================================="
