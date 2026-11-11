#!/bin/bash
# run_ti_td_side_joint.sh — mouse joint (weight-readout) fits, queued behind the FEF task-split run.  4 workers x 6 threads.
#   nohup bash runners/run_ti_td_side_joint.sh > logs/ti_td_side_joint_master.log 2>&1 &
GLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GLM_DIR" || exit 1
N_WORKERS=4; THREADS=6
export TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "waiting for the FEF task-split run and the last label-shuffle session ... $(date)"
while pgrep -f "GLM_single_neuron_coupling_tasksplit.py|GLM_ti_td_side_coupling.py" > /dev/null; do sleep 60; done
echo "CPU free $(date)"
ALL=$(python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from config.session_registry import PPC_AVC
print(' '.join(f'{m}/{d}' for m, dates in PPC_AVC.items() for d in dates))
PY
)
read -r -a S <<< "$ALL"
for w in $(seq 0 $((N_WORKERS - 1))); do
    sub=()
    for i in "${!S[@]}"; do if [ $((i % N_WORKERS)) -eq "$w" ]; then sub+=("${S[$i]}"); fi; done
    python models/unisensory/GLM_ti_td_side_joint.py --sessions "${sub[@]}" > "logs/titd_side_joint_w${w}.log" 2>&1 &
done
wait
echo "sessions fitted: $(grep -h '\-> .*success' logs/titd_side_joint_w*.log | wc -l)"
echo "TI/TD SIDE JOINT COMPLETE $(date)"
