#!/bin/bash
# ==============================================================================
# Phase 3 — context-split coupling sweep, all 40 registry sessions x 3 masks.
#
#   context : visual vs audio                -> the real test
#   sham    : random trials within context   -> measures the group-lasso penalty
#                                               handicap (the null for split-shared
#                                               is NOT zero: ~-0.015 on the pilot)
#   time    : early vs late within context   -> measures session drift, so a
#                                               positive context effect can be told
#                                               apart from non-stationarity
#                                               (sessions are context-BLOCKED, ABAB)
#
#   true context effect = csc_context - csc_sham
#   true time effect    = csc_time    - csc_sham
#
# ~44 min/session/mask => ~29 h per mask => ~88 h total.
# skip_existing is ON, so this is safe to kill and re-run: completed sessions are
# skipped and it picks up where it left off. Failed sessions delete their own
# unverified _full.pkl, so a re-run genuinely redoes them.
#
# Usage:  nohup bash runners/run_phase3_split_coupling.sh > logs/phase3_driver.log 2>&1 &
# Watch:  tail -f logs/phase3_driver.log
#         grep -c '\[success\]' logs/phase3_*.log
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

echo "=================================================================="
echo "PHASE 3 — split coupling sweep   started $(date)"
echo "  masks: context, sham, time | 40 sessions each | ~88 h total"
echo "=================================================================="

for MASK in context sham time; do
    echo ""
    echo "### mask=$MASK  starting $(date)"
    python models/unisensory/GLM_uni_coupling_split.py --mask "$MASK" \
        > "logs/phase3_${MASK}.log" 2>&1
    rc=$?
    ok=$(grep -c '\[success\]' "logs/phase3_${MASK}.log" 2>/dev/null || echo 0)
    sk=$(grep -c '\[skipped\]' "logs/phase3_${MASK}.log" 2>/dev/null || echo 0)
    er=$(grep -c '\[error\]'   "logs/phase3_${MASK}.log" 2>/dev/null || echo 0)
    echo "### mask=$MASK  finished $(date)  rc=$rc  ✓$ok –$sk ✗$er"
    if [ "$er" != "0" ]; then
        echo "### mask=$MASK  ERRORS — offending sessions:"
        grep '\[error\]' "logs/phase3_${MASK}.log" | head -20
    fi
done

echo ""
echo "=================================================================="
echo "PHASE 3 COMPLETE  $(date)"
for MASK in context sham time; do
    n=$(ls -d "$DRIVE"/*/*/GLM_uni_coupling_${MASK}_results 2>/dev/null | wc -l | tr -d ' ')
    [ "$MASK" = "context" ] && n=$(ls -d "$DRIVE"/*/*/GLM_uni_coupling_split_results 2>/dev/null | wc -l | tr -d ' ')
    echo "  $MASK: $n sessions have a results dir"
done
echo "  next: python analysis/split_coupling_analysis.py"
echo "        analysis/notebooks/Figure_4_extra_analysis_plots.ipynb"
echo "=================================================================="
