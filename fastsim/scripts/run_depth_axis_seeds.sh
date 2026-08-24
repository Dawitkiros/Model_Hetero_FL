#!/usr/bin/env bash
# Multi-seed depth-axis campaign: seeds 43,44 x 6 cells on the gated optimized path.
# Chains are balanced by measured s/round; one compiled run per GPU at a time.
set -uo pipefail

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u
conda activate mhfl
set -u
export OMP_NUM_THREADS=8

OUT=results/depth_axis_fastsim_opt
LOG=logs/depth_axis_seeds
mkdir -p "$OUT" "$LOG"

run_chain() {
  local gpu=$1; shift
  for cell in "$@"; do
    local csv="$OUT/${cell}.csv"
    if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 500 ]; then
      echo "[skip] $cell already complete"
      continue
    fi
    echo "[start] $cell on gpu $gpu at $(date +%H:%M:%S)"
    nice -n 19 python scripts/fastsim_cell.py \
      --config "conf/depth_axis/${cell}.toml" \
      --gpu "$gpu" \
      --csv "$csv" > "$LOG/${cell}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "[FAIL rc=$rc] $cell (health guard trips rc=3)"
    else
      echo "[done] $cell at $(date +%H:%M:%S)"
    fi
  done
}

run_chain 0 fjord_p010_kdon_s43 &
p0=$!
run_chain 1 fjord_p010_kdon_s44 &
p1=$!
run_chain 2 depthfl_p060_s43 depthfl_p060_s44 &
p2=$!
run_chain 3 depthfl_p030_s43 depthfl_p030_s44 &
p3=$!
run_chain 4 depthfl_p010_kdon_s43 depthfl_p010_kdon_s44 fjord_p010_kdoff_s43 &
p4=$!
run_chain 5 depthfl_p010_kdoff_s43 depthfl_p010_kdoff_s44 fjord_p010_kdoff_s44 &
p5=$!

wait $p0 $p1 $p2 $p3 $p4 $p5
echo "[campaign] all chains finished at $(date)"
