#!/bin/bash
# DepthFL + FjORD fastsim equivalence runs: six seed-42 cells, 500 rounds each,
# per-round metrics to CSV. Four per-GPU sequential chains (clean 1-per-GPU
# timing) across GPUs 0,1,4,5 (GPUs 2,3 reserved for another user).
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate sapef
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CONF=${CONF:-conf/depth_axis}
OUT=${OUT:-results/depth_axis_fastsim}
LOG=${LOG:-logs/depth_axis_fastsim}
mkdir -p "$OUT" "$LOG"

run_chain() {
  local gpu=$1; shift
  for spec in "$@"; do
    local name=${spec%%:*}; local conf=${spec#*:}
    [ -f "$OUT/${name}.csv" ] && grep -q "^500," "$OUT/${name}.csv" 2>/dev/null && continue
    nice -n 19 python scripts/fastsim_cell.py \
      --config "conf/depth_axis/${conf}" --gpu "$gpu" \
      --csv "$OUT/${name}.csv" > "$LOG/${name}.log" 2>&1
  done
}

run_chain 0 "depthfl_p060:depthfl_p060.toml" &
c0=$!
run_chain 1 "fjord_p010_kdon:fjord_p010_kdon.toml" &
c1=$!
run_chain 4 "depthfl_p030:depthfl_p030.toml" "depthfl_p010_kdoff:depthfl_p010_kdoff.toml" &
c4=$!
run_chain 5 "depthfl_p010_kdon:depthfl_p010_kdon.toml" "fjord_p010_kdoff:fjord_p010_kdoff.toml" &
c5=$!
wait "$c0" "$c1" "$c4" "$c5"
grep -h "\[cell\]" "$LOG"/*.log
echo ALL_DONE > "$LOG/orchestrator.marker"
