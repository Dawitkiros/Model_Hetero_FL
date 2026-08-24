#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate mhfl
cd "$(dirname "$0")/.."
FS="$(pwd)"
AAAI="$(cd ../flower && pwd)"
CONF=conf/surface5; OUT=results/surface5; LOG=logs/surface5
mkdir -p "$OUT/probes" "$LOG"

IFS=',' read -ra GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
NWORKERS=$(( NGPU * ${WORKERS_PER_GPU:-2} ))

Q="$LOG/queue.txt"; : > "$Q"
for n in $CONF/fs5_*.toml; do basename "$n" .toml >> "$Q"; done

worker() {
  local wid=$1; local gpu=${GPU_ARR[$((wid % NGPU))]}
  while true; do
    local n; n=$(flock "$LOG/q.lock" -c "head -n1 '$Q' && sed -i '1d' '$Q'")
    [ -z "$n" ] && break
    [ -f "$OUT/${n}_finalmodel.pt" ] && continue
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 python - "$n" "$gpu" <<'PY' > "$LOG/${n}.log" 2>&1
import sys, time
sys.path.insert(0, ".")
from fastsim.config import SimConfig
from fastsim.loop import FastSim
name, gpu = sys.argv[1], sys.argv[2]
cfg = SimConfig.from_toml(f"conf/surface5/{name}.toml")
t0 = time.time()
sim = FastSim(cfg, device=f"cuda:{gpu}", data_root="./data")
sim.run(progress_every=100)
sim.save(f"results/surface5/{name}_finalmodel.pt")
print(f"[surface5] {name} done {(time.time()-t0)/60:.1f} min")
PY
    echo "[$(date +%H:%M)] w$wid trained $n"
  done
}
for w in $(seq 0 $((NWORKERS - 1))); do worker "$w" & done
wait
grep -h "\[surface5\]" "$LOG"/fs5_*.log

: > "$Q"
for n in $CONF/fs5_*.toml; do basename "$n" .toml >> "$Q"; done
pworker() {
  local wid=$1; local gpu=${GPU_ARR[$((wid % NGPU))]}
  while true; do
    local n; n=$(flock "$LOG/q.lock" -c "head -n1 '$Q' && sed -i '1d' '$Q'")
    [ -z "$n" ] && break
    ls -d "$OUT/probes"/*"ws_${n}" >/dev/null 2>&1 && continue
    (cd "$AAAI" && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$gpu nice -n 19 \
      python probe/width_ccvr_frontier.py \
      --checkpoint "$FS/$OUT/${n}_finalmodel.pt" \
      --config "$FS/$CONF/${n}.toml" \
      --rates "0.25,0.375,0.5,0.625,0.75,1.0" --head-seeds 42 43 44 --workers 2 --skip-supervised \
      --output-root "$FS/$OUT/probes" --tag "ws_${n}") \
      > "$LOG/${n}_probe.log" 2>&1
  done
}
for w in $(seq 0 $((NWORKERS - 1))); do pworker "$w" & done
wait

python - <<'PY'
import glob, json
from statistics import mean, stdev
for ds, ps in [("c100", ["p010", "p020", "p030", "p040", "p060", "p080"]),
               ("c10", ["p010", "p030", "p060", "p080"])]:
    for p in ps:
        deltas = []
        for s in [42, 43, 44, 45, 46]:
            gl = glob.glob(f"results/surface5/probes/*ws_fs5_{ds}_{p}_s{s}/summary.json")
            if len(gl) != 1:
                print(f"{ds} {p} s{s}: MISSING probe"); break
            rows = json.load(open(gl[0]))["rows"]
            r025 = mean(r["R1_ccvr"] for r in rows if r["rate"] == 0.25)
            r100 = mean(r["R1_ccvr"] for r in rows if r["rate"] == 1.0)
            deltas.append((r025 - r100) * 100)
        else:
            print(f"{ds} {p}: core-full = {mean(deltas):+.2f} +/- {stdev(deltas):.2f} (n={len(deltas)})")
PY
echo "SURFACE5_DONE"
