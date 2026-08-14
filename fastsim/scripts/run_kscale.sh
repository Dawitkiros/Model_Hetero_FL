#!/bin/bash
# K-scaling fleet: does the sparse-regime law depend on the full-client FRACTION or COUNT?
# 18 runs: K in {100,200} x {fixed-fraction p010/p040, fixed-count N=5} x seeds 42-44,
# compared against the existing K=50 surface5 rows. Train, probe, aggregate.
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate sapef
cd "$(dirname "$0")/.."
CONF=${CONF:-conf/kscale}; OUT=${OUT:-results/kscale}; LOG=${LOG:-logs/kscale}
FS="$(pwd)"
AAAI="$(cd ../flower && pwd)"
mkdir -p "$OUT/probes" "$LOG"

IFS=',' read -ra GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
NWORKERS=$(( NGPU * ${WORKERS_PER_GPU:-3} ))

Q="$LOG/queue.txt"; : > "$Q"
for n in $CONF/fs*.toml; do basename "$n" .toml >> "$Q"; done

worker() {
  local wid=$1; local gpu=${GPU_ARR[$((wid % NGPU))]}
  while true; do
    local n; n=$(flock "$LOG/q.lock" -c "head -n1 '$Q' && sed -i '1d' '$Q'")
    [ -z "$n" ] && break
    [ -f "$OUT/${n}_finalmodel.pt" ] && continue
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nice -n 19 python - "$n" "$gpu" "$CONF" "$OUT" <<'PY' > "$LOG/${n}.log" 2>&1
import sys, time
sys.path.insert(0, ".")
from fastsim.config import SimConfig
from fastsim.loop import FastSim
name, gpu, conf, out = sys.argv[1:5]
cfg = SimConfig.from_toml(f"{conf}/{name}.toml")
t0 = time.time()
sim = FastSim(cfg, device=f"cuda:{gpu}", data_root="./data")
sim.run(progress_every=100)
sim.save(f"{out}/{name}_finalmodel.pt")
print(f"[cell] {name} done {(time.time()-t0)/60:.1f} min")
PY
    echo "[$(date +%H:%M)] w$wid done $n"
  done
}
for w in $(seq 0 $((NWORKERS - 1))); do worker "$w" & done
wait
grep -h "\[cell\]" "$LOG"/fs*.log

: > "$Q"
for n in $CONF/fs*.toml; do basename "$n" .toml >> "$Q"; done
pworker() {
  local wid=$1; local gpu=${GPU_ARR[$((wid % NGPU))]}
  while true; do
    local n; n=$(flock "$LOG/q.lock" -c "head -n1 '$Q' && sed -i '1d' '$Q'")
    [ -z "$n" ] && break
    ls -d "$OUT/probes"/*"ws_${n}" >/dev/null 2>&1 && continue
    (cd "$AAAI" && OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=$gpu nice -n 19 \
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
import glob, json, re
from collections import defaultdict
from statistics import mean, stdev
cell = defaultdict(lambda: defaultdict(dict))
for f in glob.glob("results/kscale/probes/*/summary.json"):
    m = re.search(r"ws_fsK(\d+)_(p\d+)_s(\d+)", f)
    if not m: continue
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    cell[(int(m.group(1)), m.group(2))][int(m.group(3))] = {k: mean(v) for k, v in acc.items()}
for f in glob.glob("results/surface5/probes/*/summary.json"):
    m = re.search(r"ws_fs5_c100_(p010|p040)_s(4[234])/", f)
    if not m: continue
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    cell[(50, m.group(1))][int(m.group(2))] = {k: mean(v) for k, v in acc.items()}
NFULL = {(50, "p010"): 5, (50, "p040"): 20, (100, "p005"): 5, (100, "p010"): 10,
         (100, "p040"): 40, (200, "p0025"): 5, (200, "p010"): 20, (200, "p040"): 80}
print("=== K-SCALING: fraction vs count (R1 core+CCVR %, n=3 seeds; K=50 rows from surface5 s42-44) ===")
hdr = f"{'K':>4} {'p_full':>7} {'N_full':>6} " + " ".join(f"{r:>6g}" for r in [0.25, 0.375, 0.5, 0.625, 0.75, 1.0]) + f" {'argmax':>7} {'core-full':>9}"
print(hdr)
for (K, p), seeds in sorted(cell.items()):
    ms = {r: mean(v[r] for v in seeds.values()) * 100 for r in [0.25, 0.375, 0.5, 0.625, 0.75, 1.0]}
    deltas = [(v[0.25] - v[1.0]) * 100 for v in seeds.values()]
    am = max(ms, key=ms.get)
    print(f"{K:>4} {p:>7} {NFULL[(K, p)]:>6} " + " ".join(f"{ms[r]:6.2f}" for r in [0.25, 0.375, 0.5, 0.625, 0.75, 1.0])
          + f" {am:>7g} {mean(deltas):+6.2f}+/-{stdev(deltas):.2f}")
PY
echo DONE > "$LOG/kscale.marker"
