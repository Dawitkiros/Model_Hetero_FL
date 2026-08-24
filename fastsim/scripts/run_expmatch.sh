#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate sapef
cd "$(dirname "$0")/.."
CONF=conf/expmatch; OUT=results/expmatch; LOG=logs/expmatch
FS="$(pwd)"
AAAI="$(cd ../flower && pwd)"
mkdir -p "$OUT/probes" "$LOG"

IFS=',' read -ra GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
NWORKERS=$(( NGPU * ${WORKERS_PER_GPU:-2} ))

Q="$LOG/queue.txt"; : > "$Q"
for n in $CONF/fsem*.toml; do basename "$n" .toml >> "$Q"; done

worker() {
  local wid=$1; local gpu=${GPU_ARR[$((wid % NGPU))]}
  while true; do
    local n; n=$(flock "$LOG/q.lock" -c "head -n1 '$Q' && sed -i '1d' '$Q'")
    [ -z "$n" ] && break
    [ -f "$OUT/${n}_finalmodel.pt" ] && continue
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 python - "$n" "$gpu" "$CONF" "$OUT" <<'PY' > "$LOG/${n}.log" 2>&1
import sys, time
sys.path.insert(0, ".")
from fastsim.config import SimConfig
from fastsim.loop import FastSim
name, gpu, conf, out = sys.argv[1:5]
cfg = SimConfig.from_toml(f"{conf}/{name}.toml")
t0 = time.time()
sim = FastSim(cfg, device=f"cuda:{gpu}", data_root="./data")
restricted = [pid for pid in range(cfg.num_clients) if sim.draw_n[pid] is not None]
print(f"[expmatch] {name} restricted={restricted} "
      f"pools={[len(sim.client_train[p]) for p in restricted]} "
      f"draws={[sim.draw_n[p] for p in restricted]} "
      f"weights={[sim.client_sizes[p] for p in restricted]}")
sim.run(progress_every=100)
sim.save(f"{out}/{name}_finalmodel.pt")
print(f"[cell] {name} done {(time.time()-t0)/60:.1f} min")
PY
    echo "[$(date +%H:%M)] w$wid done $n"
  done
}
for w in $(seq 0 $((NWORKERS - 1))); do worker "$w" & done
wait
grep -h "\[cell\]" "$LOG"/fsem*.log

: > "$Q"
for n in $CONF/fsem*.toml; do basename "$n" .toml >> "$Q"; done
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
import glob, json, re
from collections import defaultdict
from statistics import mean, stdev
cell = defaultdict(dict)
for f in glob.glob("results/expmatch/probes/*/summary.json"):
    m = re.search(r"ws_fsem_c100_p010_([uf]+\d+)_s(\d+)", f)
    if not m: continue
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    cell[m.group(1)][int(m.group(2))] = {k: mean(v) for k, v in acc.items()}
for f in glob.glob("results/surface5/probes/*ws_fs5_c100_p010_s4[2-6]/summary.json"):
    m = re.search(r"_s(4[2-6])/", f)
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    cell["u100"][int(m.group(1))] = {k: mean(v) for k, v in acc.items()}
print("=== MATCHED-OPTIMIZATION EXPOSURE (K=50 p010, shell pool restricted, steps matched; u100 = surface5 baseline) ===")
for f in glob.glob("results/expmatch/probes/*ws_fsemK200_p0025_f400_s4[2-6]/summary.json"):
    m = re.search(r"_s(4[2-6])/", f)
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    cell["K200f400"][int(m.group(1))] = {k: mean(v) for k, v in acc.items()}
for f in glob.glob("results/kscale/probes/*ws_fsK200_p0025_s4[2-6]/summary.json") + \
         glob.glob("results/expmatch/probes/*ws_fsemK200nat_p0025_s4[2-6]/summary.json"):
    m = re.search(r"_s(4[2-6])/", f)
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    cell["K200nat"][int(m.group(1))] = {k: mean(v) for k, v in acc.items()}
for u in ["u100", "uf100", "u050", "u025", "f025", "uf025", "K200nat", "K200f400"]:
    seeds = cell.get(u, {})
    if not seeds: continue
    deltas = [(v[0.25] - v[1.0]) * 100 for _, v in sorted(seeds.items())]
    full = [v[1.0] * 100 for _, v in sorted(seeds.items())]
    core = [v[0.25] * 100 for _, v in sorted(seeds.items())]
    print(f"{u}: core-full = {mean(deltas):+.2f} +/- {stdev(deltas):.2f} (n={len(deltas)}) "
          f"| full {mean(full):.2f} core {mean(core):.2f} "
          f"| per-seed {['%+.2f' % d for d in deltas]}")
PY
echo ALL_EXPMATCH_DONE
