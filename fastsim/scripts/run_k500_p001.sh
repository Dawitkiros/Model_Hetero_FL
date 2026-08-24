#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate sapef
cd "$(dirname "$0")/.."
CONF=conf/acfleet; OUT=results/acfleet; LOG=logs/acfleet
FS="$(pwd)"
AAAI="$(cd ../flower && pwd)"
mkdir -p "$OUT/probes" "$LOG"

IFS=',' read -ra GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}

i=0
for s in 42 43 44; do
  n="fsK500_p001_s${s}"
  gpu=${GPU_ARR[$((i % NGPU))]}
  if [ ! -f "$OUT/${n}_finalmodel.pt" ]; then
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 python - "$n" "$gpu" "$CONF" "$OUT" <<'PY' > "$LOG/${n}.log" 2>&1 &
import sys, time
sys.path.insert(0, ".")
from fastsim.config import SimConfig
from fastsim.loop import FastSim
name, gpu, conf, out = sys.argv[1:5]
cfg = SimConfig.from_toml(f"{conf}/{name}.toml")
t0 = time.time()
sim = FastSim(cfg, device=f"cuda:{gpu}", data_root="./data")
top = max(cfg.model_rates)
writers = [pid for pid in range(cfg.num_clients) if sim.rates[pid] == top]
print(f"[k500p001] {name} writers={writers} pools={[len(sim.client_train[p]) for p in writers]}")
sim.run(progress_every=100)
sim.save(f"{out}/{name}_finalmodel.pt")
print(f"[cell] {name} done {(time.time()-t0)/60:.1f} min")
PY
  fi
  i=$((i + 1))
done
wait
grep -h "\[cell\]" "$LOG"/fsK500_p001_s4[2-4].log

for s in 42 43 44; do
  n="fsK500_p001_s${s}"
  gpu=${GPU_ARR[$((s % NGPU))]}
  ls -d "$OUT/probes"/*"ws_${n}" >/dev/null 2>&1 && continue
  (cd "$AAAI" && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$gpu nice -n 19 \
    python probe/width_ccvr_frontier.py \
    --checkpoint "$FS/$OUT/${n}_finalmodel.pt" \
    --config "$FS/$CONF/${n}.toml" \
    --rates "0.25,0.375,0.5,0.625,0.75,1.0" --head-seeds 42 43 44 --workers 2 --skip-supervised \
    --output-root "$FS/$OUT/probes" --tag "ws_${n}") \
    > "$LOG/${n}_probe.log" 2>&1 &
done
wait

python - <<'PY'
import glob, json, re
from collections import defaultdict
from statistics import mean, stdev
vals = {}
for f in glob.glob("results/acfleet/probes/*ws_fsK500_p001_s4[2-4]/summary.json"):
    m = re.search(r"_s(4[2-4])/", f)
    acc = defaultdict(list)
    for r in json.load(open(f)).get("rows", []):
        if r.get("R1_ccvr") is None: continue
        acc[float(r["rate"])].append(float(r["R1_ccvr"]))
    a = {k: mean(v) * 100 for k, v in acc.items()}
    vals[int(m.group(1))] = (a[1.0], a[0.25])
gaps = [c - f for f, c in (vals[s] for s in sorted(vals))]
fulls = [vals[s][0] for s in sorted(vals)]
cores = [vals[s][1] for s in sorted(vals)]
print(f"K500 |F|=5 (share 1%): core-full = {mean(gaps):+.2f} +/- {stdev(gaps):.2f} (n={len(gaps)}) "
      f"| full {mean(fulls):.2f} core {mean(cores):.2f} "
      f"| per-seed {['%+.2f' % g for g in gaps]}")
PY
echo K500_P001_DONE
