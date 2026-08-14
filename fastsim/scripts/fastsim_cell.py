import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastsim.config import SimConfig
from fastsim.depth import DepthFLSim
from fastsim.loop import FastSim


def check_health(csv_path, num_rounds, num_classes, window=50):
    with open(csv_path) as fh:
        rows = [(int(r["round"]), float(r["acc_full"])) for r in csv.DictReader(fh)]
    expected = num_rounds - 1
    if len(rows) != expected:
        return f"row count {len(rows)} != expected {expected}"
    tail = [a for r, a in rows if r > num_rounds - window]
    floor = 2.0 / num_classes
    mean_tail = sum(tail) / len(tail)
    if mean_tail <= floor:
        return f"last-{window} mean acc_full {mean_tail:.4f} <= 2x chance {floor:.4f}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--data-root", default="./data")
    a = ap.parse_args()

    cfg = SimConfig.from_toml(a.config)
    device = f"cuda:{a.gpu}"
    t0 = time.time()
    if cfg.algorithm == "depthfl":
        sim = DepthFLSim(cfg, device=device, data_root=a.data_root)
    else:
        sim = FastSim(cfg, device=device, data_root=a.data_root)
    sim.run(progress_every=50, eval_every=a.eval_every, csv_path=a.csv)
    dt = time.time() - t0
    n = max(1, cfg.num_rounds - 1)
    print(f"[cell] {a.config} algo={cfg.algorithm} done in {dt:.1f}s "
          f"({dt / n:.2f} s/round) -> {a.csv}", flush=True)
    if a.eval_every == 1:
        problem = check_health(a.csv, cfg.num_rounds, cfg.num_classes)
        if problem is not None:
            print(f"[cell] HEALTH FAIL {a.csv}: {problem}", flush=True)
            sys.exit(3)
        print(f"[cell] HEALTH OK {a.csv}", flush=True)


if __name__ == "__main__":
    main()
