"""Aggregate the depth-axis campaign across seeds into a 3-seed table.

Per-seed statistic is the last-50-round mean (rounds > 450), matching the gate
protocol; the seed-level spread is then the std over per-seed means.
"""
import argparse
import csv
import glob
import os
import statistics as st

CELLS = [
    ("depthfl_p010_kdon", "depthfl p010 kd-on"),
    ("depthfl_p010_kdoff", "depthfl p010 kd-off"),
    ("depthfl_p030", "depthfl p030 kd-on"),
    ("depthfl_p060", "depthfl p060 kd-on"),
    ("fjord_p010_kdon", "fjord p010 kd-on"),
    ("fjord_p010_kdoff", "fjord p010 kd-off"),
]


def tail_mean(path, col, window=50):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["round"]), float(r[col])))
            except (KeyError, ValueError):
                continue
    if not rows:
        return None
    last = max(r for r, _ in rows)
    tail = [v for r, v in rows if r > last - window]
    return st.mean(tail) if tail else None


def seed_paths(d, cell):
    out = {}
    base = os.path.join(d, f"{cell}.csv")
    if os.path.exists(base):
        out[42] = base
    for p in sorted(glob.glob(os.path.join(d, f"{cell}_s*.csv"))):
        s = os.path.basename(p).rsplit("_s", 1)[1].split(".")[0]
        if s.isdigit():
            out[int(s)] = p
    return out


def fmt(vals):
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:+.4f} (n=1)"
    return f"{st.mean(vals):+.4f}+-{st.stdev(vals):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/depth_axis_fastsim_opt")
    a = ap.parse_args()

    print(f"{'cell':22s} {'n':>2s} {'acc_full':>18s} {'acc_core':>18s} {'delta (seed-level)':>20s}   per-seed delta")
    print("-" * 118)
    summary = {}
    for cell, label in CELLS:
        paths = seed_paths(a.dir, cell)
        per = {}
        for s, p in sorted(paths.items()):
            d = tail_mean(p, "delta_acc")
            if d is None:
                f_, c_ = tail_mean(p, "acc_full"), tail_mean(p, "acc_core")
                d = None if f_ is None or c_ is None else f_ - c_
            if d is not None:
                per[s] = (tail_mean(p, "acc_full"), tail_mean(p, "acc_core"), d)
        if not per:
            print(f"{label:22s}  0 {'(no data)':>18s}")
            continue
        full = [v[0] for v in per.values() if v[0] is not None]
        core = [v[1] for v in per.values() if v[1] is not None]
        delt = [v[2] for v in per.values()]
        summary[cell] = delt
        detail = " ".join(f"s{s}:{v[2]:+.3f}" for s, v in sorted(per.items()))
        print(f"{label:22s} {len(delt):2d} {fmt(full):>18s} {fmt(core):>18s} {fmt(delt):>20s}   {detail}")

    print()
    print("Fleet checks (seed-level means):")
    try:
        a10 = st.mean(summary["depthfl_p010_kdon"])
        a30 = st.mean(summary["depthfl_p030"])
        a60 = st.mean(summary["depthfl_p060"])
        print(f"  depthfl volume monotonicity p010<p030<p060: {a10:+.4f} < {a30:+.4f} < {a60:+.4f} -> {a10 < a30 < a60}")
    except KeyError:
        print("  depthfl monotonicity: incomplete")
    for on, off, name in [("depthfl_p010_kdon", "depthfl_p010_kdoff", "depthfl"),
                          ("fjord_p010_kdon", "fjord_p010_kdoff", "fjord")]:
        if on in summary and off in summary:
            von, voff = summary[on], summary[off]
            diff = st.mean(von) - st.mean(voff)
            pooled = max(1e-9, (st.pstdev(von) + st.pstdev(voff)) / 2)
            print(f"  {name} kd-on minus kd-off: {diff:+.4f} (pooled seed sd {pooled:.4f}, ratio {abs(diff)/pooled:.1f}x)"
                  f" -> {'kd harmful' if diff < 0 else 'kd helpful'}, {'separable' if abs(diff) > 2*pooled else 'WITHIN NOISE'}")


if __name__ == "__main__":
    main()
