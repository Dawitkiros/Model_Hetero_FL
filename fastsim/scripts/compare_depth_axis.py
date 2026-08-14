"""Compare fastsim DepthFL/FjORD equivalence runs against the Flower references.

Recomputes the flwr last-50-round (rounds > 450) means/stds directly from the
recorded flwr logs, reads the fastsim per-round CSVs, and prints the PASS/FAIL
table per the equivalence gate. Never mixes the two stacks in one column.
"""

import ast
import csv
import math
import os
import re
import sys

FLWR_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "depth_axis")
FASTSIM_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "depth_axis_fastsim")

# cell -> (fastsim csv basename, flwr log basename)
CELLS = [
    ("depthfl p010 kd-on",  "depthfl_p010_kdon",  "depthfl_c100_p010_s42.log"),
    ("depthfl p010 kd-off", "depthfl_p010_kdoff", "depthfl_c100_p010_nokd_s42.log"),
    ("depthfl p030 kd-on",  "depthfl_p030",       "depthfl_c100_p030_s42.log"),
    ("depthfl p060 kd-on",  "depthfl_p060",       "depthfl_c100_p060_s42.log"),
    ("fjord p010 kd-on",    "fjord_p010_kdon",    "fjord_c100_p010_s42_r3.log"),
    ("fjord p010 kd-off",   "fjord_p010_kdoff",   "fjord_c100_p010_nokd_s42_r2.log"),
]


def stats(v):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var), n


def parse_flwr(path, lo=451, hi=500):
    full, core, delta = [], [], []
    pat = re.compile(r"fit progress:\s*\((\d+),\s*[^,]+,\s*(\{.*?\}),")
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "fit progress" not in line:
                continue
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = pat.search(line)
            if not m:
                continue
            r = int(m.group(1))
            if r < lo or r > hi:
                continue
            d = ast.literal_eval(m.group(2))
            full.append(d["acc_full"]); core.append(d["acc_core"]); delta.append(d["delta_acc"])
    return full, core, delta


def parse_fastsim(path, lo=451, hi=500):
    full, core, delta = [], [], []
    if not os.path.exists(path):
        return full, core, delta
    with open(path) as fh:
        for row in csv.DictReader(fh):
            r = int(row["round"])
            if r < lo or r > hi:
                continue
            full.append(float(row["acc_full"])); core.append(float(row["acc_core"]))
            delta.append(float(row["delta_acc"]))
    return full, core, delta


def gate(fs_delta_mean, fl_delta_mean, fl_delta_std):
    sign_ok = (fs_delta_mean < 0) == (fl_delta_mean < 0)
    tol = max(0.015, 1.0 * fl_delta_std)
    mag_ok = abs(fs_delta_mean - fl_delta_mean) <= tol
    return sign_ok, mag_ok, tol


def main():
    print("=" * 118)
    print("DepthFL / FjORD equivalence gate  (last-50-round means, rounds > 450; fastsim vs flwr, labeled)")
    print("=" * 118)
    hdr = f"{'cell':<20} | {'stack':<7} {'acc_full':>16} {'acc_core':>16} {'delta_acc':>16} | {'|d dmean|':>9} {'tol':>7} {'gate':>6}"
    print(hdr)
    print("-" * 118)
    results = {}
    for label, fs_name, fl_name in CELLS:
        ff, fc, fd = parse_flwr(os.path.join(FLWR_LOG_DIR, fl_name))
        sf, sc, sd = parse_fastsim(os.path.join(FASTSIM_DIR, fs_name + ".csv"))
        flm = tuple(stats(x) for x in (ff, fc, fd))
        fsm = tuple(stats(x) for x in (sf, sc, sd))
        results[label] = (fsm, flm)
        if fsm[2][2] == 0:
            print(f"{label:<20} | fastsim  {'PENDING (no CSV rows > 450 yet)':>50}")
            print(f"{'':<20} | flwr     {flm[0][0]:>8.4f}+-{flm[0][1]:.4f} {flm[1][0]:>8.4f}+-{flm[1][1]:.4f} {flm[2][0]:>8.4f}+-{flm[2][1]:.4f}")
            print("-" * 118)
            continue
        sign_ok, mag_ok, tol = gate(fsm[2][0], flm[2][0], flm[2][1])
        verdict = "PASS" if (sign_ok and mag_ok) else "FAIL"
        dgap = abs(fsm[2][0] - flm[2][0])
        print(f"{label:<20} | fastsim  {fsm[0][0]:>8.4f}+-{fsm[0][1]:.4f} {fsm[1][0]:>8.4f}+-{fsm[1][1]:.4f} {fsm[2][0]:>8.4f}+-{fsm[2][1]:.4f} | {dgap:>9.4f} {tol:>7.4f} {verdict:>6}")
        print(f"{'':<20} | flwr     {flm[0][0]:>8.4f}+-{flm[0][1]:.4f} {flm[1][0]:>8.4f}+-{flm[1][1]:.4f} {flm[2][0]:>8.4f}+-{flm[2][1]:.4f} |{'  sign ' + ('ok' if sign_ok else 'BAD'):>26}")
        print("-" * 118)

    def dm(label):
        fsm = results[label][0]
        return fsm[2][0] if fsm[2][2] else None

    print("\nFleet-level checks:")
    d010, d030, d060 = dm("depthfl p010 kd-on"), dm("depthfl p030 kd-on"), dm("depthfl p060 kd-on")
    if None not in (d010, d030, d060):
        mono = d010 < d030 < d060
        print(f"  depthfl delta monotone p010<p030<p060 (fastsim): "
              f"{d010:.4f} < {d030:.4f} < {d060:.4f}  -> {mono}")
    fon, foff = dm("fjord p010 kd-on"), dm("fjord p010 kd-off")
    if None not in (fon, foff):
        print(f"  fjord kd-on more negative than kd-off (fastsim, 1-seed, not a hard gate): "
              f"{fon:.4f} < {foff:.4f}  -> {fon < foff}")


if __name__ == "__main__":
    main()
