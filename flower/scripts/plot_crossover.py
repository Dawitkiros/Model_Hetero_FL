from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

P100 = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
D100 = [2.69, -0.04, -1.33, -3.59, -6.06, -14.46]
S100 = [1.33, 1.03, 0.50, 0.90, 1.40, 1.76]
P10 = [0.10, 0.30, 0.60, 0.80]
D10 = [5.87, -0.02, -3.22, -16.20]
S10 = [2.51, 0.89, 0.64, 3.74]


def main() -> None:
    ap = argparse.ArgumentParser(description="core-minus-full crossover figure")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "font.family": "serif",
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.1, 1.75), constrained_layout=True)
    ax.axhline(0, color="#9a9a9a", linewidth=0.8, zorder=1)
    ax.errorbar(P100, D100, yerr=S100, color="#0072B2", linewidth=1.4, marker="o",
                markersize=3.2, capsize=1.8, elinewidth=0.8, label="CIFAR-100",
                zorder=3)
    ax.errorbar(P10, D10, yerr=S10, color="#D55E00", linewidth=1.4, marker="s",
                markersize=3.2, capsize=1.8, elinewidth=0.8, linestyle="--",
                label="CIFAR-10", zorder=3)
    ax.set_xlabel(r"full-client fraction $p_{\mathrm{full}}$")
    ax.set_ylabel(r"$\Delta_{\mathrm{dep}} = \mathrm{acc}_{\mathrm{core}} - \mathrm{acc}_{\mathrm{full}}$ (points)")
    ax.set_xticks(P100)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    fig.savefig(out_dir / "fig_crossover.pdf")
    fig.savefig(out_dir / "fig_crossover_preview.png", dpi=180)
    print(f"[ok] wrote {out_dir / 'fig_crossover.pdf'}")


if __name__ == "__main__":
    main()
