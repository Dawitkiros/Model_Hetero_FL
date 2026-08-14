from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import save_and_close

ROOT = Path(__file__).resolve().parents[1]

HETERO = {
    0.05: {
        0.10: {42: (0.2548, 0.3168)},
        0.20: {42: (0.1947, 0.3149), 43: (0.2246, 0.2742), 44: (0.2227, 0.2791)},
        0.30: {42: (0.2853, 0.2804), 43: (0.2176, 0.2443), 44: (0.3092, 0.2561)},
        0.40: {42: (0.2368, 0.2564), 43: (0.2779, 0.2338), 44: (0.3344, 0.2753)},
        0.60: {42: (0.3466, 0.3037)},
    },
    0.30: {
        0.10: {42: (0.3338, 0.4631)},
        0.20: {42: (0.4203, 0.4550)},
        0.30: {42: (0.4212, 0.4335), 43: (0.4097, 0.4319), 44: (0.4286, 0.4594)},
        0.40: {42: (0.4900, 0.4280), 43: (0.4532, 0.4462), 44: (0.4707, 0.4360)},
        0.60: {42: (0.5231, 0.4560)},
    },
    0.50: {
        0.10: {42: (0.3671, 0.4980)},
        0.20: {42: (0.4689, 0.5030)},
        0.30: {42: (0.4685, 0.4825), 43: (0.4882, 0.4804), 44: (0.4915, 0.4822)},
        0.40: {42: (0.5028, 0.4710), 43: (0.4791, 0.4744), 44: (0.5015, 0.4870)},
        0.60: {42: (0.5357, 0.4759)},
    },
}

HOMO_CORE = {0.05: 0.2793, 0.30: 0.4676, 0.50: 0.4994}

ALPHA_COLOR = {0.05: plt.cm.viridis(0.12), 0.30: plt.cm.viridis(0.52),
               0.50: plt.cm.viridis(0.80)}
ALPHA_MARKER = {0.05: "o", 0.30: "s", 0.50: "^"}
TOXIC = "#D55E00"


def apply_crossover_style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "serif",
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 0.6,
    })


def style_axis(ax, xlabel, ylabel, legend_loc):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc, handlelength=1.8)


def agg(seeds):
    af = [v[0] for v in seeds.values()]
    ac = [v[1] for v in seeds.values()]
    dd = [a - c for a, c in zip(af, ac)]
    n = len(dd)

    def ms(x):
        return st.mean(x), (st.stdev(x) if n > 1 else 0.0)

    return n, ms(af), ms(ac), ms(dd)


def crossing(ps, dmean):
    for i in range(len(ps) - 1):
        if dmean[i] < 0 <= dmean[i + 1]:
            t = -dmean[i] / (dmean[i + 1] - dmean[i])
            return ps[i] + t * (ps[i + 1] - ps[i])
    return None


def panel_transition(ax):
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
    series, crossings = {}, []
    lo, hi = 0.0, 0.0
    for a in sorted(HETERO):
        ps = sorted(HETERO[a])
        dm, de = [], []
        for p in ps:
            _, _, _, (d, s) = agg(HETERO[a][p])
            dm.append(-d * 100)
            de.append(s * 100)
        series[a] = (ps, dm, de)
        lo, hi = min(lo, min(dm)), max(hi, max(dm))
        pc = crossing(ps, [-d / 100 for d in dm])
        if pc is not None:
            crossings.append(pc)
    if crossings:
        ax.annotate(fr"boundary $p^*\approx{sum(crossings) / len(crossings):.2f}$",
                    xy=(sum(crossings) / len(crossings), hi + 1.0), ha="center",
                    va="bottom", fontsize=7, color="#333333")
    for a in sorted(HETERO):
        ps, dm, de = series[a]
        ax.errorbar(ps, dm, yerr=de, marker=ALPHA_MARKER[a], ms=5, lw=1.8,
                    color=ALPHA_COLOR[a], capsize=2, label=fr"$\alpha={a:g}$",
                    zorder=3)
    ax.set_ylim(lo - 1.5, hi + 2.6)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    style_axis(ax, xlabel=r"Full-client fraction $p_{\mathrm{full}}$",
               ylabel=r"$\Delta_{\mathrm{dep}} = \mathrm{acc}_{\mathrm{core}} - \mathrm{acc}_{\mathrm{full}}$ (points)",
               legend_loc="upper right")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", fontsize=9,
            fontweight="bold")


def panel_scissors(ax, a=0.30):
    ps = sorted(HETERO[a])
    afm, afe, acm, ace, dm = [], [], [], [], []
    for p in ps:
        _, (af, afs), (ac, acs), (d, _) = agg(HETERO[a][p])
        afm.append(af * 100); afe.append(afs * 100)
        acm.append(ac * 100); ace.append(acs * 100)
        dm.append(d)
    pc = crossing(ps, dm)
    if pc is not None:
        ax.axvline(pc, color=TOXIC, lw=1.0, ls="-", alpha=0.6, zorder=1)
    ax.axhline(HOMO_CORE[a] * 100, color="black", lw=1.0, ls=":",
               label="homo core ($r{=}0.25$)", zorder=2)
    ax.errorbar(ps, afm, yerr=afe, marker="o", ms=5, lw=1.8, color="#0072B2",
                capsize=2, label="full model ($r{=}1.0$)", zorder=4)
    ax.errorbar(ps, acm, yerr=ace, marker="s", ms=5, lw=1.8, ls="--",
                color="#6B7280", capsize=2, label="extracted core ($r{=}0.25$)",
                zorder=3)
    if pc is not None:
        ax.annotate(fr"boundary $p^*\approx{pc:.2f}$", xy=(pc, HOMO_CORE[a] * 100),
                    xytext=(pc + 0.02, HOMO_CORE[a] * 100 - 6), fontsize=7,
                    color=TOXIC, ha="left")
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    style_axis(ax, xlabel=r"Full-client fraction $p_{\mathrm{full}}$",
               ylabel="Test accuracy (%)", legend_loc="lower right")
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top", fontsize=9,
            fontweight="bold")
    ax.text(0.04, 0.85, fr"$\alpha={a:g}$", transform=ax.transAxes, ha="left",
            va="top", fontsize=8)


def write_table(path):
    rows = []
    for a in sorted(HETERO):
        ps = sorted(HETERO[a])
        first = True
        for p in ps:
            n, (af, afs), (ac, acs), (d, ds) = agg(HETERO[a][p])

            def cell(m, s):
                return fr"${m:.3f} \pm {s:.3f}$" if n > 1 else f"${m:.3f}$"

            dsign = fr"$\mathbf{{{d:+.3f}}}$" if n > 1 and abs(d) > ds else f"${d:+.3f}$"
            acol = fr"\multirow{{{len(ps)}}}{{*}}{{${a:g}$}}" if first else ""
            rows.append(f"{acol} & ${p:g}$ & {n} & {cell(af, afs)} & "
                        f"{cell(ac, acs)} & {dsign} \\\\")
            first = False
        rows.append(r"\midrule")
    body = "\n".join(rows[:-1])
    tex = (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "$\\alpha$ & $p_{\\mathrm{full}}$ & seeds & "
        "$\\mathrm{acc}_{\\mathrm{full}}$ & $\\mathrm{acc}_{\\mathrm{core}}$ & "
        "$\\Delta$ \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Width-heterogeneous HeteroFL on CIFAR-100 across Dirichlet "
        "heterogeneity $\\alpha$ and full-client fraction $p_{\\mathrm{full}}$. "
        "$\\Delta = \\mathrm{acc}_{\\mathrm{full}} - \\mathrm{acc}_{\\mathrm{core}}$ "
        "is the gap between the full model and its own extracted rate-0.25 core; "
        "$\\Delta < 0$ marks the toxic regime. Boundary-adjacent points carry three "
        "seeds (mean $\\pm$ std); a bold $\\Delta$ is separated from zero by more "
        "than one standard deviation. The homogeneous rate-0.25 core reference is "
        "$0.279 / 0.468 / 0.499$ for $\\alpha = 0.05 / 0.30 / 0.50$. The sign of "
        "$\\Delta$ flips near $p_{\\mathrm{full}} \\approx 0.3$ for every $\\alpha$.}\n"
        "\\label{tab:alpha_sweep}\n\\end{table}\n"
    )
    Path(path).write_text(tex)
    print(f"[ok] wrote {path}")


def print_table():
    print(f"\n{'alpha':<7}{'p_full':<8}{'seeds':<7}{'acc_full':<16}"
          f"{'acc_core':<16}{'delta':<16}")
    for a in sorted(HETERO):
        for p in sorted(HETERO[a]):
            n, (af, afs), (ac, acs), (d, ds) = agg(HETERO[a][p])
            print(f"{a:<7g}{p:<8g}{n:<7}{af:.3f}+/-{afs:.3f}    "
                  f"{ac:.3f}+/-{acs:.3f}    {d:+.3f}+/-{ds:.3f}")


def main():
    ap = argparse.ArgumentParser(description="alpha-sweep phase-boundary figure + table")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    apply_crossover_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    panel_transition(axA)
    panel_scissors(axB, a=0.30)
    save_and_close(fig, out_dir, "fig_alpha_sweep")
    write_table(out_dir.parent / "tab_alpha_sweep.tex")
    print_table()


if __name__ == "__main__":
    main()
