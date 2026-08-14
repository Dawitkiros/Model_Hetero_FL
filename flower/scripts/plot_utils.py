"""Publication plotting helpers for the FLoRE / toxic-shell diagnosis figures.

Adapted from FedBlockEF/scripts/plot_utils.py: same matplotlib style
(seaborn-darkgrid, cm mathtext, fonttype-42 vector text), Wong colorblind-safe
palette with hero-method emphasis, multi-seed mean/std aggregation, and
new_axes/finalize_axes/save_and_close (PDF+PNG, no titles). The method taxonomy
and palette are retargeted to the width-heterogeneous FL setting; FAB is the hero.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    def gaussian_filter1d(arr, sigma):
        return np.asarray(arr, dtype=float)


METHOD_NAMES = {
    'homo_small':  'Homo core (r=0.25)',
    'homo_large':  'Homo full (r=1.0)',
    'heterofl':    'HeteroFL',
    'fedrolex':    'FedRolex',
    'fiarse':      'FIARSE',
    'fedlagc':     'FedLAGC',
    'oracle':      'Oracle skyline',
    'ccvr':        'CCVR',
    'fab':         'FAB',
}

# Wong colorblind-safe palette; FAB (our method) is the sole blue.
PALETTE = {
    'homo_small':  '#6B7280',  # gray baseline (core floor)
    'homo_large':  '#000000',  # black ceiling
    'heterofl':    '#E69F00',  # amber
    'fedrolex':    '#CC79A7',  # reddish purple
    'fiarse':      '#D55E00',  # vermilion
    'fedlagc':     '#56B4E9',  # sky (cited, not run)
    'oracle':      '#009E73',  # green
    'ccvr':        '#8c564b',  # brown
    'fab':         '#0072B2',  # blue, our method (sole blue)
}

MARKERS = {
    'homo_small':  None,
    'homo_large':  None,
    'heterofl':    'o',
    'fedrolex':    '^',
    'fiarse':      's',
    'fedlagc':     'v',
    'oracle':      'D',
    'ccvr':        'h',
    'fab':         '*',
}

HERO_METHOD = 'fab'
METHOD_ORDER = ('homo_small homo_large heterofl fedrolex fiarse fedlagc oracle '
                'ccvr fab').split()


def line_width(method: str) -> float:
    return 2.6 if method == HERO_METHOD else 1.4


def z_order(method: str) -> int:
    return 5 if method == HERO_METHOD else 2


def method_label(token: str) -> str:
    return METHOD_NAMES.get(token, token)


def method_color(token: str) -> str | None:
    return PALETTE.get(token)


def method_marker(token: str) -> str | None:
    return MARKERS.get(token)


def method_sort_key(token: str) -> int:
    return METHOD_ORDER.index(token) if token in METHOD_ORDER else len(METHOD_ORDER)


def apply_plot_style() -> None:
    """Apply the publication-quality matplotlib style used across the papers."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    })


def new_axes(figsize: tuple[float, float] = (3.6, 2.5)):
    return plt.subplots(figsize=figsize)


def finalize_axes(
    ax,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    dedup_legend: bool | None = True,
    legend_loc: str | None = 'best',
    legend_bbox_to_anchor=None,
    legend_ncol: int = 1,
    show_legend: bool = True,
) -> None:
    """Apply axis labels, grid, and (deduplicated) legend. Titles are omitted by
    convention — the LaTeX caption carries the title."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if not show_legend:
        return
    handles, labels = ax.get_legend_handles_labels()
    if dedup_legend:
        labels_to_handles = dict(zip(labels, handles))
        handles, labels = list(labels_to_handles.values()), list(labels_to_handles.keys())
    if handles:
        ax.legend(handles, labels, frameon=False, loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol)


def save_and_close(fig, out_dir: Path, name: str) -> Path:
    """Save a figure as PDF and PNG, then close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out_dir / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] wrote {out_dir / f"{name}.pdf"}')
    return out_dir / f'{name}.pdf'


def gaussian_smoothing(values, sigma: float | None = 5.0) -> np.ndarray:
    return gaussian_filter1d(np.asarray(values, dtype=float), sigma=sigma)


def seed_mean_std(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and (ddof=1) std across the first axis (seeds)."""
    matrix = np.asarray(matrix, dtype=float)
    mean = np.nanmean(matrix, axis=0)
    std = (np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1
           else np.zeros(matrix.shape[1]))
    return mean, std
