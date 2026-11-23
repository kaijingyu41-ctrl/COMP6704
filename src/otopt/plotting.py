"""Plotting helpers shared by the figure-generation script."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402


def setup_style() -> None:
    """Compact, print-friendly defaults for a two-column paper."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": config.FIGURE_DPI,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def save(fig: plt.Figure, name: str, out_dir: Path | None = None) -> List[Path]:
    """Save a figure as both PDF (for LaTeX) and PNG (for quick viewing)."""
    out = Path(config.FIGURES_DIR if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    print(f"[figure] {out / (name + '.pdf')}")
    return paths


def trace_xy(history: Sequence[Dict[str, float]], xkey: str, ykey: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([h[xkey] for h in history], dtype=float)
    y = np.array([h.get(ykey, np.nan) for h in history], dtype=float)
    return x, y


def median_trace(
    histories: Iterable[Sequence[Dict[str, float]]],
    xkey: str,
    ykey: str,
    n_grid: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median trace of several runs, evaluated on a shared grid.

    Runs have different lengths, so every trace is interpolated (in log-y where
    possible) onto a common grid before the median is taken.
    """
    traces = [trace_xy(h, xkey, ykey) for h in histories]
    xmax = max(t[0][-1] for t in traces if len(t[0]))
    xmin = max(min(t[0][0] for t in traces if len(t[0])), 1e-12)
    grid = np.logspace(np.log10(max(xmin, 1e-9)), np.log10(max(xmax, 1e-9)), n_grid)
    stack = []
    for x, y in traces:
        if len(x) < 2:
            continue
        good = np.isfinite(y) & (y > 0)
        if good.sum() < 2:
            stack.append(np.full_like(grid, np.nan))
            continue
        logy = np.interp(grid, x[good], np.log10(y[good]), left=np.log10(y[good][0]),
                         right=np.log10(y[good][-1]))
        stack.append(10.0 ** logy)
    if not stack:
        return grid, np.full_like(grid, np.nan)
    return grid, np.nanmedian(np.vstack(stack), axis=0)


def grouped_bars(
    ax: plt.Axes,
    group_labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    colors: Dict[str, str],
    ylabel: str,
    log: bool = False,
    horizontal: bool = False,
) -> None:
    """Grouped bars, either vertical (default) or horizontal.

    The horizontal orientation puts the group labels on the y axis, where long
    instance names stay readable instead of being squeezed under the axis.
    """
    n_series = len(series)
    pos = np.arange(len(group_labels))
    width = 0.8 / max(n_series, 1)
    for k, (name, values) in enumerate(series.items()):
        offset = (k - (n_series - 1) / 2) * width
        vals = np.array(values, dtype=float)
        common = dict(label=name, color=colors.get(name, None), edgecolor="white",
                      linewidth=0.3)
        if horizontal:
            ax.barh(pos + offset, vals, width * 0.9, **common)
        else:
            ax.bar(pos + offset, vals, width * 0.9, **common)
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(group_labels)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()          # first group at the top, as in a reading order
        if log:
            ax.set_xscale("log")
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")


def annotate_legend(ax: plt.Axes, **kwargs) -> None:
    ax.legend(**kwargs)
