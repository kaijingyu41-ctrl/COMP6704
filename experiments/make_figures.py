"""Build every figure used in the report from the stored raw results.

The script only *reads* ``results/*.json`` (and the small ``.npz`` with the
handwritten-digit plans), so figures can be regenerated without re-running the
solvers.

Run with::

    python experiments/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

from common import COLORS, FIGURES_DIR, RESULTS_DIR, load_json
from otopt import config, data, plotting, reference

SHORT = {
    "ADMM (fixed rho)": "ADMM",
    "ADMM (residual balancing)": "ADMM-RB",
    "PDHG": "PDHG",
    "PDHG (restart)": "PDHG-R",
    "HiGHS (dual simplex)": "HiGHS",
}
METHODS = list(SHORT)


def short(name: str) -> str:
    return SHORT.get(name, name)


# ============================================================ F1: the problem
def fig_problem():
    inst = data.make_synthetic_instance(20, "uniform", 3)
    ref = reference.solve_reference(inst.a, inst.b, inst.C)
    rng = np.random.default_rng(3)
    pa = rng.random((20, 2))
    pb = rng.random((20, 2))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    thresh = inst.a.max() * 0.25
    for i in range(20):
        for j in range(20):
            w = ref["plan"][i, j]
            if w > thresh:
                ax.plot([pa[i, 0], pb[j, 0]], [pa[i, 1], pb[j, 1]], "-",
                        color="#1f77b4", alpha=min(1.0, 0.15 + 8 * w), lw=0.4 + 12 * w, zorder=1)
    ax.scatter(pa[:, 0], pa[:, 1], s=12, color="#d62728", zorder=3, label="supply $x_i$")
    ax.scatter(pb[:, 0], pb[:, 1], s=12, color="#2ca02c", zorder=3, label="demand $y_j$")
    ax.set_title("optimal transport plan")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.legend(loc="lower left")

    ax = axes[1]
    im = ax.imshow(ref["plan"], cmap="viridis", aspect="equal")
    ax.set_title(r"plan matrix $X^\star$ (20$\times$20)")
    ax.set_xlabel("demand index $j$"); ax.set_ylabel("supply index $i$")
    fig.colorbar(im, ax=ax, fraction=0.046, label="mass")
    return plotting.save(fig, "fig1_problem")


# ==================================================== F2/F3: convergence traces
def _trace_panel(ax, trace, xkey, ykey, title, logx=True):
    for method in METHODS:
        if method not in trace["methods"]:
            continue
        hist = trace["methods"][method]["history"]
        x, y = plotting.trace_xy(hist, xkey, ykey)
        good = np.isfinite(y) & (y > 0)
        ax.plot(x[good], y[good], color=COLORS[method], lw=1.0, label=short(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    return ax


def fig_convergence(a):
    """Objective and feasibility error against the iteration count.

    The three panels must match the caption of Figure 1 in the report, so the
    instance keys are listed explicitly rather than inferred.
    """
    traces = a["traces"]
    keys = [
        k
        for k in ("10-uniform-seed0", "200-uniform-seed0", "200-dirichlet0.2-seed0")
        if k in traces
    ]
    fig, axes = plt.subplots(2, len(keys), figsize=(6.6, 3.7), squeeze=False,
                             sharex="col")
    for c, key in enumerate(keys):
        size = key.split("-")[0]
        kind = "uniform" if "uniform" in key else "nonuniform"
        # the title must be passed to _trace_panel, which calls set_title itself
        _trace_panel(axes[0][c], traces[key], "iteration", "objective_error",
                     f"{kind} $m=n={size}$")
        _trace_panel(axes[1][c], traces[key], "iteration", "feasibility_error", "")
        # only the bottom row carries x tick labels, so nothing overlaps
        axes[0][c].tick_params(labelbottom=False)
        axes[1][c].set_xlabel("iteration")
    for r, lab in ((0, r"$|f-f_{\mathrm{ref}}|$"), (1, r"$e_{\mathrm{feas}}$")):
        axes[r][0].set_ylabel(lab)
    axes[0][-1].legend(loc="lower left")
    return plotting.save(fig, "fig2_convergence_iterations")


def fig_convergence_time(a):
    traces = a["traces"]
    keys = [k for k in ("20-uniform-seed1", "200-uniform-seed0", "200-dirichlet0.2-seed0") if k in traces]
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 2.1), squeeze=False)
    for c, key in enumerate(keys):
        _trace_panel(axes[0][c], traces[key], "time", "feasibility_error",
                     key.replace("-", " "))
        axes[0][c].set_xlabel("wall-clock time [s]")
    axes[0][0].set_ylabel(r"$e_{\mathrm{feas}}$")
    axes[0][-1].legend(loc="lower left")
    return plotting.save(fig, "fig3_convergence_time")


# ================================================= F4/F5/F6: scaling from exp B
def _runs_frame(b):
    df = pd.DataFrame(b["runs"])
    meta = pd.DataFrame(b["instances"])[["name", "size", "mass_kind", "seed"]]
    df = df.merge(meta, left_on="instance", right_on="name")
    return df


def _size_ticks(ax, sizes):
    """Log axis over problem sizes, but with plain readable tick labels."""
    ax.set_xscale("log")
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("problem size $m=n$")
    ax.minorticks_off()


def fig_scaling(b):
    df = _runs_frame(b)
    sizes = sorted(df["size"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1))

    handles = []
    for method in METHODS:
        sub = df[df.method == method]
        g = sub.groupby("size")["time"].median()
        axes[0].plot(g.index, g.values, marker="o", ms=3, color=COLORS[method],
                     label=short(method))
        hit = sub.dropna(subset=["time_to_1e-03"]).groupby("size")["time_to_1e-03"].median()
        line, = axes[1].plot(hit.index, hit.values, marker="s", ms=3,
                             color=COLORS[method], label=short(method))
        handles.append(line)          # only lines drawn in this panel are reported
    for ax, lab in zip(axes, ["median wall-clock time [s]", "median time to $10^{-3}$ [s]"]):
        _size_ticks(ax, sizes)
        ax.set_yscale("log")
        ax.set_ylabel(lab)
    axes[0].set_title("cost of the full budget")

    ref = pd.DataFrame(b["instances"])
    rtime = ref.groupby("size")[["reference_time"]].median()
    axes[2].plot(rtime.index, rtime["reference_time"], marker="D", ms=3,
                 color=COLORS["HiGHS (dual simplex)"], label="HiGHS")
    _size_ticks(axes[2], sizes)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("HiGHS solve time [s]")
    axes[2].set_title("reference solver")

    # separate legends, so no panel shows a series it does not plot
    axes[1].legend(handles=handles, loc="upper left")
    axes[2].legend(loc="upper left")
    return plotting.save(fig, "fig4_scaling")


def fig_solve_rate(b):
    df = _runs_frame(b)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, tol in zip(axes, ("1e-03", "1e-04")):
        sizes = sorted(df["size"].unique())
        series = {}
        for method in METHODS:
            sub = df[df.method == method]
            series[short(method)] = [
                100.0 * sub[sub["size"] == s][f"time_to_{tol}"].notna().mean() for s in sizes
            ]
        colors = {short(m): COLORS[m] for m in METHODS}
        plotting.grouped_bars(ax, [str(s) for s in sizes], series, colors,
                              f"solved within budget at {tol} [%]")
        ax.set_xlabel("problem size $m=n$")
        ax.set_ylim(0, 105)
        ax.set_title(f"tolerance {tol}")
    axes[1].legend(ncol=2, loc="upper right", fontsize=6)
    return plotting.save(fig, "fig5_solve_rate")


def fig_mass_effect(b):
    df = _runs_frame(b)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for ax, tol in zip(axes, ("1e-03", "1e-04")):
        data_by_method, positions, labels = [], [], []
        pos = 0
        for method in METHODS:
            for kind, off in (("uniform", -0.18), ("dirichlet0.2", 0.18)):
                vals = df[(df.method == method) & (df.mass_kind == kind)][f"iter_to_{tol}"].dropna()
                data_by_method.append(vals.values if len(vals) else np.array([np.nan]))
                positions.append(pos + off)
                labels.append("")
            pos += 1
        bp = ax.boxplot(data_by_method, positions=positions, widths=0.28, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=0.6))
        for patch, method in zip(bp["boxes"], [m for m in METHODS]):
            patch.set_facecolor(COLORS[method]); patch.set_alpha(0.55)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([short(m) for m in METHODS], rotation=15)
        ax.set_yscale("log")
        ax.set_title(f"iterations to {tol}")
    axes[0].set_ylabel("iterations")
    handles = [plt.Line2D([], [], color="0.4", marker="s", ls="", label="uniform"),
               plt.Line2D([], [], color="0.4", marker="s", ls="", mfc="none", label="Dirichlet 0.2")]
    axes[0].legend(handles=handles, loc="upper left")
    return plotting.save(fig, "fig6_mass_effect")


# ============================================ F7/F8/F9: sensitivity from exp C
def _pilot_label(name: str) -> str:
    """``syn-uniform-50-seed10`` -> ``uniform 50`` (the seeds are in the caption)."""
    parts = name.split("-")
    kind = "uniform" if "uniform" in parts else "nonunif."
    size = next((p for p in parts if p.isdigit()), "?")
    return f"{kind} {size}"


def _pilot_order(names) -> list:
    """Uniform instances first, then nonuniform, each ordered by problem size."""
    def size_of(n: str) -> int:
        return int(next(p for p in n.split("-") if p.isdigit()))

    return sorted(set(names), key=lambda n: ("uniform" not in n, size_of(n)))


def fig_parameter_sensitivity(c):
    """Three stacked panels, one per question of experiment C.

    Stacking the panels (rather than putting them side by side) gives each one
    the full figure width, so the pilot-instance names fit next to the data
    instead of being squeezed into a third of the width. The names of the two
    sweeps sit in an outside legend on the right, and the ablation panel is drawn
    as horizontal bars so its names also run horizontally. The ablation panel
    shows each enhanced variant next to the plain method it is meant to improve,
    which is the comparison the panel is there to support.
    """
    rho_df = pd.DataFrame(c["rho_sweep"])
    eta_df = pd.DataFrame(c["eta_sweep"])
    order = _pilot_order(rho_df["instance"])
    cmap = colormaps["tab10"]
    color = {name: cmap(i % 10) for i, name in enumerate(order)}

    fig, axes = plt.subplots(
        3, 1, figsize=(4.9, 5.8),
        gridspec_kw=dict(hspace=0.9, height_ratios=[1.0, 1.0, 1.35]),
    )

    rho_grid = sorted(rho_df["rho"].unique())
    for name in order:
        sub = rho_df[rho_df.instance == name].sort_values("rho")
        axes[0].plot(sub["rho"], sub["final_feasibility_error"], marker="o", ms=2.5,
                     lw=0.9, color=color[name], label=_pilot_label(name))
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xticks(rho_grid)
    axes[0].set_xticklabels([f"{g:g}" for g in rho_grid])
    axes[0].minorticks_off()
    axes[0].set_xlabel(r"ADMM penalty $\rho$")
    axes[0].set_ylabel(r"$e_{\mathrm{feas}}$ after budget")
    axes[0].set_title("(a) ADMM: penalty sensitivity")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)

    eta_grid = sorted(eta_df["eta"].unique())
    for name in order:
        sub = eta_df[eta_df.instance == name].sort_values("eta")
        axes[1].plot(sub["eta"], sub["final_feasibility_error"], marker="^", ms=2.5,
                     lw=0.9, color=color[name], label=_pilot_label(name))
        axes[1].plot(sub["eta"], sub["final_objective_error"], marker="v", ms=2.5,
                     lw=0.9, ls="--", color=color[name])
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xticks(eta_grid)
    axes[1].set_xticklabels([f"{g:g}" for g in eta_grid])
    axes[1].minorticks_off()
    axes[1].set_xlabel(r"PDHG step factor $\eta$")
    # the solid/dashed reading is given in the caption, so the axis label stays short
    axes[1].set_ylabel("error after budget")
    axes[1].set_title("(b) PDHG: step-size sensitivity")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)

    def picked(frame, **match):
        sub = frame
        for key, value in match.items():
            sub = sub[sub[key] == value]
        return sub.set_index("instance")["final_feasibility_error"]

    plain_admm = picked(rho_df, rho=config.ADMM_RHO)
    plain_pdhg = picked(eta_df, eta=config.PDHG_ETA)
    resbal = picked(pd.DataFrame(c["resbal_ablation"]), variant="residual balancing")
    restart = picked(pd.DataFrame(c["restart_ablation"]), variant="restart=True")
    precond = picked(pd.DataFrame(c["precondition_ablation"]), variant="preconditioned")

    series = {
        "ADMM": [plain_admm[i] for i in order],
        "ADMM-RB": [resbal[i] for i in order],
        "PDHG": [plain_pdhg[i] for i in order],
        "PDHG-R": [restart[i] for i in order],
        "PDHG-PC": [precond[i] for i in order],
    }
    colors = {
        "ADMM": COLORS["ADMM (fixed rho)"],
        "ADMM-RB": COLORS["ADMM (residual balancing)"],
        "PDHG": COLORS["PDHG"],
        "PDHG-R": COLORS["PDHG (restart)"],
        "PDHG-PC": "#7f7f7f",          # the preconditioned variant has no colour elsewhere
    }
    plotting.grouped_bars(axes[2], [_pilot_label(n) for n in order], series, colors,
                          r"$e_{\mathrm{feas}}$ after budget", log=True, horizontal=True)
    axes[2].set_title("(c) enhanced variants next to the plain method")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    return plotting.save(fig, "fig7_parameter_sensitivity")


def fig_ablation(c):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, frame, variants, title in (
        (axes[0], pd.DataFrame(c["resbal_ablation"]), ["fixed rho", "residual balancing"], "ADMM"),
        (axes[1], pd.DataFrame(c["restart_ablation"]), ["restart=False", "restart=True"], "PDHG"),
    ):
        insts = sorted(frame["instance"].unique())
        solved = [
            100.0 * np.mean([
                frame[(frame.variant == v) & (frame.instance == i)]["time_to_1e-03"].notna().any()
                for i in insts
            ])
            for v in variants
        ]
        x = np.arange(len(variants))
        ax.bar(x, solved, color=["#1f77b4", "#d62728"], width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(variants, rotation=10)
        ax.set_ylabel("instances solved to $10^{-3}$ [%]")
        ax.set_ylim(0, 105)
        ax.set_title(f"{title}: ablation")
    return plotting.save(fig, "fig8_ablation")


# ===================================================== F10: real digit images
def fig_digits(d, plans_path):
    z = np.load(plans_path)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4))

    for row in (1, 2):
        img_a, img_b = z[f"image_a_{row}"], z[f"image_b_{row}"]
        pa, pb = z[f"pos_a_{row}"], z[f"pos_b_{row}"]
        a, b, C, T = z[f"a_{row}"], z[f"b_{row}"], z[f"C_{row}"], z[f"ref_{row}"]

        ax = axes[row - 1][0]
        ax.imshow(img_a, cmap="gray_r")
        ax.set_title(f"source image {row}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row - 1][1]
        ax.imshow(img_b, cmap="gray_r")
        ax.set_title(f"target image {row}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row - 1][2]
        thresh = max(a.max(), b.max()) * 0.15
        for i in range(len(a)):
            for j in range(len(b)):
                if T[i, j] > thresh:
                    ax.plot([pa[i, 1], pb[j, 1]], [pa[i, 0], pb[j, 0]], "-",
                            color="#1f77b4", alpha=0.7, lw=0.3 + 10 * T[i, j])
        ax.scatter(pa[:, 1], pa[:, 0], s=3, color="#d62728")
        ax.scatter(pb[:, 1], pb[:, 0], s=3, color="#2ca02c")
        ax.invert_yaxis()
        ax.set_title("optimal coupling", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    meta = pd.DataFrame(d["instances"])
    fig.suptitle(
        f"{len(meta)} pairs; support sizes {meta.support_a.min()}--{meta.support_a.max()}",
        fontsize=7, y=1.02)
    fig.tight_layout()
    return plotting.save(fig, "fig9_digits_plans")


def fig_digits_convergence(d):
    df = pd.DataFrame(d["runs"]).merge(
        pd.DataFrame(d["instances"])[["name", "pair_kind"]], left_on="instance", right_on="name")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1))
    for ax, key, label in (
        (axes[0], "final_feasibility_error", r"$e_{\mathrm{feas}}$"),
        (axes[1], "final_objective_error", r"$|f-f_{\mathrm{ref}}|$"),
        (axes[2], "time", "wall-clock time [s]"),
    ):
        for method in METHODS:
            sub = df[df.method == method].groupby("pair_kind")[key]
            vals = [sub.median().get(k, np.nan) for k in ("cross", "same")]
            ax.plot([0, 1], vals, marker="o", ms=3, color=COLORS[method], label=short(method))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["cross-class", "same-class"])
        ax.set_ylabel(label)
        if key != "time":
            ax.set_yscale("log")
    axes[0].set_title("final accuracy on digit pairs")
    axes[0].legend(fontsize=6)
    return plotting.save(fig, "fig10_digits_convergence")


# ======================================================================== main
def main():
    plotting.setup_style()
    a = load_json(RESULTS_DIR / "experiment_a.json")
    b = load_json(RESULTS_DIR / "experiment_b.json")
    c = load_json(RESULTS_DIR / "experiment_c.json")
    d = load_json(RESULTS_DIR / "experiment_d.json")

    fig_problem()
    fig_convergence(a)
    fig_convergence_time(a)
    fig_scaling(b)
    fig_solve_rate(b)
    fig_mass_effect(b)
    fig_parameter_sensitivity(c)
    fig_ablation(c)
    fig_digits(d, RESULTS_DIR / "experiment_d_plans.npz")
    fig_digits_convergence(d)
    print(f"[done] figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
