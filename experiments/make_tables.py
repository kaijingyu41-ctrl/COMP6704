"""Generate the LaTeX tables used in the report from the stored raw results.

Writing the tables from the result files (instead of copying numbers by hand)
keeps the report and the experiments consistent, and means the tables can be
regenerated whenever the experiments are re-run.

Run with::

    python experiments/make_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR, load_json
from otopt import config
from otopt.certification import ENDPOINT_TOL

TABLES_DIR = config.ROOT / "report" / "tables"
SHORT = {
    "ADMM (fixed rho)": "ADMM",
    "ADMM (residual balancing)": "ADMM-RB",
    "PDHG": "PDHG",
    "PDHG (restart)": "PDHG-R",
}
METHODS = list(SHORT)
DASH = "--"


def num(v, digits=2) -> str:
    return DASH if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def sci(v, digits=1) -> str:
    """Scientific notation, used for error values that span many orders."""
    if v is None:
        return DASH
    v = float(v)
    if not np.isfinite(v):
        return DASH
    if v == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10.0 ** exp
    return rf"${mant:.{digits}f}{{\times}}10^{{{exp}}}$"


def integer(v) -> str:
    return DASH if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{int(round(v)):,}"


def write(name: str, body: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / name
    path.write_text(body, encoding="utf-8")
    print(f"[table] {path}")
    return path


# --------------------------------------------------------------------- helpers
def runs_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["runs"])
    meta = pd.DataFrame(payload["instances"])[["name", "size", "mass_kind", "seed"]]
    return df.merge(meta, left_on="instance", right_on="name")


def cell_rate_time(frame: pd.DataFrame, tol: str) -> str:
    """``rate (median time)`` for one group of runs."""
    col = f"time_to_{tol}"
    solved = frame[col].notna()
    rate = 100.0 * solved.mean() if len(frame) else np.nan
    med = frame.loc[solved, col].median() if solved.any() else np.nan
    if not np.isfinite(rate):
        return DASH
    return f"{rate:.0f}\\% ({num(med)})"


# ------------------------------------------------------------------ correctness
def table_correctness(a: dict) -> Path:
    """One compact table, with both errors in a single cell.

    Every entry on these instances is of order ``1e-16`` or smaller, so the two
    error measures are reported as ``objective / feasibility`` in units of
    ``1e-16``.  Full scientific notation would not fit the text width.
    """
    def scaled(v) -> str:
        if v is None or not np.isfinite(float(v)):
            return DASH
        x = float(v) * 1e16
        if x == 0.0:
            return "0"
        if x >= 100:
            s = f"{x:.0f}"
        elif x >= 10:
            s = f"{x:.1f}"
        elif x >= 1:
            s = f"{x:.2f}"
        else:
            s = f"{x:.3f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    rows = []
    for case in a["correctness"]:
        cells = " & ".join(
            f"{scaled(case[f'{m}::objective_error'])}/"
            f"{scaled(case[f'{m}::feasibility_error'])}"
            for m in METHODS
        )
        rows.append(
            f"{case['case']} & {num(case['expected_objective'], 3)} & {cells}" + r" \\"
        )
    body = "\n".join(
        [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            "Case & $f^\\star$ & " + " & ".join(SHORT[m] for m in METHODS) + r" \\",
            r"\midrule",
            r"\multicolumn{6}{l}{\emph{objective error} / \emph{feasibility error}, "
            r"in units of $10^{-16}$} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_correctness.tex", body)


# ---------------------------------------------------------------------- exp. B
def table_main(b: dict) -> Path:
    """One table, two stacked panels: solve rate and median time to tolerance."""
    df = runs_frame(b)
    sizes = sorted(df["size"].unique())
    panels = []
    for tol, label in (("1e-03", r"Tolerance $10^{-3}$"), ("1e-04", r"Tolerance $10^{-4}$")):
        lines = []
        for method in METHODS:
            cells = [
                cell_rate_time(df[(df.method == method) & (df["size"] == s)], tol) for s in sizes
            ]
            lines.append(f"{SHORT[method]} & " + " & ".join(cells) + r" \\")
        panels.append(
            "\n".join(
                [
                    r"\begin{tabular}{l" + "c" * len(sizes) + "}",
                    r"\toprule",
                    label + " & " + " & ".join(f"$m=n={s}$" for s in sizes) + r" \\",
                    r"\midrule",
                    *lines,
                    r"\bottomrule",
                    r"\end{tabular}",
                ]
            )
        )
    body = "\n".join(
        [
            r"\begin{tabular}{c}",
            panels[0],
            r"\\[1.2ex]",
            panels[1],
            r"\end{tabular}",
        ]
    )
    return write("tab_main.tex", body)


def table_mass(b: dict) -> Path:
    df = runs_frame(b)
    kinds = list(config.MASS_KINDS)
    lines = []
    for method in METHODS:
        cells = []
        for kind in kinds:
            sub = df[(df.method == method) & (df.mass_kind == kind)]
            cells.append(
                f"{cell_rate_time(sub, '1e-03')} & "
                f"{integer(sub['iter_to_1e-03'].median())} & "
                f"{sci(sub['final_feasibility_error'].median())}"
            )
        lines.append(f"{SHORT[method]} & " + " & ".join(cells) + r" \\")
    header = " & ".join(
        r"\multicolumn{3}{c}{" + ("Uniform" if k == "uniform" else "Dirichlet $\\alpha=0.2$") + "}"
        for k in kinds
    )
    sub_header = " & ".join([r"rate (time) & iters & $e_{\mathrm{feas}}$"] * len(kinds))
    body = "\n".join(
        [
            r"\begin{tabular}{l" + "ccc" * len(kinds) + "}",
            r"\toprule",
            r"Method & " + header + r" \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r" & " + sub_header + r" \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_mass.tex", body)


# ---------------------------------------------------------------------- exp. C
def table_sensitivity(c: dict) -> Path:
    rho = pd.DataFrame(c["rho_sweep"])
    eta = pd.DataFrame(c["eta_sweep"])

    rho_lines = []
    for inst, sub in rho.groupby("instance"):
        best = sub.loc[sub["final_feasibility_error"].idxmin()]
        med = sub["final_feasibility_error"].median()
        rho_lines.append(
            f"{inst} & {num(sub['final_feasibility_error'].min(), 1)} "
            f"($\\rho={best['rho']:g}$) & {num(med, 1)} \\\\"
        )

    eta_lines = []
    for inst, sub in eta.groupby("instance"):
        best = sub.loc[sub["final_feasibility_error"].idxmin()]
        eta_lines.append(
            f"{inst} & {num(sub['final_feasibility_error'].min(), 1)} "
            f"($\\eta={best['eta']:g}$) & {num(sub['final_feasibility_error'].max(), 1)} \\\\"
        )

    body = "\n".join(
        [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Instance & best $e_{\mathrm{feas}}$ (arg) & median $e_{\mathrm{feas}}$ \\",
            r"\multicolumn{3}{l}{\emph{ADMM penalty sweep},\ $\rho\in\{0.1,1,10,100,1000\}$} \\",
            r"\midrule",
            *rho_lines,
            r"\midrule",
            r"\multicolumn{3}{l}{\emph{PDHG step sweep},\ $\eta\in\{0.3,0.6,0.99\}$} \\",
            r"\midrule",
            *eta_lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_sensitivity.tex", body)


def table_ablation(c: dict) -> Path:
    resbal = pd.DataFrame(c["resbal_ablation"])
    restart = pd.DataFrame(c["restart_ablation"])
    precondition = pd.DataFrame(c.get("precondition_ablation", []))
    families = [
        ("ADMM", resbal, ["fixed rho", "residual balancing"]),
        ("PDHG", restart, ["restart=False", "restart=True"]),
    ]
    if len(precondition):
        families.append(("PDHG", precondition, ["plain", "preconditioned"]))
    lines = []
    for label, frame, variants in families:
        for v in variants:
            sub = frame[frame.variant == v]
            lines.append(
                f"{label} & {v} & {cell_rate_time(sub, '1e-03')} & "
                f"{cell_rate_time(sub, '1e-04')} & "
                f"{sci(sub['final_feasibility_error'].median())} \\\\"
            )
    body = "\n".join(
        [
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Family & Variant & rate@$10^{-3}$ (s) & rate@$10^{-4}$ (s) & median $e_{\mathrm{feas}}$ \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_ablation.tex", body)


# ---------------------------------------------------------------------- exp. E
def table_budget(e: dict) -> Path:
    """Feasibility error at 50k / 100k / 200k / 400k iterations."""
    df = pd.DataFrame(e["runs"])
    points = list(e["checkpoints"])
    lines = []
    for name in df["instance"].unique():
        sub = df[df.instance == name]
        meta = sub.iloc[0]
        label = f"{meta['size']} {'nonuniform' if meta['mass_kind'] != 'uniform' else 'uniform'} (seed {meta['seed']})"
        for k, method in enumerate(sorted(sub["method"].unique())):
            rows = sub[sub.method == method].set_index("checkpoint")
            cells = [sci(rows.loc[p, "feasibility_error"]) if p in rows.index else DASH
                     for p in points]
            head = label if k == 0 else ""
            lines.append(f"{head} & {SHORT.get(method, method)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1pt]")
    body = "\n".join(
        [
            r"\begin{tabular}{ll" + "c" * len(points) + "}",
            r"\toprule",
            r"Instance & Method & \multicolumn{" + str(len(points))
            + r"}{c}{$e_{\mathrm{feas}}$ after $k$ iterations} \\",
            r"\cmidrule(lr){3-" + str(2 + len(points)) + "}",
            r" & & " + " & ".join(f"{p // 1000}k" for p in points) + r" \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_budget.tex", body)


# ---------------------------------------------------------------------- exp. D
def table_digits(d: dict) -> Path:
    df = pd.DataFrame(d["runs"]).merge(
        pd.DataFrame(d["instances"])[["name", "pair_kind"]], left_on="instance", right_on="name"
    )
    lines = []
    for method in METHODS:
        cells = []
        for kind in ("same", "cross"):
            sub = df[(df.method == method) & (df.pair_kind == kind)]
            cells.append(
                f"{cell_rate_time(sub, '1e-03')} & "
                f"{sci(sub['final_feasibility_error'].median())} & "
                f"{sci(sub['final_objective_error'].median())}"
            )
        lines.append(f"{SHORT[method]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(
        [
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r" & \multicolumn{3}{c}{same label} & \multicolumn{3}{c}{different label} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"Method & rate (s) & $e_{\mathrm{feas}}$ & $|f-f_{\mathrm{ref}}|$ & "
            r"rate (s) & $e_{\mathrm{feas}}$ & $|f-f_{\mathrm{ref}}|$ \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_digits.tex", body)


# ---------------------------------------------------------------------- exp. G
def table_sinkhorn(g: dict) -> Path:
    """The entropic bias of the regularised baseline.

    Only the objective panel is tabulated: the feasibility error over the same runs is
    at machine precision throughout (worst case ``1.7e-15``), so a second panel of
    identical-looking entries would carry nothing the caption does not, and the
    per-instance values remain in ``results/experiment_g.json``.
    """
    df = pd.DataFrame(g["runs"])
    sizes = sorted(df["size"].unique())
    lines = []
    for eps in g["eps_grid"]:
        subset = df[df["eps"] == eps]
        cells = [sci(subset[subset["size"] == s]["final_objective_error"].median()) for s in sizes]
        lines.append(rf"$\varepsilon={eps:g}$ & " + " & ".join(cells) + r" \\")
    body = "\n".join(
        [
            r"\begin{tabular}{l" + "c" * len(sizes) + r"}",
            r"\toprule",
            r"\emph{objective error} (entropic bias) & "
            + " & ".join(f"$m=n={s}$" for s in sizes)
            + r" \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_sinkhorn.tex", body)


# ---------------------------------------------------------------------- exp. B
def table_certificate(b: dict) -> Path:
    """Certified upper bound on the suboptimality against the reference error.

    The left panel is the bound the solvers certify for the plan they return (no
    reference value involved); the right panel is the objective error measured
    against HiGHS. Both are upper bounds on the *same* signed quantity in the first
    case and absolute errors in the second, and they are medians, so the panels are
    comparable but a bound below the reference error is expected wherever the
    reference solver itself is the less accurate of the two.
    """
    df = runs_frame(b)
    sizes = sorted(df["size"].unique())
    panels = []
    for key, label in (
        ("certified_error_bound", "certified bound"),
        ("final_objective_error", r"error vs. reference"),
    ):
        lines = []
        for method in METHODS:
            sub = df[df.method == method]
            cells = [sci(sub[sub["size"] == s][key].median()) for s in sizes]
            lines.append(f"{SHORT[method]} & " + " & ".join(cells) + r" \\")
        panels.append(
            "\n".join(
                [
                    r"\begin{tabular}{l" + "c" * len(sizes) + "}",
                    r"\toprule",
                    label + " & " + " & ".join(f"$m=n={s}$" for s in sizes) + r" \\",
                    r"\midrule",
                    *lines,
                    r"\bottomrule",
                    r"\end{tabular}",
                ]
            )
        )
    body = "\n".join(
        [r"\begin{tabular}{c}", panels[0], r"\\[1.2ex]", panels[1], r"\end{tabular}"]
    )
    return write("tab_certificate.tex", body)


def table_sinkhorn_continuation(g: dict) -> Path | None:
    """Decreasing-eps Sinkhorn against matched fixed-eps controls.

    Every row was given the *same* total iteration budget, so the comparison
    isolates the schedule (and the warm start) from the budget. Four rows, two
    endpoints: at each endpoint the scheduled run and its single-eps control.
    """
    rows_in = g.get("continuation_runs")
    if not rows_in:
        print("[table] no continuation runs stored; skipping tab_sinkhorn_cont.tex")
        return None
    df = pd.DataFrame(rows_in)
    order = ["fixed 0.001", "continuation to 1e-3", "fixed 0.0001", "continuation to 1e-4"]
    label = {
        "fixed 0.001": r"single $\varepsilon=10^{-3}$",
        "continuation to 1e-3": r"schedule $\to 10^{-3}$",
        "fixed 0.0001": r"single $\varepsilon=10^{-4}$",
        "continuation to 1e-4": r"schedule $\to 10^{-4}$",
    }
    lines = []
    for variant in order:
        sub = df[df.variant == variant]
        if not len(sub):
            continue
        n_cert = int(sub["certified_at_1e-04"].sum())
        lines.append(
            f"{label[variant]} & {sci(sub['final_objective_error'].median())} & "
            f"{sci(sub['certified_error_bound'].median())} & "
            f"{n_cert}/{len(sub)} & {num(sub['time'].median())} \\\\"
        )
    body = "\n".join(
        [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Variant & median $e_{\mathrm{obj}}$ & median bound & certified@$10^{-4}$ & time (s) \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return write("tab_sinkhorn_cont.tex", body)


# ------------------------------------------------- reference-solver audit (no table)
def bracket_audit(ref: pd.DataFrame, label: str) -> dict:
    """Summarise the reference-solver audit from the stored bracket primitives.

    The rule is the one implemented in ``otopt.certification``: an interval counts
    as *valid* when the polish reached precision and the two independently computed
    endpoints agree to ``ENDPOINT_TOL``. ``f_ref`` inside a valid interval is then
    pinned to the width; otherwise the surviving statement is the *one-sided*
    ``|f_ref - f*| >= max(0, L - f_ref)``, whose positive part is nonzero only where
    ``f_ref`` sits below ``L``. Everything is recomputed from the stored columns, so
    the audit can be regenerated from an existing results file without re-running the
    experiments.
    """
    tol = ENDPOINT_TOL
    valid = (ref["reference_polish_feasibility_error"] <= tol) & (
        ref["reference_upper_bound"] >= ref["reference_lower_bound"] - tol
    )
    inside = (
        valid
        & (ref["reference_objective"] >= ref["reference_lower_bound"] - tol)
        & (ref["reference_objective"] <= ref["reference_upper_bound"] + tol)
    )
    gap = ref["reference_lower_bound"] - ref["reference_objective"]
    # only the positive part is a statement about the reference error, and only a
    # violation larger than the endpoint rounding makes that statement meaningful
    below = gap > tol
    summary = {
        "n": int(len(ref)),
        "valid": int(valid.sum()),
        "inside": int(inside.sum()),
        "max_valid_width": float(ref.loc[valid, "reference_bracket_width"].max()),
        "median_valid_width": float(ref.loc[valid, "reference_bracket_width"].median()),
        "max_error_lower_bound": float(gap.clip(lower=0.0).max()) if below.any() else 0.0,
        "below_lower": int(below.sum()),
    }
    print(f"[audit] {label}: valid {summary['valid']}/{summary['n']}"
          f" ({summary['inside']} containing f_ref), width <= {summary['max_valid_width']:.2e}"
          f" (median {summary['median_valid_width']:.2e})"
          f" | reference error >= {summary['max_error_lower_bound']:.2e} on"
          f" {summary['below_lower']} instances below L")
    return summary


# ------------------------------------------------------------------------- main
def main():
    a = load_json(RESULTS_DIR / "experiment_a.json")
    b = load_json(RESULTS_DIR / "experiment_b.json")
    c = load_json(RESULTS_DIR / "experiment_c.json")
    d = load_json(RESULTS_DIR / "experiment_d.json")
    e = load_json(RESULTS_DIR / "experiment_e.json")
    g = load_json(RESULTS_DIR / "experiment_g.json")

    table_correctness(a)
    table_main(b)
    table_mass(b)
    table_sensitivity(c)
    table_ablation(c)
    table_budget(e)
    table_digits(d)
    table_certificate(b)
    table_sinkhorn(g)
    table_sinkhorn_continuation(g)

    # a few aggregate numbers quoted in the running text
    df = runs_frame(b)
    print("\n[stats] median time-to-1e-3 by method (all sizes):")
    for method in METHODS:
        sub = df[df.method == method]
        solved = sub["time_to_1e-03"].notna().mean() * 100
        print(f"  {SHORT[method]:8s} solve rate {solved:5.1f}%  "
              f"median time {num(sub['time_to_1e-03'].median())}s  "
              f"median iters {integer(sub['iterations'].median())}")
    ref = pd.DataFrame(b["instances"])
    print("[stats] HiGHS: median time by size")
    print(ref.groupby("size")["reference_time"].median().to_string())
    if "reference_bracket_width" in ref:
        bracket_audit(ref, "experiment B (synthetic)")
        bracket_audit(pd.DataFrame(d["instances"]), "experiment D (digits)")

    print("\n[stats] dual-gap certificate (median over all 40 instances of each method):")
    for method in METHODS:
        sub = df[df.method == method]
        print(f"  {SHORT[method]:8s} bound {sub['certified_error_bound'].median():.2e}"
              f"  worst {sub['certified_error_bound'].max():.2e}"
              f"  | actual obj. err {sub['final_objective_error'].median():.2e}"
              f"  | certified@1e-4 {int(sub['certified_at_1e-04'].sum())}/40")
    print(f"  bound valid in {int(df['certificate_is_valid'].sum())}/{len(df)} runs")
    print(f"  bound below 1e-4 in {int((df['certified_error_bound'] < 1e-4).sum())}/{len(df)} runs")
    print(f"  plan feasible AND bound below 1e-4 in "
          f"{int(df['certified_at_1e-04'].sum())}/{len(df)} runs")


if __name__ == "__main__":
    main()
