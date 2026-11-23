"""Experiment A -- correctness certificates and convergence traces.

Two things are checked:

1. *Correctness.* On problems whose optimum is known analytically (a diagonal
   cost matrix, a constant cost matrix, and a planted permutation) every solver
   must reproduce both the optimal value and a feasible plan.
2. *Convergence.* On small random instances the full objective-error and
   feasibility-error traces of all four solvers are recorded, so that the
   behaviour of the methods can be compared directly.

Run with::

    python experiments/run_experiment_a.py
"""

from __future__ import annotations

import numpy as np

from common import RESULTS_DIR, certified_bracket, result_record, run_methods, save_json
from otopt import admm, config, data, metrics, pdhg, reference

TOLS = config.TOLERANCES
# the trace budget equals the benchmark budget so that the curves can be read
# against the aggregated results of experiment B
BUDGET = dict(max_iter=config.MAX_ITER, time_limit=config.TIME_LIMIT,
              log_every=config.LOG_EVERY)

TRACE_INSTANCES = (
    (10, "uniform", 0),
    (20, "uniform", 1),
    (20, "dirichlet0.2", 2),
    (100, "uniform", 0),
    (200, "uniform", 0),
    (200, "dirichlet0.2", 0),
)


# ------------------------------------------------------------------ correctness
def check_known(name, a, b, C, expected_objective, expected_plan=None):
    ref = reference.solve_reference(a, b, C)
    inst = data.Instance(name=name, a=a, b=b, C=C)
    bracket = certified_bracket(inst, ref["plan"])
    f_lower = bracket["lower"]
    row = {
        "case": name,
        "reference_objective": ref["objective"],
        "expected_objective": float(expected_objective),
        "reference_matches_expected": abs(ref["objective"] - expected_objective) < 1e-8,
        "reference_lower_bound": f_lower,
        "reference_upper_bound": bracket["upper"],
        "reference_bracket_width": bracket["width"],
        # on these instances the analytic optimum is known, so the bracket can be
        # checked against the truth rather than against the reference solver
        "bracket_contains_expected": bool(
            f_lower - 1e-9 <= expected_objective <= bracket["upper"] + 1e-9
        ),
    }
    if expected_plan is not None:
        row["reference_plan_error"] = float(np.abs(ref["plan"] - expected_plan).max())
    results = run_methods(inst, ref["objective"], TOLS, BUDGET, f_lower=f_lower)
    for method, out in results.items():
        row[f"{method}::objective_error"] = out["final_objective_error"]
        row[f"{method}::feasibility_error"] = out["final_feasibility_error"]
        row[f"{method}::certified_error_bound"] = out["certified_error_bound"]
        row[f"{method}::certificate_is_valid"] = out["certificate_is_valid"]
        row[f"{method}::certified_at_1e-04"] = out["certified_at_1e-04"]
    return row


def run_correctness():
    rows = []

    # 1) diagonal cost matrix: the optimal plan is the diagonal itself
    a = b = np.array([0.5, 0.5])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    rows.append(check_known("2x2 diagonal cost", a, b, C, 0.0, np.diag([0.5, 0.5])))

    # 2) constant cost matrix: every feasible plan costs the same
    rng = np.random.default_rng(11)
    a = rng.dirichlet(np.full(9, 0.5))
    b = rng.dirichlet(np.full(7, 0.5))
    C = np.full((9, 7), 0.37)
    rows.append(check_known("constant cost (9x7)", a, b, C, 0.37))

    # 3) planted permutation: the optimum is zero and unique
    n = 30
    a = b = np.full(n, 1.0 / n)
    perm = np.random.default_rng(12).permutation(n)
    C = np.ones((n, n))
    C[np.arange(n), perm] = 0.0
    plan = np.zeros((n, n))
    plan[np.arange(n), perm] = 1.0 / n
    rows.append(check_known("planted permutation (30x30)", a, b, C, 0.0, plan))

    return rows


# ------------------------------------------------------------------ convergence
def run_traces():
    traces = {}
    for size, kind, seed in TRACE_INSTANCES:
        inst = data.make_synthetic_instance(size, kind, seed)
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        bracket = certified_bracket(inst, ref["plan"])
        f_lower = bracket["lower"]
        out = run_methods(inst, ref["objective"], TOLS, BUDGET, f_lower=f_lower)
        key = f"{size}-{kind}-seed{seed}"
        traces[key] = {
            "meta": {"size": size, "mass_kind": kind, "seed": seed,
                     "reference_objective": ref["objective"],
                     "reference_time": ref["time"], "digits_of_ref": ref["iterations"],
                     "reference_lower_bound": f_lower,
                     "reference_upper_bound": bracket["upper"],
                     "reference_bracket_width": bracket["width"]},
            "methods": {m: result_record(o) for m, o in out.items()},
        }
        print(f"[A] trace {key}: " + ", ".join(
            f"{m} feas={o['final_feasibility_error']:.1e} obj={o['final_objective_error']:.1e}"
            for m, o in out.items()))
    return traces


def main():
    payload = {
        "budget": BUDGET,
        "tolerances": list(TOLS),
        "correctness": run_correctness(),
        "traces": run_traces(),
    }
    save_json(payload, RESULTS_DIR / "experiment_a.json")


if __name__ == "__main__":
    main()
