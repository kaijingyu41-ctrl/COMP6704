"""Experiment E -- budget study on the instances that do not reach 1e-4.

Experiments B and D use a fixed budget of 50 000 iterations. That leaves one
question open: is the residual left on the nonuniform instances an intrinsic
accuracy limit of the methods, or is it simply budget-limited? Experiment E
answers it directly by running the two methods with a budget eight times larger
(400 000 iterations) and reading the errors off the same trace at 50k, 100k, 200k
and 400k iterations.

A shrinking residual means the limit seen at 50k was a budget artefact; a flat
residual would mean the methods genuinely stall. The unbalanced sizes are the
interesting ones; the uniform instance is included as a control, because its
residual already sits at machine precision after 50k iterations.

Run with::

    python experiments/run_experiment_e.py
"""

from __future__ import annotations

from typing import Dict, List

from common import RESULTS_DIR, save_json
from otopt import admm, config, data, metrics, pdhg, reference

TOLS = config.TOLERANCES
CHECKPOINTS = (50_000, 100_000, 200_000, 400_000)
#: an eight-fold budget relative to experiments B and D
LONG_BUDGET = dict(max_iter=max(CHECKPOINTS), time_limit=300.0, log_every=200)

#: unbalanced instances first (the question under study), then a uniform control
INSTANCES = (
    (200, "dirichlet0.2", 0),
    (200, "dirichlet0.2", 1),
    (100, "dirichlet0.2", 0),
    (200, "uniform", 0),
)


def _at(history: List[Dict[str, float]], iteration: int) -> Dict[str, float] | None:
    """Trace point at ``iteration``, or ``None`` when the run never reached it.

    The guard matters here: the long budget allows 300 s, so on a slow machine a
    run can stop before 400k iterations. Relabelling the last point it did reach
    as the 400k checkpoint would report a measurement that was never made, so
    unreached checkpoints are dropped (and printed as ``--``) instead.
    """
    return metrics.trace_point_at(history, iteration)


def main():
    rows = []
    for size, kind, seed in INSTANCES:
        inst = data.make_synthetic_instance(size, kind, seed)
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        common = dict(**LONG_BUDGET, tolerances=TOLS, f_ref=ref["objective"])

        runs = {
            "ADMM (fixed rho)": admm.solve_fixed_admm(
                inst.a, inst.b, inst.C, rho=config.ADMM_RHO, **common
            ),
            "PDHG": pdhg.solve_pdhg(inst.a, inst.b, inst.C, eta=config.PDHG_ETA, **common),
        }
        for method, out in runs.items():
            for it in CHECKPOINTS:
                row = _at(out["history"], it)
                if row is None:
                    continue
                rows.append(
                    {
                        "instance": inst.name,
                        "size": size,
                        "mass_kind": kind,
                        "seed": seed,
                        "method": method,
                        "checkpoint": it,
                        "feasibility_error": row["feasibility_error"],
                        "objective_error": row.get("objective_error"),
                    }
                )
            print(
                f"[E] {inst.name:32s} {method:18s} "
                + " ".join(
                    "--" if _at(out["history"], it) is None
                    else f"{it // 1000}k={_at(out['history'], it)['feasibility_error']:.1e}"
                    for it in CHECKPOINTS
                )
            )

    save_json(
        {
            "budget": LONG_BUDGET,
            "checkpoints": list(CHECKPOINTS),
            "baseline_budget": dict(max_iter=config.MAX_ITER, time_limit=config.TIME_LIMIT),
            "tolerances": list(TOLS),
            # recorded so that this study can never silently drift from the
            # settings used by experiments B and D
            "admm_rho": config.ADMM_RHO,
            "pdhg_eta": config.PDHG_ETA,
            "runs": rows,
        },
        RESULTS_DIR / "experiment_e.json",
    )


if __name__ == "__main__":
    main()
