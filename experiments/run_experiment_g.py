"""Experiment G -- the entropically regularised (Sinkhorn) baseline.

The report solves the unregularised transport LP. The standard alternative in the
optimal-transport literature is to add an entropic penalty and run Sinkhorn
iterations, which is far cheaper and produces an exactly feasible plan, at the cost
of solving a *different* problem. This experiment quantifies that trade-off on the
same instances and with the same per-iteration budget as the main study:

* feasibility error -- how close the plan is to a transport plan;
* objective error against the *unregularised* reference -- the entropic bias;
* the certified optimality bound, which needs no reference value.

The regularisation strength is swept over ``eps``; small ``eps`` reduces the bias
but slows the iterations down. The bias sweep uses the seed-0 members of the main
grid, so the numbers can be read directly against experiment B; the decreasing-eps
continuation study that follows it runs over the whole grid instead (all five
seeds), so its conclusion does not rest on a single seed.

Run with::

    python experiments/run_experiment_g.py
"""

from __future__ import annotations

from typing import Dict, List

from common import BUDGET, RESULTS_DIR, certified_bracket, certify, save_json
from otopt import config, data, reference, sinkhorn

TOLS = config.TOLERANCES
#: regularisation strengths; the cost matrix is rescaled to [0, 1] first
EPS_GRID = (0.2, 0.1, 0.05, 0.02, 0.01)
#: the seed-0 members of the main grid (the eps-grid bias table)
INSTANCES = tuple(
    (size, kind, 0) for size in config.SIZES for kind in config.MASS_KINDS
)

# ----------------------------------------------------------------- continuation
#: decreasing-eps schedules with warm-started scaling vectors. The stages *share*
#: the standard budget, so each of these is compared against a fixed-eps run that
#: was given the same total number of iterations.
CONTINUATION_SCHEDULES = {
    "continuation to 1e-3": (0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001),
    "continuation to 1e-4": (0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001,
                             5e-4, 2e-4, 1e-4),
}
#: matched controls: a single eps with the whole budget and no warm start
CONTINUATION_CONTROLS = (1e-3, 1e-4)
#: the continuation study runs on the *whole* grid, not only the seed-0 slice, so
#: its conclusion does not rest on one seed per (size, regime) pair
CONTINUATION_INSTANCES = tuple(
    (size, kind, seed)
    for size in config.SIZES
    for kind in config.MASS_KINDS
    for seed in config.SEEDS
)

KEYS = (
    "iterations",
    "time",
    "final_feasibility_error",
    "final_objective_error",
    "iter_to_1e-03",
    "time_to_1e-03",
    "iter_to_1e-04",
    "time_to_1e-04",
    "certified_error_bound",
    "certified_error_bound_polished",
    "certified_at_1e-03",
    "certified_at_1e-04",
    "dual_violation_before_repair",
)


def main():
    meta, runs = [], []
    for size, kind, seed in INSTANCES:
        inst = data.make_synthetic_instance(size, kind, seed)
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        bracket = certified_bracket(inst, ref["plan"], ref["objective"])
        f_lower = bracket["lower"]
        meta.append(
            {
                "name": inst.name,
                "size": size,
                "mass_kind": kind,
                "seed": seed,
                "reference_objective": ref["objective"],
                "reference_time": ref["time"],
                "reference_lower_bound": f_lower,
                "reference_upper_bound": bracket["upper"],
                "reference_bracket_width": bracket["width"],
                "reference_polish_feasibility_error": bracket["polish_feasibility_error"],
                "reference_polish_reached_precision": bracket["polish_reached_precision"],
                "reference_bracket_is_valid": bracket["bracket_is_valid"],
                "reference_inside_bracket": bracket["reference_inside"],
                "reference_error_lower_bound": bracket["reference_error_lower_bound"],
                "reference_below_lower": bracket["reference_below_lower"],
                "reference_error_upper_bound": bracket["reference_error_upper_bound"],
            }
        )
        line = []
        for eps in EPS_GRID:
            out = sinkhorn.solve_sinkhorn(
                inst.a, inst.b, inst.C, eps=eps, f_ref=ref["objective"],
                tolerances=TOLS, **BUDGET,
            )
            certify(out, inst, f_lower=f_lower)
            runs.append(
                {
                    "instance": inst.name,
                    "size": size,
                    "mass_kind": kind,
                    "seed": seed,
                    "eps": eps,
                    "method": f"Sinkhorn (eps={eps:g})",
                    **{key: out[key] for key in KEYS},
                }
            )
            line.append(f"eps={eps:g}: obj={out['final_objective_error']:.1e}")
        print(f"[G] {inst.name:30s} " + "  ".join(line))

    cont_runs = run_continuation()

    save_json(
        {
            "budget": BUDGET,
            "tolerances": list(TOLS),
            "eps_grid": list(EPS_GRID),
            "continuation_schedules": {k: list(v) for k, v in CONTINUATION_SCHEDULES.items()},
            "continuation_controls": list(CONTINUATION_CONTROLS),
            "instances": meta,
            "runs": runs,
            "continuation_runs": cont_runs,
        },
        RESULTS_DIR / "experiment_g.json",
    )


def run_continuation() -> list:
    """Decreasing-eps Sinkhorn against matched fixed-eps controls.

    Two questions, one experiment. ``continuation to 1e-3``/``to 1e-4`` spread the
    standard budget over a schedule and warm-start ``(f, g)``; the paired controls
    ``fixed 1e-3``/``fixed 1e-4`` give the *same* total budget to a single ``eps``
    with no warm start. The comparison therefore isolates the scheduling, rather
    than confounding it with a larger budget or a smaller ``eps``.

    Unlike the eps-grid sweep, this runs over the *whole* evaluation grid (every
    size, both marginal regimes and all five seeds), so the reported medians and
    solve counts are not an artefact of a single seed per (size, regime) pair. Each
    instance is given its own reference solve and bracket, since it is outside the
    seed-0 slice that ``main`` stores.
    """
    rows = []
    for size, kind, seed in CONTINUATION_INSTANCES:
        inst = data.make_synthetic_instance(size, kind, seed)
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        bracket = certified_bracket(inst, ref["plan"], ref["objective"])
        f_ref = ref["objective"]
        f_lower = bracket["lower"]
        variants = {
            name: sinkhorn.solve_sinkhorn_continuation(
                inst.a, inst.b, inst.C, schedule, f_ref=f_ref, tolerances=TOLS, **BUDGET
            )
            for name, schedule in CONTINUATION_SCHEDULES.items()
        }
        for eps in CONTINUATION_CONTROLS:
            variants[f"fixed {eps:g}"] = sinkhorn.solve_sinkhorn(
                inst.a, inst.b, inst.C, eps=eps, f_ref=f_ref, tolerances=TOLS, **BUDGET
            )

        line = []
        for name, out in variants.items():
            certify(out, inst, f_lower=f_lower)
            rows.append(
                {
                    "instance": inst.name,
                    "size": size,
                    "mass_kind": kind,
                    "seed": seed,
                    "variant": name,
                    "eps_schedule": out.get("eps_schedule"),
                    "stages": out.get("stages"),
                    **{key: out[key] for key in KEYS},
                }
            )
            line.append(f"{name}: obj={out['final_objective_error']:.1e}"
                        f"/feas={out['final_feasibility_error']:.1e}")
        print(f"[G-cont] {inst.name:30s} " + "  ".join(line))
    return rows


if __name__ == "__main__":
    main()
