"""Experiment B -- scaling and marginal-imbalance study.

A full factorial grid of synthetic instances is solved by every method with an
identical budget:

* size ``m = n in {20, 50, 100, 200}``,
* balanced (uniform) or highly unbalanced (Dirichlet 0.2) marginals,
* five random seeds.

For every run we record the final accuracy, the iteration count, the wall-clock
time, and the first iteration/time at which **both** the feasibility error and
the objective error fall below 1e-3 and 1e-4.

Run with::

    python experiments/run_experiment_b.py
"""

from __future__ import annotations

import time

from common import BUDGET, RESULTS_DIR, certified_bracket, run_methods, save_json
from otopt import config, data, reference

TOLS = config.TOLERANCES
SUMMARY_KEYS = (
    "iterations",
    "time",
    "final_feasibility_error",
    "final_objective_error",
    "final_negative_mass",
    "iter_to_1e-03",
    "time_to_1e-03",
    "iter_to_1e-04",
    "time_to_1e-04",
    # dual-gap certificate: an upper bound on the suboptimality of the returned
    # plan, computed without the reference value, plus the joint feasibility flag
    "certified_error_bound",
    "certified_error_bound_polished",
    "certified_complementarity_term",
    "certified_row_term",
    "certified_col_term",
    "signed_objective_gap",
    "signed_objective_gap_vs_lower_bound",
    "certificate_is_valid",
    "certified_at_1e-03",
    "certified_at_1e-04",
    "polished_feasibility_error",
    "dual_violation_before_repair",
)


def main():
    instances_meta, runs = [], []
    grid = data.make_instance_grid(config.SIZES, config.MASS_KINDS, config.SEEDS)
    print(f"[B] {len(grid)} instances")

    for k, inst in enumerate(grid, 1):
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        bracket = certified_bracket(inst, ref["plan"], ref["objective"])
        f_lower = bracket["lower"]
        instances_meta.append(
            {
                "name": inst.name,
                **{key: inst.meta[key] for key in ("size", "mass_kind", "seed")},
                "reference_objective": ref["objective"],
                "reference_time": ref["time"],
                "reference_iterations": ref["iterations"],
                # bracket on f*: a dual feasible lower bound and the objective of
                # the polished reference plan as an upper bound, together with the
                # flags that say whether the interval actually closed and whether
                # the reference value lies inside it
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
        start = time.perf_counter()
        out = run_methods(inst, ref["objective"], TOLS, BUDGET, f_lower=f_lower)
        print(
            f"[B] {k}/{len(grid)} {inst.name} (wall {time.perf_counter() - start:.1f}s): "
            + ", ".join(
                f"{m.split(' ')[0]}{'/R' if 'residual' in m else ''}="
                f"{o['final_feasibility_error']:.0e}"
                for m, o in out.items()
            )
        )
        for method, o in out.items():
            runs.append(
                {
                    "instance": inst.name,
                    "method": method,
                    **{key: o[key] for key in SUMMARY_KEYS},
                    "rho_final": o.get("rho_final"),
                    "n_rho_changes": o.get("n_rho_changes"),
                    "n_restarts": o.get("n_restarts"),
                }
            )

    save_json(
        {
            "budget": BUDGET,
            "tolerances": list(TOLS),
            "rho_default": config.ADMM_RHO,
            "eta_default": config.PDHG_ETA,
            "instances": instances_meta,
            "runs": runs,
        },
        RESULTS_DIR / "experiment_b.json",
    )


if __name__ == "__main__":
    main()
