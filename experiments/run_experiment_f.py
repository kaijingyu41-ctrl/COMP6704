"""Experiment F -- how repeatable are the wall-clock times?

Every timing quoted in the report comes from a single run of a given
(instance, method) pair, so a reader cannot tell how much of a difference between
two methods is machine noise. This experiment repeats the *same* configuration
several times -- after one discarded warm-up run -- for the configurations whose
timings the report leans on most: the smallest and the largest regime, with uniform
and with nonuniform marginals. It reports the spread of the total wall-clock time
and of the time to tolerance, which are the two quantities the report compares.

Only ADMM and PDHG are repeated: they are the two families whose speed is
compared. Residual balancing is more than six times slower at the largest size,
which is far outside any plausible noise.

Run with::

    python experiments/run_experiment_f.py
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from common import BUDGET, RESULTS_DIR, save_json
from otopt import admm, config, data, pdhg, reference

#: repeats of each configuration, after one discarded warm-up run
REPEATS = 5

#: the smallest and the largest regime, uniform and nonuniform marginals
INSTANCES = ((20, "uniform", 0), (200, "uniform", 0), (200, "dirichlet0.2", 0))


def _admm(inst, f_ref):
    return admm.solve_fixed_admm(
        inst.a, inst.b, inst.C, rho=config.ADMM_RHO,
        tolerances=config.TOLERANCES, f_ref=f_ref, **BUDGET,
    )


def _pdhg(inst, f_ref):
    return pdhg.solve_pdhg(
        inst.a, inst.b, inst.C, eta=config.PDHG_ETA,
        tolerances=config.TOLERANCES, f_ref=f_ref, **BUDGET,
    )


def _spread(values: List[float]) -> float:
    """Relative spread of a set of timings, in percent of the median."""
    med = float(np.median(values))
    return 100.0 * (max(values) - min(values)) / med


def main():
    rows: List[Dict[str, float]] = []
    summary = []

    for size, kind, seed in INSTANCES:
        inst = data.make_synthetic_instance(size, kind, seed)
        f_ref = reference.solve_reference(inst.a, inst.b, inst.C)["objective"]

        for method, runner in (("ADMM (fixed rho)", _admm), ("PDHG", _pdhg)):
            runner(inst, f_ref)                       # warm-up, discarded

            for k in range(1, REPEATS + 1):
                out = runner(inst, f_ref)
                rows.append(
                    {
                        "instance": inst.name,
                        "size": size,
                        "mass_kind": kind,
                        "seed": seed,
                        "method": method,
                        "repeat": k,
                        "time": out["time"],
                        "time_to_1e-03": out["time_to_1e-03"],
                        "time_to_1e-04": out["time_to_1e-04"],
                        "iterations": out["iterations"],
                    }
                )

            mine = [r for r in rows if r["instance"] == inst.name and r["method"] == method]
            times = [r["time"] for r in mine]
            to_1e3 = [r["time_to_1e-03"] for r in mine if r["time_to_1e-03"] is not None]
            summary.append((inst.name, method, float(np.median(times)), _spread(times)))
            print(
                f"[F] {inst.name:30s} {method:18s} "
                f"budget {np.median(times):6.2f}s (range {min(times):.2f}-{max(times):.2f}, "
                f"spread {_spread(times):4.1f}%)  "
                f"1e-3 in {np.median(to_1e3):5.3f}s (spread {_spread(to_1e3):4.1f}%)"
                if to_1e3
                else f"[F] {inst.name:30s} {method:18s} budget {np.median(times):6.2f}s"
            )

    save_json(
        {
            "repeats": REPEATS,
            "warmup_runs": 1,
            "budget": BUDGET,
            "admm_rho": config.ADMM_RHO,
            "pdhg_eta": config.PDHG_ETA,
            "instances": [
                {"size": s, "mass_kind": k, "seed": sd} for s, k, sd in INSTANCES
            ],
            "runs": rows,
        },
        RESULTS_DIR / "experiment_f.json",
    )

    worst = max(s[3] for s in summary)
    print(f"\n[F] largest relative spread over the {len(summary)} configurations: {worst:.1f}%")


if __name__ == "__main__":
    main()
