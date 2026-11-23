"""Experiment C -- parameter sensitivity and ablations.

Three questions are investigated on a small pilot set of instances:

* how the ADMM penalty parameter ``rho`` changes the convergence behaviour;
* how the PDHG step-size factor ``eta`` changes the convergence behaviour;
* whether the residual-balancing rule for ``rho`` and the restart heuristic for
  PDHG actually help, at the accuracy levels we care about.

The pilot instances use random seeds that are **disjoint** from the seeds of the
evaluation grid in experiment B (``config.SEEDS == (0, 1, 2, 3, 4)``). The pilot
seeds are 10, 11 and 12, so the parameters chosen here are frozen on data that is
never used again for evaluation.

Run with::

    python experiments/run_experiment_c.py
"""

from __future__ import annotations

from common import BUDGET, RESULTS_DIR, save_json
from otopt import admm, config, data, pdhg, reference

TOLS = config.TOLERANCES
#: disjoint from ``config.SEEDS`` -- these instances are for tuning only
PILOT = (
    (50, "uniform", 10),
    (50, "dirichlet0.2", 11),
    (100, "uniform", 12),
    (100, "dirichlet0.2", 13),
    (200, "uniform", 14),
    (200, "dirichlet0.2", 15),
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
    "n_rho_changes",
    "n_restarts",
)


def _row(inst, tag, **kw):
    """Collect the reported fields of one run; missing keys become ``None``."""
    return {"instance": inst.name, **tag, **{k: kw.get(k) for k in KEYS}}


def main():
    instances, refs = [], {}
    for size, kind, seed in PILOT:
        inst = data.make_synthetic_instance(size, kind, seed)
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        instances.append(inst)
        refs[inst.name] = ref["objective"]

    rho_sweep, eta_sweep, resbal_ablation = [], [], []
    restart_ablation, precondition_ablation = [], []

    for inst in instances:
        f_ref = refs[inst.name]
        common = dict(**BUDGET, tolerances=TOLS, f_ref=f_ref)

        for rho in config.ADMM_RHO_GRID:
            out = admm.solve_fixed_admm(inst.a, inst.b, inst.C, rho=rho, **common)
            rho_sweep.append(_row(inst, {"rho": rho}, **out))
            print(f"[C] rho sweep {inst.name} rho={rho} feas={out['final_feasibility_error']:.1e}")

        # ablation: fixed rho vs residual balancing, at the default rho
        fixed = admm.solve_fixed_admm(inst.a, inst.b, inst.C, rho=config.ADMM_RHO, **common)
        resbal = admm.solve_residual_balancing_admm(
            inst.a, inst.b, inst.C, rho=config.ADMM_RHO,
            mu=config.ADMM_RESBAL_MU, tau=config.ADMM_RESBAL_TAU,
            adapt_every=config.ADMM_RESBAL_EVERY, adapt_until=config.ADMM_RESBAL_MAX_ADAPT,
            **common,
        )
        resbal_ablation.append(_row(inst, {"variant": "fixed rho"}, **fixed))
        resbal_ablation.append(_row(inst, {"variant": "residual balancing"}, **resbal))

        for eta in config.PDHG_ETA_GRID:
            out = pdhg.solve_pdhg(inst.a, inst.b, inst.C, eta=eta, restart=False, **common)
            eta_sweep.append(_row(inst, {"eta": eta}, **out))
            print(f"[C] eta sweep {inst.name} eta={eta} feas={out['final_feasibility_error']:.1e}")

        for restart in (False, True):
            out = pdhg.solve_pdhg(
                inst.a, inst.b, inst.C, eta=config.PDHG_ETA, restart=restart,
                restart_every=config.PDHG_RESTART_EVERY,
                restart_beta=config.PDHG_RESTART_BETA,
                **common,
            )
            restart_ablation.append(_row(inst, {"variant": f"restart={restart}"}, **out))

        # ablation: plain PDHG vs the Pock-Chambolle diagonal preconditioner
        for name, pre in (("plain", False), ("preconditioned", True)):
            out = pdhg.solve_pdhg(
                inst.a, inst.b, inst.C, eta=config.PDHG_ETA, precondition=pre, **common
            )
            precondition_ablation.append(_row(inst, {"variant": name}, **out))
            print(f"[C] precond ablation {inst.name} {name} "
                  f"feas={out['final_feasibility_error']:.1e}")

    save_json(
        {
            "budget": BUDGET,
            "tolerances": list(TOLS),
            "pilot_instances": [i.name for i in instances],
            "pilot_seeds_note": (
                "pilot seeds 10/11/12 are disjoint from the evaluation seeds "
                "0-4 used in experiment B"
            ),
            "rho_sweep": rho_sweep,
            "eta_sweep": eta_sweep,
            "resbal_ablation": resbal_ablation,
            "restart_ablation": restart_ablation,
            "precondition_ablation": precondition_ablation,
        },
        RESULTS_DIR / "experiment_c.json",
    )


if __name__ == "__main__":
    main()
