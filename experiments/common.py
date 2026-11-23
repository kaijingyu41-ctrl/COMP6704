"""Shared helpers for the experiment drivers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otopt import admm, config, metrics, pdhg, reference  # noqa: E402
from otopt.certification import certified_bracket, certify  # noqa: E402,F401

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR

#: identical iteration/time budget for every method
BUDGET = dict(max_iter=config.MAX_ITER, time_limit=config.TIME_LIMIT, log_every=config.LOG_EVERY)

#: methods implemented for this project (the HiGHS reference is handled apart)
METHOD_NAMES = ("ADMM (fixed rho)", "ADMM (residual balancing)", "PDHG", "PDHG (restart)")

#: colour-blind friendly palette
COLORS = {
    "ADMM (fixed rho)": "#1f77b4",
    "ADMM (residual balancing)": "#d62728",
    "PDHG": "#2ca02c",
    "PDHG (restart)": "#ff7f0e",
    "HiGHS (dual simplex)": "#555555",
}
MARKERS = {
    "ADMM (fixed rho)": "o",
    "ADMM (residual balancing)": "s",
    "PDHG": "^",
    "PDHG (restart)": "v",
    "HiGHS (dual simplex)": "D",
}


def solve_reference(inst) -> Dict[str, Any]:
    """HiGHS reference solution for an instance."""
    return reference.solve_reference(inst.a, inst.b, inst.C)


def run_methods(
    inst,
    f_ref: float,
    tolerances: Sequence[float],
    budget: Dict[str, Any] | None = None,
    f_lower: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Run every implemented method on one instance with the same budget."""
    b = dict(BUDGET if budget is None else budget)
    common = dict(tolerances=tuple(tolerances), f_ref=f_ref, **b)
    results = {
        "ADMM (fixed rho)": admm.solve_fixed_admm(inst.a, inst.b, inst.C, rho=config.ADMM_RHO, **common),
        "ADMM (residual balancing)": admm.solve_residual_balancing_admm(
            inst.a, inst.b, inst.C,
            rho=config.ADMM_RHO,
            mu=config.ADMM_RESBAL_MU,
            tau=config.ADMM_RESBAL_TAU,
            adapt_every=config.ADMM_RESBAL_EVERY,
            adapt_until=config.ADMM_RESBAL_MAX_ADAPT,
            **common,
        ),
        "PDHG": pdhg.solve_pdhg(inst.a, inst.b, inst.C, eta=config.PDHG_ETA, restart=False, **common),
        "PDHG (restart)": pdhg.solve_pdhg(
            inst.a, inst.b, inst.C, eta=config.PDHG_ETA, restart=True,
            restart_every=config.PDHG_RESTART_EVERY, restart_beta=config.PDHG_RESTART_BETA,
            **common,
        ),
    }
    for out in results.values():
        certify(out, inst, f_lower=f_lower)
    return results


def result_record(out: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the (large) solution matrix and dual vectors, and compact the trace."""
    rec = {k: v for k, v in out.items() if k not in ("plan", "alpha", "beta")}
    rec["history"] = metrics.downsample_history(out["history"], config.HISTORY_POINTS)
    return rec


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)
    print(f"[saved] {path}")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)!r}")
