"""Evaluation metrics for the transport LP.

The transport problem is solved to *both* primal feasibility and optimality, so
a candidate plan is never judged by its objective value alone. An infeasible
iterate can always be pushed to a low objective value by violating the marginal
constraints, which is not a valid solution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def feasibility_error(T: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """L1 violation of the two marginal constraints."""
    row = np.abs(T.sum(axis=1) - a).sum()
    col = np.abs(T.sum(axis=0) - b).sum()
    return float(row + col)


def objective_value(T: np.ndarray, C: np.ndarray) -> float:
    """Linear transport cost ``<C, T>``."""
    return float(np.sum(C * T))


def negative_mass(T: np.ndarray) -> float:
    """Total amount of negative mass (should be zero for a feasible plan)."""
    return float(np.abs(np.minimum(T, 0.0)).sum())


def kkt_residuals(
    T: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> Dict[str, float]:
    """Primal feasibility and dual/complementary-slackness residuals.

    ``alpha`` and ``beta`` are dual potentials; the dual constraint is
    ``alpha_i + beta_j <= C_ij`` and complementary slackness requires
    ``T_ij (C_ij - alpha_i - beta_j) = 0``.
    """
    dual_matrix = alpha[:, None] + beta[None, :]
    return {
        "primal_ineq": feasibility_error(T, a, b),
        "dual_violation": float(np.abs(np.minimum(C - dual_matrix, 0.0)).sum()),
        "complementarity": float(np.abs(T * (C - dual_matrix)).sum()),
    }


def dual_feasible_pair(
    alpha: np.ndarray,
    beta: np.ndarray,
    C: np.ndarray,
) -> Dict[str, object]:
    """Repair a candidate dual pair so that ``alpha_i + beta_j <= C_ij`` holds.

    ``alpha`` is kept and ``beta`` is replaced by the largest vector that makes the
    pair feasible for that ``alpha``, ``beta_j = min_i (C_ij - alpha_i)``. This is
    the repair used in first-order LP solvers to turn an approximate dual iterate
    into a dual feasible point.

    The direction of the change depends on the candidate, and it is worth being
    precise because the bound moves in the opposite direction to the dual objective:

    * if the candidate was **feasible**, every component can only grow
      (``beta_new >= beta``), so the dual objective ``a'alpha + b'beta`` does not
      decrease and the resulting bound is *tighter*;
    * if it was **infeasible**, some components are cut back to the feasible value
      and the dual objective can decrease, at the price of a looser bound.

    Either way the pair is feasible, so ``a'alpha + b'beta <= f*`` and the
    certificate of :func:`certified_suboptimality` stays valid.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    beta_repaired = np.min(C - alpha[:, None], axis=0)
    violation = float(np.max(alpha[:, None] + beta[None, :] - C))
    return {
        "alpha": alpha,
        "beta": beta_repaired,
        "max_violation_before_repair": violation,
    }


def polish_feasibility(
    T: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-15,
) -> np.ndarray:
    """Iterative proportional fitting: drive the marginals of ``T`` onto ``a``, ``b``.

    Row and column rescalings keep ``T`` non-negative, so the result is still a
    valid argument for :func:`certified_suboptimality`, and they scale between
    the two marginals only, so the support is preserved. The iteration is stopped
    as soon as the marginal error reaches ``tol``; that stopping test matters,
    because a fixed iteration count gives no guarantee that the result is
    feasible, and the caller must not treat the output as a feasible plan unless
    :func:`feasibility_error` says so. A support that cannot carry ``a`` and
    ``b`` (a zero row or column) is returned unchanged in the offending
    coordinate and keeps a non-zero residual.
    """
    X = np.maximum(np.asarray(T, dtype=float), 0.0)
    for _ in range(max_iter):
        rows = X.sum(axis=1)
        X = X * (a / np.where(rows > 0.0, rows, 1.0))[:, None]
        cols = X.sum(axis=0)
        X = X * (b / np.where(cols > 0.0, cols, 1.0))[None, :]
        if feasibility_error(X, a, b) <= tol:
            break
    return X


def certified_suboptimality(
    T: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> Dict[str, float]:
    """Upper bound on ``<C,T> - f*`` that uses no reference optimum.

    For any ``T >= 0`` and any dual feasible ``(alpha, beta)``
    (``alpha_i + beta_j <= C_ij``), weak duality (``a'alpha + b'beta <= f*``) gives

        <C,T> - f*  <=  <C - alpha - beta, T>
                        + alpha'(T1 - a) + beta'(T^T1 - b)
                    =  <C,T> - (a'alpha + b'beta).

    Every term on the right is computable from ``T`` alone, so the bound certifies
    the suboptimality of a candidate plan without solving the LP to optimality.

    Three things the bound is *not* are as important as what it is. It is an
    upper bound: a small value is good news and a large value is not evidence
    that the error is large. It is valid for infeasible ``T``, and it can then be
    zero or negative even though ``T`` is neither feasible nor optimal --- for
    instance ``T = 0`` with ``alpha = beta = 0`` gives a bound of zero. And it
    says nothing about the size of the error unless the marginal terms are
    controlled, which is why every stored run also records
    :func:`feasibility_error` and the two are read together.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    reduced = C - alpha[:, None] - beta[None, :]
    if np.any(reduced < -1e-12):
        raise ValueError("the dual pair is not feasible: alpha_i + beta_j > C_ij")
    complementarity = float(np.sum(reduced * T))
    row_term = float(alpha @ (T.sum(axis=1) - a))
    col_term = float(beta @ (T.sum(axis=0) - b))
    return {
        "bound": complementarity + row_term + col_term,
        "complementarity_term": complementarity,
        "row_term": row_term,
        "col_term": col_term,
    }


def record(
    iteration: int,
    elapsed: float,
    T: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    f_ref: Optional[float],
) -> Dict[str, float]:
    """One diagnostic row of a solver trace."""
    obj = objective_value(T, C)
    row: Dict[str, float] = {
        "iteration": int(iteration),
        "time": float(elapsed),
        "objective": obj,
        "feasibility_error": feasibility_error(T, a, b),
        "negative_mass": negative_mass(T),
    }
    if f_ref is not None:
        row["objective_error"] = abs(obj - f_ref)
    return row


def time_to_tolerance(
    history: Sequence[Dict[str, float]],
    tolerance: float,
) -> Optional[Dict[str, float]]:
    """First trace point where **both** errors are below ``tolerance``.

    Returns ``None`` when the tolerance was never reached within the budget.
    """
    for row in history:
        feas = row.get("feasibility_error", np.inf)
        obj = row.get("objective_error", np.inf)
        if feas <= tolerance and obj <= tolerance:
            return {"iteration": row["iteration"], "time": row["time"]}
    return None


def trace_point_at(
    history: Sequence[Dict[str, float]],
    iteration: int,
) -> Optional[Dict[str, float]]:
    """The trace point at exactly ``iteration``, or ``None`` if it was never reached.

    Diagnostics are recorded every ``log_every`` iterations, so a checkpoint that
    the solver actually reached appears in the trace at its own iteration count.
    Returning the last earlier point instead --- the obvious fallback --- would
    silently relabel it with a checkpoint the run never got to, which is exactly
    what happens when a time limit ends a long-budget run early. The guard is
    therefore part of the measurement, not of the presentation.
    """
    found: Optional[Dict[str, float]] = None
    for row in history:
        if row["iteration"] <= iteration:
            found = row
        else:
            break
    if found is None or found["iteration"] < iteration:
        return None
    return found


def downsample_history(
    history: Sequence[Dict[str, float]],
    max_points: int = 150,
) -> List[Dict[str, float]]:
    """Log-spaced sub-sampling of a solver trace for compact storage.

    Convergence traces span four orders of magnitude in both axes, so a
    logarithmic index selection keeps the shape of the curve while reducing the
    amount of data written to disk.
    """
    n = len(history)
    if n <= max_points:
        return list(history)
    idx = np.unique(np.round(np.logspace(0, np.log10(n - 1), max_points)).astype(int))
    return [history[int(i)] for i in idx]


def summarise_run(
    history: Sequence[Dict[str, float]],
    T: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    f_ref: float,
    tolerances: Sequence[float],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Condense a solver trace into one result record.

    ``f_ref`` is optional: the solvers, their traces and their optimality
    certificate need no reference value at all, so ``summarise_run`` works without
    one and simply omits the reference-based objective error. The benchmark passes
    it only to report that comparison alongside.
    """
    final = record(history[-1]["iteration"], history[-1]["time"], T, a, b, C, f_ref)
    out: Dict[str, object] = {
        "iterations": history[-1]["iteration"],
        "time": history[-1]["time"],
        "final_objective": final["objective"],
        "final_feasibility_error": final["feasibility_error"],
        "final_objective_error": final.get("objective_error"),
        "final_negative_mass": final["negative_mass"],
        "reference_objective": None if f_ref is None else float(f_ref),
        "history": list(history),
    }
    for tol in tolerances:
        key = f"{tol:.0e}"
        hit = time_to_tolerance(history, tol)
        out[f"iter_to_{key}"] = None if hit is None else hit["iteration"]
        out[f"time_to_{key}"] = None if hit is None else hit["time"]
    if extra:
        out.update(extra)
    return out
