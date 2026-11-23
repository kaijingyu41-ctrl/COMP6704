"""Sinkhorn iterations for the entropically regularised transport problem.

This module is the *regularised* baseline of the study. It does not solve the LP
of the report: it solves

    min_{X >= 0}  <C, X> + eps * sum_ij X_ij (log X_ij - 1)
    s.t.  X 1 = a,  X^T 1 = b,

whose optimum differs from ``f*`` by a bias of order ``eps log(mn)``. That bias is
exactly what makes the comparison interesting: Sinkhorn is by far the cheapest way
to obtain a *feasible* plan, and the question the benchmark answers is what that
speed costs in optimality for the unregularised problem.

The implementation is in the log domain, which is the numerically stable form:

    f_i = eps (log a_i - logsumexp_j((g_j - C_ij) / eps))
    g_j = eps (log b_j - logsumexp_i((f_i - C_ij) / eps))
    X_ij = exp((f_i + g_j - C_ij) / eps)

so no explicit kernel matrix is formed and ``eps`` can be small. Each iteration is
two reductions over the ``m x n`` cost matrix, i.e. ``O(mn)`` time and memory ---
the same per-iteration budget as ADMM and PDHG, which is what makes the wall-clock
comparison meaningful. The potentials passed on for the optimality certificate are
``(f, g)`` themselves, not their negation: with the plan written as
``exp((f_i + g_j - C_ij) / eps)`` they are the dual variables in the same
``max a'alpha + b'beta`` pairing as the LP dual (see the note in ``_run``). They are
feasible for the *regularised* dual only, so the certificate repairs them first, and
on this benchmark they turn out to be feasible for the LP dual already.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import metrics


def _logsumexp(Z: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(Z, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return (peak + np.log(np.sum(np.exp(Z - peak), axis=axis, keepdims=True))).squeeze(axis)


def _run(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float],
) -> Tuple[np.ndarray, list, Dict[str, Any]]:
    m, n = C.shape
    log_a = np.log(a)
    log_b = np.log(b)
    f = np.zeros(m)
    g = np.zeros(n)

    def plan() -> np.ndarray:
        return np.exp((f[:, None] + g[None, :] - C) / eps)

    history: list = []
    start = time.perf_counter()

    def snapshot(k: int) -> None:
        history.append(metrics.record(k, time.perf_counter() - start, plan(), a, b, C, f_ref))

    snapshot(0)
    k = 0
    for k in range(1, max_iter + 1):
        f = eps * (log_a - _logsumexp((g[None, :] - C) / eps, axis=1))
        g = eps * (log_b - _logsumexp((f[:, None] - C) / eps, axis=0))

        if k % log_every == 0:
            snapshot(k)

        if (time.perf_counter() - start) > time_limit:
            break

    if history[-1]["iteration"] != k:
        snapshot(k)

    info = {
        "eps": float(eps),
        # The scaling vectors f, g are the dual variables of the marginal
        # equalities in the convention used by the LP dual ``max a'alpha +
        # b'beta``: writing the Lagrangian with multipliers u, v gives u = -f and
        # v = -g, and at the optimum the Lagrangian equals ``a'f + b'g - eps``,
        # i.e. the same pairing of potentials with the marginals as the LP dual.
        # So the pair to hand to the certificate is (f, g) itself --- *not* its
        # negation. It is feasible for the regularised dual only, so the
        # certificate repairs it before use.
        "alpha": f,
        "beta": g,
        "regularised": True,
        "budget_exhausted": bool(k >= max_iter or (time.perf_counter() - start) > time_limit),
    }
    return plan(), history, info


def solve_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float] = None,
) -> Dict[str, Any]:
    """Sinkhorn iterations for the entropically regularised problem at one ``eps``."""
    T, history, info = _run(a, b, C, eps, max_iter, time_limit, log_every, tolerances, f_ref)
    out = metrics.summarise_run(history, T, a, b, C, f_ref, tolerances, info)
    out["method"] = f"Sinkhorn (eps={eps:g})"
    out["plan"] = T
    return out


def _run_continuation(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps_schedule: Sequence[float],
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float],
) -> Tuple[np.ndarray, list, Dict[str, Any]]:
    """Log-domain Sinkhorn on a decreasing ``eps`` schedule, warm-starting ``(f, g)``.

    This is the standard way to use Sinkhorn when the goal is close to the
    *unregularised* problem: run at a large ``eps`` where the iterations converge
    quickly, then shrink ``eps`` and carry the scaling vectors over. Warm starting is
    what separates it from restarting at each ``eps`` --- ``f`` and ``g`` are kept.

    The budget is spent rather than multiplied: the stages share ``max_iter``
    iterations, so the comparison against a single fixed ``eps`` given the same
    ``max_iter`` is budget-for-budget. That is the honest comparison, and it is also
    the demanding one, because the small-``eps`` stages are the slow ones.

    One continuous trace is produced, so ``time_to_tolerance`` and the certificate
    see the whole run rather than one stage of it.
    """
    m, n = C.shape
    log_a = np.log(a)
    log_b = np.log(b)
    f = np.zeros(m)
    g = np.zeros(n)

    scheme = tuple(float(e) for e in eps_schedule)
    if not scheme or min(scheme) <= 0.0:
        raise ValueError("eps_schedule must be a non-empty sequence of positive values")

    def plan(eps: float) -> np.ndarray:
        return np.exp((f[:, None] + g[None, :] - C) / eps)

    history: list = [metrics.record(0, 0.0, plan(scheme[0]), a, b, C, f_ref)]
    start = time.perf_counter()
    k = 0
    eps_used = scheme[0]
    stopped = False
    #: nominal per-stage allocation, reported for the record; the loop below
    #: recomputes the split as stages are consumed so that the total never exceeds
    #: ``max_iter`` even when the budget is smaller than the number of stages
    #: (``max_iter // len(scheme)`` alone would give every stage one iteration and
    #: overshoot).
    per_stage = max(1, int(max_iter) // len(scheme))
    remaining = max(0, int(max_iter))
    for stage, eps in enumerate(scheme):
        stages_left = len(scheme) - stage
        n_stage = remaining if stages_left == 1 else remaining // stages_left
        for _ in range(n_stage):
            k += 1
            eps_used = eps
            f = eps * (log_a - _logsumexp((g[None, :] - C) / eps, axis=1))
            g = eps * (log_b - _logsumexp((f[:, None] - C) / eps, axis=0))
            if k % log_every == 0:
                history.append(
                    metrics.record(k, time.perf_counter() - start, plan(eps), a, b, C, f_ref)
                )
            if (time.perf_counter() - start) > time_limit:
                stopped = True
                break
        remaining -= n_stage
        if stopped:
            break

    if history[-1]["iteration"] != k:
        history.append(
            metrics.record(k, time.perf_counter() - start, plan(eps_used), a, b, C, f_ref)
        )

    info = {
        "eps": eps_used,
        "eps_schedule": list(scheme),
        "stages": len(scheme),
        "iterations_per_stage": per_stage,
        "alpha": f,
        "beta": g,
        "regularised": True,
        "budget_exhausted": bool(k >= max_iter or stopped),
    }
    return plan(eps_used), history, info


def solve_sinkhorn_continuation(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps_schedule: Sequence[float],
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float] = None,
) -> Dict[str, Any]:
    """Sinkhorn on a decreasing ``eps`` schedule, within one fixed-``max_iter`` budget."""
    T, history, info = _run_continuation(
        a, b, C, eps_schedule, max_iter, time_limit, log_every, tolerances, f_ref
    )
    out = metrics.summarise_run(history, T, a, b, C, f_ref, tolerances, info)
    out["method"] = f"Sinkhorn (continuation to eps={float(eps_schedule[-1]):g})"
    out["plan"] = T
    return out
