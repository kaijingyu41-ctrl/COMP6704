"""ADMM for the discrete optimal transport linear program.

Problem
-------
``min_X <C, X>`` subject to ``X 1 = a``, ``X^T 1 = b`` and ``X >= 0``.

Splitting
---------
We introduce a second block ``Z`` that carries the non-negativity constraint::

    min_{X, Z}  <C, X> + I_A(X) + I_+(Z)     s.t.  X = Z,

where ``A = {X : X 1 = a, X^T 1 = b}`` is an affine set and ``I_+`` is the
indicator of the non-negative orthant. The scaled-form ADMM iterations are::

    X^{k+1} = Proj_A( Z^k - U^k - C / rho )
    Z^{k+1} = max( 0, X^{k+1} + U^k )
    U^{k+1} = U^k + X^{k+1} - Z^{k+1}

The projection onto ``A`` has a closed form because ``A`` is the intersection of
two marginal-constraint sets, which is an affine subspace of matrices::

    Proj_A(W) = W + r 1^T / n + 1 s^T / m - d / (m n) * 1 1^T

with ``r = a - W 1``, ``s = b - W^T 1`` and ``d = sum(a) - sum(W)``.

The non-negative block ``Z`` is the quantity we report, because it satisfies the
non-negativity constraint exactly and converges to the affine-feasible block as
the algorithm converges.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import metrics


# ------------------------------------------------------------------ projection
def project_affine(W: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of ``W`` onto ``{X : X1 = a, X^T1 = b}``."""
    m, n = W.shape
    r = a - W.sum(axis=1)
    s = b - W.sum(axis=0)
    d = float(a.sum() - W.sum())
    return W + r[:, None] / n + s[None, :] / m - d / (m * n)


# ----------------------------------------------------------------------- driver
def _run(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    rho0: float,
    adaptive: bool,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float],
    mu: float,
    tau: float,
    adapt_every: int,
    adapt_until: int,
) -> Tuple[np.ndarray, list, Dict[str, Any]]:
    m, n = C.shape
    rho = float(rho0)

    X = project_affine(np.outer(a, b), a, b)
    Z = np.outer(a, b)
    U = np.zeros_like(C)

    history: list = []
    n_rho_changes = 0
    start = time.perf_counter()

    def snapshot(k: int) -> None:
        history.append(
            metrics.record(k, time.perf_counter() - start, Z, a, b, C, f_ref)
        )

    snapshot(0)
    k = 0
    # The dual potentials of the marginal constraints come from the optimality
    # condition of the X block. X^{k+1} is the exact minimiser of
    # <C,X> + (rho/2)||X - Z^k + U^k||^2 over the affine set, so
    #     C + rho (X^{k+1} - Z^k + U^k) = alpha 1^T + 1 beta^T ,
    # where the right-hand side is the multiplier of the two marginal
    # equalities: a sum of a row vector and a column vector, i.e. an additively
    # separable matrix (rank at most two, not one). The residual must therefore be
    # formed with the Z and U that were used *inside* the projection, before they
    # are overwritten by the Z- and U-updates of the same iteration; using the
    # updated blocks destroys that exact separable structure, and the two-vector
    # representation no longer reproduces the matrix it is supposed to represent.
    stationarity = np.zeros_like(C)
    for k in range(1, max_iter + 1):
        X = project_affine(Z - U - C / rho, a, b)
        stationarity = C + rho * (X - Z + U)
        Z_prev = Z
        Z = np.maximum(0.0, X + U)
        U = U + X - Z

        if adaptive and k <= adapt_until and k % adapt_every == 0:
            primal = float(np.linalg.norm(X - Z))
            dual = rho * float(np.linalg.norm(Z - Z_prev))
            if primal > mu * dual:
                rho *= tau
                U /= tau
                n_rho_changes += 1
            elif dual > mu * primal:
                rho /= tau
                U *= tau
                n_rho_changes += 1

        if k % log_every == 0:
            snapshot(k)

        # No early stopping: every method is allowed exactly the same budget, so
        # that the comparison is unambiguous. (A residual-based stopping rule
        # would need *both* the primal and the dual residual, and the primal
        # residual alone can vanish spuriously on degenerate instances, e.g. a
        # 2x2 problem where the non-negative block equals the affine block from
        # the first iteration onwards while the dual variable is still far from
        # converged.)
        if (time.perf_counter() - start) > time_limit:
            break

    if history[-1]["iteration"] != k:
        snapshot(k)

    info = {
        "rho_final": rho,
        "n_rho_changes": n_rho_changes,
        "budget_exhausted": bool(k >= max_iter or (time.perf_counter() - start) > time_limit),
    }

    # Read the potentials off the additively separable stationarity residual of the
    # last X block: alpha is the row-wise mean after removing the column means, and
    # beta picks up what is left, so alpha[:, None] + beta[None, :] reproduces the
    # residual exactly. The splitting is fixed only up to the gauge
    # alpha -> alpha + theta, beta -> beta - theta, which leaves a'alpha + b'beta
    # unchanged because both marginals sum to one.
    alpha = (stationarity - stationarity.mean(axis=0, keepdims=True)).mean(axis=1)
    beta = (stationarity - alpha[:, None]).mean(axis=0)
    info["alpha"] = alpha
    info["beta"] = beta
    info["stationarity_residual"] = float(
        np.abs(stationarity - (alpha[:, None] + beta[None, :])).max()
    )
    return Z, history, info


# ------------------------------------------------------------------- public API
def solve_fixed_admm(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    rho: float,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float] = None,
) -> Dict[str, Any]:
    """ADMM with a constant penalty parameter ``rho``."""
    Z, history, info = _run(
        a, b, C, rho, False, max_iter, time_limit, log_every, tolerances, f_ref,
        mu=0.0, tau=1.0, adapt_every=0, adapt_until=0,
    )
    out = metrics.summarise_run(history, Z, a, b, C, f_ref, tolerances, info)
    out["method"] = "ADMM (fixed rho)"
    out["plan"] = Z
    return out


def solve_residual_balancing_admm(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    rho: float,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float] = None,
    mu: float = 10.0,
    tau: float = 2.0,
    adapt_every: int = 25,
    adapt_until: int = 1000,
) -> Dict[str, Any]:
    """ADMM with He et al.-style residual balancing (a known heuristic)."""
    Z, history, info = _run(
        a, b, C, rho, True, max_iter, time_limit, log_every, tolerances, f_ref,
        mu=mu, tau=tau, adapt_every=adapt_every, adapt_until=adapt_until,
    )
    out = metrics.summarise_run(history, Z, a, b, C, f_ref, tolerances, info)
    out["method"] = "ADMM (residual balancing)"
    out["plan"] = Z
    return out
