"""Primal-dual hybrid gradient (Chambolle-Pock) for the transport LP.

Saddle-point form
-----------------
With ``x = vec(X)``, ``c = vec(C)``, ``q = [a; b]`` and

    A x = [ X 1 ; X^T 1 ]   (an (m + n) x (mn) 0/1 operator),

the transport LP is

    min_{x >= 0} max_y  <c, x> + <y, A x - q>.

Applying the Chambolle-Pock (2011) scheme with ``g(x) = <c, x> + I_+(x)`` and
``f*(y) = <y, q>`` gives

    y^{k+1} = y^k + sigma ( A x_bar^k - q )
    x^{k+1} = max( 0, x^k - tau ( c + A^T y^{k+1} ) )
    x_bar^{k+1} = 2 x^{k+1} - x^k

with step sizes satisfying ``tau * sigma * ||A||_2^2 < 1``. For this operator
``||A||_2^2 = m + n``, so ``tau = sigma = eta / sqrt(m + n)`` with ``eta < 1`` is
safe. The operator ``A`` and its adjoint are applied through row/column sums, so
no ``(m+n) x (mn)`` dense matrix is ever formed.

An optional restart heuristic, in the spirit of the adaptive restarting used by
PDLP (Applegate et al., 2021), resets the extrapolation whenever the primal
feasibility residual fails to decrease by a fixed fraction. The comparison is with
the residual recorded at the *previous check* (every ``restart_every`` iterations),
which is the rule implemented here; comparing with the residual at the last restart
is the PDLP variant and is not implemented.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import metrics


def _operator(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``A x - q`` where ``A x = [X 1 ; X^T 1]``."""
    return np.concatenate([X.sum(axis=1) - a, X.sum(axis=0) - b])


def _adjoint(y: np.ndarray, m: int, n: int) -> np.ndarray:
    """``A^T y`` as an ``m x n`` matrix."""
    return y[:m, None] + y[None, m:]


def pc_preconditioner(m: int, n: int, eta: float = 0.99):
    """Pock-Chambolle diagonal preconditioner for this operator.

    With ``alpha = 1`` the rule of Pock & Chambolle (2011) is
    ``tau_j = 1 / sum_i |A_ij|`` and ``sigma_i = 1 / sum_j |A_ij|``. Every
    variable appears in exactly two constraints, so ``tau`` is the scalar
    ``1/2``; each row constraint contains ``n`` ones and each column constraint
    ``m`` ones, so ``sigma`` is ``1/n`` on the first ``m`` entries and ``1/m`` on
    the remaining ``n``.

    The convergence condition is ``||Sigma^{1/2} A T^{1/2}|| <= 1``. For this
    operator the bound ``||M x||^2 <= (||X1||^2/n + ||X^T1||^2/m)/2 <= 1`` holds
    with equality at ``X = 11^T/sqrt(mn)``, so the condition is tight and we
    scale both steps by ``eta < 1`` to make it strict.
    """
    tau = eta * 0.5
    sigma = np.concatenate([np.full(m, eta / n), np.full(n, eta / m)])
    return tau, sigma


def restart_decision(r: float, r_prev: float, beta: float) -> Tuple[bool, float]:
    """The restart test and the bookkeeping it implies, in one place.

    ``r`` is the primal residual at the current check (every ``restart_every``
    iterations) and ``r_prev`` the one recorded at the previous check. A restart
    fires when the residual failed to decrease by the fraction ``beta``. The
    returned reference is *always* the current residual, whether or not a restart
    fired --- so the next comparison is against the previous check, which is the
    criterion described in Section 4.2 of the report. Comparing with the residual
    at the last restart instead is the PDLP variant and is not implemented here;
    returning the pair keeps that choice in one testable place rather than in the
    loop's control flow.

    The criterion is deliberately weaker than PDLP's normalised duality gap: it
    uses only the primal residual, so a restart can fire while the dual iterate is
    still far from converged.
    """
    fire = bool(r >= (1.0 - beta) * r_prev)
    return fire, float(r)


def _run(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eta: float,
    restart: bool,
    restart_every: int,
    restart_beta: float,
    precondition: bool,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float],
) -> Tuple[np.ndarray, list, Dict[str, Any]]:
    m, n = C.shape
    if precondition:
        tau, sigma = pc_preconditioner(m, n, eta)
    else:
        tau = sigma = eta / np.sqrt(m + n)

    x = np.outer(a, b).ravel().copy()
    x_bar = x.copy()
    y = np.zeros(m + n)
    q = np.concatenate([a, b])

    history: list = []
    n_restarts = 0
    start = time.perf_counter()

    def snapshot(k: int, X: np.ndarray) -> None:
        history.append(metrics.record(k, time.perf_counter() - start, X, a, b, C, f_ref))

    snapshot(0, x.reshape(m, n))
    #: primal residual at the previous check (not at the previous restart)
    r_prev = float(np.abs(_operator(x.reshape(m, n), a, b)).sum())
    k = 0
    for k in range(1, max_iter + 1):
        # dual ascent on the (linear) conjugate of the equality indicator
        y = y + sigma * _operator(x_bar.reshape(m, n), a, b)
        # primal descent with the non-negativity prox
        gradient = C + _adjoint(y, m, n)
        x_new = np.maximum(0.0, x - tau * gradient.ravel())
        x_bar = 2.0 * x_new - x
        x = x_new

        if restart and k % restart_every == 0:
            r = float(np.abs(_operator(x.reshape(m, n), a, b)).sum())
            fire, r_prev = restart_decision(r, r_prev, restart_beta)
            if fire:
                x_bar = x.copy()          # erase the momentum / restart the method
                n_restarts += 1

        if k % log_every == 0:
            snapshot(k, x.reshape(m, n))

        if (time.perf_counter() - start) > time_limit:
            break

    if history[-1]["iteration"] != k:
        snapshot(k, x.reshape(m, n))

    info = {
        "tau": float(np.mean(tau)) if np.ndim(tau) else float(tau),
        "sigma": float(np.mean(sigma)) if np.ndim(sigma) else float(sigma),
        "precondition": bool(precondition),
        "n_restarts": n_restarts,
        "budget_exhausted": bool(k >= max_iter or (time.perf_counter() - start) > time_limit),
        # The dual iterate y is the multiplier of the marginal equalities, so
        # (-y[:m], -y[m:]) are the OT potentials alpha, beta of (2). PDHG's dual
        # feasibility A^T y + c >= 0 is exactly alpha_i + beta_j <= C_ij, so the
        # pair is feasible once the dual iterate has converged.
        "alpha": -y[:m],
        "beta": -y[m:],
    }
    return x.reshape(m, n), history, info


def solve_pdhg(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eta: float,
    max_iter: int,
    time_limit: float,
    log_every: int,
    tolerances: Sequence[float],
    f_ref: Optional[float] = None,
    restart: bool = False,
    restart_every: int = 200,
    restart_beta: float = 0.1,
    precondition: bool = False,
) -> Dict[str, Any]:
    """Plain PDHG, optionally with restarting and/or diagonal preconditioning."""
    T, history, info = _run(
        a, b, C, eta, restart, restart_every, restart_beta, precondition,
        max_iter, time_limit, log_every, tolerances, f_ref,
    )
    out = metrics.summarise_run(history, T, a, b, C, f_ref, tolerances, info)
    name = "PDHG"
    if precondition:
        name += " (preconditioned)"
    if restart:
        name += " (restart)"
    out["method"] = name
    out["plan"] = T
    return out
