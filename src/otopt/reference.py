"""High-accuracy reference solutions via the HiGHS dual simplex solver.

The reference solution is used only as a numerical yardstick: it provides the
value ``f_ref`` against which the objective error of the first-order methods is
measured, and a feasibility certificate for the instances themselves. It is a
library call (``scipy.optimize.linprog``), **not** an implementation of our own.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from . import metrics


def build_constraint_matrix(m: int, n: int):
    """Sparse equality matrix for the row and column marginal constraints.

    All ``m`` row constraints are kept and only the *last* column constraint is
    dropped. The full system has rank ``m + n - 1`` (its only linear dependency
    is ``sum_i row_i = sum_j col_j = total mass``), so removing exactly one
    equation leaves a full-row-rank system whose solution set is unchanged.
    Removing two equations would *enlarge* the feasible set. Variables are
    stored row-major, ``x[i * n + j]``.
    """
    rows, cols = [], []
    for i in range(m):                           # row marginal constraints
        rows.extend([i] * n)
        cols.extend([i * n + j for j in range(n)])
    for j in range(n - 1):                       # column marginal constraints
        r = m + j
        rows.extend([r] * m)
        cols.extend([i * n + j for i in range(m)])
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(m + (n - 1), m * n)).tocsr()
    return A


def solve_reference(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve the transport LP with HiGHS and return the optimum and its plan."""
    m, n = C.shape
    A_eq = build_constraint_matrix(m, n)
    b_eq = np.concatenate([a, b[:-1]])
    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    start = time.perf_counter()
    res = linprog(
        C.ravel(),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs-ds",
        options=options or None,
    )
    elapsed = time.perf_counter() - start

    if not res.success:
        raise RuntimeError(f"HiGHS failed on a {m}x{n} instance: {res.message}")

    T = np.asarray(res.x, dtype=np.float64).reshape(m, n)
    feas = metrics.feasibility_error(T, a, b)
    if feas > 1e-6:
        raise RuntimeError(f"HiGHS solution is not feasible (error {feas:.3e})")

    return {
        "solver": "highs-ds",
        "objective": float(res.fun),
        "plan": T,
        "feasibility_error": feas,
        "negative_mass": metrics.negative_mass(T),
        "time": elapsed,
        "iterations": int(getattr(res, "nit", 0)),
        "status": int(res.status),
    }


def build_dual_constraint_matrix(m: int, n: int):
    """Sparse inequality matrix of the dual ``alpha_i + beta_j <= C_ij``."""
    rows, cols = [], []
    for i in range(m):
        for j in range(n):
            rows.append(i * n + j)
            cols.append(i)
            rows.append(i * n + j)
            cols.append(m + j)
    data = np.ones(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m * n, m + n)).tocsr()


def solve_dual_reference(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> Dict[str, Any]:
    """Solve the dual transport LP directly to obtain optimal potentials.

    The dual is ``max a^T alpha + b^T beta`` subject to
    ``alpha_i + beta_j <= C_ij``. Solving it independently gives a certificate
    that can be used to check the complementary-slackness conditions of a primal
    plan produced by any of our solvers; this is what the KKT unit test does. The
    variables are free, so no bounds are imposed.
    """
    m, n = C.shape
    A_ub = build_dual_constraint_matrix(m, n)
    c = -np.concatenate([a, b])
    res = linprog(c, A_ub=A_ub, b_ub=C.ravel(), bounds=(None, None), method="highs-ds")
    if not res.success:
        raise RuntimeError(f"dual HiGHS solve failed: {res.message}")
    alpha = np.asarray(res.x[:m], dtype=np.float64)
    beta = np.asarray(res.x[m:], dtype=np.float64)
    return {
        "alpha": alpha,
        "beta": beta,
        "objective": float(-res.fun),
        "dual_max_violation": float(np.max(alpha[:, None] + beta[None, :] - C)),
    }
