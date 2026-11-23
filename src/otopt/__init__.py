"""Discrete optimal transport solvers: ADMM, PDHG and a HiGHS reference.

The package implements, from scratch, two families of first-order methods for the
discrete optimal transport linear program, together with the data generation,
evaluation metrics, optimality certificates and plotting utilities used by the
experiments.
"""

from . import certification, config, data, metrics, reference, admm, pdhg, sinkhorn

__all__ = [
    "certification",
    "config",
    "data",
    "metrics",
    "reference",
    "admm",
    "pdhg",
    "sinkhorn",
]
