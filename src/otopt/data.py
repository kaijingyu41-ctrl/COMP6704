"""Instance generation for the discrete optimal transport experiments.

An *instance* is a triple ``(a, b, C)`` where

* ``a`` is the supply marginal (``sum(a) == 1``),
* ``b`` is the demand marginal (``sum(b) == 1``),
* ``C`` is the ground cost matrix, rescaled to ``[0, 1]``.

Two families are provided:

* :func:`make_synthetic_instance` -- random points in the unit square with a
  squared-Euclidean cost, used for the controlled scaling study;
* :func:`make_digits_instances` -- real handwritten digit images from
  ``sklearn.datasets.load_digits``, whose grey levels are used as mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class Instance:
    """A discrete optimal transport instance."""

    name: str
    a: np.ndarray
    b: np.ndarray
    C: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return self.C.shape[0]

    @property
    def n(self) -> int:
        return self.C.shape[1]


# --------------------------------------------------------------------- helpers
def squared_distance_cost(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Matrix of squared Euclidean distances between two point sets."""
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def rescaled_cost(cost: np.ndarray) -> np.ndarray:
    """Affinely rescale a cost matrix to ``[0, 1]``.

    The cost values are shifted and scaled only; the argmin of the transport
    problem is unchanged because every feasible plan has the same total mass.
    """
    cmin = float(cost.min())
    cmax = float(cost.max())
    if cmax - cmin < 1e-12:
        return np.zeros_like(cost)
    return (cost - cmin) / (cmax - cmin)


def _mass(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return np.full(n, 1.0 / n)
    if kind.startswith("dirichlet"):
        alpha = float(kind.replace("dirichlet", ""))
        return rng.dirichlet(np.full(n, alpha))
    raise ValueError(f"unknown mass kind: {kind!r}")


def check_instance(inst: Instance, tol: float = 1e-9) -> None:
    """Raise if the instance violates the assumptions of the transport LP."""
    if inst.C.shape != (inst.a.size, inst.b.size):
        raise ValueError("cost matrix shape does not match the marginals")
    if np.any(inst.a < 0) or np.any(inst.b < 0):
        raise ValueError("negative mass")
    if abs(inst.a.sum() - 1.0) > tol or abs(inst.b.sum() - 1.0) > tol:
        raise ValueError("marginals must sum to one (balance condition)")


# ------------------------------------------------------------------- synthetic
def make_synthetic_instance(
    size: int,
    mass_kind: str,
    seed: int,
) -> Instance:
    """Random squared-Euclidean transport instance in the unit square."""
    rng = np.random.default_rng(seed)
    points_a = rng.random((size, 2))
    points_b = rng.random((size, 2))
    cost = rescaled_cost(squared_distance_cost(points_a, points_b))
    a = _mass(size, mass_kind, rng)
    b = _mass(size, mass_kind, rng)
    inst = Instance(
        name=f"syn-{mass_kind}-{size}-seed{seed}",
        a=a,
        b=b,
        C=cost,
        meta={"family": "synthetic", "size": size, "mass_kind": mass_kind, "seed": seed},
    )
    check_instance(inst)
    return inst


def make_instance_grid(
    sizes: Tuple[int, ...],
    mass_kinds: Tuple[str, ...],
    seeds: Tuple[int, ...],
) -> List[Instance]:
    """Full factorial grid of synthetic instances used in experiment B."""
    return [
        make_synthetic_instance(size, kind, seed)
        for size in sizes
        for kind in mass_kinds
        for seed in seeds
    ]


# ---------------------------------------------------------------------- digits
def _image_mass(image: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Turn an 8x8 digit image into (support positions, mass vector)."""
    flat = image.astype(np.float64).ravel()
    total = flat.sum()
    if total <= 0:
        raise ValueError("empty digit image")
    mass = flat / total
    keep = mass > eps
    mass = mass[keep]
    mass = mass / mass.sum()          # renormalise after dropping dusty pixels
    rows, cols = np.divmod(np.nonzero(keep)[0], image.shape[1])
    positions = np.column_stack([rows, cols]).astype(np.float64)
    if positions.shape[0] < 2:
        raise ValueError("degenerate digit image after thresholding")
    return positions, mass


def make_digits_instances(
    n_same: int,
    n_cross: int,
    seed: int,
    eps: float,
) -> List[Instance]:
    """Real-data instances built from the sklearn handwritten digits.

    ``n_same`` pairs share the same digit label, ``n_cross`` pairs use different
    labels. The pair list is deterministic given ``seed`` and is stored in
    ``meta`` so that the exact experiment can be reproduced.
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    images = digits.images
    labels = digits.target
    rng = np.random.default_rng(seed)

    pairs: List[Tuple[int, int]] = []
    # same-class pairs
    for label in range(10):
        if len(pairs) >= n_same:
            break
        idx = np.flatnonzero(labels == label)
        i, j = rng.choice(idx, size=2, replace=False)
        pairs.append((int(i), int(j)))
    # cross-class pairs
    seen = set()
    while len(pairs) < n_same + n_cross:
        i, j = rng.integers(0, len(labels), size=2)
        if labels[i] == labels[j] or (int(i), int(j)) in seen:
            continue
        seen.add((int(i), int(j)))
        pairs.append((int(i), int(j)))

    instances: List[Instance] = []
    for k, (i, j) in enumerate(pairs):
        pos_a, mass_a = _image_mass(images[i], eps)
        pos_b, mass_b = _image_mass(images[j], eps)
        cost = rescaled_cost(squared_distance_cost(pos_a, pos_b))
        kind = "same" if labels[i] == labels[j] else "cross"
        inst = Instance(
            name=f"digits-{kind}-{i}-{j}",
            a=mass_a,
            b=mass_b,
            C=cost,
            meta={
                "family": "digits",
                "pair_kind": kind,
                "index_a": int(i),
                "index_b": int(j),
                "label_a": int(labels[i]),
                "label_b": int(labels[j]),
            },
        )
        check_instance(inst)
        instances.append(inst)
    return instances
