"""Optimality certificates that need no reference solution.

Two things are provided:

* :func:`certify` attaches to one solver result an *upper* bound on the signed
  suboptimality ``<C,T> - f*`` of the returned plan, computed from the plan alone
  (Proposition 1 of the report), together with the flags needed to read it
  correctly.
* :func:`certified_bracket` returns a two-sided bracket on ``f*`` for one instance,
  built from an independently solved dual LP (a lower bound) and a
  feasibility-polished primal candidate (an upper bound *when the polish succeeded*,
  which is checked and reported rather than assumed). It is a numerical audit, not a
  certified interval, and what it says about the reference value is one-sided unless
  the interval closes.

The distinction between an upper bound and a lower bound is the whole point of
this module, so it is stated once here. The certificate is an *upper* bound: a
small value is good news, and a large value is not evidence that the error is
large. It is also not, on its own, a certificate of optimality --- a plan whose
marginals are wrong can have a zero or negative bound --- which is why
:func:`certify` records the plan's feasibility error alongside and exposes a
``certified_at_*`` flag that requires both.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import config, metrics, reference


#: an endpoint disagreement at this level is double-precision rounding, not a
#: failure of the polish; a signed difference beyond it is a genuine degeneracy
ENDPOINT_TOL = 1e-12


def certified_bracket(inst, plan, f_ref: float | None = None) -> Dict[str, Any]:
    """A bracket ``[lower, upper]`` on the optimal value ``f*``, and what it proves.

    ``lower`` is the objective of a dual feasible pair: the dual LP is solved
    independently and repaired to exact dual feasibility, so weak duality makes it
    a lower bound on ``f*`` --- one that holds up to floating-point rounding, since
    the dual constraint is verified by evaluating a residual rather than with
    directed rounding.

    ``upper`` is the objective of ``plan`` after feasibility polishing. Polishing
    keeps the plan non-negative and moves its marginals onto ``a`` and ``b`` by
    row/column scaling, so a *successfully* polished plan is a genuine transport
    plan, and the objective of any transport plan is an upper bound on ``f*``.
    Whether polishing succeeded is checked rather than assumed: the residual it
    leaves is recorded in ``polish_feasibility_error`` and
    ``polish_reached_precision`` is true only when that residual is at the
    resolution of :func:`metrics.feasibility_error` on double precision. That flag
    is a numerical feasibility check, not a proof: a mathematically rigorous
    interval would need directed rounding or a residual-based correction.

    ``bracket_is_valid`` is the conjunction the statistical analysis filters on:
    the polish reached precision *and* the two independently computed endpoints
    agree to :data:`ENDPOINT_TOL`. The small negative widths that appear when the
    polish has converged (a few units in the last place) are rounding and are
    accepted; a width that is still negative at that tolerance means the interval
    genuinely failed to close.

    What the bracket says about the reference value is *one-sided*, and the side is
    easy to get wrong. Because ``lower <= f*`` always,

        |f_ref - f*|  >=  max(0, lower - f_ref),

    for *any* reference value. The positive part is what carries the information,
    and it is nonzero exactly when ``f_ref`` sits *below* ``lower``: there the
    reference is in error by at least that gap and its plan cannot be exactly
    feasible. A reference value *above* ``lower`` yields nothing from this
    inequality --- ``|lower - f_ref|`` would be a positive number that says nothing
    about the error, since the two values can legitimately coincide --- so
    ``reference_error_lower_bound`` reports the positive part and
    ``reference_below_lower`` records whether the inequality was informative.

    When the interval *is* valid and ``f_ref`` lies inside it
    (``reference_inside``), the two-sided statement
    ``|f_ref - f*| <= upper - lower`` holds instead, and it is returned as
    ``reference_error_upper_bound`` (clipped at zero, since a width that is
    negative by a few units in the last place is rounding). Otherwise no upper bound
    on the reference error is available and that field is ``None``.
    """
    dual = reference.solve_dual_reference(inst.a, inst.b, inst.C)
    pair = metrics.dual_feasible_pair(dual["alpha"], dual["beta"], inst.C)
    lower = float(inst.a @ pair["alpha"] + inst.b @ pair["beta"])
    polished = metrics.polish_feasibility(plan, inst.a, inst.b)
    polish_error = metrics.feasibility_error(polished, inst.a, inst.b)
    upper = metrics.objective_value(polished, inst.C)
    width = float(upper - lower)
    reached = bool(polish_error <= ENDPOINT_TOL)
    valid = bool(reached and width >= -ENDPOINT_TOL)
    out: Dict[str, Any] = {
        "lower": lower,
        "upper": upper,
        "width": width,
        "polish_feasibility_error": polish_error,
        "polish_reached_precision": reached,
        "bracket_is_valid": valid,
    }
    if f_ref is not None:
        f_ref = float(f_ref)
        inside = bool(valid and lower - ENDPOINT_TOL <= f_ref <= upper + ENDPOINT_TOL)
        below = float(lower - f_ref)
        out["reference_error_lower_bound"] = max(0.0, below)
        out["reference_below_lower"] = bool(below > ENDPOINT_TOL)
        out["reference_inside"] = inside
        out["reference_error_upper_bound"] = max(width, 0.0) if inside else None
    return out


def certify(out: Dict[str, Any], inst, f_lower: float | None = None) -> Dict[str, Any]:
    """Attach a dual-gap certificate that needs no reference optimum.

    Each solver exposes the dual potentials it has built up (see the notes in
    ``admm.py``, ``pdhg.py`` and ``sinkhorn.py``). They are repaired into a dual
    feasible pair and used to bound the suboptimality of the returned plan; the
    bound is computed from the plan alone.

    What the bound is and is not. It is an *upper* bound on the signed gap
    ``<C,T> - f*``, so it is read together with a feasibility error: a large bound
    is not evidence of a large error, and a small bound does not by itself certify
    anything about a plan whose marginals are wrong. Each run therefore also records
    a ``certified_at_*`` flag, which is true only when the returned plan is
    simultaneously marginal-feasible at that tolerance *and* its own bound is below
    that tolerance --- that joint statement, not the bound alone, is what the report
    calls a certified run.

    Two bounds are stored. The first is applied to the plan as returned, where it is
    limited by the marginal error, because the dual prices multiply that error. The
    second is applied to a polished plan, which removes the marginal terms and
    leaves the complementarity term; it is a diagnostic for how much of the gap is
    complementarity, and the accuracy the benchmark reports is always that of the
    returned plan.

    The certificate is a statement about ``<C,T> - f*``, so validating it against
    the double-precision value ``f_ref`` tests something slightly different: the
    reference LP solve has its own error, which is the same order as the quantities
    being compared. When a certified lower bound ``f_lower <= f*`` is supplied, the
    validity flag uses it, which makes the check conservative rather than
    noise-limited.
    """
    pair = metrics.dual_feasible_pair(out["alpha"], out["beta"], inst.C)
    cert = metrics.certified_suboptimality(
        out["plan"], inst.a, inst.b, inst.C, pair["alpha"], pair["beta"]
    )
    polished = metrics.polish_feasibility(out["plan"], inst.a, inst.b)
    cert_polished = metrics.certified_suboptimality(
        polished, inst.a, inst.b, inst.C, pair["alpha"], pair["beta"]
    )
    objective = metrics.objective_value(out["plan"], inst.C)
    reference_value = out["reference_objective"]
    # the signed gap the certificate dominates, and the check that it does
    out["signed_objective_gap"] = (
        None if reference_value is None else objective - float(reference_value)
    )
    if f_lower is not None:
        out["reference_lower_bound"] = float(f_lower)
        out["signed_objective_gap_vs_lower_bound"] = objective - float(f_lower)
        out["certificate_is_valid"] = bool(
            cert["bound"] >= objective - float(f_lower) - 1e-12
        )
    elif reference_value is not None:
        out["certificate_is_valid"] = bool(
            cert["bound"] >= out["signed_objective_gap"] - 1e-12
        )
    else:
        # nothing independent to check the bound against; validity is undefined
        out["certificate_is_valid"] = None
    out["certified_error_bound"] = cert["bound"]
    out["certified_complementarity_term"] = cert["complementarity_term"]
    out["certified_row_term"] = cert["row_term"]
    out["certified_col_term"] = cert["col_term"]
    out["certified_error_bound_polished"] = cert_polished["bound"]
    out["polished_feasibility_error"] = metrics.feasibility_error(polished, inst.a, inst.b)
    out["dual_violation_before_repair"] = pair["max_violation_before_repair"]
    # A run is certified at a tolerance only if the plan we actually report is
    # feasible to it *and* the plan's own bound is below it. The bound on the
    # polished diagnostic plan is deliberately not used here.
    for tol in config.TOLERANCES:
        out[f"certified_at_{tol:.0e}"] = bool(
            out["final_feasibility_error"] <= tol and cert["bound"] <= tol
        )
    return out


def bracket_matches_known_optimum(bracket: Dict[str, Any], optimum: float, tol: float = 1e-9) -> bool:
    """Whether a bracket contains an analytically known optimum."""
    return bool(bracket["lower"] - tol <= optimum <= bracket["upper"] + tol)
