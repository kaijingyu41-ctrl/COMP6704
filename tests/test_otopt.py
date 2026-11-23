"""Unit tests for the transport LP solvers.

The suite checks the three things that can silently go wrong in an
implementation of this kind:

1. the closed-form affine projection is really the Euclidean projection;
2. the primal-dual operator and its adjoint are consistent;
3. the solvers reproduce analytically known optima and the HiGHS reference.
"""

import numpy as np
import pytest

from otopt import admm, data, metrics, pdhg, reference, sinkhorn

BUDGET = dict(max_iter=8000, time_limit=20.0, log_every=20)
TOLS = (1e-3, 1e-4)


def random_instance(m=12, n=10, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.full(m, 0.5))
    b = rng.dirichlet(np.full(n, 0.5))
    C = data.rescaled_cost(rng.random((m, n)))
    return a, b, C


# ------------------------------------------------------------------ projection
def test_projection_restores_marginals():
    a, b, C = random_instance()
    W = np.random.default_rng(1).standard_normal(C.shape) * 5.0
    P = admm.project_affine(W, a, b)
    assert np.allclose(P.sum(axis=1), a, atol=1e-12)
    assert np.allclose(P.sum(axis=0), b, atol=1e-12)


def test_projection_is_orthogonal():
    """The residual must be orthogonal to every feasible direction."""
    a, b, C = random_instance()
    W = np.random.default_rng(2).standard_normal(C.shape)
    P = admm.project_affine(W, a, b)
    R = P - W
    rng = np.random.default_rng(3)
    m, n = C.shape
    for _ in range(5):
        # a feasible direction must satisfy D 1 = 0 and D^T 1 = 0
        D = admm.project_affine(rng.standard_normal(C.shape), np.zeros(m), np.zeros(n))
        assert abs(np.sum(R * D)) < 1e-9


def test_projection_of_feasible_point_is_identity():
    a, b, C = random_instance(6, 6)
    P0 = np.outer(a, b)
    assert np.allclose(admm.project_affine(P0, a, b), P0, atol=1e-12)


# -------------------------------------------------------------------- operator
def test_operator_adjoint_consistency():
    m, n = 7, 5
    rng = np.random.default_rng(4)
    X = rng.standard_normal((m, n))
    a = np.zeros(m)
    b = np.zeros(n)
    y = rng.standard_normal(m + n)
    # <A X, y> == <X, A^T y>   (with zero marginals, A is purely linear)
    lhs = float(np.sum(pdhg._operator(X, a, b) * y))
    rhs = float(np.sum(X * pdhg._adjoint(y, m, n)))
    assert abs(lhs - rhs) < 1e-10


def test_operator_norm_bound():
    """||A||_2^2 == m + n, which justifies the Chambolle-Pock step size."""
    m, n = 4, 3
    A = np.zeros((m + n, m * n))
    for i in range(m):
        for j in range(n):
            A[i, i * n + j] = 1.0
            A[m + j, i * n + j] = 1.0
    assert abs(np.linalg.norm(A, 2) ** 2 - (m + n)) < 1e-8


# ---------------------------------------------------------------- known optima
def test_two_by_two_known_optimum():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    ref = reference.solve_reference(a, b, C)
    assert abs(ref["objective"]) < 1e-9
    assert np.allclose(ref["plan"], np.diag([0.5, 0.5]), atol=1e-6)


def test_constant_cost_objective():
    a, b, C = random_instance(9, 7, seed=5)
    C = np.full_like(C, 0.37)
    ref = reference.solve_reference(a, b, C)
    assert abs(ref["objective"] - 0.37) < 1e-8


# ------------------------------------------------------------------ consistency
@pytest.mark.parametrize("seed", [0, 1])
def test_admm_matches_reference(seed):
    a, b, C = random_instance(12, 10, seed=seed)
    ref = reference.solve_reference(a, b, C)
    out = admm.solve_fixed_admm(a, b, C, rho=10.0, f_ref=ref["objective"], tolerances=TOLS, **BUDGET)
    assert np.all(out["plan"] >= 0)
    assert out["final_feasibility_error"] < 1e-6
    assert out["final_objective_error"] < 1e-6


@pytest.mark.parametrize("seed", [0, 1])
def test_pdhg_matches_reference(seed):
    a, b, C = random_instance(12, 10, seed=seed)
    ref = reference.solve_reference(a, b, C)
    out = pdhg.solve_pdhg(a, b, C, eta=0.99, f_ref=ref["objective"], tolerances=TOLS, **BUDGET)
    assert np.all(out["plan"] >= 0)
    assert out["final_negative_mass"] == 0.0
    assert out["final_objective_error"] < 1e-3
    # the project judges a run on *both* criteria, so feasibility must be
    # asserted here as well and not only inside the benchmark scripts
    assert out["final_feasibility_error"] < 1e-3


def test_pc_preconditioner_values():
    """The Pock-Chambolle diagonal preconditioner has the expected structure."""
    tau, sigma = pdhg.pc_preconditioner(6, 4, eta=0.99)
    assert np.isclose(tau, 0.99 * 0.5)
    assert sigma.shape == (10,)
    assert np.allclose(sigma[:6], 0.99 / 4)     # row constraints contain n ones
    assert np.allclose(sigma[6:], 0.99 / 6)     # column constraints contain m ones
    # 1/tau must equal (sum_i |A_ij|)/eta = 2/eta, i.e. the exact PC rule scaled
    # by eta so that the norm condition holds strictly
    assert np.isclose(1.0 / tau, 2.0 / 0.99)


def test_preconditioned_pdhg_matches_reference():
    """Diagonal preconditioning must not break convergence or optimality."""
    a, b, C = random_instance(12, 10, seed=2)
    ref = reference.solve_reference(a, b, C)
    out = pdhg.solve_pdhg(a, b, C, eta=0.99, precondition=True,
                          f_ref=ref["objective"], tolerances=TOLS, **BUDGET)
    assert np.all(out["plan"] >= 0)
    assert out["final_negative_mass"] == 0.0
    assert out["final_objective_error"] < 1e-3
    assert out["final_feasibility_error"] < 1e-3


# ------------------------------------------------------- KKT / optimality certificate
def test_dual_repair_makes_the_pair_feasible():
    """The repair must always return a dual feasible pair, and never increase beta."""
    a, b, C = random_instance(11, 9, seed=5)
    rng = np.random.default_rng(5)
    alpha = rng.standard_normal(a.size)
    beta = rng.standard_normal(b.size) + 5.0          # deliberately infeasible
    pair = metrics.dual_feasible_pair(alpha, beta, C)
    assert pair["max_violation_before_repair"] > 0.0
    repaired = pair["alpha"][:, None] + pair["beta"][None, :] - C
    assert repaired.max() <= 1e-12
    assert np.all(pair["beta"] <= beta + 1e-12)


def test_certificate_bounds_the_signed_suboptimality():
    """``<C,T> - f*`` must never exceed the certified bound.

    The bound dominates the *signed* gap, not its absolute value: an infeasible
    plan can look cheaper than the optimum (that is the trap discussed in the
    report), and then ``<C,T> - f*`` is negative.
    """
    a, b, C = random_instance(10, 8, seed=3)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    for out in (
        admm.solve_fixed_admm(a, b, C, rho=10.0, f_ref=f_ref, tolerances=TOLS, **BUDGET),
        pdhg.solve_pdhg(a, b, C, eta=0.99, f_ref=f_ref, tolerances=TOLS, **BUDGET),
    ):
        pair = metrics.dual_feasible_pair(out["alpha"], out["beta"], C)
        cert = metrics.certified_suboptimality(out["plan"], a, b, C, pair["alpha"], pair["beta"])
        gap = metrics.objective_value(out["plan"], C) - f_ref
        assert cert["bound"] >= gap - 1e-9
        # the bound is a sum of a non-negative complementarity term and two
        # correction terms for the marginal violation
        assert cert["complementarity_term"] >= -1e-12
        assert abs(cert["bound"]
                   - (cert["complementarity_term"] + cert["row_term"] + cert["col_term"])) < 1e-12


def test_certificate_is_tight_on_a_known_instance():
    """On a 2x2 problem with a known optimum the bound certifies to machine precision."""
    a = b = np.array([0.5, 0.5])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    for out in (
        admm.solve_fixed_admm(a, b, C, rho=1.0, f_ref=0.0, tolerances=TOLS, **BUDGET),
        pdhg.solve_pdhg(a, b, C, eta=0.99, f_ref=0.0, tolerances=TOLS, **BUDGET),
    ):
        pair = metrics.dual_feasible_pair(out["alpha"], out["beta"], C)
        cert = metrics.certified_suboptimality(out["plan"], a, b, C, pair["alpha"], pair["beta"])
        # f* = 0 here, so the bound is also an upper bound on <C,T> itself
        assert cert["bound"] < 1e-9
        assert metrics.objective_value(out["plan"], C) <= cert["bound"] + 1e-12


def test_the_bound_is_an_upper_bound_and_not_a_lower_bound():
    """A zero bound says nothing unless the plan is also primal feasible.

    ``T = 0`` with ``alpha = beta = 0`` is dual feasible, so the bound is exactly
    zero, yet ``T`` is not a transport plan. The bound therefore cannot be read as
    "the error is at least ..." nor as a certificate of optimality on its own ---
    these are the two readings the report explicitly rules out.
    """
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    T = np.zeros((2, 2))
    cert = metrics.certified_suboptimality(T, a, b, C, np.zeros(2), np.zeros(2))
    assert abs(cert["bound"]) < 1e-15
    assert metrics.feasibility_error(T, a, b) > 1.0          # not feasible at all
    # the same zero bound would be produced by an optimal plan, so the bound
    # alone cannot distinguish the two; only the pair (bound, feasibility) can
    assert metrics.objective_value(T, C) <= cert["bound"] + 1e-15  # 0 <= 0


def test_certificate_does_not_understate_a_large_error():
    """The bound must never be smaller than the signed gap it dominates.

    A tight bound on an inaccurate plan, not a small bound, is what the
    certificate is for; this test pins the direction of the inequality for a plan
    that is deliberately far from optimal.
    """
    a, b, C = random_instance(8, 6, seed=11)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    pair = metrics.dual_feasible_pair(np.zeros(a.size), np.zeros(b.size), C)
    bad = np.outer(a, b) * 0.0 + 1e-3          # non-negative, badly infeasible
    cert = metrics.certified_suboptimality(bad, a, b, C, pair["alpha"], pair["beta"])
    gap = metrics.objective_value(bad, C) - f_ref
    assert cert["bound"] >= gap - 1e-9


def test_certificate_against_a_certified_lower_bound():
    """The bound must dominate ``<C,T> - L`` for a certified lower bound ``L <= f*``.

    Comparing with the primal reference value instead would test the reference's
    own accuracy as much as the certificate, so the result files are validated
    against ``L``: the dual LP, repaired to exact dual feasibility, whose objective
    is a rigorous lower bound on ``f*``.
    """
    a, b, C = random_instance(12, 10, seed=9)
    dual = reference.solve_dual_reference(a, b, C)
    repaired = metrics.dual_feasible_pair(dual["alpha"], dual["beta"], C)
    f_lower = float(a @ repaired["alpha"] + b @ repaired["beta"])
    f_ref = reference.solve_reference(a, b, C)["objective"]
    assert f_lower <= f_ref + 1e-9

    for out in (
        admm.solve_fixed_admm(a, b, C, rho=1.0, f_ref=f_ref, tolerances=TOLS, **BUDGET),
        pdhg.solve_pdhg(a, b, C, eta=0.99, f_ref=f_ref, tolerances=TOLS, **BUDGET),
    ):
        pair = metrics.dual_feasible_pair(out["alpha"], out["beta"], C)
        cert = metrics.certified_suboptimality(out["plan"], a, b, C, pair["alpha"], pair["beta"])
        assert cert["bound"] >= metrics.objective_value(out["plan"], C) - f_lower - 1e-12


def test_feasibility_polishing_reduces_the_marginal_error():
    a, b, C = random_instance(10, 10, seed=4)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    out = admm.solve_fixed_admm(a, b, C, rho=1.0, f_ref=f_ref, tolerances=TOLS, **BUDGET)
    polished = metrics.polish_feasibility(out["plan"], a, b)
    assert np.all(polished >= 0.0)
    assert (metrics.feasibility_error(polished, a, b)
            <= metrics.feasibility_error(out["plan"], a, b) + 1e-12)
    # the rescaling is expected to *reach* feasibility on a dense support, so this
    # is asserted rather than assumed: a fixed iteration count would not guarantee it
    assert metrics.feasibility_error(polished, a, b) < 1e-12


def test_polishing_cannot_repair_a_support_that_cannot_carry_the_marginals():
    """Polishing preserves the support, so it cannot fix a structural infeasibility.

    This is why the caller has to check the residual it leaves instead of assuming
    that polishing produced a feasible plan.
    """
    a = np.array([0.5, 0.5])
    b = np.array([0.25, 0.25, 0.5])
    T = np.zeros((2, 3))
    T[0, 0] = 1.0                    # row 1 and column 2 can never be filled
    polished = metrics.polish_feasibility(T, a, b, max_iter=200)
    assert metrics.feasibility_error(polished, a, b) > 0.1


def test_admm_potentials_reproduce_the_stationarity_residual():
    """The potentials must be read off the stationarity residual that defines them.

    The residual has the additively separable form ``alpha 1' + 1 beta'`` (rank at
    most two, not one), and the two extracted vectors must reproduce that matrix
    exactly. Forming the residual with the *updated* Z and U blocks would destroy
    the separability, so the two-vector representation would no longer represent
    the matrix it is supposed to represent.
    """
    a, b, C = random_instance(9, 7, seed=13)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    out = admm.solve_fixed_admm(a, b, C, rho=10.0, f_ref=f_ref, tolerances=TOLS, **BUDGET)
    assert out["stationarity_residual"] < 1e-9


def test_sinkhorn_potentials_carry_the_lp_dual_sign():
    """The Sinkhorn scaling vectors, not their negation, are the LP dual variables.

    With the plan written as ``exp((f_i + g_j - C_ij) / eps)`` and the LP dual as
    ``max a'alpha + b'beta``, the dual pair is ``(f, g)``: it is then already dual
    feasible (the slack ``C_ij - f_i - g_j = eps * log X_ij`` has the right sign),
    whereas its negation violates the constraint by an ``O(1)`` amount.
    """
    a, b, C = random_instance(9, 7, seed=8)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    out = sinkhorn.solve_sinkhorn(a, b, C, eps=0.05, f_ref=f_ref, tolerances=TOLS, **BUDGET)
    alpha, beta = out["alpha"], out["beta"]
    # slack of the dual constraint alpha_i + beta_j <= C_ij : must be non-negative
    slack = float(np.min(C - alpha[:, None] - beta[None, :]))
    # ... and the negated pair must violate it by a large amount
    violation_negated = float(np.max(-alpha[:, None] - beta[None, :] - C))
    assert slack > -1e-12
    assert violation_negated > 0.05


def test_kkt_certificate_against_independent_dual():
    """Check the KKT conditions of our plans with a dual solved independently.

    The dual LP ``max a^T alpha + b^T beta s.t. alpha_i + beta_j <= C_ij`` is
    solved directly with HiGHS, and the resulting potentials are used to verify
    dual feasibility, strong duality and complementary slackness for plans
    produced by ADMM and PDHG. This is the certificate the report refers to.
    """
    a, b, C = random_instance(10, 8, seed=3)
    dual = reference.solve_dual_reference(a, b, C)
    alpha, beta = dual["alpha"], dual["beta"]
    assert dual["dual_max_violation"] < 1e-9          # the dual point is feasible

    primal = reference.solve_reference(a, b, C)
    assert abs(dual["objective"] - primal["objective"]) < 1e-8   # strong duality

    for out in (
        admm.solve_fixed_admm(a, b, C, rho=10.0, f_ref=primal["objective"],
                              tolerances=TOLS, **BUDGET),
        pdhg.solve_pdhg(a, b, C, eta=0.99, f_ref=primal["objective"],
                        tolerances=TOLS, **BUDGET),
    ):
        T = out["plan"]
        resid = metrics.kkt_residuals(T, a, b, C, alpha, beta)
        # the test budget (8e3 iterations) is six times smaller than the
        # benchmark budget, so 1e-5 is the appropriate certificate tolerance here
        assert resid["primal_ineq"] < 1e-5
        assert resid["dual_violation"] < 1e-9
        # complementary slackness: sum_ij T_ij (C_ij - alpha_i - beta_j) ~ 0
        assert resid["complementarity"] < 1e-5


def test_adaptive_admm_runs_and_is_feasible():
    a, b, C = random_instance(10, 10, seed=7)
    ref = reference.solve_reference(a, b, C)
    out = admm.solve_residual_balancing_admm(
        a, b, C, rho=1.0, f_ref=ref["objective"], tolerances=TOLS, **BUDGET
    )
    # rho = 1 is deliberately *not* the tuned value, so only the benchmark
    # tolerance (1e-4) is required here.
    assert out["final_feasibility_error"] < 1e-4
    assert out["n_rho_changes"] >= 0


# --------------------------------------------------- regularised baseline
def test_sinkhorn_is_feasible_but_biased():
    """Sinkhorn gives an exactly feasible plan whose objective carries the bias.

    This is the property the report's baseline comparison rests on: the
    regularised solver is far more accurate on feasibility than on optimality.
    """
    a, b, C = random_instance(10, 8, seed=6)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    small = sinkhorn.solve_sinkhorn(a, b, C, eps=0.02, f_ref=f_ref,
                                    tolerances=TOLS, **BUDGET)
    large = sinkhorn.solve_sinkhorn(a, b, C, eps=0.2, f_ref=f_ref,
                                    tolerances=TOLS, **BUDGET)
    for out in (small, large):
        assert np.all(out["plan"] >= 0.0)
        assert out["final_feasibility_error"] < 1e-8      # the marginals are exact
        # the bias dominates: the objective error is orders of magnitude above the
        # feasibility error at the same iterate
        assert out["final_objective_error"] > 1e3 * out["final_feasibility_error"]
    # and the bias grows with eps
    assert large["final_objective_error"] > 5.0 * small["final_objective_error"]


def test_sinkhorn_dual_potentials_are_repaired_into_a_valid_certificate():
    a, b, C = random_instance(9, 7, seed=8)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    out = sinkhorn.solve_sinkhorn(a, b, C, eps=0.05, f_ref=f_ref,
                                  tolerances=TOLS, **BUDGET)
    pair = metrics.dual_feasible_pair(out["alpha"], out["beta"], C)
    cert = metrics.certified_suboptimality(out["plan"], a, b, C, pair["alpha"], pair["beta"])
    gap = metrics.objective_value(out["plan"], C) - f_ref
    assert cert["bound"] >= gap - 1e-9


# ------------------------------------------------------------------- validation
def test_instance_validation_rejects_unbalanced():
    a = np.array([0.6, 0.6])
    b = np.array([0.5, 0.5])
    inst = data.Instance(name="bad", a=a, b=b, C=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        data.check_instance(inst)


def test_solvers_run_without_any_reference_value():
    """The certificate needs no reference solve, so the pipeline must not require one.

    This is the difference between a method that *reports* an optimality bound and
    one that needs a solved LP to say anything: with ``f_ref=None`` the solvers and
    the certificate still work, and only the reference comparison is missing.
    """
    from otopt import certification

    a, b, C = random_instance(10, 8, seed=7)
    inst = data.Instance(name="no-ref", a=a, b=b, C=C)
    out = admm.solve_fixed_admm(a, b, C, rho=10.0, tolerances=TOLS, **BUDGET)
    assert out["final_objective_error"] is None
    assert out["reference_objective"] is None
    certification.certify(out, inst)
    assert out["certified_error_bound"] is not None
    assert out["certificate_is_valid"] is None       # nothing independent to check


def test_certified_flag_requires_feasibility_and_a_small_bound():
    """`certified` is the conjunction the report uses, not the bound alone."""
    from otopt import certification

    a, b, C = random_instance(10, 8, seed=6)
    inst = data.Instance(name="flag", a=a, b=b, C=C)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    good = admm.solve_fixed_admm(a, b, C, rho=10.0, f_ref=f_ref, tolerances=TOLS, **BUDGET)
    certification.certify(good, inst, f_lower=f_ref - 1e-9)
    assert good["certified_at_1e-04"] is True

    tiny_budget = dict(max_iter=25, time_limit=5.0, log_every=5)
    early = admm.solve_fixed_admm(a, b, C, rho=10.0, f_ref=f_ref, tolerances=TOLS,
                                  **tiny_budget)
    certification.certify(early, inst, f_lower=f_ref - 1e-9)
    assert early["final_feasibility_error"] > 1e-4
    assert early["certified_at_1e-04"] is False      # infeasible, so not certified


def test_certified_bracket_contains_a_known_optimum():
    """The bracket must hold the analytic optimum, and its width must be tiny."""
    from otopt import certification

    a = b = np.array([0.5, 0.5])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    inst = data.Instance(name="2x2", a=a, b=b, C=C)
    plan = np.diag([0.5, 0.5])
    bracket = certification.certified_bracket(inst, plan)
    assert certification.bracket_matches_known_optimum(bracket, 0.0)
    assert bracket["width"] < 1e-12
    assert bracket["polish_reached_precision"]
    assert bracket["bracket_is_valid"]


def test_bracket_degenerates_when_polishing_stalls():
    """The interval is only informative when the polish succeeded.

    A plan whose support cannot carry the marginals cannot be rescaled into one,
    so its objective is not an upper bound and the interval can be empty --- here
    the width is negative. ``bracket_is_valid`` records that, and the only
    surviving statement is the one-sided ``lower <= f*``: a primal value below it
    is provably in error.
    """
    from otopt import certification

    a, b, C = random_instance(10, 8, seed=17)
    inst = data.Instance(name="empty-support", a=a, b=b, C=C)
    empty = np.zeros_like(C)                      # support cannot carry any mass
    bracket = certification.certified_bracket(inst, empty)
    assert not bracket["polish_reached_precision"]
    assert not bracket["bracket_is_valid"]
    assert bracket["width"] < 0.0
    assert bracket["lower"] > 0.0                 # the dual bound still holds


def test_zero_bound_needs_both_primal_and_dual_optimality():
    """A primal-feasible but dual-poor pair can leave the bound strictly positive.

    On a constant-cost instance every feasible plan is optimal, so ``T`` alone
    being optimal does *not* make the bound vanish: with ``alpha = beta = 0`` the
    bound equals ``<C,T> - 0 = f* > 0``. The report states this explicitly, and
    this test pins the direction of the implication.
    """
    a = b = np.array([0.5, 0.5])
    C = np.ones((2, 2))
    T = np.diag([0.5, 0.5])                       # optimal: <C,T> = f* = 1
    assert metrics.feasibility_error(T, a, b) < 1e-15
    cert = metrics.certified_suboptimality(T, a, b, C, np.zeros(2), np.zeros(2))
    assert abs(cert["bound"] - 1.0) < 1e-12       # not zero, although T is optimal
    # ... and the bound is tight again once the dual side is optimal too
    dual = reference.solve_dual_reference(a, b, C)
    pair = metrics.dual_feasible_pair(dual["alpha"], dual["beta"], C)
    cert_ok = metrics.certified_suboptimality(T, a, b, C, pair["alpha"], pair["beta"])
    assert cert_ok["bound"] < 1e-9


def test_restart_rule_compares_with_the_previous_check():
    """The rule is the *previous check*, and the reference moves even with no restart.

    The second half of the test is the one that matters: a rule that only updated
    its reference when a restart fired would behave differently on the next check,
    and testing the predicate alone cannot see that.
    """
    from otopt import pdhg

    fire, ref = pdhg.restart_decision(r=0.9, r_prev=1.0, beta=0.1)      # 0.9 >= 0.9
    assert fire and ref == 0.9
    fire, ref = pdhg.restart_decision(r=0.8, r_prev=1.0, beta=0.1)      # a 20% drop
    assert not fire
    assert ref == 0.8            # updated although no restart fired
    fire, _ = pdhg.restart_decision(r=2.0, r_prev=1.0, beta=0.1)        # residual grew
    assert fire


def test_bracket_gives_a_lower_bound_on_the_reference_error_when_it_cannot_close():
    """``|f_ref - f*| >= max(0, L - f_ref)`` --- the positive part matters.

    A reference value below the rigorous lower bound ``L <= f*`` is provably in
    error, and that is a *one-sided* statement: no upper bound on the reference
    error is available, so the field that would carry one must be ``None``. The
    second half is the case the inequality is silent on, and the one an
    unconditional ``|L - f_ref|`` gets wrong: a reference value *above* ``L`` can
    legitimately coincide with ``f*``, so nothing is proved and the reported bound
    must be zero rather than the (positive) distance to ``L``.
    """
    from otopt import certification

    a, b, C = random_instance(10, 8, seed=19)
    inst = data.Instance(name="audit", a=a, b=b, C=C)
    dual = reference.solve_dual_reference(a, b, C)
    pair = metrics.dual_feasible_pair(dual["alpha"], dual["beta"], C)
    lower = float(a @ pair["alpha"] + b @ pair["beta"])

    f_ref_bad = lower - 1e-6                      # deliberately below any feasible f*
    bracket = certification.certified_bracket(inst, np.zeros_like(C), f_ref_bad)
    assert bracket["reference_error_lower_bound"] == abs(lower - f_ref_bad)
    assert bracket["reference_below_lower"]
    assert bracket["reference_error_upper_bound"] is None
    assert not bracket["reference_inside"]

    f_ref_high = lower + 1e-3                     # above L: nothing is proved
    bracket = certification.certified_bracket(inst, np.zeros_like(C), f_ref_high)
    assert bracket["reference_error_lower_bound"] == 0.0   # not |L - f_ref| = 1e-3
    assert not bracket["reference_below_lower"]


def test_sinkhorn_continuation_never_exceeds_the_iteration_budget():
    """The schedule must not overshoot when the budget is smaller than it.

    ``max_iter // len(schedule)`` would give every stage at least one iteration, so
    a three-iteration budget spent over four stages would run four times. The
    allocation carries its remainder forward instead, which has to hold for any
    input, including budgets far below the standard one.
    """
    a, b, C = random_instance(6, 5, seed=23)
    schedule = (0.2, 0.1, 0.05, 0.01)
    for budget in (0, 1, 2, 3, 7, 40):
        out = sinkhorn.solve_sinkhorn_continuation(
            a, b, C, eps_schedule=schedule, f_ref=None, tolerances=TOLS,
            max_iter=budget, time_limit=20.0, log_every=1,
        )
        assert out["iterations"] <= budget, (budget, out["iterations"])
        assert out["stages"] == len(schedule)


def test_bracket_bounds_the_reference_error_when_the_interval_is_valid():
    """Inside a valid interval the width *is* an upper bound on the reference error."""
    from otopt import certification

    a, b, C = random_instance(10, 8, seed=21)
    inst = data.Instance(name="audit-ok", a=a, b=b, C=C)
    ref = reference.solve_reference(a, b, C)
    bracket = certification.certified_bracket(inst, ref["plan"], ref["objective"])
    assert bracket["bracket_is_valid"]
    assert bracket["reference_inside"]
    assert bracket["reference_error_upper_bound"] == bracket["width"]
    assert bracket["width"] >= -certification.ENDPOINT_TOL


def test_trace_point_at_drops_an_unreached_checkpoint():
    """A checkpoint the run never reached must not be relabelled from an earlier point."""
    history = [{"iteration": k, "feasibility_error": 1.0 / (k + 1)} for k in (0, 200, 400, 600)]
    assert metrics.trace_point_at(history, 400)["feasibility_error"] == 1.0 / 401
    assert metrics.trace_point_at(history, 450) is None       # logged every 200 only
    assert metrics.trace_point_at(history, 800) is None       # run stopped at 600


def test_sinkhorn_continuation_shares_one_budget_across_stages():
    """The decreasing-eps schedule spends ``max_iter`` in total, not per stage.

    The report compares each schedule against a single-eps run given the *same*
    total budget, so this is the property that makes that comparison honest.
    """
    a, b, C = random_instance(10, 8, seed=6)
    f_ref = reference.solve_reference(a, b, C)["objective"]
    schedule = (0.2, 0.1, 0.05)
    out = sinkhorn.solve_sinkhorn_continuation(
        a, b, C, eps_schedule=schedule, f_ref=f_ref, tolerances=TOLS, **BUDGET)
    assert out["iterations"] <= BUDGET["max_iter"]
    assert out["eps_schedule"] == list(schedule)
    assert out["eps"] == schedule[-1]
    assert out["stages"] == len(schedule)
    # warm-started continuation still holds the marginals exactly
    assert np.all(out["plan"] >= 0.0)
    assert out["final_feasibility_error"] < 1e-8
    with pytest.raises(ValueError):
        sinkhorn.solve_sinkhorn_continuation(
            a, b, C, eps_schedule=(), f_ref=f_ref, tolerances=TOLS, **BUDGET)


def test_sinkhorn_continuation_beats_a_cold_start_at_the_same_budget():
    """Warm start is what makes a small ``eps`` affordable: same budget, smaller bias.

    At the final ``eps`` a cold run given the whole budget is still far from its own
    convergence, while the continuation arrives there with the scaling vectors
    already close, so it is closer to the unregularised optimum on the same budget.
    """
    rng = np.random.default_rng(17)
    a = rng.dirichlet(np.full(14, 0.3))
    b = rng.dirichlet(np.full(12, 0.3))
    C = data.rescaled_cost(rng.random((14, 12)))
    f_ref = reference.solve_reference(a, b, C)["objective"]
    budget = dict(max_iter=150, time_limit=20.0, log_every=10)
    cont = sinkhorn.solve_sinkhorn_continuation(
        a, b, C, eps_schedule=(0.2, 0.05, 0.01, 0.002), f_ref=f_ref,
        tolerances=TOLS, **budget)
    cold = sinkhorn.solve_sinkhorn(a, b, C, eps=0.002, f_ref=f_ref,
                                   tolerances=TOLS, **budget)
    assert cont["iterations"] <= budget["max_iter"]
    assert cold["iterations"] <= budget["max_iter"]
    assert np.all(cont["plan"] >= 0.0)
    # warm start buys accuracy, with a wide margin, at the same total budget
    assert 1.5 * cont["final_objective_error"] < cold["final_objective_error"]


def test_digits_instances_are_valid():
    insts = data.make_digits_instances(n_same=2, n_cross=2, seed=0, eps=1e-6)
    assert len(insts) == 4
    for inst in insts:
        data.check_instance(inst)
        assert abs(inst.a.sum() - 1.0) < 1e-12
