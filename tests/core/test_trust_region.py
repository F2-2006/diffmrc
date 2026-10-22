"""Trust-region minimization: adaptive radius, Steihaug-CG, descent-only.

The subproblem is the one trust-region Newton uses everywhere: minimize
the local quadratic model ``g.p + 0.5 p.Hp`` subject to ``||p|| <= radius``.
Steihaug's truncated conjugate-gradient iteration supplies the step, with
two exits that differ from ordinary CG: a direction of non-positive
curvature is followed to the boundary (there the model is unbounded below,
so the boundary is the best point on that ray), and an iterate that would
leave the ball is projected onto its boundary.

The acceptance rule is stricter than "the model improved": a step is taken
only when the energy actually decreases and the ratio of actual to
predicted decrease clears ``eta``. A wrong curvature model can therefore
cost iterations but cannot raise the energy -- a property this project
needs, because an unguarded refinement once walked the energy from -76.07
to +1.04 Ha without raising an error.
"""
import numpy as np
import pytest

from diffmrc.core.trust_region import (
    SteihaugStatus,
    TrustRegionStop,
    steihaug_cg,
    trust_region_minimize,
)


def _quadratic(centre, curvature):
    centre = np.asarray(centre, dtype=float)
    curvature = float(curvature)

    def fun(x):
        d = np.asarray(x, dtype=float) - centre
        return 0.5 * curvature * float(d @ d)

    def grad(x):
        return curvature * (np.asarray(x, dtype=float) - centre)

    def hvp(x, v):
        return curvature * np.asarray(v, dtype=float)

    return fun, grad, hvp


def _rosenbrock():
    def fun(x):
        x = np.asarray(x, dtype=float)
        return float((1.0 - x[0]) ** 2
                     + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def grad(x):
        x = np.asarray(x, dtype=float)
        return np.array([
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ])

    def hvp(x, v):
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        hxx = 2.0 - 400.0 * (x[1] - x[0] ** 2) + 800.0 * x[0] ** 2
        hxy = -400.0 * x[0]
        hyy = 200.0
        return np.array([hxx * v[0] + hxy * v[1],
                         hxy * v[0] + hyy * v[1]])

    return fun, grad, hvp


def _double_well():
    """Quartic double well: minima at x = +-1, saddle at x = 0."""
    def fun(x):
        x = np.asarray(x, dtype=float)
        return float(0.25 * (x[0] ** 2 - 1.0) ** 2 + 0.5 * x[1] ** 2)

    def grad(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] * (x[0] ** 2 - 1.0), x[1]])

    def hvp(x, v):
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.array([(3.0 * x[0] ** 2 - 1.0) * v[0], v[1]])

    return fun, grad, hvp


def test_quadratic_reaches_the_known_minimum():
    fun, grad, hvp = _quadratic([1.0, -2.0, 0.5], curvature=2.0)
    res = trust_region_minimize(fun, grad, np.zeros(3), hvp=hvp,
                                gtol=1e-10)
    assert res.stop_reason is TrustRegionStop.CONVERGED
    assert np.allclose(res.x, [1.0, -2.0, 0.5], atol=1e-9)
    assert res.fun == pytest.approx(0.0, abs=1e-16)
    assert res.grad_norm < 1e-10
    assert res.used_hessian


def test_rosenbrock_reaches_the_valley_end():
    fun, grad, hvp = _rosenbrock()
    res = trust_region_minimize(fun, grad, np.array([-1.2, 1.0]),
                                hvp=hvp, gtol=1e-10, maxiter=500)
    assert res.stop_reason is TrustRegionStop.CONVERGED
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert res.fun < 1e-12


def test_every_accepted_step_lowers_the_energy():
    fun, grad, hvp = _rosenbrock()
    res = trust_region_minimize(fun, grad, np.array([-1.2, 1.0]),
                                hvp=hvp, gtol=1e-10, maxiter=500)
    hist = np.asarray(res.energy_history, dtype=float)
    assert hist.size == res.n_accept
    assert np.all(np.diff(hist) < 0.0)


def test_an_indefinite_start_escapes_the_saddle():
    """At x = 0.1 the curvature along x is 3x^2 - 1 < 0.

    Steihaug follows that direction to the trust boundary; a correct
    implementation then finds one of the two minima. A method that stops
    where the gradient is small (or trusts a non-positive curvature as if
    it were positive) would stall at the saddle.
    """
    fun, grad, hvp = _double_well()
    start = np.array([0.1, 0.5])
    assert (3.0 * start[0] ** 2 - 1.0) < 0.0
    res = trust_region_minimize(fun, grad, start, hvp=hvp, gtol=1e-10)
    assert res.stop_reason is TrustRegionStop.CONVERGED
    assert res.fun == pytest.approx(0.0, abs=1e-14)
    assert abs(abs(res.x[0]) - 1.0) < 1e-8
    assert abs(res.x[1]) < 1e-8


def test_steihaug_follows_negative_curvature_to_the_boundary():
    g = np.array([-1.0])
    step = steihaug_cg(g, lambda v: -2.0 * np.asarray(v),
                       radius=0.5, cg_tol=1e-12)
    assert step.status is SteihaugStatus.NEGATIVE_CURVATURE
    assert np.linalg.norm(step.p) == pytest.approx(0.5, rel=1e-12)
    # Along -g the model decreases without bound; the boundary is best.
    assert float(g @ step.p) < 0.0


def test_steihaug_solves_a_quadratic_interior_when_the_ball_is_large():
    A = np.diag([1.0, 2.0, 3.0])
    x0 = np.array([2.0, -1.0, 0.5])
    g = A @ x0
    step = steihaug_cg(g, lambda v: A @ np.asarray(v), radius=10.0,
                       cg_tol=1e-12, maxiter=10)
    assert step.status is SteihaugStatus.CONVERGED
    assert np.allclose(step.p, -x0, atol=1e-10)
    assert np.linalg.norm(step.p) < 10.0


def test_every_step_stays_inside_the_radius():
    fun, grad, hvp = _rosenbrock()
    res = trust_region_minimize(fun, grad, np.array([-1.2, 1.0]),
                                hvp=hvp, gtol=1e-10, maxiter=500)
    steps = np.asarray(res.step_norms, dtype=float)
    radii = np.asarray(res.radius_history, dtype=float)
    assert steps.size == radii.size == res.n_iter
    assert np.all(steps <= radii * (1.0 + 1e-12))


def test_without_an_hvp_the_fallback_is_lbfgs():
    """The comparison path reports its own identity and its own floor.

    L-BFGS-B's line search stops at |g| ~ 5e-10 on Rosenbrock even when
    asked for 1e-10 -- the same ftol floor measured in
    ``driver.orbitals`` -- so the target here is 1e-8 and the reported
    stop reason is read off the final gradient, never off scipy's status.
    """
    fun, grad, _ = _rosenbrock()
    res = trust_region_minimize(fun, grad, np.array([-1.2, 1.0]),
                                hvp=None, gtol=1e-8, maxiter=500)
    assert not res.used_hessian
    assert res.stop_reason is TrustRegionStop.CONVERGED
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert res.grad_norm <= 1e-8
    assert res.n_hvp == 0


def test_a_zero_gradient_start_exits_without_a_step():
    fun, grad, hvp = _quadratic([0.0, 0.0], curvature=1.0)
    res = trust_region_minimize(fun, grad, np.zeros(2), hvp=hvp)
    assert res.stop_reason is TrustRegionStop.CONVERGED
    assert res.n_iter == 0
    assert res.n_accept == 0
    assert res.grad_norm == 0.0


def test_the_iteration_cap_is_reported_as_its_own_outcome():
    fun, grad, hvp = _rosenbrock()
    res = trust_region_minimize(fun, grad, np.array([-1.2, 1.0]),
                                hvp=hvp, gtol=1e-14, maxiter=1,
                                initial_radius=0.1)
    assert res.stop_reason is TrustRegionStop.ITERATION_CAP
    assert res.grad_norm > 1e-14


def test_a_wrong_gradient_sign_cannot_raise_the_energy():
    """Acceptance is on the measured energy, not on the model's promise.

    The gradient callable here reports the uphill direction (a sign slip
    of the kind that is easy to make in a new chain). Every proposed step
    then moves uphill; the ratio test rejects all of them, the radius
    shrinks, and the point must be left exactly where it started.
    """
    fun, grad, hvp = _quadratic([3.0, 3.0], curvature=1.0)
    x0 = np.array([10.0, -4.0])
    e0 = fun(x0)

    def lying_grad(x):
        return -1.0 * grad(x)

    res = trust_region_minimize(fun, lying_grad, x0, hvp=hvp,
                                gtol=1e-12, maxiter=60)
    assert res.n_accept == 0
    assert res.fun == e0
    assert res.energy_history == []
    assert res.stop_reason is TrustRegionStop.RADIUS_TOO_SMALL


def test_the_radius_grows_after_good_steps_and_respects_the_cap():
    fun, grad, hvp = _quadratic([50.0], curvature=1.0)
    res = trust_region_minimize(fun, grad, np.zeros(1), hvp=hvp,
                                gtol=1e-10, maxiter=500,
                                initial_radius=1.0, max_radius=4.0)
    radii = np.asarray(res.radius_history, dtype=float)
    assert radii.max() > 1.0
    assert radii.max() <= 4.0 * (1.0 + 1e-12)


def test_extra_arguments_reach_all_three_callables():
    centre = np.array([1.0, -1.0])
    seen = {'fun': 0, 'grad': 0, 'hvp': 0}

    def fun(x, shift):
        seen['fun'] += 1
        return 0.5 * float((np.asarray(x) - centre - shift) @
                           (np.asarray(x) - centre - shift))

    def grad(x, shift):
        seen['grad'] += 1
        return np.asarray(x) - centre - shift

    def hvp(x, v, shift):
        seen['hvp'] += 1
        return np.asarray(v)

    shift = np.array([0.5, 0.5])
    res = trust_region_minimize(fun, grad, np.zeros(2), hvp=hvp,
                                args=(shift,), gtol=1e-10)
    assert res.stop_reason is TrustRegionStop.CONVERGED
    assert np.allclose(res.x, centre + shift, atol=1e-9)
    assert seen['fun'] > 0 and seen['grad'] > 0 and seen['hvp'] > 0


def test_a_non_finite_start_is_refused():
    fun = lambda x: float('nan')
    grad = lambda x: np.zeros(2)
    with pytest.raises(ValueError, match='finite'):
        trust_region_minimize(fun, grad, np.zeros(2))


def test_the_result_reports_its_evaluation_counts():
    fun, grad, hvp = _quadratic([1.0], curvature=1.0)
    res = trust_region_minimize(fun, grad, np.zeros(1), hvp=hvp, gtol=1e-10)
    assert res.n_fun >= res.n_accept + 1
    assert res.n_grad >= res.n_accept + 1
    assert res.n_hvp >= 1
    assert res.n_cg > 0
