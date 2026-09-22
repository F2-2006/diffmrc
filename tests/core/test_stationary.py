"""Stationary-point refinement with a monotonicity guard."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.stationary import (  # noqa: E402
    curvature_report,
    refine_stationary_point,
)


def _quadratic(centre, curvature):
    """A convex function whose minimum is known."""
    centre = np.asarray(centre, dtype=float)
    curvature = float(curvature)

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(centre)
        return 0.5 * curvature * jnp.sum(d * d)

    return E


def test_refinement_finds_a_known_minimum():
    E = _quadratic([1.0, -2.0, 0.5], curvature=3.0)
    p = refine_stationary_point(E, np.zeros(3))
    assert np.allclose(np.asarray(p.theta), [1.0, -2.0, 0.5], atol=1e-7)
    assert p.grad_norm < 1e-7
    assert p.energy == pytest.approx(0.0, abs=1e-12)


def test_refinement_never_increases_the_energy():
    """The guard the production scripts lacked.

    A step that does not lower the energy must be rejected, not taken.
    Without the guard the iteration can walk uphill and report the higher
    value as a stationary point -- measured in this project as a jump
    from -76.07 to +1.04 Ha, with no error raised.
    """
    start = _quadratic([3.0, 3.0], 1.0)
    p = refine_stationary_point(start, np.array([9.0, -4.0]))
    assert p.energy <= float(start(np.array([9.0, -4.0])))


def test_a_function_with_no_stationary_point_is_not_called_converged():
    """The failure mode the downhill rule alone does not catch.

    A function that is linear in some direction lowers its energy at
    every step, so every step is accepted and the parameters walk away
    while the gradient never shrinks. This project measured |theta|
    reaching 4e13 on such a surrogate, with the whole iteration budget
    spent and no error raised.

    The gradient is the convergence criterion. Here it stays at 1, so
    the run must stop early and say the gradient stalled rather than
    report a stationary point.
    """
    def E(theta):
        t = jnp.asarray(theta)
        return t[0]

    p = refine_stationary_point(E, np.array([-1.0, 0.0]), max_iter=50)
    assert p.stalled_gradient
    assert p.grad_norm == pytest.approx(1.0, abs=1e-9)
    assert p.n_accepted < 50


def test_a_stalled_gradient_does_not_move_the_parameters_far():
    """The distance cap and the gradient test together bound the damage."""
    def E(theta):
        t = jnp.asarray(theta)
        return t[0]

    p = refine_stationary_point(E, np.zeros(3), max_iter=50)
    assert abs(float(np.asarray(p.theta)[0])) < 100.0


def test_an_already_converged_start_exits_without_touching_the_guard():
    """A zero gradient is a normal exit, not a guard stop.

    Reporting a converged point as 'the guard stopped me' would suggest
    something went wrong when nothing did.
    """
    def E(theta):
        return jnp.asarray(0.0)

    p = refine_stationary_point(E, np.zeros(3), max_iter=50)
    assert not p.guard_stopped
    assert p.n_accepted == 0
    assert p.grad_norm == 0.0


def test_refinement_reports_how_many_steps_it_took():
    E = _quadratic([1.0, -2.0], curvature=1.0)
    p = refine_stationary_point(E, np.zeros(2))
    assert p.n_accepted >= 1


def test_refinement_handles_a_flat_direction():
    """A singular Hessian must not raise; the damping covers it."""
    def E(theta):
        t = jnp.asarray(theta)
        return 1.0 * t[0] ** 2          # no curvature along t[1]

    p = refine_stationary_point(E, np.array([1.0, 1.0]))
    assert np.isfinite(p.energy)
    assert np.isfinite(p.grad_norm)


def test_refinement_is_deterministic():
    """Same start, same answer; no hidden randomness in the damping."""
    E = _quadratic([0.3, -0.7, 1.1], curvature=2.0)
    a = refine_stationary_point(E, np.zeros(3))
    b = refine_stationary_point(E, np.zeros(3))
    assert np.array_equal(np.asarray(a.theta), np.asarray(b.theta))
    assert a.n_accepted == b.n_accepted


def test_refinement_reaches_a_tight_gradient_target():
    """The energy slack must not block convergence to the target.

    Near a minimum a Newton step lowers the energy by roughly
    |g|^2 / (2 kappa). Reaching |g| < 1e-11 therefore means accepting
    energy changes around 1e-22, so a step-acceptance rule that demands a
    decrease of 1e-14 -- the value the experiment scripts used -- rejects
    every step in that regime.

    Measured before the fix: this well-conditioned quadratic stopped
    after one step at |g| = 2.4e-8, because the second step's energy
    change fell below the threshold. The guard exists to catch
    divergence, not to demand visible progress.
    """
    rng = np.random.default_rng(7)
    n = 12
    A = rng.normal(size=(n, n))
    A = A @ A.T + 0.5 * np.eye(n)
    c = rng.normal(size=n)

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(c)
        return 0.5 * d @ jnp.asarray(A) @ d

    p = refine_stationary_point(E, np.zeros(n))
    assert p.grad_norm < 1e-11
    assert np.abs(np.asarray(p.theta) - c).max() < 1e-12


def test_a_well_conditioned_quadratic_converges_in_a_few_steps():
    """Newton is exact on a quadratic; it should not need many steps."""
    rng = np.random.default_rng(11)
    n = 8
    A = rng.normal(size=(n, n))
    A = A @ A.T + np.eye(n)
    c = rng.normal(size=n)

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(c)
        return 0.5 * d @ jnp.asarray(A) @ d

    p = refine_stationary_point(E, np.zeros(n))
    assert p.n_accepted <= 4
    assert p.grad_norm < 1e-11


def test_curvature_report_counts_negative_eigenvalues():
    r = curvature_report(np.diag([-2.0, -1.0, 1.0, 3.0]))
    assert r.n_negative == 2
    assert r.smallest == pytest.approx(-2.0)
    assert r.largest == pytest.approx(3.0)


def test_curvature_report_flags_a_minimum():
    r = curvature_report(np.diag([1.0, 2.0, 3.0]))
    assert r.n_negative == 0
    assert r.is_minimum


def test_curvature_report_counts_near_zero_eigenvalues():
    """A zero curvature direction is a degeneracy, not a saddle.

    The tolerance is what keeps numerical noise from being read as
    negative curvature and a minimum from being reported as a saddle.
    """
    r = curvature_report(np.diag([1.0, 1e-10, 3.0]), negative_tol=1e-8)
    assert r.n_negative == 0
    assert r.n_near_zero == 1


def test_curvature_report_condition_number():
    r = curvature_report(np.diag([1e-4, 1.0]))
    assert r.condition_number == pytest.approx(1e4)


def test_curvature_report_symmetrizes_its_input():
    """AD Hessians are symmetric only to round-off; eigh assumes exact."""
    H = np.array([[2.0, 1.0], [1.0 + 1e-15, 3.0]])
    r = curvature_report(H)
    assert np.isfinite(r.largest)
    assert np.isfinite(r.condition_number)


def test_curvature_report_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        curvature_report(np.zeros((2, 3)))
