"""Two-stage orbital-point convergence: loose optimizer, tight refinement."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.stationary import StopReason  # noqa: E402
from diffmrc.driver.orbitals import converge_orbital_point  # noqa: E402


def _quadratic(centre, curvature=2.0):
    """A convex function whose minimum is known."""
    centre = np.asarray(centre, dtype=float)
    curvature = float(curvature)

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(centre)
        return 0.5 * curvature * jnp.sum(d * d)

    return E


def test_the_point_reaches_the_known_minimum():
    E = _quadratic([1.0, -2.0, 0.5])
    p = converge_orbital_point(E, 3)
    assert np.allclose(np.asarray(p.theta), [1.0, -2.0, 0.5], atol=1e-7)
    assert p.is_stationary
    assert p.stop_reason is StopReason.CONVERGED


def test_both_stages_report_their_own_gradient():
    """The unified entry point dropped this evidence.

    That is how "iteration limit reached" (scipy status 1) could sit in
    the code as a successful outcome: the call site saw only a gradient
    and had nothing to compare it against.
    """
    E = _quadratic([3.0], curvature=1.0)
    p = converge_orbital_point(E, 1, optimize_tol=1e-6)
    assert p.optimizer_reached
    assert p.optimizer_grad_norm <= 1e-6
    assert p.optimizer_nit >= 0
    # The refinement went further than the optimizer was asked to.
    assert p.grad_norm <= p.optimizer_grad_norm
    assert p.grad_norm < 1e-11


def test_a_loose_optimizer_target_does_not_stop_the_refinement():
    """The measured production shape: L-BFGS-B stalls, refinement walks.

    With optimize_tol set above the optimizer's practical floor the
    stage is allowed to end above it; the verdict comes from the final
    gradient, not from the optimizer's report.
    """
    E = _quadratic([2.0, -1.0], curvature=0.5)
    p = converge_orbital_point(E, 2, optimize_tol=1e-2)
    assert p.is_stationary
    assert p.grad_norm < 1e-11
    assert np.allclose(np.asarray(p.theta), [2.0, -1.0], atol=1e-6)


def test_a_function_without_a_stationary_point_is_not_stationary():
    """A gradient that never shrinks must not be called converged."""
    E = lambda t: jnp.sum(jnp.asarray(t))          # linear, no minimum
    p = converge_orbital_point(E, 3, max_steps=200)
    assert not p.is_stationary


def test_the_gradient_criterion_cannot_see_a_saddle():
    """A stationary saddle also has |g| = 0, and is reported stationary.

    This is the division of labour, not a defect: the curvature check in
    driver.run is what refuses a saddle. The shape is the measured one --
    the H2O/6-31g point from theta = 0 has three negative eigenvalues
    and refines to |g| = 3.9e-14.
    """
    def E(t):
        t = jnp.asarray(t)
        return 0.25 * t[0] ** 4 - 0.5 * t[0] ** 2 + 0.5 * t[1] ** 2

    p = converge_orbital_point(E, 2, theta0=np.zeros(2))
    assert p.is_stationary                      # |g| = 0 at the saddle
    H = np.asarray(jax.hessian(E)(np.asarray(p.theta)))
    assert np.linalg.eigvalsh(H)[0] < 0        # still a saddle


def test_the_iteration_caps_are_honoured():
    E = _quadratic([5.0] * 4, curvature=1e-3)
    p = converge_orbital_point(E, 4, max_steps=1, refine_max_iter=1)
    # A single step of each stage cannot reach the minimum from far away;
    # the point must say so rather than claim convergence.
    assert not p.is_stationary
    assert p.stop_reason is not StopReason.CONVERGED
