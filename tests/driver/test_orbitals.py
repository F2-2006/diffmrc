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


def test_the_verdict_follows_the_gradient_that_came_back():
    """Whatever the three stages return, stationarity is read off the
    gradient -- not off which stages ran or how they exited."""
    E = _quadratic([5.0] * 4, curvature=1e-3)
    p = converge_orbital_point(E, 4, max_steps=1, refine_max_iter=1)
    assert p.is_stationary == bool(p.grad_norm < p.target)
    if not p.is_stationary:
        assert p.stop_reason is not StopReason.CONVERGED


def test_a_stalled_point_is_polished_by_a_trust_region_method(monkeypatch):
    """The third stage exists because the first two stop early.

    Measured on H2O/6-31g's lower basin (2026-09-22): the two-stage
    protocol stalls at |g| = 3.7e-4 -- the damped Newton step cap is 10
    and its stall rule gives up -- while trust-ncg with the exact
    Hessian reaches 9.7e-9 at the same energy. The stall was the
    protocol's, not the chart's, so a third stage clears it.

    The two-stage path is faked here so the test is about the
    composition: when the refinement stops short of the target, the
    polish runs and its point is kept when the gradient improves.
    """
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    stalled = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                              grad_norm=3.7e-4, n_accepted=1,
                              stop_reason=StopReason.STALLED_GRADIENT,
                              target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: stalled)
    monkeypatch.setattr(orb, '_trust_region_polish',
                        lambda *a, **k: (np.array([0.1, -0.2]), -1.0000001,
                                         9.7e-9))
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert p.polished
    assert p.grad_norm == pytest.approx(9.7e-9)
    # 9.7e-9 is four orders better than the stall, and it clears the
    # driver's check threshold of 1e-8; it is still above the
    # refinement's 1e-11 target, so the point reports itself as not
    # stationary rather than rounding the claim up.
    assert p.grad_norm < 1e-8
    assert not p.is_stationary


def test_a_polish_that_does_not_improve_is_discarded(monkeypatch):
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    stalled = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                              grad_norm=1e-5, n_accepted=1,
                              stop_reason=StopReason.STALLED_GRADIENT,
                              target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: stalled)
    monkeypatch.setattr(orb, '_trust_region_polish',
                        lambda *a, **k: (np.array([0.0, 0.0]), -0.9, 1e-3))
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert not p.polished
    assert p.grad_norm == pytest.approx(1e-5)


def test_the_polish_is_skipped_when_the_refinement_already_converged(
        monkeypatch):
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    converged = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                                grad_norm=3.9e-14, n_accepted=2,
                                stop_reason=StopReason.CONVERGED,
                                target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: converged)
    called = []

    def _never(*a, **k):
        called.append(True)
        return (np.zeros(2), -1.0, 1e-12)

    monkeypatch.setattr(orb, '_trust_region_polish', _never)
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert not called
    assert not p.polished
    assert p.is_stationary


def test_the_polish_repeats_while_it_keeps_improving(monkeypatch):
    """Measured on H2O/6-31g: one pass reaches 4.6e-8, a second pass is
    what closes the last factor to the 1e-8 gate. Each pass is kept only
    if it lowers the gradient, and the loop stops at the target."""
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    stalled = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                              grad_norm=2.1e-4, n_accepted=1,
                              stop_reason=StopReason.STALLED_GRADIENT,
                              target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: stalled)
    steps = [(np.array([0.1, 0.0]), -1.000001, 4.6e-8),
             (np.array([0.2, 0.0]), -1.000002, 3.0e-9),
             (np.array([0.3, 0.0]), -1.000003, 1.0e-12)]
    calls = []

    def _polish(*a, **k):
        calls.append(True)
        return steps[len(calls) - 1]

    monkeypatch.setattr(orb, '_trust_region_polish', _polish)
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert p.polished
    assert len(calls) == 3            # third pass met the target, loop ends
    assert p.grad_norm == pytest.approx(1.0e-12)
    assert p.is_stationary


def test_hessp_matches_the_explicit_hessian_on_a_non_quadratic_energy():
    """The polish's second-order object is a jvp of the gradient. Pin it
    against jax.hessian: the two must act identically on a vector. The
    HVP path is what keeps the memory at one tangent vector instead of
    the npar^2 tangent pairs the fwd-over-fwd Hessian broadcasts --
    measured on the M5-C batch: a 402 GB allocation at npar ~ 500."""
    from diffmrc.driver.orbitals import hessp_of

    def E(t):
        return (jnp.sum(jnp.sin(t)) + 0.5 * jnp.sum(t ** 2)
                + 0.25 * t[0] ** 2 * t[1] ** 2)

    t = jnp.asarray([0.3, -0.7, 1.1, 0.2])
    v = jnp.asarray([1.0, -0.5, 0.25, 2.0])
    got = np.asarray(hessp_of(E)(t, v))
    want = np.asarray(jax.hessian(E)(t)) @ np.asarray(v)
    assert np.allclose(got, want, atol=1e-10)


def test_the_polish_descends_a_curved_valley_with_the_hvp_hessian():
    """End to end through scipy's trust-ncg with the HVP Hessian -- the
    exact call path the driver's third stage uses -- on the classic
    stalling valley (Rosenbrock)."""
    from diffmrc.driver.orbitals import _trust_region_polish

    def E(t):
        return (1.0 - t[0]) ** 2 + 100.0 * (t[1] - t[0] ** 2) ** 2

    out = _trust_region_polish(E, np.array([-1.2, 1.0]), 1e-10,
                               max_iter=200)
    assert out is not None
    th, en, gn = out
    assert np.allclose(th, [1.0, 1.0], atol=1e-4)
    assert en == pytest.approx(0.0, abs=1e-8)
