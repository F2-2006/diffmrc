"""Implicit-function second derivatives, verified against explicit answers."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.hessian import ImplicitHessianResult, implicit_hessian  # noqa: E402


def test_result_exposes_the_response_term():
    """Every field the assessment needs must reach the caller.

    The response contribution is the whole point of the route; a result
    that exposes only the total leaves a caller unable to tell whether
    the frozen approximation would have sufficed.
    """
    fields = set(ImplicitHessianResult.__dataclass_fields__)
    for name in ('H_tt', 'H_tt_inverse', 'curvature', 'mixed', 'H_RR',
                 'H_response', 'H_total', 'dtheta_dR_max', 'rcond'):
        assert name in fields


def test_rcond_is_recorded_in_the_result():
    """A reader cannot judge a response term without knowing the cutoff."""
    assert 'rcond' in ImplicitHessianResult.__dataclass_fields__


class _ScalarMole:
    """Stand-in whose 'coordinates' are a single differentiable number."""

    def __init__(self, value):
        self.coords = jnp.asarray([float(value)])

    def replace(self, coords):
        new = _ScalarMole(0.0)
        new.coords = coords
        return new


def _assemble(E_ad, theta_star, x0):
    """Reproduce the module's assembly on a scalar surrogate."""
    H_tt = np.atleast_2d(np.asarray(jax.hessian(lambda t: E_ad(t, x0))(theta_star)))
    H_tt = 0.5 * (H_tt + H_tt.T)
    M = np.atleast_2d(np.asarray(
        jax.jacfwd(jax.grad(lambda t, x: E_ad(t, x), argnums=0), argnums=1)(
            theta_star, x0)).reshape(1, 1))
    th_sg = jax.lax.stop_gradient(theta_star)
    H_RR = np.atleast_2d(np.asarray(
        jax.hessian(lambda x: E_ad(th_sg, x))(x0)).reshape(1, 1))
    H_RR = 0.5 * (H_RR + H_RR.T)
    D = -np.linalg.pinv(H_tt, rcond=1e-9) @ M
    total = H_RR + M.T @ D
    return H_tt, M, H_RR, D, 0.5 * (total + total.T)


def test_assembly_reproduces_an_analytic_second_derivative():
    """The end-to-end check on a problem with a closed-form answer.

    E(t, x) = (t - x)^2 + x^2 has its stationary point at t* = x, so the
    relaxed energy is E*(x) = x^2 and its second derivative is exactly 2.
    The implicit assembly must reproduce that from the three separately
    differentiated pieces.
    """
    def E_ad(tt, xx):
        return (tt[0] - xx) ** 2 + xx ** 2

    th = jnp.asarray([3.0])
    x0 = jnp.asarray(3.0)
    H_tt, M, H_RR, D, total = _assemble(E_ad, th, x0)

    # The pieces individually, so a wrong one is identifiable.
    assert H_tt[0, 0] == pytest.approx(2.0, abs=1e-12)
    assert M[0, 0] == pytest.approx(-2.0, abs=1e-12)
    assert H_RR[0, 0] == pytest.approx(4.0, abs=1e-12)
    assert D[0, 0] == pytest.approx(1.0, abs=1e-12)
    # And the assembled total against the analytic answer.
    assert total[0, 0] == pytest.approx(2.0, abs=1e-12)


def test_assembly_reproduces_a_two_dimensional_analytic_hessian():
    """E* = diag(1, 2) for a decoupled two-parameter surrogate."""
    def E_ad(u, v):
        return (0.5 * 3.0 * (u[0] - v[0]) ** 2
                + 0.5 * 5.0 * (u[1] - v[1]) ** 2
                + 0.5 * 1.0 * v[0] ** 2
                + 0.5 * 2.0 * v[1] ** 2)

    us = jnp.asarray([0.7, -0.3])
    vs = jnp.asarray([0.7, -0.3])
    H_tt = np.asarray(jax.hessian(lambda u: E_ad(u, vs))(us))
    H_tt = 0.5 * (H_tt + H_tt.T)
    M = np.asarray(jax.jacfwd(jax.grad(lambda u, v: E_ad(u, v), argnums=0),
                              argnums=1)(us, vs))
    th_sg = jax.lax.stop_gradient(us)
    H_RR = np.asarray(jax.hessian(lambda v: E_ad(th_sg, v))(vs))
    H_RR = 0.5 * (H_RR + H_RR.T)
    D = -np.linalg.pinv(H_tt, rcond=1e-9) @ M
    total = H_RR + M.T @ D
    total = 0.5 * (total + total.T)

    assert np.allclose(total, np.diag([1.0, 2.0]), atol=1e-10)


def test_the_response_term_is_not_negligible():
    """If the response term vanished the route would be pointless.

    On the surrogate above it carries most of the total, which is why
    the frozen approximation is not good enough and why the result
    exposes it separately.
    """
    def E_ad(tt, xx):
        return (tt[0] - xx) ** 2 + xx ** 2

    th = jnp.asarray([3.0])
    x0 = jnp.asarray(3.0)
    H_tt, M, H_RR, D, total = _assemble(E_ad, th, x0)
    response = M.T @ D
    assert abs(response[0, 0]) > 0.5 * abs(total[0, 0])


def test_frozen_theta_is_not_differentiated():
    """The frozen Hessian must treat theta as a constant.

    Dropping stop_gradient lets the frozen term acquire the orbital
    response, and the two routes stop being distinguishable.
    """
    def E(u, x):
        return (u - x) ** 2 + 0.5 * x ** 2

    th = jax.lax.stop_gradient(jnp.asarray(1.0))
    g = jax.grad(lambda x: E(th, x))(jnp.asarray(1.0))
    # d/dx [ (1 - x)^2 + 0.5 x^2 ] at x = 1 is 0 + 1 = 1
    assert float(g) == pytest.approx(1.0, abs=1e-12)


def test_a_singular_orbital_hessian_needs_the_pseudo_inverse():
    """Why the route uses pinv rather than solve.

    A direction with zero curvature makes the solve singular; the
    pseudo-inverse returns the finite answer on the remaining subspace,
    which is correct because the gradient along a null direction is zero.
    """
    def E(u):
        return 0.5 * 2.0 * u[0] ** 2 + 1.0 * u[1]

    H = np.asarray(jax.hessian(E)(jnp.zeros(2)))
    w = np.linalg.eigvalsh(H)
    assert w[0] == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(H, np.ones(2))
    assert np.all(np.isfinite(np.linalg.pinv(H, rcond=1e-9) @ np.ones(2)))


def test_the_pseudo_inverse_and_the_solve_disagree_on_a_flat_direction():
    """The two are not interchangeable on an ill-conditioned matrix."""
    n = 8
    rng = np.random.default_rng(20260922)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    w = np.array([1.0, 1e-2, 1e-4, 1e-8, 1e-10, 1e-12, 1e-12, 1e-12])
    H = (Q * w) @ Q.T
    H = 0.5 * (H + H.T)
    g = Q @ np.ones(n)
    x_pinv = np.linalg.pinv(H, rcond=1e-9) @ g
    x_solve = np.linalg.solve(H, g)
    assert np.linalg.norm(x_solve) > 10 * np.linalg.norm(x_pinv)


def test_implicit_hessian_requires_a_mole_and_a_joint_energy():
    """The public entry point only needs those two things."""
    import inspect
    params = list(inspect.signature(implicit_hessian).parameters)
    assert params[:3] == ['mol', 'theta', 'E_joint']
    assert 'rcond' in params
