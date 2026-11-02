"""The gradient-difference refinement used when the analytic Hessian fails.

The physical case is N2/6-31G CAS(4,4): the energy and its analytic gradient
are finite at the symmetric stationary point, while every entry of the
analytic orbital Hessian is NaN. The first test rebuilds that defect on a
synthetic energy whose analytic second derivative is undefined at one point
and checks that the refinement still descends. The second test uses a smooth
convex energy to check the gradient-difference Hessian against the analytic
one, the convergence of the route, and the parameter cap.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.driver.orbitals import (  # noqa: E402
    FD_HESSIAN_MAX_PARAMETERS, _finite_difference_newton,
    _trust_region_polish)


@jax.custom_jvp
def _kink(x):
    """x * sqrt(|x|): value and stated first derivative are finite at 0.

    The tangent rule states the first derivative explicitly as sqrt(|x|),
    so ``jax.grad`` is finite at zero. Differentiating that rule reaches
    ``0.5 / sqrt(|x|)`` at zero, which is not finite -- the same failure
    mode as the blocked eigensolver's Hessian at a degeneracy, while the
    energy and gradient themselves stay smooth.
    """
    return x * jnp.sqrt(jnp.abs(x))


@_kink.defjvp
def _kink_jvp(primals, tangents):
    x, = primals
    t, = tangents
    return x * jnp.sqrt(jnp.abs(x)), jnp.sqrt(jnp.abs(x)) * t


def _defective(centre=(1.0, 1.0), kink=0.02):
    """A convex energy whose analytic Hessian is not finite at the origin."""
    centre = np.asarray(centre, dtype=float)

    def E(theta):
        theta = jnp.asarray(theta)
        d = theta - jnp.asarray(centre)
        return 0.5 * jnp.sum(d * d) + kink * jnp.sum(_kink(theta))

    return E


def test_the_refinement_switches_to_gradient_differences_and_descends():
    """Finite energy and gradient, non-finite Hessian: the polish must cope.

    The probe sees the non-finite Hessian-vector product and hands the
    point to the gradient-difference Newton, which must lower the gradient
    and never raise the energy.
    """
    E = _defective()
    start = np.zeros(2)
    e_start = float(E(jnp.asarray(start)))
    g_start = float(jnp.linalg.norm(jax.grad(E)(jnp.asarray(start))))

    # The defect itself: the analytic second derivative is unusable.
    assert np.isfinite(e_start)
    assert np.all(np.isfinite(np.asarray(jax.grad(E)(jnp.asarray(start)))))
    assert not np.all(np.isfinite(np.asarray(jax.hessian(E)(jnp.asarray(start)))))

    out = _trust_region_polish(E, start, 1e-10, max_iter=200)

    assert out is not None
    theta, energy, grad_norm = out
    assert np.all(np.isfinite(theta))
    assert energy <= e_start + 1e-12
    assert grad_norm < g_start
    # The synthetic kink makes the true curvature diverge at the start
    # point (the real defect does not: N2's curvature there is finite,
    # 0.065 to 67). The gradient-difference Hessian therefore starts out
    # very large and the approach is slow; measured floor in 20
    # iterations is 1.4e-6. The smooth-energy test below is what pins the
    # gradient-difference construction to 1e-9.
    assert grad_norm < 1e-5


def test_the_gradient_difference_hessian_matches_the_analytic_one_and_caps():
    """On smooth ground the two second derivatives agree; the cap declines."""
    centre = np.array([0.3, -0.7])

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(centre)
        return 0.5 * jnp.sum(d * d) + 0.25 * jnp.sum(d ** 4)

    start = np.array([1.1, 0.4])
    out = _finite_difference_newton(E, start, 1e-12, max_iter=20)

    assert out is not None
    theta, energy, grad_norm = out
    assert grad_norm < 1e-9
    assert energy <= float(E(jnp.asarray(start))) + 1e-12
    assert np.allclose(theta, centre, atol=1e-4)

    # The construction itself: at a point where the analytic Hessian is
    # finite, the central difference of the gradient reproduces it.
    from diffmrc.core.stationary import curvature_from_gradient
    x = np.array([0.2, -0.1])
    fd = curvature_from_gradient(
        lambda t: np.asarray(jax.grad(E)(jnp.asarray(t))), x, step=1e-5)
    analytic = np.asarray(jax.hessian(E)(jnp.asarray(x)))
    assert np.allclose(fd, analytic, rtol=1e-6, atol=1e-6)

    # Above the cap the route declines instead of building a large matrix.
    assert _finite_difference_newton(
        E, start, 1e-10, max_parameters=1) is None
    assert FD_HESSIAN_MAX_PARAMETERS >= 101
