"""The trust-region stage through the optimizer's return contract.

``optimize_theta_trust_region`` is the third sibling of
``optimize_theta_scipy`` and ``optimize_theta_hand``: same arguments,
same ``(E_jit, theta, gnorm, info)`` return, so a caller can swap stages
without glue. The tests fix that contract on known minima and on the
Rosenbrock valley, where the two stages must agree on the point even
though they reach it by different paths.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core import optimizer  # noqa: E402
from diffmrc.core.optimizer import (  # noqa: E402
    optimize_theta_scipy,
    optimize_theta_trust_region,
)


def _quadratic(centre):
    centre = jnp.asarray(np.asarray(centre, dtype=float))

    def E(t):
        d = jnp.asarray(t) - centre
        return 0.5 * jnp.sum(d * d)

    return E


def _rosenbrock(t):
    x, y = t[0], t[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def test_the_stage_reaches_a_known_minimum():
    E = _quadratic([1.0, -2.0, 0.5])
    E_jit, theta, gnorm, info = optimize_theta_trust_region(
        E, 3, steps=100, g_conv=1e-10)
    assert np.allclose(np.asarray(theta), [1.0, -2.0, 0.5], atol=1e-8)
    assert gnorm < 1e-10
    assert info['ok']
    assert info['msg'].startswith('trust_region')
    assert info['nit'] >= 1
    assert info['n_accept'] >= 1
    assert info['n_hvp'] >= 1
    assert info['E'] == pytest.approx(0.0, abs=1e-16)
    assert float(E_jit(theta)) == pytest.approx(info['E'], abs=1e-16)


def test_the_stage_and_lbfgs_agree_on_the_rosenbrock_minimum():
    start = np.array([-1.2, 1.0])
    _, th_tr, g_tr, info_tr = optimize_theta_trust_region(
        _rosenbrock, 2, steps=500, g_conv=1e-8, theta0=start)
    _, th_lb, g_lb, info_lb = optimize_theta_scipy(
        _rosenbrock, 2, steps=500, g_conv=1e-8, theta0=start)
    assert np.allclose(np.asarray(th_tr), [1.0, 1.0], atol=1e-5)
    assert np.allclose(np.asarray(th_lb), [1.0, 1.0], atol=1e-4)
    assert abs(float(_rosenbrock(th_tr))
               - float(_rosenbrock(th_lb))) < 1e-12
    assert g_tr < 1e-7
    assert info_tr['ok'], info_tr


def test_arrays_travel_as_arguments_not_through_the_closure():
    shift = np.array([1.0, 2.0])

    def E(t, s):
        d = jnp.asarray(t) - jnp.asarray(s)
        return 0.5 * jnp.sum(d * d)

    E_jit, theta, gnorm, info = optimize_theta_trust_region(
        E, 2, steps=50, g_conv=1e-10, arrays=(shift,))
    assert np.allclose(np.asarray(theta), shift, atol=1e-8)
    assert info['ok']


def test_a_failing_inner_solver_returns_none_instead_of_raising(monkeypatch):
    """The caller keeps its existing fallback when the new stage cannot run.

    ``optimize_theta_scipy`` already reports a raised exception as
    ``E_jit is None`` so ``converge_orbital_point`` can fall back to the
    handwritten optimizer. The trust-region stage follows the same
    convention -- a stage that cannot run must not take the run down.
    """
    def _boom(*a, **k):
        raise RuntimeError('synthetic failure')

    monkeypatch.setattr(optimizer, 'trust_region_minimize', _boom)
    E_jit, theta, gnorm, info = optimize_theta_trust_region(
        _quadratic([0.0]), 1)
    assert E_jit is None
    assert not info['ok']
    assert 'RuntimeError' in info['msg']
    assert np.isnan(gnorm)
