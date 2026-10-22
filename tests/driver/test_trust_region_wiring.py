"""The opt-in trust-region stage in ``converge_orbital_point``.

The default path stays L-BFGS-B plus damped Newton. The new stage is
reachable through ``use_trust_region=True``, and these tests pin three
things: the switch is off by default, the two paths agree on problems
that have a unique answer, and the trust region escapes a start where
the Hessian is indefinite -- the near-degenerate shape the theta surface
of H2O/6-31g exhibits at theta = 0 (three negative eigenvalues, minimum
-1.58e-2).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.stationary import curvature_report  # noqa: E402
from diffmrc.driver import orbitals as orb  # noqa: E402
from diffmrc.driver.orbitals import converge_orbital_point  # noqa: E402


def _quadratic(centre, curvature=2.0):
    centre = np.asarray(centre, dtype=float)
    curvature = float(curvature)

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(centre)
        return 0.5 * curvature * jnp.sum(d * d)

    return E


def _rosenbrock(t):
    x, y = t[0], t[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def _double_well(t):
    return 0.25 * (t[0] ** 2 - 1.0) ** 2 + 0.5 * jnp.sum(t[1:] ** 2)


def test_the_new_stage_is_opt_in():
    E = _quadratic([1.0, -2.0, 0.5])
    default = converge_orbital_point(E, 3)
    trust = converge_orbital_point(E, 3, use_trust_region=True)
    assert default.optimizer_name == 'lbfgs-b'
    assert trust.optimizer_name == 'trust-region'
    assert default.is_stationary and trust.is_stationary
    assert trust.energy == pytest.approx(default.energy, abs=1e-12)
    assert np.allclose(np.asarray(trust.theta), [1.0, -2.0, 0.5],
                       atol=1e-8)


def test_the_trust_region_path_converges_the_rosenbrock_valley():
    p = converge_orbital_point(_rosenbrock, 2,
                               theta0=np.array([-1.2, 1.0]),
                               use_trust_region=True, max_steps=500)
    assert p.optimizer_name == 'trust-region'
    assert p.is_stationary
    assert np.allclose(np.asarray(p.theta), [1.0, 1.0], atol=1e-5)
    assert p.energy < 1e-10
    assert p.grad_norm < 1e-11
    assert p.optimizer_nit >= 1
    assert p.optimizer_message.startswith('trust_region')


def test_a_start_with_negative_curvature_reaches_a_minimum():
    """x = 0.1 sits on the flank of a saddle (f'' = 3x^2 - 1 < 0).

    The damped Newton stage cannot leave a stationary saddle, and
    L-BFGS has no curvature model; the trust region follows the negative
    direction to its boundary. The final point must be a genuine minimum:
    no negative eigenvalue in the Hessian.
    """
    start = np.array([0.1, 0.5, -0.3])
    H0 = np.asarray(jax.hessian(_double_well)(jnp.asarray(start)))
    assert curvature_report(H0).n_negative >= 1

    p = converge_orbital_point(_double_well, 3, theta0=start,
                               use_trust_region=True)
    assert p.is_stationary
    assert p.energy == pytest.approx(0.0, abs=1e-14)
    assert abs(abs(float(np.asarray(p.theta)[0])) - 1.0) < 1e-7
    assert np.allclose(np.asarray(p.theta)[1:], 0.0, atol=1e-7)
    H = np.asarray(jax.hessian(_double_well)(jnp.asarray(p.theta)))
    assert curvature_report(H).n_negative == 0


def test_an_ill_conditioned_quadratic_agrees_between_the_paths():
    """Curvature 1e-3 over four coordinates: the L-BFGS floor is real.

    Both stages must reach the same known minimum; the comparison is on
    the energy, the gradient, and the point, not on which stage was
    faster.
    """
    E = _quadratic([5.0] * 4, curvature=1e-3)
    default = converge_orbital_point(E, 4)
    trust = converge_orbital_point(E, 4, use_trust_region=True)
    assert default.is_stationary and trust.is_stationary
    assert trust.energy == pytest.approx(default.energy, abs=1e-12)
    assert trust.energy == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.asarray(trust.theta), [5.0] * 4, atol=1e-6)
    assert trust.grad_norm < 1e-11


def test_a_failed_trust_region_stage_falls_back_to_lbfgs(monkeypatch):
    """The existing fallback keeps the run alive when the stage cannot run."""
    def _none(*a, **k):
        return None, jnp.zeros(2), float('nan'), dict(
            ok=False, msg='synthetic', nit=-1)

    monkeypatch.setattr(orb, 'optimize_theta_trust_region', _none)
    p = converge_orbital_point(_quadratic([1.0, -1.0]), 2,
                               use_trust_region=True)
    assert p.optimizer_name == 'lbfgs-b'
    assert p.is_stationary
    assert np.allclose(np.asarray(p.theta), [1.0, -1.0], atol=1e-7)
