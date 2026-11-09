"""Second derivative of the matrix-free CI eigenvalue.

``core.ci_second_deriv.mf_eigenpair_hessian`` is checked against a central
difference of the first derivative that the package's own ``custom_vjp``
produces. That is the project's standard reference: automatic
differentiation establishes that the rule differentiates the right
expression, and the finite difference establishes that the expression is
the derivative of the energy.

Two higher-order AD routes are pinned as unavailable or unreliable:
reverse-over-reverse raises (the MINRES solve is a ``lax.while_loop``
with a data-dependent bound), and forward-over-reverse (``jax.hessian``)
was measured to return a wrong, asymmetric matrix on a well-conditioned
synthetic block (ground gap 0.062, diagonal -28.66 correct against
+4461.97 returned), so it is not used as a reference here.
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffmrc.core.ci_matvec import lowest_k_eigpairs_mf
from diffmrc.core.ci_second_deriv import mf_eigenpair_hessian

_NACT = 4
_N_ALPHA = 1
_N_BETA = 1
_TOL = 1e-12
# Central-difference noise: the gradient is exact to about tol / gap
# (1e-12 / 0.06), and the step is 1e-5.
_FD_TOL = 1e-5


def _random_integrals(seed):
    """Chemist-notation integrals with the eight-fold symmetry."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((_NACT, _NACT))
    h = 0.5 * (h + h.T)
    g = rng.standard_normal((_NACT,) * 4)
    g = 0.5 * (g + g.transpose(1, 0, 2, 3))
    g = 0.5 * (g + g.transpose(0, 1, 3, 2))
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    return jnp.asarray(h), jnp.asarray(g)


def _energy_fn(integ):
    def energy(p):
        h, g = integ(p)
        return lowest_k_eigpairs_mf(h, g, ncore=0, nact=_NACT,
                                    n_alpha=_N_ALPHA, n_beta=_N_BETA,
                                    k=1, tol=_TOL)[0][0]
    return energy


def _tangents(integ, p0, n_dir):
    unit = [jnp.zeros(n_dir).at[i].set(1.0) for i in range(n_dir)]
    first = [jax.jvp(integ, (p0,), (e,))[1] for e in unit]
    second = [[jax.jvp(lambda t, q=unit[i]: jax.jvp(integ, (t,), (q,))[1],
                       (p0,), (unit[j],))[1] for j in range(n_dir)]
              for i in range(n_dir)]
    return first, second


def _fd_hessian(energy, p0, n_dir, step=1e-5):
    g = jax.jit(jax.grad(energy))
    out = np.zeros((n_dir, n_dir))
    for i in range(n_dir):
        v = np.zeros(n_dir)
        v[i] = 1.0
        out[i] = (np.asarray(g(p0 + step * v)) - np.asarray(g(p0 - step * v))) \
            / (2 * step)
    return out


def test_the_hessian_matches_a_finite_difference_of_the_gradient():
    """Linear parameterization: the integral-curvature term vanishes, so this
    pins the response term and its projector convention."""
    h0, g0 = _random_integrals(5)
    hd, gd = _random_integrals(11)
    n_dir = 3
    dh = jnp.stack([hd * (i + 1.0) for i in range(n_dir)])
    dg = jnp.stack([gd * (i + 1.0) for i in range(n_dir)])

    def integ(p):
        return h0 + jnp.tensordot(p, dh, axes=1), g0 + jnp.tensordot(p, dg,
                                                                    axes=1)

    p0 = jnp.asarray([0.05, -0.04, 0.03])
    first, second = _tangents(integ, p0, n_dir)
    out = mf_eigenpair_hessian(*integ(p0), 0, _NACT, _N_ALPHA, _N_BETA,
                               first, curvatures=second)
    hess = np.asarray(out["hess"])
    assert np.max(np.abs(hess - _fd_hessian(_energy_fn(integ), p0,
                                            n_dir))) < _FD_TOL
    # a Hessian is symmetric; the response part alone is negative
    # semidefinite for a ground state
    assert np.max(np.abs(hess - hess.T)) < 1e-10
    assert np.linalg.eigvalsh(np.asarray(out["hess_response"])).max() < 1e-9
    # and the two terms add up to the total
    total = np.asarray(out["hess_response"]) + np.asarray(out["hess_curvature"])
    assert np.max(np.abs(total - hess)) < 1e-12


def test_the_curvature_term_is_required():
    """A quadratic parameterization carries a nonzero d2H/dp2; dropping the
    curvature term must miss the derivative by that term."""
    h0, g0 = _random_integrals(7)
    hd, gd = _random_integrals(13)

    def integ(p):
        s = p[0]
        return h0 + s * hd + 0.5 * s * s * h0, g0 + s * gd + 0.5 * s * s * g0

    p0 = jnp.asarray([0.07])
    first, second = _tangents(integ, p0, 1)
    energy = _energy_fn(integ)
    hess = float(np.asarray(mf_eigenpair_hessian(
        *integ(p0), 0, _NACT, _N_ALPHA, _N_BETA, first,
        curvatures=second)["hess"])[0, 0])
    fd = _fd_hessian(energy, p0, 1)[0, 0]
    assert abs(hess - fd) < _FD_TOL
    no_curv = float(np.asarray(mf_eigenpair_hessian(
        *integ(p0), 0, _NACT, _N_ALPHA, _N_BETA, first)["hess"])[0, 0])
    assert abs(no_curv - fd) > 1e-3


def test_the_gradient_matches_the_first_order_entry():
    h0, g0 = _random_integrals(17)
    hd, gd = _random_integrals(19)
    n_dir = 2
    dh = jnp.stack([hd, 2.0 * hd])
    dg = jnp.stack([gd, 2.0 * gd])

    def integ(p):
        return h0 + jnp.tensordot(p, dh, axes=1), g0 + jnp.tensordot(p, dg,
                                                                    axes=1)

    p0 = jnp.asarray([0.03, -0.02])
    first, second = _tangents(integ, p0, n_dir)
    out = mf_eigenpair_hessian(*integ(p0), 0, _NACT, _N_ALPHA, _N_BETA,
                               first, curvatures=second)
    g_ad = jax.grad(_energy_fn(integ))(p0)
    assert np.max(np.abs(np.asarray(out["grad"]) - np.asarray(g_ad))) < 1e-10
    assert abs(float(out["lam"])
               - float(_energy_fn(integ)(p0))) < 1e-12


def test_reverse_over_reverse_is_unavailable_on_this_chain():
    """The documented limit: the reverse pass hosts a data-dependent
    while_loop, so it cannot be differentiated in reverse mode again."""
    h0, g0 = _random_integrals(23)
    hd, gd = _random_integrals(29)

    def integ(p):
        return h0 + p[0] * hd, g0 + p[1] * gd

    energy = _energy_fn(integ)
    with np.testing.assert_raises(ValueError):
        jax.grad(lambda p: jax.grad(energy)(p)[0])(jnp.zeros(2))
