"""The decomposition-free lowest-eigenpair kernel (``core/lanczos.py``).

Regression for the second-derivative limit found on 2026-09-24: `eigh`,
`eigvalsh` and `eigh_safe` all return NaN at second order on a matrix with
a bit-exact degenerate pair, because the outer differentiation mode
re-enters the primal decomposition. The kernel here reaches the lowest
pair through matrix-vector products and shifted dense solves only, so no
decomposition of the large matrix is ever differentiated. Measured on the
CO/6-31G CAS(6,6) winning block (400 determinants, 24 bit-exact pairs):
the HVP is finite at every Krylov depth probed, while the decomposition
path is NaN.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def test_the_lowest_pair_matches_eigh_away_from_degeneracy():
    from diffmrc.core.lanczos import lowest_eigpair_lanczos

    rng = np.random.default_rng(0)
    B = rng.normal(size=(8, 8))
    A = jnp.asarray(B + B.T + np.diag(np.arange(8.0)))
    w, u = lowest_eigpair_lanczos(A, m=8)
    w_e, u_e = jnp.linalg.eigh(A)
    assert abs(float(w) - float(w_e[0])) < 1e-10
    assert abs(float(u @ u_e[:, 0])) > 1.0 - 1e-8


def test_the_lowest_pair_survives_a_bit_exact_degeneracy():
    from diffmrc.core.lanczos import lowest_eigpair_lanczos

    A = jnp.asarray(np.diag([-1.0, 0.5, 0.5, 2.0]))
    w, u = lowest_eigpair_lanczos(A)
    assert abs(float(w) - (-1.0)) < 1e-12
    assert abs(float(jnp.linalg.norm(u)) - 1.0) < 1e-10


def test_the_second_derivative_is_finite_where_the_decomposition_is_nan():
    """The capability the decomposition-based kernels cannot provide.

    Measured on jax 0.10.2: with a bit-exact pair away from the consumed
    column, the second derivative of ``eigh``/``eigvalsh`` of the lowest
    eigenvalue is all-NaN; this kernel's is finite. The test pins both
    halves so that a future jax change makes the contrast explicit.
    """
    from diffmrc.core.lanczos import lowest_eigpair_lanczos

    A = jnp.asarray(np.diag([-1.0, 0.5, 0.5, 2.0]))
    V = jnp.asarray(np.diag([1.0, 2.0, -0.5, 0.3]))
    f = lambda M: lowest_eigpair_lanczos(M)[0]                    # noqa: E731
    hv = np.asarray(jax.jvp(jax.grad(f), (A,), (V,))[1])
    assert np.isfinite(hv).all()
    hv_e = np.asarray(jax.jvp(
        jax.grad(lambda M: jnp.linalg.eigh(M)[0][0]), (A,), (V,))[1])
    assert not np.isfinite(hv_e).all()


def test_the_blocked_solver_runs_on_the_lanczos_kernel():
    """The ``eig_solver='lanczos'`` path through ``cs_casci_blocked``.

    Same diagonal model as the blocked-solver regression: the Sz = 0
    block carries a bit-exact excited pair, so this exercises the kernel
    inside the solver rather than standalone. Energy exact, gradient
    through the vector path finite.
    """
    pytest.importorskip("pyscfad")
    from diffmrc.core.casci import cs_casci_blocked

    h1e = jnp.asarray(np.diag([-1.0, -0.5]))
    eri = jnp.zeros((2, 2, 2, 2))
    e, _ = cs_casci_blocked(h1e, eri, ncore=0, nact=2, nele_act=2,
                            eig_solver='lanczos')
    assert abs(float(e) - (-2.0)) < 1e-10
    g = np.asarray(jax.grad(lambda h: jnp.sum(cs_casci_blocked(
        h, eri, ncore=0, nact=2, nele_act=2, eig_solver='lanczos')[1]))(h1e))
    assert np.isfinite(g).all()


def test_cluster_ift_values_match_and_the_projector_derivative_is_consistent():
    """``eigh_cluster_ift``: values exact; a smooth invariant functional's
    second derivative matches finite differences.

    The functional is the fixed-size cluster projector of the lowest two
    eigenvalues (the cluster size is fixed so the functional is smooth
    even when the pair is bit-exactly degenerate -- a membership mask
    would jump). This is the property the driver chain needs: the
    natural-orbital transform may only rotate the degenerate cluster
    freely if downstream quantities do not see it.
    """
    from diffmrc.core.eigh import eigh_cluster_ift

    A = jnp.asarray(np.diag([-1.0, -1.0, 0.5, 2.0]))
    w, u = eigh_cluster_ift(A)
    assert np.allclose(np.asarray(w), [-1.0, -1.0, 0.5, 2.0], atol=1e-12)
    assert np.allclose(np.asarray(u @ u.T), np.eye(4), atol=1e-12)
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.normal(size=(4, 4)))

    def f(M):
        _, uu = eigh_cluster_ift(M)
        P = uu[:, :2] @ uu[:, :2].T
        return jnp.sum(P * W)

    V = jnp.asarray(np.diag([1.0, 2.0, -0.5, 0.3]))
    hv = np.asarray(jax.jvp(jax.grad(f), (A,), (V,))[1])
    gp = np.asarray(jax.grad(f)(A + 1e-5 * V))
    gm = np.asarray(jax.grad(f)(A - 1e-5 * V))
    fd = float(np.sum((gp - gm) / 2e-5 * np.asarray(V)))
    ad = float(np.sum(hv * np.asarray(V)))
    assert np.isfinite(hv).all()
    assert abs(ad - fd) <= 1e-4 * max(abs(fd), 1.0)
