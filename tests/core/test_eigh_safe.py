"""``eigh_safe``: a degeneracy-safe reverse pass for ``jnp.linalg.eigh``.

Regression for the second NaN class found on 2026-09-23/24: the vector
derivative of ``eigh`` carries 1/(w_j - w_i), which is 0/0 at a
machine-precision degeneracy. Linear molecules make such degeneracies
structural -- ThO/ECP's generalized-Fock core block has a minimum gap of
2.7e-15 and its virtual block 8.9e-16, the pi and delta shells -- so the
theta gradient of the whole NEVPT2 chain came back all-NaN (t215), while
the energy stayed finite. Every earlier validation ran on H2 and H2O,
whose blocks carry no degenerate pairs.

``eigh_safe`` keeps the standard reverse pass away from degeneracy and
freezes the rotation inside a degenerate subspace, where the eigenbasis is
arbitrary anyway and NEVPT2's invariance under rotations within the core
and virtual blocks makes the choice immaterial.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _scalar(fn, n):
    """A scalar functional of (w, u) touching both outputs."""
    wgt = jnp.asarray(np.arange(1.0, n + 1.0))
    idxw = jnp.asarray(np.arange(0.0, n))

    def f(a):
        w, u = fn(a)
        return jnp.sum(w * wgt) + jnp.sum(u * idxw[:, None] * wgt[None, :])

    return f


def test_the_safe_reverse_pass_agrees_with_the_plain_one_without_degeneracy():
    """Away from degeneracy the two agree to the softening's own order.

    The coupling is diff/(diff^2 + tau^2), i.e. 1/diff with a relative
    deviation of order (tau/diff)^2 -- and the relevant gap is the one
    of the perturbed matrix, not the diagonal it was built from: the
    smallest eigenvalue gap here is 0.36, so the softening's error is
    ~8e-8 per coupling and the accumulated gradient deviation was
    measured at 3.1e-7 (2026-09-27, an earlier 1e-7 bound assumed a gap
    above 1, which this matrix does not have). Bit-for-bit agreement
    would require the hard 1/diff, whose unbounded second derivative is
    the bug this function exists to avoid.
    """
    from diffmrc.core.nevpt2.impl import eigh_safe

    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    A = jnp.asarray(B + B.T + np.diag([0.0, 2.0, 5.0, 9.0, 14.0]))
    g_std = np.asarray(jax.grad(_scalar(jnp.linalg.eigh, 5))(A))
    g_safe = np.asarray(jax.grad(_scalar(eigh_safe, 5))(A))
    assert np.abs(g_std - g_safe).max() < 1e-6


def test_the_safe_reverse_pass_survives_an_exact_degeneracy():
    from diffmrc.core.nevpt2.impl import eigh_safe

    # A block-diagonal duplication has an exactly doubled spectrum. The
    # plain reverse pass was measured to return all-NaN here (t216); this
    # test asserts only the safe side, since the plain behaviour is a
    # property of the jax build rather than of this repository.
    rng = np.random.default_rng(0)
    B2 = rng.normal(size=(3, 3))
    B2 = B2 + B2.T
    A2 = np.zeros((6, 6))
    A2[:3, :3] = B2
    A2[3:, 3:] = B2
    A2 = jnp.asarray(A2)
    g_safe = np.asarray(jax.grad(_scalar(eigh_safe, 6))(A2))
    assert np.isfinite(g_safe).all()


def test_cluster_ift_differentiates_to_the_frozen_cluster_formula():
    """The eigenvector derivative of the implicit correction must equal
    the standard frozen-cluster perturbation formula.  The sign is what
    the distinct-spectrum case pins, with a finite difference of the
    function's own value path as the independent reference: measured on
    2026-10-01, the expression differentiated to exactly minus that
    reference (t327), and the implicit-correction tests in
    ``test_lanczos`` never saw it because they check finiteness and
    self-consistency, not the absolute derivative."""
    from diffmrc.core.eigh import eigh_cluster_ift

    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A0 = jnp.asarray(Q @ np.diag([1.0, 2.5, 4.0]) @ Q.T)
    Bs = [jnp.asarray(0.5 * (b + b.T)) for b in rng.normal(size=(3, 3, 3))]
    t = rng.normal(size=3)
    t = jnp.asarray(t / np.linalg.norm(t))

    def loss(th):
        _, u = eigh_cluster_ift(A0 + sum(th[k] * Bs[k] for k in range(3)))
        return u[:, 0] @ t

    th0 = jnp.zeros(3)
    g = np.asarray(jax.grad(loss)(th0))
    d = 1e-6
    fd = np.array([
        (float(loss(th0.at[k].add(d))) - float(loss(th0.at[k].add(-d))))
        / (2 * d) for k in range(3)])
    assert np.abs(g - fd).max() < 1e-6
