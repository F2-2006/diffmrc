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
    from diffmrc.core.nevpt2.impl import eigh_safe

    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    A = jnp.asarray(B + B.T + np.diag([0.0, 2.0, 5.0, 9.0, 14.0]))
    g_std = np.asarray(jax.grad(_scalar(jnp.linalg.eigh, 5))(A))
    g_safe = np.asarray(jax.grad(_scalar(eigh_safe, 5))(A))
    assert np.abs(g_std - g_safe).max() < 1e-12


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
