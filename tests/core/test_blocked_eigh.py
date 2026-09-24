"""The blocked CI solver must be first-order safe on bit-exact degeneracies.

Regression for the third NaN class found on 2026-09-24: the raw reverse rule
of ``jnp.linalg.eigh`` forms 1/(w_j - w_i) for every eigenvector pair,
whether or not that pair carries cotangent, so one *bit-exact* degenerate
pair anywhere in a block makes the untouched entries 0 * inf = NaN.
Measured on CO/6-31G CAS(6,6): five of the seven S_z blocks carry bit-exact
pairs -- 24 of them in the 400-determinant winning block, whose own ground
state is simple -- and the raw-eigh theta gradient is all-NaN while the
energy is finite (t232r, t237). H2O and ThO never triggered it: their block
spectra have no bit-exact pairs (roundoff-level gaps are a different,
finite-failure mode handled by the same rule).

The solver therefore goes through ``core.eigh.eigh_safe``, whose
Lorentzian-softened rule freezes the rotation inside a degenerate subspace.

A diagonal two-orbital model reproduces the CO structure exactly with no
integration machinery: with ``eri = 0`` and ``h1e = diag(a, b)`` the Sz = 0
block carries the entries {2a, a+b, a+b, 2b}, so for a < b < 0 it wins with
a simple ground state and a bit-exact excited pair.
"""
import numpy as np
import pytest

pytest.importorskip("pyscfad")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _blocked_sum_ci(h1e):
    from diffmrc.core.casci import cs_casci_blocked

    _, ci = cs_casci_blocked(h1e, jnp.zeros((2, 2, 2, 2)),
                             ncore=0, nact=2, nele_act=2)
    return jnp.sum(ci)


def test_the_blocked_solver_energy_is_the_diagonal_minimum():
    from diffmrc.core.casci import cs_casci_blocked

    h1e = jnp.asarray(np.diag([-1.0, -0.5]))
    e, _ = cs_casci_blocked(h1e, jnp.zeros((2, 2, 2, 2)),
                            ncore=0, nact=2, nele_act=2)
    assert float(e) == -2.0


def test_the_blocked_solver_gradient_is_finite_at_a_bitexact_pair():
    """The winning block has a simple ground state and a bit-exact pair.

    This is the CO/6-31G configuration: the pair sits at the excited entries
    (a+b, twice), the consumed column is the ground state, and the raw rule
    was measured all-NaN through the solver's vector path (t232r). The test
    asserts the safe side only, since the raw behaviour is a property of the
    jax build; it pins that this repository does not route through it.
    """
    h1e = jnp.asarray(np.diag([-1.0, -0.5]))
    g = np.asarray(jax.grad(_blocked_sum_ci)(h1e))
    assert np.isfinite(g).all()
