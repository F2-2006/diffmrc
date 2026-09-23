"""The S_z-blocked CASCI eigensolver: the returned vector must live in the
lowest block alone.

Regression for a residue bug fixed on 2026-09-23. The per-block loop of
``cs_casci_blocked`` wrote each winning block's eigenvector at its own index
set but never erased the vector of a previously leading block, which sits at
a different index set. Whenever a later block won (the normal case, since
the S_z ordering visits the extreme blocks first), the returned "vector" was
a mixture of several S_z blocks.

Why it survived every previous validation: the energies returned by both
solvers were always correct (they come from ``eigh`` eigenvalues), and every
blocked-path check up to that date had been self-consistent -- HVP against a
finite difference of the same chain, hv.w == hw.v, gradient finiteness. A
self-consistent chain reproduces its own errors. The plain solver was the
only one externally cross-checked (3.2e-10 against PySCF on H2/6-31G), and
the mixture was harmless there only by accident of that system. The bug was
exposed by a ThO/ECP cross-check (CAS(4,4), ncore = 17) where the mixed
vector produced a 1-RDM wrong by 3.0 in max element and a NEVPT2 correction
wrong by 1.6 Ha; after the fix, that correction agrees with PySCF to 3.8e-7.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscfad = pytest.importorskip("pyscfad")


def _h2o_cas22_ints():
    """H2O/6-31G MO integrals over core+active (ncore=4, CAS(2,2)).

    H2O/6-31G is the smallest system that exercises the bug: its S_z blocks
    are ordered (0,2) -> (1,1) -> (2,0), the ground state lives in the
    middle block, so the leading vector of the first block is left behind
    by the unfixed loop.
    """
    from pyscfad import gto, scf

    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
    mol.basis = "6-31g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    C = jnp.asarray(mf.mo_coeff)
    h = jnp.asarray(mf.get_hcore())
    e = jnp.asarray(mol.intor("int2e", aosym="s1"))
    nm = 4 + 2
    h1 = jnp.einsum("ai,ab,bj->ij", C, h, C)[:nm, :nm]
    e1 = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, e)[
        :nm, :nm, :nm, :nm]
    return h1, e1


def _spin_block(det, ncore, nact):
    """(n_alpha, n_beta) active counts of a determinant bitmap.

    ``cas_dets_for2`` returns integer bitmaps (bit ``so`` set = spin orbital
    ``so`` occupied), not tuples of spin-orbital indices.
    """
    occ = [i for i in range(2 * (ncore + nact)) if (det >> i) & 1]
    n_alpha = sum(1 for so in occ if so % 2 == 0)
    return (n_alpha - ncore, len(occ) - n_alpha - ncore)


def test_the_blocked_ci_lives_in_a_single_spin_block():
    from diffmrc.core.casci import cs_casci, cs_casci_blocked
    from diffmrc.core.nevpt2.impl import cas_dets_for2

    h1, e1 = _h2o_cas22_ints()
    E_u, ci_u = cs_casci(h1, e1, ncore=4, nact=2, nele_act=2, return_ci=True)
    E_b, ci_b = cs_casci_blocked(h1, e1, ncore=4, nact=2, nele_act=2)
    ci_u = np.asarray(ci_u).reshape(-1)
    ci_b = np.asarray(ci_b).reshape(-1)

    # The two solvers describe the same state. A global sign is not fixed
    # by either solver (the plain one runs inverse iteration from a
    # positive start; the blocked one takes whatever eigh returns).
    assert float(E_b) == pytest.approx(float(E_u), abs=1e-10)
    assert abs(float(np.dot(ci_b, ci_u))) == pytest.approx(1.0, abs=1e-8)

    # The support of the blocked vector lies in exactly one S_z block.
    # Under the residue bug this set had two elements: the true ground
    # block plus the (0,2) singlet of the first-visited block, carrying
    # the full unit weight of that block's leading vector.
    dets = cas_dets_for2(4, 2, 2)
    nz = np.where(np.abs(ci_b) > 1e-8)[0]
    blocks = {_spin_block(dets[i], 4, 2) for i in nz}
    assert len(blocks) == 1


def test_the_blocked_ci_reproduces_the_plain_density_matrix():
    """The 1-RDM built from the blocked vector must match the plain one.

    This is the quantity the NEVPT2 chain consumes; the residue bug moved
    it by 3.0 in max element on ThO/ECP (where the wrong 1-RDM is
    non-physical: diagonal occupations above 2).
    """
    from diffmrc.core.casci import cs_casci, cs_casci_blocked
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.nevpt2.rdm import make_dm123_sf

    h1, e1 = _h2o_cas22_ints()
    _, ci_u = cs_casci(h1, e1, ncore=4, nact=2, nele_act=2, return_ci=True)
    _, ci_b = cs_casci_blocked(h1, e1, ncore=4, nact=2, nele_act=2)
    dets = cas_dets_for2(4, 2, 2)
    dm_u, _, _ = make_dm123_sf(jnp.asarray(ci_u).reshape(-1), dets, 2, 4)
    dm_b, _, _ = make_dm123_sf(jnp.asarray(ci_b).reshape(-1), dets, 2, 4)
    assert float(np.abs(np.asarray(dm_b) - np.asarray(dm_u)).max()) < 1e-8
