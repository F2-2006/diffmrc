"""The ECP contribution to the fixed-coefficient core-Hamiltonian gradient.

The H2O/LiH validation of the NEVPT2 relaxed gradient carries no ECP, so
it exercises only the kinetic and nuclear-attraction derivatives of the
core Hamiltonian.  On an ECP center the core Hamiltonian also carries
``ECPscalar``, whose coordinate derivative has two pieces: the basis
derivative (``ECPscalar_ipnuc``) and the center derivative
(``ECPscalar_iprinv``).  This module pins

* the AD path through the pyscfad integrals,
* the analytic assembly in ``driver.ri_scf.core_hamiltonian_gradient``,
* a central difference of ``core_hamiltonian_energy`` at fixed
  coefficients,

to one another on a fixed, dense, non-symmetric density.  The system is
HI with the LANL2DZ ECP on iodine (8 valence electrons, 13 orbitals).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscf = pytest.importorskip("pyscf")
pyscfad = pytest.importorskip("pyscfad")

_ATOM = "I 0 0 0; H 0 0 3.0406"
_BASIS = {'I': 'lanl2dz', 'H': 'def2-svp'}
_ECP = {'I': 'lanl2dz'}
_SYMBOLS = [s.split()[0] for s in _ATOM.split(';')]


def _build_ad(coords=None):
    from pyscfad import gto

    m = gto.Mole()
    m.atom = _ATOM if coords is None else [
        [_SYMBOLS[k], tuple(coords[k])] for k in range(len(_SYMBOLS))]
    m.basis = _BASIS
    m.ecp = _ECP
    m.unit = 'B'
    m.verbose = 0
    m.build()
    return m


def _build_np(coords=None):
    from pyscf import gto

    m = gto.Mole()
    m.atom = _ATOM if coords is None else [
        [_SYMBOLS[k], tuple(coords[k])] for k in range(len(_SYMBOLS))]
    m.basis = _BASIS
    m.ecp = _ECP
    m.unit = 'B'
    m.verbose = 0
    m.build()
    return m


def _density(nao):
    """A dense density assembled from random orthogonal coefficients.

    A symmetric or diagonal density would make some of the ECP derivative
    blocks invisible; the random orthogonal matrix keeps every AO block
    populated.
    """
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    occ = rng.uniform(0.0, 2.0, size=nao)
    return q, occ, (q * occ) @ q.T


def test_ecp_core_hamiltonian_gradient_matches_ad():
    """The AD gradient of Tr(D h) equals the analytic ECP-aware assembly."""
    from diffmrc.driver.ri_scf import core_hamiltonian_gradient

    m_np = _build_np()
    m_ad = _build_ad()
    assert m_np.has_ecp()
    nao = m_np.nao_nr()
    c, occ, dm = _density(nao)
    dm_j = jnp.asarray(dm)

    def hcore(mol):
        return jnp.sum(dm_j * (mol.intor('int1e_kin')
                               + mol.intor('int1e_nuc')
                               + mol.intor('ECPscalar')))

    g_ad = np.asarray(jax.grad(hcore)(m_ad).coords)
    g_ref = core_hamiltonian_gradient(m_np, c, occ=occ)
    assert np.abs(g_ref).max() > 1e-3       # the check is not vacuous
    assert np.abs(g_ad - g_ref).max() < 1e-10


def test_ecp_core_hamiltonian_gradient_matches_finite_difference():
    """The analytic assembly equals a central difference of the value."""
    from diffmrc.driver.ri_scf import (core_hamiltonian_energy,
                                       core_hamiltonian_gradient)

    m_np = _build_np()
    nao = m_np.nao_nr()
    c, occ, _ = _density(nao)
    g = core_hamiltonian_gradient(m_np, c, occ=occ)
    coords = np.asarray(m_np.atom_coords())
    step = 1e-4
    g_fd = np.zeros_like(g)
    for atom in range(m_np.natm):
        for axis in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[atom, axis] += step
            cm[atom, axis] -= step
            ep = core_hamiltonian_energy(_build_np(cp), c, occ=occ)
            em = core_hamiltonian_energy(_build_np(cm), c, occ=occ)
            g_fd[atom, axis] = (ep - em) / (2.0 * step)
    scale = max(np.abs(g_fd).max(), 1.0)
    assert np.abs(g_fd).max() > 1e-3
    assert np.abs(g - g_fd).max() / scale < 1e-8
