"""Canonical closed-shell reference response.

The finite-difference reference is the derivative of the SCF MO
coefficients themselves, not of the density: the density is invariant
under occupied-occupied rotations, so it cannot see the missing
canonical block that the property chain needs.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip('pyscf')

from pyscf import gto, scf  # noqa: E402

from diffmrc.driver.scf_response import (  # noqa: E402
    canonical_rhf_response, density_response, s1_mo)

ATOM = 'O 0 0 0; H 0.7557 0 0.588; H -0.7557 0 0.588'


def _build(coords=None):
    mol = gto.Mole()
    mol.atom = ATOM
    mol.basis = '6-31g'
    mol.build()
    if coords is not None:
        mol.set_geom_(np.asarray(coords), unit='Bohr')
        mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    return mol, mf


def _stretch(mol):
    coords = np.asarray(mol.atom_coords())
    dR = np.zeros_like(coords)
    for i in range(1, len(coords)):
        v = coords[i] - coords[0]
        dR[i] = v / np.linalg.norm(v)
    return dR / np.linalg.norm(dR)


def _aligned(a, b, S):
    """Column signs that align ``b`` to ``a``."""
    return np.sign(np.einsum('ai,ab,bi->i', a, S, b))


def test_mo_response_matches_the_scf_finite_difference():
    mol, mf = _build()
    C = np.asarray(mf.mo_coeff)
    S = mol.intor('int1e_ovlp')
    dR = _stretch(mol)
    dC, K, info = canonical_rhf_response(mf, dR)

    delta = 1e-4
    _, mfp = _build(np.asarray(mol.atom_coords()) + delta * dR)
    _, mfm = _build(np.asarray(mol.atom_coords()) - delta * dR)
    Cp = np.asarray(mfp.mo_coeff)
    Cm = np.asarray(mfm.mo_coeff)
    dC_fd = (Cp * _aligned(C, Cp, S)[None, :]
             - Cm * _aligned(C, Cm, S)[None, :]) / (2 * delta)
    assert np.max(np.abs(dC - dC_fd)) < 1e-4


def test_density_response_matches_the_scf_density_finite_difference():
    mol, mf = _build()
    S = mol.intor('int1e_ovlp')
    dR = _stretch(mol)
    dC, K, info = canonical_rhf_response(mf, dR)
    nocc = int(np.sum(np.asarray(mf.mo_occ) > 0))
    mocc = np.asarray(mf.mo_coeff)[:, :nocc]
    dP = density_response(dC[:, :nocc], mocc)

    delta = 1e-4
    _, mfp = _build(np.asarray(mol.atom_coords()) + delta * dR)
    _, mfm = _build(np.asarray(mol.atom_coords()) - delta * dR)
    dP_fd = (np.asarray(mfp.make_rdm1())
             - np.asarray(mfm.make_rdm1())) / (2 * delta)
    assert np.max(np.abs(dP - dP_fd)) < 1e-7


def test_response_is_orthonormal_in_the_overlap_metric():
    mol, mf = _build()
    dR = _stretch(mol)
    _, K, info = canonical_rhf_response(mf, dR)
    S1 = info['S1']
    assert np.max(np.abs(K + K.T + S1)) < 1e-10


def test_column_sign_gauge_follows_the_base_reference():
    mol, mf = _build()
    dR = _stretch(mol)
    C = np.asarray(mf.mo_coeff)
    dC, _, _ = canonical_rhf_response(mf, dR)
    flip = np.ones(C.shape[1])
    flip[[1, 3, 7]] = -1.0
    dC2, _, _ = canonical_rhf_response(mf, dR, base_C=C * flip[None, :])
    assert np.allclose(dC2, dC * flip[None, :], atol=1e-12)


def test_open_shell_reference_is_refused():
    mol = gto.Mole()
    mol.atom = 'C 0 0 0; H 0.94 0 0.90; H -0.94 0 0.90'
    mol.basis = '6-31g'
    mol.spin = 2
    mol.build()
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    with pytest.raises(NotImplementedError):
        canonical_rhf_response(mf, np.ones((mol.natm, 3)))
