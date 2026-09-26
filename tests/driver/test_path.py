"""Orbital signatures for the path mapping (kinetic + IAO shells).

The signatures carry no nuclear coordinates, only orbital shapes
(Bensberg & Reiher 2023, eq. 1): the kinetic-energy diagonal and the
populations on shell-resolved intrinsic atomic orbitals.  The tests need
plain PySCF (no jax), so they run on the server and skip locally.
"""
import numpy as np
import pytest

pytest.importorskip('pyscf')


def _h2o(dist=0.9584):
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = (f'O 0 0 0; H {dist} 0 0.5043; H {-dist} 0 0.5043')
    mol.unit = 'Angstrom'
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mf.mo_coeff = np.asarray(mf.mo_coeff)
    return mol, mf


def test_the_signature_shapes_and_the_core_shell_identity():
    from diffmrc.driver.path import orbital_signatures
    mol, mf = _h2o()
    kin, q, labels = orbital_signatures(mol, mf.mo_coeff)
    assert kin.shape == (mol.nao,)
    assert q.shape == (mol.nao, len(labels))
    # The most compact orbital is the oxygen 1s: its population sits on
    # the (O, 1s) shell.  This is the empirical pin of the IAO
    # column-shell association.
    core = int(np.argmax(kin))
    idx = [k for k, (atom, sh) in enumerate(labels)
           if atom == 0 and sh == '1s']
    assert idx, f'no (O, 1s) shell among {labels}'
    assert q[core, idx].sum() > 0.9


def test_the_core_orbital_matches_across_a_small_displacement():
    from diffmrc.driver.path import orbital_signatures
    from diffmrc.policy.mapping import match_orbitals
    mol1, mf1 = _h2o()
    mol2, mf2 = _h2o(dist=1.0084)          # 0.05 A stretch
    kin1, q1, labels1 = orbital_signatures(mol1, mf1.mo_coeff)
    kin2, q2, labels2 = orbital_signatures(mol2, mf2.mo_coeff)
    assert labels1 == labels2
    m = match_orbitals(kin1, q1, kin2, q2, tau_kin=0.05, tau_loc=0.05)
    core = int(np.argmax(kin1))
    assert any(i == core for (i, _, _, _) in m.pairs)
