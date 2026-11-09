"""Tests for the MS-CASPT2 / XMS-CASPT2 zeroth-order Hamiltonians.

The assertions are the ones of the development prototype, unchanged; only
the import paths moved to the release package layout.  The CH2 fixture
needs pyscf, which is an optional dependency of the stack, so the
benchmark tests skip when it is absent.
"""
import numpy as np
import pytest

from diffmrc.core.mscaspt2.model import sz0_dets, transition_1rdm


def test_sz0_dets_count():
    """C(6,3)^2 determinants for six electrons in six orbitals."""
    dets = sz0_dets(0, 6, 6)
    assert len(dets) == 20 * 20
    assert len(set(dets)) == len(dets)
    for d in dets:
        assert d.bit_count() == 6


def test_sz0_dets_rejects_odd():
    with pytest.raises(ValueError):
        sz0_dets(1, 3, 3)


def test_transition_1rdm_hermitian():
    """G^MN = (G^NM)^T for real CI vectors."""
    dets = sz0_dets(0, 4, 4)
    rng = np.random.default_rng(3)
    c1 = rng.normal(size=len(dets))
    c2 = rng.normal(size=len(dets))
    G12 = transition_1rdm(c1, c2, dets, 4)
    G21 = transition_1rdm(c2, c1, dets, 4)
    assert np.allclose(G12, G21.T, atol=1e-12)


def test_transition_1rdm_trace_is_overlap():
    """sum_p G^MN[p, p] = N_active * <M|N>."""
    dets = sz0_dets(0, 4, 4)
    rng = np.random.default_rng(5)
    c1 = rng.normal(size=len(dets))
    c2 = rng.normal(size=len(dets))
    G = transition_1rdm(c1, c2, dets, 4)
    assert np.isclose(np.trace(G), 4.0 * float(c1 @ c2), atol=1e-10)


@pytest.fixture(scope='module')
def small_system():
    pytest.importorskip('pyscf')
    from diffmrc.core.mscaspt2.bench import build_system
    return build_system('C 0 0 0; H 0.9526 0 0.5500; H -0.9526 0 0.5500',
                        'sto-3g', 2, 2, 2)


def test_model_hamiltonian_reproduces_sa_energies(small_system):
    """The model determinant Hamiltonian has the SA-CASSCF eigenvalues."""
    w = np.linalg.eigvalsh(small_system['H_model']) + small_system['e_nuc']
    for e in small_system['sa_energies']:
        assert np.min(np.abs(w - e)) < 1e-7


def test_reference_states_are_model_eigenstates(small_system):
    H = small_system['H_model']
    civs = small_system['civs']
    e = small_system['casci_energies']
    assert np.allclose(civs @ H @ civs.T, np.diag(e), atol=1e-8)


def test_ms_heff_diagonal_is_reference_plus_e2(small_system):
    from diffmrc.core.mscaspt2.ms import solve_ms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    out = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                   small_system['civs'], small_system['dets'], **common)
    diag = np.diag(out['H_eff'])
    expected = small_system['casci_energies'] + out['e2']
    assert np.allclose(diag, expected, atol=1e-9)


def test_single_state_matches_release_solver(small_system):
    """One state: MS-CASPT2 collapses onto the release single-state solve."""
    from diffmrc.core.caspt2 import caspt2_contracted_e2_coupled
    from diffmrc.core.mscaspt2.ms import averaged_fock, solve_ms

    civs = small_system['civs'][:1]
    ncore, nact, nele = (small_system['ncore'], small_system['nact'],
                         small_system['nele'])
    f = averaged_fock(small_system['h1_mo'], small_system['eri_mo'], civs,
                      small_system['dets'], ncore + nact)
    ref, _ = caspt2_contracted_e2_coupled(
        small_system['h1_mo'], small_system['eri_mo'], civs[0],
        small_system['dets'], ncore=ncore, nact=nact, nele=nele, fock=f)

    out = solve_ms(small_system['h1_mo'], small_system['eri_mo'], civs,
                   small_system['dets'], ncore=ncore, nact=nact, nele=nele,
                   H_model=small_system['H_model'],
                   det_index=small_system['det_index'])
    assert np.isclose(out['energies'][0],
                      small_system['casci_energies'][0] + ref, atol=1e-10)


def test_rotation_invariance_ms_and_xms(small_system):
    """MS-MR with the S13 eq. 1 coupling and XMS are both rotation invariant.

    The coupled MS-MR spectrum and the XMS spectrum must be unchanged
    under a unitary rotation of the reference states.  The legacy
    diagonal-projection MS-MR and the SS-SR path stay non-invariant and
    are kept as the contrast.
    """
    from diffmrc.core.mscaspt2.ms import random_unitary, solve_ms, solve_xms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    civs = small_system['civs']
    Q = random_unitary(civs.shape[0], seed=11)

    xms = solve_xms(small_system['h1_mo'], small_system['eri_mo'], civs,
                    small_system['dets'], **common)
    xms_r = solve_xms(small_system['h1_mo'], small_system['eri_mo'],
                      Q.T @ civs, small_system['dets'], **common)
    assert np.allclose(xms['energies'], xms_r['energies'], atol=1e-9)

    ms = solve_ms(small_system['h1_mo'], small_system['eri_mo'], civs,
                  small_system['dets'], **common)
    ms_r = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                    Q.T @ civs, small_system['dets'], **common)
    assert np.allclose(ms['energies'], ms_r['energies'], atol=1e-9)

    legacy = solve_ms(small_system['h1_mo'], small_system['eri_mo'], civs,
                      small_system['dets'], reference_coupling=False,
                      **common)
    legacy_r = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                        Q.T @ civs, small_system['dets'],
                        reference_coupling=False, **common)
    assert not np.allclose(legacy['energies'], legacy_r['energies'],
                           atol=1e-9)

    ss = solve_ms(small_system['h1_mo'], small_system['eri_mo'], civs,
                  small_system['dets'], fock_mode='state',
                  reference_coupling=False, **common)
    ss_r = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                    Q.T @ civs, small_system['dets'], fock_mode='state',
                    reference_coupling=False, **common)
    assert not np.allclose(ss['energies'], ss_r['energies'], atol=1e-9)


def test_coupled_ms_equals_rotated_xms(small_system):
    """The S13 eq. 1 system is the same in the original and Fock bases."""
    from diffmrc.core.mscaspt2.ms import solve_ms, solve_xms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    ms = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                  small_system['civs'], small_system['dets'], **common)
    xms = solve_xms(small_system['h1_mo'], small_system['eri_mo'],
                    small_system['civs'], small_system['dets'],
                    reference_coupling=True, **common)
    assert np.allclose(ms['energies'], xms['energies'], atol=1e-9)


def test_coupled_cross_matches_psis(small_system):
    """``cross`` from the Sylvester solve equals the psi contraction."""
    from diffmrc.core.mscaspt2.ms import coupling_matrix, solve_ms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    out = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                   small_system['civs'], small_system['dets'], **common)
    cross = coupling_matrix(out['psis'], small_system['civs'],
                            small_system['h1_mo'], small_system['eri_mo'],
                            small_system['dets'])
    assert np.allclose(out['cross'], cross, atol=1e-10)


def test_heff_symmetric(small_system):
    from diffmrc.core.mscaspt2.ms import solve_ms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    out = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                   small_system['civs'], small_system['dets'], **common)
    assert np.allclose(out['H_eff'], out['H_eff'].T, atol=1e-12)
