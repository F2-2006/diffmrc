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


def test_h0_matrix_matches_reference_assembly(small_system):
    """The vectorized H0 equals the per-determinant release assembly."""
    from diffmrc.core.caspt2 import _h0_action
    from diffmrc.core.mscaspt2.matrix_free import (
        determinant_support, h0_determinant_matrix)
    from diffmrc.core.mscaspt2.ms import averaged_fock, union_function_vectors

    ncore, nact = small_system['ncore'], small_system['nact']
    nmo = small_system['nmo']
    funcs = union_function_vectors(small_system['civs'],
                                   small_system['dets'], ncore, nact, nmo)
    dets = determinant_support(funcs)
    f = averaged_fock(small_system['h1_mo'], small_system['eri_mo'],
                      small_system['civs'], small_system['dets'],
                      ncore + nact)
    H0 = h0_determinant_matrix(f, dets)
    index = {int(d): i for i, d in enumerate(dets)}
    cache = {}
    for dd, i in index.items():
        ref = {index[t]: v for t, v in
               _h0_action({dd: 1.0}, f, nmo, cache).items() if t in index}
        lo, hi = H0.indptr[i], H0.indptr[i + 1]
        new = dict(zip(H0.indices[lo:hi].tolist(), H0.data[lo:hi]))
        assert set(ref) == set(new)
        for j, v in ref.items():
            assert abs(v - new[j]) < 1e-11


def test_projected_coupling_matches_scan(small_system):
    """The enumerated Slater-Condon coupling equals the direct scan."""
    from diffmrc.core.caspt2 import _sc_kernel
    from diffmrc.core.mscaspt2.matrix_free import (
        determinant_support, projected_coupling)
    from diffmrc.core.mscaspt2.ms import (_reference_pairs,
                                          union_function_vectors)

    ncore, nact = small_system['ncore'], small_system['nact']
    nmo = small_system['nmo']
    funcs = union_function_vectors(small_system['civs'],
                                   small_system['dets'], ncore, nact, nmo)
    dets = determinant_support(funcs)
    refs = _reference_pairs(small_system['civs'], small_system['dets'])
    Wd = projected_coupling(small_system['h1_mo'], small_system['eri_mo'],
                            refs, dets)
    h_elem, _ = _sc_kernel(small_system['h1_mo'], small_system['eri_mo'])
    index = {int(d): i for i, d in enumerate(dets)}
    Wd_ref = np.zeros_like(Wd)
    for N, ref in enumerate(refs):
        for dd, ddi in index.items():
            Wd_ref[ddi, N] = sum(c * h_elem(J, dd) for J, c in ref
                                 if (J ^ dd).bit_count() <= 4)
    assert np.allclose(Wd, Wd_ref, atol=1e-12)


def test_matrix_free_matches_dense_coupled(small_system):
    """The matrix-free solve reproduces the dense coupled system."""
    from diffmrc.core.mscaspt2.ms import solve_ms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    dense = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                     small_system['civs'], small_system['dets'], **common)
    mf = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                  small_system['civs'], small_system['dets'],
                  matrix_free=True, matrix_free_rtol=1e-12, **common)
    assert mf['matrix_free'] is True
    assert np.allclose(mf['energies'], dense['energies'], atol=1e-9)
    assert np.allclose(mf['cross'], dense['cross'], atol=1e-9)
    assert np.allclose(mf['overlap'], dense['overlap'], atol=1e-9)


def test_matrix_free_is_rotation_invariant(small_system):
    """The matrix-free coupled spectrum does not depend on the basis."""
    from diffmrc.core.mscaspt2.ms import random_unitary, solve_ms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    civs = small_system['civs']
    Q = random_unitary(civs.shape[0], seed=11)
    base = solve_ms(small_system['h1_mo'], small_system['eri_mo'], civs,
                    small_system['dets'], matrix_free=True,
                    matrix_free_rtol=1e-12, **common)
    rot = solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                   Q.T @ civs, small_system['dets'], matrix_free=True,
                   matrix_free_rtol=1e-12, **common)
    assert np.allclose(base['energies'], rot['energies'], atol=1e-9)


def test_matrix_free_requires_coupling(small_system):
    """``matrix_free`` is only defined on the coupled union path."""
    from diffmrc.core.mscaspt2.ms import solve_ms
    common = dict(ncore=small_system['ncore'], nact=small_system['nact'],
                  nele=small_system['nele'],
                  H_model=small_system['H_model'],
                  det_index=small_system['det_index'])
    with pytest.raises(ValueError):
        solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                 small_system['civs'], small_system['dets'],
                 reference_coupling=False, matrix_free=True, **common)
    with pytest.raises(ValueError):
        solve_ms(small_system['h1_mo'], small_system['eri_mo'],
                 small_system['civs'][:1], small_system['dets'],
                 matrix_free=True, **common)


def test_operator_diagonal_matches_dense(small_system):
    """``diag(V H0 V^T)`` from the determinant diagonal is exact here."""
    from diffmrc.core.mscaspt2.matrix_free import (
        coupled_reference_system_mf, operator_diagonal)
    from diffmrc.core.mscaspt2.ms import averaged_fock

    ncore, nact = small_system['ncore'], small_system['nact']
    f = averaged_fock(small_system['h1_mo'], small_system['eri_mo'],
                      small_system['civs'], small_system['dets'],
                      ncore + nact)
    system = coupled_reference_system_mf(
        small_system['h1_mo'], small_system['eri_mo'],
        small_system['civs'], small_system['dets'],
        ncore=ncore, nact=nact, nmo=small_system['nmo'], fock=f)
    V, H0 = system['V'], system['H0']
    A = (V @ H0 @ V.T).toarray()
    for shift in (0.0, -0.7, 2.5):
        diag, diag_s, diag_a = operator_diagonal(V, system['h0_diag'], shift)
        assert np.allclose(diag, np.diag(A) - shift * diag_s, atol=1e-10)


def test_matrix_free_preconditioning_matches_plain(small_system):
    """The diagonal congruence does not change the converged spectrum."""
    from diffmrc.core.mscaspt2.matrix_free import (
        coupled_reference_system_mf, solve_coupled_amplitudes_mf)
    from diffmrc.core.mscaspt2.ms import averaged_fock, reference_fock_matrix

    civs = small_system['civs']
    ncore, nact = small_system['ncore'], small_system['nact']
    f = averaged_fock(small_system['h1_mo'], small_system['eri_mo'], civs,
                      small_system['dets'], ncore + nact)
    system = coupled_reference_system_mf(
        small_system['h1_mo'], small_system['eri_mo'], civs,
        small_system['dets'], ncore=ncore, nact=nact,
        nmo=small_system['nmo'], fock=f)
    F = reference_fock_matrix(f, civs, small_system['dets'], ncore + nact)
    args = (system['V'], system['H0'], system['W'], F)
    C0, cross0, _, info0 = solve_coupled_amplitudes_mf(
        *args, rtol=1e-12, precondition=False, h0_diag=None)
    C1, cross1, _, info1 = solve_coupled_amplitudes_mf(
        *args, rtol=1e-12, precondition=True, h0_diag=system['h0_diag'])
    assert np.allclose(cross1, cross0, atol=1e-9)
    assert info1['iterations'] <= info0['iterations']
