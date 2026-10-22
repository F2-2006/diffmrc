"""AVAS selection with target-shell rules and space diagnostics.

The construction tests run on hand-built labels and identity matrices,
where the overlap eigenvalues are exact 0/1.  The molecule tests need
PySCF and are skipped when it is absent: they pin the textbook H2O
CAS(8,6) and the Valence-F rule on the three ECP f-block oxides used by
the t307 smoke (CeO, YbO, ThO).
"""
import numpy as np
import pytest

from diffmrc.policy import avas
from diffmrc.policy.projection import avas_selection as projection_avas


def _ce_labels():
    labels = [f'0 Ce 4f{m}' for m in range(7)]
    labels += [f'0 Ce 5f{m}' for m in range(7)]
    labels += ['1 O 1s', '1 O 2s', '1 O 2px', '1 O 2py', '1 O 2pz']
    return labels


def test_valence_f_takes_the_lowest_f_shell_per_ln_an_atom():
    sub = avas.target_ao_indices(None, labels=_ce_labels(),
                                 symbols=['Ce', 'O'], rule='valence-f')
    assert sub.indices.tolist() == list(range(7))
    assert sub.rule == 'valence-f'
    assert any('f shell' in line and '7 AO' in line for line in sub.evidence)
    assert sub.warnings == []


def test_valence_f_warns_when_the_shell_is_incomplete():
    labels = [f'0 Ce 4f{m}' for m in range(6)] + ['1 O 2s']
    sub = avas.target_ao_indices(None, labels=labels, symbols=['Ce', 'O'],
                                 rule='valence-f')
    assert sub.indices.size == 6
    assert any('expected 7 f orbitals' in w for w in sub.warnings)


def test_double_d_takes_the_two_lowest_d_shells():
    labels = [f'0 Fe 3d{m}' for m in range(5)]
    labels += [f'0 Fe 4d{m}' for m in range(5)]
    labels += ['1 O 2p' + a for a in 'xyz']
    sub = avas.target_ao_indices(None, labels=labels,
                                 symbols=['Fe', 'O'], rule='double-d')
    assert sub.indices.tolist() == list(range(10))
    assert sub.warnings == []


def test_custom_shells_are_per_element():
    labels = ['0 O 1s', '0 O 2s', '0 O 2px', '0 O 2py', '0 O 2pz',
              '1 H 1s', '2 H 1s']
    sub = avas.target_ao_indices(
        None, labels=labels, symbols=['O', 'H', 'H'],
        shells={'O': ['2s', '2p'], 'H': ['1s']})
    assert sub.indices.tolist() == [1, 2, 3, 4, 5, 6]


def test_a_rule_that_matches_nothing_refuses():
    with pytest.raises(ValueError):
        avas.target_ao_indices(None, labels=['0 O 1s', '1 H 1s'],
                               symbols=['O', 'H'], rule='valence-f')


def test_explicit_target_and_rule_together_are_rejected():
    with pytest.raises(ValueError):
        avas.avas_selection(None, np.eye(2), [0], occ=np.array([2.0, 0.0]),
                            S=np.eye(2), rule='valence-f')


def test_svd_selects_the_slots_that_overlap_the_target():
    S = np.eye(4)
    C = np.eye(4)
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    active, diag = avas.avas_selection(None, C, [0, 3], occ=occ, S=S)
    assert active == [0, 2]
    assert diag.nact == 2 and diag.nele == 2 and diag.ncore == 1
    assert diag.target_aos.tolist() == [0, 3]
    assert np.allclose(diag.overlap_eigenvalues['doubly_occupied'], [1, 0])
    assert np.allclose(diag.overlap_eigenvalues['empty'], [1, 0])
    assert np.allclose(diag.C_rotated[:, 2], [0, 0, 0, 1])


def test_nact_nele_override_the_threshold():
    S = np.eye(4)
    C = np.eye(4)
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    by_threshold, _ = avas.avas_selection(None, C, [0, 2], occ=occ, S=S)
    assert by_threshold == [0, 2]
    active, diag = avas.avas_selection(None, C, [0, 2], nact=1, nele=2,
                                       occ=occ, S=S)
    assert active == [0]
    assert diag.nact == 1 and diag.nele == 2 and diag.ncore == 1
    assert diag.overlap_eigenvalues['empty'][0] > 0.05


def test_inconsistent_nact_nele_raises():
    S = np.eye(4)
    C = np.eye(4)
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        avas.avas_selection(None, C, [0], nact=2, occ=occ, S=S)
    with pytest.raises(ValueError):
        avas.avas_selection(None, C, [0], nact=4, nele=2, occ=occ, S=S)


def test_singly_occupied_orbitals_are_always_active():
    S = np.eye(4)
    C = np.eye(4)
    occ = np.array([2.0, 1.0, 0.0, 0.0])
    active, diag = avas.avas_selection(None, C, [0], occ=occ, S=S)
    assert active == [0, 1]
    assert diag.nele == 3
    assert diag.blocks['singly_occupied'] == [1]
    active, diag = avas.avas_selection(None, C, [0], nact=2, nele=3,
                                       occ=occ, S=S)
    assert active == [0, 1]
    with pytest.raises(ValueError):
        avas.avas_selection(None, C, [0], nact=2, nele=2, occ=occ, S=S)


def test_fractional_occupations_are_refused():
    with pytest.raises(ValueError):
        avas.avas_selection(None, np.eye(3), [0],
                            occ=np.array([2.0, 1.5, 0.0]), S=np.eye(3))


def test_occupation_diagnostics_flags_both_ends():
    report = avas.occupation_diagnostics([0.01, 0.5, 1.99])
    assert report.weak == [0] and report.strong == [2]
    assert report.ok is False
    passing = avas.occupation_diagnostics([0.1, 1.0, 1.9])
    assert passing.ok is True and passing.weak == [] and passing.strong == []


def test_gap_diagnostics_warns_below_the_line():
    e = np.array([-1.0, -0.5, -0.4, -0.1, 0.0, 0.3])
    report = avas.orbital_gap_diagnostics(e, core=[0, 1], active=[2, 3],
                                          external=[4, 5], gap_min=0.2)
    assert report.act_int == pytest.approx(0.1)
    assert report.ext_act == pytest.approx(0.1)
    assert report.ok is False and len(report.warnings) == 2
    passing = avas.orbital_gap_diagnostics(e, core=[0, 1], active=[2],
                                           external=[4, 5], gap_min=0.05)
    assert passing.ok is True


def test_gap_indices_must_be_disjoint():
    e = np.zeros(3)
    with pytest.raises(ValueError):
        avas.orbital_gap_diagnostics(e, core=[0], active=[0], external=[2])


def test_cross_check_with_the_projection_construction():
    S = np.eye(4)
    C = np.eye(4)
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    reference = projection_avas(C, S, [0, 3], occ)
    active, diag = avas.avas_selection(None, C, [0, 3], occ=occ, S=S)
    assert active == reference.active
    for a in active:
        assert np.allclose(np.abs(diag.C_rotated[:, a]),
                           np.abs(reference.C_rotated[:, a]))


# ---------------------------------------------------------------- PySCF --


def _pyscf():
    """Import PySCF inside the test so the numpy cases run without it."""
    pytest.importorskip('pyscf')
    from pyscf import gto, scf
    return gto, scf


def _h2o(bond=0.9572):
    gto, scf = _pyscf()
    mol = gto.M(atom=f'O 0 0 0; H 0.7570 {bond} 0; H -0.7570 {bond} 0',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def test_h2o_sto3g_recovers_the_known_cas_8_6():
    mol, mf = _h2o()
    active, diag = avas.avas_selection(
        mol, mf.mo_coeff, shells={'O': ['2s', '2p'], 'H': ['1s']},
        occ=mf.mo_occ, mo_energy=mf.mo_energy)
    assert diag.target_aos.size == 6
    assert diag.nact == 6 and diag.nele == 8 and diag.ncore == 1
    assert active == [0, 1, 2, 3, 5, 6]
    lam_occ = diag.overlap_eigenvalues['doubly_occupied']
    assert np.all(lam_occ[:4] > 0.999)
    assert lam_occ[4] < 1e-6
    assert np.all(diag.overlap_eigenvalues['empty'] > 0.999)


F_BLOCK = [
    ('CeO', [('Ce', (0.0, 0.0, 0.0)), ('O', (0.0, 0.0, 3.44))], 'Ce'),
    ('YbO', [('Yb', (0.0, 0.0, 0.0)), ('O', (0.0, 0.0, 3.44))], 'Yb'),
    ('ThO', [('Th', (0.0, 0.0, 0.0)), ('O', (0.0, 0.0, 3.47))], 'Th'),
]


@pytest.mark.parametrize('name,atoms,metal', F_BLOCK)
def test_f_block_valence_f_recovers_the_f_subspace(name, atoms, metal):
    from diffmrc.analysis.composition import (ao_mask,
                                              overlap_weighted_population)
    gto, scf = _pyscf()
    mol = gto.M(atom=atoms, unit='bohr',
                basis={metal: 'stuttgart_rsc', 'O': '6-31g'},
                ecp={metal: 'stuttgart_rsc'}, verbose=0)
    mf = scf.RHF(mol).run()
    active, diag = avas.avas_selection(mol, mf.mo_coeff, rule='valence-f',
                                       occ=mf.mo_occ,
                                       mo_energy=mf.mo_energy)
    assert diag.target_aos.size == 7
    assert diag.nact == 7
    labels = mol.ao_labels()
    mask = ao_mask(labels, metal, 'f')
    population = overlap_weighted_population(diag.C_rotated, mf.get_ovlp(),
                                             mask)
    selected = np.asarray(active)
    other = np.setdiff1d(np.arange(len(population)), selected)
    assert np.all(population[selected] > 0.9)
    assert np.all(population[other] < 0.1)
    # Electron count follows the mean-field occupations of the selected
    # orbitals: the f shell is empty in CeO and ThO and filled in YbO.
    occ_sel = np.asarray(mf.mo_occ)[selected]
    n_double = int(np.sum(occ_sel > 1.5))
    n_single = int(np.sum((occ_sel > 0.5) & (occ_sel < 1.5)))
    assert diag.nele == 2 * n_double + n_single


def test_entropy_spectrum_inside_the_avas_space():
    """The entropy selector inside the AVAS space: same question, checked.

    On stretched H2O the valence space carries real static correlation,
    so the spectrum is informative.  The test records how many orbitals
    the plateau selector would keep; the AVAS space is the container, and
    the selector is allowed to keep fewer (that difference is the point
    of running the spectrum after the construction).
    """
    pytest.importorskip('pyscf')
    from pyscf import fci, mcscf

    from diffmrc.analysis.ci import dets_and_vector_from_fci, entropy_from_ci
    from diffmrc.policy.entropy_selection import plateau_selection

    mol, mf = _h2o(bond=2.5)
    active, diag = avas.avas_selection(
        mol, mf.mo_coeff, shells={'O': ['2s', '2p'], 'H': ['1s']},
        occ=mf.mo_occ, mo_energy=mf.mo_energy)
    core = [i for i in range(mol.nao_nr()) if i not in active]
    core = [i for i in core if mf.mo_occ[i] > 1.5]
    external = [i for i in range(mol.nao_nr())
                if i not in active and mf.mo_occ[i] < 0.5]
    mo = np.hstack([diag.C_rotated[:, core], diag.C_rotated[:, active],
                    diag.C_rotated[:, external]])
    cas = mcscf.CASCI(mf, diag.nact, diag.nele)
    cas.mo_coeff = mo
    cas.kernel()
    ci = np.asarray(cas.ci)
    alpha = fci.cistring.make_strings(range(diag.nact),
                                      (diag.nele + mol.spin) // 2)
    beta = fci.cistring.make_strings(range(diag.nact),
                                     (diag.nele - mol.spin) // 2)
    dets, c = dets_and_vector_from_fci(ci, alpha, beta, diag.ncore,
                                       diag.nact)
    s1 = entropy_from_ci(c, dets, ncore=diag.ncore, nact=diag.nact)
    assert s1.shape == (diag.nact,)
    assert np.isfinite(s1).all() and s1.max() > 0.14
    selection = plateau_selection(s1)
    assert selection.method == 'plateau'
    assert set(selection.active) <= set(range(diag.nact))
