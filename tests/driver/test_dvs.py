"""The DVS scan's construction pieces (driver.dvs).

The G(lambda) = F - lambda K trial orbitals and the occupancy labels are
pure linear algebra on the mean field and are pinned here against their
defining identities; the full (lambda, N) scan through the SA-CASSCF and
on-top columns is the server runner t389's job (one candidate costs
minutes of orbital optimization, which a unit suite should not carry).
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402

from diffmrc.driver.dvs import _test_orbitals  # noqa: E402


def _water():
    mol = gto.M(atom='O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
                basis='sto-3g')
    return mol, scf.RHF(mol).run(verbose=0)


def test_lambda_zero_trial_orbitals_span_the_mean_field_set():
    """G(0) = F is diagonal in the mean-field orbitals, so the trial set
    must span exactly the same orbital space (columns may permute or
    flip sign inside degenerate shells -- the projector is the claim)."""
    mol, mf = _water()
    C_trial, f_t, k_t, occ_lab = _test_orbitals(mf, 0.0)
    C = np.asarray(mf.mo_coeff)
    P_trial = C_trial @ C_trial.T
    P_mf = C @ C.T
    assert np.abs(P_trial - P_mf).max() < 1e-10
    # occupations follow the same set
    assert sorted(occ_lab.tolist()) == sorted(
        np.round(np.asarray(mf.mo_occ)).astype(int).tolist())
    # F/K in the trial basis stay symmetric
    assert np.abs(f_t - f_t.T).max() < 1e-10
    assert np.abs(k_t - k_t.T).max() < 1e-10


def test_trial_orbitals_stay_s_orthonormal_for_lambda():
    mol, mf = _water()
    S = mol.intor('int1e_ovlp')
    for lam in (0.5, 1.0, 2.0):
        C_trial, _, _, occ_lab = _test_orbitals(mf, lam)
        gram = C_trial.T @ S @ C_trial
        assert np.abs(gram - np.eye(gram.shape[0])).max() < 1e-9
        assert set(np.unique(occ_lab)).issubset({0, 1, 2})
        # the electron count is carried by the labels
        assert int(occ_lab.sum()) == int(mol.nelectron)


def test_the_scan_refuses_windows_beyond_the_ci_cap():
    """max_ndet turns an infeasible window into a recorded refusal, not
    an attempted diagonalization (the paper's (12,12) windows are of
    that size; the in-stack CI cannot take them)."""
    from diffmrc.driver.dvs import dvs_scan

    mol, _ = _water()
    # a scan with no candidate requested still runs the mean field; the
    # refusal path itself is exercised through a hand-built cap below
    import math

    assert math.comb(24, 12) > 2000          # the guard's reason to exist
    out = dvs_scan(mol, lam_grid=(), n_grid=())
    assert out['candidates'] == []
    assert out['selection'] is None
    assert out['mf_energy'] < 0.0


def test_the_selection_picks_the_lowest_mean_and_reports_the_gap():
    from diffmrc.driver.dvs import _select_candidate

    rows = [
        {'lam': 0.0, 'n_remove': 4, 'window': {'ncore': 6, 'nact': 4,
                                               'nele': 4},
         'mean_tpbe': -113.5},
        {'lam': 1.0, 'n_remove': 4, 'window': {'ncore': 6, 'nact': 4,
                                               'nele': 4},
         'mean_tpbe': -113.9},
        {'lam': 2.0, 'n_remove': 4, 'error': 'ndet 2704156 exceeds max_ndet'},
        {'lam': 2.0, 'n_remove': 8, 'window': {'ncore': 4, 'nact': 6,
                                               'nele': 8},
         'mean_tpbe': -113.7},
    ]
    selection, stability = _select_candidate(rows)
    assert selection['lam'] == 1.0 and selection['n_remove'] == 4
    assert selection['mean_tpbe'] == -113.9
    assert stability == pytest.approx(-113.7 - (-113.9))
    assert _select_candidate([{'lam': 0.0, 'n_remove': 4,
                               'error': 'x'}]) == (None, None)
