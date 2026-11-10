"""Contracts of the Kramers-adapted physical X2C complex CI.

The tests pin four things:

* the canonical construction ``e_{2k+1} = m e_{2k}^*`` produces an
  orthonormal basis whose one-particle time-reversal matrix is the
  symplectic form ``J`` (1e-12 orthonormality, 1e-10 block deviation);
* the determinant Hamiltonian built on that basis, with the core and
  active spaces cut at complete pairs, commutes with the many-body time
  reversal (1e-10) and satisfies ``T T^* = (-1)^N``;
* for the odd-electron OH system every CI level is at least doubly
  degenerate (Kramers theorem) with a within-pair gap below 1e-10;
* the unadapted energy-ordered ``argmax`` pairing is a negative control:
  it is not Kramers closed, the block deviation is of order 1e-2, the CI
  loses time-reversal invariance and a spurious Kramers gap appears.

The validation systems and the frozen-core sizes are the ones of the D
spinor pilot: H2O/6-31G and OH/6-31G, two core pairs and four active
pairs.  The H2O energy is compared against the pilot number recorded in
``final_validation.json``.  OH has several GHF stationary points: the
default core-Hamiltonian Aufbau guess can fall into a quartet-like branch
(S^2 ~ 3.75, E ~ -75.312 Eh) while the pilot doublet-like branch is at
E ~ -75.246 Eh.  The Kramers checks hold for either branch, so the OH test
pins the structural checks and only brackets the HF energy.

The symmetry tests pin the two identities of a complex chemist integral
and show that symmetrizing with the true pair relation is a no-op, while
projecting with the false real-orbital relation is not a symmetry of the
complex array and corrupts the energy.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402

from diffmrc.core import kramers, spinor_ci, x2c_df  # noqa: E402

AUX_BASIS = "def2-universal-jkfit"

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
HYDROXYL = "O 0 0 0; H 0 0 1.8"


def _mol(atom, spin):
    return gto.M(atom=atom, basis="6-31g", unit="Angstrom", spin=spin,
                 verbose=0)


@pytest.fixture(scope="module")
def water():
    mol = _mol(WATER, 0)
    hf = x2c_df.df_x2c_hf(mol, auxbasis=AUX_BASIS)
    adapted = kramers.x2c_spinor_ci(mol, 2, 4, hf=hf, adapt=True)
    return hf, adapted


@pytest.fixture(scope="module")
def hydroxyl():
    mol = _mol(HYDROXYL, 1)
    hf = x2c_df.df_x2c_hf(mol, auxbasis=AUX_BASIS)
    adapted = kramers.x2c_spinor_ci(mol, 2, 4, hf=hf, adapt=True)
    unadapted = kramers.x2c_spinor_ci(mol, 2, 4, hf=hf, adapt=False)
    return hf, adapted, unadapted


def test_synthetic_rotation_recovers_symplectic_form():
    """The construction is basis independent for an exact Kramers matrix."""
    rng = np.random.default_rng(11)
    n = 8
    target = kramers.kramers_target(n)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rot, _ = np.linalg.qr(a)
    m = rot.conj().T @ target @ rot.conj()
    w, pair_energies, seeds = kramers.kramers_adapt(m, np.arange(n))
    adapted = w.conj().T @ m @ w.conj()
    assert np.abs(w.conj().T @ w - np.eye(n)).max() < 1e-12
    assert np.abs(adapted - target).max() < 1e-12
    assert pair_energies.shape == (n // 2,)
    assert seeds.shape == (n // 2,)


def test_slice_requires_complete_pairs():
    n = 6
    h1 = np.zeros((n, n), dtype=complex)
    eri = np.zeros((n, n, n, n), dtype=complex)
    m = kramers.kramers_target(n)
    for bad_core, bad_act in ((1, 2), (2, 3), (2, 6)):
        with pytest.raises(ValueError):
            kramers.slice_active_space(h1, eri, m, bad_core, bad_act, 2)
    with pytest.raises(ValueError):
        kramers.slice_active_space(h1, eri, m, 2, 2, 3)


def test_water_adapted_basis_is_orthonormal_and_closed(water):
    hf, res = water
    assert abs(hf.e_tot - (-76.03269315736874)) < 1e-8
    assert res.basis.adapted
    assert res.basis.orthonormality < 1e-12
    assert res.basis.block_deviation < 1e-10
    assert res.basis.closed


def test_water_active_space_is_time_reversal_closed(water):
    _, res = water
    checks = res.checks
    assert checks["ndet"] == 28
    assert checks["ncore_so"] == 4
    assert checks["nact_so"] == 8
    assert checks["nele_act"] == 6
    assert checks["t_closure"] < 1e-10
    assert checks["t_square"] < 1e-12
    # Even electron count: no Kramers degeneracy is required.
    assert checks["kramers_degenerate"] is None


def _mo_eri(coeff, e_ao):
    """Four-index transform with the bra index conjugated."""
    return np.einsum("ma,nb,oc,pd,mnop->abcd", coeff.conj(), coeff,
                     coeff.conj(), coeff, e_ao, optimize=True)


def test_int2e_spinor_obeys_complex_chemist_symmetry(water):
    """The true pair symmetry is exact; the old 0.62 was a false relation.

    A complex chemist integral obeys ``E[p,q,r,s] = E[r,s,p,q]`` and
    ``E[p,q,r,s] = conj(E[q,p,s,r])``.  The real-orbital relations
    ``E[p,q,r,s] = conj(E[q,p,r,s])`` and the unconjugated eightfold
    exchange are not symmetries of a complex array; the earlier open
    problem used one of them as a pair-symmetry check.  The residual of
    that false check is 0.477 on H2O/6-31G.
    """
    _, res = water
    e = np.asarray(res.mol.intor("int2e_spinor"))
    assert np.abs(e - e.transpose(2, 3, 0, 1)).max() < 1e-12
    assert np.abs(e - e.transpose(1, 0, 3, 2).conj()).max() < 1e-12
    assert np.abs(e - e.transpose(1, 0, 2, 3).conj()).max() > 1e-2
    assert np.abs(e - e.transpose(1, 0, 3, 2)).max() > 1e-2


def test_pair_symmetrization_is_a_no_op_for_the_ci(water):
    """Averaging with ``(kl|ij)`` changes neither the array nor the spectrum."""
    hf, res = water
    mol = res.mol
    coeff = res.basis.coeff
    e = np.asarray(mol.intor("int2e_spinor"))
    e_sym = 0.5 * (e + e.transpose(2, 3, 0, 1))
    assert np.abs(e_sym - e).max() < 1e-14
    h1 = coeff.conj().T @ np.asarray(hf.hcore) @ coeff
    ncore, nact = 4, 8
    nele = int(mol.nelectron) - ncore
    h_ref, _, _ = spinor_ci.spinor_ci_matrix(
        h1, _mo_eri(coeff, e), ncore, nact, nele)
    h_sym, _, _ = spinor_ci.spinor_ci_matrix(
        h1, _mo_eri(coeff, e_sym), ncore, nact, nele)
    assert np.abs(h_ref - h_sym).max() < 1e-11
    w_ref = np.linalg.eigvalsh(h_ref)
    w_sym = np.linalg.eigvalsh(h_sym)
    assert np.abs(w_ref - w_sym).max() < 1e-10


def test_false_real_orbital_projection_is_not_a_symmetrization(water):
    """The unconjugated eightfold exchange is not a complex-array symmetry.

    Projecting on it shifts the H2O/6-31G frozen-core CI energy by about
    -2.7 Eh, so it must not be used as a repair.
    """
    hf, res = water
    mol = res.mol
    coeff = res.basis.coeff
    e = np.asarray(mol.intor("int2e_spinor"))
    e_bad = 0.5 * (e + e.transpose(1, 0, 3, 2))
    assert np.abs(e_bad - e).max() > 1e-2
    h1 = coeff.conj().T @ np.asarray(hf.hcore) @ coeff
    ncore, nact = 4, 8
    nele = int(mol.nelectron) - ncore
    w_ref = np.linalg.eigvalsh(spinor_ci.spinor_ci_matrix(
        h1, _mo_eri(coeff, e), ncore, nact, nele)[0])
    w_bad = np.linalg.eigvalsh(spinor_ci.spinor_ci_matrix(
        h1, _mo_eri(coeff, e_bad), ncore, nact, nele)[0])
    assert w_bad[0] - w_ref[0] < -1.0


def test_hydroxyl_adapted_basis_and_kramers_degeneracy(hydroxyl):
    hf, res, _ = hydroxyl
    # OH has multiple GHF stationary points.  The default core-Hamiltonian
    # Aufbau guess can fall into a quartet-like branch (S^2 ~ 3.75,
    # E ~ -75.312 Eh) on this LAPACK path, while the pilot doublet-like
    # branch is at -75.246 Eh.  The Kramers checks are independent of the
    # branch, so only convergence and a loose energy bracket are pinned.
    assert hf.converged
    assert hf.e_tot < -75.2
    checks = res.checks
    assert checks["ndet"] == 56
    assert checks["nele_act"] == 5
    assert res.basis.orthonormality < 1e-12
    assert res.basis.block_deviation < 1e-10
    assert checks["t_closure"] < 1e-10
    assert checks["t_square"] < 1e-12
    assert checks["kramers_degenerate"] is True
    assert checks["max_within_pair_gap"] < 1e-10


def test_hydroxyl_energies_are_pairwise_degenerate(hydroxyl):
    _, res, _ = hydroxyl
    w = np.linalg.eigvalsh(res.hamiltonian)
    # The lowest root is returned together with its Kramers partner.
    assert abs(w[0] - w[1]) < 1e-10


def test_unadapted_hydroxyl_breaks_time_reversal(hydroxyl):
    """Negative control: the pre-adaptation pairing is not T closed."""
    _, _, res = hydroxyl
    checks = res.checks
    assert not res.basis.adapted
    assert not res.basis.closed
    assert res.basis.block_deviation > 1e-2
    assert checks["t_closure"] > 1e-1
    assert checks["max_within_pair_gap"] > 1e-3
