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
pairs.  The energy checks are compared against the pilot numbers recorded
in ``final_validation.json`` / ``validate_v4.json``.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402

from diffmrc.core import kramers, x2c_df  # noqa: E402

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


def test_hydroxyl_adapted_basis_and_kramers_degeneracy(hydroxyl):
    hf, res, _ = hydroxyl
    assert abs(hf.e_tot - (-75.24618023940573)) < 1e-8
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
