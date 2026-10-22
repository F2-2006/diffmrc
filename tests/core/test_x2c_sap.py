"""Value-layer contracts of the SAP-X2C module.

The tests split into three groups:

1. Table and real-space potential conventions: the BSE coefficients must sum
   to -Z (they multiply unit-integral Gaussians with the sign of Eq. 16), and
   the total SAP potential must decay superpolynomially for a neutral atom.
2. Matrix elements against independent references: the scalar V^e against a
   DFT-grid quadrature, and both V^e and W^e against the point-charge 1e
   integrals in the tight-Gaussian limit (alpha -> infinity).  The latter is
   the sharpest test of the normalization factor and of libcint's spinor
   operator convention.
3. The X2C transform: exact reproduction of PySCF's 1eX2C when the SAP table
   is empty, and the H2O/6-31G energy ordering 1eX2C -> SAP-X2C -> 4c DHF.

Run with PySCF available:

    PYTHONPATH=src pytest tests/core/test_x2c_sap.py -q
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf  # noqa: E402
from pyscf.x2c import sfx2c1e, x2c  # noqa: E402

from diffmrc.core import x2c_sap as xs  # noqa: E402

GEOM_H2O = [("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -1.430523, 1.109269)),
            ("H", (0.0, 1.430523, 1.109269))]


def _mol_h2o():
    return gto.M(atom=GEOM_H2O, unit="Bohr", basis="6-31g", verbose=0)


def _mol_o_atom():
    return gto.M(atom=[("O", (0.0, 0.0, 0.0))], unit="Bohr", basis="6-31g",
                 spin=2, verbose=0)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# 1. table and real-space potential
# ---------------------------------------------------------------------------
def test_table_coefficients_sum_to_minus_z():
    table = xs.load_sap_table()
    for z in (1, 6, 8, 54, 70, 86, 118):
        coef = table[z]["coef"]
        assert abs(coef.sum() + z) < 1e-8, (z, coef.sum())
        assert np.all(table[z]["alpha"] > 0.0)


def test_sap_potential_decays_and_hits_the_nucleus():
    mol = gto.M(atom=[("Xe", (0.0, 0.0, 0.0))], unit="Bohr",
                basis="cc-pvtz-dk", verbose=0)
    far = np.array([[25.0, 0.0, 0.0], [0.0, -40.0, 0.0]])
    v_far = xs.sap_potential(mol, far)
    assert np.max(np.abs(v_far)) < 1e-8

    near = np.array([[1e-3, 0.0, 0.0]])
    v_near = xs.sap_potential(mol, near)
    bare = -54.0 / 1e-3
    # inside the 1s shell the bare Coulomb term dominates; at 1e-3 bohr the
    # residual screening is below 2% of the bare value
    assert abs(v_near[0] / bare - 1.0) < 0.02


def test_scalar_screening_matrix_matches_grid_quadrature():
    mol = _mol_h2o()
    ve = xs.sap_potential_matrix(mol, spinor=False)

    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = 4
    grids.build()
    ao = mol.eval_gto("GTOval_sph", grids.coords)
    v_screen = xs.sap_screening_potential(mol, grids.coords)
    v_num = np.einsum("g,gi,gj->ij", grids.weights * v_screen, ao, ao)
    assert _rel(ve, v_num) < 1e-4


def test_sap_matrix_is_the_sum_of_bare_and_screening():
    mol = _mol_h2o()
    ve = xs.sap_potential_matrix(mol, spinor=False)
    v_nuc = mol.intor("int1e_nuc")
    v_total = v_nuc + ve

    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = 4
    grids.build()
    ao = mol.eval_gto("GTOval_sph", grids.coords)
    v_sap = xs.sap_potential(mol, grids.coords)
    v_num = np.einsum("g,gi,gj->ij", grids.weights * v_sap, ao, ao)
    assert _rel(v_total, v_num) < 1e-4


# ---------------------------------------------------------------------------
# 2. tight-Gaussian limit: normalization and spinor operator convention
# ---------------------------------------------------------------------------
def test_tight_gaussian_reproduces_point_charge_integrals():
    mol = _mol_o_atom()
    alpha = 1e8
    aux = gto.M(atom=[("O", (0.0, 0.0, 0.0))], unit="Bohr",
                basis={"O": [[0, [alpha, 1.0]]]}, verbose=0)
    from pyscf.df.incore import aux_e2

    f = (alpha / (2.0 * np.pi)) ** 0.75
    v3 = aux_e2(mol, aux, intor="int3c2e")[:, :, 0] * f
    w3 = -aux_e2(mol, aux, intor="int3c2e_spsp1_spinor")[:, :, 0] * f
    v1 = -mol.intor("int1e_nuc") / 8.0   # unit attractive point charge
    w1 = mol.intor("int1e_spnucsp_spinor") / 8.0

    assert _rel(v3, v1) < 1e-5
    assert _rel(w3, w1) < 1e-5


def test_full_spinor_trace_equals_twice_the_scalar_pnucp_trace():
    # Basis-invariant check that does not rely on the j-adapted block layout.
    mol = gto.M(atom=[("He", (0.0, 0.0, 0.0)), ("H", (2.0, 0.0, 0.0))],
                unit="Bohr", spin=1,
                basis={"He": [[0, [0.1, 1.0]], [1, [0.2, 1.0]]],
                       "H": [[0, [0.3, 1.0]]]}, verbose=0)
    w1 = mol.intor("int1e_pnucp")
    W1 = mol.intor("int1e_spnucsp_spinor")
    assert abs(np.trace(W1) - 2.0 * np.trace(w1)) < 1e-10 * abs(np.trace(w1))


# ---------------------------------------------------------------------------
# 3. X2C transformation
# ---------------------------------------------------------------------------
def test_one_electron_x2c_reproduces_pyscf():
    mol = _mol_h2o()
    h_s = xs.x2c_hcore_1e(mol, spinor=False)
    ref_s = sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)
    assert np.abs(h_s - ref_s).max() < 1e-10

    h_2 = xs.x2c_hcore_1e(mol, spinor=True)
    ref_2 = x2c.SpinorX2CHelper(mol).get_hcore(mol)
    assert np.abs(h_2 - ref_2).max() < 1e-10


def test_sap_x2c_hcore_is_hermitian_and_changes_the_1e_hamiltonian():
    mol = _mol_h2o()
    h1 = xs.x2c_hcore_1e(mol, spinor=True)
    hs = xs.x2c_sap_hcore(mol, spinor=True)
    assert np.abs(hs - hs.conj().T).max() < 1e-10
    change = np.abs(hs - h1).max()
    assert 1e-5 < change < 1e-1


def test_scalar_sap_route_raises():
    mol = _mol_h2o()
    with pytest.raises(NotImplementedError):
        xs.x2c_sap_hcore(mol, spinor=False)


def _run_2c(mol, hcore):
    mf = x2c.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 300
    mf.get_hcore = lambda *args: hcore
    return float(mf.kernel()), mf


def test_h2o_energy_moves_from_1ex2c_toward_dhf():
    mol = _mol_h2o()
    h1 = x2c.SpinorX2CHelper(mol).get_hcore(mol)
    hs = xs.x2c_sap_hcore(mol, spinor=True)

    e_1e, mf_1e = _run_2c(mol, h1)
    e_sap, mf_sap = _run_2c(mol, hs)

    mf_dhf = scf.DHF(mol)
    mf_dhf.verbose = 0
    mf_dhf.conv_tol = 1e-12
    e_dhf = float(mf_dhf.kernel())

    err_1e = abs(e_1e - e_dhf)
    err_sap = abs(e_sap - e_dhf)
    assert err_sap < err_1e
    assert err_sap < 1e-2
    # occupied spinor energies improve as well
    e1 = np.asarray(mf_1e.mo_energy)[np.asarray(mf_1e.mo_occ) > 0]
    es = np.asarray(mf_sap.mo_energy)[np.asarray(mf_sap.mo_occ) > 0]
    ed = np.asarray(mf_dhf.mo_energy)[np.asarray(mf_dhf.mo_occ) > 0]
    assert np.max(np.abs(es - ed)) < np.max(np.abs(e1 - ed))
