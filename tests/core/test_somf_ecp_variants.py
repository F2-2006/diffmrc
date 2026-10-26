"""Contracts of the ECP SOMF variant family.

The module states that an effective core potential does not fix the split
between its one- and two-electron spin-orbit content.  The variants in
``somf_ecp.somf_variants`` are the materialization of that freedom:

* ``ecp`` is the ECP spin-orbit operator alone;
* ``valence`` adds the full valence ``J - 3/2 K`` mean field;
* ``one_center`` truncates the mean-field kernel and density to the SO-ECP
  centers;
* ``atomic_subtracted`` removes the on-site block of the density from the
  mean field;
* ``point_charge`` replaces ``ECPso`` by the effective-charge nuclear
  operator.

The tests pin the algebraic identities that connect the variants, so a
future edit cannot silently change the definitions:

* all variants are real and antisymmetric for a symmetric density;
* on a single atom the one-center truncation is exact (all AOs are on the
  one center) and the on-site subtraction removes the whole mean field,
  so ``one_center == valence`` and ``atomic_subtracted == ecp``;
* the on-site subtraction is the linearity identity
  ``h2[P - P_on] = h2[P] - h2[P_on]``;
* the point-charge variant contains no ``ECPso`` matrix element.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402

from diffmrc.core import somf as dsomf  # noqa: E402
from diffmrc.core import somf_ecp as decp  # noqa: E402


def mabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def physical_density(mol):
    """A symmetric density with ``trace(P S) = mol.nelectron``."""
    s = mol.intor("int1e_ovlp")
    p = (mol.nelectron / mol.nao_nr()) * np.linalg.inv(s)
    return 0.5 * (p + p.T)


@pytest.fixture(scope="module")
def gef():
    mol = gto.M(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
                basis={"Ge": "lanl2dz", "F": "6-31g"},
                ecp={"Ge": "ecpds28mdfso"}, spin=1, unit="Bohr",
                verbose=0)
    return mol, physical_density(mol)


@pytest.fixture(scope="module")
def ge_atom():
    mol = gto.M(atom=[("Ge", (0.0, 0.0, 0.0))],
                basis={"Ge": "lanl2dz"},
                ecp={"Ge": "ecpds28mdfso"}, spin=2, unit="Bohr",
                verbose=0)
    return mol, physical_density(mol)


@pytest.fixture(scope="module")
def gef_variants(gef):
    mol, p = gef
    return mol, p, decp.somf_variants(mol, p, contract="explicit")


def test_variant_names(gef_variants):
    _, _, v = gef_variants
    assert set(v) == {
        "ecp", "nuclear_non_ecp", "breit_pauli", "valence",
        "one_center", "atomic_subtracted", "point_charge", "double",
    }


def test_all_variants_are_antisymmetric(gef_variants):
    mol, _, v = gef_variants
    nao = mol.nao_nr()
    for name, h in v.items():
        assert h.shape == (3, nao, nao), name
        assert not np.iscomplexobj(h), name
        assert mabs(h + h.transpose(0, 2, 1)) < 1e-12, name


def test_ecp_variant_is_minus_ecpso(gef):
    mol, p = gef
    v = decp.somf_variants(mol, p, contract="explicit")
    assert mabs(v["ecp"] + decp.ecp_soc(mol)) < 1e-13


def test_point_charge_contains_no_ecpso(gef):
    mol, p = gef
    v = decp.somf_variants(mol, p, contract="explicit")
    nuc_all = decp.nuclear_soc(mol, range(mol.natm))
    h2 = dsomf.somf_two_electron(mol, p, contract="explicit")
    ref = decp.SOMF_ALPHA2_HALF * (nuc_all + h2)
    assert mabs(v["point_charge"] - ref) < 1e-12


def test_one_center_equals_full_on_a_single_atom(ge_atom):
    mol, p = ge_atom
    v = decp.somf_variants(mol, p, contract="explicit")
    assert mabs(v["one_center"] - v["valence"]) < 1e-12


def test_onsite_subtraction_removes_the_atomic_mean_field(ge_atom):
    mol, p = ge_atom
    v = decp.somf_variants(mol, p, contract="explicit")
    assert mabs(v["atomic_subtracted"] - v["ecp"]) < 1e-12


def test_onsite_subtraction_is_linear(gef):
    mol, p = gef
    so_mask = decp._center_ao_mask(mol, decp.so_ecp_centers(mol))
    p_on = p * np.outer(so_mask, so_mask)
    h2_full = dsomf.somf_two_electron(mol, p, contract="explicit")
    h2_on = dsomf.somf_two_electron(mol, p_on, contract="explicit")
    h2_eff = dsomf.somf_two_electron(mol, p - p_on, contract="explicit")
    assert mabs(h2_eff - (h2_full - h2_on)) < 1e-12


def test_one_center_truncation_is_not_trivial_on_a_molecule(gef_variants):
    _, _, v = gef_variants
    diff = mabs(v["one_center"] - v["valence"])
    scale = mabs(v["valence"])
    assert diff > 1e-3 * scale


def test_atomic_subtracted_differs_from_ecp_on_a_molecule(gef_variants):
    _, _, v = gef_variants
    diff = mabs(v["atomic_subtracted"] - v["ecp"])
    scale = mabs(v["ecp"])
    assert diff > 1e-3 * scale


def test_density_response_matches_value_contraction(gef):
    """M is the derivative of tr(D h2[P]) and is not h2[D]."""
    mol, p = gef
    rng = np.random.default_rng(11)
    # A non-symmetric direction keeps the contraction away from the
    # symmetric-density cancellation that makes the analytic value zero.
    d = rng.normal(size=p.shape)
    m = decp.somf_density_response(mol, d)
    h2d = dsomf.somf_two_electron(mol, d, contract="explicit")
    assert mabs(m - h2d) > 1e-3 * mabs(m)

    def tr_d_h2(density):
        h2 = dsomf.somf_two_electron(mol, density, contract="explicit")
        return np.einsum("kpq,pq->k", h2, d)

    h = 1e-5
    fd = (tr_d_h2(p + h * d) - tr_d_h2(p - h * d)) / (2 * h)
    analytic = np.einsum("kpq,pq->k", m, d)
    assert np.max(np.abs(fd - analytic)) < 1e-6 * np.max(np.abs(analytic))
