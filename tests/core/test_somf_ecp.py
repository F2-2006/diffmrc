"""Contracts of the valence-level SOMF operator for ECP calculations.

The definition under test is stated in ``diffmrc.core.somf_ecp``:

* on a center carrying a spin-orbit ECP, the ECP spin-orbit operator
  replaces the nuclear one-electron SOC (adding both would double count
  the core contribution);
* on every other center the nuclear SOC is included with the pyscf
  atomic charge, which is the effective charge ``Z - N_core`` on scalar
  ECP centers and the full charge otherwise;
* the two-electron part is the same ``J - 3/2 K`` mean field as in the
  all-electron module, contracted with the density of the electrons that
  are explicitly present in the calculation.

The tests also pin down the density-response kernel.  The SOMF kernel is
not symmetric under the pair exchange ``(pq) <-> (rs)``, so the density
response is not ``h2[dP]``: it needs its own contraction with the
transition density.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402

from diffmrc.core import somf as dsomf  # noqa: E402
from diffmrc.core import somf_ecp as decp  # noqa: E402


def mabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def frob(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def make_mol(atom, basis, ecp, spin):
    return gto.M(atom=atom, basis=basis, ecp=ecp, spin=spin,
                 unit='Bohr', verbose=0)


@pytest.fixture(scope='module')
def gef():
    mol = make_mol([('Ge', (0.0, 0.0, 0.0)), ('F', (0.5, 0.3, 2.4))],
                   {'Ge': 'lanl2dz', 'F': '6-31g'},
                   {'Ge': 'ecpds28mdfso'}, spin=1)
    s = mol.intor('int1e_ovlp')
    # Physical-shaped valence density: trace(P S) = mol.nelectron.
    p = (mol.nelectron / mol.nao_nr()) * np.linalg.inv(s)
    p = 0.5 * (p + p.T)
    return mol, p


@pytest.fixture(scope='module')
def water():
    return gto.M(atom='O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0',
                 basis='6-31g', unit='Angstrom', verbose=0)


def test_ecp_soc_shape_and_symmetry(gef):
    mol, _ = gef
    v = decp.ecp_soc(mol)
    assert v.shape == (3, mol.nao_nr(), mol.nao_nr())
    assert not np.iscomplexobj(v)
    # The ECP spin-orbit operator is real antisymmetric per component.
    assert mabs(v + v.transpose(0, 2, 1)) < 1e-14
    assert frob(v) > 1e-3


def test_ecp_soc_absent_raises(water):
    with pytest.raises(ValueError):
        decp.ecp_soc(water)


def test_soc_centers(gef, water):
    mol, _ = gef
    assert decp.so_ecp_centers(mol) == [0]
    assert decp.so_ecp_centers(water) == []
    assert decp.nuclear_centers(mol) == [1]
    assert decp.nuclear_centers(water) == [0, 1, 2]


def test_nuclear_soc_reproduces_pnucxp(gef, water):
    """The per-center reconstruction must equal the built-in operator."""
    for mol in (gef[0], water):
        ref = mol.intor_asymmetric('int1e_pnucxp', 3)
        got = decp.nuclear_soc(mol)
        assert mabs(got - ref) < 1e-12 * max(mabs(ref), 1.0)


def test_nuclear_soc_uses_ecp_effective_charge(gef):
    """On an ECP center the ATM charge is Z - N_core, not the full Z."""
    mol, _ = gef
    ge = decp.nuclear_soc(mol, [0])
    with mol.with_rinv_at_nucleus(0):
        prinv = np.asarray(mol.intor('int1e_prinvxp', comp=3))
    z_eff = float(mol.atom_charge(0))
    assert z_eff == 4.0
    assert mabs(ge + z_eff * prinv) < 1e-12 * max(mabs(ge), 1.0)
    # The full nuclear charge is not what pnucxp uses here.
    assert mabs(ge + 32.0 * prinv) > 1e-2


def test_nuclear_soc_additivity(gef):
    mol, _ = gef
    total = decp.nuclear_soc(mol)
    parts = decp.nuclear_soc(mol, [0]) + decp.nuclear_soc(mol, [1])
    assert mabs(total - parts) < 1e-12 * max(mabs(total), 1.0)


def test_valence_somf_terms(gef):
    """Default assembly in the +i convention.

    h = (alpha^2/2) [nuclear(no-SO centers) + valence h2] - ECPso.
    """
    mol, p = gef
    a2 = dsomf.SOMF_ALPHA2_HALF
    h = decp.valence_somf_matrix(mol, p, contract='explicit')
    ref = (a2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
                 + dsomf.somf_two_electron(mol, p, contract='explicit'))
           - decp.ecp_soc(mol))
    assert mabs(h - ref) < 1e-13 * max(mabs(ref), 1.0)


def test_valence_somf_raw_convention(gef):
    """The raw view adds the native components without scaling."""
    mol, p = gef
    h = decp.valence_somf_matrix(mol, p, contract='explicit',
                                 convention='raw')
    ref = (decp.ecp_soc(mol)
           + decp.nuclear_soc(mol, decp.nuclear_centers(mol))
           + dsomf.somf_two_electron(mol, p, contract='explicit'))
    assert mabs(h - ref) < 1e-13 * max(mabs(ref), 1.0)


def test_valence_somf_nuclear_modes(gef):
    mol, p = gef
    h_auto = decp.valence_somf_matrix(mol, p, contract='explicit',
                                      nuclear='auto')
    h_all = decp.valence_somf_matrix(mol, p, contract='explicit',
                                     nuclear='all')
    h_none = decp.valence_somf_matrix(mol, p, contract='explicit',
                                      nuclear='none')
    a2 = dsomf.SOMF_ALPHA2_HALF
    ge = a2 * decp.nuclear_soc(mol, [0])
    f = a2 * decp.nuclear_soc(mol, [1])
    assert mabs(h_auto - h_none - f) < 1e-13 * max(mabs(h_auto), 1.0)
    assert mabs(h_all - h_auto - ge) < 1e-13 * max(mabs(h_all), 1.0)


def test_valence_somf_linear_in_density(gef):
    mol, p = gef
    h = decp.valence_somf_matrix(mol, p, contract='explicit')
    h2 = decp.valence_somf_matrix(mol, 2.0 * p, contract='explicit')
    base = dsomf.somf_two_electron(mol, p, contract='explicit')
    scaled = dsomf.somf_two_electron(mol, 2.0 * p, contract='explicit')
    assert mabs(scaled - 2.0 * base) < 1e-12 * max(mabs(base), 1.0)
    delta = h2 - h - dsomf.SOMF_ALPHA2_HALF * base
    assert mabs(delta) < 1e-12 * max(dsomf.SOMF_ALPHA2_HALF * mabs(base),
                                     1e-20)


def test_valence_somf_explicit_matches_direct(gef):
    mol, p = gef
    h_exp = dsomf.somf_two_electron(mol, p, contract='explicit')
    h_dir = dsomf.somf_two_electron(mol, p, contract='direct')
    assert mabs(h_exp - h_dir) < 1e-12 * max(mabs(h_exp), 1.0)


def test_valence_somf_is_antisymmetric_and_real(gef):
    mol, p = gef
    h = decp.valence_somf_matrix(mol, p, contract='explicit')
    assert not np.iscomplexobj(h)
    assert mabs(h + h.transpose(0, 2, 1)) < 1e-12 * max(mabs(h), 1.0)


def test_valence_somf_density_check(gef):
    mol, p = gef
    with pytest.raises(ValueError):
        decp.valence_somf_matrix(mol, 0.5 * p, contract='explicit',
                                 check=True)


def test_definition_dict(gef):
    mol, _ = gef
    d = decp.somf_definition(mol)
    assert d['so_ecp_centers'] == [0]
    assert d['nuclear_centers'] == [1]
    assert d['density_electrons'] == mol.nelectron
    assert d['two_electron'] == 'valence J - 3/2 K'


def test_response_kernel_matches_density_derivative(gef):
    """M must be the functional derivative of tr(D h2[P]) with respect to P."""
    mol, _ = gef
    rng = np.random.default_rng(3)
    nao = mol.nao_nr()
    d = rng.standard_normal((nao, nao))
    d = 0.5 * (d - d.T)          # transition density of a degenerate pair
    p = rng.standard_normal((nao, nao))
    p = 0.5 * (p + p.T)          # symmetric reference density
    dp = rng.standard_normal((nao, nao))
    dp = 0.5 * (dp + dp.T)

    m = decp.somf_density_response(mol, d, contract='explicit')
    h_plus = dsomf.somf_two_electron(mol, p + 1e-6 * dp,
                                     contract='explicit')
    h_minus = dsomf.somf_two_electron(mol, p - 1e-6 * dp,
                                      contract='explicit')
    num = (np.sum(d * h_plus) - np.sum(d * h_minus)) / 2e-6
    an = float(np.sum(m * dp))
    assert abs(num - an) <= 1e-6 * max(abs(num), abs(an), 1.0)


def test_response_kernel_is_not_the_value_kernel(gef):
    """The kernel is not pair symmetric, so h2[D] is not the response."""
    mol, _ = gef
    rng = np.random.default_rng(5)
    nao = mol.nao_nr()
    d = rng.standard_normal((nao, nao))
    d = 0.5 * (d - d.T)
    m = decp.somf_density_response(mol, d, contract='explicit')
    h2d = dsomf.somf_two_electron(mol, d, contract='explicit')
    assert mabs(m - h2d) > 1e-2 * max(mabs(m), 1.0)


def test_ybf_ecp_soc_reference():
    """Reference ECPso norms on YbF (stuttgart_rsc + 6-31g)."""
    mol = make_mol([('Yb', (0.0, 0.0, 0.0)), ('F', (0.0, 0.0, 3.6))],
                   {'Yb': 'stuttgart_rsc', 'F': '6-31g'},
                   {'Yb': 'ecpds28mwbso'}, spin=1)
    v = decp.ecp_soc(mol)
    ref = [17.04161885224311, 17.04161885224311, 17.041597386144154]
    assert np.allclose([frob(v[k]) for k in range(3)], ref,
                       rtol=1e-10, atol=1e-10)
    assert decp.so_ecp_centers(mol) == [0]
    assert decp.nuclear_centers(mol) == [1]
