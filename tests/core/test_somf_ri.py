"""Contracts of the RI-SOMF and one-center SOMF operators.

The operator definition and the ORCA 6.1 section 5.26 presets are
documented in ``diffmrc.core.somf_ri``.  The integral identity

    int2e_p1vxp1[K] = eps_{KLM} (d_L phi_i d_M phi_j | 1/r12 | phi_k phi_l)

is verified directly, because the whole RI construction rests on it.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import df as pyscf_df  # noqa: E402
from pyscf import gto  # noqa: E402

from diffmrc.core import somf as dsomf  # noqa: E402
from diffmrc.core import somf_ri as dsomfri  # noqa: E402


WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'
NEON = 'Ne 0 0 0'


def _density(mol):
    s = mol.intor('int1e_ovlp')
    p = (mol.nelectron / mol.nao_nr()) * np.linalg.inv(s)
    return 0.5 * (p + p.T)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    return mol, _density(mol)


@pytest.fixture(scope='module')
def neon():
    mol = gto.M(atom=NEON, basis='6-31g', unit='Angstrom', verbose=0)
    return mol, _density(mol)


def test_derivative_identity(water):
    """int2e_p1vxp1 is the antisymmetrized mixed bra-pair derivative."""
    mol, _ = water
    nao = mol.nao_nr()
    g = mol.intor('int2e_p1vxp1', 3).reshape(3, nao, nao, nao, nao)
    t9 = mol.intor('int2e_ipvip1', 9).reshape(9, nao, nao, nao, nao)
    cand = dsomfri._cross_contract(t9)
    dev = np.abs(g - cand).max()
    assert dev < 1e-12, f'derivative identity residual {dev:.3e}'


def test_three_center_cross_factor(water):
    """int3c2e_pvxp1 is the three-component cross contraction."""
    mol, _ = water
    nao = mol.nao_nr()
    aux = pyscf_df.addons.make_auxmol(mol, auxbasis='def2-universal-jkfit')
    naux = aux.nao_nr()
    a3 = np.asarray(pyscf_df.incore.aux_e2(
        mol, aux, intor='int3c2e_pvxp1', aosym='s1')).reshape(3, nao, nao,
                                                             naux)
    t9 = np.asarray(pyscf_df.incore.aux_e2(
        mol, aux, intor='int3c2e_ipvip1', aosym='s1')).reshape(
            9, nao, nao, naux)
    cand = dsomfri._cross_contract(t9)
    assert np.abs(a3 - cand).max() < 1e-12 * max(np.abs(cand).max(), 1.0)


def test_ri_coulomb_matches_exact(water):
    mol, p = water
    j_exact = dsomfri.coulomb_exact(mol, p)
    j_ri = dsomfri.coulomb_ri(mol, p)
    rel = np.abs(j_ri - j_exact).max() / np.abs(j_exact).max()
    assert rel < 5e-3, f'RI Coulomb relative error {rel:.3e}'
    assert np.abs(j_ri + j_ri.transpose(0, 2, 1)).max() < 1e-12


def test_ri_coulomb_aux_convergence(water):
    mol, p = water
    j_exact = dsomfri.coulomb_exact(mol, p)
    errors = {}
    for basis in ('def2-SVP-RI', 'def2-TZVP-RI', 'def2-QZVP-RI'):
        j = dsomfri.coulomb_ri(mol, p, auxbasis=basis)
        errors[basis] = np.abs(j - j_exact).max() / np.abs(j_exact).max()
    assert errors['def2-TZVP-RI'] <= errors['def2-SVP-RI'] * 1.5
    assert errors['def2-QZVP-RI'] < 5e-3


def test_ri_factors_reuse(water):
    mol, p = water
    factors = dsomfri.coulomb_ri_factors(mol)
    j1 = dsomfri.coulomb_ri_from_factors(factors, p)
    j2 = dsomfri.coulomb_ri(mol, p, auxmol=None,
                            auxbasis=factors.auxbasis)
    assert np.abs(j1 - j2).max() < 1e-14 * max(np.abs(j1).max(), 1.0)


def test_one_center_equals_exact_on_single_atom(neon):
    """A single atom has only one-center quartets, so 1X is exact."""
    mol, p = neon
    k_exact = dsomfri.exchange_exact(mol, p)
    k_1c = dsomfri.exchange_one_center(mol, p)
    assert np.abs(k_1c - k_exact).max() < 1e-12 * max(
        np.abs(k_exact).max(), 1.0)


def test_amfi_equals_somf_on_single_atom(neon):
    mol, p = neon
    h_somf = dsomfri.somf_ri_two_electron(mol, p, coulomb='exact',
                                          exchange='full')
    h_amfi = dsomfri.somf_ri_two_electron(mol, p, coulomb='one_center',
                                          exchange='one_center')
    assert np.abs(h_amfi - h_somf).max() < 1e-12 * max(
        np.abs(h_somf).max(), 1.0)


def test_one_center_exchange_blocks(water):
    """Only same-atom AO blocks survive, and each equals the masked
    full-tensor contraction over the same atom's quartets."""
    mol, p = water
    nao = mol.nao_nr()
    g = mol.intor('int2e_p1vxp1', aosym='s1').reshape(3, nao, nao, nao, nao)
    blocks = dsomfri.orbital_blocks(mol)
    k_1c = dsomfri.exchange_one_center(mol, p)
    ref = np.zeros((3, nao, nao))
    on_atom = np.zeros(nao, dtype=int)
    for ia, idx in enumerate(blocks):
        on_atom[idx] = ia
        g_a = g[:, idx[:, None, None, None], idx[None, :, None, None],
                 idx[None, None, :, None], idx[None, None, None, :]]
        p_a = p[np.ix_(idx, idx)]
        ref[:, idx[:, None], idx] += (
            np.einsum('rs,kprsq->kpq', p_a, g_a, optimize=True)
            + np.einsum('rs,ksqpr->kpq', p_a, g_a, optimize=True))
    cross = on_atom[:, None] != on_atom[None, :]
    assert np.abs(k_1c[:, cross]).max() == 0.0
    assert np.abs(k_1c - ref).max() < 1e-12 * max(np.abs(ref).max(), 1.0)

    # The one-center exchange differs from the exact exchange because the
    # exact exchange also keeps quartets spread over several atoms.
    k_exact = dsomfri.exchange_exact(mol, p)
    assert np.abs(k_1c - k_exact).max() > 1e-6 * np.abs(k_exact).max()


def test_somf_preset_matches_core_somf(water):
    mol, p = water
    h_core = dsomf.somf_two_electron(mol, p, contract='explicit')
    h_new = dsomfri.somf_ri_two_electron(mol, p, coulomb='exact',
                                         exchange='full')
    assert np.abs(h_new - h_core).max() < 1e-12 * max(
        np.abs(h_core).max(), 1.0)


def test_all_routes_are_antisymmetric(water):
    mol, p = water
    for mode in ('somf', 'somf_1x', 'ri_somf_1x', 'ri_somf', 'amfi'):
        h = dsomfri.somf_ri_matrix(mol, None, p, mode=mode, check=True)
        assert np.abs(h + h.transpose(0, 2, 1)).max() < 1e-10


def test_linearity_in_density(water):
    mol, p = water
    h_a = dsomfri.somf_ri_two_electron(mol, p, coulomb='ri',
                                       exchange='one_center')
    h_b = dsomfri.somf_ri_two_electron(mol, 2.0 * p, coulomb='ri',
                                       exchange='one_center')
    assert np.abs(h_b - 2.0 * h_a).max() < 1e-12 * max(
        np.abs(h_a).max(), 1.0)


def test_zero_density_gives_zero(water):
    mol, _ = water
    z = np.zeros((mol.nao_nr(),) * 2)
    for mode in ('somf_1x', 'ri_somf_1x', 'amfi'):
        h = dsomfri.somf_ri_two_electron(
            mol, z, coulomb={'somf_1x': 'exact', 'ri_somf_1x': 'ri',
                             'amfi': 'one_center'}[mode],
            exchange='one_center')
        assert np.abs(h).max() == 0.0


def test_one_electron_mode_matches_core(water):
    mol, _ = water
    h1 = dsomfri.somf_ri_matrix(mol, mode='one_electron')
    ref = dsomf.one_electron_soc(mol)
    assert np.abs(h1 - ref).max() == 0.0


def test_mo_transform(water):
    mol, p = water
    nao = mol.nao_nr()
    c = np.linalg.qr(np.arange(nao ** 2, dtype=float)
                     .reshape(nao, nao))[0][:, :5]
    h_ao = dsomfri.somf_ri_matrix(mol, None, p, mode='ri_somf_1x')
    h_mo = dsomfri.somf_ri_matrix(mol, c, p, mode='ri_somf_1x')
    ref = np.einsum('ak,cab,bl->ckl', c, h_ao, c, optimize=True)
    assert np.abs(h_mo - ref).max() < 1e-13 * max(np.abs(ref).max(), 1.0)


def test_invalid_inputs(water):
    mol, p = water
    with pytest.raises(ValueError):
        dsomfri.somf_ri_matrix(mol, mode='nonsense')
    with pytest.raises(ValueError):
        dsomfri.somf_ri_matrix(mol, mode='ri_somf_1x')
    with pytest.raises(ValueError):
        dsomfri.somf_ri_two_electron(mol, p, coulomb='nonsense')
    with pytest.raises(ValueError):
        dsomfri.somf_ri_two_electron(mol, p, exchange='nonsense')
    with pytest.raises(ValueError):
        dsomfri.coulomb_exact(mol, p, contract='nonsense')
    with pytest.raises(ValueError):
        dsomfri.coulomb_ri(mol, p[:-1])
