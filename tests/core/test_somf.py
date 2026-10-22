"""Contracts of the SOMF two-electron operator.

The conventions under test are written down in ``diffmrc.core.somf``:
chemist-notation kernel ``g[k,p,q,r,s]``, contractions
``J = P.g[p,q,r,s]`` and ``K = P.(g[p,r,s,q] + g[s,q,p,r])``, and the raw
``int1e_pnucxp`` prefactor convention.  The reference for the exchange
index pattern is pyscf's own SOMF path
(``pyscf.prop.gtensor.uhf.get_jk``), which is contracted with the same
``int2e_p1vxp1`` kernel through ``direct_mapdm``.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402

from diffmrc.core import somf as dsomf  # noqa: E402


WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    s = mol.intor('int1e_ovlp')
    # Synthetic but physical-shaped density: symmetric, trace(P S) = nelec.
    p = (mol.nelectron / mol.nao_nr()) * np.linalg.inv(s)
    p = 0.5 * (p + p.T)
    return mol, p


def test_one_electron_operator(water):
    mol, _ = water
    h = dsomf.somf_matrix(mol, mode='one_electron')
    ref = mol.intor_asymmetric('int1e_pnucxp', 3)
    assert h.shape == (3, mol.nao_nr(), mol.nao_nr())
    assert np.abs(h - ref).max() < 1e-14
    # Real antisymmetric components.
    assert np.abs(h.imag).max() == 0.0 if np.iscomplexobj(h) else True
    assert np.abs(h + h.transpose(0, 2, 1)).max() < 1e-14


def test_one_electron_ignores_density(water):
    mol, p = water
    h1 = dsomf.somf_matrix(mol, mode='one_electron')
    h2 = dsomf.somf_matrix(mol, P=p, mode='one_electron')
    assert np.abs(h1 - h2).max() == 0.0


def test_two_electron_is_antisymmetric(water):
    mol, p = water
    h2 = dsomf.somf_two_electron(mol, p, contract='explicit')
    assert np.abs(h2 + h2.transpose(0, 2, 1)).max() < 1e-13
    assert np.linalg.norm(h2) > 1e-3


def test_two_electron_is_linear_in_density(water):
    mol, p = water
    h_a = dsomf.somf_two_electron(mol, p, contract='explicit')
    h_b = dsomf.somf_two_electron(mol, 2.0 * p, contract='explicit')
    scale = np.linalg.norm(h_b) / np.linalg.norm(h_a)
    assert abs(scale - 2.0) < 1e-12
    assert np.abs(h_b - 2.0 * h_a).max() < 1e-12 * max(1.0, np.abs(h_a).max())


def test_zero_density_gives_zero(water):
    mol, _ = water
    h2 = dsomf.somf_two_electron(mol, np.zeros((mol.nao_nr(),) * 2),
                                 contract='explicit')
    assert np.abs(h2).max() == 0.0


def test_explicit_matches_direct_route(water):
    """AO-level matrix elements: explicit einsum versus pyscf direct_mapdm."""
    mol, p = water
    h_exp = dsomf.somf_two_electron(mol, p, contract='explicit')
    h_dir = dsomf.somf_two_electron(mol, p, contract='direct')
    diff = np.abs(h_exp - h_dir).max()
    scale = max(np.abs(h_exp).max(), 1.0)
    assert diff / scale < 1e-12, f"explicit vs direct: {diff:.3e}"


def test_explicit_matches_pyscf_somf_operator(water):
    """Independent assembly from pyscf's own gtensor SOMF contraction."""
    from pyscf.prop.gtensor.uhf import get_jk
    mol, p = water
    vj, vk = get_jk(mol, (0.5 * p, 0.5 * p))
    ref = (vj[0] + vj[1]) - 1.5 * (vk[0] + vk[1])
    h = dsomf.somf_two_electron(mol, p, contract='explicit')
    assert np.abs(h - ref).max() < 1e-12 * max(np.abs(ref).max(), 1.0)


def test_auto_route_selects_explicit_for_small_system(water):
    mol, p = water
    h_auto = dsomf.somf_two_electron(mol, p, contract='auto')
    h_exp = dsomf.somf_two_electron(mol, p, contract='explicit')
    assert np.abs(h_auto - h_exp).max() == 0.0
    # A zero budget forces the direct route and must agree.
    h_dir = dsomf.somf_two_electron(mol, p, contract='auto', memory_limit=0)
    assert np.abs(h_dir - h_exp).max() < 1e-12 * max(np.abs(h_exp).max(), 1.0)


def test_mo_transform_consistency(water):
    mol, p = water
    c = np.linalg.qr(np.arange(mol.nao_nr() ** 2, dtype=float)
                     .reshape(mol.nao_nr(), mol.nao_nr()))[0][:, :7]
    h_ao = dsomf.somf_matrix(mol, None, p, contract='explicit')
    h_mo = dsomf.somf_matrix(mol, c, p, contract='explicit')
    ref = np.einsum('ak,cab,bl->ckl', c, h_ao, c, optimize=True)
    assert np.abs(h_mo - ref).max() < 1e-13 * max(np.abs(ref).max(), 1.0)


def test_prefactor_scale(water):
    mol, p = water
    h_raw = dsomf.somf_matrix(mol, None, p, contract='explicit')
    h_eh = dsomf.somf_matrix(mol, None, p, contract='explicit',
                             prefactor=True)
    assert np.abs(h_eh - dsomf.SOMF_ALPHA2_HALF * h_raw).max() == 0.0
    assert 5.0e-6 < dsomf.SOMF_ALPHA2_HALF < 2.7e-5


def test_check_accepts_valid_density(water):
    mol, p = water
    h = dsomf.somf_matrix(mol, None, p, contract='explicit', check=True)
    assert np.abs(h + h.transpose(0, 2, 1)).max() < 1e-12


def test_invalid_inputs_raise(water):
    mol, p = water
    with pytest.raises(ValueError):
        dsomf.somf_matrix(mol, mode='somf')
    with pytest.raises(ValueError):
        dsomf.somf_matrix(mol, mode='nonsense')
    with pytest.raises(ValueError):
        dsomf.somf_matrix(mol, P=p[:-1], mode='somf')
    with pytest.raises(ValueError):
        dsomf.somf_two_electron(mol, p, contract='nonsense')
    with pytest.raises(ValueError):
        dsomf.somf_matrix(mol, C=np.ones((mol.nao_nr() - 1, 2)), P=p)
    with pytest.raises(ValueError):
        dsomf.somf_matrix(mol, None, 2.0 * p, check=True)  # trace 2N
