"""Contracts of the density-fitted two-component X2C HF layer.

The conventions under test are documented in ``diffmrc.core.x2c_df`` and
were pinned numerically against pyscf 2.14 / libcint:

  * the X2C one-electron Hamiltonian reproduces ``SpinorX2CHelper``;
  * the spinor RI-JK factors reproduce pyscf's exact complex-GHF J and K
    to the density-fitting error;
  * the DF SCF energy reproduces the exact-ERI X2C UHF energy to that
    same error;
  * the analytic nuclear gradient reproduces a central difference of the
    converged DF energy (criterion 1e-5 relative);
  * the gradient satisfies translation and rotation invariance.

The fixture chain is H2O/6-31G with the def2-universal-jkfit auxiliary
basis, the same system used in the development probes.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import df, gto  # noqa: E402
from pyscf.x2c import x2c as pyscf_x2c  # noqa: E402

from diffmrc.core import x2c_df  # noqa: E402


GEOM_ANGSTROM = [('O', (0.0, 0.0, 0.0)),
                 ('H', (0.0, -0.757, 0.587)),
                 ('H', (0.0, 0.757, 0.587))]
AUX_BASIS = 'def2-universal-jkfit'


def _build_mol(coords_bohr=None):
    mol = gto.M(atom=[(s, tuple(x)) for s, x in GEOM_ANGSTROM],
                basis='6-31g', unit='Angstrom', verbose=0)
    if coords_bohr is not None:
        mol.set_geom_(coords_bohr, unit='Bohr')
    return mol


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _invariance(grad, mol):
    r = mol.atom_coords()
    trans = np.abs(grad.sum(axis=0)).max() / max(np.abs(grad).max(), 1e-30)
    rot = np.zeros(3)
    for ia in range(mol.natm):
        rot += np.cross(r[ia], grad[ia])
    rot = np.abs(rot).max() / max(np.abs(grad).max() * np.abs(r).max(),
                                  1e-30)
    return float(trans), float(rot)


@pytest.fixture(scope='module')
def water():
    mol = _build_mol()
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUX_BASIS)
    mf = pyscf_x2c.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.run()
    return mol, auxmol, mf


@pytest.fixture(scope='module')
def scf(water):
    mol, auxmol, _ = water
    return x2c_df.df_x2c_hf(mol, auxmol=auxmol)


def test_hcore_matches_pyscf(water):
    mol, _, _ = water
    h = x2c_df.df_x2c_hcore(mol)
    ref = pyscf_x2c.SpinorX2CHelper(mol).get_hcore(mol)
    assert np.abs(h - ref).max() < 1e-9
    assert _rel(h, ref) < 1e-10
    assert np.abs(h - h.conj().T).max() < 1e-10


def test_hcore_in_mo_basis(water, scf):
    mol, _, _ = water
    h = x2c_df.df_x2c_hcore(mol)
    c = scf.mo_coeff
    h_mo = x2c_df.df_x2c_hcore(mol, C=c)
    assert np.abs(h_mo - c.conj().T @ h @ c).max() < 1e-12


def test_df_factors_reproduce_exact_jk(water):
    """DF J/K against pyscf's exact spinor ERI at the exact density."""
    mol, auxmol, mf = water
    factors = x2c_df.spinor_df_factors(mol, auxmol=auxmol)
    d = mf.make_rdm1()
    vj, vk = mf.get_jk(mol, d)
    w = factors.w
    rho = np.einsum('ijP,ji->P', w, d)
    j = np.einsum('ijP,P->ij', w, rho)
    k = np.zeros_like(d)
    for p in range(factors.naux):
        wp = w[:, :, p]
        k += wp @ d @ wp
    assert _rel(j, vj) < 1e-4
    assert _rel(k, vk) < 1e-4


def test_df_scf_matches_exact_within_df_error(water, scf):
    """The DF SCF energy differs from exact-ERI X2C UHF only by DF error."""
    _, _, mf = water
    assert scf.converged
    assert scf.gradient_norm < 1e-7
    assert abs(scf.e_tot - mf.e_tot) < 1e-4


def test_derivative_factors_against_geometry_fd(water):
    """dB/dR and dS_metric/dR for one atom against central differences."""
    mol, auxmol, _ = water
    factors = x2c_df.spinor_df_factors(mol, auxmol=auxmol)
    coords = mol.atom_coords()
    step = 1e-4
    db, ds_metric, dj = factors.deriv_factors(0)
    cp = coords.copy(); cp[0, 2] += step
    cm = coords.copy(); cm[0, 2] -= step
    mp, mm = _build_mol(cp), _build_mol(cm)
    ap = df.addons.make_auxmol(mp, auxbasis=AUX_BASIS)
    am = df.addons.make_auxmol(mm, auxbasis=AUX_BASIS)
    bp = df.incore.aux_e2(mp, ap, intor='int3c2e_spinor', aosym='s1')
    bm = df.incore.aux_e2(mm, am, intor='int3c2e_spinor', aosym='s1')
    db_fd = (bp - bm) / (2 * step)
    assert _rel(db[2], db_fd) < 1e-6
    sp = ap.intor('int2c2e_sph', aosym='s1')
    sm = am.intor('int2c2e_sph', aosym='s1')
    ds_fd = (sp - sm) / (2 * step)
    assert _rel(ds_metric[2], ds_fd) < 1e-6
    assert dj.shape == ds_metric.shape
    assert np.isfinite(dj).all()


def test_gradient_against_finite_difference(water, scf):
    """Analytic gradient against a 3-point central difference (h=1e-3)."""
    mol, _, _ = water
    grad, parts = x2c_df.df_x2c_hf_gradient(mol, result=scf)
    coords = mol.atom_coords()
    step = 1e-3
    g_fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            cp = coords.copy(); cp[ia, x] += step
            cm = coords.copy(); cm[ia, x] -= step
            ep = x2c_df.df_x2c_hf(_build_mol(cp)).e_tot
            em = x2c_df.df_x2c_hf(_build_mol(cm)).e_tot
            g_fd[ia, x] = (ep - em) / (2 * step)
    assert _rel(grad, g_fd) < 2e-5

    trans, rot = _invariance(grad, mol)
    assert trans < 1e-6
    assert rot < 1e-6
    assert set(parts) == {'one_electron', 'two_electron', 'pulay', 'nuclear'}
    assert np.abs(grad - sum(parts.values())).max() < 1e-12


def test_symmetry_zero_direction(water, scf):
    """H2O lies in the yz plane: every x gradient component must vanish."""
    mol, _, _ = water
    grad, _ = x2c_df.df_x2c_hf_gradient(mol, result=scf)
    assert np.abs(grad[:, 0]).max() < 1e-6
