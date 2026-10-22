"""Contracts of the SCF-stage RI-J route.

The numerical claims are the ones measured on 101 with pyscf 2.14.0:
the RI-J matrix agrees with pyscf's own RI-J construction to 4e-12, the
fixed-coefficient RI-J gradient agrees with central differences to about
3e-8 relative at h=1e-4 Bohr, and the RIJONX gradient agrees with
central differences to about 3e-7 relative.  The exchange part of
RIJONX is dense; these tests therefore stay on H2O/CO with 6-31G.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip('pyscf')
from pyscf import gto, scf  # noqa: E402
from pyscf.df import df_jk  # noqa: E402

from diffmrc.core import ri as dri  # noqa: E402
from diffmrc.driver import ri_scf as rs  # noqa: E402

WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'
CO = 'C 0 0 0; O 0 0 1.13'
AUX = rs.RI_SCF_DEFAULT_AUX


def _system(atom, basis='6-31g'):
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged
    nocc = mol.nelectron // 2
    return mol, mf, np.asarray(mf.mo_coeff[:, :nocc], dtype=float)


@pytest.fixture(scope='module')
def water():
    return _system(WATER)


@pytest.fixture(scope='module')
def co():
    return _system(CO)


def _coeffs_from_density(dm):
    """A coefficient matrix whose closed-shell density equals ``dm``."""
    w, u = np.linalg.eigh(0.5 * (dm + dm.T))
    keep = w > 1e-12 * max(w[-1], 1.0)
    return u[:, keep] * np.sqrt(w[keep] / 2.0)[None, :]


def _displaced(mol, atom, axis, step):
    coords = np.array(mol.atom_coords(), dtype=float)
    coords[atom, axis] += step
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    return gto.M(atom=[(symbols[i], coords[i]) for i in range(mol.natm)],
                 basis=mol.basis, unit='Bohr', verbose=0)


def _fd_gradient(mol, fun, step=1.0e-4):
    grad = np.zeros((mol.natm, 3))
    for atom in range(mol.natm):
        for axis in range(3):
            plus = _displaced(mol, atom, axis, step)
            minus = _displaced(mol, atom, axis, -step)
            grad[atom, axis] = (fun(plus) - fun(minus)) / (2.0 * step)
    return grad


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_density_from_coeffs_closed_shell(water):
    mol, _, c = water
    dm = rs.density_from_coeffs(c)
    assert dm.shape == (mol.nao_nr(), mol.nao_nr())
    # MO coefficients are orthonormal in the AO overlap, so the electron
    # count is Tr(D S), not Tr(D).
    assert abs(np.trace(dm @ mol.intor('int1e_ovlp')) - mol.nelectron) < 1e-10
    assert np.abs(dm - dm.T).max() < 1e-14
    with pytest.raises(ValueError):
        rs.density_from_coeffs(np.zeros((3, 2, 2)))


def test_ri_j_energy_matches_pyscf_ri_j(water, co):
    for mol, mf, c in (water, co):
        dm = rs.density_from_coeffs(c)
        mf_df = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf_df.kernel()
        j_ref = df_jk.get_j(mf_df.with_df, dm)
        e_ref = 0.5 * float(np.sum(dm * j_ref))
        e = rs.ri_j_energy(mol, c, aux=AUX)
        # pyscf's RI-J path and the factor path use the same metric; the
        # measured matrix deviation is 4e-12, so 1e-9 on the energy is
        # a regression bound, not a physical tolerance.
        assert abs(e - e_ref) < 1e-9
        # The difference to the exact four-center Coulomb energy is the
        # RI fitting error; on these systems it is below 1e-3 Ha.
        eri = mol.intor('int2e', aosym='s1')
        j_ex = np.einsum('uvls,ls->uv', eri, dm)
        e_ex = 0.5 * float(np.sum(dm * j_ex))
        assert abs(e - e_ex) < 1.0e-3
        assert np.isfinite(e)


def test_ri_j_gradient_matches_fd(water, co):
    for mol, _, c in (water, co):
        g = rs.ri_j_gradient(mol, c, aux=AUX)
        g_fd = _fd_gradient(mol, lambda m: rs.ri_j_energy(m, c, aux=AUX))
        # measured: 3.3e-8 (H2O); the bound leaves room for BLAS
        # differences across builds.
        assert _rel(g, g_fd) < 1e-6
        assert np.abs(g.sum(axis=0)).max() < 1e-8


def test_ri_j_gradient_equals_core_ri_contraction(water):
    mol, _, c = water
    dm = rs.density_from_coeffs(c)
    auxmol, _ = dri.make_auxmol(mol, auxbasis=AUX)
    factors = dri.ri_factors(mol, auxmol)
    gamma = 0.5 * np.einsum('mn,ls->mnls', dm, dm)
    g = rs.ri_j_gradient(mol, c, aux=AUX)
    for atom in range(mol.natm):
        derivs = dri.ri_factors_deriv(mol, auxmol, atom, factors)
        vec = dri.ri_eri_deriv_contract(gamma, factors, derivs,
                                        per_axis=True)
        assert np.abs(vec - g[atom]).max() < 1e-10


def test_exact_k_gradient_matches_fd(water):
    mol, _, _ = water
    rng = np.random.default_rng(7)
    x = rng.standard_normal((mol.nao_nr(), 4))
    dm = x @ x.T
    c = _coeffs_from_density(dm)
    assert np.abs(rs.density_from_coeffs(c) - dm).max() < 1e-11
    g = rs.exact_k_gradient(mol, c)
    g_fd = _fd_gradient(mol, lambda m: rs.exact_k_energy(m, c))
    # measured with a random density: 1.8e-9 absolute
    assert _rel(g, g_fd) < 1e-6
    assert np.abs(g.sum(axis=0)).max() < 1e-8


def test_rijonx_energy_and_gradient(water, co):
    for mol, mf, c in (water, co):
        e = rs.rijonx_energy(mol, c, aux=AUX)
        # RIJONX keeps the exact exchange and fits only the Coulomb
        # term, so the deviation from conventional RHF at the RHF
        # density is the RI-J error (measured 3.8e-5 Ha on H2O).
        assert abs(e - mf.e_tot) < 1.0e-4
        g, parts = rs.rijonx_gradient(mol, c, aux=AUX, return_parts=True)
        assert g.shape == (mol.natm, 3)
        assert set(parts) == {'j', 'k', 'h', 'nuc'}
        assert np.abs(g - sum(parts.values())).max() < 1e-12
        g_fd = _fd_gradient(mol, lambda m: rs.rijonx_energy(m, c, aux=AUX))
        # measured: 2.6e-7 relative on H2O at h=1e-4
        assert _rel(g, g_fd) < 1e-5
        assert np.abs(g.sum(axis=0)).max() < 1e-8


def test_zero_density_boundary(water):
    mol, _, _ = water
    c0 = np.zeros((mol.nao_nr(), 1))
    assert abs(rs.ri_j_energy(mol, c0, aux=AUX)) < 1e-14
    assert np.abs(rs.ri_j_gradient(mol, c0, aux=AUX)).max() < 1e-14
    e = rs.rijonx_energy(mol, c0, aux=AUX)
    assert abs(e - float(mol.energy_nuc())) < 1e-12
    g = rs.rijonx_gradient(mol, c0, aux=AUX)
    assert np.abs(g - rs.nuclear_repulsion_gradient(mol)).max() < 1e-12


def test_ri_j_hessian_fd_layering(water):
    mol, _, c = water
    hess = rs.ri_j_hessian_fd(mol, c, auxbasis=AUX, step=1.0e-4)
    assert hess.shape == (3 * mol.natm, 3 * mol.natm)
    # the symmetrization is exact by construction; the residual measures
    # the finite-difference noise of the gradient surface
    assert np.abs(hess - hess.T).max() < 1e-8
    # a uniform translation is a null direction of a translationally
    # invariant energy; the measured residual is below 1e-5 Eh/Bohr^2
    trans = np.tile(np.eye(3), (mol.natm, 1))
    assert np.abs(hess @ trans).max() < 1e-4


def test_cosx_prototype_grid_convergence(water):
    mol, _, c = water
    dm = rs.density_from_coeffs(c)
    eri = mol.intor('int2e', aosym='s1')
    k_ex = np.einsum('mlns,ls->mn', eri, dm)
    e_ex = -0.25 * float(np.sum(dm * k_ex))
    e1, info1 = rs.cosx_exchange_energy(mol, c, level=1, return_info=True)
    e3, info3 = rs.cosx_exchange_energy(mol, c, level=3, return_info=True)
    # The semi-numerical factorization applies the quadrature to one AO
    # pair, so K is only approximately symmetric; measured asymmetry is
    # 1.7e-5 (level 1), 2.6e-7 (level 3) and decreases with the grid.
    # Symmetrization is therefore part of the construction, not a
    # cosmetic step.
    assert 0.0 < info1['asym'] < 1e-3
    assert info3['asym'] < 1e-5
    assert info1['ngrid'] < info3['ngrid']
    # measured: level 1 error 2.0e-5, level 3 error 4.4e-8 Ha
    assert abs(e1 - e_ex) < 1e-4
    assert abs(e3 - e_ex) < 1e-5
    assert abs(e3 - e_ex) < abs(e1 - e_ex)
    k3, _ = rs.cosx_exchange_matrix(mol, c, level=3)
    assert np.abs(k3 - k3.T).max() < 1e-12
