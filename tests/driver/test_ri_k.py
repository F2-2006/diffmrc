"""Contracts of the RI-K exchange layer and of the RIJK SCF route.

The numbers pinned here were measured with pyscf 2.14.0 on this stack:

* RI-K from the three-center factors reproduces the exchange built from
  the dense RI four-center tensor to 7e-15 (H2O/6-31G, OH/6-31G), and
  pyscf's own DF-K at the same auxiliary basis to 1e-12.
* The deviation from the *exact* four-center exchange is the fit error
  of the auxiliary basis: 4.8e-5 Eh on H2O/6-31G and 3.1e-5 Eh on
  OH/6-31G with def2-universal-jkfit.  It is not an implementation
  error and does not shrink with an implementation change; the
  implementation-level identity is the dense-RI and DF-K comparison.
* The fixed-coefficient RI-K gradient agrees with central differences
  to 3.6e-8 relative (closed shell) and 1.4e-8 (spin-resolved).
* The plain-pyscf RIJK SCF converges, its energy equals ``rijk_energy``
  at the converged density to 3e-14, and its explicit gradient agrees
  with central differences of the converged energy to 7e-8 Eh/Bohr.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip('pyscf')
from pyscf import gto, scf  # noqa: E402
from pyscf.df import df_jk  # noqa: E402

from diffmrc.core import ri as dri  # noqa: E402
from diffmrc.driver import ri_scf as rs  # noqa: E402
from diffmrc.driver import ri_scf_driver as drv  # noqa: E402

WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'
OH = 'O 0 0 0; H 0 0 0.9697'
AUX = rs.RI_SCF_DEFAULT_AUX


def _system(atom, spin=0, basis='6-31g'):
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', spin=spin, verbose=0)
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.kernel()
    assert mf.converged
    nocc = (mol.nelectron - spin) // 2
    return mol, mf, np.asarray(mf.mo_coeff[:, :nocc], dtype=float)


@pytest.fixture(scope='module')
def water():
    return _system(WATER)


@pytest.fixture(scope='module')
def oh():
    return _system(OH, spin=1)


def _displaced(mol, atom, axis, step):
    coords = np.array(mol.atom_coords(), dtype=float)
    coords[atom, axis] += step
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    return gto.M(atom=[(symbols[i], coords[i]) for i in range(mol.natm)],
                 basis=mol.basis, unit='Bohr', spin=mol.spin,
                 charge=mol.charge, verbose=0)


def _fd_gradient(mol, fun, step=1.0e-4):
    grad = np.zeros((mol.natm, 3))
    for atom in range(mol.natm):
        for axis in range(3):
            grad[atom, axis] = (fun(_displaced(mol, atom, axis, step))
                                - fun(_displaced(mol, atom, axis, -step))) \
                / (2.0 * step)
    return grad


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def _coeffs_from_density(dm):
    w, u = np.linalg.eigh(0.5 * (dm + dm.T))
    keep = w > 1e-12 * max(w[-1], 1.0)
    return u[:, keep] * np.sqrt(w[keep] / 2.0)[None, :]


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def test_ri_k_matches_dense_ri(water, oh):
    """The factor contraction equals the exchange from the dense RI ERI."""
    for mol, _, c in (water, oh):
        dm = rs.density_from_coeffs(c)
        auxmol = dri.make_auxmol(mol, auxbasis=AUX)[0]
        factors = dri.ri_factors(mol, auxmol)
        k = rs.ri_k_fock(mol, dm, factors=factors)
        eri = dri.ri_eri_ao(mol, auxmol, factors=factors)
        k_ref = np.einsum('mlns,ls->mn', eri, dm)
        del eri
        assert np.abs(k - k_ref).max() < 1e-11
        assert np.abs(k - k.T).max() < 1e-14


def test_ri_k_matches_pyscf_dfk(water):
    """The contraction equals pyscf's DF-K at the same auxiliary basis.

    pyscf's ``density_fit`` path applies the same Coulomb-metric factors
    through a blocked Cholesky route; agreement at 1e-12 pins the
    contraction and the axis conventions against an independent
    implementation.
    """
    mol, _, c = water
    dm = rs.density_from_coeffs(c)
    mf_df = scf.RHF(mol).density_fit(auxbasis=AUX)
    _, k_pyscf = df_jk.get_jk(mf_df.with_df, dm, with_j=False, with_k=True)
    k = rs.ri_k_fock(mol, dm, aux=AUX)
    assert np.abs(k - 0.5 * (k_pyscf + k_pyscf.T)).max() < 1e-10


def test_ri_k_versus_exact_exchange_is_the_fit_error(water, oh):
    """The exact-exchange deviation is the auxiliary-basis fit, not a bug.

    The RI-K energy is *by construction* the exchange of the fitted
    four-center tensor, so the physically meaningful comparison is the
    fit error.  On these small orbital bases with def2-universal-jkfit
    it is a few 1e-5 Eh; the same deviation appears in pyscf's DF-K.
    """
    for mol, mf_ref, c in (water, oh):
        dm = rs.density_from_coeffs(c)
        eri = mol.intor('int2e', aosym='s1')
        k_ex = np.einsum('mlns,ls->mn', eri, dm)
        e_ex = -0.25 * float(np.sum(dm * k_ex))
        del eri
        e_ri = rs.ri_k_energy(mol, c, aux=AUX)
        assert abs(e_ri - e_ex) < 2.0e-4
        assert abs(e_ri - e_ex) > 1.0e-6  # the fit is not exact


def test_ri_k_block_matches_full(water):
    """Blocking over the auxiliary index changes nothing but the buffer."""
    mol, _, c = water
    dm = rs.density_from_coeffs(c)
    k_full = rs.ri_k_fock(mol, dm, aux=AUX)
    for block in (1, 7, 32):
        k_blk = rs.ri_k_fock(mol, dm, aux=AUX, block=block)
        assert np.abs(k_full - k_blk).max() < 1e-13


def test_ri_k_batched_density(water):
    """A spin-stacked density is mapped density by density."""
    mol, _, c = water
    dm = rs.density_from_coeffs(c) * 0.5
    stacked = np.stack([dm, 1.5 * dm])
    k = rs.ri_k_fock(mol, stacked, aux=AUX)
    assert k.shape == stacked.shape
    assert np.abs(k[0] - rs.ri_k_fock(mol, dm, aux=AUX)).max() < 1e-13
    assert np.abs(k[1] - 1.5 * rs.ri_k_fock(mol, dm, aux=AUX)).max() < 1e-13


def test_ri_k_zero_density(water):
    mol, _, _ = water
    zero = np.zeros((mol.nao_nr(), mol.nao_nr()))
    assert np.abs(rs.ri_k_fock(mol, zero, aux=AUX)).max() < 1e-16
    assert abs(rs.ri_k_energy_from_density(mol, zero, aux=AUX)) < 1e-16


# ---------------------------------------------------------------------------
# Fixed-coefficient gradients
# ---------------------------------------------------------------------------
def test_ri_k_gradient_matches_fd(water):
    mol, _, c = water
    g = rs.ri_k_gradient(mol, c, aux=AUX)
    g_fd = _fd_gradient(mol, lambda m: rs.ri_k_energy(m, c, aux=AUX))
    assert _rel(g, g_fd) < 1e-6
    assert np.abs(g.sum(axis=0)).max() < 1e-8


def test_ri_k_gradient_random_density_matches_fd(water):
    """A non-idempotent density exercises the linear map, not an orbital set."""
    mol, _, _ = water
    rng = np.random.default_rng(11)
    x = rng.standard_normal((mol.nao_nr(), 4))
    dm = x @ x.T
    c = _coeffs_from_density(dm)
    assert np.abs(rs.density_from_coeffs(c) - dm).max() < 1e-11
    g = rs.ri_k_gradient(mol, c, aux=AUX)
    g_fd = _fd_gradient(mol, lambda m: rs.ri_k_energy(m, c, aux=AUX))
    assert _rel(g, g_fd) < 1e-6


def test_ri_k_gradient_spin_matches_fd(oh):
    """Spin-summed exchange ``-1/2`` per spin density, checked by FD."""
    mol, _, c = oh
    nocc = c.shape[1]
    occ_a = np.ones(nocc)
    occ_b = np.ones(nocc)
    occ_b[-1] = 0.0

    def e_spin(m):
        return 2.0 * (rs.ri_k_energy(m, c, occ=occ_a, aux=AUX)
                      + rs.ri_k_energy(m, c, occ=occ_b, aux=AUX))

    g = rs.ri_k_gradient_spin(mol, c, occ_alpha=occ_a, occ_beta=occ_b,
                              aux=AUX)
    g_fd = _fd_gradient(mol, e_spin)
    assert _rel(g, g_fd) < 1e-6
    assert np.abs(g.sum(axis=0)).max() < 1e-8


def test_ri_k_coordinate_tangent_matches_k_derivative(water):
    """The analytic coordinate tangent used by the driver JVP is dK/dR.

    ``driver.ri_scf_driver._RIState._make_fock_k`` propagates the
    coordinate tangent as ``dK = t1 + t1^T`` with
    ``t1[m, v] = sum_{l P} dW[m, l, P] z[l, v, P]`` and ``z = D W_P``.
    This checks that expression against central differences of the
    assembled matrix, independently of JAX/pyscfad.
    """
    mol, _, c = water
    dm = rs.density_from_coeffs(c)
    auxmol = dri.make_auxmol(mol, auxbasis=AUX)[0]
    factors = dri.ri_factors(mol, auxmol)
    w = factors.b @ factors.jhalf
    z = np.einsum('ls,svP->lvP', dm, w)
    step = 1.0e-4
    for atom in range(mol.natm):
        d = dri.ri_factors_deriv(mol, auxmol, atom, factors)
        dw = (np.einsum('cuvP,PQ->cuvQ', d.db, factors.jhalf)
              + np.tensordot(factors.b, d.djhalf,
                             axes=([2], [1])).transpose(2, 0, 1, 3))
        dwm = np.transpose(dw, (1, 2, 3, 0))          # [m, l, P, c]
        t1 = np.einsum('mlPc,lvP->mvc', dwm, z)
        dk = t1 + np.swapaxes(t1, 0, 1)               # [m, v, c]
        for axis in range(3):
            mp = _displaced(mol, atom, axis, step)
            mm = _displaced(mol, atom, axis, -step)
            cp = dri.ri_factors(mp, dri.make_auxmol(mp, auxbasis=AUX)[0])
            cm = dri.ri_factors(mm, dri.make_auxmol(mm, auxbasis=AUX)[0])
            fd = (rs.ri_k_fock(None, dm, factors=cp)
                  - rs.ri_k_fock(None, dm, factors=cm)) / (2 * step)
            assert np.abs(dk[:, :, axis] - fd).max() < 1e-6


def test_ri_k_gradient_atom_subset(water):
    mol, _, c = water
    g = rs.ri_k_gradient(mol, c, aux=AUX)
    g_sub = rs.ri_k_gradient(mol, c, aux=AUX, atoms=[1])
    assert g_sub.shape == (1, 3)
    assert np.abs(g_sub[0] - g[1]).max() < 1e-13


def test_rijk_energy_and_gradient_parts(water, oh):
    for mol, _, c in (water, oh):
        e = rs.rijk_energy(mol, c, aux=AUX)
        g, parts = rs.rijk_gradient(mol, c, aux=AUX, return_parts=True)
        assert set(parts) == {'j', 'k', 'h', 'nuc'}
        assert np.abs(g - sum(parts.values())).max() < 1e-12
        g_fd = _fd_gradient(mol, lambda m: rs.rijk_energy(m, c, aux=AUX))
        assert _rel(g, g_fd) < 1e-5


# ---------------------------------------------------------------------------
# Self-consistent RI-K on plain pyscf
# ---------------------------------------------------------------------------
def test_host_rijk_scf_energy_is_layer_energy():
    """The converged SCF energy equals the RI-K energy at its own density."""
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    mf = drv.HostRIJK_RHF(mol)
    mf.kernel()
    assert mf.converged
    nocc = mol.nelectron // 2
    c = np.asarray(mf.mo_coeff)[:, :nocc]
    e_layer = rs.rijk_energy(mol, c, aux=AUX)
    assert abs(e_layer - mf.e_tot) < 1e-9


def test_host_rijk_scf_leaves_no_dense_eri():
    """A full RI-K SCF never forms or caches the four-center tensor."""
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    mf = drv.HostRIJK_RHF(mol)
    mf._eri = None
    dm = mf.get_init_guess()
    mf.get_jk(mol, dm)
    assert mf._eri is None


def test_host_rijk_scf_gradient_matches_fd():
    """Explicit gradient of the converged RIJK energy against FD.

    The stretched OH geometry is used because at equilibrium the
    gradient components are symmetry zeros and a wrong response term
    would not show.
    """
    mol = gto.M(atom='O 0 0 0; H 0 0 2.30', basis='6-31g', spin=1,
                unit='Angstrom', verbose=0)
    mf = drv.HostRIJK_ROHF(mol)
    mf.kernel()
    assert mf.converged
    g = drv.rijk_scf_gradient(mf)

    step = 2.0e-3
    fd = np.zeros_like(g)
    for atom in range(mol.natm):
        for axis in range(3):
            values = []
            for sign in (1.0, -1.0):
                m = _displaced(mol, atom, axis, sign * step)
                ref = drv.HostRIJK_ROHF(m)
                ref.kernel()
                assert ref.converged
                values.append(ref.e_tot)
            fd[atom, axis] = (values[0] - values[1]) / (2.0 * step)
    assert np.abs(g - fd).max() < 1e-5, (
        f'explicit vs FD {np.abs(g - fd).max():.3e} Eh/Bohr')


def test_host_rijk_closed_shell_gradient_matches_fd():
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    mf = drv.HostRIJK_RHF(mol)
    mf.kernel()
    g = drv.rijk_scf_gradient(mf)
    step = 2.0e-3
    fd = np.zeros_like(g)
    for atom in range(mol.natm):
        for axis in range(3):
            values = []
            for sign in (1.0, -1.0):
                m = _displaced(mol, atom, axis, sign * step)
                ref = drv.HostRIJK_RHF(m)
                ref.kernel()
                values.append(ref.e_tot)
            fd[atom, axis] = (values[0] - values[1]) / (2.0 * step)
    assert np.abs(g - fd).max() < 1e-5


def test_host_rijk_uhf_gradient_matches_fd():
    """UHF: the spin-resolved RI-K exchange follows both spin densities."""
    mol = gto.M(atom='O 0 0 0; H 0 0 2.30', basis='6-31g', spin=1,
                unit='Angstrom', verbose=0)
    mf = drv.HostRIJK_UHF(mol)
    mf.kernel()
    assert mf.converged
    dm = np.asarray(mf.make_rdm1())
    assert dm.shape[0] == 2
    g = drv.rijk_scf_gradient(mf)
    step = 2.0e-3
    fd = np.zeros_like(g)
    for atom in range(mol.natm):
        for axis in range(3):
            values = []
            for sign in (1.0, -1.0):
                m = _displaced(mol, atom, axis, sign * step)
                ref = drv.HostRIJK_UHF(m)
                ref.kernel()
                values.append(ref.e_tot)
            fd[atom, axis] = (values[0] - values[1]) / (2.0 * step)
    assert np.abs(g - fd).max() < 1e-5, (
        f'UHF gradient vs FD {np.abs(g - fd).max():.3e} Eh/Bohr')


def test_host_ri_k_switch_changes_only_exchange():
    """Flipping ``ri_k`` moves the energy by the K fit, and both converge."""
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    mf_k = drv.HostRIJK_RHF(mol)
    mf_k.kernel()
    mf_x = drv.HostRIJK_RHF(mol, ri_k=False)
    mf_x.kernel()
    assert mf_k.converged and mf_x.converged
    assert mf_k.ri_k and not mf_x.ri_k
    d = mf_k.e_tot - mf_x.e_tot
    # measured +1.05e-5 Ha: the J fit (-3.75e-5) plus the K fit (+4.8e-5)
    assert abs(d) < 2.0e-4
    ref = scf.RHF(mol).run()
    assert abs(mf_x.e_tot - ref.e_tot) < 1e-4
