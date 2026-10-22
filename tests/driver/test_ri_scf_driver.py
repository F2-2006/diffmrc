"""Tests for the RI-J SCF driver (RIJONX route).

Acceptance scale follows the RI-J fitting error of the auxiliary basis,
not machine precision: the Coulomb term is approximated, so the SCF
energy, density and dipole move by the fit, while the Fock assembly and
the analytic RI-J Jacobian are checked against independent
implementations and finite differences at machine level.
"""
import numpy as np
import pytest

pytest.importorskip('jax')
pytest.importorskip('pyscf')
pytest.importorskip('pyscfad')

from diffmrc.driver import ri_scf
from diffmrc.driver import ri_scf_driver as drv

H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
CO = 'C 0 0 -0.5643; O 0 0 0.5643'
OH = 'O 0 0 0.0; H 0 0 0.9697'
HI = 'H 0 0 0; I 0 0 1.61'


def _mol(atom, basis='6-31g', spin=0, charge=0, ecp=None):
    from pyscfad import gto
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.spin = spin
    mol.charge = charge
    if ecp is not None:
        mol.ecp = ecp
    mol.verbose = 0
    mol.build()
    return mol


def _reference(mol):
    from pyscfad.scf import hf, rohf
    mf = rohf.ROHF(mol) if mol.spin != 0 else hf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def _run(atom, **kwargs):
    mol = _mol(atom, **kwargs)
    ref = _reference(mol)
    mf = drv.rijonx_scf(mol)
    return mol, ref, mf


@pytest.fixture(scope='module')
def h2o():
    return _run(H2O)


# ---------------------------------------------------------------------------
# Closed-shell energies, densities, dipoles
# ---------------------------------------------------------------------------
def test_h2o_energy_matches_rijonx_energy(h2o):
    """The SCF energy is the RIJONX energy at the converged density."""
    mol, ref, mf = h2o
    pmol = mol.to_pyscf()
    nocc = mol.nelectron // 2
    c = np.asarray(mf.mo_coeff)[:, :nocc]
    e_fix = ri_scf.rijonx_energy(pmol, c)
    assert abs(e_fix - mf.e_tot) < 1e-9
    assert mf.converged


def test_h2o_energy_shift_is_j_fit(h2o):
    """Energy shift against RHF is the J fitting error; measured -3.75e-5 Ha."""
    mol, ref, mf = h2o
    d = mf.e_tot - ref.e_tot
    assert abs(d) < 1e-4, f'energy shift {d:.3e} Ha exceeds the fit scale'
    assert d < 0.0


def test_h2o_density_and_dipole(h2o):
    """Density and dipole follow RHF within the fit; measured 6.5e-5 and 1.4e-5."""
    mol, ref, mf = h2o
    dm_ref = np.asarray(ref.make_rdm1())
    dm = np.asarray(mf.make_rdm1())
    assert np.abs(dm - dm_ref).max() < 1e-3
    dip_ref = np.asarray(ref.dip_moment(verbose=0))
    dip = np.asarray(mf.dip_moment(verbose=0))
    assert np.abs(dip - dip_ref).max() < 1e-3


def test_co_energy_and_density():
    """CO/6-31G: energy shift within the fit; measured -2.10e-5 Ha."""
    mol, ref, mf = _run(CO)
    d = mf.e_tot - ref.e_tot
    assert abs(d) < 1e-4, f'energy shift {d:.3e} Ha exceeds the fit scale'
    dm_ref = np.asarray(ref.make_rdm1())
    dm = np.asarray(mf.make_rdm1())
    assert np.abs(dm - dm_ref).max() < 1e-3


# ---------------------------------------------------------------------------
# Fock assembly
# ---------------------------------------------------------------------------
def test_get_jk_matches_layer(h2o):
    """J and K equal the layer results at the same density."""
    mol, ref, mf = h2o
    pmol = mol.to_pyscf()
    dm = np.asarray(ref.make_rdm1())
    vj, vk = mf.get_jk(mol, dm)
    j_ref = ri_scf.ri_j_fock(pmol, dm)
    k_ref = ri_scf.exact_k_fock(pmol, dm)
    assert np.abs(np.asarray(vj) - j_ref).max() < 1e-10
    assert np.abs(np.asarray(vk) - k_ref).max() < 1e-8


def test_get_jk_respects_switches(h2o):
    """with_j/with_k select the two branches independently."""
    mol, ref, mf = h2o
    dm = np.asarray(ref.make_rdm1())
    vj_only, vk_none = mf.get_jk(mol, dm, with_k=False)
    assert vk_none is None
    assert np.abs(np.asarray(vj_only)
                  - ri_scf.ri_j_fock(mol.to_pyscf(), dm)).max() < 1e-10
    vj_none, vk_only = mf.get_jk(mol, dm, with_j=False)
    assert vj_none is None
    assert np.abs(np.asarray(vk_only)
                  - ri_scf.exact_k_fock(mol.to_pyscf(), dm)).max() < 1e-8


def test_ri_j_does_not_form_dense_eri(h2o):
    """The Coulomb branch alone leaves pyscfad's dense ERI cache empty."""
    mol, ref, mf = h2o
    dm = np.asarray(ref.make_rdm1())
    mf.reset()
    assert mf._eri is None
    mf.get_jk(mol, dm, with_k=False)
    assert mf._eri is None


def test_batched_density(h2o):
    """A spin-stacked density is mapped spin by spin."""
    mol, ref, mf = h2o
    dm = np.asarray(ref.make_rdm1()) * 0.5
    stacked = np.stack([dm, dm])
    vj, _ = mf.get_jk(mol, stacked, with_k=False)
    j_ref = ri_scf.ri_j_fock(mol.to_pyscf(), dm)
    assert np.asarray(vj).shape == stacked.shape
    assert np.abs(np.asarray(vj)[0] - j_ref).max() < 1e-12
    assert np.abs(np.asarray(vj)[0] - np.asarray(vj)[1]).max() < 1e-12


def test_auxbasis_option_is_used():
    """A named auxiliary basis reaches the factor build."""
    mol = _mol(H2O)
    default = drv.rijonx_scf(mol)
    alternate = drv.rijonx_scf(mol, auxbasis='cc-pvdz-ri')
    assert alternate.converged and default.converged
    assert (alternate.ri_state(mol).naux
            != default.ri_state(mol).naux)


def test_range_separated_coulomb_is_refused(h2o):
    mol, ref, mf = h2o
    with pytest.raises(NotImplementedError):
        mf.get_jk(mol, np.asarray(ref.make_rdm1()), omega=0.3)


# ---------------------------------------------------------------------------
# Open shell and ECP
# ---------------------------------------------------------------------------
def test_rohf_oh_radical():
    """OH doublet: energy shift within the fit; measured -2.73e-5 Ha."""
    mol, ref, mf = _run(OH, spin=1)
    d = mf.e_tot - ref.e_tot
    assert abs(d) < 1e-4, f'energy shift {d:.3e} Ha exceeds the fit scale'
    dip_ref = np.asarray(ref.dip_moment(verbose=0))
    dip = np.asarray(mf.dip_moment(verbose=0))
    assert np.abs(dip - dip_ref).max() < 1e-3


def test_ecp_hi_energy():
    """HI with a def2-svp ECP on iodine: the ECP branch runs end to end.

    The energy shift is the J fit; measured -2.56e-5 Ha.  The identity
    against ``rijonx_energy`` also checks that the core-Hamiltonian
    helper carries the ECP scalar term.
    """
    mol, ref, mf = _run(HI, basis='def2-svp', ecp='def2-svp')
    assert mol.has_ecp()
    d = mf.e_tot - ref.e_tot
    assert abs(d) < 1e-4, f'energy shift {d:.3e} Ha exceeds the fit scale'
    dm_ref = np.asarray(ref.make_rdm1())
    dm = np.asarray(mf.make_rdm1())
    assert np.abs(dm - dm_ref).max() < 1e-3
    pmol = mol.to_pyscf()
    c = np.asarray(mf.mo_coeff)[:, :mol.nelectron // 2]
    assert abs(ri_scf.rijonx_energy(pmol, c) - mf.e_tot) < 1e-9


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
def test_ri_j_coordinate_jvp_matches_skeleton_gradient(h2o):
    """The custom JVP of the RI-J map equals the analytic RI-J gradient."""
    import jax
    import jax.numpy as jnp
    mol, ref, mf = h2o
    pmol = mol.to_pyscf()
    nocc = mol.nelectron // 2
    c = np.asarray(ref.mo_coeff)[:, :nocc]
    dm = jnp.asarray(np.asarray(ref.make_rdm1()))
    state = mf.ri_state(mol)
    fock = state.fock

    def e_j(m):
        return 0.5 * jnp.sum(dm * fock(m, dm))

    g = np.asarray(jax.grad(e_j)(mol).coords)
    g_ref = ri_scf.ri_j_gradient(pmol, c)
    assert np.abs(g - g_ref).max() < 1e-8


def test_scf_gradient_matches_finite_difference(h2o):
    """Implicit-difference SCF gradient equals central differences of E_SCF."""
    from pyscfad import gto
    mol, ref, mf = h2o
    g = np.asarray(mf.energy_grad().coords)

    step = 1.0e-3
    coords = np.asarray(mol.atom_coords())
    fd = np.zeros_like(g)
    for atom in range(mol.natm):
        for axis in range(3):
            values = []
            for sign in (1.0, -1.0):
                shifted = coords.copy()
                shifted[atom, axis] += sign * step
                m = gto.Mole()
                m.atom = [(mol.atom_pure_symbol(i), shifted[i])
                          for i in range(mol.natm)]
                m.basis = '6-31g'
                m.unit = 'Bohr'
                m.verbose = 0
                m.build()
                values.append(drv.rijonx_scf(m).e_tot)
            fd[atom, axis] = (values[0] - values[1]) / (2.0 * step)
    assert np.abs(g - fd).max() < 1e-5, (
        f'gradient vs FD {np.abs(g - fd).max():.3e} Eh/Bohr')


def test_explicit_gradient_matches_implicit(h2o):
    """The overlap-completed skeleton gradient equals the implicit gradient."""
    mol, ref, mf = h2o
    g_auto = np.asarray(mf.energy_grad().coords)
    g_exp = drv.rijonx_scf_gradient(mf)
    assert np.abs(g_auto - g_exp).max() < 1e-6, (
        f'explicit vs implicit {np.abs(g_auto - g_exp).max():.3e} Eh/Bohr')


def test_ecp_gradient_matches_finite_difference():
    """HI/def2-svp ECP gradient: explicit and implicit paths against FD.

    The ECP scalar derivative enters the core-Hamiltonian gradient and
    the auxiliary-center derivative enters the RI-J factors; both are
    exercised here.  Measured deviation 3.0e-7 Eh/Bohr.
    """
    from pyscfad import gto
    mol, ref, mf = _run(HI, basis='def2-svp', ecp='def2-svp')
    g_exp = drv.rijonx_scf_gradient(mf)
    g_imp = np.asarray(mf.energy_grad().coords)

    step = 2.0e-3
    coords = np.asarray(mol.atom_coords())
    fd = np.zeros((mol.natm, 3))
    for atom in range(mol.natm):
        for axis in range(3):
            values = []
            for sign in (1.0, -1.0):
                shifted = coords.copy()
                shifted[atom, axis] += sign * step
                m = gto.Mole()
                m.atom = [(mol.atom_pure_symbol(i), shifted[i])
                          for i in range(mol.natm)]
                m.basis = 'def2-svp'
                m.ecp = 'def2-svp'
                m.unit = 'Bohr'
                m.verbose = 0
                m.build()
                values.append(drv.rijonx_scf(m).e_tot)
            fd[atom, axis] = (values[0] - values[1]) / (2.0 * step)
    assert np.abs(g_exp - fd).max() < 1e-5
    assert np.abs(g_imp - fd).max() < 1e-5


def test_explicit_gradient_refuses_open_shell():
    mol = _mol(OH, spin=1)
    ref = _reference(mol)
    mf = drv.rijonx_scf(mol)
    with pytest.raises(NotImplementedError):
        drv.rijonx_scf_gradient(mf)
    assert ref is not None
