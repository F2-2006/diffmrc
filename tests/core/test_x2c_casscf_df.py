"""Contracts of the density-fitted spin-free X2C CASSCF response layer.

The layer covers the scalar (spin-free) limit of the two-component X2C
Hamiltonian: ``SpinFreeX2CHelper`` for the one-electron operator and the
non-relativistic Coulomb interaction in the density-fitted spatial basis.
The j-adapted two-component basis of ``core.x2c_df`` carries no spin
label, so its complex CI is a separate design (see the module docstring).

Contracts, each with the reference it is checked against:

  * the spatial DF RHF against pyscf's density-fitted sfx2c1e SCF;
  * the determinant CI against a frozen-core FCI on the same integrals;
  * the 1- and 2-RDM energies against the CI eigenvalue;
  * the generalized Fock against the SCF Fock at the determinant limit;
  * the analytic orbital gradient against a finite difference through
    the sequential Givens product;
  * the CASSCF energy and gradient against pyscf's density-fitted
    sfx2c1e CASSCF (energy and analytic gradient, an independent program);
  * the analytic nuclear gradient against a central difference of
    E(theta*, R) along the envelope-theorem path (the reference orbitals
    follow the geometry, the rotation parameters stay at the converged
    point), plus translation and rotation invariance;
  * the state-averaged state-specific gradient through the state-average
    sum rule: the weighted state gradients must reproduce pyscf's
    independent SA-CASSCF gradient;
  * the N = 1 limit of the state-averaged path against the single-state
    result.

The full re-optimization at a displaced geometry is not used as a
reference: H2O has a second orbital minimum in the same theta space and
the displaced optimization can land there.  See ``df_x2c_energies_at``.

The fixture is H2O/6-31G with def2-universal-jkfit, CAS(2,2) with four
core spatial orbitals (ten electrons).
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import df, fci, gto, mcscf, scf  # noqa: E402
from pyscf.x2c import sfx2c1e  # noqa: E402

from diffmrc.core import x2c_casscf_df as xc  # noqa: E402

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'
NCORE, NACT, NELE = 4, 2, 2


def _mol(coords_bohr=None):
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    if coords_bohr is not None:
        mol.set_geom_(coords_bohr, unit='Bohr')
    return mol


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-30))


def _invariance(grad, mol):
    r = mol.atom_coords()
    trans = np.abs(grad.sum(axis=0)).max() / max(np.abs(grad).max(), 1e-30)
    rot = np.zeros(3)
    for ia in range(mol.natm):
        rot += np.cross(r[ia], grad[ia])
    rot = np.abs(rot).max() / max(np.abs(grad).max() * np.abs(r).max(),
                                  1e-30)
    return float(trans), float(rot)


def _envelope_fd(res, root, step=1e-3):
    """Central difference of E(root; theta*, R) along the envelope path."""
    mol = res.mol
    c0 = mol.atom_coords()
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            c = c0.copy()
            c[ia, x] += step
            ep = xc.df_x2c_energies_at(_mol(c), res)[root]
            c = c0.copy()
            c[ia, x] -= step
            em = xc.df_x2c_energies_at(_mol(c), res)[root]
            fd[ia, x] = (ep - em) / (2 * step)
    return fd


@pytest.fixture(scope='module')
def water():
    mol = _mol()
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUX)
    factors = xc.spatial_df_factors(mol, auxmol=auxmol)
    h = xc.sfx2c_hcore(mol)
    s = mol.intor('int1e_ovlp')
    return mol, auxmol, factors, h, s


@pytest.fixture(scope='module')
def rhf(water):
    mol, auxmol, factors, _, _ = water
    return xc.df_x2c_rhf(mol, factors=factors)


@pytest.fixture(scope='module')
def casscf(water):
    mol = water[0]
    return xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                            g_conv=1e-6, max_cycle=400)


@pytest.fixture(scope='module')
def pyscf_casscf(water):
    mol = water[0]
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    return mc


@pytest.fixture(scope='module')
def pyscf_sa(water):
    mol = water[0]
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    return mc


def test_hcore_matches_pyscf(water):
    mol = water[0]
    h = xc.sfx2c_hcore(mol)
    ref = sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)
    assert np.abs(h - ref).max() < 1e-12
    assert np.abs(h - h.T).max() < 1e-12


def test_hcore_derivative_vs_fd(water):
    mol = water[0]
    dh = xc.sfx2c_hcore_derivatives(mol)
    c0 = mol.atom_coords()
    step = 1e-3
    err = 0.0
    scale = np.abs(dh).max()
    for ia in range(mol.natm):
        for x in range(3):
            cp = c0.copy()
            cp[ia, x] += step
            hp = xc.sfx2c_hcore(_mol(cp))
            cm = c0.copy()
            cm[ia, x] -= step
            hm = xc.sfx2c_hcore(_mol(cm))
            fd = (hp - hm) / (2 * step)
            err = max(err, np.abs(fd - dh[ia, x]).max())
    assert err / scale < 1e-5


def test_df_rhf_matches_pyscf(water, rhf):
    mol, auxmol, _, _, _ = water
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert rhf.converged
    assert abs(rhf.e_tot - mf.e_tot) < 1e-9


def test_ci_matches_frozen_core_fci(water, rhf):
    mol, auxmol, factors, h, s = water
    h_mo, w_mo, eri_ca = xc._mo_integrals(rhf.mo_coeff, h, factors,
                                          NCORE, NACT)
    nm = NCORE + NACT
    w, v, dets = xc.ci_eigensystem(h_mo[:nm, :nm], eri_ca, NCORE, NACT,
                                   NELE, nroots=3)
    h_act = h_mo[NCORE:nm, NCORE:nm].copy()
    eri_act = eri_ca[NCORE:, NCORE:, NCORE:, NCORE:].copy()
    e_core = 2 * np.trace(h_mo[:NCORE, :NCORE])
    for i in range(NCORE):
        for j in range(NCORE):
            e_core += (2 * eri_ca[i, i, j, j] - eri_ca[i, j, j, i])
    for i in range(NCORE):
        h_act += 2 * eri_ca[NCORE:nm, NCORE:nm, i, i]
        h_act -= eri_ca[NCORE:nm, i, i, NCORE:nm]
    nalpha = (2 * NCORE + NELE) // 2 - NCORE
    e_fci = fci.direct_spin1.kernel(h_act, eri_act, NACT,
                                    (nalpha, nalpha), nroots=3)[0]
    assert abs(w[0] - (e_core + e_fci[0])) < 1e-10
    assert np.linalg.norm(v[:, 0]) == pytest.approx(1.0, abs=1e-12)


def test_ci_rdm_energy(water, rhf):
    mol, auxmol, factors, h, s = water
    h_mo, w_mo, eri_ca = xc._mo_integrals(rhf.mo_coeff, h, factors,
                                          NCORE, NACT)
    nm = NCORE + NACT
    w, v, dets = xc.ci_eigensystem(h_mo[:nm, :nm], eri_ca, NCORE, NACT,
                                   NELE, nroots=1)
    g, G = xc.ci_rdms(v[:, 0], dets, 2 * nm, 2 * NCORE + NELE)
    g_sp, G_pair = xc.spin_summed_rdms(g, G, nm)
    e1 = np.einsum('pq,pq->', g_sp, h_mo[:nm, :nm])
    e2 = 0.5 * np.einsum('pqrs,pqrs->', G_pair, eri_ca)
    assert abs(e1 + e2 - w[0]) < 1e-11


def test_generalized_fock_determinant_limit(water, rhf):
    mol, auxmol, factors, h, s = water
    h_mo, w_mo, eri_ca = xc._mo_integrals(rhf.mo_coeff, h, factors,
                                          NCORE, NACT)
    nm = NCORE + NACT
    g = np.zeros((2 * nm, 2 * nm))
    nocc = 2 * NCORE + NELE
    for i in range(nocc):
        g[i, i] = 1.0
    G = (np.einsum('pq,rs->pqrs', g, g) - np.einsum('ps,rq->pqrs', g, g))
    g_sp, G_pair = xc.spin_summed_rdms(g, G, nm)
    F = xc.generalized_fock(g_sp, G_pair, h_mo[:nm, :nm],
                            w_mo[:nm, :nm, :])
    d = np.zeros((nm, nm))
    for i in range(NCORE + NELE // 2):
        d[i, i] = 1.0
    rho = np.einsum('ijP,ji->P', w_mo[:nm, :nm], d)
    j = np.einsum('ijP,P->ij', w_mo[:nm, :nm], rho)
    k = np.zeros_like(d)
    for p in range(w_mo.shape[2]):
        wp = w_mo[:nm, :nm, p]
        k += wp @ d @ wp
    f_ref = h_mo[:nm, :nm] + 2 * j - k
    occ = NCORE + NELE // 2
    assert np.abs(0.5 * F[:occ, :occ] - f_ref[:occ, :occ]).max() < 1e-11


def test_theta_gradient_vs_fd(water, rhf):
    mol, auxmol, factors, h, s = water
    nmo = h.shape[0]
    pairs = xc.orbital_pairs(NCORE, NACT, nmo)
    theta = np.zeros(len(pairs))
    theta[0] = 5e-2
    theta[5] = -7e-2
    E, g = xc._objective(theta, rhf.mo_coeff, h, factors, NCORE, NACT,
                         NELE, pairs, 1, np.array([1.0]), 0)
    step = 1e-5
    fd = np.zeros(len(pairs))
    for k in range(len(pairs)):
        tp = theta.copy()
        tp[k] += step
        tm = theta.copy()
        tm[k] -= step
        ep = xc._objective(tp, rhf.mo_coeff, h, factors, NCORE, NACT,
                           NELE, pairs, 1, np.array([1.0]), 0)[0]
        em = xc._objective(tm, rhf.mo_coeff, h, factors, NCORE, NACT,
                           NELE, pairs, 1, np.array([1.0]), 0)[0]
        fd[k] = (ep - em) / (2 * step)
    assert _rel(g, fd) < 1e-4


def test_casscf_matches_pyscf_energy_and_gradient(casscf, pyscf_casscf):
    assert casscf.gnorm < 1e-4
    assert abs(casscf.e_tot - pyscf_casscf.e_tot) < 1e-5
    g_ours, _ = xc.df_x2c_casscf_gradient(casscf.mol, casscf)
    g_ref = pyscf_casscf.nuc_grad_method().kernel()
    assert np.abs(g_ours - g_ref).max() / np.abs(g_ref).max() < 1e-4


def test_casscf_gradient_against_envelope_fd(casscf):
    grad, parts = xc.df_x2c_casscf_gradient(casscf.mol, casscf)
    fd = _envelope_fd(casscf, casscf.target)
    assert _rel(grad, fd) < 1e-4
    assert np.abs(grad - fd).max() < 5e-6


def test_casscf_gradient_invariance(casscf):
    grad, _ = xc.df_x2c_casscf_gradient(casscf.mol, casscf)
    trans, rot = _invariance(grad, casscf.mol)
    assert trans < 1e-6
    assert rot < 1e-6


def test_sa_n1_reduces_to_single_state(water, casscf):
    mol = water[0]
    sa = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                          nroots=1, g_conv=1e-6, max_cycle=400)
    g_sa, _ = xc.df_x2c_casscf_gradient(mol, sa)
    g_ss, _ = xc.df_x2c_casscf_gradient(mol, casscf)
    assert np.abs(sa.mo_coeff - casscf.mo_coeff).max() < 1e-5
    assert np.abs(g_sa - g_ss).max() < 1e-6


def test_sa_state_gradient_satisfies_sum_rule(water, pyscf_sa):
    """The weighted state gradients reproduce pyscf's SA-CASSCF gradient.

    The SA gradient is the weighted average of the state gradients, and
    pyscf computes exactly that average with its own analytic response.
    The individual re-optimized reference is not usable on H2O: a
    displaced SA optimization moves |dtheta| ~ 0.2, i.e. to the second
    orbital minimum.
    """
    mol = water[0]
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           nroots=2, target=0, g_conv=1e-6, max_cycle=400)
    g0, i0 = xc.df_x2c_sa_state_gradient(mol, res, state=0,
                                         return_info=True)
    g1, i1 = xc.df_x2c_sa_state_gradient(mol, res, state=1,
                                         return_info=True)
    avg = 0.5 * (g0 + g1)
    g_ref = pyscf_sa.nuc_grad_method().kernel()
    assert np.abs(avg - g_ref).max() < 1e-5
    assert _rel(avg, g_ref) < 1e-3
    assert i0['n_response'] > 0
    # the response terms must sum to zero over the average
    # z_0 + z_1 = -H^{-1} (g_0 + g_1) = -2 H^{-1} g_SA, so the sum is
    # bounded by the SA stationarity residual (measured 2e-7), not zero.
    resp = 0.5 * (i0['response_grad'] + i1['response_grad'])
    assert np.abs(resp).max() < 1e-5
    # the direct term is a smooth envelope-path difference: stable in h
    d_a = i0['direct']
    d_b = _envelope_fd(res, 0)
    assert np.abs(d_a - d_b).max() < 1e-5
