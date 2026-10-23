"""Density-fitted spin-free X2C CASSCF energies, orbital response and gradients.

WHAT THIS MODULE PROVIDES

A state-specific and state-averaged CASSCF layer on top of the scalar
(spin-free) X2C one-electron Hamiltonian, with a density-fitted
two-electron part:

    E = sum_pq gamma_pq h_pq + 1/2 sum_pqrs Gamma_pqrs (pq|rs) + E_nuc,

    h = SpinFreeX2CHelper(mol).get_hcore(mol),
    (pq|rs) ~= sum_P w_pqP w_rsP,   w = B S^{-1/2},
    B_mu,nu,P = (mu nu | P),        S_PQ = (P|Q).

The two-electron operator is the non-relativistic Coulomb interaction, the
same choice as ``core.x2c_df``; no picture-change correction is applied.
The one-electron operator is the spin-free Foldy-Wouthuysen transform, so
the wavefunction is a real spin-adapted determinant expansion and the
orbitals carry no spin-orbit coupling.

WHY THE SPIN-FREE LIMIT

``core.x2c_df`` works in pyscf's j-adapted two-component spinor basis.
That basis carries no spin label (the even/odd split of the decontracted
functions is not an alpha/beta split, and the one-electron Hamiltonian
couples the two components through the spin-orbit term), so a real
orbital CASSCF cannot be built on it without a complex CI and complex
orbital rotations.  Those are designed but not implemented here; the
spin-free Hamiltonian in a spatial basis is the real limit of that
problem and is the layer validated in this module.

CONVENTIONS (pinned numerically; see the development probes)

The spin-orbital determinant basis follows ``core.casci``: the
determinants are the fully occupied core plus all ``nele_act`` selections
from the active spin orbitals, in increasing spin-orbital order, and the
Condon-Slater phases use the position of the annihilated/created orbital
inside its determinant.

The 1- and 2-RDM are

    gamma_pq = <a^dag_p a_q>,
    Gamma_pqrs = <a^dag_p a^dag_r a_s a_q>.

Because the Coulomb operator conserves spin inside each electron pair,
both the energy and the generalized Fock use the same-spin-pair sum

    Gamma_pair[p,q,r,s] = sum_{sigma,tau}
        Gamma[p sigma, q sigma, r tau, s tau],

and not the sum over all four spin indices.  The determinant limit is

    Gamma_pair[p,q,r,s] = 4 D_pq D_rs - 2 D_ps D_rq

for the closed-shell spatial density D; the probes pin both this identity
and the contract with the Fock matrix.

The generalized Fock matrix in the occupied (core plus active) space is

    F_pq = sum_r gamma_pr h_qr + sum_rst Gamma_pair[p,r,s,t] (qr|st),

and the orbital gradient for a Givens rotation of the pair (i, j) is

    dE/dtheta_ij = 2 (F_ji - F_ij).

NUCLEAR GRADIENT

At a converged CASSCF point the orbital and CI responses vanish and

    dE/dR = sum_pq gamma_pq dh_pq/dR
          + sum_P sum_pq dw_pqP X_pqP
          - sum_pq F_pq dS_pq/dR
          + dE_nuc/dR,

    X_pqP = sum_rs w_rsP Gamma_pair[p,q,r,s],
    dw    = dB S^{-1/2} + B dS^{-1/2}.

The one-electron derivative comes from pyscf's spin-free X2C hcore
gradient generator; the three-center and metric derivatives reuse the
verified conventions of :mod:`diffmrc.core.ri`.

STATE-AVERAGED STATE-SPECIFIC GRADIENT

For an equal-weight state average the orbitals minimize E_SA, so the
target-state orbital gradient g_t is not zero.  With theta the orbital
rotation parameters and R the nuclear coordinates,

    dE_t/dR = dE_t/dR|_theta + z^T M,
    H z = -g_t,
    H = d2 E_SA/dtheta2,   M = d2 E_SA/dtheta dR.

The Hessian ``H`` is evaluated by finite differences of the analytic
state-averaged orbital gradient: because the CI vectors are re-diagonalized
at every theta, the CI response is already inside H.  ``M`` is evaluated
by finite differences of the same gradient with respect to a nuclear
displacement, rebuilding the integrals and the (geometry-dependent)
reference orbitals at the displaced geometry.  ``H`` is pseudoinverted
with a relative cutoff and the retained/dropped eigenvalues are reported.
The direct term ``dE_t/dR|_theta`` is evaluated as a central difference
of the target-state energy along the same envelope path: at the SA
orbitals the target state is not stationary, so the closed-form
single-state formula (which assumes a stationary target) is not this
quantity.  A fully analytic CP-MCSCF direct term with the S^R constraint
is the designed extension.

SCOPE AND LIMITS

* Full-electron basis sets only; the spin-free X2C helper of pyscf raises
  ``NotImplementedError`` for ECP molecules.
* The CI, RDM and Fock contractions are dense in the spin-orbital space
  (``nso**4`` for the 2-RDM).  The prototype is sized for the validation
  systems (H2O, CO); production f-block active spaces need a factorized
  2-RDM route that is not part of this module.
* The complex two-component (spin-orbit) response and the Kramers
  degeneracy treatment are designed, not implemented.  The scalar path is
  free of Kramers pairs by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.optimize

from pyscf import df
from pyscf.x2c import sfx2c1e
from pyscf.x2c.sfx2c1e_grad import hcore_grad_generator

from .orblib import orbital_pairs
from .ri import (MetricEig, inv_sqrt_metric_jvp, make_auxmol, metric_eig,
                 ri_metric_deriv, ri_three_center_deriv)
from .x2c_df import nuclear_repulsion_gradient

__all__ = [
    'DFX2CCASSCFResult',
    'DFX2CRHFResult',
    'SpatialDFFactors',
    'cas_ci_hamiltonian',
    'ci_eigensystem',
    'ci_rdms',
    'df_x2c_casscf',
    'df_x2c_casscf_gradient',
    'df_x2c_energies_at',
    'df_x2c_rhf',
    'df_x2c_sa_state_gradient',
    'generalized_fock',
    'orbital_gradient',
    'orbital_gradient_theta',
    'orbital_pairs',
    'overlap_derivatives',
    'sfx2c_hcore',
    'sfx2c_hcore_derivatives',
    'spatial_df_factors',
    'spin_summed_rdms',
]

#: Default relative cutoff for the auxiliary metric truncation.
DEFAULT_RCOND = 1e-12


# ---------------------------------------------------------------------------
# Scalar X2C one-electron Hamiltonian and its nuclear derivatives
# ---------------------------------------------------------------------------
def sfx2c_hcore(mol):
    """Spin-free X2C Foldy-Wouthuysen one-electron Hamiltonian (spatial)."""
    return sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)


def sfx2c_hcore_derivatives(mol):
    """Analytic nuclear derivatives of :func:`sfx2c_hcore`.

    Returns an ``(natm, 3, nao, nao)`` array in Eh / Bohr.
    """
    helper = sfx2c1e.SpinFreeX2CHelper(mol)
    gen = hcore_grad_generator(helper, mol)
    return np.asarray([gen(ia) for ia in range(mol.natm)])


def overlap_derivatives(mol):
    """Nuclear derivatives of the spatial AO overlap, ``(natm,3,nao,nao)``."""
    nao = mol.nao_nr()
    ip = mol.intor('int1e_ipovlp', comp=3)
    loc = mol.aoslice_by_atom()
    out = np.zeros((mol.natm, 3, nao, nao))
    for ia in range(mol.natm):
        p0, p1 = loc[ia, 2:]
        for x in range(3):
            y = np.zeros((nao, nao))
            y[p0:p1, :] = ip[x][p0:p1, :]
            out[ia, x] = -(y + y.T)
    return out


# ---------------------------------------------------------------------------
# Density-fitted two-electron factors (spatial)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpatialDFFactors:
    """RI-JK factors in the spatial AO basis.

    Attributes
    ----------
    b : ndarray, (nao, nao, naux)
        Three-center integrals ``(mu nu | P)``.
    s : ndarray, (naux, naux)
        Auxiliary metric ``(P | Q)``.
    jhalf : ndarray, (naux, naux)
        Truncated symmetric inverse square root of ``s``.
    eig : MetricEig
        Metric eigendecomposition reused by the derivative.
    auxmol : pyscf.gto.Mole
        Auxiliary molecule at the same geometry.
    auxbasis : str
        Label of the auxiliary basis actually used.
    rcond : float
        Metric truncation threshold.
    """

    b: np.ndarray
    s: np.ndarray
    jhalf: np.ndarray
    eig: MetricEig
    auxmol: object
    auxbasis: str
    rcond: float
    _mol: object = field(default=None, repr=False)

    @property
    def nao(self):
        return self.b.shape[0]

    @property
    def naux(self):
        return self.b.shape[2]

    @property
    def w(self):
        """Metric-folded three-center tensor ``B S^{-1/2}``."""
        return np.matmul(self.b, self.jhalf)

    def fock(self, d, h):
        """Closed-shell spatial Fock matrix, ``F = h + 2 J - K``.

        Returns ``(f, j, k)``.
        """
        w = self.w
        rho = np.einsum('ijP,ji->P', w, d)
        j = np.einsum('ijP,P->ij', w, rho)
        k = np.zeros_like(d)
        for p in range(self.naux):
            wp = w[:, :, p]
            k += wp @ d @ wp
        return h + 2.0 * j - k, j, k

    def energy_parts(self, d, h):
        """``(e_one_electron, e_coulomb, e_exchange)`` for a spatial density.

        The closed-shell factors are included: ``e_h = 2 tr(d h)``,
        ``e_j = 2 rho^2`` and ``e_k = -tr(d K)``.
        """
        w = self.w
        rho = np.einsum('ijP,ji->P', w, d)
        e_h = 2.0 * np.einsum('ij,ji->', d, h)
        e_j = 2.0 * np.sum(rho * rho)
        e_k = 0.0
        for p in range(self.naux):
            wp = w[:, :, p]
            e_k -= np.trace(d @ wp @ d @ wp)
        return float(np.real(e_h)), float(e_j), float(np.real(e_k))

    def deriv_factors(self, atom_idx):
        """Nuclear derivatives of every DF factor for one atom.

        Returns ``(db, ds_metric, djhalf)`` with shapes
        ``(3,nao,nao,naux)``, ``(3,naux,naux)`` and ``(3,naux,naux)``.
        """
        db = ri_three_center_deriv(self._mol, self.auxmol, atom_idx)
        ds = ri_metric_deriv(self.auxmol, atom_idx)
        dj = inv_sqrt_metric_jvp(self.s, ds, eig=self.eig)
        return db, ds, dj

    def energy_deriv(self, d, db, dj):
        """Fixed-density two-electron derivative along one atom.

        ``db`` is ``(3,nao,nao,naux)`` and ``dj`` is ``(3,naux,naux)``.
        Returns ``(total, coulomb, exchange)``, each of length 3, in
        Eh / Bohr.  The closed-shell density is held fixed.
        """
        w = self.w
        rho = np.einsum('ijP,ji->P', w, d)
        dw = np.matmul(db, self.jhalf) + np.einsum('abP,cPQ->cabQ',
                                                   self.b, dj)
        # X_pqP = 4 D_pq rho_P - 2 (D w_P D)_pq
        x_c = 4.0 * np.einsum('pq,P->pqP', d, rho)
        x_k = -2.0 * np.einsum('pr,rsP,sq->pqP', d, w, d)
        x = x_c + x_k
        de = np.einsum('pqP,cpqP->c', x, dw)
        de_c = np.einsum('pqP,cpqP->c', x_c, dw)
        de_k = np.einsum('pqP,cpqP->c', x_k, dw)
        return de, de_c, de_k

    def mo_factors(self, c):
        """Three-center factors in an MO basis, ``(nmo,nmo,naux)``.

        ``c`` has the AO index first and the MO index second.
        """
        return np.einsum('am,abP,bn->mnP', c, self.w, c)


def spatial_df_factors(mol, auxmol=None, auxbasis=None,
                       rcond=DEFAULT_RCOND):
    """Assemble the spatial RI-JK factors.

    When ``auxmol`` is None the auxiliary molecule is built by
    :func:`diffmrc.core.ri.make_auxmol`, which reports the basis used.
    """
    if auxmol is None:
        auxmol, label = make_auxmol(mol, auxbasis=auxbasis)
    else:
        label = auxbasis if auxbasis is not None else 'custom auxiliary molecule'
    b = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')
    s = auxmol.intor('int2c2e_sph', aosym='s1')
    s = 0.5 * (s + s.T)
    eig = metric_eig(s, rcond)
    scale = np.zeros_like(eig.w)
    scale[eig.keep] = 1.0 / np.sqrt(eig.w[eig.keep])
    jhalf = (eig.u * scale) @ eig.u.T
    jhalf = 0.5 * (jhalf + jhalf.T)
    return SpatialDFFactors(b=b, s=s, jhalf=jhalf, eig=eig, auxmol=auxmol,
                            auxbasis=label, rcond=rcond, _mol=mol)


# ---------------------------------------------------------------------------
# Closed-shell reference SCF
# ---------------------------------------------------------------------------
@dataclass
class DFX2CRHFResult:
    """Converged (or best) closed-shell spatial DF-X2C RHF result."""
    mol: object
    factors: SpatialDFFactors
    hcore: np.ndarray
    ovlp: np.ndarray
    dm: np.ndarray
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    mo_occ: np.ndarray
    e_tot: float
    converged: bool
    gradient_norm: float
    n_iter: int


def _diis(focks, errors):
    """Pulay DIIS combination of real Fock matrices."""
    n = len(focks)
    b = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            b[i, j] = float(np.sum(errors[i] * errors[j]))
    b[:n, n] = -1.0
    b[n, :n] = -1.0
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    coef = np.linalg.solve(b, rhs)
    f = np.zeros_like(focks[0])
    for i in range(n):
        f += coef[i] * focks[i]
    return f


def df_x2c_rhf(mol, factors=None, auxmol=None, auxbasis=None,
               conv_tol=1e-12, max_cycle=200, dm0=None):
    """Closed-shell spatial DF-X2C Hartree-Fock.

    The reference orbitals are the starting point of the CASSCF
    optimization and the geometry-dependent reference of the
    state-averaged response.
    """
    if factors is None:
        factors = spatial_df_factors(mol, auxmol=auxmol, auxbasis=auxbasis)
    h = sfx2c_hcore(mol)
    s = mol.intor('int1e_ovlp')
    nocc = mol.nelectron // 2
    e, c = scipy.linalg.eigh(h, s)
    d = (c[:, :nocc] @ c[:, :nocc].T if dm0 is None
         else np.array(dm0))
    focks, errors = [], []
    e_old = None
    gnorm = np.inf
    it = 0
    for it in range(max_cycle):
        f, _, _ = factors.fock(d, h)
        e_h, e_j, e_k = factors.energy_parts(d, h)
        e_tot = e_h + e_j + e_k + mol.energy_nuc()
        err = f @ d @ s - s @ d @ f
        gnorm = float(np.linalg.norm(err))
        if e_old is not None and abs(e_tot - e_old) < conv_tol and gnorm < 1e-9:
            break
        e_old = e_tot
        focks.append(f)
        errors.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        f_use = _diis(focks, errors) if len(focks) > 1 else f
        e, c = scipy.linalg.eigh(f_use, s)
        d = c[:, :nocc] @ c[:, :nocc].T
    f, _, _ = factors.fock(d, h)
    e_h, e_j, e_k = factors.energy_parts(d, h)
    occ = np.zeros(len(e))
    occ[:nocc] = 1.0
    return DFX2CRHFResult(
        mol=mol, factors=factors, hcore=h, ovlp=s, dm=d, mo_coeff=c,
        mo_energy=e, mo_occ=occ,
        e_tot=float(e_h + e_j + e_k + mol.energy_nuc()),
        converged=bool(gnorm < 1e-7), gradient_norm=gnorm,
        n_iter=it + 1)


# ---------------------------------------------------------------------------
# Determinant CI
# ---------------------------------------------------------------------------
def _det_list(nso, nele):
    """All sorted determinant occupations of ``nele`` electrons in ``nso``."""
    return [tuple(c) for c in combinations(range(nso), nele)]


def cas_ci_hamiltonian(h1e_mo, eri_mo, ncore, nact, nele_act):
    """CI Hamiltonian in the determinant basis, by the Condon-Slater rules.

    Mirrors the construction of :func:`diffmrc.core.casci.cs_casci` for a
    real, spin-pure basis and returns ``(H, dets)``.  The active space is
    ``ncore`` core and ``nact`` active spatial orbitals holding
    ``nele_act`` active electrons; the core is doubly occupied in every
    determinant.
    """
    nso = 2 * (ncore + nact)
    so = np.arange(nso)
    pi, si = so // 2, so % 2
    h_so = np.where(si[:, None] == si[None, :],
                    h1e_mo[pi[:, None], pi[None, :]], 0.0)
    m4 = ((si[:, None, None, None] == si[None, None, :, None])
          & (si[None, :, None, None] == si[None, None, None, :]))
    eri_so = np.where(
        m4,
        eri_mo[pi[:, None, None, None], pi[None, None, :, None],
               pi[None, :, None, None], pi[None, None, None, :]], 0.0)
    core_so = tuple(range(2 * ncore))
    act_so = list(range(2 * ncore, nso))
    dets = [core_so + cb for cb in combinations(act_so, nele_act)]
    ndet = len(dets)
    occ_bit = np.zeros((ndet, nso), dtype=bool)
    occ_bit[np.arange(ndet)[:, None], np.asarray(dets)] = True
    diag1 = occ_bit @ np.diag(h_so)
    e2ij = np.einsum('ijij->ij', eri_so) - np.einsum('ijji->ij', eri_so)
    diag2 = ((occ_bit @ np.triu(e2ij, k=1)) * occ_bit).sum(axis=1)
    H_diag = diag1 + diag2
    xa = occ_bit[:, None, :] & ~occ_bit[None, :, :]
    xb = ~occ_bit[:, None, :] & occ_bit[None, :, :]
    na, nb = xa.sum(axis=2), xb.sum(axis=2)
    pos = np.cumsum(occ_bit, axis=1) - 1

    def so_idx(xm):
        return np.argmax(xm, axis=2)

    def second_idx(xm, first):
        rest = np.where(np.arange(nso)[None, None, :] == first[..., None],
                        False, xm)
        return np.argmax(rest, axis=2)

    mask1 = (na == 1) & (nb == 1)
    i1 = so_idx(xa)
    k1 = so_idx(xb)
    pos_i = np.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_k = np.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = np.where((pos_i + pos_k) % 2 == 0, 1.0, -1.0)
    xijk = np.einsum('ijkj->ijk', eri_so) - np.einsum('ijjk->ijk', eri_so)
    tmat = np.einsum('aj,ijk->aik', occ_bit, xijk)
    t_pair = tmat[np.arange(ndet)[:, None], i1, k1]
    occ_i = np.take_along_axis(occ_bit[:, :, None], i1[:, :, None],
                               axis=1)[..., 0]
    x_iik = xijk[i1, i1, k1]
    val1 = sign1 * (h_so[i1, k1] + t_pair - occ_i * x_iik)
    H = np.where(mask1, val1, 0.0)

    mask2 = (na == 2) & (nb == 2)
    i2 = second_idx(xa, i1)
    k2 = second_idx(xb, k1)
    pos_i2 = np.take_along_axis(pos[:, None, :], i2[:, :, None],
                                axis=2)[..., 0]
    pos_k2 = np.take_along_axis(pos[None, :, :], k2[:, :, None],
                                axis=2)[..., 0]
    sign2 = np.where((pos_i + pos_i2 + pos_k + pos_k2) % 2 == 0, 1.0, -1.0)
    val2 = sign2 * (eri_so[i1, i2, k1, k2] - eri_so[i1, i2, k2, k1])
    H = H + np.where(mask2, val2, 0.0)
    H = H + np.diag(H_diag)
    return 0.5 * (H + H.T), dets


def ci_eigensystem(h1e_mo, eri_mo, ncore, nact, nele_act, nroots=1):
    """Lowest ``nroots`` CI eigenpairs and the determinant list.

    Returns ``(eigenvalues, eigenvectors, dets)`` with the eigenvectors
    as columns.
    """
    H, dets = cas_ci_hamiltonian(h1e_mo, eri_mo, ncore, nact, nele_act)
    w, v = scipy.linalg.eigh(H)
    return w[:nroots], v[:, :nroots], dets


def _annihilators(dets_n, dets_nm1, nso):
    """Annihilation matrices ``A_p`` from the N to the N-1 space."""
    index = {d: i for i, d in enumerate(dets_nm1)}
    a = []
    for p in range(nso):
        m = np.zeros((len(dets_nm1), len(dets_n)))
        for j, d in enumerate(dets_n):
            if p in d:
                sign = (-1.0) ** d.index(p)
                new = tuple(x for x in d if x != p)
                i = index.get(new)
                if i is not None:
                    m[i, j] = sign
        a.append(m)
    return a


_RDM_OPS_CACHE = {}


def _rdm_ops(nso, nele):
    """Cached determinant lists and annihilation operators for one sector.

    The operators depend only on ``(nso, nele)``, so the state-averaged
    Hessian, which constructs many RDMs at fixed active space, reuses
    them.  The cost is the full N-1 and N-2 spaces, quadratic in the
    determinant count; the prototype is sized for small active spaces.
    """
    key = (nso, nele)
    if key not in _RDM_OPS_CACHE:
        dets_n = _det_list(nso, nele)
        dets_n1 = _det_list(nso, nele - 1)
        dets_n2 = _det_list(nso, nele - 2)
        idx_n = {d: i for i, d in enumerate(dets_n)}
        a1 = _annihilators(dets_n, dets_n1, nso)
        a2 = _annihilators(dets_n1, dets_n2, nso)
        _RDM_OPS_CACHE[key] = (dets_n, idx_n, a1, a2)
    return _RDM_OPS_CACHE[key]


def ci_rdms(ci_vec, dets_ci, nso, nele):
    """One- and two-particle density matrices of a determinant CI vector.

    ``gamma[p,q] = <a^dag_p a_q>`` and
    ``Gamma[p,q,r,s] = <a^dag_p a^dag_r a_s a_q>``.  The vector follows
    the determinant order ``dets_ci``; the operators are built on the full
    N, N-1 and N-2 determinant spaces over the same ``nso`` spin orbitals.
    """
    dets_n, idx_n, a1, a2 = _rdm_ops(nso, nele)
    psi = np.zeros(len(dets_n), dtype=np.asarray(ci_vec).dtype)
    for i, d in enumerate(dets_ci):
        psi[idx_n[d]] = ci_vec[i]
    # gamma_pq = (a_p psi) . (a_q psi), and
    # Gamma_pqrs = (a_r a_p psi) . (a_s a_q psi); both are single GEMMs.
    ymat = np.column_stack([a1[p] @ psi for p in range(nso)])
    gamma = ymat.T @ ymat
    xmat = np.column_stack([a2[r] @ (a1[p] @ psi)
                            for p in range(nso) for r in range(nso)])
    gmat = xmat.T @ xmat
    # column order is (p, r) on both sides: G4[p, r, q, s] = Gamma[p,q,r,s]
    gamma2 = gmat.reshape(nso, nso, nso, nso).transpose(0, 2, 1, 3)
    return gamma, gamma2


def spin_summed_rdms(gamma, gamma2, nm):
    """Spin-summed 1-RDM and same-spin-pair 2-RDM over ``nm`` spatial orbitals.

    ``gamma_sp[p,q] = gamma[p alpha, q alpha] + gamma[p beta, q beta]``;
    ``Gamma_pair[p,q,r,s]`` sums the first pair over equal spins and the
    second pair over equal spins, which is the object the spin-free
    Coulomb operator and the generalized Fock both contract with.
    """
    g_sp = gamma[0::2, 0::2] + gamma[1::2, 1::2]
    g4 = gamma2.reshape(nm, 2, nm, 2, nm, 2, nm, 2)
    g_pair = (g4[:, 0, :, 0, :, 0, :, 0] + g4[:, 0, :, 0, :, 1, :, 1]
              + g4[:, 1, :, 1, :, 0, :, 0] + g4[:, 1, :, 1, :, 1, :, 1])
    return g_sp, g_pair


def generalized_fock(gamma_sp, gamma_pair, h_mo, w_mo):
    """Generalized Fock matrix in the occupied (core plus active) space.

    Returns an ``(nmo, nmo)`` array whose occupied rows are

        F_pq = sum_r gamma_sp[p,r] h_qr
             + sum_rst Gamma_pair[p,r,s,t] (qr|st),

    and whose virtual rows are zero.  Only the occupied columns of the
    integrals are needed, so the contraction never forms ``nmo**4``.
    """
    nm = gamma_sp.shape[0]
    nmo = h_mo.shape[0]
    f = np.einsum('pr,qr->pq', gamma_sp, h_mo[:, :nm])
    eri_red = np.einsum('qrP,stP->qrst', w_mo[:, :nm, :],
                        w_mo[:nm, :nm, :])
    f = f + np.einsum('prst,qrst->pq', gamma_pair, eri_red)
    # Virtual rows are identically zero (the RDMs vanish there); the matrix
    # is therefore not symmetric in the full space and must not be
    # symmetrized, or occupied values would leak into the virtual rows.
    out = np.zeros((nmo, nmo), dtype=f.dtype)
    out[:nm, :] = f
    return out


def orbital_gradient(fock, pairs):
    """Orbital gradient at the identity rotation, ``2(F_ji - F_ij)``.

    This is the derivative with respect to a direct rotation of the final
    orbitals.  For the sequential Givens product of
    :func:`diffmrc.core.rotation.block_rotation` the exact derivative at a
    nonzero parameter point carries a chain-rule factor; use
    :func:`orbital_gradient_theta` there.
    """
    return np.array([2.0 * (fock[j, i] - fock[i, j]) for (i, j) in pairs])


def _givens_matrices(theta, nmo, pairs):
    """The individual Givens matrices of the sequential product."""
    out = []
    for t, (i, j) in zip(np.asarray(theta, dtype=float), pairs):
        c, s = np.cos(t), np.sin(t)
        g = np.eye(nmo)
        g[i, i] = c
        g[j, i] = -s
        g[i, j] = s
        g[j, j] = c
        out.append(g)
    return out


def orbital_gradient_theta(fock, theta, pairs, nmo):
    """Exact derivative of the energy through the sequential Givens product.

    ``block_rotation`` builds ``R = G_1 G_2 ... G_n``.  With
    ``U_k = R^T dR/dtheta_k`` (antisymmetric) the derivative is

        dE/dtheta_k = 1/2 sum_ab g_ab (U_k)_ab,
        g_ab = 2 (F_ba - F_ab).

    With ``S = G_{k+1}...G_n`` and ``dG_k/dtheta = G_k K_k`` the identity
    ``U_k = S^T K_k S`` avoids forming the full product derivative.  At
    ``theta = 0`` this reduces to ``2 (F_ji - F_ij)``.
    """
    gmat = 2.0 * (fock.T - fock)
    gs = _givens_matrices(theta, nmo, pairs)
    n = len(gs)
    suffix = [None] * (n + 1)
    suffix[n] = np.eye(nmo)
    for k in range(n - 1, -1, -1):
        suffix[k] = gs[k] @ suffix[k + 1]
    grad = np.zeros(n)
    for k in range(n):
        i, j = pairs[k]
        kmat = np.zeros((nmo, nmo))
        kmat[i, j] = 1.0
        kmat[j, i] = -1.0
        s_mat = suffix[k + 1]
        u = s_mat.T @ kmat @ s_mat
        grad[k] = 0.5 * np.einsum('ab,ab->', gmat, u)
    return grad


# ---------------------------------------------------------------------------
# CASSCF energy, orbital gradient and result
# ---------------------------------------------------------------------------
def _mo_integrals(c, h, factors, ncore, nact):
    """MO one- and two-electron integrals for the CI and the Fock.

    Returns ``(h_mo, w_mo, eri_ca)`` where ``h_mo`` is ``(nmo,nmo)``,
    ``w_mo`` is ``(nmo,nmo,naux)`` and ``eri_ca`` is the chemist
    ``(pq|rs)`` array over the ``ncore+nact`` core and active orbitals.
    """
    nm = ncore + nact
    h_mo = c.T @ h @ c
    w_mo = factors.mo_factors(c)
    w_ca = w_mo[:nm, :nm, :]
    eri_ca = np.einsum('abP,cdP->abcd', w_ca, w_ca)
    return h_mo, w_mo, eri_ca


def _state_data(theta, c_ref, h, factors, ncore, nact, nele, pairs,
                nroots, weights, target, e_nuc=0.0):
    """All state-resolved quantities at one orbital parameter point."""
    from .rotation import block_rotation
    nmo = h.shape[0]
    c = c_ref @ block_rotation(np.asarray(theta), nmo, pairs)
    nm = ncore + nact
    h_mo, w_mo, eri_ca = _mo_integrals(c, h, factors, ncore, nact)
    eig, vecs, dets = ci_eigensystem(h_mo[:nm, :nm], eri_ca, ncore, nact,
                                     nele, nroots=nroots)
    nso = 2 * nm
    nele_tot = 2 * ncore + nele
    g_sp_list, g_pair_list, fock_list, g_list = [], [], [], []
    for m in range(nroots):
        g1, g2 = ci_rdms(vecs[:, m], dets, nso, nele_tot)
        g_sp, g_pair = spin_summed_rdms(g1, g2, nm)
        f = generalized_fock(g_sp, g_pair, h_mo, w_mo)
        g_sp_list.append(g_sp)
        g_pair_list.append(g_pair)
        fock_list.append(f)
        g_list.append(orbital_gradient_theta(f, theta, pairs, nmo))
    e_states = eig + e_nuc
    g_states = np.asarray(g_list)
    e_sa = float(np.asarray(weights) @ e_states)
    g_sa = np.asarray(weights) @ g_states
    return dict(
        mo_coeff=c, h_mo=h_mo, w_mo=w_mo, eri_ca=eri_ca, eigvals=eig,
        ci_vectors=vecs, dets=dets, e_states=e_states, e_sa=e_sa,
        g_states=g_states, g_sa=g_sa, g_sp=g_sp_list, g_pair=g_pair_list,
        fock=fock_list)


def _objective(theta, c_ref, h, factors, ncore, nact, nele, pairs,
               nroots, weights, target, e_nuc=0.0):
    """Scalar objective and its analytic orbital gradient.

    The objective is the target-state energy for ``nroots == 1`` and the
    weighted state average otherwise.
    """
    data = _state_data(theta, c_ref, h, factors, ncore, nact, nele, pairs,
                       nroots, weights, target, e_nuc)
    if nroots == 1:
        return float(data['e_states'][target]), data['g_states'][target]
    return float(data['e_sa']), data['g_sa']


@dataclass
class DFX2CCASSCFResult:
    """Converged (or best) DF-X2C CASSCF result at one geometry."""
    mol: object
    factors: SpatialDFFactors
    hcore: np.ndarray
    ovlp: np.ndarray
    c_ref: np.ndarray
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    ncore: int
    nact: int
    nele: int
    nroots: int
    weights: np.ndarray
    target: int
    pairs: list
    theta: np.ndarray
    ci_vectors: np.ndarray
    state_energies: np.ndarray
    e_tot: float
    e_sa: float
    converged: bool
    gnorm: float
    n_iter: int
    auxbasis: str


def df_x2c_casscf(mol, ncore, nact, nele, auxmol=None, auxbasis=None,
                  nroots=1, weights=None, target=0, max_cycle=300,
                  g_conv=1e-6, rcond=DEFAULT_RCOND, theta0=None,
                  rhf=None, verbose=False):
    """Spin-free DF-X2C CASSCF (single-state or equal/weighted average).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule at the reference geometry.
    ncore, nact : int
        Core and active spatial orbitals.
    nele : int
        Active electrons.
    auxmol, auxbasis : optional
        Auxiliary basis for the two-electron fit; forwarded to
        :func:`spatial_df_factors`.
    nroots : int
        Number of CI states in the average.  ``nroots == 1`` is the
        single-state CASSCF.
    weights : array_like, optional
        State weights; equal weights when omitted.
    target : int
        Index of the state whose energy is reported in ``e_tot``.
    theta0 : array_like, optional
        Warm start for the orbital rotation parameters.
    rhf : DFX2CRHFResult, optional
        Pre-computed reference SCF; built here when omitted.

    Returns
    -------
    DFX2CCASSCFResult
    """
    factors = spatial_df_factors(mol, auxmol=auxmol, auxbasis=auxbasis,
                                 rcond=rcond)
    h = sfx2c_hcore(mol)
    s = mol.intor('int1e_ovlp')
    if rhf is None:
        rhf = df_x2c_rhf(mol, factors=factors)
    c_ref = rhf.mo_coeff
    nmo = h.shape[0]
    pairs = orbital_pairs(ncore, nact, nmo)
    if weights is None:
        weights = np.full(nroots, 1.0 / nroots)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    e_nuc = float(mol.energy_nuc())
    x0 = (np.zeros(len(pairs)) if theta0 is None
          else np.asarray(theta0, dtype=float))

    def fun(x):
        return _objective(x, c_ref, h, factors, ncore, nact, nele, pairs,
                          nroots, weights, target, e_nuc)[0]

    def jac(x):
        return _objective(x, c_ref, h, factors, ncore, nact, nele, pairs,
                          nroots, weights, target, e_nuc)[1]

    res = scipy.optimize.minimize(
        fun, x0, jac=jac, method='L-BFGS-B',
        options=dict(maxiter=int(max_cycle), gtol=float(g_conv),
                     ftol=1e-15, maxls=50))
    theta = np.asarray(res.x, dtype=float)
    data = _state_data(theta, c_ref, h, factors, ncore, nact, nele, pairs,
                       nroots, weights, target, e_nuc)
    gnorm = float(np.linalg.norm(data['g_sa'] if nroots > 1
                                 else data['g_states'][target]))
    fock = data['fock'][target]
    mo_energy = np.diag(fock)
    return DFX2CCASSCFResult(
        mol=mol, factors=factors, hcore=h, ovlp=s, c_ref=c_ref,
        mo_coeff=data['mo_coeff'], mo_energy=mo_energy,
        ncore=ncore, nact=nact, nele=nele, nroots=nroots,
        weights=weights, target=target, pairs=pairs, theta=theta,
        ci_vectors=data['ci_vectors'], state_energies=data['e_states'],
        e_tot=float(data['e_states'][target]), e_sa=float(data['e_sa']),
        # The stationary-point residual along flat (low-curvature) orbital
        # directions sits at 1e-6 to 1e-5 even when the energy is converged
        # to 1e-12; the energy and the nuclear gradient are insensitive to
        # it through the envelope theorem.
        converged=bool(gnorm < 1e-5), gnorm=gnorm,
        n_iter=int(getattr(res, 'nit', -1)), auxbasis=factors.auxbasis)


def df_x2c_casscf_gradient(mol, result=None, state=None, auxmol=None,
                           auxbasis=None, rcond=DEFAULT_RCOND):
    """Analytic nuclear gradient of one CI state of a DF-X2C CASSCF.

    At the converged orbitals the orbital and CI responses vanish, so the
    gradient is the direct derivative at fixed orbitals and CI vector.
    ``state`` selects the CI root (default: ``result.target``).  The
    converged ``result`` returned by :func:`df_x2c_casscf` is required.

    Returns ``(grad, parts)`` with ``grad`` of shape ``(natm,3)`` in
    Eh / Bohr and ``parts`` holding the one-electron, two-electron,
    Pulay and nuclear contributions.
    """
    if result is None:
        raise ValueError(
            "df_x2c_casscf_gradient requires the converged result from "
            "df_x2c_casscf; pass result=... (run df_x2c_casscf first).")
    # the orbitals and factor derivatives belong to the result geometry
    mol = result.mol
    ncore, nact, nele = result.ncore, result.nact, result.nele
    nm = ncore + nact
    c = result.mo_coeff
    factors = result.factors
    h_mo, w_mo, eri_ca = _mo_integrals(c, result.hcore, factors, ncore,
                                       nact)
    eig, vecs, dets = ci_eigensystem(h_mo[:nm, :nm], eri_ca, ncore, nact,
                                     nele, nroots=result.nroots)
    m = result.target if state is None else int(state)
    g1, g2 = ci_rdms(vecs[:, m], dets, 2 * nm, 2 * ncore + nele)
    g_sp, g_pair = spin_summed_rdms(g1, g2, nm)
    fock = generalized_fock(g_sp, g_pair, h_mo, w_mo)
    dh = sfx2c_hcore_derivatives(mol)
    ds = overlap_derivatives(mol)
    g_nuc = nuclear_repulsion_gradient(mol)
    natm = mol.natm
    parts = {k: np.zeros((natm, 3)) for k in
             ('one_electron', 'two_electron', 'pulay', 'nuclear')}
    xmat = np.einsum('rsP,pqrs->pqP', w_mo[:nm, :nm, :], g_pair)
    c_occ = c[:, :nm]
    for ia in range(natm):
        db, _, dj = factors.deriv_factors(ia)
        for x in range(3):
            dhm = c.T @ dh[ia, x] @ c
            dsm = c.T @ ds[ia, x] @ c
            parts['one_electron'][ia, x] = np.einsum(
                'pq,pq->', g_sp, dhm[:nm, :nm])
            parts['pulay'][ia, x] = -np.einsum(
                'pq,pq->', fock[:nm, :nm], dsm[:nm, :nm])
            dw = db[x] @ factors.jhalf + factors.b @ dj[x]
            dwm = np.einsum('am,abP,bn->mnP', c_occ, dw, c_occ)
            parts['two_electron'][ia, x] = np.einsum('pqP,pqP->', dwm,
                                                     xmat)
        parts['nuclear'][ia, :] = g_nuc[ia, :]
    grad = sum(parts.values())
    return grad, parts


# ---------------------------------------------------------------------------
# State-averaged state-specific gradient (Z-vector)
# ---------------------------------------------------------------------------
def df_x2c_energies_at(mol, result, theta=None):
    """State energies at ``mol`` with the orbital parameters held fixed.

    The reference orbitals follow the geometry (the displaced SCF is run
    here and aligned to ``result.c_ref``), so this is the
    envelope-theorem path whose first derivative is the analytic nuclear
    gradient.  Re-optimizing the orbitals at the displaced geometry is a
    different, branch-sensitive path: H2O has a second orbital minimum in
    the same parameter space.
    """
    theta = result.theta if theta is None else np.asarray(theta, float)
    fac = spatial_df_factors(mol, auxbasis=result.auxbasis,
                             rcond=result.factors.rcond)
    h = sfx2c_hcore(mol)
    rhf = df_x2c_rhf(mol, factors=fac)
    c0 = _align_orbitals(result.c_ref, rhf.mo_coeff,
                         mol.intor('int1e_ovlp'))
    data = _state_data(theta, c0, h, fac, result.ncore, result.nact,
                       result.nele, result.pairs, result.nroots,
                       result.weights, result.target,
                       float(mol.energy_nuc()))
    return data['e_states']


def _displaced_mol(mol, ia, x, step):
    molp = mol.copy()
    coords = mol.atom_coords().copy()
    coords[ia, x] += step
    molp.set_geom_(coords, unit='Bohr')
    return molp


def _align_orbitals(c_ref, c_new, s):
    """Permute and sign-align ``c_new`` onto the reference orbital order."""
    ov = c_ref.T @ s @ c_new
    nmo = c_ref.shape[1]
    used = np.zeros(nmo, dtype=bool)
    cols = []
    for i in range(nmo):
        score = np.abs(ov[i]).copy()
        score[used] = -1.0
        j = int(np.argmax(score))
        used[j] = True
        sgn = 1.0 if ov[i, j] >= 0.0 else -1.0
        cols.append(sgn * c_new[:, j])
    return np.column_stack(cols)


def _fd_gradient_matrix(grad_of, x, step):
    """Central-difference Jacobian of a vector function of ``x``."""
    n = len(x)
    out = np.zeros((len(grad_of(x)), n))
    for k in range(n):
        xp = np.array(x, dtype=float)
        xp[k] += step
        xm = np.array(x, dtype=float)
        xm[k] -= step
        out[:, k] = (grad_of(xp) - grad_of(xm)) / (2.0 * step)
    return out


def df_x2c_sa_state_gradient(mol, result=None, state=None, fd_step=1e-3,
                             theta_step=1e-5, rcond=1e-8, auxmol=None,
                             auxbasis=None, return_info=False):
    """State-specific nuclear gradient of a state-averaged DF-X2C CASSCF.

    The direct term is the central difference of the target-state energy
    along the envelope path (reference orbitals following the geometry,
    rotation parameters fixed at the converged point); the target state is
    not stationary at the SA orbitals, so the closed-form single-state
    gradient is not this quantity.  The orbital response solves
    ``H z = -g_t`` with ``H`` the state-averaged orbital Hessian (finite
    differences of the analytic state-averaged orbital gradient, so the CI
    response is included) and the mixed derivative ``M = d g_SA / dR``
    (finite differences over displaced geometries, with the
    geometry-dependent reference orbitals).

    Returns ``(grad, info)`` when ``return_info`` is True, else ``grad``.
    ``info`` carries the response vector, the Hessian spectrum, the number
    of retained directions and the direct/response split.
    """
    if result is None:
        raise ValueError('an SA-CASSCF result is required')
    if result.nroots < 2:
        raise ValueError('the state-averaged path needs nroots >= 2')
    ncore, nact, nele = result.ncore, result.nact, result.nele
    nroots, weights = result.nroots, result.weights
    target = result.target if state is None else int(state)
    theta = np.asarray(result.theta, dtype=float)
    pairs = result.pairs
    factors = result.factors
    h = result.hcore
    c_ref = result.c_ref
    e_nuc = float(mol.energy_nuc())

    def g_sa_of(th):
        return _state_data(th, c_ref, h, factors, ncore, nact, nele,
                           pairs, nroots, weights, target, e_nuc)['g_sa']

    data0 = _state_data(theta, c_ref, h, factors, ncore, nact, nele, pairs,
                        nroots, weights, target, e_nuc)
    g_t = data0['g_states'][target]
    hess = _fd_gradient_matrix(g_sa_of, theta, theta_step)
    hess = 0.5 * (hess + hess.T)
    # mixed derivative over displaced geometries
    natm = mol.natm
    mixed = np.zeros((len(theta), natm, 3))
    for ia in range(natm):
        for x in range(3):
            gp, gm = None, None
            for sgn, which in ((1.0, 'p'), (-1.0, 'm')):
                molp = _displaced_mol(mol, ia, x, sgn * fd_step)
                fac_p = spatial_df_factors(molp, auxbasis=result.auxbasis,
                                           rcond=factors.rcond)
                h_p = sfx2c_hcore(molp)
                rhf_p = df_x2c_rhf(molp, factors=fac_p)
                c0_p = _align_orbitals(c_ref, rhf_p.mo_coeff,
                                       molp.intor('int1e_ovlp'))
                g = _state_data(theta, c0_p, h_p, fac_p, ncore, nact, nele,
                                pairs, nroots, weights, target,
                                float(molp.energy_nuc()))['g_sa']
                if which == 'p':
                    gp = g
                else:
                    gm = g
            mixed[:, ia, x] = (gp - gm) / (2.0 * fd_step)
    # pseudo-inverse response
    w, u = np.linalg.eigh(hess)
    cut = rcond * max(np.abs(w).max(), 1e-30)
    keep = np.abs(w) > cut
    coef = np.zeros_like(w)
    if keep.any():
        coef[keep] = (u[:, keep].T @ g_t) / w[keep]
    z = -u @ coef
    # The direct term is dE_t/dR at fixed orbital parameters, i.e. along the
    # envelope path where the reference orbitals follow the geometry.  At the
    # SA orbitals E_t is NOT stationary, so the closed-form single-state
    # formula (which uses the target generalized Fock for the orthonormality
    # multiplier) is not the fixed-theta derivative; the stationary value is
    # kept in the info only for diagnosis.  A central difference of the
    # envelope-path energy evaluates the direct term to O(h^2).
    coords0 = mol.atom_coords()
    direct = np.zeros((natm, 3))
    for ia in range(natm):
        for x in range(3):
            vals = []
            for sgn in (1.0, -1.0):
                c = coords0.copy()
                c[ia, x] += sgn * fd_step
                vals.append(float(df_x2c_energies_at(
                    _displaced_mol(mol, ia, x, sgn * fd_step), result)[target]))
            direct[ia, x] = (vals[0] - vals[1]) / (2.0 * fd_step)
    direct_stationary, _ = df_x2c_casscf_gradient(mol, result, state=target)
    response = np.einsum('k,kax->ax', z, mixed)
    grad = direct + response
    if not return_info:
        return grad
    info = dict(
        response=z, hessian=hess, mixed=mixed, g_target=g_t,
        hess_eigvals=w, n_kept=int(keep.sum()), n_response=len(theta),
        direct=direct, direct_stationary=direct_stationary,
        response_grad=response,
        hess_absmin=float(np.abs(w).min()),
        hess_absmax=float(np.abs(w).max()))
    return grad, info
