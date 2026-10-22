"""Density-fitted two-component X2C Hartree-Fock energies and gradients.

WHAT THIS MODULE PROVIDES

Two-component (spinor) X2C Hartree-Fock with a density-fitted (RI-JK)
two-electron part:

    E = tr(D H_X2C) + 1/2 tr(D J) - 1/2 tr(D K) + E_nuc,

    J_ij = sum_kl (ij|kl) D_lk,      K_ij = sum_kl (ik|jl) D_kl,

with the four-center ERI replaced by the RI factorization

    (ij|kl) ~= sum_P w_ijP w_klP,    w = B S^{-1/2},
    B_ijP = (ij|P),                  S_PQ = (P|Q).

The one-electron Hamiltonian H_X2C is the two-component Foldy-Wouthuysen
transform built from the decontracted spinor integrals
``int1e_ovlp_spinor``, ``int1e_spsp_spinor``, ``int1e_nuc_spinor`` and
``int1e_spnucsp_spinor`` (the same transform as pyscf's
``SpinorX2CHelper``; the values here agree to 4.5e-11 on H2O/6-31G).
The two-electron operator is the non-relativistic Dirac-Coulomb
interaction, matching pyscf's ``X2C.UHF`` class; no picture-change
correction is applied to the two-electron part.

WHY THE TWO-ELECTRON PART IS DONE IN THE SPINOR BASIS

``int3c2e_spinor`` and its ``ip1``/``ip2`` derivative counterparts exist
on this stack and operate on the contracted two-component spinor basis,
so the RI factors are built directly in the basis the SCF uses.  The
auxiliary metric and its derivatives are the usual spatial ones
(``int2c2e_sph``, ``int2c2e_ip1/ip2``).

CONVENTIONS (pinned numerically; see the development probes)

The pyscf complex-GHF convention for the spinor ERI array is

    J_ij = sum_kl E[i,j,k,l] D[l,k],
    K_ij = sum_kl E[i,k,j,l] D[k,l],

which for w = B S^{-1/2} becomes

    rho_P = sum_ij w_ijP D_ji,
    J     = sum_P w_P rho_P,
    K     = sum_P w_P D w_P.

The three-center tensor is Hermitian in the AO pair,
``B_ijP = conj(B_jiP)``; the derivative with respect to the second AO
center therefore carries a complex conjugate:

    dB/dR_A = -( ip1 restricted to A on the bra
               + conj(ip1 with the AO pair axes exchanged) restricted
                 to A on the ket
               + ip2 restricted to A on the auxiliary center ).

The conjugate was verified by finite differences of B (2.5e-9 relative
over all atoms and directions on H2O/6-31G).

NUCLEAR GRADIENT

At a converged SCF the density response collapses to the Pulay term:

    dE/dR = sum_ij D_ij dH_ji/dR
          + dE_2e/dR |_D
          - sum_ij W_ij dS_ji/dR
          + dE_nuc/dR,
    W = C_occ diag(eps_i n_i) C_occ^dag.

The fixed-density two-electron derivative is evaluated in factor space.
With G_P = rho_P D^T - (D w_P D)^T,

    dE_2e = sum_ijP G_ijP dw_ijP,
    dw    = dB S^{-1/2} + B dS^{-1/2},
    dE_2e = sum (G S^{-1/2}) dB + sum M dS^{-1/2},
    M_QP  = sum_ij B_ijQ G_ijP.

The metric inverse square root derivative uses the degeneracy-safe
Daleckii-Krein form already implemented in :mod:`diffmrc.core.ri`.

THE X2C TRANSFORM DERIVATIVE

dH_X2C/dR comes from the Wave 3 two-component derivative work: analytic
nuclear derivatives of the four input integrals, the implicit eigenbasis
derivative of the picture-change matrix X (complement-space sum, safe at
the exact Kramers degeneracy), and a transform-layer central difference
along the analytic integral directions for the polynomial and matrix
square-root chain.  The transform-layer step is 1e-4 Bohr; the Wave 3
report measures its error at about 1e-7 relative on H2O.

SCOPE AND LIMITS

* Full-electron basis sets only.  pyscf's two-component X2C helper
  raises ``NotImplementedError`` for ECP molecules, and this module
  inherits that limit.
* This is the HF layer.  The state-specific SA-CASSCF response terms
  (target-state CI response and orbital Z-vector) are not implemented
  here; see the Wave 7 report for the design.
* The gradient is validated against finite differences of the same DF
  energy and against the independent translation/rotation invariants;
  the DF J/K matrices are validated against the exact spinor ERI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from pyscf import df, lib
from pyscf.x2c import x2c as pyscf_x2c

from .ri import (MetricEig, atom_ao_mask, inv_sqrt_metric_jvp,
                 make_auxmol, metric_eig)

__all__ = [
    'DFX2CHFResult',
    'LIGHT_SPEED',
    'SpinorDFFactors',
    'X2CTransform',
    'df_x2c_hcore',
    'df_x2c_hf',
    'df_x2c_hf_gradient',
    'energy_weighted_density',
    'nuclear_repulsion_gradient',
    'spinor_df_factors',
    'spinor_overlap_derivatives',
]

LIGHT_SPEED = lib.param.LIGHT_SPEED

#: Default reciprocal condition number for the auxiliary metric truncation.
DEFAULT_RCOND = 1e-12

#: Step (Bohr) of the transform-layer central difference in dH_X2C/dR.
TRANSFORM_STEP = 1e-4


# ---------------------------------------------------------------------------
# Two-component X2C transform: values
# ---------------------------------------------------------------------------
def spinor_contraction(contr):
    """Expand a spatial contraction matrix to the two-component spinor basis.

    ``contr`` has shape ``(nao_nr_decontracted, nao_nr_contracted)``; the
    result is block diagonal in the interleaved spinor ordering used by
    the pyscf ``_spinor`` integrals.
    """
    npp, ncc = contr.shape
    cc = np.zeros((npp * 2, ncc * 2))
    cc[0::2, 0::2] = contr
    cc[1::2, 1::2] = contr
    return cc


def _x2c_xmatrix(t, v, w, s, c):
    """Picture-change matrix X from the generalized eigenproblem.

    Returns ``(x, e, a)`` with all ``2 * nao`` eigenpairs of

        h a = m a e,   h = [[v, t], [t, w/(4c^2) - t]],
                       m = [[s, 0], [0, t/(2c^2)]],

    and ``X = C_s C_l^{-1}`` from the positive-energy branch.
    """
    nao = s.shape[0]
    n2 = nao * 2
    h = np.zeros((n2, n2), dtype=complex)
    m = np.zeros((n2, n2), dtype=complex)
    h[:nao, :nao] = v
    h[:nao, nao:] = t
    h[nao:, :nao] = t
    h[nao:, nao:] = w * (.25 / c ** 2) - t
    m[:nao, :nao] = s
    m[nao:, nao:] = t * (.5 / c ** 2)
    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao, nao:]
    cs = a[nao:, nao:]
    x = scipy.linalg.solve(cl.T, cs.T).T
    return x, e, a


def _metric_renormalization(s, s1):
    """R with ``R^dag s1 R = s`` (pyscf ``x2c._get_r``)."""
    w, v = scipy.linalg.eigh(s)
    idx = w > 1e-14
    v = v[:, idx]
    w_sqrt = np.sqrt(w[idx])
    w_invsqrt = 1 / w_sqrt
    snesc = np.dot(np.dot(v.conj().T, s1), v)
    r_mid = np.einsum('i,ij,j->ij', w_invsqrt, snesc, w_invsqrt)
    w1, v1 = scipy.linalg.eigh(r_mid)
    idx1 = w1 > 1e-14
    v1 = v1[:, idx1]
    r_mid = np.dot(v1 / np.sqrt(w1[idx1]), v1.conj().T)
    r = np.einsum('i,ij,j->ij', w_invsqrt, r_mid, w_sqrt)
    return np.dot(np.dot(v, r), v.conj().T)


def _fw_hcore(t, v, w, s, x, c):
    """Foldy-Wouthuysen one-electron Hamiltonian from X."""
    s1 = s + np.dot(np.dot(x.conj().T, t), x) * (.5 / c ** 2)
    tx = np.dot(t, x)
    h1 = (v + tx + tx.conj().T - np.dot(x.conj().T, tx) +
          np.dot(np.dot(x.conj().T, w), x) * (.25 / c ** 2))
    r = _metric_renormalization(s, s1)
    return np.dot(np.dot(r.conj().T, h1), r)


def _derivative_integral(dint, p0, p1, op=None, z=0.0):
    """Derivative of a spinor integral with respect to atom A's position.

    ``dint`` is a ``(n, n)`` bra-derivative integral; ``op`` is the
    operator-derivative integral for the nuclear-attraction operators.
    The recipe (including the hermitian completion and the separate
    operator term) is the one fixed numerically in Wave 3.
    """
    y = np.zeros_like(dint)
    y[p0:p1, :] = dint[p0:p1, :]
    out = -(y + y.conj().T)
    if op is not None and z != 0.0:
        out = out - z * (op + op.conj().T)
    return out


# ---------------------------------------------------------------------------
# Two-component X2C transform: derivatives
# ---------------------------------------------------------------------------
class X2CTransform:
    """Two-component X2C transform at one geometry.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Orbital molecule (spatial basis).
    c : float
        Speed of light; defaults to the pyscf constant.

    Attributes
    ----------
    xmol : Mole
        Decontracted spinor molecule used for the transform.
    cc : ndarray
        Spinor contraction matrix, shape ``(2 * nao_xmol, 2 * nao_mol)``.
    nao : int
        Number of decontracted spinor functions.
    """

    def __init__(self, mol, c=LIGHT_SPEED):
        self.mol = mol
        self.c = c
        helper = pyscf_x2c.SpinorX2CHelper(mol)
        self.xmol, self.contr = helper.get_xmol(mol)
        self.nao = self.xmol.nao_2c()
        self.cc = spinor_contraction(self.contr)
        xm = self.xmol
        self.s = xm.intor_symmetric('int1e_ovlp_spinor')
        self.t = xm.intor_symmetric('int1e_spsp_spinor') * .5
        self.v = xm.intor_symmetric('int1e_nuc_spinor')
        self.w = xm.intor_symmetric('int1e_spnucsp_spinor')
        self.slices = xm.aoslice_2c_by_atom()
        self.z = xm.atom_charges()

    def xmatrix(self, s=None, t=None, v=None, w=None):
        """Picture-change matrix at the given integrals (defaults: stored)."""
        s = self.s if s is None else s
        t = self.t if t is None else t
        v = self.v if v is None else v
        w = self.w if w is None else w
        return _x2c_xmatrix(t, v, w, s, self.c)[0]

    def hcore_uncontracted(self, s=None, t=None, v=None, w=None, x=None):
        """FW Hamiltonian in the decontracted spinor basis."""
        s = self.s if s is None else s
        t = self.t if t is None else t
        v = self.v if v is None else v
        w = self.w if w is None else w
        if x is None:
            x = self.xmatrix(s, t, v, w)
        return _fw_hcore(t, v, w, s, x, self.c)

    def hcore(self, **kw):
        """FW Hamiltonian in the contracted spinor basis."""
        h1 = self.hcore_uncontracted(**kw)
        return np.dot(np.dot(self.cc.conj().T, h1), self.cc)

    def d_integrals(self):
        """Analytic derivative integrals, ``dict[(atom, axis)] -> (ds,dt,dv,dw)``."""
        xm = self.xmol
        ds_b = xm.intor('int1e_ipovlp_spinor', comp=3)
        dt_b = xm.intor('int1e_ipkin_spinor', comp=3)
        dv_b = xm.intor('int1e_ipnuc_spinor', comp=3)
        dw_b = xm.intor('int1e_ipspnucsp_spinor', comp=3)
        out = {}
        for ia in range(xm.natm):
            p0, p1 = self.slices[ia, 2:]
            with xm.with_rinv_at_nucleus(ia):
                ipr = xm.intor('int1e_iprinv_spinor', comp=3)
                ipwr = xm.intor('int1e_ipsprinvsp_spinor', comp=3)
            for ix in range(3):
                out[(ia, ix)] = (
                    _derivative_integral(ds_b[ix], p0, p1),
                    _derivative_integral(dt_b[ix], p0, p1),
                    _derivative_integral(dv_b[ix], p0, p1, ipr[ix],
                                         self.z[ia]),
                    _derivative_integral(dw_b[ix], p0, p1, ipwr[ix],
                                         self.z[ia]),
                )
        return out

    def dx_implicit(self, dints):
        """Implicit derivative of X from the generalized eigenproblem.

        Only complement-space (negative-energy) contributions enter.  The
        within-branch part cancels exactly in
        ``dX = (dB - X dA) A^{-1}``, which keeps the formula finite at
        the exact Kramers degeneracy of the positive branch.
        """
        c = self.c
        nao = self.nao
        n2 = 2 * nao
        x, e, a = _x2c_xmatrix(self.t, self.v, self.w, self.s, c)
        sel = np.arange(nao, n2)
        comp = np.arange(0, nao)
        amat = a[:nao, sel]
        ainv = scipy.linalg.inv(amat)
        gap = e[sel][None, :] - e[comp][:, None]
        ac = a.conj().T
        out = {}
        for key, (ds, dt, dv, dw) in dints.items():
            dh = np.zeros((n2, n2), dtype=complex)
            dm = np.zeros((n2, n2), dtype=complex)
            dh[:nao, :nao] = dv
            dh[:nao, nao:] = dt
            dh[nao:, :nao] = dt
            dh[nao:, nao:] = dw * (.25 / c ** 2) - dt
            dm[:nao, :nao] = ds
            dm[nao:, nao:] = dt * (.5 / c ** 2)
            tt = ac @ dh @ a
            uu = ac @ dm @ a
            beta = (tt[np.ix_(comp, sel)] -
                    uu[np.ix_(comp, sel)] * e[sel][None, :]) / gap
            da = a[:, comp] @ beta
            out[key] = (da[nao:, :] - np.dot(x, da[:nao, :])) @ ainv
        return out

    def hcore_derivatives(self, step=TRANSFORM_STEP):
        """dH_X2C/dR in the contracted spinor basis, shape ``(natm,3,n,n)``.

        The integral derivatives are analytic; the polynomial and matrix
        square-root chain is differentiated by a transform-layer central
        difference along those analytic directions, and the X response
        comes from :meth:`dx_implicit`.  The Wave 3 report measured the
        transform-layer error at about 1e-7 relative on H2O.
        """
        dints = self.d_integrals()
        dx = self.dx_implicit(dints)
        x0 = self.xmatrix()
        homo = (self.s, self.t, self.v, self.w)
        out = np.zeros((self.xmol.natm, 3, self.nao, self.nao),
                       dtype=complex)
        for (ia, ix), (ds, dt, dv, dw) in dints.items():
            sp = (self.s + step * ds, self.t + step * dt,
                  self.v + step * dv, self.w + step * dw)
            sm = (self.s - step * ds, self.t - step * dt,
                  self.v - step * dv, self.w - step * dw)
            dh_int = (self.hcore_uncontracted(*sp, x=x0) -
                      self.hcore_uncontracted(*sm, x=x0)) / (2 * step)
            dxx = dx[(ia, ix)]
            dh_x = (self.hcore_uncontracted(*homo, x=x0 + step * dxx) -
                    self.hcore_uncontracted(*homo, x=x0 - step * dxx)) / (2 * step)
            out[ia, ix] = dh_int + dh_x
        return np.einsum('pi,axpq,qj->axij', self.cc.conj(), out, self.cc)


def df_x2c_hcore(mol, C=None, c=LIGHT_SPEED, return_deriv=False,
                 step=TRANSFORM_STEP):
    """Two-component X2C one-electron Hamiltonian.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Orbital molecule.
    C : ndarray, optional
        If given, the Hamiltonian is returned in this basis,
        ``C^dag H C`` (for example MO coefficients).
    c : float
        Speed of light.
    return_deriv : bool
        Also return ``dH/dR``, shape ``(natm,3,n,n)`` (same basis as H).
    step : float
        Transform-layer step for the chain derivative, in Bohr.

    Returns
    -------
    h : ndarray, (n,n)
        Complex Hermitian two-component Hamiltonian.
    dh : ndarray, (natm,3,n,n), only when ``return_deriv`` is True.
    """
    tr = X2CTransform(mol, c=c)
    h = tr.hcore()
    if C is not None:
        cc = np.asarray(C)
        h = np.dot(np.dot(cc.conj().T, h), cc)
    if not return_deriv:
        return h
    dh = tr.hcore_derivatives(step=step)
    if C is not None:
        cc = np.asarray(C)
        dh = np.einsum('pi,axpq,qj->axij', cc.conj(), dh, cc)
    return h, dh


# ---------------------------------------------------------------------------
# DF factors in the two-component spinor basis
# ---------------------------------------------------------------------------
def _spinor_atom_mask(mol, atom_idx):
    """Boolean mask of the spinor AO functions on ``atom_idx``."""
    nao = mol.nao_2c()
    mask = np.zeros(nao, dtype=bool)
    p0, p1 = mol.aoslice_2c_by_atom()[atom_idx, 2:]
    mask[p0:p1] = True
    return mask


@dataclass(frozen=True)
class SpinorDFFactors:
    """RI-JK factors in the contracted two-component spinor basis.

    Attributes
    ----------
    b : ndarray, (nao, nao, naux), complex
        Three-center integrals ``(ij|P)``; Hermitian in the AO pair.
    s : ndarray, (naux, naux)
        Auxiliary metric.
    jhalf : ndarray, (naux, naux)
        Truncated symmetric inverse square root of ``s``.
    eig : MetricEig
        Metric eigendecomposition reused by the derivative.
    auxmol : pyscf.gto.Mole
        Auxiliary molecule.
    auxbasis : str
        Label of the auxiliary basis.
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

    def density_fit_charge(self, d):
        """``rho_P = sum_ij w_ijP D_ji`` (real for Hermitian D)."""
        return np.einsum('ijP,ji->P', self.w, d)

    def fock(self, d, h):
        """DF Coulomb and exchange matrices and the Fock matrix.

        Returns ``(f, j, k)`` with ``f = h + j - k``.
        """
        w = self.w
        rho = np.einsum('ijP,ji->P', w, d)
        j = np.einsum('ijP,P->ij', w, rho)
        k = np.zeros_like(d)
        for p in range(self.naux):
            wp = w[:, :, p]
            k += wp @ d @ wp
        return h + j - k, j, k

    def energy_parts(self, d, h):
        """``(e_one_electron, e_coulomb, e_exchange)`` without E_nuc."""
        w = self.w
        rho = np.einsum('ijP,ji->P', w, d)
        e_h = np.einsum('ij,ji->', d, h)
        e_j = 0.5 * np.sum(rho * rho)
        e_k = 0.0
        for p in range(self.naux):
            wp = w[:, :, p]
            e_k -= 0.5 * np.trace(d @ wp @ d @ wp)
        return float(e_h.real), float(e_j.real), float(e_k.real)

    def deriv_factors(self, atom_idx):
        """Nuclear derivatives of every DF factor for one atom.

        Returns ``(db, ds_metric, djhalf)`` with shapes
        ``(3,nao,nao,naux)``, ``(3,naux,naux)`` and ``(3,naux,naux)``.
        The second-AO-center term is the complex-conjugate transpose of
        the first-center derivative integral; the conjugate is required
        for the complex spinor tensor and was verified by finite
        differences (2.5e-9 relative).
        """
        mol, auxmol = self._mol, self.auxmol
        ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1_spinor',
                               aosym='s1')
        ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2_spinor',
                               aosym='s1')
        m_ao = _spinor_atom_mask(mol, atom_idx)
        m_aux = atom_ao_mask(auxmol, atom_idx)
        first = ip1 * m_ao[:, None, None]
        second = ip1.transpose(0, 2, 1, 3).conj() * m_ao[None, :, None]
        aux = ip2 * m_aux[None, None, :]
        db = -(first + second + aux)

        ip1m = auxmol.intor('int2c2e_ip1', aosym='s1')
        ip2m = auxmol.intor('int2c2e_ip2', aosym='s1')
        ds = -(ip1m * m_aux[:, None] + ip2m * m_aux[None, :])
        ds = 0.5 * (ds + ds.transpose(0, 2, 1))
        dj = inv_sqrt_metric_jvp(self.s, ds, eig=self.eig)
        return db, ds, dj

    def energy_deriv(self, d, db, dj):
        """Fixed-density two-electron derivative along one atom/axis.

        ``db`` is ``(nao,nao,naux)`` and ``dj`` is ``(naux,naux)`` for a
        single Cartesian component.  Returns ``(total, coulomb, exchange)``
        in Eh/Bohr; the density is held fixed.
        """
        w = self.w
        rho = np.einsum('ijP,ji->P', w, d)
        dw = np.matmul(db, self.jhalf) + np.matmul(self.b, dj)
        de_j = np.einsum('P,ijP,ji->', rho, dw, d)
        de_k = 0.0
        for p in range(self.naux):
            wp = w[:, :, p]
            dwp = dw[:, :, p]
            de_k -= np.trace(d @ dwp @ d @ wp)
        return float((de_j + de_k).real), float(de_j.real), float(de_k.real)

    # The parent molecule is needed by deriv_factors but is not part of the
    # value-level factor set.
    _mol: object = field(default=None, repr=False)


def spinor_df_factors(mol, auxmol=None, auxbasis=None,
                      rcond=DEFAULT_RCOND):
    """Assemble the RI-JK factors in the contracted spinor basis.

    When ``auxmol`` is None the auxiliary molecule is built by
    :func:`diffmrc.core.ri.make_auxmol`, which tries the standard JK
    fitting sets and reports the basis actually used.
    """
    if auxmol is None:
        auxmol, label = make_auxmol(mol, auxbasis=auxbasis)
    else:
        label = auxbasis if auxbasis is not None else 'custom auxiliary molecule'
    b = df.incore.aux_e2(mol, auxmol, intor='int3c2e_spinor', aosym='s1')
    s = auxmol.intor('int2c2e_sph', aosym='s1')
    s = 0.5 * (s + s.T)
    eig = metric_eig(s, rcond)
    scale = np.zeros_like(eig.w)
    scale[eig.keep] = 1.0 / np.sqrt(eig.w[eig.keep])
    jhalf = (eig.u * scale) @ eig.u.T
    jhalf = 0.5 * (jhalf + jhalf.T)
    return SpinorDFFactors(b=b, s=s, jhalf=jhalf, eig=eig, auxmol=auxmol,
                           auxbasis=label, rcond=rcond, _mol=mol)


# ---------------------------------------------------------------------------
# SCF
# ---------------------------------------------------------------------------
def _initial_density(mol, h, s):
    """Aufbau density from the one-electron Hamiltonian."""
    e, c = scipy.linalg.eigh(h, s)
    occ = np.zeros(len(e))
    occ[:mol.nelectron] = 1.0
    return (c * occ) @ c.conj().T


def _diis_fock(focks, errors):
    """Pulay DIIS combination of complex Fock matrices."""
    n = len(focks)
    b = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            b[i, j] = np.real(np.vdot(errors[i], errors[j]))
    b[:n, n] = -1.0
    b[n, :n] = -1.0
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    coef = np.linalg.solve(b, rhs)
    f = np.zeros_like(focks[0])
    for i in range(n):
        f += coef[i] * focks[i]
    return f


@dataclass
class DFX2CHFResult:
    """Converged (or best) DF-X2C HF result at one geometry.

    The density and orbitals are in the contracted spinor basis.
    ``e_tot`` includes the nuclear repulsion.
    """
    mol: object
    factors: SpinorDFFactors
    hcore: np.ndarray
    ovlp: np.ndarray
    dm: np.ndarray
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    mo_occ: np.ndarray
    fock: np.ndarray
    e_tot: float
    e_one_electron: float
    e_coulomb: float
    e_exchange: float
    e_nuc: float
    converged: bool
    gradient_norm: float
    n_iter: int


def df_x2c_hf(mol, auxmol=None, auxbasis=None, dm0=None, conv_tol=1e-11,
              max_cycle=200, rcond=DEFAULT_RCOND, verbose=False):
    """Density-fitted two-component X2C Hartree-Fock (complex GHF).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Orbital molecule.
    auxmol : Mole, optional
        Pre-built auxiliary molecule.
    auxbasis : str, optional
        Auxiliary basis name; used only when ``auxmol`` is None.
    dm0 : ndarray, optional
        Initial density in the contracted spinor basis.
    conv_tol : float
        Energy-change threshold for SCF convergence.
    max_cycle : int
        Maximum number of SCF iterations.
    rcond : float
        Metric truncation threshold.
    verbose : bool
        Print SCF progress.

    Returns
    -------
    DFX2CHFResult
    """
    factors = spinor_df_factors(mol, auxmol=auxmol, auxbasis=auxbasis,
                                rcond=rcond)
    transform = X2CTransform(mol)
    h = transform.hcore()
    s = mol.intor_symmetric('int1e_ovlp_spinor')
    nelec = mol.nelectron
    d = _initial_density(mol, h, s) if dm0 is None else np.array(dm0)
    focks, errors = [], []
    e_old = None
    mo_e = mo_c = occ = None
    gnorm = np.inf
    for it in range(max_cycle):
        f, _, _ = factors.fock(d, h)
        e_h, e_j, e_k = factors.energy_parts(d, h)
        e_tot = e_h + e_j + e_k + mol.energy_nuc()
        err = f @ d @ s - s @ d @ f
        gnorm = float(np.linalg.norm(err))
        if verbose and (it % 10 == 0 or it == max_cycle - 1):
            print('  scf %3d  E %.12f  |FDS-SDF| %.2e' % (it, e_tot, gnorm))
        if e_old is not None and abs(e_tot - e_old) < conv_tol and gnorm < 1e-8:
            break
        e_old = e_tot
        focks.append(f)
        errors.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        f_use = _diis_fock(focks, errors) if len(focks) > 1 else f
        mo_e, mo_c = scipy.linalg.eigh(f_use, s)
        occ = np.zeros(len(mo_e))
        occ[:nelec] = 1.0
        d = (mo_c * occ) @ mo_c.conj().T
    f, _, _ = factors.fock(d, h)
    e_h, e_j, e_k = factors.energy_parts(d, h)
    return DFX2CHFResult(
        mol=mol, factors=factors, hcore=h, ovlp=s, dm=d, mo_coeff=mo_c,
        mo_energy=mo_e, mo_occ=occ, fock=f, e_tot=float(e_h + e_j + e_k +
                                                        mol.energy_nuc()),
        e_one_electron=e_h, e_coulomb=e_j, e_exchange=e_k,
        e_nuc=float(mol.energy_nuc()), converged=bool(gnorm < 1e-7),
        gradient_norm=gnorm, n_iter=it + 1)


def energy_weighted_density(result):
    """``W = C_occ diag(eps_i n_i) C_occ^dag`` in the spinor basis."""
    c = result.mo_coeff
    occ = result.mo_occ
    e = result.mo_energy
    oc = occ > 0
    m = c[:, oc] * (e[oc] * occ[oc])
    return m @ c[:, oc].conj().T


def spinor_overlap_derivatives(mol):
    """dS/dR of the contracted spinor overlap, shape ``(natm,3,n,n)``."""
    n = mol.nao_2c()
    ip = mol.intor('int1e_ipovlp_spinor', comp=3)
    loc = mol.aoslice_2c_by_atom()
    out = np.zeros((mol.natm, 3, n, n), dtype=complex)
    for ia in range(mol.natm):
        p0, p1 = loc[ia, 2:]
        for x in range(3):
            out[ia, x] = _derivative_integral(ip[x], p0, p1)
    return out


def nuclear_repulsion_gradient(mol):
    """Nuclear repulsion gradient, shape ``(natm,3)`` in Eh/Bohr."""
    z = mol.atom_charges()
    r = mol.atom_coords()
    dr = r[:, None, :] - r
    dist = np.linalg.norm(dr, axis=2)
    idx = np.diag_indices(z.size)
    dist[idx] = 1e100
    rinv = 1. / dist
    rinv[idx] = 0.
    return np.einsum('i,j,ijx,ij->ix', -z, z, dr, rinv ** 3)


def df_x2c_hf_gradient(mol, result=None, auxmol=None, auxbasis=None,
                       dm0=None, transform_step=TRANSFORM_STEP,
                       rcond=DEFAULT_RCOND, verbose=False):
    """Analytic nuclear gradient of the DF-X2C HF energy.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Orbital molecule at the reference geometry.
    result : DFX2CHFResult, optional
        A converged result; when None the SCF is run here.
    auxmol, auxbasis, dm0, rcond, verbose
        Forwarded to :func:`df_x2c_hf` when ``result`` is None.
    transform_step : float
        Step of the transform-layer chain derivative in dH_X2C/dR.

    Returns
    -------
    grad : ndarray, (natm,3), Eh/Bohr
    parts : dict
        ``{'one_electron', 'two_electron', 'pulay', 'nuclear'}``,
        each ``(natm,3)``.
    """
    if result is None:
        result = df_x2c_hf(mol, auxmol=auxmol, auxbasis=auxbasis, dm0=dm0,
                           rcond=rcond, verbose=verbose)
    transform = X2CTransform(mol)
    dh = transform.hcore_derivatives(step=transform_step)
    ds = spinor_overlap_derivatives(mol)
    d = result.dm
    wmat = energy_weighted_density(result)
    natm = mol.natm
    g_h = np.zeros((natm, 3))
    g_2e = np.zeros((natm, 3))
    g_p = np.zeros((natm, 3))
    for ia in range(natm):
        db, _, dj = result.factors.deriv_factors(ia)
        for x in range(3):
            g_h[ia, x] = np.einsum('ij,ji->', d, dh[ia, x]).real
            de2, _, _ = result.factors.energy_deriv(d, db[x], dj[x])
            g_2e[ia, x] = de2
            g_p[ia, x] = -np.einsum('ij,ji->', wmat, ds[ia, x]).real
    g_nuc = nuclear_repulsion_gradient(mol)
    grad = g_nuc + g_h + g_2e + g_p
    return grad, {'one_electron': g_h, 'two_electron': g_2e,
                  'pulay': g_p, 'nuclear': g_nuc}
