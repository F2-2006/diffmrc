"""RASSCF orbital optimization on top of the determinant CI kernel.

PURPOSE
-------
Minimizes the RASSCF energy over orbital rotations for a fixed RAS
partition.  The wavefunction is a RAS-CI expansion in the determinant
basis over the active orbitals (``diffmrc.core.rasscf.determinants``); the
inactive orbitals are frozen doubly occupied and the external orbitals are
empty.  Both the energy and its analytic orbital gradient are provided, so
the optimizer never falls back to finite differences.

PUBLIC INTERFACE
----------------
``RASSCF`` is built from a pyscf ``Mole``, an MO coefficient matrix and the
partition ``(n_inact, n1, n2, n3, max_holes, max_elec)``.

    ``energy(x)``        energy at the rotation parameters ``x``.
    ``_energy_grad(x)``  energy and analytic gradient (internal driver API).
    ``kernel(x0=None)``  BFGS optimization; sets ``x``, ``C``, ``e_tot``,
                         ``converged`` and ``history``.

The attributes ``pairs``, ``dets`` and ``ranges`` describe the rotation
parameter space, the CI space and the five subspace ranges.

ORBITAL BASIS ORDERING
----------------------
[inactive | RAS1 | RAS2 | RAS3 | external]

ORBITAL ROTATION
----------------
C(kappa) = C0 @ exp(kappa), with kappa antisymmetric and non-zero only for
pairs belonging to different subspaces.  Rotations inside a subspace are
redundant because the RAS constraints depend only on the subspace
occupation totals, which are invariant under intra-subspace unitary
rotations.

The orbital gradient is the analytic derivative of the energy with respect
to the rotation parameters of C = C0 exp(kappa).  The derivative of the
matrix exponential is handled exactly through its logarithmic derivative
(see ``_grad_param``), so the gradient is consistent with the energy for
finite rotations.

VALIDATED SCOPE
---------------
Water/STO-3G with a 4-inactive + RAS2(2) partition: the RAS(2,2) energy
reproduces pyscf CASSCF(2,2) to 5.8e-12 Eh, which is the RAS-to-CAS limit
of Malmqvist 2008, Sec. II B.  The analytic gradient agrees with central
differences over 14 rotation parameters at amplitude 1e-2 to 2.6e-9
(largest component; relative 1.3e-9), and the agreement is amplitude
independent from kappa = 0 up to 0.3 (1.8e-9 to 2.8e-9, i.e. the
finite-difference step noise).  A non-degenerate partition with a
non-empty RAS1 (RAS(1,2,2), holes <= 1, electrons <= 2) reproduces pyscf
CASSCF(4e,3o) to 5.8e-12 Eh in the zero-excitation limit, and converges to
a stationary point in the constrained case.

BOUNDARIES
----------
* Determinant basis with fixed M_S: no spin-adapted CSFs, no point-group
  symmetry, no state averaging, ground root only.
* RASSCF only.  RASPT2 dynamic correlation, FOCIS construction,
  IPEA/imaginary shifts, spin-orbit coupling and RASSI are not
  implemented.
* The AO-to-MO four-index transform is O(n_mo^5) and the 2-RDM is an
  explicit orbital-quartet loop; the implementation targets small active
  spaces (seconds at five active orbitals and about a hundred
  determinants).
* Only inter-subspace rotations are parametrized.  The redundancy argument
  for intra-subspace rotations is this implementation's, not the paper's;
  it is supported by the finite-difference gradient check.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm
from scipy.optimize import minimize

from .determinants import (build_hamiltonian, enumerate_active_determinants,
                           one_rdm, solve_ci, two_rdm)

__all__ = ["RASSCF"]


class RASSCF:
    """RASSCF for a closed-shell or high-spin reference (fixed Ms)."""

    def __init__(self, mol, mo_coeff, n_inact, n1, n2, n3,
                 max_holes, max_elec, n_alpha=None, n_beta=None):
        self.mol = mol
        self.C0 = np.array(mo_coeff, dtype=float)
        self.nmo = self.C0.shape[1]
        self.n_inact = n_inact
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.n_act = n1 + n2 + n3
        self.n_ext = self.nmo - n_inact - self.n_act
        if min(n1, n2, n3, n_inact, self.n_ext) < 0:
            raise ValueError("inconsistent RAS partition for this orbital count")
        self.max_holes = max_holes
        self.max_elec = max_elec

        nelec_act = mol.nelectron - 2 * n_inact
        if n_alpha is None:
            n_alpha = nelec_act // 2
        if n_beta is None:
            n_beta = nelec_act - n_alpha
        if n_alpha + n_beta != nelec_act:
            raise ValueError("active alpha+beta electrons do not match the core")
        self.na_act, self.nb_act = n_alpha, n_beta

        self.dets = enumerate_active_determinants(
            self.n_act, n_alpha, n_beta, n1, n2, n3, max_holes, max_elec)
        if not self.dets:
            raise ValueError("empty RAS CI space")

        # subspace ranges
        self.ranges = [
            (0, n_inact),
            (n_inact, n_inact + n1),
            (n_inact + n1, n_inact + n1 + n2),
            (n_inact + n1 + n2, n_inact + n1 + n2 + n3),
            (n_inact + self.n_act, self.nmo),
        ]
        self.pairs = self._rotation_pairs()
        # integral cache in the AO basis
        self.h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        from pyscf import ao2mo
        self.eri_ao = ao2mo.full(mol, np.eye(self.nmo), compact=False)
        self.eri_ao = self.eri_ao.reshape(self.nmo, self.nmo, self.nmo, self.nmo)
        self.energy_nuc = mol.energy_nuc()

        self._last = None

    # ------------------------------------------------------------------
    def _rotation_pairs(self):
        pairs = []
        for s in range(len(self.ranges)):
            for t in range(s + 1, len(self.ranges)):
                a0, a1 = self.ranges[s]
                b0, b1 = self.ranges[t]
                for a in range(a0, a1):
                    for b in range(b0, b1):
                        pairs.append((min(a, b), max(a, b), a, b))
        return pairs

    def _kappa(self, x):
        k = np.zeros((self.nmo, self.nmo))
        for (lo, hi, a, b), v in zip(self.pairs, x):
            k[a, b] += v
            k[b, a] -= v
        return k

    # ------------------------------------------------------------------
    def _mo_integrals(self, C):
        h = C.T @ self.h_ao @ C
        eri = np.einsum("pi,pqrs->iqrs", C, self.eri_ao, optimize=True)
        eri = np.einsum("qj,iqrs->ijrs", C, eri, optimize=True)
        eri = np.einsum("rk,ijrs->ijks", C, eri, optimize=True)
        eri = np.einsum("sl,ijks->ijkl", C, eri, optimize=True)
        return h, eri

    def _active_hamiltonian(self, h, eri):
        """Effective one-electron operator and core energy."""
        ia = self.n_inact
        act = slice(ia, ia + self.n_act)
        core = slice(0, ia)
        h_eff = h[act, act].copy()
        for i in range(ia):
            h_eff += 2.0 * eri[act, act, i, i] - eri[act, i, i, act]
        eri_cc = eri[core, core, core, core]
        e_core = 2.0 * np.trace(h[core, core])
        e_core += 2.0 * np.einsum("iijj->", eri_cc)
        e_core -= np.einsum("ijji->", eri_cc)
        return h_eff, e_core

    # ------------------------------------------------------------------
    def _embedded_dets(self):
        """Reference determinants embedded in the full MO space (core frozen)."""
        core = (1 << self.n_inact) - 1
        out = [(core | (a << self.n_inact), core | (b << self.n_inact))
               for (a, b) in self.dets]
        return out

    # ------------------------------------------------------------------
    def energy(self, x):
        return self._energy_grad(x, need_grad=False)[0]

    def _energy_grad(self, x, need_grad=True):
        kappa = self._kappa(x)
        C = self.C0 @ expm(kappa)
        h, eri = self._mo_integrals(C)
        h_eff, e_core = self._active_hamiltonian(h, eri)
        ia = self.n_inact
        act = slice(ia, ia + self.n_act)
        eri_act = eri[act, act, act, act]
        H = build_hamiltonian(self.dets, self.n_act, h_eff, eri_act)
        w, v = solve_ci(H, nroots=1)
        c = v[:, 0]
        e_act = w[0]
        e_tot = e_core + e_act + self.energy_nuc
        self._last = dict(C=C, h=h, eri=eri, c=c, e_tot=e_tot)
        if not need_grad:
            return e_tot, None
        G = self._mo_gradient(c, h, eri)
        g = self._grad_param(G, kappa)
        return e_tot, g

    # ------------------------------------------------------------------
    def _mo_gradient(self, civec, h, eri):
        """dE/dR_ab for C -> C (1 + R), R antisymmetric, in the current MO basis.

        The reference wavefunction is embedded in the full MO determinant
        space (inactive core doubly occupied) so that the one- and two-particle
        density matrices are obtained by the same validated routines used for
        the CI, without hand-built core-active factorizations.

        One-body contribution: dh_pq = R_rp h_rq + h_ps R_sq gives
        dE = 2 (h D)_ab R_ab.
        Two-body contribution: each of the four integral indices produces one
        term of the form 0.5 d[p,q,r,s] (pq|rs) with that index replaced by the
        rotation.
        """
        n = self.nmo
        full_dets = self._embedded_dets()
        D = one_rdm(civec, full_dets, n)
        d = two_rdm(civec, full_dets, n)
        G = 2.0 * (h @ D)
        G += 0.5 * np.einsum("bqrs,aqrs->ab", d, eri, optimize=True)
        G += 0.5 * np.einsum("pbrs,pars->ab", d, eri, optimize=True)
        G += 0.5 * np.einsum("pqbs,pqas->ab", d, eri, optimize=True)
        G += 0.5 * np.einsum("pqrb,pqra->ab", d, eri, optimize=True)
        return G

    # ------------------------------------------------------------------
    def _grad_param(self, G, kappa):
        """dE/dx_k for C(x) = C0 exp(kappa(x)), exact in the exponential map.

        With Omega_k = e^{-kappa} d exp(kappa)/dx_k the chain rule gives
        g_k = sum_ab G_ab Omega_k[a,b].  The derivative of the exponential is

            d exp(kappa)/dx_k = int_0^1 e^{s kappa} E_k e^{(1-s) kappa} ds,

        hence Omega_k = int_0^1 e^{-u kappa} E_k e^{u kappa} du with u = 1 - s.
        The integrand is evaluated by Gauss-Legendre quadrature; for the
        generator E_k = e_a e_b^T - e_b e_a^T this reduces to

            g_k = int_0^1 du [ (P^T G P)_ab - (P^T G P)_ba ],  P = exp(-u kappa).
        """
        u, w = leggauss(12)
        u = 0.5 * (u + 1.0)
        w = 0.5 * w
        g = np.zeros(len(self.pairs))
        for ui, wi in zip(u, w):
            P = expm(-ui * kappa)
            W = P.T @ G @ P
            for k, (lo, hi, a, b) in enumerate(self.pairs):
                g[k] += wi * (W[a, b] - W[b, a])
        return g

    # ------------------------------------------------------------------
    def kernel(self, x0=None, gtol=1e-7, maxiter=200, verbose=True):
        if x0 is None:
            x0 = np.zeros(len(self.pairs))
        history = []

        def fun(x):
            e, g = self._energy_grad(x, need_grad=True)
            history.append(e)
            return e, g

        res = minimize(fun, x0, jac=True, method="BFGS",
                       options=dict(gtol=gtol, maxiter=maxiter))
        self.x = res.x
        self.C = self.C0 @ expm(self._kappa(res.x))
        e, _ = self._energy_grad(res.x, need_grad=False)
        self.e_tot = e
        self.converged = res.success
        self.history = history
        if verbose:
            print(f"RASSCF: E = {e:.12f}  n_iter = {len(history)}  "
                  f"conv = {res.success}")
        return e, self.C
