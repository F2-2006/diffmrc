"""Transition densities and oscillator strengths from state-averaged CASSCF.

Given a converged state-averaged CASSCF object and its molecule, this
module builds the one-particle transition density between every pair of
roots, contracts it with the length (position) and velocity (gradient)
integrals, and returns oscillator strengths, Einstein A coefficients and
radiative lifetimes.

Public interface
----------------
``transition_density(ci_i, ci_j, ncas, nelecas)``
    One-particle transition density in pyscf's FCI convention.
``transition_moments(mc, mol)``
    Length- and velocity-gauge transition moments for every root pair,
    as two ``(nroots, nroots, 3)`` arrays.
``oscillator(dE_h, mu_len, mu_vel)``
    Length and velocity oscillator strengths for one pair.
``einstein_a(dE_h, f)``
    Spontaneous-emission Einstein coefficient for one pair.
``transition_table(mc, mol, states)``
    Per-pair rows (transition energy, moments, oscillator strengths,
    Einstein A) and per-root radiative lifetimes.
``ci_list(mc)``, ``e_states(mc)``
    Root enumeration that works for state-averaged and single-root
    objects alike.

Units
-----
Transition energies are Hartree (reported also in eV, cm^-1 and nm).
Length-gauge moments are in atomic units of dipole (e a0) and are also
reported in Debye; velocity-gauge moments are in e a0^-1.  Oscillator
strengths are dimensionless.  Einstein coefficients are s^-1 and
lifetimes are s.

Conventions
-----------
Transition densities follow pyscf's FCI convention

    dm1[p,q] = <bra| a^dag_q a_p |ket>,

so a one-electron operator O is contracted as

    <bra|O|ket> = sum_pq dm1[p,q] * O[q,p].

Length gauge uses the position operator r (a0); velocity gauge uses the
gradient (a0^-1), obtained from pyscf's ``int1e_ipovlp`` = <grad p|q>
with a sign flip from integration by parts.

Oscillator strengths are the spin-free, non-relativistic forms

    f_L = (2/3) dE |<r>|^2 ,      f_V = (2/(3 dE)) |<grad>|^2

with dE in Hartree and matrix elements in atomic units.  Both are
per-component values of one non-degenerate pair of spin-free states; the
degeneracy factors that convert them to term-averaged quantities are not
applied here and must be supplied by the caller when comparing with
experiment.

Einstein A is spontaneous emission only:

    A[j<-i] = 7.4217e-22 * nu[Hz]^2 * f

so the lifetime returned here is the purely radiative lifetime of the
spin-free state; internal conversion, intersystem crossing, quenching and
spin-forbidden channels are all outside it.

Boundaries
----------
Only the method layer is transferred.  The prototype's numerical layer --
the f-f absolute-intensity pipeline (Mulliken 4f/5d active-space
selection, the end-to-end runner and its JSON records) -- is deliberately
not part of this module: the active space drifts between the geometries
of that study, so those absolute f-f intensities are not reproducible at
the method level.  A caller that needs absolute intensities must supply
its own geometry-stable active space and treat the result as a method
demonstration, not a prediction.

Validated scope
---------------
The method layer reproduces the prototype numbers.  On the LiH X->A
ladder the velocity-gauge oscillator strength converges onto the
length-gauge value (cc-pVDZ CAS(2,2) ratio 0.374; cc-pVQZ CAS(2,10)
ratio 1.118 with |ratio - 1| at least halved).  The permanent dipole
agrees with pyscf's own AO-density contraction to better than 1e-8 D, and
the AO gradient matrix is antisymmetric to machine precision.  These are
the criteria carried by ``tests/driver/test_ff_transition.py``.
"""
from __future__ import annotations

import numpy as np
from pyscf.fci import direct_spin1

__all__ = [
    'ci_list', 'e_states', 'transition_density', 'transition_moments',
    'oscillator', 'einstein_a', 'transition_table',
    'HARTREE_TO_EV', 'HARTREE_TO_CM1', 'AU_DIPOLE_DEBYE', 'A_FACTOR',
]

_FCI = direct_spin1.FCI()  # plain solver: the SA wrapper expects CI lists

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_CM1 = 219474.6313702
AU_DIPOLE_DEBYE = 2.541746473
HARTREE_TO_J = 4.3597447222071e-18
PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
# A[s^-1] = A_FACTOR * nu[Hz]^2 * f
A_FACTOR = 7.4217e-22


def ci_list(mc):
    """CI vectors as a list, whether or not the run was state-averaged."""
    ci = mc.ci
    return list(ci) if isinstance(ci, (list, tuple)) else [ci]


def e_states(mc):
    """Per-root energies in Hartree, for SA and single-root runs alike."""
    if hasattr(mc, "e_states"):
        return [float(x) for x in mc.e_states]
    return [float(mc.e_tot)]


def transition_density(ci_i, ci_j, ncas, nelecas):
    """One-particle transition density between two CI vectors.

    Uses pyscf's FCI convention ``dm1[p,q] = <i| a^dag_q a_p |j>``; see
    the module docstring for the operator contraction.
    """
    return _FCI.trans_rdm1(ci_i, ci_j, ncas, nelecas)


def transition_moments(mc, mol):
    """Length- and velocity-gauge transition moments between all SA roots.

    Returns ``(mu, vl)``, each ``(nroots, nroots, 3)`` in atomic units.
    ``mu`` is the length-gauge moment (e a0) and is symmetric in the root
    pair; ``vl`` is the velocity-gauge moment (e a0^-1) and is
    antisymmetric.
    """
    Cact = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    dip_ao = mol.intor("int1e_r", comp=3)
    grad_ao = -mol.intor("int1e_ipovlp", comp=3)  # <p|grad|q>
    dip = np.einsum("pi,xpq,qj->xij", Cact, dip_ao, Cact)
    grad = np.einsum("pi,xpq,qj->xij", Cact, grad_ao, Cact)
    cis = ci_list(mc)
    n = len(cis)
    mu = np.zeros((n, n, 3))
    vl = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(i, n):
            dm1 = transition_density(cis[i], cis[j], mc.ncas, mc.nelecas)
            a = -np.einsum("pq,xqp->x", dm1, dip)
            b = np.einsum("pq,xqp->x", dm1, grad)
            mu[i, j] = mu[j, i] = a
            vl[i, j] = b
            vl[j, i] = -b
    return mu, vl


def oscillator(dE_h, mu_len, mu_vel):
    """Length- and velocity-gauge oscillator strengths for one pair."""
    f_len = (2.0 / 3.0) * dE_h * float(np.dot(mu_len, mu_len))
    f_vel = (2.0 / 3.0) / dE_h * float(np.dot(mu_vel, mu_vel))
    return f_len, f_vel


def einstein_a(dE_h, f):
    """Spontaneous-emission Einstein coefficient A[j<-i] in s^-1."""
    nu = dE_h * HARTREE_TO_J / PLANCK
    return A_FACTOR * nu * nu * f


def transition_table(mc, mol, states):
    """Per-pair transition rows and per-root radiative lifetimes.

    ``states`` is a sequence of records with an ``E_hartree`` entry, one
    per root in the order of ``ci_list(mc)``.  Returns ``(rows, tau, mu,
    vl)``; only upward pairs (``j > i``) are tabulated, and a pair whose
    transition energy is not positive is skipped.
    """
    mu, vl = transition_moments(mc, mol)
    n = len(states)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            dE = states[j]["E_hartree"] - states[i]["E_hartree"]
            if dE <= 0:
                continue
            f_l, f_v = oscillator(dE, mu[i, j], vl[i, j])
            lam_nm = 1e9 * PLANCK * C_LIGHT / (dE * HARTREE_TO_J)
            nrm = float(np.linalg.norm(mu[i, j]))
            rows.append({
                "i": i, "j": j,
                "dE_hartree": dE,
                "dE_eV": dE * HARTREE_TO_EV,
                "dE_cm1": dE * HARTREE_TO_CM1,
                "lambda_nm": lam_nm,
                "mu_len_au": [float(x) for x in mu[i, j]],
                "mu_len_debye": [float(x) * AU_DIPOLE_DEBYE
                                 for x in mu[i, j]],
                "mu_len_norm_au": nrm,
                "mu_vel_au": [float(x) for x in vl[i, j]],
                "mu_vel_norm_au": float(np.linalg.norm(vl[i, j])),
                "f_len": f_l, "f_vel": f_v,
                "gauge_ratio": (f_v / f_l) if f_l != 0.0 else None,
                "gauge_rel_diff": (abs(f_v - f_l) / abs(f_l)) if f_l else None,
                "A_s-1": einstein_a(dE, f_l),
                "A_from_fvel_s-1": einstein_a(dE, f_v),
                "velocity_energy_ratio": (float(np.linalg.norm(vl[i, j])) /
                                          (dE * nrm)) if nrm > 0 else None,
            })
    tau = []
    for i in range(n):
        tot = sum(r["A_s-1"] for r in rows if r["j"] == i)
        tau.append({"root": i, "sum_A_down_s-1": tot,
                    "tau_rad_s": (1.0 / tot) if tot > 0 else None})
    return rows, tau, mu, vl
