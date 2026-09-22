"""Implicit-function second derivatives of the orbital-rotation energy.

The orbital parameters theta are defined by a stationarity condition,
g(theta*(R), R) = dE/dtheta = 0, so the relaxed energy as a function of
the geometry is E*(R) = E(theta*(R), R). Differentiating the condition
gives the orbital response

    dtheta*/dR = -H_tt^-1 M,     M = d2E/dtheta dR,  H_tt = d2E/dtheta2

and the second derivative of the relaxed energy is

    d2E*/dR2 = H_RR + M^T (-H_tt^-1 M)

with H_RR the Hessian at frozen theta. The second term is the response
contribution; dropping it gives the frozen approximation, which this
project measured to be off by 1e-5 in the frequencies, and which moves
every frequency in the same direction once included.

Why a pseudo-inverse, not a solve
---------------------------------

The orbital Hessian is singular along directions that leave the energy
invariant -- rotations within a block, for instance. For N2 it reaches a
condition number of 2.6e15. A direct solve inverts that near-null
direction and amplifies whatever the right-hand side has there; the
pseudo-inverse drops it, which is correct because the gradient component
along an exact zero-curvature direction is zero and contributes nothing
to dtheta*/dR.

The cutoff is relative (``rcond``), matching ``numpy.linalg.pinv``. An
earlier script in this project used an absolute singular-value threshold
of 1e-3; on a matrix with condition number 1e12 the two retain five and
two singular values respectively and produce solutions differing by
99.9%. Whether that difference reaches a physical quantity depends on
the component of M along the discarded directions, which
``experiments/verify_rcond_choice.py`` measures on a real system rather
than assuming.

Conventions that matter
-----------------------

* ``theta`` must be ``stop_gradient``-ed in the frozen Hessian, or the
  frozen term silently acquires the orbital response and the two routes
  stop being distinguishable.
* Every Hessian is symmetrized as ``0.5*(H + H.T)``: an AD Hessian is
  symmetric only to round-off, and both ``eigvalsh`` and ``pinv`` assume
  exact symmetry.
* ``M`` is deliberately left unsymmetrized. Symmetry is a property of
  the assembled ``H_total``, not of ``M``.
* The ``.coords`` attribute of a differentiated Mole is nested
  differently depending on whether the outermost operation is ``grad``
  or ``jacfwd``/``hessian`` of a ``grad``. Taking the wrong level is a
  silent shape error, so the two helpers below are kept separate.
"""
from dataclasses import dataclass

import numpy as np
import jax

from .orblib import orbital_pairs  # noqa: F401  (re-exported for callers)
from .rotation import block_rotation

__all__ = ['ImplicitHessianResult', 'make_joint_energy_fn', 'implicit_hessian']

DEFAULT_RCOND = 1e-9


@dataclass
class ImplicitHessianResult:
    """The assembled second derivative and the diagnostics behind it."""

    H_tt: np.ndarray
    H_tt_inverse: np.ndarray
    curvature: object
    mixed: np.ndarray
    H_RR: np.ndarray
    H_response: np.ndarray
    H_total: np.ndarray
    dtheta_dR_max: float
    rcond: float


def make_joint_energy_fn(ncore, nact, nele, nmo, pairs, pscf,
                         use_blocked=True):
    """Build E(theta, mol), differentiable in both arguments.

    The reference orbitals and the integrals are recomputed inside the
    function from the Mole. Holding them fixed across a displaced
    geometry represents a different set of orbitals; this project
    recorded the resulting error as 4e-3 Eh/bohr for the mildest case.

    Parameters
    ----------
    ncore, nact, nele : int
        Active-space definition.
    nmo : int
        Number of basis functions.
    pairs : list of (int, int)
        Orbital rotation pairs. Must include core<->virtual.
    pscf : module
        ``pyscfad.scf``, injected to avoid a hard dependency here.

    Returns
    -------
    callable
        ``(mol, theta) -> energy``.
    """
    from .casci import cs_casci_blocked

    def E_joint(m_, th_):
        mf_ = pscf.RHF(m_)
        mf_.kernel()
        C_ = mf_.mo_coeff
        h_ = m_.intor('int1e_kin') + m_.intor('int1e_nuc')
        e_ = m_.intor('int2e', aosym='s1')
        C2 = C_ @ block_rotation(th_, nmo, pairs)
        nm = ncore + nact
        h1 = jax.numpy.einsum('ai,ab,bj->ij', C2, h_, C2)[:nm, :nm]
        e1 = jax.numpy.einsum('pi,qj,rk,sl,pqrs->ijkl', C2, C2, C2, C2,
                              e_)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + m_.energy_nuc()

    return E_joint


def _mixed_derivative(E_joint, mol, theta):
    """M = d2E/dtheta dR, taken as jacfwd of the theta-gradient."""
    g_of = jax.grad(E_joint, argnums=1)
    M = jax.jacfwd(g_of, argnums=0)(mol, theta)
    n_par = int(np.asarray(theta).size)
    return np.asarray(M.coords).reshape(n_par, -1)


def _frozen_hessian(E_joint, mol, theta, n_par):
    """H_RR = d2E/dR2 with theta held fixed."""
    th_sg = jax.lax.stop_gradient(theta)

    def E_frozen(m_):
        return E_joint(m_, th_sg)

    H = jax.jacfwd(jax.grad(E_frozen))(mol)
    return np.asarray(H.coords.coords).reshape(n_par, n_par)


def implicit_hessian(mol, theta, E_joint, rcond=DEFAULT_RCOND):
    """Assemble the relaxed-geometry Hessian at a stationary theta.

    Parameters
    ----------
    mol : pyscfad Mole
        Reference geometry.
    theta : array_like
        Orbital parameters at the stationary point for that geometry.
    E_joint : callable
        ``(mol, theta) -> energy``, as built by `make_joint_energy_fn`.
    rcond : float
        Relative cutoff for the pseudo-inverse of the orbital Hessian.

    Returns
    -------
    ImplicitHessianResult
        The total Hessian plus every intermediate, so a caller can see
        how much of it came from the orbital response rather than only
        the total.
    """
    from .stationary import curvature_report

    theta = jax.numpy.asarray(np.asarray(theta, dtype=float))
    n_par = int(np.asarray(theta).size)

    H_tt = np.asarray(jax.hessian(lambda t: E_joint(mol, t))(theta))
    H_tt = 0.5 * (H_tt + H_tt.T)

    M = _mixed_derivative(E_joint, mol, theta)
    n_r = M.shape[1]

    H_RR = _frozen_hessian(E_joint, mol, theta, n_r)
    H_RR = 0.5 * (H_RR + H_RR.T)

    H_tt_inv = np.linalg.pinv(H_tt, rcond=rcond)
    D = -H_tt_inv @ M
    H_resp = M.T @ D
    H_total = H_RR + H_resp
    H_total = 0.5 * (H_total + H_total.T)

    return ImplicitHessianResult(
        H_tt=H_tt, H_tt_inverse=H_tt_inv,
        curvature=curvature_report(H_tt),
        mixed=M, H_RR=H_RR, H_response=H_resp, H_total=H_total,
        dtheta_dR_max=float(np.abs(D).max()) if D.size else 0.0,
        rcond=float(rcond),
    )
