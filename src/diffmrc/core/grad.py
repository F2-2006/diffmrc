"""Nuclear gradient for the differentiable CASSCF chain.

FORM
----

    C0 = pscf.RHF(m_).mo_coeff              # a function of the geometry
    C_ = C0 @ R(stop_gradient(theta_star))  # theta frozen at the converged point

Both elements are required.

1. THE REFERENCE ORBITALS MUST BE A FUNCTION OF THE GEOMETRY.
   C0 is a coefficient matrix in the atomic-orbital basis, and the AO basis
   moves with the nuclei. Holding a fixed numerical matrix across a displaced
   geometry represents a physically different set of orbitals.

2. THETA MUST BE FROZEN AT THE STATIONARY POINT OBTAINED FROM THOSE ORBITALS.
   theta is defined relative to the reference orbitals, so the same theta with
   a geometry-dependent C0 gives the same physical orbitals; holding theta
   fixed lets the orbitals follow the reference set as it moves, which is what
   the envelope theorem requires.

VERIFICATION
------------

Against the closed-form analytic gradient of an independent program, and
against two reference-free conditions - translational invariance
(sum_a dE/dR_a = 0) and rotational invariance (sum_a R_a x dE/dR_a = 0) -
which hold to 1e-14 or better.

APPROACHES THAT DO NOT WORK
---------------------------

1. Taking C0 from a converged CASSCF calculation. C_CASSCF(R0) is a product of
   that geometry and is not the orbital set of a displaced geometry; the error
   is of order 4e-3 Eh/bohr.

2. Rotating the reference orbitals onto the CASSCF orbitals with theta. The
   rotation set spans inter-block rotations only, whereas the active block also
   requires intra-block rotations; the residual is 0.97.

3. Holding C0 fixed at the reference geometry. Same failure as (1), giving a
   spurious gradient of order 4e-1 Eh/bohr.

A CAUTION ON TEST SYSTEMS. Water is a poor verification target for this layer:
CAS(2,2) barely changes its orbitals, so approach (1) returns a plausible-
looking 3e-5 on water while failing by three orders of magnitude on CO. Test on
systems where the orbitals actually move.

Agreement with a finite difference of the same code path is necessary but not
sufficient: it establishes only that the derivative and the energy are
consistent with each other, not that either is correct. An independent
reference is required.

USAGE
-----

    from grad import (make_energy_fn, make_grad_fn, solve_theta_star,
                      nuclear_gradient)

    E_cas = make_energy_fn(ncore, nact, nele)
    th_star, gnorm = solve_theta_star(mol, C_rhf, E_theta, npar, optimize_theta)
    g = nuclear_gradient(mol, make_grad_fn(nmo, pairs, th_star, E_cas, pscf))
    # g has shape (natm, 3), in Eh/bohr
"""
import numpy as np
import jax
from jax import numpy as jnp

from .rotation import block_rotation
from .nevpt2.m4 import cas_ci_vector


def make_energy_fn(ncore, nact, nele, use_blocked=False):
    """Build a differentiable function from a Mole to the CAS energy.

    Parameters
    ----------
    ncore, nact, nele : int
        Active-space definition.
    use_blocked : bool
        Use the S_z-blocked diagonalization (cs_casci_blocked). Required for
        second derivatives; identical in the first derivative.
    """
    if use_blocked:
        from .casci import cs_casci_blocked as _solver

        def _E(C, h, e, en):
            nm = ncore + nact
            Cj = jnp.asarray(C)
            h1 = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
            e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj, e)[
                :nm, :nm, :nm, :nm]
            return _solver(h1, e1, ncore=ncore, nact=nact, nele_act=nele)[0] + en
    else:
        def _E(C, h, e, en):
            return cas_ci_vector(C, h, e, en, ncore=ncore, nact=nact,
                                 nele_act=nele)[0]
    return _E


def make_grad_fn(nmo, pairs, th_star, E_cas_of, pscf):
    """Build a differentiable function from a Mole to the nuclear gradient.

    Parameters
    ----------
    nmo : int
        Number of basis functions.
    pairs : list[(int, int)]
        Orbital rotation pairs; must include core<->virtual
        (see orblib.orbital_pairs).
    th_star : jnp.ndarray
        Stationary point obtained starting from the RHF orbitals.
    E_cas_of : callable(C, h1e, eri, enuc) -> energy
    pscf : module
        pyscfad.scf, passed in to avoid a hard dependency in this module.
    """
    TH_SG = jax.lax.stop_gradient(jnp.asarray(th_star))

    def E_geo(m_):
        mf_ = pscf.RHF(m_); mf_.kernel()
        C_ = mf_.mo_coeff                          # geometry-dependent
        h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
        e_ = m_.intor("int2e", aosym="s1")
        return E_cas_of(C_ @ block_rotation(TH_SG, nmo, pairs), h_, e_,
                        m_.energy_nuc())

    return E_geo


def solve_theta_star(mol, C_rhf, E_theta, npar, optimize_theta, **kw):
    """Find theta*, starting from the RHF orbitals.

    Parameters
    ----------
    mol : pyscfad Mole
        Mole at the reference geometry.
    C_rhf : jnp.ndarray
        RHF orbital coefficients at the reference geometry.
    E_theta : callable(th) -> energy
    npar : int
        Number of rotation parameters.
    optimize_theta : callable
        The orbital optimizer (see optim_theta.optimize_theta).

    Returns
    -------
    th_star : jnp.ndarray
    gnorm : float
        Norm of the orbital gradient at the reported point; a useful
        self-check. Values above 1e-6 indicate the optimization has not
        converged and the result should not be used.
    """
    _, th, g = optimize_theta(mol, C_rhf, E_theta, npar, **kw)
    return jnp.asarray(np.asarray(th)), float(g)


def nuclear_gradient(mol, E_geo_fn):
    """Nuclear gradient in Eh/bohr, returned as an (natm, 3) array.

    Parameters
    ----------
    mol : pyscfad Mole
        Target geometry.
    E_geo_fn : callable(Mole) -> scalar energy
        Constructed by make_grad_fn.

    Returns
    -------
    g : np.ndarray, shape (natm, 3)

    Notes
    -----
    The result of jax.grad on a Mole is itself a Mole-shaped pytree; only the
    .coords field holds the nuclear coordinates.
    """
    g = jax.grad(E_geo_fn)(mol)
    return np.asarray(g.coords)
