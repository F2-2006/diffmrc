"""The electric dipole moment of the CASSCF layer, differentiable.

Why this file exists: the P0 spectroscopy deliverable needs a second
observable from the *same* chain that produced the Hessian, so that the
double-harmonic IR intensities carry the same provenance as the
frequencies.  The chain already materialises every ingredient -- the
canonical orbitals, the CI vector, and through ``core.nevpt2.rdm`` the
spin-summed active 1-RDM -- so the dipole is one more contraction on the
way out and no new approximation enters.

The pure assembly functions are kept numpy-only so that the block
bookkeeping (which is where a density bug would live) is testable on an
interpreter without jax; the jax factory is a thin wrapper around the
same two functions.
"""
import numpy as np

__all__ = ['dipole_from_density', 'mo_density',
           'make_casscf_dipole_theta_fn']


def mo_density(ncore, nact, nmo, dm1_act):
    """The spin-summed MO 1-RDM: 2 on core, ``dm1_act`` on active, else 0.

    The convention is the chain's own: the reference SCF orders ncore
    doubly occupied orbitals first, the nact active orbitals next.
    Getting this block wrong is silent -- every downstream number stays
    finite -- so the shape is checked here rather than trusted.
    """
    dm1_act = np.asarray(dm1_act, dtype=float)
    if dm1_act.shape != (nact, nact):
        raise ValueError(
            'the active block must be (nact, nact) = ({0}, {0}), got {1}'
            .format(nact, dm1_act.shape))
    d = np.zeros((nmo, nmo))
    d[:ncore, :ncore] = 2.0 * np.eye(ncore)
    d[ncore:ncore + nact, ncore:ncore + nact] = dm1_act
    return d


def dipole_from_density(C, D_mo, dip_ao, charges, coords):
    """``mu = sum_A Z_A R_A - Tr(C D_mo C^T . dip)`` (shape (3,)).

    The electronic trace enters with a minus sign -- ``D`` is the electron
    density with trace N_elec, and the electron charge is -1; dropping
    the sign is invisible in every invariant-free check and only the
    translation test below catches it (a neutral molecule would then
    drift with the origin).

    ``dip_ao`` is ``mol.intor('int1e_r')``, shape (3, nao, nao), taken
    from the coordinate origin; the nuclear term uses the same origin,
    and for a neutral molecule the result is origin-independent, which
    is what allows dipoles of displaced geometries to be differenced.
    """
    C = np.asarray(C, dtype=float)
    D = np.asarray(D_mo, dtype=float)
    d_ao = C @ D @ C.T
    elec = -np.einsum('pq,xpq->x', d_ao, np.asarray(dip_ao, dtype=float))
    nuc = np.einsum('a,ax->x', np.asarray(charges, dtype=float),
                    np.asarray(coords, dtype=float))
    return elec + nuc


def make_casscf_dipole_theta_fn(C0_, h_, e_, dip_, charges, coords, *,
                                ncore, nact, nele, nmo, pairs, dets,
                                ci_solver='plain', eig_solver='default'):
    """``theta -> mu(theta)`` at the CASSCF layer, differentiable.

    Deliberately a sibling of ``nevpt2.make_nevpt2_theta_fn``: the same
    ``_casscf_solution`` produces the orbitals and CI vector (so the
    dipole cannot pick up a different convention than the energies), and
    the dipole is the contraction of ``dipole_from_density`` on the way
    out.  ``dip_``, ``charges`` and ``coords`` are arrays, not Mole
    access, so the function is jit- and grad-safe the same way the
    energy functions are.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.rdm import make_dm123_sf
    from .nevpt2 import _casscf_solution

    ncore = int(ncore)
    nact = int(nact)
    nmo = int(nmo)
    charges_ = jnp.asarray(charges, dtype=float)
    coords_ = jnp.asarray(coords, dtype=float)

    def mu_theta(th_):
        _, C_can, ci2, _, _, _ = _casscf_solution(
            C0_, h_, e_, th_, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
            pairs=pairs, ci_solver=ci_solver, eig_solver=eig_solver)
        dm1, _, _ = make_dm123_sf(ci2, dets, nact, ncore)
        d_mo = jnp.zeros((nmo, nmo))
        d_mo = d_mo.at[:ncore, :ncore].set(2.0 * jnp.eye(ncore))
        d_mo = d_mo.at[ncore:ncore + nact, ncore:ncore + nact].set(dm1)
        d_ao = C_can @ d_mo @ C_can.T
        return (-jnp.einsum('pq,xpq->x', d_ao, dip_)
                + jnp.einsum('a,ax->x', charges_, coords_))

    return mu_theta
