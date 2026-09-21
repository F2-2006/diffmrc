"""Standard definition of the orbital-rotation parameter set.

Orbital relaxation is parametrized by an exponential map of antisymmetric
rotation parameters, C = C0 @ R(theta), so that the orbital-optimization step
becomes a continuous function of a fixed-size parameter vector and the envelope
theorem applies to the first derivative.

THE PARAMETER SET MATTERS. The rotations must span every non-redundant
inter-block pair:

    (i, a)  core <-> active
    (i, v)  core <-> virtual
    (a, v)  active <-> virtual

Intra-block rotations (core<->core, active<->active, virtual<->virtual) leave
the energy invariant and are omitted as redundant.

Omitting the core<->virtual rotations leaves the orbital optimization at a
non-stationary point. For N2/6-31G with CAS(2,2), evaluated against an
independent CASSCF implementation:

    parameter set                    E_cas                deviation (kcal/mol)
    ------------------------------------------------------------------------
    (i,a) + (a,v)    [32 params]   -108.903578776806     0.045036
    + (i,v)          [92 params]   -108.903650545967     8e-10
    + (i,j)         [107 params]   -108.903650545967     8e-10

Adding the intra-core pairs changes nothing, confirming that they are
redundant; omitting core<->virtual changes the answer in the fourth decimal
place of a relative energy.

This set must be used together with `block_rotation`, which accumulates Givens
rotations in the order the pairs are listed.

Usage:

    from orblib import orbital_pairs
    pairs = orbital_pairs(ncore=6, nact=2, nmo=18)

    # or infer nmo from the shape of C0
    pairs = orbital_pairs_from_C0(C0, ncore=6, nact=2)
"""
import numpy as np

__all__ = ['orbital_pairs', 'orbital_pairs_from_C0', 'n_pairs']


def orbital_pairs(ncore, nact, nmo, include_cv=True):
    """Non-redundant inter-block rotation pairs for a Givens parametrization.

    Parameters
    ----------
    ncore : int
        Number of frozen core orbitals.
    nact : int
        Number of active orbitals.
    nmo : int
        Total number of orbitals.
    include_cv : bool
        Whether to include core<->virtual rotations. Default True, which is the
        correct choice; set False only to reproduce results obtained with the
        incomplete parameter set.

    Returns
    -------
    list[(int, int)]
        Ordered as (i,a), then (i,v), then (a,v).
    """
    act_lo, act_hi = ncore, ncore + nact
    vir_lo = ncore + nact

    pairs = []
    # core <-> active
    for i in range(ncore):
        for a in range(act_lo, act_hi):
            pairs.append((i, a))
    # core <-> virtual
    if include_cv:
        for i in range(ncore):
            for v in range(vir_lo, nmo):
                pairs.append((i, v))
    # active <-> virtual
    for a in range(act_lo, act_hi):
        for v in range(vir_lo, nmo):
            pairs.append((a, v))
    return pairs


def orbital_pairs_from_C0(C0, ncore, nact, include_cv=True):
    """Infer nmo from the shape of C0 and build the parameter set."""
    nmo = np.asarray(C0).shape[1]
    return orbital_pairs(ncore, nact, nmo, include_cv=include_cv)


def n_pairs(ncore, nact, nmo, include_cv=True):
    """Number of parameters, without materializing the list."""
    return len(orbital_pairs(ncore, nact, nmo, include_cv=include_cv))
