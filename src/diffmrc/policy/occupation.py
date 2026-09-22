"""Partitioning orbitals by their occupation numbers.

Everything the selection rules need to know about *where* the electrons
are, expressed as plain functions over an occupation vector. Keeping this
apart from the rules themselves means the rules read as chemistry, while
these functions are tested on hand-built numbers.

Nothing here knows about elements or shells; that lives in ``rules``.
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = ['OccupationPartition', 'BlockPartition', 'split_by_occupation',
           'select_valence', 'partition_blocks']

DOUBLY_THRESHOLD = 1.5
SINGLY_THRESHOLD = 0.5


@dataclass
class OccupationPartition:
    """Orbital indices grouped by how many electrons they carry."""

    singly: list = field(default_factory=list)
    doubly: list = field(default_factory=list)
    empty: list = field(default_factory=list)


@dataclass
class BlockPartition:
    """Orbital indices grouped by their role in a CAS calculation."""

    core: list = field(default_factory=list)
    active: list = field(default_factory=list)
    virtual: list = field(default_factory=list)
    ncore: int = 0
    n_elec_active: int = 0


def split_by_occupation(occ, doubly_threshold=DOUBLY_THRESHOLD,
                        singly_threshold=SINGLY_THRESHOLD):
    """Group orbital indices into singly occupied, doubly occupied, empty.

    Parameters
    ----------
    occ : array_like
        Occupation numbers, one per orbital. A converged SCF leaves these
        a few digits away from integers, so the grouping uses thresholds
        rather than equality.
    doubly_threshold : float
        Above this an orbital counts as doubly occupied.
    singly_threshold : float
        Above this, and below `doubly_threshold`, it counts as singly
        occupied.

    Returns
    -------
    OccupationPartition
        Three ascending index lists.

    Raises
    ------
    ValueError
        If `occ` is not one-dimensional.
    """
    occ = np.asarray(occ, dtype=float)
    if occ.ndim != 1:
        raise ValueError(f'occ must be one-dimensional, got shape {occ.shape}')
    return OccupationPartition(
        singly=[int(i) for i in np.where(
            (occ > singly_threshold) & (occ < doubly_threshold))[0]],
        doubly=[int(i) for i in np.where(occ > doubly_threshold)[0]],
        empty=[int(i) for i in np.where(occ < singly_threshold)[0]],
    )


def select_valence(occ, n_orbitals):
    """Choose a main-group valence shell from occupation structure.

    The valence shell of a main-group atom is its singly occupied
    orbitals plus the highest doubly occupied ones, which is what the
    Aufbau filling leaves at the top of the occupied block.

    Parameters
    ----------
    occ : array_like
        Occupation numbers.
    n_orbitals : int
        How many orbitals the shell should contain.

    Returns
    -------
    list[int] or None
        Ascending orbital indices, or None when the request cannot be
        satisfied from the occupation structure. A caller must treat None
        as a refusal, not as an empty selection.

    Notes
    -----
    Taking the *highest* doubly occupied orbitals is a position
    assumption: it holds when the SCF ordering follows the Aufbau
    principle, which it does for the main-group atoms this rule targets.
    It is not valid for a transition metal with several partially filled
    d shells, where the ordering of the d block relative to the s and p
    blocks is not fixed.
    """
    if n_orbitals < 1:
        return None
    partition = split_by_occupation(occ)
    n_from_doubly = n_orbitals - len(partition.singly)
    if n_from_doubly < 0 or len(partition.doubly) < n_from_doubly:
        return None
    if n_from_doubly == 0:
        return sorted(partition.singly)
    return sorted(partition.doubly[-n_from_doubly:] + partition.singly)


def partition_blocks(occ, active):
    """Split all orbitals into core, active, and virtual.

    The three blocks are disjoint and cover every orbital. An orbital
    listed as active is never also counted as core, which is what keeps
    the electron count consistent with the nuclear charge.

    Parameters
    ----------
    occ : array_like
        Occupation numbers.
    active : sequence of int
        Orbital indices chosen for the active space.

    Returns
    -------
    BlockPartition

    Raises
    ------
    ValueError
        If an index in `active` is out of range.
    """
    occ = np.asarray(occ, dtype=float)
    active = sorted({int(i) for i in active})
    if active and (active[0] < 0 or active[-1] >= len(occ)):
        raise ValueError(
            f'active index out of range for {len(occ)} orbitals: {active}'
        )
    active_set = set(active)
    partition = split_by_occupation(occ)
    core = [i for i in partition.doubly if i not in active_set]
    core_set = set(core)
    virtual = [i for i in range(len(occ))
               if i not in core_set and i not in active_set]
    n_elec = sum(2 if occ[i] > DOUBLY_THRESHOLD else 1 for i in active)
    return BlockPartition(core=core, active=active, virtual=virtual,
                          ncore=len(core), n_elec_active=int(n_elec))
