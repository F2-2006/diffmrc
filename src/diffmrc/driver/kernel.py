"""Assembling the numerical path the driver runs.

The pieces come from the frozen kernel, the policy layer and
``driver.orbitals``; this module decides the order and the reference
they are measured against. Only one step needs pyscfad -- the SCF that
produces the reference orbitals -- and it is isolated in its own
function so that every other piece can be tested without the kernel
installed.

The partition of the orbitals is taken from the occupations, never from
the index order. See ``partition_from_occupations`` for the measurement
that made that a rule.
"""
from dataclasses import dataclass, field

import numpy as np

from .orbitals import converge_orbital_point

__all__ = ['ActiveSpace', 'partition_from_occupations']


@dataclass
class ActiveSpace:
    """The three orbital blocks, in the order the kernel expects."""

    core: list = field(default_factory=list)
    active: list = field(default_factory=list)
    virtual: list = field(default_factory=list)
    n_elec: int = 0

    @property
    def ncas(self):
        return len(self.active)

    @property
    def ncore(self):
        return len(self.core)

    @property
    def mo_order(self):
        """The column order that puts core first, then active, virtual."""
        return list(self.core) + list(self.active) + list(self.virtual)


def partition_from_occupations(occ, active_indices):
    """Split a reference occupation vector around an active set.

    Parameters
    ----------
    occ : array_like
        Occupation numbers of the reference (SCF) orbitals.
    active_indices : sequence of int
        Orbitals selected for the active space.

    Returns
    -------
    ActiveSpace

    Raises
    ------
    ValueError
        If an index falls outside the reference, or ``occ`` is not
        one-dimensional.

    Notes
    -----
    The core and virtual blocks are read off the occupations
    (``occ > 0.5`` counts as occupied), so an orbital ordering that
    departs from aufbau still yields a consistent partition. The
    electron count is the sum over the active block, which is what makes
    the ``active_electrons`` self-check meaningful: it compares this
    number against what the selection rule intended.
    """
    occ = np.asarray(occ, dtype=float)
    if occ.ndim != 1:
        raise ValueError(
            f'occ must be one-dimensional, got shape {occ.shape}')
    active = sorted({int(i) for i in active_indices})
    for i in active:
        if not 0 <= i < occ.size:
            raise ValueError(
                f'active orbital index {i} is outside 0..{occ.size - 1}'
            )
    act = set(active)
    core = [i for i in range(occ.size) if i not in act and occ[i] > 0.5]
    virtual = [i for i in range(occ.size)
               if i not in act and occ[i] <= 0.5]
    n_elec = 0
    for i in active:
        if occ[i] > 1.5:
            n_elec += 2
        elif occ[i] > 0.5:
            n_elec += 1
    return ActiveSpace(core=core, active=active, virtual=virtual,
                       n_elec=n_elec)
