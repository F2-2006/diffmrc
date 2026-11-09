"""MRCI configuration space construction.

PURPOSE
-------
Partitions the determinants of an active-plus-external orbital window
into the three configuration classes of Werner & Knowles,
J. Chem. Phys. 1988, 89, 5803, Eq. (1): internal (no external
electrons), singly external, and doubly external.  The union is the
uncontracted MRCISD space used by ``workflow.run_mrci``.

PUBLIC INTERFACE
----------------
``build_space``, ``n_external_electrons``, ``reference_vector``,
``active_pairs``, ``external_pairs``.

CONVENTIONS
-----------
Orbital layout inside the CI space: spatial orbitals 0..nact-1 are
active, nact..nact+next-1 are external.  Core orbitals are folded into
the effective one-electron Hamiltonian and are not part of the CI space.
Determinant bitmasks follow ``diffmrc.core.mrci.determinants``.

VALIDATED SCOPE
---------------
The resulting spaces are the ones used in the H2O and CH2 benchmarks of
``tests/core/test_mrci.py``; the uncontracted space equals full CI for a
two-electron active space.

BOUNDARIES
----------
* Only one M_S sector is built (``ms`` defaults to 0 and is the only
  tested value).
* The partition uses the external-electron count only; no symmetry or
  RAS restrictions are applied.
"""

from __future__ import annotations

import itertools

import numpy as np

from .determinants import dets_with_ms


def n_external_electrons(mask: int, nact: int, norb_ci: int) -> int:
    """Number of electrons in external spin-orbitals."""
    ext = 0
    for p in range(nact, norb_ci):
        ext += (mask >> p) & 1
    for p in range(norb_ci + nact, 2 * norb_ci):
        ext += (mask >> p) & 1
    return ext


def build_space(nact: int, next_: int, n_elec: int, ms: float = 0.0):
    """Build the MRCISD determinant space.

    Returns
    -------
    internal, singles, doubles : list[int]
        Determinant bitmasks over 2*(nact+next_) spin-orbitals, alpha block
        first.  The union internal + singles + doubles is the uncontracted
        MRCISD space.
    """
    norb_ci = nact + next_
    all_dets = dets_with_ms(norb_ci, n_elec, ms)
    internal, singles, doubles = [], [], []
    for d in all_dets:
        k = n_external_electrons(d, nact, norb_ci)
        if k == 0:
            internal.append(d)
        elif k == 1:
            singles.append(d)
        elif k == 2:
            doubles.append(d)
    return internal, singles, doubles


def reference_vector(h_int: np.ndarray, internal: list[int], root: int = 0):
    """Diagonalize the internal-space Hamiltonian (CASCI) and return one root."""
    w, v = np.linalg.eigh(h_int)
    return w[root], v[:, root], w


def active_pairs(nact: int):
    """Spatial active-orbital pairs (i >= j)."""
    return [(i, j) for i in range(nact) for j in range(i + 1)]


def external_pairs(nact: int, next_: int):
    """Spatial external-orbital pairs (a >= b), indices in CI-space numbering."""
    return [(a, b) for a in range(nact, nact + next_) for b in range(nact, a + 1)]
