"""Active spaces by projection onto a target subspace.

Two rules share one piece of mathematics, the overlap-metric projector
P = X (X^T S X)^{-1} X^T S onto the span of target directions X:

- AVAS (Sayfutyarova et al. 2017): targets are atomic orbitals, the
  occupied and virtual blocks are rotated by the eigenvectors of the
  projected overlaps, and the rotated slots whose overlap with the
  target is above a threshold enter the active space.  The rotated
  coefficient matrix is returned because the active orbitals are
  combinations, not a subset, of the input MOs; ``active`` indexes the
  rotated slots.
- iCAS (Lei et al. 2021): targets are arbitrary probe directions;
  orbital i's weight in probe x is n_{x,i} = y^T G^{-1} y with
  y = X^T S c_i and G = X^T S X; each probe claims its highest-weight
  unclaimed orbital, and a gap below the tolerance refuses the
  assignment (the probe set does not separate the orbitals).

Refusal is a result: the probe rule returns None, AVAS raises only for
ill-formed input.  AVAS's own caveat travels: the spectra can falsify a
target choice but never confirm it.  Open shells are refused (the
source's option 3, keeping every singly occupied orbital active, is the
recorded extension).  Ported from the fBlockKit implementations
(read-only reference); the AVAS target set here is the calculation's own
AO subset, the route the source describes in its outlook and the only
one open to the f block, where the standard minimal auxiliary sets have
no coverage.
"""
from dataclasses import dataclass, field

import numpy as np

from .rules import Candidate

__all__ = ['AVAS_DEFAULT_THRESHOLD', 'AvasChoice', 'avas_selection',
           'probe_candidate']

#: AVAS overlap threshold; the source uses 0.05-0.1, the only
#: user-chosen number of the construction.
AVAS_DEFAULT_THRESHOLD = 0.05


@dataclass
class AvasChoice:
    """The rotated-coefficient active space and the spectra it came from."""

    active: list = field(default_factory=list)
    n_electrons: int = 0
    C_rotated: np.ndarray = field(default_factory=lambda: np.array([]))
    spectra_occupied: np.ndarray = field(
        default_factory=lambda: np.array([]))
    spectra_virtual: np.ndarray = field(
        default_factory=lambda: np.array([]))
    evidence: list = field(default_factory=list)


def _projected_spectrum(C_block, S, X):
    """Eigenvalues (descending) and rotation of the target projector.

    The projected overlap matrix is B = (S C)_A^T sigma^-1 (S C)_A with
    (S C)_A = X^T S C and sigma = X^T S X (source eqs. 4-9); its
    eigenvectors rotate the block so the leading columns carry the
    target overlap.
    """
    SC_A = X.T @ S @ C_block
    sigma = X.T @ S @ X
    B = SC_A.T @ np.linalg.solve(sigma, SC_A)
    B = 0.5 * (B + B.T)
    w, U = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    return w[order], U[:, order]


def avas_selection(C, S, target_aos, occ, *,
                   threshold=AVAS_DEFAULT_THRESHOLD):
    """AVAS active space from target atomic orbitals (closed shells only).

    Parameters
    ----------
    C : array_like
        MO coefficients, shape (nao, nmo).
    S : array_like
        AO overlap, shape (nao, nao).
    target_aos : sequence of int
        Rows of ``C``/``S`` forming the target set (the calculation's own
        AOs; a minimal auxiliary basis is not usable for the f block,
        where the standard minimal sets have no coverage).
    occ : array_like
        MO occupations; values in {0, 2} required (open shells refused).
    threshold : float
        Overlap eigenvalue above which a rotated slot is active.

    Returns
    -------
    AvasChoice
        ``active`` indexes the rotated slots; ``C_rotated`` shares the
        column order.  ``n_electrons`` counts 2 per selected occupied
        slot.
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    occ = np.asarray(occ, dtype=float)
    if C.ndim != 2 or S.shape != (C.shape[0], C.shape[0]):
        raise ValueError(f'C {C.shape} and S {S.shape} do not describe '
                         f'the same AO space')
    if occ.shape != (C.shape[1],):
        raise ValueError(f'occ has shape {occ.shape} for {C.shape[1]} '
                         f'orbitals')
    bad = np.unique(occ[(occ > 1e-8) & (np.abs(occ - 2.0) > 1e-8)])
    if bad.size:
        raise ValueError(
            f'fractional or singly occupied orbitals {bad.tolist()}; '
            f'open-shell AVAS keeps every singly occupied orbital active '
            f'(the source option 3) and is not implemented here -- '
            f'closed shells only')
    if not 0.0 < threshold < 1.0:
        raise ValueError(f'threshold {threshold} outside (0, 1)')
    occ_mask = occ > 1.0
    rows = np.asarray(sorted({int(a) for a in target_aos}), dtype=int)
    if rows.size == 0 or rows.min() < 0 or rows.max() >= C.shape[0]:
        raise ValueError(f'target rows {rows.tolist()} outside the AO '
                         f'space of size {C.shape[0]}')
    X = np.eye(C.shape[0])[:, rows]

    w_occ, U_occ = _projected_spectrum(C[:, occ_mask], S, X)
    w_vir, U_vir = _projected_spectrum(C[:, ~occ_mask], S, X)
    keep_occ = w_occ > threshold
    keep_vir = w_vir > threshold

    C_rotated = C.copy()
    C_rotated[:, occ_mask] = C[:, occ_mask] @ U_occ
    C_rotated[:, ~occ_mask] = C[:, ~occ_mask] @ U_vir
    active = ([int(i) for i in np.where(occ_mask)[0][keep_occ]]
              + [int(i) for i in np.where(~occ_mask)[0][keep_vir]])
    n_el = 2 * int(keep_occ.sum())
    evidence = [
        f'target {rows.size} AO(s); occupied block keeps '
        f'{int(keep_occ.sum())} of {int(occ_mask.sum())} (eigenvalues '
        f'{np.round(w_occ, 4).tolist()}), virtual block keeps '
        f'{int(keep_vir.sum())} of {int((~occ_mask).sum())} '
        f'(eigenvalues {np.round(w_vir, 4).tolist()}) at threshold '
        f'{threshold}',
        'the spectra can falsify a target choice but not confirm it '
        '(source caveat); all-small virtual overlaps mean the '
        'antibonding target character is ligand-centred',
    ]
    return AvasChoice(active, n_el, C_rotated, w_occ, w_vir, evidence)


def probe_candidate(C, S, probes, *, tol=1e-2, pool=None):
    """iCAS assignment: each probe subspace claims its best orbital.

    ``probes`` is a sequence of probe directions or subspaces; each is a
    length-nao vector or a (nao, k) matrix.  Orbital i's weight in probe
    x is ``n_{x,i} = y^T G^{-1} y`` with ``y = X^T S c_i`` and
    ``G = X^T S X``.  Each probe takes the unclaimed orbital of largest
    weight (ties resolve to the smaller orbital index); if the smallest
    claimed weight minus the largest unclaimed one is below ``tol`` the
    assignment is refused (the probe set does not separate the
    orbitals).

    Returns a ``Candidate`` (evidence records every weight and the gap)
    or None on refusal; a caller must treat None as a refusal.
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    if C.ndim != 2 or S.shape != (C.shape[0], C.shape[0]):
        raise ValueError(f'C {C.shape} and S {S.shape} do not describe '
                         f'the same AO space')
    probes = [np.asarray(X, dtype=float) for X in probes]
    probes = [X[:, None] if X.ndim == 1 else X for X in probes]
    pool = (list(range(C.shape[1])) if pool is None
            else [int(i) for i in pool])
    if len(probes) == 0 or len(probes) >= len(pool):
        raise ValueError(f'{len(probes)} probes for {len(pool)} pooled '
                         f'orbitals; each probe needs a distinct orbital')
    n_i = np.zeros((len(probes), len(pool)))
    for x, X in enumerate(probes):
        if X.shape[0] != C.shape[0]:
            raise ValueError(f'probe {x} has {X.shape[0]} AO rows for '
                             f'{C.shape[0]}')
        G = X.T @ S @ X
        Y = X.T @ S @ C[:, pool]                    # (k, npool)
        n_i[x] = np.einsum('ki,ij,kj->i', Y, np.linalg.inv(G), Y)
    chosen, evidence = [], []
    free = list(range(len(pool)))
    for x in range(len(probes)):
        best = max(free, key=lambda t: (n_i[x, t], -pool[t]))
        free.remove(best)
        chosen.append(best)
        gap = n_i[x, best] - (max(n_i[x, t] for t in free) if free
                              else 0.0)
        evidence.append(
            f'probe {x}: orbital {pool[best]} with n = '
            f'{n_i[x, best]:.6g}, gap to the next unclaimed = {gap:.3g}')
        if gap < tol:
            evidence.append(f'gap {gap:.3g} below {tol:g}; the probe set '
                            f'does not separate the orbitals, refusing')
            return None
    active = sorted(pool[t] for t in chosen)
    evidence.append(f'{len(probes)} probes selected {active}')
    return Candidate(active=active, per_element={}, evidence=evidence)
