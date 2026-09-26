"""Consistent active spaces along a path: the Bensberg-Reiher mapping.

The criterion needs no nuclear coordinates, only orbital shapes: two
orbitals match when the kinetic-energy difference and the shell-resolved
IAO population difference are both below their tolerances (Bensberg &
Reiher, JPCL 2023, eq. 1),

    |t_i - t_j| < tau_kin  and  sum_a |q_i^a - q_j^a| < tau_loc,

with t the kinetic expectation and q^a the population on IAO shell a.
Orbitals without a match are the ones that change most along the path
(the bond-making/breaking orbitals) and must stay in the active space at
every structure; the threshold suggestion is the minimum of the
non-matchable count over a sweep of tau (their eq. 8).

Simplification, recorded rather than hidden: the source's set-valued
machinery (within-structure grouping, bidirectional set mapping and the
consistency class of non-matchable orbitals) is reduced here to one-to-
one greedy minimum-distance matching under the same criterion.  For
non-degenerate spectra the two constructions agree; for degenerate
manifolds whose spread exceeds tau the simplified version sends the
whole manifold to the non-matchable set, which errs on the safe side
(the orbitals stay active).

Why here: the policy layer selects at a single geometry; this module is
how a selection travels along a path without the sudden kink that an
inconsistently re-selected space puts into a potential-energy curve.
The driver-side signature construction (kinetic diagonal and IAO shells)
lives in ``driver.path``; this module consumes plain numpy arrays.
"""
from dataclasses import dataclass, field

import numpy as np

from .rules import Candidate

__all__ = ['OrbitalMapping', 'match_orbitals', 'tau_sweep',
           'path_consistent_candidate']


@dataclass
class OrbitalMapping:
    """A one-to-one matching and the two non-matchable sets.

    ``pairs`` holds ``(i, j, dt, dq)``: an orbital index in each
    structure and the two margins that admitted the pair.
    """

    pairs: list = field(default_factory=list)
    unmatched_a: list = field(default_factory=list)
    unmatched_b: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


def _as_signatures(kin, q, name):
    kin = np.asarray(kin, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    if q.ndim != 2 or q.shape[0] != kin.size:
        raise ValueError(f'{name}: {kin.size} kinetic entries but '
                         f'populations of shape {q.shape}')
    return kin, q


def match_orbitals(kin_a, q_a, kin_b, q_b, *, tau_kin, tau_loc):
    """Greedy minimum-distance matching under the two-part criterion.

    Admissible edges need both margins below their tolerances; edges are
    then taken greedily in order of the normalized distance
    ``|dt|/tau_kin + sum|dq|/tau_loc`` with the index pair as a
    deterministic tie-break, so the result does not depend on array
    layout.  Tolerances must be positive; a zero tolerance admits no
    edge by the strict inequalities and is a programming error here.
    """
    kin_a, q_a = _as_signatures(kin_a, q_a, 'a')
    kin_b, q_b = _as_signatures(kin_b, q_b, 'b')
    if q_a.shape[1] != q_b.shape[1]:
        raise ValueError(f'the structures carry different shell vectors: '
                         f'{q_a.shape[1]} vs {q_b.shape[1]}')
    if not (tau_kin > 0.0) or not (tau_loc > 0.0):
        raise ValueError(f'tolerances must be positive, got tau_kin='
                         f'{tau_kin}, tau_loc={tau_loc}')
    dt = np.abs(kin_a[:, None] - kin_b[None, :])
    dq = np.abs(q_a[:, None, :] - q_b[None, :, :]).sum(axis=2)
    admissible = (dt < tau_kin) & (dq < tau_loc)
    weight = dt / tau_kin + dq / tau_loc
    edges = sorted((tuple(int(v) for v in e)
                    for e in np.argwhere(admissible)),
                   key=lambda e: (weight[e[0], e[1]], e))
    used_a, used_b, pairs = set(), set(), []
    for i, j in edges:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, float(dt[i, j]), float(dq[i, j])))
    pairs.sort()
    unmatched_a = [i for i in range(kin_a.size) if i not in used_a]
    unmatched_b = [j for j in range(kin_b.size) if j not in used_b]
    evidence = [
        f'{len(pairs)} matched, {len(unmatched_a)} unmatched on the '
        f'anchor side, {len(unmatched_b)} on the target side '
        f'(tau_kin={tau_kin:g}, tau_loc={tau_loc:g})',
    ]
    for (i, j, d_t, d_q) in pairs[:3]:
        evidence.append(f'{i} -> {j}: |dt| = {d_t:.3e}, |dq| = {d_q:.3e}')
    for i in unmatched_a:
        evidence.append(f'anchor orbital {i} has no match inside the '
                        f'tolerances')
    for j in unmatched_b:
        evidence.append(f'target orbital {j} has no match inside the '
                        f'tolerances')
    return OrbitalMapping(pairs, unmatched_a, unmatched_b, evidence)


def tau_sweep(kin_a, q_a, kin_b, q_b, taus):
    """``(tau, n_unmatched_a, n_unmatched_b)`` over a tolerance ladder.

    The source's threshold suggestion is the minimum of the total count
    (their eq. 8).  Their count is explicitly not monotone in tau (a
    looser threshold can make the set-level mapping inconsistent); the
    simplified one-to-one matching is closer to monotone, recorded as a
    boundary of this port.
    """
    rows = []
    for tau in taus:
        m = match_orbitals(kin_a, q_a, kin_b, q_b, tau_kin=float(tau),
                           tau_loc=float(tau))
        rows.append((float(tau), len(m.unmatched_a), len(m.unmatched_b)))
    return rows


def path_consistent_candidate(kin_a, q_a, kin_b, q_b, base_active, *,
                              tau_kin, tau_loc, max_nonmatchable=None):
    """Transport an anchor candidate to a second geometry.

    ``active_b = image(base_active) union non-matchable(B)``: the mapped
    image keeps the anchor's selection, and every orbital the mapping
    cannot identify must stay active because it is one of the changing
    orbitals the source's whole protocol exists for.  An anchor orbital
    without a match that was not in the base is a changing orbital the
    base missed; that is reported in the evidence rather than silently
    absorbed.

    Returns None (a refusal) when ``max_nonmatchable`` is given and the
    total non-matchable count exceeds it: the source's own condition is
    that a mapping whose unmatched set grew large says nothing.
    """
    if not len(list(base_active)):
        raise ValueError('the base candidate is empty; there is nothing '
                         'to transport')
    m = match_orbitals(kin_a, q_a, kin_b, q_b, tau_kin=tau_kin,
                       tau_loc=tau_loc)
    base = {int(i) for i in base_active}
    image = sorted(j for (i, j, _, _) in m.pairs if i in base)
    active = sorted(set(image) | set(m.unmatched_b))
    total = len(m.unmatched_a) + len(m.unmatched_b)
    evidence = list(m.evidence)
    missed = [i for i in m.unmatched_a if i not in base]
    if missed:
        evidence.append(
            f'the base missed the changing anchor orbital(s) {missed}; '
            f'they have no image and the target space carries the '
            f'target-side unmatched set instead')
    if max_nonmatchable is not None and total > max_nonmatchable:
        evidence.append(
            f'{total} non-matchable orbitals exceed max_nonmatchable '
            f'{max_nonmatchable}; the mapping is not specific enough, '
            f'refusing')
        return None
    evidence.append(f'target active space {active} '
                    f'(image {image} + unmatched {sorted(m.unmatched_b)})')
    return Candidate(active=active, per_element={}, evidence=evidence)
