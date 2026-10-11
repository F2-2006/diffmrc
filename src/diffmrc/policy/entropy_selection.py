"""Entropy-driven active-space selection: the spectrum becomes a choice.

The verification module judges a rule-based candidate against the
entropy spectrum; that comparison presupposes the candidate exists.
When the chemical rules refuse -- a composition outside their scope, a
mixed orbital they will not label -- the spectrum itself has to supply
the answer, and this module builds it. It reads the same two objects
the verification reads: the single-orbital spectrum
(:mod:`diffmrc.analysis.entropy`) and the mutual-information matrix
(:func:`diffmrc.analysis.ci.mutual_information`; exact from a CI vector
through the eigenproblem of the 16x16 pair density, per
Stein/Pantazis/Krewald 2019).

Two selectors and one check:

- :func:`plateau_selection` is the autoCAS protocol's own selection
  step. The source implementation (SCINE autoCAS) normalizes the
  spectrum, sweeps a threshold over [0, 1) in steps, and reads off the
  widest run of steps over which the orbital count does not change; the
  count of that plateau is the active-space size, taken top-ranked
  first. :func:`diffmrc.analysis.plateau.detect_plateaus` already finds
  the runs; this function turns the largest one into the orbital
  choice. With no plateau it falls back to the weak-correlation line
  (orbitals above 2% of the largest entropy, Stein & Reiher 2016), the
  same fallback the source describes and the verification module
  reports in its NO_PLATEAU branch.

- :func:`mi_selection` grows a set from the largest single-orbital
  entropy by repeatedly adding the orbital with the strongest mutual
  information to the set, stopping when no outside orbital exceeds a
  threshold fraction of the largest pair value. This is a choice of
  this package, not a literature algorithm: autoCAS itself selects by
  the s1 plateau alone -- its interface carries s2 and
  mutual-information arrays without consuming them. The fraction
  threshold therefore has no literature value and is a documented
  default that travels in the evidence.

- :func:`mi_closure` checks any candidate -- rule-based, plateau-based,
  hand-given -- against the pair structure: an outside orbital with
  strong mutual information to the set is a *leak* (the candidate
  misses an orbital that entangles with it); an inside orbital with no
  strong partner is *isolated* (the candidate includes an orbital the
  entanglement does not couple to the rest). The report names both
  directions rather than a verdict: either can be legitimate (a leak
  can sit on a plateau boundary; an isolated member can be an unpaired
  orbital with no partner).

A spectrum is only read this way in a localized-orbital basis (the
autoCAS precondition, and this project's t177 measurement: a truncated
space degrades into numerical noise). The thresholds are not mixing
lines: 0.14 and the 2% fraction are Stein & Reiher's, the MI fraction
is this package's own.
"""
from dataclasses import dataclass, field

import numpy as np

from ..analysis.diagnostics import (
    SINGLE_REFERENCE_THRESHOLD,
    WEAK_CORRELATION_THRESHOLD,
    is_single_reference,
)
from ..analysis.plateau import detect_plateaus

__all__ = ['EntropySelection', 'plateau_selection', 'mi_selection',
           'MI_FRACTION_DEFAULT', 'MiClosureReport', 'mi_closure']

#: Threshold fraction of the largest mutual-information value below
#: which a pair counts as uncoupled. A package choice, not a literature
#: value (see the module docstring); it travels in every evidence list.
MI_FRACTION_DEFAULT = 0.05

#: Entropies agreeing to this tolerance rank as equal and order by index;
#: physically degenerate partners differ only in floating-point tails
#: (the same convention as policy.ranking).
_TIE_TOL = 1e-12


@dataclass
class EntropySelection:
    """The selected active space and the evidence for every step.

    A refusal carries ``active == []`` with the reason in ``evidence``
    (the convention of every selection rule in this layer).
    """

    active: list = field(default_factory=list)
    method: str = ''
    plateau_height: int = 0
    evidence: list = field(default_factory=list)


def _validate_spectrum(s1):
    s1 = np.asarray(s1, dtype=float).reshape(-1)
    if not np.isfinite(s1).all():
        raise ValueError('the entropy spectrum contains non-finite '
                         'values; fix the extraction, not the selection')
    return s1


def plateau_selection(s1, plateau_values=10, threshold_step=0.01):
    """Select by the largest plateau of the entropy spectrum.

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies over the full space.
    plateau_values, threshold_step : int, float
        Plateau-detection settings, forwarded to the analysis layer
        (defaults are the autoCAS protocol's 10 and 0.01).

    Returns
    -------
    EntropySelection
        ``active == []`` for an empty spectrum or a single-reference
        one (largest entropy below the 0.14 line): no active space is
        justified, which is a result, not an error.
    """
    s1 = _validate_spectrum(s1)
    if s1.size == 0:
        return EntropySelection(
            [], 'plateau', 0, ['empty entropy spectrum; nothing to select'])
    largest = float(np.max(s1))
    if is_single_reference(s1, threshold=SINGLE_REFERENCE_THRESHOLD):
        return EntropySelection(
            [], 'plateau', 0,
            [f'largest single-orbital entropy {largest:.3e} is below the '
             f'{SINGLE_REFERENCE_THRESHOLD} multireference criterion '
             f'(Stein & Reiher 2016); no active space is justified'])

    order = list(np.argsort(-s1, kind='stable'))
    heights, _ = detect_plateaus(s1, plateau_values=plateau_values,
                                 threshold_step=threshold_step)
    if heights:
        k = int(heights[0])
        active = sorted(int(i) for i in order[:k])
        selected = s1[order[:k]]
        evidence = [
            f'plateau height {k} (the widest run of {threshold_step:g}-wide '
            f'threshold steps over which the orbital count is constant; '
            f'autoCAS protocol)',
            f'selection: {k} orbitals, entropy range '
            f'{float(selected.min()):.4f} to {float(selected.max()):.4f}; '
            f'first orbital outside the plateau at '
            f'{float(s1[order[k]]):.4f}' if k < s1.size else
            f'selection: all {k} orbitals',
        ]
        return EntropySelection(active, 'plateau', k, evidence)

    cutoff = WEAK_CORRELATION_THRESHOLD * largest
    active = sorted(int(i) for i in np.where(s1 >= cutoff)[0])
    evidence = [
        f'no plateau at {plateau_values} x {threshold_step:g}; falling '
        f'back to the weak-correlation line '
        f'({WEAK_CORRELATION_THRESHOLD:.0%} of the largest entropy, '
        f'Stein & Reiher 2016): cutoff {cutoff:.4f}',
        f'selection: {len(active)} orbitals at or above the line',
    ]
    return EntropySelection(active, 'plateau', 0, evidence)


def _validate_pair_matrix(I, n):
    I = np.asarray(I, dtype=float)
    if I.shape != (n, n):
        raise ValueError(f'the mutual-information matrix has shape '
                         f'{I.shape} for {n} orbitals')
    if not np.isfinite(I).all():
        raise ValueError('the mutual-information matrix contains '
                         'non-finite values; fix the extraction, not the '
                         'selection')
    if not np.allclose(I, I.T, atol=1e-8):
        raise ValueError('the mutual-information matrix is not symmetric; '
                         'it is not a pair measure then')
    return I


def mi_selection(s1, I, mi_fraction=MI_FRACTION_DEFAULT, seed=None):
    """Grow a set by the strongest mutual information, from the top-entropy seed.

    The growth rule: at each step add the outside orbital with the
    largest mutual information to any member of the set; stop when that
    best candidate does not strictly exceed ``mi_fraction`` times the
    largest pair value in the matrix. The seed is the largest-entropy
    orbital unless given (the open-shell top orbital is the physical
    start for f-block systems).

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies, one per orbital.
    I : array_like
        Mutual-information matrix (symmetric, zeros on the diagonal).
    mi_fraction : float
        Stop threshold as a fraction of the largest pair value; a
        package default (no literature value exists -- see the module
        docstring) reported in the evidence.
    seed : int, optional
        Starting orbital. Defaults to the largest entropy (ties: the
        lowest index).

    Returns
    -------
    EntropySelection
        ``active == []`` when the spectrum refuses (empty,
        single-reference) or the seed is out of range (a caller error
        reported, not crashed).
    """
    s1 = _validate_spectrum(s1)
    if s1.size == 0:
        return EntropySelection(
            [], 'mutual_information', 0,
            ['empty entropy spectrum; nothing to select'])
    I = _validate_pair_matrix(I, s1.size)
    if is_single_reference(s1, threshold=SINGLE_REFERENCE_THRESHOLD):
        return EntropySelection(
            [], 'mutual_information', 0,
            [f'largest single-orbital entropy {float(np.max(s1)):.3e} is '
             f'below the {SINGLE_REFERENCE_THRESHOLD} multireference '
             f'criterion; no active space is justified'])
    if not 0.0 <= mi_fraction < 1.0:
        raise ValueError(f'mi_fraction {mi_fraction} outside [0, 1)')

    if seed is None:
        start = min(range(s1.size),
                    key=lambda i: (-round(float(s1[i]) / _TIE_TOL), i))
    else:
        start = int(seed)
        if not 0 <= start < s1.size:
            return EntropySelection(
                [], 'mutual_information', 0,
                [f'seed {seed} outside the {s1.size} orbitals; refusing'])

    off_diagonal = I[~np.eye(s1.size, dtype=bool)]
    largest_pair = float(off_diagonal.max()) if off_diagonal.size else 0.0
    threshold = mi_fraction * largest_pair

    active = [start]
    steps = []
    while True:
        best_i, best_v = None, -np.inf
        for i in range(s1.size):
            if i in active:
                continue
            v = max(float(I[i, j]) for j in active)
            if v > best_v:
                best_i, best_v = i, v
        if best_i is None or best_v <= threshold:
            break
        active.append(best_i)
        steps.append((best_i, best_v))
    active.sort()

    evidence = [
        f'seed {start} (largest single-orbital entropy '
        f'{float(s1[start]):.4f})' if seed is None else f'seed {start} '
        f'(given; entropy {float(s1[start]):.4f})',
        f'stop threshold {threshold:.4f} = {mi_fraction:.0%} of the '
        f'largest pair value {largest_pair:.4f} (this package\'s '
        f'fraction; autoCAS itself selects by the s1 plateau alone)',
    ]
    for added, v in steps:
        evidence.append(f'added {added} (max mutual information to the set '
                        f'{v:.4f})')
    if not steps:
        evidence.append('no orbital outside the seed exceeded the '
                        'threshold: the set stays a single orbital')
    evidence.append(f'selection: {len(active)} orbitals {active}')
    return EntropySelection(active, 'mutual_information', 0, evidence)


@dataclass
class MiClosureReport:
    """Which candidate orbitals are uncoupled, and which outside ones entangle."""

    threshold: float = 0.0
    leaks: list = field(default_factory=list)
    isolated: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


def mi_closure(s1, I, candidate, mi_fraction=MI_FRACTION_DEFAULT):
    """Check a candidate against the pair structure of the wavefunction.

    Two directions are reported, neither as a verdict:

    *leaks* -- outside orbitals whose largest mutual information to the
    set exceeds the threshold. A leak is a candidate boundary that the
    pair structure does not support (the strongest case: an orbital
    entangled with the set as strongly as its own members are).

    *isolated* -- members whose largest mutual information to the rest
    of the set does not reach the threshold. An isolated member
    (a single unpaired orbital, for instance) can be legitimate; a
    block of them suggests the candidate joined weakly correlated
    orbitals, which is what the CANDIDATE_TOO_LARGE verdict of
    :func:`diffmrc.policy.verification.verify_candidate` measures from
    the spectrum side.

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies (used for size and finiteness only).
    I : array_like
        Mutual-information matrix.
    candidate : iterable of int
        Orbital indices to check, in any order, duplicates allowed.
    mi_fraction : float
        Same fraction rule as :func:`mi_selection`.

    Returns
    -------
    MiClosureReport
        Empty leak and isolated lists mean the candidate is closed at
        this threshold. A single-member candidate reports no isolated
        list (the partner question is meaningless there); an empty
        candidate reports everything as a leak by construction and the
        evidence says so.
    """
    s1 = _validate_spectrum(s1)
    I = _validate_pair_matrix(I, s1.size)
    members = sorted({int(i) for i in candidate})
    out_of_range = [i for i in members if not 0 <= i < s1.size]
    if out_of_range:
        raise ValueError(f'candidate orbitals {out_of_range} outside the '
                         f'{s1.size}-orbital space')
    if not 0.0 <= mi_fraction < 1.0:
        raise ValueError(f'mi_fraction {mi_fraction} outside [0, 1)')

    off_diagonal = I[~np.eye(s1.size, dtype=bool)]
    largest_pair = float(off_diagonal.max()) if off_diagonal.size else 0.0
    threshold = mi_fraction * largest_pair

    outside = [i for i in range(s1.size) if i not in members]
    leaks = [i for i in outside
             if members and max(float(I[i, j]) for j in members) > threshold]
    isolated = []
    if len(members) >= 2:
        isolated = [
            i for i in members
            if max(float(I[i, j]) for j in members if j != i) < threshold
        ]

    evidence = [
        f'threshold {threshold:.4f} = {mi_fraction:.0%} of the largest '
        f'pair value {largest_pair:.4f}; the two directions can be '
        f'legitimate (a leak can sit on a plateau boundary, an isolated '
        f'member can be an unpaired orbital)'
    ]
    if not members:
        evidence.append('empty candidate: no boundary exists, so no leaks '
                        'are reported; the spectrum selectors in this '
                        'module build candidates instead')
    elif leaks:
        strongest = max(leaks,
                        key=lambda i: max(float(I[i, j]) for j in members))
        evidence.append(
            f'{len(leaks)} leak(s): {leaks}; strongest is {strongest} with '
            f'mutual information '
            f'{max(float(I[strongest, j]) for j in members):.4f} to the set')
    else:
        evidence.append('no leaks: no outside orbital exceeds the '
                        'threshold against the set')
    if len(members) >= 2:
        if isolated:
            evidence.append(f'{len(isolated)} isolated member(s): '
                            f'{isolated} -- their largest mutual '
                            f'information to the rest of the set is below '
                            f'the threshold')
        else:
            evidence.append('no isolated members: every member couples to '
                            'the set above the threshold')
    return MiClosureReport(threshold, leaks, isolated, evidence)
