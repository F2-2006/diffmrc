"""Recombining subset entropies and judging a rule-based candidate.

The verdict compares what the chemical rules proposed against what the
wavefunction's entropy spectrum shows. Agreement licenses an automatic
answer. Disagreement is reported with the direction of the mismatch, so
a caller can widen the space or hand the decision back to the user.

The most important case is the one where nothing can be decided. A
spectrum whose largest entropy lies below the multireference criterion
means the system is not strongly correlated at all, and no active space
should be selected automatically. Reporting a plateau there would be
reporting structure in numerical noise: the spectrum is normalized by
its own maximum before the plateau sweep, so a set of round-off values
acquires a staircase. That failure was measured on a lanthanide atom,
where a truncated reference space left every entropy at about 1e-6 yet
the sweep reported a plateau of height ten.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..analysis.diagnostics import (
    SINGLE_REFERENCE_THRESHOLD,
    WEAK_CORRELATION_THRESHOLD,
    is_single_reference,
    zs1,
)
from ..analysis.plateau import detect_plateaus

__all__ = ['Verdict', 'VerificationResult', 'recombine_entropies',
           'verify_candidate', 'OVERLAP_MIN_SIGMA', 'NEVPT2_SHIFT_MAX_EV',
           'EV_TO_WAVENUMBER', 'OverlapReport', 'active_space_overlap',
           'ShiftReport', 'correlation_shift']


class Verdict(Enum):
    """What comparing the candidate against the spectrum concluded."""

    AGREES = 'agrees'
    CANDIDATE_TOO_SMALL = 'candidate_too_small'
    CANDIDATE_TOO_LARGE = 'candidate_too_large'
    NO_PLATEAU = 'no_plateau'
    SINGLE_REFERENCE = 'single_reference'


@dataclass
class VerificationResult:
    verdict: Verdict
    plateau_heights: list = field(default_factory=list)
    normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    n_candidate: int = 0
    zs1: float = 0.0
    is_single_reference: bool = True
    weakly_correlated: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


def recombine_entropies(per_subset, n_orbitals):
    """Merge per-subset entropies into one spectrum over all orbitals.

    Each subset gives an entropy for the orbitals it contains, and an
    orbital may appear in several subsets. It gets the largest of its
    estimates: restricting an orbital's environment to a subset can only
    lower the measured entanglement, so the widest view available is the
    best estimate.

    Parameters
    ----------
    per_subset : sequence
        Either bare mappings from orbital index to entropy, or
        ``(mapping, _)`` pairs, which is what a solver returning an extra
        value per subset produces.
    n_orbitals : int
        Size of the full spectrum to return.

    Returns
    -------
    numpy.ndarray
        Entropy per orbital, length `n_orbitals`. Orbital indices that
        appeared in no subset are zero.
    """
    best = np.zeros(n_orbitals, dtype=float)
    seen = np.zeros(n_orbitals, dtype=bool)
    for item in per_subset:
        mapping = item[0] if isinstance(item, tuple) else item
        for index, value in dict(mapping).items():
            index = int(index)
            if 0 <= index < n_orbitals and (not seen[index]
                                            or value > best[index]):
                best[index] = float(value)
                seen[index] = True
    return best


def verify_candidate(s1, candidate, plateau_values=10, threshold_step=0.01):
    """Compare an entropy spectrum against a rule-based candidate.

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies over the full space.
    candidate : sequence of int
        Orbital indices the rules proposed.
    plateau_values, threshold_step : int, float
        Plateau-detection settings, forwarded to the analysis layer.

    Returns
    -------
    VerificationResult
        The verdict, the plateau heights found, and the evidence for it.
    """
    s1 = np.asarray(s1, dtype=float)
    candidate = sorted({int(i) for i in candidate})
    evidence = []

    if is_single_reference(s1, threshold=SINGLE_REFERENCE_THRESHOLD):
        largest = float(np.max(s1)) if s1.size else 0.0
        evidence.append(
            f'largest single-orbital entropy {largest:.3e} is below the '
            f'multireference criterion {SINGLE_REFERENCE_THRESHOLD}; no '
            f'active space is justified'
        )
        return VerificationResult(
            verdict=Verdict.SINGLE_REFERENCE, plateau_heights=[],
            n_candidate=len(candidate),
            zs1=zs1(s1) if s1.size else 0.0,
            is_single_reference=True, evidence=evidence,
        )

    heights, normalized = detect_plateaus(
        s1, plateau_values=plateau_values, threshold_step=threshold_step)
    result = VerificationResult(
        verdict=Verdict.NO_PLATEAU, plateau_heights=heights,
        normalized=normalized, n_candidate=len(candidate),
        zs1=zs1(s1), is_single_reference=False,
    )

    # The largest plateau is the correlated-orbital count. The count of
    # *all* orbitals also forms a step at low thresholds, but it lasts
    # only until the threshold crosses the weakest entropies. For a
    # genuinely correlated spectrum the weak tail sits near zero, so that
    # step is a sweep point or two wide and never reaches the required
    # plateau length; when it does reach it, the entropies are all
    # comparable and the spectrum is separating nothing, which is the
    # noise case the floor above has already rejected.
    if not heights:
        cutoff = WEAK_CORRELATION_THRESHOLD * float(np.max(s1))
        result.weakly_correlated = [int(i) for i in np.where(s1 < cutoff)[0]]
        result.evidence = evidence + [
            f'no plateau above the {SINGLE_REFERENCE_THRESHOLD} floor; '
            f'excluding the {len(result.weakly_correlated)} orbitals below '
            f'{WEAK_CORRELATION_THRESHOLD:.0%} of the maximum'
        ]
        return result

    height = heights[0]
    if height == len(candidate):
        result.verdict = Verdict.AGREES
        result.evidence = evidence + [
            f'plateau height {height} matches the {len(candidate)} '
            f'proposed orbitals'
        ]
    elif height > len(candidate):
        result.verdict = Verdict.CANDIDATE_TOO_SMALL
        result.evidence = evidence + [
            f'plateau height {height} exceeds the {len(candidate)} '
            f'proposed orbitals; the candidate misses strongly correlated '
            f'orbitals'
        ]
    else:
        result.verdict = Verdict.CANDIDATE_TOO_LARGE
        result.evidence = evidence + [
            f'plateau height {height} is below the {len(candidate)} '
            f'proposed orbitals; the candidate includes weakly correlated '
            f'orbitals'
        ]
    return result


OVERLAP_MIN_SIGMA = 1.1e-6
"""Smallest active-space overlap singular value that still counts as safe.

Sayfutyarova and coworkers proposed the smallest singular value of the
overlap between the initial and the final active spaces as an
orbital-rotation diagnostic; King and Gagliardi measured it over 1120
calculations and report that this threshold separates a ~6% subset whose
mean error is higher by 0.08 eV -- significant, but only at the extreme
low end.  The check flags *vanished* orbitals, it does not rank quality.
"""

NEVPT2_SHIFT_MAX_EV = 1.1
"""Bao and Truhlar's suggested bound on the correlation shift, in eV.

King and Gagliardi retested it as a weak classifier: the flagged ~7% of
calculations carry a mean error higher by 0.34 eV.  The eV value is
evidential; the check itself is unit-agnostic and takes its threshold.
"""

# 1 eV expressed in wavenumbers, derived from CODATA SI constants:
# E = h c nu  =>  nu [cm^-1] = e [C] / (h [J s] * c [m/s]) / 100.
EV_TO_WAVENUMBER = (1.602176634e-19 / (6.62607015e-34 * 299792458.0)
                    / 100.0)


@dataclass
class OverlapReport:
    singular_values: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_min: float = 0.0
    vanished: bool = False
    evidence: list = field(default_factory=list)


def active_space_overlap(C_initial, C_final, S, threshold=OVERLAP_MIN_SIGMA):
    """Singular values of the initial-to-final active-space overlap.

    ``C_initial`` and ``C_final`` are the active-space coefficient blocks
    (nao x nact) before and after the orbital optimization, ``S`` the
    atomic-orbital overlap.  ``S_change = C_final^T S C_initial``; a
    singular value near zero means an orbital was rotated out completely
    -- the candidate described a space the converged solution does not
    occupy.  This is the one check in this module that looks at the
    space's own geometry rather than at the entropy spectrum.
    """
    C0 = np.asarray(C_initial, dtype=float)
    C1 = np.asarray(C_final, dtype=float)
    S = np.asarray(S, dtype=float)
    if C0.ndim != 2 or C1.ndim != 2 or C0.shape != C1.shape:
        raise ValueError(
            f'active blocks must share one shape, got {C0.shape} and '
            f'{C1.shape}')
    if S.shape != (C0.shape[0], C0.shape[0]):
        raise ValueError(
            f'S shape {S.shape} does not match {C0.shape[0]} atomic '
            f'orbitals')
    svals = np.linalg.svd(C1.T @ S @ C0, compute_uv=False)
    sigma_min = float(svals[-1])
    vanished = sigma_min < threshold
    if vanished:
        note = ('; an orbital was rotated out during the optimization'
                if not np.isclose(sigma_min, 0.0)
                else '; an orbital is exactly orthogonal to the initial '
                     'space')
    else:
        note = ''
    evidence = [
        f'active-space overlap sigma_min = {sigma_min:.3e} '
        f'({"below" if vanished else "above"} the {threshold:.1e} '
        f'threshold{note})'
    ]
    return OverlapReport(singular_values=svals, sigma_min=sigma_min,
                         vanished=vanished, evidence=evidence)


@dataclass
class ShiftReport:
    gap: float = 0.0
    reliable: bool = True
    threshold: float = 0.0
    evidence: list = field(default_factory=list)


def correlation_shift(delta_casscf, delta_nevpt2, *, threshold):
    """|dE_NEVPT2 - dE_CASSCF| for a differential property, and its flag.

    The property differences are at whatever level the caller reports --
    a bond energy, an excitation energy -- and the threshold is keyword
    required in the *same unit*: the function refuses to guess it, since
    the published 1.1 eV value silently means 8872 cm^-1 when the
    property is a wavenumber (``NEVPT2_SHIFT_MAX_EV`` and
    ``EV_TO_WAVENUMBER`` are provided for that conversion).  Exceeding
    the threshold is a warning, not a verdict: King and Gagliardi's
    measurement makes it a weak classifier, not a filter.
    """
    gap = abs(float(delta_nevpt2) - float(delta_casscf))
    reliable = gap <= threshold
    evidence = [
        f'correlation shift |dE_NEVPT2 - dE_CASSCF| = {gap:.4g} against '
        f'the threshold {threshold:.4g} '
        f'({"within" if reliable else "beyond"}; beyond the threshold the '
        f'flagged results carry a higher mean error but are not excluded)'
    ]
    return ShiftReport(gap=gap, reliable=reliable, threshold=threshold,
                       evidence=evidence)
