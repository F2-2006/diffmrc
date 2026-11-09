"""Locating a crossing between two potential-energy surfaces.

WHY THIS EXISTS

A crossing between two electronic states is a zero of the gap

    Delta(x) = E_0(x) - E_1(x)

along a coordinate x (a bond length, a bending angle, a seam
parameter).  The number is easy to produce and easy to produce
wrongly.  A bond-length scan of MnO and CrO in this project found gaps
that changed sign only where the active orbitals had jumped to another
solution (projection overlap 0.665 against 0.995 elsewhere); treating
that sign change as a crossing would report a seam that does not exist.
The same scan found a second, smoother candidate at 2.90 A that was
also a branch change.  A crossing locator that only looks at the sign
of the gap cannot tell those cases apart.

The criteria enforced here are the three that separated them:

  * the gap has opposite signs on adjacent points (a bracket exists);
  * the gap is monotone on both sides of the bracket, over a stated
    window of intervals;
  * no point in the bracket is a branch jump, measured by the overlap
    of the active orbital space with the previous geometry.

A bracket that fails any of the three is reported with the reason and
the failing points, not silently converted into a number.  The
crossing coordinate itself is always reported when a bracket exists,
because it is a property of the numbers; ``status`` says whether it is
licenced as a physical crossing.

INPUT AND OUTPUT

The input is a sequence of :class:`ScanPoint`: one coordinate, the two
state energies ``(E_0, E_1)`` at that coordinate, optionally the two
gradients ``(dE_0/dx, dE_1/dx)`` and the active-space overlap with the
previous point.  The gap is ``E_0 - E_1``, so the state whose relative
stabilization is being tracked goes first.

The output is a :class:`CrossingReport`: the bracket, the linearly
interpolated crossing coordinate, the gap and slope on both sides, the
gradient difference of the two states, an uncertainty from the local
curvature model next to the conservative bracket half-width, and the
criteria verdict.

UNCERTAINTY

Two numbers are reported, because they answer different questions.
``step_bound`` is the half-width of the bracket: the largest error a
linear interpolation can make when the only thing known is that the
zero lies between the two points.  ``uncertainty`` is the difference
between the linear root and a quadratic root fitted through the
neighbouring point: it estimates the interpolation error from the
curvature actually present in the scan.  When the quadratic model has no
root inside the bracket -- the usual situation when the curvature is
large relative to the step -- no model correction is claimed and the
caller is left with the bracket bound.

NUMERICS

Everything here is NumPy on plain floats.  No electronic-structure
library is imported, so the criteria can be tested on analytic surfaces
and on stored scans without a kernel installation.  This module is
additive: it consumes numbers produced elsewhere and does not touch any
existing kernel path.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ['CrossingStatus', 'ScanPoint', 'NearestGap', 'CrossingReport',
           'TwoStartCertificate', 'locate_crossing', 'two_start_certificate',
           'orbital_space_overlap', 'matched_orbital_overlaps']


class CrossingStatus(str, Enum):
    """What the data licence.

    ``FOUND`` is the only value that means a physical crossing was
    located.  The other values are distinct failure modes, because they
    call for different follow-up: a branch jump needs a finer scan or a
    state-tracking protocol, a non-monotone bracket needs a decision
    about which feature is the crossing, and ``NO_CROSSING`` inside the
    scanned range may simply mean the range is too short.
    """

    FOUND = 'found'
    NO_CROSSING = 'no_crossing'
    BRANCH_JUMP = 'branch_jump'
    NON_MONOTONIC = 'non_monotonic'
    INSUFFICIENT_DATA = 'insufficient_data'


@dataclass(frozen=True)
class ScanPoint:
    """One geometry of a two-state scan.

    ``energies`` are ordered ``(E_0, E_1)``; the gap is ``E_0 - E_1``.
    ``gradients`` are the derivatives of the same two energies with
    respect to ``coordinate`` (not the Cartesian nuclear gradient: the
    caller projects it onto the scan direction first).  ``overlap_previous``
    is the active-orbital-space overlap with the previous point of the
    scan, in [0, 1]; ``None`` means it was not measured, which is
    different from a measured zero and is treated as unknown.
    """

    coordinate: float
    energies: tuple
    gradients: tuple = None
    overlap_previous: float = None


@dataclass(frozen=True)
class NearestGap:
    """The closest approach to a crossing when no bracket exists.

    ``closure_distance`` is ``|gap / slope|`` when the local slope points
    toward zero, i.e. the distance the coordinate would have to travel
    under the local linear model to close the gap.  It is a scale
    estimate, not an extrapolated crossing: the model is not expected to
    hold over that distance.
    """

    coordinate: float
    gap: float
    slope: float = None
    closure_distance: float = None


@dataclass(frozen=True)
class CrossingReport:
    """The crossing and the evidence for it.

    ``crossing`` is the linear interpolation of the bracket; it is
    reported even when ``status`` is ``BRANCH_JUMP`` or
    ``NON_MONOTONIC`` so that the caller can see what the raw numbers
    would have given.  ``uncertainty`` is the curvature-model correction
    and is ``None`` when the model does not produce a root inside the
    bracket or when the bracket is not licenced.  ``step_bound`` is
    always the bracket half-width.

    ``all_brackets`` lists every sign change in the scan, in coordinate
    order, so a scan with more than one candidate does not hide the
    others behind the first one.
    """

    status: CrossingStatus
    bracket: tuple = None
    crossing: float = None
    gap_left: float = None
    gap_right: float = None
    slope_left: float = None
    slope_right: float = None
    gradient_difference: float = None
    uncertainty: float = None
    step_bound: float = None
    model_correction: float = None
    monotonic: bool = False
    monotonic_checked: bool = False
    branch_clean: bool = True
    branch_jump_indices: tuple = ()
    all_brackets: tuple = ()
    nearest: NearestGap = None
    messages: tuple = ()


@dataclass(frozen=True)
class TwoStartCertificate:
    """Whether two converged starts reached the same stationary point.

    The verdict is one of

    ``same_stationary_point``
        the energies agree within ``energy_tol``, both orbital gradients
        are below ``grad_tol`` and the active-space overlap is at least
        ``overlap_tol``;
    ``distinct_stationary_points``
        both gradients are converged but the solutions differ in energy
        or in orbital space (or both);
    ``unconverged``
        at least one orbital gradient is above ``grad_tol``, so the
        energy comparison says nothing about stationary points;
    ``inconsistent``
        the orbitals agree but the energies do not, which cannot happen
        for two evaluations of the same stationary point and points to a
        stale energy, a different active space, or a different
        Hamiltonian.
    """

    verdict: str
    energy_difference: float
    gradient_norms: tuple
    overlap: float
    reasons: tuple = ()


def _as_point_list(points):
    points = list(points)
    if len(points) < 2:
        raise ValueError('a crossing scan needs at least two points')
    x = np.array([float(p.coordinate) for p in points], dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError('scan coordinates must be finite')
    if not np.all(np.diff(x) > 0.0):
        raise ValueError('scan coordinates must be strictly increasing')
    e = np.array([[float(v) for v in p.energies] for p in points], dtype=float)
    if e.shape[1] != 2:
        raise ValueError('each ScanPoint needs exactly two energies')
    if not np.all(np.isfinite(e)):
        raise ValueError('scan energies must be finite')
    return x, e


def _point_continuous(points, index, threshold):
    """True when point ``index`` continues the previous point.

    A missing overlap is treated as unknown and does not block the
    bracket; the report records how many checks were actually made.
    """
    if index == 0:
        return True
    overlap = points[index].overlap_previous
    if overlap is None:
        return True
    return float(overlap) >= float(threshold)


def _quadratic_crossing(x, gap, i):
    """Root of a quadratic through the bracket and one outer point.

    Returns ``(root, correction)`` or ``(None, None)``.  The outer point
    is the left neighbour when it exists and the right neighbour
    otherwise.  The returned root must lie inside the bracket; otherwise
    the local quadratic does not describe the sign change and no
    correction is claimed.
    """
    if i > 0:
        idx = (i - 1, i, i + 1)
    elif i + 2 < len(x):
        idx = (i, i + 1, i + 2)
    else:
        return None, None
    xs = np.array([x[k] for k in idx], dtype=float)
    gs = np.array([gap[k] for k in idx], dtype=float)
    coeff = np.polyfit(xs, gs, 2)
    a, b, c = coeff
    scale = max(abs(b), abs(c), 1e-300)
    if abs(a) <= 1e-12 * scale:
        # The three points are collinear to numerical precision.  The
        # linear root is the whole answer; a quadratic fit would return
        # one huge spurious root next to it.
        return None, 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None, None
    sqrt_disc = np.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    lo, hi = x[i], x[i + 1]
    inside = [r for r in roots if lo <= r <= hi]
    if not inside:
        return None, None
    root = min(inside, key=lambda r: abs(r - 0.5 * (lo + hi)))
    linear = x[i] + (-gap[i]) * (x[i + 1] - x[i]) / (gap[i + 1] - gap[i])
    return root, root - linear


def _slope_at(x, gap, index, direction):
    """One-sided difference of the gap, or ``None`` at the ends."""
    other = index + direction
    if other < 0 or other >= len(x):
        return None
    return float((gap[other] - gap[index]) / (x[other] - x[index]))


def _gradients_available(points):
    """True when every point carries both state gradients."""
    return all(p.gradients is not None and len(p.gradients) == 2
               for p in points)


def locate_crossing(points, *, overlap_threshold=0.90, monotonic_window=2):
    """Locate the crossing of two surfaces sampled on a coordinate grid.

    Parameters
    ----------
    points : sequence of ScanPoint
        Strictly increasing ``coordinate``.  ``energies`` are
        ``(E_0, E_1)`` and the gap is ``E_0 - E_1``.
    overlap_threshold : float
        A point whose ``overlap_previous`` is below this value is a
        branch jump.  A bracket that touches one is not licenced, which
        is the rule the MnO/CrO scans used: 0.90 separated the measured
        branch changes (0.665, 0.670) from the continuous points
        (0.995 and above).
    monotonic_window : int
        Number of intervals on each side of the bracket over which the
        gap must be monotone.  The bracket itself counts as one interval
        on each side.

    Returns
    -------
    CrossingReport
    """
    x, e = _as_point_list(points)
    gap = e[:, 0] - e[:, 1]
    n = len(points)
    window = max(1, int(monotonic_window))

    branch_jumps = tuple(
        k for k in range(1, n)
        if points[k].overlap_previous is not None
        and float(points[k].overlap_previous) < float(overlap_threshold))
    overlap_checks = sum(1 for p in points[1:]
                         if p.overlap_previous is not None)

    # Exact zeros are crossings at a grid point.  Handle them before the
    # sign-change search, which requires a strict product < 0.
    exact = [k for k in range(n) if gap[k] == 0.0]

    brackets = []
    for k in range(n - 1):
        if gap[k] * gap[k + 1] < 0.0:
            brackets.append((k, k + 1))

    if not brackets and not exact:
        k = int(np.argmin(np.abs(gap)))
        slope = None
        if 0 < k < n - 1:
            slope = float((gap[k + 1] - gap[k - 1]) / (x[k + 1] - x[k - 1]))
        elif k == 0:
            slope = _slope_at(x, gap, k, +1)
        else:
            slope = _slope_at(x, gap, k, -1)
        closure = None
        if slope is not None and slope != 0.0 and gap[k] * slope < 0.0:
            closure = float(abs(gap[k] / slope))
        elif slope is not None and slope != 0.0:
            # At an endpoint the one-sided slope can point away from
            # zero while the unobserved side points toward it.  The
            # magnitude is still the local closure scale; report it and
            # say that the direction is extrapolated.
            closure = float(abs(gap[k] / slope))
        diffs = np.diff(gap)
        monotone = bool(len(diffs) and (np.all(diffs > 0.0)
                                        or np.all(diffs < 0.0)))
        messages = []
        if branch_jumps:
            messages.append('branch jumps at indices %s'
                            % (branch_jumps,))
        if closure is not None and gap[k] * slope >= 0.0:
            messages.append('the nearest point is a scan endpoint and the '
                            'closure distance is an extrapolation in the '
                            'unobserved direction')
        return CrossingReport(
            status=CrossingStatus.NO_CROSSING,
            monotonic=monotone,
            monotonic_checked=bool(len(diffs)),
            branch_clean=not branch_jumps,
            branch_jump_indices=branch_jumps,
            nearest=NearestGap(coordinate=float(x[k]), gap=float(gap[k]),
                               slope=slope, closure_distance=closure),
            messages=tuple(messages))

    if exact:
        k = exact[0]
        clean = _point_continuous(points, k, overlap_threshold)
        if k + 1 < n:
            clean = clean and _point_continuous(points, k + 1,
                                                overlap_threshold)
        status = CrossingStatus.FOUND if clean else CrossingStatus.BRANCH_JUMP
        return CrossingReport(
            status=status,
            bracket=(float(x[k]), float(x[k])),
            crossing=float(x[k]),
            gap_left=0.0, gap_right=0.0,
            uncertainty=0.0, step_bound=0.0, model_correction=0.0,
            monotonic=True, monotonic_checked=False,
            branch_clean=clean,
            branch_jump_indices=branch_jumps,
            all_brackets=((float(x[k]), float(x[k])),))

    # Evaluate every sign change; report the first licenced one.  A scan
    # with two candidates is a real possibility (a seam crossed twice),
    # so the others are listed rather than dropped.
    evaluated = []
    for i, j in brackets:
        clean = _point_continuous(points, i, overlap_threshold) and \
            _point_continuous(points, j, overlap_threshold)
        # The direction follows the sign change: a gap that runs from
        # negative to positive must increase on both sides, one that
        # runs from positive to negative must decrease.  Requiring an
        # increase unconditionally would refuse every crossing whose
        # states are ordered the other way round -- measured on the CH2
        # scan, where the gap E(triplet) - E(singlet) decreases through
        # the crossing at 81.7 degrees.
        direction = 1.0 if gap[i] < 0.0 else -1.0
        left_diffs = direction * np.diff(gap[max(0, i - window):i + 1])
        right_diffs = direction * np.diff(gap[j:min(n, j + window + 1)])
        checked = (len(left_diffs) > 0 or len(right_diffs) > 0)
        monotone = bool(np.all(left_diffs > 0.0)) and \
            bool(np.all(right_diffs > 0.0))
        xc = float(x[i] - gap[i] * (x[j] - x[i]) / (gap[j] - gap[i]))
        root, correction = _quadratic_crossing(x, gap, i)
        evaluated.append(dict(
            i=i, j=j, clean=clean, monotone=monotone, checked=checked,
            xc=xc, correction=correction))

    licenced = [r for r in evaluated if r['clean'] and r['monotone']]
    chosen = licenced[0] if licenced else evaluated[0]
    if licenced:
        status = CrossingStatus.FOUND
    elif not chosen['clean']:
        status = CrossingStatus.BRANCH_JUMP
    else:
        status = CrossingStatus.NON_MONOTONIC

    i, j = chosen['i'], chosen['j']
    xc = chosen['xc']
    g0, g1 = float(gap[i]), float(gap[j])
    if status is CrossingStatus.FOUND:
        uncertainty = (abs(chosen['correction'])
                       if chosen['correction'] is not None else None)
    else:
        uncertainty = None
    step_bound = 0.5 * float(x[j] - x[i])

    grad_diff = None
    if _gradients_available(points):
        lo, hi = x[i], x[j]
        t = 0.0 if hi == lo else (xc - lo) / (hi - lo)
        g_lo = np.asarray(points[i].gradients, dtype=float)
        g_hi = np.asarray(points[j].gradients, dtype=float)
        pair = (1.0 - t) * g_lo + t * g_hi
        grad_diff = float(pair[0] - pair[1])
    else:
        grad_diff = float((g1 - g0) / (x[j] - x[i]))

    messages = []
    if len(licenced) > 1:
        messages.append('multiple licenced brackets; the first is reported '
                        'and all are listed in all_brackets')
    if status is CrossingStatus.BRANCH_JUMP:
        messages.append('a bracket endpoint continues with overlap below '
                        '%.3g; the crossing is not licenced'
                        % float(overlap_threshold))
    if status is CrossingStatus.NON_MONOTONIC:
        messages.append('the gap is not monotone on both sides of the '
                        'bracket over a window of %d intervals' % window)
    if overlap_checks == 0 and n > 1:
        messages.append('no overlap values were supplied; branch continuity '
                        'was not checked')

    return CrossingReport(
        status=status,
        bracket=(float(x[i]), float(x[j])),
        crossing=xc,
        gap_left=g0,
        gap_right=g1,
        slope_left=_slope_at(x, gap, i, -1),
        slope_right=_slope_at(x, gap, j, +1),
        gradient_difference=grad_diff,
        uncertainty=uncertainty,
        step_bound=step_bound,
        model_correction=chosen['correction'],
        monotonic=bool(chosen['monotone']),
        monotonic_checked=bool(chosen['checked']),
        branch_clean=bool(chosen['clean']),
        branch_jump_indices=branch_jumps,
        all_brackets=tuple((float(x[a]), float(x[b])) for a, b in brackets),
        messages=tuple(messages))


def orbital_space_overlap(mo_a, mo_b, overlap_matrix, ncore, ncas):
    """Overlap of two active spaces, invariant to rotations inside them.

    ``M = A^T S B`` for the active columns; the returned value is
    ``sum(M * M) / ncas``.  An active-space rotation ``A -> A U`` leaves
    it unchanged, which is what makes it a branch diagnostic rather than
    an orbital-phase diagnostic.  One for identical spaces, zero for
    orthogonal spaces.
    """
    a = np.asarray(mo_a, dtype=float)[:, ncore:ncore + ncas]
    b = np.asarray(mo_b, dtype=float)[:, ncore:ncore + ncas]
    s = np.asarray(overlap_matrix, dtype=float)
    m = a.T @ s @ b
    return float(np.sum(m * m) / ncas)


def matched_orbital_overlaps(mo_a, mo_b, overlap_matrix, ncore, ncas):
    """Singular values of the active-space overlap matrix.

    These are the overlaps after the best orthogonal matching of the two
    active sets, one per orbital, sorted descending.  Unlike the summed
    overlap they show whether a single orbital changed; a space that
    differs in one column gives one small singular value and the rest
    near one.
    """
    a = np.asarray(mo_a, dtype=float)[:, ncore:ncore + ncas]
    b = np.asarray(mo_b, dtype=float)[:, ncore:ncore + ncas]
    s = np.asarray(overlap_matrix, dtype=float)
    return np.linalg.svd(a.T @ s @ b, compute_uv=False)


def two_start_certificate(energy_a, energy_b, grad_norm_a, grad_norm_b,
                          overlap, *, energy_tol=1e-6, grad_tol=1e-4,
                          overlap_tol=0.99):
    """Decide whether two converged starts found the same stationary point.

    The energy alone cannot decide: two different local minima can lie
    within a millihartree of each other, and one unconverged run can lie
    anywhere.  The certificate therefore requires all three pieces of
    evidence -- energy, orbital gradient and active-space overlap -- and
    reports which one failed when the verdict is not agreement.

    The defaults are tied to the convergence settings this project uses:
    a CASSCF run converged to 1e-8 Eh has an orbital-gradient target of
    about 1e-4 (pyscf takes the square root of the energy tolerance), so
    ``grad_tol=1e-4`` is the loosest value at which both runs can be
    called stationary.  A run that is to be used as evidence should be
    re-converged with a tighter ``conv_tol_grad``; the MnO two-start
    check in this project used 1e-6.
    """
    de = float(energy_a) - float(energy_b)
    ga, gb = float(grad_norm_a), float(grad_norm_b)
    ov = float(overlap)
    reasons = []
    if max(ga, gb) > grad_tol:
        reasons.append('orbital gradient %.3e or %.3e exceeds %.3e'
                       % (ga, gb, grad_tol))
        return TwoStartCertificate('unconverged', de, (ga, gb), ov,
                                   tuple(reasons))
    same_energy = abs(de) <= float(energy_tol)
    same_orbitals = ov >= float(overlap_tol)
    if same_energy and same_orbitals:
        reasons.append('energies agree to %.3e and overlap %.4f is above '
                       '%.3f' % (abs(de), ov, overlap_tol))
        return TwoStartCertificate('same_stationary_point', de, (ga, gb), ov,
                                   tuple(reasons))
    if same_energy and not same_orbitals:
        reasons.append('energies agree but overlap %.4f is below %.3f: '
                       'two distinct solutions at the same energy'
                       % (ov, overlap_tol))
        return TwoStartCertificate('distinct_stationary_points', de, (ga, gb),
                                   ov, tuple(reasons))
    if not same_energy and same_orbitals:
        reasons.append('orbitals agree (overlap %.4f) but energies differ '
                       'by %.3e Eh, which one stationary point cannot '
                       'produce' % (ov, abs(de)))
        return TwoStartCertificate('inconsistent', de, (ga, gb), ov,
                                   tuple(reasons))
    reasons.append('energies differ by %.3e Eh and overlap %.4f is below '
                   '%.3f' % (abs(de), ov, overlap_tol))
    return TwoStartCertificate('distinct_stationary_points', de, (ga, gb), ov,
                               tuple(reasons))
