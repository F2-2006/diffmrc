"""Plateau detection on the normalized single-orbital entropy spectrum.

Given single-orbital entropies, normalize by the largest and sweep a
threshold tau from 0 to 1. The count

    N(tau) = #{ i : s1_i / max(s1) > tau }

is a descending staircase: it drops by one each time tau crosses an
orbital's normalized entropy. Where the spectrum has a gap between
strongly and weakly correlated orbitals, the staircase stays flat for a
long stretch. The height of that plateau is the number of orbitals that
belong in the active space.

The protocol follows the one implemented in the autoCAS program; see
C. J. Stein, M. Reiher, J. Comput. Chem. 40, 2216 (2019).

The threshold step sets the resolution of the sweep and the required
plateau length sets how flat a stretch must be to count. Both are exposed
so that a caller can loosen the criterion for spectra that have no sharp
gap. A smooth spectrum with no gap yields no plateau, which is the
intended outcome: the caller should then fall back on chemical rules
rather than invent an active space.
"""
import numpy as np

from .diagnostics import SINGLE_REFERENCE_THRESHOLD

__all__ = ['detect_plateaus', 'count_above', 'DEFAULT_PLATEAU_VALUES',
           'DEFAULT_THRESHOLD_STEP', 'DEFAULT_MIN_ENTROPY']

DEFAULT_PLATEAU_VALUES = 10
DEFAULT_THRESHOLD_STEP = 0.01
DEFAULT_MIN_ENTROPY = SINGLE_REFERENCE_THRESHOLD


def count_above(s1, thresholds):
    """Count how many entries exceed each threshold.

    Parameters
    ----------
    s1 : array_like
        Values to test, typically entropies already normalized.
    thresholds : array_like
        Thresholds to test, on the same scale as `s1`.

    Returns
    -------
    counts : numpy.ndarray
        Integer array, same length as `thresholds`.
    """
    s1 = np.asarray(s1, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    return np.array([int(np.sum(s1 > t)) for t in thresholds], dtype=int)


def detect_plateaus(s1, plateau_values=DEFAULT_PLATEAU_VALUES,
                    threshold_step=DEFAULT_THRESHOLD_STEP,
                    min_entropy=DEFAULT_MIN_ENTROPY):
    """Find plateaus in the normalized entropy spectrum.

    The spectrum is normalized by its largest value before the sweep, so a
    spectrum that is pure numerical noise acquires a relative structure
    that is not chemical. A system whose orbitals are all essentially
    unentangled would otherwise be reported as having a well-defined
    plateau, and an active space would be selected from noise. The
    ``min_entropy`` guard prevents that: the protocol only applies once at
    least one orbital is entangled enough to make the system genuinely
    multireference.

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies.
    plateau_values : int
        Number of consecutive constant steps required to call a stretch a
        plateau. Must be at least 1.
    threshold_step : float
        Step size of the threshold sweep, in (0, 1].
    min_entropy : float
        The sweep is skipped and no plateau is reported unless the largest
        entropy reaches this value. Defaults to the multireference
        criterion of the selection protocol: a wavefunction is treated as
        multiconfigurational only if some single-orbital entropy exceeds
        0.14. Pass 0.0 to run the sweep on any spectrum, however small.

    Returns
    -------
    heights : list[int]
        Distinct plateau heights, in descending order. Empty when the
        spectrum has no plateau or falls below `min_entropy`.
    normalized : numpy.ndarray
        The entropies divided by their maximum, sorted in descending order.
        All zeros when the maximum is zero. Returned even when the guard
        suppresses the plateau, so a caller can inspect the spectrum.

    Raises
    ------
    ValueError
        If `plateau_values` is below one or `threshold_step` is not in
        (0, 1].
    """
    if plateau_values < 1:
        raise ValueError(f'plateau_values must be >= 1, got {plateau_values}')
    if not 0.0 < threshold_step <= 1.0:
        raise ValueError(
            f'threshold_step must lie in (0, 1], got {threshold_step}'
        )

    s1 = np.asarray(s1, dtype=float)
    if s1.size == 0:
        return [], np.array([])
    s_max = float(np.max(s1))
    if s_max <= 0.0:
        return [], np.zeros_like(s1)

    normalized = np.sort(s1 / s_max)[::-1]
    if s_max < min_entropy:
        # Every orbital is below the multireference criterion. Normalizing
        # would manufacture structure that the wavefunction does not have.
        return [], normalized

    thresholds = np.arange(0.0, 1.0, threshold_step)
    counts = count_above(normalized, thresholds)

    heights = []
    current = counts[0]
    run = 1
    for n in counts:
        if n == current:
            run += 1
        else:
            run = 1
            current = n
        if run >= plateau_values:
            heights.append(int(n))

    return sorted(set(heights), reverse=True), normalized
