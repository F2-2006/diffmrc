"""Diagnostics that decide whether a multireference treatment is warranted.

The Zs(1) diagnostic averages the single-orbital entropies over their
theoretical maximum,

    Zs(1) = (1 / (n * ln 4)) * sum_i s1_i

and approaches one when every orbital is maximally entangled. Values above
0.2 indicate pronounced multireference character; between 0.1 and 0.2 a
single-reference method may still be adequate and is usually cheaper.

Two thresholds accompany it. A system whose largest single-orbital entropy
falls below the single-reference threshold carries too little static
correlation to justify an active space. Orbitals whose entropy falls below
the weak-correlation threshold, relative to the largest, are too weakly
correlated to include.

References
----------
C. J. Stein, M. Reiher, Mol. Phys. 115, 2110 (2017).
"""
import numpy as np

__all__ = ['zs1', 'is_single_reference', 'SINGLE_REFERENCE_THRESHOLD',
           'WEAK_CORRELATION_THRESHOLD', 'MULTIREFERENCE_ZS1_THRESHOLD']

SINGLE_REFERENCE_THRESHOLD = 0.14
WEAK_CORRELATION_THRESHOLD = 0.02
MULTIREFERENCE_ZS1_THRESHOLD = 0.2

_LN4 = float(np.log(4.0))


def zs1(s1):
    """Evaluate the Zs(1) multireference diagnostic.

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies.

    Returns
    -------
    float
        The diagnostic, in [0, 1]. Values above 0.2 indicate pronounced
        multireference character.

    Raises
    ------
    ValueError
        If `s1` is empty.
    """
    s1 = np.asarray(s1, dtype=float)
    if s1.size == 0:
        raise ValueError('cannot evaluate Zs(1) on an empty spectrum')
    return float(np.sum(s1) / (s1.size * _LN4))


def is_single_reference(s1, threshold=SINGLE_REFERENCE_THRESHOLD):
    """Decide whether a single-reference treatment is adequate.

    An empty spectrum counts as single reference: with no active orbitals
    there is nothing that would justify the cost of a multireference
    calculation.

    Parameters
    ----------
    s1 : array_like
        Single-orbital entropies.
    threshold : float
        A system whose largest entropy falls below this value is treated
        as single reference.

    Returns
    -------
    bool
        True if the system is weakly correlated.
    """
    s1 = np.asarray(s1, dtype=float)
    if s1.size == 0:
        return True
    return bool(np.max(s1) < threshold)
