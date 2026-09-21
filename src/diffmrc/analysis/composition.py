"""Orbital composition analysis by element and angular momentum.

Given molecular orbital coefficients and the atomic-orbital overlap matrix,
report how much of a given element and angular momentum each orbital
carries. The report cites these numbers when it claims that an active space
holds, for instance, the europium 4f shell.

Only the angular-momentum *letter* is matched here. Principal quantum
numbers are deliberately not interpreted: the ANO basis sets label their
f shells 6f, 7f and 8f with no 4f among them, so matching on the principal
quantum number would silently find nothing. The element is matched exactly
rather than as a substring, so that O does not select Os.
"""
import numpy as np

__all__ = ['ao_mask', 'overlap_weighted_population']


def _shell_letters(shell):
    """Strip the principal quantum number, leaving the angular-momentum part.

    ``'6f'`` becomes ``'f'``, ``'2px'`` becomes ``'px'``, ``'5d'`` becomes
    ``'d'``. Labels that carry no letter at all yield the empty string.
    """
    return ''.join(ch for ch in shell if not ch.isdigit()).lower()


def ao_mask(labels, element, angular_momentum):
    """Build a boolean mask selecting atomic orbitals of one element and shell.

    Parameters
    ----------
    labels : sequence of str
        PySCF-style AO labels such as ``"3 O 2pz"``. The element is taken
        as the second whitespace-separated field and the shell as the third.
    element : str
        Element symbol, matched exactly against the second field.
    angular_momentum : str
        One of ``"s"``, ``"p"``, ``"d"``, ``"f"``. Matched against the
        leading letter of the shell, so ``"p"`` selects ``2px`` but not
        ``1s``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``len(labels)``. Labels that do not split
        into at least three fields select nothing.
    """
    out = []
    for label in labels:
        parts = str(label).split()
        if len(parts) < 3:
            out.append(False)
            continue
        elem = parts[1]
        letters = _shell_letters(parts[2])
        out.append(elem == element and letters.startswith(angular_momentum))
    return np.array(out, dtype=bool)


def overlap_weighted_population(C, S, mask):
    """Weighted population of each orbital on a set of atomic orbitals.

    The weighting uses the overlap matrix rather than the raw coefficient
    squares, so that the result is comparable between atomic orbitals of
    different angular momentum.

    Parameters
    ----------
    C : array_like
        Molecular orbital coefficients, shape ``(nao, nmo)``.
    S : array_like
        Atomic-orbital overlap matrix, shape ``(nao, nao)``.
    mask : array_like
        Boolean mask of length ``nao`` selecting the atomic orbitals.
        Non-boolean input is treated as a selector for nonzero entries.

    Returns
    -------
    numpy.ndarray
        Weighted population of each molecular orbital, shape ``(nmo,)``.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent.
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    mask = np.asarray(mask)

    if C.ndim != 2:
        raise ValueError(f'C must be two-dimensional, got shape {C.shape}')
    nao, _ = C.shape
    if S.shape != (nao, nao):
        raise ValueError(
            f'S shape {S.shape} does not match C dimension {nao}'
        )
    if mask.shape != (nao,):
        raise ValueError(
            f'mask length {mask.shape} does not match {nao} atomic orbitals'
        )
    mask = mask.astype(bool)

    # Weight each coefficient by the overlap it accumulates with the masked
    # atomic orbitals; the absolute value keeps the result a population.
    weighted = np.abs(S[mask, :] @ C) * np.abs(C[mask, :])
    return weighted.sum(axis=0)
