"""Localized orbitals and the overlapping split of a large space.

Entanglement-based orbital selection assumes the orbitals are localized.
The reason is physical: static correlation is a local effect, so in a
localized basis a given orbital is strongly entangled with one or a few
others and the single-orbital entropy means something. In a canonical
(delocalized) basis the same correlation is spread over the whole space,
the entropy per orbital drops, and the spectrum stops separating strongly
from weakly correlated orbitals.

The second ingredient is how to handle a space too large for an exact
solve. Splitting it into overlapping subsets, solving each, and
recombining the entropies keeps every orbital's information. Truncating
the space instead discards orbitals outright: measured on a lanthanide
atom, a CAS(19,25) truncation collapsed the ground state to a single
determinant whose entropies are identically zero, leaving only numerical
noise to select from.

References
----------
C. J. Stein, M. Reiher, J. Comput. Chem. 40, 2216 (2019), Sec.
"Ground-State Calculations with Very Large Valence Spaces" and Fig. 4.
"""
import numpy as np

__all__ = ['localize_orbitals', 'overlapping_split']

LOCALIZATION_METHODS = ('pm', 'boys', 'ibo')


def localize_orbitals(mol, C, method='pm'):
    """Transform canonical molecular orbitals into a localized set.

    Parameters
    ----------
    mol : pyscf Mole
        Molecule the orbitals belong to.
    C : array_like
        Canonical orbital coefficients, shape ``(nao, nmo)``.
    method : str
        ``'pm'`` for Pipek-Mezey, the default: it keeps sigma and pi
        separated, which suits the correlation patterns this package
        targets. ``'boys'`` for Foster-Boys, ``'ibo'`` for Huzinaga.

    Returns
    -------
    numpy.ndarray
        Localized coefficients, same shape as `C`.

    Raises
    ------
    ValueError
        If `method` is not recognized.
    """
    if method not in LOCALIZATION_METHODS:
        raise ValueError(
            f'unknown localization method {method!r}; '
            f'expected one of {LOCALIZATION_METHODS}'
        )
    from pyscf import lo
    C = np.asarray(C, dtype=float)
    if method == 'pm':
        return np.asarray(lo.PipekMezey(mol, C).kernel(), dtype=float)
    if method == 'boys':
        return np.asarray(lo.Boys(mol, C).kernel(), dtype=float)
    return np.asarray(lo.ibo.ibo(mol, C), dtype=float)


def overlapping_split(space, l_max):
    """Split a space into overlapping subsets of at most `l_max` orbitals.

    The first half of the space is the occupied block and the second half
    the virtual block; each is divided into equal subgroups and every
    occupied subgroup is combined with every virtual one. For an
    18-orbital space with ``l_max = 6`` that is three subgroups of three
    on each side and nine subsets of six -- the construction in the
    reference figure, a CAS(18,18) into nine CAS(6,6).

    Overlap is the point: every orbital appears in `n_groups` subsets, so
    no orbital's entanglement is lost from the recombined spectrum.

    Parameters
    ----------
    space : sequence of int
        Orbital indices to split, ordered so that the first half is the
        occupied block and the second half the virtual block.
    l_max : int
        Maximum subset size. Must be even and divide the space length
        exactly, so that the subgroups come out even.

    Returns
    -------
    list[list[int]]
        Ascending index lists, ``(L / l_max) ** 2`` of them. When the
        space already fits within `l_max`, a single subset holding all
        of it.

    Raises
    ------
    ValueError
        If `l_max` is not positive and even, or the space length is not
        an exact multiple of `l_max`.
    """
    space = sorted(int(i) for i in space)
    if l_max < 1:
        raise ValueError(f'l_max must be positive, got {l_max}')
    if len(space) <= l_max:
        return [space] if space else []
    if l_max % 2 != 0:
        raise ValueError(
            f'l_max must be even to split into occupied and virtual '
            f'halves, got {l_max}'
        )
    if len(space) % l_max != 0:
        raise ValueError(
            f'space of {len(space)} orbitals is not an exact multiple of '
            f'l_max {l_max}'
        )

    n_groups = len(space) // l_max
    half = len(space) // 2
    group = half // n_groups
    occupied = [space[i * group:(i + 1) * group] for i in range(n_groups)]
    virtual = [space[half + i * group:half + (i + 1) * group]
               for i in range(n_groups)]
    return [sorted(occ + vir) for occ in occupied for vir in virtual]
