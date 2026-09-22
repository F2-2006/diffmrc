"""Element and shell composition, and ranking candidates by it.

The weighting itself lives in the analysis layer, where it is tested on
hand-built labels. This module adds the policy decision on top: given a
set of candidate orbitals and their weights, which ones to keep -- and,
crucially, when the weights are too close together to justify keeping any
of them.

That last part is the point. A ranking over near-equal weights is
arbitrary, and an arbitrary selection silently produces a plausible but
wrong active space. Every ranking here is therefore allowed to fail.

The label matching is delegated rather than reimplemented. A previous
implementation in this project took a fixed character position out of the
shell string, which matched nothing for fluorine's ``'2px'`` and matched
only the two-digit contractions of a lanthanide's f shell. The failure
was silent: an all-zero weight vector sorted by ``argsort`` returns the
index order, so the selection landed on the innermost core orbitals.
"""
from dataclasses import dataclass, field

import numpy as np

from ..analysis.composition import ao_mask

__all__ = ['WeightResult', 'RankResult', 'element_shell_weights',
           'rank_by_weight']

DEFAULT_MIN_RATIO = 2.0


@dataclass
class WeightResult:
    """Per-orbital weights on a shell, and how many functions were matched."""

    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    n_ao: int = 0


@dataclass
class RankResult:
    """A split of candidates into kept and dropped, with its margin."""

    keep: list = field(default_factory=list)
    drop: list = field(default_factory=list)
    discrimination: float = 0.0


def element_shell_weights(mol, C, element, angular_momentum):
    """Weighted population of each MO on one element's shell.

    The weighting uses the overlap matrix rather than raw coefficient
    squares, so the result is comparable between shells of different
    angular momentum.

    Parameters
    ----------
    mol : pyscf-like
        Provides ``ao_labels()`` and ``intor('int1e_ovlp')``.
    C : array_like
        Molecular orbital coefficients, shape ``(nao, nmo)``.
    element : str
        Element symbol, matched exactly.
    angular_momentum : str
        One of ``'s'``, ``'p'``, ``'d'``, ``'f'``.

    Returns
    -------
    WeightResult
        ``weights`` has length ``nmo``. ``n_ao`` is the number of atomic
        orbitals matched and is zero when the element or shell is absent;
        a caller must check it before trusting the weights.
    """
    C = np.asarray(C, dtype=float)
    mask = ao_mask(mol.ao_labels(), element, angular_momentum)
    n_ao = int(mask.sum())
    if n_ao == 0:
        return WeightResult(weights=np.zeros(C.shape[1]), n_ao=0)
    S = np.asarray(mol.intor('int1e_ovlp'), dtype=float)
    weighted = np.abs(S[mask, :] @ C) * np.abs(C[mask, :])
    return WeightResult(weights=weighted.sum(axis=0), n_ao=n_ao)


def rank_by_weight(candidates, weights, n_take, min_ratio=DEFAULT_MIN_RATIO):
    """Rank candidate orbitals by weight and split off the top `n_take`.

    The ranking is only as good as the set it is given. A weight says
    how much of a shell an orbital carries; it does not say whether the
    orbital is occupied, and an antibonding orbital can carry more of a
    ligand's character than the bonding orbital below it. Ranking the
    whole orbital space by a ligand weight and taking the top three once
    put three *empty* orbitals into an active space while dropping every
    occupied ligand orbital. Partition by occupation first, then rank
    within the block you actually want.

    Parameters
    ----------
    candidates : sequence of int
        Orbital indices to rank.
    weights : array_like
        Per-orbital weights, indexed the same way as the full orbital set.
    n_take : int
        How many to keep.
    min_ratio : float
        The smallest acceptable ratio between the weakest kept and the
        strongest dropped candidate. Below it the ranking carries no
        information and the function refuses.

    Returns
    -------
    RankResult or None
        None when the weights cannot justify a split: fewer candidates
        than requested, every weight zero, or a discrimination below
        `min_ratio`. A caller must treat None as a refusal, never as an
        empty selection.
    """
    candidates = [int(i) for i in candidates]
    weights = np.asarray(weights, dtype=float)
    if n_take < 0 or len(candidates) < n_take:
        return None
    if n_take == 0:
        return RankResult(keep=[], drop=sorted(candidates),
                          discrimination=float('inf'))

    w = np.array([weights[i] for i in candidates], dtype=float)
    if not np.any(w > 0.0):
        return None

    order = sorted(range(len(candidates)), key=lambda k: -w[k])
    keep = [candidates[k] for k in order[:n_take]]
    drop = [candidates[k] for k in order[n_take:]]
    if not drop:
        return RankResult(keep=sorted(keep), drop=[],
                          discrimination=float('inf'))

    w_keep_min = min(w[k] for k in order[:n_take])
    w_drop_max = max(w[k] for k in order[n_take:])
    if w_drop_max <= 0.0:
        # Everything dropped is exactly zero, so the split is absolute.
        return RankResult(keep=sorted(keep), drop=sorted(drop),
                          discrimination=float('inf'))
    discrimination = float(w_keep_min / w_drop_max)
    if discrimination < min_ratio:
        return None
    return RankResult(keep=sorted(keep), drop=sorted(drop),
                      discrimination=discrimination)
