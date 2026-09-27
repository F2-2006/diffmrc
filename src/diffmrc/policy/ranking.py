"""Ranked-orbital active-space selection with approximate pair coefficients.

King and Gagliardi separate *how the orbitals are made* from *how they
are ranked*: every doubly occupied orbital and every window virtual gets
an importance number, and the active space is the top of that ranking,
trimmed until the configuration count fits a cap.  The cap is the
source's eq. (2),

    N_CSF = C(L, alpha) C(L, beta) - C(L, alpha+1) C(L, beta-1),

and the ranking is the approximate pair coefficient (APC) of eqs. (18),
(19): each occupied-virtual pair is treated as a two-configuration model
whose coefficient replaces the exchange integral by the exchange-matrix
diagonal, c = -x / (D + sqrt(x^2 + D^2)) with x = 0.5 K_aa and D the
orbital-energy difference; the one-orbital entropies follow from binary
entropies of the gathered |c|^2 (eqs. 12, 13).

Why here: this is the size-control rule the plateau method cannot give
(the plateau height does not fix a size) while the NEVPT2 chain has a
hard memory ceiling, so a ranking with an explicit configuration cap is
the decision that closes that loop.

Integral convention: ``eri_mo`` is the kernel's layout, pinned by two
independent consumers to the chemist (pq|rs) form -- the Condon-Slater
double-excitation element in ``core/casci.py`` equals (ik|jl) under it,
and the core-valence mean field in ``core/nevpt2/impl.py`` contracts it
as 2 (pq|cc) - (pc|cq).  Under that layout the exchange diagonal
``K_aa = sum_i (ai|ai)`` is ``eri_mo[a, i, a, i]``, which is what
``exchange_diagonal`` sums; the fBlockKit port pinned the same identity
against ORCA's K-matrix export.

Adopted from the fBlockKit implementation (its analysis/apc.py fixed
these conventions first); this port consumes plain numpy arrays and
carries the source's deficiencies as evidence: APC systematically
overestimates the occupied entropies (R^2 0.64 against DMRG) and
degrades where the HF determinant is a poor approximation.
"""
from dataclasses import dataclass, field
from math import comb

import numpy as np

__all__ = ['RankedOrbitalChoice', 'n_csf', 'exchange_diagonal',
           'ranked_orbital_selection']

#: Entropies agreeing to this tolerance count as equal and rank by index;
#: physically degenerate partners differ only in floating-point tails that
#: can depend on the BLAS the integrals were transformed under.
_TIE_TOL = 1e-12


def _comb0(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def n_csf(n_orb, n_ele):
    """Configuration-state count of the source's eq. (2)."""
    if n_orb < 1 or n_ele < 0 or n_ele > 2 * n_orb:
        raise ValueError(f'n_orb={n_orb}, n_ele={n_ele} outside the '
                         f'physical range')
    n_alpha = (n_ele + 1) // 2
    n_beta = n_ele - n_alpha
    return (_comb0(n_orb, n_alpha) * _comb0(n_orb, n_beta)
            - _comb0(n_orb, n_alpha + 1) * _comb0(n_orb, n_beta - 1))


def exchange_diagonal(eri_mo, occupied, virtual):
    """0.5 K_aa = sum_{i in occupied} (ai|ai) for each window virtual.

    ``eri_mo`` follows the kernel's four-index order (chemist (pq|rs),
    see the module docstring); the sum is the quantity the source pins
    as ``-diag(C K C^T)[a]`` against the fBlockKit/ORCA K-matrix export.
    """
    eri = np.asarray(eri_mo, dtype=float)
    occ = np.asarray(occupied, dtype=int)
    vir = np.asarray(virtual, dtype=int)
    n = eri.shape[0] if eri.ndim == 4 else -1
    if n < 1 or any(s != n for s in eri.shape):
        raise ValueError(f'eri_mo must be square in all four axes, got '
                         f'{eri.shape}')
    if occ.size == 0 or vir.size == 0:
        raise ValueError('occupied and virtual index sets must be '
                         'non-empty')
    return np.einsum('aiai->a', eri[np.ix_(vir, occ, vir, occ)])


def _binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    q = p[mask]
    out[mask] = -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))
    return out


@dataclass
class RankedOrbitalChoice:
    """The trimmed active set and the ranking it came from.

    A refusal carries ``active == []`` and the reason in ``evidence``.
    """

    active: list = field(default_factory=list)
    entropies: dict = field(default_factory=dict)
    n_csf: int = 0
    evidence: list = field(default_factory=list)


def ranked_orbital_selection(eps_occupied, eps_virtual, exchange_virtual, *,
                             occupied=None, virtual=None, singly_occupied=None,
                             n_csf_cap, min_occupied=1, min_virtual=2):
    """Rank occupied and window-virtual orbitals, trim to the CSF cap.

    Parameters
    ----------
    eps_occupied, eps_virtual : array_like
        Orbital energies (or Fock diagonals) of the doubly occupied set
        and of the virtual window, any unit as long as both share it.
    exchange_virtual : array_like
        0.5 K_aa = sum_i (ai|ai) per window virtual, same unit as eps.
    occupied, virtual : array_like of int, optional
        Index labels for the two blocks, returned in ``active`` and
        ``entropies``.  Defaults: 0..n_occ-1 and n_occ..n_occ+n_vir-1.
    singly_occupied : array_like of int, optional
        Index labels of singly occupied orbitals (open-shell
        references).  The source's rule applies: each is given the
        highest entropy, ln 2, and enters the space unconditionally --
        it is never a drop candidate -- while its electron is counted
        in the CSF cap, whose electron budget becomes 2 n_occ + n_sing.
    n_csf_cap : int
        Maximum configuration count; the loop drops the lowest-ranked
        orbital until the count fits.
    min_occupied, min_virtual : int
        The source's solver-stability floor (at least one occupied and
        two unoccupied orbitals in the space); a drop that would violate
        it is skipped.

    Returns
    -------
    RankedOrbitalChoice
        ``active == []`` when the cap cannot be reached within the floor
        (a refusal, not an error).
    """
    ei = np.asarray(eps_occupied, dtype=float).reshape(-1)
    ea = np.asarray(eps_virtual, dtype=float).reshape(-1)
    x = np.asarray(exchange_virtual, dtype=float).reshape(-1)
    if ei.size == 0 or ea.size == 0:
        raise ValueError('the occupied set and the virtual window must '
                         'both be non-empty')
    if x.shape != ea.shape:
        raise ValueError(f'exchange has shape {x.shape} for {ea.shape} '
                         f'virtual orbitals')
    occ_idx = (list(range(ei.size)) if occupied is None
               else [int(i) for i in occupied])
    vir_idx = (list(range(ei.size, ei.size + ea.size))
               if virtual is None else [int(i) for i in virtual])
    sing_idx = ([] if singly_occupied is None
                else [int(i) for i in singly_occupied])
    if len(occ_idx) != ei.size or len(vir_idx) != ea.size:
        raise ValueError('index label lists must match the array sizes')
    if n_csf_cap < 1:
        raise ValueError(f'n_csf_cap {n_csf_cap} must be positive')

    D = ea[None, :] - ei[:, None]                       # (nocc, nvir)
    c = -x[None, :] / (D + np.sqrt(x[None, :] ** 2 + D ** 2))
    c2 = c * c
    s_occ = _binary_entropy(c2.sum(axis=1) / (1.0 + c2.sum(axis=1)))
    s_vir = _binary_entropy(c2.sum(axis=0) / (1.0 + c2.sum(axis=0)))

    ent = {}
    for label, value in zip(occ_idx, s_occ):
        ent[label] = float(value)
    for label, value in zip(vir_idx, s_vir):
        ent[label] = float(value)
    for label in sing_idx:
        ent[label] = float(np.log(2.0))
    kept_occ, kept_vir, kept_sing, dropped = (list(occ_idx), list(vir_idx),
                                              list(sing_idx), [])
    evidence = [
        f'pool {len(kept_occ)} occupied + {len(kept_vir)} virtual ranked '
        f'by APC (eqs. 12/13/18/19); the source overestimates occupied '
        f'entropies (R^2 0.64 vs DMRG) and degrades where the HF '
        f'determinant is a poor approximation'
    ]
    if kept_sing:
        evidence.append(
            f'{len(kept_sing)} singly occupied {sorted(kept_sing)} given '
            f'the highest entropy (ln 2) and entered unconditionally '
            f'(source open-shell rule)')

    def _count():
        return n_csf(len(kept_occ) + len(kept_sing) + len(kept_vir),
                     2 * len(kept_occ) + len(kept_sing))

    while _count() > n_csf_cap:
        cands = []
        if len(kept_occ) > min_occupied:
            cands += [(ent[i], 0, i) for i in kept_occ]
        if len(kept_vir) > min_virtual:
            cands += [(ent[i], 1, i) for i in kept_vir]
        if not cands:
            evidence.append(
                f'cap {n_csf_cap} unreachable within the '
                f'{min_occupied}-occupied/{min_virtual}-virtual floor; '
                f'refusing')
            return RankedOrbitalChoice([], ent, 0, evidence)
        # Lowest entropy first; near-ties (1e-12) rank by label.
        cands.sort(key=lambda t: (round(t[0] / _TIE_TOL), t[2]))
        _, block, label = cands[0]
        (kept_occ if block == 0 else kept_vir).remove(label)
        dropped.append(label)

    active = sorted(kept_occ + kept_sing + kept_vir)
    count = _count()
    if dropped:
        evidence.append(f'dropped {sorted(dropped)} (lowest ranked) to '
                        f'reach N_CSF = {count} <= {n_csf_cap}')
    evidence.append(
        f'selection: {len(kept_occ)} occupied + {len(kept_vir)} virtual, '
        f'N_CSF = {count}; bias is expected where the HF determinant is '
        f'a poor approximation')
    return RankedOrbitalChoice(active, ent, count, evidence)
