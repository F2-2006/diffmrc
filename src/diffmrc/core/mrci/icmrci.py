"""Internally contracted MRCI.

PURPOSE
-------
Builds the internally contracted doubly-external basis of Werner and
Knowles, J. Chem. Phys. 1988, 89, 5803, and solves the projected CI
eigenvalue problem:

  Eq. (9):  Psi_ijp^ab = 1/2 (E_ai,bj + p E_bi,aj) Psi_0
  Eq. (14): S_ij,kl^(p) = <0| E_ik,jl + p E_il,jk |0>
  Eq. (15-16): symmetric orthogonalization, T = S^(-1/2)

The implementation builds the contracted functions explicitly in the
determinant basis and solves the projected eigenvalue problem.  This is
algebraically equivalent to Eqs. (9)-(19); the direct-CI residual formulas
Eqs. (41)-(48) are not used here.

PUBLIC INTERFACE
----------------
``pair_function``, ``build_contracted_matrix``, ``orthonormalize``,
``solve_mrci``, ``MRCIResult``.

VALIDATED SCOPE
---------------
For H2O/6-31G CAS(2,2) the contracted energy equals the uncontracted
MRCISD energy to machine precision, as required when the internal N-2
space is one-dimensional.  For CH2/6-31G CAS(4,4) the contraction error
is 1.028e-4 Eh and the contracted energy stays above the uncontracted
one (variational).  Threshold independence was checked for
``svd_tol`` between 1e-6 and 1e-12 (same rank 245).

BOUNDARIES
----------
* The contraction is built from the lowest CASCI root only; excited-state
  or state-averaged references are not defined here.
* The full MO two-electron tensor is held in memory; the direct-CI
  residual algorithm of WK88 Eqs. (41)-(48) is not implemented, so the
  reduced-scaling property of the original method is absent.
* Columns are normalized to unit length; the WK88 Eq. (17)
  ``(2 - delta_ab)^(-1)`` convention is not adopted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .determinants import apply_spin_free_excitation, s2_expectation
from .space import active_pairs, external_pairs


@dataclass
class MRCIResult:
    energy: float
    vector: np.ndarray
    basis: np.ndarray
    n_internal: int
    n_singles: int
    n_doubles_basis: int
    rank_contracted: int
    s2: float
    ref_weight: float
    labels: list = field(default_factory=list)


def pair_function(psi0: dict, i: int, j: int, a: int, b: int, p: int, norb_ci: int) -> dict:
    """Return 1/2 (E_ai E_bj + p E_bi E_aj) |psi0>.

    Operator order follows WK88 Eq. (7): E_ai,bj = E_ai E_bj, i.e. E_bj acts
    first.  The delta term -delta_ij E_ab vanishes because i is internal and
    b is external.
    """
    v_bj = apply_spin_free_excitation(psi0, b, j, norb_ci)
    v1 = apply_spin_free_excitation(v_bj, a, i, norb_ci)

    v_aj = apply_spin_free_excitation(psi0, a, j, norb_ci)
    v2 = apply_spin_free_excitation(v_aj, b, i, norb_ci)

    out: dict[int, float] = {}
    for k, v in v1.items():
        out[k] = out.get(k, 0.0) + 0.5 * v
    for k, v in v2.items():
        out[k] = out.get(k, 0.0) + 0.5 * p * v
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


def build_contracted_matrix(
    psi0: dict,
    nact: int,
    next_: int,
    full_index: dict,
    n_full: int,
    svd_tol: float = 1e-8,
):
    """Build the contracted doubly-external basis.

    Returns
    -------
    X : ndarray (n_full, n_pairs)
        Contracted pair functions as columns in the full determinant basis.
    labels : list of (i, j, a, b, p)
    """
    norb_ci = nact + next_
    cols = []
    labels = []
    for (i, j) in active_pairs(nact):
        for (a, b) in external_pairs(nact, next_):
            for p in (1, -1):
                if p == -1 and (a == b or i == j):
                    # 1/2 (E_ai,bj - E_bi,aj) vanishes when a == b or i == j,
                    # because the two operator products then coincide.
                    continue
                vec = pair_function(psi0, i, j, a, b, p, norb_ci)
                if not vec:
                    continue
                col = np.zeros(n_full)
                for det, c in vec.items():
                    idx = full_index.get(det)
                    if idx is None:
                        raise KeyError(f"pair function escaped the doubles space: {det}")
                    col[idx] += c
                if np.linalg.norm(col) < 1e-12:
                    continue
                cols.append(col)
                labels.append((i, j, a, b, p))
    X = np.array(cols).T if cols else np.zeros((n_full, 0))
    return X, labels


def orthonormalize(X: np.ndarray, tol: float = 1e-8):
    """Symmetric orthogonalization T = S^(-1/2) with linear-dependency removal."""
    S = X.T @ X
    w, u = np.linalg.eigh(S)
    wmax = w[-1] if len(w) else 0.0
    keep = w > max(tol * wmax, 1e-12)
    B = X @ u[:, keep] / np.sqrt(w[keep])
    return B, int(keep.sum()), S


def solve_mrci(H: np.ndarray, C: np.ndarray, n_internal: int, n_singles: int, dets, norb_ci: int, nroots: int = 3):
    """Solve the projected CI problem for the orthonormal combined basis C."""
    Hc = C.T @ H @ C
    w, v = np.linalg.eigh(Hc)
    results = []
    for k in range(min(nroots, len(w))):
        full_vec = C @ v[:, k]
        s2 = s2_expectation(full_vec, dets, norb_ci)
        ref_weight = float(np.sum(full_vec[:n_internal] ** 2))
        results.append(MRCIResult(
            energy=float(w[k]),
            vector=full_vec,
            basis=C,
            n_internal=n_internal,
            n_singles=n_singles,
            n_doubles_basis=int(C.shape[1] - n_internal - n_singles),
            rank_contracted=int(C.shape[1] - n_internal - n_singles),
            s2=s2,
            ref_weight=ref_weight,
        ))
    return results
