"""Single-orbital entanglement entropy.

The active-space selection protocol introduced by Stein and Reiher ranks
orbitals by the entanglement entropy of their one-orbital reduced density
matrix. The matrix has four eigenvalues in the occupation basis
{|0>, |up>, |down>, |up-down>}:

    w = [1 - n_a - n_b + p2,  n_a - p2,  n_b - p2,  p2]

with n_a and n_b the alpha and beta occupations of the orbital and p2 the
probability of finding it doubly occupied. The entropy is

    s1 = -sum_i w_i ln w_i

bounded above by ln 4. Orbitals that are always empty, singly occupied, or
doubly occupied have zero entropy; orbitals whose occupation fluctuates
strongly approach ln 4.

Conventions
-----------
``n_a``, ``n_b`` and ``p2`` must all be expressed in the *same* convention.
Mixing a spin-summed two-particle density with spin-resolved occupations
inflates ``p2`` by a factor of two for doubly occupied orbitals, which
clips to a weight vector of ``[0, 0.5, 0.5, 0]`` and reports ``ln 2`` for an
orbital that is in fact unentangled. See ``rdm.py`` for the extraction step
where that mistake would be made.

References
----------
C. J. Stein, M. Reiher, J. Chem. Theory Comput. 12, 1760 (2016).
C. J. Stein, M. Reiher, J. Comput. Chem. 40, 2216 (2019).
"""
import numpy as np

LN4 = float(np.log(4.0))
MAX_ENTROPY = LN4


def single_orbital_entropy(n_a, n_b, p2):
    """Compute the single-orbital entropy for a set of orbitals.

    Parameters
    ----------
    n_a, n_b : array_like
        Alpha and beta occupation numbers of each orbital. Any shape.
    p2 : array_like
        Probability that each orbital is doubly occupied, same shape.

    Returns
    -------
    s1 : numpy.ndarray
        Entropy of each orbital, in the range [0, ln 4], of the same shape
        as the inputs.

    Raises
    ------
    ValueError
        If the three inputs do not all have the same shape.
    """
    n_a = np.asarray(n_a, dtype=float)
    n_b = np.asarray(n_b, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if not (n_a.shape == n_b.shape == p2.shape):
        raise ValueError(
            f'shape mismatch: n_a{n_a.shape}, n_b{n_b.shape}, p2{p2.shape}'
        )

    # Stack the four occupation-basis weights on a leading axis. For
    # physical occupations the four entries are already non-negative and
    # sum to one.
    w = np.empty((4,) + n_a.shape, dtype=float)
    w[0] = 1.0 - n_a - n_b + p2
    w[1] = n_a - p2
    w[2] = n_b - p2
    w[3] = p2

    # Round-off can push a weight a few ulps outside [0, 1], and a
    # malformed input can push it far outside. Clipping alone would leave
    # the weights summing to something other than one, so renormalize:
    # the result is the entropy of the nearest valid distribution and
    # stays inside [0, ln 4] by construction.
    w = np.clip(w, 0.0, 1.0)
    total = w.sum(axis=0)
    w = np.divide(w, total, out=np.full_like(w, 0.25),
                  where=total > 0.0)

    positive = w > 0.0
    contrib = np.where(positive, w * np.log(np.where(positive, w, 1.0)), 0.0)
    s1 = -contrib.sum(axis=0)
    # A tiny negative can survive the sum at the extremes.
    return np.maximum(s1, 0.0)
