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


def eigenvalue_entropy(w, xp=None):
    """``-sum w ln w`` over the positive eigenvalues of a spectrum.

    The entropy formula shared by the four occupation weights and the
    16 eigenvalues of a pair density; separated here so the numpy and
    the differentiable routes run one expression instead of two free to
    drift apart. Negative round-off is clipped to zero; the log takes a
    protected argument so a zero eigenvalue cannot send ``0 * log(0)``
    through a backward pass (the same guard as in
    ``single_orbital_entropy``).

    Parameters
    ----------
    w : array_like
        Eigenvalues (any one-dimensional array works; only positive
        entries contribute).
    xp : module, optional
        Array backend; defaults to numpy.

    Returns
    -------
    float (numpy) or scalar array (jax)
    """
    if xp is None:
        xp = np
    w = xp.maximum(xp.asarray(w, dtype=float), 0.0)
    positive = w > 0.0
    contrib = xp.where(positive,
                       w * xp.log(xp.where(positive, w, 1.0)), 0.0)
    return -contrib.sum()


def single_orbital_entropy(n_a, n_b, p2, xp=None):
    """Compute the single-orbital entropy for a set of orbitals.

    Parameters
    ----------
    n_a, n_b : array_like
        Alpha and beta occupation numbers of each orbital. Any shape.
    p2 : array_like
        Probability that each orbital is doubly occupied, same shape.
    xp : module, optional
        Array backend; defaults to numpy. The chain passes ``jax.numpy``
        so that the differentiable path and this module share one
        formula instead of two copies free to drift apart.

    Returns
    -------
    s1 : ndarray
        Entropy of each orbital, in the range [0, ln 4], of the same shape
        as the inputs.

    Raises
    ------
    ValueError
        If the three inputs do not all have the same shape.
    """
    if xp is None:
        xp = np
    n_a = xp.asarray(n_a, dtype=float)
    n_b = xp.asarray(n_b, dtype=float)
    p2 = xp.asarray(p2, dtype=float)
    if not (n_a.shape == n_b.shape == p2.shape):
        raise ValueError(
            f'shape mismatch: n_a{n_a.shape}, n_b{n_b.shape}, p2{p2.shape}'
        )

    # Stack the four occupation-basis weights on a leading axis. For
    # physical occupations the four entries are already non-negative and
    # sum to one.
    w = xp.stack([1.0 - n_a - n_b + p2, n_a - p2, n_b - p2, p2])

    # Round-off can push a weight a few ulps outside [0, 1], and a
    # malformed input can push it far outside. Clipping alone would leave
    # the weights summing to something other than one, so renormalize; a
    # zero-sum vector (only reachable from malformed input) keeps the
    # long-standing 0.25 fallback, which reads as maximum entropy.
    w = xp.clip(w, 0.0, 1.0)
    total = w.sum(axis=0)
    w = w / xp.where(total > 0.0, total, 1.0)
    w = xp.where(total > 0.0, w, 0.25)

    positive = w > 0.0
    contrib = xp.where(positive,
                       w * xp.log(xp.where(positive, w, 1.0)), 0.0)
    s1 = -contrib.sum(axis=0)
    # A tiny negative can survive the sum at the extremes.
    return xp.maximum(s1, 0.0)
