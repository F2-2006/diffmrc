"""Bridge from reduced density matrices to single-orbital entropies.

The entropy module works on plain occupation numbers; this module extracts
them from the density matrices a wavefunction calculation produces. Keeping
the two apart means the entropy formulas can be validated on hand-built
numbers, and the extraction convention -- which is easy to get wrong -- is
isolated in one place.

Conventions
-----------

The two-particle density matrix is assumed to be **spin-summed**, the
convention of PySCF's ``fcisolver.make_rdm2``:

    rdm2[p, q, r, s] = sum_{sigma,tau} <p+_sigma q+_tau s_tau r_sigma>

so that the same-orbital element counts both spin orderings,

    rdm2[i, i, i, i] = 2 * <n_{i,alpha} n_{i,beta}>.

Note which builder this is *not*: the kernel's ``make_dm123_sf`` follows
PySCF ``fci.rdm.make_dm123``'s pre-reorder convention,
``2pdm[p,q,r,s] = <p+ q r+ s>``, whose same-orbital element is instead
``(n_alpha + n_beta)^2 = diag(dm1) + 2 p2`` (measured to ten decimals on
this project's builder, 2026-09-27; ``reorder_dm123`` is what would bring
it to the form this module expects).  Feeding that builder's ``dm2`` here
with ``spin_summed=True`` would halve a quantity that is not ``2 p2``.

The one-orbital entropy needs the probability of double occupation, which is
``<n_a n_b>``, not twice that. Passing the raw element through gives a doubly
occupied orbital ``p2 = 2``; the weight vector then clips to a half-and-half
mixture of the empty and doubly occupied states and the orbital is reported
with an entropy of ln 2 when it is in fact a closed shell. Pass
``spin_summed=False`` to supply a spin-resolved two-particle matrix, whose
diagonal is ``<n_a n_b>`` directly.

The one-particle matrices must be **spin-resolved** -- ``[alpha, beta]`` --
because the entropy formula needs the two spin occupations separately. Note
that this differs from the two-particle convention above; a calculation that
returns one of each kind is exactly the situation this module guards
against.
"""
import numpy as np

from .entropy import single_orbital_entropy

__all__ = ['entropy_from_rdms']


def entropy_from_rdms(rdm1s, rdm2, spin_summed=True):
    """Compute single-orbital entropies from reduced density matrices.

    Parameters
    ----------
    rdm1s : sequence of two array_like
        ``[alpha_1rdm, beta_1rdm]``, each of shape ``(ncas, ncas)`` and
        spin-resolved, so that ``diag(rdm1s[0])`` holds the alpha
        occupations.
    rdm2 : array_like
        Two-particle density matrix of shape ``(ncas, ncas, ncas, ncas)``
        in the ordering ``rdm2[p, q, r, s] = <p+ q+ s r>``.
    spin_summed : bool
        Whether `rdm2` follows the spin-summed convention. Default True,
        which matches PySCF and this project's kernel and causes the
        same-orbital diagonal to be halved.

    Returns
    -------
    s1 : numpy.ndarray
        Single-orbital entropy of each active orbital, shape ``(ncas,)``.

    Raises
    ------
    ValueError
        If the inputs have inconsistent shapes.
    """
    if len(rdm1s) != 2:
        raise ValueError(
            f'expected two spin blocks in rdm1s, got {len(rdm1s)}'
        )
    rdm1a = np.asarray(rdm1s[0], dtype=float)
    rdm1b = np.asarray(rdm1s[1], dtype=float)
    rdm2 = np.asarray(rdm2, dtype=float)

    if rdm1a.shape != rdm1b.shape or rdm1a.ndim != 2 \
            or rdm1a.shape[0] != rdm1a.shape[1]:
        raise ValueError(
            f'rdm1s must be two square matrices of equal shape, got '
            f'{rdm1a.shape} and {rdm1b.shape}'
        )
    ncas = rdm1a.shape[0]
    if rdm2.shape != (ncas, ncas, ncas, ncas):
        raise ValueError(
            f'rdm2 shape {rdm2.shape} does not match rdm1s dimension '
            f'{ncas}'
        )

    n_a = np.diag(rdm1a).copy()
    n_b = np.diag(rdm1b).copy()
    p2 = np.array([rdm2[i, i, i, i] for i in range(ncas)], dtype=float)
    if spin_summed:
        # Spin-summed diagonal counts both orderings of the pair.
        p2 = p2 * 0.5
    return single_orbital_entropy(n_a, n_b, p2)
