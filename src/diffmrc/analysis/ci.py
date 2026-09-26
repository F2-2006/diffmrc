"""Single-orbital occupations and entropies straight from a CI vector.

The kernel's determinant convention is followed, not interpreted: a
determinant is an integer bitmap, bit ``s`` is spin orbital ``s``, the
alpha orbital of active orbital ``j`` sits at bit ``2*(ncore+j)`` and the
beta at the next bit (``core/nevpt2/m4.py``, ``cas_dets_for2``).

Why this exists next to the density-matrix route: ``make_dm123_sf``
assembles full one-, two- and three-particle matrices, and its cost grows
steeply with the active space; the occupation numbers a selection
criterion needs are three bit tests per determinant and orbital, with no
density matrix at all.  This is the route that makes entropy-based
selection reach the active-space sizes the f-element targets live at.

The masks are built with Python's arbitrary-precision integers and then
consumed as fixed arrays, so a frozen core of any size is fine as long as
the *active* window fits a 64-bit word; a larger window raises rather
than wrapping silently.
"""
import numpy as np

from .entropy import single_orbital_entropy

__all__ = ['occupation_masks', 'occupations_from_ci', 'entropy_from_ci']


def occupation_masks(dets, ncore=0, nact=2):
    """0/1 masks of which determinants occupy each active spin orbital.

    Returns ``(mask_a, mask_b, mask_2)``: which determinants hold the
    alpha electron of each active orbital, the beta electron, and both.
    Built once per determinant list and reused across evaluations, which
    is what keeps the per-call cost at one matmul per occupation.
    """
    if ncore < 0 or nact < 1:
        raise ValueError(f'ncore={ncore}, nact={nact} outside the '
                         f'physical range')
    span = 2 * nact
    if span > 63:
        raise ValueError(
            f'active space of {nact} orbitals spans {span} spin orbitals, '
            f'past the 63-bit window these bitmaps are read with')
    shift = 2 * ncore
    mask = (1 << span) - 1
    active = np.array([(int(d) >> shift) & mask for d in dets],
                      dtype=np.int64)
    bits = (active[:, None] & (1 << np.arange(span))[None, :]) != 0
    mask_a = bits[:, 0::2]
    mask_b = bits[:, 1::2]
    return mask_a, mask_b, mask_a & mask_b


def occupations_from_ci(c_ci, dets, ncore=0, nact=2):
    """``(n_alpha, n_beta, p2)`` per active orbital of a CI vector.

    ``n_x,i = sum_I |c_I|^2 [spin orbital x of orbital i occupied in I]``
    and ``p2,i`` the same with both spins; the weights are the quadratic
    map's, so the whole extraction is linear in ``|c_I|^2``.
    """
    c = np.asarray(c_ci, dtype=float)
    if c.ndim != 1:
        raise ValueError(f'c_ci must be one-dimensional, got shape '
                         f'{c.shape}')
    mask_a, mask_b, mask_2 = occupation_masks(dets, ncore=ncore, nact=nact)
    if c.shape[0] != mask_a.shape[0]:
        raise ValueError(
            f'c_ci has {c.shape[0]} coefficients for {mask_a.shape[0]} '
            f'determinants')
    w = c * c
    return (w @ mask_a.astype(float), w @ mask_b.astype(float),
            w @ mask_2.astype(float))


def entropy_from_ci(c_ci, dets, ncore=0, nact=2):
    """Single-orbital entropies of the active space of a CI vector.

    The differentiable sibling of this function is
    ``driver.entropy.single_orbital_entropy_from_ci``; the two are bound
    by a test that evaluates both on the same vector.
    """
    n_a, n_b, p2 = occupations_from_ci(c_ci, dets, ncore=ncore, nact=nact)
    return single_orbital_entropy(n_a, n_b, p2)
