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

__all__ = ['occupation_masks', 'occupations_from_ci', 'entropy_from_ci',
           'pair_scatter', 'pair_density', 'two_orbital_entropy',
           'mutual_information']


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


def pair_scatter(dets, i, j, ncore=0, nact=2):
    """Static bit bookkeeping of the pair (i, j): (inv, pat, sign, ngroups).

    The pieces from which :func:`pair_density` assembles
    ``rho = V^T V``: the environment group id of every determinant
    (``inv``), its 16-pattern index (``pat``), the crossing-parity sign,
    and the number of environment groups.  Separated so the
    differentiable sibling (``driver.entropy``) can rebuild V as a
    segment sum over the CI vector instead of re-deriving the bit
    bookkeeping -- the components are functions of the determinant list
    alone and are hoisted out of any differentiation loop.

    The order i < j fixes the bit layout of ``pat`` and is enforced.

    The bit arithmetic runs on the active-relative bitmap (the
    determinant shifted past the frozen core), not on the global one:
    a global layout needs 2*(ncore + nact) bits and overflows int64 for
    the f-block chains (LaF: ncore 31 -> 69 bits; measured 2026-10-11
    when the first real-system mutual-information run crashed here on
    pair 0/2).  The shift is exact, not an approximation: the frozen
    core is fully occupied in every determinant of the CI space, so
    the core's contribution to the crossing parity is the same global
    sign for every amplitude -- one constant phase -- and the reduced
    density rho = V^T V is invariant under it.  (This is the same
    active-relative convention as the core.nevpt2.rdm bitmap fix,
    fee6311.)
    """
    dets = list(dets)
    if not (0 <= i < j < nact):
        raise ValueError(f'pair ({i}, {j}) outside the active space of '
                         f'{nact} orbitals; the order i < j fixes the '
                         f'bit layout')
    base = 2 * ncore
    g = (2 * i, 2 * i + 1, 2 * j, 2 * j + 1)
    pair_mask = 0
    for x in g:
        pair_mask |= 1 << x
    weight = (1, 2, 4, 8)
    active_mask = (1 << (2 * nact)) - 1
    nd = len(dets)
    pat = np.zeros(nd, dtype=np.int64)
    parity = np.zeros(nd, dtype=np.int64)
    env_key = np.zeros(nd, dtype=np.int64)
    for idx, d in enumerate(dets):
        d = (int(d) >> base) & active_mask
        p = 0
        for t, x in enumerate(g):
            if (d >> x) & 1:
                p |= weight[t]
                lower = (d & ((1 << x) - 1)) & ~pair_mask
                parity[idx] += bin(lower).count('1')
        pat[idx] = p
        env_key[idx] = d & ~pair_mask
    sign = np.where(parity % 2 == 0, 1.0, -1.0)
    keys, inv = np.unique(env_key, return_inverse=True)
    return inv, pat, sign, int(keys.size)


def pair_density(c_ci, dets, i, j, ncore=0, nact=2):
    """Two-orbital reduced density of the pair (i, j), shape (16, 16).

    The pair's Fock space has 16 occupation patterns of the four spin
    orbitals; the reduced state is the partial trace of |psi><psi| over
    the environment, built without fermionic-operator bookkeeping: each
    determinant factors into (pair pattern) x (environment), picking up
    the crossing parity p = (-1)^{number of occupied environment spin
    orbitals below each occupied pair spin orbital}, and the reduced
    matrix is the Gram matrix of the environment-indexed amplitudes,
    rho = V^T V.  Patterns of different electron count decouple
    automatically (no environment can host both amplitudes).

    The 16x16 basis does not diagonalize itself (two patterns of equal
    N and S_z can mix), so two-orbital spectroscopies rest on the
    eigenproblem of rho, not on occupation counting
    (Stein/Pantazis/Krewald 2019).  The order i < j fixes the bit
    layout and is enforced.
    """
    c = np.asarray(c_ci, dtype=float)
    dets = list(dets)
    if c.ndim != 1:
        raise ValueError(f'c_ci must be one-dimensional, got shape '
                         f'{c.shape}')
    if not (0 <= i < j < nact):
        raise ValueError(f'pair ({i}, {j}) outside the active space of '
                         f'{nact} orbitals; the order i < j fixes the '
                         f'bit layout')
    if c.shape[0] != len(dets):
        raise ValueError(f'c_ci has {c.shape[0]} coefficients for '
                         f'{len(dets)} determinants')
    inv, pat, sign, ngroups = pair_scatter(dets, i, j, ncore=ncore,
                                           nact=nact)
    V = np.zeros((ngroups, 16))
    np.add.at(V, (inv, pat), c * sign)
    return V.T @ V


def two_orbital_entropy(c_ci, dets, i, j, ncore=0, nact=2):
    """s_ij(2) = -sum_r w_r ln w_r over the 16 eigenvalues of the pair."""
    rho = pair_density(c_ci, dets, i, j, ncore=ncore, nact=nact)
    w = np.linalg.eigvalsh(0.5 * (rho + rho.T))
    w = w[w > 1e-14]
    return float(-(w * np.log(w)).sum())


def mutual_information(c_ci, dets, ncore=0, nact=2):
    """I_ij = s1_i + s1_j - s_ij(2) for every active orbital pair.

    The diagonal is zero by convention (an orbital carries no mutual
    information with itself).  I_ij >= 0 up to round-off; no pair-sum
    rule is assumed.  Cost is O(ndet) per pair in the current
    Python-level accumulation, which bounds this to the spaces the
    bitmask route already serves.
    """
    s1 = entropy_from_ci(c_ci, dets, ncore=ncore, nact=nact)
    I = np.zeros((nact, nact))
    for i in range(nact):
        for j in range(i + 1, nact):
            s2 = two_orbital_entropy(c_ci, dets, i, j, ncore=ncore,
                                     nact=nact)
            I[i, j] = I[j, i] = float(s1[i] + s1[j] - s2)
    return I
