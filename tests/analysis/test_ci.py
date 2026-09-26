"""Occupations and entropies straight from a CI vector's bitmaps.

The convention under test is the kernel's: determinant ``d`` is an integer
bitmap, bit ``s`` is spin orbital ``s``, with the alpha orbital of active
orbital ``j`` at bit ``2*(ncore+j)`` and the beta at the next bit.  The
last test binds this module to the kernel's RDM builder on a real CI
vector, so the convention is pinned against the chain, not against the
comment above it.
"""
import numpy as np
import pytest

from diffmrc.analysis.ci import (
    entropy_from_ci,
    occupation_masks,
    occupations_from_ci,
)


def _det(occupied):
    d = 0
    for so in occupied:
        d |= 1 << so
    return d


# One core orbital (spin orbitals 0, 1) and a CAS(2e,2o) over 2..5.
_DETS = [_det([0, 1, 2, 3]), _det([0, 1, 4, 5]), _det([0, 1, 2, 5])]


def test_masks_pick_the_active_window_only():
    ma, mb, m2 = occupation_masks(_DETS, ncore=1, nact=2)
    assert ma.shape == mb.shape == m2.shape == (3, 2)
    assert np.array_equal(ma, [[True, False], [False, True],
                               [True, False]])
    assert np.array_equal(mb, [[True, False], [False, True],
                               [False, True]])
    assert np.array_equal(m2, [[True, False], [False, True],
                               [False, False]])


def test_occupations_count_weights_deterministically():
    c = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)])
    n_a, n_b, p2 = occupations_from_ci(c, _DETS, ncore=1, nact=2)
    assert np.allclose(n_a, [0.7, 0.3])
    assert np.allclose(n_b, [0.5, 0.5])
    assert np.allclose(p2, [0.5, 0.3])


def test_a_pinned_orbital_carries_no_entropy_and_a_toggling_one_ln2():
    """Two determinants split the weight evenly between orbitals 0 and 1.

    Each of the two orbitals is doubly occupied in one determinant and
    empty in the other, half the time each: the maximally fluctuating
    two-state pattern, entropy ln 2. The remaining orbitals are empty.
    """
    dets = [_det([0, 1]), _det([2, 3])]
    c = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    s1 = entropy_from_ci(c, dets, ncore=0, nact=8)
    assert s1.shape == (8,)
    assert np.allclose(s1[:2], np.log(2.0), atol=1e-12)
    assert np.allclose(s1[2:], 0.0, atol=1e-12)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        occupations_from_ci(np.ones(2), _DETS, ncore=1, nact=2)


def test_the_active_window_must_fit_a_64_bit_word():
    dets = [_det([0])]
    with pytest.raises(ValueError):
        occupation_masks(dets, ncore=0, nact=32)


def test_the_two_routes_agree_on_a_real_ci_vector():
    """The bit-count against the kernel's RDM builder, same vector.

    Confined to the S_z = 0 sector so that n_alpha == n_beta per orbital,
    which is what lets the spin-summed RDM diagonal be split in two.
    """
    pytest.importorskip('jax')
    from diffmrc.analysis.entropy import single_orbital_entropy
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.nevpt2.rdm import make_dm123_sf

    ncore, nact, nele = 0, 4, 4
    dets = cas_dets_for2(ncore, nact, nele)
    rng = np.random.default_rng(11)
    c = rng.normal(size=len(dets))
    keep = []
    for d in dets:
        a = sum(1 for i in range(0, 2 * nact, 2) if (d >> i) & 1)
        b = sum(1 for i in range(1, 2 * nact, 2) if (d >> i) & 1)
        keep.append(a == b)
    c = np.where(keep, c, 0.0)
    c = c / np.linalg.norm(c)

    s_bit = entropy_from_ci(c, dets, ncore=ncore, nact=nact)
    dm1, dm2, _ = make_dm123_sf(c, dets, nact, ncore)
    dm1 = np.asarray(dm1)
    dm2 = np.asarray(dm2)
    n_a = np.diag(dm1) / 2.0
    p2 = np.array([dm2[i, i, i, i] for i in range(nact)]) / 2.0
    s_rdm = single_orbital_entropy(n_a, n_a, p2)
    assert np.allclose(s_bit, s_rdm, atol=1e-10)
