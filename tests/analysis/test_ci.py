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


def test_the_bit_count_agrees_with_explicit_second_quantization():
    """n_a, n_b and p2 against rdm1_spin / rdm2_spin, same vector.

    The spin-orbital builders apply the operators to the determinants
    directly, so they are an independent implementation of the convention
    the masks encode; a random vector suffices and no spin symmetry is
    used anywhere.
    """
    pytest.importorskip('jax')
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.nevpt2.rdm import rdm1_spin, rdm2_spin

    ncore, nact, nele = 0, 2, 2
    dets = cas_dets_for2(ncore, nact, nele)
    rng = np.random.default_rng(5)
    c = rng.normal(size=len(dets))
    c = c / np.linalg.norm(c)

    n_a, n_b, p2 = occupations_from_ci(c, dets, ncore=ncore, nact=nact)
    g1 = np.asarray(rdm1_spin(c, dets, ncas=nact))
    g2 = np.asarray(rdm2_spin(c, dets, ncas=nact))
    assert np.allclose(n_a, np.diag(g1)[0::2], atol=1e-12)
    assert np.allclose(n_b, np.diag(g1)[1::2], atol=1e-12)
    # Gamma[p,q,r,s] = <a+_p a+_q a_s a_r>; the double occupancy is
    # <n_a n_b> = <a+_alpha a+_beta a_beta a_alpha> at [alpha, beta,
    # alpha, beta].
    assert np.allclose(p2, [g2[2 * i, 2 * i + 1, 2 * i, 2 * i + 1]
                            for i in range(nact)], atol=1e-12)


def test_the_production_builder_agrees_under_its_measured_convention():
    """The bit-count against make_dm123_sf at CAS(4,4), same vector.

    That builder follows PySCF ``fci.rdm.make_dm123``'s *pre-reorder*
    convention, 2pdm[p,q,r,s] = <p+ q r+ s>, whose same-orbital element
    is (n_a + n_b)^2 -- measured here as
    dm2[i,i,i,i] = diag(dm1)[i] + 2 p2[i] (a single |alpha beta>
    determinant gives 4, not the 2 of the reordered/normal convention
    <p+ q+ s r> that PySCF's fcisolver.make_rdm2 and
    analysis.rdm.entropy_from_rdms use). Halving *this* builder's
    diagonal would not give p2, so the conversion is written out rather
    than assumed.
    """
    pytest.importorskip('jax')
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.nevpt2.rdm import make_dm123_sf

    ncore, nact, nele = 0, 4, 4
    dets = cas_dets_for2(ncore, nact, nele)
    rng = np.random.default_rng(11)
    c = rng.normal(size=len(dets))
    c = c / np.linalg.norm(c)

    n_a, n_b, p2 = occupations_from_ci(c, dets, ncore=ncore, nact=nact)
    dm1, dm2, _ = make_dm123_sf(c, dets, nact, ncore)
    dm1 = np.asarray(dm1)
    dm2 = np.asarray(dm2)
    diag = np.diag(dm1)
    assert np.allclose(n_a + n_b, diag, atol=1e-10)
    assert np.allclose(p2, [(dm2[i, i, i, i] - diag[i]) / 2.0
                            for i in range(nact)], atol=1e-10)


def test_the_pre_reorder_diagonal_on_a_single_determinant():
    """|alpha beta> on one orbital: the builder's diagonal is 4, not 2.

    The sharpest single point of the convention difference. Under the
    normal spin-summed convention the value would be 2 p2 = 2; the
    pre-reorder convention gives (n_a + n_b)^2 = 4. Exact, integer
    arithmetic -- no tolerance.
    """
    pytest.importorskip('jax')
    from diffmrc.core.nevpt2.rdm import make_dm123_sf

    dm1, dm2, _ = make_dm123_sf(np.array([1.0]), [3], ncas=1, ncore=0)
    assert float(np.asarray(dm1)[0, 0]) == 2.0
    assert float(np.asarray(dm2)[0, 0, 0, 0]) == 4.0


from diffmrc.analysis.ci import (  # noqa: E402
    mutual_information,
    pair_density,
    two_orbital_entropy,
)


def test_a_single_determinant_is_a_product_state_everywhere():
    dets = [_det([0, 1, 4, 5])]
    c = np.array([1.0])
    I = mutual_information(c, dets, ncore=1, nact=2)
    assert I.shape == (2, 2)
    assert np.allclose(I, 0.0, atol=1e-12)
    s2 = two_orbital_entropy(c, dets, 0, 1, ncore=1, nact=2)
    assert s2 == pytest.approx(0.0, abs=1e-12)


def test_a_pure_pair_state_carries_2ln2_against_an_inert_one():
    """(|up-dn 0> + |0 up-dn>)/sqrt(2) on orbitals 0 and 1, orbital 2 inert.

    The (0, 1) pair sits in a pure superposition of two orthogonal
    patterns with equal weight and is a product with the inert orbital:
    s2(0, 1) = 0 while s1 = ln 2 each, so I_01 = 2 ln 2, the maximum
    for a two-orbital singlet.  Orbital 2 is doubly occupied
    throughout: s1_2 = 0 and I_02 = ln 2 + 0 - ln 2 = 0.
    """
    dets = [_det([0, 1, 4, 5]), _det([2, 3, 4, 5])]
    c = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    assert two_orbital_entropy(c, dets, 0, 1, ncore=0, nact=3) == \
        pytest.approx(0.0, abs=1e-12)
    I = mutual_information(c, dets, ncore=0, nact=3)
    assert I[0, 1] == pytest.approx(2.0 * np.log(2.0), abs=1e-12)
    assert I[1, 0] == pytest.approx(2.0 * np.log(2.0), abs=1e-12)
    assert I[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert I[1, 2] == pytest.approx(0.0, abs=1e-12)


def test_the_pair_density_is_hermitian_positive_and_unit_trace():
    dets = [_det([0, 1, 4, 5]), _det([2, 3, 4, 5]), _det([0, 2])]
    c = np.array([0.5, 0.5, np.sqrt(0.5)])
    rho = pair_density(c, dets, 0, 1, ncore=0, nact=3)
    assert rho.shape == (16, 16)
    assert np.allclose(rho, rho.T, atol=1e-12)
    w = np.linalg.eigvalsh(rho)
    assert w.min() > -1e-10
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_the_orbital_order_is_enforced():
    dets = [_det([0, 1])]
    with pytest.raises(ValueError):
        pair_density(np.array([1.0]), dets, 1, 0, ncore=0, nact=2)
    with pytest.raises(ValueError):
        pair_density(np.array([1.0]), dets, 0, 0, ncore=0, nact=2)


def test_the_pair_density_is_core_shift_invariant():
    """A 30-orbital frozen core must not change the pair density.

    The bit arithmetic is active-relative; a global layout needs
    2*(ncore + nact) bits and overflows int64 here (the measured LaF
    crash, 2026-10-11: ncore 31, a 69-bit global determinant).  The
    frozen core is fully occupied in every determinant, so its
    contribution to the crossing parity is one constant sign for every
    amplitude and rho = V^T V is invariant under the shift.
    """
    nact = 2
    # 2 electrons, Sz = 0; active bits (a0, b0, a1, b1)
    patterns = [0b0011, 0b1100, 0b0101, 0b1010]
    c = np.array([0.6, 0.4, 0.3, 0.7])
    c = c / np.linalg.norm(c)
    rho_small = pair_density(c, patterns, 0, 1, ncore=0, nact=nact)
    ncore = 30
    core = (1 << (2 * ncore)) - 1
    shifted = [(p << (2 * ncore)) | core for p in patterns]
    rho_shifted = pair_density(c, shifted, 0, 1, ncore=ncore, nact=nact)
    assert np.allclose(rho_shifted, rho_small, atol=1e-12)


def test_dets_and_vector_from_fci_matches_hand_computed_layout():
    """ncore=1, nact=2: active j=0 -> bits 2 (alpha) / 3 (beta), j=1 ->
    bits 4 / 5; enumeration alpha outer, beta inner, both ascending."""
    from diffmrc.analysis.ci import dets_and_vector_from_fci

    alpha = [0b01, 0b10]
    beta = [0b01, 0b10]
    ci = np.array([[0.7, 0.1], [0.1, 0.7071]])
    dets, c = dets_and_vector_from_fci(ci, alpha, beta, ncore=1, nact=2)
    assert dets == [12, 36, 24, 48]
    assert np.allclose(c, [0.7, 0.1, 0.1, 0.7071])


def test_the_fci_vector_length_must_match_the_string_lists():
    from diffmrc.analysis.ci import dets_and_vector_from_fci

    with pytest.raises(ValueError, match='do not belong'):
        dets_and_vector_from_fci(np.zeros(3), [0b01, 0b10], [0b01, 0b10],
                                 ncore=0, nact=2)
