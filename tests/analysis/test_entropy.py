"""Single-orbital entropy: analytic limits and symmetry."""
import numpy as np
import pytest

from diffmrc.analysis.entropy import single_orbital_entropy

LN4 = np.log(4.0)


def test_empty_orbital_has_zero_entropy():
    s1 = single_orbital_entropy(np.array([0.0]), np.array([0.0]),
                                np.array([0.0]))
    assert s1[0] == pytest.approx(0.0, abs=1e-12)


def test_doubly_occupied_orbital_has_zero_entropy():
    s1 = single_orbital_entropy(np.array([1.0]), np.array([1.0]),
                                np.array([1.0]))
    assert s1[0] == pytest.approx(0.0, abs=1e-12)


def test_singly_occupied_alpha_orbital_has_zero_entropy():
    s1 = single_orbital_entropy(np.array([1.0]), np.array([0.0]),
                                np.array([0.0]))
    assert s1[0] == pytest.approx(0.0, abs=1e-12)


def test_maximally_entangled_orbital_approaches_ln4():
    """Equal weights on all four states gives ln4.

    The weight vector is [1 - n_a - n_b + p2, n_a - p2, n_b - p2, p2].
    It is flat at 1/4 exactly when n_a = n_b = 1/2 and p2 = 1/4, which is
    the maximally entangled orbital: each spin is half occupied and the
    pair is doubly occupied a quarter of the time.
    """
    s1 = single_orbital_entropy(np.array([0.5]), np.array([0.5]),
                                np.array([0.25]))
    assert s1[0] == pytest.approx(LN4, abs=1e-12)


def test_entropy_at_half_occupation_is_not_ln4():
    """Half-occupied spins alone do not saturate the entropy.

    With p2 = 0 the weights are [0, 1/2, 1/2, 0], a classical mixture of
    two states, so the entropy is ln 2 rather than ln 4. This is the case
    that distinguishes a genuinely entangled orbital from a merely
    open-shell one.
    """
    s1 = single_orbital_entropy(np.array([0.5]), np.array([0.5]),
                                np.array([0.0]))
    assert s1[0] == pytest.approx(np.log(2.0), abs=1e-12)


def test_entropy_never_exceeds_ln4():
    rng = np.random.default_rng(20260921)
    n_a = rng.uniform(0, 1, 200)
    n_b = rng.uniform(0, 1, 200)
    p2 = np.minimum(n_a, n_b) * rng.uniform(0, 1, 200)
    s1 = single_orbital_entropy(n_a, n_b, p2)
    assert np.all(s1 <= LN4 + 1e-12)
    assert np.all(s1 >= -1e-12)


def test_alpha_beta_symmetry():
    """Swapping the two spins leaves the entropy unchanged."""
    rng = np.random.default_rng(7)
    n_a = rng.uniform(0, 1, 50)
    n_b = rng.uniform(0, 1, 50)
    p2 = np.minimum(n_a, n_b) * rng.uniform(0, 1, 50)
    assert np.allclose(single_orbital_entropy(n_a, n_b, p2),
                       single_orbital_entropy(n_b, n_a, p2))


def test_accepts_plain_lists():
    """array_like input works, not just ndarray."""
    s1 = single_orbital_entropy([0.0], [0.0], [0.0])
    assert s1.shape == (1,)
    assert s1[0] == pytest.approx(0.0, abs=1e-12)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        single_orbital_entropy(np.zeros(3), np.zeros(2), np.zeros(3))


def test_multidimensional_input_preserves_shape():
    """The 4-vector of weights is stacked on the leading axis."""
    n_a = np.full((2, 3), 0.5)
    n_b = np.full((2, 3), 0.5)
    p2 = np.full((2, 3), 0.25)
    s1 = single_orbital_entropy(n_a, n_b, p2)
    assert s1.shape == (2, 3)
