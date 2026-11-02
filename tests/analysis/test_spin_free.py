"""Spin-free (Pittner) entropy against the manifold-averaged reading."""
import numpy as np
import pytest

from diffmrc.analysis.ci import (manifold_entropy_from_ci,  # noqa: F401
                                 spin_free_orbital_entropy,
                                 spin_free_weights)


def test_the_three_weights_are_normalized():
    for n_a, n_b, p2 in [(1.0, 1.0, 0.0), (0.7, 0.7, 0.01), (0.0, 0.0, 0.0)]:
        w = np.asarray(spin_free_weights(n_a, n_b, p2))
        assert w.shape == (3,)
        assert w.sum() == pytest.approx(1.0, abs=1e-14)


def test_a_pure_singly_occupied_orbital_has_zero_spin_free_entropy():
    """With one electron, the spin orientation is merged away."""
    assert spin_free_orbital_entropy(1.0, 1.0, 0.0) == 0.0


def test_the_manifold_reading_exceeds_the_spin_free_one_by_the_spin_term():
    """S_manifold = S_spin_free + (N - 2 p2) ln 2 for a spin-adapted average.

    The equal-weight average over 2S+1 components gives n_alpha = n_beta =
    N/2; the manifold four-state entropy then keeps the ln 2 of every
    singly occupied electron that the spin-free construction merges.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        w0, w1, w2 = rng.dirichlet(np.ones(3))
        n = w1 + 2.0 * w2
        p2 = w2
        w = np.array([w0, w1 / 2.0, w1 / 2.0, w2])
        w = w[w > 0.0]
        s_manifold = -(w * np.log(w)).sum()
        s_free = spin_free_orbital_entropy(n / 2.0, n / 2.0, p2)
        assert s_manifold - s_free == pytest.approx(
            (n - 2.0 * p2) * np.log(2.0), abs=1e-12)
