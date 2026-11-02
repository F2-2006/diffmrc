"""Deterministic orbital phase convention: invariants and stability."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.gauge import (canonical_mo_phases,  # noqa: E402
                                canonical_mo_phases_np)


def _random_coefficients(seed=3, nao=12, nmo=8):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
    return q


def test_the_convention_is_invariant_to_column_signs():
    """Flipping any set of columns must not change the canonical result."""
    C = _random_coefficients()
    flips = np.ones(C.shape[1])
    flips[::2] = -1.0
    a = np.asarray(canonical_mo_phases(jnp.asarray(C)))
    b = np.asarray(canonical_mo_phases(jnp.asarray(C * flips)))
    assert np.allclose(a, b, atol=0.0)


def test_the_convention_is_idempotent():
    C = _random_coefficients(seed=11)
    once = np.asarray(canonical_mo_phases(jnp.asarray(C)))
    twice = np.asarray(canonical_mo_phases(jnp.asarray(once)))
    assert np.array_equal(once, twice)


def test_a_tie_inside_the_band_does_not_flip_the_sign():
    """Symmetric partners equal to roundoff must not decide the phase.

    Two entries inside the band tie; the lowest index wins, so the sign
    is the same whether the roundoff happens to order them one way or
    the other.
    """
    C = np.zeros((6, 1))
    C[2, 0] = 0.6
    C[4, 0] = 0.6 + 2e-16
    C[0, 0] = -0.5
    a = np.asarray(canonical_mo_phases(jnp.asarray(C)))
    C2 = C.copy()
    C2[2, 0], C2[4, 0] = C[4, 0], C[2, 0]
    b = np.asarray(canonical_mo_phases(jnp.asarray(C2)))
    # The first entry inside the band (row 2) is made positive in both.
    assert a[2, 0] > 0.0
    assert b[2, 0] > 0.0
    assert a[0, 0] < 0.0 and b[0, 0] < 0.0


def test_it_rejects_a_one_dimensional_array():
    with pytest.raises(ValueError):
        canonical_mo_phases(jnp.zeros(5))


def test_the_numpy_sibling_matches_the_jax_function():
    """The X2C layer uses the NumPy rule; the two must not drift apart."""
    C = _random_coefficients(seed=5)
    flips = np.ones(C.shape[1])
    flips[1::3] = -1.0
    flipped = C * flips
    a = np.asarray(canonical_mo_phases(jnp.asarray(flipped)))
    b = canonical_mo_phases_np(flipped)
    assert np.array_equal(a, b)


def test_the_numpy_sibling_handles_an_empty_column_set():
    C = canonical_mo_phases_np(np.zeros((4, 0)))
    assert C.shape == (4, 0)


def test_the_numpy_sibling_rejects_a_one_dimensional_array():
    with pytest.raises(ValueError):
        canonical_mo_phases_np(np.zeros(5))
