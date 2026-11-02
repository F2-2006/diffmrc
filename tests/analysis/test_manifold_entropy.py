"""Manifold-merged single-orbital entropy.

The criterion under test is Stein and Reiher's four-state entropy, read
after the degenerate components of a spin multiplet are merged. The
reason is the one the selection protocol needs: the ``M_S = +S``
component of a high-spin determinant carries one spin per open shell and
therefore reports zero entropy for every one of those shells, although
all of them are degenerate and belong in the active space.

The determinant ordering is the kernel's, so the vectors here are built
with ``cas_dets_for2`` and not with a private convention.
"""
import numpy as np
import pytest

from diffmrc.analysis.ci import (
    entropy_from_ci,
    manifold_entropy_from_ci,
    manifold_occupations,
)

LN2 = float(np.log(2.0))


def _vector(dets, occupied):
    v = np.zeros(len(dets))
    v[dets.index(occupied)] = 1.0
    return v


def _dets_for(ncore, nact, nele):
    """The kernel's determinant list, rebuilt without the pyscf import chain.

    ``core.nevpt2.impl.cas_dets_for2`` is the same construction; it is
    imported here through ``pyscfad``, which this unit test does not need.
    The bitmap convention is pinned by the test below, not by the comment.
    """
    from itertools import combinations

    nso = 2 * (ncore + nact)
    core_so = list(range(2 * ncore))
    act_so = list(range(2 * ncore, nso))
    dets = []
    for comb in combinations(act_so, nele):
        det = 0
        for so in core_so + list(comb):
            det |= 1 << so
        dets.append(det)
    return dets


# CAS(2e,2o): two electrons over spin orbitals 0..3.
_DETS = _dets_for(0, 2, 2)
_ALPHA_PAIR = 1 | (1 << 2)      # spin orbitals 0 and 2
_BETA_PAIR = 1 << 1 | (1 << 3)  # spin orbitals 1 and 3
_CLOSED = 1 | (1 << 1)          # spin orbitals 0 and 1


def test_a_high_spin_component_has_zero_entropy():
    """One spin per open shell: the four weights are [0, 1, 0, 0]."""
    s1 = manifold_entropy_from_ci(_vector(_DETS, _ALPHA_PAIR), _DETS,
                                  nact=2)
    assert np.allclose(s1, 0.0)


def test_merging_the_multiplet_restores_the_spin_isobaric_occupations():
    """The alpha and beta components together give one electron of
    unknown spin per orbital: weights [0, 1/2, 1/2, 0] and entropy ln 2.
    """
    vectors = np.column_stack([_vector(_DETS, _ALPHA_PAIR),
                               _vector(_DETS, _BETA_PAIR)])
    s1 = manifold_entropy_from_ci(vectors, _DETS, nact=2)
    assert np.allclose(s1, LN2)


def test_a_closed_shell_determinant_has_zero_entropy():
    s1 = manifold_entropy_from_ci(_vector(_DETS, _CLOSED), _DETS, nact=2)
    assert np.allclose(s1, 0.0)


def test_equal_weights_are_the_default():
    vectors = np.column_stack([_vector(_DETS, _ALPHA_PAIR),
                               _vector(_DETS, _BETA_PAIR)])
    default = manifold_entropy_from_ci(vectors, _DETS, nact=2)
    explicit = manifold_entropy_from_ci(vectors, _DETS, nact=2,
                                        weights=[0.5, 0.5])
    assert np.allclose(default, explicit)


def test_a_zero_weight_drops_the_state():
    vectors = np.column_stack([_vector(_DETS, _ALPHA_PAIR),
                               _vector(_DETS, _BETA_PAIR)])
    s1 = manifold_entropy_from_ci(vectors, _DETS, nact=2, weights=[1.0, 0.0])
    assert np.allclose(s1, 0.0)


def test_normalization_does_not_change_the_result():
    v = _vector(_DETS, _ALPHA_PAIR)
    plain = manifold_entropy_from_ci(v, _DETS, nact=2)
    scaled = manifold_entropy_from_ci(3.0 * v, _DETS, nact=2)
    assert np.allclose(plain, scaled)


def test_one_vector_reproduces_the_single_state_route():
    """The manifold route with one state is the existing extraction."""
    rng = np.random.default_rng(7)
    v = rng.normal(size=len(_DETS))
    v /= np.linalg.norm(v)
    assert np.allclose(manifold_entropy_from_ci(v, _DETS, nact=2),
                       entropy_from_ci(v, _DETS, nact=2))


def test_occupations_average_linearly():
    """The mixture is assembled on (n_a, n_b, p2), not on a density."""
    vectors = np.column_stack([_vector(_DETS, _ALPHA_PAIR),
                               _vector(_DETS, _BETA_PAIR)])
    n_a, n_b, p2 = manifold_occupations(vectors, _DETS, nact=2)
    assert np.allclose(n_a, 0.5)
    assert np.allclose(n_b, 0.5)
    assert np.allclose(p2, 0.0)


def test_negative_weights_are_refused():
    vectors = np.column_stack([_vector(_DETS, _ALPHA_PAIR),
                               _vector(_DETS, _BETA_PAIR)])
    with pytest.raises(ValueError, match='non-negative'):
        manifold_entropy_from_ci(vectors, _DETS, nact=2,
                                 weights=[1.5, -0.5])


def test_more_weights_than_states_are_refused():
    with pytest.raises(ValueError, match='cannot weight'):
        manifold_entropy_from_ci(_vector(_DETS, _ALPHA_PAIR), _DETS,
                                 nact=2, weights=[0.5, 0.5])


def test_zero_weights_are_refused():
    with pytest.raises(ValueError, match='sum to zero'):
        manifold_entropy_from_ci(_vector(_DETS, _ALPHA_PAIR), _DETS,
                                 nact=2, weights=[0.0])


def test_an_empty_column_is_refused():
    v = _vector(_DETS, _ALPHA_PAIR)
    with pytest.raises(ValueError, match='zero norm'):
        manifold_entropy_from_ci(np.column_stack([v, np.zeros_like(v)]),
                                 _DETS, nact=2)


def test_the_differentiable_route_agrees_with_the_numpy_route():
    """The jax sibling is bound to the numpy extraction on the same input."""
    jnp = pytest.importorskip('jax.numpy')
    import jax
    jax.config.update('jax_enable_x64', True)
    from diffmrc.driver.entropy import (
        manifold_single_orbital_entropy_from_ci,
    )

    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(len(_DETS), 2))
    vectors /= np.linalg.norm(vectors, axis=0)
    expected = manifold_entropy_from_ci(vectors, _DETS, nact=2)
    got = manifold_single_orbital_entropy_from_ci(vectors, _DETS, nact=2)
    assert np.allclose(np.asarray(got), expected, atol=1e-12)
