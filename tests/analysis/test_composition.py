"""Orbital composition analysis by element and angular momentum."""
import numpy as np
import pytest

from diffmrc.analysis.composition import (
    ao_mask,
    overlap_weighted_population,
)

LABELS = [
    '0 O 1s', '1 O 2s', '2 O 2px', '3 O 2py', '4 O 2pz',
    '5 H 1s', '6 H 1s',
]


def test_ao_mask_selects_by_element_and_shell():
    mask = ao_mask(LABELS, 'O', 'p')
    assert mask.tolist() == [False, False, True, True, True, False, False]


def test_ao_mask_selects_hydrogen_s():
    mask = ao_mask(LABELS, 'H', 's')
    assert mask.tolist() == [False, False, False, False, False, True, True]


def test_ao_mask_matches_element_exactly():
    """A substring match would confuse O with Os or Og."""
    labels = ['0 O 2p', '1 Os 5d']
    mask = ao_mask(labels, 'O', 'p')
    assert mask.tolist() == [True, False]


def test_ao_mask_on_absent_shell_is_all_false():
    mask = ao_mask(LABELS, 'O', 'd')
    assert not mask.any()


def test_ao_mask_reads_high_angular_momentum_shells():
    """Lanthanide ANO sets label their f shells 6f, 7f, 8f -- no 4f.

    Matching the angular-momentum letter rather than the principal quantum
    number is what makes this work.
    """
    labels = ['0 Eu 6f', '1 Eu 7f', '2 Eu 5d', '3 Eu 6s']
    assert ao_mask(labels, 'Eu', 'f').tolist() == [True, True, False, False]
    assert ao_mask(labels, 'Eu', 'd').tolist() == [False, False, True, False]


def test_ao_mask_handles_general_contracted_labels():
    """PySCF labels contracted shells such as '4d' and '5d' alike."""
    labels = ['0 Lu 4d', '1 Lu 5d', '2 Lu 4f']
    assert ao_mask(labels, 'Lu', 'd').tolist() == [True, True, False]
    assert ao_mask(labels, 'Lu', 'f').tolist() == [False, False, True]


def test_ao_mask_ignores_unparseable_labels():
    mask = ao_mask(['garbage', '', '3 O'], 'O', 'p')
    assert not mask.any()


def test_ao_mask_returns_a_boolean_array():
    mask = ao_mask(LABELS, 'O', 'p')
    assert mask.dtype == bool
    assert mask.shape == (len(LABELS),)


def test_overlap_weighted_population_of_an_atomic_orbital():
    """A pure 2pz orbital has unit weighted population on that AO."""
    C = np.zeros((7, 1))
    C[4, 0] = 1.0
    S = np.eye(7)
    mask = ao_mask(LABELS, 'O', 'p')
    w = overlap_weighted_population(C, S, mask)
    assert w.shape == (1,)
    assert w[0] == 1.0


def test_overlap_weighted_population_sums_over_masked_aos():
    C = np.zeros((7, 1))
    C[2, 0] = np.sqrt(0.5)
    C[3, 0] = np.sqrt(0.5)
    S = np.eye(7)
    mask = ao_mask(LABELS, 'O', 'p')
    w = overlap_weighted_population(C, S, mask)
    assert w[0] == pytest.approx(1.0, abs=1e-12)


def test_overlap_weighted_population_vanishes_outside_the_mask():
    C = np.zeros((7, 1))
    C[0, 0] = 1.0          # O 1s, not in the p mask
    S = np.eye(7)
    mask = ao_mask(LABELS, 'O', 'p')
    w = overlap_weighted_population(C, S, mask)
    assert w[0] == 0.0


def test_overlap_weighted_population_handles_several_orbitals():
    C = np.eye(7)[:, :3]
    S = np.eye(7)
    mask = ao_mask(LABELS, 'O', 'p')
    w = overlap_weighted_population(C, S, mask)
    assert w.shape == (3,)
    assert np.allclose(w, [0.0, 0.0, 1.0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        overlap_weighted_population(np.zeros((7, 2)), np.eye(6),
                                    np.ones(7, dtype=bool))


def test_non_square_overlap_raises():
    with pytest.raises(ValueError):
        overlap_weighted_population(np.zeros((7, 2)), np.eye(7)[:6],
                                    np.ones(7, dtype=bool))


def test_wrong_mask_length_raises():
    with pytest.raises(ValueError):
        overlap_weighted_population(np.zeros((7, 2)), np.eye(7),
                                    np.ones(6, dtype=bool))


def test_mask_is_normalized_to_boolean():
    """An integer mask is accepted and treated as a selector."""
    C = np.zeros((7, 1))
    C[4, 0] = 1.0
    S = np.eye(7)
    mask = np.zeros(7, dtype=int)
    mask[4] = 1
    assert overlap_weighted_population(C, S, mask)[0] == 1.0
