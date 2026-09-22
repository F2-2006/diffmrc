"""Localization and the overlapping split of a large space."""
import numpy as np
import pytest

from diffmrc.policy.localize import localize_orbitals, overlapping_split


def test_split_covers_the_whole_space():
    """Every orbital must appear in at least one subset.

    The subsets overlap by construction, so an orbital may appear in
    several; what must not happen is an orbital appearing in none,
    because its entanglement would then be lost from the spectrum.
    """
    space = list(range(18))
    subsets = overlapping_split(space, 6)
    covered = {i for s in subsets for i in s}
    assert covered == set(space)


def test_split_of_the_documented_example():
    """A CAS(18,18) into nine CAS(6,6), as in the reference figure."""
    subsets = overlapping_split(list(range(18)), 6)
    assert len(subsets) == 9
    assert all(len(s) == 6 for s in subsets)


def test_split_count_follows_the_quadratic_scaling():
    """Eq. (6) of the reference: N = L**2 / l_max**2."""
    subsets = overlapping_split(list(range(12)), 4)
    assert len(subsets) == 9          # (12 / 4) ** 2


def test_split_pairs_occupied_with_virtual():
    """Each subset holds half occupied and half virtual orbitals.

    The first half of the space is the occupied block; a subset must draw
    from both halves, which is what makes the correlation between an
    occupied and a virtual orbital visible.
    """
    space = list(range(18))
    occupied = set(space[:9])
    for subset in overlapping_split(space, 6):
        assert len(set(subset) & occupied) == 3
        assert len(set(subset) - occupied) == 3


def test_split_of_a_space_already_small_enough_is_a_single_subset():
    assert overlapping_split(list(range(5)), 10) == [list(range(5))]


def test_split_of_an_empty_space_is_empty():
    assert overlapping_split([], 6) == []


def test_split_rejects_a_nonpositive_l_max():
    with pytest.raises(ValueError):
        overlapping_split(list(range(6)), 0)


def test_split_rejects_an_odd_l_max():
    """An odd limit cannot be halved into occupied and virtual parts."""
    with pytest.raises(ValueError):
        overlapping_split(list(range(6)), 3)


def test_split_rejects_an_inexact_fit():
    with pytest.raises(ValueError):
        overlapping_split(list(range(7)), 4)


def test_subsets_are_ascending_and_within_the_space():
    space = list(range(18))
    for s in overlapping_split(space, 6):
        assert s == sorted(s)
        assert set(s) <= set(space)


def test_split_accepts_an_unordered_input():
    """The caller's ordering does not matter; the function sorts first."""
    shuffled = [7, 0, 5, 3, 1, 6, 4, 2]
    subsets = overlapping_split(shuffled, 4)
    assert sorted({i for s in subsets for i in s}) == list(range(8))


def test_split_of_eight_orbitals_gives_four_subsets():
    subsets = overlapping_split(list(range(8)), 4)
    assert len(subsets) == 4
    assert all(len(s) == 4 for s in subsets)


def test_localize_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        localize_orbitals(None, np.eye(2), method='nonsense')
