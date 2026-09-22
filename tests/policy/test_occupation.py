"""Partitioning orbitals by their occupation."""
import numpy as np
import pytest

from diffmrc.policy.occupation import (
    partition_blocks,
    select_valence,
    split_by_occupation,
)


def test_split_reads_a_closed_shell():
    occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0])
    p = split_by_occupation(occ)
    assert p.doubly == [0, 1, 2]
    assert p.singly == []
    assert p.empty == [3, 4]


def test_split_reads_a_high_spin_atom():
    """Seven singly occupied orbitals above a closed shell."""
    occ = np.array([2.0] * 28 + [1.0] * 7 + [0.0] * 5)
    p = split_by_occupation(occ)
    assert len(p.doubly) == 28
    assert p.singly == list(range(28, 35))
    assert len(p.empty) == 5


def test_split_tolerates_near_integer_occupations():
    """A converged SCF leaves occupations a few digits off the integers."""
    occ = np.array([2.0000001, 1.9999998, 1.0000002, 0.0000001])
    p = split_by_occupation(occ)
    assert p.doubly == [0, 1]
    assert p.singly == [2]
    assert p.empty == [3]


def test_split_rejects_non_1d_input():
    with pytest.raises(ValueError):
        split_by_occupation(np.zeros((2, 2)))


# The two fixtures below are the occupation structures measured on the F
# and O atoms in the ANO basis; the t176 script reports [2, 3, 4] for
# both. They deliberately include a core, so that "topmost doubly
# occupied" is a real instruction rather than a synonym for "all the
# doubly occupied orbitals".
F_OCC = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0])
O_OCC = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 0.0])


def test_select_valence_of_fluorine():
    """1s2 2s2 2p_a2 2p_b2 2p_c1: one singly plus the topmost two doubly."""
    assert select_valence(F_OCC, 3) == [2, 3, 4]


def test_select_valence_of_oxygen():
    """1s2 2s2 2p_a2 2p_b1 2p_c1: two singly plus the topmost doubly."""
    assert select_valence(O_OCC, 3) == [2, 3, 4]


def test_select_valence_does_not_dip_into_the_core():
    """The point of "topmost": the 1s orbital must not be selected.

    A naive "take the first doubly occupied orbitals" returns [0, 1, 2]
    here, pulling the 1s core into the active space.
    """
    picked = select_valence(F_OCC, 3)
    assert 0 not in picked
    assert 1 not in picked


def test_select_valence_sorts_its_result():
    """The returned indices are ascending, whatever the occupation order."""
    occ = np.array([1.0, 2.0, 1.0, 2.0])
    picked = select_valence(occ, 3)
    assert picked == sorted(picked)


def test_select_valence_returns_none_when_impossible():
    """Not enough doubly occupied orbitals to fill the request."""
    occ = np.array([2.0, 1.0, 1.0, 1.0, 0.0])
    assert select_valence(occ, 5) is None


def test_select_valence_rejects_absurd_request():
    assert select_valence(np.array([2.0, 2.0]), 0) is None
    assert select_valence(np.array([2.0, 2.0]), -1) is None


def test_partition_blocks_keeps_core_and_active_disjoint():
    """An active orbital must never also be counted as core.

    Counting it twice is what makes the electron count disagree with the
    nuclear charge and trips the downstream sanity check.
    """
    occ = np.array([2.0, 2.0, 2.0, 1.0, 0.0])
    b = partition_blocks(occ, [2, 3])
    assert b.active == [2, 3]
    assert b.core == [0, 1]
    assert b.virtual == [4]
    assert set(b.core) & set(b.active) == set()


def test_partition_blocks_counts_active_electrons():
    occ = np.array([2.0, 2.0, 1.0, 1.0, 0.0])
    b = partition_blocks(occ, [2, 3])
    assert b.n_elec_active == 2
    assert b.ncore == 2


def test_partition_blocks_counts_doubly_occupied_active_as_two():
    occ = np.array([2.0, 2.0, 2.0, 1.0, 0.0])
    b = partition_blocks(occ, [1, 2, 3])
    assert b.n_elec_active == 5


def test_partition_blocks_covers_every_orbital_exactly_once():
    occ = np.array([2.0, 1.0, 1.0, 0.0, 0.0])
    b = partition_blocks(occ, [1, 2])
    everything = sorted(b.core + b.active + b.virtual)
    assert everything == list(range(len(occ)))


def test_partition_blocks_rejects_out_of_range_active():
    with pytest.raises(ValueError):
        partition_blocks(np.zeros(3), [0, 5])


def test_partition_blocks_accepts_an_empty_active_space():
    occ = np.array([2.0, 2.0, 0.0])
    b = partition_blocks(occ, [])
    assert b.active == []
    assert b.n_elec_active == 0
    assert sorted(b.core + b.virtual) == [0, 1, 2]
