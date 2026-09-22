"""Assembling the end-to-end path out of the frozen kernel's pieces."""
import numpy as np
import pytest

from diffmrc.driver.kernel import ActiveSpace, partition_from_occupations


def test_partition_splits_around_the_active_set():
    occ = np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    space = partition_from_occupations(occ, [2, 3])
    assert space.core == [0, 1]
    assert space.active == [2, 3]
    assert space.virtual == [4, 5]
    assert space.n_elec == 2


def test_partition_reads_occupations_not_index_order():
    """The failure this exists to prevent.

    A multi-solution SCF can leave the singly occupied orbitals in a
    block the aufbau order does not predict. Assuming 'the first ncore
    orbitals are doubly occupied, the next nact are active' then selects
    empty orbitals: this project measured a CASCI 18.67 Ha above the
    ROHF it started from, which is impossible for an exact solver.
    """
    occ = np.array([2.0, 0.0, 2.0, 1.0, 0.0, 0.0])
    space = partition_from_occupations(occ, [1, 3])
    assert space.core == [0, 2]
    assert space.active == [1, 3]
    assert space.n_elec == 1


def test_partition_counts_electrons_from_the_occupations():
    occ = np.array([2.0, 2.0, 2.0, 0.0])
    space = partition_from_occupations(occ, [1, 2])
    assert space.n_elec == 4
    assert space.ncas == 2


def test_partition_rejects_an_index_outside_the_reference():
    occ = np.array([2.0, 0.0])
    with pytest.raises(ValueError):
        partition_from_occupations(occ, [2])


def test_partition_deduplicates_the_active_set():
    occ = np.array([2.0, 1.0, 0.0])
    space = partition_from_occupations(occ, [1, 1])
    assert space.active == [1]
