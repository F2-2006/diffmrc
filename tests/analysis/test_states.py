"""Spin recognition and physical-state grouping from a root table.

The convention under test follows ``analysis/ci.py``: a determinant is an
integer bitmap, bit ``s`` is spin orbital ``s``, the alpha orbital of
active orbital ``j`` at bit ``2*(ncore+j)`` and the beta at the next bit.

The fixtures are the textbook roots of a two-electron, two-orbital space:
the closed shell and the open-shell singlet carry <S^2> = 0, the three
triplet components carry <S^2> = 2, and these values are the ground truth
the implementation is pinned against -- no reference to the kernel.
"""
import numpy as np
import pytest

from diffmrc.analysis.states import (det_abs_sz, select_groups,
                                     spin_square, state_groups)


def _det(occupied):
    d = 0
    for so in occupied:
        d |= 1 << so
    return d


# (2e, 2o), ncore = 0: all six determinants of the space.
_DETS = [_det([0, 1]),   # |1a 1b>
         _det([0, 2]),   # |1a 2a>
         _det([0, 3]),   # |1a 2b>
         _det([1, 2]),   # |1b 2a>
         _det([1, 3]),   # |1b 2b>
         _det([2, 3])]   # |2a 2b>


def _vec(entries, n=len(_DETS)):
    c = np.zeros(n)
    for idx, val in entries:
        c[idx] = val
    return c


def test_det_abs_sz_is_half_the_spin_excess():
    assert det_abs_sz(_det([0, 1])) == 0.0
    assert det_abs_sz(_det([0, 2])) == 1.0
    assert det_abs_sz(_det([1, 3])) == 1.0
    assert det_abs_sz(_det([0, 3])) == 0.0
    assert det_abs_sz(_det([0, 1, 2, 3])) == 0.0
    # A closed core contributes equally to both spins and cancels.
    assert det_abs_sz(_det([0, 1, 2, 3, 4, 5])) == 0.0


def test_spin_square_textbook_roots():
    s = 1.0 / np.sqrt(2.0)
    # closed shell and open-shell singlet
    assert spin_square(_vec([(0, 1.0)]), _DETS) == pytest.approx(0.0, abs=1e-12)
    assert spin_square(_vec([(2, s), (3, -s)]), _DETS) == \
        pytest.approx(0.0, abs=1e-12)
    # the three triplet components, M = +1, 0, -1
    assert spin_square(_vec([(1, 1.0)]), _DETS) == pytest.approx(2.0, abs=1e-12)
    assert spin_square(_vec([(2, s), (3, s)]), _DETS) == \
        pytest.approx(2.0, abs=1e-12)
    assert spin_square(_vec([(4, 1.0)]), _DETS) == pytest.approx(2.0, abs=1e-12)
    # a closed-shell mixture stays a singlet
    assert spin_square(_vec([(0, s), (5, s)]), _DETS) == \
        pytest.approx(0.0, abs=1e-12)


def test_spin_square_sees_contamination():
    """Half singlet, half M = +1 triplet: <S^2> averages to 1."""
    s = 1.0 / np.sqrt(2.0)
    assert spin_square(_vec([(0, s), (1, s)]), _DETS) == \
        pytest.approx(1.0, abs=1e-12)


def test_spin_square_phase_of_the_open_shell_singlet_does_not_matter():
    s = 1.0 / np.sqrt(2.0)
    c = _vec([(2, s), (3, -s)])
    assert spin_square(c, _DETS) == pytest.approx(0.0, abs=1e-12)
    assert spin_square(-c, _DETS) == pytest.approx(0.0, abs=1e-12)


def _triplet_components():
    s = 1.0 / np.sqrt(2.0)
    return {
        '+1': _vec([(1, 1.0)]),
        '0': _vec([(2, s), (3, s)]),
        '-1': _vec([(4, 1.0)]),
    }


def test_state_groups_merge_degenerate_spin_components():
    comps = _triplet_components()
    energies = [0.0, 1.0, 1.0, 1.0, 2.0]
    civecs = np.stack([_vec([(0, 1.0)]), comps['+1'], comps['0'],
                       comps['-1'], _vec([(5, 1.0)])], axis=1)
    rows, groups = state_groups(energies, civecs, _DETS)
    assert len(rows) == 5
    assert len(groups) == 3
    g0, g1, g2 = groups
    assert g0['roots'] == [0]
    assert g0['multiplicity'] == 1
    assert g0['E'] == pytest.approx(0.0, abs=1e-12)
    assert sorted(g1['roots']) == [1, 2, 3]
    assert g1['multiplicity'] == 3
    assert g2['roots'] == [4]
    assert g2['multiplicity'] == 1
    # every row carries its own spin diagnostics; the group merges them
    assert rows[0]['multiplicity'] == 1
    assert rows[2]['multiplicity'] == 3
    assert rows[2]['szs'] == [0.0]
    assert g1['szs'] == [0.0, 1.0]


def test_state_groups_flag_a_spin_impure_degeneracy():
    """Two degenerate roots of different S are not one physical state."""
    energies = [1.0, 1.0]
    civecs = np.stack([_vec([(0, 1.0)]), _vec([(1, 1.0)])], axis=1)
    _, groups = state_groups(energies, civecs, _DETS)
    assert len(groups) == 1
    assert groups[0]['multiplicity'] is None
    assert groups[0]['s2'] is None


def test_state_groups_pick_the_usable_representative():
    """A component whose observables are undefined hands the group over."""
    comps = _triplet_components()
    energies = [1.0, 1.0, 1.0]
    civecs = np.stack([comps['+1'], comps['0'], comps['-1']], axis=1)
    usable = [False, True, False]
    _, groups = state_groups(energies, civecs, _DETS, usable=usable)
    assert groups[0]['representative'] == 1
    assert groups[0]['usable'] is True
    _, groups2 = state_groups(energies, civecs, _DETS,
                              usable=[False, False, False])
    assert groups2[0]['usable'] is False
    assert groups2[0]['representative'] == 0  # falls back to lowest


def test_state_groups_respect_the_energy_tolerance():
    c = [_vec([(0, 1.0)]), _vec([(5, 1.0)])]
    civecs = np.stack(c, axis=1)
    near = [1.0, 1.0 + 1e-12]
    _, merged = state_groups(near, civecs, _DETS, tol=1e-9)
    assert len(merged) == 1
    _, split = state_groups(near, civecs, _DETS, tol=1e-14)
    assert len(split) == 2


def test_select_groups_filters_by_multiplicity_and_energy():
    comps = _triplet_components()
    energies = [0.0, 1.0, 1.0, 1.0, 2.0]
    civecs = np.stack([_vec([(0, 1.0)]), comps['+1'], comps['0'],
                       comps['-1'], _vec([(5, 1.0)])], axis=1)
    _, groups = state_groups(energies, civecs, _DETS)
    singlets = select_groups(groups, multiplicity=1, n=2)
    assert [g['E'] for g in singlets] == [0.0, 2.0]
    triplets = select_groups(groups, multiplicity=3)
    assert triplets[0]['multiplicity'] == 3
    assert len(triplets) == 1
    assert select_groups(groups, multiplicity=5) == []
    assert select_groups(groups, multiplicity=1, n=1)[0]['E'] == 0.0
