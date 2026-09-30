"""Orbital identity rules on synthetic coefficient matrices.

The fixtures are the three frontier patterns the M5-C batches met: a
delocalized pi (two carbons on one axis -- which must not read as one
"centre"), an atom-centred lone pair, and a degenerate linear two-pi
frontier whose naive window crosses components.  The AO labels are the
kernel's format, but the numbers are hand-built.
"""
import numpy as np

from diffmrc.analysis.orbitals import (classify_orbital, degenerate_pairs,
                                       frontier_report, heavy_atoms,
                                       p_axis_profile)

# AO layout: C0 (px,py,pz), C1 (px,py,pz), O0 (px,py,pz)
_LABELS = ["0 C 2px", "0 C 2py", "0 C 2pz",
           "1 C 2px", "1 C 2py", "1 C 2pz",
           "0 O 2px", "0 O 2py", "0 O 2pz"]


def _C(columns):
    C = np.zeros((len(_LABELS), len(columns)))
    for k, col in enumerate(columns):
        for i, v in col.items():
            C[i, k] = v
    return C


def test_heavy_atoms_lists_atom_keys_not_elements():
    assert heavy_atoms(_LABELS) == ["0 C", "1 C", "0 O"]


def test_a_delocalised_pi_is_single_axis_and_not_centred():
    pa = 0.5
    C = _C([{0: pa, 3: pa}])          # C0 px + C1 px
    prof = p_axis_profile(C, _LABELS, 0, heavy_atoms(_LABELS))
    ident = classify_orbital(prof)
    assert ident['axis'] == 'x'
    assert ident['kind'] == 'pi'
    assert ident['centre'] is None


def test_a_lone_pair_is_single_axis_and_centred():
    C = _C([{8: 1.0}])                # O0 pz alone
    ident = classify_orbital(
        p_axis_profile(C, _LABELS, 0, heavy_atoms(_LABELS)))
    assert ident['axis'] == 'z'
    assert ident['kind'] == 'lone'
    assert ident['centre'] == '0 O'


def test_a_sigma_is_not_a_frontier_member():
    C = _C([{0: 0.5, 3: 0.5, 5: 0.5}])   # mixed axes
    ident = classify_orbital(
        p_axis_profile(C, _LABELS, 0, heavy_atoms(_LABELS)))
    assert ident['kind'] == 'other'


def test_degenerate_pairs_are_per_side_of_the_border():
    e = np.array([-1.0, -0.5, -0.5, 0.2, 0.2, 0.9])
    # nocc=3: both the occupied pair and the virtual pair survive;
    # nocc=4: the (3, 4) pair straddles the border and is dropped.
    assert degenerate_pairs(e, nocc=3) == [(1, 2), (3, 4)]
    assert degenerate_pairs(e, nocc=4) == [(1, 2)]


def test_a_linear_two_pi_frontier_suggests_the_three_orbital_window():
    """HOMO/HOMO-1 degenerate on x and y, LUMO on x: the naive (HOMO,
    LUMO) pair crosses components, and the suggestion is (nocc-2, 3)."""
    C = _C([{0: .5, 3: .5},              # MO0 (HOMO-1): x pi
            {1: .5, 4: .5},              # MO1 (HOMO):   y pi
            {0: .5, 3: -.5},             # MO2 (LUMO):   x pi*
            {1: .5, 4: -.5}])            # MO3:          y pi*
    e = np.array([-0.5, -0.5, 0.1, 0.1])
    rep = frontier_report(C, _LABELS, e, nocc=2, atoms=heavy_atoms(_LABELS),
                          target='ppi', nelectron=8)
    assert rep['homo_in_degenerate_pair']
    assert rep['cross_component']
    assert rep['suggested_window'] == {'ncore': 0, 'nact': 3, 'nele': 8}


def test_an_amide_lone_pair_below_the_pi_frontier_is_found():
    """HOMO is a conjugated pi; the npi target must reach the lone pair
    two orbitals below, widening the window to (ncore=lone_index)."""
    C = _C([{6: 1.0},                     # MO0: O lone pair (z)
            {1: .5, 4: .5},               # MO1 (HOMO): pi (y)
            {1: .5, 4: -.5}])             # MO2 (LUMO): pi*
    e = np.array([-0.6, -0.45, 0.1])
    rep = frontier_report(C, _LABELS, e, nocc=2, atoms=heavy_atoms(_LABELS),
                          target='npi', nelectron=4)
    assert rep['lone_below_homo'] == 0
    assert rep['suggested_window'] == {'ncore': 0, 'nact': 3, 'nele': 4}


def test_a_lone_pair_homo_keeps_the_two_orbital_window():
    C = _C([{1: .5, 4: .5},               # MO0: pi
            {6: 1.0},                     # MO1 (HOMO): O lone pair
            {1: .5, 4: -.5}])             # MO2 (LUMO): pi*
    e = np.array([-0.5, -0.45, 0.1])
    rep = frontier_report(C, _LABELS, e, nocc=2, atoms=heavy_atoms(_LABELS),
                          target='npi', nelectron=4)
    assert rep['homo']['kind'] == 'lone'
    assert rep['suggested_window'] == {'ncore': 1, 'nact': 2, 'nele': 2}
