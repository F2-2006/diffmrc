"""Entropy-driven selection: plateau choice, MI growth, closure reports."""
import numpy as np
import pytest

from diffmrc.policy.entropy_selection import (
    MI_FRACTION_DEFAULT,
    mi_closure,
    mi_selection,
    plateau_selection,
)


# A flat top block over a weak tail: the canonical autoCAS picture.
FLAT = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.02, 0.01, 0.005])


def test_plateau_selection_finds_the_flat_block():
    result = plateau_selection(FLAT)
    assert result.active == [0, 1, 2, 3, 4, 5, 6]
    assert result.plateau_height == 7
    assert any('plateau height 7' in line for line in result.evidence)


def test_plateau_selection_selects_by_entropy_not_by_index():
    shifted = np.roll(FLAT, 1)     # the weak orbital moves to index 0
    result = plateau_selection(shifted)
    assert result.active == [1, 2, 3, 4, 5, 6, 7]


def test_plateau_selection_falls_back_to_the_weak_correlation_line():
    # Steps of 8.5% plus a two-level tail (0.06, 0.015): every step
    # between consecutive counts is narrower than the ten-step plateau
    # window, so detect_plateaus finds nothing, and the 0.015 tail sits
    # below the 2% line while the 0.06 does not.
    staircase = np.append(1.0 - 0.085 * np.arange(11), [0.06, 0.015])
    result = plateau_selection(staircase)
    assert result.plateau_height == 0
    assert any('no plateau' in line for line in result.evidence)
    assert any('weak-correlation line' in line for line in result.evidence)
    assert result.active == list(range(12))    # the 0.015 tail is dropped


def test_plateau_selection_reads_a_uniform_spectrum_as_all_orbitals():
    # Every entropy comparable (the all-orbitals count is itself the
    # plateau): the selection is the whole space, which is the honest
    # answer -- nothing is weakly correlated enough to drop.
    uniform = np.linspace(0.5, 0.45, 8)
    result = plateau_selection(uniform)
    assert result.plateau_height == 8
    assert result.active == list(range(8))


def test_plateau_selection_refuses_a_single_reference_spectrum():
    result = plateau_selection(np.array([0.1, 0.05, 0.01]))
    assert result.active == []
    assert any('multireference criterion' in line
               for line in result.evidence)


def test_plateau_selection_refuses_an_empty_spectrum():
    assert plateau_selection(np.array([])).active == []


def test_plateau_selection_rejects_non_finite_values():
    with pytest.raises(ValueError, match='non-finite'):
        plateau_selection(np.array([0.5, np.nan]))


def _pair_matrix(n, pairs):
    I = np.zeros((n, n))
    for (i, j), v in pairs.items():
        I[i, j] = I[j, i] = v
    return I


def test_mi_selection_grows_along_the_chain():
    s1 = np.array([0.8, 0.5, 0.3, 0.2])
    I = _pair_matrix(4, {(0, 1): 0.5, (1, 2): 0.1})
    result = mi_selection(s1, I)
    # seed 0 -> 1 (0.5) -> 2 (0.1, still above 5% of 0.5 = 0.025) -> stop
    assert result.active == [0, 1, 2]
    assert any('added 1' in line for line in result.evidence)
    assert any('added 2' in line for line in result.evidence)


def test_mi_selection_stops_at_a_stricter_fraction():
    s1 = np.array([0.8, 0.5, 0.3, 0.2])
    I = _pair_matrix(4, {(0, 1): 0.5, (1, 2): 0.1})
    result = mi_selection(s1, I, mi_fraction=0.3)
    # threshold 0.15: the 0.1 link to orbital 2 does not qualify
    assert result.active == [0, 1]
    assert any('threshold 0.1500' in line for line in result.evidence)


def test_mi_selection_honours_an_explicit_seed():
    s1 = np.array([0.8, 0.5, 0.3, 0.2])
    I = _pair_matrix(4, {(0, 1): 0.5, (2, 3): 0.4})
    result = mi_selection(s1, I, seed=2)
    assert result.active == [2, 3]


def test_mi_selection_refuses_single_reference_and_bad_seed():
    weak = np.array([0.1, 0.05, 0.01, 0.001])
    I = _pair_matrix(4, {(0, 1): 0.5})
    assert mi_selection(weak, I).active == []
    strong = np.array([0.8, 0.5, 0.3, 0.2])
    assert mi_selection(strong, I, seed=99).active == []


def test_mi_selection_rejects_an_asymmetric_matrix():
    s1 = np.array([0.8, 0.5])
    I = np.array([[0.0, 0.5], [0.4, 0.0]])
    with pytest.raises(ValueError, match='not symmetric'):
        mi_selection(s1, I)


def test_mi_closure_reports_a_leak_at_the_boundary():
    s1 = np.array([0.8, 0.5, 0.3, 0.2])
    I = _pair_matrix(4, {(0, 1): 0.5, (1, 2): 0.4})
    report = mi_closure(s1, I, [0, 1])
    assert report.leaks == [2]
    assert report.isolated == []
    assert any('leak' in line for line in report.evidence)


def test_mi_closure_reports_isolated_members():
    s1 = np.array([0.8, 0.5, 0.3, 0.2])
    I = _pair_matrix(4, {(0, 1): 0.5, (2, 3): 0.4})
    report = mi_closure(s1, I, [0, 3])
    assert 3 in report.isolated          # no link to 0
    assert 3 in report.leaks or 1 in report.leaks  # 1 entangles with 0


def test_mi_closure_is_scale_invariant():
    s1 = np.array([0.8, 0.5, 0.3, 0.2])
    I = _pair_matrix(4, {(0, 1): 0.5, (1, 2): 0.4})
    report = mi_closure(s1, I, [0, 1])
    scaled = mi_closure(s1, I * 10.0, [0, 1])
    assert report.leaks == scaled.leaks
    assert report.isolated == scaled.isolated


def test_mi_closure_with_a_single_member_skips_the_partner_question():
    s1 = np.array([0.8, 0.5, 0.3])
    I = _pair_matrix(3, {(0, 1): 0.5})
    report = mi_closure(s1, I, [0])
    assert report.isolated == []         # meaningless for one member
    assert report.leaks == [1]


def test_mi_closure_with_an_empty_candidate():
    s1 = np.array([0.8, 0.5])
    I = _pair_matrix(2, {(0, 1): 0.5})
    report = mi_closure(s1, I, [])
    assert report.leaks == [] and report.isolated == []
    assert any('empty candidate' in line for line in report.evidence)


def test_mi_closure_rejects_out_of_range_members():
    s1 = np.array([0.8, 0.5])
    I = _pair_matrix(2, {(0, 1): 0.5})
    with pytest.raises(ValueError, match='outside'):
        mi_closure(s1, I, [0, 5])


def test_mi_fraction_default_is_documented():
    # the module docstring pins this as a package choice, not a literature
    # value; the constant is the single source
    assert 0.0 < MI_FRACTION_DEFAULT < 1.0
