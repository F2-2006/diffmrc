"""The perturbation-layer convergence criterion on surfaces with known answers."""
import numpy as np
import pytest

from diffmrc.core.perturbation_convergence import (
    DEFAULT_CURVATURE_THRESHOLD,
    DEFAULT_E2_TOLERANCE,
    BranchPoint,
    assess_perturbation_convergence,
    branch_spread,
    first_order_bound,
    split_spectrum,
)


def _diag(values):
    return np.diag(np.asarray(values, dtype=float))


def test_accepts_a_converged_quadratic_minimum():
    """A minimum with a tiny residual gradient and no flat mode."""
    H = _diag([1.0, 2.0, 3.0])
    s = np.array([0.1, -0.2, 0.05])
    g = np.array([1e-9, -2e-9, 1e-9])
    report = assess_perturbation_convergence(g, s, H)
    assert report.value == 'accept'
    assert report.certified
    assert report.n_negative == 0
    assert report.first_order_bound < DEFAULT_E2_TOLERANCE
    assert report.required_gradient > 1e-9


def test_rejects_a_saddle_even_with_a_zero_gradient():
    """The known bad point: the linear model has the wrong sign there."""
    H = _diag([-0.1, 1.0, 2.0])
    g = np.zeros(3)
    s = np.array([0.1, 0.0, 0.0])
    report = assess_perturbation_convergence(g, s, H)
    assert report.value == 'reject'
    assert not report.certified
    assert report.n_negative == 1
    assert any('saddle' in reason for reason in report.reasons)


def test_rejects_a_flat_direction_that_carries_e2():
    """The OH CAS(3,2) failure mode: the energy is flat, e2 is not."""
    H = _diag([0.0, 1.0, 1.0])
    g = np.zeros(3)
    s = np.array([0.01, 0.0, 0.0])
    report = assess_perturbation_convergence(g, s, H)
    assert report.value == 'reject'
    assert report.n_soft == 1
    assert any('flat direction' in reason for reason in report.reasons)


def test_accepts_a_flat_direction_without_e2_coupling():
    """The active-active redundancy: the correction is invariant along it."""
    H = _diag([0.0, 1.0, 1.0])
    g = np.zeros(3)
    s = np.array([0.0, 0.1, 0.0])
    report = assess_perturbation_convergence(g, s, H)
    assert report.value == 'accept'
    assert report.n_soft == 1
    assert report.soft_e2_coupling == pytest.approx(0.0, abs=1e-15)


def test_a_flat_coupling_below_the_linear_floor_needs_a_probe():
    """Below the floor and without a finite probe the second order is open."""
    H = _diag([0.0, 1.0, 1.0])
    g = np.zeros(3)
    s = np.array([1e-6, 0.0, 0.0])
    report = assess_perturbation_convergence(g, s, H, null_angle=0.1)
    assert report.value == 'undetermined'
    assert any('null_probe_max' in warning for warning in report.warnings)


def test_a_finite_probe_above_the_tolerance_rejects():
    H = _diag([0.0, 1.0, 1.0])
    g = np.zeros(3)
    s = np.array([1e-9, 0.0, 0.0])
    report = assess_perturbation_convergence(g, s, H, null_probe_max=1e-4)
    assert report.value == 'reject'
    assert report.null_probe_max == pytest.approx(1e-4)


def test_a_finite_probe_below_the_tolerance_accepts():
    H = _diag([0.0, 1.0, 1.0])
    g = np.zeros(3)
    s = np.array([1e-9, 0.0, 0.0])
    report = assess_perturbation_convergence(g, s, H, null_probe_max=1e-9)
    assert report.value == 'accept'


def test_a_small_probe_does_not_hide_a_large_linear_coupling():
    """A probe along one direction can miss the coupling direction."""
    H = _diag([0.0, 1.0, 1.0])
    g = np.zeros(3)
    s = np.array([0.01, 0.0, 0.0])
    report = assess_perturbation_convergence(g, s, H, null_probe_max=1e-9)
    assert report.value == 'reject'
    assert any('maximum-coupling' in warning for warning in report.warnings)


def test_the_first_order_bound_bounds_the_exact_quadratic_error():
    """E = t.H.t/2, e2 = s.t: the exact error is -s.H^-1.g."""
    H = _diag([0.5, 2.0, 4.0])
    s = np.array([0.3, -0.1, 0.2])
    distance = np.array([2e-6, -1e-6, 3e-6])
    g = H @ distance
    exact = float(s @ distance)
    out = first_order_bound(g, s, H)
    assert out['prediction'] == pytest.approx(-exact, rel=1e-12, abs=1e-18)
    assert out['bound'] >= abs(exact) - 1e-18


def test_the_bound_is_tight_when_s_is_parallel_to_the_residual():
    H = _diag([0.5, 2.0])
    distance = np.array([1e-6, 0.0])
    g = H @ distance
    s = np.array([0.3, 0.0])
    out = first_order_bound(g, s, H)
    assert out['bound'] == pytest.approx(abs(float(s @ distance)))


def test_an_unconverged_gradient_is_rejected_with_the_required_value():
    H = _diag([0.5, 2.0])
    g = np.array([1e-3, 0.0])
    s = np.array([0.3, 0.0])
    report = assess_perturbation_convergence(g, s, H, e2_tolerance=1e-6)
    assert report.value == 'reject'
    assert report.required_gradient == pytest.approx(
        1e-6 / (0.3 / 0.5), rel=1e-12)
    assert report.gradient_norm > report.required_gradient


def test_a_same_branch_repeat_difference_above_the_tolerance_rejects():
    H = _diag([1.0, 2.0])
    g = np.zeros(2)
    s = np.array([0.1, 0.1])
    report = assess_perturbation_convergence(
        g, s, H, repeat_delta_e2=1e-5, repeat_overlap=0.99)
    assert report.value == 'reject'
    assert any('repeat difference' in reason for reason in report.reasons)


def test_a_different_branch_repeat_difference_is_not_a_convergence_error():
    """Low overlap means the difference is branch spread, not drift."""
    H = _diag([1.0, 2.0])
    g = np.zeros(2)
    s = np.array([0.1, 0.1])
    report = assess_perturbation_convergence(
        g, s, H, repeat_delta_e2=1.8e-2, repeat_overlap=0.2)
    assert report.value == 'accept'
    assert any('branch spread' in warning for warning in report.warnings)


def test_a_non_finite_input_is_undetermined():
    report = assess_perturbation_convergence(
        np.array([np.nan, 0.0]), np.zeros(2), np.eye(2))
    assert report.value == 'undetermined'
    assert not report.certified


def test_the_curvature_threshold_sits_between_the_measured_scales():
    """Noise floor 3.3e-5, physical saddle 1.58e-2, geometric mean 7.2e-4."""
    noise = 3.3e-5
    physical = 1.58e-2
    geometric_mean = np.sqrt(noise * physical)
    assert geometric_mean == pytest.approx(7.2e-4, rel=0.05)
    assert noise < DEFAULT_CURVATURE_THRESHOLD < physical


def test_split_spectrum_classifies_the_three_bands():
    H = _diag([-2e-2, 1e-5, 5e-1])
    split = split_spectrum(H, curvature_threshold=1e-3)
    assert split.n_negative == 1
    assert split.n_soft == 1
    # ``hard`` is defined by magnitude, so the negative mode is hard too.
    assert int(split.hard.sum()) == 2
    assert split.lowest == pytest.approx(-2e-2, rel=1e-12)


def test_branch_spread_selects_by_energy_and_flags_the_e2_spread():
    points = [
        BranchPoint('saddle', -75.9848818, -0.1282449, hessian_lowest=-1.6e-2,
                    certified=False),
        BranchPoint('minimum', -76.0060407, -0.1095038, hessian_lowest=7e-8),
    ]
    report = branch_spread(points, e2_tolerance=1e-6)
    assert report.selected.label == 'minimum'
    assert report.n_certified == 1
    assert report.e2_spread == pytest.approx(0.0, abs=1e-15)
    assert not report.ambiguous
    assert report.rows[0]['certified'] is False


def test_branch_spread_between_two_certified_minima_is_ambiguous():
    points = [
        BranchPoint('a', -76.00, -0.1090),
        BranchPoint('b', -75.99, -0.1100),
    ]
    report = branch_spread(points, e2_tolerance=1e-6)
    assert report.selected.label == 'a'
    assert report.e2_spread == pytest.approx(1e-3, rel=1e-12)
    assert report.ambiguous
    assert any('branch-dependent' in reason for reason in report.reasons)


def test_branch_spread_without_a_certified_point_reports_none():
    report = branch_spread([BranchPoint('s', 0.0, 0.0, certified=False)])
    assert report.selected is None
    assert report.n_certified == 0
    assert any('no certified' in reason for reason in report.reasons)


def test_branch_spread_flags_a_degenerate_energy_pair():
    points = [
        BranchPoint('a', -76.0000000, -0.1090),
        BranchPoint('b', -76.0000005, -0.1100),
    ]
    report = branch_spread(points, e2_tolerance=1e-6, energy_tol=1e-6)
    assert report.degenerate
