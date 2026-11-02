"""Report-only gating information at the active-space boundary.

Two facts are recorded for every run without touching a verdict:

* the smallest eigenvalue of the orbital Hessian, so a minimum with a
  zero mode can be recognised as "flat direction, not certified";
* the reference core-active and active-virtual gaps, so an exactly
  degenerate shell split by the boundary can be named.

The boundary-degeneracy self-check is optional and is not wired into the
driver. These tests pin the report text, the threshold behaviour on the
measured N2 and HF gaps, and the fact that the four original checks are
unchanged when the optional check is not requested.
"""
import numpy as np

from diffmrc.core.stationary import CurvatureReport
from diffmrc.driver.kernel import (ActiveSpace, BOUNDARY_DEGENERACY_TOL,
                                   FLAT_DIRECTION_TOL, _active_virtual_gap,
                                   _boundary_degeneracy,
                                   _boundary_degeneracy_evidence,
                                   _core_active_gap,
                                   _flat_direction_evidence)
from diffmrc.driver.run import checks_from_result


def _report(n_negative, smallest):
    return CurvatureReport(n_negative=int(n_negative),
                           smallest=float(smallest),
                           largest=max(abs(float(smallest)), 1.0),
                           condition_number=1.0, n_near_zero=0)


def _space(core, active, virtual):
    return ActiveSpace(core=list(core), active=list(active),
                       virtual=list(virtual))


def _split_core_active():
    """Core 2 and active 3 are exactly degenerate at the reference level."""
    energies = np.array([-2.0, -1.0, -0.5, -0.5 + 1e-15, 0.2, 0.4, 0.6])
    return _space([0, 1, 2], [3, 4], [5, 6]), energies


def _split_active_virtual():
    """Active 4 and virtual 5 are exactly degenerate at the reference level."""
    energies = np.array([-2.0, -1.0, -0.5, -0.1, 0.2, 0.2 + 1e-14, 0.6])
    return _space([0, 1, 2], [3, 4], [5, 6]), energies


def test_flat_direction_is_reported_for_a_zero_mode_minimum():
    text = _flat_direction_evidence(_report(0, 5.22e-8))

    assert text is not None
    assert 'flat direction, not certified' in text
    assert '5.220e-08' in text


def test_flat_direction_is_not_reported_for_a_true_minimum():
    assert _flat_direction_evidence(_report(0, 0.555)) is None


def test_flat_direction_respects_the_threshold():
    assert _flat_direction_evidence(_report(0, FLAT_DIRECTION_TOL)) is None
    assert _flat_direction_evidence(_report(0, 0.5 * FLAT_DIRECTION_TOL)) \
        is not None


def test_flat_direction_is_not_reported_for_a_saddle():
    """A negative eigenvalue is already reported by the curvature check."""
    assert _flat_direction_evidence(_report(1, -2.8e-2)) is None


def test_flat_direction_is_not_reported_when_there_are_no_rotation_parameters():
    """An empty Hessian reports smallest=0; that is not a zero mode."""
    assert _flat_direction_evidence(_report(0, 0.0), npar=0) is None


def test_core_active_exact_degeneracy_is_reported():
    space, energies = _split_core_active()

    gap = _core_active_gap(space, energies)
    report = _boundary_degeneracy(space, energies)

    assert gap[:2] == (2, 3)
    assert gap[2] < BOUNDARY_DEGENERACY_TOL
    assert report['split'] is True
    assert report['core_active'][:2] == (2, 3)
    assert 'a degenerate shell is split across the space boundary' in \
        report['evidence'][0]
    assert 'core-active gap between orbitals 2 and 3' in report['evidence'][0]


def test_active_virtual_exact_degeneracy_is_reported():
    space, energies = _split_active_virtual()

    gap = _active_virtual_gap(space, energies)
    report = _boundary_degeneracy(space, energies)

    assert gap[:2] == (4, 5)
    assert gap[2] < BOUNDARY_DEGENERACY_TOL
    assert report['split'] is True
    assert 'active-virtual gap between orbitals 4 and 5' in \
        report['evidence'][0]


def test_n2_gap_of_8e_minus_3_is_not_flagged():
    """The measured N2/6-31G core-active gap must stay outside the check."""
    energies = np.array([-4.0, -3.0, -2.0, -1.0, -0.5,
                         0.008144764, 0.1, 0.2, 0.3, 0.5, 0.6])
    space = _space([0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10])

    report = _boundary_degeneracy(space, energies)

    assert report['split'] is False
    assert report['evidence'] == []


def test_a_gap_of_1e_minus_4_is_not_flagged():
    """Threshold behaviour, not the measured HF value.

    The 1e-4 Ha figure is the occupation-fallback degeneracy tolerance,
    not the final HF core-active gap. The measured HF final space [4, 5]
    has a 2.8e-15 Ha gap and is flagged by the next test; it is refused
    by the gradient check either way.
    """
    energies = np.array([-3.0, -2.0, -1.0, -0.5, -0.5 + 1e-4, 0.7, 0.9])
    space = _space([0, 1, 2], [3, 4], [5, 6])

    report = _boundary_degeneracy(space, energies)

    assert report['split'] is False
    assert report['evidence'] == []


def test_hf_final_core_active_gap_is_exact_degeneracy():
    """The measured HF/6-31G final space [4, 5]: gap 2.8e-15 Ha.

    This is a true positive, not a false one: the same degenerate pi
    shell is split as in HCl and HI. The verdict is unchanged because
    HF is already refused by its 4.2e-8 gradient.
    """
    energies = np.array([-3.0, -2.0, -1.0, -0.5, -0.5 + 2.8e-15, 0.7, 0.9])
    space = _space([0, 1, 2, 3], [4, 5], [6])

    report = _boundary_degeneracy(space, energies)

    assert report['split'] is True
    assert 'core-active gap between orbitals 3 and 4' in report['evidence'][0]


def test_threshold_is_1e_minus_8_hartree():
    assert BOUNDARY_DEGENERACY_TOL == 1e-8


def test_co_final_active_virtual_gap_is_exact_degeneracy():
    """The measured CO/6-31G final [6, 7]: gap 4.2e-16 Ha on the virtual side.

    The missing pi* partner is outside the space, so this is a true
    positive on the active-virtual boundary. CO is already refused by
    curvature in the same run.
    """
    energies = np.array([-4.0, -3.0, -2.0, -1.0, -0.7, -0.5,
                         0.05, 0.1, 0.1 + 4.2e-16, 0.6])
    space = _space([0, 1, 2, 3, 4, 5], [6, 7], [8, 9])

    report = _boundary_degeneracy(space, energies)

    assert report['split'] is True
    assert 'active-virtual gap between orbitals 7 and 8' in report['evidence'][0]


def test_a_wide_gap_produces_no_boundary_evidence():
    energies = np.array([-2.0, -1.0, -0.5, 0.1, 0.2, 0.5, 0.6])
    space = _space([0, 1, 2], [3, 4], [5, 6])
    report = _boundary_degeneracy(space, energies)

    assert _boundary_degeneracy_evidence(report) == []


def test_optional_boundary_check_is_absent_by_default():
    checks = checks_from_result(1e-12, 0, -1.0, -0.9, 2, 2)

    assert [c.name for c in checks] == [
        'orbital_gradient', 'hessian_curvature', 'variational',
        'active_electrons']
    assert all(c.passed for c in checks)


def test_optional_boundary_check_can_be_requested_without_a_split():
    checks = checks_from_result(1e-12, 0, -1.0, -0.9, 2, 2,
                                boundary_degeneracy=False)

    assert len(checks) == 5
    assert checks[-1].name == 'boundary_degeneracy'
    assert checks[-1].passed is True


def test_optional_boundary_check_fails_on_a_split_exact_degeneracy():
    space, energies = _split_core_active()
    report = _boundary_degeneracy(space, energies)

    checks = checks_from_result(1e-12, 0, -1.0, -0.9, 2, 2,
                                boundary_degeneracy=report)

    assert len(checks) == 5
    assert checks[-1].name == 'boundary_degeneracy'
    assert checks[-1].passed is False
    assert 'a degenerate shell is split' in checks[-1].detail


def test_the_four_original_checks_are_identical_with_the_optional_check():
    baseline = checks_from_result(1e-12, 0, -1.0, -0.9, 2, 2)
    extended = checks_from_result(1e-12, 0, -1.0, -0.9, 2, 2,
                                  boundary_degeneracy=False)

    assert [(c.name, c.passed, c.detail) for c in extended[:4]] == \
        [(c.name, c.passed, c.detail) for c in baseline]
