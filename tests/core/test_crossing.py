"""Crossing location on analytic surfaces and on a stored CH2 scan.

The analytic cases are the point of the module: a linear pair of
surfaces has a crossing that can be written down, so the interpolation,
the uncertainty and the gradient difference are checked against exact
numbers rather than against another implementation.  The CH2 case is the
end-to-end regression: a real CAS(2,2)/6-31G scan whose crossing angle
was reported as 81.74 degrees, and whose data are stored in this file so
the test does not depend on a private project directory.
"""
import math

import numpy as np
import pytest

from diffmrc.core.crossing import (
    CrossingStatus,
    ScanPoint,
    locate_crossing,
    matched_orbital_overlaps,
    orbital_space_overlap,
    two_start_certificate,
)

RAD_PER_DEG = math.pi / 180.0


def _linear_points(xs, slope_a=1.0, slope_b=-1.0, offset_a=0.0,
                   offset_b=0.0, overlap=None, gradients=True):
    points = []
    for k, x in enumerate(xs):
        ea = offset_a + slope_a * x
        eb = offset_b + slope_b * x
        g = (slope_a, slope_b) if gradients else None
        points.append(ScanPoint(coordinate=float(x), energies=(ea, eb),
                                gradients=g,
                                overlap_previous=(None if k == 0 else overlap)))
    return points


def test_a_linear_crossing_is_exact():
    """E0 = x, E1 = -x: the crossing is at zero and nothing else is."""
    # The grid avoids zero: an exact zero takes the grid-point branch,
    # which is tested separately.
    points = _linear_points(np.linspace(-0.55, 0.45, 11))
    report = locate_crossing(points)
    assert report.status is CrossingStatus.FOUND
    assert report.crossing == pytest.approx(0.0, abs=1e-15)
    assert report.bracket == pytest.approx((-0.05, 0.05))
    assert report.step_bound == pytest.approx(0.05)
    # The surfaces are linear, so the curvature model adds nothing.
    assert report.uncertainty == pytest.approx(0.0, abs=1e-12)
    assert report.gradient_difference == pytest.approx(2.0, abs=1e-12)


def test_a_quadratic_surface_beats_the_linear_root():
    """E0 = x^2 + x - 1, E1 = -x + 1 cross at -1 + sqrt(3)."""
    exact = -1.0 + math.sqrt(3.0)
    xs = np.linspace(0.4, 1.2, 9)
    points = []
    for k, x in enumerate(xs):
        ea = x * x + x - 1.0
        eb = -x + 1.0
        points.append(ScanPoint(
            coordinate=float(x), energies=(ea, eb),
            gradients=(2.0 * x + 1.0, -1.0),
            overlap_previous=(None if k == 0 else 1.0)))
    report = locate_crossing(points)
    assert report.status is CrossingStatus.FOUND
    assert abs(report.crossing - exact) < 0.01
    # The quadratic correction must move the answer toward the exact root
    # and must be reported with the right sign.
    linear_error = abs((report.crossing - report.model_correction) - exact)
    assert abs(report.crossing - exact) < linear_error
    assert report.uncertainty == pytest.approx(abs(report.model_correction))


def test_the_step_length_controls_the_bracket_bound():
    # Neither grid contains zero, so both take the bracket branch.
    coarse = locate_crossing(_linear_points(np.linspace(-0.95, 0.95, 6)))
    fine = locate_crossing(_linear_points(np.linspace(-0.95, 0.97, 41)))
    assert coarse.step_bound == pytest.approx(0.19)
    assert fine.step_bound == pytest.approx(0.024)


def test_a_crossing_from_positive_to_negative_is_accepted():
    """The monotonicity direction follows the sign change, not a fixed
    increase: CH2's triplet-minus-singlet gap decreases through its
    crossing, and a fixed-increase rule refused it."""
    xs = np.linspace(-0.55, 0.45, 11)
    points = [ScanPoint(coordinate=float(x), energies=(-x, x + 0.0),
                        gradients=(-1.0, 1.0),
                        overlap_previous=None if k == 0 else 1.0)
              for k, x in enumerate(xs)]
    report = locate_crossing(points)
    assert report.status is CrossingStatus.FOUND
    assert report.crossing == pytest.approx(0.0, abs=1e-12)


def test_a_non_monotone_bracket_is_refused():
    """A wiggle next to the bracket is the signature of a second feature."""
    xs = np.linspace(-0.55, 0.45, 11)
    points = _linear_points(xs)
    # Push one point on the left side up so the gap is not monotone there.
    broken = list(points)
    p = broken[3]
    broken[3] = ScanPoint(coordinate=p.coordinate,
                          energies=(p.energies[0] + 0.3, p.energies[1]),
                          gradients=p.gradients,
                          overlap_previous=p.overlap_previous)
    report = locate_crossing(broken, monotonic_window=2)
    assert report.status is CrossingStatus.NON_MONOTONIC
    assert report.uncertainty is None
    assert any('monotone' in m for m in report.messages)


def test_a_branch_jump_blocks_the_bracket():
    xs = np.linspace(-0.55, 0.45, 11)
    points = _linear_points(xs, overlap=1.0)
    jump = list(points)
    p = jump[5]
    jump[5] = ScanPoint(coordinate=p.coordinate, energies=p.energies,
                        gradients=p.gradients, overlap_previous=0.665)
    report = locate_crossing(jump, overlap_threshold=0.90)
    assert report.status is CrossingStatus.BRANCH_JUMP
    assert report.crossing is not None          # the raw number is visible
    assert report.uncertainty is None           # but it is not licenced
    assert report.branch_jump_indices == (5,)
    assert not report.branch_clean


def test_a_clean_scan_records_that_continuity_was_checked():
    xs = np.linspace(-0.55, 0.45, 11)
    report = locate_crossing(_linear_points(xs, overlap=0.999))
    assert report.status is CrossingStatus.FOUND
    assert report.branch_clean
    assert report.branch_jump_indices == ()


def test_no_crossing_reports_the_closure_distance():
    """Parallel surfaces never cross; the local slope says how far away."""
    xs = np.linspace(0.0, 1.0, 11)
    points = [ScanPoint(coordinate=float(x), energies=(x + 0.5, x))
              for x in xs]
    report = locate_crossing(points)
    assert report.status is CrossingStatus.NO_CROSSING
    assert report.crossing is None
    assert report.nearest is not None
    assert report.nearest.gap == pytest.approx(0.5)
    assert report.nearest.slope == pytest.approx(0.0, abs=1e-12)
    assert report.nearest.closure_distance is None


def test_an_exact_zero_on_the_grid_is_a_crossing():
    xs = np.linspace(-1.0, 1.0, 11)          # includes zero
    points = [ScanPoint(coordinate=float(x), energies=(x, -x))
              for x in xs]
    report = locate_crossing(points)
    assert report.status is CrossingStatus.FOUND
    assert report.crossing == pytest.approx(0.0, abs=1e-15)
    assert report.bracket == (0.0, 0.0)
    assert report.uncertainty == pytest.approx(0.0)


def test_two_brackets_are_both_listed():
    """A gap that changes sign twice must not hide the second candidate."""
    xs = np.linspace(-2.0, 2.0, 17)
    points = [ScanPoint(coordinate=float(x),
                        energies=((x - 1.03) * (x + 1.03), 0.0))
              for x in xs]
    report = locate_crossing(points)
    assert report.status is CrossingStatus.FOUND
    assert len(report.all_brackets) == 2
    assert report.bracket[0] < report.bracket[1]


def test_scan_point_ordering_is_enforced():
    points = [ScanPoint(coordinate=1.0, energies=(0.0, 1.0)),
              ScanPoint(coordinate=0.5, energies=(1.0, 0.0))]
    with pytest.raises(ValueError):
        locate_crossing(points)
    with pytest.raises(ValueError):
        locate_crossing(points[:1])


def test_orbital_overlap_is_invariant_to_rotations_inside_the_space():
    rng = np.random.default_rng(7)
    n = 10
    a = np.linalg.qr(rng.normal(size=(n, n)))[0]
    s = np.eye(n)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    b = a.copy()
    b[:, 0:3] = a[:, 0:3] @ q
    assert orbital_space_overlap(a, b, s, 0, 3) == pytest.approx(1.0)
    vals = matched_orbital_overlaps(a, b, s, 0, 3)
    assert np.allclose(np.sort(vals), 1.0)
    # One changed column: the summed overlap drops, the singular-value
    # spectrum shows which orbital moved.
    c = a.copy()
    c[:, 2] = a[:, 5]
    assert orbital_space_overlap(a, c, s, 0, 3) == pytest.approx(2.0 / 3.0)
    vals = np.sort(matched_orbital_overlaps(a, c, s, 0, 3))
    assert vals[0] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(vals[1:], 1.0)


def test_two_start_certificate_distinguishes_the_four_outcomes():
    same = two_start_certificate(-10.0, -10.0 + 1e-8, 1e-7, 2e-7, 0.9999)
    assert same.verdict == 'same_stationary_point'
    distinct = two_start_certificate(-10.0, -10.0 - 1.2e-3, 1e-7, 1e-7, 0.66)
    assert distinct.verdict == 'distinct_stationary_points'
    loose = two_start_certificate(-10.0, -10.0 - 1.2e-3, 2e-2, 1e-7, 0.66)
    assert loose.verdict == 'unconverged'
    stale = two_start_certificate(-10.0, -10.0 - 1.2e-3, 1e-7, 1e-7, 0.9999)
    assert stale.verdict == 'inconsistent'
    assert distinct.energy_difference == pytest.approx(1.2e-3)
    assert distinct.overlap == pytest.approx(0.66)


# CH2/6-31G, CAS(2,2), r(C-H) = 1.079 A, 1-degree scan (2026-11-05).
# Columns: theta / deg, E(singlet) / Eh, E(triplet) / Eh,
# dE/dtheta(singlet) / Eh rad^-1, dE/dtheta(triplet) / Eh rad^-1,
# overlap of the singlet active space with the previous point,
# overlap of the triplet active space with the previous point.
_CH2 = np.array([
    [78.0, -38.844783128842, -38.840492265667,
     -0.102237613240, -0.168516621264, np.nan, np.nan],
    [79.0, -38.846532269013, -38.843395578765,
     -0.098198895085, -0.164183164766, 0.999558, 0.999068],
    [80.0, -38.848210899513, -38.846223505708,
     -0.094158238925, -0.159878247454, 0.999561, 0.999063],
    [81.0, -38.849819012262, -38.848976557318,
     -0.090118601306, -0.155603330264, 0.999564, 0.999058],
    [82.0, -38.851356650993, -38.851655270061,
     -0.086082999366, -0.151359887110, 0.999567, 0.999053],
    [83.0, -38.852823911367, -38.854260206078,
     -0.082054308553, -0.147149382798, 0.999569, 0.999047],
    [84.0, -38.854220940338, -38.856791952848,
     -0.078035573680, -0.142973253192, 0.999572, 0.999041],
    [85.0, -38.855547936151, -38.859251122537,
     -0.074029537731, -0.138832892184, 0.999574, 0.999035],
    [86.0, -38.856805146610, -38.861638351082,
     -0.070039218508, -0.134729628784, 0.999577, 0.999028],
])


def _ch2_points():
    points = []
    for k, row in enumerate(_CH2):
        overlap = None
        if k > 0:
            overlap = float(min(row[5], row[6]))
        points.append(ScanPoint(
            coordinate=float(row[0]),
            energies=(float(row[1]), float(row[2])),
            gradients=(float(row[3]) * RAD_PER_DEG,
                       float(row[4]) * RAD_PER_DEG),
            overlap_previous=overlap))
    return points


def test_the_stored_ch2_scan_reproduces_its_reported_crossing():
    """The regression that ties the module to the 2026-11-05 result.

    The report gives 81.74 deg from the 1-degree scan, a bracket bound
    of +-0.5 deg and a fitted uncertainty of about 0.01 deg.  The
    gradient difference is +1.141 mEh/deg.
    """
    report = locate_crossing(_ch2_points())
    assert report.status is CrossingStatus.FOUND
    assert report.bracket == (81.0, 82.0)
    assert report.crossing == pytest.approx(81.738, abs=0.01)
    assert report.step_bound == pytest.approx(0.5)
    assert report.uncertainty is not None
    assert report.uncertainty < 0.01
    assert report.gradient_difference * 1e3 == pytest.approx(1.141, abs=0.005)
    assert report.monotonic and report.monotonic_checked
    assert report.branch_clean


def test_importing_the_crossing_module_does_not_load_the_kernel():
    """core.crossing is numerics only; it must stand without pyscf or jax."""
    import subprocess
    import sys
    code = ('import sys; import diffmrc.core.crossing; '
            'heavy = [m for m in ("pyscf", "pyscfad", "jax") '
            'if m in sys.modules]; print("HEAVY:", heavy)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert 'HEAVY: []' in out.stdout, out.stdout
