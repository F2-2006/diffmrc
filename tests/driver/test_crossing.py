"""Scan orchestration: order, branch tracking, and the locator hand-off.

The evaluator is a plain callable here, which is the point of the
separation: the ordering logic is checked without a kernel, and a
kernel-backed scan is the same loop with a different callable.
"""
import pytest

from diffmrc.core.crossing import CrossingStatus, ScanPoint
from diffmrc.driver.crossing import CrossingScan, scan_two_state_crossing


def test_the_scan_passes_every_coordinate_in_order_and_keeps_state():
    seen = []

    def evaluate(x, previous):
        seen.append((x, previous))
        state = (previous or 0) + 1
        return ScanPoint(coordinate=x, energies=(x, -x),
                         gradients=(1.0, -1.0),
                         overlap_previous=None if previous is None else 1.0), \
            state

    result = scan_two_state_crossing(evaluate, [-0.5, -0.25, 0.25, 0.5])
    assert isinstance(result, CrossingScan)
    assert [s[0] for s in seen] == [-0.5, -0.25, 0.25, 0.5]
    assert seen[0][1] is None
    assert seen[1][1] == 1
    assert seen[3][1] == 3
    assert result.report.status is CrossingStatus.FOUND
    assert result.report.crossing == pytest.approx(0.0)


def test_the_scan_forwards_the_overlap_threshold():
    def evaluate(x, previous):
        # A branch jump right at the bracket.
        overlap = 0.5 if x == 0.25 else 1.0
        if previous is None:
            overlap = None
        return ScanPoint(coordinate=x, energies=(x, -x),
                         gradients=(1.0, -1.0),
                         overlap_previous=overlap), True

    result = scan_two_state_crossing(evaluate, [-0.5, -0.25, 0.25, 0.5],
                                     overlap_threshold=0.9)
    assert result.report.status is CrossingStatus.BRANCH_JUMP
    assert result.report.branch_jump_indices == (2,)


def test_an_empty_scan_is_refused():
    with pytest.raises(ValueError):
        scan_two_state_crossing(lambda x, p: (None, None), [])


@pytest.mark.slow
def test_ch2_bending_scan_with_the_casscf_evaluator():
    """The kernel-backed path on the system whose crossing is known.

    CH2/6-31G CAS(2,2), r(C-H) = 1.079 A, H-C-H angle 80-82 degrees.
    The stored scan of 2026-11-05 gives 81.74 degrees; the evaluator is
    expected to reproduce the bracket and the crossing to within the
    1-degree grid.
    """
    import math

    import numpy as np
    import pytest as _pytest
    _pytest.importorskip('pyscf')
    from pyscf import gto

    from diffmrc.driver.crossing import make_casscf_evaluator

    ang = 1.8897259886
    r_ch = 1.079

    def build_mol(theta, spin):
        half = math.radians(0.5 * theta)
        x = r_ch * math.sin(half)
        y = r_ch * math.cos(half)
        return gto.M(atom=[('C', (0.0, 0.0, 0.0)),
                           ('H', (x, y, 0.0)),
                           ('H', (-x, y, 0.0))],
                     basis='6-31g', charge=0, spin=int(spin),
                     unit='Angstrom', verbose=0)

    def project(theta, grad):
        half = math.radians(0.5 * theta)
        r = r_ch * ang
        v1 = np.array([0.5 * r * math.cos(half),
                       -0.5 * r * math.sin(half), 0.0])
        v2 = np.array([-0.5 * r * math.cos(half),
                       -0.5 * r * math.sin(half), 0.0])
        return float(np.sum(grad[1] * v1 + grad[2] * v2))

    evaluate = make_casscf_evaluator(
        build_mol, spins=(2, 0), active=(3, 4),
        conv_tol=1e-9, conv_tol_grad=1e-6, project_gradient=project)
    result = scan_two_state_crossing(evaluate, [80.0, 81.0, 82.0])
    report = result.report
    assert report.status is CrossingStatus.FOUND
    assert report.bracket == (81.0, 82.0)
    assert report.crossing == pytest.approx(81.74, abs=0.05)
    assert report.branch_clean
