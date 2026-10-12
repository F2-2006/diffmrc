"""Diatomic scan fits: the t78/t88 unit chain and the audit numbers."""
import numpy as np
import pytest

from diffmrc.analysis.vibrations import (SCAN_SPAN_MAX, fit_scan_parabola,
                                         multipoint_scan_estimates,
                                         omega_of_curvature)


def test_clean_parabola_recovers_vertex_and_curvature():
    xv, spring = 1.95, 0.6           # spring = d2E/dr2 = 2 * quadratic coeff
    grid = np.array([xv - 0.03, xv, xv + 0.03])
    y = 0.5 * spring * (grid - xv) ** 2 - 10.0
    fit = fit_scan_parabola(grid, y, mu_amu=1.0006)
    assert fit.vertex == pytest.approx(xv, abs=1e-9)
    assert fit.k == pytest.approx(spring, rel=1e-6)
    assert fit.vertex_in_window
    assert fit.omega == pytest.approx(2106.5, rel=2e-3)


def test_the_lah_anchor_reproduces_the_literature_frequency():
    """The 2026-10-11 unit-chain fix was validated against the H&H
    1428.7 cm^-1 value on this exact record; pin it in the package."""
    grid = np.array([2.002, 2.032, 2.062])
    y = np.array([-8490.867676065, -8490.867484359, -8490.867044288])
    mu = 138.90547 * 1.007825 / (138.90547 + 1.007825)
    fit = fit_scan_parabola(grid, y, mu)
    assert fit.vertex == pytest.approx(1.9938, abs=1e-3)
    assert fit.omega == pytest.approx(1428.6, abs=1.5)


def test_a_crossed_basin_flags_on_the_span():
    # The centre point jumped into another solution (0.015 Ha step vs the
    # physical 2e-4 .. 1.2e-3 Ha scale).
    grid = np.array([1.99, 2.02, 2.05])
    y = np.array([-10.0, -9.985, -9.9846])
    fit = fit_scan_parabola(grid, y, mu_amu=17.0)
    assert fit.span > SCAN_SPAN_MAX


def test_a_displaced_window_shows_as_an_outside_vertex():
    grid = np.array([2.01, 2.04, 2.07])      # minimum at 1.95, window right of it
    y = 0.5 * (grid - 1.95) ** 2
    fit = fit_scan_parabola(grid, y, mu_amu=100.0)
    assert not fit.vertex_in_window


def test_the_three_estimates_agree_on_a_mildly_anharmonic_curve():
    x = np.array([1.92, 1.95, 1.98, 2.01, 2.04])
    d = x - 1.975
    y = 0.5 * d ** 2 + 0.02 * d ** 4
    est = multipoint_scan_estimates(x, y, mu_amu=20.0)
    assert set(est) == {'q3', 'q5', 'q4'}
    omegas = [est[key].omega for key in ('q3', 'q5', 'q4')]
    assert min(omegas) > 0
    assert max(omegas) / min(omegas) < 1.05
    for fit in est.values():
        assert fit.vertex_in_window
        assert fit.vertex == pytest.approx(1.975, abs=2e-3)


def test_curvature_chain_refuses_a_negative_spring_constant():
    assert np.isnan(omega_of_curvature(-0.5, 10.0))
    assert np.isnan(omega_of_curvature(0.5, 0.0))


def test_fit_refuses_fewer_than_three_points():
    with pytest.raises(ValueError):
        fit_scan_parabola([1.0, 2.0], [0.0, 1.0], 10.0)
