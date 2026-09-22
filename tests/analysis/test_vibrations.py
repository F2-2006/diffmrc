"""Vibrational frequencies from a Hessian, in mass-weighted coordinates."""
import numpy as np
import pytest

from diffmrc.analysis.vibrations import (
    AMU2ME,
    HA2CM,
    mass_weighted_hessian,
    trans_rot_projector,
    wavenumbers,
)

H2O_COORDS = np.array([[0.0, 0.0, 0.0],
                       [0.7557, 0.0, 0.5880],
                       [-0.7557, 0.0, 0.5880]])
H2O_MASSES = np.array([15.9949, 1.00783, 1.00783])
CO_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1283]])
CO_MASSES = np.array([12.0, 15.9949])


def _stretch_hessian(n_atoms, k):
    """A Hessian with one bond-stretching force constant along each axis."""
    H = np.zeros((3 * n_atoms, 3 * n_atoms))
    for axis in range(3):
        H[axis, axis] = k
        H[3 + axis, 3 + axis] = k
        H[axis, 3 + axis] = -k
        H[3 + axis, axis] = -k
    return H


def test_projector_counts_six_modes_for_a_nonlinear_molecule():
    proj = trans_rot_projector(H2O_COORDS, H2O_MASSES)
    assert proj.n_trans_rot == 6
    assert proj.P.shape == (9, 9)


def test_projector_counts_five_modes_for_a_linear_molecule():
    """The regression that a hard-coded 6 would miss.

    A linear molecule has only two independent rotations, so one extra
    mode is genuinely vibrational. Dropping six instead of five silently
    discards a real frequency, and two of the systems this package was
    validated on -- CO and N2 -- are linear.
    """
    proj = trans_rot_projector(CO_COORDS, CO_MASSES)
    assert proj.n_trans_rot == 5
    assert proj.P.shape == (6, 6)


def test_projector_is_idempotent():
    proj = trans_rot_projector(H2O_COORDS, H2O_MASSES)
    assert np.allclose(proj.P @ proj.P, proj.P, atol=1e-12)


def test_projector_is_symmetric():
    proj = trans_rot_projector(H2O_COORDS, H2O_MASSES)
    assert np.allclose(proj.P, proj.P.T, atol=1e-12)


def test_projector_annihilates_translation():
    """Translating every atom must lie in the projected-out space."""
    proj = trans_rot_projector(H2O_COORDS, H2O_MASSES)
    m3 = np.repeat(H2O_MASSES, 3)
    for axis in range(3):
        t = np.zeros(9)
        t[axis::3] = np.sqrt(m3[axis::3])
        assert np.allclose(proj.P @ t, 0.0, atol=1e-12)


def test_projector_rejects_a_mismatched_mass_count():
    with pytest.raises(ValueError):
        trans_rot_projector(H2O_COORDS, np.array([1.0, 2.0]))


def test_mass_weighting_divides_by_sqrt_mass():
    """Every element is divided by sqrt(m_i m_j), cross terms included.

    Two atoms of mass 1 and 4 give factors of 1 within the first atom, 2
    between the atoms, and 4 within the second -- so a matrix of ones
    comes back as those factors.
    """
    H = np.ones((6, 6))
    masses = np.array([1.0, 4.0])
    Hmw = mass_weighted_hessian(H, masses)
    assert Hmw[0, 0] == pytest.approx(1.0)
    assert Hmw[0, 3] == pytest.approx(0.5)
    assert Hmw[0, 4] == pytest.approx(0.5)
    assert Hmw[3, 3] == pytest.approx(0.25)


def test_mass_weighting_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        mass_weighted_hessian(np.eye(6), np.array([1.0, 2.0, 3.0]))


def test_constants_are_mutually_consistent():
    """The conversion factor is derived, not a second literal."""
    assert HA2CM / np.sqrt(AMU2ME) == pytest.approx(5140.487143730784,
                                                    rel=1e-12)


def test_wavenumbers_of_a_known_diatomic():
    """A diatomic has exactly one vibration, with an analytic value.

    In mass-weighted coordinates the single eigenvalue of a stretching
    force constant k is k/mu with mu the reduced mass.
    """
    k = 0.5  # Ha/Bohr^2
    r = wavenumbers(_stretch_hessian(2, k), CO_COORDS, CO_MASSES)
    assert r.n_trans_rot == 5
    assert r.freqs_cm.shape == (1,)
    mu = CO_MASSES[0] * CO_MASSES[1] / CO_MASSES.sum()
    expected = np.sqrt(k / mu) * HA2CM / np.sqrt(AMU2ME)
    assert r.freqs_cm[0] == pytest.approx(expected, rel=1e-9)


def test_wavenumbers_of_water_give_three_modes():
    rng = np.random.default_rng(20260922)
    A = rng.normal(size=(9, 9))
    r = wavenumbers(A @ A.T, H2O_COORDS, H2O_MASSES)
    assert r.freqs_cm.shape == (3,)
    assert r.n_trans_rot == 6


def test_wavenumbers_report_imaginary_modes():
    """A negative eigenvalue is an imaginary frequency and must be counted
    rather than silently absolute-valued."""
    H = np.zeros((9, 9))
    for axis in range(3):
        H[axis, axis] = -0.5
        H[3 + axis, 3 + axis] = -0.5
        H[axis, 3 + axis] = 0.5
        H[3 + axis, axis] = 0.5
    r = wavenumbers(H, H2O_COORDS, H2O_MASSES)
    assert r.n_imag >= 1


def test_wavenumbers_zero_modes_are_small():
    """The projected-out modes must come back numerically zero.

    This is the check that the projector did its job; without it a caller
    cannot tell a real low frequency from an unprojected translation.
    """
    rng = np.random.default_rng(20260922)
    A = rng.normal(size=(9, 9))
    r = wavenumbers(A @ A.T, H2O_COORDS, H2O_MASSES)
    assert r.n_trans_rot == 6
    assert np.all(r.zero_mode_magnitudes < 1e-8)


def test_wavenumbers_are_sorted():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(9, 9))
    r = wavenumbers(A @ A.T, H2O_COORDS, H2O_MASSES)
    assert np.all(np.diff(r.freqs_cm) >= 0)


def test_wavenumbers_reject_a_mismatched_coordinate_count():
    with pytest.raises(ValueError):
        wavenumbers(np.eye(6), H2O_COORDS, H2O_MASSES)
