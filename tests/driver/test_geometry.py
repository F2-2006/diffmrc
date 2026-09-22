"""Geometry optimization driven by analytic gradients."""
import numpy as np
import pytest

from diffmrc.driver.geometry import GeometryResult, optimize_geometry


def _harmonic(centre):
    """A convex surrogate with an analytic gradient."""
    centre = np.asarray(centre, dtype=float).ravel()

    def energy_and_gradient(flat):
        flat = np.asarray(flat, dtype=float)
        d = flat - centre
        return float(0.5 * d @ d), d

    return energy_and_gradient


def test_optimization_finds_a_known_minimum():
    centre = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]])
    r = optimize_geometry(_harmonic(centre), np.full(6, 0.3))
    assert isinstance(r, GeometryResult)
    assert np.allclose(r.coords.ravel(), centre.ravel(), atol=1e-6)
    assert r.converged


def test_result_reports_the_force_norm():
    centre = np.array([[0.0, 0.0, 0.0]])
    r = optimize_geometry(_harmonic(centre), np.array([1.0, 1.0, 1.0]))
    assert r.grad_norm < 1e-6


def test_result_reshapes_coordinates_to_atoms():
    r = optimize_geometry(_harmonic(np.zeros(9)), np.full(9, 0.1), n_atoms=3)
    assert r.coords.shape == (3, 3)


def test_result_preserves_a_two_dimensional_starting_shape():
    centre = np.zeros((3, 3))
    r = optimize_geometry(_harmonic(centre), np.full((3, 3), 0.1))
    assert r.coords.shape == (3, 3)


def test_an_impossible_target_is_not_reported_as_converged():
    """A function with no stationary point must not claim success.

    Relying on the optimizer's own flag is not enough: a budget
    exhaustion can return success-shaped output. The force norm is the
    independent check.
    """
    def energy_and_gradient(flat):
        flat = np.asarray(flat, dtype=float)
        return float(np.sum(flat)), np.ones_like(flat)

    r = optimize_geometry(energy_and_gradient, np.zeros(3), max_steps=5)
    assert not r.converged
    assert r.grad_norm > 1e-6


def test_the_step_budget_is_respected():
    def energy_and_gradient(flat):
        flat = np.asarray(flat, dtype=float)
        return float(0.5 * flat @ flat + 0.1 * flat.sum()), flat + 0.1

    r = optimize_geometry(energy_and_gradient, np.full(3, 5.0), max_steps=1)
    assert r.n_steps <= 1


def test_the_gradient_convention_matches_the_rest_of_the_package():
    """The caller supplies dE/dR, as core.grad.nuclear_gradient does.

    Supplying a gradient and receiving a descent direction is the
    convention everywhere else here; getting it backwards makes the
    optimizer climb away from the minimum.
    """
    centre = np.array([[2.0, -1.0, 0.5]])
    r = optimize_geometry(_harmonic(centre), np.zeros(3))
    assert np.allclose(r.coords.ravel(), centre.ravel(), atol=1e-6)


def test_a_result_carries_a_message_from_the_optimizer():
    r = optimize_geometry(_harmonic(np.zeros(3)), np.ones(3))
    assert r.message


def test_tightening_the_tolerance_can_leave_a_run_unconverged():
    """The tolerance is the caller's to set, and it is honoured."""
    centre = np.zeros(3)
    strict = optimize_geometry(_harmonic(centre), np.ones(3),
                               grad_tol=1e-14, max_steps=2)
    loose = optimize_geometry(_harmonic(centre), np.ones(3), grad_tol=1e-5)
    assert not strict.converged
    assert loose.converged
