"""Unit contracts of the Newton polish used by the DF-X2C CASSCF driver.

The integration cover is ``test_x2c_casscf_df.py``; these checks pin the
polish mechanics on a synthetic quadratic so that a failure localizes to
the refinement stage rather than to the CASSCF chain.
"""
import numpy as np

from diffmrc.core import x2c_casscf_df as xc


def test_polish_converges_on_quadratic_with_flat_direction():
    """Newton removes the gradient without touching the null direction."""
    hess = np.diag([1.0, 10.0, 0.0])
    lin = np.array([1e-4, -2e-4, 0.0])

    def eval_point(x):
        return (float(0.5 * x @ hess @ x + lin @ x),
                hess @ x + lin)

    theta, energy, grad, history = xc._newton_polish(
        eval_point, np.zeros(3), g_tol=1e-10, max_iter=5, fd_step=1e-5,
        rcond=1e-10, max_step=0.3)
    assert np.linalg.norm(grad) < 1e-10
    assert energy < 0.0
    assert history and all(step['accepted'] for step in history)
    assert abs(theta[2]) < 1e-14


def test_polish_rejects_a_step_without_resolved_curvature():
    """A gradient with no curvature behind it yields no step at all."""
    def eval_point(x):
        return float(x[0]), np.array([1.0])

    theta, energy, grad, history = xc._newton_polish(
        eval_point, np.zeros(1), g_tol=1e-10, max_iter=3, fd_step=1e-5,
        rcond=1e-10, max_step=0.3)
    assert np.linalg.norm(grad) == 1.0
    assert np.array_equal(theta, np.zeros(1))
    assert history[0]['accepted'] is False
    assert history[0]['note']
