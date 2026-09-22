"""Geometry optimization driven by analytic nuclear gradients.

The optimizer is L-BFGS-B, the same choice as in ``core.optimizer``, where
the module docstring records why the hand-written alternative was
abandoned: its line-search fallback was unreachable and its initial
Hessian scaling was fixed, so it either overstepped or collapsed into
microscopic steps.

Convergence is reported from the force norm as well as from the
optimizer's own flag. A step budget that runs out can return a
success-shaped result, and a geometry that is still far from stationary
must not be reported as converged.

Units: the caller supplies coordinates and gradients in Bohr and
Eh/Bohr, which is the convention of ``core.grad.nuclear_gradient``.
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = ['GeometryResult', 'ElectronicStructureFailure', 'optimize_geometry']

DEFAULT_GRAD_TOL = 1e-5
DEFAULT_MAX_STEPS = 200


class ElectronicStructureFailure(RuntimeError):
    """Raised by the energy callback when the underlying calculation failed.

    This is a different failure from the geometry not converging. A
    geometry that has not converged is a usable intermediate structure;
    an electronic-structure failure means the energy at that geometry is
    not trustworthy at all, so the optimizer must stop rather than take
    a step based on it.

    Raising this from `energy_and_gradient` lets `optimize_geometry`
    report the two apart instead of collapsing both into "something went
    wrong".
    """


@dataclass
class GeometryResult:
    """An optimized geometry and the evidence it converged."""

    coords: np.ndarray = field(default_factory=lambda: np.array([]))
    energy: float = float('nan')
    grad_norm: float = float('inf')
    n_steps: int = 0
    converged: bool = False
    message: str = ''
    failure: str = ''
    """Empty when the run completed; otherwise which kind of failure it was.

    ``'electronic_structure'`` means the energy callback raised
    `ElectronicStructureFailure`, so the geometry is wherever the
    optimizer had reached and no further steps were taken.
    ``'not_converged'`` means the optimizer stopped on its own terms
    without reaching the force tolerance.
    """


def optimize_geometry(energy_and_gradient, coords0, n_atoms=None,
                      max_steps=DEFAULT_MAX_STEPS, grad_tol=DEFAULT_GRAD_TOL):
    """Minimize the energy over nuclear coordinates.

    Parameters
    ----------
    energy_and_gradient : callable
        Takes a flat coordinate array and returns ``(energy, gradient)``,
        with the gradient in the same length units as the coordinates.
    coords0 : array_like
        Starting coordinates, flat or shaped.
    n_atoms : int, optional
        Number of atoms, used to shape the returned coordinates. When
        omitted the shape of `coords0` is preserved if it was 2-D.
    max_steps : int
        Iteration cap.
    grad_tol : float
        Force norm below which the geometry is called converged.

    Returns
    -------
    GeometryResult
        ``converged`` requires both the force norm to be below `grad_tol`
        and the optimizer not to have reported a failure.
    """
    from scipy.optimize import minimize

    coords0 = np.asarray(coords0, dtype=float)
    original_shape = coords0.shape
    flat0 = coords0.ravel()

    if n_atoms is not None:
        shape = (int(n_atoms), 3)
    elif len(original_shape) == 2:
        shape = original_shape
    else:
        shape = (-1, 3)

    # The energy callback can fail for a reason that has nothing to do
    # with the geometry -- an SCF that will not converge, an active space
    # that turned out to be wrong. Those must stop the run rather than be
    # optimized through, so they are caught separately from the
    # geometry-not-converged outcome.
    state = {'failure': '', 'message': ''}

    def fun(x):
        try:
            e, g = energy_and_gradient(x)
        except ElectronicStructureFailure as exc:
            state['failure'] = 'electronic_structure'
            state['message'] = str(exc)[:200]
            raise
        return float(e), np.asarray(g, dtype=float).ravel()

    try:
        res = minimize(fun, flat0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=int(max_steps), ftol=1e-14,
                                    gtol=float(grad_tol), maxls=50))
        x = np.asarray(res.x, dtype=float)
        energy, gradient = fun(x)
        grad_norm = float(np.linalg.norm(gradient))
        n_steps = int(getattr(res, 'nit', 0))
        message = str(res.message)[:80]
        # Two independent conditions, both required.
        #
        # scipy's own flag alone is not enough: a line search that cannot
        # make progress can report success with a large gradient left.
        # The gradient alone is not enough either: a run that exhausted
        # its iteration budget stops with whatever force it had reached,
        # and calling that converged hides that the optimizer gave up
        # rather than finished. L-BFGS-B reports status 1 for "iteration
        # limit reached", so `success` -- not the status code -- is what
        # indicates a genuine stop.
        optimizer_ok = bool(res.success)
        converged = bool(optimizer_ok and grad_norm < grad_tol)
        failure = '' if converged else 'not_converged'
    except ElectronicStructureFailure:
        x = flat0
        energy = float('nan')
        grad_norm = float('inf')
        n_steps = 0
        message = state['message']
        converged = False
        failure = 'electronic_structure'

    return GeometryResult(
        coords=x.reshape(shape), energy=energy, grad_norm=grad_norm,
        n_steps=n_steps, converged=converged, message=message,
        failure=failure,
    )
