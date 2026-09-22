"""Producing the orbital stationary point that the checks are applied to.

The point comes from two stages with different jobs.

``optimize`` runs scipy L-BFGS-B from the reference orbitals. Its
practical floor is not zero: on H2O/6-31g it stops with |g| about 1.5e-5
because its line search can no longer lower the energy (ftol=1e-14), not
because it exhausted its budget -- 2000 and 8000 steps stop bit-for-bit
identically. The stage's target is therefore set loosely (1e-6), above
the floor, so that reaching it is informative and missing it is not a
failure.

``refine`` runs damped Newton steps from ``core.stationary``. Measured:
two steps take |g| from 1.5e-5 to 3.9e-14, so the 1e-11 default target
is reachable with four orders of margin.

Asking the optimizer for 1e-12 instead -- which several scripts did --
makes every run end with a non-convergence report at 1.5e-5 that the
call site cannot tell apart from a real failure. Splitting the two
stages separates "the optimizer cannot go further" from "the point is
not stationary", and each outcome is reported as itself.

The convergence verdict is the gradient. scipy's L-BFGS-B reports
status 1 for "iteration limit reached"; the status and the message are
kept as evidence, but they are never promoted to a convergence claim.

One limitation is worth stating here because the callers depend on it:
a stationary saddle also has |g| = 0, so the gradient criterion alone
reports it as stationary. The gate for that is the curvature check in
``driver.run``, not this module. On H2O/6-31g the point reached from
theta = 0 has three negative Hessian eigenvalues and still refines to
|g| = 3.9e-14.
"""
from dataclasses import dataclass

from ..core.optimizer import optimize_theta_hand, optimize_theta_scipy
from ..core.stationary import refine_stationary_point

__all__ = ['OrbitalPoint', 'converge_orbital_point', 'OPTIMIZER_GRAD_TOL',
           'REFINE_TARGET']

OPTIMIZER_GRAD_TOL = 1e-6
REFINE_TARGET = 1e-11
MAX_STEPS = 2000
REFINE_MAX_ITER = 80


@dataclass
class OrbitalPoint:
    """The refined point, with both stages' evidence kept.

    ``grad_norm`` is the final gradient; ``optimizer_grad_norm`` is what
    the optimizer stopped at, before refinement. ``optimizer_reached``
    says whether the loose stage met its own target -- a run that misses
    it is not a failure, it is the case refinement exists for.
    """

    theta: object
    energy: float
    grad_norm: float
    optimizer_grad_norm: float
    optimizer_message: str
    optimizer_nit: int
    optimizer_reached: bool
    refinement: object

    @property
    def is_stationary(self):
        """True only when the final gradient reached the refine target."""
        return bool(self.refinement.is_stationary)

    @property
    def stop_reason(self):
        return self.refinement.stop_reason

    @property
    def target(self):
        return self.refinement.target

    @property
    def n_accepted(self):
        return self.refinement.n_accepted


def converge_orbital_point(E_theta, npar, theta0=None,
                           optimize_tol=OPTIMIZER_GRAD_TOL,
                           refine_target=REFINE_TARGET,
                           max_steps=MAX_STEPS,
                           refine_max_iter=REFINE_MAX_ITER):
    """Optimize with a loose target, then refine to the tight one.

    Parameters
    ----------
    E_theta : callable
        Differentiable function of the parameter vector.
    npar : int
        Number of parameters.
    theta0 : array_like, optional
        Starting point; zeros when omitted.
    optimize_tol : float
        Target for the L-BFGS-B stage.
    refine_target : float
        Target for the Newton stage, and the one the verdict uses.
    max_steps, refine_max_iter : int
        Iteration caps for the two stages.

    Returns
    -------
    OrbitalPoint
    """
    info = {}
    E_jit, theta, g_after, info = optimize_theta_scipy(
        E_theta, npar, steps=max_steps, g_conv=optimize_tol, theta0=theta0)
    if E_jit is None:
        # scipy raised; the handwritten fallback keeps the run alive.
        E_jit, theta, g_after, info = optimize_theta_hand(
            E_theta, npar, steps=max_steps, g_conv=optimize_tol,
            theta0=theta0)

    sp = refine_stationary_point(E_theta, theta, max_iter=refine_max_iter,
                                 target=refine_target)
    g_after = float(g_after)
    return OrbitalPoint(
        theta=sp.theta,
        energy=float(sp.energy),
        grad_norm=float(sp.grad_norm),
        optimizer_grad_norm=g_after,
        optimizer_message=str(info.get('msg', '')),
        optimizer_nit=int(info.get('nit', -1)),
        optimizer_reached=bool(g_after <= optimize_tol),
        refinement=sp,
    )
