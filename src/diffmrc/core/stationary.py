"""Refining a stationary point of the orbital-rotation energy.

The optimizer in ``optimizer`` finds a point where the orbital gradient is
small; this module takes that point and refines it with damped Newton
steps until the gradient stops improving. The damping and the acceptance
test are what keep the refinement honest.

Why the acceptance test is not optional
---------------------------------------

Without it the iteration can take a step that raises the energy and then
report the higher value as the stationary point. This project measured
exactly that: an unguarded refinement moved the energy from -76.07 to
+1.04 Ha and printed no error, leaving a result that looks plausible and
is not a minimum.

The guard existed only in the later experiment scripts. The scripts that
produced the production numbers ran bare, unguarded iterations, so moving
this path into a module is the point at which the guard stops being
optional.

The iteration also stops when no step lowers the energy, rather than
spinning to the iteration cap. On an already-converged point that is the
normal outcome, which is why ``guard_stopped`` is reported as data rather
than raised as an error.
"""
from dataclasses import dataclass

import numpy as np
import jax

__all__ = ['StationaryPoint', 'CurvatureReport', 'refine_stationary_point',
           'curvature_report']

# The energy test is a guard against divergence, not a measure of
# progress, so it allows a step that leaves the energy unchanged to
# within floating-point noise. Requiring a *significant* decrease would
# contradict the gradient target, and the two are quantitatively linked:
# near convergence a Newton step lowers the energy by about
# |g|^2 / (2 kappa), so reaching |g| < 1e-11 means accepting energy
# changes of order 1e-22. A decrease requirement of 1e-14 -- the value
# the experiment scripts used -- rejects every step in that regime and
# strands the refinement at |g| ~ 1e-8.
#
# Measured on a well-conditioned quadratic: with the 1e-14 decrease
# requirement the run stopped after one step at |g| = 2.4e-8; allowing
# a noise-sized increase reached |g| = 3.5e-16 in two steps.
_ENERGY_SLACK = 1e-14
_LAMBDA_INIT = 1e-8
_LAMBDA_MIN = 1e-12
_LAMBDA_UP = 10.0
_LAMBDA_DOWN = 0.3
_INNER_TRIES = 14

# A cap on how far a single Newton step may move the parameters. The
# downhill-only rule is not sufficient on its own: a function with no
# stationary point (linear in some direction) lowers the energy at every
# step, so every step is accepted and the parameters walk off to
# infinity while the energy keeps "improving". Measured in this project:
# a linear surrogate reached |theta| ~ 4e13 and spent the whole iteration
# budget. A step larger than this means the Newton model has stopped
# describing the function, which is the signal to stop.
_MAX_STEP_NORM = 10.0


@dataclass
class StationaryPoint:
    """A refined point, with the evidence that it is stationary.

    ``guard_stopped`` says the loop ended because it could not make
    progress. Read it together with ``grad_norm``: a guard stop with a
    gradient still above ``target`` means the point is **not** a
    stationary point, and the caller must not treat the energy as
    belonging to one.
    """

    theta: object
    energy: float
    grad_norm: float
    n_accepted: int = 0
    guard_stopped: bool = False
    stalled_gradient: bool = False


@dataclass
class CurvatureReport:
    """Eigenvalue summary of a Hessian."""

    n_negative: int
    smallest: float
    largest: float
    condition_number: float
    n_near_zero: int

    @property
    def is_minimum(self):
        """A minimum has no negative curvature."""
        return self.n_negative == 0


def curvature_report(H, negative_tol=1e-8):
    """Summarize the eigenvalue spectrum of a Hessian.

    Parameters
    ----------
    H : array_like
        Square symmetric matrix; only its symmetric part is used, because
        an automatic-differentiation Hessian is symmetric only to
        round-off and ``eigvalsh`` assumes exact symmetry.
    negative_tol : float
        An eigenvalue below the negative of this counts as negative. The
        tolerance matters: an AD Hessian in a flat direction carries
        eigenvalues of order 1e-14 that are numerical noise, not
        curvature, and counting them as negative would report a
        minimum as a saddle point.

    Returns
    -------
    CurvatureReport
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f'H must be square, got shape {H.shape}')
    H = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(H)
    smallest = float(w[0])
    largest = float(w[-1])
    scale = max(abs(largest), abs(smallest), 1e-30)
    return CurvatureReport(
        n_negative=int((w < -negative_tol).sum()),
        smallest=smallest,
        largest=largest,
        condition_number=float(scale / max(abs(smallest), 1e-30)),
        n_near_zero=int((np.abs(w) < negative_tol).sum()),
    )


def refine_stationary_point(E_theta, theta0, max_iter=80, target=1e-11):
    """Damped Newton refinement that only ever moves downhill.

    Parameters
    ----------
    E_theta : callable
        Differentiable function of the parameter vector.
    theta0 : array_like
        Starting point, typically the output of
        ``optimizer.optimize_theta``.
    max_iter : int
        Iteration cap.
    target : float
        Stop when the gradient norm falls below this.

    Returns
    -------
    StationaryPoint
        ``guard_stopped`` is True when the loop ended because no step
        lowered the energy. That is a normal outcome on a converged
        point, not an error.
    """
    import jax.numpy as jnp

    grad = jax.jit(jax.grad(E_theta))
    hess = jax.jit(jax.hessian(E_theta))

    theta = jnp.asarray(np.asarray(theta0, dtype=float))
    energy = float(E_theta(theta))
    lam = _LAMBDA_INIT
    n_accepted = 0
    guard_stopped = False
    stalled_gradient = False
    gnorm = float(jnp.linalg.norm(grad(theta)))

    for _ in range(max_iter):
        if gnorm < target:
            break
        H = np.asarray(hess(theta))
        H = 0.5 * (H + H.T)
        g = np.asarray(grad(theta), dtype=float)
        accepted = False
        for _try in range(_INNER_TRIES):
            try:
                step = np.linalg.solve(H + lam * np.eye(H.shape[0]), g)
            except np.linalg.LinAlgError:
                lam *= _LAMBDA_UP
                continue
            if not np.all(np.isfinite(step)) \
                    or np.linalg.norm(step) > _MAX_STEP_NORM:
                # The Newton model has stopped describing the function.
                lam *= _LAMBDA_UP
                continue
            candidate = theta - jnp.asarray(step)
            e_new = float(E_theta(candidate))
            if e_new <= energy + _ENERGY_SLACK:
                theta, energy = candidate, e_new
                lam = max(lam * _LAMBDA_DOWN, _LAMBDA_MIN)
                accepted = True
                n_accepted += 1
                break
            # The step was rejected; damp harder and try again.
            lam *= _LAMBDA_UP
        if not accepted:
            guard_stopped = True
            break

        g_new = float(jnp.linalg.norm(grad(theta)))
        # Lowering the energy is necessary but not sufficient. A function
        # with no stationary point (linear in some direction) lowers the
        # energy at every step, so the downhill rule alone lets the
        # parameters walk away while the gradient never shrinks -- this
        # project measured |theta| reaching 4e13 on such a surrogate.
        # The gradient is the convergence criterion; a step that does not
        # reduce it means the iteration is no longer approaching a
        # stationary point.
        if g_new >= gnorm * (1.0 - 1e-12):
            theta, energy, gnorm = theta, energy, g_new
            stalled_gradient = True
            break
        gnorm = g_new

    return StationaryPoint(theta=theta, energy=energy, grad_norm=gnorm,
                           n_accepted=n_accepted, guard_stopped=guard_stopped,
                           stalled_gradient=stalled_gradient)
