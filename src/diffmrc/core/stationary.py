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
from enum import Enum

import numpy as np
import jax

__all__ = ['StationaryPoint', 'StopReason', 'CurvatureReport',
           'refine_stationary_point', 'curvature_report']

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
# Newton-CG controls: the damped-Newton step solves (H + lam I) x = g by
# conjugate gradients on Hessian-vector products.  The explicit
# ``jax.hessian`` alternative is fwd-over-fwd and broadcasts every
# intermediate over npar^2 tangents -- measured on the M5-C batch, it
# died on a 402 GB allocation at npar ~ 500 (nmo 42-48).  The HVP route
# carries one tangent per reverse pass; the same exact Hessian.  The CG
# budget is deliberately shallow: an inexact Newton step only has to
# feed the damping ladder and the downhill acceptance, which correct it;
# measured on the M5-C batch, a deep solver (tol 1e-10, 100 iters) put
# single molecules at 5-8 h of HVP dispatch.
_CG_TOL = 1e-6
_CG_MAXITER = 40


class StopReason(Enum):
    """Why the refinement loop ended.

    These are mutually exclusive outcomes, and they call for different
    follow-up. In particular ``CONVERGED`` and ``ITERATION_CAP`` are both
    "the loop finished"; only the first means the point is stationary.
    Reporting them as two booleans let a run exhaust its budget with a
    gradient of order 1e3 and set neither flag -- measured on an
    ill-conditioned quadratic started far from the solution, where 80
    accepted steps still left |g| = 4.0e3.

    This is the same shape as treating an optimizer's status code as a
    convergence signal, which this project has already been bitten by
    once.
    """

    CONVERGED = 'converged'
    GUARD_STOPPED = 'guard_stopped'
    STALLED_GRADIENT = 'stalled_gradient'
    ITERATION_CAP = 'iteration_cap'


@dataclass
class StationaryPoint:
    """A refined point, with the evidence that it is stationary.

    ``is_stationary`` is the only thing a caller should gate on. It is
    true when the gradient actually reached the target, not merely when
    the loop finished.
    """

    theta: object
    energy: float
    grad_norm: float
    n_accepted: int = 0
    stop_reason: StopReason = StopReason.ITERATION_CAP
    target: float = 1e-11

    @property
    def is_stationary(self):
        """True only when the gradient reached the target."""
        return bool(self.grad_norm < self.target)

    @property
    def guard_stopped(self):
        return self.stop_reason is StopReason.GUARD_STOPPED

    @property
    def stalled_gradient(self):
        return self.stop_reason is StopReason.STALLED_GRADIENT


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


def _cg_solve(matvec, b, tol=_CG_TOL, maxiter=_CG_MAXITER):
    """Solve ``matvec(x) = b`` by conjugate gradients.

    The Newton step of the refinement needs ``(H + lam I) x = g``; with
    the Hessian reaching this code only through Hessian-vector products,
    CG is the solver that never materializes it.  Positive-definiteness
    is guaranteed by the shift for lam >= _LAMBDA_MIN; when a direction
    is numerically flat the iteration stops rather than dividing by a
    vanishing denominator, and the caller's step-norm guard decides what
    to do with the result.
    """
    x = np.zeros_like(b)
    r = np.array(b, dtype=float)
    b_norm2 = float(r @ r)
    if b_norm2 == 0.0:
        return x
    p = r.copy()
    rs = b_norm2
    for _ in range(maxiter):
        Ap = np.asarray(matvec(p), dtype=float)
        denom = float(p @ Ap)
        if abs(denom) < 1e-300:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if rs_new <= (tol ** 2) * b_norm2:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def refine_stationary_point(E_theta, theta0, max_iter=80, target=1e-11):
    """Damped Newton refinement that only ever moves downhill.

    The Newton step solves ``(H + lam I) x = g`` by conjugate gradients
    on Hessian-vector products (``_cg_solve``); the Hessian itself is
    never materialized -- the explicit ``jax.hessian`` broadcast every
    intermediate over npar^2 tangents and was killed by a 402 GB
    allocation at npar ~ 500 on the M5-C batch.  The damping ladder, the
    step-norm guard and the downhill acceptance are unchanged.

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

    grad_plain = jax.grad(E_theta)
    grad = jax.jit(grad_plain)

    def _hvp_raw(theta, v):
        return jax.jvp(grad_plain, (theta,), (v,))[1]

    hvp = jax.jit(_hvp_raw)

    theta = jnp.asarray(np.asarray(theta0, dtype=float))
    energy = float(E_theta(theta))
    lam = _LAMBDA_INIT
    n_accepted = 0
    gnorm = float(jnp.linalg.norm(grad(theta)))
    # Default to the cap: it is the reason that needs no break to reach.
    reason = StopReason.ITERATION_CAP

    for _ in range(max_iter):
        if gnorm < target:
            reason = StopReason.CONVERGED
            break
        g = np.asarray(grad(theta), dtype=float)
        accepted = False
        for _try in range(_INNER_TRIES):
            def _matvec(v):
                return np.asarray(hvp(theta, jnp.asarray(v)),
                                  dtype=float) + lam * v

            step = _cg_solve(_matvec, g)
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
            reason = StopReason.GUARD_STOPPED
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
            reason = StopReason.STALLED_GRADIENT
            break
        gnorm = g_new
    else:
        # The loop ran to exhaustion without any break. Whether that left
        # us at a stationary point is decided by the gradient, not by the
        # absence of a break, so re-check before reporting CONVERGED.
        if gnorm < target:
            reason = StopReason.CONVERGED

    return StationaryPoint(theta=theta, energy=energy, grad_norm=gnorm,
                           n_accepted=n_accepted, stop_reason=reason,
                           target=float(target))
