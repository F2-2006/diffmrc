"""Trust-region minimization with a Steihaug-CG subproblem.

WHY THIS MODULE EXISTS

The orbital-parameter (theta) optimizer in ``core.optimizer`` uses
L-BFGS-B, and ``core.stationary`` adds a damped Newton refinement. Both
read curvature only indirectly: L-BFGS infers it from gradient changes
and the damped Newton forces positive definiteness through a damping
ladder. On near-degenerate surfaces neither is the right tool.

* L-BFGS has no model of negative curvature, so it can stop where the
  gradient vanishes and the curvature is negative. The project measured
  such a point on the H2O/6-31g theta surface: |g| = 3.9e-14 with three
  negative Hessian eigenvalues (smallest -1.58e-2). The gradient
  criterion alone cannot distinguish it from a minimum.
* The damped Newton refinement cannot leave such a point either: at a
  stationary point the Newton step ``-H^-1 g`` is zero, and negative
  curvature changes only the step direction, not its size. The same
  refinement also stalled on the lower H2O basin at |g| ~ 4e-4 when its
  step cap was reached.

A trust-region method addresses both. The step minimizes the local model
inside a ball whose radius adapts to the measured agreement between model
and function, and a direction of non-positive curvature is followed to
the ball's boundary -- where the model is lowest along that ray. The
acceptance test uses the measured energy, so a wrong curvature model can
cost iterations but cannot raise the energy. That property is required
here: an unguarded refinement in this project once walked the energy from
-76.07 to +1.04 Ha while reporting no error.

WHAT THIS MODULE PROVIDES

``trust_region_minimize(fun, grad, x0, hvp=None, ...)``
    The outer loop. With an HVP callable it runs the iteration described
    above; without one it falls back to scipy's L-BFGS-B on the same
    ``fun``/``grad``, reported through the same result type, so a caller
    can compare stages without glue code.

``steihaug_cg(g, matvec, radius, ...)``
    The subproblem solver, exposed for tests and for callers that want
    the step without the outer loop.

CONVENTIONS

* ``fun(x, *args) -> float``, ``grad(x, *args) -> array`` and
  ``hvp(x, v, *args) -> array``. The HVP argument order matches
  ``driver.orbitals.hessp_of`` and scipy's ``hessp``; a JAX
  ``jvp`` of the gradient plugs in directly.
* ``x`` is a 1-D float64 array. The trust region uses the Euclidean
  norm, the same chart the theta parameters live in.
* Only steps that lower the measured energy are taken: the ratio test
  must clear ``eta`` *and* ``fun(x + p) < fun(x)``. A rejected step
  leaves the point and the gradient untouched.
* ``args`` travel to every call as positional trailing arguments. Large
  reference arrays must go through ``args`` rather than a closure, or a
  jitted caller will embed them in the traced graph (the contract of
  ``optimizer.optimize_theta_scipy``).
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = ['SteihaugStatus', 'SteihaugStep', 'TrustRegionStop',
           'TrustRegionResult', 'steihaug_cg', 'trust_region_minimize']

# Nocedal & Wright, Numerical Optimization, Algorithm 4.1. A step is
# accepted when the agreement ratio clears eta; the radius shrinks when
# the ratio is poor and grows only when the step reached the boundary
# and the model was accurate there.
_RHO_SHRINK = 0.25
_RHO_GROW = 0.75
# Default CG budget. One HVP costs about one gradient evaluation, so a
# deep subproblem solve is not free; an inexact step is enough because
# the outer ratio test corrects it. ``core.stationary`` settled on 40 for
# the same reason (measured 5-8 h of HVP dispatch at 100 iterations on
# the M5-C batch).
_DEFAULT_CG_MAXITER = 50


class SteihaugStatus(Enum):
    """Why the truncated CG iteration returned."""

    ZERO_GRADIENT = 'zero_gradient'
    CONVERGED = 'converged'
    NEGATIVE_CURVATURE = 'negative_curvature'
    RADIUS_BOUNDARY = 'radius_boundary'
    ITERATION_CAP = 'iteration_cap'
    NONFINITE = 'nonfinite'


class TrustRegionStop(Enum):
    """Why the outer loop returned.

    The names carry the same distinction as ``core.stationary.StopReason``:
    ``CONVERGED`` means the gradient reached the target; the other
    outcomes say the loop ended for a different reason and must not be
    read as convergence.
    """

    CONVERGED = 'converged'
    RADIUS_TOO_SMALL = 'radius_too_small'
    ITERATION_CAP = 'iteration_cap'


@dataclass
class SteihaugStep:
    """One approximate solution of the trust-region subproblem.

    ``p`` minimizes the model approximately inside the ball, ``Hp`` is
    the model curvature applied to it (carried along incrementally so the
    caller does not pay an extra HVP), and ``iterations`` counts the
    Hessian-vector products that were spent.
    """

    p: np.ndarray
    Hp: np.ndarray
    iterations: int
    status: SteihaugStatus

    @property
    def boundary_hit(self):
        """True when the step lies on the trust-region boundary."""
        return self.status in (SteihaugStatus.NEGATIVE_CURVATURE,
                               SteihaugStatus.RADIUS_BOUNDARY)


@dataclass
class TrustRegionResult:
    """The point, the evidence, and the cost of reaching it.

    ``energy_history`` holds the accepted energies only (one entry per
    accepted step), so a caller can check monotone descent without
    re-evaluating the function. ``radius_history``, ``step_norms``,
    ``rho_history`` and ``cg_iterations`` have one entry per subproblem
    solve, including rejected ones.
    """

    x: np.ndarray
    fun: float
    grad_norm: float
    n_iter: int = 0
    n_accept: int = 0
    n_reject: int = 0
    radius: float = 0.0
    stop_reason: TrustRegionStop = TrustRegionStop.ITERATION_CAP
    used_hessian: bool = True
    n_fun: int = 0
    n_grad: int = 0
    n_hvp: int = 0
    n_cg: int = 0
    energy_history: list = field(default_factory=list)
    radius_history: list = field(default_factory=list)
    rho_history: list = field(default_factory=list)
    step_norms: list = field(default_factory=list)
    cg_iterations: list = field(default_factory=list)


def _boundary_step(p, d, radius):
    """Distance tau along ``d`` that puts ``p + tau*d`` on the sphere.

    Solves ``||p + tau d||^2 = radius^2`` for the non-negative root.
    With ``p = 0`` this reduces to ``radius / ||d||``.
    """
    dd = float(d @ d)
    if dd <= 0.0:
        return 0.0
    pd = float(p @ d)
    pp = float(p @ p)
    disc = pd * pd + dd * (radius * radius - pp)
    return (-pd + np.sqrt(max(disc, 0.0))) / dd


def steihaug_cg(g, matvec, radius, cg_tol=0.1, maxiter=None):
    """Approximately minimize ``g.p + 0.5 p.Hp`` subject to ``||p|| <= radius``.

    Steihaug's truncated conjugate-gradient method. Two exits differ
    from ordinary CG:

    * a direction with ``d.Hd <= 0`` means the model is unbounded below
      along it, so the best point in the ball on that ray is the
      boundary (``p + tau d`` with ``||p + tau d|| = radius``);
    * an iterate that would leave the ball is projected onto its
      boundary by the same rule.

    The iteration stops early when the residual falls below
    ``cg_tol * ||g||`` (an inexact Newton step; Eisenstat-Walker style),
    or after ``maxiter`` products.

    Parameters
    ----------
    g : array_like
        Gradient at the expansion point; the model's linear term.
    matvec : callable
        ``matvec(v) -> H v``.
    radius : float
        Trust-region radius.
    cg_tol : float
        Relative residual target.
    maxiter : int, optional
        HVP budget; ``min(g.size, 50)`` when omitted.

    Returns
    -------
    SteihaugStep
    """
    g = np.asarray(g, dtype=float).ravel()
    n = g.size
    radius = float(radius)
    p = np.zeros(n)
    Hp = np.zeros(n)
    r = g.copy()
    d = -r
    rs = float(r @ r)
    gnorm = np.sqrt(rs)
    if n == 0 or gnorm == 0.0:
        return SteihaugStep(p=p, Hp=Hp, iterations=0,
                            status=SteihaugStatus.ZERO_GRADIENT)
    if radius <= 0.0:
        return SteihaugStep(p=p, Hp=Hp, iterations=0,
                            status=SteihaugStatus.RADIUS_BOUNDARY)
    if maxiter is None:
        maxiter = min(n, _DEFAULT_CG_MAXITER)
    target = float(cg_tol) * gnorm

    for it in range(1, int(maxiter) + 1):
        if np.sqrt(rs) <= target:
            return SteihaugStep(p=p, Hp=Hp, iterations=it - 1,
                                status=SteihaugStatus.CONVERGED)
        Hd = np.asarray(matvec(d), dtype=float).ravel()
        if Hd.size != n or not np.all(np.isfinite(Hd)):
            return SteihaugStep(p=p, Hp=Hp, iterations=it,
                                status=SteihaugStatus.NONFINITE)
        dHd = float(d @ Hd)
        if dHd <= 0.0:
            tau = _boundary_step(p, d, radius)
            return SteihaugStep(p=p + tau * d, Hp=Hp + tau * Hd,
                                iterations=it,
                                status=SteihaugStatus.NEGATIVE_CURVATURE)
        alpha = rs / dHd
        p_new = p + alpha * d
        if np.linalg.norm(p_new) >= radius:
            tau = _boundary_step(p, d, radius)
            return SteihaugStep(p=p + tau * d, Hp=Hp + tau * Hd,
                                iterations=it,
                                status=SteihaugStatus.RADIUS_BOUNDARY)
        p = p_new
        Hp = Hp + alpha * Hd
        r = r + alpha * Hd
        rs_new = float(r @ r)
        if np.sqrt(rs_new) <= target:
            return SteihaugStep(p=p, Hp=Hp, iterations=it,
                                status=SteihaugStatus.CONVERGED)
        d = -r + (rs_new / rs) * d
        rs = rs_new

    return SteihaugStep(p=p, Hp=Hp, iterations=int(maxiter),
                        status=SteihaugStatus.ITERATION_CAP)


def _fallback_lbfgs(fun, grad, x0, args, gtol, maxiter):
    """L-BFGS-B on the same callables, for the no-HVP comparison path."""
    import scipy.optimize as sopt

    counts = {'fun': 0, 'grad': 0}

    def f_np(x):
        counts['fun'] += 1
        return float(fun(np.asarray(x, dtype=float), *args))

    def g_np(x):
        counts['grad'] += 1
        return np.asarray(grad(np.asarray(x, dtype=float), *args),
                          dtype=float).ravel()

    e0 = f_np(x0)
    if not np.isfinite(e0):
        raise ValueError(f'initial function value is not finite: {e0}')
    res = sopt.minimize(
        f_np, x0, jac=g_np, method='L-BFGS-B',
        options=dict(maxiter=int(maxiter), gtol=float(gtol),
                     ftol=1e-14, maxls=50))
    x = np.asarray(res.x, dtype=float)
    g = g_np(x)
    gnorm = float(np.linalg.norm(g))
    stop = (TrustRegionStop.CONVERGED if gnorm <= gtol
            else TrustRegionStop.ITERATION_CAP)
    return TrustRegionResult(
        x=x, fun=float(res.fun), grad_norm=gnorm,
        n_iter=int(getattr(res, 'nit', -1)),
        stop_reason=stop, used_hessian=False,
        n_fun=counts['fun'], n_grad=counts['grad'])


def trust_region_minimize(fun, grad, x0, hvp=None, args=(),
                          gtol=1e-8, maxiter=200,
                          initial_radius=1.0, max_radius=10.0,
                          min_radius=None, eta=0.15,
                          radius_up=2.0, radius_down=0.25,
                          cg_tol=0.1, cg_maxiter=None):
    """Minimize a differentiable function with an adaptive trust region.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args) -> float``.
    grad : callable
        ``grad(x, *args) -> array``.
    x0 : array_like
        Starting point, 1-D.
    hvp : callable, optional
        ``hvp(x, v, *args) -> H v``. When omitted the function runs
        scipy L-BFGS-B instead and reports ``used_hessian = False``;
        this is the comparison path, not an approximation to the
        Newton model.
    args : tuple
        Extra positional arguments for all three callables.
    gtol : float
        Stop when ``||grad|| <= gtol``.
    maxiter : int
        Subproblem-solve budget (rejected steps count).
    initial_radius, max_radius : float
        Trust-region controls. A step never exceeds ``max_radius``.
    min_radius : float, optional
        Floor at which the loop stops with ``RADIUS_TOO_SMALL``;
        defaults to ``1e-12 * max(1, ||x0||)``.
    eta : float
        Minimum agreement ratio for accepting a step. The measured
        energy must also decrease.
    radius_up, radius_down : float
        Multipliers applied when agreement is good at the boundary, or
        poor.
    cg_tol : float
        Relative residual target of the inner CG.
    cg_maxiter : int, optional
        Inner HVP budget; ``min(n, 50)`` when omitted.

    Returns
    -------
    TrustRegionResult

    Raises
    ------
    ValueError
        If ``x0`` is not 1-D, or the initial energy or gradient is not
        finite. A broken chain should fail at the start, not after
        paying for iterations.
    """
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1:
        raise ValueError(f'x0 must be one-dimensional, got shape {x.shape}')
    x = x.copy()
    n = x.size
    if n == 0:
        e0 = float(fun(x, *args))
        return TrustRegionResult(
            x=x, fun=e0, grad_norm=0.0, n_iter=0,
            stop_reason=TrustRegionStop.CONVERGED, n_fun=1)

    if hvp is None:
        return _fallback_lbfgs(fun, grad, x, args, gtol, maxiter)

    counts = {'fun': 0, 'grad': 0, 'hvp': 0}

    def f_np(z):
        counts['fun'] += 1
        val = float(fun(np.asarray(z, dtype=float), *args))
        return val

    def g_np(z):
        counts['grad'] += 1
        return np.asarray(grad(np.asarray(z, dtype=float), *args),
                          dtype=float).ravel()

    def mv(z, v):
        counts['hvp'] += 1
        return np.asarray(hvp(np.asarray(z, dtype=float),
                              np.asarray(v, dtype=float), *args),
                          dtype=float).ravel()

    energy = f_np(x)
    if not np.isfinite(energy):
        raise ValueError(f'initial function value is not finite: {energy}')
    g = g_np(x)
    if g.size != n:
        raise ValueError(f'gradient has shape {g.shape}, expected ({n},)')
    if not np.all(np.isfinite(g)):
        raise ValueError('initial gradient is not finite')
    gnorm = float(np.linalg.norm(g))

    radius = float(initial_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f'initial_radius must be positive, got {radius}')
    if max_radius is None:
        max_radius = np.inf
    radius = min(radius, float(max_radius))
    if min_radius is None:
        min_radius = 1e-12 * max(1.0, float(np.linalg.norm(x)))
    min_radius = float(min_radius)

    res = TrustRegionResult(x=x, fun=energy, grad_norm=gnorm, radius=radius)
    stop = TrustRegionStop.ITERATION_CAP

    for _ in range(int(maxiter)):
        if gnorm <= gtol:
            stop = TrustRegionStop.CONVERGED
            break

        step = steihaug_cg(g, lambda v: mv(x, v), radius,
                           cg_tol=cg_tol, maxiter=cg_maxiter)
        res.n_iter += 1
        res.n_cg += step.iterations
        res.radius_history.append(radius)
        res.step_norms.append(float(np.linalg.norm(step.p)))
        res.cg_iterations.append(step.iterations)

        p, Hp = step.p, step.Hp
        predicted = -float(g @ p + 0.5 * p @ Hp)
        finite_step = bool(np.all(np.isfinite(p)) and np.all(np.isfinite(Hp)))
        if step.status is SteihaugStatus.ZERO_GRADIENT:
            stop = TrustRegionStop.CONVERGED
            break
        if not finite_step or float(np.linalg.norm(p)) == 0.0:
            res.rho_history.append(float('-inf'))
            radius = max(radius * radius_down, min_radius)
            res.radius = radius
            if radius <= min_radius:
                stop = TrustRegionStop.RADIUS_TOO_SMALL
                break
            continue

        candidate = x + p
        e_new = f_np(candidate)
        actual = energy - e_new
        rho = (actual / predicted
               if predicted > 0.0 and np.isfinite(e_new) else float('-inf'))
        res.rho_history.append(float(rho))
        accepted = bool(np.isfinite(e_new) and predicted > 0.0
                        and actual > 0.0 and rho > eta)

        if accepted:
            x = candidate
            energy = e_new
            g = g_np(x)
            gnorm = float(np.linalg.norm(g))
            res.n_accept += 1
            res.energy_history.append(energy)
        else:
            res.n_reject += 1

        if rho < _RHO_SHRINK:
            radius = max(radius * radius_down, min_radius)
        elif rho > _RHO_GROW and step.boundary_hit:
            radius = min(radius * radius_up, float(max_radius))
        res.radius = radius

        if radius <= min_radius:
            stop = (TrustRegionStop.CONVERGED if gnorm <= gtol
                    else TrustRegionStop.RADIUS_TOO_SMALL)
            break
    else:
        stop = (TrustRegionStop.CONVERGED if gnorm <= gtol
                else TrustRegionStop.ITERATION_CAP)

    res.x = x
    res.fun = energy
    res.grad_norm = gnorm
    res.stop_reason = stop
    res.n_fun = counts['fun']
    res.n_grad = counts['grad']
    res.n_hvp = counts['hvp']
    return res
