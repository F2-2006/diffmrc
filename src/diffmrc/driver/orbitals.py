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

The single-stage route above is not the only entry point.
``converge_orbital_point_two_stage`` splits the parameter set by the
role of the rotations: the well-conditioned core-active and
active-virtual subset converges first, and a Newton refinement on the
full energy then removes the core-virtual component. The split exists
because measured on HI/def2-SVP CAS(2,2) the full L-BFGS-B stage
stalled at a gradient of 3.0e-4 while the subset reached 4.3e-8,
whereas dropping core-virtual from the model entirely leaves the point
unconverged in directions that shift the energy. ``embed_theta`` maps
the subset vector to the full list by pair identity, never by
position. The driver takes the split route when the core-virtual block
exceeds twice the remaining parameter count: that is the regime where
the joint L-BFGS stage is dominated by many large-curvature rotations
with small optimal values. Below it the single-stage route is kept, so
systems it already converges keep their previous numbers.

Both routes refine through a second derivative. When the analytic one
is not finite -- the blocked eigensolver's Hessian at an exact
degeneracy, while the energy and gradient are smooth -- the refinement
rebuilds it from symmetric differences of the analytic gradient
(``_finite_difference_newton``). Measured on N2/6-31G CAS(4,4) on
2026-11-01: the analytic Hessian was NaN in every entry, the point sat
at a gradient of 7.3e-5, and two gradient-difference Newton steps took
it to 8.8e-14 in 5.8 s, which is what lets the run meet its 1e-8
criterion without changing the criterion.
"""
import time
from dataclasses import dataclass

import numpy as np

from ..core.optimizer import (optimize_theta_hand, optimize_theta_scipy,
                              optimize_theta_trust_region)
from ..core.stationary import (StopReason, StationaryPoint,
                               _ENERGY_SLACK, _INNER_TRIES, _LAMBDA_DOWN,
                               _LAMBDA_INIT, _LAMBDA_MIN, _LAMBDA_UP,
                               _MAX_STEP_NORM, curvature_from_gradient,
                               refine_stationary_point)

__all__ = ['OrbitalPoint', 'TwoStagePoint', 'converge_orbital_point',
           'converge_orbital_point_two_stage', 'embed_theta', 'hessp_of',
           'optimize_theta_trust_region', 'optimizer_stage',
           'should_split_rotation_set',
           'OPTIMIZER_GRAD_TOL', 'REFINE_TARGET', 'SECOND_STAGE_FALLBACK_TOL',
           'SPLIT_CV_RATIO']

# The first-stage target. The measured floor of L-BFGS-B on this energy is
# 1.5e-5, so a tighter target keeps the stage iterating in a flat valley
# instead of handing over to the Newton refinement; measured on HI/def2-SVP,
# a 1e-6 target ran past 55 minutes while 1e-4 reached 4.3e-8 in 128 s.
OPTIMIZER_GRAD_TOL = 1e-4
REFINE_TARGET = 1e-11
MAX_STEPS = 2000
REFINE_MAX_ITER = 80

# The second stage of the two-stage route accepts the embedded first-stage
# point as converged when the full-space Newton refinement reaches this
# gradient. It is deliberately looser than REFINE_TARGET: the acceptance
# criterion for the end-to-end path is 1e-6, and the short L-BFGS-B fallback
# below only runs when the Newton stage cannot reach it.
SECOND_STAGE_FALLBACK_TOL = 1e-6
SECOND_STAGE_LBFGS_STEPS = 40

# The driver splits the rotation set when the core-virtual block is more
# than this multiple of the rest. Measured ratios: HI/def2-SVP CAS(2,2)
# 204/58 = 3.5 (the joint L-BFGS stage stalled and its gradient rose),
# H2O/6-31G CAS(2,2) 28/22 = 1.3 and N2/6-31G CAS(4,4) 45/56 = 0.8 (both
# converge without the split). The ratio is a property of the partition
# and is known before any energy is evaluated.
SPLIT_CV_RATIO = 2.0

# The refinement's second-order object is the analytic Hessian, reached as a
# jvp of the gradient. At an exact degeneracy that route is undefined while
# the energy and its gradient are smooth: measured on N2/6-31G CAS(4,4) on
# 2026-11-01, all 10201 entries of the analytic Hessian at the symmetric
# stationary point were NaN, the gradient there was 7.3e-5, and the point is
# a minimum (the gradient-difference curvature has zero negative
# eigenvalues). The fallback below builds the curvature from symmetric
# differences of the analytic gradient instead.
#
# Cost, measured on the same system (npar = 101, JAX on one CPU thread): one
# gradient 0.03 s, one Hessian 2.9-3.2 s (2 x 101 evaluations), the whole
# two-step refinement 5.8 s. The matrix is 101^2 float64 = 82 kB. The
# analytic route it replaces spent 63.7 s and produced no usable entry, so
# the fallback is cheaper as well as correct.
#
# The cap matches the curvature probe in ``driver/kernel.py``
# (CURVATURE_FD_MAX_PARAMETERS). Above it the fallback declines rather than
# spending minutes per Newton step on a rotation space this project has
# never refined to 1e-11; the caller keeps the optimizer's point and the run
# reports its gradient, exactly as it does when the curvature probe refuses.
FD_HESSIAN_MAX_PARAMETERS = 300
FD_HESSIAN_STEP = 1e-5
# One Hessian costs 2 x npar gradient evaluations, so the iteration cap is
# deliberately short. Measured on N2/6-31G from the L-BFGS-B point: two
# steps reach 8.8e-14. Even a start displaced by 1e-2 needs fewer than ten
# before it stalls at the floating-point floor of the energy.
FD_NEWTON_MAX_ITER = 20


def should_split_rotation_set(ncore, nact, nmo):
    """Whether the two-stage route is used for one orbital partition.

    The criterion is the size of the core-virtual block relative to the
    core-active and active-virtual blocks. The core-virtual rotations
    are the ones whose curvature is large and whose optimal values are
    small; when they dominate the parameter set, a joint L-BFGS-B stage
    spends its steps on them and can stall (HI/def2-SVP: 204 of 262
    parameters, gradient 3.0e-4 and rising).

    Parameters
    ----------
    ncore, nact, nmo : int
        Block sizes of the partition.

    Returns
    -------
    bool
        True when the core-virtual block exceeds ``SPLIT_CV_RATIO`` times
        the rest of the rotation set.
    """
    ncore, nact, nmo = int(ncore), int(nact), int(nmo)
    nvir = max(nmo - ncore - nact, 0)
    n_cv = ncore * nvir
    n_rest = ncore * nact + nact * nvir
    return n_cv > SPLIT_CV_RATIO * n_rest


def embed_theta(subset_pairs, full_pairs, theta_subset, dtype=float):
    """Lift a parameter vector from a pair subset to the full pair list.

    The parameter vector has one entry per rotation pair, and the two
    pair lists agree on the ordering *within* each block
    (core-active, core-virtual, active-virtual) but differ in which
    blocks are present. Slicing the vector by position would therefore
    scatter the active-virtual entries into core-virtual slots; this
    function maps by pair identity instead.

    Parameters
    ----------
    subset_pairs : sequence of (int, int)
        The pairs the subset vector was built on, in the order used to
        build it.
    full_pairs : sequence of (int, int)
        The target pair list.
    theta_subset : array_like
        Values, one per entry of `subset_pairs`.
    dtype : type
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Length ``len(full_pairs)``; pairs absent from the subset are zero.
    """
    full_index = {tuple(pair): k for k, pair in enumerate(full_pairs)}
    theta = np.asarray(theta_subset, dtype=dtype)
    if theta.ndim != 1 or theta.shape[0] != len(subset_pairs):
        raise ValueError(
            f'theta_subset has shape {theta.shape}, expected '
            f'({len(subset_pairs)},)')
    out = np.zeros(len(full_pairs), dtype=dtype)
    for k, pair in enumerate(subset_pairs):
        key = tuple(pair)
        if key not in full_index:
            raise ValueError(f'pair {key} is not in the full pair list')
        out[full_index[key]] = theta[k]
    return out


def hessp_of(E_theta):
    """Hessian-vector product of ``E_theta`` as a jitted callable.

    The trust-region polish consumes the second-order object through
    scipy's ``hessp`` interface.  ``jax.hessian`` on the same function is
    mathematically identical but its fwd-over-fwd contraction broadcasts
    every intermediate over the npar^2 tangent pairs; measured on the
    M5-C batch (npar ~ 500, nmo ~ 42-48), the first polish to fire was
    killed by a 402 GB allocation.  A ``jax.jvp`` of the gradient carries
    a single tangent vector through one reverse pass -- same exact
    Hessian, memory O(one vector) plus the jitted reverse graph.
    """
    import jax
    from jax import numpy as jnp

    g = jax.grad(E_theta)

    # The returned callable takes the optional runtime arrays after
    # (x, v): hvp(x, v, *arrays), matching the optimizer convention.
    @jax.jit
    def hvp(x, v, *arr):
        return jax.jvp(lambda t: g(t, *arr), (x,), (v,))[1]

    return hvp


def _finite_difference_newton(E_theta, theta0, target, max_iter=20,
                              arrays=(), step=FD_HESSIAN_STEP,
                              max_parameters=FD_HESSIAN_MAX_PARAMETERS):
    """Damped Newton refinement with a gradient-difference Hessian.

    This is the route taken when the analytic Hessian at the parameter
    point is not finite. The blocked eigensolver's second derivative is
    undefined at an exact degeneracy while the energy and its first
    derivative are smooth, so the curvature is rebuilt from symmetric
    differences of the analytic gradient -- the same construction
    :func:`diffmrc.core.stationary.curvature_from_gradient` applies for
    the curvature probe, and the standard second derivative of a smooth
    function.

    Everything else is the refinement's own rule, and it is imported
    rather than retyped: the damping ladder
    (``_LAMBDA_INIT``/``_LAMBDA_MIN``/``_LAMBDA_UP``/``_LAMBDA_DOWN``),
    the step-norm cap, the energy slack, the inner try budget and the
    stall test. A step is accepted only when it does not raise the energy
    and the following gradient is strictly lower; a step that lowers the
    energy without lowering the gradient stops the iteration.

    Parameters
    ----------
    E_theta : callable
        Differentiable energy of the parameter vector.
    theta0 : array_like
        Starting point, normally the optimizer's output.
    target : float
        Stop when the gradient norm falls below this.
    max_iter : int
        Iteration cap; further capped by ``FD_NEWTON_MAX_ITER`` because
        each iteration rebuilds a dense Hessian.
    arrays : tuple
        Extra runtime arguments appended to ``E_theta(theta, *arrays)``.
    step : float
        Symmetric-difference step, in theta units.
    max_parameters : int
        Refuse above this dimension.

    Returns
    -------
    (ndarray, float, float) or None
        ``(theta, energy, grad_norm)`` when the gradient improved, None
        when the dimension is above the cap, the analytic gradient is
        not finite, or no step was accepted.
    """
    import jax
    from jax import numpy as jnp

    x = np.asarray(theta0, dtype=float).reshape(-1)
    n = x.size
    if n > int(max_parameters):
        return None

    E_jit = jax.jit(E_theta)
    grad_plain = jax.grad(E_theta)
    grad_jit = jax.jit(grad_plain)

    def energy_at(t):
        return float(E_jit(jnp.asarray(t), *arrays))

    def gradient_at(t):
        return np.asarray(grad_jit(jnp.asarray(t), *arrays), dtype=float)

    energy = energy_at(x)
    g = gradient_at(x)
    if not (np.isfinite(energy) and np.all(np.isfinite(g))):
        return None
    gnorm = float(np.linalg.norm(g))
    gnorm0 = gnorm
    lam = _LAMBDA_INIT
    eye = np.eye(n)
    accepted_any = False

    for _ in range(min(int(max_iter), FD_NEWTON_MAX_ITER)):
        if gnorm < target:
            break
        hessian = curvature_from_gradient(
            gradient_at, x, step=step, max_parameters=max_parameters)
        accepted = False
        for _try in range(_INNER_TRIES):
            try:
                delta = np.linalg.solve(hessian + lam * eye, g)
            except np.linalg.LinAlgError:
                lam *= _LAMBDA_UP
                continue
            if (not np.all(np.isfinite(delta))
                    or np.linalg.norm(delta) > _MAX_STEP_NORM):
                lam *= _LAMBDA_UP
                continue
            candidate = x - delta
            e_new = energy_at(candidate)
            if e_new <= energy + _ENERGY_SLACK:
                x, energy = candidate, e_new
                lam = max(lam * _LAMBDA_DOWN, _LAMBDA_MIN)
                accepted = True
                accepted_any = True
                break
            lam *= _LAMBDA_UP
        if not accepted:
            break
        g_new = gradient_at(x)
        gn_new = float(np.linalg.norm(g_new))
        if gn_new >= gnorm * (1.0 - 1e-12):
            gnorm = gn_new
            break
        g, gnorm = g_new, gn_new

    if not accepted_any and gnorm >= gnorm0:
        return None
    return x, float(energy), float(gnorm)


def _trust_region_polish(E_theta, theta0, target, max_iter=200, arrays=()):
    """A trust-region step from a point the two stages could not settle.

    Measured on H2O/6-31g's lower basin (2026-09-22): the two-stage
    protocol stalls at |g| = 3.7e-4 -- the damped Newton caps a step at
    10 and its stall rule gives up -- while trust-ncg with the exact
    Hessian reaches 9.7e-9 at the same energy and 8.8e-8 Eh lower. The
    stall was the protocol's, not the chart's.

    The Hessian reaches scipy as ``hessp`` (``hessp_of``: a jvp of the
    gradient) rather than as a materialized ``jax.hessian``; the two are
    the same object, but the fwd-over-fwd form broadcasts every
    intermediate over npar^2 tangents and died on a 402 GB allocation
    at npar ~ 500 in the M5-C batch.

    Returns ``(theta, energy, grad_norm)`` or None when scipy cannot
    start; also None when the gradient did not improve, so the caller
    keeps the point it had.

    When the analytic Hessian is not finite at ``theta0`` -- the exact
    degeneracy documented beside ``FD_HESSIAN_MAX_PARAMETERS`` -- the
    second-order object is unusable and the same point is refined by
    :func:`_finite_difference_newton` instead. The probe is one
    Hessian-vector product with a vector of ones: the Hessian is
    symmetric, so a non-finite entry in row i makes ``(H @ 1)[i]``
    non-finite. A finite probe keeps the existing trust-ncg path
    unchanged.
    """
    import scipy.optimize as sopt

    import jax
    from jax import numpy as jnp

    E_jit = jax.jit(E_theta)
    g_jit = jax.jit(jax.grad(E_theta))
    hvp_jit = hessp_of(E_theta)
    x0 = np.asarray(theta0, dtype=float)

    probe = np.asarray(hvp_jit(jnp.asarray(x0),
                               jnp.ones_like(jnp.asarray(x0)),
                               *arrays), dtype=float)
    if not np.all(np.isfinite(probe)):
        improved = _finite_difference_newton(
            E_theta, x0, target, max_iter=max_iter, arrays=arrays)
        if improved is None:
            return None
        g0 = float(np.linalg.norm(
            np.asarray(g_jit(jnp.asarray(x0), *arrays), dtype=float)))
        if not np.isfinite(g0) or improved[2] >= g0:
            return None
        return improved

    def fun(x):
        return float(E_jit(jnp.asarray(x), *arrays))

    def jac(x):
        return np.asarray(g_jit(jnp.asarray(x), *arrays), dtype=float)

    def hessp(x, p):
        return np.asarray(hvp_jit(jnp.asarray(x), jnp.asarray(p), *arrays),
                          dtype=float)

    try:
        res = sopt.minimize(fun, x0, jac=jac, hessp=hessp,
                            method='trust-ncg',
                            options=dict(gtol=float(target),
                                         maxiter=int(max_iter)))
    except Exception:
        return None
    x = np.asarray(res.x, dtype=float)
    g = float(np.linalg.norm(np.asarray(g_jit(jnp.asarray(x), *arrays),
                                        dtype=float)))
    if not (np.all(np.isfinite(x)) and np.isfinite(g)
            and np.isfinite(float(res.fun))):
        # A finite probe can still be followed by a non-finite step; the
        # caller must not receive a point that cannot be compared.
        return None
    return x, float(res.fun), g


@dataclass
class OrbitalPoint:
    """The refined point, with both stages' evidence kept.

    ``grad_norm`` is the final gradient; ``optimizer_grad_norm`` is what
    the optimizer stopped at, before refinement. ``optimizer_reached``
    says whether the loose stage met its own target -- a run that misses
    it is not a failure, it is the case refinement exists for.
    ``optimizer_name`` says which stage ran: ``'lbfgs-b'`` is the
    default path, ``'trust-region'`` is opt-in.
    """

    theta: object
    energy: float
    grad_norm: float
    optimizer_grad_norm: float
    optimizer_message: str
    optimizer_nit: int
    optimizer_reached: bool
    refinement: object
    polished: bool = False
    optimizer_name: str = 'lbfgs-b'

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


def optimizer_stage(E_theta, npar, theta0=None,
                    optimize_tol=OPTIMIZER_GRAD_TOL,
                    max_steps=MAX_STEPS, arrays=(),
                    use_trust_region=False):
    """The loose first stage alone, with its own verdict.

    Split out of :func:`converge_orbital_point` so the driver can ask
    "does the full parameter set converge at this level?" before paying
    for the refinement, and then hand the same result back instead of
    running it twice. The substitution order -- trust region, scipy
    L-BFGS-B, handwritten -- is unchanged.

    Returns
    -------
    dict
        ``E_jit``, ``theta``, ``grad_norm``, ``info``, ``name`` and
        ``reached`` (``grad_norm <= optimize_tol``).
    """
    name = 'lbfgs-b'
    E_jit = None
    if use_trust_region:
        E_jit, theta, g_after, info = optimize_theta_trust_region(
            E_theta, npar, steps=max_steps, g_conv=optimize_tol,
            theta0=theta0, arrays=arrays)
        if E_jit is not None:
            name = 'trust-region'
    if E_jit is None:
        info = {}
        E_jit, theta, g_after, info = optimize_theta_scipy(
            E_theta, npar, steps=max_steps, g_conv=optimize_tol,
            theta0=theta0, arrays=arrays)
    if E_jit is None:
        # scipy raised; the handwritten fallback keeps the run alive.
        E_jit, theta, g_after, info = optimize_theta_hand(
            E_theta, npar, steps=max_steps, g_conv=optimize_tol,
            theta0=theta0, arrays=arrays)
    return dict(E_jit=E_jit, theta=theta, grad_norm=float(g_after),
                info=info, name=name,
                reached=bool(float(g_after) <= optimize_tol))


def converge_orbital_point(E_theta, npar, theta0=None,
                           optimize_tol=OPTIMIZER_GRAD_TOL,
                           refine_target=REFINE_TARGET,
                           max_steps=MAX_STEPS,
                           refine_max_iter=REFINE_MAX_ITER, arrays=(),
                           use_trust_region=False):
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
    arrays : tuple
        Extra runtime arguments appended to every ``E_theta(theta, *arrays)``
        call (see ``optimizer.optimize_theta_scipy``); pass the AO
        integrals here, not through the closure.
    use_trust_region : bool
        When True the first stage is ``optimize_theta_trust_region``
        (Newton-CG with an adaptive trust region) instead of L-BFGS-B.
        The refinement and polish stages are unchanged. The default stays
        L-BFGS-B, so existing results are reproduced bit for bit.

    Returns
    -------
    OrbitalPoint
    """
    stage = optimizer_stage(
        E_theta, npar, theta0=theta0, optimize_tol=optimize_tol,
        max_steps=max_steps, arrays=arrays,
        use_trust_region=use_trust_region)
    E_jit = stage['E_jit']
    theta = stage['theta']
    g_after = stage['grad_norm']
    info = stage['info']
    optimizer_name = stage['name']

    sp = refine_stationary_point(E_theta, theta, max_iter=refine_max_iter,
                                 target=refine_target, arrays=arrays)
    polished = False
    if not sp.is_stationary:
        # The two stages stopped above the target: the chart may still
        # hold a stationary point, and a method without the step cap can
        # find it. Take another pass while the gradient keeps improving
        # -- measured on H2O/6-31g, the first pass reaches 4.6e-8 and a
        # second is what closes the last factor.
        th, en, gn = sp.theta, float(sp.energy), float(sp.grad_norm)
        for _ in range(3):
            improved = _trust_region_polish(E_theta, th, refine_target,
                                            max_iter=2 * refine_max_iter,
                                            arrays=arrays)
            if improved is None or improved[2] >= gn:
                break
            th, en, gn = improved
            polished = True
            if gn < refine_target:
                break
        if polished:
            sp = StationaryPoint(
                theta=th, energy=en, grad_norm=gn,
                n_accepted=sp.n_accepted,
                stop_reason=(StopReason.CONVERGED
                             if gn < refine_target
                             else StopReason.STALLED_GRADIENT),
                target=float(refine_target))
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
        polished=polished,
        optimizer_name=optimizer_name,
    )


@dataclass
class TwoStagePoint:
    """The full-space point reached through a two-stage optimization.

    ``point`` is the outcome of the second stage, and the properties
    below forward to it, so call sites that consumed an ``OrbitalPoint``
    keep working. ``first_stage`` keeps the constrained point, and
    ``embedded_theta`` is the full-length vector it mapped to -- the
    starting point the second stage refined. ``stages_report`` renders
    the whole record for ``algorithm_data``.
    """

    point: object
    first_stage: object
    embedded_theta: object
    stage1_seconds: float
    stage2_seconds: float
    second_optimizer: str = 'newton'
    fallback_used: bool = False
    newton_gradient: float = float('nan')
    newton_stop_reason: str = ''
    embedded_gradient: float = float('nan')

    @property
    def theta(self):
        return self.point.theta

    @property
    def energy(self):
        return self.point.energy

    @property
    def grad_norm(self):
        return self.point.grad_norm

    @property
    def is_stationary(self):
        return bool(self.point.is_stationary)

    @property
    def stop_reason(self):
        return self.point.stop_reason

    @property
    def target(self):
        return self.point.target

    @property
    def n_accepted(self):
        return self.point.n_accepted

    @property
    def polished(self):
        return self.point.polished

    @property
    def refinement(self):
        return self.point.refinement

    @property
    def optimizer_grad_norm(self):
        """The first stage's optimizer gradient, not the final one."""
        return self.first_stage.optimizer_grad_norm

    @property
    def optimizer_message(self):
        return self.first_stage.optimizer_message

    @property
    def optimizer_nit(self):
        return self.first_stage.optimizer_nit

    @property
    def optimizer_reached(self):
        return self.first_stage.optimizer_reached

    @property
    def optimizer_name(self):
        return self.first_stage.optimizer_name

    def stages_report(self):
        """A JSON-ready record of both stages."""
        first = self.first_stage
        second = self.point
        return dict(
            n_parameters=dict(subset=int(len(first.theta)),
                              full=int(len(self.embedded_theta))),
            stage1=dict(
                gradient=float(first.optimizer_grad_norm),
                energy=float(first.energy),
                gradient_after_refinement=float(first.grad_norm),
                stop_reason=str(getattr(first.stop_reason, 'value',
                                        first.stop_reason)),
                optimizer=first.optimizer_name,
                optimizer_message=str(first.optimizer_message),
                optimizer_iterations=int(first.optimizer_nit),
                reached=bool(first.optimizer_reached),
                seconds=float(self.stage1_seconds),
            ),
            stage2=dict(
                gradient=float(second.grad_norm),
                energy=float(second.energy),
                energy_change=float(second.energy - first.energy),
                stop_reason=str(getattr(second.stop_reason, 'value',
                                        second.stop_reason)),
                optimizer=self.second_optimizer,
                newton_gradient=self.newton_gradient,
                newton_stop_reason=self.newton_stop_reason,
                embedded_gradient=self.embedded_gradient,
                polished=bool(second.polished),
                fallback_used=bool(self.fallback_used),
                iterations=int(second.n_accepted),
                seconds=float(self.stage2_seconds),
            ),
        )


def converge_orbital_point_two_stage(E_subset, npar_subset, E_full,
                                     npar_full, subset_pairs, full_pairs,
                                     theta0=None,
                                     optimize_tol=OPTIMIZER_GRAD_TOL,
                                     refine_target=REFINE_TARGET,
                                     max_steps=MAX_STEPS,
                                     refine_max_iter=REFINE_MAX_ITER,
                                     arrays=(), use_trust_region=False,
                                     second_max_iter=None):
    """Converge on a rotation subset, then polish on the full set.

    WHY THIS EXISTS

    The core-virtual rotations make the full parameter set ill
    conditioned: measured on HI/def2-SVP CAS(2,2), the full L-BFGS-B
    stage stopped at a gradient of 3.0e-4 after 30 iterations and its
    gradient rose from 5.4e-3 to 5.6e-2 while the line search thrashed,
    whereas the 58-parameter subset (core-active and active-virtual
    only) reached 4.3e-8 in 128 s. Dropping core-virtual entirely is not
    an option -- an unconverged point in those directions changes the
    reported energy on pseudopotential systems -- so the split is by
    role: the well-conditioned subset decides the basin, and a Newton
    refinement on the full energy removes the core-virtual gradient.

    The first stage is :func:`converge_orbital_point` on the subset; the
    second starts from the embedded vector (core-virtual entries zero)
    and runs the damped Newton refinement on the full energy, followed
    by the same trust-region polish as the single-stage route when the
    target is not reached. A short L-BFGS-B pass and a second Newton
    refinement are used only if the gradient is still above
    ``SECOND_STAGE_FALLBACK_TOL``; the better point is kept.

    The driver selects this route by the size of the core-virtual block
    (see :func:`should_split_rotation_set`), not by a trial run: the
    block is the source of the ill conditioning, and its size is known
    before any energy is evaluated.

    Parameters
    ----------
    E_subset, npar_subset, E_full, npar_full :
        The two energies and their parameter counts.
    subset_pairs, full_pairs : sequence of (int, int)
        The pair lists the two parameter vectors were built on. The
        subset must be contained in the full list.
    theta0 : array_like, optional
        Starting point on the subset parameter vector, forwarded to the
        first stage.
    optimize_tol, refine_target, max_steps, refine_max_iter, arrays,
    use_trust_region :
        Forwarded to :func:`converge_orbital_point`.
    second_max_iter : int, optional
        Newton iteration cap for the second stage; defaults to
        ``refine_max_iter``.

    Returns
    -------
    TwoStagePoint
    """
    started = time.perf_counter()
    first = converge_orbital_point(
        E_subset, npar_subset, theta0=theta0, optimize_tol=optimize_tol,
        refine_target=refine_target, max_steps=max_steps,
        refine_max_iter=refine_max_iter, arrays=arrays,
        use_trust_region=use_trust_region)
    stage1_seconds = time.perf_counter() - started

    theta_full = embed_theta(subset_pairs, full_pairs, first.theta)
    if theta_full.shape[0] != int(npar_full):
        raise ValueError(
            f'embedded vector has length {theta_full.shape[0]}, expected '
            f'{int(npar_full)}; the full pair list and npar disagree')
    stage2_started = time.perf_counter()
    # The size of the core-virtual component the second stage has to
    # remove. One gradient evaluation, kept because it is the number
    # that explains the stage's energy change.
    import jax
    import jax.numpy as jnp
    embedded_gradient = float(jnp.linalg.norm(
        jax.grad(E_full)(jnp.asarray(theta_full, dtype=float))))
    second_iter = (refine_max_iter if second_max_iter is None
                   else int(second_max_iter))
    stage2_target = float(refine_target)
    sp = refine_stationary_point(E_full, theta_full, max_iter=second_iter,
                                 target=stage2_target, arrays=arrays)
    newton_gradient = float(sp.grad_norm)
    newton_reason = sp.stop_reason
    polished = False
    if not sp.is_stationary and newton_gradient > SECOND_STAGE_FALLBACK_TOL:
        # Same rescue as the single-stage route: a stalled Newton step
        # hands over to a trust-region step, and only a strictly lower
        # gradient is accepted. It runs only when the Newton point misses
        # the acceptance level -- below that, the extra passes cost more
        # than the factor they recover.
        th, en, gn = sp.theta, float(sp.energy), float(sp.grad_norm)
        for _ in range(3):
            improved = _trust_region_polish(E_full, th, stage2_target,
                                            max_iter=2 * second_iter,
                                            arrays=arrays)
            if improved is None or improved[2] >= gn:
                break
            th, en, gn = improved
            polished = True
            if gn < stage2_target:
                break
        if polished:
            sp = StationaryPoint(
                theta=th, energy=en, grad_norm=gn,
                n_accepted=sp.n_accepted,
                stop_reason=(StopReason.CONVERGED
                             if gn < stage2_target
                             else StopReason.STALLED_GRADIENT),
                target=float(stage2_target))

    second_optimizer = 'newton'
    fallback_used = False
    if float(sp.grad_norm) > SECOND_STAGE_FALLBACK_TOL:
        # The Newton stage alone did not reach the acceptance level. A
        # short L-BFGS-B pass from the refined point is cheap when it
        # works, and a fresh Newton refinement after it removes the
        # residual; the candidate is kept only when its gradient is
        # lower than the point already in hand.
        fallback_used = True
        second_optimizer = 'newton+lbfgs-b'
        _, theta_fb, _, _ = optimize_theta_scipy(
            E_full, npar_full, steps=SECOND_STAGE_LBFGS_STEPS,
            g_conv=SECOND_STAGE_FALLBACK_TOL, theta0=sp.theta,
            arrays=arrays)
        sp_fb = refine_stationary_point(
            E_full, theta_fb, max_iter=second_iter, target=stage2_target,
            arrays=arrays)
        if float(sp_fb.grad_norm) < float(sp.grad_norm):
            sp = sp_fb
    stage2_seconds = time.perf_counter() - stage2_started

    point = OrbitalPoint(
        theta=sp.theta,
        energy=float(sp.energy),
        grad_norm=float(sp.grad_norm),
        optimizer_grad_norm=float(first.optimizer_grad_norm),
        optimizer_message=str(first.optimizer_message),
        optimizer_nit=int(first.optimizer_nit),
        optimizer_reached=bool(first.optimizer_reached),
        refinement=sp,
        polished=polished,
        optimizer_name=first.optimizer_name,
    )
    return TwoStagePoint(
        point=point,
        first_stage=first,
        embedded_theta=theta_full,
        stage1_seconds=float(stage1_seconds),
        stage2_seconds=float(stage2_seconds),
        second_optimizer=second_optimizer,
        fallback_used=fallback_used,
        newton_gradient=float(newton_gradient),
        newton_stop_reason=str(getattr(newton_reason, 'value',
                                       newton_reason)),
        embedded_gradient=float(embedded_gradient),
    )


def auto_window(mol, mf, family):
    """Frontier identity and a suggested active window for one family.

    ``mol`` is a built Mole; ``mf`` its converged mean-field solution
    (anything carrying ``mo_energy`` and ``mo_coeff``).  ``family`` is
    ``'npi'`` or ``'ppi'``.  The suggestion logic lives in
    ``analysis.orbitals.frontier_report``: the (HOMO, LUMO) pair for a
    standard frontier, widened to three orbitals when a degenerate
    occupied pi pair or a lone pair below a conjugated pi frontier is
    in play.  The full report is returned -- the suggestion is a
    suggestion, and everything that produced it travels with it.

    The report also carries ``policy_window``: the same facts run
    through the policy layer's parameterized rule (``policy.windows``),
    whose default parameters reproduce the analysis suggestion exactly.
    The window the caller should act on is the policy one; the analysis
    field stays as the checkable default.
    """
    from ..analysis.orbitals import frontier_report, heavy_atoms
    from ..policy.windows import suggest_window

    labels = mol.ao_labels()
    nocc = mol.nelectron // 2
    rep = frontier_report(
        np.asarray(mf.mo_coeff), labels, np.asarray(mf.mo_energy),
        nocc, heavy_atoms(labels),
        target=family, nelectron=int(mol.nelectron))
    rep['policy_window'] = suggest_window(dict(rep, nocc=nocc),
                                          family=family,
                                          nelectron=int(mol.nelectron))
    return rep
