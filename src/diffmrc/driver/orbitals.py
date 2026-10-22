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

import numpy as np

from ..core.optimizer import (optimize_theta_hand, optimize_theta_scipy,
                              optimize_theta_trust_region)
from ..core.stationary import (StopReason, StationaryPoint,
                               refine_stationary_point)

__all__ = ['OrbitalPoint', 'converge_orbital_point', 'hessp_of',
           'optimize_theta_trust_region',
           'OPTIMIZER_GRAD_TOL', 'REFINE_TARGET']

OPTIMIZER_GRAD_TOL = 1e-6
REFINE_TARGET = 1e-11
MAX_STEPS = 2000
REFINE_MAX_ITER = 80


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
    """
    import scipy.optimize as sopt

    import jax
    from jax import numpy as jnp

    E_jit = jax.jit(E_theta)
    g_jit = jax.jit(jax.grad(E_theta))
    hvp_jit = hessp_of(E_theta)
    x0 = np.asarray(theta0, dtype=float)

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
    optimizer_name = 'lbfgs-b'
    E_jit = None
    if use_trust_region:
        E_jit, theta, g_after, info = optimize_theta_trust_region(
            E_theta, npar, steps=max_steps, g_conv=optimize_tol,
            theta0=theta0, arrays=arrays)
        if E_jit is not None:
            optimizer_name = 'trust-region'
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
