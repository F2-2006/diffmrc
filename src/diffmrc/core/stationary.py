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
           'StationaryCandidate', 'SelectionResult',
           'SpectrumPoint', 'StationarySpectrum', 'BasinHoppingStep',
           'BasinHoppingResult',
           'refine_stationary_point', 'curvature_report',
           'select_stationary_point', 'cluster_stationary_points',
           'basin_hopping_search']

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
    if w.size == 0:
        # No rotation directions at all (e.g. an active space that
        # already spans the whole basis, H2/STO-3G CAS(2,2)): there is
        # no curvature that could be negative, and the condition number
        # is undefined -- one keeps the report finite.
        return CurvatureReport(n_negative=0, smallest=0.0, largest=0.0,
                               condition_number=1.0, n_near_zero=0)
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


@dataclass
class StationaryCandidate:
    """One refined stationary point entering a multistart selection.

    ``curvature`` is optional: without it the point can be stationary but
    cannot be certified as a minimum. ``label`` names the starting point
    and is the primary tie-break; two candidates with equal energy and
    gradient are ordered by it, then by the canonical form of ``theta``, so
    the selection does not depend on the order in which the runs finished.
    """

    theta: object
    energy: float
    grad_norm: float
    curvature: object = None
    label: str = ''

    @property
    def is_minimum(self):
        return self.curvature is not None and self.curvature.is_minimum


@dataclass
class SelectionResult:
    """The outcome of :func:`select_stationary_point`.

    ``status`` is one of

    ``'no_stationary'``
        No candidate reached the gradient target; ``chosen`` is None.
    ``'minimum'`` / ``'minimum_degenerate'``
        The chosen point is a certified minimum (no negative curvature). The
        second form says another certified minimum lies within
        ``energy_tol`` of it, so the energy alone does not identify one
        point.
    ``'saddle_only'``
        Curvature reports exist and every stationary candidate has a
        negative direction. ``chosen`` is the lowest such point, reported
        for diagnosis, and ``is_minimum`` is False.
    ``'curvature_unverified'``
        No candidate carries a curvature report; the lowest stationary
        point is returned but cannot be certified.
    """

    chosen: object
    status: str
    n_candidates: int
    n_stationary: int
    n_minima: int
    energy_spread: float = None
    second_energy: float = None
    reason: str = ''

    @property
    def is_minimum(self):
        return self.chosen is not None and self.chosen.is_minimum


def _canonical_theta(theta):
    """Order-independent tie-break key for a parameter vector."""
    arr = np.asarray(theta, dtype=float).reshape(-1)
    return tuple(np.round(arr, 12).tolist())


def select_stationary_point(candidates, target=1e-11, energy_tol=1e-6):
    """Deterministically select one stationary point among refined runs.

    The CASSCF energy is non-convex in the orbital parameters, so different
    starts can reach different stationary points and a bare "lowest run"
    rule is not stable. This function makes the choice explicit and
    reproducible:

    1. Keep only candidates whose gradient reached ``target``.
    2. Prefer certified minima (``curvature.is_minimum``). If none is
       certified, report the state honestly instead of calling a saddle a
       minimum.
    3. Among the pool, order by ``(energy, grad_norm, label, theta)``.
       Every component is a deterministic function of the run, so the
       selection does not depend on completion order.
    4. Mark the result degenerate when a second certified minimum lies
       within ``energy_tol`` (relative to ``max(1, |E|)``); ``second_energy``
       is the next distinct energy among *all* stationary candidates (a
       lower saddle is useful context) for the caller to report.

    This is a selection rule, not a search strategy. The candidate set is
    built by :func:`cluster_stationary_points` from runs the caller
    refined; :func:`basin_hopping_search` is the search over starting
    points, and it remains a heuristic that supplies no global-minimum
    guarantee.
    """
    candidates = list(candidates)
    n_candidates = len(candidates)
    stationary = [c for c in candidates if float(c.grad_norm) < float(target)]
    if not stationary:
        return SelectionResult(
            chosen=None, status='no_stationary', n_candidates=n_candidates,
            n_stationary=0, n_minima=0,
            reason='no candidate reached the gradient target')

    minima = [c for c in stationary if c.is_minimum]
    if minima:
        pool = minima
        base = 'minimum'
    elif any(c.curvature is not None for c in stationary):
        pool = stationary
        base = 'saddle_only'
    else:
        pool = stationary
        base = 'curvature_unverified'

    def key(c):
        return (float(c.energy), float(c.grad_norm), str(c.label),
                _canonical_theta(c.theta))

    pool = sorted(pool, key=key)
    chosen = pool[0]
    tol_abs = float(energy_tol) * max(1.0, abs(float(chosen.energy)))
    family = [c for c in pool
              if abs(float(c.energy) - float(chosen.energy)) <= tol_abs]
    # The next distinct energy is reported over *all* stationary candidates,
    # not only the minima pool: a lower saddle is relevant context for the
    # caller even though it is not eligible for selection.
    higher = [float(c.energy) for c in stationary
              if abs(float(c.energy) - float(chosen.energy)) > tol_abs]
    second = min(higher) if higher else None
    energies = [float(c.energy) for c in stationary]
    spread = max(energies) - min(energies)
    status = base + ('_degenerate' if len(family) > 1 else '')
    reason = (
        'selected the lowest %s among %d stationary candidates; '
        '%d carry curvature reports and %d are minima'
        % ('minimum' if base == 'minimum' else 'stationary point',
           len(stationary),
           sum(1 for c in stationary if c.curvature is not None),
           len(minima)))
    if len(family) > 1:
        reason += ('; %d minima lie within energy_tol, so the energy does '
                   'not identify one point' % len(family))
    return SelectionResult(chosen=chosen, status=status,
                           n_candidates=n_candidates,
                           n_stationary=len(stationary), n_minima=len(minima),
                           energy_spread=float(spread),
                           second_energy=second, reason=reason)


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


def refine_stationary_point(E_theta, theta0, max_iter=80, target=1e-11,
                            arrays=()):
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
    arrays : tuple
        Extra runtime arguments appended to every ``E_theta(theta, *arrays)``
        call; the same contract as ``optimizer.optimize_theta_scipy``
        (large reference arrays must not sit in a jit closure).
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

    from .jit_cache import cached_jit

    def _build_grad(fn):
        return jax.jit(jax.grad(fn))

    def _build_hvp(fn):
        plain = jax.grad(fn)

        def raw(theta, v, *arr):
            return jax.jvp(lambda t: plain(t, *arr), (theta,), (v,))[1]

        return jax.jit(raw)

    # A refinement called once per start would otherwise rebuild and
    # recompile both graphs on every call; the cache lives on E_theta, so
    # the cost is paid once per energy function (core.jit_cache records the
    # measurement).
    grad = cached_jit(E_theta, 'grad', _build_grad)
    hvp = cached_jit(E_theta, 'hvp', _build_hvp)

    theta = jnp.asarray(np.asarray(theta0, dtype=float))
    energy = float(E_theta(theta, *arrays))
    lam = _LAMBDA_INIT
    n_accepted = 0
    gnorm = float(jnp.linalg.norm(grad(theta, *arrays)))
    # Default to the cap: it is the reason that needs no break to reach.
    reason = StopReason.ITERATION_CAP

    for _ in range(max_iter):
        if gnorm < target:
            reason = StopReason.CONVERGED
            break
        g = np.asarray(grad(theta, *arrays), dtype=float)
        accepted = False
        for _try in range(_INNER_TRIES):
            def _matvec(v):
                return np.asarray(hvp(theta, jnp.asarray(v), *arrays),
                                  dtype=float) + lam * v

            step = _cg_solve(_matvec, g)
            if not np.all(np.isfinite(step)) \
                    or np.linalg.norm(step) > _MAX_STEP_NORM:
                # The Newton model has stopped describing the function.
                lam *= _LAMBDA_UP
                continue
            candidate = theta - jnp.asarray(step)
            e_new = float(E_theta(candidate, *arrays))
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

        g_new = float(jnp.linalg.norm(grad(theta, *arrays)))
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


def curvature_from_gradient(grad_fn, theta, step=1e-5, max_parameters=None):
    """Hessian from symmetric differences of an analytic gradient.

    WHY THIS EXISTS

    The analytic second derivative of the blocked eigensolver is
    undefined at an exact degeneracy: on N/6-31G CAS(3,3) quartet and on
    N2/6-31G with a degenerate frontier pair, the orbital Hessian at the
    symmetric stationary point is entirely NaN while the energy and the
    analytic gradient there are finite. The energy itself is smooth, so
    its curvature exists; what fails is one numerical route to it.

    This function takes the other route. The gradient is analytic and
    finite, and a symmetric difference of it is the standard second
    derivative of a smooth function, accurate to O(step**2). It is the
    same practice the NEVPT2 frequencies use.

    Parameters
    ----------
    grad_fn : callable
        Analytic gradient, ``theta -> grad``.
    theta : array_like
        Point at which the curvature is wanted.
    step : float
        Symmetric-difference step, in the units of ``theta``.
    max_parameters : int, optional
        Refuse above this dimension. The cost is ``2 n`` gradient
        evaluations, and for an f-block rotation space that is not a
        diagnostic anyone will wait for.

    Returns
    -------
    ndarray
        Symmetrized Hessian of shape ``(n, n)``.

    Raises
    ------
    ValueError
        If ``max_parameters`` is given and the dimension exceeds it.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n = theta.size
    if max_parameters is not None and n > int(max_parameters):
        raise ValueError(
            f'the gradient-difference Hessian costs 2 x {n} gradient '
            f'evaluations, above the cap of {int(max_parameters)} '
            f'parameters; the analytic route must be repaired for a space '
            f'this large')
    H = np.empty((n, n), dtype=float)
    for j in range(n):
        shift = np.zeros(n)
        shift[j] = step
        plus = np.asarray(grad_fn(theta + shift), dtype=float).reshape(-1)
        minus = np.asarray(grad_fn(theta - shift), dtype=float).reshape(-1)
        H[:, j] = (plus - minus) / (2.0 * step)
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# Multistart spectra and a basin-hopping search.
#
# The functions above refine one point and select among candidates the caller
# supplied. The functions below build the candidate set. The search is a
# heuristic: it samples the basins and reports what it found with the evidence
# it has. It does not prove that the lowest point in the spectrum is the
# global minimum, and no stopping rule below may be read as such a proof.
# ---------------------------------------------------------------------------


@dataclass
class SpectrumPoint:
    """One distinct stationary point, with the runs that reached it.

    ``representative`` is the lowest-energy member of the cluster. Every
    property forwards to it, so a spectrum point can be consumed wherever a
    :class:`StationaryCandidate` is expected. ``strict`` records whether the
    representative reached the strict gradient target; a point that did not
    can still be a usable stationary point in a flat direction, but it must
    not be called strictly stationary.
    """

    representative: StationaryCandidate
    n_members: int = 1
    labels: tuple = ()
    strict: bool = False

    @property
    def theta(self):
        return self.representative.theta

    @property
    def energy(self):
        return float(self.representative.energy)

    @property
    def grad_norm(self):
        return float(self.representative.grad_norm)

    @property
    def curvature(self):
        return self.representative.curvature

    @property
    def label(self):
        return str(self.representative.label)

    @property
    def is_minimum(self):
        return self.representative.is_minimum

    @property
    def has_curvature(self):
        return self.curvature is not None


@dataclass
class StationarySpectrum:
    """The distinct stationary points reached by a set of refined runs.

    ``points`` is ordered by ``(energy, gradient, label, theta)``, the same
    order :func:`select_stationary_point` uses, so two spectra built from the
    same candidates in a different completion order are identical.
    """

    points: tuple
    n_candidates: int
    n_stationary: int
    n_distinct: int
    target: float
    energy_tol: float = 1e-6
    strict_target: float = 1e-11

    @property
    def best(self):
        """The lowest-energy distinct point, or None for an empty spectrum."""
        return self.points[0] if self.points else None

    @property
    def n_energy_levels(self):
        """Distinct energies, merging orbital orientations of one level.

        ``n_distinct`` counts (energy, orbital) points; on a degenerate
        manifold the two differ. Measured on N2/6-31G CAS(2,2): one energy
        and six active-space orientations.
        """
        levels = []
        for point in sorted(self.points, key=lambda p: p.energy):
            if not any(_same_energy(point.energy, level, self.energy_tol)
                       for level in levels):
                levels.append(point.energy)
        return len(levels)

    @property
    def n_minima(self):
        return sum(1 for p in self.points if p.is_minimum)

    @property
    def n_certified(self):
        return sum(1 for p in self.points if p.has_curvature)

    @property
    def certificate_coverage(self):
        """Fraction of distinct points that carry a curvature report."""
        if not self.points:
            return 0.0
        return float(self.n_certified) / len(self.points)

    @property
    def spread(self):
        """Energy range across distinct points, in Hartree."""
        if not self.points:
            return 0.0
        energies = [p.energy for p in self.points]
        return float(max(energies) - min(energies))

    @property
    def selection(self):
        """The deterministic selection over the distinct representatives."""
        return select_stationary_point(
            [p.representative for p in self.points],
            target=self.target, energy_tol=self.energy_tol)


def _same_energy(a, b, energy_tol):
    ea, eb = float(a), float(b)
    return abs(ea - eb) <= float(energy_tol) * max(1.0, abs(ea), abs(eb))


def cluster_stationary_points(candidates, target=1e-11, energy_tol=1e-6,
                              same_point=None, strict_target=1e-11):
    """Group refined candidates into distinct stationary points.

    A candidate enters the spectrum only when its gradient reached
    ``target``; the others are counted but not clustered, because a point
    that is not stationary has no energy identity to compare.

    Two candidates are the same point when their energies agree to
    ``energy_tol`` (relative to ``max(1, |E|)``) **and** ``same_point``, when
    given, returns True. The energy test alone is not sufficient: two
    genuinely different solutions can lie within a millihartree of each
    other, and a single point refined from two starts can differ by more
    than floating-point noise. The orbital test belongs to the caller
    because it needs the AO overlap matrix and the MO coefficients; the
    experiment scripts pass ``core.crossing.orbital_space_overlap``, which
    is invariant to rotations inside the active space.

    The clustering is greedy against the cluster representative, and the
    candidates are sorted by ``(energy, gradient, label, theta)`` before the
    pass, so the result does not depend on the order the runs finished.

    Parameters
    ----------
    candidates : iterable of StationaryCandidate
    target : float
        Gradient norm below which a candidate is called stationary.
    energy_tol : float
        Relative energy tolerance for the same-point test.
    same_point : callable, optional
        ``same_point(candidate_a, candidate_b) -> bool``. When omitted only
        the energy test is applied.
    strict_target : float
        Gradient norm for the ``strict`` flag on each spectrum point.

    Returns
    -------
    StationarySpectrum
    """
    candidates = list(candidates)
    stationary = [c for c in candidates
                  if float(c.grad_norm) < float(target)]

    def key(c):
        return (float(c.energy), float(c.grad_norm), str(c.label),
                _canonical_theta(c.theta))

    stationary.sort(key=key)
    clusters = []
    for candidate in stationary:
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            if not _same_energy(candidate.energy, representative.energy,
                                energy_tol):
                continue
            if same_point is not None and not same_point(candidate,
                                                         representative):
                continue
            cluster.append(candidate)
            placed = True
            break
        if not placed:
            clusters.append([candidate])

    points = []
    for cluster in clusters:
        representative = cluster[0]
        points.append(SpectrumPoint(
            representative=representative,
            n_members=len(cluster),
            labels=tuple(str(c.label) for c in cluster),
            strict=bool(float(representative.grad_norm)
                        < float(strict_target))))
    points.sort(key=lambda p: (p.energy, p.grad_norm, p.label,
                               _canonical_theta(p.theta)))
    return StationarySpectrum(
        points=tuple(points), n_candidates=len(candidates),
        n_stationary=len(stationary), n_distinct=len(points),
        target=float(target), energy_tol=float(energy_tol),
        strict_target=float(strict_target))


@dataclass
class BasinHoppingStep:
    """One move of a basin-hopping walk, recorded for the trace.

    ``is_new`` means the move reached an energy not present among the
    stationary points so far. It is deliberately an energy test rather
    than an orbital test. The spectrum keeps the orbital identity, but a
    search that counted a new orbital orientation of the same solution as
    news would never stop on a degenerate manifold: measured on
    N2/6-31G CAS(2,2), eight converged starts gave one energy and six
    distinct active-space orientations (the two pi components are exactly
    degenerate).
    """

    step: int
    move: str
    label: str
    energy: float
    grad_norm: float
    is_stationary: bool
    has_curvature: bool
    is_minimum: bool
    is_new: bool
    accepted: bool
    note: str = ''


@dataclass
class BasinHoppingResult:
    """The outcome of :func:`basin_hopping_search`.

    ``stop_reason`` is one of ``'step_budget'``, ``'time_budget'`` or
    ``'no_new_points'``. Each is a decidable predicate on the run; none of
    them says the global minimum was found.
    """

    best: object
    spectrum: StationarySpectrum
    trace: tuple
    n_steps: int
    n_accepted: int
    n_new: int
    stop_reason: str
    seconds: float
    initial: object = None

    @property
    def certificate_coverage(self):
        return self.spectrum.certificate_coverage

    @property
    def certified(self):
        """True when the reported best is a certified minimum."""
        return self.best is not None and self.best.is_minimum


def basin_hopping_search(converge, theta0, moves, same_point=None,
                         target=1e-11, energy_tol=1e-6, strict_target=1e-11,
                         max_steps=100, max_no_new=None, max_seconds=None,
                         temperature=0.0, energy_slack=1e-8,
                         require_curvature=True, coverage_gate=False,
                         novelty_window=None,
                         seed=0, initial_label='start'):
    """Basin-hopping search over starting points for the orbital optimizer.

    One move is: propose a starting point, converge it with the caller's
    refinement, and decide whether the walker moves there. The proposal is a
    callable supplied by the caller, so orbital rotations, geometry
    displacements and reference-orbital permutations are all the same kind
    of object to this function.

    ACCEPTANCE

    A move is eligible only when the refined point reached the gradient
    target and, when ``require_curvature`` is True, carries a curvature
    report. Without that gate a walker can drift onto a point whose
    "minimum" status is unknown, and the search would then report an
    uncertified point as its best. An eligible move is accepted when its
    energy is at most ``energy_slack`` above the walker's current energy;
    with ``temperature > 0`` an uphill eligible move is accepted with the
    Metropolis probability ``exp(-dE / temperature)``. ``temperature = 0``
    (the default) is a greedy walk: deterministic given the seed, and the
    cheaper of the two.

    STOPPING RULES

    All three are decidable and are checked before each move:

    ``step_budget``
        ``max_steps`` moves have been made.
    ``time_budget``
        ``max_seconds`` of wall clock have elapsed. A single move can
        overrun the budget; the check is per move, not inside one.
    ``no_new_points``
        ``max_no_new`` consecutive moves produced no new stationary
        *energy* (the orbital identity is reported in the spectrum but
        does not reset the counter: degenerate orientations of one
        solution would otherwise keep the walk alive forever). When
        ``coverage_gate`` is True this rule fires
        only if every distinct point found so far carries a curvature
        report; otherwise the walk continues to the step or time budget,
        because a spectrum with uncertified points cannot support the
        selection regardless of how long it has been stable.

    The search does not guarantee the global minimum. It reports the
    spectrum it reached, the best certified point, and the certificate
    coverage, and the caller must state the budget alongside the result.

    Parameters
    ----------
    converge : callable
        ``converge(theta0, label) -> StationaryCandidate``. The caller owns
        the optimizer and the refinement; this function only decides where
        to start them.
    theta0 : array_like
        Starting point of the walk, already in the parameter space of
        ``converge``.
    moves : sequence of (str, callable)
        Each callable is ``propose(rng, theta) -> theta_start``. They are
        tried round-robin in the listed order, so the trace is reproducible.
    same_point : callable, optional
        Forwarded to :func:`cluster_stationary_points`.
    target, energy_tol, strict_target : float
        Forwarded to :func:`cluster_stationary_points`.
    max_steps, max_no_new, max_seconds : int, int, float
        Budgets; see the stopping rules.
    temperature : float
        Metropolis temperature in Hartree. Zero means greedy acceptance.
    energy_slack : float
        Energy increase (Hartree) tolerated for a greedy acceptance.
    require_curvature : bool
        Gate acceptance on a curvature report.
    coverage_gate : bool
        Require full certificate coverage before the no-new-points rule
        may fire.
    novelty_window : float, optional
        Energy window (Hartree) above the lowest stationary point found.
        A new point outside the window is still recorded in the spectrum
        and still counts in ``n_new``, but it does not reset the
        no-new-points counter. The window exists because the theta chart
        contains high-energy stationary points in which a virtual orbital
        occupies a core slot: measured on H2O/6-31G, a large orbital kick
        reached a minimum 1.15 Ha above the ground state. Such a point is
        stationary in the chart and useless as a search result, and
        letting it reset the counter delays the stop without adding
        information.
    seed : int
        Seed of the generator handed to the move callables.
    initial_label : str
        Label of the starting point.

    Returns
    -------
    BasinHoppingResult
    """
    moves = list(moves)
    if not moves:
        raise ValueError('moves must contain at least one proposal')
    for entry in moves:
        if len(entry) != 2 or not callable(entry[1]):
            raise ValueError(
                'each move must be a (name, callable) pair, got %r'
                % (entry,))
    if max_steps < 0:
        raise ValueError(f'max_steps must be non-negative, got {max_steps}')
    if max_no_new is not None and max_no_new < 1:
        raise ValueError(
            f'max_no_new must be at least 1, got {max_no_new}')

    import time

    rng = np.random.default_rng(int(seed))
    started = time.time()

    def elapsed():
        return time.time() - started

    initial = converge(np.asarray(theta0, dtype=float), initial_label)
    current = initial
    candidates = [initial]
    trace = []
    n_accepted = 0
    no_new = 0
    n_new = 0
    stop_reason = 'step_budget'
    step = 0
    while step < int(max_steps):
        if max_seconds is not None and elapsed() >= float(max_seconds):
            stop_reason = 'time_budget'
            break
        if max_no_new is not None and no_new >= int(max_no_new):
            if coverage_gate:
                coverage = cluster_stationary_points(
                    candidates, target=target, energy_tol=energy_tol,
                    same_point=same_point,
                    strict_target=strict_target).certificate_coverage
                if coverage >= 1.0:
                    stop_reason = 'no_new_points'
                    break
            else:
                stop_reason = 'no_new_points'
                break

        name, propose = moves[step % len(moves)]
        label = '%s-%d' % (name, step)
        start = propose(rng, np.asarray(current.theta, dtype=float))
        candidate = converge(np.asarray(start, dtype=float), label)

        # A point is new when it is not the same point as any representative
        # already in the spectrum. Comparing cluster counts instead would be
        # wrong in the rare case where a new point bridges two clusters and
        # merges them: the count falls while a new point was in fact found.
        is_new = False
        if float(candidate.grad_norm) < float(target):
            is_new = not any(
                _same_energy(candidate.energy, other.energy, energy_tol)
                for other in candidates
                if float(other.grad_norm) < float(target))
        candidates.append(candidate)
        counts_as_new = is_new
        if is_new and novelty_window is not None:
            stationary = [other.energy for other in candidates
                          if float(other.grad_norm) < float(target)]
            if stationary:
                lowest = min(stationary)
                if float(candidate.energy) > lowest + float(novelty_window):
                    counts_as_new = False
        if counts_as_new:
            n_new += 1
            no_new = 0
        else:
            no_new += 1

        is_stationary = bool(float(candidate.grad_norm) < float(target))
        has_curvature = candidate.curvature is not None
        usable = is_stationary and (has_curvature or not require_curvature)
        accepted = False
        note = ''
        if not usable:
            note = ('not stationary' if not is_stationary
                    else 'no curvature report')
        else:
            delta = float(candidate.energy) - float(current.energy)
            if delta <= float(energy_slack):
                accepted = True
            elif float(temperature) > 0.0:
                probability = float(np.exp(-delta / float(temperature)))
                if rng.random() < probability:
                    accepted = True
                else:
                    note = 'metropolis reject'
            else:
                note = 'higher energy'
        if accepted:
            current = candidate
            n_accepted += 1

        trace.append(BasinHoppingStep(
            step=step, move=str(name), label=label,
            energy=float(candidate.energy),
            grad_norm=float(candidate.grad_norm),
            is_stationary=is_stationary, has_curvature=has_curvature,
            is_minimum=bool(candidate.is_minimum),
            is_new=is_new, accepted=accepted, note=note))
        step += 1
    else:
        # The while loop ended because the step budget was reached.
        stop_reason = 'step_budget'

    spectrum = cluster_stationary_points(
        candidates, target=target, energy_tol=energy_tol,
        same_point=same_point, strict_target=strict_target)
    selection = spectrum.selection
    best = selection.chosen
    return BasinHoppingResult(
        best=best, spectrum=spectrum, trace=tuple(trace),
        n_steps=len(trace), n_accepted=n_accepted, n_new=n_new,
        stop_reason=stop_reason, seconds=float(elapsed()), initial=initial)
