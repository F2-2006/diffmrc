"""Convergence criterion for the perturbation-layer correction.

The CASSCF energy is stationary in the orbital parameters, so its error is
second order in the residual orbital gradient while the NEVPT2 correction
``e2`` is first order: a small energy change does not imply a small change
of the correction.  Measured on H2O/6-31G CAS(2,2), between optimizations
converged to 1e-4 and 1e-6 the CASSCF energy moves by 1.3e-9 Eh while the
correction moves by 1.1e-5 Eh.

This module answers one question, and only that question:

    at this orbital point, is ``e2`` determined to the requested tolerance
    by the local data, and is the point a minimum of the CASSCF energy?

It does **not** answer which stationary point is the physical target.  The
criterion is deliberately branch-free: every quantity it uses is a
derivative at the point in hand, except for the optional repeat check,
which compares two points that have already been certified to lie on the
same branch by their wavefunction overlap.  Branch selection is left to
:func:`branch_spread`, which uses only the variational CASSCF energy and
reports the ``e2`` spread as a residual.

The local model
---------------

Write ``E(t)`` for the CASSCF energy and ``e2(t)`` for the correction, with
``t`` the orbital-rotation parameters.  At a stationary point ``t*``,

    g = dE/dt,   H = d2E/dt2,   s = de2/dt,
    t - t* ~= -H^-1 g,   e2(t) - e2(t*) ~= -s . H^-1 g.

The first-order bound is therefore ``|H^-1 s| |g|``.  It is a *local*
statement: it needs no other stationary point, no branch label and no
starting point.

Three failure modes survive a bare gradient threshold, and each has a
distinct signature:

1. **Saddle.**  ``H`` has a negative eigenvalue.  The Newton displacement
   ``-H^-1 g`` is not a descent direction and the linear model has the
   wrong sign along that mode.  Measured on H2O/6-31G CAS(2,2): the
   zero-start stationary point has ``lambda_min = -1.58e-2`` and lies
   21.2 mEh above the minimum, with an 18.7 mEh different correction.
2. **Energy-flat direction that carries e2.**  ``H`` has a zero (or
   numerically zero) eigenvalue but ``s`` has a component along that
   eigenvector.  The energy does not fix the position along the mode, so a
   finite move can change ``e2`` without any energy penalty and no
   gradient threshold can bound it.  Measured on OH/6-31G CAS(3,2): four
   starts with the *same* CASSCF energy to 1e-12, orbital gradients below
   2.5e-6, and corrections from -0.0754 to -0.0873 Eh (a 12 mEh spread).
3. **Residual gradient too large for the requested accuracy.**  The
   ordinary case: ``|H^-1 s| |g|`` exceeds the tolerance.

A flat direction that does *not* carry ``e2`` is harmless and common: the
active-active rotation is a redundancy of the CAS-CI parametrization, and
after the canonicalization the correction is invariant along it.  The
H2O/6-31G minimum has such a mode with ``s . v = 2e-12``.

Thresholds and where they come from
-----------------------------------

``curvature_threshold = 1e-3``
    The measured soft-mode noise of the value-faithful Hessian (central
    differences of the analytic gradient) reaches ``3.3e-5`` on certified
    minima; the smallest physical negative curvature measured here is
    ``1.58e-2`` (the H2O saddle).  The geometric mean is ``7.2e-4``; the
    default rounds it to ``1e-3``.  Eigenvalues inside the band are
    neither certified positive nor certified negative, so they are treated
    as flat and tested with the null-direction rule.  The raw
    ``curvature_report`` default of ``1e-8`` is below the noise floor of
    this route and would call the H2O minimum a saddle.

``e2_tolerance = 1e-6`` Eh
    The requested accuracy of the correction.  It is the only physical
    constant in the criterion; the gradient threshold is derived per point
    as ``e2_tolerance / |H_hard^-1 s_hard|`` rather than fixed.

``null_angle = 0.1`` rad
    The declared amplitude of the finite probe along a flat direction.
    The measured first-order separation is wide: controlled modes carry
    ``|s . v| <= 1e-8`` Eh/rad (including the exact redundancy),
    uncontrolled ones ``>= 1e-2`` Eh/rad.  The first order is not the
    whole story: the OH/6-31G CAS(3,2) ROHF point has ``|s_soft| = 1e-12``
    but a finite probe of ``2.7e-3`` Eh, because the second-order term
    dominates on an exactly flat mode.  That is why a supplied finite probe
    is the primary test and the linear term is kept only as a floor.

Verdicts
--------

``accept``
    The point is a certified minimum, the first-order bound is within the
    tolerance, no flat direction carries ``e2``, and (when supplied) the
    repeat difference between same-branch sets is within the tolerance.
``reject``
    A positive failure: negative curvature above the noise floor, a flat
    direction that carries ``e2``, a first-order bound above the tolerance,
    or a same-branch repeat difference above the tolerance.  For an
    unconverged point the last case means "not converged enough", not
    "physically wrong".
``undetermined``
    The evidence is insufficient: a non-finite input, a missing Hessian, or
    a flat direction with an e2 coupling below the linear detection floor
    but no finite probe to rule out a second-order variation.

Branch dependence that remains
------------------------------

Two certified minima can carry corrections that differ by more than the
tolerance.  The local criterion accepts both, correctly: each is converged
on its own branch.  :func:`branch_spread` then reports the spread and
selects by the CASSCF energy alone, because that is the only variational
quantity available; selecting by ``e2`` would be circular.  When the
spread exceeds the tolerance the reported correction is branch-dependent
by that amount, and the report says so instead of hiding it behind a
single number.
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = ['DEFAULT_E2_TOLERANCE', 'DEFAULT_CURVATURE_THRESHOLD',
           'DEFAULT_NULL_ANGLE', 'DEFAULT_OVERLAP_MIN',
           'SpectrumSplit', 'PerturbationConvergenceReport',
           'BranchPoint', 'BranchSpreadReport',
           'split_spectrum', 'first_order_bound',
           'assess_perturbation_convergence', 'branch_spread']

# Requested accuracy of the correction, in hartree.  The one physical
# constant of the criterion.
DEFAULT_E2_TOLERANCE = 1e-6
# Geometric mean of the measured Hessian noise (3.3e-5) and the smallest
# physical negative curvature (1.58e-2); see the module docstring.
DEFAULT_CURVATURE_THRESHOLD = 1e-3
# Declared amplitude of the finite probe along a flat direction.
DEFAULT_NULL_ANGLE = 0.1
# Two orbital sets with an active-space overlap below this are different
# branches; their correction difference is branch spread, not a
# convergence error, and must not be used as a repeat test.
DEFAULT_OVERLAP_MIN = 0.9


@dataclass
class SpectrumSplit:
    """Eigen-decomposition of the orbital Hessian split by scale.

    ``hard`` modes have ``|lambda| > curvature_threshold`` and support the
    linear model.  ``soft`` modes are inside the noise band and are tested
    for an ``e2`` coupling instead.  ``negative`` modes are the soft or
    hard modes below ``-curvature_threshold``.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    hard: np.ndarray
    soft: np.ndarray
    negative: np.ndarray
    curvature_threshold: float

    @property
    def lowest(self):
        return float(self.eigenvalues[0]) if self.eigenvalues.size else 0.0

    @property
    def largest(self):
        return float(self.eigenvalues[-1]) if self.eigenvalues.size else 0.0

    @property
    def n_negative(self):
        return int(np.count_nonzero(self.negative))

    @property
    def n_soft(self):
        return int(np.count_nonzero(self.soft))

    @property
    def is_minimum(self):
        return self.n_negative == 0

    def project(self, vector):
        """Return ``(hard, soft)`` components of ``vector``."""
        vector = np.asarray(vector, dtype=float).reshape(-1)
        hard = (self.eigenvectors[:, self.hard].T @ vector
                if self.hard.size else np.zeros(0))
        soft = (self.eigenvectors[:, self.soft].T @ vector
                if self.soft.size else np.zeros(0))
        return hard, soft

    def invert_hard(self, vector):
        """``H_hard^-1`` applied to ``vector`` in the eigenbasis."""
        hard, _ = self.project(vector)
        return hard / self.eigenvalues[self.hard]


def split_spectrum(hessian, curvature_threshold=DEFAULT_CURVATURE_THRESHOLD):
    """Symmetrize, diagonalize and split the Hessian by magnitude."""
    H = np.asarray(hessian, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f'H must be square, got shape {H.shape}')
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        raise ValueError('H contains non-finite entries')
    w, V = np.linalg.eigh(H)
    threshold = float(curvature_threshold)
    if threshold <= 0.0:
        raise ValueError('curvature_threshold must be positive')
    hard = np.abs(w) > threshold
    soft = ~hard
    negative = w < -threshold
    return SpectrumSplit(eigenvalues=w, eigenvectors=V, hard=hard, soft=soft,
                         negative=negative, curvature_threshold=threshold)


def first_order_bound(gradient, e2_gradient, hessian, *,
                      curvature_threshold=DEFAULT_CURVATURE_THRESHOLD):
    """Return the first-order quantities of the local model.

    Returns a dict with ``sensitivity`` (``|H_hard^-1 s_hard|``),
    ``bound`` (``sensitivity * |g_hard|``), ``prediction``
    (``-s_hard . H_hard^-1 g_hard``), and the norms of the hard and soft
    components of both gradients.
    """
    g = np.asarray(gradient, dtype=float).reshape(-1)
    s = np.asarray(e2_gradient, dtype=float).reshape(-1)
    if g.shape != s.shape:
        raise ValueError(f'gradient and e2_gradient differ in length: '
                         f'{g.size} vs {s.size}')
    split = split_spectrum(hessian, curvature_threshold)
    if g.size != split.eigenvalues.size:
        raise ValueError(f'gradient length {g.size} does not match the '
                         f'Hessian dimension {split.eigenvalues.size}')
    g_hard, g_soft = split.project(g)
    s_hard, s_soft = split.project(s)
    inv_s = split.invert_hard(s)
    sensitivity = float(np.linalg.norm(inv_s))
    bound = sensitivity * float(np.linalg.norm(g_hard))
    prediction = -float(inv_s @ g_hard)
    return {
        'split': split,
        'sensitivity': sensitivity,
        'bound': bound,
        'prediction': prediction,
        'gradient_hard_norm': float(np.linalg.norm(g_hard)),
        'gradient_soft_norm': float(np.linalg.norm(g_soft)),
        'e2_gradient_hard_norm': float(np.linalg.norm(s_hard)),
        'e2_gradient_soft_norm': float(np.linalg.norm(s_soft)),
        'required_gradient': (DEFAULT_E2_TOLERANCE / sensitivity
                              if sensitivity > 0.0 else float('inf')),
    }


@dataclass
class PerturbationConvergenceReport:
    """The verdict and the evidence behind it.

    ``value`` is ``accept``, ``reject`` or ``undetermined``.  ``certified``
    is the boolean a caller should gate on; it is true only for ``accept``.
    """

    value: str
    certified: bool
    gradient_norm: float
    gradient_max: float
    e2_gradient_norm: float
    hessian_lowest: float
    hessian_largest: float
    n_negative: int
    n_soft: int
    curvature_threshold: float
    e2_tolerance: float
    sensitivity: float
    first_order_bound: float
    first_order_prediction: float
    required_gradient: float
    soft_e2_coupling: float
    null_probe_max: float = None
    repeat_delta_e2: float = None
    repeat_overlap: float = None
    checks: tuple = field(default_factory=tuple)
    reasons: tuple = field(default_factory=tuple)
    warnings: tuple = field(default_factory=tuple)

    def as_dict(self):
        return {
            'value': self.value,
            'certified': self.certified,
            'gradient_norm': self.gradient_norm,
            'gradient_max': self.gradient_max,
            'e2_gradient_norm': self.e2_gradient_norm,
            'hessian_lowest': self.hessian_lowest,
            'hessian_largest': self.hessian_largest,
            'n_negative': self.n_negative,
            'n_soft': self.n_soft,
            'curvature_threshold': self.curvature_threshold,
            'e2_tolerance': self.e2_tolerance,
            'sensitivity': self.sensitivity,
            'first_order_bound': self.first_order_bound,
            'first_order_prediction': self.first_order_prediction,
            'required_gradient': self.required_gradient,
            'soft_e2_coupling': self.soft_e2_coupling,
            'null_probe_max': self.null_probe_max,
            'repeat_delta_e2': self.repeat_delta_e2,
            'repeat_overlap': self.repeat_overlap,
            'checks': [dict(c) for c in self.checks],
            'reasons': list(self.reasons),
            'warnings': list(self.warnings),
        }


def _check(name, status, detail, value=None, threshold=None):
    entry = {'name': name, 'status': status, 'detail': detail}
    if value is not None:
        entry['value'] = float(value)
    if threshold is not None:
        entry['threshold'] = float(threshold)
    return entry


def assess_perturbation_convergence(
        gradient, e2_gradient, hessian, *,
        e2_tolerance=DEFAULT_E2_TOLERANCE,
        curvature_threshold=DEFAULT_CURVATURE_THRESHOLD,
        null_angle=DEFAULT_NULL_ANGLE,
        null_probe_max=None,
        repeat_delta_e2=None,
        repeat_overlap=None,
        overlap_min=DEFAULT_OVERLAP_MIN):
    """Decide whether the correction is converged at one orbital point.

    Parameters
    ----------
    gradient : array_like
        ``dE_cas/dt`` at the point.
    e2_gradient : array_like
        ``de2/dt`` at the point.
    hessian : array_like
        ``d2E_cas/dt2`` at the point.  The value-faithful Hessian is the
        central difference of the analytic gradient; an AD Hessian through
        an eigensolver carries frozen-rule semantics and is not the Hessian
        of the value.
    e2_tolerance : float
        Requested accuracy of the correction, in hartree.
    curvature_threshold : float
        Magnitude below which a Hessian eigenvalue counts as flat.  See the
        module docstring for the measured provenance.
    null_angle : float
        Amplitude of the finite probe along a flat direction, in radians.
        Used to turn an ``e2`` coupling ``|s . v|`` into an estimated
        variation ``|s . v| * null_angle``.
    null_probe_max : float, optional
        The largest measured ``|e2(t + null_angle * v) - e2(t)|`` over the
        flat directions.  When supplied it is combined with the linear
        estimate by ``max``: the finite probe is the honest test because a
        flat direction has no curvature to bound the second-order term, and
        the linear term guards against a probe that misses the
        maximum-coupling direction inside a degenerate soft subspace.
    repeat_delta_e2 : float, optional
        ``|e2(a) - e2(b)|`` between two independently converged orbital
        sets.  Applied only when ``repeat_overlap >= overlap_min``; a lower
        overlap means different branches and the difference is branch
        spread, not a convergence error.
    repeat_overlap : float, optional
        Wavefunction or active-space overlap of the two sets, in [0, 1].
    overlap_min : float
        Threshold above which two sets count as the same branch.

    Returns
    -------
    PerturbationConvergenceReport
    """
    g = np.asarray(gradient, dtype=float).reshape(-1)
    s = np.asarray(e2_gradient, dtype=float).reshape(-1)
    tol = float(e2_tolerance)
    if tol <= 0.0:
        raise ValueError('e2_tolerance must be positive')
    if g.size != s.size:
        raise ValueError(f'gradient and e2_gradient differ in length: '
                         f'{g.size} vs {s.size}')
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(s))
            and np.all(np.isfinite(np.asarray(hessian, dtype=float)))):
        return PerturbationConvergenceReport(
            value='undetermined', certified=False,
            gradient_norm=float(np.linalg.norm(g)),
            gradient_max=float(np.abs(g).max()) if g.size else 0.0,
            e2_gradient_norm=float(np.linalg.norm(s)),
            hessian_lowest=float('nan'), hessian_largest=float('nan'),
            n_negative=0, n_soft=0,
            curvature_threshold=float(curvature_threshold),
            e2_tolerance=tol, sensitivity=float('nan'),
            first_order_bound=float('nan'),
            first_order_prediction=float('nan'),
            required_gradient=float('nan'), soft_e2_coupling=float('nan'),
            checks=(_check('finite_inputs', 'undetermined',
                           'the gradient, the e2 gradient or the Hessian '
                           'contains non-finite entries'),),
            reasons=('non-finite input; no verdict is possible',))

    split = split_spectrum(hessian, curvature_threshold)
    if g.size != split.eigenvalues.size:
        raise ValueError(f'gradient length {g.size} does not match the '
                         f'Hessian dimension {split.eigenvalues.size}')
    g_hard, g_soft = split.project(g)
    s_hard, s_soft = split.project(s)
    inv_s = split.invert_hard(s)
    sensitivity = float(np.linalg.norm(inv_s))
    bound = sensitivity * float(np.linalg.norm(g_hard))
    prediction = -float(inv_s @ g_hard)
    required = tol / sensitivity if sensitivity > 0.0 else float('inf')
    soft_coupling = float(np.linalg.norm(s_soft))

    checks = []
    reasons = []
    warnings = []

    # 1. Curvature: a certified negative eigenvalue is a saddle.
    if split.n_negative:
        checks.append(_check(
            'curvature_minimum', 'fail',
            f'{split.n_negative} Hessian eigenvalue(s) below '
            f'-{split.curvature_threshold:.1e}; lowest '
            f'{split.lowest:.3e}. The Newton direction is not a descent '
            f'direction and the linear model has the wrong sign there',
            value=split.lowest, threshold=-split.curvature_threshold))
        reasons.append(
            f'saddle: lowest Hessian eigenvalue {split.lowest:.3e} is below '
            f'-{split.curvature_threshold:.1e}')
    else:
        checks.append(_check(
            'curvature_minimum', 'pass',
            f'no Hessian eigenvalue below -{split.curvature_threshold:.1e}; '
            f'lowest {split.lowest:.3e}',
            value=split.lowest, threshold=-split.curvature_threshold))

    # 2. Flat directions: harmless only when the correction is invariant
    #    along them.
    if split.n_soft == 0:
        null_value = 0.0
        checks.append(_check(
            'flat_direction', 'pass',
            'no Hessian eigenvalue inside the flat band '
            f'|lambda| <= {split.curvature_threshold:.1e}'))
    else:
        linear = soft_coupling * float(null_angle)
        # The finite probe is the honest test; the linear term is kept as a
        # floor because a probe along one direction can miss the coupling
        # direction inside a degenerate soft subspace.
        null_value = (max(float(null_probe_max), linear)
                      if null_probe_max is not None else linear)
        if null_probe_max is not None:
            if null_value > tol:
                checks.append(_check(
                    'flat_direction', 'fail',
                    f'{split.n_soft} flat direction(s); a finite probe at '
                    f'{null_angle:g} rad changes e2 by {null_value:.3e}, '
                    f'above the tolerance {tol:.3e}. The energy does not '
                    f'fix the position along the mode',
                    value=null_value, threshold=tol))
                reasons.append(
                    f'e2 varies by {null_value:.3e} along an energy-flat '
                    f'direction (probe {null_angle:g} rad)')
            else:
                checks.append(_check(
                    'flat_direction', 'pass',
                    f'{split.n_soft} flat direction(s); a finite probe at '
                    f'{null_angle:g} rad changes e2 by {null_value:.3e}',
                    value=null_value, threshold=tol))
            if linear > float(null_probe_max) and linear > 0.1 * tol:
                warnings.append(
                    f'the linear soft-space coupling {linear:.3e} exceeds '
                    f'the probe {float(null_probe_max):.3e}; the probe may '
                    f'not lie along the maximum-coupling direction')
        else:
            if linear > tol:
                checks.append(_check(
                    'flat_direction', 'fail',
                    f'{split.n_soft} flat direction(s); |s_soft| = '
                    f'{soft_coupling:.3e} gives a first-order variation of '
                    f'{linear:.3e} at {null_angle:g} rad, above the '
                    f'tolerance {tol:.3e}',
                    value=linear, threshold=tol))
                reasons.append(
                    f'e2 couples to an energy-flat direction: |s_soft| = '
                    f'{soft_coupling:.3e}, variation {linear:.3e} at '
                    f'{null_angle:g} rad')
            elif soft_coupling > 0.0:
                checks.append(_check(
                    'flat_direction', 'undetermined',
                    f'{split.n_soft} flat direction(s); the first-order '
                    f'variation {linear:.3e} is below the tolerance '
                    f'{tol:.3e}, but a finite probe was not supplied and a '
                    f'flat direction has no curvature to bound the '
                    f'second-order term',
                    value=linear, threshold=tol))
                reasons.append(
                    'flat direction present; the e2 coupling is below the '
                    'linear detection floor but no finite probe rules out a '
                    'second-order variation')
                warnings.append(
                    'supply null_probe_max (or measure e2 along the flat '
                    'eigenvectors) to certify this point')
            else:
                checks.append(_check(
                    'flat_direction', 'pass',
                    f'{split.n_soft} flat direction(s) carry no e2 coupling '
                    f'(|s_soft| = {soft_coupling:.3e}); the correction is '
                    f'invariant along them',
                    value=soft_coupling))

    # 3. Residual-gradient bound on the hard subspace.
    if bound > tol:
        checks.append(_check(
            'first_order_bound', 'fail',
            f'|H_hard^-1 s_hard| |g_hard| = {bound:.3e} exceeds the '
            f'tolerance {tol:.3e}; the orbital gradient must reach '
            f'{required:.3e} on this point',
            value=bound, threshold=tol))
        reasons.append(
            f'first-order e2 bound {bound:.3e} exceeds the tolerance '
            f'{tol:.3e}; required orbital gradient {required:.3e}')
    else:
        checks.append(_check(
            'first_order_bound', 'pass',
            f'|H_hard^-1 s_hard| |g_hard| = {bound:.3e} <= {tol:.3e}',
            value=bound, threshold=tol))

    # 4. Repeatability between independently converged sets on this branch.
    if repeat_delta_e2 is not None:
        delta = abs(float(repeat_delta_e2))
        same_branch = (repeat_overlap is None
                       or float(repeat_overlap) >= float(overlap_min))
        if not same_branch:
            warnings.append(
                f'the repeat sets overlap by {float(repeat_overlap):.3f} < '
                f'{float(overlap_min):.3f}; their e2 difference '
                f'{delta:.3e} is branch spread and is not used as a '
                f'convergence test')
        elif delta > tol:
            checks.append(_check(
                'repeatability', 'fail',
                f'two independently converged sets on the same branch '
                f'differ in e2 by {delta:.3e}, above the tolerance '
                f'{tol:.3e}',
                value=delta, threshold=tol))
            reasons.append(
                f'same-branch repeat difference {delta:.3e} exceeds the '
                f'tolerance {tol:.3e}')
        else:
            checks.append(_check(
                'repeatability', 'pass',
                f'same-branch repeat difference {delta:.3e} <= {tol:.3e}',
                value=delta, threshold=tol))

    failed = [c for c in checks if c['status'] == 'fail']
    undetermined = [c for c in checks if c['status'] == 'undetermined']
    if failed:
        value = 'reject'
    elif undetermined:
        value = 'undetermined'
    else:
        value = 'accept'
    if value == 'accept':
        reasons.append(
            'certified: the point is a minimum, the first-order bound is '
            f'within {tol:.1e} Eh, and no flat direction carries e2')
    elif value == 'undetermined' and not reasons:
        reasons.append('the evidence is insufficient for a verdict')

    return PerturbationConvergenceReport(
        value=value, certified=(value == 'accept'),
        gradient_norm=float(np.linalg.norm(g)),
        gradient_max=float(np.abs(g).max()) if g.size else 0.0,
        e2_gradient_norm=float(np.linalg.norm(s)),
        hessian_lowest=split.lowest, hessian_largest=split.largest,
        n_negative=split.n_negative, n_soft=split.n_soft,
        curvature_threshold=split.curvature_threshold,
        e2_tolerance=tol, sensitivity=sensitivity,
        first_order_bound=bound, first_order_prediction=prediction,
        required_gradient=required, soft_e2_coupling=soft_coupling,
        null_probe_max=(None if null_probe_max is None else float(null_probe_max)),
        repeat_delta_e2=(None if repeat_delta_e2 is None
                         else float(repeat_delta_e2)),
        repeat_overlap=(None if repeat_overlap is None
                        else float(repeat_overlap)),
        checks=tuple(checks), reasons=tuple(reasons),
        warnings=tuple(warnings))


@dataclass
class BranchPoint:
    """One certified stationary point entering a branch comparison."""

    label: str
    energy: float
    e2: float
    gradient_norm: float = 0.0
    hessian_lowest: float = 0.0
    certified: bool = True
    theta: object = None
    overlap_to_reference: float = None


@dataclass
class BranchSpreadReport:
    """The branch dependence that the local criterion cannot remove.

    ``selected`` is the lowest-energy certified point, which is the
    declared selection rule: the CASSCF energy is variational and
    gauge-invariant, the correction is neither, so the correction must not
    be used to choose the branch.  ``ambiguous`` is true when two certified
    points differ in ``e2`` by more than the tolerance; the reported
    correction then carries that spread as an irreducible branch
    dependence.
    """

    selected: object
    n_certified: int
    energy_spread: float
    e2_spread: float
    ambiguous: bool
    degenerate: bool
    rows: tuple
    reasons: tuple

    def as_dict(self):
        return {
            'selected': (None if self.selected is None
                         else self.selected.label),
            'n_certified': self.n_certified,
            'energy_spread': self.energy_spread,
            'e2_spread': self.e2_spread,
            'ambiguous': self.ambiguous,
            'degenerate': self.degenerate,
            'rows': [dict(r) for r in self.rows],
            'reasons': list(self.reasons),
        }


def branch_spread(points, *, e2_tolerance=DEFAULT_E2_TOLERANCE,
                  energy_tol=1e-6):
    """Report the e2 spread across certified stationary points.

    Parameters
    ----------
    points : sequence of BranchPoint
        Refined stationary points, one per distinct orbital solution.
        Several refinements of the same solution must be merged by the
        caller (by wavefunction overlap); passing them separately would be
        reported as a spurious degenerate family.  Points whose
        ``certified`` is false are listed for context but excluded from the
        selection and the spread.
    e2_tolerance : float
        The correction difference above which the branch spread is larger
        than the requested accuracy.
    energy_tol : float
        Two minima within this energy are reported as degenerate; the
        energy then does not identify one point either.

    Returns
    -------
    BranchSpreadReport
    """
    points = list(points)
    certified = [p for p in points if p.certified]
    rows = tuple({
        'label': p.label,
        'energy': float(p.energy),
        'e2': float(p.e2),
        'gradient_norm': float(p.gradient_norm),
        'hessian_lowest': float(p.hessian_lowest),
        'certified': bool(p.certified),
        'overlap_to_reference': (None if p.overlap_to_reference is None
                                 else float(p.overlap_to_reference)),
    } for p in points)
    if not certified:
        return BranchSpreadReport(
            selected=None, n_certified=0, energy_spread=0.0, e2_spread=0.0,
            ambiguous=False, degenerate=False, rows=rows,
            reasons=('no certified stationary point was supplied',))
    ordered = sorted(certified, key=lambda p: float(p.energy))
    selected = ordered[0]
    energies = [float(p.energy) for p in certified]
    e2s = [float(p.e2) for p in certified]
    energy_spread = max(energies) - min(energies)
    e2_spread = max(e2s) - min(e2s)
    ambiguous = e2_spread > float(e2_tolerance)
    degenerate = (len(ordered) > 1
                  and abs(float(ordered[1].energy)
                          - float(selected.energy)) <= float(energy_tol))
    reasons = [
        f'{len(certified)} certified stationary point(s); the lowest '
        f'CASSCF energy selects {selected.label}'
    ]
    if ambiguous:
        reasons.append(
            f'the correction differs by {e2_spread:.3e} across certified '
            f'points, above the tolerance {float(e2_tolerance):.3e}; the '
            f'reported value is branch-dependent by that amount')
    if degenerate:
        reasons.append(
            f'the second certified point lies within {float(energy_tol):.1e} '
            f'Eh of the selected one; the energy does not identify one '
            f'point either')
    return BranchSpreadReport(
        selected=selected, n_certified=len(certified),
        energy_spread=float(energy_spread), e2_spread=float(e2_spread),
        ambiguous=bool(ambiguous), degenerate=bool(degenerate), rows=rows,
        reasons=tuple(reasons))
