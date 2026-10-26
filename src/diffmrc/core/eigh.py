#!/usr/bin/env python
"""Degeneracy-safe eigendecomposition for the differentiable core.

WHY THIS MODULE EXISTS

The reverse-mode rule of ``jnp.linalg.eigh`` carries the factor
1/(w_j - w_i) for every eigenvector pair. Two failure modes follow, both
observed in production:

1) Roundoff-level gaps. Linear molecules have exact pi/delta degeneracies,
   so the computed gap of a symmetry-degenerate pair sits at the level of
   the computation's own roundoff (1e-15 .. 1e-16) and the factor is ~1e15:
   finite, but large enough to swamp the legitimate derivatives. Measured
   on ThO/ECP: the generalized-Fock core block (17 orbitals) has a minimum
   gap of 2.7e-15 and the virtual block (75 orbitals) 8.9e-16; with the raw
   rule the theta gradient of the CASSCF energy is all-NaN.

2) Bit-exact gaps. When the computed eigenvalues of a pair come out equal
   to the last bit, the factor is 1/0 = inf, and the reverse pass forms it
   for every pair whether or not that pair carries cotangent: the untouched
   entries are 0 * inf = NaN. Measured on CO/6-31G CAS(6,6): the S_z block
   CI matrices of the blocked solver contain 24 bit-exact pairs in the
   winning block alone (t237), and the raw rule returns NaN for a functional
   that consumes only the ground column -- the CASSCF theta gradient is
   all-NaN while the energy itself is finite (t232r).

THE RULE

The eigenbasis inside a degenerate subspace is arbitrary, so its derivative
is not fixed by the physics; the honest choice is to freeze the rotation
there. The JVP rule couples the subspaces through a Lorentzian,
C_ij = L_ij * diff/(diff^2 + tau^2) with L = U^T dA U and tau = 1e-4. For
|diff| >> tau it reproduces 1/diff to relative order (tau/diff)^2 -- below
1e-8 for every gap above 1e-2, which covers all validation systems -- and
near zero it goes linearly to zero (the freeze). A hard cutoff was tried
and rejected: it leaves 1/diff^3 spikes just outside its threshold whose
second derivative overflows.

The softening lives in the JVP rule (and the reverse pass, which jax builds
by transposing it), not in a hand-written VJP. That matters: with a
custom_vjp the forward tangent of u still comes from the standard eigh JVP,
so forward-over-reverse keeps the raw 1/diff and overflows to NaN as soon
as something consumes the eigenvectors -- while segments that only consume
the eigenvalues look fine because the offending tangent is dead code (t219).

SCOPE: first order under THIS rule. Micro-tested 2026-09-24 on a bit-exact
pair: this rule is finite for both the eigenvalue and the eigenvector paths,
including when the degeneracy lies away from the consumed column, whereas
the raw rule is NaN in all of those configurations. Second derivatives
remain NaN under this rule (and under ``eigh_mask``) -- for eigenvalue-only
functionals too, and for eigh, eigvalsh and the two rules alike -- because
the outer differentiation mode re-enters the primal decomposition and the
raw 1/diff returns.

CONTAINER: ``jnp.linalg.eigh`` returns the named tuple ``EighResult`` in
every jax version tested here (0.6.2, 0.10.2, 0.11.2); older versions may
return a plain tuple. A ``custom_jvp`` rule must return its tangent in the
same pytree container as the primal, so a hard-coded ``(dw, du)`` tuple
raises ``TypeError`` under ``grad(jit(f))`` (the outer trace mixes with the
traced primal) while ``eager`` and ``jit(grad(f))`` keep working
(reproduced on both 0.6.2 and 0.11.2 on 2026-10-26). Both rules below
rebuild the tangent from the actual primal output's tree structure, so the
same code works for either container.

The explicit second-order rule is ``eigh_cluster_ift`` below: it removes
the decomposition from the differentiated path entirely (stopped primal +
implicit per-column correction), so the second derivative is self-
consistent. The boundary audit (t57, 2026-10-11, N2 CAS(6,6) with
bit-exact degeneracies) measured HVP finiteness across four directions:
under the IFT route all three completed directions were finite
(|HVP| = 1.2e5 .. 6.6e5, 0 NaN entries), while ``eigh_mask`` returned
NaN in every direction (204/204). Second-order quantities on degenerate
systems should therefore route through ``eigh_cluster_ift``; the finite
values still carry large condition numbers (magnitudes ~1e5 on N2), so
regularization awareness remains part of the contract. Finite differences
of the first-order gradient stay the independent cross-check.

``eigh_soft2`` (P0.5, 2026-10-16) is the softening counterpart of
``eigh_cluster_ift``: it keeps the Lorentzian first-order rule of
``eigh_safe`` -- measured bit-equal to it, 6.4e-15 -- while replacing the
exact cluster resolvent by a Tikhonov-softened one, so its couplings stay
bounded by 1/(2 tau) instead of diverging with the physical gap and its
second derivative is finite without any hard spike. Its second derivative
is still a *frozen* rule (the decomposition is stopped at the evaluation
point): the channel through the decomposition's own response is absent,
the same "frozen gauge vs finite differences" gap documented at t269. FD
of the AD gradient recovers the full value-function derivative (measured
1.9% on the bit-exact fixture, 7e-6 on a well-separated control); see the
function docstring for the numbers.

USED BY

- ``core.nevpt2.impl.canonicalize_jax`` (active natural orbitals, sub-Fock
  blocks): the ThO/ECP case above.
- ``core.casci.cs_casci_blocked`` (per-S_z-block CI lowest eigenpair): the
  CO/6-31G case above. The block construction guarantees only that the
  winning block's ground state is non-degenerate; it does not keep the
  excited part of a block free of bit-exact pairs, and the raw rule is NaN
  through those pairs even though only column 0 is consumed.
"""
import json as _json
import os as _os
from collections import namedtuple

import jax
jax.config.update("jax_enable_x64", True)
# Persistent JAX compilation cache (performance): without it every
# per-molecule process pays the jit cost again (measured 1797 s / 2960 s /
# 16827 s for E / grad / hvp). Set DIFFMRC_JAX_CACHE=<dir> to reuse the
# cache across processes (first process compiles, later ones hit it).
# Configuration-only change, no numerical effect.
_jax_cache = _os.environ.get('DIFFMRC_JAX_CACHE')
if _jax_cache:
    jax.config.update('jax_compilation_cache_dir', _jax_cache)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
from jax import numpy as jnp
import numpy as _np

__all__ = ['eigh_safe', 'make_eigh_mask', 'eigh_mask', 'eigh_cluster_ift',
           'eigh_soft2', 'set_trace_sink', 'cluster_tolerance',
           'ClusterDecision', 'lowest_pair_auto']

# Diagnostic hook (added 2026-10-10 after the t397 lesson: the
# cluster-member scan had to monkey-patch this frozen module from
# outside, missed two other module-level bindings of the same functions,
# and silently recorded nothing). The sink belongs inside the kernel:
# it records every call as one JSON line and never alters a return
# value -- every failure inside it is swallowed, so a trace can never
# change a calculation.
_TRACE_SINK = None


def set_trace_sink(path):
    """Record kernel eigh calls as JSON lines to ``path`` (None = off).

    ``DIFFMRC_EIGH_TRACE`` seeds the same sink at import time. The hook
    is diagnostics only: values and derivatives are untouched.
    """
    global _TRACE_SINK
    _TRACE_SINK = path


def _trace_call(name, a):
    sink = _TRACE_SINK or _os.environ.get('DIFFMRC_EIGH_TRACE')
    if not sink:
        return
    try:
        arr = _np.asarray(a)             # raises under tracing; swallowed
        if arr.ndim != 2 or arr.shape[0] < 2:
            return
        w = _np.sort(_np.linalg.eigvalsh(arr))
        gaps = _np.diff(w)
        rec = {'solver': name, 'n': int(w.size),
               'min_gap': float(gaps.min()),
               'n_near': int((gaps < 1e-6).sum())}
        with open(sink, 'a', encoding='utf-8') as fh:
            fh.write(_json.dumps(rec) + '\n')
    except Exception:
        pass


def _primal_tangent(out, dw, du):
    """Tangent with the same pytree structure as an eigh primal output.

    The container returned by ``jnp.linalg.eigh`` is not fixed across jax
    versions: the tested versions (0.6.2, 0.10.2, 0.11.2) return the named
    tuple ``EighResult``, while older versions may return a plain tuple of
    ``(eigenvalues, eigenvectors)``. A ``custom_jvp`` rule must return its
    tangent in the *same* container as the primal, so the structure is
    taken from the actual primal output instead of being hard-coded.
    Measured on jax 0.6.2 and 0.11.2: a plain ``(dw, du)`` tangent raises
    ``TypeError: Custom JVP rule ... must produce ... the same
    container/pytree structure`` as soon as the function is traced inside
    ``jit`` and differentiated from the outside (``grad(jit(f))``);
    ``eager`` and ``jit(grad(f))`` hid the mismatch because neither path
    mixes the outer trace with a traced primal.
    """
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(out), [dw, du])


# ---------------------------------------------------------------------------
# Cluster-tolerance decision rule.
#
# The measured band (CO/6-31G and N2/cc-pVDZ at their stationary points,
# t280/t286/t287; recorded in SESSION_LOG_APPEND_20260923b.md section 21):
# symmetry pairs that are bit-exact at theta = 0 split by floating-point
# noise at theta* to 1.9e-8 .. 1.155e-5, while the smallest physical gap of
# the same spectra is 1.9e-2. The admissible tolerance is therefore a band,
# not a number, and it has to be derived from the spectrum in hand.
#
# The rule below makes the two-scale assumption explicit: gaps at or below
# ``noise_scale`` are numerical, gaps above it are physical. The tolerance
# is the geometric mean of the two scales, so it sits at the centre of the
# log band and stays clear of both edges; the ratio test rejects spectra in
# which the two scales are not separated, which is exactly the case the
# README describes as having no admissible tolerance (fallback: the
# decomposition-free Lanczos kernel). With the default relative noise
# scale (3e-5 of max|w|), the recorded CO spectra give tol = 1.26e-3
# (1-RDM block) and 7.55e-4 (Fock virtual block on a unit scale); supplying
# the measured noise 1.155e-5 gives 4.69e-4 for the Fock block. All are
# inside the measured window 1e-5 .. 1e-2 and inside the range where t286
# measured the chain-level AD-vs-FD residual to be 2.29e-4 at three
# different tolerances.
#
# The rule is a decision helper, not a differentiable function: it converts
# the spectrum to NumPy and returns a Python-level mode string.
# ---------------------------------------------------------------------------

# Relative numerical-noise scale. The measured upper edge of the split band
# is 1.155e-5 on spectra whose largest magnitude is order 1-10. The default
# has to keep the floor above that edge even for a unit-scale spectrum, so
# it is set at 3e-5: on the recorded CO spectra this gives floor 5.9e-5
# (1-RDM block, scale 1.98) and >= 3e-5 (Fock block), both above the
# measured 1.155e-5 and both three orders below the physical gaps.
_CLUSTER_REL_NOISE = 3e-5
# Required separation between the noise scale and the smallest physical
# gap. The recorded spectra have a ratio of 1.6e3 (dm1: 4.6e-8 vs 2.7e-2)
# to 5.7e3 (Fock virtual: 3.4e-6 vs 1.9e-2).
_CLUSTER_MIN_RATIO = 100.0

ClusterDecision = namedtuple(
    'ClusterDecision', ['mode', 'tol', 'noise_floor', 'physical_min',
                        'ratio', 'n_noise_gaps', 'n_physical_gaps'])


def cluster_tolerance(w, noise_scale=None, rel_noise=_CLUSTER_REL_NOISE,
                      min_ratio=_CLUSTER_MIN_RATIO):
    """Choose the degenerate-cluster tolerance from a spectrum.

    Parameters
    ----------
    w : array_like
        Eigenvalues (any order).
    noise_scale : float or None
        Absolute numerical-noise scale. When omitted it is estimated as
        ``rel_noise * max(1, max|w|)``. Supply a measured value when one is
        available (e.g. the split of a symmetry pair under a known
        perturbation).
    rel_noise, min_ratio : float
        Defaults documented above.

    Returns
    -------
    ClusterDecision
        ``mode`` is one of

        ``'none'``
            No gap lies at or below the noise floor: the spectrum has no
            numerical cluster, ``tol = 0`` (the freeze is disabled).
        ``'cluster'``
            A noise cluster exists and is separated from the physical
            spectrum by at least ``min_ratio``: ``tol`` is the geometric
            mean of the noise floor and the smallest physical gap.
        ``'cluster_all'``
            Every gap lies at or below the noise floor: the whole spectrum
            is one numerical cluster and ``tol`` is the noise floor.
        ``'lanczos'``
            The smallest physical gap is less than ``min_ratio`` times the
            noise floor: no tolerance separates the two scales. The caller
            must use a decomposition-free kernel on the pair it needs
            (:func:`lowest_pair_auto` does this).

    The two-scale assumption cannot be tested from one spectrum: a
    *physical* gap that falls below the noise floor is indistinguishable
    from numerical splitting. That is the residual open case of the
    README, and it is why the fallback exists rather than a lower
    tolerance.
    """
    w = _np.sort(_np.asarray(w, dtype=float).reshape(-1))
    n = w.size
    if n < 2:
        return ClusterDecision('none', 0.0, 0.0, float('inf'), float('inf'),
                               0, 0)
    gaps = _np.diff(w)
    scale = max(1.0, float(_np.max(_np.abs(w))))
    floor = (float(noise_scale) if noise_scale is not None
             else float(rel_noise) * scale)
    below = gaps <= floor
    above = ~below
    n_noise = int(below.sum())
    n_phys = int(above.sum())
    if n_phys == 0:
        return ClusterDecision('cluster_all', floor, floor, float('inf'),
                               float('inf'), n_noise, 0)
    physical_min = float(_np.min(gaps[above]))
    ratio = physical_min / floor if floor > 0.0 else float('inf')
    if n_noise == 0:
        # The smallest gap is already physical: nothing to freeze.
        return ClusterDecision('none', 0.0, floor, physical_min, ratio,
                               0, n_phys)
    if ratio < float(min_ratio):
        return ClusterDecision('lanczos', None, floor, physical_min, ratio,
                               n_noise, n_phys)
    tol = float(_np.sqrt(floor * physical_min))
    return ClusterDecision('cluster', tol, floor, physical_min, ratio,
                           n_noise, n_phys)


def lowest_pair_auto(a, **kwargs):
    """Lowest eigenpair with the tolerance decision applied.

    ``'cluster'`` and ``'cluster_all'`` route through
    :func:`eigh_cluster_ift` at the chosen tolerance, ``'none'`` through
    :func:`eigh_safe`, and ``'lanczos'`` through the decomposition-free
    kernel of ``core.lanczos``. This is an eager helper: the mode is a
    Python-level branch on the values, so it cannot be traced inside
    ``jit``. Differentiable callers that need a fixed rule should choose
    the kernel and tolerance themselves.

    The decision needs the spectrum, which this helper obtains with a dense
    ``jnp.linalg.eigvalsh``. It therefore targets the small dense blocks
    the rule was derived on (1-RDM and Fock blocks, dense CI blocks), not a
    matrix-free operator large enough that a decomposition is what one is
    trying to avoid; in that case pass a measured ``noise_scale`` and use
    the kernel directly.

    Returns ``(lam, u, decision)``; the eigenvalue is a 0-d array for the
    decomposition routes and a scalar array for the Lanczos route.
    """
    w_all = _np.asarray(jnp.linalg.eigvalsh(a))
    dec = cluster_tolerance(w_all, **kwargs)
    if dec.mode == 'lanczos':
        from .lanczos import lowest_eigpair_lanczos
        lam, u = lowest_eigpair_lanczos(a)
        return lam, u, dec
    if dec.mode == 'none':
        w, U = eigh_safe(a)
        return w[0], U[:, 0], dec
    w, U = eigh_cluster_ift(a, tol=dec.tol)
    return w[0], U[:, 0], dec


@jax.custom_jvp
def eigh_safe(a):
    """``jnp.linalg.eigh`` with a first-order-safe degeneracy rule.

    The primal is the standard decomposition; only the derivative rule is
    replaced (see the module docstring for the rule, its measurements and
    its first-order scope).
    """
    _trace_call('eigh_safe', a)
    return jnp.linalg.eigh(a)


@eigh_safe.defjvp
def _eigh_safe_jvp(primals, tangents):
    (a,), (da,) = primals, tangents
    out = jnp.linalg.eigh(a)
    w, u = out
    L = u.T @ (0.5 * (da + da.T)) @ u
    diff = w[None, :] - w[:, None]
    tau = 1e-4
    F = diff / (diff ** 2 + tau ** 2)
    n = w.shape[0]
    C = L * F * (1.0 - jnp.eye(n))       # diagonal stays zero (normalization)
    dw = jnp.diag(L)
    du = u @ C
    return out, _primal_tangent(out, dw, du)


def make_eigh_mask(tol=1e-6):
    """``eigh_safe`` with a hard mask instead of the Lorentzian softening.

    ``F = keep/diff`` with ``keep = |diff| > tol``: eigenvector pairs whose
    eigenvalues lie within ``tol`` are frozen with an *identically zero*
    coupling at every order, and pairs outside the mask get the exact
    ``1/diff`` -- more accurate than the Lorentzian at small-but-real gaps
    (relative deviation ``(tau/diff)^2``, e.g. 1e-2 at diff = 1e-3 with
    tau = 1e-4), and, crucially, free of the Lorentzian's ``1/tau^2``
    second-order term. That term was measured to contaminate the driver
    chain: at CO/6-31G theta = 0 (near-degenerate pi occupations entering
    the canonicalization) the AD-HVP read +4.10e3 while the finite
    difference of the gradient read +16.39 (t245, rel 0.996). The mask is
    the all-orders version of "freeze the rotation inside a degenerate
    subspace": the intra-cluster block couplings vanish identically.

    The hard cutoff needs the smallest *real* gap to sit well above
    ``tol`` -- the 1/diff^3 term of a pair just outside the mask is
    bounded by tol^-3, so with tol = 1e-6 every physical gap above ~1e-3
    is safe. Systems whose gap spectrum has structure near the tolerance
    should keep the Lorentzian route instead.
    """
    @jax.custom_jvp
    def eigh_mask(a):
        _trace_call('eigh_mask', a)
        return jnp.linalg.eigh(a)

    @eigh_mask.defjvp
    def _eigh_mask_jvp(primals, tangents):
        (a,), (da,) = primals, tangents
        out = jnp.linalg.eigh(a)
        w, u = out
        L = u.T @ (0.5 * (da + da.T)) @ u
        diff = w[None, :] - w[:, None]
        keep = jnp.abs(diff) > tol
        F = jnp.where(keep, 1.0 / jnp.where(keep, diff, 1.0), 0.0)
        n = w.shape[0]
        C = L * F * (1.0 - jnp.eye(n))
        return out, _primal_tangent(out, jnp.diag(L), u @ C)

    return eigh_mask


_EIGH_MASK = make_eigh_mask(1e-6)
eigh_mask = _EIGH_MASK                    # module-level instance, tol = 1e-6


def eigh_cluster_ift(a, tol=1e-8):
    """Eigendecomposition whose derivatives come from implicit solves.

    ``eigh_safe`` and ``eigh_mask`` replace the derivative *rule* but still
    call ``jnp.linalg.eigh`` inside it, so a second differentiation
    re-enters the primal decomposition and the raw 1/0 at a bit-exact
    degeneracy returns. Measured (t268, 2026-09-25): with the natural-
    orbital transform's derivative cut (``stop_gradient``) the CO chain's
    AD-HVP matched FD/VD to 5e-10; with either rule in place it was off by
    375x. This function removes the decomposition from the differentiated
    path entirely: the primal (w, u) is stopped, and each column is moved
    by one implicit correction

        U_new_i = u_i + (I - P_i) X_i,
        M_i X_i = (I - P_i) A u_i,
        M_i     = (I - P_i)(A - w_i I)(I - P_i) + P_i,

    with P_i the projector onto u_i's degenerate cluster (eigenvalues
    within ``tol`` of w_i). X_i is *identically zero* at the solution
    (both terms of (A - w_i)u_i either lie in the cluster or vanish by
    the projection), so values are untouched; the derivative of the
    expression is the standard eigenvector perturbation formula with
    intra-cluster rotations frozen identically at every order, and the
    second derivative is self-consistent because ``a`` is *live* inside
    M_i and the right-hand side (the same mechanism as the Lanczos
    kernel's bordered IFT step).

    ``tol`` must separate the degenerate clusters from the smallest real
    gap, and at theta* that band is not empty. At theta = 0 the symmetry
    pairs are bit-exact (1e-15), so any small tol works there; at the
    minimizer the rotated basis splits the same pairs to 1.9e-8 .. 1.2e-5
    (floating noise, not physics; t277/t280), which straddles the 1e-8
    signature default -- the pairs then fall outside the clusters and the
    derivative takes the raw 1/gap resolvent, measured as the 450x
    AD-vs-FD disagreement at theta* (t275/t282). The physical gaps start
    at 1.9e-2, three to six orders above the split band, so the working
    window is ~1e-7 .. 1e-2; use 1e-4 (mid-band, t286), which is what
    driver-level callers pass (t418). The signature default is kept at
    1e-8 for bit-exact spectra only.
    """
    _trace_call('eigh_cluster_ift', a)
    w, u = jnp.linalg.eigh(a)
    w_sg = jax.lax.stop_gradient(w)
    u_sg = jax.lax.stop_gradient(u)
    n = a.shape[0]
    eye = jnp.eye(n)
    close = jnp.abs(w_sg[None, :] - w_sg[:, None]) <= tol
    u_new = []
    for i in range(n):
        ui = u_sg[:, i]
        wi = w_sg[i]
        Pi = (u_sg * close[i][None, :]) @ u_sg.T          # projector of cluster(i)
        Ii = eye - Pi
        M = Ii @ (a - wi * eye) @ Ii + Pi
        # The minus sign is the derivative direction: the first-order
        # eigenvector equation is (A - w_i) du = -(I - P) dA u_i, so the
        # live expression must differentiate to -(I - P) M^-1 (I - P) dA u_i.
        # The value is untouched either way (x is identically zero at the
        # solution); measured on 2026-10-01, the positive sign returned the
        # whole vector derivative reversed (AD = -FD exactly, t327).
        xi = -jnp.linalg.solve(M, Ii @ (a @ ui))
        u_new.append(ui + Ii @ xi)
    u_out = jnp.stack(u_new, axis=1)
    # Rayleigh quotient of the *live* matrix at the stopped vectors: equals
    # w at the solution and carries u^T dA u as its derivative.
    w_out = jnp.diag(u_sg.T @ (a @ u_sg))
    return w_out, u_out


def eigh_soft2(a, tau=1e-4, tol=1e-4):
    """Second-order-safe eigendecomposition with a softened resolvent (P0.5).

    Same skeleton as :func:`eigh_cluster_ift` -- the decomposition (w, u) is
    stopped at the evaluation point and every output is an explicit smooth
    function of the live matrix -- but the exact cluster resolvent
    ``M^-1`` is replaced by the Tikhonov-softened one, applied per column as

        U_new_i = u_i - C_i (C_i^2 + tau^2 I)^-1 (C_i u_i),
        C_i     = (I - P_i)(a - w_i^ray I)(I - P_i),

    with ``w_i^ray = u_i^T a u_i`` the live Rayleigh quotient, ``P_i`` the
    projector onto the columns whose stopped eigenvalues lie within ``tol``
    of ``w_i``, and the small-residual grouping chosen so the solve's
    constant is not felt in the values (the correction is a value-zero
    expression; the solve acts on a ~1e-16 right-hand side).

    WHY THIS FORM. The first derivative of this expression is the
    Lorentzian rule of :func:`eigh_safe` -- measured against it on 5x5
    well-separated matrices: max deviation 6.4e-15 (eigenvectors) and
    2.8e-17 (eigenvalues), 2026-10-16. The cluster projector zeroes the
    intra-cluster couplings identically at every order (the flat freeze of
    :func:`eigh_mask`), which is what removes the ``F'(0) = 1/tau^2``
    gauge curvature that an unclustered Lorentzian carries: on the 4x4
    bit-exact-pair fixture below, the unclustered variant's HVP read
    +7.5e6 (scaling exactly as 1/tau^2 across tau = 1e-2 .. 1e-4) while
    this function reads +0.59 against a value-level reference of +0.70.
    Outside the cluster the softened coupling is bounded by 1/(2 tau),
    so the second derivative has no raw 1/gap^2 growth (the boundary
    audit's large-condition-number components); the trade is the
    (tau/gap)^2 first-order deviation of the Lorentzian, unchanged from
    :func:`eigh_safe`.

    WHAT THE SECOND DERIVATIVE IS. The frozen rule's second derivative
    differentiates the softened perturbation series, not the value
    function: the channel where the decomposition itself responds to the
    perturbation is absent (it cannot be represented without the raw
    1/gap). On the gauge-invariant loss of the fixture the AD HVP is
    +0.593 against a value-level reference +0.704 (16%); on an adversarial
    non-gauge loss on a well-separated matrix the gap can reach O(1/gn^2)
    scale. This is the same frozen-vs-FD semantics measured at t269
    (chain-level 7% residual) and t283 (5e-6 at CO/6-31G theta*). The FD
    of the AD gradient does carry the full derivative and is the accurate
    second-order route for this family (measured 1.9% on the fixture and
    7e-6 on the control, versus the value-level FD).

    VALUES. The correction vanishes identically at the evaluation point
    (every factor is a residual that is zero at the point), so the values
    are the plain ``jnp.linalg.eigh`` values: bit-equal on the fixture
    (du = 0.0) and 2.2e-15 on the control.

    ``tol`` must separate the numerical-split band from the smallest
    physical gap. Measured window on CO/6-31G at theta*: split band
    1.9e-8 .. 1.2e-5, physical gaps from 1.9e-2, so anything in
    ~1e-7 .. 1e-2 works; the default 1e-4 sits mid-band (t286). ``tol=0``
    disables the freeze for spectra without near-degeneracies.
    """
    _trace_call('eigh_soft2', a)
    a = 0.5 * (a + a.T)
    w, u = jnp.linalg.eigh(a)
    w0 = jax.lax.stop_gradient(w)
    u0 = jax.lax.stop_gradient(u)
    n = a.shape[0]
    eye = jnp.eye(n)
    tau2 = tau * tau
    close = jnp.abs(w0[None, :] - w0[:, None]) <= tol
    cols = []
    for i in range(n):
        ui = u0[:, i]
        wi = ui @ (a @ ui)
        Pi = (u0 * close[i][None, :]) @ u0.T
        Ii = eye - Pi
        C = Ii @ (a - wi * eye) @ Ii
        b = Ii @ ((a - wi * eye) @ ui)
        y = jnp.linalg.solve(C @ C + tau2 * eye, b)
        cols.append(ui - C @ y)
    return jnp.diag(u0.T @ (a @ u0)), jnp.stack(cols, axis=1)
