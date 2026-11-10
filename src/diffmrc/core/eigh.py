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
           'separation_certificate', 'ClusterDecision', 'lowest_pair_auto']

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
# Two scales, measured on CO, N2, H2O and OH (6-31G; active 1-RDM and Fock
# core/virtual blocks; 2026-11-10):
#
#   numerical   Split of a pair that is exactly degenerate in exact
#               arithmetic, under a random basis rotation inside the block:
#               1e-15 .. 5e-14 in float64 (thread count 1 and 4 alike) and
#               1e-7 .. 4e-7 in float32. At a *converged* stationary point
#               this is the whole noise band: the rotation is a similarity
#               transform and the spectrum is invariant.
#
#   residual    At a stationary point converged only to an orbital gradient
#               g, a rotation away from the optimum changes the active
#               space at O(g) and splits the same pairs by 0.1 .. 2 x |dE|
#               (measured through the production CI kernel: eps = 1e-4
#               gives 1e-6 .. 6e-5, eps = 1e-3 gives 4e-5 .. 6e-4; the
#               energy change is linear in eps, so the split tracks the
#               gradient). The 1e-8 .. 1.2e-5 band recorded at the CO
#               theta* is this residual, not floating point.
#
#   physical    Smallest non-degenerate gap of the same blocks: CO 1.5e-2
#               (dm1) / 7.5e-3 (Fock virtual), N2 5.2e-2 / 6.1e-2 (Fock
#               core 3.3e-3), H2O 3.8e-4 (dm1, CAS(8,8)), OH 1.0e-2. A
#               physical gap moves by at most 5e-14 under the gauge
#               replicates.
#
# The separation is therefore not guaranteed a priori: it is a property of
# the spectrum *and* of the orbital convergence. The rule takes the band
# from the caller when it is measured (``noise_band``), and otherwise falls
# back to a relative prior; a gap above the band but within
# ``_CLUSTER_MIN_RATIO`` of it is not certified as physical and routes to
# the decomposition-free kernel. A physical gap *inside* the band cannot be
# seen from one spectrum; :func:`separation_certificate` classifies it by
# stability across independent constructions (a numerical split moves, a
# physical gap does not).
#
# The rule is a decision helper, not a differentiable function: it converts
# the spectrum to NumPy and returns a Python-level mode string.
# ---------------------------------------------------------------------------

# Relative prior for the noise scale, used only when no measured band is
# supplied. It corresponds to an orbital gradient residual of ~1e-4 on a
# spectrum whose largest magnitude is order 1 (measured split ~ 0.1-2 x the
# gradient), which is the loose end of the production runs.
_CLUSTER_REL_NOISE = 3e-5
# Required separation between the band's upper edge and the smallest
# physical gap. Measured stability margin: numerical gaps have
# value/spread <= 3 across gauge replicates, physical gaps >= 1e3, so any
# threshold in 10 .. 1e3 separates the classes; 30 leaves a factor of 10
# for the sampling error of a finite replicate set.
_CLUSTER_MIN_RATIO = 30.0
# Default width of a band given as a single measured scale: hi / lo.
# Measured: float64 gauge splits span 1e-15 .. 5e-14 (50x), float32
# 1e-7 .. 4e-7 (4x), production splits at fixed eps span ~20x.
_CLUSTER_BAND_WIDTH = 30.0
# A gap is stable (physical) when its value exceeds its across-replicate
# spread by this factor. Numerical gaps move by O(100%) of their value;
# physical gaps move by <= 1e-13 relative.
_CLUSTER_STABILITY = 10.0
# Relative check on the decomposition-free kernel's value before its pair
# is accepted (see lowest_pair_auto): the kernel cannot resolve two
# eigenvalues closer than ~||A||/m^2, and on such a spectrum it converges
# to the wrong member of the pair. Measured worst case on the four systems'
# blocks after the refill fix: 3.2e-7 relative.
_FALLBACK_CHECK = 1e-6

ClusterDecision = namedtuple(
    'ClusterDecision', ['mode', 'tol', 'noise_floor', 'physical_min',
                        'ratio', 'n_noise_gaps', 'n_physical_gaps',
                        'certified', 'band_lo', 'fallback'])

_MODES = ('none', 'cluster', 'cluster_all', 'separation_unverified')


def _decision(mode, tol, noise_floor, physical_min, ratio, n_noise, n_phys,
              certified, band_lo):
    return ClusterDecision(mode, tol, noise_floor, physical_min, ratio,
                           n_noise, n_phys, certified, band_lo, '')


def cluster_tolerance(w, noise_scale=None, rel_noise=_CLUSTER_REL_NOISE,
                      min_ratio=_CLUSTER_MIN_RATIO, noise_band=None,
                      certified=None):
    """Choose the degenerate-cluster tolerance from a spectrum.

    Parameters
    ----------
    w : array_like
        Eigenvalues (any order).
    noise_scale : float or None
        Measured upper edge of the numerical-noise band, absolute. When
        omitted the relative prior ``rel_noise * max(1, max|w|)`` is used
        and the decision is marked uncertified.
    rel_noise, min_ratio : float
        Defaults documented above.
    noise_band : (lo, hi) or None
        Measured band, lower and upper edge. Takes precedence over
        ``noise_scale``. Obtain it from :func:`separation_certificate` when
        replicate constructions are available.
    certified : bool or None
        Override the certification flag. ``None`` marks a supplied band or
        scale as certified and the relative prior as uncertified.

    Returns
    -------
    ClusterDecision
        ``mode`` is one of

        ``'none'``
            No gap lies at or below the band, and the smallest gap is at
            least ``min_ratio`` times the band: nothing to freeze,
            ``tol = 0``.
        ``'cluster'``
            A numerical cluster exists and the smallest gap above the band
            is at least ``min_ratio`` times the band: ``tol`` is the
            geometric mean of the band's upper edge and that gap.
        ``'cluster_all'``
            Every gap lies at or below a *certified* band: the whole
            spectrum is one numerical cluster and ``tol`` is the band's
            upper edge.
        ``'separation_unverified'``
            A gap lies above the band but closer than ``min_ratio`` times
            it, or the whole spectrum is inside an *uncertified* band, or
            every gap is within the band while the band is uncertified.
            No tolerance can be certified from this spectrum; the caller
            must use a decomposition-free kernel on the pair it needs
            (:func:`lowest_pair_auto` routes to ``core.lanczos``, verifies
            its value, and degrades to the softened dense rule if the
            kernel cannot resolve the pair).

    Notes
    -----
    A physical gap *inside* the band is indistinguishable from numerical
    splitting in one spectrum, and this function will cluster it. That is
    the residual assumption of the single-spectrum route. It is removed
    only by :func:`separation_certificate`, which sees the same physical
    gap reproduced across independent constructions and classifies it by
    stability rather than by magnitude.
    """
    w = _np.sort(_np.asarray(w, dtype=float).reshape(-1))
    n = w.size
    if n < 2:
        return _decision('none', 0.0, 0.0, float('inf'), float('inf'),
                         0, 0, True, 0.0)
    gaps = _np.diff(w)
    scale = max(1.0, float(_np.max(_np.abs(w))))
    if noise_band is not None:
        lo, hi = (float(noise_band[0]), float(noise_band[1]))
        if not (0.0 <= lo <= hi):
            raise ValueError('noise_band must satisfy 0 <= lo <= hi')
        cert = True if certified is None else bool(certified)
    elif noise_scale is not None:
        hi = float(noise_scale)
        if hi < 0.0:
            raise ValueError('noise_scale must be non-negative')
        lo = hi / _CLUSTER_BAND_WIDTH
        cert = True if certified is None else bool(certified)
    else:
        hi = float(rel_noise) * scale
        lo = hi / _CLUSTER_BAND_WIDTH
        cert = False if certified is None else bool(certified)
    below = gaps <= hi
    above = ~below
    n_noise = int(below.sum())
    n_phys = int(above.sum())
    if n_phys == 0:
        if cert:
            return _decision('cluster_all', hi, hi, float('inf'),
                             float('inf'), n_noise, 0, True, lo)
        # An uncertified prior cannot turn a whole spectrum into one
        # numerical cluster: any of these gaps may be physical.
        return _decision('separation_unverified', None, hi, float('inf'),
                         float('inf'), n_noise, 0, False, lo)
    physical_min = float(_np.min(gaps[above]))
    ratio = physical_min / hi if hi > 0.0 else float('inf')
    if ratio < float(min_ratio):
        # Includes the no-cluster case: a smallest gap that close to the
        # band is not certified as physical (the old rule returned 'none'
        # here and differentiated it with a bare 1/gap).
        return _decision('separation_unverified', None, hi, physical_min,
                         ratio, n_noise, n_phys, cert, lo)
    if n_noise == 0:
        return _decision('none', 0.0, hi, physical_min, ratio, 0, n_phys,
                         cert, lo)
    tol = float(_np.sqrt(hi * physical_min))
    return _decision('cluster', tol, hi, physical_min, ratio, n_noise,
                     n_phys, cert, lo)


def separation_certificate(spectra, min_ratio=_CLUSTER_MIN_RATIO,
                           stability=_CLUSTER_STABILITY):
    """Classify gaps by stability across independent constructions.

    Parameters
    ----------
    spectra : (k, n) array_like
        ``k >= 2`` spectra of the *same* physical matrix, each produced by
        an independent numerical construction: a different gauge rotation
        inside the block, a different BLAS thread count, or a different
        precision. The constructions must vary the noise source; rerunning
        the same code with the same inputs produces zero spread and is
        rejected. Sorted internally.
    min_ratio, stability : float
        Defaults documented above.

    Returns
    -------
    ClusterDecision
        Same fields as :func:`cluster_tolerance`, with ``mode`` from
        ``'none'``, ``'cluster'``, ``'cluster_all'`` or
        ``'separation_unverified'``. Every returned decision is certified:
        the band comes from the observed spread. A gap is *numerical* when
        its median value is at most ``stability`` times its across-replicate
        spread, and *physical* otherwise. The decision is
        ``'separation_unverified'`` when

        - every gap is numerical but the replicate set shows no variation
          at all (no evidence), or
        - a physical gap is not separated from the numerical band by
          ``min_ratio``, or
        - a physical gap lies inside the numerical band (its value is at
          or below the largest numerical gap).

    The last case is the one a single spectrum cannot see: a physical gap
    inside the band is clustered by :func:`cluster_tolerance`, while here
    it is stable across constructions and therefore detected.
    """
    arr = _np.sort(_np.asarray(spectra, dtype=float), axis=1)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError('need at least two spectra of equal length')
    k, n = arr.shape
    if n < 2:
        return _decision('none', 0.0, 0.0, float('inf'), float('inf'),
                         0, 0, True, 0.0)
    gaps = _np.diff(arr, axis=1)
    value = _np.median(gaps, axis=0)
    spread = gaps.max(axis=0) - gaps.min(axis=0)
    if float(spread.max()) <= 0.0:
        # No variation: the replicate set cannot certify anything.
        return _decision('separation_unverified', None, float('nan'),
                         float('inf'), float('inf'), 0, 0, False, 0.0)
    stable = (value > 0.0) & (value > float(stability) * spread)
    n_phys = int(stable.sum())
    n_noise = int((~stable).sum())
    if n_noise == 0:
        return _decision('none', 0.0, 0.0, float(_np.min(value)), float('inf'),
                         0, n_phys, True, 0.0)
    band_hi = float(_np.max(_np.abs(gaps[:, ~stable])))
    band_lo = float(_np.min(_np.abs(gaps[:, ~stable])))
    if band_hi <= 0.0:
        # Bit-exact degeneracies reproduced in every construction: the
        # cluster is real but its width is below the representable scale.
        band_hi = float(_np.finfo(float).eps
                        * max(1.0, float(_np.max(_np.abs(arr)))))
        band_lo = min(band_lo, band_hi)
    if n_phys == 0:
        return _decision('cluster_all', band_hi, band_hi, float('inf'),
                         float('inf'), n_noise, 0, True, band_lo)
    physical_min = float(_np.min(value[stable]))
    ratio = physical_min / band_hi
    if physical_min <= band_hi or ratio < float(min_ratio):
        # A physical gap inside the band, or one too close to it: the two
        # scales cannot be separated by a tolerance.
        return _decision('separation_unverified', None, band_hi,
                         physical_min, ratio, n_noise, n_phys, True, band_lo)
    tol = float(_np.sqrt(band_hi * physical_min))
    return _decision('cluster', tol, band_hi, physical_min, ratio, n_noise,
                     n_phys, True, band_lo)


def lowest_pair_auto(a, replicate_spectra=None, **kwargs):
    """Lowest eigenpair with the tolerance decision applied.

    ``'cluster'`` and ``'cluster_all'`` route through
    :func:`eigh_cluster_ift` at the chosen tolerance and ``'none'`` through
    :func:`eigh_safe`. ``'separation_unverified'`` routes through the
    decomposition-free kernel of ``core.lanczos`` and *verifies its value*
    against the dense lowest eigenvalue: the kernel cannot resolve two
    eigenvalues closer than about ``||A|| / m^2`` and converges to the
    wrong member of such a pair, so a value off by more than
    ``_FALLBACK_CHECK`` relative degrades to :func:`eigh_safe` and the
    decision records ``fallback = 'dense_safe'``. The alternative -- a
    wrong eigenpair returned silently -- is the failure this verification
    was added for (measured on a 5x5 matrix with gaps 3e-6 .. 1e-2: the
    kernel returned the first excited state, energy off by the full gap).

    Parameters
    ----------
    a : (n, n) array
        Symmetric matrix.
    replicate_spectra : (k, n) array_like or None
        When given, the decision comes from :func:`separation_certificate`
        on these spectra instead of from the single spectrum of ``a``.
        They must be independent constructions of the same block (see
        there).
    **kwargs
        Forwarded to :func:`cluster_tolerance` (without replicates) or to
        :func:`separation_certificate` (with replicates; only
        ``min_ratio`` and ``stability`` are meaningful).

    This is an eager helper: the mode is a Python-level branch on the
    values, so it cannot be traced inside ``jit``. Differentiable callers
    that need a fixed rule should choose the kernel and tolerance
    themselves.

    Returns ``(lam, u, decision)``; the decision's ``fallback`` field
    records which route produced the pair: ``''`` for the decomposition
    routes, ``'lanczos'`` for a verified decomposition-free pair,
    ``'dense_safe'`` for a dense pair after a failed verification.
    """
    w_all = _np.asarray(jnp.linalg.eigvalsh(a))
    if replicate_spectra is not None:
        kw = {k: kwargs[k] for k in ('min_ratio', 'stability')
              if k in kwargs}
        dec = separation_certificate(replicate_spectra, **kw)
    else:
        dec = cluster_tolerance(w_all, **kwargs)
    if dec.mode == 'separation_unverified':
        from .lanczos import lowest_eigpair_lanczos
        lam, u = lowest_eigpair_lanczos(a)
        scale = max(1.0, float(_np.max(_np.abs(w_all))))
        if abs(float(lam) - float(w_all[0])) <= _FALLBACK_CHECK * scale:
            return lam, u, dec._replace(fallback='lanczos')
        w, U = eigh_safe(a)
        return w[0], U[:, 0], dec._replace(fallback='dense_safe')
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
