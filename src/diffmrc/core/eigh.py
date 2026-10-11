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

import jax
jax.config.update("jax_enable_x64", True)
# JAX 持久编译缓存（性能，2026-10-10）：批处理每分子一个进程时，jit 编译
# （实测 E 1797 s / grad 2960 s / hvp 16827 s，t393）此前每进程重付；
# 设 DIFFMRC_JAX_CACHE=<dir> 后跨进程复用（首进程编译、后续命中）。
# 配置级改动，零数值影响；与本模块探针同一「允许的 core 例外」类别。
_jax_cache = _os.environ.get('DIFFMRC_JAX_CACHE')
if _jax_cache:
    jax.config.update('jax_compilation_cache_dir', _jax_cache)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
from jax import numpy as jnp
import numpy as _np

__all__ = ['eigh_safe', 'make_eigh_mask', 'eigh_mask', 'eigh_cluster_ift',
           'set_trace_sink']

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
    w, u = jnp.linalg.eigh(a)
    L = u.T @ (0.5 * (da + da.T)) @ u
    diff = w[None, :] - w[:, None]
    tau = 1e-4
    F = diff / (diff ** 2 + tau ** 2)
    n = w.shape[0]
    C = L * F * (1.0 - jnp.eye(n))       # diagonal stays zero (normalization)
    dw = jnp.diag(L)
    du = u @ C
    return (w, u), (dw, du)


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
        w, u = jnp.linalg.eigh(a)
        L = u.T @ (0.5 * (da + da.T)) @ u
        diff = w[None, :] - w[:, None]
        keep = jnp.abs(diff) > tol
        F = jnp.where(keep, 1.0 / jnp.where(keep, diff, 1.0), 0.0)
        n = w.shape[0]
        C = L * F * (1.0 - jnp.eye(n))
        return (w, u), (jnp.diag(L), u @ C)

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
    gap; on the CO chain the clusters are bit-exact (1e-15) and the
    smallest real gap is 1.5e-2, so the default 1e-8 is safe.
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
