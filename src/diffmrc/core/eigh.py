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

SCOPE: first order only. Micro-tested 2026-09-24 on a bit-exact pair: this
rule is finite for both the eigenvalue and the eigenvector paths, including
when the degeneracy lies away from the consumed column, whereas the raw
rule is NaN in all of those configurations. Second derivatives remain NaN
-- for eigenvalue-only functionals too, and for eigh, eigvalsh and this
rule alike -- because the outer differentiation mode re-enters the primal
decomposition and the raw 1/diff returns. A second-order-safe decomposition
needs an explicit second-order rule; until then, second-order quantities on
systems with bit-exact degeneracies must avoid the decomposition (finite
differences of the first-order gradient are the interim route).

USED BY

- ``core.nevpt2.impl.canonicalize_jax`` (active natural orbitals, sub-Fock
  blocks): the ThO/ECP case above.
- ``core.casci.cs_casci_blocked`` (per-S_z-block CI lowest eigenpair): the
  CO/6-31G case above. The block construction guarantees only that the
  winning block's ground state is non-degenerate; it does not keep the
  excited part of a block free of bit-exact pairs, and the raw rule is NaN
  through those pairs even though only column 0 is consumed.
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

__all__ = ['eigh_safe']


@jax.custom_jvp
def eigh_safe(a):
    """``jnp.linalg.eigh`` with a first-order-safe degeneracy rule.

    The primal is the standard decomposition; only the derivative rule is
    replaced (see the module docstring for the rule, its measurements and
    its first-order scope).
    """
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
