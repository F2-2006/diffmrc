#!/usr/bin/env python
"""Lowest eigenpair without any decomposition of a large or degenerate matrix.

WHY THIS MODULE EXISTS

`core.eigh.eigh_safe` fixes the first-order reverse pass of an
eigendecomposition, but the second derivative remains NaN whenever the
matrix carries a bit-exact degenerate pair (measured 2026-09-24 on
jax 0.10.2): the outer differentiation mode re-enters the primal
decomposition and the raw 1/(w_j - w_i) returns -- for `eigh`,
`eigvalsh` and `eigh_safe` alike, and even for eigenvalue-only
functionals with a simple ground state. Linear molecules make this the
normal case rather than an exception: the CO/6-31G CAS(6,6) S_z-block CI
matrices carry bit-exact pairs in five of seven blocks (24 in the
400-determinant winning block). First-order gradients were fixed by
routing `cs_casci_blocked` through `eigh_safe` (commit a89693d); second
derivatives -- the HVP, the lambda_min spectrum, the NEVPT2-relaxed
frequencies -- need a construction that never differentiates a
decomposition at any level.

THE CONSTRUCTION (two levels, both decomposition-free)

Outer level: a fixed-depth Lanczos iteration with full reorthogonalization
builds an orthonormal basis Q (n x m) of the Krylov space of a
deterministic generic start vector, and the matrix is projected to
T = Q^T A Q (m x m). Every operation on the large matrix is a
matrix-vector product, an inner product or a norm.

Inner level: the lowest pair of T by **shifted inverse iteration, no
decomposition** (``_lowest_pair_solve``), in two phases -- a deep shift at
T's tridiagonal Gershgorin lower bound, then a fine shift at
Rayleigh - eps. All operations are dense solves on the m x m matrix,
matrix-vector products and inner products.

DERIVATIVES: the forward iteration is run under ``stop_gradient`` and the
returned pair is corrected by one implicit-function (Newton) step on the
eigen-equation, whose bordered (n+1) system is the only differentiated
object. The reason is measured, not stylistic (t249b, 2026-09-24): the
Lanczos basis's sensitivity to A is amplified by prod_k 1/beta_k --
exponentially in the depth. The *value* is variational and stays exact
(FD of the value is a clean plateau), but differentiating through the
basis carries a derivative error that grows with m: the second
derivative of trace(T) -- a purely smooth functional of the same chain --
measured AD 3.2e17 against finite difference 4.9e9 against value-based
estimate 5.5e5 at m = 48, while the same chain at m = 24 agreed to 2e-5.
With the Newton correction all derivatives come from the bordered solve
(condition ~ 1/gap, no 1/beta), which is the same implicit route the
Hessian work already uses for the orbital fixed point.

A ridge of 1e-8 x max|A| keeps the bordered solve finite at a degenerate
lowest eigenvalue, where the system is singular and the eigenvector's
derivative is genuinely undefined; the ridge only keeps values finite
(the derivatives of losing blocks are never consumed -- they multiply
zero cotangents, and a dense solve's vjp is linear in the cotangent, so
no 0*inf pathology arises).

DESIGN POINTS LEARNED WHILE BUILDING IT (t244/t244b/t244c, 2026-09-24)

- The small solve must be decomposition-free. With an `eigh_safe`-based
  inner solve the second derivative tracked the appearance of a bit-exact
  tie *inside T*: the admission of A's exact pair into the Ritz spectrum
  at m = 96 gave T a bit-exact zero gap, the HVP-vs-FD ratio degraded
  monotonically (m = 24: 9e-9; 48: 1.6e-3; 64: 6.5e-2; 96: 2.6) and
  turned NaN at m = 128. The disease recurs at every level that
  decomposes; the solve-only inner solve is immune to ties in T, which
  also removed the earlier depth cap: m may equal n, where the Krylov
  space spans the whole space and the Ritz pair is exact.
- The start vector is a deterministic generic vector, not the constant
  one: a symmetric start confines the Krylov space to the symmetric
  sector of a block-duplicated spectrum, and the iteration then closes
  onto an invariant subspace where the naive refill divided by a
  near-zero vector (measured |<u|u_e>| = 1.17).
- The inner iteration needs a shift that is *provably* below the lowest
  eigenvalue. A fine shift of Rayleigh - eps is not enough: when the deep
  phase has not converged into the ground basin, that shift lands above
  the lowest eigenvalue and the iteration converges to the wrong pair
  (measured t244d: a 12x12 block-duplicated case returned a value 0.33
  off). The residual-anchored shift (Rayleigh - residual - eps) is below
  the lowest eigenvalue throughout the ground basin.
- The shift must also stay a *bounded* distance below the eigenvalue
  (the _FINE_EPS floor). A bare residual anchor drives the shift to
  within ~1e-12 of the eigenvalue; the differentiated solves then carry
  Jacobians of order 1/gap^2, and the terms that cancel analytically
  stop cancelling in float64. Measured on the CO winning block: the
  value stayed exact (FD plateau across three steps) while the AD second
  derivative drifted from 1.6e-3 (m = 48) to 2.98 (m = 128) away from
  the finite-difference value. A 1e-3 floor bounds the condition number
  at ~1e3 and leaves the iteration's convergence rate (1e-3/gap per
  step) ample.

Semantics: the returned pair is the (converged) Rayleigh-Ritz pair of the
Krylov subspace, a smooth function of A; its derivatives are those of the
algorithm, equal to the true eigenpair derivatives up to the convergence
error.

SCOPE AND STATUS

Prototype (2026-09-24 night). Integration into `cs_casci_blocked` is
selected with ``eig_solver='lanczos'``. Accuracy degrades gracefully with
m; callers that need the last digits raise m.
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

__all__ = ['lowest_eigpair_lanczos']

# Below this Krylov coefficient the new direction is treated as exhausted
# (the space has closed onto an invariant subspace) and the basis is
# refilled from a deterministic generic vector. Both the division and the
# refill normalization carry denominator floors: a raw 0/0 in the untaken
# branch of a where() still reaches the reverse pass as 0 * inf and
# infects the selected branch.
_BETA_FLOOR = 1e-10
_FILL_SEED = 1.7
_FILL_SEED2 = 2.9
# Margin under the Gershgorin bound (covers the roundoff-level
# off-tridiagonal leakage of a reorthogonalized Lanczos projection) and
# the floor under the residual-anchored fine-phase shift. The floor
# bounds the shifted matrix's condition number at ~1/_FINE_EPS: driving
# the shift arbitrarily close to the eigenvalue (as a bare residual
# anchor does) makes the differentiated solves' Jacobians grow like
# 1/gap^2, and the analytically cancelling terms then fail to cancel in
# float64 -- measured on the CO winning block: the value stayed exact
# while the AD HVP drifted to 0.2-3x the finite-difference value as the
# Krylov depth grew.
_GERSH_MARGIN = 1e-8
# The fine shift keeps the residual anchor minus this floor. A floor near
# zero (1e-9) makes each fine step a near-projection and keeps the forward
# *value* stable under perturbation (a 1e-3 floor was measured to let the
# state selection jump between perturbed points, t244e); the derivative
# side is handled by the Newton correction and does not see the shift.
_FINE_EPS = 1e-9
# Ridge on the bordered IFT system, relative to the matrix scale.
_RIDGE = 1e-8


def _generic_start(n):
    """A deterministic, structure-free vector (no random state)."""
    idx = jnp.arange(1.0, n + 1.0)
    v = jnp.sin(_FILL_SEED * idx + 0.4) + 0.35 * jnp.cos(_FILL_SEED2 * idx)
    return v / jnp.linalg.norm(v)


def _lowest_pair_solve(t, n_deep=60, n_fine=4):
    """Lowest eigenpair of a small symmetric matrix, no decomposition.

    Phase 1 (deep): a fixed shift at the tridiagonal Gershgorin lower
    bound lies below the lowest eigenvalue by construction, so the
    iteration enters the lowest state's basin from any start. The phase
    must run long enough that the returned Rayleigh quotient sits *below
    the first excited level*; with a shallow-first estimate that is not
    automatic -- an earlier 10-step version landed the fine shift above
    the lowest eigenvalue on random block-duplicated spectra and the
    iteration then converged to the wrong pair (measured t244d: a 12x12
    case returned a value 0.33 away from the true one).

    Phase 2 (fine): the shift is ``Rayleigh - residual - eps``. Inside the
    ground basin the residual bound places it strictly below the lowest
    eigenvalue (Krylov-Weinstein), so the shifted matrix stays
    nonsingular, and as the residual shrinks the shift approaches the
    eigenvalue and the phase converges in a few steps.
    """
    m = t.shape[0]
    if m == 1:
        return t[0, 0], jnp.ones((1,))
    d = jnp.diag(t)
    beta = jnp.abs(jnp.diag(t, 1))
    row = d - jnp.concatenate([jnp.zeros(1), beta]) \
        - jnp.concatenate([beta, jnp.zeros(1)])
    sigma_deep = jnp.min(row) - _GERSH_MARGIN
    eye = jnp.eye(m)
    v = _generic_start(m)

    def step(vv, sigma):
        w = jnp.linalg.solve(t - sigma * eye, vv)
        return w / jnp.linalg.norm(w)

    for _ in range(n_deep):
        v = step(v, sigma_deep)
    for _ in range(n_fine):
        lam = v @ t @ v
        res = jnp.linalg.norm(t @ v - lam * v)
        v = step(v, lam - res - _FINE_EPS)
    lam = v @ t @ v
    return lam, v


def _lanczos_forward(a, m=None, v0=None):
    """Forward iteration: lowest Ritz pair of the m-step Krylov space.

    No derivative contract of its own -- the public entry runs this under
    ``stop_gradient`` and corrects it with an implicit step; see the
    module docstring.
    """
    n = a.shape[0]
    if n == 1:
        return a[0, 0], jnp.ones((1,))
    if m is None:
        m = min(int(n), 48)
    # Never let the space complete: at m = n the reorthogonalized remainder
    # vanishes, the deterministic refill produces a near-zero column and Q
    # loses orthonormality (measured: a returned vector 6e-4 off unit norm,
    # and the value 6e-4 off the eigenvalue). One dimension short is
    # harmless -- the Ritz pair of an (n-1)-dimensional Krylov space is
    # still variational, and the inner iteration supplies the accuracy.
    m = max(2, min(int(m), max(1, int(n) - 1)))
    a = 0.5 * (a + a.T)
    q = _generic_start(n) if v0 is None else v0 / jnp.linalg.norm(v0)
    Q = q[:, None]
    fill0 = _generic_start(n + 1)[: n]
    for _ in range(m - 1):
        w = a @ q
        # Full reorthogonalization, twice (the second pass removes the
        # round-off the first reintroduces).
        w = w - Q @ (Q.T @ w)
        w = w - Q @ (Q.T @ w)
        beta = jnp.linalg.norm(w)
        alive = beta > _BETA_FLOOR
        fill = fill0 - Q @ (Q.T @ fill0)
        fill = fill - Q @ (Q.T @ fill)
        fnorm = jnp.linalg.norm(fill)
        fill = fill / jnp.where(fnorm > _BETA_FLOOR, fnorm, 1.0)
        q = jnp.where(alive, w / jnp.where(alive, beta, 1.0), fill)
        Q = jnp.concatenate([Q, q[:, None]], axis=1)
    T = Q.T @ (a @ Q)
    T = 0.5 * (T + T.T)
    lam, v_small = _lowest_pair_solve(T)
    return lam, Q @ v_small


def _ift_step(a, lam, u):
    """One Newton step on ``(A - lam I) u = 0, u^T u = 1`` (bordered solve).

    Used twice: inside the forward polish (plain arithmetic), and as the
    differentiated correction whose bordered solve carries every
    derivative of the public entry.
    """
    n = a.shape[0]
    res = a @ u - lam * u
    nrm = 0.5 * (u @ u - 1.0)
    K = jnp.zeros((n + 1, n + 1))
    K = K.at[:n, :n].set(a - lam * jnp.eye(n))
    K = K.at[:n, n].set(-u)
    K = K.at[n, :n].set(u)
    rhs = jnp.concatenate([-res, -jnp.reshape(nrm, (1,))])
    ridge = _RIDGE * jnp.maximum(1.0, jnp.max(jnp.abs(a)))
    d = jnp.linalg.solve(K + ridge * jnp.eye(n + 1), rhs)
    return lam + d[n], u + d[:n]


def lowest_eigpair_lanczos(a, m=None, v0=None):
    """Lowest eigenpair of a symmetric matrix, differentiable to any order.

    The forward iterate (:func:`_lanczos_forward`) is polished by two
    Newton steps on the eigen-equation and then stopped; the returned pair
    moves by one more Newton step whose bordered solve is the only
    differentiated object. Three consequences: (i) the *value* reaches
    machine precision regardless of the Krylov depth (the polish corrects
    a shallow Ritz pair by orders of magnitude per step -- measured on a
    7-of-8 Krylov space: 1.2e-3 -> exact); (ii) every derivative -- first
    and second -- is the standard implicit-function derivative of the
    eigenpair, with no prod 1/beta amplification from the Krylov basis
    (the amplification is measured and documented in the module
    docstring); (iii) the ridge keeps the bordered solve finite at a
    degenerate lowest eigenvalue, where the eigenvector derivative is
    genuinely undefined.

    Parameters
    ----------
    a : (n, n) array
        Symmetric matrix.
    m : int, optional
        Krylov depth; defaults to ``min(n, 48)`` and is capped at ``n - 1``
        (a completed space breaks the basis's orthonormality through the
        refill branch; see the code comment). The polish removes the value
        dependence on m; deeper m only improves the starting pair.
    v0 : (n,) array, optional
        Start vector; defaults to a fixed generic vector.

    Returns
    -------
    (w0, u0) : the lowest eigenpair, ``||u0|| = 1``.
    """
    lam0, u0 = _lanczos_forward(a, m=m, v0=v0)
    n = a.shape[0]
    if n == 1:
        return lam0, u0
    for _ in range(2):
        lam0, u0 = _ift_step(a, lam0, u0)
    lam = jax.lax.stop_gradient(lam0)
    u = jax.lax.stop_gradient(u0)
    # Two chained live IFT steps, not one. The single-step tail drops the
    # *stopped* forward's own A-dependence through the term
    # (d u_new / d u0) * (d u0 / d A) -- and while d u_new / d u0 vanishes
    # quadratically with the residual, d u0 / d A does not: the forward's
    # vector is a product of Lanczos-basis directions whose sensitivity to
    # A grows like prod 1/beta. Measured (t261/t263, 2026-09-25): the
    # vector functional `sum(u)` had an AD second derivative 50% off the
    # FD/VD value on the CO winning block, while the eigenvalue functional
    # (whose stopped input has an O(1) sensitivity) agreed to 2e-6. Chaining
    # a second live step makes the total input-sensitivity O(residual^2) of
    # the first live step (~1e-24 after the polish), which kills the term.
    lam1, u1 = _ift_step(a, lam, u)
    return _ift_step(a, lam1, u1)
