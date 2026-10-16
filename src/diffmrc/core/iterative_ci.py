import jax

jax.config.update("jax_enable_x64", True)
from jax import lax
from jax import numpy as jnp

from .safe_math import safe_divide

__all__ = ['minres_solve', 'davidson_lowest_k', 'lowest_k_eigpairs',
           'ci_matrix_from_integrals', 'iterative_ci_energies',
           'iterative_ci_ground_ci']

# Deterministic seed-block constants (structure-free start directions).
_SEED_A = 1.7
_SEED_B = 2.9
# Below this column norm an expansion direction counts as exhausted and is
# replaced by a deterministic refill, the core/lanczos.py convention.
_DIR_FLOOR = 1e-10
# Relative floor on |theta - H_ii| in the Davidson preconditioner: the
# amplification stays bounded near theta = H_ii, and the sign of a tiny
# denominator is preserved (the no-0/0 pattern of core/safe_math.py).
_PRECOND_FLOOR = 1e-3
# Denominator floor of the MINRES recurrence.
_MINRES_FLOOR = 1e-300
_MINRES_TOL = 1e-12
# Relative gap below which a returned pair counts as (near-)degenerate;
# matches the cluster tolerance of core.eigh.eigh_cluster_ift.
_CLUSTER_TOL = 1e-8
# Health thresholds of the cluster-complement solve. A degenerate partner
# outside the cluster makes the system singular: the residual stalls, or --
# when the right-hand side has no null component -- MINRES converges onto a
# minimum-residual point reached by drifting along the null direction, so
# the iterate is huge while the residual is small. Both signatures are
# needed; the residual alone misses the second. The growth ratio is
# dimensionless: |x| * (_CLUSTER_TOL * scale) / |rhs| > _STUCK_GROWTH means
# the iterate is larger than any solution with complement gaps at or above
# the cluster tolerance could justify.
_STUCK_RES = 1e-6
_STUCK_GROWTH = 10.0
# Discovery rounds per pair: a d-fold hidden cluster needs d - 1 rounds, so
# this covers hidden clusters up to six-fold; beyond it the ridge applies.
_DISCOVERY_ROUNDS = 4


def _seed_block(n, m):
    """m deterministic structure-free start directions (no random state)."""
    idx = jnp.arange(1.0, n + 1.0)
    cols = []
    for j in range(m):
        a = _SEED_A + 0.61 * j
        b = _SEED_B + 0.37 * j
        cols.append(jnp.sin(a * idx + 0.4 + 1.1 * j)
                    + 0.35 * jnp.cos(b * idx))
    return _orthonormalize(jnp.stack(cols, axis=1))


def _orthonormalize(w):
    """Orthonormal columns with a deterministic sign gauge (R diagonal >= 0)."""
    q, r = jnp.linalg.qr(w)
    return q * jnp.where(jnp.diag(r) >= 0.0, 1.0, -1.0)[None, :]


def _orth_against(v, w):
    """w orthogonalized against the orthonormal columns of v, twice."""
    w = w - v @ (v.T @ w)
    w = w - v @ (v.T @ w)
    return _orthonormalize(w)


def _expand(v, res, theta, diag):
    """Append one preconditioned-residual block to the Davidson subspace.

    ``diag`` (optional) is the diagonal of the operator; the preconditioner
    is the standard Davidson 1/(theta_j - H_ii) with a sign-preserving
    magnitude floor, so no denominator can reach 0. Directions whose
    projected norm lies below ``_DIR_FLOOR`` are exhausted and are replaced
    by deterministic refill directions orthogonal to ``v`` (the
    ``core/lanczos.py`` pattern).
    """
    if diag is not None:
        den = theta[None, :] - diag[:, None]
        floor = _PRECOND_FLOOR * jnp.maximum(1.0, jnp.max(jnp.abs(diag)))
        den = jnp.where(jnp.abs(den) >= floor, den,
                        jnp.where(den >= 0.0, floor, -floor))
        res = res / den
    w = res - v @ (v.T @ res)
    w = w - v @ (v.T @ w)
    col = jnp.linalg.norm(w, axis=0)
    fresh = _orth_against(v, _seed_block(v.shape[0], res.shape[1]))
    unit = safe_divide(w, col[None, :], _DIR_FLOOR)
    cand = jnp.where((col > _DIR_FLOOR)[None, :], unit, fresh)
    return jnp.concatenate([v, _orth_against(v, cand)], axis=1)


def davidson_lowest_k(matvec, n, k=1, tol=1e-10, max_cycles=60, n_expand=3,
                      diag=None, v0=None):
    """Lowest k eigenpairs of a symmetric operator, matrix-vector products only.

    Block Davidson with reorthogonalized expansions, a thick restart per
    cycle (the leading Ritz vectors plus the residual directions), and a
    convergence test on the true residual norms. Every operation on the
    large operator is a matvec; the only dense eigendecomposition acts on
    the small projected matrix.

    Parameters
    ----------
    matvec : callable
        ``x -> A x`` for a vector or a block of columns.
    n : int
        Operator dimension (a black-box matvec cannot report it).
    k : int
        Number of lowest eigenpairs to return.
    tol : float
        Convergence threshold on the largest residual norm
        ``||A x_i - theta_i x_i||``.
    max_cycles : int
        Cycle cap; the returned residuals say whether it was reached.
    n_expand : int
        Expansions per cycle.
    diag : array or None
        Operator diagonal for the Davidson preconditioner.
    v0 : array or None
        Start vector(s), placed before the deterministic seeds.

    Returns
    -------
    (theta, x, res, cycles, spectrum, ritz_block)
        Ascending eigenvalues, orthonormal eigenvector columns (deterministic
        sign gauge: the largest-magnitude entry of each column is positive),
        the residual norms, the number of cycles executed, the full Ritz
        spectrum of the final subspace (its lowest k entries are ``theta``;
        the rest approximate the eigenvalues above the returned set), and
        the full Ritz vectors of that subspace. The spectrum and the block
        let the implicit reverse pass read a degenerate cluster in full,
        including partners above the returned k.

    The forward path uses ``lax.while_loop`` with a data-dependent condition,
    so it cannot itself be reverse-differentiated; the differentiable entry
    is :func:`lowest_k_eigpairs`, which runs this under ``stop_gradient`` and
    supplies derivatives from the implicit-function rule.
    """
    n = int(n)
    k = max(1, min(int(k), n))
    m0 = min(n, 2 * k + 2)
    room = n - m0
    n_ex = 0 if room < k else min(int(n_expand), room // k)

    seeds = _seed_block(n, m0)
    if v0 is None:
        v = seeds
    else:
        v0a = jnp.asarray(v0)
        if v0a.ndim == 1:
            v0a = v0a[:, None]
        nv = min(m0, v0a.shape[1])
        base = _orthonormalize(v0a[:, :nv])
        if m0 - nv > 0:
            base = jnp.concatenate(
                [base, _orth_against(base, seeds[:, :m0 - nv])], axis=1)
        v = _orthonormalize(base)

    def _ritz(vv):
        t = vv.T @ matvec(vv)
        t = 0.5 * (t + t.T)
        w, y = jnp.linalg.eigh(t)
        theta = w[:k]
        block = vv @ y
        r = matvec(block[:, :k]) - block[:, :k] * theta[None, :]
        return theta, block, r, w

    def _cycle(vv):
        for _ in range(n_ex):
            theta, _, r, _ = _ritz(vv)
            vv = _expand(vv, r, theta, diag)
        theta, block, r, w = _ritz(vv)
        # Thick restart: the leading Ritz vectors (the converged block plus
        # the next-lowest ones) and the residual directions, filling m0
        # columns exactly. Keeping only the k returned Ritz vectors discards
        # the rest of the accumulated direction information each cycle and
        # was measured to slow the residual history by more than an order of
        # magnitude at fixed work (40 x 40, k = 1).
        keep = block[:, :m0 - k]
        return (_orthonormalize(
            jnp.concatenate([keep, _orth_against(keep, r)], axis=1)),
            theta, block, r, w)

    # One cycle before the loop: every carry state then has the post-expansion
    # shapes, so the loop's carry types are static.
    v1, theta1, block1, r1, w1 = _cycle(v)
    carry = (v1, jnp.int32(1), theta1, block1, r1, w1)

    def cond(c):
        _, cyc, _, _, r, _ = c
        return (cyc < max_cycles) & (jnp.max(jnp.linalg.norm(r, axis=0)) > tol)

    def body(c):
        vv, cyc, _, _, _, _ = c
        v_new, theta, block, r, w = _cycle(vv)
        return (v_new, cyc + 1, theta, block, r, w)

    _, cycles, theta, block, r, spectrum = lax.while_loop(cond, body, carry)
    res = jnp.linalg.norm(r, axis=0)
    # Deterministic sign gauge: the largest-magnitude entry is positive.
    x = block[:, :k]
    piv = jnp.argmax(jnp.abs(x), axis=0)
    signs = jnp.where(x[piv, jnp.arange(k)] >= 0.0, 1.0, -1.0)
    return theta, x * signs[None, :], res, cycles, spectrum, block


def minres_solve(matvec, b, tol=_MINRES_TOL, max_iter=None):
    """Solve ``A x = b`` for a symmetric (possibly indefinite) operator.

    MINRES (Paige-Saunders) in the unpreconditioned form: a short-recurrence
    Lanczos process with Givens rotations, so the storage is O(n) and every
    access to A is a matvec. This is the linear kernel of the implicit
    reverse pass of :func:`lowest_k_eigpairs`, whose shifted-projected
    operator is indefinite for every pair above the ground state (CG applies
    only to the ground pair, whose complement spectrum is strictly positive).

    Deviations between the recurrence's residual estimate and the true
    residual are round-off; the returned ``res`` is the recurrence's own
    ``|phibar|``, which coincides with the 2-norm of the residual in exact
    arithmetic. Every denominator in the recurrence carries a floor, so a
    breakdown returns a finite iterate with a large reported residual rather
    than an inf.

    Returns
    -------
    (x, res, iters)
        The iterate, the final residual estimate divided by ``||b||``
        (relative; 0.0 for ``b = 0``), and the iterations executed.
    """
    n = b.shape[0]
    if max_iter is None:
        max_iter = min(1000, 4 * n + 20)
    b_norm = jnp.linalg.norm(b)
    zero = jnp.zeros_like(b)
    carry = (
        jnp.int32(0), zero, b, b, zero, zero,
        jnp.float64(0.0), b_norm, jnp.float64(0.0), jnp.float64(0.0),
        b_norm, jnp.float64(-1.0), jnp.float64(0.0), b_norm,
    )

    def cond(c):
        it, _, _, _, _, _, _, _, _, _, _, _, _, res = c
        return ((it < max_iter) & (b_norm > _MINRES_FLOOR)
                & (res > tol * b_norm))

    def body(c):
        (it, x, r1, r2, w, w2, oldb, beta, dbar, epsln, phibar, cs, sn,
         _) = c
        beta_s = jnp.maximum(beta, _MINRES_FLOOR)
        v = r2 / beta_s
        y = matvec(v)
        # From the second iteration on, remove the component along r1.
        prev = jnp.where(it >= 1, beta / jnp.maximum(oldb, _MINRES_FLOOR),
                         0.0)
        y = y - prev * r1
        alfa = v @ y
        y = y - (alfa / beta_s) * r2
        beta_n = jnp.linalg.norm(y)
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        gamma = jnp.maximum(jnp.sqrt(gbar * gbar + beta_n * beta_n),
                            _MINRES_FLOOR)
        cs_n = gbar / gamma
        sn_n = beta_n / gamma
        phi = cs_n * phibar
        phibar_n = sn_n * phibar
        w_n = (v - oldeps * w2 - delta * w) / gamma
        x_n = x + phi * w_n
        return (it + 1, x_n, r2, y, w_n, w, beta, beta_n, -cs * beta_n,
                sn * beta_n, phibar_n, cs_n, sn_n, jnp.abs(phibar_n))

    it, x, *_, res = lax.while_loop(cond, body, carry)
    return x, res / jnp.maximum(b_norm, _MINRES_FLOOR), it


def _complement_matvec(a, lam, proj, eps):
    """Symmetric operator ``I (A - lam) I + P + eps I``, ``I = 1 - P``.

    The cluster block is carried as the identity, so a right-hand side
    orthogonal to the cluster yields a solution with exactly zero cluster
    components.
    """
    def mv(y):
        p = y - proj @ y
        q = a @ p - lam * p
        return proj @ y + (q - proj @ q) + eps * p
    return mv


def _unhealthy(x, res, rhs_norm, scale):
    """Whether a complement solve failed to produce a trustworthy iterate."""
    growth = (jnp.linalg.norm(x) * (_CLUSTER_TOL * scale)
              / jnp.maximum(rhs_norm, _MINRES_FLOOR))
    return (res > _STUCK_RES) | (growth > _STUCK_GROWTH)


def _pullback_solve(a, lam, proj, u_bar, ridge, scale):
    """Solve the cluster-complement system of one returned pair.

    The cluster known to the forward (the Ritz columns within the tolerance
    of ``lam``) is frozen out first. If a degenerate partner is absent from
    the Ritz space -- reachable only when the operator carries the partner's
    direction outside the whole Krylov space of the run, which a structured
    start can produce -- the complement is singular, and the solve says so
    through one of two signatures (:func:`_unhealthy`). The iterate's own
    direction is then a candidate null vector; when the Rayleigh quotient
    confirms it, the partner enters the cluster and the solve restarts on
    the larger complement, so the returned derivative follows the frozen
    convention for the partner as well. The round cap is
    ``_DISCOVERY_ROUNDS``; beyond it ``ridge`` (relative to ``max|A|``)
    regularizes the remaining complement, which pins a finite representative
    at a scale of order ``ridge^-1`` -- the last net, and the only case in
    which the ridge contributes to a returned value.
    """
    rhs = u_bar - proj @ u_bar
    x, res, _ = minres_solve(_complement_matvec(a, lam, proj, 0.0), rhs)
    carry = (proj, x, rhs, res, jnp.int32(0))

    def cond(c):
        proj_c, x_c, rhs_c, res_c, rounds = c
        v = x_c / jnp.maximum(jnp.linalg.norm(x_c), _MINRES_FLOOR)
        ray = jnp.abs(v @ (a @ v) - lam * (v @ v))
        w = v - proj_c @ v
        return ((rounds < _DISCOVERY_ROUNDS)
                & _unhealthy(x_c, res_c, jnp.linalg.norm(rhs_c), scale)
                & (ray < _CLUSTER_TOL * scale)
                & (jnp.linalg.norm(w) > _DIR_FLOOR))

    def body(c):
        proj_c, x_c, rhs_c, res_c, rounds = c
        v = x_c / jnp.maximum(jnp.linalg.norm(x_c), _MINRES_FLOOR)
        w = v - proj_c @ v
        w = w / jnp.maximum(jnp.linalg.norm(w), _DIR_FLOOR)
        proj_c = proj_c + jnp.outer(w, w)
        rhs_c = u_bar - proj_c @ u_bar
        x_c, res_c, _ = minres_solve(_complement_matvec(a, lam, proj_c, 0.0),
                                     rhs_c)
        return (proj_c, x_c, rhs_c, res_c, rounds + 1)

    proj, x, rhs, res, _ = lax.while_loop(cond, body, carry)
    net = _unhealthy(x, res, jnp.linalg.norm(rhs), scale)
    x = lax.cond(net,
                 lambda _: minres_solve(
                     _complement_matvec(a, lam, proj, ridge * scale), rhs)[0],
                 lambda _: x, None)
    return x


def _ift_pullback(a, ws, vs, spectrum, ritz_block, ct_w, ct_v, ridge):
    """Cotangent of the matrix for the returned lowest-k eigenpairs.

    For a simple eigenpair ``(lam, u)`` of ``A`` with ``||u|| = 1`` the
    implicit function theorem on ``((A - lam) u, u^T u - 1) = 0`` gives the
    pullback of cotangents ``(lam_bar, u_bar)`` as

        A_bar = -0.5 * (xi u^T + u xi^T),
        xi    = x - lam_bar * u,
        x     = solve of I (A - lam) I x = I u_bar,   I = 1 - P,

    with ``P`` the projector onto the degenerate cluster of ``lam`` inside
    the final Ritz block (``P = u u^T`` for a simple eigenvalue, which makes
    the system the standard Sternheimer equation on the orthocomplement).
    The eigenvalue part is exact and solve-free (``x = 0`` when
    ``u_bar = 0``), and the vector part is one MINRES solve per pair.

    Degeneracy, the complete rule: the cluster is read from the full Ritz
    spectrum of the converged subspace, so it covers partners inside the
    returned set and partners above it -- a k = 1 call on a two-fold
    degenerate ground state has its partner outside ``ws`` entirely. A
    partner absent even from the Ritz space is discovered by the solve
    itself and enters the cluster (:func:`_pullback_solve`). Restricting the
    solve to the orthocomplement of the whole cluster is the
    ``core/eigh.py`` frozen-rotation convention: the intra-cluster rotation
    carries no derivative, the per-pair contributions of a complete cluster
    sum to the regular projector derivative, and a complete cluster costs no
    ridge at all. ``ridge`` fires only when the discovery rounds are
    exhausted, where the derivative is genuinely undefined and a bounded
    finite representative replaces it, the ``core/lanczos.py`` convention.
    """
    k = ws.shape[0]
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(a)))
    a_bar = jnp.zeros_like(a)
    for i in range(k):
        lam = ws[i]
        u = vs[:, i]
        close = jnp.abs(spectrum - lam) <= _CLUSTER_TOL * scale
        proj = (ritz_block * close[None, :]) @ ritz_block.T
        x = _pullback_solve(a, lam, proj, ct_v[:, i], ridge, scale)
        x = x - proj @ x
        xi = x - ct_w[i] * u
        a_bar = a_bar - 0.5 * (jnp.outer(xi, u) + jnp.outer(u, xi))
    return 0.5 * (a_bar + a_bar.T)


def _lowest_k_primal(a, k, tol, max_cycles, n_expand, ridge):
    """Value path: Davidson under stop_gradient (no derivative of the loop)."""
    ws, vs, _, _, _, _ = davidson_lowest_k(lambda x: a @ x, n=a.shape[0],
                                           k=k, tol=tol,
                                           max_cycles=max_cycles,
                                           n_expand=n_expand, diag=jnp.diag(a))
    return lax.stop_gradient(ws), lax.stop_gradient(vs)


def _lowest_k_fwd(a, k, tol, max_cycles, n_expand, ridge):
    ws, vs, _, _, spec, block = davidson_lowest_k(
        lambda x: a @ x, n=a.shape[0], k=k, tol=tol, max_cycles=max_cycles,
        n_expand=n_expand, diag=jnp.diag(a))
    return ((lax.stop_gradient(ws), lax.stop_gradient(vs)),
            (a, ws, vs, spec, block))


def _lowest_k_bwd(k, tol, max_cycles, n_expand, ridge, res, ct):
    a, ws, vs, spec, block = res
    return (_ift_pullback(a, ws, vs, spec, block, ct[0], ct[1], ridge),)


_lowest_k_solve = jax.custom_vjp(_lowest_k_primal,
                                 nondiff_argnums=(1, 2, 3, 4, 5))
_lowest_k_solve.defvjp(_lowest_k_fwd, _lowest_k_bwd)


def lowest_k_eigpairs(a, k=1, tol=1e-10, max_cycles=60, n_expand=3,
                      ridge=1e-8):
    """Lowest k eigenpairs of a symmetric matrix, matvec-only and differentiable.

    WHY THIS MODULE EXISTS
        The in-stack CI solver diagonalizes its Hamiltonian densely
        (``core.casci``): O(dim^2) storage and O(dim^3) time in the
        determinant count, which places the (12,12)-and-above active spaces
        of the f-block targets beyond reach. This entry is the M1 milestone
        of the iterative-CI route (project record
        二A6_可微迭代CI_调研_20261002.md, route A): the forward solve works
        on matrix-vector products alone, and the reverse pass is an implicit
        function of the converged eigenpairs rather than a record of the
        iteration.

    FORWARD
        Block Davidson (:func:`davidson_lowest_k`) under ``stop_gradient``:
        the iteration -- subspace growth, restarts, the small projected
        eigendecomposition -- never appears in any derivative.

    REVERSE
        ``custom_vjp`` whose VJP is the standard eigenpair perturbation rule
        evaluated by solving the Sternheimer system on the complement of
        each eigenvector (:func:`_ift_pullback`). The residuals keep the
        operator, the pairs and the final Ritz block; every derivative-side
        linear solve goes through MINRES on the same matvec, at O(n) working
        memory per pair. The accuracy of the reverse rule
        is set by the forward residual, not by the derivative step: with
        ``||(A - theta) u|| <= tol`` the returned derivatives are those of
        the exact pair to O(tol / gap). The loop's own trajectory carries no
        derivative (the ``core/lanczos.py`` lesson: differentiating through
        a Krylov basis amplifies prod 1/beta).

    DEGENERACY
        The pullback is exact for simple pairs and follows the frozen-
        rotation convention of ``core/eigh.py`` inside a degenerate cluster:
        the intra-cluster rotation is removed from the solve, so the
        per-pair contributions of a complete cluster sum to the regular
        projector derivative and such a cluster costs no ridge. Partners
        that the forward's Ritz space missed are discovered by the solve
        and enter the cluster; the ridge fires only when that discovery is
        exhausted, where the derivative is genuinely undefined and a
        bounded finite representative replaces it. The individual
        eigenvectors of a degenerate cluster are not smooth functions of
        the matrix (their internal rotation is arbitrary), so a finite
        difference of a single returned column is meaningless there while
        the cluster sum is well defined; :func:`_ift_pullback` and
        :func:`_pullback_solve` document the rule and its measurements.

    A NOTE ON VALIDATION
        The reverse rule is limited by the forward residual, not by the
        solve: the returned derivative is that of the exact pair to
        O(residual / gap). A finite-difference cross-check is therefore
        bounded by the forward ``tol`` divided by the step -- with the
        default ``tol = 1e-10`` and a step of 1e-6 the check cannot go below
        about 1e-4, and it needs ``tol <= 1e-13`` to reach the 1e-8 band.

    SCOPE
        First-order implicit derivatives; the VJP itself is not built for a
        second differentiation. ``a`` is a dense matrix in this milestone --
        the M1 acceptance stage -- while the solver and the reverse kernel
        only ever touch ``matvec``, so the matrix-free M3 stage replaces the
        operator construction without touching this code.

    Returns
    -------
    (ws, vs)
        The k lowest eigenvalues and an (n, k) block of orthonormal
        eigenvectors, deterministic sign gauge.
    """
    return _lowest_k_solve(a, k, tol, max_cycles, n_expand, ridge)


def _casci_determinants(ncore, nact, nele_act):
    """The determinant list of ``core.casci.cs_casci``, in its construction order."""
    from itertools import combinations
    nso = 2 * (ncore + nact)
    core_so = tuple(range(2 * ncore))
    return [core_so + c
            for c in combinations(range(2 * ncore, nso), nele_act)]


def ci_matrix_from_integrals(h1e_mo, eri_mo, ncore=0, nact=2, nele_act=2):
    """Dense CI matrix over the full determinant space, built by ``core.casci``.

    The determinant list and its order are those of ``cs_casci``, and the
    Hamiltonian construction is its ``_block_H`` with the full index set, so
    the matrix is the one the dense solver diagonalizes. The import of
    ``core.casci`` is function-scoped because that module imports ``pyscfad``
    at module level; this entry therefore needs pyscfad installed even
    though neither the construction nor the solver uses it.
    """
    from .casci import _block_H
    dets = _casci_determinants(ncore, nact, nele_act)
    return _block_H(h1e_mo, eri_mo, dets, list(range(len(dets))),
                    ncore, nact, nele_act)


def iterative_ci_energies(h1e_mo, eri_mo, ncore=0, nact=2, nele_act=2, k=1,
                          **kwargs):
    """The k lowest CASCI energies of the given MO integrals, differentiable.

    Same determinant space as ``cs_casci``; ``k = 1`` with the defaults
    reproduces its ground-state energy in the residual tolerance.
    """
    h = ci_matrix_from_integrals(h1e_mo, eri_mo, ncore, nact, nele_act)
    ws, _ = lowest_k_eigpairs(h, k=k, **kwargs)
    return ws


def iterative_ci_ground_ci(h1e_mo, eri_mo, ncore=0, nact=2, nele_act=2,
                           **kwargs):
    """Ground-state energy and CI vector, in the ``cs_casci`` layout."""
    h = ci_matrix_from_integrals(h1e_mo, eri_mo, ncore, nact, nele_act)
    ws, vs = lowest_k_eigpairs(h, k=1, **kwargs)
    return ws[0], vs[:, 0]
