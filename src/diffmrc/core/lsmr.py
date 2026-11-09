"""Least-squares response solves without an iteration cap.

WHY THIS MODULE EXISTS

``core.hessian.ImplicitHessianHVPOps`` solves the orbital response
equation ``H_tt x = M v`` matrix-free, once per geometric column.  The
orbital Hessian is symmetric but badly conditioned: on H2O/6-31G
CAS(2,2) its eigenvalues span ``[-1.6e-2, 8.6e+1]`` and on CO/6-31G
CAS(2,2) ``[1e-11, 8.8e+1]``.  LSMR needs hundreds of iterations to
drive the residual to the level where the assembled Hessian matches
the dense pseudo-inverse route, and ``scipy.sparse.linalg.lsmr`` caps
the iteration count at ``min(m, n)`` whenever ``maxiter`` is left at
``None`` -- which is below what these systems need.  The response
solve therefore used a hand-set cap of ``8 n_par`` (at least 200).

A fixed cap is the wrong control.  It is simultaneously too large for
an easy column and too small for a hard one, and it says nothing about
whether the returned vector is converged.  What the solve actually
needs is

* a stopping rule stated in the quantity the caller cares about --
  the relative residual ``||r|| / ||b||``;
* no iteration cap by default, with wall-clock and stagnation guards
  so a hopeless system terminates instead of spinning;
* deflation of the near-null orbital directions, which are what make
  LSMR slow in the first place.

WHAT DEFLATION MEANS HERE

The dense route applies ``numpy.linalg.pinv(H_tt, rcond)``: directions
with ``|lambda| < rcond * lambda_max`` are dropped, the rest are
inverted.  The matrix-free route has to reproduce that action without
forming the matrix.  Deflation splits the space into

    V   spanned by the eigenvectors of the smallest |lambda|,
    V^perp   everything else,

solves the ``V`` block exactly from its Rayleigh quotients (applying
the same ``rcond`` cutoff inside it), and hands ``V^perp`` to LSMR.
LSMR then sees an operator whose spectrum is bounded away from zero by
the deflation threshold instead of one that reaches 1e-13 of
``lambda_max``.

The basis is built matrix-free by :func:`near_null_basis`: seed with
the coordinate directions that carry the smallest diagonal entries,
expand with a few block-Krylov steps, and Rayleigh-Ritz.  On the two
test systems the small eigenvectors are localised on one or two
rotation pairs (participation ratio 1.0-2.6), so the diagonal seed
finds them; the Krylov expansion supplies the coupling that the
diagonal alone misses.

Because the basis is approximate, one split solve is not exact.  Each
refinement round re-solves the block with the current residual and
sends the complement part back to LSMR, which converges to the exact
pseudo-inverse solution without requiring the Ritz vectors to be
accurate to machine precision.

STORAGE

``lsmr_solve`` keeps six vectors of length ``n`` regardless of the
iteration count; nothing of size ``n**2`` is formed.  Memory is
therefore bounded by construction, and the only guard the caller has
to supply is wall-clock time.
"""
import time

import numpy as np

__all__ = ['sym_ortho', 'lsmr_solve', 'near_null_basis',
           'deflated_solve', 'projected_operator']


def sym_ortho(a, b):
    """Givens rotation ``(c, s, r)`` with ``[c s; -s c] [a; b] = [r; 0]``.

    Branching on the larger magnitude keeps ``tau`` below one, which is
    what makes the rotation stable when one component dominates.
    """
    if b == 0.0:
        return float(np.sign(a)), 0.0, float(abs(a))
    if a == 0.0:
        return 0.0, float(np.sign(b)), float(abs(b))
    if abs(b) > abs(a):
        tau = a / b
        s = float(np.sign(b)) / np.sqrt(1.0 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = float(np.sign(a)) / np.sqrt(1.0 + tau * tau)
        s = c * tau
        r = a / c
    return float(c), float(s), float(r)


def lsmr_solve(matvec, b, rtol=1e-10, atol=0.0, maxiter=None, conlim=1e15,
               time_limit=None, stall_iters=200, record=False):
    """LSMR for a symmetric operator, stopped on the relative residual.

    The recurrences are those of Fong and Saunders (2011), the same
    ones ``scipy.sparse.linalg.lsmr`` implements, so the two agree
    iteration for iteration; what differs is the control.

    Parameters
    ----------
    matvec : callable
        ``x -> A x``.  ``A`` must be symmetric, so the same callable
        stands in for the transpose product.
    b : array_like
        Right-hand side.
    rtol : float
        Stop when ``||r|| / ||b|| <= rtol``.  This is the criterion the
        caller can interpret: it bounds the relative residual of the
        response equation directly.
    atol : float
        Optional second criterion on ``||A^T r|| / (||A|| ||r||)``.
        Zero disables it.  It matters for a consistent system with an
        exact null direction, where the residual cannot fall below the
        null-space component of ``b`` but the normal equations can
        still be satisfied; use it only when the right-hand side is
        known to be orthogonal to the null space.
    maxiter : int or None
        Iteration cap.  ``None`` -- the default -- means no cap: the
        solve runs until it converges or one of the guards fires.
    conlim : float
        Stop when the condition estimate exceeds this value.  The
        default is loose on purpose; the point of the module is to
        solve ill-conditioned systems, not to refuse them.
    time_limit : float or None
        Wall-clock guard in seconds.  When it expires the current
        iterate is returned with ``istop = 9`` so the caller can see
        that the budget, not the tolerance, ended the solve.
    stall_iters : int
        Stagnation guard: if ``||r||`` has not improved by more than a
        factor ``1 - 1e-14`` over this many iterations, stop with
        ``istop = 10``.  Without it a round-off floor would be an
        infinite loop once the cap is removed.
    record : bool
        Keep the residual history.

    Returns
    -------
    x : ndarray
    info : dict
        ``itn``, ``istop``, ``normr``, ``normar``, ``normA``, ``condA``,
        ``normx``, ``reason`` (a word for the same thing as ``istop``)
        and, when ``record`` is set, ``history``.
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.size
    x = np.zeros(n)
    normb = float(np.linalg.norm(b))
    info = dict(itn=0, istop=0, normr=normb, normar=0.0, normA=0.0,
                condA=1.0, normx=0.0, reason='zero_rhs', history=[])
    if normb == 0.0 or n == 0:
        return x, info

    u = b / normb
    v = np.asarray(matvec(u), dtype=float).reshape(-1)
    alpha = float(np.linalg.norm(v))
    info['normA'] = alpha
    info['normar'] = alpha * normb
    if alpha == 0.0:
        info['reason'] = 'zero_operator'
        return x, info
    v = v / alpha

    zetabar = alpha * normb
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0
    h = v.copy()
    hbar = np.zeros(n)

    betadd = normb
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0
    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normr = normb
    normar = alpha * normb
    condA = 1.0
    normx = 0.0

    t_start = time.perf_counter()
    best = normr
    last_improve = 0
    itn = 0
    history = []
    reason = 'maxiter'
    istop = 7

    while True:
        if maxiter is not None and itn >= maxiter:
            reason, istop = 'maxiter', 7
            break
        itn += 1

        u = np.asarray(matvec(v), dtype=float).reshape(-1) - alpha * u
        beta = float(np.linalg.norm(u))
        if beta > 0.0:
            u = u / beta
            v = np.asarray(matvec(u), dtype=float).reshape(-1) - beta * v
            alpha = float(np.linalg.norm(v))
            if alpha > 0.0:
                v = v / alpha

        chat, shat, alphahat = sym_ortho(alphabar, 0.0)
        rhoold = rho
        c, s, rho = sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar = hbar * (-(thetabar * rho / (rhoold * rhobarold))) + h
        x = x + (zeta / (rho * rhobar)) * hbar
        h = h * (-(thetanew / rho)) + v

        # Apply Qhat_{k,2k+1} (damp is zero here), then Q_{k,k+1}; the
        # residuals of the bidiagonalization are tracked through the
        # same rotations that update x, which is what makes the ||r||
        # estimate free.
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = np.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        normA2 = normA2 + beta * beta
        normA = np.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        normar = abs(zetabar)
        normx = float(np.linalg.norm(x))
        if record:
            history.append(float(normr))

        test1 = normr / normb
        test2 = (normar / (normA * normr)) if (normA * normr) > 0 else np.inf
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / normb)
        rtol_eff = rtol + atol * normA * normx / normb

        if normr < best * (1.0 - 1e-14):
            best = normr
            last_improve = itn

        if itn >= 1 and 1.0 + t1 <= 1.0:
            reason, istop = 'machine_residual', 4
            break
        if test3 <= 1.0 / conlim:
            reason, istop = 'condition_limit', 3
            break
        if atol > 0.0 and test2 <= atol:
            reason, istop = 'normal_equations', 2
            break
        if test1 <= rtol_eff:
            reason, istop = 'relative_residual', 1
            break
        if itn - last_improve >= stall_iters:
            reason, istop = 'stagnation', 10
            break
        if time_limit is not None and (
                time.perf_counter() - t_start) > time_limit:
            reason, istop = 'time_limit', 9
            break

    info.update(itn=itn, istop=istop, normr=float(normr),
                normar=float(normar), normA=float(normA),
                condA=float(condA), normx=float(normx), reason=reason,
                history=history)
    return x, info


def projected_operator(matvec, basis):
    """``x -> (I - P) A (I - P) x`` with ``P`` the projector on ``basis``."""
    B = np.asarray(basis, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    def proj(v):
        return v - B @ (B.T @ v)

    def op(v):
        v = np.asarray(v, dtype=float).reshape(-1)
        return proj(np.asarray(matvec(proj(v)), dtype=float).reshape(-1))

    return op


def _orthonormalize(X, tol=1e-12):
    """Modified Gram-Schmidt with a second pass, dropping dependent columns."""
    Q = np.zeros_like(X)
    k = 0
    for j in range(X.shape[1]):
        v = X[:, j].copy()
        for _ in range(2):
            for i in range(k):
                v -= (Q[:, i] @ v) * Q[:, i]
        nv = np.linalg.norm(v)
        if nv > tol * max(np.linalg.norm(X[:, j]), 1.0):
            Q[:, k] = v / nv
            k += 1
    return Q[:, :k]


def _rayleigh_ritz(op_matvec, B):
    """Ritz pairs of the symmetric operator in the span of ``B``."""
    W = np.column_stack([np.asarray(op_matvec(B[:, j]), dtype=float)
                         for j in range(B.shape[1])])
    T = 0.5 * (B.T @ W + (B.T @ W).T)
    w, U = np.linalg.eigh(T)
    V = B @ U
    resid = np.column_stack([
        np.asarray(op_matvec(V[:, j]), dtype=float) - w[j] * V[:, j]
        for j in range(w.size)])
    return w, V, np.linalg.norm(resid, axis=0)


def near_null_basis(matvec, n, diag=None, cond_target=100.0,
                    rel_cutoff=None, tol_rel=1.0e-10, seed_extra=6,
                    add_per_round=6, max_rounds=30, max_size=None,
                    lambda_max=None, power_iters=12, rng_seed=20261109):
    """Orthonormal basis of the smallest-|lambda| directions, matrix-free.

    The deflation threshold is stated as a target condition number:
    every direction with ``|lambda| < lambda_max / cond_target`` is
    wanted.  That is the quantity that decides LSMR's cost, and it is
    also what removes the near-null directions that make the dense
    pseudo-inverse and the matrix-free solve disagree.

    The subspace is grown greedily.  The seed is the block of
    coordinate directions carrying the smallest diagonal entries; the
    diagonal is the cheapest matrix-free indicator of the soft
    directions and the Hessian route already computes it.  Each round
    solves the projected eigenproblem inside the current coordinate
    set, then adds the coordinates that carry the largest residual
    ``A v - theta v`` outside the set.  A round is certified when every
    wanted Ritz pair has residual below ``tol_rel * lambda_max`` and at
    least three vectors of headroom separate the wanted pairs from the
    block edge; an uncertified basis is reported as such and the caller
    must not deflate with it.

    Parameters
    ----------
    matvec : callable
        Symmetric operator product.
    n : int
        Dimension.
    diag : ndarray or None
        Diagonal of the operator.  ``None`` computes it, at ``n``
        products.
    cond_target : float
        Deflate until the remaining spectrum has condition number about
        ``cond_target``.  The default 100 leaves LSMR a well-conditioned
        complement.
    rel_cutoff : float or None
        Threshold relative to ``lambda_max``.  ``None`` derives it as
        ``1 / cond_target``.
    tol_rel : float
        Certification threshold on ``||A v - theta v|| / lambda_max``.
        Measured: a residual of 3e-4 relative left the complement as
        ill-conditioned as the original operator, so this is not a
        formality.
    seed_extra, add_per_round : int
        Seed block size is ``max(8, seed_extra + 2)``; each round adds
        at most ``add_per_round`` coordinates.
    max_size : int or None
        Cap on the coordinate-set size; ``None`` means the full space.
    lambda_max : float or None
        Largest |eigenvalue|; ``None`` estimates it by power iteration
        on the squared operator.
    power_iters : int
        Power iterations for the ``lambda_max`` estimate.

    Returns
    -------
    basis : ndarray, shape (n, k)
    ritz : ndarray, shape (k,)
    info : dict
        ``lambda_max``, ``rel_cutoff``, ``n_wanted``, ``block_size``,
        ``certified``, ``ritz_residual``, ``matvecs``, ``rounds``.
    """
    n = int(n)
    info = dict(lambda_max=0.0, rel_cutoff=0.0, n_wanted=0, block_size=0,
                certified=False, ritz_residual=np.inf, matvecs=0, rounds=0,
                diag_source='supplied')
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0), info

    calls = [0]

    def op(v):
        calls[0] += 1
        return np.asarray(matvec(v), dtype=float).reshape(-1)

    if lambda_max is None:
        rng = np.random.default_rng(rng_seed)
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(power_iters):
            w = op(op(v))
            nw = float(np.linalg.norm(w))
            if nw == 0.0:
                break
            v = w / nw
            lam = np.sqrt(nw)
        lambda_max = float(lam)
    info['lambda_max'] = float(lambda_max)
    if lambda_max == 0.0:
        return np.zeros((n, 0)), np.zeros(0), info

    if rel_cutoff is None:
        rel_cutoff = 1.0 / float(cond_target)
    info['rel_cutoff'] = float(rel_cutoff)
    cutoff = float(rel_cutoff) * lambda_max

    if diag is None:
        diag = np.array([op(np.eye(1, n, j).reshape(-1))[j]
                         for j in range(n)])
        info['diag_source'] = 'computed'
    diag = np.asarray(diag, dtype=float).reshape(-1)

    S = list(np.argsort(np.abs(diag))[:max(8, seed_extra + 2)])
    S = sorted(set(int(i) for i in S))
    if max_size is None:
        max_size = n
    max_size = int(min(max_size, n))

    basis = np.zeros((n, 0))
    ritz = np.zeros(0)
    for rnd in range(int(max_rounds)):
        B = _orthonormalize(np.eye(n)[:, S])
        W = np.column_stack([op(B[:, j]) for j in range(B.shape[1])])
        T = 0.5 * (B.T @ W + (B.T @ W).T)
        w, U = np.linalg.eigh(T)
        order = np.argsort(np.abs(w))
        w = w[order]
        U = U[:, order]
        Rmat = W @ U - (B @ U) * w
        resid = np.linalg.norm(Rmat, axis=0)
        n_wanted = int(np.sum(np.abs(w) < cutoff))
        wanted = max(n_wanted, 1)
        headroom = B.shape[1] - n_wanted
        certified = bool(n_wanted > 0
                         and resid[:n_wanted].max() <= tol_rel * lambda_max
                         and headroom >= 3)
        info.update(n_wanted=n_wanted, block_size=int(B.shape[1]),
                    certified=certified,
                    ritz_residual=float(resid[:wanted].max()),
                    rounds=rnd + 1)
        basis = B @ U[:, :wanted]
        ritz = w[:wanted]
        if certified:
            break
        if B.shape[1] >= max_size or B.shape[1] >= n:
            break
        r = np.linalg.norm(Rmat[:, :wanted], axis=1)
        r[S] = 0.0
        room = max(1, min(add_per_round, max_size - B.shape[1]))
        new = [int(i) for i in np.argsort(-r)[:room] if r[i] > 0.0]
        if not new:
            break
        S = sorted(set(S) | set(new))

    info['matvecs'] = calls[0]
    if basis.shape[1] == 0:
        return np.zeros((n, 0)), np.zeros(0), info
    return basis, ritz, info


def _block_pinv_action(B, lam, rcond, lam_scale):
    """Return ``v -> B pinv(B^T A B, rcond) B^T v`` by the eigenvalue rule."""
    keep = np.abs(lam) >= rcond * lam_scale
    vals = lam[keep]
    Uk = B[:, keep]

    def apply(v):
        if vals.size == 0:
            return np.zeros_like(v)
        return Uk @ ((Uk.T @ v) / vals)

    return apply, int(keep.sum())


def deflated_solve(matvec, b, basis=None, ritz=None, rcond=1e-9,
                   rtol=1e-12, atol=1e-12, refine=3, maxiter=None,
                   conlim=1e15, time_limit=None, lambda_max=None,
                   stall_iters=40, fallback=True):
    """Solve ``A x = b`` with the pseudo-inverse action, small block deflated.

    The deflated block is inverted from its Rayleigh quotients with the
    same relative cutoff ``numpy.linalg.pinv`` uses, so a direction
    below ``rcond * lambda_max`` contributes nothing -- exactly the
    dense route's rule.  The complement goes to :func:`lsmr_solve` on
    the projected operator, which is the part that has a chance to be
    well conditioned.

    ``refine`` rounds of residual correction follow.  One round would be
    exact if the basis were an exact invariant subspace; with an
    approximate one, each round removes what the previous round left,
    and the loop stops as soon as the projected residual -- the part the
    solve can actually reduce -- is below ``rtol * ||b||``.  The total
    residual is reported separately: on a singular operator it cannot
    fall below the null-space component of ``b``, which the
    pseudo-inverse deliberately discards.

    ``fallback`` re-solves from scratch with plain LSMR when the
    deflated route has not converged; the answer is then the
    unaccelerated one rather than a wrong one.

    Returns
    -------
    x : ndarray
    info : dict
        ``rounds``, per-round LSMR diagnostics (``solves``), total
        ``iterations``, ``relative_residual`` (projected),
        ``total_residual``, ``deflated``, ``dropped``, ``reason``.
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.size
    normb = float(np.linalg.norm(b))
    info = dict(rounds=0, iterations=0, relative_residual=1.0,
                total_residual=1.0, solves=[], deflated=0, dropped=0,
                reason='lsmr_only', basis=None)
    if normb == 0.0:
        info.update(relative_residual=0.0, total_residual=0.0)
        return np.zeros(n), info
    if basis is None or np.asarray(basis).size == 0:
        x, sub = lsmr_solve(matvec, b, rtol=rtol, atol=atol,
                            maxiter=maxiter, conlim=conlim,
                            time_limit=time_limit, stall_iters=stall_iters)
        info.update(rounds=1, iterations=sub['itn'],
                    relative_residual=sub['normr'] / normb,
                    total_residual=sub['normr'] / normb,
                    solves=[sub], reason='lsmr_only')
        return x, info

    B = np.asarray(basis, dtype=float)
    lam = np.asarray(ritz, dtype=float).reshape(-1)
    lam_scale = float(np.abs(lam).max()) if lam.size else 0.0
    if lambda_max is not None:
        lam_scale = max(lam_scale, float(lambda_max))
    block_apply, n_kept = _block_pinv_action(B, lam, rcond, lam_scale)
    info['deflated'] = int(B.shape[1])
    info['dropped'] = int(B.shape[1] - n_kept)

    def project(v):
        return v - B @ (B.T @ v)

    op = projected_operator(matvec, B)
    x = block_apply(b)

    def residuals(v):
        r = b - np.asarray(matvec(v), dtype=float).reshape(-1)
        return r, project(r)

    r, r_perp = residuals(x)
    rel = float(np.linalg.norm(r_perp) / normb)
    total_iters = 0
    rounds = 0
    for _ in range(max(1, int(refine))):
        if rel <= rtol:
            break
        rounds += 1
        if np.linalg.norm(r_perp) > 0.0:
            dw, sub = lsmr_solve(op, r_perp, rtol=min(rtol, 1e-12),
                                 atol=atol, maxiter=maxiter, conlim=conlim,
                                 time_limit=time_limit,
                                 stall_iters=stall_iters)
            total_iters += sub['itn']
            info['solves'].append(sub)
        else:
            dw = np.zeros(n)
            info['solves'].append(dict(itn=0, istop=0, normr=0.0,
                                       reason='empty_complement'))
        x = x + block_apply(r) + dw
        r, r_perp = residuals(x)
        rel = float(np.linalg.norm(r_perp) / normb)

    total = float(np.linalg.norm(r) / normb)
    if rel > rtol and fallback:
        # The deflated route did not converge.  Return the
        # unaccelerated answer rather than an unconverged accelerated
        # one: whatever error plain LSMR has is the status quo, and an
        # unconverged deflated solution has no such guarantee.
        x_fb, sub = lsmr_solve(matvec, b, rtol=rtol, atol=atol,
                               maxiter=maxiter, conlim=conlim,
                               time_limit=time_limit,
                               stall_iters=stall_iters)
        r = b - np.asarray(matvec(x_fb), dtype=float).reshape(-1)
        rel = float(np.linalg.norm(project(r)) / normb)
        total = float(np.linalg.norm(r) / normb)
        info['solves'].append(sub)
        total_iters += sub['itn']
        x = x_fb
        info['reason'] = 'fallback_lsmr'
    else:
        info['reason'] = 'converged' if rel <= rtol else 'refine_exhausted'

    last = info['solves'][-1] if info['solves'] else {}
    info.update(rounds=rounds, iterations=total_iters,
                relative_residual=rel, total_residual=total,
                itn=int(total_iters),
                istop=int(last.get('istop', 0)))
    return x, info
