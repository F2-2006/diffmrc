"""Second derivative of the matrix-free CI eigenvalue.

WHY THIS MODULE EXISTS
    ``core.ci_matvec.lowest_k_eigpairs_mf`` carries a first-order implicit
    reverse: a ``custom_vjp`` whose VJP is the Sternheimer solve of the
    converged eigenpair. The reverse pass contains a MINRES iteration
    (``lax.while_loop`` with a data-dependent bound), which has no
    transposition rule, so reverse-over-reverse (``jax.grad`` of
    ``jax.grad``, ``jax.vjp`` of the gradient) raises
    "Reverse-mode differentiation does not work for lax.while_loop".
    Forward-over-reverse (``jax.hessian`` = ``jacfwd(jacrev)``) does run,
    but it is not a usable route. It agrees with this module on
    N2/6-31G CAS(2,2) (1.8e-14) and yet returns a wrong, asymmetric
    matrix on a well-conditioned synthetic block (ground gap 0.062; the
    diagonal entry -28.66 is reproduced as +4461.97): the tangent it
    propagates through the MINRES while_loop is a different recurrence
    from the solve itself and can diverge, silently. Symmetry of the
    returned matrix is the cheap diagnostic, and it fails there. The
    route also broadcasts every intermediate over the parameter count
    (the cost recorded for the dense implicit Hessian in
    ``core.hessian``) and is unusable wherever the assembly needs the
    transposed (adjoint) solve, as
    ``core.hessian.ImplicitHessianHVPOps`` does.
    This module supplies the rule explicitly instead: one forward solve,
    one matrix-free application and one MINRES solve per direction,
    working memory O(n_det) throughout. It is the second-order complement
    of the first-order route that the README's "Active-space size" entry
    records as validated.

THE RULE
    Write the block Hamiltonian as ``H(p) = F(h(p), e(p))``, with ``F``
    the Condon-Slater assembly of ``core.ci_matvec._apply``. ``F`` is
    linear in the integrals ``(h, e)`` but the integrals are nonlinear in
    the parameters ``p`` (an orbital rotation acts quartically on the
    two-electron tensor), so the second derivative of the lowest
    eigenvalue splits into two parts:

        d2 lam / dp_i dp_j = u^T F(h_ij, e_ij) u
                             + 2 u^T F(h_i, e_i) (du/dp_j),

    where ``h_i = dh/dp_i``, ``h_ij = d2h/dp_i dp_j``, and ``du/dp_j``
    solves the Sternheimer equation on the orthocomplement of the
    degenerate cluster of ``lam``:

        P (H - lam) P (du/dp_j) = -P F(h_j, e_j) u,
        P = 1 - sum_{v in cluster} v v^T.

    The first term is the curvature of the integrals at frozen CI vector;
    the second is the CI response. Both are needed: on the N2/6-31G
    CAS(6,6) chain measured here the first term is +63.3 on the (0,0)
    entry while the response term is -3.1, and dropping either one is
    wrong by more than a factor of ten. The response term alone is
    negative semidefinite (the eigenvector is variational, so the
    eigenvalue is concave in any parameter at frozen integrals); the
    total has no fixed sign, because the integral-curvature term is
    positive on the same directions.

    The projector convention is the first-order one
    (``core.iterative_ci._ift_pullback``: the frozen-rotation rule of
    ``core.eigh``), including the cluster discovery of
    ``core.ci_matvec._mf_pullback_solve``.

COST
    One matrix-free application per direction for ``F(h_i, e_i) u`` and
    one MINRES solve per direction for ``du/dp_j``; the curvature term
    costs one application per direction pair on the second integral
    tangents, which the caller supplies. Nothing of size ``n_det**2`` is
    formed. The Davidson forward runs once. This module is the CI-level
    block of a second derivative, not a differentiable function of its
    own inputs: it is evaluated at concrete integrals, in the same way
    ``core.hessian.implicit_hessian_columnwise`` is.

WHAT IT DOES NOT DO
    The orbital-level response is not assembled here. The relaxed second
    derivative needs the CI response to the orbital parameters and to the
    geometry (``u_theta``, ``u_R``) inside the mixed and orbital-Hessian
    blocks; this module supplies the per-direction response solve and the
    two contractions those blocks are built from, and the caller supplies
    the integral tangents and curvatures.
"""
import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from .ci_matvec import (_CLUSTER_TOL, _apply, _mf_forward, _mf_pullback_solve,
                        ci_block_operator, string_tables)

__all__ = ['mf_eigenpair_hessian']


def mf_eigenpair_hessian(h1e_mo, eri_mo, ncore, nact, n_alpha, n_beta,
                         tangents, curvatures=None, tol=1e-10, max_cycles=60,
                         n_expand=3, ridge=1e-8):
    """Lowest eigenpair of one S_z block and its second derivative.

    Parameters
    ----------
    h1e_mo, eri_mo : array
        MO-space integrals of the full (core + active) window, in the
        layout ``core.ci_matvec`` expects (chemist notation).
    ncore, nact : int
        Core and active orbital counts.
    n_alpha, n_beta : int
        Active-space spin counts of the block (the block label of
        ``core.casci.cs_casci_blocked`` minus ``ncore`` on the alpha side).
    tangents : sequence of (array, array)
        First parameter tangents of the integrals: ``(dh_i, de_i)``.
        ``F`` is linear in the integrals, so one JVP per direction gives
        ``F(h_i, e_i) u`` exactly.
    curvatures : sequence of sequences of (array, array), optional
        Second parameter tangents ``(dh_ij, de_ij)``, indexed
        ``[i][j]``. Required whenever the integrals are nonlinear in the
        parameters (an orbital rotation, a nuclear displacement). Pass
        ``None`` only when they are linear, where the term is exactly
        zero.
    tol, max_cycles, n_expand :
        Davidson controls, as in :func:`core.ci_matvec.lowest_k_eigpairs_mf`.
    ridge : float
        Regularization of the response solve when the cluster discovery is
        exhausted (``core.ci_matvec._mf_pullback_solve``).

    Returns
    -------
    dict
        ``lam`` : the lowest eigenvalue of the block.
        ``u`` : its eigenvector, in the block layout of ``ci_matvec``.
        ``grad`` : ``d lam / dp_i`` (Hellmann-Feynman, one per tangent).
        ``hess`` : ``(n_dir, n_dir)`` matrix of ``d2 lam / dp_i dp_j``.
        ``hess_response`` : the response part alone (negative
            semidefinite for the ground state).
        ``hess_curvature`` : the integral-curvature part alone.
        ``residual``, ``cycles`` : convergence record of the forward run.
        ``cluster_size`` : number of Ritz vectors inside the cluster
            tolerance of ``lam`` (1 for a simple state).
        ``n_det`` : dimension of the block.
    """
    tangents = [(jnp.asarray(dh), jnp.asarray(de)) for dh, de in tangents]
    n_dir = len(tangents)
    if curvatures is None:
        curvatures = [[(jnp.zeros_like(h1e_mo), jnp.zeros_like(eri_mo))
                       for _ in range(n_dir)] for _ in range(n_dir)]
    with jax.ensure_compile_time_eval():
        tab = string_tables(nact, n_alpha, n_beta)
    phase = tab['phase']

    def apply_global(hh, ee, y):
        return phase * _apply(hh, ee, tab, ncore, nact, phase * y)

    ws, vs, res, cyc, spec, block = _mf_forward(
        h1e_mo, eri_mo, tab, ncore, nact, 1, tol, max_cycles, n_expand)
    lam = ws[0]
    u = vs[:, 0]

    matvec, diag = ci_block_operator(h1e_mo, eri_mo, ncore, nact, n_alpha,
                                     n_beta, tab)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(diag)))
    close = jnp.abs(spec - lam) <= _CLUSTER_TOL * scale
    cols = block * close[None, :]

    # First tangents: F(h_i, e_i) u at the frozen eigenvector.
    g = [jax.jvp(lambda hh, ee: apply_global(hh, ee, u),
                 (h1e_mo, eri_mo), (dh, de))[1] for dh, de in tangents]
    # Sternheimer solve of the same direction; the projector is applied
    # inside _mf_pullback_solve and once more here, matching the reverse pass.
    du = []
    for gj in g:
        x = _mf_pullback_solve(matvec, lam, cols, -gj, ridge, scale)
        du.append(x - cols @ (cols.T @ x))

    hess_resp = jnp.zeros((n_dir, n_dir))
    hess_curv = jnp.zeros((n_dir, n_dir))
    for i in range(n_dir):
        for j in range(n_dir):
            hess_resp = hess_resp.at[i, j].set(2.0 * (g[i] @ du[j]))
            dh2, de2 = curvatures[i][j]
            hess_curv = hess_curv.at[i, j].set(
                u @ apply_global(dh2, de2, u))
    hess = hess_curv + hess_resp
    asym = jnp.max(jnp.abs(hess - hess.T))
    hess = 0.5 * (hess + hess.T)
    hess_resp = 0.5 * (hess_resp + hess_resp.T)
    hess_curv = 0.5 * (hess_curv + hess_curv.T)

    return dict(lam=lam, u=u,
                grad=jnp.asarray([gi @ u for gi in g]),
                hess=hess, hess_response=hess_resp,
                hess_curvature=hess_curv, hess_asymmetry=asym,
                residual=res[0], cycles=cyc,
                cluster_size=int(jnp.sum(close)),
                n_det=int(tab['nI'] * tab['nJ']))
