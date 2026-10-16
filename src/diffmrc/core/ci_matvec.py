import numpy as np
from itertools import combinations

import jax

jax.config.update("jax_enable_x64", True)
from jax import lax
from jax import numpy as jnp

from .iterative_ci import davidson_lowest_k, minres_solve

__all__ = ['block_determinant_count', 'string_tables', 'ci_block_operator',
           'ci_block_matvec', 'lowest_k_eigpairs_mf',
           'iterative_ci_energies_mf']

_CLUSTER_TOL = 1e-8
_DIR_FLOOR = 1e-10
_MINRES_FLOOR = 1e-300
_STUCK_RES = 1e-6
_STUCK_GROWTH = 10.0
_DISCOVERY_ROUNDS = 4
# Pattern-chunk size of the accumulation; bounds the largest temporary of a
# scatter to chunk * n_src * n_strings entries.
_CHUNK = 8
# Fixed width of the batched block apply: the whole chain then compiles a
# single shape, so a Davidson run pays one compilation however its subspace
# grows. All blocks are a handful of columns wide.
_BLOCK_WIDTH = 8


def block_determinant_count(nact, n_alpha, n_beta):
    from math import comb
    return comb(nact, n_alpha) * comb(nact, n_beta)


def _strings(nact, n_occ):
    """Sorted n_occ-subset bitmasks of nact orbitals, with orbital positions.

    ``pos[j, k]`` counts the occupied orbitals below j in string k, i.e. the
    position j occupies in that string's ordered list -- the quantity the
    Condon-Slater signs are built from in ``core.casci._block_H`` (its
    ``pos`` array), restricted here to one spin block.
    """
    masks = sorted(sum(1 << i for i in c)
                   for c in combinations(range(nact), n_occ))
    masks = np.asarray(masks, dtype=np.int64)
    pos = np.zeros((nact, masks.size), dtype=np.int64)
    for j in range(nact):
        for k, m in enumerate(masks):
            pos[j, k] = bin(int(m) & ((1 << j) - 1)).count('1')
    return masks, pos


def _excitation_table(nact, n_occ, masks, index, pos, order):
    """Excitation patterns of one spin block (1 = singles, 2 = doubles).

    Each pattern is a (removed, added) pair of orbital sets of size
    ``order``; its sources are the strings carrying every removed orbital
    and none of the added ones. That count is combinatorial and therefore
    identical for every pattern, which keeps all arrays statically shaped.
    The sign mirrors ``core.casci._block_H``'s position sum inside this
    block: ``(-1) ** (sum pos[removed, source] + sum pos[added, target])``.
    """
    if n_occ < order or nact - order < n_occ:
        patterns = []
    elif order == 1:
        patterns = [(i, a) for i in range(nact) for a in range(nact)
                    if i != a]
    else:
        patterns = [(i, j, a, b)
                    for i in range(nact) for j in range(i + 1, nact)
                    for a in range(nact) if a not in (i, j)
                    for b in range(a + 1, nact) if b not in (i, j)]
    if not patterns:
        return (jnp.zeros((0, order), dtype=jnp.int64),
                jnp.zeros((0, order), dtype=jnp.int64),
                jnp.zeros((0, 0), dtype=jnp.int64),
                jnp.zeros((0, 0), dtype=jnp.int64),
                jnp.zeros((0, 0)))
    n_src = len(list(combinations(range(nact - 2 * order), n_occ - order)))
    rem = np.array([p[:order] for p in patterns], dtype=np.int64)
    add = np.array([p[order:] for p in patterns], dtype=np.int64)
    src = np.zeros((len(patterns), n_src), dtype=np.int64)
    tgt = np.zeros((len(patterns), n_src), dtype=np.int64)
    sign = np.zeros((len(patterns), n_src))
    for p, pat in enumerate(patterns):
        keep = [k for k, m in enumerate(masks)
                if all((int(m) >> pat[c]) & 1 for c in range(order))
                and not any((int(m) >> pat[order + c]) & 1
                            for c in range(order))]
        assert len(keep) == n_src
        src[p] = keep
        for s, k in enumerate(keep):
            m = int(masks[k])
            for c in range(order):
                m ^= (1 << pat[c]) | (1 << pat[order + c])
            j = index[m]
            tgt[p, s] = j
            parity = 0
            for c in range(order):
                parity += int(pos[pat[c], k]) + int(pos[pat[order + c], j])
            sign[p, s] = 1.0 if parity % 2 == 0 else -1.0
    # Pad the pattern axis to a whole number of chunks: the accumulation
    # loops slice fixed-size chunks inside a traced loop, and the zero-sign
    # filler rows contribute nothing (their values are multiplied by 0).
    pad = (-len(patterns)) % _CHUNK
    if pad:
        rem = np.vstack([rem, np.zeros((pad, order), dtype=np.int64)])
        add = np.vstack([add, np.zeros((pad, order), dtype=np.int64)])
        src = np.vstack([src, np.zeros((pad, n_src), dtype=np.int64)])
        tgt = np.vstack([tgt, np.zeros((pad, n_src), dtype=np.int64)])
        sign = np.vstack([sign, np.zeros((pad, n_src))])
    return (jnp.asarray(rem), jnp.asarray(add), jnp.asarray(src),
            jnp.asarray(tgt), jnp.asarray(sign))


def string_tables(nact, n_alpha, n_beta):
    """Static pattern tables of an S_z block (numpy build, jnp payload).

    ``phase`` converts between the two determinant-sign conventions: the
    kernel works in the alpha-first order, where the sign of an excitation
    is local to its spin block, while ``core.casci`` orders the spin
    orbitals by the interleaved global index. The matrices are related by
    ``H_global = D H_alpha-first D`` with ``D = diag(phase)`` and
    ``phase(d) = (-1) ** #{(a in I, b in J): a > b}``; both sides only ever
    see the operator, so the convention is applied by two sign vectors.
    """
    masks_a, pos_a = _strings(nact, n_alpha)
    masks_b, pos_b = _strings(nact, n_beta)
    index_a = {int(m): i for i, m in enumerate(masks_a)}
    index_b = {int(m): i for i, m in enumerate(masks_b)}
    tab = {
        'nI': int(masks_a.size), 'nJ': int(masks_b.size), 'nact': int(nact),
        'n_alpha': int(n_alpha), 'n_beta': int(n_beta),
        'masks_a': jnp.asarray(masks_a), 'masks_b': jnp.asarray(masks_b),
        'a1': _excitation_table(nact, n_alpha, masks_a, index_a, pos_a, 1),
        'a2': _excitation_table(nact, n_alpha, masks_a, index_a, pos_a, 2),
        'b1': _excitation_table(nact, n_beta, masks_b, index_b, pos_b, 1),
        'b2': _excitation_table(nact, n_beta, masks_b, index_b, pos_b, 2),
    }
    bits = np.arange(nact)
    tab['occ_a'] = jnp.asarray(
        ((masks_a[:, None] >> bits[None, :]) & 1).astype(np.float64))
    tab['occ_b'] = jnp.asarray(
        ((masks_b[:, None] >> bits[None, :]) & 1).astype(np.float64))
    lowcnt = np.zeros((nact, masks_b.size), dtype=np.int64)
    for a in range(nact):
        for k, m in enumerate(masks_b):
            lowcnt[a, k] = bin(int(m) & ((1 << a) - 1)).count('1')
    parity = (tab['occ_a'] @ jnp.asarray(lowcnt)) % 2
    tab['phase'] = jnp.where(parity == 0, 1.0, -1.0).reshape(-1)
    return tab


def _folded(h1e_mo, eri_mo, ncore, nact):
    """Active-space integrals with the closed-shell core folded in.

    Mirrors ``core.casci._block_H``'s frozen core: it contributes 2 J - K to
    the one-electron part and a constant to the diagonal. Chemist-notation
    MO integrals throughout.
    """
    h = h1e_mo[ncore:ncore + nact, ncore:ncore + nact]
    g = eri_mo[ncore:ncore + nact, ncore:ncore + nact,
               ncore:ncore + nact, ncore:ncore + nact]
    if ncore == 0:
        return h, g, jnp.zeros(())
    h_eff = (h
             + 2.0 * jnp.einsum('pqcc->pq',
                                eri_mo[ncore:ncore + nact,
                                       ncore:ncore + nact, :ncore, :ncore])
             - jnp.einsum('pccq->pq',
                          eri_mo[ncore:ncore + nact, :ncore, :ncore,
                                 ncore:ncore + nact]))
    core_block = eri_mo[:ncore, :ncore, :ncore, :ncore]
    # Closed-shell core constant: two electrons per core orbital, and the
    # two-electron sum runs over ordered spatial pairs of 2J - K so that
    # the same-orbital Coulomb (cc|cc) is kept exactly once.
    e_core = (2.0 * jnp.trace(h1e_mo[:ncore, :ncore])
              + 2.0 * jnp.einsum('ccdd->', core_block)
              - jnp.einsum('cddc->', core_block))
    return h_eff, g, e_core


def _diagonal(h1e_mo, eri_mo, tab, ncore, nact):
    """Diagonal of the block Hamiltonian, in the block's determinant order."""
    h_eff, g, e_core = _folded(h1e_mo, eri_mo, ncore, nact)
    occ_a, occ_b = tab['occ_a'], tab['occ_b']
    wd = jnp.einsum('iijj->ij', g) - jnp.einsum('ijji->ij', g)
    da = (occ_a @ jnp.diag(h_eff)
          + (occ_a @ jnp.triu(wd, 1) * occ_a).sum(axis=1))
    db = (occ_b @ jnp.diag(h_eff)
          + (occ_b @ jnp.triu(wd, 1) * occ_b).sum(axis=1))
    coul = occ_a @ jnp.einsum('iijj->ij', g) @ occ_b.T
    return (da[:, None] + db[None, :] + coul + e_core).reshape(-1)


def _acc_alpha(sig, src, tgt, sign, cpl, x2):
    """``sig[tgt, :] += sign * cpl * x2[src, :]`` over alpha-string sources.

    ``cpl`` broadcasts against the (n_pat, n_src, n_J) shape: a per-pattern
    scalar is passed as (n_pat, 1, 1), a per-(pattern, source) value as
    (n_pat, n_src, 1), a full field as (n_pat, n_src, n_J). The pattern
    axis is walked by a traced loop over fixed chunks (tables are padded to
    a whole number of chunks at construction), so the compiled graph holds
    one scatter per excitation class however many patterns there are; the
    zero-sign filler rows contribute nothing.
    """
    n_pat = src.shape[0]
    if n_pat == 0:
        return sig
    n_chunks = n_pat // _CHUNK
    nJ = sig.shape[1]

    def body(i, s):
        c0 = i * _CHUNK
        src_c = lax.dynamic_slice_in_dim(src, c0, _CHUNK, axis=0)
        tgt_c = lax.dynamic_slice_in_dim(tgt, c0, _CHUNK, axis=0)
        sign_c = lax.dynamic_slice_in_dim(sign, c0, _CHUNK, axis=0)
        cpl_c = lax.dynamic_slice_in_dim(cpl, c0, _CHUNK, axis=0)
        val = sign_c[:, :, None] * cpl_c * x2[src_c, :]
        return s.at[tgt_c[:, :, None],
                    jnp.arange(nJ)[None, None, :]].add(val)

    return lax.fori_loop(0, n_chunks, body, sig)


def _acc_beta(sig, src, tgt, sign, cpl, x2):
    """``sig[:, tgt] += sign * cpl * x2[:, src]`` over beta-string sources."""
    n_pat = src.shape[0]
    if n_pat == 0:
        return sig
    n_chunks = n_pat // _CHUNK
    nI = sig.shape[0]

    def body(i, s):
        c0 = i * _CHUNK
        src_c = lax.dynamic_slice_in_dim(src, c0, _CHUNK, axis=0)
        tgt_c = lax.dynamic_slice_in_dim(tgt, c0, _CHUNK, axis=0)
        sign_c = lax.dynamic_slice_in_dim(sign, c0, _CHUNK, axis=0)
        cpl_c = lax.dynamic_slice_in_dim(cpl, c0, _CHUNK, axis=0)
        val = (sign_c[:, :, None] * cpl_c
               * x2[:, src_c].transpose(1, 2, 0))
        return s.at[jnp.arange(nI)[None, None, :],
                    tgt_c[:, :, None]].add(val)

    return lax.fori_loop(0, n_chunks, body, sig)


def _acc_alpha_beta(sig, src_a, tgt_a, sign_a, src_b, tgt_b, sign_b, cval, x2):
    """The opposite-spin double class: one alpha single times one beta single.

    The (alpha-chunk, beta-chunk) pairs are flattened into one traced loop;
    ``cval`` is the (n_pat_a, n_pat_b) integral field of the pattern pairs.
    """
    n_pa, n_pb = src_a.shape[0], src_b.shape[0]
    if n_pa == 0 or n_pb == 0:
        return sig
    n_ca, n_cb = n_pa // _CHUNK, n_pb // _CHUNK

    def body(idx, s):
        ia, ib = idx // n_cb, idx % n_cb
        ca0, cb0 = ia * _CHUNK, ib * _CHUNK
        src_a_c = lax.dynamic_slice_in_dim(src_a, ca0, _CHUNK, axis=0)
        tgt_a_c = lax.dynamic_slice_in_dim(tgt_a, ca0, _CHUNK, axis=0)
        sign_a_c = lax.dynamic_slice_in_dim(sign_a, ca0, _CHUNK, axis=0)
        src_b_c = lax.dynamic_slice_in_dim(src_b, cb0, _CHUNK, axis=0)
        tgt_b_c = lax.dynamic_slice_in_dim(tgt_b, cb0, _CHUNK, axis=0)
        sign_b_c = lax.dynamic_slice_in_dim(sign_b, cb0, _CHUNK, axis=0)
        cval_a = lax.dynamic_slice_in_dim(cval, ca0, _CHUNK, axis=0)
        cval_c = lax.dynamic_slice_in_dim(cval_a, cb0, _CHUNK, axis=1)
        xa = x2[src_a_c, :]
        val = (sign_a_c[:, :, None, None]
               * sign_b_c[None, None, :, :]
               * cval_c[:, None, :, None]
               * xa[:, :, src_b_c])
        return s.at[tgt_a_c[:, :, None, None],
                    tgt_b_c[None, None, :, :]].add(val)

    return lax.fori_loop(0, n_ca * n_cb, body, sig)


def _apply(h1e_mo, eri_mo, tab, ncore, nact, x):
    """One matrix-free application of the block Hamiltonian (alpha-first).

    The terms mirror ``core.casci._block_H``: diagonal; alpha and beta
    singles (one-electron part, same-spin exchange sums, opposite-spin
    spectator Coulomb); same-spin doubles; and the opposite-spin double
    class, whose exchange partner vanishes by spin, leaving the single
    integral (ia|jb) per pattern pair.
    """
    h_eff, g, e_core = _folded(h1e_mo, eri_mo, ncore, nact)
    occ_a, occ_b = tab['occ_a'], tab['occ_b']
    nI, nJ = tab['nI'], tab['nJ']
    orb = jnp.arange(int(tab['nact']))
    x2 = x.reshape(nI, nJ)
    sig = _diagonal(h1e_mo, eri_mo, tab, ncore, nact).reshape(nI, nJ) * x2

    # alpha singles: (h + same-spin exchange) on the alpha side, the beta
    # spectator Coulomb on the beta side. The j = i self-term of the
    # Slater-Condon sum vanishes identically ((ia|ii) = (ii|ia)), so the
    # exclusion of core.casci._block_H drops out of the algebra.
    rem, add, src, tgt, sign = tab['a1']
    i, a = rem[:, 0], add[:, 0]
    w1 = (g[i[:, None], a[:, None], orb[None, :], orb[None, :]]
          - g[i[:, None], orb[None, :], orb[None, :], a[:, None]])
    same = (occ_a @ w1.T)[src, jnp.arange(i.size)[:, None]] + h_eff[i, a][:, None]
    spec = (occ_b @ g[i[:, None], a[:, None], orb[None, :],
                      orb[None, :]].T).T
    sig = _acc_alpha(sig, src, tgt, sign, same[:, :, None] + spec[:, None, :],
                     x2)

    # beta singles: the mirror image
    rem, add, src, tgt, sign = tab['b1']
    j, b = rem[:, 0], add[:, 0]
    w2 = (g[j[:, None], b[:, None], orb[None, :], orb[None, :]]
          - g[j[:, None], orb[None, :], orb[None, :], b[:, None]])
    same = (occ_b @ w2.T)[src, jnp.arange(j.size)[:, None]] + h_eff[j, b][:, None]
    spec = (occ_a @ g[j[:, None], b[:, None], orb[None, :],
                      orb[None, :]].T).T
    sig = _acc_beta(sig, src, tgt, sign, same[:, :, None] + spec[:, None, :],
                    x2)

    # same-spin doubles
    rem, add, src, tgt, sign = tab['a2']
    cval = (g[rem[:, 0], add[:, 0], rem[:, 1], add[:, 1]]
            - g[rem[:, 0], add[:, 1], rem[:, 1], add[:, 0]])[:, None, None]
    sig = _acc_alpha(sig, src, tgt, sign, cval, x2)

    rem, add, src, tgt, sign = tab['b2']
    cval = (g[rem[:, 0], add[:, 0], rem[:, 1], add[:, 1]]
            - g[rem[:, 0], add[:, 1], rem[:, 1], add[:, 0]])[:, None, None]
    sig = _acc_beta(sig, src, tgt, sign, cval, x2)

    # opposite-spin doubles
    tab_a1, tab_b1 = tab['a1'], tab['b1']
    cval = g[tab_a1[0][:, 0][:, None], tab_a1[1][:, 0][:, None],
             tab_b1[0][:, 0][None, :], tab_b1[1][:, 0][None, :]]
    sig = _acc_alpha_beta(sig, tab_a1[2], tab_a1[3], tab_a1[4],
                          tab_b1[2], tab_b1[3], tab_b1[4], cval, x2)
    return sig.reshape(-1)


def ci_block_operator(h1e_mo, eri_mo, ncore, nact, n_alpha, n_beta, tab=None):
    """Matrix-free operator of the (n_alpha, n_beta) S_z block.

    Returns ``(matvec, diag)``; ``matvec`` applies the block Hamiltonian
    without ever forming it, in the same determinant sign convention as
    ``core.casci``. The determinant order is alpha-string-major,
    beta-string-minor. Working memory is O(n_det) per vector plus the
    static pattern tables; one application costs one pass over the block's
    excitation connections.
    """
    if tab is None:
        tab = string_tables(nact, n_alpha, n_beta)
    phase = tab['phase']

    def mv1(x):
        return phase * _apply(h1e_mo, eri_mo, tab, ncore, nact, phase * x)

    def matvec(x):
        if x.ndim == 1:
            return mv1(x)
        # The block applies each column independently (the scatter of one
        # column never touches another), so batching them with a vmap is
        # bit-identical to a sequential loop while running several times
        # faster. Every chunk is zero-padded to a fixed width so that the
        # whole chain sees exactly one shape: with a width that followed
        # the Davidson's growing subspace, each new width re-triggered a
        # full recompilation, and at n_det ~ 8.5e5 those compilations
        # dominated the wall clock. Padding is exact (the operator is
        # linear, a zero column maps to zero), so values are unchanged.
        n, m = x.shape
        outs = []
        for c0 in range(0, m, _BLOCK_WIDTH):
            c1 = min(c0 + _BLOCK_WIDTH, m)
            block = jnp.zeros((n, _BLOCK_WIDTH), dtype=x.dtype)
            block = block.at[:, :c1 - c0].set(x[:, c0:c1])
            y = jax.vmap(mv1, in_axes=1, out_axes=1)(block)
            outs.append(y[:, :c1 - c0])
        return jnp.concatenate(outs, axis=1)

    return matvec, _diagonal(h1e_mo, eri_mo, tab, ncore, nact)


def ci_block_matvec(h1e_mo, eri_mo, ncore, nact, n_alpha, n_beta, tab=None):
    return ci_block_operator(h1e_mo, eri_mo, ncore, nact, n_alpha, n_beta,
                             tab)[0]


def _mf_pullback_solve(matvec, lam, cluster_cols, u_bar, ridge, scale):
    """The cluster-complement solve of one pair, operator given as a matvec.

    Same rule as ``core.iterative_ci._pullback_solve``, with the cluster
    projector applied through its columns. The discovered-partner slots are
    preallocated (``_DISCOVERY_ROUNDS`` zero columns), so the whole loop is
    shape-static and trace-safe.
    """
    n, n_cluster = cluster_cols.shape
    n_slots = n_cluster + _DISCOVERY_ROUNDS

    def make_proj(cols_bytes):
        def proj(y):
            return cols_bytes @ (cols_bytes.T @ y)
        return proj

    def op(cols_bytes, eps):
        def mv(y):
            p = y - cols_bytes @ (cols_bytes.T @ y)
            q = matvec(p) - lam * p
            return (cols_bytes @ (cols_bytes.T @ y)) \
                + (q - cols_bytes @ (cols_bytes.T @ q)) + eps * p
        return mv

    def unhealthy(x, res, rhs_norm):
        growth = (jnp.linalg.norm(x) * (_CLUSTER_TOL * scale)
                  / jnp.maximum(rhs_norm, _MINRES_FLOOR))
        return (res > _STUCK_RES) | (growth > _STUCK_GROWTH)

    cols0 = jnp.concatenate([cluster_cols,
                             jnp.zeros((n, _DISCOVERY_ROUNDS))], axis=1)
    rhs0 = u_bar - cols0 @ (cols0.T @ u_bar)
    x0, res0, _ = minres_solve(op(cols0, 0.0), rhs0)
    carry = (cols0, x0, rhs0, res0, jnp.int32(0))

    def cond(c):
        cols_c, x_c, rhs_c, res_c, rounds = c
        v = x_c / jnp.maximum(jnp.linalg.norm(x_c), _MINRES_FLOOR)
        ray = jnp.abs(v @ matvec(v) - lam * (v @ v))
        w = v - cols_c @ (cols_c.T @ v)
        return ((rounds < _DISCOVERY_ROUNDS)
                & unhealthy(x_c, res_c, jnp.linalg.norm(rhs_c))
                & (ray < _CLUSTER_TOL * scale)
                & (jnp.linalg.norm(w) > _DIR_FLOOR))

    def body(c):
        cols_c, x_c, rhs_c, res_c, rounds = c
        v = x_c / jnp.maximum(jnp.linalg.norm(x_c), _MINRES_FLOOR)
        w = v - cols_c @ (cols_c.T @ v)
        w = w / jnp.maximum(jnp.linalg.norm(w), _DIR_FLOOR)
        cols_c = cols_c.at[:, n_cluster + rounds].set(w)
        rhs_c = u_bar - cols_c @ (cols_c.T @ u_bar)
        x_c, res_c, _ = minres_solve(op(cols_c, 0.0), rhs_c)
        return (cols_c, x_c, rhs_c, res_c, rounds + 1)

    cols, x, rhs, res, _ = lax.while_loop(cond, body, carry)
    x = lax.cond(unhealthy(x, res, jnp.linalg.norm(rhs)),
                 lambda _: minres_solve(op(cols, ridge * scale), rhs)[0],
                 lambda _: x, None)
    return x


def _mf_forward(h1e_mo, eri_mo, tab, ncore, nact, k, tol, max_cycles,
                n_expand):
    matvec, diag = ci_block_operator(h1e_mo, eri_mo, ncore, nact,
                                     tab['n_alpha'], tab['n_beta'], tab)
    n = tab['nI'] * tab['nJ']
    return davidson_lowest_k(matvec, n, k=k, tol=tol, max_cycles=max_cycles,
                             n_expand=n_expand, diag=diag)


def lowest_k_eigpairs_mf(h1e_mo, eri_mo, ncore=0, nact=2, n_alpha=1,
                         n_beta=1, k=1, tol=1e-10, max_cycles=60,
                         n_expand=3, ridge=1e-8):
    """The matrix-free differentiable lowest-k solver of one S_z block.

    Forward: Davidson on the matrix-free operator
    (:func:`ci_block_operator`) under ``stop_gradient``. Reverse: the
    implicit rule of ``core.iterative_ci`` with every operator access a
    matvec; the matrix cotangent is never formed, because the VJP of one
    pair with respect to the integrals is the gradient of the scalar
    ``xi @ H(h, e) @ u`` through the same matrix-free application. Working
    memory is O(n_det) per vector on both sides, plus the saved eigenpairs
    and Ritz block; the O(n_det^2) object of the dense route never exists.

    The discovery thresholds use ``max(1, max|diag|)`` as the operator's
    scale, the diagonal being the only O(n_det) spectral information the
    matrix-free route holds without extra matvecs.
    """
    tab = string_tables(nact, n_alpha, n_beta)
    phase = tab['phase']

    def apply_global(hh, ee, y):
        return phase * _apply(hh, ee, tab, ncore, nact, phase * y)

    def operator_scale(h, e):
        return jnp.maximum(1.0, jnp.max(jnp.abs(
            _diagonal(h, e, tab, ncore, nact))))

    def primal(h, e):
        ws, vs, _, _, _, _ = _mf_forward(h, e, tab, ncore, nact, k, tol,
                                         max_cycles, n_expand)
        return lax.stop_gradient(ws), lax.stop_gradient(vs)

    def fwd(h, e):
        ws, vs, res, cyc, spec, block = _mf_forward(
            h, e, tab, ncore, nact, k, tol, max_cycles, n_expand)
        return ((lax.stop_gradient(ws), lax.stop_gradient(vs)),
                (h, e, ws, vs, spec, block, operator_scale(h, e)))

    def bwd(res, ct):
        h, e, ws, vs, spec, block, scale = res
        ct_w, ct_v = ct
        matvec, _ = ci_block_operator(h, e, ncore, nact, n_alpha, n_beta,
                                      tab)
        h_bar = jnp.zeros_like(h)
        e_bar = jnp.zeros_like(e)
        for p in range(ws.shape[0]):
            lam = ws[p]
            u = vs[:, p]
            close = jnp.abs(spec - lam) <= _CLUSTER_TOL * scale
            cols = block * close[None, :]
            x = _mf_pullback_solve(matvec, lam, cols, ct_v[:, p], ridge,
                                   scale)
            x = x - cols @ (cols.T @ x)
            xi = x - ct_w[p] * u
            g_h, g_e = jax.grad(
                lambda hh, ee: xi @ apply_global(hh, ee, u),
                argnums=(0, 1))(h, e)
            h_bar = h_bar - g_h
            e_bar = e_bar - g_e
        return h_bar, e_bar

    solve = jax.custom_vjp(primal)
    solve.defvjp(fwd, bwd)
    return solve(h1e_mo, eri_mo)


def iterative_ci_energies_mf(h1e_mo, eri_mo, ncore=0, nact=2, nele_act=2,
                             k=1, **kwargs):
    """The k lowest energies of the closed-shell S_z block, differentiable.

    The block is ``(nele_act // 2, nele_act // 2)`` in the active space and
    the energies exclude the nuclear repulsion, exactly like
    ``core.casci.cs_casci``.
    """
    n_a = nele_act // 2
    ws, _ = lowest_k_eigpairs_mf(h1e_mo, eri_mo, ncore=ncore, nact=nact,
                                 n_alpha=n_a, n_beta=nele_act - n_a, k=k,
                                 **kwargs)
    return ws
