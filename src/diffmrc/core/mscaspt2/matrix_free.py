"""Matrix-free coupled amplitude solve for the MS-CASPT2 union system.

PURPOSE
-------
The S13 eq. 1 reference-space coupling is projected on the union of the
contracted functions of all references, which gives the linear system

    A C - S C F = -W,
    A = V H0 V^T,  S = V V^T,

where ``V`` is the sparse function-by-determinant matrix and ``H0`` the
sparse one-body zeroth-order Hamiltonian in the determinant basis.  The
dense route in :mod:`diffmrc.core.mscaspt2.ms` assembles ``S`` and ``A``
as ``n_fn x n_fn`` matrices and diagonalizes ``S`` to regularize the
linear dependencies.  On N2 CAS(6,6)/6-31G ``n_fn = 24156``, so both
matrices together need 8.70 GiB and the overlap eigensolve does not fit
in an 8 GB job.

This module never forms either matrix.  The two operator actions are

    S c = V (V^T c),      A c = V (H0 (V^T c)),

and the shifted systems of the Sylvester solve are handed to
:func:`diffmrc.core.lsmr.lsmr_solve`, whose stopping rule is the relative
residual ``||r|| / ||b||`` with no iteration cap.  Memory is then
``nnz(V)`` plus ``nnz(H0)`` plus a handful of length-``nd`` vectors,
independent of ``n_fn**2``.

WHY THE MINIMUM-NORM SOLUTION IS THE PHYSICAL ONE
-------------------------------------------------
``S`` is singular: the union basis is linearly dependent, and every
vector in the null space of ``V^T`` is also annihilated by ``A``, since
``A c = V H0 (V^T c)``.  The right-hand side is ``W = V W_d`` and is
therefore orthogonal to that null space.  LSMR started from zero
generates its Krylov space from the right-hand side, so the iterates
stay in ``range(V)`` and the solve returns the minimum-norm solution --
the same object the dense route returns when the discarded overlap
eigenvalues are exact zeros.  For the solutions ``C`` and ``C + N`` with
``V^T N = 0`` the two physical outputs agree,

    cross   = C^T W    (N^T W = (V^T N)^T W_d = 0),
    overlap = C^T S C  (S N = 0),

so the effective Hamiltonian does not depend on which representative of
the solution set the solver lands on.

The near-null directions that the dense route drops at
``rtol * max(eig(S), 1)`` are not removed explicitly here.  On the
validated H2O system every dropped eigenvalue is at round-off
(``|lambda| <= 3e-14`` against ``lambda_max = 18``), so the regularized
and the unregularized problems have the same solution; a
``drop_null_rows`` switch removes identically zero function vectors
before the solve, which is exact and shrinks the Krylov space.

PRECONDITIONING
---------------
The union mixes the eight contraction classes, whose functions carry
very different normalizations, so the unscaled operator has a wide
spectrum and LSMR needs thousands of iterations (3697 on H2O, 7501 on N2
without reaching the tolerance).  A diagonal congruence

    D^-1/2 (A - E S) D^-1/2,   D = |diag(A - E S)|

is the cheap remedy: ``diag(S)_i = ||v_i||**2`` and
``diag(A)_i = sum_d V[i, d]**2 <d|H0|d>`` both cost one pass over
``nnz(V)``, because the off-diagonal part of ``H0`` cancels in the
diagonal of ``V H0 V^T`` for these contracted functions (measured
6.1e-16 relative on H2O; the identity is asserted in the tests).  On
H2O the scaling takes the solve from 3697 to 155 iterations without
moving the spectrum, and on N2 it is what makes the single-state solve
converge at all.

H0 ASSEMBLY
-----------
``H0 = sum_pq f[pq] E(pq)`` with ``E(pq) = sum_sigma a^dag_(p sigma)
a_(q sigma)`` is a one-body operator, so its determinant matrix elements
follow from occupation bits alone.  :func:`h0_determinant_matrix` builds
the sparse matrix with vectorized bit operations over the whole
determinant list at once, instead of one Python call per determinant.
The element rule is the one of ``caspt2._e1`` (annihilate ``q``, create
``p``); the builder reproduces the reference sparse assembly
element-for-element on H2O (2.1e-14), 54.9 s against 636 s for the
8052-function N2 support.

The projected coupling ``W = V W_d`` is built by
:func:`projected_coupling`, which scatters the singles and doubles of
every reference determinant with the release Slater-Condon phases
instead of evaluating the kernel on every (reference, determinant) pair
(4.1 s against a projected 94 s on the N2 single-state support,
element-for-element identical).

PUBLIC INTERFACE
----------------
    ``determinant_support``, ``function_matrix``, ``h0_diagonal``,
    ``h0_determinant_matrix``, ``projected_coupling``, ``operator_diagonal``,
    ``coupled_reference_system_mf``, ``solve_coupled_amplitudes_mf``,
    ``CoupledOperator``.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from ..lsmr import lsmr_solve

__all__ = ['determinant_support', 'function_matrix', 'h0_diagonal',
           'h0_determinant_matrix', 'projected_coupling',
           'coupled_reference_system_mf', 'solve_coupled_amplitudes_mf',
           'CoupledOperator', 'operator_diagonal', 'block_preconditioner']


def spin_orbital_integrals(h1e_mo, eri_mo):
    """Spin-orbital ``h`` and ``(pq|rs)`` tensors of ``caspt2._sc_kernel``."""
    h1e_mo = np.asarray(h1e_mo)
    eri_mo = np.asarray(eri_mo)
    nmo = h1e_mo.shape[0]
    nso = 2 * nmo
    pi = np.arange(nso) // 2
    si = np.arange(nso) % 2
    h_so = np.where(si[:, None] == si[None, :],
                    h1e_mo[pi[:, None], pi[None, :]], 0.0)
    m4 = ((si[:, None, None, None] == si[None, None, :, None])
          & (si[None, :, None, None] == si[None, None, None, :]))
    eri_so = np.where(
        m4, eri_mo[pi[:, None, None, None], pi[None, None, :, None],
                   pi[None, :, None, None], pi[None, None, None, :]], 0.0)
    return h_so, eri_so


def _parity(x):
    """``(-1)**popcount(x)`` as a float array."""
    return np.where(np.bitwise_count(x) & 1, -1.0, 1.0)


def projected_coupling(h1e_mo, eri_mo, refs, dets):
    """``Wd[d, N] = sum_J c_J <J|H|d>`` over the determinant support.

    ``refs`` is one list of ``(determinant, coefficient)`` pairs per
    state (the layout of ``ms._reference_pairs``); ``dets`` is the sorted
    determinant array of the union support.  Only determinants that
    differ from a reference by a single or double spin-orbital
    replacement couple, so the reference pair list is enumerated into
    singles and doubles and scattered into the ``nd`` axis, instead of
    scanning every ``(reference, determinant)`` pair.  The element rules
    (phases, the spectator sum of the single-excitation element and the
    zero diagonal) are those of ``caspt2._sc_kernel``.
    """
    h_so, eri_so = spin_orbital_integrals(h1e_mo, eri_mo)
    nso = h_so.shape[0]
    nd = dets.size
    allso = np.arange(nso, dtype=np.int64)
    out = np.zeros((nd, len(refs)))
    for N, ref in enumerate(refs):
        pos_parts, val_parts = [], []
        for J, cj in ref:
            occ_bits = (J >> allso) & 1
            occ = allso[occ_bits.astype(bool)]
            unocc = allso[~occ_bits.astype(bool)]
            weight = float(cj)

            # Single excitations: remove i, create k.
            if occ.size and unocc.size:
                i_arr = np.repeat(occ, unocc.size)
                k_arr = np.tile(unocc, occ.size)
                tgt = J ^ (np.int64(1) << i_arr) ^ (np.int64(1) << k_arr)
                phase = _parity(J & ((np.int64(1) << i_arr) - 1)) * \
                    _parity(tgt & ((np.int64(1) << k_arr) - 1))
                spec = (eri_so[i_arr[:, None], occ[None, :],
                               k_arr[:, None], occ[None, :]]
                        - eri_so[i_arr[:, None], occ[None, :], occ[None, :],
                                 k_arr[:, None]])
                same = occ[None, :] == i_arr[:, None]
                spec = np.where(same, 0.0, spec).sum(axis=1)
                val = h_so[i_arr, k_arr] + spec
                pos_parts.append(tgt)
                val_parts.append(phase * val * weight)

            # Double excitations: remove i1 < i2, create k1 < k2.
            if occ.size < 2 or unocc.size < 2:
                continue
            i_pairs = np.array([(a, b) for ai, a in enumerate(occ)
                                for b in occ[ai + 1:]], dtype=np.int64)
            k_pairs = np.array([(a, b) for ai, a in enumerate(unocc)
                                for b in unocc[ai + 1:]], dtype=np.int64)
            I1 = np.repeat(i_pairs[:, 0], k_pairs.shape[0])
            I2 = np.repeat(i_pairs[:, 1], k_pairs.shape[0])
            K1 = np.tile(k_pairs[:, 0], i_pairs.shape[0])
            K2 = np.tile(k_pairs[:, 1], i_pairs.shape[0])
            tgt2 = (J ^ (np.int64(1) << I1) ^ (np.int64(1) << I2)
                    ^ (np.int64(1) << K1) ^ (np.int64(1) << K2))
            phase2 = (_parity(J & ((np.int64(1) << I1) - 1))
                      * _parity(J & ((np.int64(1) << I2) - 1))
                      * _parity(tgt2 & ((np.int64(1) << K1) - 1))
                      * _parity(tgt2 & ((np.int64(1) << K2) - 1)))
            val2 = (eri_so[I1, I2, K1, K2] - eri_so[I1, I2, K2, K1])
            pos_parts.append(tgt2)
            val_parts.append(phase2 * val2 * weight)

        if not pos_parts:
            continue
        pos_all = np.concatenate(pos_parts)
        val_all = np.concatenate(val_parts)
        col = np.searchsorted(dets, pos_all)
        np.minimum(col, nd - 1, out=col)
        keep = dets[col] == pos_all
        out[:, N] = np.bincount(col[keep], weights=val_all[keep],
                                minlength=nd)
    return out


def determinant_support(functions):
    """Sorted determinant bitmaps reached by a list of function vectors."""
    support = set()
    for _, vec in functions:
        support.update(vec)
    return np.array(sorted(support), dtype=np.int64)


def function_matrix(functions, dets):
    """Sparse ``V`` (functions x determinants) over the sorted ``dets``.

    The coordinate arrays are preallocated from the per-row lengths, so
    the 14.6 million entries of the N2 union never pass through Python
    lists (which would cost an order of magnitude more memory than the
    final CSR).
    """
    index = {int(d): i for i, d in enumerate(dets)}
    n = len(functions)
    counts = np.fromiter((len(vec) for _, vec in functions), dtype=np.int64,
                         count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=float)
    at = 0
    for i, (_, vec) in enumerate(functions):
        m = len(vec)
        indices[at:at + m] = [index[dd] for dd in vec]
        data[at:at + m] = list(vec.values())
        at += m
    V = sp.csr_matrix((data, indices, indptr),
                      shape=(n, len(dets)))
    V.eliminate_zeros()
    return V


def _e_pq_targets(dets, p, q, spin):
    """Row positions, target columns and signs of one spin channel of E(pq).

    ``dets`` is the sorted determinant bitmap array.  The action is
    ``a^dag_(p,spin) a_(q,spin)`` with the fermionic phase of the release
    kernel: annihilate ``q`` first, then create ``p``, each with the
    occupation parity below its own spin orbital.  Returns the source row
    positions, the column of the target determinant (only where the
    target is in the support) and the sign.
    """
    qs = 2 * q + spin
    ps = 2 * p + spin
    occ_q = ((dets >> qs) & 1).astype(bool)
    if p == q:
        rows = np.flatnonzero(occ_q)
        return rows, rows, np.ones(rows.size)
    occ_p = ((dets >> ps) & 1).astype(bool)
    rows = np.flatnonzero(occ_q & ~occ_p)
    if rows.size == 0:
        return rows, rows, np.zeros(0)
    d = dets[rows]
    tgt = d ^ np.int64((1 << qs) | (1 << ps))
    s1 = np.bitwise_count(d & np.int64((1 << qs) - 1))
    s2 = np.bitwise_count((d ^ np.int64(1 << qs))
                          & np.int64((1 << ps) - 1))
    sign = 1.0 - 2.0 * ((s1 + s2) & 1)
    pos = np.searchsorted(dets, tgt)
    np.minimum(pos, dets.size - 1, out=pos)
    valid = dets[pos] == tgt
    return rows[valid], pos[valid], sign[valid]


def h0_diagonal(fock, dets):
    """Diagonal of ``H0`` in the determinant basis.

    ``<d|E(p, p)|d>`` is the occupation of spatial orbital ``p``, so the
    diagonal is ``sum_p f[p, p] n_p(d)`` and costs one pass over the
    determinant list per orbital.
    """
    fock = np.asarray(fock, dtype=float)
    nmo = fock.shape[0]
    dets = np.asarray(dets, dtype=np.int64)
    diag = np.zeros(dets.size)
    for p in range(nmo):
        fpp = fock[p, p]
        if abs(fpp) <= 1e-14:
            continue
        occ = (((dets >> np.int64(2 * p)) & 1)
               + ((dets >> np.int64(2 * p + 1)) & 1))
        diag += fpp * occ
    return diag


def h0_determinant_matrix(fock, dets):
    """Sparse ``H0 = sum_pq f[pq] E(pq)`` over the sorted determinant list.

    The diagonal is the occupation-weighted Fock diagonal, the
    off-diagonals come from :func:`_e_pq_targets`.  Two passes are made
    over the operator pairs: one to count the nonzeros so the three
    coordinate arrays can be allocated once, one to fill them.  Targets
    outside the determinant list are dropped, which keeps the operator
    symmetric (the reverse element is dropped by the same rule).
    """
    fock = np.asarray(fock, dtype=float)
    nmo = fock.shape[0]
    dets = np.asarray(dets, dtype=np.int64)
    nd = dets.size
    nz_pairs = [(p, q) for p in range(nmo) for q in range(nmo)
                if p != q and abs(fock[p, q]) > 1e-14]

    counts = []
    total = nd
    for p, q in nz_pairs:
        n_pair = 0
        for spin in (0, 1):
            rows, _, _ = _e_pq_targets(dets, p, q, spin)
            n_pair += rows.size
        counts.append(n_pair)
        total += n_pair

    data = np.zeros(total, dtype=float)
    rows_out = np.zeros(total, dtype=np.int32)
    cols_out = np.zeros(total, dtype=np.int32)

    data[:nd] = h0_diagonal(fock, dets)
    rows_out[:nd] = np.arange(nd, dtype=np.int32)
    cols_out[:nd] = np.arange(nd, dtype=np.int32)

    at = nd
    for (p, q), n_pair in zip(nz_pairs, counts):
        fpq = fock[p, q]
        for spin in (0, 1):
            rows, cols, sign = _e_pq_targets(dets, p, q, spin)
            n = rows.size
            data[at:at + n] = fpq * sign
            rows_out[at:at + n] = rows.astype(np.int32, copy=False)
            cols_out[at:at + n] = cols.astype(np.int32, copy=False)
            at += n
    if at != total:
        raise RuntimeError('H0 assembly count mismatch '
                           f'({at} of {total})')
    H0 = sp.coo_matrix((data, (rows_out, cols_out)), shape=(nd, nd)).tocsr()
    H0.sum_duplicates()
    return H0


def coupled_reference_system_mf(h1e_mo, eri_mo, civs, dets, *, ncore, nact,
                                nmo=None, fock=None, functions=None,
                                coupling='enumerate'):
    """Assemble ``V``, ``H0`` and ``W`` without forming ``S`` or ``A``.

    Same objects as :func:`diffmrc.core.mscaspt2.ms.coupled_reference_system`
    minus the two ``n_fn x n_fn`` matrices.  The union functions come from
    the release kernel ``_class_function_vectors``.  ``coupling`` selects
    how the projected Hamiltonian coupling ``W`` is built:
    ``'enumerate'`` scatters the single and double excitations of every
    reference determinant (cost ``n_ref * (n_occ n_vir + C(n_occ,2)
    C(n_vir,2))``), ``'scan'`` is the direct reference implementation
    that evaluates the release ``_sc_kernel`` on every (reference,
    determinant) pair (cost ``n_ref * nd``).  The two agree
    element-for-element; ``'scan'`` is kept as the cross-check.
    """
    from ..caspt2 import _sc_kernel
    from .ms import _reference_pairs, union_function_vectors

    civs = np.asarray(civs, dtype=float)
    if nmo is None:
        nmo = np.asarray(h1e_mo).shape[0]
    if fock is None:
        raise ValueError('the reference-space coupling requires a Fock '
                         'matrix')
    if functions is None:
        functions, labels = union_function_vectors(civs, dets, ncore, nact,
                                                   nmo, with_labels=True)
    else:
        labels = None

    dets_sorted = determinant_support(functions)
    index = {int(d): i for i, d in enumerate(dets_sorted)}
    V = function_matrix(functions, dets_sorted)
    H0 = h0_determinant_matrix(fock, dets_sorted)
    h0_diag = h0_diagonal(fock, dets_sorted)

    refs = _reference_pairs(civs, dets)
    if coupling == 'enumerate':
        Wd = projected_coupling(h1e_mo, eri_mo, refs, dets_sorted)
    elif coupling == 'scan':
        h_elem, _ = _sc_kernel(h1e_mo, eri_mo)
        Wd = np.zeros((dets_sorted.size, civs.shape[0]))
        for N, ref in enumerate(refs):
            for dd, ddi in index.items():
                Wd[ddi, N] = sum(c * h_elem(J, dd) for J, c in ref
                                 if (J ^ dd).bit_count() <= 4)
    else:
        raise ValueError("coupling must be 'enumerate' or 'scan'")
    W = np.asarray(V @ Wd)
    return {'functions': functions, 'labels': labels, 'index': index,
            'dets': dets_sorted,
            'V': V, 'H0': H0, 'h0_diag': h0_diag, 'Wd': Wd, 'W': W,
            'coupling': coupling,
            'nd': int(dets_sorted.size), 'nfn': int(V.shape[0])}


def block_preconditioner(V, labels, rtol=1e-8):
    """Blockwise ``S**-1/2`` from the contraction-class labels.

    The union functions arrive in eight contraction classes per
    reference.  The linear dependencies of the union basis are mostly
    within a class, and the class normalizations differ widely, so the
    block-diagonal part of ``S`` already captures most of what makes the
    raw basis badly scaled.  Each block ``S_b = V_b V_b^T`` is formed
    densely (the largest N2 block is a few thousand functions), its
    eigenvalues below ``rtol * max(eig, 1)`` are dropped -- these are
    exactly the within-block null directions of the full ``S``, so
    dropping them does not change the physical amplitudes -- and the
    block map is ``T_b = U_b / sqrt(lambda_b)``.

    Returns the list of ``(indices, T_b)`` pairs and the total reduced
    dimension.
    """
    blocks = {}
    for i, lab in enumerate(labels):
        blocks.setdefault(lab, []).append(i)
    out = []
    for lab in blocks:
        idx = np.asarray(blocks[lab], dtype=np.int64)
        Vb = V[idx]
        Sb = np.asarray((Vb @ Vb.T).toarray())
        Sb = 0.5 * (Sb + Sb.T)
        ev, U = np.linalg.eigh(Sb)
        ev_max = max(float(ev.max()), 1.0)
        keep = ev > rtol * ev_max
        if not np.any(keep):
            continue
        out.append((idx, U[:, keep] / np.sqrt(ev[keep])))
    size = int(sum(T.shape[1] for _, T in out))
    return out, size


def operator_diagonal(V, h0_diag, shift):
    """Diagonal of ``A - shift * S``, matrix-free.

    ``diag(S)_i = ||v_i||^2`` is a row sum of ``V**2``.  For ``diag(A)``
    the off-diagonal part of ``H0`` contributes nothing on the validated
    systems: ``sum_{d != d'} V[i, d] <d|H0|d'> V[i, d']`` cancels to
    round-off because the contracted functions are sums and differences
    of operator products (measured 6.1e-16 relative on H2O, and the
    element rule is checked against the exact diagonal in the tests), so
    ``diag(A)_i = sum_d V[i, d]**2 <d|H0|d>``.  Both terms cost one pass
    over ``nnz(V)``.
    """
    vsq = V.multiply(V)
    diag_s = np.asarray(vsq.sum(axis=1)).reshape(-1)
    diag_a = np.asarray(vsq @ h0_diag).reshape(-1)
    return diag_a - float(shift) * diag_s, diag_s, diag_a


class CoupledOperator:
    """``c -> A c - shift * S c`` with ``A = V H0 V^T`` and ``S = V V^T``.

    The three sparse products are timed separately (``time_vt``,
    ``time_h0``, ``time_v``) so a run reports where the matrix-vector
    product spends its time instead of leaving it to an estimate.

    ``scale`` is a diagonal congruence: the returned operator applies
    ``D**-1/2 (A - shift S) D**-1/2`` to its argument.  The scaling is
    what makes the solve practical -- the union functions of the eight
    contraction classes carry very different normalizations, and the
    unscaled operator needs 24 times the iterations on H2O.
    """

    def __init__(self, V, H0, shift, drop_rows=None, scale=None):
        self.V = V
        self.H0 = H0
        self.shift = float(shift)
        self.drop_rows = drop_rows
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self.calls = 0
        self.time_vt = 0.0
        self.time_h0 = 0.0
        self.time_v = 0.0

    def __call__(self, c):
        self.calls += 1
        c = np.asarray(c, dtype=float).reshape(-1)
        if self.scale is not None:
            c = c * self.scale
        if self.drop_rows is not None:
            c = c.copy()
            c[self.drop_rows] = 0.0
        t0 = time.perf_counter()
        t = self.V.T @ c
        t1 = time.perf_counter()
        t = self.H0 @ t - self.shift * t
        t2 = time.perf_counter()
        out = np.asarray(self.V @ t).reshape(-1)
        t3 = time.perf_counter()
        self.time_vt += t1 - t0
        self.time_h0 += t2 - t1
        self.time_v += t3 - t2
        if self.drop_rows is not None:
            out[self.drop_rows] = 0.0
        if self.scale is not None:
            out = out * self.scale
        return out


def solve_coupled_amplitudes_mf(V, H0, W, reference_fock, *, rtol=1e-10,
                                atol=0.0, maxiter=None, time_limit=None,
                                drop_null_rows=True, precondition='auto',
                                h0_diag=None, labels=None, block_rtol=1e-8,
                                columns=None, record=False):
    """Solve ``A C - S C F = -W`` matrix-free and return the amplitudes.

    ``F`` is diagonalized, which turns the Sylvester equation into one
    shifted system per reference,

        (A - E_j S) c_j = -(W U)[:, j],   C = C' U^T,

    each solved with :func:`diffmrc.core.lsmr.lsmr_solve` to a relative
    residual ``rtol``.

    ``precondition`` selects the change of unknown.  ``'auto'`` (the
    default) uses ``'diag'`` when the ``h0_diag`` diagonal is supplied and
    ``'block'`` otherwise.  ``'block'`` builds the
    block-diagonal ``S**-1/2`` of :func:`block_preconditioner`
    (``y -> T y``, ``u -> T^T u``); this is a congruence, so the physical
    amplitudes are unchanged.  ``'diag'`` applies
    ``D**-1/2 (A - E_j S) D**-1/2`` with ``D = |diag(A - E_j S)|``
    (:func:`operator_diagonal`).  ``'none'`` solves in the raw union
    basis.  ``h0_diag`` is the determinant-basis diagonal of ``H0``
    (``h0_diagonal``), needed by ``'diag'``; ``block_rtol`` is the
    within-block eigenvalue cutoff.

    ``drop_null_rows`` removes function rows that are identically zero
    from the unknown vector; the physical outputs are invariant under
    that restriction because such a row contributes nothing to ``S``,
    ``A`` or ``W``.

    ``columns`` restricts the solve to that subset of the shifted
    systems; the remaining columns of ``C`` stay zero and ``cross`` is
    only complete once every column has been filled in, which is what a
    long run split across jobs needs.

    Returns ``(C, cross, overlap, info)`` with
    ``cross = C^T W = <Psi_M^(1)|H|N>``, ``overlap = C^T S C`` and a
    per-column list of LSMR diagnostics.
    """
    V = sp.csr_matrix(V)
    H0 = sp.csr_matrix(H0)
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    nfn = V.shape[0]

    drop_rows = None
    if drop_null_rows:
        row_nnz = np.diff(V.indptr)
        zero_rows = np.flatnonzero(row_nnz == 0)
        if zero_rows.size:
            drop_rows = zero_rows

    F = np.asarray(reference_fock, dtype=float)
    ef, U = np.linalg.eigh(0.5 * (F + F.T))
    wp = W @ U
    Cp = np.zeros((nfn, wp.shape[1]))
    infos = []
    dropped_scale = None
    floor_rows = None
    diag_s = diag_a = None
    if precondition is True:
        precondition = 'auto'
    elif precondition is False:
        precondition = 'none'
    if precondition == 'auto':
        # the diagonal congruence is the cheaper of the two and the one
        # measured to converge on both validated systems; the block map
        # stays opt-in because its dense block Gram is expensive to build
        precondition = 'diag' if h0_diag is not None else 'block'
    if precondition == 'block' and labels is None:
        raise ValueError("precondition='block' requires the class labels")
    blocks_on = precondition in ('auto', 'block') and labels is not None
    blocks = None
    block_size = 0
    block_time = 0.0
    if blocks_on:
        t_blk = time.perf_counter()
        blocks, block_size = block_preconditioner(V, labels, rtol=block_rtol)
        block_time = time.perf_counter() - t_blk
    if (blocks is None and precondition in ('auto', 'diag')
            and h0_diag is not None):
        vsq = V.multiply(V)
        diag_s = np.asarray(vsq.sum(axis=1)).reshape(-1)
        diag_a = np.asarray(vsq @ np.asarray(h0_diag, dtype=float)).reshape(-1)
        del vsq
    offsets = None
    if blocks is not None:
        offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([T.shape[1] for _, T in blocks])

    def block_apply(y):
        c = np.zeros(nfn)
        for k, (idx, T) in enumerate(blocks):
            c[idx] = T @ y[offsets[k]:offsets[k + 1]]
        return c

    def block_adjoint(u):
        y = np.zeros(block_size)
        for k, (idx, T) in enumerate(blocks):
            y[offsets[k]:offsets[k + 1]] = T.T @ u[idx]
        return y

    solve_cols = (range(wp.shape[1]) if columns is None
                  else [int(j) for j in columns])
    for j in solve_cols:
        scale = None
        if diag_s is not None and blocks is None:
            mdiag = diag_a - ef[j] * diag_s
            mag = np.abs(mdiag)
            floor = 1e-12 * max(float(mag.max()), 1e-300)
            scale = 1.0 / np.sqrt(np.maximum(mag, floor))
            if dropped_scale is None:
                # only rows that are identically zero in V are dropped;
                # a nonzero row is scaled, never removed, because removing
                # it would restrict the unknown vector
                zero_rows_all = np.diff(V.indptr) == 0
                scale[zero_rows_all] = 0.0
                dropped_scale = int(np.sum(zero_rows_all))
                floor_rows = int(np.sum(mag <= floor))
        base = CoupledOperator(V, H0, ef[j], drop_rows=drop_rows, scale=scale)
        if blocks is not None:
            def op(y, base=base):
                return block_adjoint(base(block_apply(y)))
            # a zero function row is a null direction of its block and is
            # already excluded from the block map, so no explicit drop
            b = block_adjoint(-wp[:, j])
        else:
            op = base
            b = -wp[:, j]
            if scale is not None:
                b = b * scale
            if drop_rows is not None:
                b = b.copy()
                b[drop_rows] = 0.0
        x, sub = lsmr_solve(op, b, rtol=rtol, atol=atol, maxiter=maxiter,
                            time_limit=time_limit, record=record)
        if blocks is not None:
            x = block_apply(x)
        elif scale is not None:
            x = x * scale
        sub['matvecs'] = base.calls
        sub['shift'] = float(ef[j])
        sub['relres'] = float(sub['normr'] / max(np.linalg.norm(b), 1e-300))
        sub['time_vt'] = base.time_vt
        sub['time_h0'] = base.time_h0
        sub['time_v'] = base.time_v
        infos.append(sub)
        Cp[:, j] = x
    C = Cp @ U.T

    cross = C.T @ W
    t = np.asarray(V.T @ C)
    overlap = t.T @ t
    info = {'columns': infos,
            'iterations': int(sum(sub['itn'] for sub in infos)),
            'matvecs': int(sum(sub['matvecs'] for sub in infos)),
            'max_relres': float(max(sub['relres'] for sub in infos)),
            'dropped_rows': int(0 if drop_rows is None else drop_rows.size),
            'dropped_scale_rows': int(0 if dropped_scale is None
                                      else dropped_scale),
            'floor_rows': int(0 if floor_rows is None else floor_rows),
            'precondition': ('block' if blocks is not None
                             else 'diag' if diag_s is not None else 'none'),
            'block_size': int(block_size),
            'block_seconds': float(block_time),
            'reasons': [sub['reason'] for sub in infos],
            'time_vt': float(sum(sub['time_vt'] for sub in infos)),
            'time_h0': float(sum(sub['time_h0'] for sub in infos)),
            'time_v': float(sum(sub['time_v'] for sub in infos))}
    return C, cross, overlap, info
