"""Sparse re-assembly of the coupled CASPT2 system at CAS(6,6).

The public coupled path assembles S and A with a Python double loop over
all function pairs and caches one H0 action per determinant.  At N2/6-31G
CAS(6,6) the function space has 5532 functions whose determinant union is
about 2.3 million, so both the pair loop and the cache are impractical.

This probe builds the same objects with sparse linear algebra:

    F[i, d] = coefficient of determinant d in function i
    S = F F^T,   W = F g,   A = F H F^T

where H is the sparse matrix of the state-specific generalized Fock over
the determinant union.  Function basis, state 1-RDM, generalized Fock and
Slater-Condon kernel come from the library, and H is built from the same
bit rules as ``_h0_action`` (occupied -> empty moves plus the occupation
diagonal).  On H2O the assembly reproduces the public entry point to
2e-16, so agreement here checks the assembly algebra.

Usage:
    python caspt2_sparse_assembly.py --ref n2_ref.npz [--library] [--block 200000]
    python caspt2_sparse_assembly.py --ref h2o_ref.npz --library
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from diffmrc.core.caspt2 import (_class_function_vectors, _generalized_fock,
                                 _sc_kernel, _state_1rdm)

CLASSES = ['V0_ijab', 'Vm1_iab', 'Vm2_ab', 'V1_ija', 'V2_ij', 'V0_ia',
           'Vm1_a', 'V1_i']


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def build_h0(detarr, f, block=200000):
    """Sparse H0 over the sorted determinant array ``detarr``.

    Same rules as ``_h0_action``: E(p,q) moves an electron from spin
    orbital q to p with the annihilate-then-create sign, E(p,p) adds
    f[p,p] once per electron in spatial orbital p.  Targets outside the
    determinant array are dropped (they cannot reach any K_i).
    """
    nd = len(detarr)
    nmo = f.shape[0]
    nso = 2 * nmo
    rows, cols, vals = [], [], []
    d = np.int64(1)
    for b0 in range(0, nd, block):
        b1 = min(b0 + block, nd)
        D = detarr[b0:b1]
        occ = ((D[:, None] >> np.arange(nso, dtype=np.int64)[None, :])
               & d).astype(bool)
        lowpop = np.zeros((b1 - b0, nso + 1), dtype=np.int32)
        for so in range(nso):
            lowpop[:, so + 1] = lowpop[:, so] + occ[:, so]
        src0 = np.arange(b0, b1, dtype=np.int64)
        # Diagonal: sum_p f_pp * n_p, n_p in {0,1,2}.
        diag = np.zeros(b1 - b0)
        for p in range(nmo):
            diag += f[p, p] * (occ[:, 2 * p].astype(np.int32)
                               + occ[:, 2 * p + 1].astype(np.int32))
        rows.append(src0)
        cols.append(src0)
        vals.append(diag)
        for p in range(nmo):
            for q in range(nmo):
                fpq = f[p, q]
                if p == q or fpq == 0.0:
                    continue
                for spin in (0, 1):
                    qs, ps = 2 * q + spin, 2 * p + spin
                    valid = occ[:, qs] & ~occ[:, ps]
                    if not valid.any():
                        continue
                    src = src0[valid]
                    low_q = lowpop[valid, qs]
                    low_p = lowpop[valid, ps]
                    sign = np.where((low_q + low_p
                                     - (1 if ps > qs else 0)) % 2,
                                    -1.0, 1.0)
                    tgt = ((D[valid] ^ (d << qs)) | (d << ps))
                    ipos = np.searchsorted(detarr, tgt)
                    ok = ipos < nd
                    ok[ok] = detarr[ipos[ok]] == tgt[ok]
                    rows.append(ipos[ok])
                    cols.append(src[ok])
                    vals.append(fpq * sign[ok])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    log('H0 entries %.3e; building csr' % len(vals))
    H = sp.coo_matrix((vals, (rows, cols)), shape=(nd, nd)).tocsr()
    H.sum_duplicates()
    log('H0 nnz %.3e' % H.nnz)
    return H


def solve_system(S, A, W, e0, labels, rtol=1e-8):
    ev, evec = np.linalg.eigh(S)
    keep = ev > rtol * max(float(ev.max()), 1.0)
    T = evec[:, keep] / np.sqrt(ev[keep])
    M = T.T @ (A - e0 * S) @ T
    b = -T.T @ W
    y = np.linalg.solve(M, b)
    x = T @ y
    e2 = float(W @ x)
    bd = {k: 0.0 for k in CLASSES}
    for name, wi, xi in zip(labels, W, x):
        bd[name] += wi * xi
    return e2, bd


def sparse_coupled(h1, eri, ci, dets, ncore, nact, nele, block=200000):
    nmo = h1.shape[0]
    ref_pairs = [(int(x), float(c)) for x, c in zip(dets, ci)
                 if abs(c) > 1e-14]
    log('state 1-RDM')
    D = _state_1rdm(ref_pairs, nmo)
    f = _generalized_fock(h1, eri, D)
    h_elem, _ = _sc_kernel(h1, eri)
    ref_arr = np.array([J for J, _ in ref_pairs], dtype=np.int64)
    ref_c = np.array([c for _, c in ref_pairs])
    g_cache = {}

    def g_of(dd):
        if dd not in g_cache:
            dd = int(dd)
            near = np.nonzero(np.bitwise_count(ref_arr ^ dd) <= 4)[0]
            g_cache[dd] = sum(ref_c[i] * h_elem(int(ref_arr[i]), dd)
                              for i in near)
        return g_cache[dd]

    log('function vectors')
    functions = _class_function_vectors(ref_pairs, ncore, nact, nmo)
    labels = [name for name, _ in functions]
    nfn = len(functions)
    detset = set()
    for _, vec in functions:
        detset.update(vec.keys())
    detarr = np.array(sorted(detset), dtype=np.int64)
    nd = len(detarr)
    log('functions %d, determinants %d' % (nfn, nd))

    log('F assembly')
    pos = {int(d): i for i, d in enumerate(detarr.tolist())}
    rows = np.empty(sum(len(v) for _, v in functions), dtype=np.int32)
    cols = np.empty_like(rows)
    vals = np.empty(len(rows))
    k = 0
    for i, (_, vec) in enumerate(functions):
        for d, c in vec.items():
            rows[k] = i
            cols[k] = pos[d]
            vals[k] = c
            k += 1
    F = sp.coo_matrix((vals, (rows, cols)), shape=(nfn, nd)).tocsr()
    del pos
    log('F nnz %d' % F.nnz)

    gv = np.array([g_of(int(x)) for x in detarr.tolist()])
    W = F @ gv
    log('W done; S = F F^T')
    S = (F @ F.T).toarray()
    S = 0.5 * (S + S.T)
    log('S done; building H0')

    H = build_h0(detarr, f, block=block)
    log('H F^T')
    HF = H @ F.T
    log('HF nnz %.3e; A = F HF' % HF.nnz)
    A = (F @ HF).toarray()
    A = 0.5 * (A + A.T)
    e0 = float(np.sum(f * D))
    log('solving reduced system')
    e2, bd = solve_system(S, A, W, e0, labels)
    return e2, bd, nfn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--library', action='store_true')
    ap.add_argument('--block', type=int, default=200000)
    args = ap.parse_args()

    d = np.load(args.ref)
    h1, eri, ci = d['h1_mo'], d['eri_mo'], d['ci']
    dets = list(d['dets'])
    ncore, nact, nele = int(d['ncore']), int(d['nact']), int(d['nele'])
    log('reference: ncore=%d nact=%d nele=%d nmo=%d'
        % (ncore, nact, nele, h1.shape[0]))

    e2, bd, nfn = sparse_coupled(h1, eri, ci, dets, ncore, nact, nele,
                                 block=args.block)
    print('sparse coupled total %+18.12f  nfn=%d' % (e2, nfn))
    for k in CLASSES:
        print('  %-9s %+18.12f' % (k, bd[k]))
    print('  sum classes %+18.12f' % sum(bd.values()))

    if args.library:
        from diffmrc.core.caspt2 import caspt2_contracted_e2_coupled
        e2l, bdl = caspt2_contracted_e2_coupled(
            h1, eri, ci, dets, ncore=ncore, nact=nact, nele=nele)
        print('library coupled total %+18.12f' % e2l)
        print('difference sparse-library %.3e' % (e2 - e2l))


if __name__ == '__main__':
    main()
