"""Sparse coupled CASPT2 with the TUVA / ITUV classes included.

Same sparse assembly as the Wave 11 script (F F^T, F H F^T with the
state-specific generalized Fock), but the function basis replaces the
two single-excitation placeholders by the plain FIC products

    Vm1_a (TUVA): E(a,t) E(u,v) |0>,  a in virtual, t,u,v in active
    V1_i  (ITUV): E(t,i) E(u,v) |0>,  i in core,    t,u,v in active

The other six classes keep the library basis, so the change isolates the
two classes (the full projected solve is basis-invariant within a class).

Usage:
    python caspt2_sparse_extended.py --ref n2_ref.npz [--rtol 1e-6]
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from diffmrc.core.caspt2 import (_apply_chain, _class_function_vectors,
                                 _generalized_fock, _sc_kernel, _state_1rdm)

CLASSES = ['V0_ijab', 'Vm1_iab', 'Vm2_ab', 'V1_ija', 'V2_ij', 'V0_ia',
           'Vm1_a', 'V1_i']


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def extended_functions(ref_pairs, ncore, nact, nmo):
    """Library functions minus the two single-excitation classes, plus
    the plain-product TUVA / ITUV functions."""
    nocc = ncore + nact
    core = list(range(ncore))
    act = list(range(ncore, nocc))
    vir = list(range(nocc, nmo))
    out = [(name, vec) for name, vec in
           _class_function_vectors(ref_pairs, ncore, nact, nmo)
           if name not in ('Vm1_a', 'V1_i')]
    log('library functions kept %d' % len(out))
    for a in vir:
        for t in act:
            for u in act:
                for v in act:
                    out.append(('Vm1_a', _apply_chain(ref_pairs,
                                                      [(a, t), (u, v)])))
    for i in core:
        for t in act:
            for u in act:
                for v in act:
                    out.append(('V1_i', _apply_chain(ref_pairs,
                                                     [(t, i), (u, v)])))
    log('extended functions %d' % len(out))
    return out


def build_h0(detarr, f, block=200000):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--rtol', type=float, default=1e-6)
    ap.add_argument('--block', type=int, default=200000)
    args = ap.parse_args()

    z = np.load(args.ref)
    h1 = np.asarray(z['h1_mo'])
    eri = np.asarray(z['eri_mo'])
    ci = np.asarray(z['ci']).ravel()
    dets = [int(x) for x in z['dets']]
    ncore, nact, nele = int(z['ncore']), int(z['nact']), int(z['nele'])
    nmo = h1.shape[0]
    log('reference: ncore=%d nact=%d nele=%d nmo=%d'
        % (ncore, nact, nele, nmo))

    ref_pairs = [(int(d), float(c)) for d, c in zip(dets, ci)
                 if abs(c) > 1e-14]
    D = _state_1rdm(ref_pairs, nmo)
    f = _generalized_fock(h1, eri, D)
    h_elem, _ = _sc_kernel(h1, eri)
    ref_arr = np.array([J for J, _ in ref_pairs], dtype=np.int64)
    ref_c = np.array([c for _, c in ref_pairs])
    g_cache = {}

    def g_of(dd):
        g = g_cache.get(dd)
        if g is None:
            near = np.nonzero(np.bitwise_count(ref_arr ^ np.int64(dd)) <= 4)[0]
            g = float(sum(ref_c[i] * h_elem(int(ref_arr[i]), dd)
                          for i in near))
            g_cache[dd] = g
        return g

    functions = extended_functions(ref_pairs, ncore, nact, nmo)
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
    rows = np.empty(sum(len(v) for _, v in functions), dtype=np.int64)
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
    H = build_h0(detarr, f, block=args.block)
    log('H F^T')
    HF = H @ F.T
    log('HF nnz %.3e; A = F HF' % HF.nnz)
    A = (F @ HF).toarray()
    A = 0.5 * (A + A.T)
    e0 = float(np.sum(f * D))
    log('solving reduced system, rtol=%.1e' % args.rtol)
    ev, evec = np.linalg.eigh(S)
    evmax = max(float(ev.max()), 1.0)
    keep = ev > args.rtol * evmax
    log('S rank %d / %d' % (int(keep.sum()), nfn))
    T = evec[:, keep] / np.sqrt(ev[keep])
    M = T.T @ (A - e0 * S) @ T
    b = -T.T @ W
    y = np.linalg.solve(M, b)
    x = T @ y
    e2 = float(W @ x)
    bd = {name: 0.0 for name in CLASSES}
    for name, wi, xi in zip(labels, W, x):
        bd[name] += wi * xi
    print('sparse coupled total %+18.12f  nfn=%d' % (e2, nfn), flush=True)
    for kk in CLASSES:
        print('  %-9s %+18.12f' % (kk, bd[kk]), flush=True)
    print('  sum classes %+18.12f' % sum(bd.values()), flush=True)


if __name__ == '__main__':
    main()
