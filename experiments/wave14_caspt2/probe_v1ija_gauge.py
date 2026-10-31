"""Gauge dependence of the V1_ija diagonal approximation on N2.

The plain-product class basis of V1_ija has a degenerate overlap spectrum.
The diagonal approximation taken in the orthonormal basis then depends on
the basis chosen inside each degenerate block.  This probe evaluates the
same class with (a) the raw LAPACK eigenbasis, (b) the block gauge that
diagonalizes the Fock matrix inside each block (the production gauge),
(c) random rotations inside each degenerate block, and (d) the exact
class-block solve H2, which is gauge invariant.  The spread across (a)
and (c) is the honest uncertainty of the H1 variant; the H2 value is the
gauge-invariant answer of the same class space.

Usage: python probe_v1ija_gauge.py <n2_ref.npz>
"""
import sys

import numpy as np

from diffmrc.core.caspt2 import (_apply_chain, _plain_class_ops, _sc_kernel)

TOL = 1e-9


def forms(vecs, w, de_of, e0):
    ds = sorted(set().union(*(v.keys() for v in vecs)))
    pos = {d: i for i, d in enumerate(ds)}
    rows, cols, vals = [], [], []
    for i, vec in enumerate(vecs):
        for d, c in vec.items():
            rows.append(i)
            cols.append(pos[d])
            vals.append(c)
    import scipy.sparse as sp
    F = sp.coo_matrix((vals, (rows, cols)),
                      shape=(len(vecs), len(ds))).tocsr()
    S = (F @ F.T).toarray()
    de = np.asarray([de_of(d) for d in ds])
    A = (F.multiply((de - e0)[None, :]) @ F.T).toarray()
    return 0.5 * (S + S.T), 0.5 * (A + A.T), np.asarray(w)


def h1_raw(S, A, w, nthresh=1e-6):
    ev, U = np.linalg.eigh(S)
    keep = ev > nthresh * ev.max()
    T = U[:, keep] / np.sqrt(ev[keep])
    M = T.T @ A @ T
    Wt = T.T @ w
    return -float(np.sum(Wt * Wt / np.diag(M)))


def h1_gauge(S, A, w, nthresh=1e-6, random_state=None):
    ev, U = np.linalg.eigh(S)
    ev_max = float(ev.max())
    keep = ev > nthresh * ev_max
    U = U[:, keep]
    evk = ev[keep]
    start = 0
    rng = (np.random.default_rng(random_state)
           if random_state is not None else None)
    for i in range(1, len(evk) + 1):
        if i == len(evk) or evk[i] - evk[i - 1] > TOL * ev_max:
            if i - start > 1:
                block = U[:, start:i]
                if rng is None:
                    Ab = block.T @ A @ block
                    _, V = np.linalg.eigh(0.5 * (Ab + Ab.T))
                else:
                    q, _ = np.linalg.qr(rng.normal(size=(i - start,
                                                         i - start)))
                    V = q
                U[:, start:i] = block @ V
            start = i
    T = U / np.sqrt(evk)
    M = T.T @ A @ T
    Wt = T.T @ w
    return -float(np.sum(Wt * Wt / np.diag(M)))


def h2(S, A, w, nthresh=1e-6):
    ev, U = np.linalg.eigh(S)
    keep = ev > nthresh * ev.max()
    T = U[:, keep] / np.sqrt(ev[keep])
    M = 0.5 * ((T.T @ A @ T) + (T.T @ A @ T).T)
    Wt = T.T @ w
    return float(Wt @ np.linalg.solve(M, -Wt))


def main(path, cls='V1_ija'):
    z = np.load(path)
    h1 = np.asarray(z['h1_mo'])
    eri = np.asarray(z['eri_mo'])
    ci = np.asarray(z['ci']).ravel()
    dets = [int(x) for x in z['dets']]
    ncore, nact, nele = int(z['ncore']), int(z['nact']), int(z['nele'])
    eps = np.asarray(z['eps'], dtype=float)
    nmo = h1.shape[0]
    nocc = ncore + nact
    ref = [(d, float(c)) for d, c in zip(dets, ci) if abs(c) > 1e-14]
    h_elem, so_bits = _sc_kernel(h1, eri)

    def g_of(dd):
        return sum(cJ * h_elem(J, dd) for J, cJ in ref
                   if (J ^ dd).bit_count() <= 4)

    n0 = np.zeros(nmo)
    for dd, c in ref:
        for so in so_bits(dd):
            n0[so // 2] += c * c
    e0 = float(eps @ n0)

    def de_of(dd):
        return sum(((dd >> (2 * p)) & 3).bit_count() * eps[p]
                   for p in range(nmo))

    core = list(range(ncore))
    act = list(range(ncore, nocc))
    vir = list(range(nocc, nmo))
    ops = _plain_class_ops(cls, core, act, vir)
    vecs = [_apply_chain(ref, o) for o in ops]
    w = np.asarray([sum(c * g_of(d) for d, c in v.items()) for v in vecs])
    S, A, w = forms(vecs, w, de_of, e0)
    ev_s = np.linalg.eigvalsh(S)
    rank = int((ev_s > 1e-6 * ev_s.max()).sum())
    print('%s nfn %d rank(1e-6) %d' % (cls, len(vecs), rank))
    print('H1 raw LAPACK     %+.9f' % h1_raw(S, A, w))
    print('H1 block-A gauge  %+.9f' % h1_gauge(S, A, w))
    vals = [h1_gauge(S, A, w, random_state=s) for s in range(5)]
    print('H1 random gauges  ' + ' '.join('%+.9f' % v for v in vals))
    print('H1 spread (random) %.3e' % (max(vals) - min(vals)))
    print('H2 exact solve    %+.9f' % h2(S, A, w))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'V1_ija')
