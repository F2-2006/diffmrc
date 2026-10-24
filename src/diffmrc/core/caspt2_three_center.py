#!/usr/bin/env python
"""Contracted CASPT2 with the coupling assembled from three-center factors.

This is the value-layer companion of
:func:`diffmrc.core.caspt2.caspt2_contracted_e2`.  The class definitions,
the overlaps ``S``, the diagonal-Fock expectations ``A`` and the V0_ia
block equation are the same; only the ERI-dependent coupling ``W`` is
built differently.

Instead of a four-center MO tensor ``eri_mo[a, b, c, d]``, the routine
takes a three-center factor ``u_mo`` with

    eri_mo[a, b, c, d] = sum_P u_mo[a, b, P] u_mo[c, d, P]

(the metric factor already folded in, as returned by
``diffmrc.core.ri.ri_eri_mo``).  Every Slater-Condon ERI term is
scattered into a pair-density vector ``Z[a, b, P]`` and the coupling is
one contraction ``sum_{a b P} Z[a, b, P] u_mo[a, b, P]``.  Neither the
four-center MO tensor nor the spin-orbital tensor is formed, so the
working set scales as ``nmo^2 * naux`` instead of ``nmo^4`` (or
``16 * nmo^4`` for the spin-orbital array of the materialized path).

Normalisation follows the materialized path exactly: the two orderings
of a double excitation enter with relative weight -1 for the ``VM``
component (``cross - direct``; the OpenMolcas ``mkrhsh`` sqrt(3) is an
overall scale that cancels in ``W^2 / (A - E0 S)``).  Keeping the two
paths on the same convention is what makes their class tables
comparable.

Scope: value layer only.  The construction is plain numpy and is not
wired into the jax chain; the differentiable production path remains
:func:`diffmrc.core.caspt2.caspt2_contracted_e2_jax`.
"""
from collections import defaultdict

import numpy as np

from .caspt2 import _apply2, _e1

__all__ = ['caspt2_contracted_e2_three_center']


def _so_bits(nso):
    """Occupied spin-orbital list of a determinant bitmap, with a cache."""
    cache = {}

    def f(x):
        if x not in cache:
            cache[x] = [i for i in range(nso) if (x >> i) & 1]
        return cache[x]
    return f


def _h_terms(J, D, h1e):
    """Terms of ``<J|H|D>`` for determinant bitmaps.

    Mirrors ``caspt2._sc_kernel.h_elem`` term by term and returns
    ``(h_terms, e_terms)``: ``(p, q, coef)`` multiplying ``h1e[p, q]``
    and ``(a, b, c, d, coef)`` multiplying ``eri_mo[a, b, c, d]`` in the
    chemist layout.  The ERI index order follows the spin-orbital folding
    of ``_sc_kernel``::

        eri_so[i, j, k, l] = eri_mo[i // 2, k // 2, j // 2, l // 2]

    A term is an exact rewrite of the corresponding ``h_elem`` branch, so
    contracting the terms with the integrals reproduces ``h_elem``
    elementwise.
    """
    if J == D:
        return (), ()
    dj, dk = J & ~D, D & ~J
    n = dj.bit_count()
    if n != dk.bit_count() or n > 2:
        return (), ()
    h_terms, e_terms = [], []
    if n == 1:
        i = (dj & -dj).bit_length() - 1
        k = (dk & -dk).bit_length() - 1
        sgn = (-1.0) ** ((J & ((1 << i) - 1)).bit_count()
                         + (D & ((1 << k) - 1)).bit_count())
        if (i & 1) != (k & 1):
            return (), ()
        pi_, pk_ = i >> 1, k >> 1
        h_terms.append((pi_, pk_, sgn))
        common = J & D
        while common:
            so = (common & -common).bit_length() - 1
            common &= common - 1
            q = so >> 1
            e_terms.append((pi_, pk_, q, q, sgn))
            if (so & 1) == (i & 1):
                e_terms.append((pi_, q, q, pk_, -sgn))
    else:
        bits_j = [b for b in range(dj.bit_length()) if (dj >> b) & 1]
        bits_k = [b for b in range(dk.bit_length()) if (dk >> b) & 1]
        i1, i2 = bits_j
        k1, k2 = bits_k
        sgn = (-1.0) ** ((J & ((1 << i1) - 1)).bit_count()
                         + (J & ((1 << i2) - 1)).bit_count()
                         + (D & ((1 << k1) - 1)).bit_count()
                         + (D & ((1 << k2) - 1)).bit_count())
        if (i1 & 1) == (k1 & 1) and (i2 & 1) == (k2 & 1):
            e_terms.append((i1 >> 1, k1 >> 1, i2 >> 1, k2 >> 1, sgn))
        if (i1 & 1) == (k2 & 1) and (i2 & 1) == (k1 & 1):
            e_terms.append((i1 >> 1, k2 >> 1, i2 >> 1, k1 >> 1, -sgn))
    return tuple(h_terms), tuple(e_terms)


def _g_terms(dd, ref_pairs, h1e, cache):
    """Hamiltonian terms of ``g(D) = sum_J c_J <J|H|D>`` for one target."""
    if dd in cache:
        return cache[dd]
    hacc = defaultdict(float)
    eacc = defaultdict(float)
    for J, cJ in ref_pairs:
        if (J ^ dd).bit_count() > 4:
            continue
        h_terms, e_terms = _h_terms(J, dd, h1e)
        for p, q, coef in h_terms:
            hacc[(p, q)] += cJ * coef
        for a, b, c, d, coef in e_terms:
            eacc[(a, b, c, d)] += cJ * coef
    cache[dd] = (dict(hacc), dict(eacc))
    return cache[dd]


def _w_three_center(kacc, h1e, u, cache, ref_pairs):
    """``W = sum_D kacc[D] g(D)`` with ``eri = U U^T``.

    The ERI coefficient of every term is scattered into the pair-density
    tensor ``Z[a, b, P]`` (one length-``naux`` vector per term) and the
    result is contracted with ``U`` in a single pass.  No four-center
    array is ever formed.
    """
    nmo = h1e.shape[0]
    wh = 0.0
    z = None
    for dd, kc in kacc.items():
        if kc == 0.0:
            continue
        h_terms, e_terms = _g_terms(dd, ref_pairs, h1e, cache)
        for (p, q), coef in h_terms.items():
            wh += kc * coef * h1e[p, q]
        if e_terms:
            if z is None:
                z = np.zeros((nmo, nmo, u.shape[2]))
            for (a, b, c, d), coef in e_terms.items():
                z[a, b] += (kc * coef) * u[c, d]
    if z is None:
        return wh
    return wh + float(np.einsum('abp,abp->', z, u))


def caspt2_contracted_e2_three_center(h1e_mo, u_mo, ci, dets, *, ncore,
                                      nact, nele, eps):
    """Contracted CASPT2 with ``W`` assembled from three-center factors.

    Parameters match :func:`diffmrc.core.caspt2.caspt2_contracted_e2`
    except that the four-center MO ERI is replaced by ``u_mo``, the
    metric-folded three-center tensor in the MO basis (see the module
    docstring).  Returns ``(e2_total, breakdown)`` keyed by class name.

    The eight classes, the relative weight of the two orderings (VP +1,
    VM -1) and the V0_ia block solve are the same as the materialized
    path, so on the same reference the two routines agree to roundoff.
    """
    h1e = np.asarray(h1e_mo)
    u = np.asarray(u_mo)
    ci = np.asarray(ci).ravel()
    eps = np.asarray(eps, dtype=float)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError('h1e_mo must be a square matrix')
    nmo = h1e.shape[0]
    if u.ndim != 3 or u.shape[0] != nmo or u.shape[1] != nmo:
        raise ValueError('u_mo must have shape (nmo, nmo, naux)')
    if eps.shape != (nmo,):
        raise ValueError('eps must have length nmo')
    nocc = ncore + nact
    ref_pairs = [(int(d), float(c)) for d, c in zip(dets, ci)
                 if abs(c) > 1e-14]
    so_bits = _so_bits(2 * nmo)
    g_cache = {}

    def w_of(kacc):
        return _w_three_center(kacc, h1e, u, g_cache, ref_pairs)

    def wsa(qs, ps, vm=False, pair='both'):
        if len(qs) == 1:
            q, p = qs[0], ps[0]
            kacc = defaultdict(float)
            for I, cI in ref_pairs:
                for d1, cc in _e1(I, p, q):
                    kacc[d1] += cI * cc
        else:
            q1, q2 = qs
            p1, p2 = ps
            # Same operator content as the materialized path: the VM
            # component keeps cross - direct (relative weight -1); the
            # OpenMolcas sqrt(3) is an overall scale and cancels in
            # W^2 / (A - E0 S).
            fac = -1.0 if vm else 1.0
            kacc = defaultdict(float)
            for I, cI in ref_pairs:
                if pair in ('both', 'direct'):
                    for d2, cc in _apply2(I, p1, q1, p2, q2).items():
                        kacc[d2] += cI * cc
                if pair in ('both', 'cross'):
                    for d2, cc in _apply2(I, p1, q2, p2, q1).items():
                        kacc[d2] += fac * cI * cc
        w = w_of(kacc)
        s = sum(cc * cc for cc in kacc.values())
        occ = np.zeros(nmo)
        for dd, cc in kacc.items():
            c2 = cc * cc
            for so in so_bits(dd):
                occ[so >> 1] += c2
        return w, s, float(eps @ occ)

    n0 = np.zeros(nmo)
    for dd, c in ref_pairs:
        for so in so_bits(dd):
            n0[so >> 1] += c * c
    e0_zero = float(eps @ n0)

    core = list(range(ncore))
    act = list(range(ncore, nocc))
    vir = list(range(nocc, nmo))

    def pairs(pool_a, pool_b):
        if pool_b is None:
            return [(x,) for x in pool_a]
        if pool_a is pool_b:
            return [(pool_a[i], pool_a[j]) for i in range(len(pool_a))
                    for j in range(i, len(pool_a))]
        return [(x, y) for x in pool_a for y in pool_b]

    classes = [
        ('V0_ijab', core, core, vir, vir),
        ('Vm1_iab', act, core, vir, vir),
        ('Vm2_ab', act, act, vir, vir),
        ('V1_ija', core, core, act, vir),
        ('V2_ij', core, core, act, act),
        ('Vm1_a', act, None, vir, None),
        ('V1_i', core, None, act, None),
    ]
    breakdown = {}
    total = 0.0
    for name, o1, o2, i1c, i2c in classes:
        acc = 0.0
        for qs in pairs(o1, o2):
            if len(qs) == 2 and qs[0] == qs[1] and qs[0] in act \
                    and qs[0] != act[0]:
                continue
            for ps in pairs(i1c, i2c):
                if len(ps) == 2 and ps[0] == ps[1] and ps[0] in act \
                        and ps[0] != act[0]:
                    continue
                w, s, a = wsa(qs, ps, vm=False)
                dnm = a - e0_zero * s
                if abs(dnm) > 1e-10:
                    acc -= w * w / dnm
                if len(qs) == 2 and qs[0] != qs[1] and ps[0] != ps[1]:
                    wm, sm, am = wsa(qs, ps, vm=True)
                    dnm2 = am - e0_zero * sm
                    if abs(dnm2) > 1e-10:
                        acc -= wm * wm / dnm2
        breakdown[name] = acc
        total += acc

    acc_ia = 0.0
    de_cache = {}

    def de_of(dd):
        if dd not in de_cache:
            tot = 0.0
            for p in range(nmo):
                tot += ((dd >> (2 * p)) & 3).bit_count() * eps[p]
            de_cache[dd] = tot
        return de_cache[dd]

    def kacc_of(mode, uu, j, a, vv):
        kacc = defaultdict(float)
        for I, cI in ref_pairs:
            terms = (_apply2(I, a, j, vv, uu) if mode == 'cross'
                     else _apply2(I, a, uu, vv, j))
            for d2, cc in terms.items():
                kacc[d2] += cI * cc
        return kacc

    for iu, uu in enumerate(act):
        for vv in act[iu:]:
            modes = ('cross',) if vv == uu else ('cross', 'direct')
            for j in core:
                for a in vir:
                    ks = [kacc_of(m, uu, j, a, vv) for m in modes]
                    ws = [w_of(k) for k in ks]
                    n = len(ks)
                    mmat = np.zeros((n, n))
                    for p in range(n):
                        for q in range(n):
                            shared = ks[p].keys() & ks[q].keys()
                            mmat[p, q] = sum(ks[p][d] * ks[q][d]
                                             * (de_of(d) - e0_zero)
                                             for d in shared)
                    if n == 1:
                        if abs(mmat[0, 0]) > 1e-10:
                            acc_ia -= ws[0] * ws[0] / mmat[0, 0]
                    else:
                        det = (mmat[0, 0] * mmat[1, 1]
                               - mmat[0, 1] * mmat[1, 0])
                        if abs(det) > 1e-12:
                            x = np.linalg.solve(mmat, -np.asarray(ws))
                            acc_ia += float(np.dot(ws, x))
    breakdown['V0_ia'] = acc_ia
    total += acc_ia
    return total, breakdown
