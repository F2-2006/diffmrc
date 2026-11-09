"""Determinant algebra and Slater-Condon Hamiltonian construction.

PURPOSE
-------
Provides the determinant primitives used by the MRCI space construction
and the dense Hamiltonian build: second-quantized creation and
annihilation on bitmask determinants, spin-free excitation operators,
the <S^2> expectation value, and the Slater-Condon matrix elements for
an orthonormal spin-orbital basis.

CONVENTIONS
-----------
* 2*norb spin orbitals, alpha block first: P = 0..norb-1 is alpha,
  P = norb..2*norb-1 is beta.  A determinant is a Python int bitmask.
* ``build_hamiltonian`` takes ``h1`` and ``eri`` in the MO basis with the
  frozen core already folded in; ``eri`` is in chemist notation,
  ``eri[p, q, r, s] = (pq|rs)``.
* ``dets_with_ms`` enumerates one M_S sector only; all callers in this
  package use M_S = 0.

VALIDATED SCOPE
---------------
The dense Hamiltonian reproduces pyscf FCI in the same CI space to
1e-12 Eh for H2O/6-31G CAS(2,2) and CH2/STO-3G CAS(4,4), and the
CH2/6-31G spectrum matches pyscf FCI root by root (tests/core/test_mrci.py).

BOUNDARIES
----------
* Bitmask determinants do not carry point-group or spin-adaptation
  labels; spin purity is checked a posteriori through ``s2_expectation``.
* ``build_hamiltonian`` is a dense O(N * n_occ^2 * n_virt^2) Python
  loop; it is the memory and time bottleneck of the package.
"""

from __future__ import annotations

import itertools

import numpy as np


def annihilate(mask: int, orb: int):
    """Apply a_orb.  Returns (new_mask, sign) or None."""
    if not (mask >> orb) & 1:
        return None
    sign = -1 if (mask & ((1 << orb) - 1)).bit_count() % 2 else 1
    return mask ^ (1 << orb), sign


def create(mask: int, orb: int):
    """Apply a^dagger_orb.  Returns (new_mask, sign) or None."""
    if (mask >> orb) & 1:
        return None
    sign = -1 if (mask & ((1 << orb) - 1)).bit_count() % 2 else 1
    return mask | (1 << orb), sign


def apply_single(mask: int, dst: int, src: int):
    """Apply a^dagger_dst a_src.  Returns (new_mask, sign) or None."""
    r = annihilate(mask, src)
    if r is None:
        return None
    m, s1 = r
    r = create(m, dst)
    if r is None:
        return None
    m, s2 = r
    return m, s1 * s2


def occupied(mask: int, nso: int) -> list[int]:
    return [p for p in range(nso) if (mask >> p) & 1]


def spin_block(p: int, norb: int) -> int:
    return p // norb


def build_hamiltonian(dets, norb, h1, eri):
    """Dense Hamiltonian in a determinant basis.

    Parameters
    ----------
    dets : list[int]
        Determinant bitmasks over 2*norb spin-orbitals.
    norb : int
        Number of spatial orbitals.
    h1 : ndarray (norb, norb)
        One-electron integrals in the MO basis (core contribution folded in).
    eri : ndarray (norb, norb, norb, norb)
        Two-electron integrals in chemist notation, eri[p,q,r,s] = (pq|rs).

    Returns
    -------
    ndarray (len(dets), len(dets))
    """
    index = {d: i for i, d in enumerate(dets)}
    nso = 2 * norb
    n = len(dets)
    H = np.zeros((n, n))

    for j, det in enumerate(dets):
        occ = occupied(det, nso)
        virt = [p for p in range(nso) if not (det >> p) & 1]

        # Diagonal
        e = 0.0
        for p in occ:
            sp, pp = divmod(p, norb)
            e += h1[pp, pp]
        for a in range(len(occ)):
            sp, p = divmod(occ[a], norb)
            for b in range(len(occ)):
                sq, q = divmod(occ[b], norb)
                e += 0.5 * eri[p, p, q, q]
                if sp == sq:
                    e -= 0.5 * eri[p, q, q, p]
        H[j, j] += e

        # Single excitations: a^dagger_dst a_src
        for src in occ:
            ss, ps = divmod(src, norb)
            for dst in virt:
                sd, pd = divmod(dst, norb)
                if sd != ss:
                    continue
                r = apply_single(det, dst, src)
                if r is None:
                    continue
                m, sign = r
                i = index.get(m)
                if i is None:
                    continue
                val = h1[pd, ps]
                for o in occ:
                    if o == src:
                        continue
                    so, po = divmod(o, norb)
                    val += eri[pd, ps, po, po]
                    if sd == so:
                        val -= eri[pd, po, po, ps]
                H[i, j] += sign * val

        # Double excitations: a^dagger_dst1 a^dagger_dst2 a_src2 a_src1
        for i1 in range(len(occ)):
            for i2 in range(i1 + 1, len(occ)):
                src1, src2 = occ[i1], occ[i2]
                s1, p1 = divmod(src1, norb)
                s2, p2 = divmod(src2, norb)
                for a1 in range(len(virt)):
                    for a2 in range(a1 + 1, len(virt)):
                        dst1, dst2 = virt[a1], virt[a2]
                        t1, q1 = divmod(dst1, norb)
                        t2, q2 = divmod(dst2, norb)
                        r = annihilate(det, src1)
                        if r is None:
                            continue
                        m, sg = r
                        r = annihilate(m, src2)
                        if r is None:
                            continue
                        m, s = r
                        sg *= s
                        r = create(m, dst2)
                        if r is None:
                            continue
                        m, s = r
                        sg *= s
                        r = create(m, dst1)
                        if r is None:
                            continue
                        m, s = r
                        sg *= s
                        i = index.get(m)
                        if i is None:
                            continue
                        val = 0.0
                        if t1 == s1 and t2 == s2:
                            val += eri[q1, p1, q2, p2]
                        if t1 == s2 and t2 == s1:
                            val -= eri[q1, p2, q2, p1]
                        H[i, j] += sg * val

    return H


def apply_spin_free_excitation(vec: dict, dst: int, src: int, norb: int) -> dict:
    """Apply the spin-summed operator E_dst,src = sum_sigma a^dag_dst,s a_src,s.

    Parameters
    ----------
    vec : dict[int, float]
        Sparse vector over determinant bitmasks.
    dst, src : int
        Spatial orbital indices.
    norb : int
        Number of spatial orbitals.

    Returns
    -------
    dict[int, float]
    """
    out: dict[int, float] = {}
    for det, coeff in vec.items():
        for spin in (0, 1):
            r = apply_single(det, dst + spin * norb, src + spin * norb)
            if r is None:
                continue
            m, sign = r
            out[m] = out.get(m, 0.0) + sign * coeff
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


def apply_spin_free_excitation_batch(vec: dict, pairs, norb: int) -> dict:
    """Apply several E_dst,src operators in the order given (first is leftmost)."""
    cur = vec
    for dst, src in reversed(pairs):
        cur = apply_spin_free_excitation(cur, dst, src, norb)
    return cur


def s2_expectation(vec: np.ndarray, dets, norb: int) -> float:
    """<S^2> = Sz(Sz+1) + ||S+ psi||^2 for a normalized CI vector.

    S+ raises Ms by one, so its image lies outside the Ms sector spanned by
    ``dets``.  The norm is therefore accumulated over the image determinants
    directly instead of being indexed back into the basis.
    """
    sz = 0.0
    for k, det in enumerate(dets):
        na = (det & ((1 << norb) - 1)).bit_count()
        nb = (det >> norb).bit_count()
        sz += (na - nb) / 2.0 * abs(vec[k]) ** 2
    # S+ = sum_p a^dag_{p,alpha} a_{p,beta}
    splus: dict[int, float] = {}
    for k, det in enumerate(dets):
        c = vec[k]
        if c == 0.0:
            continue
        for p in range(norb):
            r = apply_single(det, p, p + norb)
            if r is None:
                continue
            m, sign = r
            splus[m] = splus.get(m, 0.0) + sign * c
    norm2 = sum(v * v for v in splus.values())
    return sz * (sz + 1.0) + float(norm2)


def dets_with_ms(norb: int, n_elec: int, ms: float = 0.0):
    """All determinants with n_elec electrons and the given Ms over 2*norb spin-orbitals."""
    two_ms = int(round(2 * ms))
    out = []
    nso = 2 * norb
    for occ in itertools.combinations(range(nso), n_elec):
        mask = 0
        for p in occ:
            mask |= 1 << p
        na = (mask & ((1 << norb) - 1)).bit_count()
        nb = (mask >> norb).bit_count()
        if na - nb == two_ms:
            out.append(mask)
    return out
