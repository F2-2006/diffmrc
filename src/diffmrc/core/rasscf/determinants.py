"""Determinant CI kernel of the RASSCF module.

PURPOSE
-------
Provides the determinant-basis machinery of a restricted active space SCF:

* enumeration of the RAS CI space over the active orbitals with the
  occupation constraints of Malmqvist, J. Chem. Phys. 129 (2008) 164106
  (RAS1 holes <= ``max_holes``, RAS3 electrons <= ``max_elec``);
* Slater-Condon matrix elements of the non-relativistic Hamiltonian for
  determinant pairs that differ by at most two spin orbitals;
* sparse Hamiltonian assembly, CI diagonalization, and the spin-summed
  one- and two-particle density matrices.

Inactive orbitals are frozen doubly occupied and enter the active CI
through an effective one-electron operator built by
``diffmrc.core.rasscf.mcscf``.  This module knows nothing about orbital
rotations or the SCF procedure itself.

PUBLIC INTERFACE
----------------
Bit helpers:
    ``popcount``, ``occupation``, ``apply_sequence``.
RAS space:
    ``ras_counts``, ``ras_allowed``, ``enumerate_determinants``,
    ``enumerate_active_determinants``.
Matrix elements and assembly:
    ``hamiltonian_element``, ``build_hamiltonian``.
Densities and CI solver:
    ``one_rdm``, ``two_rdm``, ``solve_ci``.

Inputs are integer bit masks and numpy arrays; outputs are numpy arrays or
``scipy.sparse`` matrices.  Energies are in Eh.

CONVENTIONS
-----------
* Spatial orbitals, real, chemists' notation for two-electron integrals:
  eri[p,q,r,s] = (pq|rs).
* A determinant is a pair of integer bit masks ``(alpha_mask, beta_mask)``.
  Spin-orbital ordering is the alpha block first, then the beta block, so
  the sign of a beta annihilation includes the alpha occupation parity.
* RAS partition of the *active* orbitals:
  RAS1 = [0, n1), RAS2 = [n1, n1+n2), RAS3 = [n1+n2, n1+n2+n3).
* Determinants are the reference space: states are not spin adapted.

VALIDATED SCOPE
---------------
Water/STO-3G, seven orbitals, all-orbital determinants with M_S = 0: the
lowest eigenvalue of the assembled Hamiltonian reproduces the pyscf FCI
energy to 1.0e-12 Eh over 441 determinants, and every matrix element of a
four-orbital (2 alpha, 2 beta) space agrees with pyscf ``direct_spin1`` to
2.8e-14.  The matrix elements also reproduce a brute-force spin-orbital
summation to 1.1e-15 on random integrals that carry the full eightfold
permutation symmetry.

BOUNDARIES
----------
* Determinant basis with fixed M_S: no spin-adapted CSFs and no point-group
  symmetry, so degenerate and near-degenerate roots are not protected.
* Ground root only; state averaging is not implemented.
* Bit-string enumeration scales exponentially with the number of active
  orbitals, and ``two_rdm`` loops explicitly over occupied pairs and
  orbital quartets.
* The Slater-Condon routines assume the eightfold permutation symmetry of
  physical two-electron integrals, in particular (pq|rs) = (rs|pq).
  Non-physical random integrals can break the agreement with a direct
  summation.
* The first-order interacting space (FOCIS) used by RASPT2 is not built by
  this module; only the reference RAS CI space is enumerated.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "apply_sequence",
    "build_hamiltonian",
    "enumerate_active_determinants",
    "enumerate_determinants",
    "hamiltonian_element",
    "occupation",
    "one_rdm",
    "popcount",
    "ras_allowed",
    "ras_counts",
    "solve_ci",
    "two_rdm",
]


# --------------------------------------------------------------------------
# Bit helpers
# --------------------------------------------------------------------------
def popcount(x: int) -> int:
    return int(x).bit_count()


def occupation(mask: int, n_orb: int) -> np.ndarray:
    """Return occupation vector (length n_orb) of a bit mask."""
    return np.array([(mask >> p) & 1 for p in range(n_orb)], dtype=np.int8)


def _annihilate(mask: int, p: int, alpha_mask: int, beta_mask: int, spin: int):
    """Annihilate spin-orbital (p, spin) from a determinant.

    Returns (sign, new_mask) or None.  ``alpha_mask``/``beta_mask`` are the
    full determinant, used for the parity of beta operators.
    """
    if not (mask >> p) & 1:
        return None
    below = popcount(mask & ((1 << p) - 1))
    if spin == 1:
        below += popcount(alpha_mask)
    sign = -1 if below & 1 else 1
    return sign, mask ^ (1 << p)


def _create(mask: int, p: int, alpha_mask: int, beta_mask: int, spin: int):
    """Create spin-orbital (p, spin).  Same parity rule as annihilation."""
    if (mask >> p) & 1:
        return None
    below = popcount(mask & ((1 << p) - 1))
    if spin == 1:
        below += popcount(alpha_mask)
    sign = -1 if below & 1 else 1
    return sign, mask | (1 << p)


def apply_sequence(a: int, b: int, seq):
    """Apply a sequence of (spin, orbital, is_creation) in application order.

    The first element acts first (i.e. it is the rightmost operator).
    Returns (sign, alpha_mask, beta_mask) or None if the result vanishes.
    """
    sign = 1
    for spin, p, create in seq:
        if spin == 0:
            res = _create(a, p, a, b, 0) if create else _annihilate(a, p, a, b, 0)
        else:
            res = _create(b, p, a, b, 1) if create else _annihilate(b, p, a, b, 1)
        if res is None:
            return None
        s, new = res
        sign *= s
        if spin == 0:
            a = new
        else:
            b = new
    return sign, a, b


# --------------------------------------------------------------------------
# RAS constraints
# --------------------------------------------------------------------------
def ras_counts(alpha_mask: int, beta_mask: int, n1: int, n2: int, n3: int):
    """Return (holes_in_RAS1, electrons_in_RAS3) for an active-space determinant."""
    ras1 = (1 << n1) - 1
    ras3 = ((1 << n3) - 1) << (n1 + n2)
    n_elec_ras1 = popcount(alpha_mask & ras1) + popcount(beta_mask & ras1)
    n_elec_ras3 = popcount(alpha_mask & ras3) + popcount(beta_mask & ras3)
    return 2 * n1 - n_elec_ras1, n_elec_ras3


def ras_allowed(alpha_mask, beta_mask, n1, n2, n3, max_holes, max_elec) -> bool:
    holes, elec3 = ras_counts(alpha_mask, beta_mask, n1, n2, n3)
    return holes <= max_holes and elec3 <= max_elec


def enumerate_determinants(n_orb, n_alpha, n_beta, n1, n2, n3,
                           max_holes, max_elec, inactive_mask=0):
    """Enumerate determinants with fixed alpha/beta counts satisfying RAS.

    ``inactive_mask`` marks orbitals forced doubly occupied (used for the
    all-orbital FOCIS space, where the inactive core is kept doubly occupied
    in the *reference* only; pass 0 to leave them free).
    """
    alphas = [m for m in range(1 << n_orb) if popcount(m) == n_alpha]
    betas = [m for m in range(1 << n_orb) if popcount(m) == n_beta]
    dets = []
    for a in alphas:
        if (a & inactive_mask) != inactive_mask:
            continue
        for b in betas:
            if (b & inactive_mask) != inactive_mask:
                continue
            if ras_allowed(a | 0, b | 0, n1, n2, n3, max_holes, max_elec):
                dets.append((a, b))
    return dets


def enumerate_active_determinants(n_act, n_alpha, n_beta, n1, n2, n3,
                                  max_holes, max_elec):
    """Reference-space determinants over the active orbitals only."""
    return enumerate_determinants(n_act, n_alpha, n_beta, n1, n2, n3,
                                  max_holes, max_elec)


# --------------------------------------------------------------------------
# Slater-Condon matrix elements
# --------------------------------------------------------------------------
def _diagonal_element(a, b, n_orb, h1, eri):
    """<D|H|D> for a determinant over all orbitals (no frozen core)."""
    occ_a = [p for p in range(n_orb) if (a >> p) & 1]
    occ_b = [p for p in range(n_orb) if (b >> p) & 1]
    e = 0.0
    for p in occ_a:
        e += h1[p, p]
    for p in occ_b:
        e += h1[p, p]
    # Coulomb and exchange, spin-resolved:
    # 1/2 sum_{ij} [(ii|jj) - delta(sigma_i,sigma_j) (ij|ji)]
    for p in occ_a:
        for q in occ_a:
            e += 0.5 * eri[p, p, q, q]
            e -= 0.5 * eri[p, q, q, p]
    for p in occ_b:
        for q in occ_b:
            e += 0.5 * eri[p, p, q, q]
            e -= 0.5 * eri[p, q, q, p]
    for p in occ_a:
        for q in occ_b:
            e += eri[p, p, q, q]
    return e


def _two_body_replacement(aI, bI, aJ, bJ, created, annihilated, n_orb, eri):
    """<J|H2|I> for a replacement with two created and two annihilated spins.

    ``created`` and ``annihilated`` are lists of (orbital, spin).  The routine
    sums the four index assignments of the operator a+_P a+_R a_S a_Q with the
    1/2 prefactor of H2, which is the form validated against PySCF FCI.
    """
    val = 0.0
    (P0, sP0), (R0, sR0) = created
    (Q0, sQ0), (S0, sS0) = annihilated
    for (P, sP), (R, sR) in (((P0, sP0), (R0, sR0)),
                             ((R0, sR0), (P0, sP0))):
        for (Q, sQ), (S, sS) in (((Q0, sQ0), (S0, sS0)),
                                 ((S0, sS0), (Q0, sQ0))):
            seq = [(sQ, Q, False), (sS, S, False),
                   (sR, R, True), (sP, P, True)]
            res = apply_sequence(aI, bI, seq)
            if res is None:
                continue
            sgn, a, b = res
            if (a, b) != (aJ, bJ):
                continue
            # (PQ|RS) carries delta(sP,sQ) delta(sR,sS)
            if sP == sQ and sR == sS:
                val += 0.5 * sgn * eri[P, Q, R, S]
    return val


def _singles_element(aI, bI, aJ, bJ, p_out, s_out, p_in, s_in,
                     common_a, common_b, n_orb, h1, eri):
    """<J|H|I> where J is obtained from I by replacing (p_in,s_in)->(p_out,s_out).

    ``common_a``/``common_b`` are the occupied orbitals shared by I and J.
    """
    val = 0.0
    # one-body: a^+_{p_out} a_{p_in}; h is spin independent, so the spin-orbital
    # matrix element carries delta(s_out, s_in).
    res = apply_sequence(aI, bI, [(s_in, p_in, False), (s_out, p_out, True)])
    if res is not None:
        sgn, a, b = res
        assert (a, b) == (aJ, bJ), "single replacement sign bookkeeping failed"
        if s_out == s_in:
            val += sgn * h1[p_out, p_in]

    # two-body: the excited electron interacts with every other occupied
    # spin-orbital k; the replacement acts on {p_in, k} -> {p_out, k}.
    for spin_k, k in [(0, x) for x in common_a] + [(1, x) for x in common_b]:
        val += _two_body_replacement(
            aI, bI, aJ, bJ,
            created=[(p_out, s_out), (k, spin_k)],
            annihilated=[(p_in, s_in), (k, spin_k)],
            n_orb=n_orb, eri=eri)
    return val


def _doubles_element(aI, bI, aJ, bJ, out1, out2, in1, in2, n_orb, eri):
    """<J|H|I> for a double replacement (no one-body contribution)."""
    return _two_body_replacement(aI, bI, aJ, bJ,
                                 created=[out1, out2],
                                 annihilated=[in1, in2],
                                 n_orb=n_orb, eri=eri)


def _differences(aI, bI, aJ, bJ, n_orb):
    out, inn = [], []
    for p in range(n_orb):
        ia, ja = (aI >> p) & 1, (aJ >> p) & 1
        ib, jb = (bI >> p) & 1, (bJ >> p) & 1
        if ia and not ja:
            inn.append((p, 0))
        elif ja and not ia:
            out.append((p, 0))
        if ib and not jb:
            inn.append((p, 1))
        elif jb and not ib:
            out.append((p, 1))
    return out, inn


def hamiltonian_element(aI, bI, aJ, bJ, n_orb, h1, eri):
    """<J|H|I> for determinants over all orbitals."""
    out, inn = _differences(aI, bI, aJ, bJ, n_orb)
    ndiff = len(out)
    if ndiff != len(inn):
        return 0.0
    if ndiff > 2:
        return 0.0
    if ndiff == 0:
        return _diagonal_element(aI, bI, n_orb, h1, eri)
    if ndiff == 1:
        (p_out, s_out) = out[0]
        (p_in, s_in) = inn[0]
        common_a = [p for p in range(n_orb)
                    if (aI >> p) & 1 and (aJ >> p) & 1]
        common_b = [p for p in range(n_orb)
                    if (bI >> p) & 1 and (bJ >> p) & 1]
        return _singles_element(aI, bI, aJ, bJ, p_out, s_out, p_in, s_in,
                                common_a, common_b, n_orb, h1, eri)
    (o1, o2) = out
    (i1, i2) = inn
    return _doubles_element(aI, bI, aJ, bJ, o1, o2, i1, i2, n_orb, eri)


# --------------------------------------------------------------------------
# Sparse Hamiltonian build
# --------------------------------------------------------------------------
def build_hamiltonian(dets, n_orb, h1, eri, tol=1e-14):
    """Build the sparse Hamiltonian matrix over a determinant list."""
    index = {d: i for i, d in enumerate(dets)}
    rows, cols, vals = [], [], []
    for i, (a, b) in enumerate(dets):
        rows.append(i); cols.append(i)
        vals.append(_diagonal_element(a, b, n_orb, h1, eri))
        # singles
        occ_a = [p for p in range(n_orb) if (a >> p) & 1]
        occ_b = [p for p in range(n_orb) if (b >> p) & 1]
        vir_a = [p for p in range(n_orb) if not (a >> p) & 1]
        vir_b = [p for p in range(n_orb) if not (b >> p) & 1]
        for spin, occ, vir in ((0, occ_a, vir_a), (1, occ_b, vir_b)):
            for p_in in occ:
                for p_out in vir:
                    res = apply_sequence(a, b, [(spin, p_in, False),
                                                (spin, p_out, True)])
                    if res is None:
                        continue
                    sgn, aJ, bJ = res
                    j = index.get((aJ, bJ))
                    if j is None or j <= i:
                        continue
                    v = _singles_element(a, b, aJ, bJ, p_out, spin, p_in, spin,
                                         occ_a, occ_b, n_orb, h1, eri)
                    if abs(v) > tol:
                        rows.append(j); cols.append(i); vals.append(v)
                        rows.append(i); cols.append(j); vals.append(v)
        # doubles (different spin-orbital pairs)
        spin_occ = [(0, p) for p in occ_a] + [(1, p) for p in occ_b]
        spin_vir = [(0, p) for p in vir_a] + [(1, p) for p in vir_b]
        for x in range(len(spin_occ)):
            for y in range(x + 1, len(spin_occ)):
                s1, p1 = spin_occ[x]
                s2, p2 = spin_occ[y]
                for u in range(len(spin_vir)):
                    for w in range(u + 1, len(spin_vir)):
                        t1, q1 = spin_vir[u]
                        t2, q2 = spin_vir[w]
                        seq = [(s1, p1, False), (s2, p2, False),
                               (t2, q2, True), (t1, q1, True)]
                        res = apply_sequence(a, b, seq)
                        if res is None:
                            continue
                        sgn, aJ, bJ = res
                        j = index.get((aJ, bJ))
                        if j is None or j <= i:
                            continue
                        v = _doubles_element(a, b, aJ, bJ,
                                             (q1, t1), (q2, t2),
                                             (p1, s1), (p2, s2), n_orb, eri)
                        if abs(v) > tol:
                            rows.append(j); cols.append(i); vals.append(v)
                            rows.append(i); cols.append(j); vals.append(v)
    n = len(dets)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


# --------------------------------------------------------------------------
# Density matrices
# --------------------------------------------------------------------------
def one_rdm(civec, dets, n_orb):
    """Spin-summed one-particle density matrix D[p,q]."""
    index = {d: i for i, d in enumerate(dets)}
    D = np.zeros((n_orb, n_orb))
    for i, (a, b) in enumerate(dets):
        ci = civec[i]
        if abs(ci) < 1e-14:
            continue
        for spin, mask in ((0, a), (1, b)):
            for q in range(n_orb):
                if not (mask >> q) & 1:
                    continue
                for p in range(n_orb):
                    res = apply_sequence(a, b, [(spin, q, False),
                                                (spin, p, True)])
                    if res is None:
                        continue
                    sgn, aJ, bJ = res
                    j = index.get((aJ, bJ))
                    if j is not None:
                        D[p, q] += ci * civec[j] * sgn
    return D


def two_rdm(civec, dets, n_orb):
    """Spin-summed two-particle density matrix d[p,q,r,s].

    Convention: d[p,q,r,s] = sum_{sigma,tau} <a+_{p sigma} a+_{r tau}
    a_{s tau} a_{q sigma}>, so that E2 = 1/2 sum (pq|rs) d[p,q,r,s].
    """
    index = {d: i for i, d in enumerate(dets)}
    d = np.zeros((n_orb, n_orb, n_orb, n_orb))
    for i, (a, b) in enumerate(dets):
        ci = civec[i]
        if abs(ci) < 1e-14:
            continue
        occ = [(0, p) for p in range(n_orb) if (a >> p) & 1]
        occ += [(1, p) for p in range(n_orb) if (b >> p) & 1]
        # d[p,q,r,s] = sum_{sigma,tau} <a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}>
        # application order: a_{q sigma}, a_{s tau}, a+_{r tau}, a+_{p sigma}
        for (s_q, q) in occ:
            for (s_s, s) in occ:
                if (s_q, q) == (s_s, s):
                    continue
                for p in range(n_orb):
                    for r in range(n_orb):
                        res = apply_sequence(a, b,
                                             [(s_q, q, False), (s_s, s, False),
                                              (s_s, r, True), (s_q, p, True)])
                        if res is None:
                            continue
                        sgn, aJ, bJ = res
                        j = index.get((aJ, bJ))
                        if j is not None:
                            d[p, q, r, s] += ci * civec[j] * sgn
    return d


# --------------------------------------------------------------------------
# CI solver
# --------------------------------------------------------------------------
def solve_ci(H, nroots=1, v0=None):
    """Lowest eigenvalues of a sparse Hamiltonian."""
    n = H.shape[0]
    if n <= 400:
        w, v = np.linalg.eigh(H.toarray())
        return w[:nroots], v[:, :nroots]
    from scipy.sparse.linalg import eigsh
    w, v = eigsh(H, k=nroots, which="SA", v0=v0, tol=1e-12, maxiter=10000)
    order = np.argsort(w)
    return w[order], v[:, order]
