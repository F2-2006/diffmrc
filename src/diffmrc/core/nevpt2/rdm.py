#!/usr/bin/env python
"""Active-space RDMs for NEVPT2, spin-free and in the norm order of PySCF fci.rdm.

Second quantization is applied programmatically on a bitmap representation to
build the active-space one- and two-particle density matrices, following the
FCI3pdm_kern_sf spin-free convention:

    dm1[p,q]     = sum_sigma <a^dag_p,sigma a_q,sigma>
    dm2[p,q,r,s] = sum_{sigma,tau} <a^dag_p,sigma a^dag_q,tau a_s,tau a_r,sigma>

The numerical results are calibrated against the independent implementation.
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from .m4 import apply_seq  # bitmap utilities


def _apply_4(det, anns, cres):
    """Four-operator sequence a^dag_p a^dag_q a_s a_r; annihilations first, both ascending."""
    return apply_seq(det, anns, cres)


def rdm1_spin(c_ci, dets, ncas=2):
    """Spin-orbital one-particle RDM: gamma[p,q] = sum_IJ c_I c_J <D_I|a^dag_p a_q|D_J>."""
    nso = 2 * ncas
    rdm = jnp.zeros((nso, nso))
    for p in range(nso):
        for q in range(nso):
            m = jnp.zeros((len(dets), len(dets)))
            for I, det in enumerate(dets):
                for J, det2 in enumerate(dets):
                    dK, sig = apply_seq(det2, [q], [p])
                    m = m.at[I, J].set(sig if dK == det else 0.0)
            rdm = rdm.at[p, q].set(jnp.einsum("i,ij,j->", c_ci, m, c_ci))
    return rdm


def rdm2_spin(c_ci, dets, ncas=2):
    """Spin-orbital two-particle RDM: Gamma[p,q,r,s] = sum_IJ c_I c_J <D_I|a^dag_p a^dag_q a_s a_r|D_J>."""
    nso = 2 * ncas
    rdm = jnp.zeros((nso, nso, nso, nso))
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    m = jnp.zeros((len(dets), len(dets)))
                    for I, det in enumerate(dets):
                        for J, det2 in enumerate(dets):
                            # <D_I| a^dag_p a^dag_q a_s a_r |D_J>: annihilate r,s then create q,p
                            dK, sig = apply_seq(det2, [s, r], [q, p])
                            m = m.at[I, J].set(sig if dK == det else 0.0)
                    rdm = rdm.at[p, q, r, s].set(jnp.einsum("i,ij,j->", c_ci, m, c_ci))
    return rdm


def make_dm123_sf(c_ci, dets, ncas=2, ncore=0):
    """Return dm1, dm2, dm3 in density order, following the FCI3pdm_kern_sf convention.

    With ncore > 0 the first 2*ncore spin orbitals are a fully occupied core and
    the active space spans MOs ncore through ncore + ncas - 1.

    Vectorization: a double loop over determinant pairs and spin-orbital pairs
    would perform about 1.2e8 indexed updates for CAS(6,6) with 924
    determinants, which exhausts memory. The construction is instead a single
    four-dimensional mask,

        M[p,q,I,J] = <D_I|a^dag_p a_q|D_J>

    built from single-excitation masks and fermionic phases, as in the CI
    solver.
    """
    nso = 2 * (ncore + ncas)
    ndet = len(dets)
    # ---- Occupation bitmap. The determinants are integer bitmaps, so they
    #      are decomposed by bitwise AND rather than used as indices. ----
    det_arr = jnp.asarray(dets, dtype=int)[:, None]      # (ndet, 1)
    occ_bit = (det_arr & (1 << jnp.arange(nso))[None, :]) != 0  # (ndet, nso) bool
    # ---- Single-excitation matching: a^dag_p a_q with q in D_J, p not in D_J,
    #      and D_I = D_J - {q} + {p} ----
    XA = occ_bit[:, None, :] & ~occ_bit[None, :, :]  # diff_a[I,J] = D_I \ D_J
    XB = ~occ_bit[:, None, :] & occ_bit[None, :, :]  # diff_b[I,J] = D_J \ D_I
    nA, nB = XA.sum(axis=2), XB.sum(axis=2)
    mask1 = (nA == 1) & (nB == 1)                    # (ndet, ndet)
    # Phase: occupied positions, as in the CI solver (cumsum minus one).
    pos = jnp.cumsum(occ_bit, axis=1) - 1
    i1 = jnp.argmax(XA, axis=2)                      # p: spin orbital unique to D_I
    k1 = jnp.argmax(XB, axis=2)                      # q: spin orbital unique to D_J
    pos_p = jnp.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_q = jnp.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = jnp.where((pos_p + pos_q) % 2 == 0, 1.0, -1.0)  # (ndet, ndet)
    # ---- Constructing M[sp,sq] for all pairs at once costs about 1.7 GB, and
    #      in an eager loop several copies accumulate, which exhausted memory
    #      at 150 GB for 924 determinants. Constructing and caching one
    #      (ndet, ndet) block per spin-orbital pair uses 144 pairs times
    #      0.85 MB, about 122 MB, for CAS(6,6). ----
    so = jnp.arange(nso)
    eye_IJ = jnp.eye(ndet)
    M = {}
    for sp in range(nso):
        for sq in range(nso):
            m = mask1 & (i1 == sp) & (k1 == sq)     # (ndet,ndet) single-excitation mask
            mf = jnp.where(m, sign1, 0.0)           # single-excitation term with phase
            if sp == sq:                            # number-operator diagonal
                mf = mf + jnp.where(eye_IJ, occ_bit[:, sp], 0.0)
            M[(sp, sq)] = mf
    def slice_M(sp, sq):
        return M[(sp, sq)]
    # ---- dm1, with the spin sum ----
    dm1 = jnp.zeros((ncas, ncas))
    for p in range(ncas):
        for q in range(ncas):
            dm1 = dm1.at[p, q].set(
                jnp.einsum("i,ij,j->", c_ci, slice_M(2 * (ncore + p), 2 * (ncore + q)), c_ci)
                + jnp.einsum("i,ij,j->", c_ci, slice_M(2 * (ncore + p) + 1, 2 * (ncore + q) + 1), c_ci))
    # ---- dm2 and dm3 in density order; the contraction formulas are as usual ----
    dm2 = jnp.zeros((ncas, ncas, ncas, ncas))
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    val = jnp.zeros(())
                    for sig in (0, 1):
                        for tau in (0, 1):
                            Mp = slice_M(2 * (ncore + p) + sig, 2 * (ncore + q) + sig)
                            Mr = slice_M(2 * (ncore + r) + tau, 2 * (ncore + s) + tau)
                            val = val + jnp.einsum("i,ik,kj,j->", c_ci, Mp, Mr, c_ci)
                    dm2 = dm2.at[p, q, r, s].set(val)
    dm3 = jnp.zeros((ncas,) * 6)
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    for t in range(ncas):
                        for u in range(ncas):
                            val = jnp.zeros(())
                            for sig in (0, 1):
                                for tau in (0, 1):
                                    for ups in (0, 1):
                                        Mp = slice_M(2 * (ncore + p) + sig, 2 * (ncore + q) + sig)
                                        Mr = slice_M(2 * (ncore + r) + tau, 2 * (ncore + s) + tau)
                                        Mt = slice_M(2 * (ncore + t) + ups, 2 * (ncore + u) + ups)
                                        val = val + jnp.einsum("i,ik,kl,lj,j->",
                                                               c_ci, Mp, Mr, Mt, c_ci)
                            dm3 = dm3.at[p, q, r, s, t, u].set(val)
    return dm1, dm2, dm3
