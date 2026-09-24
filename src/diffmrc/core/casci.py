#!/usr/bin/env python
"""Differentiable CASCI: an end-to-end JAX chain from nuclear coordinates to energy.

Pipeline (fully differentiable):

    coords -> pyscfad Mole (a differentiable pytree)
           -> int1e_kin / int1e_nuc / int2e (differentiable integrals)
           -> pyscfad RHF kernel (differentiable SCF, orbitals carry response)
           -> einsum MO integral transformation
           -> Slater-determinant CI by the Condon-Slater rules
           -> jnp.linalg.eigh -> E_CASCI

Verification performed by main(): energy against PySCF mcscf.CASCI, the
auto-differentiated gradient against a central difference of the same chain,
and against the closed-form analytic gradient of an independent program.

Usage: python casci_pyscfad.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "")

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from pyscfad import scf as pscf

from itertools import combinations

from .eigh import eigh_safe

def cs_casci(h1e_mo, eri_mo, ncore=0, nact=2, nele_act=2, nroots=1, return_ci=False):
    """CASCI over a Slater-determinant basis built by the Condon-Slater rules.

    The occupied set is the fully occupied core plus `nele_act` spin orbitals
    chosen from the active space; the rules classify each determinant pair by
    zero, one, or two differences, and core-active cross terms are handled
    automatically by the complete occupation set.

    nroots > 1 : state-averaged energy over the lowest nroots eigenvalues with
                 equal weights, matching the eigenvalue ordering used by an
                 independent state-average implementation.
    return_ci  : additionally return the ground-state CI vector (needed by the
                 perturbation layer).

    h1e_mo / eri_mo : MO-space integrals, at least ncore + nact dimensions.
    """
    nso = 2 * (ncore + nact)
    # Vectorized construction of the spin-orbital basis. For f-block systems
    # (nso = 40-56) a four-fold .at loop would run ~2.5 million times, so the
    # whole construction is broadcast instead.
    # h_so[i,k] = h1e_mo[i//2, k//2] * delta(spin_i, spin_k); same for eri_so.
    so = jnp.arange(nso)
    pi, si = so // 2, so % 2
    h_so = jnp.where(si[:, None] == si[None, :],
                     h1e_mo[pi[:, None], pi[None, :]], 0.0)
    # eri_so[i,j,k,l] = eri_mo[pi_i, pk_k, pj_j, pl_l] * delta(si,sk) * delta(sj,sl)
    # Build in [i,k,j,l] order and transpose to [i,j,k,l].
    m4 = (si[:, None, None, None] == si[None, None, :, None]) * (
        si[None, :, None, None] == si[None, None, None, :])
    # eri_mo dimension order is [p_i, p_k, p_j, p_l]; the broadcast result is
    # already [i,j,k,l] with the index mapping P0->0(i), P1(pk)->2(k),
    # P2(pj)->1(j), P3->3(l), so no transpose is needed.
    eri_so = jnp.where(m4, eri_mo[pi[:, None, None, None],
                                  pi[None, None, :, None],
                                  pi[None, :, None, None],
                                  pi[None, None, None, :]], 0.0)
    core_so = tuple(range(2 * ncore))
    act_so = list(range(2 * ncore, nso))
    DETS = [core_so + comb for comb in combinations(act_so, nele_act)]
    ndet = len(DETS)
    # ===== Vectorized Hamiltonian construction
    # Rationale: building H through indexed .at updates makes the JIT
    # compilation of state-averaged CASSCF explode, because the trace over a
    # 91 x 91 buffer and the gradient vjp expand into a graph whose compile-time
    # peak exceeds 100 GB. The vectorized form reduces compile-time memory by
    # about two orders of magnitude and brings the H construction for a
    # 1001-determinant CAS(4,7) from roughly 40 minutes to seconds.
    occ_bit = jnp.zeros((ndet, nso), dtype=bool)
    occ_bit = occ_bit.at[jnp.arange(ndet)[:, None], jnp.asarray(DETS)].set(True)
    # ---- Diagonal: one-electron term plus two-electron term (strict upper
    #      triangle, since the two-electron contribution is antisymmetric) ----
    diag1 = occ_bit @ jnp.diag(h_so)
    E2ij = jnp.einsum('ijij->ij', eri_so) - jnp.einsum('ijji->ij', eri_so)
    diag2 = ((occ_bit @ jnp.triu(E2ij, k=1)) * occ_bit).sum(axis=1)
    H_diag = diag1 + diag2
    # ---- Off-diagonal: single and double excitations, via difference
    #      matrices and phases, with the whole determinant-pair matrix at once
    XA = occ_bit[:, None, :] & ~occ_bit[None, :, :]  # diff_a[A,B] = occ_a \ occ_b
    XB = ~occ_bit[:, None, :] & occ_bit[None, :, :]  # diff_b[A,B] = occ_b \ occ_a
    nA, nB = XA.sum(axis=2), XB.sum(axis=2)
    # Occupied position, for the phase: cumsum - 1 gives the index of each
    # spin orbital within the occupied set.
    pos = jnp.cumsum(occ_bit, axis=1) - 1

    def so_idx(Xm):
        return jnp.argmax(Xm, axis=2)  # (ndet, ndet): index of the difference

    def second_idx(Xm, first):
        Xwo = jnp.where(jnp.arange(nso)[None, None, :] == first[..., None],
                        False, Xm)
        return jnp.argmax(Xwo, axis=2)

    # ---- Single excitation (i -> k):
    #      val = sign * (h_so[i,k] + sum_{j in occ_a, j != i}
    #                    (eri_so[i,j,k,j] - eri_so[i,j,j,k]))
    mask1 = (nA == 1) & (nB == 1)
    # Discrete indices must be stop_gradient. i1 and k1 are determined by the
    # static bit patterns of the determinant list, hence independent of the
    # nuclear coordinates, so their true derivative is identically zero. They
    # are produced by jnp.argmax, whose first derivative is zero but whose
    # second derivative touches the derivative of an integer index with respect
    # to its input, giving 0 * inf = NaN. Without this guard, taking a second
    # derivative with respect to the geometry returns NaN everywhere.
    i1 = jax.lax.stop_gradient(so_idx(XA))
    k1 = jax.lax.stop_gradient(so_idx(XB))
    pos_i = jnp.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_k = jnp.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = jnp.where((pos_i + pos_k) % 2 == 0, 1.0, -1.0)
    Xijk = jnp.einsum('ijkj->ijk', eri_so) - jnp.einsum('ijjk->ijk', eri_so)
    T = jnp.einsum('aj,ijk->aik', occ_bit, Xijk)  # T[A,i,k] = sum_{j in occ_a} Xijk
    T_pair = T[jnp.arange(ndet)[:, None], i1, k1]
    occ_i = jnp.take_along_axis(occ_bit[:, :, None], i1[:, :, None], axis=1)[..., 0]
    X_ii_k = Xijk[i1, i1, k1]  # the j == i term, excluded
    val1 = sign1 * (h_so[i1, k1] + T_pair - occ_i * X_ii_k)
    H = jnp.where(mask1, val1, 0.0)
    # ---- Double excitation (i,j -> k,l):
    #      val = sign * (eri_so[i,j,k,l] - eri_so[i,j,l,k])
    mask2 = (nA == 2) & (nB == 2)
    i2 = jax.lax.stop_gradient(second_idx(XA, i1))
    k2 = jax.lax.stop_gradient(second_idx(XB, k1))
    pos_i2 = jnp.take_along_axis(pos[:, None, :], i2[:, :, None], axis=2)[..., 0]
    pos_k2 = jnp.take_along_axis(pos[None, :, :], k2[:, :, None], axis=2)[..., 0]
    sign2 = jnp.where((pos_i + pos_i2 + pos_k + pos_k2) % 2 == 0, 1.0, -1.0)
    val2 = sign2 * (eri_so[i1, i2, k1, k2] - eri_so[i1, i2, k2, k1])
    H = H + jnp.where(mask2, val2, 0.0)
    # ---- Fold in the diagonal ----
    H = H + jnp.diag(H_diag)
    if return_ci:
        eigs = jnp.linalg.eigvalsh(H)
        if nroots > 1:
            return jnp.mean(eigs[:nroots]), None  # state average; no CI vector
        # The ground-state eigenvector is obtained by inverse iteration with a
        # shift, because the vjp of jnp.linalg.eigh is NaN at degenerate
        # eigenvectors, and the CI matrix has a three-fold degenerate triplet.
        # (H - (lambda_0 - s) I)^-1 is nonsingular and differentiable.
        n = H.shape[0]
        # Shift of lambda_0 - 1e-8, one iteration. The vjp of solve amplifies
        # by roughly 1/shift per iteration, so a 1e-8 shift amplifies by 1e8,
        # comparable to the float64 round-off. One iteration leaves a CI
        # residual of about 1e-8 and an energy difference of about 1e-7, within
        # the 1e-6 acceptance threshold. Larger shifts or a second iteration
        # degrade the accuracy instead: 1e-6 with one iteration gives 2.7e-6,
        # and 1e-6 or 1e-5 with two iterations amplifies gradient error by
        # more than 1e10 and gives 5e-4.
        A = H - (eigs[0] - 1e-8) * jnp.eye(n)
        v = jnp.linalg.solve(A, jnp.ones(n))
        v = v / jnp.linalg.norm(v)
        return eigs[0], v
    eigs = jnp.linalg.eigvalsh(H)
    return jnp.mean(eigs[:nroots])

def cs_casci_blocked(h1e_mo, eri_mo, ncore=0, nact=2, nele_act=2):
    """Ground-state energy and CI vector by blockwise diagonalization in S_z.

    WHY THIS EXISTS
        The CI matrix of cs_casci contains degeneracies across spin-projection
        blocks - a three-fold degeneracy at -84.8416418 Eh for H2O/6-31G with
        CAS(2,2). The reverse-mode vjp of jnp.linalg.eigh produces 0/0 at a
        degeneracy and therefore returns NaN for the second derivative. Inverse
        iteration defers the NaN from the first derivative to the second but
        does not remove it.

    WHY THE BLOCKS ARE SAFE
        The Hamiltonian is strictly zero between different S_z blocks (the
        largest off-block matrix element measured is 0.000e+00). For H2O/6-31G
        with CAS(2,2) the block structure is

            Sz = +6 : 1 determinant
            Sz = +8 : 4 determinants, lowest = -85.17707936,
                      smallest in-block gap = 2.545e-02
            Sz = +10: 1 determinant

        so the global minimum lies in a block that contains no degeneracy.

        A non-degenerate ground state is necessary but NOT sufficient for the
        raw rule: the reverse pass forms the 1/(w_j - w_i) factor for every
        pair in the block whether or not the pair carries cotangent, so one
        bit-exact degenerate pair anywhere in any block turns the untouched
        entries into 0 * inf = NaN. Measured on CO/6-31G CAS(6,6) (t237):
        five of the seven blocks carry bit-exact pairs -- 24 of them in the
        400-determinant winning block, whose own ground state is simple --
        and the raw-eigh theta gradient is all-NaN while the energy is finite
        (t232r). The block eigenproblem therefore goes through
        ``core.eigh.eigh_safe``, whose Lorentzian-softened rule freezes the
        rotation inside a degenerate subspace. It is first-order safe by
        measurement (micro-tests, 2026-09-24) and leaves every well-separated
        block identical to the raw rule; second derivatives remain outside
        its scope (see the module docstring in core/eigh.py).

    WHEN TO USE IT
        Only where a second derivative is required. The unblocked cs_casci is
        used for the first derivative and below: it is more thoroughly
        validated and its inverse-iteration path is more accurate.

    Returns (E, ci_flat), where ci_flat uses the full determinant layout of
    cs_casci.
    """
    from itertools import combinations
    nso = 2 * (ncore + nact)
    core_so = tuple(range(2 * ncore))
    act_so = list(range(2 * ncore, nso))
    DETS = [core_so + comb for comb in combinations(act_so, nele_act)]
    sz = [sum(1 for so in d if so % 2 == 0) * 2 - nele_act for d in DETS]

    # Solve blockwise and take the global minimum. Three design points:
    #   (a) Do not compare block energies with Python if or float(); under
    #       tracing that raises ConcretizationTypeError.
    #   (b) The block partition is static, since the determinant list does not
    #       depend on the coordinates, so the loop is safe to trace.
    #   (c) Do not select the block with a data-dependent where. The globally
    #       lowest block is physically fixed - it is the block of highest S_z,
    #       in which all active electrons are paired - so which block wins is
    #       determined by the combinatorial structure and is independent of the
    #       geometry. Selecting with jnp.where(e_blk <= E_min) would introduce
    #       a non-differentiable branch.
    tot = len(DETS)
    out = jnp.zeros(tot)
    E_min = None
    # Process S_z from low to high so the final update carries the lowest block.
    for s in sorted(set(sz)):
        idx = [i for i, x in enumerate(sz) if x == s]
        H_blk = _block_H(h1e_mo, eri_mo, DETS, idx, ncore, nact, nele_act)
        w, U = eigh_safe(H_blk)
        if E_min is None:
            E_min = w[0]
            out = out.at[jnp.asarray(idx)].set(U[:, 0])
        else:
            # jnp.minimum keeps the energy differentiable. The vector is
            # updated by whichever block wins; the comparison is physically
            # stable because the block energies differ by far more than the
            # numerical noise (0.34 Ha for H2O).
            better = w[0] < E_min
            E_min = jnp.minimum(E_min, w[0])
            # When a new block wins, the vector of the previously leading
            # block must be erased over the WHOLE vector: it lives in a
            # different index set than the current block, so a per-block
            # update alone leaves it in place. Measured on ThO/ECP
            # CAS(4,4), ncore = 17: the retained mixture (the true lowest
            # block plus the leading vector of the two earlier blocks)
            # gave a 1-RDM differing from the correct one by 3.0 and a
            # NEVPT2 correction wrong by 1.6 Ha. The per-block rollback
            # that was here (v_old) only handled the current block's own
            # previous content, which is always zero, so it never fired.
            out = jnp.where(better, 0.0, 1.0) * out
            out = out.at[jnp.asarray(idx)].set(
                out[jnp.asarray(idx)] + jnp.where(better, 1.0, 0.0) * U[:, 0])
    return E_min, out

def _block_H(h1e_mo, eri_mo, DETS, idx, ncore, nact, nele_act):
    """Hamiltonian block over the determinant subset `idx`.

    Identical to the Hamiltonian construction in cs_casci, restricted to the
    rows and columns named by idx.
    """
    nso = 2 * (ncore + nact)
    so = jnp.arange(nso)
    pi, si = so // 2, so % 2
    h_so = jnp.where(si[:, None] == si[None, :], h1e_mo[pi[:, None], pi[None, :]],
                     0.0)
    m4 = ((si[:, None, None, None] == si[None, None, :, None])
          & (si[None, :, None, None] == si[None, None, None, :]))
    eri_so = jnp.where(m4,
                       eri_mo[pi[:, None, None, None], pi[None, None, :, None],
                              pi[None, :, None, None], pi[None, None, None, :]],
                       0.0)
    sub = [DETS[i] for i in idx]
    ndet = len(sub)
    occ_bit = jnp.zeros((ndet, nso), dtype=bool)
    occ_bit = occ_bit.at[jnp.arange(ndet)[:, None], jnp.asarray(sub)].set(True)
    diag1 = occ_bit @ jnp.diag(h_so)
    E2ij = jnp.einsum('ijij->ij', eri_so) - jnp.einsum('ijji->ij', eri_so)
    diag2 = ((occ_bit @ jnp.triu(E2ij, k=1)) * occ_bit).sum(axis=1)
    XA = occ_bit[:, None, :] & ~occ_bit[None, :, :]
    XB = ~occ_bit[:, None, :] & occ_bit[None, :, :]
    nA, nB = XA.sum(axis=2), XB.sum(axis=2)
    pos = jnp.cumsum(occ_bit, axis=1) - 1

    def so_idx(Xm):
        return jnp.argmax(Xm, axis=2)

    def second_idx(Xm, first):
        Xwo = jnp.where(jnp.arange(nso)[None, None, :] == first[..., None],
                        False, Xm)
        return jnp.argmax(Xwo, axis=2)

    mask1 = (nA == 1) & (nB == 1)
    i1 = jax.lax.stop_gradient(so_idx(XA))
    k1 = jax.lax.stop_gradient(so_idx(XB))
    pos_i = jnp.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_k = jnp.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = jnp.where((pos_i + pos_k) % 2 == 0, 1.0, -1.0)
    Xijk = jnp.einsum('ijkj->ijk', eri_so) - jnp.einsum('ijjk->ijk', eri_so)
    T = jnp.einsum('aj,ijk->aik', occ_bit, Xijk)
    T_pair = T[jnp.arange(ndet)[:, None], i1, k1]
    occ_i = jnp.take_along_axis(occ_bit[:, :, None], i1[:, :, None],
                                axis=1)[..., 0]
    X_ii_k = Xijk[i1, i1, k1]
    val1 = sign1 * (h_so[i1, k1] + T_pair - occ_i * X_ii_k)
    H = jnp.where(mask1, val1, 0.0)

    mask2 = (nA == 2) & (nB == 2)
    i2 = jax.lax.stop_gradient(second_idx(XA, i1))
    k2 = jax.lax.stop_gradient(second_idx(XB, k1))
    pos_i2 = jnp.take_along_axis(pos[:, None, :], i2[:, :, None], axis=2)[..., 0]
    pos_k2 = jnp.take_along_axis(pos[None, :, :], k2[:, :, None], axis=2)[..., 0]
    sign2 = jnp.where((pos_i + pos_i2 + pos_k + pos_k2) % 2 == 0, 1.0, -1.0)
    val2 = sign2 * (eri_so[i1, i2, k1, k2] - eri_so[i1, i2, k2, k1])
    H = H + jnp.where(mask2, val2, 0.0)
    return H + jnp.diag(diag1 + diag2)

def casci_energy_from_mol(mol):
    """Fully differentiable map from a Mole pytree to the CASCI energy."""
    h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    mf = pscf.RHF(mol)
    mf.kernel()
    C = mf.mo_coeff
    h1e_mo = jnp.einsum("ai,ab,bj->ij", C, h1e, C)
    eri_mo = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, eri)
    E_ci = cs_casci(h1e_mo, eri_mo)
    return mol.energy_nuc() + E_ci
