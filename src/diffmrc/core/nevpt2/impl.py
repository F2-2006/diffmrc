#!/usr/bin/env python
"""Differentiable NEVPT2, organized by intermediate.

Modules:

  make_dm123_sf      active-space one- and two-particle density matrices in the
                     spin-free density ordering <p^dag q r^dag s>, following the
                     FCI3pdm_kern_sf convention of the reference implementation
  make_gfock_mo_energy
                     generalized Fock diagonalization, giving a differentiable
                     set of orbital energies; the differentiable counterpart of
                     canonicalization
  make_a7, srs_e2    the dominant double-external class, assembled from norm,
                     h, and diff

Validation: the numpy implementation of the double-external class reproduces
the reference to 0.00.

For H2/6-31G in CAS(2,2) with no core, only the double-external class
contributes: the single-external class vanishes by Brillouin's theorem, the
core classes vanish because there is no core, and the core-virtual double
excitations vanish because there are only two electrons.
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from .rdm import make_dm123_sf
from .f3ca import fdm2fdm3_jax


def make_gfock_mo_energy(h1e_mo, eri_mo, dm1_sf, ncore=0, nact=2):
    """Generalized Fock diagonalization over the full MO space, giving a
    differentiable set of orbital energies, matching canonicalization.

    F_pq = h_pq + sum_rs D_rs [2(pq|rs) - (pr|qs)], with D the active-space
    density; the core and virtual blocks are empty.
    """
    nmo = h1e_mo.shape[0]
    D = jnp.zeros((nmo, nmo))
    for i in range(nact):
        for j in range(nact):
            D = D.at[ncore + i, ncore + j].set(dm1_sf[i, j])
    F = h1e_mo.copy()
    for p in range(nmo):
        for q in range(nmo):
            v = jnp.zeros(())
            for r in range(nmo):
                for s in range(nmo):
                    v = v + D[r, s] * (2 * eri_mo[p, q, r, s] - eri_mo[p, r, q, s])
            F = F.at[p, q].set(h1e_mo[p, q] + v)
    eps = jnp.linalg.eigvalsh(F)
    return eps


def make_a7(h1e, h2e, dm1, dm2, dm3):
    """The make_a7 intermediate, reproducing the reference to 0.00.

    h1e and h2e are the active-space integrals, with h2e in the transposed
    convention (0,2,1,3).
    """
    delta = jnp.eye(dm2.shape[0])
    rm2 = jnp.einsum("iljk->ijkl", dm2) - jnp.einsum("ik,jl->ijkl", dm1, delta)
    rm3 = (jnp.einsum("injmkl->ijklmn", dm3)
           - jnp.einsum("jn,imkl->ijklmn", delta, dm2)
           - jnp.einsum("km,ijln->ijklmn", delta, rm2)
           - jnp.einsum("kn,ijml->ijklmn", delta, rm2))
    a7 = (-jnp.einsum("bi,pqia->pqab", h1e, rm2)
          - jnp.einsum("ai,pqbi->pqab", h1e, rm2)
          - jnp.einsum("kbij,pqkija->pqab", h2e, rm3)
          - jnp.einsum("kaij,pqkibj->pqab", h2e, rm3)
          - jnp.einsum("baij,pqij->pqab", h2e, rm2))
    return rm2, a7


def srs_e2(h1e_act, h2e_act, h2e_v, dm1, dm2, dm3, mo_energy, nocc=2):
    """Double-external class, the dominant contribution: E2 = -sum_ab norm/(diff + h/norm)."""
    rm2, a7 = make_a7(h1e_act, h2e_act, dm1, dm2, dm3)
    norm = 0.5 * jnp.einsum("rsqp,rsba,pqba->rs", h2e_v, h2e_v, rm2)
    h = 0.5 * jnp.einsum("rsqp,rsba,pqab->rs", h2e_v, h2e_v, a7)
    diff = mo_energy[nocc:, None] + mo_energy[None, nocc:]
    norm_safe = jnp.where(jnp.abs(norm) > 1e-12, norm, jnp.zeros_like(norm))
    e2 = -(norm_safe / (diff + jnp.where(jnp.abs(norm) > 1e-12, h / norm, 0.0) + 1e-30)).sum()
    return e2, norm, h


def nevpt2_e2_full(h1e_mo, eri_mo, c_ci, dets, E0_elec, mo_energy=None,
                  ncore=0, nact=2, nocc=2):
    """Full differentiable NEVPT2. For H2 with no core only the double-external
    class contributes; the single-external class vanishes by Brillouin's
    theorem.

    mo_energy holds the orbital energies: from RHF for a fixed-orbital
    calculation, from the generalized Fock diagonalization for a CASSCF one.
    """
    h1e_act = h1e_mo[ncore:ncore + nact, ncore:ncore + nact]
    eri_act = jnp.einsum("ijkl->ikjl", eri_mo[ncore:ncore + nact, ncore:ncore + nact,
                                             ncore:ncore + nact, ncore:ncore + nact])
    h2e_v = jnp.einsum("aibj->abij", eri_mo[nocc:, ncore:ncore + nact,
                                             nocc:, ncore:ncore + nact])
    dm1, dm2, dm3 = make_dm123_sf(c_ci, dets, nact)
    if mo_energy is None:
        eps = make_gfock_mo_energy(h1e_mo, eri_mo, dm1, ncore, nact)
    else:
        eps = mo_energy
    e2, norm, h = srs_e2(h1e_act, eri_act, h2e_v, dm1, dm2, dm3, eps, nocc)
    return e2, eps


def _nele_act_from_dets(dets, ncore):
    """Infer the number of active electrons from the determinant bitmaps."""
    return bin(dets[0]).count('1') - 2 * ncore


def make_a16_simple(h1e, h2e, dm3):
    """The make_a16 contraction for a two-electron system, where the
    fourth-order density terms vanish and only the h1e and h2e contractions
    with the three-particle density remain.

    h1e is active, h2e uses the transposed (0,2,1,3) convention, and dm3 is
    the three-particle density in density order.
    """
    n = h1e.shape[0]
    a16 = (-jnp.einsum("ib,rpqiac->pqrabc", h1e, dm3)
           + jnp.einsum("ia,rpqbic->pqrabc", h1e, dm3)
           - jnp.einsum("ci,rpqbai->pqrabc", h1e, dm3)
           - jnp.einsum("kbia,rpqcki->pqrabc", h2e, dm3)
           - jnp.einsum("kbaj,rpqjkc->pqrabc", h2e, dm3)
           + jnp.einsum("cbij,rpqjai->pqrabc", h2e, dm3))
    fdm2 = jnp.einsum("kbij,rpajki->prab", h2e, dm3)
    for i in range(n):
        a16 = a16.at[:, i, :, :, :, i].add(fdm2)
    a16 = (a16
           + jnp.einsum("jbij,rpqiac->pqrabc", h2e, dm3)
           - jnp.einsum("cjka,rpqbjk->pqrabc", h2e, dm3)
           + jnp.einsum("jcij,rpqbai->pqrabc", h2e, dm3))
    return a16


def make_a16_full(h1e, h2e, dm3, f3ca, f3ac):
    """The full make_a16 contraction, including the fourth-order density terms.

    f3ca and f3ac are the outputs of the fourth-order contraction, in the
    symmetrized [j,i,k,l,p,q] layout.
    """
    n = h1e.shape[0]
    a16 = (-jnp.einsum("ib,rpqiac->pqrabc", h1e, dm3)
           + jnp.einsum("ia,rpqbic->pqrabc", h1e, dm3)
           - jnp.einsum("ci,rpqbai->pqrabc", h1e, dm3)
           - f3ca.transpose(1, 4, 0, 2, 5, 3)
           - jnp.einsum("kbia,rpqcki->pqrabc", h2e, dm3)
           - jnp.einsum("kbaj,rpqjkc->pqrabc", h2e, dm3)
           + jnp.einsum("cbij,rpqjai->pqrabc", h2e, dm3))
    fdm2 = jnp.einsum("kbij,rpajki->prab", h2e, dm3)
    for i in range(n):
        a16 = a16.at[:, i, :, :, :, i].add(fdm2)
    a16 = (a16
           + f3ac.transpose(1, 2, 0, 4, 3, 5)
           - f3ca.transpose(1, 2, 0, 4, 3, 5)
           + jnp.einsum("jbij,rpqiac->pqrabc", h2e, dm3)
           - jnp.einsum("cjka,rpqbjk->pqrabc", h2e, dm3)
           + jnp.einsum("jcij,rpqbai->pqrabc", h2e, dm3))
    return a16


def make_a17(h1e, h2e, dm2, dm3):
    h1e = h1e - jnp.einsum("mjjn->mn", h2e)
    return (-jnp.einsum("pi,cabi->abcp", h1e, dm2)
            - jnp.einsum("kpij,cabjki->abcp", h2e, dm3))


def make_a19(h1e, h2e, dm1, dm2):
    h1e = h1e - jnp.einsum("mjjn->mn", h2e)
    return (-jnp.einsum("pi,ai->ap", h1e, dm1)
            - jnp.einsum("kpij,ajki->ap", h2e, dm2))


def sr_e2(h2e_v3, h1e_v, h1e_act, h2e_act, dm1, dm2, dm3, mo_energy, nocc=2,
          c_ci=None, links_a=None, links_b=None, na=None, nb=None):
    """Single-external class: E2 = -sum_m norm/(diff + ener/norm).

    When c_ci is supplied, the full make_a16 contraction is used, including
    the fourth-order density terms.
    """
    if c_ci is not None:
        norb = h1e_act.shape[0]
        eri = h2e_act.transpose(0, 2, 1, 3)
        _, f3ca = fdm2fdm3_jax(eri, c_ci, range(na), 0, nb, norb, links_a, links_b, na, nb,
                               kern='f3ca')
        _, f3ac = fdm2fdm3_jax(eri, c_ci, range(na), 0, nb, norb, links_a, links_b, na, nb,
                               kern='f3ac')
        a16 = make_a16_full(h1e_act, h2e_act, dm3, f3ca, f3ac)
    else:
        a16 = make_a16_simple(h1e_act, h2e_act, dm3)
    a17 = make_a17(h1e_act, h2e_act, dm2, dm3)
    a19 = make_a19(h1e_act, h2e_act, dm1, dm2)
    ener = (jnp.einsum("ipqr,pqrabc,iabc->i", h2e_v3, a16, h2e_v3)
            + 2 * jnp.einsum("ipqr,pqra,ia->i", h2e_v3, a17, h1e_v)
            + jnp.einsum("ip,pa,ia->i", h1e_v, a19, h1e_v))
    norm = (jnp.einsum("ipqr,rpqbac,iabc->i", h2e_v3, dm3, h2e_v3)
            + 2 * jnp.einsum("ipqr,rpqa,ia->i", h2e_v3, dm2, h1e_v)
            + jnp.einsum("ip,pa,ia->i", h1e_v, dm1, h1e_v))
    diff = mo_energy[nocc:]
    idx = jnp.abs(norm) > 1e-12
    e2 = -(jnp.where(idx, norm, 0.0)
           / (diff + jnp.where(idx, ener / norm, 0.0) + 1e-30)).sum()
    return e2


def nevpt2_e2_full2(h1e_mo, eri_mo, c_ci, dets, E0_elec, mo_energy=None,
                    ncore=0, nact=2, nocc=2, links_a=None, links_b=None):
    """Full NEVPT2 over the single- and double-external classes.

    With a core present, the effective one-electron integrals are
    h1eff = h + core VHF. When links_a and links_b are supplied, the
    single-external class uses the full make_a16 contraction including the
    fourth-order density terms.
    """
    # Core VHF contribution, vhfcore = 2 J_full - K_full, numerically
    # calibrated against the reference.
    # Vectorization: a triple Python loop over ncore, nmo, and nmo performs
    # about 2.35 million indexed updates for nmo = 280 and ncore = 30, taking
    # minutes. A single einsum is mathematically identical:
    #   Σ_c 2·eri[p,q,c,c]  → 2·einsum('pqcc->pq', eri[:,:,:ncore,:ncore])
    #   Σ_c   eri[p,c,c,q]  →   einsum('pccq->pq', eri[:,:ncore,:ncore,:])
    if ncore > 0:
        vhf = (2.0 * jnp.einsum('pqcc->pq', eri_mo[:, :, :ncore, :ncore])
               - jnp.einsum('pccq->pq', eri_mo[:, :ncore, :ncore, :]))
    else:
        vhf = jnp.zeros_like(h1e_mo)
    h1eff = h1e_mo + vhf
    h1e_act = h1eff[ncore:ncore + nact, ncore:ncore + nact]
    eri_act = jnp.einsum("ijkl->ikjl", eri_mo[ncore:ncore + nact, ncore:ncore + nact,
                                             ncore:ncore + nact, ncore:ncore + nact])
    h2e_v = jnp.einsum("aibj->abij", eri_mo[nocc:, ncore:ncore + nact,
                                             nocc:, ncore:ncore + nact])
    h2e_v3 = jnp.einsum("mipq->mpiq", eri_mo[nocc:, ncore:ncore + nact,
                                             ncore:ncore + nact, ncore:ncore + nact])
    h1e_v = h1eff[nocc:, ncore:ncore + nact] - jnp.einsum("mbbn->mn", h2e_v3)
    dm1, dm2, dm3 = make_dm123_sf(c_ci, dets, nact, ncore)
    eps = mo_energy if mo_energy is not None else make_gfock_mo_energy(h1e_mo, eri_mo, dm1, ncore, nact)
    if links_a is not None and c_ci is not None:
        if c_ci.ndim == 1:
            # Full-determinant 1D CI vector to (na, nb) form, by a static
            # index gather and a fermionic phase, both differentiable
            from .f3ca import ab_index_map
            nele_act = _nele_act_from_dets(dets, ncore)
            idx_ab, phase_ab = ab_index_map(ncore, nact, nele_act)
            ci_ab = c_ci[idx_ab] * phase_ab
            # The projection onto S_z = 0 must be renormalized.
            #
            # The determinant list of the CI solver is generated over spin
            # orbitals without fixing the alpha and beta counts, so the basis
            # spans every S_z component. For CAS(4,4) with four core orbitals
            # that is 70 determinants of which 36 have S_z = 0; for CAS(6,6)
            # it is 924 against 400. This step keeps only the S_z = 0 subset.
            #
            # When the ground-state vector has weight in other S_z sectors - 
            # for C2 the projected norm came out at 0.7071 with a weight
            # distribution of 0.25, 0.5, 0.25 - the projection gives a norm
            # below one, which scales down every density matrix and the
            # fourth-order contraction and produces deviations of order 1e-2
            # in the single- and double-external classes. Renormalizing brings
            # C2 into exact agreement with the reference root, the prefactor
            # going from 0.7071 to 1,
            # which served as the cross-check that identified the problem.
            _n = jnp.linalg.norm(ci_ab)
            ci_ab = ci_ab / _n
        else:
            ci_ab = c_ci
        na, nb = ci_ab.shape
        e2_sr = sr_e2(h2e_v3, h1e_v, h1e_act, eri_act, dm1, dm2, dm3, eps, nocc,
                      ci_ab, links_a, links_b, na, nb)
        if ncore > 0:
            from .core import e2_core
            eri = eri_act.transpose(0, 2, 1, 3)
            _, f3ca = fdm2fdm3_jax(eri, ci_ab, range(na), 0, nb, nact, links_a, links_b,
                                   na, nb, kern='f3ca')
            _, f3ac = fdm2fdm3_jax(eri, ci_ab, range(na), 0, nb, nact, links_a, links_b,
                                   na, nb, kern='f3ac')
            e2_core_ = e2_core(h1e_act, eri_act, dm1, dm2, dm3, eps, ncore, nact,
                               f3ca, f3ac, eri_mo=eri_mo, h1eff=h1eff)
            e2_sr = e2_sr + e2_core_
    else:
        e2_sr = sr_e2(h2e_v3, h1e_v, h1e_act, eri_act, dm1, dm2, dm3, eps, nocc)
    e2_srs, norm, h = srs_e2(h1e_act, eri_act, h2e_v, dm1, dm2, dm3, eps, nocc)
    return e2_sr + e2_srs, eps


@jax.custom_jvp
def eigh_safe(a):
    """``jnp.linalg.eigh`` with a degeneracy-safe first-order propagation.

    The vector derivative of ``eigh`` carries a factor 1/(w_j - w_i). At a
    machine-precision degeneracy that factor is 0/0 and returns NaN, which
    then poisons every derivative through the rotated orbitals. Linear
    molecules make this the norm rather than the exception: ThO's
    generalized-Fock core block (17 orbitals) has a minimum eigenvalue gap
    of 2.7e-15 and its virtual block (75 orbitals) 8.9e-16 -- the pi and
    delta shells are exactly degenerate by symmetry. Every previous
    validation ran on H2 and H2O, whose blocks carry no such pairs.

    In a degenerate subspace the eigenbasis is arbitrary, so any choice of
    its derivative is as good as another; the honest choice is to freeze
    the rotation there, which is exactly what the NEVPT2 invariance under
    rotations within the core and virtual blocks permits.

    The softening lives in the JVP rule (and the reverse pass, which jax
    builds by transposing it), not in a hand-written VJP. That matters:
    with a custom_vjp the forward tangent of ``u`` still comes from the
    standard ``eigh`` JVP, so forward-over-reverse (jvp of grad) keeps the
    raw 1/diff = 1e15 and overflows to NaN as soon as something consumes
    the eigenVECTORS -- while segments that only consume the eigenvalues
    look fine because the offending tangent is dead code. Measured: with
    the custom_vjp version, the eigenvalue-only HVP segment was finite and
    the segment that feeds the rotated orbitals into a second CASCI was
    NaN (t219).

    The coupling is Lorentzian-softened, C_ij = L_ij * diff/(diff^2 + tau^2)
    with L = U^T dA U. For |diff| >> tau it is 1/diff to relative order
    (tau/diff)^2 -- below 1e-8 for every gap above 1e-2, which covers all
    validation systems; near zero it goes linearly to zero (the freeze).
    A hard cutoff would leave 1/diff^3 spikes just outside its threshold
    whose second derivative overflows.

    Measured on ThO/ECP (CAS(4,4), ncore=17): with the plain eigh the theta
    gradient of the joint energy is all-NaN; with a custom_vjp version the
    first order is finite but the HVP stays NaN; with this JVP version both
    are finite.
    """
    return jnp.linalg.eigh(a)


@eigh_safe.defjvp
def _eigh_safe_jvp(primals, tangents):
    (a,), (da,) = primals, tangents
    w, u = jnp.linalg.eigh(a)
    L = u.T @ (0.5 * (da + da.T)) @ u
    diff = w[None, :] - w[:, None]
    tau = 1e-4
    F = diff / (diff ** 2 + tau ** 2)
    n = w.shape[0]
    C = L * F * (1.0 - jnp.eye(n))       # diagonal stays zero (normalization)
    dw = jnp.diag(L)
    du = u @ C
    return (w, u), (dw, du)


def canonicalize_jax(C, h1e, eri, c_ci, ncore=0, nact=2, nele_act=2, occ2_idx=(), occ0_idx=()):
    """Differentiable canonicalization, following mcscf.canonicalize with
    cas_nat=True.

    1) Active natural orbitals: diagonalize the active one-particle density,
       rotate within the active space, and re-solve the CI.
    2) Subspace Fock diagonalization: the doubly occupied block spans the
       core plus the occupied active orbitals, the empty block the rest of the
       active space plus the external orbitals, and the remainder is left as
       is. This yields the orbitals and the orbital energies.

    Returns (C_can, ci_can, eps).

    occ2_idx and occ0_idx are static block parameters, because an occupation
    threshold cannot be determined dynamically under differentiation. For
    H2/6-31G the threshold is never approached; for LiH the blocks must be
    supplied from the measured occupation numbers.
    """
    nmo = C.shape[1]
    nocc = ncore + nact
    h1e_mo = jnp.einsum("ai,ab,bj->ij", C, h1e, C)
    eri_mo = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, eri)
    # 1) Active natural orbitals from the one-particle density
    dm1, _, _ = make_dm123_sf(c_ci, cas_dets_for2(ncore, nact, nele_act), nact, ncore)
    w_nat, U_nat = eigh_safe(dm1)
    # The reference orders active natural orbitals by descending occupation
    # number, whereas eigh returns ascending eigenvalues, so the columns are
    # reversed.
    U_nat = U_nat[:, ::-1]
    C1 = C.at[:, ncore:nocc].set(C[:, ncore:nocc] @ U_nat)
    # Re-solve the CI in the natural-orbital basis
    h1e_mo1 = jnp.einsum("ai,ab,bj->ij", C1, h1e, C1)
    eri_mo1 = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C1, C1, C1, C1, eri)
    E0, ci1 = _casci_ci(h1e_mo1, eri_mo1, ncore, nact, nele_act)
    # 2) Generalized Fock matrix, F = h + J - K/2, in the ORIGINAL orbital
    #    basis and with the original active density
    dm1_orig, _, _ = make_dm123_sf(c_ci, cas_dets_for2(ncore, nact, nele_act), nact, ncore)
    D = jnp.zeros((nmo, nmo))
    for i in range(ncore):
        D = D.at[i, i].set(2.0)
    for i in range(nact):
        for j in range(nact):
            D = D.at[ncore + i, ncore + j].set(dm1_orig[i, j])
    # The Fock matrix in the original orbital basis, F = h + J - K/2.
    #
    # Vectorization: a four-fold Python loop over nmo^4 performs about 3.1
    # million indexed updates for O3 with 42 orbitals, which makes
    # compilation explode. The einsum form is identical.
    #
    # Conventions in the reference get_fock:
    #   J = einsum('pqrs,rs->pq', eri, dm), the Coulomb term (p q|r s)
    #   K = einsum('psrq,rs->pq', eri, dm), the exchange term (p s|r q)
    F = h1e_mo + jnp.einsum("pqrs,rs->pq", eri_mo, D) \
        - 0.5 * jnp.einsum("psrq,rs->pq", eri_mo, D)
    # Transform to the natural-orbital basis: F_C1 = U^T F U, with U being the
    # identity outside the active block and the natural-orbital rotation
    # inside it
    U_full = jnp.eye(nmo).at[ncore:nocc, ncore:nocc].set(U_nat)
    F = U_full.T @ F @ U_full
    # Block diagonalization, matching the reference:
    #   - Only the core and external blocks are diagonalized; the active space
    #     is left in its natural-orbital basis and is not rotated.
    #   - The active orbital energies are the diagonal of the Fock matrix in
    #     that basis, while the core and virtual energies are the block
    #     eigenvalues.
    # An earlier version diagonalized the active block as well, which shifted
    # the active orbital energies by 9e-6.
    occ2 = [ncore + i for i in occ2_idx]
    occ0 = [ncore + i for i in occ0_idx]
    blocks = [list(range(ncore)) + occ2, occ0 + list(range(nocc, nmo))]
    blocks = [b for b in blocks if b]
    active_idx = [ncore + i for i in range(nact)]
    C2 = C1.copy()
    eps_map = {}
    for blk in blocks:
        idx = jnp.asarray(blk, dtype=int)
        if len(blk) > 1:
            f_blk = F[idx][:, idx]
            w, u = eigh_safe(f_blk)
            C2 = C2.at[:, blk].set(C1[:, blk] @ u)
            for a, i in enumerate(blk):
                eps_map[i] = w[a]
        else:
            eps_map[blk[0]] = F[blk[0], blk[0]]
    # Active orbitals: the Fock diagonal in the natural-orbital basis, unrotated
    for i in active_idx:
        eps_map[i] = F[i, i]
    eps = jnp.stack([eps_map[i] for i in range(nmo)])  # reorder by orbital index
    # Final CI solve in the canonical orbital basis
    h1e_mo2 = jnp.einsum("ai,ab,bj->ij", C2, h1e, C2)
    eri_mo2 = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C2, C2, C2, C2, eri)
    E0c, ci2 = _casci_ci(h1e_mo2, eri_mo2, ncore, nact, nele_act)
    return C2, ci2, eps


def cas_dets_for2(ncore=0, nact=2, nele_act=2):
    from itertools import combinations
    nso = 2 * (ncore + nact)
    core_so = list(range(2 * ncore))
    act_so = list(range(2 * ncore, nso))
    dets = []
    for comb in combinations(act_so, nele_act):
        det = 0
        for so in core_so + list(comb):
            det |= (1 << so)
        dets.append(det)
    return dets


def _casci_ci(h1e_mo, eri_mo, ncore, nact, nele_act=2):
    """Ground-state CAS energy and CI vector, returned as (E_elec, ci)."""
    from ..casci import cs_casci
    nact_mo = ncore + nact
    e, ci = cs_casci(h1e_mo[:nact_mo, :nact_mo], eri_mo[:nact_mo, :nact_mo, :nact_mo, :nact_mo],
                     ncore=ncore, nact=nact, nele_act=nele_act, return_ci=True)
    return e, ci
