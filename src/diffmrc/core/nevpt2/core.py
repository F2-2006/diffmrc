"""NEVPT2 core classes: Si, Sij, Sir, Sijr, Srsi, Sijrs.

A differentiable reimplementation of the core-class terms of the reference
NEVPT2 implementation. Input conventions match nevpt2_impl:

    h1e_act : active effective one-electron integrals, paired with h2e in the
              (pr|qs) convention
    h2e_act : active two-electron integrals, (p r|q s) ordering
    h2e_v*  : mixed outer/inner blocks in the slicing convention used throughout

Each term is validated against an independent implementation by comparing the
individual norm and h arrays, not only the summed energy.
"""
from jax import numpy as jnp


def make_hdm1(dm1):
    delta = jnp.eye(dm1.shape[0])
    return 2.0 * delta - dm1.transpose(1, 0)


def make_hdm2(dm1, dm2):
    delta = jnp.eye(dm2.shape[0])
    dm2x = jnp.einsum('ikjl->ijkl', dm2) - jnp.einsum('jk,il->ijkl', delta, dm1)
    return (jnp.einsum('klij->ijkl', dm2x)
            + jnp.einsum('il,kj->ijkl', delta, dm1)
            + jnp.einsum('jk,li->ijkl', delta, dm1)
            - 2.0 * jnp.einsum('ik,lj->ijkl', delta, dm1)
            - 2.0 * jnp.einsum('jl,ki->ijkl', delta, dm1)
            - 2.0 * jnp.einsum('il,jk->ijkl', delta, delta)
            + 4.0 * jnp.einsum('ik,jl->ijkl', delta, delta))


def make_hdm3(dm1, dm2, dm3, hdm1, hdm2):
    delta = jnp.eye(dm3.shape[0])
    return (- jnp.einsum('pb,qrac->pqrabc', delta, hdm2)
            - jnp.einsum('br,pqac->pqrabc', delta, hdm2)
            + jnp.einsum('bq,prac->pqrabc', delta, hdm2) * 2.0
            + jnp.einsum('ap,bqcr->pqrabc', delta, dm2) * 2.0
            - jnp.einsum('ap,cr,bq->pqrabc', delta, delta, dm1) * 4.0
            + jnp.einsum('cr,bqap->pqrabc', delta, dm2) * 2.0
            - jnp.einsum('bqapcr->pqrabc', dm3)
            + jnp.einsum('ar,pc,bq->pqrabc', delta, delta, dm1) * 2.0
            - jnp.einsum('ar,bqcp->pqrabc', delta, dm2))


def make_a23(h1e, h2e, dm1, dm2, dm3):
    return (- jnp.einsum('ip,caib->abcp', h1e, dm2)
            - jnp.einsum('pijk,cajbik->abcp', h2e, dm3)
            + 2.0 * jnp.einsum('bp,ca->abcp', h1e, dm1)
            + 2.0 * jnp.einsum('pibk,caik->abcp', h2e, dm2))


def make_a25(h1e, h2e, dm1, dm2):
    return (- jnp.einsum('pi,ai->ap', h1e, dm1)
            - jnp.einsum('pijk,jaik->ap', h2e, dm2)
            + 2.0 * jnp.einsum('ap->pa', h1e)
            + 2.0 * jnp.einsum('piaj,ij->ap', h2e, dm1))


def make_a22_full(h1e, h2e, dm2, dm3, f3ca, f3ac):
    """The full make_a22 contraction, including the fourth-order density terms."""
    a22 = (- jnp.einsum('pb,kipjac->ijkabc', h1e, dm3)
           - jnp.einsum('pa,kibjpc->ijkabc', h1e, dm3)
           + jnp.einsum('cp,kibjap->ijkabc', h1e, dm3)
           + jnp.einsum('cqra,kibjqr->ijkabc', h2e, dm3)
           - jnp.einsum('qcpq,kibjap->ijkabc', h2e, dm3)
           - f3ac.transpose(1, 5, 0, 2, 4, 3))
    fdm2 = jnp.einsum('pqrb,kiqcpr->ikbc', h2e, dm3)
    for i in range(h2e.shape[0]):
        a22 = a22.at[:, i, :, i, :, :].add(-fdm2)
    a22 = (a22
           - jnp.einsum('pqab,kiqjpc->ijkabc', h2e, dm3)
           + jnp.einsum('pcrb,kiajpr->ijkabc', h2e, dm3)
           + jnp.einsum('cqrb,kiqjar->ijkabc', h2e, dm3)
           - f3ac.transpose(1, 3, 0, 4, 2, 5)
           + f3ca.transpose(1, 3, 0, 4, 2, 5)
           + 2.0 * jnp.einsum('jb,kiac->ijkabc', h1e, dm2)
           + 2.0 * jnp.einsum('pjrb,kiprac->ijkabc', h2e, dm3))
    fdm2 = (jnp.einsum('pa,kipc->ikac', h1e, dm2)
            - jnp.einsum('cp,kiap->ikac', h1e, dm2)
            - jnp.einsum('cqra,kiqr->ikac', h2e, dm2)
            + jnp.einsum('qcpq,kiap->ikac', h2e, dm2)
            + jnp.einsum('pqra,kiqcpr->ikac', h2e, dm3)
            - jnp.einsum('rcpq,kiaqrp->ikac', h2e, dm3))
    for i in range(h2e.shape[0]):
        a22 = a22.at[:, i, :, :, i, :].add(fdm2 * 2)
    return a22


def si_e2(h1e_act, h2e_act, h2e_v_si, h1e_v_si, dm1, dm2, dm3, f3ca, f3ac,
          mo_energy_core):
    """Si class, single excitation out of the core: E2 = -sum norm/(diff + ener/norm).

    h2e_v_si : active-core mixed integrals
    h1e_v_si : h1eff over active and core; diff = -eps_core
    """
    ncas = h1e_act.shape[0]
    a22 = make_a22_full(h1e_act, h2e_act, dm2, dm3, f3ca, f3ac)
    a23 = make_a23(h1e_act, h2e_act, dm1, dm2, dm3)
    a25 = make_a25(h1e_act, h2e_act, dm1, dm2)
    delta = jnp.eye(ncas)
    dm3_h = jnp.einsum('abef,cd->abcdef', dm2, delta) * 2 - dm3.transpose(0, 1, 3, 2, 4, 5)
    dm2_h = jnp.einsum('ab,cd->abcd', dm1, delta) * 2 - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2 * delta - dm1.transpose(1, 0)
    ener = (jnp.einsum('qpir,pqrabc,baic->i', h2e_v_si, a22, h2e_v_si)
            + jnp.einsum('qpir,pqra,ai->i', h2e_v_si, a23, h1e_v_si) * 2.0
            + jnp.einsum('pi,pa,ai->i', h1e_v_si, a25, h1e_v_si))
    norm = (jnp.einsum('qpir,rpqbac,baic->i', h2e_v_si, dm3_h, h2e_v_si)
            + jnp.einsum('qpir,rpqa,ai->i', h2e_v_si, dm2_h, h1e_v_si) * 2.0
            + jnp.einsum('pi,pa,ai->i', h1e_v_si, dm1_h, h1e_v_si))
    diff = -mo_energy_core
    idx = jnp.abs(norm) > 1e-12
    # Guarded denominator: a raw 0/0 in the untaken branch of where() still
    # propagates NaN -- the reverse pass of the division computes 0/0 when
    # the cotangent is zeroed by that same where.
    norm_safe = jnp.where(idx, norm, 1.0)
    e2 = -(jnp.where(idx, norm, 0.0)
           / (diff + jnp.where(idx, ener / norm_safe, 0.0) + 1e-30)).sum()
    return e2


def make_a3(h1e, h2e, dm1, dm2, hdm1):
    delta = jnp.eye(dm2.shape[0])
    return (jnp.einsum('ia,ip->pa', h1e, hdm1)
            + 2.0 * jnp.einsum('ijka,pj,ik->pa', h2e, delta, dm1)
            - jnp.einsum('ijka,jpik->pa', h2e, dm2))


def sijr_e2(h2e_v, hdm1, a3, mo_energy_vir, mo_energy_core, ncore):
    """Sijr class, double excitation from the core into the virtual space; reduced over r <= s."""
    ci_diag = jnp.diag_indices(ncore)
    ci_triu = jnp.triu_indices(ncore)
    norm = (2.0 * jnp.einsum('rpji,raji,pa->rji', h2e_v, h2e_v, hdm1)
            - 1.0 * jnp.einsum('rpji,raij,pa->rji', h2e_v, h2e_v, hdm1))
    norm = norm + norm.transpose(0, 2, 1)
    norm = norm.at[:, ci_diag[0], ci_diag[1]].multiply(0.5)
    h = (2.0 * jnp.einsum('rpji,raji,pa->rji', h2e_v, h2e_v, a3)
         - 1.0 * jnp.einsum('rpji,raij,pa->rji', h2e_v, h2e_v, a3))
    h = h + h.transpose(0, 2, 1)
    h = h.at[:, ci_diag[0], ci_diag[1]].multiply(0.5)
    diff = (mo_energy_vir[:, None, None] - mo_energy_core[None, :, None]
            - mo_energy_core[None, None, :])
    norm_t = norm[:, ci_triu[0], ci_triu[1]]
    h_t = h[:, ci_triu[0], ci_triu[1]]
    diff_t = diff[:, ci_triu[0], ci_triu[1]]
    idx = jnp.abs(norm_t) > 1e-12
    norm_t_safe = jnp.where(idx, norm_t, 1.0)
    return -(jnp.where(idx, norm_t, 0.0)
             / (diff_t + jnp.where(idx, h_t / norm_t_safe, 0.0)
                + 1e-30)).sum()


def make_k27(h1e, h2e, dm1, dm2):
    return (- jnp.einsum('ai,pi->pa', h1e, dm1)
            - jnp.einsum('iajk,pkij->pa', h2e, dm2)
            + jnp.einsum('iaji,pj->pa', h2e, dm1))


def srsi_e2(h2e_v, dm1, k27, mo_energy_vir, mo_energy_core, nvir):
    """Srsi class, double excitation from the virtual space into the core; reduced over r <= s."""
    vi_diag = jnp.diag_indices(nvir)
    vi_triu = jnp.triu_indices(nvir)
    norm = (2.0 * jnp.einsum('rsip,rsia,pa->rsi', h2e_v, h2e_v, dm1)
            - 1.0 * jnp.einsum('rsip,sria,pa->rsi', h2e_v, h2e_v, dm1))
    norm = norm + norm.transpose(1, 0, 2)
    norm = norm.at[vi_diag].multiply(0.5)
    h = (2.0 * jnp.einsum('rsip,rsia,pa->rsi', h2e_v, h2e_v, k27)
         - 1.0 * jnp.einsum('rsip,sria,pa->rsi', h2e_v, h2e_v, k27))
    h = h + h.transpose(1, 0, 2)
    h = h.at[vi_diag].multiply(0.5)
    diff = (mo_energy_vir[:, None, None] + mo_energy_vir[None, :, None]
            - mo_energy_core[None, None, :])
    norm_t = norm[vi_triu]
    h_t = h[vi_triu]
    diff_t = diff[vi_triu]
    idx = jnp.abs(norm_t) > 1e-12
    norm_t_safe = jnp.where(idx, norm_t, 1.0)
    return -(jnp.where(idx, norm_t, 0.0)
             / (diff_t + jnp.where(idx, h_t / norm_t_safe, 0.0)
                + 1e-30)).sum()


def make_a9(h1e, h2e, hdm1, hdm2, hdm3):
    a9 = jnp.einsum('ib,pqai->pqab', h1e, hdm2)
    a9 = a9 + jnp.einsum('ijib,pqaj->pqab', h2e, hdm2) * 2.0
    a9 = a9 - jnp.einsum('ijjb,pqai->pqab', h2e, hdm2)
    a9 = a9 - jnp.einsum('ijkb,pkqaij->pqab', h2e, hdm3)
    a9 = a9 + jnp.einsum('ia,pqib->pqab', h1e, hdm2)
    a9 = a9 - jnp.einsum('ijja,pqib->pqab', h2e, hdm2)
    a9 = a9 - jnp.einsum('ijba,pqji->pqab', h2e, hdm2)
    a9 = a9 + jnp.einsum('ijia,pqjb->pqab', h2e, hdm2) * 2.0
    a9 = a9 - jnp.einsum('ijka,pqkjbi->pqab', h2e, hdm3)
    return a9


def sij_e2(h2e_v, hdm2, a9, mo_energy_core, ncore):
    """Sij class: double excitation within the core.

    SUMMATION CONVENTION. The sum runs over all ncore^2 pairs (i, j); there is
    no i <= j reduction. This is the single largest source of discrepancy
    encountered during development, and the convention is established by reading
    the reference implementation rather than inferred.

    The reference Sij calls _norm_to_energy(norm, h, -diff) with norm and h as
    complete (ncore, ncore) matrices. Its source comments are explicit for the
    two neighbouring classes and silent here:

        Sijr : "We sum norm and h only over i <= j"  - symmetrize, then take the
                                                       upper triangle
        Srsi : "We sum norm and h only over r <= s"  - likewise
        Sij  : no such comment, no symmetrization, no triangle - full sum

    The norm matrix is symmetric, so the full sum is the diagonal plus twice the
    off-diagonal, whereas taking the upper triangle counts the off-diagonal once
    and therefore omits exactly half of it. In one test this accounted for 99%
    of a 1.98e-4 Eh discrepancy.

    The numerical threshold is aligned with the reference at 1e-14; the earlier
    value here was 1e-12.
    """
    norm = 0.5 * jnp.einsum('qpij,baij,pqab->ij', h2e_v, h2e_v, hdm2)
    h = 0.5 * jnp.einsum('qpij,baij,pqab->ij', h2e_v, h2e_v, a9)
    diff = mo_energy_core[:, None] + mo_energy_core[None, :]
    norm_t = norm.reshape(-1)
    h_t = h.reshape(-1)
    diff_t = diff.reshape(-1)
    idx = jnp.abs(norm_t) > 1e-14
    norm_t_safe = jnp.where(idx, norm_t, 1.0)
    return -(jnp.where(idx, norm_t, 0.0)
             / (-diff_t + jnp.where(idx, h_t / norm_t_safe, 0.0)
                + 1e-30)).sum()


def make_a12(h1e, h2e, dm1, dm2, dm3):
    return (jnp.einsum('ia,qpib->pqab', h1e, dm2)
            - jnp.einsum('bi,qpai->pqab', h1e, dm2)
            + jnp.einsum('ijka,qpjbik->pqab', h2e, dm3)
            - jnp.einsum('kbij,qpajki->pqab', h2e, dm3)
            - jnp.einsum('bjka,qpjk->pqab', h2e, dm2)
            + jnp.einsum('jbij,qpai->pqab', h2e, dm2))


def make_a13(h1e, h2e, dm1, dm2, dm3):
    delta = jnp.eye(dm3.shape[0])
    a13 = -jnp.einsum('ia,qbip->pqab', h1e, dm2)
    a13 = a13 + jnp.einsum('pa,qb->pqab', h1e, dm1) * 2.0
    a13 = a13 + jnp.einsum('bi,qiap->pqab', h1e, dm2)
    a13 = a13 - jnp.einsum('pa,bi,qi->pqab', delta, h1e, dm1) * 2.0
    a13 = a13 - jnp.einsum('ijka,qbjpik->pqab', h2e, dm3)
    a13 = a13 + jnp.einsum('kbij,qjapki->pqab', h2e, dm3)
    a13 = a13 + jnp.einsum('blma,qmlp->pqab', h2e, dm2)
    a13 = a13 + jnp.einsum('kpma,qbkm->pqab', h2e, dm2) * 2.0
    a13 = a13 - jnp.einsum('bpma,qm->pqab', h2e, dm1) * 2.0
    a13 = a13 - jnp.einsum('lbkl,qkap->pqab', h2e, dm2)
    a13 = a13 - jnp.einsum('ap,mbkl,qlmk->pqab', delta, h2e, dm2) * 2.0
    a13 = a13 + jnp.einsum('ap,lbkl,qk->pqab', delta, h2e, dm1) * 2.0
    return a13


def sir_e2(h2e_v1, h2e_v2, h1e_v, dm1, dm2, dm3, a12, a13, mo_energy_core,
           mo_energy_vir):
    """Sir class: single excitation between core and virtual."""
    norm = (jnp.einsum('rpiq,raib,qpab->ir', h2e_v1, h2e_v1, dm2) * 2.0
            - jnp.einsum('rpiq,rabi,qpab->ir', h2e_v1, h2e_v2, dm2)
            - jnp.einsum('rpqi,raib,qpab->ir', h2e_v2, h2e_v1, dm2)
            + jnp.einsum('raqi,rabi,qb->ir', h2e_v2, h2e_v2, dm1) * 2.0
            - jnp.einsum('rpqi,rabi,qbap->ir', h2e_v2, h2e_v2, dm2)
            + jnp.einsum('rpqi,raai,qp->ir', h2e_v2, h2e_v2, dm1)
            + jnp.einsum('rpiq,ri,qp->ir', h2e_v1, h1e_v, dm1) * 4.0
            - jnp.einsum('rpqi,ri,qp->ir', h2e_v2, h1e_v, dm1) * 2.0
            + jnp.einsum('ri,ri->ir', h1e_v, h1e_v) * 2.0)
    h = (jnp.einsum('rpiq,raib,pqab->ir', h2e_v1, h2e_v1, a12) * 2.0
         - jnp.einsum('rpiq,rabi,pqab->ir', h2e_v1, h2e_v2, a12)
         - jnp.einsum('rpqi,raib,pqab->ir', h2e_v2, h2e_v1, a12)
         + jnp.einsum('rpqi,rabi,pqab->ir', h2e_v2, h2e_v2, a13))
    diff = mo_energy_core[:, None] - mo_energy_vir[None, :]
    idx = jnp.abs(norm) > 1e-12
    norm_safe = jnp.where(idx, norm, 1.0)
    return -(jnp.where(idx, norm, 0.0)
             / (-diff + jnp.where(idx, h / norm_safe, 0.0) + 1e-30)).sum()


def sijrs_e2(cvcv, mo_energy_core, mo_energy_vir, ncore, nvir):
    """Sijrs class: double excitation between core and virtual.

    This term involves only integrals and a denominator, no density matrix. The
    reference returns (norm, e) with e equal to the energy contribution, since
    the denominator is already folded into the intermediate.

    cvcv is the core x virtual x core x virtual block in (c v|c v) order.
    """
    eia = mo_energy_core[:, None] - mo_energy_vir[None, :]
    norm = 0.0
    e2 = 0.0
    for i in range(ncore):
        djba = (eia.reshape(-1, 1) + eia[i].reshape(1, -1)).ravel()
        gi = cvcv[i * nvir:(i + 1) * nvir].reshape(nvir, ncore, nvir).transpose(1, 2, 0)
        t2i = (gi.ravel() / djba).reshape(ncore, nvir, nvir)
        theta = gi * 2 - gi.transpose(0, 2, 1)
        norm = norm + jnp.einsum('jab,jab', gi, theta)
        e2 = e2 + jnp.einsum('jab,jab', t2i, theta)
    return e2, norm


def e2_core(h1e_act, h2e_act, dm1, dm2, dm3, mo_energy, ncore, nact, f3ca, f3ac,
            eri_mo=None, h1eff=None):
    """Total second-order energy over the core classes.

    h2e_act  : active block in (p r|q s) order
    mo_energy: canonical orbital energies
    eri_mo, h1eff : full-space MO integrals in (p r|q s) order, from which the
                    mixed blocks are sliced
    """
    nocc = ncore + nact
    nvir = eri_mo.shape[0] - nocc
    eps_core = mo_energy[:ncore]
    eps_vir = mo_energy[nocc:]
    # ---- Slices equivalent to the reference's block layout. The mapping was
    #      established by verifying all 24 index permutations against the
    #      reference blocks.
    #      With t = einsum('prqs->pqrs', eri_mo), so that t[p,q,r,s] equals
    #      eri_mo[p,r,q,s], the blocks are direct slices: ----
    #   h2e_v_si   = t[act,act,core,act]   （PySCF: ppaa[act,core].T(0,2,1,3)）
    #   h2e_v_sij  = t[act,act,core,core]  （PySCF: papa[:c,:,:c].T(1,3,0,2)）
    #   h2e_v_sijr = t[vir,act,core,core]  （PySCF: pacv[:c].T(3,1,2,0)）
    #   h2e_v_srsi = t[vir,vir,core,act]   （PySCF: pacv[nocc:].T(3,0,2,1)）
    #   h2e_v1_sir = t[vir,act,core,act]   （PySCF: ppaa[nocc:,:c].T(0,2,1,3)）
    #   h2e_v2_sir = t[vir,act,act,core]   （PySCF: papa[nocc:,:,:c].T(0,3,1,2)）
    t = jnp.einsum('prqs->pqrs', eri_mo)
    h2e_v_si = t[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
    h1e_v_si = h1eff[ncore:nocc, :ncore]
    h2e_v_sij = t[ncore:nocc, ncore:nocc, :ncore, :ncore]
    h2e_v_sijr = t[nocc:, ncore:nocc, :ncore, :ncore]
    h2e_v_srsi = t[nocc:, nocc:, :ncore, ncore:nocc]
    h2e_v1_sir = t[nocc:, ncore:nocc, :ncore, ncore:nocc]
    h2e_v2_sir = t[nocc:, ncore:nocc, ncore:nocc, :ncore]
    h1e_v_sir = h1eff[nocc:, :ncore]
    # Sijrs: cvcv in packed (ncore*nvir, ncore*nvir) form, verified to machine precision
    cvcv = eri_mo[:ncore, nocc:, :ncore, nocc:].reshape(ncore * nvir, ncore * nvir)

    hdm1 = make_hdm1(dm1)
    hdm2 = make_hdm2(dm1, dm2)
    hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2)

    e2 = 0.0
    if ncore > 0:
        e2 += si_e2(h1e_act, h2e_act, h2e_v_si, h1e_v_si, dm1, dm2, dm3, f3ca, f3ac,
                    eps_core)
        e_sijrs, _ = sijrs_e2(cvcv, eps_core, eps_vir, ncore, nvir)
        e2 += e_sijrs
        a3 = make_a3(h1e_act, h2e_act, dm1, dm2, hdm1)
        e2 += sijr_e2(h2e_v_sijr, hdm1, a3, eps_vir, eps_core, ncore)
        k27 = make_k27(h1e_act, h2e_act, dm1, dm2)
        e2 += srsi_e2(h2e_v_srsi, dm1, k27, eps_vir, eps_core, nvir)
        a9 = make_a9(h1e_act, h2e_act, hdm1, hdm2, hdm3)
        e2 += sij_e2(h2e_v_sij, hdm2, a9, eps_core, ncore)
        a12 = make_a12(h1e_act, h2e_act, dm1, dm2, dm3)
        a13 = make_a13(h1e_act, h2e_act, dm1, dm2, dm3)
        e2 += sir_e2(h2e_v1_sir, h2e_v2_sir, h1e_v_sir, dm1, dm2, dm3, a12, a13,
                     eps_core, eps_vir)
    return e2
