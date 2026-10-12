#!/usr/bin/env python
"""Differentiable CASPT2: minimal core (P1 pilot, first class).

This module implements the active-to-double-virtual class of an
uncontracted CASPT2 in the semicanonical (Roos) zeroth order.  The
class coupling ``norm`` is the same physical quantity the NEVPT2
double-external class (``nevpt2.impl.srs_e2``) assembles -- what
differs between the two methods is the denominator:

    NEVPT2 (Dyall zeroth order):  D = eps_a + eps_b + h/norm
    CASPT2 (Roos zeroth order):   D = eps_a + eps_b - E0

with E0 = sum_p occ_p eps_p the diagonal-Fock reference expectation
(core doubly occupied, active natural occupations, virtual empty).
``eps`` must be the semicanonical Fock diagonal -- the
``canonicalize_jax`` output -- because both denominators are defined
only for that basis.

The class carries the whole H2 correlation energy (no core, no single
external by symmetry), which is the first anchor:
server/caspt2_anchor/h2_cas22.log, ORCA CASPT2 = -0.004043415 Eh.
"""
import logging  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp  # noqa: E402

import numpy as np  # noqa: E402

_log = logging.getLogger(__name__)

if os.environ.get('CASPT2_DEBUG'):
    # Back-compat for the pre-logging switch (retired 2026-10-12): the
    # environment variable still forces the debug records out, even
    # without the CLI's -v.
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s %(name)s] %(message)s')
    _log.setLevel(logging.DEBUG)

__all__ = ['caspt2_double_external_e2', 'caspt2_full_e2',
           'caspt2_contracted_e2']


def caspt2_full_e2(h1e_mo, eri_mo, ci, dets, *, ncore, nact, nele, eps,
                   occ_variant='rdm'):
    """Uncontracted CASPT2 energy over the explicit external space.

    One Hamiltonian build over the *full valence* determinant list --
    core fully occupied, the ``nele`` electrons spread over active +
    virtual -- restricted to its S_z = 0 block, applied to the reference
    vector embedded in that list, reads every external coupling at once:

        v_m = (H @ psi0)_m,      m in the external space
        E2  = sum_m v_m^2 / (E0^0 - E_m^0)

    with the diagonal-Fock (semicanonical Roos) zeroth order:
    E0^0 = sum_p n_p eps_p -- n_p = 2 for core, the CI 1-RDM diagonal for
    active (``occ_variant='rdm'``) or integer occupations of the leading
    determinant (``'integer'``) -- and E_m^0 = sum of eps over the
    integer occupations of m.  ``H`` is ``core.casci._block_H``, the same
    bitwise construction the CI solver uses; no Slater-Condon formula is
    re-derived here.

    Returns ``(e2_total, breakdown)`` where ``breakdown`` maps
    ``(core_holes, virtual_electrons)`` to the summed contribution --
    the tags line up with the external-class picture (0/1/2 core holes =
    V0/Vm1/Vm2 in the ORCA decomposition).  Reference-space determinants
    (the CI space itself) are excluded.
    """
    from .casci import _block_H
    from .nevpt2.impl import cas_dets_for2, make_dm123_sf

    nmo = h1e_mo.shape[0]
    nvir = nmo - ncore - nact
    dets = [int(d) for d in dets]
    c = np.asarray(ci).ravel()

    # Full valence list: core occupied + nele electrons over act + virt.
    dets_full = [int(d) for d in cas_dets_for2(ncore, nact + nvir, nele)]
    nso = 2 * (ncore + nact + nvir)
    amask = 0
    for so in range(0, nso, 2):
        amask |= (1 << so)

    def sz_of(d):
        # S_z of the *full* determinant: the doubly occupied core
        # contributes equal alpha and beta, so 2*n_alpha - n_electrons
        # uses the total electron count.
        return 2 * bin(d & amask).count('1') - (2 * ncore + nele)

    idx0 = [i for i, d in enumerate(dets_full) if sz_of(d) == 0]
    pos = {d: i for i, d in enumerate(dets_full)}

    # Reference embedding is the identity: cas_dets_for2 returns full
    # spin-orbital bitmaps (core bits set, active bits in place), so the
    # active-space determinants are already keys of the full list -- the
    # virtual region is simply unoccupied.
    vec0 = np.zeros(len(dets_full))
    ref_idx = set()
    for d, ci_d in zip(dets, c):
        j = pos[d]
        vec0[j] = ci_d
        ref_idx.add(j)

    # Frozen-core effective one-electron integrals.
    if ncore > 0:
        vhf = (2.0 * jnp.einsum('pqcc->pq', eri_mo[:, :, :ncore, :ncore])
               - jnp.einsum('pccq->pq', eri_mo[:, :ncore, :ncore, :]))
        h1eff = h1e_mo + vhf
    else:
        h1eff = h1e_mo

    # _block_H wants occupation lists (so indices), the format
    # cs_casci_blocked feeds it -- not the integer bitmaps.  Cast to an
    # explicit int64 matrix: a bare nested list can reach jax as float.
    dets_full_so = np.array(
        [[so for so in range(nso) if (int(d) >> so) & 1]
         for d in dets_full], dtype=np.int64)
    H = _block_H(h1eff, eri_mo, dets_full_so, idx0, ncore, nact + nvir, nele)
    v = jnp.asarray(vec0)[jnp.asarray(idx0)]
    v = np.asarray(H @ v)
    _log.debug('dets_full=%d idx0=%d ref_in_block=%d vec0_nnz=%d '
               '|H|max=%.3e |v|max=%.3e',
               len(dets_full), len(idx0), len(ref_idx & set(idx0)),
               int(np.count_nonzero(vec0)),
               float(np.abs(np.asarray(H)).max()),
               float(np.abs(v).max()))

    # Zeroth-order reference energy.
    if occ_variant == 'rdm':
        dm1, _, _ = make_dm123_sf(jnp.asarray(c), dets, nact, ncore)
        occ0 = np.concatenate([np.full(ncore, 2.0),
                               np.asarray(dm1).diagonal(),
                               np.zeros(nvir)])
    else:
        lead = dets[int(np.argmax(np.abs(c)))]
        occ0 = np.concatenate([
            np.full(ncore, 2.0),
            [float(((lead >> (2 * ncore + 2 * t)) & 3).bit_count())
             for t in range(nact)],
            np.zeros(nvir)])
    eps = np.asarray(eps, dtype=float)
    e0_zero = float(occ0 @ eps)

    breakdown = {}
    total = 0.0
    diag = []
    for i_local, j in enumerate(idx0):
        if j in ref_idx:
            continue
        d = dets_full[j]
        occ = np.array([bin((d >> (2 * p)) & 3).count('1')
                        for p in range(nmo)], dtype=float)
        e_m = float(occ @ eps)
        denom = e0_zero - e_m
        if abs(denom) < 1e-8:
            continue
        contrib = float(v[i_local] ** 2) / denom
        holes = int(np.sum(2.0 - occ[:ncore]))
        n_vir_e = int(np.sum(occ[ncore + nact:]))
        key = (holes, n_vir_e)
        breakdown[key] = breakdown.get(key, 0.0) + contrib
        total += contrib
        if os.environ.get('CASPT2_DEBUG'):
            diag.append((abs(contrib), contrib, v[i_local], denom,
                         d, holes, n_vir_e))
    if _log.isEnabledFor(logging.DEBUG):
        diag.sort(reverse=True)
        _log.debug('e0_zero=%.6f', e0_zero)
        for a, c, vv, dn, d, h, nv in diag[:5]:
            _log.debug('top: contrib=%+.6f v=%+.6f denom=%+.6f '
                       'det=%d holes=%d n_vir=%d', c, vv, dn, d, h, nv)
    return total, breakdown


def caspt2_double_external_e2(h1e_mo, eri_mo, c_ci, dets, *,
                              ncore=0, nact=2, nocc=None, mo_energy=None):
    """Active-to-double-virtual CASPT2 class energy.

    Inputs follow ``nevpt2_e2_full2``: full-MO ``h1e_mo``/``eri_mo``,
    the CI vector in the orbital basis, the determinant bitmaps, the
    space definition.  ``mo_energy`` is the semicanonical Fock diagonal
    (``canonicalize_jax`` output).  Returns ``(e2, norm, diff)`` --
    the class energy and the coupling and denominator arrays for the
    record.
    """
    from .nevpt2.impl import (make_a7, make_dm123_sf,
                              make_gfock_mo_energy)

    if nocc is None:
        nocc = ncore + nact
    nmo = h1e_mo.shape[0]

    # Effective one-electron integrals including the frozen-core VHF,
    # exactly as the NEVPT2 chain slices them.
    if ncore > 0:
        vhf = (2.0 * jnp.einsum('pqcc->pq', eri_mo[:, :, :ncore, :ncore])
               - jnp.einsum('pccq->pq', eri_mo[:, :ncore, :ncore, :]))
    else:
        vhf = jnp.zeros_like(h1e_mo)
    h1eff = h1e_mo + vhf
    h1e_act = h1eff[ncore:nocc, ncore:nocc]
    eri_act = jnp.einsum(
        "ijkl->ikjl",
        eri_mo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc])
    h2e_v = jnp.einsum("aibj->abij", eri_mo[nocc:, ncore:nocc,
                                            nocc:, ncore:nocc])

    dm1, dm2, dm3 = make_dm123_sf(c_ci, dets, nact, ncore)
    eps = mo_energy if mo_energy is not None else make_gfock_mo_energy(
        h1e_mo, eri_mo, dm1, ncore, nact)

    # The class coupling: the same contraction srs_e2 assembles (squared
    # coupling per virtual pair, active space contracted in).
    rm2, _a7 = make_a7(h1e_act, eri_act, dm1, dm2, dm3)
    norm = 0.5 * jnp.einsum("rsqp,rsba,pqba->rs", h2e_v, h2e_v, rm2)

    # CASPT2 zeroth-order reference: the diagonal-Fock expectation.
    occ = jnp.zeros(nmo)
    occ = occ.at[:ncore].set(2.0)
    occ = occ.at[ncore:nocc].set(jnp.diag(dm1))
    e0_zero = jnp.sum(occ * eps)

    diff = eps[nocc:, None] + eps[None, nocc:] - e0_zero
    diff_safe = jnp.where(jnp.abs(diff) > 1e-12, diff, jnp.ones_like(diff))
    e2 = -jnp.sum(jnp.where(jnp.abs(diff) > 1e-12, norm / diff_safe, 0.0))
    return float(e2), np.asarray(norm), np.asarray(diff)


def caspt2_contracted_e2(h1e_mo, eri_mo, ci, dets, *, ncore, nact, nele, eps):
    """Contracted (internally contracted) CASPT2, diagonal-complete scheme.

    Follows the OpenMolcas conventions established by line-by-line source
    comparison (2026-10-02 session; OpenMolcas v26.06 src/caspt2):

    - The eight excitation classes are organised by ('out', 'in') orbital
      pools; each unordered pair ''with repetition'' of pool indices labels
      one contracted function
      ``K = E(p1 q1) E(p2 q2) +- E(p1 q2) E(p2 q1)`` acting on the CI
      vector (spin-summed E; full-electron integrals -- the frozen-core
      VHF shortcut is invalid once core holes appear).
    - Molcas normalisation (mkrhsh.F90): VP = cross sum; VM = sqrt(3) *
      cross difference (exists only when q1 != q2 and p1 != p2).  Both
      components enter e2 independently.
    - ``W = <0|H|K>``, ``S = <K|K>``, ``A = <K|F|K>`` with F approximated
      by its semicanonical diagonal eps;  ``e2 = -W^2 / (A - E0 * S)``
      with E0 the reference's F expectation.
    - Same-orbital rows of the active pool are proportional replicas
      across different active orbitals (measured on H2O: K(act2,act2)
      ~ -K(act1,act1)/42, so their W^2/D coincide); only the pool's
      leading orbital contributes those rows.  Core and virtual pools
      have no such replicas and are summed in full.  (The single-
      excitation class V0_ia has W = 0 identically -- its physical
      contribution lives in the off-diagonal F coupling, out of scope of
      this diagonal-complete scheme.)

    Anchors (ratios to ORCA CASPT2 over the five dominant classes):
    H2O/6-31G CAS(2,2) 1.02 / 0.94 / 1.00 / 0.89 / 0.96 (V0_ijab, Vm1_iab,
    Vm2_ab, V1_ija, V2_ij);  CO/6-31G CAS(2,2) total within 0.03%.

    Returns (e2_total, breakdown) with breakdown keyed by class name.
    """
    from .nevpt2.impl import cas_dets_for2  # noqa: F401  (API parity)

    h1e_mo = np.asarray(h1e_mo)
    eri_mo = np.asarray(eri_mo)
    ci = np.asarray(ci).ravel()
    eps = np.asarray(eps, dtype=float)
    nmo = h1e_mo.shape[0]
    nocc = ncore + nact
    nso = 2 * nmo

    ref_pairs = [(int(d), float(c)) for d, c in zip(dets, ci)
                 if abs(c) > 1e-14]

    pi = np.arange(nso) // 2
    si = np.arange(nso) % 2
    h_so = np.where(si[:, None] == si[None, :],
                    h1e_mo[pi[:, None], pi[None, :]], 0.0)
    m4 = ((si[:, None, None, None] == si[None, None, :, None])
          & (si[None, :, None, None] == si[None, None, None, :]))
    eri_so = np.where(
        m4, eri_mo[pi[:, None, None, None], pi[None, None, :, None],
                   pi[None, :, None, None], pi[None, None, None, :]], 0.0)

    _bits_cache = {}

    def so_bits(x):
        if x not in _bits_cache:
            _bits_cache[x] = [i for i in range(nso) if (x >> i) & 1]
        return _bits_cache[x]

    def pc(x):
        return bin(x).count('1')

    def h_elem(J, K):
        """<J|H|K> (Slater-Condon; verified bitwise against core.casci._block_H)."""
        if J == K:
            return 0.0
        dj, dk = J & ~K, K & ~J
        n = pc(dj)
        if n != pc(dk) or n > 2:
            return 0.0
        if n == 1:
            i = (dj & -dj).bit_length() - 1
            k = (dk & -dk).bit_length() - 1
            sgn = (-1.0) ** (pc(J & ((1 << i) - 1)) + pc(K & ((1 << k) - 1)))
            tot = h_so[i, k]
            for j in so_bits(J & K):
                tot += eri_so[i, j, k, j] - eri_so[i, j, j, k]
            return sgn * tot
        i1, i2 = so_bits(dj)
        k1, k2 = so_bits(dk)
        sgn = (-1.0) ** (pc(J & ((1 << i1) - 1)) + pc(J & ((1 << i2) - 1))
                         + pc(K & ((1 << k1) - 1)) + pc(K & ((1 << k2) - 1)))
        return sgn * (eri_so[i1, i2, k1, k2] - eri_so[i1, i2, k2, k1])

    def ann(dd, so):
        if not (dd >> so) & 1:
            return None
        return dd ^ (1 << so), (-1.0) ** pc(dd & ((1 << so) - 1))

    def cre(dd, so):
        if (dd >> so) & 1:
            return None
        return dd | (1 << so), (-1.0) ** pc(dd & ((1 << so) - 1))

    def e1(dd, p, q):
        out = []
        for spin in (0, 1):
            r = ann(dd, 2 * q + spin)
            if r is None:
                continue
            d1, s1 = r
            r2 = cre(d1, 2 * p + spin)
            if r2 is None:
                continue
            d2, s2 = r2
            out.append((d2, s1 * s2))
        return out

    def apply2(dd, p1, q1, p2, q2):
        acc = defaultdict(float)
        for d1, c1 in e1(dd, p1, q1):
            for d2, c2 in e1(d1, p2, q2):
                acc[d2] += c1 * c2
        return acc

    _g = {}

    def g_of(dd):
        if dd not in _g:
            _g[dd] = sum(cJ * h_elem(J, dd) for J, cJ in ref_pairs
                         if pc(J ^ dd) <= 4)
        return _g[dd]

    def wsa(qs, ps, vm):
        """(W, S, A) for the contracted function of the given pair labels."""
        if len(qs) == 1:
            q, p = qs[0], ps[0]
            kacc = defaultdict(float)
            for I, cI in ref_pairs:
                for d1, cc in e1(I, p, q):
                    kacc[d1] += cI * cc
        else:
            q1, q2 = qs
            p1, p2 = ps
            fac = -np.sqrt(3.0) if vm else 1.0
            kacc = defaultdict(float)
            for I, cI in ref_pairs:
                for d2, cc in apply2(I, p1, q1, p2, q2).items():
                    kacc[d2] += cI * cc
                for d2, cc in apply2(I, p1, q2, p2, q1).items():
                    kacc[d2] += fac * cI * cc
        w = sum(cc * g_of(dd) for dd, cc in kacc.items())
        s = sum(cc * cc for cc in kacc.values())
        occ = np.zeros(nmo)
        for dd, cc in kacc.items():
            w2 = cc * cc
            for so in so_bits(dd):
                occ[so // 2] += w2
        return w, s, float(eps @ occ)

    n0 = np.zeros(nmo)
    for dd, c in ref_pairs:
        for so in so_bits(dd):
            n0[so // 2] += c * c
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
        ('V0_ia', core, None, vir, None),
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
    return total, breakdown
