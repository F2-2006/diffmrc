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

Two engines realise the same scheme: ``caspt2_contracted_e2`` (numpy
reference, with the bitwise-verified Slater-Condon coupling and the
debug hooks) and ``caspt2_contracted_e2_jax`` (a jax twin of the same
construction, traceable in the CI coefficients).
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
           'caspt2_contracted_e2', 'caspt2_contracted_e2_jax',
           'caspt2_contracted_e2_coupled']


def _sc_kernel(h1e_mo, eri_mo):
    """Slater-Condon matrix elements over spin-orbital occupation bitmaps.

    Builds the spin-orbital integrals from the (chemist-notation) MO
    integrals and returns ``(h_elem, so_bits)``: ``h_elem(J, K)`` is
    ``<J|H|K>`` for determinant bitmaps, zero on the diagonal (a class
    function never overlaps its own reference coupling) and beyond
    rank-two differences; ``so_bits(x)`` lists the occupied spin orbitals
    of a bitmap with a cache.  Shared by the numpy reference and the jax
    twin so the two engines cannot drift apart.
    """
    h1e_mo = np.asarray(h1e_mo)
    eri_mo = np.asarray(eri_mo)
    nmo = h1e_mo.shape[0]
    nso = 2 * nmo
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

    return h_elem, so_bits


def _ann(dd, so):
    """Annihilate ``so`` in a determinant bitmap (None if empty)."""
    if not (dd >> so) & 1:
        return None
    return dd ^ (1 << so), (-1.0) ** ((dd & ((1 << so) - 1)).bit_count())


def _cre(dd, so):
    """Create ``so`` in a determinant bitmap (None if occupied)."""
    if (dd >> so) & 1:
        return None
    return dd | (1 << so), (-1.0) ** ((dd & ((1 << so) - 1)).bit_count())


def _e1(dd, p, q):
    """Spin-summed E(p, q) applied to a determinant bitmap.

    Returns the list of (target bitmap, fermionic sign) over both spin
    channels, matching the element rules the coupling build has always
    used.
    """
    out = []
    for spin in (0, 1):
        r = _ann(dd, 2 * q + spin)
        if r is None:
            continue
        d1, s1 = r
        r2 = _cre(d1, 2 * p + spin)
        if r2 is None:
            continue
        d2, s2 = r2
        out.append((d2, s1 * s2))
    return out


def _apply2(dd, p1, q1, p2, q2):
    """E(p2 q2) E(p1 q1) applied to a determinant bitmap ->
    {target bitmap: coefficient}."""
    acc = defaultdict(float)
    for d1, c1 in _e1(dd, p1, q1):
        for d2, c2 in _e1(d1, p2, q2):
            acc[d2] += c1 * c2
    return acc


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
    - Molcas normalisation (mkrhsh.F90): VP = cross + direct; VM =
      sqrt(3) * (cross - direct) (exists only when q1 != q2 and
      p1 != p2).  The sqrt(3) is an overall scale: it multiplies the
      whole VM function and cancels in ``W^2 / (A - E0 S)``, so the
      operator content used here is ``cross - direct``, i.e. the two
      orderings carry relative weight -1.  Both components enter e2
      independently.
    - ``W = <0|H|K>``, ``S = <K|K>``, ``A = <K|F|K>`` with F approximated
      by its semicanonical diagonal eps;  ``e2 = -W^2 / (A - E0 * S)``
      with E0 the reference's F expectation.
    - Same-orbital rows of the active pool are proportional replicas
      across different active orbitals (measured on H2O: K(act2,act2)
      ~ -K(act1,act1)/42, so their W^2/D coincide); only the pool's
      leading orbital contributes those rows.  Core and virtual pools
      have no such replicas and are summed in full.
    - The single-excitation class V0_ia (Molcas case D, AIVX; ORCA
      class table) is not the plain single excitation E(a,j)|0> -- that
      one is Brillouin-forbidden at a CASSCF reference (W = 0 measured,
      and it is why Vm1_a and V1_i come out as clean zeros, matching
      ORCA).  Case D instead dresses the excitation with an active-pair
      reorganization: per unordered active pair (u, v) -- the double
      classes' pair convention -- two component functions (Molcas stores
      two slots per pair, src/caspt2 ADDRHSD1/D2),

          cross  K = E(v,u) E(a,j)|0>   -- coupling dominated by (aj|vu)
          direct K = E(v,j) E(a,u)|0>   -- coupling dominated by (au|vj)

      which coincide into a single component for the diagonal pairs
      (v == u), and are solved per block as the projected amplitude
      equation (see the class block below).

    Anchors (ratios to the ORCA CASPT2 class table, full-FIC column):
    H2O/6-31G CAS(2,2) 1.017 / 1.011 / 0.999 / 1.009 / 0.955 (V0_ijab,
    Vm1_iab, Vm2_ab, V1_ija, V2_ij), eight-class total 1.0103;
    CO/6-31G CAS(2,2) 1.009 / 1.011 / 0.963 / 0.999 / 0.987, total
    1.0026;  V0_ia target -0.011135 (ORCA H2O).  This path keeps the
    diagonal-Fock zeroth order (the CASPT2-D-level approximation); the
    full-Fock coupled path is ``caspt2_contracted_e2_coupled``.

    Returns (e2_total, breakdown) with breakdown keyed by class name.
    """
    h1e_mo = np.asarray(h1e_mo)
    eri_mo = np.asarray(eri_mo)
    ci = np.asarray(ci).ravel()
    eps = np.asarray(eps, dtype=float)
    nmo = h1e_mo.shape[0]
    nocc = ncore + nact

    ref_pairs = [(int(d), float(c)) for d, c in zip(dets, ci)
                 if abs(c) > 1e-14]

    h_elem, so_bits = _sc_kernel(h1e_mo, eri_mo)

    _g = {}

    def g_of(dd):
        if dd not in _g:
            _g[dd] = sum(cJ * h_elem(J, dd) for J, cJ in ref_pairs
                         if (J ^ dd).bit_count() <= 4)
        return _g[dd]

    def wsa(qs, ps, vm, pair='both'):
        """(W, S, A) for the contracted function of the given pair labels.

        ``pair`` selects the operator combination of the two-pair branch:
        'both' builds E(p1 q1)E(p2 q2) + fac E(p1 q2)E(p2 q1) as before;
        'direct' and 'cross' keep a single term, which is how the two
        independent amplitudes of the V0_ia class are built.
        """
        if len(qs) == 1:
            q, p = qs[0], ps[0]
            kacc = defaultdict(float)
            for I, cI in ref_pairs:
                for d1, cc in _e1(I, p, q):
                    kacc[d1] += cI * cc
        else:
            q1, q2 = qs
            p1, p2 = ps
            # mkrhsh writes VM = sqrt(3) (cross - direct); the sqrt(3)
            # scales the whole function and cancels in W^2/(A - E0 S).
            fac = -1.0 if vm else 1.0
            kacc = defaultdict(float)
            for I, cI in ref_pairs:
                if pair in ('both', 'direct'):
                    for d2, cc in _apply2(I, p1, q1, p2, q2).items():
                        kacc[d2] += cI * cc
                if pair in ('both', 'cross'):
                    for d2, cc in _apply2(I, p1, q2, p2, q1).items():
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

    # V0_ia (Molcas case D): one inactive hole, one virtual particle and
    # an active-pair reorganization.  Per unordered active pair (u <= v)
    # -- the double classes' pair convention -- and each (core j,
    # virtual a) the class has two component functions, the 'cross' and
    # 'direct' functions of the docstring, built with qs = (u, j) and
    # ps = (a, v):
    #   direct = apply2(p1, q1, p2, q2) = E(v,j) E(a,u)  -- u -> a, j -> v
    #   cross  = apply2(p1, q2, p2, q1) = E(v,u) E(a,j)  -- j -> a, u -> v
    # For a diagonal pair (v == u) the two components coincide as
    # vectors, so the block carries ONE amplitude there.
    #
    # The components of a block are strongly overlapping (the cross and
    # direct vectors share determinants with opposing weights), so the
    # block is solved as the projected amplitude equation
    #
    #     M x = -W,   M[p, q] = sum_D K_p[D] K_q[D] (de_D - E0),
    #
    # the exact projection of the occupation-diagonal H0 - E0 onto the
    # block, with W_p = <0|H|K_p> and E2 = sum_p W_p x_p.  Independent
    # per-function contributions (the strictly diagonal reading) overcount
    # the H2O class by 1.59x and CO by 1.79x against ORCA (t414 round 1);
    # the block solve lands both systems within 2.3% (round 3) and
    # reproduces the block orthogonality of the Molcas case structure.
    acc_ia = 0.0
    _de_cache = {}

    def de_of(dd):
        if dd not in _de_cache:
            tot = 0.0
            for p in range(nmo):
                tot += ((dd >> (2 * p)) & 3).bit_count() * eps[p]
            _de_cache[dd] = tot
        return _de_cache[dd]

    def kacc_of(mode, u, j, a, v):
        kacc = defaultdict(float)
        for I, cI in ref_pairs:
            terms = (_apply2(I, a, j, v, u) if mode == 'cross'
                     else _apply2(I, a, u, v, j))
            for d2, cc in terms.items():
                kacc[d2] += cI * cc
        return kacc

    for iu, u in enumerate(act):
        for v in act[iu:]:
            modes = ('cross',) if v == u else ('cross', 'direct')
            for j in core:
                for a in vir:
                    ks = [kacc_of(m, u, j, a, v) for m in modes]
                    ws = [sum(cc * g_of(dd) for dd, cc in k.items())
                          for k in ks]
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


def caspt2_contracted_e2_jax(h1e_mo, eri_mo, ci, dets, *, ncore, nact, nele,
                             eps):
    """Differentiable twin of :func:`caspt2_contracted_e2`.

    Same diagonal-complete contracted scheme, same class definitions and
    the same diagonal denominator ``A - E0 * S``; verified elementwise
    against the numpy reference (tests/core/test_caspt2.py, t414).

    Every contracted function is linear in the CI vector by
    construction, so its coupling W, overlap S and Fock expectation A
    are static quadratic forms ``c^T M c``.  The matrices are built once
    with numpy from the shared Slater-Condon kernel; the evaluation is a
    chain of jax array operations, so ``jax.grad`` of the returned total
    flows through the whole construction.  The static build costs
    ``ndet``-squared Slater-Condon elements per function: intended for
    pilot active spaces, not for production-size CI lists.

    Returns ``(e2_total, breakdown)`` with jax scalars.
    """
    h_elem, _ = _sc_kernel(h1e_mo, eri_mo)
    eps = np.asarray(eps, dtype=float)
    nmo = eps.shape[0]
    nocc = ncore + nact
    dets = [int(d) for d in dets]
    ndet = len(dets)

    de_cache = {}

    def de_of(dd):
        """Per-determinant Fock energy, sum_p n_p(D) eps_p."""
        if dd not in de_cache:
            tot = 0.0
            for p in range(nmo):
                tot += ((dd >> (2 * p)) & 3).bit_count() * eps[p]
            de_cache[dd] = tot
        return de_cache[dd]

    def v_single(p, q):
        return [dict(_e1(Ibit, p, q)) for Ibit in dets]

    def v_two(p1, q1, p2, q2, pair='both', fac=1.0):
        out = []
        for Ibit in dets:
            if pair == 'direct':
                out.append(dict(_apply2(Ibit, p1, q1, p2, q2)))
            elif pair == 'cross':
                out.append(dict(_apply2(Ibit, p1, q2, p2, q1)))
            else:
                acc = defaultdict(float)
                for tgt, sgn in _apply2(Ibit, p1, q1, p2, q2).items():
                    acc[tgt] += sgn
                for tgt, sgn in _apply2(Ibit, p1, q2, p2, q1).items():
                    acc[tgt] += fac * sgn
                out.append(dict(acc))
        return out

    def gram_forms(va, vb):
        """(S, A) quadratic-form matrices between two component sets:
        S[i, j] = <va_i|vb_j> and A[i, j] = the same with the per-
        determinant Fock energy de_D as weight."""
        q = np.zeros((ndet, ndet))
        qa = np.zeros((ndet, ndet))
        for i, vi in enumerate(va):
            if not vi:
                continue
            for j, vj in enumerate(vb):
                if not vj:
                    continue
                qij = 0.0
                qaij = 0.0
                for tgt in vi.keys() & vj.keys():
                    qij += vi[tgt] * vj[tgt]
                    qaij += vi[tgt] * vj[tgt] * de_of(tgt)
                q[i, j] = qij
                qa[i, j] = qaij
        return q, qa

    def w_form(vrefs):
        """The coupling quadratic form W = c^T mw c of one component set."""
        mw = np.zeros((ndet, ndet))
        for i, vi in enumerate(vrefs):
            if not vi:
                continue
            for j in range(ndet):
                mwij = 0.0
                for tgt, sgn in vi.items():
                    mwij += sgn * h_elem(dets[j], tgt)
                mw[i, j] = mwij
        return mw

    def forms(vrefs):
        """Static (W, S, A) quadratic-form matrices of one function."""
        mw = w_form(vrefs)
        q, qa = gram_forms(vrefs, vrefs)
        return mw, q, qa

    ci = jnp.asarray(ci).ravel()
    e0 = jnp.sum(ci * ci * jnp.asarray([de_of(d) for d in dets]))

    def contrib(fmat):
        mw, q, qa = fmat
        w = ci @ jnp.asarray(mw) @ ci
        s = ci @ jnp.asarray(q) @ ci
        a = ci @ jnp.asarray(qa) @ ci
        dnm = a - e0 * s
        ok = jnp.abs(dnm) > 1e-10
        return -jnp.where(ok, w * w / jnp.where(ok, dnm, 1.0), 0.0)

    core = list(range(ncore))
    act = list(range(ncore, nocc))
    vir = list(range(nocc, nmo))

    def pairs_list(o1, o2):
        if o2 is None:
            return [(x,) for x in o1]
        if o1 is o2:
            return [(o1[i], o1[j]) for i in range(len(o1))
                    for j in range(i, len(o1))]
        return [(x, y) for x in o1 for y in o2]

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
        for qs in pairs_list(o1, o2):
            if len(qs) == 2 and qs[0] == qs[1] and qs[0] in act \
                    and qs[0] != act[0]:
                continue
            for ps in pairs_list(i1c, i2c):
                if len(ps) == 2 and ps[0] == ps[1] and ps[0] in act \
                        and ps[0] != act[0]:
                    continue
                if len(qs) == 1:
                    acc += contrib(forms(v_single(ps[0], qs[0])))
                else:
                    acc += contrib(forms(v_two(ps[0], qs[0], ps[1], qs[1])))
                    if qs[0] != qs[1] and ps[0] != ps[1]:
                        # VM relative weight -1, as in the numpy engine
                        # (the mkrhsh sqrt(3) is an overall scale).
                        acc += contrib(forms(v_two(ps[0], qs[0], ps[1], qs[1],
                                                   fac=-1.0)))
        breakdown[name] = acc
        total += acc

    # V0_ia (Molcas case D): per unordered active pair one or two
    # components, solved per block as M x = -W with
    # M[p, q] = sum_D K_p K_q (de_D - E0), exactly as in the numpy
    # reference.  Every entry is a static quadratic form in c.
    acc_ia = 0.0
    for iu, u in enumerate(act):
        for v in act[iu:]:
            modes = ('cross',) if v == u else ('cross', 'direct')
            for j in core:
                for a in vir:
                    vls = [v_two(a, u, v, j, pair=m) for m in modes]
                    ws = [ci @ jnp.asarray(w_form(x)) @ ci for x in vls]
                    if len(vls) == 1:
                        q, qa = gram_forms(vls[0], vls[0])
                        m00 = (ci @ jnp.asarray(qa) @ ci
                               - e0 * (ci @ jnp.asarray(q) @ ci))
                        ok = jnp.abs(m00) > 1e-10
                        acc_ia += -jnp.where(
                            ok, ws[0] * ws[0] / jnp.where(ok, m00, 1.0), 0.0)
                    else:
                        mm = []
                        for p in range(len(vls)):
                            row = []
                            for q in range(len(vls)):
                                g, ga = gram_forms(vls[p], vls[q])
                                row.append(ci @ jnp.asarray(ga) @ ci
                                           - e0 * (ci @ jnp.asarray(g) @ ci))
                            mm.append(jnp.stack(row))
                        mmat = jnp.stack(mm)
                        wvec = jnp.stack(ws)
                        det = (mmat[0, 0] * mmat[1, 1]
                               - mmat[0, 1] * mmat[1, 0])
                        ok = jnp.abs(det) > 1e-12
                        eye = jnp.eye(len(vls))
                        x = jnp.linalg.solve(
                            mmat + jnp.where(ok, 0.0, 1.0) * eye, -wvec)
                        acc_ia += jnp.where(ok, jnp.dot(wvec, x), 0.0)
    breakdown['V0_ia'] = acc_ia
    total += acc_ia
    return total, breakdown


# ---------------------------------------------------------------------------
# Coupled internally contracted CASPT2 (full generalized-Fock zeroth order)
# ---------------------------------------------------------------------------

def _apply_chain(ref_pairs, ops):
    """Apply E(p1 q1) E(p2 q2) ... to the reference CI expansion.

    ``ops`` is the product written left to right; the rightmost operator
    acts first.  Returns a determinant -> coefficient dict (the same
    representation the class builders use).
    """
    acc = defaultdict(float)
    for I, cI in ref_pairs:
        vec = {I: cI}
        for p, q in reversed(ops):
            nxt = defaultdict(float)
            for dd, cc in vec.items():
                for tgt, sgn in _e1(dd, p, q):
                    nxt[tgt] += cc * sgn
            vec = nxt
        for tgt, cc in vec.items():
            acc[tgt] += cc
    return dict(acc)


def _state_1rdm(ref_pairs, nmo):
    """Spin-summed state 1-RDM ``D[p, q] = <0|E(p, q)|0>``."""
    D = np.zeros((nmo, nmo))
    for p in range(nmo):
        for q in range(nmo):
            acc = defaultdict(float)
            for I, cI in ref_pairs:
                for tgt, sgn in _e1(I, p, q):
                    acc[tgt] += cI * sgn
            D[p, q] = sum(cJ * acc.get(J, 0.0) for J, cJ in ref_pairs)
    return D


def _generalized_fock(h1e_mo, eri_mo, D):
    """State-specific generalized Fock matrix.

    ``f[p, q] = h[p, q] + sum_rs D[r, s] ((pq|rs) - (pr|qs)/2)``.  Its
    diagonal is the semicanonical orbital energy array ``eps`` when the
    reference orbitals are semicanonical; the off-diagonal blocks connect
    different orbital pools and are the source of the inter-class
    coupling in CASPT2 (the diagonal approximation drops them).
    """
    return (h1e_mo
            + np.einsum('rs,pqrs->pq', D, eri_mo)
            - 0.5 * np.einsum('rs,prqs->pq', D, eri_mo))


def _pairs_with_repetition(a, b):
    """Index pairs of one class: ordered cross products for different
    pools, unordered pairs with repetition inside one pool."""
    if b is None:
        return [(x,) for x in a]
    if a is b:
        return [(a[i], a[j]) for i in range(len(a)) for j in range(i, len(a))]
    return [(x, y) for x in a for y in b]


def _class_function_vectors(ref_pairs, ncore, nact, nmo):
    """All internally contracted functions of the eight CASPT2 classes.

    The operators follow the Molcas case structure: one function per
    unordered internal index pair (and per external pair), with the
    two-orderings combination of the doubly excited classes.  The
    relative combination is the Molcas ``mkrhsh`` one, ``VP = A + B`` and
    ``VM = A - B`` (the sqrt(3) in the source is an overall scale that
    cancels in ``W**2 / (A - E0 S)`` and in the linear solve).
    """
    nocc = ncore + nact
    core = list(range(ncore))
    act = list(range(ncore, nocc))
    vir = list(range(nocc, nmo))
    functions = []

    classes = [
        ('V0_ijab', core, core, vir, vir),
        ('Vm1_iab', act, core, vir, vir),
        ('Vm2_ab', act, act, vir, vir),
        ('V1_ija', core, core, act, vir),
        ('V2_ij', core, core, act, act),
        ('Vm1_a', act, None, vir, None),
        ('V1_i', core, None, act, None),
    ]
    for name, o1, o2, i1, i2 in classes:
        for qs in _pairs_with_repetition(o1, o2):
            for ps in _pairs_with_repetition(i1, i2):
                if len(qs) == 1:
                    functions.append(
                        (name, _apply_chain(ref_pairs,
                                            [(ps[0], qs[0])])))
                    continue
                p1, p2 = ps
                q1, q2 = qs
                ka = _apply_chain(ref_pairs, [(p1, q1), (p2, q2)])
                if p1 == p2 or q1 == q2:
                    functions.append((name, ka))
                    continue
                kb = _apply_chain(ref_pairs, [(p1, q2), (p2, q1)])
                vp = defaultdict(float)
                vm = defaultdict(float)
                for dd, cc in ka.items():
                    vp[dd] += cc
                    vm[dd] += cc
                for dd, cc in kb.items():
                    vp[dd] += cc
                    vm[dd] -= cc
                functions.append((name, dict(vp)))
                functions.append((name, dict(vm)))

    # V0_ia (Molcas case D / ORCA ITUA-ITAU): one inactive hole, one
    # virtual particle and an active-pair reorganization.  Per unordered
    # active pair (u <= v), inactive j and virtual a the two operator
    # products are independent amplitudes; for u == v they coincide and
    # only one function is kept.
    for iu, u in enumerate(act):
        for v in act[iu:]:
            modes = ('cross',) if v == u else ('cross', 'direct')
            for j in core:
                for a in vir:
                    for mode in modes:
                        ops = ([(a, j), (v, u)] if mode == 'cross'
                               else [(a, u), (v, j)])
                        functions.append(
                            ('V0_ia', _apply_chain(ref_pairs, ops)))
    return functions


def _h0_action(vec, f, nmo, det_cache):
    """Apply ``H0 = sum_pq f[p, q] E(p, q)`` to a function vector."""
    acc = defaultdict(float)
    for dd, c in vec.items():
        hmap = det_cache.get(dd)
        if hmap is None:
            m = defaultdict(float)
            for p in range(nmo):
                for q in range(nmo):
                    fpq = f[p, q]
                    if abs(fpq) > 1e-14:
                        for tgt, sgn in _e1(dd, p, q):
                            m[tgt] += fpq * sgn
            hmap = dict(m)
            det_cache[dd] = hmap
        for tgt, sgn in hmap.items():
            acc[tgt] += c * sgn
    return acc


def caspt2_contracted_e2_coupled(h1e_mo, eri_mo, ci, dets, *, ncore, nact,
                                 nele, fock=None, rtol=1e-8,
                                 return_system=False):
    """Contracted CASPT2 with the full generalized-Fock zeroth order.

    This is the coupled (inter-class) companion of
    :func:`caspt2_contracted_e2`.  The diagonal path evaluates each
    contracted function with its own denominator

        e2 = -W**2 / (A - E0 S),

    which is the CASPT2-D approximation.  The full zeroth-order
    Hamiltonian is the state-specific generalized Fock operator

        H0 = sum_pq f[p, q] E(p, q),

    whose off-diagonal blocks (inactive-active, active-virtual,
    inactive-virtual) connect determinants of different excitation
    classes.  The amplitudes then solve the projected linear system

        (A - E0 S) x = -W,
        A[p, q] = <K_p|H0|K_q>,  S[p, q] = <K_p|K_q>,
        W[p] = <K_p|H|0>,        E0 = <0|H0|0>,

    and the class decomposition of the non-variational energy is
    ``e2_class = sum_{p in class} W[p] x[p]``.  The functions are the
    eight-class internally contracted basis of
    :func:`_class_function_vectors`; ``S`` is regularly linearly
    dependent, so it is diagonalized and eigenvectors with eigenvalue
    below ``rtol * max(1, eig_max)`` are dropped (the ``NThresh`` role
    in the reference implementation).

    Parameters
    ----------
    h1e_mo, eri_mo : array
        MO one- and two-electron integrals (chemist notation).
    ci : array
        Reference CI vector over ``dets``.
    dets : sequence of int
        Determinant bitmaps (core spin orbitals first).
    ncore, nact, nele : int
        Space definition.
    fock : array, optional
        Generalized Fock matrix.  When omitted it is built from the CI
        vector and the integrals (state-specific 1-RDM).
    rtol : float
        Relative eigenvalue cutoff of the overlap matrix.
    return_system : bool
        Also return the assembled system (matrices, solution, labels) for
        diagnostics and tests.

    Returns
    -------
    (e2_total, breakdown) or (e2_total, breakdown, system)

    Notes
    -----
    This path is a numpy pilot implementation: the assembly costs
    ``nfn**2`` function-pair contractions plus one ``H0`` application per
    function, intended for small active spaces (the same regime as the
    existing contracted engine).  It has no jax twin; the differentiable
    path remains :func:`caspt2_contracted_e2_jax`.
    """
    h1e_mo = np.asarray(h1e_mo)
    eri_mo = np.asarray(eri_mo)
    ci = np.asarray(ci).ravel()
    nmo = h1e_mo.shape[0]
    if h1e_mo.shape != (nmo, nmo) or eri_mo.shape != (nmo,) * 4:
        raise ValueError('integral array shapes do not match')

    ref_pairs = [(int(d), float(c)) for d, c in zip(dets, ci)
                 if abs(c) > 1e-14]
    D = _state_1rdm(ref_pairs, nmo)
    if fock is None:
        f = _generalized_fock(h1e_mo, eri_mo, D)
    else:
        f = np.asarray(fock, dtype=float)
        if f.shape != (nmo, nmo):
            raise ValueError('fock matrix shape does not match')

    h_elem, _ = _sc_kernel(h1e_mo, eri_mo)
    g_cache = {}

    def g_of(dd):
        if dd not in g_cache:
            g_cache[dd] = sum(cJ * h_elem(J, dd) for J, cJ in ref_pairs
                              if (J ^ dd).bit_count() <= 4)
        return g_cache[dd]

    functions = _class_function_vectors(ref_pairs, ncore, nact, nmo)
    nfn = len(functions)
    labels = [name for name, _ in functions]

    S = np.zeros((nfn, nfn))
    A = np.zeros((nfn, nfn))
    W = np.zeros(nfn)
    h0_cache = {}
    h0_vec = [None] * nfn
    for i, (_, vec) in enumerate(functions):
        W[i] = sum(c * g_of(dd) for dd, c in vec.items())
        h0_vec[i] = _h0_action(vec, f, nmo, h0_cache)
    for i in range(nfn):
        vi = functions[i][1]
        for j in range(i, nfn):
            vj = functions[j][1]
            shared = vi.keys() & vj.keys()
            S[i, j] = S[j, i] = sum(vi[dd] * vj[dd] for dd in shared)
            aij = sum(vi[dd] * h0_vec[j].get(dd, 0.0) for dd in vi)
            A[i, j] = A[j, i] = aij
    S = 0.5 * (S + S.T)
    A = 0.5 * (A + A.T)

    e0 = float(np.sum(f * D))

    ev, evec = np.linalg.eigh(S)
    ev_max = max(float(ev.max()), 1.0)
    keep = ev > rtol * ev_max
    if not np.any(keep):
        raise np.linalg.LinAlgError('empty overlap spectrum')
    T = evec[:, keep] / np.sqrt(ev[keep])
    M = T.T @ (A - e0 * S) @ T
    b = -T.T @ W
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(b)):
        raise np.linalg.LinAlgError('non-finite coupled CASPT2 system')
    try:
        y = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        _log.warning('coupled CASPT2: singular reduced matrix, using '
                     'least squares')
        y = np.linalg.lstsq(M, b, rcond=1e-12)[0]
    x = T @ y
    e2 = float(W @ x)

    breakdown = {name: 0.0 for name in
                 ('V0_ijab', 'Vm1_iab', 'Vm2_ab', 'V1_ija', 'V2_ij',
                  'V0_ia', 'Vm1_a', 'V1_i')}
    for name, wi, xi in zip(labels, W, x):
        breakdown[name] += wi * xi

    if return_system:
        system = dict(S=S, A=A, W=W, E0=e0, x=x, T=T, labels=labels,
                      fock=f, dm1=D, functions=functions)
        return e2, breakdown, system
    return e2, breakdown
