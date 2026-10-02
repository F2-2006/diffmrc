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
import os  # noqa: E402

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp  # noqa: E402

import numpy as np  # noqa: E402

__all__ = ['caspt2_double_external_e2', 'caspt2_full_e2']


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
    if os.environ.get('CASPT2_DEBUG'):
        print(f'[caspt2] dets_full={len(dets_full)} idx0={len(idx0)} '
              f'ref_in_block={len(ref_idx & set(idx0))} '
              f'vec0_nnz={int(np.count_nonzero(vec0))} '
              f'|H|max={float(np.abs(np.asarray(H)).max()):.3e} '
              f'|v|max={float(np.abs(v).max()):.3e}', flush=True)

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
    if os.environ.get('CASPT2_DEBUG'):
        diag.sort(reverse=True)
        print(f'[caspt2] e0_zero={e0_zero:.6f}')
        for a, c, vv, dn, d, h, nv in diag[:5]:
            print(f'[caspt2] top: contrib={c:+.6f} v={vv:+.6f} '
                  f'denom={dn:+.6f} det={d} holes={h} n_vir={nv}')
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
