#!/usr/bin/env python
"""Determinant-space recalculation of the ECP SOMF variant table.

WHY THIS DRIVER EXISTS

``compare_variants.py`` builds every state-interaction splitting from pyscf
``trans_rdm1`` transition densities, which are spin-summed charge densities,
and from a two-root model space inside one M_S sector.  Both choices are
wrong for a spin-orbit matrix element:

* the SOC operator is spin dependent, so the matrix element needs the
  spin-resolved (determinant-level) coupling, not the charge density;
* an atomic term such as Ge+ 2P spans three spatial components, and a
  two-root subset omits the component that carries most of the L_+ S_- /
  L_- S_+ coupling.

This driver recomputes the Ge+, Ge, GeF, YbF and CeO variant tables with
the determinant-space spin-orbital CI of ``diffmrc.core.spinor_ci``.  For
every variant the SOC operator is added as a one-body spin-orbital term
(``H_SO = i sum_k h_k s_k``) and the state-interaction matrix is the
projection of the full determinant Hamiltonian onto the spin-free term.

Routes recorded per variant
  ``det_term``   determinant-space projection onto the full spin-free term
                 (all spatial components, both M_S sectors).  This is the
                 physically defined value.
  ``det_pair``   the same projection restricted to the roots of the legacy
                 model space (one M_S sector), so that the effect of the
                 model-space truncation is visible separately.
  ``legacy_pair`` the archived spin-summed assembly (pyscf ``trans_rdm1``
                 charge density plus the 2x2 Kramers tensor) on the same
                 model space and the same operator.  The ratio
                 ``legacy_pair / det_pair`` is the density-combination
                 defect at fixed operator and fixed model space.

The variant operators themselves are built by
``diffmrc.core.somf_ecp.somf_variants`` plus the free-atom subtraction of
``compare_variants.py``; this driver does not change the operator family,
only the state-interaction assembly.

Run (single thread):
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python \
      experiments/somf_ecp/recalc_variants_determinant.py \
      --systems ge_plus,ge_atom,gef,ybf,ceo
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import itertools
import json
import sys
import time

import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.data import nist

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from diffmrc.core import somf as dsomf          # noqa: E402
from diffmrc.core import somf_ecp as decp       # noqa: E402
from diffmrc.core import spinor_ci as sci       # noqa: E402
from diffmrc.core import rassi                  # noqa: E402

CM = 219474.6313702
A2 = 0.5 * float(nist.ALPHA) ** 2
SPIN_HALF = [np.array([[0, 1], [1, 0]], dtype=complex) / 2,
             np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
             np.array([[1, 0], [0, -1]], dtype=complex) / 2]


# ---------------------------------------------------------------------------
# Determinant algebra
# ---------------------------------------------------------------------------
def _annihilate(key, x):
    if not (key >> x) & 1:
        return None
    sign = -1.0 if bin(key & ((1 << x) - 1)).count("1") % 2 else 1.0
    return key ^ (1 << x), sign


def _create(key, x):
    if (key >> x) & 1:
        return None
    sign = -1.0 if bin(key & ((1 << x) - 1)).count("1") % 2 else 1.0
    return key ^ (1 << x), sign


def spin_orbital_integrals(h1, h2, norb):
    """Embed scalar active-space integrals in the spin-orbital basis.

    Same embedding as the Wave 18 benchmark and
    ``recalc_determinant_space.py``: the one-body term is spin diagonal,
    the two-body term couples equal spins inside each electron pair.
    """
    nso = 2 * norb
    h1so = np.zeros((nso, nso), dtype=complex)
    eri = np.zeros((nso, nso, nso, nso), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            h1so[2 * p, 2 * q] = h1[p, q]
            h1so[2 * p + 1, 2 * q + 1] = h1[p, q]
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    v = h2[p, q, r, s]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            eri[2 * p + sp, 2 * q + sp,
                                2 * r + sq, 2 * s + sq] = v
    return h1so, eri


def soc_one_body(hk, norb):
    """Spin-orbital matrix of ``i sum_k hk[k] s_k`` (hk already alpha^2/2)."""
    nso = 2 * norb
    op = np.zeros((nso, nso), dtype=complex)
    for k in range(3):
        for p in range(norb):
            for q in range(norb):
                if hk[k, p, q] == 0.0:
                    continue
                for a in range(2):
                    for b in range(2):
                        op[2 * p + a, 2 * q + b] += (1j * hk[k, p, q]
                                                     * SPIN_HALF[k][a, b])
    return 0.5 * (op + op.conj().T)


def transition_density_spin_summed(ci_i, ci_j, dets, ncas):
    """``D[p,q] = sum_sigma <i| a^dag_{p sigma} a_{q sigma} |j>``.

    This is what pyscf ``trans_rdm1`` returns; the driver uses it only to
    reproduce the archived assembly.
    """
    index = {sum(1 << x for x in d): k for k, d in enumerate(dets)}
    nso = 2 * ncas
    d = np.zeros((ncas, ncas), dtype=complex)
    for jcol, occ in enumerate(dets):
        cj = ci_j[jcol]
        if cj == 0.0:
            continue
        key = sum(1 << x for x in occ)
        for q_so in occ:
            step = _annihilate(key, q_so)
            if step is None:
                continue
            key1, s1 = step
            for p_so in range(nso):
                if (p_so & 1) != (q_so & 1):
                    continue
                step = _create(key1, p_so)
                if step is None:
                    continue
                key2, s2 = step
                row = index.get(key2)
                if row is None:
                    continue
                ci = ci_i[row]
                if ci == 0.0:
                    continue
                d[p_so >> 1, q_so >> 1] += np.conj(ci) * cj * s1 * s2
    return d


def spin_z_expectation(ci, dets):
    """<S_z> of a determinant-space CI vector."""
    sz = 0.0
    for k, occ in enumerate(dets):
        c = abs(ci[k]) ** 2
        if c == 0.0:
            continue
        sz += c * 0.5 * sum(1 if (x & 1) == 0 else -1 for x in occ)
    return sz


def groups_of(energies, tol=1e-6):
    """Degeneracy groups of a spectrum, tolerance in Eh."""
    order = np.argsort(energies)
    out, cur = [], [int(order[0])]
    for k in order[1:]:
        if abs(energies[k] - energies[cur[-1]]) < tol:
            cur.append(int(k))
        else:
            out.append(cur)
            cur = [int(k)]
    out.append(cur)
    return out


def legacy_kramers_levels(energies_eh, tmat, pref=1j):
    """Archived spin-summed assembly: 2x2 Kramers blocks per root pair.

    ``energies_eh`` are in Eh, ``tmat[i, j, k] = sum_pq D_ij hk_k`` in Eh.
    """
    n = len(energies_eh)
    h = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        h[2 * i:2 * i + 2, 2 * i:2 * i + 2] += energies_eh[i] * np.eye(2)
        for j in range(n):
            for k in range(3):
                h[2 * i:2 * i + 2, 2 * j:2 * j + 2] += (pref * tmat[i, j, k]
                                                        * SPIN_HALF[k])
    return np.linalg.eigvalsh(h)


def det_si_levels(vectors, energies_eh, op_det):
    """Project a determinant-space one-body operator onto CI vectors."""
    h = vectors.conj().T @ op_det @ vectors + np.diag(energies_eh)
    return np.linalg.eigvalsh(0.5 * (h + h.conj().T))


def frob(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def _sz_so(ncas):
    nso = 2 * ncas
    sz = np.zeros((nso, nso), dtype=complex)
    for p in range(ncas):
        sz[2 * p, 2 * p] = 0.5
        sz[2 * p + 1, 2 * p + 1] = -0.5
    return sz


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------
def build_gef(geometry="bent", ecp=None):
    atoms = {"bent": [("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
             "linear": [("Ge", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.4))]}
    return gto.M(atom=atoms[geometry], basis={"Ge": "lanl2dz", "F": "6-31g"},
                 ecp=ecp or {"Ge": "ecpds28mdfso"}, spin=1, unit="Bohr",
                 verbose=0)


def free_atom_h2(mol, atom_index, basis, ecp, spin, unit="Bohr"):
    """Valence mean field of the free atom in the molecular basis."""
    symbol = mol.atom_symbol(atom_index)
    last = None
    for sp in (spin, 0, 2, 4, 6):
        atom = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis,
                     ecp=ecp, spin=sp, unit=unit, verbose=0)
        mf = scf.ROHF(atom)
        mf.conv_tol = 1e-11
        mf.max_cycle = 500
        mf.verbose = 0
        try:
            mf.kernel()
        except Exception as exc:          # pragma: no cover - SCF dependent
            last = exc
            continue
        if mf.converged:
            p = np.asarray(mf.make_rdm1())
            if p.ndim == 3:
                p = p[0] + p[1]
            p = 0.5 * (p + p.T)
            return atom, p, dsomf.somf_two_electron(atom, p,
                                                    contract="explicit"), sp
        last = RuntimeError("free %s atom SCF spin=%s did not converge"
                            % (symbol, sp))
    raise RuntimeError(str(last))


def embed_atom_h2(mol, atom_index, h2_atom):
    mask = decp._center_ao_mask(mol, [atom_index])
    idx = np.where(mask)[0]
    if idx.size != h2_atom.shape[1]:
        raise RuntimeError("atomic and molecular AO block sizes differ")
    out = np.zeros((3, mol.nao_nr(), mol.nao_nr()))
    out[:, idx[:, None], idx[None, :]] = h2_atom
    return out


def variant_operators(mol, p, free_atom_args, contract="explicit"):
    """All operators of the family, in the +i convention, plus free-atom."""
    ops = decp.somf_variants(mol, p, contract=contract)
    atom, p_atom, h2_atom, spin_used = free_atom_h2(mol, 0, **free_atom_args)
    h2_atom_emb = embed_atom_h2(mol, 0, h2_atom)
    ops["free_atom_subtracted"] = (
        A2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
              + dsomf.somf_two_electron(mol, p, contract=contract)
              - h2_atom_emb) - decp.ecp_soc(mol))
    meta = dict(free_atom_spin=int(spin_used),
                free_atom_nelec=int(atom.nelectron),
                free_atom_nao=int(atom.nao_nr()),
                free_atom_h2_frob=frob(h2_atom))
    return ops, meta


# ---------------------------------------------------------------------------
# Determinant-space evaluation of a variant set
# ---------------------------------------------------------------------------
def evaluate_variants(c_act, ncas, nele_act, ops, target_group, w0, v0,
                      dets, pair_states, ecore, cm=CM):
    """State-interaction splittings for every operator variant.

    ``target_group`` indexes the spin-free term; ``pair_states`` is the
    list of state indices of the legacy model space (one M_S sector).
    """
    space = rassi.DetSpace(norb=ncas, nelec=nele_act)
    keys = [sum(1 << x for x in d) for d in dets]
    if [int(x) for x in space.masks] != keys:
        raise RuntimeError("determinant ordering mismatch between "
                           "spinor_ci and rassi")

    term = np.asarray(target_group, dtype=int)
    pair = np.asarray(pair_states, dtype=int)
    v_term = v0[:, term]
    v_pair = v0[:, pair]
    e_pair = w0[pair] + ecore

    out = {}
    for name, hk in ops.items():
        hk_mo = np.einsum("ap,kab,bq->kpq", c_act, hk, c_act, optimize=True)
        op_det = rassi.one_body_matrix(soc_one_body(hk_mo, ncas), space)

        lev_term = det_si_levels(v_term, w0[term], op_det) * cm
        lev_pair = det_si_levels(v_pair, w0[pair], op_det) * cm

        # Archived assembly on the same model space and the same operator.
        n = len(pair)
        tmat = np.zeros((n, n, 3))
        for a in range(n):
            for b in range(n):
                d = transition_density_spin_summed(
                    v0[:, pair[a]], v0[:, pair[b]], dets, ncas)
                for k in range(3):
                    tmat[a, b, k] = float(np.real(np.sum(d * hk_mo[k])))
        lev_legacy = legacy_kramers_levels(e_pair, tmat) * cm

        span_term = float(lev_term[-1] - lev_term[0])
        span_pair = float(lev_pair[-1] - lev_pair[0])
        span_legacy = float(lev_legacy[-1] - lev_legacy[0])
        out[name] = dict(
            frob=frob(hk),
            det_term_cm=span_term,
            det_term_levels_cm=[float(x - lev_term[0]) for x in lev_term],
            det_pair_cm=span_pair,
            det_pair_levels_cm=[float(x - lev_pair[0]) for x in lev_pair],
            legacy_pair_cm=span_legacy,
            legacy_pair_levels_cm=[float(x - lev_legacy[0])
                                   for x in lev_legacy],
            ratio_legacy_over_det_pair=(span_legacy / span_pair
                                        if span_pair else float("nan")),
            ratio_det_pair_over_term=(span_pair / span_term
                                      if span_term else float("nan")),
        )
    return out


def term_and_pair(h1, h2, ncas, nele_act, pair_energy_cm=None, tol=1e-6):
    """Determinant-space spin-free spectrum and the legacy model space.

    Returns ``(w0, v0, dets, ecore, groups, target, pair_states)``.  The
    target group is the one whose energy matches ``pair_energy_cm`` (the
    lowest group when that is None).  ``pair_states`` are the two states
    of the target group with the highest <S_z>.
    """
    h1so, eso = spin_orbital_integrals(np.asarray(h1), np.asarray(h2), ncas)
    h0, dets, ecore = sci.spinor_ci_matrix(h1so, eso, 0, 2 * ncas, nele_act)
    w0, v0 = np.linalg.eigh(h0)
    groups = groups_of(w0, tol=tol)

    target = None
    if pair_energy_cm is not None:
        for g in groups:
            e_rel = (w0[g[0]] - w0[0]) * CM
            if abs(e_rel - pair_energy_cm) < 1.0:
                target = g
                break
    if target is None:
        target = groups[0]

    # Resolve the M_S content of the (possibly mixed) degenerate group and
    # order its states by <S_z> so that the legacy one-sector model space
    # is well defined.
    vg = v0[:, target]
    op_sz = rassi.one_body_matrix(_sz_so(ncas),
                                  rassi.DetSpace(norb=ncas, nelec=nele_act))
    msz = vg.conj().T @ op_sz @ vg
    _, u = np.linalg.eigh(0.5 * (msz + msz.conj().T))
    vg_rot = vg @ u
    ms = np.real(np.diag(u.conj().T @ msz @ u))
    order = np.argsort(-ms)
    n_pair = int(np.count_nonzero(np.abs(ms - ms[order[0]]) < 0.1))
    n_pair = min(2, n_pair)
    v0 = v0.copy()
    v0[:, target] = vg_rot[:, order]
    # After the permutation the highest-<S_z> states sit in the first
    # ``n_pair`` columns of the term block.
    pair_states = [int(target[k]) for k in range(n_pair)]
    return w0, v0, dets, ecore, groups, target, pair_states


def term_density(v0, target, dets, ncas):
    """Spin-summed active-space density averaged over the term."""
    rdm = np.zeros((2 * ncas, 2 * ncas), dtype=complex)
    for k in target:
        rdm += sci.spinor_ci_rdm1(v0[:, k], dets, 2 * ncas)
    rdm /= len(target)
    d = np.zeros((ncas, ncas))
    for p in range(ncas):
        for q in range(ncas):
            d[p, q] = (rdm[2 * p, 2 * q] + rdm[2 * p + 1, 2 * q + 1]).real
    return 0.5 * (d + d.T)


def _group_record(groups, w0, ecore):
    return [dict(deg=len(g),
                 energy_cm=float((w0[g[0]] + ecore) * CM
                                 - (w0[0] + ecore) * CM),
                 spread_cm=float((w0[g[-1]] - w0[g[0]]) * CM))
            for g in groups[:6]]


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def run_ge_plus(contract="explicit"):
    t0 = time.time()
    mol = gto.M(atom=[("Ge", (0, 0, 0))], basis={"Ge": "lanl2dz"},
                ecp={"Ge": "ecpds28mdfso"}, charge=1, spin=1,
                unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    ncas = mol.nao_nr()
    nele = mol.nelectron
    mc = mcscf.CASCI(mf, ncas, nele)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2 = np.asarray(mc.get_h2eff())
    if h2.ndim == 2:
        h2 = ao2mo.restore(1, h2, ncas)
    e_legacy, _ = mc.fcisolver.kernel(h1, h2, ncas, (2, 1), nroots=3)
    legacy_gaps = (np.atleast_1d(e_legacy) + ecore) * CM
    c_act = np.asarray(mc.mo_coeff)

    w0, v0, dets, ecore2, groups, target, pair_states = term_and_pair(
        h1, h2, ncas, nele)
    d_act = term_density(v0, target, dets, ncas)
    p_ao = c_act @ d_act @ c_act.T
    p_ao = 0.5 * (p_ao + p_ao.T)

    ops, meta = variant_operators(
        mol, p_ao, dict(basis={"Ge": "lanl2dz"},
                        ecp={"Ge": "ecpds28mdfso"}, spin=2), contract)
    res = evaluate_variants(c_act, ncas, nele, ops, target, w0, v0, dets,
                            pair_states, ecore)

    # Every pair of the term under the ECPso operator: shows that the
    # archived two-root model space has no unique value.
    hk_mo = np.einsum("ap,kab,bq->kpq", c_act, ops["ecp"], c_act,
                      optimize=True)
    space = rassi.DetSpace(norb=ncas, nelec=nele)
    op_det = rassi.one_body_matrix(soc_one_body(hk_mo, ncas), space)
    spans = []
    for pr in itertools.combinations(target, 2):
        lev = det_si_levels(v0[:, list(pr)], w0[list(pr)], op_det) * CM
        spans.append(float(lev[-1] - lev[0]))

    # The archived 2x2 Kramers assembly over the same fifteen pairs: this
    # separates "the model space has no unique answer" from "the archived
    # formula is unstable inside a fixed model space".
    legacy_spans = []
    for pr in itertools.combinations(target, 2):
        pr = list(pr)
        tm = np.zeros((2, 2, 3))
        for a in range(2):
            for b in range(2):
                d = transition_density_spin_summed(v0[:, pr[a]], v0[:, pr[b]],
                                                   dets, ncas)
                for k in range(3):
                    tm[a, b, k] = float(np.real(np.sum(d * hk_mo[k])))
        lev = legacy_kramers_levels(w0[pr] + ecore, tm) * CM
        legacy_spans.append(float(lev[-1] - lev[0]))

    return dict(
        system="ge_plus",
        setup=dict(basis={"Ge": "lanl2dz"}, ecp={"Ge": "ecpds28mdfso"},
                   charge=1, cas=dict(ncas=int(ncas), nele=int(nele))),
        nao=int(mol.nao_nr()),
        legacy_roots_cm=[float(x - legacy_gaps[0]) for x in legacy_gaps],
        det_groups=_group_record(groups, w0, ecore),
        target_term=dict(degeneracy=int(len(target)),
                         energy_cm=float((w0[target[0]] + ecore) * CM
                                         - (w0[0] + ecore) * CM),
                         ms_of_pair=[float(spin_z_expectation(v0[:, k], dets))
                                     for k in pair_states]),
        density=dict(trace_ps=float(np.einsum(
            "pq,pq->", p_ao, mol.intor("int1e_ovlp"))), nelec=int(nele)),
        free_atom=meta,
        experiment_ge2p_cm=1767.0,
        variants=res,
        ecp_pair_spans_cm=dict(min=min(spans), max=max(spans),
                               all=[round(x, 4) for x in spans]),
        ecp_legacy_pair_spans_cm=dict(min=min(legacy_spans),
                                      max=max(legacy_spans),
                                      all=[round(x, 4) for x in legacy_spans]),
        runtime_s=time.time() - t0)


def run_ge_atom(contract="explicit"):
    t0 = time.time()
    mol = gto.M(atom=[("Ge", (0, 0, 0))], basis={"Ge": "lanl2dz"},
                ecp={"Ge": "ecpds28mdfso"}, spin=2, unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    ncas = mol.nao_nr()
    nele = mol.nelectron
    mc = mcscf.CASCI(mf, ncas, nele)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2 = np.asarray(mc.get_h2eff())
    if h2.ndim == 2:
        h2 = ao2mo.restore(1, h2, ncas)
    e_legacy, _ = mc.fcisolver.kernel(h1, h2, ncas, (3, 1), nroots=3)
    legacy_gaps = (np.atleast_1d(e_legacy) + ecore) * CM
    c_act = np.asarray(mc.mo_coeff)

    w0, v0, dets, ecore2, groups, target, pair_states = term_and_pair(
        h1, h2, ncas, nele)
    d_act = term_density(v0, target, dets, ncas)
    p_ao = c_act @ d_act @ c_act.T
    p_ao = 0.5 * (p_ao + p_ao.T)
    ops, meta = variant_operators(
        mol, p_ao, dict(basis={"Ge": "lanl2dz"},
                        ecp={"Ge": "ecpds28mdfso"}, spin=2), contract)
    res = evaluate_variants(c_act, ncas, nele, ops, target, w0, v0, dets,
                            pair_states, ecore)

    lev = np.array(res["ecp"]["det_term_levels_cm"])
    clusters = []
    for x in lev:
        if clusters and abs(x - clusters[-1][0]) < 1.0:
            clusters[-1][1] += 1
        else:
            clusters.append([float(x), 1])
    return dict(
        system="ge_atom",
        setup=dict(basis={"Ge": "lanl2dz"}, ecp={"Ge": "ecpds28mdfso"},
                   spin=2, cas=dict(ncas=int(ncas), nele=int(nele))),
        nao=int(mol.nao_nr()),
        legacy_roots_cm=[float(x - legacy_gaps[0]) for x in legacy_gaps],
        det_groups=_group_record(groups, w0, ecore),
        target_term=dict(degeneracy=int(len(target)),
                         energy_cm=float((w0[target[0]] + ecore) * CM
                                         - (w0[0] + ecore) * CM)),
        ecp_term_clusters=[[round(a, 3), b] for a, b in clusters],
        density=dict(trace_ps=float(np.einsum(
            "pq,pq->", p_ao, mol.intor("int1e_ovlp"))), nelec=int(nele)),
        free_atom=meta,
        experiment_ge3p_span_cm=1409.9609,
        experiment_ge3p_intervals_cm=[557.1341, 852.8268],
        variants=res,
        runtime_s=time.time() - t0)


def _run_molecular(name, mol, ncore, ncas, nele_act, nroots, pair_index,
                   free_atom_args, extra, contract="explicit",
                   sa_casscf_states=None):
    """Shared body for GeF / YbF."""
    t0 = time.time()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 400
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("%s ROHF did not converge" % name)
    c = np.asarray(mf.mo_coeff)
    if sa_casscf_states is None:
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.5))
        active = list(range(ncore, ncore + ncas))
        frozen = [i for i in range(nocc) if i not in set(active)]
        virt = [i for i in range(c.shape[1]) if i >= nocc
                and i not in set(active)]
        c_ord = c[:, frozen + active + virt]
        mc = mcscf.CASCI(mf, ncas, nele_act, ncore=ncore)
        mc.mo_coeff = c_ord
        mc.verbose = 0
        h1, ecore = mc.get_h1eff()
        h2 = np.asarray(mc.get_h2eff())
        if h2.ndim == 2:
            h2 = ao2mo.restore(1, h2, ncas)
        e_raw, _ = mc.fcisolver.kernel(h1, h2, ncas, nele_act, nroots=nroots)
        legacy_energies = np.atleast_1d(np.asarray(e_raw, float)) + ecore
    else:
        mc = mcscf.CASSCF(mf, ncas, nele_act)
        mc.verbose = 0
        mc.conv_tol = 1e-10
        mc.max_cycle = 200
        mc.state_average_([1.0 / sa_casscf_states] * sa_casscf_states)
        mc.fcisolver.nroots = sa_casscf_states
        mc.kernel()
        if not mc.converged:
            raise RuntimeError("%s SA-CASSCF did not converge" % name)
        c_ord = np.asarray(mc.mo_coeff)
        h1, ecore = mc.get_h1eff()
        h2 = np.asarray(mc.get_h2eff())
        if h2.ndim == 2:
            h2 = ao2mo.restore(1, h2, ncas)
        legacy_energies = np.atleast_1d(
            np.asarray(mc.fcisolver.e_states, dtype=float)) + ecore
    legacy_gaps = (legacy_energies - legacy_energies[0]) * CM
    pair_energy_cm = float(legacy_gaps[pair_index[0]])

    w0, v0, dets, ecore2, groups, target, pair_states = term_and_pair(
        h1, h2, ncas, nele_act, pair_energy_cm=pair_energy_cm)
    d_act = term_density(v0, target, dets, ncas)
    c_act = c_ord[:, ncore:ncore + ncas]
    c_core = c_ord[:, :ncore]
    p_ao = 2.0 * c_core @ c_core.T + c_act @ d_act @ c_act.T
    p_ao = 0.5 * (p_ao + p_ao.T)

    ops, meta = variant_operators(mol, p_ao, free_atom_args, contract)
    res = evaluate_variants(c_act, ncas, nele_act, ops, target, w0, v0, dets,
                            pair_states, ecore)
    out = dict(
        system=name,
        setup=dict(ncore=int(ncore), ncas=int(ncas), nele_act=int(nele_act),
                   nroots=int(nroots), sa_casscf_states=sa_casscf_states),
        nao=int(mol.nao_nr()),
        legacy_roots_cm=[float(x) for x in legacy_gaps],
        det_groups=_group_record(groups, w0, ecore),
        target_term=dict(degeneracy=int(len(target)),
                         energy_cm=float((w0[target[0]] + ecore) * CM
                                         - (w0[0] + ecore) * CM),
                         ms_of_pair=[float(spin_z_expectation(v0[:, k], dets))
                                     for k in pair_states]),
        density=dict(trace_ps=float(np.einsum(
            "pq,pq->", p_ao, mol.intor("int1e_ovlp"))),
            nelec=int(mol.nelectron)),
        free_atom=meta,
        variants=res,
        runtime_s=time.time() - t0)
    out.update(extra)
    return out


def run_gef(contract="explicit"):
    mol = build_gef("bent")
    return _run_molecular(
        "gef", mol, ncore=2, ncas=6, nele_act=mol.nelectron - 4, nroots=2,
        pair_index=(0, 1),
        free_atom_args=dict(basis={"Ge": "lanl2dz"},
                            ecp={"Ge": "ecpds28mdfso"}, spin=2),
        extra=dict(geometry="bent F=(0.5,0.3,2.4) Bohr",
                   basis={"Ge": "lanl2dz", "F": "6-31g"},
                   ecp={"Ge": "ecpds28mdfso"}),
        contract=contract, sa_casscf_states=2)


def run_ybf(contract="explicit"):
    mol = gto.M(atom=[("Yb", (0, 0, 0)), ("F", (0, 0, 3.6))],
                basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr", verbose=0)
    return _run_molecular(
        "ybf", mol, ncore=22, ncas=6, nele_act=7, nroots=5,
        pair_index=(3, 4),
        free_atom_args=dict(basis={"Yb": "stuttgart_rsc"},
                            ecp={"Yb": "ecpds28mwbso"}, spin=0),
        extra=dict(geometry="R(Yb-F)=3.6 Bohr",
                   basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                   ecp={"Yb": "ecpds28mwbso"},
                   active_window=[22, 23, 24, 25, 26, 27]),
        contract=contract)


def run_ceo(contract="explicit"):
    t0 = time.time()
    mol = gto.M(atom=[("Ce", (0, 0, 0)), ("O", (0, 0, 3.4))],
                basis={"Ce": "stuttgart_rsc", "O": "6-31g"},
                ecp={"Ce": "ecpds28mwbso"}, spin=2, unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 400
    mf.init_guess = "atom"
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("CeO ROHF did not converge")
    c = np.asarray(mf.mo_coeff)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.5))
    active = list(range(18, 27))
    frozen = [i for i in range(nocc) if i not in set(active)]
    virt = [i for i in range(c.shape[1]) if i >= nocc and i not in set(active)]
    c_ord = c[:, frozen + active + virt]
    ncore, ncas = len(frozen), len(active)
    nele_act = mol.nelectron - 2 * ncore
    mc = mcscf.CASCI(mf, ncas, nele_act, ncore=ncore)
    mc.mo_coeff = c_ord
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2 = np.asarray(mc.get_h2eff())
    if h2.ndim == 2:
        h2 = ao2mo.restore(1, h2, ncas)
    e_raw, _ = mc.fcisolver.kernel(h1, h2, ncas, nele_act, nroots=6)
    legacy_energies = np.atleast_1d(np.asarray(e_raw, float)) + ecore
    legacy_gaps = (legacy_energies - legacy_energies[0]) * CM
    pair_energy_cm = float(legacy_gaps[3])
    w0, v0, dets, ecore2, groups, target, pair_states = term_and_pair(
        h1, h2, ncas, nele_act, pair_energy_cm=pair_energy_cm)
    d_act = term_density(v0, target, dets, ncas)
    c_act = c_ord[:, ncore:ncore + ncas]
    c_core = c_ord[:, :ncore]
    p_ao = 2.0 * c_core @ c_core.T + c_act @ d_act @ c_act.T
    p_ao = 0.5 * (p_ao + p_ao.T)
    ops, meta = variant_operators(
        mol, p_ao, dict(basis={"Ce": "stuttgart_rsc"},
                        ecp={"Ce": "ecpds28mwbso"}, spin=2), contract)
    res = evaluate_variants(c_act, ncas, nele_act, ops, target, w0, v0, dets,
                            pair_states, ecore)
    return dict(
        system="ceo",
        setup=dict(ncore=int(ncore), ncas=int(ncas), nele_act=int(nele_act),
                   nroots=6),
        nao=int(mol.nao_nr()),
        legacy_roots_cm=[float(x) for x in legacy_gaps],
        det_groups=_group_record(groups, w0, ecore),
        target_term=dict(degeneracy=int(len(target)),
                         energy_cm=float((w0[target[0]] + ecore) * CM
                                         - (w0[0] + ecore) * CM)),
        density=dict(trace_ps=float(np.einsum(
            "pq,pq->", p_ao, mol.intor("int1e_ovlp"))),
            nelec=int(mol.nelectron)),
        free_atom=meta, variants=res, runtime_s=time.time() - t0)


RUNNERS = {
    "ge_plus": run_ge_plus,
    "ge_atom": run_ge_atom,
    "gef": run_gef,
    "ybf": run_ybf,
    "ceo": run_ceo,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="ge_plus,ybf")
    ap.add_argument("--out", default=os.path.join(
        HERE, "results_ecp_somf_variants_determinant.json"))
    ap.add_argument("--contract", default="explicit",
                    choices=("explicit", "auto", "direct"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "route_new": ("determinant-space spin-orbital CI "
                         "(diffmrc.core.spinor_ci), SOC as a one-body term"),
           "route_old": ("state interaction from spin-summed transition "
                         "densities and 2x2 Kramers blocks; archived"),
           "units": "cm^-1",
           "alpha2_half": A2,
           "pyscf": pyscf_version,
           "numpy": np.__version__,
           "contract": args.contract,
           "systems": {}}
    for name in args.systems.split(","):
        name = name.strip()
        if not name:
            continue
        print("== %s ==" % name, flush=True)
        res["systems"][name] = RUNNERS[name](contract=args.contract)
        print(json.dumps(res["systems"][name], indent=1), flush=True)
    res["runtime_s"] = time.time() - t0
    out = os.path.abspath(args.out)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
