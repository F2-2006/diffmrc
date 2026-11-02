#!/usr/bin/env python
"""ECP SOMF variant comparison and the free-atom double-counting correction.

Systems
  ge_atom   neutral Ge, CAS(4,4) over 4s4p, M_S = 1 sector.  The three 3P
            spatial components give the atomic fine structure in the
            p^2 coupling.  Because the p^2 effective spin-orbit constant
            is A = zeta/2, this system calibrates the sign and scale of
            the ECPso operator rather than providing a one-line
            comparison with the observed intervals.
  ge_plus   Ge+, CAS(3,4), the one-electron 2P term.  Its fine structure
            is the clean one-electron calibration: the observed Ge II
            2P splitting is 1767 cm^-1.
  gef       GeF, SA-CASSCF CAS(9,6), two roots; the ECP analogue of the
            all-electron OF calibration.
  ybf       YbF, fixed ROHF reference plus a CASCI window (the same
            window as the earlier value-layer report).  The 2Pi pair is
            the two degenerate roots above the ground state.
  ceo       CeO, fixed ROHF triplet reference plus a CASCI window.  The
            active space is the two singly occupied Ce orbitals and the
            low-lying virtuals; the reported pair is a model-window
            degeneracy, not a physical CeO excitation energy.

Variants
  ecp, nuclear_non_ecp, breit_pauli, valence, one_center,
  atomic_subtracted, point_charge, double
  All operators are returned in the +i convention by
  ``diffmrc.core.somf_ecp.somf_variants`` and the state-interaction
  matrix is assembled with the same +i prefactor.

Free-atom subtraction
  For every system with an SO-ECP center, the free atom is computed in
  the same basis and ECP, its spin-summed density is used to build the
  valence mean field, and that matrix is embedded in the molecular
  SO-ECP AO block.  The ``free_atom_subtracted`` operator is

      ECPso + (a^2/2)[nuclear_non_ecp + h2[molecule] - h2[free atom]].

  On the free atom it reduces to ECPso by construction.  It is the
  variant that assumes the ECP already contains the free-atom valence
  mean field.  ``atomic_subtracted`` (the on-site block of the molecular
  density) is the cheaper approximation to it.

The output JSON records every operator's Frobenius norm, the selected
pair's transition tensor, and the state-interaction level splitting.

WARNING (2026-11-02): every state-interaction splitting here is assembled
from pyscf ``trans_rdm1`` spin-summed transition densities; see the
warning in ``validate_ecp_somf.py``.  The defect is operator- and
geometry-dependent (measured on the GeF bent setup: 1.0235 for ECPso,
1.0250 for the valence SOMF operator, 1.0912 for nuclear-only, 1.0848 for
two-electron-only).  The variant ordering and the free-atom-subtraction
construction are unaffected in structure, but the absolute splittings and
the Ge+ 2P calibration offset quoted in the README (11.6%) must be
re-derived with ``recalc_determinant_space.py`` before they are cited.
No archived JSON of this script is distributed; the printed numbers are
diagnostic only.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import sys
import time

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.data import nist

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from diffmrc.core import somf as dsomf          # noqa: E402
from diffmrc.core import somf_ecp as decp       # noqa: E402

CM = 219474.6313702
A2 = 0.5 * float(nist.ALPHA) ** 2
SPIN_HALF = [np.array([[0, 1], [1, 0]], dtype=complex) / 2,
             np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
             np.array([[1, 0], [0, -1]], dtype=complex) / 2]


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def frob(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def density_sum(mf):
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    return 0.5 * (dm + dm.T)


def free_atom_h2(mol, atom_index, basis, ecp, spin, unit="Bohr"):
    """Valence mean field of the free atom in the molecular basis.

    The requested spin is tried first; open-shell f atoms can fail to
    converge in a single ROHF run, so a small ladder of multiplicities is
    tried before giving up.  The caller records which one was used.
    """
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
        except Exception as exc:  # pragma: no cover - SCF dependent
            last = exc
            continue
        if mf.converged:
            p = density_sum(mf)
            h2 = dsomf.somf_two_electron(atom, p, contract="explicit")
            return atom, mf, h2, sp
        last = RuntimeError(f"free {symbol} atom SCF spin={sp} did not converge")
    raise RuntimeError(str(last))


def embed_atom_h2(mol, atom_index, h2_atom):
    mask = decp._center_ao_mask(mol, [atom_index])
    idx = np.where(mask)[0]
    if idx.size != h2_atom.shape[1]:
        raise RuntimeError("atomic and molecular AO block sizes differ")
    out = np.zeros((3, mol.nao_nr(), mol.nao_nr()))
    out[:, idx[:, None], idx[None, :]] = h2_atom
    return out


def si_pair(es, T, pref):
    """2x2 Kramers-pair state interaction for a doublet pair."""
    n = len(es)
    h = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        h[2 * i:2 * i + 2, 2 * i:2 * i + 2] += es[i] * np.eye(2)
        for j in range(n):
            for k in range(3):
                h[2 * i:2 * i + 2, 2 * j:2 * j + 2] += (
                    pref * T[i, j, k] * SPIN_HALF[k])
    w = np.linalg.eigvalsh(h)
    return w


def pair_blocks(mol, C, act, D, pair, es, variants):
    ca = C[:, act]
    i0, i1 = pair
    out = {}
    for name, h in variants.items():
        hmo = np.einsum("ap,kab,bq->kpq", ca, h, ca, optimize=True)
        T = np.zeros((2, 2, 3))
        for a in range(2):
            for b in range(2):
                d = D[i0 + a][i0 + b]
                for k in range(3):
                    T[a, b, k] = float(np.sum(d * hmo[k]))
        w = si_pair(es[[i0, i1]], T, 1j)
        out[name] = dict(
            frob=frob(h),
            T_norm=float(np.linalg.norm(T[0, 1])),
            levels_cm=[float((x - w[0]) * CM) for x in w],
            splitting_cm=float((w[2] - w[0]) * CM),
        )
    return out


def run_ge_atom():
    t0 = time.time()
    mol = gto.M(atom=[("Ge", (0, 0, 0))], basis={"Ge": "lanl2dz"},
                ecp={"Ge": "ecpds28mdfso"}, spin=2, unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    ncas = mol.nao_nr()
    mc = mcscf.CASCI(mf, ncas, 4)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1, h2e, ncas, (3, 1), nroots=3)
    es = np.atleast_1d(e_raw) + ecore
    C = np.asarray(mc.mo_coeff)
    D = {}
    for i in range(3):
        for j in range(3):
            D[(i, j)] = np.asarray(
                mc.fcisolver.trans_rdm1(ci[i], ci[j], ncas, (3, 1)))
    p = sum(D[(i, i)] for i in range(3)) / 3.0
    p_ao = C @ p @ C.T
    p_ao = 0.5 * (p_ao + p_ao.T)
    variants = decp.somf_variants(mol, p_ao, contract="explicit")
    _, _, h2_atom, _ = free_atom_h2(mol, 0, {"Ge": "lanl2dz"},
                                        {"Ge": "ecpds28mdfso"}, 2)
    # For the single atom the free-atom matrix is h2_atom itself.
    variants["free_atom_subtracted"] = (
        A2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
              + dsomf.somf_two_electron(mol, p_ao, contract="explicit")
              - h2_atom) - decp.ecp_soc(mol))
    # p^2 3P: the one-particle spin-orbit constant of the term is A = zeta/2.
    levels = {}
    for name, h in variants.items():
        hmo = np.einsum("ap,kab,bq->kpq", C, h, C, optimize=True)
        T = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T[i, j, k] = float(np.sum(D[(i, j)] * hmo[k]))
        levels[name] = dict(frob=frob(h), T_norm=float(np.linalg.norm(T)))
    return dict(
        system="ge_atom",
        nao=int(mol.nao_nr()),
        nelectron=int(mol.nelectron),
        scalar_gaps_cm=[float(x) for x in (es - es[0]) * CM],
        variants=levels,
        note=("p^2 3P; the many-electron effective constant is A = zeta/2, "
              "so the raw T norm is not compared one-to-one with the "
              "observed intervals"),
        runtime_s=time.time() - t0,
    )


def run_ge_plus():
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
    mc = mcscf.CASCI(mf, ncas, mol.nelectron)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1, h2e, ncas, (2, 1), nroots=2)
    es = np.atleast_1d(e_raw) + ecore
    C = np.asarray(mc.mo_coeff)
    D = {}
    for i in range(2):
        for j in range(2):
            D[(i, j)] = np.asarray(
                mc.fcisolver.trans_rdm1(ci[i], ci[j], ncas, (2, 1)))
    p = sum(D[(i, i)] for i in range(2)) / 2.0
    p_ao = C @ p @ C.T
    p_ao = 0.5 * (p_ao + p_ao.T)
    variants = decp.somf_variants(mol, p_ao, contract="explicit")
    _, _, h2_atom, _ = free_atom_h2(mol, 0, {"Ge": "lanl2dz"},
                                        {"Ge": "ecpds28mdfso"}, 2)
    variants["free_atom_subtracted"] = (
        A2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
              + dsomf.somf_two_electron(mol, p_ao, contract="explicit")
              - h2_atom) - decp.ecp_soc(mol))
    out = {}
    for name, h in variants.items():
        hmo = np.einsum("ap,kab,bq->kpq", C, h, C, optimize=True)
        T = np.zeros((2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i, j, k] = float(np.sum(D[(i, j)] * hmo[k]))
        w = si_pair(es, T, 1j)
        out[name] = dict(frob=frob(h), T_norm=float(np.linalg.norm(T[0, 1])),
                         splitting_cm=float((w[2] - w[0]) * CM))
    return dict(
        system="ge_plus",
        nao=int(mol.nao_nr()),
        nelectron=int(mol.nelectron),
        scalar_gaps_cm=[float(x) for x in (es - es[0]) * CM],
        variants=out,
        experiment_ge2p_cm=1767.0,
        note=("one-electron 2P calibration; the observed Ge II 2P "
              "splitting is about 1767 cm^-1"),
        runtime_s=time.time() - t0,
    )


def run_gef():
    t0 = time.time()
    mol = gto.M(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
                basis={"Ge": "lanl2dz", "F": "6-31g"},
                ecp={"Ge": "ecpds28mdfso"}, spin=1, unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    ncore, nact, nstates = 2, 6, 2
    nele_act = mol.nelectron - 2 * ncore
    mc = mcscf.CASSCF(mf, nact, nele_act)
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([1.0 / nstates] * nstates)
    mc.fcisolver.nroots = nstates
    mc.kernel()
    C = np.asarray(mc.mo_coeff)
    act = slice(ncore, ncore + nact)
    es = np.atleast_1d(np.asarray(mc.fcisolver.e_states, dtype=float))
    ci = np.asarray(mc.ci)
    D = [[np.asarray(mc.fcisolver.trans_rdm1(ci[i], ci[j], nact, nele_act))
          for j in range(nstates)] for i in range(nstates)]
    p = np.asarray(mc.make_rdm1(), dtype=float)
    p = 0.5 * (p + p.T)
    variants = decp.somf_variants(mol, p, contract="explicit")
    _, _, h2_atom, _ = free_atom_h2(mol, 0, {"Ge": "lanl2dz"},
                                        {"Ge": "ecpds28mdfso"}, 2)
    h2_atom_emb = embed_atom_h2(mol, 0, h2_atom)
    variants["free_atom_subtracted"] = (
        A2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
              + dsomf.somf_two_electron(mol, p, contract="explicit")
              - h2_atom_emb) - decp.ecp_soc(mol))
    blocks = pair_blocks(mol, C, act, D, (0, 1), es, variants)
    return dict(
        system="gef",
        nao=int(mol.nao_nr()),
        nelectron=int(mol.nelectron),
        cas=dict(ncore=ncore, nact=nact, nele_act=int(nele_act),
                 nstates=nstates),
        cas_energy=float(mc.e_tot),
        scalar_gaps_cm=[float(x) for x in (es - es[0]) * CM],
        variants=blocks,
        runtime_s=time.time() - t0,
    )


def run_ybf():
    t0 = time.time()
    mol = gto.M(atom=[("Yb", (0, 0, 0)), ("F", (0, 0, 3.6))],
                basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    c = np.asarray(mf.mo_coeff)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.5))
    active = [22, 23, 24, 25, 26, 27]
    frozen = [i for i in range(nocc) if i not in set(active)]
    virt = [i for i in range(c.shape[1])
            if i >= nocc and i not in set(active)]
    c_ord = c[:, frozen + active + virt]
    ncore = len(frozen)
    ncas = len(active)
    nele_act = mol.nelectron - 2 * ncore
    mc = mcscf.CASCI(mf, ncas, nele_act, ncore=ncore)
    mc.mo_coeff = c_ord
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1, h2e, ncas, nele_act, nroots=5)
    es = np.atleast_1d(np.asarray(e_raw, float)) + ecore
    D = [[np.asarray(mc.fcisolver.trans_rdm1(ci[i], ci[j], ncas, nele_act))
          for j in range(5)] for i in range(5)]
    ca = c_ord[:, ncore:ncore + ncas]
    cf = c_ord[:, :ncore]
    p_core = 2.0 * cf @ cf.T
    p_core = 0.5 * (p_core + p_core.T)
    gaps = (es - es[0]) * CM
    pair, best = None, 1e9
    for i in range(1, 5):
        for j in range(i + 1, 5):
            if gaps[i] > 1000.0 and gaps[j] > 1000.0:
                d = abs(gaps[i] - gaps[j])
                if d < best:
                    best, pair = d, (i, j)
    if pair is None or best > 1.0:
        raise RuntimeError("no degenerate pair in the YbF window")
    i0, i1 = pair
    p_pair = p_core + ca @ (0.5 * (D[i0][i0] + D[i1][i1])) @ ca.T
    p_pair = 0.5 * (p_pair + p_pair.T)
    variants = decp.somf_variants(mol, p_pair, contract="direct")
    _, _, h2_atom, _ = free_atom_h2(mol, 0, {"Yb": "stuttgart_rsc"},
                                        {"Yb": "ecpds28mwbso"}, 0)
    h2_atom_emb = embed_atom_h2(mol, 0, h2_atom)
    variants["free_atom_subtracted"] = (
        A2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
              + dsomf.somf_two_electron(mol, p_pair, contract="direct")
              - h2_atom_emb) - decp.ecp_soc(mol))
    blocks = pair_blocks(mol, c_ord, slice(ncore, ncore + ncas), D, pair,
                         es, variants)
    return dict(
        system="ybf",
        nao=int(mol.nao_nr()),
        nelectron=int(mol.nelectron),
        window=dict(ncore=int(ncore), ncas=int(ncas),
                    nele_act=int(nele_act), active=active,
                    pair=[int(i0), int(i1)]),
        cas_energy=float(es[0]),
        scalar_gaps_cm=[float(x) for x in gaps],
        pair_gap_cm=float(gaps[i0]),
        variants=blocks,
        runtime_s=time.time() - t0,
    )


def run_ceo():
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
    virt = [i for i in range(c.shape[1])
            if i >= nocc and i not in set(active)]
    c_ord = c[:, frozen + active + virt]
    ncore = len(frozen)
    ncas = len(active)
    nele_act = mol.nelectron - 2 * ncore
    mc = mcscf.CASCI(mf, ncas, nele_act, ncore=ncore)
    mc.mo_coeff = c_ord
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1, h2e, ncas, nele_act, nroots=6)
    es = np.atleast_1d(np.asarray(e_raw, float)) + ecore
    D = [[np.asarray(mc.fcisolver.trans_rdm1(ci[i], ci[j], ncas, nele_act))
          for j in range(6)] for i in range(6)]
    ca = c_ord[:, ncore:ncore + ncas]
    cf = c_ord[:, :ncore]
    p_core = 2.0 * cf @ cf.T
    p_core = 0.5 * (p_core + p_core.T)
    gaps = (es - es[0]) * CM
    pair, best = None, 1e9
    for i in range(1, 6):
        for j in range(i + 1, 6):
            if gaps[i] > 100.0 and gaps[j] > 100.0:
                d = abs(gaps[i] - gaps[j])
                if d < best:
                    best, pair = d, (i, j)
    if pair is None or best > 5.0:
        raise RuntimeError("no degenerate pair in the CeO window")
    i0, i1 = pair
    p_pair = p_core + ca @ (0.5 * (D[i0][i0] + D[i1][i1])) @ ca.T
    p_pair = 0.5 * (p_pair + p_pair.T)
    variants = decp.somf_variants(mol, p_pair, contract="direct")
    atom_note = None
    try:
        _, _, h2_atom, atom_spin = free_atom_h2(
            mol, 0, {"Ce": "stuttgart_rsc"}, {"Ce": "ecpds28mwbso"}, 2)
        h2_atom_emb = embed_atom_h2(mol, 0, h2_atom)
        variants["free_atom_subtracted"] = (
            A2 * (decp.nuclear_soc(mol, decp.nuclear_centers(mol))
                  + dsomf.somf_two_electron(mol, p_pair, contract="direct")
                  - h2_atom_emb) - decp.ecp_soc(mol))
        atom_note = {"free_atom_spin": int(atom_spin)}
    except RuntimeError as exc:
        atom_note = {"free_atom_skipped": str(exc)}
    blocks = pair_blocks(mol, c_ord, slice(ncore, ncore + ncas), D, pair,
                         es, variants)
    return dict(
        system="ceo",
        nao=int(mol.nao_nr()),
        nelectron=int(mol.nelectron),
        window=dict(ncore=int(ncore), ncas=int(ncas),
                    nele_act=int(nele_act), active=active,
                    pair=[int(i0), int(i1)]),
        cas_energy=float(es[0]),
        scalar_gaps_cm=[float(x) for x in gaps],
        pair_gap_cm=float(gaps[i0]),
        variants=blocks,
        free_atom_note=atom_note,
        runtime_s=time.time() - t0,
    )


RUNNERS = {
    "ge_atom": run_ge_atom,
    "ge_plus": run_ge_plus,
    "gef": run_gef,
    "ybf": run_ybf,
    "ceo": run_ceo,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default=",".join(RUNNERS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = {}
    for name in args.systems.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"== {name}", flush=True)
        out[name] = RUNNERS[name]()
        print(json.dumps(out[name], indent=2), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)


if __name__ == "__main__":
    main()
