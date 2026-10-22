#!/usr/bin/env python
"""Value-layer validation of the valence-level SOMF operator with ECPs.

Systems
  GeF  SA-CASSCF CAS(9,6), two roots: the two components of the 2Pi
       ground state.  The splitting is a direct SOC observable and GeF
       is the ECP analogue of the all-electron OF calibration.
  YbF  ROHF + CASCI in a fixed window (three F 2p-derived orbitals, the
       6s sigma, two virtuals).  This is a model window, not the full
       YbF spectrum; it exercises the operator in an f-zone valence
       basis and the 4f/semicore reference density.

Operator variants (all in the state-interaction layer)
  ecp        ECPso only                       H = -i sum_k T_k s_k
  somf       (a^2/2)(nuclear + h2[P]) - ECPso H = +i sum_k T_k s_k
  somf_act   same, but h2 built from the active-space density only
  nuclear    (a^2/2) nuclear only             H = +i sum_k T_k s_k
  double     ECPso + (a^2/2)(nuclear_all + h2)  (double-counting view)

The report separates the ECPso one-electron value from the valence SOMF
value; no claim of a unique physical decomposition is made.
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
PAULI = np.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)
SPIN = 0.5 * PAULI


def build_gef():
    return gto.M(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
                 basis={"Ge": "lanl2dz", "F": "6-31g"},
                 ecp={"Ge": "ecpds28mdfso"}, spin=1, unit="Bohr",
                 verbose=0)


def build_ybf():
    return gto.M(atom=[("Yb", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))],
                 basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                 ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr",
                 verbose=0)


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def frob(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def si_energies(es, T, pref):
    """State-interaction eigenvalues for scalar energies es and T[I,J,k]."""
    n = len(es)
    h = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        h[2 * i, 2 * i] = es[i]
        h[2 * i + 1, 2 * i + 1] = es[i]
        for j in range(n):
            for k in range(3):
                h[2 * i:2 * i + 2, 2 * j:2 * j + 2] += \
                    pref * T[i, j, k] * SPIN[k]
    herm = maxabs(h - h.conj().T)
    w = np.linalg.eigvalsh(h)
    return dict(levels_cm=[float((x - es[0]) * CM) for x in w],
                levels_eh=[float(x) for x in w],
                hermiticity=herm,
                splitting_cm=(float((w[2] - w[0]) * CM) if 2 * n >= 3
                              else None),
                kramers_cm=[float(abs(x) * CM) for x in
                            (w[1::2] - w[0::2])])


def t_matrix(mol, C, act, D, h):
    """T[I,J,k] = sum_pq D[I,J]_pq h^MO_k,pq on the active slice."""
    n = len(D)
    T = np.zeros((n, n, 3))
    ca = C[:, act]
    hmo = np.einsum("ap,kab,bq->kpq", ca, h, ca, optimize=True)
    for i in range(n):
        for j in range(n):
            for k in range(3):
                T[i, j, k] = float(np.sum(D[i][j] * hmo[k]))
    return T, hmo


def run_gef(contract="explicit"):
    t0 = time.time()
    mol = build_gef()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.max_cycle = 300
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
    D = {}
    for i in range(nstates):
        for j in range(nstates):
            D[(i, j)] = np.asarray(mc.fcisolver.trans_rdm1(
                ci[i], ci[j], nact, nele_act))
    Dm = [[D[(i, j)] for j in range(nstates)] for i in range(nstates)]

    # Densities: state-averaged (mc.make_rdm1) and active-only.
    p_sa = np.asarray(mc.make_rdm1(), dtype=float)
    p_sa = 0.5 * (p_sa + p_sa.T)
    ca = C[:, act]
    p_act_ao = ca @ Dm[0][0] @ ca.T
    p_act_ao = 0.5 * (p_act_ao + p_act_ao.T)

    # Operator variants.
    ecp = decp.ecp_soc(mol)
    h2_sa = dsomf.somf_two_electron(mol, p_sa, contract=contract)
    nuc_f = decp.nuclear_soc(mol, decp.nuclear_centers(mol))
    h_somf = A2 * (nuc_f + h2_sa) - ecp
    h_somf_act = A2 * (nuc_f + dsomf.somf_two_electron(
        mol, p_act_ao, contract=contract)) - ecp
    h_nuc = A2 * nuc_f
    nuc_all = decp.nuclear_soc(mol, range(mol.natm))
    h_double = A2 * (nuc_all + h2_sa) - ecp

    variants = {
        "ecp": (h_somf * 0.0 + ecp, -1j),
        "somf": (h_somf, 1j),
        "somf_active_density": (h_somf_act, 1j),
        "nuclear": (h_nuc, 1j),
        "double": (h_double, 1j),
    }
    out = dict(
        system="GeF", nao=int(mol.nao_nr()), nelectron=int(mol.nelectron),
        cas=dict(ncore=ncore, nact=nact, nele_act=int(nele_act),
                 nstates=nstates),
        cas_energy=float(mc.e_tot), converged=bool(mc.converged),
        scalar_states=[float(x) for x in es],
        scalar_gaps_cm=[float((x - es[0]) * CM) for x in es],
        density=dict(trace_ps_sa=float(np.einsum(
                         "pq,pq->", p_sa, mol.intor("int1e_ovlp"))),
                     trace_ps_act=float(np.einsum(
                         "pq,pq->", p_act_ao, mol.intor("int1e_ovlp"))),
                     e2_sa_frob=[frob(h2_sa[k]) for k in range(3)]),
        variants={})
    for name, (h, pref) in variants.items():
        T, _ = t_matrix(mol, C, act, Dm, h)
        si = si_energies(es, T, pref)
        out["variants"][name] = dict(
            prefactor=("i" if pref == 1j else "-i"),
            T=[[[float(x) for x in T[i, j]] for j in range(nstates)]
               for i in range(nstates)],
            T_norm=[float(np.linalg.norm(T[i, j]))
                    for i in range(nstates) for j in range(nstates)],
            si=si)
    out["runtime_s"] = time.time() - t0
    return out


def run_ybf(contract="direct", nroots=5):
    t0 = time.time()
    mol = build_ybf()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.max_cycle = 300
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
    h1eff, ecore = mc.get_h1eff()
    h2eff = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1eff, h2eff, ncas, nele_act,
                                    nroots=nroots)
    es = np.atleast_1d(np.asarray(e_raw, dtype=float)) + ecore
    ci = np.asarray(ci)
    C = c_ord
    act = slice(ncore, ncore + ncas)
    Dm = [[np.asarray(mc.fcisolver.trans_rdm1(ci[i], ci[j], ncas,
                                              nele_act))
           for j in range(nroots)] for i in range(nroots)]
    ca = C[:, act]
    cf = C[:, :ncore]
    p_core = 2.0 * cf @ cf.T
    p_core = 0.5 * (p_core + p_core.T)

    # The 2Pi pair: the two degenerate roots above the ground state.
    gaps = (es - es[0]) * CM
    pair = None
    best = 1e9
    for i in range(1, nroots):
        for j in range(i + 1, nroots):
            if gaps[i] > 1000.0 and gaps[j] > 1000.0:
                d = abs(gaps[i] - gaps[j])
                if d < best:
                    best = d
                    pair = (i, j)
    if pair is None or best > 1.0:
        raise RuntimeError("no degenerate excited pair found: gaps=%s"
                           % np.array2string(gaps, precision=2))
    i0, i1 = pair
    p_pair_act = 0.5 * (Dm[i0][i0] + Dm[i1][i1])
    p_pair = p_core + ca @ p_pair_act @ ca.T
    p_pair = 0.5 * (p_pair + p_pair.T)
    p_ground = p_core + ca @ Dm[0][0] @ ca.T
    p_ground = 0.5 * (p_ground + p_ground.T)
    p_sa = p_core + ca @ (sum(Dm[i][i] for i in range(nroots))
                          / nroots) @ ca.T
    p_sa = 0.5 * (p_sa + p_sa.T)

    ecp = decp.ecp_soc(mol)
    h2_pair = dsomf.somf_two_electron(mol, p_pair, contract=contract)
    h2_ground = dsomf.somf_two_electron(mol, p_ground, contract=contract)
    h2_sa = dsomf.somf_two_electron(mol, p_sa, contract=contract)
    nuc_f = decp.nuclear_soc(mol, decp.nuclear_centers(mol))
    h_somf = A2 * (nuc_f + h2_pair) - ecp
    h_somf_ground = A2 * (nuc_f + h2_ground) - ecp
    h_somf_sa = A2 * (nuc_f + h2_sa) - ecp
    h_nuc = A2 * nuc_f
    nuc_all = decp.nuclear_soc(mol, range(mol.natm))
    h_double = A2 * (nuc_all + h2_pair) - ecp
    variants = {
        "ecp": (ecp, -1j),
        "somf": (h_somf, 1j),
        "somf_ground_density": (h_somf_ground, 1j),
        "somf_sa_density": (h_somf_sa, 1j),
        "nuclear": (h_nuc, 1j),
        "double": (h_double, 1j),
    }

    def pair_block(h, pref):
        hmo = np.einsum("ap,kab,bq->kpq", ca, h, ca, optimize=True)
        T = np.zeros((2, 2, 3))
        for a in range(2):
            for b in range(2):
                d = Dm[i0 + a][i0 + b]
                for k in range(3):
                    T[a, b, k] = float(np.sum(d * hmo[k]))
        es_pair = es[[i0, i1]]
        return T, si_energies(es_pair, T, pref)

    # Sigma-Pi couplings from the ground state to each pair component.
    def cross_block(h, pref):
        out = {}
        hmo = np.einsum("ap,kab,bq->kpq", ca, h, ca, optimize=True)
        for j in (i0, i1):
            T = np.zeros(3)
            for k in range(3):
                T[k] = float(np.sum(Dm[0][j] * hmo[k]))
            out[str(j)] = [float(x) for x in T]
        return out

    out = dict(
        system="YbF", nao=int(mol.nao_nr()), nelectron=int(mol.nelectron),
        window=dict(ncore=int(ncore), ncas=int(ncas),
                    nele_act=int(nele_act), active=active,
                    nroots=int(nroots), pair=[int(i0), int(i1)],
                    note=("fixed ROHF reference, CASCI model window: three "
                          "F 2p-derived orbitals, the 6s sigma, two "
                          "virtuals; the 2Pi pair is the two degenerate "
                          "roots above the ground state")),
        cas_energy=float(es[0]), converged=True,
        scalar_states=[float(x) for x in es],
        scalar_gaps_cm=[float(x) for x in gaps],
        pair_gap_cm=float(gaps[i0]),
        density=dict(trace_ps_pair=float(np.einsum(
                         "pq,pq->", p_pair, mol.intor("int1e_ovlp"))),
                     trace_ps_ground=float(np.einsum(
                         "pq,pq->", p_ground, mol.intor("int1e_ovlp"))),
                     trace_ps_sa=float(np.einsum(
                         "pq,pq->", p_sa, mol.intor("int1e_ovlp"))),
                     e2_pair_frob=[frob(h2_pair[k]) for k in range(3)]),
        variants={})
    for name, (h, pref) in variants.items():
        T, si = pair_block(h, pref)
        out["variants"][name] = dict(
            prefactor=("i" if pref == 1j else "-i"),
            pair_T=[[[float(x) for x in T[a, b]] for b in range(2)]
                    for a in range(2)],
            pair_T_norm=float(np.linalg.norm(T[0, 1])),
            pair_si=si,
            sigma_pi=cross_block(h, pref))
    out["runtime_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ecp_somf.json")
    ap.add_argument("--systems", default="gef,ybf")
    ap.add_argument("--contract", default="explicit")
    args = ap.parse_args()
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "alpha2_half": A2,
           "definition": "valence-level SOMF; see somf_ecp module",
           "systems": {}}
    if "gef" in args.systems:
        print("== GeF ==", flush=True)
        res["systems"]["gef"] = run_gef(contract=args.contract)
    if "ybf" in args.systems:
        print("== YbF ==", flush=True)
        res["systems"]["ybf"] = run_ybf(
            contract="direct" if args.contract == "auto" else args.contract)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
