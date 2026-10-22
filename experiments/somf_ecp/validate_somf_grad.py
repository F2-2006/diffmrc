#!/usr/bin/env python
"""SOC-gradient prototype for the valence-level SOMF operator with an ECP.

The target is the SOC vector

    T_k(R) = sum_pq D01_pq(R) h_k,pq(R),
    h = (a^2/2) (nuclear + h2[P]) - ECPso,

where D01 is the transition 1-RDM of the two lowest scalar states (the
2Pi pair of GeF), P is the state-averaged reference density, and the
scalar states come from the fixed-reference model in
``soc_grad_model.py``.

Derivative chain (R = one nuclear coordinate):

    dT = sum D01 [ (a^2/2)(d nuclear + dh2|_P) - d ECPso ]
       + sum dD01 h
       + (a^2/2) sum M dP

with

    M_rs = sum_pq D01_pq K[p,q,r,s]
         = somf_density_response(mol, D01_ao)

the density-response kernel.  The kernel is not symmetric under
(pq) <-> (rs), so M is not ``h2[D01]``.  The three integral derivatives
(d ECPso, d nuclear, dh2|_P) are central differences at the integral
layer because no analytic derivative kernels exist for the ECP spin-orbit
operator or for ``int2e_p1vxp1``.  Everything else (dC from Loewdin
re-orthonormalization, dD01 and dP from first-order CI response) is
analytic.

Validation: the analytic chain is compared with a full-chain central
difference in which the scalar CI, the transition and state densities,
the reference density, and the whole SOC operator are recomputed at
R +/- delta.

Boundaries: fixed reference orbitals, no variational orbital relaxation,
small active space, one geometric direction at a time.  This is a
prototype, not a production gradient.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import os as _os
import sys
import time

import numpy as np
from pyscf import scf
from pyscf.data import nist

HERE = _os.path.dirname(_os.path.abspath(__file__))
SRC = _os.path.abspath(_os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import soc_grad_model as gm                      # noqa: E402
from diffmrc.core import somf as dsomf           # noqa: E402
from diffmrc.core import somf_ecp as decp        # noqa: E402

A2 = 0.5 * float(nist.ALPHA) ** 2
CM = 219474.6313702
ATOM = [("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))]
BASIS = {"Ge": "lanl2dz", "F": "6-31g"}
ECP = {"Ge": "ecpds28mdfso"}
SPIN = 1
NCORE, NACT = 2, 6
DISPLACED = 1          # F center
H_INT = 1e-5           # integral-layer step, Bohr


def build(disp=None, axis=0, delta=0.0):
    d = None if disp is None else (disp, axis, delta)
    return gm.build_mol(ATOM, BASIS, ECP, SPIN, disp=d)


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def vec_rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def somf_operator(mol, p_ao):
    """Valence SOMF operator in the +i convention, AO basis."""
    ecp = decp.ecp_soc(mol)
    nuc = decp.nuclear_soc(mol, decp.nuclear_centers(mol))
    h2 = dsomf.somf_two_electron(mol, p_ao, contract="explicit")
    return A2 * (nuc + h2) - ecp, dict(ecp=ecp, nuc=nuc, h2=h2)


def t_vector(mol, ci, c_act, d01_act, h_ao):
    hmo = np.einsum("ap,kab,bq->kpq", c_act, h_ao, c_act, optimize=True)
    return np.array([float(np.sum(d01_act * hmo[k])) for k in range(3)])


def full_chain(disp_atom, axis, delta, cref, ci, v_ref):
    mol = build(disp_atom, axis, delta)
    sol = gm.solve_scalar(mol, cref, ci, phase_ref=v_ref)
    dm = gm.state_rdms(sol, ci, nstates=2)
    p_ao, _ = gm.active_density(sol, ci, dm)
    h, pieces = somf_operator(mol, p_ao)
    c = sol["c"]
    ca = c[:, NCORE:NCORE + NACT]
    t = t_vector(mol, ci, ca, dm[0][1], h)
    return t, sol, dm, p_ao, h, pieces


def analytic_chain(sol0, dm0, p_ao0, h0, pieces0, ci, axis):
    mol = sol0["mol"]
    c = sol0["c"]
    cf = c[:, :NCORE]
    ca = c[:, NCORE:NCORE + NACT]
    d01 = dm0[0][1]
    d01_ao = ca @ d01 @ ca.T   # transition density: do NOT symmetrize

    # --- scalar response: dC, dH, first-order CI coefficients ----------
    ds, dh1, deri = gm.ao_derivatives_ecp(mol, DISPLACED, axis)
    dc = gm.lowdin_basis_deriv(sol0["cref"], sol0["s"], ds)
    dh1m = dc.T @ sol0["h1"] @ c + c.T @ dh1 @ c + c.T @ sol0["h1"] @ dc
    derm = gm.eri_transform_deriv(c, dc, sol0["eri"], deri)
    dh1e = ci.effective_h1_deriv(dh1m, derm)
    deria = ci.active_eri(derm)
    ccoef = gm.ci_response(sol0, ci, dh1e, deria, ntracked=2)

    d_all_0 = gm.trans_all(sol0, ci, 0)
    d_all_1 = gm.trans_all(sol0, ci, 1)
    dd01 = np.zeros((NACT, NACT))
    for j in range(ci.ndet):
        if ccoef[0, j] != 0.0:
            dd01 += ccoef[0, j] * d_all_1[j]
        if ccoef[1, j] != 0.0:
            # trans_all(k) returns D_{j,k}; the second sum needs D_{0,j}.
            dd01 += ccoef[1, j] * d_all_0[j].T
    dp0 = gm.density_response(sol0, ci, ccoef, 0, d_all_0)
    dp1 = gm.density_response(sol0, ci, ccoef, 1, d_all_1)
    dp_act = 0.5 * (dp0 + dp1)

    dc_f = dc[:, :NCORE]
    dc_a = dc[:, NCORE:NCORE + NACT]
    p_act = 0.5 * (dm0[0][0] + dm0[1][1])
    dp_ao = 2.0 * (dc_f @ cf.T + cf @ dc_f.T)
    dp_ao += dc_a @ p_act @ ca.T
    dp_ao += ca @ dp_act @ ca.T
    dp_ao += ca @ p_act @ dc_a.T
    dp_ao = 0.5 * (dp_ao + dp_ao.T)

    # --- integral-layer derivatives ------------------------------------
    molp = build(DISPLACED, axis, +H_INT)
    molm = build(DISPLACED, axis, -H_INT)
    decp_d = (decp.ecp_soc(molp) - decp.ecp_soc(molm)) / (2 * H_INT)
    dnuc_d = (decp.nuclear_soc(molp, decp.nuclear_centers(molp))
              - decp.nuclear_soc(molm, decp.nuclear_centers(molm)))         / (2 * H_INT)
    h2p = dsomf.somf_two_electron(molp, p_ao0, contract="explicit")
    h2m = dsomf.somf_two_electron(molm, p_ao0, contract="explicit")
    dh2_d = (h2p - h2m) / (2 * H_INT)
    dh_int = A2 * (dnuc_d + dh2_d) - decp_d

    # --- density-response matrix ---------------------------------------
    m_resp = decp.somf_density_response(mol, d01_ao, contract="explicit")

    # --- MO-basis assembly (same structure as the validated chain) -----
    h_act = np.einsum("ap,kab,bq->kpq", ca, h0, ca, optimize=True)
    dh_act = (np.einsum("ap,kab,bq->kpq", dc_a, h0, ca, optimize=True)
              + np.einsum("ap,kab,bq->kpq", ca, dh_int, ca, optimize=True)
              + np.einsum("ap,kab,bq->kpq", ca, h0, dc_a, optimize=True))
    term_int = np.array([float(np.sum(d01 * dh_act[k])) for k in range(3)])
    term_dd = np.array([float(np.sum(dd01 * h_act[k])) for k in range(3)])
    term_dp = A2 * np.einsum("pq,kpq->k", dp_ao, m_resp)
    d_t = term_int + term_dd + term_dp
    d_t_nodp = term_int + term_dd
    return dict(dT=d_t, dT_no_response=d_t_nodp,
                term_integral=term_int, term_dD=term_dd,
                term_density_response=term_dp,
                d01_ao=d01_ao, dd01=dd01, dp_ao=dp_ao,
                dd_ao=np.zeros_like(h0),
                response_matrix_norm=maxabs(m_resp),
                ccoef_norm=float(np.abs(ccoef).max()),
                dc_norm=maxabs(dc))


def main():
    global DISPLACED
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="somf_grad.json")
    ap.add_argument("--deltas", default="1e-2,1e-3,1e-4")
    ap.add_argument("--atom", type=int, default=DISPLACED)
    args = ap.parse_args()
    DISPLACED = int(args.atom)
    t0 = time.time()

    mol0 = build()
    mf = scf.ROHF(mol0)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.max_cycle = 300
    mf.kernel()
    cref = np.asarray(mf.mo_coeff)
    ci = gm.CIModel(NCORE, NACT, mol0.nelectron - 2 * NCORE, SPIN)
    sol0 = gm.solve_scalar(mol0, cref, ci)
    dm0 = gm.state_rdms(sol0, ci, nstates=2)
    p_ao0, p_act0 = gm.active_density(sol0, ci, dm0)
    h0, pieces0 = somf_operator(mol0, p_ao0)
    c0 = sol0["c"]
    ca0 = c0[:, NCORE:NCORE + NACT]
    t0vec = t_vector(mol0, ci, ca0, dm0[0][1], h0)
    print("nao=%d ndet=%d E0=%.10f E1=%.10f gap_cm=%.6f"
          % (mol0.nao_nr(), ci.ndet, sol0["e"][0], sol0["e"][1],
             (sol0["e"][1] - sol0["e"][0]) * CM), flush=True)
    print("T0=%s |T0|=%.6e split_cm=%.4f"
          % (np.array2string(t0vec, precision=8),
             float(np.linalg.norm(t0vec)),
             float(np.linalg.norm(t0vec) * CM)), flush=True)
    print("density trace(PS)=%.9f  h2 frob=%s"
          % (float(np.einsum("pq,pq->", p_ao0,
                             mol0.intor("int1e_ovlp"))),
             [float(np.linalg.norm(pieces0["h2"][k])) for k in range(3)]),
          flush=True)

    deltas = [float(x) for x in args.deltas.split(",") if x]
    out = dict(system="GeF", displaced_atom=int(DISPLACED),
               nao=int(mol0.nao_nr()), ndet=int(ci.ndet),
               t0=[float(x) for x in t0vec],
               t0_norm=float(np.linalg.norm(t0vec)),
               t0_split_cm=float(np.linalg.norm(t0vec) * CM),
               gap_cm=float((sol0["e"][1] - sol0["e"][0]) * CM),
               deltas=deltas, h_int=H_INT, directions={})

    for axis in range(3):
        row = analytic_chain(sol0, dm0, p_ao0, h0, pieces0, ci, axis)
        print("\naxis %d analytic: dT=%s"
              % (axis, np.array2string(row["dT"], precision=8)), flush=True)
        print("  terms integral=%s dD=%s dP=%s"
              % (np.array2string(row["term_integral"], precision=8),
                 np.array2string(row["term_dD"], precision=8),
                 np.array2string(row["term_density_response"],
                                 precision=8)), flush=True)
        scans = []
        for delta in deltas:
            tp, solp, dmp, pp, hp, _ = full_chain(DISPLACED, axis, +delta,
                                                  cref, ci, sol0["v"][:, :2])
            tm, solm, dmm, pm, hm, _ = full_chain(DISPLACED, axis, -delta,
                                                  cref, ci, sol0["v"][:, :2])
            fd = (tp - tm) / (2 * delta)
            scans.append(dict(
                delta=float(delta), fd=[float(x) for x in fd],
                rel=vec_rel(row["dT"], fd),
                rel_no_response=vec_rel(row["dT_no_response"], fd),
                abs_dev=[float(x) for x in np.abs(row["dT"] - fd)],
                fd_norm=float(np.linalg.norm(fd))))
            print("  delta %.0e fd=%s rel=%.3e rel_no_resp=%.3e"
                  % (delta, np.array2string(fd, precision=8),
                     scans[-1]["rel"], scans[-1]["rel_no_response"]),
                  flush=True)
        out["directions"][str(axis)] = dict(
            analytic=[float(x) for x in row["dT"]],
            analytic_no_response=[float(x) for x in row["dT_no_response"]],
            terms=dict(integral=[float(x) for x in row["term_integral"]],
                       dD=[float(x) for x in row["term_dD"]],
                       density_response=[float(x) for x in
                                         row["term_density_response"]]),
            response_matrix_norm=float(row["response_matrix_norm"]),
            dc_norm=float(row["dc_norm"]),
            scans=scans,
            max_rel=float(min(s["rel"] for s in scans)),
            max_rel_no_response=float(min(s["rel_no_response"]
                                          for s in scans)))
    out["runtime_s"] = time.time() - t0
    path = _os.path.abspath(args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
