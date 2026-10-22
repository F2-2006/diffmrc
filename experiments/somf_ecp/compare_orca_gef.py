#!/usr/bin/env python
"""ORCA cross-check of the two-electron SOMF contraction in an ECP basis.

ORCA 6.1.1 accepts a scalar ECP and can build either the effective
nuclear-charge operator (SOCType 1) or the SOMF operator (SOCType 3)
from it.  The built-in ECP library has no spin-orbit coefficients, so
this is not a check of the ECPso term; it is a check of the two pieces
that are shared with the valence SOMF definition:

  1. the per-center effective-charge nuclear SOC (SOCType 1), and
  2. the valence J - 3/2 K contraction (SOCType 3 minus the one-electron
     part), in an ECP valence basis.

Both codes use Ge/LANL2DZ (Hay-Wadt, 28-core) + F/6-31G, the same
geometry, and SA-CASSCF CAS(9,6) with two roots.  The splitting is the
2Pi value; ORCA prints 47.87 cm^-1 (SOCType 1) and 26.00 cm^-1
(SOCType 3).

Run with the pyscf environment used by the project.  The ORCA numbers
are constants in this script and are also recorded in the report.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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

from diffmrc.core import somf as dsomf      # noqa: E402
from diffmrc.core import somf_ecp as decp   # noqa: E402

CM = 219474.6313702
A2 = 0.5 * float(nist.ALPHA) ** 2
PAULI = np.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)
SPIN = 0.5 * PAULI
ORCA = {"soc_type_1_cm": 47.87, "soc_type_3_cm": 26.00,
        "two_electron_only_cm": 14.14}


def main():
    t0 = time.time()
    mol = gto.M(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
                basis={"Ge": "lanl2dz", "F": "6-31g"},
                ecp={"Ge": "lanl2dz"}, spin=1, unit="Bohr", verbose=0)
    print("has_ecp=%s has_ecp_soc=%s nelec=%d nao=%d"
          % (mol.has_ecp(), mol.has_ecp_soc(), mol.nelectron,
             mol.nao_nr()))
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
    D = [[np.asarray(mc.fcisolver.trans_rdm1(ci[i], ci[j], nact, nele_act))
          for j in range(nstates)] for i in range(nstates)]
    p = np.asarray(mc.make_rdm1(), dtype=float)
    p = 0.5 * (p + p.T)
    print("CAS E=%.10f converged=%s scalar gaps cm=%s"
          % (mc.e_tot, mc.converged,
             np.array2string((es - es[0]) * CM, precision=6)))

    nuc = decp.nuclear_soc(mol, range(mol.natm))
    h2 = dsomf.somf_two_electron(mol, p, contract="explicit")
    h_1e = A2 * nuc
    h_somf = A2 * (nuc + h2)

    ca = C[:, act]

    def split(h, pref):
        hmo = np.einsum("ap,kab,bq->kpq", ca, h, ca, optimize=True)
        T = np.zeros((2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i, j, k] = float(np.sum(D[i][j] * hmo[k]))
        H = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            H[2 * i, 2 * i] = es[i]
            H[2 * i + 1, 2 * i + 1] = es[i]
            for j in range(2):
                for k in range(3):
                    H[2 * i:2 * i + 2, 2 * j:2 * j + 2] += \
                        pref * T[i, j, k] * SPIN[k]
        w = np.linalg.eigvalsh(H)
        return dict(splitting_cm=float((w[2] - w[0]) * CM),
                    levels_cm=[float((x - es[0]) * CM) for x in w],
                    T=[[[float(x) for x in T[i, j]] for j in range(2)]
                       for i in range(2)],
                    hermiticity=float(np.abs(H - H.conj().T).max()))

    one = split(h_1e, 1j)
    somf = split(h_somf, 1j)
    two_e = split(A2 * h2, 1j)
    out = dict(
        system="GeF scalar ECP",
        basis={"Ge": "lanl2dz", "F": "6-31g"}, ecp={"Ge": "lanl2dz"},
        cas=dict(ncore=ncore, nact=nact, nele_act=int(nele_act),
                 nstates=nstates),
        cas_energy=float(mc.e_tot), converged=bool(mc.converged),
        scalar_gaps_cm=[float((x - es[0]) * CM) for x in es],
        density_trace=float(np.einsum("pq,pq->", p,
                                      mol.intor("int1e_ovlp"))),
        nuclear=one, somf=somf, two_electron_only=two_e,
        orca=ORCA,
        ratio_pyscf=somf["splitting_cm"] / one["splitting_cm"],
        ratio_orca=ORCA["soc_type_3_cm"] / ORCA["soc_type_1_cm"],
        abs_dev_1e_cm=one["splitting_cm"] - ORCA["soc_type_1_cm"],
        abs_dev_somf_cm=somf["splitting_cm"] - ORCA["soc_type_3_cm"],
        abs_dev_2e_cm=two_e["splitting_cm"] - ORCA["two_electron_only_cm"],
        ratio_2e_pyscf=(two_e["splitting_cm"] / one["splitting_cm"]),
        ratio_2e_orca=(ORCA["two_electron_only_cm"]
                       / ORCA["soc_type_1_cm"]),
        runtime_s=time.time() - t0)
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "results_compare_orca_gef.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
