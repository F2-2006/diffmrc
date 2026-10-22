"""Integral-layer verification of the SAP matrix derivative design.

For each atom and Cartesian direction this script compares the analytic
derivatives of V^e and W^e against central finite differences:

    dV^e/dR_B = -(P_B ip1 + (P_B ip1)^T)
                - sum_{k on B} w_k ip2_k
    dW^e/dR_B = -(P_B J + (P_B J)^dagger)
                - sum_{k on B} w_k K_k

where ip1/ip2 are the scalar 3-centre derivatives and J/K are the spinor
``int3c2e_ipspsp1_spinor`` / ``int3c2e_spsp1ip2_spinor`` derivatives.
The finite difference moves the atom, and with it the SAP auxiliary
functions on that atom, so both the basis-motion and auxiliary-motion
terms are exercised.

This validates the derivative *integrals* only.  The X2C chain derivative
(dX, dR, the SCF gradient assembly) is not implemented.
"""
import json
import os

import numpy as np
from pyscf import gto
from pyscf.df.incore import aux_e2

from diffmrc.core import x2c_sap as xs

GEOM = [("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.430523, 1.109269)),
        ("H", (0.0, 1.430523, 1.109269))]
STEP = 1e-4


def _mol(geom):
    return gto.M(atom=geom, unit="Bohr", basis="6-31g", verbose=0)


def _mats(mol, table):
    aux = xs.sap_auxmol(mol, table=table)
    weights = xs.sap_aux_weights(mol, aux, table=table)
    ve = np.einsum("ijp,p->ij", aux_e2(mol, aux, intor="int3c2e"), weights)
    we = np.einsum("ijp,p->ij",
                   aux_e2(mol, aux, intor="int3c2e_spsp1_spinor"), weights)
    return aux, weights, ve, we


def _shift(geom, ia, ix, delta):
    out = [(s, np.array(x, dtype=float)) for s, x in geom]
    out[ia][1][ix] += delta
    return out


def _analytic(mol, aux, weights, ia, ix):
    nao = mol.nao_nr()
    p0, p1 = mol.aoslice_by_atom()[ia, 2:]
    sh0, sh1, a0, a1 = aux.aoslice_by_atom()[ia]
    aux_mask = np.zeros(aux.nao_nr())
    aux_mask[a0:a1] = 1.0

    proj = np.zeros(nao)
    proj[p0:p1] = 1.0
    ip1 = aux_e2(mol, aux, intor="int3c2e_ip1", comp=3)[ix]
    ip2 = aux_e2(mol, aux, intor="int3c2e_ip2", comp=3)[ix]
    t1 = np.einsum("m,mnk,k->mn", proj, ip1, weights)
    t2 = np.einsum("mnk,k,k->mn", ip2, weights, aux_mask)
    dve = -(t1 + t1.T + t2)

    proj2 = np.zeros(2 * nao)
    proj2[np.arange(p0, p1) * 2] = 1.0
    proj2[np.arange(p0, p1) * 2 + 1] = 1.0
    j1 = aux_e2(mol, aux, intor="int3c2e_ipspsp1_spinor")[ix]
    j2 = aux_e2(mol, aux, intor="int3c2e_spsp1ip2_spinor")[ix]
    u1 = np.einsum("m,mnk,k->mn", proj2, j1, weights)
    u2 = np.einsum("mnk,k,k->mn", j2, weights, aux_mask)
    dwe = -(u1 + u1.conj().T + u2)
    return dve, dwe


def main():
    table = xs.load_sap_table()
    mol = _mol(GEOM)
    aux, weights, _, _ = _mats(mol, table)
    out = {"step_bohr": STEP, "rows": {}}
    for ia in range(mol.natm):
        for ix in range(3):
            ve_p, we_p = _mats(_mol(_shift(GEOM, ia, ix, +STEP)), table)[2:]
            ve_m, we_m = _mats(_mol(_shift(GEOM, ia, ix, -STEP)), table)[2:]
            fd_ve = (ve_p - ve_m) / (2 * STEP)
            fd_we = (we_p - we_m) / (2 * STEP)
            dve, dwe = _analytic(mol, aux, weights, ia, ix)
            out["rows"]["atom%d_dir%d" % (ia, ix)] = {
                "dve_rel": float(np.linalg.norm(dve - fd_ve) /
                                 np.linalg.norm(fd_ve)),
                "dwe_rel": float(np.linalg.norm(dwe - fd_we) /
                                 np.linalg.norm(fd_we)),
                "dve_maxabs": float(np.abs(dve - fd_ve).max()),
                "dwe_maxabs": float(np.abs(dwe - fd_we).max()),
            }
    worst_ve = max(row["dve_rel"] for row in out["rows"].values())
    worst_we = max(row["dwe_rel"] for row in out["rows"].values())
    out["worst_dve_rel"] = worst_ve
    out["worst_dwe_rel"] = worst_we
    out["pass"] = bool(worst_ve < 1e-7 and worst_we < 1e-6)
    text = json.dumps(out, indent=2)
    print(text)
    return out


if __name__ == "__main__":
    main()
