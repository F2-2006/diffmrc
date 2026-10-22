#!/usr/bin/env python
"""Pin down the per-center nuclear SOC construction.

``int1e_pnucxp`` is the sum over centers of Z_A (r-R_A) x p / |r-R_A|^3
(up to a sign convention).  This probe checks:
  * whether the ATM charge used by pnucxp is the ECP effective charge or
    the full nuclear charge;
  * the sign and origin convention of ``int1e_prinvxp`` relative to
    pnucxp, per center;
  * whether a per-center sum reproduces pnucxp exactly.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from pyscf import gto


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def frob(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def report(tag, mol):
    pn = mol.intor_asymmetric("int1e_pnucxp", 3)
    print("\n[%s] nao=%d natm=%d charges=%s" % (tag, mol.nao_nr(), mol.natm,
                                                mol.atom_charges().tolist()))
    print("  pnucxp frob=%s" % [frob(pn[k]) for k in range(3)])
    for ia in range(mol.natm):
        with mol.with_rinv_at_nucleus(ia):
            v = np.asarray(mol.intor("int1e_prinvxp", comp=3))
            va = np.asarray(mol.intor_asymmetric("int1e_prinvxp", 3))
        print("  center %d %s: |prinvxp|_F=%s  asym_vs_intor=%.2e"
              % (ia, mol.atom_symbol(ia),
                 [frob(v[k]) for k in range(3)], maxabs(v - va)))
        for sign in (+1.0, -1.0):
            for ztag, z in (("Zeff", float(mol.atom_charge(ia))),
                            ("Zfull", float(gto.charge(mol.atom_symbol(ia))))):
                pass
    # Per-center reconstruction with both sign choices and both charges.
    for sign in (+1.0, -1.0):
        for ztag in ("Zeff", "Zfull"):
            acc = np.zeros_like(pn)
            for ia in range(mol.natm):
                with mol.with_rinv_at_nucleus(ia):
                    v = np.asarray(mol.intor("int1e_prinvxp", comp=3))
                z = (float(mol.atom_charge(ia)) if ztag == "Zeff"
                     else float(gto.charge(mol.atom_symbol(ia))))
                acc += sign * z * v
            print("  sum(%s %s*prinvxp) - pnucxp: maxabs=%.3e rel=%.3e"
                  % (ztag, "+" if sign > 0 else "-", maxabs(acc - pn),
                     maxabs(acc - pn) / max(maxabs(pn), 1e-300)))


if __name__ == "__main__":
    f_atom = gto.M(atom=[("F", (0.0, 0.0, 0.0))], basis="6-31g",
                   spin=1, unit="Bohr", verbose=0)
    report("F atom, no ECP", f_atom)

    ge_ecp = gto.M(atom=[("Ge", (0.0, 0.0, 0.0))], basis="lanl2dz",
                   ecp={"Ge": "ecpds28mdfso"}, spin=2, unit="Bohr", verbose=0)
    report("Ge atom, SO-ECP", ge_ecp)

    ge_all = gto.M(atom=[("Ge", (0.0, 0.0, 0.0))], basis="lanl2dz",
                   spin=0, unit="Bohr", verbose=0)
    report("Ge atom, no ECP", ge_all)

    gef = gto.M(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
                basis={"Ge": "lanl2dz", "F": "6-31g"},
                ecp={"Ge": "ecpds28mdfso"}, spin=1, unit="Bohr", verbose=0)
    report("GeF, Ge SO-ECP", gef)

    h2o = gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
                basis="6-31g", unit="Angstrom", verbose=0)
    report("water, all-electron", h2o)
