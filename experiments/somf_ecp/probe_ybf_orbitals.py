#!/usr/bin/env python
"""Inspect the YbF ROHF orbital ladder for active-space selection.

Prints orbital energies, occupations (from the ROHF density), and the
dominant atom/angular composition of each MO, so that the 6s / F 2p
active window can be chosen by composition instead of by a blind index
slice.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from pyscf import gto, scf

MOL = gto.M(atom=[("Yb", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))],
            basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
            ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr", verbose=0)


def main():
    mol = MOL
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.max_cycle = 300
    mf.kernel()
    print("converged=%s E=%.10f nelec=%d nao=%d"
          % (mf.converged, mf.e_tot, mol.nelectron, mol.nao_nr()))
    c = np.asarray(mf.mo_coeff)
    e = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    s = mol.intor("int1e_ovlp")
    # Mulliken population per atom for each MO.
    pop = np.zeros((mol.natm, c.shape[1]))
    for ia in range(mol.natm):
        sl = mol.aoslice_by_atom()[ia]
        p0, p1 = int(sl[2]), int(sl[3])
        pop[ia] = np.einsum("pi,pq,qi->i", c[p0:p1], s[p0:p1], c)
    print("idx  energy(Eh)  occ   Yb      F")
    for i in range(c.shape[1]):
        print("%3d  %10.5f  %4.1f  %7.3f %7.3f"
              % (i, e[i], occ[i], pop[0, i], pop[1, i]))
    # Atomic orbital labels for reference.
    print("AO labels:", mol.ao_labels())


if __name__ == "__main__":
    main()
