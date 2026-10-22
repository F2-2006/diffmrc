#!/usr/bin/env python
"""State structure of YbF in a 6s / F 2p active window.

The YbF ROHF orbitals put the 4f and semicore shells below the F 2p
manifold.  This probe freezes every occupied orbital except the three
F 2p-derived orbitals and the singly occupied 6s sigma orbital, adds two
lowest virtuals, and runs a CASCI with three roots.  The point is to see
whether the lowest roots are the 2Sigma+ ground state plus the two
components of 2Pi, and to record the transition 1-RDMs.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from pyscf import gto, mcscf, scf

CM = 219474.6313702


def build():
    return gto.M(atom=[("Yb", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))],
                 basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                 ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr",
                 verbose=0)


def main():
    mol = build()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.max_cycle = 300
    mf.kernel()
    c = np.asarray(mf.mo_coeff)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.5))
    print("nocc=%d nao=%d" % (nocc, mol.nao_nr()))

    # Active window: F 2p manifold (22, 23, 24), the 6s sigma (25), and
    # the two lowest virtuals (26, 27).  Frozen: all other occupied MOs.
    active = [22, 23, 24, 25, 26, 27]
    active_set = set(active)
    frozen = [i for i in range(nocc) if i not in active_set]
    virt = [i for i in range(c.shape[1]) if i >= nocc and i not in active_set]
    order = frozen + active + virt
    c_ord = c[:, order]
    ncore = len(frozen)
    print("ncore=%d ncas=%d frozen=%s active=%s"
          % (ncore, len(active), frozen, active))

    mc = mcscf.CASCI(mf, len(active), mol.nelectron - 2 * ncore,
                     ncore=ncore)
    mc.mo_coeff = c_ord
    mc.verbose = 0
    mc.fcisolver.nroots = 3
    mc.kernel()
    h1eff, ecore = mc.get_h1eff()
    h2eff = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1eff, h2eff, len(active),
                                    mol.nelectron - 2 * ncore, nroots=3)
    es = np.atleast_1d(np.asarray(e_raw, dtype=float)) + ecore
    print("CASCI e_tot=%s converged=%s" % (np.asarray(mc.e_tot), mc.converged))
    print("state energies (Eh): %s" % np.array2string(es, precision=8))
    print("gaps (cm^-1): %s"
          % np.array2string((es - es[0]) * CM, precision=4))

    ci = np.asarray(ci)
    ncas = len(active)
    nelecas = mol.nelectron - 2 * ncore
    print("ci shape=%s" % (ci.shape,))
    for i in range(ci.shape[0]):
        for j in range(ci.shape[0]):
            d = np.asarray(mc.fcisolver.trans_rdm1(ci[i], ci[j], ncas,
                                                   nelecas))
            if i == j:
                print("D%d%d diag sum=%.6f sym=%.2e occ=%s"
                      % (i, i, float(np.trace(d)),
                         float(np.abs(d - d.T).max()),
                         np.array2string(np.diag(d), precision=3)))
            else:
                print("D%d%d trace=%.3e sym=%.2e anti=%.2e norm=%.3e"
                      % (i, j, float(np.trace(d)),
                         float(np.abs(d - d.T).max()),
                         float(np.abs(d + d.T).max()),
                         float(np.linalg.norm(d))))


if __name__ == "__main__":
    main()
