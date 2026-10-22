#!/usr/bin/env python
"""Which YbF CASCI roots carry a nonzero SOC matrix element?

Runs the fixed active window with eight roots, prints the diagonal
active-space occupations and the norm of the ECPso SOC vector between
the ground state and every root.  A nonzero entry identifies a Sigma-Pi
type partner; the degenerate pair with identical occupations is the
model 2Pi manifold used for the value-layer check.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys

import numpy as np
from pyscf import gto, mcscf, scf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from diffmrc.core import somf_ecp as decp  # noqa: E402

CM = 219474.6313702


def main():
    mol = gto.M(atom=[("Yb", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))],
                basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr", verbose=0)
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
    ncore, ncas = len(frozen), len(active)
    nele = mol.nelectron - 2 * ncore
    mc = mcscf.CASCI(mf, ncas, nele, ncore=ncore)
    mc.mo_coeff = c_ord
    mc.verbose = 0
    nroots = 8
    h1eff, ecore = mc.get_h1eff()
    h2eff = mc.get_h2eff()
    e_raw, ci = mc.fcisolver.kernel(h1eff, h2eff, ncas, nele,
                                    nroots=nroots)
    es = np.atleast_1d(np.asarray(e_raw, dtype=float)) + ecore
    ci = np.asarray(ci)
    print("nroots=%d nele=%d ncas=%d" % (nroots, nele, ncas))
    print("gaps cm:", np.array2string((es - es[0]) * CM, precision=2))

    ecp = decp.ecp_soc(mol)
    ca = c_ord[:, ncore:ncore + ncas]
    hmo = np.einsum("ap,kab,bq->kpq", ca, ecp, ca, optimize=True)
    for j in range(nroots):
        d = np.asarray(mc.fcisolver.trans_rdm1(ci[0], ci[j], ncas, nele))
        t = np.array([float(np.sum(d * hmo[k])) for k in range(3)])
        d0 = np.asarray(mc.fcisolver.trans_rdm1(ci[j], ci[j], ncas, nele))
        print("root %d gap %10.2f occ %s |T0j| %.6e T=%s"
              % (j, (es[j] - es[0]) * CM,
                 np.array2string(np.diag(d0), precision=2),
                 float(np.linalg.norm(t)),
                 np.array2string(t, precision=6)))


if __name__ == "__main__":
    main()
