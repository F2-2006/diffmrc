#!/usr/bin/env python
"""Probe the pyscf ECP/SOC interface for a valence-level SOMF operator.

Questions answered here (all numbers printed, no pass/fail):
  1. What nuclear charge does ``int1e_pnucxp`` use on an ECP center?
  2. Can the per-center nuclear SOC be rebuilt from ``int1e_prinvxp``
     with the ``with_rinv_at_nucleus`` origin, and does the sum equal
     ``int1e_pnucxp``?
  3. Which atoms carry a spin-orbit ECP, and how are the ECP data
     stored on ``mol``?
  4. Does the two-electron SOMF kernel ``int2e_p1vxp1`` work in an ECP
     valence basis (including f functions), and does the direct route
     agree with the explicit einsum route?
  5. Is the two-electron SOMF bilinear form symmetric in its two
     density arguments?
  6. What are the sizes and norms of ECPso versus the nuclear SOC?
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time

import numpy as np
from pyscf import gto, scf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from diffmrc.core import somf as dsomf  # noqa: E402


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def frob(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def build(spec):
    return gto.M(atom=spec["atom"], basis=spec["basis"], ecp=spec["ecp"],
                 spin=spec["spin"], charge=0, unit="Bohr", verbose=0)


GEF = dict(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
           basis={"Ge": "lanl2dz", "F": "6-31g"},
           ecp={"Ge": "ecpds28mdfso"}, spin=1)
YBF = dict(atom=[("Yb", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))],
           basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
           ecp={"Yb": "ecpds28mwbso"}, spin=1)
CEF = dict(atom=[("Ce", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))],
           basis={"Ce": "stuttgart_rsc", "F": "6-31g"},
           ecp={"Ce": "ecpds28mwbso"}, spin=1)


def probe_charges(name, spec):
    mol = build(spec)
    print("\n=== %s ===" % name)
    print("nao=%d nao_2c=%d nelectron=%d nbas=%d"
          % (mol.nao_nr(), mol.nao_2c(), mol.nelectron, mol.nbas))
    print("has_ecp=%s has_ecp_soc=%s" % (mol.has_ecp(), mol.has_ecp_soc()))
    print("atom_symbols=%s" % [mol.atom_symbol(i) for i in range(mol.natm)])
    print("atom_charges (effective)=%s" % mol.atom_charges().tolist())
    print("_ecp keys=%s" % list(mol._ecp.keys()))
    for key, val in mol._ecp.items():
        print("  _ecp[%r]: type=%s len=%d" % (key, type(val), len(val)))
        for entry in val:
            print("    entry type=%s len=%s first=%s"
                  % (type(entry), len(entry) if hasattr(entry, "__len__")
                     else "?", entry[0] if hasattr(entry, "__len__") else entry))

    # Nuclear SOC: pnucxp versus the per-center rinv cross p construction.
    pnuc = mol.intor_asymmetric("int1e_pnucxp", 3)
    acc = np.zeros_like(pnuc)
    per_atom = []
    for ia in range(mol.natm):
        with mol.with_rinv_at_nucleus(ia):
            v = mol.intor("int1e_prinvxp", comp=3)
        z = float(mol.atom_charge(ia))
        acc += z * v
        per_atom.append(dict(atom=ia, symbol=mol.atom_symbol(ia), z=z,
                             frob=frob(z * v), maxabs=maxabs(z * v)))
    print("pnucxp frob=%s" % [frob(pnuc[k]) for k in range(3)])
    print("sum Z_eff*prinvxp - pnucxp maxabs=%.3e (scale %.3e)"
          % (maxabs(acc - pnuc), maxabs(pnuc)))
    for row in per_atom:
        print("  center %d %-3s Z_eff=%6.2f |Z*prinvxp|_F=%.6e"
              % (row["atom"], row["symbol"], row["z"], row["frob"]))

    if mol.has_ecp_soc():
        v = np.asarray(mol.intor("ECPso"))
        print("ECPso shape=%s frob=%s maxabs=%.6e antisym=%s"
              % (v.shape, [frob(v[k]) for k in range(3)], maxabs(v),
                 [maxabs(v[k] + v[k].T) for k in range(3)]))

    # A physical valence density from ROHF (converged or not, only for
    # kernel checks); symmetrized to satisfy the SOMF contract.
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.max_cycle = 200
    try:
        mf.kernel()
    except Exception as exc:                                  # noqa: BLE001
        print("ROHF failed: %s: %s" % (type(exc).__name__, exc))
    dm = mf.make_rdm1()
    if isinstance(dm, np.ndarray) and dm.ndim == 3:
        dm = dm[0] + dm[1]
    dm = np.asarray(dm, dtype=float)
    dm = 0.5 * (dm + dm.T)
    print("ROHF converged=%s E=%.10f trace(P S)=%.9f"
          % (mf.converged, mf.e_tot,
             float(np.einsum("pq,pq->", dm, mol.intor("int1e_ovlp")))))

    t0 = time.time()
    try:
        h_dir = dsomf.somf_two_electron(mol, dm, contract="direct")
        t_dir = time.time() - t0
        print("two-electron direct ok in %.2f s, frob=%s"
              % (t_dir, [frob(h_dir[k]) for k in range(3)]))
    except Exception as exc:                                  # noqa: BLE001
        print("two-electron direct failed: %s: %s"
              % (type(exc).__name__, exc))
        h_dir = None

    if mol.nao_nr() <= 40:
        t0 = time.time()
        h_exp = dsomf.somf_two_electron(mol, dm, contract="explicit")
        t_exp = time.time() - t0
        print("two-electron explicit ok in %.2f s" % t_exp)
        if h_dir is not None:
            print("explicit vs direct maxabs=%.3e rel=%.3e"
                  % (maxabs(h_exp - h_dir),
                     maxabs(h_exp - h_dir) / max(maxabs(h_exp), 1e-300)))
        # Bilinear symmetry B(D,P) = sum D * h2[P].
        p2 = dm.copy()
        b_dp = float(np.sum(dm * h_exp))
        h_d = dsomf.somf_two_electron(mol, dm, contract="explicit")
        b_pd = float(np.sum(dm * h_d))
        print("bilinear B(P,P)=%.10e (trivially symmetric)" % b_dp)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(dm.shape)
        x = 0.5 * (x + x.T)
        y = rng.standard_normal(dm.shape)
        y = 0.5 * (y + y.T)
        b_xy = float(np.sum(x * dsomf.somf_two_electron(mol, y,
                                                        contract="explicit")))
        b_yx = float(np.sum(y * dsomf.somf_two_electron(mol, x,
                                                        contract="explicit")))
        print("bilinear symmetry B(x,y)=%.10e B(y,x)=%.10e rel=%.3e"
              % (b_xy, b_yx, abs(b_xy - b_yx) / max(abs(b_xy), 1e-300)))

    # Derivative-kernel audit for the two-electron SOC kernel.
    for intname in ("int2e_ip1_p1vxp1", "int2e_p1vxp1_ip1",
                    "int2e_ipip1_p1vxp1"):
        try:
            mol.intor(intname, comp=3)
            print("derivative kernel %s: available" % intname)
        except Exception as exc:                              # noqa: BLE001
            print("derivative kernel %s: %s" % (intname,
                                                str(exc).split("\n")[0][:90]))


def probe_states(name, spec, ncore, nact, nstates):
    """State structure of the scalar problem (acceptance only)."""
    from pyscf import mcscf
    mol = build(spec)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    nele_act = mol.nelectron - 2 * ncore
    mc = mcscf.CASSCF(mf, nact, nele_act)
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([1.0 / nstates] * nstates)
    mc.fcisolver.nroots = nstates
    mc.kernel()
    es = np.atleast_1d(np.asarray(mc.fcisolver.e_states, dtype=float))
    print("\n=== states %s CAS(%d,%d) nroots=%d converged=%s ==="
          % (name, nele_act, nact, nstates, mc.converged))
    print("state energies (Eh): %s" % np.array2string(es, precision=8))
    print("gaps (cm^-1): %s"
          % np.array2string((es - es[0]) * 219474.6313702, precision=4))
    print("Etot=%.10f" % mc.e_tot)
    return mol, mf, mc


if __name__ == "__main__":
    probe_charges("GeF SO-ECP", GEF)
    probe_charges("YbF SO-ECP", YBF)
    probe_charges("CeF SO-ECP", CEF)
    probe_states("GeF", GEF, ncore=2, nact=6, nstates=2)
