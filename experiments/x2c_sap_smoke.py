"""Smoke and benchmark driver for the SAP-X2C value layer.

Run with the PySCF/pyscfad environment on the compute server:

    python x2c_sap_smoke.py --system h2o --out h2o.json

The script prints a JSON block with the matrix checks and the SCF energies
needed for the benchmark report.  It is not a pytest file; the numerical
assertions live in tests/core/test_x2c_sap.py.
"""
import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_threads = os.environ.get("DIFFMRC_THREADS", "1")
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = _threads
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
from pyscf import gto, scf
from pyscf.x2c import sfx2c1e, x2c

from diffmrc.core import x2c_sap as xs

SYSTEMS = {
    "h2o": dict(atom=[("O", (0.0, 0.0, 0.0)),
                      ("H", (0.0, -1.430523, 1.109269)),
                      ("H", (0.0, 1.430523, 1.109269))],
                unit="Bohr", basis="6-31g", charge=0, spin=0),
    "xeh": dict(atom=[("Xe", (0.0, 0.0, 0.0)),
                      ("H", (0.0, 0.0, 1.75))],
                unit="Bohr", basis={"Xe": "cc-pvtz-dk", "H": "cc-pvdz"},
                charge=1, spin=0),
    "xe": dict(atom=[("Xe", (0.0, 0.0, 0.0))],
               unit="Bohr", basis="cc-pvtz-dk", charge=0, spin=0),
    # Negative control, not a production comparison: the 3-21g decontraction
    # is ill-conditioned for the X2C transform (Xe 1s X2C vs 4c DHF differs
    # by 13 Eh, total energy by 32 Eh).  Kept so the failure is reproducible;
    # use cc-pVTZ-DK for any accuracy claim.
    "xe321": dict(atom=[("Xe", (0.0, 0.0, 0.0))],
                  unit="Bohr", basis="3-21g", charge=0, spin=0),
    "xeh321": dict(atom=[("Xe", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.75))],
                   unit="Bohr", basis={"Xe": "3-21g", "H": "3-21g"},
                   charge=1, spin=0),
    "yb": dict(atom=[("Yb", (0.0, 0.0, 0.0))],
               unit="Bohr", basis="cc-pvdz-dk", charge=0, spin=0),
}


def build(system):
    spec = SYSTEMS[system]
    mol = gto.Mole()
    mol.atom = spec["atom"]
    mol.unit = spec["unit"]
    mol.basis = spec["basis"]
    mol.charge = spec["charge"]
    mol.spin = spec["spin"]
    mol.verbose = 0
    mol.build()
    return mol


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run_rhf_custom(mol, hcore):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.get_hcore = lambda *args: hcore
    e = mf.kernel()
    return float(e), mf


def run_uhf_custom(mol, hcore):
    mf = x2c.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 300
    mf.get_hcore = lambda *args: hcore
    e = mf.kernel()
    return float(e), mf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="h2o", choices=sorted(SYSTEMS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mol = build(args.system)
    out = {"system": args.system, "nao": mol.nao_nr(),
           "nao_2c": mol.nao_2c(), "nelectron": mol.nelectron,
           "basis": str(mol.basis), "charge": mol.charge, "spin": mol.spin}

    table = xs.load_sap_table()
    out["table"] = {
        "elements": len(table),
        "sum_coef_H": float(sum(table[1]["coef"])),
        "sum_coef_Xe": float(sum(table[54]["coef"])),
        "sum_coef_Yb": float(sum(table[70]["coef"])),
    }

    # --- matrix checks -----------------------------------------------------
    t0 = time.time()
    ve_s = xs.sap_potential_matrix(mol, spinor=False)
    ve_2, we_2 = xs.sap_matrices(mol, spinor=True)
    out["matrix_time_s"] = time.time() - t0
    out["ve"] = {
        "shape_scalar": list(ve_s.shape),
        "shape_spinor": list(ve_2.shape),
        "herm_scalar": float(np.abs(ve_s - ve_s.T).max()),
        "herm_spinor": float(np.abs(ve_2 - ve_2.conj().T).max()),
        "trace_scalar": float(np.trace(ve_s)),
        "norm_scalar": float(np.linalg.norm(ve_s)),
        "we_norm_spinor": float(np.linalg.norm(we_2)),
        "we_herm_spinor": float(np.abs(we_2 - we_2.conj().T).max()),
        "scalar_trace_vs_spinor_half_trace": float(
            np.trace(ve_s) - 0.5 * np.real(np.trace(ve_2))),
    }

    # --- 1eX2C reproduction (SAP off) --------------------------------------
    h1e_2 = xs.x2c_hcore_1e(mol, spinor=True)
    ref_2 = x2c.SpinorX2CHelper(mol).get_hcore(mol)
    out["one_x2c_spinor"] = {
        "maxabs": float(np.abs(h1e_2 - ref_2).max()),
        "rel": rel(h1e_2, ref_2),
    }
    if mol.nelectron % 2 == 0:
        h1e_s = xs.x2c_hcore_1e(mol, spinor=False)
        ref_s = sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)
        out["one_x2c_scalar"] = {
            "maxabs": float(np.abs(h1e_s - ref_s).max()),
            "rel": rel(h1e_s, ref_s),
        }

    # --- SAP-X2C matrix ----------------------------------------------------
    t0 = time.time()
    hs_2 = xs.x2c_sap_hcore(mol, spinor=True)
    out["sap_hcore_time_s"] = time.time() - t0
    out["sap_hcore"] = {
        "spinor_maxabs_vs_1e": float(np.abs(hs_2 - h1e_2).max()),
        "spinor_norm_change": rel(hs_2, h1e_2),
        "spinor_herm": float(np.abs(hs_2 - hs_2.conj().T).max()),
    }

    # --- energies ----------------------------------------------------------
    out["energies"] = {}
    if mol.nelectron % 2 == 0:
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        out["energies"]["rhf_nr"] = float(mf.kernel())
        mf2 = scf.RHF(mol)
        mf2.verbose = 0
        mf2.conv_tol = 1e-12
        out["energies"]["rhf_sfx2c1e_pyscf"] = float(mf2.sfx2c1e().kernel())

    t0 = time.time()
    e_1e_2, mf_1e = run_uhf_custom(mol, ref_2)
    out["energies"]["uhf_1e_x2c"] = e_1e_2
    out["energies"]["uhf_1e_time_s"] = time.time() - t0

    t0 = time.time()
    e_sap_2, mf_sap = run_uhf_custom(mol, hs_2)
    out["energies"]["uhf_sap_x2c"] = e_sap_2
    out["energies"]["uhf_sap_time_s"] = time.time() - t0

    t0 = time.time()
    mf_dhf = scf.DHF(mol)
    mf_dhf.verbose = 0
    mf_dhf.conv_tol = 1e-12
    mf_dhf.max_cycle = 300
    e_dhf = float(mf_dhf.kernel())
    out["energies"]["dhf_4c"] = e_dhf
    out["energies"]["dhf_time_s"] = time.time() - t0

    out["energies"]["sap_minus_1e_2c"] = e_sap_2 - e_1e_2
    out["energies"]["err_1e_vs_dhf"] = e_1e_2 - e_dhf
    out["energies"]["err_sap_vs_dhf"] = e_sap_2 - e_dhf

    # occupied spinor energies and low-lying splittings
    try:
        e1 = np.asarray(mf_1e.mo_energy)
        es = np.asarray(mf_sap.mo_energy)
        ed = np.asarray(mf_dhf.mo_energy)
        o1 = np.asarray(mf_1e.mo_occ) > 0
        os_ = np.asarray(mf_sap.mo_occ) > 0
        od = np.asarray(mf_dhf.mo_occ) > 0
        out["spinor_energies"] = {
            "1e_occ": [float(x) for x in e1[o1]],
            "sap_occ": [float(x) for x in es[os_]],
            "dhf_occ": [float(x) for x in ed[od]],
        }
        out["spinor_energies"]["max_abs_1e_minus_dhf"] = float(
            np.max(np.abs(e1[o1] - ed[od])))
        out["spinor_energies"]["max_abs_sap_minus_dhf"] = float(
            np.max(np.abs(es[os_] - ed[od])))
    except Exception as exc:
        out["spinor_energies_error"] = "%s: %s" % (type(exc).__name__, exc)

    text = json.dumps(out, indent=2, default=float)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
