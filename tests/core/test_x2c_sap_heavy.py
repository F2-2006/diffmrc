"""Heavy SAP-X2C checks on XeH+ (gated).

The two-component SCF and the four-component DHF reference take minutes per
cycle on the 101 server, so the test runs only when explicitly requested:

    DIFFMRC_X2C_SAP_HEAVY=1 PYTHONPATH=src pytest \
        tests/core/test_x2c_sap_heavy.py -q

The same numbers are produced by experiments/x2c_sap_smoke.py --system xeh.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DIFFMRC_X2C_SAP_HEAVY", "") != "1",
    reason="set DIFFMRC_X2C_SAP_HEAVY=1 to run the XeH+ SCF benchmarks")

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf  # noqa: E402
from pyscf.x2c import x2c  # noqa: E402

from diffmrc.core import x2c_sap as xs  # noqa: E402


def _mol_xeh():
    return gto.M(atom=[("Xe", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.75))],
                 unit="Bohr", basis={"Xe": "cc-pvtz-dk", "H": "cc-pvdz"},
                 charge=1, spin=0, verbose=0)


def _run_2c(mol, hcore):
    mf = x2c.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 300
    mf.get_hcore = lambda *args: hcore
    return float(mf.kernel()), mf


def test_xeh_sap_x2c_moves_toward_dhf():
    mol = _mol_xeh()
    h1 = x2c.SpinorX2CHelper(mol).get_hcore(mol)
    hs = xs.x2c_sap_hcore(mol, spinor=True)

    e_1e, mf_1e = _run_2c(mol, h1)
    e_sap, mf_sap = _run_2c(mol, hs)

    mf_dhf = scf.DHF(mol)
    mf_dhf.verbose = 0
    mf_dhf.conv_tol = 1e-12
    mf_dhf.max_cycle = 300
    e_dhf = float(mf_dhf.kernel())

    assert abs(e_sap - e_dhf) < abs(e_1e - e_dhf)
    e1 = np.asarray(mf_1e.mo_energy)[np.asarray(mf_1e.mo_occ) > 0]
    es = np.asarray(mf_sap.mo_energy)[np.asarray(mf_sap.mo_occ) > 0]
    ed = np.asarray(mf_dhf.mo_energy)[np.asarray(mf_dhf.mo_occ) > 0]
    assert np.max(np.abs(es - ed)) < np.max(np.abs(e1 - ed))
