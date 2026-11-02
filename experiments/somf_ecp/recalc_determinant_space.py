#!/usr/bin/env python
"""Determinant-space recalculation of the archived GeF SO-ECP numbers.

WHY THIS DRIVER EXISTS

``validate_ecp_somf.py`` and ``compare_orca_gef.py`` build the
state-interaction matrix from pyscf ``trans_rdm1`` transition densities.
That routine returns the spin-summed charge density (D_alpha + D_beta),
while the spin-orbit matrix element is a spin-resolved determinant
coupling; every splitting assembled this way is high by a
system-dependent factor (2.9% for the GeF ECPso value, 0.3-0.4% for the
all-electron OH/OF calibration).  The archived JSON files are kept as
history.

This driver recomputes the same systems with the determinant-space
spin-orbital CI of ``diffmrc.core.spinor_ci`` -- the route used by the
F-paper correction and by the Wave 18 SOC benchmark -- and reruns the old
assembly in the same process, so that the shift is attributed operator by
operator rather than across two independent runs.

Systems
  fp_paper      linear Ge-F = 3.4 Bohr, ECPso only.  Frozen F-paper setup;
                the new value must reproduce 1329.607404 cm^-1 and the old
                assembly must reproduce 1368.084143 cm^-1.
  validate      the bent GeF of ``validate_ecp_somf.py`` with its five
                operator variants (ecp, somf, somf_active_density, nuclear,
                double).
  compare_orca  the scalar-ECP GeF of ``compare_orca_gef.py`` with its
                three operators (nuclear, somf, two_electron_only).

Route
  The active-space one- and two-electron integrals come from the
  state-averaged CASSCF (``get_h1eff`` / ``get_h2eff``, frozen core
  folded in).  They are embedded in the spin-orbital basis exactly as in
  the Wave 18 benchmark, the spin-free Hamiltonian is diagonalized in the
  determinant basis, and each SOC operator is added as a one-body
  spin-orbital term (``H_SO = i sum_k h_k s_k``).  The state-interaction
  matrix is the projection of the full determinant Hamiltonian onto the
  spin-free states up to and including the target term.

Output
  ``results_ecp_somf_gef_recalc.json`` next to this file.  The archived
  result files are not modified.

Run (compute node, single thread):
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python \
      experiments/somf_ecp/recalc_determinant_space.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import sys
import time

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.data import nist

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from diffmrc.core import somf as dsomf          # noqa: E402
from diffmrc.core import somf_ecp as decp       # noqa: E402
from diffmrc.core import spinor_ci as sci       # noqa: E402

CM = 219474.6313702
A2 = 0.5 * float(nist.ALPHA) ** 2
SPIN = [np.array([[0, 1], [1, 0]], dtype=complex) / 2,
        np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
        np.array([[1, 0], [0, -1]], dtype=complex) / 2]


# ---------------------------------------------------------------------------
# Molecules and the reference SA-CASSCF
# ---------------------------------------------------------------------------
def build_gef(geometry, ecp):
    atoms = {"bent": [("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
             "linear": [("Ge", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.4))]}
    return gto.M(atom=atoms[geometry], basis={"Ge": "lanl2dz", "F": "6-31g"},
                 ecp=ecp, spin=1, unit="Bohr", verbose=0)


def sa_casscf(mol, ncore, nact, nstates, scf_tol=1e-12,
              cas_tol=1e-10, grad_tol=1e-5, spin_sector=None,
              guesses=("default",)):
    """State-averaged CASSCF with an explicit, recorded SCF ladder.

    ``guesses=("default",)`` reproduces the archived experiment scripts
    (one ROHF run, conv_tol 1e-12).  The F-paper setup instead uses the
    minao/atom/huckel ladder and conv_tol 1e-10 of the Wave 18 benchmark,
    which is what fixes the frozen 1329.607404 cm^-1 value.
    """
    best = None
    for guess in guesses:
        mf = scf.ROHF(mol)
        if guess != "default":
            mf.init_guess = guess
        mf.conv_tol = scf_tol
        mf.verbose = 0
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("ROHF did not converge (%s)" % guess)
        if best is None or mf.e_tot < best.e_tot:
            best = mf
    mf = best
    nele_act = mol.nelectron - 2 * ncore
    mc = mcscf.CASSCF(mf, nact, nele_act)
    mc.verbose = 0
    mc.conv_tol = cas_tol
    mc.conv_tol_grad = grad_tol
    mc.max_cycle = 200
    if spin_sector is not None:
        mc.fcisolver.spin = spin_sector
    mc.state_average_([1.0 / nstates] * nstates)
    mc.fcisolver.nroots = nstates
    mc.kernel()
    if not mc.converged:
        raise RuntimeError("SA-CASSCF did not converge")
    es = np.atleast_1d(np.asarray(mc.fcisolver.e_states, dtype=float))
    return mf, mc, es


# ---------------------------------------------------------------------------
# Candidate energy grouping (same tolerance as the Wave 18 benchmark)
# ---------------------------------------------------------------------------
def groups_of(energies, tol=1e-5):
    order = np.argsort(energies)
    out, cur = [], [int(order[0])]
    for k in order[1:]:
        if abs(energies[k] - energies[cur[-1]]) < tol:
            cur.append(int(k))
        else:
            out.append(cur)
            cur = [int(k)]
    out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Determinant-space route
# ---------------------------------------------------------------------------
def spin_orbital_integrals(h1, h2, norb):
    """Embed scalar active-space integrals in the spin-orbital basis.

    This is the embedding of the Wave 18 benchmark
    (``code/soc_bench.py``) that ``core.spinor_ci`` was validated with:
    a spin-independent one-body term is diagonal in spin, and the two
    electron integral is nonzero only when the two creation and the two
    annihilation indices carry equal spins.
    """
    nso = 2 * norb
    h1so = np.zeros((nso, nso), dtype=complex)
    eri = np.zeros((nso, nso, nso, nso), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            h1so[2 * p, 2 * q] = h1[p, q]
            h1so[2 * p + 1, 2 * q + 1] = h1[p, q]
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    v = h2[p, q, r, s]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            eri[2 * p + sp, 2 * q + sp,
                                2 * r + sq, 2 * s + sq] = v
    return h1so, eri


def soc_one_body(hk, norb):
    """One-body spin-orbital SOC matrix for H_SO = i sum_k h_k s_k."""
    nso = 2 * norb
    op = np.zeros((nso, nso), dtype=complex)
    for k in range(3):
        for p in range(norb):
            for q in range(norb):
                if hk[k, p, q] == 0.0:
                    continue
                for a in range(2):
                    for b in range(2):
                        op[2 * p + a, 2 * q + b] += (1j * hk[k, p, q]
                                                     * SPIN[k][a, b])
    return 0.5 * (op + op.conj().T)


def determinant_route(mol, mc, hk, ncore, ncas, nele):
    """State-interaction splitting from the full determinant Hamiltonian.

    ``hk`` is (3, nao, nao) in the ``+i`` convention.  Returns the span of
    the lowest spin-free term after SOC and the unperturbed term list.
    """
    c_act = np.asarray(mc.mo_coeff)[:, ncore:ncore + ncas]
    h1, ecore = mc.get_h1eff()
    h2 = np.asarray(mc.get_h2eff())
    if h2.ndim == 2:
        h2 = ao2mo.restore(1, h2, ncas)
    h1so, eso = spin_orbital_integrals(np.asarray(h1), h2, ncas)
    h0mat, _, _ = sci.spinor_ci_matrix(h1so, eso, 0, 2 * ncas, nele)
    w0, v0 = np.linalg.eigh(h0mat)
    w0e = w0 + ecore
    groups = groups_of(w0e)
    target = groups[0]
    if len(target) < 3:
        raise RuntimeError("the lowest spin-free term is not degenerate: %s"
                           % [len(g) for g in groups[:3]])
    nwin = int(np.count_nonzero(w0e <= w0e[target[0]] + 1e-6))
    if nwin != len(target):
        raise RuntimeError("spin-free window %d differs from the target "
                           "degeneracy %d" % (nwin, len(target)))
    vw = v0[:, :nwin]
    hk_mo = np.einsum("ap,kab,bq->kpq", c_act, hk, c_act, optimize=True)
    h1so_soc = h1so + soc_one_body(hk_mo, ncas)
    hfull, _, _ = sci.spinor_ci_matrix(h1so_soc, eso, 0, 2 * ncas, nele)
    hsi = vw.conj().T @ hfull @ vw
    levels = np.linalg.eigvalsh(0.5 * (hsi + hsi.conj().T)) + ecore
    return dict(
        span_cm=float((levels[-1] - levels[0]) * CM),
        levels_cm=[float((x - levels[0]) * CM) for x in levels],
        si_window_states=int(nwin),
        scalar_groups=[dict(energy_cm=float((w0e[g[0]] - w0e[0]) * CM),
                            degeneracy=len(g)) for g in groups[:4]],
        scalar_gap_cm=float((w0e[1] - w0e[0]) * CM),
        ndet=int(h0mat.shape[0]))


# ---------------------------------------------------------------------------
# Archived (spin-summed) route, rerun for the operator-by-operator ratio
# ---------------------------------------------------------------------------
def old_route(mc, es, ncore, nact, nele_act, ops):
    """Archived state-interaction assembly, ``pref`` per operator."""
    C = np.asarray(mc.mo_coeff)
    ca = C[:, ncore:ncore + nact]
    ci = np.asarray(mc.ci)
    nstates = len(es)
    dmat = {}
    for i in range(nstates):
        for j in range(nstates):
            dmat[(i, j)] = np.asarray(mc.fcisolver.trans_rdm1(
                ci[i], ci[j], nact, nele_act))
    out = {}
    for name, (h, pref) in ops.items():
        hmo = np.einsum("ap,kab,bq->kpq", ca, h, ca, optimize=True)
        t = np.zeros((nstates, nstates, 3))
        for i in range(nstates):
            for j in range(nstates):
                for k in range(3):
                    t[i, j, k] = float(np.sum(dmat[(i, j)] * hmo[k]))
        hm = np.zeros((2 * nstates, 2 * nstates), dtype=complex)
        for i in range(nstates):
            hm[2 * i, 2 * i] = es[i]
            hm[2 * i + 1, 2 * i + 1] = es[i]
            for j in range(nstates):
                for k in range(3):
                    hm[2 * i:2 * i + 2, 2 * j:2 * j + 2] += \
                        pref * t[i, j, k] * SPIN[k]
        w = np.linalg.eigvalsh(0.5 * (hm + hm.conj().T))
        out[name] = dict(
            splitting_cm=float((w[2] - w[0]) * CM),
            levels_cm=[float((x - es[0]) * CM) for x in w],
            hermiticity=float(np.abs(hm - hm.conj().T).max()))
    return out


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def run_fp_paper():
    mol = build_gef("linear", {"Ge": "ecpds28mdfso"})
    ecp = decp.ecp_soc(mol)
    # Primary run: the SCF/CASSCF settings of the frozen gef_recalc setup.
    _, mc, es = sa_casscf(mol, 2, 6, 2, scf_tol=1e-10, cas_tol=1e-9,
                          grad_tol=1e-5, spin_sector=1,
                          guesses=("minao", "atom", "huckel"))
    old = old_route(mc, es, 2, 6, mol.nelectron - 4, {"ecp": (ecp, -1j)})
    new = determinant_route(mol, mc, -ecp, 2, 6, mol.nelectron - 4)
    # Cross-check: the archived validate_ecp_somf / t452 tolerances.  The
    # two runs differ only in the converged SA-CASSCF point; the spread is
    # recorded so that the 5e-5 cm^-1 ambiguity is visible.
    _, mc2, es2 = sa_casscf(mol, 2, 6, 2)
    old2 = old_route(mc2, es2, 2, 6, mol.nelectron - 4,
                     {"ecp": (ecp, -1j)})
    new2 = determinant_route(mol, mc2, -ecp, 2, 6,
                             mol.nelectron - 4)
    return dict(system="GeF linear 3.4 Bohr, ECPso",
                ecp={"Ge": "ecpds28mdfso"}, basis={"Ge": "lanl2dz",
                                                   "F": "6-31g"},
                settings=dict(scf_tol=1e-10, cas_tol=1e-9,
                              grad_tol=1e-5, spin_sector=1,
                              guesses=["minao", "atom", "huckel"]),
                casscf_e=float(mc.e_tot),
                route_old=old["ecp"], route_new=new,
                ratio_old_over_new=old["ecp"]["splitting_cm"] / new["span_cm"],
                crosscheck_archived_settings=dict(
                    settings=dict(scf_tol=1e-12, cas_tol=1e-10,
                                  guesses=["default"]),
                    casscf_e=float(mc2.e_tot),
                    route_old=old2["ecp"], route_new=new2,
                    ratio_old_over_new=(old2["ecp"]["splitting_cm"]
                                        / new2["span_cm"])))


def run_validate():
    mol = build_gef("bent", {"Ge": "ecpds28mdfso"})
    _, mc, es = sa_casscf(mol, 2, 6, 2)
    ncore, nact = 2, 6
    nele_act = mol.nelectron - 2 * ncore
    C = np.asarray(mc.mo_coeff)
    ca = C[:, ncore:ncore + nact]
    ci = np.asarray(mc.ci)
    d00 = np.asarray(mc.fcisolver.trans_rdm1(ci[0], ci[0], nact, nele_act))
    p_sa = np.asarray(mc.make_rdm1(), dtype=float)
    p_sa = 0.5 * (p_sa + p_sa.T)
    p_act_ao = ca @ d00 @ ca.T
    p_act_ao = 0.5 * (p_act_ao + p_act_ao.T)
    ecp = decp.ecp_soc(mol)
    nuc_f = decp.nuclear_soc(mol, decp.nuclear_centers(mol))
    nuc_all = decp.nuclear_soc(mol, range(mol.natm))
    h2_sa = dsomf.somf_two_electron(mol, p_sa, contract="explicit")
    h2_act = dsomf.somf_two_electron(mol, p_act_ao, contract="explicit")
    old_ops = {
        "ecp": (ecp, -1j),
        "somf": (A2 * (nuc_f + h2_sa) - ecp, 1j),
        "somf_active_density": (A2 * (nuc_f + h2_act) - ecp, 1j),
        "nuclear": (A2 * nuc_f, 1j),
        "double": (A2 * (nuc_all + h2_sa) - ecp, 1j),
    }
    new_ops = {
        "ecp": -ecp,
        "somf": A2 * (nuc_f + h2_sa) - ecp,
        "somf_active_density": A2 * (nuc_f + h2_act) - ecp,
        "nuclear": A2 * nuc_f,
        "double": A2 * (nuc_all + h2_sa) - ecp,
    }
    old = old_route(mc, es, ncore, nact, nele_act, old_ops)
    out = {}
    for name, hk in new_ops.items():
        new = determinant_route(mol, mc, hk, ncore, nact, nele_act)
        out[name] = dict(old=old[name], new=new,
                         ratio_old_over_new=(old[name]["splitting_cm"]
                                             / new["span_cm"]))
    return dict(system="GeF bent, SO-ECP (validate_ecp_somf setup)",
                basis={"Ge": "lanl2dz", "F": "6-31g"},
                ecp={"Ge": "ecpds28mdfso"},
                settings=dict(scf_tol=1e-12, cas_tol=1e-10, guesses=["default"]),
                casscf_e=float(mc.e_tot),
                variants=out)


def run_compare_orca():
    mol = build_gef("bent", {"Ge": "lanl2dz"})
    _, mc, es = sa_casscf(mol, 2, 6, 2)
    ncore, nact = 2, 6
    nele_act = mol.nelectron - 2 * ncore
    p = np.asarray(mc.make_rdm1(), dtype=float)
    p = 0.5 * (p + p.T)
    nuc = decp.nuclear_soc(mol, range(mol.natm))
    h2 = dsomf.somf_two_electron(mol, p, contract="explicit")
    old_ops = {
        "nuclear": (A2 * nuc, 1j),
        "somf": (A2 * (nuc + h2), 1j),
        "two_electron_only": (A2 * h2, 1j),
    }
    new_ops = {"nuclear": A2 * nuc, "somf": A2 * (nuc + h2),
               "two_electron_only": A2 * h2}
    old = old_route(mc, es, ncore, nact, nele_act, old_ops)
    out = {}
    for name, hk in new_ops.items():
        new = determinant_route(mol, mc, hk, ncore, nact, nele_act)
        out[name] = dict(old=old[name], new=new,
                         ratio_old_over_new=(old[name]["splitting_cm"]
                                             / new["span_cm"]))
    orca = {"soc_type_1_cm": 47.87, "soc_type_3_cm": 26.00,
            "two_electron_only_cm": 14.14}
    return dict(system="GeF bent, scalar ECP (compare_orca_gef setup)",
                basis={"Ge": "lanl2dz", "F": "6-31g"},
                ecp={"Ge": "lanl2dz"},
                settings=dict(scf_tol=1e-12, cas_tol=1e-10, guesses=["default"]),
                casscf_e=float(mc.e_tot),
                orca=orca,
                ratio_pyscf_new=(out["somf"]["new"]["span_cm"]
                                 / out["nuclear"]["new"]["span_cm"]),
                ratio_orca=orca["soc_type_3_cm"] / orca["soc_type_1_cm"],
                variants=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HERE, "results_ecp_somf_gef_recalc.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "route_new": ("determinant-space spin-orbital CI "
                         "(diffmrc.core.spinor_ci), SOC as a one-body term"),
           "route_old": ("state interaction from pyscf trans_rdm1 transition "
                         "densities (spin-summed); archived, history only"),
           "units": "cm^-1",
           "alpha2_half": A2,
           "systems": {}}
    for name, fn in (("fp_paper", run_fp_paper),
                     ("validate", run_validate),
                     ("compare_orca", run_compare_orca)):
        print("== %s ==" % name, flush=True)
        res["systems"][name] = fn()
        print(json.dumps(res["systems"][name], indent=1), flush=True)
    res["runtime_s"] = time.time() - t0
    out = os.path.abspath(args.out)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
