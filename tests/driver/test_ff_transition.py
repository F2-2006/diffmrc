"""The f-f transition method layer (driver.ff_transition).

The prototype's numerical layer (the f-f absolute-intensity pipeline) is
not part of the merged module, so the end-to-end case here runs the
method layer against a small state-averaged CASSCF and checks the
consistency of the table it produces.  The LiH gauge ladder is the
prototype's convergence criterion: the velocity form must approach the
length form as the wavefunction improves.

Note: these tests run real SCF and CASSCF.  On the local Windows
pyscf build the MKL thread layer must be sequential or the ao2mo/FCI
stage dies natively with 0xc06d007f (exit 127, no traceback); the
environment is therefore fixed before pyscf is imported.
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pyscf = pytest.importorskip("pyscf")  # noqa: E402
from pyscf import gto, mcscf, scf  # noqa: E402
from pyscf.fci import direct_spin1  # noqa: E402

from diffmrc.driver import ff_transition as ff  # noqa: E402

_FCI = direct_spin1.FCI()


def sa_run(atom, basis, charge, spin, ncas, nelecas, nroots):
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 400
    mf.kernel()
    assert mf.converged, "reference SCF did not converge"
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = 1e-10
    if nroots == 1:
        mc.kernel()
    else:
        mc = mcscf.state_average(mc, tuple([1.0 / nroots] * nroots))
        mc.kernel()
    assert mc.converged, "SA-CASSCF did not converge"
    return mol, mf, mc


@pytest.fixture(scope="module")
def lih_sto3g():
    """Minimal end-to-end case: LiH/STO-3G, CAS(2,2), three roots."""
    mol, _, mc = sa_run("Li 0 0 0; H 0 0 1.595", "sto-3g", 0, 0, 2, 2, 3)
    states = [{"E_hartree": e} for e in ff.e_states(mc)]
    rows, tau, mu, vl = ff.transition_table(mc, mol, states)
    return mol, mc, states, rows, tau, mu, vl


def test_the_ao_gradient_matrix_and_the_velocity_moments_are_antisymmetric():
    """The gradient integral is antisymmetric, so the velocity-gauge
    transition moments must be antisymmetric in the root pair."""
    worst = 0.0
    for atom, basis in (("Li 0 0 0; H 0 0 1.595", "cc-pvdz"),
                        ("H 0 0 0; H 0 0 0.7414", "cc-pvdz")):
        mol, _, mc = sa_run(atom, basis, 0, 0, 2, 2, 3)
        grad = -mol.intor("int1e_ipovlp", comp=3)
        worst = max(worst, float(
            np.abs(grad + np.transpose(grad, (0, 2, 1))).max()))
        _, vl = ff.transition_moments(mc, mol)
        assert np.abs(vl + np.transpose(vl, (1, 0, 2))).max() < 1e-12
    assert worst < 1e-12, f"max|G + G^T| = {worst:.2e} (threshold 1e-12)"


def test_the_permanent_dipole_agrees_with_pyscfs_ao_density_route():
    """The module contraction and pyscf's own AO density must give the
    same permanent dipole for LiH/CAS(2,2)."""
    mol, _, mc = sa_run("Li 0 0 0; H 0 0 1.595", "cc-pvdz", 0, 0, 2, 2, 1)
    dip_ao = mol.intor("int1e_r", comp=3)
    nuc = np.einsum("a,ax->x", mol.atom_charges(), mol.atom_coords())
    ci = ff.ci_list(mc)[0]
    dm1 = _FCI.make_rdm1(ci, mc.ncas, mc.nelecas)
    Cact = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    Ccore = mc.mo_coeff[:, :mc.ncore]
    route_a = nuc \
        - np.einsum("pq,xqp->x", dm1,
                    np.einsum("pi,xpq,qj->xij", Cact, dip_ao, Cact)) \
        - 2.0 * np.einsum("pi,xpq,qi->x", Ccore, dip_ao, Ccore)
    route_b = nuc - np.einsum("pq,xpq->x", mc.make_rdm1(), dip_ao)
    diff_d = float(np.abs(route_a - route_b).max() * ff.AU_DIPOLE_DEBYE)
    val_d = float(route_a[2] * ff.AU_DIPOLE_DEBYE)
    assert diff_d < 1e-8, f"|A-B| = {diff_d:.2e} D (threshold 1e-8 D)"
    assert abs(val_d + 5.596) < 0.02, f"dipole {val_d:.6f} D"


def test_the_velocity_gauge_converges_onto_the_length_gauge():
    """For an allowed transition the velocity form must approach the
    length form as the wavefunction improves (LiH X->A ladder)."""
    ladder = (("cc-pvdz", 2), ("cc-pvdz", 6), ("cc-pvtz", 6),
              ("cc-pvtz", 10), ("cc-pvqz", 10))
    rows = []
    for basis, ncas in ladder:
        mol, _, mc = sa_run("Li 0 0 0; H 0 0 1.595", basis, 0, 0, ncas, 2, 3)
        mu, vl = ff.transition_moments(mc, mol)
        energies = ff.e_states(mc)
        best = None
        m = len(ff.ci_list(mc))
        for i in range(m):
            for j in range(i + 1, m):
                dE = energies[j] - energies[i]
                f_l, f_v = ff.oscillator(dE, mu[i, j], vl[i, j])
                if f_l > 1e-6 and (best is None or f_l > best[0]):
                    best = (f_l, f_v, f_v / f_l)
        rows.append((f"{basis}/CAS(2,{ncas})", best[0], best[1], best[2]))
    first = abs(rows[0][3] - 1.0)
    last = abs(rows[-1][3] - 1.0)
    detail = "; ".join(f"{n}: f_L={fl:.4f} f_V={fv:.4f} ratio={r:.3f}"
                       for n, fl, fv, r in rows)
    assert last < 0.25, f"{detail} | |ratio-1| {first:.3f} -> {last:.3f}"
    assert last < first / 2.0, (
        f"{detail} | |ratio-1| {first:.3f} -> {last:.3f} "
        f"(threshold: < 0.25 and at least halved)")


def test_the_transition_table_is_internally_consistent(lih_sto3g):
    """End to end: the table covers every upward pair and its derived
    columns agree with their definitions."""
    _, mc, states, rows, tau, mu, vl = lih_sto3g
    assert mc.converged
    n = len(states)
    assert n == len(ff.ci_list(mc))
    assert len(rows) == n * (n - 1) // 2
    assert len(tau) == n
    for row in rows:
        assert row["f_len"] >= 0.0 and row["f_vel"] >= 0.0
        assert abs(row["dE_hartree"] * ff.HARTREE_TO_EV
                   - row["dE_eV"]) < 1e-12
        assert abs(row["dE_hartree"] * ff.HARTREE_TO_CM1
                   - row["dE_cm1"]) < 1e-6
        assert row["mu_len_norm_au"] == pytest.approx(
            float(np.linalg.norm(row["mu_len_au"])))
        assert row["mu_vel_norm_au"] == pytest.approx(
            float(np.linalg.norm(row["mu_vel_au"])))
        if row["f_len"] > 0.0:
            assert row["gauge_ratio"] == pytest.approx(
                row["f_vel"] / row["f_len"])
        if row["mu_len_norm_au"] > 0.0:
            assert row["velocity_energy_ratio"] == pytest.approx(
                row["mu_vel_norm_au"] / (row["dE_hartree"]
                                         * row["mu_len_norm_au"]))
    for item in tau:
        total = sum(r["A_s-1"] for r in rows if r["j"] == item["root"])
        assert item["sum_A_down_s-1"] == pytest.approx(total)
        if total > 0.0:
            assert item["tau_rad_s"] == pytest.approx(1.0 / total)
        else:
            assert item["tau_rad_s"] is None


def test_every_transition_row_carries_the_documented_fields(lih_sto3g):
    """Schema of the row records: every delivered key is present and the
    gauge bookkeeping is self-consistent."""
    _, _, _, rows, _, mu, vl = lih_sto3g
    required = {"i", "j", "dE_hartree", "dE_eV", "dE_cm1", "lambda_nm",
                "mu_len_au", "f_len", "f_vel", "gauge_ratio", "A_s-1"}
    assert rows
    for row in rows:
        assert required <= set(row)
        assert 0 <= row["i"] < row["j"] < mu.shape[0]
        assert len(row["mu_len_au"]) == 3
        assert len(row["mu_vel_au"]) == 3
        assert row["A_s-1"] == pytest.approx(
            ff.einstein_a(row["dE_hartree"], row["f_len"]))
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            assert np.allclose(mu[i, j], mu[j, i])
            assert np.allclose(vl[i, j], -vl[j, i])
