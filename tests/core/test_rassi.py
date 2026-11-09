"""Contracts of the RASSI state-interaction module.

Two layers are tested.

Unit properties on hand-built model spaces:

* determinant-space ordering and S_- normalization;
* S^2 eigenvalues of the generated multiplets;
* Hermiticity of the spin-orbit operator and of the state-interaction
  matrix;
* equivalence of the two assembly routes (determinant projection versus
  spin-resolved transition densities);
* the Delta S = 0, +-1 selection rule.

End-to-end OH X ^2Pi benchmark (geometry and protocol of the project SOC
benchmark): ROHF/6-31G, r = 1.8324 Bohr, SA-CASSCF(7,6) with two roots,
model space of two spatial components times two M_S components.  The
frozen splittings are 205.883 cm^-1 (one-electron Breit-Pauli) and
128.829 cm^-1 (SOMF), anchored against ORCA 6.1.1.  The same run checks
the two assembly routes and the Omega = +-1/2, +-3/2 assignment.
"""
import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, mcscf, scf  # noqa: E402

from diffmrc.core import rassi  # noqa: E402
from diffmrc.core import somf as dsomf  # noqa: E402
from diffmrc.core import spinor_ci as sci  # noqa: E402


CM = 219474.6313702
TOL_CM = 5e-3
OH_GEOMETRY = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8324))]
OH_REFERENCE = {"one_electron": 205.883, "somf": 128.829}


# ---------------------------------------------------------------------------
# Unit-level helpers
# ---------------------------------------------------------------------------
def random_soc(norb, seed=7):
    """Random real antisymmetric (3, norb, norb) tensor."""
    rng = np.random.default_rng(seed)
    hk = rng.normal(size=(3, norb, norb))
    return 0.5 * (hk - hk.transpose(0, 2, 1))


def random_spin_free_integrals(norb, seed=5):
    """Random spatial integrals with the 8-fold permutational symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    g = rng.normal(size=(norb, norb, norb, norb))
    g = 0.25 * (g + g.transpose(1, 0, 2, 3)
                + g.transpose(0, 1, 3, 2) + g.transpose(1, 0, 3, 2))
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    nso = 2 * norb
    h1so = np.zeros((nso, nso))
    eri = np.zeros((nso, nso, nso, nso))
    for p in range(norb):
        for q in range(norb):
            h1so[2 * p, 2 * q] = h1[p, q]
            h1so[2 * p + 1, 2 * q + 1] = h1[p, q]
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    for a in (0, 1):
                        for b in (0, 1):
                            eri[2 * p + a, 2 * q + a,
                                2 * r + b, 2 * s + b] = g[p, q, r, s]
    return h1so, eri


def det_ms(space, i):
    det = space.dets[i]
    na = sum(1 for x in det if x % 2 == 0)
    return 0.5 * (na - (len(det) - na))


def det_vector(space, occ):
    vec = np.zeros(len(space), dtype=complex)
    vec[space.index[sum(1 << i for i in occ)]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_lowering_triplet():
    space = rassi.DetSpace(2, 2)
    # M_S = 1: both alpha spin orbitals (0 and 2) occupied.
    high = rassi.SpinFreeState(0.0, 1.0, 1.0, det_vector(space, (0, 2)))
    m0 = rassi.lower_component(high, space)
    expected = np.zeros(len(space), dtype=complex)
    expected[space.index[(1 << 1) | (1 << 2)]] = 1.0 / np.sqrt(2.0)
    expected[space.index[(1 << 0) | (1 << 3)]] = 1.0 / np.sqrt(2.0)
    assert np.allclose(m0.ci, expected)
    assert abs(m0.ms - 0.0) < 1e-12


def test_s2_eigenvalues():
    space = rassi.DetSpace(2, 2)
    _, s2 = rassi.spin_operators(space)
    triplet = rassi.build_multiplet(
        rassi.SpinFreeState(0.0, 1.0, 1.0, det_vector(space, (0, 2))), space)
    for st in triplet:
        val = float(np.real(st.ci.conj() @ s2 @ st.ci))
        assert abs(val - 2.0) < 1e-10
    singlet = rassi.SpinFreeState(
        0.0, 0.0, 0.0,
        (det_vector(space, (0, 3)) - det_vector(space, (1, 2))) / np.sqrt(2.0))
    val = float(np.real(singlet.ci.conj() @ s2 @ singlet.ci))
    assert abs(val) < 1e-10
    assert abs(np.vdot(singlet.ci, triplet[1].ci)) < 1e-10


def test_two_routes_agree():
    space = rassi.DetSpace(3, 3)
    hk = random_soc(3)
    o = rassi.soc_one_body(hk, 3)
    assert np.allclose(o, o.conj().T)

    # Build a few states from a random Hermitian spin-free Hamiltonian.
    rng = np.random.default_rng(11)
    a = rng.normal(size=(len(space), len(space)))
    h0 = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(h0)
    states = [rassi.SpinFreeState(float(w[i]), 1.5, 1.5, v[:, i])
              for i in range(3)]
    states = [s for st in states for s in rassi.build_multiplet(st, space)]
    h_a = rassi.soc_matrix_determinant(
        states, rassi.one_body_matrix(o, space))
    h_b = rassi.soc_matrix_transition(states, hk, space)
    assert np.allclose(h_a, h_b, atol=1e-10)
    assert np.allclose(h_a, h_a.conj().T, atol=1e-10)


def test_delta_s_selection():
    space = rassi.DetSpace(3, 4)
    _, s2 = rassi.spin_operators(space)
    h0 = rassi.ci_matrix(*random_spin_free_integrals(3, seed=5), space)
    w, v = np.linalg.eigh(h0)
    ms_diag = np.array([det_ms(space, i) for i in range(len(space))])
    states = []
    for i in range(len(w)):
        ms = float(np.sum(np.abs(v[:, i]) ** 2 * ms_diag))
        s2v = float(np.real(v[:, i].conj() @ s2 @ v[:, i]))
        s = 0.5 * (np.sqrt(max(1.0 + 4.0 * s2v, 0.0)) - 1.0)
        s = round(2 * s) / 2
        if abs(ms - s) > 1e-6:
            continue
        high = rassi.SpinFreeState(float(w[i]), s, ms, v[:, i])
        states.extend(rassi.build_multiplet(high, space))

    hk = random_soc(3)
    h = rassi.soc_matrix_determinant(
        states, rassi.one_body_matrix(rassi.soc_one_body(hk, 3), space))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if abs(si.s - sj.s) > 1.0 + 1e-9:
                assert abs(h[i, j]) < 1e-10, (si.s, sj.s, h[i, j])


# ---------------------------------------------------------------------------
# OH benchmark
# ---------------------------------------------------------------------------
def _spin_orbital_integrals(h1, h2, norb):
    """Embed spatial active-space integrals in the spin-orbital basis."""
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
                    for a in (0, 1):
                        for b in (0, 1):
                            eri[2 * p + a, 2 * q + a,
                                2 * r + b, 2 * s + b] = v
    return h1so, eri


def _angular_momentum_mo(mol, c_act):
    a = np.asarray(mol.intor("int1e_cg_irxp"), dtype=float)
    lz = c_act.T @ (1j * a[2]) @ c_act
    return 0.5 * (lz + lz.conj().T)


def _sector_indices(space):
    """Determinant indices grouped by M_S = (N_alpha - N_beta)/2."""
    groups = {}
    for i, det in enumerate(space.dets):
        na = sum(1 for x in det if x % 2 == 0)
        nb = len(det) - na
        groups.setdefault(0.5 * (na - nb), []).append(i)
    return groups


def _diagonalize_sectors(h0, space):
    """Diagonalize each M_S block; return states with definite M_S."""
    out = []
    for ms, idx in sorted(_sector_indices(space).items(), reverse=True):
        idx = np.array(idx)
        w, v = np.linalg.eigh(h0[np.ix_(idx, idx)])
        for k in range(len(w)):
            full = np.zeros(len(space), dtype=complex)
            full[idx] = v[:, k]
            out.append((float(w[k]), float(ms), full))
    return out


def _classify_and_build(h0, space, emax, s2_mat):
    """Return multiplets (lists of SpinFreeState) below ``emax``."""
    states = _diagonalize_sectors(h0, space)
    multiplets = []
    for energy, ms, vec in states:
        if energy > emax:
            continue
        s2 = float(np.real(vec.conj() @ s2_mat @ vec))
        s = 0.5 * (np.sqrt(max(1.0 + 4.0 * s2, 0.0)) - 1.0)
        s = round(s * 2.0) / 2.0
        if abs(ms - s) > 1e-6:
            continue
        high = rassi.SpinFreeState(energy, s, ms, vec,
                                   label="S=%.1f,E=%.6f" % (s, energy))
        multiplets.append(rassi.build_multiplet(high, space))
    multiplets.sort(key=lambda m: m[0].energy)
    return multiplets


def _rotate_group_by_lz(group, lz_mat):
    """Rotate a degenerate group into eigenstates of L_z."""
    if len(group) < 2:
        return group
    v = np.column_stack([g.ci for g in group])
    m = v.conj().T @ lz_mat @ v
    _, u = np.linalg.eigh(0.5 * (m + m.conj().T))
    out = []
    for k in range(len(group)):
        ci = v @ u[:, k]
        lam = float(np.real(ci.conj() @ lz_mat @ ci))
        out.append(rassi.SpinFreeState(
            group[0].energy, group[0].s, group[0].ms, ci,
            label="S=%.1f,Lam=%+.1f" % (group[0].s, round(lam, 1))))
    return out


@pytest.fixture(scope="module")
def oh():
    """SA-CASSCF(7,6) reference and scalar determinant Hamiltonian of OH."""
    mol = gto.M(atom=OH_GEOMETRY, basis="6-31g", unit="Bohr", spin=1,
                verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    assert mf.converged

    nroots = 2
    mc = mcscf.CASSCF(mf, 6, 7)
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([1.0 / nroots] * nroots)
    mc.fcisolver.nroots = nroots
    mc.kernel()
    assert mc.converged

    ncore, ncas = int(mc.ncore), int(mc.ncas)
    nele = int(sum(mc.nelecas)) if isinstance(mc.nelecas, (tuple, list)) \
        else int(mc.nelecas)
    c = np.asarray(mc.mo_coeff)
    c_act = c[:, ncore:ncore + ncas]
    h1, _ = mc.get_h1eff()
    h2 = np.asarray(mc.get_h2eff())
    if h2.ndim == 2:
        h2 = ao2mo.restore(1, h2, ncas)

    p_ao = np.asarray(mc.make_rdm1())
    if p_ao.ndim == 3:
        p_ao = p_ao[0] + p_ao[1]
    p_ao = 0.5 * (p_ao + p_ao.T)

    h1so, eri = _spin_orbital_integrals(h1, h2, ncas)
    h0, dets, ecore_ci = sci.spinor_ci_matrix(h1so, eri, 0, 2 * ncas, nele)
    space = rassi.DetSpace(ncas, nele)
    assert [tuple(d) for d in dets] == [tuple(d) for d in space.dets]
    h0 = h0 + ecore_ci * np.eye(len(space))

    lz_mo = _angular_momentum_mo(mol, c_act)
    lz_so = np.zeros((2 * ncas, 2 * ncas), dtype=complex)
    for p in range(ncas):
        for q in range(ncas):
            lz_so[2 * p, 2 * q] = lz_mo[p, q]
            lz_so[2 * p + 1, 2 * q + 1] = lz_mo[p, q]
    lz_mat = rassi.one_body_matrix(lz_so, space)
    sz_mat, s2_mat = rassi.spin_operators(space)

    return dict(mol=mol, space=space, h0=h0, lz=lz_mat, sz=sz_mat,
                s2=s2_mat, c_act=c_act, p_ao=p_ao, ncas=ncas)


@pytest.fixture(scope="module")
def oh_model(oh):
    """Spin-free ^2Pi model space with states rotated into L_z eigenstates."""
    space = oh["space"]
    e_ground = float(np.linalg.eigvalsh(oh["h0"])[0])
    multiplets = _classify_and_build(oh["h0"], space, e_ground + 0.05,
                                     oh["s2"])

    by_term = {}
    for m in multiplets:
        high = m[0]
        by_term.setdefault((round(high.energy, 6), high.s), []).append(high)
    multiplets = []
    for highs in by_term.values():
        for st in _rotate_group_by_lz(highs, oh["lz"]):
            multiplets.append(rassi.build_multiplet(st, space))
    multiplets.sort(key=lambda m: m[0].energy)
    states = [st for m in multiplets for st in m]

    return dict(states=states, multiplets=multiplets,
                lz_states=rassi.soc_matrix_determinant(states, oh["lz"]),
                sz_states=rassi.soc_matrix_determinant(states, oh["sz"]))


@pytest.mark.parametrize("mode", ["one_electron", "somf"])
def test_oh_term_splitting(oh, oh_model, mode):
    """Frozen OH ^2Pi splitting, route equivalence, Hermiticity and Omega."""
    states = oh_model["states"]
    jz_states = oh_model["lz_states"] + oh_model["sz_states"]

    if mode == "one_electron":
        hk = dsomf.somf_matrix(oh["mol"], oh["c_act"], None,
                               mode="one_electron", prefactor=True)
    else:
        hk = dsomf.somf_matrix(oh["mol"], oh["c_act"], oh["p_ao"],
                               mode="somf", prefactor=True)
    hk = np.asarray(hk, dtype=float)

    o_det = rassi.one_body_matrix(rassi.soc_one_body(hk, oh["ncas"]),
                                  oh["space"])
    h_soc = rassi.soc_matrix_determinant(states, o_det)
    h_soc_b = rassi.soc_matrix_transition(states, hk, oh["space"])
    assert float(np.abs(h_soc - h_soc_b).max()) < 1e-12

    so, diag = rassi.diagonalize(states, h_soc, jz_states)
    assert diag["hermiticity"] < 1e-10

    energies = np.array([s.energy for s in so])
    groups = rassi.kramers_pairs(energies)
    assert [len(g) for g in groups] == [2, 2]
    splitting = float((energies[groups[1][0]] - energies[0]) * CM)
    assert splitting == pytest.approx(OH_REFERENCE[mode], abs=TOL_CM)

    # ^2Pi: Omega = +-1/2 for the lower and +-3/2 for the upper Kramers pair.
    omega = sorted(abs(float(so[i].omega)) for i in range(len(so)))
    assert omega == pytest.approx([0.5, 0.5, 1.5, 1.5], abs=1e-6)


def test_oh_determinant_space_size(oh):
    """The CAS(7,6) active space gives the expected determinant count."""
    assert len(oh["space"]) == len(list(itertools.combinations(range(12), 7)))
