"""Contracts of the zero-field-splitting kernel against pyscf references.

The module under test builds a spin-free determinant Hamiltonian, its
spin-summed active-space 1-RDM and the one-electron spin operator on the
same bitmap convention.  Three independent pyscf routes are used as
references:

* ``pyscf.fci.direct_spin1.FCI.kernel`` for the lowest energy of every
  M_S block;
* ``pyscf.fci.spin_op.spin_square`` for the spin label of that root;
* ``FCI.make_rdm1`` for the natural occupations of the same root.

The bitmap convention of the module is documented in
``diffmrc.core.zfs``: bit ``2*j`` is the alpha and bit ``2*j+1`` the beta
spin orbital of active orbital ``j``.  The frozen core is closed shell,
so the constant bit shift relative to the kernel convention does not
change <S^2>.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci, gto, mcscf, scf  # noqa: E402

from diffmrc.analysis.states import spin_square  # noqa: E402
from diffmrc.core import zfs as zc  # noqa: E402


NCORE = 2
NACT = 4
# OH/6-31G has nine electrons; CAS(4,4) with two frozen core orbitals
# leaves five active electrons.
GEOMETRY = "O 0 0 0; H 0 0 1.8"


@pytest.fixture(scope="module")
def oh():
    """ROHF reference and active-space integrals of OH/6-31G."""
    mol = gto.M(atom=GEOMETRY, basis="6-31g", spin=1, verbose=0, unit="B")
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    h1, eri, _ = zc.active_integrals(mol, mf.mo_coeff, NCORE, NACT)
    nele = mol.nelectron - 2 * NCORE
    blocks, all_masks, index = zc.det_space(NACT, nele)
    return dict(mol=mol, mf=mf, h1=h1, eri=eri, nele=nele, blocks=blocks,
                all_masks=all_masks, index=index)


def test_det_space_partition(oh):
    """The M_S blocks partition all determinants of the active space."""
    from math import comb
    blocks = oh["blocks"]
    assert sum(len(m) for m in blocks.values()) == comb(2 * NACT, oh["nele"])
    assert sum(len(m) for m in blocks.values()) == len(oh["all_masks"])
    assert len(oh["index"]) == len(oh["all_masks"])


def test_block_energies_match_pyscf_fci(oh):
    """Every M_S block: lowest eigenvalue and <S^2> against pyscf FCI."""
    h1, eri = oh["h1"], oh["eri"]
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    for key in sorted(oh["blocks"]):
        masks = oh["blocks"][key]
        ham = zc.block_hamiltonian(h1, eri, masks, NACT)
        w, v = np.linalg.eigh(ham)
        e_ref, c_ref = solver.kernel(h1, eri, NACT, key)
        assert abs(float(w[0]) - float(e_ref)) < 1e-9, key
        s2 = float(spin_square(v[:, 0], masks))
        s2_ref = fci.spin_op.spin_square(c_ref, NACT, key)[0]
        assert abs(s2 - float(s2_ref)) < 1e-6, key


def test_natural_occupations_match_pyscf(oh):
    """Spin-summed active-space 1-RDM against pyscf ``make_rdm1``."""
    h1, eri = oh["h1"], oh["eri"]
    key = (2, 3)
    masks = oh["blocks"][key]
    ham = zc.block_hamiltonian(h1, eri, masks, NACT)
    w, v = np.linalg.eigh(ham)
    conn = zc.build_connections(oh["all_masks"], oh["index"], NACT)
    rdm = zc.rdm1_state(dict(ci=np.ascontiguousarray(v[:, 0]), masks=masks),
                        conn, oh["index"])
    occ = np.sort(np.linalg.eigvalsh(rdm))[::-1]

    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    e_ref, c_ref = solver.kernel(h1, eri, NACT, key)
    rdm_ref = solver.make_rdm1(c_ref, NACT, key)
    occ_ref = np.sort(np.linalg.eigvalsh(rdm_ref))[::-1]

    assert abs(float(np.trace(rdm)) - float(oh["nele"])) < 1e-8
    assert float(np.abs(occ - occ_ref).max()) < 1e-7


def test_spin_operator_reproduces_ms(oh):
    """The one-electron S operator gives <S_z> = M_S and vanishing <S_x>, <S_y>."""
    key = (2, 3)
    masks = oh["blocks"][key]
    ham = zc.block_hamiltonian(oh["h1"], oh["eri"], masks, NACT)
    w, v = np.linalg.eigh(ham)
    ci = np.ascontiguousarray(v[:, 0])
    state = dict(ci=ci, masks=masks)
    conn = zc.build_connections(oh["all_masks"], oh["index"], NACT)

    values = []
    for sp in zc.spin_orbital_components(np.zeros((3, NACT, NACT)), "S"):
        mat = zc.onee_matrix(conn, sp)
        values.append(zc.state_matrix([state], mat, oh["index"])[0, 0].real)

    assert abs(values[2] - 0.5 * (key[0] - key[1])) < 1e-10
    assert max(abs(values[0]), abs(values[1])) < 1e-10

    s2 = float(spin_square(ci, masks))
    assert abs(s2 - 0.75) < 1e-6


def test_l_operator_phase_matches_pnucxp():
    """The physical L uses the opposite raw sign to ``int1e_pnucxp``.

    ``int1e_cg_irxp`` and ``int1e_pnucxp`` are both r x p type tensors
    but libcint stores them with opposite signs (see the module
    docstring).  On a single atom with C = identity the dominant
    <p_x|.|p_y> element is +1 for ``irxp`` and negative for ``pnucxp``;
    the module must apply ``I_L = -1j`` to the former so that the
    orbital Zeeman operator and the SOC orbital part share one phase.
    """
    mol = gto.M(atom="B 0 0 0", basis="sto-3g", spin=1, verbose=0,
                unit="B")
    irxp = mol.intor("int1e_cg_irxp", 3)
    pnucxp = mol.intor("int1e_pnucxp", 3)
    # p_x and p_y are AOs 2 and 3 in the minimal basis.
    assert irxp[2, 2, 3] > 0.9
    assert pnucxp[2, 2, 3] < -1.0
    # The module's physical L_z element is -1j * irxp.
    l3 = zc.l_mo_operator(mol, np.eye(mol.nao_nr()))
    lz = zc.spin_orbital_components(l3, "L")[2]
    assert abs(lz[2 * 2, 2 * 3] - (-1j * irxp[2, 2, 3])) < 1e-12
    assert abs(zc.I_L - (-zc.I_SOC)) < 1e-15


def test_g_tensor_pure_spin_doublets():
    """Spin-only doublets reproduce the analytic pseudo-spin g values.

    For the |+/-M> doublet of a spin-S multiplet with ``M = L + g_e S``
    and L = 0, the projection gives ``g_parallel = 2 M g_e`` and
    ``g_perp = (S + 1/2) g_e`` for M = 1/2.  This is the benchmark that
    a quenched-orbital system must meet; it is *not* ``g_e`` unless
    S = 1/2.
    """
    cases = [
        (0.5, 0.5, [zc.GE, zc.GE, zc.GE]),
        (1.5, 0.5, [2.0 * zc.GE, 2.0 * zc.GE, zc.GE]),
        (2.5, 0.5, [3.0 * zc.GE, 3.0 * zc.GE, zc.GE]),
        (2.5, 2.5, [5.0 * zc.GE, 0.0, 0.0]),
    ]
    for S, M, expected in cases:
        ms = np.arange(-S, S + 1.0, 1.0)
        sz, sx, sy = zc.spin_matrices(S, ms)
        mats = [zc.GE * sx, zc.GE * sy, zc.GE * sz]
        vecs = np.zeros((len(ms), 2), dtype=complex)
        vecs[int(np.argmin(np.abs(ms + M))), 0] = 1.0
        vecs[int(np.argmin(np.abs(ms - M))), 1] = 1.0
        _, sv = zc.g_tensor_doublet(vecs, mats, 2)
        assert np.abs(sv - np.asarray(expected)).max() < 1e-10, (S, M, sv)


def test_g_tensor_rejects_non_kramers_pair():
    """A pair with a non-zero projected trace is not a Kramers doublet."""
    ms = np.array([-1.0, 0.0, 1.0])
    sz, sx, sy = zc.spin_matrices(1.0, ms)
    mats = [zc.GE * sx, zc.GE * sy, zc.GE * sz]
    vecs = np.zeros((3, 2), dtype=complex)
    vecs[2, 0] = 1.0     # M_S = +1
    vecs[1, 1] = 1.0     # M_S = 0
    with pytest.raises(ValueError, match="Kramers pair"):
        zc.g_tensor_doublet(vecs, mats, 2)


def test_g_tensor_rejects_bad_shapes():
    """Only a two-state Kramers doublet has a pseudo-spin-1/2 g-tensor."""
    with pytest.raises(ValueError, match="ndoublet=2"):
        zc.g_tensor_doublet(np.eye(3), [np.eye(3)] * 3, ndoublet=3)
    vecs = np.zeros((3, 2), dtype=complex)
    vecs[0, 0] = vecs[1, 0] = 1.0     # first column not normalized
    vecs[2, 1] = 1.0
    with pytest.raises(ValueError, match="orthonormal"):
        zc.g_tensor_doublet(vecs, [np.eye(3)] * 3, 2)


def test_solve_scalar_warns_when_nkeep_truncates_window(oh):
    """A per-block cap inside the energy window must be visible."""
    with pytest.warns(RuntimeWarning, match="truncates states"):
        zc.solve_scalar(oh["h1"], oh["eri"], NACT, oh["nele"],
                        nkeep=1, ecut=1.0)


@pytest.fixture(scope="module")
def b_atom_p1():
    """B/STO-3G in the AO basis: an exact 2s2 2p1 2P ground state."""
    mol = gto.M(atom="B 0 0 0", basis="sto-3g", spin=1, verbose=0,
                unit="B")
    C = np.eye(mol.nao_nr())
    ncore, nact, nele = 1, 4, 3
    h1, eri, _ = zc.active_integrals(mol, C, ncore, nact)
    states = zc.solve_scalar(h1, eri, nact, nele, nkeep=24, ecut=2.0)
    blocks, all_masks, index = zc.det_space(nact, nele)
    conn = zc.build_connections(all_masks, index, nact)
    rdm = zc.rdm1_state(states[0], conn, index)
    nmo = C.shape[1]
    P_mo = np.zeros((nmo, nmo))
    for i in range(ncore):
        P_mo[i, i] = 2.0
    P_mo[ncore:ncore + nact, ncore:ncore + nact] = rdm
    P_ao = C @ P_mo @ C.T
    # One-electron SOC: exact for this analytic test.  The SOMF mean
    # field of a single 2P component breaks the spherical symmetry (see
    # the module docstring).
    h3 = zc.soc_mo_operator(mol, C, P_ao, mode="one_electron")
    h3a = h3[:, ncore:ncore + nact, ncore:ncore + nact]
    hso = zc.state_matrix(states, zc.onee_matrix(
        conn, zc.spin_orbital_blocks(h3a, "soc")), index)
    vals, vecs, _ = zc.si_diagonalize(states, hso)
    l3 = zc.l_mo_operator(mol, C)[:, ncore:ncore + nact, ncore:ncore + nact]
    smat = [zc.state_matrix(states, zc.onee_matrix(conn, m), index)
            for m in zc.spin_orbital_components(
                np.zeros((3, nact, nact)), "S")]
    lmat = [zc.state_matrix(states, zc.onee_matrix(conn, m), index)
            for m in zc.spin_orbital_components(l3, "L")]
    mats = [lmat[k] + zc.GE * smat[k] for k in range(3)]
    return vals, vecs, mats


def test_g_tensor_analytic_p1_atom(b_atom_p1):
    """B/STO-3G 2P_1/2 doublet: the analytic Lande factor, isotropic.

    This is the end-to-end regression for the operator phases: with the
    old ``I_SOC`` phase on ``int1e_cg_irxp`` the same calculation gives
    2.0008 instead of ``1 - (g_e - 1)/3 = 0.665894``.
    """
    vals, vecs, mats = b_atom_p1
    # 2P_1/2 is the lowest doublet; 2P_3/2 is a fourfold level above it.
    assert abs((vals[1] - vals[0]) * zc.CM1) < 1e-6
    assert (vals[2] - vals[0]) * zc.CM1 > 1.0
    g, sv = zc.g_tensor_doublet(vecs, mats, 2)
    analytic = 1.0 - (zc.GE - 1.0) / 3.0
    assert np.abs(sv - analytic).max() < 1e-5, sv
    # The g matrix itself is pseudo-spin-frame dependent; its singular
    # values are the isotropic analytic value.
    assert g.shape == (3, 3)


def test_g_tensor_analytic_oh_pi_doublet():
    """OH 2Pi_{3/2}: g_parallel = 2 (Lambda + g_e Sigma) = 2 + g_e.

    A clean Hund's case (a) 2Pi state has Lambda = 1, Sigma = 1/2,
    Omega = 3/2; the two doublet components carry
    M_z = +/- (1 + g_e/2), so the pseudo-spin g is (2 + g_e, 0, 0).
    This is the molecular counterpart of the p^1 atomic test: the
    orbital angular momentum is unquenched, so the orbital Zeeman phase
    enters at first order.
    """
    mol = gto.M(atom=[("O", (0, 0, 0)), ("H", (0, 0, 1.8324))],
                basis="6-31g", spin=1, unit="B", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASSCF(mf, 6, 7)
    mc.state_average_([0.5, 0.5])
    mc.fix_spin_(ss=0.75, shift=0.5)
    mc.conv_tol = 1e-9
    mc.verbose = 0
    mc.kernel()
    C = np.asarray(mc.mo_coeff, dtype=float)
    ncore, nact, nele = 1, 6, 7

    h1, eri, _ = zc.active_integrals(mol, C, ncore, nact)
    states = zc.solve_scalar(h1, eri, nact, nele, nkeep=400, ecut=1.0)
    blocks, all_masks, index = zc.det_space(nact, nele)
    conn = zc.build_connections(all_masks, index, nact)
    rdm = zc.rdm1_state(states[0], conn, index)
    nmo = C.shape[1]
    P_mo = np.zeros((nmo, nmo))
    for i in range(ncore):
        P_mo[i, i] = 2.0
    P_mo[ncore:ncore + nact, ncore:ncore + nact] = rdm
    P_ao = C @ P_mo @ C.T

    h3 = zc.soc_mo_operator(mol, C, P_ao, mode="somf")
    h3a = h3[:, ncore:ncore + nact, ncore:ncore + nact]
    hso = zc.state_matrix(states, zc.onee_matrix(
        conn, zc.spin_orbital_blocks(h3a, "soc")), index)
    vals, vecs, _ = zc.si_diagonalize(states, hso)
    # The 2Pi_{3/2} ground doublet is the lowest Kramers pair.
    assert abs((vals[1] - vals[0]) * zc.CM1) < 1e-6

    l3 = zc.l_mo_operator(mol, C)[:, ncore:ncore + nact, ncore:ncore + nact]
    smat = [zc.state_matrix(states, zc.onee_matrix(conn, m), index)
            for m in zc.spin_orbital_components(
                np.zeros((3, nact, nact)), "S")]
    lmat = [zc.state_matrix(states, zc.onee_matrix(conn, m), index)
            for m in zc.spin_orbital_components(l3, "L")]
    mats = [lmat[k] + zc.GE * smat[k] for k in range(3)]
    g, sv = zc.g_tensor_doublet(vecs, mats, 2)
    # The molecule lies along z; the doublet is axial.
    assert abs(abs(g[2, 2]) - (2.0 + zc.GE)) < 1e-3, g
    assert max(abs(g[0, :]).max(), abs(g[1, :]).max()) < 1e-2, g
