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
from pyscf import fci, gto, scf  # noqa: E402

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
