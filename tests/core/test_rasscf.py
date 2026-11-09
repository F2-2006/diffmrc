"""Contracts of the RASSCF module: determinant CI, densities and gradient.

Four independent references pin the determinant kernel on water/STO-3G:

* the Slater-Condon matrix elements against a brute-force summation over
  spin orbitals, on random integrals that carry the eightfold permutation
  symmetry;
* the assembled Hamiltonian element by element against pyscf
  ``direct_spin1`` on the same integrals;
* the lowest eigenvalue over the full determinant space against pyscf FCI
  (441 determinants, M_S = 0);
* the RAS(2,2) energy against pyscf CASSCF(2,2), which is the RAS-to-CAS
  limit of Malmqvist, J. Chem. Phys. 129 (2008) 164106, Sec. II B.

The orbital gradient is checked against central differences of the energy
over all 14 rotation parameters at amplitude 1e-2, with the same
exponential parameterization C = C0 exp(kappa) that the energy uses.  The
development report quotes 1e-6 Eh as the FCI criterion; the assertion below
keeps the tighter 1e-9 Eh of the development test (measured 1e-12 Eh).

A non-degenerate partition, RAS(1,2,2) with holes <= 1 and electrons <= 2,
is validated end to end against the same code with the constraints removed
(slow): 51 determinants against 100, and the constrained energy lies
1.542 mEh above the unconstrained one, as the variational principle
requires.
"""
import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, fci, gto, mcscf, scf  # noqa: E402
from pyscf.fci import cistring, direct_spin1  # noqa: E402

from diffmrc.core.rasscf import (  # noqa: E402
    RASSCF,
    apply_sequence,
    build_hamiltonian,
    enumerate_determinants,
    hamiltonian_element,
    solve_ci,
)

GEOMETRY = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"

# Frozen values of the development prototype (results/ras_demo.json).
E_RAS_CONSTRAINED = -75.008310663346
DELTA_RAS_CAS = 1.542e-3


@pytest.fixture(scope="module")
def h2o():
    """RHF/STO-3G reference of water: seven orbitals, ten electrons."""
    mol = gto.M(atom=GEOMETRY, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def _brute_element(aI, bI, aJ, bJ, n_orb, h1, eri):
    """<J|H|I> by explicit summation over spin orbitals."""
    val = 0.0
    # one-body
    for p in range(n_orb):
        for q in range(n_orb):
            for sp in (0, 1):
                for sq in (0, 1):
                    if sp != sq:
                        continue
                    res = apply_sequence(aI, bI, [(sq, q, False), (sp, p, True)])
                    if res is None:
                        continue
                    sgn, a, b = res
                    if (a, b) == (aJ, bJ):
                        val += sgn * h1[p, q]
    # two-body
    for p, q, r, s in itertools.product(range(n_orb), repeat=4):
        for sp, sq, sr, ss in itertools.product((0, 1), repeat=4):
            if sp != sq or sr != ss:
                continue
            res = apply_sequence(aI, bI, [(sq, q, False), (ss, s, False),
                                          (sr, r, True), (sp, p, True)])
            if res is None:
                continue
            sgn, a, b = res
            if (a, b) == (aJ, bJ):
                val += 0.5 * sgn * eri[p, q, r, s]
    return val


def test_matrix_elements():
    """Slater-Condon elements against a brute-force spin-orbital sum.

    Three orbitals, two alpha and two beta electrons, all 81 determinant
    pairs.  The random two-electron integrals are symmetrized to the full
    eightfold permutation symmetry of physical (pq|rs), in particular
    (pq|rs) = (rs|pq): the diagonal element keeps only (pp|qq), and the
    routine relies on that symmetry.  Without it the comparison fails at
    order one, which is a defect of such a test input, not of the kernel.
    """
    rng = np.random.default_rng(7)
    n_orb = 3
    h1 = rng.normal(size=(n_orb, n_orb))
    h1 = 0.5 * (h1 + h1.T)
    eri = rng.normal(size=(n_orb,) * 4)
    eri = (eri + eri.transpose(1, 0, 2, 3)
           + eri.transpose(0, 1, 3, 2) + eri.transpose(1, 0, 3, 2)) / 4.0
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))

    dets = [(a, b) for a in range(1 << n_orb) if a.bit_count() == 2
            for b in range(1 << n_orb) if b.bit_count() == 2]

    worst = 0.0
    n_checked = 0
    for (aI, bI) in dets:
        for (aJ, bJ) in dets:
            ref = _brute_element(aI, bI, aJ, bJ, n_orb, h1, eri)
            got = hamiltonian_element(aI, bI, aJ, bJ, n_orb, h1, eri)
            worst = max(worst, abs(ref - got))
            n_checked += 1
    assert n_checked == 81
    assert worst < 1e-12, f"matrix element mismatch {worst:.3e}"


def _pyscf_matrix(h1, eri, norb, na, nb):
    """Dense Hamiltonian from pyscf's direct_spin1 contraction."""
    strs_a = cistring.gen_strings4orblist(range(norb), na)
    strs_b = cistring.gen_strings4orblist(range(norb), nb)
    dim = len(strs_a) * len(strs_b)
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, (na, nb), 0.5)
    H = np.zeros((dim, dim))
    for k in range(dim):
        c = np.zeros(dim)
        c[k] = 1.0
        H[:, k] = direct_spin1.contract_2e(h2e, c, norb, (na, nb))
    return H, strs_a, strs_b


def test_matrix_vs_pyscf(h2o):
    """Every Hamiltonian element against pyscf on the same integrals.

    Four orbitals with two alpha and two beta electrons.  PySCF's own
    contraction sees the same integrals, so a sign or index error in the
    Slater-Condon routines shows up directly.
    """
    mol, mf = h2o
    norb_use = 4
    C = mf.mo_coeff[:, :norb_use]
    norb = norb_use
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.full(mol, C, compact=False).reshape(norb, norb, norb, norb)

    na = nb = 2
    dets = enumerate_determinants(norb, na, nb, 0, 0, 0, 0, 0)
    Hmine = build_hamiltonian(dets, norb, h1, eri).toarray()

    Href, strs_a, strs_b = _pyscf_matrix(h1, eri, norb, na, nb)

    perm = []
    for (a, b) in dets:
        ia = cistring.str2addr(norb, na, a)
        ib = cistring.str2addr(norb, nb, b)
        perm.append(ia * len(strs_b) + ib)
    Hmine_sorted = Hmine[np.ix_(perm, perm)]

    d = np.abs(Hmine_sorted - Href)
    assert d.max() < 1e-9, f"Hamiltonian mismatch vs pyscf {d.max():.3e}"


def test_fci_vs_pyscf(h2o):
    """The full-space determinant CI against pyscf FCI.

    No RAS restriction, so the determinant space is complete for the fixed
    alpha/beta counts and the lowest eigenvalue must equal pyscf's FCI
    energy on the same orbitals.
    """
    mol, mf = h2o
    nocc = mol.nelectron // 2
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(
        norb, norb, norb, norb)

    dets = enumerate_determinants(norb, nocc, nocc, 0, 0, 0, 0, 0)
    assert len(dets) == 441
    H = build_hamiltonian(dets, norb, h1, eri)
    w, _ = solve_ci(H, nroots=1)
    e_mine = w[0] + mol.energy_nuc()

    e_pyscf = fci.FCI(mol, mf.mo_coeff).kernel()[0]
    # The development report quotes 1e-6 Eh; this keeps the tighter 1e-9 Eh
    # of the development test.  The measured deviation is 1e-12 Eh, which is
    # the convergence threshold of pyscf's own FCI.
    assert abs(e_mine - e_pyscf) < 1e-9, \
        f"FCI mismatch {e_mine - e_pyscf:.3e}"


def test_casscf_limit(h2o):
    """RAS(2,2) with both excitation maxima zero is CASSCF(2,2).

    Malmqvist 2008, Sec. II B: with RAS1 fully occupied and RAS3 empty,
    i.e. both maxima zero, one identically obtains the CASSCF wavefunction
    for the RAS2 space.
    """
    mol, mf = h2o
    ras = RASSCF(mol, mf.mo_coeff, n_inact=4, n1=0, n2=2, n3=0,
                 max_holes=0, max_elec=0)
    e_ras, _ = ras.kernel(verbose=False)

    cas = mcscf.CASSCF(mf, 2, 2)
    cas.verbose = 0
    e_cas = cas.kernel()[0]

    assert abs(e_ras - e_cas) < 1e-8, f"CAS limit failed {e_ras - e_cas:.3e}"


def test_gradient_fd(h2o):
    """Analytic orbital gradient against central differences.

    The energy uses C = C0 exp(kappa), so the gradient must be the exact
    derivative of the exponential map.  A gradient derived from the
    linearized C(1+kappa) would deviate here by O(||kappa||) and would let
    the optimizer stop at a non-stationary point.
    """
    mol, mf = h2o
    # 4 inactive + RAS2(2) + 1 external = CAS(2,2) in 7 orbitals
    ras = RASSCF(mol, mf.mo_coeff, n_inact=4, n1=0, n2=2, n3=0,
                 max_holes=0, max_elec=0)
    rng = np.random.default_rng(3)
    x = 1e-2 * rng.normal(size=len(ras.pairs))
    _, g = ras._energy_grad(x, need_grad=True)

    h = 1e-5
    gfd = np.zeros_like(g)
    for k in range(len(x)):
        xp = x.copy()
        xp[k] += h
        xm = x.copy()
        xm[k] -= h
        gfd[k] = (ras.energy(xp) - ras.energy(xm)) / (2 * h)
    err = np.max(np.abs(g - gfd))
    rel = err / max(1e-8, np.max(np.abs(gfd)))
    assert rel < 1e-5, f"gradient mismatch {err:.3e} (rel {rel:.3e})"


@pytest.mark.slow
def test_ras_vs_cas_nonzero_holes(h2o):
    """A genuine RAS constraint against the same code without constraints.

    Partition n_inact = 2, RAS1 = 1 (1b2), RAS2 = 2 (3a1, 1b1), RAS3 = 2
    (4a1, 2b2) with holes <= 1 and electrons <= 2.  It admits the
    1b1^2 -> 4a1^2 and 1b1^2 -> 2b2^2 double excitations while excluding
    two holes in RAS1 and more than two electrons in RAS3, giving 51
    determinants against 100 for the unconstrained CAS(6e,5o) in the same
    five active orbitals.  The constrained energy must lie above the
    unconstrained one by 1.542 mEh.
    """
    mol, mf = h2o
    ras = RASSCF(mol, mf.mo_coeff, n_inact=2, n1=1, n2=2, n3=2,
                 max_holes=1, max_elec=2)
    assert len(ras.dets) == 51
    e_ras, _ = ras.kernel(verbose=False)
    if not ras.converged:
        e_ras, _ = ras.kernel(x0=ras.x, verbose=False)

    cas = RASSCF(mol, mf.mo_coeff, n_inact=2, n1=1, n2=2, n3=2,
                 max_holes=2, max_elec=4)
    assert len(cas.dets) == 100
    e_cas, _ = cas.kernel(verbose=False)
    if not cas.converged:
        e_cas, _ = cas.kernel(x0=cas.x, verbose=False)

    assert abs(e_ras - E_RAS_CONSTRAINED) < 1e-6, \
        f"constrained RAS energy {e_ras:.12f}"
    assert abs((e_ras - e_cas) - DELTA_RAS_CAS) < 1e-3, \
        f"RAS above CAS by {1e3 * (e_ras - e_cas):.3f} mEh"
