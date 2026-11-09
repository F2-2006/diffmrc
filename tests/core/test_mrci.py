"""Contracts of the determinant-basis MRCI kernel against pyscf references.

The kernel builds the MRCISD determinant space (internal, singly external,
doubly external), assembles the dense Slater-Condon Hamiltonian, solves
CASCI in the internal space, diagonalizes the uncontracted space, and
builds the internally contracted doubly-external basis of Werner &
Knowles (J. Chem. Phys. 1988, 89, 5803, Eqs. 9-19).  The Davidson
correction follows the Langhoff-Davidson form generalized by reference
weight.

The reference values are frozen from the validated prototype (2026-11-05,
pyscf 2.14.0) and are checked, not merely printed:

* H2O/6-31G CAS(2,2): the two-electron active space makes the internal
  N-2 space one-dimensional, so the contracted energy must equal the
  uncontracted one, and both must equal pyscf FCI in the same CI space.
* CH2/STO-3G CAS(4,4): one external orbital, so uncontracted MRCISD is
  the full CI of that space.
* CH2/6-31G CAS(4,4): the contraction actually truncates.  Frozen values:
  uncontracted MRCISD lies 9.147e-4 Eh above pyscf FCI and the contracted
  energy lies 1.028e-4 Eh above the uncontracted one.  CH2 has a triplet
  ground state, so the pyscf reference requests several roots and keeps
  the lowest; a single-root Davidson started from the closed-shell guess
  converges to the first singlet and would show a spurious 14 mH error.
* H2...H2 / 6-31G and He...He / cc-pVDZ at 50 A: truncated SD-CI is not
  size consistent.  Frozen deviations Delta = E(dimer) - 2 E(monomer):
  +2.59e-4 and +1.07e-4 Eh at the uncontracted MRCISD level, -5.28e-5
  and +2.02e-6 Eh with the Davidson correction.  The tolerance is 5e-5 Eh.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci  # noqa: E402

from diffmrc.core.mrci import run_mrci  # noqa: E402
from diffmrc.core.mrci.pyscf_iface import build_reference  # noqa: E402


H2O_GEOMETRY = """
O 0.000000  0.000000  0.117300
H 0.000000  0.757000 -0.469200
H 0.000000 -0.757000 -0.469200
"""

# Singlet CH2, C-H 1.11 A, H-C-H 102.4 deg.
CH2_GEOMETRY = """
C 0.000000  0.000000  0.000000
H 0.865000  0.696000  0.000000
H -0.865000 0.696000  0.000000
"""


def pyscf_fci_lowest(ref, nroots=4):
    """Lowest pyscf FCI eigenvalue in the same CI space.

    CH2 has a triplet ground state.  A single-root Davidson started from
    the closed-shell guess converges to the first singlet instead of the
    triplet ground state, so several roots are requested and the lowest
    is kept.
    """
    neleca = ref.nact_elec // 2
    nelecb = ref.nact_elec - neleca
    e = fci.direct_spin1.kernel(
        ref.h1, ref.eri, ref.norb_ci, (neleca, nelecb), nroots=nroots
    )
    return float(np.min(np.atleast_1d(e[0]))) + ref.e_core


@pytest.fixture(scope="module")
def h2o_cas22():
    """H2O/6-31G CAS(2,2) and the pyscf FCI in the same 2e/9o CI space."""
    ref = build_reference(H2O_GEOMETRY, "6-31G", nact=2, nact_elec=2)
    res = run_mrci(ref, roots=3, verbose=False)
    ref_fci = build_reference(
        H2O_GEOMETRY, "6-31G", nact=ref.norb_ci, nact_elec=2
    )
    return ref, res, ref_fci


@pytest.fixture(scope="module")
def ch2_sto3g():
    ref = build_reference(CH2_GEOMETRY, "STO-3G", nact=4, nact_elec=4)
    res = run_mrci(ref, roots=4, verbose=False)
    return ref, res, pyscf_fci_lowest(ref)


@pytest.fixture(scope="module")
def ch2_631g():
    ref = build_reference(CH2_GEOMETRY, "6-31G", nact=4, nact_elec=4)
    res = run_mrci(ref, roots=4, verbose=False)
    return ref, res, pyscf_fci_lowest(ref)


def test_h2o_casci_matches_pyscf(h2o_cas22):
    ref, res, _ = h2o_cas22
    assert abs(res["e_casci"] - ref.e_casci) <= 1e-8


def test_h2o_contraction_is_exact_for_two_electrons(h2o_cas22):
    _, res, ref_fci = h2o_cas22
    assert abs(res["e_uncontracted"] - ref_fci.e_casci) <= 1e-12
    assert abs(res["e_ic"] - res["e_uncontracted"]) <= 1e-12
    assert res["n_doubles_basis"] < res["n_doubles_uncontracted"]


def test_ch2_sto3g_is_full_ci(ch2_sto3g):
    ref, res, e_fci = ch2_sto3g
    assert abs(res["e_casci"] - ref.e_casci) <= 1e-8
    assert abs(res["e_uncontracted"] - e_fci) <= 1e-12
    assert abs(res["e_ic"] - res["e_uncontracted"]) <= 1e-12
    assert abs(res["s2_uncontracted"] - 2.0) <= 1e-6
    assert abs(res["s2_ic"] - 2.0) <= 1e-6


def test_ch2_631g_matches_frozen_contraction_values(ch2_631g):
    ref, res, e_fci = ch2_631g
    assert abs(res["e_casci"] - ref.e_casci) <= 1e-8

    # Frozen prototype values, tolerance 1e-5 Eh.
    assert abs((res["e_uncontracted"] - e_fci) - 9.147e-4) <= 1e-5
    assert abs((res["e_ic"] - res["e_uncontracted"]) - 1.028e-4) <= 1e-5

    # The contracted space is a subspace of the uncontracted one.
    assert res["e_ic"] >= res["e_uncontracted"] - 1e-9
    assert res["e_uncontracted"] >= e_fci - 1e-9

    # Both roots are the triplet ground state.
    assert abs(res["s2_uncontracted"] - 2.0) <= 1e-6
    assert abs(res["s2_ic"] - 2.0) <= 1e-6


R_SEP = 50.0  # fragment separation in Angstrom

H2_MONOMER = """
H 0.000000  0.000000 -0.370000
H 0.000000  0.000000  0.370000
"""

HE_MONOMER = """
He 0.000000  0.000000  0.000000
"""


def h2_dimer(r):
    z = r / 2.0
    return f"""
H 0.000000  0.000000 {-(z + 0.37):.6f}
H 0.000000  0.000000 {-(z - 0.37):.6f}
H 0.000000  0.000000  {(z - 0.37):.6f}
H 0.000000  0.000000  {(z + 0.37):.6f}
"""


def he_dimer(r):
    z = r / 2.0
    return f"""
He 0.000000  0.000000 {-z:.6f}
He 0.000000  0.000000  {z:.6f}
"""


def _size_consistency(name, atom_mono, atom_dimer, basis, nact_frag, nelec_frag):
    """Delta = E(dimer) - 2 E(monomer) for every MRCI level."""
    mono = build_reference(atom_mono, basis, nact=nact_frag,
                           nact_elec=nelec_frag)
    res_m = run_mrci(mono, roots=2, verbose=False)

    dimer = build_reference(atom_dimer, basis, nact=2 * nact_frag,
                            nact_elec=2 * nelec_frag)
    res_d = run_mrci(dimer, roots=2, verbose=False)

    return {key: res_d[key] - 2.0 * res_m[key]
            for key in ("e_casci", "e_uncontracted", "e_ic",
                        "e_uncontracted_q", "e_ic_q")}


@pytest.mark.slow
def test_size_consistency_deviations():
    """Frozen size-consistency deviations of truncated SD-CI and MRCI+Q.

    Each fragment uses CAS(2,2) in the monomer and CAS(4,4) in the dimer;
    the monomer MRCISD is exact, so the dimer deviation is the truncated
    CI error.  CASCI is size consistent by construction.
    """
    h2 = _size_consistency("H2...H2", H2_MONOMER, h2_dimer(R_SEP),
                           "6-31G", 2, 2)
    he = _size_consistency("He...He", HE_MONOMER, he_dimer(R_SEP),
                           "cc-pVDZ", 2, 2)

    # CASCI consistency is limited only by the SCF convergence threshold.
    assert abs(h2["e_casci"]) <= 1e-7
    assert abs(he["e_casci"]) <= 1e-7

    # MRCISD and MRCI+Q deviations, frozen values with a 5e-5 Eh tolerance.
    assert abs(h2["e_uncontracted"] - 2.592e-4) <= 5e-5
    assert abs(h2["e_uncontracted_q"] - (-5.284e-5)) <= 5e-5
    assert abs(he["e_uncontracted"] - 1.072e-4) <= 5e-5
    assert abs(he["e_uncontracted_q"] - 2.020e-6) <= 5e-5
