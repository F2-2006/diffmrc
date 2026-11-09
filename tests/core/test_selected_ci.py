"""Contracts of the selected-CI kernel (HCI/ASCI) and its NEVPT2 bridge.

The kernel supplies a determinant-space Hamiltonian, the heat-bath CI
selector of Holmes, Tubman & Umrigar (J. Chem. Theory Comput. 12, 3674
(2016)) and the adaptive sampling CI selector of Levine et al.
(J. Chem. Theory Comput. 16, 2340 (2020)).  The NEVPT2 bridge maps a
selected determinant list onto the pyscf alpha-major CI matrix and calls
``pyscf.mrpt.nevpt2.NEVPT``.

The reference numbers are frozen from the validated prototype (2026-11-09,
pyscf 2.14.0) and are checked, not merely printed:

* H2O/cc-pVDZ CAS(8,6), 225 determinants: the HCI eps1 scan key points
  (eps1 = 1e-2, 1e-3, 3e-4, 1e-4, 3e-5) select 20, 46, 59, 63, 65
  determinants with variational deviations 4.113e-4, 4.137e-6, 6.339e-8,
  3.675e-10 and machine zero; the full space returns CASCI.
* The Hamiltonian reproduces pyscf ``fci.direct_spin1.pspace``
  elementwise; the cache is exact.
* The NEVPT2 bridge reproduces a direct ``NEVPT(mc).kernel()`` call to
  1e-12 Eh on the full-space wavefunction.
* HCI returns determinants and coefficients in matching order, which is
  not the enumeration order; elementwise comparison without alignment
  through ``dets`` is invalid.
* N2/6-31G CAS(6,6) reproduces the 1% stopping-rule defect: at
  eps1 = 1e-4 the default ``new_fraction=0.01`` stops after 103
  determinants and misses a determinant with coefficient 1.725e-2,
  leaving a 6.330e-4 Eh deviation; ``new_fraction=0`` reaches 104
  determinants and the exact energy.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, mcscf, scf  # noqa: E402
from pyscf.fci import cistring, direct_spin1  # noqa: E402
from pyscf.mrpt.nevpt2 import NEVPT  # noqa: E402

from diffmrc.core.selected_ci import (  # noqa: E402
    ASCI,
    ActiveSpaceHamiltonian,
    HCI,
    enumerate_determinants,
)
from diffmrc.core.selected_ci.nevpt2_interface import (  # noqa: E402
    embed_ci_vector,
    selected_ci_nevpt2,
)


H2O_GEOMETRY = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
N2_GEOMETRY = "N 0 0 0; N 0 0 1.4"


def build_casci(atom, basis, ncas, nelecas, canonicalize=False):
    """CASCI reference, active-space Hamiltonian and determinant list."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.run()
    if canonicalize:
        mo_coeff, ci, mo_energy = mc.canonicalize(ci=mc.ci, cas_natorb=True)
        mc.mo_coeff = mo_coeff
        mc.ci = ci
        mc.mo_energy = mo_energy
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, np.asarray(mc.get_h2eff()), ncas)
    ham = ActiveSpaceHamiltonian(h1e, h2e, ncas)
    dets = enumerate_determinants(ncas, mc.nelecas[0], mc.nelecas[1])
    ham.enable_cache(dets)
    return dict(
        mol=mol, mc=mc, h1e=h1e, h2e=h2e, ncas=ncas, ecore=ecore,
        ham=ham, dets=dets, nelecas=mc.nelecas,
        reference=mc.e_tot - ecore,
    )


@pytest.fixture(scope="module")
def h2o_cas86():
    """H2O/cc-pVDZ CAS(8,6): 225 determinants."""
    return build_casci(H2O_GEOMETRY, "cc-pvdz", 6, 8)


@pytest.fixture(scope="module")
def h2o_cas44():
    """H2O/6-31G CAS(4,4): 36 determinants, the prototype unit system."""
    return build_casci(H2O_GEOMETRY, "6-31g", 4, 4)


@pytest.fixture(scope="module")
def n2_cas66():
    """N2/6-31G CAS(6,6): 400 determinants."""
    return build_casci(N2_GEOMETRY, "6-31g", 6, 6)


@pytest.fixture(scope="module")
def h2o_cas86_canonical():
    """H2O/cc-pVDZ CAS(8,6) in canonical natural orbitals for NEVPT2."""
    return build_casci(H2O_GEOMETRY, "cc-pvdz", 6, 8, canonicalize=True)


def exact_pt2(ham, selected, coeffs, energy, all_dets):
    """Brute-force Epstein-Nesbet sum over every external determinant."""
    selected_set = set(selected)
    correction = 0.0
    for det in all_dets:
        if det in selected_set:
            continue
        numerator = 0.0
        for det_i, coefficient in zip(selected, coeffs):
            value = ham.matrix_element(det, det_i)
            if value != 0.0:
                numerator += value * coefficient
        denominator = energy - ham.diagonal(det)
        if abs(denominator) > 1e-12:
            correction += numerator * numerator / denominator
    return correction


# ---------------------------------------------------------------------------
# Hamiltonian against pyscf
# ---------------------------------------------------------------------------
def test_hamiltonian_matches_pyscf_pspace(h2o_cas44, n2_cas66):
    """Dense matrix, cached matrix and CASCI energy against pyscf."""
    for system in (h2o_cas44, n2_cas66):
        h1e, h2e = system["h1e"], system["h2e"]
        ncas, nelecas = system["ncas"], system["nelecas"]
        dets = system["dets"]

        na = cistring.num_strings(ncas, nelecas[0])
        nb = cistring.num_strings(ncas, nelecas[1])
        _, reference = direct_spin1.pspace(
            h1e, h2e, ncas, nelecas, np=na * nb
        )
        index = [cistring.str2addr(ncas, nelecas[0], det[0]) * nb
                 + cistring.str2addr(ncas, nelecas[1], det[1])
                 for det in dets]
        reordered = reference[np.ix_(index, index)]

        cached = system["ham"].dense_matrix(dets)
        assert np.max(np.abs(cached - reordered)) <= 1e-10

        uncached = ActiveSpaceHamiltonian(h1e, h2e, ncas)
        ours = uncached.dense_matrix(dets)
        assert np.max(np.abs(ours - cached)) <= 1e-12

        energy = float(np.linalg.eigvalsh(ours)[0])
        assert abs(energy - system["reference"]) <= 1e-10


# ---------------------------------------------------------------------------
# HCI eps1 scan (frozen prototype points)
# ---------------------------------------------------------------------------
HCI_SCAN = [
    (1e-2, 20, 4.113e-4, 1e-7),
    (1e-3, 46, 4.137e-6, 1e-9),
    (3e-4, 59, 6.339e-8, 1e-10),
    (1e-4, 63, 3.675e-10, 1e-12),
    # At 3e-5 the selected space reproduces CASCI to machine precision.
    (3e-5, 65, None, 1e-10),
]


@pytest.mark.parametrize("eps1, expected_ndet, expected_dev, tol", HCI_SCAN)
def test_hci_eps_scan(h2o_cas86, eps1, expected_ndet, expected_dev, tol):
    """Frozen HCI selection counts and variational deviations."""
    system = h2o_cas86
    hci = HCI(system["ham"], system["nelecas"][0], system["nelecas"][1],
              eps1=eps1).run()
    assert len(hci.dets) == expected_ndet
    deviation = hci.energy - system["reference"]
    if expected_dev is None:
        assert abs(deviation) <= tol
    else:
        assert abs(deviation - expected_dev) <= tol


def test_hci_full_space_matches_casci(h2o_cas86):
    """eps1 = 0 with the strict stop rule covers the full space."""
    system = h2o_cas86
    hci = HCI(system["ham"], system["nelecas"][0], system["nelecas"][1],
              eps1=0.0, new_fraction=0.0).run()
    assert len(hci.dets) == len(system["dets"])
    assert set(hci.dets) == set(system["dets"])
    assert abs(hci.energy - system["reference"]) <= 1e-12


def test_hci_and_asci_pt2_match_bruteforce(h2o_cas44):
    """PT2 sums on full and truncated spaces match explicit enumeration."""
    system = h2o_cas44
    ham, all_dets = system["ham"], system["dets"]
    neleca, nelecb = system["nelecas"]

    full = HCI(ham, neleca, nelecb, eps1=0.0, new_fraction=0.0).run()
    assert len(full.dets) == len(all_dets)
    assert abs(full.energy - system["reference"]) <= 1e-10
    correction, _ = full.pt2(eps2=0.0)
    assert abs(correction - exact_pt2(
        ham, full.dets, full.coeffs, full.energy, all_dets)) <= 1e-10

    truncated = HCI(ham, neleca, nelecb, eps1=0.1).run()
    correction, _ = truncated.pt2(eps2=0.0)
    assert abs(correction - exact_pt2(
        ham, truncated.dets, truncated.coeffs, truncated.energy,
        all_dets)) <= 1e-10

    asci_full = ASCI(ham, neleca, nelecb, cdets=len(all_dets),
                     tdets=len(all_dets)).run()
    assert len(asci_full.dets) == len(all_dets)
    assert abs(asci_full.energy - system["reference"]) <= 1e-10
    correction, _ = asci_full.pt2()
    assert abs(correction - exact_pt2(
        ham, asci_full.dets, asci_full.coeffs, asci_full.energy,
        all_dets)) <= 1e-10

    asci_truncated = ASCI(ham, neleca, nelecb, cdets=5, tdets=10).run()
    assert asci_truncated.energy >= system["reference"] - 1e-10


def test_asci_reaches_casci(h2o_cas86):
    """ASCI at 1/3 of the space and in the full-space limit."""
    system = h2o_cas86
    neleca, nelecb = system["nelecas"]
    ham, all_dets = system["ham"], system["dets"]

    third = ASCI(ham, neleca, nelecb, cdets=75, tdets=75).run()
    assert len(third.dets) == 75
    assert abs(third.energy - system["reference"]) <= 1e-12

    full = ASCI(ham, neleca, nelecb, cdets=len(all_dets),
                tdets=len(all_dets)).run()
    assert len(full.dets) == len(all_dets)
    assert abs(full.energy - system["reference"]) <= 1e-12


# ---------------------------------------------------------------------------
# Determinant ordering
# ---------------------------------------------------------------------------
def test_determinant_order_alignment(h2o_cas44):
    """Coefficients align with ``dets``, not with the enumeration order."""
    system = h2o_cas44
    mc, ham, dets = system["mc"], system["ham"], system["dets"]
    neleca, nelecb = system["nelecas"]

    hci = HCI(ham, neleca, nelecb, eps1=0.0, new_fraction=0.0).run()
    assert len(set(hci.dets)) == len(hci.dets)
    assert set(hci.dets) == set(dets)

    ncas = system["ncas"]
    by_det = {
        det: mc.ci[cistring.str2addr(ncas, neleca, det[0]),
                   cistring.str2addr(ncas, nelecb, det[1])]
        for det in dets
    }

    aligned = np.array([by_det[det] for det in hci.dets])
    overlap = abs(float(np.dot(hci.coeffs, aligned)))
    assert abs(overlap - 1.0) <= 1e-10

    if hci.dets != dets:
        misaligned = np.array([by_det[det] for det in dets])
        sign = np.sign(np.dot(hci.coeffs, misaligned))
        naive_diff = float(np.max(np.abs(sign * hci.coeffs - misaligned)))
        assert naive_diff > 1e-8


# ---------------------------------------------------------------------------
# NEVPT2 bridge
# ---------------------------------------------------------------------------
def test_nevpt2_bridge_matches_pyscf(h2o_cas86_canonical):
    """Full-space bridge versus a direct pyscf NEVPT2 call."""
    system = h2o_cas86_canonical
    mc, ham, dets = system["mc"], system["ham"], system["dets"]
    neleca, nelecb = system["nelecas"]
    ncas = system["ncas"]

    solver = NEVPT(mc)
    solver.canonicalized = True
    e_direct = float(solver.kernel())

    na = cistring.num_strings(ncas, neleca)
    nb = cistring.num_strings(ncas, nelecb)
    assert mc.ci.shape == (na, nb)

    coeffs = np.array([
        mc.ci[cistring.str2addr(ncas, neleca, det[0]),
              cistring.str2addr(ncas, nelecb, det[1])]
        for det in dets
    ])
    embedded = embed_ci_vector(dets, coeffs, ncas, neleca, nelecb)
    assert np.max(np.abs(embedded - mc.ci)) <= 1e-12

    full = HCI(ham, neleca, nelecb, eps1=0.0, new_fraction=0.0).run()
    aligned = np.array([
        mc.ci[cistring.str2addr(ncas, neleca, det[0]),
              cistring.str2addr(ncas, nelecb, det[1])]
        for det in full.dets
    ])
    assert abs(abs(float(np.dot(full.coeffs, aligned))) - 1.0) <= 1e-10

    result = selected_ci_nevpt2(mc, full.dets, full.coeffs, full.energy,
                                canonicalized=True)
    assert abs(result["e_corr"] - e_direct) <= 1e-12


# ---------------------------------------------------------------------------
# N2 stopping-rule regression
# ---------------------------------------------------------------------------
def test_n2_stopping_rule_defect(n2_cas66):
    """The 1% rule stops one determinant early; strict mode is exact.

    At eps1 = 1e-4 the second iteration holds 103 determinants and adds
    one; 1% of 103 is 1.03, so the single addition triggers the stop.
    The missing determinant has coefficient 1.725e-2 in the FCI vector,
    which costs 6.330e-4 Eh.  The defect is a resolution limit of the
    published rule, not an implementation deviation.
    """
    system = n2_cas66
    ham, neleca, nelecb = system["ham"], system["nelecas"][0], system["nelecas"][1]

    default = HCI(ham, neleca, nelecb, eps1=1e-4).run()
    assert len(default.dets) == 103
    deviation = default.energy - system["reference"]
    assert abs(deviation - 6.330e-4) <= 1e-6
    assert deviation > 5e-4

    strict = HCI(ham, neleca, nelecb, eps1=1e-4, new_fraction=0.0).run()
    assert len(strict.dets) == 104
    assert abs(strict.energy - system["reference"]) <= 1e-10
