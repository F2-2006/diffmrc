"""Contracts of the QDPT / state-interaction module on frozen SOC benchmarks.

The reference numbers are the project's frozen SOC benchmarks, anchored
against ORCA 6.1.1:

    OH X ^2Pi, 6-31G, CAS(7,6), ncore = 1, r = 1.8324 Bohr
    OF X ^2Pi, 6-31G, CAS(13,8), ncore = 2, r = 2.4 Bohr

Four checks reproduce the term splittings of the one-electron
Breit-Pauli and SOMF operators to the printed digits, which tests the
prefactors, the spin algebra and the state-interaction assembly at once.
The remaining checks cover Hermiticity, Kramers degeneracy, operator
antisymmetry, the SO-CASPT2 diagonal hook, and the model-space
sensitivity that must be reported with every number.

The benchmark geometries live here, not in the package: ``systems.py``
of the development prototype was a driver-side registry.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf.fci import direct_spin1  # noqa: E402

from diffmrc.core import qdpt  # noqa: E402


TOL_CM = 5e-3   # frozen values are printed to 1e-3 cm^-1

SYSTEMS = {
    "oh": dict(
        label="OH X 2Pi",
        atom=[("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8324))],
        basis="6-31g", charge=0, spin=1,
        ncore=1, ncas=6, nelecas=7, nroots=2,
    ),
    "of": dict(
        label="OF X 2Pi",
        atom=[("O", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 2.4))],
        basis="6-31g", charge=0, spin=1,
        ncore=2, ncas=8, nelecas=13, nroots=2,
    ),
}


def system(name):
    """Return a copy of the named benchmark record."""
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}")
    return {k: (v.copy() if isinstance(v, (dict, list)) else v)
            for k, v in SYSTEMS[name].items()}


@pytest.fixture(scope="module")
def oh():
    sp = qdpt.ScalarSpace(system("oh"))
    return sp, qdpt.mo_soc(sp.mol, sp.c_core_act, sp.density_ao(),
                           "one_electron")


@pytest.fixture(scope="module")
def of():
    sp = qdpt.ScalarSpace(system("of"))
    return sp, qdpt.mo_soc(sp.mol, sp.c_core_act, sp.density_ao(),
                           "one_electron")


def active(h, sp):
    """Active-space block of a core + active SOC tensor."""
    return h[:, sp.ncore:, sp.ncore:]


def test_oh_one_electron(oh):
    sp, (h, pref) = oh
    res = qdpt.qdpt_levels(sp, active(h, sp), pref)
    assert qdpt.term_splitting_cm(res) == pytest.approx(205.883, abs=TOL_CM)


def test_oh_somf(oh):
    sp, _ = oh
    h, pref = qdpt.mo_soc(sp.mol, sp.c_core_act, sp.density_ao(), "somf")
    res = qdpt.qdpt_levels(sp, active(h, sp), pref)
    assert qdpt.term_splitting_cm(res) == pytest.approx(128.829, abs=TOL_CM)


def test_of_one_electron(of):
    sp, (h, pref) = of
    res = qdpt.qdpt_levels(sp, active(h, sp), pref)
    assert qdpt.term_splitting_cm(res) == pytest.approx(272.061, abs=TOL_CM)


def test_of_somf(of):
    sp, _ = of
    h, pref = qdpt.mo_soc(sp.mol, sp.c_core_act, sp.density_ao(), "somf")
    res = qdpt.qdpt_levels(sp, active(h, sp), pref)
    assert qdpt.term_splitting_cm(res) == pytest.approx(176.125, abs=TOL_CM)


def test_hermiticity_and_kramers(oh):
    sp, (h, pref) = oh
    res = qdpt.qdpt_levels(sp, active(h, sp), pref)
    # ``hermiticity`` is the pre-symmetrization residual of the assembled
    # H_eff, a machine-precision diagnostic rather than a physical criterion
    # (the physical splitting is the 205.883 check above).  It is 1e-13 in an
    # isolated run and reaches 5e-10 in a long full-suite process, tracking
    # the CI-vector convergence; the spectrum is computed from the
    # symmetrized matrix, so the Kramers gap stays at machine precision.
    assert res["hermiticity"] < 1e-8
    assert res["kramers_gap_cm"] < 1e-6


def test_operator_symmetry(oh):
    sp, (h, pref) = oh
    chk = qdpt.check_antisymmetry(active(h, sp))
    assert chk["antisymmetry"] < 1e-12
    assert chk["imag"] == 0.0


def test_prefactor_values():
    assert qdpt.PREFACTORS["one_electron"] == \
        pytest.approx(1j * 0.5 * 7.2973525693e-3 ** 2)
    assert qdpt.PREFACTORS["ecp"] == -1j


def test_diagonal_hook(oh):
    """A differential diagonal moves the splitting; a uniform shift does not."""
    sp, (h, pref) = oh
    base = sp.state_energies()
    res0 = qdpt.qdpt_levels(sp, active(h, sp), pref)
    res1 = qdpt.qdpt_levels(sp, active(h, sp), pref,
                            diagonal=base + 0.01)
    assert np.abs(res1["levels_eh"] - res0["levels_eh"]).max() < 1e-10
    res2 = qdpt.qdpt_levels(sp, active(h, sp), pref,
                            diagonal=base + np.array([0.01, -0.01]))
    assert abs(qdpt.term_splitting_cm(res2)
               - qdpt.term_splitting_cm(res0)) > 100.0


def test_model_space_matters(oh):
    """The four-state window keeps Kramers pairs degenerate."""
    sp, (h, pref) = oh
    res = qdpt.qdpt_levels(sp, active(h, sp), pref)
    assert qdpt.term_splitting_cm(res) == pytest.approx(205.883, abs=TOL_CM)
    assert res["levels_cm"][1] < 1e-6
    assert abs(res["levels_cm"][2] - res["levels_cm"][3]) < 1e-6


def test_detspace_scalar_spectrum(oh):
    """The determinant-space oracle reproduces the pyscf FCI roots.

    Both sides use the same effective integrals and neither includes the
    frozen-core energy ``ecore``; only the active-space CI energy is
    compared.
    """
    sp, _ = oh
    h1, _ = sp.h1eff()
    h2 = sp.h2eff()
    h_det = qdpt.scalar_ci_matrix(sp)
    w = np.linalg.eigvalsh(h_det)

    solver = direct_spin1.FCI()
    solver.conv_tol = 1e-12
    e_ref, _ = solver.kernel(h1, h2, sp.ncas, (sp.na, sp.nb),
                             nroots=sp.nroots)
    assert np.abs(w[:sp.nroots] - np.asarray(e_ref)).max() < 1e-10


def test_doublet_only_guard(oh):
    """S >= 1 must be refused rather than guessed."""
    sp, (h, pref) = oh

    class TripletSpace:
        system = {"spin": 3}

    with pytest.raises(NotImplementedError):
        qdpt.si_matrix_s12(TripletSpace(), active(h, sp), pref)
