"""The implicit-differentiation ROHF reference: interface and checks.

The lanthanide cases behind the module docstring are three-to-six
minute server calculations.  They are kept here behind
``DIFFMRC_SCFAD_LANTHANIDE=1`` so the default suite stays small: the
same code path is exercised on OH and H2, and the slow tests pin the
YbF analytic/finite-difference agreement and the CeO branch
repeatability that the merge is accepted on.
"""
import os

import numpy as np
import pytest

pytest.importorskip("pyscfad")
pytest.importorskip("jax")

from diffmrc.driver import scf_ad  # noqa: E402

LANTHANIDE = os.environ.get("DIFFMRC_SCFAD_LANTHANIDE") == "1"
_YBF = [("Yb", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 3.6))]
_CEO = [("Ce", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 3.4))]


def _oh(cart=False, atoms=None):
    from pyscfad import gto
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0 0 1.83" if atoms is None else atoms
    mol.basis = "sto-3g"
    mol.unit = "Bohr"
    mol.spin = 1
    mol.cart = cart
    mol.verbose = 0
    mol.build()
    return mol


def _h2():
    from pyscfad import gto
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "6-31g"
    mol.spin = 0
    mol.verbose = 0
    mol.build()
    return mol


def _shifted(atoms, iatom, axis, step):
    out = []
    for i, (symbol, xyz) in enumerate(atoms):
        x = np.array(xyz, dtype=float)
        if i == iatom:
            x[axis] += step
        out.append((symbol, tuple(x)))
    return out


def _lanthanide(atoms, basis, ecp, spin):
    from pyscfad import gto
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.ecp = ecp
    mol.spin = spin
    mol.unit = "Bohr"
    mol.verbose = 0
    mol.build()
    return mol


def _ybf(atoms=None):
    return _lanthanide(
        _YBF if atoms is None else atoms,
        {"Yb": "stuttgart_rsc", "F": "6-31g"},
        {"Yb": "ecpds28mwbso"}, 1)


def _ceo():
    return _lanthanide(
        _CEO, {"Ce": "stuttgart_rsc", "O": "6-31g"},
        {"Ce": "ecpds28mwbso"}, 2)


def test_the_names_are_exported():
    from diffmrc.driver.scf_protocol import GUESSES

    assert scf_ad.GUESS_ORDER is GUESSES
    assert len(scf_ad.GUESS_ORDER) == 5
    assert scf_ad.GUESS_ORDER[0] is None
    assert scf_ad.RELAX_GRID
    assert all(len(pair) == 2 for pair in scf_ad.RELAX_GRID)
    for name in ("SCFReference", "implicit_diff", "rohf_energy",
                 "rohf_gradient", "scf_reference"):
        assert name in scf_ad.__all__
        assert callable(getattr(scf_ad, name))


def test_implicit_diff_restores_the_previous_flag():
    from pyscfad import config

    previous = config.scf_implicit_diff
    with scf_ad.implicit_diff(True):
        assert config.scf_implicit_diff is True
    assert config.scf_implicit_diff is previous


def test_the_coordinate_view_has_one_leaf_and_keeps_settings():
    import jax

    mol = _oh()
    view = scf_ad._coordinate_view(mol)
    leaves = jax.tree_util.tree_leaves(view)
    assert len(leaves) == 1
    assert np.shape(leaves[0]) == (2, 3)

    assert scf_ad._coordinate_view(_oh(cart=True)).cart is True

    # A coordinate-only view is returned unchanged, so repeated calls do
    # not rebuild the molecule.
    assert scf_ad._coordinate_view(view) is view


def test_a_plain_pyscf_molecule_is_rejected():
    from pyscf import gto
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1,
                verbose=0)
    with pytest.raises(TypeError):
        scf_ad._coordinate_view(mol)


def test_oh_reference_energy_and_gradient():
    mol = _oh()
    ref = scf_ad.scf_reference(mol)
    assert ref.e_tot < 0.0
    assert ref.spread == pytest.approx(0.0, abs=1e-9)

    e = float(np.asarray(scf_ad.rohf_energy(mol, dm0=ref.dm)))
    assert e == pytest.approx(ref.e_tot, abs=1e-9)

    g = scf_ad.rohf_gradient(mol, dm0=ref.dm)
    g2 = scf_ad.rohf_gradient(mol, dm0=ref.dm)
    assert np.array_equal(g, g2)

    # Central difference of the same single-start protocol.
    step = 1e-3
    base = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))]
    energies = []
    for sign in (+1, -1):
        moved = _oh(atoms=_shifted(base, 0, 2, sign * step))
        energies.append(
            scf_ad.scf_reference(moved, guesses=(), dm0_guess=ref.dm).e_tot)
    fd = (energies[0] - energies[1]) / (2 * step)
    assert abs(g[0, 2] - fd) / abs(fd) < 1e-5


def test_a_closed_shell_reference_converges():
    mol = _h2()
    ref = scf_ad.scf_reference(mol)
    assert ref.e_tot == pytest.approx(-1.1267553, abs=1e-6)
    assert ref.spread == pytest.approx(0.0, abs=1e-9)
    g = scf_ad.rohf_gradient(mol, dm0=ref.dm)
    assert np.all(np.isfinite(g))


def test_no_converged_candidate_is_refused():
    mol = _oh()
    with pytest.raises(RuntimeError, match="no ROHF candidate converged"):
        scf_ad.scf_reference(mol, conv_tol=1e-12, max_cycle=1)


def test_the_energy_refuses_an_unconverged_fixed_point():
    mol = _oh()
    with pytest.raises(RuntimeError, match="did not converge"):
        scf_ad.rohf_energy(mol, conv_tol=1e-12, max_cycle=1)


@pytest.mark.skipif(not LANTHANIDE,
                    reason="set DIFFMRC_SCFAD_LANTHANIDE=1 to run")
def test_ybf_implicit_gradient_matches_analytic_and_differs_from_unrolled():
    import jax
    from pyscf import gto as pyscf_gto
    from pyscf import scf as pyscf_scf
    from pyscfad import scf

    mol = _ybf()
    ref = scf_ad.scf_reference(mol)

    plain = pyscf_gto.M(atom=_YBF, basis={"Yb": "stuttgart_rsc",
                                          "F": "6-31g"},
                        ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr",
                        verbose=0)
    pmf = pyscf_scf.ROHF(plain)
    pmf.conv_tol = 1e-10
    pmf.max_cycle = 200
    pmf.verbose = 0
    pmf.kernel()
    assert pmf.converged
    assert pmf.e_tot == pytest.approx(ref.e_tot, abs=1e-6)
    analytic = np.asarray(pmf.nuc_grad_method().kernel())

    g = scf_ad.rohf_gradient(mol, dm0=ref.dm)
    relative = np.linalg.norm(g - analytic) / np.linalg.norm(analytic)
    assert relative <= 1e-6

    # The defect this module exists to remove: the unrolled path, on the
    # same coordinate-only pytree, returns 3.42 times the analytic
    # magnitude in this setup.
    def unrolled_energy(m):
        mf = scf.ROHF(m)
        mf.conv_tol = 1e-10
        mf.max_cycle = 200
        mf.verbose = 0
        mf.kernel()
        return mf.e_tot

    with scf_ad.implicit_diff(False):
        unrolled = np.asarray(
            jax.grad(unrolled_energy)(scf_ad._coordinate_view(mol)).coords)
    relation = np.linalg.norm(unrolled - analytic) / np.linalg.norm(analytic)
    assert relation > 1.0
    assert abs(unrolled[0, 2]) / abs(analytic[0, 2]) == pytest.approx(
        3.42, abs=0.01)

    step = 1e-3
    finite = np.zeros_like(g)
    for iatom in (0, 1):
        values = []
        for sign in (+1, -1):
            moved = _shifted(_YBF, iatom, 2, sign * step)
            displaced = scf_ad.scf_reference(_ybf(moved),
                                             dm0_guess=ref.dm)
            # A branch jump would show up as a tenth-of-a-Hartree step.
            assert displaced.e_tot == pytest.approx(ref.e_tot, abs=0.01)
            values.append(displaced.e_tot)
        finite[iatom, 2] = (values[0] - values[1]) / (2 * step)
    fd_relative = np.linalg.norm(g - finite) / np.linalg.norm(finite)
    assert fd_relative <= 1e-5


@pytest.mark.skipif(not LANTHANIDE,
                    reason="set DIFFMRC_SCFAD_LANTHANIDE=1 to run")
def test_ceo_reference_and_gradient_are_repeatable():
    mol = _ceo()
    first = scf_ad.scf_reference(mol)
    second = scf_ad.scf_reference(mol)
    # CeO has two solutions 0.112 Eh apart with identical occupations;
    # the spread is the evidence that the ladder saw both.
    assert first.spread > 0.1
    assert first.e_tot == second.e_tot
    assert first.guess == second.guess

    g1 = scf_ad.rohf_gradient(mol, dm0=first.dm)
    g2 = scf_ad.rohf_gradient(mol, dm0=second.dm)
    assert np.array_equal(g1, g2)
    assert g1[0, 2] == pytest.approx(2.24555783e-2, rel=1e-6)
    assert g1[1, 2] == pytest.approx(-2.24555783e-2, rel=1e-6)
