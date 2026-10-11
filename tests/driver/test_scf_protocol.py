"""The graduated SCF protocol: exported names and a smoke run.

Graduated from the f-block experiment scripts on 2026-10-11; these
tests pin the interface the chains rely on (five guesses, the damping
grid, the relay candidate) without exercising multi-solution systems --
those live in the experiment records.
"""
import pytest

pytest.importorskip("pyscf")


def test_the_names_are_exported():
    from diffmrc.driver.scf_protocol import (GUESSES, RELAX_GRID,
                                             scf_protocol)
    assert len(GUESSES) == 5 and GUESSES[0] is None
    assert RELAX_GRID and all(len(pair) == 2 for pair in RELAX_GRID)
    assert callable(scf_protocol)


def test_h2_converges_and_reports_a_stage():
    from pyscf import gto

    from diffmrc.driver.scf_protocol import scf_protocol

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf, e, reliable, spread, stage = scf_protocol(mol, 0, "h2-smoke")
    assert mf is not None and mf.converged
    assert reliable is True
    assert e == pytest.approx(-1.1167593, abs=1e-4)
    assert isinstance(stage, str) and stage


def test_a_relay_density_joins_the_candidates():
    from pyscf import gto

    from diffmrc.driver.scf_protocol import scf_protocol

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf0, e0, *_ = scf_protocol(mol, 0, "h2-first")
    dm = mf0.make_rdm1()
    mf, e, reliable, spread, stage = scf_protocol(mol, 0, "h2-relay",
                                                  dm0_guess=dm)
    assert mf.converged and reliable
    assert e == pytest.approx(float(e0), abs=1e-9)
