"""The state-averaged CASSCF + per-state NEVPT2 pipeline (driver layer).

All additive: the core exposes ``cs_casci(nroots>1, return_ci=True)`` (the
equal-weight mean energy and the per-root CI vectors) and the NEVPT2
contraction consumes one vector at a time; the only new construction is
the average-density canonicalization, pinned against the core routine by
a k=1 identity test.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _symmetric_ints(n, seed=0):
    """Synthetic chemically-symmetric integrals on n MOs (test scaffolding,
    not physical): p<->q, r<->s and (pq)<->(rs) symmetries enforced."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = rng.normal(size=(n, n, n, n)) * 0.1
    eri = 0.25 * (eri + eri.transpose(1, 0, 2, 3)
                  + eri.transpose(0, 1, 3, 2)
                  + eri.transpose(2, 3, 0, 1))
    return jnp.asarray(h), jnp.asarray(eri)


def test_sa_cas_solution_matches_the_core_state_average():
    from diffmrc.core.casci import cs_casci
    from diffmrc.driver.nevpt2 import sa_cas_solution

    h, eri = _symmetric_ints(4)
    ncore, nact, nele, nroots = 0, 2, 2, 2
    e_sa, cis = sa_cas_solution(h, eri, ncore=ncore, nact=nact, nele=nele,
                                nroots=nroots)
    e_ref = cs_casci(h, eri, ncore=ncore, nact=nact, nele_act=nele,
                     nroots=nroots)
    assert float(e_sa) == pytest.approx(float(e_ref), abs=1e-12)
    assert np.asarray(cis).shape[1] == nroots
    gram = np.asarray(cis).T @ np.asarray(cis)
    assert np.allclose(gram, np.eye(nroots), atol=1e-10)


def test_sa_canonicalization_reduces_to_the_core_routine_for_one_vector():
    from diffmrc.core.nevpt2.impl import canonicalize_jax
    from diffmrc.driver.nevpt2 import canonicalize_sa_jax, sa_cas_solution

    h, eri = _symmetric_ints(4)
    ncore, nact, nele = 0, 2, 2
    _, cis = sa_cas_solution(h, eri, ncore=ncore, nact=nact, nele=nele,
                             nroots=1)
    C0 = jnp.eye(4)
    C_core, ci_core, eps_core = canonicalize_jax(
        C0, h, eri, cis[:, 0], ncore=ncore, nact=nact, nele_act=nele)
    C_sa, cis_sa, eps_sa = canonicalize_sa_jax(
        C0, h, eri, cis, ncore=ncore, nact=nact, nele=nele)
    assert np.allclose(np.asarray(C_sa), np.asarray(C_core), atol=1e-12)
    assert np.allclose(np.asarray(eps_sa), np.asarray(eps_core), atol=1e-12)
    assert np.allclose(np.asarray(cis_sa)[:, 0], np.asarray(ci_core),
                       atol=1e-12)


def test_sa_canonicalization_keeps_two_vectors_orthonormal():
    from diffmrc.driver.nevpt2 import canonicalize_sa_jax, sa_cas_solution

    h, eri = _symmetric_ints(4)
    ncore, nact, nele = 0, 2, 2
    _, cis = sa_cas_solution(h, eri, ncore=ncore, nact=nact, nele=nele,
                             nroots=2)
    _, cis2, _ = canonicalize_sa_jax(jnp.eye(4), h, eri, cis, ncore=ncore,
                                     nact=nact, nele=nele)
    gram = np.asarray(cis2).T @ np.asarray(cis2)
    assert np.allclose(gram, np.eye(2), atol=1e-10)


def _h2_631g():
    from pyscfad import gto

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "6-31g"
    mol.build()
    return mol


def _pairs(ncore, nact, nmo):
    from diffmrc.core.orblib import orbital_pairs

    return orbital_pairs(ncore, nact, nmo, include_cv=True)


def test_sa_joint_fn_reduces_to_the_single_root_joint_fn():
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn, make_nevpt2_sa_fn

    mol = _h2_631g()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    t = jnp.zeros(len(pairs))
    for scale in (0.0, 1.0):
        f1 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                                  e2_scale=scale)
        fs = make_nevpt2_sa_fn(ncore, nact, nele, nmo, pairs, builder,
                               nroots=1, e2_scale=scale)
        a = float(f1(mol, t))
        out = fs(mol, t)
        b = float(out["E_sa"]) + float(out["states"][0][1])  # scaled e2
        assert b == pytest.approx(a, abs=1e-10)


def test_sa_joint_fn_reports_per_state_energies():
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_sa_fn

    mol = _h2_631g()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    fs = make_nevpt2_sa_fn(ncore, nact, nele, nmo, pairs, builder, nroots=2,
                           e2_scale=0.0)
    out = fs(mol, jnp.zeros(len(pairs)))
    assert len(out["states"]) == 2
    Ek = [float(e) for e, _ in out["states"]]
    assert Ek[0] < Ek[1]
    assert float(np.mean(Ek)) == pytest.approx(float(out["E_sa"]), abs=1e-8)


def test_sa_vertical_excitations_reference_state_zero():
    from diffmrc.driver.nevpt2 import sa_vertical_excitations

    states = [(-1.0, -0.01), (-0.9, -0.03)]
    out = sa_vertical_excitations(states)
    assert out[0]["dE_casscf"] == 0.0
    assert out[0]["dE_nevpt2"] == 0.0
    assert out[1]["dE_casscf"] == pytest.approx(0.1, abs=1e-14)
    # dE_nevpt2 = 0.1 + (e2_1 - e2_0) = 0.1 - 0.02 = 0.08
    assert out[1]["dE_nevpt2"] == pytest.approx(0.08, abs=1e-14)


def test_sa_theta_fn_matches_the_joint_fn_at_fixed_geometry():
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import (make_nevpt2_sa_fn,
                                       make_nevpt2_sa_theta_fn)

    mol = _h2_631g()
    nmo = mol.nao_nr()
    ncore, nact, nele, nroots = 0, 2, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    joint = make_nevpt2_sa_fn(ncore, nact, nele, nmo, pairs, builder,
                              nroots=nroots)
    theta_only = make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                         builder, nroots=nroots)
    th = jnp.zeros(len(pairs))
    a = joint(mol, th)
    b = theta_only(th)
    assert float(a["E_sa"]) == pytest.approx(float(b["E_sa"]), abs=1e-10)
    for (ea, e2a), (eb, e2b) in zip(a["states"], b["states"]):
        assert float(ea) == pytest.approx(float(eb), abs=1e-10)
        assert float(e2a) == pytest.approx(float(e2b), abs=1e-10)
