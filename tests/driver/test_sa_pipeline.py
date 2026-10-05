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


def test_sa_root_table_groups_into_physical_states():
    """The raw root table is not a state table: the triplet arrives as
    three degenerate M components, and state_groups must merge them into
    one multiplicity-3 state while leaving the singlets alone.  This is
    the analysis path the M5-C driver script used to pick targets by
    eye, now pinned on a real SA solve."""
    from pyscfad import scf

    from diffmrc.analysis.states import select_groups, state_groups
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.driver.nevpt2 import make_nevpt2_sa_fn

    mol = _h2_631g()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    fs = make_nevpt2_sa_fn(ncore, nact, nele, nmo, pairs, builder,
                           nroots=6, e2_scale=0.0)
    out = fs(mol, jnp.zeros(len(pairs)))
    dets = cas_dets_for2(ncore, nact, nele)
    energies = [float(e) for e, _ in out["states"]]
    rows, groups = state_groups(energies, np.asarray(out["cis"]), dets)

    triplets = select_groups(groups, multiplicity=3)
    assert len(triplets) == 1
    assert len(triplets[0]['roots']) == 3          # M = +1, 0, -1
    assert triplets[0]['szs'] == [0.0, 1.0]
    singlets = select_groups(groups, multiplicity=1, n=None)
    assert len(singlets) == 3
    assert len(groups) == 4
    assert all(g['multiplicity'] is not None for g in groups)


def test_sa_nevpt2_state_table_matches_the_manual_assembly():
    """The driver-level table is the glue the M5-C scripts assemble by
    hand (state_groups on the CI vectors + the representative's vertical
    excitations); it must reproduce that composition value for value."""
    from pyscfad import scf

    from diffmrc.analysis.states import state_groups
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.driver.nevpt2 import (make_nevpt2_sa_theta_fn,
                                       sa_nevpt2_state_table,
                                       sa_vertical_excitations)

    mol = _h2_631g()
    nmo = mol.nao_nr()
    ncore, nact, nele, nroots = 0, 2, 2, 6
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    comp = make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                   builder, nroots=nroots, e2_scale=1.0)
    out = comp(jnp.zeros(len(pairs)))
    dets = cas_dets_for2(ncore, nact, nele)

    rows, groups = sa_nevpt2_state_table(out, dets)
    energies = [float(e) for e, _ in out["states"]]
    usable = [bool(np.isfinite(float(e2))) for _, e2 in out["states"]]
    _, ref_groups = state_groups(energies, np.asarray(out["cis"]), dets,
                                 usable=usable)
    vees = sa_vertical_excitations(out["states"])
    assert len(rows) == len(out["states"])
    assert len(groups) == len(ref_groups) == 4
    for g, rg in zip(groups, ref_groups):
        assert g["roots"] == rg["roots"]
        assert g["multiplicity"] == rg["multiplicity"]
        r = vees[g["representative"]]
        assert g["dE_casscf"] == pytest.approx(float(r["dE_casscf"]),
                                               abs=1e-12)
        a = g["dE_nevpt2"]
        b = float(r["dE_nevpt2"])
        assert (a == pytest.approx(b, abs=1e-12)
                or (np.isnan(a) and np.isnan(b)))
    assert sum(1 for g in groups if g["multiplicity"] == 3) == 1


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


def test_sa_state_weights_single_point_reduces_to_the_single_root():
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                       make_nevpt2_sa_fn)

    mol = _h2_631g()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    th = jnp.zeros(len(pairs))
    joint = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                                 e2_scale=0.0)
    sa = make_nevpt2_sa_fn(ncore, nact, nele, nmo, pairs, builder, nroots=2,
                           e2_scale=0.0, state_weights=[1.0, 0.0])
    assert float(sa(mol, th)["E_sa"]) == pytest.approx(
        float(joint(mol, th)), abs=1e-8)


def test_sa_state_weights_scaling_and_equal_default_agree():
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_sa_fn

    mol = _h2_631g()
    nmo = mol.nao_nr()
    pairs = _pairs(0, 2, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    th = jnp.zeros(len(pairs))

    def run(w=None):
        fn = make_nevpt2_sa_fn(0, 2, 2, nmo, pairs, builder, nroots=2,
                               e2_scale=0.0, state_weights=w)
        return float(fn(mol, th)["E_sa"])

    a = run()
    assert run([0.5, 0.5]) == pytest.approx(a, abs=1e-10)
    assert run([1.0, 1.0]) == pytest.approx(a, abs=1e-10)  # normalized


def _ecp_dimer(heavy, r_bohr, ecp_basis):
    from pyscfad import gto

    mol = gto.Mole()
    mol.atom = [(heavy, (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, r_bohr))]
    mol.unit = "bohr"
    mol.basis = {heavy: ecp_basis, "O": "6-31g"}
    mol.ecp = {heavy: ecp_basis}
    mol.charge = 0
    mol.spin = 0
    mol.build()
    return mol


def test_sa_pipeline_smoke_on_a_lanthanide_ecp_system():
    """CeO/stuttgart_rsc: the SA solve, average-density canonicalization and
    per-state NEVPT2 run end-to-end on ECP integrals with an f element --
    finite numbers, correlation corrections negative, sane energy scale.
    (Theta stays at zero here; the f-block theta-optimization scale is a
    documented follow-up -- npar ~1e3, see the plan file.)"""
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_sa_theta_fn

    mol = _ecp_dimer("Ce", 3.44, "stuttgart_rsc")
    nmo = mol.nao_nr()
    ncore, nact, nele, nroots = 18, 2, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    comp = make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                   builder, nroots=nroots, e2_scale=1.0)
    out = comp(jnp.zeros(len(pairs)))
    E_sa = float(out["E_sa"])
    Ek = np.array([float(e) for e, _ in out["states"]])
    e2 = np.array([float(c) for _, c in out["states"]])
    assert np.isfinite(E_sa) and np.isfinite(Ek).all() and np.isfinite(e2).all()
    assert (e2 <= 1e-9).all()          # correlation lowers the energy
    assert -600.0 < E_sa < -500.0      # CeO/ECP energy scale


def test_sa_pipeline_smoke_on_an_actinide_ecp_system():
    """ThO/stuttgart_rsc: same smoke for the actinide side."""
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_sa_theta_fn

    mol = _ecp_dimer("Th", 3.47, "stuttgart_rsc")
    nmo = mol.nao_nr()
    ncore, nact, nele, nroots = 18, 2, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)  # noqa: E731
    comp = make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                   builder, nroots=nroots, e2_scale=1.0)
    out = comp(jnp.zeros(len(pairs)))
    E_sa = float(out["E_sa"])
    Ek = np.array([float(e) for e, _ in out["states"]])
    e2 = np.array([float(c) for _, c in out["states"]])
    assert np.isfinite(E_sa) and np.isfinite(Ek).all() and np.isfinite(e2).all()
    assert (e2 <= 1e-9).all()
    assert -600.0 < E_sa < -400.0      # ThO/ECP energy scale


def test_sa_mcpdft_table_matches_the_engine_per_state():
    """H2CO/STO-3G, npi window, two roots (the second is the triplet
    npi*, S^2 = 2): the table's per-state one-step tPBE column against
    pyscf mcpdft.CASCI evaluated at the table's own canonical orbitals.
    The engine sits at exactly our point, so the comparison isolates the
    table's densities and on-top evaluation -- measured 3e-14 (t387).
    """
    from pyscf import gto, mcpdft, scf

    from diffmrc.driver.nevpt2 import sa_mcpdft_table_from_mol

    mol = gto.M(atom='C 0 0 0; O 0 0 1.205; H 0.9425 0 -0.5875; '
                     'H -0.9425 0 -0.5875',
                basis='sto-3g')
    table = sa_mcpdft_table_from_mol(mol, 'npi', lambda m: scf.RHF(m),
                                     nroots=2)
    e_tpbe = np.asarray(table['e_tpbe'])
    assert e_tpbe.shape == (2,)
    assert np.all(np.isfinite(e_tpbe))
    assert e_tpbe[0] < e_tpbe[1]
    assert 'dE_mcpdft' in table['groups'][0]

    mf = scf.RHF(mol).run(verbose=0)
    mcp_casci = getattr(mcpdft, 'CASCI', None)
    if mcp_casci is None:
        from pyscf.mcpdft.casci import CASCI as mcp_casci
    nele = int(table['window']['nele'])
    eng = mcp_casci(mf, 'tPBE', int(table['window']['nact']),
                    (nele // 2, nele // 2))
    eng.mo_coeff = np.asarray(table['out']['C'], dtype=float)
    eng.fcisolver.nroots = 2
    eng.kernel()
    e_eng = np.atleast_1d(np.asarray(
        eng.e_states if hasattr(eng, 'e_states') else eng.e_tot, dtype=float))
    assert np.abs(e_tpbe[:2] - e_eng[:2]).max() < 1e-7
