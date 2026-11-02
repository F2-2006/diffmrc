"""The deduplicated Si correction: one assembly plus a class selection.

``nevpt2_e2_full2_si_exact`` used to rebuild all six core classes twice
and cancel five of them against themselves.  Only the Si class differs
between the masked and the exact form, so the current path evaluates the
release total once and applies ``si_exact - si_masked``.

These tests pin the equivalence against the old three-pass value and
check the two halves separately, so a wrong class selection cannot hide
inside a cancelling total.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _five_core_classes(blocks, eps, ncore, nact, eri_mo):
    """The five core classes other than Si, assembled as ``e2_core`` does.

    This restates the reference's slices and contraction order rather
    than calling a shared helper, so a slip in the class selection cannot
    be copied into both sides of the comparison.
    """
    from diffmrc.core.nevpt2.core import (make_a3, make_a9, make_a12,
                                          make_a13, make_hdm1, make_hdm2,
                                          make_hdm3, make_k27, sij_e2,
                                          sijr_e2, sir_e2, sijrs_e2,
                                          srsi_e2)

    h1e_act = blocks['h1e_act']
    eri_act = blocks['eri_act']
    dm1, dm2, dm3 = blocks['dm1'], blocks['dm2'], blocks['dm3']
    nocc = ncore + nact
    nvir = eri_mo.shape[0] - nocc
    eps_core = eps[:ncore]
    eps_vir = eps[nocc:]
    t = jnp.einsum('prqs->pqrs', eri_mo)
    h2e_v_sij = t[ncore:nocc, ncore:nocc, :ncore, :ncore]
    h2e_v_sijr = t[nocc:, ncore:nocc, :ncore, :ncore]
    h2e_v_srsi = t[nocc:, nocc:, :ncore, ncore:nocc]
    h2e_v1_sir = t[nocc:, ncore:nocc, :ncore, ncore:nocc]
    h2e_v2_sir = t[nocc:, ncore:nocc, ncore:nocc, :ncore]
    h1e_v_sir = blocks['h1eff'][nocc:, :ncore]
    cvcv = eri_mo[:ncore, nocc:, :ncore, nocc:].reshape(ncore * nvir,
                                                        ncore * nvir)
    hdm1 = make_hdm1(dm1)
    hdm2 = make_hdm2(dm1, dm2)
    hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2)
    e = sijrs_e2(cvcv, eps_core, eps_vir, ncore, nvir)[0]
    a3 = make_a3(h1e_act, eri_act, dm1, dm2, hdm1)
    e = e + sijr_e2(h2e_v_sijr, hdm1, a3, eps_vir, eps_core, ncore)
    k27 = make_k27(h1e_act, eri_act, dm1, dm2)
    e = e + srsi_e2(h2e_v_srsi, dm1, k27, eps_vir, eps_core, nvir)
    a9 = make_a9(h1e_act, eri_act, hdm1, hdm2, hdm3)
    e = e + sij_e2(h2e_v_sij, hdm2, a9, eps_core, ncore)
    a12 = make_a12(h1e_act, eri_act, dm1, dm2, dm3)
    a13 = make_a13(h1e_act, eri_act, dm1, dm2, dm3)
    e = e + sir_e2(h2e_v1_sir, h2e_v2_sir, h1e_v_sir, dm1, dm2, dm3,
                   a12, a13, eps_core, eps_vir)
    return e


def _h2o_631g_core_arrays():
    """NEVPT2 inputs on H2O/6-31G CAS(2,2) with a four-orbital core."""
    pytest.importorskip("pyscfad")
    from pyscfad import gto, scf

    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.orblib import orbital_pairs
    from diffmrc.driver.nevpt2 import _casscf_solution, _spin_links

    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0 0 0.757; H 0 0.586 -0.478"
    mol.basis = "6-31g"
    mol.unit = "B"
    mol.verbose = 0
    mol.build()
    nmo = mol.nao_nr()
    ncore, nact, nele = 4, 2, 2
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    theta = jnp.zeros(len(pairs))
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    E0c, _C, ci, eps, h1c, eric = _casscf_solution(
        mf.mo_coeff, mf.get_hcore(), mol.intor("int2e", aosym="s1"), theta,
        ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
        ci_solver="blocked")
    dets = cas_dets_for2(ncore, nact, nele)
    la, lb = _spin_links(nact, nele)
    return dict(h1e_mo=h1c, eri_mo=eric, c_ci=jnp.asarray(ci), dets=dets,
                E0=E0c, eps=eps, ncore=ncore, nact=nact, nocc=ncore + nact,
                links_a=la, links_b=lb)


def test_the_deduplicated_si_correction_reproduces_the_reference():
    """One assembly plus a class selection against the three-pass value."""
    from diffmrc.core.nevpt2.core import e2_core
    from diffmrc.core.nevpt2.impl import nevpt2_e2_full2
    from diffmrc.driver.nevpt2_zvector import (
        _core_blocks, _e2_core_exact_si, _si_inputs, _si_value,
        nevpt2_e2_full2_si_exact)

    a = _h2o_631g_core_arrays()
    kw = dict(ncore=a["ncore"], nact=a["nact"], nocc=a["nocc"],
              links_a=a["links_a"], links_b=a["links_b"])

    e2_ref, eps = nevpt2_e2_full2(a["h1e_mo"], a["eri_mo"], a["c_ci"],
                                  a["dets"], a["E0"], mo_energy=a["eps"],
                                  **kw)
    blocks = _core_blocks(a["h1e_mo"], a["eri_mo"], a["c_ci"], a["dets"],
                          a["ncore"], a["nact"], a["nocc"], eps,
                          a["links_a"], a["links_b"])

    e2_core_masked = e2_core(blocks["h1e_act"], blocks["eri_act"],
                             blocks["dm1"], blocks["dm2"], blocks["dm3"],
                             eps, a["ncore"], a["nact"], blocks["f3ca"],
                             blocks["f3ac"], eri_mo=a["eri_mo"],
                             h1eff=blocks["h1eff"])
    e2_core_exact = _e2_core_exact_si(blocks, eps, a["ncore"], a["nact"],
                                      a["eri_mo"])

    si_in = _si_inputs(blocks, eps, a["ncore"], a["nact"], a["eri_mo"])
    si_masked = _si_value(si_in, "masked")
    si_exact = _si_value(si_in, "exact")
    five = _five_core_classes(blocks, eps, a["ncore"], a["nact"], a["eri_mo"])

    # All differences are measured against the magnitude of the core-class
    # sum, not against the Si term: ``e2_core - five`` cancels two numbers
    # of order 0.07 Eh to leave a Si term of order 1e-8 Eh, so a relative
    # tolerance on the Si term alone would measure the cancellation, not
    # the class selection.
    scale = max(abs(float(e2_core_masked)), abs(float(e2_ref)), 1e-30)

    def rel(x, y):
        return float(abs(x - y) / scale)

    # 1) the Si class, recovered from each assembly and from the selector
    assert rel(float(e2_core_masked - five), float(si_masked)) < 1e-14
    assert rel(float(e2_core_exact - five), float(si_exact)) < 1e-14
    # 2) the five untouched classes are the same in both assemblies
    five_masked = float(e2_core_masked - si_masked)
    five_exact = float(e2_core_exact - si_exact)
    assert rel(five_masked, five_exact) < 1e-14
    assert rel(five_masked, float(five)) < 1e-14
    # 3) the total against the reconstructed three-pass value
    old = float(e2_ref - e2_core_masked + e2_core_exact)
    new, _ = nevpt2_e2_full2_si_exact(a["h1e_mo"], a["eri_mo"], a["c_ci"],
                                      a["dets"], a["E0"],
                                      mo_energy=a["eps"], **kw)
    new = float(new)
    assert rel(new, old) < 1e-14
    # The comparison is not vacuous: the two Si forms differ here.
    assert abs(float(si_exact) - float(si_masked)) > 0.0


def test_the_deduplicated_path_is_the_release_path_without_a_core():
    """With the core classes switched off the Si correction is skipped and
    the total must be the release total, bit for bit.

    ``links_a is None`` is the production switch that drops every core
    class inside ``nevpt2_e2_full2``; it is the same branch the exact-Si
    wrapper must take, and it must not touch the number.
    """
    from diffmrc.core.nevpt2.impl import nevpt2_e2_full2
    from diffmrc.driver.nevpt2_zvector import nevpt2_e2_full2_si_exact

    a = _h2o_631g_core_arrays()
    kw = dict(ncore=a["ncore"], nact=a["nact"], nocc=a["nocc"],
              links_a=None, links_b=None)
    e_ref, _ = nevpt2_e2_full2(a["h1e_mo"], a["eri_mo"], a["c_ci"],
                               a["dets"], a["E0"], mo_energy=a["eps"], **kw)
    e_new, _ = nevpt2_e2_full2_si_exact(a["h1e_mo"], a["eri_mo"], a["c_ci"],
                                        a["dets"], a["E0"],
                                        mo_energy=a["eps"], **kw)
    assert float(e_new) == float(e_ref)
