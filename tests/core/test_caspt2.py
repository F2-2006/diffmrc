"""Contracted CASPT2 engine: regression of the ORCA-anchored class values."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscf = pytest.importorskip("pyscf")


def _ints(Cm, h1e, eri):
    h1 = jnp.einsum('ai,ab,bj->ij', Cm, h1e, Cm)
    z = jnp.einsum('pi,pqrs->iqrs', Cm, eri)
    z = jnp.einsum('qj,iqrs->ijrs', Cm, z)
    z = jnp.einsum('rk,ijrs->ijks', Cm, z)
    return h1, jnp.einsum('sl,ijks->ijkl', Cm, z)


def test_h2o_contracted_classes_regression():
    """H2O/6-31G CAS(2,2): pinned from the 2026-10-02 server run (t328w,
    rep-act diagonal-complete scheme).  ORCA per-class anchors live in
    server/caspt2_anchor/h2o_cas22.log; measured ratios to ORCA are
    V0_ijab 1.02 / Vm1_iab 0.94 / Vm2_ab 1.00 / V1_ija 0.89 / V2_ij 0.96.
    """
    from pyscf import gto, scf, mcscf

    from diffmrc.core.casci import cs_casci
    from diffmrc.core.caspt2 import caspt2_contracted_e2
    from diffmrc.core.nevpt2.impl import canonicalize_jax, cas_dets_for2

    mol = gto.M(atom='O 0 0 0; H 0.7557 0 0.588; H -0.7557 0 0.588',
                basis='6-31g')
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.kernel()
    C = jnp.asarray(mc.mo_coeff)
    h1e = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    eri = jnp.asarray(mol.intor('int2e', aosym='s1'))

    h1_mo, eri_mo = _ints(C, h1e, eri)
    _, ci = cs_casci(h1_mo[:6, :6], eri_mo[:6, :6, :6, :6],
                     ncore=4, nact=2, nele_act=2, nroots=1, return_ci=True)
    C2, ci2, eps = canonicalize_jax(C, h1e, eri, ci, ncore=4, nact=2,
                                   nele_act=2)
    h1_mo2, eri_mo2 = _ints(C2, h1e, eri)
    dets = cas_dets_for2(4, 2, 2)
    total, bd = caspt2_contracted_e2(h1_mo2, eri_mo2, ci2, dets,
                                     ncore=4, nact=2, nele=2, eps=eps)
    assert bd['V0_ijab'] == pytest.approx(-0.047421, abs=3e-4)
    assert bd['Vm1_iab'] == pytest.approx(-0.030215, abs=3e-4)
    assert bd['Vm2_ab'] == pytest.approx(-0.008006, abs=3e-4)
    assert bd['V1_ija'] == pytest.approx(-0.021235, abs=3e-4)
    assert bd['V2_ij'] == pytest.approx(-0.005060, abs=3e-4)
    assert np.isfinite(total)
    assert total == pytest.approx(-0.111936, abs=1e-3)
