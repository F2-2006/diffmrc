"""Contracted CASPT2 engine: ORCA-anchored classes and the jax twin."""
from itertools import combinations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _cas_dets(ncore, nact, nele):
    """Determinant bitmaps in the layout of nevpt2.impl.cas_dets_for2
    (core spin orbitals first, then active combinations).  Local copy so
    the synthetic tests stay importable without the pyscfad-dependent
    modules."""
    core_so = list(range(2 * ncore))
    act_so = list(range(2 * ncore, 2 * (ncore + nact)))
    dets = []
    for comb in combinations(act_so, nele):
        det = 0
        for so in core_so + list(comb):
            det |= 1 << so
        dets.append(det)
    return dets


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
    The V0_ia class (Molcas case D) is asserted against its ORCA value
    within the 5% acceptance band; the total is asserted within 10% of
    the ORCA total (the pre-V0_ia total sat at 12%).
    """
    pyscf = pytest.importorskip("pyscf")
    pytest.importorskip("pyscfad")  # the diffmrc chain imports it
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
    # Brillouin-forbidden singles: ORCA reports -0.000000000 / -0.000000002.
    assert abs(bd['Vm1_a']) < 1e-8
    assert abs(bd['V1_i']) < 1e-8
    # Molcas case D anchor: ORCA -0.011135077577 (5% acceptance band).
    assert bd['V0_ia'] == pytest.approx(-0.011135077577, rel=0.05)
    assert total == pytest.approx(-0.12709574191632, rel=0.10)


def _synthetic_case(seed=7, ncore=2, nact=2, nele=2, nmo=6):
    """A small deterministic pseudo-system: symmetric one- and two-electron
    integrals, a normalized CI vector on the S_z = 0 block.  Only used to
    compare the two engines and to check the gradient -- no physical
    meaning."""
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(nmo, nmo))
    h1 = 0.5 * (h1 + h1.T)
    e4 = rng.normal(size=(nmo,) * 4)
    e4 = 0.5 * (e4 + e4.transpose(0, 1, 3, 2))
    e4 = 0.5 * (e4 + e4.transpose(1, 0, 2, 3))
    e4 = 0.5 * (e4 + e4.transpose(2, 3, 0, 1))
    # Well-separated positive orbital energies keep the diagonal
    # denominators away from the guard threshold, so the AD/FD check is
    # not limited by conditioning.
    eps = np.linspace(1.0, 2.0, nmo)
    dets = _cas_dets(ncore, nact, nele)
    amask = sum(1 << so for so in range(0, 2 * (ncore + nact), 2))
    ci = np.zeros(len(dets))
    for i, d in enumerate(dets):
        # S_z = 0 determinants only, for a physical-looking reference.
        if bin(d & amask).count('1') == (2 * ncore + nele) // 2:
            ci[i] = rng.normal()
    ci = ci / np.linalg.norm(ci)
    return h1, e4, eps, dets, ci


def test_jax_twin_matches_numpy_reference():
    """The two engines build the same class sums elementwise."""
    from diffmrc.core.caspt2 import (caspt2_contracted_e2,
                                     caspt2_contracted_e2_jax)

    h1, e4, eps, dets, ci = _synthetic_case()
    e_np, bd_np = caspt2_contracted_e2(h1, e4, ci, dets, ncore=2, nact=2,
                                       nele=2, eps=eps)
    e_jx, bd_jx = caspt2_contracted_e2_jax(jnp.asarray(h1), jnp.asarray(e4),
                                           jnp.asarray(ci), dets, ncore=2,
                                           nact=2, nele=2,
                                           eps=jnp.asarray(eps))
    assert float(e_jx) == pytest.approx(e_np, abs=1e-10)
    for name in bd_np:
        assert float(bd_jx[name]) == pytest.approx(bd_np[name], abs=1e-10)


def test_jax_twin_grad_ad_vs_fd():
    """jax.grad of the contracted energy w.r.t. the CI coefficients agrees
    with central finite differences (the differentiable link of the
    chain)."""
    from diffmrc.core.caspt2 import caspt2_contracted_e2_jax

    h1, e4, eps, dets, ci = _synthetic_case()
    h1_j, e4_j, eps_j = (jnp.asarray(h1), jnp.asarray(e4),
                         jnp.asarray(eps))

    def energy(c):
        return caspt2_contracted_e2_jax(h1_j, e4_j, c, dets, ncore=2, nact=2,
                                        nele=2, eps=eps_j)[0]

    g_ad = np.asarray(jax.grad(energy)(jnp.asarray(ci)))
    delta = 1e-6
    g_fd = np.zeros_like(ci)
    for k in range(len(ci)):
        e_p = np.array(ci)  # copy: the perturbations must not alias
        e_m = np.array(ci)
        e_p[k] += delta
        e_m[k] -= delta
        g_fd[k] = (float(energy(jnp.asarray(e_p)))
                   - float(energy(jnp.asarray(e_m)))) / (2 * delta)
    assert np.all(np.isfinite(g_ad))
    assert np.allclose(g_ad, g_fd, rtol=1e-5, atol=1e-8)
