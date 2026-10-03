"""Regression tests for core.mcpdft (MC-PDFT on-top energy, tPBE).

Anchors: H2/STO-3G CAS(2,2). Engine E_ot = -0.733794563942 is stable
across environments (pyscf 2.14 on 101, side-loaded env on 82; versions
2.8.0/2.12.1/2.14.0 are bitwise equal on identical orbitals, t373). The
tests recompute the engine reference in-process as well, so they do not
depend on the stored golden alone.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("pyscfad")
from pyscf import gto, scf, mcscf, mcpdft  # noqa: E402
from pyscf.mcpdft import _dms  # noqa: E402
from pyscfad.dft import libxc as pxc  # noqa: E402

from diffmrc.core.casci import cs_casci_blocked  # noqa: E402
from diffmrc.core.nevpt2.impl import cas_dets_for2  # noqa: E402
from diffmrc.core import mcpdft as mcpmod  # noqa: E402

H2_E_OT = -0.733794563942
TOL = 1e-10


def _setup():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, 2, 2).run(verbose=0)
    mcp = mcpdft.CASSCF(mf, 'tPBE', 2, 2).run(verbose=0)
    nao = mol.nao_nr()
    ncore, ncas = mc.ncore, mc.ncas
    C = mc.mo_coeff
    h1m = jnp.asarray(C.T @ mf.get_hcore() @ C)
    e1m = jnp.asarray(np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C,
                                mol.intor('int2e', aosym='s1')))
    sl = ncore + ncas
    E, ci = cs_casci_blocked(h1m[:sl, :sl], e1m[:sl, :sl, :sl, :sl],
                             ncore=ncore, nact=ncas, nele_act=2)
    DETS = cas_dets_for2(ncore, ncas, 2)
    grid = mcp.grids
    coords, weights = grid.coords, grid.weights
    ngr = coords.shape[0]
    raw = np.asarray(mcp.otfnal._numint.eval_ao(mol, coords, deriv=1))
    assert raw.shape == (4, ngr, nao)
    aoT, aoT_g = raw[0], raw[1:4].transpose(1, 0, 2)
    Cact = C[:, ncore:ncore + ncas]
    acas = aoT @ Cact
    dm1s = np.asarray(mcp.fcisolver.make_rdm1s(np.asarray(mc.ci), ncas,
                                               (1, 1)))
    d1f = _dms.casdm1s_to_dm1s(mcp, dm1s, mo_coeff=C, ncore=ncore)

    def gd(D):
        return (np.einsum('gr,rs,gs->g', aoT, D, aoT),
                2.0 * np.einsum('gxr,rs,gs->gx', aoT_g, D, aoT))
    ra, ga = gd(d1f[0])
    rb, gb = gd(d1f[1])
    return dict(mcp=mcp, ci=ci, DETS=DETS, ncas=ncas, ncore=ncore,
                acas=acas, weights=weights, ra=ra, ga=ga, rb=rb, gb=gb,
                Cact=Cact)


def test_pi_factorization_matches_loop_form():
    d = _setup()
    W = mcpmod.w_sigma(d["ci"], d["DETS"], d["ncas"], d["ncore"])
    Pi_fac = np.asarray(mcpmod.pi_active(W, jnp.asarray(d["acas"])))
    # loop form (A - B_ss) through build_M
    M = mcpmod.build_M(d["ci"], d["DETS"], 2 * (d["ncore"] + d["ncas"]))
    c = jnp.asarray(d["ci"])
    ncas, ncore = d["ncas"], d["ncore"]
    A = np.zeros((ncas,) * 4)
    Bss = np.zeros((ncas,) * 4)
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    a = b = 0.0
                    for sg in (0, 1):
                        for t in (0, 1):
                            a += float(c @ (M[(2 * (ncore + p) + sg,
                                               2 * (ncore + q) + sg)]
                                            @ M[(2 * (ncore + r) + t,
                                                 2 * (ncore + s) + t)]) @ c)
                        b += float(c @ (M[(2 * (ncore + p) + sg,
                                           2 * (ncore + q) + sg)]
                                        @ M[(2 * (ncore + r) + sg,
                                             2 * (ncore + s) + sg)]) @ c)
                    A[p, q, r, s] = a
                    Bss[p, q, r, s] = b
    acas = d["acas"]
    Pi_loop = 0.5 * np.einsum('gi,gj,gk,gl,ijkl->g', acas, acas, acas, acas,
                              A - Bss)
    assert np.abs(Pi_fac - Pi_loop).max() < 1e-14


def test_eot_matches_engine_and_golden():
    d = _setup()
    E_ot = float(mcpmod.mcpdft_eot(
        d["ci"], d["DETS"], d["ncas"], d["ncore"],
        d["ra"], d["ga"], d["rb"], d["gb"], d["ra"], d["rb"],
        jnp.asarray(d["acas"]), jnp.asarray(d["weights"]), pxc.eval_xc))
    assert abs(E_ot - float(d["mcp"].e_ot)) < TOL
    assert abs(E_ot - H2_E_OT) < TOL


def test_eot_differentiable_in_ci():
    d = _setup()
    acas_j = jnp.asarray(d["acas"])
    w_j = jnp.asarray(d["weights"])

    def f(cc):
        return mcpmod.mcpdft_eot(cc, d["DETS"], d["ncas"], d["ncore"],
                                 d["ra"], d["ga"], d["rb"], d["gb"],
                                 d["ra"], d["rb"], acas_j, w_j, pxc.eval_xc)
    g = np.asarray(jax.grad(f)(jnp.asarray(d["ci"])))
    assert np.isfinite(g).all()
    eps = 1e-6
    for i in range(len(g)):
        cp = np.array(d["ci"], copy=True)
        cm = np.array(d["ci"], copy=True)
        cp[i] += eps
        cm[i] -= eps
        fd = (float(f(jnp.asarray(cp))) - float(f(jnp.asarray(cm)))) / (2 * eps)
        assert abs(fd - g[i]) < 1e-5


def _random_points(seed, n, rho_lo, rho_hi, smax):
    """Translated-layout points: log-uniform spin densities, reduced
    gradients up to smax (|grad rho_s| = s * 2 kF rho_s)."""
    rng = np.random.default_rng(seed)
    rho_t = np.zeros((2, 4, n))
    for s in (0, 1):
        rho_s = 10.0 ** rng.uniform(np.log10(rho_lo), np.log10(rho_hi), n)
        kf = (3 * np.pi ** 2 * rho_s) ** (1.0 / 3.0)
        gmag = rng.uniform(0.0, smax, n) * 2 * kf * rho_s
        direc = rng.normal(size=(3, n))
        direc /= np.sqrt((direc ** 2).sum(0))
        rho_t[s, 0] = rho_s
        rho_t[s, 1:4] = direc * gmag
    return rho_t


def test_pbe_kernel_matches_libxc_pointwise():
    """t385: the jnp PBE kernel against libxc over the physical domain.

    Machine precision for densities from 1e-8 up, at reduced gradients up
    to 3. Below about 1e-13 libxc's internal low-density guards differ
    from the closed-form evaluation (up to ~1e-4 absolute there, where
    grid weights put the on-top integral contribution below 1e-25; the
    deviation and its domain are recorded by the runner t385/f).
    """
    from pyscf.dft import libxc as pyscf_libxc

    rho_t = _random_points(seed=7, n=4000, rho_lo=1e-8, rho_hi=1e1, smax=3.0)
    ref = np.asarray(pyscf_libxc.eval_xc('PBE', rho_t, spin=1, deriv=0)[0])
    ours = np.asarray(mcpmod.pbe_exc(jnp.asarray(rho_t)))
    assert np.abs(ours - ref).max() < 1e-12


def test_pbe_kernel_is_jit_and_grad_safe():
    """The property the driver's optimizer and hessian need -- and the
    reason the kernel exists: pyscfad's libxc wrapper traces into a C
    call and cannot be jitted (t384)."""
    rho_t = jnp.asarray([[0.3, 0.01, 0.0, 0.0],
                         [0.2, -0.02, 0.01, 0.0]])[:, :, None]
    v_eager = mcpmod.pbe_exc(rho_t)
    v_jit = jax.jit(mcpmod.pbe_exc)(rho_t)
    assert abs(float(v_eager[0]) - float(v_jit[0])) < 1e-14
    g = jax.grad(lambda r: jnp.sum(mcpmod.pbe_exc(r)))(rho_t)
    assert np.isfinite(np.asarray(g)).all()
    with pytest.raises(ValueError):
        mcpmod.pbe_eval_xc('PBE', rho_t, spin=0)

