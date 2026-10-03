"""Regression tests for the MC-PDFT theta chain (driver.kernel).

Mirrors the fast core of the server runner t384 (which additionally
exercises the optimization end to end on the production path). The
levels to read failures against:

- H2/STO-3G anchors sit at machine precision (t384 A);
- H2/6-31G anchors sit at the engine's FCI convergence floor, ~1e-8:
  the engine's davidson vector converges to ~1e-7, the MC-PDFT energy
  is first order in that density error, while our ci comes from a full
  block decomposition (t384 B/C);
- ``pyscf.mcpdft.CASSCF`` is the one-step protocol -- CASSCF-driven
  orbitals with a single tPBE evaluation, which is also the DVS paper's
  protocol (t384 F). The theta chain here minimizes the on-top energy
  itself, so its converged point is stationary for the MC-PDFT surface
  and lies below the one-step value.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("pyscfad")
from pyscf import gto, scf, mcpdft  # noqa: E402
from pyscfad import gto as agto, scf as ascf  # noqa: E402

from diffmrc.core.casci import cs_casci_blocked  # noqa: E402
from diffmrc.core.mcpdft import w_sigma  # noqa: E402
from diffmrc.core.nevpt2.impl import cas_dets_for2  # noqa: E402
from diffmrc.driver.kernel import (ActiveSpace,  # noqa: E402
                                   mcpdft_energy_function,
                                   mcpdft_terms_function,
                                   partition_from_occupations, search_point)

H2_STO3G_E_TOT = -1.156628479816


def _engine(basis='sto-3g'):
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis=basis)
    mf = scf.RHF(mol).run(verbose=0)
    mcp = mcpdft.CASSCF(mf, 'tPBE', 2, 2)
    if basis != 'sto-3g':
        mcp.fcisolver.conv_tol = 1e-12
    mcp.run(verbose=0)
    nmo = mol.nao_nr()
    space = ActiveSpace(core=list(range(mcp.ncore)),
                        active=list(range(mcp.ncore, mcp.ncore + mcp.ncas)),
                        virtual=list(range(mcp.ncore + mcp.ncas, nmo)),
                        n_elec=2)
    return mol, mf, mcp, space


def test_theta_zero_matches_engine_and_golden():
    mol, mf, mcp, space = _engine('sto-3g')
    terms, npar = mcpdft_terms_function(mol, mcp.mo_coeff, space)
    E_tot, E_ot, E_mcwfn, E_cas = (float(x) for x in terms(jnp.zeros(npar)))
    assert abs(E_tot - float(mcp.e_tot)) < 1e-10
    assert abs(E_ot - float(mcp.e_ot)) < 1e-10
    assert abs(E_mcwfn - (float(mcp.e_tot) - float(mcp.e_ot))) < 1e-10
    assert abs(E_cas + float(mol.energy_nuc())
               - float(mcp.e_mcscf)) < 1e-10
    assert abs(E_tot - H2_STO3G_E_TOT) < 1e-9


def test_dm1s_from_w_matches_engine():
    mol, mf, mcp, space = _engine('sto-3g')
    ncore, ncas = mcp.ncore, mcp.ncas
    C = np.asarray(mcp.mo_coeff)
    h1m = jnp.asarray(C.T @ mf.get_hcore() @ C)
    e1m = jnp.asarray(np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C,
                                mol.intor('int2e', aosym='s1')))
    sl = ncore + ncas
    _, ci = cs_casci_blocked(h1m[:sl, :sl], e1m[:sl, :sl, :sl, :sl],
                             ncore=ncore, nact=ncas, nele_act=2)
    W = w_sigma(ci, cas_dets_for2(ncore, ncas, 2), ncas, ncore)
    dm1s = np.asarray(jnp.einsum('spqI,I->spq', W, ci))
    eng = np.asarray(mcp.fcisolver.make_rdm1s(np.asarray(mcp.ci), ncas,
                                              (1, 1)))
    assert np.abs(dm1s - eng).max() < 1e-12


def test_theta_gradient_matches_fd():
    mol, mf, mcp, space = _engine('6-31g')
    terms, npar = mcpdft_terms_function(mol, mcp.mo_coeff, space)

    def E(t):
        return terms(t)[0]

    rng = np.random.default_rng(3)
    eps = 1e-5
    for th in (np.zeros(npar),
               rng.normal(scale=0.05, size=npar)):
        g_ad = np.asarray(jax.grad(E)(jnp.asarray(th)))
        g_fd = np.zeros(npar)
        for i in range(npar):
            tp, tm = np.array(th), np.array(th)
            tp[i] += eps
            tm[i] -= eps
            g_fd[i] = (float(E(jnp.asarray(tp)))
                       - float(E(jnp.asarray(tm)))) / (2 * eps)
        assert np.abs(g_ad - g_fd).max() < 1e-8


def test_production_path_jit_hessian_and_relaxation():
    """The pyscfad path with the theta chain's own optimizer.

    The two properties the jnp kernel exists for -- jit and hessian --
    plus the relaxed minimum: stationary (|grad| ~ 1e-14 in t384) and
    strictly below the RHF-start energy.
    """
    mol_ad = agto.Mole()
    mol_ad.atom = 'H 0 0 0; H 0 0 0.74'
    mol_ad.basis = '6-31g'
    mol_ad.build()
    mf_ad = ascf.RHF(mol_ad)
    mf_ad.kernel(verbose=0)
    space = partition_from_occupations(np.asarray(mf_ad.mo_occ), [0, 1])
    E_theta, npar = mcpdft_energy_function(
        mol_ad, np.asarray(mf_ad.mo_coeff), space)
    assert npar == 4
    E0 = float(E_theta(jnp.zeros(npar)))
    assert abs(float(jax.jit(E_theta)(jnp.zeros(npar))) - E0) < 1e-12
    H = jax.hessian(E_theta)(jnp.zeros(npar))
    assert np.isfinite(np.asarray(H)).all()
    point = search_point(E_theta, npar, tries=1, optimize_tol=1e-6,
                         refine_target=1e-11)
    assert float(point.grad_norm) < 1e-8
    assert point.energy < E0 - 1e-6
