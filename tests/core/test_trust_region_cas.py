"""Small-CAS theta optimization through the trust-region stage.

Gated on pyscfad: the reference orbitals come from RHF and the energy
path runs the packaged CASCI kernel. The systems and reference energies
are the ones recorded by the project's theta-landscape work:

* H2O/6-31g CAS(2,2) from theta = 0: stationary saddle,
  E = -75.984881801724, three negative Hessian eigenvalues.
* H2O/6-31g CAS(2,2) from the seeded perturbed start (0.3 * N(0,1),
  seed 1000, the "rand0" of probe_h2o_lower_basin): lower branch,
  E = -76.006040634709, 13.28 kcal/mol below the saddle.
* CO/6-31g CAS(2,2): both stages must land on the same branch from the
  same start.

These are integration checks; the unit behaviour is in
``test_trust_region.py`` and ``test_optimizer_trust_region.py``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscfad = pytest.importorskip("pyscfad")
from pyscfad import gto as pgto, scf as pscf  # noqa: E402

from diffmrc.core.casci import cs_casci_blocked  # noqa: E402
from diffmrc.core.optimizer import (  # noqa: E402
    optimize_theta_scipy,
    optimize_theta_trust_region,
)
from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.core.rotation import block_rotation  # noqa: E402
from diffmrc.core.stationary import curvature_report  # noqa: E402

H2O_ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
CO_ATOM = 'C 0 0 0; O 0 0 1.1283'
SEED = 1000
PERTURB = 0.3

# Project reference values (probe_saddle_refined / probe_h2o_lower_basin,
# 2026-09-22).
H2O_SADDLE_E = -75.984881801724
H2O_LOWER_E = -76.006040634709


def _theta_energy(atom, ncore, nact, nele, basis='6-31g'):
    mol = pgto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.build()
    nmo = mol.nao_nr()
    mf = pscf.RHF(mol)
    mf.kernel()
    C0 = np.asarray(mf.mo_coeff)
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)

    def E_cas(C):
        Cj = jnp.asarray(C)
        nm = ncore + nact
        h1 = jnp.einsum('ai,ab,bj->ij', Cj, h1e, Cj)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                        eri)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + enuc

    E_theta = lambda t: E_cas(C0 @ block_rotation(t, nmo, pairs))
    return E_theta, len(pairs)


def _spectrum(E_theta, theta):
    H = np.asarray(jax.hessian(E_theta)(jnp.asarray(theta)))
    return curvature_report(H)


@pytest.fixture(scope='module')
def h2o():
    return _theta_energy(H2O_ATOM, ncore=4, nact=2, nele=2)


def test_h2o_theta_zero_reaches_the_recorded_saddle(h2o):
    """The trust region reproduces the stationary point the project pinned.

    The point is a saddle, and that is the surface's property, not the
    optimizer's: the endpoint's Hessian must still carry negative
    curvature. A trust region does not change which stationary point a
    start converges to; it is the start that selects the basin.
    """
    E_theta, npar = h2o
    E_jit, theta, gnorm, info = optimize_theta_trust_region(
        E_theta, npar, steps=2000, g_conv=1e-8, theta0=np.zeros(npar))
    E = float(E_theta(jnp.asarray(theta)))
    assert abs(E - H2O_SADDLE_E) < 1e-5, (E, H2O_SADDLE_E)
    assert gnorm < 1e-5
    assert info['ok'], info
    sp = _spectrum(E_theta, theta)
    assert sp.n_negative >= 1, sp


def test_h2o_lower_branch_is_reachable_from_the_documented_start(h2o):
    """The near-degenerate case: the same surface has a second solution.

    From the seeded perturbed start the project's L-BFGS protocol reached
    -76.006040634709 and stalled at |g| ~ 4e-4; the trust region must
    reach the same branch with a smaller gradient, and the endpoint must
    be a minimum (no negative curvature).
    """
    E_theta, npar = h2o
    rng = np.random.default_rng(SEED)
    theta0 = PERTURB * rng.standard_normal(npar)
    E_jit, theta, gnorm, info = optimize_theta_trust_region(
        E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
    E = float(E_theta(jnp.asarray(theta)))
    assert E < -76.0, E
    assert abs(E - H2O_LOWER_E) < 1e-4, (E, H2O_LOWER_E)
    assert gnorm < 1e-5
    sp = _spectrum(E_theta, theta)
    assert sp.n_negative == 0, sp

    _, th_lb, g_lb, info_lb = optimize_theta_scipy(
        E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
    E_lb = float(E_theta(jnp.asarray(th_lb)))
    assert abs(E_lb - E) < 1e-4, (E_lb, E)


def test_co_branches_agree_between_lbfgs_and_the_trust_region():
    E_theta, npar = _theta_energy(CO_ATOM, ncore=5, nact=2, nele=2)
    starts = [('zeros', np.zeros(npar))]
    rng = np.random.default_rng(SEED)
    starts.append(('rand0', PERTURB * rng.standard_normal(npar)))
    for name, theta0 in starts:
        _, th_tr, g_tr, info_tr = optimize_theta_trust_region(
            E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
        _, th_lb, g_lb, info_lb = optimize_theta_scipy(
            E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
        E_tr = float(E_theta(jnp.asarray(th_tr)))
        E_lb = float(E_theta(jnp.asarray(th_lb)))
        assert abs(E_tr - E_lb) < 1e-5, (name, E_tr, E_lb)
        assert g_tr < 1e-4, (name, g_tr)
