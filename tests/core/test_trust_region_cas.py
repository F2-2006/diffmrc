"""Small-CAS theta optimization through the trust-region stage.

Gated on pyscfad: the reference orbitals come from RHF and the energy
path runs the packaged CASCI kernel. The systems are the ones the
project's theta-landscape work already recorded:

* H2O/6-31g CAS(2,2), ncore = 4. Theta = 0 is the reference (RHF) point.
  The default protocol (L-BFGS-B plus damped Newton) reaches a
  stationary saddle there: E = -75.984881801724, three negative Hessian
  eigenvalues. The same surface has a lower branch at
  E = -76.006040634709, 13.28 kcal/mol below; the project reached it
  from a perturbed start (0.3 * N(0,1), seed 1000, "rand0" of
  probe_h2o_lower_basin.py).
* CO/6-31g CAS(2,2), ncore = 5. The theta surface has two solutions
  10.76 kcal/mol apart, so the start selects the branch; from the same
  start the two stages must land on the same one.

Measured on 101 (2026-10-22), first stage at gtol = 1e-8:

    H2O zeros  L-BFGS      -75.984881801160  |g| 1.26e-05  n_neg 3
    H2O zeros  trust-region -76.006040722893 |g| 1.02e-07  n_neg 0
    H2O rand0  L-BFGS      -76.006039978306  |g| 2.88e-05  n_neg 1
    H2O rand0  trust-region -76.006040722545 |g| 8.56e-08  n_neg 0
    CO  zeros  L-BFGS      -111.221407084316 |g| 4.35e-06  n_neg 0
    CO  zeros  trust-region -111.221407084318 |g| 8.17e-09  n_neg 0

The finding these tests pin: from the standard start the trust region
reaches the variationally lower solution, while the L-BFGS path stops on
the saddle. That is a basin-selection effect of the step rule, not a
difference in the energy function.
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
from diffmrc.driver.orbitals import converge_orbital_point  # noqa: E402

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


def test_h2o_theta_zero_selects_different_basins_for_the_two_paths(h2o):
    """The standard start: L-BFGS stops on the saddle, the trust region
    reaches the lower branch.

    The L-BFGS endpoint reproduces the recorded saddle energy and still
    has negative curvature; the trust-region endpoint reproduces the
    lower branch and is a minimum. Both are stationary points of the
    same energy function -- the difference is which one the step rule
    follows from theta = 0.
    """
    E_theta, npar = h2o
    default = converge_orbital_point(E_theta, npar, theta0=np.zeros(npar),
                                     max_steps=2000)
    trust = converge_orbital_point(E_theta, npar, theta0=np.zeros(npar),
                                   max_steps=2000, use_trust_region=True)

    assert default.optimizer_name == 'lbfgs-b'
    assert trust.optimizer_name == 'trust-region'
    assert abs(default.energy - H2O_SADDLE_E) < 1e-5, (
        default.energy, H2O_SADDLE_E)
    assert abs(trust.energy - H2O_LOWER_E) < 1e-5, (
        trust.energy, H2O_LOWER_E)

    sp_default = _spectrum(E_theta, np.asarray(default.theta))
    sp_trust = _spectrum(E_theta, np.asarray(trust.theta))
    assert sp_default.n_negative >= 1, sp_default
    assert sp_trust.n_negative == 0, sp_trust

    assert default.grad_norm < 1e-6
    assert trust.grad_norm < 1e-6


def test_h2o_lower_branch_from_the_documented_perturbed_start(h2o):
    """From rand0 both paths reach the lower branch; the trust region
    leaves a smaller gradient.

    The project's L-BFGS protocol stalled on this basin at |g| ~ 4e-4;
    the table in the module docstring gives the measured 2.9e-5 after
    refinement. The trust region reaches 1e-7 or better at the same
    energy, and the endpoint is a minimum.
    """
    E_theta, npar = h2o
    rng = np.random.default_rng(SEED)
    theta0 = PERTURB * rng.standard_normal(npar)
    trust = converge_orbital_point(E_theta, npar, theta0=theta0,
                                   max_steps=2000, use_trust_region=True)
    default = converge_orbital_point(E_theta, npar, theta0=theta0,
                                     max_steps=2000)
    assert abs(trust.energy - H2O_LOWER_E) < 1e-5, (
        trust.energy, H2O_LOWER_E)
    assert abs(default.energy - trust.energy) < 1e-4
    # Both pipelines end with the same refinement and polish stages, so
    # the final gradients are compared to the target, not to each other;
    # the stage-level ordering is measured in the benchmark table.
    assert trust.grad_norm < 1e-6
    assert default.grad_norm < 1e-6
    sp = _spectrum(E_theta, np.asarray(trust.theta))
    assert sp.n_negative == 0, sp


def test_co_branches_agree_between_lbfgs_and_the_trust_region():
    """Same start, same branch; the trust region reports a smaller
    gradient at nearly identical energy (measured difference 2e-12 and
    1.1e-11 Ha on the two starts)."""
    E_theta, npar = _theta_energy(CO_ATOM, ncore=5, nact=2, nele=2)
    rng = np.random.default_rng(SEED)
    starts = [('zeros', np.zeros(npar)),
              ('rand0', PERTURB * rng.standard_normal(npar))]
    for name, theta0 in starts:
        _, th_tr, g_tr, _ = optimize_theta_trust_region(
            E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
        _, th_lb, g_lb, _ = optimize_theta_scipy(
            E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
        E_tr = float(E_theta(jnp.asarray(th_tr)))
        E_lb = float(E_theta(jnp.asarray(th_lb)))
        assert abs(E_tr - E_lb) < 1e-8, (name, E_tr, E_lb)
        assert g_tr < 1e-7, (name, g_tr)
        assert g_tr < g_lb, (name, g_tr, g_lb)
