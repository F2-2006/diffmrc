"""Feasibility probe for the H2O/cc-pVDZ NEVPT2 gradient validation.

Measures, at theta = 0 (not yet the stationary point):

  * basis size, rotation-parameter count, SCF convergence;
  * the cost of one CASSCF value+gradient and one NEVPT2 value+gradient;
  * a nuclear-gradient differentiation of the CASSCF chain (the pyscfad
    integral-derivative memory) and, if it fits, of the NEVPT2 chain.

No validation claim is made here; this only decides whether the full
staged run fits the machine budget.
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["MKL_CBWR"] = "COMPATIBLE"
_src = os.environ.get('DIFFMRC_SRC')
if _src:
    sys.path.insert(0, _src)

import numpy as np                                                # noqa: E402
import jax                                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp                                      # noqa: E402
from pyscfad import gto as pgto, scf as pscf                      # noqa: E402

from diffmrc.core.orblib import orbital_pairs                     # noqa: E402
from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,          # noqa: E402
                                   nevpt2_nuclear_gradient)
from diffmrc.driver.nevpt2_zvector import (                       # noqa: E402
    _components_from_si_exact,
    align_reference_blocks,
    make_nevpt2_joint_fn_si_exact)

ATOM = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
BASIS = os.environ.get('BSV_BASIS', 'cc-pvdz')
NACT, NELE = 2, 2


def rhf_tight(m):
    mf = pscf.RHF(m)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    return mf


def main():
    t0 = time.time()
    mol = pgto.Mole()
    mol.atom = ATOM
    mol.basis = BASIS
    mol.build()
    nmo = mol.nao_nr()
    ncore = mol.nelectron // 2 - NELE // 2
    pairs = orbital_pairs(ncore, NACT, nmo, include_cv=True)
    print('basis = %s   nao = %d   ncore = %d   npar = %d   natm = %d'
          % (BASIS, nmo, ncore, len(pairs), mol.natm), flush=True)

    mf = rhf_tight(mol)
    mf.kernel()
    print('SCF converged = %s  cycles = %s  E = %.12f  [%.1fs]'
          % (bool(mf.converged), getattr(mf, 'cycles', '?'),
             float(mf.e_tot), time.time() - t0), flush=True)

    theta = jnp.zeros(len(pairs))

    t1 = time.time()
    f0 = make_nevpt2_joint_fn(ncore, NACT, NELE, nmo, pairs, rhf_tight,
                              e2_scale=0.0, ci_solver='blocked',
                              eig_solver='lanczos')
    E0, g0 = jax.value_and_grad(lambda t: f0(mol, t))(theta)
    print('CASSCF value+grad (cold) = %.12f  |gtheta|max = %.3e  [%.1fs]'
          % (float(E0), float(np.abs(g0).max()), time.time() - t1), flush=True)

    t1 = time.time()
    E0b, g0b = jax.value_and_grad(lambda t: f0(mol, t))(theta)
    print('CASSCF value+grad (warm) [%.2fs]'
          % (time.time() - t1), flush=True)

    t1 = time.time()
    f1 = make_nevpt2_joint_fn(ncore, NACT, NELE, nmo, pairs, rhf_tight,
                              e2_scale=1.0, ci_solver='blocked',
                              eig_solver='lanczos')
    E1, g1 = jax.value_and_grad(lambda t: f1(mol, t))(theta)
    print('NEVPT2 value+grad (cold) = %.12f  e2 = %.10f  '
          '|gtheta|max = %.3e  [%.1fs]'
          % (float(E1), float(E1 - E0), float(np.abs(g1).max()),
             time.time() - t1), flush=True)

    t1 = time.time()
    E1b, g1b = jax.value_and_grad(lambda t: f1(mol, t))(theta)
    print('NEVPT2 value+grad (warm) = %.12f  [%.2fs]'
          % (float(E1b), time.time() - t1), flush=True)

    t1 = time.time()
    try:
        gR = nevpt2_nuclear_gradient(mol, lambda m_: f0(m_, theta))
        print('CASSCF nuclear grad (AD) |g| = %.3e  [%.1fs]'
              % (float(np.abs(gR).max()), time.time() - t1), flush=True)
    except Exception as ex:                                       # noqa: BLE001
        print('CASSCF nuclear grad FAILED: %s: %s  [%.1fs]'
              % (type(ex).__name__, str(ex)[:200], time.time() - t1),
              flush=True)

    # Alignment-at-base check: the map must be the identity there.
    C = jnp.asarray(np.asarray(mf.mo_coeff))
    S = jnp.asarray(mol.intor('int1e_ovlp'))
    C_al = align_reference_blocks(C, S, C, [(0, ncore),
                                            (ncore, ncore + NACT),
                                            (ncore + NACT, nmo)])
    print('alignment at base: max|C_al - C| = %.3e'
          % float(np.abs(np.asarray(C_al) - np.asarray(C)).max()),
          flush=True)

    print('total wall %.1fs' % (time.time() - t0), flush=True)
    print('PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
