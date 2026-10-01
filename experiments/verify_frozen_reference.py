#!/usr/bin/env python
"""Standalone check that the packaged kernel reproduces the frozen reference.

This is the pytest test in ``tests/core/test_frozen_reference.py`` written as
a plain script, so that it can run in an environment that has the numerical
dependencies but not pytest. Both must agree; if they ever diverge, the test
file is the source of truth.

Usage:

    python verify_frozen_reference.py [path/to/src]

The optional argument prepends a source directory to ``sys.path`` so the
package can be checked before it is installed.
"""
import os
import sys

# The kernel drives JAX through pyscfad, which requires double precision.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

if len(sys.argv) > 1:
    sys.path.insert(0, os.path.abspath(sys.argv[1]))

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from pyscfad import gto as pgto, scf as pscf

from diffmrc.core.rotation import block_rotation
from diffmrc.core.orblib import orbital_pairs
from diffmrc.core.casci import cs_casci_blocked
from diffmrc.core.optimizer import optimize_theta

# H2O / 6-31G, CAS(2,2). Pre-migration value, confirmed by an A/B run
# against the original flat-layout modules. See tests/core/test_frozen_reference.py
# for the provenance and for why this differs from the manuscript value.
H2O_631G_CAS22_E_BASELINE = -75.9848818014
H2O_ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
TOL = 1e-5


def compute_energy():
    mol = pgto.Mole()
    mol.atom = H2O_ATOM
    mol.basis = '6-31g'
    mol.build()
    nmo = mol.nao_nr()

    mf = pscf.RHF(mol)
    mf.kernel()
    C0 = mf.mo_coeff

    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    ncore = mol.nelectron // 2 - 1
    pairs = orbital_pairs(ncore, 2, nmo, include_cv=True)

    def E_cas(C):
        Cj = jnp.asarray(C)
        h1m = jnp.einsum('ai,ab,bj->ij', Cj, h1e, Cj)[:ncore + 2, :ncore + 2]
        e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                         eri)[:ncore + 2, :ncore + 2, :ncore + 2, :ncore + 2]
        return cs_casci_blocked(h1m, e1m, ncore=ncore, nact=2,
                                nele_act=2)[0] + enuc

    E_theta = lambda t: E_cas(C0 @ block_rotation(t, nmo, pairs))
    _, th, gnorm = optimize_theta(None, None, E_theta, len(pairs), steps=2000,
                                  g_conv=1e-12, theta0=np.zeros(len(pairs)))
    return float(E_theta(jnp.asarray(np.asarray(th)))), len(pairs), gnorm


def main():
    print('=' * 68)
    print('Regression baseline check: H2O / 6-31G, CAS(2,2)')
    print('=' * 68)

    E, npar, gnorm = compute_energy()
    dev = abs(E - H2O_631G_CAS22_E_BASELINE)

    print(f'  parameter count       {npar}  (expected 50)')
    print(f'  orbital gradient      {gnorm:.3e}')
    print(f'  computed energy       {E:.10f} Eh')
    print(f'  pre-migration value   {H2O_631G_CAS22_E_BASELINE:.10f} Eh')
    print(f'  deviation             {dev:.3e}  (tolerance {TOL:.0e})')

    ok_energy = dev < TOL
    ok_npar = npar == 50
    ok_conv = gnorm < 1e-4
    print()
    print(f'  energy    {"PASS" if ok_energy else "FAIL"}')
    print(f'  npar      {"PASS" if ok_npar else "FAIL"}')
    print(f'  converged {"PASS" if ok_conv else "FAIL"}')
    print()
    if ok_energy and ok_npar and ok_conv:
        print('RESULT: kernel numerics are unchanged by the migration.')
        return 0
    print('RESULT: the kernel moved. Investigate before trusting the package.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
