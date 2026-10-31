#!/usr/bin/env python
"""Check the RIJONX SCF route as the reference generator for SOMF.

The SOMF operator itself has no RI variant: the two-electron kernel is
contracted four-center.  What can be accelerated is the SCF stage that
produces the starting orbitals.  This probe:

1. runs the exact ROHF and ``driver.ri_scf_driver.rijonx_scf`` (RI-J
   Coulomb, exact exchange) on OH and OF;
2. starts an exact pyscf CASSCF from the RIJONX orbitals and compares the
   CASSCF energy and the JAX SOMF splitting with the exact-ROHF-started
   run;
3. reports timings and the auxiliary-basis size.

Run on 101:
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python probe_rijonx_reference.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import sys
import time

import numpy as np
from pyscf import gto, mcscf, scf

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from diffmrc.core import somf as dsomf
from diffmrc.core import somf_jax as jsomf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_oh_of as ref  # noqa: E402

MEMORY_GUARD = 20 * 2 ** 30


def cas_data(mol, mo_coeff, mo_occ, spec, nstates=2):
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.mo_coeff = np.asarray(mo_coeff)
    mf.mo_occ = np.asarray(mo_occ)
    mf.converged = True
    mc = mcscf.CASSCF(mf, spec['nact'], mol.nelectron - 2 * spec['ncore'])
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([1.0 / nstates] * nstates)
    mc.fcisolver.nroots = nstates
    mc.kernel()
    es = np.asarray(mc.fcisolver.e_states)[:nstates]
    c = np.asarray(mc.mo_coeff)
    act = slice(spec['ncore'], spec['ncore'] + spec['nact'])
    ca = c[:, act]
    dm = np.asarray(mc.make_rdm1())
    d = np.zeros((nstates, nstates, spec['nact'], spec['nact']))
    for i in range(nstates):
        for j in range(nstates):
            d[i, j] = np.asarray(mc.fcisolver.trans_rdm1(
                mc.ci[i], mc.ci[j], spec['nact'], mc.nelecas))
    return dict(mc=mc, es=es, ca=ca, dm=dm, d=d)


def jax_splitting(mol, data):
    ints = jsomf.somf_integrals(mol, memory_limit=MEMORY_GUARD)
    return float(jsomf.somf_splitting_jax(
        jnp.asarray(data['dm']), ints.h1, ints.g,
        jnp.asarray(data['ca']), jnp.asarray(data['es']),
        jnp.asarray(data['d'])))


def run(name, spec):
    print(f'===== {name} =====')
    mol = gto.M(atom=spec['atom'], basis='6-31g', unit='Bohr',
                spin=spec['spin'], verbose=0)

    t0 = time.time()
    mf_exact = scf.ROHF(mol)
    mf_exact.conv_tol = 1e-12
    mf_exact.verbose = 0
    mf_exact.kernel()
    t_exact = time.time() - t0
    print('exact ROHF      E = %.12f  conv = %s  t = %.2f s'
          % (mf_exact.e_tot, mf_exact.converged, t_exact))

    t0 = time.time()
    from pyscfad import gto as adgto
    from diffmrc.driver.ri_scf_driver import RIJONX_ROHF
    # pyscfad's SCF needs a pyscfad Mole.  The multireference stage below
    # uses the equivalent plain pyscf Mole and the converged orbitals.
    mol_ad = adgto.Mole(atom=spec['atom'], basis='6-31g', unit='Bohr',
                     spin=spec['spin'], verbose=0)
    # The pyscfad kernel reads conv_tol from the object; passing it through
    # rijonx_scf(..., conv_tol=...) duplicates the positional argument in
    # this pyscfad version.
    mf_ri = RIJONX_ROHF(mol_ad)
    mf_ri.conv_tol = 1e-12
    mf_ri.verbose = 0
    mf_ri.kernel()
    t_ri = time.time() - t0
    naux = mf_ri.ri_state().auxmol.nao_nr()
    print('RIJONX ROHF     E = %.12f  conv = %s  t = %.2f s  naux = %d'
          % (float(mf_ri.e_tot), mf_ri.converged, t_ri, naux))
    print('E(RIJONX) - E(exact) = %.3e Eh'
          % (float(mf_ri.e_tot) - float(mf_exact.e_tot)))

    d_exact = cas_data(mol, mf_exact.mo_coeff, mf_exact.mo_occ, spec)
    d_ri = cas_data(mol, mf_ri.mo_coeff, mf_ri.mo_occ, spec)
    print('CASSCF exact-start E = %.12f' % d_exact['mc'].e_tot)
    print('CASSCF RIJONX-start E = %.12f  dE = %.3e Eh'
          % (d_ri['mc'].e_tot, d_ri['mc'].e_tot - d_exact['mc'].e_tot))
    s_exact = jax_splitting(mol, d_exact)
    s_ri = jax_splitting(mol, d_ri)
    print('SOMF exact-start  = %.10f cm^-1' % s_exact)
    print('SOMF RIJONX-start = %.10f cm^-1  diff = %.3e cm^-1'
          % (s_ri, s_ri - s_exact))
    return dict(name=name, e_exact=float(mf_exact.e_tot),
                e_rijonx=float(mf_ri.e_tot), naux=int(naux),
                t_exact=t_exact, t_rijonx=t_ri,
                cas_exact=float(d_exact['mc'].e_tot),
                cas_rijonx=float(d_ri['mc'].e_tot),
                somf_exact_cm=s_exact, somf_rijonx_cm=s_ri)


if __name__ == '__main__':
    out = {}
    for name in ('oh', 'of'):
        out[name] = run(name, ref.SYSTEMS[name])
    print(out)
