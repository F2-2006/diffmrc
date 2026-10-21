#!/usr/bin/env python
"""Performance options in diffmrc: CI-solver routing, MO-transform order,
and the jit reference-array form.

The script has four independent parts:

  1. ``resolve_ci_solver`` turns ``ci_solver='auto'`` into ``'plain'`` or
     ``'blocked'`` from the determinant count.  The two anchors printed here
     are 924 determinants (CAS(6,6)) and 12870 determinants (CAS(8,8)).
  2. ``stepwise_mo_ints`` against the one-shot four-operand contraction on a
     small random tensor.  The two forms agree to round-off; the production
     difference is in what XLA does at nao = 96 (see the Manual), and the
     timings printed here are for a small array only.
  3. ``canonicalize_jax`` on H2O/6-31G CAS(2,2) with
     ``ci_solver='auto'``, ``'plain'``, and ``'blocked'``: orbitals, orbital
     energies, and the CAS energy must agree; the CI vector may differ by a
     global phase and by the plain solver's inverse-iteration residual.
  4. The jit reference-array form: one executable serves several arrays,
     whereas a closure over the arrays compiles once per array.  The compile
     counts are the point; the wall-clock difference at this size is small.

Run from the repository root with the package importable:

    PYTHONPATH=src python examples/performance_options.py

The environment variables below keep the timings reproducible on a shared
machine; they are not required for correctness.
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import time
from math import comb

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp

from diffmrc.core.casci import cs_casci
from diffmrc.core.nevpt2.impl import (BLOCKED_CI_MIN_DET,
                                      canonicalize_jax,
                                      resolve_ci_solver,
                                      stepwise_mo_ints)


def banner(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


def timed(fn, *args, repeat=5):
    """Best wall clock of a jitted callable after one warm-up call.

    The best of several runs is less sensitive to load spikes than a
    mean or a median on a shared machine.
    """
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(min(ts)), out


def part_routing():
    banner('1. ci_solver routing by determinant count')
    print('BLOCKED_CI_MIN_DET = {}'.format(BLOCKED_CI_MIN_DET))
    for nact, nele in ((6, 6), (8, 8)):
        ndet = comb(2 * nact, nele)
        auto = resolve_ci_solver('auto', nact, nele)
        print('CAS({0},{0})  ndet={1:6d}  auto -> {2:8s}'
              .format(nact, ndet, auto))
    print('explicit request wins: plain -> {}, blocked -> {}'.format(
        resolve_ci_solver('plain', 8, 8),
        resolve_ci_solver('blocked', 6, 6)))
    print('ndet = 924 is below the threshold, 12870 above it.')


def part_stepwise():
    banner('2. stepwise MO transform vs one-shot four-operand einsum')
    nao = 10
    key = jax.random.PRNGKey(7)
    key, kc, kh, ke = jax.random.split(key, 4)
    # A random orthogonal orbital rotation; the overlap structure of a real
    # coefficient matrix is not needed to compare two algebraic orderings.
    C, _ = jnp.linalg.qr(jax.random.normal(kc, (nao, nao)))
    h = jax.random.normal(kh, (nao, nao))
    eri = jax.random.normal(ke, (nao, nao, nao, nao))

    def one_shot(Cx, h_, e_):
        h1 = jnp.einsum('ai,ab,bj->ij', Cx, h_, Cx)
        eri_mo = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cx, Cx, Cx, Cx, e_)
        return h1, eri_mo

    one_jit = jax.jit(one_shot)
    step_jit = jax.jit(stepwise_mo_ints)
    t_one, (h1_one, eri_one) = timed(one_jit, C, h, eri)
    t_step, (h1_step, eri_step) = timed(step_jit, C, h, eri)

    print('array: nao = {}, random C/h/eri'.format(nao))
    print('max |d h1|  = {:.3e}'.format(
        float(jnp.max(jnp.abs(h1_one - h1_step)))))
    print('max |d eri| = {:.3e}'.format(
        float(jnp.max(jnp.abs(eri_one - eri_step)))))
    print('one-shot  {:.3e} s   stepwise {:.3e} s'
          .format(t_one, t_step))
    print('At nao = 10 both run in microseconds; the ordering is not a')
    print('small-array speedup.  At nao = 96 on physical AO integrals it')
    print('changes both the runtime and the numerical branch; see the')
    print('Manual chapter.')


def part_ci_solver():
    banner("3. ci_solver on H2O/6-31G CAS(2,2)")

    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0; H 0.7572 0 0.5859; H -0.7572 0 0.5859',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    C0 = jnp.asarray(mf.mo_coeff)
    h1e = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    eri = jnp.asarray(mol.intor('int2e', aosym='s1'))
    ncore, nact, nele = 0, 2, 2
    nm = ncore + nact

    h1m, e1m = stepwise_mo_ints(C0, h1e, eri)
    e_cas0, ci0 = cs_casci(h1m[:nm, :nm], e1m[:nm, :nm, :nm, :nm],
                           ncore=ncore, nact=nact, nele_act=nele,
                           return_ci=True)
    print('initial CAS energy = {:.15f} Eh'.format(float(e_cas0)))

    results = {}
    for solver in ('auto', 'plain', 'blocked'):
        t0 = time.perf_counter()
        C2, ci2, eps = canonicalize_jax(C0, h1e, eri, ci0,
                                        ncore=ncore, nact=nact,
                                        nele_act=nele, ci_solver=solver)
        h1c, e1c = stepwise_mo_ints(C2, h1e, eri)
        e_cas = cs_casci(h1c[:nm, :nm], e1c[:nm, :nm, :nm, :nm],
                         ncore=ncore, nact=nact, nele_act=nele)
        dt = time.perf_counter() - t0
        results[solver] = (np.asarray(C2), np.asarray(eps),
                           np.asarray(ci2), float(e_cas), dt)
        print('{0:8s} E_CAS = {1:.15f} Eh   first-call wall = {2:.2f} s'
              .format(solver, float(e_cas), dt))

    C_ref, eps_ref, ci_ref, e_ref, _ = results['plain']
    for solver in ('auto', 'blocked'):
        C2, eps, ci, e_cas, _ = results[solver]
        dC = np.max(np.abs(C2 - C_ref))
        deps = np.max(np.abs(eps - eps_ref))
        dE = abs(e_cas - e_ref)
        # The CI vector is defined up to a global phase; align it before
        # taking the difference.  The blocked solver returns the exact
        # eigenvector, the plain solver an inverse-iteration vector with a
        # residual of order 1e-8.
        sign = 1.0 if np.dot(ci, ci_ref) >= 0 else -1.0
        dci = np.max(np.abs(sign * ci - ci_ref))
        print('{0:8s} vs plain: max|dC| = {1:.1e}  max|deps| = {2:.1e}'
              '  |dE_CAS| = {3:.1e}  max|dci| = {4:.1e}'
              .format(solver, dC, deps, dE, dci))

    print('Orbitals and orbital energies are the intended invariants; the')
    print('CI vector differs by phase and by the plain solver residual.')


def part_jit_arrays():
    banner('4. jit reference-array form vs closure constants')
    nao = 16
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 4)
    Cs = []
    for k in keys[:3]:
        C, _ = jnp.linalg.qr(jax.random.normal(k, (nao, nao)))
        Cs.append(C)
    hs = [jax.random.normal(k, (nao, nao)) for k in jax.random.split(keys[3], 3)]
    es = [jax.random.normal(k, (nao, nao, nao, nao))
          for k in jax.random.split(keys[1], 3)]

    def scalar_from(Cx, h_, e_):
        h1, eri_mo = stepwise_mo_ints(Cx, h_, e_)
        return jnp.sum(h1 * h1) + jnp.sum(eri_mo * eri_mo)

    def closure_form(C0_):
        def f(h_, e_):
            return scalar_from(C0_, h_, e_)
        return f

    # One executable, the arrays are traced arguments.
    arg_jit = jax.jit(scalar_from)
    # One executable per reference array.
    closure_jits = [jax.jit(closure_form(C)) for C in Cs]

    t0 = time.perf_counter()
    vals_arg = [arg_jit(C, h, e) for C, h, e in zip(Cs, hs, es)]
    jax.block_until_ready(vals_arg)
    t_arg = time.perf_counter() - t0

    t0 = time.perf_counter()
    vals_clo = [f(h, e) for f, h, e in zip(closure_jits, hs, es)]
    jax.block_until_ready(vals_clo)
    t_clo = time.perf_counter() - t0

    n_arg = arg_jit._cache_size()
    n_clo = sum(f._cache_size() for f in closure_jits)
    dmax = float(max(abs(float(a - b))
                     for a, b in zip(vals_arg, vals_clo)))
    print('three reference arrays, one theta-independent scalar each')
    print('array-argument form: jit objects = 1, compilations = {}'
          .format(n_arg))
    print('closure form:        jit objects = 3, compilations = {}'
          .format(n_clo))
    print('total first-call wall: array form {:.3f} s, closure form {:.3f} s'
          .format(t_arg, t_clo))
    print('max |value difference| = {:.3e}'.format(dmax))
    print('At production size the compile is the cost that matters: one')
    print('executable serves every geometry whose arrays share a shape.')


def main():
    part_routing()
    part_stepwise()
    part_ci_solver()
    part_jit_arrays()
    print()
    print('performance_options.py finished')


if __name__ == '__main__':
    main()
