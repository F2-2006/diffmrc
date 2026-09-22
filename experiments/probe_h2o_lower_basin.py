#!/usr/bin/env python
"""Follow-up to probe_saddle_refined: is the lower H2O basin real?

probe_saddle_refined found:

    theta = 0  -> L-BFGS-B -> refine -> stationary saddle  E = -75.984881801724
    perturbed  -> L-BFGS-B -> refine -> lower points       E = -76.006040  (~13 kcal/mol lower)
                                        (three starts agree to ~1e-6 Ha,
                                         refinement stalled at |g| ~ 1e-4)

A 13 kcal/mol spread between two "CAS(2,2)/6-31g" solutions of water is
either a genuine feature of the theta-space landscape or a defect in the
energy function the experiments use. The variational criterion cannot
decide between them alone, so this script brings in two independent
checks:

  1. PySCF's own CASSCF(2,2), from the RHF orbitals and from a swapped
     ordering. Same ansatz, independent implementation: whichever energy
     it converges to is attainable.
  2. Orbital identity at each theta point: which RHF orbital each rotated
     orbital descends from. This shows what the lower basin actually is.

  3. Whether the lower point is stationary: re-run L-BFGS-B from it with
     a tight target and report the gradient it reaches.

Usage:
    python probe_h2o_lower_basin.py [system] [basis] [nthread]
"""
import os
import sys

_SYS = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
_BASIS = sys.argv[2] if len(sys.argv) > 2 else '6-31g'
_NT = sys.argv[3] if len(sys.argv) > 3 else '4'

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = _NT
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp                                    # noqa: E402

from pyscfad import gto as pgto, scf as pscf                    # noqa: E402

from diffmrc.core.rotation import block_rotation                # noqa: E402
from diffmrc.core.orblib import orbital_pairs                   # noqa: E402
from diffmrc.core.casci import cs_casci_blocked                 # noqa: E402
from diffmrc.core.optimizer import optimize_theta               # noqa: E402
from diffmrc.core.stationary import (                           # noqa: E402
    curvature_report,
    refine_stationary_point,
)

SYSTEMS = {
    'H2O': ('O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880', 4, 2, 2),
    'CO':  ('C 0 0 0; O 0 0 1.1283', 5, 2, 2),
    'N2':  ('N 0 0 0; N 0 0 1.0977', 5, 2, 2),
}


def orbital_labels(C, C0, S, ncore, nact):
    """Which reference orbital each rotated orbital descends from."""
    W = C.T @ S @ C0
    out = []
    for j in range(C.shape[1]):
        i = int(np.argmax(np.abs(W[:, j])))
        block = ('core' if j < ncore else
                 'act' if j < ncore + nact else 'virt')
        out.append((j, block, i, float(W[i, j])))
    return out


def main():
    if _SYS not in SYSTEMS:
        print(f'unknown system {_SYS!r}')
        return 1
    atom, ncore, nact, nele = SYSTEMS[_SYS]

    print('=' * 78)
    print(f'Is the lower basin real?  {_SYS}/{_BASIS}')
    print('=' * 78)

    mol = pgto.Mole()
    mol.atom = atom
    mol.basis = _BASIS
    mol.build()
    nmo = mol.nao_nr()

    mf = pscf.RHF(mol)
    mf.kernel()
    C0 = np.asarray(mf.mo_coeff)
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    S = mol.intor('int1e_ovlp')
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    npar = len(pairs)

    def E_cas(C):
        Cj = jnp.asarray(C)
        nm = ncore + nact
        h1 = jnp.einsum('ai,ab,bj->ij', Cj, h1e, Cj)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                        eri)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + enuc

    E_theta = lambda t: E_cas(C0 @ block_rotation(t, nmo, pairs))
    E_jit = jax.jit(E_theta)
    g_jit = jax.jit(jax.grad(E_theta))
    h_jit = jax.jit(jax.hessian(E_theta))

    # ---------------- 1. independent implementation ----------------
    print()
    print('  --- PySCF CASSCF({}, {}) reference (same ansatz, '
          'independent code) ---'.format(nele, nact))
    from pyscf import gto as gto2, scf as scf2, mcscf
    mol2 = gto2.M(atom=atom, basis=_BASIS, verbose=0)
    mf2 = scf2.RHF(mol2)
    mf2.kernel()
    for tag, swap in (('rhf order', None), ('swap orbitals 3<->6', (3, 6))):
        mo = np.array(mf2.mo_coeff)
        if swap:
            mo[:, [swap[0], swap[1]]] = mo[:, [swap[1], swap[0]]]
        mc = mcscf.CASSCF(mf2, nact, nele)
        mc.verbose = 0
        E_ref = float(mc.kernel(mo)[0])
        print(f'    {tag:>20}: E = {E_ref:.12f}  conv={bool(mc.converged)}')

    # ---------------- 2. theta-space points ----------------
    def point(t0, steps, gconv, tag):
        _, th, _ = optimize_theta(None, None, E_theta, npar, steps=steps,
                                  g_conv=gconv, theta0=t0)
        sp = refine_stationary_point(E_theta, th, max_iter=200,
                                     target=1e-11)
        th = jnp.asarray(np.asarray(sp.theta))
        E = float(E_jit(th))
        g = float(jnp.linalg.norm(g_jit(th)))
        rep = curvature_report(0.5 * (np.asarray(h_jit(th))
                                      + np.asarray(h_jit(th)).T))
        print(f'    {tag:>24}: E = {E:.12f}  |g| = {g:.2e}  '
              f'{sp.stop_reason.name}  n_neg = {rep.n_negative}')
        return th, E, g, sp

    print()
    print('  --- theta-space points ---')
    th_a, E_a, g_a, sp_a = point(np.zeros(npar), 2000, 1e-12, 'zeros')
    rng = np.random.default_rng(1000)
    th_b, E_b, g_b, sp_b = point(0.3 * rng.standard_normal(npar), 800,
                                 1e-6, 'rand0(0.3)')

    # is the lower point stationary?  re-optimize from it, tight target
    print()
    print('  --- re-optimize from the lower point (tight target) ---')
    _, th_b2, _ = optimize_theta(None, None, E_theta, npar, steps=3000,
                                 g_conv=1e-10, theta0=th_b)
    sp_b2 = refine_stationary_point(E_theta, th_b2, max_iter=200,
                                    target=1e-11)
    th_b2 = jnp.asarray(np.asarray(sp_b2.theta))
    E_b2 = float(E_jit(th_b2))
    g_b2 = float(jnp.linalg.norm(g_jit(th_b2)))
    rep_b2 = curvature_report(0.5 * (np.asarray(h_jit(th_b2))
                                     + np.asarray(h_jit(th_b2)).T))
    print(f'    from lower point: E = {E_b2:.12f}  |g| = {g_b2:.2e}  '
          f'{sp_b2.stop_reason.name}  n_neg = {rep_b2.n_negative}')

    # ---------------- 3. what the orbitals are ----------------
    print()
    print('  --- active/core orbital identity (index <- reference orbital) ---')
    for tag, th in (('zeros (saddle)', th_a), ('rand0 (lower)', th_b2)):
        C = np.asarray(C0 @ block_rotation(th, nmo, pairs))
        labs = orbital_labels(C, C0, S, ncore, nact)
        core_l = ' '.join(f'{j}<-{i}({w:+.2f})' for j, b, i, w in labs
                          if b == 'core')
        act_l = ' '.join(f'{j}<-{i}({w:+.2f})' for j, b, i, w in labs
                         if b == 'act')
        print(f'    {tag}:')
        print(f'      core: {core_l}')
        print(f'      act : {act_l}')

    print()
    print('=' * 78)
    print('  Interpretation guide:')
    print('    - if PySCF finds the lower energy too, the basin is real')
    print('    - the "x<-7(0.87)" style labels say which RHF orbital each')
    print('      rotated orbital descends from; the active pair decides')
    print('      the physics')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
