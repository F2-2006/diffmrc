#!/usr/bin/env python
"""Can the theta chart be settled at the low basin, or is it flat by nature?

In the low basin the theta-space point stalls at |g| ~ 4e-4 with a
softest Hessian eigenvalue of 1.7e-6. One refinement step raises the
gradient instead of lowering it: the Newton direction along the soft
mode is about 200 units long, the step cap allows 10, and a partial
step lands where the gradient is larger.

So the optimizer and the refinement both stop for reasons of their own.
The question this script answers is whether a stationary point EXISTS
in the chart at that energy: move a long way along the soft eigenvector
- both signs - re-optimize from each landing point, and see whether any
of them settles below |g| ~ 4e-4 at the same energy.

If none do, the chart is flat by nature there and the residual gradient
is the chart's, not the optimizer's. That is a statement worth having
before anyone reads a dtheta*/dR computed in this basin.

Usage:
    python probe_low_basin_escape.py [system] [basis] [nthread]
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
from diffmrc.core.stationary import refine_stationary_point     # noqa: E402

SYSTEMS = {
    'H2O': ('O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880', 4, 2, 2),
}


def main():
    if _SYS not in SYSTEMS:
        print(f'unknown system {_SYS!r}')
        return 1
    atom, ncore, nact, nele = SYSTEMS[_SYS]

    print('=' * 78)
    print(f'Can the theta chart settle in the low basin?  {_SYS}/{_BASIS}')
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

    # ---- the low-basin stall point (same protocol as the diagnostics) ----
    rng = np.random.default_rng(1000)
    _, th_opt, _ = optimize_theta(None, None, E_theta, npar, steps=3000,
                                  g_conv=1e-8,
                                  theta0=0.3 * rng.standard_normal(npar))
    sp = refine_stationary_point(E_theta, th_opt, max_iter=200,
                                 target=1e-11)
    th0 = jnp.asarray(np.asarray(sp.theta))
    E0 = float(E_jit(th0))
    g0 = float(jnp.linalg.norm(g_jit(th0)))
    print()
    print(f'  stall point: E = {E0:.12f}  |g| = {g0:.3e}  '
          f'{sp.stop_reason.name}')

    # ---- the soft eigenvector ----
    H = np.asarray(h_jit(th0))
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    v = V[:, 0]
    print(f'  soft mode: lambda_min = {w[0]:+.4e}  '
          f'lambda_1 = {w[1]:+.4e}  kappa = {w[-1] / max(abs(w[0]), 1e-30):.2e}')

    # ---- walk along it, both signs ----
    print()
    print('  E along theta0 + s * v_min:')
    best = (E0, th0, g0, 0.0)
    for s in (-40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        th_s = jnp.asarray(np.asarray(th0) + s * v)
        e_s = float(E_jit(th_s))
        print(f'    s = {s:+7.1f}   E = {e_s:.12f}   dE = {e_s - E0:+.3e}',
              flush=True)
        if e_s < best[0] - 1e-10:
            best = (e_s, th_s, float(jnp.linalg.norm(g_jit(th_s))), s)

    # ---- re-optimize from the best landing point ----
    print()
    print(f'  lowest along the walk: s = {best[3]:+.1f}  E = {best[0]:.12f}  '
          f'|g| = {best[2]:.3e}')
    if best[3] != 0.0:
        _, th_re, _ = optimize_theta(None, None, E_theta, npar, steps=3000,
                                     g_conv=1e-10, theta0=best[1])
        sp_re = refine_stationary_point(E_theta, th_re, max_iter=200,
                                        target=1e-11)
        th_re = jnp.asarray(np.asarray(sp_re.theta))
        E_re = float(E_jit(th_re))
        g_re = float(jnp.linalg.norm(g_jit(th_re)))
        H_re = np.asarray(h_jit(th_re))
        w_re = np.linalg.eigvalsh(0.5 * (H_re + H_re.T))
        print(f'  re-optimized from there: E = {E_re:.12f}  |g| = {g_re:.3e}  '
              f'{sp_re.stop_reason.name}')
        print(f'  spectrum: min = {w_re[0]:+.3e}  n_negative = '
              f'{int((w_re < -1e-8).sum())}')

    print()
    print('=' * 78)
    print('  If nothing below |g| ~ 4e-4 is reached at this energy, the')
    print('  residual gradient is the chart\'s flatness, not a solver failure;')
    print('  dtheta*/dR in this basin inherits that caveat and must say so.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
