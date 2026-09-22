#!/usr/bin/env python
"""Is the low basin's stalled gradient the chart's, or the optimizer's?

In the low basin the two-stage protocol settles at |g| ~ 2e-4 to 4e-4
while the energy matches PySCF's own solution to 9e-8 Ha. Two readings:

  the chart's   the point is a minimum of the theta-chart energy and
                the residual gradient is what double precision can
                represent there;
  the optimizer's  a stationary point exists nearby and the damped
                Newton (step cap 10, stall rule) cannot reach it.

The escape probe walked along the softest eigenvector only, which
proves nothing about other directions. This probe uses a trust-region
method with the exact Hessian: trust-region steps are not capped at 10,
they accept a step only when the quadratic model predicts a decrease,
and they fail loudly instead of stalling silently. If |g| drops to
1e-10, the stall was the optimizer's; if it does not, the chart is the
limit.

Usage:
    python probe_low_basin_gradient.py [system] [basis] [nthread]
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
}


def main():
    if _SYS not in SYSTEMS:
        print(f'unknown system {_SYS!r}')
        return 1
    atom, ncore, nact, nele = SYSTEMS[_SYS]

    print('=' * 78)
    print(f'Chart or optimizer?  {_SYS}/{_BASIS}')
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

    # ---- the two-stage protocol's stall point ----
    rng = np.random.default_rng(1000)
    _, th_opt, _ = optimize_theta(None, None, E_theta, npar, steps=3000,
                                  g_conv=1e-8,
                                  theta0=0.3 * rng.standard_normal(npar))
    sp = refine_stationary_point(E_theta, th_opt, max_iter=200,
                                 target=1e-11)
    th = jnp.asarray(np.asarray(sp.theta))
    print()
    print(f'  two-stage stall point: E = {float(E_jit(th)):.12f}  '
          f'|g| = {sp.grad_norm:.3e}  {sp.stop_reason.name}')

    # ---- trust-region with the exact Hessian, from the same point ----
    import scipy.optimize as sopt

    print()
    print('  trust-region from the stall point (exact Hessian):')
    for method in ('trust-exact', 'trust-krylov', 'trust-ncg'):
        calls = {'n': 0}

        def fun(x):
            calls['n'] += 1
            return float(E_jit(jnp.asarray(x)))

        def jac(x):
            return np.asarray(g_jit(jnp.asarray(x)), dtype=float)

        def hess(x):
            H = np.asarray(h_jit(jnp.asarray(x)), dtype=float)
            return 0.5 * (H + H.T)

        res = sopt.minimize(fun, np.asarray(th, dtype=float), jac=jac,
                            hess=hess, method=method,
                            options=dict(gtol=1e-10, maxiter=200))
        g_new = float(np.linalg.norm(np.asarray(
            g_jit(jnp.asarray(res.x)), dtype=float)))
        rep = curvature_report(0.5 * (hess(res.x) + hess(res.x).T))
        print(f'    {method:>12}: E = {float(res.fun):.12f}  '
              f'|g| = {g_new:.3e}  nit = {res.nit}  '
              f'n_negative = {rep.n_negative}  calls = {calls["n"]}',
              flush=True)
        print(f'                  status {res.status}: {res.message}')

    print()
    print('=' * 78)
    print('  Reading: |g| ~ 1e-10 from any of these => the two-stage')
    print('  protocol stalled on its own step cap or stall rule, and the')
    print('  chart does contain a stationary point at this energy.')
    print('  |g| stuck at 1e-4 for all three => the chart is the limit.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
