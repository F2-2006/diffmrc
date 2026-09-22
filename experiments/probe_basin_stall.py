#!/usr/bin/env python
"""Why does the theta-space optimization stall at |g| ~ 4e-4 in the lower basin?

probe_h2o_lower_basin established that the lower basin (E = -76.0060407,
confirmed by PySCF's CASSCF from a swapped ordering) is real. But the
theta-space optimization in that basin stalls at |g| ~ 4e-4 in two
independent runs, whereas PySCF converges there. Three candidates:

  (a) the parametrization cannot represent that point;
  (b) a precision floor: near a stiff point the energy gain per step
      falls below double precision, so both L-BFGS-B (ftol) and the
      refinement (stall rule) stop;
  (c) the refinement's stop rule is too strict in that regime.

Test (a) directly: take PySCF's converged orbitals for the lower basin
and evaluate OUR energy function at them. If the energies agree, the
parametrization's image contains the basin and (a) is refuted. The
Hessian spectrum at the stall point separates (b) from (c): a stiff
spectrum (large smallest eigenvalue) means the available energy at
|g| = 4e-4 is below double precision.

Usage:
    python probe_basin_stall.py [system] [basis] [nthread]
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
    print(f'Why does the lower-basin optimization stall at |g| ~ 4e-4?  '
          f'{_SYS}/{_BASIS}')
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

    # ---- the stall point: random start, optimize with a tight target ----
    rng = np.random.default_rng(1000)
    t0 = 0.3 * rng.standard_normal(npar)
    _, th_opt, g_opt = optimize_theta(None, None, E_theta, npar,
                                      steps=3000, g_conv=1e-8, theta0=t0)
    th_opt = jnp.asarray(np.asarray(th_opt))
    E_opt = float(E_jit(th_opt))
    g_opt = float(jnp.linalg.norm(g_jit(th_opt)))
    print()
    print(f'  after optimizer alone : E = {E_opt:.12f}  |g| = {g_opt:.3e}')

    sp = refine_stationary_point(E_theta, th_opt, max_iter=200, target=1e-11)
    th_ref = jnp.asarray(np.asarray(sp.theta))
    E_ref = float(E_jit(th_ref))
    print(f'  after refinement      : E = {E_ref:.12f}  |g| = '
          f'{sp.grad_norm:.3e}  {sp.stop_reason.name}  '
          f'steps accepted = {sp.n_accepted}')

    # ---- spectrum at the stall point ----
    H = np.asarray(h_jit(th_ref))
    H = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(H)
    rep = curvature_report(H)
    print()
    print(f'  Hessian at stall point: min = {w[0]:+.4e}  max = {w[-1]:+.4e}  '
          f'kappa = {rep.condition_number:.3e}  n_negative = {rep.n_negative}')
    print(f'  smallest 5 eigenvalues: '
          f'{", ".join(f"{x:+.3e}" for x in w[:5])}')
    # Energy still available along the softest direction, by the standard
    # estimate. If this is below double precision at this energy scale,
    # every finite-difference-based stopper (L-BFGS-B ftol) fires here.
    if w[0] > 0:
        print(f'  |g|^2/(2*lambda_min) = {sp.grad_norm ** 2 / (2 * w[0]):.3e} Ha '
              f'available')
    print(f'  double-precision resolution at E = {E_ref:.1f}: '
          f'{abs(E_ref) * 2.2e-16:.3e} Ha')

    # ---- (a) can the parametrization reach PySCF's lower solution? ----
    print()
    print('  --- PySCF lower solution, fed to our energy function ---')
    from pyscf import gto as gto2, scf as scf2, mcscf
    mol2 = gto2.M(atom=atom, basis=_BASIS, verbose=0)
    mf2 = scf2.RHF(mol2)
    mf2.kernel()
    mo = np.array(mf2.mo_coeff)
    mo[:, [3, 6]] = mo[:, [6, 3]]
    mc = mcscf.CASSCF(mf2, nact, nele)
    mc.verbose = 0
    E_pyscf = float(mc.kernel(mo)[0])
    C_pyscf = np.asarray(mc.mo_coeff)
    # E_theta takes parameters; here we already have orbitals, so the
    # orbital-input energy function is the right one to call.
    E_ours_at_pyscf = float(E_cas(jnp.asarray(C_pyscf)))
    print(f'    PySCF   E(swapped order)  = {E_pyscf:.12f}  conv={bool(mc.converged)}')
    print(f'    ours at PySCF orbitals    = {E_ours_at_pyscf:.12f}')
    print(f'    difference               = {E_ours_at_pyscf - E_pyscf:+.3e} Ha')
    print(f'    our theta-space minimum  = {E_ref:.12f}')
    print(f'    theta-min vs PySCF orbitals = {E_ref - E_ours_at_pyscf:+.3e} Ha')

    print()
    print('=' * 78)
    print('  Reading:')
    print('    difference ~ 0  => the parametrization reaches the basin;')
    print('                       candidate (a) is refuted and the stall is')
    print('                       a precision/stopping phenomenon')
    print('    lam_min large   => the energy available at |g| ~ 4e-4 is near')
    print('                       the double-precision floor')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
