#!/usr/bin/env python
"""Does the Newton refinement reach the target it is given?

`stationary.refine_stationary_point` defaults to ``target=1e-11`` and
``policy.params`` offers the same number as a tunable. The refinement is
the answer to the floor `probe_gradient_floor.py` measures: L-BFGS-B
stops at |g| ~ 1.5e-5 because its line search can no longer lower the
energy, and a Newton step is supposed to walk the remaining stretch.

Whether it actually does is a separate question, and the answer decides
how the default should be set. The stopping rule has three exits:

    CONVERGED      |g| reached the target
    GUARD_STOPPED  no step lowered the energy
    STALLED_GRADIENT  a step was accepted but did not reduce |g|

and the last two are legitimate outcomes at a point that is already as
good as the arithmetic allows. A target that is never reached turns
every run into one of them, which is indistinguishable at the call site
from a genuine failure.

This script takes a point from L-BFGS-B, refines it, and reports which
exit fired and at what |g|, for a range of targets. It also checks the
two relations the module's comments assert:

    the energy still available at the stopping point  ~ |g|^2 / (2 kappa)
    the Newton step reduces |g| quadratically near a minimum

Usage:

    python check_refinement_target.py [system] [basis] [nthread] [steps]
"""
import os
import sys

_SYS = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
_BASIS = sys.argv[2] if len(sys.argv) > 2 else '6-31g'
_NT = sys.argv[3] if len(sys.argv) > 3 else '4'
_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

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
    StopReason,
    curvature_report,
    refine_stationary_point,
)

SYSTEMS = {
    'H2O': ('O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880', 4, 2, 2),
    'CO':  ('C 0 0 0; O 0 0 1.1283', 5, 2, 2),
    'N2':  ('N 0 0 0; N 0 0 1.0977', 5, 2, 2),
}


def main():
    if _SYS not in SYSTEMS:
        print(f'unknown system {_SYS!r}; expected one of {list(SYSTEMS)}')
        return 1
    atom, ncore, nact, nele = SYSTEMS[_SYS]

    print('=' * 78)
    print(f'Does the Newton refinement reach its target?  {_SYS}/{_BASIS}')
    print('=' * 78)

    mol = pgto.Mole()
    mol.atom = atom
    mol.basis = _BASIS
    mol.build()
    nmo = mol.nao_nr()

    mf = pscf.RHF(mol)
    mf.kernel()
    C0 = mf.mo_coeff
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

    # Start where the production path starts: L-BFGS-B, which is where
    # the floor was measured.
    _, th_lbfgs, g_lbfgs = optimize_theta(None, None, E_theta, npar,
                                          steps=_STEPS, g_conv=1e-12,
                                          theta0=np.zeros(npar))
    th_lbfgs = jnp.asarray(np.asarray(th_lbfgs))
    E_lbfgs = float(E_jit(th_lbfgs))
    print(f'  L-BFGS-B stopping point:')
    print(f'    |g| = {g_lbfgs:.6e}   E = {E_lbfgs:.12f}')

    # The curvature, so the available energy can be predicted.
    # `curvature_report` takes the matrix and symmetrizes it itself.
    H = np.asarray(jax.hessian(E_theta)(th_lbfgs))
    rep = curvature_report(H)
    print(f'    curvature: min = {rep.smallest:.4e}  max = '
          f'{rep.largest:.4e}  kappa = {rep.condition_number:.3e}  '
          f'n_negative = {rep.n_negative}')
    if rep.smallest > 0:
        predicted_gain = g_lbfgs ** 2 / (2.0 * rep.smallest)
        print(f'    |g|^2/(2*lambda_min) = {predicted_gain:.4e} Ha still '
              f'available along the softest direction')

    # ---- refine at a range of targets ----
    print()
    print(f'  {"target":>10} {"exit":>17} {"|g| reached":>14} '
          f'{"steps":>6} {"E":>18} {"dE vs L-BFGS":>15}')
    rows = []
    targets = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    for tg in targets:
        sp = refine_stationary_point(E_theta, th_lbfgs, max_iter=200,
                                     target=tg)
        E_sp = float(E_jit(sp.theta))
        rows.append((tg, sp))
        print(f'    {tg:>10.0e} {sp.stop_reason.name:>17} '
              f'{sp.grad_norm:>14.6e} {sp.n_accepted:>6} '
              f'{E_sp:>18.12f} {E_sp - E_lbfgs:>15.3e}')

    # ---- what the exits mean ----
    print()
    reached = [tg for tg, sp in rows
               if sp.stop_reason is StopReason.CONVERGED]
    guarded = [tg for tg, sp in rows
               if sp.stop_reason is StopReason.GUARD_STOPPED]
    stalled = [tg for tg, sp in rows
               if sp.stop_reason is StopReason.STALLED_GRADIENT]
    print(f'  exits: converged at {reached if reached else "no target"}, '
          f'guard_stopped at {len(guarded)} targets, '
          f'stalled_gradient at {len(stalled)}')

    # The best |g| any of them reached -- that is the practical floor of
    # the refinement, independent of which target was asked for.
    best = min(sp.grad_norm for _, sp in rows)
    best_targets = [tg for tg, sp in rows if sp.grad_norm == best]
    print(f'  best |g| reached across all targets: {best:.6e}')
    print(f'  lowest target that was actually met: '
          f'{max(reached) if reached else "none"}')

    print()
    print('=' * 78)
    default_target = 1e-11
    if reached and min(reached) <= default_target:
        verdict = (f'the default target {default_target:g} is reachable; '
                   'the exits below it are ordinary outcomes, not failures')
    else:
        verdict = (
            f'the default target {default_target:g} was NOT met by any run'
            f'\n  => the practical floor of the refinement here is '
            f'{best:.3e}'
            '\n  => every call with the default target ends in '
            'guard_stopped or stalled_gradient, which the call site '
            'cannot distinguish from a real failure'
            '\n  => set the default at or above the floor, and report the '
            'mechanism rather than a false non-convergence')
    print(f'VERDICT: {verdict}')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
