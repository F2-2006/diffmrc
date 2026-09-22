#!/usr/bin/env python
"""Is the orbital gradient stuck on a plateau, or is it just noise?

Two independent scripts in this project stop at |g| ~ 1.5e-5 while
asking for 1e-12: the implicit-Hessian expansion iteration reaches
1.57e-05, and `verify_rcond_choice.py` stops at 1.547e-05 on H2O. Both
are *bit-identical* when given four times as many steps, which rules out
"needs more iterations" as the explanation.

That leaves two possibilities, and they call for opposite responses:

  A. the point is a genuine minimum along the directions the optimizer
     can move in -- the gradient is fine, the step simply cannot proceed
     because of the line search's own limits. Then |g| is a real
     residual and the convergence target must be set above it.

  B. the computed gradient is noise at the 1e-5 level. Then |g| measures
     the noise, no target below it is reachable, and -- more seriously --
     every convergence test in the project that uses a gradient norm
     below ~1e-5 has been testing the noise, not the physics.

The test that separates them is whether the reported gradient agrees
with the energy it claims to differentiate. Along the normalized -g,

    E(a) - E0 = -a*|g| + (C/2)*a^2

so the slope of the measured curve must come out as -|g| and the
curvature must be positive. A gradient that is noise will not reproduce
its own magnitude in the slope of the energy.

Note that the two terms cross near a = 2|g|/C, which for |g| = 1.5e-5
and C ~ 80 is a = 4e-7. Every step larger than that *raises* the energy
because it overshoots the minimum -- so "the energy went up" on its own
is not evidence of a bad gradient, and the slope has to be fitted rather
than read off a single step.

The energies involved run down to 1e-11 Hartree, so they must be
compared at full double precision, with no re-convergence of the SCF
between evaluations.

Usage:

    python probe_gradient_floor.py [system] [basis] [nthread] [steps]

Writes nothing; prints the measurement and a verdict.
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

    print('=' * 76)
    print(f'Orbital gradient: real residual or noise floor?  {_SYS}/{_BASIS}')
    print('=' * 76)

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

    _, th_star, gnorm = optimize_theta(None, None, E_theta, npar,
                                       steps=_STEPS, g_conv=1e-12,
                                       theta0=np.zeros(npar))
    th_star = jnp.asarray(np.asarray(th_star))
    g = g_jit(th_star)
    gnorm = float(jnp.linalg.norm(g))
    E0 = float(E_jit(th_star))
    print(f'  theta*: |g| = {gnorm:.6e}   E = {E0:.12f}   '
          f'npar = {npar}   steps = {_STEPS}')

    # ---- 1. is the gradient reproducible? ----
    # A deterministic function returns the same value every time, so this
    # rules out only the crudest form of nondeterminism. It is worth
    # printing because "noise" in the sense used here is not run-to-run
    # jitter but the disagreement between the gradient and the finite
    # difference of the energy it claims to differentiate.
    reps = [np.asarray(g_jit(th_star)) for _ in range(3)]
    spread = max(float(np.abs(r - reps[0]).max()) for r in reps)
    print(f'  gradient across 3 evaluations: max spread = {spread:.3e}')
    if spread == 0.0:
        print('    => bit-identical; the pipeline is deterministic')

    # ---- 2. the energy along -ghat ----
    # Along the *normalized* direction the expansion is
    #     E(a) = E0 - a*|g| + (C/2)*a^2
    # so the linear term is |g| itself, not |g|^2. With |g| = 1.5e-5 and
    # a curvature of order 10 the two terms cross near a ~ 1e-6, and a
    # step larger than 2|g|/C overshoots the minimum and *raises* the
    # energy. That would look like a bad gradient if the linear term were
    # not in the model, so both terms are fitted from the data rather
    # than assumed.
    gdir = g / (jnp.linalg.norm(g) + 1e-300)
    steps = [10.0 ** e for e in (0, -2, -3, -4, -5, -6)]
    energies = [float(E_jit(th_star - a * gdir)) for a in steps]
    print()
    print('  moving along the normalized -ghat:')
    print(f'    linear term predicts -a*|g| with |g| = {gnorm:.6e}')
    print(f'    {"a":>10} {"E - E0":>18} {"-a*|g|":>14} '
          f'{"(E-E0) - (-a*|g|)":>20}')
    for a, Ea in zip(steps, energies):
        dE = Ea - E0
        lin = -a * gnorm
        print(f'    {a:>10.0e} {dE:>18.6e} {lin:>14.6e} {dE - lin:>20.6e}')

    # ---- 3. fit the curvature, then test the model it implies ----
    # The curvature is fitted and |g| is *measured*, not fitted: a free
    # two-parameter fit is dominated by the quadratic term at these step
    # sizes and returns a slope that is a small difference of large
    # numbers, so it is not a sensitive test of the gradient. Holding
    # |g| at its measured value and fitting only C makes the residual a
    # genuine test of whether the gradient explains the energy.
    fit_a = np.array([1e-2, 1e-3, 1e-4])
    fit_dE = np.array([energies[1] - E0, energies[2] - E0,
                       energies[3] - E0])
    # E - E0 + a*|g| = (C/2) a^2  =>  C = 2*sum(dE' * a^2) / sum(a^4)
    lhs = fit_dE + fit_a * gnorm
    C = 2.0 * float(np.sum(lhs * fit_a ** 2) / np.sum(fit_a ** 4))
    print()
    print(f'  curvature fitted with |g| held at its measured value:')
    print(f'    C = {C:.4f}')
    print(f'    intercept check: E - E0 + a*|g| should scale as a^2')
    for a, Ea in zip(fit_a, fit_dE):
        model = -a * gnorm + 0.5 * C * a ** 2
        rel = (Ea - model) / model if model else float('nan')
        print(f'      a = {a:>9.0e}: measured {Ea:>13.6e}  '
              f'model {model:>13.6e}  relative {rel:>10.2e}')

    # The strongest single test: predict the smallest step from the
    # parameters and compare. There the two terms are comparable in size,
    # so neither can hide an error in the other.
    a_small, E_small = steps[-1], energies[-1] - E0
    pred_small = -a_small * gnorm + 0.5 * C * a_small ** 2
    rel_small = abs(E_small - pred_small) / abs(pred_small)

    print()
    print('=' * 76)
    if gnorm == 0.0:
        verdict = 'the gradient is exactly zero; nothing to decide'
    elif C > 0 and rel_small < 0.05:
        astar = gnorm / C
        drop = gnorm ** 2 / (2 * C)
        verdict = (
            f'|g| held at {gnorm:.6e} and a positive curvature C = '
            f'{C:.3f} reproduce the energy at every step, including the '
            f'smallest, to {rel_small:.1%}'
            '\n  => CASE A: the gradient is a real residual, not noise.'
            f'\n  => the minimum lies {astar:.4e} away along -ghat, with '
            f'{drop:.3e} Ha still to gain.'
            '\n  => any target below ~|g| is unreachable without the '
            'optimizer taking that last step; set the target above the '
            'residual and say so, rather than reporting a false failure.')
    else:
        verdict = (
            f'|g| = {gnorm:.6e} with curvature {C:.4f} fails to reproduce '
            f'the measured energies (smallest step off by {rel_small:.1%})'
            '\n  => the reported gradient disagrees with the energy it '
            'differentiates: CASE B, a noise floor.')
    print(f'VERDICT: {verdict}')
    print('=' * 76)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
