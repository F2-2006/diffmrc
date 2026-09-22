#!/usr/bin/env python
"""At which point does the negative curvature belong?

The P3 report measured the Hessian of the orbital-rotation energy at the
L-BFGS-B stopping point -- a point whose gradient norm is 1.5e-5, not a
stationary point -- and concluded from it that the point is a saddle. The
curvature of a function at a non-stationary point is not the curvature of
the nearby minimum, so the conclusion needs the spectrum at the refined
point to stand.

This script computes the spectrum at both:

    P0  the L-BFGS-B stopping point     |g| ~ 1.5e-5
    P1  the Newton-refined point        |g| ~ 4e-14

and labels the softest directions by the orbital pair (i, j) they rotate,
so a negative eigenvalue can be attributed rather than merely reported.

If P0 has negative curvature and P1 does not, the saddle belonged to the
non-stationary point; the refined point is a minimum, and the sentence
"theta* is a saddle" must be withdrawn. If the negative curvature survives
at P1, the point is a genuine saddle, and the script runs the project's
multi-start protocol: perturbed starts, optimize, refine, and a
variational verdict (casscf-multiple-minima).

Usage:
    python probe_saddle_refined.py [system] [basis] [nthread] [n_restart]
"""
import os
import sys

_SYS = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
_BASIS = sys.argv[2] if len(sys.argv) > 2 else '6-31g'
_NT = sys.argv[3] if len(sys.argv) > 3 else '4'
_NRESTART = int(sys.argv[4]) if len(sys.argv) > 4 else 4

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

NEG_TOL = 1e-8   # the report's threshold for calling an eigenvalue negative


def spectrum(H, pairs, tag):
    """Eigenvalues of the symmetric part, with the softest directions
    attributed to the orbital pairs they rotate."""
    Hs = 0.5 * (np.asarray(H) + np.asarray(H).T)
    w, V = np.linalg.eigh(Hs)
    rep = curvature_report(Hs)
    print(f'  {tag}: min = {w[0]:+.6e}  max = {w[-1]:+.6e}  '
          f'kappa = {rep.condition_number:.3e}  n_negative = {rep.n_negative}')
    for k in range(3):
        v = V[:, k]
        idx = np.argsort(-np.abs(v))[:4]
        lab = '  '.join(f'({pairs[i][0]}<->{pairs[i][1]})={v[i]:+.3f}'
                        for i in idx)
        print(f'      eig[{k}] = {w[k]:+.6e}   top weight: {lab}')
    for k in range(1, 4):
        v = V[:, -k]
        idx = np.argsort(-np.abs(v))[:4]
        lab = '  '.join(f'({pairs[i][0]}<->{pairs[i][1]})={v[i]:+.3f}'
                        for i in idx)
        print(f'      eig[-{k}] = {w[-k]:+.6e}   top weight: {lab}')
    return w


def main():
    if _SYS not in SYSTEMS:
        print(f'unknown system {_SYS!r}; expected one of {list(SYSTEMS)}')
        return 1
    atom, ncore, nact, nele = SYSTEMS[_SYS]

    print('=' * 78)
    print(f'Where does the negative curvature belong?  {_SYS}/{_BASIS}')
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
    h_jit = jax.jit(jax.hessian(E_theta))

    # ---------------- P0: L-BFGS-B stopping point ----------------
    print()
    print('  --- P0: L-BFGS-B stopping point (g_conv=1e-12, as the report) ---')
    _, th0, g0 = optimize_theta(None, None, E_theta, npar, steps=2000,
                                g_conv=1e-12, theta0=np.zeros(npar))
    th0 = jnp.asarray(np.asarray(th0))
    E0 = float(E_jit(th0))
    g0 = float(jnp.linalg.norm(g_jit(th0)))
    print(f'    |g| = {g0:.6e}   E = {E0:.12f}')
    w0 = spectrum(np.asarray(h_jit(th0)), pairs, 'P0 curvature')

    # ---------------- P1: Newton-refined point ----------------
    print()
    print('  --- P1: Newton-refined point (target=1e-11) ---')
    sp = refine_stationary_point(E_theta, th0, max_iter=200, target=1e-11)
    th1 = jnp.asarray(np.asarray(sp.theta))
    E1 = float(E_jit(th1))
    print(f'    exit = {sp.stop_reason.name}   |g| = {sp.grad_norm:.6e}   '
          f'steps = {sp.n_accepted}')
    print(f'    E = {E1:.12f}   dE vs P0 = {E1 - E0:+.6e}')
    w1 = spectrum(np.asarray(h_jit(th1)), pairs, 'P1 curvature')

    nn0 = int((w0 < -NEG_TOL).sum())
    nn1 = int((w1 < -NEG_TOL).sum())

    # ---------------- verdict / escalation ----------------
    print()
    print('=' * 78)
    if nn1 == 0 and nn0 > 0:
        print('VERDICT: the negative curvature belonged to P0, the '
              'non-stationary point.')
        print('  At the stationary point P1 the Hessian has no negative '
              'eigenvalue')
        print(f'  (n_negative: P0 = {nn0}, P1 = {nn1}). The refined theta* is '
              'a minimum;')
        print('  "theta* is a saddle" was a statement about a point whose '
              'gradient was 1.5e-5.')
        print('  The P3 report\'s correction of the |g|^2/(2 lambda_min) '
              'estimate stands,')
        print('  because it was about that same stopping point.')
    elif nn1 == 0 and nn0 == 0:
        print('VERDICT: no negative curvature at either point. The saddle '
              'claim does not reproduce.')
    else:
        print(f'VERDICT: negative curvature SURVIVES at the stationary point '
              f'(P1 n_negative = {nn1}).')
        print('  The point is a genuine saddle. Running the multi-start '
              'protocol:')
        print()
        starts = [('zeros', np.zeros(npar))]
        for s in range(_NRESTART):
            rng = np.random.default_rng(1000 + s)
            starts.append((f'rand{s}(0.3)', 0.3 * rng.standard_normal(npar)))
        rows = []
        for name, t0 in starts:
            _, th, _ = optimize_theta(None, None, E_theta, npar, steps=800,
                                      g_conv=1e-6, theta0=t0)
            sp_i = refine_stationary_point(E_theta, th, max_iter=200,
                                           target=1e-11)
            thi = jnp.asarray(np.asarray(sp_i.theta))
            Ei = float(E_jit(thi))
            Hi = np.asarray(h_jit(thi))
            rep_i = curvature_report(0.5 * (Hi + Hi.T))
            rows.append((name, Ei, sp_i.grad_norm, rep_i.n_negative,
                         sp_i.stop_reason.name))
            print(f'    {name:>12}  E = {Ei:.12f}  |g| = {sp_i.grad_norm:.2e}'
                  f'  n_neg = {rep_i.n_negative}  {sp_i.stop_reason.name}')
        best = min(rows, key=lambda r: r[1])
        print()
        print(f'  variational lowest: {best[0]}  E = {best[1]:.12f}  '
              f'n_neg = {best[3]}')
        print(f'  P1 - lowest = {E1 - best[1]:+.6e} Ha')
        if abs(E1 - best[1]) < 1e-10 and best[3] == 0:
            print('  => P1 (zeros start) IS the lowest minimum found; the '
                  'saddle is elsewhere in theta space')
        elif E1 - best[1] > 1e-10:
            print('  => a lower minimum exists; P1 is a saddle sitting above '
                  'it. The production path must start from more than theta=0.')
        else:
            print('  => the lowest point found is itself not a minimum '
                  '(n_neg > 0); widen the search before concluding.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
