"""Orbital-convergence criterion for the NEVPT2 correction.

The CASSCF energy is stationary in theta, so its error is second order in
the residual orbital gradient g while the correction e2 is first order:

    e2(theta*) - e2(theta) ~= s . delta_theta,
    delta_theta = -H^-1 g,   s = d e2/dtheta,   H = d2 E_casscf/dtheta2.

This script measures the relation on a real system by walking the theta
optimizer down a ladder of convergence thresholds, and turns it into an
executable criterion.  It also runs the same optimization from distinct
starting points to measure the branch spread that a convergence
threshold alone cannot see.

Stages (all at one fixed geometry):
  ladder   optimize theta at several g_conv values, record E_cas, e2,
           |g|max and |s|max at each point (one jitted theta function)
  bound    at the tightest point build H, solve H^-1 s, and compare the
           first-order prediction with the measured e2 drift of every
           ladder point
  branch   re-optimize from random starts at one threshold and report the
           spread of E_cas and e2

Usage:
  python nevpt2_convergence.py <system> <out.json> [stages]
  system in {h2o, co}
"""
import json
import os
import sys
import time

import numpy as np
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from pyscfad import gto as pgto, scf as pscf

from diffmrc.core.orblib import orbital_pairs
from diffmrc.driver.nevpt2 import make_nevpt2_theta_fn

SYSTEMS = {
    'h2o': dict(atom='O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
                ncore=4, nact=2, nele=2, basis='6-31g'),
    'co': dict(atom='C 0 0 0; O 0 0 1.1283', ncore=6, nact=2, nele=2,
               basis='6-31g'),
}
THRESHOLDS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


def build(system):
    cfg = SYSTEMS[system]
    mol = pgto.Mole()
    mol.atom = cfg['atom']
    mol.basis = cfg['basis']
    mol.build()
    nmo = mol.nao_nr()
    ncore, nact, nele = cfg['ncore'], cfg['nact'], cfg['nele']
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)

    def builder(m):
        mf = pscf.RHF(m)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        return mf

    f_cas = make_nevpt2_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                 builder, e2_scale=0.0, ci_solver='blocked',
                                 eig_solver='lanczos')
    f_nev = make_nevpt2_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                 builder, e2_scale=1.0, ci_solver='blocked',
                                 eig_solver='lanczos')
    return mol, nmo, pairs, f_cas, f_nev


def theta_quantities(f_cas, f_nev, theta):
    """(E_cas, e2, g_cas, s_e2) at one theta, all numpy."""

    def cas(t):
        return f_cas.components(t)[0]

    def corr(t):
        return f_nev.components(t)[1]

    E = f_cas.components(theta)[0]
    e2 = f_nev.components(theta)[1]
    g = jax.grad(cas)(theta)
    s = jax.grad(corr)(theta)
    return (float(E), float(e2), np.asarray(g, dtype=float),
            np.asarray(s, dtype=float))


def stage_ladder(f_cas, f_nev, npar, out, start='zeros'):
    rows = []
    if start == 'lower':
        # The zeros start lands on a saddle of the H2O theta surface
        # (measured H_min = -1.6e-2 and a 21 mEh lower minimum reached
        # from random starts).  Walk the ladder on the lower branch.
        rng = np.random.default_rng(7)
        _, theta0, _ = _optimize(f_cas, npar, 1e-8,
                                 jnp.asarray(rng.normal(scale=0.05,
                                                        size=npar)))
        theta = theta0
        print('lower-branch start: E %.12f' % float(f_cas(theta)), flush=True)
    else:
        theta = jnp.zeros(npar)
    for thr in THRESHOLDS:
        t0 = time.time()
        E_jit, theta, gnorm = _optimize(f_cas, npar, thr, theta)
        E, e2, g, s = theta_quantities(f_cas, f_nev, theta)
        rows.append(dict(threshold=thr, E_cas=E, e2=e2,
                         g_norm=gnorm, g_max=float(np.abs(g).max()),
                         s_norm=float(np.linalg.norm(s)),
                         s_max=float(np.abs(s).max()),
                         theta=np.asarray(theta).tolist(),
                         seconds=time.time() - t0))
        print('thr %.0e  E %.12f  e2 %+.12f  |g|max %.3e  |s|max %.3e'
              '  (%.1f s)' % (thr, E, e2, rows[-1]['g_max'],
                              rows[-1]['s_max'], rows[-1]['seconds']),
              flush=True)
    out['ladder'] = rows
    return rows


def _optimize(f, npar, g_conv, theta0):
    """L-BFGS-B on the joint energy, jitted, with a step cap."""
    import scipy.optimize as sopt
    E_jit = jax.jit(f)
    g_jit = jax.jit(jax.grad(f))

    def fun(x):
        return float(E_jit(jnp.asarray(x)))

    def jac(x):
        return np.asarray(g_jit(jnp.asarray(x)), dtype=float)

    res = sopt.minimize(fun, np.asarray(theta0, dtype=float), jac=jac,
                        method='L-BFGS-B',
                        options=dict(maxiter=3000, gtol=float(g_conv),
                                     ftol=1e-15, maxls=60))
    th = jnp.asarray(np.asarray(res.x, dtype=float))
    gn = float(jnp.linalg.norm(g_jit(th)))
    return E_jit, th, gn


def stage_bound(f_cas, f_nev, npar, out):
    rows = out['ladder']
    tight = min(rows, key=lambda r: r['g_max'])
    th = jnp.asarray(np.asarray(tight['theta'], dtype=float))

    def cas(t):
        return f_cas.components(t)[0]

    def corr(t):
        return f_nev.components(t)[1]

    # Value-faithful theta Hessian: a central difference of the jitted
    # theta gradient (the same construction the relaxed-gradient protocol
    # uses; an AD Hessian through the eigensolver carries frozen-rule
    # semantics and is not the Hessian of the value).
    g_jit = jax.jit(jax.grad(cas))
    n = int(th.shape[0])
    step = 1e-5
    H = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = step
        gp = np.asarray(g_jit(th + jnp.asarray(e)), dtype=float)
        gm = np.asarray(g_jit(th - jnp.asarray(e)), dtype=float)
        H[:, k] = (gp - gm) / (2.0 * step)
    H = 0.5 * (H + H.T)
    s = np.asarray(jax.grad(corr)(th), dtype=float)
    ev = np.linalg.eigvalsh(H)
    kappa = float(np.linalg.cond(H))
    pinv = np.linalg.pinv(H, rcond=1e-8)
    z = pinv @ s
    # First-order e2 error bound per unit residual gradient.
    slope = float(np.linalg.norm(z))
    print('H spectrum [%.3e, %.3e]  cond %.3e  |H^-1 s| %.6e'
          % (ev[0], ev[-1], kappa, slope), flush=True)
    pred = []
    for r in rows:
        th_r = jnp.asarray(np.asarray(r['theta'], dtype=float))
        g = np.asarray(g_jit(th_r), dtype=float)
        d_pred = -float(s @ pinv @ g)
        d_meas = r['e2'] - tight['e2']
        pred.append(dict(threshold=r['threshold'], g_max=r['g_max'],
                         de2_measured=d_meas, de2_predicted=d_pred,
                         ratio=(d_pred / d_meas if d_meas != 0 else None)))
        print('  thr %.0e  g_max %.2e  de2 meas %+.3e  pred %+.3e'
              % (r['threshold'], r['g_max'], d_meas, d_pred), flush=True)
    out['bound'] = dict(slope=slope, H_min=float(ev[0]), H_max=float(ev[-1]),
                        cond=kappa, s=s.tolist(), rows=pred)


def stage_branch(f_cas, f_nev, npar, out, g_conv=1e-8, nrand=3, seed=7):
    rng = np.random.default_rng(seed)
    starts = [np.zeros(npar)] + [rng.normal(scale=0.05, size=npar)
                                 for _ in range(nrand)]
    rows = []
    for k, st in enumerate(starts):
        E_jit, th, gn = _optimize(f_cas, npar, g_conv, jnp.asarray(st))
        E, e2, g, s = theta_quantities(f_cas, f_nev, th)
        rows.append(dict(start=k, E_cas=E, e2=e2, g_max=float(np.abs(g).max()),
                         theta=np.asarray(th).tolist()))
        print('start %d  E %.12f  e2 %+.12f  |g|max %.2e'
              % (k, E, e2, rows[-1]['g_max']), flush=True)
    out['branch'] = rows
    Es = [r['E_cas'] for r in rows]
    e2s = [r['e2'] for r in rows]
    print('E spread %.3e  e2 spread %.3e'
          % (max(Es) - min(Es), max(e2s) - min(e2s)), flush=True)


def main(system, out_path, stages):
    mol, nmo, pairs, f_cas, f_nev = build(system)
    npar = len(pairs)
    print('%s: nmo %d npar %d  ncore %d nact %d nele %d'
          % (system, nmo, npar, SYSTEMS[system]['ncore'],
             SYSTEMS[system]['nact'], SYSTEMS[system]['nele']), flush=True)
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out = json.load(fh)
    want = stages.split(',') if stages else ['ladder', 'bound', 'branch']
    start = os.environ.get('CONV_START', 'zeros')
    if 'ladder' in want:
        stage_ladder(f_cas, f_nev, npar, out, start=start)
        out['start_mode'] = start
        with open(out_path, 'w') as fh:
            json.dump(out, fh, indent=1)
    if 'bound' in want:
        stage_bound(f_cas, f_nev, npar, out)
        with open(out_path, 'w') as fh:
            json.dump(out, fh, indent=1)
    if 'branch' in want:
        stage_branch(f_cas, f_nev, npar, out)
        with open(out_path, 'w') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else '')
