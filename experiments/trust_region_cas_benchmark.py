#!/usr/bin/env python
"""Compare the trust-region theta stage with L-BFGS-B on small CAS cases.

The systems are the ones this project already uses for theta-space
landscape checks:

* H2O / 6-31g, CAS(2,2), ncore = 4. From theta = 0 the protocol reaches
  a stationary point with three negative Hessian eigenvalues
  (E = -75.984881801724); a perturbed start reaches a lower branch
  (E = -76.006040634709), 13.28 kcal/mol below. The pair is the
  near-degenerate case: one surface, two stationary points, one of them
  a saddle.
* CO / 6-31g, CAS(2,2), ncore = 5. The known A/B solutions of the theta
  surface differ by 10.76 kcal/mol, so the start also matters here.

The comparison is on the same start and the same gradient target:
energy, gradient norm, subproblem iterations, accepted and rejected
steps, and the Hessian spectrum at the endpoint. Cost is reported as
Hessian-vector products, not wall time: the machine is shared.

Usage:
    python trust_region_cas_benchmark.py [H2O|CO|all] [basis] [nthread]

Environment: JAX_PLATFORMS=cpu is forced when unset. The script needs
pyscfad and writes nothing outside stdout unless --json is given.
"""
import argparse
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = os.environ.get(_v, '4')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp                                    # noqa: E402

from pyscfad import gto as pgto, scf as pscf                    # noqa: E402

from diffmrc.core.casci import cs_casci_blocked                 # noqa: E402
from diffmrc.core.optimizer import (optimize_theta_scipy,       # noqa: E402
                                    optimize_theta_trust_region)
from diffmrc.core.orblib import orbital_pairs                   # noqa: E402
from diffmrc.core.rotation import block_rotation                # noqa: E402
from diffmrc.core.stationary import curvature_report            # noqa: E402
from diffmrc.driver.orbitals import converge_orbital_point      # noqa: E402

SYSTEMS = {
    'H2O': ('O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880', 4, 2, 2),
    'CO':  ('C 0 0 0; O 0 0 1.1283', 5, 2, 2),
}
SEED = 1000
PERTURB = 0.3


def build(system, basis):
    atom, ncore, nact, nele = SYSTEMS[system]
    mol = pgto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.build()
    nmo = mol.nao_nr()
    mf = pscf.RHF(mol)
    mf.kernel()
    C0 = np.asarray(mf.mo_coeff)
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)

    def E_cas(C):
        Cj = jnp.asarray(C)
        nm = ncore + nact
        h1 = jnp.einsum('ai,ab,bj->ij', Cj, h1e, Cj)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                        eri)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + enuc

    E_theta = lambda t: E_cas(C0 @ block_rotation(t, nmo, pairs))
    return E_theta, len(pairs), (ncore, nact, nele), nmo


def spectrum(E_theta, theta):
    """Eigenvalue summary at a point; npar is small here (50-66)."""
    H = np.asarray(jax.hessian(E_theta)(jnp.asarray(theta)))
    return curvature_report(H)


def run_case(system, basis):
    E_theta, npar, blocks, nmo = build(system, basis)
    rng = np.random.default_rng(SEED)
    pad = PERTURB * rng.standard_normal(npar)
    starts = [('zeros', np.zeros(npar)), ('rand0', pad)]

    print(f'\n{"=" * 88}')
    print(f'{system}/{basis}  CAS{blocks}  npar={npar}  nmo={nmo}')
    print(f'{"=" * 88}')
    header = (f'{"start":>8} {"method":>13} {"E (Ha)":>17} {"|g|":>10} '
              f'{"nit":>5} {"acc":>4} {"rej":>4} {"hvp":>5} '
              f'{"n_neg":>6} {"radius":>9}')
    print(header)
    print('-' * len(header))

    rows = []
    for name, theta0 in starts:
        for method in ('lbfgs-b', 'trust-region'):
            if method == 'lbfgs-b':
                E_jit, theta, gnorm, info = optimize_theta_scipy(
                    E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
                n_hvp, n_accept, n_reject, radius = 0, 0, 0, 0.0
            else:
                E_jit, theta, gnorm, info = optimize_theta_trust_region(
                    E_theta, npar, steps=2000, g_conv=1e-8, theta0=theta0)
                n_hvp = info.get('n_hvp', 0)
                n_accept = info.get('n_accept', 0)
                n_reject = info.get('n_reject', 0)
                radius = info.get('radius', 0.0)
            E = float(E_theta(jnp.asarray(theta)))
            sp = spectrum(E_theta, theta)
            row = dict(system=system, basis=basis, start=name,
                       method=method, energy=E, grad_norm=float(gnorm),
                       nit=int(info.get('nit', -1)),
                       n_accept=int(n_accept), n_reject=int(n_reject),
                       n_hvp=int(n_hvp), n_negative=int(sp.n_negative),
                       radius=float(radius),
                       stop=str(info.get('msg', '')))
            rows.append(row)
            print(f'{name:>8} {method:>13} {E:>17.12f} {gnorm:>10.2e} '
                  f'{row["nit"]:>5} {row["n_accept"]:>4} '
                  f'{row["n_reject"]:>4} {row["n_hvp"]:>5} '
                  f'{row["n_negative"]:>6} {row["radius"]:>9.2e}')
    return rows


def run_pipeline_case(system, basis):
    """The production path: stage, damped-Newton refinement, polish.

    The stage table above compares optimizers on one problem. This table
    compares what a caller actually gets from ``converge_orbital_point``:
    the final gradient (after refinement), the stage that ran, and the
    Hessian spectrum at the endpoint. The default path must stay
    L-BFGS-B; the trust-region path is opt-in.
    """
    E_theta, npar, blocks, nmo = build(system, basis)
    rng = np.random.default_rng(SEED)
    pad = PERTURB * rng.standard_normal(npar)
    starts = [('zeros', np.zeros(npar)), ('rand0', pad)]

    print(f'\n{"=" * 96}')
    print(f'{system}/{basis}  CAS{blocks}  npar={npar}  production pipeline')
    print(f'{"=" * 96}')
    header = (f'{"start":>8} {"stage":>13} {"E (Ha)":>17} {"|g| final":>11} '
              f'{"|g| stage":>11} {"nit":>5} {"n_neg":>6} {"stop":>18}')
    print(header)
    print('-' * len(header))

    rows = []
    for name, theta0 in starts:
        for use_tr in (False, True):
            point = converge_orbital_point(
                E_theta, npar, theta0=theta0, max_steps=2000,
                use_trust_region=use_tr)
            theta = np.asarray(point.theta)
            sp = spectrum(E_theta, theta)
            row = dict(system=system, basis=basis, start=name,
                       path=point.optimizer_name,
                       energy=float(point.energy),
                       grad_norm=float(point.grad_norm),
                       stage_grad_norm=float(point.optimizer_grad_norm),
                       nit=int(point.optimizer_nit),
                       n_negative=int(sp.n_negative),
                       stop=str(point.stop_reason.value),
                       polished=bool(point.polished))
            rows.append(row)
            print(f'{name:>8} {point.optimizer_name:>13} '
                  f'{row["energy"]:>17.12f} {row["grad_norm"]:>11.2e} '
                  f'{row["stage_grad_norm"]:>11.2e} {row["nit"]:>5} '
                  f'{row["n_negative"]:>6} {row["stop"]:>18}')
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('system', nargs='?', default='all',
                    choices=['H2O', 'CO', 'all'])
    ap.add_argument('basis', nargs='?', default='6-31g')
    ap.add_argument('--json', default=None)
    ap.add_argument('--pipeline', action='store_true',
                    help='also run the full converge_orbital_point path')
    args = ap.parse_args(argv)

    systems = list(SYSTEMS) if args.system == 'all' else [args.system]
    rows = []
    for system in systems:
        rows.extend(run_case(system, args.basis))
    if args.pipeline:
        for system in systems:
            rows.extend(run_pipeline_case(system, args.basis))

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(rows, fh, indent=2)
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
