"""Open-shell NEVPT2 frozen gradient and the CAS(3,2) zero-response theorem.

OH/6-31G, ROHF, doublet, ncore=3, nact=2, nele=3.  For a three-electron
active space every determinant of the active CI space differs from the
reference by one spin orbital, so the active CI matrix in the canonical
ROHF basis is diagonal and the CASSCF stationary point coincides with
the ROHF orbitals at every geometry.  The response term of the relaxed
NEVPT2 gradient therefore vanishes identically on this system; what can
be validated is the frozen-theta gradient and the fact that theta* = 0.

Measured quantities, all along the O-H stretch direction:
  e2                      NEVPT2 correction at theta = 0
  |g_cas(0)|max           CASSCF theta gradient at the claimed stationarity
  |s|max = |d e2/dtheta|  correlation source (nonzero, but multiplied by
                          dtheta*/dR = 0 in the relaxed path)
  dE/dR|theta (AD)        frozen partial derivative
  dE/dR|theta (FD)        central difference of the same chain
  theta residual          max |g_cas| at the displaced geometries

Usage: python open_shell_check.py <out.json>
"""
import json
import os
import sys

import numpy as np
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from pyscfad import gto as pgto, scf as pscf

from diffmrc.core.orblib import orbital_pairs
from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                   nevpt2_nuclear_gradient)

# 0.9697 Angstrom expressed in bohr, so the coordinate displacements below
# (also bohr) act on the geometry the validation report describes.
ATOM = 'O 0 0 0; H 0 0 1.83245'
NCORE, NACT, NELE = 3, 2, 3
V = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])   # stretch direction


def build(coords=None):
    mol = pgto.Mole()
    if coords is None:
        mol.atom = ATOM
    else:
        mol.atom = [['O', tuple(coords[0])], ['H', tuple(coords[1])]]
    mol.basis = '6-31g'
    mol.spin = 1
    mol.unit = 'B'
    mol.build()
    return mol


def main(out_path):
    mol = build()
    nmo = mol.nao_nr()
    pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
    npar = len(pairs)

    def builder(m):
        mf = pscf.ROHF(m)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        return mf

    f = make_nevpt2_joint_fn(NCORE, NACT, NELE, nmo, pairs, builder,
                             e2_scale=1.0, ci_solver='blocked',
                             eig_solver='lanczos')
    th0 = jnp.zeros(npar)
    E_cas, e2 = f.components(mol, th0)
    g_cas = jax.grad(lambda t: f.components(mol, t)[0])(th0)
    s = jax.grad(lambda t: f.components(mol, t)[1])(th0)
    print('nmo %d npar %d  E_cas %.12f  e2 %+.12f' % (
        nmo, npar, float(E_cas), float(e2)), flush=True)
    print('|g_cas(0)|max %.3e  |s|max %.3e'
          % (float(jnp.abs(g_cas).max()), float(jnp.abs(s).max())),
          flush=True)

    g = np.asarray(nevpt2_nuclear_gradient(
        mol, lambda m_: f(m_, th0)), dtype=float)
    d_ad = float(np.sum(g * V))
    print('frozen AD dE/dR %.12e' % d_ad, flush=True)

    coords0 = np.asarray(mol.atom_coords())
    fd = {}
    for delta in (3e-4, 1e-4):
        vals = []
        for sgn in (+1.0, -1.0):
            mol_d = build(coords0 + sgn * delta * V)
            vals.append(float(f(mol_d, th0)))
        fd[str(delta)] = (vals[0] - vals[1]) / (2.0 * delta)
        print('frozen FD d=%.0e  %.12e  rel %.2e' % (
            delta, fd[str(delta)], abs(fd[str(delta)] - d_ad)
            / max(abs(d_ad), 1e-30)), flush=True)

    # theta stationarity at displaced geometries (the zero-response premise)
    th_res = {}
    for delta in (3e-4, 1e-3):
        rows = []
        for sgn in (+1.0, -1.0):
            mol_d = build(coords0 + sgn * delta * V)
            gg = jax.grad(lambda t: f.components(mol_d, t)[0])(th0)
            rows.append(float(jnp.abs(gg).max()))
        th_res[str(delta)] = max(rows)
        print('theta residual at d=%.0e: %.2e' % (delta, th_res[str(delta)]),
              flush=True)

    rec = dict(system='OH/6-31G CAS(3,2)', nmo=nmo, npar=npar,
               E_cas=float(E_cas), e2=float(e2),
               g_cas_max=float(jnp.abs(g_cas).max()),
               s_max=float(jnp.abs(s).max()),
               d_ad=d_ad, fd=fd, theta_residual=th_res,
               gradient=g.tolist())
    with open(out_path, 'w') as fh:
        json.dump(rec, fh, indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'open_shell.json')
