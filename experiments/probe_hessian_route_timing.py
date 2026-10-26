"""Probe: where the time goes in the three Hessian routes.

Runs the dense, columnwise and matrix-free routes on H2O/STO-3G and
prints each stage as soon as it finishes, so a slow stage is visible
without waiting for the whole comparison.  Also reports the LSMR
iteration count per geometric column.

Usage: python probe_hessian_route_timing.py [atol] [maxiter]
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pyscfad import gto, scf  # noqa: E402

from diffmrc.core.hessian import (implicit_hessian,  # noqa: E402
                                  implicit_hessian_columnwise,
                                  implicit_hessian_hvp, make_joint_energy_fn,
                                  make_theta_energy_fn)
from diffmrc.core.orblib import orbital_pairs  # noqa: E402

ATOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-10
MAXITER = int(sys.argv[2]) if len(sys.argv) > 2 else None

mol = gto.Mole()
mol.atom = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
mol.basis = 'sto-3g'
mol.verbose = 0
mol.build()
nmo = mol.nao_nr()
ncore, nact, nele = 1, 2, 2
pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
theta = jnp.zeros(len(pairs))
print(f'H2O/STO-3G  nmo={nmo} natm={mol.natm} npar={len(pairs)} '
      f'ngeom={3 * mol.natm}  atol={ATOL} maxiter={MAXITER}', flush=True)

E_joint = make_joint_energy_fn(ncore, nact, nele, nmo, pairs, scf,
                               scf=scf.RHF)
E_tt = make_theta_energy_fn(mol, ncore, nact, nele, nmo, pairs, scf,
                            scf=scf.RHF)

t0 = time.time()
dense = implicit_hessian(mol, theta, E_joint)
print(f'  dense       {time.time() - t0:7.1f}s  '
      f'|H|max={np.abs(dense.H_total).max():.6e}', flush=True)

t0 = time.time()
col = implicit_hessian_columnwise(mol, theta, E_joint, E_tt=E_tt)
print(f'  columnwise  {time.time() - t0:7.1f}s  '
      f'|col-dense|={np.abs(col.H_total - dense.H_total).max():.3e}',
      flush=True)

t0 = time.time()
hvp = implicit_hessian_hvp(mol, theta, E_joint, E_tt=E_tt,
                           atol=ATOL, btol=ATOL, maxiter=MAXITER)
print(f'  hvp         {time.time() - t0:7.1f}s  '
      f'|hvp-dense|={np.abs(hvp.H_total - dense.H_total).max():.3e}',
      flush=True)
print(f'  lsmr iterations = {hvp.lsmr_iterations.tolist()}', flush=True)
print(f'  lsmr istop      = {hvp.lsmr_istop.tolist()}', flush=True)
print(f'  lsmr residual   = {hvp.lsmr_residual.tolist()}', flush=True)
