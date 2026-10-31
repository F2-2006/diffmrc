"""Per-process wall time and peak RSS of the RIJONX gradients.

Each invocation runs one system and one route, so ``ru_maxrss`` is that
route's peak rather than the maximum over both.  Routes:

    explicit   driver.rijonx_scf_gradient (analytic skeleton + response)
    implicit   mf.energy_grad()           (pyscfad implicit differentiation)

Usage: python probe_rijonx_gradient_cost.py <route> <system>
"""
import json
import os
import resource
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

from pyscfad import gto  # noqa: E402
from diffmrc.driver import ri_scf_driver as drv  # noqa: E402

SYSTEMS = {
    'oh': dict(atom='O 0 0 0; H 0 0 1.83', basis='6-31g', ecp=None,
               spin=1),
    'of': dict(atom='O 0 0 0; F 0 0 2.60', basis='6-31g', ecp=None,
               spin=1),
    'io-ecp': dict(atom='I 0 0 0; O 0 0 3.59', basis='def2-svp',
                   ecp='def2-svp', spin=1),
}


def build(spec):
    mol = gto.Mole()
    mol.atom = spec['atom']
    mol.basis = spec['basis']
    if spec['ecp'] is not None:
        mol.ecp = spec['ecp']
    mol.spin = int(spec['spin'])
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build()
    return mol


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


route = sys.argv[1]
system = sys.argv[2]
spec = SYSTEMS[system]
mol = build(spec)

t0 = time.time()
mf = drv.rijonx_scf(mol)
scf_s = time.time() - t0
if not mf.converged:
    raise RuntimeError('SCF did not converge')

t0 = time.time()
if route == 'explicit':
    grad = np.asarray(drv.rijonx_scf_gradient(mf), dtype=float)
elif route == 'implicit':
    grad = np.asarray(mf.energy_grad().coords, dtype=float)
else:
    raise ValueError(f'unknown route {route}')
grad_s = time.time() - t0

print(json.dumps(dict(system=system, route=route, basis=spec['basis'],
                      ecp=spec['ecp'], nao=int(mol.nao_nr()),
                      natm=int(mol.natm), scf_s=float(scf_s),
                      grad_s=float(grad_s),
                      peak_rss_gib=float(peak_gib()),
                      grad_max=float(np.abs(grad).max())), indent=2))
