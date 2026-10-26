"""Probe: open-shell RIJONX gradient with an effective core potential.

The open-shell response uses the core Hamiltonian, which carries the
ECP scalar term, and the skeleton gradient uses the ECP derivative
terms already exercised by the closed-shell HI test.  This probe checks
the combination on an IO doublet: iodine is described by def2-svp with
its ECP, oxygen all-electron.

Usage: python probe_rijonx_open_shell_ecp.py [step]
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

from pyscfad import gto  # noqa: E402
from diffmrc.driver import ri_scf_driver as drv  # noqa: E402

STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0e-3
ATOM = 'I 0 0 0; O 0 0 3.59'
BASIS = 'def2-svp'
ECP = 'def2-svp'


def build():
    mol = gto.Mole()
    mol.atom = ATOM
    mol.basis = BASIS
    mol.ecp = ECP
    mol.spin = 1
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build()
    return mol


def displaced(mol, atom, axis, step):
    coords = np.array(mol.atom_coords(), dtype=float, copy=True)
    coords[atom, axis] += step
    m = gto.Mole()
    m.atom = [(mol.atom_pure_symbol(i), coords[i]) for i in range(mol.natm)]
    m.basis = BASIS
    m.ecp = ECP
    m.spin = 1
    m.unit = 'Bohr'
    m.verbose = 0
    m.build()
    return m


mol = build()
print(f'IO doublet: nao={mol.nao_nr()} natm={mol.natm} '
      f'ecp={mol.has_ecp()} nelec={mol.nelec}', flush=True)
t0 = time.time()
mf = drv.rijonx_scf(mol)
print(f'  e_tot={mf.e_tot:.12f} converged={mf.converged} '
      f'scf={time.time() - t0:.1f}s', flush=True)
print(f'  mo_occ={np.asarray(mf.mo_occ)}', flush=True)

t0 = time.time()
g_exp = np.asarray(drv.rijonx_scf_gradient(mf), dtype=float)
t_exp = time.time() - t0
t0 = time.time()
g_imp = np.asarray(mf.energy_grad().coords, dtype=float)
t_imp = time.time() - t0

fd = np.zeros_like(g_exp)
t0 = time.time()
for ia in range(mol.natm):
    for ax in range(3):
        vals = [drv.rijonx_scf(displaced(mol, ia, ax, s * STEP)).e_tot
                for s in (1.0, -1.0)]
        fd[ia, ax] = (vals[0] - vals[1]) / (2.0 * STEP)
t_fd = time.time() - t0

print(f'  explicit ({t_exp:.2f}s):\n{g_exp}', flush=True)
print(f'  implicit ({t_imp:.2f}s):\n{g_imp}', flush=True)
print(f'  FD ({t_fd:.2f}s):\n{fd}', flush=True)
print(f'  max |explicit - implicit| = '
      f'{np.abs(g_exp - g_imp).max():.3e} Eh/Bohr', flush=True)
print(f'  max |explicit - FD|       = '
      f'{np.abs(g_exp - fd).max():.3e} Eh/Bohr', flush=True)
print(f'  max |implicit - FD|       = '
      f'{np.abs(g_imp - fd).max():.3e} Eh/Bohr', flush=True)
