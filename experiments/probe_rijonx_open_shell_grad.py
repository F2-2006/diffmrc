"""Probe: does pyscfad's implicit RIJONX gradient hold for open-shell ROHF?

The release notes state that open-shell gradients are reachable through
``mf.energy_grad()`` but were never checked against finite differences.
This script measures that gap on OH and OF before any explicit gradient
is implemented.  It also records the density/occupation layout so the
explicit route can be built against the object actually returned by the
SCF, not against an assumption.

Usage: python probe_rijonx_open_shell_grad.py [step]
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

SYSTEMS = {
    'OH': ('O 0 0 0; H 0 0 1.83', '6-31g', 1),
    'OF': ('O 0 0 0; F 0 0 2.60', '6-31g', 1),
}


def build(atom, basis, spin):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.spin = int(spin)
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build()
    return mol


def displaced(mol, atom, axis, step):
    coords = np.array(mol.atom_coords(), dtype=float, copy=True)
    coords[atom, axis] += step
    m = gto.Mole()
    m.atom = [(mol.atom_pure_symbol(i), coords[i]) for i in range(mol.natm)]
    m.basis = mol.basis
    m.unit = 'Bohr'
    m.verbose = 0
    m.spin = int(mol.spin)
    m.build()
    return m


for name, (atom, basis, spin) in SYSTEMS.items():
    print('=' * 78, flush=True)
    print(f'{name}  basis={basis}  spin={spin}', flush=True)
    mol = build(atom, basis, spin)
    t0 = time.time()
    mf = drv.rijonx_scf(mol)
    t_scf = time.time() - t0
    dm = np.asarray(mf.make_rdm1())
    print(f'  nao={mol.nao_nr()} natm={mol.natm} e_tot={mf.e_tot:.12f} '
          f'converged={mf.converged} scf={t_scf:.1f}s', flush=True)
    print(f'  mo_occ={np.asarray(mf.mo_occ)}', flush=True)
    print(f'  dm shape={dm.shape}', flush=True)
    print(f'  nelec={mol.nelec}', flush=True)

    t0 = time.time()
    g = np.asarray(mf.energy_grad().coords, dtype=float)
    t_grad = time.time() - t0
    print(f'  implicit gradient ({t_grad:.1f}s):\n{g}', flush=True)

    fd = np.zeros_like(g)
    t0 = time.time()
    for ia in range(mol.natm):
        for ax in range(3):
            vals = []
            for sign in (1.0, -1.0):
                vals.append(drv.rijonx_scf(displaced(mol, ia, ax,
                                                     sign * STEP)).e_tot)
            fd[ia, ax] = (vals[0] - vals[1]) / (2.0 * STEP)
    t_fd = time.time() - t0
    dev = np.abs(g - fd)
    print(f'  FD gradient (step={STEP}, {t_fd:.1f}s):\n{fd}', flush=True)
    print(f'  max |implicit - FD| = {dev.max():.3e} Eh/Bohr', flush=True)
