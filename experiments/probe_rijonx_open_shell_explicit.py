"""Probe: explicit RIJONX gradients for ROHF and UHF against FD.

The release notes left open-shell gradients to pyscfad's implicit
``energy_grad()`` without a finite-difference check.  This script
compares three independent objects on OH and OF:

    explicit  = driver.rijonx_scf_gradient (analytic skeleton + response)
    implicit  = mf.energy_grad()           (pyscfad implicit function)
    FD        = central difference of the converged RIJONX energy

and reports the cost of each.  A stretched geometry is used alongside
the equilibrium one so the comparison is not made at a gradient that is
already near zero.

Usage: python probe_rijonx_open_shell_explicit.py [step]
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

CASES = [
    ('OH-ROHF', 'O 0 0 0; H 0 0 1.83', 1, False),
    ('OH-ROHF-stretched', 'O 0 0 0; H 0 0 2.30', 1, False),
    ('OF-ROHF', 'O 0 0 0; F 0 0 2.60', 1, False),
    ('OH-UHF', 'O 0 0 0; H 0 0 1.83', 1, True),
    ('OH-UHF-stretched', 'O 0 0 0; H 0 0 2.30', 1, True),
    ('OF-UHF', 'O 0 0 0; F 0 0 2.60', 1, True),
]


def build(atom, spin):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = '6-31g'
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


def fd_gradient(mol, uhf):
    g = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for ax in range(3):
            vals = [drv.rijonx_scf(displaced(mol, ia, ax, s * STEP),
                                   uhf=uhf).e_tot
                    for s in (1.0, -1.0)]
            g[ia, ax] = (vals[0] - vals[1]) / (2.0 * STEP)
    return g


for name, atom, spin, uhf in CASES:
    print('=' * 78, flush=True)
    print(f'{name}  spin={spin}  uhf={uhf}', flush=True)
    mol = build(atom, spin)
    t0 = time.time()
    mf = drv.rijonx_scf(mol, uhf=uhf)
    t_scf = time.time() - t0
    print(f'  nao={mol.nao_nr()} e_tot={mf.e_tot:.12f} '
          f'converged={mf.converged} scf={t_scf:.1f}s', flush=True)
    print(f'  mo_occ={np.asarray(mf.mo_occ)}', flush=True)

    t0 = time.time()
    g_exp = np.asarray(drv.rijonx_scf_gradient(mf), dtype=float)
    t_exp = time.time() - t0
    t0 = time.time()
    g_imp = np.asarray(mf.energy_grad().coords, dtype=float)
    t_imp = time.time() - t0
    t0 = time.time()
    g_fd = fd_gradient(mol, uhf)
    t_fd = time.time() - t0

    print(f'  explicit ({t_exp:.2f}s):\n{g_exp}', flush=True)
    print(f'  implicit ({t_imp:.2f}s):\n{g_imp}', flush=True)
    print(f'  FD step={STEP} ({t_fd:.2f}s):\n{g_fd}', flush=True)
    print(f'  max |explicit - implicit| = '
          f'{np.abs(g_exp - g_imp).max():.3e} Eh/Bohr', flush=True)
    print(f'  max |explicit - FD|       = '
          f'{np.abs(g_exp - g_fd).max():.3e} Eh/Bohr', flush=True)
    print(f'  max |implicit - FD|       = '
          f'{np.abs(g_imp - g_fd).max():.3e} Eh/Bohr', flush=True)
