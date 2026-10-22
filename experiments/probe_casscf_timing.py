"""Timing breakdown of the DF-X2C CASSCF response path."""
import time

import numpy as np
from pyscf import gto

from diffmrc.core import x2c_casscf_df as xc

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'
NCORE, NACT, NELE = 4, 2, 2


def mol_at(coords=None):
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
    return mol


def tic(label, t0):
    print('%-32s %8.3f s' % (label, time.time() - t0), flush=True)
    return time.time()


def main():
    mol = mol_at()
    t0 = time.time()
    fac = xc.spatial_df_factors(mol, auxbasis=AUX)
    t0 = tic('spatial_df_factors', t0)
    h = xc.sfx2c_hcore(mol)
    t0 = tic('sfx2c_hcore', t0)
    rhf = xc.df_x2c_rhf(mol, factors=fac)
    t0 = tic('df_x2c_rhf', t0)
    nmo = h.shape[0]
    pairs = xc.orbital_pairs(NCORE, NACT, nmo)
    theta = np.zeros(len(pairs))
    for nr in (1, 2):
        t = time.time()
        for _ in range(10):
            xc._state_data(theta, rhf.mo_coeff, h, fac, NCORE, NACT, NELE,
                           pairs, nr, np.full(nr, 1.0 / nr), 0)
        t0 = tic('state_data x10 nroots=%d' % nr, t)
    t = time.time()
    res1 = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                            g_conv=1e-8)
    t0 = tic('casscf single (optimize)', t)
    print('  n_iter', res1.n_iter, 'gnorm', res1.gnorm, flush=True)
    t = time.time()
    res2 = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                            nroots=2, target=0, g_conv=1e-8,
                            theta0=res1.theta)
    t0 = tic('casscf SA(2) (optimize)', t)
    print('  n_iter', res2.n_iter, 'gnorm', res2.gnorm, flush=True)
    t = time.time()
    grad, info = xc.df_x2c_sa_state_gradient(mol, res2, state=0,
                                             return_info=True)
    t0 = tic('sa_state_gradient total', t)
    print('  hess eigmin/max %.3e %.3e n_kept %d' %
          (info['hess_absmin'], info['hess_absmax'], info['n_kept']),
          flush=True)
    t = time.time()
    c0 = mol.atom_coords()
    c = c0.copy()
    c[0, 2] += 1e-3
    rp = xc.df_x2c_casscf(mol_at(c), NCORE, NACT, NELE, auxbasis=AUX,
                          nroots=2, target=0, g_conv=1e-8,
                          theta0=res2.theta)
    t0 = tic('one displaced SA optimization', t)
    del rp


if __name__ == '__main__':
    main()
