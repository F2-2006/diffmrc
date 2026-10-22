"""Orbital-gradient consistency at the converged CASSCF point."""
import numpy as np
from pyscf import gto

from diffmrc.core import x2c_casscf_df as xc

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'
NCORE, NACT, NELE = 4, 2, 2


def main():
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    fac = xc.spatial_df_factors(mol, auxbasis=AUX)
    h = xc.sfx2c_hcore(mol)
    rhf = xc.df_x2c_rhf(mol, factors=fac)
    pairs = xc.orbital_pairs(NCORE, NACT, h.shape[0])
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           g_conv=1e-6, max_cycle=400)
    print('opt gnorm %.3e n_iter %d E %.12f' %
          (res.gnorm, res.n_iter, res.e_tot), flush=True)
    E, g = xc._objective(res.theta, rhf.mo_coeff, h, fac, NCORE, NACT,
                         NELE, pairs, 1, np.array([1.0]), 0)
    step = 1e-5
    fd = np.zeros(len(pairs))
    for k in range(len(pairs)):
        tp = res.theta.copy()
        tp[k] += step
        ep = xc._objective(tp, rhf.mo_coeff, h, fac, NCORE, NACT, NELE,
                           pairs, 1, np.array([1.0]), 0)[0]
        tm = res.theta.copy()
        tm[k] -= step
        em = xc._objective(tm, rhf.mo_coeff, h, fac, NCORE, NACT, NELE,
                           pairs, 1, np.array([1.0]), 0)[0]
        fd[k] = (ep - em) / (2 * step)
    print('gnorm %.3e  fd norm %.3e' %
          (np.linalg.norm(g), np.linalg.norm(fd)), flush=True)
    print('max |g - fd| %.3e  rel %.3e' %
          (np.abs(g - fd).max(),
           np.abs(g - fd).max() / max(np.abs(fd).max(), 1e-30)), flush=True)
    print('largest |fd| %.3e at pair %s' %
          (np.abs(fd).max(), pairs[int(np.argmax(np.abs(fd)))]), flush=True)


if __name__ == '__main__':
    main()
