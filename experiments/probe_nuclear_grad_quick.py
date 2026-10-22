"""Quick single-state nuclear-gradient check by two references.

Reference A: the analytic gradient of pyscf's density-fitted sfx2c1e
CASSCF, an independent program.

Reference B: the central difference of E(theta*, R) along the
envelope-theorem path -- the reference orbitals follow the geometry, the
orbital rotation parameters stay at the converged point.  This is the
quantity the analytic gradient differentiates; re-optimizing the orbitals
at each displaced geometry is branch-sensitive on H2O, whose theta space
has a second minimum.
"""
import time

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.x2c import sfx2c1e

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


def main():
    mol = mol_at()
    t0 = time.time()
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           g_conv=1e-6, max_cycle=300)
    print('opt %.2f s E %.12f gnorm %.3e it %d' %
          (time.time() - t0, res.e_tot, res.gnorm, res.n_iter), flush=True)
    grad, parts = xc.df_x2c_casscf_gradient(mol, res)
    print('grad:', np.array2string(grad, precision=10), flush=True)
    # reference A: pyscf density-fitted sfx2c1e CASSCF gradient
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    g_ref = mc.nuc_grad_method().kernel()
    print('pyscf DF CASSCF grad:', np.array2string(g_ref, precision=10),
          flush=True)
    print('max |ours - pyscf_df| %.3e  rel %.3e' %
          (np.abs(grad - g_ref).max(),
           np.abs(grad - g_ref).max() / np.abs(g_ref).max()), flush=True)
    # reference B: fixed-theta FD along the envelope-theorem path
    pairs = res.pairs
    c_ref = res.c_ref
    cache = {}

    def e_fixed(coords):
        key = tuple(np.round(coords, 10).ravel())
        if key not in cache:
            molp = mol_at(coords)
            fac = xc.spatial_df_factors(molp, auxbasis=AUX)
            hp = xc.sfx2c_hcore(molp)
            rhfp = xc.df_x2c_rhf(molp, factors=fac)
            c0 = xc._align_orbitals(c_ref, rhfp.mo_coeff,
                                    molp.intor('int1e_ovlp'))
            cache[key] = xc._state_data(
                res.theta, c0, hp, fac, NCORE, NACT, NELE, pairs, 1,
                np.array([1.0]), 0, float(molp.energy_nuc()))['e_states'][0]
        return float(cache[key])

    c0 = mol.atom_coords()
    step = 1e-3
    fd = np.zeros_like(grad)
    t1 = time.time()
    for ia in range(mol.natm):
        for x in range(3):
            c = c0.copy()
            c[ia, x] += step
            ep = e_fixed(c)
            c = c0.copy()
            c[ia, x] -= step
            em = e_fixed(c)
            fd[ia, x] = (ep - em) / (2 * step)
    print('fixed-theta FD:', np.array2string(fd, precision=10), flush=True)
    print('max |ours - fd| %.3e  rel %.3e  (%.1f s, %d evals)' %
          (np.abs(grad - fd).max(),
           np.abs(grad - fd).max() / np.abs(fd).max(),
           time.time() - t1, len(cache)), flush=True)


if __name__ == '__main__':
    main()
