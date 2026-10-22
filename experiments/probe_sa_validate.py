"""Validate the corrected SA state-specific gradient.

Checks:
  * the state-average sum rule 0.5 (g0 + g1) against pyscf's SA-CASSCF
    gradient (the SA gradient is the weighted average of the state
    gradients);
  * the branch behaviour of a re-optimized displaced SA calculation,
    which decides whether a re-optimization finite difference is a usable
    individual-state reference on H2O.
"""
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
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           nroots=2, target=0, g_conv=1e-6, max_cycle=400)
    grads = {}
    for root in (0, 1):
        g, info = xc.df_x2c_sa_state_gradient(mol, res, state=root,
                                              return_info=True)
        grads[root] = g
        print('root', root)
        print(' direct\n', np.array2string(info['direct'], precision=8))
        print(' stationary direct\n',
              np.array2string(info['direct_stationary'], precision=8))
        print(' response\n', np.array2string(info['response_grad'],
                                             precision=8))
        print(' total\n', np.array2string(g, precision=8))
        print(' |g_t| %.3e |z| %.3e hess %.3e..%.3e nkept %d' %
              (np.linalg.norm(info['g_target']),
               np.linalg.norm(info['response']),
               info['hess_absmin'], info['hess_absmax'], info['n_kept']),
              flush=True)
    avg = 0.5 * (grads[0] + grads[1])
    print('SA average\n', np.array2string(avg, precision=8), flush=True)
    # pyscf SA-CASSCF gradient
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    g_sa = np.asarray(mc.nuc_grad_method().kernel())
    print('pyscf SA\n', np.array2string(g_sa, precision=8), flush=True)
    print('max |avg - pyscf| %.3e rel %.3e' %
          (np.abs(avg - g_sa).max(),
           np.abs(avg - g_sa).max() / np.abs(g_sa).max()), flush=True)
    # branch behaviour of a re-optimized displaced SA
    c0 = mol.atom_coords()
    for sgn in (1.0, -1.0):
        c = c0.copy()
        c[0, 2] += sgn * 1e-3
        rp = xc.df_x2c_casscf(mol_at(c), NCORE, NACT, NELE, auxbasis=AUX,
                              nroots=2, target=0, g_conv=1e-6,
                              max_cycle=400, theta0=res.theta)
        print('displaced sign %+d: |dtheta| %.3e  e_states %s' %
              (sgn, np.linalg.norm(rp.theta - res.theta),
               np.array2string(rp.state_energies, precision=8)), flush=True)


if __name__ == '__main__':
    main()
