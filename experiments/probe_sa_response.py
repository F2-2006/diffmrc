"""Breakdown of the state-averaged state-specific response."""
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


def main():
    mol = mol_at()
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           nroots=2, target=0, g_conv=1e-6, max_cycle=400)
    print('SA states', res.state_energies, 'gnorm', res.gnorm, flush=True)
    grad, info = xc.df_x2c_sa_state_gradient(mol, res, state=0,
                                             return_info=True)
    step = 1e-3
    c0 = mol.atom_coords()
    fd = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            c = c0.copy()
            c[ia, x] += step
            ep = xc.df_x2c_energies_at(mol_at(c), res)[0]
            c = c0.copy()
            c[ia, x] -= step
            em = xc.df_x2c_energies_at(mol_at(c), res)[0]
            fd[ia, x] = (ep - em) / (2 * step)
    print('direct\n', np.array2string(info['direct'], precision=8), flush=True)
    print('response\n', np.array2string(info['response_grad'],
                                        precision=8), flush=True)
    print('analytic\n', np.array2string(grad, precision=8), flush=True)
    print('fd\n', np.array2string(fd, precision=8), flush=True)
    print('fd - direct\n', np.array2string(fd - info['direct'],
                                           precision=8), flush=True)
    w = info['hess_eigvals']
    print('hess eig min/max %.3e %.3e  n_kept %d' %
          (info['hess_absmin'], info['hess_absmax'], info['n_kept']),
          flush=True)
    print('ten smallest |eig|', np.sort(np.abs(w))[:10], flush=True)
    print('|g_t| %.3e  |z| %.3e' %
          (np.linalg.norm(info['g_target']), np.linalg.norm(info['response'])),
          flush=True)
    print('mixed max %.3e' % np.abs(info['mixed']).max(), flush=True)


if __name__ == '__main__':
    main()
