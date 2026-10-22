#!/usr/bin/env python
"""Replayable validation of the DF-X2C HF energy and analytic gradient.

Reports, on H2O/6-31G with the def2-universal-jkfit auxiliary basis:

  * X2C hcore against pyscf's SpinorX2CHelper;
  * DF J/K against the exact spinor ERI at the exact density;
  * DF SCF energy against pyscf's exact-ERI X2C UHF;
  * dB/dR and dS_metric/dR against geometry finite differences;
  * the analytic gradient against 3-point and 5-point differences of the
    converged DF energy, plus translation/rotation invariants.

Run on 101 with the pyscfad214 environment, single-thread BLAS:

    python df_x2c_gradient_validation.py [output.json]
"""
import json
import sys
import time

import numpy as np

from pyscf import df, gto
from pyscf.x2c import x2c as pyscf_x2c

from diffmrc.core import x2c_df

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX_BASIS = 'def2-universal-jkfit'


def build_mol(coords_bohr=None):
    mol = gto.M(atom=[(s, tuple(x)) for s, x in GEOM], basis='6-31g',
                unit='Angstrom', verbose=0)
    if coords_bohr is not None:
        mol.set_geom_(coords_bohr, unit='Bohr')
    return mol


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) /
                 max(np.linalg.norm(b), 1e-30))


def invariance(grad, mol):
    r = mol.atom_coords()
    trans = np.abs(grad.sum(axis=0)).max() / max(np.abs(grad).max(), 1e-30)
    rot = np.zeros(3)
    for ia in range(mol.natm):
        rot += np.cross(r[ia], grad[ia])
    rot = np.abs(rot).max() / max(np.abs(grad).max() * np.abs(r).max(),
                                  1e-30)
    return float(trans), float(rot)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'x2c_df_validation.json'
    t0 = time.time()
    res = {}
    mol = build_mol()
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUX_BASIS)
    res['nao_2c'] = mol.nao_2c()
    res['naux'] = auxmol.nao_nr()

    h = x2c_df.df_x2c_hcore(mol)
    h_ref = pyscf_x2c.SpinorX2CHelper(mol).get_hcore(mol)
    res['hcore_maxabs'] = float(np.abs(h - h_ref).max())
    res['hcore_rel'] = rel(h, h_ref)

    mf = pyscf_x2c.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.run()
    res['e_exact_pyscf'] = float(mf.e_tot)

    t1 = time.time()
    scf = x2c_df.df_x2c_hf(mol, auxmol=auxmol)
    res['scf_time_s'] = time.time() - t1
    res['e_df_scf'] = scf.e_tot
    res['e_df_minus_exact'] = scf.e_tot - mf.e_tot
    res['scf_converged'] = bool(scf.converged)
    res['scf_gradient_norm'] = scf.gradient_norm
    res['scf_n_iter'] = scf.n_iter
    res['e_components'] = {
        'one_electron': scf.e_one_electron,
        'coulomb': scf.e_coulomb,
        'exchange': scf.e_exchange,
        'nuclear': scf.e_nuc,
    }

    d = mf.make_rdm1()
    vj, vk = mf.get_jk(mol, d)
    w = scf.factors.w
    rho = np.einsum('ijP,ji->P', w, d)
    j_df = np.einsum('ijP,P->ij', w, rho)
    k_df = np.zeros_like(d)
    for p in range(scf.factors.naux):
        wp = w[:, :, p]
        k_df += wp @ d @ wp
    res['J_rel_vs_exact'] = rel(j_df, vj)
    res['K_rel_vs_exact'] = rel(k_df, vk)

    coords = mol.atom_coords()
    db_bad, ds_bad = 0.0, 0.0
    for ia in range(mol.natm):
        db, ds_metric, dj = scf.factors.deriv_factors(ia)
        for x in range(3):
            cp = coords.copy(); cp[ia, x] += 1e-4
            cm = coords.copy(); cm[ia, x] -= 1e-4
            mp, mm = build_mol(cp), build_mol(cm)
            ap = df.addons.make_auxmol(mp, auxbasis=AUX_BASIS)
            am = df.addons.make_auxmol(mm, auxbasis=AUX_BASIS)
            bp = df.incore.aux_e2(mp, ap, intor='int3c2e_spinor',
                                  aosym='s1')
            bm = df.incore.aux_e2(mm, am, intor='int3c2e_spinor',
                                  aosym='s1')
            db_bad = max(db_bad, rel(db[x], (bp - bm) / 2e-4))
            sp = ap.intor('int2c2e_sph', aosym='s1')
            sm = am.intor('int2c2e_sph', aosym='s1')
            ds_bad = max(ds_bad, rel(ds_metric[x], (sp - sm) / 2e-4))
    res['dB_rel_vs_fd'] = float(db_bad)
    res['dSmetric_rel_vs_fd'] = float(ds_bad)

    grad, parts = x2c_df.df_x2c_hf_gradient(mol, result=scf)
    res['grad'] = grad.tolist()
    res['grad_invariance'] = invariance(grad, mol)
    res['grad_parts_max'] = {k: float(np.abs(v).max())
                             for k, v in parts.items()}

    def energy_at(c):
        return x2c_df.df_x2c_hf(build_mol(c)).e_tot

    g3 = np.zeros((mol.natm, 3))
    g5 = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            e = {}
            for k in (-2, -1, 1, 2):
                c = coords.copy(); c[ia, x] += k * 1e-3
                e[k] = energy_at(c)
            g3[ia, x] = (e[1] - e[-1]) / 2e-3
            g5[ia, x] = (-e[2] + 8 * e[1] - 8 * e[-1] + e[-2]) / 12e-3
    res['grad_vs_fd3'] = rel(grad, g3)
    res['grad_vs_fd5'] = rel(grad, g5)
    res['fd3_vs_fd5'] = rel(g3, g5)
    res['grad_fd5_maxabs'] = float(np.abs(grad - g5).max())
    res['time_s'] = time.time() - t0

    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=1)
    for key in ('hcore_maxabs', 'hcore_rel', 'e_df_minus_exact',
                'J_rel_vs_exact', 'K_rel_vs_exact', 'dB_rel_vs_fd',
                'dSmetric_rel_vs_fd', 'grad_vs_fd3', 'grad_vs_fd5',
                'fd3_vs_fd5'):
        print('%-22s %.3e' % (key, res[key]))
    print('invariance', res['grad_invariance'])
    print('time %.1f s -> %s' % (res['time_s'], out_path))


if __name__ == '__main__':
    main()
