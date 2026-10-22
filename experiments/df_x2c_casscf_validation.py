"""Validation run for the spin-free DF-X2C CASSCF response layer.

Stages (``--mode``):

  single : CASSCF energy against pyscf's density-fitted and exact-ERI
           sfx2c1e CASSCF; the analytic gradient against pyscf's
           density-fitted CASSCF gradient, against central and five-point
           differences of E(theta*, R) along the envelope-theorem path
           (the reference orbitals follow the geometry, the rotation
           parameters stay at the converged point), and the
           translation/rotation invariants.
  sa     : equal-weight state average over the two lowest roots; the
           state-specific gradients (roots 0 and 1) from the envelope-path
           direct term plus the Z-vector response; the state-average sum
           rule against pyscf's independent SA-CASSCF gradient; the branch
           evidence that a displaced re-optimization is not an
           individual-state reference; and the N = 1 limit against the
           single-state result.

Re-optimizing the orbitals at a displaced geometry is not used as a
reference: H2O has a second orbital minimum in the same theta space and
the displaced optimization can land there.  The envelope path is the
quantity the analytic gradient differentiates.

The JSON is printed and optionally written.
"""
import argparse
import json
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


def mol_at(coords_bohr=None):
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    if coords_bohr is not None:
        mol.set_geom_(coords_bohr, unit='Bohr')
    return mol


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def invariants(grad, mol):
    r = mol.atom_coords()
    trans = float(np.abs(grad.sum(axis=0)).max() / max(np.abs(grad).max(),
                                                       1e-30))
    rot = np.zeros(3)
    for ia in range(mol.natm):
        rot += np.cross(r[ia], grad[ia])
    rot = float(np.abs(rot).max() / max(np.abs(grad).max()
                                        * np.abs(r).max(), 1e-30))
    return trans, rot


class EnvelopeCache:
    """E(theta*, R) at displaced geometries, orbitals following the geometry."""

    def __init__(self, result):
        self.result = result
        self.cache = {}

    def __call__(self, coords):
        key = tuple(np.round(np.asarray(coords, float).ravel(), 10))
        if key not in self.cache:
            self.cache[key] = np.asarray(
                xc.df_x2c_energies_at(mol_at(coords), self.result), float)
        return self.cache[key]

    def __len__(self):
        return len(self.cache)


def finite_difference(mol, energies_of, step, stencil, root=0):
    c0 = mol.atom_coords()
    grad = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            if stencil == 3:
                c = c0.copy()
                c[ia, x] += step
                ep = energies_of(c)[root]
                c = c0.copy()
                c[ia, x] -= step
                em = energies_of(c)[root]
                grad[ia, x] = (ep - em) / (2 * step)
            elif stencil == 5:
                vals = []
                for k in (1, -1, 2, -2):
                    c = c0.copy()
                    c[ia, x] += k * step
                    vals.append(energies_of(c)[root])
                ep1, em1, ep2, em2 = vals
                grad[ia, x] = (-ep2 + 8 * ep1 - 8 * em1 + em2) / (12 * step)
            else:
                raise ValueError(stencil)
    return grad


def pyscf_reference(mol, density_fit):
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    return mc


def stage_single(out, step=1e-3):
    mol = mol_at()
    t0 = time.time()
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           g_conv=1e-6, max_cycle=400)
    out['casscf'] = dict(
        e_tot=float(res.e_tot), e_sa=float(res.e_sa),
        gnorm=float(res.gnorm), n_iter=int(res.n_iter),
        n_pairs=len(res.pairs), seconds=time.time() - t0)
    mc_df = pyscf_reference(mol, density_fit=True)
    mc_ex = pyscf_reference(mol, density_fit=False)
    out['pyscf_df_casscf'] = float(mc_df.e_tot)
    out['pyscf_exact_casscf'] = float(mc_ex.e_tot)
    out['casscf_minus_pyscf_df'] = float(res.e_tot - mc_df.e_tot)
    out['casscf_minus_pyscf_exact'] = float(res.e_tot - mc_ex.e_tot)
    grad, parts = xc.df_x2c_casscf_gradient(mol, res)
    out['gradient'] = grad.tolist()
    out['gradient_parts_max'] = {k: float(np.abs(v).max())
                                 for k, v in parts.items()}
    out['gradient_scale'] = float(np.abs(grad).max())
    tr, ro = invariants(grad, mol)
    out['translation_invariance'] = tr
    out['rotation_invariance'] = ro
    g_df = mc_df.nuc_grad_method().kernel()
    out['pyscf_df_gradient'] = np.asarray(g_df).tolist()
    out['gradient_vs_pyscf_df'] = dict(
        max_abs=float(np.abs(grad - g_df).max()),
        rel=rel(grad, g_df))
    cache = EnvelopeCache(res)
    for name, stencil in (('central', 3), ('five_point', 5)):
        t1 = time.time()
        fd = finite_difference(mol, cache, step, stencil)
        out[name] = dict(
            rel=rel(grad, fd), max_abs=float(np.abs(grad - fd).max()),
            fd=fd.tolist(), seconds=time.time() - t1)
    out['n_energy_evaluations'] = len(cache)
    return res


def stage_sa(out, step=1e-3):
    mol = mol_at()
    t0 = time.time()
    res = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           nroots=2, target=0, g_conv=1e-6, max_cycle=400)
    out['sa'] = dict(e_states=[float(x) for x in res.state_energies],
                     e_sa=float(res.e_sa), gnorm=float(res.gnorm),
                     n_iter=int(res.n_iter), seconds=time.time() - t0)
    grads, infos = {}, {}
    for root in (0, 1):
        t1 = time.time()
        grad, info = xc.df_x2c_sa_state_gradient(mol, res, state=root,
                                                 return_info=True)
        grads[root], infos[root] = grad, info
        out[f'sa_state_{root}'] = dict(
            grad=grad.tolist(), direct=info['direct'].tolist(),
            direct_stationary=info['direct_stationary'].tolist(),
            response=info['response_grad'].tolist(),
            g_target=info['g_target'].tolist(),
            hess_absmin=info['hess_absmin'],
            hess_absmax=info['hess_absmax'],
            n_kept=info['n_kept'], n_response=info['n_response'],
            seconds=time.time() - t1)
    # state-average sum rule against pyscf's independent SA-CASSCF gradient
    avg = 0.5 * (grads[0] + grads[1])
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    g_ref = np.asarray(mc.nuc_grad_method().kernel())
    out['sa_sum_rule'] = dict(
        average=avg.tolist(), pyscf=g_ref.tolist(),
        max_abs=float(np.abs(avg - g_ref).max()),
        rel=rel(avg, g_ref),
        response_sum_max=float(np.abs(0.5 * (infos[0]['response_grad']
                                             + infos[1]['response_grad'])).max()))
    # branch evidence: a displaced SA optimization from theta* moves far
    c0 = mol.atom_coords()
    c = c0.copy()
    c[0, 2] += step
    rp = xc.df_x2c_casscf(mol_at(c), NCORE, NACT, NELE, auxbasis=AUX,
                          nroots=2, target=0, g_conv=1e-6, max_cycle=400,
                          theta0=res.theta)
    out['sa_branch'] = dict(
        dtheta_norm=float(np.linalg.norm(rp.theta - res.theta)),
        e_states=[float(x) for x in rp.state_energies],
        note='a displaced re-optimization is not an individual-state '
             'reference when |dtheta| is O(0.1)')
    sa1 = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                           nroots=1, g_conv=1e-6, max_cycle=400)
    single = xc.df_x2c_casscf(mol, NCORE, NACT, NELE, auxbasis=AUX,
                              g_conv=1e-6, max_cycle=400)
    g_sa1, _ = xc.df_x2c_casscf_gradient(mol, sa1)
    g_single, _ = xc.df_x2c_casscf_gradient(mol, single)
    out['sa_n1'] = dict(
        max_coeff_diff=float(np.abs(sa1.mo_coeff - single.mo_coeff).max()),
        max_grad_diff=float(np.abs(g_sa1 - g_single).max()),
        e_diff=float(sa1.e_tot - single.e_tot))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('single', 'sa', 'both'),
                    default='single')
    ap.add_argument('--json', default=None)
    ap.add_argument('--step', type=float, default=1e-3)
    args = ap.parse_args()
    out = {}
    if args.mode in ('single', 'both'):
        stage_single(out, step=args.step)
    if args.mode in ('sa', 'both'):
        stage_sa(out, step=args.step)
    text = json.dumps(out, indent=2, default=float)
    print(text)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            fh.write(text)


if __name__ == '__main__':
    main()
