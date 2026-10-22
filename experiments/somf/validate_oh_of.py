#!/usr/bin/env python
"""Acceptance run for the SOMF two-electron operator on OH and OF.

Compares the stack against ORCA 6.1.1 (same geometry, basis, active space,
two state-averaged roots):

    system   one-electron (ORCA)   SOMF (ORCA)
    OH       205.88 cm^-1          128.83 cm^-1
    OF       272.06 cm^-1          176.12 cm^-1

Three criteria are reported:
  (1) absolute splitting deviation against ORCA;
  (2) relative deviation;
  (3) the screening ratio SOMF/one-electron, which removes any overall
      convention offset of the one-electron operator between the two
      codes.  A scale-corrected SOMF value
      ``our_somf * orca_1e / our_1e`` is also printed.

SI assembly follows the reference SI convention of this stack:
H_{Is,Js'} = E_I delta_IJ delta_ss' + i (alpha^2/2) sum_k T^IJ_k s_k,
with T^IJ_k = sum_pq D^IJ_pq h_k[pq].  The module returns the raw operator
in the ``int1e_pnucxp`` convention, so the same prefactor is used for the
one-electron and SOMF operators.

Run:
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python validate_oh_of.py \
      --out results_somf.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import argparse
import json
import time

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.data import nist

from diffmrc.core import somf as dsomf

ALPHA2 = float(nist.ALPHA) ** 2
CM = 219474.6313632
SPIN = 0.5 * np.array([[[0, 1], [1, 0]],
                       [[0, -1j], [1j, 0]],
                       [[1, 0], [0, -1]]], dtype=complex)

SYSTEMS = {
    'oh': dict(atom=[('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.8324))],
               spin=1, ncore=1, nact=6,
               orca_1e=205.88, orca_somf=128.83),
    'of': dict(atom=[('O', (0.0, 0.0, 0.0)), ('F', (0.0, 0.0, 2.4))],
               spin=1, ncore=2, nact=8,
               orca_1e=272.06, orca_somf=176.12),
}


def si_splitting(es, T, nstates):
    H = np.zeros((2 * nstates, 2 * nstates), dtype=complex)
    pref = 1j * ALPHA2 / 2.0
    for i in range(nstates):
        H[2 * i, 2 * i] = es[i]
        H[2 * i + 1, 2 * i + 1] = es[i]
        for j in range(nstates):
            for k in range(3):
                H[2 * i:2 * i + 2, 2 * j:2 * j + 2] += \
                    pref * T[i, j, k] * SPIN[k]
    w = np.linalg.eigvalsh(H)
    return float((w[2] - w[0]) * CM), float(np.abs(H - H.conj().T).max())


def transition_tensor(D, h_mo, nstates):
    T = np.zeros((nstates, nstates, 3))
    for i in range(nstates):
        for j in range(nstates):
            for k in range(3):
                T[i, j, k] = float(np.sum(D[i, j] * h_mo[k]))
    return T


def run(name, spec):
    t0 = time.time()
    mol = gto.M(atom=spec['atom'], basis='6-31g', unit='Bohr',
                spin=spec['spin'], verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('ROHF did not converge')
    nstates = 2
    mc = mcscf.CASSCF(mf, spec['nact'], mol.nelectron - 2 * spec['ncore'])
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([1.0 / nstates] * nstates)
    mc.fcisolver.nroots = nstates
    mc.kernel()
    if not mc.converged:
        raise RuntimeError('CASSCF did not converge')

    es = np.asarray(mc.fcisolver.e_states)[:nstates]
    C = np.asarray(mc.mo_coeff)
    act = slice(spec['ncore'], spec['ncore'] + spec['nact'])
    Ca = C[:, act]
    ci = mc.ci

    dm_sa = np.asarray(mc.make_rdm1())
    dm_cas = mc.fcisolver.states_make_rdm1(ci, mc.ncas, mc.nelecas)
    Ccore = C[:, :mc.ncore]
    dm_states = [2.0 * Ccore @ Ccore.T + Ca @ np.asarray(d) @ Ca.T
                 for d in dm_cas]
    s = mol.intor('int1e_ovlp')
    tra = [float(np.einsum('pq,pq->', d, s)) for d in [dm_sa] + dm_states]

    D = {}
    for i in range(nstates):
        for j in range(nstates):
            D[i, j] = np.asarray(mc.fcisolver.trans_rdm1(
                ci[i], ci[j], spec['nact'], mc.nelecas))

    # AO-level cross-check: explicit einsum versus pyscf direct_mapdm.
    h2_exp = dsomf.somf_two_electron(mol, dm_sa, contract='explicit')
    h2_dir = dsomf.somf_two_electron(mol, dm_sa, contract='direct')
    ao_diff = float(np.abs(h2_exp - h2_dir).max())
    ao_scale = float(np.abs(h2_exp).max())
    ao_antisym = float(np.abs(h2_exp + h2_exp.transpose(0, 2, 1)).max())
    geo_sym = float(np.abs(h2_dir + h2_dir.transpose(0, 2, 1)).max())

    h1_mo = dsomf.somf_matrix(mol, Ca, None, mode='one_electron')
    h_somf_mo = dsomf.somf_matrix(mol, Ca, dm_sa, mode='somf')
    h_somf_state = [dsomf.somf_matrix(mol, Ca, d, mode='somf')
                    for d in dm_states]

    def split(h_mo):
        return si_splitting(es, transition_tensor(D, h_mo, nstates), nstates)

    s1e, herm1 = split(h1_mo)
    ssomf, herms = split(h_somf_mo)
    s_state = [split(h) for h in h_somf_state]

    ratio = ssomf / s1e
    orca_ratio = spec['orca_somf'] / spec['orca_1e']
    scaled = ssomf * spec['orca_1e'] / s1e
    out = dict(
        system=name,
        nao=int(mol.nao_nr()), nelec=int(mol.nelectron),
        nelecas=[int(x) for x in mc.nelecas], ncas=int(mc.ncas),
        ncore=int(mc.ncore), nstates=nstates,
        cas_energy=float(mc.e_tot),
        scalar_gap_cm=float((es[1] - es[0]) * CM),
        density_trace=[float(x) for x in tra],
        density_state_diff=float(np.abs(dm_states[0] - dm_states[1]).max()),
        ao_check=dict(explicit_vs_direct_maxdiff=ao_diff,
                      operator_maxabs=ao_scale,
                      relative=ao_diff / ao_scale,
                      antisym_explicit=ao_antisym,
                      antisym_direct=geo_sym),
        one_electron_cm=s1e,
        somf_cm=ssomf,
        somf_state_density_cm=[x[0] for x in s_state],
        somf_state_density_spread_cm=float(
            max(x[0] for x in s_state) - min(x[0] for x in s_state)),
        hermiticity=[herm1, herms],
        orca=dict(one_electron=spec['orca_1e'], somf=spec['orca_somf']),
        deviation_1e_cm=s1e - spec['orca_1e'],
        deviation_1e_rel=(s1e - spec['orca_1e']) / spec['orca_1e'],
        deviation_somf_cm=ssomf - spec['orca_somf'],
        deviation_somf_rel=(ssomf - spec['orca_somf']) / spec['orca_somf'],
        ratio_this=ratio,
        ratio_orca=orca_ratio,
        ratio_rel=(ratio - orca_ratio) / orca_ratio,
        scale_corrected_somf_cm=scaled,
        scale_corrected_dev_cm=scaled - spec['orca_somf'],
        scale_corrected_dev_rel=(scaled - spec['orca_somf']) / spec['orca_somf'],
        wall_s=float(time.time() - t0),
    )
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', default='oh,of')
    ap.add_argument('--out', default='results_somf.json')
    args = ap.parse_args()
    results = {}
    for name in args.systems.split(','):
        print(f'===== {name} =====')
        results[name] = run(name.strip(), SYSTEMS[name.strip()])
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('written', args.out)


if __name__ == '__main__':
    main()
