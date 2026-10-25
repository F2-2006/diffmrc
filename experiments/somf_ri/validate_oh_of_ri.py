#!/usr/bin/env python
"""End-to-end validation of RI-SOMF and the one-center exchange on OH/OF.

Reproduces the SA-CASSCF / SI pipeline of ``experiments/somf/validate_oh_of.py``
and compares five operator routes:

    exact          J(exact) + K(exact)              ORCA SOCFlags 1,4,4,0
    somf_1x        J(exact) + K(one-center)         ORCA SOCFlags 1,2,3,0
    ri_somf_1x     J(RI)    + K(one-center)         ORCA SOCFlags 1,3,3,0
    ri_somf_4x     J(RI)    + K(exact)              ORCA SOCFlags 1,3,4,0
    amfi           J(1c)    + K(one-center)         ORCA SOCFlags 1,4,3,0, maxcenter 1

The ORCA numbers below were produced with ORCA 6.1.1 at the same geometry,
basis and active space (``experiments/somf_ri/orca/*.in``).

Run:
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python validate_oh_of_ri.py \
      --out results_oh_of_ri.json
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
from pyscf import gto, mcscf, scf
from pyscf.data import nist

from diffmrc.core import somf as dsomf
from diffmrc.core import somf_ri as dsomfri

ALPHA2 = float(nist.ALPHA) ** 2
CM = 219474.6313632
SPIN = 0.5 * np.array([[[0, 1], [1, 0]],
                       [[0, -1j], [1j, 0]],
                       [[1, 0], [0, -1]]], dtype=complex)

SYSTEMS = {
    'oh': dict(atom=[('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.8324))],
               spin=1, ncore=1, nact=6),
    'of': dict(atom=[('O', (0.0, 0.0, 0.0)), ('F', (0.0, 0.0, 2.4))],
               spin=1, ncore=2, nact=8),
}

# ORCA 6.1.1, same geometry/basis/CAS, def2/J for the RI routes.
ORCA = {
    'oh': dict(soc1=205.88, somf=128.83, somf_1x=129.31,
               ri_somf_1x=129.31, ri_somf_4x=128.83, amfi=128.44),
    'of': dict(soc1=272.07, somf=176.13, somf_1x=175.74,
               ri_somf_1x=175.74, ri_somf_4x=176.13, amfi=176.42),
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


def reference_state(name, spec):
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
    D = {}
    for i in range(nstates):
        for j in range(nstates):
            D[i, j] = np.asarray(mc.fcisolver.trans_rdm1(
                ci[i], ci[j], spec['nact'], mc.nelecas))
    return dict(mol=mol, es=es, C=C, Ca=Ca, dm_sa=dm_sa, D=D,
                nstates=nstates, cas_energy=float(mc.e_tot))


def run(name, spec):
    t0 = time.time()
    ref = reference_state(name, spec)
    mol, es, Ca, dm_sa = ref['mol'], ref['es'], ref['Ca'], ref['dm_sa']
    D, nstates = ref['D'], ref['nstates']

    h1 = dsomf.somf_matrix(mol, Ca, None, mode='one_electron')
    s1, h1_herm = si_splitting(es, transition_tensor(D, h1, nstates),
                               nstates)

    # Four-center reference from core.somf.
    h_ref = dsomf.somf_matrix(mol, Ca, dm_sa, mode='somf')
    s_ref, ref_herm = si_splitting(es, transition_tensor(D, h_ref, nstates),
                                   nstates)

    # AO-level diagnostics.
    j_exact = dsomfri.coulomb_exact(mol, dm_sa)
    k_exact = dsomfri.exchange_exact(mol, dm_sa)
    k_1c = dsomfri.exchange_one_center(mol, dm_sa)
    j_1c = dsomfri.coulomb_one_center(mol, dm_sa)
    j_ri = dsomfri.coulomb_ri(mol, dm_sa)
    ao = dict(
        j_max=float(np.abs(j_exact).max()),
        k_max=float(np.abs(k_exact).max()),
        j_ri_absmax=float(np.abs(j_ri - j_exact).max()),
        j_ri_rel=float(np.abs(j_ri - j_exact).max()
                       / np.abs(j_exact).max()),
        j_1c_absmax=float(np.abs(j_1c - j_exact).max()),
        j_1c_rel=float(np.abs(j_1c - j_exact).max()
                       / np.abs(j_exact).max()),
        k_1c_absmax=float(np.abs(k_1c - k_exact).max()),
        k_1c_rel=float(np.abs(k_1c - k_exact).max()
                       / np.abs(k_exact).max()),
        antisym=dict(
            j_exact=float(np.abs(j_exact + j_exact.transpose(0, 2, 1)).max()),
            j_ri=float(np.abs(j_ri + j_ri.transpose(0, 2, 1)).max()),
            j_1c=float(np.abs(j_1c + j_1c.transpose(0, 2, 1)).max()),
            k_exact=float(np.abs(k_exact + k_exact.transpose(0, 2, 1)).max()),
            k_1c=float(np.abs(k_1c + k_1c.transpose(0, 2, 1)).max()),
        ),
    )

    routes = {}
    for label, mode in (
            ('somf', 'somf'),
            ('somf_1x', 'somf_1x'),
            ('ri_somf_1x', 'ri_somf_1x'),
            ('ri_somf_4x', None),
            ('amfi', 'amfi')):
        if mode is None:
            h = dsomfri.somf_ri_matrix(mol, Ca, dm_sa, coulomb='ri',
                                       exchange='full')
        else:
            h = dsomfri.somf_ri_matrix(mol, Ca, dm_sa, mode=mode)
        split, herm = si_splitting(es, transition_tensor(D, h, nstates),
                                   nstates)
        routes[label] = dict(splitting_cm=split, hermiticity=herm)

    # Auxiliary-basis convergence of the RI Coulomb term (splitting level).
    aux_scan = {}
    for auxbasis in ('def2-SVP-RI', 'def2-TZVP-RI', 'def2-QZVP-RI',
                     'def2-universal-jkfit'):
        try:
            h = dsomfri.somf_ri_matrix(mol, Ca, dm_sa,
                                       coulomb='ri', exchange='one_center',
                                       auxbasis=auxbasis)
        except Exception as exc:
            aux_scan[auxbasis] = dict(error=f'{type(exc).__name__}: {exc}')
            continue
        split, _ = si_splitting(es, transition_tensor(D, h, nstates),
                                nstates)
        j = dsomfri.coulomb_ri(mol, dm_sa, auxbasis=auxbasis)
        aux_scan[auxbasis] = dict(
            splitting_cm=split,
            j_absmax_vs_exact=float(np.abs(j - j_exact).max()),
            j_rel_vs_exact=float(np.abs(j - j_exact).max()
                                 / np.abs(j_exact).max()),
            naux=int(dsomfri.coulomb_ri_factors(
                mol, auxbasis=auxbasis).naux))

    out = dict(
        system=name,
        nao=int(mol.nao_nr()),
        nelecas=[int(x) for x in (spec['nact'], mol.nelectron
                                  - 2 * spec['ncore'])],
        ncore=int(spec['ncore']),
        cas_energy=ref['cas_energy'],
        one_electron_cm=s1,
        orca=ORCA[name],
        routes=routes,
        deviations_vs_exact_somf={
            k: v['splitting_cm'] - s_ref for k, v in routes.items()},
        deviations_vs_orca={
            k: v['splitting_cm'] - ORCA[name][k] for k, v in routes.items()},
        relative_vs_orca={
            k: (v['splitting_cm'] - ORCA[name][k]) / ORCA[name][k]
            for k, v in routes.items()},
        ao_diagnostics=ao,
        aux_basis_scan=aux_scan,
        hermiticity=dict(one_electron=h1_herm, reference=ref_herm),
        wall_s=float(time.time() - t0),
    )
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', default='oh,of')
    ap.add_argument('--out', default='results_oh_of_ri.json')
    args = ap.parse_args()
    results = {}
    for name in args.systems.split(','):
        print(f'===== {name} =====')
        results[name.strip()] = run(name.strip(), SYSTEMS[name.strip()])
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('written', args.out)


if __name__ == '__main__':
    main()
