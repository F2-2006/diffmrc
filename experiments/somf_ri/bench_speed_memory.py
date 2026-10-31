#!/usr/bin/env python
"""Wall time and peak memory of four-center SOMF versus RI-SOMF(1X).

Each route runs in a fresh process.  The density is a converged RHF
density; the operator cost does not depend on how P was obtained.

Routes
------
``exact_1x_explicit``   four-center J + one-center K, materialized nao^4
``exact_1x_direct``     four-center J + one-center K, pyscf direct map
``exact_full_explicit`` four-center J + four-center K, materialized
``exact_full_direct``   four-center J + four-center K, direct map
``ri_1x``               RI J + one-center K (no nao^4 array)
``ri_j``                RI J only (isolated cost)
``one_center_k``        one-center K only (isolated cost)

Run (one route per process):
  PYTHONPATH=<pkg>/src python bench_speed_memory.py --system w4 --route ri_1x
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import argparse
import json
import resource
import time

import numpy as np
from pyscf import gto, scf

from diffmrc.core import somf_ri as dsomfri


def water_chain(n, sep=2.8):
    lines = []
    for i in range(n):
        z = i * sep
        lines += [f'O 0 0 {z:.6f}',
                  f'H 0.757 0.587 {z:.6f}',
                  f'H -0.757 0.587 {z:.6f}']
    return '; '.join(lines)


SYSTEMS = {
    'w2': water_chain(2),
    'w4': water_chain(4),
}


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='w4', choices=sorted(SYSTEMS))
    ap.add_argument('--basis', default='cc-pVDZ')
    ap.add_argument('--route', required=True,
                    choices=['exact_1x_explicit', 'exact_1x_direct',
                             'exact_full_explicit', 'exact_full_direct',
                             'ri_1x', 'ri_j', 'one_center_k'])
    ap.add_argument('--aux', default='def2-universal-jkfit')
    args = ap.parse_args()

    mol = gto.M(atom=SYSTEMS[args.system], basis=args.basis,
                unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF did not converge')
    p = np.asarray(mf.make_rdm1())
    nao = mol.nao_nr()
    base_mb = peak_mb()

    t0 = time.time()
    if args.route == 'exact_1x_explicit':
        j = dsomfri.coulomb_exact(mol, p, contract='explicit')
        h = j - 1.5 * dsomfri.exchange_one_center(mol, p)
        naux = None
    elif args.route == 'exact_1x_direct':
        j = dsomfri.coulomb_exact(mol, p, contract='direct')
        h = j - 1.5 * dsomfri.exchange_one_center(mol, p)
        naux = None
    elif args.route == 'exact_full_explicit':
        j = dsomfri.coulomb_exact(mol, p, contract='explicit')
        h = j - 1.5 * dsomfri.exchange_exact(mol, p, contract='explicit')
        naux = None
    elif args.route == 'exact_full_direct':
        j = dsomfri.coulomb_exact(mol, p, contract='direct')
        h = j - 1.5 * dsomfri.exchange_exact(mol, p, contract='direct')
        naux = None
    elif args.route == 'ri_1x':
        factors = dsomfri.coulomb_ri_factors(mol, auxbasis=args.aux)
        j = dsomfri.coulomb_ri_from_factors(factors, p)
        k = dsomfri.exchange_one_center(mol, p)
        h = j - 1.5 * k
        naux = factors.naux
    elif args.route == 'ri_j':
        factors = dsomfri.coulomb_ri_factors(mol, auxbasis=args.aux)
        h = dsomfri.coulomb_ri_from_factors(factors, p)
        naux = factors.naux
    else:
        h = dsomfri.exchange_one_center(mol, p)
        naux = None
    wall = time.time() - t0

    peak = peak_mb()
    out = dict(system=args.system, basis=args.basis, route=args.route,
               nao=int(nao), naux=naux, wall_s=wall, peak_mb=peak,
               base_mb=base_mb, operator_increment_mb=peak - base_mb,
               operator_max=float(np.abs(h).max()),
               four_center_bytes=3 * 8.0 * nao ** 4,
               ri_peak_bytes=(4 * 8.0 * nao ** 2 * naux) if naux else None)
    print(json.dumps(out))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'bench_raw.jsonl'), 'a') as fh:
        fh.write(json.dumps(out) + '\n')


if __name__ == '__main__':
    main()
