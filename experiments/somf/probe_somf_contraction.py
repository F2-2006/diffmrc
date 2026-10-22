#!/usr/bin/env python
"""Probe 2: determine the explicit int2e_p1vxp1 contraction that reproduces
pyscf's SOMF direct (J) and exchange (K) maps, so that diffmrc can implement
the operator with plain numpy einsum and a documented index convention.

Run with pyscf available: JAX_PLATFORMS=cpu python probe_somf_contraction.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.prop.gtensor.uhf import get_jk


def build():
    mol = gto.M(atom=[('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.8324))],
                basis='6-31g', unit='Bohr', spin=1, verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASSCF(mf, 6, 7)
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.state_average_([0.5, 0.5])
    mc.fcisolver.nroots = 2
    mc.kernel()
    return mol, mc


mol, mc = build()
P = np.asarray(mc.make_rdm1())
S = mol.intor('int1e_ovlp')
print('trace(P S) =', np.einsum('pq,pq->', P, S))

vj, vk = get_jk(mol, (0.5 * P, 0.5 * P))
j_dir = vj[0] + vj[1]
k_dir = vk[0] + vk[1]
print('j_dir norm', np.linalg.norm(j_dir), 'k_dir norm', np.linalg.norm(k_dir))

g = mol.intor('int2e_p1vxp1', aosym='s1')
print('g shape', g.shape, 'norm', np.linalg.norm(g))
g = g.reshape(3, mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())

# symmetry probes for component 0
g0 = g[0]
print('g0 - g0^T(bra)  :', np.abs(g0 - g0.transpose(1, 0, 2, 3)).max())
print('g0 - g0^T(ket)  :', np.abs(g0 - g0.transpose(0, 1, 3, 2)).max())
print('g0 + g0(swap b,k):', np.abs(g0 + g0.transpose(2, 3, 0, 1)).max())
print('g0 + g0(1<->3)  :', np.abs(g0 + g0.transpose(0, 3, 2, 1)).max())

cands_j = {
    'pqrs': np.einsum('rs,kpqrs->kpq', P, g),
    'pqsr': np.einsum('rs,kpqsr->kpq', P, g),
    'rspq': np.einsum('rs,krspq->kpq', P, g),
}
print('--- J candidates vs direct ---')
for k, v in cands_j.items():
    print(f'  J[{k}] maxdiff {np.abs(v - j_dir).max():.3e}  norm {np.linalg.norm(v):.6e}')

cands_k = {
    'prsq': np.einsum('rs,kprsq->kpq', P, g),
    'psrq': np.einsum('rs,kpsrq->kpq', P, g),
    'sqpr': np.einsum('rs,ksqpr->kpq', P, g),
    'qprs': np.einsum('rs,kqprs->kpq', P, g),
    'rpsq': np.einsum('rs,krpsq->kpq', P, g),
    'qpsr': np.einsum('rs,kqpsr->kpq', P, g),
}
print('--- K candidates vs direct ---')
for k, v in cands_k.items():
    print(f'  K[{k}] maxdiff {np.abs(v - k_dir).max():.3e}  norm {np.linalg.norm(v):.6e}')

# pair sums for the exchange part (ORCA: (pr|g|sq) + (sq|g|pr))
print('--- K pair sums vs direct ---')
for a, b in (('prsq', 'sqpr'), ('prsq', 'psrq'), ('prsq', 'rpsq'),
             ('prsq', 'qprs')):
    v = cands_k[a] + cands_k[b]
    print(f'  K[{a}+{b}] maxdiff {np.abs(v - k_dir).max():.3e}  '
          f'norm {np.linalg.norm(v):.6e}')

h2 = cands_j['pqrs'] - 1.5 * (cands_k['prsq'] + cands_k['sqpr'])
h2_dir = j_dir - 1.5 * k_dir
print('antisym of explicit h2:', np.abs(h2 + h2.transpose(0, 2, 1)).max())
print('explicit vs direct h2 :', np.abs(h2 - h2_dir).max())
