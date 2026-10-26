"""Reproduce the N2 CAS(6,6) default-path class table.

Loads the saved semicanonical reference and calls the release default
path.  Prints the per-class breakdown against the ORCA CASPT2-D table.
"""
import json
import sys
import time

import numpy as np

from diffmrc.core.caspt2 import caspt2_contracted_e2

ORCA_D = {
    'V0_ijab': -0.008032535, 'Vm1_iab': -0.016383398,
    'Vm2_ab': -0.012732384, 'V1_ija': -0.001224040,
    'V2_ij': -0.007510645, 'V0_ia': -0.020418190,
    'Vm1_a': -0.003608944, 'V1_i': -0.001610598,
}
ORCA_D_TOTAL = -0.071520733

if len(sys.argv) < 2:
    raise SystemExit('usage: run_n2_default.py <n2_ref.npz>')
z = np.load(sys.argv[1])
h1 = np.asarray(z['h1_mo'])
eri = np.asarray(z['eri_mo'])
ci = np.asarray(z['ci']).ravel()
dets = [int(x) for x in z['dets']]
ncore, nact, nele = int(z['ncore']), int(z['nact']), int(z['nele'])
eps = np.asarray(z['eps'], dtype=float)

t0 = time.time()
e2, bd = caspt2_contracted_e2(h1, eri, ci, dets, ncore=ncore, nact=nact,
                              nele=nele, eps=eps)
print('total %+.12f  ratio %.6f  (%.1f s)' % (e2, e2 / ORCA_D_TOTAL,
                                              time.time() - t0))
out = {}
for k, v in bd.items():
    out[k] = float(v)
    print('  %-9s %+.12f  ratio %.6f' % (k, float(v), float(v) / ORCA_D[k]))
print(json.dumps({'total': float(e2), 'classes': out}))
