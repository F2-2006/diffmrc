"""N2 CAS(6,6) default vs ortho class schemes on the saved reference."""
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

ref = sys.argv[1]
schemes = sys.argv[2].split(',') if len(sys.argv) > 2 else ['default', 'ortho']
z = np.load(ref)
args = (np.asarray(z['h1_mo']), np.asarray(z['eri_mo']),
        np.asarray(z['ci']).ravel(), [int(x) for x in z['dets']])
kw = dict(ncore=int(z['ncore']), nact=int(z['nact']), nele=int(z['nele']),
          eps=np.asarray(z['eps'], dtype=float))
opts = {'default': {}, 'keep': dict(active_row='keep'),
        'ortho': dict(class_diagonal='ortho'),
        'ortho_ia': dict(class_diagonal='ortho', v0_ia='class'),
        'block': dict(class_diagonal='block'),
        'block_ia': dict(class_diagonal='block', v0_ia='class')}
out = {}
for name in schemes:
    t0 = time.time()
    e2, bd = caspt2_contracted_e2(*args, **kw, **opts[name])
    print('[%s] total %+.12f ratio %.6f (%.1f s)'
          % (name, e2, e2 / ORCA_D_TOTAL, time.time() - t0), flush=True)
    for k, v in bd.items():
        r = (float(v) / ORCA_D[k]) if ORCA_D[k] else float('nan')
        print('  %-9s %+.12f ratio %8.4f' % (k, float(v), r), flush=True)
    out[name] = dict(total=float(e2),
                     classes={k: float(v) for k, v in bd.items()})
print('JSON ' + json.dumps(out), flush=True)
