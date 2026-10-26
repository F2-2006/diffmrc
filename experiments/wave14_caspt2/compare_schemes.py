"""Compare CASPT2 class schemes on a saved reference.

Runs the release default (VP/VM per-function diagonal, replica row rule),
the same path with every active row kept, and the plain-product basis with
the orthonormalized class diagonal.  An ORCA class table can be supplied
as a JSON file to print ratios.

Usage:
  python compare_schemes.py <ref.npz> [orca_classes.json] [schemes]
  schemes: comma list out of default,keep,ortho (default: all three)
"""
import json
import sys
import time

import numpy as np

from diffmrc.core.caspt2 import caspt2_contracted_e2

SCHEMES = {
    'default': dict(),
    'keep': dict(active_row='keep'),
    'ortho': dict(class_diagonal='ortho'),
    'ortho_ia': dict(class_diagonal='ortho', v0_ia='class'),
    'block': dict(class_diagonal='block'),
    'block_ia': dict(class_diagonal='block', v0_ia='class'),
}


def main(ref, orca_path=None, schemes='default,keep,ortho'):
    z = np.load(ref)
    kw = dict(ncore=int(z['ncore']), nact=int(z['nact']), nele=int(z['nele']))
    args = (np.asarray(z['h1_mo']), np.asarray(z['eri_mo']),
            np.asarray(z['ci']).ravel(), [int(x) for x in z['dets']])
    eps = np.asarray(z['eps'], dtype=float)
    orca = None
    if orca_path:
        with open(orca_path) as f:
            orca = json.load(f)
    print('reference %s  ncore=%d nact=%d nele=%d  ndet=%d' % (
        ref, kw['ncore'], kw['nact'], kw['nele'], len(args[3])))
    results = {}
    for name in schemes.split(','):
        if not name:
            continue
        t0 = time.time()
        total, bd = caspt2_contracted_e2(*args, eps=eps, **kw,
                                         **SCHEMES[name])
        results[name] = dict(total=float(total),
                             classes={k: float(v) for k, v in bd.items()})
        print('\n[%s] total %+.12f  (%.1f s)' % (name, total, time.time() - t0))
        for k, v in bd.items():
            line = '  %-9s %+.12f' % (k, float(v))
            if orca is not None and k in orca and orca[k] != 0.0:
                line += '  ratio %.4f' % (float(v) / orca[k])
            print(line)
        if orca is not None:
            tot = orca.get('_total')
            if tot:
                print('  total ratio %.4f' % (float(total) / tot))
    print('\nJSON ' + json.dumps(results))


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None,
         sys.argv[3] if len(sys.argv) > 3 else 'default,keep,ortho')
