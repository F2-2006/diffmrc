"""t163: GG-ASG A2 -- exhaustive six-orbital windows vs the greedy path.

How suboptimal is the exact greedy growth?  On N2/6-31G every
six-orbital window out of the twelve non-core orbitals is enumerated
(C(12,6) = 924 CASCI energies, single jit compile reused across
windows), the global best is located, and the comparison against the
greedy sequence's final window (t162) quantifies the price of
greediness.  A greedy path that lands on the global optimum makes the
algorithm's case strong; a measurable gap bounds the value of any
look-ahead.

The window is a column selection of C0 (any six orthogonal columns --
CASCI depends on the subspace, not on a contiguous layout); core
columns stay fixed.  Single jit over the traced coefficient matrix, so
the 924 evaluations share one compilation.

Usage (101): JAX_PLATFORMS=cpu python greedy_growth_exhaustive.py
"""
import itertools
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from boundary_coupling_probe import build, energy_at  # noqa: E402

H2K = 627.5094740631


def main():
    name = 'N2'
    if '--sys' in sys.argv:
        name = sys.argv[sys.argv.index('--sys') + 1]
    nele = {'N2': 2, 'CO': 6}[name]
    mol, C0, h_, e_, ncore, nact0, nmo = build(name)
    ncand = nmo - ncore
    nwin = nact0 + 4
    n_pairs = len(orbital_pairs(ncore, nwin, nmo))
    print(f'=== {name}: enumerate C({ncand},{nwin}) = '
          f'{len(list(itertools.combinations(range(ncand), nwin)))} '
          f'windows of {nwin} orbitals (core {ncore} fixed) ===',
          flush=True)

    core_cols = list(range(ncore))
    rest = list(range(ncore, nmo))

    # one jit over the traced permutation array: columns are selected by
    # gadget indexing, so all 924 windows share a single compilation (a
    # static window tuple would retrace per window)
    def energy_perm(perm_arr):
        C = C0[:, perm_arr]
        return energy_at(C, h_, e_, ncore, nwin, nele, nmo,
                         jnp.zeros(n_pairs))

    fe = jax.jit(energy_perm)
    results = {}
    t0 = time.time()
    for k, combo in enumerate(itertools.combinations(rest, nwin)):
        perm = np.asarray(core_cols + list(combo)
                          + [c for c in rest if c not in combo])
        E = float(fe(jnp.asarray(perm)))
        results[combo] = E
        if k % 100 == 0:
            print(f'   {k} windows, best so far '
                  f'{min(results.values()):.6f}  ({time.time() - t0:.0f}s)',
                  flush=True)
    best = min(results, key=results.get)
    print(f'   global best window {best}  E={results[best]:.6f}', flush=True)
    top = sorted(results, key=results.get)[:10]
    for combo in top:
        print(f'   top: {combo}  E={results[combo]:.6f}', flush=True)
    # comparison target: the greedy path's final window (column ids in
    # the ORIGINAL C0), read from t162's record when present
    rec = dict(name=name, nwin=nwin, n_windows=len(results),
               best_window=[int(c) for c in best],
               best_energy=results[best],
               top10=[[int(c) for c in w] for w in top],
               energies={' '.join(map(str, w)): results[w] for w in results})
    with open(f't163_{name}.json', 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1)
    print('t163_DONE', flush=True)


if __name__ == '__main__':
    main()
