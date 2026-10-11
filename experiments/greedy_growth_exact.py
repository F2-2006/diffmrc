"""t162: GG-ASG v1 -- exact greedy growth, with the one-derivative proxy
measured against it.

Background (t161, 2026-10-11): the one-derivative boundary probe was
falsified as a gain predictor on N2 -- the v with zero first derivative
(the symmetry-forbidden group) carried the third-largest addition gain,
because its gain lives in the double-excitation space the first
derivative cannot see.  Algorithm A's v1 therefore grows by the exact
addition gain (small n_vir makes this affordable), and this script is
also the proxy-quality arena: at the first step it computes both the
exact gain ordering and the one-derivative ordering, and reports the
rank agreement.  A cheap proxy that tracks the exact order on several
systems later becomes the large-n_vir route; one that does not, does
not.

The exact greedy loop: W = W0 (user-given); at each step every outside
virtual v is tried by the true CASCI gain E(W) - E(W + v); the best v
enters the active block (column moved to the block end); stop when the
best gain falls below f_stop of the first step's, or at nact_max.

Usage (101): JAX_PLATFORMS=cpu python greedy_growth_exact.py [--sys CO]
"""
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

# name: (nele, nact_max, f_stop)
RUNS = {'N2': (2, 6, 0.1), 'CO': (6, 10, 0.1)}


def proxy_order(C0, h_, e_, ncore, nact, nele, nmo, vir_lo):
    """One-derivative boundary ordering (the t161 probe) at the given C."""
    pairs = orbital_pairs(ncore, nact, nmo)

    def energy(th):
        return energy_at(C0, h_, e_, ncore, nact, nele, nmo, th)

    g = np.asarray(jax.grad(energy)(jnp.zeros(len(pairs))))
    best = {}
    for k, (p, q) in enumerate(pairs):
        if p >= ncore and q >= vir_lo:
            v = q - vir_lo
            best[v] = max(best.get(v, 0.0), abs(float(g[k])))
    return best


def run_sys(name):
    nele, nact_max, f_stop = RUNS[name]
    mol, C0, h_, e_, ncore, nact0, nmo = build(name)
    print(f'=== {name}: start CAS({nele},{nact0}), grow to at most '
          f'{nact_max} ===', flush=True)

    C = C0
    nact = nact0
    orig = list(range(nmo))            # orig[i] = column i's initial index
    steps = []
    proxy_vs_exact = None
    margin_first = None
    t_start = time.time()
    while nact < nact_max:
        vir_lo = ncore + nact
        E_cur = float(energy_at(C, h_, e_, ncore, nact, nele, nmo,
                                jnp.zeros(len(orbital_pairs(ncore, nact,
                                                           nmo)))))
        drops = {}
        for v_pos in range(vir_lo, nmo):
            v_abs = v_pos
            perm = (list(range(vir_lo)) + [v_abs]
                    + list(range(vir_lo, v_abs))
                    + list(range(v_abs + 1, nmo)))
            C1 = C[:, perm]
            E_add = float(energy_at(C1, h_, e_, ncore, nact + 1, nele,
                                    nmo,
                                    jnp.zeros(
                                        len(orbital_pairs(ncore, nact + 1,
                                                          nmo)))))
            drops[v_pos] = E_cur - E_add
        order = sorted(drops, key=lambda v: -drops[v])
        best_v = order[0]
        if margin_first is None:
            margin_first = drops[best_v]
            # proxy arena: exact gains vs one-derivative order, step 1
            prox = proxy_order(C, h_, e_, ncore, nact, nele, nmo, vir_lo)
            exact_rank = {v: i for i, v in enumerate(order)}
            px = [prox.get(v - vir_lo, 0.0) for v in order]
            rx = np.argsort(np.argsort([-x for x in px]))   # 0 = highest
            ry = np.arange(len(order))
            n = len(order)
            rho = 1.0 - 6.0 * float(((rx - ry) ** 2).sum()) / (n * (n * n - 1))
            top1_proxy_pos = int(rx[0])       # proxy rank of the exact top1
            proxy_vs_exact = dict(
                exact_order=[int(orig[v]) for v in order],
                exact_gain_kcal=[drops[v] * H2K for v in order],
                proxy_value=[float(x) for x in px],
                spearman=float(rho), exact_top1_proxy_rank=top1_proxy_pos)
            print(f'   step 1 exact order (kcal/mol): '
                  f'{[round(drops[v] * H2K, 4) for v in order]}', flush=True)
            print(f'   step 1 proxy values: '
                  f'{[f"{x:.2e}" for x in px]}', flush=True)
            print(f'   proxy-vs-exact spearman = {rho:+.3f}; exact top1 '
                  f'has proxy rank {top1_proxy_pos}', flush=True)
        stop = (margin_first is not None
                and drops[best_v] < f_stop * margin_first)
        print(f'   step {len(steps) + 1}: +v{orig[best_v]:<3d} '
              f'gain={drops[best_v] * H2K:+.4f} kcal/mol  '
              f'E={E_cur:.6f}  ({time.time() - t_start:.0f}s)'
              + ('   [STOP: below f_stop margin]' if stop else ''),
              flush=True)
        steps.append(dict(v_orig=int(orig[best_v]),
                          gain_kcal=drops[best_v] * H2K,
                          E_cur=E_cur, nact=nact + 1))
        if stop:
            break
        # move best_v to the end of the active block
        perm = (list(range(vir_lo)) + [best_v]
                + list(range(vir_lo, best_v))
                + list(range(best_v + 1, nmo)))
        C = C[:, perm]
        orig = [orig[i] for i in perm]
        nact += 1
    print(f'   final window CAS({nele},{nact}) started from '
          f'CAS({nele},{nact0})', flush=True)

    rec = dict(name=name, nact0=nact0, nact_final=nact, nele=nele,
               steps=steps, proxy_vs_exact=proxy_vs_exact,
               f_stop=f_stop, nact_max=nact_max)
    with open(f't162_{name}.json', 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main():
    names = list(RUNS)
    if '--sys' in sys.argv:
        names = sys.argv[sys.argv.index('--sys') + 1].split(',')
    for n in names:
        run_sys(n)
    print('t162_DONE', flush=True)


if __name__ == '__main__':
    main()
