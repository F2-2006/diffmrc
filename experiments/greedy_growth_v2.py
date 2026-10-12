"""t162-v2: exact greedy growth with the adjacent-step stop rule.

The v1 stop (gain < f_stop x the FIRST step) was fooled by
complementarity: the N2 run stopped at a 0.78 kcal step and missed a
1.26 kcal step waiting behind it (a pi-pair partner).  The v2 rule
compares against the PREVIOUS step instead -- m(t) < f_stop * m(t-1)
-- so a rebound in the margin is never a stopping point, and the same
N2 run continues to the global optimum.  Same protocol otherwise.

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

Pricing kernel (2026-10-12): the loop needs CI ENERGIES only; the
inherited jax chain's XLA compile explodes with system size (CO: one
candidate priced for more than 19 core-hours without finishing, vs
~0.6 for N2's whole run).  ``--kernel pyscf`` computes a plain-pyscf
CASCI value for the same window in seconds; the jax kernel stays the
default for the recorded semantics and the step-1 proxy arena.

Usage (101): JAX_PLATFORMS=cpu python greedy_growth_v2.py [--sys CO]
             [--kernel pyscf] [--no-proxy]
"""
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from boundary_coupling_probe import SYS, build, energy_at  # noqa: E402

H2K = 627.5094740631

# name: (nele, nact_max, f_stop)
RUNS = {'N2': (2, 6, 0.1), 'CO': (6, 10, 0.1)}


def build_pyscf(name):
    """The same molecule on the plain-pyscf side (pricing kernel 'pyscf')."""
    from pyscf import gto as pgto, scf as pscf

    atom, basis, nele, nact = SYS[name]
    mol = pgto.M(atom=atom, basis=basis, charge=0, spin=0, verbose=0)
    mf = pscf.RHF(mol)
    mf.kernel()
    ncore = (mol.nelectron - nele) // 2
    return mol, mf, ncore, nact, mol.nao_nr()


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


def run_sys(name, kernel='jax', proxy=True):
    nele, nact_max, f_stop = RUNS[name]
    if kernel == 'pyscf':
        mol_p, mf_p, ncore, nact0, nmo = build_pyscf(name)
        from pyscf import fci as pfci
        from pyscf import mcscf as pmcscf

        def price(Cmat, nact_):
            mf2 = mf_p.copy()
            mf2.mo_coeff = np.asarray(Cmat)
            mc = pmcscf.CASCI(mf2, nact_, nele)
            mc.fcisolver = pfci.direct_spin1.FCI(mol_p)
            mc.fcisolver.spin = 0
            mc.fcisolver.conv_tol = 1e-10
            mc.fcisolver.max_cycle = 500
            return float(mc.kernel()[0])

        C = np.asarray(mf_p.mo_coeff)
        h_ = e_ = None                      # jax-side objects not needed
    else:
        mol, C0, h_, e_, ncore, nact0, nmo = build(name)

        def price(Cmat, nact_):
            return float(energy_at(
                Cmat, h_, e_, ncore, nact_, nele, nmo,
                jnp.zeros(len(orbital_pairs(ncore, nact_, nmo)))))

        C = C0
    print(f'=== {name}: start CAS({nele},{nact0}), grow to at most '
          f'{nact_max}  [{kernel} kernel] ===', flush=True)
    nact = nact0
    orig = list(range(nmo))            # orig[i] = column i's initial index
    steps = []
    proxy_vs_exact = None
    margin_first = None
    margin_prev = None
    t_start = time.time()
    while nact < nact_max:
        vir_lo = ncore + nact
        E_cur = price(C, nact)
        drops = {}
        for v_pos in range(vir_lo, nmo):
            v_abs = v_pos
            perm = (list(range(vir_lo)) + [v_abs]
                    + list(range(vir_lo, v_abs))
                    + list(range(v_abs + 1, nmo)))
            C1 = C[:, perm]
            t_v = time.time()
            E_add = price(C1, nact + 1)
            drops[v_pos] = E_cur - E_add
            print(f'   step {len(steps) + 1} candidate '
                  f'{v_pos - vir_lo + 1}/{nmo - vir_lo}: '
                  f'gain={drops[v_pos] * H2K:+.4f} kcal/mol '
                  f'({time.time() - t_v:.0f}s)', flush=True)
        order = sorted(drops, key=lambda v: -drops[v])
        best_v = order[0]
        prev = margin_prev
        margin_prev = drops[best_v]
        if margin_first is None:
            margin_first = drops[best_v]
        if proxy and kernel == 'jax' and proxy_vs_exact is None:
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
        stop = (margin_prev is not None and drops[best_v] > 0
                and drops[best_v] < f_stop * margin_prev)
        ratio = (f'{drops[best_v] / prev:.3f}x' if prev
                 else 'first step')
        print(f'   step {len(steps) + 1}: +v{orig[best_v]:<3d} '
              f'gain={drops[best_v] * H2K:+.4f} kcal/mol  '
              f'E={E_cur:.6f}  (vs prev {ratio})'
              + ('   [STOP: below f_stop of previous step]'
                 if stop else ''), flush=True)
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
               f_stop=f_stop, nact_max=nact_max, kernel=kernel,
               proxy_arena=bool(proxy and kernel == 'jax'))
    with open(f't162_{name}.json', 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main():
    names = list(RUNS)
    if '--sys' in sys.argv:
        names = sys.argv[sys.argv.index('--sys') + 1].split(',')
    kernel = 'jax'
    if '--kernel' in sys.argv:
        kernel = sys.argv[sys.argv.index('--kernel') + 1]
    if kernel not in ('jax', 'pyscf'):
        raise SystemExit(f'unknown kernel {kernel!r}; use jax or pyscf')
    proxy = '--no-proxy' not in sys.argv
    for n in names:
        run_sys(n, kernel=kernel, proxy=proxy)
    print('t162_DONE', flush=True)


if __name__ == '__main__':
    main()
