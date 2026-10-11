"""t161: boundary coupling -- dE/dtheta on active<->virtual channels.

Pathway 2 of the differentiable-selection programme, mechanism check.
The probe: the chain's E(theta) is the CI energy of the rotated orbitals
(no orbital optimization).  At a converged CASSCF point every
partition-preserving rotation has zero derivative (stationarity), so the
boundary signal exists only in the CASCI reading of the chain -- which
is also this project's primary De protocol.  For each virtual v the
coupling number is max |dE/dtheta| over the channels (a, v), a in the
active block; the candidate ordering is this number, descending.

The mechanism check (the part that makes this an algorithm rather than
a diagnostic): add the top-ranked virtual to the active space and
measure the CASCI energy drop; the drop ordering is predicted by the
gradient ordering.  The check is cheap on purpose -- five additions
(top 3, bottom 2 of the ranking) decide whether the one-backpropagation
ranking carries real information.

Honest scope: the gradient measures the coupling of the boundary
between the two spaces (the derivative of the CI energy along the
rotation that would replace an active orbital by v, evaluated at zero).
It is a local response, not a full re-relaxation cost; the addition
check calibrates exactly that gap.  Literature boundary to verify
before any novelty claim: perturbative subspace selection (CASPT2
P-space, selected-CI growth) uses the same intuition through
amplitudes; this route uses the variational derivative of the
differentiable chain instead.

Usage (101): JAX_PLATFORMS=cpu python boundary_coupling_probe.py
"""
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.driver.nevpt2 import _casscf_solution  # noqa: E402

# name: (atom, basis, nele, nact)
SYS = {
    'N2': ('N 0 0 0; N 0 0 1.10', '6-31g', 2, 2),
    'CO': ('C 0 0 0; O 0 0 1.128', '6-31g', 6, 6),
}


def build(name):
    from pyscfad import gto, scf

    atom, basis, nele, nact = SYS[name]
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    nmo = mol.nao_nr()
    ncore = (mol.nelectron - nele) // 2
    C0 = jnp.asarray(mf.mo_coeff)
    h_ = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    e_ = jnp.asarray(mol.intor('int2e', aosym='s1'))
    return mol, C0, h_, e_, ncore, nact, nmo


def energy_at(C0, h_, e_, ncore, nact, nele, nmo, theta):
    E0c, *_ = _casscf_solution(C0, h_, e_, theta, ncore=ncore, nact=nact,
                               nele=nele, nmo=nmo,
                               pairs=orbital_pairs(ncore, nact, nmo))
    return E0c


def run_sys(name):
    mol, C0, h_, e_, ncore, nact, nmo = build(name)
    nele = SYS[name][2]
    vir_lo = ncore + nact
    pairs = orbital_pairs(ncore, nact, nmo)
    nvir = nmo - vir_lo
    print(f'=== {name}: ncore={ncore} nact={nact} nvir={nvir} '
          f'({len(pairs)} rotation channels) ===', flush=True)

    t0 = time.time()
    g = jax.grad(lambda th: energy_at(C0, h_, e_, ncore, nact, nele, nmo,
                                      th))(jnp.zeros(len(pairs)))
    g = np.asarray(g)
    print(f'   gradient of E over all channel(s) in '
          f'{time.time() - t0:.0f}s', flush=True)

    # av channels: p in the active block, q in the virtual block
    av = [(p, q, k) for k, (p, q) in enumerate(pairs)
          if p >= ncore and q >= vir_lo]
    best = {}
    for p, q, k in av:
        v = q - vir_lo
        val = abs(float(g[k]))
        if val > best.get(v, (0.0, None))[0]:
            best[v] = (val, p - ncore)     # (coupling, active partner)
    ranking = sorted(best, key=lambda v: -best[v][0])
    for v in ranking[:10]:
        print(f'   v+{v:2d}: max|dE/dtheta| = {best[v][0]:.6e} '
              f'(strongest partner a={best[v][1]})', flush=True)

    # ---- mechanism check: add the virtual and measure the CASCI drop ----
    E0 = float(energy_at(C0, h_, e_, ncore, nact, nele, nmo,
                         jnp.zeros(len(pairs))))
    picks = ranking[:3] + ranking[-2:]
    drops = {}
    for v in picks:
        v_abs = vir_lo + v
        # move column v_abs to slot vir_lo (the end of the active block)
        perm = (list(range(vir_lo)) + [v_abs]
                + list(range(vir_lo, v_abs))
                + list(range(v_abs + 1, nmo)))
        C1 = C0[:, perm]
        E1 = float(energy_at(C1, h_, e_, ncore, nact + 1, nele, nmo,
                             jnp.zeros(
                                 len(orbital_pairs(ncore, nact + 1, nmo)))))
        drops[v] = E0 - E1
        print(f'   add v+{v:2d}: CASCI drop = {drops[v] * 627.5094740631:+.4f}'
              f' kcal/mol   (gradient rank '
              f'{ranking.index(v) + 1}/{len(ranking)})', flush=True)

    # rank correlation between the gradient coupling and the measured drop
    n = len(picks)
    rx = np.argsort(np.argsort([best[v][0] for v in picks]))
    ry = np.argsort(np.argsort([drops[v] for v in picks]))
    d2 = ((rx - ry) ** 2).sum()
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    print(f'   rank correlation (spearman, n={n}): rho = {rho:+.3f}',
          flush=True)

    rec = dict(name=name, ncore=ncore, nact=nact, nvir=nvir,
               n_channels=len(pairs),
               coupling={int(v): best[v][0] for v in ranking},
               gradient_rank=[int(v) for v in ranking],
               drops_kcal={int(v): drops[v] * 627.5094740631
                           for v in drops},
               spearman=float(rho))
    with open(f't161_{name}.json', 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main():
    names = list(SYS)
    if '--sys' in sys.argv:
        names = sys.argv[sys.argv.index('--sys') + 1].split(',')
    for n in names:
        run_sys(n)
    print('t161_DONE', flush=True)


if __name__ == '__main__':
    main()
