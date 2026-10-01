"""Regression test for the production gradient module.

Runs the gradient through the public interface of grad.py on three systems and
compares against a golden standard. Only converged reference points are
compared: an unconverged orbital optimization enters a central difference as an
intermediate value rather than as a stationary point, which inflates the
apparent deviation by orders of magnitude. The number of such points is
reported alongside the residual.

Usage: python t155_grad_regression.py [NT]
"""
import os
import sys
import json
import time
import gc

NT = sys.argv[1] if len(sys.argv) > 1 else '16'
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np                                               # noqa: E402
import jax; jax.config.update("jax_enable_x64", True)             # noqa: E402
from jax import numpy as jnp                                     # noqa: E402
import pyscfad                                                    # noqa: E402
from pyscfad import gto as pgto, scf as pscf                      # noqa: E402
from pyscf import gto, scf, mcscf                                # noqa: E402
from diffmrc.core.orblib import orbital_pairs                                 # noqa: E402
from diffmrc.core.optimizer import optimize_theta                           # noqa: E402

# Exercised only through the public interface of the production module
from grad import (make_energy_fn, make_grad_fn, solve_theta_star,
                  nuclear_gradient)

BASIS = '6-31g'
NACT, NELE = 2, 2
H = 1e-3                             # central-difference step used for the golden standard
TOL = 1e-5                           # regression tolerance; the reference implementation reaches 3.5e-8 to 1.0e-6

SYSTEMS = {
    'H2O': 'O 0 0 0; H 0.7572 0 0.5859; H -0.7572 0 0.5859',
    'N2':  'N 0 0 0; N 0 0 1.0977',
    'CO':  'C 0 0 0; O 0 0 1.1283',
}
# regression baseline for the three systems
BASELINE = {'CO': 8.1e-07, 'H2O': 3.5e-08, 'N2': 1.0e-06}

print('=' * 100, flush=True)
print(f'Gradient regression test for grad.py  T={NT}', flush=True)
print('=' * 100, flush=True)

RESULT = dict(basis=BASIS, nthread=NT, h=H, tol=TOL, systems={})
all_ok = True

for SYS, ATOMS in SYSTEMS.items():
    print('\n' + '=' * 100, flush=True)
    print(f'  {SYS}/{BASIS}', flush=True)
    print('=' * 100, flush=True)

    mol = pgto.Mole(); mol.atom = ATOMS; mol.basis = BASIS; mol.build()
    nmo = mol.nao_nr(); natm = mol.natm
    ncore = mol.nelectron // 2 - NELE // 2
    pairs = orbital_pairs(ncore, NACT, nmo, include_cv=True)
    npar = len(pairs)
    print(f'  nmo={nmo} natm={natm} ncore={ncore} npar={npar}', flush=True)

    # ---------- 1) theta* through the production module ----------
    E_cas = make_energy_fn(ncore, NACT, NELE, use_blocked=False)
    mf = pscf.RHF(mol); mf.kernel()
    C_rhf = jnp.asarray(np.asarray(mf.mo_coeff))
    h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    enuc = mol.energy_nuc()

    from diffmrc.core.rotation import block_rotation
    E_theta = lambda t: E_cas(C_rhf @ block_rotation(t, nmo, pairs),
                              h1e, eri, enuc)
    t0 = time.time()
    th_star, gn = solve_theta_star(mol, C_rhf, E_theta, npar,
                                   optimize_theta, steps=2000,
                                   g_conv=1e-12, label=SYS)
    print(f'  theta*: |g|={gn:.3e}   ({time.time()-t0:.0f}s)', flush=True)
    if gn > 1e-6:
        print(f'  |g| is large; the result may not be trustworthy', flush=True)

    # ---------- 2) Gradient through the production module ----------
    E_geo = make_grad_fn(nmo, pairs, th_star, E_cas, pscf)
    t0 = time.time()
    g_ad = nuclear_gradient(mol, E_geo)          # module API
    print(f'  AD gradientdone  {time.time()-t0:.0f}s', flush=True)
    print(f'    g = {np.array2string(g_ad.ravel(), precision=8)}', flush=True)

    # ---------- 3) Golden standard: pointwise finite differences ----------
    coords0 = np.asarray(mol.atom_coords())
    syms = [mol.atom_symbol(i) for i in range(natm)]

    def mk_str(c_bohr):
        ca = c_bohr * 0.529177210903
        return "; ".join([f"{syms[i]} {ca[i,0]:.12f} {ca[i,1]:.12f} "
                          f"{ca[i,2]:.12f}" for i in range(natm)])

    def gold_e(astr):
        """Strict CASSCF. The thresholds were tightened after an earlier run
        produced nine of eighteen unconverged points."""
        m = gto.M(atom=astr, basis=BASIS, verbose=0)
        mf = scf.RHF(m); mf.max_cycle = 400; mf.conv_tol = 1e-12
        mf.kernel()
        if not mf.converged:
            return None, False
        mc = mcscf.CASSCF(mf, NACT, NELE)
        mc.conv_tol = 1e-13; mc.conv_tol_grad = 1e-8
        mc.max_cycle_macro = 2000; mc.fcisolver.conv_tol = 1e-14
        mc.verbose = 0
        try:
            mc.kernel()
        except Exception:
            return None, False
        return float(mc.e_tot), bool(mc.converged)

    print(f'  3) Golden-standard finite differences, {2*natm*3} points...',
          flush=True)
    t0 = time.time()
    g_gold = np.full((natm, 3), np.nan)
    convs = np.zeros((natm, 3), dtype=bool)
    for a in range(natm):
        for k in range(3):
            cp = coords0.copy(); cp[a, k] += H
            cm = coords0.copy(); cm[a, k] -= H
            Ep, cvp = gold_e(mk_str(cp))
            Em, cvm = gold_e(mk_str(cm))
            ok_pt = (Ep is not None) and (Em is not None) and cvp and cvm
            convs[a, k] = ok_pt
            if ok_pt:
                g_gold[a, k] = (Ep - Em) / (2 * H)
    nbad = int((~convs).sum())
    print(f'    done in {time.time()-t0:.0f}s  unconverged points = {nbad}/'
          f'{convs.size}', flush=True)

    # ---------- 4) Assessment over converged points only ----------
    dev_map = np.abs(g_ad - g_gold)
    dev_valid = dev_map[convs] if convs.any() else np.array([np.inf])
    dev = float(dev_valid.max())
    base = BASELINE.get(SYS, np.nan)
    ok = dev < TOL
    all_ok = all_ok and ok
    print(f'\n  largest deviation over converged points = {dev:.3e}   '
          f'(baseline {base:.1e})   {"passed" if ok else "failed"}', flush=True)
    if nbad:
        dev_bad = dev_map[~convs]
        print(f'     (apparent deviations at unconverged points = '
              f'{np.array2string(dev_bad, precision=2)}'
              f' -- an unconverged CASSCF is an intermediate value, so these\n'
              f'              numbers are not usable)', flush=True)
    # Invariance check, which needs no external reference
    print(f'    invariance check: |sum_atoms ∂E/∂R| = '
          f'{np.abs(g_ad.sum(0)).max():.2e}  '
          f'|sum R×∂E/∂R| = '
          f'{np.abs(sum(np.cross(coords0[i], g_ad[i]) for i in range(natm))).max():.2e}',
          flush=True)
    RESULT['systems'][SYS] = dict(
        npar=npar, grad_theta=float(gn), n_bad_gold=nbad,
        n_converged=int(convs.sum()),
        g_ad=g_ad.ravel().tolist(),
        g_gold=np.nan_to_num(g_gold).ravel().tolist(),
        convs=convs.ravel().tolist(),
        dev_converged=dev, baseline=float(base) if base == base else None,
        pass_=ok)

if all_ok:
    print(f'\nall systems passed (tolerance {TOL:.0e}, converged points only)',
          flush=True)
else:
    print(f'\nFAIL: at least one system did not pass', flush=True)

outp = 'results/t155_grad_regression.json'
json.dump(RESULT, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print(f'\nt155_{"OK" if all_ok else "FAIL"}_DONE', flush=True)
gc.collect()
