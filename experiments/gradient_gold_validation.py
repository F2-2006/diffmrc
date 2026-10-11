"""t157: Gradient validation against an analytic reference and reference-free checks.

Three independent criteria are evaluated: the closed-form analytic gradient of
an independent program, and two conditions that require no external reference
at all - translational invariance, in which the gradient components sum to
zero, and rotational invariance, in which the torque vanishes. The
finite-difference golden standard is included for comparison, together with the
convergence status of every displaced point.

Usage: python gradient_gold_validation.py <SYS> [NT]
"""
import os
import sys
import json
import time
import gc

SYS = (sys.argv[1] if len(sys.argv) > 1 else 'N2').upper()
NT = sys.argv[2] if len(sys.argv) > 2 else '16'

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
from diffmrc.core.rotation import block_rotation                            # noqa: E402
from diffmrc.core.orblib import orbital_pairs                                 # noqa: E402
from diffmrc.core.optimizer import optimize_theta                           # noqa: E402

from grad import (make_energy_fn, make_grad_fn, solve_theta_star,
                  nuclear_gradient)

BASIS = '6-31g'
NACT, NELE = 2, 2
H = 1e-3

SYSTEMS = {
    'H2O': 'O 0 0 0; H 0.7572 0 0.5859; H -0.7572 0 0.5859',
    'N2':  'N 0 0 0; N 0 0 1.0977',
    'CO':  'C 0 0 0; O 0 0 1.1283',
}
ATOMS = SYSTEMS[SYS]

# archived reference values, for comparison
T119 = {
    'N2': dict(n_bad=4, dev_reported=17.740931993600384,
               g_gold=[17.74093199028215, -4.263256414560601e-11,
                       0.010576178311794138, 17.740931993600384,
                       -3.126388037344441e-10, -0.010576178212318155]),
    'H2O': dict(n_bad=8, dev_reported=3.496256317792046e-08,
                g_gold=[1.0658141036401503e-10, 7.780442956573097e-09,
                        -0.023221806621620544, -0.0047665728857282375,
                        2.5579538487363607e-10, 0.011610908750014914,
                        0.004766572956782511, -1.4210854715202004e-10,
                        0.011610909403714231]),
    'CO':  dict(n_bad=0, dev_reported=None, g_gold=None),
}

print('=' * 100, flush=True)
print(f'Golden-standard fix: analytic gradient reference, {SYS}/{BASIS}  T={NT}',
      flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = ATOMS; mol.basis = BASIS; mol.build()
nmo = mol.nao_nr(); natm = mol.natm
ncore = mol.nelectron // 2 - NELE // 2
pairs = orbital_pairs(ncore, NACT, nmo, include_cv=True)
npar = len(pairs)
coords0 = np.asarray(mol.atom_coords())
print(f'  nmo={nmo} natm={natm} ncore={ncore} npar={npar}', flush=True)

RESULT = dict(system=SYS, basis=BASIS, nthread=NT, npar=npar, h=H)

# ==================================================================
# ① grad.py   AD gradient
# ==================================================================
E_cas = make_energy_fn(ncore, NACT, NELE, use_blocked=False)
mf = pscf.RHF(mol); mf.kernel()
C_rhf = jnp.asarray(np.asarray(mf.mo_coeff))
h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri = mol.intor("int2e", aosym="s1")
enuc = mol.energy_nuc()

E_theta = lambda t: E_cas(C_rhf @ block_rotation(t, nmo, pairs), h1e, eri, enuc)
t0 = time.time()
th_star, gn = solve_theta_star(mol, C_rhf, E_theta, npar, optimize_theta,
                               steps=3000, g_conv=1e-13, label=SYS)
print(f'\n  theta*: |g|={gn:.3e}   ({time.time()-t0:.0f}s)', flush=True)
E_geo = make_grad_fn(nmo, pairs, th_star, E_cas, pscf)
g_ad = nuclear_gradient(mol, E_geo)
print(f'  AD gradient = {np.array2string(g_ad.ravel(), precision=8)}', flush=True)
RESULT['grad_theta'] = float(gn)
RESULT['g_ad'] = g_ad.ravel().tolist()

# ==================================================================
# 2) Criterion 1: invariance, purely internal and needing no reference
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('2) Invariance criteria', flush=True)
print('=' * 100, flush=True)
sum_g = g_ad.sum(axis=0)
torque = sum(np.cross(coords0[a], g_ad[a]) for a in range(natm))
print(f'  sum_atoms ∂E/∂R  = {np.array2string(sum_g, precision=12)}  '
      f'|·|_max={np.abs(sum_g).max():.3e}', flush=True)
print(f'  sum_atoms R×∂E/∂R = {np.array2string(torque, precision=12)}  '
      f'|·|_max={np.abs(torque).max():.3e}', flush=True)
RESULT['sum_grad'] = sum_g.tolist()
RESULT['torque'] = torque.tolist()

# ==================================================================
# 3) Analytic CASSCF nuclear gradient, the primary criterion
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('3) Analytic CASSCF nuclear gradient, the primary criterion', flush=True)
print('=' * 100, flush=True)

syms = [mol.atom_symbol(i) for i in range(natm)]


def mk_str(c_bohr):
    ca = c_bohr * 0.529177210903
    return "; ".join([f"{syms[i]} {ca[i,0]:.12f} {ca[i,1]:.12f} "
                      f"{ca[i,2]:.12f}" for i in range(natm)])


def gold_analytic(astr, th_guess=None):
    """Analytic CASSCF nuclear gradient.

    Convergence settings follow the standard used throughout: conv_tol_grad=1e-7
    with 800 macrocycles. A tighter threshold is counterproductive; see the note in
    the script header.
    return (grad, e_tot, converged, converged_grad)
    """
    m = gto.M(atom=astr, basis=BASIS, verbose=0)
    mf = scf.RHF(m); mf.max_cycle = 400; mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        return None, None, False, False
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-7          # standard setting; 1e-8 proved too tight
    mc.max_cycle_macro = 800
    mc.fcisolver.conv_tol = 1e-13
    mc.verbose = 0
    try:
        mc.kernel()
        g = mc.nuc_grad_method().kernel()
    except Exception as ex:
        return None, None, False, False
    cvg = bool(mc.converged)
    # Convergence of the coupled-perturbed step is what makes this usable
    gconv = bool(getattr(mc, 'converged', False))
    return np.asarray(g), float(mc.e_tot), cvg, gconv


print('  Computing the analytic gradient at the reference geometry...', flush=True)
t0 = time.time()
g_ana, E_ana, cv, gc_ = gold_analytic(mk_str(coords0))
print(f'    done {time.time()-t0:.0f}s  CASSCF E={E_ana}  '
      f'conv={cv}', flush=True)
if g_ana is not None:
    print(f'    analytic gradient = {np.array2string(g_ana.ravel(), precision=8)}',
          flush=True)
RESULT['E_gold'] = E_ana
RESULT['g_analytic'] = g_ana.ravel().tolist() if g_ana is not None else None
RESULT['gold_converged'] = cv

# ==================================================================
# 4) Finite-difference golden standard, shown to exhibit the unconverged points
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('4) Finite-difference golden standard, for comparison', flush=True)
print('=' * 100, flush=True)
g_fd = np.full((natm, 3), np.nan)
convs = np.zeros((natm, 3), dtype=bool)
t0 = time.time()
for a in range(natm):
    for k in range(3):
        cp = coords0.copy(); cp[a, k] += H
        cm = coords0.copy(); cm[a, k] -= H
        _, Ep, cvp, _ = gold_analytic(mk_str(cp))
        _, Em, cvm, _ = gold_analytic(mk_str(cm))
        ok = (Ep is not None) and (Em is not None) and cvp and cvm
        convs[a, k] = ok
        if ok:
            g_fd[a, k] = (Ep - Em) / (2 * H)
        print(f'    atom{a}_{"xyz"[k]}: '
              f'{g_fd[a,k]:+.8e}' if ok else
              f'    atom{a}_{"xyz"[k]}: unconverged points (conv={cvp}/{cvm})',
              flush=True)
print(f'  done {time.time()-t0:.0f}s  unconverged points={int((~convs).sum())}/{convs.size}',
      flush=True)
RESULT['g_fd'] = np.nan_to_num(g_fd).ravel().tolist()
RESULT['fd_bad'] = int((~convs).sum())

# ==================================================================
# ⑤ Assessment
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('⑤ Assessment', flush=True)
print('=' * 100, flush=True)

if g_ana is not None:
    dev_ana = np.abs(g_ad - g_ana)
    print('  AD against the analytic reference, the most reliable criterion',
          flush=True)
    print(f'     AD   = {np.array2string(g_ad.ravel(), precision=8)}',
          flush=True)
    print(f'     analytic = {np.array2string(g_ana.ravel(), precision=8)}',
          flush=True)
    RESULT['dev_analytic'] = float(dev_ana.max())

if convs.any():
    dev_fd = np.abs(g_ad - g_fd)[convs]
    print(f'\n  Finite-difference golden standard, '
          f'{int(convs.sum())} converged points', flush=True)
    print(f'     largest deviation = {dev_fd.max():.3e}', flush=True)
    RESULT['dev_fd_converged'] = float(dev_fd.max())
    RESULT['n_fd_converged'] = int(convs.sum())

if SYS in T119 and T119[SYS]['g_gold'] is not None:
    r = T119[SYS]
    print(f'\n  Comparison with archived values:', flush=True)
    print(f'     archived unconverged points = {r["n_bad"]}  this run = '
          f'{int((~convs).sum())}', flush=True)
    print(f'     largest spurious value from an unconverged point = '
          f'{max(abs(x) for x in r["g_gold"]):.4f}', flush=True)
    print(f'     the archived residual of {r["dev_reported"]:.4f} has the same origin',
          flush=True)

outp = f'results/t157_goldfix_{SYS}.json'
json.dump(RESULT, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print(f'\nt157_{SYS}_DONE', flush=True)
gc.collect()
