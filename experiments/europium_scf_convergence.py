"""t156: Convergence strategies for the europium atom.

The europium ground state has a half-filled f shell, and the restricted
open-shell SCF converges poorly from standard initial guesses. This script
compares convergence strategies - cycle limits, damping, level shifting, DIIS
suppression, and Newton-Raphson - and selects among the converged solutions by
the variational principle.

Usage: python europium_scf_convergence.py [NT]
"""
import os
import sys
import json
import time
import gc

NT = sys.argv[1] if len(sys.argv) > 1 else '8'
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np                                               # noqa: E402
from pyscf import gto, scf                                       # noqa: E402

H2K = 627.5094740631
SPIN = 7                         # Eu: [Xe]4f⁷6s^2  =>  2S+1 = 8

print('=' * 100, flush=True)
print(f'Europium atomic end: ROHF convergence diagnostic  T={NT}', flush=True)
print('=' * 100, flush=True)

EBASE = -10839.5724292080        # unconverged reference from an earlier run

RESULTS = {}


def report(tag, mf, t0, extra=''):
    ok = bool(mf.converged)
    dE = (mf.e_tot - EBASE) * H2K
    print(f'  {tag:<34} E={mf.e_tot:.10f}  conv={ok}  '
          f'ncyc={mf.cycles}  {time.time()-t0:.0f}s  Δvs_base={dE:+.2f} kcal'
          f'{extra}', flush=True)
    RESULTS[tag] = dict(E=float(mf.e_tot), conv=ok,
                        ncyc=int(mf.cycles or -1),
                        t=round(time.time() - t0, 1))
    return ok


def make_mol():
    return gto.M(atom='Eu 0 0 0', basis='ano', spin=SPIN, verbose=0)


# ==================================================================
# 1) Baseline, reproducing the failure described above
# ==================================================================
print('\n  1) Baseline, reproducing the earlier failure', flush=True)
m = make_mol()
t0 = time.time()
mf = scf.ROHF(m).x2c1e()
mf.init_guess = 'atom'
mf.max_cycle = 250
mf.conv_tol = 1e-9
mf.verbose = 0
try:
    mf.kernel()
    report('① baseline(max_cycle=250)', mf, t0)
except Exception as ex:
    print(f'  ✗ {type(ex).__name__}: {str(ex)[:80]}', flush=True)

# ==================================================================
# 2) A larger cycle limit
# ==================================================================
print('\n  2) Larger cycle limit, max_cycle=1000', flush=True)
m = make_mol()
t0 = time.time()
mf = scf.ROHF(m).x2c1e()
mf.init_guess = 'atom'
mf.max_cycle = 1000
mf.conv_tol = 1e-9
mf.verbose = 0
try:
    mf.kernel()
    C2 = mf.mo_coeff.copy()
    report('② max_cycle=1000', mf, t0)
except Exception as ex:
    print(f'  ✗ {type(ex).__name__}: {str(ex)[:80]}', flush=True)
    C2 = None

# ==================================================================
# 3) Damping and level shifting
# ==================================================================
print('\n  3) Damping and level shifting', flush=True)
for tag, dmp, shift in (('③a damping=0.3', 0.3, 0.0),
                        ('③b damping=0.3 shift=0.5', 0.3, 0.5),
                        ('③c damping=0.5 shift=1.0', 0.5, 1.0)):
    m = make_mol()
    t0 = time.time()
    mf = scf.ROHF(m).x2c1e()
    mf.init_guess = 'atom'
    mf.max_cycle = 400
    mf.conv_tol = 1e-9
    mf.diis = False                 # DIIS off; it often aggravates oscillation
    mf.damp = dmp
    mf.level_shift = shift
    mf.verbose = 0
    try:
        mf.kernel()
        report(tag, mf, t0)
    except Exception as ex:
        print(f'  ✗ {tag}: {type(ex).__name__}: {str(ex)[:60]}', flush=True)

# ==================================================================
# 4) Newton-Raphson SCF, quadratically convergent and most effective here
# ==================================================================
print('\n  ④ Newton-Raphson SCF', flush=True)
m = make_mol()
t0 = time.time()
mf = scf.ROHF(m).x2c1e()
mf.init_guess = 'atom'
mf.max_cycle = 200
mf.conv_tol = 1e-9
mf.verbose = 0
try:
    mfn = mf.newton()
    mfn.max_cycle = 150
    mfn.conv_tol = 1e-9
    mfn.verbose = 0
    mfn.kernel()
    C4 = mfn.mo_coeff.copy()
    report('④ Newton-Raphson', mfn, t0)
except Exception as ex:
    print(f'  ✗ ④ {type(ex).__name__}: {str(ex)[:100]}', flush=True)
    C4 = None

# ==================================================================
# 5) Conventional SCF started from the Newton orbitals
# ==================================================================
if C4 is not None:
    print('\n  5) Conventional SCF from the Newton orbitals', flush=True)
    m = make_mol()
    t0 = time.time()
    mf = scf.ROHF(m).x2c1e()
    mf.max_cycle = 100
    mf.conv_tol = 1e-9
    mf.verbose = 0
    try:
        mf.kernel(C4)
        report('⑤ ROHF from Newton orbitals', mf, t0)
    except Exception as ex:
        print(f'  ✗ ⑤ {type(ex).__name__}: {str(ex)[:80]}', flush=True)

# ==================================================================
# 6) Selection by the variational principle
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('6) Selection by the variational principle', flush=True)
print('=' * 100, flush=True)
good = {k: v for k, v in RESULTS.items() if v['conv']}
if not good:
    print('  FAIL: no strategy converged; manual intervention required',
          flush=True)
else:
    best = min(good, key=lambda k: good[k]['E'])
    Emin = good[best]['E']
    Emax = max(v['E'] for v in good.values())
    print(f'  converged strategies = {len(good)}', flush=True)
    print(f'  lowest by the variational principle = {best}   E = {Emin:.10f}', flush=True)
    print(f'  spread over converged solutions = {(Emax-Emin)*H2K:.4f} kcal/mol', flush=True)
    print('   (a large spread means the europium atomic end also has multiple\n'
          '   solutions, which must be resolved by the variational principle)',
          flush=True)
    for k, v in sorted(good.items(), key=lambda kv: kv[1]['E']):
        print(f'    {k:<34} {v["E"]:.10f}  '
              f'Δ={(v["E"]-Emin)*H2K:+.4f} kcal', flush=True)

RESULTS['_best'] = (min(good, key=lambda k: good[k]['E']) if good else None)
RESULTS['_spread_kcal'] = (float((max(v['E'] for v in good.values())
                                  - min(v['E'] for v in good.values()))
                                 * H2K) if good else None)

outp = 'results/t156_eu_scf.json'
json.dump(RESULTS, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print('\nt156_DONE', flush=True)
gc.collect()
