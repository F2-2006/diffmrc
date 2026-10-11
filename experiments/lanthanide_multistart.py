"""t142: Lanthanide dissociation energies with multi-start orbital optimization.

Each atomic and molecular end is computed from several SCF initial guesses;
only converged solutions are retained, and the lowest is selected by the
variational principle. Active spaces follow the conservation criterion: the
number of active electrons and orbitals on the molecular side equals the sum
of the corresponding quantities on the separated-atom side.

Usage: python lanthanide_multistart.py <LaF|LaO|EuF|EuO> [nthread]
"""
import os
import sys
import json
import time
import gc

NAME = sys.argv[1] if len(sys.argv) > 1 else 'LaF'
NT = sys.argv[2] if len(sys.argv) > 2 else '8'

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np                                               # noqa: E402
from pyscf import gto, scf, mcscf, mrpt                          # noqa: E402

H2K = 627.5094740631

# Molecular definitions follow the same geometry, spin, and window
MOLS = {
    'LaF': ("La 0 0 0; F 0 0 2.02", 0, 2, 7, None, 156.56, 'La', 'F'),
    'LaO': ("La 0 0 0; O 0 0 1.79", 1, 7, 10,
            [29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 189.56, 'La', 'O'),
    'EuF': ("Eu 0 0 0; F 0 0 2.06", 8, 12, 10,
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], 130.98, 'Eu', 'F'),
    'EuO': ("Eu 0 0 0; O 0 0 1.81", 7, 13, 10,
            [29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 114.1, 'Eu', 'O'),
}
ATOM_SPIN = {'La': 1, 'F': 1, 'O': 2, 'Eu': 7}
ATOM_CAS = {'La': (7, 3), 'F': (3, 5), 'O': (3, 4), 'Eu': (7, 7)}
GUESSES = ('default', 'atom', '1e', 'huckel')


def scf_multistart_atom(el, verbose=True):
    """Atomic end: multi-start ROHF, retaining only the converged solutions
    and selecting the lowest."""
    spin = ATOM_SPIN[el]
    m = gto.M(atom=f'{el} 0 0 0', basis='ano', spin=spin, verbose=0)
    cands = {}
    for g in GUESSES:
        t0 = time.time()
        mf = (scf.ROHF(m) if spin > 0 else scf.RHF(m)).x2c1e()
        if g != 'default':
            mf.init_guess = g
        mf.max_cycle = 250
        mf.conv_tol = 1e-9
        mf.verbose = 0
        try:
            mf.kernel()
        except Exception as ex:
            if verbose:
                print(f'       [{el}/{g}] exception {type(ex).__name__}', flush=True)
            continue
        cands[g] = mf
        if verbose:
            print(f'       [{el}/{g:8s}] E={mf.e_tot:.10f} conv={mf.converged} '
                  f'ncyc={mf.cycles} {time.time()-t0:.0f}s', flush=True)
    good = {g: mf for g, mf in cands.items() if mf.converged}
    if not good:
        if verbose:
            print(f'       FAIL {el}: no converged initial guess', flush=True)
        return None
    best_g = min(good, key=lambda g: good[g].e_tot)
    if verbose:
        spread = (max(mf.e_tot for mf in good.values())
                  - min(mf.e_tot for mf in good.values())) * H2K
        print(f'       {el} best initial guess = {best_g}  '
              f'spread over converged solutions = {spread:.4f} kcal/mol', flush=True)
    return good[best_g]


def atom_energy(el):
    mf = scf_multistart_atom(el)
    if mf is None:
        return None
    ncas, nele = ATOM_CAS[el]
    mc = mcscf.CASSCF(mf, ncas, nele)
    mc.conv_tol = 1e-8
    mc.conv_tol_grad = 1e-6
    mc.fcisolver.conv_tol = 1e-10
    mc.max_cycle_macro = 400
    mc.verbose = 0
    E_cas = mc.kernel()[0]
    e2 = mrpt.NEVPT(mc).kernel()
    print(f'       [{el}] CASSCF E_cas={E_cas:.8f} conv={mc.converged}  '
          f'E2={e2:.8f}  E_tot={E_cas+e2:.8f}', flush=True)
    return dict(E_cas=float(E_cas), e2=float(e2),
                E_tot=float(E_cas) + float(e2),
                conv=bool(mc.converged), guess=mf.init_guess)


def mol_energy(atomstr, spin, nele, ncas, act_orbs):
    """Molecular end: multi-start ROHF, then CASSCF and NEVPT2 from the best."""
    m = gto.M(atom=atomstr, basis='ano', spin=spin, verbose=0)
    cands = {}
    for g in GUESSES:
        t0 = time.time()
        mf = (scf.ROHF(m) if spin > 0 else scf.RHF(m)).x2c1e()
        if g != 'default':
            mf.init_guess = g
        mf.max_cycle = 250
        mf.conv_tol = 1e-9
        mf.verbose = 0
        try:
            mf.kernel()
        except Exception as ex:
            print(f'     [mol/{g}] exception {type(ex).__name__}', flush=True)
            continue
        cands[g] = mf
        print(f'     [mol/{g:8s}] E={mf.e_tot:.10f} conv={mf.converged} '
              f'ncyc={mf.cycles} {time.time()-t0:.0f}s', flush=True)
    good = {g: mf for g, mf in cands.items() if mf.converged}
    if not good:
        print('     FAIL: no converged initial guess for the molecule', flush=True)
        return None
    best_g = min(good, key=lambda g: good[g].e_tot)
    spread = (max(mf.e_tot for mf in good.values())
              - min(mf.e_tot for mf in good.values())) * H2K
    print(f'     best initial guess = {best_g}  spread over converged solutions = {spread:.4f} kcal/mol',
          flush=True)
    mf = good[best_g]

    if act_orbs is not None:
        ncore = (m.nelectron - nele) // 2
        nso = mf.mo_coeff.shape[1]
        core = list(range(ncore))
        vir = [i for i in range(nso) if i not in core and i not in act_orbs]
        mo = mf.mo_coeff[:, core + list(act_orbs) + vir]
        mc = mcscf.CASSCF(mf, ncas, nele, ncore=ncore)
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-6
        mc.fcisolver.conv_tol = 1e-10
        mc.max_cycle_macro = 400
        mc.verbose = 0
        E_cas = mc.kernel(mo)[0]
    else:
        mc = mcscf.CASSCF(mf, ncas, nele)
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-6
        mc.fcisolver.conv_tol = 1e-10
        mc.max_cycle_macro = 400
        mc.verbose = 0
        E_cas = mc.kernel()[0]
    e2 = mrpt.NEVPT(mc).kernel()
    print(f'     [mol] CASSCF E_cas={E_cas:.8f} conv={mc.converged}  '
          f'E2={e2:.8f}  E_tot={E_cas+e2:.8f}', flush=True)
    return dict(E_cas=float(E_cas), e2=float(e2),
                E_tot=float(E_cas) + float(e2),
                conv=bool(mc.converged), guess=best_g)


def main():
    if NAME not in MOLS:
        print(f'Unknown system; choose from {list(MOLS)}')
        sys.exit(1)
    atomstr, spin, nele, ncas, act, de_ref, a1, a2 = MOLS[NAME]
    print('=' * 100, flush=True)
    print(f'Lanthanide layer, multi-start with convergence filtering: '
          f'{NAME}  T={NT}', flush=True)
    print(f'  {atomstr}   CAS({nele},{ncas})   window={act}   '
          f'reference De={de_ref}', flush=True)
    print('=' * 100, flush=True)

    t0 = time.time()
    res = dict(system=NAME, basis='ano', x2c1e=True, nele=nele, ncas=ncas,
               geometry=atomstr, act_window=str(act), de_ref=de_ref,
               method='multistart ROHF -> CASSCF + NEVPT2 (PySCF)',
               nthread=NT, mkl_cbwr='COMPATIBLE', guesses=list(GUESSES))

    print('\n  1) Atomic end, multi-start', flush=True)
    at = {}
    for el in (a1, a2):
        r = atom_energy(el)
        if r is None:
            print(f'   ✗ {el} no converged solution; stopping', flush=True)
            return
        at[el] = r
    E_sum = sum(v['E_tot'] for v in at.values())
    print(f'     sum over atoms = {E_sum:.8f}', flush=True)
    res['atoms'] = at
    res['E_atom_sum'] = float(E_sum)

    print('\n  2) Molecular end, multi-start', flush=True)
    r = mol_energy(atomstr, spin, nele, ncas, act)
    if r is None:
        print('   ✗ molecular endfailed', flush=True)
        return
    De = (E_sum - r['E_tot']) * H2K
    print(f'\n  De = {De:.4f}   reference {de_ref}   Δ = {De - de_ref:+.4f} kcal/mol',
          flush=True)
    res['mol'] = r
    res['De'] = float(De)
    res['dDe'] = float(De - de_ref)
    res['t'] = round(time.time() - t0, 1)

    outp = f'results/t142_A_{NAME}.json'
    json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
              default=str, indent=1)
    print(f'\n>>> {outp}', flush=True)
    print('\nt142_DONE', flush=True)
    gc.collect()


if __name__ == '__main__':
    main()
