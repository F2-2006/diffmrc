"""t158: Europium dissociation energies with convergence-accelerating settings.

Convergence settings applied to the initial SCF must be cleared before entering
the orbital optimization; otherwise the optimization retraverses the SCF with
those settings and can reach a different solution. The variational ordering
between the orbital optimization and its starting point is checked after every
call, since a violation indicates that the two calculations described different
electronic states.

Usage: python europium_dissociation_rerun.py <EuF|EuO> [NT]
"""
import os
import sys
import json
import time
import gc

NAME = sys.argv[1] if len(sys.argv) > 1 else 'EuF'
NT = sys.argv[2] if len(sys.argv) > 2 else '8'

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np                                               # noqa: E402
from pyscf import gto, scf, mcscf, mrpt                          # noqa: E402

H2K = 627.5094740631

MOLS = {
    'EuF': ("Eu 0 0 0; F 0 0 2.06", 8, 12, 10,
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], 130.98, 'F'),
    'EuO': ("Eu 0 0 0; O 0 0 1.81", 7, 13, 10,
            [29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 114.1, 'O'),
}

# The initial guess is decisive and must be listed explicitly.
#
# An earlier version of this script copied only damp, level_shift, and diis,
# omitting mf.init_guess = "atom". The same strategy then returned
# E = -10686.48 without converging, against -10839.92 with it: a difference
# of 153 Ha. Parameters must be checked line by line against the original.
#
# This version therefore takes init_guess as the first dimension of the scan.
GUESSES = ('atom', None)

EU_STRATS = ([('atom+damp0.5_shift1.0', 'atom',
               dict(damp=0.5, level_shift=1.0, diis=False)),
              ('atom+damp0.3_shift1.0', 'atom',
               dict(damp=0.3, level_shift=1.0, diis=False)),
              ('atom+default_scf', 'atom', dict()),
              ('default+damp0.5_shift1.0', None,
               dict(damp=0.5, level_shift=1.0, diis=False)),
              ('newton(atom)', 'atom', 'newton'),
              ('newton(default)', None, 'newton')])

OTHER_STRATS = [('atom+damp0.5_shift1.0', 'atom',
                 dict(damp=0.5, level_shift=1.0, diis=False)),
                ('atom+default_scf', 'atom', dict()),
                ('default+default_scf', None, dict()),
                ('newton(default)', None, 'newton'),
                ('atom+newton', 'atom', 'newton')]


def run_scf(mol, spin, strat, tag, verbose=True):
    """Run ROHF according to one strategy, returning (mf, converged).

    Three elements must match the original exactly:
        init_guess='atom'  +  max_cycle=400  +  conv_tol=1e-9
      init_guess="atom", max_cycle=400, conv_tol=1e-9
    Omitting the first gave E = -10686.48 rather than -10839.92.
    The initial guess is decisive, so it is scanned explicitly here.
    """
    name, guess, cfg = strat
    t0 = time.time()
    mf = (scf.ROHF(mol) if spin > 0 else scf.RHF(mol)).x2c1e()
    mf.max_cycle = 400
    mf.conv_tol = 1e-9
    mf.verbose = 0
    if guess is not None:
        mf.init_guess = guess          # decisive; see the note above
    try:
        if cfg == 'newton':
            mfn = mf.newton()
            mfn.max_cycle = 200
            mfn.conv_tol = 1e-9
            mfn.verbose = 0
            mfn.kernel()
            mf = mfn
        else:
            for k, v in cfg.items():
                setattr(mf, k, v)
            mf.kernel()
    except Exception as ex:
        if verbose:
            print(f'       [{tag}/{name:18s}] exception {type(ex).__name__}: '
                  f'{str(ex)[:50]}', flush=True)
        return None, False
    if verbose:
        print(f'       [{tag}/{name:18s}] E={mf.e_tot:.10f} '
              f'conv={mf.converged} ncyc={mf.cycles} {time.time()-t0:.0f}s',
              flush=True)
    return mf, bool(mf.converged)


def cas_nevpt2(mf, ncas, nele, act_orbs=None, mol=None):
    """CASSCF + NEVPT2

    Fixed a variational-ordering violation.
    An earlier version passed an mf object carrying damp=0.5, level_shift=1.0,
    and diis=False directly to mcscf.CASSCF. The reference implementation then
    re-ran the SCF with those settings and reached a different solution:
        ROHF best   E = -10839.9186926779
        CASSCF      E = -10839.6704809500    higher by 0.248 Ha
    Since the active space contains the starting determinant, the
    orbital-optimized energy cannot lie above the energy of the orbitals it
    started from, so the result was invalid. The lanthanide script that does
    not set damp or level_shift reaches the correct value, 19.15 kcal/mol
    """
    # Clear the SCF-only settings, which would otherwise steer the SCF that
    # the CASSCF step runs internally
    for attr, default in (('damp', 0.0), ('level_shift', 0.0),
                          ('diis', None), ('diis_space', None)):
        try:
            setattr(mf, attr, default)
        except Exception:
            pass

    # Variational check: CASSCF must not lie above its ROHF starting point
    E_ref = float(mf.e_tot)

    if act_orbs is not None and mol is not None:
        ncore = (mol.nelectron - nele) // 2
        nso = mf.mo_coeff.shape[1]
        core = list(range(ncore))
        vir = [i for i in range(nso) if i not in core and i not in act_orbs]
        mo = mf.mo_coeff[:, core + list(act_orbs) + vir]
        mc = mcscf.CASSCF(mf, ncas, nele, ncore=ncore)
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-6
        mc.fcisolver.conv_tol = 1e-10
        mc.max_cycle_macro = 800
        mc.verbose = 0
        E = mc.kernel(mo)[0]
    else:
        mc = mcscf.CASSCF(mf, ncas, nele)
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-6
        mc.fcisolver.conv_tol = 1e-10
        mc.max_cycle_macro = 800
        mc.verbose = 0
        E = mc.kernel()[0]
    e2 = mrpt.NEVPT(mc).kernel()
    # Variational check
    if E > E_ref + 1e-8:
        print(f'       VARIATIONAL VIOLATION: CASSCF={E:.10f} > ROHF={E_ref:.10f}  '
              f'({(E-E_ref)*627.5094740631:+.2f} kcal/mol) -- result not usable',
              flush=True)
    return float(E), float(e2), bool(mc.converged)


def main():
    if NAME not in MOLS:
        print(f'Unknown system {NAME}; choose from {list(MOLS)}')
        sys.exit(1)
    atomstr, spin, nele, ncas, act, de_ref, other = MOLS[NAME]
    print('=' * 106, flush=True)
    print(f'Europium rerun with convergence-accelerating settings: '
          f'{NAME}  T={NT}', flush=True)
    print(f'  {atomstr}  CAS({nele},{ncas})  reference De={de_ref}', flush=True)
    print('=' * 106, flush=True)

    t_all = time.time()
    res = dict(system=NAME, basis='ano', x2c1e=True, nele=nele, ncas=ncas,
               geometry=atomstr, de_ref=de_ref, nthread=NT,
               method='convergence-accelerated ROHF -> CASSCF + NEVPT2')

    # ---------- ① Eu atomic end ----------
    print('\n  1) Europium atomic end', flush=True)
    eu_mol = gto.M(atom='Eu 0 0 0', basis='ano', spin=7, verbose=0)
    eu_sols = []
    for st in EU_STRATS:
        mf, ok = run_scf(eu_mol, 7, st, 'Eu')
        if ok:
            eu_sols.append((st[0], mf))
    if not eu_sols:
        print('   ✗ Eu no converged solution; stopping', flush=True)
        return
    best_name, best_mf = min(eu_sols, key=lambda kv: kv[1].e_tot)
    spread = (max(mf.e_tot for _, mf in eu_sols)
              - min(mf.e_tot for _, mf in eu_sols)) * H2K
    print(f'     lowest by the variational principle = {best_name}  converged solutions = {len(eu_sols)}  '
          f'  spread = {spread:.2f} kcal/mol', flush=True)
    E_eu_cas, e2_eu, ok_eu = cas_nevpt2(best_mf, 7, 7)
    E_eu = E_eu_cas + e2_eu
    print(f'     [Eu] CASSCF={E_eu_cas:.8f} conv={ok_eu}  E2={e2_eu:.8f}  '
          f'E_tot={E_eu:.8f}', flush=True)
    res['Eu'] = dict(strategy=best_name, E_cas=E_eu_cas, e2=e2_eu,
                     E_tot=E_eu, conv=ok_eu, n_sols=len(eu_sols),
                     spread_kcal=float(spread))

    # ---------- 2) The other atom ----------
    print(f'\n  ② {other} atomic end', flush=True)
    o_spin = {'F': 1, 'O': 2}[other]
    o_cas = {'F': (3, 5), 'O': (3, 4)}[other]
    o_mol = gto.M(atom=f'{other} 0 0 0', basis='ano', spin=o_spin, verbose=0)
    o_sols = []
    for st in OTHER_STRATS:
        mf, ok = run_scf(o_mol, o_spin, st, other)
        if ok:
            o_sols.append((st[0], mf))
    if not o_sols:
        print(f'   ✗ {other} no converged solution; stopping', flush=True)
        return
    bo_name, bo_mf = min(o_sols, key=lambda kv: kv[1].e_tot)
    E_o_cas, e2_o, ok_o = cas_nevpt2(bo_mf, o_cas[0], o_cas[1])
    E_o = E_o_cas + e2_o
    print(f'     best for {other} = {bo_name}  E_tot={E_o:.8f}', flush=True)
    res['other'] = dict(element=other, strategy=bo_name, E_cas=E_o_cas,
                        e2=e2_o, E_tot=E_o, conv=ok_o)

    E_atom_sum = E_eu + E_o
    res['E_atom_sum'] = float(E_atom_sum)
    print(f'     sum over atoms = {E_atom_sum:.8f}', flush=True)

    # ---------- ③ molecular end ----------
    print(f'\n  3) {NAME} molecular end, multi-strategy', flush=True)
    m_mol = gto.M(atom=atomstr, basis='ano', spin=spin, verbose=0)
    m_sols = []
    for st in OTHER_STRATS:
        mf, ok = run_scf(m_mol, spin, st, 'mol')
        if ok:
            m_sols.append((st[0], mf))
    if not m_sols:
        print('   ✗ molecular endno converged solution', flush=True)
        return
    bm_name, bm_mf = min(m_sols, key=lambda kv: kv[1].e_tot)
    mspread = (max(mf.e_tot for _, mf in m_sols)
               - min(mf.e_tot for _, mf in m_sols)) * H2K
    print(f'     lowest by the variational principle = {bm_name}  converged solutions = {len(m_sols)}  '
          f'  spread = {mspread:.2f} kcal/mol', flush=True)
    E_m_cas, e2_m, ok_m = cas_nevpt2(bm_mf, ncas, nele, act, m_mol)
    E_m = E_m_cas + e2_m
    print(f'     [mol] CASSCF={E_m_cas:.8f} conv={ok_m}  E2={e2_m:.8f}  '
          f'E_tot={E_m:.8f}', flush=True)
    res['mol'] = dict(strategy=bm_name, E_cas=E_m_cas, e2=e2_m, E_tot=E_m,
                      conv=ok_m, n_sols=len(m_sols),
                      spread_kcal=float(mspread))

    # ---------- ④ De ----------
    De = (E_atom_sum - E_m) * H2K
    print('\n' + '=' * 106, flush=True)
    print(f'  De = {De:.4f}   reference {de_ref}   '
          f'Δ = {De - de_ref:+.4f} kcal/mol', flush=True)
    print('=' * 106, flush=True)
    res['De'] = float(De)
    res['dDe'] = float(De - de_ref)
    res['t'] = round(time.time() - t_all, 1)

    outp = f'results/t158_A_{NAME}.json'
    json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
              default=str, indent=1)
    print(f'\n>>> {outp}', flush=True)
    print(f'\nt158_{NAME}_DONE', flush=True)
    gc.collect()


if __name__ == '__main__':
    main()
