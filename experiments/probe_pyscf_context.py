#!/usr/bin/env python
"""Which context decides the PySCF swapped-ordering result?

Three experiments disagreed on what CASSCF(2,2) converges to for H2O/6-31g
when the starting orbitals have columns 3 and 6 exchanged:

    probe_h2o_lower_basin.py     -76.006040723014
    probe_basin_stall.py         -75.229918957983
    probe_pyscf_two_solutions.py -75.229918958034 (fresh mf per run)

The basin script ran a CASSCF from the RHF ordering first and reused the
same mf object; the others did not. This script reruns the basin block
verbatim, then two controls that vary one thing at a time:

    A  loop sharing mf2          (basin, verbatim)
    B  swap only, same process, fresh mf
    C  loop sharing mf3, but fresh mol/mf per iteration

Whichever context reproduces -76.006 identifies the hidden state.

Usage:
    python probe_pyscf_context.py [basis]
"""
import os
import sys

_BASIS = sys.argv[1] if len(sys.argv) > 1 else '6-31g'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '4'

import numpy as np                                              # noqa: E402

# Imported first, exactly as the basin script does, so that any patching
# pyscfad performs on the pyscf namespace is present here too.
from pyscfad import gto as pgto, scf as pscf                    # noqa: E402

ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'


def run_block(label, order, fresh_each=False):
    """order: iterable of (tag, swap-or-None)."""
    from pyscf import gto as gto2, scf as scf2, mcscf
    mol2 = gto2.M(atom=ATOM, basis=_BASIS, verbose=0)
    mf2 = scf2.RHF(mol2)
    mf2.kernel()
    S = mol2.intor('int1e_ovlp')
    out = []
    for tag, swap in order:
        if fresh_each:
            mol2 = gto2.M(atom=ATOM, basis=_BASIS, verbose=0)
            mf2 = scf2.RHF(mol2)
            mf2.kernel()
        mo = np.array(mf2.mo_coeff)
        if swap:
            mo[:, [swap[0], swap[1]]] = mo[:, [swap[1], swap[0]]]
        mc = mcscf.CASSCF(mf2, 2, 2)
        mc.verbose = 0
        E = float(mc.kernel(mo)[0])
        C = np.asarray(mc.mo_coeff)
        W = C.T @ S @ np.asarray(mf2.mo_coeff)
        labs = '  '.join(f'{j}<-{int(np.argmax(np.abs(W[:, j])))}'
                         for j in (4, 5))
        print(f'  [{label}] {tag:>14}: E = {E:.12f}  conv={bool(mc.converged)}'
              f'  active: {labs}', flush=True)
        out.append(E)
    return out


def main():
    print('=' * 78)
    print(f'PySCF context test  H2O/{_BASIS}')
    print('=' * 78)

    print()
    print('  A: loop sharing one mf2 (the basin script, verbatim)')
    run_block('A', (('rhf order', None), ('swap 3<->6', (3, 6))))

    print()
    print('  B: swap only, same process, fresh mf')
    run_block('B', (('swap 3<->6', (3, 6)),))

    print()
    print('  C: loop, but a fresh mol/mf per iteration')
    run_block('C', (('rhf order', None), ('swap 3<->6', (3, 6))),
              fresh_each=True)

    print()
    print('=' * 78)
    print('  A gives -76.006  => the shared mf object carries the state')
    print('  B gives -76.006  => the process history (not mf sharing) does')
    print('  none do          => the basin number is not reproducible and')
    print('                      the independent-confirmation claim needs')
    print('                      to be withdrawn')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
