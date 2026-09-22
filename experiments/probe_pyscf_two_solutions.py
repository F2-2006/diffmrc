#!/usr/bin/env python
"""Which PySCF starting orderings reach which H2O CASSCF(2,2) solution?

Two experiments disagreed. probe_h2o_lower_basin.py reported

    rhf order   : -75.984881801185
    swap 3 <-> 6: -76.006040723014

while probe_basin_stall.py, running the swap in a different context,
reported -75.229918957983 for the same nominal setup. One of the two
carries hidden state; a claim of independent confirmation depends on
which.

Candidate: PySCF's CASSCF writes its converged orbitals back into the
mean-field object, so a second run that re-reads mf.mo_coeff starts from
the first run's answer, not from the RHF orbitals. This script runs each
ordering in a fresh process state, printing the reference-orbital
identity of the final active pair, so the answer does not depend on
what ran before.

Usage:
    python probe_pyscf_two_solutions.py [basis] [nthread]
"""
import os
import sys

_BASIS = sys.argv[1] if len(sys.argv) > 1 else '6-31g'
_NT = sys.argv[2] if len(sys.argv) > 2 else '4'

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = _NT

import numpy as np                                              # noqa: E402
from pyscf import gto, scf, mcscf                                # noqa: E402

ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
NCORE_EXPECTED = 4
NACT, NELE = 2, 2


def active_labels(C, C0, S):
    """Which reference orbitals the two CASSCF-active columns descend from."""
    W = C.T @ S @ C0
    out = []
    for j in (NCORE_EXPECTED, NCORE_EXPECTED + 1):
        i = int(np.argmax(np.abs(W[:, j])))
        out.append(f'{j}<-{i}({W[i, j]:+.2f})')
    return '  '.join(out)


def fresh():
    """A new mol/mf every time, so no run can contaminate the next."""
    mol = gto.M(atom=ATOM, basis=_BASIS, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def run(tag, swap):
    mol, mf = fresh()
    C0 = np.array(mf.mo_coeff)
    mo = np.array(C0)
    if swap is not None:
        mo[:, [swap[0], swap[1]]] = mo[:, [swap[1], swap[0]]]
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.verbose = 0
    E = float(mc.kernel(mo)[0])
    S = mol.intor('int1e_ovlp')
    print(f'  {tag:>18}: E = {E:.12f}  conv={bool(mc.converged)}  '
          f'active: {active_labels(np.asarray(mc.mo_coeff), C0, S)}')
    return E, mf


def main():
    print('=' * 78)
    print(f'PySCF CASSCF(2,2) starting orderings  H2O/{_BASIS}')
    print('=' * 78)
    run('rhf order', None)
    run('swap 3<->6', (3, 6))
    run('swap 3<->6 (again)', (3, 6))
    run('swap 4<->6', (4, 6))

    # Does a CASSCF run write back into the mean-field object?  This is
    # the hidden state the two experiments could have differed on.
    mol, mf = fresh()
    before = np.array(mf.mo_coeff)
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.verbose = 0
    mc.kernel(before)
    after = np.array(mf.mo_coeff)
    print()
    print(f'  mf.mo_coeff changed by mc.kernel: '
          f'{float(np.abs(after - before).max()):.3e}')
    E2 = float(mcscf.CASSCF(mf, NACT, NELE).kernel(np.array(mf.mo_coeff))[0])
    print(f'  CASSCF from the possibly-updated orbitals: E = {E2:.12f}')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
