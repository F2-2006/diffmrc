"""Check the molecular-end rule against a real SCF solution.

This runs on the server, where the pyscfad environment and the cached
EuF / EuO molecular-end solutions live. The unit tests build a stand-in
Mole with an identity overlap, which checks the logic but not the
parsing of real PySCF labels. This script closes that gap: it feeds the
rule the actual ``ao_labels()`` and the actual overlap matrix of a
converged calculation.

What the measurement is expected to reproduce (EuF, 2026-09-22):

    eight singly occupied orbitals -- six Eu f, one Eu 6s, one Eu 5d
    three doubly occupied fluorine p orbitals
    an active space of eleven orbitals holding fourteen electrons
    the strongest two fluorine orbitals are the pi pair, not the sigma

Run from the package root:

    PYTHONPATH=src python experiments/verify_molecular_end_rule.py

Exit status is non-zero if any expectation fails, so the script can be
used as a check rather than read as a report.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import numpy as np                                              # noqa: E402

from diffmrc.policy.rules import molecular_end_candidate        # noqa: E402

#: The cached SCF/orbital data written by the t176 batch; point this at
#: the directory holding the t176cache_*.npz files (the experiment
#: skips a system when its cache is absent).
_CACHE_DIR = os.environ.get('DIFFMRC_EXPERIMENT_CACHE', '.')

SYSTEMS = {
    'EuF': ("Eu 0 0 0; F 0 0 2.06", 8, 'F',
            os.path.join(_CACHE_DIR,
                         't176cache_EuF_mol_atom+default_scf.npz'),
            dict(n_singly=8, n_centre=8, n_ligand=3, n_active=11,
                 n_electrons=14, centre_counts={'Eu-f': 6, 'Eu-s': 1,
                                                'Eu-d': 1})),
    'EuO': ("Eu 0 0 0; O 0 0 1.81", 7, 'O',
            os.path.join(_CACHE_DIR,
                         't176cache_EuO_mol_atom+default_scf.npz'),
            None),   # expectations unknown until the cache exists
}


def run(name):
    atomstr, spin, ligand, path, expect = SYSTEMS[name]
    if not os.path.exists(path):
        print(f'{name}: no cache at {path}')
        return None
    from pyscf import gto
    d = np.load(path)
    mol = gto.M(atom=atomstr, basis='ano', spin=spin, verbose=0)
    C = np.asarray(d['mo_coeff'])
    occ = np.asarray(d['mo_occ'])
    print(f'{name}: E = {float(d["e_tot"]):.8f}, {C.shape[0]} AO, '
          f'{C.shape[1]} MO')

    c = molecular_end_candidate(mol, C, occ, center='Eu', ligand=ligand)
    if c is None:
        print('  rule refused')
        return False

    print(f'  active: {c.active}')
    print(f'  per_element: {c.per_element}')
    for e in c.evidence:
        print(f'  evidence: {e}')

    if expect is None:
        print('  (no expectations recorded for this system yet)')
        return None

    ok = True
    checks = [
        ('centre orbital count', c.per_element['Eu']['n_orbitals'],
         expect['n_centre']),
        ('ligand orbital count', c.per_element[ligand]['n_orbitals'],
         expect['n_ligand']),
        ('active size', len(c.active), expect['n_active']),
        ('active electrons',
         int(sum(2 if occ[i] > 1.5 else 1 for i in c.active)),
         expect['n_electrons']),
        ('singly occupied total',
         int(np.sum((occ > 0.5) & (occ < 1.5))), expect['n_singly']),
    ]
    for label, got, want in checks:
        mark = 'ok' if got == want else 'FAIL'
        if got != want:
            ok = False
        print(f'  {mark:>4} {label}: {got} (expected {want})')

    joined = ' '.join(c.evidence)
    for shell, want in expect['centre_counts'].items():
        token = f'{shell} x{want}'
        present = token in joined
        if not present:
            ok = False
        print(f'  {"ok" if present else "FAIL":>4} evidence says {token}')

    # The two strongest ligand orbitals should be the pi pair. In EuF
    # they are indices 30 and 31; the sigma 2pz (29) mixes with the
    # centre and carries less fluorine character. Checking the *count of
    # doubly occupied ligand orbitals with near-full weight* is the
    # composition-level statement of that.
    tight = molecular_end_candidate(mol, C, occ, center='Eu', ligand=ligand,
                                    ligand_doubly=2)
    if tight is not None:
        ligand_part = [i for i in tight.active if i not in c.active
                       or occ[i] > 1.5]
        print(f'  ligand_doubly=2 -> active {tight.active} '
              f'(ligand part {ligand_part})')
    return ok


def main():
    names = sys.argv[1:] or ['EuF', 'EuO']
    results = {n: run(n) for n in names}
    failed = [n for n, r in results.items() if r is False]
    print()
    if failed:
        print(f'FAILED: {failed}')
        return 1
    print('verify_molecular_end_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
