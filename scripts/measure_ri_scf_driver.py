"""Measure the RI-J SCF driver: energies, densities, iterations, time, memory.

The reference SCF and the RIJONX SCF are run on the same molecule.  Each
mode is warmed once before the timed run so the reported wall time is not
dominated by the first-call compilation of the pyscfad kernels.  Peak
memory is measured in child processes (``ru_maxrss``) so both modes start
from the same interpreter baseline.

Usage::

    python measure_ri_scf_driver.py [--json results.json] [--gradient]

The ``--gradient`` switch adds the H2O gradient checks: the implicit
gradient, the explicit overlap-completed gradient and central differences
of the converged energy.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

CASES = {
    'h2o': dict(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', spin=0),
    'co': dict(atom='C 0 0 -0.5643; O 0 0 0.5643',
               basis='6-31g', spin=0),
    'oh': dict(atom='O 0 0 0.0; H 0 0 0.9697',
               basis='6-31g', spin=1),
    'hi': dict(atom='H 0 0 0; I 0 0 1.61',
               basis='def2-svp', spin=0, ecp='def2-svp'),
}


def build(case):
    from pyscfad import gto
    spec = CASES[case]
    mol = gto.Mole()
    mol.atom = spec['atom']
    mol.basis = spec['basis']
    mol.spin = spec['spin']
    if spec.get('ecp'):
        mol.ecp = spec['ecp']
    mol.verbose = 0
    mol.build()
    return mol


def _build_at(mol, coords, case):
    """A molecule rebuilt at displaced coordinates (Bohr)."""
    from pyscfad import gto
    spec = CASES[case]
    m = gto.Mole()
    m.atom = [(mol.atom_pure_symbol(i), coords[i]) for i in range(mol.natm)]
    m.basis = spec['basis']
    m.spin = spec['spin']
    if spec.get('ecp'):
        m.ecp = spec['ecp']
    m.unit = 'Bohr'
    m.verbose = 0
    m.build()
    return m


def reference(case):
    from pyscfad.scf import hf, rohf
    mol = build(case)
    mf = rohf.ROHF(mol) if mol.spin != 0 else hf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mol, mf


def rijonx(case):
    from diffmrc.driver.ri_scf_driver import rijonx_scf
    mol = build(case)
    return mol, rijonx_scf(mol)


def timed(ctor, case):
    """Warm once (kernel compilation), then time a reset SCF."""
    mol = build(case)
    mf = ctor(mol)
    mf.verbose = 0
    mf.kernel()
    mf.reset()
    cycles = []

    def callback(envs):
        cycles.append(int(envs['cycle']) + 1)

    mf.callback = callback
    t0 = time.perf_counter()
    mf.kernel()
    wall = time.perf_counter() - t0
    mf.callback = None
    return mol, mf, wall, (cycles[-1] if cycles else None)


def ref_ctor(case):
    from pyscfad.scf import hf, rohf
    spin = CASES[case]['spin']
    return rohf.ROHF if spin != 0 else hf.RHF


def rij_ctor(case):
    from diffmrc.driver.ri_scf_driver import RIJONX_RHF, RIJONX_ROHF
    spin = CASES[case]['spin']
    return RIJONX_ROHF if spin != 0 else RIJONX_RHF


def measure(case, gradient=False):
    from diffmrc.driver import ri_scf
    from diffmrc.driver.ri_scf_driver import rijonx_scf, rijonx_scf_gradient

    mol = build(case)
    spec = CASES[case]
    _, ref, t_ref, n_ref = timed(ref_ctor(case), case)
    _, mf, t_rij, n_rij = timed(rij_ctor(case), case)

    dm_ref = np.asarray(ref.make_rdm1())
    dm = np.asarray(mf.make_rdm1())
    dip_ref = np.asarray(ref.dip_moment(verbose=0))
    dip = np.asarray(mf.dip_moment(verbose=0))
    out = {
        'case': case,
        'spin': spec['spin'],
        'ecp': bool(spec.get('ecp')),
        'nao': int(mol.nao),
        'nelectron': int(mol.nelectron),
        'e_rhf': float(ref.e_tot),
        'e_rijonx': float(mf.e_tot),
        'dE': float(mf.e_tot - ref.e_tot),
        'max_ddm': float(np.abs(dm - dm_ref).max()),
        'dip_rhf': dip_ref.tolist(),
        'dip_rijonx': dip.tolist(),
        'max_ddip': float(np.abs(dip - dip_ref).max()),
        'cycles_rhf': n_ref,
        'cycles_rijonx': n_rij,
        'wall_rhf': t_ref,
        'wall_rijonx': t_rij,
        'converged': bool(mf.converged and ref.converged),
    }
    if dm.ndim == 3:
        out['max_ddm_alpha'] = float(np.abs(dm[0] - dm_ref[0]).max())
        out['max_ddm_beta'] = float(np.abs(dm[1] - dm_ref[1]).max())
        out['nelec_alpha_rhf'] = float(np.trace(dm_ref[0]))
        out['nelec_beta_rhf'] = float(np.trace(dm_ref[1]))
        out['nelec_alpha_rijonx'] = float(np.trace(dm[0]))
        out['nelec_beta_rijonx'] = float(np.trace(dm[1]))

    state = mf.ri_state(mol)
    naux = int(state.naux)
    out['naux'] = naux
    if naux is not None:
        out['ri_working_set_bytes'] = int(8 * mol.nao ** 2 * naux)
    out['dense_eri_bytes'] = int(8 * mol.nao ** 4)
    out['dense_eri_deriv_bytes'] = int(24 * mol.nao ** 4)

    if spec['spin'] == 0:
        pmol = mol.to_pyscf()
        nocc = mol.nelectron // 2
        c = np.asarray(mf.mo_coeff)[:, :nocc]
        out['e_rijonx_from_density'] = float(ri_scf.rijonx_energy(pmol, c))
        out['energy_identity_gap'] = (out['e_rijonx_from_density']
                                      - out['e_rijonx'])

    if gradient and case == 'h2o':
        g_imp = np.asarray(mf.energy_grad().coords)
        g_exp = np.asarray(rijonx_scf_gradient(mf))
        step = 1.0e-3
        coords = np.asarray(mol.atom_coords())
        fd = np.zeros_like(g_imp)
        for atom in range(mol.natm):
            for axis in range(3):
                values = []
                for sign in (1.0, -1.0):
                    shifted = coords.copy()
                    shifted[atom, axis] += sign * step
                    values.append(rijonx_scf(_build_at(
                        mol, shifted, case)).e_tot)
                fd[atom, axis] = (values[0] - values[1]) / (2.0 * step)
        out['gradient_implicit'] = g_imp.tolist()
        out['gradient_explicit'] = g_exp.tolist()
        out['gradient_fd'] = fd.tolist()
        out['grad_max_implicit_vs_fd'] = float(np.abs(g_imp - fd).max())
        out['grad_max_explicit_vs_fd'] = float(np.abs(g_exp - fd).max())
        out['grad_max_explicit_vs_implicit'] = float(np.abs(g_exp - g_imp).max())

        # unit-level check: the custom JVP against the analytic RI-J gradient
        import jax
        import jax.numpy as jnp
        dm0 = jnp.asarray(np.asarray(ref.make_rdm1()))
        state = mf.ri_state(mol)
        fock = state.fock

        def e_j(m):
            return 0.5 * jnp.sum(dm0 * fock(m, dm0))

        g_jvp = np.asarray(jax.grad(e_j)(mol).coords)
        g_rij = ri_scf.ri_j_gradient(pmol, np.asarray(ref.mo_coeff)[:, :nocc])
        out['ri_j_max_jvp_vs_analytic'] = float(np.abs(g_jvp - g_rij).max())
    return out


def child_rss(case, mode):
    if mode == 'rhf':
        ctor = ref_ctor(case)
    else:
        ctor = rij_ctor(case)
    mol = build(case)
    mf = ctor(mol)
    mf.verbose = 0
    mf.kernel()
    print('RSS_KB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def peak_rss(case, mode):
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--child', case, mode],
        capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith('RSS_KB'):
            return int(line.split()[1])
    raise RuntimeError(f'child failed: {proc.stderr[-800:]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json')
    parser.add_argument('--gradient', action='store_true')
    parser.add_argument('--child', nargs=2)
    args = parser.parse_args()
    if args.child:
        child_rss(*args.child)
        return
    results = []
    for case in CASES:
        row = measure(case, gradient=args.gradient)
        row['rss_rhf_kb'] = peak_rss(case, 'rhf')
        row['rss_rijonx_kb'] = peak_rss(case, 'rij')
        results.append(row)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')


if __name__ == '__main__':
    main()
