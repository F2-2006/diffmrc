#!/usr/bin/env python
"""RI/DF approximation to the four-center ERIs: assembly and derivatives.

System: H2O / 6-31G, CAS(2,2) with ncore = 4 (frozen oxygen 1s).

The script walks the public interface of ``diffmrc.core.ri``:

  1. auxiliary molecule and RI factors (B, S, S^{-1/2});
  2. RI assembly against the pyscf DF reconstruction and the dense ERI;
  3. RI-CASCI against dense CASCI;
  4. nuclear derivative of the three-center tensor against central
     differences;
  5. materialized ERI derivative, its contraction, and the JAX rule
     against the same finite differences.

Every number printed here is a measured quantity of this run.  The
expected orders of magnitude are given in the tutorial chapter
``manual/tutorial/chapters/10-ri-df.tex``.

Run from the repository root, with the package importable:

    python examples/ri_df.py
    PYTHONPATH=src python examples/ri_df.py
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import time

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)

from pyscf import gto, scf  # noqa: E402

from diffmrc.core import ri  # noqa: E402
from diffmrc.core.casci import cs_casci  # noqa: E402


ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
BASIS = '6-31g'
NCORE, NACT, NELE = 4, 2, 2
FD_STEP = 1e-4  # Bohr


def mo_integrals(c, h1e, eri_ao):
    """MO integrals by four sequential two-operand contractions.

    The one-shot four-operand einsum is avoided on purpose: at large
    sizes XLA lowers it with only ~1e-6 absolute accuracy, which the
    small gaps of a generalized Fock matrix amplify (see the manual's
    differentiable-chain chapter).
    """
    h1 = c.T @ h1e @ c
    t = np.einsum('pi,pqrs->iqrs', c, eri_ao)
    t = np.einsum('qj,iqrs->ijrs', c, t)
    t = np.einsum('rk,ijrs->ijks', c, t)
    eri_mo = np.einsum('sl,ijks->ijkl', c, t)
    return h1, eri_mo


def rel_dev(a, b):
    """max|a - b| / max|b|, the deviation used throughout the records."""
    scale = np.abs(b).max()
    return np.abs(a - b).max() / scale if scale > 0.0 else np.abs(a - b).max()


def displaced_mol(mol, atom, axis, step):
    """Rebuild the molecule with one Cartesian component displaced."""
    coords = mol.atom_coords(unit='Bohr').copy()
    coords[atom, axis] += step
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    return gto.M(atom=[(s, c) for s, c in zip(symbols, coords)],
                 basis=mol.basis, unit='Bohr', verbose=0)


def pair_symmetrize(g):
    """Project a four-index tensor onto the eightfold ERI symmetry."""
    g = 0.25 * (g + g.transpose(1, 0, 2, 3)
                + g.transpose(0, 1, 3, 2) + g.transpose(1, 0, 3, 2))
    return 0.5 * (g + g.transpose(2, 3, 0, 1))


def main():
    t_start = time.perf_counter()

    print('RI/DF example: H2O/6-31G, CAS(2,2), ncore=4')
    print('=' * 62)

    mol = gto.M(atom=ATOM, basis=BASIS, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    auxmol, label = ri.make_auxmol(mol)
    factors = ri.ri_factors(mol, auxmol)

    print('\n[0] factors')
    print(f'    auxiliary basis          {label}')
    print(f'    nao / naux               {factors.nao} / {factors.naux}')
    print(f'    metric condition number  {factors.cond:.3e}')
    print(f'    metric directions dropped {factors.eig.n_dropped}'
          f'  (rcond {factors.rcond:.0e})')
    print(f'    B shape                  {factors.b.shape}'
          '   (mu, nu, aux)')

    # ------------------------------------------------------------------
    # 1. Assembly
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    eri_ri = ri.ri_eri_ao(mol, auxmol, factors=factors)
    eri_dense = mol.intor('int2e', aosym='s1')
    naux = factors.naux
    bf = factors.b.reshape(-1, naux)
    ref = (bf @ np.linalg.solve(factors.s, bf.T)).reshape(eri_ri.shape)
    t_asm = time.perf_counter() - t0

    delta = eri_ri - eri_dense
    print('\n[1] assembly')
    print(f'    identity residual vs pyscf DF solve   '
          f'{np.abs(eri_ri - ref).max():.3e}')
    print(f'    dense vs RI: RMS                      '
          f'{np.sqrt(np.mean(delta ** 2)):.3e}')
    print(f'    dense vs RI: max abs                  '
          f'{np.abs(delta).max():.3e}')
    print(f'    assembly wall clock                   {t_asm:.2f} s')

    # ------------------------------------------------------------------
    # 2. RI-CASCI against dense CASCI
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    c = mf.mo_coeff
    h1_mo, eri_mo_dense = mo_integrals(c, mf.get_hcore(), eri_dense)
    b_mo, eri_mo_ri = ri.ri_eri_mo(c, factors.b, factors.jhalf)
    nm = NCORE + NACT
    e_dense, ci_dense = cs_casci(h1_mo[:nm, :nm],
                                 eri_mo_dense[:nm, :nm, :nm, :nm],
                                 ncore=NCORE, nact=NACT, nele_act=NELE,
                                 return_ci=True)
    e_ri, ci_ri = cs_casci(h1_mo[:nm, :nm],
                           eri_mo_ri[:nm, :nm, :nm, :nm],
                           ncore=NCORE, nact=NACT, nele_act=NELE,
                           return_ci=True)
    t_casci = time.perf_counter() - t0
    ci_dense = np.asarray(ci_dense).reshape(-1)
    ci_ri = np.asarray(ci_ri).reshape(-1)

    print('\n[2] RI-CASCI')
    print(f'    E_CASCI dense                         {float(e_dense):.10f} Eh')
    print(f'    E_CASCI RI                            {float(e_ri):.10f} Eh')
    print(f'    RI - dense                            '
          f'{float(e_ri) - float(e_dense):+.3e} Eh')
    print(f'    CI overlap                            '
          f'{abs(float(ci_dense @ ci_ri)):.10f}')
    print(f'    wall clock (integrals + 2 CI)         {t_casci:.2f} s')

    # ------------------------------------------------------------------
    # 3. Three-center derivative against central differences
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    worst_three = 0.0
    worst_atom = None
    for atom in range(mol.natm):
        d3 = ri.ri_three_center_deriv(mol, auxmol, atom)
        for axis in range(3):
            mp = displaced_mol(mol, atom, axis, FD_STEP)
            mm = displaced_mol(mol, atom, axis, -FD_STEP)
            ap, _ = ri.make_auxmol(mp)
            am, _ = ri.make_auxmol(mm)
            fd = (ri.ri_three_center(mp, ap)
                  - ri.ri_three_center(mm, am)) / (2.0 * FD_STEP)
            if np.abs(fd).max() < 1e-10:
                continue
            dev = rel_dev(d3[axis], fd)
            if dev > worst_three:
                worst_three = dev
                worst_atom = (atom, axis)
    t_three = time.perf_counter() - t0

    print('\n[3] three-center derivative dB/dR vs central differences')
    print(f'    step                                  {FD_STEP:.0e} Bohr')
    print(f'    worst relative deviation              {worst_three:.3e}'
          f'  (atom {worst_atom[0]}, axis {worst_atom[1]})')
    print(f'    wall clock                            {t_three:.2f} s')

    # ------------------------------------------------------------------
    # 4. Materialized ERI derivative, contraction, and the JAX rule
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    worst_eri = 0.0
    derivs = []
    for atom in range(mol.natm):
        d_eri = ri.ri_eri_deriv(mol, auxmol, atom, factors=factors)
        derivs.append(d_eri)
        for axis in range(3):
            mp = displaced_mol(mol, atom, axis, FD_STEP)
            mm = displaced_mol(mol, atom, axis, -FD_STEP)
            ap, _ = ri.make_auxmol(mp)
            am, _ = ri.make_auxmol(mm)
            fd = (ri.ri_eri_ao(mp, ap)
                  - ri.ri_eri_ao(mm, am)) / (2.0 * FD_STEP)
            if np.abs(fd).max() < 1e-10:
                continue
            worst_eri = max(worst_eri, rel_dev(d_eri[axis], fd))

    rng = np.random.default_rng(5)
    gamma = pair_symmetrize(rng.standard_normal(eri_ri.shape))
    d0 = ri.ri_factors_deriv(mol, auxmol, 0, factors)
    scalar = ri.ri_eri_deriv_contract(gamma, factors, d0)
    vector = ri.ri_eri_deriv_contract(gamma, factors, d0, per_axis=True)
    contract_gap = abs(scalar - float(np.sum(gamma * derivs[0])))

    val, tangent = ri.ri_eri_jvp(factors.b, factors.jhalf,
                                 d0.db[1], d0.djhalf[1])
    jax_gap = rel_dev(np.asarray(tangent), derivs[0][1])
    t_eri = time.perf_counter() - t0

    print('\n[4] ERI derivative and contractions')
    print(f'    d(ERI)/dR vs central differences      {worst_eri:.3e}'
          '   (relative, worst component)')
    print(f'    contraction vs materialized tensor    {contract_gap:.3e}')
    print(f'    per-axis vector vs scalar sum         '
          f'{abs(scalar - float(np.sum(vector))):.3e}')
    print(f'    JAX custom_jvp vs analytic derivative {jax_gap:.3e}')

    eri_jax = np.asarray(ri.ri_eri_from_factors(factors.b, factors.jhalf))
    print(f'    JAX value vs numpy assembly           '
          f'{np.abs(eri_jax - eri_ri).max():.3e}')
    print(f'    wall clock                            {t_eri:.2f} s')

    print('\n' + '=' * 62)
    print(f'total wall clock                          '
          f'{time.perf_counter() - t_start:.2f} s')
    print('RI/DF example finished')


if __name__ == '__main__':
    main()
