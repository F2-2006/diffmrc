#!/usr/bin/env python
"""RI-NEVPT2: the same second-order correction from RI and dense integrals.

System: H2O / 6-31G, CAS(2,2) with ncore = 4 (frozen oxygen 1s).

The RI three-center tensor is transformed to the MO basis by
``diffmrc.core.ri.ri_eri_mo`` and handed to the NEVPT2 contraction
``diffmrc.core.nevpt2.impl.nevpt2_e2_full2`` exactly where the dense MO
ERI would enter.  Three numbers are reported:

  * ``e2 dense``          dense ERI, dense CI vector;
  * ``e2 RI, same CI``    RI ERI, the dense CI vector (isolates the
                          integral approximation);
  * ``e2 RI, end to end`` RI ERI and the CI vector solved from it
                          (includes the small CI shift).

This is a fixed-orbital comparison: the RHF orbitals are not relaxed for
the active space, so the absolute e2 is not a production NEVPT2 value.
The quantity being validated is the RI-minus-dense difference, which is a
property of the integrals alone.

Run from the repository root, with the package importable:

    python examples/ri_nevpt2.py
    PYTHONPATH=src python examples/ri_nevpt2.py
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import time

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp  # noqa: E402

from pyscf import gto, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402

from diffmrc.core import ri  # noqa: E402
from diffmrc.core.casci import cs_casci  # noqa: E402
from diffmrc.core.nevpt2.impl import cas_dets_for2, nevpt2_e2_full2  # noqa: E402


ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
BASIS = '6-31g'
NCORE, NACT, NELE = 4, 2, 2


def mo_integrals(c, h1e, eri_ao):
    """MO integrals by four sequential two-operand contractions."""
    h1 = c.T @ h1e @ c
    t = np.einsum('pi,pqrs->iqrs', c, eri_ao)
    t = np.einsum('qj,iqrs->ijrs', c, t)
    t = np.einsum('rk,ijrs->ijks', c, t)
    eri_mo = np.einsum('sl,ijks->ijkl', c, t)
    return h1, eri_mo


def spin_links(nact, nele):
    """Per-spin CI link tables, the convention of the driver layer.

    ``nevpt2_e2_full2`` takes one link table per spin for the
    single-external class; the tables depend only on the active space.
    """
    na = nele // 2
    nb = nele - na
    la = cistring.gen_linkstr_index(range(nact), na)
    lb = cistring.gen_linkstr_index(range(nact), nb)
    fmt = lambda links: [[tuple(x) for x in links[s]]  # noqa: E731
                         for s in range(len(links))]
    return fmt(la), fmt(lb)


def nevpt2_correction(h1_mo, eri_mo, c_ci, dets, e0, mo_energy, links):
    """The strong-correction energy at fixed orbitals."""
    e2, _ = nevpt2_e2_full2(
        jnp.asarray(h1_mo), jnp.asarray(eri_mo), jnp.asarray(c_ci),
        dets, e0, mo_energy=jnp.asarray(mo_energy),
        ncore=NCORE, nact=NACT, nocc=NCORE + NACT,
        links_a=links[0], links_b=links[1])
    return float(e2)


def casci(h1_mo, eri_mo):
    nm = NCORE + NACT
    e, ci = cs_casci(h1_mo[:nm, :nm], eri_mo[:nm, :nm, :nm, :nm],
                     ncore=NCORE, nact=NACT, nele_act=NELE,
                     return_ci=True)
    return float(e), np.asarray(ci).reshape(-1)


def main():
    t_start = time.perf_counter()

    print('RI-NEVPT2 example: H2O/6-31G, CAS(2,2), ncore=4')
    print('=' * 62)

    mol = gto.M(atom=ATOM, basis=BASIS, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    auxmol, label = ri.make_auxmol(mol)
    factors = ri.ri_factors(mol, auxmol)

    print('\n[0] setup')
    print(f'    auxiliary basis          {label}')
    print(f'    nao / naux               {factors.nao} / {factors.naux}')
    print(f'    metric condition number  {factors.cond:.3e}')

    c = mf.mo_coeff
    h1_mo, eri_mo_dense = mo_integrals(c, mf.get_hcore(),
                                       mol.intor('int2e', aosym='s1'))
    _, eri_mo_ri = ri.ri_eri_mo(c, factors.b, factors.jhalf)

    e_dense, ci_dense = casci(h1_mo, eri_mo_dense)
    e_ri, ci_ri = casci(h1_mo, eri_mo_ri)

    print('\n[1] CASCI reference')
    print(f'    E_CASCI dense                         {e_dense:.10f} Eh')
    print(f'    E_CASCI RI                            {e_ri:.10f} Eh')
    print(f'    RI - dense                            {e_ri - e_dense:+.3e} Eh')
    print(f'    CI overlap                            '
          f'{abs(float(ci_dense @ ci_ri)):.10f}')

    dets = cas_dets_for2(NCORE, NACT, NELE)
    links = spin_links(NACT, NELE)
    mo_energy = mf.mo_energy

    print('\n[2] NEVPT2 correction (fixed RHF orbitals)')
    t0 = time.perf_counter()
    e2_dense = nevpt2_correction(h1_mo, eri_mo_dense, ci_dense, dets,
                                 e_dense, mo_energy, links)
    e2_ri_same = nevpt2_correction(h1_mo, eri_mo_ri, ci_dense, dets,
                                   e_ri, mo_energy, links)
    e2_ri = nevpt2_correction(h1_mo, eri_mo_ri, ci_ri, dets,
                              e_ri, mo_energy, links)
    t_e2 = time.perf_counter() - t0

    print(f'    e2 dense                              {e2_dense:+.10f} Eh')
    print(f'    e2 RI, same CI                        {e2_ri_same:+.10f} Eh')
    print(f'    e2 RI, end to end                     {e2_ri:+.10f} Eh')
    print(f'    RI - dense, same CI                   '
          f'{e2_ri_same - e2_dense:+.3e} Ha')
    print(f'    RI - dense, end to end                '
          f'{e2_ri - e2_dense:+.3e} Ha')
    print(f'    wall clock (3 contractions)           {t_e2:.2f} s')

    print('\n' + '=' * 62)
    print(f'total wall clock                          '
          f'{time.perf_counter() - t_start:.2f} s')
    print('RI-NEVPT2 example finished')


if __name__ == '__main__':
    main()
