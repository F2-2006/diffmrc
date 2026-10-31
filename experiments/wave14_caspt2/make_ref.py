"""Build and save semicanonical reference data for CASPT2 class studies.

The reference chain matches the frozen anchors: RHF -> CASSCF -> CASCI ->
canonicalize_jax.  Saves the canonical-basis MO integrals, the CI vector,
the determinant bitmaps and the orbital energies as an npz, so later
class-scheme comparisons do not repeat the CASSCF.

Usage:  python make_ref.py <out.npz> <system>
        system in {h2o, co, n2}
"""
import sys

import numpy as np
import jax.numpy as jnp

from pyscf import gto, scf, mcscf

from diffmrc.core.casci import cs_casci
from diffmrc.core.nevpt2.impl import canonicalize_jax, cas_dets_for2

SYSTEMS = {
    'h2o': dict(atom='O 0 0 0; H 0.7557 0 0.588; H -0.7557 0 0.588',
                ncore=4, nact=2, nele=2),
    'co': dict(atom='C 0 0 0; O 0 0 1.1283', ncore=6, nact=2, nele=2),
    'n2': dict(atom='N 0 0 0; N 0 0 1.0977', ncore=4, nact=6, nele=6),
}


def ints(Cm, h1e, eri):
    h1 = jnp.einsum('ai,ab,bj->ij', Cm, h1e, Cm)
    z = jnp.einsum('pi,pqrs->iqrs', Cm, eri)
    z = jnp.einsum('qj,iqrs->ijrs', Cm, z)
    z = jnp.einsum('rk,ijrs->ijks', Cm, z)
    return h1, jnp.einsum('sl,ijks->ijkl', Cm, z)


def main(out, system):
    cfg = SYSTEMS[system]
    ncore, nact, nele = cfg['ncore'], cfg['nact'], cfg['nele']
    mol = gto.M(atom=cfg['atom'], basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, nact, nele)
    mc.kernel()
    C = jnp.asarray(mc.mo_coeff)
    h1e = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    eri = jnp.asarray(mol.intor('int2e', aosym='s1'))
    h1_mo, eri_mo = ints(C, h1e, eri)
    nm = ncore + nact
    e_ci, ci = cs_casci(h1_mo[:nm, :nm], eri_mo[:nm, :nm, :nm, :nm],
                        ncore=ncore, nact=nact, nele_act=nele, nroots=1,
                        return_ci=True)
    C2, ci2, eps = canonicalize_jax(C, h1e, eri, ci, ncore=ncore,
                                    nact=nact, nele_act=nele)
    h1c, eric = ints(C2, h1e, eri)
    E2_ci, ci3 = cs_casci(h1c[:nm, :nm], eric[:nm, :nm, :nm, :nm],
                          ncore=ncore, nact=nact, nele_act=nele, nroots=1,
                          return_ci=True)
    dets = cas_dets_for2(ncore, nact, nele)
    np.savez(out, h1_mo=np.asarray(h1c), eri_mo=np.asarray(eric),
             ci=np.asarray(ci3).ravel(), dets=np.asarray(dets, dtype=np.int64),
             eps=np.asarray(eps, dtype=float), ncore=ncore, nact=nact,
             nele=nele, casscf=float(mc.e_tot), rhf=float(mf.e_tot),
             ci_energy=float(E2_ci))
    print('%s: CASSCF %.12f  CI %.12f  nmo %d' % (
        system, float(mc.e_tot), float(E2_ci), mol.nao_nr()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'h2o')
