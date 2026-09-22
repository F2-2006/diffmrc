#!/usr/bin/env python
"""Does the pseudo-inverse cutoff change a physical answer?

Two cutoffs coexist in this project's history:

* a *relative* one, ``numpy.linalg.pinv(H, rcond=1e-9)``, used by the
  implicit-Hessian route;
* an *absolute* one, keeping singular values above 1e-3, used by an
  earlier comparison script.

On a synthetic matrix with condition number 1e12 they retain five and two
singular values respectively and their solutions differ by 99.9%. That is
alarming only if the discarded directions couple to the mixed derivative
M = d2E/dtheta dR. If they do not, the two conventions agree on every
physical quantity and the choice is cosmetic.

This script measures the coupling on a real system rather than assuming
either way. It reports, for each convention:

* how many singular values of H_tt survive,
* the largest component of M along the discarded directions, relative to
  the largest component of M overall,
* the resulting dtheta*/dR and how much the two differ.

The verdict at the bottom says whether the choice matters for this
system, and with what margin.

Usage:

    python verify_rcond_choice.py [H2O|CO|N2] [basis] [nthread]
"""
import os
import sys

_SYS = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
_BASIS = sys.argv[2] if len(sys.argv) > 2 else '6-31g'
_NT = sys.argv[3] if len(sys.argv) > 3 else '4'

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = _NT
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp

from pyscfad import gto as pgto, scf as pscf

from diffmrc.core.rotation import block_rotation
from diffmrc.core.orblib import orbital_pairs
from diffmrc.core.casci import cs_casci_blocked
from diffmrc.core.optimizer import optimize_theta

RELATIVE_RCOND = 1e-9
ABSOLUTE_CUTOFF = 1e-3

SYSTEMS = {
    'H2O': ('O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880', 4, 2, 2),
    'CO':  ('C 0 0 0; O 0 0 1.1283', 5, 2, 2),
    'N2':  ('N 0 0 0; N 0 0 1.0977', 5, 2, 2),
}


def main():
    if _SYS not in SYSTEMS:
        print(f'unknown system {_SYS!r}; expected one of {list(SYSTEMS)}')
        return 1
    atom, ncore, nact, nele = SYSTEMS[_SYS]

    print('=' * 76)
    print(f'Pseudo-inverse cutoff: does it change the answer?  {_SYS}/{_BASIS}')
    print('=' * 76)

    mol = pgto.Mole()
    mol.atom = atom
    mol.basis = _BASIS
    mol.build()
    nmo = mol.nao_nr()
    nR = 3 * mol.natm

    mf = pscf.RHF(mol)
    mf.kernel()
    C0 = mf.mo_coeff
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    npar = len(pairs)

    def E_cas(C):
        Cj = jnp.asarray(C)
        nm = ncore + nact
        h1 = jnp.einsum('ai,ab,bj->ij', Cj, h1e, Cj)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                        eri)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + enuc

    E_theta = lambda t: E_cas(C0 @ block_rotation(t, nmo, pairs))
    _, th_star, gnorm = optimize_theta(None, None, E_theta, npar, steps=2000,
                                       g_conv=1e-12,
                                       theta0=np.zeros(npar))
    th_star = jnp.asarray(np.asarray(th_star))
    print(f'  theta*:  |g| = {gnorm:.3e}   npar = {npar}   nR = {nR}')

    def E_joint(m_, th_):
        mf_ = pscf.RHF(m_)
        mf_.kernel()
        C_ = mf_.mo_coeff
        h_ = m_.intor('int1e_kin') + m_.intor('int1e_nuc')
        e_ = m_.intor('int2e', aosym='s1')
        Cj = C_ @ block_rotation(th_, nmo, pairs)
        nm = ncore + nact
        h1 = jnp.einsum('ai,ab,bj->ij', Cj, h_, Cj)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                        e_)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + m_.energy_nuc()

    H_tt = np.asarray(jax.hessian(lambda t: E_theta(t))(th_star))
    H_tt = 0.5 * (H_tt + H_tt.T)
    g_of = jax.grad(E_joint, argnums=1)
    M = np.asarray(jax.jacfwd(g_of, argnums=0)(mol, th_star).coords
                   ).reshape(npar, nR)

    # ---- the two conventions ----
    U, s, Vt = np.linalg.svd(H_tt)
    keep_rel = s > RELATIVE_RCOND * s[0]
    keep_abs = s > ABSOLUTE_CUTOFF

    def pinv(mask):
        return (Vt[mask].T / s[mask]) @ U[:, mask].T

    D_rel = -pinv(keep_rel) @ M
    D_abs = -pinv(keep_abs) @ M

    print()
    print('  singular values retained')
    print(f'    relative  rcond={RELATIVE_RCOND:g} : {int(keep_rel.sum())} '
          f'of {s.size}')
    print(f'    absolute  s > {ABSOLUTE_CUTOFF:g}      : {int(keep_abs.sum())} '
          f'of {s.size}')
    print(f'    spectrum: max={s[0]:.4e}  min={s[-1]:.4e}  '
          f'kappa={s[0]/max(s[-1], 1e-300):.2e}')

    dropped = keep_rel & ~keep_abs
    if dropped.any():
        # The question the whole script exists to answer: does M have any
        # component along the directions one convention discards?
        M_energy = float(np.abs(M).max())
        U_drop = U[:, dropped]
        leaked = np.linalg.norm(U_drop.T @ M, axis=0)
        leak_max = float(leaked.max())
        print()
        print('  coupling of M to the discarded directions')
        print(f'    directions in relative-but-not-absolute: '
              f'{int(dropped.sum())}')
        print(f'    |M|_max                    = {M_energy:.4e}')
        print(f'    |M| along discarded, max   = {leak_max:.4e}')
        if M_energy > 0:
            print(f'    ratio                      = {leak_max / M_energy:.3e}')
        leak_ratio = leak_max / max(M_energy, 1e-300)
    else:
        leak_ratio = 0.0
        print()
        print('  the two conventions retain the same directions')
        print('  => the choice cannot matter for this system')

    print()
    print('  resulting orbital response')
    print(f'    |dtheta*/dR|_max  relative = {np.abs(D_rel).max():.4e}')
    print(f'    |dtheta*/dR|_max  absolute = {np.abs(D_abs).max():.4e}')
    denom = max(np.abs(D_rel).max(), 1e-300)
    rel_diff = float(np.abs(D_rel - D_abs).max() / denom)
    print(f'    relative difference          = {rel_diff:.3e}')

    print()
    print('=' * 76)
    if not dropped.any():
        verdict = ('the two conventions are identical for this system'
                   '\n  => either may be used; keep the relative one for '
                   'its scale invariance')
    elif leak_ratio < 1e-6 and rel_diff < 1e-6:
        verdict = ('the conventions differ only in directions M does not '
                   'couple to'
                   '\n  => physically equivalent for this system; the '
                   'relative cutoff is the safer default')
    else:
        verdict = ('the conventions give materially different responses'
                   '\n  => the choice must be recorded with every result, '
                   'and the two must not be compared across conventions')
    print(f'VERDICT: {verdict}')
    print('=' * 76)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
