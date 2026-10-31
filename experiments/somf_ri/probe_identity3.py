#!/usr/bin/env python
"""Verify the derivative identity and the three-center RI factorization.

libcint defines (src/autocode/intor2.c)

    int2e_p1vxp1[K] = eps_{KLM} (d_L phi_i d_M phi_j | 1/r12 | phi_k phi_l)

with both derivatives on the bra pair.  This probe checks

  (1) the four-center identity against int2e_ipvip1 (9 components, the
      component order is L * 3 + M, L on the first AO, M on the second),
  (2) the three-center factors from int3c2e_ipvip1 / int3c2e_sph,
  (3) the Coulomb-metric RI contraction of the J term,

for water, OH and CH2O in 6-31G.

Run:
  PYTHONPATH=<pkg>/src python probe_identity3.py
"""
import json
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import numpy as np
from pyscf import df, gto

EPS = np.zeros((3, 3, 3))
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1.0

SYSTEMS = {
    'water': ('O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0', '6-31g', 0),
    'oh': ('O 0 0 0; H 0 0 1.8324', '6-31g', 1),
    'ch2o': ('C 0 0 0; O 0 0 1.2; H 0.94 0 0.6; H -0.94 0 0.6', '6-31g', 0),
}


def cross(t9):
    """A[K] = eps_{KLM} t[L * 3 + M] for t9 of shape (9, ...)."""
    t = t9.reshape(3, 3, *t9.shape[1:])
    return np.einsum('klm,lm...->k...', EPS, t)


def fit(ref, cand):
    den = float(np.sum(cand * cand))
    num = float(np.sum(ref * cand))
    c = num / den
    res = float(np.abs(ref - c * cand).max())
    return c, res, res / float(np.abs(ref).max())


def run(name, atom, basis, spin):
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', spin=spin, verbose=0)
    nao = mol.nao_nr()
    g = mol.intor('int2e_p1vxp1', 3).reshape(3, nao, nao, nao, nao)
    t4 = mol.intor('int2e_ipvip1', 9).reshape(9, nao, nao, nao, nao)
    cand = cross(t4)
    const, resid, rel = fit(g, cand)

    aux = df.addons.make_auxmol(mol, auxbasis='def2-universal-jkfit')
    naux = aux.nao_nr()
    t3 = df.incore.aux_e2(mol, aux, intor='int3c2e_ipvip1', aosym='s1')
    b3 = df.incore.aux_e2(mol, aux, intor='int3c2e_sph', aosym='s1')
    s = aux.intor('int2c2e_sph', aosym='s1')
    s = 0.5 * (s + s.T)
    print(name, 'nao', nao, 'naux', naux,
          't3 shape', np.shape(t3), 'b3 shape', np.shape(b3))

    t3 = t3.reshape(9, nao, nao, naux)
    a3 = cross(t3)                       # (3, nao, nao, naux)
    b3 = b3.reshape(nao, nao, naux)      # (nao, nao, naux)

    # Synthetic physical-shaped density: trace(P S) = nelectron.
    ovlp = mol.intor('int1e_ovlp')
    p = (mol.nelectron / nao) * np.linalg.inv(ovlp)
    p = 0.5 * (p + p.T)

    j_exact = np.einsum('rs,kpqrs->kpq', p, g)
    w = np.einsum('uvP,uv->P', b3, p)
    z = np.linalg.solve(s, w)
    j_ri = np.einsum('kpqP,P->kpq', a3, z)
    # Same contraction without RI, as a control on the identity algebra.
    j_id = np.einsum('rs,kpqrs->kpq', p, cand)

    out = dict(
        system=name, basis=basis, nao=int(nao), naux=int(naux),
        identity=dict(constant=const, max_abs_residual=resid,
                      relative=rel),
        j_exact_norm=float(np.abs(j_exact).max()),
        j_identity_vs_exact=float(np.abs(j_id - const * j_exact).max()),
        j_ri_vs_exact=fit(j_exact, j_ri),
        metric_cond=float(np.linalg.cond(s)),
    )
    print(json.dumps(out, indent=2))
    return out


def main():
    results = {}
    for name, (atom, basis, spin) in SYSTEMS.items():
        results[name] = run(name, atom, basis, spin)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'probe_identity3.json'), 'w') as fh:
        json.dump(results, fh, indent=2)


if __name__ == '__main__':
    main()
