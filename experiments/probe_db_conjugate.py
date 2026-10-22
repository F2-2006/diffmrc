#!/usr/bin/env python
"""Check the role of the complex conjugate in the second-AO-center dB term.

Compares the analytic three-center derivative with and without the
conjugate against a geometry finite difference for one atom/direction.
"""
import numpy as np
from pyscf import df, gto
from pyscf.gto.mole import ATOM_OF

GEOM = [('O', (0.0, 0.0, 0.0)), ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'


def build(coords=None):
    m = gto.M(atom=[(s, tuple(x)) for s, x in GEOM], basis='6-31g',
              unit='Angstrom', verbose=0)
    if coords is not None:
        m.set_geom_(coords, unit='Bohr')
    return m


def aux_mask(auxmol, ia):
    loc = auxmol.ao_loc_nr()
    mask = np.zeros(auxmol.nao_nr(), dtype=bool)
    for ib in range(auxmol.nbas):
        if int(auxmol._bas[ib, ATOM_OF]) == ia:
            mask[loc[ib]:loc[ib + 1]] = True
    return mask


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


mol = build()
auxmol = df.addons.make_auxmol(mol, auxbasis=AUX)
ia, ix, step = 0, 2, 1e-4
coords = mol.atom_coords()
cp = coords.copy(); cp[ia, ix] += step
cm = coords.copy(); cm[ia, ix] -= step
mp, mm = build(cp), build(cm)
ap, am = df.addons.make_auxmol(mp, auxbasis=AUX), df.addons.make_auxmol(mm, auxbasis=AUX)
bp = df.incore.aux_e2(mp, ap, intor='int3c2e_spinor', aosym='s1')
bm = df.incore.aux_e2(mm, am, intor='int3c2e_spinor', aosym='s1')
db_fd = (bp - bm) / (2 * step)

ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1_spinor', aosym='s1')
ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2_spinor', aosym='s1')
p0, p1 = mol.aoslice_2c_by_atom()[ia, 2:]
m_ao = np.zeros(mol.nao_2c(), dtype=bool); m_ao[p0:p1] = True
m_aux = aux_mask(auxmol, ia)
aux = ip2[ix] * m_aux[None, None, :]
first = ip1[ix] * m_ao[:, None, None]
second_conj = ip1[ix].transpose(1, 0, 2).conj() * m_ao[None, :, None]
second_plain = ip1[ix].transpose(1, 0, 2) * m_ao[None, :, None]
with_conj = -(first + second_conj + aux)
without = -(first + second_plain + aux)
print('with conjugate   rel %.3e' % rel(with_conj, db_fd))
print('without conjugate rel %.3e' % rel(without, db_fd))
