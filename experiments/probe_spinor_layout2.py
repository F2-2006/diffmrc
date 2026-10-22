"""Second probe: structure of the decontracted spinor integrals."""
import numpy as np
from pyscf import gto

from diffmrc.core import x2c_df

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]

mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
tr = x2c_df.X2CTransform(mol)
xm = tr.xmol
s = xm.intor_symmetric('int1e_ovlp_spinor')
t = xm.intor_symmetric('int1e_spsp_spinor') * .5
v = xm.intor_symmetric('int1e_nuc_spinor')
w = xm.intor_symmetric('int1e_spnucsp_spinor')
s_sp = xm.intor_symmetric('int1e_ovlp')
t_sp = xm.intor_symmetric('int1e_kin') * .5
print('xmol nao_nr', xm.nao_nr(), 'nao_2c', xm.nao_2c())
print('s shape', s.shape, 'spatial ovlp shape', s_sp.shape)
for name, x in [('s', s), ('t', t), ('v', v), ('w', w)]:
    ee = x[0::2, 0::2]
    oo = x[1::2, 1::2]
    eo = x[0::2, 1::2]
    print(name, 'ee norm %.3e oo norm %.3e eo norm %.3e' %
          (np.linalg.norm(ee), np.linalg.norm(oo), np.linalg.norm(eo)))
    print('   ee vs oo maxdiff %.3e' % np.abs(ee - oo).max())
print('s_ee vs spatial ovlp maxdiff', np.abs(s[0::2, 0::2] - s_sp).max())
print('t_ee vs spatial kin/2 maxdiff', np.abs(t[0::2, 0::2] - t_sp).max())
print('s_eo max', np.abs(s[0::2, 1::2]).max())
print('s_oo eigenvalues head', np.linalg.eigvalsh(s[1::2, 1::2])[:5])
print('s_ee eigenvalues head', np.linalg.eigvalsh(s[0::2, 0::2])[:5])
# what does the s[0::2,1::2] block look like for a single p shell?
# print 6x6 blocks of the O atom's first p shell
print('s 6x6 block rows 2..8, cols 2..8 (decontracted):')
print(np.round(s[2:8, 2:8], 5))
print('s 6x6 cross ee-eo rows 2:8 even, cols 2:8 odd:')
print(np.round(s[2:8:2, 3:8:2], 5))
# compare hcore built by pyscf on the same xmol
from pyscf.x2c import x2c as px
h1 = px._x2c1e_get_hcore(t, v, w, s, 137.035999084)
print('h1 ee/oo maxdiff', np.abs(h1[0::2, 0::2] - h1[1::2, 1::2]).max())
print('h1 eo max', np.abs(h1[0::2, 1::2]).max())
print('h1 ee norm', np.linalg.norm(h1[0::2, 0::2]))
