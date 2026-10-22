"""Probe the spinor basis layout used by the two-component integrals."""
import numpy as np
from pyscf import df, gto

from diffmrc.core import x2c_df

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]

mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
auxmol = df.addons.make_auxmol(mol, auxbasis='def2-universal-jkfit')
tr = x2c_df.X2CTransform(mol)
h = tr.hcore()
s = mol.intor_symmetric('int1e_ovlp_spinor')
fac = x2c_df.spinor_df_factors(mol, auxmol=auxmol)
w = fac.w
n = mol.nao_nr()
print('nao_nr', n, 'nao_2c', mol.nao_2c(), 'xnao', tr.nao)
print('slices_2c', tr.slices)
print('xaoslice', tr.xmol.aoslice_2c_by_atom())
cc = tr.cc
print('cc shape', cc.shape)
nz = np.abs(cc) > 1e-14
print('cc nnz per row (first 6 rows)', nz.sum(axis=1)[:6])
print('cc nnz per col (first 6 cols)', nz.sum(axis=0)[:6])
# block structure tests, block = [first n, last n]
print('h block off-spin max', np.abs(h[:n, n:]).max())
print('h block alpha vs beta maxdiff', np.abs(h[:n, :n] - h[n:, n:]).max())
print('h interleaved off-spin max', np.abs(h[0::2, 1::2]).max())
print('s block off-spin max', np.abs(s[:n, n:]).max())
print('s block alpha vs beta maxdiff', np.abs(s[:n, :n] - s[n:, n:]).max())
print('s interleaved off-spin max', np.abs(s[0::2, 1::2]).max())
print('w block off-spin max', np.abs(w[:n, n:, :]).max())
print('w block alpha vs beta maxdiff', np.abs(w[:n, :n, :] - w[n:, n:, :]).max())
print('w interleaved off-spin max', np.abs(w[0::2, 1::2, :]).max())
print('w hermitian pair residual',
      np.abs(w - w.transpose(1, 0, 2).conj()).max())
# check what the reference helper's contraction is
from pyscf.x2c import x2c as px
helper = px.SpinorX2CHelper(mol)
xmol2, contr2 = helper.get_xmol(mol)
print('contr shape', contr2.shape)
print('contr nnz', (np.abs(contr2) > 1e-14).sum(), 'of', contr2.size)
print('contr row nnz first 6', (np.abs(contr2) > 1e-14).sum(axis=1)[:6])
print('contr col nnz first 6', (np.abs(contr2) > 1e-14).sum(axis=0)[:6])
# compare with interleaved vs block expansion
cc_int = np.zeros_like(cc)
cc_int[0::2, 0::2] = contr2
cc_int[1::2, 1::2] = contr2
cc_blk = np.zeros_like(cc)
cc_blk[:contr2.shape[0], :contr2.shape[1]] = contr2
cc_blk[contr2.shape[0]:, contr2.shape[1]:] = contr2
print('cc equals interleaved', np.abs(cc - cc_int).max())
print('cc equals block', np.abs(cc - cc_blk).max())
