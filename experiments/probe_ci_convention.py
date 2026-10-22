"""Isolate the CI convention on the spatial DF-CASSCF integrals."""
import numpy as np
import scipy.linalg
from pyscf import df, fci, gto
from pyscf.x2c import sfx2c1e

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]


def main():
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis='def2-universal-jkfit')
    h = sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)
    s = mol.intor('int1e_ovlp')
    b = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')
    saux = auxmol.intor('int2c2e_sph', aosym='s1')
    ev, u = scipy.linalg.eigh(saux)
    keep = ev > 1e-12
    print('metric eig min/max', ev.min(), ev.max(),
          'n_keep', int(keep.sum()), 'n_neg', int((ev < 0).sum()))
    jh = (u[:, keep] / np.sqrt(ev[keep])) @ u[:, keep].T
    w = np.matmul(b, jh)
    nocc = mol.nelectron // 2
    e, c = scipy.linalg.eigh(h, s)
    d = c[:, :nocc] @ c[:, :nocc].T
    for it in range(100):
        rho = np.einsum('ijP,ji->P', w, d)
        j = np.einsum('ijP,P->ij', w, rho)
        k = np.zeros_like(d)
        for p in range(w.shape[2]):
            wp = w[:, :, p]
            k += wp @ d @ wp
        f = h + 2 * j - k
        en = (2 * np.einsum('ij,ji->', d, h) + 2 * np.sum(rho ** 2)
              - np.trace(d @ k))
        if it and abs(en - e_old) < 1e-13:
            break
        e_old = en
        e, c = scipy.linalg.eigh(f, s)
        d = c[:, :nocc] @ c[:, :nocc].T
    print('E_elec_rhf', en, 'E_tot', en + mol.energy_nuc())
    print('orbital energies', np.round(e, 6))
    h1 = c.T @ h @ c
    wm = np.einsum('am,abP,bn->mnP', c, w, c)
    eri = np.einsum('abP,cdP->abcd', wm, wm)
    # determinant energy over the 5 occupied spatial orbitals
    e_det = 2 * np.trace(h1[:nocc, :nocc])
    for i in range(nocc):
        for j in range(nocc):
            e_det += 2 * eri[i, i, j, j] - eri[i, j, j, i]
    print('E_det_from_mo', e_det)
    print('E_det - E_elec', e_det - en)
    e2_ao = 2 * np.sum(rho ** 2) - np.trace(d @ k)
    e2_mo = 0.0
    for i in range(nocc):
        for j in range(nocc):
            e2_mo += 2 * eri[i, i, j, j] - eri[i, j, j, i]
    print('E2 AO', e2_ao, 'E2 MO', e2_mo, 'diff', e2_mo - e2_ao)
    print('orthonormality c.T s c - I', np.abs(c.T @ s @ c - np.eye(c.shape[1])).max())
    exact_ao = mol.intor('int2e_sph', aosym='s1')
    print('exact (00|11) AO-MO',
          np.einsum('a,b,c,d,abcd->', c[:, 0], c[:, 0], c[:, 1], c[:, 1], exact_ao))
    print('exact (00|00) AO-MO',
          np.einsum('a,b,c,d,abcd->', c[:, 0], c[:, 0], c[:, 0], c[:, 0], exact_ao))
    print('(00|11) MO', eri[0, 0, 1, 1], 'AO', np.dot(wm[0, 0], wm[1, 1]))
    print('wm00 norm', np.linalg.norm(wm[0, 0]), 'wm11 norm', np.linalg.norm(wm[1, 1]))
    df_ao = np.einsum('abP,cdP->abcd', w, w)
    c0, c1 = c[:, 0], c[:, 1]
    print('DF (00|11) 4idx', np.einsum('a,b,c,d,abcd->', c0, c0, c1, c1, df_ao))
    print('DF (00|11) wm  ', np.dot(wm[0, 0], wm[1, 1]))
    print('exact AO (0 0|1 1)', exact_ao[0, 0, 1, 1],
          'DF AO (0 0|1 1)', df_ao[0, 0, 1, 1])
    print('exact AO (0 0|0 0)', exact_ao[0, 0, 0, 0],
          'DF AO (0 0|0 0)', df_ao[0, 0, 0, 0])
    print('b shape', b.shape, 'saux shape', saux.shape)
    print('saux eig min/max kept', ev[keep].min(), ev[keep].max())
    b00 = b[0, 0, :]
    b11 = b[1, 1, :]
    print('b00 norm', np.linalg.norm(b00), 'b11 norm', np.linalg.norm(b11))
    print('(00|11) via solve', b00 @ np.linalg.solve(saux, b11))
    print('(00|00) via solve', b00 @ np.linalg.solve(saux, b00))
    print('(00|11) via w', np.dot(w[0, 0], w[1, 1]))
    print('ao (00|11)', sum(w[:, :, P][0, 0] * w[:, :, P][1, 1] for P in range(w.shape[2])) if False else 'skip')
    print('h1 diag occupied', np.round(np.diag(h1)[:nocc], 6))
    print('(ii|ii) first 5', np.round([eri[i, i, i, i] for i in range(nocc)], 6))
    # pyscf FCI in the full 5-orbital occupied space, 10 electrons
    e_fci5 = fci.direct_spin1.kernel(h1[:nocc, :nocc], eri[:nocc, :nocc, :nocc, :nocc],
                                     nocc, (nocc, nocc), nroots=1)[0]
    print('FCI 5 orb 10e', e_fci5, 'diff to det', e_fci5 - e_det)
    # our cs_casci on the same 5 orbitals
    from diffmrc.core.casci import cs_casci
    e_cs5 = float(cs_casci(h1[:nocc, :nocc], eri[:nocc, :nocc, :nocc, :nocc],
                           ncore=0, nact=nocc, nele_act=10, nroots=1))
    print('cs_casci 5 orb 10e', e_cs5, 'diff to fci', e_cs5 - e_fci5)
    # cs_casci with a frozen core of 4
    e_cs = float(cs_casci(h1[:6, :6], eri[:6, :6, :6, :6], ncore=4, nact=2,
                          nele_act=2, nroots=1))
    print('cs_casci ncore=4 nact=2 nele=2', e_cs, 'diff to det', e_cs - e_det)
    # frozen-core FCI reference
    nc, na = 4, 2
    h_act = h1[nc:nc + na, nc:nc + na].copy()
    eri_act = eri[nc:nc + na, nc:nc + na, nc:nc + na, nc:nc + na].copy()
    e_core = 2 * np.trace(h1[:nc, :nc])
    for i in range(nc):
        for j in range(nc):
            e_core += 2 * eri[i, i, j, j] - eri[i, j, j, i]
    for i in range(nc):
        h_act += 2 * eri[nc:nc + na, nc:nc + na, i, i]
        h_act -= eri[nc:nc + na, i, i, nc:nc + na]
    ef = fci.direct_spin1.kernel(h_act, eri_act, na, (1, 1), nroots=1)[0]
    print('frozen-core FCI', e_core + ef, 'diff to cs', e_core + ef - e_cs)


if __name__ == '__main__':
    main()
