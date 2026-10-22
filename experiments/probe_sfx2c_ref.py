"""Probe the spin-free (scalar) X2C path: hcore, derivatives, pyscf CASSCF."""
import json

import numpy as np
import scipy.linalg
from pyscf import df, gto, mcscf, scf
from pyscf.x2c import sfx2c1e
from pyscf.x2c.sfx2c1e_grad import hcore_grad_generator

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'


def build(coords=None):
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
    return mol


def main():
    out = {}
    mol = build()
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUX)
    helper = sfx2c1e.SpinFreeX2CHelper(mol)
    h = helper.get_hcore(mol)
    out['h_shape'] = list(h.shape)
    out['h_sym'] = float(np.abs(h - h.T).max())
    # derivative vs finite difference of the hcore
    dh = hcore_grad_generator(helper, mol)
    natm = mol.natm
    coords0 = mol.atom_coords()
    step = 1e-3
    errs = []
    for ia in range(natm):
        da = np.asarray(dh(ia))
        for x in range(3):
            cp = coords0.copy()
            cp[ia, x] += step
            hp = helper.get_hcore(build(cp))
            cm = coords0.copy()
            cm[ia, x] -= step
            hm = helper.get_hcore(build(cm))
            fd = (hp - hm) / (2 * step)
            errs.append(float(np.abs(fd - da[x]).max()))
    out['dh_vs_fd_max'] = max(errs)
    out['dh_scale'] = float(max(np.abs(np.asarray(dh(ia))).max()
                                for ia in range(natm)))
    out['e_nuc'] = float(mol.energy_nuc())
    # pyscf CASSCF with the spin-free X2C hcore, exact ERI
    mf = scf.RHF(mol)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    out['e_rhf_sfx2c'] = float(mf.e_tot)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    out['e_casscf_sfx2c'] = float(mc.e_tot)
    try:
        g = mc.nuc_grad_method().kernel()
        out['casscf_grad'] = g.tolist()
        out['casscf_grad_max'] = float(np.abs(g).max())
    except Exception as ex:
        out['casscf_grad_error'] = '%s: %s' % (type(ex).__name__, str(ex)[:80])
    # pyscf CASSCF with density fitting
    try:
        mf2 = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf2 = sfx2c1e.sfx2c1e(mf2)
        mf2.conv_tol = 1e-12
        mf2.kernel()
        out['e_rhf_sfx2c_df'] = float(mf2.e_tot)
        mc2 = mcscf.CASSCF(mf2, 2, 2)
        mc2.conv_tol = 1e-12
        mc2.max_cycle = 200
        mc2.kernel()
        out['e_casscf_sfx2c_df'] = float(mc2.e_tot)
        out['df_minus_exact'] = float(mc2.e_tot - mc.e_tot)
    except Exception as ex:
        out['df_error'] = '%s: %s' % (type(ex).__name__, str(ex)[:120])
    # our own spatial DF RHF energy
    s = mol.intor('int1e_ovlp')
    b = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')
    saux = auxmol.intor('int2c2e_sph', aosym='s1')
    e, u = scipy.linalg.eigh(saux)
    keep = e > 1e-12
    jh = (u[:, keep] / np.sqrt(e[keep])) @ u[:, keep].T
    w = np.matmul(b, jh)
    nocc = mol.nelectron // 2
    d = np.zeros_like(h)
    e2, c2 = scipy.linalg.eigh(h, s)
    d = c2[:, :nocc] @ c2[:, :nocc].T
    for _ in range(100):
        rho = np.einsum('ijP,ji->P', w, d)
        j = np.einsum('ijP,P->ij', w, rho)
        k = np.zeros_like(d)
        for p_ in range(w.shape[2]):
            wp = w[:, :, p_]
            k += wp @ d @ wp
        f = h + 2 * j - k
        e_tot = (2 * np.einsum('ij,ji->', d, h) + 2 * np.sum(rho ** 2)
                 - np.trace(d @ k))
        if _ and abs(e_tot - e_old) < 1e-13:
            break
        e_old = e_tot
        e2, c2 = scipy.linalg.eigh(f, s)
        d = c2[:, :nocc] @ c2[:, :nocc].T
    out['e_rhf_spatial_df'] = float(e_tot)
    out['rhf_df_minus_pyscf_df'] = float(e_tot - out.get('e_rhf_sfx2c_df',
                                                         np.nan))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
