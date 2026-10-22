"""Numerical probes that pin the conventions of the DF-X2C CASSCF layer.

The scalar (spin-free) X2C Hamiltonian is used: the one-electron operator is
``SpinFreeX2CHelper`` and the two-electron part is the non-relativistic
Coulomb in the density-fitted spatial basis.  This is the real/spin-free
limit of the two-component module in ``x2c_df``; the j-adapted two-component
basis has no spin label and its complex CI is out of scope here.

Probe list
----------
A. The spatial DF RHF against pyscf's density-fitted sfx2c1e RHF.
B. The CI layer: our determinant Hamiltonian against an independent
   frozen-core FCI (pyscf ``fci.direct_spin1``) on the same integrals.
C. The 1- and 2-RDM against finite differences of the CI eigenvalue.
D. The generalized Fock against the SCF Fock matrix in the
   single-determinant limit.
E. The analytic orbital (theta) gradient against a finite difference.
"""
import json
from itertools import combinations

import numpy as np
import scipy.linalg
from pyscf import df, fci, gto, scf
from pyscf.x2c import sfx2c1e

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'
NCORE, NACT, NELE = 4, 2, 2


def build(coords=None):
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
    return mol


def spatial_rhf(h, s, w, nelec, conv_tol=1e-13, max_cycle=200):
    """Closed-shell RHF in the spatial basis with DF two-electron integrals."""
    nao = h.shape[0]
    nocc = nelec // 2
    e, c = scipy.linalg.eigh(h, s)
    d = c[:, :nocc] @ c[:, :nocc].T
    e_old = None
    for it in range(max_cycle):
        rho = np.einsum('ijP,ji->P', w, d)
        j = np.einsum('ijP,P->ij', w, rho)
        k = np.zeros_like(d)
        for p in range(w.shape[2]):
            wp = w[:, :, p]
            k += wp @ d @ wp
        f = h + 2 * j - k
        e_tot = (2 * np.einsum('ij,ji->', d, h) + 2 * np.sum(rho ** 2)
                 - np.trace(d @ k))
        err = f @ d @ s - s @ d @ f
        gnorm = np.linalg.norm(err)
        if e_old is not None and abs(e_tot - e_old) < conv_tol and gnorm < 1e-9:
            break
        e_old = e_tot
        e, c = scipy.linalg.eigh(f, s)
        d = c[:, :nocc] @ c[:, :nocc].T
    return dict(e_tot=e_tot, mo=c, mo_e=e, dm=d, gnorm=gnorm, n_iter=it + 1)


def det_list(nso, nele):
    return [tuple(c) for c in combinations(range(nso), nele)]


def _annihilators(dets_n, dets_nm1, nso):
    """A_p: (N-1, N) annihilation matrices with the sorted-occupation sign."""
    index = {d: i for i, d in enumerate(dets_nm1)}
    A = []
    for p in range(nso):
        m = np.zeros((len(dets_nm1), len(dets_n)))
        for j, d in enumerate(dets_n):
            if p in d:
                sign = (-1.0) ** d.index(p)
                new = tuple(x for x in d if x != p)
                i = index.get(new)
                if i is not None:
                    m[i, j] = sign
        A.append(m)
    return A


def ci_rdms(vec, nso, nele):
    """gamma and Gamma for a CI vector, in the full spin-orbital space.

    gamma[p,q] = <a^dag_p a_q>, Gamma[p,q,r,s] = <a^dag_p a^dag_r a_s a_q>.
    The determinant list of the CI (core + active combinations) is a subset
    of the full N-electron list; the vector is expanded onto the full list.
    """
    dets_n = det_list(nso, nele)
    dets_n1 = det_list(nso, nele - 1)
    dets_n2 = det_list(nso, nele - 2)
    index_n = {d: i for i, d in enumerate(dets_n)}
    # the caller's vector follows its own list; map by determinant content
    return dets_n, dets_n1, dets_n2, index_n


def ci_rdms_from_vector(vec, dets_ci, nso, nele):
    """RDMs with the CI determinant list ``dets_ci`` (sorted tuples)."""
    dets_n = det_list(nso, nele)
    dets_n1 = det_list(nso, nele - 1)
    dets_n2 = det_list(nso, nele - 2)
    idx_n = {d: i for i, d in enumerate(dets_n)}
    psi = np.zeros(len(dets_n), dtype=vec.dtype)
    for i, d in enumerate(dets_ci):
        psi[idx_n[d]] = vec[i]
    A1 = _annihilators(dets_n, dets_n1, nso)
    A2 = _annihilators(dets_n1, dets_n2, nso)
    # gamma
    g = np.zeros((nso, nso), dtype=vec.dtype)
    for p in range(nso):
        for q in range(nso):
            g[p, q] = psi @ (A1[p].T @ A1[q]) @ psi
    # Gamma
    X = {}
    for p in range(nso):
        for r in range(nso):
            X[(p, r)] = A2[r] @ (A1[p] @ psi)
    G = np.zeros((nso, nso, nso, nso), dtype=vec.dtype)
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    G[p, q, r, s] = X[(p, r)] @ X[(q, s)]
    return g, G


def spinor_ints(h1e, eri):
    nso = 2 * h1e.shape[0]
    pi = np.arange(nso) // 2
    si = np.arange(nso) % 2
    hso = np.where(si[:, None] == si[None, :], h1e[np.ix_(pi, pi)], 0.0)
    # chemist (pq|rs): spin conservation within each electron pair
    eso = np.where(
        (si[:, None, None, None] == si[None, :, None, None])
        & (si[None, None, :, None] == si[None, None, None, :]),
        eri[np.ix_(pi, pi, pi, pi)], 0.0)
    return hso, eso, pi, si


def generalized_fock(g, G, h1e, eri, spin=False):
    hso, eso, pi, si = spinor_ints(h1e, eri)
    F = np.einsum('pr,qr->pq', g, hso) + np.einsum('prst,qrst->pq', G, eso)
    if spin:
        return F[0::2, 0::2]
    return F


def main():
    out = {}
    mol = build()
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUX)
    h = sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)
    s = mol.intor('int1e_ovlp')
    b = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')
    saux = auxmol.intor('int2c2e_sph', aosym='s1')
    e, u = scipy.linalg.eigh(saux)
    keep = e > 1e-12
    jh = (u[:, keep] / np.sqrt(e[keep])) @ u[:, keep].T
    w = np.matmul(b, jh)
    out['nao'] = int(h.shape[0])
    out['naux'] = int(w.shape[2])
    out['metric_cond'] = float(e[keep].max() / e[keep].min())
    # ---- A. RHF
    rhf = spatial_rhf(h, s, w, mol.nelectron)
    out['e_rhf'] = float(rhf['e_tot'] + mol.energy_nuc())
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    out['e_rhf_pyscf_df'] = float(mf.e_tot)
    out['rhf_diff'] = float(out['e_rhf'] - mf.e_tot)
    c = rhf['mo']
    h1 = c.T @ h @ c
    wm = np.einsum('am,abP,bn->mnP', c, w, c)
    eri = np.einsum('abP,cdP->abcd', wm, wm)
    nm = NCORE + NACT
    h_ca = h1[:nm, :nm]
    e_ca = eri[:nm, :nm, :nm, :nm]
    from diffmrc.core.casci import cs_casci
    e_ci, ci_vec = cs_casci(h_ca, e_ca, ncore=NCORE, nact=NACT,
                            nele_act=NELE, nroots=1, return_ci=True)
    out['e_casci'] = float(e_ci)
    ci_vec = np.asarray(ci_vec).ravel()
    # ---- B. frozen-core FCI on the same active integrals
    h_act = h1[NCORE:nm, NCORE:nm].copy()
    eri_act = eri[NCORE:nm, NCORE:nm, NCORE:nm, NCORE:nm].copy()
    e_core = 2 * np.trace(h1[:NCORE, :NCORE])
    for i in range(NCORE):
        for j in range(NCORE):
            e_core += 2 * eri[i, i, j, j] - eri[i, j, j, i]
    for i in range(NCORE):
        h_act += 2 * eri[NCORE:nm, NCORE:nm, i, i]
        h_act -= eri[NCORE:nm, i, i, NCORE:nm]
    nalpha = (2 * NCORE + NELE) // 2 - NCORE
    e_fci = fci.direct_spin1.kernel(h_act, eri_act, NACT,
                                    (nalpha, nalpha), nroots=3)[0]
    out['e_fci'] = float(e_core + e_fci[0])
    out['e_casci_minus_fci'] = float(e_ci - (e_core + e_fci[0]))
    # ---- C. RDMs
    nso = 2 * nm
    dets_ci = [tuple(range(2 * NCORE)) + tuple(cb)
               for cb in combinations(range(2 * NCORE, nso), NELE)]
    g, G = ci_rdms_from_vector(ci_vec, dets_ci, nso, 2 * NCORE + NELE)
    hso, eso, pi, si = spinor_ints(h_ca, e_ca)
    e_rdm = (np.einsum('pq,pq->', g, hso)
             + 0.5 * np.einsum('pqrs,pqrs->', eso, G))
    out['e_rdm_minus_ci'] = float(e_rdm - e_ci)
    # determinant limit of the RDM machinery
    psi_det = np.zeros(1)
    psi_det[0] = 1.0
    gd, Gd = ci_rdms_from_vector(psi_det, [dets_ci[0]], nso, 2 * NCORE + NELE)
    idx = np.arange(2 * NCORE + NELE)
    gd_ref = np.zeros((nso, nso))
    gd_ref[idx, idx] = 1.0
    Gd_ref = (np.einsum('pq,rs->pqrs', gd_ref, gd_ref)
              - np.einsum('ps,rq->pqrs', gd_ref, gd_ref))
    out['det_gamma_diff'] = float(np.abs(gd - gd_ref).max())
    out['det_gamma2_diff'] = float(np.abs(Gd - Gd_ref).max())
    e2_det_rdm = 0.5 * np.einsum('pqrs,pqrs->', Gd, eso)
    e2_det_direct = 0.5 * sum(eso[p, p, q, q] - eso[p, q, q, p]
                              for p in idx for q in idx)
    out['det_e2_rdm'] = float(e2_det_rdm)
    out['det_e2_direct'] = float(e2_det_direct)
    out['det_e2_diff'] = float(e2_det_rdm - e2_det_direct)
    # gamma against a symmetric finite difference in the spatial h1e
    eps = 1e-6
    g_fd = np.zeros((nm, nm))
    for a in range(nm):
        for b in range(a, nm):
            hp = h_ca.copy()
            hp[a, b] += eps
            if b != a:
                hp[b, a] += eps
            ep = float(cs_casci(hp, e_ca, ncore=NCORE, nact=NACT,
                                nele_act=NELE, nroots=1))
            hm = h_ca.copy()
            hm[a, b] -= eps
            if b != a:
                hm[b, a] -= eps
            em = float(cs_casci(hm, e_ca, ncore=NCORE, nact=NACT,
                                nele_act=NELE, nroots=1))
            g_fd[a, b] = g_fd[b, a] = (ep - em) / (2 * eps)
    g_aa = g[0::2, 0::2].real
    div = np.where(np.eye(nm) > 0, 2.0, 4.0)
    out['gamma_fd_maxdiff'] = float(np.abs(g_fd / div - g_aa).max())
    out['gamma_scale'] = float(np.abs(g_aa).max())
    # ---- D. generalized Fock in the determinant limit
    # spatial generalised Fock from the spin-summed RDMs
    g_sp0 = np.zeros((nm, nm))
    for i in range(NCORE + NELE // 2):
        g_sp0[i, i] = 2.0
    G_pairs0 = np.zeros((nm, nm, nm, nm))
    for p in range(NCORE + NELE // 2):
        for s in range(NCORE + NELE // 2):
            G_pairs0[p, p, s, s] += 4.0
    for p in range(NCORE + NELE // 2):
        for r in range(NCORE + NELE // 2):
            G_pairs0[p, r, r, p] -= 2.0
    F_sp0 = (np.einsum('pr,qr->pq', g_sp0, h_ca)
             + np.einsum('prst,qrst->pq', G_pairs0, e_ca))
    d0 = np.zeros((nm, nm))
    for i in range(NCORE + NELE // 2):
        d0[i, i] = 1.0
    rho = np.einsum('ijP,ji->P', wm[:nm, :nm], d0)
    j = np.einsum('ijP,P->ij', wm[:nm, :nm], rho)
    k = np.zeros_like(d0)
    for p in range(wm.shape[2]):
        wp = wm[:nm, :nm, p]
        k += wp @ d0 @ wp
    F_ref = h_ca + 2 * j - k
    occ = NCORE + NELE // 2
    out['fock_maxdiff'] = float(np.abs(0.5 * F_sp0[:occ, :occ]
                                       - F_ref[:occ, :occ]).max())
    out['fock_scale'] = float(np.abs(F_ref[:occ, :occ]).max())
    # ---- E. analytic theta gradient against FD
    # spatial rotation of one pair (i, j) applied to the reference orbitals
    from diffmrc.core.rotation import block_rotation
    nmo = h1.shape[0]
    pairs = [(i, j) for i in range(NCORE) for j in range(NCORE, nm)]
    pairs += [(i, j) for i in range(NCORE) for j in range(nm, nmo)]
    pairs += [(i, j) for i in range(NCORE, nm) for j in range(nm, nmo)]
    # spin-summed 1-RDM and same-spin-pair 2-RDM on the spatial orbitals
    g_sp = g[0::2, 0::2] + g[1::2, 1::2]
    G_pairs = np.zeros((nm, nm, nm, nm))
    G4 = G.reshape(nm, 2, nm, 2, nm, 2, nm, 2)
    # Gamma[p,q,r,s] with q and s the second indices
    G_pairs = (G4[:, 0, :, 0, :, 0, :, 0] + G4[:, 0, :, 0, :, 1, :, 1]
               + G4[:, 1, :, 1, :, 0, :, 0] + G4[:, 1, :, 1, :, 1, :, 1])
    G_pairs = G_pairs.transpose(0, 1, 2, 3)

    def e_theta(th):
        r = block_rotation(np.asarray(th), h1.shape[0], pairs)
        cc = c @ r
        hm = cc.T @ h @ cc
        ww = np.einsum('am,abP,bn->mnP', cc, w, cc)
        ee = np.einsum('abP,cdP->abcd', ww[:nm, :nm], ww[:nm, :nm])
        return float(cs_casci(hm[:nm, :nm], ee, ncore=NCORE, nact=NACT,
                              nele_act=NELE, nroots=1))

    # analytic gradient at theta = 0 from the spatial generalized Fock
    g_sp_full = np.zeros((nmo, nmo))
    g_sp_full[:nm, :nm] = g_sp
    G_full = np.zeros((nmo, nmo, nmo, nmo))
    G_full[:nm, :nm, :nm, :nm] = G_pairs
    F_sp = (np.einsum('pr,qr->pq', g_sp_full, h1)
            + np.einsum('prst,qrst->pq', G_full, eri))
    g_th = np.array([2.0 * (F_sp[j, i] - F_sp[i, j]) for (i, j) in pairs])
    # FD, with a one-hot perturbation per pair
    fd = np.zeros(len(pairs))
    hstep = 1e-5
    for k in range(len(pairs)):
        th = np.zeros(len(pairs))
        th[k] = hstep
        ep = e_theta(th)
        th[k] = -hstep
        em = e_theta(th)
        fd[k] = (ep - em) / (2 * hstep)
    out['theta_grad_maxdiff'] = float(np.abs(g_th - fd).max())
    out['theta_grad_scale'] = float(np.abs(fd).max())
    out['theta_grad_rel'] = float(np.abs(g_th - fd).max()
                                  / max(np.abs(fd).max(), 1e-30))
    out['n_pairs'] = len(pairs)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
