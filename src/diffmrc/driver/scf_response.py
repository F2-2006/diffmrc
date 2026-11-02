"""Analytic canonical RHF reference response for the excited-state chain.

The property chain in ``driver.excited`` is evaluated on the canonical
RHF orbitals that the SCF reference returns.  PySCF's analytic CPHF
(``pyscf.hessian.rhf.solve_mo1``) solves for the occupied response in the
gauge where the occupied-occupied block is ``-S1/2``; that gauge keeps
the density derivative exact but does not keep the individual orbitals
canonical, and it says nothing about the virtual columns.  The chain
needs the coefficient derivative of the orbitals the SCF actually
returns, so three pieces are added:

    dC = C K,   K = -S1/2 + U,   U^T = -U

    U_oo[i,j] = e1[i,j] / (eps[j] - eps[i])                (i != j)
    U_vo[a,i] = K_mo1[a,i] + S1[a,i]/2
    U_ov[i,a] = -U_vo[a,i]
    U_vv[a,b] = [f1[a,b] - S1[a,b](eps[a]+eps[b])/2]
                / (eps[b] - eps[a])                       (a != b)

``e1`` is the first-order occupied Fock block returned by ``solve_mo1``;
``f1 = h1 + v1`` is the first-order Fock matrix, with ``v1`` the response
of the Coulomb/exchange potential to the CPHF density derivative.  The
occupied-virtual block follows from ``U`` being antisymmetric, and the
virtual-virtual block from the canonical condition in the virtual
space.  ``-S1/2`` is the symmetric part forced by
``C(R)^T S(R) C(R) = I``.

Only a closed-shell reference is supported.  PySCF 2.13 has no ROHF
CPHF: ``ROHF.Hessian()`` raises ``NotImplementedError`` and the RHF
response kernel constructs a closed-shell density.  An open-shell
reference is refused here instead of returning a wrong response.

The optional ``base_C`` rotates the result into the column-sign gauge of
a reference MO matrix.  pyscf and pyscfad choose independent signs for
the same orbitals, and the property chain is evaluated on the pyscfad
reference; without this alignment a sign flip on one column changes the
response of a different state.
"""
import numpy as np

__all__ = ['canonical_rhf_response', 'density_response', 's1_mo']


def s1_mo(mol, C, dR):
    """First-order overlap in the MO basis along ``dR``.

    ``S1[p,q] = C_p^T (dS/dR) C_q`` for the unit direction ``dR``.  The
    two atom contributions are accumulated from ``int1e_ipovlp`` with the
    same sign convention as ``pyscf.hessian.rhf.solve_mo1``.
    """
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    S1 = np.zeros((C.shape[1], C.shape[1]))
    for ia in range(mol.natm):
        _, _, p0, p1 = aoslices[ia]
        s1ao = np.zeros((3, C.shape[0], C.shape[0]))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        for x in range(3):
            if dR[ia, x] != 0.0:
                S1 += dR[ia, x] * C.T @ s1ao[x] @ C
    return S1


def density_response(dC_occ, mocc):
    """Closed-shell AO density derivative from the occupied response."""
    return 2.0 * (dC_occ @ mocc.T + mocc @ dC_occ.T)


def canonical_rhf_response(mf, dR, base_C=None):
    """Canonical AO coefficient derivative of a closed-shell RHF reference.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        A converged closed-shell reference at the base geometry.
    dR : (natm, 3) array
        Displacement direction in Bohr.  The response is linear in it, so
        no normalization is applied.
    base_C : (nao, nmo) array_like, optional
        Reference MO matrix whose column-sign gauge the result should
        use.  When omitted the PySCF gauge of ``mf.mo_coeff`` is kept.

    Returns
    -------
    dC : (nao, nmo)
        ``dC[:, i]`` is the derivative of the i-th AO coefficient column.
    K : (nmo, nmo)
        ``dC = C K``, in the same column gauge as ``dC``.
    info : dict
        ``K_mo1``, ``S1``, ``e1``, ``Uoo``, ``Uvv`` and the density
        response, for diagnostics and tests.

    Raises
    ------
    NotImplementedError
        For an open-shell occupation.  PySCF 2.13 has no ROHF CPHF.
    """
    from pyscf.hessian import rhf as hrhf

    mol = mf.mol
    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    if np.any((occ != 0.0) & (occ != 2.0)):
        raise NotImplementedError(
            'canonical_rhf_response requires a closed-shell reference; '
            'PySCF 2.13 has no ROHF solve_mo1 and the RHF response kernel '
            'builds a closed-shell density')
    occidx = np.where(occ > 0)[0]
    viridx = np.where(occ == 0)[0]
    nocc = len(occidx)
    eps = np.asarray(mf.mo_energy)
    dR = np.asarray(dR, dtype=float)
    if dR.shape != (mol.natm, 3):
        raise ValueError('dR has shape {0}, expected ({1}, 3)'
                         .format(dR.shape, mol.natm))
    S0 = mol.intor('int1e_ovlp')

    hess = mf.Hessian()
    h1ao = hrhf.make_h1(hess, mf.mo_coeff, mf.mo_occ)
    mo1s, e1s = hrhf.solve_mo1(mf, mf.mo_energy, mf.mo_coeff, mf.mo_occ,
                               h1ao)
    dC_occ = sum(dR[ia, x] * mo1s[ia][x]
                 for ia in range(mol.natm) for x in range(3))
    e1 = sum(dR[ia, x] * e1s[ia][x]
             for ia in range(mol.natm) for x in range(3))
    K_mo1 = C.T @ S0 @ dC_occ

    S1 = s1_mo(mol, C, dR)

    # Occupied-occupied canonical rotation: the CPHF gauge sets the
    # occupied block to -S1/2 and leaves the first-order Fock block e1
    # behind; the rotation that removes its off-diagonal part is e1/gap.
    Uoo = np.zeros((nocc, nocc))
    for i in range(nocc):
        for j in range(nocc):
            if i != j:
                Uoo[i, j] = e1[i, j] / (eps[j] - eps[i])

    # Virtual-virtual canonical rotation.  The virtual block of the
    # first-order Fock matrix is not returned by solve_mo1; rebuild it
    # from the fixed-density derivative h1 and the response v1 to the
    # CPHF density derivative.
    mocc = C[:, occidx]
    dm1 = density_response(dC_occ, mocc)
    vresp = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)
    v1mo = C.T @ np.asarray(vresp(dm1[None])[0]) @ C
    h1mo = sum(dR[ia, x] * C.T @ np.asarray(h1ao[ia][x]) @ C
               for ia in range(mol.natm) for x in range(3))
    f1 = h1mo + v1mo
    Uvv = np.zeros((len(viridx), len(viridx)))
    for a in range(len(viridx)):
        for b in range(len(viridx)):
            if a != b:
                ia, ib = viridx[a], viridx[b]
                num = f1[ia, ib] - 0.5 * S1[ia, ib] * (eps[ia] + eps[ib])
                Uvv[a, b] = num / (eps[ib] - eps[ia])

    K = -0.5 * S1.copy()
    K[np.ix_(occidx, occidx)] += Uoo
    Uvo = K_mo1[np.ix_(viridx, occidx)] + 0.5 * S1[np.ix_(viridx, occidx)]
    K[np.ix_(viridx, occidx)] += Uvo
    K[np.ix_(occidx, viridx)] += -Uvo.T
    K[np.ix_(viridx, viridx)] += Uvv

    dC = C @ K
    if base_C is not None:
        base_C = np.asarray(base_C)
        s0 = np.sign(np.einsum('ai,ab,bi->i', C, S0, base_C))
        dC = dC * s0[None, :]
        K = K * s0[:, None] * s0[None, :]

    return dC, K, {'K_mo1': K_mo1, 'S1': S1, 'e1': e1, 'Uoo': Uoo,
                   'Uvv': Uvv, 'dm1': dm1}
