"""Fixed-reference scalar chain used by the SOC-gradient prototype.

This is a self-contained version of the chain-test convention used in
this stack: one ROHF reference at the reference geometry, Loewdin
re-orthonormalization at every displaced geometry, a fixed frozen core
and active space, and a deterministic FCI in the active space.  The
scalar Hamiltonian is differentiated analytically; the SOC integrals
(ECPso, the effective-charge nuclear term, and the two-electron SOC
kernel) are differenced at the integral layer because no analytic
derivative kernels are available for them.

The module provides no production gradient.  It supports the prototype
in ``validate_somf_grad.py``: the full-chain finite difference, the
analytic scalar response (dC, dD, dP), and the SOMF-specific terms.
"""
import math

import numpy as np
from pyscf import gto, scf
from pyscf.fci import direct_spin1


def build_mol(atom, basis, ecp, spin, disp=None, unit="Bohr"):
    atoms = [(s, [float(x) for x in c]) for s, c in atom]
    if disp is not None:
        iatm, axis, delta = disp
        atoms[iatm][1][axis] += delta
    return gto.M(atom=[(s, tuple(c)) for s, c in atoms], basis=basis,
                 ecp=ecp, spin=spin, charge=0, unit=unit, verbose=0)


def ao_integrals_ecp(mol):
    s = mol.intor("int1e_ovlp")
    h1 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    if mol.has_ecp():
        h1 = h1 + mol.intor("ECPscalar")
    eri = mol.intor("int2e", aosym="s1")
    hsoc = np.asarray(mol.intor("ECPso"), dtype=float)
    return s, h1, eri, hsoc


def ao_derivatives_ecp(mol, atom, axis):
    """Analytic d/dR of S, h1 (kin+nuc+ECPscalar) and int2e.

    The assembly of the local ECP terms follows the pyscf RHF gradient:
    ``v = -Z_eff * int1e_iprinv + ECPscalar_iprinv + int1e_ipkin +
    int1e_ipnuc + ECPscalar_ipnuc`` with the row block of the moved
    center added on the bra side.
    """
    nao = mol.nao_nr()
    sl = mol.aoslice_by_atom()[atom]
    p0, p1 = int(sl[2]), int(sl[3])
    blk = np.zeros(nao, dtype=bool)
    blk[p0:p1] = True

    ipov = mol.intor("int1e_ipovlp")[axis]
    ds = -(blk[:, None] * ipov + blk[None, :] * ipov.T)

    # ECP operator-derivative terms follow pyscf/grad/rhf.py: the local
    # ECP contribution to the core Hamiltonian (ECPscalar_ipnuc) enters
    # h1 for every center, but the ECP operator derivative with respect
    # to the moved center (ECPscalar_iprinv) exists only when the moved
    # center is itself an ECP center.
    ecp_atoms = set()
    if mol.has_ecp():
        ecp_atoms = set(int(x) for x in mol._ecpbas[:, 0])
    hcore = -(mol.intor("int1e_ipkin")[axis]
              + mol.intor("int1e_ipnuc")[axis])
    with mol.with_rinv_at_nucleus(atom):
        vrinv = mol.intor("int1e_iprinv")[axis].copy()
        if mol.has_ecp():
            ecp_ipnuc = mol.intor("ECPscalar_ipnuc", comp=3)[axis].copy()
            ecp_iprinv = mol.intor("ECPscalar_iprinv", comp=3)[axis].copy()
    if mol.has_ecp():
        hcore = hcore - ecp_ipnuc
    vrinv *= -mol.atom_charge(atom)
    if atom in ecp_atoms:
        vrinv += ecp_iprinv
    vrinv[p0:p1, :] += hcore[p0:p1, :]
    dh1 = vrinv + vrinv.T

    ip1 = mol.intor("int2e_ip1", aosym="s1")[axis]
    deri = -(blk[:, None, None, None] * ip1
             + blk[None, :, None, None] * ip1.transpose(1, 0, 2, 3)
             + blk[None, None, :, None] * ip1.transpose(2, 3, 0, 1)
             + blk[None, None, None, :] * ip1.transpose(2, 3, 1, 0))
    return ds, dh1, deri


def lowdin_basis(cref, s):
    m = cref.T @ s @ cref
    w, u = np.linalg.eigh(m)
    return cref @ (u * w ** -0.5) @ u.T


def lowdin_basis_deriv(cref, s, ds):
    """dC for C(R) = Cref M^{-1/2}, M = Cref^T S Cref."""
    m = cref.T @ s @ cref
    dm = cref.T @ ds @ cref
    w, u = np.linalg.eigh(m)
    f = w ** -0.5
    num = f[:, None] - f[None, :]
    den = w[:, None] - w[None, :]
    deg = np.abs(den) < 1e-10
    fprime = -0.5 * w ** -1.5
    limit = 0.5 * (fprime[:, None] + fprime[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        fdiv = np.where(deg, limit, num / np.where(deg, 1.0, den))
    idx = np.diag_indices(len(w))
    fdiv[idx] = fprime
    return cref @ (u @ (fdiv * (u.T @ dm @ u)) @ u.T)


def mo_integrals(c, h1, eri):
    h1m = c.T @ h1 @ c
    erm = np.einsum("pi,qj,rk,sl,pqrs->ijkl", c, c, c, c, eri,
                    optimize=True)
    return h1m, erm


def eri_transform_deriv(c, dc, eri, deri):
    out = np.einsum("pi,qj,rk,sl,pqrs->ijkl", c, c, c, c, deri,
                    optimize=True)
    for pos in range(4):
        cs = [c, c, c, c]
        cs[pos] = dc
        out += np.einsum("pi,qj,rk,sl,pqrs->ijkl",
                         cs[0], cs[1], cs[2], cs[3], eri, optimize=True)
    return out


class CIModel:
    """Deterministic FCI matrix in a fixed active space (small CAS)."""

    def __init__(self, ncore, nact, nele_act, spin):
        self.ncore = int(ncore)
        self.nact = int(nact)
        na = (nele_act + spin) // 2
        nb = nele_act - na
        self.nelec = (int(na), int(nb))
        self.nstr = (math.comb(self.nact, na), math.comb(self.nact, nb))
        self.ndet = self.nstr[0] * self.nstr[1]

    def effective_h1(self, h1m, erm):
        sl = slice(self.ncore, self.ncore + self.nact)
        h1e = np.array(h1m[sl, sl], copy=True)
        for i in range(self.ncore):
            h1e += 2.0 * erm[sl, sl, i, i] - erm[sl, i, i, sl]
        return h1e

    def effective_h1_deriv(self, dh1m, derm):
        sl = slice(self.ncore, self.ncore + self.nact)
        dh1e = np.array(dh1m[sl, sl], copy=True)
        for i in range(self.ncore):
            dh1e += 2.0 * derm[sl, sl, i, i] - derm[sl, i, i, sl]
        return dh1e

    def active_eri(self, erm):
        sl = slice(self.ncore, self.ncore + self.nact)
        return erm[sl, sl, sl, sl]

    def _h2e(self, h1e, eria):
        return direct_spin1.absorb_h1e(h1e, eria, self.nact, self.nelec,
                                        0.5)

    def apply(self, h1e, eria, vecs):
        h2e = self._h2e(h1e, eria)
        flat = vecs.ndim == 1
        arr = vecs.reshape((-1, self.ndet)) if not flat else vecs[None, :]
        out = np.empty_like(arr)
        for k in range(arr.shape[0]):
            out[k] = direct_spin1.contract_2e(
                h2e, arr[k].reshape(self.nstr), self.nact,
                self.nelec).ravel()
        return out[0] if flat else out

    def matrix(self, h1e, eria):
        return self.apply(h1e, eria, np.eye(self.ndet)).T

    def trans_rdm(self, va, vb):
        return direct_spin1.trans_rdm1(va.reshape(self.nstr),
                                        vb.reshape(self.nstr),
                                        self.nact, self.nelec)


def solve_scalar(mol, cref, ci, phase_ref=None):
    """Solve the scalar fixed-reference CI, keeping the full eigenbasis.

    The two lowest roots are the tracked states; the complete eigenbasis
    is retained because the first-order CI response sums over every
    determinant-space eigenstate.
    """
    s, h1, eri, hsoc = ao_integrals_ecp(mol)
    c = lowdin_basis(cref, s)
    h1m, erm = mo_integrals(c, h1, eri)
    h1e = ci.effective_h1(h1m, erm)
    eria = ci.active_eri(erm)
    hmat = ci.matrix(h1e, eria)
    e_full, v_full = np.linalg.eigh(hmat)
    for k in range(2):
        if phase_ref is not None:
            if v_full[:, k] @ phase_ref[:, k] < 0.0:
                v_full[:, k] *= -1.0
        else:
            imax = int(np.argmax(np.abs(v_full[:, k])))
            if v_full[imax, k] < 0.0:
                v_full[:, k] *= -1.0
    return dict(mol=mol, s=s, h1=h1, eri=eri, hsoc=hsoc, c=c,
                h1m=h1m, erm=erm, h1e=h1e, eria=eria, hmat=hmat,
                e=e_full, v=v_full, cref=cref)


def state_rdms(sol, ci, nstates=2):
    """Transition 1-RDMs among the tracked states."""
    v = sol["v"]
    return [[ci.trans_rdm(v[:, i], v[:, j]) for j in range(nstates)]
            for i in range(nstates)]


def active_density(sol, ci, dmats, weights=None):
    """Full AO density of a weighted sum of active-state 1-RDMs."""
    if weights is None:
        weights = np.ones(len(dmats)) / len(dmats)
    p_act = sum(w * dmats[k][k] for k, w in enumerate(weights))
    c = sol["c"]
    cf = c[:, :ci.ncore]
    ca = c[:, ci.ncore:ci.ncore + ci.nact]
    p = 2.0 * cf @ cf.T + ca @ p_act @ ca.T
    return 0.5 * (p + p.T), p_act


def ci_response(sol, ci, dh1e, deria, ntracked=2):
    """First-order CI coefficients c[k, j] for the tracked roots.

    Uses the full determinant-space eigenbasis of the scalar FCI.
    """
    e, v = sol["e"], sol["v"]
    c = np.zeros((ntracked, ci.ndet))
    for k in range(ntracked):
        hder = ci.apply(dh1e, deria, v[:, k])
        for j in range(ci.ndet):
            if j == k:
                continue
            c[k, j] = (v[:, j] @ hder) / (e[k] - e[j])
    return c


def trans_all(sol, ci, k):
    """Transition RDMs D_{j,k} for every eigenstate j."""
    v = sol["v"]
    return np.array([ci.trans_rdm(v[:, j], v[:, k])
                     for j in range(ci.ndet)])


def density_response(sol, ci, ccoef, k, d_all):
    """dP_k = sum_j c[k,j] (D_{j,k} + D_{k,j})."""
    d = np.zeros((ci.nact, ci.nact))
    for j in range(ci.ndet):
        if ccoef[k, j] == 0.0:
            continue
        d += ccoef[k, j] * (d_all[j] + d_all[j].T)
    return d
