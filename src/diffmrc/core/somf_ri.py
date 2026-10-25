"""RI-SOMF: acceleration and one-center truncation of the SOMF operator.

WHY THIS MODULE EXISTS

``core.somf`` implements the four-center SOMF two-electron operator.  Its
kernel is the spin-same-orbit integral

    g_K = (P* i CROSS P j | 1/r12 | k l)
        = eps_{KLM} (d_L phi_i d_M phi_j | 1/r12 | phi_k phi_l),

where ``int2e_p1vxp1`` is libcint's definition (src/autocode/intor2.c;
verified to 3.6e-15 against ``int2e_ipvip1`` in
``experiments/somf_ri/probe_identity3.py``).  Both derivatives sit on the
bra pair, so the integral is *not* a scalar function of ``r_ij`` and the
naive Coulomb-metric factorization with metric ``(P|g_K|Q)`` is invalid:
that ``metric`` is a three-component object, antisymmetric in ``(P,Q)``
and with zero diagonal, hence indefinite rather than positive definite.

The derivative form above, however, is an ordinary Coulomb integral of
the differentiated bra pair against an undifferentiated ket density.  The
Coulomb kernel is positive definite, so the standard Dunlap metric
applies:

    (d_L phi_i d_M phi_j | 1/r12 | phi_k phi_l)
        ~= sum_{PQ} (d_L phi_i d_M phi_j | P) S^{-1}_{PQ} (Q | phi_k phi_l)

with ``S_{PQ} = (P|Q)``.  This is the mathematical content of the
"Coulomb via RI" flag (``SOCFlags`` second entry = 3) in the ORCA 6.1
manual, section 5.26.  RI-SOMF therefore means: fit the differentiated
*bra* pair (the J term) with the Coulomb metric, and treat the exchange
term with the ``1X`` one-center approximation.  The module returns
numpy arrays, mirroring ``core.somf``.

ORCA 6.1 manual, section 5.26 (``spinorbit.html``), keyword table:

    SOMF(1X)    = SOCType 3, SOCFlags 1,2,3,0, SOCMaxCenter 4
    RI-SOMF(1X) = SOCType 3, SOCFlags 1,3,3,0, SOCMaxCenter 4
    AMFI        = SOCType 3, SOCFlags 1,4,3,0, SOCMaxCenter 1

``SOCFlags`` entries: (1) one-electron on/off, (2) Coulomb route
(0 omit, 1 numeric, 2 seminumeric, 3 RI, 4 exact), (3) exchange route
(0 omit, 1 local X-alpha, 2 reversed X-alpha, 3 one-center exact
including spin-other-orbit, 4 full analytic), (4) local DFT correlation.
``SOCMaxCenter`` is the "max. number of centers to include in the
integrals".  Hence the SOMF family keeps the full four-center Coulomb
term and truncates only the exchange; AMFI truncates both.

OPERATOR DEFINITION

The two-electron part follows ``core.somf`` exactly:

    J[k,p,q] = sum_{rs} P_{rs} g[k,p,q,r,s]
    K[k,p,q] = sum_{rs} P_{rs} ( g[k,p,r,s,q] + g[k,s,q,p,r] )
    h2       = J - 3/2 K

with the same raw ``int1e_pnucxp`` prefactor convention (no alpha^2/2,
no i).  ``exchange='one_center'`` keeps only quartets whose four AO
functions sit on one atom; ``exchange='full'`` is the four-center term.

ROUTES

``coulomb='ri'``     three-center factors, memory ``3 nao^2 naux`` for
                     ``int3c2e_pvxp1`` plus ``nao^2 naux`` for
                     ``int3c2e_sph``.
``coulomb='exact'``  four-center ``int2e_p1vxp1``.
``coulomb='one_center'`` one-center four-center blocks (AMFI).

``exchange='full'``  four-center ``int2e_p1vxp1``.
``exchange='one_center'`` per-atom sub-molecule blocks.
``exchange='none'``  J only.

NOT IMPLEMENTED

* JAX differentiability.  The RI factors are exposed
  (``coulomb_ri_factors`` / ``coulomb_ri_from_factors``) so that a
  derivative layer can reuse the same contraction later.
* RI for the exchange term.  The exchange kernel carries the same
  derivative structure, but the reference 1X operator truncates the
  exchange to one center instead; fitting the exchange bra pair keeps
  both the operator index and a density index on the fitted object, a
  more complex contraction than the J term.
* Response of ``P`` (the operator is density dependent) and nuclear
  derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import df, gto

from .somf import DEFAULT_MEMORY_LIMIT, SOMF_ALPHA2_HALF  # noqa: F401
from .ri import (DEFAULT_RCOND, inv_sqrt_metric, make_auxmol,
                 metric_eig)

__all__ = [
    'SOMF_ALPHA2_HALF',
    'DEFAULT_MEMORY_LIMIT',
    'CoulombFactors',
    'orbital_blocks',
    'coulomb_exact',
    'coulomb_ri_factors',
    'coulomb_ri_from_factors',
    'coulomb_ri',
    'coulomb_one_center',
    'exchange_exact',
    'exchange_one_center',
    'one_center_two_electron',
    'somf_ri_two_electron',
    'somf_ri_matrix',
]

#: Levi-Civita symbol, used to contract the two bra-pair derivatives.
_LEVI_CIVITA = np.zeros((3, 3, 3))
_LEVI_CIVITA[0, 1, 2] = _LEVI_CIVITA[1, 2, 0] = _LEVI_CIVITA[2, 0, 1] = 1.0
_LEVI_CIVITA[0, 2, 1] = _LEVI_CIVITA[2, 1, 0] = _LEVI_CIVITA[1, 0, 2] = -1.0

# ORCA 6.1 manual section 5.26 presets: (coulomb, exchange).
_PRESETS = {
    'somf': ('exact', 'full'),
    'somf_1x': ('exact', 'one_center'),
    'ri_somf_1x': ('ri', 'one_center'),
    'ri_somf': ('ri', 'full'),
    'amfi': ('one_center', 'one_center'),
}


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def _cross_contract(t9):
    """``A[K] = eps_{KLM} t[L, M]`` for a 9-component integral tensor.

    libcint orders the two derivative components component-major,
    ``comp = L * 3 + M`` with ``L`` on the first AO and ``M`` on the
    second.  The identity was verified to 3.6e-15 in
    ``experiments/somf_ri/probe_identity3.py``.
    """
    t = np.asarray(t9).reshape(3, 3, *np.shape(t9)[1:])
    return np.einsum('klm,lm...->k...', _LEVI_CIVITA, t, optimize=True)


def _as_density(mol, P):
    nao = mol.nao_nr()
    P = np.asarray(P, dtype=float)
    if P.shape != (nao, nao):
        raise ValueError(f'P must have shape ({nao}, {nao}), got {P.shape}')
    return P


# ---------------------------------------------------------------------------
# Orbital blocks
# ---------------------------------------------------------------------------
def orbital_blocks(mol):
    """AO index arrays, one per atom, in the molecule's AO order."""
    nao = mol.nao_nr()
    loc = mol.ao_loc_nr()
    blocks = []
    for ia in range(mol.natm):
        idx = []
        for ib in range(mol.nbas):
            if int(mol._bas[ib, gto.mole.ATOM_OF]) == ia:
                idx.extend(range(int(loc[ib]), int(loc[ib + 1])))
        blocks.append(np.asarray(idx, dtype=int))
    if sum(len(b) for b in blocks) != nao:
        raise RuntimeError('orbital blocks do not cover all AOs')
    return blocks


def _atom_mol(mol, atom_index):
    """Single-atom molecule in the same basis, unit and representation.

    The basis functions of the block are identical to the full molecule's
    block, so two-electron integrals over that block coincide.  The caller
    verifies this through the overlap block in :func:`_atom_kernel`.
    """
    symbol = mol.atom_symbol(atom_index)
    coord = mol.atom_coord(atom_index, unit='Bohr')
    basis = mol.basis
    if isinstance(basis, dict):
        basis = {symbol: basis[symbol]}
    # The sub-molecule only supplies basis functions; its occupation is
    # irrelevant, but the build must be consistent.  Use the parity of the
    # (effective) nuclear charge as the spin.
    sub_spin = int(round(mol.atom_charge(atom_index))) % 2
    kwargs = dict(atom=[(symbol, coord)], basis=basis, unit='Bohr',
                  spin=sub_spin, verbose=0, cart=bool(mol.cart))
    if getattr(mol, 'ecp', None):
        ecp = mol.ecp
        if isinstance(ecp, dict):
            ecp = {symbol: ecp[symbol]} if symbol in ecp else None
        if ecp:
            kwargs['ecp'] = ecp
    return gto.M(**kwargs)


def _atom_kernel(mol, atom_index, indices, memory_limit, mode):
    """SOMF kernel restricted to one atom, and the atom's density.

    ``mode='explicit'`` materializes ``3 nA^4`` doubles;
    ``mode='direct'`` delegates to pyscf's density-fitting-free map.
    """
    sub = _atom_mol(mol, atom_index)
    if sub.nao_nr() != len(indices):
        raise RuntimeError(
            f'atom {atom_index}: sub-molecule has {sub.nao_nr()} AOs, '
            f'full block has {len(indices)}')
    # The overlap block must match, otherwise the AO ordering or the
    # representation (cartesian/spherical) differs.
    s_full = mol.intor('int1e_ovlp')[np.ix_(indices, indices)]
    s_sub = sub.intor('int1e_ovlp')
    dev = float(np.abs(s_full - s_sub).max())
    if dev > 1e-10:
        raise RuntimeError(
            f'atom {atom_index}: sub-molecule overlap block differs by '
            f'{dev:.3e}')
    n = len(indices)
    need = 3 * 8.0 * n ** 4
    if mode == 'auto':
        mode = 'explicit' if need <= memory_limit else 'direct'
    if mode == 'explicit':
        g = sub.intor('int2e_p1vxp1', aosym='s1').reshape(3, n, n, n, n)
        return g, sub
    return None, sub


def _atom_jk_direct(sub, P_A):
    from pyscf.prop.gtensor.uhf import get_jk
    vj, vk = get_jk(sub, (0.5 * P_A, 0.5 * P_A))
    return vj[0] + vj[1], vk[0] + vk[1]


# ---------------------------------------------------------------------------
# Coulomb term
# ---------------------------------------------------------------------------
def coulomb_exact(mol, P, contract='auto', memory_limit=None):
    """Four-center Coulomb-like SOMF term ``J``, shape ``(3, nao, nao)``."""
    nao = mol.nao_nr()
    P = _as_density(mol, P)
    if contract not in ('auto', 'explicit', 'direct'):
        raise ValueError(f'unknown contract route: {contract!r}')
    if contract == 'auto':
        limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
        contract = 'explicit' if 8.0 * 3 * nao ** 4 <= limit else 'direct'
    if contract == 'explicit':
        g = mol.intor('int2e_p1vxp1', aosym='s1').reshape(3, nao, nao,
                                                          nao, nao)
        return np.einsum('rs,kpqrs->kpq', P, g, optimize=True)
    from pyscf.prop.gtensor.uhf import get_jk
    vj, _ = get_jk(mol, (0.5 * P, 0.5 * P))
    return vj[0] + vj[1]


@dataclass(frozen=True)
class CoulombFactors:
    """Three-center RI factors of the SOMF Coulomb term.

    Attributes
    ----------
    a : ndarray, (3, nao, nao, naux)
        ``eps_{KLM} (d_L phi_mu d_M phi_nu | P)``; the two derivatives sit
        on the AO pair and the centre-derivative signs cancel.
    b : ndarray, (nao, nao, naux)
        ``(mu nu | P)``.
    s : ndarray, (naux, naux)
        Auxiliary metric ``(P|Q)``.
    jhalf : ndarray, (naux, naux)
        Truncated ``S^{-1/2}``.
    auxbasis : str
    cond : float
    n_dropped : int
    """
    a: np.ndarray
    b: np.ndarray
    s: np.ndarray
    jhalf: np.ndarray
    auxbasis: str
    cond: float
    n_dropped: int

    @property
    def nao(self):
        return self.b.shape[0]

    @property
    def naux(self):
        return self.b.shape[2]


def coulomb_ri_factors(mol, auxmol=None, auxbasis=None,
                       rcond=DEFAULT_RCOND):
    """Assemble the three-center factors of the SOMF Coulomb term."""
    if auxmol is None:
        auxmol, label = make_auxmol(mol, auxbasis)
    else:
        label = auxbasis or 'custom auxiliary molecule'
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    # libcint's int3c2e_pvxp1 is exactly the three-component cross
    # contraction of int3c2e_ipvip1; using it avoids the 9-component
    # intermediate (verified to 2.2e-15 in experiments/somf_ri).
    a = df.incore.aux_e2(mol, auxmol, intor='int3c2e_pvxp1', aosym='s1')
    a = np.asarray(a, dtype=float).reshape(3, nao, nao, naux)
    b = np.asarray(df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph',
                                    aosym='s1'), dtype=float)
    b = b.reshape(nao, nao, naux)
    s = auxmol.intor('int2c2e_sph', aosym='s1')
    s = 0.5 * (s + s.T)
    eig = metric_eig(s, rcond)
    jhalf = inv_sqrt_metric(s, rcond, eig=eig)
    return CoulombFactors(a=a, b=b, s=s, jhalf=jhalf, auxbasis=label,
                          cond=eig.cond, n_dropped=eig.n_dropped)


def coulomb_ri_from_factors(factors, P):
    """Contract the three-center factors with a density matrix.

    ``w_P = sum_{uv} (mu nu|P) P_{uv}`` is the standard RI density fit;
    ``J_K[mu,nu] = sum_PQ a[K,mu,nu,P] S^{-1}_{PQ} w_Q``.
    """
    P = np.asarray(P, dtype=float)
    nao = factors.nao
    if P.shape != (nao, nao):
        raise ValueError(f'P must have shape ({nao}, {nao}), got {P.shape}')
    w = np.einsum('uvP,uv->P', factors.b, P, optimize=True)
    z = factors.jhalf @ (factors.jhalf @ w)
    return np.einsum('kpqP,P->kpq', factors.a, z, optimize=True)


def coulomb_ri(mol, P, auxmol=None, auxbasis=None, rcond=DEFAULT_RCOND):
    """RI Coulomb-like SOMF term ``J``, shape ``(3, nao, nao)``."""
    factors = coulomb_ri_factors(mol, auxmol=auxmol, auxbasis=auxbasis,
                                 rcond=rcond)
    return coulomb_ri_from_factors(factors, P)


def coulomb_one_center(mol, P, memory_limit=None, contract='auto'):
    """One-center Coulomb-like SOMF term (the AMFI ``SOCMaxCenter 1`` limit)."""
    P = _as_density(mol, P)
    nao = mol.nao_nr()
    limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
    out = np.zeros((3, nao, nao))
    for ia, idx in enumerate(orbital_blocks(mol)):
        if len(idx) == 0:
            continue
        g, sub = _atom_kernel(mol, ia, idx, limit, contract)
        P_A = P[np.ix_(idx, idx)]
        if g is None:
            j, _ = _atom_jk_direct(sub, P_A)
            out[:, idx[:, None], idx] += j
        else:
            out[:, idx[:, None], idx] += np.einsum(
                'rs,kpqrs->kpq', P_A, g, optimize=True)
    return out


# ---------------------------------------------------------------------------
# Exchange term
# ---------------------------------------------------------------------------
def exchange_exact(mol, P, contract='auto', memory_limit=None):
    """Four-center SOMF exchange term ``K``, shape ``(3, nao, nao)``."""
    nao = mol.nao_nr()
    P = _as_density(mol, P)
    if contract not in ('auto', 'explicit', 'direct'):
        raise ValueError(f'unknown contract route: {contract!r}')
    if contract == 'auto':
        limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
        contract = 'explicit' if 8.0 * 3 * nao ** 4 <= limit else 'direct'
    if contract == 'explicit':
        g = mol.intor('int2e_p1vxp1', aosym='s1').reshape(3, nao, nao,
                                                          nao, nao)
        return (np.einsum('rs,kprsq->kpq', P, g, optimize=True)
                + np.einsum('rs,ksqpr->kpq', P, g, optimize=True))
    from pyscf.prop.gtensor.uhf import get_jk
    _, vk = get_jk(mol, (0.5 * P, 0.5 * P))
    return vk[0] + vk[1]


def exchange_one_center(mol, P, memory_limit=None, contract='auto'):
    """One-center SOMF exchange term, shape ``(3, nao, nao)``.

    Only AO quartets with all four functions on the same atom contribute
    (ORCA ``SOCFlags`` third entry = 3, "one-center exact exchange,
    including spin-other-orbit").  The one-center blocks are evaluated in
    per-atom sub-molecules; the overlap block check in
    :func:`_atom_kernel` guarantees identical basis functions.
    """
    P = _as_density(mol, P)
    nao = mol.nao_nr()
    limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
    out = np.zeros((3, nao, nao))
    for ia, idx in enumerate(orbital_blocks(mol)):
        if len(idx) == 0:
            continue
        g, sub = _atom_kernel(mol, ia, idx, limit, contract)
        P_A = P[np.ix_(idx, idx)]
        if g is None:
            _, k = _atom_jk_direct(sub, P_A)
            out[:, idx[:, None], idx] += k
        else:
            out[:, idx[:, None], idx] += (
                np.einsum('rs,kprsq->kpq', P_A, g, optimize=True)
                + np.einsum('rs,ksqpr->kpq', P_A, g, optimize=True))
    return out


def one_center_two_electron(mol, P, memory_limit=None, contract='auto'):
    """One-center ``J - 3/2 K`` (the AMFI two-electron operator)."""
    P = _as_density(mol, P)
    nao = mol.nao_nr()
    limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
    out = np.zeros((3, nao, nao))
    for ia, idx in enumerate(orbital_blocks(mol)):
        if len(idx) == 0:
            continue
        g, sub = _atom_kernel(mol, ia, idx, limit, contract)
        P_A = P[np.ix_(idx, idx)]
        if g is None:
            j, k = _atom_jk_direct(sub, P_A)
        else:
            j = np.einsum('rs,kpqrs->kpq', P_A, g, optimize=True)
            k = (np.einsum('rs,kprsq->kpq', P_A, g, optimize=True)
                 + np.einsum('rs,ksqpr->kpq', P_A, g, optimize=True))
        out[:, idx[:, None], idx] += j - 1.5 * k
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def somf_ri_two_electron(mol, P, coulomb='ri', exchange='one_center',
                         auxmol=None, auxbasis=None, rcond=DEFAULT_RCOND,
                         memory_limit=None, contract='auto'):
    """Two-electron SOMF operator with selectable Coulomb/exchange routes.

    Parameters
    ----------
    coulomb : {'ri', 'exact', 'one_center', 'none'}
    exchange : {'full', 'one_center', 'none'}
    """
    if coulomb not in ('ri', 'exact', 'one_center', 'none'):
        raise ValueError(f'unknown coulomb route: {coulomb!r}')
    if exchange not in ('full', 'one_center', 'none'):
        raise ValueError(f'unknown exchange route: {exchange!r}')
    P = _as_density(mol, P)
    nao = mol.nao_nr()

    if coulomb == 'ri':
        j = coulomb_ri(mol, P, auxmol=auxmol, auxbasis=auxbasis,
                       rcond=rcond)
    elif coulomb == 'exact':
        j = coulomb_exact(mol, P, contract=contract,
                          memory_limit=memory_limit)
    elif coulomb == 'one_center':
        j = coulomb_one_center(mol, P, memory_limit=memory_limit,
                               contract=contract)
    else:
        j = np.zeros((3, nao, nao))

    if exchange == 'full':
        k = exchange_exact(mol, P, contract=contract,
                           memory_limit=memory_limit)
    elif exchange == 'one_center':
        k = exchange_one_center(mol, P, memory_limit=memory_limit,
                                contract=contract)
    else:
        k = np.zeros((3, nao, nao))

    return j - 1.5 * k


def somf_ri_matrix(mol, C=None, P=None, mode='ri_somf_1x',
                   coulomb=None, exchange=None, auxmol=None, auxbasis=None,
                   rcond=DEFAULT_RCOND, memory_limit=None, contract='auto',
                   prefactor=False, check=False):
    """SOMF SOC operator with the RI / one-center routes of this module.

    ``mode`` selects an ORCA 6.1 section 5.26 preset:

    ==============  ==========  ============
    mode            coulomb     exchange
    ==============  ==========  ============
    ``somf``        exact       full
    ``somf_1x``     exact       one_center
    ``ri_somf_1x``  ri          one_center
    ``ri_somf``     ri          full
    ``amfi``        one_center  one_center
    ==============  ==========  ============

    ``coulomb`` / ``exchange`` override the corresponding preset entry.
    ``mode='one_electron'`` returns only ``int1e_pnucxp``.
    """
    from .somf import one_electron_soc

    if mode == 'one_electron':
        h = one_electron_soc(mol)
    else:
        if mode not in _PRESETS:
            raise ValueError(f'unknown mode: {mode!r}')
        if coulomb is None or exchange is None:
            pc, pe = _PRESETS[mode]
            coulomb = pc if coulomb is None else coulomb
            exchange = pe if exchange is None else exchange
        if P is None:
            raise ValueError(f"mode={mode!r} requires the reference density P")
        P = _as_density(mol, P)
        h = one_electron_soc(mol) + somf_ri_two_electron(
            mol, P, coulomb=coulomb, exchange=exchange, auxmol=auxmol,
            auxbasis=auxbasis, rcond=rcond, memory_limit=memory_limit,
            contract=contract)

    nao = mol.nao_nr()
    if C is not None:
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] != nao:
            raise ValueError(f'C must have shape ({nao}, n), got {C.shape}')
        h = np.einsum('ak,cab,bl->ckl', C, h, C, optimize=True)

    if prefactor:
        h = SOMF_ALPHA2_HALF * h

    if check:
        asym = float(np.abs(h + h.transpose(0, 2, 1)).max())
        if asym > 1e-10:
            raise ValueError(
                f'operator is not antisymmetric: max|h + h^T| = {asym:.3e}')
    return h
