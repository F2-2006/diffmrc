"""Spin-orbit mean-field (SOMF) two-electron operator.

WHY THIS MODULE EXISTS

The SOC layer of this stack assembles the state-interaction (SI) matrix
from the one-electron Breit-Pauli operator
``int1e_pnucxp`` only.  On OH the one-electron operator gives a 2Pi
splitting of 206.52 cm^-1, against 205.88 cm^-1 for ORCA with
``SOCFlags 1,0,0,0`` (0.31%), while the ORCA default SOMF operator gives
128.83 cm^-1.  The missing term is the two-electron part: the
spin-same-orbit and spin-other-orbit interactions screened by the
ground-state density.  This module supplies that operator.

FORMULA (ORCA 6.1 manual, section 5.26; Hess et al., Chem. Phys. Lett.
251, 365 (1996); Neese, J. Chem. Phys. 122, 034107 (2005), Eq. (19))

    h^{K}_{pq} = (p|h^{1el}_K|q)
                 + sum_{rs} P_{rs} [ (pq|g_K|rs)
                                     - 3/2 (pr|g_K|sq)
                                     - 3/2 (sq|g_K|pr) ],

with the one-electron operator

    h^{1el}_K = (alpha^2/2) sum_A Z_A l_{iA,K} r_{iA}^{-3}

and the two-electron kernel

    g_K(r_i, r_j) = -(alpha^2/2) l_{ij,K} r_{ij}^{-3},

where ``l_{ij} = (r_i - r_j) x p_i``.  ``P`` is the spin-summed charge
density matrix of the reference state, ``P_{pq} = <0|E_q^p|0>``, so that
``trace(P S) = N_elec`` in a non-orthogonal AO basis.

INTEGRAL CONVENTION (fixed here; the same convention as ``int1e_pnucxp``)

``libcint`` exposes the two-electron kernel as ``int2e_p1vxp1``.  In this
module the tensor is stored as

    g[k, p, q, r, s] = (pq|g_K|rs),   k = x, y, z,

i.e. chemist notation with both indices of the first electron pair first.
The tensor is symmetric under ``r <-> s`` but not under ``p <-> q``; the
SOMF combination below is antisymmetric in ``p <-> q`` after contraction
with a symmetric ``P`` (verified: max|h + h^T| <= 9e-16).

The contractions implemented are

    J[k, p, q] = sum_{rs} P_{rs} g[k, p, q, r, s]
    K[k, p, q] = sum_{rs} P_{rs} ( g[k, p, r, s, q] + g[k, s, q, p, r] )
    h2         = J - 3/2 K

The index patterns are not guessed: they were fixed by comparing against
pyscf's own SOMF machinery (``pyscf.prop.gtensor.uhf.get_jk``, which
contracts ``int2e_p1vxp1`` through ``_vhf.direct_mapdm``).  Explicit
einsum versus that route agrees to 7.1e-15 on OH/6-31G; see
``experiments/somf/probe_somf_contraction.py``.

PREFACTOR

Neither ``int1e_pnucxp`` nor ``int2e_p1vxp1`` carries ``alpha^2/2`` or
``i``.  The returned array therefore follows the convention of
``int1e_pnucxp``: real and antisymmetric, and the SI assembly multiplies
by ``i * alpha^2/2`` (the reference SI convention of this stack).
``prefactor=True`` returns ``(alpha^2/2) * h`` (units Eh, still real;
the caller then uses ``i * sum_k h_k s_k``).

TWO ROUTES

* ``contract='explicit'``: materializes ``(3, nao, nao, nao, nao)`` and
  uses the three einsums above.  Transparent, used by the unit tests.
* ``contract='direct'``: delegates the two contractions to pyscf's
  density-fitting-free ``direct_mapdm`` path, so no ``nao^4`` array is
  built.  This is the production route for larger basis sets.
* ``contract='auto'`` (default): explicit below ``memory_limit`` bytes
  of kernel storage, direct above.

DENSITY DEPENDENCE AND DERIVATIVES

``h2`` is linear in ``P`` and the operator is not a fixed one-electron
operator: ``P`` is the ground-state density of the state whose SOC matrix
elements are being computed.  For a nuclear derivative of an SOC matrix
element ``T = sum_{pq} D_{pq} h_{pq}[P]`` the chain therefore contains

    dT/dR = sum D (dh1e/dR) + sum D (dh2e/dR)|_P + sum D (dh2e/dP) dP/dR,

where the last term is the density response (and orbital-response terms
enter through D and C).  This module implements the value layer only: it
returns h[P] for a fixed P.  A gradient implementation must add the
response term; the Hellmann-Feynman expression alone is incomplete.

The operator also differs between wavefunction methods (CASSCF vs MRCI
densities), exactly as the ORCA manual states; the caller decides which
``P`` is physical for the target states.

NOT IMPLEMENTED

* RI-SOMF: the kernel is not a function of ``r_ij`` alone, so the usual
  Coulomb-metric RI factorization of ``core.ri`` does not apply directly.
  Four-center integrals are the only route implemented here.
* ``SOCMaxCenter`` / AMFI one-center truncation of the exchange.
* JAX differentiability: the value layer is numpy.  Making ``P`` and the
  integral tensors differentiable inputs is a separate step.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "one_electron_soc",
    "somf_two_electron",
    "somf_matrix",
    "SOMF_ALPHA2_HALF",
]

# alpha^2 / 2 in atomic units (CODATA value used by pyscf.data.nist).
SOMF_ALPHA2_HALF = 0.5 * 7.2973525693e-3 ** 2

# Default budget for the explicit (3, nao, nao, nao, nao) kernel, in bytes.
# 1 GiB holds nao up to about 75; larger systems use the direct route.
DEFAULT_MEMORY_LIMIT = 1 << 30


def one_electron_soc(mol):
    """One-electron Breit-Pauli SOC integrals, shape ``(3, nao, nao)``.

    ``mol.intor_asymmetric`` is the call pyscf itself uses for this
    operator (``pyscf.prop.gtensor.uhf.make_h01_soc1e``).  On this stack
    it equals ``mol.intor('int1e_pnucxp')`` to 3e-15; the asymmetric form
    is used so that no symmetry is imposed on an antisymmetric operator.
    """
    return np.asarray(mol.intor_asymmetric("int1e_pnucxp", 3), dtype=float)


def somf_two_electron(mol, P, contract="auto", memory_limit=None,
                      aosym="s1"):
    """Two-electron SOMF mean-field operator ``(3, nao, nao)`` in the AO basis.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule providing ``int2e_p1vxp1``.
    P : array_like, shape (nao, nao)
        Spin-summed reference charge density matrix in the AO basis,
        ``trace(P S) = N_elec``.  The SOMF contraction is defined for a
        symmetric charge density; this function does not symmetrize the
        input, so a non-symmetric ``P`` produces a non-antisymmetric
        operator (caught by ``check=True`` in :func:`somf_matrix`).
    contract : {'auto', 'explicit', 'direct'}
        Integral contraction route, see the module docstring.
    memory_limit : float, optional
        Byte budget for the explicit kernel in the ``'auto'`` route.
    aosym : str
        ``aosym`` argument forwarded to ``mol.intor`` in the explicit
        route (``'s1'`` keeps all index permutations).

    Returns
    -------
    ndarray, shape (3, nao, nao)
        ``J - 3/2 K`` in the same raw convention as ``int1e_pnucxp``.
        Real; antisymmetric up to roundoff (``max|h + h^T| < 1e-14`` for
        a converged symmetric ``P``).
    """
    nao = mol.nao_nr()
    P = np.asarray(P, dtype=float)
    if P.shape != (nao, nao):
        raise ValueError(f"P must have shape ({nao}, {nao}), got {P.shape}")

    if contract not in ("auto", "explicit", "direct"):
        raise ValueError(f"unknown contract route: {contract!r}")

    if contract == "auto":
        limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
        need = 8.0 * 3 * nao ** 4
        contract = "explicit" if need <= limit else "direct"

    if contract == "explicit":
        return _two_electron_explicit(mol, P, aosym=aosym)
    return _two_electron_direct(mol, P)


def _two_electron_explicit(mol, P, aosym="s1"):
    if aosym != "s1":
        raise ValueError("the explicit route requires aosym='s1'")
    nao = mol.nao_nr()
    g = np.asarray(mol.intor("int2e_p1vxp1", aosym=aosym), dtype=float)
    g = g.reshape(3, nao, nao, nao, nao)
    j = np.einsum("rs,kpqrs->kpq", P, g, optimize=True)
    k1 = np.einsum("rs,kprsq->kpq", P, g, optimize=True)
    k2 = np.einsum("rs,ksqpr->kpq", P, g, optimize=True)
    return j - 1.5 * (k1 + k2)


def _two_electron_direct(mol, P):
    # pyscf's SOMF implementation: the same kernel through direct_mapdm.
    # The map is linear in the density, so the spin-summed P can be split
    # into two equal spin channels without changing J[P] or K[P].
    from pyscf.prop.gtensor.uhf import get_jk

    vj, vk = get_jk(mol, (0.5 * P, 0.5 * P))
    return (vj[0] + vj[1]) - 1.5 * (vk[0] + vk[1])


def somf_matrix(mol, C=None, P=None, mode="somf", contract="auto",
                memory_limit=None, prefactor=False, check=False):
    """SOMF SOC operator in an AO or MO basis.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    C : array_like, shape (nao, n), optional
        Orbital coefficients.  ``None`` returns the operator in the AO
        basis.
    P : array_like, shape (nao, nao), required for ``mode='somf'``
        Spin-summed reference charge density matrix in the AO basis.
    mode : {'somf', 'one_electron'}
        ``'one_electron'`` returns only ``int1e_pnucxp`` (P is ignored).
    prefactor : bool
        If True, multiply by ``alpha^2/2`` (units Eh, still real and
        antisymmetric; the SI assembly then uses ``i * sum_k h_k s_k``).
        If False (default), the raw convention of ``int1e_pnucxp`` is
        kept and the SI assembly multiplies by ``i * alpha^2/2``.
    check : bool
        If True, run structural checks and raise on failure.

    Returns
    -------
    ndarray, shape (3, n, n)
        Real SOC operator components in the basis set by ``C``.

    Notes
    -----
    The two-electron part is linear in ``P`` and uses ``P`` as the
    ground-state density; it is *not* a fixed one-electron operator.
    """
    if mode not in ("somf", "one_electron"):
        raise ValueError(f"unknown mode: {mode!r}")

    nao = mol.nao_nr()
    h = one_electron_soc(mol)

    if mode == "somf":
        if P is None:
            raise ValueError("mode='somf' requires the reference density P")
        P = np.asarray(P, dtype=float)
        if P.shape != (nao, nao):
            raise ValueError(f"P must have shape ({nao}, {nao}), got {P.shape}")
        if check:
            s = mol.intor("int1e_ovlp")
            nelec = float(np.einsum("pq,pq->", P, s))
            if abs(nelec - mol.nelectron) > 1e-6:
                raise ValueError(
                    f"trace(P S) = {nelec:.9f} differs from mol.nelectron "
                    f"= {mol.nelectron}")
        h = h + somf_two_electron(mol, P, contract=contract,
                                  memory_limit=memory_limit)

    if C is not None:
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] != nao:
            raise ValueError(f"C must have shape ({nao}, n), got {C.shape}")
        h = np.einsum("ak,cab,bl->ckl", C, h, C, optimize=True)

    if prefactor:
        h = SOMF_ALPHA2_HALF * h

    if check:
        asym = float(np.abs(h + h.transpose(0, 2, 1)).max())
        if asym > 1e-10:
            raise ValueError(
                f"operator is not antisymmetric: max|h + h^T| = {asym:.3e}")
    return h
