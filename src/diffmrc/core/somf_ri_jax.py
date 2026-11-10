"""Differentiable (JAX) RI-SOMF operator.

WHY THIS MODULE EXISTS

``core.somf_ri`` is a numpy value layer: it builds the RI Coulomb factors
``a``, ``b``, ``jhalf`` and the one-center exchange kernels, then
contracts them with a density.  ``core.somf_jax`` is the differentiable
four-center path; it must materialize the ``(3, nao, nao, nao, nao)``
kernel, which is 24 * nao**4 bytes.  Neither supplies a differentiable
RI-SOMF operator.  This module closes that gap: the RI-SOMF(1X) operator
and the state-interaction splitting become pure-JAX functions of the
reference density ``P``, with an analytic JVP/VJP that never touches a
four-center tensor.

OPERATOR

The value layer defines, with ``g[k,p,q,r,s] = (pq|g_K|rs)``,

    J[k,p,q] = sum_rs P_rs g[k,p,q,r,s]
    K[k,p,q] = sum_rs P_rs ( g[k,p,r,s,q] + g[k,s,q,p,r] )
    h2       = J - 3/2 K

and RI-SOMF(1X) replaces ``J`` by

    w_P = sum_uv b[uv,P] P_uv,          b[uv,P] = (uv|P)
    z   = S^{-1} w = jhalf (jhalf w),   S[P,Q] = (P|Q), jhalf = S^{-1/2}
    J[k,mu,nu] = sum_P a[k,mu,nu,P] z_P,
    a[k,mu,nu,P] = eps_{KLM} (d_L mu d_M nu | P),

while ``K`` is evaluated exactly on each one-center block.  Every object
here is constant at fixed geometry; only ``P`` (and, for the SI
assembly, ``C``) is differentiated.

JVP / VJP

``ri_coulomb_jax`` carries a ``custom_jvp`` whose tangent is the exact
RI derivative:

    dw = b dP + db P
    dz = djhalf (jhalf w) + jhalf (djhalf w) + jhalf jhalf dw
    dJ = a dz + da z

The ``djhalf`` tangent is supplied by the host; ``core.ri`` computes it
with the Daleckii-Krein divided differences of ``x^{-1/2}``
(``inv_sqrt_metric_jvp``), which stay finite at degenerate metric
eigenvalues.  Reverse-mode transposes this same linear map, so the
density response never materializes a ``nao**4`` array: the working set
is ``3 nao**2 naux`` (factors) plus the block kernels.  ``ri_coulomb_vjp``
exposes the transposed map explicitly for callers that assemble a
nuclear gradient by hand.

The one-center exchange has no geometry dependence in the AO basis: all
four basis functions of a block sit on one atom, so translating that atom
translates the whole quartet and the integral is invariant.  Its density
response is the blockwise contraction below, which is linear in ``P``.
A caller that needs ``d/dR`` of the operator must supply ``dP/dR`` and
the ``a``/``b``/``jhalf`` tangents; the latter come from
``core.somf_ri.coulomb_ri_factor_derivs``.

BOUNDARIES

* ``a`` is the three-center SOMF bra factor.  libcint in this stack has
  no derivative of ``int3c2e_pvxp1`` (nor of ``int2e_p1vxp1``); the
  analytic nuclear derivative of ``a`` needs third-derivative integrals
  and is not available.  ``coulomb_ri_factor_derivs`` therefore returns
  ``da`` by central differences of the analytic three-center integral
  and marks that link.  The JVP itself is exact for any ``da``; the gap
  is in the integral, not in this module.
* The one-electron operator ``h1`` is a constant input.  Its nuclear
  derivative is the caller's responsibility (libcint has no
  ``int1e_pnucxp`` derivative either).
* Only ``exchange='one_center'`` is implemented.  A four-center exchange
  would reintroduce the ``nao**4`` kernel this module exists to avoid.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from .somf import SOMF_ALPHA2_HALF
from .somf_jax import (CM_PER_EH, SPIN_MATRICES, si_splitting_jax,
                       transition_tensor_jax)

__all__ = [
    "CM_PER_EH",
    "SPIN_MATRICES",
    "SomfRIIntegrals",
    "somf_ri_integrals",
    "ri_coulomb_jax",
    "ri_coulomb_jvp",
    "ri_coulomb_vjp",
    "one_center_exchange_jax",
    "somf_ri_two_electron_jax",
    "somf_ri_matrix_jax",
    "somf_ri_splitting_jax",
]


class SomfRIIntegrals(NamedTuple):
    """Constant RI-SOMF tensors at one geometry.

    Attributes
    ----------
    h1 : jax.Array, shape (3, nao, nao)
        One-electron Breit-Pauli SOC, raw ``int1e_pnucxp`` convention.
    a : jax.Array, shape (3, nao, nao, naux)
        ``eps_{KLM} (d_L mu d_M nu | P)``.
    b : jax.Array, shape (nao, nao, naux)
        ``(mu nu | P)``.
    jhalf : jax.Array, shape (naux, naux)
        Truncated symmetric ``S^{-1/2}`` of the auxiliary metric.
    indices : tuple of jax.Array
        AO indices of each one-center exchange block.
    kernels : tuple of jax.Array
        ``g_A[k,p,q,r,s]`` of each block, in the block's own AO order.
    """
    h1: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    jhalf: jnp.ndarray
    indices: tuple
    kernels: tuple


def somf_ri_integrals(mol, factors=None, auxmol=None, auxbasis=None,
                      rcond=None, memory_limit=None):
    """Materialize the constant RI-SOMF tensors for a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    factors : CoulombFactors, optional
        Pre-built factor container from
        :func:`diffmrc.core.somf_ri.coulomb_ri_factors`.
    auxmol, auxbasis, rcond
        Forwarded to :func:`coulomb_ri_factors` when ``factors`` is None.
    memory_limit : float, optional
        Byte budget for the per-atom explicit exchange kernels.  Each
        atom costs ``24 n_A**4`` bytes; the check is per atom, not on the
        full ``nao**4`` tensor.

    Returns
    -------
    SomfRIIntegrals
    """
    from .somf import one_electron_soc
    from .somf_ri import coulomb_ri_factors, exchange_one_center_kernels

    if factors is None:
        kwargs = {}
        if rcond is not None:
            kwargs["rcond"] = rcond
        factors = coulomb_ri_factors(mol, auxmol=auxmol,
                                     auxbasis=auxbasis, **kwargs)
    if factors.a.ndim != 4 or factors.a.shape[0] != 3:
        raise ValueError("a must have shape (3, nao, nao, naux)")
    nao = factors.nao
    if factors.a.shape[1:3] != (nao, nao):
        raise ValueError("a and b disagree on nao")
    blocks = exchange_one_center_kernels(mol, memory_limit=memory_limit)
    return SomfRIIntegrals(
        h1=jnp.asarray(one_electron_soc(mol), dtype=jnp.float64),
        a=jnp.asarray(factors.a, dtype=jnp.float64),
        b=jnp.asarray(factors.b, dtype=jnp.float64),
        jhalf=jnp.asarray(factors.jhalf, dtype=jnp.float64),
        indices=tuple(jnp.asarray(idx, dtype=jnp.int32) for idx in blocks[0]),
        kernels=tuple(jnp.asarray(g, dtype=jnp.float64) for g in blocks[1]),
    )


# ---------------------------------------------------------------------------
# RI Coulomb term
# ---------------------------------------------------------------------------
def _ri_coulomb_parts(P, a, b, jhalf):
    """Intermediate quantities shared by the value and the JVP."""
    w = jnp.einsum("uvP,uv->P", b, P)
    jw = jnp.matmul(jhalf, w)
    z = jnp.matmul(jhalf, jw)
    return w, jw, z


@jax.custom_jvp
def ri_coulomb_jax(P, a, b, jhalf):
    """RI Coulomb-like SOMF term ``J``, shape ``(3, nao, nao)``.

    Parameters
    ----------
    P : array_like, shape (nao, nao)
        Spin-summed reference density; may be traced.
    a : array_like, shape (3, nao, nao, naux)
    b : array_like, shape (nao, nao, naux)
    jhalf : array_like, shape (naux, naux)

    The derivative rule is exact for all four arguments; see
    :func:`ri_coulomb_jvp`.
    """
    _, _, z = _ri_coulomb_parts(P, a, b, jhalf)
    return jnp.einsum("kpqP,P->kpq", a, z)


def ri_coulomb_jvp(P, a, b, jhalf, dP, da, db, djhalf):
    """Value and directional derivative of :func:`ri_coulomb_jax`.

    Parameters
    ----------
    P, a, b, jhalf
        Primals, as in :func:`ri_coulomb_jax`.
    dP, da, db, djhalf
        Tangent arrays of the same shapes.  ``djhalf`` is the directional
        derivative of ``S^{-1/2}``; ``core.ri.inv_sqrt_metric_jvp``
        supplies it for a metric displacement.

    Returns
    -------
    (jax.Array, jax.Array)
        ``J`` and ``dJ``, both shape ``(3, nao, nao)``.
    """
    w, jw, z = _ri_coulomb_parts(P, a, b, jhalf)
    dw = (jnp.einsum("uvP,uv->P", b, dP)
          + jnp.einsum("uvP,uv->P", db, P))
    dz = (jnp.matmul(djhalf, jw)
          + jnp.matmul(jhalf, jnp.matmul(djhalf, w))
          + jnp.matmul(jhalf, jnp.matmul(jhalf, dw)))
    value = jnp.einsum("kpqP,P->kpq", a, z)
    tangent = (jnp.einsum("kpqP,P->kpq", a, dz)
               + jnp.einsum("kpqP,P->kpq", da, z))
    return value, tangent


@ri_coulomb_jax.defjvp
def _ri_coulomb_jvp_rule(primals, tangents):
    return ri_coulomb_jvp(*primals, *tangents)


def ri_coulomb_vjp(P, a, b, jhalf, cotangent):
    """Explicit reverse-mode map of :func:`ri_coulomb_jax`.

    Parameters
    ----------
    P, a, b, jhalf
        Primals at which the VJP is evaluated.
    cotangent : array_like, shape (3, nao, nao)
        Cotangent of ``J``.

    Returns
    -------
    tuple of jax.Array
        ``(dP, da, db, djhalf)`` with the shapes of the primals.

    Derivation: with ``z = jhalf jhalf w`` and symmetric ``jhalf``,

        zbar = a^T G,  wbar = jhalf jhalf zbar,
        dP   = b wbar,  db = P wbar,
        da   = G z^T,
        djhalf = zbar (jhalf w)^T + (jhalf zbar) w^T.

    The cost is ``O(nao**2 naux)`` and no ``nao**4`` array is formed.
    A physical ``djhalf`` is symmetric, so only the symmetric part of the
    returned matrix enters a directional derivative; the exact pullback
    is kept unsymmetrized so that ``jax.vjp`` of the plain contraction
    matches element for element.
    """
    G = jnp.asarray(cotangent)
    w, jw, z = _ri_coulomb_parts(P, a, b, jhalf)
    zbar = jnp.einsum("kpq,kpqP->P", G, a)
    wbar = jnp.matmul(jhalf, jnp.matmul(jhalf, zbar))
    dP = jnp.einsum("uvP,P->uv", b, wbar)
    db = jnp.einsum("uv,P->uvP", P, wbar)
    da = jnp.einsum("kpq,P->kpqP", G, z)
    # dz = djhalf (jhalf w) + jhalf (djhalf w) + jhalf jhalf dw, so the
    # exact pullback is the two outer products below.  It is not
    # symmetric; a symmetric tangent djhalf only sees its symmetric part,
    # which is why the custom JVP (not this helper) is the production path.
    djhalf = (jnp.einsum("R,S->RS", zbar, jw)
              + jnp.einsum("R,S->RS", jnp.matmul(jhalf, zbar), w))
    return dP, da, db, djhalf


# ---------------------------------------------------------------------------
# One-center exchange
# ---------------------------------------------------------------------------
def one_center_exchange_jax(P, indices, kernels):
    """One-center SOMF exchange term ``K``, shape ``(3, nao, nao)``.

    Only AO quartets with all four functions on one atom contribute.  The
    block kernels are constant at fixed geometry (a one-center quartet is
    translation invariant), so this term has no clamped-nucleus
    derivative; its density response is the linear map implemented here.

    Parameters
    ----------
    P : array_like, shape (nao, nao)
    indices : sequence of integer arrays
        AO indices of each block.
    kernels : sequence of arrays
        ``g_A[k,p,q,r,s]`` in each block's own AO order.
    """
    P = jnp.asarray(P)
    nao = P.shape[0]
    out = jnp.zeros((3, nao, nao), dtype=P.dtype)
    for idx, g in zip(indices, kernels):
        idx = jnp.asarray(idx)
        if idx.shape[0] == 0:
            continue
        P_A = P[jnp.ix_(idx, idx)]
        k = (jnp.einsum("rs,kprsq->kpq", P_A, g)
             + jnp.einsum("rs,ksqpr->kpq", P_A, g))
        out = out.at[:, idx[:, None], idx].add(k)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def somf_ri_two_electron_jax(P, ints):
    """Differentiable RI-SOMF(1X) two-electron operator, ``(3, nao, nao)``."""
    P = jnp.asarray(P)
    nao = ints.b.shape[0]
    if P.shape != (nao, nao):
        raise ValueError(f"P must have shape ({nao}, {nao}), got {P.shape}")
    j = ri_coulomb_jax(P, ints.a, ints.b, ints.jhalf)
    k = one_center_exchange_jax(P, ints.indices, ints.kernels)
    return j - 1.5 * k


def somf_ri_matrix_jax(P, ints, C=None, prefactor=False):
    """Differentiable RI-SOMF(1X) operator in the AO or an MO basis.

    Parameters
    ----------
    P : array_like, shape (nao, nao)
        Reference density; may be traced.
    ints : SomfRIIntegrals
    C : array_like, shape (nao, n), optional
    prefactor : bool
        Multiply by ``alpha^2/2`` when True; the raw convention is kept
        otherwise.
    """
    P = jnp.asarray(P)
    nao = ints.h1.shape[-1]
    if ints.h1.shape != (3, nao, nao):
        raise ValueError(f"h1 must have shape (3, {nao}, {nao}), got "
                         f"{ints.h1.shape}")
    if P.shape != (nao, nao):
        raise ValueError(f"P must have shape ({nao}, {nao}), got {P.shape}")
    h = ints.h1 + somf_ri_two_electron_jax(P, ints)
    if C is not None:
        C = jnp.asarray(C)
        if C.ndim != 2 or C.shape[0] != nao:
            raise ValueError(f"C must have shape ({nao}, n), got {C.shape}")
        h = jnp.einsum("ak,cab,bl->ckl", C, h, C)
    if prefactor:
        h = SOMF_ALPHA2_HALF * h
    return h


def somf_ri_splitting_jax(P, ints, C, es, D, levels=(0, 2), route="auto"):
    """RI-SOMF(1X) SI splitting as a differentiable function of ``P``.

    Parameters
    ----------
    P : array_like, shape (nao, nao)
        Reference density; traced value.
    ints : SomfRIIntegrals
    C : array_like, shape (nao, nact)
        Active orbital coefficients.
    es : array_like, shape (nstates,)
    D : array_like, shape (nstates, nstates, nact, nact)
        Transition 1-RDMs in the active space.
    levels, route
        Forwarded to :func:`diffmrc.core.somf_jax.si_splitting_jax`.

    Returns
    -------
    jax.Array, scalar
        Splitting in cm^-1.
    """
    h_ao = somf_ri_matrix_jax(P, ints)
    h_mo = jnp.einsum("ak,cab,bl->ckl", jnp.asarray(C), h_ao,
                      jnp.asarray(C))
    T = transition_tensor_jax(jnp.asarray(D), h_mo)
    return si_splitting_jax(jnp.asarray(es), T, levels=levels, route=route)
