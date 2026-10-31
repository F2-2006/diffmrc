"""Differentiable (JAX) spin-orbit mean-field operator.

WHY THIS MODULE EXISTS

``diffmrc.core.somf`` implements the SOMF two-electron operator as a numpy
value layer.  The SOC matrix elements that enter the state-interaction (SI)
assembly depend on the reference density ``P`` through that operator, so a
differentiable SOC chain needs the map ``P -> h2[P]`` to live in JAX.  The
kernel is linear in ``P``;

    h2_k[P] = sum_rs P_rs [ g[k,p,q,r,s] - 3/2 g[k,p,r,s,q]
                            - 3/2 g[k,s,q,p,r] ],

so the implementation is a constant integral tensor contracted with a
traced density.  No custom gradient is required for the operator itself:
reverse-mode JAX gives the density-response contraction

    d/dP_rs sum_pq D_pq h2_k[P]_pq
        = sum_pq D_pq [ g[k,p,q,r,s] - 3/2 g[k,p,r,s,q]
                        - 3/2 g[k,s,q,p,r] ]
        = M_k[r,s],

which is exactly ``somf_ecp.somf_density_response``.  ``tests/core/
test_somf_jax.py`` checks this identity against the analytic contraction.

ROUTE AND MEMORY

Only the explicit route is implemented.  The integral tensor
``g[k,p,q,r,s]`` is constant and materialized once, and the three
contractions above are three ``jnp.einsum`` calls.  The storage is
``24 * nao^4`` bytes (``3 * nao^4`` float64 entries):

    nao =  40  ->  0.06 GB      nao = 100  ->  2.4 GB
    nao =  60  ->  0.31 GB      nao = 128  ->  6.4 GB
    nao =  75  ->  0.76 GB      nao = 150  -> 12.2 GB
    nao =  96  ->  2.0 GB       nao = 175  -> 22.5 GB

Reverse-mode keeps the kernel alive and forms transposed tangents, so the
peak is a small multiple of that budget.  ``somf_integrals`` accepts an
explicit ``memory_limit`` and raises with the estimate before the tensor is
built.

The numpy direct route (``pyscf.prop.gtensor.uhf.get_jk`` through
``direct_mapdm``) never materializes ``g`` and is the production route for
large basis sets, but it runs inside pyscf's C kernels and cannot be traced.
It is not differentiable here: a ``jax.custom_vjp`` around it would still
need a contraction of the four-index kernel in the backward pass, and the
density-response contraction ``M`` is not one of the ``J``/``K`` maps that
``direct_mapdm`` exposes.  A differentiable direct route therefore needs a
dedicated integral-driven contraction and is out of scope for this module.
Use the value layer for large ``nao`` when gradients are not required.

SI ASSEMBLY AND EIGENVALUES

``si_hamiltonian_jax`` reproduces the reference SI convention of this
stack, ``H_{Is,Js'} = E_I delta_IJ delta_ss' + i (alpha^2/2) sum_k T^IJ_k
(s_k)_{ss'}``, with ``T^IJ_k = sum_pq D^IJ_pq h_k[pq]``.  The matrix is
complex Hermitian and its levels are Kramers doublets.

The splitting needs eigenvalues only, so ``route='complex'`` (default)
uses ``jnp.linalg.eigh`` with a custom JVP whose tangent is
``diag(U^H dA U)``.  The eigenvalue derivative carries no
``1/(w_j - w_i)`` factor and is therefore finite at exact degeneracy by
construction; the eigenvector part, which is where the project's
``eigh_safe`` applies its Lorentzian softening, is not formed.

``route='doubled'`` is a cross-check: for ``H = A + iB`` with ``A`` real
symmetric and ``B`` real antisymmetric, the real symmetric matrix
``M = [[A, -B], [B, A]]`` has the spectrum of ``H`` repeated twice, so the
eigenvalues of ``H`` are ``M[0], M[2], M[4], ...``.  The same
eigenvalue-only rule is applied to ``M``.  Both routes agree to roundoff;
the doubled one doubles the matrix size and is kept for verification.

``diffmrc.core.eigh.eigh_safe`` is not used on the SI matrix.  Its JVP is
written for real symmetric input (``u.T``), and under JAX 0.10.2 it
returns a plain tuple while ``jnp.linalg.eigh`` returns an ``EighResult``
namedtuple; grad-of-jit then raises a custom-JVP structure error.  The
frozen core module is not modified here; the eigenvalue-only rule above
covers the SI use case without eigenvectors.

A gauge transform that makes the two-state matrix real directly (for
example ``diag(1, 1, -i, -i)``) is *not* general: the off-diagonal spin
block ``sum_k T_k s_k`` is a general Hermitian 2x2 matrix, and a phase
gauge can remove its imaginary part only when the transition tensor has
support on a single Cartesian component.

BOUNDARIES

* RI-SOMF is not implemented; the kernel is not a function of ``r_ij``
  alone, so the Coulomb-metric factorization of ``core.ri`` does not apply.
* ECP SOMF (``somf_ecp``) remains a numpy value layer.  The valence-level
  ECP operator has its own one-electron substitution and double-counting
  question; no differentiable version is claimed here.
* Nuclear-coordinate gradients are not provided.  This module supplies
  ``d(splitting)/dP``; the orbital and CI response that turn ``dP/dR`` into
  a geometry derivative are the driver's responsibility.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from .somf import SOMF_ALPHA2_HALF, one_electron_soc

__all__ = [
    "CM_PER_EH",
    "SPIN_MATRICES",
    "SomfIntegrals",
    "explicit_kernel_bytes",
    "somf_integrals",
    "somf_two_electron_jax",
    "somf_matrix_jax",
    "eigvalsh_hermitian",
    "transition_tensor_jax",
    "si_hamiltonian_jax",
    "si_hamiltonian_doubled_jax",
    "si_splitting_jax",
    "somf_splitting_jax",
]

# Same conversion as the value-layer acceptance script.
CM_PER_EH = 219474.6313632

# Spin matrices of the reference SI assembly (validate_oh_of.py): one half
# of the Pauli matrices, i.e. s = sigma / 2.  The factor 1/2 is part of the
# convention: omitting it doubles every splitting.
SPIN_MATRICES = 0.5 * jnp.asarray([
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
])


class SomfIntegrals(NamedTuple):
    """Constant integral tensors of the SOMF operator.

    Attributes
    ----------
    h1 : jax.Array, shape (3, nao, nao)
        One-electron Breit-Pauli SOC in the raw ``int1e_pnucxp``
        convention (no ``alpha^2/2``, no ``i``).
    g : jax.Array, shape (3, nao, nao, nao, nao)
        Two-electron SOC kernel ``g[k,p,q,r,s] = (pq|g_K|rs)`` in chemist
        notation, raw convention, constant under differentiation.
    """

    h1: jnp.ndarray
    g: jnp.ndarray


def explicit_kernel_bytes(nao):
    """Storage of the explicit kernel in bytes, ``24 * nao**4``."""
    nao = int(nao)
    return 24 * nao ** 4


def somf_integrals(mol, memory_limit=None):
    """Materialize the constant SOMF integrals for a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule providing ``int1e_pnucxp`` and ``int2e_p1vxp1``.
    memory_limit : float, optional
        Byte budget for the explicit kernel.  When given and exceeded, a
        ``MemoryError`` is raised before ``intor`` is called.

    Returns
    -------
    SomfIntegrals
        ``h1`` and ``g`` as float64 JAX arrays.
    """
    nao = int(mol.nao_nr())
    need = explicit_kernel_bytes(nao)
    if memory_limit is not None and need > memory_limit:
        raise MemoryError(
            f"explicit SOMF kernel needs {need / 2**30:.3f} GiB "
            f"(24 * nao^4 bytes, nao = {nao}) but memory_limit is "
            f"{float(memory_limit) / 2**30:.3f} GiB; use the numpy value "
            f"layer with contract='direct' for this system size")
    h1 = jnp.asarray(one_electron_soc(mol), dtype=jnp.float64)
    g = jnp.asarray(mol.intor("int2e_p1vxp1", aosym="s1"),
                    dtype=jnp.float64).reshape(3, nao, nao, nao, nao)
    return SomfIntegrals(h1=h1, g=g)


def somf_two_electron_jax(P, g):
    """Differentiable two-electron SOMF operator ``J - 3/2 K``.

    Parameters
    ----------
    P : array_like, shape (nao, nao)
        Spin-summed reference density.  May be a traced value; the
        contraction is linear in ``P``.
    g : array_like, shape (3, nao, nao, nao, nao)
        Constant kernel ``g[k,p,q,r,s]`` from :func:`somf_integrals`.

    Returns
    -------
    jax.Array, shape (3, nao, nao)
        ``J - 3/2 K`` in the raw ``int1e_pnucxp`` convention.
    """
    P = jnp.asarray(P)
    g = jnp.asarray(g)
    if g.ndim != 5 or g.shape[0] != 3 or g.shape[1] != g.shape[2] \
            or g.shape[3] != g.shape[4] or g.shape[1] != g.shape[3]:
        raise ValueError(
            f"g must have shape (3, nao, nao, nao, nao), got {g.shape}")
    nao = g.shape[1]
    if P.shape != (nao, nao):
        raise ValueError(f"P must have shape ({nao}, {nao}), got {P.shape}")
    j = jnp.einsum("rs,kpqrs->kpq", P, g)
    k1 = jnp.einsum("rs,kprsq->kpq", P, g)
    k2 = jnp.einsum("rs,ksqpr->kpq", P, g)
    return j - 1.5 * (k1 + k2)


def somf_matrix_jax(P, h1, g, C=None, mode="somf", prefactor=False):
    """Differentiable SOMF operator in an AO or MO basis.

    Parameters
    ----------
    P : array_like, shape (nao, nao), optional
        Reference density; required for ``mode='somf'``.  May be traced.
    h1 : array_like, shape (3, nao, nao)
        One-electron operator from :func:`somf_integrals`.
    g : array_like, shape (3, nao, nao, nao, nao)
        Two-electron kernel from :func:`somf_integrals`.
    C : array_like, shape (nao, n), optional
        Orbital coefficients.  ``None`` returns the AO basis.
    mode : {'somf', 'one_electron'}
        ``'one_electron'`` ignores ``P``.
    prefactor : bool
        Multiply by ``alpha^2/2`` (units Eh) when True; the raw
        ``int1e_pnucxp`` convention is kept otherwise.

    Returns
    -------
    jax.Array, shape (3, n, n)
    """
    if mode not in ("somf", "one_electron"):
        raise ValueError(f"unknown mode: {mode!r}")
    h1 = jnp.asarray(h1)
    nao = h1.shape[-1]
    if h1.shape != (3, nao, nao):
        raise ValueError(f"h1 must have shape (3, {nao}, {nao}), got "
                         f"{h1.shape}")

    h = h1
    if mode == "somf":
        if P is None:
            raise ValueError("mode='somf' requires the reference density P")
        P = jnp.asarray(P)
        if P.shape != (nao, nao):
            raise ValueError(f"P must have shape ({nao}, {nao}), got "
                             f"{P.shape}")
        h = h + somf_two_electron_jax(P, g)

    if C is not None:
        C = jnp.asarray(C)
        if C.ndim != 2 or C.shape[0] != nao:
            raise ValueError(f"C must have shape ({nao}, n), got {C.shape}")
        h = jnp.einsum("ak,cab,bl->ckl", C, h, C)

    if prefactor:
        h = SOMF_ALPHA2_HALF * h
    return h


@jax.custom_jvp
def eigvalsh_hermitian(a):
    """Eigenvalues of a Hermitian matrix with an exact first derivative.

    The tangent of an eigenvalue is the diagonal of the perturbation in
    the eigenbasis, ``dw_i = (U^H dA U)_ii``.  It contains no
    ``1/(w_j - w_i)`` factor, so the derivative is finite at exact
    degeneracy; this is why the complex route does not need the
    eigenvector softening of ``eigh_safe``.
    """
    return jnp.linalg.eigvalsh(a)


@eigvalsh_hermitian.defjvp
def _eigvalsh_hermitian_jvp(primals, tangents):
    (a,), (da,) = primals, tangents
    w, u = jnp.linalg.eigh(a)
    da = 0.5 * (da + jnp.conj(jnp.swapaxes(da, -1, -2)))
    rot = jnp.conj(jnp.swapaxes(u, -1, -2)) @ da @ u
    dw = jnp.real(jnp.diagonal(rot, axis1=-2, axis2=-1))
    return w, dw


def transition_tensor_jax(D, h_mo):
    """SOC transition tensor ``T[..., k] = sum_pq D[..., p, q] h_k[p, q]``.

    Parameters
    ----------
    D : array_like, shape (..., n, n)
        Transition 1-RDM (``(i, j, p, q)`` in the SI assembly) or a single
        density.
    h_mo : array_like, shape (3, n, n)
        SOC operator in the same orbital basis, raw convention.

    Returns
    -------
    jax.Array, shape (..., 3)
    """
    return jnp.einsum("...pq,kpq->...k", D, h_mo)


def si_hamiltonian_jax(es, T):
    """Complex Hermitian SI matrix of the reference convention.

    ``H_{Is,Js'} = E_I delta_IJ delta_ss'
                   + i (alpha^2/2) sum_k T^IJ_k (s_k)_{ss'}``.

    Parameters
    ----------
    es : array_like, shape (n,)
        Scalar state energies in Eh.
    T : array_like, shape (n, n, 3)
        Transition tensor in the raw SOC convention.

    Returns
    -------
    jax.Array, shape (2n, 2n), complex128
    """
    es = jnp.asarray(es)
    T = jnp.asarray(T)
    n = es.shape[0]
    if T.shape != (n, n, 3):
        raise ValueError(f"T must have shape ({n}, {n}, 3), got {T.shape}")
    h = jnp.diag(jnp.repeat(es, 2)).astype(jnp.complex128)
    pref = 1j * SOMF_ALPHA2_HALF
    for k in range(3):
        h = h + pref * jnp.kron(T[:, :, k], SPIN_MATRICES[k])
    return h


def si_hamiltonian_doubled_jax(es, T):
    """Real symmetric embedding of the complex SI matrix.

    With ``H = A + iB`` (``A`` symmetric, ``B`` antisymmetric) the
    matrix ``M = [[A, -B], [B, A]]`` is real symmetric and has the
    spectrum of ``H`` repeated twice.  The eigenvalues of ``H`` are the
    even-indexed eigenvalues of ``M``.

    Parameters
    ----------
    es : array_like, shape (n,)
    T : array_like, shape (n, n, 3)

    Returns
    -------
    jax.Array, shape (4n, 4n), float64
    """
    h = si_hamiltonian_jax(es, T)
    h = 0.5 * (h + jnp.conj(h).T)
    a = jnp.real(h)
    b = jnp.imag(h)
    m = jnp.block([[a, -b], [b, a]])
    return 0.5 * (m + m.T)


def si_splitting_jax(es, T, levels=(0, 2), route="auto"):
    """Splitting of the two lowest SI doublets in cm^-1.

    Parameters
    ----------
    es : array_like, shape (n,)
        Scalar state energies in Eh.
    T : array_like, shape (n, n, 3)
        Transition tensor in the raw SOC convention.
    levels : tuple of int
        ``(lower, upper)`` eigenvalue indices of the ``2n`` SI matrix.
        The default ``(0, 2)`` follows the value-layer acceptance script
        for two states: each Kramers doublet contributes two eigenvalues.
    route : {'auto', 'complex', 'doubled'}
        ``'complex'`` uses :func:`eigvalsh_hermitian` on the Hermitian
        matrix (default); ``'doubled'`` applies the same rule to the real
        symmetric embedding and is a cross-check.

    Returns
    -------
    jax.Array, scalar
        Splitting in cm^-1.
    """
    es = jnp.asarray(es)
    T = jnp.asarray(T)
    if route == "auto":
        route = "complex"
    if route == "complex":
        w = eigvalsh_hermitian(si_hamiltonian_jax(es, T))
        return (w[levels[1]] - w[levels[0]]) * CM_PER_EH
    if route == "doubled":
        w = eigvalsh_hermitian(si_hamiltonian_doubled_jax(es, T))
        return (w[2 * levels[1]] - w[2 * levels[0]]) * CM_PER_EH
    raise ValueError(f"unknown route: {route!r}")


def somf_splitting_jax(P, h1, g, C, es, D, levels=(0, 2), route="auto"):
    """SOMF splitting as a differentiable function of the density ``P``.

    Parameters
    ----------
    P : array_like, shape (nao, nao)
        Reference density; traced value.
    h1, g : array_like
        Constants from :func:`somf_integrals`.
    C : array_like, shape (nao, nact)
        Active orbital coefficients at which the operator is transformed.
    es : array_like, shape (nstates,)
    D : array_like, shape (nstates, nstates, nact, nact)
        Transition 1-RDMs in the active space.
    levels, route
        Forwarded to :func:`si_splitting_jax`.

    Returns
    -------
    jax.Array, scalar
        Splitting in cm^-1.
    """
    h_ao = somf_matrix_jax(P, h1, g)
    h_mo = jnp.einsum("ak,cab,bl->ckl", C, h_ao, C)
    T = transition_tensor_jax(D, h_mo)
    return si_splitting_jax(es, T, levels=levels, route=route)
