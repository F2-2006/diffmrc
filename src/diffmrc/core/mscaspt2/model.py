"""Model-space algebra for the multistate CASPT2 zeroth-order Hamiltonians.

PURPOSE
-------
Provides the closed-shell ``S_z = 0`` determinant space, the spin-summed
transition 1-RDMs, the generalized Fock matrix, and the electronic model
Hamiltonian that the MS-CASPT2 / XMS-CASPT2 effective Hamiltonian is built
on.

Determinants are integer bitmaps over spin orbitals with the core spin
orbitals first, the same convention as
``diffmrc.core.nevpt2.impl.cas_dets_for2``.  The module works in the
``S_z = 0`` sector so that every model state has a well-defined
multiplicity and the state-averaged CASSCF orbitals refer to the same
states that enter the multistate treatment.

PUBLIC INTERFACE
----------------
Determinant space:
    ``sz0_dets``, ``dets_to_so``, ``model_hamiltonian``.
Densities and Fock matrices:
    ``transition_1rdm``, ``ci_1rdm``, ``state_fock``, ``embed_density``.

Inputs are numpy arrays; integrals are in chemist notation and in the MO
basis.  Energies are in Eh and exclude the nuclear repulsion term, which
the caller adds.

CONVENTIONS
-----------
* Spin orbital ``2*i`` is alpha and ``2*i + 1`` is beta for spatial
  orbital ``i``; a determinant bitmap has bit ``2*i + sigma`` set when
  that spin orbital is occupied.
* ``transition_1rdm(c_left, c_right, ...)`` returns
  ``G[p, q] = <left|E(p, q)|right>`` with ``E(p, q)`` the spin-summed
  excitation operator; only the first ``norb`` (occupied plus active)
  spatial orbitals are addressed.
* ``state_fock`` builds the full ``nmo x nmo`` generalized Fock matrix
  ``f[p, q] = h[p, q] + sum_rs D[r, s] ((pq|rs) - (pr|qs)/2)`` because
  the zeroth-order Hamiltonian acts on virtual orbitals as well.

VALIDATED SCOPE
---------------
Unit tests cover the determinant count and uniqueness, the odd-electron
rejection, the transposition symmetry ``G^MN = (G^NM)^T``, the trace
identity ``sum_p G^MN[p, p] = N_active <M|N>``, and the reproduction of
the SA-CASSCF eigenvalues by the model Hamiltonian.  The numpy model
Hamiltonian was checked elementwise against the jax
``diffmrc.core.casci._block_H`` kernel it replaces (CH2/STO-3G CAS(2,2),
2 states): the two matrices agree to machine precision.  See
``diffmrc.core.mscaspt2.ms`` for the multistate validation numbers.

BOUNDARIES
----------
* ``S_z = 0`` only, and the active electron count must be even; states of
  other ``M_S`` components are not generated.
* The model Hamiltonian is the electronic Hamiltonian of the active space
  plus frozen core (no nuclear repulsion, no relativistic terms).
* ``transition_1rdm`` is a dense pilot implementation with a Python loop
  over orbital pairs; it is intended for small active spaces.
* The core is assumed closed shell, matching ``sz0_dets``.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np


def sz0_dets(ncore, nact, nele):
    """Determinants with equal alpha and beta counts in the active space.

    Parameters
    ----------
    ncore : int
        Number of doubly occupied core spatial orbitals.
    nact : int
        Number of active spatial orbitals.
    nele : int
        Number of active electrons; must be even.

    Returns
    -------
    list of int
        Determinant bitmaps, core spin orbitals first.
    """
    if nele % 2:
        raise ValueError("sz0_dets requires an even active electron count")
    na = nele // 2
    core = list(range(2 * ncore))
    act_a = [2 * i for i in range(ncore, ncore + nact)]
    act_b = [2 * i + 1 for i in range(ncore, ncore + nact)]
    dets = []
    for ca in combinations(act_a, na):
        for cb in combinations(act_b, na):
            det = 0
            for so in core + list(ca) + list(cb):
                det |= 1 << so
            dets.append(det)
    return dets


def dets_to_so(dets):
    """Bitmap determinants to sorted spin-orbital occupation tuples."""
    out = []
    for det in dets:
        out.append(tuple(i for i in range(det.bit_length())
                         if (det >> i) & 1))
    return out


def _determinant_hamiltonian(h1e_mo, eri_mo, dets, ncore, nact):
    """Electronic Hamiltonian over the determinant bitmaps (numpy).

    Slater-Condon construction on the spin-orbital integrals: diagonal
    occupation terms, one- and two-spin-orbital replacements with the
    parity of the spectator occupations.  This is a numpy transcription of
    the jax kernel ``diffmrc.core.casci._block_H`` (same expressions, same
    index and sign conventions) so that this layer does not require the
    optional pyscfad dependency.  The returned matrix is not symmetrized;
    ``model_hamiltonian`` does that.
    """
    h1e_mo = np.asarray(h1e_mo, dtype=float)
    eri_mo = np.asarray(eri_mo, dtype=float)
    nso = 2 * (ncore + nact)
    so = np.arange(nso)
    pi, si = so // 2, so % 2
    h_so = np.where(si[:, None] == si[None, :],
                    h1e_mo[pi[:, None], pi[None, :]], 0.0)
    m4 = ((si[:, None, None, None] == si[None, None, :, None])
          & (si[None, :, None, None] == si[None, None, None, :]))
    eri_so = np.where(
        m4, eri_mo[pi[:, None, None, None], pi[None, None, :, None],
                   pi[None, :, None, None], pi[None, None, None, :]], 0.0)

    sub = dets_to_so(dets)
    ndet = len(sub)
    occ = np.zeros((ndet, nso), dtype=bool)
    for row, occupied in enumerate(sub):
        occ[row, list(occupied)] = True

    diag1 = occ @ np.diag(h_so)
    e2ij = np.einsum('ijij->ij', eri_so) - np.einsum('ijji->ij', eri_so)
    diag2 = ((occ @ np.triu(e2ij, k=1)) * occ).sum(axis=1)

    xa = occ[:, None, :] & ~occ[None, :, :]
    xb = ~occ[:, None, :] & occ[None, :, :]
    na, nb = xa.sum(axis=2), xb.sum(axis=2)
    pos = np.cumsum(occ, axis=1) - 1

    def first_index(xm):
        return np.argmax(xm, axis=2)

    def second_index(xm, first):
        xwo = xm & ~(so[None, None, :] == first[..., None])
        return np.argmax(xwo, axis=2)

    mask1 = (na == 1) & (nb == 1)
    i1 = first_index(xa)
    k1 = first_index(xb)
    pos_i = np.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_k = np.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = np.where((pos_i + pos_k) % 2 == 0, 1.0, -1.0)
    xijk = np.einsum('ijkj->ijk', eri_so) - np.einsum('ijjk->ijk', eri_so)
    t = np.einsum('aj,ijk->aik', occ, xijk)
    t_pair = t[np.arange(ndet)[:, None], i1, k1]
    occ_i = np.take_along_axis(occ[:, :, None], i1[:, :, None],
                               axis=1)[..., 0]
    x_ii_k = xijk[i1, i1, k1]
    val1 = sign1 * (h_so[i1, k1] + t_pair - occ_i * x_ii_k)
    h = np.where(mask1, val1, 0.0)

    mask2 = (na == 2) & (nb == 2)
    i2 = second_index(xa, i1)
    k2 = second_index(xb, k1)
    pos_i2 = np.take_along_axis(pos[:, None, :], i2[:, :, None],
                                axis=2)[..., 0]
    pos_k2 = np.take_along_axis(pos[None, :, :], k2[:, :, None],
                                axis=2)[..., 0]
    sign2 = np.where((pos_i + pos_i2 + pos_k + pos_k2) % 2 == 0, 1.0, -1.0)
    val2 = sign2 * (eri_so[i1, i2, k1, k2] - eri_so[i1, i2, k2, k1])
    h = h + np.where(mask2, val2, 0.0)
    return h + np.diag(diag1 + diag2)


def model_hamiltonian(h1e_mo, eri_mo, dets, ncore, nact, nele):
    """Electronic Hamiltonian over the determinant list (no nuclear term).

    ``dets`` are bitmaps; the Slater-Condon construction is done on the
    spin-orbital integrals of the ``ncore + nact`` occupied-plus-active
    orbitals.  The result is symmetrized to remove round-off asymmetry.
    """
    h = _determinant_hamiltonian(h1e_mo, eri_mo, dets, ncore, nact)
    return 0.5 * (h + h.T)


def _ann(dd, so):
    """Annihilate spin orbital ``so`` in a bitmap (``None`` if empty)."""
    if not (dd >> so) & 1:
        return None
    return dd ^ (1 << so), (-1.0) ** ((dd & ((1 << so) - 1)).bit_count())


def _cre(dd, so):
    """Create spin orbital ``so`` in a bitmap (``None`` if occupied)."""
    if (dd >> so) & 1:
        return None
    return dd | (1 << so), (-1.0) ** ((dd & ((1 << so) - 1)).bit_count())


def _e1(dd, p, q):
    """Spin-summed E(p, q) on a determinant bitmap (both spin channels)."""
    out = []
    for spin in (0, 1):
        r = _ann(dd, 2 * q + spin)
        if r is None:
            continue
        d1, s1 = r
        r2 = _cre(d1, 2 * p + spin)
        if r2 is None:
            continue
        d2, s2 = r2
        out.append((d2, s1 * s2))
    return out


def transition_1rdm(c_left, c_right, dets, norb):
    """Spin-summed transition 1-RDM ``G[p, q] = <L|E(p, q)|R>``.

    Only the first ``norb`` spatial orbitals are addressed; the
    model-space Fock matrix never needs the virtual block.
    """
    index = {d: i for i, d in enumerate(dets)}
    cr = np.asarray(c_right, dtype=float)
    cl = np.asarray(c_left, dtype=float)
    G = np.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            acc = defaultdict(float)
            for j, det in enumerate(dets):
                c = cr[j]
                if abs(c) < 1e-14:
                    continue
                for tgt, sgn in _e1(det, p, q):
                    acc[tgt] += c * sgn
            total = 0.0
            for tgt, val in acc.items():
                i = index.get(tgt)
                if i is not None:
                    total += cl[i] * val
            G[p, q] = total
    return G


def state_fock(h1e_mo, eri_mo, D):
    """Full generalized Fock matrix.

    ``f[p, q] = h[p, q] + sum_rs D[r, s] ((pq|rs) - (pr|qs)/2)``, the
    same construction as ``diffmrc.core.caspt2._generalized_fock``.  The
    whole matrix is needed because the zeroth-order Hamiltonian acts on
    virtual orbitals as well; ``D`` must be ``nmo x nmo`` with zeros in
    the virtual block.
    """
    h = np.asarray(h1e_mo)
    e = np.asarray(eri_mo)
    D = np.asarray(D)
    if D.shape != h.shape:
        raise ValueError("density must be nmo x nmo")
    return (h
            + np.einsum('rs,pqrs->pq', D, e)
            - 0.5 * np.einsum('rs,prqs->pq', D, e))


def embed_density(D_block, nmo):
    """Place an occupied-plus-active density block into an nmo matrix."""
    D_block = np.asarray(D_block)
    D = np.zeros((nmo, nmo))
    n = D_block.shape[0]
    D[:n, :n] = D_block
    return D


def ci_1rdm(ci, dets, norb):
    """State 1-RDM ``D[p, q] = <0|E(p, q)|0>``."""
    return transition_1rdm(ci, ci, dets, norb)
