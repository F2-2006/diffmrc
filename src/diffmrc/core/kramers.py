"""Kramers-adapted spinor bases and time-reversal-closed active spaces.

WHY THIS MODULE EXISTS

A physical two-component X2C reference is a complex generalized Hartree-Fock
(GHF) wavefunction.  Its one-particle time-reversal operator is antiunitary,

    T |i> = sum_j U[j, i] |j>,        U^dag U = 1,   U^* U = -1,

and in the j-adapted spinor basis it is the matrix returned by
:func:`diffmrc.core.spinor_ci.kramers_unitary`.  In a Kramers-restricted
reference the HF spinors come in exact pairs (i, T i), so every determinant
space built from complete pairs is invariant under the many-body time-
reversal operator and the complex CI of :mod:`diffmrc.core.spinor_ci` is
closed under it.

A general GHF solution, and in particular the X2C GHF of an odd-electron
system, is *not* Kramers restricted: the spinors are only approximately
paired, and pairing them by the largest element of the overlap of their
time-reversed images gives pairs whose 2x2 time-reversal block differs from
the canonical symplectic form by 3e-2 on OH/6-31G.  Slicing core and active
orbitals from such an ordering breaks the many-body time-reversal invariance
of the CI Hamiltonian (measured ``T H^* T^dag - H`` of order 5 on OH/6-31G
with a frozen core) and leaves a spurious Kramers gap of order 0.05 Eh.

This module supplies the canonical repair: rotate the occupied and virtual
spaces with the Kramers-adapted basis

    e_{2k+1} = m e_{2k}^*,

where ``m = c^dag S U c^*`` is the one-particle time-reversal matrix in the
orthonormal MO basis.  Each seed orbital is orthogonalized against the
columns already taken, its time-reversed image is orthogonalized against the
seed and the same set, and the pair is appended.  The resulting rotation is
unitary in the MO metric and the adapted basis is orthonormal to machine
precision.  The adapted one-particle matrix is the symplectic form
``J = kron(1, [[0, -1], [1, 0]])`` to 1e-15 on both validation systems.

WHAT IS IMPLEMENTED

* :func:`time_reversal_mo_matrix`: ``m = c^dag S U c^*`` for a complex MO
  basis that is orthonormal in the spinor metric ``S``.
* :func:`kramers_adapt`: canonical Kramers pair construction, returning the
  unitary rotation and the pair energies.
* :func:`build_kramers_basis`: the adapted coefficient matrix, its
  time-reversal matrix, and the measured orthonormality and block deviation.
  ``adapt=False`` reproduces the energy-ordered ``argmax`` pairing that the
  earlier probe used; it is a diagnostics path, not a production option.
* :func:`slice_active_space`: core/active slicing that requires even
  boundaries, i.e. complete pairs on both sides of the cut.
* :func:`x2c_spinor_ci`: the physical X2C complex-CI entry point.  It runs
  the density-fitted two-component X2C GHF of :mod:`diffmrc.core.x2c_df`,
  transforms ``mol.intor('int2e_spinor')`` to the adapted MO basis, slices
  the requested core and active pairs, and reports the time-reversal and
  Kramers diagnostics of the resulting determinant Hamiltonian.

CONVENTIONS

The two-electron array is the pyscf complex-spinor array in the ordering of
:mod:`diffmrc.core.spinor_ci`::

    J_ij = sum_kl E[i,j,k,l] D[l,k],
    K_ij = sum_kl E[i,k,j,l] D[k,l].

The MO ERI is the four-index transform with the bra index conjugated::

    E^{MO}_{abcd} = sum_{mnop} c^*_{ma} c_{nb} c^*_{oc} c_{pd} E^{AO}_{mnop}.

The one-particle time-reversal matrix satisfies ``m^dag m = 1`` and
``m^* m = -1`` when ``c^dag S c = 1``; the adapted matrix is
``m' = W^dag m W^*``.

SCOPE AND LIMITS

* Value layer.  The CI Hamiltonian, its spectrum and the symmetry checks are
  implemented; no orbital optimization, response or derivative is added
  here, and the module does not change any existing return value.
* The Kramers adaptation is a unitary rotation of the virtual space as well
  as the occupied space, so orbital energies after adaptation are the
  energies of the seed spinors of each pair; they are not eigenvalues of the
  adapted Fock matrix.
* The ``adapt=False`` path exists to reproduce and diagnose a broken
  construction.  It is not a scientific option: on an odd-electron X2C GHF
  it produces a CI that is not time-reversal invariant.
* The non-relativistic-limit comparison against pyscf ``fci.direct_spin1``
  (see :mod:`diffmrc.core.spinor_ci` tests) validates the determinant
  algebra and the ERI convention.  It is not a relativistic FCI reference:
  no independent two-component FCI exists for these X2C Hamiltonians, and
  the comparison must not be reported as one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spinor_ci, x2c_df

__all__ = [
    'KramersBasis',
    'KramersActiveSpace',
    'X2CSpinorCIResult',
    'argmax_pair_ordering',
    'build_kramers_basis',
    'kramers_adapt',
    'kramers_target',
    'slice_active_space',
    'time_reversal_mo_matrix',
    'x2c_spinor_ci',
]


def _maxabs(a):
    return float(np.abs(a).max())


def kramers_target(nspinor):
    """Canonical symplectic form ``J = kron(1, [[0,-1],[1,0]])``."""
    n = int(nspinor)
    if n % 2:
        raise ValueError("a Kramers basis needs an even spinor count")
    j2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    return np.kron(np.eye(n // 2), j2)


def time_reversal_mo_matrix(mol, mo_coeff):
    """One-particle time reversal in an orthonormal complex MO basis.

    ``m = c^dag S U c^*`` with ``U = kramers_unitary(mol)`` and ``S`` the
    spinor overlap.  The MO basis must satisfy ``c^dag S c = 1``.
    """
    c = np.asarray(mo_coeff, dtype=complex)
    s = np.asarray(mol.intor_symmetric('int1e_ovlp_spinor'))
    u = spinor_ci.kramers_unitary(mol)
    return c.conj().T @ s @ u @ c.conj()


def _orthogonalize(vector, columns):
    """Two-pass Gram-Schmidt projection against orthonormal ``columns``."""
    v = np.array(vector, dtype=complex)
    for _ in range(2):
        for col in columns:
            v -= col * np.vdot(col, v)
    return v


def kramers_adapt(m, mo_energy=None, tol=1e-8):
    """Canonical Kramers pair rotation from a time-reversal matrix.

    Parameters
    ----------
    m : ndarray, (n, n), complex
        ``c^dag S U c^*`` in an orthonormal MO basis.
    mo_energy : ndarray, (n,), optional
        Orbital energies used to order the pairs; index order when omitted.
    tol : float
        Residual norm below which a direction is treated as already spanned.

    Returns
    -------
    rotation : ndarray, (n, n), complex
        Unitary ``W``; the adapted coefficients are ``c @ W``.
    pair_energies : ndarray, (n//2,)
        Seed energy of each pair, in pair order.
    seeds : ndarray, (n//2,)
        MO index of the seed spinor of each pair.

    Notes
    -----
    The construction is ``e_{2k+1} = m e_{2k}^*`` with the seed and its image
    orthogonalized against the columns already taken.  The columns are
    therefore orthonormal by construction and ordered pair by pair; the
    energy ordering only decides which pair is built first.
    """
    m = np.asarray(m, dtype=complex)
    n = m.shape[0]
    if m.shape != (n, n):
        raise ValueError("m must be square")
    if n % 2:
        raise ValueError("a Kramers basis needs an even spinor count")
    energies = (np.arange(n, dtype=float) if mo_energy is None
                else np.asarray(mo_energy, dtype=float))
    if energies.shape != (n,):
        raise ValueError("mo_energy must have one entry per spinor")
    columns = []
    pair_energies = []
    seeds = []
    used = np.zeros(n, dtype=bool)
    for i in np.argsort(energies, kind='stable'):
        if used[i]:
            continue
        seed = np.zeros(n, dtype=complex)
        seed[i] = 1.0
        seed = _orthogonalize(seed, columns)
        nseed = np.linalg.norm(seed)
        if nseed < tol:
            continue
        seed /= nseed
        partner = m @ seed.conj()
        partner = _orthogonalize(partner, columns)
        partner -= seed * np.vdot(seed, partner)
        npartner = np.linalg.norm(partner)
        if npartner < tol:
            continue
        partner /= npartner
        used[i] = True
        used[int(np.argmax(np.abs(seed)))] = True
        used[int(np.argmax(np.abs(partner)))] = True
        pair_energies.append(float(energies[i]))
        seeds.append(int(i))
        columns.extend([seed, partner])
    if len(columns) != n:
        raise RuntimeError(
            "Kramers construction produced %d of %d spinors" %
            (len(columns), n))
    rotation = np.array(columns, dtype=complex).T
    return rotation, np.asarray(pair_energies), np.asarray(seeds)


def argmax_pair_ordering(m, mo_energy=None):
    """Energy-ordered ``argmax`` pairing: the pre-adaptation construction.

    Each unused spinor is paired with the unused spinor whose time-reversal
    matrix element with it is largest.  The returned permutation places the
    pairs contiguously.  This is the construction the earlier probe used; it
    is kept for the negative control and is not a production option.
    """
    m = np.asarray(m, dtype=complex)
    n = m.shape[0]
    if n % 2:
        raise ValueError("a Kramers basis needs an even spinor count")
    energies = (np.arange(n, dtype=float) if mo_energy is None
                else np.asarray(mo_energy, dtype=float))
    scores_all = np.abs(m)
    used = np.zeros(n, dtype=bool)
    order = []
    for i in np.argsort(energies, kind='stable'):
        if used[i]:
            continue
        scores = scores_all[:, i].copy()
        scores[used] = -1.0
        scores[i] = -1.0
        j = int(np.argmax(scores))
        if scores[j] < 0.0:
            raise RuntimeError("no partner left for spinor %d" % i)
        order.extend([int(i), j])
        used[i] = True
        used[j] = True
    return np.asarray(order, dtype=int)


@dataclass(frozen=True)
class KramersBasis:
    """A Kramers-adapted (or deliberately unadapted) spinor basis.

    Attributes
    ----------
    coeff : ndarray, (nao_2c, n2)
        Adapted MO coefficients in the AO spinor basis.
    rotation : ndarray, (n2, n2)
        Unitary ``W`` with ``coeff = mo_coeff @ W``.
    unitary : ndarray, (n2, n2)
        Time-reversal matrix in the adapted basis, ``W^dag m W^*``.
    mo_energy : ndarray, (n2,)
        Orbital energy of each adapted column (the pair's seed energy).
    pair_energies : ndarray, (n2//2,)
        Seed energy of each pair.
    orthonormality : float
        ``max |coeff^dag S coeff - 1|``.
    block_deviation : float
        ``max |unitary - J|``.
    adapted : bool
        True for the canonical construction, False for the argmax pairing.
    """

    coeff: np.ndarray
    rotation: np.ndarray
    unitary: np.ndarray
    mo_energy: np.ndarray
    pair_energies: np.ndarray
    orthonormality: float
    block_deviation: float
    adapted: bool

    @property
    def nspinor(self):
        return self.coeff.shape[1]

    @property
    def n_pairs(self):
        return self.nspinor // 2

    @property
    def closed(self):
        """Whether the basis is Kramers closed to the construction tolerance.

        The tolerance is the block deviation recorded at construction time,
        compared with a fixed 1e-8 threshold; an unadapted basis of an
        odd-electron X2C GHF fails it by orders of magnitude.
        """
        return bool(self.block_deviation < 1e-8)


def build_kramers_basis(mol, mo_coeff, mo_energy, adapt=True, tol=1e-8):
    """Build a Kramers-adapted basis from complex X2C/GHF orbitals.

    ``adapt=False`` orders the orbitals by the energy-ordered ``argmax``
    pairing instead of rotating them; the returned ``block_deviation`` then
    measures how far that construction is from Kramers closure.
    """
    c = np.asarray(mo_coeff, dtype=complex)
    energies = np.asarray(mo_energy, dtype=float)
    n = c.shape[1]
    if n % 2:
        raise ValueError("a Kramers basis needs an even spinor count")
    m = time_reversal_mo_matrix(mol, c)
    if adapt:
        w, pair_energies, _ = kramers_adapt(m, energies, tol=tol)
    else:
        order = argmax_pair_ordering(m, energies)
        w = np.eye(n, dtype=complex)[:, order]
        pair_energies = energies[order][0::2]
    m_adapted = w.conj().T @ m @ w.conj()
    coeff = c @ w
    s = np.asarray(mol.intor_symmetric('int1e_ovlp_spinor'))
    ortho = _maxabs(coeff.conj().T @ s @ coeff - np.eye(n))
    target = kramers_target(n)
    return KramersBasis(
        coeff=coeff, rotation=w, unitary=m_adapted,
        mo_energy=np.repeat(pair_energies, 2), pair_energies=pair_energies,
        orthonormality=ortho, block_deviation=_maxabs(m_adapted - target),
        adapted=bool(adapt))


@dataclass(frozen=True)
class KramersActiveSpace:
    """Core/active slicing with complete Kramers pairs on both sides.

    ``h1`` and ``eri`` are the full adapted-basis arrays; the CI layer
    truncates them internally to ``ncore_so + nact_so``.  ``unitary`` is the
    active block ``m[ncore:ncore+nact, ncore:ncore+nact]``.
    """

    h1: np.ndarray
    eri: np.ndarray
    unitary: np.ndarray
    ncore_so: int
    nact_so: int
    nele_act: int
    label: str = ''

    @property
    def nspinor_total(self):
        return self.h1.shape[0]


def slice_active_space(h1, eri, unitary, ncore_so, nact_so, nele_act,
                       label=''):
    """Slice a T-closed core/active space from a pair-ordered basis.

    ``ncore_so`` and ``nact_so`` must be even: an odd boundary would split a
    Kramers pair and make the determinant space non-invariant under time
    reversal.  The check is structural; whether the underlying basis is
    actually Kramers closed is reported by :class:`KramersBasis`.
    """
    ncore_so = int(ncore_so)
    nact_so = int(nact_so)
    nele_act = int(nele_act)
    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    unitary = np.asarray(unitary)
    nso = h1.shape[0]
    if ncore_so < 0 or nact_so < 0:
        raise ValueError("core and active spinor counts must be >= 0")
    if ncore_so % 2 or nact_so % 2:
        raise ValueError(
            "core and active spinor counts must be even (complete Kramers "
            "pairs); the pair-ordering is produced by kramers_adapt")
    if ncore_so + nact_so > nso:
        raise ValueError("core + active exceeds the spinor basis")
    if nele_act < 0 or nele_act > nact_so:
        raise ValueError("nele_act must lie in [0, nact_so]")
    if unitary.shape != (nso, nso):
        raise ValueError("unitary must have the full basis dimension")
    act = slice(ncore_so, ncore_so + nact_so)
    return KramersActiveSpace(
        h1=h1, eri=eri, unitary=unitary[act, act], ncore_so=ncore_so,
        nact_so=nact_so, nele_act=nele_act, label=label)


@dataclass
class X2CSpinorCIResult:
    """Physical X2C complex CI in a Kramers-adapted, T-closed basis."""

    mol: object
    hf: object
    basis: KramersBasis
    space: KramersActiveSpace
    hamiltonian: np.ndarray
    dets: list
    ecore: float
    energies: np.ndarray
    checks: dict


def x2c_spinor_ci(mol, ncore_pairs, nact_pairs, auxbasis=None, adapt=True,
                  nroots=1, hf=None, tol=1e-8):
    """Complex X2C CI in the Kramers-adapted spinor basis.

    The physical route is: density-fitted two-component X2C GHF, Kramers
    adaptation of the MO space, four-index transform of
    ``mol.intor('int2e_spinor')``, and the determinant CI of
    :func:`diffmrc.core.spinor_ci.spinor_ci_matrix` on ``nact_pairs`` active
    pairs in front of ``ncore_pairs`` core pairs.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule; full-electron basis sets only (the two-component X2C
        helper of pyscf rejects ECP molecules).
    ncore_pairs, nact_pairs : int
        Core and active Kramers pairs, i.e. twice as many spinors.
    auxbasis : str, optional
        Auxiliary basis for the DF SCF; pyscf default when omitted.
    adapt : bool
        True (default) uses the canonical Kramers construction.  False
        reproduces the energy-ordered ``argmax`` pairing and is a
        diagnostics-only negative control.
    nroots : int
        Number of lowest CI eigenvalues returned.
    hf : DFX2CHFResult, optional
        Pre-computed two-component X2C GHF; run here when omitted.
    tol : float
        Residual tolerance of the Kramers construction.

    Returns
    -------
    X2CSpinorCIResult
        ``checks`` carries ``orthonormality``, ``kramers_block_deviation``,
        ``t_closure`` (``max |T H^* T^dag - H|``), ``t_square``
        (``max |T T^* - (-1)^N|``) and, for an odd active electron count,
        ``kramers_degenerate`` and ``max_within_pair_gap``.

    Notes
    -----
    The validated outputs of this entry are the basis and symmetry
    checks.  The frozen-core *energies* inherit the open convention
    defect of :mod:`diffmrc.core.spinor_ci`: its reduction is pinned on
    arrays that satisfy ``E[i,j,i,j] = E[i,j,j,i]``, which the physical
    pyscf spinor array does not.  Do not report these energies as
    frozen-core X2C CI energies until that reduction is re-derived; the
    T-closure and Kramers checks do not depend on it.
    """
    if hf is None:
        hf = x2c_df.df_x2c_hf(mol, auxbasis=auxbasis)
    basis = build_kramers_basis(mol, hf.mo_coeff, hf.mo_energy, adapt=adapt,
                                tol=tol)
    c = basis.coeff
    h1 = c.conj().T @ np.asarray(hf.hcore) @ c
    e_ao = np.asarray(mol.intor('int2e_spinor'))
    eri = np.einsum('ma,nb,oc,pd,mnop->abcd', c.conj(), c, c.conj(), c,
                    e_ao, optimize=True)
    ncore_so = 2 * int(ncore_pairs)
    nact_so = 2 * int(nact_pairs)
    nele_act = int(mol.nelectron) - ncore_so
    space = slice_active_space(h1, eri, basis.unitary, ncore_so, nact_so,
                               nele_act,
                               label='x2c-%s' % ('adapted' if adapt
                                                 else 'unadapted'))
    H, dets, ecore = spinor_ci.spinor_ci_matrix(
        h1, eri, ncore_so, nact_so, nele_act)
    w = np.linalg.eigvalsh(H)
    td = spinor_ci.time_reversal_determinant_matrix(space.unitary, dets)
    parity = (-1.0) ** nele_act
    checks = {
        'orthonormality': basis.orthonormality,
        'kramers_block_deviation': basis.block_deviation,
        't_closure': _maxabs(td @ H.conj() @ td.conj().T - H),
        't_square': _maxabs(td @ td.conj() - parity * np.eye(len(dets))),
        'ncore_so': ncore_so,
        'nact_so': nact_so,
        'nele_act': nele_act,
        'ndet': len(dets),
    }
    if nele_act % 2:
        ok, gaps = spinor_ci.kramers_pairing(w, tol=1e-8)
        checks['kramers_degenerate'] = bool(ok)
        checks['max_within_pair_gap'] = (float(np.max(gaps[0::2]))
                                         if gaps.size else 0.0)
    else:
        checks['kramers_degenerate'] = None
        checks['max_within_pair_gap'] = None
    return X2CSpinorCIResult(
        mol=mol, hf=hf, basis=basis, space=space, hamiltonian=H, dets=dets,
        ecore=float(ecore), energies=w[:int(nroots)], checks=checks)
