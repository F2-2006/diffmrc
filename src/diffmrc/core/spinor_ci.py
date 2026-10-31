"""Complex two-component configuration interaction over spinor orbitals.

WHY THIS MODULE EXISTS

The differentiable multireference stack is built on real, spin-adapted
spatial orbitals.  A two-component X2C reference has complex spinors with
exact Kramers degeneracy, and both properties break assumptions of the
real-orbital CI layer: the Hamiltonian is complex Hermitian, the Kramers
partner is not a separate spin block, and every many-electron level is at
least doubly degenerate.  This module supplies a minimal complex CI that
works directly in a spinor basis, together with the Kramers-safe
eigenvalue derivative that the real-orbital layer does not need.

WHAT IS IMPLEMENTED

* ``spinor_ci_matrix`` / ``spinor_ci_solve``: determinant-basis CI in an
  arbitrary orthonormal complex spinor basis, with a frozen core folded
  into a core Fock operator and a core energy.
* ``spinor_ci_matrix_jax`` / ``spinor_ci_energies_jax``: the same matrix
  as a differentiable function of the one- and two-electron integrals.
* ``eigvalsh_kramers_safe``: an eigenvalue-only custom JVP whose tangent
  is ``diag(U^dag dH U)``.  It contains no inverse eigenvalue gap, so it
  stays finite at an exact Kramers degeneracy where a plain reverse-mode
  ``eigh`` returns ``0 * inf = NaN``.
* ``kramers_unitary``: the antiunitary one-particle time-reversal matrix
  in the j-adapted spinor basis, built from ``sph2spinor_coeff``.  It
  satisfies ``U^dag U = 1`` and ``U^* U = -1``.
* ``kramers_pairing``: a spectrum check that every level is at least
  doubly degenerate.

CONVENTIONS

The two-electron array follows ``mol.intor('int2e_spinor')`` and the
density-fitting factor convention of :mod:`diffmrc.core.x2c_df`:

    J_ij = sum_kl E[i,j,k,l] D[l,k],
    K_ij = sum_kl E[i,k,j,l] D[k,l].

The second-quantized Hamiltonian whose expectation reproduces the pyscf
GHF energy with this array is

    H = sum_pq h_pq a^dag_p a_q
        + 1/2 sum_pqrs E[p,q,r,s] a^dag_p a^dag_r a_s a_q.

That ordering was pinned numerically: for every determinant tested, the
brute-force action of this operator reproduces
``1/2 tr(D J) - 1/2 tr(D K)`` of pyscf to machine precision (H2O/6-31G
spinor basis, same-spin and mixed determinants).

With a frozen core the active-space Hamiltonian is

    H_eff = E_core
            + sum_pq f_pq a^dag_p a_q
            + 1/2 sum_pqrs E[p,q,r,s] a^dag_p a^dag_r a_s a_q,

    E_core = sum_i h_ii + 1/2 sum_ij (E[i,i,j,j] - E[i,j,j,i]),
    f_pq   = h_pq + sum_i (E[p,q,i,i] - E[p,i,i,q]),

where i, j run over the core spinors and p, q, r, s over the active
spinors.  These are the only formulas in the module that are not a direct
transcription of the operator; they are pinned in the tests against a
full-space construction that keeps the core in the determinant list.

PYSF SPINOR ERI CONVENTION (MEASURED BOUNDARY)

``mol.intor('int2e_spinor')`` is reproduced to 6.7e-16 by the explicit
``sph2spinor_coeff`` contraction, but the resulting array does not obey
the chemist pair symmetry ``E[p,q,r,s] = conj(E[q,p,r,s])`` (measured
maximum violation 0.618 on H2O/6-31G).  The ordering used here is the
one that reproduces the pyscf GHF determinant energy with that array.
With it the complex CI reproduces pyscf FCI in the non-relativistic
spin-orbital limit and the exact Kramers theorem on a synthetic
time-reversal-invariant Hamiltonian.  On the physical X2C HF MO basis,
however, the many-body time reversal does not commute with the CI
Hamiltonian built from the pyscf spinor array (measured
``T H^* T^T - H = 143`` on OH/6-31G with a time-reversal-closed active
space).  A production two-component CI therefore needs a
convention-aware ERI transform, or an explicitly orthonormal spinor
basis with a validated four-index transform.  That interface is the
remaining gap; the determinant algebra itself is validated.

SCOPE AND LIMITS

* The CI is a value layer plus an eigenvalue-only JAX path.  It is not a
  full Kramers-restricted CASSCF: there is no complex unitary orbital
  rotation, no orbital Z-vector, and no state-specific response.
* The ground-state energy derivative with respect to a time-reversal-even
  one-electron perturbation is well defined at a Kramers degeneracy.  A
  derivative of an individual eigenvector is not: the pair is exactly
  degenerate, so any eigenvector basis inside the pair is arbitrary.
* ECPs are not supported by the two-component X2C transform in pyscf
  (``SpinorX2CHelper.get_hcore`` raises ``NotImplementedError`` for an
  ECP molecule); this module inherits that boundary.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "core_fock",
    "eigvalsh_kramers_safe",
    "kramers_pairing",
    "kramers_unitary",
    "spinor_ci_energies_jax",
    "spinor_ci_matrix",
    "spinor_ci_matrix_jax",
    "spinor_ci_rdm1",
    "spinor_ci_solve",
    "spinor_determinants",
    "time_reversal_determinant_matrix",
]


# ---------------------------------------------------------------------------
# Determinants and fermionic operators
# ---------------------------------------------------------------------------
def spinor_determinants(nspinor, nelec):
    """Sorted occupation tuples of ``nelec`` electrons in ``nspinor`` spinors."""
    return [tuple(c) for c in itertools.combinations(range(nspinor), nelec)]


def _popcount_below(key, x):
    """Number of occupied spinors with index below ``x``."""
    return int(key & ((1 << x) - 1)).bit_count()


def _annihilate(key, x):
    """Apply ``a_x``; return ``(new_key, sign)`` or ``None``."""
    if not (key >> x) & 1:
        return None
    sign = -1.0 if _popcount_below(key, x) % 2 else 1.0
    return key ^ (1 << x), sign


def _create(key, x):
    """Apply ``a^dag_x``; return ``(new_key, sign)`` or ``None``."""
    if (key >> x) & 1:
        return None
    sign = -1.0 if _popcount_below(key, x) % 2 else 1.0
    return key ^ (1 << x), sign


# ---------------------------------------------------------------------------
# Core Fock
# ---------------------------------------------------------------------------
def core_fock(h1, eri, ncore_so):
    """Fold the first ``ncore_so`` spinors into a core energy and Fock matrix.

    Parameters
    ----------
    h1 : ndarray, (nso, nso), complex
        One-electron Hamiltonian in the spinor basis.
    eri : ndarray, (nso, nso, nso, nso), complex
        Two-electron integrals in the pyscf spinor convention.
    ncore_so : int
        Number of fully occupied core spinors (counted from index 0).

    Returns
    -------
    f : ndarray, (nact, nact), complex
        Active-space Fock matrix.
    ecore : float
        Core energy (real part of the core determinant energy).
    """
    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    nso = h1.shape[0]
    ncore_so = int(ncore_so)
    if not 0 <= ncore_so <= nso:
        raise ValueError("ncore_so must lie between 0 and nso")
    act = slice(ncore_so, nso)
    h_aa = h1[act, act]
    eri_aaaa = eri[act, act, act, act]
    f = h_aa.copy()
    ecore = 0.0 + 0.0j
    for i in range(ncore_so):
        # The exchange contraction has two orderings in the two-electron
        # operator; for a complex ERI they differ and the one-body
        # operator that reproduces the single-excitation matrix element
        # carries their average.
        f += (eri[act, act, i, i]
              - 0.5 * (eri[act, i, i, act] + eri[act, i, act, i]))
        ecore += h1[i, i]
        for j in range(ncore_so):
            ecore += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])
    return f, float(np.real(ecore))


# ---------------------------------------------------------------------------
# Determinant Hamiltonian
# ---------------------------------------------------------------------------
def _build_matrix(h_one, eri_act, occ_lists, keys, index, nso_act):
    """Determinant matrix of the active Hamiltonian by operator action."""
    ndet = len(keys)
    H = np.zeros((ndet, ndet), dtype=complex)

    # One-body: h_pq a^dag_p a_q.
    for jcol, key in enumerate(keys):
        for q in occ_lists[jcol]:
            step = _annihilate(key, q)
            if step is None:
                continue
            key1, s1 = step
            for p in range(nso_act):
                step = _create(key1, p)
                if step is None:
                    continue
                key2, s2 = step
                irow = index.get(key2)
                if irow is not None:
                    H[irow, jcol] += h_one[p, q] * s1 * s2

    # Two-body: 1/2 E[p,q,r,s] a^dag_p a^dag_r a_s a_q.
    for jcol, key in enumerate(keys):
        occ = occ_lists[jcol]
        for q in occ:
            step = _annihilate(key, q)
            if step is None:
                continue
            key1, s1 = step
            for s in occ:
                if s == q:
                    continue
                step = _annihilate(key1, s)
                if step is None:
                    continue
                key2, s2 = step
                for r in range(nso_act):
                    step = _create(key2, r)
                    if step is None:
                        continue
                    key3, s3 = step
                    for p in range(nso_act):
                        if p == r:
                            continue
                        step = _create(key3, p)
                        if step is None:
                            continue
                        key4, s4 = step
                        irow = index.get(key4)
                        if irow is not None:
                            H[irow, jcol] += (0.5 * eri_act[p, q, r, s]
                                              * s1 * s2 * s3 * s4)
    return 0.5 * (H + H.conj().T)


def spinor_ci_matrix(h1, eri, ncore_so, nact, nele_act, full=False):
    """CI Hamiltonian in the determinant basis of the active spinors.

    Parameters
    ----------
    h1 : ndarray, (nso, nso), complex
        One-electron Hamiltonian; ``nso = ncore_so + nact`` when
        ``full=False``.
    eri : ndarray, (nso, nso, nso, nso), complex
        Two-electron integrals in the pyscf spinor convention.
    ncore_so : int
        Number of fully occupied core spinors.
    nact : int
        Number of active spinors.
    nele_act : int
        Number of active electrons.
    full : bool
        If True, build the matrix in the full determinant space with the
        core kept in every determinant.  This is the reference
        construction used to validate the core-Fock reduction; it is not
        the production route.

    Returns
    -------
    H : ndarray, (ndet, ndet), complex Hermitian
    dets : list of tuple of int
        Occupied spinor indices of each determinant.
    ecore : float
        Core energy added to the diagonal (zero when ``full=True``).
    """
    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    nso = h1.shape[0]
    nso_req = ncore_so + nact
    if nso > nso_req:
        h1 = h1[:nso_req, :nso_req]
        eri = eri[:nso_req, :nso_req, :nso_req, :nso_req]
        nso = nso_req
    elif nso < nso_req:
        raise ValueError("the input basis is smaller than ncore_so + nact")
    if full:
        dets = [tuple(range(ncore_so)) + c
                for c in itertools.combinations(range(ncore_so, nso),
                                                nele_act)]
        keys = [sum(1 << x for x in d) for d in dets]
        index = {k: i for i, k in enumerate(keys)}
        H = _build_matrix(h1, eri, dets, keys, index, nso)
        return H, dets, 0.0
    f, ecore = core_fock(h1, eri, ncore_so)
    eri_act = eri[ncore_so:, ncore_so:, ncore_so:, ncore_so:]
    dets = spinor_determinants(nact, nele_act)
    keys = [sum(1 << x for x in d) for d in dets]
    index = {k: i for i, k in enumerate(keys)}
    H = _build_matrix(f, eri_act, dets, keys, index, nact)
    H = H + ecore * np.eye(len(dets), dtype=complex)
    return H, dets, ecore


def spinor_ci_solve(h1, eri, ncore_so, nact, nele_act, nroots=1):
    """Lowest ``nroots`` complex CI eigenpairs.

    Returns ``(eigenvalues, eigenvectors, dets, ecore)``; the eigenvectors
    are columns and the energies include the core energy.
    """
    H, dets, ecore = spinor_ci_matrix(h1, eri, ncore_so, nact, nele_act)
    w, v = np.linalg.eigh(H)
    return w[:nroots], v[:, :nroots], dets, ecore


def spinor_ci_rdm1(ci_vec, dets, nspinor):
    """One-particle density matrix ``gamma[p,q] = <a^dag_p a_q>``.

    ``dets`` is the determinant list and ``nspinor`` the number of spinors
    in the determinant space.  The returned matrix is Hermitian for a
    normalized eigenvector.
    """
    vec = np.asarray(ci_vec, dtype=complex)
    keys = [sum(1 << x for x in d) for d in dets]
    index = {k: i for i, k in enumerate(keys)}
    gamma = np.zeros((nspinor, nspinor), dtype=complex)
    for jcol, occ in enumerate(dets):
        if vec[jcol] == 0:
            continue
        for q in occ:
            step = _annihilate(keys[jcol], q)
            if step is None:
                continue
            key1, s1 = step
            for p in range(nspinor):
                step = _create(key1, p)
                if step is None:
                    continue
                key2, s2 = step
                irow = index.get(key2)
                if irow is not None:
                    gamma[p, q] += np.conj(vec[irow]) * vec[jcol] * s1 * s2
    return 0.5 * (gamma + gamma.conj().T)


# ---------------------------------------------------------------------------
# JAX path
# ---------------------------------------------------------------------------
def _ci_structure(ncore_so, nact, nele_act):
    """Sparse constant structure of the active CI matrix.

    The matrix is linear in the active Fock matrix and in the active ERI,
    so the determinant-level signs and row/column indices are constants.
    Returns a dict of index/sign arrays for the one- and two-body parts.
    """
    dets = spinor_determinants(nact, nele_act)
    keys = [sum(1 << x for x in d) for d in dets]
    index = {k: i for i, k in enumerate(keys)}
    one_rows, one_cols, one_signs, one_p, one_q = [], [], [], [], []
    two_rows, two_cols, two_signs = [], [], []
    two_p, two_q, two_r, two_s = [], [], [], []
    for jcol, key in enumerate(keys):
        occ = dets[jcol]
        for q in occ:
            step = _annihilate(key, q)
            if step is None:
                continue
            key1, s1 = step
            for p in range(nact):
                step = _create(key1, p)
                if step is None:
                    continue
                key2, s2 = step
                irow = index.get(key2)
                if irow is not None:
                    one_rows.append(irow); one_cols.append(jcol)
                    one_signs.append(s1 * s2); one_p.append(p); one_q.append(q)
        for q in occ:
            step = _annihilate(key, q)
            if step is None:
                continue
            key1, s1 = step
            for s in occ:
                if s == q:
                    continue
                step = _annihilate(key1, s)
                if step is None:
                    continue
                key2, s2 = step
                for r in range(nact):
                    step = _create(key2, r)
                    if step is None:
                        continue
                    key3, s3 = step
                    for p in range(nact):
                        if p == r:
                            continue
                        step = _create(key3, p)
                        if step is None:
                            continue
                        key4, s4 = step
                        irow = index.get(key4)
                        if irow is not None:
                            two_rows.append(irow); two_cols.append(jcol)
                            two_signs.append(0.5 * s1 * s2 * s3 * s4)
                            two_p.append(p); two_q.append(q)
                            two_r.append(r); two_s.append(s)
    return {
        "ndet": len(dets),
        "dets": dets,
        "one_rows": np.asarray(one_rows, dtype=np.int64),
        "one_cols": np.asarray(one_cols, dtype=np.int64),
        "one_signs": np.asarray(one_signs, dtype=np.float64),
        "one_p": np.asarray(one_p, dtype=np.int64),
        "one_q": np.asarray(one_q, dtype=np.int64),
        "two_rows": np.asarray(two_rows, dtype=np.int64),
        "two_cols": np.asarray(two_cols, dtype=np.int64),
        "two_signs": np.asarray(two_signs, dtype=np.float64),
        "two_p": np.asarray(two_p, dtype=np.int64),
        "two_q": np.asarray(two_q, dtype=np.int64),
        "two_r": np.asarray(two_r, dtype=np.int64),
        "two_s": np.asarray(two_s, dtype=np.int64),
    }


def _require_jax():
    try:
        import jax  # noqa: F401
        from jax import numpy as jnp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("the JAX path of diffmrc.core.spinor_ci "
                          "requires jax") from exc
    import jax
    from jax import numpy as jnp
    return jax, jnp


def spinor_ci_matrix_jax(h1, eri, ncore_so, nact, nele_act, structure=None):
    """Active CI matrix as a differentiable function of the integrals.

    Parameters
    ----------
    h1 : array, (nso, nso), complex
    eri : array, (nso, nso, nso, nso), complex
    ncore_so, nact, nele_act : int
    structure : dict, optional
        Result of ``_ci_structure``; recomputed when None.

    Returns
    -------
    H : jax array, (ndet, ndet), complex
    ecore : jax scalar
    """
    jax, jnp = _require_jax()
    h1 = jnp.asarray(h1)
    eri = jnp.asarray(eri)
    nso_req = ncore_so + nact
    if h1.shape[0] > nso_req:
        h1 = h1[:nso_req, :nso_req]
        eri = eri[:nso_req, :nso_req, :nso_req, :nso_req]
    if structure is None:
        structure = _ci_structure(ncore_so, nact, nele_act)
    act = slice(ncore_so, ncore_so + nact)
    h_aa = h1[act, act]
    eri_aa = eri[act, act, act, act]
    f = h_aa
    ecore = jnp.array(0.0, dtype=h1.dtype)
    for i in range(ncore_so):
        f = (f + eri[act, act, i, i]
             - 0.5 * (eri[act, i, i, act] + eri[act, i, act, i]))
        ecore = ecore + h1[i, i]
        for j in range(ncore_so):
            ecore = ecore + 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])
    H = jnp.zeros((structure["ndet"], structure["ndet"]),
                  dtype=jnp.result_type(h1, eri))
    H = H.at[structure["one_rows"], structure["one_cols"]].add(
        structure["one_signs"] * f[structure["one_p"], structure["one_q"]])
    H = H.at[structure["two_rows"], structure["two_cols"]].add(
        structure["two_signs"] * eri_aa[
            structure["two_p"], structure["two_q"],
            structure["two_r"], structure["two_s"]])
    H = 0.5 * (H + jnp.conj(H.T))
    return H, ecore


def eigvalsh_kramers_safe(a):
    """Hermitian eigenvalues with an eigenvalue-only, Kramers-safe JVP.

    The tangent is ``diag(U^dag dA U)``.  It contains no inverse gap, so
    it is finite at an exact Kramers degeneracy, where the plain
    reverse-mode rule of ``eigh`` evaluates ``0 / 0``.
    """
    jax, jnp = _require_jax()
    _eigvalsh = getattr(eigvalsh_kramers_safe, "_impl", None)
    if _eigvalsh is None:
        @jax.custom_jvp
        def _impl(a):
            return jnp.linalg.eigvalsh(a)

        @_impl.defjvp
        def _impl_jvp(primals, tangents):
            (a,), (da,) = primals, tangents
            w, u = jnp.linalg.eigh(a)
            dw = jnp.real(jnp.einsum("pi,pq,qi->i", jnp.conj(u), da, u))
            return w, dw

        eigvalsh_kramers_safe._impl = _impl
        _eigvalsh = _impl
    return _eigvalsh(a)


def spinor_ci_energies_jax(h1, eri, ncore_so, nact, nele_act, nroots=1,
                           structure=None):
    """Lowest ``nroots`` active CI energies, differentiable in ``h1``/``eri``.

    The value is ``eigvalsh`` of the active matrix; the derivative uses
    :func:`eigvalsh_kramers_safe`.  The returned energies include the core
    energy through the matrix diagonal.
    """
    jax, jnp = _require_jax()
    if structure is None:
        structure = _ci_structure(ncore_so, nact, nele_act)
    H, ecore = spinor_ci_matrix_jax(h1, eri, ncore_so, nact, nele_act,
                                    structure=structure)
    w = eigvalsh_kramers_safe(H)
    # The eigenvalues of a Hermitian matrix are real; the core energy is
    # real for a Hermitian Hamiltonian.  Returning a real vector keeps
    # ``jax.grad`` applicable without a holomorphic flag.
    return jnp.real(w[:nroots]) + jnp.real(ecore)


# ---------------------------------------------------------------------------
# Kramers symmetry
# ---------------------------------------------------------------------------
def kramers_unitary(mol):
    """One-particle time-reversal matrix in the j-adapted spinor basis.

    Returns ``U`` such that ``T phi_i = sum_j U[j, i] phi_j`` for the
    antiunitary time-reversal operator.  In an orthonormal spinor basis
    ``U`` satisfies ``U^dag U = 1`` and ``U^* U = -1``.
    """
    coeff = np.asarray(mol.sph2spinor_coeff())  # (2, nao_nr, nao_2c)
    nspin, nao_nr, nao_2c = coeff.shape
    if nspin != 2:
        raise ValueError("sph2spinor_coeff must carry two spin components")
    c = coeff.reshape(nspin * nao_nr, nao_2c)
    metric = np.kron(np.eye(2), mol.intor("int1e_ovlp"))
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    t0 = -1.0j * np.kron(sigma_y, np.eye(nao_nr))
    # phi = chi C with chi the spin-spatial basis; the antiunitary action
    # conjugates the coefficients, so C U = T0 C^* and
    # U = (C^dag M C)^{-1} C^dag M T0 C^*.
    gram = c.conj().T @ metric @ c
    return np.linalg.solve(gram, c.conj().T @ metric @ t0 @ c.conj())


def kramers_pairing(eigenvalues, tol=1e-8):
    """Check that every level is at least doubly degenerate.

    Returns ``(ok, gaps)`` where ``gaps`` is the array of consecutive
    level spacings.  A spectrum of an even-electron time-reversal-
    invariant Hamiltonian has each level at least twofold degenerate.
    """
    w = np.sort(np.asarray(eigenvalues, dtype=float))
    if w.size % 2:
        return False, np.diff(w)
    gaps = np.diff(w)
    ok = bool(np.all(gaps[0::2] < tol)) if gaps.size else True
    return ok, gaps


def time_reversal_determinant_matrix(unitary, dets):
    """Many-electron time-reversal matrix in a determinant basis.

    For an antiunitary one-particle ``U`` the determinant action is the
    minor ``det(U[j_a, i_b])`` with the ordering of the occupation tuples.
    The result ``T`` satisfies ``T H^* T^dag = H`` for a time-reversal-
    invariant Hamiltonian and ``T T^* = (-1)^N``.
    """
    nso = unitary.shape[0]
    ndet = len(dets)
    t = np.zeros((ndet, ndet), dtype=complex)
    idx = {d: i for i, d in enumerate(dets)}
    for jcol, occ in enumerate(dets):
        cols = np.asarray(occ, dtype=int)
        for nocc in itertools.combinations(range(nso), len(occ)):
            irow = idx.get(tuple(nocc))
            if irow is None:
                continue
            minor = unitary[np.ix_(np.asarray(nocc, dtype=int), cols)]
            t[irow, jcol] = np.linalg.det(minor)
    return t
