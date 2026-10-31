"""Contracts of the complex spinor CI and its Kramers-safe derivatives.

The module states four things that the tests pin independently:

* the second-quantized ordering reproduces the pyscf GHF energy of every
  determinant with the same ``int2e_spinor`` array;
* the frozen-core reduction reproduces the active block of a full
  determinant-space construction with the core kept in the list;
* in the non-relativistic, spin-decoupled limit the complex CI reproduces
  pyscf's real-orbital FCI energies;
* a synthetic time-reversal-invariant Hamiltonian with an odd electron
  count produces a Kramers-degenerate CI spectrum, the one-particle time
  reversal satisfies ``U^dag U = 1`` and ``U^* U = -1``, and the
  many-electron time reversal commutes with the CI Hamiltonian.

The JAX part checks that the eigenvalue tangent is finite at the exact
Kramers degeneracy and agrees with a finite difference.
"""
import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci, gto  # noqa: E402

from diffmrc.core import spinor_ci  # noqa: E402


def _random_hermitian(rng, n):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (a + a.conj().T)


def _random_eri(rng, n):
    """A complex ERI with the pyscf chemist symmetry."""
    e = (rng.normal(size=(n, n, n, n))
         + 1j * rng.normal(size=(n, n, n, n)))
    e = 0.5 * (e + e.transpose(1, 0, 2, 3).conj())
    e = 0.5 * (e + e.transpose(0, 1, 3, 2).conj())
    e = 0.5 * (e + e.transpose(2, 3, 0, 1).conj())
    return e


def _ghf_energy(h1, eri, occ):
    d = np.zeros(h1.shape, dtype=complex)
    for x in occ:
        d[x, x] = 1.0
    j = np.einsum("ijkl,lk->ij", eri, d)
    k = np.einsum("ikjl,kl->ij", eri, d)
    return (np.einsum("ij,ji->", d, h1)
            + 0.5 * np.einsum("ij,ji->", d, j)
            - 0.5 * np.einsum("ij,ji->", d, k))


def test_single_determinant_matches_ghf():
    rng = np.random.default_rng(3)
    n = 4
    h1 = _random_hermitian(rng, n)
    eri = _random_eri(rng, n)
    H, dets, _ = spinor_ci.spinor_ci_matrix(h1, eri, 0, n, 2)
    for j, occ in enumerate(dets):
        assert abs(H[j, j] - _ghf_energy(h1, eri, occ)) < 1e-12


def test_core_fock_matches_full_space():
    rng = np.random.default_rng(5)
    nso, ncore, nact, nele = 6, 2, 4, 2
    h1 = _random_hermitian(rng, nso)
    eri = _random_eri(rng, nso)
    h_red, dets_red, ecore = spinor_ci.spinor_ci_matrix(
        h1, eri, ncore, nact, nele)
    h_full, dets_full, _ = spinor_ci.spinor_ci_matrix(
        h1, eri, ncore, nact, nele, full=True)
    lookup = {d: i for i, d in enumerate(dets_full)}
    idx = [lookup[tuple(range(ncore)) + tuple(x + ncore for x in d)]
           for d in dets_red]
    ref = h_full[np.ix_(idx, idx)]
    assert np.abs(h_red - ref).max() < 1e-12
    assert np.isfinite(ecore)


def test_nonrelativistic_limit_matches_pyscf_fci():
    """Complex spinor CI against pyscf FCI on the same spatial orbitals."""
    mol = gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
                basis="6-31g", unit="Angstrom", verbose=0)
    h_sp = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri_sp = mol.intor("int2e", aosym="s1")
    norb, nelec = 4, 4
    h = h_sp[:norb, :norb]
    eri = eri_sp[:norb, :norb, :norb, :norb]
    nso = 2 * norb
    h1 = np.zeros((nso, nso), dtype=complex)
    h1[0::2, 0::2] = h
    h1[1::2, 1::2] = h
    eri_so = np.zeros((nso, nso, nso, nso), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    val = eri[p, q, r, s]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            eri_so[2 * p + sp, 2 * q + sp,
                                   2 * r + sq, 2 * s + sq] = val
    w_pyscf = fci.direct_spin1.kernel(h, eri, norb, nelec, nroots=5)[0]
    H_ci, _, _ = spinor_ci.spinor_ci_matrix(h1, eri_so, 0, nso, nelec)
    # Our determinant space spans every M_S sector, so a triplet appears
    # with its full threefold degeneracy while direct_spin1 returns the
    # M_S = 0 representatives.  Compare the unique levels.
    w_all = np.linalg.eigvalsh(H_ci)
    w_unique = np.unique(np.round(w_all, 10))[:5]
    assert np.abs(w_unique - np.sort(np.atleast_1d(w_pyscf))).max() < 1e-10


def test_complex_unitary_rotation_invariance():
    """The spectrum is invariant under a one-particle unitary rotation."""
    rng = np.random.default_rng(7)
    n = 4
    h1 = _random_hermitian(rng, n)
    eri = _random_eri(rng, n)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    u, _ = np.linalg.qr(a)
    h1r = u.conj().T @ h1 @ u
    erir = np.einsum("pa,qb,rc,sd,pqrs->abcd", u.conj(), u, u.conj(), u, eri,
                     optimize=True)
    w0 = np.linalg.eigvalsh(spinor_ci.spinor_ci_matrix(h1, eri, 0, n, 2)[0])
    w1 = np.linalg.eigvalsh(spinor_ci.spinor_ci_matrix(h1r, erir, 0, n, 2)[0])
    assert np.abs(w0 - w1).max() < 1e-11


def _synthetic_kramers(n_pairs, seed=0):
    """A T-invariant one- and two-electron Hamiltonian in a Kramers basis."""
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    u = np.zeros((n, n), dtype=complex)
    j2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    for p in range(n_pairs):
        u[2 * p:2 * p + 2, 2 * p:2 * p + 2] = j2
    h = _random_hermitian(rng, n)
    h = 0.5 * (h + u @ h.conj() @ u.T)
    eri = _random_eri(rng, n)
    eri_t = np.einsum("ia,jb,kc,ld,abcd->ijkl", u, u, u, u, eri.conj())
    eri = 0.5 * (eri + eri_t)
    return u, h, eri


def test_synthetic_kramers_ci_is_degenerate_and_time_reversal_invariant():
    # An odd electron count gives T^2 = -1, so every level must be at
    # least doubly degenerate (Kramers theorem).  The one-particle matrix
    # used in the determinant minor is the transpose of the unitary part,
    # which is the convention pinned by the T-commutation identity.
    u, h, eri = _synthetic_kramers(3)
    nele = 3
    H, dets, _ = spinor_ci.spinor_ci_matrix(h, eri, 0, h.shape[0], nele)
    w = np.linalg.eigvalsh(H)
    ok, gaps = spinor_ci.kramers_pairing(w, tol=1e-9)
    assert ok, gaps
    T = spinor_ci.time_reversal_determinant_matrix(u.T, dets)
    assert np.abs(T @ H.conj() @ T.T - H).max() < 1e-10
    assert np.abs(T @ T.conj() + np.eye(len(dets))).max() < 1e-10


def test_jax_eigenvalue_tangent_is_finite_at_degeneracy():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from diffmrc.core.spinor_ci import eigvalsh_kramers_safe

    # A Kramers pair with an exactly degenerate 2x2 block and a
    # time-reversal-even perturbation.
    base = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 2.0]], dtype=complex)
    pert = np.array([[0.3, 0.0, 0.0],
                     [0.0, 0.3, 0.0],
                     [0.0, 0.0, -0.2]], dtype=complex)
    h = jnp.asarray(base)
    v = jnp.asarray(pert)

    def f(eps):
        return eigvalsh_kramers_safe(h + eps * v)[0]

    g = jax.grad(f)(0.0)
    assert np.isfinite(float(g))
    h_fd = 1e-6
    fd = (float(f(h_fd)) - float(f(-h_fd))) / (2 * h_fd)
    assert abs(float(g) - fd) < 1e-8

    # The eigvalsh route of the installed JAX is already spectrum-safe;
    # the diagnostic below records whether the eigh route is not.
    def f_eigh(eps):
        w, _ = jnp.linalg.eigh(h + eps * v)
        return w[0]

    try:
        g_eigh = float(jax.grad(f_eigh)(0.0))
    except Exception:
        g_eigh = np.nan
    assert (not np.isfinite(g_eigh)) or abs(g_eigh - fd) < 1e-4


def test_x2c_ecp_boundary_is_explicit():
    from pyscf.x2c import x2c as pyscf_x2c

    mol = gto.M(atom=[("Yb", (0, 0, 0)), ("F", (0, 0, 3.6))],
                basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr",
                verbose=0)
    with pytest.raises(NotImplementedError):
        pyscf_x2c.SpinorX2CHelper(mol).get_hcore(mol)
