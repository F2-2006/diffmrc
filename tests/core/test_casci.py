"""The S_z-blocked CASCI eigensolver: the returned vector must live in the
lowest block alone.

Regression for a residue bug fixed on 2026-09-23. The per-block loop of
``cs_casci_blocked`` wrote each winning block's eigenvector at its own index
set but never erased the vector of a previously leading block, which sits at
a different index set. Whenever a later block won (the normal case, since
the S_z ordering visits the extreme blocks first), the returned "vector" was
a mixture of several S_z blocks.

Why it survived every previous validation: the energies returned by both
solvers were always correct (they come from ``eigh`` eigenvalues), and every
blocked-path check up to that date had been self-consistent -- HVP against a
finite difference of the same chain, hv.w == hw.v, gradient finiteness. A
self-consistent chain reproduces its own errors. The plain solver was the
only one externally cross-checked (3.2e-10 against PySCF on H2/6-31G), and
the mixture was harmless there only by accident of that system. The bug was
exposed by a ThO/ECP cross-check (CAS(4,4), ncore = 17) where the mixed
vector produced a 1-RDM wrong by 3.0 in max element and a NEVPT2 correction
wrong by 1.6 Ha; after the fix, that correction agrees with PySCF to 3.8e-7.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscfad = pytest.importorskip("pyscfad")


def _h2o_cas22_ints():
    """H2O/6-31G MO integrals over core+active (ncore=4, CAS(2,2)).

    H2O/6-31G is the smallest system that exercises the bug: its S_z blocks
    are ordered (0,2) -> (1,1) -> (2,0), the ground state lives in the
    middle block, so the leading vector of the first block is left behind
    by the unfixed loop.
    """
    from pyscfad import gto, scf

    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
    mol.basis = "6-31g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    C = jnp.asarray(mf.mo_coeff)
    h = jnp.asarray(mf.get_hcore())
    e = jnp.asarray(mol.intor("int2e", aosym="s1"))
    nm = 4 + 2
    h1 = jnp.einsum("ai,ab,bj->ij", C, h, C)[:nm, :nm]
    e1 = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, e)[
        :nm, :nm, :nm, :nm]
    return h1, e1


def _spin_block(det, ncore, nact):
    """(n_alpha, n_beta) active counts of a determinant bitmap.

    ``cas_dets_for2`` returns integer bitmaps (bit ``so`` set = spin orbital
    ``so`` occupied), not tuples of spin-orbital indices.
    """
    occ = [i for i in range(2 * (ncore + nact)) if (det >> i) & 1]
    n_alpha = sum(1 for so in occ if so % 2 == 0)
    return (n_alpha - ncore, len(occ) - n_alpha - ncore)


def test_the_blocked_ci_lives_in_a_single_spin_block():
    from diffmrc.core.casci import cs_casci, cs_casci_blocked
    from diffmrc.core.nevpt2.impl import cas_dets_for2

    h1, e1 = _h2o_cas22_ints()
    E_u, ci_u = cs_casci(h1, e1, ncore=4, nact=2, nele_act=2, return_ci=True)
    E_b, ci_b = cs_casci_blocked(h1, e1, ncore=4, nact=2, nele_act=2)
    ci_u = np.asarray(ci_u).reshape(-1)
    ci_b = np.asarray(ci_b).reshape(-1)

    # The two solvers describe the same state. A global sign is not fixed
    # by either solver (the plain one runs inverse iteration from a
    # positive start; the blocked one takes whatever eigh returns).
    assert float(E_b) == pytest.approx(float(E_u), abs=1e-10)
    assert abs(float(np.dot(ci_b, ci_u))) == pytest.approx(1.0, abs=1e-8)

    # The support of the blocked vector lies in exactly one S_z block.
    # Under the residue bug this set had two elements: the true ground
    # block plus the (0,2) singlet of the first-visited block, carrying
    # the full unit weight of that block's leading vector.
    dets = cas_dets_for2(4, 2, 2)
    nz = np.where(np.abs(ci_b) > 1e-8)[0]
    blocks = {_spin_block(dets[i], 4, 2) for i in nz}
    assert len(blocks) == 1


def test_the_blocked_ci_reproduces_the_plain_density_matrix():
    """The 1-RDM built from the blocked vector must match the plain one.

    This is the quantity the NEVPT2 chain consumes; the residue bug moved
    it by 3.0 in max element on ThO/ECP (where the wrong 1-RDM is
    non-physical: diagonal occupations above 2).
    """
    from diffmrc.core.casci import cs_casci, cs_casci_blocked
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.nevpt2.rdm import make_dm123_sf

    h1, e1 = _h2o_cas22_ints()
    _, ci_u = cs_casci(h1, e1, ncore=4, nact=2, nele_act=2, return_ci=True)
    _, ci_b = cs_casci_blocked(h1, e1, ncore=4, nact=2, nele_act=2)
    dets = cas_dets_for2(4, 2, 2)
    dm_u, _, _ = make_dm123_sf(jnp.asarray(ci_u).reshape(-1), dets, 2, 4)
    dm_b, _, _ = make_dm123_sf(jnp.asarray(ci_b).reshape(-1), dets, 2, 4)
    assert float(np.abs(np.asarray(dm_b) - np.asarray(dm_u)).max()) < 1e-8


def _random_symmetric_ints(n=6, seed=5):
    """A synthetic Hamiltonian's ingredients (no chemistry, pure algebra).

    The two-electron tensor carries the chemist symmetries p<->q, r<->s
    and (pq)<->(rs), so the CI matrix is exactly Hermitian and the two
    eigensolver routes (rotations and inverse iteration) target the same
    operator.  A tensor with only the p<->q symmetry leaves the CI matrix
    non-symmetric: the decomposition reads one triangle while the
    inverse-iteration solve reads the full matrix, and the two ground
    vectors then disagree (measured before this fix).
    """
    rng = np.random.default_rng(seed)
    h1e = rng.normal(size=(n, n))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(size=(n, n, n, n))
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    return jnp.asarray(h1e), jnp.asarray(eri)


def test_the_casci_and_cas_dets_orderings_agree():
    """The two determinant constructions must be the same list, in order.

    ``cs_casci`` builds its Slater basis with ``combinations`` over spin
    orbitals after the core block; the analysis-layer consumers read CI
    vectors against ``cas_dets_for2``.  If the two lists ever diverge,
    every per-state number is silently permuted, so the ordering is
    pinned here instead of assumed.
    """
    from itertools import combinations

    from diffmrc.core.nevpt2.impl import cas_dets_for2

    ncore, nact, nele = 0, 3, 2
    ref = [sum(1 << x for x in (tuple(range(2 * ncore)) + c))
           for c in combinations(range(2 * ncore,
                                     2 * (ncore + nact)), nele)]
    assert list(cas_dets_for2(ncore, nact, nele)) == ref


def test_state_average_vectors_are_orthonormal_and_match_the_single_root():
    from diffmrc.core.casci import cs_casci

    h1e, eri = _random_symmetric_ints()
    E, V = cs_casci(h1e, eri, ncore=0, nact=3, nele_act=2, nroots=2,
                    return_ci=True)
    V = np.asarray(V)
    assert V.ndim == 2 and V.shape[1] == 2
    assert np.allclose(V.T @ V, np.eye(2), atol=1e-10)
    # The ground vector of the state-averaged call agrees with the
    # single-root path (inverse iteration) up to its own residual.
    _, v1 = cs_casci(h1e, eri, ncore=0, nact=3, nele_act=2, nroots=1,
                     return_ci=True)
    overlap = abs(float(np.dot(np.asarray(v1), V[:, 0])))
    assert abs(overlap - 1.0) < 1e-6


def test_state_average_energy_is_bit_identical_with_and_without_vectors():
    from diffmrc.core.casci import cs_casci

    h1e, eri = _random_symmetric_ints(seed=7)
    E_pairs, _ = cs_casci(h1e, eri, ncore=0, nact=3, nele_act=2, nroots=2,
                          return_ci=True)
    E_plain = cs_casci(h1e, eri, ncore=0, nact=3, nele_act=2, nroots=2)
    assert np.asarray(E_pairs) == np.asarray(E_plain)


def test_per_state_entropies_from_the_state_average_vectors():
    from diffmrc.analysis.ci import entropy_from_ci
    from diffmrc.core.casci import cs_casci
    from diffmrc.core.nevpt2.impl import cas_dets_for2

    h1e, eri = _random_symmetric_ints(seed=11)
    _, V = cs_casci(h1e, eri, ncore=0, nact=3, nele_act=2, nroots=2,
                    return_ci=True)
    dets = cas_dets_for2(0, 3, 2)
    for k in range(2):
        s_k = entropy_from_ci(np.asarray(V[:, k]), dets, ncore=0, nact=3)
        assert np.all(np.isfinite(s_k))
        assert np.all(s_k >= -1e-12)
        assert np.all(s_k <= np.log(4) + 1e-12)


# ---------------------------------------------------------------------------
# eig_solver='davidson': the matrix-free CI route.
# ---------------------------------------------------------------------------


def test_the_matrix_free_solver_reproduces_the_dense_block():
    """Davidson on the matrix-free operator against the dense block eigh.

    The two routes build the Hamiltonian by independent code paths
    (``core.ci_matvec`` assembles a string-excitation operator, the dense
    path applies the Condon-Slater rules determinant by determinant) and
    use different determinant orderings; the returned vector is mapped
    back into the ``cs_casci`` layout by ``_mf_global_order``. Agreement
    of both the energy and the vector pins the mapping and the sign
    convention.
    """
    from diffmrc.core.casci import cs_casci_blocked

    h1, e1 = _h2o_cas22_ints()
    E_d, ci_d = cs_casci_blocked(h1, e1, ncore=4, nact=2, nele_act=2,
                                 eig_solver='eigh_safe')
    E_m, ci_m = cs_casci_blocked(h1, e1, ncore=4, nact=2, nele_act=2,
                                 eig_solver='davidson')
    assert float(E_m) == pytest.approx(float(E_d), abs=1e-9)
    overlap = abs(float(np.asarray(ci_d) @ np.asarray(ci_m)))
    assert overlap == pytest.approx(1.0, abs=1e-9)
    # The vector must still live in one S_z block alone.
    from diffmrc.core.nevpt2.impl import cas_dets_for2

    dets = cas_dets_for2(4, 2, 2)
    nz = np.where(np.abs(np.asarray(ci_m)) > 1e-8)[0]
    blocks = {_spin_block(dets[i], 4, 2) for i in nz}
    assert len(blocks) == 1


def test_the_matrix_free_solver_matches_the_dense_gradient():
    """First derivative through both CI routes against a central difference.

    ``h1e`` is the only differentiated argument; its tangent is restricted
    to the symmetric subspace because the Hamiltonian always symmetrizes
    the one-electron part.
    """
    from diffmrc.core.casci import cs_casci_blocked

    h1, e1 = _h2o_cas22_ints()
    rng = np.random.default_rng(31)
    D = jnp.asarray(0.5 * (rng.normal(size=h1.shape)
                           + rng.normal(size=h1.shape).T))
    D = 0.5 * (D + D.T)

    def energy(h, solver):
        return cs_casci_blocked(h, e1, ncore=4, nact=2, nele_act=2,
                                eig_solver=solver)[0]

    step = 1e-6
    for solver in ('eigh_safe', 'davidson'):
        ad = float(jnp.sum(jax.grad(lambda h: energy(h, solver))(h1) * D))
        fd = (float(energy(h1 + step * D, solver))
              - float(energy(h1 - step * D, solver))) / (2.0 * step)
        assert abs(ad - fd) < 1e-8 * max(1.0, abs(fd)), solver

    g_d = jax.grad(lambda h: energy(h, 'eigh_safe'))(h1)
    g_m = jax.grad(lambda h: energy(h, 'davidson'))(h1)
    assert float(jnp.max(jnp.abs(g_d - g_m))) < 1e-9


def test_the_matrix_free_solver_gradient_survives_jit():
    """``grad(jit(E))`` must equal the eager gradient.

    This is the path that exposed a lowering failure in the matrix-free
    chain: ``string_tables`` built its constant tables with ``jnp.asarray``
    inside the trace, the operator closures captured them, and lowering a
    ``custom_vjp`` that closes over a ``DynamicJaxprTracer`` constant
    failed with "No constant handler for type: DynamicJaxprTracer".
    Building the tables under ``jax.ensure_compile_time_eval`` keeps them
    concrete. The test pins both the value and the gradient.
    """
    from diffmrc.core.casci import cs_casci_blocked

    h1, e1 = _h2o_cas22_ints()

    def E(h):
        return cs_casci_blocked(h, e1, ncore=4, nact=2, nele_act=2,
                                eig_solver='davidson')[0]

    g_eager = jax.grad(E)(h1)
    g_jit = jax.grad(jax.jit(E))(h1)
    assert np.isfinite(np.asarray(g_jit)).all()
    assert float(jnp.max(jnp.abs(g_eager - g_jit))) < 1e-10
    assert float(jax.jit(E)(h1)) == pytest.approx(float(E(h1)), abs=1e-12)
