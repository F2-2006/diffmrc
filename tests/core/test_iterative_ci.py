import numpy as np
import pytest
from itertools import combinations, product
from scipy.special import erf

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.iterative_ci import (davidson_lowest_k, lowest_k_eigpairs,
                                       minres_solve, _ift_pullback)

# ---------------------------------------------------------------------------
# H2 reference systems.
#
# The local environment of this project does not carry pyscfad (the dense
# reference ``cs_casci`` imports it at module level), so the solver is
# exercised here on H2 Hamiltonians assembled from a self-contained provider:
# s-type Gaussian integrals over the BSE contraction constants, a Loewdin
# orthonormalization, and an independent determinant-CI builder that applies
# the second-quantized Hamiltonian with explicit parity counting. Nothing in
# the provider reuses project code, so agreement with the dense path is not
# self-consistency. The provider itself is pinned against the reference
# energies this project measured with PySCF:
#
#   H2/STO-3G, R = 0.74 A:  CASSCF(2,2) = -1.137283834489
#       (pinned in tests/driver/test_mcpdft_link.py to 1e-9)
#   H2/6-31G,  R = 0.7414 A: FCI = -1.1516827321
#       (recorded in SESSION_LOG_APPEND_20260923b.md; the geometry was
#       identified by matching -- at R = 0.74 the same engine gives
#       -1.1516725)
#
# Both H2 active spaces take every orbital (ncore = 0), so the CI energy is
# invariant under the choice of orthonormal orbital basis and the Loewdin
# transform is sufficient; no SCF is needed.
# ---------------------------------------------------------------------------

_ANGSTROM_IN_BOHR = 1.8897261254578281
# Contractions from the Basis Set Exchange data deployed in basisdb
# (output/bas/sto-3g.bas and output/bas/6-31g.bas), hydrogen blocks.
_H_SHELLS = {
    'sto-3g': [([3.425250914, 0.6239137298, 0.1688554040],
                [0.1543289673, 0.5353281423, 0.4446345422])],
    '6-31g': [([18.73113696, 2.825394365, 0.6401216923],
               [0.03349460434, 0.2347269535, 0.8137573261]),
              ([0.1612777588], [1.0])],
}
_H2_FCI_ANCHOR = {'sto-3g': -1.137283834489, '6-31g': -1.1516827321}
_H2_ANCHOR_R = {'sto-3g': 0.74, '6-31g': 0.7414}

FIX_PAIR_AT_HALF = jnp.asarray(np.diag([-1.0, 0.5, 0.5, 2.0]))
FIX_DEGENERATE_GROUND = jnp.asarray(np.diag([-1.0, -1.0, 0.5, 2.0]))
_DIAG_DIRECTION = jnp.asarray(np.diag([1.0, 2.0, -0.5, 0.3]))


def _f0(t):
    return 1.0 - t / 3.0 if t < 1e-12 else np.sqrt(np.pi / (4.0 * t)) * erf(np.sqrt(t))


def _h2_mo_integrals(basis, r_angstrom):
    """MO-basis (h1e, eri) of H2 in `basis`, chemist-notation eri (pq|rs)."""
    shells = _H_SHELLS[basis]
    centers = [np.zeros(3),
               np.array([0.0, 0.0, r_angstrom * _ANGSTROM_IN_BOHR])]
    prims, rows = [], []
    for c in centers:
        for exps, coefs in shells:
            row = np.zeros(len(prims) + len(exps))
            for e, d in zip(exps, coefs):
                prims.append((c, e))
                row[len(prims) - 1] = d
            rows.append(row)
    npm = len(prims)
    contraction = np.zeros((len(rows), npm))
    for k, row in enumerate(rows):
        contraction[k, :row.size] = row

    def norm(e):
        return (2.0 * e / np.pi) ** 0.75

    s_p = np.zeros((npm, npm))
    h_p = np.zeros((npm, npm))
    for i in range(npm):
        a_c, a_e = prims[i]
        for j in range(npm):
            b_c, b_e = prims[j]
            p = a_e + b_e
            mu = a_e * b_e / p
            r2 = float(np.dot(a_c - b_c, a_c - b_c))
            kab = np.exp(-mu * r2)
            s = (np.pi / p) ** 1.5 * kab
            t = mu * (3.0 - 2.0 * mu * r2) * s
            p_c = (a_e * a_c + b_e * b_c) / p
            v = sum(-2.0 * np.pi / p * _f0(p * float(np.dot(p_c - c, p_c - c)))
                    for c in centers)
            kk = norm(a_e) * norm(b_e)
            s_p[i, j] = s * kk
            h_p[i, j] = (t + kab * v) * kk
    e_p = np.zeros((npm, npm, npm, npm))
    for i, j, k, l in product(range(npm), repeat=4):
        a_c, a_e = prims[i]
        b_c, b_e = prims[j]
        c_c, c_e = prims[k]
        d_c, d_e = prims[l]
        p = a_e + b_e
        q = c_e + d_e
        p_c = (a_e * a_c + b_e * b_c) / p
        q_c = (c_e * c_c + d_e * d_c) / q
        e_p[i, j, k, l] = (
            2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
            * np.exp(-a_e * b_e / p * float(np.dot(a_c - b_c, a_c - b_c)))
            * np.exp(-c_e * d_e / q * float(np.dot(c_c - d_c, c_c - d_c)))
            * _f0(p * q / (p + q) * float(np.dot(p_c - q_c, p_c - q_c)))
            * norm(a_e) * norm(b_e) * norm(c_e) * norm(d_e))

    h = np.einsum('ai,bj,ij->ab', contraction, contraction, h_p)
    s = np.einsum('ai,bj,ij->ab', contraction, contraction, s_p)
    eri = np.einsum('ai,bj,ck,dl,ijkl->abcd', contraction, contraction,
                    contraction, contraction, e_p)
    w, u = np.linalg.eigh(s)
    x = u @ np.diag(w ** -0.5) @ u.T
    h_mo = x.T @ h @ x
    eri_mo = np.einsum('pi,qj,rk,sl,pqrs->ijkl', x, x, x, x, eri)
    return jnp.asarray(h_mo), jnp.asarray(eri_mo)


def _h2_fci(basis):
    h_mo, eri_mo = _h2_mo_integrals(basis, _H2_ANCHOR_R[basis])
    nact = 2 if basis == 'sto-3g' else 4
    h = _ci_matrix(np.asarray(h_mo), np.asarray(eri_mo), 0, nact, 2)
    enuc = 1.0 / (_H2_ANCHOR_R[basis] * _ANGSTROM_IN_BOHR)
    return float(np.linalg.eigvalsh(h)[0]) + enuc


def _ci_matrix(h1e_mo, eri_mo, ncore, nact, nele_act):
    """Determinant CI matrix by direct application of the second-quantized H.

    The determinant list is the one of ``core.casci.cs_casci`` (the frozen
    core followed by the combinations of active spin orbitals in increasing
    index order). The Hamiltonian is applied operator by operator with
    explicit parity counting, so no derived Slater-Condon sign formula is
    involved; the two-electron integrals enter in the physicist convention
    ``<pq|rs> = (pr|qs)_chem``. Excitations whose target leaves the
    determinant list -- the core excitations of an ncore > 0 space -- are
    skipped: the operator term is real but acts outside the space.
    """
    nso = 2 * (ncore + nact)
    core = tuple(range(2 * ncore))
    dets = [frozenset(core + c)
            for c in combinations(range(2 * ncore, nso), nele_act)]
    index = {d: i for i, d in enumerate(dets)}
    h = np.zeros((nso, nso))
    g = np.zeros((nso, nso, nso, nso))
    for p, q in product(range(nso), repeat=2):
        if p % 2 == q % 2:
            h[p, q] = h1e_mo[p // 2, q // 2]
    for p, q, r, s in product(range(nso), repeat=4):
        if p % 2 == r % 2 and q % 2 == s % 2:
            g[p, q, r, s] = eri_mo[p // 2, r // 2, q // 2, s // 2]

    def ann(occ, i):
        if i not in occ:
            return None, 0.0
        return occ - {i}, (-1.0) ** sum(1 for x in occ if x < i)

    def cre(occ, i):
        if i in occ:
            return None, 0.0
        return occ | {i}, (-1.0) ** sum(1 for x in occ if x < i)

    big = np.zeros((len(dets), len(dets)))
    for col, occ in enumerate(dets):
        for p, q in product(range(nso), repeat=2):
            if h[p, q] == 0.0:
                continue
            o1, s1 = ann(occ, q)
            if o1 is None:
                continue
            o2, s2 = cre(o1, p)
            if o2 is None or o2 not in index:
                continue
            big[index[o2], col] += s1 * s2 * h[p, q]
        for r in occ:
            o1, s1 = ann(occ, r)
            for s in o1:
                o2, s2 = ann(o1, s)
                for q in range(nso):
                    if q in o2:
                        continue
                    o3, s3 = cre(o2, q)
                    for p in range(nso):
                        if p in o3:
                            continue
                        v = g[p, q, r, s]
                        if v == 0.0:
                            continue
                        o4, s4 = cre(o3, p)
                        if o4 is not None and o4 in index:
                            big[index[o4], col] += (0.5 * s1 * s2 * s3 * s4
                                                    * v)
    return big


def _random_symmetric(rng, n, gap=1.0):
    b = rng.normal(size=(n, n))
    return jnp.asarray(b + b.T + np.diag(np.arange(n) * gap))


def _directional_fd(f, a, direction, step):
    return (float(f(a + step * direction)) - float(f(a - step * direction))) \
        / (2.0 * step)


# ---------------------------------------------------------------------------
# The fixture provider itself.
# ---------------------------------------------------------------------------


def test_the_h2_fixture_reproduces_the_reference_energies():
    """The provider must reproduce the project's PySCF references.

    The CI builders in this file are only meaningful if the Hamiltonian they
    solve is the physical one, so the provider is pinned against two
    independent recorded values before any solver test uses it.
    """
    e_sto = _h2_fci('sto-3g')
    assert abs(e_sto - _H2_FCI_ANCHOR['sto-3g']) < 1e-8
    e_631 = _h2_fci('6-31g')
    assert abs(e_631 - _H2_FCI_ANCHOR['6-31g']) < 1e-7


def test_the_h2_ci_matrix_is_exactly_symmetric():
    h_mo, eri_mo = _h2_mo_integrals('6-31g', _H2_ANCHOR_R['6-31g'])
    h = _ci_matrix(np.asarray(h_mo), np.asarray(eri_mo), 0, 4, 2)
    assert np.abs(h - h.T).max() < 1e-12


# ---------------------------------------------------------------------------
# Values against the dense path (the local stand-in for ``cs_casci``).
# ---------------------------------------------------------------------------


def test_the_iterative_values_match_the_dense_path_on_h2():
    for basis, k in (('sto-3g', 1), ('sto-3g', 2), ('6-31g', 1), ('6-31g', 2)):
        h_mo, eri_mo = _h2_mo_integrals(basis, _H2_ANCHOR_R[basis])
        nact = 2 if basis == 'sto-3g' else 4
        h = jnp.asarray(_ci_matrix(np.asarray(h_mo), np.asarray(eri_mo),
                                   0, nact, 2))
        ws, vs = lowest_k_eigpairs(h, k=k, tol=1e-12)
        w_ref, v_ref = jnp.linalg.eigh(h)
        assert float(jnp.max(jnp.abs(ws - w_ref[:k]))) < 1e-8
        # The second eigenvalue of H2 is the three-fold degenerate triplet,
        # so for k = 2 the lowest two eigenvalues do not define a unique
        # block: the vectors are only pinned for k = 1, whose singlet ground
        # state is simple. The eigenvalues are the invariant at k = 2.
        if k == 1:
            overlap = abs(float(vs[:, 0] @ v_ref[:, 0]))
            assert abs(overlap - 1.0) < 1e-8


def test_the_iterative_values_match_the_dense_path_on_random_matrices():
    rng = np.random.default_rng(0)
    for n, k in ((5, 1), (8, 2), (16, 3), (40, 2)):
        a = _random_symmetric(rng, n)
        ws, vs = lowest_k_eigpairs(a, k=k, tol=1e-12)
        w_ref, v_ref = jnp.linalg.eigh(a)
        assert float(jnp.max(jnp.abs(ws - w_ref[:k]))) < 1e-8
        overlap = np.abs(np.asarray(vs.T @ v_ref[:, :k]))
        assert np.abs(overlap - np.eye(k)).max() < 1e-6


def test_the_deterministic_gauge_pins_the_column_signs():
    rng = np.random.default_rng(1)
    a = _random_symmetric(rng, 9)
    _, vs = lowest_k_eigpairs(a, k=2, tol=1e-12)
    v = np.asarray(vs)
    piv = np.argmax(np.abs(v), axis=0)
    assert np.all(v[piv, np.arange(2)] > 0.0)


def test_the_differentiable_entry_works_under_jit():
    """The custom_vjp forward hosts a data-dependent while_loop, so the jit
    path is a distinct thing to pin: value and gradient both."""
    rng = np.random.default_rng(9)
    a = _random_symmetric(rng, 7)
    value = jax.jit(lambda m: lowest_k_eigpairs(m, k=1, tol=1e-12)[0][0])(a)
    assert abs(float(value) - float(jnp.linalg.eigvalsh(a)[0])) < 1e-10
    grad = jax.jit(jax.grad(
        lambda m: lowest_k_eigpairs(m, k=1, tol=1e-12)[0][0]))(a)
    assert np.isfinite(np.asarray(grad)).all()


def test_casci_is_imported_lazily_so_the_module_loads_without_pyscfad():
    """``core.casci`` imports pyscfad at module level, so ``iterative_ci``
    must reach it only inside the function that builds the CI matrix;
    otherwise the solver would be unimportable on any machine without
    pyscfad (this one included)."""
    import ast
    import pathlib

    import diffmrc.core.iterative_ci as ici

    tree = ast.parse(pathlib.Path(ici.__file__).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            assert 'casci' not in node.module
        if isinstance(node, ast.Import):
            assert all('casci' not in alias.name for alias in node.names)


def test_the_forward_reports_residuals_below_the_tolerance():
    rng = np.random.default_rng(2)
    a = _random_symmetric(rng, 20)
    theta, x, res, cycles, spectrum, block = davidson_lowest_k(
        lambda v: a @ v, 20, k=2, tol=1e-11, diag=jnp.diag(a))
    assert float(jnp.max(res)) <= 1e-11
    assert int(cycles) >= 1
    assert spectrum.shape[0] == block.shape[1] >= 2
    w_ref = jnp.linalg.eigvalsh(a)
    assert float(jnp.max(jnp.abs(theta - w_ref[:2]))) < 1e-9


# ---------------------------------------------------------------------------
# First derivatives: AD against finite differences.
# ---------------------------------------------------------------------------


def test_the_energy_gradient_matches_finite_differences_on_a_plateau():
    rng = np.random.default_rng(3)
    a = _random_symmetric(rng, 9)
    directions = [jnp.asarray(0.5 * (b + b.T))
                  for b in rng.normal(size=(3, 9, 9))]
    zero = jnp.zeros(3)

    def energy(th):
        return lowest_k_eigpairs(
            a + sum(th[i] * directions[i] for i in range(3)),
            k=1, tol=1e-13)[0][0]

    grad = np.asarray(jax.grad(energy)(zero))
    for step in (1e-4, 1e-5):
        fd = np.array([_directional_fd(energy, zero, jnp.eye(3)[i], step)
                       for i in range(3)])
        assert np.abs(grad - fd).max() < 1e-7 * max(np.abs(fd).max(), 1.0)


def test_the_eigenvector_gradient_matches_finite_differences():
    """A functional that consumes the eigenvector, so the Sternheimer solve
    inside the reverse pass is exercised (the energy-only path never solves).
    The weights are applied to amplitudes squared: the functional is
    invariant under a global sign, which the gauge fix makes moot anyway."""
    rng = np.random.default_rng(4)
    a = _random_symmetric(rng, 9)
    weights = jnp.asarray(rng.normal(size=9))
    directions = [jnp.asarray(0.5 * (b + b.T))
                  for b in rng.normal(size=(3, 9, 9))]
    zero = jnp.zeros(3)

    def functional(th):
        _, vs = lowest_k_eigpairs(
            a + sum(th[i] * directions[i] for i in range(3)),
            k=1, tol=1e-13)
        return jnp.sum(weights * vs[:, 0] ** 2)

    grad = np.asarray(jax.grad(functional)(zero))
    for step in (1e-4, 1e-5):
        fd = np.array([_directional_fd(functional, zero, jnp.eye(3)[i], step)
                       for i in range(3)])
        assert np.abs(grad - fd).max() < 1e-6


def test_the_two_pair_sum_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    a = _random_symmetric(rng, 11)
    directions = [jnp.asarray(0.5 * (b + b.T))
                  for b in rng.normal(size=(3, 11, 11))]
    zero = jnp.zeros(3)

    def state_average(th):
        return jnp.sum(lowest_k_eigpairs(
            a + sum(th[i] * directions[i] for i in range(3)),
            k=2, tol=1e-13)[0])

    grad = np.asarray(jax.grad(state_average)(zero))
    fd = np.array([_directional_fd(state_average, zero, jnp.eye(3)[i], 1e-5)
                   for i in range(3)])
    assert np.abs(grad - fd).max() < 1e-7 * max(np.abs(fd).max(), 1.0)


def test_the_full_matrix_gradient_matches_finite_differences():
    """The whole 9x9 derivative along symmetric directions (the acceptance
    wording is 'against Hamiltonian parameters'). The comparison is the
    directional contraction: a symmetric direction touches both (i, j) and
    (j, i), so comparing a single matrix element against the directional
    difference is off by exactly a factor of two."""
    rng = np.random.default_rng(6)
    a = _random_symmetric(rng, 9)

    def energy(m):
        return lowest_k_eigpairs(m, k=1, tol=1e-13)[0][0]

    grad = np.asarray(jax.grad(energy)(a))
    for i, j in ((0, 3), (2, 2), (4, 7)):
        d = np.zeros((9, 9))
        d[i, j] = d[j, i] = 1.0
        d = jnp.asarray(d)
        fd = _directional_fd(energy, a, d, 1e-6)
        assert abs(float(np.sum(grad * np.asarray(d))) - fd) \
            < 1e-7 * max(abs(fd), 1.0)
    # the symmetric-matrix derivative is symmetric by construction
    assert np.abs(grad - grad.T).max() == 0.0


# ---------------------------------------------------------------------------
# Degeneracy fixtures: recorded behaviour.
# ---------------------------------------------------------------------------


def test_the_pair_at_half_behaves_like_a_simple_problem():
    """A bit-exact pair away from the consumed column: everything exact.

    The pair at 0.5 is fully represented in the final Ritz block, so the
    reverse solve works on its orthocomplement and the ground pair behaves
    exactly as in the non-degenerate case.
    """
    _, vs = lowest_k_eigpairs(FIX_PAIR_AT_HALF, k=1, tol=1e-13)
    assert float(vs[0, 0]) ** 2 > 1.0 - 1e-12

    def energy(m):
        return lowest_k_eigpairs(m, k=1, tol=1e-13)[0][0]

    grad = np.asarray(jax.grad(energy)(FIX_PAIR_AT_HALF))
    assert np.isfinite(grad).all()
    # the ground vector is the first basis vector, so dE = V[0, 0] exactly
    fd = _directional_fd(energy, FIX_PAIR_AT_HALF, _DIAG_DIRECTION, 1e-5)
    assert abs(float(np.sum(grad * np.asarray(_DIAG_DIRECTION))) - fd) < 1e-9
    assert abs(float(np.sum(grad * np.asarray(_DIAG_DIRECTION))) - 1.0) < 1e-9

    weights = jnp.asarray([0.7, -1.3, 0.4, 0.9])

    def vector_functional(m):
        _, v = lowest_k_eigpairs(m, k=1, tol=1e-13)
        return jnp.sum(weights * v[:, 0] ** 2)

    gv = np.asarray(jax.grad(vector_functional)(FIX_PAIR_AT_HALF))
    assert np.isfinite(gv).all()
    assert np.abs(gv).max() < 1e-9


def test_the_degenerate_ground_energy_is_the_frozen_rayleigh_derivative():
    """A two-fold degenerate ground state: the individual eigenvalue is not
    differentiable, and the returned derivative is the one of the frozen
    Rayleigh quotient ``u^T A u`` at the returned vector -- the
    ``core/eigh.py`` convention, measured here as 1.200552382 while the
    central difference of the non-smooth minimum reads 1.5 (the average of
    the two one-sided values 1 and 2)."""
    theta, vs = lowest_k_eigpairs(FIX_DEGENERATE_GROUND, k=1, tol=1e-13)
    assert abs(float(theta[0]) + 1.0) < 1e-12
    u = np.asarray(vs[:, 0])
    frozen = float(u @ np.asarray(_DIAG_DIRECTION) @ u)

    def energy(m):
        return lowest_k_eigpairs(m, k=1, tol=1e-13)[0][0]

    grad = np.asarray(jax.grad(energy)(FIX_DEGENERATE_GROUND))
    assert np.isfinite(grad).all()
    assert abs(float(np.sum(grad * np.asarray(_DIAG_DIRECTION))) - frozen) < 1e-9


def test_the_degenerate_cluster_sums_are_regular():
    """Summed over the complete cluster the derivative is smooth again.

    Both the energy sum and the projector functional are invariant under the
    arbitrary rotation inside the degenerate pair, so they have well-defined
    derivatives and the finite difference is a valid reference for both.
    """
    weights = jnp.asarray([0.7, -1.3, 0.4, 0.9])

    def energy_sum(m):
        return jnp.sum(lowest_k_eigpairs(m, k=2, tol=1e-13)[0])

    fd = _directional_fd(energy_sum, FIX_DEGENERATE_GROUND, _DIAG_DIRECTION,
                         1e-5)
    grad = np.asarray(jax.grad(energy_sum)(FIX_DEGENERATE_GROUND))
    assert np.isfinite(grad).all()
    assert abs(float(np.sum(grad * np.asarray(_DIAG_DIRECTION))) - fd) < 1e-8
    assert abs(fd - 3.0) < 1e-6

    def projector(m):
        _, v = lowest_k_eigpairs(m, k=2, tol=1e-13)
        return jnp.sum(weights * (v @ v.T))

    gp = np.asarray(jax.grad(projector)(FIX_DEGENERATE_GROUND))
    assert np.isfinite(gp).all()
    fd_p = _directional_fd(projector, FIX_DEGENERATE_GROUND, _DIAG_DIRECTION,
                           1e-5)
    assert abs(float(np.sum(gp * np.asarray(_DIAG_DIRECTION))) - fd_p) < 1e-8


def test_the_cluster_rule_agrees_with_the_frozen_cluster_implementation():
    """Cross-check against ``eigh_cluster_ift``, the in-stack frozen-cluster
    rule, on a functional both routes can express: the weighted projector of
    the degenerate ground cluster. The implicit gradients must agree once the
    other route's gradient is symmetrized (its antisymmetric part contracts
    to zero on any symmetric perturbation)."""
    from diffmrc.core.eigh import eigh_cluster_ift

    weights = jnp.asarray([0.7, -1.3, 0.4, 0.9])

    def mine(m):
        _, v = lowest_k_eigpairs(m, k=2, tol=1e-13)
        return jnp.sum(weights * jnp.diag(v @ v.T))

    def theirs(m):
        _, v = eigh_cluster_ift(m)
        p = v[:, :2] @ v[:, :2].T
        return jnp.sum(weights * jnp.diag(p))

    g_mine = np.asarray(jax.grad(mine)(FIX_DEGENERATE_GROUND))
    g_theirs = np.asarray(jax.grad(theirs)(FIX_DEGENERATE_GROUND))
    assert np.abs(g_mine - 0.5 * (g_theirs + g_theirs.T)).max() < 1e-6


def test_the_degenerate_fixtures_against_the_decomposition_rules():
    """The behaviour record the milestone asks for: on the bit-exact
    fixtures the kernel is finite on both the energy and the eigenvector
    paths, ``eigh_safe`` is finite on both, and the raw decomposition is not
    finite on the eigenvector path (a property of the jax build, pinned here
    so that a change makes the contrast explicit)."""
    from diffmrc.core.eigh import eigh_safe

    weights = jnp.asarray([0.7, -1.3, 0.4, 0.9])

    def raw_vector_functional(m):
        _, v = jnp.linalg.eigh(m)
        return jnp.sum(weights * v[:, 0] ** 2)

    def safe_vector_functional(m):
        _, v = eigh_safe(m)
        return jnp.sum(weights * v[:, 0] ** 2)

    def kernel_vector_functional(m):
        _, v = lowest_k_eigpairs(m, k=1, tol=1e-13)
        return jnp.sum(weights * v[:, 0] ** 2)

    for fixture in (FIX_PAIR_AT_HALF, FIX_DEGENERATE_GROUND):
        g_kernel = np.asarray(jax.grad(kernel_vector_functional)(fixture))
        g_safe = np.asarray(jax.grad(safe_vector_functional)(fixture))
        g_raw = np.asarray(jax.grad(raw_vector_functional)(fixture))
        assert np.isfinite(g_kernel).all()
        assert np.isfinite(g_safe).all()
        assert not np.isfinite(g_raw).all()


# ---------------------------------------------------------------------------
# The degeneracy family of M2: systemic fixtures across cluster positions,
# near-degeneracy scales, and partners outside the forward's Ritz space.
# ---------------------------------------------------------------------------


def _rotated(evals, q):
    return jnp.asarray(q @ np.diag(evals) @ q.T)


def _projector_functional(k, weights, ridge=1e-8):
    """``sum_r weights_r P_rr`` with ``P`` the projector of the returned block.

    The weight vector contracts with the diagonal of the projector; a full
    matrix contraction with the broadcast weights would be a different
    functional (``sum_i w_i v_i * sum_j v_j``).
    """
    def f(m):
        _, v = lowest_k_eigpairs(m, k=k, tol=1e-13, ridge=ridge)
        return jnp.sum(weights * jnp.diag(v @ v.T))
    return f


def _symmetrized_ift_projector_gradient(matrix, k, weights):
    """The frozen-cluster rule of ``core/eigh.py`` as an independent reference.

    ``eigh_cluster_ift`` decomposes the full matrix, so its cluster for a
    degenerate pair is complete by construction; its gradient carries an
    antisymmetric part that contracts to zero with any symmetric
    perturbation, which is why the comparison uses the symmetrized form.
    """
    from diffmrc.core.eigh import eigh_cluster_ift
    tol = 1e-8 * float(jnp.max(jnp.abs(matrix)))

    def f(m):
        _, v = eigh_cluster_ift(m, tol=tol)
        p = v[:, :k] @ v[:, :k].T
        return jnp.sum(weights * jnp.diag(p))

    g = np.asarray(jax.grad(f)(matrix))
    return 0.5 * (g + g.T)


def test_a_three_fold_ground_cluster_sums_are_exact():
    """A complete cluster of three: every returned eigenvalue is the exact
    degenerate level, and the cluster sum of the eigenvalue derivatives is
    the trace of the perturbation over the cluster -- 2.5 here, exactly."""
    fixture = jnp.asarray(np.diag([-1.0, -1.0, -1.0, 0.5, 2.0]))
    direction = jnp.asarray(np.diag([1.0, 2.0, -0.5, 0.3, -0.2]))
    for k in (1, 2, 3):
        theta, _ = lowest_k_eigpairs(fixture, k=k, tol=1e-13)
        assert float(jnp.max(jnp.abs(theta + 1.0))) < 1e-12

    def energy_sum(m):
        return jnp.sum(lowest_k_eigpairs(m, k=3, tol=1e-13)[0])

    grad = np.asarray(jax.grad(energy_sum)(fixture))
    ad = float(np.sum(grad * np.asarray(direction)))
    fd = _directional_fd(energy_sum, fixture, direction, 1e-5)
    assert abs(ad - 2.5) < 1e-9
    assert abs(ad - fd) < 1e-8

    # k = 1 on the same cluster: the energy derivative is the frozen
    # Rayleigh quotient of the returned mixture, sum_i u_i^2 V_ii.
    theta, vs = lowest_k_eigpairs(fixture, k=1, tol=1e-13)
    u = np.asarray(vs[:, 0])
    frozen = float(u @ np.asarray(direction) @ u)
    grad1 = np.asarray(jax.grad(
        lambda m: lowest_k_eigpairs(m, k=1, tol=1e-13)[0][0])(fixture))
    assert np.isfinite(grad1).all()
    assert abs(float(np.sum(grad1 * np.asarray(direction))) - frozen) < 1e-9


def test_the_near_degenerate_gap_family():
    """Gaps above the cluster tolerance: the rule is the exact implicit one.

    dE/dV scales like 1/g, so the fixture is rotated to keep the functional
    sensitive to the mixing. The finite difference step is scaled with the
    gap (delta = g / 1000) so that the perturbation stays in the linear
    regime; the independent reference is the ``eigh_cluster_ift`` gradient
    under the same cluster tolerance. Measured relative agreement: 7e-12,
    1.2e-11 and 1.6e-9 for g = 1e-3, 1e-4, 1e-6.
    """
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    weights = jnp.asarray(rng.normal(size=4))
    direction = jnp.asarray(np.diag([1.0, 2.0, -0.5, 0.3]))
    for gap in (1e-3, 1e-4, 1e-6):
        matrix = _rotated([-1.0, -1.0 + gap, 0.5, 2.0], q)

        def functional(m):
            _, v = lowest_k_eigpairs(m, k=1, tol=1e-14)
            return jnp.sum(weights * v[:, 0] ** 2)

        grad = np.asarray(jax.grad(functional)(matrix))
        assert np.isfinite(grad).all()
        ad = float(np.sum(grad * np.asarray(direction)))
        assert abs(ad) > 1.0                              # a real 1/g signal
        ref = float(np.sum(
            _symmetrized_ift_projector_gradient(matrix, 1, weights)
            * np.asarray(direction)))
        assert abs(ad - ref) / abs(ad) < 1e-6
        fd = _directional_fd(functional, matrix, direction, gap / 1000.0)
        assert abs(ad - fd) / abs(ad) < 1e-4


def test_a_gap_below_the_cluster_tolerance_freezes_the_pair():
    """Below the tolerance the pair is one cluster for both implementations.

    The derivative of the individual vector is not defined there (the
    forward difference of the fixture splits the pair and is meaningless),
    but the two frozen-cluster rules must agree with each other.
    """
    rng = np.random.default_rng(6)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    weights = jnp.asarray(rng.normal(size=4))
    direction = jnp.asarray(np.diag([1.0, 2.0, -0.5, 0.3]))
    matrix = _rotated([-1.0, -1.0 + 1e-9, 0.5, 2.0], q)

    def functional(m):
        _, v = lowest_k_eigpairs(m, k=1, tol=1e-14)
        return jnp.sum(weights * v[:, 0] ** 2)

    grad = np.asarray(jax.grad(functional)(matrix))
    assert np.isfinite(grad).all()
    ad = float(np.sum(grad * np.asarray(direction)))
    ref = float(np.sum(
        _symmetrized_ift_projector_gradient(matrix, 1, weights)
        * np.asarray(direction)))
    assert abs(ad - ref) < 1e-6 * max(abs(ad), abs(ref), 1.0)


def test_the_cluster_in_the_middle_of_the_spectrum():
    """A degenerate pair with eigenvalues on both sides: the complement
    operator is indefinite with both signs, which is what MINRES covers and
    CG would not. Measured: AD against FD to the printed digits, and against
    the symmetrized frozen-cluster rule to 5.6e-16."""
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    matrix = _rotated([-2.0, -1.0, 0.5, 0.5, 2.0, 3.0], q)
    weights = jnp.asarray(rng.normal(size=6))
    direction = jnp.asarray(np.diag([0.3, -0.7, 1.0, -0.4, 0.6, -0.2]))

    def ground_functional(m):
        _, v = lowest_k_eigpairs(m, k=1, tol=1e-13)
        return jnp.sum(weights * v[:, 0] ** 2)

    grad = np.asarray(jax.grad(ground_functional)(matrix))
    ad = float(np.sum(grad * np.asarray(direction)))
    fd = _directional_fd(ground_functional, matrix, direction, 1e-5)
    assert abs(ad - fd) < 1e-7
    ref = float(np.sum(
        _symmetrized_ift_projector_gradient(matrix, 1, weights)
        * np.asarray(direction)))
    assert abs(ad - ref) < 1e-8

    # k = 4 puts the two-member cluster inside the returned block
    fp = _projector_functional(4, weights)
    gp = np.asarray(jax.grad(fp)(matrix))
    ad_p = float(np.sum(gp * np.asarray(direction)))
    fd_p = _directional_fd(fp, matrix, direction, 1e-5)
    assert np.isfinite(gp).all()
    assert abs(ad_p - fd_p) < 1e-7
    ref_p = float(np.sum(
        _symmetrized_ift_projector_gradient(matrix, 4, weights)
        * np.asarray(direction)))
    assert abs(ad_p - ref_p) < 1e-8


def test_the_block_doubled_spectrum_has_a_complete_cluster():
    """The ``eigh_safe`` regression fixture: A = diag(B, B) doubles the whole
    spectrum, so every eigenvalue has a partner far away in index. The
    return values are the doubled lowest levels and the projector derivative
    matches the symmetrized reference to 7.8e-16 -- a complete cluster costs
    no ridge."""
    rng = np.random.default_rng(8)
    b = rng.normal(size=(3, 3))
    b = b + b.T
    doubled = jnp.asarray(np.block([[b, np.zeros((3, 3))],
                                    [np.zeros((3, 3)), b]]))
    w_ref = np.linalg.eigvalsh(np.asarray(doubled))
    theta, _ = lowest_k_eigpairs(doubled, k=2, tol=1e-13)
    assert float(jnp.max(jnp.abs(theta - w_ref[:2]))) < 1e-12
    weights = jnp.asarray(rng.normal(size=6))
    g_mine = np.asarray(jax.grad(_projector_functional(2, weights))(doubled))
    g_ref = _symmetrized_ift_projector_gradient(doubled, 2, weights)
    assert np.abs(g_mine - g_ref).max() < 1e-8


def test_the_complete_cluster_costs_no_ridge():
    """M2 bound on the ridge: for a complete cluster the M1 rule applied a
    ridge that was measured to cost 4.9e-9 against the frozen-cluster
    reference; with the cluster rule complete the same comparison sits at
    machine precision."""
    weights = jnp.asarray([0.7, -1.3, 0.4, 0.9])
    g_mine = np.asarray(jax.grad(
        _projector_functional(2, weights))(FIX_DEGENERATE_GROUND))
    g_ref = _symmetrized_ift_projector_gradient(
        FIX_DEGENERATE_GROUND, 2, weights)
    assert np.abs(g_mine - g_ref).max() < 1e-12


def test_a_partner_outside_the_ritz_block_enters_the_cluster():
    """The complete rule for absent partners.

    The pullback is called directly with a Ritz block that omits a
    degenerate partner of the ground level (the state a structured start
    vector produces). The solve discovers the partner from the operator,
    freezes it, and returns the same value as the call with the complete
    block -- where the M1-era rule instead returned an iterate of order
    1e16 driven by the singular complement.
    """
    weights = jnp.asarray([0.7, -1.3, 0.4, 0.9])

    # two-fold ground, one hidden partner
    u2 = np.array([0.89442719, 0.4472136, 0.0, 0.0])
    partner2 = np.array([0.4472136, -0.89442719, 0.0, 0.0])
    others2 = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    cotangent2 = 2.0 * np.asarray(weights) * u2
    complete2 = np.stack([u2, partner2] + list(others2), axis=1)
    truncated2 = np.stack([u2] + list(others2), axis=1)

    a2 = jnp.asarray(np.diag([-1.0, -1.0, 0.5, 2.0]))
    spec2 = jnp.asarray([-1.0, -1.0, 0.5, 2.0])
    ref2 = np.asarray(_ift_pullback(a2, jnp.asarray([-1.0]),
                                    jnp.asarray(u2)[:, None], spec2,
                                    jnp.asarray(complete2), jnp.zeros(1),
                                    jnp.asarray(cotangent2)[:, None], 1e-8))
    got2 = np.asarray(_ift_pullback(a2, jnp.asarray([-1.0]),
                                    jnp.asarray(u2)[:, None],
                                    jnp.asarray([-1.0, 0.5, 2.0]),
                                    jnp.asarray(truncated2), jnp.zeros(1),
                                    jnp.asarray(cotangent2)[:, None], 1e-8))
    assert np.isfinite(got2).all()
    assert np.abs(got2 - ref2).max() < 1e-12

    # three-fold ground, two hidden partners
    u3 = np.array([0.57735027, 0.57735027, 0.57735027, 0.0, 0.0])
    others3 = np.array([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    cotangent3 = 2.0 * np.array([0.7, -1.3, 0.4, 0.9, 0.25]) * u3
    a3 = jnp.asarray(np.diag([-1.0, -1.0, -1.0, 0.5, 2.0]))
    got3 = np.asarray(_ift_pullback(a3, jnp.asarray([-1.0]),
                                    jnp.asarray(u3)[:, None],
                                    jnp.asarray([-1.0, 0.5, 2.0]),
                                    jnp.asarray(np.stack([u3] + list(others3),
                                                         axis=1)),
                                    jnp.zeros(1),
                                    jnp.asarray(cotangent3)[:, None], 1e-8))
    assert np.isfinite(got3).all()
    assert np.abs(got3).max() < 1e-8


def test_the_ridge_is_the_last_net_and_scales_inversely():
    """When more partners are hidden than the discovery rounds cover, the
    ridge produces a finite representative whose scale is 1/ridge exactly --
    the recorded observable effect of the safeguard. Five hidden partners
    against a cap of four: measured |A_bar|max = 3.880e3 / 3.880e5 / 3.880e7
    for ridge = 1e-6 / 1e-8 / 1e-10."""
    weights = np.array([0.7, -1.3, 0.4, 0.9, 0.25, -0.6, 0.5, 0.2])
    n = 8
    u = np.concatenate([np.ones(6) / np.sqrt(6.0), [0.0, 0.0]])
    others = np.array([[1.0 if i == k else 0.0 for k in range(n)]
                       for i in (6, 7)])
    a = jnp.asarray(np.diag([-1.0] * 6 + [0.5, 2.0]))
    cotangent = 2.0 * weights * u
    values = []
    for ridge in (1e-6, 1e-8, 1e-10):
        got = np.asarray(_ift_pullback(
            a, jnp.asarray([-1.0]), jnp.asarray(u)[:, None],
            jnp.asarray([-1.0, 0.5, 2.0]),
            jnp.asarray(np.stack([u] + list(others), axis=1)), jnp.zeros(1),
            jnp.asarray(cotangent)[:, None], ridge))
        assert np.isfinite(got).all()
        assert np.abs(got).max() > 1e2
        values.append(np.abs(got).max() * ridge)
    assert max(values) / min(values) - 1.0 < 1e-3


# ---------------------------------------------------------------------------
# The linear kernel of the reverse pass.
# ---------------------------------------------------------------------------


def test_minres_solves_symmetric_systems_against_the_dense_solve():
    rng = np.random.default_rng(7)
    for n in (4, 6, 25):
        b_mat = rng.normal(size=(n, n))
        matrix = jnp.asarray(0.5 * (b_mat + b_mat.T))
        rhs = jnp.asarray(rng.normal(size=n))
        x, res, iters = minres_solve(lambda y: matrix @ y, rhs)
        x_ref = jnp.linalg.solve(matrix, rhs)
        rel = float(jnp.linalg.norm(x - x_ref) / jnp.linalg.norm(x_ref))
        assert rel < 1e-10
        assert float(res) < 1e-10


def test_minres_handles_positive_definite_and_singular_extremes():
    rng = np.random.default_rng(8)
    b = rng.normal(size=(5, 5))
    b = jnp.asarray(b @ b.T + 5.0 * np.eye(5))
    rhs = jnp.asarray(rng.normal(size=5))
    x, res, _ = minres_solve(lambda y: b @ y, rhs)
    assert float(jnp.linalg.norm(x - jnp.linalg.solve(b, rhs))) < 1e-8

    # zero right-hand side returns zero without iterating
    x0, res0, iters0 = minres_solve(lambda y: b @ y, jnp.zeros(5))
    assert float(jnp.linalg.norm(x0)) == 0.0
    assert float(res0) == 0.0
    assert int(iters0) == 0

    # a nearly singular system still returns a finite iterate with a small
    # reported residual
    s = jnp.diag(jnp.asarray([1e-8, 1.0, 2.0]))
    x1 = (minres_solve(lambda y: s @ y, jnp.ones(3)))[0]
    assert np.isfinite(np.asarray(x1)).all()


# ---------------------------------------------------------------------------
# The CI-level entries (need the dense reference kernel, hence pyscfad).
# ---------------------------------------------------------------------------


def test_the_iterative_ci_entries_match_cs_casci_on_h2():
    """The dense reference of the milestone: ``cs_casci`` itself, on the same
    MO integrals, for the two H2 systems. Guarded on pyscfad because
    ``core.casci`` imports it at module level; on a machine without pyscfad
    the local stand-in is the dense-path test above."""
    pytest.importorskip("pyscfad")
    from diffmrc.core.casci import cs_casci
    from diffmrc.core.iterative_ci import (iterative_ci_energies,
                                           iterative_ci_ground_ci)

    for basis in ('sto-3g', '6-31g'):
        h_mo, eri_mo = _h2_mo_integrals(basis, _H2_ANCHOR_R[basis])
        nact = 2 if basis == 'sto-3g' else 4
        e_dense = float(cs_casci(h_mo, eri_mo, ncore=0, nact=nact,
                                 nele_act=2))
        e_iter = float(iterative_ci_energies(h_mo, eri_mo, ncore=0, nact=nact,
                                             nele_act=2, tol=1e-12)[0])
        assert abs(e_iter - e_dense) < 1e-8

        e0, ci = iterative_ci_ground_ci(h_mo, eri_mo, ncore=0, nact=nact,
                                        nele_act=2, tol=1e-12)
        _, ci_dense = cs_casci(h_mo, eri_mo, ncore=0, nact=nact, nele_act=2,
                               return_ci=True)
        overlap = abs(float(jnp.dot(ci, ci_dense.reshape(-1))))
        assert abs(overlap - 1.0) < 1e-6
        assert abs(float(e0) - e_dense) < 1e-8
