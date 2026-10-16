import numpy as np
import pytest
from itertools import combinations

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.ci_matvec import (block_determinant_count, ci_block_operator,
                                    iterative_ci_energies_mf,
                                    lowest_k_eigpairs_mf,
                                    string_tables)  # noqa: E402
from tests.core.test_iterative_ci import (_h2_mo_integrals, _ci_matrix,
                                          _H2_ANCHOR_R)  # noqa: E402


def _sym8(a):
    """The eight-fold chemist symmetries of a two-electron tensor.

    Physical MO integrals carry (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq); a
    random tensor fed to the comparison must carry them too, or the two
    implementations legitimately disagree (the first synthetic fixture here
    was not symmetrized and produced exactly that false alarm).
    """
    terms = [a]
    for g in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        terms = terms + [t.transpose(g) for t in terms]
    return sum(terms) / 8.0


def _block_reference(h_mo, eri_mo, ncore, nact, nele_act, n_alpha, n_beta):
    """The dense S_z block of the independent builder, in the grid order.

    The block matrix is materialized by the brute-force second-quantized
    builder of the iterative-CI test module (itself pinned against the
    project's PySCF reference energies) and reordered into the
    (alpha-string, beta-string) determinant order of
    :func:`ci_block_operator`. The comparison is exact -- the matrix-free
    kernel converts its internal alpha-first sign convention through the
    per-determinant phase -- so the test needs no sign matching.
    """
    full = _ci_matrix(np.asarray(h_mo), np.asarray(eri_mo), ncore, nact,
                      nele_act)
    core = tuple(range(2 * ncore))
    dets = [frozenset(core + c) for c in
            combinations(range(2 * ncore, 2 * (ncore + nact)), nele_act)]

    def counts(d):
        n_a = sum(1 for so in d if so % 2 == 0)
        return n_a - ncore, len(d) - n_a - ncore

    idx = [i for i, d in enumerate(dets) if counts(d) == (n_alpha, n_beta)]
    block = full[np.ix_(idx, idx)]
    keys = [tuple(sorted(dets[i])) for i in idx]
    tab = string_tables(nact, n_alpha, n_beta)
    masks_a = np.asarray(tab['masks_a'])
    masks_b = np.asarray(tab['masks_b'])

    def key(ma, mb):
        return tuple(sorted(
            list(range(2 * ncore)) +
            [2 * ncore + 2 * i for i in range(nact) if (int(ma) >> i) & 1] +
            [2 * ncore + 2 * j + 1 for j in range(nact) if (int(mb) >> j) & 1]))

    order = np.array([keys.index(key(masks_a[I], masks_b[J]))
                      for I in range(len(masks_a))
                      for J in range(len(masks_b))])
    # grid index k carries the block determinant order[k], so the grid matrix
    # equals the block matrix re-indexed by order.
    return block[np.ix_(order, order)]


def _matvec_matrix(matvec, n):
    return np.column_stack([np.asarray(matvec(jnp.eye(n)[:, k]))
                            for k in range(n)])


def _block_error(h_mo, eri_mo, ncore, nact, n_alpha, n_beta):
    tab = string_tables(nact, n_alpha, n_beta)
    matvec, _ = ci_block_operator(h_mo, eri_mo, ncore, nact, n_alpha,
                                  n_beta, tab)
    n = tab['nI'] * tab['nJ']
    mat = _matvec_matrix(matvec, n)
    assert np.abs(mat - mat.T).max() < 1e-12
    ref = _block_reference(h_mo, eri_mo, ncore, nact, n_alpha + n_beta,
                           n_alpha, n_beta)
    return np.abs(mat - ref).max()


def test_the_operator_matches_the_independent_builder_on_h2():
    """Every block of H2/6-31G against the independent dense builder.

    The CAS(4,2) space covers all excitation classes of this active space
    at once (alpha/beta singles, same-spin doubles, opposite-spin doubles);
    blocks from zero to four electrons of one spin are compared.
    """
    h_mo, eri_mo = _h2_mo_integrals('6-31g', _H2_ANCHOR_R['6-31g'])
    for n_alpha, n_beta in ((1, 1), (2, 0), (0, 2), (2, 2), (3, 1), (4, 0)):
        assert _block_error(h_mo, eri_mo, 0, 4, n_alpha, n_beta) < 1e-12 \
            + 0.0, (n_alpha, n_beta)


def test_the_operator_matches_the_independent_builder_on_synthetic_integrals():
    """Random chemist-symmetric integrals, blocks (0,2), (1,1), (2,2)."""
    rng = np.random.default_rng(3)
    nact = 6
    b = rng.normal(size=(nact, nact))
    h_mo = jnp.asarray(0.5 * (b + b.T))
    eri_mo = jnp.asarray(_sym8(rng.normal(size=(nact,) * 4)))
    for n_alpha, n_beta in ((0, 2), (1, 1), (2, 2)):
        assert _block_error(h_mo, eri_mo, 0, nact, n_alpha,
                            n_beta) < 1e-12, (n_alpha, n_beta)


def test_the_core_folding_matches_the_independent_builder():
    """Frozen-core folding against the independent dense builder.

    With ``ncore > 0`` the kernel folds the closed shell into an effective
    one-electron operator (2 J - K per core orbital) plus a constant, while
    the dense reference carries the core explicitly in every determinant.
    The two must agree elementwise; the test also pins the constant's
    strange-looking ordered-pair form, whose naive (cc|cc) - (cc'|c'c)
    version cancels the same-orbital Coulomb term.
    """
    rng = np.random.default_rng(7)
    for ncore, nact, nele, n_alpha, n_beta in ((1, 4, 2, 1, 1),
                                               (2, 4, 2, 1, 1),
                                               (2, 5, 4, 2, 2),
                                               (3, 6, 6, 3, 3)):
        nmo = ncore + nact
        b = rng.normal(size=(nmo, nmo))
        h_mo = jnp.asarray(0.5 * (b + b.T))
        eri_mo = jnp.asarray(_sym8(rng.normal(size=(nmo,) * 4)))
        assert _block_error(h_mo, eri_mo, ncore, nact, n_alpha,
                            n_beta) < 1e-12, (ncore, nact)


def test_the_matrix_free_energies_match_the_dense_block():
    """Davidson on the matrix-free operator against the dense block eigh."""
    h_mo, eri_mo = _h2_mo_integrals('6-31g', _H2_ANCHOR_R['6-31g'])
    matvec, _ = ci_block_operator(h_mo, eri_mo, 0, 4, 2, 2)
    mat = _matvec_matrix(matvec, 36)
    w_ref = np.linalg.eigvalsh(mat)[:2]
    ws, vs = lowest_k_eigpairs_mf(h_mo, eri_mo, ncore=0, nact=4, n_alpha=2,
                                  n_beta=2, k=2, tol=1e-12)
    assert np.abs(np.asarray(ws) - w_ref).max() < 1e-10
    assert abs(float(vs[:, 0] @ vs[:, 0]) - 1.0) < 1e-12

    ws2 = iterative_ci_energies_mf(h_mo, eri_mo, ncore=0, nact=4, nele_act=4,
                                   k=1, tol=1e-12)
    assert abs(float(ws2[0]) - w_ref[0]) < 1e-10


def test_the_matrix_free_gradient_matches_finite_differences():
    """AD through the matrix-free chain against a central difference.

    The VJP with respect to the integrals is the gradient of
    ``xi @ H(h, e) u`` through the same matrix-free application, so the
    dense matrix never exists on either side of the chain. Measured
    relative agreement at delta = 1e-6: 2.4e-12.
    """
    h_mo, eri_mo = _h2_mo_integrals('6-31g', _H2_ANCHOR_R['6-31g'])
    rng = np.random.default_rng(5)
    dh = jnp.asarray(rng.normal(size=h_mo.shape))
    dh = 0.5 * (dh + dh.T)
    de = jnp.asarray(_sym8(rng.normal(size=eri_mo.shape)))

    def energy(h, e):
        return lowest_k_eigpairs_mf(h, e, ncore=0, nact=4, n_alpha=2,
                                    n_beta=2, k=1, tol=1e-13)[0][0]

    g_h, g_e = jax.grad(energy, argnums=(0, 1))(h_mo, eri_mo)
    assert np.isfinite(np.asarray(g_h)).all()
    assert np.isfinite(np.asarray(g_e)).all()
    ad = float(jnp.sum(g_h * dh) + jnp.sum(g_e * de))
    step = 1e-6
    fd = (float(energy(h_mo + step * dh, eri_mo + step * de))
          - float(energy(h_mo - step * dh, eri_mo - step * de))) / (2 * step)
    assert abs(ad - fd) < 1e-8 * abs(fd)


def test_the_block_dimensions_and_the_empty_spin_sector():
    assert block_determinant_count(6, 3, 3) == 400
    assert block_determinant_count(12, 6, 6) == 853776
    rng = np.random.default_rng(4)
    nact = 4
    b = rng.normal(size=(nact, nact))
    h_mo = jnp.asarray(0.5 * (b + b.T))
    eri_mo = jnp.asarray(_sym8(rng.normal(size=(nact,) * 4)))
    assert _block_error(h_mo, eri_mo, 0, nact, 0, 2) < 1e-12
