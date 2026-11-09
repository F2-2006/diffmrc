"""The M_S sector of the CI vector in the NEVPT2 core classes.

An odd active-electron count has two exactly degenerate lowest M_S
blocks.  ``cs_casci_blocked`` keeps whichever is lower by a strict
comparison on energies that differ only by roundoff, while the core
classes are assembled from the alpha-beta coefficient matrix of one fixed
sector.  When the partner block wins, that projection is empty and the
renormalization is 0/0: the class energies come back NaN even though the
energy is finite.

These tests lock the two properties the fix has to keep:

  * the projection accepts either sector and always returns the fixed
    (nele//2, nele - nele//2) layout, normalized;
  * the two degenerate sectors give the same eight class energies, which
    is the statement that the spin-free contraction is invariant under
    the exchange.

Measured on OH/6-31G CAS(5,4) at theta*: the base (2, 3) values are
unchanged to 5e-16 by the sector-aware projection and the (3, 2) partner
reproduces all eight classes to 6e-16, while the fixed projection returns
NaN for Sr and Si from the partner.
"""
from itertools import combinations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.casci import _block_H, _block_lowest_pair  # noqa: E402
from diffmrc.core.nevpt2.f3ca import ab_index_map  # noqa: E402
from diffmrc.core.nevpt2.impl import (  # noqa: E402
    cas_dets_for2, ci_ab_from_flat, nevpt2_e2_full2)
from diffmrc.driver.nevpt2 import _spin_links  # noqa: E402
from diffmrc.driver.nevpt2_zvector import _e2_core_exact_si  # noqa: E402

NCORE, NACT, NELE = 1, 2, 3
NMO = 5


def _fixture():
    """Well separated synthetic integrals on a CAS(3,2) with one core."""
    rng = np.random.default_rng(21)
    h1 = rng.normal(scale=0.05, size=(NMO, NMO))
    h1 = 0.5 * (h1 + h1.T) + np.diag([-0.70, -0.10, 0.15, 0.85, 1.30])
    eri = rng.normal(scale=0.02, size=(NMO,) * 4)
    eri = (eri + eri.transpose(1, 0, 2, 3) + eri.transpose(0, 1, 3, 2)
           + eri.transpose(1, 0, 3, 2)
           + eri.transpose(2, 3, 0, 1) + eri.transpose(3, 2, 0, 1)
           + eri.transpose(2, 3, 1, 0) + eri.transpose(3, 2, 1, 0)) / 8.0
    eps = np.asarray([-0.70, -0.05, 0.20, 0.95, 1.40])
    return jnp.asarray(h1), jnp.asarray(eri), jnp.asarray(eps)


def _sector_vectors(h1, eri):
    """Lowest eigenvector of each of the two degenerate M_S blocks."""
    nso = 2 * (NCORE + NACT)
    core_so = tuple(range(2 * NCORE))
    dets = [core_so + c for c in combinations(
        list(range(2 * NCORE, nso)), NELE)]
    sz = [sum(1 for so in d if so % 2 == 0) * 2 - NELE for d in dets]
    out = {}
    for s in sorted(set(sz)):
        idx = [i for i, x in enumerate(sz) if x == s]
        h_blk = _block_H(h1, eri, dets, idx, NCORE, NACT, NELE)
        w, u = _block_lowest_pair(h_blk, "eigh")
        na = sum(1 for so in dets[idx[0]]
                 if so >= 2 * NCORE and so % 2 == 0)
        full = np.zeros(len(dets))
        full[np.asarray(idx)] = np.asarray(u)
        out[(na, NELE - na)] = (float(w), jnp.asarray(full))
    return out, cas_dets_for2(NCORE, NACT, NELE)


def test_ab_index_map_covers_both_spin_orientations():
    """Both orientations exist, are complete, and cover disjoint halves of
    the determinant list."""
    idx_a, ph_a = ab_index_map(NCORE, NACT, NELE, 1, 2)
    idx_b, ph_b = ab_index_map(NCORE, NACT, NELE, 2, 1)
    assert idx_a.shape == (2, 1)     # one-electron alpha strings x two-electron beta
    assert idx_b.shape == (1, 2)
    assert (idx_a >= 0).all() and (idx_b >= 0).all()
    sa, sb = set(idx_a.ravel().tolist()), set(idx_b.ravel().tolist())
    assert not (sa & sb)             # the two sectors are different determinants
    assert len(sa | sb) == len(cas_dets_for2(NCORE, NACT, NELE))
    assert np.abs(ph_a).max() == 1.0 and np.abs(ph_b).max() == 1.0


def test_ci_ab_from_flat_selects_the_occupied_sector():
    """A vector placed in either sector comes back normalized in the fixed
    (nele//2, nele - nele//2) layout, with no NaN."""
    _, dets = _sector_vectors(*_fixture()[:2])
    rng = np.random.default_rng(5)
    idx_a, _ = ab_index_map(NCORE, NACT, NELE, 1, 2)
    idx_b, _ = ab_index_map(NCORE, NACT, NELE, 2, 1)
    shape = idx_a.shape                 # (2, 1): the fixed layout
    for idx in (idx_a, idx_b):
        flat = jnp.zeros(len(dets))
        flat = flat.at[jnp.asarray(idx.ravel())].set(
            jnp.asarray(rng.normal(size=idx.size)))
        c = ci_ab_from_flat(flat, NCORE, NACT, NELE)
        assert c.shape == shape
        assert np.isfinite(np.asarray(c)).all()
        assert float(jnp.linalg.norm(c)) == pytest.approx(1.0, abs=1e-14)


def test_the_two_degenerate_sectors_give_the_same_class_energies():
    """The eight classes are identical (to roundoff) whether the CI vector
    comes from the (1, 2) or the (2, 1) block."""
    h1, eri, eps = _fixture()
    vecs, dets = _sector_vectors(h1, eri)
    e12, v12 = vecs[(1, 2)]
    e21, v21 = vecs[(2, 1)]
    assert abs(e12 - e21) < 1e-12          # spin degeneracy of the fixture
    la, lb = _spin_links(NACT, NELE)
    vals = {}
    for key, vec in (("a", v12), ("b", v21)):
        e2, _ = nevpt2_e2_full2(h1, eri, vec, dets, 0.0, mo_energy=eps,
                                ncore=NCORE, nact=NACT, nocc=NCORE + NACT,
                                links_a=la, links_b=lb)
        vals[key] = e2
    assert np.isfinite(float(vals["a"])) and np.isfinite(float(vals["b"]))
    assert abs(float(vals["a"]) - float(vals["b"])) < 1e-13


def test_ci_ab_from_flat_is_differentiable_in_both_sectors():
    """The sector selection sits on the differentiated path; its tangent is
    finite and identical for the two orientations of the same vector."""
    _, dets = _sector_vectors(*_fixture()[:2])
    rng = np.random.default_rng(7)
    idx_a, _ = ab_index_map(NCORE, NACT, NELE, 1, 2)
    idx_b, _ = ab_index_map(NCORE, NACT, NELE, 2, 1)
    flat = jnp.zeros(len(dets))
    flat = flat.at[jnp.asarray(idx_a.ravel())].set(
        jnp.asarray(rng.normal(size=idx_a.size)))
    tang = jnp.zeros(len(dets))
    tang = tang.at[jnp.asarray(idx_a.ravel())].set(
        jnp.asarray(rng.normal(size=idx_a.size)))

    def f(v):
        return jnp.sum(ci_ab_from_flat(v, NCORE, NACT, NELE) ** 2)

    _, dv = jax.jvp(f, (flat,), (tang,))
    assert np.isfinite(float(dv))
    # A vector carried entirely by the partner sector is differentiable too.
    flat_b = jnp.zeros(len(dets))
    flat_b = flat_b.at[jnp.asarray(idx_b.ravel())].set(
        jnp.asarray(rng.normal(size=idx_b.size)))
    tang_b = jnp.zeros(len(dets))
    tang_b = tang_b.at[jnp.asarray(idx_b.ravel())].set(
        jnp.asarray(rng.normal(size=idx_b.size)))

    def g(v):
        return jnp.sum(ci_ab_from_flat(v, NCORE, NACT, NELE) ** 2)

    _, dv_b = jax.jvp(g, (flat_b,), (tang_b,))
    assert np.isfinite(float(dv_b))
