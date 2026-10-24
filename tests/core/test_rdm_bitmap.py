"""Regression tests for the active-relative RDM bitmap (fix 2026-10-10).

The determinant bitmaps from ``cas_dets_for2`` span the full SO space; the
old code fed them straight to ``jnp.asarray``, which overflows int64 once
nso grows (LaF/cc-pVTZ-DK: 350 bits -> OverflowError).  The fix shifts the
fixed core bits out before the conversion; these tests pin both ends: the
physical invariants hold on a small system, and a large-nso construction
no longer raises.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.nevpt2.impl import cas_dets_for2
from diffmrc.core.nevpt2.rdm import make_dm123_sf


def _norm_ci(ndet, seed=7):
    rng = np.random.default_rng(seed)
    ci = jnp.asarray(rng.standard_normal(ndet))
    return ci / jnp.linalg.norm(ci)


def test_dm1_trace_equals_active_electrons():
    dets = cas_dets_for2(4, 4, 4)
    ci = _norm_ci(len(dets))
    dm1, _, _ = make_dm123_sf(ci, dets, 4, 4)
    assert abs(float(jnp.trace(dm1)) - 4.0) < 1e-10


def test_max_rank_skips_higher_rdms_and_preserves_dm1():
    """canonicalize_jax consumes only dm1.

    The dm3 contraction of make_dm123_sf scales as ncas**6 * ndet**4 and
    dominates the canonicalization at CAS(6,6).  max_rank=1 must skip it
    while leaving dm1 bit-identical; max_rank=2 keeps dm2 and skips dm3.
    """
    dets = cas_dets_for2(0, 2, 2)
    ci = _norm_ci(len(dets))
    dm1_full, dm2_full, dm3_full = make_dm123_sf(ci, dets, 2, 0)
    dm1_one, dm2_one, dm3_one = make_dm123_sf(ci, dets, 2, 0, max_rank=1)
    dm1_two, dm2_two, dm3_two = make_dm123_sf(ci, dets, 2, 0, max_rank=2)
    assert np.array_equal(np.asarray(dm1_full), np.asarray(dm1_one))
    assert np.array_equal(np.asarray(dm1_full), np.asarray(dm1_two))
    assert dm2_one is None and dm3_one is None
    assert dm2_two is not None and dm3_two is None
    assert np.allclose(np.asarray(dm2_full), np.asarray(dm2_two), atol=0,
                       rtol=0)


def test_large_nso_no_overflow():
    # ncore=100 builds bitmaps of 208 bits; the raw integers exceed int64
    # (asserted below), and before the fix make_dm123_sf raised
    # OverflowError inside jnp.asarray.
    dets = cas_dets_for2(100, 4, 4)
    assert max(int(d) for d in dets) > (1 << 63)
    ci = _norm_ci(len(dets))
    dm1, _, _ = make_dm123_sf(ci, dets, 4, 100)
    assert abs(float(jnp.trace(dm1)) - 4.0) < 1e-10
