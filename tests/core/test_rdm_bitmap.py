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


def test_large_nso_no_overflow():
    # ncore=100 builds bitmaps of 208 bits; the raw integers exceed int64
    # (asserted below), and before the fix make_dm123_sf raised
    # OverflowError inside jnp.asarray.
    dets = cas_dets_for2(100, 4, 4)
    assert max(int(d) for d in dets) > (1 << 63)
    ci = _norm_ci(len(dets))
    dm1, _, _ = make_dm123_sf(ci, dets, 4, 100)
    assert abs(float(jnp.trace(dm1)) - 4.0) < 1e-10
