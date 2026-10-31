"""The spectrum-driven cluster-tolerance rule and its routing helper.

The measured band this rule encodes is recorded in
``SESSION_LOG_APPEND_20260923b.md`` section 21 and in the module docstring
of ``core/eigh.py``: at the CO/6-31G and N2/cc-pVDZ stationary points the
symmetry pairs that are bit-exact at theta = 0 split to 1.9e-8 .. 1.155e-5
by floating-point noise, while the smallest physical gaps of the same
spectra are 1.9e-2 and above. A tolerance inside that band is admissible;
below it the derivative takes a bare 1/gap resolvent, above it distinct
states are clustered.

These tests use the spectra recorded by the t280 probe (quoted in the
session log) and synthetic spectra, so they run without pyscfad.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.eigh import (cluster_tolerance, eigh_safe,
                               lowest_pair_auto)  # noqa: E402

# t280, CO/6-31G at theta*: the active 1-RDM spectrum.
_DM1_SPECTRUM = [0.01974, 0.04643, 0.04643, 1.95352, 1.95352, 1.98036]
# t280, CO/6-31G at theta*: the consecutive gaps of the Fock virtual block,
# with the two numerically split pairs (3.358e-6, 1.155e-5) and the three
# physical gaps (1.9e-2, 7.7e-2, 2.2e-1).
_FOCK_VIRTUAL_GAPS = [3.358e-6, 1.155e-5, 1.9e-2, 7.7e-2, 2.2e-1]


def _synthetic_from_gaps(gaps, x0=0.0):
    return np.concatenate([[x0], x0 + np.cumsum(gaps)])


def test_the_recorded_dm1_spectrum_gives_an_admissible_tolerance():
    dec = cluster_tolerance(_DM1_SPECTRUM)
    assert dec.mode == 'cluster'
    # The two noise gaps are the 4.616e-8 and 1.856e-8 entries; the
    # smallest physical gap is 2.669e-2.
    assert dec.n_noise_gaps == 2
    assert dec.physical_min == pytest.approx(2.669e-2, rel=1e-3)
    # The default relative floor on this scale is 5.9e-5, above the measured
    # split edge 1.155e-5 and well below the physical gap.
    assert 1e-7 < dec.noise_floor < 1e-3
    # The measured admissible window from t286 is 1e-5 .. 1e-2.
    assert 1e-5 < dec.tol < 1e-2
    # The tolerance must sit strictly between the two measured scales.
    assert dec.noise_floor < dec.tol < dec.physical_min


def test_the_recorded_fock_gaps_give_an_admissible_tolerance():
    spectrum = _synthetic_from_gaps(_FOCK_VIRTUAL_GAPS)
    # The absolute noise scale measured on this block; supplying the
    # measured value is the supported route when it is known.
    dec = cluster_tolerance(spectrum, noise_scale=1.155e-5)
    assert dec.mode == 'cluster'
    assert dec.n_noise_gaps == 2
    assert dec.physical_min == pytest.approx(1.9e-2, rel=1e-12)
    assert dec.noise_floor < dec.tol < dec.physical_min
    assert 1e-5 < dec.tol < 1e-2
    # The default relative floor also classifies this block correctly when
    # the spectrum is on a unit scale.
    dec_default = cluster_tolerance(spectrum)
    assert dec_default.mode == 'cluster'
    assert dec_default.noise_floor < dec_default.tol < dec_default.physical_min


def test_a_separated_spectrum_without_noise_needs_no_freeze():
    dec = cluster_tolerance([0.0, 0.5, 1.2, 3.0])
    assert dec.mode == 'none'
    assert dec.tol == 0.0
    assert dec.n_noise_gaps == 0


def test_an_exact_degeneracy_is_clustered_at_the_noise_floor():
    dec = cluster_tolerance([-1.0, -1.0, 0.5, 2.0])
    assert dec.mode == 'cluster'
    assert dec.n_noise_gaps == 1
    assert dec.tol > dec.noise_floor
    # The physical gap 1.5 must stay outside the cluster.
    assert dec.tol < 1.5


def test_no_admissible_tolerance_falls_back_to_lanczos():
    # The smallest physical gap sits only a factor of 3 above the noise
    # floor: no tolerance separates the two scales (README's fallback case).
    spectrum = _synthetic_from_gaps([3e-5, 9e-5, 1.9e-2, 1.0])
    dec = cluster_tolerance(spectrum, noise_scale=3e-5)
    assert dec.mode == 'lanczos'
    assert dec.tol is None
    assert dec.ratio < 100.0


def test_a_physical_gap_below_the_floor_is_the_documented_open_case():
    """The rule cannot distinguish it from noise; this pins the limitation.

    The README states that a physical gap inside the noise band makes the
    tolerance rule unsolvable from the spectrum alone. The rule then
    clusters it (rather than silently exposing it); the documented remedy
    is to supply a measured ``noise_scale`` small enough to separate the
    two, or to route the pair through the Lanczos kernel.
    """
    spectrum = _synthetic_from_gaps([1e-8, 3e-6, 1e-6, 1e-2])
    dec = cluster_tolerance(spectrum, noise_scale=3e-5)
    assert dec.mode == 'cluster'
    assert dec.n_noise_gaps == 3          # the 1e-6 physical gap is inside
    # The honest statement: the spectrum alone cannot flag it. What the
    # rule does guarantee is that the chosen threshold stays above the
    # supplied noise scale and below the first gap above it.
    assert dec.noise_floor < dec.tol < dec.physical_min


def _rotated_pair_fixture(split):
    """Two-dimensional rotated spectrum with a numerical pair split."""
    q = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    w = np.diag([0.5, 0.5 + split])
    return jnp.asarray(q @ w @ q.T)


def test_lowest_pair_auto_routes_and_returns_the_eigenpair():
    a = jnp.asarray(np.diag([-1.0, 0.4, 0.7, 2.0]))
    lam, u, dec = lowest_pair_auto(a)
    assert dec.mode == 'none'
    assert abs(float(lam) + 1.0) < 1e-10
    assert np.isfinite(np.asarray(u)).all()


def test_lowest_pair_auto_falls_back_to_lanczos_when_ambiguous():
    # Gaps [3e-5, 9e-5, ...] with an explicit floor at 3e-5 (ratio 3).
    a = jnp.diag(jnp.asarray([0.0, 3e-5, 1.2e-4, 1.9e-2, 1.0]))
    lam, u, dec = lowest_pair_auto(a, noise_scale=3e-5)
    assert dec.mode == 'lanczos'
    w_ref = np.linalg.eigvalsh(np.asarray(a))
    assert abs(float(lam) - w_ref[0]) < 1e-10
    assert np.isfinite(np.asarray(u)).all()


def test_lowest_pair_auto_agrees_with_eigh_safe_without_degeneracy():
    rng = np.random.default_rng(21)
    b = rng.normal(size=(6, 6))
    a = jnp.asarray(b + b.T + np.diag(np.arange(6.0)))
    lam, u, _ = lowest_pair_auto(a)
    w, U = eigh_safe(a)
    assert abs(float(lam) - float(w[0])) < 1e-10
    assert abs(abs(float(u @ U[:, 0])) - 1.0) < 1e-8


def test_lowest_pair_auto_cluster_route_returns_the_ground_pair():
    """The ``cluster`` branch routes through ``eigh_cluster_ift``."""
    w = _synthetic_from_gaps([1e-8, 4e-8, 2.6e-2, 1.0])
    a = jnp.diag(jnp.asarray(w))
    lam, u, dec = lowest_pair_auto(a)
    assert dec.mode == 'cluster'
    assert dec.noise_floor < dec.tol < dec.physical_min
    assert abs(float(lam) - w[0]) < 1e-10
    assert abs(float(u @ u) - 1.0) < 1e-10
