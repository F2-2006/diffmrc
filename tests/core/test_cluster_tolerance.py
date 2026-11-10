"""The spectrum-driven cluster-tolerance rule and its routing helper.

The measured bands this rule encodes are recorded in the module comment of
``core/eigh.py`` and in ``开发_OP_简并容差_20261110/报告.txt`` (2026-11-10):
on CO, N2, H2O and OH the split of an exactly degenerate pair under a
random gauge rotation is 1e-15 .. 5e-14 in float64 and 1e-7 .. 4e-7 in
float32, while the smallest physical gaps of the same blocks are 3.8e-4
(H2O dm1) and above. The 1e-8 .. 1.2e-5 band recorded earlier at the CO
theta* is the response to a residual orbital gradient, not floating point.

These tests use the recorded spectra, synthetic spectra and replicated
constructions; they run without pyscfad.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.eigh import (cluster_tolerance, eigh_cluster_ift,
                               eigh_safe, lowest_pair_auto,
                               separation_certificate)  # noqa: E402

# t280, CO/6-31G at theta*: the active 1-RDM spectrum.
_DM1_SPECTRUM = [0.01974, 0.04643, 0.04643, 1.95352, 1.95352, 1.98036]
# t280, CO/6-31G at theta*: the consecutive gaps of the Fock virtual block,
# with the two numerically split pairs (3.358e-6, 1.155e-5) and the three
# physical gaps (1.9e-2, 7.7e-2, 2.2e-1).
_FOCK_VIRTUAL_GAPS = [3.358e-6, 1.155e-5, 1.9e-2, 7.7e-2, 2.2e-1]


def _synthetic_from_gaps(gaps, x0=0.0):
    return np.concatenate([[x0], x0 + np.cumsum(gaps)])


def _rotated_matrix(spectrum, seed):
    rng = np.random.default_rng(seed)
    n = len(spectrum)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    a = q @ np.diag(np.asarray(spectrum, dtype=float)) @ q.T
    return jnp.asarray(0.5 * (a + a.T))


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
    # The relative floor is a prior, not a measurement.
    assert dec.certified is False


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
    assert dec.certified is True
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


def test_no_admissible_tolerance_returns_separation_unverified():
    # The smallest physical gap sits only a factor of 3 above the noise
    # floor: no tolerance separates the two scales.
    spectrum = _synthetic_from_gaps([3e-5, 9e-5, 1.9e-2, 1.0])
    dec = cluster_tolerance(spectrum, noise_scale=3e-5)
    assert dec.mode == 'separation_unverified'
    assert dec.tol is None
    assert dec.ratio < 30.0


def test_a_smallest_gap_above_the_band_is_not_certified():
    """The no-cluster hole of the previous rule.

    With no gap at or below the band, the old rule returned ``'none'`` and
    differentiated a gap only 10x the band with a bare 1/gap. The ratio
    test now applies to this branch too.
    """
    spectrum = np.array([0.0, 1e-6, 1.0])
    dec = cluster_tolerance(spectrum, noise_band=(1e-8, 1e-7))
    assert dec.mode == 'separation_unverified'
    assert dec.n_noise_gaps == 0
    assert dec.ratio == pytest.approx(10.0)
    # The same spectrum with the smallest gap far above the band is safe.
    dec_ok = cluster_tolerance(np.array([0.0, 1e-3, 1.0]),
                               noise_band=(1e-8, 1e-7))
    assert dec_ok.mode == 'none'


def test_a_measured_band_keeps_a_physical_gap_out_of_the_cluster():
    """A physical gap above the band must not be swallowed by the tol.

    With the relative prior the 2e-5 gap is classified as noise and the
    tolerance becomes 5.5e-4, above the physical gap. With the measured
    band the same gap is physical and the tolerance stays below it.
    """
    spectrum = _synthetic_from_gaps([1e-9, 2e-5, 1e-2])
    prior = cluster_tolerance(spectrum)
    assert prior.mode == 'cluster'
    assert prior.tol > 2e-5          # the old behaviour: the gap is frozen
    measured = cluster_tolerance(spectrum, noise_band=(1e-9, 1e-8))
    assert measured.mode == 'cluster'
    assert measured.tol < 2e-5       # certified: the gap stays live
    assert measured.physical_min == pytest.approx(2e-5, rel=1e-12)


def test_an_uncertified_band_cannot_cluster_the_whole_spectrum():
    spectrum = np.array([0.0, 1e-9, 3e-9, 1e-8])
    assert cluster_tolerance(spectrum).mode == 'separation_unverified'
    dec = cluster_tolerance(spectrum, noise_band=(1e-9, 1e-8))
    assert dec.mode == 'cluster_all'
    assert dec.certified is True
    assert dec.tol == pytest.approx(1e-8)


def test_certificate_separates_noise_from_physics():
    # Two exactly degenerate states (numerical split) and a physical gap
    # of 3e-6 that is invariant under the gauge replicate.
    base = [0.0, 0.0, 3e-6, 1.9e-2, 1.0]
    reps = np.stack([np.linalg.eigvalsh(np.asarray(_rotated_matrix(base, s)))
                     for s in range(8)])
    dec = separation_certificate(reps)
    assert dec.mode == 'cluster'
    assert dec.certified is True
    assert dec.n_noise_gaps == 1
    assert dec.physical_min == pytest.approx(3e-6, rel=1e-6)
    assert dec.noise_floor < 1e-12
    assert dec.tol < 3e-6
    # A single spectrum with the relative prior cannot see that the 3e-6
    # gap is physical; the certificate can.
    single = cluster_tolerance(np.sort(reps[0]))
    assert single.mode in ('cluster_all', 'cluster')


def test_certificate_flags_a_physical_gap_inside_the_band():
    # The 1e-6 gap is numerical (it moves by O(100%) across constructions);
    # the 3e-8 gap is stable physical but lies inside the numerical band.
    gaps = [0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6, 2.5e-6, 3.0e-6]
    rows = []
    for g in gaps:
        rows.append(np.sort([0.0, 3e-8, 0.5, 0.5 + g, 1.0]))
    dec = separation_certificate(np.stack(rows))
    assert dec.mode == 'separation_unverified'
    assert dec.tol is None
    assert dec.physical_min == pytest.approx(3e-8, rel=1e-6)
    assert dec.physical_min <= dec.noise_floor
    # The single-spectrum rule clusters the physical gap silently: the
    # 3e-8 gap is below every band it can justify from one spectrum, so the
    # tolerance is set by the next gap and swallows it.
    single = cluster_tolerance(np.sort(rows[0]), noise_band=(1e-8, 1e-6))
    assert single.mode == 'cluster'
    assert single.tol > 3e-8


def test_certificate_rejects_a_replicate_set_without_variation():
    rows = np.stack([np.array([0.0, 1e-9, 1e-2, 1.0])] * 4)
    dec = separation_certificate(rows)
    assert dec.mode == 'separation_unverified'
    assert dec.certified is False


def test_a_physical_gap_below_the_floor_is_the_single_spectrum_limitation():
    """The rule cannot distinguish it from noise; this pins the limitation.

    A gap inside the band is clustered by the single-spectrum route. The
    documented remedy is a measured band plus
    :func:`separation_certificate`, which sees the gap stay put across
    constructions; a bare spectrum cannot.
    """
    spectrum = _synthetic_from_gaps([1e-8, 3e-6, 1e-6, 1e-2])
    dec = cluster_tolerance(spectrum, noise_scale=3e-5)
    assert dec.mode == 'cluster'
    assert dec.n_noise_gaps == 3          # the 1e-6 physical gap is inside
    assert dec.noise_floor < dec.tol < dec.physical_min


def test_lowest_pair_auto_routes_and_returns_the_eigenpair():
    a = jnp.asarray(np.diag([-1.0, 0.4, 0.7, 2.0]))
    lam, u, dec = lowest_pair_auto(a)
    assert dec.mode == 'none'
    assert abs(float(lam) + 1.0) < 1e-10
    assert np.isfinite(np.asarray(u)).all()


def test_lowest_pair_auto_agrees_with_eigh_safe_without_degeneracy():
    rng = np.random.default_rng(21)
    b = rng.normal(size=(6, 6))
    a = jnp.asarray(b + b.T + np.diag(np.arange(6.0)))
    lam, u, _ = lowest_pair_auto(a)
    w, U = eigh_safe(a)
    assert abs(float(lam) - float(w[0])) < 1e-10
    assert abs(abs(float(u @ U[:, 0])) - 1.0) < 1e-8


def test_lowest_pair_auto_returns_unverified_and_a_correct_value():
    # Gaps [3e-5, 9e-5, ...] with an explicit floor at 3e-5 (ratio 3).
    a = jnp.diag(jnp.asarray([0.0, 3e-5, 1.2e-4, 1.9e-2, 1.0]))
    lam, u, dec = lowest_pair_auto(a, noise_scale=3e-5)
    assert dec.mode == 'separation_unverified'
    assert dec.tol is None
    assert dec.fallback == 'lanczos'
    w_ref = np.linalg.eigvalsh(np.asarray(a))
    assert abs(float(lam) - w_ref[0]) < 1e-10
    assert np.isfinite(np.asarray(u)).all()


def test_lowest_pair_auto_verifies_the_lanczos_value_and_degrades():
    """The decomposition-free kernel cannot resolve a 3e-6 gap at n = 5.

    Its Ritz vector is a mixture of the two lowest states and the Newton
    polish converges to the first excited state; without the value check
    this would be returned silently with an error equal to the full gap.
    """
    # Seed 3 is one of the measured failures (11 of the 20 seeds tried):
    # the returned value is the 3e-6 excited state against a 1e-16 ground
    # state, so the relative check (2e-6 on this scale) rejects it.
    a = _rotated_matrix([0.0, 3e-6, 1.9e-2, 1.0, 2.0], seed=3)
    lam, u, dec = lowest_pair_auto(a, noise_band=(1e-9, 1e-6))
    assert dec.mode == 'separation_unverified'
    assert dec.fallback == 'dense_safe'
    w_ref = np.linalg.eigvalsh(np.asarray(a))
    assert abs(float(lam) - w_ref[0]) < 1e-12
    assert np.isfinite(np.asarray(u)).all()


def test_frozen_physical_gap_corrupts_the_first_derivative():
    """The failure the certificate prevents, measured against FD.

    A physical gap of 2e-5 is misclassified as noise by the relative prior
    (band 6e-5), so the geometric-mean tolerance becomes 7.7e-4 and the
    eigenvector derivative loses the 1/2e-5 coupling. The certified band
    (1e-15) keeps the tolerance below the gap and recovers FD.
    """
    rng = np.random.default_rng(11)
    n = 5
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    spectrum = np.array([0.0, 2e-5, 1e-2, 1.0, 2.0])
    a0 = q @ np.diag(spectrum) @ q.T
    a0 = 0.5 * (a0 + a0.T)
    b = rng.normal(size=(n, n))
    b = 0.5 * (b + b.T)
    da = rng.normal(size=(n, n))
    da = 0.5 * (da + da.T)

    def value(x):
        w, u = np.linalg.eigh(a0 + x * da)
        return float(u[:, 0] @ b @ u[:, 0])

    def ad(x, tol):
        w, u = eigh_cluster_ift(jnp.asarray(a0 + x * da), tol=tol)
        u0 = u[:, 0]
        return u0 @ (jnp.asarray(b) @ u0)

    h = 1e-8
    fd = (value(h) - value(-h)) / (2 * h)
    g_frozen = float(jax.grad(lambda x: ad(x, 7.75e-4))(0.0))
    g_cert = float(jax.grad(lambda x: ad(x, 1.4e-10))(0.0))
    assert abs(g_frozen - fd) / abs(fd) > 0.5
    assert abs(g_cert - fd) / abs(fd) < 1e-5


def test_lowest_pair_auto_cluster_route_returns_the_ground_pair():
    """The ``cluster`` branch routes through ``eigh_cluster_ift``."""
    w = _synthetic_from_gaps([1e-8, 4e-8, 2.6e-2, 1.0])
    a = jnp.diag(jnp.asarray(w))
    lam, u, dec = lowest_pair_auto(a)
    assert dec.mode == 'cluster'
    assert dec.noise_floor < dec.tol < dec.physical_min
    assert abs(float(lam) - w[0]) < 1e-10
    assert abs(float(u @ u) - 1.0) < 1e-10
