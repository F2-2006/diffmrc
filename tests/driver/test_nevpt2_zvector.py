"""The Si-channel exact form and the orbital Z-vector response."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


# ---------------------------------------------------------------------------
# Si channel: the value mask is a discontinuity, not a removable singularity
# ---------------------------------------------------------------------------

# Measured on H2O/6-31G CAS(2,2) at the CASSCF stationary point: the two
# near-zero channels carry these invariants, and the two healthy channels
# the ones below.  They are the fixture the derivative test is built on.
_NEAR = (7.607595e-13, 1.339689e-12)
_HEALTHY = (6.896498e-07, 1.257363e-06)
_DIFF = -0.72


def test_si_energy_modes_agree_away_from_the_mask():
    """Away from |n| = 1e-12 the two algebraic forms are the same value."""
    from diffmrc.driver.nevpt2_zvector import si_energy_from_invariants

    norm = jnp.asarray([_HEALTHY[0], 1.168083e-07])
    ener = jnp.asarray([_HEALTHY[1], 2.129651e-07])
    diff = jnp.asarray([-_DIFF, -0.72])
    a = float(si_energy_from_invariants(norm, ener, diff, mode='exact'))
    b = float(si_energy_from_invariants(norm, ener, diff, mode='masked'))
    assert abs(a - b) < 1e-14
    assert abs(a) > 1e-9            # the test is not vacuous


def test_the_mask_drops_a_channel_the_exact_form_keeps():
    """At n = 7.6e-13 the release mask returns zero; the exact form returns
    -n^2/(d n + h), a value of 2.5e-13 that is negligible for the energy
    but whose derivative is not."""
    from diffmrc.driver.nevpt2_zvector import si_energy_from_invariants

    n = jnp.asarray([_NEAR[0]])
    h = jnp.asarray([_NEAR[1]])
    d = jnp.asarray([-_DIFF])
    exact = float(si_energy_from_invariants(n, h, d, mode='exact'))
    masked = float(si_energy_from_invariants(n, h, d, mode='masked'))
    d0 = -float(_DIFF)
    analytic = -float(_NEAR[0]) ** 2 / (d0 * float(_NEAR[0])
                                        + float(_NEAR[1]))
    assert masked == 0.0
    assert exact == pytest.approx(analytic, abs=1e-18)
    assert exact != 0.0


def test_the_exact_form_has_the_analytic_channel_slopes():
    """df/dn and df/dh follow from the value formula; at the near-zero
    channel they are O(0.5-1), which is why dropping the channel costs a
    real derivative even though it costs only 2.5e-13 in energy."""
    from diffmrc.driver.nevpt2_zvector import si_energy_from_invariants

    n0, h0, d0 = float(_NEAR[0]), float(_NEAR[1]), -float(_DIFF)
    den = d0 * n0 + h0
    dfdn = -(2.0 * n0 * den - n0 ** 2 * d0) / den ** 2
    dfdh = n0 ** 2 / den ** 2
    assert abs(dfdn) > 0.4
    assert abs(dfdh) > 0.08

    def f(n, h):
        return si_energy_from_invariants(jnp.asarray([n]), jnp.asarray([h]),
                                         jnp.asarray([d0]), mode='exact')

    eps = 1e-17
    fd_n = (float(f(n0 + eps, h0)) - float(f(n0 - eps, h0))) / (2 * eps)
    fd_h = (float(f(n0, h0 + eps)) - float(f(n0, h0 - eps))) / (2 * eps)
    assert fd_n == pytest.approx(dfdn, rel=1e-4, abs=1e-6)
    assert fd_h == pytest.approx(dfdh, rel=1e-4, abs=1e-6)


def test_the_exact_form_is_smooth_at_the_zero_channel():
    """AD of the exact form agrees with a step below the channel's
    variation scale; the mask's AD is zero there because both points of
    the difference lie under the threshold.  This is the boundary the
    exact form removes, stated as a test rather than as prose."""
    from diffmrc.driver.nevpt2_zvector import si_energy_from_invariants

    n0, h0, d0 = float(_NEAR[0]), float(_NEAR[1]), -float(_DIFF)

    def energy(t, mode):
        # A ray through the channel invariants with the measured seed-5
        # slopes (dn/dc = -0.28, dh/dc = +0.52); n and h vary linearly in
        # the displacement to first order through the CI.
        n = jnp.asarray(n0 - 2.8e-1 * t)
        h = jnp.asarray(h0 + 5.2e-1 * t)
        return si_energy_from_invariants(n, h, jnp.asarray(d0), mode=mode)

    ad = float(jax.jvp(lambda t: energy(t, 'exact'), (0.0,), (1.0,))[1])
    ad_mask = float(jax.jvp(lambda t: energy(t, 'masked'), (0.0,), (1.0,))[1])
    fd = (float(energy(1e-13, 'exact')) - float(energy(-1e-13, 'exact'))) / 2e-13
    assert abs(ad) > 0.1
    assert ad == pytest.approx(fd, rel=1e-3)
    assert abs(ad_mask) < 1e-4 * abs(ad)


# ---------------------------------------------------------------------------
# Orbital Z-vector: the adjoint form must be the same number as s^T D
# ---------------------------------------------------------------------------

def _symmetric_with_a_soft_mode(n=8, seed=3):
    """(H, soft eigenvector): eigenvalues 0.5..100 and one at 3.8e-6.

    The ratio 3.8e-6/100 = 3.8e-8 sits between rcond = 1e-7 and 1e-12,
    so the two cutoffs separate cleanly -- the same ordering as the H2O
    CASSCF Hessian (soft mode 3.8e-6, kappa 2.3e7)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.linspace(0.5, 100.0, n)
    lam[-1] = 3.8e-6
    return (q * lam) @ q.T, q[:, -1]


def test_orbital_zvector_solves_the_adjoint_equation():
    from diffmrc.driver.nevpt2_zvector import orbital_zvector

    h, soft = _symmetric_with_a_soft_mode()
    rng = np.random.default_rng(4)
    s = rng.normal(size=h.shape[0])
    s -= soft * (s @ soft)               # no component along the soft mode
    z, info = orbital_zvector(h, s, rcond=1e-7)
    assert np.abs(h @ z - s).max() < 1e-10
    assert info['residual_relative'] < 1e-10
    assert np.all(np.isfinite(z))


def test_the_lagrangian_form_equals_the_direct_response_operator():
    from diffmrc.driver.nevpt2_zvector import (lagrangian_relaxed_gradient,
                                               orbital_zvector)

    h, _soft = _symmetric_with_a_soft_mode()
    rng = np.random.default_rng(5)
    s = rng.normal(size=h.shape[0])
    m = rng.normal(size=(h.shape[0], 6))
    g = rng.normal(size=6)
    z, _ = orbital_zvector(h, s, rcond=1e-7)
    adjoint = lagrangian_relaxed_gradient(g, z, m)
    d = -np.linalg.pinv(h, rcond=1e-7) @ m
    direct = g + s @ d
    assert np.abs(adjoint - direct).max() < 1e-10


def test_a_zero_source_reduces_to_the_frozen_gradient():
    from diffmrc.driver.nevpt2_zvector import (lagrangian_relaxed_gradient,
                                               orbital_zvector)

    h, _soft = _symmetric_with_a_soft_mode()
    m = np.zeros((h.shape[0], 6))
    g = np.linspace(-0.1, 0.1, 6)
    z, info = orbital_zvector(h, np.zeros(h.shape[0]), rcond=1e-7)
    assert np.abs(z).max() == 0.0
    assert info['residual'] == 0.0
    assert np.array_equal(lagrangian_relaxed_gradient(g, z, m), g)


def test_rcond_cuts_the_soft_mode_from_the_response():
    """The soft mode is a null direction of the gradient source on the H2O
    CASSCF surface; with rcond=1e-7 its contribution is discarded, while a
    looser cutoff keeps it and the solve amplifies roundoff."""
    from diffmrc.driver.nevpt2_zvector import orbital_zvector

    h, soft = _symmetric_with_a_soft_mode()
    s = soft.copy()                 # a source purely along the soft mode
    z_tight, _ = orbital_zvector(h, s, rcond=1e-7)
    z_loose, _ = orbital_zvector(h, s, rcond=1e-12)
    assert np.abs(z_tight).max() < 1e-12
    assert np.abs(z_loose).max() > 1e5


# ---------------------------------------------------------------------------
# The chain-level assembly (needs pyscfad)
# ---------------------------------------------------------------------------

def _h2o_sto3g():
    gto = pytest.importorskip("pyscfad.gto").Mole
    mol = gto()
    mol.atom = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
    mol.basis = "sto-3g"
    mol.build()
    return mol


def test_the_exact_si_chain_matches_the_release_total_with_a_core():
    """Both assemblies run on identical blocks; the correction is the
    difference of the two core-class sums.  On H2O/STO-3G (ncore=1) the
    Si channels are away from the mask, so the totals must agree to
    roundoff.  A drift here would mean the recomputed blocks differ from
    the ones the release total used."""
    pytest.importorskip("pyscfad")
    from pyscfad import scf

    from diffmrc.core.orblib import orbital_pairs
    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn
    from diffmrc.driver.nevpt2_zvector import make_nevpt2_joint_fn_si_exact

    mol = _h2o_sto3g()
    nmo = mol.nao_nr()
    ncore, nact, nele = 1, 2, 2
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    theta = jnp.zeros(len(pairs))
    builder = lambda m: scf.RHF(m)  # noqa: E731
    f_ref = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                                 e2_scale=1.0, ci_solver="blocked")
    f_new = make_nevpt2_joint_fn_si_exact(ncore, nact, nele, nmo, pairs,
                                          builder, e2_scale=1.0,
                                          ci_solver="blocked")
    e_ref = float(f_ref(mol, theta))
    e_new = float(f_new(mol, theta))
    assert abs(e_new - e_ref) < 1e-11
    # The e2_scale = 0 anchor: both functions are the CASSCF energy and
    # the correction path is skipped entirely.
    f0_ref = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                                  e2_scale=0.0, ci_solver="blocked")
    f0_new = make_nevpt2_joint_fn_si_exact(ncore, nact, nele, nmo, pairs,
                                           builder, e2_scale=0.0,
                                           ci_solver="blocked")
    # Two dispatches of the SCF re-run the kernel; the measured spread of
    # the CASSCF energy across dispatches on this fixture is 6e-14, the
    # same re-dispatch roundoff the release anchor test documents.
    assert abs(float(f0_new(mol, theta)) - float(f0_ref(mol, theta))) < 1e-12


def test_the_exact_si_chain_is_the_release_chain_without_a_core():
    """ncore=0 has no core classes; the correction must be identically
    zero and the two chains must agree bitwise through the same trace."""
    pytest.importorskip("pyscfad")
    from pyscfad import gto, scf

    from diffmrc.core.orblib import orbital_pairs
    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn
    from diffmrc.driver.nevpt2_zvector import make_nevpt2_joint_fn_si_exact

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "6-31g"
    mol.build()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    theta = jnp.zeros(len(pairs))
    builder = lambda m: scf.RHF(m)  # noqa: E731
    f_ref = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                                 e2_scale=1.0)
    f_new = make_nevpt2_joint_fn_si_exact(ncore, nact, nele, nmo, pairs,
                                          builder, e2_scale=1.0)
    assert float(f_new(mol, theta)) == float(f_ref(mol, theta))
