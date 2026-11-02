"""Two-stage orbital-point convergence: loose optimizer, tight refinement."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.stationary import StopReason  # noqa: E402
from diffmrc.driver.orbitals import (  # noqa: E402
    converge_orbital_point, converge_orbital_point_two_stage, embed_theta)


def _quadratic(centre, curvature=2.0):
    """A convex function whose minimum is known."""
    centre = np.asarray(centre, dtype=float)
    curvature = float(curvature)

    def E(theta):
        d = jnp.asarray(theta) - jnp.asarray(centre)
        return 0.5 * curvature * jnp.sum(d * d)

    return E


def test_the_point_reaches_the_known_minimum():
    E = _quadratic([1.0, -2.0, 0.5])
    p = converge_orbital_point(E, 3)
    assert np.allclose(np.asarray(p.theta), [1.0, -2.0, 0.5], atol=1e-7)
    assert p.is_stationary
    assert p.stop_reason is StopReason.CONVERGED


def test_both_stages_report_their_own_gradient():
    """The unified entry point dropped this evidence.

    That is how "iteration limit reached" (scipy status 1) could sit in
    the code as a successful outcome: the call site saw only a gradient
    and had nothing to compare it against.
    """
    E = _quadratic([3.0], curvature=1.0)
    p = converge_orbital_point(E, 1, optimize_tol=1e-6)
    assert p.optimizer_reached
    assert p.optimizer_grad_norm <= 1e-6
    assert p.optimizer_nit >= 0
    # The refinement went further than the optimizer was asked to.
    assert p.grad_norm <= p.optimizer_grad_norm
    assert p.grad_norm < 1e-11


def test_a_loose_optimizer_target_does_not_stop_the_refinement():
    """The measured production shape: L-BFGS-B stalls, refinement walks.

    With optimize_tol set above the optimizer's practical floor the
    stage is allowed to end above it; the verdict comes from the final
    gradient, not from the optimizer's report.
    """
    E = _quadratic([2.0, -1.0], curvature=0.5)
    p = converge_orbital_point(E, 2, optimize_tol=1e-2)
    assert p.is_stationary
    assert p.grad_norm < 1e-11
    assert np.allclose(np.asarray(p.theta), [2.0, -1.0], atol=1e-6)


def test_a_function_without_a_stationary_point_is_not_stationary():
    """A gradient that never shrinks must not be called converged."""
    E = lambda t: jnp.sum(jnp.asarray(t))          # linear, no minimum
    p = converge_orbital_point(E, 3, max_steps=200)
    assert not p.is_stationary


def test_the_gradient_criterion_cannot_see_a_saddle():
    """A stationary saddle also has |g| = 0, and is reported stationary.

    This is the division of labour, not a defect: the curvature check in
    driver.run is what refuses a saddle. The shape is the measured one --
    the H2O/6-31g point from theta = 0 has three negative eigenvalues
    and refines to |g| = 3.9e-14.
    """
    def E(t):
        t = jnp.asarray(t)
        return 0.25 * t[0] ** 4 - 0.5 * t[0] ** 2 + 0.5 * t[1] ** 2

    p = converge_orbital_point(E, 2, theta0=np.zeros(2))
    assert p.is_stationary                      # |g| = 0 at the saddle
    H = np.asarray(jax.hessian(E)(np.asarray(p.theta)))
    assert np.linalg.eigvalsh(H)[0] < 0        # still a saddle


def test_the_verdict_follows_the_gradient_that_came_back():
    """Whatever the three stages return, stationarity is read off the
    gradient -- not off which stages ran or how they exited."""
    E = _quadratic([5.0] * 4, curvature=1e-3)
    p = converge_orbital_point(E, 4, max_steps=1, refine_max_iter=1)
    assert p.is_stationary == bool(p.grad_norm < p.target)
    if not p.is_stationary:
        assert p.stop_reason is not StopReason.CONVERGED


def test_a_stalled_point_is_polished_by_a_trust_region_method(monkeypatch):
    """The third stage exists because the first two stop early.

    Measured on H2O/6-31g's lower basin (2026-09-22): the two-stage
    protocol stalls at |g| = 3.7e-4 -- the damped Newton step cap is 10
    and its stall rule gives up -- while trust-ncg with the exact
    Hessian reaches 9.7e-9 at the same energy. The stall was the
    protocol's, not the chart's, so a third stage clears it.

    The two-stage path is faked here so the test is about the
    composition: when the refinement stops short of the target, the
    polish runs and its point is kept when the gradient improves.
    """
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    stalled = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                              grad_norm=3.7e-4, n_accepted=1,
                              stop_reason=StopReason.STALLED_GRADIENT,
                              target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: stalled)
    monkeypatch.setattr(orb, '_trust_region_polish',
                        lambda *a, **k: (np.array([0.1, -0.2]), -1.0000001,
                                         9.7e-9))
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert p.polished
    assert p.grad_norm == pytest.approx(9.7e-9)
    # 9.7e-9 is four orders better than the stall, and it clears the
    # driver's check threshold of 1e-8; it is still above the
    # refinement's 1e-11 target, so the point reports itself as not
    # stationary rather than rounding the claim up.
    assert p.grad_norm < 1e-8
    assert not p.is_stationary


def test_a_polish_that_does_not_improve_is_discarded(monkeypatch):
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    stalled = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                              grad_norm=1e-5, n_accepted=1,
                              stop_reason=StopReason.STALLED_GRADIENT,
                              target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: stalled)
    monkeypatch.setattr(orb, '_trust_region_polish',
                        lambda *a, **k: (np.array([0.0, 0.0]), -0.9, 1e-3))
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert not p.polished
    assert p.grad_norm == pytest.approx(1e-5)


def test_the_polish_is_skipped_when_the_refinement_already_converged(
        monkeypatch):
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    converged = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                                grad_norm=3.9e-14, n_accepted=2,
                                stop_reason=StopReason.CONVERGED,
                                target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: converged)
    called = []

    def _never(*a, **k):
        called.append(True)
        return (np.zeros(2), -1.0, 1e-12)

    monkeypatch.setattr(orb, '_trust_region_polish', _never)
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert not called
    assert not p.polished
    assert p.is_stationary


def test_the_polish_repeats_while_it_keeps_improving(monkeypatch):
    """Measured on H2O/6-31g: one pass reaches 4.6e-8, a second pass is
    what closes the last factor to the 1e-8 gate. Each pass is kept only
    if it lowers the gradient, and the loop stops at the target."""
    from diffmrc.core.stationary import StopReason, StationaryPoint
    from diffmrc.driver import orbitals as orb

    stalled = StationaryPoint(theta=jnp.zeros(2), energy=-1.0,
                              grad_norm=2.1e-4, n_accepted=1,
                              stop_reason=StopReason.STALLED_GRADIENT,
                              target=1e-11)
    monkeypatch.setattr(orb, 'refine_stationary_point',
                        lambda *a, **k: stalled)
    steps = [(np.array([0.1, 0.0]), -1.000001, 4.6e-8),
             (np.array([0.2, 0.0]), -1.000002, 3.0e-9),
             (np.array([0.3, 0.0]), -1.000003, 1.0e-12)]
    calls = []

    def _polish(*a, **k):
        calls.append(True)
        return steps[len(calls) - 1]

    monkeypatch.setattr(orb, '_trust_region_polish', _polish)
    p = orb.converge_orbital_point(lambda t: 0.0, 2)
    assert p.polished
    assert len(calls) == 3            # third pass met the target, loop ends
    assert p.grad_norm == pytest.approx(1.0e-12)
    assert p.is_stationary


def test_hessp_matches_the_explicit_hessian_on_a_non_quadratic_energy():
    """The polish's second-order object is a jvp of the gradient. Pin it
    against jax.hessian: the two must act identically on a vector. The
    HVP path is what keeps the memory at one tangent vector instead of
    the npar^2 tangent pairs the fwd-over-fwd Hessian broadcasts --
    measured on the M5-C batch: a 402 GB allocation at npar ~ 500."""
    from diffmrc.driver.orbitals import hessp_of

    def E(t):
        return (jnp.sum(jnp.sin(t)) + 0.5 * jnp.sum(t ** 2)
                + 0.25 * t[0] ** 2 * t[1] ** 2)

    t = jnp.asarray([0.3, -0.7, 1.1, 0.2])
    v = jnp.asarray([1.0, -0.5, 0.25, 2.0])
    got = np.asarray(hessp_of(E)(t, v))
    want = np.asarray(jax.hessian(E)(t)) @ np.asarray(v)
    assert np.allclose(got, want, atol=1e-10)


def test_embedding_maps_pairs_by_identity_not_by_position():
    """Core-virtual entries are zero and subset values land on their pairs.

    The two pair lists order their blocks identically but the full list
    carries an extra core-virtual block in the middle, so a positional
    copy would move every active-virtual value into a core-virtual slot.
    A shuffled subset must produce the same full vector.
    """
    from diffmrc.core.orblib import orbital_pairs

    ncore, nact, nmo = 2, 3, 9
    subset = orbital_pairs(ncore, nact, nmo, include_cv=False)
    full = orbital_pairs(ncore, nact, nmo, include_cv=True)
    theta = np.arange(len(subset), dtype=float)

    out = embed_theta(subset, full, theta)
    index = {pair: k for k, pair in enumerate(full)}
    assert out.shape == (len(full),)
    for k, pair in enumerate(subset):
        assert out[index[pair]] == theta[k]
    nvir = nmo - ncore - nact
    assert len(full) == ncore * nact + ncore * nvir + nact * nvir
    for i in range(ncore):
        for v in range(ncore + nact, nmo):
            assert out[index[(i, v)]] == 0.0

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(subset))
    shuffled = embed_theta([subset[i] for i in perm], full, theta[perm])
    assert np.array_equal(out, shuffled)

    with pytest.raises(ValueError):
        embed_theta(subset, full, theta[:-1])
    with pytest.raises(ValueError):
        embed_theta([(0, nmo + 1)], full, np.zeros(1))


def test_the_split_rule_matches_the_measured_block_ratios():
    """The route criterion is the size of the core-virtual block.

    The three partitions are the measured ones: HI/def2-SVP CAS(2,2)
    (204 core-virtual pairs against 58), H2O/6-31G CAS(2,2) (28 against
    22) and N2/6-31G CAS(4,4) (45 against 56). Only the first is the
    regime where the joint L-BFGS stage stalled.
    """
    from diffmrc.driver.orbitals import should_split_rotation_set

    assert should_split_rotation_set(12, 2, 31)
    assert not should_split_rotation_set(4, 2, 13)
    assert not should_split_rotation_set(5, 4, 18)
    assert not should_split_rotation_set(0, 2, 5)


def test_two_stage_converges_the_coupling_the_subset_route_leaves():
    """A synthetic quadratic whose subset optimum is not the full one.

    The subset minimum leaves the complement coordinates at zero; the
    full minimum has both blocks displaced. The first stage must stop at
    the subset optimum, and the second must remove the full-space
    gradient. Nothing here is an electronic-structure statement -- it
    pins the composition and the mapping.
    """
    nfull = 6
    full_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    subset_pairs = [(0, 1), (0, 3), (1, 3)]
    subset_index = [full_pairs.index(pair) for pair in subset_pairs]

    rng = np.random.default_rng(7)
    a = rng.standard_normal((nfull, nfull))
    hess = a @ a.T / nfull + np.eye(nfull)
    lin = rng.standard_normal(nfull)

    def E_full(t):
        t = jnp.asarray(t)
        return 0.5 * t @ hess @ t - lin @ t

    def E_subset(u):
        t = jnp.zeros(nfull).at[jnp.asarray(subset_index)].set(u)
        return E_full(t)

    point = converge_orbital_point_two_stage(
        E_subset, len(subset_pairs), E_full, nfull, subset_pairs,
        full_pairs, optimize_tol=1e-6, refine_target=1e-10,
        max_steps=500, refine_max_iter=60)

    target = np.linalg.solve(hess, lin)
    grad_embedded = jax.grad(E_full)(point.embedded_theta)
    assert float(jnp.linalg.norm(grad_embedded)) > 1e-2
    assert point.grad_norm < 1e-8
    assert np.allclose(np.asarray(point.theta), target, atol=1e-6)
    assert point.energy == pytest.approx(-0.5 * lin @ target, abs=1e-9)
    assert point.energy < float(E_full(point.embedded_theta)) - 1e-4

    report = point.stages_report()
    assert report['n_parameters'] == dict(subset=3, full=6)
    assert report['stage1']['seconds'] >= 0.0
    assert report['stage2']['seconds'] >= 0.0
    assert report['stage2']['energy_change'] < 0.0


def test_the_polish_descends_a_curved_valley_with_the_hvp_hessian():
    """End to end through scipy's trust-ncg with the HVP Hessian -- the
    exact call path the driver's third stage uses -- on the classic
    stalling valley (Rosenbrock)."""
    from diffmrc.driver.orbitals import _trust_region_polish

    def E(t):
        return (1.0 - t[0]) ** 2 + 100.0 * (t[1] - t[0] ** 2) ** 2

    out = _trust_region_polish(E, np.array([-1.2, 1.0]), 1e-10,
                               max_iter=200)
    assert out is not None
    th, en, gn = out
    assert np.allclose(th, [1.0, 1.0], atol=1e-4)
    assert en == pytest.approx(0.0, abs=1e-8)
