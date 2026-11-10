"""Multistart spectra and the basin-hopping search.

The tests here use analytic one-dimensional surfaces and the real refinement
kernel: no optimizer is stubbed, so what is checked is the search logic on
top of the same damped-Newton refinement the production path uses.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from diffmrc.core.stationary import (  # noqa: E402
    StationaryCandidate,
    basin_hopping_search,
    cluster_stationary_points,
    curvature_report,
    refine_stationary_point,
)
from diffmrc.driver.orbitals import converge_orbital_point  # noqa: E402

import jax.numpy as jnp  # noqa: E402


def _curv(*eigs):
    return curvature_report(np.diag(np.asarray(eigs, dtype=float)))


def _candidate(energy, label, grad_norm=1e-12, curvature='min'):
    if curvature == 'none':
        cur = None
    elif curvature == 'min':
        cur = _curv(0.2, 0.5)
    else:
        cur = _curv(*curvature)
    return StationaryCandidate(theta=np.asarray([energy, 0.0]),
                               energy=energy, grad_norm=grad_norm,
                               curvature=cur, label=label)


# ---------------------------------------------------------------------------
# Spectrum construction.
# ---------------------------------------------------------------------------


def test_spectrum_clusters_runs_that_reached_the_same_point():
    a = _candidate(-1.0, 'a')
    b = _candidate(-1.0 + 1e-9, 'b')
    c = _candidate(-0.9, 'c')
    spectrum = cluster_stationary_points([a, b, c])
    assert spectrum.n_distinct == 2
    assert spectrum.n_stationary == 3
    assert spectrum.points[0].n_members == 2
    assert spectrum.points[0].labels == ('a', 'b')
    assert spectrum.points[0].energy == pytest.approx(-1.0)
    assert spectrum.points[1].n_members == 1


def test_spectrum_keeps_isoenergetic_points_apart_with_an_orbital_test():
    """Energy alone cannot separate two different solutions at one energy."""
    a = _candidate(-1.0, 'a')
    b = _candidate(-1.0, 'b')
    spectrum = cluster_stationary_points([a, b], same_point=lambda x, y: False)
    assert spectrum.n_distinct == 2
    assert spectrum.points[0].energy == pytest.approx(-1.0)


def test_spectrum_ignores_non_stationary_candidates():
    good = _candidate(-1.0, 'good', grad_norm=1e-12)
    bad = _candidate(-2.0, 'bad', grad_norm=1e-3)
    spectrum = cluster_stationary_points([good, bad], target=1e-8)
    assert spectrum.n_candidates == 2
    assert spectrum.n_stationary == 1
    assert spectrum.n_distinct == 1
    assert spectrum.best.energy == pytest.approx(-1.0)


def test_spectrum_is_independent_of_completion_order():
    a = _candidate(-1.0, 'a', grad_norm=3e-12)
    b = _candidate(-1.0, 'b', grad_norm=1e-12)
    c = _candidate(-0.5, 'c')
    forward = cluster_stationary_points([a, b, c])
    backward = cluster_stationary_points([c, b, a])
    assert forward.points[0].label == backward.points[0].label == 'b'
    assert forward.points[0].n_members == backward.points[0].n_members == 2
    assert forward.spread == pytest.approx(backward.spread)


def test_energy_levels_merge_orbital_orientations_of_one_solution():
    """Degenerate orientations are one level and several points.

    Measured on N2/6-31G CAS(2,2): eight converged starts gave one energy
    and six active-space orientations of the same solution.
    """
    a = _candidate(-1.0, 'a')
    b = _candidate(-1.0, 'b')
    c = _candidate(-0.5, 'c')
    spectrum = cluster_stationary_points(
        [a, b, c], same_point=lambda x, y: False)
    assert spectrum.n_distinct == 3
    assert spectrum.n_energy_levels == 2


def test_spectrum_reports_certificate_coverage():
    certified = _candidate(-1.0, 'a')
    uncertified = _candidate(-0.5, 'b', curvature='none')
    spectrum = cluster_stationary_points([certified, uncertified])
    assert spectrum.certificate_coverage == pytest.approx(0.5)
    assert spectrum.n_certified == 1
    assert spectrum.n_minima == 1


def test_spectrum_strict_flag_uses_the_strict_target():
    loose = _candidate(-1.0, 'loose', grad_norm=1e-9)
    tight = _candidate(-0.5, 'tight', grad_norm=1e-13)
    spectrum = cluster_stationary_points(
        [loose, tight], target=1e-6, strict_target=1e-11)
    flags = {p.label: p.strict for p in spectrum.points}
    assert flags['loose'] is False
    assert flags['tight'] is True


def test_spectrum_selection_prefers_a_certified_minimum_over_a_lower_saddle():
    saddle = _candidate(-2.0, 'saddle', curvature=(1.0, -0.5))
    minimum = _candidate(-1.0, 'minimum')
    spectrum = cluster_stationary_points([saddle, minimum])
    selection = spectrum.selection
    assert selection.chosen is minimum
    assert selection.status == 'minimum'


def test_empty_spectrum_has_a_zero_coverage_and_no_best():
    spectrum = cluster_stationary_points([])
    assert spectrum.n_distinct == 0
    assert spectrum.best is None
    assert spectrum.certificate_coverage == 0.0
    assert spectrum.spread == 0.0


# ---------------------------------------------------------------------------
# Basin hopping on a double well.
#
# E(x) = (x^2 - 1)^2 + 0.2 x has two minima, the left one lower by about
# 0.4 Eh. Refinement is downhill-only, so a start in the right basin cannot
# cross the barrier; a move that proposes a point in the left basin is what
# the search has to exploit.
# ---------------------------------------------------------------------------


def _double_well(x):
    t = jnp.asarray(x)[0]
    return (t * t - 1.0) ** 2 + 0.2 * t


def _double_well_hessian(x):
    t = float(np.asarray(x)[0])
    return np.asarray([[12.0 * t * t - 4.0]])


def _double_well_converge(theta0, label):
    point = converge_orbital_point(
        _double_well, 1, theta0=np.asarray(theta0, dtype=float))
    return StationaryCandidate(
        theta=point.theta, energy=float(point.energy),
        grad_norm=float(point.grad_norm),
        curvature=curvature_report(_double_well_hessian(point.theta)),
        label=label)


def _fixed_start(value):
    """A proposal that always returns one point: a controlled move."""
    def propose(rng, theta):
        return np.asarray([float(value)])
    return propose


def _jitter(scale):
    def propose(rng, theta):
        return np.asarray(theta) + scale * rng.standard_normal(
            np.shape(theta))
    return propose


def test_basin_hopping_finds_the_lower_well():
    result = basin_hopping_search(
        _double_well_converge, np.asarray([0.5]),
        [('cross', _fixed_start(-0.5))],
        max_steps=4, max_no_new=2, seed=0)
    assert result.best is not None
    assert result.best.energy < _double_well(np.asarray([0.5]))
    assert result.n_new == 1
    assert result.trace[0].accepted
    assert result.trace[0].is_new
    assert result.spectrum.n_distinct == 2
    assert result.certified


def test_the_result_keeps_the_initial_point_not_the_final_walker():
    """``initial`` is the point the walk started from, not where it ended.

    A first version stored the walker's current point at return time, which
    made a cost-to-first-hit analysis read index 0 as the starting point
    even when the start was a saddle. The field has to be the start.
    """
    start_energy = _double_well_converge(np.asarray([0.5]), 'x').energy
    result = basin_hopping_search(
        _double_well_converge, np.asarray([0.5]),
        [('cross', _fixed_start(-0.5))], max_steps=4, max_no_new=2, seed=0)
    assert result.initial.label == 'start'
    assert result.initial.energy == pytest.approx(start_energy, abs=1e-10)
    assert result.best.energy < result.initial.energy


def test_basin_hopping_does_not_walk_to_a_higher_well_when_greedy():
    result = basin_hopping_search(
        _double_well_converge, np.asarray([-0.5]),
        [('cross', _fixed_start(0.5))],
        max_steps=3, max_no_new=2, seed=0)
    assert not result.trace[0].accepted
    assert result.trace[0].note == 'higher energy'
    assert result.n_accepted == 0
    # The point is still recorded in the spectrum even though it was rejected.
    assert result.spectrum.n_distinct == 2
    assert result.best.energy == pytest.approx(
        float(_double_well_converge(np.asarray([-0.5]), 'x').energy))


def test_basin_hopping_metropolis_accepts_an_uphill_move():
    result = basin_hopping_search(
        _double_well_converge, np.asarray([-0.5]),
        [('cross', _fixed_start(0.5))],
        max_steps=3, max_no_new=3, temperature=1e6, seed=0)
    assert result.trace[0].accepted
    assert result.n_accepted >= 1
    assert result.trace[0].energy > _double_well_converge(
        np.asarray([-0.5]), 'x').energy


def test_basin_hopping_stops_after_a_run_without_new_points():
    def converge(theta0, label):
        p = refine_stationary_point(
            lambda t: jnp.sum(jnp.asarray(t) ** 2), theta0)
        return StationaryCandidate(
            theta=p.theta, energy=float(p.energy), grad_norm=float(p.grad_norm),
            curvature=_curv(2.0, 2.0), label=label)

    result = basin_hopping_search(
        converge, np.zeros(2), [('jitter', _jitter(1e-3))],
        max_steps=50, max_no_new=3, seed=1)
    assert result.stop_reason == 'no_new_points'
    assert result.n_steps == 3
    assert result.spectrum.n_distinct == 1


def test_basin_hopping_reports_the_step_budget():
    def converge(theta0, label):
        p = refine_stationary_point(
            lambda t: jnp.sum(jnp.asarray(t) ** 2), theta0)
        return StationaryCandidate(
            theta=p.theta, energy=float(p.energy), grad_norm=float(p.grad_norm),
            curvature=_curv(2.0, 2.0), label=label)

    result = basin_hopping_search(
        converge, np.zeros(2), [('jitter', _jitter(1e-3))],
        max_steps=5, max_no_new=None, seed=1)
    assert result.stop_reason == 'step_budget'
    assert result.n_steps == 5


def test_basin_hopping_reports_the_time_budget_before_the_first_move():
    result = basin_hopping_search(
        _double_well_converge, np.asarray([0.5]),
        [('cross', _fixed_start(-0.5))],
        max_steps=10, max_seconds=0.0, seed=0)
    assert result.stop_reason == 'time_budget'
    assert result.n_steps == 0
    assert result.best is not None


def test_basin_hopping_refuses_a_point_without_a_curvature_report():
    def converge(theta0, label):
        p = refine_stationary_point(
            lambda t: jnp.sum(jnp.asarray(t) ** 2), theta0)
        return StationaryCandidate(
            theta=p.theta, energy=float(p.energy), grad_norm=float(p.grad_norm),
            curvature=None, label=label)

    result = basin_hopping_search(
        converge, np.zeros(2), [('jitter', _jitter(1e-3))],
        max_steps=3, max_no_new=3, require_curvature=True, seed=1)
    assert result.n_accepted == 0
    assert result.trace[0].note == 'no curvature report'
    assert not result.certified
    assert result.certificate_coverage == 0.0


def test_basin_hopping_coverage_gate_blocks_the_no_new_stop():
    def converge(theta0, label):
        p = refine_stationary_point(
            lambda t: jnp.sum(jnp.asarray(t) ** 2), theta0)
        return StationaryCandidate(
            theta=p.theta, energy=float(p.energy), grad_norm=float(p.grad_norm),
            curvature=None, label=label)

    gated = basin_hopping_search(
        converge, np.zeros(2), [('jitter', _jitter(1e-3))],
        max_steps=4, max_no_new=1, coverage_gate=True, seed=1)
    ungated = basin_hopping_search(
        converge, np.zeros(2), [('jitter', _jitter(1e-3))],
        max_steps=4, max_no_new=1, coverage_gate=False, seed=1)
    assert gated.stop_reason == 'step_budget'
    assert ungated.stop_reason == 'no_new_points'


def test_novelty_window_ignores_a_high_energy_stationary_point():
    """A chart stationary point far above the best must not reset the counter.

    The theta chart contains stationary points in which a virtual orbital
    occupies a core slot; measured on H2O/6-31G one sits 1.15 Ha above the
    ground state. It is a real stationary point of the chart and a useless
    search result, so it belongs in the spectrum but not in the novelty
    counter that decides when to stop.
    """
    low = _candidate(-1.0, 'low')
    high = _candidate(10.0, 'high')

    def run(window):
        sequence = [low, high, low, low, low, low]

        def converge(theta0, label):
            return sequence.pop(0)

        return basin_hopping_search(
            converge, np.zeros(1), [('fixed', _fixed_start(0.0))],
            max_steps=4, max_no_new=2, novelty_window=window, seed=0)

    gated = run(0.5)
    ungated = run(None)
    assert gated.stop_reason == ungated.stop_reason == 'no_new_points'
    assert gated.n_steps == 2
    assert ungated.n_steps == 3
    assert gated.spectrum.n_distinct == 2
    assert ungated.spectrum.n_distinct == 2


def test_basin_hopping_is_deterministic_for_a_seed():
    first = basin_hopping_search(
        _double_well_converge, np.asarray([0.5]),
        [('jitter', _jitter(0.3))], max_steps=6, max_no_new=None, seed=7)
    second = basin_hopping_search(
        _double_well_converge, np.asarray([0.5]),
        [('jitter', _jitter(0.3))], max_steps=6, max_no_new=None, seed=7)
    assert first.trace == second.trace
    assert first.n_accepted == second.n_accepted
    assert first.n_new == second.n_new


def test_the_cached_graph_reads_runtime_arrays():
    """The compiled graph is reused, but the runtime arrays are not baked in.

    ``refine_stationary_point`` builds its gradient and Hessian-vector
    graphs once per energy function. The contract is that large reference
    arrays travel through ``arrays=``; a cache that captured them would
    refine every later start towards the first one's centre.
    """
    def energy(theta, centre):
        d = jnp.asarray(theta) - jnp.asarray(centre)
        return 0.5 * jnp.sum(d * d)

    first = refine_stationary_point(energy, np.zeros(2),
                                    arrays=(np.ones(2),))
    second = refine_stationary_point(energy, np.zeros(2),
                                     arrays=(3.0 * np.ones(2),))
    assert np.allclose(np.asarray(first.theta), 1.0, atol=1e-9)
    assert np.allclose(np.asarray(second.theta), 3.0, atol=1e-9)


def test_basin_hopping_rejects_an_empty_move_list():
    with pytest.raises(ValueError, match='at least one'):
        basin_hopping_search(_double_well_converge, np.zeros(1), [])


def test_basin_hopping_rejects_a_move_without_a_callable():
    with pytest.raises(ValueError, match='pair'):
        basin_hopping_search(_double_well_converge, np.zeros(1),
                             [('broken', None)])


def test_basin_hopping_rejects_a_non_positive_no_new_budget():
    with pytest.raises(ValueError, match='max_no_new'):
        basin_hopping_search(_double_well_converge, np.zeros(1),
                             [('jitter', _jitter(0.1))], max_no_new=0)
