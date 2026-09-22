"""Plateau detection on the normalized entropy spectrum."""
import numpy as np
import pytest

from diffmrc.analysis.plateau import detect_plateaus, count_above


def test_gap_produces_a_plateau_at_the_gap_height():
    """Seven strongly correlated orbitals, a clear gap, then weak ones."""
    s1 = np.array([1.30, 1.28, 1.27, 1.25, 1.24, 1.22, 1.20,
                   0.30, 0.25, 0.20])
    plateaus, _ = detect_plateaus(s1, plateau_values=10, threshold_step=0.01)
    assert 7 in plateaus


def test_uniform_spectrum_has_no_plateau():
    """A smooth decay gives no sustained plateau."""
    s1 = np.linspace(1.0, 0.01, 100)
    plateaus, _ = detect_plateaus(s1, plateau_values=10, threshold_step=0.01)
    assert plateaus == []


def test_returns_normalized_entropies_in_descending_order():
    s1 = np.array([0.5, 1.0, 0.25])
    _, norm = detect_plateaus(s1)
    assert np.allclose(norm, [1.0, 0.5, 0.25])


def test_all_zero_entropy_returns_empty():
    plateaus, _ = detect_plateaus(np.zeros(5))
    assert plateaus == []


def test_empty_input_returns_empty():
    plateaus, norm = detect_plateaus(np.array([]))
    assert plateaus == []
    assert norm.size == 0


def test_plateaus_are_sorted_descending_and_deduplicated():
    s1 = np.array([1.30, 1.28, 1.27, 1.25, 1.24, 1.22, 1.20,
                   0.30, 0.25, 0.20])
    plateaus, _ = detect_plateaus(s1, plateau_values=5, threshold_step=0.01)
    assert plateaus == sorted(set(plateaus), reverse=True)


def test_count_above_matches_brute_force():
    s1 = np.array([1.0, 0.5, 0.5, 0.25])
    th = np.array([0.0, 0.3, 0.5, 0.9])
    got = count_above(s1, th)
    want = [int((s1 > t).sum()) for t in th]
    assert list(got) == want


def test_count_above_returns_integers():
    got = count_above(np.array([1.0, 0.5]), np.array([0.0]))
    assert got.dtype.kind in 'iu'


def test_plateau_values_controls_sensitivity():
    """A shorter required run detects more plateaus on the same spectrum."""
    s1 = np.array([1.30, 1.28, 1.27, 1.20, 0.30, 0.25, 0.20, 0.15])
    strict, _ = detect_plateaus(s1, plateau_values=30, threshold_step=0.01)
    loose, _ = detect_plateaus(s1, plateau_values=5, threshold_step=0.01)
    assert len(loose) >= len(strict)


def test_single_orbital_spectrum_plateaus_at_one():
    """One orbital normalizes to 1.0, so N(tau) holds at 1 for every tau.

    The staircase never descends and the protocol reports a plateau of
    height one. That is the correct reading of the definition, though a
    one-orbital active space is not usually what a caller wants; the
    chemical rules in the policy layer are what guard against it.
    """
    plateaus, norm = detect_plateaus(np.array([1.3]))
    assert np.allclose(norm, [1.0])
    assert plateaus == [1]


def test_numerical_noise_does_not_produce_a_plateau():
    """The regression that motivates the absolute floor.

    A spectrum of pure round-off has a *relative* structure once it is
    normalized by its own maximum: in this example three orbitals carry
    2e-06 and seven carry 1e-06, so the staircase holds at ten over a
    wide threshold range and the sweep reports a plateau of height ten.
    Every one of those entropies is five orders of magnitude below the
    0.14 that the selection protocol requires before a wavefunction is
    called multiconfigurational, so the plateau is an artefact of the
    normalization and no active space should be selected from it.
    """
    s1 = np.array([2e-06, 2e-06, 2e-06] + [1e-06] * 7 + [0.0])
    plateaus, normalized = detect_plateaus(s1)
    assert plateaus == []
    # The spectrum is still returned for inspection.
    assert normalized[0] == pytest.approx(1.0)


def test_guard_yields_to_a_genuine_multireference_spectrum():
    """The floor must not suppress a real plateau."""
    s1 = np.array([1.30, 1.28, 1.27, 1.25, 1.24, 1.22, 1.20,
                   0.30, 0.25, 0.20])
    plateaus, _ = detect_plateaus(s1, plateau_values=10, threshold_step=0.01)
    assert 7 in plateaus


def test_min_entropy_can_be_disabled():
    """Passing zero restores the unguarded sweep, for callers that want it."""
    s1 = np.array([2e-06, 2e-06, 2e-06] + [1e-06] * 7 + [0.0])
    plateaus, _ = detect_plateaus(s1, min_entropy=0.0)
    assert 10 in plateaus


def test_guard_uses_the_single_reference_threshold_by_default():
    """The default floor is the protocol's 0.14, imported not duplicated."""
    from diffmrc.analysis.diagnostics import SINGLE_REFERENCE_THRESHOLD
    from diffmrc.analysis.plateau import DEFAULT_MIN_ENTROPY
    assert DEFAULT_MIN_ENTROPY == SINGLE_REFERENCE_THRESHOLD


def test_spectrum_just_below_the_floor_is_suppressed():
    """A boundary check: 0.13 is below the criterion, 0.15 is above."""
    below = np.array([0.13, 0.12, 0.11, 0.10])
    above = np.array([0.15, 0.12, 0.11, 0.10])
    assert detect_plateaus(below)[0] == []
    assert detect_plateaus(above)[0] != []
