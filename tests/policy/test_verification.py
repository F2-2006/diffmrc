"""Entropy-spectrum recombination and the two-source verdict."""
import numpy as np
import pytest

from diffmrc.policy.verification import (
    Verdict,
    recombine_entropies,
    verify_candidate,
)


def test_recombination_takes_the_largest_estimate_per_orbital():
    """An orbital may appear in several subsets; keep the largest entropy.

    Restricting an orbital's environment to a subset can only lower the
    measured entanglement, so the widest view available is the best
    estimate.
    """
    per_subset = [
        ({0: 0.9, 1: 0.2}, None),
        ({1: 0.7, 2: 0.4}, None),
    ]
    s1 = recombine_entropies(per_subset, 3)
    assert s1[0] == pytest.approx(0.9)
    assert s1[1] == pytest.approx(0.7)
    assert s1[2] == pytest.approx(0.4)


def test_recombination_accepts_bare_mappings():
    s1 = recombine_entropies([{0: 0.5}, {1: 0.7}], 2)
    assert np.allclose(s1, [0.5, 0.7])


def test_recombination_leaves_unseen_orbitals_at_zero():
    s1 = recombine_entropies([({1: 0.5}, None)], 3)
    assert s1[0] == 0.0
    assert s1[1] == pytest.approx(0.5)
    assert s1[2] == 0.0


def test_recombination_ignores_out_of_range_indices():
    s1 = recombine_entropies([({0: 0.5, 9: 0.9}, None)], 2)
    assert np.allclose(s1, [0.5, 0.0])


def test_agreement_verdict():
    """Seven entangled orbitals above a gap, three weakly correlated."""
    s1 = np.array([1.25, 1.23, 1.22, 1.20, 1.19, 1.17, 1.15,
                   0.05, 0.04, 0.03])
    r = verify_candidate(s1, candidate=[0, 1, 2, 3, 4, 5, 6])
    assert r.verdict is Verdict.AGREES
    assert r.plateau_heights == [7]


def test_candidate_too_small_is_reported():
    """Nine orbitals carry entropy but the candidate holds seven."""
    s1 = np.array([1.25, 1.23, 1.22, 1.20, 1.19, 1.17, 1.15, 1.13, 1.11,
                   0.05, 0.04])
    r = verify_candidate(s1, candidate=[0, 1, 2, 3, 4, 5, 6])
    assert r.verdict is Verdict.CANDIDATE_TOO_SMALL
    assert r.plateau_heights == [9]
    assert any('misses' in e for e in r.evidence)


def test_candidate_too_large_is_reported():
    """Only three orbitals are entangled; the candidate holds seven."""
    s1 = np.array([1.25, 1.23, 1.22, 0.05, 0.04, 0.03, 0.02, 0.01])
    r = verify_candidate(s1, candidate=list(range(7)))
    assert r.verdict is Verdict.CANDIDATE_TOO_LARGE
    assert r.plateau_heights == [3]
    assert any('weakly correlated' in e for e in r.evidence)


def test_a_noise_spectrum_verdicts_as_single_reference():
    """The regression that matters most.

    Every entropy here is five orders of magnitude below the 0.14 the
    protocol requires. The correct answer is to refuse, not to report a
    plateau: normalizing would turn the round-off into a staircase and an
    active space would be chosen from noise.
    """
    s1 = np.array([2e-06, 2e-06, 2e-06] + [1e-06] * 7 + [0.0])
    r = verify_candidate(s1, candidate=list(range(7)))
    assert r.verdict is Verdict.SINGLE_REFERENCE
    assert r.plateau_heights == []
    assert r.is_single_reference
    assert any('below the multireference criterion' in e for e in r.evidence)


def test_spectrum_just_below_the_criterion_is_single_reference():
    s1 = np.array([0.13, 0.12, 0.11, 0.10])
    assert verify_candidate(s1, candidate=[0, 1]).verdict \
        is Verdict.SINGLE_REFERENCE


def test_no_plateau_verdict_when_the_spectrum_is_smooth():
    """Above the floor but with no gap: the protocol falls back to 1-2%."""
    s1 = np.linspace(1.0, 0.01, 60)
    r = verify_candidate(s1, candidate=list(range(10)))
    assert r.verdict is Verdict.NO_PLATEAU
    assert r.plateau_heights == []
    assert r.weakly_correlated is not None
    assert any('excluding' in e for e in r.evidence)


def test_result_reports_zs1_and_the_single_reference_test():
    s1 = np.full(6, np.log(4.0))
    r = verify_candidate(s1, candidate=list(range(6)))
    assert r.zs1 == pytest.approx(1.0, abs=1e-12)
    assert not r.is_single_reference


def test_an_empty_candidate_is_compared_against_the_plateau():
    """A candidate of nothing can still be judged against the spectrum."""
    s1 = np.array([1.25, 1.23, 1.22, 1.20, 1.19, 1.17, 1.15,
                   0.05, 0.04, 0.03])
    r = verify_candidate(s1, candidate=[])
    assert r.verdict is Verdict.CANDIDATE_TOO_SMALL
    assert r.n_candidate == 0


def test_duplicate_candidate_indices_are_counted_once():
    s1 = np.array([1.25, 1.23, 1.22, 1.20, 1.19, 1.17, 1.15,
                   0.05, 0.04, 0.03])
    r = verify_candidate(s1, candidate=[0, 0, 1, 1, 2, 3, 4, 5, 6])
    assert r.n_candidate == 7
    assert r.verdict is Verdict.AGREES


def test_the_plateau_settings_are_forwarded():
    """A looser plateau length finds plateaus a stricter one would not."""
    s1 = np.array([1.25, 1.23, 1.22, 1.20, 1.19, 0.05, 0.04, 0.03])
    strict = verify_candidate(s1, candidate=list(range(5)),
                              plateau_values=40, threshold_step=0.01)
    loose = verify_candidate(s1, candidate=list(range(5)),
                             plateau_values=5, threshold_step=0.01)
    assert len(loose.plateau_heights) >= len(strict.plateau_heights)
