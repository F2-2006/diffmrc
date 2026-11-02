"""Curvature-aware fallback decisions.

The fallback space under test is the one the occupation criterion builds
when no element rule covers the composition. It completes degenerate
partners, and that completion can land on an orbital saddle: D3h ozone
converges to a gradient of 1.0e-4 with thirteen negative eigenvalues, so
a retry gate that only reads the gradient never fires. These tests fix
the comparison rules and the evidence they must leave behind, without
running the numerical kernel.
"""
from diffmrc.core.stationary import CurvatureReport
from diffmrc.driver.kernel import (_candidate_evidence, _non_minimum_evidence,
                                   prefer_candidate)


def _record(active, n_negative, min_eigenvalue, gradient, energy):
    return dict(active=list(active), n_negative=int(n_negative),
                min_eigenvalue=float(min_eigenvalue),
                gradient=float(gradient), energy=float(energy))


def test_zero_negative_curvature_beats_a_lower_energy_saddle():
    """A minimum wins even when the saddle is the lower point."""
    saddle = _record([10, 11, 12], 13, -1.5408e-2, 1.0e-4,
                     -224.0547703661)
    minimum = _record([11, 12], 0, 5.5e-2, 5.2e-7, -224.0400000000)

    chosen, reason = prefer_candidate(saddle, minimum)

    assert chosen is minimum
    assert 'minimum' in reason


def test_both_minima_choose_the_lower_energy():
    higher = _record([11, 12], 0, 4.0e-2, 8.0e-7, -224.0400000000)
    lower = _record([10, 11, 12], 0, 4.0e-2, 9.0e-7, -224.0500000000)

    chosen, reason = prefer_candidate(higher, lower)

    assert chosen is lower
    assert '-224.0500000000 < -224.0400000000' in reason


def test_both_saddles_choose_the_fewer_negative_eigenvalues():
    """The rule is curvature count, not energy, when both are saddles."""
    fewer_but_higher = _record([11, 12], 2, -3.995e-2, 5.2e-7,
                               -224.0500000000)
    more_but_lower = _record([10, 11, 12], 13, -1.5408e-2, 1.0e-4,
                             -224.0600000000)

    chosen, reason = prefer_candidate(fewer_but_higher, more_but_lower)

    assert chosen is fewer_but_higher
    assert 'negative' in reason
    assert '2 vs 13' in reason


def test_a_non_stationary_point_never_beats_a_stationary_saddle():
    """Curvature at a point the optimizer did not reach certifies nothing."""
    unconverged = _record([10, 11, 12], 0, 9.0e-2, 1.0e-2, -224.0600000000)
    saddle = _record([11, 12], 3, -4.0e-2, 4.0e-6, -224.0500000000)

    chosen, reason = prefer_candidate(unconverged, saddle)

    assert chosen is saddle
    assert 'stationary' in reason


def test_candidate_evidence_carries_all_four_decision_numbers():
    record = _record([11, 12], 2, -3.995e-2, 5.2e-7, -224.0555435885)

    line = _candidate_evidence(record)

    assert 'n_negative=2' in line
    assert 'min_eigenvalue=-3.995000e-02' in line
    assert 'gradient=5.200e-07' in line
    assert 'energy=-224.0555435885' in line


def test_non_minimum_evidence_names_the_curvature_and_the_missing_partner():
    curvature = CurvatureReport(n_negative=2, smallest=-3.995e-2,
                                largest=8.0e-2, condition_number=2.0,
                                n_near_zero=0)

    text = _non_minimum_evidence([11, 12], curvature, 5.2e-7,
                                 gap=(10, 11, 0.137))

    assert 'not an orbital minimum' in text
    assert '2 negative eigenvalue' in text
    assert '-3.995000e-02' in text
    assert '0.137 Ha' in text
    assert 'near-degenerate partner is frozen in the core' in text
    assert 'too small' in text


def test_non_minimum_evidence_reports_a_split_degenerate_pair():
    """The measured D3h ozone case: 10 and 11 are degenerate, one is core."""
    curvature = CurvatureReport(n_negative=2, smallest=-3.995e-2,
                                largest=8.0e-2, condition_number=2.0,
                                n_near_zero=0)

    text = _non_minimum_evidence([11, 12], curvature, 5.2e-7,
                                 gap=(10, 11, 5.0e-7))

    assert 'is degenerate with the lowest active orbital (11)' in text
    assert '5.000e-07 Ha' in text
    assert 'one component of the degenerate pair is frozen in the core' in text


def test_non_minimum_evidence_reports_an_unconverged_point_as_such():
    curvature = CurvatureReport(n_negative=1, smallest=-1.0e-2,
                                largest=5.0e-2, condition_number=5.0,
                                n_near_zero=0)

    text = _non_minimum_evidence([4, 5], curvature, 2.0e-2)

    assert 'not stationary' in text
    assert 'not a minimum certificate' in text


def test_a_wide_core_gap_is_reported_without_the_near_degenerate_claim():
    curvature = CurvatureReport(n_negative=3, smallest=-5.0e-2,
                                largest=1.0, condition_number=20.0,
                                n_near_zero=0)

    text = _non_minimum_evidence([4, 5], curvature, 1.0e-7,
                                 gap=(3, 4, 0.65))

    assert '0.650 Ha' in text
    assert 'near-degenerate partner is frozen' not in text
