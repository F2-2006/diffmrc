"""Multireference diagnostics from the entropy spectrum."""
import numpy as np
import pytest

from diffmrc.analysis.diagnostics import (
    SINGLE_REFERENCE_THRESHOLD,
    WEAK_CORRELATION_THRESHOLD,
    is_single_reference,
    zs1,
)

LN4 = np.log(4.0)


def test_zs1_of_maximally_entangled_orbitals_is_one():
    """All orbitals at ln4 saturates the diagnostic."""
    s1 = np.full(6, LN4)
    assert zs1(s1) == pytest.approx(1.0, abs=1e-12)


def test_zs1_of_noninteracting_orbitals_is_zero():
    assert zs1(np.zeros(6)) == pytest.approx(0.0, abs=1e-12)


def test_zs1_scales_linearly():
    a = zs1(np.full(4, 0.4))
    b = zs1(np.full(4, 0.8))
    assert b == pytest.approx(2.0 * a, rel=1e-12)


def test_zs1_is_independent_of_orbital_count_at_fixed_filling():
    """Doubling the spectrum at the same per-orbital entropy leaves Zs(1)."""
    assert zs1(np.full(4, 0.5)) == pytest.approx(zs1(np.full(8, 0.5)),
                                                 rel=1e-12)


def test_zs1_rejects_empty_input():
    with pytest.raises(ValueError):
        zs1(np.array([]))


def test_thresholds_have_the_documented_defaults():
    assert SINGLE_REFERENCE_THRESHOLD == pytest.approx(0.14)
    assert WEAK_CORRELATION_THRESHOLD == pytest.approx(0.02)


def test_low_entropy_system_is_single_reference():
    assert is_single_reference(np.array([0.05, 0.04, 0.03]))


def test_high_entropy_system_is_multireference():
    assert not is_single_reference(np.array([1.30, 1.28, 1.27]))


def test_empty_spectrum_is_treated_as_single_reference():
    """No active orbitals means nothing justifies a multireference treatment."""
    assert is_single_reference(np.array([]))


def test_threshold_argument_is_honoured():
    s1 = np.array([0.20, 0.10])
    assert not is_single_reference(s1, threshold=0.14)
    assert is_single_reference(s1, threshold=0.25)
