"""End-to-end use of the analysis layer on hand-built density matrices.

These tests exercise the whole analysis path -- density matrices to
entropies to plateau detection to diagnostics -- without touching the
numerical kernel, which is the point of keeping the layer separate.

The numbers mirror the shape of the f-element problem this project targets:
a set of nearly degenerate, strongly correlated orbitals sitting above a
closed shell, which is what makes a multireference treatment necessary.
"""
import numpy as np

from diffmrc.analysis.diagnostics import is_single_reference, zs1
from diffmrc.analysis.plateau import detect_plateaus
from diffmrc.analysis.rdm import entropy_from_rdms


def _rdms(occ_a, occ_b, p2_diag):
    """Assemble spin-summed density matrices with prescribed occupations.

    ``p2_diag`` is the probability of double occupation; the stored
    two-particle diagonal is twice that, following the kernel's
    convention.
    """
    ncas = len(occ_a)
    rdm2 = np.zeros((ncas, ncas, ncas, ncas))
    for i in range(ncas):
        rdm2[i, i, i, i] = 2.0 * p2_diag[i]
    return [np.diag(occ_a), np.diag(occ_b)], rdm2


def test_weakly_correlated_system_is_flagged_single_reference():
    """Occupations pinned at 0 or 1 give vanishing entropy."""
    occ_a = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    occ_b = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    p2 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rdm1s, rdm2 = _rdms(occ_a, occ_b, p2)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert is_single_reference(s1)
    assert zs1(s1) < 0.2


def test_strongly_correlated_system_produces_a_plateau_and_flags_multireference():
    """Fluctuating occupations on the first four orbitals build entropy."""
    ncas = 8
    occ_a = [0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 0.0, 0.0]
    occ_b = [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0]
    p2 = [0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 0.0, 0.0]
    rdm1s, rdm2 = _rdms(occ_a, occ_b, p2)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert not is_single_reference(s1)
    assert zs1(s1) > 0.1
    plateaus, normalized = detect_plateaus(s1, plateau_values=5,
                                           threshold_step=0.01)
    # The four fluctuating orbitals stand above the four pinned ones.
    assert 4 in plateaus
    assert np.isclose(normalized[0], 1.0)


def test_pinned_orbitals_carry_no_entropy_in_a_correlated_system():
    """The closed shell below the active window must not contribute.

    A doubly occupied orbital reported with entropy would inflate the
    plateau height and pull spurious orbitals into the active space. This
    is the failure the spin-summed halving prevents.
    """
    ncas = 6
    occ_a = [1.0, 1.0, 0.6, 0.6, 0.0, 0.0]
    occ_b = [1.0, 1.0, 0.6, 0.6, 0.0, 0.0]
    p2 = [1.0, 1.0, 0.2, 0.2, 0.0, 0.0]
    rdm1s, rdm2 = _rdms(occ_a, occ_b, p2)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert np.allclose(s1[:2], 0.0, atol=1e-12)
    assert np.allclose(s1[4:], 0.0, atol=1e-12)
    assert np.all(s1[2:4] > 0.1)


def test_plateau_height_matches_the_number_of_correlated_orbitals():
    """Seven strongly correlated orbitals above a gap, as in a 4f shell."""
    ncas = 12
    occ_a = [1.0] * 4 + [0.62] * 7 + [0.0]
    occ_b = [1.0] * 4 + [0.55] * 7 + [0.0]
    p2 = [1.0] * 4 + [0.18] * 7 + [0.0]
    rdm1s, rdm2 = _rdms(occ_a, occ_b, p2)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    plateaus, _ = detect_plateaus(s1, plateau_values=10, threshold_step=0.01)
    assert plateaus, 'expected at least one plateau'
    assert plateaus[0] == 7
    assert not is_single_reference(s1)


def test_analysis_layer_needs_nothing_but_numpy():
    """The layer runs on plain lists, with no kernel objects involved."""
    occ_a = [0.7, 1.0]
    occ_b = [0.5, 1.0]
    p2 = [0.2, 1.0]
    rdm1s = [np.diag(occ_a), np.diag(occ_b)]
    rdm2 = np.zeros((2, 2, 2, 2))
    for i in range(2):
        rdm2[i, i, i, i] = 2.0 * p2[i]
    s1 = entropy_from_rdms(rdm1s.tolist() if hasattr(rdm1s, 'tolist')
                           else rdm1s, rdm2)
    assert s1.shape == (2,)
    assert np.all(np.isfinite(s1))
