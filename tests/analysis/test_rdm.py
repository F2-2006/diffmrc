"""Bridge from reduced density matrices to single-orbital entropies."""
import numpy as np
import pytest

from diffmrc.analysis.rdm import entropy_from_rdms

LN2 = float(np.log(2.0))


def _determinant_rdms(occ_alpha, occ_beta, ncas, spin_summed=True):
    """Build the density matrices of a single determinant.

    Parameters
    ----------
    spin_summed : bool
        Store the two-particle diagonal as ``2 * n_a * n_b`` when True,
        matching PySCF's ``make_rdm2`` and this project's kernel, and as
        ``n_a * n_b`` when False.
    """
    rdm1a = np.diag(np.asarray(occ_alpha, dtype=float))
    rdm1b = np.diag(np.asarray(occ_beta, dtype=float))
    rdm2 = np.zeros((ncas, ncas, ncas, ncas))
    factor = 2.0 if spin_summed else 1.0
    for i in range(ncas):
        rdm2[i, i, i, i] = factor * occ_alpha[i] * occ_beta[i]
    return [rdm1a, rdm1b], rdm2


def test_closed_shell_determinant_has_zero_entropy():
    """A closed-shell determinant has no entanglement anywhere."""
    ncas = 4
    rdm1s, rdm2 = _determinant_rdms([1, 1, 0, 0], [1, 1, 0, 0], ncas)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert s1.shape == (ncas,)
    assert np.allclose(s1, 0.0, atol=1e-12)


def test_high_spin_determinant_has_zero_entropy():
    """Singly occupied alpha orbitals are unentangled."""
    ncas = 4
    rdm1s, rdm2 = _determinant_rdms([1, 1, 1, 0], [1, 1, 0, 0], ncas)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert np.allclose(s1, 0.0, atol=1e-12)


def test_all_occupied_orbitals_of_a_filled_shell_are_unentangled():
    ncas = 8
    rdm1s, rdm2 = _determinant_rdms([1] * 6 + [0] * 2, [1] * 6 + [0] * 2,
                                    ncas)
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert np.allclose(s1, 0.0, atol=1e-12)


def test_spin_resolved_input_is_accepted_explicitly():
    """The same determinant in the other convention gives the same answer."""
    ncas = 4
    rdm1s, rdm2 = _determinant_rdms([1, 1, 0, 0], [1, 1, 0, 0], ncas,
                                    spin_summed=False)
    s1 = entropy_from_rdms(rdm1s, rdm2, spin_summed=False)
    assert np.allclose(s1, 0.0, atol=1e-12)


def test_halving_is_what_keeps_a_doubly_occupied_orbital_unentangled():
    """The regression that motivates the ``spin_summed`` default.

    A spin-summed two-particle diagonal holds ``2 * n_a * n_b``. A
    doubly occupied orbital therefore carries the value 2, which is not
    a probability. Halved, it becomes 1 and the orbital is correctly
    reported as unentangled. Read as-is, the weight vector clips to a
    half-and-half mixture of the empty and doubly occupied states, and
    the orbital is reported as carrying a spurious ln 2.
    """
    ncas = 2
    rdm1s, rdm2 = _determinant_rdms([1, 0], [1, 0], ncas)
    # The determinant's diagonal entry is 2 * 1 * 1 = 2.
    assert rdm2[0, 0, 0, 0] == pytest.approx(2.0)

    # Correct handling: double occupation is certain, so no entanglement.
    assert np.allclose(entropy_from_rdms(rdm1s, rdm2), 0.0, atol=1e-12)
    # Treating the spin-summed value as a probability reports ln 2.
    spurious = entropy_from_rdms(rdm1s, rdm2, spin_summed=False)
    assert spurious[0] == pytest.approx(LN2, abs=1e-12)


def test_partially_occupied_orbitals_differ_between_the_conventions():
    """The two conventions are genuinely different inputs.

    For a fractionally occupied orbital the halving factor does not
    cancel: the two readings give different entropies. A caller who
    passes spin-summed data with ``spin_summed=False`` gets a plausible
    but wrong number, which is why the default is stated explicitly.
    """
    ncas = 1
    n_a, n_b, p2_true = 0.7, 0.5, 0.2
    rdm1s = [np.diag([n_a]), np.diag([n_b])]
    rdm2_summed = np.zeros((1, 1, 1, 1))
    rdm2_summed[0, 0, 0, 0] = 2.0 * p2_true
    rdm2_resolved = np.zeros((1, 1, 1, 1))
    rdm2_resolved[0, 0, 0, 0] = p2_true

    correct = entropy_from_rdms(rdm1s, rdm2_summed, spin_summed=True)
    also_correct = entropy_from_rdms(rdm1s, rdm2_resolved,
                                     spin_summed=False)
    misread = entropy_from_rdms(rdm1s, rdm2_summed, spin_summed=False)

    assert correct[0] == pytest.approx(also_correct[0], abs=1e-12)
    assert abs(misread[0] - correct[0]) > 1e-3


def test_mismatched_rdm2_shape_raises():
    rdm1s = [np.eye(3), np.eye(3)]
    with pytest.raises(ValueError):
        entropy_from_rdms(rdm1s, np.zeros((3, 3, 3)))


def test_rdm1s_must_have_two_spin_blocks():
    with pytest.raises(ValueError):
        entropy_from_rdms([np.eye(3)], np.zeros((3, 3, 3, 3)))


def test_rdm1s_blocks_must_be_square_and_equal():
    with pytest.raises(ValueError):
        entropy_from_rdms([np.zeros((3, 2)), np.zeros((3, 3))],
                          np.zeros((3, 3, 3, 3)))


def test_off_diagonal_rdm1_does_not_affect_the_entropy():
    """Only the diagonal occupations enter the one-orbital entropy."""
    ncas = 3
    rdm1s, rdm2 = _determinant_rdms([1, 1, 0], [1, 1, 0], ncas)
    baseline = entropy_from_rdms(rdm1s, rdm2)
    rdm1s = [rdm1s[0].copy(), rdm1s[1].copy()]
    rdm1s[0][0, 1] = 0.3
    rdm1s[0][1, 0] = 0.3
    assert np.allclose(entropy_from_rdms(rdm1s, rdm2), baseline, atol=1e-12)


def test_output_lies_in_the_physical_range():
    rng = np.random.default_rng(11)
    ncas = 5
    a = rng.uniform(0, 1, ncas)
    b = rng.uniform(0, 1, ncas)
    rdm1s = [np.diag(a), np.diag(b)]
    rdm2 = np.zeros((ncas, ncas, ncas, ncas))
    for i in range(ncas):
        rdm2[i, i, i, i] = min(a[i], b[i])
    s1 = entropy_from_rdms(rdm1s, rdm2)
    assert np.all(s1 >= 0.0)
    assert np.all(s1 <= np.log(4.0) + 1e-12)
