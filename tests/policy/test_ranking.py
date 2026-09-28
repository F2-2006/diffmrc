"""The ranked-orbital (APC) selection.

The ranking and the cap follow King & Gagliardi, JCTC 2021 (the
approximate pair coefficient, eqs. 12/13/18/19, and the N_CSF cap of
eq. 2); the source's own preset anchors -- max(7,6) -> 490, (8,8) ->
1764, (10,10) -> 19404, (12,12) -> 226512 -- are reproduced by n_csf.
"""
import numpy as np
import pytest

from diffmrc.policy.ranking import (
    RankedOrbitalChoice,
    n_csf,
    ranked_orbital_selection,
)


def test_n_csf_reproduces_the_source_presets():
    assert n_csf(7, 6) == 490
    assert n_csf(8, 8) == 1764
    assert n_csf(10, 10) == 19404
    assert n_csf(12, 12) == 226512


def test_the_pair_coefficient_and_entropies_follow_the_closed_form():
    # One occupied, one virtual: c = -x / (D + sqrt(x^2 + D^2)).
    choice = ranked_orbital_selection(
        np.array([-0.5]), np.array([0.5]), np.array([0.1]),
        n_csf_cap=3)
    assert isinstance(choice, RankedOrbitalChoice)
    c = -0.1 / (1.0 + np.sqrt(0.01 + 1.0))
    p = c * c / (1.0 + c * c)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert choice.entropies[0] == pytest.approx(h)
    assert choice.entropies[1] == pytest.approx(h)
    assert choice.active == [0, 1]
    assert choice.n_csf == 3


def test_the_cap_trims_the_lowest_ranked_and_respects_the_floor():
    eps_i = np.array([-0.6, -0.5, -0.4])
    eps_a = np.array([0.4, 0.5, 0.6])
    x = np.array([0.30, 0.10, 0.02])
    # (6e, 6o) would give 175 CSFs; the cap forces drops down to <= 50,
    # and each drop must keep >= 1 occupied and >= 2 virtual orbitals.
    choice = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=50)
    assert choice.n_csf <= 50
    assert len(choice.active) < 6
    n_occ = sum(1 for i in choice.active if i < 3)
    n_vir = len(choice.active) - n_occ
    assert n_occ >= 1 and n_vir >= 2
    assert any('dropped' in line for line in choice.evidence)


def test_an_unreachable_cap_refuses_with_evidence():
    choice = ranked_orbital_selection(
        np.array([-0.5, -0.4]), np.array([0.4, 0.8]),
        np.array([0.2, 0.1]), n_csf_cap=1)
    assert choice.active == []
    assert any('cap' in line for line in choice.evidence)


def test_near_equal_entropies_rank_by_index():
    eps_i = np.array([-0.5, -0.5])
    eps_a = np.array([0.5, 0.5])
    x = np.array([0.1, 0.1])
    a = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=50)
    b = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=50,
                                 occupied=[1, 0], virtual=[3, 2])
    assert set(a.entropies) == {0, 1, 2, 3}
    assert set(b.entropies) == {0, 1, 2, 3}
    # The labels follow the caller's index lists; equal inputs keep
    # equal entropies.
    assert a.entropies[0] == pytest.approx(b.entropies[1], abs=1e-12)
    assert a.entropies[1] == pytest.approx(b.entropies[0], abs=1e-12)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        ranked_orbital_selection(np.zeros(2), np.zeros(3), np.zeros(2),
                                 n_csf_cap=100)


def test_singly_occupied_orbitals_enter_unconditionally():
    # The source's open-shell rule: a singly occupied orbital is given
    # the highest entropy and enters unconditionally.  Counts: three
    # occupied + one singly occupied + three virtual give L=7, ele=7 ->
    # n_csf(7,7) = 784; without the singly occupied orbital the same
    # window gives n_csf(6,6) = 175.
    eps_i = np.array([-0.6, -0.5, -0.4])
    eps_a = np.array([0.4, 0.5, 0.6])
    x = np.array([0.30, 0.10, 0.02])
    wide = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=10**6,
                                    singly_occupied=[10])
    assert wide.n_csf == 784
    assert 10 in wide.active
    assert wide.entropies[10] == pytest.approx(np.log(2.0))
    # Under a binding cap the singly occupied orbital is never dropped.
    tight = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=50,
                                     singly_occupied=[10])
    assert tight.n_csf <= 50
    assert 10 in tight.active
    assert len(tight.active) < 7


def test_the_singly_occupied_rule_defaults_off():
    # Without the keyword the counted electron budget excludes any
    # singly occupied orbital, matching the pre-extension behaviour.
    choice = ranked_orbital_selection(np.array([-0.6, -0.5, -0.4]),
                                      np.array([0.4, 0.5, 0.6]),
                                      np.array([0.30, 0.10, 0.02]),
                                      n_csf_cap=10**6)
    assert choice.n_csf == 175


def test_virtual_removal_rebalances_occupied_and_virtual_entropies():
    # The source's bias correction: N times, the highest-entropy virtual
    # is removed from the sums of eqs. 5/6 and the entropies recomputed;
    # the removed orbitals are finally assigned the highest remaining
    # entropy.  Closed-form check on one occupied and two virtuals.
    eps_i = np.array([-0.5])
    eps_a = np.array([0.5, 0.6])
    x = np.array([0.1, 0.2])
    base = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=10**6)
    adj = ranked_orbital_selection(eps_i, eps_a, x, n_csf_cap=10**6,
                                   virtual_removal=1)
    D = np.array([1.0, 1.1])
    c2 = (x / (D + np.sqrt(x * x + D * D))) ** 2

    def h(p):
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    s_i0 = h(c2.sum() / (1 + c2.sum()))
    s_a1 = h(c2[0] / (1 + c2[0]))
    s_a2 = h(c2[1] / (1 + c2[1]))
    assert base.entropies[0] == pytest.approx(s_i0)
    assert base.entropies[1] == pytest.approx(s_a1)
    assert base.entropies[2] == pytest.approx(s_a2)
    # a2 carries the higher virtual entropy and is the one removed.
    assert s_a2 > s_a1
    # After one removal: occupied entropy recomputed without a2, the
    # remaining virtual is unchanged, and a2 takes the highest entropy
    # left (which here is the shared one-virtual value).
    s_i1 = h(c2[0] / (1 + c2[0]))
    assert adj.entropies[0] == pytest.approx(s_i1)
    assert adj.entropies[1] == pytest.approx(s_a1)
    assert adj.entropies[2] == pytest.approx(max(s_i1, s_a1))
    assert adj.entropies[0] < base.entropies[0]


def test_virtual_removal_bounds_are_rejected():
    # At least one virtual must remain in the sums of eqs. 5/6.
    with pytest.raises(ValueError):
        ranked_orbital_selection(np.array([-0.5]), np.array([0.5]),
                                 np.array([0.1]), n_csf_cap=10,
                                 virtual_removal=1)
    with pytest.raises(ValueError):
        ranked_orbital_selection(np.array([-0.5]), np.array([0.5, 0.6]),
                                 np.array([0.1, 0.2]), n_csf_cap=10,
                                 virtual_removal=-1)
