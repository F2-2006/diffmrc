"""Consistent active spaces along a path: the matching and the transport.

The criterion, the non-matchable set and the tau sweep follow Bensberg &
Reiher, JPCL 2023 (eqs. 1 and 8 in the card's numbering): two orbitals
match when both the kinetic-energy difference and the shell-population
difference are below their tolerances; orbitals without a match are the
changing (bond-making/breaking) ones and must stay active; the threshold
suggestion is the minimum of the non-matchable count over tau.  All
fixtures here are synthetic signatures, so every expectation is
computable by hand.
"""
import numpy as np
import pytest

from diffmrc.policy.mapping import (
    OrbitalMapping,
    match_orbitals,
    path_consistent_candidate,
    tau_sweep,
)
from diffmrc.policy.rules import Candidate


def test_identical_signatures_match_one_to_one():
    kin = np.array([10.0, 1.0, 0.5])
    q = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    m = match_orbitals(kin, q, kin, q, tau_kin=1e-9, tau_loc=1e-9)
    assert isinstance(m, OrbitalMapping)
    assert [p[:2] for p in m.pairs] == [(0, 0), (1, 1), (2, 2)]
    assert m.unmatched_a == [] and m.unmatched_b == []
    assert m.evidence


def test_a_changed_orbital_lands_in_the_non_matchable_set():
    kin_a = np.array([10.0, 1.0])
    kin_b = np.array([10.001, 1.5])
    q = np.array([[1.0, 0.0], [0.0, 0.0]])
    m = match_orbitals(kin_a, q, kin_b, q, tau_kin=0.01, tau_loc=1e-3)
    assert [p[:2] for p in m.pairs] == [(0, 0)]
    assert m.unmatched_a == [1] and m.unmatched_b == [1]


def test_the_transport_adds_the_non_matchable_orbitals():
    kin_a = np.array([10.0, 1.0])
    kin_b = np.array([10.001, 1.5])
    q = np.array([[1.0, 0.0], [0.0, 0.0]])
    c = path_consistent_candidate(kin_a, q, kin_b, q, [0],
                                  tau_kin=0.01, tau_loc=1e-3)
    assert isinstance(c, Candidate)
    assert c.active == [0, 1]
    # The anchor's unmatched orbital was not in the base: the base
    # missed a changing orbital and the evidence must say so.
    assert any('missed' in line for line in c.evidence)


def test_a_base_that_covers_everything_carries_no_warning():
    kin_a = np.array([10.0, 1.0])
    kin_b = np.array([10.001, 1.5])
    q = np.array([[1.0, 0.0], [0.0, 0.0]])
    c = path_consistent_candidate(kin_a, q, kin_b, q, [0, 1],
                                  tau_kin=0.01, tau_loc=1e-3)
    assert c.active == [0, 1]
    assert not any('missed' in line for line in c.evidence)


def test_the_margin_can_refuse_the_transport():
    kin_a = np.array([10.0, 1.0])
    kin_b = np.array([10.001, 1.5])
    q = np.array([[1.0, 0.0], [0.0, 0.0]])
    out = path_consistent_candidate(kin_a, q, kin_b, q, [0],
                                    tau_kin=0.01, tau_loc=1e-3,
                                    max_nonmatchable=1)
    assert out is None


def test_the_tau_sweep_reports_counts_and_finds_the_minimum():
    kin_a = np.array([0.0, 0.5, 1.0])
    kin_b = np.array([0.02, 0.55, 3.0])
    q = np.zeros((3, 1))
    rows = tau_sweep(kin_a, q, kin_b, q, [0.01, 0.06, 0.2])
    assert rows[0] == pytest.approx((0.01, 3, 3))
    assert rows[1] == pytest.approx((0.06, 1, 1))
    assert rows[2] == pytest.approx((0.2, 1, 1))
    best = min(rows, key=lambda r: r[1] + r[2])
    assert best[0] == pytest.approx(0.06)


def test_ties_resolve_to_the_smaller_index_deterministically():
    kin_a = np.array([1.0])
    q_a = np.array([[0.5]])
    kin_b = np.array([1.0, 1.0])
    q_b = np.array([[0.5], [0.5]])
    m = match_orbitals(kin_a, q_a, kin_b, q_b, tau_kin=0.1, tau_loc=0.1)
    assert [p[:2] for p in m.pairs] == [(0, 0)]


def test_ill_formed_inputs_are_rejected():
    kin = np.array([1.0, 2.0])
    q = np.zeros((2, 1))
    with pytest.raises(ValueError):
        match_orbitals(kin, q, kin, q, tau_kin=0.0, tau_loc=0.1)
    with pytest.raises(ValueError):
        match_orbitals(kin, q[:1], kin, q, tau_kin=0.1, tau_loc=0.1)
    with pytest.raises(ValueError):
        match_orbitals(kin, q, kin, np.zeros((2, 2)),
                       tau_kin=0.1, tau_loc=0.1)
    with pytest.raises(ValueError):
        path_consistent_candidate(kin, q, kin, q, [],
                                  tau_kin=0.1, tau_loc=0.1)
