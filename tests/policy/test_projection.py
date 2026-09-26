"""Subspace-projection selection rules: AVAS and the iCAS probe.

Both rules project coefficient vectors onto a target subspace in the
overlap metric; AVAS targets atomic orbitals, the iCAS probe targets
arbitrary probe directions.  The tests use an orthonormal metric
(S = I) and identity coefficients, where the spectra are exact 0/1.
"""
import numpy as np
import pytest

from diffmrc.policy.projection import (
    AvasChoice,
    avas_selection,
    probe_candidate,
)
from diffmrc.policy.rules import Candidate


def test_avas_selects_the_slots_that_overlap_the_target():
    S = np.eye(4)
    C = np.eye(4)
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    # target = AO 0 (reached by occupied slot 0) and AO 3 (reached by
    # virtual slot 2); active indexes the rotated slots.
    choice = avas_selection(C, S, [0, 3], occ)
    assert isinstance(choice, AvasChoice)
    assert choice.active == [0, 2]
    assert choice.n_electrons == 2
    assert np.allclose(choice.C_rotated[:, 2], [0, 0, 0, 1])


def test_avas_drops_weak_overlaps_with_the_threshold():
    # A deterministic rotation puts 99% of target AO 0 into the occupied
    # block and 1% into the virtual block: the strong side is kept, the
    # 1% side is below the threshold and dropped.
    c_ = np.sqrt(0.99)
    s_ = np.sqrt(0.01)
    Q = np.array([[c_, 0.0, s_, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [-s_, 0.0, c_, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    choice = avas_selection(Q, np.eye(4), [0], occ)
    assert choice.active == [0]
    assert choice.n_electrons == 2
    assert choice.spectra_virtual.max() < 0.05


def test_avas_refuses_open_shells():
    with pytest.raises(ValueError):
        avas_selection(np.eye(3), np.eye(3), [0], np.array([2.0, 1.0, 0.0]))


def test_the_probe_rule_picks_one_orbital_per_probe_with_a_gap():
    S = np.eye(3)
    C = np.eye(3)
    c = probe_candidate(C, S, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert isinstance(c, Candidate)
    assert c.active == [0, 2]
    assert any('gap' in line for line in c.evidence)


def test_a_near_degenerate_probe_gap_refuses():
    probe = [[1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]]
    assert probe_candidate(np.eye(2), np.eye(2), probe, tol=0.5) is None
