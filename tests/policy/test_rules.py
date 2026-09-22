"""Rule-based candidate generation from chemical structure."""
import numpy as np
import pytest

from diffmrc.policy.rules import (
    SHELL_RULES,
    Candidate,
    candidate_from_elements,
    plan_from_elements,
    dominant_shell,
)


class _FakeMol:
    """Enough of a Mole for the composition check to run."""

    def __init__(self, labels):
        self._labels = labels

    def ao_labels(self):
        return self._labels

    def intor(self, name):
        return np.eye(len(self._labels))


def _labels_for(n):
    return [f'{i} Eu 1s' for i in range(n)]


def test_shell_rules_cover_lanthanide_and_main_group():
    assert 'lanthanide_f' in SHELL_RULES
    assert 'main_group_p' in SHELL_RULES
    assert SHELL_RULES['lanthanide_f']['n_orbitals'] == 7
    assert SHELL_RULES['main_group_p']['n_orbitals'] == 3


def test_candidate_for_a_bare_lanthanide_atom():
    """Seven singly occupied orbitals are the half-filled f shell."""
    occ = np.array([2.0] * 28 + [1.0] * 7 + [0.0] * 5)
    c = candidate_from_elements(_FakeMol(_labels_for(len(occ))),
                                np.eye(len(occ)), occ,
                                center='Eu', center_n=7)
    assert isinstance(c, Candidate)
    assert c.active == list(range(28, 35))
    assert c.per_element['Eu']['n_orbitals'] == 7
    assert c.evidence


def test_candidate_refuses_when_the_center_shell_is_incomplete():
    """Four singly occupied orbitals cannot be a half-filled f shell."""
    occ = np.array([2.0] * 5 + [1.0] * 4 + [0.0] * 3)
    c = candidate_from_elements(_FakeMol(_labels_for(len(occ))),
                                np.eye(len(occ)), occ,
                                center='Eu', center_n=7)
    assert c is None


def test_candidate_refuses_when_there_are_too_many_singly_occupied():
    """Eight is not seven either; a near-miss must not be accepted."""
    occ = np.array([2.0] * 5 + [1.0] * 8 + [0.0] * 2)
    c = candidate_from_elements(_FakeMol(_labels_for(len(occ))),
                                np.eye(len(occ)), occ,
                                center='Eu', center_n=7)
    assert c is None


def test_candidate_for_a_ligand_uses_the_valence_rule():
    """A free main-group atom's valence shell.

    This is the atom-end rule and it is what the ligand branch
    implements. Its limits are documented in the module: it does not
    describe the same atom inside a molecule, where the occupation
    pattern is different.
    """
    occ = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0])
    labels = ['0 F 1s', '1 F 2s', '2 F 2px', '3 F 2py', '4 F 2pz',
              '5 F 3s', '6 F 3px']
    c = candidate_from_elements(_FakeMol(labels), np.eye(len(labels)), occ,
                                center='Eu', ligand='F', ligand_n=3)
    assert c.active == [2, 3, 4]
    assert c.per_element['F']['n_orbitals'] == 3


def test_the_lanthanide_rule_refuses_a_main_group_atom():
    """Seven singly occupied orbitals is a lanthanide signature, not
    fluorine's; the centre path must not accept whatever it is given."""
    occ = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0])
    labels = ['0 F 1s', '1 F 2s', '2 F 2px', '3 F 2py', '4 F 2pz',
              '5 F 3s', '6 F 3px']
    c = candidate_from_elements(_FakeMol(labels), np.eye(len(labels)), occ,
                                center='F', center_n=3)
    assert c is None


# The measured structure of EuF at the molecular end (t176, 2026-09-21):
# thirty doubly occupied orbitals, then eight singly occupied, of which
# seven are the europium f shell and one carries the fluorine.
#
# Note what this means for `select_valence`. On a free fluorine atom the
# valence shell is two doubly occupied orbitals plus one singly occupied
# one, which is what that function assumes. In the molecule the picture
# is different: fluorine's share appears as a *single* singly occupied
# orbital, and there is no pair of doubly occupied orbitals to take. So
# the valence rule is an atom-end rule, and the molecule needs a
# different identification -- see the module note below.
EUF_MOL_OCC = np.array([2.0] * 30 + [1.0] * 8 + [0.0] * 4)
EUF_MOL_SINGLY = list(range(30, 38))


def test_the_molecular_end_is_not_described_by_the_valence_rule():
    """A documented limitation, pinned so it cannot regress silently.

    `select_valence` looks for the valence shell as some singly occupied
    orbitals plus the highest doubly occupied ones. That describes a free
    main-group atom. In a molecule the ligand's share appears differently
    -- for fluorine in EuF it is one singly occupied orbital, with no
    doubly occupied pair to take -- so the rule returns None. The
    molecular case is handled by `dominant_shell` instead, which labels
    orbitals by composition.
    """
    c = candidate_from_elements(
        _FakeMol([f'{i} Eu 1s' for i in range(len(EUF_MOL_OCC))]),
        np.eye(len(EUF_MOL_OCC)), EUF_MOL_OCC,
        center='Eu', ligand='F', ligand_n=3)
    assert c is None


def test_the_centre_alone_is_still_identified_in_the_molecule():
    """Without naming a ligand, the eight singly occupied orbitals do not
    match the seven-orbital f shell either, so the call refuses."""
    c = candidate_from_elements(
        _FakeMol([f'{i} Eu 1s' for i in range(len(EUF_MOL_OCC))]),
        np.eye(len(EUF_MOL_OCC)), EUF_MOL_OCC, center='Eu', center_n=7)
    assert c is None


def test_candidate_evidence_names_the_structure():
    occ = np.array([2.0] * 5 + [1.0] * 7)
    c = candidate_from_elements(_FakeMol(_labels_for(len(occ))),
                                np.eye(len(occ)), occ, center='Eu', center_n=7)
    joined = ' '.join(c.evidence).lower()
    assert 'singly occupied' in joined
    assert 'f shell' in joined


def test_candidate_records_the_number_of_functions_matched():
    """The caller needs to know whether the basis actually had f functions."""
    occ = np.array([2.0] * 5 + [1.0] * 7)
    labels = ['0 Eu 1s'] * 5 + ['5 Eu 4f-3'] * 7
    c = candidate_from_elements(_FakeMol(labels), np.eye(len(labels)), occ,
                                center='Eu', center_n=7)
    assert c.per_element['Eu']['n_ao_f'] == 7


def test_candidate_notes_a_basis_without_f_functions():
    """A rule that matches on structure alone still runs, but says so."""
    occ = np.array([2.0] * 5 + [1.0] * 7)
    c = candidate_from_elements(_FakeMol(_labels_for(len(occ))),
                                np.eye(len(occ)), occ, center='Eu', center_n=7)
    assert c.per_element['Eu']['n_ao_f'] == 0
    assert any('no f functions' in e for e in c.evidence)


def test_candidate_ligand_refuses_when_the_shell_is_too_large():
    occ = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    c = candidate_from_elements(_FakeMol(_labels_for(len(occ))),
                                np.eye(len(occ)), occ,
                                center='F', center_n=3)
    assert c is None




# ---------------------------------------------------------------------
# Labelling orbitals by their dominant composition.
#
# The weights below were measured on EuF's molecular end (t176,
# 2026-09-22) over eight singly occupied orbitals. All eight are singly
# occupied, so nothing about the occupations separates them; the
# composition does.
# ---------------------------------------------------------------------
EUF_SINGLY = [32, 33, 34, 35, 36, 37, 38, 39]
EUF_WEIGHTS = {
    'Eu-s': {32: 0.0000, 33: 0.0000, 34: 0.0000, 35: 0.0000,
             36: 0.8661, 37: 0.0000, 38: 0.0000, 39: 0.0000},
    'Eu-d': {32: 0.0005, 33: 0.0005, 34: 0.0028, 35: 0.0000,
             36: 0.0418, 37: 0.0000, 38: 0.0000, 39: 0.9966},
    'Eu-f': {32: 0.9984, 33: 0.9984, 34: 0.9972, 35: 0.9999,
             36: 0.0001, 37: 0.9999, 38: 0.9999, 39: 0.0039},
    'F-p': {32: 0.0011, 33: 0.0011, 34: 0.0000, 35: 0.0000,
            36: 0.0092, 37: 0.0000, 38: 0.0000, 39: 0.0000},
}


def _shell_weights(overrides=None, n=40):
    out = {}
    for name, mapping in EUF_WEIGHTS.items():
        w = np.zeros(n)
        for k, v in mapping.items():
            w[k] = v
        if overrides and name in overrides:
            for k, v in overrides[name].items():
                w[k] = v
        out[name] = w
    return out


def test_dominant_shell_labels_the_measured_orbitals():
    """Six f, one s, one d -- which is what the chemistry says.

    This is the labelling that both a plain threshold and a two-shell
    ratio get wrong. A threshold of 0.5 on the f weight accepts 48 of
    EuF's 76 orbitals; the ratio f/(f+p) labels orbital 39 as f because
    both its weights are near zero and the ratio is 0.0039/0.0039.
    """
    labels = dominant_shell(EUF_SINGLY, _shell_weights())
    assert labels[32] == 'Eu-f'
    assert labels[33] == 'Eu-f'
    assert labels[34] == 'Eu-f'
    assert labels[35] == 'Eu-f'
    assert labels[37] == 'Eu-f'
    assert labels[38] == 'Eu-f'
    assert labels[36] == 'Eu-s'
    assert labels[39] == 'Eu-d'


def test_dominant_shell_counts_six_f_orbitals_not_seven():
    """The measured count contradicts the atom-end assumption.

    A free europium atom shows seven singly occupied f orbitals. Its
    molecular end shows six, with the 6s and 5d taking two of the eight
    singly occupied slots. A rule that assumes seven and hunts for
    orbitals to fill the count is what made the earlier attempts place
    the wrong orbitals.
    """
    labels = dominant_shell(EUF_SINGLY, _shell_weights())
    n_f = sum(1 for v in labels.values() if v == 'Eu-f')
    assert n_f == 6


def test_dominant_shell_refuses_a_mixed_orbital():
    """No shell above the threshold means the label would be arbitrary."""
    weights = _shell_weights(overrides={'Eu-f': {32: 0.3}})
    labels = dominant_shell([32], weights, min_weight=0.5)
    assert labels[32] is None


def test_dominant_shell_respects_the_threshold():
    weights = _shell_weights()
    strict = dominant_shell([36], weights, min_weight=0.9)
    loose = dominant_shell([36], weights, min_weight=0.5)
    assert strict[36] is None          # 0.866 is below 0.9
    assert loose[36] == 'Eu-s'


def test_dominant_shell_handles_an_empty_candidate_list():
    assert dominant_shell([], _shell_weights()) == {}


def test_dominant_shell_ignores_shells_not_supplied():
    """Only the shells given are considered."""
    weights = {'Eu-f': _shell_weights()['Eu-f']}
    labels = dominant_shell([36], weights)
    assert labels[36] is None          # nothing dominates in this subset
