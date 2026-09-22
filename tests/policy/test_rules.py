"""Rule-based candidate generation from chemical structure."""
import numpy as np
import pytest

from diffmrc.policy.rules import SHELL_RULES, Candidate, candidate_from_elements


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
    """Fluorine's valence shell, on the structure measured for it.

    A main-group atom is reached through the `ligand` argument; the bare
    `center` path is the lanthanide rule and correctly refuses fluorine.
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


def test_candidate_accepts_a_center_with_an_explicit_ligand():
    """Both halves of a diatomic can be proposed in one call."""
    occ = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 0.0])
    labels = ['0 O 1s', '1 O 2s', '2 O 2px', '3 O 2py', '4 O 2pz', '5 O 3s']
    c = candidate_from_elements(_FakeMol(labels), np.eye(len(labels)), occ,
                                center='Eu', ligand='O', ligand_n=3)
    assert c.active == [2, 3, 4]
    assert 'O' in c.per_element
