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
    -- one singly occupied orbital for fluorine in EuF, with no doubly
    occupied pair to take -- so the rule returns None.

    Returning None is the correct behaviour here: the function does not
    know how to identify the ligand's shell in this situation, and
    guessing would place the wrong orbitals in the active space. What
    the molecule needs is a rule that identifies the shell by orbital
    composition rather than by occupation pattern, which is future work.
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
