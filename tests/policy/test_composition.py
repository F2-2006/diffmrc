"""Element and shell composition, and ranking candidates by it."""
import numpy as np
import pytest

from diffmrc.policy.composition import element_shell_weights, rank_by_weight


class _FakeMol:
    """Minimal stand-in so the weighting can be tested without pyscf."""

    def __init__(self, labels, overlap):
        self._labels = labels
        self._overlap = overlap

    def ao_labels(self):
        return self._labels

    def intor(self, name):
        assert name == 'int1e_ovlp'
        return self._overlap


# Label forms taken from a real PySCF run: the shell carries its principal
# quantum number and, for f functions, a magnetic quantum number suffix.
LABELS = ['0 Eu 1s', '1 Eu 4f-3', '2 Eu 4f+0', '3 Eu 5d+0',
          '4 F 2px', '5 F 2py', '6 F 2pz']


def _mol():
    return _FakeMol(LABELS, np.eye(len(LABELS)))


def test_weights_find_the_lanthanide_f_shell():
    """A fixed-position slice finds only the highest f contraction here.

    The ANO f labels run '4f-3' through '10f+3'; taking the third
    character of the shell string matches 'f' only for the two-digit
    contractions, so most of the shell is missed. Matching the letter
    finds all of it.
    """
    C = np.eye(len(LABELS))
    r = element_shell_weights(_mol(), C, 'Eu', 'f')
    assert r.n_ao == 2
    assert r.weights[1] > 0.9
    assert r.weights[2] > 0.9


def test_weights_find_the_ligand_p_shell():
    """The case that produced a core-orbital selection.

    For '2px' the third character is 'x', so a fixed-position slice finds
    no p functions at all for fluorine.
    """
    C = np.eye(len(LABELS))
    r = element_shell_weights(_mol(), C, 'F', 'p')
    assert r.n_ao == 3
    assert r.weights[4] > 0.9
    assert r.weights[5] > 0.9
    assert r.weights[6] > 0.9


def test_weights_do_not_mix_shells_of_the_same_element():
    """Europium's d functions must not appear in its f weight."""
    C = np.eye(len(LABELS))
    r = element_shell_weights(_mol(), C, 'Eu', 'd')
    assert r.n_ao == 1
    assert r.weights[3] > 0.9
    assert r.weights[1] < 0.1


def test_absent_shell_returns_zero_weights_and_zero_count():
    C = np.eye(len(LABELS))
    r = element_shell_weights(_mol(), C, 'F', 'f')
    assert r.n_ao == 0
    assert np.allclose(r.weights, 0.0)


def test_weights_have_one_entry_per_molecular_orbital():
    C = np.eye(len(LABELS))
    r = element_shell_weights(_mol(), C, 'Eu', 'f')
    assert r.weights.shape == (len(LABELS),)


def test_weights_survive_a_contracted_orbital():
    """A molecular orbital spread over two labelled functions of one shell."""
    C = np.zeros((len(LABELS), 1))
    C[1, 0] = np.sqrt(0.5)
    C[2, 0] = np.sqrt(0.5)
    r = element_shell_weights(_mol(), C, 'Eu', 'f')
    assert r.weights[0] == pytest.approx(1.0)


def test_rank_by_weight_takes_the_top_entries():
    w = np.array([0.1, 0.9, 0.8, 0.05])
    r = rank_by_weight([0, 1, 2, 3], w, 2)
    assert r.keep == [1, 2]
    assert r.drop == [0, 3]


def test_rank_by_weight_refuses_when_the_order_is_not_meaningful():
    """All-equal weights carry no information; the ranking is arbitrary.

    This is the guard that turns a silent garbage selection -- argsort
    over zeros returns the index order, which selects the core -- into a
    refusal.
    """
    w = np.zeros(4)
    assert rank_by_weight([0, 1, 2, 3], w, 2) is None


def test_rank_by_weight_refuses_when_keep_and_drop_are_indistinguishable():
    """A lanthanide's f weight spreads over many functions, so the split
    between the seventh and eighth candidate can be arbitrary even when
    the weights are not exactly zero."""
    w = np.array([0.500, 0.499, 0.498, 0.497])
    assert rank_by_weight([0, 1, 2, 3], w, 2, min_ratio=2.0) is None


def test_rank_by_weight_reports_the_discrimination():
    w = np.array([1.0, 0.5, 0.01, 0.005])
    r = rank_by_weight([0, 1, 2, 3], w, 2)
    assert r.discrimination == pytest.approx(0.5 / 0.01)


def test_rank_by_weight_keeps_everything_when_asked_for_all():
    w = np.array([1.0, 0.5, 0.01])
    r = rank_by_weight([0, 1, 2], w, 3)
    assert r.keep == [0, 1, 2]
    assert r.drop == []


def test_rank_by_weight_returns_none_when_short():
    w = np.array([1.0, 0.5])
    assert rank_by_weight([0, 1], w, 3) is None


def test_rank_by_weight_keeps_its_result_sorted():
    """Ascending indices, so the result can be compared across runs."""
    w = np.array([0.1, 0.9, 0.8, 0.05])
    r = rank_by_weight([3, 2, 1, 0], w, 2)
    assert r.keep == sorted(r.keep)
    assert r.drop == sorted(r.drop)


def test_rank_by_weight_handles_a_zero_weight_candidate_in_the_keep_set():
    """Selection is by weight, not by index, so a zero can be kept while
    a nonzero is dropped only if the weights say so."""
    w = np.array([1.0, 0.5, 0.0])
    r = rank_by_weight([0, 1, 2], w, 2)
    assert r.keep == [0, 1]
    assert r.drop == [2]
