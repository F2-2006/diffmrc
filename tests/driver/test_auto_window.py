"""auto_window extraction and forwarding, on hand-built objects.

The analysis rules themselves are tested in tests/analysis; these tests
pin the driver-side contract: which quantities are pulled off the
molecule and mean-field objects, and that the family and the electron
count travel through to the suggestion.  No pyscfad is needed -- the
mean field is a stand-in carrying the two arrays the extraction reads.
"""
import numpy as np
import pytest

pytest.importorskip("jax")     # the driver imports the jax-backed kernel

from diffmrc.driver.orbitals import auto_window  # noqa: E402

_LABELS = ["0 C 2px", "0 C 2py", "0 C 2pz",
           "1 C 2px", "1 C 2py", "1 C 2pz",
           "0 O 2px", "0 O 2py", "0 O 2pz"]


class _FakeMol:
    def __init__(self, nelectron):
        self.nelectron = nelectron

    def ao_labels(self):
        return list(_LABELS)


class _FakeMF:
    def __init__(self, coeff, energy):
        self.mo_coeff = coeff
        self.mo_energy = energy


def _coeff(columns):
    C = np.zeros((len(_LABELS), len(columns)))
    for k, col in enumerate(columns):
        for i, v in col.items():
            C[i, k] = v
    return C


def test_auto_window_forwards_a_lone_pair_frontier():
    # MO0 pi, MO1 lone (HOMO), MO2 pi* -- the standard npi case
    C = _coeff([{1: .5, 4: .5}, {6: 1.0}, {1: .5, 4: -.5}])
    e = np.array([-0.5, -0.45, 0.1])
    rep = auto_window(_FakeMol(4), _FakeMF(C, e), 'npi')
    assert rep['homo']['kind'] == 'lone'
    assert rep['homo']['centre'] == '0 O'
    assert rep['suggested_window'] == {'ncore': 1, 'nact': 2, 'nele': 2}


def test_auto_window_widens_a_degenerate_two_pi_frontier():
    # HOMO/HOMO-1 degenerate on x and y, LUMO on x: the naive pair
    # crosses components and the suggestion is three orbitals.
    C = _coeff([{0: .5, 3: .5}, {1: .5, 4: .5},
                {0: .5, 3: -.5}, {1: .5, 4: -.5}])
    e = np.array([-0.5, -0.5, 0.1, 0.1])
    rep = auto_window(_FakeMol(4), _FakeMF(C, e), 'ppi')
    assert rep['cross_component']
    assert rep['suggested_window'] == {'ncore': 0, 'nact': 3, 'nele': 4}
