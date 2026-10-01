"""The window rule: policy parameters reproduce the analysis default.

Same fixtures as tests/analysis/test_orbitals.py, wrapped in the layer
that owns the *rule*: the analysis report supplies the facts, the policy
function turns them into the window triple, and with default parameters
the two must agree exactly (the analysis layer's suggestion is the
rule's default, not a separate implementation).
"""
import numpy as np

from diffmrc.analysis.orbitals import frontier_report, heavy_atoms
from diffmrc.policy.windows import suggest_window

_LABELS = ["0 C 2px", "0 C 2py", "0 C 2pz",
           "1 C 2px", "1 C 2py", "1 C 2pz",
           "0 O 2px", "0 O 2py", "0 O 2pz"]


def _C(columns):
    C = np.zeros((len(_LABELS), len(columns)))
    for k, col in enumerate(columns):
        for i, v in col.items():
            C[i, k] = v
    return C


def _report(columns, energies, nocc, target, nelectron):
    rep = dict(frontier_report(_C(columns), _LABELS, np.asarray(energies),
                               nocc, heavy_atoms(_LABELS), target=target,
                               nelectron=nelectron))
    rep['nocc'] = nocc
    return rep


def test_the_rule_reproduces_the_default_on_a_lone_pair_frontier():
    rep = _report([{1: .5, 4: .5}, {6: 1.0}, {1: .5, 4: -.5}],
                  [-0.5, -0.45, 0.1], 2, 'npi', 4)
    got = suggest_window(rep, family='npi', nelectron=4)
    assert got == {'nact': 2, 'pattern': 'adjacent', 'ncore': 1, 'nele': 2}
    assert got['ncore'] == rep['suggested_window']['ncore']
    assert got['nact'] == rep['suggested_window']['nact']
    assert got['nele'] == rep['suggested_window']['nele']


def test_the_rule_reproduces_the_default_on_a_crossed_frontier():
    rep = _report([{0: .5, 3: .5}, {1: .5, 4: .5},
                   {0: .5, 3: -.5}, {1: .5, 4: -.5}],
                  [-0.5, -0.5, 0.1, 0.1], 2, 'ppi', 8)
    got = suggest_window(rep, family='ppi', nelectron=8)
    assert got == {'nact': 3, 'pattern': 'cross', 'ncore': 0, 'nele': 8}
    assert (got['ncore'], got['nact'], got['nele']) == (
        rep['suggested_window']['ncore'], rep['suggested_window']['nact'],
        rep['suggested_window']['nele'])


def test_the_rule_reproduces_the_default_on_the_amide_reach_case():
    rep = _report([{6: 1.0}, {1: .5, 4: .5}, {1: .5, 4: -.5}],
                  [-0.6, -0.45, 0.1], 2, 'npi', 4)
    got = suggest_window(rep, family='npi', nelectron=4)
    assert got == {'nact': 3, 'pattern': 'reach', 'ncore': 0, 'nele': 4}
    assert got['ncore'] == rep['suggested_window']['ncore']


def test_the_parameters_are_the_policy_lever():
    rep = _report([{0: .5, 3: .5}, {1: .5, 4: .5},
                   {0: .5, 3: -.5}, {1: .5, 4: -.5}],
                  [-0.5, -0.5, 0.1, 0.1], 2, 'ppi', 8)
    got = suggest_window(rep, family='ppi', nelectron=8, cross_window=4)
    assert got['nact'] == 4


def test_without_the_space_bookkeeping_the_rule_still_names_the_pattern():
    rep = _report([{1: .5, 4: .5}, {6: 1.0}, {1: .5, 4: -.5}],
                  [-0.5, -0.45, 0.1], 2, 'npi', 4)
    rep.pop('nocc')
    got = suggest_window(rep, family='npi', nelectron=4)
    assert got['pattern'] == 'adjacent'
    assert got['nact'] == 2
    assert got['ncore'] is None and got['nele'] is None
