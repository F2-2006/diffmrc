"""The vertical-excitation report, checked on hand-built data."""
import json

from diffmrc.report.vee import VeeReportData, render_vee_json, render_vee_text


def _data():
    return VeeReportData(
        geometry='formaldehyde.xyz',
        family='npi',
        window={'ncore': 7, 'nact': 2, 'nele': 2},
        grad_norm=3.4e-11,
        states=[
            {'index': 0, 'multiplicity': 1, 'szs': [0.0],
             'dE_casscf': 0.0, 'dE_nevpt2': 0.0},
            {'index': 1, 'multiplicity': 3, 'szs': [0.0, 1.0],
             'dE_casscf': 3.55, 'dE_nevpt2': 3.97},
        ],
        frontier={'homo': {'kind': 'lone', 'axis': 'z', 'centre': '1 O'},
                  'homo_minus_1': {'kind': 'pi', 'axis': 'y',
                                   'centre': None},
                  'lumo': {'kind': 'pi', 'axis': 'y', 'centre': None},
                  'cross_component': False, 'lone_below_homo': None,
                  'occupied_degenerate_pairs': []},
        provenance={'basis': '6-31g'},
    )


def test_the_text_report_carries_the_window_and_the_states():
    text = render_vee_text(_data())
    assert 'vertical excitations (npi)' in text
    assert 'ncore=7' in text
    assert '+3.970000' in text       # the NEVPT2 column
    assert 'lone' in text            # the frontier identity


def test_the_text_report_carries_the_suggestion_caveat():
    text = render_vee_text(_data())
    assert 'suggestion' in text
    assert 'not a decision' in text


def test_json_roundtrips_the_plain_fields():
    payload = json.loads(render_vee_json(_data()))
    assert payload['window'] == {'ncore': 7, 'nact': 2, 'nele': 2}
    assert payload['states'][1]['dE_nevpt2'] == 3.97
    assert payload['frontier']['homo']['kind'] == 'lone'
    assert payload['family'] == 'npi'
