"""The report's content, checked on hand-built data."""
import json

from diffmrc.report.summary import ReportData, render_json, render_text


def _data(**kw):
    base = dict(
        energy=-76.0, geometry='O 0 0 0; H 0.76 0 0.59; H -0.76 0 0.59',
        frequencies=[1595.0, 3657.0, 3756.0],
        active_space=dict(orbitals=[2, 3, 4], n_electrons=5,
                          evidence=['F: valence shell', 'plateau 3 matches']),
        checks=[dict(name='orbital_gradient', passed=True,
                     detail='|g| = 1.0e-10 (threshold 1e-08)'),
                dict(name='hessian_curvature', passed=True,
                     detail='0 negative eigenvalues')],
        decision=dict(level='fast', reasons=[], suggestions=[]),
        provenance=dict(params={'gradient_tol': 1e-8},
                        sources={'gradient_tol': 'default'},
                        versions={'diffmrc': '0.1.0'}, geometry_input='water'),
    )
    base.update(kw)
    return ReportData(**base)


def test_text_report_states_the_energy_and_geometry():
    out = render_text(_data())
    assert '-76.000000' in out
    assert 'O 0 0 0' in out


def test_text_report_lists_every_check_with_its_value():
    """Per CP2K's geometry optimizer, each criterion is printed with its
    measured value, its threshold and a yes/no -- not one summary flag."""
    out = render_text(_data())
    assert 'orbital_gradient' in out
    assert '1.0e-10' in out
    assert 'hessian_curvature' in out


def test_text_report_marks_each_check_pass_or_fail():
    d = _data(checks=[dict(name='orbital_gradient', passed=False,
                           detail='|g| = 1.0e+00 (threshold 1e-08)'),
                      dict(name='hessian_curvature', passed=True,
                           detail='0 negative eigenvalues')])
    out = render_text(d)
    assert 'YES' in out and 'NO' in out


def test_text_report_states_the_active_space_and_why():
    out = render_text(_data())
    assert '[2, 3, 4]' in out
    assert 'valence shell' in out


def test_text_report_says_which_parameters_the_user_chose():
    """A reader must be able to tell a surprising result from a
    surprising input."""
    out = render_text(_data())
    assert 'gradient_tol' in out
    assert 'default' in out


def test_a_refusal_leads_with_the_refusal():
    """A reader must not have to reach the end to learn the number is
    not to be trusted."""
    d = _data(
        energy=float('nan'),
        decision=dict(level='refused',
                      reasons=['orbital_gradient: |g| = 1.0e+00'],
                      suggestions=['tighten the orbital optimization']),
    )
    out = render_text(d)
    head = '\n'.join(out.splitlines()[:8]).upper()
    assert 'REFUS' in head


def test_a_refusal_lists_the_reasons_and_the_way_forward():
    d = _data(decision=dict(level='refused',
                            reasons=['hessian_curvature: 3 negative'],
                            suggestions=['run the multi-start search']))
    out = render_text(d)
    assert '3 negative' in out
    assert 'multi-start' in out


def test_json_report_is_valid_and_round_trips():
    parsed = json.loads(render_json(_data()))
    assert parsed['energy'] == -76.0
    assert parsed['active_space']['orbitals'] == [2, 3, 4]


def test_json_report_keeps_every_check():
    parsed = json.loads(render_json(_data()))
    assert len(parsed['checks']) == 2
    assert parsed['checks'][0]['name'] == 'orbital_gradient'


def test_json_report_carries_the_provenance():
    parsed = json.loads(render_json(_data()))
    assert 'versions' in parsed['provenance']
    assert parsed['provenance']['sources']['gradient_tol'] == 'default'


def test_a_refusal_is_machine_readable_too():
    d = _data(decision=dict(level='refused', reasons=['x'], suggestions=['y']))
    parsed = json.loads(render_json(d))
    assert parsed['decision']['level'] == 'refused'


def test_missing_frequencies_do_not_break_the_report():
    out = render_text(_data(frequencies=[]))
    assert 'frequenc' in out.lower()


def test_an_imaginary_frequency_is_flagged_in_the_text():
    """A negative frequency means a saddle point, not a molecule."""
    out = render_text(_data(frequencies=[-120.0, 1595.0, 3657.0]))
    assert 'imaginary' in out.lower()


def test_the_json_report_keeps_frequencies_as_numbers():
    parsed = json.loads(render_json(_data()))
    assert parsed['frequencies'] == [1595.0, 3657.0, 3756.0]


def test_an_empty_check_list_is_stated_not_hidden():
    """No checks is not the same as all checks passing."""
    out = render_text(_data(checks=[]))
    assert 'no self-check' in out.lower() or 'none' in out.lower()


def test_the_on_top_block_renders_only_when_present():
    d = _data()
    assert 'E_ot' not in render_text(d)

    d.on_top = {'method': 'mcpdft0',
                'protocol': 'one-step: CASSCF-driven orbitals, one on-top '
                            'evaluation (tPBE), 0.25-mixed (tPBE0)',
                'hyb': 0.25, 'e_casscf': -1.137283834489,
                'e_ot': -0.733794563942, 'e_mcwfn': -0.422833915874}
    text = render_text(d)
    assert 'on-top decomposition' in text
    assert '-0.7337945639' in text
    assert 'hybrid lambda' in text
    payload = json.loads(render_json(d))
    assert payload['on_top']['method'] == 'mcpdft0'
    assert payload['on_top']['hyb'] == 0.25
