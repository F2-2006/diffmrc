"""The command-line surface."""
import pytest

from diffmrc.cli.main import build_parser, main

WATER = '3\nwater\nO 0 0 0\nH 0.76 0 0.59\nH -0.76 0 0.59\n'


def _write(tmp_path, text=WATER, name='mol.xyz'):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_parser_accepts_a_minimal_invocation():
    args = build_parser().parse_args(['mol.xyz'])
    assert args.geometry == 'mol.xyz'


def test_parser_defaults_come_from_the_parameter_table():
    """No default is written twice."""
    from diffmrc.policy.params import defaults
    args = build_parser().parse_args(['mol.xyz'])
    assert args.plateau_values == defaults()['plateau_values']
    assert args.min_entropy == defaults()['min_entropy']
    assert args.gradient_tol == defaults()['gradient_tol']


def test_parser_accepts_charge_and_spin():
    args = build_parser().parse_args(['mol.xyz', '--charge', '0',
                                      '--spin', '8'])
    assert args.charge == 0
    assert args.spin == 8


def test_parser_accepts_a_manual_active_space():
    args = build_parser().parse_args(['mol.xyz', '--active-space', '2-4'])
    assert args.active_space == '2-4'


def test_parser_accepts_a_level():
    args = build_parser().parse_args(['mol.xyz', '--level', 'fast'])
    assert args.level == 'fast'


def test_show_defaults_prints_every_parameter(capsys):
    """A user must be able to see what the program decided for them."""
    rc = main(['--show-defaults'])
    out = capsys.readouterr().out
    from diffmrc.policy.params import PARAMS
    for key in PARAMS:
        assert key in out, key
    assert rc == 0


def test_show_defaults_includes_the_description(capsys):
    main(['--show-defaults'])
    out = capsys.readouterr().out
    from diffmrc.policy.params import PARAMS
    assert PARAMS['min_entropy'].help[:30] in out


def test_check_validates_without_computing(tmp_path, capsys):
    p = _write(tmp_path)
    rc = main(['--check', str(p)])
    assert rc == 0
    assert 'ok' in capsys.readouterr().out.lower()


def test_check_reports_a_missing_file(tmp_path, capsys):
    rc = main(['--check', str(tmp_path / 'nope.xyz')])
    assert rc != 0
    assert 'nope.xyz' in capsys.readouterr().err


def test_check_reports_a_malformed_file(tmp_path, capsys):
    p = _write(tmp_path, 'this is not an xyz file\n')
    rc = main(['--check', str(p)])
    assert rc != 0
    assert capsys.readouterr().err


def test_check_explains_a_bad_element(tmp_path, capsys):
    """A wrong atom symbol should say which one, not just fail."""
    p = _write(tmp_path, '2\nbad\nXx 0 0 0\nH 0 0 0.74\n')
    rc = main(['--check', str(p)])
    assert rc != 0
    err = capsys.readouterr().err
    assert 'Xx' in err or 'element' in err.lower()


def test_an_unknown_flag_exits_nonzero():
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(['mol.xyz', '--no-such-flag'])
    assert e.value.code != 0


def test_an_out_of_range_parameter_is_rejected_before_computing(
        tmp_path, capsys):
    """Validation happens up front, so a wrong number costs nothing."""
    p = _write(tmp_path)
    rc = main([str(p), '--gradient-tol', '1e9'])
    assert rc == 2
    assert 'gradient_tol' in capsys.readouterr().err


def test_a_negative_spin_is_rejected(tmp_path, capsys):
    p = _write(tmp_path)
    rc = main([str(p), '--spin', '-1'])
    assert rc == 2
    assert 'spin' in capsys.readouterr().err.lower()


def test_an_inconsistent_charge_and_spin_are_reported(tmp_path, capsys):
    """The PySCF message for this is the one worth copying.

    Its electron-count error names the value, the reason and the common
    misconception: mol.spin = 2S, not 2S+1.
    """
    p = _write(tmp_path, '1\nhydrogen\nH 0 0 0\n')
    rc = main([str(p), '--charge', '0', '--spin', '7'])
    assert rc == 2
    err = capsys.readouterr().err
    assert 'spin' in err.lower()


def test_a_malformed_active_space_is_rejected(tmp_path, capsys):
    p = _write(tmp_path)
    rc = main([str(p), '--active-space', 'not-a-range'])
    assert rc == 2
    assert 'active' in capsys.readouterr().err.lower()


def test_the_default_level_is_the_one_that_can_run():
    """The robust level refuses until the verification exists.

    Defaulting to it would make every run end in a refusal whose reason
    is a missing feature rather than a property of the system. The
    escalation path stays: a fast run whose checks fail reports that it
    should be re-run at the robust level.
    """
    from diffmrc.cli.main import build_parser

    args = build_parser().parse_args(['mol.xyz'])
    assert args.level == 'fast'


def test_a_failed_self_check_is_not_reported_as_success(tmp_path):
    """Exit 0 means success; a run whose checks failed is not one.

    Surfaced by the server validation on 2026-09-22: a fast run that
    landed on a saddle (three negative eigenvalues) printed its energy
    and exited 0, which a script would read as a quotable result.
    """
    import diffmrc.cli.main as main_mod
    from diffmrc.driver.state import Decision, Level, SelfCheck

    p = _write(tmp_path)

    class _Result:
        energy = -75.0
        frequencies = None
        algorithm_data = {'geometry': 'x'}
        decision = Decision(
            level=Level.ROBUST,
            checks=[SelfCheck('orbital_gradient', True, 'ok'),
                    SelfCheck('hessian_curvature', False, '3 negatives')],
            reasons=['hessian_curvature: 3 negatives'],
            suggestions=[])

    original = main_mod._compute
    main_mod._compute = lambda *a, **k: _Result()
    try:
        rc = main_mod.main([str(p), '--outdir', str(tmp_path)])
    finally:
        main_mod._compute = original
    assert rc != 0
