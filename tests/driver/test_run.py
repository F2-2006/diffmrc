"""Assembling the self-checks that gate a reported result."""
import pytest

from diffmrc.driver.run import RunResult, checks_from_result
from diffmrc.driver.state import Decision, Level, next_level


def _good():
    return dict(grad_norm=1e-10, n_negative=0, e_casscf=-76.0,
                e_reference=-75.9, n_elec_active=7, n_elec_expected=7)


def _by_name(checks):
    return {c.name: c for c in checks}


def test_all_four_checks_pass_on_a_good_result():
    checks = checks_from_result(**_good())
    assert all(c.passed for c in checks)
    assert {c.name for c in checks} == {
        'orbital_gradient', 'hessian_curvature', 'variational',
        'active_electrons',
    }


def test_the_checks_are_ordered_deterministically():
    a = [c.name for c in checks_from_result(**_good())]
    b = [c.name for c in checks_from_result(**_good())]
    assert a == b


def test_a_loose_gradient_fails_its_check():
    """The failure of a manuscript table row that stopped at 8.9e-7."""
    args = _good()
    args['grad_norm'] = 8.9e-7
    assert not _by_name(checks_from_result(**args))['orbital_gradient'].passed


def test_a_saddle_point_fails_the_curvature_check():
    """Two manuscript rows used saddle points, with 3 and 1 negatives."""
    for n_neg in (1, 3):
        args = _good()
        args['n_negative'] = n_neg
        assert not _by_name(checks_from_result(**args))['hessian_curvature'].passed


def test_an_energy_above_the_reference_fails_variationally():
    """The identity CASSCF <= starting ROHF, violated by 0.248 Ha once."""
    args = _good()
    args['e_casscf'] = args['e_reference'] + 0.248
    assert not _by_name(checks_from_result(**args))['variational'].passed


def test_equality_at_the_reference_passes_variationally():
    """A CASSCF that reproduces its reference exactly is fine."""
    args = _good()
    args['e_casscf'] = args['e_reference']
    assert _by_name(checks_from_result(**args))['variational'].passed


def test_a_wrong_electron_count_fails_its_check():
    """The label-parsing bug that selected core orbitals."""
    args = _good()
    args['n_elec_active'] = 13
    assert not _by_name(checks_from_result(**args))['active_electrons'].passed


def test_every_check_carries_a_detail_quoting_the_numbers():
    """A failure has to be diagnosable from its own message."""
    for c in checks_from_result(**_good()):
        assert c.detail


def test_failing_a_check_escalates_then_refuses():
    """The full path: fast fails, robust fails, refuse."""
    args = _good()
    args['grad_norm'] = 1.0
    d1 = next_level(Level.FAST, checks_from_result(**args))
    assert d1.level is Level.ROBUST
    d2 = next_level(Level.ROBUST, checks_from_result(**args))
    assert d2.level is Level.REFUSED
    assert d2.suggestions


def test_a_good_result_stays_at_the_fast_level():
    d = next_level(Level.FAST, checks_from_result(**_good()))
    assert d.level is Level.FAST


def test_run_result_reports_trustworthiness():
    good = RunResult(energy=-76.0, decision=Decision(level=Level.FAST))
    assert good.trustworthy
    bad = RunResult(energy=float('nan'),
                    decision=Decision(level=Level.REFUSED))
    assert not bad.trustworthy


def test_an_unfinished_run_is_not_trustworthy():
    """No decision yet means no basis for trusting the number."""
    assert not RunResult().trustworthy


def test_the_thresholds_are_function_arguments():
    """A caller can tighten them without editing the module."""
    args = _good()
    args['grad_norm'] = 1e-9
    assert _by_name(checks_from_result(**args))['orbital_gradient'].passed
    assert not _by_name(checks_from_result(**args,
                                           gradient_tol=1e-12)
                        )['orbital_gradient'].passed


def test_a_robust_result_without_verification_fails_its_check():
    """The robust level promises a verification this build does not do.

    Refusing is the design: a number that looks reasonable but rests on
    an unverified active space will be quoted. Until the verification
    exists, the honest outcome at the robust level is a refusal that
    names the gap.
    """
    from diffmrc.driver.run import checks_from_result

    checks = checks_from_result(
        grad_norm=1e-10, n_negative=0, e_casscf=-76.0, e_reference=-75.9,
        n_elec_active=7, n_elec_expected=7,
        active_space_verified=False)
    named = {c.name: c for c in checks}
    assert 'active_space' in named
    assert not named['active_space'].passed


def test_a_fast_result_has_no_active_space_check():
    from diffmrc.driver.run import checks_from_result

    checks = checks_from_result(
        grad_norm=1e-10, n_negative=0, e_casscf=-76.0, e_reference=-75.9,
        n_elec_active=7, n_elec_expected=7)
    assert {c.name for c in checks} == {
        'orbital_gradient', 'hessian_curvature', 'variational',
        'active_electrons',
    }
