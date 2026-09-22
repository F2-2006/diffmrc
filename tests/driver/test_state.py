"""The escalation state machine and its refusal path."""
import pytest

from diffmrc.driver.state import Decision, Level, SelfCheck, next_level


def _ok(name='check'):
    return SelfCheck(name=name, passed=True, detail='fine')


def _bad(name='orbital_gradient'):
    return SelfCheck(name=name, passed=False, detail='not converged')


def test_all_checks_passing_keeps_the_fast_level():
    d = next_level(Level.FAST, [_ok('gradient'), _ok('variational')])
    assert d.level is Level.FAST
    assert d.reasons == []


def test_a_failing_check_escalates_to_robust():
    d = next_level(Level.FAST, [_bad('orbital_gradient')])
    assert d.level is Level.ROBUST


def test_a_failing_check_at_robust_refuses():
    d = next_level(Level.ROBUST, [_bad('orbital_gradient')])
    assert d.level is Level.REFUSED


def test_refusal_carries_every_failing_check():
    d = next_level(Level.ROBUST,
                   [_bad('orbital_gradient'), _ok('variational'),
                    _bad('hessian_curvature')])
    names = [c.name for c in d.checks if not c.passed]
    assert names == ['orbital_gradient', 'hessian_curvature']


def test_refusal_is_not_an_exception():
    """Refusing is a result the caller can report, not a crash."""
    d = next_level(Level.ROBUST, [_bad('orbital_gradient')])
    assert isinstance(d, Decision)
    assert d.reasons


def test_refusal_always_offers_a_way_forward():
    d = next_level(Level.ROBUST, [_bad('orbital_gradient')])
    assert d.suggestions


def test_refusal_suggests_something_specific_to_the_failure():
    d = next_level(Level.ROBUST, [_bad('variational')])
    assert any('active space' in s for s in d.suggestions)


def test_an_unknown_check_still_yields_a_suggestion():
    d = next_level(Level.ROBUST, [_bad('something_new')])
    assert d.level is Level.REFUSED
    assert d.suggestions


def test_a_refusal_does_not_escalate_further():
    """Once refused, another failing check keeps it refused."""
    d = next_level(Level.REFUSED, [_bad('orbital_gradient')])
    assert d.level is Level.REFUSED


def test_no_checks_is_treated_as_passing():
    """Nothing to fail means nothing blocks the cheap path."""
    d = next_level(Level.FAST, [])
    assert d.level is Level.FAST


def test_the_check_names_survive_into_the_reasons():
    d = next_level(Level.ROBUST, [_bad('variational')])
    assert any('variational' in r for r in d.reasons)


def test_the_detail_survives_into_the_reasons():
    d = next_level(Level.ROBUST,
                   [SelfCheck('variational', False, '0.248 Ha above ROHF')])
    assert any('0.248' in r for r in d.reasons)


def test_duplicate_failures_yield_one_suggestion_each():
    d = next_level(Level.ROBUST,
                   [_bad('variational'), _bad('variational')])
    assert len(d.suggestions) == len(set(d.suggestions))


def test_escalation_reports_the_reason_it_escalated():
    d = next_level(Level.FAST, [_bad('orbital_gradient')])
    assert d.reasons
    assert any('robust' in s for s in d.suggestions)
