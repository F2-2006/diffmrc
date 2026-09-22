"""The single source of parameter defaults."""
import pytest

from diffmrc.policy.params import (
    PARAMS,
    EffectiveParams,
    Param,
    defaults,
    effective,
    validate,
)


def test_every_param_declares_the_documented_fields():
    """A parameter without a description cannot be presented to a user."""
    for key, p in PARAMS.items():
        assert isinstance(p, Param), key
        assert p.help, f'{key} has no description'
        assert p.layer in ('core', 'analysis', 'policy', 'driver', 'report'), key


def test_every_param_declares_a_range_or_a_choice_set():
    """A parameter with neither cannot be validated."""
    for key, p in PARAMS.items():
        assert (p.lo is not None and p.hi is not None) or p.choices, key


def test_defaults_are_within_their_declared_ranges():
    """A default outside its own range is a contradiction."""
    for key, p in PARAMS.items():
        if p.lo is not None:
            assert p.default >= p.lo, key
        if p.hi is not None:
            assert p.default <= p.hi, key
        if p.choices:
            assert p.default in p.choices, key


def test_defaults_returns_a_plain_mapping():
    d = defaults()
    assert isinstance(d, dict)
    assert set(d) == set(PARAMS)


def test_validate_accepts_the_defaults():
    assert validate(defaults()) == []


def test_validate_rejects_an_out_of_range_value():
    p = PARAMS['gradient_tol']
    problems = validate({'gradient_tol': p.hi * 10})
    assert problems
    assert 'gradient_tol' in problems[0]


def test_validate_rejects_a_value_below_the_range():
    problems = validate({'gradient_tol': -1.0})
    assert problems


def test_validate_rejects_an_unknown_key():
    problems = validate({'no_such_parameter': 1})
    assert problems
    assert 'no_such_parameter' in problems[0]


def test_validate_rejects_a_bad_choice():
    problems = validate({'localization': 'nonsense'})
    assert problems
    assert 'localization' in problems[0]


def test_validate_reports_every_problem_not_just_the_first():
    """A user fixing one error at a time would retry repeatedly."""
    problems = validate({'no_such_parameter': 1, 'gradient_tol': 1e9,
                         'localization': 'nonsense'})
    assert len(problems) >= 3


def test_effective_merges_overrides_over_defaults():
    out = effective({'gradient_tol': 1e-6})
    assert out['gradient_tol'] == 1e-6
    assert out['plateau_values'] == defaults()['plateau_values']


def test_effective_marks_where_each_value_came_from():
    """The report has to say which values the user chose.

    That distinction is what lets a reader tell a surprising result from
    a surprising input.
    """
    out = effective({'gradient_tol': 1e-6})
    assert out.sources['gradient_tol'] == 'user'
    assert out.sources['plateau_values'] == 'default'


def test_effective_records_only_the_keys_the_user_touched():
    out = effective({'gradient_tol': 1e-6})
    assert 'plateau_values' in out.sources
    assert out.sources['plateau_values'] == 'default'
    assert sum(1 for v in out.sources.values() if v == 'user') == 1


def test_effective_rejects_an_invalid_override():
    with pytest.raises(ValueError):
        effective({'gradient_tol': 1e9})


def test_effective_is_readable_like_a_mapping():
    out = effective({'gradient_tol': 1e-6})
    assert isinstance(out, EffectiveParams)
    assert out['gradient_tol'] == 1e-6
    assert 'gradient_tol' in out


def test_the_gradient_tolerance_has_one_value_not_three():
    """The conflict this module exists to resolve.

    The experiment scripts carried 1e-8, 1e-12 and 1e-13 in different
    places for the same physical quantity.
    """
    assert isinstance(PARAMS['gradient_tol'].default, float)


def test_the_multireference_threshold_is_the_documented_one():
    """0.14 comes from the selection protocol; it is not a free choice."""
    assert PARAMS['min_entropy'].default == pytest.approx(0.14)


def test_every_param_names_the_layer_that_consumes_it():
    layers = {p.layer for p in PARAMS.values()}
    assert 'driver' in layers
    assert 'policy' in layers
