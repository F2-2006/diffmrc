"""Bidirectional growth: move pricing, the down move, stop rules, cap."""
import pytest

from diffmrc.policy.growth import grow_window


def oracle_from_weights(weights):
    """E(window) = -sum(weights[i] for i in window)."""
    def evaluate(ncore, nact, order):
        return -sum(weights[i] for i in order[ncore:ncore + nact])
    return evaluate


def test_grows_upward_toward_the_informative_columns():
    w = [0.0, 4.0, 3.0, 0.0, 0.0, 0.0]
    res = grow_window(oracle_from_weights(w), ncore0=1, nact0=1,
                      n_columns=6, cap=10, f_stop=0.05)
    # up (+3.0) then a dead step (0.0) which stops the loop
    assert [s.direction for s in res.steps] == ['up', 'up']
    assert res.steps[0].column == 2
    assert res.steps[0].gain == pytest.approx(3.0)
    assert res.steps[0].nact == 1        # state before the move
    assert res.ncore_final == 1
    assert res.nact_final == 2


def test_down_move_pulls_the_top_core_orbital():
    # col 1 is the top core column and worth as much as a frozen f hole;
    # col 2 sits in the seed window; upward there is nothing to gain.
    w = [0.0, 6.0, 1.0, 0.0, 0.0, 0.0]
    res = grow_window(oracle_from_weights(w), ncore0=2, nact0=1,
                      n_columns=6, cap=10, f_stop=0.05)
    first = res.steps[0]
    assert first.direction == 'down'
    assert first.column == 1             # by original column index
    assert first.gain == pytest.approx(6.0)
    assert res.ncore_final == 1          # the core gave one orbital back
    assert res.nact_final == 2
    assert res.steps[1].gain == pytest.approx(0.0)   # then both dead


def test_first_rule_stops_early_where_step_rule_continues():
    # A complementarity-shaped gain profile: a small marginal step after
    # a large first step, then a second wind. The first-step baseline
    # stops at 0.78; the adjacent-step baseline survives it and picks up
    # the 5.0 that follows.
    w = [0.0, 16.0, 6.2, 0.78, 5.0, 0.02, 0.0]
    ev = oracle_from_weights(w)
    r1 = grow_window(ev, 1, 0, 7, cap=10, f_stop=0.1, stop_rule='first')
    assert len(r1.steps) == 3
    assert r1.nact_final == 2
    r2 = grow_window(ev, 1, 0, 7, cap=10, f_stop=0.1, stop_rule='step')
    assert len(r2.steps) == 5
    assert r2.nact_final == 4
    assert any(s.gain == pytest.approx(5.0) for s in r2.steps)


def test_cap_limits_growth():
    w = [1.0] * 6
    res = grow_window(oracle_from_weights(w), 1, 1, 6, cap=3, f_stop=0.05)
    assert res.nact_final == 3
    assert len(res.steps) == 2           # one step per iteration below cap


def test_unknown_stop_rule_refuses():
    with pytest.raises(ValueError):
        grow_window(oracle_from_weights([0.0]), 1, 0, 1, stop_rule='nope')
