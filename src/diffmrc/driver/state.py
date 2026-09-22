"""How much effort to spend, and when to stop and say so.

The machine has three states. The cheap one is a single orbital
optimization started from the reference orbitals. If a self-check fails,
the robust state retries from several starting points and verifies the
active space against the entanglement spectrum. If that also fails, the
driver refuses.

Refusing is the point of the design. A number that looks reasonable but
came from a saddle point, an unconverged optimization, or a wrongly
chosen active space is worse than no number, because it will be quoted.
The self-checks are all cheap and all independent of any external
program, so each is a genuine test rather than a comparison against
another implementation that might share the same mistake. Each one
corresponds to a failure that actually occurred in this project:

* the orbital gradient -- a manuscript table row once stopped at 8.9e-7
* the curvature -- two other rows were saddle points, with one and three
  negative eigenvalues
* the variational bound -- an active space once left CASSCF 0.248 Ha
  above the ROHF it started from
* the active electron count -- a label-parsing bug once selected core
  orbitals, which the downstream sanity check caught only by accident
"""
from dataclasses import dataclass, field
from enum import Enum

__all__ = ['Level', 'SelfCheck', 'Decision', 'next_level']


class Level(Enum):
    FAST = 'fast'
    ROBUST = 'robust'
    REFUSED = 'refused'


@dataclass
class SelfCheck:
    """One cheap, externally-independent condition the result must meet."""

    name: str
    passed: bool
    detail: str = ''


@dataclass
class Decision:
    """What to do next, and why."""

    level: Level
    checks: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


_SUGGESTIONS = {
    'orbital_gradient': 'tighten the orbital optimization, or supply a '
                        'starting point closer to the solution',
    'hessian_curvature': 'the reported point is a saddle point; run the '
                         'multi-start search to find a minimum',
    'variational': 'the active space does not contain the reference '
                   'configuration; check the orbital partition',
    'active_electrons': 'the active space and its electron count disagree; '
                        'the selection rule did not match this system',
    'active_space': 'specify the active space explicitly, or relax the '
                    'plateau criterion',
}


def next_level(current, checks):
    """Advance the state machine given the outcome of the self-checks.

    Parameters
    ----------
    current : Level
        The level that just finished.
    checks : sequence of SelfCheck
        Its self-checks.

    Returns
    -------
    Decision
        The next level, the checks that drove it, and -- when the machine
        refuses -- the reasons and what to try instead.
    """
    checks = list(checks)
    failed = [c for c in checks if not c.passed]

    if current is Level.REFUSED:
        return Decision(
            level=Level.REFUSED, checks=checks,
            reasons=['already refused; no further escalation'],
            suggestions=['report the failure and the checks that produced it'],
        )

    if not failed:
        return Decision(level=current, checks=checks)

    if current is Level.FAST:
        return Decision(
            level=Level.ROBUST, checks=checks,
            reasons=[f'{c.name}: {c.detail}' for c in failed],
            suggestions=['escalating to the robust level'],
        )

    suggestions = []
    for c in failed:
        hint = _SUGGESTIONS.get(c.name)
        if hint and hint not in suggestions:
            suggestions.append(hint)
    if not suggestions:
        suggestions = ['inspect the failing check and the run log']
    return Decision(
        level=Level.REFUSED, checks=checks,
        reasons=[f'{c.name}: {c.detail}' for c in failed],
        suggestions=suggestions,
    )
