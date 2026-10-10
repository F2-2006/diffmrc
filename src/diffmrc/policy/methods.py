"""The registry of reported-energy methods.

One table answers, for every value of the ``method`` parameter: how the
reported energy is formed (``on_top``, ``hyb``) and what the report
layer should print about it (``ground_protocol``, ``vee_protocol``).
Before this table existed, the same string literals and
``method in ('mcpdft', 'mcpdft0')`` comparisons were scattered across
``driver.kernel``, ``driver.run``, ``cli.main`` and the parameter
choices -- four places to touch for one new method, four chances for
the report text to drift from the dispatch.

The table lives in the policy layer because ``policy.params`` declares
the choice list and the import contract forbids policy -> driver. It
carries *data* only; the dispatch behaviour stays in the layers that
own it (the driver runs the factories, the CLI renders the fields) and
reads the data from here.

Adding a method: add one entry here, extend the driver's factories if
it needs new calls, and extend any consumer switch on ``on_top``.
"""

from dataclasses import dataclass

__all__ = ['MethodSpec', 'METHODS', 'method_spec', 'on_top_method']


@dataclass(frozen=True)
class MethodSpec:
    """One reported-energy method and its report semantics.

    Parameters
    ----------
    name : str
        The ``method`` parameter value.
    hyb : float
        Mixing of the CASSCF energy into the reported value (mcpdft0 =
        0.25, the tPBE0 hybrid); 0 for the pure methods.
    on_top : bool
        True when the reported energy is a one-step on-top evaluation
        at CASSCF-driven orbitals rather than a variational expectation
        value -- the flag every consumer switch keys on.
    ground_protocol : str or None
        The ground-state report's protocol line; None for methods whose
        ground-state report carries no protocol field (casci).
    vee_protocol : str
        The vertical-excitation report's protocol line (every method
        has one).
    """

    name: str
    hyb: float = 0.0
    on_top: bool = False
    ground_protocol: str = None
    vee_protocol: str = ''


METHODS = {
    'casci': MethodSpec(
        name='casci',
        vee_protocol='equal-weight SA-CASSCF, per-state NEVPT2',
    ),
    'mcpdft': MethodSpec(
        name='mcpdft',
        on_top=True,
        ground_protocol='one-step: CASSCF-driven orbitals, one on-top '
                        'evaluation (tPBE)',
        vee_protocol='equal-weight SA-CASSCF, per-state one-step MC-PDFT '
                     '(tPBE) and NEVPT2',
    ),
    'mcpdft0': MethodSpec(
        name='mcpdft0',
        hyb=0.25,
        on_top=True,
        ground_protocol='one-step: CASSCF-driven orbitals, one on-top '
                        'evaluation (tPBE), 0.25-mixed with the CASSCF '
                        'energy (tPBE0)',
        vee_protocol='equal-weight SA-CASSCF, per-state NEVPT2 with the '
                     'one-step tPBE0 column (0.25-mixed)',
    ),
}


def method_spec(name):
    """The spec for ``name``; unknown names raise with the choices."""
    try:
        return METHODS[str(name)]
    except KeyError:
        raise ValueError(
            f'method: {name!r} is not a known method; '
            f'choose one of {sorted(METHODS)}') from None


def on_top_method(name):
    """True when ``name`` is a known on-top method (tolerant of None)."""
    spec = METHODS.get(str(name))
    return spec is not None and spec.on_top
