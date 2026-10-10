"""The method registry: one table behind every 'method' dispatch.

Before the registry the 'method' parameter was consumed with literal
comparisons in four places (the params choices, driver.kernel's ground
state, driver.run's vertical excitations, cli's report fields), and the
report texts lived next to each comparison. This test freezes the
mapping in ``policy.methods`` and asserts it reproduces the previous
literals exactly -- a refactor, not a behaviour change.

The table lives in the policy layer because ``policy.params`` needs the
choice list and the import contract forbids policy -> driver; the
dispatch *behaviour* stays in the layers that own it.
"""
import pytest

from diffmrc.policy.methods import METHODS, method_spec, on_top_method
from diffmrc.policy.params import PARAMS

# The pre-registry literals, copied from the code being replaced.
_GROUND_TPBE = 'one-step: CASSCF-driven orbitals, one on-top evaluation (tPBE)'
_VEE_CASCI = 'equal-weight SA-CASSCF, per-state NEVPT2'
_VEE_MCPDFT = ('equal-weight SA-CASSCF, per-state one-step MC-PDFT '
               '(tPBE) and NEVPT2')
_VEE_MCPDFT0 = ('equal-weight SA-CASSCF, per-state NEVPT2 with the '
                'one-step tPBE0 column (0.25-mixed)')


def test_registry_covers_the_declared_choices():
    assert set(PARAMS['method'].choices) == set(METHODS)
    assert PARAMS['method'].default in METHODS


def test_hyb_values():
    assert METHODS['casci'].hyb == 0.0
    assert METHODS['mcpdft'].hyb == 0.0
    assert METHODS['mcpdft0'].hyb == 0.25


def test_ground_protocol_texts_match_the_kernel_literals():
    assert METHODS['casci'].ground_protocol is None
    assert METHODS['mcpdft'].ground_protocol == _GROUND_TPBE
    assert METHODS['mcpdft0'].ground_protocol == (
        _GROUND_TPBE + ', 0.25-mixed with the CASSCF energy (tPBE0)')


def test_vee_protocol_texts_match_the_cli_literals():
    assert METHODS['casci'].vee_protocol == _VEE_CASCI
    assert METHODS['mcpdft'].vee_protocol == _VEE_MCPDFT
    assert METHODS['mcpdft0'].vee_protocol == _VEE_MCPDFT0


def test_on_top_flags():
    assert not METHODS['casci'].on_top
    assert METHODS['mcpdft'].on_top
    assert METHODS['mcpdft0'].on_top


def test_on_top_helper_tolerates_absent_and_unknown_names():
    assert on_top_method('mcpdft') and on_top_method('mcpdft0')
    assert not on_top_method('casci')
    assert not on_top_method(None)
    assert not on_top_method('bogus')


def test_method_spec_rejects_unknown_names_with_the_choices():
    with pytest.raises(ValueError) as exc:
        method_spec('mcpdft2')
    assert 'mcpdft2' in str(exc.value)
    assert 'casci' in str(exc.value)          # the choices are listed
    assert method_spec('mcpdft0').hyb == 0.25  # known names pass through


def test_protocol_texts_are_nonempty_for_every_method():
    for name, spec in METHODS.items():
        assert spec.vee_protocol, name
