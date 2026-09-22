"""The layer directions the architecture depends on.

`lint-imports` is what enforces these; the tests below document the
intent so a reader learns the architecture here, and a reader of the lint
output gets the exact violation.
"""
import pathlib

import pytest


def test_policy_and_driver_import():
    import diffmrc.policy  # noqa: F401
    import diffmrc.driver  # noqa: F401


def _package_sources(package_name):
    """Every .py file in a package, read from disk.

    Reading from disk rather than importing keeps this test independent of
    import side effects, and catches a forbidden import in a module that
    nothing else has imported yet.
    """
    import diffmrc
    root = pathlib.Path(diffmrc.__file__).parent
    package = root / package_name
    return sorted(package.rglob('*.py'))


@pytest.mark.parametrize('layer', ['analysis', 'core'])
def test_lower_layers_never_reference_the_upper_ones(layer):
    """analysis and core must not name policy, driver, report, or cli.

    This is the separation that lets the analysis layer be validated on
    hand-built numbers, and the kernel be validated against reference
    energies, without either dragging in the layers above.
    """
    forbidden = ('policy', 'driver', 'report', 'cli')
    offenders = []
    for path in _package_sources(layer):
        text = path.read_text(encoding='utf-8')
        for word in forbidden:
            if f'diffmrc.{word}' in text or f'from ..{word}' in text \
                    or f'from .{word}' in text:
                offenders.append(f'{path.name}: {word}')
    assert not offenders, f'{layer} reaches upward: {offenders}'


def test_policy_never_references_the_driver():
    offenders = []
    for path in _package_sources('policy'):
        text = path.read_text(encoding='utf-8')
        if 'diffmrc.driver' in text or 'from ..driver' in text:
            offenders.append(path.name)
    assert not offenders, f'policy reaches into driver: {offenders}'


def test_report_never_references_the_layers_below_or_above():
    """Report takes data classes only, so it is tested on hand-built input."""
    offenders = []
    for path in _package_sources('report'):
        text = path.read_text(encoding='utf-8')
        for word in ('core', 'policy', 'driver', 'cli'):
            if f'diffmrc.{word}' in text or f'from ..{word}' in text \
                    or f'from .{word}' in text:
                offenders.append(f'{path.name}: {word}')
    assert not offenders, f'report reaches into other layers: {offenders}'
