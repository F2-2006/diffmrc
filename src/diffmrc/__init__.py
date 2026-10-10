"""End-to-end differentiable multireference electronic structure stack.

The public surface is this module's ``__all__`` and nothing else: the
two entry points of ``driver.run``, the result container, and the
command-line ``main``. Deep module paths remain reachable but are
internal -- importing ``diffmrc`` itself stays light (no jax, no
pyscf), so ``diffmrc --check`` works without the numerical kernel
installed; the heavy modules load on first attribute access.
"""

__version__ = "0.1.0"

__all__ = ['run_from_geometry', 'vertical_excitations_from_geometry',
           'RunResult', 'main']

_LAZY = {
    'run_from_geometry': ('diffmrc.driver.run', 'run_from_geometry'),
    'vertical_excitations_from_geometry': (
        'diffmrc.driver.run', 'vertical_excitations_from_geometry'),
    'RunResult': ('diffmrc.driver.run', 'RunResult'),
    'main': ('diffmrc.cli.main', 'main'),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}') from None
    import importlib

    return getattr(importlib.import_module(module_name), attribute)
