"""The public surface: existence, importability, and deliberate smallness.

Users should reach the stack through ``diffmrc.<name>`` only (the
GradDFT single-exit convention); deep module paths stay internal. The
surface is kept small on purpose and this test freezes that choice --
adding a name is a decision, not an accident.
"""
import inspect


def test_public_names_importable():
    import diffmrc
    assert diffmrc.__all__, 'the package must declare a public surface'
    for name in diffmrc.__all__:
        obj = getattr(diffmrc, name)
        assert obj is not None, name


def test_the_two_entry_points_are_callable():
    import diffmrc
    assert callable(diffmrc.run_from_geometry)
    assert callable(diffmrc.vertical_excitations_from_geometry)
    assert callable(diffmrc.main)
    sig = inspect.signature(diffmrc.run_from_geometry)
    assert 'geometry_file' in sig.parameters


def test_version_is_a_string():
    import diffmrc
    assert isinstance(diffmrc.__version__, str) and diffmrc.__version__


def test_public_surface_stays_small():
    import diffmrc
    assert len(diffmrc.__all__) <= 8


def test_importing_the_package_stays_light():
    """No jax/pyscf at import time: --check must work without the kernel."""
    import subprocess
    import sys
    code = ('import sys, diffmrc; '
            'heavy = [m for m in ("jax", "pyscf", "pyscfad") '
            'if m in sys.modules]; '
            'print("HEAVY:", heavy)')
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert 'HEAVY: []' in out.stdout, out.stdout
