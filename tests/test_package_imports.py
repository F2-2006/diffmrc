"""Smoke test: the package installs and its two lowest layers import."""


def test_package_imports():
    import diffmrc
    assert hasattr(diffmrc, '__version__')


def test_core_subpackage_imports():
    import diffmrc.core  # noqa: F401


def test_analysis_subpackage_imports():
    import diffmrc.analysis  # noqa: F401
