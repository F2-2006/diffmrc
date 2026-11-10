"""The per-callable compiled-function cache."""
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from diffmrc.core.jit_cache import cached_jit  # noqa: E402


def _factory(fn):
    return ('compiled', id(fn))


def test_same_callable_and_kind_returns_the_same_object():
    def f(x):
        return x

    first = cached_jit(f, 'grad', _factory)
    second = cached_jit(f, 'grad', _factory)
    assert first is second


def test_different_kinds_are_cached_separately():
    def f(x):
        return x

    grad = cached_jit(f, 'grad', _factory)
    hvp = cached_jit(f, 'hvp', _factory)
    assert grad is not hvp
    assert grad[0] == hvp[0] == 'compiled'
    assert grad[1] == hvp[1] == id(f)


def test_different_callables_do_not_share_a_cache():
    def f(x):
        return x

    def g(x):
        return x

    assert cached_jit(f, 'grad', _factory) == cached_jit(f, 'grad', _factory)
    assert cached_jit(f, 'grad', _factory) != cached_jit(g, 'grad', _factory)


def test_the_factory_runs_once_per_key():
    calls = []

    def f(x):
        return x

    def factory(fn):
        calls.append(fn)
        return object()

    cached_jit(f, 'grad', factory)
    cached_jit(f, 'grad', factory)
    assert len(calls) == 1


def test_an_object_that_cannot_hold_attributes_still_works():
    class Slotted:
        __slots__ = ()

        def __call__(self, x):
            return x

    fn = Slotted()
    first = cached_jit(fn, 'grad', _factory)
    second = cached_jit(fn, 'grad', _factory)
    assert first is not second
    assert first[0] == 'compiled'
