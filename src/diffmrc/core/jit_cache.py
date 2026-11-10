"""Per-callable cache for compiled JAX functions.

WHY THIS EXISTS

``jax.jit`` builds a fresh wrapper every time it is called, and a wrapper
carries its own compilation cache. A routine that constructs
``jax.jit(jax.grad(f))`` inside its body therefore re-traces, re-lowers and
re-compiles the graph on every invocation, even when ``f`` is the same
object and the shapes have not changed. Measured on N2/6-31G CAS(2,2)
(92 parameters): the first gradient call costs 1.24 s and the second
0.007 s, and a fresh wrapper pays the 1.24 s again. A multistart run calls
the same optimizer dozens of times on the same energy function, so the
compilation, not the arithmetic, sets the cost.

The same problem in a different form is recorded in ``core.rotation``: the
unrolled Givens loop made the compile grow with the pair count, and the
``lax.scan`` rewrite fixed it. This module fixes the other half -- the
wrapper being rebuilt per call.

WHAT IT DOES

``cached_jit(fn, kind, factory)`` stores the object returned by
``factory(fn)`` on ``fn`` itself, under a per-kind key, and returns the
same object on later calls with the same ``fn`` and ``kind``. The cache
lives and dies with the callable, so two systems with two energy functions
never share a compiled graph, and nothing is kept alive after the caller
drops the function.

The cache key is the callable object and the kind string, not the shapes;
``jax.jit`` already keys its own cache on shapes and dtypes, so a call with
new shapes compiles once and is cached by the wrapper as usual.

Callables that do not accept attributes (some builtins) fall back to an
uncached ``factory(fn)``. That is slower and never wrong.
"""
__all__ = ['cached_jit']

_ATTRIBUTE = '_diffmrc_jit_cache'


def cached_jit(fn, kind, factory):
    """Return ``factory(fn)``, cached on ``fn`` under ``kind``.

    Parameters
    ----------
    fn : callable
        The function the compiled object wraps.
    kind : str
        Distinguishes several compiled forms of the same function (for
        example ``'grad'`` and ``'hvp'``).
    factory : callable
        ``factory(fn)`` builds the object on a cache miss.

    Returns
    -------
    object
        The cached object, or a fresh one when ``fn`` cannot hold the
        cache attribute.
    """
    cache = getattr(fn, _ATTRIBUTE, None)
    if cache is None:
        cache = {}
        try:
            setattr(fn, _ATTRIBUTE, cache)
        except (AttributeError, TypeError):
            return factory(fn)
    key = str(kind)
    compiled = cache.get(key)
    if compiled is None:
        compiled = factory(fn)
        cache[key] = compiled
    return compiled
