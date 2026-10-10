#!/usr/bin/env python
"""Gradient-safe primitives for the differentiable core.

WHY THIS MODULE EXISTS

Three separate incidents in this project trace to the same trap
family -- a mathematically harmless singularity in the *values* that
becomes NaN poisoning in the *derivatives*:

* a negative floating-point noise (-4e-19) entered a fractional power
  inside the on-top density, produced 2001 NaN points and sent the
  optimizer off (t384d; fixed by clamping the intermediate to its
  positive part, both for the density and for 1 +/- zeta);
* a 0/0 in the *untaken* branch of a ``jnp.where`` reached the reverse
  pass as 0 * inf = NaN and infected the selected branch too (commit
  5e640c9; fixed by flooring the untaken denominator before the
  select);
* roundoff-level eigenvalue gaps made the raw 1/(w_j - w_i) rule
  overflow (see ``core/eigh.py``, a different but adjacent instance of
  "guard the reverse pass, not just the value").

The household rule from those incidents: guard the *denominator* or
the *base* on both branches, never assume the untaken branch is dead
in the reverse pass. These primitives fold the pattern into one place
with the gradient semantics written down, following ``pyscfad``'s
``safe_sqrt`` convention (whose docstring states grad(sqrt)(0) = inf
against grad(safe_sqrt)(0) = 0).

USAGE RULE (for new code; existing call sites migrate gradually)

Use these primitives wherever a value is a ratio, a positive part, or
a guard for a branch; do not open-code ``jnp.maximum(x, 0)`` /
``x / jnp.maximum(d, floor)`` / the two-step where-floor pattern.
The behaviours are identical; the point is that there is one place to
audit, extend and test.
"""
from jax import numpy as jnp

__all__ = ['positive_part', 'safe_divide', 'guarded_ratio']


def positive_part(x):
    """``max(x, 0)`` with a finite gradient everywhere.

    Value: negative entries become 0. Gradient: the negative branch
    carries derivative 0 (raw ``sqrt`` of a negative base would be NaN;
    a negative part feeding a fractional power is the t384d case). The
    tie-break at exactly 0 is a backend detail and not part of the
    contract -- both 0 and 0.5 are acceptable there.
    """
    return jnp.maximum(x, 0.0)


def safe_divide(numerator, denominator, floor=1e-30):
    """``numerator / denominator`` with the denominator floored.

    Value: denominators below ``floor`` (including 0 and the negative
    noise of a physically positive quantity) are read as ``floor``, so
    the result is large but finite. Gradient: finite wherever the
    floored value is finite; the raw form's 1/0 = inf (and its
    0 * inf product in an untaken branch) cannot occur.

    ``floor`` is positive by construction: this primitive is for
    quantities that are physically nonnegative (densities, occupation
    weights, 1 +/- zeta). A signed denominator needs an explicit
    symmetric floor instead.
    """
    return numerator / jnp.maximum(denominator, floor)


def guarded_ratio(numerator, denominator, keep, floor=1e-30):
    """``numerator / denominator`` where ``keep``, else 0 -- gradient safe.

    The exact pattern of commit 5e640c9, generalised: the untaken
    branch's denominator is replaced by 1 *before* the select, so the
    reverse pass never forms ``0 * inf``; the outer select then zeroes
    the branch's cotangent. Both branches, values and derivatives, stay
    finite no matter what the untaken inputs are.

    ``keep`` is a boolean array broadcastable against the inputs.
    """
    kept = jnp.asarray(keep)
    den = jnp.where(kept, denominator, jnp.ones_like(denominator))
    den = jnp.maximum(den, floor)
    return jnp.where(kept, numerator / den, jnp.zeros_like(numerator))
