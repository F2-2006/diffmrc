"""Semantics and gradient contracts of the safe-math primitives.

This stack hit the same trap family three times before the module
existed: a negative floating-point noise entering a fractional power
(t384d, the on-top density), 0/0 in the *untaken* branch of a ``where``
reaching the reverse pass as 0 * inf (commit 5e640c9, the finite-basis
degeneracy guard). The primitives fold the fixes into one place and
document their gradient semantics -- the pyscfad ``safe_sqrt`` pattern,
where the docstring states how the safe gradient differs from the raw
one.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffmrc.core import safe_math as sm


def test_positive_part_value_and_grad():
    x = jnp.array([-1e-19, 0.0, 2.0])
    y = sm.positive_part(x)
    assert np.array_equal(np.asarray(y), np.array([0.0, 0.0, 2.0]))
    g = jax.grad(lambda v: sm.positive_part(v).sum())(x)
    assert np.isfinite(np.asarray(g)).all()
    # the negative branch contributes zero; the exact tie-break at 0 is
    # a backend detail and deliberately not asserted
    assert g[0] == 0.0 and g[2] == 1.0


def test_safe_divide_value_and_grad():
    num = jnp.array([1.0, 3.0])
    den = jnp.array([0.0, 2.0])
    y = sm.safe_divide(num, den, floor=1e-30)
    assert np.isfinite(np.asarray(y)).all()
    assert np.allclose(np.asarray(y), np.array([1e30, 1.5]), rtol=1e-12)
    g = jax.grad(lambda d: sm.safe_divide(jnp.ones(2), d, floor=1e-30).sum())(den)
    assert np.isfinite(np.asarray(g)).all()


def test_guarded_ratio_value():
    num = jnp.array([0.0, 3.0])
    den = jnp.array([0.0, 2.0])
    keep = jnp.array([False, True])
    y = sm.guarded_ratio(num, den, keep)
    # the untaken branch is 0, not 0/0
    assert np.array_equal(np.asarray(y), np.array([0.0, 1.5]))


def test_guarded_ratio_no_nan_grad():
    # regression for 5e640c9: a 0/0 in the untaken branch must not reach
    # the reverse pass. Differentiate w.r.t. both inputs.
    num = jnp.array([0.0, 3.0])
    den = jnp.array([0.0, 2.0])
    keep = jnp.array([False, True])
    g_num = jax.grad(lambda n: sm.guarded_ratio(n, den, keep).sum())(num)
    g_den = jax.grad(lambda d: sm.guarded_ratio(num, d, keep).sum())(den)
    assert np.isfinite(np.asarray(g_num)).all()
    assert np.isfinite(np.asarray(g_den)).all()
    assert np.array_equal(np.asarray(g_num), np.array([0.0, 0.5]))
    assert np.array_equal(np.asarray(g_den), np.array([0.0, -0.75]))


def test_primitives_compose_under_jit():
    f = jax.jit(lambda x: sm.positive_part(x) + sm.safe_divide(x, x, 1e-30))
    y = f(jnp.array([0.0, 2.0]))
    assert np.isfinite(np.asarray(y)).all()
