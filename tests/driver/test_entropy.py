"""The differentiable entropy path.

Every test here needs jax, so the module skips as a whole off the server.
The gradient test is the point of the module: it compares AD of the jax
route against a central difference of the *numpy* route, so the two
implementations are bound by the same number at two levels.
"""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.analysis.ci import entropy_from_ci  # noqa: E402
from diffmrc.core.nevpt2.impl import cas_dets_for2  # noqa: E402
from diffmrc.driver.entropy import (  # noqa: E402
    entropy_from_ci_masks,
    single_orbital_entropy_from_ci,
)


def _sz0_vector(ncore, nact, nele, seed=7):
    """A normalized vector confined to the S_z = 0 determinants."""
    dets = cas_dets_for2(ncore, nact, nele)
    rng = np.random.default_rng(seed)
    c = rng.normal(size=len(dets))
    keep = []
    for d in dets:
        a = sum(1 for i in range(0, 2 * (ncore + nact), 2) if (d >> i) & 1)
        b = sum(1 for i in range(1, 2 * (ncore + nact), 2) if (d >> i) & 1)
        keep.append(a == b)
    c = np.where(keep, c, 0.0)
    return dets, c / np.linalg.norm(c)


def test_the_jax_path_reproduces_the_numpy_extraction():
    dets, c = _sz0_vector(0, 4, 4)
    s_np = entropy_from_ci(c, dets, ncore=0, nact=4)
    s_jx = np.asarray(single_orbital_entropy_from_ci(
        jnp.asarray(c), dets, 0, 4))
    assert np.allclose(s_jx, s_np, atol=1e-12)


def test_the_mask_level_entry_point_agrees_with_the_convenience_one():
    from diffmrc.analysis.ci import occupation_masks
    dets, c = _sz0_vector(0, 4, 4)
    masks = occupation_masks(dets, ncore=0, nact=4)
    s_a = np.asarray(entropy_from_ci_masks(jnp.asarray(c), masks))
    s_b = np.asarray(single_orbital_entropy_from_ci(
        jnp.asarray(c), dets, 0, 4))
    assert np.allclose(s_a, s_b, atol=1e-14)


def test_the_gradient_matches_a_finite_difference_of_the_numpy_route():
    """AD differentiates the same function the numpy route evaluates."""
    dets, c = _sz0_vector(0, 4, 4)
    v = np.random.default_rng(3).normal(size=c.shape)

    def f_np(x):
        return float(np.sum(entropy_from_ci(x, dets, ncore=0, nact=4)))

    eps = 1e-6
    fd = (f_np(c + eps * v) - f_np(c - eps * v)) / (2 * eps)

    g = jax.grad(lambda x: jnp.sum(
        single_orbital_entropy_from_ci(x, dets, 0, 4)))(jnp.asarray(c))
    ad = float(jnp.asarray(g) @ jnp.asarray(v))
    assert np.isfinite(ad)
    assert abs(ad - fd) < 1e-8 * max(1.0, abs(fd))
