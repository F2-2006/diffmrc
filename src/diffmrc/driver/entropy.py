"""The single-orbital entropy of a chain's CI vector, differentiable.

Why this exists next to ``analysis.ci``: the analysis function is numpy
and pins the convention, but the chain needs s1 as a function of the CI
vector -- the vector is itself a function of the orbital parameters, and
asking how stable an entropy plateau is under re-optimization means
differentiating through this extraction.  The masks are static (the
determinants do not move), so the differentiable part is one square and
one matmul per occupation, then the shared formula from
``analysis.entropy`` run on ``jax.numpy``.

Cost note: `entropy_from_ci_masks` is the hot path. A caller evaluating
s1 many times (an optimizer, a derivative sweep) must hoist
``analysis.ci.occupation_masks`` out of the loop; it walks the
determinant list in Python and must not run per step.
"""
from ..analysis.ci import occupation_masks

__all__ = ['entropy_from_ci_masks', 'single_orbital_entropy_from_ci']


def entropy_from_ci_masks(c_ci, masks):
    """s1 over the active space given precomputed masks (jax)."""
    from jax import numpy as jnp
    from ..analysis.entropy import single_orbital_entropy

    mask_a, mask_b, mask_2 = masks
    w = jnp.asarray(c_ci, dtype=jnp.float64) ** 2
    n_a = w @ jnp.asarray(mask_a, dtype=jnp.float64)
    n_b = w @ jnp.asarray(mask_b, dtype=jnp.float64)
    p2 = w @ jnp.asarray(mask_2, dtype=jnp.float64)
    return single_orbital_entropy(n_a, n_b, p2, xp=jnp)


def single_orbital_entropy_from_ci(c_ci, dets, ncore=0, nact=2):
    """s1 over the active space of ``c_ci``, differentiable (jax).

    The sibling of ``analysis.ci.entropy_from_ci``; a test evaluates both
    on the same vector, and the gradient test binds this route's AD to a
    finite difference of that sibling.
    """
    return entropy_from_ci_masks(c_ci, occupation_masks(dets, ncore, nact))
