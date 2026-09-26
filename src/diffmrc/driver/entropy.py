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

__all__ = ['entropy_from_ci_masks', 'single_orbital_entropy_from_ci',
           'make_casscf_entropy_theta_fn']


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


def make_casscf_entropy_theta_fn(C0_, h_, e_, dets, *, ncore, nact, nele,
                                 nmo, pairs, ci_solver='plain',
                                 eig_solver='default'):
    """``theta -> s1(theta)`` of the chain's CI vector, differentiable.

    The entropy sibling of ``make_casscf_dipole_theta_fn``: the same
    ``_casscf_solution`` produces the CI vector, and the bitmask
    extraction is the way out.  The masks are built once (determinants
    are static); differentiating through the extraction touches only the
    quadratic map and the shared formula, so a threshold-edge orbital
    becomes a diagnosable margin (the derivative of its occupation)
    instead of a step.
    """
    from .nevpt2 import _casscf_solution

    masks = occupation_masks(dets, ncore=ncore, nact=nact)

    def s1_theta(th_):
        _, _, ci2, _ = _casscf_solution(
            C0_, h_, e_, th_, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
            pairs=pairs, ci_solver=ci_solver, eig_solver=eig_solver)
        return entropy_from_ci_masks(ci2, masks)

    return s1_theta
