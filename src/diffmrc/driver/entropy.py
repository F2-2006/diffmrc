"""The entropy observables of a chain's CI vector, differentiable.

Why this exists next to ``analysis.ci``: the analysis functions are
numpy and pin the conventions, but the chain needs s1 and the mutual
information as functions of the CI vector -- the vector is itself a
function of the orbital parameters, and asking how stable an entropy
plateau or a pair entanglement is under re-optimization means
differentiating through these extractions.  The masks and the pair
scatters are static (the determinants do not move), so the
differentiable part is one square and one matmul per occupation for s1,
and one segment sum plus a 16x16 eigenproblem per pair for the mutual
information, then the shared formulas from ``analysis.entropy`` run on
``jax.numpy``.

Cost note: `entropy_from_ci_masks` is the hot path, and the pair route
multiplies its cost by the number of pairs asked for. A caller
evaluating s1 many times (an optimizer, a derivative sweep) must hoist
``analysis.ci.occupation_masks`` out of the loop, and a caller wanting
one pair's entanglement must not compute the whole triangle; both walk
the determinant list in Python and must not run per step.
"""
from ..analysis.ci import occupation_masks

__all__ = ['entropy_from_ci_masks', 'single_orbital_entropy_from_ci',
           'make_casscf_entropy_theta_fn', 'pair_density_from_scatter',
           'mutual_information_from_ci', 'make_casscf_mi_theta_fn']


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
        _, _, ci2, _, _, _ = _casscf_solution(
            C0_, h_, e_, th_, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
            pairs=pairs, ci_solver=ci_solver, eig_solver=eig_solver)
        return entropy_from_ci_masks(ci2, masks)

    return s1_theta


def pair_density_from_scatter(c_ci, inv, pat, sign, ngroups):
    """The pair density ``rho = V^T V`` from precomputed scatter (jax).

    The differentiable sibling of ``analysis.ci.pair_density``: the CI
    vector enters only through the weighted segment sum

        V[g, mu] = sum_{det in group g} c_det * sign_det * [pat_det = mu]

    whose transpose (the backward pass) is a gather, so the assembly is
    smooth in ``c_ci``. ``inv``, ``pat``, ``sign`` and ``ngroups`` come
    from :func:`diffmrc.analysis.ci.pair_scatter` and depend on the
    determinant list alone -- hoist them out of any differentiation
    loop (the same discipline as the masks above).
    """
    from jax import numpy as jnp

    c = jnp.asarray(c_ci, dtype=jnp.float64)
    W = ((jnp.asarray(pat)[:, None] == jnp.arange(16)[None, :])
         * jnp.asarray(sign, dtype=jnp.float64)[:, None])
    V = (jnp.zeros((ngroups, 16), dtype=jnp.float64)
         .at[jnp.asarray(inv)].add(W * c[:, None]))
    return V.T @ V


def mutual_information_from_ci(c_ci, dets, scatters, ncore=0, nact=2):
    """``I_ij = s1_i + s1_j - s2_ij`` over the pairs in ``scatters`` (jax).

    ``scatters`` maps ``(i, j)`` (ascending, the order
    :func:`diffmrc.analysis.ci.pair_scatter` enforces) to the tuple it
    returns. Only the listed pairs are computed: the full matrix costs
    one 16x16 eigenproblem per pair, so a caller asking about one
    candidate boundary passes just those pairs instead of the whole
    triangle. The result is symmetric by construction and the diagonal
    is zero by convention, matching the numpy sibling
    (:func:`diffmrc.analysis.ci.mutual_information`), which a test
    binds this route to at 1e-12.

    The eigenvalue pass uses ``eigvalsh``: the entropy is a symmetric
    function of the spectrum, and the eigenvalue derivative carries no
    1/gap, so degeneracies that would poison an eigenvector route are
    harmless here.
    """
    from jax import numpy as jnp

    from ..analysis.entropy import eigenvalue_entropy

    s1 = single_orbital_entropy_from_ci(c_ci, dets, ncore, nact)
    I = jnp.zeros((nact, nact), dtype=jnp.asarray(s1).dtype)
    for (i, j), (inv, pat, sign, ngroups) in scatters.items():
        if not 0 <= i < j < nact:
            raise ValueError(f'pair ({i}, {j}) outside the active space '
                             f'of {nact} orbitals')
        rho = pair_density_from_scatter(c_ci, inv, pat, sign, ngroups)
        s2 = eigenvalue_entropy(jnp.linalg.eigvalsh(rho), xp=jnp)
        value = s1[i] + s1[j] - s2
        I = I.at[i, j].set(value)
        I = I.at[j, i].set(value)
    return I


def make_casscf_mi_theta_fn(C0_, h_, e_, dets, *, ncore, nact, nele, nmo,
                            pairs, ci_solver='plain',
                            eig_solver='default'):
    """``theta -> I(theta)`` of the chain's CI vector, differentiable.

    The mutual-information sibling of ``make_casscf_entropy_theta_fn``:
    the same ``_casscf_solution`` produces the CI vector, and the pair
    scatters are built once (the determinants are static), so
    differentiating touches only the segment sums, one 16x16
    eigenproblem per pair, and the shared entropy formula. A caller
    reads the derivative to see which pair entanglements the orbital
    parameters actually influence -- the stability question behind a
    selection boundary drawn at a mutual-information threshold.
    """
    from ..analysis.ci import pair_scatter
    from .nevpt2 import _casscf_solution

    scatters = {}
    for i in range(nact):
        for j in range(i + 1, nact):
            scatters[(i, j)] = pair_scatter(dets, i, j, ncore=ncore,
                                            nact=nact)

    def I_theta(th_):
        _, _, ci2, _, _, _ = _casscf_solution(
            C0_, h_, e_, th_, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
            pairs=pairs, ci_solver=ci_solver, eig_solver=eig_solver)
        return mutual_information_from_ci(ci2, dets, scatters,
                                          ncore=ncore, nact=nact)

    return I_theta
