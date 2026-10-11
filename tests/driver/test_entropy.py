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


def _h2_chain():
    from pyscfad import gto, scf

    from diffmrc.core.orblib import orbital_pairs

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    ncore, nact, nele, nmo = 0, 2, 2, mol.nao_nr()
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    C0 = jnp.asarray(mf.mo_coeff)
    h_ = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    e_ = jnp.asarray(mol.intor('int2e', aosym='s1'))
    dets = cas_dets_for2(ncore, nact, nele)
    return C0, h_, e_, dets, ncore, nact, nele, nmo, pairs


def test_the_entropy_factory_matches_the_direct_extraction():
    pytest.importorskip("pyscfad")
    from diffmrc.driver.entropy import make_casscf_entropy_theta_fn
    from diffmrc.driver.nevpt2 import _casscf_solution

    C0, h_, e_, dets, ncore, nact, nele, nmo, pairs = _h2_chain()
    th0 = jnp.zeros((len(pairs),))
    fn = make_casscf_entropy_theta_fn(
        C0, h_, e_, dets, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
        pairs=pairs)
    s_factory = np.asarray(fn(th0))
    _, _, ci2, _, _, _ = _casscf_solution(C0, h_, e_, th0, ncore=ncore,
                                          nact=nact, nele=nele, nmo=nmo,
                                          pairs=pairs)
    s_direct = entropy_from_ci(np.asarray(ci2), dets, ncore=ncore,
                               nact=nact)
    assert np.allclose(s_factory, s_direct, atol=1e-12)


def _scatters_for(dets, nact):
    from diffmrc.analysis.ci import pair_scatter
    return {(i, j): pair_scatter(dets, i, j, 0, nact)
            for i in range(nact) for j in range(i + 1, nact)}


def test_the_jax_pair_density_matches_the_numpy_one():
    from diffmrc.analysis.ci import pair_density, pair_scatter
    from diffmrc.driver.entropy import pair_density_from_scatter

    dets, c = _sz0_vector(0, 4, 4)
    inv, pat, sign, ngroups = pair_scatter(dets, 0, 2, 0, 4)
    rho_np = pair_density(c, dets, 0, 2, ncore=0, nact=4)
    rho_jx = np.asarray(pair_density_from_scatter(
        jnp.asarray(c), inv, pat, sign, ngroups))
    assert np.allclose(rho_jx, rho_np, atol=1e-12)


def test_the_jax_mutual_information_reproduces_the_numpy_matrix():
    from diffmrc.analysis.ci import mutual_information
    from diffmrc.driver.entropy import mutual_information_from_ci

    dets, c = _sz0_vector(0, 4, 4)
    scatters = _scatters_for(dets, 4)
    I_np = mutual_information(c, dets, ncore=0, nact=4)
    I_jx = np.asarray(mutual_information_from_ci(
        jnp.asarray(c), dets, scatters, ncore=0, nact=4))
    assert np.allclose(I_jx, I_np, atol=1e-12)
    assert np.allclose(np.diag(I_jx), 0.0, atol=1e-14)


def test_the_mi_gradient_matches_a_finite_difference_of_the_numpy_route():
    """AD differentiates the same function the numpy route evaluates."""
    from diffmrc.analysis.ci import mutual_information
    from diffmrc.driver.entropy import mutual_information_from_ci

    dets, c = _sz0_vector(0, 4, 4)
    scatters = _scatters_for(dets, 4)
    v = np.random.default_rng(11).normal(size=c.shape)

    def f_np(x):
        return float(mutual_information(x, dets, ncore=0, nact=4)[0, 1])

    eps = 1e-6
    fd = (f_np(c + eps * v) - f_np(c - eps * v)) / (2 * eps)

    def f_jx(x):
        I = mutual_information_from_ci(x, dets, scatters, ncore=0,
                                       nact=4)
        return I[0, 1]

    g = jax.grad(f_jx)(jnp.asarray(c))
    ad = float(jnp.asarray(g) @ jnp.asarray(v))
    assert np.isfinite(ad)
    assert abs(ad - fd) < 1e-6 * max(1.0, abs(fd))


def test_the_mi_factory_matches_the_direct_extraction():
    pytest.importorskip("pyscfad")
    from diffmrc.analysis.ci import mutual_information
    from diffmrc.driver.entropy import make_casscf_mi_theta_fn
    from diffmrc.driver.nevpt2 import _casscf_solution

    C0, h_, e_, dets, ncore, nact, nele, nmo, pairs = _h2_chain()
    th0 = jnp.zeros((len(pairs),))
    fn = make_casscf_mi_theta_fn(
        C0, h_, e_, dets, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
        pairs=pairs)
    I_factory = np.asarray(fn(th0))
    _, _, ci2, _, _, _ = _casscf_solution(C0, h_, e_, th0, ncore=ncore,
                                          nact=nact, nele=nele, nmo=nmo,
                                          pairs=pairs)
    I_direct = mutual_information(np.asarray(ci2), dets, ncore=ncore,
                                  nact=nact)
    assert np.allclose(I_factory, I_direct, atol=1e-12)


def test_the_mi_factory_is_differentiable_in_theta():
    pytest.importorskip("pyscfad")
    from diffmrc.driver.entropy import make_casscf_mi_theta_fn
    from diffmrc.driver.nevpt2 import _casscf_solution

    C0, h_, e_, dets, ncore, nact, nele, nmo, pairs = _h2_chain()
    th0 = jnp.zeros((len(pairs),))
    fn = make_casscf_mi_theta_fn(
        C0, h_, e_, dets, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
        pairs=pairs)

    def value(th):
        return fn(th)[0, 1]

    g = jax.grad(value)(th0)
    assert np.isfinite(np.asarray(g)).all()

    v = np.random.default_rng(5).normal(size=(len(pairs),))
    # finite difference of the numpy route, one direction
    h = 1e-6

    def f_direct(th):
        _, _, ci2, _, _, _ = _casscf_solution(
            C0, h_, e_, jnp.asarray(th), ncore=ncore, nact=nact,
            nele=nele, nmo=nmo, pairs=pairs)
        from diffmrc.analysis.ci import mutual_information as mi_np
        return float(mi_np(np.asarray(ci2), dets, ncore=ncore,
                           nact=nact)[0, 1])

    fd = (f_direct(np.asarray(th0) + h * v)
          - f_direct(np.asarray(th0) - h * v)) / (2 * h)
    ad = float(jnp.asarray(g) @ jnp.asarray(v))
    assert abs(ad - fd) < 1e-5 * max(1.0, abs(fd))


def test_the_entropy_gradient_matches_a_finite_difference():
    pytest.importorskip("pyscfad")
    from diffmrc.driver.entropy import make_casscf_entropy_theta_fn

    C0, h_, e_, dets, ncore, nact, nele, nmo, pairs = _h2_chain()
    npar = len(pairs)
    fn = make_casscf_entropy_theta_fn(
        C0, h_, e_, dets, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
        pairs=pairs)
    th0 = jnp.zeros((npar,))
    v = jnp.asarray(np.random.default_rng(3).normal(size=npar))
    v = v / jnp.linalg.norm(v)
    g = jax.grad(lambda t: jnp.sum(fn(t)))(th0)
    ad = float(g @ v)
    eps = 1e-5
    fd = float((jnp.sum(fn(th0 + eps * v))
                - jnp.sum(fn(th0 - eps * v))) / (2 * eps))
    assert np.isfinite(ad)
    assert abs(ad - fd) < 1e-7 * max(1.0, abs(fd))
