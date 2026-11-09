"""Implicit-function second derivatives, verified against explicit answers."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core.hessian import (  # noqa: E402
    ImplicitHessianResult, implicit_hessian, implicit_hessian_columnwise,
    implicit_hessian_hvp)


def test_result_exposes_the_response_term():
    """Every field the assessment needs must reach the caller.

    The response contribution is the whole point of the route; a result
    that exposes only the total leaves a caller unable to tell whether
    the frozen approximation would have sufficed.
    """
    fields = set(ImplicitHessianResult.__dataclass_fields__)
    for name in ('H_tt', 'H_tt_inverse', 'curvature', 'mixed', 'H_RR',
                 'H_response', 'H_total', 'dtheta_dR_max', 'rcond'):
        assert name in fields


def test_rcond_is_recorded_in_the_result():
    """A reader cannot judge a response term without knowing the cutoff."""
    assert 'rcond' in ImplicitHessianResult.__dataclass_fields__


@jax.tree_util.register_pytree_node_class
class _ScalarMole:
    """Stand-in whose 'coordinates' are a single differentiable number.

    Registered as a pytree so the blocked and matrix-free routes can
    push a tangent through it exactly as they do through a pyscfad Mole.
    """

    def __init__(self, value):
        self.coords = jnp.asarray([float(value)])

    def tree_flatten(self):
        return (self.coords,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        new = cls(0.0)
        new.coords = children[0]
        return new

    def replace(self, coords):
        new = _ScalarMole(0.0)
        new.coords = coords
        return new


def _assemble(E_ad, theta_star, x0):
    """Reproduce the module's assembly on a scalar surrogate."""
    H_tt = np.atleast_2d(np.asarray(jax.hessian(lambda t: E_ad(t, x0))(theta_star)))
    H_tt = 0.5 * (H_tt + H_tt.T)
    M = np.atleast_2d(np.asarray(
        jax.jacfwd(jax.grad(lambda t, x: E_ad(t, x), argnums=0), argnums=1)(
            theta_star, x0)).reshape(1, 1))
    th_sg = jax.lax.stop_gradient(theta_star)
    H_RR = np.atleast_2d(np.asarray(
        jax.hessian(lambda x: E_ad(th_sg, x))(x0)).reshape(1, 1))
    H_RR = 0.5 * (H_RR + H_RR.T)
    D = -np.linalg.pinv(H_tt, rcond=1e-9) @ M
    total = H_RR + M.T @ D
    return H_tt, M, H_RR, D, 0.5 * (total + total.T)


def test_assembly_reproduces_an_analytic_second_derivative():
    """The end-to-end check on a problem with a closed-form answer.

    E(t, x) = (t - x)^2 + x^2 has its stationary point at t* = x, so the
    relaxed energy is E*(x) = x^2 and its second derivative is exactly 2.
    The implicit assembly must reproduce that from the three separately
    differentiated pieces.
    """
    def E_ad(tt, xx):
        return (tt[0] - xx) ** 2 + xx ** 2

    th = jnp.asarray([3.0])
    x0 = jnp.asarray(3.0)
    H_tt, M, H_RR, D, total = _assemble(E_ad, th, x0)

    # The pieces individually, so a wrong one is identifiable.
    assert H_tt[0, 0] == pytest.approx(2.0, abs=1e-12)
    assert M[0, 0] == pytest.approx(-2.0, abs=1e-12)
    assert H_RR[0, 0] == pytest.approx(4.0, abs=1e-12)
    assert D[0, 0] == pytest.approx(1.0, abs=1e-12)
    # And the assembled total against the analytic answer.
    assert total[0, 0] == pytest.approx(2.0, abs=1e-12)


def test_assembly_reproduces_a_two_dimensional_analytic_hessian():
    """E* = diag(1, 2) for a decoupled two-parameter surrogate."""
    def E_ad(u, v):
        return (0.5 * 3.0 * (u[0] - v[0]) ** 2
                + 0.5 * 5.0 * (u[1] - v[1]) ** 2
                + 0.5 * 1.0 * v[0] ** 2
                + 0.5 * 2.0 * v[1] ** 2)

    us = jnp.asarray([0.7, -0.3])
    vs = jnp.asarray([0.7, -0.3])
    H_tt = np.asarray(jax.hessian(lambda u: E_ad(u, vs))(us))
    H_tt = 0.5 * (H_tt + H_tt.T)
    M = np.asarray(jax.jacfwd(jax.grad(lambda u, v: E_ad(u, v), argnums=0),
                              argnums=1)(us, vs))
    th_sg = jax.lax.stop_gradient(us)
    H_RR = np.asarray(jax.hessian(lambda v: E_ad(th_sg, v))(vs))
    H_RR = 0.5 * (H_RR + H_RR.T)
    D = -np.linalg.pinv(H_tt, rcond=1e-9) @ M
    total = H_RR + M.T @ D
    total = 0.5 * (total + total.T)

    assert np.allclose(total, np.diag([1.0, 2.0]), atol=1e-10)


def test_the_response_term_is_not_negligible():
    """If the response term vanished the route would be pointless.

    On the surrogate above it carries most of the total, which is why
    the frozen approximation is not good enough and why the result
    exposes it separately.
    """
    def E_ad(tt, xx):
        return (tt[0] - xx) ** 2 + xx ** 2

    th = jnp.asarray([3.0])
    x0 = jnp.asarray(3.0)
    H_tt, M, H_RR, D, total = _assemble(E_ad, th, x0)
    response = M.T @ D
    assert abs(response[0, 0]) > 0.5 * abs(total[0, 0])


def test_frozen_theta_is_not_differentiated():
    """The frozen Hessian must treat theta as a constant.

    Dropping stop_gradient lets the frozen term acquire the orbital
    response, and the two routes stop being distinguishable.
    """
    def E(u, x):
        return (u - x) ** 2 + 0.5 * x ** 2

    th = jax.lax.stop_gradient(jnp.asarray(1.0))
    g = jax.grad(lambda x: E(th, x))(jnp.asarray(1.0))
    # d/dx [ (1 - x)^2 + 0.5 x^2 ] at x = 1 is 0 + 1 = 1
    assert float(g) == pytest.approx(1.0, abs=1e-12)


def test_a_singular_orbital_hessian_needs_the_pseudo_inverse():
    """Why the route uses pinv rather than solve.

    A direction with zero curvature makes the solve singular; the
    pseudo-inverse returns the finite answer on the remaining subspace,
    which is correct because the gradient along a null direction is zero.
    """
    def E(u):
        return 0.5 * 2.0 * u[0] ** 2 + 1.0 * u[1]

    H = np.asarray(jax.hessian(E)(jnp.zeros(2)))
    w = np.linalg.eigvalsh(H)
    assert w[0] == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(H, np.ones(2))
    assert np.all(np.isfinite(np.linalg.pinv(H, rcond=1e-9) @ np.ones(2)))


def test_the_pseudo_inverse_and_the_solve_disagree_on_a_flat_direction():
    """The two are not interchangeable on an ill-conditioned matrix."""
    n = 8
    rng = np.random.default_rng(20260922)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    w = np.array([1.0, 1e-2, 1e-4, 1e-8, 1e-10, 1e-12, 1e-12, 1e-12])
    H = (Q * w) @ Q.T
    H = 0.5 * (H + H.T)
    g = Q @ np.ones(n)
    x_pinv = np.linalg.pinv(H, rcond=1e-9) @ g
    x_solve = np.linalg.solve(H, g)
    assert np.linalg.norm(x_solve) > 10 * np.linalg.norm(x_pinv)


def test_implicit_hessian_requires_a_mole_and_a_joint_energy():
    """The public entry point only needs those two things."""
    import inspect
    params = list(inspect.signature(implicit_hessian).parameters)
    assert params[:3] == ['mol', 'theta', 'E_joint']
    assert 'rcond' in params


# ---------------------------------------------------------------------------
# Blocked and matrix-free routes
# ---------------------------------------------------------------------------
def _E_scalar_mole(m, t):
    """The analytic surrogate with the joint-energy argument order."""
    x = m.coords[0]
    return (t[0] - x) ** 2 + x ** 2


def test_columnwise_assembly_reproduces_the_analytic_hessian():
    """Column-by-column traversal is the same assembly, not an approximation.

    The closed-form surrogate has H_tt = 2, M = -2, H_RR = 4 and
    H_total = 2.  A blocked route that changed any of those would be
    detectable here without a molecular calculation.
    """
    th = jnp.asarray([3.0])
    x0 = _ScalarMole(3.0)
    res = implicit_hessian_columnwise(
        x0, th, _E_scalar_mole,
        tangent_fn=lambda m, v: _ScalarMole(float(v[0])))

    assert res.H_tt[0, 0] == pytest.approx(2.0, abs=1e-12)
    assert res.mixed[0, 0] == pytest.approx(-2.0, abs=1e-12)
    assert res.H_RR[0, 0] == pytest.approx(4.0, abs=1e-12)
    assert res.H_total[0, 0] == pytest.approx(2.0, abs=1e-12)


def test_hvp_route_reproduces_the_analytic_hessian():
    """The matrix-free response solve is exact on a nonsingular surrogate."""
    th = jnp.asarray([3.0])
    x0 = _ScalarMole(3.0)
    res = implicit_hessian_hvp(
        x0, th, _E_scalar_mole,
        tangent_fn=lambda m, v: _ScalarMole(float(v[0])),
        atol=1e-13, btol=1e-13)

    assert res.H_total[0, 0] == pytest.approx(2.0, abs=1e-9)
    assert res.H_tt_diagonal[0] == pytest.approx(2.0, abs=1e-12)
    # 1 = relative residual, 2 = normal equations, 4 = machine floor.
    # On a one-dimensional operator the first iterate is already at the
    # floor, so 4 is the expected code here.
    assert res.lsmr_istop[0] in (0, 1, 2, 4)
    assert res.maxiter is None           # no cap by default
    assert res.deflation['certified'] in (True, False)


def test_the_hvp_route_reports_its_deflation_decision():
    """A reader must be able to see whether the small block was deflated.

    The scalar surrogate has one parameter and one eigenvalue, far above
    any deflation threshold relative to itself, so nothing is wanted and
    the solve must fall back to the full operator without claiming a
    deflation it did not make.
    """
    th = jnp.asarray([3.0])
    x0 = _ScalarMole(3.0)
    res = implicit_hessian_hvp(
        x0, th, _E_scalar_mole,
        tangent_fn=lambda m, v: _ScalarMole(float(v[0])),
        atol=1e-13, btol=1e-13)
    info = res.deflation
    assert info is not None
    assert info['n_wanted'] == 0
    assert info['certified'] is False
    assert res.solver_reason[0] in ('relative_residual', 'machine_residual',
                                    'normal_equations', 'stagnation')


def test_the_deflated_route_matches_the_plain_route_on_a_surrogate():
    """Turning deflation on must not change the answer."""
    th = jnp.asarray([3.0])
    x0 = _ScalarMole(3.0)
    kw = dict(tangent_fn=lambda m, v: _ScalarMole(float(v[0])),
              atol=1e-13, btol=1e-13)
    a = implicit_hessian_hvp(x0, th, _E_scalar_mole, deflate=False, **kw)
    b = implicit_hessian_hvp(x0, th, _E_scalar_mole, deflate=True, **kw)
    assert np.abs(a.H_total - b.H_total).max() < 1e-12


def _lih_sto3g():
    """A small pyscfad system that is cheap enough for a unit test.

    H2O/STO-3G already costs about 8 minutes for the three routes on
    the reference server; LiH has 11 orbital parameters and 6 geometric
    directions and exercises the same code paths.
    """
    pytest.importorskip('pyscfad')
    from pyscfad import gto, scf

    from diffmrc.core.hessian import (make_joint_energy_fn,
                                      make_theta_energy_fn)
    from diffmrc.core.orblib import orbital_pairs

    mol = gto.Mole()
    mol.atom = 'Li 0 0 0; H 0 0 1.6'
    mol.basis = 'sto-3g'
    mol.verbose = 0
    mol.build()
    nmo = mol.nao_nr()
    ncore, nact, nele = 1, 2, 2
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    E_joint = make_joint_energy_fn(ncore, nact, nele, nmo, pairs, scf,
                                   scf=scf.RHF)
    E_tt = make_theta_energy_fn(mol, ncore, nact, nele, nmo, pairs, scf,
                                scf=scf.RHF)
    return mol, jnp.zeros(len(pairs)), E_joint, E_tt


def test_columnwise_matches_vectorized_on_a_molecular_energy():
    """LiH/STO-3G: the two traversal orders produce the same matrices."""
    mol, theta, E_joint, E_tt = _lih_sto3g()
    dense = implicit_hessian(mol, theta, E_joint)
    col = implicit_hessian_columnwise(mol, theta, E_joint, E_tt=E_tt)
    assert np.abs(col.H_tt - dense.H_tt).max() < 1e-10
    assert np.abs(col.mixed - dense.mixed).max() < 1e-10
    assert np.abs(col.H_RR - dense.H_RR).max() < 1e-10
    assert np.abs(col.H_total - dense.H_total).max() < 1e-10


def test_hvp_matches_the_dense_assembly_and_is_symmetric():
    """LiH/STO-3G: matrix-free columns and HVP symmetry.

    The system is small enough to compare every column against the
    dense assembly; the symmetry check uses two random directions so it
    also exercises the transpose action of M.
    """
    mol, theta, E_joint, E_tt = _lih_sto3g()
    dense = implicit_hessian(mol, theta, E_joint)
    hvp = implicit_hessian_hvp(mol, theta, E_joint, E_tt=E_tt,
                              atol=1e-9, btol=1e-9)
    assert np.abs(hvp.H_total - dense.H_total).max() < 1e-6
    assert np.abs(hvp.H_tt_diagonal - np.diag(dense.H_tt)).max() < 1e-8

    rng = np.random.default_rng(20261026)
    n = hvp.n_geom
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    hu = hvp.hvp(u)
    hv = hvp.hvp(v)
    assert abs(float(u @ hv) - float(v @ hu)) < 1e-5 * max(
        1.0, float(np.abs(u @ hv)))


class _FixedField:
    """A mean field that returns one fixed coefficient matrix."""

    def __init__(self, C):
        self.mo_coeff = jnp.asarray(C)

    def kernel(self):
        return None


def test_the_theta_chart_is_invariant_to_reference_column_signs():
    """A frozen theta must describe the same physical rotation whatever
    sign the eigensolver returned for the reference columns.  The joint
    and the theta-only factories both canonicalize the columns before
    the rotation; without it the two flipped references give different
    energies and gradients at the same theta."""
    pytest.importorskip('pyscfad')
    from pyscfad import gto

    from diffmrc.core.hessian import (make_joint_energy_fn,
                                      make_theta_energy_fn)
    from diffmrc.core.orblib import orbital_pairs

    mol = gto.Mole()
    mol.atom = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
    mol.basis = '6-31g'
    mol.verbose = 0
    mol.build()
    nmo = mol.nao_nr()
    ncore, nact, nele = 4, 2, 2
    pairs = orbital_pairs(ncore, nact, nmo)
    npar = len(pairs)
    rng = np.random.default_rng(31)
    C, _ = np.linalg.qr(rng.normal(size=(nmo, nmo)))
    C_flip = C.copy()
    C_flip[:, ::3] *= -1.0
    theta = jnp.asarray(0.01 * np.arange(npar))

    def theta_fn(Cx):
        return make_theta_energy_fn(
            mol, ncore, nact, nele, nmo, pairs, None,
            scf=lambda m: _FixedField(Cx))

    a = theta_fn(C)
    b = theta_fn(C_flip)
    assert float(a(theta)) == pytest.approx(float(b(theta)), abs=1e-12)
    assert np.abs(np.asarray(jax.grad(a)(theta))
                  - np.asarray(jax.grad(b)(theta))).max() < 1e-12

    def joint_fn(Cx):
        return make_joint_energy_fn(ncore, nact, nele, nmo, pairs, None,
                                    scf=lambda m: _FixedField(Cx))

    ja = joint_fn(C)
    jb = joint_fn(C_flip)
    assert float(ja(mol, theta)) == pytest.approx(
        float(jb(mol, theta)), abs=1e-12)
    ga = np.asarray(jax.grad(lambda t: ja(mol, t))(theta))
    gb = np.asarray(jax.grad(lambda t: jb(mol, t))(theta))
    assert np.abs(ga - gb).max() < 1e-12
