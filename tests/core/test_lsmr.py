"""Least-squares response solves: no iteration cap, deflation, fallback.

These tests use dense synthetic operators, so they run anywhere; the
molecular checks live in ``test_hessian.py`` and skip without pyscfad.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, lsmr

from diffmrc.core.lsmr import (deflated_solve, lsmr_solve, near_null_basis,
                               sym_ortho)


def _spd(n, cond, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return 0.5 * (q @ np.diag(np.logspace(-np.log10(cond), 0, n)) @ q.T
                  + (q @ np.diag(np.logspace(-np.log10(cond), 0, n)) @ q.T).T)


def _soft_mode_operator(n=60, seed=3):
    """Diagonally dominant with a localized near-null direction.

    The smallest eigenvalue is 1e-7 of the largest, which is the
    regime the orbital Hessian reaches on CO/6-31G.
    """
    rng = np.random.default_rng(seed)
    d = np.concatenate([[1e-7], np.linspace(1e-2, 1.0, 19),
                        np.linspace(1.0, 90.0, n - 20)])
    g = rng.normal(size=(n, n))
    a = np.diag(d) + 0.01 * (g + g.T) / np.sqrt(n)
    return 0.5 * (a + a.T)


def test_sym_ortho_is_a_rotation():
    for a, b in ((3.0, 4.0), (0.0, 2.0), (-2.0, 0.0), (1e-300, 1.0)):
        c, s, r = sym_ortho(a, b)
        assert c * c + s * s == pytest.approx(1.0, abs=1e-14)
        assert c * a + s * b == pytest.approx(r, rel=1e-12, abs=1e-14)
        assert -s * a + c * b == pytest.approx(0.0, abs=1e-12)


def test_lsmr_matches_scipy_iterate_for_iterate():
    """The recurrences are scipy's; only the control differs."""
    a = _spd(30, 1e3, 5)
    b = np.random.default_rng(7).normal(size=30)
    ao = LinearOperator((30, 30), matvec=lambda x: a @ x,
                        rmatvec=lambda x: a @ x, dtype=float)
    x_scipy = lsmr(ao, b, atol=0.0, btol=1e-14, conlim=1e15, maxiter=8)[0]
    x_mine, info = lsmr_solve(lambda v: a @ v, b, rtol=1e-14, maxiter=8)
    assert info['istop'] == 7
    assert np.linalg.norm(x_mine - x_scipy) < 1e-10 * np.linalg.norm(x_scipy)


def test_no_iteration_cap_by_default():
    """scipy stops at min(m, n); the point of this module is not to.

    A condition number of 1e6 needs more than n iterations to reach a
    1e-12 relative residual, which is exactly the regime the response
    equation lives in.
    """
    n = 40
    a = _spd(n, 1e6, 11)
    b = np.random.default_rng(13).normal(size=n)
    x_scipy, istop, itn, _ = lsmr(
        LinearOperator((n, n), matvec=lambda v: a @ v,
                       rmatvec=lambda v: a @ v, dtype=float),
        b, atol=0.0, btol=1e-12, conlim=1e15, maxiter=None)[:4]
    assert istop == 7 and itn == n            # capped, not converged
    x_mine, info = lsmr_solve(lambda v: a @ v, b, rtol=1e-12)
    assert info['itn'] > n
    assert info['reason'] in ('relative_residual', 'machine_residual')
    pinv = np.linalg.pinv(a, rcond=1e-12) @ b
    assert np.linalg.norm(x_mine - pinv) < 1e-6 * np.linalg.norm(pinv)
    assert np.linalg.norm(x_mine - pinv) < np.linalg.norm(x_scipy - pinv)


def test_stagnation_guard_terminates():
    """Without a cap, a round-off floor must not become an endless loop."""
    n = 20
    rng = np.random.default_rng(17)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = np.concatenate([[0.0], np.logspace(-6, 0, n - 1)])
    a = q @ np.diag(d) @ q.T
    a = 0.5 * (a + a.T)
    b = rng.normal(size=n)                     # inconsistent: needs the floor
    x, info = lsmr_solve(lambda v: a @ v, b, rtol=1e-16, atol=0.0,
                         stall_iters=20)
    assert info['istop'] in (1, 2, 4, 10)
    assert info['itn'] < 500


def test_normal_equations_stop_fires_on_a_singular_operator():
    n = 30
    rng = np.random.default_rng(19)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = np.concatenate([[0.0], np.logspace(-5, 2, n - 1)])
    a = q @ np.diag(d) @ q.T
    a = 0.5 * (a + a.T)
    b = rng.normal(size=n)                     # has a null-space component
    x, info = lsmr_solve(lambda v: a @ v, b, rtol=1e-14, atol=1e-12)
    assert info['reason'] == 'normal_equations'
    pinv = np.linalg.pinv(a, rcond=1e-9) @ b
    assert np.linalg.norm(x - pinv) < 1e-6 * np.linalg.norm(pinv)


def test_near_null_basis_is_certified_on_a_soft_mode_operator():
    a = _soft_mode_operator()
    n = a.shape[0]
    basis, ritz, info = near_null_basis(lambda v: a @ v, n, diag=np.diag(a))
    assert info['certified']
    assert basis.shape == (n, info['n_wanted'])
    assert info['ritz_residual'] < 1e-8 * info['lambda_max']
    assert np.abs(basis.T @ basis - np.eye(basis.shape[1])).max() < 1e-10
    exact = np.linalg.eigvalsh(a)
    assert np.abs(np.sort(ritz) - exact[:ritz.size]).max() < 1e-6


def test_near_null_basis_refuses_a_delocalized_operator():
    """A capped block that cannot resolve the soft directions must say so."""
    n = 60
    rng = np.random.default_rng(23)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = np.concatenate([np.linspace(1e-6, 1e-2, 5), np.linspace(0.2, 100, 55)])
    a = 0.5 * (q @ np.diag(d) @ q.T + (q @ np.diag(d) @ q.T).T)
    basis, ritz, info = near_null_basis(lambda v: a @ v, n, max_size=24)
    assert not info['certified']


def test_deflated_solve_reproduces_the_pseudo_inverse():
    a = _soft_mode_operator()
    n = a.shape[0]
    b = np.random.default_rng(29).normal(size=n)
    basis, ritz, info = near_null_basis(lambda v: a @ v, n, diag=np.diag(a))
    x, dinfo = deflated_solve(lambda v: a @ v, b, basis, ritz, rcond=1e-9,
                              rtol=1e-12)
    pinv = np.linalg.pinv(a, rcond=1e-9) @ b
    assert dinfo['reason'] == 'converged'
    assert np.linalg.norm(x - pinv) < 1e-8 * np.linalg.norm(pinv)
    assert dinfo['deflated'] == basis.shape[1]


def test_deflation_drops_directions_below_rcond():
    """A direction under the cutoff must contribute nothing, as in pinv."""
    n = 40
    rng = np.random.default_rng(31)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = np.concatenate([[1e-12], np.linspace(0.5, 50, n - 1)])
    a = 0.5 * (q @ np.diag(d) @ q.T + (q @ np.diag(d) @ q.T).T)
    b = rng.normal(size=n)
    basis, ritz, info = near_null_basis(lambda v: a @ v, n, diag=np.diag(a))
    assert info['certified']
    x, dinfo = deflated_solve(lambda v: a @ v, b, basis, ritz, rcond=1e-9,
                              rtol=1e-12)
    # the null direction is q[:, 0]; its component in x must vanish
    null_component = float(q[:, 0] @ x)
    pinv = np.linalg.pinv(a, rcond=1e-9) @ b
    assert abs(null_component) < 1e-8 * np.linalg.norm(pinv)
    assert np.linalg.norm(x - pinv) < 1e-6 * np.linalg.norm(pinv)


def test_deflated_solve_is_accurate_on_a_soft_mode_operator():
    """The case that breaks a residual-only criterion.

    With five near-null directions the plain solve satisfies its
    residual test and is still wrong by 1e-4 relative; the deflated
    solve reproduces the pseudo-inverse.
    """
    n = 60
    rng = np.random.default_rng(37)
    d = np.concatenate([np.linspace(1e-9, 1e-7, 5), np.linspace(1.0, 90, n - 5)])
    g = rng.normal(size=(n, n))
    a = np.diag(d) + 0.01 * (g + g.T) / np.sqrt(n)
    a = 0.5 * (a + a.T)
    b = rng.normal(size=n)
    pinv = np.linalg.pinv(a, rcond=1e-9) @ b
    basis, ritz, info = near_null_basis(lambda v: a @ v, n, diag=np.diag(a))
    assert info['certified']
    x, dinfo = deflated_solve(lambda v: a @ v, b, basis, ritz, rcond=1e-9,
                              rtol=1e-12)
    assert np.linalg.norm(x - pinv) < 1e-8 * np.linalg.norm(pinv)


def test_fallback_returns_the_plain_answer_when_the_basis_is_bad():
    """An unconverged deflated solution must not be returned as the answer."""
    n = 50
    rng = np.random.default_rng(41)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = np.concatenate([np.linspace(1e-6, 1e-2, 4), np.linspace(0.2, 100, 46)])
    a = 0.5 * (q @ np.diag(d) @ q.T + (q @ np.diag(d) @ q.T).T)
    b = rng.normal(size=n)
    # a deliberately useless one-vector basis
    basis = q[:, [10]]
    ritz = np.array([d[10]])
    x_def, info_def = deflated_solve(lambda v: a @ v, b, basis, ritz,
                                     rcond=1e-12, rtol=1e-12, fallback=True)
    x_plain, info_plain = lsmr_solve(lambda v: a @ v, b, rtol=1e-12,
                                     atol=1e-12)
    if info_def['reason'] == 'fallback_lsmr':
        assert np.allclose(x_def, x_plain, rtol=0, atol=1e-12)
    else:
        assert info_def['reason'] == 'converged'
