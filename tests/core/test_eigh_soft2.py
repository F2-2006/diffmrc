"""``eigh_soft2``: a second-order-safe softened rule for ``jnp.linalg.eigh``.

Regression for the P0.5 boundary (roadmap 2026-09-23): every rule in this
module was first-order safe, but the second derivative on matrices with
*bit-exact* degenerate pairs was either NaN (``eigh``, ``eigvalsh``,
``eigh_safe``, ``eigh_mask``) or carried the cluster-tolerance band of
``eigh_cluster_ift``. ``eigh_soft2`` freezes the degenerate cluster
identically at every order and softens the physical couplings instead of
taking the exact 1/gap, which bounds them by 1/(2 tau) and makes the
second derivative finite without a hard spike.

The fixture is a 4x4 mirror matrix whose spectrum is two bit-exact
doubled pairs (gaps 0.0, sqrt(2), 0.0; verified bit-equal through
``np.linalg.eigvalsh`` and ``jnp.linalg.eigh``), i.e. the CO/6-31G
blocked-CI structure in miniature. Measurements recorded 2026-10-16
(symmetric probe directions only -- a non-symmetric probe exercises the
rules' differing reading of the tangent and is not meaningful):

  - values: du = 0.0 (bit-equal) on the fixture, 2.2e-15 on a control;
  - first order vs ``eigh_safe`` on well-separated 5x5 matrices:
    max |du| deviation 6.4e-15, |dw| 2.8e-17 (same tau, same Lorentzian);
  - AD HVP on the fixture (gauge-invariant loss): +0.593, finite, versus
    the unclustered Lorentzian's +7.5e6 (exactly 1/tau^2-scaled);
  - FD of the AD gradient (the accurate second-order route for this
    family): 0.717 against the value-level reference 0.704 (1.9%) -- the
    frozen rule's own AD second derivative reads 16% there, the same
    frozen-vs-FD gap documented at t269.

The tests assert the shipped behaviour only; the raw-rule NaN remains a
property of the jax build, not of this repository.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _fixture():
    """4x4 with two bit-exact degenerate pairs and cross couplings."""
    return jnp.asarray(np.array([[1.0, 0.0, 0.5, 0.0],
                                 [0.0, 1.0, 0.0, 0.5],
                                 [0.5, 0.0, 2.0, 0.0],
                                 [0.0, 0.5, 0.0, 2.0]]))


def _control():
    """4x4 with all gaps physical (no pair within the cluster tol)."""
    return jnp.asarray(np.array([[1.00, 0.0, 0.5, 0.0],
                                 [0.0, 1.03, 0.0, 0.5],
                                 [0.5, 0.0, 2.0, 0.0],
                                 [0.0, 0.5, 0.0, 2.06]]))


def _probe_dirs(n=4, seed=7):
    rng = np.random.default_rng(seed)
    sym = lambda m: 0.5 * (m + m.T)  # noqa: E731
    return (jnp.asarray(sym(rng.normal(size=(n, n)))),
            jnp.asarray(sym(rng.normal(size=(n, n)))))


def _gauge_invariant_loss(fn, n=4):
    """Equal weights within each bit-exact cluster + cluster projectors.

    Mirrors the production chains: the energy is invariant under the
    arbitrary basis inside a degenerate subspace, so the loss depends on
    the cluster spans, not on individual columns.
    """
    bm = np.array([[0.3, -0.2, 0.1, 0.4], [-0.2, 0.7, 0.5, -0.1],
                   [0.1, 0.5, -0.4, 0.2], [0.4, -0.1, 0.2, 0.9]])
    bm = jnp.asarray(0.5 * (bm + bm.T))
    cw = jnp.asarray([1.0, 2.5])[: n // 2]

    def f(a):
        w, u = fn(a)
        out = jnp.sum(w * jnp.repeat(cw, 2))
        for k in range(n // 2):
            p = u[:, 2 * k:2 * k + 2] @ u[:, 2 * k:2 * k + 2].T
            out = out + jnp.sum(p * bm)
        return out

    return f


def test_values_are_the_plain_eigh_values():
    """Correction factors are zero at the evaluation point, values untouched."""
    from diffmrc.core.eigh import eigh_soft2

    for A in (_fixture(), _control()):
        w_ref, u_ref = jnp.linalg.eigh(A)
        w, u = eigh_soft2(A)
        assert float(jnp.abs(w - w_ref).max()) < 1e-12
        assert float(jnp.abs(u - u_ref).max()) < 1e-12


def test_the_first_order_rule_agrees_with_eigh_safe():
    """Same tau, same Lorentzian: the two rules are bit-equal at first order.

    Measured max deviations 6.4e-15 (vectors) and 2.8e-17 (values) over
    three seeds; the threshold below is a margin, not the measurement.
    """
    from diffmrc.core.eigh import eigh_safe, eigh_soft2

    sym = lambda m: 0.5 * (m + m.T)  # noqa: E731
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        B = rng.normal(size=(5, 5))
        A = jnp.asarray(B + B.T + np.diag([0.0, 2.0, 5.0, 9.0, 14.0]))
        V = jnp.asarray(sym(rng.normal(size=(5, 5))))
        du_s = np.asarray(jax.jvp(lambda x: eigh_safe(x)[1], (A,), (V,))[1])
        du_2 = np.asarray(jax.jvp(lambda x: eigh_soft2(x)[1], (A,), (V,))[1])
        dw_s = np.asarray(jax.jvp(lambda x: eigh_safe(x)[0], (A,), (V,))[1])
        dw_2 = np.asarray(jax.jvp(lambda x: eigh_soft2(x)[0], (A,), (V,))[1])
        assert np.abs(du_s - du_2).max() < 1e-12
        assert np.abs(dw_s - dw_2).max() < 1e-12


def test_second_order_is_finite_on_a_bitexact_pair():
    """jvp(grad) and hessian through the fixture: all finite.

    The plain rule and ``eigh_safe`` return NaN here (the B9 boundary);
    the unclustered softened variant reads +7.5e6 (1/tau^2 gauge
    curvature). This asserts the shipped rule: finite and of the
    value-level order (+0.593 measured against value-FD +0.704).
    """
    from diffmrc.core.eigh import eigh_soft2

    A = _fixture()
    V1, V2 = _probe_dirs()
    f = _gauge_invariant_loss(eigh_soft2)
    g = jax.grad(f)
    hv = np.asarray(jax.jvp(g, (A,), (V2,))[1])
    assert np.isfinite(hv).all()
    ad = float(jnp.sum(V1 * hv))
    assert abs(ad) < 100.0            # unclustered variant: 7.5e6
    H = np.asarray(jax.hessian(f)(A))
    assert np.isfinite(H).all()
    Hv1 = np.asarray(jax.jvp(g, (A,), (V1,))[1])
    # Hessian symmetry: v1 . (H v2) = v2 . (H v1)
    assert abs(float(jnp.sum(V1 * hv)) - float(jnp.sum(V2 * Hv1))) < 1e-6


def test_fd_of_the_gradient_matches_the_value_level_reference():
    """The FD of the AD gradient is the accurate second-order route.

    On the fixture (bit-exact pairs, gauge-invariant loss) the finite
    difference of the AD gradient reproduces the value-level central
    difference of the loss: measured 0.717 vs 0.704 (1.9%), while the
    frozen AD HVP itself reads 16% off -- the documented frozen-vs-FD
    gap. The threshold is a margin around the measurement.
    """
    from diffmrc.core.eigh import eigh_soft2

    A = _fixture()
    V1, V2 = _probe_dirs()
    f_s = _gauge_invariant_loss(eigh_soft2)
    f_v = _gauge_invariant_loss(jnp.linalg.eigh)
    g = jax.grad(f_s)
    d = 1e-4
    fd_grad = float(jnp.sum(V2 * (g(A + d * V1) - g(A - d * V1)) / (2 * d)))
    fd_val = (float(f_v(A + d * V1 + d * V2)) - float(f_v(A + d * V1 - d * V2))
              - float(f_v(A - d * V1 + d * V2)) + float(f_v(A - d * V1 - d * V2))
              ) / (4 * d * d)
    assert abs(fd_grad - fd_val) / abs(fd_val) < 0.1


def test_a_split_pair_stays_exact_and_finite():
    """The theta*-style near-degenerate band (split ~1e-6) is clustered.

    The pair split by 3e-6 sits inside the default tol = 1e-4 and is
    frozen like a bit-exact one: values stay the plain eigh values and
    the gradient is finite.
    """
    from diffmrc.core.eigh import eigh_soft2

    A = jnp.asarray(np.array([[1.0, 0.0, 0.5, 0.0],
                              [0.0, 1.0 + 3e-6, 0.0, 0.5],
                              [0.5, 0.0, 2.0, 0.0],
                              [0.0, 0.5, 0.0, 2.0]]))
    w_ref, u_ref = jnp.linalg.eigh(A)
    w, u = eigh_soft2(A)
    assert float(jnp.abs(w - w_ref).max()) < 1e-12
    assert float(jnp.abs(u - u_ref).max()) < 1e-12
    g = np.asarray(jax.grad(_gauge_invariant_loss(eigh_soft2))(A))
    assert np.isfinite(g).all()
