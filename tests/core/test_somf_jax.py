"""Contracts of the differentiable SOMF layer (``diffmrc.core.somf_jax``).

The value-layer contracts are covered by ``test_somf.py``.  This file
checks that the JAX path reproduces them, that reverse-mode differentiation
gives the analytic density-response contraction of ``somf_ecp``, and that
the SI splitting is differentiable at exact degeneracy.

The AD-FD comparisons use a synthetic two-state problem built from the
water density; the OH/OF acceptance run lives in
``experiments/somf/validate_oh_of_jax.py``.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import expm  # noqa: E402
from pyscf import gto  # noqa: E402

from diffmrc.core import somf as dsomf  # noqa: E402
from diffmrc.core import somf_ecp as decp  # noqa: E402
from diffmrc.core import somf_jax as jsomf  # noqa: E402


WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    s = mol.intor('int1e_ovlp')
    p = (mol.nelectron / mol.nao_nr()) * np.linalg.inv(s)
    p = 0.5 * (p + p.T)
    return mol, p


@pytest.fixture(scope='module')
def ints(water):
    mol, _ = water
    return jsomf.somf_integrals(mol)


def _physical_tensor(nstates=2, seed=3):
    """Transition tensor with T[i,i] = 0 and T[j,i] = -T[i,j]."""
    rng = np.random.default_rng(seed)
    t = np.zeros((nstates, nstates, 3))
    for i in range(nstates):
        for j in range(i + 1, nstates):
            v = rng.normal(size=3)
            t[i, j] = v
            t[j, i] = -v
    return t


def _orthonormal_active(mol, nact, seed=5):
    """S-orthonormal active orbitals, ``C^T S C = I``."""
    s = mol.intor('int1e_ovlp')
    w, v = np.linalg.eigh(s)
    x = v @ np.diag(w ** -0.5) @ v.T
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(mol.nao_nr(), nact)))[0]
    c = x @ q
    return c


def _active_problem(mol, p, nact=6, nstates=2, seed=7):
    """Synthetic SI problem consistent with the water density."""
    c = _orthonormal_active(mol, nact)
    d_act = c.T @ p @ c
    rng = np.random.default_rng(seed)
    d = np.zeros((nstates, nstates, nact, nact))
    for i in range(nstates):
        a = rng.normal(size=(nact, nact))
        d[i, i] = a + a.T
    r = rng.normal(size=(nact, nact))
    d[0, 1] = r
    d[1, 0] = r.T
    es = np.asarray([0.0, 0.02])
    return c, d_act, d, es


def test_integrals_match_value_layer(water, ints):
    mol, _ = water
    nao = mol.nao_nr()
    h1 = np.asarray(ints.h1)
    g = np.asarray(ints.g)
    assert h1.dtype == np.float64 and g.dtype == np.float64
    assert np.abs(h1 - dsomf.one_electron_soc(mol)).max() == 0.0
    ref = mol.intor('int2e_p1vxp1', aosym='s1').reshape(3, nao, nao, nao, nao)
    assert np.abs(g - ref).max() == 0.0


def test_two_electron_matches_value_layer(water, ints):
    mol, p = water
    h_jax = np.asarray(jsomf.somf_two_electron_jax(p, ints.g))
    h_ref = dsomf.somf_two_electron(mol, p, contract='explicit')
    scale = max(np.abs(h_ref).max(), 1.0)
    assert np.abs(h_jax - h_ref).max() / scale < 1e-13


def test_matrix_matches_value_layer(water, ints):
    mol, p = water
    c = np.linalg.qr(np.arange(mol.nao_nr() ** 2, dtype=float)
                     .reshape(mol.nao_nr(), mol.nao_nr()))[0][:, :7]
    h_ao = np.asarray(jsomf.somf_matrix_jax(p, ints.h1, ints.g))
    ref_ao = dsomf.somf_matrix(mol, None, p, contract='explicit')
    assert np.abs(h_ao - ref_ao).max() / np.abs(ref_ao).max() < 1e-13

    h_mo = np.asarray(jsomf.somf_matrix_jax(p, ints.h1, ints.g, C=c))
    ref_mo = dsomf.somf_matrix(mol, c, p, contract='explicit')
    assert np.abs(h_mo - ref_mo).max() / np.abs(ref_mo).max() < 1e-13

    h_eh = np.asarray(jsomf.somf_matrix_jax(p, ints.h1, ints.g,
                                            prefactor=True))
    assert np.abs(h_eh - dsomf.SOMF_ALPHA2_HALF * h_ao).max() == 0.0

    h_1e = np.asarray(jsomf.somf_matrix_jax(None, ints.h1, ints.g,
                                            mode='one_electron'))
    assert np.abs(h_1e - np.asarray(ints.h1)).max() == 0.0


def test_two_electron_is_linear_and_zero(water, ints):
    _, p = water
    h_a = jsomf.somf_two_electron_jax(p, ints.g)
    h_b = jsomf.somf_two_electron_jax(2.0 * p, ints.g)
    assert np.abs(np.asarray(h_b - 2.0 * h_a)).max() < 1e-12 * \
        max(np.abs(np.asarray(h_a)).max(), 1.0)
    h0 = jsomf.somf_two_electron_jax(jnp.zeros_like(p), ints.g)
    assert np.abs(np.asarray(h0)).max() == 0.0


def test_two_electron_is_antisymmetric(water, ints):
    _, p = water
    h = np.asarray(jsomf.somf_two_electron_jax(p, ints.g))
    assert np.abs(h + h.transpose(0, 2, 1)).max() < 1e-13


def test_density_response_matches_analytic_contraction(water, ints):
    """AD of ``sum D h2[P]`` equals ``somf_ecp.somf_density_response``."""
    mol, p = water
    rng = np.random.default_rng(11)
    d = rng.normal(size=(mol.nao_nr(),) * 2)

    def weighted(x, w):
        h2 = jsomf.somf_two_electron_jax(x, ints.g)
        return jnp.sum(w[:, None, None] * d[None, :, :] * h2)

    m_ref = decp.somf_density_response(mol, d, contract='explicit')
    for k in range(3):
        w = np.zeros(3)
        w[k] = 1.0
        g = np.asarray(jax.grad(lambda x, w=w: weighted(x, w))(p))
        ref = np.asarray(m_ref[k])
        assert np.abs(g - ref).max() / max(np.abs(ref).max(), 1.0) < 1e-12

    w = rng.normal(size=3)
    g = np.asarray(jax.grad(lambda x: weighted(x, w))(p))
    ref = np.einsum('k,krs->rs', w, np.asarray(m_ref))
    assert np.abs(g - ref).max() / max(np.abs(ref).max(), 1.0) < 1e-12


def test_si_assembly_matches_the_reference_convention():
    """Same numbers as the value-layer SI script, including s = sigma/2."""
    from pyscf.data import nist
    es = np.asarray([0.0, 0.02])
    t = _physical_tensor()

    def numpy_splitting(es, t):
        spin = 0.5 * np.asarray([
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]]], dtype=complex)
        pref = 1j * float(nist.ALPHA) ** 2 / 2.0
        h = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            h[2 * i:2 * i + 2, 2 * i:2 * i + 2] += es[i] * np.eye(2)
            for j in range(2):
                for k in range(3):
                    h[2 * i:2 * i + 2, 2 * j:2 * j + 2] += \
                        pref * t[i, j, k] * spin[k]
        w = np.linalg.eigvalsh(h)
        return float((w[2] - w[0]) * 219474.6313632)

    ref = numpy_splitting(es, t)
    got = float(jsomf.si_splitting_jax(es, t))
    assert abs(got - ref) < 1e-10 * max(abs(ref), 1.0)


def test_doubled_form_has_the_complex_spectrum():
    es = np.asarray([0.0, 0.02])
    t = _physical_tensor()
    h_complex = np.asarray(jsomf.si_hamiltonian_jax(es, t))
    h_doubled = np.asarray(jsomf.si_hamiltonian_doubled_jax(es, t))
    assert np.abs(h_doubled - h_doubled.T).max() < 1e-15
    w_complex = np.linalg.eigvalsh(h_complex)
    w_doubled = np.linalg.eigvalsh(h_doubled)
    assert w_doubled.shape == (2 * w_complex.shape[0],)
    pairs = np.repeat(w_complex, 2)
    assert np.abs(w_doubled - pairs).max() < 1e-13 * np.abs(pairs).max()
    s_doubled = float(jsomf.si_splitting_jax(es, t, route='doubled'))
    s_complex = float(jsomf.si_splitting_jax(es, t, route='complex'))
    assert abs(s_doubled - s_complex) < 1e-9 * max(abs(s_complex), 1.0)


def test_splitting_gradient_matches_fd_in_soc_scale():
    es = np.asarray([0.0, 0.02])
    t = _physical_tensor()

    def f(c):
        return jsomf.si_splitting_jax(es, c * t)

    ad = float(jax.grad(f)(1.0))
    for h in (1e-4, 1e-5):
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2.0 * h))
        assert abs(ad - fd) <= 1e-6 * max(abs(fd), 1.0), (h, ad, fd)


def test_both_splitting_routes_give_the_same_gradient():
    es = np.asarray([0.0, 0.02])
    t = _physical_tensor()

    def f(c, route):
        return jsomf.si_splitting_jax(es, c * t, route=route)

    g_doubled = float(jax.grad(lambda c: f(c, 'doubled'))(1.0))
    g_complex = float(jax.grad(lambda c: f(c, 'complex'))(1.0))
    assert abs(g_doubled - g_complex) <= 1e-8 * max(abs(g_doubled), 1.0)


def test_jit_grad_of_splitting_is_finite_and_matches_eager():
    es = np.asarray([0.0, 0.02])
    t = _physical_tensor()
    f = lambda c: jsomf.si_splitting_jax(es, c * t)  # noqa: E731
    g_eager = float(jax.grad(f)(1.0))
    g_jit = float(jax.grad(jax.jit(f))(1.0))
    assert np.isfinite(g_jit)
    assert abs(g_jit - g_eager) <= 1e-10 * max(abs(g_eager), 1.0)
    def f_d(c):
        return jsomf.si_splitting_jax(es, c * t, route='doubled')

    g_doubled = float(jax.grad(jax.jit(f_d))(1.0))
    assert abs(g_doubled - g_eager) <= 1e-8 * max(abs(g_eager), 1.0)


def test_splitting_gradient_is_finite_at_exact_degeneracy():
    es = np.asarray([0.0, 0.02])
    t = _physical_tensor()

    def f(c):
        return jsomf.si_splitting_jax(es, c * t)

    # c = 0: the 4x4 SI matrix is exactly block-degenerate (E0, E0, E1, E1).
    ad = float(jax.grad(f)(0.0))
    assert np.isfinite(ad)
    assert abs(ad) < 1e-9
    ad_jit = float(jax.grad(jax.jit(f))(0.0))
    assert np.isfinite(ad_jit)
    assert abs(ad_jit) < 1e-9
    value = float(f(0.0))
    assert np.isfinite(value)
    assert abs(value - 0.02 * jsomf.CM_PER_EH) < 1e-8


def test_zero_soc_tensor_is_finite():
    es = np.asarray([0.0, 0.02])
    t = np.zeros((2, 2, 3))
    s = float(jsomf.si_splitting_jax(es, t))
    assert np.isfinite(s)
    g = jax.grad(lambda x: jsomf.si_splitting_jax(es, x))(t)
    assert np.all(np.isfinite(np.asarray(g)))


def test_end_to_end_density_ad_matches_fd(water, ints):
    mol, p = water
    c, _, d, es = _active_problem(mol, p)

    def f(scale):
        return jsomf.somf_splitting_jax((1.0 + scale) * p,
                                        ints.h1, ints.g, c, es, d)

    ad = float(jax.grad(f)(0.0))
    assert abs(float(f(0.0))) > 1.0  # nonzero SOC scale
    for h in (1e-4, 1e-5):
        fd = float((f(h) - f(-h)) / (2.0 * h))
        assert abs(ad - fd) <= 1e-6 * max(abs(fd), 1.0), (h, ad, fd)


def test_end_to_end_orbital_rotation_ad_matches_fd(water, ints):
    """Rotate the active orbitals and the active density together."""
    mol, p = water
    c, d_act, d, es = _active_problem(mol, p)
    nact = c.shape[1]
    rng = np.random.default_rng(13)
    a = rng.normal(size=(nact, nact))
    a = (a - a.T) / np.linalg.norm(a - a.T)
    p_inactive = p - c @ d_act @ c.T

    def f(theta):
        u = expm(theta * a)
        cr = c @ u
        pr = p_inactive + cr @ d_act @ cr.T
        return jsomf.somf_splitting_jax(pr, ints.h1, ints.g, cr, es, d)

    ad = float(jax.grad(f)(0.0))
    assert abs(float(f(0.0))) > 1.0
    for h in (1e-4, 1e-5):
        fd = float((f(h) - f(-h)) / (2.0 * h))
        assert abs(ad - fd) <= 1e-5 * max(abs(fd), 1.0), (h, ad, fd)


def test_jit_grad_agrees(water, ints):
    _, p = water
    f = jax.jit(lambda x: jnp.sum(jsomf.somf_two_electron_jax(x, ints.g)))
    g_jit = np.asarray(jax.grad(f)(p))
    g_raw = np.asarray(jax.grad(
        lambda x: jnp.sum(jsomf.somf_two_electron_jax(x, ints.g)))(p))
    assert np.abs(g_jit - g_raw).max() <= 1e-12 * max(np.abs(g_raw).max(), 1.0)


def test_explicit_kernel_memory_estimate(water):
    mol, _ = water
    nao = mol.nao_nr()
    assert jsomf.explicit_kernel_bytes(nao) == 24 * nao ** 4
    with pytest.raises(MemoryError):
        jsomf.somf_integrals(mol, memory_limit=1)


def test_invalid_inputs(water, ints):
    mol, p = water
    nao = mol.nao_nr()
    with pytest.raises(ValueError):
        jsomf.somf_two_electron_jax(p[:-1], ints.g)
    with pytest.raises(ValueError):
        jsomf.somf_two_electron_jax(p, ints.g[:, :, :-1])
    with pytest.raises(ValueError):
        jsomf.somf_matrix_jax(p, ints.h1, ints.g, mode='nonsense')
    with pytest.raises(ValueError):
        jsomf.somf_matrix_jax(None, ints.h1, ints.g)
    with pytest.raises(ValueError):
        jsomf.somf_matrix_jax(p, ints.h1, ints.g,
                              C=np.ones((nao - 1, 2)))
    with pytest.raises(ValueError):
        jsomf.si_hamiltonian_jax(np.zeros(2), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        jsomf.si_hamiltonian_doubled_jax(np.zeros(2), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        jsomf.si_splitting_jax(np.zeros(2), np.zeros((2, 2, 3)),
                               route='nonsense')
