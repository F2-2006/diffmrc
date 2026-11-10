"""Contracts of the differentiable RI-SOMF layer.

``test_somf_ri.py`` covers the value layer and ``test_somf_jax.py`` the
four-center differentiable path.  This file checks the RI-SOMF(1X) JAX
path against the value layer, the analytic JVP/VJP against JAX's own
linearization of the plain contraction, the density response against
finite differences, and the factor derivative layer (``db``, ``djhalf``
analytic; ``da`` a central difference because libcint has no
``int3c2e_pvxp1`` derivative).
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from pyscf import gto  # noqa: E402

from diffmrc.core import somf_ri as dsomfri  # noqa: E402
from diffmrc.core import somf_ri_jax as jsomfri  # noqa: E402
from diffmrc.core.ri import ri_metric_deriv  # noqa: E402


WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    s = mol.intor('int1e_ovlp')
    p = (mol.nelectron / mol.nao_nr()) * np.linalg.inv(s)
    p = 0.5 * (p + p.T)
    return mol, p


@pytest.fixture(scope='module')
def ri(water):
    mol, p = water
    auxmol, label = dsomfri.make_auxmol(mol)
    factors = dsomfri.coulomb_ri_factors(mol, auxmol=auxmol)
    ints = jsomfri.somf_ri_integrals(mol, factors=factors)
    return auxmol, label, factors, ints


def _orthonormal_active(mol, nact, seed=5):
    s = mol.intor('int1e_ovlp')
    w, v = np.linalg.eigh(s)
    x = v @ np.diag(w ** -0.5) @ v.T
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(mol.nao_nr(), nact)))[0]
    return x @ q


def _active_problem(mol, p, nact=6, nstates=2, seed=7):
    c = _orthonormal_active(mol, nact)
    rng = np.random.default_rng(seed)
    d = np.zeros((nstates, nstates, nact, nact))
    for i in range(nstates):
        a = rng.normal(size=(nact, nact))
        d[i, i] = a + a.T
    r = rng.normal(size=(nact, nact))
    d[0, 1] = r
    d[1, 0] = r.T
    return c, d, np.asarray([0.0, 0.02])


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()
                 / max(np.abs(np.asarray(b)).max(), 1.0))


def test_integrals_match_value_layer(water, ri):
    mol, _ = water
    auxmol, label, factors, ints = ri
    assert np.abs(np.asarray(ints.a) - factors.a).max() == 0.0
    assert np.abs(np.asarray(ints.b) - factors.b).max() == 0.0
    assert np.abs(np.asarray(ints.jhalf) - factors.jhalf).max() == 0.0
    blocks = dsomfri.orbital_blocks(mol)
    assert len(ints.indices) == len(blocks)
    for idx, g, ref in zip(ints.indices, ints.kernels, blocks):
        assert np.array_equal(np.asarray(idx), ref)
        na = len(ref)
        assert np.asarray(g).shape == (3, na, na, na, na)


def test_two_electron_matches_value_layer(water, ri):
    mol, p = water
    _, _, _, ints = ri
    h_jax = np.asarray(jsomfri.somf_ri_two_electron_jax(p, ints))
    h_ref = (dsomfri.coulomb_ri(mol, p)
             - 1.5 * dsomfri.exchange_one_center(mol, p))
    assert _rel(h_jax, h_ref) < 1e-12


def test_matrix_matches_value_layer(water, ri):
    mol, p = water
    _, _, _, ints = ri
    c = _orthonormal_active(mol, 6)
    h_ao = np.asarray(jsomfri.somf_ri_matrix_jax(p, ints))
    ref_ao = dsomfri.somf_ri_matrix(mol, None, p, mode='ri_somf_1x')
    assert _rel(h_ao, ref_ao) < 1e-12

    h_mo = np.asarray(jsomfri.somf_ri_matrix_jax(p, ints, C=c))
    ref_mo = dsomfri.somf_ri_matrix(mol, c, p, mode='ri_somf_1x')
    assert _rel(h_mo, ref_mo) < 1e-12


def test_ri_jvp_matches_plain_contraction(water, ri):
    _, p = water
    auxmol, _, factors, _ = ri
    nao, naux = factors.nao, factors.naux
    rng = np.random.default_rng(11)
    a = rng.normal(size=(3, nao, nao, naux))
    b = rng.normal(size=(nao, nao, naux))
    jh = rng.normal(size=(naux, naux))
    jh = 0.5 * (jh + jh.T)
    dP = rng.normal(size=(nao, nao))
    da = rng.normal(size=a.shape)
    db = rng.normal(size=b.shape)
    dj = rng.normal(size=jh.shape)
    dj = 0.5 * (dj + dj.T)

    def plain(x, y, z, w):
        ww = jnp.einsum('uvP,uv->P', z, x)
        zz = jnp.matmul(w, jnp.matmul(w, ww))
        return jnp.einsum('kpqP,P->kpq', y, zz)

    val, tan = jsomfri.ri_coulomb_jvp(p, a, b, jh, dP, da, db, dj)
    v_ref, t_ref = jax.jvp(plain, (p, a, b, jh), (dP, da, db, dj))
    assert _rel(val, v_ref) < 1e-14
    assert _rel(tan, t_ref) < 1e-12


def test_ri_vjp_matches_autodiff(water, ri):
    _, p = water
    auxmol, _, factors, _ = ri
    nao, naux = factors.nao, factors.naux
    rng = np.random.default_rng(13)
    a = rng.normal(size=(3, nao, nao, naux))
    b = rng.normal(size=(nao, nao, naux))
    jh = rng.normal(size=(naux, naux))
    jh = 0.5 * (jh + jh.T)
    g = rng.normal(size=(3, nao, nao))

    def plain(x, y, z, w):
        ww = jnp.einsum('uvP,uv->P', z, x)
        zz = jnp.matmul(w, jnp.matmul(w, ww))
        return jnp.einsum('kpqP,P->kpq', y, zz)

    got = jsomfri.ri_coulomb_vjp(p, a, b, jh, g)
    _, pull = jax.vjp(plain, p, a, b, jh)
    ref = pull(g)
    for x, y in zip(got, ref):
        assert _rel(x, y) < 1e-12


def test_one_center_exchange_matches_value_layer(water, ri):
    mol, p = water
    _, _, _, ints = ri
    got = np.asarray(jsomfri.one_center_exchange_jax(
        p, ints.indices, ints.kernels))
    ref = dsomfri.exchange_one_center(mol, p)
    assert _rel(got, ref) < 1e-12


def test_one_center_kernel_is_translation_invariant(water):
    mol, _ = water
    _, kernels0 = dsomfri.exchange_one_center_kernels(mol)
    for axis in range(3):
        h = 1e-4
        plus = dsomfri._displaced_mol(mol, 0, axis, h)
        minus = dsomfri._displaced_mol(mol, 0, axis, -h)
        _, kp = dsomfri.exchange_one_center_kernels(plus)
        _, km = dsomfri.exchange_one_center_kernels(minus)
        dev = max(np.abs(x - y).max() for x, y in zip(kp, km))
        assert dev == 0.0, (axis, dev)


def test_density_response_matches_fd(water, ri):
    mol, p = water
    _, _, _, ints = ri
    c, d, es = _active_problem(mol, p)

    def f(scale):
        return jsomfri.somf_ri_splitting_jax((1.0 + scale) * p, ints, c,
                                             es, d)

    ad = float(jax.grad(f)(0.0))
    assert abs(float(f(0.0))) > 1.0
    for h in (1e-4, 1e-5):
        fd = float((f(h) - f(-h)) / (2.0 * h))
        assert abs(ad - fd) <= 1e-6 * max(abs(fd), 1.0), (h, ad, fd)

    rng = np.random.default_rng(17)
    dp = rng.normal(size=(mol.nao_nr(),) * 2)
    dp = 0.5 * (dp + dp.T)
    g = jax.grad(lambda x: jsomfri.somf_ri_splitting_jax(x, ints, c, es, d))(p)
    g = np.asarray(g)
    for h in (1e-4, 1e-5):
        fp = float(jsomfri.somf_ri_splitting_jax(p + h * dp, ints, c, es, d))
        fm = float(jsomfri.somf_ri_splitting_jax(p - h * dp, ints, c, es, d))
        fd = (fp - fm) / (2.0 * h)
        pred = float(np.sum(g * dp))
        assert abs(pred - fd) <= 1e-5 * max(abs(fd), 1.0), (h, pred, fd)


def test_density_response_is_the_analytic_contraction(water, ri):
    """AD equals ``sum_ij w_ij . M_ij`` with ``M`` built from the factors."""
    mol, p = water
    _, _, _, ints = ri
    c, d, es = _active_problem(mol, p)
    h_ao = np.asarray(jsomfri.somf_ri_matrix_jax(p, ints))
    h_mo = np.einsum('ak,cab,bl->ckl', c, h_ao, c, optimize=True)
    t0 = np.einsum('ijpq,kpq->ijk', d, h_mo)
    w = np.asarray(jax.grad(
        lambda t: jsomfri.si_splitting_jax(jnp.asarray(es), t))(t0))
    d_ao = np.einsum('ca,ijab,db->ijcd', c, d, c, optimize=True)

    m = np.zeros((3, mol.nao_nr(), mol.nao_nr()))
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            m += np.einsum('k,krs->rs', w[i, j],
                           _ri_density_response(ints, d_ao[i, j]))
    g = np.asarray(jax.grad(
        lambda x: jsomfri.somf_ri_splitting_jax(x, ints, c, es, d))(p))
    assert _rel(g, m) < 1e-11


def _ri_density_response(ints, d_ao):
    """``d/dP_rs sum_pq d_ao[pq] h2_k[pq]`` for the RI-SOMF(1X) operator.

    Coulomb: ``z_P = sum_pq d_ao[pq] a[k,p,q,P]``, ``wbar = S^{-1} z``,
    ``M[k,r,s] = sum_P b[rs,P] wbar_P``.  One-center exchange: the block
    kernel contracted with the block of ``d_ao``.
    """
    nao = d_ao.shape[0]
    m = np.zeros((3, nao, nao))
    a = np.asarray(ints.a)
    b = np.asarray(ints.b)
    jh = np.asarray(ints.jhalf)
    for k in range(3):
        z = np.einsum('pq,pqP->P', d_ao, a[k])
        wbar = jh @ (jh @ z)
        m[k] += np.einsum('rsP,P->rs', b, wbar)
    for idx, g in zip(ints.indices, ints.kernels):
        idx = np.asarray(idx)
        d_a = d_ao[np.ix_(idx, idx)]
        g = np.asarray(g)
        m[:, idx[:, None], idx] += -1.5 * (
            np.einsum('pq,kprsq->krs', d_a, g)
            + np.einsum('pq,ksqpr->krs', d_a, g))
    return m


def test_factor_db_matches_fd(water, ri):
    mol, _ = water
    auxmol, label, factors, _ = ri
    d = dsomfri.coulomb_ri_factor_derivs(mol, auxmol, 0, factors)
    h = 1e-5
    fp = dsomfri.coulomb_ri_factors(
        dsomfri._displaced_mol(mol, 0, 2, h),
        auxmol=dsomfri.make_auxmol(dsomfri._displaced_mol(mol, 0, 2, h),
                                   auxbasis=label)[0])
    fm = dsomfri.coulomb_ri_factors(
        dsomfri._displaced_mol(mol, 0, 2, -h),
        auxmol=dsomfri.make_auxmol(dsomfri._displaced_mol(mol, 0, 2, -h),
                                   auxbasis=label)[0])
    db_fd = (fp.b - fm.b) / (2.0 * h)
    assert _rel(d.db[2], db_fd) < 1e-9


def test_factor_djhalf_satisfies_the_sylvester_identity(water, ri):
    mol, _ = water
    auxmol, _, factors, _ = ri
    d = dsomfri.coulomb_ri_factor_derivs(mol, auxmol, 0, factors)
    ds = ri_metric_deriv(auxmol, 0)[2]
    j = factors.jhalf
    resid = d.djhalf[2] @ j + j @ d.djhalf[2] + j @ j @ ds @ j @ j
    scale = np.abs(j @ j @ ds @ j @ j).max()
    assert np.abs(resid).max() / scale < 1e-10


def test_factor_da_matches_fd(water, ri):
    mol, _ = water
    auxmol, label, factors, _ = ri
    d = dsomfri.coulomb_ri_factor_derivs(mol, auxmol, 0, factors,
                                         a_step=1e-4)
    h = 1e-5
    fp = dsomfri.coulomb_ri_factors(
        dsomfri._displaced_mol(mol, 0, 2, h),
        auxmol=dsomfri.make_auxmol(dsomfri._displaced_mol(mol, 0, 2, h),
                                   auxbasis=label)[0])
    fm = dsomfri.coulomb_ri_factors(
        dsomfri._displaced_mol(mol, 0, 2, -h),
        auxmol=dsomfri.make_auxmol(dsomfri._displaced_mol(mol, 0, 2, -h),
                                   auxbasis=label)[0])
    da_fd = (fp.a - fm.a) / (2.0 * h)
    assert _rel(d.da[2], da_fd) < 1e-6


def test_da_analytic_is_marked_unavailable(water, ri):
    mol, _ = water
    auxmol, _, factors, _ = ri
    with pytest.raises(NotImplementedError):
        dsomfri.coulomb_ri_factor_derivs(mol, auxmol, 0, factors,
                                         a_deriv='analytic')


def test_invalid_inputs(water, ri):
    mol, p = water
    auxmol, _, factors, ints = ri
    nao = mol.nao_nr()
    with pytest.raises(ValueError):
        jsomfri.somf_ri_two_electron_jax(p[:-1], ints)
    with pytest.raises(ValueError):
        jsomfri.somf_ri_matrix_jax(p, ints, C=np.ones((nao - 1, 2)))
    with pytest.raises(ValueError):
        dsomfri.coulomb_ri_factor_derivs(mol, auxmol, 0, factors,
                                         a_deriv='nonsense')
    with pytest.raises(MemoryError):
        dsomfri.exchange_one_center_kernels(mol, memory_limit=1)
