"""Contracts of the differentiable RI/DF layer.

The conventions under test are written down in ``diffmrc.core.ri``; the
sign and axis claims there are the ones that pyscf/libcint deliver on this
stack (verified independently by isolated finite differences, see
the development validation script).  These tests are regression guards for that
module, not a replacement for the validation script: the script measures
the physical RI error and the chain-level gradients, which are too slow
for a unit test.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402

from diffmrc.core import ri as dri  # noqa: E402


WATER = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0'


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='6-31g', unit='Angstrom', verbose=0)
    auxmol, label = dri.make_auxmol(mol)
    factors = dri.ri_factors(mol, auxmol)
    return mol, auxmol, factors, label


def _build_like(mol, coords_bohr):
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    return gto.M(atom=[(symbols[i], c) for i, c in enumerate(coords_bohr)],
                 basis=mol.basis, unit='Bohr', verbose=0)


def _displaced(mol, atom, axis, step):
    coords = mol.atom_coords()
    cp = coords.copy(); cp[atom, axis] += step
    cm = coords.copy(); cm[atom, axis] -= step
    return cp, cm


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def _pair_symmetrize(g):
    g = 0.25 * (g + g.transpose(1, 0, 2, 3)
                + g.transpose(0, 1, 3, 2) + g.transpose(1, 0, 3, 2))
    return 0.5 * (g + g.transpose(2, 3, 0, 1))


def _ao2mo(c, eri):
    t = np.tensordot(eri, c, axes=([3], [0]))
    t = np.tensordot(t, c, axes=([2], [0]))
    t = np.tensordot(t, c, axes=([1], [0]))
    t = np.tensordot(t, c, axes=([0], [0]))
    return t.transpose(3, 2, 1, 0)


def test_auxmol_reports_basis_and_shapes(water):
    mol, auxmol, factors, label = water
    assert auxmol.nao_nr() > 0
    assert label
    assert factors.b.shape == (mol.nao_nr(), mol.nao_nr(), auxmol.nao_nr())
    assert factors.s.shape == (auxmol.nao_nr(), auxmol.nao_nr())
    assert factors.jhalf.shape == factors.s.shape


def test_axis_order_and_pair_symmetry(water):
    mol, auxmol, factors, _ = water
    b = factors.b
    # the two AO indices are first and symmetric, the auxiliary index is last
    assert np.abs(b - b.transpose(1, 0, 2)).max() < 1e-12
    assert np.abs(factors.s - factors.s.T).max() < 1e-12
    # the auxiliary index is the last and has the auxiliary dimension; the
    # assembly identity test below pins the (mu, nu, P) interpretation
    assert b.shape[2] == factors.naux


def test_metric_inverse_sqrt(water):
    _, _, factors, _ = water
    s, j = factors.s, factors.jhalf
    eye = np.eye(s.shape[0])
    assert np.abs(j @ s @ j - eye).max() < 1e-9
    assert np.abs(j - j.T).max() < 1e-12


def test_metric_rcond_truncation():
    # a metric with one numerically dead direction: the truncated inverse
    # square root must not diverge and must annihilate that direction
    w = np.array([1.0, 2.0, 1e-14])
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))
    s = (q * w) @ q.T
    j = dri.inv_sqrt_metric(s, rcond=1e-12)
    assert np.isfinite(j).all()
    assert np.abs(j @ s @ j - (np.eye(3) - np.outer(q[:, 2], q[:, 2]))).max() < 1e-10


def test_inv_sqrt_metric_jvp_matches_fd():
    rng = np.random.default_rng(3)
    n = 6
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w = np.linspace(1.0, 2.0, n)
    s = (q * w) @ q.T
    ds = rng.standard_normal((n, n)); ds = 0.5 * (ds + ds.T)
    an = dri.inv_sqrt_metric_jvp(s, ds)
    fd = (dri.inv_sqrt_metric(s + 1e-6 * ds)
          - dri.inv_sqrt_metric(s - 1e-6 * ds)) / 2e-6
    assert _rel(an, fd) < 1e-8


def test_assembly_identity_against_metric_solve(water):
    mol, auxmol, factors, _ = water
    eri = dri.ri_eri_ao(mol, auxmol, factors=factors)
    naux = factors.naux
    bf = factors.b.reshape(-1, naux)
    x = np.linalg.solve(factors.s, bf.T)
    ref = (bf @ x).reshape(factors.nao, factors.nao, factors.nao, factors.nao)
    assert np.abs(eri - ref).max() < 1e-10
    # the RI tensor is symmetric under the pair exchange
    assert np.abs(eri - eri.transpose(2, 3, 0, 1)).max() == 0.0


def test_three_center_derivative_fd(water):
    mol, auxmol, factors, _ = water
    step = 1e-4
    for atom in range(mol.natm):
        d = dri.ri_factors_deriv(mol, auxmol, atom, factors)
        for axis in range(3):
            cp, cm = _displaced(mol, atom, axis, step)
            mp, mm = _build_like(mol, cp), _build_like(mol, cm)
            ap, _ = dri.make_auxmol(mp)
            am, _ = dri.make_auxmol(mm)
            fd = (dri.ri_three_center(mp, ap)
                  - dri.ri_three_center(mm, am)) / (2 * step)
            if np.abs(fd).max() < 1e-10:
                continue
            assert _rel(d.db[axis], fd) < 1e-8


def test_metric_derivative_fd(water):
    mol, auxmol, factors, _ = water
    step = 1e-4
    for atom in range(mol.natm):
        d = dri.ri_factors_deriv(mol, auxmol, atom, factors)
        for axis in range(3):
            cp, cm = _displaced(mol, atom, axis, step)
            mp, mm = _build_like(mol, cp), _build_like(mol, cm)
            ap, _ = dri.make_auxmol(mp)
            am, _ = dri.make_auxmol(mm)
            fd = (dri.ri_metric(ap) - dri.ri_metric(am)) / (2 * step)
            if np.abs(fd).max() < 1e-10:
                continue
            assert _rel(d.ds[axis], fd) < 1e-8


def test_eri_derivative_fd(water):
    mol, auxmol, factors, _ = water
    deri = dri.ri_eri_deriv(mol, auxmol, 0, factors=factors)
    step = 1e-4
    for axis in range(3):
        cp, cm = _displaced(mol, 0, axis, step)
        mp, mm = _build_like(mol, cp), _build_like(mol, cm)
        ap, _ = dri.make_auxmol(mp)
        am, _ = dri.make_auxmol(mm)
        fd = (dri.ri_eri_ao(mp, ap) - dri.ri_eri_ao(mm, am)) / (2 * step)
        if np.abs(fd).max() < 1e-10:
            continue
        # the metric inverse square root amplifies the central-difference
        # truncation; the measured deviation at this step is 5e-9
        assert _rel(deri[axis], fd) < 1e-7


def test_contraction_matches_materialized(water):
    mol, auxmol, factors, _ = water
    rng = np.random.default_rng(5)
    gamma = _pair_symmetrize(rng.standard_normal(
        (factors.nao, factors.nao, factors.nao, factors.nao)))
    deri = dri.ri_eri_deriv(mol, auxmol, 0, factors=factors)
    d = dri.ri_factors_deriv(mol, auxmol, 0, factors)
    scalar = dri.ri_eri_deriv_contract(gamma, factors, d)
    vec = dri.ri_eri_deriv_contract(gamma, factors, d, per_axis=True)
    assert vec.shape == (3,)
    assert abs(scalar - float(np.sum(gamma * deri))) < 1e-10
    assert abs(scalar - float(vec.sum())) < 1e-12


def test_mo_transform(water):
    mol, auxmol, factors, _ = water
    rng = np.random.default_rng(7)
    norb = 6
    c, _ = np.linalg.qr(rng.standard_normal((factors.nao, norb)))
    eri = dri.ri_eri_ao(mol, auxmol, factors=factors)
    b_mo, eri_mo = dri.ri_eri_mo(c, factors.b, factors.jhalf)
    # three-center reference, one AO index at a time
    ref = np.tensordot(factors.b, c, axes=([1], [0]))    # [mu,P,j]
    ref = np.tensordot(ref, c, axes=([0], [0]))          # [P,j,i]
    ref = np.matmul(ref.transpose(2, 1, 0), factors.jhalf)
    assert b_mo.shape == (norb, norb, factors.naux)
    assert _rel(b_mo, ref) < 1e-11
    assert eri_mo.shape == (norb, norb, norb, norb)
    assert _rel(eri_mo, _ao2mo(c, eri)) < 1e-11


def test_jax_interface(water):
    mol, auxmol, factors, _ = water
    d = dri.ri_factors_deriv(mol, auxmol, 0, factors)
    eri_np = dri.ri_eri_ao(mol, auxmol, factors=factors)
    eri_jax = np.asarray(dri.ri_eri_from_factors(factors.b, factors.jhalf))
    assert np.abs(eri_jax - eri_np).max() < 1e-12
    deri = dri.ri_eri_deriv(mol, auxmol, 0, factors=factors)
    val, jvp = dri.ri_eri_jvp(factors.b, factors.jhalf, d.db[1], d.djhalf[1])
    assert np.abs(np.asarray(val) - eri_np).max() < 1e-12
    assert _rel(np.asarray(jvp), deri[1]) < 1e-10
    # the custom rule is consistent with ordinary autodiff through the
    # factor contraction
    g = jax.grad(lambda b: dri.ri_eri_from_factors(b, factors.jhalf).sum())(factors.b)
    assert np.isfinite(np.asarray(g)).all()
