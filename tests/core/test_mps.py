"""Tests for diffmrc.core.mps (graduated DMRG-line utilities).

The synthetic fixtures need only numpy.  The molecule regression is
guarded by an importorskip on pyscf; it pins the calibrated N2/6-31G
reference for the recorded start window {6,7} under the pinned basis
(total energy -108.8679136653, measured 2026-10-16; see the module
docstring for the physics record and the reproduction margins).
"""
import numpy as np
import pytest

from diffmrc.core.mps import pin_degenerate_columns


def _random_orthogonal(rng, n):
    q, _r = np.linalg.qr(rng.normal(size=(n, n)))
    return q


def _same_up_to_column_sign(A, B, atol=1e-14):
    """True when A and B agree column-wise up to a global sign per column.

    Non-degenerate columns pass through the pinning untouched except for
    the phase convention (per-column sign of the max-|entry|), so the
    bitwise-untouched assertion is 'equal up to sign'.
    """
    assert A.shape == B.shape
    for c in range(A.shape[1]):
        if np.allclose(A[:, c], B[:, c], atol=atol):
            continue
        if np.allclose(A[:, c], -B[:, c], atol=atol):
            continue
        return False
    return True


def _pair_fixture(seed=7, n=6):
    """Random orthogonal C with a designed 2-fold group at columns (2,3).

    x2 is built so that its restriction to the pair span has a real
    eigenvalue gap; the input pair block is a random mixture of those
    axes, mimicking the process-dependent representatives.
    """
    rng = np.random.default_rng(seed)
    C = _random_orthogonal(rng, n)
    # mix the pair columns so the input is NOT x2-aligned
    theta = rng.uniform(0.3, 1.2)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    C[:, 2:4] = C[:, 2:4] @ rot
    eps = np.array([0.0, 1.0, 2.0, 2.0, 5.0, 6.0])
    # x2: identity on the pair axes' orthogonal complement, eigenvalues
    # (3.0, 1.0) along the pair's internal axes e1/e2
    axes = _random_orthogonal(rng, n)
    d = np.array([0.0, 0.0, 3.0, 1.0, 0.0, 0.0])
    x2 = axes @ np.diag(d) @ axes.T
    x2 = 0.5 * (x2 + x2.T)
    return C, eps, x2


def test_pair_axes_order_span_phases():
    C, eps, x2 = _pair_fixture()
    out, n_rot = pin_degenerate_columns(C, eps, x2)
    assert n_rot == 2
    P = out[:, 2:4]
    # (a) x2 block diagonal to numerical precision
    blk = P.T @ x2 @ P
    assert abs(blk[0, 1]) < 1e-12
    # (b) descending <x2>
    assert blk[0, 0] > blk[1, 1] + 0.5
    # (c) pair span preserved
    assert np.allclose(P @ P.T, C[:, 2:4] @ C[:, 2:4].T, atol=1e-12)
    # (d) non-degenerate columns untouched up to the phase convention
    keep = [0, 1, 4, 5]
    assert _same_up_to_column_sign(out[:, keep], C[:, keep])
    # only the phase may differ: norms are bitwise preserved
    assert np.array_equal(np.abs(out[:, keep]), np.abs(C[:, keep]))
    # (e) phase convention
    for c in range(out.shape[1]):
        imax = int(np.argmax(np.abs(out[:, c])))
        assert out[imax, c] > 0
    # (f) idempotent
    out2, n_rot2 = pin_degenerate_columns(out, eps, x2)
    assert n_rot2 == 2
    assert np.allclose(out2, out, atol=1e-14)


def test_invariance_to_input_rotation_inside_group():
    C, eps, x2 = _pair_fixture(seed=11)
    rng = np.random.default_rng(3)
    theta = rng.uniform(0, np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    C2 = C.copy()
    C2[:, 2:4] = C2[:, 2:4] @ rot
    out1, _ = pin_degenerate_columns(C, eps, x2)
    out2, _ = pin_degenerate_columns(C2, eps, x2)
    assert np.allclose(out1, out2, atol=1e-12)


def test_group_window_and_gap_handling():
    C, eps, x2 = _pair_fixture(seed=5)
    eps = np.array([0.0, 1.0, 2.0, 2.0 + 0.5e-7, 5.0, 6.0])
    out, n_rot = pin_degenerate_columns(C, eps, x2)      # inside tol
    assert n_rot == 2
    eps2 = np.array([0.0, 1.0, 2.0, 2.0 + 5.0e-7, 5.0, 6.0])
    out2, n_rot2 = pin_degenerate_columns(C, eps2, x2)   # outside tol
    assert n_rot2 == 0
    assert _same_up_to_column_sign(out2, C)
    assert np.array_equal(np.abs(out2), np.abs(C))


def test_no_groups_changes_only_phases():
    rng = np.random.default_rng(2)
    C = _random_orthogonal(rng, 5)
    eps = np.arange(5.0)
    x2 = rng.normal(size=(5, 5))
    x2 = 0.5 * (x2 + x2.T)
    out, n_rot = pin_degenerate_columns(C, eps, x2)
    assert n_rot == 0
    assert _same_up_to_column_sign(out, C)
    assert np.array_equal(np.abs(out), np.abs(C))


def test_n2_6_31g_start_window_reference():
    """Calibrated molecule regression (needs pyscf).

    Raw RHF for N2/6-31G at 1.10 A: -108.867618373058 (three engines
    agreed to 3e-13).  The raw CAS(2,2) on the {6,7} columns is NOT
    reproducible across processes (0.02 Ha scatter); under the pinned
    basis it is -108.8679136653 with n_rot = 8 (four degenerate pairs),
    stable across fresh builds to the printed digits.
    """
    pytest.importorskip('pyscf')
    from pyscf import gto, scf, mcscf, fci
    mol = gto.M(atom='N 0 0 0; N 0 0 1.10', basis='6-31g', charge=0,
                spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    assert abs(mf.e_tot - (-108.867618373058)) < 1e-9
    rx = mol.intor('int1e_r')[0]
    C, n_rot = pin_degenerate_columns(np.asarray(mf.mo_coeff),
                                      np.asarray(mf.mo_energy), rx @ rx)
    assert n_rot == 8
    mf2 = mf.copy()
    mf2.mo_coeff = C
    mc = mcscf.CASCI(mf2, 2, 2)
    mc.fcisolver = fci.direct_spin1.FCI(mol)
    mc.fcisolver.spin = 0
    mc.fcisolver.conv_tol = 1e-10
    mc.fcisolver.max_cycle = 500
    e_tot = float(mc.kernel()[0])
    assert abs(e_tot - (-108.8679136653)) < 1e-9
