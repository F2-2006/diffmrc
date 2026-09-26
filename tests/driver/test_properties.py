"""The dipole assembly: block bookkeeping and the AO contraction.

The assembly is pure numpy on purpose: it is the part of the dipole path
that can be wrong without any jax present, so it is the part pinned
here.  The jax-dependent chain tests skip without jax, so this file
keeps working on a bare interpreter.
"""
import numpy as np
import pytest

from diffmrc.driver.properties import dipole_from_density, mo_density


def test_mo_density_puts_two_on_core_occupation_on_active_zero_elsewhere():
    dm1 = np.array([[1.7, 0.1], [0.1, 0.3]])
    d = mo_density(ncore=2, nact=2, nmo=5, dm1_act=dm1)
    assert np.allclose(np.diag(d), [2.0, 2.0, 1.7, 0.3, 0.0])
    assert np.allclose(d, d.T)
    assert np.isclose(np.trace(d), 4.0 + 2.0)   # 2*ncore + Tr(dm1)


def test_mo_density_rejects_a_mismatched_active_block():
    with pytest.raises(ValueError, match='active'):
        mo_density(ncore=2, nact=2, nmo=5, dm1_act=np.zeros((3, 3)))


def test_dipole_contraction_is_invariant_under_core_rotations():
    """mu must not depend on rotations of the *equally occupied* columns.

    The core block is 2*I, so any rotation inside it cancels exactly
    (Q (2I) Q^T = 2I).  The active block is not isotropic on its own:
    rotating the active columns requires rotating the active density by
    the same orthogonal matrix, which is the second half of the test.
    Both statements together are what make the contraction safe on a
    non-canonical basis and pin the block bookkeeping.
    """
    rng = np.random.default_rng(7)
    nao, nmo = 6, 6
    dip = rng.normal(size=(3, nao, nao))
    dip = 0.5 * (dip + dip.transpose(0, 2, 1))
    C = np.linalg.qr(rng.normal(size=(nao, nmo)))[0]
    dm1 = np.array([[1.5, 0.2], [0.2, 0.5]])
    D = mo_density(2, 2, nmo, dm1)
    zero2, zero23 = np.zeros(2), np.zeros((2, 3))
    mu0 = dipole_from_density(C, D, dip, zero2, zero23)
    Qc = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    Qa = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    C2 = C.copy()
    C2[:, :2] = C[:, :2] @ Qc
    mu_core = dipole_from_density(C2, D, dip, zero2, zero23)
    assert np.allclose(mu0, mu_core, atol=1e-12)
    D2 = mo_density(2, 2, nmo, Qa @ dm1 @ Qa.T)
    C3 = C.copy()
    C3[:, 2:4] = C[:, 2:4] @ Qa
    mu_act = dipole_from_density(C3, D2, dip, zero2, zero23)
    assert np.allclose(mu0, mu_act, atol=1e-12)


def test_translating_a_neutral_molecule_leaves_the_dipole_alone():
    """Shift every atom by t and the dipole must not move.

    Translating the basis functions by t shifts each electronic integral
    by t*S_pq (shown here in an orthonormal AO basis, S = 1) and each
    nuclear term by Z_A t; for a neutral molecule the two shifts cancel
    because Tr(D) equals the total charge.  This is the statement that
    lets dipoles of displaced geometries be differenced at all.
    """
    rng = np.random.default_rng(11)
    nao = 4
    C = np.linalg.qr(rng.normal(size=(nao, nao)))[0]
    D = mo_density(1, 2, 4, np.array([[1.6, 0.2], [0.2, 0.4]]))
    assert np.isclose(np.trace(D), 4.0)          # neutral: 4 electrons
    charges = np.array([2.0, 2.0])
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])
    dip = rng.normal(size=(3, nao, nao))
    dip = 0.5 * (dip + dip.transpose(0, 2, 1))
    t = np.array([0.7, -0.3, 1.1])
    mu0 = dipole_from_density(C, D, dip, charges, coords)
    dip_t = dip + t[:, None, None] * np.eye(nao)
    mu1 = dipole_from_density(C, D, dip_t, charges, coords + t)
    assert np.allclose(mu0, mu1, atol=1e-12)


# --------------------------------------------------------------- jax-gated
# The chain tests import jax and pyscfad and skip without them, exactly
# like tests/driver/test_nevpt2.py, so the numpy assembly above keeps
# running on a bare interpreter.

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

pyscfad = pytest.importorskip('pyscfad')


def _h2o_chain(shift=None):
    """H2O/STO-3G CAS(2,2), optionally translated.

    Deliberately polar: on H2 every dipole and every theta-derivative
    vanishes by symmetry, so a comparison there would be zero against
    zero and prove nothing (the lesson this project already learned on
    the HVP test).
    """
    from pyscfad import gto, scf

    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.orblib import orbital_pairs

    xyz = np.array([[0.0, 0.0, 0.0],
                    [0.7557, 0.0, 0.5880],
                    [-0.7557, 0.0, 0.5880]])
    if shift is not None:
        xyz = xyz + np.asarray(shift)
    mol = gto.Mole()
    mol.atom = [[sym, tuple(x)] for sym, x in
                zip(('O', 'H', 'H'), xyz)]
    mol.unit = 'Angstrom'
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    ncore = mol.nelectron // 2 - 1
    nact, nele = 2, 2
    nmo = mol.nao_nr()
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    ref = dict(
        C=jnp.asarray(np.asarray(mf.mo_coeff)),
        h=jnp.asarray(np.asarray(mol.intor('int1e_kin')
                                 + mol.intor('int1e_nuc'))),
        e=jnp.asarray(np.asarray(mol.intor('int2e', aosym='s1'))),
        dip=jnp.asarray(np.asarray(mol.intor('int1e_r'))),
        chg=jnp.asarray(np.asarray(mol.atom_charges(), dtype=float)),
        crd=jnp.asarray(np.asarray(mol.atom_coords())))
    return ref, ncore, nact, nele, nmo, pairs, cas_dets_for2(ncore, nact,
                                                             nele)


def _mu_fn(ref, ncore, nact, nele, nmo, pairs, dets):
    from diffmrc.driver.properties import make_casscf_dipole_theta_fn
    return make_casscf_dipole_theta_fn(
        ref['C'], ref['h'], ref['e'], ref['dip'], ref['chg'], ref['crd'],
        ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs, dets=dets)


def test_the_chained_dipole_is_invariant_under_a_translation():
    """Shift the whole molecule and the dipole must not move.

    The synthetic test above checks the bookkeeping; this one runs the
    same statement through the real chain -- SCF, CI, canonicalization,
    RDM -- with real integrals, where the overlap matrix makes the
    electronic shift non-trivial.
    """
    shift = np.array([0.31, -0.17, 0.42])
    out = []
    for s in (shift, -shift):
        ref, ncore, nact, nele, nmo, pairs, dets = _h2o_chain(s)
        mu = _mu_fn(ref, ncore, nact, nele, nmo, pairs, dets)
        out.append(np.asarray(mu(jnp.zeros(len(pairs))), dtype=float))
    assert np.allclose(out[0], out[1], atol=1e-9)


def test_the_theta_derivative_of_the_dipole_matches_its_own_difference():
    """AD of mu against a central difference of mu(theta).

    Both evaluate the same function, so this pins the differentiation of
    the new contraction (the physics is anchored against PySCF on the
    server, not here).  Two perturbation directions keep the test cheap
    while still exercising the canonicalization and the CI inside the
    differentiated path.
    """
    ref, ncore, nact, nele, nmo, pairs, dets = _h2o_chain()
    mu = _mu_fn(ref, ncore, nact, nele, nmo, pairs, dets)
    th0 = jnp.zeros(len(pairs))
    jac = np.asarray(jax.jacfwd(mu)(th0))
    assert jac.shape == (3, len(pairs))
    assert np.abs(jac).max() > 1e-6          # the derivative is not trivial
    d = 1e-5
    for k in (0, len(pairs) // 2):
        e = np.zeros(len(pairs))
        e[k] = 1.0
        fd = (np.asarray(mu(th0 + d * jnp.asarray(e)))
              - np.asarray(mu(th0 - d * jnp.asarray(e)))) / (2 * d)
        assert np.allclose(jac[:, k], fd, atol=1e-9, rtol=1e-5)
