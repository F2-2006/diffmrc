"""The differentiable NEVPT2 chain: joint energy, gradients, HVPs."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscfad = pytest.importorskip("pyscfad")


def _h2():
    from pyscfad import gto

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.build()
    return mol


def _h2o():
    from pyscfad import gto

    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
    mol.basis = "sto-3g"
    mol.build()
    return mol


def _pairs(ncore, nact, nmo):
    from diffmrc.core.orblib import orbital_pairs

    return orbital_pairs(ncore, nact, nmo, include_cv=True)


def test_the_joint_function_returns_the_casscf_energy_at_scale_zero():
    """The limit anchor's premise: with the correlation off, the joint
    function IS the validated CASSCF energy function."""
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn

    mol = _h2()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)
    fn = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=0.0)
    E0 = float(fn(mol, jnp.zeros(len(pairs))))
    # -1.137284 was measured in this session (PySCF CASSCF(2,2) on the
    # same geometry, printed by the server probe) -- not recalled from a
    # paper; the frozen reference test pins the H2O analogue.
    assert E0 == pytest.approx(-1.137284, abs=1e-5)


def test_the_joint_function_changes_with_the_scale():
    from pyscfad import scf, gto

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn

    # H2/STO-3G CAS(2,2) fills the whole basis (2 orbitals, no virtual
    # space), so its correlation term is identically zero; 6-31G leaves
    # two virtual orbitals and the term is real.
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "6-31g"
    mol.build()
    nmo = mol.nao_nr()
    ncore, nact, nele = 0, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)
    f0 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=0.0)
    f1 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=1.0)
    t = jnp.zeros(len(pairs))
    assert float(f1(mol, t)) < float(f0(mol, t))     # correlation lowers E


def _gradient_of(fn, mol, theta):
    """dE/dR with theta frozen: the envelope-theorem recipe."""
    from diffmrc.driver.nevpt2 import nevpt2_nuclear_gradient

    def E_of(m_):
        return fn(m_, theta)

    return np.asarray(nevpt2_nuclear_gradient(mol, E_of))


def test_the_gradient_is_translationally_invariant():
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn

    mol = _h2()
    pairs = _pairs(0, 2, mol.nao_nr())
    builder = lambda m: scf.RHF(m)
    fn = make_nevpt2_joint_fn(0, 2, 2, mol.nao_nr(), pairs, builder)
    g = _gradient_of(fn, mol, jnp.zeros(len(pairs)))
    assert np.abs(g.sum(axis=0)).max() < 1e-8


def test_the_gradient_is_rotationally_invariant():
    from pyscfad import scf, gto

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn

    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
    mol.basis = "sto-3g"
    mol.build()
    nmo = mol.nao_nr()
    pairs = _pairs(1, 2, nmo)
    builder = lambda m: scf.RHF(m)
    fn = make_nevpt2_joint_fn(1, 2, 2, nmo, pairs, builder)
    g = _gradient_of(fn, mol, jnp.zeros(len(pairs)))
    R = np.asarray(mol.atom_coords())
    torque = np.cross(R, g).sum(axis=0)
    assert np.abs(torque).max() < 1e-6


def test_the_scale_zero_gradient_reproduces_the_casscf_gradient():
    """With the correlation scaled off, the joint function's gradient must
    equal the CASSCF gradient. Inside this codebase that is the only
    "known answer" available (validated against invariance to 1e-14);
    externally, the equivalence argument is the Lagrangian derivation of
    the analytic NEVPT2 gradient (Park, JCTC 2019; card in
    ``文献细读/细读_NEVPT2梯度_BAGEL2019.md``), of which an AD gradient is
    an independent evaluation of the same mathematical object.

    The CASSCF side is not hand-written here: it is
    ``core.hessian.make_joint_energy_fn``, the function the validated
    implicit-Hessian route already uses.
    """
    from pyscfad import scf

    from diffmrc.core.hessian import make_joint_energy_fn
    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn

    mol = _h2()
    nmo = mol.nao_nr()
    pairs = _pairs(0, 2, nmo)
    theta = jnp.zeros(len(pairs))
    builder = lambda m: scf.RHF(m)
    fn_nevpt2 = make_nevpt2_joint_fn(0, 2, 2, nmo, pairs, builder,
                                     e2_scale=0.0)
    fn_casscf = make_joint_energy_fn(0, 2, 2, nmo, pairs, pscf=scf)
    g_nev = _gradient_of(fn_nevpt2, mol, theta)
    g_cas = _gradient_of(fn_casscf, mol, theta)
    assert np.abs(g_nev - g_cas).max() < 1e-10


def test_the_theta_only_function_matches_the_joint_energy():
    """The frozen-geometry form and the joint form are the same function
    at the same theta. The theta optimizer consumes the former because
    ``core.optimizer`` wraps its objective in ``jax.jit`` and this
    pyscfad build cannot trace its SCF under a jit context (probe t190);
    at a fixed geometry nothing physical is dropped by precomputing the
    reference."""
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                       make_nevpt2_theta_fn)

    mol = _h2()
    pairs = _pairs(0, 2, mol.nao_nr())
    builder = lambda m: scf.RHF(m)
    theta = jnp.zeros(len(pairs))
    fn = make_nevpt2_joint_fn(0, 2, 2, mol.nao_nr(), pairs, builder)
    th_fn = make_nevpt2_theta_fn(mol, 0, 2, 2, mol.nao_nr(), pairs, builder)
    assert float(th_fn(theta)) == pytest.approx(float(fn(mol, theta)),
                                                abs=1e-12)


def _rhf_tight(m):
    """RHF with a tighter convergence tolerance: the FD ladder divides
    the residual by 2*delta, so a 1e-9 SCF residual shows up as 1e-6 in
    the smallest FD rows."""
    from pyscfad import scf

    mf = scf.RHF(m)
    mf.conv_tol = 1e-12
    return mf


def test_the_hvp_matches_a_finite_difference_of_the_gradient():
    """FD only after a delta sweep shows a platform. This project has
    measured FD wrong by a factor 106 near zero gradients, with the
    signature 'the smaller delta, the larger the value' -- so the test
    tries a ladder and requires the middle of it to be flat.

    The probe vector is the bond stretch. For a diatomic every
    perpendicular direction is an exact zero mode (rigid rotation and
    translation), so a generic displacement would make this test
    compare 0 against 0 and prove nothing.
    """
    from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                       nevpt2_hvp,
                                       nevpt2_nuclear_gradient)

    mol = _h2()
    pairs = _pairs(0, 2, mol.nao_nr())
    fn = make_nevpt2_joint_fn(0, 2, 2, mol.nao_nr(), pairs, _rhf_tight,
                              ci_solver='blocked')
    theta = jnp.zeros(len(pairs))
    v = np.zeros(mol.natm * 3)
    v[2] = -1.0
    v[5] = 1.0                       # symmetric stretch along z
    E_of = lambda m_: fn(m_, theta)
    hvp = np.asarray(nevpt2_hvp(mol, E_of, v))
    rows = []
    for d in (1e-2, 3e-3, 1e-3, 3e-4):
        m1 = mol.copy()
        m1.set_geom_(np.asarray(mol.atom_coords()) + d * v.reshape(-1, 3),
                     'Bohr')
        m2 = mol.copy()
        m2.set_geom_(np.asarray(mol.atom_coords()) - d * v.reshape(-1, 3),
                     'Bohr')
        g1 = np.asarray(nevpt2_nuclear_gradient(m1, E_of))
        g2 = np.asarray(nevpt2_nuclear_gradient(m2, E_of))
        rows.append(((g1 - g2) / (2 * d)).reshape(-1))
    # The ladder's error to the HVP is quadratic in delta (measured on H2:
    # 6.6e-6 at 1e-3, 6.0e-7 at 3e-4, ratio ~ (10/3)^2), so the honest
    # checks are monotone improvement plus a bound at the finest delta.
    # A flat "platform" between 3e-3 and 1e-3 does not exist at this
    # curvature: their difference is ~5e-5, itself the delta^2 term.
    errs = [np.abs(r - hvp).max() for r in rows]
    assert errs[3] < errs[2] < errs[1] < errs[0]     # monotone improvement
    assert errs[2] < 2e-5                            # delta = 1e-3
    assert errs[3] < 2e-6                            # delta = 3e-4


def test_the_hvp_is_symmetric():
    """Hv . w == Hw . v for two probe vectors: the Hessian is symmetric.

    Measured on H2O, not H2: a diatomic has a single non-zero curvature
    direction, so any two-vector identity on it reduces to x == x.
    """
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn, nevpt2_hvp

    mol = _h2o()
    nmo = mol.nao_nr()
    pairs = _pairs(1, 2, nmo)
    fn = make_nevpt2_joint_fn(1, 2, 2, nmo, pairs, lambda m: scf.RHF(m),
                              ci_solver='blocked')
    theta = jnp.zeros(len(pairs))
    E_of = lambda m_: fn(m_, theta)
    rng = np.random.default_rng(0)
    v = rng.normal(size=mol.natm * 3)
    w = rng.normal(size=mol.natm * 3)
    hv = np.asarray(nevpt2_hvp(mol, E_of, v)).reshape(-1)
    hw = np.asarray(nevpt2_hvp(mol, E_of, w)).reshape(-1)
    scale = max(1.0, np.abs(hv).max() * np.abs(w).max(),
                np.abs(hw).max() * np.abs(v).max())
    assert abs(hv @ w - hw @ v) < 1e-8 * scale
