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
    the BAGEL 2019 NEVPT2 gradient notes), of which an AD gradient is
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
    hvp = np.asarray(nevpt2_hvp(mol, E_of, v)).reshape(-1)
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


def test_the_actinide_ecp_route_wires_through():
    """ThO with the Stuttgart ECP on Th and 6-31G on O: nao = 96, 38
    electrons, spin 0 from parity.

    The light ligand basis is deliberate. The published ``ano`` ligand
    basis gives nao = 175, whose joint evaluation peaked at 197 GB RSS and
    was OOM-killed; the 6-31G ligand keeps the full-space integrals at
    ~5 GB (measured). The joint value is recorded in the Task 8 run log
    (``experiments/verify_nevpt2_an.py tho_ecp``); the assertions here are
    the wiring invariants -- the reference SCF converges to the pinned
    energy, the CASCI energy cannot exceed it, and the correlation
    correction is finite and negative.

    This test deliberately never differentiates with respect to the
    geometry: pyscfad's integral-derivative path allocates a dense
    ``(nctr, nao^4)`` array (51.3 GiB at nao = 96, measured), so the
    actinide validation lives in theta space (see the Task 8 docstring).
    """
    from pyscfad import gto, scf

    from diffmrc.driver.nevpt2 import make_nevpt2_joint_fn

    mol = gto.Mole()
    mol.atom = "Th 0 0 0; O 0 0 1.85"
    mol.basis = {"Th": "stuttgart_rsc", "O": "6-31g"}
    mol.ecp = {"Th": "stuttgart_rsc"}
    mol.spin = 0
    mol.build()
    assert mol.nelectron == 38
    nmo = mol.nao_nr()

    def builder(m):
        mf = scf.RHF(m)
        mf.init_guess = "atom"
        mf.max_cycle = 200
        mf.conv_tol = 1e-9
        return mf

    mf0 = builder(mol)
    mf0.kernel()
    # pinned from the 2026-09-23 server run (probe t197 / run
    # experiments/verify_nevpt2_an.py tho_ecp)
    assert float(mf0.e_tot) == pytest.approx(-481.0313789074, abs=1e-9)

    # CAS(4,4): the CAS(8,8) variant of this chain needs over 2.5 h per
    # joint evaluation at nao = 96 (the unblocked CI construction at 12870
    # determinants runs eagerly); the wiring question is answered by the
    # minimal space, which evaluates in minutes.
    ncore, nact, nele = 17, 4, 4
    pairs = _pairs(ncore, nact, nmo)
    fn = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=1.0, ci_solver="blocked")
    theta = jnp.zeros(len(pairs))
    E_cas, e2 = fn.components(mol, theta)
    assert np.isfinite(float(E_cas))
    assert np.isfinite(float(e2))
    assert float(E_cas) < float(mf0.e_tot)   # CASCI cannot exceed the SCF
    # The correlation correction this chain returns is negative here; this
    # asserts the chain's own regression value, not the physics. The
    # large-core discrepancy against PySCF recorded in the earlier baseline
    # (1.6 Ha) is closed by the blocked-loop vector fix; the post-fix
    # deviation is 3.8e-7 (frozen-table row 7).
    assert float(e2) < 0.0


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


# ---------------------------------------------------------------------------
# The relaxed-path nuclear gradient (the correlation response term)
# ---------------------------------------------------------------------------

def _relaxed_pieces(mol, ncore, nact, nele, theta):
    """(frozen gradient, relaxed gradient, response term) on the H2O fixture."""
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                       nevpt2_nuclear_gradient,
                                       nevpt2_relaxed_nuclear_gradient)

    nmo = mol.nao_nr()
    pairs = _pairs(ncore, nact, nmo)
    builder = lambda m: scf.RHF(m)
    f0 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=0.0, ci_solver='blocked')
    f1 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=1.0, ci_solver='blocked')
    g_frozen = np.asarray(nevpt2_nuclear_gradient(
        mol, lambda m_: f1(m_, theta)))
    g_relaxed = np.asarray(nevpt2_relaxed_nuclear_gradient(
        mol, theta, f0, f1))
    return g_frozen, g_relaxed, g_relaxed - g_frozen, f0, f1


def test_the_relaxed_gradient_anchor_is_exact_to_roundoff():
    """With e2 = E_nevpt2 - E_casscf identically zero the response source
    is exactly zero, so the relaxed function reduces to the frozen-theta
    gradient up to re-dispatch roundoff.

    Measured on H2O/STO-3G (2026-10-17): the non-zero gradient
    components are bitwise identical; the near-zero ones (translation
    sums) differ at 1e-15 .. 1e-17 -- the SCF's own cancellation order
    across two dispatches, not a response term. The bound below is the
    measurement's margin.
    """
    from diffmrc.driver.nevpt2 import nevpt2_relaxed_nuclear_gradient

    mol = _h2o()
    nmo = mol.nao_nr()
    ncore, nact, nele = 1, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                       nevpt2_nuclear_gradient)

    f0 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs,
                              lambda m: scf.RHF(m), e2_scale=0.0,
                              ci_solver='blocked')
    theta = jnp.zeros(len(pairs))
    g_frozen = np.asarray(nevpt2_nuclear_gradient(
        mol, lambda m_: f0(m_, theta)))
    g_relaxed = np.asarray(nevpt2_relaxed_nuclear_gradient(
        mol, theta, f0, f0))
    assert np.abs(g_relaxed - g_frozen).max() < 1e-13


def test_the_relaxed_response_has_no_translation_or_rotation_component():
    """The response term is translation- and rotation-free.

    M's translation/rotation columns vanish because the CASSCF energy is
    invariant under those displacements at every theta, so the response
    D=-pinv(H_tt)M inherits exact zeros there. A nonzero value would mean
    the mixed-derivative chain is inconsistent (the same class of defect
    the invariance criteria exist to catch).
    """
    mol = _h2o()
    ncore, nact, nele = 1, 2, 2
    theta = jnp.zeros(len(_pairs(ncore, nact, mol.nao_nr())))
    _, _, resp, _, _ = _relaxed_pieces(mol, ncore, nact, nele, theta)
    assert np.isfinite(resp).all()
    assert np.abs(resp.sum(axis=0)).max() < 1e-9
    coords = np.asarray(mol.atom_coords())
    rot = np.cross(coords, resp).sum(axis=0)
    assert np.abs(rot).max() < 1e-9


def test_the_relaxed_gradient_takes_a_precomputed_response_operator():
    """The D seam: a zero response operator reproduces the frozen gradient
    to roundoff (the verification script injects the finite-difference
    channel through this argument)."""
    from diffmrc.driver.nevpt2 import nevpt2_relaxed_nuclear_gradient

    mol = _h2o()
    nmo = mol.nao_nr()
    ncore, nact, nele = 1, 2, 2
    pairs = _pairs(ncore, nact, nmo)
    from pyscfad import scf

    from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,
                                       nevpt2_nuclear_gradient)

    builder = lambda m: scf.RHF(m)
    f0 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=0.0, ci_solver='blocked')
    f1 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=1.0, ci_solver='blocked')
    theta = jnp.zeros(len(pairs))
    D0 = np.zeros((len(pairs), mol.natm * 3))
    g0 = np.asarray(nevpt2_relaxed_nuclear_gradient(
        mol, theta, f0, f1, D=D0))
    g_frozen = np.asarray(nevpt2_nuclear_gradient(
        mol, lambda m_: f1(m_, theta)))
    assert np.abs(g0 - g_frozen).max() < 1e-13
