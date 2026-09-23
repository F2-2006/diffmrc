"""The differentiable NEVPT2 chain, assembled in the driver layer.

The composition mirrors the validated per-class chain of the original
t-scripts (t121/t123/t132), which is the only path in this project whose
NEVPT2 numbers were checked against an independent implementation:

    CASCI in the theta-rotated basis  (provides the CI vector)
      -> canonicalize_jax             (active natural orbitals + sub-Fock
                                       diagonalization; the NEVPT2
                                       denominators are defined only for
                                       the canonical set)
      -> full-space MO integrals in the canonical basis
      -> CASCI again                  (energy and CI in that basis)
      -> nevpt2_e2_full2 with the per-spin link tables

Two facts about ``nevpt2_e2_full2`` shape this module and were verified
by reading the frozen source rather than assumed:

  * it slices ``eri_mo[nocc:, ...]`` for the external classes and hands
    the full ``eri_mo`` to ``e2_core`` for the core-class mixed blocks,
    so the integrals handed in must span the *whole* MO space -- a
    truncated ``[:ncore+nact]`` array silently drops the virtual space;
  * with ``links_a``/``links_b`` left at ``None`` the ``ncore > 0``
    branch skips the entire core-class contribution (Si, Sij, Sir,
    Sijr, Srsi, Sijrs). For any system with a frozen core -- i.e. all of
    the f-block ones -- the links are mandatory, not an optimisation.

``e2_scale`` exists for one reason: with it set to zero the joint
function reduces to the CASSCF energy function, whose nuclear gradient
is already validated against invariance to 1e-14. That is the only
"known answer" this chain has.

Two entry points share the same core (``_components_from``):

  * ``make_nevpt2_joint_fn``  -- ``(mol, theta)``, the SCF inside, for
    everything that differentiates with respect to the geometry;
  * ``make_nevpt2_theta_fn``  -- ``theta`` only, the reference computed
    once at construction, for the theta optimizer, which wraps its
    objective in ``jax.jit`` (a context this pyscfad build cannot trace
    its SCF in -- see the factory docstring).
"""
import numpy as np

__all__ = ['make_nevpt2_joint_fn', 'make_nevpt2_theta_fn',
           'nevpt2_nuclear_gradient', 'nevpt2_hvp',
           'nevpt2_relaxed_hessian']


def _spin_links(nact, nele):
    """Per-spin CI link tables (static numpy; depend only on the space)."""
    from pyscf.fci import cistring

    na = nele // 2
    nb = nele - na
    la = cistring.gen_linkstr_index(range(nact), na)
    lb = cistring.gen_linkstr_index(range(nact), nb)
    fmt = lambda l: [[tuple(x) for x in l[s]] for s in range(len(l))]
    return fmt(la), fmt(lb)


def _components_from(C0_, h_, e_, enuc, th_, *, ncore, nact, nele, nmo,
                     pairs, dets, links_a, links_b, ci_solver='plain',
                     want_e2=True):
    """``(E_casscf, E2)`` from reference quantities already in hand.

    Contains no SCF call and no Mole access -- theta and the precomputed
    arrays only. Both the joint function (which computes the reference
    from the Mole first) and the frozen-geometry theta function (which
    computes it once at construction) funnel through here, so the two
    cannot drift apart.

    ``ci_solver`` selects the CI eigensolver, following the instruction in
    ``core.casci.cs_casci_blocked``'s own docstring: the plain solver for
    the first derivative and below (more thoroughly validated), the
    S_z-blocked solver wherever a second derivative is required (the
    plain one's ``eigh`` vjp produces NaN at the cross-S_z degeneracy --
    measured here as 0/0 for H2O/6-31G CAS(2,2), whose CI matrix is
    three-fold degenerate in the unblocked layout).
    """
    from jax import numpy as jnp

    from ..core.casci import cs_casci_blocked
    from ..core.nevpt2.impl import (_casci_ci, canonicalize_jax,
                                    nevpt2_e2_full2)
    from ..core.rotation import block_rotation

    nm = ncore + nact

    def ci_solve(h1, eri):
        if ci_solver == 'blocked':
            return cs_casci_blocked(h1[:nm, :nm], eri[:nm, :nm, :nm, :nm],
                                    ncore=ncore, nact=nact, nele_act=nele)
        if ci_solver == 'plain':
            return _casci_ci(h1, eri, ncore, nact, nele)
        raise ValueError('unknown ci_solver: {!r}'.format(ci_solver))

    C2 = jnp.asarray(C0_) @ block_rotation(th_, nmo, pairs)

    def mo_ints(Cx):
        h1 = jnp.einsum('ai,ab,bj->ij', Cx, h_, Cx)
        eri = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cx, Cx, Cx, Cx, e_)
        return h1, eri

    h1_, eri_ = mo_ints(C2)
    _, ci = ci_solve(h1_, eri_)
    C_can, _, eps = canonicalize_jax(C2, h_, e_, ci, ncore=ncore,
                                     nact=nact, nele_act=nele)
    h1c, eric = mo_ints(C_can)
    E0c, ci2 = ci_solve(h1c, eric)
    if not want_e2:
        # The caller multiplies e2 by zero; skipping the expensive
        # contraction is exact (0 * x = 0 in IEEE, and the VJP of a term
        # multiplied by zero is zero), and on actinide-sized virtual
        # spaces it is the difference between minutes and hours.
        return E0c + enuc, jnp.zeros(())
    e2, _ = nevpt2_e2_full2(h1c, eric, jnp.asarray(ci2), dets, E0c,
                            mo_energy=eps, ncore=ncore, nact=nact,
                            nocc=ncore + nact,
                            links_a=links_a, links_b=links_b)
    return E0c + enuc, e2


def make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, scf_builder,
                         e2_scale=1.0, ci_solver='plain'):
    """Build ``E(mol, theta) = E_casscf + e2_scale * E2_nevpt2``.

    Parameters
    ----------
    ncore, nact, nele : int
        Active-space definition.
    nmo : int
        Number of basis functions.
    pairs : list of (int, int)
        Orbital rotation pairs; must include core<->virtual.
    scf_builder : callable
        ``mol -> mean field``. The builder declares the reference
        *flavour* only -- it constructs and configures the object
        (RHF/ROHF, X2C, ECP, damping) and the joint function runs
        ``kernel()`` on it, following ``make_joint_energy_fn``. For the
        f block the flavour is the caller's decision: a lanthanide or
        actinide run needs ROHF (open shell) with or without X2C, and an
        ECP route needs no X2C at all -- ``lambda m: pscf.ROHF(m).x2c1e()``
        versus ``lambda m: pscf.ROHF(m)``. An earlier version hard-coded
        ``pscf.RHF``, which silently excluded every open-shell system
        this project actually studies.
    e2_scale : float
        Multiplier on the correlation energy; 0 gives the CASSCF
        function exactly, which is the validation anchor.

    Returns
    -------
    callable
        ``(mol, theta) -> jax scalar``, differentiable in both.
    """
    from ..core.nevpt2.impl import cas_dets_for2

    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              dets=dets, links_a=links_a, links_b=links_b,
              ci_solver=ci_solver)

    def components(m_, th_, want_e2=True):
        """CASSCF energy and NEVPT2 correction, both differentiable."""
        mf_ = scf_builder(m_)
        mf_.kernel()
        # get_hcore carries the ECP and the X2C rotation; the raw
        # int1e_kin + int1e_nuc sum would silently drop both.
        return _components_from(mf_.mo_coeff, mf_.get_hcore(),
                                m_.intor('int2e', aosym='s1'),
                                m_.energy_nuc(), th_, want_e2=want_e2, **kw)

    def E_joint(m_, th_):
        E_cas, e2 = components(m_, th_, want_e2=(e2_scale != 0.0))
        return E_cas + e2_scale * e2

    E_joint.components = components      # the anchor test uses both
    return E_joint


def make_nevpt2_theta_fn(mol, ncore, nact, nele, nmo, pairs, scf_builder,
                         e2_scale=1.0, ci_solver='plain'):
    """Theta-only joint energy at a fixed geometry (safe under jax.jit).

    The reference SCF and the AO integrals are computed once here; the
    returned function depends on theta alone. The theta optimizer needs
    this form: ``core.optimizer`` wraps its objective in ``jax.jit``, and
    this pyscfad build cannot trace the SCF under a jit context -- it
    raises ``TracerArrayConversionError`` inside ``get_occ`` (probe
    ``code/t190_probe_theta_grad.py``: ``jax.grad`` and ``jax.hessian``
    trace the SCF fine, ``jax.jit`` does not, both with the default and a
    1e-12 convergence tolerance). The same pattern is used by
    ``driver.kernel.energy_function``, which takes ``C0`` as an argument
    for exactly this reason.

    At a fixed geometry the frozen-reference form is the same function as
    the joint one: the SCF would run once either way, so no physics is
    dropped -- only the (irrelevant, here) geometry dependence.
    """
    from ..core.nevpt2.impl import cas_dets_for2

    mf_ = scf_builder(mol)
    mf_.kernel()
    C0_ = mf_.mo_coeff
    h_ = mf_.get_hcore()
    e_ = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              dets=dets, links_a=links_a, links_b=links_b,
              ci_solver=ci_solver)

    def E_theta(th_):
        E_cas, e2 = _components_from(C0_, h_, e_, enuc, th_,
                                     want_e2=(e2_scale != 0.0), **kw)
        return E_cas + e2_scale * e2

    E_theta.components = lambda th_: _components_from(  # noqa: E731
        C0_, h_, e_, enuc, th_, **kw)
    return E_theta


def nevpt2_nuclear_gradient(mol, E_of):
    """dE/dR for an energy function that rebuilds its reference inside.

    The two requirements are the ones ``core.grad`` documents: the
    reference orbitals must be a function of the geometry (which
    ``E_of`` guarantees by rebuilding them), and any rotation
    parameters must be frozen (which the caller guarantees by closing
    over a fixed theta). The differentiation itself follows the
    validated route of ``core.grad.nuclear_gradient``: the pyscfad Mole
    is a pytree, and only its ``coords`` field carries the nuclear
    coordinates.

    Parameters
    ----------
    mol : pyscfad Mole
        Target geometry.
    E_of : callable(Mole) -> scalar
        The frozen-theta energy; typically ``lambda m: fn(m, theta)``.

    Returns
    -------
    numpy.ndarray, shape (natm, 3), in Eh/bohr
    """
    import jax

    g = jax.grad(E_of)(mol)
    return np.asarray(g.coords)


def nevpt2_hvp(mol, E_of, v):
    """Hessian-vector product, forward-over-reverse -- one ``jax.jvp``.

    Cheaper than assembling the matrix and exact to machine precision,
    which is what makes it the primary route (the full matrix is built
    only when a caller needs its spectrum). The direction is applied as
    a pytree tangent: the pyscfad Mole pytree's coordinate leaf is the
    (natm, 3) array, the same leaf ``core.hessian`` reads Hessians back
    out of (``H.coords.coords``).

    ``E_of`` must be built with ``ci_solver='blocked'``: a second
    derivative through the plain CI eigensolver returns NaN wherever the
    unblocked CI matrix carries a cross-S_z degeneracy (measured: NaN
    already at the CASCI stage for H2O/6-31G CAS(2,2)). This is the
    instruction in ``cs_casci_blocked``'s own docstring.

    Parameters
    ----------
    mol : pyscfad Mole
    E_of : callable(Mole) -> scalar
        The frozen-theta energy.
    v : array_like, shape (natm, 3) or (3 * natm,)
        The direction. For a diatomic, every perpendicular direction is
        an exact zero mode (rigid rotation/translation), so a probe
        meant to test anything has to include the bond stretch.

    Returns
    -------
    numpy.ndarray, shape (natm, 3)  --  H @ v
    """
    import jax
    from jax import numpy as jnp
    from jax import tree_util

    v_arr = jnp.asarray(np.asarray(v, dtype=float).reshape(-1, 3))
    coords_ref = np.asarray(mol.atom_coords())
    leaves, treedef = tree_util.tree_flatten(mol)
    idx = [i for i, l in enumerate(leaves)
           if hasattr(l, 'shape') and tuple(np.shape(l)) == coords_ref.shape
           and np.allclose(np.asarray(l), coords_ref)]
    if len(idx) != 1:
        raise ValueError('cannot locate the coordinate leaf of the Mole pytree')
    tan_leaves = [jnp.zeros_like(l) for l in leaves]
    tan_leaves[idx[0]] = v_arr
    tangent = tree_util.tree_unflatten(treedef, tan_leaves)
    _, hv = jax.jvp(jax.grad(E_of), (mol,), (tangent,))
    return np.asarray(hv.coords)


def nevpt2_relaxed_hessian(mol, theta, E_casscf, E_nevpt2, rcond=None):
    """The NEVPT2 Hessian along the CASSCF-relaxed orbital path.

    The correlation energy is a correction evaluated on the CASSCF
    reference, so the orbital response entering its second derivative is
    the *CASSCF* one -- dtheta*/dR comes from the CASSCF Lagrangian --
    while the frozen curvature and the mixed term come from the NEVPT2
    energy itself. In Park's Lagrangian this is exactly the split between
    the CASSCF Z-vector (the response operator) and the NEVPT2 source
    terms (``文献细读/细读_NEVPT2梯度_BAGEL2019.md``, Eqs. 23-28).

    Optimizing theta on the SC-NEVPT2 surface is not an alternative: that
    surface is not smooth (the SC energy lacks invariance under inactive
    rotations, and its denominators cross zero). Measured here: the
    theta optimizer walks into a singularity from the zeros start
    (E = -3.2e7, |g| = 3.6e14) *and* from the CASSCF minimum
    (E = -1.1e6, |g| = 3.4e15).

    Returns the same record shape as ``core.hessian.implicit_hessian``,
    with ``H_total = H_RR(nevpt2) - M_nevpt2^T H_theta_theta(casscf)^-1
    M_nevpt2``.
    """
    import jax

    from ..core.hessian import (DEFAULT_RCOND, ImplicitHessianResult,
                                _frozen_hessian, _mixed_derivative)
    from ..core.stationary import curvature_report

    rcond = DEFAULT_RCOND if rcond is None else float(rcond)
    theta = jax.numpy.asarray(np.asarray(theta, dtype=float))

    H_tt = np.asarray(jax.hessian(lambda t: E_casscf(mol, t))(theta))
    H_tt = 0.5 * (H_tt + H_tt.T)
    M = _mixed_derivative(E_nevpt2, mol, theta)
    n_r = M.shape[1]
    H_RR = _frozen_hessian(E_nevpt2, mol, theta, n_r)
    H_RR = 0.5 * (H_RR + H_RR.T)

    H_tt_inv = np.linalg.pinv(H_tt, rcond=rcond)
    D = -H_tt_inv @ M
    H_resp = M.T @ D
    H_total = H_RR + H_resp
    H_total = 0.5 * (H_total + H_total.T)

    return ImplicitHessianResult(
        H_tt=H_tt, H_tt_inverse=H_tt_inv,
        curvature=curvature_report(H_tt),
        mixed=M, H_RR=H_RR, H_response=H_resp, H_total=H_total,
        dtheta_dR_max=float(np.abs(D).max()) if D.size else 0.0,
        rcond=float(rcond),
    )
