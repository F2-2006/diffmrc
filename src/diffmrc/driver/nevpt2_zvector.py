"""Z-vector response for the NEVPT2 nuclear gradient, exact Si channels.

Motivation
----------
``nevpt2_nuclear_gradient`` differentiates the frozen-theta energy: the
SCF reference and the CI vector are rebuilt at every displaced geometry,
but the CASSCF orbital-rotation parameters theta stay at their stationary
values.  Along the physical path theta follows the CASSCF stationarity
condition dE_casscf/dtheta = 0, and the correlation energy needs its own
response term because NEVPT2 is not variational in the orbitals:

    g_relaxed = g_frozen - M^T H_tt^{-1} s,
    H_tt = d2E_casscf/dtheta2,   M = d2E_casscf/dtheta dR,
    s    = d e2/dtheta,

with e2 = E_nevpt2 - E_casscf.  ``z = H_tt^{-1} s`` is the orbital
Z-vector (the adjoint variable of the stationarity constraint); the
correction ``-M^T z`` needs one linear solve per state, not one per
Cartesian direction.

The CI coordinates need no separate multiplier in this reduced space.
Both energy functions solve the CI eigenvalue problem internally, so
their theta and R derivatives are taken along the CI-relaxed manifold;
the CI block of the full Hessian is replaced by its Schur complement,
which is exactly the H_tt above.  Working in the full (theta, c) space
would add a CI multiplier, but that multiplier is annihilated by the
variational CI condition for E_casscf and only reshuffles the same
reduced Hessian.

Si-class boundary
-----------------
The release core evaluates the Si class as

    e2_si = -sum_i n_i / (d_i + h_i / n_i)  for |n_i| > 1e-12,

i.e. -sum_i n_i^2 / (d_i n_i + h_i) with a *value* mask.  With a frozen
core, symmetry can drive individual channel norms to 1e-13.  Dropping
such a channel changes the energy by ~1e-13, but its derivative does not
vanish: at n = 1e-13 and h = 2e-12 the exact form has

    df/dn = -(2 n (d n + h) - n^2 d) / (d n + h)^2 ~ -0.5,

so the masked chain loses an O(0.1-1) CI derivative per channel.  The
mask is a discontinuity in the energy, not a removable singularity: the
function itself is analytic wherever d_i n_i + h_i != 0, and its
derivative at the base point is well defined.  This module therefore
evaluates the same algebra with the guard moved to the denominator,

    e2_si = -sum_i n_i^2 / (d_i n_i + h_i),

and keeps every channel whose denominator is not numerically zero.

A finite difference of this form needs a step below the channel's
variation scale (measured: ~1e-11 in CI units on H2O/6-31G); larger
steps measure the ray slope of the n ~ t, h ~ t regime, which is a
different number.  The analytic derivative is the one this module
returns; the verification script validates it against both the small-
step finite difference and the derivative chain
df/dc = sum_i (df/dn_i dn_i/dc + df/dh_i dh_i/dc).

Scope
-----
Additive prototype.  No existing function or signature is modified: the
exact-Si energy is assembled by correcting the release total with the
difference between the two evaluations of the core classes on identical
inputs.  Other classes keep their release masks; their channels are far
from zero on the validation system.  Production wiring would replace the
correction with a mode argument inside ``core.nevpt2.core.e2_core``.
"""
import numpy as np

__all__ = ['si_e2_exact',
           'si_energy_from_invariants',
           'si_channel_table',
           'nevpt2_e2_full2_si_exact',
           'make_nevpt2_joint_fn_si_exact',
           'orbital_zvector',
           'lagrangian_relaxed_gradient',
           'nevpt2_relaxed_gradient_zvector']


def _asarray(a):
    from jax import numpy as jnp
    return jnp.asarray(a)


def si_energy_from_invariants(norm, ener, diff, mode='exact', guard=1e-30):
    """Si energy from channel invariants.

    ``mode='exact'`` evaluates ``-sum n^2/(d n + h)`` guarding the
    denominator; ``mode='masked'`` reproduces the release value mask
    (``|n| > 1e-12``) and exists only for side-by-side diagnostics and
    tests.  The two modes agree away from the mask threshold and differ by
    the value of the dropped channels (~6e-13 on H2O/6-31G) at a
    symmetry-zero channel; their derivatives differ by O(0.1-1) because
    the mask is a discontinuity, not a removable singularity.
    """
    from jax import numpy as jnp

    n = jnp.asarray(norm)
    h = jnp.asarray(ener)
    d = jnp.asarray(diff)
    if mode == 'masked':
        idx = jnp.abs(n) > 1e-12
        n_safe = jnp.where(idx, n, 1.0)
        return -(jnp.where(idx, n, 0.0)
                 / (d + jnp.where(idx, h / n_safe, 0.0) + 1e-30)).sum()
    if mode != 'exact':
        raise ValueError('mode must be exact or masked, got {!r}'.format(mode))
    den = d * n + h
    ok = jnp.abs(den) > guard
    return -(jnp.where(ok, n * n / jnp.where(ok, den, 1.0), 0.0)).sum()


def si_e2_exact(h1e_act, h2e_act, h2e_v_si, h1e_v_si, dm1, dm2, dm3,
                f3ca, f3ac, mo_energy_core, guard=1e-30):
    """Si class with a denominator guard and no value mask.

    Same algebra as ``core.nevpt2.core.si_e2``; the only change is where
    the near-zero guard sits.  ``guard`` is an absolute floor on
    ``|d_i n_i + h_i|`` and is far below any physical denominator on the
    validation system (measured 2.2e-12); a channel whose denominator
    really crosses zero is a genuine degeneracy of the perturbation
    expansion and is dropped rather than regularized silently.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.core import make_a22_full, make_a23, make_a25

    ncas = h1e_act.shape[0]
    a22 = make_a22_full(h1e_act, h2e_act, dm2, dm3, f3ca, f3ac)
    a23 = make_a23(h1e_act, h2e_act, dm1, dm2, dm3)
    a25 = make_a25(h1e_act, h2e_act, dm1, dm2)
    delta = jnp.eye(ncas)
    dm3_h = (jnp.einsum('abef,cd->abcdef', dm2, delta) * 2
             - dm3.transpose(0, 1, 3, 2, 4, 5))
    dm2_h = jnp.einsum('ab,cd->abcd', dm1, delta) * 2 - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2 * delta - dm1.transpose(1, 0)
    ener = (jnp.einsum('qpir,pqrabc,baic->i', h2e_v_si, a22, h2e_v_si)
            + jnp.einsum('qpir,pqra,ai->i', h2e_v_si, a23, h1e_v_si) * 2.0
            + jnp.einsum('pi,pa,ai->i', h1e_v_si, a25, h1e_v_si))
    norm = (jnp.einsum('qpir,rpqbac,baic->i', h2e_v_si, dm3_h, h2e_v_si)
            + jnp.einsum('qpir,rpqa,ai->i', h2e_v_si, dm2_h, h1e_v_si) * 2.0
            + jnp.einsum('pi,pa,ai->i', h1e_v_si, dm1_h, h1e_v_si))
    return si_energy_from_invariants(norm, ener, -mo_energy_core,
                                     mode='exact', guard=guard)


def si_channel_table(h1e_act, h2e_act, h2e_v_si, h1e_v_si, dm1, dm2, dm3,
                     f3ca, f3ac, mo_energy_core):
    """Per-channel Si diagnostics: n_i, h_i, denominator, value, slopes.

    Returns a dict of numpy arrays.  ``dfdn`` and ``dfdh`` are the partial
    derivatives of -n^2/(d n + h), the quantities that turn a small norm
    into a sizeable derivative when the denominator is also small.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.core import make_a22_full, make_a23, make_a25

    ncas = h1e_act.shape[0]
    a22 = make_a22_full(h1e_act, h2e_act, dm2, dm3, f3ca, f3ac)
    a23 = make_a23(h1e_act, h2e_act, dm1, dm2, dm3)
    a25 = make_a25(h1e_act, h2e_act, dm1, dm2)
    delta = jnp.eye(ncas)
    dm3_h = (jnp.einsum('abef,cd->abcdef', dm2, delta) * 2
             - dm3.transpose(0, 1, 3, 2, 4, 5))
    dm2_h = jnp.einsum('ab,cd->abcd', dm1, delta) * 2 - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2 * delta - dm1.transpose(1, 0)
    ener = (jnp.einsum('qpir,pqrabc,baic->i', h2e_v_si, a22, h2e_v_si)
            + jnp.einsum('qpir,pqra,ai->i', h2e_v_si, a23, h1e_v_si) * 2.0
            + jnp.einsum('pi,pa,ai->i', h1e_v_si, a25, h1e_v_si))
    norm = (jnp.einsum('qpir,rpqbac,baic->i', h2e_v_si, dm3_h, h2e_v_si)
            + jnp.einsum('qpir,rpqa,ai->i', h2e_v_si, dm2_h, h1e_v_si) * 2.0
            + jnp.einsum('pi,pa,ai->i', h1e_v_si, dm1_h, h1e_v_si))
    n = np.asarray(norm, dtype=float)
    h = np.asarray(ener, dtype=float)
    d = np.asarray(mo_energy_core, dtype=float)
    den = -d * n + h
    with np.errstate(divide='ignore', invalid='ignore'):
        f = -n ** 2 / den
        dfdn = -(2.0 * n * den - n ** 2 * d) / den ** 2
        dfdh = n ** 2 / den ** 2
    return {'norm': n, 'ener': h, 'denominator': den, 'value': f,
            'dfdn': dfdn, 'dfdh': dfdh}


def _core_blocks(h1e_mo, eri_mo, c_ci, dets, ncore, nact, nocc, eps,
                 links_a, links_b):
    """Recompute the inputs of ``core.nevpt2.core.e2_core`` exactly.

    The sequence repeats ``nevpt2_e2_full2`` so that the two evaluations
    of the core classes (release and exact-Si) sit on identical arrays;
    any drift here would show up as a value difference at the base point,
    which the driver test bounds.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.f3ca import ab_index_map, fdm2fdm3_jax
    from ..core.nevpt2.impl import _nele_act_from_dets, make_dm123_sf

    if ncore > 0:
        vhf = (2.0 * jnp.einsum('pqcc->pq', eri_mo[:, :, :ncore, :ncore])
               - jnp.einsum('pccq->pq', eri_mo[:, :ncore, :ncore, :]))
    else:
        vhf = jnp.zeros_like(h1e_mo)
    h1eff = h1e_mo + vhf
    h1e_act = h1eff[ncore:ncore + nact, ncore:ncore + nact]
    eri_act = jnp.einsum('ijkl->ikjl',
                         eri_mo[ncore:ncore + nact, ncore:ncore + nact,
                                ncore:ncore + nact, ncore:ncore + nact])
    dm1, dm2, dm3 = make_dm123_sf(c_ci, dets, nact, ncore)
    ci_ab = c_ci
    if c_ci.ndim == 1:
        nele_act = _nele_act_from_dets(dets, ncore)
        idx_ab, phase_ab = ab_index_map(ncore, nact, nele_act)
        ci_ab = c_ci[idx_ab] * phase_ab
        ci_ab = ci_ab / jnp.linalg.norm(ci_ab)
    eri_t = eri_act.transpose(0, 2, 1, 3)
    na, nb = ci_ab.shape
    _, f3ca = fdm2fdm3_jax(eri_t, ci_ab, range(na), 0, nb, nact, links_a,
                           links_b, na, nb, kern='f3ca')
    _, f3ac = fdm2fdm3_jax(eri_t, ci_ab, range(na), 0, nb, nact, links_a,
                           links_b, na, nb, kern='f3ac')
    return dict(h1e_act=h1e_act, eri_act=eri_act, dm1=dm1, dm2=dm2,
                dm3=dm3, f3ca=f3ca, f3ac=f3ac, h1eff=h1eff)


def _e2_core_exact_si(blocks, eps, ncore, nact, eri_mo):
    """``core.nevpt2.core.e2_core`` with the Si class swapped.

    A copy of the release assembly restricted to the core classes; the
    slices and the contraction order are identical to ``e2_core``, and
    only the Si call differs.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.core import (make_a3, make_a9, make_a12, make_a13,
                                    make_hdm1, make_hdm2, make_hdm3,
                                    make_k27, sij_e2, sijr_e2, sir_e2,
                                    sijrs_e2, srsi_e2)

    h1e_act = blocks['h1e_act']
    eri_act = blocks['eri_act']
    dm1, dm2, dm3 = blocks['dm1'], blocks['dm2'], blocks['dm3']
    f3ca, f3ac = blocks['f3ca'], blocks['f3ac']
    nocc = ncore + nact
    nvir = eri_mo.shape[0] - nocc
    eps_core = eps[:ncore]
    eps_vir = eps[nocc:]
    t = jnp.einsum('prqs->pqrs', eri_mo)
    h2e_v_si = t[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
    h1e_v_si = blocks['h1eff'][ncore:nocc, :ncore]
    h2e_v_sij = t[ncore:nocc, ncore:nocc, :ncore, :ncore]
    h2e_v_sijr = t[nocc:, ncore:nocc, :ncore, :ncore]
    h2e_v_srsi = t[nocc:, nocc:, :ncore, ncore:nocc]
    h2e_v1_sir = t[nocc:, ncore:nocc, :ncore, ncore:nocc]
    h2e_v2_sir = t[nocc:, ncore:nocc, ncore:nocc, :ncore]
    h1e_v_sir = blocks['h1eff'][nocc:, :ncore]
    cvcv = eri_mo[:ncore, nocc:, :ncore, nocc:].reshape(ncore * nvir,
                                                        ncore * nvir)
    hdm1 = make_hdm1(dm1)
    hdm2 = make_hdm2(dm1, dm2)
    hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2)

    e2 = si_e2_exact(h1e_act, eri_act, h2e_v_si, h1e_v_si, dm1, dm2, dm3,
                     f3ca, f3ac, eps_core)
    e_sijrs, _ = sijrs_e2(cvcv, eps_core, eps_vir, ncore, nvir)
    e2 = e2 + e_sijrs
    a3 = make_a3(h1e_act, eri_act, dm1, dm2, hdm1)
    e2 = e2 + sijr_e2(h2e_v_sijr, hdm1, a3, eps_vir, eps_core, ncore)
    k27 = make_k27(h1e_act, eri_act, dm1, dm2)
    e2 = e2 + srsi_e2(h2e_v_srsi, dm1, k27, eps_vir, eps_core, nvir)
    a9 = make_a9(h1e_act, eri_act, hdm1, hdm2, hdm3)
    e2 = e2 + sij_e2(h2e_v_sij, hdm2, a9, eps_core, ncore)
    a12 = make_a12(h1e_act, eri_act, dm1, dm2, dm3)
    a13 = make_a13(h1e_act, eri_act, dm1, dm2, dm3)
    e2 = e2 + sir_e2(h2e_v1_sir, h2e_v2_sir, h1e_v_sir, dm1, dm2, dm3,
                     a12, a13, eps_core, eps_vir)
    return e2


def nevpt2_e2_full2_si_exact(h1e_mo, eri_mo, c_ci, dets, E0_elec,
                             mo_energy=None, ncore=0, nact=2, nocc=2,
                             links_a=None, links_b=None):
    """``nevpt2_e2_full2`` with the exact-Si core classes.

    The release total is computed as is; then both core assemblies are
    evaluated on the same recomputed blocks and the difference is applied.
    With ``ncore == 0`` or without link tables the release path skips the
    core classes and this function is the release function.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.core import e2_core
    from ..core.nevpt2.impl import nevpt2_e2_full2

    e2_ref, eps = nevpt2_e2_full2(
        h1e_mo, eri_mo, c_ci, dets, E0_elec, mo_energy=mo_energy,
        ncore=ncore, nact=nact, nocc=nocc, links_a=links_a, links_b=links_b)
    if ncore <= 0 or links_a is None or c_ci is None:
        return e2_ref, eps
    blocks = _core_blocks(h1e_mo, eri_mo, c_ci, dets, ncore, nact, nocc,
                          eps, links_a, links_b)
    e2_masked = e2_core(blocks['h1e_act'], blocks['eri_act'], blocks['dm1'],
                        blocks['dm2'], blocks['dm3'], eps, ncore, nact,
                        blocks['f3ca'], blocks['f3ac'], eri_mo=eri_mo,
                        h1eff=blocks['h1eff'])
    e2_exact = _e2_core_exact_si(blocks, eps, ncore, nact, eri_mo)
    return e2_ref - e2_masked + e2_exact, eps


def _components_from_si_exact(C0_, h_, e_, enuc, th_, *, ncore, nact, nele,
                              nmo, pairs, dets, links_a, links_b,
                              ci_solver='plain', eig_solver='default',
                              want_e2=True):
    """``driver.nevpt2._components_from`` with the exact-Si e2."""
    from jax import numpy as jnp

    from .nevpt2 import _casscf_solution

    E0c, _C_can, ci2, eps, h1c, eric = _casscf_solution(
        C0_, h_, e_, th_, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
        pairs=pairs, ci_solver=ci_solver, eig_solver=eig_solver)
    if not want_e2:
        return E0c + enuc, jnp.zeros(())
    e2, _ = nevpt2_e2_full2_si_exact(
        h1c, eric, jnp.asarray(ci2), dets, E0c, mo_energy=eps, ncore=ncore,
        nact=nact, nocc=ncore + nact, links_a=links_a, links_b=links_b)
    return E0c + enuc, e2


def make_nevpt2_joint_fn_si_exact(ncore, nact, nele, nmo, pairs,
                                  scf_builder, e2_scale=1.0,
                                  ci_solver='plain', eig_solver='default'):
    """Build ``E(mol, theta)`` with the exact-Si core classes.

    Same contract as ``driver.nevpt2.make_nevpt2_joint_fn``; use it to
    measure what the Si-channel treatment changes in the frozen and
    relaxed gradients without touching the release chain.
    """
    from ..core.nevpt2.impl import cas_dets_for2
    from .nevpt2 import _spin_links

    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              dets=dets, links_a=links_a, links_b=links_b,
              ci_solver=ci_solver, eig_solver=eig_solver)

    def components(m_, th_, want_e2=True):
        mf_ = scf_builder(m_)
        mf_.kernel()
        return _components_from_si_exact(
            mf_.mo_coeff, mf_.get_hcore(),
            m_.intor('int2e', aosym='s1'), m_.energy_nuc(), th_,
            want_e2=want_e2, **kw)

    def E_joint(m_, th_):
        E_cas, e2 = components(m_, th_, want_e2=(e2_scale != 0.0))
        return E_cas + e2_scale * e2

    E_joint.components = components
    return E_joint


def orbital_zvector(H_tt, source, rcond=None, symmetrize=True):
    """Solve the adjoint (Z-vector) equation ``H_tt z = source``.

    Returns ``(z, info)``.  ``rcond`` is forwarded to ``numpy.linalg.pinv``
    and is a *relative* cutoff, as in ``core.hessian``; the H2O CASSCF
    Hessian carries one soft mode (eigenvalue 3.8e-6, kappa 2.3e7) whose
    response is not part of the gradient source, so pass 1e-7 rather than
    the looser default when reproducing the relaxed-gradient numbers.
    ``info`` records the residual of the kept subspace, the eigenvalues
    and the projection of the source on the discarded modes.
    """
    H = np.asarray(H_tt, dtype=float)
    s = np.asarray(source, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] != s.shape[0]:
        raise ValueError('H_tt must be square and match the source length')
    if symmetrize:
        H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    pinv = np.linalg.pinv(H, rcond=rcond)
    z = pinv @ s
    resid = H @ z - s
    scale = max(float(np.abs(s).max()), 1.0)
    info = dict(eigenvalues=eig, condition=float(np.linalg.cond(H)),
                rcond=(None if rcond is None else float(rcond)),
                residual=float(np.abs(resid).max()),
                residual_relative=float(np.abs(resid).max() / scale),
                source_norm=float(np.linalg.norm(s)),
                z_norm=float(np.linalg.norm(z)))
    return z, info


def lagrangian_relaxed_gradient(g_frozen, z, M):
    """``g_frozen - M^T z``: the Lagrangian form of the response term.

    ``z = H_tt^{-1} s`` and ``M = d2E_casscf/dtheta dR``; the correction
    equals ``s^T D`` with ``D = -H_tt^{-1} M``, the response operator the
    relaxed-gradient driver uses.  Both forms are the same number; the
    adjoint form needs one solve for all Cartesian directions.
    """
    g = np.asarray(g_frozen, dtype=float)
    M = np.asarray(M, dtype=float)
    z = np.asarray(z, dtype=float)
    if M.shape[0] != z.shape[0]:
        raise ValueError('M rows must match the Z-vector length')
    return g - (M.T @ z).reshape(g.shape)


def nevpt2_relaxed_gradient_zvector(mol, theta, E_casscf, E_nevpt2,
                                    rcond=None, H_tt=None, M=None,
                                    source=None, g_frozen=None):
    """Relaxed-path NEVPT2 gradient through the orbital Z-vector.

    Parameters mirror ``driver.nevpt2.nevpt2_relaxed_nuclear_gradient``;
    every expensive piece can be injected (``H_tt``, ``M``, ``source``,
    ``g_frozen``) so a verification script can reuse a finite-difference
    response operator and a previously differentiated energy.

    Returns ``(gradient, info)`` with ``info`` holding the Z-vector solve
    diagnostics and the response share along the probe directions the
    caller may add.
    """
    import jax

    from .nevpt2 import nevpt2_nuclear_gradient

    theta = jax.numpy.asarray(np.asarray(theta, dtype=float))
    if g_frozen is None:
        g_frozen = nevpt2_nuclear_gradient(
            mol, lambda m_: E_nevpt2(m_, theta))
    g_frozen = np.asarray(g_frozen, dtype=float)
    if source is None:
        source = np.asarray(
            jax.grad(lambda t: E_nevpt2(mol, t))(theta)
            - jax.grad(lambda t: E_casscf(mol, t))(theta))
    if H_tt is None:
        H_tt = np.asarray(jax.hessian(lambda t: E_casscf(mol, t))(theta))
    if M is None:
        from ..core.hessian import _mixed_derivative
        M = _mixed_derivative(E_casscf, mol, theta)
    M = np.asarray(M, dtype=float)
    z, info = orbital_zvector(H_tt, source, rcond=rcond)
    g = lagrangian_relaxed_gradient(g_frozen, z, M)
    info['response'] = (g - g_frozen).tolist()
    info['response_norm'] = float(np.linalg.norm(g - g_frozen))
    info['g_frozen'] = g_frozen.tolist()
    return g, info
