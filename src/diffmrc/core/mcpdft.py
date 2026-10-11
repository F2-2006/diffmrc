#!/usr/bin/env python
"""MC-PDFT on-top energy (tPBE) on a frozen grid -- jnp, differentiable.

Value layer and first derivatives are validated against the pyscf engine
(t344/t355/t379 in this project's worked examples):

  - translation convention (Li Manni et al. 2014; pyscf
    ``get_rho_translated``): gradient rows follow
    ``rho'_t = (grad rho / 2) (1 +- zeta)`` with NO ``grad zeta`` term;
  - the on-top density uses the convention-free identity

        Pi = Pi_act + (rho_f rho_f - rho_a rho_a)

    with ``rho_f`` the spin densities of the full (core + active) one-body
    density and ``rho_a`` those of the active part alone, and

        Pi_act(g) = 1/2 [ ||G0 + G1||^2 - ||G0||^2 - ||G1||^2 ],
        G_s(g)[I] = sum_pq acas[g,p] acas[g,q] (M[(2p+s,2q+s)]^T c)[I],

    which is the factorized form of ``1/2 sum phi^4 (A - B_ss)`` derived
    from the operator identities ``E_pq E_rs = delta_qr E_ps - G2_tot`` and
    ``B_ss = delta_qr E_ps - G2_same``; it removes the ncas^4 tensor and all
    block loops (verified bitwise: 6.6e-17 against the loop form, t379);
  - the xc kernel (exc for the translated spin densities) is injected by the
    caller -- under pyscfad, ``pyscfad.dft.libxc.eval_xc`` is differentiable
    through a custom jvp, so the whole E_ot is differentiable in the CI
    vector and the active-orbital grid values.

The grid is frozen by construction: all geometry dependence sits in the
caller-supplied ``acas`` and densities.
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

__all__ = [
    "build_M",
    "w_sigma",
    "pi_active",
    "eot_integrand",
    "eot_energy",
    "mcpdft_eot",
    "pbe_exc",
    "pbe_eval_xc",
    "exchange_energy",
]


def build_M(c_ci, dets, nso):
    """Single-excitation blocks M[(sp, sq)][I, J] = <D_I|a^dag_sp a_sq|D_J>.

    Same construction as ``core.nevpt2.rdm.make_dm123_sf`` (bitmap masks and
    fermionic phases), returned as a dict keyed by spin-orbital pairs.
    """
    if nso > 63:
        # 1 << k wraps silently past bit 62 of a signed int64; refuse
        # loudly instead of building corrupted masks (same 63-bit window
        # as analysis.ci and the rdm bitmap routes).
        raise ValueError(
            f'nso={nso} spin orbitals exceed the 63-bit window the '
            f'occupation bitmaps are stored in; shift to an '
            f'active-relative layout first')
    det_arr = jnp.asarray(dets, dtype=int)[:, None]
    occ_bit = (det_arr & (1 << jnp.arange(nso))[None, :]) != 0
    XA = occ_bit[:, None, :] & ~occ_bit[None, :, :]
    XB = ~occ_bit[:, None, :] & occ_bit[None, :, :]
    nA, nB = XA.sum(axis=2), XB.sum(axis=2)
    mask1 = (nA == 1) & (nB == 1)
    pos = jnp.cumsum(occ_bit, axis=1) - 1
    i1 = jnp.argmax(XA, axis=2)
    k1 = jnp.argmax(XB, axis=2)
    pos_p = jnp.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_q = jnp.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = jnp.where((pos_p + pos_q) % 2 == 0, 1.0, -1.0)
    eye = jnp.eye(len(dets))
    M = {}
    for sp in range(nso):
        for sq in range(nso):
            m = mask1 & (i1 == sp) & (k1 == sq)
            mf = jnp.where(m, sign1, 0.0)
            if sp == sq:
                mf = mf + jnp.where(eye, occ_bit[:, sp], 0.0)
            M[(sp, sq)] = mf
    return M


def w_sigma(c_ci, dets, ncas, ncore=0):
    """W[s, p, q, I] = (M[(2p+s, 2q+s)]^T c)[I]; shape (2, ncas, ncas, ndet)."""
    M = build_M(c_ci, dets, 2 * (ncore + ncas))
    ndet = len(dets)
    W = jnp.zeros((2, ncas, ncas, ndet))
    for sg in (0, 1):
        for p in range(ncas):
            for q in range(ncas):
                W = W.at[sg, p, q].set(
                    M[(2 * (ncore + p) + sg, 2 * (ncore + q) + sg)].T @ c_ci)
    return W


def pi_active(W, acas):
    """Active-space on-top pair density on the grid.

    W from :func:`w_sigma`; ``acas`` of shape (ngrids, ncas). Returns
    ``Pi_act`` of shape (ngrids,).  Differentiable in both arguments.
    """
    Gs = jnp.einsum('gp,gq,spqI->sgI', acas, acas, W)
    G = Gs.sum(axis=0)
    return 0.5 * ((G * G).sum(axis=1) - (Gs * Gs).sum(axis=(0, 2)))


def eot_integrand(rho_a, g_a, rho_b, g_b, Pi, eval_xc, otxc='tPBE'):
    """Per-grid-point E_ot integrand (weights not yet applied).

    ``rho_a``/``rho_b`` are spin densities, ``g_a``/``g_b`` their gradients
    (ngrids, 3).  ``eval_xc`` is the xc kernel, called as
    ``eval_xc('PBE', rho_t, spin=1, deriv=0)[0]`` (e.g. the differentiable
    ``pyscfad.dft.libxc.eval_xc``).  The sqrt-guard keeps the R >= 1 branch
    NaN-free under AD (no 0 x inf).
    """
    rho_tot = rho_a + rho_b
    rho_avg = 0.5 * rho_tot
    gtot = (g_a + g_b).T                      # (3, ngr)
    safe = rho_avg >= 0.5e-15
    den = jnp.where(safe, rho_avg ** 2, 1.0)
    # The pair density is non-negative; a value below zero is float noise
    # (measured -4e-19 on a real grid, t384d). Fed through the ratio it
    # makes R < 0, hence zeta = sqrt(1 - R) > 1 and one translated density
    # negative at the 1e-18 level -- which the kernel's fractional powers
    # cannot take.
    Pi_pos = jnp.maximum(Pi, 0.0)
    R = jnp.where(safe, Pi_pos / den, jnp.where(Pi_pos == 0, 0.0, 1.0))
    zeta_raw = jnp.sqrt(jnp.where(R < 1.0, 1.0 - R, 1.0))
    zeta = jnp.where(R < 1.0, zeta_raw, 0.0)
    up = rho_avg * (1 + zeta)
    dn = rho_avg * (1 - zeta)
    up_g = 0.5 * (1 + zeta) * gtot
    dn_g = 0.5 * (1 - zeta) * gtot
    rho_t = jnp.stack([jnp.concatenate([up[None], up_g]),
                       jnp.concatenate([dn[None], dn_g])])
    exc = eval_xc('PBE', rho_t, spin=1, deriv=0)[0]
    return exc * (up + dn)


def eot_energy(rho_a, g_a, rho_b, g_b, Pi, weights, eval_xc, otxc='tPBE'):
    """E_ot = sum_g integrand * weight."""
    return jnp.dot(eot_integrand(rho_a, g_a, rho_b, g_b, Pi, eval_xc, otxc),
                   weights)


def mcpdft_eot(c_ci, dets, ncas, ncore, rho_a, g_a, rho_b, g_b,
               rho_a_act, rho_b_act, acas, weights, eval_xc, otxc='tPBE'):
    """Full on-top energy for one CI vector on a frozen grid.

    ``rho_a_act``/``rho_b_act`` are the active-part spin densities (the
    ``rho_f rho_f - rho_a rho_a`` correction); pass the full densities when
    ncore == 0.
    """
    W = w_sigma(c_ci, dets, ncas, ncore)
    Pi = pi_active(W, acas) + (rho_a * rho_b - rho_a_act * rho_b_act)
    return eot_energy(rho_a, g_a, rho_b, g_b, Pi, weights, eval_xc, otxc)


# ---------------------------------------------------------------------------
# The PBE energy density in jnp: the kernel the translated functional needs.
#
# Reusing pyscfad's libxc wrapper pins the whole chain to eager mode -- the
# wrapper is jax.custom_jvp over the C library, and under jit its primal is
# traced, which the C call cannot take (t384 measured it). The driver's
# optimizer jits its gradient and the curvature check runs jax.hessian, so
# the on-top kernel has to be traceable; PBE is closed-form algebra, so it
# is written out here. Validated pointwise against pyscf's libxc
# ``eval_xc('PBE', rho, spin=1, deriv=0)`` (t385).
#
# Conventions: ``rho_t`` is the (2, 4, ngrids) channel layout of the chain
# -- ``rho_t[s, 0]`` the spin density, ``rho_t[s, 1:4]`` its gradient --
# and the return value is the energy density per unit volume (the sum over
# spins), which is what libxc returns under ``spin=1, deriv=0``.

_PI = jnp.pi
# PBE exchange: F(s) = 1 + kappa - kappa / (1 + mu s^2 / kappa).
_PBE_KAPPA = 0.804
_PBE_BETA = 0.06672455060314922
_PBE_GAMMA = 0.031090690869654895          # (1 - ln 2) / pi^2
_PBE_MU = _PBE_BETA * _PI ** 2 / 3.0
_EX_PREF = -1.5 * (3.0 / (4.0 * _PI)) ** (1.0 / 3.0)
_TWO_43 = 2.0 ** (4.0 / 3.0)
_FZZ0 = 8.0 / (9.0 * (_TWO_43 - 2.0))      # f''(0) of the zeta interpolation

# PW92 parameters: (A, alpha1, beta1..beta4) for eps_c(rs, 0), eps_c(rs, 1)
# and the spin stiffness alpha_c(rs). The A values are the original PW92
# ones, not the rounded values of the PBE paper appendix -- libxc's
# GGA_C_PBE carries the original (its LDA part differs from LDA_C_PW,
# which rounds A_0 to 0.031091, by ~5e-6 relative at small rs; measured
# in t385c).
_PW92_UNPOL = (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
_PW92_POL = (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
_PW92_ALPHA = (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)

# Spin-polarized PBE exchange: libxc evaluates the enhancement on the
# doubled spin density (the unpolarized-form exchange scaled per spin),
# i.e. s_libxc^2 = s^2 / 2^(2/3) with s^2 = |grad rho_s|^2/(4 kF^2 rho_s^2)
# and kF = (3 pi^2 rho_s)^(1/3). Measured to ten digits against
# eval_xc('GGA_X_PBE') at s = 0.5/1/2 (t385c).
_EX_S2_FACTOR = 2.0 ** (-2.0 / 3.0)


def _pw92(rs, params):
    """One PW92 term: -2A(1 + a1 rs) ln(1 + 1/(2A Q(rs))), p = 1."""
    A, a1, b1, b2, b3, b4 = params
    q = b1 * jnp.sqrt(rs) + b2 * rs + b3 * rs ** 1.5 + b4 * rs ** 2
    return -2.0 * A * (1.0 + a1 * rs) * jnp.log1p(1.0 / (2.0 * A * q))


def _pbe_exchange_density(rho_a, ga, rho_b, gb):
    """PBE exchange energy density (per volume); spin-scaled LDA times
    the enhancement F(s) = 1 + kappa - kappa / (1 + mu s^2 / kappa)."""
    def _channel(rho_s, g_s):
        safe = jnp.where(rho_s > 1e-30, rho_s, 1.0)
        kf = (3.0 * _PI ** 2 * safe) ** (1.0 / 3.0)
        s2 = (g_s / (2.0 * kf * safe)) ** 2 * _EX_S2_FACTOR
        s2 = jnp.where(rho_s > 1e-30, s2, 0.0)
        fx = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
        return _EX_PREF * safe ** (4.0 / 3.0) * fx * jnp.where(
            rho_s > 1e-30, 1.0, 0.0)

    return _channel(rho_a, ga) + _channel(rho_b, gb)


def _pbe_correlation_density(rho_a, rho_b, gp):
    """PBE correlation energy density (per volume): PW92 LDA plus the
    gradient term H(rs, zeta, t). ``gp`` is the gradient vector of the
    total density, shape (3, ngrids)."""
    rho = rho_a + rho_b
    rho_safe = jnp.where(rho > 1e-30, rho, 1.0)
    rs = (3.0 / (4.0 * _PI * rho_safe)) ** (1.0 / 3.0)
    zeta = jnp.where(rho > 1e-30, (rho_a - rho_b) / rho_safe, 0.0)
    # |zeta| <= 1 physically; a translation noise of order 1e-18 in one
    # spin density can push it past +-1, and (1 -+ zeta)**(4/3) of a
    # negative number is NaN. Clamping at zero restores the exact
    # boundary value (the powers vanish there).
    one_p = jnp.maximum(1.0 + zeta, 0.0)
    one_m = jnp.maximum(1.0 - zeta, 0.0)
    phi = 0.5 * (one_p ** (2.0 / 3.0) + one_m ** (2.0 / 3.0))
    fz = (one_p ** (4.0 / 3.0) + one_m ** (4.0 / 3.0) - 2.0) / (_TWO_43 - 2.0)
    ec0 = _pw92(rs, _PW92_UNPOL)
    ec1 = _pw92(rs, _PW92_POL)
    # The spin stiffness uses the opposite sign convention: it is positive
    # and raises the energy as zeta grows.
    ac = -_pw92(rs, _PW92_ALPHA)
    ec_lda = (ec0 + ac * fz / _FZZ0 * (1.0 - zeta ** 4)
              + (ec1 - ec0) * fz * zeta ** 4)
    phi_safe = jnp.where(phi > 1e-30, phi, 1.0)
    kf_tot = (3.0 * _PI ** 2 * rho_safe) ** (1.0 / 3.0)
    ks = jnp.sqrt(4.0 * kf_tot / _PI)
    gt = jnp.sqrt((gp ** 2).sum(axis=0))
    t2 = (gt / (2.0 * phi_safe * ks * rho_safe)) ** 2
    t2 = jnp.where(rho > 1e-30, t2, 0.0)
    a_g = (_PBE_BETA / _PBE_GAMMA) / (
        jnp.exp(-ec_lda / (_PBE_GAMMA * phi_safe ** 3)) - 1.0)
    h = _PBE_GAMMA * phi_safe ** 3 * jnp.log1p(
        (_PBE_BETA / _PBE_GAMMA) * t2 * (1.0 + a_g * t2)
        / (1.0 + a_g * t2 + a_g ** 2 * t2 ** 2))
    return rho_safe * (ec_lda + h) * jnp.where(rho > 1e-30, 1.0, 0.0)


def pbe_exc(rho_t):
    """The PBE exchange-correlation energy per particle, in jnp.

    ``rho_t`` has the (2, 4, ngrids) channel layout of the translated
    densities; the return matches pyscf libxc
    ``eval_xc('PBE', rho_t, spin=1, deriv=0)[0]`` -- the per-particle
    energy, which the chain multiplies by the total density. Validated
    pointwise against libxc (t385). Differentiable and jit-safe at every
    order: no C call, no custom derivative.
    """
    rho_a, rho_b = rho_t[0, 0], rho_t[1, 0]
    ga = jnp.sqrt((rho_t[0, 1:4] ** 2).sum(axis=0))
    gb = jnp.sqrt((rho_t[1, 1:4] ** 2).sum(axis=0))
    rho = rho_a + rho_b
    rho_safe = jnp.where(rho > 1e-30, rho, 1.0)
    eps = (_pbe_exchange_density(rho_a, ga, rho_b, gb)
           + _pbe_correlation_density(rho_a, rho_b,
                                      rho_t[0, 1:4] + rho_t[1, 1:4]))
    exc = eps / rho_safe
    return jnp.where(rho > 1e-30, exc, 0.0)


def pbe_eval_xc(xc_code, rho, spin=0, relativity=0, deriv=0, omega=None,
                verbose=None):
    """``eval_xc``-compatible adapter around :func:`pbe_exc`.

    Same call signature and return convention as the pyscf/pyscfad kernels
    the chain accepts (``exc`` first), but restricted to the channel the
    translated functional actually asks for: ``'PBE'``, ``spin=1``,
    ``deriv=0``. Unlike the pyscfad wrapper this is jit- and hessian-safe.
    """
    if xc_code != 'PBE' or spin != 1 or deriv != 0:
        raise ValueError(
            f"the jnp PBE kernel implements eval_xc('PBE', spin=1, deriv=0) "
            f"only, got ({xc_code!r}, spin={spin}, deriv={deriv})")
    return (pbe_exc(rho), None, None, None)


def exchange_energy(D_a, D_b, eri):
    """HF-like exchange energy of the full spin-resolved AO densities.

    ``E_x = -1/2 sum_sigma <K[D_sigma] | D_sigma>`` with
    ``K[D][p,q] = sum_rs (pr|qs) D[r,s]`` -- the engine's ``E_x`` inside
    ``energy_mcwfn`` (its ``vk`` from ``get_jk`` on the full density).
    It is *not* the wavefunction's exact exchange: the engine's own
    comment says as much, and the hybrid tPBE0 mixes exactly this term
    back into the classical part (t390 pinned the assembly:
    ``E = classical + lambda (E_x + E_c) + (1 - lambda) E_ot`` with
    ``lambda = 0.25``).  For the total energy the pair ``E_x + E_c``
    is obtained without any 2-RDM convention: the standard RDM split
    gives ``E_x + E_c = <Vee> - E_j = E_CASSCF - classical`` (checked
    to the digit, t391), which is why the chain carries the hybrid as
    the convex combination ``(1 - lambda) E_tPBE + lambda E_CASSCF``
    and this primitive stays a diagnostic.
    """
    def _k(D):
        return jnp.einsum('prqs,rs->pq', eri, D)

    K_a, K_b = _k(D_a), _k(D_b)
    return -0.5 * ((K_a * D_a).sum() + (K_b * D_b).sum())
