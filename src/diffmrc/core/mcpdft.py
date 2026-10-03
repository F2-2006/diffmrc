#!/usr/bin/env python
"""MC-PDFT on-top energy (tPBE) on a frozen grid -- jnp, differentiable.

Value layer and first derivatives are validated against the pyscf engine
(t344/t355/t379 in the project worked examples):

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
]


def build_M(c_ci, dets, nso):
    """Single-excitation blocks M[(sp, sq)][I, J] = <D_I|a^dag_sp a_sq|D_J>.

    Same construction as ``core.nevpt2.rdm.make_dm123_sf`` (bitmap masks and
    fermionic phases), returned as a dict keyed by spin-orbital pairs.
    """
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
    R = jnp.where(safe, Pi / den, jnp.where(Pi == 0, 0.0, 1.0))
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
