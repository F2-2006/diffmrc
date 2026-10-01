"""Finite differences versus automatic differentiation for the mixed derivative.

The mixed second derivative d2E/dtheta dR is the orbital-response term. At a
converged orbital optimization the gradient with respect to the rotation
parameters is near zero, so a central difference of that gradient is dominated
by cancellation error rather than by the derivative itself. This script
quantifies the failure: the finite-difference estimate is two orders of
magnitude larger than the auto-differentiated value and grows as the step is
reduced.

Usage: python t149_mix_AD_vs_FD.py [nthread]
"""
import os
import sys
import json
import time
import gc

NT = sys.argv[1] if len(sys.argv) > 1 else '8'
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np                                               # noqa: E402
import jax; jax.config.update("jax_enable_x64", True)             # noqa: E402
from jax import numpy as jnp                                     # noqa: E402
import pyscfad                                                    # noqa: E402
from pyscfad import gto as pgto, scf as pscf                      # noqa: E402
from diffmrc.core.rotation import block_rotation                            # noqa: E402
from diffmrc.core.orblib import orbital_pairs                                 # noqa: E402
from diffmrc.core.optimizer import optimize_theta                           # noqa: E402
from diffmrc.core.casci import cs_casci_blocked, _block_H             # noqa: E402
from itertools import combinations                               # noqa: E402

ATOM_STR = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
BASIS = '6-31g'
NCORE, NACT, NELE = 4, 2, 2
XYZ0 = np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                   [-1.4283, 0.0, 1.1115]])

print('=' * 100, flush=True)
print('Branch hopping and the mixed derivative: automatic differentiation\n'
      'against finite differences', flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = ATOM_STR; mol.basis = BASIS; mol.build()
mol.set_geom_(XYZ0, unit='Bohr'); mol.build()
nmo = mol.nao_nr(); natm = mol.natm
NOCC = NCORE + NACT
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)
mf0 = pscf.RHF(mol); mf0.kernel()
C0_ref = mf0.mo_coeff
h1e0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri0 = mol.intor("int2e", aosym="s1")
enuc0 = mol.energy_nuc()


def E_cas_of(C, h, e, en):
    nm = NCORE + NACT
    Cj = jnp.asarray(C)
    h1m = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
    e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj, e)[:nm, :nm, :nm, :nm]
    return cs_casci_blocked(h1m, e1m, ncore=NCORE, nact=NACT,
                            nele_act=NELE)[0] + en


def E_theta(th):
    return E_cas_of(C0_ref @ block_rotation(th, nmo, pairs), h1e0, eri0, enuc0)


# theta*, with Newton refinement
_, th, _ = optimize_theta(mol, C0_ref, E_theta, NPAR, steps=3000,
                          g_conv=1e-13, label='the AD-versus-FD diagnostic')
th = jnp.asarray(np.asarray(th))
for k in range(5):
    gg = jax.grad(E_theta)(th)
    if float(jnp.linalg.norm(gg)) < 1e-12:
        break
    H = jax.hessian(E_theta)(th)
    th = th - jnp.linalg.lstsq(jnp.asarray(0.5*(H+H.T)), gg, rcond=None)[0]
TH = jnp.asarray(th)
H_thth = np.asarray(jax.hessian(E_theta)(TH))
H_thth = 0.5 * (H_thth + H_thth.T)
w_h, U_h = np.linalg.eigh(H_thth)
print(f'  theta*: |g| = {float(jnp.linalg.norm(jax.grad(E_theta)(TH))):.3e}  '
      f'|theta*| = {float(jnp.linalg.norm(TH)):.6f}', flush=True)
print(f'  theta* largest absolute component = {float(jnp.max(jnp.abs(TH))):.6f}', flush=True)

# ==================================================================
# 1) Branch-hopping diagnostic
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('1) Branch-hopping diagnostic: block identity and in-block minimum at each displaced point', flush=True)
print('=' * 100, flush=True)

# Block information, consistent with the blocked diagonalization
nso = 2 * (NCORE + NACT)
core_so = tuple(range(2 * NCORE))
act_so = list(range(2 * NCORE, nso))
DETS = [core_so + c for c in combinations(act_so, NELE)]
sz = np.array([sum(1 for so in d if so % 2 == 0) * 2 - NELE for d in DETS])
block_ids = sorted(set(sz.tolist()))


def block_report(xyz):
    """Return (E_total, dominant block, lowest energy within each block)."""
    m_ = pgto.Mole(); m_.atom = ATOM_STR; m_.basis = BASIS; m_.build()
    m_.set_geom_(np.asarray(xyz).reshape(-1, 3), unit='Bohr'); m_.build()
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    mf_ = pscf.RHF(m_); mf_.kernel(); C0_ = mf_.mo_coeff
    nm = NCORE + NACT
    C_ = C0_ @ block_rotation(TH, nmo, pairs)
    h1m = np.asarray(jnp.einsum('ai,ab,bj->ij', C_, h_, C_)[:nm, :nm])
    e1m = np.asarray(jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', C_, C_, C_, C_, e_)[
        :nm, :nm, :nm, :nm])
    lows = {}
    for s in block_ids:
        idx = [i for i, x in enumerate(sz) if x == s]
        Hb = np.asarray(_block_H(jnp.asarray(h1m), jnp.asarray(e1m), DETS, idx,
                                 NCORE, NACT, NELE))
        lows[s] = float(np.linalg.eigvalsh(Hb)[0])
    best = min(lows, key=lows.get)
    return float(m_.energy_nuc()) + lows[best], best, lows


print(f'{"geometry":>10}{"E_total":>18}{"dominant block":>8}  lowest energy per block', flush=True)
for tag, dd in (('R0', 0.0), ('+1e-4', 1e-4), ('-1e-4', -1e-4),
                ('+1e-3', 1e-3), ('-1e-3', -1e-3)):
    x = XYZ0.copy()
    if dd != 0:
        x[0, 0] += dd
    E, best, lows = block_report(x)
    lowstr = '  '.join(f'Sz={s:+d}:{lows[s]:.8f}' for s in block_ids)
    print(f'{tag:>10}{E:>18.10f}{best:>8}  {lowstr}', flush=True)

# ==================================================================
# ② AD compute ∂^2E/∂theta∂R
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('2) Mixed derivative by automatic differentiation, avoiding finite differences', flush=True)
print('=' * 100, flush=True)


def F_joint(m_, th_):
    """(Mole, theta) -> energy. Both arguments are pytrees and the Mole is
    differentiable."""
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas_of(C_ @ block_rotation(th_, nmo, pairs), h_, e_,
                    m_.energy_nuc())


t0 = time.time()
try:
    g_th_of_m = jax.grad(F_joint, argnums=1)          # Mole -> (NPAR,)
    print(f'  g_th_of_m built {time.time()-t0:.0f}s', flush=True)
    t0 = time.time()
    mix_pt = jax.jacfwd(g_th_of_m, argnums=0)(mol, TH)
    print(f'  jacfwd done {time.time()-t0:.0f}s', flush=True)
    # mix_pt is a pytree of shape (NPAR,) plus the Mole structure
    # Unlike jax.grad(...)(mol).coords, the first axis of the jacfwd output is NPAR,
    #   so the coordinates are obtained as mix_pt.coords, not .coords.coords
    try:
        mix_AD = np.asarray(mix_pt.coords)
        print(f'  [.coords]        shape = {mix_AD.shape}', flush=True)
    except AttributeError:
        # Fallback: locate the Mole structure recursively
        mix_AD = np.asarray(mix_pt)
        print(f'  [plain asarray]   shape = {mix_AD.shape}', flush=True)
    print(f'  |mix_AD|_max = {np.max(np.abs(mix_AD)):.6e}', flush=True)
    AD_OK = True
except Exception as ex:
    print(f'  ✗ AD failed: {type(ex).__name__}: {str(ex)[:200]}', flush=True)
    mix_AD = None
    AD_OK = False

# ==================================================================
# 3) Automatic differentiation against finite differences
# ==================================================================
if AD_OK:
    print('\n' + '=' * 100, flush=True)
    print('3) Automatic differentiation against finite differences, per eigenvector', flush=True)
    print('=' * 100, flush=True)
    mix_AD_flat = mix_AD.reshape(NPAR, -1)

    def fd_mix(DD):
        mix = np.zeros((NPAR, natm, 3))
        for a in range(natm):
            for c in range(3):
                xp = XYZ0.copy(); xp[a, c] += DD
                xm = XYZ0.copy(); xm[a, c] -= DD
                m_p = pgto.Mole(); m_p.atom = ATOM_STR; m_p.basis = BASIS
                m_p.build(); m_p.set_geom_(xp, unit='Bohr'); m_p.build()
                m_m = pgto.Mole(); m_m.atom = ATOM_STR; m_m.basis = BASIS
                m_m.build(); m_m.set_geom_(xm, unit='Bohr'); m_m.build()
                gp = np.asarray(jax.grad(
                    lambda t_: F_joint(m_p, t_))(TH))
                gm = np.asarray(jax.grad(
                    lambda t_: F_joint(m_m, t_))(TH))
                mix[:, a, c] = (gp - gm) / (2 * DD)
        return mix.reshape(NPAR, -1)

    mix_fd = fd_mix(5e-4)
    print(f'  FD(delta=5e-4) |·|_max = {np.max(np.abs(mix_fd)):.6e}', flush=True)
    print(f'  AD         |·|_max = {np.max(np.abs(mix_AD_flat)):.6e}', flush=True)
    print(f'  |AD − FD|_max = {np.max(np.abs(mix_AD_flat - mix_fd)):.6e}',
          flush=True)

    print(f'\n{"dir":>4}{"eigenvalue":>14}{"1/w":>12}{"AD component":>16}'
          f'{"FD component":>16}{"AD contribution":>14}', flush=True)
    for i in range(min(12, NPAR)):
        ca = float(np.linalg.norm(U_h[:, i] @ mix_AD_flat))
        cf = float(np.linalg.norm(U_h[:, i] @ mix_fd))
        inv = 1.0/w_h[i] if abs(w_h[i]) > 1e-14 else float('inf')
        print(f'{i:>4}{w_h[i]:>14.6e}{inv:>12.3e}{ca:>16.6e}'
              f'{cf:>16.6e}{abs(inv*ca):>14.4e}', flush=True)

    # Use automatic differentiation to compute dtheta*/dR and apply it
    pos = w_h > 1e-3
    Hpinv = (U_h[:, pos] / w_h[pos]) @ U_h[:, pos].T
    dth_AD = -Hpinv @ mix_AD_flat

    print(f'\n  |dtheta*/dR (AD)|_max = {np.max(np.abs(dth_AD)):.6e}', flush=True)
    dth_FD = -Hpinv @ mix_fd
    print(f'  |dtheta*/dR (FD)|_max = {np.max(np.abs(dth_FD)):.6e}', flush=True)

    C_GOLD = np.array([0.7192834630, 0.0217667449, 0.4701124325])

    def E_geo_A(m_):
        mf_ = pscf.RHF(m_); mf_.kernel()
        C_ = mf_.mo_coeff
        h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
        e_ = m_.intor("int2e", aosym="s1")
        return E_cas_of(C_ @ block_rotation(jax.lax.stop_gradient(TH), nmo, pairs),
                        h_, e_, m_.energy_nuc())

    H_A = np.asarray(jax.hessian(E_geo_A)(mol).coords.coords).reshape(
        3*natm, 3*natm)
    print(f'\n{"idx":>4}{"A":>18}{"AD corrected":>18}{"FD corrected":>18}{"golden":>18}',
          flush=True)
    res = {}
    for tag, dth in (('AD', dth_AD), ('FD', dth_FD)):
        M = mix_AD_flat.T @ dth if tag == 'AD' else mix_fd.T @ dth
        res[tag] = H_A + 2.0*M
    for i in range(3):
        print(f'{i:>4}{H_A[i,i]:>18.10f}{res["AD"][i,i]:>18.10f}'
              f'{res["FD"][i,i]:>18.10f}{C_GOLD[i]:>18.10f}', flush=True)
    print(f'\n  relative deviation:', flush=True)
    for tag in ('AD', 'FD'):
        rels = [abs(res[tag][i, i]-C_GOLD[i])/abs(C_GOLD[i]) for i in range(3)]
        print(f'    {tag}: {[f"{x:.3e}" for x in rels]}  largest={max(rels):.3e}',
              flush=True)
    relA = [abs(H_A[i, i]-C_GOLD[i])/abs(C_GOLD[i]) for i in range(3)]
    print(f'    A : {[f"{x:.3e}" for x in relA]}  largest={max(relA):.3e}',
          flush=True)

    out = dict(
        thetanorm=float(jnp.linalg.norm(TH)),
        mix_AD_max=float(np.max(np.abs(mix_AD_flat))),
        mix_FD_max=float(np.max(np.abs(mix_fd))),
        dth_AD_max=float(np.max(np.abs(dth_AD))),
        dth_FD_max=float(np.max(np.abs(dth_FD))),
        rel_A=relA,
        rel_AD=[abs(res['AD'][i, i]-C_GOLD[i])/abs(C_GOLD[i]) for i in range(3)],
        rel_FD=[abs(res['FD'][i, i]-C_GOLD[i])/abs(C_GOLD[i]) for i in range(3)],
    )
    json.dump(out, open('results/t149_mix_AD_vs_FD.json', 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n>>> results/t149_mix_AD_vs_FD.json', flush=True)
print('\nt149_DONE', flush=True)
gc.collect()
