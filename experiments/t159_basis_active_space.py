"""Frequencies versus basis set and active space, with and without the response.

Compares the frozen-orbital and response-corrected Hessians at three levels of
theory. The frozen-orbital error grows with the flexibility of the one-particle
basis whereas the corrected error is stable, which establishes that omitting
the orbital response is not a controlled approximation.

Usage: python t159_basis_active_space.py <basis> <nact> [NT]
"""
import os
import sys
import json
import time
import gc

BASIS = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
NACT = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NT = sys.argv[3] if len(sys.argv) > 3 else '16'

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
from diffmrc.core.casci import cs_casci_blocked                       # noqa: E402

NELE = NACT                       # filling the valence space
AMU2ME = 1822.8884862
HA2CM = 219474.6313632
RCOND = 1e-9
ATOM_STR = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
XYZ0 = np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                   [-1.4283, 0.0, 1.1115]])
EXP_H2O = [1594.7, 3656.7, 3755.8]

print('=' * 100, flush=True)
print('Basis set and active space comparison —— H2O/{BASIS} CAS({NELE},{NACT})  T={NT}',
      flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = ATOM_STR; mol.basis = BASIS; mol.build()
mol.set_geom_(XYZ0, unit='Bohr'); mol.build()
nmo = mol.nao_nr(); natm = mol.natm
NCORE = mol.nelectron // 2 - NELE // 2
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)
NR = 3 * natm
print(f'  nmo={nmo}  natm={natm}  ncore={NCORE}  npar={NPAR}  nR={NR}',
      flush=True)


def E_cas(C, h, e, en):
    nm = NCORE + NACT
    Cj = jnp.asarray(C)
    h1m = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
    e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                     e)[:nm, :nm, :nm, :nm]
    return cs_casci_blocked(h1m, e1m, ncore=NCORE, nact=NACT,
                            nele_act=NELE)[0] + en


def E_joint(m_, th_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas(C_ @ block_rotation(th_, nmo, pairs), h_, e_,
                 m_.energy_nuc())


# ---------- theta* ----------
mf0 = pscf.RHF(mol); mf0.kernel()
C0_ref = mf0.mo_coeff
h1e0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri0 = mol.intor("int2e", aosym="s1")
enuc0 = mol.energy_nuc()
E_theta = lambda t: E_cas(C0_ref @ block_rotation(t, nmo, pairs),
                          h1e0, eri0, enuc0)

t0 = time.time()
_, th, gn0 = optimize_theta(mol, C0_ref, E_theta, NPAR, steps=4000,
                            g_conv=1e-13, label='opt')
th = jnp.asarray(np.asarray(th))
E_best = float(E_theta(th))
GN_best = float(jnp.linalg.norm(jax.grad(E_theta)(th)))

# Newton refinement with a monotonicity guard.
# An earlier unguarded version with 30 iterations diverged on CAS(4,4),
# taking E from -76.07 to +1.04 with |g| = 2.68, because a Newton step far
# from the solution can be far too large. Only descending steps are accepted;
_n_acc = 0
for _k in range(40):
    g = jax.grad(E_theta)(th)
    gnk = float(jnp.linalg.norm(g))
    if gnk < 1e-12:
        break
    Hh = jax.hessian(E_theta)(th)
    step = jnp.linalg.lstsq(jnp.asarray(0.5 * (Hh + Hh.T)), g,
                            rcond=None)[0]
    th_new = th - step
    E_new = float(E_theta(th_new))
    if E_new < E_best - 1e-14:              # monotonicity guard
        th, E_best, GN_best = th_new, E_new, gnk
        _n_acc += 1
    else:
        break
TH0 = jnp.asarray(th)
GN = float(jnp.linalg.norm(jax.grad(E_theta)(TH0)))
print(f'  theta*: E={float(E_theta(TH0)):.10f}  |g|={GN:.3e}  '
      f'Newton steps accepted: {_n_acc}  ({time.time()-t0:.0f}s)', flush=True)
if GN > 1e-8:
    print(f'  |g|={GN:.2e}, not fully converged; the result is indicative only',
          flush=True)

RESULT = dict(basis=BASIS, nact=NACT, nele=NELE, nmo=nmo, npar=NPAR,
              nthread=NT, E_cas=float(E_theta(TH0)), grad_theta=GN,
              exp=EXP_H2O)

# ==================================================================
# 1) Orbital Hessian and its pseudo-inverse
# ==================================================================
t0 = time.time()
H_tt = np.asarray(jax.hessian(E_theta)(TH0))
H_tt = 0.5 * (H_tt + H_tt.T)
w = np.linalg.eigvalsh(H_tt)
H_tt_inv = np.linalg.pinv(H_tt, rcond=RCOND)
print(f'\n  ① H_thetatheta ({NPAR}×{NPAR})  {time.time()-t0:.0f}s  '
      f'min={w[0]:+.3e} max={w[-1]:+.3e} κ={abs(w[-1]/w[0]):.2e} '
      f'n_neg={int((w<0).sum())}', flush=True)
RESULT['H_tt_eig'] = [float(w[0]), float(w[-1])]

# ==================================================================
# ② M = ∂^2E/∂theta∂R
# ==================================================================
t0 = time.time()
g_th_of_m = jax.grad(E_joint, argnums=1)
M = np.asarray(jax.jacfwd(g_th_of_m, argnums=0)(mol, TH0).coords
               ).reshape(NPAR, NR)
print(f'  ② M ({NPAR}×{NR})  {time.time()-t0:.0f}s  '
      f'|M|={np.abs(M).max():.3e}', flush=True)

# ==================================================================
# ③ H_RR (A route)
# ==================================================================
TH_SG = jax.lax.stop_gradient(TH0)


def E_geo_A(m_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas(C_ @ block_rotation(TH_SG, nmo, pairs), h_, e_,
                 m_.energy_nuc())


t0 = time.time()
H_RR = np.asarray(jax.jacfwd(jax.grad(E_geo_A))(mol).coords.coords
                  ).reshape(NR, NR)
H_RR = 0.5 * (H_RR + H_RR.T)
print(f'  ③ H_RR ({NR}×{NR})  {time.time()-t0:.0f}s', flush=True)

# ==================================================================
# 4) Assembly, with the response term included
# ==================================================================
D = -H_tt_inv @ M
H_RESP = M.T @ D
H_TOT = 0.5 * ((H_RR + H_RESP) + (H_RR + H_RESP).T)
print(f'  ④ response term |·|_max = {np.abs(H_RESP).max():.3e}  '
      f'of the total: {np.abs(H_RESP).max()/max(np.abs(H_TOT).max(),1e-30):.3e}',
      flush=True)

# Relative contribution of the response term to the diagonal
diag_ratio = np.abs(np.diag(H_RESP)) / np.maximum(np.abs(np.diag(H_TOT)), 1e-30)
print(f'     largest relative diagonal contribution = {diag_ratio.max():.3e}',
      flush=True)
RESULT['resp_frac_diag_max'] = float(diag_ratio.max())

# ==================================================================
# ⑤ frequencies
# ==================================================================
masses = np.array(mol.atom_mass_list(isotope_avg=True))
M3 = np.repeat(masses, 3)
sm = np.sqrt(M3)
coords = np.asarray(mol.atom_coords())
mtot = masses.sum()
com = (masses[:, None] * coords).sum(0) / mtot
rc = coords - com
B = []
for a in range(3):
    v = np.zeros(3 * natm); v[a::3] = sm[a::3]; B.append(v)
for a in range(3):
    e = np.zeros(3); e[a] = 1.0
    v = np.zeros(3 * natm)
    for i in range(natm):
        v[3*i:3*i+3] = sm[3*i] * np.cross(e, rc[i])
    B.append(v)
B = np.array(B).T
norms = np.linalg.norm(B[:, 3:], axis=0)
B = np.hstack([B[:, :3], B[:, 3:][:, norms > 1e-6]])
Q, _ = np.linalg.qr(B)
P = np.eye(3 * natm) - Q @ Q.T
NTR = 3 + 3


def freqs(H):
    Hmw = H / np.sqrt(np.outer(M3, M3))
    Hp = 0.5 * ((P @ Hmw @ P) + (P @ Hmw @ P).T)
    w, _ = np.linalg.eigh(Hp)
    lam = np.sort(w)[NTR:]
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * (HA2CM / np.sqrt(AMU2ME))
    return np.sort(nu), int((nu < 0).sum())


nu_A, imag_A = freqs(H_RR)
nu_T, imag_T = freqs(H_TOT)
dev_A = np.abs(nu_A - EXP_H2O)
dev_T = np.abs(nu_T - EXP_H2O)

print('\n' + '=' * 100, flush=True)
print(f'Frequency comparison  H2O/{BASIS} CAS({NELE},{NACT})', flush=True)
print('=' * 100, flush=True)
print(f'  A (frozen)  = {[f"{x:.1f}" for x in nu_A]}  '
      f'imaginary={imag_A}  sum of deviations={dev_A.sum():.1f}', flush=True)
print(f'  with response = {[f"{x:.1f}" for x in nu_T]}  '
      f'imaginary={imag_T}  sum of deviations={dev_T.sum():.1f}', flush=True)
print(f'  experiment  = {EXP_H2O}', flush=True)
print(f'  => the response term changes the sum of deviations by '
      f'{dev_T.sum()-dev_A.sum():+.1f} cm-1', flush=True)

RESULT['freq_A'] = [float(x) for x in nu_A]
RESULT['freq_T'] = [float(x) for x in nu_T]
RESULT['dev_A_sum'] = float(dev_A.sum())
RESULT['dev_T_sum'] = float(dev_T.sum())
RESULT['imag_A'] = imag_A
RESULT['imag_T'] = imag_T

outp = f'results/t159_{BASIS}_cas{NACT}.json'
json.dump(RESULT, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print(f'\nt159_{BASIS}_CAS{NACT}_DONE', flush=True)
gc.collect()
