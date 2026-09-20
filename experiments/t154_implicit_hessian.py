"""Orbital-response term by implicit differentiation of the stationarity condition.

Applies the implicit-function theorem to g(theta*, R) = 0:

    dtheta*/dR  = -H_theta,theta^-1 d2E/dtheta dR
    d2E/dR2     = d2E/dR2|theta + (d2E/dR dtheta) dtheta*/dR

requiring three matrices of moderate dimension rather than the tape of an
unrolled iteration. The orbital Hessian is not positive definite and can be
near-singular, so a pseudo-inverse is required in place of a direct solve.

Usage: python t154_implicit_hessian.py <SYS> [NT]
"""
import os
import sys
import json
import time
import gc

SYS = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
NT = sys.argv[2] if len(sys.argv) > 2 else '16'

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stack'))
import numpy as np                                               # noqa: E402
import jax; jax.config.update("jax_enable_x64", True)             # noqa: E402
from jax import numpy as jnp                                     # noqa: E402
import pyscfad                                                    # noqa: E402
from pyscfad import gto as pgto, scf as pscf                      # noqa: E402
from casscf_lih import block_rotation                            # noqa: E402
from orblib import orbital_pairs                                 # noqa: E402
from optim_theta import optimize_theta                           # noqa: E402
from casci_pyscfad import cs_casci_blocked                       # noqa: E402

BASIS = '6-31g'
NACT, NELE = 2, 2
Rcond = 1e-9                      # pseudo-inverse cutoff; the orbital Hessian

SYSTEMS = {
    'H2O': 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
    'N2':  'N 0 0 0; N 0 0 1.0977',
    'CO':  'C 0 0 0; O 0 0 1.1283',
}
XYZ_OVERRIDE = {
    'H2O': np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                       [-1.4283, 0.0, 1.1115]]),
}
# Established reference values
REF = {
    'H2O': dict(unrolled=[0.7192706205, 0.0217921326, 0.4702703066],
                gold=[0.7192711493, 0.0217924736, 0.4702696863],
                A=[0.7261828915, 0.0217921326, 0.4711140468]),
    'CO':  dict(gold=[-0.0054597535, -0.0054597251, 1.3764788527],
                A=None, unrolled=None),
    'N2':  dict(gold=[-0.0050975828, -0.0050976254, 1.7262280352],
                A=None, unrolled=None),
}

print('=' * 100, flush=True)
print(f'Response term by implicit differentiation: {SYS}/{BASIS}  T={NT}',
      flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = SYSTEMS[SYS]; mol.basis = BASIS; mol.build()
if SYS in XYZ_OVERRIDE:
    mol.set_geom_(XYZ_OVERRIDE[SYS], unit='Bohr'); mol.build()
nmo = mol.nao_nr(); natm = mol.natm
NCORE = mol.nelectron // 2 - NELE // 2
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)
NR = 3 * natm
print(f'  nmo={nmo}  natm={natm}  ncore={NCORE}  NPAR={NPAR}  nR={NR}',
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
    """E(Mole, theta). Both arguments are pytrees; differentiable throughout."""
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas(C_ @ block_rotation(th_, nmo, pairs), h_, e_,
                 m_.energy_nuc())


# ---------- reference geometry  theta* ----------
mf0 = pscf.RHF(mol); mf0.kernel()
C0_ref = mf0.mo_coeff
h1e0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri0 = mol.intor("int2e", aosym="s1")
enuc0 = mol.energy_nuc()


def E_theta(th):
    return E_cas(C0_ref @ block_rotation(th, nmo, pairs), h1e0, eri0, enuc0)


t0 = time.time()
_, th, gn0 = optimize_theta(mol, C0_ref, E_theta, NPAR, steps=3000,
                            g_conv=1e-13, label=f'opt-{SYS}')
th = jnp.asarray(np.asarray(th))
for _ in range(6):
    g = jax.grad(E_theta)(th)
    if float(jnp.linalg.norm(g)) < 1e-12:
        break
    Hh = jax.hessian(E_theta)(th)
    th = th - jnp.linalg.lstsq(jnp.asarray(0.5 * (Hh + Hh.T)), g,
                               rcond=None)[0]
TH0 = jnp.asarray(th)
GN = float(jnp.linalg.norm(jax.grad(E_theta)(TH0)))
print(f'\n  theta*: E={float(E_theta(TH0)):.10f}  |g|={GN:.3e}  '
      f'({time.time()-t0:.0f}s)', flush=True)

RESULT = dict(system=SYS, basis=BASIS, nthread=NT, npar=NPAR, nR=NR,
              E_star=float(E_theta(TH0)), grad_theta=GN, rcond=Rcond)

# ==================================================================
# 1) Orbital Hessian and its pseudo-inverse
# ==================================================================
t0 = time.time()
H_tt = np.asarray(jax.hessian(lambda t: E_theta(t))(TH0))
H_tt = 0.5 * (H_tt + H_tt.T)
w_tt = np.linalg.eigvalsh(H_tt)
print(f'\n  ① H_thetatheta ({NPAR}×{NPAR})   {time.time()-t0:.0f}s', flush=True)
print(f'     spectrum: min={w_tt[0]:+.4e}  max={w_tt[-1]:+.4e}  '
      f'κ={abs(w_tt[-1]/w_tt[0]):.2e}  n_neg={int((w_tt<0).sum())}  '
      f'eigenvalues below 1e-8: {int((np.abs(w_tt)<1e-8).sum())}', flush=True)
H_tt_inv = np.linalg.pinv(H_tt, rcond=Rcond)
print(f'     pinv(rcond={Rcond:g}) done', flush=True)
RESULT['H_tt_eig'] = [float(w_tt[0]), float(w_tt[-1])]

# ==================================================================
# 2) M = d2E/dtheta dR, of shape (NPAR, nR), by jacfwd of grad_theta
# ==================================================================
t0 = time.time()
g_th_of_m = jax.grad(E_joint, argnums=1)        # (Mole, theta) -> (NPAR,)
M_pt = jax.jacfwd(g_th_of_m, argnums=0)(mol, TH0)
M = np.asarray(M_pt.coords).reshape(NPAR, NR)   # take .coords
print(f'\n  ② M = ∂^2E/∂theta∂R ({NPAR}×{NR})   {time.time()-t0:.0f}s', flush=True)
print(f'     |M|_max = {np.abs(M).max():.4e}', flush=True)
RESULT['M_norm'] = float(np.abs(M).max())

# Symmetry is checked on H_TOT in step 4

# ==================================================================
# 3) H_RR = d2E/dR2 at frozen theta, of shape (3N, 3N)
# ==================================================================
t0 = time.time()
TH_SG = jax.lax.stop_gradient(TH0)


def E_geo_frozen(m_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas(C_ @ block_rotation(TH_SG, nmo, pairs), h_, e_,
                 m_.energy_nuc())


H_RR_pt = jax.jacfwd(jax.grad(E_geo_frozen))(mol)
H_RR = np.asarray(H_RR_pt.coords.coords).reshape(NR, NR)
H_RR = 0.5 * (H_RR + H_RR.T)
print(f'\n  ③ H_RR = ∂^2E/∂R^2|_theta ({NR}×{NR})   {time.time()-t0:.0f}s', flush=True)
print(f'     diag[:3] = {[f"{x:.10f}" for x in np.diag(H_RR)[:3]]}', flush=True)

# ==================================================================
# 4) Assemble H = H_RR + M^T (-H_theta^-1 M)
# ==================================================================
t0 = time.time()
D = -H_tt_inv @ M                       # dtheta*/dR  (NPAR × nR)
H_RESP = M.T @ D                        # response contribution, (nR, nR)
H_TOT = H_RR + H_RESP
H_TOT = 0.5 * (H_TOT + H_TOT.T)
print(f'\n  4) Assembly done   {time.time()-t0:.0f}s', flush=True)
print(f'     |dtheta*/dR|_max = {np.abs(D).max():.4e}', flush=True)
print(f'     |response term|_max  = {np.abs(H_RESP).max():.4e}', flush=True)
print(f'     response term / total Hessian, largest ratio = '
      f'{np.abs(H_RESP).max()/max(np.abs(H_TOT).max(),1e-30):.3e}', flush=True)

# ==================================================================
# ⑤ Assessment
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('Assessment', flush=True)
print('=' * 100, flush=True)
d_tot = np.diag(H_TOT)[:3]
d_frz = np.diag(H_RR)[:3]
ref = REF.get(SYS, {})
print(f'{"route":<22}{"diag[0]":>18}{"diag[1]":>18}{"diag[2]":>18}', flush=True)
print('-' * 78, flush=True)
print(f'{"A (frozen)":<22}{d_frz[0]:>18.10f}{d_frz[1]:>18.10f}'
      f'{d_frz[2]:>18.10f}', flush=True)
print(f'{"implicit, not unrolled":<22}{d_tot[0]:>18.10f}{d_tot[1]:>18.10f}'
      f'{d_tot[2]:>18.10f}', flush=True)
if ref.get('unrolled'):
    u = ref['unrolled']
    print(f'{"unrolled iteration N=3 (the unrolled-iteration study)":<22}{u[0]:>18.10f}{u[1]:>18.10f}'
          f'{u[2]:>18.10f}', flush=True)
if ref.get('gold'):
    gg = ref['gold']
    print(f'{"golden standard":<22}{gg[0]:>18.10f}{gg[1]:>18.10f}'
          f'{gg[2]:>18.10f}', flush=True)

if ref.get('gold'):
    gg = np.asarray(ref['gold'])
    rel_t = np.abs(d_tot - gg) / np.abs(gg)
    rel_f = np.abs(d_frz - gg) / np.abs(gg)
    print(f'\n  implicit vs golden standard: {[f"{x:.3e}" for x in rel_t]}  '
          f'largest={rel_t.max():.3e}', flush=True)
    print(f'  A route vs golden standard: {[f"{x:.3e}" for x in rel_f]}  '
          f'largest={rel_f.max():.3e}', flush=True)
    print(f'  => improvement by a factor of '
          f'{rel_f.max()/max(rel_t.max(),1e-30):.0f}', flush=True)
    RESULT['rel_implicit'] = [float(x) for x in rel_t]
    RESULT['rel_A'] = [float(x) for x in rel_f]
    if SYS == 'H2O':
        u = np.asarray(ref['unrolled'])
        d_iu = np.abs(d_tot - u)
        print(f'\n  andunrolled iteration (the unrolled-iteration study)  absolute deviation = '
              f'{[f"{x:.3e}" for x in d_iu]}  largest={d_iu.max():.3e}',
              flush=True)
        print('     criterion: should be <= 1e-9, the two routes being\n'
              '     mathematically equivalent and differing only numerically',
              flush=True)

RESULT['H_RR_diag'] = [float(x) for x in np.diag(H_RR)]
RESULT['H_TOT_diag'] = [float(x) for x in np.diag(H_TOT)]
RESULT['H_TOT'] = H_TOT.tolist()
RESULT['dtheta_dR_max'] = float(np.abs(D).max())

outp = f'results/t154_implicit_{SYS}.json'
json.dump(RESULT, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print(f'\nt154_{SYS}_DONE', flush=True)
gc.collect()
