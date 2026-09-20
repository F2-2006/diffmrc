"""Generalization of the unrolled-iteration response to CO and N2.

The full Newton expansion is exact but its cost scales with both the number of
rotation parameters and the number of iterations. This script tests a
frozen-Hessian quasi-Newton variant, in which the Hessian is evaluated once at
the reference geometry; the fixed point is still defined by g(theta) = 0, so
the derivative with respect to geometry retains the complete response. The
golden standard uses a warm start from the converged reference parameters to
track a single solution branch.

Usage: python t152_generalize.py <H2O|CO|N2> <freeze|full|gold|all> [NT]
"""
import os
import sys
import json
import time
import gc

SYS = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'freeze'
NT = sys.argv[3] if len(sys.argv) > 3 else '16'

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stack'))
import numpy as np                                                # noqa: E402
import jax; jax.config.update("jax_enable_x64", True)              # noqa: E402
from jax import numpy as jnp                                      # noqa: E402
import pyscfad                                                     # noqa: E402
from pyscfad import gto as pgto, scf as pscf                       # noqa: E402
from casscf_lih import block_rotation                             # noqa: E402
from orblib import orbital_pairs                                  # noqa: E402
from optim_theta import optimize_theta                            # noqa: E402
from casci_pyscfad import cs_casci_blocked                        # noqa: E402

BASIS = '6-31g'
NACT, NELE = 2, 2
REG = 1e-6
DELTA = 2e-3                      # central-difference step, as in the golden standard
N_LIST_FREEZE = (1, 2, 3, 5)
N_LIST_FULL = (1, 3)

SYSTEMS = {
    'H2O': 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
    'N2':  'N 0 0 0; N 0 0 1.0977',
    'CO':  'C 0 0 0; O 0 0 1.1283',
}
# H2O uses the geometry of the golden-standard refinement, so that the
# Newton-refined reference values can be compared directly
XYZ_OVERRIDE = {
    'H2O': np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                       [-1.4283, 0.0, 1.1115]]),
}
# The Newton-refined golden standard for H2O: the three O displacements
GOLD_H2O = np.array([0.7192711493, 0.0217924736, 0.4702696863])

print('=' * 108, flush=True)
print(f'Hessian generalization: {SYS}/{BASIS}  MODE={MODE}  T={NT}', flush=True)
print('=' * 108, flush=True)

mol = pgto.Mole(); mol.atom = SYSTEMS[SYS]; mol.basis = BASIS; mol.build()
if SYS in XYZ_OVERRIDE:
    mol.set_geom_(XYZ_OVERRIDE[SYS], unit='Bohr'); mol.build()
XYZ0 = np.asarray(mol.atom_coords())
nmo = mol.nao_nr(); natm = mol.natm
NCORE = mol.nelectron // 2 - NELE // 2
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)
print(f'  nmo={nmo}  natm={natm}  ncore={NCORE}  nact={NACT}  '
      f'NPAR={NPAR}  dof={3*natm}', flush=True)


def E_cas(C, h, e, en):
    nm = NCORE + NACT
    Cj = jnp.asarray(C)
    h1m = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
    e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                     e)[:nm, :nm, :nm, :nm]
    return cs_casci_blocked(h1m, e1m, ncore=NCORE, nact=NACT,
                            nele_act=NELE)[0] + en


# ==================================================================
# reference geometry  theta*
# ==================================================================
mf0 = pscf.RHF(mol); mf0.kernel()
C0_ref = mf0.mo_coeff
h1e0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri0 = mol.intor("int2e", aosym="s1")
enuc0 = mol.energy_nuc()


def E_theta(th):
    return E_cas(C0_ref @ block_rotation(th, nmo, pairs), h1e0, eri0, enuc0)


def newton_refine(E_of, th, n_max=6, tol=1e-12):
    """Newton refinement to |g| < tol, outside any tracing context."""
    th = jnp.asarray(np.asarray(th))
    k = 0
    for k in range(n_max):
        g = jax.grad(E_of)(th)
        if float(jnp.linalg.norm(g)) < tol:
            break
        H = jax.hessian(E_of)(th)
        th = th - jnp.linalg.lstsq(jnp.asarray(0.5 * (H + H.T)), g,
                                   rcond=None)[0]
    return th, float(jnp.linalg.norm(jax.grad(E_of)(th))), k


t0 = time.time()
_, th, gn0 = optimize_theta(mol, C0_ref, E_theta, NPAR, steps=3000,
                            g_conv=1e-13, label=f'opt-{SYS}')
TH0, GN_STAR, _ = newton_refine(E_theta, th)
E_STAR = float(E_theta(TH0))
print(f'\n  theta*: E={E_STAR:.10f}  |g|={GN_STAR:.3e}  '
      f'|theta*|={float(jnp.linalg.norm(TH0)):.6f}  ({time.time()-t0:.0f}s)',
      flush=True)

# ---------- H0 and its inverse, both constant ----------
H0r = jax.hessian(E_theta)(TH0)
H0 = np.asarray(0.5 * (H0r + H0r.T))
w0 = np.linalg.eigvalsh(H0)
H0_INV = np.linalg.inv(H0 + REG * np.eye(NPAR))
print(f'  H0 spectrum: min={w0[0]:+.4e}  max={w0[-1]:+.4e}  '
      f'κ={abs(w0[-1]/w0[0]):.2e}  n_neg={int((w0 < 0).sum())}', flush=True)


# ==================================================================
# Unrolled iterations, three variants
# ==================================================================
def _setup(m_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return C_, h_, e_, m_.energy_nuc()


def E_unrolled_full(m_, n_steps):
    """Full Newton expansion."""
    C_, h_, e_, en_ = _setup(m_)

    def E_of(t):
        return E_cas(C_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    def body(_, t):
        g = jax.grad(E_of)(t)
        H = jax.hessian(E_of)(t)
        H = 0.5 * (H + H.T)
        return t - jnp.linalg.solve(H + REG * jnp.eye(NPAR), g)

    tf = jax.lax.fori_loop(0, n_steps, body, TH0)
    return E_of(tf)


def E_unrolled_frozen(m_, n_steps):
    """Frozen-Hessian quasi-Newton expansion.

    H0 is constant, so each step costs one gradient evaluation rather than one
    Hessian. The fixed point is still defined by g(theta) = 0, so the derivative
    with respect to geometry retains the complete response.
    """
    C_, h_, e_, en_ = _setup(m_)

    def E_of(t):
        return E_cas(C_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    def body(_, t):
        return t - (H0_INV @ jax.grad(E_of)(t))

    tf = jax.lax.fori_loop(0, n_steps, body, TH0)
    return E_of(tf)


def E_geo_A(m_):
    """Frozen-orbital route, the baseline against which H0 is compared."""
    C_, h_, e_, en_ = _setup(m_)
    TH_SG = jax.lax.stop_gradient(TH0)
    return E_cas(C_ @ block_rotation(TH_SG, nmo, pairs), h_, e_, en_)


def geo_hessian(E_fn):
    """Geometry Hessian of shape (3N, 3N), and the elapsed time."""
    t0 = time.time()
    H = jax.hessian(E_fn)(mol)
    Hs = np.asarray(H.coords.coords).reshape(3 * natm, 3 * natm)
    return Hs, time.time() - t0


RESULT = dict(system=SYS, basis=BASIS, mode=MODE, nthread=NT, npar=NPAR,
              nmo=nmo, natm=natm, ncore=NCORE, E_star=E_STAR,
              grad_theta=GN_STAR, H0_eig=[float(w0[0]), float(w0[-1])])

# ==================================================================
# 1) Self-consistency at the reference geometry
# ==================================================================
print('\n  1) Self-consistency at the reference geometry', flush=True)
_DO_FULL_CHECK = MODE in ('full', 'all')
for N in (1, 3, 5):
    for tag, fn in (('frozen', E_unrolled_frozen), ('full', E_unrolled_full)):
        if tag == 'full' and (not _DO_FULL_CHECK or N not in N_LIST_FULL):
            continue
        try:
            d = float(fn(mol, N)) - E_STAR
            print(f'     {tag:>6} N={N}: Δ = {d:+.3e}', flush=True)
        except Exception as ex:
            print(f'     {tag:>6} N={N}: ✗ {type(ex).__name__}: {str(ex)[:80]}',
                  flush=True)

# ==================================================================
# ② A route
# ==================================================================
if MODE in ('freeze', 'full', 'all'):
    print('\n  2) Frozen-orbital route', flush=True)
    H_A, tA = geo_hessian(E_geo_A)
    dA = H_A.diagonal()[:3]
    print(f'     diag[:3] = {[f"{x:.10f}" for x in dA]}   {tA:.0f}s', flush=True)
    RESULT['A_diag'] = [float(x) for x in dA]
    RESULT['A_time'] = tA

# ==================================================================
# 3) Frozen-Hessian expansion
# ==================================================================
if MODE in ('freeze', 'all'):
    print('\n  3) Frozen-Hessian expansion, the cost-reduced variant', flush=True)
    for N in N_LIST_FREEZE:
        try:
            Hf, tf = geo_hessian(lambda m_: E_unrolled_frozen(m_, N))
            df = Hf.diagonal()[:3]
            rel = [abs(df[i] - GOLD_H2O[i]) / abs(GOLD_H2O[i])
                   for i in range(3)] if SYS == 'H2O' else None
            s = (f'  relative deviation vs golden standard: {[f"{x:.3e}" for x in rel]}  '
                  f'largest={max(rel):.3e}') if rel else ''
            print(f'     N={N}: {[f"{x:.10f}" for x in df]}   {tf:.0f}s{s}',
                  flush=True)
            RESULT[f'frozen_N{N}'] = dict(diag=[float(x) for x in df],
                                          time=tf, rel=rel)
        except Exception as ex:
            print(f'     N={N}: ✗ {type(ex).__name__}: {str(ex)[:120]}',
                  flush=True)
        gc.collect()

# ==================================================================
# 4) Full Newton expansion, checking that freezing introduces no extra error
# ==================================================================
if MODE in ('full', 'all'):
    print('\n  4) Full Newton expansion, for comparison', flush=True)
    for N in N_LIST_FULL:
        try:
            Hn, tn = geo_hessian(lambda m_: E_unrolled_full(m_, N))
            dn = Hn.diagonal()[:3]
            rel = [abs(dn[i] - GOLD_H2O[i]) / abs(GOLD_H2O[i])
                   for i in range(3)] if SYS == 'H2O' else None
            s = (f'  relative deviation vs golden standard: {[f"{x:.3e}" for x in rel]}  '
                  f'largest={max(rel):.3e}') if rel else ''
            print(f'     N={N}: {[f"{x:.10f}" for x in dn]}   {tn:.0f}s{s}',
                  flush=True)
            RESULT[f'full_N{N}'] = dict(diag=[float(x) for x in dn],
                                        time=tn, rel=rel)
        except Exception as ex:
            print(f'     N={N}: ✗ {type(ex).__name__}: {str(ex)[:120]}',
                  flush=True)
        gc.collect()

# ==================================================================
# 5) Golden standard: re-optimize theta from theta* at every displaced point
# ==================================================================
if MODE in ('gold', 'all'):
    print('\n  5) Golden standard by central difference, delta=%.0e' % DELTA, flush=True)
    ND = 3

    def mk_mol(xyz):
        m_ = pgto.Mole(); m_.atom = SYSTEMS[SYS]; m_.basis = BASIS
        m_.build()
        m_.set_geom_(np.asarray(xyz).reshape(-1, 3), unit='Bohr'); m_.build()
        return m_

    def E_opt(xyz, th_start):
        m_ = mk_mol(xyz)
        h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
        e_ = m_.intor("int2e", aosym="s1")
        mf_ = pscf.RHF(m_); mf_.kernel(); C0_ = mf_.mo_coeff
        en_ = m_.energy_nuc()

        def Eth_(t):
            return E_cas(C0_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

        _, thx, g0 = optimize_theta(m_, None, Eth_, NPAR, steps=800,
                                    g_conv=1e-11, theta0=np.asarray(th_start),
                                    label='')
        thx, gfin, _ = newton_refine(Eth_, thx)
        return float(Eth_(thx)), g0, gfin

    def shift(i, j, di, dj):
        R = XYZ0.copy()
        if i is not None:
            R[i // 3, i % 3] += di
        if j is not None:
            R[j // 3, j % 3] += dj
        return R

    t0 = time.time()
    E0g, g0g, gfg = E_opt(XYZ0, TH0)
    print(f'    reference point E={E0g:.12f}  |g|: {g0g:.1e}  ->  {gfg:.1e}', flush=True)

    H_G = np.zeros((ND, ND))
    gworst = 0.0
    for i in range(ND):
        Ep, _, gp = E_opt(shift(i, None, +DELTA, 0), TH0)
        Em, _, gm = E_opt(shift(i, None, -DELTA, 0), TH0)
        H_G[i, i] = (Ep - 2 * E0g + Em) / DELTA**2
        gworst = max(gworst, gp, gm)
        print(f'    H[{i},{i}] = {H_G[i,i]:+.10f}   (|g|={gp:.1e}/{gm:.1e})',
              flush=True)
    for i in range(ND):
        for j in range(i + 1, ND):
            Epp, _, _ = E_opt(shift(i, j, +DELTA, +DELTA), TH0)
            Epm, _, _ = E_opt(shift(i, j, +DELTA, -DELTA), TH0)
            Emp, _, _ = E_opt(shift(i, j, -DELTA, +DELTA), TH0)
            Emm, _, _ = E_opt(shift(i, j, -DELTA, -DELTA), TH0)
            H_G[i, j] = H_G[j, i] = (Epp - Epm - Emp + Emm) / (4 * DELTA**2)
            print(f'    H[{i},{j}] = {H_G[i,j]:+.10f}', flush=True)

    print(f'    elapsed {time.time()-t0:.0f}s  worst |g|={gworst:.1e}', flush=True)
    if SYS == 'H2O':
        print(f'    difference from the Newton-refined reference = '
              f'{np.max(np.abs(H_G.diagonal()[:3] - GOLD_H2O)):.3e}', flush=True)
    RESULT['gold_diag'] = [float(x) for x in H_G.diagonal()]
    RESULT['gold_matrix'] = H_G.tolist()
    RESULT['gold_worst_g'] = gworst
    RESULT['gold_time'] = time.time() - t0

# ==================================================================
# Summary
# ==================================================================
print('\n' + '=' * 108, flush=True)
print('Summary', flush=True)
print('=' * 108, flush=True)

GOLD = GOLD_H2O if SYS == 'H2O' else np.asarray(
    RESULT.get('gold_diag', [np.nan] * 3))[:3]

rows = []
if 'A_diag' in RESULT:
    rows.append(('A (frozen)', np.asarray(RESULT['A_diag']),
                 RESULT.get('A_time', 0)))
for k, v in RESULT.items():
    if k.startswith('frozen_N'):
        rows.append((f'frozen expansion {k.split("_N")[1]:>2}', np.asarray(v['diag']),
                     v['time']))
    if k.startswith('full_N'):
        rows.append((f'full Newton {k.split("_N")[1]:>2}', np.asarray(v['diag']),
                     v['time']))
if MODE in ('gold', 'all'):
    rows.append(('golden standard', np.asarray(RESULT['gold_diag']), RESULT['gold_time']))

print(f'{"route":<16}{"dof0":>16}{"dof1":>16}{"dof2":>16}{"largest relative deviation":>14}'
      f'{"elapsed/s":>10}', flush=True)
print('-' * 90, flush=True)
for tag, v, tt in rows:
    rel = np.abs(v - GOLD) / np.abs(GOLD)
    print(f'{tag:<16}{v[0]:>16.10f}{v[1]:>16.10f}{v[2]:>16.10f}'
          f'{np.max(rel):>14.3e}{tt:>10.0f}', flush=True)

outp = f'results/t152_{SYS}_{MODE}.json'
json.dump(RESULT, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print(f'\nt152_{SYS}_{MODE}_DONE', flush=True)
gc.collect()
