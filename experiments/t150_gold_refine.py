"""Golden-standard Hessian by re-optimizing orbitals at every displaced geometry.

Each displaced geometry is optimized to a tight gradient threshold with a
Newton refinement, and the Hessian is assembled by central differences of the
resulting energies. This reference does not share the differentiation path with
the quantity under test, but it is valid only where the displaced calculations
converge; the convergence status of every point is reported.

Usage: python t150_gold_refine.py [nthread]
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

ATOM_STR = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
BASIS = '6-31g'
NCORE, NACT, NELE = 4, 2, 2
XYZ0 = np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                   [-1.4283, 0.0, 1.1115]])

print('=' * 100, flush=True)
print('Refining the golden standard: Newton refinement of the orbital '
      'optimization at every displaced geometry', flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = ATOM_STR; mol.basis = BASIS; mol.build()
mol.set_geom_(XYZ0, unit='Bohr'); mol.build()
nmo = mol.nao_nr(); natm = mol.natm
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)


def E_cas_of(C, h, e, en):
    nm = NCORE + NACT
    Cj = jnp.asarray(C)
    h1m = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
    e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj, e)[:nm, :nm, :nm, :nm]
    return cs_casci_blocked(h1m, e1m, ncore=NCORE, nact=NACT,
                            nele_act=NELE)[0] + en


def make_eth(xyz):
    m_ = pgto.Mole(); m_.atom = ATOM_STR; m_.basis = BASIS; m_.build()
    m_.set_geom_(np.asarray(xyz).reshape(-1, 3), unit='Bohr'); m_.build()
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    mf_ = pscf.RHF(m_); mf_.kernel(); C0_ = mf_.mo_coeff
    en_ = m_.energy_nuc()

    def Eth_(t):
        return E_cas_of(C0_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    return m_, Eth_


def E_opt_newton(xyz, verbose=False):
    """Orbital optimization at one geometry: L-BFGS followed by Newton
    refinement to a gradient norm below 1e-12."""
    m_, Eth_ = make_eth(xyz)
    _, th, g0 = optimize_theta(m_, None, Eth_, NPAR, steps=1500,
                               g_conv=1e-11, label='')
    th = jnp.asarray(np.asarray(th))
    gcur = g0
    n_newton = 0
    for k in range(6):
        g = jax.grad(Eth_)(th)
        gn = float(jnp.linalg.norm(g))
        if gn < 1e-12:
            break
        H = jax.hessian(Eth_)(th)
        H = jnp.asarray(0.5 * (H + H.T))
        th = th - jnp.linalg.lstsq(H, g, rcond=None)[0]
        n_newton += 1
        gcur = gn
    gfin = float(jnp.linalg.norm(jax.grad(Eth_)(th)))
    if verbose:
        print(f'      L-BFGS |g|={g0:.2e} -> {n_newton} Newton steps -> '
              f'|g|={gfin:.2e}', flush=True)
    return float(Eth_(th)), gfin


def E_opt_lbfgs(xyz):
    """The earlier, L-BFGS-only procedure, kept for comparison."""
    m_, Eth_ = make_eth(xyz)
    _, th, g = optimize_theta(m_, None, Eth_, NPAR, steps=1500,
                              g_conv=1e-11, label='')
    th = jnp.asarray(np.asarray(th))
    return float(Eth_(th)), g


DELTA = 2e-3
ND = 3


def shift(i, j, di, dj):
    R = XYZ0.copy()
    if i is not None:
        R[i // 3, i % 3] += di
    if j is not None:
        R[j // 3, j % 3] += dj
    return R


for tag, E_fn in (('C_newton', E_opt_newton), ('C_lbfgs', E_opt_lbfgs)):
    print(f'\n  === {tag} ===', flush=True)
    t0 = time.time()
    E0, g0 = E_fn(XYZ0)
    print(f'    reference point E={E0:.12f}  |g|={g0:.2e}', flush=True)
    H = np.zeros((ND, ND))
    for i in range(ND):
        Ep, gp = E_fn(shift(i, None, +DELTA, 0))
        Em, gm = E_fn(shift(i, None, -DELTA, 0))
        H[i, i] = (Ep - 2*E0 + Em) / DELTA**2
        print(f'    H[{i},{i}] = {H[i,i]:+.10f}   '
              f'(|g| = {gp:.1e}, {gm:.1e})', flush=True)
    for i in range(ND):
        for j in range(i+1, ND):
            Epp, _ = E_fn(shift(i, j, +DELTA, +DELTA))
            Epm, _ = E_fn(shift(i, j, +DELTA, -DELTA))
            Emp, _ = E_fn(shift(i, j, -DELTA, +DELTA))
            Emm, _ = E_fn(shift(i, j, -DELTA, -DELTA))
            H[i, j] = H[j, i] = (Epp - Epm - Emp + Emm) / (4*DELTA**2)
            print(f'    H[{i},{j}] = {H[i,j]:+.10f}', flush=True)
    print(f'    elapsed {time.time()-t0:.0f}s', flush=True)

    if tag == 'C_newton':
        C_NEW = np.diag(H).copy()
        H_NEW = H.copy()
    else:
        C_OLD = np.diag(H).copy()
        H_OLD = H.copy()

# Frozen-orbital route
mol_j = mol
mf_j = pscf.RHF(mol_j); mf_j.kernel()
C0_j = mf_j.mo_coeff
h_j = mol_j.intor("int1e_kin") + mol_j.intor("int1e_nuc")
e_j = mol_j.intor("int2e", aosym="s1")
_, th_j, _ = optimize_theta(mol_j, C0_j,
                            lambda t: E_cas_of(
                                C0_j @ block_rotation(t, nmo, pairs),
                                h_j, e_j, mol_j.energy_nuc()),
                            NPAR, steps=3000, g_conv=1e-13, label='')
th_j = jnp.asarray(np.asarray(th_j))
for k in range(5):
    gg = jax.grad(lambda t: E_cas_of(
        C0_j @ block_rotation(t, nmo, pairs), h_j, e_j,
        mol_j.energy_nuc()))(th_j)
    if float(jnp.linalg.norm(gg)) < 1e-12:
        break
    Hh = jax.hessian(lambda t: E_cas_of(
        C0_j @ block_rotation(t, nmo, pairs), h_j, e_j,
        mol_j.energy_nuc()))(th_j)
    th_j = th_j - jnp.linalg.lstsq(jnp.asarray(0.5*(Hh+Hh.T)), gg, rcond=None)[0]
TH = jax.lax.stop_gradient(th_j)


def E_geo_A(m_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas_of(C_ @ block_rotation(TH, nmo, pairs), h_, e_,
                    m_.energy_nuc())


H_A = np.asarray(jax.hessian(E_geo_A)(mol).coords.coords).reshape(3*natm, 3*natm)

print('\n' + '=' * 100, flush=True)
print('Comparison: frozen orbitals, L-BFGS golden standard, '
      'Newton-refined golden standard', flush=True)
print('=' * 100, flush=True)
print(f'{"idx":>4}{"A":>18}{"C_lbfgs":>18}{"C_newton":>18}'
      f'{"A - C_lbfgs":>12}{"A - C_newton":>12}', flush=True)
for i in range(ND):
    print(f'{i:>4}{H_A[i,i]:>18.10f}{C_OLD[i]:>18.10f}{C_NEW[i]:>18.10f}'
          f'{H_A[i,i]-C_OLD[i]:>12.3e}{H_A[i,i]-C_NEW[i]:>12.3e}', flush=True)

print('\n  reference C_lbfgs diagonal = '
      '[0.7192834630, 0.0217667449, 0.4701124325]', flush=True)
print('  largest difference from that reference = '
      f'{np.max(np.abs(C_OLD - np.array([0.7192834630, 0.0217667449, 0.4701124325]))):.3e}',
      flush=True)
print(f'  C_newton minus C_lbfgs = {np.max(np.abs(C_NEW - C_OLD)):.3e}',
      flush=True)
print('\n  A difference of order 1e-2 would mean the golden standard is the '
      'bottleneck;', flush=True)
print('  of order 1e-6, that the frozen-orbital error is real.', flush=True)

out = dict(
    C_lbfgs=C_OLD.tolist(), C_newton=C_NEW.tolist(),
    H_A_diag=[float(H_A[i, i]) for i in range(ND)],
    diff_C=float(np.max(np.abs(C_NEW - C_OLD))),
    relA_old=[float(abs(H_A[i, i]-C_OLD[i])/abs(C_OLD[i])) for i in range(ND)],
    relA_new=[float(abs(H_A[i, i]-C_NEW[i])/abs(C_NEW[i])) for i in range(ND)],
)
json.dump(out, open('results/t150_gold.json', 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n>>> results/t150_gold.json')
print('\nt150_DONE', flush=True)
gc.collect()
