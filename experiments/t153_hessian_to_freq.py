"""Harmonic vibrational frequencies from the mass-weighted Hessian.

The Hessian is mass-weighted, the translational and rotational modes are
projected out, and the remaining eigenvalues are converted to wavenumbers. The
conversion factor is sqrt(lambda) * 5140.487, where the numerical factor
follows from 1 Ha = 219474.6313632 cm^-1 and 1 amu = 1822.8884862 electron
masses.

Usage: python t153_hessian_to_freq.py <H2O|CO|N2> [NT]
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

import numpy as np                                               # noqa: E402
import jax; jax.config.update("jax_enable_x64", True)             # noqa: E402
from jax import numpy as jnp                                     # noqa: E402
import pyscfad                                                    # noqa: E402
from pyscfad import gto as pgto, scf as pscf                      # noqa: E402
from pyscf.data import nist                                       # noqa: E402
from diffmrc.core.rotation import block_rotation                            # noqa: E402
from diffmrc.core.orblib import orbital_pairs                                 # noqa: E402
from diffmrc.core.optimizer import optimize_theta                           # noqa: E402
from diffmrc.core.casci import cs_casci_blocked                       # noqa: E402

BASIS = '6-31g'
NACT, NELE = 2, 2
REG = 1e-6
AMU2ME = 1822.8884862
HA2CM = 219474.6313632

SYSTEMS = {
    'H2O': 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
    'N2':  'N 0 0 0; N 0 0 1.0977',
    'CO':  'C 0 0 0; O 0 0 1.1283',
}
XYZ_OVERRIDE = {
    'H2O': np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                       [-1.4283, 0.0, 1.1115]]),
}
# Experimental fundamentals, in cm-1, for reference
EXP = {'H2O': [1594.7, 3656.7, 3755.8],
       'CO':  [2143.3],
       'N2':  [2358.6]}

print('=' * 100, flush=True)
print('Differentiable Hessian to vibrational frequencies —— {SYS}/{BASIS}  T={NT}', flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = SYSTEMS[SYS]; mol.basis = BASIS; mol.build()
if SYS in XYZ_OVERRIDE:
    mol.set_geom_(XYZ_OVERRIDE[SYS], unit='Bohr'); mol.build()
XYZ0 = np.asarray(mol.atom_coords())
nmo = mol.nao_nr(); natm = mol.natm
NCORE = mol.nelectron // 2 - NELE // 2
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)
NDOF = 3 * natm

masses = np.array([mol.atom_mass_list(isotope_avg=True)[i]
                   for i in range(natm)])          # amu
M3 = np.repeat(masses, 3)
print(f'  nmo={nmo}  natm={natm}  NPAR={NPAR}  NDOF={NDOF}', flush=True)
print(f'  masses (amu) = {[f"{m:.4f}" for m in masses]}', flush=True)


# ------------------------------------------------------------------
# Mass weighting and projection
# ------------------------------------------------------------------
def make_proj(coords, m3):
    """Build the translational and rotational projector in mass-weighted coordinates.

    Mass-weighted coordinates: q_i = sqrt(m_i) x_i
    Translation vectors: sqrt(m_i) e_a for a = x, y, z
    Rotation vectors: sqrt(m_i) (n_a x R_i)
    """
    N = len(m3) // 3
    sm = np.sqrt(m3)
    B = []
    # Translation
    for a in range(3):
        v = np.zeros(3 * N)
        v[a::3] = sm[a::3]
        B.append(v)
    # Rotation about the centre of mass
    m_tot = masses.sum()
    com = (masses[:, None] * coords).sum(0) / m_tot
    rc = coords - com
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        v = np.zeros(3 * N)
        for i in range(N):
            cross = np.cross(e, rc[i])
            v[3 * i:3 * i + 3] = sm[3 * i] * cross
        B.append(v)
    B = np.array(B).T                        # (3N, 6)
    # Linear molecule: the smallest rotation-vector norm is zero, so only two
    norms = np.linalg.norm(B[:, 3:], axis=0)
    keep = norms > 1e-6
    n_trans_rot = 3 + int(keep.sum())
    B = np.hstack([B[:, :3], B[:, 3:][:, keep]])
    # Orthonormalize
    Q, _ = np.linalg.qr(B)
    P = np.eye(3 * N) - Q @ Q.T
    return P, n_trans_rot, Q


P_MW, NTR, Q_MW = make_proj(XYZ0, M3)
print(f'  translational and rotational modes = {NTR}  =>  vibrational modes = {NDOF - NTR}', flush=True)


def freqs_from_hess(H, label):
    """(3N, 3N) Hessian in Ha/Bohr^2 to wavenumbers, after projection."""
    H = 0.5 * (H + H.T)
    Hmw = H / np.sqrt(np.outer(M3, M3))
    Hp = P_MW @ Hmw @ P_MW
    Hp = 0.5 * (Hp + Hp.T)
    w, V = np.linalg.eigh(Hp)
    # Keep only the vibrational subspace, discarding the NTR zero modes
    idx = np.argsort(np.abs(w))[NTR:]        # drop the NTR smallest in magnitude
    lam = w[idx]
    # ν̃[cm^-1] = sqrt(lambda[Ha/(Bohr^2·amu)]) × sqrt(Ha -> cm^-1 / amu -> m_e)
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * (HA2CM / np.sqrt(AMU2ME))
    nu = np.sort(nu)
    imag = int((nu < 0).sum())
    return nu, imag, np.sort(np.abs(w))[:NTR + 1]


# ------------------------------------------------------------------
# reference geometry theta* and H0
# ------------------------------------------------------------------
mf0 = pscf.RHF(mol); mf0.kernel()
C0_ref = mf0.mo_coeff
h1e0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri0 = mol.intor("int2e", aosym="s1")
enuc0 = mol.energy_nuc()


def E_cas(C, h, e, en):
    nm = NCORE + NACT
    Cj = jnp.asarray(C)
    h1m = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
    e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                     e)[:nm, :nm, :nm, :nm]
    return cs_casci_blocked(h1m, e1m, ncore=NCORE, nact=NACT,
                            nele_act=NELE)[0] + en


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
    th = th - jnp.linalg.lstsq(jnp.asarray(0.5 * (Hh + Hh.T)), g, rcond=None)[0]
TH0 = jnp.asarray(th)
GN = float(jnp.linalg.norm(jax.grad(E_theta)(TH0)))
print(f'\n  theta*: E={float(E_theta(TH0)):.10f}  |g|={GN:.3e}  '
      f'({time.time()-t0:.0f}s)', flush=True)

H0r = jax.hessian(E_theta)(TH0)
H0_INV = np.linalg.inv(np.asarray(0.5 * (H0r + H0r.T)) + REG * np.eye(NPAR))


def _setup(m_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return C_, h_, e_, m_.energy_nuc()


def E_geo_A(m_):
    C_, h_, e_, en_ = _setup(m_)
    TG = jax.lax.stop_gradient(TH0)
    return E_cas(C_ @ block_rotation(TG, nmo, pairs), h_, e_, en_)


def E_frozen(m_, n_steps):
    C_, h_, e_, en_ = _setup(m_)

    def E_of(t):
        return E_cas(C_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    def body(_, t):
        return t - (H0_INV @ jax.grad(E_of)(t))

    return E_of(jax.lax.fori_loop(0, n_steps, body, TH0))


def E_full(m_, n_steps):
    C_, h_, e_, en_ = _setup(m_)

    def E_of(t):
        return E_cas(C_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    def body(_, t):
        g = jax.grad(E_of)(t)
        H = jax.hessian(E_of)(t)
        H = 0.5 * (H + H.T)
        return t - jnp.linalg.solve(H + REG * jnp.eye(NPAR), g)

    return E_of(jax.lax.fori_loop(0, n_steps, body, TH0))


# ------------------------------------------------------------------
# Hessian for each route, then frequencies
# ------------------------------------------------------------------
PLANS = [('A (frozen)', E_geo_A, 0),
         ('frozen expansion N=2', lambda m_: E_frozen(m_, 2), 0),
         ('frozen expansion N=3', lambda m_: E_frozen(m_, 3), 0),
         ('full Newton N=3', lambda m_: E_full(m_, 3), 0)]

RESULT = dict(system=SYS, basis=BASIS, nthread=NT, natm=natm, npar=NPAR,
              masses=masses.tolist(), n_vib=NDOF - NTR,
              exp=EXP.get(SYS, []), grad_theta=GN)

print('\n' + '=' * 100, flush=True)
print('Vibrational frequencies', flush=True)
print('=' * 100, flush=True)

for tag, fn, _ in PLANS:
    try:
        t0 = time.time()
        H = jax.hessian(fn)(mol)
        Hs = np.asarray(H.coords.coords).reshape(NDOF, NDOF)
        dt = time.time() - t0
        nu, imag, zeros = freqs_from_hess(Hs, tag)
        print(f'\n  {tag}   ({dt:.0f}s)', flush=True)
        print(f'    frequencies [cm^-1] = {[f"{x:.1f}" for x in nu]}', flush=True)
        print(f'    imaginary modes = {imag}   zero modes |lambda| = '
              f'{[f"{abs(z):.1e}" for z in zeros]}', flush=True)
        RESULT[tag] = dict(freqs=[float(x) for x in nu], n_imag=imag,
                           time=dt)
    except Exception as ex:
        print(f'\n  {tag}: ✗ {type(ex).__name__}: {str(ex)[:150]}', flush=True)
    gc.collect()

if SYS in EXP:
    print(f'\n  experiment (fundamentals, for reference)= {EXP[SYS]}', flush=True)
    print('  Note: this calculation is harmonic, at the CAS(2,2)/6-31G level, '
          'so a systematic overestimate is expected', flush=True)

outp = f'results/t153_freq_{SYS}.json'
json.dump(RESULT, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
          default=str, indent=1)
print(f'\n>>> {outp}', flush=True)
print(f'\nt153_{SYS}_DONE', flush=True)
gc.collect()
