"""Orbital-response term by unrolling the orbital optimization into the graph.

The orbital optimization is expanded as a fixed number of Newton steps inside
the computational graph, so that back-propagation through the chain accumulates
the response automatically. Three details are required: the self-consistent
field step that supplies the reference orbitals must be evaluated outside the
iteration construct, since it cannot be traced; the orbital Hessian requires
regularization because it has negative eigenvalues at the reference geometry;
and the iteration must start from the converged parameters.

Usage: python t151_unrolled_theta.py [nthread]
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
from diffmrc.core.casci import cs_casci_blocked                       # noqa: E402

ATOM_STR = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
BASIS = '6-31g'
NCORE, NACT, NELE = 4, 2, 2
XYZ0 = np.asarray([[0.0, 0.0, 0.0], [1.4283, 0.0, 1.1115],
                   [-1.4283, 0.0, 1.1115]])

print('=' * 100, flush=True)
print('Orbital response by unrolled iteration: theta*(R)', flush=True)
print('=' * 100, flush=True)

mol = pgto.Mole(); mol.atom = ATOM_STR; mol.basis = BASIS; mol.build()
mol.set_geom_(XYZ0, unit='Bohr'); mol.build()
nmo = mol.nao_nr(); natm = mol.natm
pairs = orbital_pairs(NCORE, NACT, nmo, include_cv=True)
NPAR = len(pairs)
print(f'  nmo={nmo} natm={natm} NPAR={NPAR}', flush=True)


def E_cas_of(C, h, e, en):
    nm = NCORE + NACT
    Cj = jnp.asarray(C)
    h1m = jnp.einsum('ai,ab,bj->ij', Cj, h, Cj)[:nm, :nm]
    e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj, e)[:nm, :nm, :nm, :nm]
    return cs_casci_blocked(h1m, e1m, ncore=NCORE, nact=NACT,
                            nele_act=NELE)[0] + en


# ---------- theta* at the reference geometry ----------
mf0 = pscf.RHF(mol); mf0.kernel()
C0_ref = mf0.mo_coeff
h1e0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri0 = mol.intor("int2e", aosym="s1")
enuc0 = mol.energy_nuc()


def E_theta(th):
    return E_cas_of(C0_ref @ block_rotation(th, nmo, pairs), h1e0, eri0, enuc0)


_, th, _ = optimize_theta(mol, C0_ref, E_theta, NPAR, steps=3000,
                          g_conv=1e-13, label='opt')
th = jnp.asarray(np.asarray(th))
for k in range(5):
    gg = jax.grad(E_theta)(th)
    if float(jnp.linalg.norm(gg)) < 1e-12:
        break
    H = jax.hessian(E_theta)(th)
    th = th - jnp.linalg.lstsq(jnp.asarray(0.5*(H+H.T)), gg, rcond=None)[0]
TH0 = jnp.asarray(th)
print(f'  theta_ref: |g| = {float(jnp.linalg.norm(jax.grad(E_theta)(TH0))):.3e}  '
      f'|theta| = {float(jnp.linalg.norm(TH0)):.6f}', flush=True)

# ==================================================================
# unrolled iteration
# ==================================================================
print('\n' + '=' * 100, flush=True)
print('Unrolled iteration: pure-JAX Newton steps from the converged reference', flush=True)
print('=' * 100, flush=True)


def E_and_grad(m_, th_):
    """Return (E, g_theta) for a given (Mole, theta); differentiable throughout.

    The SCF must be evaluated outside the iteration construct, since it cannot
    """
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    en_ = m_.energy_nuc()

    def E_of(t):
        return E_cas_of(C_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    return E_of(th_), jax.grad(E_of)(th_)


def unrolled_theta(m_, th_init, n_steps, damp=1.0, reg=1e-6):
    """Unrolled Newton iteration with lambda regularization, inside the graph.

    KEY POINT: pscf.RHF(m_).kernel() cannot be placed inside lax.fori_loop,
    because the iterative SCF solve triggers TracerArrayConversionError under
    tracing. The SCF is therefore evaluated once, outside the loop, and only
    theta is iterated. In this context C0 does not depend on theta, so that is
    correct: the reference orbitals vary with geometry while theta is held.
    """
    # ---- SCF and integrals evaluated once, outside the loop ----
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    en_ = m_.energy_nuc()

    def E_of(t):
        return E_cas_of(C_ @ block_rotation(t, nmo, pairs), h_, e_, en_)

    def body(_, t):
        g = jax.grad(E_of)(t)
        H = jax.hessian(E_of)(t)
        H = 0.5 * (H + H.T)
        Hr = H + reg * jnp.eye(NPAR)          # lambda regularization
        step = jnp.linalg.solve(Hr, g)
        return t - damp * step

    th_ = jax.lax.fori_loop(0, n_steps, body, th_init)
    return th_


def E_unrolled(m_, n_steps, th_init, damp=1.0, reg=1e-6):
    """Energy after the unrolled iteration."""
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    th_f = unrolled_theta(m_, th_init, n_steps, damp, reg)
    return E_cas_of(C_ @ block_rotation(th_f, nmo, pairs), h_, e_,
                    m_.energy_nuc())


# 1) Check that the expansion reproduces E(theta*) at the REFERENCE geometry
print('\n  1) Self-consistency at the reference geometry: the expansion should stay put', flush=True)
for N in (0, 1, 3, 5, 8):
    e_ref = float(E_theta(TH0))
    e_un = float(E_unrolled(mol, N, TH0))
    print(f'     N={N}: E_unrolled={e_un:.12f}  Δ vs E(theta*)={e_un-e_ref:+.3e}',
          flush=True)

# 2) Compare first derivatives
print('\n  2) First derivative: expansion, frozen orbitals, golden standard', flush=True)
GOLD = None
try:
    t0 = time.time()
    g_un = np.asarray(jax.grad(lambda m_: E_unrolled(m_, 3, TH0))(mol).coords)
    print(f'     unrolled(N=3): {time.time()-t0:.0f}s  '
          f'g = {np.round(g_un, 8).tolist()}', flush=True)
except Exception as ex:
    print(f'     ✗ unrolled grad failed: {type(ex).__name__}: {str(ex)[:150]}',
          flush=True)
    g_un = None

# A routegradient
TH_SG = jax.lax.stop_gradient(TH0)


def E_geo_A(m_):
    mf_ = pscf.RHF(m_); mf_.kernel()
    C_ = mf_.mo_coeff
    h_ = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
    e_ = m_.intor("int2e", aosym="s1")
    return E_cas_of(C_ @ block_rotation(TH_SG, nmo, pairs), h_, e_,
                    m_.energy_nuc())


g_A = np.asarray(jax.grad(E_geo_A)(mol).coords)
print(f'     A (frozen) : g = {np.round(g_A, 8).tolist()}', flush=True)
if g_un is not None:
    print(f'     |unrolled − A|_max = {np.max(np.abs(g_un - g_A)):.3e}',
          flush=True)

# 3) Second derivative (Hessian): the first three diagonal elements only,
print('\n  3) Hessian diagonal: expansion against frozen orbitals', flush=True)
C_OLD = np.array([0.7192834630, 0.0217667449, 0.4701124325])
C_NEW = np.array([0.7192711493, 0.0217924736, 0.4702696863])
H_A_diag = np.asarray(jax.hessian(E_geo_A)(mol).coords.coords).reshape(
    3*natm, 3*natm).diagonal()[:3]
print(f'     A (frozen) : {[f"{x:.10f}" for x in H_A_diag]}', flush=True)
print(f'     C_lbfgs    : {[f"{x:.10f}" for x in C_OLD]}', flush=True)
print(f'     C_newton   : {[f"{x:.10f}" for x in C_NEW]}', flush=True)

for N in (3, 5):
    try:
        t0 = time.time()
        H_un = np.asarray(jax.hessian(
            lambda m_: E_unrolled(m_, N, TH0))(mol).coords.coords)
        H_un = H_un.reshape(3*natm, 3*natm)
        print(f'     unrolled(N={N}): {[f"{x:.10f}" for x in H_un.diagonal()[:3]]}'
              f'   {time.time()-t0:.0f}s', flush=True)
        rel = [abs(H_un[i, i]-C_NEW[i])/abs(C_NEW[i]) for i in range(3)]
        relA = [abs(H_A_diag[i]-C_NEW[i])/abs(C_NEW[i]) for i in range(3)]
        print(f'        relative deviation from the Newton-refined reference: {[f"{x:.3e}" for x in rel]}  '
              f'largest={max(rel):.3e}', flush=True)
        print(f'        frozen-orbital deviation, for comparison:     {[f"{x:.3e}" for x in relA]}  '
              f'largest={max(relA):.3e}', flush=True)
    except Exception as ex:
        print(f'     ✗ unrolled(N={N}) Hessian failed: '
              f'{type(ex).__name__}: {str(ex)[:150]}', flush=True)

print('\nt151_DONE', flush=True)
gc.collect()
