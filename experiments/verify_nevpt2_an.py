"""Actinide validation of the differentiable NEVPT2 chain (Phase 0, Task 8).

    tho_ecp   ThO, Stuttgart ECP on Th + 6-31G on O, RHF, frontier active space
    uf_ecp    UF,  Stuttgart ECP on U + 6-31G on F, ROHF (spin from parity)

THE BOUNDARY THAT RESHAPED THIS SCRIPT (measured 2026-09-23, guarded probe
``code/t201_guarded_grad_stage.py``): **nuclear gradients on actinide-sized
systems are blocked upstream, not here.** Differentiating any energy with
respect to the geometry needs the geometry derivative of the AO integrals,
and pyscfad materializes it densely:

    pyscfad/gto/_moleintor_jvp.py:722  _fill_grad0
    grad = numpy.zeros((comp, len(ctr_coeff), nao, nao, nao, nao))
    -> MemoryError: 51.3 GiB for (1, 81, 96, 96, 96, 96) at nao = 96

One array. The unguarded run peaked past 200 GB and took the machine's
memory with it. Scaling is nctr * nao^4, so no ligand-basis choice fixes
it; every previously validated gradient in this project was nao <= 30,
where the same array fits trivially.

What that leaves, and what this script therefore does: theta derivatives
do NOT need the integral geometry derivative. The chain's automatic
differentiation is validated here at actinide scale in theta space:

  [1] energy cross-check against PySCF's own CASCI + SC-NEVPT2 on the
      same ECP molecule (the Task-3 anchor is unusable on ECP routes:
      ``core.hessian.make_joint_energy_fn`` hard-codes
      ``int1e_kin + int1e_nuc`` and silently drops the ECP)
  [2] theta-gradient finiteness at theta = 0
  [3] a theta-space Hessian-vector product against a finite-difference
      ladder in theta (two deltas; monotone improvement is the check)
  [4] the nuclear-gradient attempt itself, recorded as blocked with the
      verbatim allocation error -- attempted, not skipped silently

Boundary for every quotation of these numbers: scalar-relativistic only;
SOC out of scope (roadmap P4); and the nuclear-derivative layer of the
actinide protocol awaits a non-dense integral-derivative path.

Usage:  PYTHONPATH=src python experiments/verify_nevpt2_an.py <case> [NT]
"""
import json
import os
import sys
import time

CASE = sys.argv[1] if len(sys.argv) > 1 else 'tho_ecp'
NT = sys.argv[2] if len(sys.argv) > 2 else '4'
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from pyscfad import gto as pgto, scf as pscf  # noqa: E402

from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,  # noqa: E402
                                   make_nevpt2_theta_fn)

CASES = {
    'tho_ecp': dict(atom='Th 0 0 0; O 0 0 1.85',
                    basis={'Th': 'stuttgart_rsc', 'O': '6-31g'},
                    ecp={'Th': 'stuttgart_rsc'}, spin=0),
    'uf_ecp': dict(atom='U 0 0 0; F 0 0 2.03',
                   basis={'U': 'stuttgart_rsc', 'F': '6-31g'},
                   ecp={'U': 'stuttgart_rsc'}, spin=None),
}


def build(case):
    cfg = CASES[case]
    mol = pgto.Mole()
    mol.atom = cfg['atom']
    mol.basis = cfg['basis']
    if cfg['ecp']:
        mol.ecp = cfg['ecp']
    if cfg['spin'] is not None:
        mol.spin = cfg['spin']
        mol.build()
        return mol, cfg
    # parity fixes the spin; the count is only known after a build, and a
    # wrong parity makes build raise -- flip on the consistency error.
    mol.spin = 0
    try:
        mol.build()
    except RuntimeError:
        mol.spin = 1
        mol.build()
    print('  spin from parity: nelec={} -> spin={}'.format(
        mol.nelectron, mol.spin), flush=True)
    return mol, cfg


def frontier_space(mf):
    occ = np.asarray(mf.mo_occ)
    nocc = int((occ > 0).sum())
    # keep the four highest occupied active: on the ThO ladder a
    # three-orbital window splits a degenerate core pair (mo 15/16 at
    # -0.484128), the exact inactive degeneracy SC-NEVPT2 lacks
    # invariance against.
    ncore = nocc - 4
    nact = 8
    nele = int(round(occ[:ncore + nact].sum() - 2 * ncore))
    return ncore, nact, nele


def main():
    t_start = time.time()
    mol, cfg = build(CASE)
    nmo = mol.nao_nr()
    print('=' * 86, flush=True)
    print('NEVPT2 actinide validation -- {}  nao={}  nelec={}  spin={}'.format(
        CASE, nmo, mol.nelectron, mol.spin), flush=True)
    print('=' * 86, flush=True)

    mf0 = pscf.RHF(mol) if mol.nelectron % 2 == 0 else pscf.ROHF(mol)
    mf0.init_guess = 'atom'
    mf0.max_cycle = 200
    mf0.conv_tol = 1e-9
    mf0.kernel()
    print('  SCF E = {:.10f}  conv={}  [{:.0f}s]'.format(
        float(mf0.e_tot), bool(mf0.converged), time.time() - t_start),
        flush=True)

    ncore, nact, nele = frontier_space(mf0)
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    print('  frontier CAS({},{})  ncore={}  NPAR={}'.format(
        nele, nact, ncore, len(pairs)), flush=True)

    def builder(m):
        mf = pscf.RHF(m) if m.nelectron % 2 == 0 else pscf.ROHF(m)
        mf.init_guess = 'atom'
        mf.max_cycle = 200
        mf.conv_tol = 1e-9
        return mf

    fn = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=1.0, ci_solver='blocked')
    fn0 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                               e2_scale=0.0, ci_solver='blocked')
    theta = jnp.zeros(len(pairs))

    out = dict(case=CASE, nao=nmo, nelec=mol.nelectron, spin=int(mol.spin),
               ncore=ncore, nact=nact, nele=nele, npair=len(pairs),
               scf=float(mf0.e_tot))

    # ---- [1] energy cross-check vs PySCF CASCI + SC-NEVPT2 (same ECP) ----
    t0 = time.time()
    try:
        from pyscf import gto as gto2, scf as scf2, mcscf, mrpt
        m2 = gto2.M(atom=cfg['atom'], basis=cfg['basis'],
                    ecp=cfg['ecp'], spin=int(mol.spin), verbose=0)
        mf2 = (scf2.RHF(m2) if m2.nelectron % 2 == 0 else scf2.ROHF(m2))
        mf2.kernel()
        mc = mcscf.CASCI(mf2, nact, nele)
        mc.ncore = ncore
        mc.verbose = 0
        mc.kernel()
        e2_ref = float(mrpt.NEVPT(mc).kernel())
        E_ours, e2_ours = fn.components(mol, theta)
        E_ours = float(E_ours)
        e2_ours = float(e2_ours)
        dE = abs((E_ours + e2_ours) - (float(mc.e_tot) + e2_ref))
        ok1 = dE < 1e-6
        print('  [1] E(ours)  = {:.10f}  e2 = {:.10f}'.format(
            E_ours + e2_ours, e2_ours), flush=True)
        print('  [1] E(pyscf) = {:.10f}  e2 = {:.10f}'.format(
            float(mc.e_tot) + e2_ref, e2_ref), flush=True)
        print('  [1] |dE| = {:.3e} (thr 1e-6)  {}  [{:.0f}s]'.format(
            dE, 'PASS' if ok1 else 'FAIL', time.time() - t0), flush=True)
        out['crosscheck'] = dict(E_ours=E_ours, e2_ours=e2_ours,
                                 E_pyscf=float(mc.e_tot) + e2_ref,
                                 e2_pyscf=e2_ref, dE=dE, ok=bool(ok1))
    except Exception as ex:  # noqa: BLE001 -- record, keep going
        print('  [1] EXC {}: {}'.format(type(ex).__name__, str(ex)[:100]),
              flush=True)
        out['crosscheck'] = dict(error=repr(ex)[:200])

    # ---- [2] theta gradient at theta = 0 (frozen-geometry form) ----
    t0 = time.time()
    th_fn = make_nevpt2_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                 builder, e2_scale=1.0,
                                 ci_solver='blocked')
    g_th = np.asarray(jax.grad(th_fn)(theta))
    ok2 = bool(np.isfinite(g_th).all())
    print('  [2] theta-gradient finite = {}  |g|max = {:.3e}  [{:.0f}s]'.format(
        ok2, float(np.abs(g_th).max()), time.time() - t0), flush=True)
    out['theta_grad'] = dict(ok=ok2, gmax=float(np.abs(g_th).max()),
                             norm=float(np.linalg.norm(g_th)))

    # ---- [3] theta-space HVP vs FD ladder ----
    t0 = time.time()
    npar = g_th.size
    rng = np.random.default_rng(0)
    v = rng.normal(size=npar)
    v = v / np.linalg.norm(v)
    g_fn = jax.grad(th_fn)
    _, hv = jax.jvp(g_fn, (theta,), (jnp.asarray(v),))
    hv = np.asarray(hv)
    errs = {}
    for d in (3e-3, 1e-3):
        gp = np.asarray(g_fn(jnp.asarray(d * v)))
        gm = np.asarray(g_fn(jnp.asarray(-d * v)))
        errs[d] = float(np.abs((gp - gm) / (2 * d) - hv).max())
    ok3 = bool(np.isfinite(hv).all()) and errs[1e-3] < errs[3e-3]
    print('  [3] theta-HVP finite = {}   |FD-HVP| d=3e-3: {:.3e}  '
          'd=1e-3: {:.3e}  单调 {}  [{:.0f}s]'.format(
              bool(np.isfinite(hv).all()), errs[3e-3], errs[1e-3], ok3,
              time.time() - t0), flush=True)
    out['theta_hvp'] = dict(finite=bool(np.isfinite(hv).all()),
                            err_3e3=errs[3e-3], err_1e3=errs[1e-3],
                            ok=ok3)

    # ---- [4] nuclear gradient: attempt, record the upstream block ----
    from diffmrc.driver.nevpt2 import nevpt2_nuclear_gradient
    try:
        g = np.asarray(nevpt2_nuclear_gradient(mol, lambda m_: fn(m_, theta)))
        out['nuclear_grad'] = dict(ok=True,
                                   sum_g=float(np.abs(g.sum(axis=0)).max()))
        print('  [4] nuclear gradient unexpectedly fits: |sum g| = {:.3e}'.format(
            out['nuclear_grad']['sum_g']), flush=True)
    except Exception as ex:
        out['nuclear_grad'] = dict(blocked=True,
                                   error='{}: {}'.format(
                                       type(ex).__name__, str(ex)[:180]))
        print('  [4] nuclear gradient BLOCKED upstream: {}: {}'.format(
            type(ex).__name__, str(ex)[:110]), flush=True)

    outp = 'results/verify_nevpt2_an_{}.json'.format(CASE)
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=1, default=str)
    print('>>> {}   total {:.0f}s'.format(outp, time.time() - t_start),
          flush=True)
    print('verify_nevpt2_an_{}_DONE'.format(CASE), flush=True)


if __name__ == '__main__':
    main()
