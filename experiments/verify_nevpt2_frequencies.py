"""H2O/6-31G CAS(2,2): NEVPT2 versus CASSCF harmonic spectra.

The deliverable is the pair of spectra and their difference -- not agreement
with either. The correlation correction is expected to lower the stretching
modes by tens of cm^-1; the difference is the result.

Both columns use the same relaxed recipe, reusing validated machinery:
converge theta on the surface's own joint energy (driver.orbitals.
converge_orbital_point, multi-start, lowest kept), then assemble the relaxed
Hessian with the implicit route (core.hessian.implicit_hessian). A
frozen-theta Hessian is not the frequency of the method -- it omits the
orbital response -- so it is not the headline here; the HVP machinery itself
is validated separately (code/t188_nevpt2_task4_hvp.py).

Usage:  PYTHONPATH=src python experiments/verify_nevpt2_frequencies.py [NT]
"""
import gc
import json
import os
import sys
import time

NT = sys.argv[1] if len(sys.argv) > 1 else '4'
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

from diffmrc.analysis.vibrations import wavenumbers  # noqa: E402
from diffmrc.core.hessian import implicit_hessian  # noqa: E402
from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,  # noqa: E402
                                   make_nevpt2_theta_fn)
from diffmrc.driver.orbitals import converge_orbital_point  # noqa: E402

ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
XYZ_BOHR = np.asarray([[0.0, 0.0, 0.0],
                       [1.4283, 0.0, 1.1115],
                       [-1.4283, 0.0, 1.1115]])
BASIS = '6-31g'
NACT, NELE = 2, 2
EXP = [1594.7, 3656.7, 3755.8]        # H2O fundamentals, cm^-1, for reference


def rhf_tight(m):
    mf = pscf.RHF(m)
    mf.conv_tol = 1e-12
    return mf


def _keep_best(best, pt, tag):
    """A runaway is not a candidate.

    The theta optimizer can walk into a NEVPT2 denominator singularity --
    measured on this system from the zeros start: E = -3.2e7,
    |g| = 3.6e14. Ranking by energy alone would then select it, which is
    this project's recorded failure mode ("energy monotonicity is a
    guard, not a criterion"). A candidate must additionally be finite
    and stationary to a sane tolerance.
    """
    ok = (np.isfinite(pt.energy) and np.isfinite(pt.grad_norm)
          and pt.grad_norm < 1e-3)
    if not ok:
        print('    ({}) rejected: E = {:.6g}  |g| = {:.3e}'.format(
            tag, pt.energy, pt.grad_norm), flush=True)
        return best
    if best is None or pt.energy < best.energy:
        return pt
    return best


def main():
    mol = pgto.Mole()
    mol.atom = ATOM
    mol.basis = BASIS
    mol.build()
    mol.set_geom_(XYZ_BOHR, unit='Bohr')
    mol.build()
    nmo = mol.nao_nr()
    natm = mol.natm
    ncore = mol.nelectron // 2 - NELE // 2
    pairs = orbital_pairs(ncore, NACT, nmo, include_cv=True)
    npar = len(pairs)
    coords = np.asarray(mol.atom_coords())
    masses = np.array(mol.atom_mass_list(isotope_avg=True))
    print('=' * 90, flush=True)
    print('NEVPT2 vs CASSCF frequencies -- H2O/{}  CAS({},{})  ncore={}  '
          'NPAR={}  T={}'.format(BASIS, NELE, NACT, ncore, npar, NT),
          flush=True)
    print('=' * 90, flush=True)

    E_cas = make_nevpt2_joint_fn(ncore, NACT, NELE, nmo, pairs, rhf_tight,
                                 e2_scale=0.0, ci_solver='blocked')
    E_nev = make_nevpt2_joint_fn(ncore, NACT, NELE, nmo, pairs, rhf_tight,
                                 e2_scale=1.0, ci_solver='blocked')
    # The theta optimizer wraps its objective in jax.jit, which cannot
    # trace the SCF in this pyscfad build (probe t190); the frozen-geometry
    # form is the same function at a fixed geometry. The joint functions
    # stay for the implicit Hessian, which never jits. All four carry
    # ci_solver='blocked': this stage takes second derivatives (the theta
    # refinement's Newton step, the implicit H_tt and mixed terms), and the
    # plain eigensolver returns NaN at the cross-S_z degeneracy of this
    # CI matrix (probe t193: NaN for plain, finite for blocked).
    E_th_cas = make_nevpt2_theta_fn(mol, ncore, NACT, NELE, nmo, pairs,
                                    rhf_tight, e2_scale=0.0,
                                    ci_solver='blocked')
    E_th_nev = make_nevpt2_theta_fn(mol, ncore, NACT, NELE, nmo, pairs,
                                    rhf_tight, e2_scale=1.0,
                                    ci_solver='blocked')

    # ---- theta* on the CASSCF surface (multi-start; the low basin of this
    #      system is reachable only from perturbed starts, see the theta-space
    #      double-minimum record) ----
    rng = np.random.default_rng(0)
    starts = [np.zeros(npar)] + [0.05 * rng.normal(size=npar) for _ in range(2)]
    pt_cas = None
    for k, t_start in enumerate(starts):
        t_wall = time.time()
        pt = converge_orbital_point(E_th_cas, npar, theta0=jnp.asarray(t_start))
        print('  CAS   start {}: E = {:.12f}  |g| = {:.2e}  polished={}  '
              '[{:.0f}s]'.format(k, pt.energy, pt.grad_norm,
                                 getattr(pt, 'polished', None),
                                 time.time() - t_wall), flush=True)
        pt_cas = _keep_best(pt_cas, pt, 'CAS start {}'.format(k))

    # ---- theta* on the NEVPT2 surface (warm start from the CASSCF point) ----
    # The zeros start is deliberately not among the candidates: on the
    # NEVPT2 surface it walks into a denominator singularity (measured
    # 2026-09-23: E = -3.2e7, |g| = 3.6e14 within 275 s). A perturbed warm
    # start probes the neighbouring basin instead.
    pt_nev = None
    rng2 = np.random.default_rng(1)
    nev_starts = [pt_cas.theta,
                  pt_cas.theta + 0.01 * jnp.asarray(rng2.normal(size=npar))]
    for k, t0 in enumerate(nev_starts):
        t_start = time.time()
        pt = converge_orbital_point(E_th_nev, npar, theta0=jnp.asarray(t0))
        print('  NEV   start {}: E = {:.12f}  |g| = {:.2e}  polished={}  '
              '[{:.0f}s]'.format(k, pt.energy, pt.grad_norm,
                                 getattr(pt, 'polished', None),
                                 time.time() - t_start), flush=True)
        pt_nev = _keep_best(pt_nev, pt, 'NEV start {}'.format(k))

    e2_at_cas_theta = float(E_th_nev(pt_cas.theta)) - pt_cas.energy
    print('  E2 (NEVPT2 correlation at the CASSCF point) = '
          '{:+.10f} Ha'.format(e2_at_cas_theta), flush=True)

    # ---- relaxed Hessians through the validated implicit route ----
    out = dict(system='H2O', basis=BASIS, ncore=ncore, nact=NACT, nele=NELE,
               npar=npar, natm=natm)
    for tag, E_of, pt in (('casscf', E_cas, pt_cas),
                          ('nevpt2', E_nev, pt_nev)):
        t0 = time.time()
        res = implicit_hessian(mol, pt.theta, E_of)
        kappa = float(np.linalg.cond(res.H_tt))
        nu = wavenumbers(res.H_total, coords, masses)
        freqs = [float(x) for x in nu.freqs_cm]
        zeros = [float(z) for z in nu.zero_mode_magnitudes]
        print('  [{}] relaxed Hessian: |g_theta| = {:.2e}  kappa(H_tt) = '
              '{:.2e}  dtheta_dR_max = {:.2e}  [{:.0f}s]'.format(
                  tag, pt.grad_norm, kappa, res.dtheta_dR_max,
                  time.time() - t0), flush=True)
        print('  [{}] zero modes |lambda| = {}'.format(
            tag, ['{:.1e}'.format(z) for z in zeros]), flush=True)
        print('  [{}] frequencies [cm^-1] = {}'.format(
            tag, ['{:.1f}'.format(x) for x in freqs]), flush=True)
        out[tag] = dict(theta=[float(x) for x in np.asarray(pt.theta)],
                        grad_norm=float(pt.grad_norm), kappa_Htt=kappa,
                        freqs_cm=freqs, zero_modes=zeros,
                        E=float(pt.energy))
        gc.collect()

    f_cas = np.asarray(out['casscf']['freqs_cm'])
    f_nev = np.asarray(out['nevpt2']['freqs_cm'])
    n = min(f_cas.size, f_nev.size)
    print('', flush=True)
    print('  pair (relaxed):  CASSCF    {}'.format(
        ['{:.1f}'.format(x) for x in f_cas]), flush=True)
    print('                   NEVPT2    {}'.format(
        ['{:.1f}'.format(x) for x in f_nev]), flush=True)
    print('                   difference {}'.format(
        ['{:+.1f}'.format(x) for x in (f_nev[:n] - f_cas[:n])]), flush=True)
    print('                   experiment {}'.format(EXP), flush=True)
    out['difference_cm'] = [float(x) for x in (f_nev[:n] - f_cas[:n])]

    outp = 'results/verify_nevpt2_frequencies_H2O.json'
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=1, default=str)
    print('', flush=True)
    print('>>> {}'.format(outp), flush=True)
    print('verify_nevpt2_frequencies_DONE', flush=True)


if __name__ == '__main__':
    main()
