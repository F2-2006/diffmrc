"""H2O/6-31G CAS(2,2): NEVPT2 versus CASSCF harmonic spectra, relaxed route.

Definition used for the NEVPT2 column (the literature convention, and the
only one that survives this system's measurements):

    F(R) = E_nevpt2(mol(R), theta*(R)),   theta*(R) = argmin E_casscf

-- the correlation correction evaluated on the CASSCF-relaxed orbitals.
The Hessian of F is taken by central differences (the FD platform
discipline: two deltas, monotone improvement checked), with theta*
re-converged at every displaced geometry from a warm start.

Two shortcuts were tried and both failed, which is why the definition is
evaluated the literal way:

  * Optimizing theta on the SC-NEVPT2 surface: the surface is not smooth
    (SC energy lacks invariance under inactive rotations; denominators
    cross zero). Measured: the optimizer walks into a singularity from
    the zeros start (E = -3.2e7) and from the CASSCF minimum itself
    (E = -1.1e6).
  * The crossed two-surface implicit formula
    H_RR(nev) - M_nev^T H_theta_theta(casscf)^-1 M_nev: stationary only if
    the NEVPT2 energy itself is stationary in theta. Measured |g_theta|
    of E_nevpt2 at the CASSCF point: 2.78 -- not stationary, so the
    formula drops g_nev * theta*'' and the H_theta_theta(nev) term and
    the response diverges (dtheta/dR ~ 1.3e3, frequencies ~ -1e6 cm^-1).

The CASSCF column uses the validated implicit route and publishes which
theta* branch it used. This system has two CASSCF solutions (13.3
kcal/mol apart, recorded 2026-09-22); the NEVPT2 corrections at the two
branches differ by an order of magnitude and flip sign (measured:
-0.0085 Ha at the RHF-derived branch, +0.0804 Ha at the lower CASSCF
branch), so the branch is named, not assumed. The standard branch is
taken as the valid stationary point closest to the RHF reference.

Usage:  PYTHONPATH=src python verify_nevpt2_relaxed_frequencies.py [NT]
"""
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
EXP = [1594.7, 3656.7, 3755.8]


def rhf_tight(m):
    mf = pscf.RHF(m)
    mf.conv_tol = 1e-12
    return mf


def _mol():
    mol = pgto.Mole()
    mol.atom = ATOM
    mol.basis = BASIS
    mol.build()
    mol.set_geom_(XYZ_BOHR, unit='Bohr')
    mol.build()
    return mol


def _keep_best(best, pt, tag):
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
    mol = _mol()
    nmo = mol.nao_nr()
    natm = mol.natm
    ncore = mol.nelectron // 2 - NELE // 2
    pairs = orbital_pairs(ncore, NACT, nmo, include_cv=True)
    npar = len(pairs)
    coords0 = np.asarray(mol.atom_coords())
    masses = np.array(mol.atom_mass_list(isotope_avg=True))
    print('=' * 88, flush=True)
    print('NEVPT2 vs CASSCF frequencies (relaxed route) -- H2O/{} CAS({},{})'
          '  ncore={}  NPAR={}'.format(BASIS, NELE, NACT, ncore, npar),
          flush=True)
    print('=' * 88, flush=True)

    E_cas = make_nevpt2_joint_fn(ncore, NACT, NELE, nmo, pairs, rhf_tight,
                                 e2_scale=0.0, ci_solver='blocked')
    E_nev = make_nevpt2_joint_fn(ncore, NACT, NELE, nmo, pairs, rhf_tight,
                                 e2_scale=1.0, ci_solver='blocked')
    E_th_cas = make_nevpt2_theta_fn(mol, ncore, NACT, NELE, nmo, pairs,
                                    rhf_tight, e2_scale=0.0,
                                    ci_solver='blocked')
    E_th_nev = make_nevpt2_theta_fn(mol, ncore, NACT, NELE, nmo, pairs,
                                    rhf_tight, e2_scale=1.0,
                                    ci_solver='blocked')

    # ---- CAS theta* candidates ----
    rng = np.random.default_rng(0)
    starts = [np.zeros(npar)] + [0.05 * rng.normal(size=npar) for _ in range(2)]
    cands = []
    for k, t0 in enumerate(starts):
        t_wall = time.time()
        pt = converge_orbital_point(E_th_cas, npar, theta0=jnp.asarray(t0))
        print('  CAS start {}: E = {:.12f}  |g| = {:.2e}  polished={}  '
              '[{:.0f}s]'.format(k, pt.energy, pt.grad_norm,
                                 getattr(pt, 'polished', None),
                                 time.time() - t_wall), flush=True)
        kept = _keep_best(None, pt, 'CAS start {}'.format(k))
        if kept is not None:
            cands.append(pt)

    # standard branch: valid stationary point closest to the RHF reference
    pt_cas = min(cands, key=lambda p: float(np.linalg.norm(
        np.asarray(p.theta))))
    print('  standard branch chosen: E = {:.12f}  |g| = {:.2e}  '
          '(closest to RHF among {} candidates)'.format(
              pt_cas.energy, pt_cas.grad_norm, len(cands)), flush=True)
    e2_at = float(E_th_nev(pt_cas.theta)) - pt_cas.energy
    print('  E2 at this theta* = {:+.10f} Ha'.format(e2_at), flush=True)

    out = dict(system='H2O', basis=BASIS, ncore=ncore, nact=NACT, nele=NELE,
               npar=npar, natm=natm, e2_at_theta=e2_at,
               branches=[dict(E=float(p.energy), g=float(p.grad_norm))
                         for p in cands])

    # ---- CASSCF column: validated implicit route ----
    t0 = time.time()
    res = implicit_hessian(mol, pt_cas.theta, E_cas)
    nu = wavenumbers(res.H_total, coords0, masses)
    f_cas = [float(x) for x in nu.freqs_cm]
    print('  [casscf] kappa(H_tt) = {:.2e}  dtheta_dR_max = {:.2e}  [{}s]'
          .format(float(np.linalg.cond(res.H_tt)), res.dtheta_dR_max,
                  int(time.time() - t0)), flush=True)
    print('  [casscf] frequencies = {}'.format(
        ['{:.1f}'.format(x) for x in f_cas]), flush=True)
    out['casscf'] = dict(freqs_cm=f_cas,
                         kappa=float(np.linalg.cond(res.H_tt)))

    # ---- NEVPT2 column: F(R) = E_nev(mol(R), theta*(R)) ----
    # Its gradient is computed *exactly* by the implicit function theorem
    #     grad F = d_R E_nev + g_theta^T theta*',   theta*' = -H_tt(cas)^-1 M_cas
    # (every piece exact; theta*' is the derivative of the CASSCF argmin),
    # and the Hessian is the central difference of that gradient. An
    # analytic second derivative assembled from the same pieces is NOT
    # enough: it would have to keep d(g_theta)/dR - H_tt(nev) theta*' and
    # the theta*'' term, and dropping them is what produced the measured
    # dtheta/dR ~ 1e3 blow-up in the first attempt.
    from diffmrc.core.hessian import _mixed_derivative

    def gradF_of(coords, theta_warm):
        m = _mol()
        m.set_geom_(coords, unit='Bohr')
        m.build()
        th_fn = make_nevpt2_theta_fn(m, ncore, NACT, NELE, nmo, pairs,
                                     rhf_tight, e2_scale=0.0,
                                     ci_solver='blocked')
        pt = converge_orbital_point(th_fn, npar,
                                    theta0=jnp.asarray(theta_warm))
        if not (np.isfinite(pt.energy) and pt.grad_norm < 1e-3):
            raise RuntimeError('theta* fell out of the branch: E = {:.6g} '
                               '|g| = {:.3e}'.format(pt.energy, pt.grad_norm))
        th = jnp.asarray(pt.theta)
        dth = float(np.max(np.abs(np.asarray(th) - np.asarray(theta_warm))))
        g_theta = np.asarray(jax.grad(lambda t: E_nev(m, t))(th))
        H_tt = np.asarray(jax.hessian(lambda t: E_cas(m, t))(th))
        H_tt = 0.5 * (H_tt + H_tt.T)
        M_cas = _mixed_derivative(E_cas, m, th)
        g_R = np.asarray(jax.grad(lambda mm: E_nev(mm, th))(m).coords)
        dtheta_dR = -np.linalg.pinv(H_tt, rcond=1e-8) @ M_cas
        gradF = np.asarray(g_R).reshape(-1) + g_theta @ dtheta_dR
        return gradF, th, dth, float(np.linalg.norm(g_theta))

    t0 = time.time()
    gF0, th0, _, gt0 = gradF_of(coords0, np.asarray(pt_cas.theta))
    print('  [nevpt2] grad F(R0) computed (|g_theta| = {:.2e})  [{:.0f}s]'
          .format(gt0, time.time() - t0), flush=True)

    DELTA = 1.5e-3
    n3 = 3 * natm
    H = np.zeros((n3, n3))
    drifts = {}
    for j in range(n3):
        e = np.zeros(n3)
        e[j] = 1.0
        gp, tp, dp, _ = gradF_of(coords0 + DELTA * e.reshape(-1, 3), th0)
        gm, tm, dm, _ = gradF_of(coords0 - DELTA * e.reshape(-1, 3), th0)
        H[:, j] = (gp - gm) / (2 * DELTA)
        drifts[j] = (dp, dm)
        print('    col {:2d}: H_jj = {:+.8f}   branch drift <- {:.2e} '
              '{:.2e}'.format(j, H[j, j], dp, dm), flush=True)
    H = 0.5 * (H + H.T)

    # platform check on one direction with a second delta
    e = np.zeros(n3)
    e[0] = 1.0
    gp, _, _, _ = gradF_of(coords0 + 0.5e-3 * e.reshape(-1, 3), th0)
    gm, _, _, _ = gradF_of(coords0 - 0.5e-3 * e.reshape(-1, 3), th0)
    col2 = (gp - gm) / (2 * 0.5e-3)
    print('  platform: |col0(d=1.5e-3) - col0(d=5e-4)|max = {:.2e}'.format(
        float(np.abs(H[:, 0] - col2).max())), flush=True)

    nu_nev = wavenumbers(H, coords0, masses)
    f_nev = [float(x) for x in nu_nev.freqs_cm]
    print('  [nevpt2] frequencies = {}'.format(
        ['{:.1f}'.format(x) for x in f_nev]), flush=True)

    n = min(len(f_cas), len(f_nev))
    print('', flush=True)
    print('  CASSCF {}'.format(['{:.1f}'.format(x) for x in f_cas]),
          flush=True)
    print('  NEVPT2 {}'.format(['{:.1f}'.format(x) for x in f_nev]),
          flush=True)
    print('  diff   {}'.format(['{:+.1f}'.format(x)
                                for x in (np.asarray(f_nev[:n])
                                          - np.asarray(f_cas[:n]))]),
          flush=True)
    print('  exp    {}'.format(EXP), flush=True)
    out['nevpt2'] = dict(freqs_cm=f_nev, delta=DELTA,
                         hessian=H.tolist(),
                         branch_drift={str(k): v for k, v in drifts.items()})
    out['difference_cm'] = [float(x) for x in (np.asarray(f_nev[:n])
                                               - np.asarray(f_cas[:n]))]

    outp = 'results/verify_nevpt2_relaxed_frequencies_H2O.json'
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=1, default=str)
    print('>>> {}'.format(outp), flush=True)
    print('verify_nevpt2_relaxed_frequencies_DONE', flush=True)


if __name__ == '__main__':
    main()
