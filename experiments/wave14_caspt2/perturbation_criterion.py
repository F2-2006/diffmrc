"""Measure the perturbation-layer convergence criterion on real systems.

The criterion lives in ``diffmrc.core.perturbation_convergence``.  This
script supplies the data it is meant to consume and tests it on known good
and known bad points:

  systems   H2O/6-31G CAS(2,2), CH2/6-31G CAS(2,2) singlet,
            OH/6-31G CAS(3,2) and OH/6-31G CAS(5,4)
  points    several independent theta starts per system, refined to a
            stationary point and (where useful) Newton-refined with the
            value-faithful Hessian
  local     g = dE/dt, s = de2/dt, H = d2E/dt2, the eigen-split, the
  data      first-order bound, the finite probe along flat directions
  branches  energy, correction, active-space overlap, wavefunction
            overlap and theta distance between every pair of points

Stages are selected on the command line so a slow system can be run alone:

    python perturbation_criterion.py h2o out.json [stages]

with stages a comma list over {starts, ladder, repeat, nullprobe}.  The
JSON is rewritten after every stage, so an interrupted run keeps what it
has already measured.

The value-faithful Hessian is a central difference of the analytic
gradient.  An AD Hessian through the eigensolver carries frozen-rule
semantics and is not the Hessian of the value; ``experiments/
wave14_caspt2/nevpt2_convergence.py`` documents the same choice.
"""
import json
import os
import sys
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from pyscf import gto as pgto, scf as pscf

from diffmrc.core.perturbation_convergence import (
    DEFAULT_E2_TOLERANCE,
    assess_perturbation_convergence,
    branch_spread,
    split_spectrum,
    BranchPoint,
)
from diffmrc.core.orblib import orbital_pairs
from diffmrc.driver.nevpt2 import _casscf_solution, make_nevpt2_theta_fn

SYSTEMS = {
    'h2o': dict(atom='O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
                ncore=4, nact=2, nele=2, basis='6-31g', mult=1,
                starts=[('zeros', None, 0.0), ('r7_s0.05', 7, 0.05),
                        ('r11_s0.20', 11, 0.20), ('r23_s0.50', 23, 0.50)],
                ladder=True, repeat=True, soft_probe=2, refine=True),
    'ch2': dict(atom='C 0 0 0; H 0.865 0 0.695; H -0.865 0 0.695',
                ncore=2, nact=2, nele=2, basis='6-31g', mult=1,
                starts=[('zeros', None, 0.0), ('r7_s0.05', 7, 0.05),
                        ('r11_s0.20', 11, 0.20)],
                ladder=True, repeat=True, soft_probe=2, refine=True),
    'oh32': dict(atom='O 0 0 0; H 0 0 0.9697',
                 ncore=3, nact=2, nele=3, basis='6-31g', mult=2,
                 starts=[('zeros', None, 0.0), ('r7_s0.05', 7, 0.05),
                         ('r11_s0.20', 11, 0.20), ('r23_s0.50', 23, 0.50)],
                 ladder=False, repeat=False, soft_probe=3, refine=False),
    'oh54': dict(atom='O 0 0 0; H 0 0 0.9697',
                 ncore=2, nact=4, nele=5, basis='6-31g', mult=2,
                 starts=[('zeros', None, 0.0), ('r7_s0.05', 7, 0.05)],
                 ladder=False, repeat=False, soft_probe=1, refine=False,
                 eager_e2=True),
}


# ---------------------------------------------------------------------------
# Chain construction and elementary measurements
# ---------------------------------------------------------------------------

def build(name):
    cfg = SYSTEMS[name]
    mol = pgto.Mole()
    mol.atom = cfg['atom']
    mol.basis = cfg['basis']
    mol.spin = cfg['mult'] - 1
    mol.build()
    nmo = mol.nao_nr()
    pairs = orbital_pairs(cfg['ncore'], cfg['nact'], nmo, include_cv=True)

    def builder(m):
        mf = pscf.RHF(m) if cfg['mult'] == 1 else pscf.ROHF(m)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        return mf

    f_cas = make_nevpt2_theta_fn(mol, cfg['ncore'], cfg['nact'], cfg['nele'],
                                 nmo, pairs, builder, e2_scale=0.0,
                                 ci_solver='blocked', eig_solver='lanczos')
    f_nev = make_nevpt2_theta_fn(mol, cfg['ncore'], cfg['nact'], cfg['nele'],
                                 nmo, pairs, builder, e2_scale=1.0,
                                 ci_solver='blocked', eig_solver='lanczos')
    return mol, nmo, pairs, f_cas, f_nev


def optimize(E_jit, g_jit, theta0, gtol):
    import scipy.optimize as sopt
    res = sopt.minimize(lambda x: float(E_jit(jnp.asarray(x))),
                        np.asarray(theta0, dtype=float),
                        jac=lambda x: np.asarray(g_jit(jnp.asarray(x)),
                                                 dtype=float),
                        method='L-BFGS-B',
                        options=dict(maxiter=4000, gtol=float(gtol),
                                     ftol=1e-15, maxls=60))
    return jnp.asarray(res.x), int(res.nit)


def hessian(g_jit, theta, npar, step=1e-5):
    H = np.zeros((npar, npar))
    for k in range(npar):
        e = np.zeros(npar)
        e[k] = step
        gp = np.asarray(g_jit(theta + jnp.asarray(e)), dtype=float)
        gm = np.asarray(g_jit(theta - jnp.asarray(e)), dtype=float)
        H[:, k] = (gp - gm) / (2.0 * step)
    return 0.5 * (H + H.T)


def newton_refine(E_jit, g_jit, theta, npar, target=1e-10, max_iter=5):
    """Damped Newton with the value-faithful Hessian and a downhill guard."""
    theta = jnp.asarray(theta)
    for _ in range(max_iter):
        g = np.asarray(g_jit(theta), dtype=float)
        if np.linalg.norm(g) < target:
            break
        H = hessian(g_jit, theta, npar)
        lam = 1e-8
        accepted = False
        e_now = float(E_jit(theta))
        for _try in range(16):
            try:
                step = np.linalg.solve(H + lam * np.eye(npar), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            if not np.all(np.isfinite(step)) or np.linalg.norm(step) > 10.0:
                lam *= 10.0
                continue
            cand = theta - jnp.asarray(step)
            if float(E_jit(cand)) <= e_now + 1e-14:
                theta = cand
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
    return theta


def casscf_orbitals(f_cas, theta, name):
    """Canonical orbitals and CI vector at theta (private driver helper)."""
    cfg = SYSTEMS[name]
    C0, h, e = f_cas.arrays
    nmo = C0.shape[1]
    pairs = orbital_pairs(cfg['ncore'], cfg['nact'], nmo, include_cv=True)
    E0, C_can, ci, eps, h1c, eric = _casscf_solution(
        C0, h, e, theta, ncore=cfg['ncore'], nact=cfg['nact'],
        nele=cfg['nele'], nmo=nmo, pairs=pairs, ci_solver='blocked',
        eig_solver='lanczos')
    return C_can, ci


def wavefunction_overlap(mol, Ca, cia, Cb, cib, ncore, nact, nele):
    """|<Psi_a|Psi_b>| for two CAS-CI states in their canonical bases.

    The overlap factorizes into the closed-shell core determinant and the
    active determinant sum.  The spin-orbital convention is the one used
    by ``core.nevpt2``: spin orbital 2p is alpha and 2p+1 is beta of
    spatial orbital p.
    """
    from diffmrc.core.nevpt2.impl import cas_dets_for2

    S = mol.intor('int1e_ovlp')
    Ca = np.asarray(Ca, dtype=float)
    Cb = np.asarray(Cb, dtype=float)
    S_core = Ca[:, :ncore].T @ S @ Cb[:, :ncore]
    det_core = float(np.linalg.det(S_core)) ** 2
    M = Ca[:, ncore:ncore + nact].T @ S @ Cb[:, ncore:ncore + nact]
    nso = 2 * nact
    Sso = np.zeros((nso, nso))
    Sso[0::2, 0::2] = M
    Sso[1::2, 1::2] = M
    dets = cas_dets_for2(ncore, nact, nele)
    occ = [[j for j in range(nso) if (int(d) >> (2 * ncore)) & (1 << j)]
           for d in dets]
    ca = np.asarray(cia, dtype=float).reshape(-1)
    cb = np.asarray(cib, dtype=float).reshape(-1)
    total = 0.0
    for i, di in enumerate(occ):
        if ca[i] == 0.0:
            continue
        for j, dj in enumerate(occ):
            if cb[j] == 0.0:
                continue
            total += ca[i] * cb[j] * np.linalg.det(Sso[np.ix_(di, dj)])
    return abs(float(det_core * total))


def point_record(name, theta, f_cas, f_nev, E_jit, g_jit,
                 s_fn, npar, with_orbitals=True):
    """All local quantities at one theta, plus the canonical orbitals."""
    t0 = time.time()
    E = float(E_jit(theta))
    g = np.asarray(g_jit(theta), dtype=float)
    e2 = float(f_nev.components(theta)[1])
    s = np.asarray(s_fn(theta), dtype=float)
    H = hessian(g_jit, theta, npar)
    split = split_spectrum(H)
    rec = dict(
        E_cas=E, e2=e2,
        g_norm=float(np.linalg.norm(g)),
        g_max=float(np.abs(g).max()),
        s_norm=float(np.linalg.norm(s)),
        hessian_lowest=split.lowest,
        hessian_largest=split.largest,
        n_negative=split.n_negative,
        n_soft=split.n_soft,
        theta=[float(x) for x in np.asarray(theta, dtype=float)],
        H=[[float(x) for x in row] for row in H],
        g=[float(x) for x in g],
        s=[float(x) for x in s],
        seconds=time.time() - t0,
    )
    if with_orbitals:
        C_can, ci = casscf_orbitals(f_cas, theta, name)
        rec['C_can'] = [[float(x) for x in row] for row in np.asarray(C_can)]
        rec['ci'] = [float(x) for x in np.asarray(ci).reshape(-1)]
    return rec


def null_probe(f_nev, theta, H, s, n_soft_max, null_angle=0.1):
    """Finite e2 change along the soft subspace of the Hessian.

    The eigenvector basis inside a degenerate soft subspace is arbitrary, so
    probing a few eigenvectors can miss the direction that carries e2.  The
    first direction probed is therefore the normalized soft-space component
    of ``s`` itself, which maximizes the first-order coupling by
    construction.  The softest eigenvectors are probed as well, for the
    diagnosis of which mode carries the variation.
    """
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    soft = np.where(np.abs(w) <= 1e-3)[0]
    e2_0 = float(f_nev.components(theta)[1])
    s = np.asarray(s, dtype=float)
    directions = []
    if soft.size:
        s_soft = V[:, soft].T @ s
        norm = float(np.linalg.norm(s_soft))
        if norm > 1e-14:
            directions.append(('soft_s', V[:, soft] @ (s_soft / norm),
                               float(norm)))
    for k in soft[np.argsort(np.abs(w[soft]))][:int(n_soft_max)]:
        directions.append(('eig_%d' % k, V[:, k], float(V[:, k] @ s)))
    rows = []
    for label, v, projection in directions:
        up = float(f_nev.components(theta + null_angle * jnp.asarray(v))[1])
        dn = float(f_nev.components(theta - null_angle * jnp.asarray(v))[1])
        rows.append(dict(direction=label, s_projection=projection,
                         de2_up=up - e2_0, de2_down=dn - e2_0,
                         max_abs=float(max(abs(up - e2_0), abs(dn - e2_0)))))
    return rows


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_starts(name, out, mol, nmo, pairs, f_cas, f_nev,
                 eager_e2=False, refine=True):
    cfg = SYSTEMS[name]
    npar = len(pairs)
    E_jit = jax.jit(f_cas)
    g_jit = jax.jit(jax.grad(f_cas))
    if eager_e2:
        corr = lambda t: f_nev.components(t)[1]
        s_fn = jax.grad(corr)
    else:
        s_fn = jax.jit(jax.grad(lambda t: f_nev.components(t)[1]))

    rows = []
    raw_rows = []
    for label, seed, scale in cfg['starts']:
        t0 = time.time()
        theta0 = (np.zeros(npar) if seed is None
                  else np.random.default_rng(seed).normal(scale=scale,
                                                          size=npar))
        theta, nit = optimize(E_jit, g_jit, theta0, 1e-6)
        print('  %-10s opt nit %4d  E %.12f  |g| %.2e  (%.1f s)'
              % (label, nit, float(E_jit(theta)),
                 float(jnp.linalg.norm(g_jit(theta))), time.time() - t0),
              flush=True)
        rec = point_record(name, theta, f_cas, f_nev, E_jit, g_jit,
                           s_fn, npar)
        rec['label'] = label
        rec['nit'] = nit
        raw_rows.append(rec)
        if refine:
            t0 = time.time()
            theta_r = newton_refine(E_jit, g_jit, theta, npar)
            rec_ref = point_record(name, theta_r, f_cas, f_nev,
                                   E_jit, g_jit, s_fn, npar)
            rec_ref['label'] = label + '+newton'
            rec_ref['nit'] = 0
            rec_ref['refined_from'] = label
            rec_ref['seconds_refine'] = time.time() - t0
            print('  %-10s newton |g| %.2e  e2 %+.12f  (%.1f s)'
                  % (label, rec_ref['g_norm'], rec_ref['e2'],
                     rec_ref['seconds_refine']), flush=True)
            rows.append(rec_ref)
        else:
            rows.append(rec)
    out['points'] = rows
    out['unrefined'] = raw_rows
    return rows


def stage_ladder(name, out, mol, nmo, pairs, f_cas, f_nev):
    """Walk the optimizer down a gradient ladder from the lowest minimum."""
    cfg = SYSTEMS[name]
    npar = len(pairs)
    E_jit = jax.jit(f_cas)
    g_jit = jax.jit(jax.grad(f_cas))
    s_fn = jax.jit(jax.grad(lambda t: f_nev.components(t)[1]))
    # Walk down from the *starting point* that produced the reference, not
    # from the converged point: starting at the stationary point leaves
    # every rung at the same place and measures nothing.
    ref = out.get('reference')
    if not ref:
        cand = [p for p in (out.get('points') or [])
                if p.get('n_negative', 0) == 0]
        ref = (min(cand, key=lambda p: p['E_cas'])['label'] if cand else '')
    base = ref.split('+')[0]
    entry = next((s for s in cfg['starts'] if s[0] == base), None)
    if entry is None or entry[1] is None:
        theta0 = jnp.zeros(npar)
    else:
        theta0 = jnp.asarray(np.random.default_rng(entry[1]).normal(
            scale=entry[2], size=npar))
    rows = []
    for gtol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        theta, nit = optimize(E_jit, g_jit, theta0, gtol)
        theta0 = theta
        rec = point_record(name, theta, f_cas, f_nev, E_jit, g_jit,
                           s_fn, npar, with_orbitals=False)
        rec['label'] = 'gtol_%.0e' % gtol
        rec['gtol'] = gtol
        rec['nit'] = nit
        print('  ladder %.0e: E %.12f e2 %+.12f |g| %.2e |s| %.2e'
              % (gtol, rec['E_cas'], rec['e2'], rec['g_norm'],
                 rec['s_norm']), flush=True)
        rows.append(rec)
    out['ladder'] = rows
    return rows


def stage_repeat(name, out, mol, nmo, pairs, f_cas, f_nev):
    """Two independent refinements near the lowest minimum."""
    cfg = SYSTEMS[name]
    npar = len(pairs)
    E_jit = jax.jit(f_cas)
    g_jit = jax.jit(jax.grad(f_cas))
    s_fn = jax.jit(jax.grad(lambda t: f_nev.components(t)[1]))
    points = [p for p in (out.get('points') or [])
              if p.get('n_negative', 0) == 0]
    if not points:
        out['repeat'] = []
        return []
    start = min(points, key=lambda p: p['E_cas'])
    theta_ref = jnp.asarray(start['theta'])
    rows = []
    for seed in (101, 202):
        rng = np.random.default_rng(seed)
        theta0 = theta_ref + jnp.asarray(rng.normal(scale=1e-3, size=npar))
        theta, nit = optimize(E_jit, g_jit, theta0, 1e-8)
        theta = newton_refine(E_jit, g_jit, theta, npar)
        rec = point_record(name, theta, f_cas, f_nev, E_jit, g_jit,
                           s_fn, npar)
        rec['label'] = 'repeat_%d' % seed
        rec['nit'] = nit
        rec['de2_to_reference'] = rec['e2'] - start['e2']
        rec['overlap_to_reference'] = wavefunction_overlap(
            mol, np.asarray(start['C_can']), np.asarray(start['ci']),
            np.asarray(rec['C_can']), np.asarray(rec['ci']),
            cfg['ncore'], cfg['nact'], cfg['nele'])
        print('  repeat %d: E %.12f e2 %+.12f |g| %.2e de2 %+.2e ov %.6f'
              % (seed, rec['E_cas'], rec['e2'], rec['g_norm'],
                 rec['de2_to_reference'], rec['overlap_to_reference']),
              flush=True)
        rows.append(rec)
    out['repeat'] = rows
    return rows


def stage_nullprobe(name, out, mol, nmo, pairs, f_cas, f_nev):
    """Finite e2 probes along the flat Hessian directions of every point."""
    cfg = SYSTEMS[name]
    rows_all = ((out.get('points') or []) + (out.get('repeat') or [])
                + (out.get('unrefined') or []) + (out.get('ladder') or []))
    for p in rows_all:
        H = np.asarray(p['H'], dtype=float)
        theta = jnp.asarray(np.asarray(p['theta']))
        rows = null_probe(f_nev, theta, H, p['s'], cfg['soft_probe'])
        p['null_probe'] = rows
        p['null_probe_max'] = (max(r['max_abs'] for r in rows)
                               if rows else 0.0)
        print('  probe %-12s max |de2| %.3e'
              % (p['label'], p['null_probe_max']), flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(name, out, mol, nmo, pairs):
    """Pairwise overlaps, criterion verdicts and the branch report."""
    cfg = SYSTEMS[name]
    points = out.get('points') or []
    labels = [p['label'] for p in points]
    ov = np.zeros((len(points), len(points)))
    dtheta = np.zeros((len(points), len(points)))
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            dtheta[i, j] = np.linalg.norm(np.asarray(a['theta'])
                                          - np.asarray(b['theta']))
            if i == j:
                ov[i, j] = 1.0
                continue
            ov[i, j] = wavefunction_overlap(
                mol, np.asarray(a['C_can']), np.asarray(a['ci']),
                np.asarray(b['C_can']), np.asarray(b['ci']),
                cfg['ncore'], cfg['nact'], cfg['nele'])
    out['overlap'] = ov.tolist()
    out['theta_distance'] = dtheta.tolist()
    out['labels'] = labels

    verdicts = []
    for p in points:
        g = np.asarray(p['g'], dtype=float)
        s = np.asarray(p['s'], dtype=float)
        H = np.asarray(p['H'], dtype=float)
        report = assess_perturbation_convergence(
            g, s, H, e2_tolerance=DEFAULT_E2_TOLERANCE,
            null_probe_max=p.get('null_probe_max'))
        p['criterion'] = report.as_dict()
        verdicts.append(dict(label=p['label'], value=report.value,
                             bound=report.first_order_bound,
                             required_gradient=report.required_gradient,
                             n_negative=report.n_negative,
                             n_soft=report.n_soft,
                             soft_e2_coupling=report.soft_e2_coupling,
                             null_probe_max=report.null_probe_max,
                             reasons=list(report.reasons)))
        print('  %-14s %-12s bound %.2e req|g| %.2e lam_min %.2e '
              'soft %d s_soft %.2e probe %.2e'
              % (p['label'], report.value, report.first_order_bound,
                 report.required_gradient, report.hessian_lowest,
                 report.n_soft, report.soft_e2_coupling,
                 report.null_probe_max or 0.0), flush=True)
    out['verdicts'] = verdicts

    repeats = []
    for r in out.get('repeat') or []:
        report = assess_perturbation_convergence(
            np.asarray(r['g'], dtype=float), np.asarray(r['s'], dtype=float),
            np.asarray(r['H'], dtype=float),
            e2_tolerance=DEFAULT_E2_TOLERANCE,
            null_probe_max=r.get('null_probe_max'),
            repeat_delta_e2=r.get('de2_to_reference'),
            repeat_overlap=r.get('overlap_to_reference'))
        r['criterion'] = report.as_dict()
        repeats.append(dict(label=r['label'], value=report.value,
                            repeat_delta_e2=report.repeat_delta_e2,
                            repeat_overlap=report.repeat_overlap,
                            reasons=list(report.reasons)))
        print('  %-14s %-12s repeat de2 %.2e overlap %.6f'
              % (r['label'], report.value, report.repeat_delta_e2 or 0.0,
                 report.repeat_overlap or 0.0), flush=True)
    out['repeat_criterion'] = repeats

    # Unrefined L-BFGS points and ladder rungs: these are the samples where a
    # gradient threshold alone can look converged while e2 has not converged.
    accepted = [p for p, v in zip(points, verdicts)
                if v['value'] == 'accept']
    reference = (min(accepted, key=lambda p: p['E_cas']) if accepted
                 else min(points, key=lambda p: p['E_cas'])
                 if points else None)
    out['reference'] = reference['label'] if reference else None
    for key in ('unrefined', 'ladder'):
        checked = []
        for r in out.get(key) or []:
            report = assess_perturbation_convergence(
                np.asarray(r['g'], dtype=float),
                np.asarray(r['s'], dtype=float),
                np.asarray(r['H'], dtype=float),
                e2_tolerance=DEFAULT_E2_TOLERANCE,
                null_probe_max=r.get('null_probe_max'))
            r['criterion'] = report.as_dict()
            if reference is not None:
                r['de2_to_reference'] = r['e2'] - reference['e2']
                r['e2_converged'] = (abs(r['de2_to_reference'])
                                     <= DEFAULT_E2_TOLERANCE)
            else:
                r['de2_to_reference'] = None
                r['e2_converged'] = None
            r['criterion_correct'] = (
                None if r['e2_converged'] is None
                else (report.certified == r['e2_converged']))
            checked.append(dict(
                label=r['label'], gtol=r.get('gtol'),
                g_norm=r['g_norm'], e2=r['e2'],
                de2_to_reference=r['de2_to_reference'],
                e2_converged=r['e2_converged'],
                value=report.value, certified=report.certified,
                bound=report.first_order_bound,
                required_gradient=report.required_gradient,
                null_probe_max=report.null_probe_max,
                correct=r['criterion_correct']))
            print('  %-8s %-10s |g| %.2e de2 %+.2e converged %s -> %s %s'
                  % (key, r['label'], r['g_norm'],
                     r['de2_to_reference'] or 0.0, r['e2_converged'],
                     report.value, r['criterion_correct']), flush=True)
        out[key + '_criterion'] = checked

    # Branch report: one entry per distinct orbital solution.  Several
    # starts can reach the same solution with different convergence
    # quality; counting them separately would report a spurious
    # "degenerate" family.  Group by wavefunction overlap and keep the
    # lowest-energy certified member (or the lowest-energy member when
    # none is certified).
    n = len(points)
    assigned = [False] * n
    groups = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [j for j in range(n) if ov[i, j] >= 0.9]
        for j in group:
            assigned[j] = True
        groups.append(group)
    bp = []
    for group in groups:
        certified = [j for j in group if verdicts[j]['value'] == 'accept']
        pool = certified if certified else group
        j = min(pool, key=lambda k: points[k]['E_cas'])
        p = points[j]
        bp.append(BranchPoint(
            label=p['label'], energy=p['E_cas'], e2=p['e2'],
            gradient_norm=p['g_norm'],
            hessian_lowest=p['criterion']['hessian_lowest'],
            certified=bool(certified), theta=p['theta']))
        if len(group) > 1:
            print('  group: %s (%d starts) certified %s'
                  % (p['label'], len(group), bool(certified)), flush=True)
    spread = branch_spread(bp, e2_tolerance=DEFAULT_E2_TOLERANCE)
    out['branch'] = spread.as_dict()
    print('  branch: selected %s  e2 spread %.3e  ambiguous %s'
          % (spread.selected.label if spread.selected else None,
             spread.e2_spread, spread.ambiguous), flush=True)
    return out


def main(name, out_path, stages):
    cfg = SYSTEMS[name]
    mol, nmo, pairs, f_cas, f_nev = build(name)
    print('%s: nmo %d npar %d' % (name, nmo, len(pairs)), flush=True)
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out = json.load(fh)
    want = stages.split(',') if stages else ['starts', 'ladder', 'repeat',
                                             'nullprobe', 'analyse']
    if 'starts' in want:
        stage_starts(name, out, mol, nmo, pairs, f_cas, f_nev,
                     eager_e2=cfg.get('eager_e2', False),
                     refine=cfg.get('refine', True))
        _save(out, out_path)
    if 'ladder' in want and cfg['ladder']:
        stage_ladder(name, out, mol, nmo, pairs, f_cas, f_nev)
        _save(out, out_path)
    if 'repeat' in want and cfg['repeat']:
        stage_repeat(name, out, mol, nmo, pairs, f_cas, f_nev)
        _save(out, out_path)
    if 'nullprobe' in want:
        stage_nullprobe(name, out, mol, nmo, pairs, f_cas, f_nev)
        _save(out, out_path)
    if 'analyse' in want:
        analyse(name, out, mol, nmo, pairs)
        _save(out, out_path)


def _save(out, path):
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else '')
