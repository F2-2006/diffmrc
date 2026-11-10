#!/usr/bin/env python
"""Stationary-point spectra and a basin-hopping search on CASSCF surfaces.

The open problem is that ``core.stationary.select_stationary_point`` makes
the *selection* deterministic but leaves the *search* to the caller. This
script builds the spectrum the selection consumes, then compares three ways
of generating it at equal budget:

    multistart   one reference start plus random orbital rotations
    bh-orbit     basin hopping with orbital-rotation kicks only
    bh-geometry  basin hopping with orbital kicks and geometry displacements

Every strategy refines with the production path
(``driver.orbitals.converge_orbital_point``) under the same settings, so the
comparison is of starting-point generation, not of optimizers.

The "global minimum" is the lowest certified minimum found by any run in the
whole experiment -- a best-known reference, not a proof. A strategy is
credited with a hit when its own selected point lies within 1e-6 (relative)
of that energy and carries a positive-definite curvature report.

Usage
-----
    python stationary_spectrum_search.py --phase spectrum
    python stationary_spectrum_search.py --phase compare --repeats 5 --budget 6
    python stationary_spectrum_search.py --phase alt-space
    python stationary_spectrum_search.py --phase stop-rule
    python stationary_spectrum_search.py --phase all

The JSON written by ``--phase spectrum`` (``--spectrum-json``) is reused by
the compare phase, so the reference energies come from the same run.
"""
import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp                                    # noqa: E402

from pyscf import gto, scf                                      # noqa: E402

from diffmrc.core.crossing import orbital_space_overlap         # noqa: E402
from diffmrc.core.gauge import canonical_mo_phases              # noqa: E402
from diffmrc.core.orblib import orbital_pairs                   # noqa: E402
from diffmrc.core.rotation import block_rotation                # noqa: E402
from diffmrc.core.stationary import (                           # noqa: E402
    StationaryCandidate, basin_hopping_search,
    cluster_stationary_points, curvature_report)
from diffmrc.driver.kernel import (                             # noqa: E402
    energy_function, partition_from_occupations)
from diffmrc.driver.orbitals import converge_orbital_point      # noqa: E402
from diffmrc.core.optimizer import optimize_theta_scipy         # noqa: E402

# 6-31G geometries. The active windows are the HOMO/LUMO pair of each RHF
# reference, which is the smallest CAS with a non-trivial orbital rotation;
# the closed-shell pi^4 windows of N2 and CO rotate nothing and were
# dropped after a probe showed every start returning the RHF energy.
SYSTEMS = {
    'H2O': dict(atom='O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880',
                active=[4, 5], charge=0, spin=0),
    'CH2': dict(atom='C 0 0 0; H 1.11 0 -0.3096; H -1.11 0 -0.3096',
                active=[3, 4], charge=0, spin=0),
    'N2': dict(atom='N 0 0 0; N 0 0 1.0977', active=[6, 7],
               charge=0, spin=0),
    'CO': dict(atom='C 0 0 0; O 0 0 1.1283', active=[6, 7],
               charge=0, spin=0),
}

# The alternative window is CAS(4,4) in every case: the HOMO-1..LUMO+1
# manifold, which for N2 and CO is the two pi / two pi* set. A different
# active space is a different Hamiltonian, so its spectrum is reported on
# its own and never merged with the CAS(2,2) one.
ALT_SPACES = {
    'H2O': dict(atom=SYSTEMS['H2O']['atom'], active=[3, 4, 5, 6],
                charge=0, spin=0),
    'CH2': dict(atom=SYSTEMS['CH2']['atom'], active=[2, 3, 4, 5],
                charge=0, spin=0),
    'N2': dict(atom=SYSTEMS['N2']['atom'], active=[5, 6, 7, 8],
               charge=0, spin=0),
    'CO': dict(atom=SYSTEMS['CO']['atom'], active=[5, 6, 7, 8],
               charge=0, spin=0),
}

BASIS = '6-31g'
TARGET = 1e-6          # gradient norm admitted as stationary
STRICT_TARGET = 1e-11  # gradient norm for the strict flag
ENERGY_TOL = 1e-4      # relative, for clustering distinct points
OVERLAP_TOL = 0.99     # active-space overlap for the same point

# A bounded convergence for a multistart driver. The trust-region first
# stage was measured at 9-20 s per start on N2/6-31G, but a start far from
# every basin can spend the whole trust-region budget and take minutes; the
# L-BFGS stage with a 150-step cap keeps the per-start cost predictable,
# and the refinement then reaches 1e-7. The energies of the basins agree
# with the tighter 1e-8 runs to 1e-10, which is well below the 1e-4
# clustering tolerance.
CONVERGE_KW = dict(max_steps=150, use_trust_region=False,
                   optimize_tol=1e-4, refine_target=1e-7,
                   refine_max_iter=25)


class CASSystem:
    """One (geometry, basis, active space) CASSCF energy function."""

    def __init__(self, name, atom, active, basis=BASIS, charge=0, spin=0):
        self.name = name
        self.atom = atom
        self.basis = basis
        self.active = list(active)
        self.charge = int(charge)
        self.spin = int(spin)
        self.mol = gto.M(atom=atom, basis=basis, charge=self.charge,
                         spin=self.spin, verbose=0)
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        self.C0 = np.asarray(self.mf.mo_coeff, dtype=float)
        self.occ = np.asarray(self.mf.mo_occ, dtype=float)
        self.space = partition_from_occupations(self.occ, self.active)
        self.E, self.npar = energy_function(self.mol, self.C0, self.space)
        self.nmo = self.mol.nao_nr()
        self.pairs = orbital_pairs(self.space.ncore, self.space.ncas,
                                   self.nmo, include_cv=True)
        self.ncore = int(self.space.ncore)
        self.ncas = int(self.space.ncas)
        self.nele = int(self.space.n_elec)
        self.S = self.mol.intor('int1e_ovlp')
        self.C0o = np.asarray(canonical_mo_phases(
            jnp.asarray(self.C0, dtype=float)))[:, self.space.mo_order]
        self._hess = jax.jit(jax.hessian(self.E))

    def converge(self, theta0, label, **overrides):
        kwargs = dict(CONVERGE_KW)
        kwargs.update(overrides)
        point = converge_orbital_point(
            self.E, self.npar, theta0=np.asarray(theta0, dtype=float),
            **kwargs)
        theta = np.asarray(point.theta, dtype=float)
        curvature, route = self.curvature(theta)
        return StationaryCandidate(
            theta=theta, energy=float(point.energy),
            grad_norm=float(point.grad_norm), curvature=curvature,
            label=str(label))

    def curvature(self, theta):
        """Curvature at a point, AD first and gradient differences second."""
        from diffmrc.core.stationary import curvature_from_gradient

        hessian = np.asarray(self._hess(jnp.asarray(theta)))
        if np.all(np.isfinite(hessian)):
            return curvature_report(hessian), 'ad'
        g_jit = jax.jit(jax.grad(self.E))
        grad_fn = lambda x: np.asarray(                              # noqa: E731
            g_jit(jnp.asarray(x)), dtype=float)
        hessian = curvature_from_gradient(grad_fn, theta, step=1e-5,
                                          max_parameters=400)
        return curvature_report(hessian), 'fd'

    def mo(self, theta):
        rotation = block_rotation(np.asarray(theta, dtype=float), self.nmo,
                                  self.pairs)
        return np.asarray(self.C0o @ rotation, dtype=float)

    def overlap(self, theta_a, theta_b):
        return orbital_space_overlap(self.mo(theta_a), self.mo(theta_b),
                                     self.S, self.ncore, self.ncas)

    def same_point(self, candidate_a, candidate_b):
        return self.overlap(candidate_a.theta,
                            candidate_b.theta) >= OVERLAP_TOL

    def spectrum(self, candidates):
        return cluster_stationary_points(
            candidates, target=TARGET, energy_tol=ENERGY_TOL,
            same_point=self.same_point, strict_target=STRICT_TARGET)


# ---------------------------------------------------------------------------
# Start generation.
# ---------------------------------------------------------------------------


def rotation_kick(scale):
    """A proposal that adds a Gaussian orbital rotation of size ``scale``."""
    def propose(rng, theta):
        return np.asarray(theta) + scale * rng.standard_normal(
            np.shape(theta))
    return propose


class GeometryKick:
    """A proposal that transfers a branch found at a displaced geometry.

    The geometry move does not optimize the nuclei. It displaces them,
    converges the orbitals there with a short L-BFGS run, and returns the
    resulting rotation pattern as the starting point at the reference
    geometry. The optimization target therefore stays the reference
    geometry -- energies and active-space overlaps remain comparable -- and
    the move only changes which orbital branch the refinement starts from.

    Two displacements are prebuilt per system so the compiled displaced
    energy functions are reused; each kick picks one at random.
    """

    def __init__(self, system, sigma=0.08, count=2, seed=0):
        self.system = system
        self.displaced = []
        rng = np.random.default_rng(seed)
        # The geometry is kept as the input string; parse it rather than
        # reading pyscf's internal atom list.
        atoms = []
        for token in system.atom.split(';'):
            parts = token.split()
            if parts:
                atoms.append((parts[0],
                              np.asarray(parts[1:4], dtype=float)))
        for _ in range(count):
            coords = np.asarray([a[1] for a in atoms], dtype=float)
            shifted = coords + sigma * rng.standard_normal(coords.shape)
            atom_str = '; '.join(
                '%s %.10f %.10f %.10f' % (a[0], *shifted[k])
                for k, a in enumerate(atoms))
            try:
                other = CASSystem(system.name, atom_str, system.active,
                                  basis=system.basis,
                                  charge=system.charge, spin=system.spin)
            except Exception:
                continue
            if other.nmo != system.nmo or other.npar != system.npar:
                continue
            self.displaced.append(other)

    def __call__(self, rng, theta):
        if not self.displaced:
            return np.asarray(theta)
        other = self.displaced[int(rng.integers(len(self.displaced)))]
        _, start, _, info = optimize_theta_scipy(
            other.E, other.npar, steps=100, g_conv=1e-3,
            theta0=np.asarray(theta, dtype=float))
        if not np.all(np.isfinite(np.asarray(start))):
            return np.asarray(theta)
        return np.asarray(start, dtype=float)


def build_spectrum(system, n_random=6, n_perturb=3, seed=0):
    """Zero start, random rotations, then perturbations of the best point."""
    rng = np.random.default_rng(seed)
    candidates = [system.converge(np.zeros(system.npar), 'zero')]
    scales = (0.1, 0.3, 1.0)
    for k in range(n_random):
        scale = scales[k % len(scales)]
        start = scale * rng.standard_normal(system.npar)
        candidates.append(system.converge(start, 'rand%.1f-%d' % (scale, k)))
    first = system.spectrum(candidates)
    best = first.best
    if best is not None:
        for k in range(n_perturb):
            sigma = (0.05, 0.2)[k % 2]
            start = np.asarray(best.theta, dtype=float) \
                + sigma * rng.standard_normal(system.npar)
            candidates.append(system.converge(start,
                                              'pert%.2f-%d' % (sigma, k)))
    return candidates


def spectrum_rows(system, spectrum):
    """One row per distinct point, with overlap to the lowest point."""
    best = spectrum.best
    rows = []
    for point in spectrum.points:
        overlap = 1.0 if best is None else system.overlap(
            point.theta, best.theta)
        rows.append(dict(
            label=point.label,
            energy=float(point.energy),
            delta=float(point.energy - best.energy) if best is not None else 0.0,
            grad_norm=float(point.grad_norm),
            n_negative=(None if point.curvature is None
                        else int(point.curvature.n_negative)),
            min_eigenvalue=(None if point.curvature is None
                            else float(point.curvature.smallest)),
            is_minimum=bool(point.is_minimum),
            strict=bool(point.strict),
            n_members=int(point.n_members),
            labels=list(point.labels),
            overlap_to_lowest=float(overlap)))
    return rows


def print_spectrum(system, spectrum):
    rows = spectrum_rows(system, spectrum)
    print('  distinct points: %d of %d stationary (%d candidates)'
          % (spectrum.n_distinct, spectrum.n_stationary,
             spectrum.n_candidates))
    print('  %-18s %16s %10s %10s %7s %7s %5s %s'
          % ('label', 'E (Ha)', 'dE (mHa)', '|g|', 'n_neg', 'min_eig',
             'hits', 'overlap'))
    for row in rows:
        print('  %-18s %16.10f %10.3f %10.2e %7s %7.1e %5d %.3f'
              % (row['label'], row['energy'], 1e3 * row['delta'],
                 row['grad_norm'],
                 '-' if row['n_negative'] is None else row['n_negative'],
                 float('nan') if row['min_eigenvalue'] is None
                 else row['min_eigenvalue'],
                 row['n_members'], row['overlap_to_lowest']))


# ---------------------------------------------------------------------------
# Phases.
# ---------------------------------------------------------------------------


def phase_spectrum(names, out_path):
    result = {}
    for name in names:
        print('=' * 78)
        system = CASSystem(name, **SYSTEMS[name])
        print('%s/%s  CAS(%d,%d)  npar=%d  ncore=%d  E_RHF=%.10f'
              % (name, BASIS, system.ncas, system.nele, system.npar,
                 system.ncore, system.mf.e_tot))
        started = time.time()
        candidates = build_spectrum(system, seed=11)
        spectrum = system.spectrum(candidates)
        print_spectrum(system, spectrum)
        print('  wall %.1f s' % (time.time() - started))
        result[name] = dict(
            npar=system.npar, ncore=system.ncore, ncas=system.ncas,
            nele=system.nele,
            e_rhf=float(system.mf.e_tot),
            points=spectrum_rows(system, spectrum),
            n_candidates=spectrum.n_candidates,
            n_stationary=spectrum.n_stationary,
            n_distinct=spectrum.n_distinct,
            n_energy_levels=spectrum.n_energy_levels,
            spread=float(spectrum.spread),
            certificate_coverage=float(spectrum.certificate_coverage),
            seconds=float(time.time() - started))
        # Written after every system: a slow start on the next system must
        # not cost the spectra already measured.
        if out_path:
            with open(out_path, 'w', encoding='utf-8') as handle:
                json.dump(result, handle, indent=2)
    if out_path:
        print('wrote %s' % out_path)
    return result


def multistart_run(system, budget, seed):
    rng = np.random.default_rng(seed)
    # One reference start plus ``budget`` random starts, matching the
    # basin-hopping budget of one initial refinement plus ``budget`` moves.
    candidates = [system.converge(np.zeros(system.npar), 'zero')]
    for k in range(int(budget)):
        scale = float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))
        candidates.append(system.converge(
            scale * rng.standard_normal(system.npar), 'ms%d' % k))
    spectrum = system.spectrum(candidates)
    return spectrum, candidates


def bh_run(system, budget, seed, geometry_move=None):
    moves = [('rot0.05', rotation_kick(0.05)),
             ('rot0.3', rotation_kick(0.3)),
             ('rot0.8', rotation_kick(0.8))]
    if geometry_move is not None:
        moves.append(('geom', geometry_move))
    started = time.time()
    result = basin_hopping_search(
        system.converge, np.zeros(system.npar), moves,
        same_point=system.same_point, target=TARGET,
        energy_tol=ENERGY_TOL, strict_target=STRICT_TARGET,
        max_steps=int(budget), max_no_new=None, temperature=0.0,
        require_curvature=True, novelty_window=0.2, seed=seed)
    seconds = time.time() - started
    return result.spectrum, result, seconds


def _first_hit(energies, minima, reference, tol=1e-6):
    """Converge index of the first certified minimum at the reference.

    ``None`` means the run never reached the reference. The index is the
    number of refinements the strategy had paid for when it first held the
    answer, so it measures cost at equal reliability.
    """
    if reference is None:
        return None
    for index, (energy, is_minimum) in enumerate(zip(energies, minima)):
        if not is_minimum:
            continue
        if abs(float(energy) - float(reference)) \
                <= tol * max(1.0, abs(float(reference))):
            return index
    return None


def _is_hit(candidate, reference, tol=1e-6):
    if candidate is None or reference is None:
        return False
    if not candidate.is_minimum:
        return False
    return abs(float(candidate.energy) - float(reference)) \
        <= tol * max(1.0, abs(float(reference)))


def phase_compare(names, repeats, budget, spectrum_json, out_path):
    references = {}
    if spectrum_json and os.path.exists(spectrum_json):
        with open(spectrum_json, encoding='utf-8') as handle:
            stored = json.load(handle)
        for name, entry in stored.items():
            energies = [p['energy'] for p in entry['points']
                        if p['is_minimum']]
            references[name] = min(energies) if energies else None

    records = []
    for name in names:
        print('=' * 78)
        print('%s: baseline vs basin hopping' % name)
        system = CASSystem(name, **SYSTEMS[name])
        # Pay the compilation of the energy, gradient and Hessian-vector
        # graphs before any strategy is timed. The cache lives on E_theta,
        # so without this warm-up the first strategy would be charged for
        # the other two as well.
        system.converge(np.zeros(system.npar), 'warmup')
        geometry_move = GeometryKick(system, seed=0)
        strategies = [
            ('multistart', lambda seed: multistart_run(system, budget, seed)),
            ('bh-orbit', lambda seed: bh_run(system, budget, seed)),
            ('bh-geometry', lambda seed: bh_run(system, budget, seed,
                                                geometry_move)),
        ]
        per_strategy = {key: [] for key, _ in strategies}
        for seed in range(int(repeats)):
            for key, runner in strategies:
                started = time.time()
                outcome = runner(seed)
                elapsed = time.time() - started
                if key == 'multistart':
                    spectrum, entries = outcome
                    bh = None
                else:
                    spectrum, bh, _seconds = outcome
                    entries = [bh.initial] + list(bh.trace)
                selection = spectrum.selection
                best = selection.chosen
                records.append(dict(
                    system=name, strategy=key, seed=seed,
                    energy=(None if best is None else float(best.energy)),
                    certified=bool(best is not None and best.is_minimum),
                    status=selection.status,
                    n_distinct=int(spectrum.n_distinct),
                    n_stationary=int(spectrum.n_stationary),
                    n_candidates=int(spectrum.n_candidates),
                    seconds=float(elapsed),
                    first_hit=None,
                    entry_energies=[float(e.energy) for e in entries],
                    entry_minima=[bool(e.is_minimum) for e in entries],
                    stop_reason=(None if bh is None else bh.stop_reason),
                    n_accepted=(None if bh is None else int(bh.n_accepted))))
                per_strategy[key].append(records[-1])
        known = [r['energy'] for r in records
                 if r['system'] == name and r['energy'] is not None
                 and r['certified']]
        reference = references.get(name)
        if known:
            reference = min(known) if reference is None else min(
                reference, min(known))
        print('  reference (best certified minimum): %s'
              % ('none' if reference is None else '%.12f' % reference))
        for key, _ in strategies:
            for row in per_strategy[key]:
                row['first_hit'] = _first_hit(
                    row['entry_energies'], row['entry_minima'], reference)
            rows = per_strategy[key]
            hits = sum(1 for row in rows
                       if _is_hit(_row_candidate(row), reference))
            seconds = [row['seconds'] for row in rows]
            hits_at = [row['first_hit'] for row in rows
                       if row['first_hit'] is not None]
            print('  %-12s hit %d/%d  E_best %s  n_distinct %.1f  '
                  'first hit median %s  median %.1f s  total %.0f s'
                  % (key, hits, len(rows),
                     'none' if not any(row['energy'] is not None
                                       for row in rows)
                     else '%.10f' % min(row['energy'] for row in rows
                                        if row['energy'] is not None),
                     float(np.mean([row['n_distinct'] for row in rows])),
                     'none' if not hits_at else '%.1f' % float(
                         np.median(hits_at)),
                     float(np.median(seconds)), float(np.sum(seconds))))
        references[name] = reference

    summary = {}
    for name in names:
        summary[name] = {}
        rows = [r for r in records if r['system'] == name]
        reference = references.get(name)
        for key in ('multistart', 'bh-orbit', 'bh-geometry'):
            subset = [r for r in rows if r['strategy'] == key]
            hits = sum(1 for row in subset
                       if _is_hit(_row_candidate(row), reference))
            summary[name][key] = dict(
                hits=hits, trials=len(subset),
                hit_rate=(hits / len(subset) if subset else 0.0),
                mean_seconds=float(np.mean([r['seconds'] for r in subset])),
                median_seconds=float(np.median(
                    [r['seconds'] for r in subset])),
                mean_distinct=float(np.mean(
                    [r['n_distinct'] for r in subset])),
                median_first_hit=(None if not [
                    r['first_hit'] for r in subset
                    if r['first_hit'] is not None] else float(np.median(
                        [r['first_hit'] for r in subset
                         if r['first_hit'] is not None]))))
    overall = {}
    for key in ('multistart', 'bh-orbit', 'bh-geometry'):
        subset = [r for r in records if r['strategy'] == key]
        hits = sum(1 for row in subset
                   if _is_hit(_row_candidate(row),
                              references.get(row['system'])))
        overall[key] = dict(hits=hits, trials=len(subset),
                            hit_rate=(hits / len(subset) if subset else 0.0),
                            mean_seconds=float(np.mean(
                                [r['seconds'] for r in subset])),
                            median_first_hit=(None if not [
                                r['first_hit'] for r in subset
                                if r['first_hit'] is not None] else float(
                                    np.median([r['first_hit'] for r in subset
                                               if r['first_hit'] is not None]))))
    print('=' * 78)
    print('overall hit rates (best-known reference per system)')
    for key, entry in overall.items():
        print('  %-12s %d/%d = %.2f  mean %.1f s'
              % (key, entry['hits'], entry['trials'], entry['hit_rate'],
                 entry['mean_seconds']))
    payload = dict(references=references, summary=summary, overall=overall,
                   records=records, repeats=int(repeats), budget=int(budget))
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        print('wrote %s' % out_path)
    return payload


def _row_candidate(row):
    """Adapter so a recorded run answers the hit test like a candidate."""
    if row.get('energy') is None:
        return None

    class _Recorded:
        energy = row['energy']
        is_minimum = row['certified']
    return _Recorded()


def phase_alt_space(names, out_path):
    result = {}
    for name in names:
        if name not in ALT_SPACES:
            continue
        print('=' * 78)
        print('%s/%s  alternative active space %s'
              % (name, BASIS, ALT_SPACES[name]['active']))
        system = CASSystem(name, **ALT_SPACES[name])
        print('  npar=%d  ncore=%d  ncas=%d  nele=%d'
              % (system.npar, system.ncore, system.ncas, system.nele))
        candidates = build_spectrum(system, n_random=6, n_perturb=2, seed=23)
        spectrum = system.spectrum(candidates)
        print_spectrum(system, spectrum)
        result[name] = dict(active=list(system.active), npar=system.npar,
                            points=spectrum_rows(system, spectrum),
                            n_distinct=spectrum.n_distinct,
                            spread=float(spectrum.spread),
                            certificate_coverage=float(
                                spectrum.certificate_coverage))
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        print('wrote %s' % out_path)
    return result


def phase_stop_rule(name, spectrum_json):
    print('=' * 78)
    print('%s: stopping rules' % name)
    system = CASSystem(name, **SYSTEMS[name])
    reference = None
    if spectrum_json and os.path.exists(spectrum_json):
        with open(spectrum_json, encoding='utf-8') as handle:
            stored = json.load(handle)
        energies = [p['energy'] for p in stored[name]['points']
                    if p['is_minimum']]
        reference = min(energies) if energies else None
    if reference is None:
        print('  no spectrum reference; the hit column will read False')
    for max_no_new in (3, 5):
        result = basin_hopping_search(
            system.converge, np.zeros(system.npar),
            [('rot0.05', rotation_kick(0.05)),
             ('rot0.3', rotation_kick(0.3)),
             ('rot0.8', rotation_kick(0.8))],
            same_point=system.same_point, target=TARGET,
            energy_tol=ENERGY_TOL, strict_target=STRICT_TARGET,
            max_steps=20, max_no_new=max_no_new, coverage_gate=True,
            require_curvature=True, novelty_window=0.2, seed=31)
        hit = _is_hit(result.best, reference)
        print('  max_no_new=%d -> %s after %d moves, %d new, '
              'coverage %.2f, best %s, hit=%s'
              % (max_no_new, result.stop_reason, result.n_steps,
                 result.n_new, result.certificate_coverage,
                 'none' if result.best is None
                 else '%.12f' % result.best.energy, hit))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', default='spectrum',
                        choices=('spectrum', 'compare', 'alt-space',
                                 'stop-rule', 'all'))
    parser.add_argument('--systems', default='H2O,CH2,N2,CO')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--budget', type=int, default=6)
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--spectrum-json', default=None)
    args = parser.parse_args(argv)

    names = [s.strip() for s in args.systems.split(',') if s.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    spectrum_json = args.spectrum_json or os.path.join(
        args.outdir, 'spectrum_631g.json')
    compare_json = os.path.join(args.outdir, 'compare_631g.json')
    alt_json = os.path.join(args.outdir, 'alt_space_631g.json')

    if args.phase in ('spectrum', 'all'):
        phase_spectrum(names, spectrum_json)
    if args.phase in ('compare', 'all'):
        phase_compare(names, args.repeats, args.budget, spectrum_json,
                      compare_json)
    if args.phase in ('alt-space', 'all'):
        phase_alt_space(names, alt_json)
    if args.phase in ('stop-rule', 'all'):
        phase_stop_rule(names[0], spectrum_json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
