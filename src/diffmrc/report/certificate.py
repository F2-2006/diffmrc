"""Branch certificates for multireference electronic-structure runs.

A branch certificate does not answer "which solution is correct".  It
answers a narrower and reproducible question: "is the point in hand the
declared target branch, under the declared evidence, tolerances and
environment?"  The answer is three-valued: ``accept``, ``reject`` or
``undetermined``.  A tolerance violation never falls back to the nearest
branch; it returns ``undetermined``.

Public interface
----------------
``build_certificate(case, ...)``
    Build a certificate from a case record.
``validate_certificate(cert, ...)``
    Validate a certificate and recompute its verdict.
``classify_degeneracies(values, ...)``
    Classify near-degenerate pairs of a spectrum.
``two_electron_singlet_rotation_replicates(...)`` and
``single_orbital_entropy(...)``
    Canonical gauge-interval construction for the two-electron singlet.
``canonical_json``, ``sha256_hex``, ``file_sha256``, ``payload_hash``,
``tolerance_id`` and ``environment_hash``
    Deterministic serialization and identity hashes.
``main(argv)``
    Command-line entry point with the ``generate`` and ``verify``
    subcommands.

The JSON Schema is shipped next to this module as
``branch_certificate.schema.json``; the verifier uses it when
``jsonschema`` is installed and falls back to the structural checks when
it is not.

Validated scope
---------------
Two shipped example certificates (N2/cc-pVDZ and CO/6-31G, both
CAS(2,2)), a 24-pair degeneracy fixture, hash stability under a changed
timestamp, hash sensitivity to a changed tolerance, environment field or
code commit, the undetermined-on-violation rule, positive-match
rejection, the canonical-interval arithmetic and the CLI exit codes are
pinned by ``tests/report/test_certificate.py``.  No electronic-structure
calculation is run by the tests.

The schema has three hard constraints, and each one is enforced here:

1.  Gauge-dependent quantities carry a canonical interval built from at
    least three independent orbital rotations performed inside the same
    active space.  A rotation that changes the active-space composition
    is a physical difference, not gauge spread, and is refused.  The
    degeneracy criterion threshold and the partner set are recorded.  If
    the interval is wider than the real difference between candidates,
    the observable is marked non-discriminating and is excluded from the
    acceptance decision.

2.  Tolerances are written together with the code commit.  The
    tolerance identifier is a hash over the tolerance values, the commit
    and the declared code version.

3.  A tolerance violation yields ``undetermined``, not the nearest
    branch.  ``reject`` is reserved for evidence that positively matches
    a declared *other* branch.

The environment (BLAS/LAPACK, JAX/XLA, effective OpenMP thread count,
precision mode) enters the payload hash.  A change in any of those
fields is a new certificate identity.  A certificate does not claim
thread-insensitivity unless a measured difference is recorded and the
tolerance is shown to be larger than that difference.

The module is self-contained: the standard library only.  NumPy and JAX
are imported only by :func:`collect_environment`, and only to record
their versions; a missing package does not stop certificate generation.

Boundaries
----------
The certificate is only as good as its declared evidence.  It does not
discover branches, and it does not certify that the branch inventory is
complete: an incomplete inventory is recorded, hashed and reported as a
residual risk.  It does not claim thread insensitivity unless a measured
difference is recorded and the tolerance is shown to be larger than that
difference.  The verdict is about the declared branch identity under the
declared tolerances, not about the physical accuracy of the underlying
method.

Command line
------------

    python -m diffmrc.report.certificate generate case.json -o certificate.json
    python -m diffmrc.report.certificate verify certificate.json

Exit codes for ``verify``: 0 accept, 1 reject, 2 undetermined,
3 invalid document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = '1.0'
KIND = 'diffmrc.branch-certificate'
MIN_ROTATIONS = 3

MODULE_PATH = Path(__file__).resolve()
SCHEMA_PATH = MODULE_PATH.parent / 'branch_certificate.schema.json'

DEFAULT_TOLERANCES = {
    'gradient_norm_max': 1e-5,
    'energy_match_hartree': 5e-4,
    'occupation_match_max': 0.05,
    'overlap_min': 0.95,
    'curvature_min_eigenvalue': -1e-4,
    'canonical_interval_energy_max_hartree': 1e-8,
    'degeneracy_relative_threshold': 1e-6,
    'degeneracy_noise_floor_relative': 1e-12,
    'policy': {
        'require_complete_environment': False,
        'allow_incomplete_branch_inventory': True,
    },
}

ENVIRONMENT_COMPONENTS = (
    'python', 'numpy', 'blas', 'lapack', 'jax', 'jaxlib', 'xla',
    'pyscf', 'pyscfad',
)
ENVIRONMENT_SCALARS = ('platform', 'omp_num_threads_effective',
                       'precision_mode')


# ---------------------------------------------------------------------------
# Canonical serialization and hashing
# ---------------------------------------------------------------------------

def _normalize_for_json(value):
    """Return a JSON-safe copy with stable float conventions.

    ``-0.0`` and ``0.0`` compare equal but serialize differently; a hash
    must not depend on the sign of a zero.  Non-finite floats are refused
    rather than silently serialized as ``NaN`` or ``Infinity``.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'non-finite float in certificate payload: {value!r}')
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    return value


def canonical_json(value):
    """Deterministic JSON text used for every hash in this module."""
    return json.dumps(_normalize_for_json(value), sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def sha256_hex(value):
    """SHA-256 of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    """SHA-256 of a file's bytes, or ``None`` when the file is absent."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def tolerance_id(tolerances, versions):
    """Hash binding the tolerance values to the code commit and version."""
    return sha256_hex({
        'tolerances': {k: v for k, v in tolerances.items()
                       if k != 'tolerance_id'},
        'code_commit': versions.get('code_commit'),
        'code_version': versions.get('code_version'),
    })


def environment_hash(environment):
    """Hash covering both the measurement and generation environments."""
    return sha256_hex({
        'measurement': environment.get('measurement'),
        'generation': environment.get('generation'),
        'thread_sensitivity': environment.get('thread_sensitivity'),
        'status': environment.get('status'),
        'missing': environment.get('missing', []),
    })


def payload_hash(certificate):
    """Hash of the certificate without its identity and timestamp fields."""
    body = {k: v for k, v in certificate.items()
            if k not in ('certificate_id', 'payload_hash', 'created_utc')}
    return sha256_hex(body)


# ---------------------------------------------------------------------------
# Environment collection
# ---------------------------------------------------------------------------

def _component(name, version, build=None):
    return {'name': str(name), 'version': str(version),
            **({'build': build} if build is not None else {})}


def _numpy_blas_lapack():
    """Best-effort BLAS/LAPACK record from NumPy's build configuration."""
    try:
        import numpy as np
    except Exception:
        return (_component('unrecorded', 'unrecorded'),
                _component('unrecorded', 'unrecorded'))
    record = {}
    try:
        config = np.show_config(mode='dicts')
        deps = config.get('Build Dependencies', {})
        for key in ('blas', 'lapack'):
            entry = deps.get(key, {})
            name = entry.get('name') or 'unrecorded'
            version = entry.get('version') or 'unrecorded'
            openblas = entry.get('openblas configuration')
            record[key] = _component(name, version, build=openblas)
    except Exception:
        pass
    return (record.get('blas', _component('unrecorded', 'unrecorded')),
            record.get('lapack', _component('unrecorded', 'unrecorded')))


def _package_component(module_name, display_name=None):
    try:
        module = __import__(module_name)
    except Exception:
        return _component(display_name or module_name, 'not installed')
    version = getattr(module, '__version__', None)
    if version is None:
        try:
            from importlib.metadata import version as _version
            version = _version(module_name)
        except Exception:
            version = 'unrecorded'
    return _component(display_name or module_name, version)


def _effective_omp_threads():
    raw = os.environ.get('OMP_NUM_THREADS')
    if raw is None or not raw.strip():
        return 'runtime default (OMP_NUM_THREADS unset)'
    try:
        return int(raw.strip())
    except ValueError:
        return raw.strip()


def collect_environment():
    """Record the environment in which this process is running.

    NumPy and JAX are optional: a missing package is recorded as
    ``not installed`` instead of raising.  The returned record matches
    the schema's ``environment_record`` definition.
    """
    blas, lapack = _numpy_blas_lapack()
    try:
        cpu_count = os.cpu_count()
    except Exception:
        cpu_count = None
    return {
        'python': _component('CPython', platform.python_version()),
        'platform': (f'{platform.system()} {platform.machine()}; '
                     f'cpu_count={cpu_count}'),
        'numpy': _package_component('numpy'),
        'blas': blas,
        'lapack': lapack,
        'jax': _package_component('jax'),
        'jaxlib': _package_component('jaxlib'),
        'xla': _package_component('jaxlib', 'XLA (jaxlib)'),
        'pyscf': _package_component('pyscf'),
        'pyscfad': _package_component('pyscfad'),
        'omp_num_threads_effective': _effective_omp_threads(),
        'precision_mode': ('float64 (declared by the certificate module; '
                           'the certificate arithmetic is pure Python float)'),
    }


def _missing_environment_fields(record):
    missing = []
    for key in ENVIRONMENT_COMPONENTS:
        item = record.get(key)
        if not isinstance(item, dict):
            missing.append(f'{key}')
        elif item.get('version') in (None, '', 'unrecorded'):
            missing.append(f'{key}.version')
    for key in ENVIRONMENT_SCALARS:
        if record.get(key) in (None, ''):
            missing.append(key)
    return missing


def _normalize_environment(case_environment, *, collect_if_auto=True):
    """Fill the measurement/generation records and compute the hash."""
    raw = dict(case_environment or {})
    measurement = raw.get('measurement')
    if not isinstance(measurement, dict):
        raise ValueError('environment.measurement is required')
    generation = raw.get('generation')
    if generation == 'auto':
        generation = collect_environment() if collect_if_auto else None
    if not isinstance(generation, dict):
        raise ValueError('environment.generation must be a record or "auto"')
    thread = raw.get('thread_sensitivity') or {
        'measured': False,
        'tolerance_insensitive_claim': False,
        'note': 'not measured; no thread-insensitivity is claimed',
    }
    if thread.get('tolerance_insensitive_claim') and not thread.get('measured'):
        raise ValueError('a thread-insensitivity claim requires a '
                         'measured thread sensitivity')
    missing = sorted(set(_missing_environment_fields(measurement))
                     | set(_missing_environment_fields(generation)))
    environment = {
        'measurement': measurement,
        'generation': generation,
        'thread_sensitivity': thread,
        'status': 'complete' if not missing else 'partial',
        'missing': missing,
    }
    environment['environment_hash'] = environment_hash(environment)
    return environment


# ---------------------------------------------------------------------------
# Gauge intervals
# ---------------------------------------------------------------------------

def single_orbital_entropy(occupation, double_occupancy):
    """Single-orbital entropy from the four occupation-basis weights.

    Same formula as ``diffmrc.analysis.entropy.single_orbital_entropy``
    for a closed-shell orbital (``n_a = n_b = occupation / 2``).
    """
    n = float(occupation)
    p2 = float(double_occupancy)
    weights = [1.0 - n + p2, n / 2.0 - p2, n / 2.0 - p2, p2]
    weights = [min(max(w, 0.0), 1.0) for w in weights]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    weights = [w / total for w in weights]
    return -sum(w * math.log(w) for w in weights if w > 0.0)


def _rotation_matrix_2x2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s], [s, c]]


def two_electron_singlet_rotation_replicates(natural_occupations,
                                             amplitude_rad,
                                             seeds,
                                             excluded_index=1):
    """Gauge replicates for a two-electron closed-shell singlet.

    In a two-electron, two-orbital singlet the measured natural
    occupations determine the state up to a sign: with
    ``|Psi> = c1 |11> + c2 |22>``, ``n_i = 2 c_i^2``.  A rotation ``U``
    of the two active natural orbitals maps the double-occupancy
    amplitude of the rotated orbital ``k`` to
    ``c1 U[0,k]^2 + c2 U[1,k]^2`` and its occupation to
    ``2 (c1^2 U[0,k]^2 + c2^2 U[1,k]^2)``.  The entropy then follows from
    the four weights.  This is an exact gauge transform of the measured
    state, not a new electronic-structure calculation.

    Parameters
    ----------
    natural_occupations : sequence of two floats
        Measured natural occupations of the active space; they must sum
        to 2 within 1e-6.
    amplitude_rad : float
        Maximum absolute rotation angle.  The angle of each replicate is
        drawn uniformly from ``[-amplitude_rad, amplitude_rad]``.
    seeds : sequence of int
        One seed per independent rotation.
    excluded_index : int
        Index of the rotated orbital tracked as the excluded orbital.
        Only 0 and 1 are valid for this two-orbital model.

    Returns
    -------
    list of dict
        Records with ``rotation_id``, ``seed``, ``matrix``,
        ``matrix_sha256``, ``occupation`` and ``entropy``.
    """
    occ = [float(x) for x in natural_occupations]
    if len(occ) != 2:
        raise ValueError('the two-electron singlet model needs exactly two '
                         f'natural occupations, got {len(occ)}')
    if abs(sum(occ) - 2.0) > 1e-6:
        raise ValueError('the two natural occupations must sum to 2 for a '
                         f'two-electron singlet, got {sum(occ)!r}')
    if not (0.0 <= occ[0] <= 2.0 and 0.0 <= occ[1] <= 2.0):
        raise ValueError(f'natural occupations outside [0, 2]: {occ!r}')
    if int(excluded_index) not in (0, 1):
        raise ValueError('excluded_index must be 0 or 1 in the two-orbital '
                         f'model, got {excluded_index!r}')
    amplitude = float(amplitude_rad)
    if amplitude <= 0.0:
        raise ValueError('rotation amplitude must be positive')
    c1 = math.sqrt(occ[0] / 2.0)
    c2 = math.sqrt(occ[1] / 2.0)
    records = []
    for seed in seeds:
        rng = random.Random(int(seed))
        angle = rng.uniform(-amplitude, amplitude)
        matrix = _rotation_matrix_2x2(angle)
        u1 = matrix[0][int(excluded_index)]
        u2 = matrix[1][int(excluded_index)]
        occupation = 2.0 * (c1 * c1 * u1 * u1 + c2 * c2 * u2 * u2)
        double_occ = (c1 * u1 * u1 + c2 * u2 * u2) ** 2
        all_occupations = []
        for column in (0, 1):
            v1, v2 = matrix[0][column], matrix[1][column]
            all_occupations.append(
                2.0 * (c1 * c1 * v1 * v1 + c2 * c2 * v2 * v2))
        records.append({
            'rotation_id': f'rot-{int(seed)}',
            'seed': int(seed),
            'angle_rad': angle,
            'matrix': matrix,
            'matrix_sha256': sha256_hex(matrix),
            'occupation': occupation,
            'entropy': single_orbital_entropy(occupation, double_occ),
            'all_occupations': all_occupations,
        })
    return records


def _interval(values):
    values = [float(v) for v in values]
    if not values:
        raise ValueError('an interval needs at least one value')
    low, high = min(values), max(values)
    return {'low': low, 'high': high, 'width': high - low,
            'n_values': len(values)}


def _build_canonical_interval(case_interval, target, evidence, tolerances):
    if not isinstance(case_interval, dict):
        raise ValueError('canonical_interval is required')
    protocol = case_interval.get('protocol')
    if not protocol:
        raise ValueError('canonical_interval.protocol is required')
    same_active_space = bool(case_interval.get('same_active_space', True))
    composition_unchanged = bool(
        case_interval.get('composition_unchanged', True))
    if composition_unchanged != same_active_space:
        raise ValueError('same_active_space and composition_unchanged must '
                         'agree: a composition change is a physical '
                         'difference, not gauge spread')

    excluded = [o for o in evidence['excluded_orbitals']
                if o.get('is_lowest_correlation')]
    if len(excluded) != 1:
        raise ValueError('evidence.excluded_orbitals must contain exactly one '
                         'entry with is_lowest_correlation=true')
    excluded = excluded[0]

    if protocol == 'precomputed':
        rotations = list(case_interval.get('rotations') or [])
        for rotation in rotations:
            rotation.setdefault(
                'matrix_sha256',
                sha256_hex(rotation.get('matrix')))
    elif protocol == 'two_electron_singlet_reconstruction':
        rotations = two_electron_singlet_rotation_replicates(
            evidence['natural_occupations'],
            case_interval.get('rotation_amplitude_rad', 0.5),
            case_interval.get('seeds') or [101, 202, 303],
            excluded_index=case_interval.get('excluded_index', 1),
        )
        rotations = [
            {k: r[k] for k in ('rotation_id', 'seed', 'matrix',
                               'matrix_sha256', 'occupation', 'entropy',
                               'all_occupations')}
            | {'energy_change_hartree': 0.0,
               'energy_change_source': ('exact invariance of an '
                                        'active-space rotation')}
            for r in rotations
        ]
    else:
        raise ValueError(f'unknown canonical_interval.protocol: {protocol!r}')

    reference = case_interval.get('reference') or {
        'occupation': excluded['basis_dependent_occupation'],
        'entropy': excluded['entropy'],
    }
    reference = {'occupation': float(reference['occupation']),
                 'entropy': float(reference['entropy'])}

    for rotation in rotations:
        if 'energy_change_hartree' not in rotation:
            rotation['energy_change_hartree'] = 0.0
        if 'matrix_sha256' not in rotation:
            raise ValueError('each precomputed rotation needs matrix_sha256 '
                             'or matrix')

    n_rotations = len(rotations)
    occ_values = [reference['occupation']] + [r['occupation']
                                              for r in rotations]
    ent_values = [reference['entropy']] + [r['entropy'] for r in rotations]
    energy_max = max((abs(float(r['energy_change_hartree']))
                      for r in rotations), default=0.0)
    energy_tol = float(
        tolerances['canonical_interval_energy_max_hartree'])
    quality = {
        'min_rotations_met': n_rotations >= MIN_ROTATIONS,
        'same_active_space': same_active_space,
        'energy_invariant': energy_max <= energy_tol,
    }
    active_space_blob = {
        'ncore': target['active_space']['ncore'],
        'nact': target['active_space']['nact'],
        'nelec': target['active_space']['nelec'],
        'active_indices': target['active_space']['active_indices'],
    }
    return {
        'protocol': protocol,
        'same_active_space': same_active_space,
        'active_space_sha256': sha256_hex(active_space_blob),
        'composition_unchanged': composition_unchanged,
        'n_rotations': n_rotations,
        'rotation_amplitude_rad': case_interval.get('rotation_amplitude_rad'),
        'rotations': rotations,
        'reference': reference,
        'occupation_interval': _interval(occ_values),
        'entropy_interval': _interval(ent_values),
        'energy_invariance_max_hartree': energy_max,
        'quality': quality,
        'discriminating': {
            'occupation': None,
            'entropy': None,
            'candidate_occupation_difference': None,
            'candidate_entropy_difference': None,
            'reason': 'no competitor branches were supplied',
        },
    }


# ---------------------------------------------------------------------------
# Degeneracy
# ---------------------------------------------------------------------------

def classify_degeneracies(values, relative_threshold=1e-6,
                          noise_floor_relative=1e-12,
                          declared_partner_set=None):
    """Classify near-degenerate pairs of a spectrum.

    The criterion is the one fixed by the brainstorm record:

        |v_i - v_j| / max(|v_i|, |v_j|) < max(10 * noise_floor, threshold)

    and the returned ``threshold_used`` says which of the two terms
    dominated.  ``declared_partner_set`` is the set of pairs the caller
    has already resolved (for example by symmetry); detected pairs not
    covered by it are reported as unresolved.
    """
    values = [float(v) for v in values]
    threshold = float(relative_threshold)
    noise = float(noise_floor_relative)
    effective = max(10.0 * noise, threshold)
    threshold_used = ('10x_noise_floor' if 10.0 * noise >= threshold
                      else 'relative_threshold')
    detected = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            scale = max(abs(values[i]), abs(values[j]), 1e-300)
            gap = abs(values[i] - values[j]) / scale
            if gap < effective:
                detected.append({'i': i, 'j': j,
                                 'value_i': values[i],
                                 'value_j': values[j],
                                 'relative_gap': gap})
    declared = {(int(p['i']), int(p['j'])):
                float(p.get('relative_gap', 0.0))
                for p in (declared_partner_set or [])}
    unresolved = [p for p in detected if (p['i'], p['j']) not in declared]
    if not detected:
        status = 'clear'
    elif unresolved:
        status = 'degenerate_unresolved'
    else:
        status = 'degenerate_resolved'
    return {
        'criterion': ('|v_i - v_j| / max(|v_i|, |v_j|) < '
                      'max(10 * noise_floor, relative_threshold)'),
        'relative_threshold': threshold,
        'noise_floor_relative': noise,
        'threshold_used': threshold_used,
        'effective_threshold': effective,
        'partner_set': declared_partner_set or [],
        'n_partners': len(detected),
        'detected_pairs': detected,
        'unresolved_partners': [
            {'i': p['i'], 'j': p['j'],
             'relative_gap': p['relative_gap']} for p in unresolved],
        'status': status,
    }


def _build_degeneracy(case_degeneracy, evidence, tolerances):
    raw = dict(case_degeneracy or {})
    values = raw.get('spectrum', evidence['natural_occupations'])
    result = classify_degeneracies(
        values,
        relative_threshold=raw.get(
            'relative_threshold',
            tolerances['degeneracy_relative_threshold']),
        noise_floor_relative=raw.get(
            'noise_floor_relative',
            tolerances['degeneracy_noise_floor_relative']),
        declared_partner_set=raw.get('declared_partner_set'),
    )
    result['related_measurements'] = raw.get('related_measurements', [])
    result['in_scope'] = bool(raw.get('in_scope', True))
    result['spectrum'] = [float(v) for v in values]
    return result


# ---------------------------------------------------------------------------
# Branch comparison
# ---------------------------------------------------------------------------

def _max_occupation_delta(a, b):
    if len(a) != len(b):
        raise ValueError('occupation spectra must have the same length: '
                         f'{len(a)} vs {len(b)}')
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _compare_branches(target, evidence, branches, canonical, tolerances):
    competitors = list(branches.get('competitors', []))
    energy_tol = float(tolerances['energy_match_hartree'])
    occ_tol = float(tolerances['occupation_match_max'])
    rows = []
    matched = []
    for branch in competitors:
        label = str(branch.get('label', 'unnamed'))
        energy_delta = abs(float(evidence['energy_hartree'])
                           - float(branch['energy_hartree']))
        occ_delta = _max_occupation_delta(
            evidence['natural_occupations'],
            branch['natural_occupations'])
        row = {
            'label': label,
            'energy_delta_hartree': energy_delta,
            'occupation_max_delta': occ_delta,
            'gradient_norm': branch.get('gradient_norm'),
            'curvature_n_negative': (branch.get('curvature') or {}).get(
                'n_negative'),
            'curvature_lowest_eigenvalue': (branch.get('curvature') or {}).get(
                'lowest_eigenvalue'),
            'excluded_orbital_entropy': branch.get('excluded_orbital_entropy'),
            'source': branch.get('source'),
        }
        if branch.get('excluded_orbital_entropy') is not None:
            row['entropy_delta'] = abs(
                float(evidence['excluded_orbitals'][0]['entropy'])
                - float(branch['excluded_orbital_entropy']))
        rows.append(row)
        if energy_delta <= energy_tol and occ_delta <= occ_tol:
            matched.append(label)

    occ_diffs = [r['occupation_max_delta'] for r in rows]
    ent_diffs = [r['entropy_delta'] for r in rows if 'entropy_delta' in r]
    occ_difference = min(occ_diffs) if occ_diffs else None
    ent_difference = min(ent_diffs) if ent_diffs else None
    occ_width = canonical['occupation_interval']['width']
    ent_width = canonical['entropy_interval']['width']
    occ_discriminating = (None if occ_difference is None
                          else occ_width < occ_difference)
    ent_discriminating = (None if ent_difference is None
                          else ent_width < ent_difference)
    reason_bits = []
    if occ_difference is None:
        reason_bits.append('no competitor occupation spectrum was supplied')
    else:
        reason_bits.append(
            f'occupation interval width {occ_width:.6g} vs nearest '
            f'competitor difference {occ_difference:.6g}: '
            f'{"discriminating" if occ_discriminating else "non-discriminating"}')
    if ent_difference is None:
        reason_bits.append('no competitor entropy was supplied')
    else:
        reason_bits.append(
            f'entropy interval width {ent_width:.6g} vs nearest '
            f'competitor difference {ent_difference:.6g}: '
            f'{"discriminating" if ent_discriminating else "non-discriminating"}')
    canonical['discriminating'] = {
        'occupation': occ_discriminating,
        'entropy': ent_discriminating,
        'candidate_occupation_difference': occ_difference,
        'candidate_entropy_difference': ent_difference,
        'reason': '; '.join(reason_bits),
    }
    target_curv = evidence.get('curvature') or {}
    curvature_discriminates = None
    if target_curv.get('n_negative') is not None and any(
            r['curvature_n_negative'] is not None for r in rows):
        target_negative = int(target_curv.get('n_negative', 0)) > 0
        curvature_discriminates = any(
            (int(r['curvature_n_negative']) > 0) != target_negative
            for r in rows if r['curvature_n_negative'] is not None)
    margins = {
        'rows': rows,
        'matched_competitors': matched,
        'min_energy_delta_hartree': min(
            (r['energy_delta_hartree'] for r in rows), default=None),
        'min_occupation_delta': occ_difference,
        'min_entropy_delta': ent_difference,
        'curvature_discriminates': curvature_discriminates,
    }
    return margins


# ---------------------------------------------------------------------------
# Case normalization and certificate construction
# ---------------------------------------------------------------------------

def _require(mapping, key, where):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f'{where}.{key} is required')
    return mapping[key]


def _normalize_target(raw):
    target = dict(raw)
    for key in ('system', 'geometry', 'charge', 'spin', 'multiplicity',
                'basis', 'active_space', 'state', 'branch_label',
                'declared_energy_hartree', 'declared_natural_occupations',
                'identity_criteria'):
        _require(target, key, 'target')
    space = dict(target['active_space'])
    for key in ('ncore', 'nact', 'nelec', 'active_indices'):
        _require(space, key, 'target.active_space')
    if len(space['active_indices']) != int(space['nact']):
        raise ValueError('target.active_space.active_indices length does not '
                         f'match nact: {len(space["active_indices"])} vs '
                         f'{space["nact"]}')
    state = dict(target['state'])
    if state.get('kind') not in ('minimum', 'stationary_point'):
        raise ValueError('target.state.kind must be "minimum" or '
                         '"stationary_point"')
    target['active_space'] = space
    target['state'] = state
    return target


def _normalize_evidence(raw):
    evidence = dict(raw)
    for key in ('energy_hartree', 'orbital_gradient_norm', 'gradient_chart',
                'natural_occupations', 'excluded_orbitals'):
        _require(evidence, key, 'evidence')
    evidence.setdefault(
        'natural_occupation_sum',
        float(sum(float(x) for x in evidence['natural_occupations'])))
    evidence.setdefault('curvature', None)
    evidence.setdefault('overlap_to_reference', None)
    return evidence


def _normalize_branches(raw):
    branches = dict(raw or {})
    competitors = list(branches.get('competitors', []))
    for i, branch in enumerate(competitors):
        for key in ('label', 'energy_hartree', 'natural_occupations'):
            _require(branch, key, f'branches.competitors[{i}]')
    inventory = dict(branches.get('inventory') or {})
    inventory.setdefault('complete', True)
    inventory.setdefault('n_solutions_reported', len(competitors) + 1)
    inventory.setdefault('n_solutions_recorded', len(competitors) + 1)
    inventory.setdefault('missing', 0)
    return {'competitors': competitors, 'inventory': inventory}


def _normalize_versions(raw, *, certificate_module_sha, schema_sha):
    versions = dict(raw or {})
    commit = str(versions.get('code_commit', '')).strip()
    if not commit:
        raise ValueError('versions.code_commit is required; a tolerance '
                         'value without its code commit is not a certificate')
    if len(commit) < 7:
        raise ValueError('versions.code_commit must be at least 7 characters')
    versions.setdefault('code_version', 'unrecorded')
    versions['certificate_module_sha256'] = certificate_module_sha
    versions['schema_sha256'] = schema_sha
    return versions


def _check(name, status, detail, value=None, threshold=None):
    entry = {'name': name, 'status': status, 'detail': detail}
    if value is not None:
        entry['value'] = value
    if threshold is not None:
        entry['threshold'] = threshold
    return entry


def _evaluate(*, target, evidence, canonical, degeneracy, environment,
              versions, tolerances, branches, comparison):
    """Run the certificate checks and return a three-valued verdict.

    ``fail`` is reserved for evidence that positively matches another
    declared branch.  A tolerance violation is ``undetermined``.
    """
    checks = []
    warnings = []
    risks = []

    def add(check):
        checks.append(check)

    # 1. Version binding.
    if versions.get('code_commit') and versions.get(
            'certificate_module_sha256'):
        add(_check('version_binding', 'pass',
                   f'tolerances bound to commit {versions["code_commit"]} '
                   f'and module sha256 {versions["certificate_module_sha256"][:12]}'))
    else:
        add(_check('version_binding', 'undetermined',
                   'code commit or certificate module hash is missing; a '
                   'tolerance without its code identity is not usable'))

    # 2. Environment declaration.
    env_missing = environment.get('missing', [])
    strict_env = bool(tolerances.get('policy', {}).get(
        'require_complete_environment', False))
    if environment.get('status') == 'complete':
        add(_check('environment_declared', 'pass',
                   'measurement and generation environments are complete'))
    elif strict_env:
        add(_check('environment_declared', 'undetermined',
                   'the environment record is partial and the policy '
                   f'requires completeness; missing: {env_missing}',
                   value=env_missing))
    else:
        add(_check('environment_declared', 'pass',
                   'the environment record is partial; missing fields are '
                   f'declared and hashed: {env_missing}',
                   value=env_missing))
        warnings.append(
            'environment record is partial (missing: '
            + ', '.join(env_missing)
            + '); the certificate identity includes the declared gaps')

    # 3. Gradient.
    grad = float(evidence['orbital_gradient_norm'])
    grad_tol = float(tolerances['gradient_norm_max'])
    if grad <= grad_tol:
        add(_check('gradient_converged', 'pass',
                   f'orbital gradient {grad:.3e} <= {grad_tol:.3e}',
                   value=grad, threshold=grad_tol))
    else:
        add(_check('gradient_converged', 'undetermined',
                   f'orbital gradient {grad:.3e} exceeds the tolerance '
                   f'{grad_tol:.3e}; this is a tolerance violation, not a '
                   'branch rejection',
                   value=grad, threshold=grad_tol))

    # 4. Curvature, required only for a declared minimum.
    kind = target['state']['kind']
    curvature = evidence.get('curvature')
    if kind == 'stationary_point':
        add(_check('curvature_minimum', 'not_applicable',
                   'the target is declared as a stationary point, not a '
                   'minimum; curvature is not required'))
    elif not curvature:
        add(_check('curvature_minimum', 'undetermined',
                   'the target is declared as a minimum but no curvature '
                   'report is present'))
    else:
        lowest = float(curvature['lowest_eigenvalue'])
        n_negative = int(curvature['n_negative'])
        curv_tol = float(tolerances['curvature_min_eigenvalue'])
        cross = curvature.get('cross_check') or {}
        cross_text = ''
        if cross:
            cross_text = (f'; cross-check {cross.get("method")}: lowest '
                          f'{float(cross.get("lowest_eigenvalue", 0.0)):.3e}, '
                          f'n_negative {cross.get("n_negative")}')
        if n_negative == 0 and lowest > curv_tol:
            add(_check('curvature_minimum', 'pass',
                       f'no negative curvature; lowest {lowest:.3e} > '
                       f'{curv_tol:.3e}{cross_text}',
                       value=lowest, threshold=curv_tol))
        else:
            add(_check('curvature_minimum', 'undetermined',
                       f'curvature does not certify a minimum: lowest '
                       f'{lowest:.3e}, n_negative {n_negative}, threshold '
                       f'{curv_tol:.3e}{cross_text}',
                       value=lowest, threshold=curv_tol))

    # 5. Target declaration match.
    energy_delta = abs(float(evidence['energy_hartree'])
                       - float(target['declared_energy_hartree']))
    energy_tol = float(tolerances['energy_match_hartree'])
    if energy_delta <= energy_tol:
        add(_check('energy_matches_declaration', 'pass',
                   f'|E - E_declared| = {energy_delta:.3e} <= '
                   f'{energy_tol:.3e}',
                   value=energy_delta, threshold=energy_tol))
    else:
        add(_check('energy_matches_declaration', 'undetermined',
                   f'|E - E_declared| = {energy_delta:.3e} exceeds '
                   f'{energy_tol:.3e}; the measurement does not match the '
                   'declared target',
                   value=energy_delta, threshold=energy_tol))

    occ_delta = _max_occupation_delta(
        evidence['natural_occupations'],
        target['declared_natural_occupations'])
    occ_tol = float(tolerances['occupation_match_max'])
    if occ_delta <= occ_tol:
        add(_check('occupation_matches_declaration', 'pass',
                   f'max natural-occupation change {occ_delta:.3e} <= '
                   f'{occ_tol:.3e}',
                   value=occ_delta, threshold=occ_tol))
    else:
        add(_check('occupation_matches_declaration', 'undetermined',
                   f'max natural-occupation change {occ_delta:.3e} exceeds '
                   f'{occ_tol:.3e}',
                   value=occ_delta, threshold=occ_tol))

    # 6. Canonical interval protocol.
    q = canonical['quality']
    if q['min_rotations_met'] and q['same_active_space'] and q['energy_invariant']:
        add(_check('canonical_interval_protocol', 'pass',
                   f'{canonical["n_rotations"]} independent rotations in the '
                   f'same active space; max energy change '
                   f'{canonical["energy_invariance_max_hartree"]:.3e} Ha',
                   value=canonical['n_rotations'],
                   threshold=MIN_ROTATIONS))
    else:
        add(_check('canonical_interval_protocol', 'undetermined',
                   'the canonical interval does not meet the hard constraint: '
                   f'n_rotations={canonical["n_rotations"]} '
                   f'(min {MIN_ROTATIONS}), '
                   f'same_active_space={q["same_active_space"]}, '
                   f'energy_invariant={q["energy_invariant"]}',
                   value=canonical['n_rotations'],
                   threshold=MIN_ROTATIONS))

    # 7. Discrimination: at least one identity observable must separate
    #    the target from the competitors.
    disc = canonical['discriminating']
    energy_identity = ('energy' in {
        c['observable'] for c in target['identity_criteria']})
    curvature_identity = ('curvature' in {
        c['observable'] for c in target['identity_criteria']})
    overlap_identity = ('overlap' in {
        c['observable'] for c in target['identity_criteria']})
    energy_margin = comparison['min_energy_delta_hartree']
    energy_discriminating = (None if energy_margin is None
                             else energy_margin > energy_tol)
    overlap_value = evidence.get('overlap_to_reference')
    overlap_discriminating = (None if overlap_value is None
                              else float(overlap_value) >= float(
                                  tolerances['overlap_min']))
    available = []
    if disc['occupation']:
        available.append('excluded_orbital_occupation')
    if disc['entropy']:
        available.append('excluded_orbital_entropy')
    if energy_identity and energy_discriminating:
        available.append('energy')
    if curvature_identity and comparison.get('curvature_discriminates'):
        available.append('curvature')
    if overlap_identity and overlap_discriminating:
        available.append('overlap')
    if not branches['competitors']:
        add(_check('observable_discriminates', 'not_applicable',
                   'no competitor branches were supplied; discrimination '
                   'between candidates is not applicable'))
    elif available:
        add(_check('observable_discriminates', 'pass',
                   'the target is separable from the competitors by: '
                   + ', '.join(available),
                   value=available))
    else:
        add(_check('observable_discriminates', 'undetermined',
                   'no identity observable separates the target from the '
                   'competitors: the canonical interval is wider than the '
                   'candidate difference and no energy, curvature or '
                   'overlap criterion discriminates',
                   value=available))

    # 8. Positive match to another branch -> reject.
    if comparison['matched_competitors']:
        add(_check('branch_identity', 'fail',
                   'the evidence matches declared competitor branch(es): '
                   + ', '.join(comparison['matched_competitors']),
                   value=comparison['matched_competitors']))
    else:
        add(_check('branch_identity', 'pass',
                   'the evidence does not match any declared competitor '
                   'within the energy and occupation tolerances'))

    # 9. Branch inventory completeness.
    inventory = branches['inventory']
    allow_incomplete = bool(tolerances.get('policy', {}).get(
        'allow_incomplete_branch_inventory', True))
    if inventory.get('complete'):
        add(_check('branch_inventory', 'pass',
                   f'{inventory.get("n_solutions_recorded")} of '
                   f'{inventory.get("n_solutions_reported")} branches are '
                   'recorded; the inventory is declared complete'))
    elif allow_incomplete:
        add(_check('branch_inventory', 'pass',
                   f'the inventory is incomplete: '
                   f'{inventory.get("n_solutions_recorded")} of '
                   f'{inventory.get("n_solutions_reported")} branches are '
                   f'recorded (missing {inventory.get("missing")}); the '
                   'policy permits the certificate with a residual risk',
                   value=inventory.get('missing')))
        risks.append(
            'branch inventory is incomplete: '
            f'{inventory.get("missing")} branch(es) have no per-branch '
            'record; an unrecorded competitor closer than the energy '
            'tolerance cannot be excluded')
    else:
        add(_check('branch_inventory', 'undetermined',
                   'the branch inventory is incomplete and the policy '
                   'requires a complete inventory',
                   value=inventory.get('missing')))

    # 10. Degeneracy.
    if degeneracy['status'] == 'degenerate_unresolved':
        add(_check('degeneracy_resolved', 'undetermined',
                   f'{len(degeneracy["unresolved_partners"])} degenerate '
                   'partner pair(s) are not covered by the declared partner '
                   'set',
                   value=degeneracy['unresolved_partners']))
    elif degeneracy['status'] == 'degenerate_resolved':
        add(_check('degeneracy_resolved', 'pass',
                   f'{degeneracy["n_partners"]} degenerate partner pair(s) '
                   f'are covered by the declared set; threshold used: '
                   f'{degeneracy["threshold_used"]} '
                   f'({degeneracy["effective_threshold"]:.3e})',
                   value=degeneracy['n_partners']))
    else:
        add(_check('degeneracy_resolved', 'pass',
                   'no degenerate partner pair was detected above the '
                   f'effective threshold {degeneracy["effective_threshold"]:.3e}',
                   value=0))

    # 11. Thread sensitivity must not be silently assumed away.
    thread = environment.get('thread_sensitivity') or {}
    if thread.get('tolerance_insensitive_claim'):
        if thread.get('measured') and thread.get('difference') is not None \
                and float(thread['difference']) <= energy_tol:
            add(_check('thread_sensitivity', 'pass',
                       'the measured thread-induced difference is within the '
                       'energy tolerance',
                       value=thread['difference'], threshold=energy_tol))
        else:
            add(_check('thread_sensitivity', 'undetermined',
                       'a thread-insensitivity claim is made without a '
                       'measured difference below the energy tolerance',
                       value=thread.get('difference'), threshold=energy_tol))
    else:
        add(_check('thread_sensitivity', 'pass',
                   'no thread-insensitivity is claimed; the effective thread '
                   'count is part of the hashed environment',
                   value=environment['measurement'].get(
                       'omp_num_threads_effective')))

    failed = [c for c in checks if c['status'] == 'fail']
    undetermined = [c for c in checks if c['status'] == 'undetermined']
    if failed:
        value = 'reject'
    elif undetermined:
        value = 'undetermined'
    else:
        value = 'accept'

    reasons = []
    for check in failed + undetermined:
        reasons.append(f'{check["name"]}: {check["detail"]}')
    if value == 'accept':
        reasons.append(
            'all required checks passed; the evidence identifies the '
            f'declared branch {target["branch_label"]}')
    if disc['occupation'] is False:
        warnings.append('the excluded-orbital occupation is non-discriminating '
                        '(canonical interval wider than the candidate '
                        'difference) and is not used for acceptance')
    if disc['entropy'] is False:
        warnings.append('the excluded-orbital entropy is non-discriminating '
                        '(canonical interval wider than the candidate '
                        'difference) and is not used for acceptance')
    return {
        'value': value,
        'reasons': reasons,
        'warnings': warnings,
        'residual_risks': risks,
        'checks': checks,
    }


def build_certificate(case, *, created_utc=None, certificate_module_sha=None,
                      schema_sha=None, collect_generation_environment=True):
    """Build a certificate from a case record.

    Parameters
    ----------
    case : dict
        Case record with ``target``, ``evidence``, ``branches``,
        ``canonical_interval``, ``degeneracy``, ``tolerances``,
        ``versions``, ``environment`` and ``provenance``.
    created_utc : str, optional
        ISO 8601 timestamp.  It is excluded from the payload hash, so a
        fixed value is not needed for reproducible identity.
    certificate_module_sha, schema_sha : str, optional
        Overrides used by tests; by default they are read from the files.
    """
    target = _normalize_target(case['target'])
    evidence = _normalize_evidence(case['evidence'])
    branches = _normalize_branches(case.get('branches'))
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances.update(case.get('tolerances') or {})
    policy = dict(DEFAULT_TOLERANCES['policy'])
    policy.update((case.get('tolerances') or {}).get('policy') or {})
    tolerances['policy'] = policy

    module_sha = certificate_module_sha or file_sha256(MODULE_PATH)
    schema_sha = schema_sha or file_sha256(SCHEMA_PATH)
    versions = _normalize_versions(case.get('versions'),
                                   certificate_module_sha=module_sha,
                                   schema_sha=schema_sha)
    environment = _normalize_environment(
        case.get('environment'),
        collect_if_auto=collect_generation_environment)
    tolerance_digest = tolerance_id(tolerances, versions)
    tolerances['tolerance_id'] = tolerance_digest

    canonical = _build_canonical_interval(
        case.get('canonical_interval'), target, evidence, tolerances)
    degeneracy = _build_degeneracy(
        case.get('degeneracy'), evidence, tolerances)
    comparison = _compare_branches(
        target, evidence, branches, canonical, tolerances)
    verdict = _evaluate(
        target=target, evidence=evidence, canonical=canonical,
        degeneracy=degeneracy, environment=environment, versions=versions,
        tolerances=tolerances, branches=branches, comparison=comparison)

    certificate = {
        'schema_version': SCHEMA_VERSION,
        'kind': KIND,
        'created_utc': created_utc or datetime.now(timezone.utc).isoformat(
            timespec='seconds'),
        'target': target,
        'evidence': evidence,
        'branches': branches,
        'canonical_interval': canonical,
        'degeneracy': degeneracy,
        'tolerances': tolerances,
        'versions': versions,
        'environment': environment,
        'verdict': verdict,
        'provenance': dict(case.get('provenance') or {
            'sources': [],
            'generated_by': 'certificate.py',
        }),
    }
    digest = payload_hash(certificate)
    certificate['payload_hash'] = digest
    certificate['certificate_id'] = 'sha256:' + digest
    return certificate


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class CertificateInvalid(Exception):
    """Raised when a document cannot be read as a certificate."""


def _require_keys(mapping, keys, where):
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise CertificateInvalid(f'{where} is missing key(s): {missing}')


def _structural_check(cert):
    _require_keys(cert, ('schema_version', 'kind', 'certificate_id',
                         'payload_hash', 'created_utc', 'target', 'evidence',
                         'branches', 'canonical_interval', 'degeneracy',
                         'tolerances', 'versions', 'environment', 'verdict',
                         'provenance'), 'certificate')
    if cert['schema_version'] != SCHEMA_VERSION:
        raise CertificateInvalid(
            f'unsupported schema_version: {cert["schema_version"]!r}')
    if cert['kind'] != KIND:
        raise CertificateInvalid(f'unsupported kind: {cert["kind"]!r}')
    for key in ('target', 'evidence', 'branches', 'canonical_interval',
                'degeneracy', 'tolerances', 'versions', 'environment',
                'verdict', 'provenance'):
        if not isinstance(cert[key], dict):
            raise CertificateInvalid(f'{key} must be an object')
    for key in ('system', 'basis', 'branch_label', 'declared_energy_hartree',
                'declared_natural_occupations', 'active_space', 'state',
                'identity_criteria'):
        if key not in cert['target']:
            raise CertificateInvalid(f'target is missing {key!r}')
    for key in ('energy_hartree', 'orbital_gradient_norm',
                'natural_occupations', 'excluded_orbitals'):
        if key not in cert['evidence']:
            raise CertificateInvalid(f'evidence is missing {key!r}')
    for key in ('occupation_interval', 'entropy_interval', 'rotations',
                'n_rotations', 'quality', 'discriminating'):
        if key not in cert['canonical_interval']:
            raise CertificateInvalid(
                f'canonical_interval is missing {key!r}')
    if cert['verdict'].get('value') not in ('accept', 'reject',
                                            'undetermined'):
        raise CertificateInvalid('verdict.value must be accept, reject or '
                                 'undetermined')


def _schema_validate_optional(cert):
    """Validate against the shipped JSON Schema when jsonschema is present."""
    try:
        import jsonschema
    except Exception:
        return {'available': False, 'valid': None, 'errors': []}
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(cert), key=lambda e: list(e.path))
        return {
            'available': True,
            'valid': not errors,
            'errors': [f'{"/".join(map(str, e.path))}: {e.message}'
                       for e in errors[:20]],
        }
    except Exception as exc:
        return {'available': True, 'valid': False,
                'errors': [f'{type(exc).__name__}: {exc}']}


def _recompute(cert):
    """Recompute the derived fields and the verdict from the payload.

    The verifier does not trust the recorded ``quality`` flags, the
    interval endpoints, the degeneracy status or the environment status:
    all of them are re-derived from the primary records.
    """
    tolerances = dict(cert['tolerances'])
    environment = dict(cert['environment'])
    missing = sorted(
        set(_missing_environment_fields(environment.get('measurement') or {}))
        | set(_missing_environment_fields(environment.get('generation') or {})))
    environment['missing'] = missing
    environment['status'] = 'complete' if not missing else 'partial'
    environment['environment_hash'] = environment_hash(environment)
    versions = dict(cert['versions'])
    tolerances['tolerance_id'] = tolerance_id(tolerances, versions)

    canonical = dict(cert['canonical_interval'])
    rotations = list(canonical.get('rotations') or [])
    reference = canonical.get('reference') or {'occupation': 0.0,
                                               'entropy': 0.0}
    canonical['n_rotations'] = len(rotations)
    canonical['occupation_interval'] = _interval(
        [reference['occupation']] + [float(r['occupation'])
                                     for r in rotations])
    canonical['entropy_interval'] = _interval(
        [reference['entropy']] + [float(r['entropy']) for r in rotations])
    energy_max = max((abs(float(r.get('energy_change_hartree', 0.0)))
                      for r in rotations), default=0.0)
    canonical['energy_invariance_max_hartree'] = energy_max
    canonical['quality'] = {
        'min_rotations_met': len(rotations) >= MIN_ROTATIONS,
        'same_active_space': bool(canonical.get('same_active_space')),
        'energy_invariant': energy_max <= float(
            tolerances['canonical_interval_energy_max_hartree']),
    }

    degeneracy = _build_degeneracy(
        {
            'spectrum': cert['degeneracy'].get(
                'spectrum', cert['evidence']['natural_occupations']),
            'declared_partner_set': cert['degeneracy'].get('partner_set', []),
            'relative_threshold': cert['degeneracy'].get(
                'relative_threshold',
                tolerances['degeneracy_relative_threshold']),
            'noise_floor_relative': cert['degeneracy'].get(
                'noise_floor_relative',
                tolerances['degeneracy_noise_floor_relative']),
            'related_measurements': cert['degeneracy'].get(
                'related_measurements', []),
            'in_scope': cert['degeneracy'].get('in_scope', True),
        },
        cert['evidence'], tolerances)

    branches = cert['branches']
    comparison = _compare_branches(
        cert['target'], cert['evidence'], branches, canonical, tolerances)
    verdict = _evaluate(
        target=cert['target'], evidence=cert['evidence'], canonical=canonical,
        degeneracy=degeneracy, environment=environment,
        versions=versions, tolerances=tolerances, branches=branches,
        comparison=comparison)
    return canonical, environment, tolerances, degeneracy, verdict


def validate_certificate(cert, *, strict_environment=None,
                         check_module_sha=False):
    """Validate a certificate and recompute its verdict.

    Returns a report dict.  ``valid`` is True when the document is
    structurally readable and all recorded hashes match the recomputed
    ones.  The recomputed verdict is authoritative; a mismatch with the
    recorded verdict is reported as an error.
    """
    errors = []
    warnings = []
    try:
        _structural_check(cert)
    except CertificateInvalid as exc:
        return {'valid': False, 'schema_ok': False,
                'errors': [str(exc)], 'warnings': [],
                'verdict_recorded': None, 'verdict_recomputed': None,
                'checks': []}

    schema_report = _schema_validate_optional(cert)
    if schema_report['available'] and not schema_report['valid']:
        errors.extend(schema_report['errors'])

    recorded_payload = cert.get('payload_hash')
    recomputed_payload = payload_hash(cert)
    payload_ok = recorded_payload == recomputed_payload
    if not payload_ok:
        errors.append(
            f'payload hash mismatch: recorded {recorded_payload}, '
            f'recomputed {recomputed_payload}')

    recorded_env = cert['environment'].get('environment_hash')
    recomputed_env = environment_hash(cert['environment'])
    env_ok = recorded_env == recomputed_env
    if not env_ok:
        errors.append(
            f'environment hash mismatch: recorded {recorded_env}, '
            f'recomputed {recomputed_env}')

    recorded_tol = cert['tolerances'].get('tolerance_id')
    recomputed_tol = tolerance_id(cert['tolerances'], cert['versions'])
    tol_ok = recorded_tol == recomputed_tol
    if not tol_ok:
        errors.append(
            f'tolerance id mismatch: recorded {recorded_tol}, '
            f'recomputed {recomputed_tol}')

    if check_module_sha:
        expected = file_sha256(MODULE_PATH)
        recorded = cert['versions'].get('certificate_module_sha256')
        if recorded != expected:
            errors.append(
                'certificate module hash does not match the running module: '
                f'recorded {recorded}, current {expected}')

    recomputed_canonical, recomputed_environment, recomputed_tolerances, \
        recomputed_degeneracy, recomputed_verdict = _recompute(cert)

    interval_ok = True
    recorded_interval = cert['canonical_interval']
    for key in ('occupation_interval', 'entropy_interval'):
        recorded = recorded_interval.get(key, {})
        recomputed = recomputed_canonical[key]
        if any(abs(float(recorded.get(k, 0.0)) - recomputed[k]) > 1e-12
               for k in ('low', 'high', 'width')):
            interval_ok = False
            errors.append(
                f'{key} is inconsistent with the recorded rotations and '
                f'reference: recorded {recorded}, recomputed {recomputed}')
    for i, rotation in enumerate(recorded_interval.get('rotations') or []):
        matrix = rotation.get('matrix')
        recorded_hash = rotation.get('matrix_sha256')
        if matrix is not None:
            expected = sha256_hex(matrix)
            if recorded_hash != expected:
                interval_ok = False
                errors.append(
                    f'rotation {i} matrix hash mismatch: recorded '
                    f'{recorded_hash}, recomputed {expected}')

    recorded_verdict = cert['verdict'].get('value')
    verdict_ok = recorded_verdict == recomputed_verdict['value']
    if not verdict_ok:
        errors.append(
            f'recorded verdict {recorded_verdict!r} differs from the '
            f'recomputed verdict {recomputed_verdict["value"]!r}')

    if strict_environment is not None:
        policy = dict(recomputed_tolerances.get('policy') or {})
        policy['require_complete_environment'] = bool(strict_environment)
        recomputed_tolerances['policy'] = policy
        recomputed_verdict = _evaluate(
            target=cert['target'], evidence=cert['evidence'],
            canonical=recomputed_canonical,
            degeneracy=recomputed_degeneracy,
            environment=recomputed_environment, versions=cert['versions'],
            tolerances=recomputed_tolerances, branches=cert['branches'],
            comparison=_compare_branches(
                cert['target'], cert['evidence'], cert['branches'],
                recomputed_canonical, recomputed_tolerances))

    if cert['environment'].get('status') == 'partial':
        warnings.append('the environment record is partial; the missing '
                        'fields are part of the hashed identity')
    if not cert['branches']['inventory'].get('complete'):
        warnings.append('the branch inventory is incomplete; an unrecorded '
                        'competitor closer than the energy tolerance cannot '
                        'be excluded')
    for warning in cert['verdict'].get('warnings', []):
        if warning not in warnings:
            warnings.append(warning)

    valid = not errors and (schema_report['valid'] is not False)
    return {
        'valid': valid,
        'schema_ok': schema_report['valid'],
        'payload_hash_ok': payload_ok,
        'environment_hash_ok': env_ok,
        'tolerance_id_ok': tol_ok,
        'interval_ok': interval_ok,
        'verdict_recorded': recorded_verdict,
        'verdict_recomputed': recomputed_verdict['value'],
        'verdict_consistent': verdict_ok,
        'checks': recomputed_verdict['checks'],
        'errors': errors,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_for_json(value), indent=2,
                               ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def _cmd_generate(args):
    case = load_json(args.case)
    certificate = build_certificate(
        case, collect_generation_environment=not args.no_collect_environment)
    write_json(args.output, certificate)
    print(f'wrote {args.output}')
    print(f'certificate_id {certificate["certificate_id"]}')
    print(f'verdict {certificate["verdict"]["value"]}')
    for reason in certificate['verdict']['reasons']:
        print(f'  {reason}')
    return 0


def _cmd_verify(args):
    cert = load_json(args.certificate)
    report = validate_certificate(
        cert, strict_environment=args.strict_environment,
        check_module_sha=args.check_module_sha)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f'valid            {report["valid"]}')
        print(f'verdict recorded {report["verdict_recorded"]}')
        print(f'verdict recomputed {report["verdict_recomputed"]}')
        print(f'payload hash ok  {report["payload_hash_ok"]}')
        print(f'environment ok   {report["environment_hash_ok"]}')
        print(f'tolerance id ok  {report["tolerance_id_ok"]}')
        for error in report['errors']:
            print(f'error: {error}')
        for warning in report['warnings']:
            print(f'warning: {warning}')
    if not report['valid']:
        return 3
    return {'accept': 0, 'reject': 1, 'undetermined': 2}.get(
        report['verdict_recomputed'], 3)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate and verify diffmrc branch certificates.')
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate', help='build a certificate from a case')
    gen.add_argument('case', help='case JSON file')
    gen.add_argument('-o', '--output', required=True,
                     help='output certificate JSON file')
    gen.add_argument('--no-collect-environment', action='store_true',
                     help='do not probe the local environment for the '
                          'generation record')
    gen.set_defaults(func=_cmd_generate)
    ver = sub.add_parser('verify', help='validate a certificate')
    ver.add_argument('certificate', help='certificate JSON file')
    ver.add_argument('--json', action='store_true', help='JSON report')
    ver.add_argument('--strict-environment', action='store_true',
                     default=None, help='require a complete environment')
    ver.add_argument('--check-module-sha', action='store_true',
                     help='require the recorded module hash to match the '
                          'running module')
    ver.set_defaults(func=_cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
