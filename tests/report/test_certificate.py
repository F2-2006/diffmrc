"""Tests for the branch-certificate module.

The tests use the two shipped example cases (N2/cc-pVDZ CAS(2,2) with
three stationary points, CO/6-31G CAS(2,2) with the A/B pair) and the
24-pair degeneracy fixture.  No electronic-structure calculation is run.
"""
import copy
import json
from pathlib import Path

import pytest

from diffmrc.report import certificate as C

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
N2_CASE = FIXTURES / 'n2_ccpvdz_cas22.case.json'
CO_CASE = FIXTURES / 'co_631g_cas22.case.json'
N2_CERT = FIXTURES / 'n2_ccpvdz_cas22.certificate.json'
CO_CERT = FIXTURES / 'co_631g_cas22.certificate.json'
SCHEMA = C.SCHEMA_PATH

FIXED_GENERATION = {
    'python': {'name': 'CPython', 'version': '3.14.2'},
    'platform': 'Windows AMD64; cpu_count=24',
    'numpy': {'name': 'numpy', 'version': '2.4.2'},
    'blas': {'name': 'scipy-openblas', 'version': '0.3.31.dev'},
    'lapack': {'name': 'scipy-openblas', 'version': '0.3.31.dev'},
    'jax': {'name': 'jax', 'version': 'not installed'},
    'jaxlib': {'name': 'jaxlib', 'version': 'not installed'},
    'xla': {'name': 'XLA (jaxlib)', 'version': 'not installed'},
    'pyscf': {'name': 'pyscf', 'version': 'not installed'},
    'pyscfad': {'name': 'pyscfad', 'version': 'not installed'},
    'omp_num_threads_effective': 'runtime default (OMP_NUM_THREADS unset)',
    'precision_mode': 'float64',
}


def load(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def case_with_fixed_generation(path):
    case = load(path)
    case['environment']['generation'] = copy.deepcopy(FIXED_GENERATION)
    return case


@pytest.fixture(scope='module')
def n2_case():
    return case_with_fixed_generation(N2_CASE)


@pytest.fixture(scope='module')
def co_case():
    return case_with_fixed_generation(CO_CASE)


# ---------------------------------------------------------------------------
# Shipped examples
# ---------------------------------------------------------------------------

def test_n2_example_certificate_is_valid_and_accepts():
    cert = load(N2_CERT)
    report = C.validate_certificate(cert)
    assert report['valid'], report['errors']
    assert report['verdict_recomputed'] == 'accept'
    assert cert['canonical_interval']['n_rotations'] == 3
    assert cert['degeneracy']['status'] == 'clear'


def test_co_example_certificate_is_valid_and_accepts():
    cert = load(CO_CERT)
    report = C.validate_certificate(cert)
    assert report['valid'], report['errors']
    assert report['verdict_recomputed'] == 'accept'
    assert cert['canonical_interval']['n_rotations'] == 3
    assert cert['branches']['inventory']['complete'] is True


def test_examples_validate_against_the_json_schema():
    jsonschema = pytest.importorskip('jsonschema')
    schema = load(SCHEMA)
    validator = jsonschema.Draft202012Validator(schema)
    for path in (N2_CERT, CO_CERT):
        errors = sorted(validator.iter_errors(load(path)),
                        key=lambda e: list(e.path))
        assert not errors, [f'{"/".join(map(str, e.path))}: {e.message}'
                            for e in errors]


# ---------------------------------------------------------------------------
# Scenario 1: identical input gives an identical hash
# ---------------------------------------------------------------------------

def test_same_input_gives_the_same_hash_even_with_a_different_timestamp(n2_case):
    first = C.build_certificate(
        n2_case, created_utc='2026-11-05T00:00:00+00:00')
    second = C.build_certificate(
        n2_case, created_utc='2026-11-06T12:34:56+00:00')
    assert first['payload_hash'] == second['payload_hash']
    assert first['certificate_id'] == second['certificate_id']
    assert first['created_utc'] != second['created_utc']


def test_regenerating_with_the_recorded_generation_environment_is_stable():
    shipped = load(N2_CERT)
    case = load(N2_CASE)
    case['environment']['generation'] = copy.deepcopy(
        shipped['environment']['generation'])
    first = C.build_certificate(case, created_utc='2026-11-05T00:00:00+00:00')
    second = C.build_certificate(case, created_utc='2026-11-07T00:00:00+00:00')
    assert first['certificate_id'] == second['certificate_id']


# ---------------------------------------------------------------------------
# Scenario 2: tampering with a tolerance or an environment field changes
#             the hash
# ---------------------------------------------------------------------------

def test_changing_a_tolerance_changes_the_hash_and_the_tolerance_id(n2_case):
    base = C.build_certificate(n2_case)
    changed = copy.deepcopy(n2_case)
    changed['tolerances']['gradient_norm_max'] = 1e-6
    other = C.build_certificate(changed)
    assert other['certificate_id'] != base['certificate_id']
    assert other['tolerances']['tolerance_id'] != base['tolerances']['tolerance_id']
    assert other['payload_hash'] != base['payload_hash']


def test_changing_an_environment_field_changes_the_hash(co_case):
    base = C.build_certificate(co_case)
    changed = copy.deepcopy(co_case)
    changed['environment']['measurement']['omp_num_threads_effective'] = 4
    other = C.build_certificate(changed)
    assert other['certificate_id'] != base['certificate_id']
    assert other['environment']['environment_hash'] != \
        base['environment']['environment_hash']


def test_changing_the_blas_version_changes_the_hash(co_case):
    base = C.build_certificate(co_case)
    changed = copy.deepcopy(co_case)
    changed['environment']['measurement']['blas']['version'] = '2024.0'
    other = C.build_certificate(changed)
    assert other['certificate_id'] != base['certificate_id']


def test_changing_the_code_commit_changes_the_tolerance_id(co_case):
    base = C.build_certificate(co_case)
    changed = copy.deepcopy(co_case)
    changed['versions']['code_commit'] = '0' * 40
    other = C.build_certificate(changed)
    assert other['tolerances']['tolerance_id'] != \
        base['tolerances']['tolerance_id']
    assert other['certificate_id'] != base['certificate_id']


def test_a_tampered_payload_is_detected_by_the_verifier():
    cert = load(CO_CERT)
    tampered = copy.deepcopy(cert)
    tampered['evidence']['energy_hartree'] += 1e-6
    report = C.validate_certificate(tampered)
    assert report['valid'] is False
    assert report['payload_hash_ok'] is False
    assert any('payload hash mismatch' in e for e in report['errors'])


def test_a_tampered_environment_hash_is_detected_by_the_verifier():
    cert = load(CO_CERT)
    tampered = copy.deepcopy(cert)
    tampered['environment']['measurement']['omp_num_threads_effective'] = 1
    report = C.validate_certificate(tampered)
    assert report['valid'] is False
    assert report['environment_hash_ok'] is False


# ---------------------------------------------------------------------------
# Scenario 3: a tolerance violation is undetermined, not the nearest branch
# ---------------------------------------------------------------------------

def test_gradient_tolerance_violation_is_undetermined_not_reject(n2_case):
    changed = copy.deepcopy(n2_case)
    changed['tolerances']['gradient_norm_max'] = 1e-9
    cert = C.build_certificate(changed)
    assert cert['verdict']['value'] == 'undetermined'
    gradient = [c for c in cert['verdict']['checks']
                if c['name'] == 'gradient_converged'][0]
    assert gradient['status'] == 'undetermined'
    assert not [c for c in cert['verdict']['checks']
                if c['name'] == 'branch_identity' and c['status'] == 'fail']


def test_too_few_rotations_is_undetermined(co_case):
    changed = copy.deepcopy(co_case)
    changed['canonical_interval']['seeds'] = [101, 202]
    cert = C.build_certificate(changed)
    assert cert['canonical_interval']['n_rotations'] == 2
    assert cert['verdict']['value'] == 'undetermined'
    check = [c for c in cert['verdict']['checks']
             if c['name'] == 'canonical_interval_protocol'][0]
    assert check['status'] == 'undetermined'


def test_a_composition_change_is_not_treated_as_gauge_spread(co_case):
    changed = copy.deepcopy(co_case)
    changed['canonical_interval']['same_active_space'] = False
    changed['canonical_interval']['composition_unchanged'] = False
    cert = C.build_certificate(changed)
    assert cert['verdict']['value'] == 'undetermined'
    check = [c for c in cert['verdict']['checks']
             if c['name'] == 'canonical_interval_protocol'][0]
    assert check['status'] == 'undetermined'


def test_contradictory_composition_flags_are_refused(co_case):
    changed = copy.deepcopy(co_case)
    changed['canonical_interval']['same_active_space'] = True
    changed['canonical_interval']['composition_unchanged'] = False
    with pytest.raises(ValueError):
        C.build_certificate(changed)


def test_strict_environment_policy_turns_a_partial_record_into_undetermined(
        co_case):
    changed = copy.deepcopy(co_case)
    changed['tolerances']['policy']['require_complete_environment'] = True
    cert = C.build_certificate(changed)
    assert cert['environment']['status'] == 'partial'
    assert cert['verdict']['value'] == 'undetermined'


def test_strict_branch_inventory_policy_turns_n2_into_undetermined(n2_case):
    changed = copy.deepcopy(n2_case)
    changed['tolerances']['policy']['allow_incomplete_branch_inventory'] = False
    cert = C.build_certificate(changed)
    assert cert['verdict']['value'] == 'undetermined'


# ---------------------------------------------------------------------------
# Rejection: evidence that positively matches another declared branch
# ---------------------------------------------------------------------------

def test_evidence_matching_a_competitor_branch_is_rejected(co_case):
    changed = copy.deepcopy(co_case)
    competitor = changed['branches']['competitors'][0]
    changed['evidence']['energy_hartree'] = competitor['energy_hartree']
    changed['evidence']['natural_occupations'] = list(
        competitor['natural_occupations'])
    changed['evidence']['orbital_gradient_norm'] = competitor['gradient_norm']
    changed['evidence']['curvature'] = copy.deepcopy(competitor['curvature'])
    changed['evidence']['excluded_orbitals'][0]['natural_occupation'] = \
        competitor['natural_occupations'][1]
    changed['evidence']['excluded_orbitals'][0][
        'basis_dependent_occupation'] = competitor['natural_occupations'][1]
    changed['evidence']['excluded_orbitals'][0]['entropy'] = \
        C.single_orbital_entropy(competitor['natural_occupations'][1],
                                 competitor['natural_occupations'][1] / 2.0)
    cert = C.build_certificate(changed)
    assert cert['verdict']['value'] == 'reject'
    identity = [c for c in cert['verdict']['checks']
                if c['name'] == 'branch_identity'][0]
    assert identity['status'] == 'fail'
    assert 'CO-6-31G-A' in identity['detail']


# ---------------------------------------------------------------------------
# Degeneracy
# ---------------------------------------------------------------------------

def test_the_co_24_pair_fixture_is_classified_as_resolved():
    fixture = load(FIXTURES / 'co_degenerate_pairs.json')
    declared = [{'i': i, 'j': j, 'relative_gap': 0.0}
                for i, j in fixture['pairs']]
    result = C.classify_degeneracies(
        fixture['values'], relative_threshold=1e-6,
        noise_floor_relative=1e-12, declared_partner_set=declared)
    assert result['n_partners'] == 24
    assert result['status'] == 'degenerate_resolved'
    assert result['threshold_used'] == 'relative_threshold'
    assert result['effective_threshold'] == pytest.approx(1e-6)
    assert not result['unresolved_partners']


def test_the_co_24_pair_fixture_is_unresolved_without_a_declared_set():
    fixture = load(FIXTURES / 'co_degenerate_pairs.json')
    result = C.classify_degeneracies(
        fixture['values'], relative_threshold=1e-6,
        noise_floor_relative=1e-12, declared_partner_set=[])
    assert result['n_partners'] == 24
    assert result['status'] == 'degenerate_unresolved'
    assert len(result['unresolved_partners']) == 24


def test_degeneracy_threshold_follows_the_larger_term():
    result = C.classify_degeneracies(
        [1.0, 1.0000005], relative_threshold=1e-6,
        noise_floor_relative=1e-12, declared_partner_set=[])
    assert result['n_partners'] == 1
    assert result['threshold_used'] == 'relative_threshold'
    result = C.classify_degeneracies(
        [1.0, 1.0000005], relative_threshold=1e-9,
        noise_floor_relative=1e-5, declared_partner_set=[])
    assert result['threshold_used'] == '10x_noise_floor'
    assert result['effective_threshold'] == pytest.approx(1e-4)


def test_an_unresolved_degeneracy_makes_the_certificate_undetermined(co_case):
    changed = copy.deepcopy(co_case)
    changed['degeneracy']['spectrum'] = [1.9, 1.9, 0.1, 0.1]
    changed['degeneracy']['declared_partner_set'] = []
    cert = C.build_certificate(changed)
    assert cert['degeneracy']['status'] == 'degenerate_unresolved'
    assert cert['verdict']['value'] == 'undetermined'


def test_the_co_certificate_records_the_out_of_scope_24_pair_measurement():
    cert = load(CO_CERT)
    related = cert['degeneracy']['related_measurements']
    assert len(related) == 1
    assert related[0]['n_partner_pairs'] == 24
    assert related[0]['in_scope'] is False
    assert 'CAS(6,6)' in related[0]['space']


# ---------------------------------------------------------------------------
# Environment and tolerance provenance
# ---------------------------------------------------------------------------

def test_a_thread_insensitivity_claim_without_a_measurement_is_refused(co_case):
    changed = copy.deepcopy(co_case)
    changed['environment']['thread_sensitivity'] = {
        'measured': False,
        'tolerance_insensitive_claim': True,
    }
    with pytest.raises(ValueError):
        C.build_certificate(changed)


def test_a_measured_thread_difference_larger_than_the_tolerance_is_recorded(
        co_case):
    cert = C.build_certificate(co_case)
    thread = cert['environment']['thread_sensitivity']
    assert thread['measured'] is True
    assert thread['tolerance_insensitive_claim'] is False
    assert thread['difference'] > cert['tolerances']['energy_match_hartree']


def test_a_missing_code_commit_is_refused(co_case):
    changed = copy.deepcopy(co_case)
    del changed['versions']['code_commit']
    with pytest.raises(ValueError):
        C.build_certificate(changed)


# ---------------------------------------------------------------------------
# Canonical interval arithmetic
# ---------------------------------------------------------------------------

def test_rotation_replicates_preserve_the_spectrum_and_record_three_angles():
    replicates = C.two_electron_singlet_rotation_replicates(
        [1.953255, 0.046745], 0.5, [11, 22, 33])
    assert len(replicates) == 3
    assert len({r['matrix_sha256'] for r in replicates}) == 3
    for record in replicates:
        assert 0.0 <= record['occupation'] <= 2.0
        assert record['entropy'] >= 0.0
        assert len(record['matrix']) == 2
        assert sum(record['all_occupations']) == pytest.approx(2.0)


def test_the_rotation_model_refuses_a_non_singlet_spectrum():
    with pytest.raises(ValueError):
        C.two_electron_singlet_rotation_replicates(
            [1.9, 0.2], 0.5, [1, 2, 3])


def test_the_canonical_interval_includes_the_reference_value():
    cert = load(CO_CERT)
    interval = cert['canonical_interval']['occupation_interval']
    reference = cert['canonical_interval']['reference']['occupation']
    assert interval['n_values'] == 4
    assert interval['low'] <= reference <= interval['high']


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_verify_returns_zero_for_an_accepted_certificate():
    assert C.main(['verify', str(N2_CERT)]) == 0


def test_cli_verify_returns_two_for_an_undetermined_certificate(
        tmp_path, n2_case):
    changed = copy.deepcopy(n2_case)
    changed['tolerances']['gradient_norm_max'] = 1e-9
    cert = C.build_certificate(changed)
    path = tmp_path / 'undetermined.json'
    C.write_json(path, cert)
    assert C.main(['verify', str(path)]) == 2
