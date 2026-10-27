"""Prediction provenance and interpretation guards, without rerunning weights."""
import json
from pathlib import Path
import subprocess

import pytest

from pbpk_kb.blueprints import pk_predictor as predictor


ROOT = Path(__file__).resolve().parents[2]


def raw_prediction():
    return {
        'VDss_L_kg': 1.7,
        'Volume_of_distribution_(VDss)_folderror': 2.4,
        'Volume_of_distribution_(VDss)_lowerbound': 1.7 / 2.4,
        'Volume_of_distribution_(VDss)_upperbound': 1.7 * 2.4,
        'CL_mL_min_kg': 3.5,
        'Clearance_(CL)_folderror': 2.,
        'Clearance_(CL)_lowerbound': 1.75,
        'Clearance_(CL)_upperbound': 7.,
        'Fraction_unbound_in_plasma_(fup)': .1,
        'Fraction_unbound_in_plasma_(fup)_folderror': 3.,
        'Fraction_unbound_in_plasma_(fup)_lowerbound': .1 / 3,
        # The author's 3.0.1 implementation clips the multiplier, not product.
        'Fraction_unbound_in_plasma_(fup)_upperbound': .1,
    }


def normalise(endpoint, raw=None, similarity=.26, overlap=True):
    return predictor._normalise_record(endpoint, raw or raw_prediction(), {
        'mean_top5_similarity_standardized': similarity,
        'training_overlap': overlap,
    }, {'predictor': 'PKSmart', 'version': '3.0.1'}, 'pubchem:4192')


def test_native_units_are_not_exponentiated_or_confused_with_oral_clearance():
    vdss, cl, fu = [normalise(endpoint) for endpoint in ['vdss', 'cl', 'fu']]
    assert (vdss['value'], vdss['unit'], vdss['definition']) == (1.7, 'L/kg', 'vdss_iv')
    assert (cl['value'], cl['unit'], cl['definition']) == (3.5, 'mL/min/kg', 'cl_systemic_iv')
    assert (fu['value'], fu['unit'], fu['definition']) == (.1, 'fraction', 'fu_plasma')
    for record in [vdss, cl, fu]:
        assert record['source_kind'] == 'predicted'
        assert record['training_overlap'] is True
        assert record['eligible_as_measured_evidence'] is False
        assert record['independent_validation_evidence'] is False
        assert record['automatic_parameter_override_allowed'] is False
        assert record['context']['population'].startswith('unknown')


def test_fu_upstream_bound_bug_is_retained_and_not_promoted_to_confidence_interval():
    uncertainty = normalise('fu')['uncertainty']
    assert uncertainty['upstream_upper_bound'] == .1
    assert uncertainty['nominal_upper_bound'] == pytest.approx(.3)
    assert uncertainty['coverage_probability'] is None
    assert uncertainty['calibrated_prediction_interval'] is False
    assert uncertainty['valid_as_measurement_interval'] is False
    assert 'clips the fold-error' in uncertainty['bound_review']
    raw = raw_prediction()
    raw['Fraction_unbound_in_plasma_(fup)'] = .8
    assert normalise('fu', raw)['uncertainty']['nominal_upper_bound'] == 1.


def test_subunit_error_estimate_is_unknown_not_an_inverted_interval():
    raw = raw_prediction()
    raw['Clearance_(CL)_folderror'] = .7
    uncertainty = normalise('cl', raw)['uncertainty']
    assert uncertainty['kind'] == 'unknown'
    assert uncertainty['fold_error'] is None
    assert uncertainty['nominal_lower_bound'] is None
    assert uncertainty['nominal_upper_bound'] is None


@pytest.mark.parametrize('endpoint,key,value', [
    ('cl', 'CL_mL_min_kg', 0), ('vdss', 'VDss_L_kg', float('nan')),
    ('vdss', 'VDss_L_kg', float('inf')), ('fu', 'Fraction_unbound_in_plasma_(fup)', 1.1),
])
def test_impossible_predictions_fail_explicitly(endpoint, key, value):
    raw = raw_prediction()
    raw[key] = value
    with pytest.raises(ValueError, match='invalid physical prediction'):
        normalise(endpoint, raw)


def test_similarity_screen_uses_software_threshold_without_clinical_claim():
    record = normalise('vdss', similarity=.25)
    assert record['applicability'] == 'in_domain'
    assert record['applicability_details']['threshold'] == .25
    assert record['applicability_details']['upstream_and_project_domains_identical'] is False
    assert 'not a validated clinical' in record['applicability_details']['threshold_source']
    assert normalise('vdss', similarity=.249)['applicability'] == 'out_of_domain'


@pytest.mark.parametrize('smiles,identity', [('', 'x'), ('CC\nO', 'x'), ('CC name', 'x'), ('CC', ''), ('C' * 4097, 'x')])
def test_unbounded_or_ambiguous_inputs_are_rejected(smiles, identity):
    with pytest.raises(ValueError):
        predictor._validate_request(smiles, identity)


def test_resources_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError, match='inside the project'):
        predictor._inside('../outside.json', tmp_path, must_exist=False)


def test_worker_receives_minimal_environment_without_account_secrets(monkeypatch):
    if not (ROOT / predictor.ENV_DIRECTORY / 'bin/python').is_file():
        pytest.skip('Project PKSmart environment is not installed')
    monkeypatch.setenv('PK_PREDICTOR_TEST_SECRET', 'must-not-reach-worker')
    seen = {}
    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, json.dumps({'status': 'completed', 'records': [{}, {}, {}]}), '')
    monkeypatch.setattr(predictor.subprocess, 'run', fake_run)
    predictor.predict_pk('CC', 'test:structure', ROOT)
    assert 'PK_PREDICTOR_TEST_SECRET' not in seen['env']
    assert set(seen['env']) == {'PATH', 'LANG', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS',
                               'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                               'PYTHONNOUSERSITE', 'TMPDIR', 'XDG_CACHE_HOME'}
    assert json.loads(seen['input'])['smiles'] == 'CC'
    assert seen['env']['XDG_CACHE_HOME'].startswith(str(ROOT))


def test_identity_receipt_mismatch_fails_before_execution(tmp_path, monkeypatch):
    if not (ROOT / predictor.ENV_DIRECTORY / 'bin/python').is_file():
        pytest.skip('Project PKSmart environment is not installed')
    monkeypatch.setattr(predictor.subprocess, 'run', lambda *a, **kw: pytest.fail('Worker must not start'))
    source = Path(__file__).relative_to(ROOT)
    with pytest.raises(ValueError, match='checksum mismatch'):
        predictor.predict_pk('CC', 'test:structure', ROOT, input_reference={
            'path': str(source), 'sha256': '0' * 64, 'locator': 'identity-only test'})


def test_compatibility_is_explicit_and_cannot_accept_an_arbitrary_package(monkeypatch):
    with pytest.raises(ValueError, match='Unknown explicitly requested'):
        predictor.predict_pk('CC', 'test:structure', ROOT, preprocessing_compatibility='latest')
    assert predictor._verified_compatibility(ROOT, None) is None
    with pytest.raises(ValueError, match='Unknown explicitly requested'):
        predictor._verified_compatibility(ROOT, '../other-environment')


def test_compatibility_bytes_are_checked_before_import(monkeypatch):
    wheel = ROOT / predictor.ARTIFACT_DIRECTORY / predictor.COMPATIBILITY_WHEEL
    if not wheel.exists():
        pytest.skip('Optional official compatibility artifact not installed')
    monkeypatch.setattr(predictor, '_sha', lambda path: '0' * 64)
    with pytest.raises(ValueError, match='wheel checksum mismatch'):
        predictor._verified_compatibility(ROOT, predictor.COMPATIBILITY_NAME)


def test_upstream_failure_is_preserved_without_automatic_compatibility_retry(monkeypatch):
    if not (ROOT / predictor.ENV_DIRECTORY / 'bin/python').is_file():
        pytest.skip('Project PKSmart environment is not installed')
    calls = []
    def fail(command, **kwargs):
        calls.append(json.loads(kwargs['input']))
        return subprocess.CompletedProcess(command, 1, '', 'Official PKSmart structure standardization failed')
    monkeypatch.setattr(predictor.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='standardization failed'):
        predictor.predict_pk('CC', 'test:structure', ROOT)
    assert len(calls) == 1
    assert calls[0]['preprocessing_compatibility'] is None
