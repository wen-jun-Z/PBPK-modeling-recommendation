"""Synthetic repeated-dose records only; no real held-out clinical values."""
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('clari_scorer_test', ROOT/'scripts/score_clarithromycin_methods_endpoints.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, *, native_failed=False, all_failed=False):
    monkeypatch.setattr(score, 'ROOT', tmp_path)
    monkeypatch.setattr(score.ARTIFACTS, 'ROOT', tmp_path)
    # Source-suite authentication has its own existing all-57 predictor tests;
    # this fixture has no real suite and isolates the endpoint/global barrier.
    monkeypatch.setattr(score, 'authenticate_source_registry', lambda *_: None)
    program = tmp_path/'score.py'
    program.write_text('# synthetic scorer')
    helper = tmp_path/'scripts/score_locked_aggregate_curves.py'
    helper.parent.mkdir()
    helper.write_text('# synthetic artifact helper')
    monkeypatch.setattr(score, '__file__', str(program))
    selections = []
    for i, entry in enumerate(score.ARTIFACTS.expected_selections().values()):
        failed = all_failed or i == 0
        selections.append({**entry, 'model_key': None if failed else 'one',
            'prediction_available': not failed, 'upstream_status': 'blocked' if failed else 'completed',
            **({'failure': 'synthetic upstream failure'} if failed else {})})
    models = {} if all_failed else {'one': {'action': 'partition_pksim_standard', 'values': [1, 2]}}
    sealed = {'schema_version': 'sealed-primary-table-pk-1', 'case_id': score.CASE_ID, 'endpoints': [
        {'endpoint': 'Cmax_ss_day5', 'mean': 200., 'sd': 10., 'unit': 'ug/L', 'statistic': 'arithmetic_mean',
         'endpoint_n': 23, 'window': 'day5 nominal sampled maximum'},
        {'endpoint': 'AUCtau_day5', 'mean': 1200., 'sd': 100., 'unit': 'ug*h/L', 'statistic': 'arithmetic_mean',
         'endpoint_n': 23, 'window': [96, 108]}]}
    sealed_path = tmp_path/'sealed/observations.json'
    sealed_sha = save(sealed_path, sealed)
    design = {'schema_version': 'pbpk-independent-repeated-endpoint-design-v1', 'case_id': score.CASE_ID,
        'compound': 'clarithromycin', 'publication_cluster_id': 'PMC10333645', 'trial_cluster_ids': ['NCT02052336'],
        'scientific_results_accessed_by_algorithm': False,
        'population': {'n': 23, 'enrolled_n': 24, 'sex': 'male'},
        'sealed_observation_reference': {'path': str(sealed_path), 'sha256': sealed_sha},
        'evidence_exclusion': {'exclude_entire_trial': True,
            'exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training': True,
            'raw_source_files': ['raw_outcome.xml']}}
    design_path = tmp_path/'design.json'
    design_sha = save(design_path, design)
    qualification = {'strict_main_eligible': False, 'qualification_tier': 'transportability_sensitivity',
        'qualification_reasons': ['synthetic representative-subject and uncertain formulation approximation']}
    protocol = {'case_id': score.CASE_ID, 'design_sha256': design_sha, 'evaluation_contract': score.EVALUATION_CONTRACT,
        'prediction_aggregation': 'single_male_reference', 'models': models, 'model_keys': list(models),
        'selections': selections, **qualification, 'sex_weights': {'male': 1.}, 'molecular_weight_g_mol': 100.,
        'sampling_time_min': score.ABSOLUTE_MINUTES, 'sampling_time_h': score.SAMPLE_HOURS, 'sampling_origin_h': 96,
        'native_concentration_column': score.COLUMN,
        'regimen': {'doses_mg': [500]*9, 'administrations_h': list(range(0, 97, 12)), 'first_dose_origin_h': 0,
            'index_dose_h': 96, 'evaluation_window_h': [96, 108], 'source_tenth_dose_confirmed': False,
            'final_boundary_policy': 'left-limit before any hypothetical tenth dose; do not assert source confirmed order'},
        'approximation': {'reference_subjects_only': True, 'individual_covariates_available': False,
            'source_generated_height_not_observed': True, 'demographic_n': 24, 'endpoint_n': 23,
            'physiology_population': 'European_ICRP_2002', 'source_weibull_preserved': True, 'source_TDI_and_Pgp_preserved': True}}
    protocol_path = tmp_path/'protocol.json'
    protocol_sha = save(protocol_path, protocol)
    rows = []
    if models:
        values = [1., 2., 4., 3., 2., 1., .5, .25]
        row = {'model_key': 'one', 'sex': 'male', 'status': 'native_failed' if native_failed else 'completed',
            'native_execution_invoked': True, 'time_min': score.ABSOLUTE_MINUTES, 'concentration': None if native_failed else values,
            'unit': 'µmol/l', 'time_unit': 'min', 'error': 'synthetic native failure' if native_failed else None}
        if not native_failed:
            path = tmp_path/'native.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Time [min]', score.COLUMN])
                writer.writerows(sorted([(0, 0), (5000, 1), (5790, 6), *zip(score.ABSOLUTE_MINUTES, values)]))
            raw = b'<synthetic-native-model/>'
            archive = tmp_path/'model.pkml.gz'
            archive.write_bytes(gzip.compress(raw, mtime=0))
            row.update(native_csv_path=str(path), native_csv_sha256=score.sha(path), native_time_column='Time [min]',
                native_concentration_column=score.COLUMN,
                pkml_archives=[{'path': str(archive), 'sha256': score.sha(archive),
                    'original_path': str(tmp_path/'model.pkml'), 'original_byte_size': len(raw),
                    'original_sha256': hashlib.sha256(raw).hexdigest()}])
        rows.append(row)
    prediction = {'case_id': score.CASE_ID, 'models': rows, 'selections': selections, 'protocol_sha256': protocol_sha}
    prediction_path = tmp_path/'predictions.json'
    prediction_sha = save(prediction_path, prediction)
    case = {'case_id': score.CASE_ID, 'design_path': str(design_path), 'design_sha256': design_sha,
            'protocol_path': str(protocol_path), 'protocol_sha256': protocol_sha}
    config = {'cases': [case], 'models': models, 'selections': selections, **qualification,
        'evaluation_contract': score.EVALUATION_CONTRACT,
        'expected_native_jobs': [{'case_id': score.CASE_ID, 'model_key': k, 'sex': 'male'} for k in models],
        'frozen_input_sha256': {str(program): score.sha(program), str(design_path): design_sha}}
    config_path = tmp_path/'freeze.json'
    config_sha = save(config_path, config)
    lock = {'schema_version': 'pbpk-external-predictions-lock-v1', 'model_selection_frozen': True,
        'all_registered_predictions_terminal': True, 'method_job_count': 57, 'expected_prediction_rows': 57,
        'config_freeze': {'path': str(config_path), 'sha256': config_sha},
        'predictions': [{**case, 'path': str(prediction_path), 'sha256': prediction_sha}],
        'registered_native_jobs': len(rows), 'native_execution_attempts': len(rows),
        'completed_native_jobs': sum(r['status'] == 'completed' for r in rows)}
    lock_path = tmp_path/'lock.json'
    save(lock_path, lock)
    return lock_path, lock, config, protocol, prediction


def rebind(path, lock, config, protocol, prediction):
    item = lock['predictions'][0]
    item['protocol_sha256'] = save(Path(item['protocol_path']), protocol)
    config['cases'][0]['protocol_sha256'] = item['protocol_sha256']
    prediction['protocol_sha256'] = item['protocol_sha256']
    item['sha256'] = save(Path(item['path']), prediction)
    lock['config_freeze']['sha256'] = save(Path(lock['config_freeze']['path']), config)
    save(path, lock)


def forbid_outcomes(monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if 'sealed' in path.parts or path.name == 'raw_outcome.xml':
            pytest.fail('Outcome access before a complete prediction barrier')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)


def test_primary_units_accumulated_predose_and_dense_peak_are_separate(tmp_path, monkeypatch):
    path, *_ = fixture(tmp_path, monkeypatch)
    forbid_outcomes(monkeypatch)
    _, config, case = score.validate_locked_inputs(path)
    value = case['model_endpoint_predictions']['one']
    assert value['primary']['Cmax_ss_day5'] == 400
    assert value['primary']['AUCtau_day5'] == pytest.approx(1650)
    assert value['auxiliary']['native_grid_Cmax_day5_ug_L'] == 600
    assert value['auxiliary']['native_grid_AUCtau_day5_ug_h_L'] == pytest.approx(1875)
    assert len(config['selections']) == 57


@pytest.mark.parametrize('damage', ['missing_method', 'changed_metric', 'promoted_tier', 'wrong_counts',
    'sealed_input', 'raw_input', 'single_dose', 'shifted_origin', 'invented_tenth', 'discard_TDI',
    'missing_model', 'shifted_time', 'negative_tail', 'csv_binding', 'archive', 'truncated_history'])
def test_invalid_lock_is_rejected_before_any_sealed_access(tmp_path, monkeypatch, damage):
    path, lock, config, protocol, prediction = fixture(tmp_path, monkeypatch)
    if damage == 'missing_method': config['selections'].pop()
    elif damage == 'changed_metric': config['evaluation_contract'] = {'version': 'posthoc'}
    elif damage == 'promoted_tier': config['strict_main_eligible'] = True
    elif damage == 'wrong_counts': lock['native_execution_attempts'] = 0
    elif damage == 'sealed_input': config['frozen_input_sha256']['sealed/observations.json'] = '0'*64
    elif damage == 'raw_input': config['frozen_input_sha256']['raw_outcome.xml'] = '0'*64
    elif damage == 'single_dose': protocol['regimen']['administrations_h'] = [96]
    elif damage == 'shifted_origin': protocol['sampling_origin_h'] = 0
    elif damage == 'invented_tenth': protocol['regimen']['source_tenth_dose_confirmed'] = True
    elif damage == 'discard_TDI': protocol['approximation']['source_TDI_and_Pgp_preserved'] = False
    elif damage == 'missing_model': prediction['models'] = []
    elif damage == 'shifted_time': prediction['models'][0]['time_min'] = [t+1 for t in score.ABSOLUTE_MINUTES]
    elif damage == 'csv_binding': prediction['models'][0]['concentration'][1] += 1
    elif damage == 'archive': prediction['models'][0]['pkml_archives'][0]['original_sha256'] = '0'*64
    else:
        row = prediction['models'][0]
        csv_path = Path(row['native_csv_path'])
        lines = csv_path.read_text().splitlines(True)
        if damage == 'negative_tail': lines[2] = '5000,-1\n'
        else: lines.pop(1)
        csv_path.write_text(''.join(lines))
        row['native_csv_sha256'] = score.sha(csv_path)
    rebind(path, lock, config, protocol, prediction)
    forbid_outcomes(monkeypatch)
    with pytest.raises(ValueError): score.score_locked(path, tmp_path/'scores')


@pytest.mark.parametrize('mode', ['one_upstream_failure', 'all_upstream_failed', 'native_failed'])
def test_scoring_preserves_57_rows_and_public_receipt_hides_outcomes(tmp_path, monkeypatch, mode):
    path, *_ = fixture(tmp_path, monkeypatch, all_failed=mode == 'all_upstream_failed', native_failed=mode == 'native_failed')
    receipt = score.score_locked(path, tmp_path/'scores')
    expected = 56 if mode == 'one_upstream_failure' else 0
    assert receipt['method_rows_scored'] == 57 and receipt['strict_main_eligible'] is False
    assert all(x == {'evaluable': expected, 'failed_retained': 57-expected, 'total': 57}
               for x in receipt['endpoint_counts'].values())
    assert 'observed_arithmetic_mean' not in json.dumps(receipt)
    assert 'absolute_log_error' not in json.dumps(receipt)
    results = json.loads((tmp_path/'scores/sealed/endpoint_scores.json').read_text())['results']
    assert len(results) == 57
    if expected:
        endpoint = next(e for e in results[1]['endpoints'] if e['endpoint'] == 'Cmax_ss_day5')
        assert endpoint['absolute_log_error'] == pytest.approx(math.log(2))
    forbid_outcomes(monkeypatch)
    with pytest.raises(FileExistsError): score.score_locked(path, tmp_path/'scores')


def test_zero_prediction_is_failure_without_epsilon():
    source = {'endpoint': 'Cmax_ss_day5', 'mean': 200., 'sd': 10., 'endpoint_n': 23, 'unit': 'ug/L'}
    result = score.compare_endpoint(0, source)
    assert result['evaluable'] is False and result['absolute_log_error'] is None
    assert result['prediction'] == 0


def test_source_registry_rejection_precedes_observations(tmp_path, monkeypatch):
    path, *_ = fixture(tmp_path, monkeypatch)
    def reject(*_):
        raise ValueError('source suite does not match')
    monkeypatch.setattr(score, 'authenticate_source_registry', reject)
    forbid_outcomes(monkeypatch)
    with pytest.raises(ValueError, match='source suite'):
        score.score_locked(path, tmp_path/'scores')


@pytest.mark.parametrize('times,values', [([0, 1], [1]), ([1, 0], [1, 2]), ([0, 1], [1, -2]), ([0, 1], [1, float('nan')])])
def test_incomplete_or_invalid_auc_domain_is_never_repaired(times, values):
    with pytest.raises(ValueError): score.trapezoid(times, values)
