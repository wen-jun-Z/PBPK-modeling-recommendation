#!/usr/bin/env python3
"""Score frozen Clari repeated-dose endpoint predictions after the global lock.

Endpoint values and method errors stay in a sealed output. The public receipt
reports execution, evaluability and the complete registered denominator only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('clari_native_artifact_checks', ROOT/'scripts/score_locked_aggregate_curves.py')
ARTIFACTS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ARTIFACTS)

CASE_ID = 'clarithromycin_oh2023_alone_day5_500mg_bid'
COLUMN = 'Organism|PeripheralVenousBlood|Clarithromycin|Plasma (Peripheral Venous Blood) [µmol/l]'
SAMPLE_HOURS = [0, 1, 2, 3, 4, 6, 8, 12]
ABSOLUTE_MINUTES = [60*(96+x) for x in SAMPLE_HOURS]

EVALUATION_CONTRACT = {
    'version': 'pbpk-clarithromycin-repeated-endpoints-v1',
    'case_id': 'clarithromycin_oh2023_alone_day5_500mg_bid',
    'compound': 'clarithromycin',
    'independent_unit': 'one drug, one publication/trial; seeds and methods are not independent studies',
    'method_job_count': 57,
    'qualification_tier': 'transportability_sensitivity',
    'strict_main_eligible': False,
    'nominal_dose_times_h': [0, 12, 24, 36, 48, 60, 72, 84, 96],
    'dose_mg_each': 500,
    'endpoint_window_h': [96, 108],
    'sampling_hours_from_last_morning_dose': [0, 1, 2, 3, 4, 6, 8, 12],
    'sampling_convention': 'original nominal grid; no time shifts or extrapolation; last sample uses the registered next-dose left-limit convention, which the paper does not resolve explicitly',
    'population_estimator': 'one male reference with enrolled demographic means; not a simulated distribution or the mean of individual NCA outcomes',
    'primary_endpoints': {
        'Cmax_ss_day5': {'unit': 'ug/L', 'estimator': 'maximum of the eight registered nominal plasma samples'},
        'AUCtau_day5': {'unit': 'ug*h/L', 'estimator': 'linear trapezoidal integral over all eight nominal samples from 96 to 108 h'},
    },
    'clinical_nca_limitation': 'published arithmetic individual-endpoint means are compared with reference-person endpoint proxies; the clinical integration algorithm is not assumed to be reproduced',
    'auxiliary_endpoints': 'native-grid maximum and linear trapezoidal AUC over the same window, reported separately as numerical-resolution sensitivity; not substituted for the primary estimators',
    'concentration_conversion': 'native umol/L multiplied by the frozen source molecular weight in g/mol gives ug/L; native time in minutes divided by 60 gives hours',
    'error': 'absolute natural logarithm of prediction/observation; signed prediction/observation ratio additionally reports direction',
    'nonpositive_or_nonfinite_endpoint': 'explicitly unevaluable; no epsilon, clipping, zero filling or silent omission',
    'denominator': 'all 57 registered method choices and both endpoints, including upstream and native failures',
    'observation_access': 'only after all selections, inputs, original native curves, archives and terminal predictions are authenticated and the immutable output is reserved',
    'confidence_intervals': 'no cross-drug confidence interval from one drug or from repeated seeds',
}


def local(value):
    return ARTIFACTS.local(value)


def read(path):
    return ARTIFACTS.read(path)


def sha(path):
    return ARTIFACTS.sha(path)


def verified(path, checksum, *, outcome=False, denied=()):
    return ARTIFACTS.verified(path, checksum, outcome=outcome, denied=denied)


def authenticate_source_registry(config, protocol, design_path):
    """Rebuild the actual completed source-method registry, without observations."""
    for key in ('input_package_path', 'input_package_sha256', 'suite_path'):
        if config.get(key) != protocol.get(key) or not protocol.get(key):
            raise ValueError('The executed source package/suite must be bound in both freeze and protocol')
    package = verified(protocol['input_package_path'], protocol['input_package_sha256'])
    script = ROOT/'scripts/predict_clarithromycin_methods_external.py'
    spec = importlib.util.spec_from_file_location('clari_score_source_contract', script)
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    registered = predictor.authenticated_registry(local(protocol['suite_path']), package, local(design_path))
    if any(config[key] != registered[key] for key in ('selections', 'models', 'frozen_input_sha256')):
        raise ValueError('The frozen method choices or source hashes differ from the completed registered suite')
    values = [p['Value'] for p in registered['template']['instance']['compound']['Parameters']
              if p['Name'] == 'Molecular weight']
    if len(values) != 1 or values[0] != protocol['molecular_weight_g_mol']:
        raise ValueError('Concentration conversion must use the unchanged source molecular weight')


def validate_locked_inputs(lock_path):
    """Authenticate the entire submitted denominator before any outcome access."""
    lock_path = local(lock_path)
    if any(p.lower().startswith('sealed') for p in lock_path.relative_to(ARTIFACTS.ROOT).parts):
        raise ValueError('An observation file cannot be a prediction lock')
    lock = read(lock_path)
    if (lock.get('schema_version') != 'pbpk-external-predictions-lock-v1'
            or lock.get('model_selection_frozen') is not True
            or lock.get('all_registered_predictions_terminal') is not True
            or type(lock.get('method_job_count')) is not int or lock['method_job_count'] != 57
            or type(lock.get('expected_prediction_rows')) is not int or lock['expected_prediction_rows'] != 57):
        raise ValueError('Complete 57-method repeated-dose prediction lock required')
    ref = lock['config_freeze']
    config = read(verified(ref['path'], ref['sha256']))
    if config.get('evaluation_contract') != EVALUATION_CONTRACT:
        raise ValueError('The endpoint evaluation contract must be frozen before prediction')
    selections, models = config['selections'], config['models']
    ARTIFACTS.validate_selections(selections, models)
    cases, records = config['cases'], lock['predictions']
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(records, list) or len(records) != 1:
        raise ValueError('Exactly one registered Clari study is required')
    planned, item = cases[0], records[0]
    if (item.get('case_id') != CASE_ID or any(item[k] != planned[k] for k in
            ('case_id', 'design_path', 'design_sha256', 'protocol_path', 'protocol_sha256'))):
        raise ValueError('Design/protocol identity changed after the pre-run freeze')
    design = read(verified(item['design_path'], item['design_sha256']))
    excluded = design['evidence_exclusion']
    denied = {local(design['sealed_observation_reference']['path']),
              *(local(p) for p in excluded.get('raw_source_files', [])),
              *(local(p) for p in excluded.get('curator_only_source_paths', []))}
    if (design.get('schema_version') != 'pbpk-independent-repeated-endpoint-design-v1'
            or design.get('compound') != 'clarithromycin'
            or design.get('publication_cluster_id') != 'PMC10333645'
            or design.get('trial_cluster_ids') != ['NCT02052336']
            or design.get('scientific_results_accessed_by_algorithm') is not False
            or excluded.get('exclude_entire_trial') is not True
            or excluded.get('exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training') is not True):
        raise ValueError('Wrong study identity or incomplete outcome isolation')
    protocol = read(verified(item['protocol_path'], item['protocol_sha256'], denied=denied))
    prediction = read(verified(item['path'], item['sha256'], denied=denied))
    if any(x.get('case_id') != CASE_ID for x in (design, protocol, prediction)):
        raise ValueError('Prediction and design case identities differ')
    if (protocol.get('design_sha256') != item['design_sha256']
            or prediction.get('protocol_sha256') != item['protocol_sha256']
            or protocol.get('evaluation_contract') != EVALUATION_CONTRACT
            or protocol.get('prediction_aggregation') != 'single_male_reference'
            or protocol.get('selections') != selections or prediction.get('selections') != selections
            or protocol.get('models') != models or not isinstance(protocol.get('model_keys'), list)
            or set(protocol['model_keys']) != set(models) or len(protocol['model_keys']) != len(models)):
        raise ValueError('Predicted selections or estimand differ from the global freeze')
    qualification = {k: config.get(k) for k in ('strict_main_eligible', 'qualification_tier', 'qualification_reasons')}
    if (qualification['strict_main_eligible'] is not False
            or qualification['qualification_tier'] != 'transportability_sensitivity'
            or not isinstance(qualification['qualification_reasons'], list)
            or not qualification['qualification_reasons']
            or any(not isinstance(r, str) or not r.strip() for r in qualification['qualification_reasons'])
            or any(protocol.get(k) != v for k, v in qualification.items())):
        raise ValueError('Predeclared approximation tier cannot be promoted or omitted')
    pop = design['population']
    mw = protocol.get('molecular_weight_g_mol')
    if (not ARTIFACTS.finite(mw) or mw <= 0 or protocol.get('sex_weights') != {'male': 1.0}
            or type(pop.get('n')) is not int or pop['n'] != 23
            or type(pop.get('enrolled_n')) is not int or pop['enrolled_n'] != 24
            or pop.get('sex') != 'male'):
        raise ValueError('The source 24-enrolled/23-completer male reference must stay explicit')
    if (not ARTIFACTS.same_grid(protocol.get('sampling_time_min'), ABSOLUTE_MINUTES)
            or not ARTIFACTS.same_grid(protocol.get('sampling_time_h'), SAMPLE_HOURS)
            or protocol.get('sampling_origin_h') != 96
            or protocol.get('native_concentration_column') != COLUMN):
        raise ValueError('Repeated-dose day-5 time origin and entire nominal grid must be preserved')
    regimen = protocol.get('regimen', {})
    approximation = protocol.get('approximation', {})
    if (regimen.get('doses_mg') != [500]*9
            or not ARTIFACTS.same_grid(regimen.get('administrations_h'), list(range(0, 97, 12)))
            or regimen.get('first_dose_origin_h') != 0 or regimen.get('index_dose_h') != 96
            or regimen.get('evaluation_window_h') != [96, 108]
            or regimen.get('source_tenth_dose_confirmed') is not False
            or regimen.get('final_boundary_policy') != 'left-limit before any hypothetical tenth dose; do not assert source confirmed order'):
        raise ValueError('All nine nominal doses and the unresolved final-dose convention must stay frozen')
    for key, expected in {'reference_subjects_only': True, 'individual_covariates_available': False,
                          'source_generated_height_not_observed': True, 'demographic_n': 24,
                          'endpoint_n': 23, 'physiology_population': 'European_ICRP_2002',
                          'source_weibull_preserved': True, 'source_TDI_and_Pgp_preserved': True}.items():
        if approximation.get(key) != expected or type(approximation.get(key)) is not type(expected):
            raise ValueError('Source mechanism or reference-population assumptions changed: ' + key)
    rows = prediction.get('models')
    wanted = {(key, 'male') for key in models}
    if (not isinstance(rows, list) or len(rows) != len(wanted)
            or {(r.get('model_key'), r.get('sex')) for r in rows} != wanted):
        raise ValueError('One terminal male native record per selected model is required')
    for row in rows:
        status, values = row.get('status'), row.get('concentration')
        invoked = row.get('native_execution_invoked')
        if (status not in {'completed', 'native_failed', 'not_run_resource_limit'} or type(invoked) is not bool
                or status == 'completed' and not invoked or status == 'not_run_resource_limit' and invoked
                or row.get('time_unit') != 'min'
                or str(row.get('unit')).replace('µ', 'u').replace('μ', 'u') != 'umol/l'
                or not ARTIFACTS.same_grid(row.get('time_min'), ABSOLUTE_MINUTES)):
            raise ValueError('Invalid terminal status, native units or repeated-dose grid')
        if status == 'completed':
            if (not isinstance(values, list) or len(values) != 8
                    or any(not ARTIFACTS.finite(v) or v < 0 for v in values) or row.get('error') is not None):
                raise ValueError('A successful prediction needs all eight finite nonnegative values')
            ARTIFACTS.validate_native_artifacts(row, protocol, ABSOLUTE_MINUTES, denied)
        elif values is not None or not row.get('error'):
            raise ValueError('Failed native predictions need null values and an explicit reason')
    jobs = config.get('expected_native_jobs')
    if (not isinstance(jobs, list) or len(jobs) != len(wanted)
            or {(j['case_id'], j['model_key'], j['sex']) for j in jobs} != {(CASE_ID, k, s) for k, s in wanted}
            or type(lock.get('registered_native_jobs')) is not int or lock['registered_native_jobs'] != len(wanted)
            or type(lock.get('native_execution_attempts')) is not int
            or lock['native_execution_attempts'] != sum(r['native_execution_invoked'] for r in rows)
            or type(lock.get('completed_native_jobs')) is not int
            or lock['completed_native_jobs'] != sum(r['status'] == 'completed' for r in rows)):
        raise ValueError('Native registration, attempt and completion counts differ')
    frozen = config.get('frozen_input_sha256')
    if not isinstance(frozen, dict) or not frozen:
        raise ValueError('Executed source and input hashes are required')
    for path, checksum in frozen.items():
        verified(path, checksum, denied=denied)
    authenticate_source_registry(config, protocol, item['design_path'])
    endpoint_values = {r['model_key']: endpoint_predictions(r, protocol) for r in rows}
    return lock, config, {'case_id': CASE_ID, 'design': design, 'protocol': protocol,
                          'native_rows': {r['model_key']: r for r in rows}, 'qualification': qualification,
                          'model_endpoint_predictions': endpoint_values,
                          'denied': denied, 'lock_sha256': sha(lock_path)}


def trapezoid(times, values):
    if (len(times) != len(values) or len(times) < 2
            or any(not ARTIFACTS.finite(v) for v in [*times, *values])
            or any(b <= a for a, b in zip(times, times[1:])) or any(v < 0 for v in values)):
        raise ValueError('AUC requires a complete, finite, ordered nonnegative curve')
    return math.fsum((b-a)*(left+right)/2 for a, b, left, right in zip(times, times[1:], values, values[1:]))


def endpoint_predictions(row, protocol):
    """Calculate both fixed primary estimators and separate dense-grid auxiliaries."""
    if row['status'] != 'completed':
        return None
    mw = protocol['molecular_weight_g_mol']
    values = [v*mw for v in row['concentration']]
    primary = {'Cmax_ss_day5': max(values), 'AUCtau_day5': trapezoid(SAMPLE_HOURS, values)}
    path = verified(row['native_csv_path'], row['native_csv_sha256'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    points = [(float(r['Time [min]']), float(r[COLUMN])*mw) for r in records]
    if (not points or not math.isclose(points[0][0], 0, rel_tol=0, abs_tol=1e-8)
            or not math.isclose(points[-1][0], 6480, rel_tol=0, abs_tol=1e-8)):
        raise ValueError('The complete native repeated-dose history must remain from 0 through 108 h')
    # Exact registered boundary values have already been checked against native
    # interpolation. Keep original interior points; never extend the interval.
    dense = [(5760.0, values[0]), *(p for p in points if 5760 < p[0] < 6480), (6480.0, values[-1])]
    auxiliary = {'native_grid_Cmax_day5_ug_L': max(v for _, v in dense),
                 'native_grid_AUCtau_day5_ug_h_L': trapezoid([t/60 for t, _ in dense], [v for _, v in dense]),
                 'native_grid_points_in_window': len(dense), 'primary_estimates_unchanged': True}
    return {'primary': primary, 'auxiliary': auxiliary}


def read_outcomes(case):
    """First and only observation access, exclusively after lock validation."""
    ref = case['design']['sealed_observation_reference']
    sealed = read(verified(ref['path'], ref['sha256'], outcome=True))
    if sealed.get('schema_version') != 'sealed-primary-table-pk-1' or sealed.get('case_id') != CASE_ID:
        raise ValueError('Unexpected sealed clinical endpoint source')
    rows = sealed.get('endpoints')
    expected = EVALUATION_CONTRACT['primary_endpoints']
    if (not isinstance(rows, list) or len(rows) != 2
            or {r.get('endpoint') for r in rows} != set(expected)):
        raise ValueError('Both original repeated-dose source endpoints are required')
    for row in rows:
        if (row.get('unit') != expected[row['endpoint']]['unit']
                or row.get('statistic') != 'arithmetic_mean'
                or type(row.get('endpoint_n')) is not int or row['endpoint_n'] != 23
                or not ARTIFACTS.finite(row.get('mean')) or row['mean'] <= 0
                or not ARTIFACTS.finite(row.get('sd')) or row['sd'] < 0):
            raise ValueError('Sealed endpoint unit, arithmetic estimand or source value contract changed')
        window = 'day5 nominal sampled maximum' if row['endpoint'] == 'Cmax_ss_day5' else [96, 108]
        if row.get('window') != window:
            raise ValueError('Sealed steady-state observation window differs from the registered target')
    return {r['endpoint']: r for r in rows}


def compare_endpoint(predicted, source, failure=None):
    observed = source['mean']
    evaluable = failure is None and ARTIFACTS.finite(predicted) and predicted > 0
    if not evaluable and failure is None:
        failure = 'Nonpositive or nonfinite reference endpoint; no clipping or epsilon'
    delta = math.log(predicted)-math.log(observed) if evaluable else None
    return {'endpoint': source['endpoint'], 'unit': source['unit'], 'evaluable': evaluable,
            'failure': failure, 'prediction': predicted, 'observed_arithmetic_mean': observed,
            'observed_sd': source['sd'], 'endpoint_n': source['endpoint_n'],
            'absolute_log_error': abs(delta) if evaluable else None,
            'prediction_observation_ratio': predicted/observed if evaluable else None}


def score_locked(lock_path, output):
    output = local(output)
    if output.exists():
        raise FileExistsError('Evaluation outputs are immutable; refused before reading observations')
    lock, config, case = validate_locked_inputs(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    sealed_dir = output/'sealed'
    sealed_dir.mkdir(mode=0o700)
    if sha(lock_path) != case['lock_sha256']:
        raise ValueError('The prediction lock changed before observation access')
    for path, checksum in config['frozen_input_sha256'].items():
        verified(path, checksum, denied=case['denied'])
    observed = read_outcomes(case)
    rows = []
    for selection in config['selections']:
        key = selection['model_key']
        native = case['native_rows'].get(key)
        prediction = case['model_endpoint_predictions'].get(key)
        reason = selection.get('failure') if key is None else native.get('error') if prediction is None else None
        if prediction is None and not reason:
            reason = 'Registered selection has no successful native prediction'
        endpoints = [compare_endpoint(None if prediction is None else prediction['primary'][name], source, reason)
                     for name, source in observed.items()]
        rows.append({'selection': selection, 'endpoints': endpoints,
                     'auxiliary_resolution_sensitivity': None if prediction is None else prediction['auxiliary'],
                     'status': 'evaluated' if all(e['evaluable'] for e in endpoints) else 'failed_retained_in_denominator'})
    if len(rows) != 57:
        raise ValueError('The all-method denominator changed during endpoint scoring')
    result = {'schema_version': 'pbpk-clarithromycin-endpoint-scores-v1',
              'case_id': CASE_ID, 'evaluation_contract': EVALUATION_CONTRACT,
              'method_job_denominator': 57, 'independent_drugs': 1, 'independent_studies': 1,
              **case['qualification'], 'results': rows,
              'lock_sha256': case['lock_sha256'], 'confidence_interval_computed': False,
              'scientific_acceptance_threshold_not_invented': True}
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if len(payload) > 8*1024**2:
        raise ValueError('Sealed endpoint output exceeded its bounded size')
    target = sealed_dir/'endpoint_scores.json'
    with target.open('xb') as stream:
        os.chmod(target, 0o600)
        stream.write(payload)
    counts = {}
    for name in EVALUATION_CONTRACT['primary_endpoints']:
        endpoints = [e for row in rows for e in row['endpoints'] if e['endpoint'] == name]
        evaluable = sum(e['evaluable'] for e in endpoints)
        counts[name] = {'evaluable': evaluable, 'failed_retained': 57-evaluable, 'total': 57}
    receipt = {'status': 'scored_and_sealed', 'method_jobs_expected': 57, 'method_rows_scored': 57,
               'independent_drugs': 1, 'independent_studies': 1, **case['qualification'],
               'metrics_exposed_to_method_development': False, 'lock_sha256': case['lock_sha256'],
               'endpoint_counts': counts, 'scorer_sha256': sha(Path(__file__)),
               'native_artifact_validator_sha256': sha(ROOT/'scripts/score_locked_aggregate_curves.py')}
    with (output/'scoring_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(score_locked(args.lock, args.output)), flush=True)


if __name__ == '__main__':
    main()
