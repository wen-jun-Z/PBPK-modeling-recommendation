#!/usr/bin/env python3
"""Bounded study-separated comparison of the implemented PBPK blueprint method.

The learned CL/fu predictions are frozen. Independent development studies may
calibrate at most two construction parameters and select one of two partition
formulas. The fixed-formula comparator receives exactly the same information.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from pbpk_kb.blueprints.development_curve_blueprint import (
    _training_document, fit_development_mapping, predict_with_development_mapping,
)
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint
from pbpk_kb.blueprints.evidence_evaluation import score_case
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/blueprint_design/ml_hybrid_pilot_20260928'
FORMULAS = ('PKSimStandard', 'RodgersRowland')
METHODS = ('curve_fixed_partition', 'ours_blueprint')
DEVELOPMENT = {
    'caffeine': 'caffeine_wahllander_1989',
    'midazolam': 'midazolam_schwagmeier_1998_5mg',
    'alfentanil': 'alfentanil_phimmasone_2001_iv',
}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def reference(path, location=''):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': sha256(path), 'location': location}


def verify(pins):
    for name, expected in pins.items():
        if sha256(inside(name, ROOT, exists=True)) != expected:
            raise ValueError('Frozen source changed: ' + name)


def prepare(output, workers, budget):
    if output.exists():
        raise FileExistsError('Preparation output already exists; use --run-prepared')
    output.mkdir(parents=True)
    inventory = BASE/'inputs/clinical_candidates.json'
    historical = ROOT/'experiments/blueprint_design/ten_drug_endpoints_20260923/evaluation/protocol.json'
    candidates = read(inventory)['cases']
    old_cases = {c['case_id']: c for c in read(historical)['cases']}
    pins = {}

    def pin(path):
        path = inside(path, ROOT, exists=True)
        pins[str(path.relative_to(ROOT))] = sha256(path)

    for p in [Path(__file__).resolve(), inventory, historical,
              BASE/'run/clinical_parameter_predictions_locked.json',
              BASE/'run/parameter_split.json', BASE/'run/model_manifest.json',
              BASE/'run/parameter_models.joblib',
              *sorted((ROOT/'src/pbpk_kb/blueprints').glob('*.py')),
              *sorted((ROOT/'src/pbpk_kb/model_runtime/R').glob('*.R'))]:
        pin(p)
    cases, checks = [], []
    for case in candidates:
        drug = case['drug']
        development = old_cases[DEVELOPMENT[drug]]
        target_source = ROOT/case['observations_path']
        source = ROOT/development['observations_path']
        pin(target_source); pin(source)
        if source == target_source or sha256(source) == sha256(target_source):
            raise ValueError('Development and evaluation data must be distinct studies')
        original = read(source)
        folder = output/'inputs'/drug
        folder.mkdir(parents=True)
        normalized = {**deepcopy(original), 'compound_id': case['compound_id'],
            'study_group_id': development['case_id'], 'study_id': development['study_id'],
            'role': 'calibration_data_reconstruction', 'matrix': 'plasma',
            'time_unit': 'min', 'unit': 'umol/l',
            'original_role': original.get('role'),
            'metadata_normalization_source': reference(source, 'All original observations preserved exactly'),
            'historically_seen_study': True, 'newly_blinded_validation': False}
        # No point removal, smoothing, time shifts, or concentration conversion.
        assert normalized['observations'] == original['observations']
        obs_path = folder/'development_observations.json'
        save(obs_path, normalized); pin(obs_path)
        training = {**reference(obs_path, 'metadata-normalized development curve'),
            'role': 'calibration_data_reconstruction', 'case_id': development['case_id']}
        request_path = BASE/'run/inputs'/drug/'rf_direct/request.json'
        pin(request_path)
        request = read(request_path)
        request['target'].pop('terminal_fit', None)
        # Keep the frozen RF drug ledger and evidence unchanged. No target-study
        # CL, Vss, fu, or historical fitted parameters may be reintroduced.
        for item in request['evidence']:
            pin(ROOT/item['source']['path'])
        pin(ROOT/request['evidence_ledger']['path'])
        request['assumptions'] = [
            'Historical study-separated development comparison; not newly blinded validation.',
            'CL and fu use the identical frozen structure-derived predictions as the RF-direct comparator.',
            'No target evaluation concentrations or published PK summaries enter construction or fitting.',
            'A separate clinical development study calibrates logMA and a common clearance multiplier.',
            'Both formulas receive the same development curve; selection uses only its frozen regularized objective.',
            'Predicted Vss is not a measured anchor and is not used in this development calibration.',
            'One representative adult approximates a study group; total clearance remains lumped into liver.',
        ]
        # Remove only the newly allowed development source, if historically
        # forbidden, and preserve every other exclusion including test data.
        allowed = {str(source.relative_to(ROOT)), str(obs_path.relative_to(ROOT))}
        allowed_hashes = {sha256(source), sha256(obs_path)}
        removed = [f for f in request.get('forbidden_sources', [])
                   if f['path'] in allowed or f['sha256'] in allowed_hashes]
        request['forbidden_sources'] = [f for f in request.get('forbidden_sources', [])
            if f['path'] not in allowed and f['sha256'] not in allowed_hashes]
        if not any(f['sha256'] == sha256(target_source) for f in request['forbidden_sources']):
            request['forbidden_sources'].append(reference(target_source, 'Evaluation outcomes; never construction inputs'))
        evaluation_request = deepcopy(request)
        development_request = deepcopy(request)
        for key in ('human', 'dosing', 'target'):
            development_request[key] = deepcopy(development['request'][key])
        development_request['target'].pop('terminal_fit', None)
        save(folder/'evaluation_request.json', evaluation_request)
        save(folder/'development_request.json', development_request)
        pin(folder/'evaluation_request.json'); pin(folder/'development_request.json')
        for formula in FORMULAS:
            req = deepcopy(development_request); req['partition_method'] = formula
            EvidenceBlueprintRequest.model_validate(req)
            document = _training_document(training, req, ROOT)
            if document['observations'] != original['observations']:
                raise ValueError('Metadata normalization changed clinical observations')
            constructed = assemble_evidence_blueprint(req, ROOT)
            if constructed['status'] != 'assembled':
                raise ValueError('Preflight assembly failed: '+drug+'/'+formula)
            checks.append({'drug': drug, 'formula': formula, 'status': 'assembled'})
        cases.append({'case_id': case['case_id'], 'drug': drug, 'compound_id': case['compound_id'],
            'study_id': case['study_id'], 'observations_path': case['observations_path'],
            'development_case_id': development['case_id'], 'development_study_id': development['study_id'],
            'development_point_count': len(original['observations']),
            'development_request_path': str((folder/'development_request.json').relative_to(ROOT)),
            'evaluation_request_path': str((folder/'evaluation_request.json').relative_to(ROOT)),
            'training_source': training, 'removed_development_exclusions': removed})
    protocol = {'schema_version': 'own-blueprint-study-separated-pilot-1', 'prepared_at': now(),
        'cases': cases, 'methods': list(METHODS), 'partition_candidates': list(FORMULAS),
        'resources': {'workers': workers, 'threads_per_native_job': 2,
            'fit_calls_per_candidate': budget, 'maximum_native_calls': 6*budget+6,
            'native_timeout_seconds': 180, 'maximum_native_output_bytes_per_call': 64*1024**2},
        'objective': 'mean(log(P/O)^2) + 0.1*(log(CL/ML_CL)/log(2))^2 + 0.01*((logMA-logP)/2.5)^2',
        'parameter_limits': {'logMA': 'physical-property logP +/- 2.5', 'clearance_scale': [0.25, 4.0]},
        'selection': 'Minimum eligible development objective; PKSimStandard preferred on exact ties',
        'vss_anchor': None, 'vss_anchor_reason': 'No independent measured anchor selected; ML Vss is not a measurement',
        'failure_recovery': False, 'retained_failures': True,
        'same_input_fair_comparison': ['curve_fixed_partition', 'ours_blueprint'],
        'unequal_information_baselines': ['median_direct', 'rf_direct', 'rf_vss_mapping'],
        'information_note': 'Old three methods have no target-drug clinical development curve; no pure algorithm superiority claim against them.',
        'evaluation_studies_used_for_fit_or_selection': False,
        'all_evaluation_predictions_locked_before_scoring': True,
        'historical_studies_previously_seen': True, 'newly_blinded_independent_validation': False,
        'all_11_clinical_endpoints_validated': False,
        'no_llm_calls': True, 'no_parameter_model_retraining': True,
        'preflight_checks': checks, 'pins': pins}
    save(output/'protocol.json', protocol)
    save(output/'progress.json', {'stage': 'prepared', 'updated_at': now(),
        'fit_jobs_total': 6, 'fit_jobs_completed': 0, 'prediction_jobs_completed': 0})
    return protocol


def execute(output, protocol):
    verify(protocol['pins'])
    if (output/'runner.json').exists():
        raise FileExistsError('This prepared experiment has already been started')
    save(output/'runner.json', {'pid': os.getpid(), 'started_at': now(), 'project': str(ROOT)})
    fits, results = {}, {}
    started = time.monotonic()
    cases = protocol['cases']

    def progress(stage, **extra):
        save(output/'progress.json', {'stage': stage, 'updated_at': now(), 'pid': os.getpid(),
            'elapsed_seconds': round(time.monotonic()-started, 1),
            'fit_jobs_completed': len(fits), 'fit_jobs_total': 6,
            'prediction_jobs_completed': len(results), 'prediction_jobs_total': 6, **extra})

    def fit(case, formula):
        drug = case['drug']; req = read(ROOT/case['development_request_path'])
        req['partition_method'] = formula
        folder = output/'fits'/drug/formula
        try:
            result = fit_development_mapping(reference_request=req, training_source=case['training_source'],
                vss_anchor=None, output_dir=folder, project_root=ROOT,
                max_evaluations=protocol['resources']['fit_calls_per_candidate'], recover_failed_trials=False)
            result['mapping_path'] = str((folder/'mapping_locked.json').relative_to(ROOT))
        except Exception as exc:
            result = {'status': 'failed', 'eligible': False, 'reason': type(exc).__name__+': '+str(exc)}
        return result

    progress('fit_development_studies')
    with ThreadPoolExecutor(max_workers=protocol['resources']['workers']) as pool:
        futures = {pool.submit(fit, c, f): c['drug']+'/'+f for c in cases for f in FORMULAS}
        for future in as_completed(futures):
            key = futures[future]; fits[key] = future.result()
            save(output/'fits.json', fits)
            progress('fit_development_studies', latest={'job': key, 'status': fits[key]['status']})
            print(json.dumps({'fit': key, 'status': fits[key]['status'], 'eligible': fits[key].get('eligible')}), flush=True)
    verify(protocol['pins'])
    selections = {}
    for case in cases:
        drug = case['drug']
        eligible = [fits[drug+'/'+f] for f in FORMULAS if fits[drug+'/'+f].get('eligible')]
        selected = min(eligible, key=lambda f: (f['decision']['objective'],
            f['construction_partition_method'] != 'PKSimStandard')) if eligible else None
        fixed = fits[drug+'/PKSimStandard']
        selections[drug] = {
            'ours_blueprint': selected['mapping_path'] if selected else None,
            'curve_fixed_partition': fixed['mapping_path'] if fixed.get('eligible') else None,
        }
    map_pins = {f['mapping_path']: sha256(ROOT/f['mapping_path']) for f in fits.values() if f.get('mapping_path')}
    save(output/'mappings_lock.json', {'locked_at': now(), 'selections': selections, 'mapping_hashes': map_pins})
    jobs, aliases = {}, {}
    for case in cases:
        for method in METHODS:
            key = case['drug']+'/'+method
            path = selections[case['drug']][method]
            if not path:
                results[key] = {'case_id': case['case_id'], 'study_id': case['study_id'], 'drug': case['drug'], 'method': method,
                    'result': {'status': 'mapping_unavailable', 'reason': 'No eligible development candidate; no silent fallback'}}
                continue
            identity = (case['drug'], path)
            if identity not in jobs:
                jobs[identity] = (case, method, path)
            aliases[key] = jobs[identity][0]['drug']+'/'+jobs[identity][1]

    def predict(job):
        case, method, path = job
        folder = output/'cases'/case['drug']/method
        request = read(ROOT/case['evaluation_request_path'])
        request['partition_method'] = read(ROOT/path)['construction_partition_method']
        try:
            verify(map_pins)
            result = predict_with_development_mapping(request=request, mapping_path=ROOT/path,
                output_dir=folder, project_root=ROOT)
            result.update(prediction_path=str((folder/'prediction_locked.json').relative_to(ROOT)),
                endpoints_path=str((folder/'endpoints.json').relative_to(ROOT)), mapping_path=path,
                partition_method=request['partition_method'])
        except Exception as exc:
            result = {'status': 'failed', 'reason': type(exc).__name__+': '+str(exc), 'mapping_path': path}
        return {'case_id': case['case_id'], 'study_id': case['study_id'], 'drug': case['drug'], 'method': method, 'result': result}

    progress('predict_evaluation_studies')
    with ThreadPoolExecutor(max_workers=protocol['resources']['workers']) as pool:
        futures = {pool.submit(predict, job): job for job in jobs.values()}
        for future in as_completed(futures):
            row = future.result(); key = row['drug']+'/'+row['method']; results[key] = row
            progress('predict_evaluation_studies', latest={'job': key, 'status': row['result']['status']})
    for key, origin in aliases.items():
        if key != origin:
            results[key] = {**deepcopy(results[origin]), 'method': key.split('/')[1], 'alias_of': origin}
    locks = {}
    for key, row in results.items():
        if row['result']['status'] == 'predicted':
            locks[key] = {**reference(ROOT/row['result']['prediction_path'], 'evaluation prediction locked before scoring'),
                'endpoints': reference(ROOT/row['result']['endpoints_path'], 'native endpoints')}
    save(output/'predictions_lock.json', {'locked_at': now(), 'predictions': locks, 'aliases': aliases,
        'all_attempts_completed_before_scoring': True})
    verify(protocol['pins']); verify(map_pins)
    save(output/'job_results.json', results)
    progress('score_evaluation_studies')
    rows = []
    for case in cases:
        # This is the first loading of target-study numeric concentrations.
        observation = read(ROOT/case['observations_path'])
        request = read(ROOT/case['evaluation_request_path'])
        times = [p['time_min'] for p in observation['observations']]
        request['target']['terminal_fit'] = {'start_h': times[-4]/60, 'end_h': times[-1]/60,
            'min_points': 4, 'min_adjusted_r2': .98, 'min_span_half_lives': 1.0,
            'max_auc_extrapolated_fraction': .2, 'max_aumc_extrapolated_fraction': .2}
        for method in METHODS:
            key = case['drug']+'/'+method; row = results[key]
            if key in locks:
                try:
                    locked = locks[key]
                    if sha256(ROOT/locked['path']) != locked['sha256']:
                        raise ValueError('Evaluation prediction changed after lock')
                    row['metrics'] = score_case(read(ROOT/locked['path']), observation, request)
                    row['scoring_status'] = 'completed'
                except Exception as exc:
                    row['scoring_status'] = 'failed'; row['scoring_error'] = type(exc).__name__+': '+str(exc)
            else:
                row['scoring_status'] = 'prediction_unavailable'
            m = row.get('metrics', {})
            rows.append({'drug': case['drug'], 'method': method, 'study_id': case['study_id'],
                'development_study_id': case['development_study_id'],
                'prediction_status': row['result']['status'], 'scoring_status': row['scoring_status'],
                'gmfe': m.get('gmfe'), 'ln_rmse': m.get('log_rmse'), 'point_count': m.get('point_count'),
                'auc_fold_error': m.get('auc', {}).get('fold_error'),
                'sampled_cmax_fold_error': m.get('sampled_cmax', {}).get('fold_error'),
                'c_last_fold_error': m.get('c_last', {}).get('fold_error'),
                'tmax_absolute_error_h': m.get('sampled_tmax', {}).get('absolute_error_h'),
                'failure_reason': row.get('scoring_error', row['result'].get('reason'))})
    scored = sum(r.get('scoring_status') == 'completed' for r in results.values())
    actual_calls = sum(len(list((output/'fits'/c['drug']/f).glob('attempt_*'))) for c in cases for f in FORMULAS) + len(jobs)
    summary = {'status': 'completed' if len(locks) == scored == 6 else 'completed_with_failures',
        'finished_at': now(), 'elapsed_seconds': round(time.monotonic()-started, 1),
        'clinical_drugs': 3, 'attempted_clinical_predictions': 6, 'unique_evaluation_runs': len(jobs),
        'native_evaluations_total': actual_calls, 'successful_native_predictions': len(locks),
        'successful_clinical_scores': scored, 'selections': selections, 'fits': fits, 'results': results,
        'clinical_cases_previously_seen': True, 'all_11_clinical_endpoints_validated': False,
        'comparison_interpretation': protocol['information_note'],
        'limitations': ['Historical studies are not blinded or prospective validation.',
            'The same development study fits parameters and selects formula; selection is in-sample.',
            'No independent measured Vss anchor; predicted CL regularization may be biased.',
            'Budget exhaustion can return a feasible point without numerical convergence.',
            'Group mean curves with representative physiology do not provide individual prediction intervals.']}
    save(output/'summary.json', summary)
    with (output/'clinical_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    progress(summary['status'], native_evaluations_total=actual_calls)
    print(json.dumps({'status': summary['status'], 'scored': scored, 'native_calls': actual_calls,
        'seconds': summary['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'own_method')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--max-evaluations', type=int, choices=range(6, 25), default=18)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(); output = inside(args.output, ROOT)
    if args.status:
        print(json.dumps(read(output/'progress.json'), indent=2)); return
    try:
        protocol = read(output/'protocol.json') if args.run_prepared else prepare(output, args.workers, args.max_evaluations)
        if args.prepare_only:
            print(json.dumps({'status': 'prepared', 'preflight': protocol['preflight_checks']})); return
        execute(output, protocol)
    except Exception as exc:
        if output.exists():
            previous = read(output/'progress.json') if (output/'progress.json').exists() else {}
            save(output/'progress.json', {**previous, 'stage': 'failed', 'updated_at': now(),
                'failure_reason': type(exc).__name__+': '+str(exc)})
        raise


if __name__ == '__main__':
    main()
