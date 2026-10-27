#!/usr/bin/env python3
"""Freeze complete clinical references, run existing PBPK constructors, score 11 endpoints.

This runner never fits against evaluation concentrations or selects parameters
using their errors. Numerical and source qualification precede every simulation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from pbpk_kb.blueprints.development_curve_blueprint import predict_with_development_mapping
from pbpk_kb.blueprints.evidence_blueprint import run_evidence_blueprint
from pbpk_kb.blueprints.full_endpoint_evaluation import qualify_endpoint_references, score_full_endpoints
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    cohort_path = inside(args.cohort, ROOT, exists=True)
    output = inside(args.output, ROOT)
    if output.exists():
        raise FileExistsError('Use a new output directory; locked results cannot be overwritten')
    cohort = read(cohort_path)
    cases = cohort['cases']
    assert cases and len({c['case_id'] for c in cases}) == len(cases)
    assert len(cases) <= 12, 'This is a bounded small pilot'
    pins = {str(cohort_path.relative_to(ROOT)): sha256(cohort_path)}
    for path in [Path(__file__), *list((ROOT/'src/pbpk_kb/blueprints').glob('*.py'))]:
        pins[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in cohort['source_hashes'].items():
        path = inside(name, ROOT, exists=True)
        assert sha256(path) == digest, 'Clinical source changed: '+name
        pins[name] = digest
    prepared, qualification = {}, {}
    for case in cases:
        cid = case['case_id']
        assert cid.replace('_', '').isalnum(), 'Case IDs must be simple path components'
        assert case['clinical_source_review_passed'] is True
        assert case['evaluation_used_for_fitting'] is False
        request_path = inside(case['request_path'], ROOT, exists=True)
        observed_path = inside(case['observations_path'], ROOT, exists=True)
        req, obs = read(request_path), read(observed_path)
        for path in (request_path, observed_path):
            pins[str(path.relative_to(ROOT))] = sha256(path)
        forbidden = {sha256(observed_path)}
        for evidence in req['evidence']:
            source = evidence['source']; path = inside(source['path'], ROOT, exists=True)
            assert sha256(path) == source['sha256'] and sha256(path) not in forbidden
            pins[str(path.relative_to(ROOT))] = sha256(path)
        ledger = req['evidence_ledger']; path = inside(ledger['path'], ROOT, exists=True)
        assert sha256(path) == ledger['sha256']
        pins[str(path.relative_to(ROOT))] = sha256(path)
        if case.get('mapping_path'):
            path = inside(case['mapping_path'], ROOT, exists=True)
            pins[str(path.relative_to(ROOT))] = sha256(path)
            training = read(path)['training_source']
            assert training['sha256'] not in forbidden
            training_path = inside(training['path'], ROOT, exists=True)
            assert sha256(training_path) == training['sha256']
            pins[str(training_path.relative_to(ROOT))] = sha256(training_path)
        kwargs = {'terminal_fit': case['terminal_fit'], 'initial_condition': case.get('initial_condition')}
        qualification[cid] = qualify_endpoint_references(obs, req, **kwargs)
        prepared[cid] = (req, obs, kwargs)
    output.mkdir(parents=True)
    save(output/'reference_qualification.json', qualification)
    incomplete = {cid: q['unavailable_endpoints'] for cid, q in qualification.items() if not q['all_11_qualified']}
    if incomplete:
        save(output/'progress.json', {'status': 'blocked_incomplete_references', 'cases': incomplete})
        raise ValueError('No simulation started: incomplete references '+str(incomplete))
    protocol = {'created_at': now(), 'cases': cases, 'pins': pins,
        'all_references_qualified_before_prediction': True,
        'parameter_selection_from_evaluation': False, 'fitting_performed_in_this_run': False,
        'case_selection_basis': 'clinical-source and reference completeness, never prediction errors',
        'resources': {'workers': args.workers, 'cpu_threads_per_native_job': 2, 'gpu_required': False},
        'study_separation': cohort['study_separation'],
        'independent_validation': False,
        'limitation': 'Historical public-data pilot; source inspection is not blinded; no new-drug generalization claim'}
    save(output/'protocol.json', protocol)
    save(output/'runner.json', {'pid': os.getpid(), 'project': str(ROOT), 'started_at': now()})
    started = time.monotonic()
    results = {}

    def progress(status):
        payload = {'status': status, 'updated_at': now(), 'cases_total': len(cases),
                   'predictions_finished': len(results), 'elapsed_seconds': round(time.monotonic()-started, 2)}
        save(output/'progress.json', payload)
        print(json.dumps(payload), flush=True)

    def verify_pins():
        for name, digest in pins.items():
            if sha256(ROOT/name) != digest:
                raise ValueError('Frozen input or code changed: '+name)

    def predict(case):
        req = prepared[case['case_id']][0]
        folder = output/'cases'/case['case_id']
        folder.parent.mkdir(exist_ok=True)
        try:
            if case.get('mapping_path'):
                return predict_with_development_mapping(request=req, mapping_path=ROOT/case['mapping_path'],
                    output_dir=folder, project_root=ROOT)
            return run_evidence_blueprint(request=req, output_dir=folder, project_root=ROOT, execute=True)
        except Exception as exc:
            return {'status': 'failed', 'error': type(exc).__name__+': '+str(exc)}

    progress('predicting')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(predict, case): case['case_id'] for case in cases}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
            progress('predicting')
    verify_pins()
    locks = {}
    for case in cases:
        path = output/'cases'/case['case_id']/'prediction_locked.json'
        if results[case['case_id']].get('status') == 'predicted' and path.is_file():
            locks[case['case_id']] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    save(output/'predictions_lock.json', {'created_at': now(), 'all_predictions_locked_before_scoring': True,
                                         'predictions': locks})
    progress('scoring')
    scores, rows = {}, []
    for case in cases:
        cid = case['case_id']; req, obs, kwargs = prepared[cid]
        curve = None
        if cid in locks:
            path = ROOT/locks[cid]['path']; assert sha256(path) == locks[cid]['sha256']
            curve = read(path)
        score = score_full_endpoints(obs, req, curve, **kwargs,
            prediction_failure_reason=results[cid].get('error', results[cid].get('status')) if curve is None else None)
        assert score['reference_qualification'] == qualification[cid]
        scores[cid] = score
        for endpoint, metric in score['endpoints'].items():
            rows.append({'case_id': cid, 'drug': req['compound_name'], 'study': case['study'],
                'method': case['method'], 'dose_mg': req['dosing']['dose_mg'],
                'infusion_minutes': req['dosing']['infusion_minutes'], 'endpoint': endpoint,
                **{key: metric.get(key) for key in ('status','unit','prediction','reference','fold_error',
                    'absolute_error_h','gmfe','log_rmse','reference_type','prediction_reason','reference_reason')}})
    save(output/'scores.json', scores)
    with (output/'endpoint_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    verify_pins()
    summary = {'status': 'completed' if all(s['scored_endpoints'] == 11 for s in scores.values()) else 'completed_with_prediction_failures',
        'finished_at': now(), 'case_count': len(cases), 'compound_count': len({c['drug'] for c in cases}),
        'all_11_references_per_case': True, 'native_predictions_completed': len(locks),
        'endpoint_comparisons_attempted': len(rows),
        'endpoint_comparisons_completed': sum(s['scored_endpoints'] for s in scores.values()),
        'case_metrics': {cid: {'curve_gmfe': s['endpoints']['plasma_concentration_time']['gmfe'],
            'scored_endpoints': s['scored_endpoints'], 'scalar_within_twofold': sum(
                r['fold_error'] is not None and r['fold_error'] <= 2 for r in s['endpoints'].values()),
            'scalar_fold_denominator': 9} for cid, s in scores.items()},
        'elapsed_seconds': round(time.monotonic()-started, 2),
        'no_fitting_or_selection_on_evaluation_data': True, 'source_and_code_hashes_unchanged': True,
        'independent_validation': False}
    save(output/'summary.json', summary)
    progress(summary['status'])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
