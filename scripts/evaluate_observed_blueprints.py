#!/usr/bin/env python3
"""Run the existing blueprint algorithm on explicitly curated clinical studies.

Every case needs real observations; there is no hypothetical dose grid. Fit and
formula selection use the declared development study, then all predictions are
locked before comparison scoring. Reading sources for curation is disclosed;
this small development experiment is not a prospective clinical validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import time
import zipfile

from pbpk_kb.blueprints.development_curve_blueprint import fit_development_mapping, predict_with_development_mapping
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint
from pbpk_kb.blueprints.evidence_evaluation import aggregate_cases, observed_rows, score_case
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('PKSimStandard', 'RodgersRowland')
DEVELOPMENT = 'calibration_data_reconstruction'
COMPARISON = 'separate_study_development_comparison'
WITHIN_STUDY = {'same_study_population_development_comparison', 'same_study_individual_development_comparison'}


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--max-evaluations', type=int, choices=range(6, 55), default=18)
    parser.add_argument('--recover-failed-trials', action='store_true')
    args = parser.parse_args(argv)
    output = inside(args.output, ROOT)
    if output.exists():
        raise FileExistsError('Use a new output directory; completed results cannot be overwritten')
    if args.recover_failed_trials and 'recover_failed_trials' not in inspect.signature(fit_development_mapping).parameters:
        raise ValueError('Failure recovery is not implemented in the current algorithm')
    start = time.monotonic()
    cases, anchors, pins = [], {}, {}

    def pin(path):
        path = inside(path, ROOT, exists=True)
        pins[str(path.relative_to(ROOT))] = sha256(path)
        return path

    def verify():
        for name, digest in pins.items():
            if sha256(ROOT / name) != digest:
                raise ValueError('Frozen input or source changed: ' + name)

    for cohort_path in args.cohort:
        cohort = load(pin(cohort_path))
        for item in cohort['cases']:
            case = deepcopy(item)
            for key in ('request_path', 'observations_path'):
                case[key] = str(pin(case[key]).relative_to(ROOT))
            case['request'] = EvidenceBlueprintRequest.model_validate(load(ROOT/case['request_path'])).model_dump(mode='json', exclude_none=True)
            req = case['request']
            if case['role'] not in {DEVELOPMENT, COMPARISON, *WITHIN_STUDY} or case['compound_id'] != req['compound_id']:
                raise ValueError('Case identity/role mismatch')
            if req['dosing']['route'] != 'intravenous':
                raise ValueError('The current calibrated algorithm supports single-dose IV only')
            # Validate curator output before any simulation; no values enter
            # fitting except the explicitly declared development document.
            observed = load(pin(case['observations_path']))
            observed_rows(observed)
            if (observed.get('compound_id') != req['compound_id'] or observed.get('role') != case['role']
                    or observed.get('time_unit') != 'min' or observed.get('unit') != 'umol/l'
                    or observed.get('matrix') != 'plasma'):
                raise ValueError('Curated observations need matching identity, role and units')
            if any(r.get('unit','umol/l') != 'umol/l' or r.get('time_unit','min') != 'min' for r in observed['observations']):
                raise ValueError('Per-observation units disagree with the document')
            times, _ = observed_rows(observed)
            case['evaluation_target'] = deepcopy(req['target'])
            if 'terminal_fit' not in case['evaluation_target'] and len(times) >= 4:
                case['evaluation_target']['terminal_fit'] = {'start_h':times[-4]/60, 'end_h':times[-1]/60,
                    'min_points':4, 'min_adjusted_r2':.98, 'min_span_half_lives':1.,
                    'max_auc_extrapolated_fraction':.2, 'max_aumc_extrapolated_fraction':.2}
            terminal = case['evaluation_target'].get('terminal_fit')
            if terminal and not (times[0]/60 <= terminal['start_h'] < terminal['end_h'] == times[-1]/60):
                raise ValueError('Predeclared scoring window must fit observed sample times')
            blueprint = assemble_evidence_blueprint(req, ROOT)
            if blueprint['status'] != 'assembled':
                raise ValueError('Incomplete evidence: ' + case['case_id'])
            for source in blueprint['manifest']['sources']:
                pin(source['path'])
            cases.append(case)
        for drug, anchor in cohort.get('vss_anchors', {}).items():
            if anchor is not None:
                pin(anchor['source']['path'])
            anchors[drug] = anchor
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Require nonempty, unique cases')
    drugs = sorted({c['drug'] for c in cases})
    references = {}
    for drug in drugs:
        own = [c for c in cases if c['drug'] == drug]
        development = [c for c in own if c['role'] == DEVELOPMENT]
        comparison = [c for c in own if c['role'] != DEVELOPMENT]
        if len(development) != 1 or not comparison:
            raise ValueError('Require exactly one development case and a declared comparison per drug')
        ref = development[0]
        for c in comparison:
            same_study = c['study_id'] == ref['study_id']
            if same_study != (c['role'] in WITHIN_STUDY):
                raise ValueError('Comparison role must correctly identify whether the study is shared')
        if any(any(c['request'][k] != ref['request'][k] for k in ('compound_id','compound_name','evidence','evidence_ledger')) for c in own):
            raise ValueError('Drug evidence must be fixed across study requests')
        forbidden = {}
        for c in own:
            for source in c['request'].get('forbidden_sources', []):
                path = pin(source['path'])
                canonical = str(path.relative_to(ROOT))
                if source['sha256'] != pins[canonical]:
                    raise ValueError('Forbidden source checksum mismatch')
                forbidden[canonical] = {**source, 'path':canonical}
        if any(c['observations_path'] not in forbidden for c in comparison):
            raise ValueError('Comparison observations must be explicitly forbidden during fitting')
        for c in own:
            c['request']['forbidden_sources'] = list(forbidden.values())
        references[drug] = ref
    source_paths = [Path(__file__), *list((ROOT/'src/pbpk_kb/blueprints').glob('*.py')),
                    *list((ROOT/'src/pbpk_kb/model_runtime/R').glob('*.R'))]
    for path in source_paths:
        pin(path)
    output.mkdir(parents=True)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(ROOT)))
    save(output/'runner.json', {'pid':os.getpid(), 'started_at':now(), 'project':str(ROOT)})
    save(output/'protocol.json', {'cases':cases, 'pins':pins, 'vss_anchors':anchors,
        'partition_methods':METHODS, 'max_evaluations':args.max_evaluations,
        'recover_failed_trials':args.recover_failed_trials,
        'comparison_observations_accessed_for_curation':True, 'independent_validation':False,
        'data_scope':'Observed adult single-dose IV studies; no hypothetical grid',
        'selection':'Minimum fixed development objective; comparison errors never select a candidate',
        'source_model_parameters_used':False, 'started_at':now()})
    fits, selections, predictions = {}, {}, {}

    def progress(stage):
        save(output/'progress.json', {'stage':stage, 'updated_at':now(),
            'fits_completed':len(fits), 'fits_total':2*len(drugs),
            'predictions_completed':len(predictions), 'predictions_total':len(cases),
            'elapsed_seconds':time.monotonic()-start})

    def fit(drug, method):
        case = references[drug]
        req = deepcopy(case['request']); req['partition_method'] = method
        folder = output/'fits'/drug/method
        source = {'path':case['observations_path'], 'sha256':pins[case['observations_path']],
                  'case_id':case['case_id'], 'role':DEVELOPMENT}
        kwargs = {'recover_failed_trials':True} if args.recover_failed_trials else {}
        try:
            result = fit_development_mapping(reference_request=req, training_source=source,
                vss_anchor=anchors.get(drug), output_dir=folder, project_root=ROOT,
                max_evaluations=args.max_evaluations, **kwargs)
            return {**result, 'mapping_path':str((folder/'mapping_locked.json').relative_to(ROOT))}
        except Exception as exc:
            return {'status':'failed', 'eligible':False, 'reason':type(exc).__name__+': '+str(exc)}

    progress('fit_existing_algorithm' if not args.recover_failed_trials else 'fit_recovery_algorithm')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fit,d,m):d+'/'+m for d in drugs for m in METHODS}
        for future in as_completed(jobs):
            key = jobs[future]; fits[key] = future.result()
            progress('fit')
            print(json.dumps({'candidate':key, 'status':fits[key]['status'], 'decision':fits[key].get('decision')}), flush=True)
    for drug in drugs:
        eligible = [fits[drug+'/'+m] for m in METHODS if fits[drug+'/'+m].get('eligible')]
        if eligible:
            original = references[drug]['request']['partition_method']
            best = min(eligible, key=lambda f:(f['decision']['objective'], f['construction_partition_method'] != original))
            selections[drug] = {'path':best['mapping_path'], 'sha256':sha256(ROOT/best['mapping_path']),
                                'partition_method':best['construction_partition_method']}
    verify()
    save(output/'models_locked.json', {'fits':fits,'selections':selections,'locked_at':now()})

    def predict(case):
        selected = selections.get(case['drug'])
        if selected is None:
            return {'status':'not_predicted','reason':'No feasible development candidate'}
        req = deepcopy(case['request']); req['partition_method'] = selected['partition_method']
        try:
            if sha256(ROOT/selected['path']) != selected['sha256']:
                raise ValueError('Locked mapping changed')
            return predict_with_development_mapping(request=req, mapping_path=ROOT/selected['path'],
                output_dir=output/'cases'/case['case_id'], project_root=ROOT)
        except Exception as exc:
            return {'status':'failed','reason':type(exc).__name__+': '+str(exc)}

    progress('predict_locked_models')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(predict,c):c['case_id'] for c in cases}
        for future in as_completed(jobs):
            key = jobs[future]; predictions[key] = future.result()
            progress('predict_locked_models')
            print(json.dumps({'case':key, 'status':predictions[key]['status']}), flush=True)
    verify()
    locks = {key:sha256(output/'cases'/key/'prediction_locked.json') for key,value in predictions.items() if value['status']=='predicted'}
    save(output/'predictions_locked.json', {'hashes':locks, 'locked_at':now()})
    results, rows = [], []
    for case in cases:
        cid = case['case_id']
        row = {k:case[k] for k in ('case_id','drug','compound_id','study_id','role')}
        row.update(result=predictions[cid], scoring_status='prediction_unavailable')
        if cid in locks:
            observed = load(ROOT/case['observations_path'])
            req = deepcopy(case['request'])
            req['target'] = case['evaluation_target']
            curve_path = output/'cases'/cid/'prediction_locked.json'
            if sha256(curve_path) != locks[cid]:
                raise ValueError('Locked prediction changed')
            try:
                row.update(metrics=score_case(load(curve_path), observed, req), scoring_status='completed')
            except Exception as exc:
                row.update(scoring_status='failed', scoring_error=type(exc).__name__+': '+str(exc))
        results.append(row)
        m = row.get('metrics', {})
        rows.append({**{k:row[k] for k in ('case_id','drug','role','scoring_status')},
            'study_id':case['study_id'], 'dose_mg':case['request']['dosing']['dose_mg'],
            'infusion_minutes':case['request']['dosing'].get('infusion_minutes'),
            'point_count':m.get('point_count'),
            'gmfe':m.get('gmfe'), 'log_rmse':m.get('log_rmse'),
            **{k+'_'+label:m.get(k,{}).get(field) for k, fields in {
                'auc':{'observed_umol_h_l':'observed_umol_h_l','predicted_umol_h_l':'predicted_umol_h_l'},
                'sampled_cmax':{'observed_umol_l':'observed_umol_l','predicted_umol_l':'predicted_umol_l'},
                'c_last':{'observed_umol_l':'observed_umol_l','predicted_umol_l':'predicted_umol_l'},
                'sampled_tmax':{'observed_h':'observed_h','predicted_h':'predicted_h','absolute_error_h':'absolute_error_h'}
            }.items() for label,field in fields.items()},
            **{k+'_fold_error':m.get(k,{}).get('fold_error') for k in ('auc','sampled_cmax','c_last','terminal_half_life')}})
    status = 'completed' if all(r['scoring_status']=='completed' for r in results) else 'completed_with_failures'
    summary = {'status':status, 'compound_count':len(drugs), 'case_count':len(cases),
        'successful_predictions':len(locks), 'independent_validation':False,
        'fits':fits, 'selections':selections, 'results':results, 'aggregates':aggregate_cases(results),
        'native_fit_calls':sum(f.get('evaluation_count',0) for f in fits.values()),
        'elapsed_seconds':time.monotonic()-start, 'finished_at':now()}
    save(output/'summary.json', summary)
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    progress(status)
    print(json.dumps({'status':status,'seconds':summary['elapsed_seconds'],'aggregates':summary['aggregates']}),flush=True)
    return 0 if status=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
