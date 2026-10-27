#!/usr/bin/env python3
"""Four-drug, study-separated development comparison of middle-out blueprints.

Only development concentrations enter fitting and candidate selection. Historical
comparison outcomes are read after every decision and prediction has been locked.
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

from pbpk_kb.blueprints.development_curve_blueprint import fit_development_mapping, predict_with_development_mapping
from pbpk_kb.blueprints.evidence_evaluation import score_case, aggregate_cases
from pbpk_kb.blueprints.prediction_quality import assess_prediction_quality
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/blueprint_design/ten_drug_endpoints_20260923'
PREVIOUS = ROOT / 'experiments/blueprint_design/pk_anchors_20260926'
BATCH = ROOT / 'experiments/blueprint_design/middle_out_20260926'
DRUGS = ('caffeine', 'midazolam', 'sufentanil', 'moxifloxacin')
METHODS = ('PKSimStandard', 'RodgersRowland')


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BATCH / 'evaluation')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--max-evaluations', type=int, choices=range(6, 25), default=18)
    parser.add_argument('--reuse-candidates', type=Path,
                        help='Completed pre-correction run; reuse only candidates unaffected by the isolated CL/AUMC repair')
    args = parser.parse_args(argv)
    output = inside(args.output, ROOT)
    if output.exists():
        raise FileExistsError('Choose a new directory; prior experiments cannot be overwritten')
    frozen = load(BASE / 'evaluation/protocol.json')
    anchors = load(PREVIOUS / 'inputs/anchors.json')
    cases = [deepcopy(c) for c in frozen['cases'] if c['scenario_kind'] == 'clinical_study' and c['drug'] in DRUGS]
    references = {d: next(c for c in cases if c['drug'] == d and c['role'] == 'calibration_data_reconstruction') for d in DRUGS}
    started = time.monotonic()
    pins = {}
    files = [Path(__file__).resolve(), BASE / 'evaluation/protocol.json', PREVIOUS / 'inputs/anchors.json',
        PREVIOUS / 'evaluation_verified/summary.json',
        *list((ROOT / 'src/pbpk_kb/blueprints').glob('*.py')),
        *list((ROOT / 'src/pbpk_kb/model_runtime/R').glob('*.R'))]
    training_sources = {}
    for c in cases:
        path = inside(c['observations_path'], ROOT, exists=True)
        if sha256(path) != frozen['pins'][str(path.relative_to(ROOT))]:
            raise ValueError('Frozen observations changed')
        files.extend([path, BASE / 'final_results/cases' / c['case_id'] / 'clinical_comparison.json'])
        if c['request']['evidence'] != references[c['drug']]['request']['evidence']:
            raise ValueError('Drug evidence must remain identical across studies')
        if c['case_id'] == references[c['drug']]['case_id']:
            if c['drug'] == 'caffeine':
                # The legacy file lacks document-level identity/units. This
                # explicit metadata normalization preserves every sample.
                normalized = BATCH / 'inputs/caffeine_development_observations.json'
                doc = load(normalized)
                if doc['source']['sha256'] != sha256(path) or doc['observations'] != load(path)['observations']:
                    raise ValueError('Caffeine metadata normalization must not change observations')
                path = normalized
                files.append(path)
            training_sources[c['drug']] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                'role': c['role'], 'case_id': c['case_id']}
    for path in files:
        pins[str(path.relative_to(ROOT))] = sha256(path)

    previous_fits, reuse_provenance = {}, None
    if args.reuse_candidates:
        previous_dir = inside(args.reuse_candidates, ROOT, exists=True)
        old_protocol = load(previous_dir/'protocol.json')
        old_summary = load(previous_dir/'summary.json')
        if old_summary['status'] != 'completed' or old_protocol['cases'] != cases:
            raise ValueError('Candidate reuse requires a completed identical case protocol')
        if old_protocol['training_sources'] != training_sources or old_protocol['resources']['fit_calls_per_candidate'] != args.max_evaluations:
            raise ValueError('Cannot reuse candidates with changed data or evaluation budget')
        archive_path = previous_dir/'code_before_integral_fix/manifest.json'
        archive = load(archive_path)
        allowed_repairs = {'src/pbpk_kb/blueprints/development_curve_blueprint.py',
            'src/pbpk_kb/blueprints/pk_anchor_moments.py', 'scripts/evaluate_development_curve_blueprints.py'}
        for relative,expected in old_protocol['pins'].items():
            if relative in allowed_repairs:
                record = archive[relative]
                copied = inside(record['copy'], ROOT, exists=True)
                if record['sha256'] != expected or sha256(copied) != expected:
                    raise ValueError('Missing verified pre-correction source snapshot')
                pins[record['copy']] = expected
            elif sha256(ROOT/relative) != expected:
                raise ValueError('Reuse would mix changed inputs or objective code: '+relative)
        for path in (previous_dir/'protocol.json', previous_dir/'summary.json', archive_path):
            pins[str(path.relative_to(ROOT))] = sha256(path)
        previous_fits = old_summary['fits']
        for fit in previous_fits.values():
            if fit.get('mapping_path'):
                path = ROOT/fit['mapping_path']
                pins[str(path.relative_to(ROOT))] = sha256(path)
                # Any CL-unavailable trial in a CL-only candidate can now
                # change the search trajectory; do not reuse that candidate.
                affected = False
                for result_path in sorted(path.parent.glob('attempt_*/result.json')):
                    result = load(result_path)
                    pins[str(result_path.relative_to(ROOT))] = sha256(result_path)
                    affected |= (fit.get('target_vss_l') is None and result.get('status') == 'predicted'
                                 and result.get('clearance_l_h') is None)
                fit['affected_by_cl_integral_correction'] = affected
        reuse_provenance = {'directory':str(previous_dir.relative_to(ROOT)),
            'source_changes':sorted(allowed_repairs), 'original_source_archive':str(archive_path.relative_to(ROOT)),
            'policy':'Reuse only eligible candidates without any CL-unavailable trial when Vss is not required; objective and all data unchanged',
            'no_comparison_metric_used_for_reuse':True}

    def verify_pins():
        for name, digest in pins.items():
            if sha256(ROOT / name) != digest:
                raise ValueError('Frozen input/code changed during execution: ' + name)

    output.mkdir(parents=True)
    save(output / 'protocol.json', {'started_at': now(), 'cases': cases, 'pins': pins,
        'reference_cases': {d: c['case_id'] for d,c in references.items()}, 'training_sources': training_sources,
        'partition_candidates': METHODS, 'independent_validation': False,
        'purpose': 'Historical development evaluation; bounded middle-out PBPK with two continuous parameters',
        'selection': 'Minimum frozen regularized objective on development curve; prefer original formula on exact tie',
        'selection_is_in_sample': True, 'separate_selection_study_available': False,
        'candidate_reuse': reuse_provenance,
        'comparison_access': 'Only after all parameter/formula decisions and predictions are locked',
        'objective': 'mean(log(P/O)^2) + .1*(log(CL/CLref)/log(2))^2 + optional .1*(log(Vss/Vssref)/log(2))^2 + .01*((logMA-logP)/2.5)^2',
        'regularization_not_physiological_CI_or_independent_observations': True,
        'no_additional_endpoint_loss': 'AUC/Cmax/Clast are derived from the same curve, and are scored separately',
        'parameter_limits': {'lipophilicity': 'source logP +/- 2.5', 'clearance_scale': [.25,4.]},
        'controls': ['original evidence', 'human physiology', 'fu', 'pKa', 'dose', 'infusion', 'digitized observation times'],
        'sampling_limitation': 'Hohmann nominal first sample 5 min versus digitized 4.8 min; preserved identically for all methods, no fitted time shift',
        'comparators': ['frozen_original', 'pk_anchored', 'curve_fixed_partition', 'curve_selected'],
        'all_failures_retained': True, 'resources': {'workers': args.workers, 'threads_per_native_job':2,
            'fit_calls_per_candidate': args.max_evaluations, 'candidate_count': 8,
            'maximum_native_output_bytes_per_call': 64*1024**2, 'native_timeout_seconds':180}})
    save(output / 'runner.json', {'pid': os.getpid(), 'started_at': now(), 'project': str(ROOT)})
    fits, predictions = {}, {}

    def progress(stage):
        save(output / 'progress.json', {'stage': stage, 'updated_at': now(),
            'elapsed_seconds': time.monotonic()-started, 'candidates_completed': len(fits),
            'candidates_total': 8, 'predictions_completed': len(predictions)})

    progress('fit_development_curves')

    def fit(drug, method):
        previous = previous_fits.get(drug+'/'+method)
        if previous and previous.get('eligible') and not previous.get('affected_by_cl_integral_correction'):
            return {**previous, 'reused_from':str(previous_dir.relative_to(ROOT)),
                    'reuse_reason':'All prior callback CL values available; isolated CL/AUMC repair cannot change this trajectory'}
        req = deepcopy(references[drug]['request'])
        req['partition_method'] = method
        folder = output / 'fits' / drug / method
        try:
            result = fit_development_mapping(reference_request=req, training_source=training_sources[drug],
                vss_anchor=anchors[drug]['target_vss'], output_dir=folder, project_root=ROOT,
                max_evaluations=args.max_evaluations)
            return {**result, 'mapping_path': str((folder/'mapping_locked.json').relative_to(ROOT))}
        except Exception as e:
            return {'status': 'failed', 'eligible': False, 'error': type(e).__name__+': '+str(e)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fit,d,m):(d,m) for d in DRUGS for m in METHODS}
        for future in as_completed(futures):
            drug,method = futures[future]
            key = drug + '/' + method
            fits[key] = future.result()
            progress('fit_development_curves')
            print(json.dumps({'candidate':key,'status':fits[key]['status'],
                'decision':fits[key].get('decision')}), flush=True)
    selections = {}
    for drug in DRUGS:
        original = references[drug]['request']['partition_method']
        candidates = [fits[drug+'/'+m] for m in METHODS if fits[drug+'/'+m].get('eligible')]
        if candidates:
            chosen = min(candidates,key=lambda x:(x['decision']['objective'], x['construction_partition_method'] != original))
            selections[drug] = {'mapping_path':chosen['mapping_path'], 'sha256':sha256(ROOT/chosen['mapping_path']),
                'partition_method':chosen['construction_partition_method'], 'objective':chosen['decision']['objective']}
    verify_pins()
    mapping_hashes = {k:sha256(ROOT/v['mapping_path']) for k,v in fits.items() if v.get('eligible')}
    save(output / 'all_mappings_locked.json', {'locked_at':now(), 'selections':selections,
        'candidate_mapping_hashes':mapping_hashes, 'fits':fits})

    # Reuse an identical mapping/case prediction for the fixed-formula ablation.
    aliases, jobs = {}, {}
    for case in cases:
        cid, drug = case['case_id'], case['drug']
        selected = selections.get(drug)
        fixed = fits[drug+'/'+case['request']['partition_method']]
        paths = {'curve_selected':selected['mapping_path'] if selected else None,
                 'curve_fixed_partition':fixed.get('mapping_path') if fixed.get('eligible') else None}
        for version,path in paths.items():
            key = cid + '/' + version
            if path is None:
                predictions[key] = {'status':'not_predicted','reason':'No eligible candidate; no silent fallback'}
                continue
            jobkey = (cid,path)
            if jobkey not in jobs:
                jobs[jobkey] = (key,case,path,sha256(ROOT/path))
            aliases[key] = jobs[jobkey][0]
    progress('predict_locked_models')

    def predict(job):
        key,case,path,digest = job
        try:
            if sha256(ROOT/path) != digest:
                raise ValueError('Frozen mapping changed')
            req = deepcopy(case['request'])
            req['partition_method'] = load(ROOT/path)['construction_partition_method']
            return predict_with_development_mapping(request=req, mapping_path=ROOT/path,
                output_dir=output/'cases'/key, project_root=ROOT)
        except Exception as e:
            return {'status':'failed','error':type(e).__name__+': '+str(e)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(predict,job):job[0] for job in jobs.values()}
        for future in as_completed(futures):
            key = futures[future]
            predictions[key] = future.result()
            progress('predict_locked_models')
            print(json.dumps({'prediction':key,'status':predictions[key]['status']}),flush=True)
    locks = {k:sha256(output/'cases'/k/'prediction_locked.json') for k,v in predictions.items() if v['status']=='predicted'}
    save(output/'all_predictions_locked.json', {'locked_at':now(),'predictions':locks,'aliases':aliases})
    verify_pins()
    progress('score_locked_predictions')
    old = {r['case_id']:r for r in load(PREVIOUS/'evaluation_verified/summary.json')['results']}
    results, comparisons = [], []
    for case in cases:
        cid = case['case_id']
        observed = load(ROOT/case['observations_path'])
        req = deepcopy(case['request'])
        times = [r['time_min'] for r in observed['observations']]
        req['target']['terminal_fit'] = {'start_h':times[-4]/60,'end_h':times[-1]/60,
            'min_points':4,'min_adjusted_r2':.98,'min_span_half_lives':1.,
            'max_auc_extrapolated_fraction':.2,'max_aumc_extrapolated_fraction':.2}
        for version in ('frozen_original','pk_anchored','curve_fixed_partition','curve_selected'):
            row = {k:case[k] for k in ('case_id','drug','compound_id','study_id','role')}
            row.update(version=version, observation_sha256=sha256(ROOT/case['observations_path']))
            if version in ('frozen_original','pk_anchored'):
                row['metrics'] = old[cid]['baseline_metrics' if version=='frozen_original' else 'metrics']
                row['scoring_status'] = 'completed'
            else:
                key = cid+'/'+version
                actual = aliases.get(key,key)
                row['prediction_key'] = actual
                row['result'] = predictions[actual]
                if actual in locks:
                    curve_path = output/'cases'/actual/'prediction_locked.json'
                    if sha256(curve_path) != locks[actual]:
                        raise ValueError('Prediction changed before scoring')
                    row['metrics'] = score_case(load(curve_path),observed,req)
                    row['quality'] = assess_prediction_quality(row['metrics'])
                    row['scoring_status'] = 'completed'
                else:
                    row['scoring_status'] = 'prediction_unavailable'
            results.append(row)
            metrics = row.get('metrics',{})
            comparisons.append({**{k:row[k] for k in ('case_id','drug','role','version')},
                'scoring_status':row['scoring_status'], 'gmfe':metrics.get('gmfe'),'ln_rmse':metrics.get('log_rmse'),
                **{name+'_fold_error':metrics.get(name,{}).get('fold_error') for name in ('auc','sampled_cmax','c_last','terminal_half_life')}})
    verify_pins()
    main_rows = [r for r in results if r['version']=='curve_selected']
    successful = sum(r['scoring_status']=='completed' for r in main_rows)
    summary = {'status':'completed' if successful==len(cases) else 'completed_with_failures',
        'case_count':len(cases),'successful_predictions':successful,'compound_count':len(DRUGS),
        'independent_validation':False,'selections':selections,'fits':fits,'results':results,
        'aggregates':{v:aggregate_cases([r for r in results if r['version']==v]) for v in ('frozen_original','pk_anchored','curve_fixed_partition','curve_selected')},
        'unique_native_prediction_count':len(jobs),
        'reused_candidate_count':sum('reused_from' in f for f in fits.values()),
        'new_native_fit_calls':sum(f.get('evaluation_count',0) for f in fits.values() if 'reused_from' not in f),
        'elapsed_seconds':time.monotonic()-started,'finished_at':now()}
    save(output/'summary.json',summary)
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(comparisons[0]))
        writer.writeheader();writer.writerows(comparisons)
    progress(summary['status'])
    print(json.dumps({'status':summary['status'],'predictions':successful,'seconds':summary['elapsed_seconds']}),flush=True)
    return 0 if successful==len(cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
