#!/usr/bin/env python3
"""Four-drug development comparison of PK-evidence-constrained construction.

No observed curve is loaded until all construction mappings are frozen. The
old ten-drug results remain immutable. All failed cases stay in the report.
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

from pbpk_kb.blueprints.pk_anchored_blueprint import fit_pk_mapping, predict_with_pk_mapping
from pbpk_kb.blueprints.evidence_evaluation import score_case, aggregate_cases
from pbpk_kb.blueprints.prediction_quality import assess_prediction_quality
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/blueprint_design/ten_drug_endpoints_20260923'
BATCH = ROOT / 'experiments/blueprint_design/pk_anchors_20260926'
DRUGS = ('caffeine', 'midazolam', 'sufentanil', 'moxifloxacin')


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BATCH / 'evaluation')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--max-evaluations', type=int, choices=range(4, 25), default=24)
    args = parser.parse_args(argv)
    output = inside(args.output, ROOT)
    if output.exists():
        raise FileExistsError('Choose a new output directory; prior runs are immutable')
    frozen = load(BASE / 'evaluation/protocol.json')
    anchors = load(BATCH / 'inputs/anchors.json')
    cases = [deepcopy(c) for c in frozen['cases'] if c['scenario_kind'] == 'clinical_study' and c['drug'] in DRUGS]
    references = {drug: next(c for c in cases if c['drug'] == drug and c['role'] == 'calibration_data_reconstruction') for drug in DRUGS}
    for c in cases:
        if c['request']['evidence'] != references[c['drug']]['request']['evidence']:
            raise ValueError('Drug evidence must be identical across reference and comparison')
    output.mkdir(parents=True)
    started = time.monotonic()
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in [BASE / 'evaluation/protocol.json', BATCH / 'inputs/anchors.json',
        Path(__file__).resolve(), *[ROOT/'src/pbpk_kb/blueprints'/name for name in ('pk_anchor_optimizer.py', 'pk_anchored_blueprint.py', 'prediction_quality.py')]]}
    for path in (ROOT/'src/pbpk_kb/blueprints').glob('*.py'):
        pins[str(path.relative_to(ROOT))] = sha256(path)
    for path in (ROOT/'src/pbpk_kb/model_runtime/R').glob('*.R'):
        pins[str(path.relative_to(ROOT))] = sha256(path)
    for case in cases:
        path = inside(case['observations_path'], ROOT, exists=True)
        relative = str(path.relative_to(ROOT))
        if sha256(path) != frozen['pins'][relative]:
            raise ValueError('Frozen observations changed: ' + relative)
        pins[relative] = sha256(path)
        path = BASE/'final_results/cases'/case['case_id']/'clinical_comparison.json'
        pins[str(path.relative_to(ROOT))] = sha256(path)
    def verify_pins():
        for relative, expected in pins.items():
            if sha256(ROOT/relative) != expected:
                raise ValueError('Frozen source or code changed during run: ' + relative)
    save(output / 'protocol.json', {'started_at': now(), 'cases': cases, 'reference_cases': {d:c['case_id'] for d,c in references.items()},
        'pins': pins, 'targets': anchors, 'purpose': 'historical development; PK anchors only; no curve fitting',
        'independent_validation': False, 'observation_access': 'only after every mapping and prediction is locked',
        'unchanged_controls': ['fu', 'pKa', 'physiology', 'dosing', 'source_evidence'],
        'fit_targets': ['supplied systemic CL', 'curated IV Vss if available'], 'all_failures_retained': True,
        'partition_policy': 'preserve original; only if PK constraints fail, try the other registered formula; no curve-based selection',
        'maximum_partition_candidates_per_drug':2,
        'resources': {'workers': args.workers, 'cpu_threads_per_native_job': 2, 'solver': 'CPU CVODES',
                      'maximum_fit_evaluations_per_drug': args.max_evaluations, 'maximum_native_output_bytes': 64*1024**2},
        'research_screening_fold_threshold': 2., 'optimization_target_fold_threshold': 1.5,
        'thresholds_are_project_research_conventions_not_clinical_acceptance': True})
    save(output / 'runner.json', {'pid': os.getpid(), 'started_at': now(), 'project': str(ROOT)})
    fits, predictions = {}, {}
    def progress(stage):
        save(output / 'progress.json', {'stage': stage, 'updated_at': now(), 'elapsed_seconds': time.monotonic()-started,
            'fit_completed': len(fits), 'fit_total': len(DRUGS), 'fits_converged': sum(v.get('converged', False) for v in fits.values()),
            'predictions_completed': len(predictions), 'predictions_total': len(cases)})
    progress('fit_pk_evidence')
    def fit(drug):
        try:
            request=deepcopy(references[drug]['request'])
            first=fit_pk_mapping(reference_request=request, vss_anchor=anchors[drug]['target_vss'],
                output_dir=output/'fits'/drug, project_root=ROOT, max_evaluations=args.max_evaluations)
            first['mapping_path']=str((output/'fits'/drug/'mapping_locked.json').relative_to(ROOT))
            if first.get('converged'):
                return first
            request['partition_method']='RodgersRowland' if request['partition_method']=='PKSimStandard' else 'PKSimStandard'
            second=fit_pk_mapping(reference_request=request, vss_anchor=anchors[drug]['target_vss'],
                output_dir=output/'fits'/(drug+'_alternative'), project_root=ROOT, max_evaluations=args.max_evaluations)
            second.update(mapping_path=str((output/'fits'/(drug+'_alternative')/'mapping_locked.json').relative_to(ROOT)),
                primary_attempt_status=first['status'], selection_reason='primary PK evidence constraints failed; comparison curves unopened')
            return second
        except Exception as e:
            return {'status': 'failed', 'converged': False, 'error': type(e).__name__ + ': ' + str(e)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fit, drug):drug for drug in DRUGS}
        for f in as_completed(futures):
            drug = futures[f];fits[drug] = f.result();progress('fit_pk_evidence')
            print(json.dumps({'drug':drug, 'fit_status':fits[drug]['status'], 'converged':fits[drug].get('converged',False)}), flush=True)
    mapping_hashes = {d:sha256(ROOT/fits[d]['mapping_path']) for d in DRUGS if fits[d].get('converged')}
    verify_pins()
    save(output / 'all_mappings_locked.json', {'locked_at':now(), 'mapping_sha256':mapping_hashes, 'fits':fits})
    progress('predict_all_cases')
    def predict(case):
        drug = case['drug']
        if drug not in mapping_hashes:
            return {'status':'not_predicted', 'reason':'PK evidence constraints did not converge; no silent fallback'}
        try:
            path=ROOT/fits[drug]['mapping_path']
            if sha256(path)!=mapping_hashes[drug]:raise ValueError('Mapping changed')
            request=deepcopy(case['request'])
            request['partition_method']=fits[drug]['construction_partition_method']
            return predict_with_pk_mapping(request=request, mapping_path=path,
                output_dir=output/'cases'/case['case_id'], project_root=ROOT)
        except Exception as e:
            return {'status':'failed','error':type(e).__name__+': '+str(e)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(predict,c):c['case_id'] for c in cases}
        for f in as_completed(futures):
            cid=futures[f];predictions[cid]=f.result();progress('predict_all_cases')
            print(json.dumps({'case':cid,'status':predictions[cid]['status']}),flush=True)
    locks={cid:sha256(output/'cases'/cid/'prediction_locked.json') for cid,r in predictions.items() if r['status']=='predicted'}
    save(output/'all_predictions_locked.json',{'locked_at':now(),'predictions':locks})
    verify_pins()
    progress('score_locked_predictions')
    results=[];comparisons=[]
    # This is the first read of concentration outcomes in this runner.
    for case in cases:
        cid=case['case_id'];row={k:case[k] for k in ('case_id','drug','compound_id','study_id','role')}
        row['result']=predictions[cid]
        baseline=load(BASE/'final_results/cases'/cid/'clinical_comparison.json')
        row['baseline_metrics']=baseline.get('metrics',baseline)
        if cid in locks:
            curve_path=output/'cases'/cid/'prediction_locked.json'
            if sha256(curve_path)!=locks[cid]:raise ValueError('Prediction changed before scoring')
            observed_path=inside(case['observations_path'], ROOT, exists=True)
            observed=load(observed_path)
            row['observation_sha256']=sha256(observed_path)
            req=deepcopy(case['request'])
            times=[r['time_min'] for r in observed['observations']]
            if len(times)>=4:
                req['target']['terminal_fit']={'start_h':times[-4]/60,'end_h':times[-1]/60,
                    'min_points':4,'min_adjusted_r2':.98,'min_span_half_lives':1.,
                    'max_auc_extrapolated_fraction':.2,'max_aumc_extrapolated_fraction':.2}
            row['metrics']=score_case(load(curve_path),observed,req)
            row['quality']=assess_prediction_quality(row['metrics'])
            row['scoring_status']='completed'
        else:
            row['scoring_status']='prediction_unavailable'
        results.append(row)
        for version,metrics in [('baseline',row['baseline_metrics']),('pk_anchored',row.get('metrics',{}))]:
            comparisons.append({**{k:row[k] for k in ('case_id','drug','role')},'version':version,
                'gmfe':metrics.get('gmfe'),'ln_rmse':metrics.get('log_rmse'),
                **{name+'_fold_error':metrics.get(name,{}).get('fold_error') for name in ('auc','sampled_cmax','c_last','terminal_half_life')}})
    baseline_rows=[{**r,'metrics':r['baseline_metrics'],'scoring_status':'completed'} for r in results]
    verify_pins()
    summary={'status':'completed' if len(locks)==len(cases) else 'completed_with_failures',
        'case_count':len(cases),'successful_predictions':len(locks),'compound_count':len(DRUGS),
        'independent_validation':False,'fits':fits,'results':results,
        'baseline_aggregate':aggregate_cases(baseline_rows),'candidate_aggregate':aggregate_cases(results),
        'elapsed_seconds':time.monotonic()-started,'finished_at':now()}
    save(output/'summary.json',summary)
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    progress(summary['status'])
    print(json.dumps({'status':summary['status'],'predictions':len(locks),'total':len(cases),'seconds':summary['elapsed_seconds']}),flush=True)
    return 0 if summary['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
