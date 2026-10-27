#!/usr/bin/env python3
"""Historical development evaluation of context correction and bounded fitting."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions
from pbpk_kb.blueprints.source_adaptation import load_source_model, adapt_source_model
from pbpk_kb.blueprints.source_adaptation_audit import audit_execution
from pbpk_kb.blueprints.source_calibration import fit_bounded_source_parameters
from pbpk_kb.blueprints.source_context import apply_source_context, install_calibrated_parameters
from pbpk_kb.blueprints.source_output_grid import ensure_output_times
from pbpk_kb.blueprints.source_workflow import inside, read_curve, save, sha256
from source_context_cases import build_case_manifest, load_observations
from watch_blueprint_suite import ProcessScope

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def summarize_score(rows):
    studies = {}
    for row in rows:
        if row['log_rmse'] is None:
            return None
        studies.setdefault(row['study_group_id'], []).append(row['log_rmse'] ** 2)
    return sum(sum(v)/len(v) for v in studies.values()) / len(studies) if studies else None


def score(scenarios, curves, name):
    lock = {'method': name, 'seed': 0, 'candidate_id': name, 'status': 'completed',
            'predictions_sha256': lock_predictions(curves)}
    return evaluate_locked_predictions(scenarios, curves, lock, device='cpu')


def native_job(case, arm, snapshot, grid, directory, spec, value):
    started = time.monotonic()
    destination = directory / case['case_id'] / arm
    destination.mkdir(parents=True, exist_ok=False)
    result = {'case_id': case['case_id'], 'arm': arm, 'status': 'failed', 'curve': None}
    try:
        manifest = execute_snapshot(snapshot=snapshot, output_dir=destination/'native', project_root=ROOT,
                                    timeout_seconds=300, max_output_bytes=128*1024**2, cpu_threads=2)
        if not manifest['engineering_accepted']:
            raise ValueError(str(manifest.get('error') or manifest.get('stop_reason') or manifest.get('checks')))
        # Independent source-protocol expansion also covers unchanged multi-
        # schema regimens; context/fitting never edits administration here.
        expected = deepcopy(case['native_expectations']['adapted'])
        if case['compound_name'] == 'Rifampicin':
            expected['expectations'].append({'path': spec['path'], 'value': value, 'unit': spec['unit']})
        audit = audit_execution(snapshot=snapshot, execution_manifest=manifest, output_dir=destination/'readback',
                                project_root=ROOT, **expected)
        if not audit['accepted']:
            raise ValueError('Native readback failed: '+str(audit.get('reasons')))
        curve = read_curve(destination/'native', manifest, case['target_output'])
        def present(t):
            index = bisect_left(curve['time_min'], t)
            return any(abs(t-curve['time_min'][i]) <= grid['time_tolerance_min']
                       for i in (index-1,index) if 0 <= i < len(curve['time_min']))
        missing = [t for t in grid['estimated_output_times_min'] if not present(t)]
        if missing:
            raise ValueError('Native output misses requested time points: '+str(missing[:10]))
        pkml = destination/'native'/manifest['pkml_outputs'][0]['path']
        result.update(status='passed', curve=curve, native_readback_verified=True,
                      native_time_coverage_verified=True, pkml_path=str(pkml), pkml_sha256=sha256(pkml),
                      execution_manifest=manifest['manifest_path'])
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    result['elapsed_seconds'] = round(time.monotonic()-started, 3)
    save(destination/'result.json', result)
    return result


def run(args):
    output = inside(args.output, ROOT)
    if output.exists() or not 1 <= args.workers <= 4:
        raise ValueError('Use a new project output and 1..4 workers')
    output.mkdir(parents=True)
    begin = time.monotonic()
    save(output/'runner.json', {**ProcessScope().state(os.getpid()), 'started_at': now()})
    save(ROOT/'runtime/deployment/source_adaptation_pilot.json', {'output':str(output.relative_to(ROOT)), 'registered_at':now()})
    state = {'status':'running','stage':'freeze','started_at':now(), 'completed_native_jobs':0,
             'planned_native_jobs':0,'successful_native_jobs':0,'jobs':{}}

    def update(**values):
        state.update(values, elapsed_seconds=round(time.monotonic()-begin, 2))
        save(output/'progress.json',state)

    update()
    try:
        cases_manifest = build_case_manifest(ROOT)
        cases = cases_manifest['cases']
        native_spec = cases_manifest['rifampicin_pint_spec']
        spec = {'parameter_id':'intestinal_permeability',
                'path':'Rifampicin|Specific intestinal permeability (transcellular)',
                'unit':native_spec['unit'], 'source_value':native_spec['source_value'],
                'lower_bound':native_spec['search_bounds'][0], 'upper_bound':native_spec['search_bounds'][1],
                'scaling':'log', 'evidence_reference':native_spec['source']['path']+'#PI OSP10/P int; bounded 0.25..4x development search window'}
        code_paths = ['scripts/evaluate_source_context.py','scripts/source_context_cases.py','scripts/source_adaptation_cases.py']
        code_paths += ['src/pbpk_kb/blueprints/'+n+'.py' for n in ['source_context','source_adaptation',
            'source_adaptation_audit','source_calibration','source_output_grid','source_workflow',
            'design_runtime','parameter_batch','design_scoring','heldout_evaluation']]
        code_paths += ['src/pbpk_kb/model_runtime/R/'+n+'.R' for n in ['blueprint_snapshot_run',
                       'blueprint_parameter_batch','source_adaptation_audit']]
        protocol = {'frozen_at':now(),'cases':cases_manifest,'parameter_specs':[spec],
                    'code':[{'path':p,'sha256':sha256(ROOT/p)} for p in code_paths],
                    'selection_rule':'Accept calibrated candidate only if study-equal selection log-MSE is strictly below source; otherwise retain source parameter.',
                    'parameter_transfer':'Shared drug permeability applies to ALL Rifampicin scenarios including IV; no route-dependent intrinsic value.',
                    'food_rule':'FeSSIF is a fixed sensitivity hypothesis only; never promote based on Bornemann scores.',
                    'clinical_tier':'historical_development; source-model fitting overlap, not blind validation',
                    'max_native_jobs':50, 'max_rounds':2, 'max_output_points_exclusive':8192, 'workers':args.workers}
        save(output/'protocol.json',protocol)
        prepared, results, aliases, jobs = {}, {}, {}, []
        for case in cases:
            ref = case['source']
            source = load_source_model(ROOT/ref['path'],ref['sha256'],ref['simulation_name'],ROOT)
            blueprint = adapt_source_model(source,{**case['request'],'simulation_name':case['case_id']},ROOT)
            if blueprint['status'] != 'adapted':
                raise ValueError(case['case_id']+str(blueprint['reasons']))
            context = apply_source_context(blueprint,case['context_request'],ROOT)
            if context['status'] != 'adapted':
                raise ValueError(case['case_id']+str(context['reasons']))
            prepared[case['case_id']] = {'blueprint':blueprint,'context':context}
            for arm, snapshot in [('source',blueprint['snapshot']),('context',context['snapshot'])]:
                grid = ensure_output_times(snapshot,case['required_times_min'],ROOT,max_points=8192)
                prepared[case['case_id']][arm] = grid
                if arm == 'context' and not case['context_request']:
                    aliases[(case['case_id'],arm)] = (case['case_id'],'source')
                else:
                    jobs.append((case,arm,grid['snapshot'],grid['manifest'],spec['source_value']))
        save(output/'prepared_models.json', prepared)
        update(cases=len(cases),adaptations_accepted=len(cases),stage='source_and_context_execution')

        def execute_jobs(batch):
            if state['planned_native_jobs'] + len(batch) > 50:
                raise ValueError('Native construction budget exceeded')
            update(planned_native_jobs=state['planned_native_jobs']+len(batch))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(native_job,c,a,s,g,output,spec,v):(c['case_id'],a) for c,a,s,g,v in batch}
                for future in as_completed(futures):
                    key = futures[future]
                    result = future.result()
                    results[key] = result
                    state['jobs']['/'.join(key)] = {k:v for k,v in result.items() if k!='curve'}
                    update(completed_native_jobs=state['completed_native_jobs']+1,
                           successful_native_jobs=state['successful_native_jobs']+int(result['status']=='passed'))
                    print('/'.join(key),result['status'],result.get('error',''),flush=True)

        execute_jobs(jobs)
        for key, target in aliases.items():
            results[key] = {**results[target],'arm':key[1],'shared_prediction':list(target)}
        update(stage='bounded_calibration')
        training, calibration_rows = [], []
        for case in cases:
            if case['role'] != 'calibration':
                continue
            rows = load_observations(case,ROOT,role='calibration')
            calibration_rows.extend(rows)
            if not case['calibration_eligible']:
                continue
            native = results[(case['case_id'],'context')]
            if native['status'] != 'passed':
                raise ValueError('Calibration model failed native validation')
            for row in rows:
                if row['status'] != 'ready':
                    raise ValueError('Declared eligible calibration observations are not valid')
                training.append({'case_id':row['scenario_id'],'study_group_id':row['study_group_id'],
                                 'role':'calibration','pkml_path':Path(native['pkml_path']),
                                 'target_output':case['target_output'],'observations':row['observations']})
        save(output/'calibration_observations.json',calibration_rows)
        fitted = fit_bounded_source_parameters(training,[spec],output/'calibration',ROOT,max_rounds=2,device='auto')
        if fitted['status'] != 'completed':
            raise ValueError('Bounded source calibration failed; see calibration_result.json')
        selected = fitted['selected_vector']
        update(stage='selection_predictions')
        selection_cases = [c for c in cases if c['role']=='selection']
        candidate_jobs = []
        for case in selection_cases:
            grid = prepared[case['case_id']]['context']
            snapshot = install_calibrated_parameters(grid['snapshot'],[spec],selected,'calibration/calibration_result.json')
            candidate_jobs.append((case,'candidate',snapshot,grid['manifest'],selected[0]))
        execute_jobs(candidate_jobs)
        # The full candidate is already fixed by calibration. Selection may only
        # choose that candidate or source; it cannot adjust its values again.
        save(output/'selection_predictions_locked.json',{'locked_at':now(),'jobs':[
            results[(c['case_id'],a)] for c in selection_cases for a in ['context','candidate']]})
        selection_rows = []
        for case in selection_cases:
            selection_rows.extend(dict(row,case_id=case['case_id']) for row in load_observations(case,ROOT,role='selection'))
        selection_scores = {}
        for arm in ['context','candidate']:
            curves = {r['scenario_id']:results[(r['case_id'],arm)]['curve'] for r in selection_rows}
            evaluated = score(selection_rows,curves,arm)
            selection_scores[arm] = evaluated
        source_loss = summarize_score(selection_scores['context']['rows'])
        candidate_loss = summarize_score(selection_scores['candidate']['rows'])
        promote = candidate_loss is not None and source_loss is not None and candidate_loss < source_loss
        decision = {'locked_at':now(),'source_selection_log_mse':source_loss,'candidate_selection_log_mse':candidate_loss,
                    'candidate_promoted':promote,'candidate_values':selected,
                    'retained_values':selected if promote else [spec['source_value']],
                    'evaluation_concentrations_read':False}
        save(output/'selection.json',{'decision':decision,'evaluations':selection_scores})
        update(stage='fixed_candidate_predictions')
        batch = []
        # Always report the fitted candidate on every registered Rifampicin case,
        # even if selection rejected it; no unsuccessful experiment disappears.
        for case in cases:
            key = (case['case_id'],'candidate')
            if key in results:
                continue
            if case['compound_name'] != 'Rifampicin':
                results[key] = {**results[(case['case_id'],'context')],'arm':'candidate','shared_prediction':[case['case_id'],'context']}
                continue
            grid = prepared[case['case_id']]['context']
            snapshot = install_calibrated_parameters(grid['snapshot'],[spec],selected,'calibration/calibration_result.json')
            batch.append((case,'candidate',snapshot,grid['manifest'],selected[0]))
        execute_jobs(batch)
        # An unvalidated food hypothesis is not the algorithm's default output.
        for case in cases:
            key = (case['case_id'],'selected')
            arm = ('candidate' if promote else 'context') if case['compound_name']=='Rifampicin' else 'source'
            results[key] = {**results[(case['case_id'],arm)],'arm':'selected','shared_prediction':[case['case_id'],arm]}
        save(output/'predictions_locked.json',{'locked_at':now(),'jobs':{'/'.join(k):v for k,v in results.items()}})
        lock_sha = sha256(output/'predictions_locked.json')
        update(stage='historical_evaluation')
        observations=[]
        for case in cases:
            observations.extend(dict(row,case_id=case['case_id']) for row in load_observations(
                case,ROOT,role=case['role'],predictions_locked=True))
        save(output/'observations.json',observations)
        evaluations={}
        for arm in ['source','context','candidate','selected']:
            evaluations[arm]={}
            for role in sorted({c['role'] for c in cases}):
                rows=[r for r in observations if r['dataset_role']==role]
                curves={r['scenario_id']:results[(r['case_id'],arm)]['curve'] for r in rows}
                evaluations[arm][role]=score(rows,curves,arm+'_'+role)
        save(output/'evaluations.json',evaluations)
        summary={'completed_at':now(),'case_count':len(cases),'curve_count':len(observations),
                 'native_jobs':state['completed_native_jobs'],'native_passed':state['successful_native_jobs'],
                 'calibration_counts':fitted['counts'],'selection':decision,
                 'calibration_data_counts':{'registered_curves':len(calibration_rows),'fitted_curves':len(training),
                     'blocked_curves':len(calibration_rows)-len(training)},
                 'blocked_calibration_cases':[{'case_id':c['case_id'],'reason':c['calibration_ineligibility_reason']}
                     for c in cases if c['role']=='calibration' and not c['calibration_eligible']],
                 'prediction_lock_sha256':lock_sha,'independent_blind_evaluation':False,
                 'code_unchanged_during_execution':all(sha256(ROOT/r['path'])==r['sha256'] for r in protocol['code']),
                 'elapsed_seconds':round(time.monotonic()-begin,2),
                 'metrics':{a:{r:v['tiers']['reconstruction'] for r,v in rs.items()} for a,rs in evaluations.items()}}
        save(output/'summary.json',summary)
        update(status='completed',stage='completed',candidate_promoted=promote,calibration_counts=fitted['counts'])
        print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},ensure_ascii=False,indent=2),flush=True)
        return summary
    except Exception as error:
        update(status='failed',stage='failed',reason=f'{type(error).__name__}: {error}')
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    summary=run(args)
    return 0 if summary['code_unchanged_during_execution'] and summary['native_jobs']==summary['native_passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
