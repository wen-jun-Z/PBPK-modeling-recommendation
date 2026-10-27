#!/usr/bin/env python3
"""Independent ten-drug structure/stability pilot. Never modifies the paper."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from run_blueprint_learning import Controller, Paused, now, read, ref, checked, verify
import run_blueprint_evidence_policy as inputs
import run_blueprint_capability as previous

VERSION = 'robust-blueprint-structure-1'
DEFAULT = ROOT / 'experiments/blueprint_design/robust_structure_20260930'
BASE = ROOT / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
CAPABILITY = ROOT / 'experiments/blueprint_design/blueprint_capability_20260930'
METHODS = ('current', 'evidence_flexible', 'balanced_exchange')
SCENARIOS = ('available', 'missing_fu', 'missing_cl', 'missing_both')
FORMULAS = ('PKSimStandard', 'RodgersRowland')
ENGINE_NAMES = {'current': 'pbpk_kb.blueprint_policy.transport_calibration',
    'evidence_flexible': 'pbpk_kb.blueprint_policy.robust_exchange_calibration',
    'balanced_exchange': 'pbpk_kb.blueprint_policy.robust_exchange_calibration'}
api = previous.api


def own_run(path):
    raw = Path(path) if Path(path).is_absolute() else ROOT / path
    run = inside(raw, ROOT)
    if raw.is_symlink() or run.parent != DEFAULT.parent or not run.name.startswith('robust_structure_'):
        raise ValueError('Use a dedicated immediate robust_structure_ experiment directory')
    return run


def engine(method): return importlib.import_module(ENGINE_NAMES[method])
def variant(method): return 'flow_exchange' if method == 'current' else method
def key(cid, formula): return cid + '__' + formula


def signature(request, target, training, method):
    from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
    request = EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)
    target = EvidenceBlueprintRequest.model_validate(target).model_dump(mode='json', exclude_none=True)
    if previous.values(request) != previous.values(target):
        raise ValueError('Target drug evidence differs from the development construction')
    return method + '_' + previous.signature(request, target, training, method)[:24]


def paper_pins():
    """Guard existing publication sources and all their result assets, read-only."""
    folders = [ROOT / 'results', BASE / 'paper', CAPABILITY / 'paper']
    return {str(p.relative_to(ROOT)): sha256(p) for folder in folders for p in folder.rglob('*')
            if p.is_file() and not p.is_symlink()}


def fill_visible(rows, fallback):
    """Fill only absent fu/total CL; never inspect hidden source evidence."""
    selected = deepcopy(rows); actions = []
    if len({r['parameter'] for r in selected}) != len(selected):
        raise ValueError('Duplicate primary evidence')
    for parameter in ('fraction_unbound', 'total_clearance'):
        existing = [r for r in selected if r['parameter'] == parameter]
        if len(existing) > 1: raise ValueError('Conflicting primary evidence')
        if existing: continue
        candidates = [r for r in fallback if r['parameter'] == parameter and r.get('basis') == 'predicted']
        if len(candidates) != 1: raise ValueError('One authorized RF fallback required for ' + parameter)
        selected.append(deepcopy(candidates[0])); actions.append({'parameter': parameter, 'action': 'rf_fallback'})
    by_name = {r['parameter']: r for r in selected}
    if not 0 < float(by_name['fraction_unbound']['value']) <= 1:
        raise ValueError('Invalid fu')
    total = by_name['total_clearance']
    if total['unit'] != 'ml/min/kg' or not math.isfinite(float(total['value'])) or float(total['value']) <= 0:
        raise ValueError('Invalid total clearance')
    renal = by_name.get('renal_clearance')
    if renal and (renal['unit'] != total['unit'] or not 0 <= float(renal['value']) < float(total['value'])):
        raise ValueError('RF total clearance is incompatible with retained renal evidence')
    return selected, actions


def candidate_checks(records, summary):
    """Never claim nonregression on a smaller successful intersection."""
    def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    def curve(row): return row.get('curve_metrics',{}).get('gmfe')
    baseline={(r['case_id'],r['scenario']):r for r in records if r['method']=='current'}
    output={}
    for method in METHODS[1:]:
        conditions=[]; regressions=[]; endpoints=[]; incomplete=[]
        for scenario in SCENARIOS:
            new=next(r for r in summary if r['method']==method and r['scenario']==scenario)
            old=next(r for r in summary if r['method']=='current' and r['scenario']==scenario)
            a,b=new['curve_gmfe']['estimate'],old['curve_gmfe']['estimate']
            conditions.append(new['scored']==old['scored']==10 and new['failed']==old['failed']==0
                and finite(a) and finite(b) and a<=b+1e-12)
        for row in (r for r in records if r['method']==method):
            old=baseline.get((row['case_id'],row['scenario']),{})
            a,b=curve(row),curve(old)
            if not (finite(a) and finite(b) and a>0 and b>0):
                incomplete.append({'drug':row['drug'],'scenario':row['scenario']});continue
            if a>b*(1+1e-10): regressions.append({'drug':row['drug'],'scenario':row['scenario'],'ratio':a/b})
            for name, value in row.get('endpoints',{}).items():
                prior=old.get('endpoints',{}).get(name,{})
                a,b=value.get('error'),prior.get('error')
                if finite(a) and finite(b) and value.get('error_metric')==prior.get('error_metric') and a>b+1e-10:
                    endpoints.append({'drug':row['drug'],'scenario':row['scenario'],'endpoint':name,
                        'error_metric':value['error_metric'],'candidate_error':a,'current_error':b})
        output[method]={'all_four_aggregate_curve_errors_nonincreasing':all(conditions),
            'individual_curve_regressions':regressions,'endpoint_error_regressions':endpoints,'incomplete_pairs':incomplete,
            'automatic_promotion':False,'paper_updated':False,
            'interpretation':'Development result only; individual and endpoint errors can worsen even if aggregate improves.'}
    return output


def prepare(run, workers=14):
    run = own_run(run)
    if run.exists(): raise FileExistsError('Preserve existing run; use resume')
    if type(workers) is not int or not 1 <= workers <= 14: raise ValueError('Use 1–14 workers')
    original, old_state = read(BASE / 'protocol.json'), read(BASE / 'state.json')
    cap, cap_state = read(CAPABILITY / 'protocol.json'), read(CAPABILITY / 'state.json')
    if old_state['status'] != 'complete' or cap_state['status'] != 'complete':
        raise ValueError('Frozen controls must be complete')
    checked(old_state['protocol']); checked(cap_state['protocol'])
    pins = {}; api.pin_refs([original, old_state, cap, cap_state], pins)
    publication = paper_pins(); verify(publication)
    contracts = {m: {'module': ENGINE_NAMES[m], 'version': engine(m).VERSION,
        'config': deepcopy(engine(m).CONFIG), 'code_pins': engine(m)._code_pins(ROOT)} for m in METHODS}
    for contract in contracts.values(): pins.update(contract['code_pins'])
    cached = {}
    for unit in cap['units']:
        if (unit['method'], unit['sampling']) != ('evidence_policy', 'full') or unit['scenario'] not in ('available', 'missing_fu'):
            continue
        value = deepcopy(cap_state['computations'][unit['computation_id']])
        mapping = checked(value['mapping']); target = checked(value['request'])
        base = read((ROOT / value['mapping']['path']).parent / 'input.json')['request']
        train = checked(mapping['training_source'])
        cid = signature(base, target, train, 'current')
        value.update(reused_from=ref(CAPABILITY / 'state.json'), original_resources=value.get('resources', {}),
            resources={'total_native_calls': 0, 'linear_solver_calls': 0, 'wall_seconds': 0, 'cache_reused': True})
        cached[cid] = value
    run.mkdir(); (run / 'inputs').mkdir()
    specs, reused, units, audit = {}, {}, [], []
    changed = {s: [] for s in SCENARIOS}
    cases = deepcopy(original['cases'])
    for case in cases:
        base, target = (checked(case['requests']['evidence'][k]) for k in ('development', 'target'))
        rf = checked(case['requests']['rf']['development'])
        training = checked(original['computations'][case['case_id'] + '__full__flow_exchange']['training_source'])
        train_ref = {**inputs.store_input(run, inputs.clean_training(training)), 'role': 'calibration_data_reconstruction'}
        reference_values = {r['parameter']: r['value'] for r in base['evidence']}
        for scenario in SCENARIOS:
            removed = ({'fraction_unbound'} if scenario == 'missing_fu' else {'total_clearance'} if scenario == 'missing_cl'
                       else {'fraction_unbound', 'total_clearance'} if scenario == 'missing_both' else set())
            rows = [inputs.visible_record(r) for r in base['evidence'] if r['parameter'] not in removed]
            fallback = [inputs.visible_record(r) for r in rf['evidence'] if r['parameter'] in ('fraction_unbound', 'total_clearance')]
            rows, fallback, view = inputs.bind_view(run, rows, fallback)
            selected, actions = fill_visible(rows, fallback)
            request = inputs.selected_request(run, base, selected)
            predict_request = deepcopy(request)
            for field in ('human', 'dosing', 'target'): predict_request[field] = deepcopy(target[field])
            changed_parameters = [r['parameter'] for r in selected if r['value'] != reference_values.get(r['parameter'])]
            if changed_parameters: changed[scenario].append(case['case_id'])
            audit.append({'case_id': case['case_id'], 'scenario': scenario, 'removed_parameters': sorted(removed),
                'changed_parameters': changed_parameters, 'actions': actions, 'visible_evidence': view,
                'hidden_reference_not_passed_to_fit': {r['parameter']: r['value'] for r in base['evidence'] if r['parameter'] in removed}})
            for method in METHODS:
                cid = signature(request, predict_request, training, method)
                units.append({'id': '__'.join((case['case_id'], scenario, method)), 'case_id': case['case_id'],
                    'scenario': scenario, 'sampling': 'full', 'method': method, 'computation_id': cid,
                    'scientific_inputs_changed': bool(changed_parameters)})
                if cid in cached: reused[cid] = cached[cid]
                elif cid not in specs:
                    specs[cid] = {'case_id': case['case_id'], 'sampling': 'full', 'method': method,
                        'request': inputs.store_input(run, request), 'target_request': inputs.store_input(run, predict_request),
                        'training_source': train_ref}
    for obj in (cases, specs, reused, audit): api.pin_refs(obj, pins)
    for name in ('scripts/run_robust_blueprint.py', 'scripts/run_blueprint_capability.py',
        'scripts/run_blueprint_evidence_policy.py', 'scripts/run_blueprint_learning.py',
        'scripts/run_blueprint_transport.py', 'scripts/report_blueprint_validation.py'):
        pins[name] = sha256(ROOT / name)
    if len(units) != 120 or len(cases) != 10: raise ValueError('Ten drug × four inputs × three methods required')
    protocol = {'version': VERSION, 'created_at': now(), 'cases': cases, 'methods': list(METHODS),
        'sampling': ['full'], 'scenarios': list(SCENARIOS), 'formulas': list(FORMULAS), 'units': units,
        'computations': specs, 'cached_ids': sorted(reused), 'contracts': contracts, 'input_audit': audit,
        'changed_case_ids': changed, 'source_pins': pins, 'paper_guard': publication,
        'source_protocol': ref(BASE / 'protocol.json'), 'workers': workers, 'native_threads': 2,
        'native_timeout_seconds': 180, 'monitor_seconds': 1200, 'directory_limit_bytes': 10 * 1024**3,
        'scope': 'Ten previously viewed development drugs; full development curves only; no new blind validation.',
        'stability': {'curve': 'geomean(max(P_missing/P_available,P_available/P_missing)) on fixed target sampling schedule, without clinical outcomes',
            'endpoints': 'matched NCA predicted endpoints compared between input scenarios; Tmax absolute hours, other positive endpoints symmetric fold change; undefined remain missing',
            'primary_denominator': 'only drugs whose scientific inputs actually change; equal-input controls separate',
            'descriptive_drift_threshold': 1.25, 'threshold_is_clinical_acceptance': False},
        'adoption': {'automatic': False, 'all_scenario_gmfe_nonincrease_required': True,
            'secondary_endpoint_regressions_reported': True, 'individual_drug_regressions_reported': True,
            'no_target_driven_per_drug_model_selection': True,
            'warning': 'A development nonregression result cannot guarantee future clinical errors will not increase.'},
        'bootstrap': {'seed': 20260930, 'resamples': 2000, 'cluster': 'clinical study; Pletz drugs together'},
        'limitations': ['Missing inputs are deliberate removals from existing evidence, not newly observed missingness.',
            'Six drugs already use RF fu and CL; deleting/refilling identical values supplies no additional robustness evidence.',
            'Clinical/scientific validity of retained renal inputs is not implied by preservation.',
            'No paper source, figure, table or PDF is replaced, regenerated or added.']}
    verify(pins); verify(publication)
    save(run / 'protocol.json', protocol)
    save(run / 'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': ref(run / 'protocol.json'),
        'created_at': now(), 'fits': {}, 'computations': reused, 'monitor': {}})
    return {'unique_new_fits': 2 * len(specs), 'new_predictions': len(specs), 'cached_predictions': len(reused),
        'logical_units': len(units), 'changed_drugs': {s: len(v) for s,v in changed.items()}, 'paper_guard_files': len(publication)}


def load(run, locked=False):
    protocol, state = read(run / 'protocol.json'), read(run / 'state.json')
    if checked(state['protocol']) != protocol or protocol['version'] != VERSION or state['version'] != VERSION:
        raise ValueError('Protocol changed')
    verify(protocol['source_pins']); verify(protocol['paper_guard'])
    if protocol['cases'] != checked(protocol['source_protocol'])['cases']:
        raise ValueError('Clinical cases changed')
    expected = {(c['case_id'], s, m) for c in protocol['cases'] for s in SCENARIOS for m in METHODS}
    actual = [(u['case_id'], u['scenario'], u['method']) for u in protocol['units']]
    if len(actual) != len(expected) or set(actual) != expected: raise ValueError('Comparison matrix changed')
    for cid, spec in protocol['computations'].items():
        req, target, train = (checked(spec[k]) for k in ('request', 'target_request', 'training_source'))
        if cid != signature(req, target, train, spec['method']): raise ValueError('Computation identity changed')
    fits = {key(cid, f) for cid in protocol['computations'] for f in FORMULAS}
    computations = set(protocol['computations']) | set(protocol['cached_ids'])
    if not set(state['fits']) <= fits or not set(protocol['cached_ids']) <= set(state['computations']) <= computations:
        raise ValueError('Unexpected checkpoint')
    api.pin_refs([state['fits'], state['computations']], {})
    if locked and (set(state['fits']) != fits or set(state['computations']) != computations or not state.get('all_attempts_locked_before_scoring')
            or not state.get('monitor_stopped_before_scoring') or state['status'] not in ('predictions_locked', 'complete')
            or any(v.get('status') not in ('predicted','failed') for v in state['computations'].values())
            or any(v.get('status') not in ('fitted','failed') for v in state['fits'].values())
            or not state.get('formula_selection')):
        raise ValueError('All attempts must lock before scoring')
    if locked:
        selection=checked(state['formula_selection'])
        if selection.get('basis')!='development_objective_only' or set(selection.get('choices',{}))!=set(protocol['computations']):
            raise ValueError('All development formula choices must lock before scoring')
    return protocol, state


def execute(run):
    with (run / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, state = load(run)
        if state['status'] == 'complete': return {'status': 'complete', 'reused': True}
        for method, contract in protocol['contracts'].items():
            if engine(method).VERSION != contract['version'] or engine(method).CONFIG != contract['config']:
                raise ValueError('Engine contract changed')
        controller = Controller(run, {**protocol, 'source_pins': {**protocol['source_pins'], **protocol['paper_guard']}}, state)
        controller.persist(status='fitting', started_at=state.get('started_at', now()), pause_reason=None)
        monitor = threading.Thread(target=controller.watch, daemon=True); monitor.start()
        def fit_one(cid, spec, formula):
            controller.check()
            if key(cid, formula) in state['fits']: checked(state['fits'][key(cid, formula)]['mapping']); return
            req = checked(spec['request']); req['partition_method'] = formula
            mapping = engine(spec['method']).fit_mapping(reference_request=req, training_source=spec['training_source'],
                output_dir=run / 'work' / cid / formula, project_root=ROOT,
                variant=variant(spec['method']), before_trial=controller.check)
            api.fit_costs(mapping, variant(spec['method']))
            with controller.lock:
                state['fits'][key(cid, formula)] = {'mapping': mapping['mapping'], 'status': mapping['status'],
                    'eligible': mapping['eligible'], 'native_calls': mapping['total_native_calls'],
                    'linear_solver_calls': mapping['linear_solver_calls']}
                controller.persist()
        def predict_one(cid, spec):
            controller.check()
            if cid in state['computations']: api.pin_refs(state['computations'][cid], {}); return
            choice = checked(state['formula_selection'])['choices'][cid]
            mappings = [checked(state['fits'][key(cid,f)]['mapping']) for f in FORMULAS]
            resources = {'total_native_calls': sum(m['total_native_calls'] for m in mappings),
                'linear_solver_calls': sum(m['linear_solver_calls'] for m in mappings),
                'wall_seconds': sum(m['elapsed_seconds'] for m in mappings), 'cache_reused': False,
                'fit_cost_count_exact': all(m.get('cost_count_exact',True) for m in mappings)}
            if choice is None: outcome = {'status': 'failed', 'failure': 'No eligible formula'}
            else:
                req = checked(spec['target_request']); req['partition_method'] = choice['formula']
                folder = run / 'work' / cid / 'target'
                if folder.exists() and not (folder/'completion.json').exists() and any(p.name != '.calibration.lock' for p in folder.iterdir()):
                    folder = folder.with_name('target_retry_1')
                    if folder.exists() and not (folder/'completion.json').exists(): raise Paused('Two partial prediction attempts retained')
                predicted = engine(spec['method']).predict_mapping(request=req, mapping_path=ROOT / choice['mapping']['path'],
                    output_dir=folder, project_root=ROOT, before_trial=controller.check)
                cost = read(folder/'result.json')
                resources['total_native_calls'] += cost['actual_new_calls']; resources['linear_solver_calls'] += cost['linear_solver_calls']
                resources['wall_seconds'] += cost['elapsed_seconds']; resources['prior_incomplete_costs_unknown'] = folder.name == 'target_retry_1'
                outcome = {'status': predicted['status'], 'request': ref(folder/'request.json'), 'mapping': choice['mapping'],
                    'selected_formula': choice['formula'], 'engine': spec['method']}
                if predicted['status'] == 'predicted':
                    outcome.update(prediction=ref(folder/'prediction_locked.json'), completion=ref(folder/'completion.json'),
                        dense_endpoints=ref(folder/'endpoints.json'))
                else: outcome['failure'] = predicted
            outcome['resources'] = resources
            with controller.lock: state['computations'][cid] = outcome; controller.persist()
        def phase(items, fn):
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                jobs = {pool.submit(fn,*args): args[0] for args in items}
                for future in as_completed(jobs):
                    try: future.result()
                    except Exception as exc:
                        with controller.lock:
                            if not controller.halted.is_set():
                                state.setdefault('interruptions',[]).append({'unit': jobs[future], 'error':str(exc), 'time':now()})
                                controller.pause(str(exc))
            controller.check(force=True)
        try:
            items = sorted(protocol['computations'].items())
            phase([(cid,spec,f) for cid,spec in items for f in FORMULAS], fit_one)
            choices = {}
            for cid,spec in items:
                best = api.choose_formula([checked(state['fits'][key(cid,f)]['mapping']) for f in FORMULAS])
                choices[cid] = None if best is None else {'formula': best['construction_partition_method'],
                    'mapping': state['fits'][key(cid,best['construction_partition_method'])]['mapping']}
            document = {'version': VERSION, 'basis':'development_objective_only','choices':choices}
            path = run/'formulas_locked.json'
            if path.exists() and read(path) != document: raise ValueError('Locked formula choice changed')
            save(path,document)
            controller.persist(status='predicting', formula_selection=ref(path), formulas_locked_before_target_prediction=True)
            phase(items,predict_one)
            controller.closed.set(); monitor.join(timeout=30)
            if monitor.is_alive(): raise Paused('Monitor did not stop')
            controller.persist(status='predictions_locked',all_attempts_locked_before_scoring=True,
                monitor_stopped_before_scoring=True,locked_at=now())
            result = report(run)
            controller.persist(status='complete',completed_at=now(),results=ref(run/'results.json'))
            return {'status':'complete','units':len(result['units'])}
        except Exception as exc:
            if not controller.halted.is_set(): controller.pause(str(exc))
            raise
        finally: controller.closed.set(); monitor.join(timeout=30)


def report(run):
    import numpy as np
    from report_blueprint_validation import Store, _case_reference, _score_prediction, positive, clustered_geomean, paired_ratio
    protocol,state = load(run,locked=True)
    if state['status']=='complete': result=checked(state['results']); verify(result['source_pins']); return result
    store=Store(ROOT)
    for path,digest in protocol['source_pins'].items(): store.checksum(path,digest)
    store.ref(state['protocol']); generated={}; api.pin_refs([state['fits'],state['computations'],state['formula_selection']],generated)
    for path,digest in generated.items(): store.checksum(path,digest)
    # No target outcome has been decoded in preparation, fitting or prediction.
    references={c['case_id']:_case_reference(c,store) for c in protocol['cases']}
    cases={c['case_id']:c for c in protocol['cases']}; records=[]; cache={}
    for unit in protocol['units']:
        computation=state['computations'][unit['computation_id']]; case=cases[unit['case_id']]
        row={**unit,'drug':case['drug'],'cluster_id':case['cluster_id'],'study_id':case['study_id'],
            'attempt_status':computation['status'],'prediction_source':computation.get('prediction'),'mapping_source':computation.get('mapping')}
        if unit['computation_id'] not in cache:
            try:
                if computation['status']!='predicted': raise ValueError(str(computation.get('failure')))
                clinical,curve,endpoints=references[unit['case_id']]
                scored,_,_=_score_prediction(case,clinical,curve,endpoints,computation,store); scored.pop('curve',None)
                scored['scoring_status']='scored' if positive(scored['curve_metrics'].get('gmfe')) else 'partial'
            except (ValueError,KeyError,TypeError,OverflowError) as exc: scored={'scoring_status':'failed','failure':str(exc)}
            cache[unit['computation_id']]=scored
        row.update(deepcopy(cache[unit['computation_id']])); records.append(row)
    summary=[]; clusters={cid:c['cluster_id'] for cid,c in cases.items()}
    for scenario in SCENARIOS:
        for method in METHODS:
            rows=[r for r in records if r['scenario']==scenario and r['method']==method]
            scores={r['case_id']:r.get('curve_metrics',{}).get('gmfe') for r in rows if positive(r.get('curve_metrics',{}).get('gmfe'))}
            ep={}
            for name in sorted({e for r in rows for e in r.get('endpoints',{})}):
                vals=[r['endpoints'][name]['error'] for r in rows if r.get('endpoints',{}).get(name,{}).get('error_metric')=='fold_error' and positive(r['endpoints'][name].get('error'))]
                if vals: ep[name]={'n':len(vals),'error_metric':'fold_error','geomean_fold_error':math.exp(float(np.mean(np.log(vals)))), 'within_twofold':sum(v<=2 for v in vals)}
                if name=='tmax':
                    absolute=[r['endpoints'][name]['error'] for r in rows if r.get('endpoints',{}).get(name,{}).get('error_metric')=='absolute_error_h' and isinstance(r['endpoints'][name].get('error'),(int,float)) and math.isfinite(r['endpoints'][name]['error'])]
                    ep[name]={'n':len(absolute),'error_metric':'absolute_error_h','mean_absolute_error_h':float(np.mean(absolute)) if absolute else None}
            summary.append({'scenario':scenario,'method':method,'scored':len(scores),'failed':10-len(scores),
                'curve_gmfe':clustered_geomean(scores,{c:clusters[c] for c in scores},seed=20260930),
                'within_twofold':sum(v<=2 for v in scores.values()),'endpoint_errors':ep})
    ratios={}
    for scenario in SCENARIOS:
        def scores(method): return {r['case_id']:r.get('curve_metrics',{}).get('gmfe') for r in records if r['scenario']==scenario and r['method']==method}
        ratios[scenario]={m:paired_ratio(scores(m),scores('current'),clusters,expected_ids=list(cases),seed=20260930) for m in METHODS[1:]}
    stability=[]
    for row in records:
        if row['scenario']=='available': continue
        full=next(r for r in records if r['case_id']==row['case_id'] and r['scenario']=='available' and r['method']==row['method'])
        item={k:row[k] for k in ('case_id','drug','cluster_id','method','scenario','scientific_inputs_changed')}
        item['status']='failed'; item['endpoint_prediction_drift']={}
        if row['attempt_status']==full['attempt_status']=='predicted':
            a,b=(store.ref(r['prediction_source']) for r in (row,full))
            times=np.asarray(cases[row['case_id']]['nca_policy']['sampling_time_h'])*60
            valid=(times>0)&(times>=max(a['time_min'][0],b['time_min'][0]))&(times<=min(a['time_min'][-1],b['time_min'][-1]))
            pp=np.interp(times[valid],a['time_min'],a['concentration']); qq=np.interp(times[valid],b['time_min'],b['concentration'])
            ok=(pp>0)&(qq>0)&np.isfinite(pp)&np.isfinite(qq)
            if ok.all() and len(pp):
                d=np.maximum(pp/qq,qq/pp); item.update(status='compared',points=len(d),curve_prediction_drift=math.exp(float(np.mean(np.log(d)))),max_point_drift=float(max(d)))
            for endpoint,v in row.get('endpoints',{}).items():
                w=full.get('endpoints',{}).get(endpoint,{})
                # Clinical-NCA predictions only; observed outcome does not enter drift.
                p=v.get('value'); q=w.get('value')
                if endpoint=='tmax' and isinstance(p,(int,float)) and isinstance(q,(int,float)) and math.isfinite(p) and math.isfinite(q):
                    if v.get('unit') != w.get('unit') or v.get('unit') not in ('h','min'):
                        raise ValueError('Tmax drift requires matching hour/minute units')
                    item['tmax_prediction_shift_h']=abs(p-q)/(60 if v['unit']=='min' else 1)
                elif endpoint!='tmax' and positive(p) and positive(q): item['endpoint_prediction_drift'][endpoint]=max(p/q,q/p)
        stability.append(item)
    adoption=candidate_checks(records,summary)
    result={'version':VERSION,'protocol':ref(run/'protocol.json'),'scope':protocol['scope'],'units':records,'summary':summary,
        'candidate_to_current_curve_ratio':ratios,'stability':stability,'adoption_checks':adoption,
        'changed_case_ids':protocol['changed_case_ids'],'unique_scored_predictions':len(cache),'source_pins':dict(store.pins),
        'paper_guard_verified':True,'paper_updated':False,'limitations':protocol['limitations']}
    store.unchanged(); verify(protocol['paper_guard']); save(run/'results.json',result)
    table=[{'scenario':r['scenario'],'method':r['method'],'scored':r['scored'],'failed':r['failed'],
        'curve_gmfe':r['curve_gmfe']['estimate'],'ci95_low':(r['curve_gmfe']['ci95'] or [None,None])[0],
        'ci95_high':(r['curve_gmfe']['ci95'] or [None,None])[1],'within_twofold':r['within_twofold']} for r in summary]
    with (run/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return result


def status(run):
    import psutil
    p,s=read(run/'protocol.json'),read(run/'state.json')
    result={'status':s['status'],'fits_completed':len(s['fits']),'fits_total':2*len(p['computations']),
        'predictions_completed':sum(cid in s['computations'] for cid in p['computations']),'predictions_total':len(p['computations']),
        'reused_predictions':len(p['cached_ids']),'paper_guard_unchanged':all(sha256(ROOT/k)==v for k,v in p['paper_guard'].items()),
        'monitor':s.get('monitor'),'pause_reason':s.get('pause_reason')}
    if (run/'pid.json').exists():
        pid=read(run/'pid.json'); result.update(pid=pid['pid'],alive=False)
        try:
            proc=psutil.Process(pid['pid']); result['alive']=proc.uids().real==os.getuid() and abs(proc.create_time()-pid['create_time'])<.01 and proc.status()!='zombie'
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return result


def background(run):
    import psutil
    if status(run).get('alive'): raise ValueError('Project batch already active')
    with (run/'run.log').open('a') as log:
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--run-dir',str(run)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    save(run/'pid.json',{'pid':child.pid,'uid':os.getuid(),'create_time':psutil.Process(child.pid).create_time(),'script':str(Path(__file__).relative_to(ROOT))})
    return {'pid':child.pid,'log':str((run/'run.log').relative_to(ROOT))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','status','report','pause','resume','cleanup'))
    p.add_argument('--run-dir',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=14);p.add_argument('--background',action='store_true')
    a=p.parse_args();run=own_run(a.run_dir)
    if a.command=='prepare': out=prepare(run,a.workers)
    elif a.command=='status': out=status(run)
    elif a.command=='run':out=background(run) if a.background else execute(run)
    elif a.command=='pause':save(run/'PAUSE',{'time':now()});out={'status':'pause_requested'}
    elif a.command=='resume':
        if status(run).get('alive'):raise ValueError('Project batch already active')
        (run/'PAUSE').unlink(missing_ok=True);out=background(run) if a.background else execute(run)
    elif a.command=='cleanup':
        _,state=load(run);out={'removed_count':0,'fits_checked':len(state['fits']),'reason':'No per-evaluation artifacts written; source-bound native artifacts remain required.'}
    else:
        with (run/'.run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out={'units':len(report(run)['units']),'status':'reported'}
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
