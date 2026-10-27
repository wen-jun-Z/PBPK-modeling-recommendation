#!/usr/bin/env python3
"""Frozen ten-drug evidence-policy development experiment; no target-guided fitting."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprint_policy.evidence_decision import decide
from run_blueprint_learning import Controller, Paused, now, read, ref, checked, verify, own_run

VERSION = 'blueprint-evidence-policy-1'
DEFAULT = ROOT / 'experiments/blueprint_design/evidence_policy_20260929'
BASE = ROOT / 'experiments/blueprint_design/validation_20260929'
POLICIES = ('evidence_policy', 'fixed_template', 'rf_fill', 'no_conflict_guard')
CLASSICAL = ('two_compartment', 'two_compartment_prior')
METHODS = POLICIES + CLASSICAL
SCENARIOS = ('available', 'missing_fu', 'conflicting_fu', 'wrong_species_fu',
             'missing_required_pbpk_formula', 'unsupported_ddi')
SAMPLING = ('full', 'five')
FORMULAS = ('PKSimStandard', 'RodgersRowland')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def store_input(run, value):
    """Content-addressed immutable inputs; filenames do not reveal perturbation truth."""
    path = run / 'inputs' / (digest(value) + '.json')
    if path.exists():
        if read(path) != value: raise ValueError('Immutable input mismatch')
    else: save(path, value)
    return {**ref(path), 'location': 'Frozen visible input'}


def clean_training(document):
    keys = ('schema_version', 'compound_id', 'compound_name', 'study_id', 'role',
            'matrix', 'time_unit', 'unit', 'molecular_weight_g_mol', 'human', 'dosing')
    result = {k: deepcopy(document[k]) for k in keys if k in document}
    result['observations'] = [{k: row[k] for k in ('time_min', 'concentration_umol_per_l')}
                              for row in document['observations']]
    return result


def visible_record(record):
    keys = ('parameter', 'compound_id', 'value', 'unit', 'basis', 'context', 'prediction_method')
    return {k: deepcopy(record[k]) for k in keys if k in record}


def scenario_evidence(request, rf_request, scenario, case_id):
    """Dataset generator, not a decision algorithm. Truth is stored outside visible inputs."""
    rows = deepcopy(request['evidence'])
    fallback = [deepcopy(e) for e in rf_request['evidence'] if e['parameter'] == 'fraction_unbound']
    original = next(e for e in rows if e['parameter'] == 'fraction_unbound')
    audit = {'primary_sources': deepcopy(rows), 'fallback_sources': deepcopy(fallback),
             'perturbation': scenario, 'synthetic': scenario not in ('available', 'missing_fu')}
    if scenario == 'missing_fu': rows = [e for e in rows if e['parameter'] != 'fraction_unbound']
    elif scenario == 'conflicting_fu':
        changed = deepcopy(original)
        changed['value'] = float(original['value']) * .5
        index = rows.index(original)
        rows.insert(index if int(digest(case_id)[:8], 16) % 2 else index + 1, changed)
    elif scenario == 'wrong_species_fu':
        original.setdefault('context', {})['species'] = 'rat'
    elif scenario == 'missing_required_pbpk_formula':
        rows = [e for e in rows if e['parameter'] != 'molecular_formula']
    elif scenario not in ('available', 'unsupported_ddi'): raise ValueError('Unknown scenario')
    return [visible_record(e) for e in rows], [visible_record(e) for e in fallback], audit


def bind_view(run, rows, fallback):
    view = store_input(run, {'records': rows + fallback})
    bound = []
    for index, row in enumerate(rows + fallback):
        bound.append({**deepcopy(row), 'source': {**view, 'location': f'Record {index + 1}'}})
    return bound[:len(rows)], bound[len(rows):], view


def selected_request(run, template, evidence):
    result = {k: deepcopy(template[k]) for k in ('entry_reason', 'compound_id', 'compound_name',
               'human', 'dosing', 'target', 'partition_method')}
    result['evidence'] = deepcopy(evidence)
    result['evidence_ledger'] = store_input(run, {'evidence': evidence})
    result['assumptions'] = ['Frozen adult single-IV plasma development experiment; population defaults are not patient observations.',
        'Solubility placeholders apply only to dissolved IV administration. Selected RF estimates are not clinical observations.']
    result['forbidden_sources'] = deepcopy(template.get('forbidden_sources', []))
    return result


def computation_signature(request, training, target, allowed):
    """Exact computational identity, excluding provenance only; each alias retains its own audit."""
    values = sorted([{k: deepcopy(e[k]) for k in ('parameter', 'compound_id', 'value', 'unit', 'basis', 'context', 'prediction_method') if k in e}
                     for e in request['evidence']], key=lambda e: e['parameter'])
    return {'compound_id': request['compound_id'], 'evidence': values,
        'human': request['human'], 'dosing': request['dosing'], 'target': request['target'],
        'target_human': target['human'], 'target_dosing': target['dosing'], 'target_endpoint_request': target['target'],
        'training_source': training, 'allowed_parameters': list(allowed),
        'screen_parameters': True, 'max_evaluations_per_formula': 24, 'prior_weight': .1,
        'formulas': list(FORMULAS)}


def prepare(run, workers=14):
    run = own_run(run)
    if run.exists(): raise FileExistsError('Keep existing batch; resume it or choose a new run directory')
    if not 1 <= workers <= 14: raise ValueError('Use 1 to 14 workers')
    old = read(BASE / 'protocol.json'); verify(old['source_pins'])
    run.mkdir(); (run / 'inputs').mkdir()
    decisions, units, computations, audit = [], [], {}, []
    for case in old['cases']:
        base = checked(case['requests']['evidence']['development'])
        target = checked(case['requests']['evidence']['target'])
        rf = checked(case['requests']['rf']['development'])
        training = {s: {**store_input(run, clean_training(checked(case['sampling_sources'][s]))),
                        'role': 'calibration_data_reconstruction'} for s in SAMPLING}
        if len(checked(training['five'])['observations']) != 5: raise ValueError('Five-point isolation failed')
        # Classical receives only MW and the permitted CL prior, no full physicochemical record.
        mw = next(e['value'] for e in base['evidence'] if e['parameter'] == 'molecular_weight')
        classical_case = {k: deepcopy(case[k]) for k in ('case_id', 'compound_id', 'study_id', 'target_request_metadata', 'nca_policy')}
        classical_case['physchem'] = store_input(run, {'identity': {'molecular_weight_g_mol': mw}})
        for scenario in SCENARIOS:
            rows, fallback, provenance = scenario_evidence(base, rf, scenario, case['case_id'])
            rows, fallback, view = bind_view(run, rows, fallback)
            visible = selected_request(run, base, rows)
            purpose = {'endpoints': base['target']['endpoints'], 'population_condition': 'reference_adult',
                       'mechanism': 'ddi' if scenario == 'unsupported_ddi' else 'parent_plasma'}
            audit.append({'case_id': case['case_id'], 'scenario': scenario, 'view': view, **provenance})
            for method in METHODS:
                identity = '__'.join((case['case_id'], scenario, method))
                is_native = method in POLICIES
                expected = scenario not in (('conflicting_fu', 'missing_required_pbpk_formula', 'unsupported_ddi') if is_native else ('unsupported_ddi',))
                if is_native:
                    decision = decide(visible, fallback, purpose=purpose, method=method)
                else:
                    # These adapters do not model DDI; molecular formula/fu are not required by two-compartment PK.
                    decision = {'status': 'blocked' if purpose['mechanism'] != 'parent_plasma' else 'ready',
                        'reasons': ['unsupported_ddi'] if purpose['mechanism'] != 'parent_plasma' else [],
                        'actions': [], 'selected_evidence': [e for e in rows if e['parameter'] in
                            ('total_clearance', 'hepatic_clearance', 'renal_clearance', 'elimination_scope')]}
                item = {'id': identity, 'case_id': case['case_id'], 'scenario': scenario, 'method': method,
                    'status': decision['status'], 'expected_ready': expected, 'decision': decision, 'visible_evidence': view}
                decisions.append(item)
                for sampling in SAMPLING:
                    unit = {k: item[k] for k in ('case_id', 'scenario', 'method')}
                    unit.update(id='__'.join((case['case_id'], scenario, sampling, method)), sampling=sampling,
                                status='planned' if decision['status'] == 'ready' else 'blocked', decision_id=identity)
                    if decision['status'] == 'ready':
                        selected = selected_request(run, base, decision['selected_evidence'])
                        predicted = deepcopy(selected)
                        for field in ('human', 'dosing', 'target'): predicted[field] = deepcopy(target[field])
                        if is_native:
                            allowed = decision['calibration']['allowed_parameters']
                            signature = computation_signature(selected, training[sampling], predicted, allowed)
                        else:
                            allowed = []
                            signature = {'kind': 'classical', 'case': classical_case, 'training': training[sampling],
                                'with_cl_prior': method == 'two_compartment_prior',
                                'clearance_evidence': [{k:v for k,v in e.items() if k != 'source'} for e in selected['evidence']],
                                'human': selected['human'], 'dosing': selected['dosing'], 'target': selected['target']}
                        compid = ('native_' if is_native else 'classical_') + digest(signature)[:24]
                        unit['policy_request'] = store_input(run, selected)
                        unit['computation_id'] = compid
                        unit['computation_signature_sha256'] = digest(signature)
                        if compid not in computations:
                            computations[compid] = {'kind': 'native' if is_native else 'classical',
                                'case_id': case['case_id'], 'sampling': sampling, 'method': method,
                                'request': unit['policy_request'], 'target_request': store_input(run, predicted),
                                'training_source': training[sampling], 'allowed_parameters': allowed,
                                'signature': signature, 'representative_unit': unit['id']}
                            if not is_native: computations[compid]['case'] = classical_case
                    units.append(unit)
    pins = dict(old['source_pins'])
    code = [Path(__file__), ROOT/'scripts/report_blueprint_evidence_policy.py', ROOT/'scripts/run_blueprint_learning.py',
        ROOT/'scripts/report_blueprint_validation.py', ROOT/'scripts/blueprint_validation_classical.py',
        ROOT/'scripts/run_classical_pk_comparator.py', *sorted((ROOT/'src/pbpk_kb/blueprint_policy').glob('*.py')),
        *sorted((ROOT/'src/pbpk_kb/blueprint_learning').glob('*.py'))]
    for path in code + [BASE/'protocol.json', *sorted((run/'inputs').glob('*.json'))]:
        pins[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {'version': VERSION, 'created_at': now(), 'source_pins': pins, 'cases': old['cases'],
        'methods': list(METHODS), 'scenarios': list(SCENARIOS), 'sampling': list(SAMPLING),
        'decisions': decisions, 'units': units, 'computations': computations, 'audit_only_not_passed_to_methods': audit,
        'workers': workers, 'native_threads': 2, 'native_timeout_seconds': 180, 'monitor_seconds': 1200,
        'directory_limit_bytes': 10*1024**3,
        'constraints': {'max_evaluations_per_formula': 24, 'formulas': list(FORMULAS), 'screen_parameters': True,
            'screening_shared_across_native_policies': True, 'rf_fallback_parameters': ['fraction_unbound'],
            'classical_starts': 16, 'classical_max_nfev_per_start': 400,
            'two_compartment_parameters_fitted_not_supplied': ['CL','Vc','Q','Vp'],
            'target_outcomes_read_only_after_all_attempts_locked': True,
            'aliases_are_not_independent_experiments': True},
        'interpretation': 'Ten previously viewed development drugs; engineered faults are not clinical observations. No zero-shot claim.',
        'retention': 'Preserve scores, failed attempts, candidate summaries and selected artifacts; remove only nonselected successful native execution directories.'}
    save(run/'protocol.json', protocol)
    save(run/'state.json', {'version': VERSION, 'status': 'prepared', 'created_at': now(),
        'protocol': ref(run/'protocol.json'), 'computations': {}, 'fits': {}, 'monitor': {},
        'engineering_counts': dict(Counter((d['method'] + ':' + d['status']) for d in decisions))})
    return {'decisions': len(decisions), 'logical_units': len(units), 'unique_computations': len(computations),
        'unique_native_fits': 2*sum(c['kind']=='native' for c in computations.values())}


def execute(run):
    from pbpk_kb.blueprint_policy.native_calibration import fit_mapping, predict_mapping, cleanup_unselected
    from pbpk_kb.blueprints.evidence_preserving_calibration import _infrastructure_problem
    from blueprint_validation_classical import fit_predict
    from report_blueprint_evidence_policy import report
    run = own_run(run)
    with (run/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, state = read(run/'protocol.json'), read(run/'state.json')
        if protocol['version'] != VERSION or state['version'] != VERSION: raise ValueError('Wrong batch version')
        if state['status'] == 'complete': return
        controller = Controller(run, protocol, state)
        controller.check(force=True)
        monitor = threading.Thread(target=controller.watch, daemon=True); monitor.start()
        controller.persist(status='running', started_at=state.get('started_at', now()))
        def persist_result(identity, result):
            with controller.lock:
                state['computations'][identity] = result
                controller.persist()
        def one(identity, config):
            controller.check()
            if identity in state['computations']:
                for field in ('prediction','request','fit','mapping'):
                    if state['computations'][identity].get(field): checked(state['computations'][identity][field])
                return
            folder = run/'work'/identity
            request = checked(config['request'])
            started = time.monotonic()
            if config['kind'] == 'classical':
                try:
                    result = fit_predict(case=config['case'], request=request, training_source=config['training_source'],
                        output_dir=folder, project_root=ROOT, with_cl_prior=config['method']=='two_compartment_prior')
                except RuntimeError as exc:
                    if str(exc) != 'No finite subset-only classical fit; all attempts retained': raise
                    result = {'status': 'failed', 'failure': str(exc), 'fit': ref(folder/'fit.json'),
                        'resources': {'wall_seconds': time.monotonic()-started}}
            else:
                fits = []
                for formula in FORMULAS:
                    controller.check()
                    fitid = identity+'__'+formula
                    current_request = {**deepcopy(request), 'partition_method': formula}
                    fitfolder = folder/formula
                    mapping = fit_mapping(reference_request=current_request, training_source=config['training_source'],
                        output_dir=fitfolder, project_root=ROOT, max_evaluations=24,
                        allowed_parameters=config['allowed_parameters'], screen_parameters=True,
                        before_trial=controller.check, prior_weight=.1)
                    fits.append(mapping)
                    with controller.lock:
                        state['fits'][fitid] = {'mapping': mapping['mapping'], 'eligible': mapping['eligible'],
                            'status': mapping['status'], 'native_calls': mapping['total_native_calls'],
                            'screening': mapping.get('screening')}
                        controller.persist()
                    cleaned = cleanup_unselected(output_dir=fitfolder, project_root=ROOT)
                    with controller.lock:
                        state['fits'][fitid]['cleanup'] = cleaned; controller.persist()
                valid = [f for f in fits if f['eligible']]
                if not valid:
                    result = {'status': 'failed', 'failure': 'No eligible native fit',
                        'resources': {'total_native_calls': sum(f['total_native_calls'] for f in fits),
                                      'actual_new_native_calls': sum(f['total_native_calls'] for f in fits),
                                      'wall_seconds': sum(f.get('elapsed_seconds',0) for f in fits)}}
                else:
                    chosen = min(valid, key=lambda f: (f['decision']['objective'], f['construction_partition_method']))
                    predicted = checked(config['target_request']); predicted['partition_method'] = chosen['construction_partition_method']
                    controller.check()
                    out = folder/'target'
                    # A resumed partial target is retained. One explicit retry is allowed;
                    # a second incomplete attempt pauses rather than writing indefinitely.
                    if out.exists() and not (out/'completion.json').exists() and any(p.name!='.calibration.lock' for p in out.iterdir()):
                        out = folder/'target_retry_1'
                        if out.exists() and not (out/'completion.json').exists() and any(p.name!='.calibration.lock' for p in out.iterdir()):
                            raise Paused('Two incomplete target executions retained; inspect infrastructure before further retries')
                    native = predict_mapping(request=predicted, mapping_path=ROOT/chosen['mapping']['path'], output_dir=out, project_root=ROOT)
                    if _infrastructure_problem(native): raise Paused('Target execution infrastructure error')
                    result = {'status': 'predicted' if native['status']=='predicted' else 'failed',
                        'native_status':native['status'], 'request': ref(out/'request.json'), 'mapping': chosen['mapping'],
                        'resources': {'total_native_calls': 1+sum(f['total_native_calls'] for f in fits)+(out.name=='target_retry_1'),
                            'actual_new_native_calls': 1+sum(f['total_native_calls'] for f in fits)+(out.name=='target_retry_1'),
                            'wall_seconds': native.get('elapsed_seconds',0)+sum(f.get('elapsed_seconds',0) for f in fits),
                            'resource_scope': 'Cumulative unique computation; aliases add no work. Interrupted target timing may be incomplete.',
                            'target_retry_used':out.name=='target_retry_1',
                            'failed_native_trials': sum(sum(r['status']!='ok' for r in f['evaluations']) for f in fits)}}
                    if native['status']=='predicted': result['prediction'] = ref(out/'prediction_locked.json')
                    else: result['failure'] = native
            persist_result(identity, result)
        try:
            # Same worker pool bounds all native children. Classical analytic fits do not spawn native tasks.
            jobs = sorted(protocol['computations'].items(), key=lambda kv:(kv[1]['kind']!='native',kv[1]['case_id'],kv[1]['sampling'],kv[0]))
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                futures = {pool.submit(one, identity, config): identity for identity,config in jobs}
                for future in as_completed(futures):
                    try: future.result()
                    except Exception as exc:
                        with controller.lock:
                            if not isinstance(exc, Paused) or not controller.halted.is_set():
                                state.setdefault('interruptions', []).append({'computation_id': futures[future], 'time':now(),
                                    'error': type(exc).__name__+': '+str(exc)})
                            if not controller.halted.is_set(): controller.pause(type(exc).__name__+': '+str(exc))
            controller.check(force=True)
            if len(state['computations']) != len(protocol['computations']): raise ValueError('Incomplete prediction matrix')
            controller.persist(status='predictions_locked', all_attempts_locked_before_scoring=True, locked_at=now())
            report(run, project_root=ROOT)
            controller.persist(status='complete', completed_at=now(), results=ref(run/'results.json'))
        except Exception as exc:
            if not controller.halted.is_set(): controller.pause(type(exc).__name__+': '+str(exc))
            raise
        finally:
            controller.closed.set(); monitor.join(timeout=2)


def status(run):
    run=own_run(run); state=read(run/'state.json'); protocol=read(run/'protocol.json')
    progress=[]
    for path in (run/'work').glob('*/*/progress.json'):
        try: progress.append(read(path))
        except (ValueError,FileNotFoundError): pass
    result={'status': state['status'], 'engineering_counts':state.get('engineering_counts'),
        'decisions':len(protocol['decisions']), 'logical_units':len(protocol['units']),
        'computations_completed':len(state['computations']), 'computations_total':len(protocol['computations']),
        'fits_completed':len(state['fits']), 'fits_total':2*sum(c['kind']=='native' for c in protocol['computations'].values()),
        'native_calls_recorded':sum(p.get('total_native_calls',0) for p in progress),
        'monitor':state.get('monitor'), 'pause_reason':state.get('pause_reason')}
    if (run/'pid.json').exists():
        import psutil
        pid=read(run/'pid.json'); result['pid']=pid['pid']; result['alive']=False
        try:
            proc=psutil.Process(pid['pid'])
            result['alive']=proc.uids().real==os.getuid() and abs(proc.create_time()-pid['create_time'])<.01 and proc.status()!='zombie'
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return result


def background(run):
    import psutil
    run=own_run(run)
    if (run/'pid.json').exists() and status(run).get('alive'): raise ValueError('This batch is already running')
    with (run/'run.log').open('a') as log:
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--run-dir',str(run)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    save(run/'pid.json',{'pid':proc.pid,'uid':os.getuid(),'create_time':psutil.Process(proc.pid).create_time(),
        'started_at':now(),'script':str(Path(__file__).relative_to(ROOT))})
    return {'pid':proc.pid,'log':str((run/'run.log').relative_to(ROOT))}


def cleanup(run):
    from pbpk_kb.blueprint_policy.native_calibration import cleanup_unselected
    run=own_run(run)
    if status(run).get('alive'): raise ValueError('Let the runner perform automatic fit cleanup while active')
    output=[]
    for path in sorted((run/'work').glob('*/*/mapping_locked.json')):
        output.append(cleanup_unselected(output_dir=path.parent,project_root=ROOT))
    return {'fits_checked':len(output),'removed_count':sum(x['removed_count'] for x in output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','report','pause','resume','cleanup'])
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args(); run=own_run(args.run_dir)
    if args.command=='prepare': result=prepare(run,args.workers)
    elif args.command=='status': result=status(run)
    elif args.command=='pause':
        save(run/'PAUSE',{'requested_at':now(),'reason':'User requested no new task submissions'})
        result={'status':'pause_requested'}
    elif args.command=='resume':
        if status(run).get('alive'): raise ValueError('Wait for the paused runner to finish its current calls')
        (run/'PAUSE').unlink(missing_ok=True)
        state=read(run/'state.json'); state.pop('pause_reason',None); state['status']='prepared'; save(run/'state.json',state)
        result=background(run) if args.background else execute(run)
    elif args.command=='run': result=background(run) if args.background else execute(run)
    elif args.command=='cleanup': result=cleanup(run)
    else:
        from report_blueprint_evidence_policy import report
        result=report(run,project_root=ROOT)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
