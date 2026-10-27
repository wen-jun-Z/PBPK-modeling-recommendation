#!/usr/bin/env python3
"""Shared clinical cohort: learned parameters -> executed PBPK -> locked scoring.

Published construction ideas are adapted to a common input contract; this is
not an exact reproduction of the original authors' closed QSAR predictors.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

VERSION = 'shared-clinical-pbpk-1'
DEFAULT = ROOT / 'experiments/blueprint_design/clinical_pbpk_comparison_20261004'
SOURCE = ROOT / 'experiments/blueprint_design/cross_drug_20260930/protocol.json'
METHODS = ('rf_standard', 'ml_standard', 'ml_five_partition', 'ours')
LABELS = {'rf_standard': 'RF + fixed PK-Sim', 'ml_standard': 'Shared ML + fixed PK-Sim',
          'ml_five_partition': 'Shared ML + five-partition ensemble', 'ours': 'Our PBPK blueprint'}
FORMULAS = ('PKSimStandard', 'RodgersRowland', 'Berezhkovskiy', 'PoulinTheil', 'Schmitt')
CODE = ('scripts/run_clinical_pbpk_comparison.py',
        'src/pbpk_kb/blueprint_learning/clinical_shared_prior.py',
        'src/pbpk_kb/blueprint_policy/clinical_pbpk_comparison.py',
        'src/pbpk_kb/blueprint_learning/rich_parameter_prior.py',
        'src/pbpk_kb/blueprint_policy/learned_exchange_design.py',
        'src/pbpk_kb/blueprint_policy/capacity_guided_design.py',
        'src/pbpk_kb/blueprint_policy/permeability_model.py',
        'src/pbpk_kb/blueprints/extended_prediction.py',
        'src/pbpk_kb/blueprints/evidence_blueprint.py',
        'src/pbpk_kb/blueprints/design_runtime.py',
        'src/pbpk_kb/blueprints/evidence_native_audit.py',
        'src/pbpk_kb/blueprints/native_edits.py',
        'src/pbpk_kb/blueprints/source_adaptation_audit.py',
        'src/pbpk_kb/blueprints/source_workflow.py',
        'src/pbpk_kb/blueprints/endpoint_prediction.py',
        'src/pbpk_kb/blueprints/terminal_pk.py',
        'src/pbpk_kb/blueprints/pk_parameter_learning.py',
        'src/pbpk_kb/blueprint_policy/axes_calibration.py',
        'src/pbpk_kb/blueprint_policy/perfusion_export.py',
        'src/pbpk_kb/blueprints/clinical_prediction_nca.py',
        'scripts/score_cross_drug_blueprint.py', 'scripts/score_expanded_tail.py',
        'scripts/report_blueprint_validation.py', 'scripts/render_pbpk_method_comparison.py',
        'scripts/render_current_clinical_table.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = inside(path, ROOT, exists=True)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def checked(reference):
    path = inside(ROOT / reference['path'], ROOT, exists=True)
    if sha256(path) != reference['sha256']:
        raise ValueError('Source hash changed: ' + reference['path'])
    return read(path)


def own(path):
    path = inside(path, ROOT)
    if path.parent != DEFAULT.parent or not path.name.startswith('clinical_pbpk_comparison_'):
        raise ValueError('Use a project-owned clinical_pbpk_comparison_ directory')
    return path


def paper_pins():
    return {str(p.relative_to(ROOT)): sha256(p) for p in
            (ROOT/'results/manuscript/main.tex', ROOT/'results/manuscript/pbpk_paper_zh.pdf')}


def verify(protocol, code_patches=None):
    for path, expected in {**protocol['source_pins'], **protocol['code_pins'], **protocol['paper_guard']}.items():
        actual = sha256(inside(ROOT/path, ROOT, exists=True))
        if actual != expected:
            patch = (code_patches or {}).get(path, {})
            if (path not in protocol['code_pins'] or patch.get('old_sha256') != expected or
                    patch.get('new_sha256') != actual or not patch.get('reason')):
                raise ValueError('Frozen source changed: ' + path)


def collect_refs(value, pins):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            path = inside(ROOT/value['path'], ROOT, exists=True)
            if sha256(path) != value['sha256']:
                raise ValueError('Input receipt differs: ' + value['path'])
            pins[value['path']] = value['sha256']
        for child in value.values():
            collect_refs(child, pins)
    elif isinstance(value, list):
        for child in value:
            collect_refs(child, pins)


def prepare(run=DEFAULT, workers=14):
    run = own(run)
    if (run/'protocol.json').exists():
        raise FileExistsError('Preserve the existing protocol; use run to resume')
    if not 1 <= workers <= 14:
        raise ValueError('Workers must be 1..14')
    source = read(SOURCE)
    cases, pins = [], {str(SOURCE.relative_to(ROOT)): sha256(SOURCE)}
    for original in source['cases']:
        case = {key: deepcopy(original[key]) for key in ('case_id', 'compound_id', 'drug', 'study_id',
                'cluster_id', 'cohort', 'reference_kind', 'reference_source', 'physchem', 'nca_policy')}
        case['dataset'] = 'reported_clinical_10' if case['cohort'] == 'original' else 'osp_observed_10'
        case['source_sensitivity_flags'] = original.get('sensitivity_flags', [])
        # This source contains requests, not development/target concentrations.
        request = checked(original['cross_drug_requests'])['target']
        case['request_source'] = original['cross_drug_requests']
        chemical = read(ROOT/case['physchem'])
        identity = chemical['identity']
        case['smiles'] = identity.get('isomeric_smiles') or identity.get('canonical_connectivity_smiles')
        if not case['smiles']:
            raise ValueError('Missing molecular identity: ' + case['drug'])
        if request['compound_id'] != case['compound_id'] or request['dosing']['route'] != 'intravenous':
            raise ValueError('Cohort identity/route differs')
        required = {'molecular_weight', 'molecular_formula', 'logp', 'pka_types'}
        if not required <= {e['parameter'] for e in request['evidence']}:
            raise ValueError('Missing physicochemical input')
        # Old predicted CL/fu are discarded, and no clinical renal split is retained.
        request['evidence'] = [e for e in request['evidence'] if e['parameter'] not in
                              {'total_clearance', 'hepatic_clearance', 'renal_clearance', 'fraction_unbound'}]
        request['partition_method'] = 'PKSimStandard'
        request['target'].pop('terminal_fit', None)
        request['assumptions'] = ['Common frozen physicochemical inputs; no target clinical curve or PK scalar input.',
            'CL and fu will be independently re-predicted excluding all cohort drugs and generic scaffolds.',
            'Native total plasma CL is a lumped hepatic sink, without additional glomerular filtration.',
            'Previously viewed development cases; not a new blinded clinical validation.']
        case['request_template'] = request
        case['forbidden_clinical_sources'] = [original['reference_source'], original['full_training_source'],
                                             original['five_training_source']]
        collect_refs(case, pins)
        pins[case['physchem']] = sha256(ROOT/case['physchem'])
        cases.append(case)
    if len(cases) != 20 or len({c['compound_id'] for c in cases}) != 20:
        raise ValueError('All twenty distinct prespecified drugs are required')
    protocol = {'version': VERSION, 'created_at': now(), 'source_protocol': ref(SOURCE), 'cases': cases,
        'methods': list(METHODS), 'method_labels': LABELS, 'formulas': list(FORMULAS), 'workers': workers,
        'native_threads': 2, 'native_timeout_seconds': 180, 'directory_limit_bytes': 10*1024**3,
        'monitor_seconds': 1200, 'source_pins': pins, 'code_pins': {name: sha256(ROOT/name) for name in CODE},
        'paper_guard': paper_pins(), 'clinical_training_points_per_drug': 0,
        'parameter_training': 'Human PK data; exclude all target identities and generic scaffolds; missing-label losses masked.',
        'comparison_scope': 'RF/native and shared-rich-ML/native, five-formula, blueprint; literature-adapted, not original author QSAR replication.',
        'dataset_interpretation': 'Two source strata of one common twenty-drug development cohort, not two independent external datasets.',
        'references': ['https://doi.org/10.1007/s00204-024-03764-9', 'https://doi.org/10.1021/acs.jmedchem.5c00340'],
        'endpoints': ['plasma_concentration_time', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life',
                      'auc_inf', 'clearance', 'vz', 'mrt', 'vss'],
        'scoring': 'Drug-equal curve GMFE, log-RMSE, endpoint FE, within-twofold, coverage; clustered paired bootstrap 2000 seed20261004.',
        'prediction_lock_before_scoring': True, 'automatic_promotion': False, 'paper_updated': False}
    run.mkdir(parents=True, exist_ok=True)
    save(run/'protocol.json', protocol)
    save(run/'state.json', {'version': VERSION, 'protocol': ref(run/'protocol.json'), 'status': 'prepared',
        'created_at': now(), 'progress': {}, 'outcomes': {}, 'all_predictions_locked_before_scoring': False})
    return {'status': 'prepared', 'drugs': len(cases), 'comparisons': len(cases)*len(METHODS),
            'datasets': {'reported_clinical_10': 10, 'osp_observed_10': 10}}


class Pause(RuntimeError):
    pass


class Monitor:
    def __init__(self, run, protocol, state):
        self.run, self.protocol, self.state = run, protocol, state
        self.lock, self.done, self.halted = threading.RLock(), threading.Event(), threading.Event()
        self.last_hash = self.last_size = 0.

    def update(self, **values):
        with self.lock:
            self.state.update(values, updated_at=now())
            save(self.run/'state.json', self.state)

    def check(self, force=False):
        with self.lock:
            if self.halted.is_set() or (self.run/'PAUSE').exists():
                raise Pause(self.state.get('pause_reason', 'Pause requested'))
            tick = time.monotonic()
            if force or tick-self.last_hash >= self.protocol['monitor_seconds']:
                verify(self.protocol, self.state.get('runtime_code_patches'))
                self.last_hash = tick
            if force or tick-self.last_size >= 60:
                files = list(self.run.rglob('*'))
                if any(p.is_symlink() for p in files):
                    raise Pause('Unexpected symlink in experiment')
                size = 0
                for path in files:
                    try:
                        if path.is_file():
                            size += path.stat().st_size
                    except FileNotFoundError:
                        # Atomic JSON replacement can remove a temporary path.
                        continue
                if size > self.protocol['directory_limit_bytes']:
                    raise Pause('Batch exceeds 10 GiB; stop new submissions')
                self.update(monitor={'checked_at': now(), 'bytes': size, 'source_check_interval_seconds': 1200})
                self.last_size = tick

    def watch(self):
        while not self.done.wait(15):
            try:
                self.check()
            except Exception as error:
                self.halted.set()
                self.update(pause_reason=str(error))


def build_request(case, parameters, source, folder):
    request = deepcopy(case['request_template'])
    request['evidence'] = [e for e in request['evidence'] if e['parameter'] not in
                          {'total_clearance', 'hepatic_clearance', 'renal_clearance', 'fraction_unbound'}]
    request['forbidden_sources'] = [{'path': item['path'], 'sha256': item['sha256'],
                                    'location': item.get('location') or item.get('json_pointer') or
                                                item.get('role') or 'Sealed clinical source'}
                                   for item in case['forbidden_clinical_sources']]
    predictor = {'name': 'shared_scaffold_excluded_parameter_model', 'version': VERSION,
                 'training_overlap': False, 'applicability': {'target_structure_and_scaffold_excluded': True,
                                                            'human_IV_parameter_model': True}}
    for name, unit in (('total_clearance', 'ml/min/kg'), ('fraction_unbound', 'fraction')):
        value = float(parameters[name])
        if not math.isfinite(value) or value <= 0 or (name == 'fraction_unbound' and value > 1):
            raise ValueError('Invalid predicted ' + name)
        request['evidence'].append({'parameter': name, 'compound_id': case['compound_id'], 'value': value,
            'unit': unit, 'basis': 'predicted', 'source': {**source, 'location': case['compound_id']+'/'+name},
            'context': {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous'},
            'prediction_method': predictor, 'notes': 'Clinical target and development labels were not used.'})
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/'evidence.json', {'evidence': request['evidence']})
    request['evidence_ledger'] = {**ref(folder/'evidence.json'), 'location': 'Common cohort construction inputs'}
    save(folder/'request.json', request)
    return request


def prior_evidence(case, parameters, source, name, unit):
    return [{'parameter': name, 'compound_id': case['compound_id'], 'value': float(parameters[name]),
        'unit': unit, 'basis': 'predicted', 'source': {**source, 'location': case['compound_id']+'/'+name},
        'context': {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous',
                    'study_role': 'independent_parameters', 'independent_of_target': True},
        'prediction_method': {'name': 'shared_fixed_rich_ensemble', 'version': VERSION,
            'training_overlap': False, 'applicability': {'target_structure_and_scaffold_excluded': True,
                                                       'human_IV_parameter_model': True}}}]


def run_case(case, prior, source, run, monitor):
    from pbpk_kb.blueprint_policy import clinical_pbpk_comparison as methods
    from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
    cid = case['case_id']; folder = run/'models'/cid
    with monitor.lock:
        if cid in monitor.state['outcomes']:
            checked(monitor.state['outcomes'][cid])
            return cid
    monitor.check()
    values = prior['parameters'][case['compound_id']]
    rf_values = prior['rf_parameters'][case['compound_id']]
    request = build_request(case, values, source, folder/'rich_inputs')
    rf_request = build_request(case, rf_values, source, folder/'rf_inputs')
    entries, formula_results = {}, {}
    started = time.monotonic()

    def trial(key, function):
        monitor.check()
        output = folder/(key+'.json')
        with monitor.lock:
            progress = monitor.state['progress'].setdefault(cid, {'trials': {}})
            previous = progress['trials'].get(key)
        if previous:
            return checked(previous)
        if output.exists():
            preserve_interrupted(output)
        stamp = time.monotonic()
        try:
            working = folder/key
            cache = working/'result.json'
            result = None
            if working.exists():
                if cache.exists():
                    candidate = read(cache)
                    # Recover only a completed adapter receipt with unchanged inputs.
                    expected_request = rf_request if key == 'rf_standard' else request
                    if (candidate.get('version') != methods.VERSION or
                            candidate.get('request_sha256') != methods._digest(expected_request) or
                            candidate.get('status') not in ('predicted', 'failed')):
                        raise Pause('Uncommitted adapter receipt differs: '+str(cache.relative_to(ROOT)))
                    collect_refs(candidate, {})
                    result = {**candidate, 'result_ref': ref(cache)}
                else:
                    # An interrupted process is not a scientific prediction failure.
                    preserve_interrupted(working)
            if result is None:
                result = function()
            result.setdefault('status', 'predicted')
            if result['status'] == 'failed':
                result.setdefault('failure', result.get('reason', 'Prediction failed'))
                if result.get('reason') == 'native_execution_failed' and result.get('execution_ref'):
                    execution = checked(result['execution_ref'])
                    process = execution.get('process', {})
                    result['infrastructure_failure'] = bool(process.get('stop_reason') or
                        process.get('status') == 'blocked' or process.get('error'))
                    result['execution_failure_detail'] = {k:process.get(k) for k in
                        ('status', 'stop_reason', 'returncode', 'error')}
        except Exception as error:
            if isinstance(error, Pause):
                raise
            infrastructure = isinstance(error, (OSError, ImportError)) or type(error).__name__ == 'InfrastructurePause'
            result = {'status': 'failed', 'failure': type(error).__name__+': '+str(error),
                      'infrastructure_failure': infrastructure}
        result['elapsed_seconds'] = time.monotonic()-stamp
        save(output, result)
        with monitor.lock:
            progress['trials'][key] = ref(output)
            count = monitor.state.get('consecutive_infrastructure_errors', 0)
            count = count+1 if result.get('infrastructure_failure') else 0
            monitor.update(consecutive_infrastructure_errors=count)
            if count >= 3:
                monitor.halted.set()
                monitor.update(pause_reason='Three consecutive infrastructure failures')
        return result

    for formula in FORMULAS:
        formula_results[formula] = trial('native_'+formula,
            lambda formula=formula: methods.run_native_partition(request, folder/('native_'+formula), ROOT, formula))
    entries['ml_standard'] = deepcopy(formula_results['PKSimStandard'])
    entries['rf_standard'] = trial('rf_standard',
        lambda: methods.run_native_partition(rf_request, folder/'rf_standard', ROOT, 'PKSimStandard'))

    def ensemble():
        failed = [name for name, value in formula_results.items() if value['status'] != 'predicted']
        if failed:
            return {'status': 'failed', 'failure': 'Five-of-five ensemble requires all formulas: '+', '.join(failed)}
        curve = methods.mean_native_curves([formula_results[name]['curve'] for name in FORMULAS])
        return {'status': 'predicted', 'curve': curve, 'endpoints': summarize_extended_prediction(curve, request),
                'formula_members': list(FORMULAS), 'model_kind': 'Five separately executed PBPK models; arithmetic concentration ensemble'}
    entries['ml_five_partition'] = trial('ml_five_partition', ensemble)
    # Runner timing is outside the adapter's immutable result receipt.
    reusable_native = None
    if formula_results['PKSimStandard']['status'] == 'predicted':
        receipt = formula_results['PKSimStandard']['result_ref']
        reusable_native = {**checked(receipt), 'result_ref': receipt}
    entries['ours'] = trial('ours', lambda: methods.run_ours(request,
        prior_evidence(case, values, source, 'vss', 'l/kg'),
        prior_evidence(case, values, source, 'terminal_half_life', 'h'), folder/'ours', ROOT,
        native_result=reusable_native))
    for name, value in entries.items():
        if value['status'] == 'predicted':
            if value['endpoints'].get('observations_used') is not False:
                raise ValueError('Endpoint summary did not certify observation-free prediction')
            value['endpoints']['comparison_observations_used'] = False
            value['endpoints']['endpoint_estimator_uses_observations'] = False
        value['request'] = rf_request if name == 'rf_standard' else request
        value['method'] = name
        value['condition'] = 'no_clinical_curve'
        value['target_observations_used'] = False
    bundle = {'case_id': cid, 'compound_id': case['compound_id'], 'protocol': monitor.state['protocol'],
        'entries': entries, 'elapsed_seconds': time.monotonic()-started, 'locked_at': now()}
    save(folder/'predictions.json', bundle)
    with monitor.lock:
        monitor.state['outcomes'][cid] = ref(folder/'predictions.json')
        monitor.update(completed_drugs=len(monitor.state['outcomes']))
    return cid


def preserve_interrupted(path):
    """Retain interrupted work within this case; never overwrite or score it."""
    index = 1
    while path.with_name(path.name+f'.interrupted_{index}').exists():
        index += 1
    path.rename(path.with_name(path.name+f'.interrupted_{index}'))


def run_experiment(run=DEFAULT):
    from pbpk_kb.blueprint_learning.clinical_shared_prior import fit_shared_prior
    run = own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(run/'state.json'); protocol = checked(state['protocol'])
        verify(protocol, state.get('runtime_code_patches'))
        if state['status'] == 'complete':
            return checked(state['results'])['summary']
        monitor = Monitor(run, protocol, state)
        thread = threading.Thread(target=monitor.watch, daemon=True)
        monitor.update(status='training_parameters', pid=os.getpid(), started_at=state.get('started_at', now()))
        thread.start()
        try:
            monitor.check(force=True)
            targets = [{'compound_id': c['compound_id'], 'smiles': c['smiles']} for c in protocol['cases']]
            def training_progress(event):
                monitor.check()
                monitor.update(training_progress=event)
                print(json.dumps({'training_progress': event}), flush=True)
            prior = fit_shared_prior(targets, run/'parameter_model', ROOT, n_jobs=8, device='auto',
                                     progress=training_progress)
            source = ref(run/'parameter_model/predictions_audit.json')
            monitor.update(status='running_pbpk', parameter_predictions=source)
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                futures = [pool.submit(run_case, c, prior, source, run, monitor) for c in protocol['cases']]
                for future in as_completed(futures):
                    future.result()
            monitor.check(force=True)
            if set(state['outcomes']) != {c['case_id'] for c in protocol['cases']}:
                raise Pause('Prediction matrix is incomplete')
            monitor.update(status='predictions_locked', all_predictions_locked_before_scoring=True,
                           predictions_locked_at=now())
        except Exception as error:
            monitor.halted.set()
            monitor.update(status='paused', pause_reason=type(error).__name__+': '+str(error))
            raise
        finally:
            monitor.done.set(); thread.join(timeout=20)
    return report(run)


def summarize(rows, cases):
    import report_blueprint_validation as scores
    strata = {'all_20': cases, 'reported_clinical_10': [c for c in cases if c['cohort']=='original'],
              'osp_observed_10': [c for c in cases if c['cohort']=='extension']}
    summaries, comparisons = [], {}
    for name, group in strata.items():
        ids = {c['case_id'] for c in group}; clusters = {c['case_id']: c['cluster_id'] for c in group}
        values = {}
        for method in METHODS:
            selected = [r for r in rows if r['case_id'] in ids and r['method']==method]
            valid = {r['case_id']: r['curve_metrics']['gmfe'] for r in selected
                     if scores.positive(r.get('curve_metrics', {}).get('gmfe'))}
            values[method] = valid
            summaries.append({'dataset': name, 'method': method, 'attempted': len(group), 'scored': len(valid),
                'failed_or_unscorable': len(group)-len(valid), 'within_twofold': sum(x <= 2 for x in valid.values()),
                'curve_gmfe': scores.clustered_geomean(valid, {k:clusters[k] for k in valid}, seed=20261004),
                'endpoint_errors': endpoint_summary(selected)})
        comparisons[name] = {m: scores.paired_ratio(values['ours'], values[m], clusters,
                                                   expected_ids=ids, seed=20261004) for m in METHODS if m!='ours'}
    return summaries, comparisons


def endpoint_summary(rows):
    from pbpk_kb.blueprints.clinical_prediction_nca import UNITS
    result = {}
    for endpoint in UNITS:
        items = [r.get('endpoints', {}).get(endpoint, {}) for r in rows]
        groups = sorted({x.get('reference_kind','unavailable') for x in items})
        result[endpoint] = {}
        for kind in groups:
            selected = [x for x in items if x.get('reference_kind','unavailable') == kind]
            errors = [x['error'] for x in selected if x.get('error') is not None]
            result[endpoint][kind] = {'scored': len(errors), 'available_rows': len(selected),
                'attempted': len(rows), 'weak_tail': sum(bool(x.get('weak_tail')) for x in selected),
                'gmfe': math.exp(sum(math.log(x) for x in errors)/len(errors)) if errors and endpoint!='tmax' else None,
                'mae_h': sum(errors)/len(errors) if errors and endpoint=='tmax' else None,
                'within_twofold': sum(x<=2 for x in errors) if endpoint!='tmax' else None}
    return result


def report(run=DEFAULT):
    import report_blueprint_validation as legacy
    from score_cross_drug_blueprint import score_curve_reference, _native_fields
    from render_current_clinical_table import native_value
    from pbpk_kb.blueprints.clinical_prediction_nca import UNITS
    run = own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(run/'state.json'); protocol = checked(state['protocol'])
        verify(protocol, state.get('runtime_code_patches'))
        if not state.get('all_predictions_locked_before_scoring') or set(state['outcomes'])!={c['case_id'] for c in protocol['cases']}:
            raise ValueError('Every method must finish or fail and lock before opening clinical observations')
        loaded = [(c, state['outcomes'][c['case_id']], checked(state['outcomes'][c['case_id']])) for c in protocol['cases']]
        for case, reference, bundle in loaded:
            if (bundle['protocol'] != state['protocol'] or set(bundle['entries']) != set(METHODS) or
                    bundle.get('case_id') != case['case_id'] or bundle.get('compound_id') != case['compound_id']):
                raise ValueError('Prediction matrix differs from frozen protocol')
            for method, entry in bundle['entries'].items():
                if entry.get('method') != method or entry.get('request', {}).get('compound_id') != case['compound_id']:
                    raise ValueError('Prediction drug or method identity differs')
                if entry['status'] not in ('predicted', 'failed'):
                    raise ValueError('Unfinished prediction')
                if entry['status']=='predicted':
                    _native_fields(entry)
        store, rows = legacy.Store(ROOT), []
        for case, reference, bundle in loaded:
            for method, entry in bundle['entries'].items():
                row = {key: case[key] for key in ('case_id', 'compound_id', 'drug', 'study_id', 'cluster_id', 'dataset')}
                row.update(method=method, attempt_status=entry['status'], scoring_status='failed',
                           failure=entry.get('failure'), endpoints={}, elapsed_seconds=entry.get('elapsed_seconds'))
                if entry['status']=='predicted':
                    try:
                        if case['reference_kind']=='observed_curve':
                            row.update(score_curve_reference(case, entry, store))
                        else:
                            clinical, curve, endpoints = legacy._case_reference(case, store)
                            base = '/entries/'+method
                            unit = {'prediction': {**reference, 'json_pointer': base+'/curve'},
                                    'request': {**reference, 'json_pointer': base+'/request'}}
                            output, _, _ = legacy._score_prediction(case, clinical, curve, endpoints, unit, store)
                            output.pop('curve', None)
                            for endpoint, value in output['endpoints'].items():
                                value['reference_kind'] = 'original_reported' if value['reference'] is not None else 'unavailable'
                                value['native_prediction'], _, value['native_prediction_reason'] = native_value(
                                    entry['endpoints']['endpoints'][endpoint], endpoint, value['unit'], entry['request']['human']['weight_kg'])
                            row.update(output, scoring_status='scored' if legacy.positive(output['curve_metrics'].get('gmfe')) else 'partial')
                    except (ValueError, ArithmeticError, KeyError) as error:
                        if 'Source lock mismatch' in str(error) or 'Source changed during reporting' in str(error):
                            raise
                        row.update(scoring_status='scoring_failed', failure=type(error).__name__+': '+str(error))
                rows.append(row)
        store.unchanged()
        summaries, comparisons = summarize(rows, protocol['cases'])
        result = {'version': VERSION, 'protocol': state['protocol'], 'units': rows, 'summary': summaries,
            'paired_ours_over_comparator': comparisons, 'predictions_locked_before_scoring': True,
            'limitations': ['Previously viewed development cohort, not new blinded validation.',
                'Common input adaptations of published construction ideas, not exact original QSAR replication.',
                'Shared physicochemical evidence includes measured pKa; this is not fully SMILES-only.',
                'Two source strata are not two independent large public datasets.',
                'Original scalar reports and observed-curve-derived window references are reported separately.',
                'No reference is manufactured for missing terminal/organ endpoints.'], 'paper_updated': False}
        save(run/'results.json', result)
        fields = ['drug', 'study_id', 'dataset', 'method', 'status', 'endpoint', 'unit', 'prediction',
                  'native_prediction', 'observation', 'reference_kind', 'error', 'error_metric',
                  'curve_gmfe', 'curve_ln_rmse', 'n', 'weak_tail', 'reason']
        with (run/'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in rows:
                base = {k:row[k] for k in ('drug','study_id','dataset','method')}
                metric = row.get('curve_metrics', {})
                writer.writerow({**base, 'status': row['scoring_status'], 'endpoint': 'plasma_concentration_time',
                    'reference_kind': 'clinical_observed_curve', 'curve_gmfe': metric.get('gmfe'),
                    'curve_ln_rmse': metric.get('ln_rmse'), 'n': metric.get('n'), 'reason': row.get('failure')})
                for endpoint, unit in UNITS.items():
                    value = row['endpoints'].get(endpoint, {})
                    writer.writerow({**base, 'status': row['scoring_status'], 'endpoint': endpoint,
                        'unit': value.get('unit',unit), 'prediction': value.get('value'),
                        'native_prediction': value.get('native_prediction'), 'observation': value.get('reference'),
                        'reference_kind': value.get('reference_kind','unavailable'), 'error': value.get('error'),
                        'error_metric': value.get('error_metric'), 'weak_tail': value.get('weak_tail'),
                        'reason': value.get('reason') or value.get('reference_reason') or row.get('failure')})
        lines = ['# Shared clinical ML–PBPK comparison', '',
            'Twenty previously viewed drugs; two source strata; no own-drug clinical calibration. All predictions were locked before scoring.', '',
            '| Dataset | Method | Scored / attempted | Curve GMFE (95% CI) | Within 2-fold |',
            '|---|---|---:|---:|---:|']
        for row in summaries:
            score = row['curve_gmfe']; estimate = score['estimate']; ci = score.get('ci95')
            number = '—' if estimate is None else f'{estimate:.3f}'
            if ci: number += f' ({ci[0]:.3f}–{ci[1]:.3f})'
            lines.append(f"| {row['dataset']} | {LABELS[row['method']]} | {row['scored']}/{row['attempted']} | {number} | {row['within_twofold']} |")
        lines.extend(['', 'Full endpoint predictions, references and errors: [results.csv](results.csv).', '',
            'The fixed native and five-partition pipelines are adaptations using a shared, retrained predictor. They are not exact reproductions of Geci/Jia QSAR models.',
            'The blueprint uses independently predicted Vss and half-life to set capacity and exchange. No clinical target value selects a parameter or formula.',
            'Missing clinical references remain empty. Failures remain in attempted denominators. Current paper files are unchanged.', '',
            'Control: `runtime/app-env/bin/python -B scripts/run_clinical_pbpk_comparison.py status`.',
            'Resume: same command with `run --background`; final scoring: `report`; remove temporary files: `cleanup`.'])
        (run/'README.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete', completed_at=now(), results=ref(run/'results.json'),
                     result_table=ref(run/'results.csv'), result_readme=ref(run/'README.md'))
        save(run/'state.json', state)
        return {'status':'complete', 'summary': summaries, 'comparisons':comparisons}


def status(run=DEFAULT):
    run = own(run)
    if not (run/'state.json').exists(): return {'status':'not_prepared'}
    state = read(run/'state.json')
    completed = len(state.get('outcomes', {}))
    return {key:state.get(key) for key in ('status','pid','started_at','updated_at','completed_at','pause_reason','monitor','training_progress')} | {
        'completed_drugs': completed, 'total_drugs':20,
        'completed_trials': sum(len(p.get('trials',{})) for p in state.get('progress',{}).values()),
        'expected_trials':160, 'result_table':state.get('result_table')}


def cleanup(run=DEFAULT):
    run = own(run); state=read(run/'state.json')
    if state['status']!='complete': raise ValueError('Cleanup requires complete locked results')
    checked(state['results'])
    table = inside(ROOT/state['result_table']['path'], ROOT, exists=True)
    if sha256(table) != state['result_table']['sha256']:
        raise ValueError('Final result table changed')
    removed=0
    for path in run.rglob('*.tmp'):
        if path.is_file() and not path.is_symlink(): path.unlink(); removed+=1
    return {'removed_temporary_files':removed, 'retained':'Model packages, locked predictions, provenance and failed diagnostics'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status','report','cleanup'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args(); run=own(args.run_dir)
    if args.command=='run' and args.background:
        with (run/'run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            environment=os.environ.copy()
            environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (run/'run.log').open('ab') as log:
                process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--run-dir',str(run)],
                    cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(json.dumps({'pid':process.pid,'log':str(run/'run.log')}));return
    actions={'prepare':lambda:prepare(run,args.workers),'run':lambda:run_experiment(run),
             'status':lambda:status(run),'report':lambda:report(run),'cleanup':lambda:cleanup(run)}
    print(json.dumps(actions[args.command](),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
