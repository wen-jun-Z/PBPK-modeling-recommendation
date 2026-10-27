#!/usr/bin/env python3
"""Frozen 31-drug clinical validation of a training-only clearance prior.

Only predicted CL changes. The source-qualified evidence policy, other priors,
human physiology and learned-exchange-design-2 remain unchanged. Predictions
and explicit failures are all locked before the clinical scorer opens targets.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_qualified_binding_validation as source_runner
import run_distribution_consistency_validation as infra
import run_joint_partition_validation as statistics
from pbpk_kb.blueprint_policy import clearance_uncertainty
from pbpk_kb.blueprints.source_workflow import save, sha256

base = source_runner.base
value, retained, clone = base.value, base.retained, base.clone
VERSION = 'foundation-blueprint-validation-1'
PRIOR_VERSION = 'foundation-clearance-prior-1'
CANDIDATE = 'foundation_clearance'
SOURCE = source_runner.DEFAULT
BATCH = ROOT/'experiments/blueprint_design/foundation_prior_20261007'
DEFAULT = BATCH/'clinical'
PRIOR = BATCH/'prior/predictions.json'
CONDITIONS = base.CONDITIONS
FIXED_PARAMETERS = ('fraction_unbound', 'vss', 'terminal_half_life', 'mrt')


def own(path):
    path = value.inside(path, ROOT)
    if path != DEFAULT:
        raise ValueError('Use only foundation_prior_20261007/clinical')
    return path


def validate_prior(prior, original, source_receipt):
    """Keep the new predictor contract distinct from the old selected recipe."""
    if prior.get('version') != PRIOR_VERSION or prior.get('result_sha256') != base.training_digest(
            {k:v for k,v in prior.items() if k != 'result_sha256'}):
        raise ValueError('Foundation prior version or checksum differs')
    if prior.get('source_parameter_prior') != source_receipt:
        raise ValueError('The current union-excluded parameter prior must be bound explicitly')
    ids = set(original['parameters'])
    if set(prior.get('parameters', {})) != ids or set(prior.get('per_compound_audit', {})) != ids:
        raise ValueError('Retain the complete frozen inference/exclusion target cohort')
    if prior.get('targets') != original['targets']:
        raise ValueError('Target structures or exclusion cohort differs')
    audit = prior.get('audit', {})
    for flag in ('target_labels_used', 'target_observation_files_opened', 'training_structure_overlap',
                 'training_generic_scaffold_overlap', 'clinical_results_used_for_selection'):
        if audit.get(flag) is not False:
            raise ValueError('Training-only prior certification missing: '+flag)
    pool_count = audit.get('eligible_human_training_structures', audit.get('human_training_structures'))
    if pool_count != original['audit']['human_training_structures']:
        raise ValueError('The frozen training structure exclusion differs')
    fit_count, calibration_count = audit.get('final_CL_context_count'), audit.get('CL_calibration_count')
    eligible_cl = audit.get('eligible_CL_labels')
    members = audit.get('human_parameter_training_indices', {}).get('total_clearance')
    if (not all(type(n) is int and n > 0 for n in (fit_count, calibration_count, eligible_cl))
            or fit_count+calibration_count != eligible_cl or eligible_cl > pool_count
            or not isinstance(members, list) or len(set(members)) != fit_count):
        raise ValueError('CL context-fit and held-out calibration counts/membership must be explicit')
    if not prior.get('method_name') or not prior.get('algorithm_version'):
        raise ValueError('Actual method name and algorithm version are required')
    for name, unit in original['units'].items():
        if prior.get('units', {}).get(name, '').lower() != unit.lower():
            raise ValueError('Prior unit differs: '+name)
    for cid in ids:
        row = prior['parameters'][cid]
        if not value.scores.positive(row.get('total_clearance')) or isinstance(row['total_clearance'], bool):
            raise ValueError('Invalid total clearance: '+cid)
        for name in FIXED_PARAMETERS:
            if row.get(name) != original['parameters'][cid][name]:
                raise ValueError('Only predicted CL may change: '+cid+'/'+name)
        subject = prior['per_compound_audit'][cid]
        if subject.get('human_training_overlap') is not False or subject.get('target_structure_and_scaffold_excluded') is not True:
            raise ValueError('Target training exclusion missing: '+cid)
        if any(subject.get(k) != original['per_compound_audit'][cid][k]
               for k in ('normalized_identity', 'generic_scaffold')):
            raise ValueError('Target chemical identity differs: '+cid)
    return prior


def checked_prior(path, source):
    receipt = value.old.ref(value.inside(path, ROOT, exists=True))
    prior = value.old.checked(receipt)
    original = value.old.checked(source['parameter_prior'])
    validate_prior(prior, original, source['parameter_prior'])
    if not {c['compound_id'] for c in source['cases']} <= set(prior['parameters']):
        raise ValueError('Each clinical case requires a certified prediction')
    for case in source['cases']:
        identity = base.normalize_structure(case['smiles'])['identity_smiles']
        if identity != prior['per_compound_audit'][case['compound_id']]['normalized_identity']:
            raise ValueError('Clinical identity differs from prior inference')
    for field in ('source_refs', 'code_pins'):
        if not prior.get(field):
            raise ValueError('Missing frozen training dependencies: '+field)
        base.merge_pins({}, prior[field])
    value.old.collect_refs(training_receipts(prior), {})
    return prior, receipt


def training_receipts(prior):
    """The author's checkpoint and fitted context are distinct reproducibility inputs."""
    return {name: prior[name] for name in ('checkpoint_receipt', 'context_receipt')}


def replace_clearance(original, prior, receipt):
    """Return an exact identity unless a supported predicted CL value changes."""
    task = deepcopy(original)
    audit = {'changed': False, 'reason': None, 'prior': receipt,
        'source_request_sha256': value.model._digest(original['request'])}
    if task.get('construction_blocked'):
        audit['reason'] = 'Pre-existing chemical construction blocker retained; CL not applied'
        return task, audit
    rows = [r for r in task['request']['evidence'] if r['parameter'] == 'total_clearance']
    if len(rows) != 1:
        raise ValueError('Each supported request must contain exactly one selected total CL')
    row = rows[0]
    if row['basis'] in {'measured', 'derived'}:
        audit['reason'] = 'Independent total CL and organ clearance remain fixed'
        return task, audit
    if row['basis'] != 'predicted':
        raise ValueError('Unexpected clearance basis')
    if any(r['parameter'] in {'renal_clearance', 'hepatic_clearance'} for r in task['request']['evidence']):
        raise ValueError('Predicted total CL cannot overwrite an independent organ split')
    if row['unit'].lower() != 'ml/min/kg':
        raise ValueError('Predicted CL requires the frozen weight-normalized unit')
    val = prior['parameters'][task['request']['compound_id']]['total_clearance']
    if not value.scores.positive(val) or isinstance(val, bool):
        raise ValueError('Invalid predicted clearance')
    audit.update(before=float(row['value']), after=float(val), unit='mL/min/kg')
    if float(val) == float(row['value']):
        audit['reason'] = 'Exactly equal CL value; entire original input and provenance retained with explicit prior equality receipt'
        return task, audit
    cid = task['request']['compound_id']
    row.update(value=float(val), source={**receipt, 'location': cid+'/total_clearance'},
        prediction_method={'name': prior['method_name'], 'version': PRIOR_VERSION,
            'algorithm_version': prior['algorithm_version'], 'training_overlap': False,
            'applicability': {'human_IV_parameter_model': True, 'target_structure_and_scaffold_excluded': True}},
        notes='Training-only foundation CL prediction; all union target identities/scaffolds excluded; no clinical curve fit')
    task['core']['foundation_clearance_prior'] = receipt
    audit.update(changed=True, reason='Predicted CL replaced; all other evidence and sidecars retained')
    if task['sidecars'] != original['sidecars']:
        raise ValueError('Vss and half-life must remain exactly unchanged')
    return task, audit


def source_outcome(original, receipt, pins):
    outcome = value.old.checked(receipt)
    entry = outcome['entry']
    if (outcome.get('key') != original['key'] or not outcome.get('locked_at')
            or entry.get('request') != original['request']
            or entry.get('target_observations_used') is not False
            or entry.get('status') not in {'predicted', 'failed'}
            or retained.key_for(original['request'], original['sidecars'], original['core']) != original['key']):
        raise ValueError('Source is not an exact-input prediction-locked result')
    if not original.get('construction_blocked') and entry.get('request_sha256') != value.model._digest(original['request']):
        raise ValueError('Source request digest differs')
    value.old.collect_refs(outcome, pins); pins[receipt['path']] = receipt['sha256']
    return outcome


def input_decision(task, prior, source_protocol):
    """Bind legacy evidence to verified frozen inputs without relabelling it."""
    records = [row for row in task['request']['evidence'] if row['parameter'] == 'total_clearance']
    if len(records) != 1:
        raise ValueError('One frozen total CL record is required')
    record = records[0]; legacy = None
    if record['basis'] in {'measured', 'derived'}:
        original = value.old.checked(task['source_prediction_receipt'])
        if (original['entry']['request'] != task['request'] or not original.get('locked_at')
                or original['entry'].get('target_observations_used') is not False):
            raise ValueError('Legacy CL evidence must be the exact previously locked request')
        # The source-qualified protocol is hash-verified before this function.
        legacy = {'record': deepcopy(record), 'source_qualified_protocol': source_protocol,
                  'source_outcome': task['source_prediction_receipt']}
    decision = clearance_uncertainty.decide(task['request'],
        prior['uncertainty'][task['request']['compound_id']], reviewed_legacy_source=legacy,
        construction_blocked=task.get('construction_blocked'))
    return {**decision, 'request_sha256': value.model._digest(task['request'])}


def prepare(run=DEFAULT, prior_path=PRIOR, workers=14):
    run = own(run)
    if (run/'protocol.json').exists():
        raise FileExistsError('Protocol is frozen; resume with run')
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 14:
        raise ValueError('workers must be 1..14')
    state = value.old.read(SOURCE/'state.json')
    source = value.old.checked(state['protocol']); value.verify(source)
    if (state['status'] != 'complete' or state.get('all_predictions_locked_before_scoring') is not True
            or set(state['outcomes']) != set(source['tasks'])):
        raise ValueError('Completed source-qualified baseline required')
    prior, receipt = checked_prior(prior_path, source)
    pins, code = deepcopy(source['source_pins']), deepcopy(source['code_pins'])
    for ref in (state['protocol'], state['results'], receipt):
        base.merge_pins(pins, {ref['path']: ref['sha256']})  # Do not parse source clinical results.
    base.merge_pins(pins, prior['source_refs']); base.merge_pins(code, prior['code_pins'])
    value.old.collect_refs(training_receipts(prior), pins)
    base.merge_pins(code, {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))})
    for module in (base, source_runner, infra, statistics, clearance_uncertainty,
                   base.execution, base.execution.separate):
        path = Path(module.__file__)
        base.merge_pins(code, {str(path.resolve().relative_to(ROOT)): sha256(path)})
    run.mkdir(parents=True, exist_ok=True)
    (run/'inputs').mkdir(exist_ok=True)
    tasks, units, baselines, audits, decisions = {}, [], {}, {}, {}
    transformed = {}
    for oldkey, original in source['tasks'].items():
        if (original['core']['adapter_version'] != 'clinical-pbpk-comparison-1'
                or original['core']['design_version'] != 'learned-exchange-design-2'):
            raise ValueError('Retain the unchanged selected PBPK engine')
        outcome_ref = state['outcomes'][oldkey]
        source_outcome(original, outcome_ref, pins)
        task, audit = replace_clearance(original, prior, receipt)
        key = retained.key_for(task['request'], task['sidecars'], task['core'])
        task.update(key=key, candidate=CANDIDATE, native_source_ref=None,
            source_prediction_receipt=outcome_ref, source_exact_input_digest=source_runner.exact_input_digest(original),
            reuse_value_receipt=outcome_ref if not audit['changed'] else None)
        if audit['changed']:
            ledger = run/'inputs'/(key[:20]+'.json')
            save(ledger, {'version': VERSION, 'clearance_prior': receipt, 'evidence': task['request']['evidence']})
            task['request']['evidence_ledger'] = {**value.old.ref(ledger), 'location': 'Exact CL-only construction inputs'}
            pins[str(ledger.relative_to(ROOT))] = sha256(ledger)
            value.model._check_request(task['request'])
            for rows in task['sidecars'].values():
                value.model._check_sidecars(task['request'], rows, ROOT)
            built = value.model.assemble_evidence_blueprint(task['request'], ROOT)
            if built['status'] != 'assembled':
                raise ValueError(task['case_id']+': new CL request failed input preflight '+str(built.get('gaps')))
        elif source_runner.exact_input_digest(task) != source_runner.exact_input_digest(original):
            raise ValueError('A cached prediction requires fully identical inputs')
        if key != retained.key_for(task['request'], task['sidecars'], task['core']):
            raise ValueError('New ledger changed the scientific fingerprint')
        if key in tasks and source_runner.exact_input_digest(task) != source_runner.exact_input_digest(tasks[key]):
            raise ValueError('Input fingerprint collision')
        decisions[key] = input_decision(task, prior, state['protocol'])
        tasks[key] = task; audits[key] = audit; transformed[oldkey] = key
    for unit in source['units']:
        key = transformed[unit['key']]
        units.append({**deepcopy(unit), 'key': key, 'source_key': unit['key'], 'clearance_changed': audits[key]['changed']})
        if unit['condition'] == 'all_ml':
            baselines[unit['case_id']] = {'fingerprint': key}
    matrix = {(c['case_id'], con) for c in source['cases'] for con in CONDITIONS}
    if len(units) != len(matrix) or {(u['case_id'], u['condition']) for u in units} != matrix:
        raise ValueError('Retain every source case and condition')
    source_keys = {(u['case_id'], u['condition']): u['key'] for u in source['units']}
    stability_ids = [c['case_id'] for c in source['cases']
        if source_keys[c['case_id'], 'available'] != source_keys[c['case_id'], 'missing_clearance']]
    protocol = deepcopy(source)
    protocol.update(version=VERSION, candidate=CANDIDATE, created_at=value.old.now(), tasks=tasks, units=units,
        baselines=baselines, source_pins=pins, code_pins=code, source_qualified_protocol=state['protocol'],
        source_qualified_results=state['results'], parameter_prior=receipt,
        source_parameter_prior=source['parameter_prior'], parameter_mode='foundation_clearance_only',
        parameter_target_count=len(prior['parameters']), input_adjustments=audits,
        uncertainty_policy_version=clearance_uncertainty.VERSION, uncertainty_decisions=decisions,
        stability_case_ids=stability_ids,
        adoption_rule={'source': 'Prespecified foundation CL development gate',
            'coverage_all_conditions_not_decreased': True,
            'available_and_missing_clearance_common_gmfe_not_increased': True,
            'meaningful_benefit_count_exceeds_harm_in_both_primary_conditions': True,
            'fixed_source_actual_missing_cl_drift_and_error_ratio_not_increased': True,
            'automatically_update_selected_version': False},
        workers=workers, native_threads=2, native_timeout_seconds=180, monitor_seconds=1200,
        directory_limit_bytes=10*1024**3, directory_budget_scope=str(BATCH.relative_to(ROOT)),
        new_unique_predictions=sum(t['reuse_value_receipt'] is None for t in tasks.values()),
        expected_new_native_calls=sum(t['reuse_value_receipt'] is None for t in tasks.values()),
        no_clinical_curve_fitting=True, new_training_calls=0, new_independent_clinical_samples=0,
        cache_only=False, publication_snapshot=value.old.paper_pins(), paper_guard={},
        unchanged_parameters=list(FIXED_PARAMETERS), design_version='learned-exchange-design-2',
        limitations=[
            'Previously viewed 31-drug development cohort; original twenty and added eleven compounds are separate strata, not new blinded experiments.',
            'Source-reported scalar endpoints and finite-window curve-derived endpoints remain separate; absent references are not fabricated.',
            'General binding evidence can have unknown underlying-study overlap; explicit source qualification and original contexts remain unchanged.',
            'Only predicted total CL is replaced by a training-only prior. Independent CL and organ clearance, fu/Vss/half/MRT and the human PBPK engine remain fixed.',
            'The eligible human pool is unchanged, but CL fitting excludes its held-out interval-calibration subset; the new and original CL predictors do not use identical context-fit sample counts.',
            'All 31 cases have already been inspected in development. Current fitting excludes all 40 structures/scaffolds; historical recipe selection used labels from Ethinylestradiol, Montelukast, Paroxetine and Pitavastatin.',
            'Exact-input reused predictions are not new fits, compounds, patients or independent experiments.',
            'All original evidence-eligible denominators and chemical construction failures remain; a returned prediction is not clinical qualification.'])
    value.verify(protocol, publication=True)
    save(run/'protocol.json', protocol)
    save(run/'state.json', {'version': VERSION, 'candidate': CANDIDATE, 'status': 'prepared',
        'protocol': value.old.ref(run/'protocol.json'), 'created_at': value.old.now(),
        'outcomes': {}, 'all_predictions_locked_before_scoring': False})
    return {'status': 'prepared', 'clinical_cases': len(source['cases']), 'logical_units': len(units),
        'unique_predictions': len(tasks), 'new_native_calls_expected': protocol['expected_new_native_calls'],
        'reused_exact_predictions': len(tasks)-protocol['new_unique_predictions'],
        'retained_construction_blockers': sum(bool(t.get('construction_blocked')) for t in tasks.values())}


def invoke(task, folder):
    """Recover only a complete outcome bound to this request and predictor."""
    result_path = folder/'result.json'
    entry = None
    if result_path.exists():
        result = value.old.read(result_path)
        if (result.get('experiment_fingerprint') != task['key'] or result.get('experiment_candidate') != CANDIDATE
                or result.get('request_sha256') != value.model._digest(task['request'])):
            raise ValueError('Recovery result belongs to different inputs')
        value.old.collect_refs(result, {})
        if infra.infrastructure_failure(result) is None:
            entry = {**result, 'result_ref': value.old.ref(result_path)}
    if entry is None:
        if folder.exists():
            value.old.preserve_interrupted(folder)
        entry = base.execution.separate._executor(task)(task['request'], task['sidecars']['vss'],
            task['sidecars']['half'], folder, ROOT, native_result=None)
    if infra.infrastructure_failure(entry) is not None:
        raise value.old.Pause('Native infrastructure failure is not a clinical prediction failure')
    if (entry.get('request_sha256') != value.model._digest(task['request'])
            or entry.get('observations_used') is not False or entry.get('status') not in {'predicted', 'failed'}):
        raise ValueError('Prediction identity or observation boundary differs')
    if entry.get('native_solver_calls') != 1 or not 0 <= entry.get('linear_trajectory_calls', 0) <= 33:
        raise ValueError('Native/linear call budget differs')
    return entry


def solve(task, run, monitor):
    if task['reuse_value_receipt'] is not None:
        original = value.old.checked(task['reuse_value_receipt'])
        if original['entry']['request'] != task['request']:
            raise value.old.Pause('Cached request differs')
        return retained.solve(task, run, monitor)
    key = task['key']
    with monitor.lock:
        if key in monitor.state['outcomes']:
            item = value.old.checked(monitor.state['outcomes'][key])
            if item['key'] != key or item['entry']['request'] != task['request']:
                raise value.old.Pause('Locked outcome differs')
            return key
    monitor.check(); folder = run/'models'/key[:20]; started = time.monotonic()
    try:
        entry = deepcopy(invoke(task, folder))
        entry.update(request=deepcopy(task['request']), target_observations_used=False, condition='no_clinical_curve')
        if entry['status'] == 'predicted':
            if entry['endpoints'].get('observations_used') is not False:
                raise ValueError('Clinical observations used during endpoint prediction')
            entry['endpoints'].update(comparison_observations_used=False, endpoint_estimator_uses_observations=False)
            value._native_fields(entry)
        else:
            entry.setdefault('failure', entry.get('reason', 'Prediction failed'))
        item = {'key': key, 'case_id': task['case_id'], 'entry': entry, 'reused': False,
            'source_prediction_receipt': task['source_prediction_receipt'],
            'new_native_calls': entry.get('native_solver_calls', 0),
            'new_linear_trajectory_calls': entry.get('linear_trajectory_calls', 0),
            'elapsed_seconds': time.monotonic()-started, 'locked_at': value.old.now()}
        path = run/'models'/f'{key[:20]}.json'; save(path, item)
    except Exception as error:
        with monitor.lock:
            count = monitor.state.get('consecutive_infrastructure_errors', 0)+1
            monitor.update(consecutive_infrastructure_errors=count, pause_reason=type(error).__name__+': '+str(error))
            if count >= 3: monitor.halted.set()
        save(run/'models'/f'{key[:20]}.infrastructure.json', {'key': key,
            'reason': type(error).__name__+': '+str(error), 'scientific_failure': False, 'time': value.old.now()})
        raise value.old.Pause('Execution exception retained for review: '+str(error)) from error
    with monitor.lock:
        monitor.state['outcomes'][key] = value.old.ref(path)
        monitor.update(completed_unique=len(monitor.state['outcomes']), consecutive_infrastructure_errors=0)
    return key


def stability_statistics(result, previous, protocol):
    """Use the source-defined missing-CL intervention group, never score selection."""
    ids = set(protocol['stability_case_ids'])
    cases = [c for c in protocol['cases'] if c['case_id'] in ids]
    if len(cases) != len(ids):
        raise ValueError('Unknown prespecified missing-clearance case')
    def summarize(data):
        available = infra._case_rows(data, 'available', cases)
        missing = infra._case_rows(data, 'missing_clearance', cases)
        a = {cid: row['curve_metrics'].get('gmfe') for cid, row in available.items()}
        m = {cid: row['curve_metrics'].get('gmfe') for cid, row in missing.items()}
        drift = {cid: (row.get('prediction_drift') or {}).get('curve_geometric_fold_change')
                 for cid, row in missing.items()}
        return {'missing_over_available_error': base.paired(m, a, cases),
            'prediction_drift': base.drug_equal(drift, cases),
            'available_scored': sum(value.scores.positive(v) for v in a.values()),
            'missing_clearance_scored': sum(value.scores.positive(v) for v in m.values())}
    source, candidate = summarize(previous), summarize(result)
    checks = {}
    for metric in ('missing_over_available_error', 'prediction_drift'):
        before, after = source[metric]['estimate'], candidate[metric]['estimate']
        checks[metric+'_not_increased'] = (value.scores.positive(before) and value.scores.positive(after)
                                          and after <= before*(1+1e-12))
    checks['scored_coverage_not_decreased'] = all(candidate[k] >= source[k]
        for k in ('available_scored', 'missing_clearance_scored'))
    return {'case_ids': sorted(ids), 'case_selection': 'Frozen source available versus missing-CL input difference',
        'source': source, 'candidate': candidate, 'checks': checks, 'passed': all(checks.values()),
        'clinical_safety_established': False,
        'interpretation': 'Prespecified development nonregression gate; error ratios and drift are not calibrated clinical risk'}


def report(run=DEFAULT):
    run = own(run)
    bound = value.old.read(run/'state.json'); protocol = value.old.checked(bound['protocol'])
    if (not bound.get('all_predictions_locked_before_scoring')
            or set(bound['outcomes']) != set(protocol['tasks'])):
        raise ValueError('Lock every prediction before target scoring')
    if set(protocol['uncertainty_decisions']) != set(protocol['tasks']):
        raise ValueError('Every exact input needs its pre-score uncertainty decision')
    for key, decision in protocol['uncertainty_decisions'].items():
        if (decision['request_sha256'] != value.model._digest(protocol['tasks'][key]['request'])
                or decision.get('observations_used') is not False):
            raise ValueError('Input-only uncertainty decision differs')
    def scoring_save(path, payload):
        if Path(path) == run/'state.json' and payload.get('status') == 'complete':
            payload = {**payload, 'status': 'scoring'}
        save(path, payload)
    # Publish complete only after both scoring and source-paired statistics exist.
    info = clone(base.report, own=own, VERSION=VERSION, save=scoring_save)(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = value.old.read(run/'state.json'); protocol = value.old.checked(state['protocol'])
        result = value.old.checked(state['results'])
        previous = value.old.checked(protocol['source_qualified_results'])
        comparison = statistics.comparison_statistics(result, previous, protocol)
        primary_gate = statistics.adoption_gate(comparison)
        stability_gate = stability_statistics(result, previous, protocol)
        comparison.update(primary_gate=primary_gate, stability_gate=stability_gate,
            adoption_pass=bool(primary_gate['passed'] and stability_gate['passed']),
            automatic_version_update=False)
        uncertainty = clearance_uncertainty.summarize(result['units'], protocol['uncertainty_decisions'])
        result.update(candidate=CANDIDATE, design_version='learned-exchange-design-2',
            source_qualified_results=protocol['source_qualified_results'],
            foundation_prior_comparison=comparison, input_adjustments=protocol['input_adjustments'],
            source_missingness=previous['missingness'], source_available_over_all_ml=previous['available_over_all_ml'],
            unchanged_parameters=list(FIXED_PARAMETERS), new_training_calls=0,
            uncertainty_decisions=protocol['uncertainty_decisions'], uncertainty_validation=uncertainty,
            new_independent_clinical_samples=0, automatic_version_update=False)
        value.verify(protocol); save(run/'results.json', result)
        state.update(status='complete', completed_at=value.old.now(), results=value.old.ref(run/'results.json'))
        save(run/'state.json', state)
        lines = ['# Foundation clearance prior: unchanged human PBPK validation', '',
            'Only predicted total clearance changes. Independently sourced CL/renal inputs, source-qualified fu, Vss, half-life, MRT and learned-exchange-design-2 remain fixed. No clinical curve fitting or per-drug method selection occurs.', '',
            '| Condition | Source scored / requested | Candidate scored / requested | Source common GMFE | Candidate common GMFE | At least 5% benefit / harm / paired drugs |',
            '|---|---:|---:|---:|---:|---:|']
        def fmt(x): return '—' if x is None else f'{x:.4f}'
        for condition, row in comparison['candidate_over_source_qualified']['all'].items():
            counts = row['meaningful_changes']
            lines.append(f"| {condition} | {row['source_scored']}/{row['requested']} | {row['candidate_scored']}/{row['requested']} | {fmt(row['common_source_curve']['estimate'])} | {fmt(row['common_candidate_curve']['estimate'])} | {counts['benefit']} / {counts['harm']} / {counts['common_scored_drugs']} |")
        lines += ['', 'All 31 requested drugs, five repeated input conditions, eleven outputs and two chemical blockers remain. The original evidence-eligible group is retained rather than narrowed after source qualification. Old twenty and additional eleven drugs are reported separately, not treated as new independent blind validation.',
            'The paired study-cluster bootstrap has 2,000 draws, seed 20261007 and preserved cluster multiplicity. Source-reported endpoints and curve-derived finite-window endpoints remain separate; unavailable references are empty. Every failure stays in coverage and returned-with-error-below-2/3/5 denominators.',
            'Missing-CL errors and prediction drift are reported alongside available-input quality. Lower error on returned cases alone cannot establish useful rejection or clinical safety. Exact cached predictions are not new independent experiments. The paper is unchanged.', '']
        lines += ['CL interval decisions are frozen before scoring and never delete point forecasts. Independently sourced CL has unquantified uncertainty, not zero risk; the two construction blockers are counted separately. CL-only AUC∞/CL range decisions do not assess the other nine output uncertainties. Risk–coverage and high-error AUROC are descriptive clinical diagnostics, not calibrated clinical risk.', '']
        lines += [f"Prespecified development adoption gate passed: {comparison['adoption_pass']}. Primary coverage/accuracy/net-benefit gate: {primary_gate['passed']}; fixed-group missing-CL stability gate: {stability_gate['passed']}. This does not automatically change the selected model or paper.", '']
        (run/'README.md').write_text('\n'.join(lines))
    return {**info, 'candidate': CANDIDATE, 'new_training_calls': 0}


class Monitor(retained.Monitor):
    """The batch budget includes the supplied prior as well as clinical work."""
    def check(self, force=False):
        super().check(force)
        tick = time.monotonic()
        if force or tick-getattr(self, 'last_batch_size_check', 0.) >= 60:
            total = 0
            for path in BATCH.rglob('*'):
                if path.is_symlink():
                    raise value.old.Pause('Unexpected symlink in the bounded foundation batch')
                try:
                    if path.is_file(): total += path.stat().st_size
                except FileNotFoundError:
                    pass  # An atomic save may replace its temporary path.
            if total > self.protocol['directory_limit_bytes']:
                self.halted.set()
                raise value.old.Pause('Foundation batch exceeds 10 GiB; stop new submissions')
            self.last_batch_size_check = tick
            self.update(batch_bytes=total)


def execute(run=DEFAULT):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        return clone(retained.execute, own=own, solve=solve, Monitor=Monitor, report=report)(own(run))


def status(run=DEFAULT): return clone(base.status, own=own)(own(run))
def cleanup(run=DEFAULT): return clone(base.cleanup, own=own)(own(run))


def background(run=DEFAULT):
    run = own(run)
    # A second process may start, but the project run.lock forbids duplicate work.
    with (run/'run.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    save(run/'pid.json', {'pid': child.pid, 'project': str(ROOT), 'created_at': value.old.now()})
    return {'status': 'started', 'pid': child.pid, 'log': str(run/'run.log')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'cleanup'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--workers', type=int, default=14); parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': output = prepare(args.run_dir, args.prior, args.workers)
    elif args.command == 'run': output = background(args.run_dir) if args.background else execute(args.run_dir)
    else: output = globals()[args.command](args.run_dir)
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
