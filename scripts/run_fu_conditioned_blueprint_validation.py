#!/usr/bin/env python3
"""CL-only evidence conditioning on the frozen source-qualified PBPK inputs.

Only qualified measured/derived fu may adjust a predicted CL. Independent CL,
renal evidence, fu, Vss, half-life, MRT and learned-exchange-design-2 are fixed.
All logical conditions and failures are retained and prediction-locked before
the clinical targets are read. No clinical fitting or predictor training occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_foundation_blueprint_validation as runtime
from pbpk_kb.blueprints.source_workflow import save, sha256

base, value, retained, clone = runtime.base, runtime.value, runtime.retained, runtime.clone
source_runner, statistics, infra = runtime.source_runner, runtime.statistics, runtime.infra
VERSION = 'fu-conditioned-blueprint-validation-1'
PRIOR_VERSION = 'fu-conditioned-clearance-prior-1'
CANDIDATE = 'fu_conditioned_clearance'
MODULE = 'pbpk_kb.blueprint_learning.fu_conditioned_clearance'
SOURCE = source_runner.DEFAULT
BATCH = ROOT/'experiments/blueprint_design/fu_conditioned_clearance_20261007'
DEFAULT = BATCH/'clinical'
PRIOR = BATCH/'prior/predictions.json'
CONDITIONS = base.CONDITIONS


def own(path):
    path = value.inside(path, ROOT)
    if path != DEFAULT:
        raise ValueError('Use only fu_conditioned_clearance_20261007/clinical')
    return path


def validate_prior(prior, original, receipt):
    if prior.get('version') != PRIOR_VERSION or prior.get('result_sha256') != base.training_digest(
            {k:v for k,v in prior.items() if k != 'result_sha256'}):
        raise ValueError('Fu-conditioned prior version or checksum differs')
    if prior.get('source_parameter_prior') != receipt:
        raise ValueError('The frozen current parameter prior must be explicit')
    for name in ('parameters', 'targets', 'units'):
        if prior.get(name) != original[name]:
            raise ValueError('All original 40-target baseline parameters and identities stay fixed: '+name)
    if set(prior.get('per_compound_audit', {})) != set(original['parameters']):
        raise ValueError('Every excluded target needs its inference audit')
    for cid, old in original['per_compound_audit'].items():
        audit = prior['per_compound_audit'][cid]
        if (audit.get('human_training_overlap') is not False
                or audit.get('target_structure_and_scaffold_excluded') is not True
                or any(audit.get(k) != old[k] for k in ('normalized_identity', 'generic_scaffold'))):
            raise ValueError('Target identity/scaffold isolation differs: '+cid)
    for name in ('target_labels_used', 'target_observation_files_opened', 'training_structure_overlap',
                 'training_generic_scaffold_overlap', 'clinical_results_used_for_selection'):
        if prior.get('audit', {}).get(name) is not False:
            raise ValueError('Training-only selection certification missing: '+name)
    selection = prior.get('selection', {})
    if (selection.get('kind') not in {'none', 'conditioned', 'offset'}
            or isinstance(selection.get('beta'), bool) or selection.get('beta') not in (0., .25, .5, 1.)
            or (selection['kind'] == 'none' and selection['beta'] != 0)):
        raise ValueError('A fixed training-selected candidate and beta are required')
    return prior


def checked_prior(path, source):
    receipt = value.old.ref(value.inside(path, ROOT, exists=True)); prior = value.old.checked(receipt)
    original = value.old.checked(source['parameter_prior'])
    validate_prior(prior, original, source['parameter_prior'])
    for name in ('source_refs', 'code_pins'):
        if not prior.get(name): raise ValueError('Missing training dependencies: '+name)
        base.merge_pins({}, prior[name])
    value.old.collect_refs(prior['model_receipt'], {})
    for case in source['cases']:
        identity = base.normalize_structure(case['smiles'])['identity_smiles']
        if identity != prior['per_compound_audit'][case['compound_id']]['normalized_identity']:
            raise ValueError('Clinical compound differs from excluded inference target')
    return prior, receipt


def eligibility(task):
    """Source receipts are checked by prepare; qualification never uses scores."""
    if task.get('construction_blocked'):
        return False, 'Existing chemical construction failure retained'
    request = task['request']; evidence = request['evidence']
    cl = [r for r in evidence if r['parameter'] == 'total_clearance']
    fu = [r for r in evidence if r['parameter'] == 'fraction_unbound']
    if len(cl) != 1 or len(fu) != 1:
        raise ValueError('Exactly one frozen CL and fu record are required')
    if cl[0]['basis'] in {'measured', 'derived'}:
        return False, 'Independent CL and organ clearance remain protected'
    if cl[0]['basis'] != 'predicted': raise ValueError('Unexpected total-CL basis')
    if fu[0]['basis'] == 'predicted':
        return False, 'No qualified independent plasma fu; exact original inputs retained'
    if fu[0]['basis'] not in {'measured', 'derived'}:
        return False, 'An assumed binding value cannot update predicted clearance'
    context, source = fu[0].get('context', {}), fu[0].get('source', {})
    if context.get('general_protein_binding_adopted_for_plasma') is True:
        return False, 'General binding adopted as plasma is not qualifying fu evidence'
    if (fu[0].get('compound_id') != request['compound_id'] or fu[0].get('unit') != 'fraction'
            or context.get('species') != 'human' or context.get('matrix') != 'plasma'
            or context.get('independent_of_target') is False
            or context.get('study_role') in {'target', 'evaluation', 'test', 'validation', 'heldout'}
            or context.get('target_curve_source_used') is True or context.get('target_observations_used') is True):
        raise ValueError('Frozen fu is not qualified non-target human plasma evidence')
    for ref in request.get('forbidden_sources', []):
        if source.get('path') == ref.get('path') or source.get('sha256') == ref.get('sha256'):
            raise ValueError('Clinical target evidence must not drive fu conditioning')
    if any(r['parameter'] in {'renal_clearance', 'hepatic_clearance'} for r in evidence):
        raise ValueError('Predicted CL must not overwrite independent organ clearance')
    val = fu[0].get('value')
    if isinstance(val, bool) or not value.scores.positive(val) or val > 1:
        raise ValueError('Plasma fu must be finite in (0,1]')
    return True, 'Source-qualified measured/derived plasma fu and missing independent CL'


def transformed_task(original, prior, receipt, predict_update=None):
    task = deepcopy(original); eligible, reason = eligibility(original)
    audit = {'eligible': eligible, 'changed': False, 'reason': reason,
        'prior': receipt, 'target_observations_used': False,
        'source_request_sha256': value.model._digest(original['request'])}
    if not eligible: return task, audit
    records = {row['parameter']: row for row in task['request']['evidence']}
    cl, fu = records['total_clearance'], records['fraction_unbound']; cid = task['request']['compound_id']
    baseline = prior['parameters'][cid]
    if cl['unit'] != 'ml/min/kg' or cl['value'] != baseline['total_clearance']:
        raise ValueError('Apply one update to the original baseline CL only')
    predict_update = predict_update or importlib.import_module(MODULE).predict_update
    update = predict_update(ROOT, prior, cid, fu['value'], baseline['fraction_unbound'], evidence_status=fu['basis'])
    factor = update.get('factor'); beta = update.get('beta')
    raw, bounded, applied = (update.get(name) for name in
        ('raw_delta_log10', 'bounded_delta_log10', 'applied_delta_log10'))
    if (isinstance(factor, bool) or not value.scores.positive(factor) or not .5-1e-12 <= factor <= 2+1e-12
            or isinstance(beta, bool) or beta not in (0., .25, .5, 1.)
            or beta != prior['selection']['beta'] or update.get('candidate') != prior['selection']['kind']
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in (raw, bounded, applied))
            or update.get('uses_clinical_observations') is not False
            or update.get('observed_fu') != fu['value'] or update.get('prior_fu') != baseline['fraction_unbound']):
        raise ValueError('Invalid bounded, observation-free fu update receipt')
    cap = math.log10(2.)
    if (not math.isclose(bounded, max(-cap, min(cap, raw)), rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(applied, beta*bounded, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(float(factor), 10**applied, rel_tol=1e-12)):
        raise ValueError('Factor must equal 10**(beta * clip(raw log shift, +/-log10(2)))')
    if (beta == 0 or fu['value'] == baseline['fraction_unbound']) and factor != 1:
        raise ValueError('Zero beta or identical fu must be an exact no-op')
    audit.update(update=deepcopy(update), fu_evidence=deepcopy(fu), before=float(cl['value']),
        after=float(cl['value']*factor), unit='mL/min/kg',
        shift_clipped=not math.isclose(raw, bounded, rel_tol=1e-12, abs_tol=1e-12))
    if factor == 1:
        audit['reason'] = 'Exactly unit update factor; original request, core and ledger preserved'
        return task, audit
    cl.update(value=audit['after'], source={**receipt, 'location': cid+'/fu_conditioned_clearance'},
        prediction_method={'name': 'fu_conditioned_clearance', 'version': PRIOR_VERSION,
            'training_overlap': False, 'applicability': {'human_IV_parameter_model': True,
                'target_structure_and_scaffold_excluded': True}},
        notes='Frozen training-selected fu-conditioned relative CL update; all other parameters retained; no clinical curve fitting')
    task['core']['fu_conditioned_clearance_prior'] = receipt
    audit.update(changed=True, reason='Only predicted CL updated from source-qualified independent plasma fu')
    return task, audit


def prepare(run=DEFAULT, prior_path=PRIOR, workers=14):
    run = own(run)
    if (run/'protocol.json').exists(): raise FileExistsError('Protocol is frozen; resume with run')
    if type(workers) is not int or not 1 <= workers <= 14: raise ValueError('workers must be 1..14')
    state = value.old.read(SOURCE/'state.json'); source = value.old.checked(state['protocol']); value.verify(source)
    if (state['status'] != 'complete' or state.get('all_predictions_locked_before_scoring') is not True
            or set(state['outcomes']) != set(source['tasks'])):
        raise ValueError('Completed source-qualified baseline required')
    prior, receipt = checked_prior(prior_path, source); module = importlib.import_module(MODULE)
    if module.VERSION != PRIOR_VERSION: raise ValueError('Predictor code and artifact versions differ')
    pins, code = deepcopy(source['source_pins']), deepcopy(source['code_pins'])
    for ref in (state['protocol'], state['results'], receipt):
        base.merge_pins(pins, {ref['path']: ref['sha256']})  # Baseline scores remain unread.
    base.merge_pins(pins, prior['source_refs']); base.merge_pins(code, prior['code_pins'])
    value.old.collect_refs(prior['model_receipt'], pins)
    base.merge_pins(code, {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))})
    for dependency in (runtime, base, statistics, infra, source_runner, module,
                       base.execution, base.execution.separate):
        path = Path(dependency.__file__).resolve(); base.merge_pins(code, {str(path.relative_to(ROOT)): sha256(path)})
    run.mkdir(parents=True, exist_ok=True); (run/'inputs').mkdir(exist_ok=True)
    tasks, transformations, audits = {}, {}, {}
    for oldkey, original in source['tasks'].items():
        if (original['core']['adapter_version'] != 'clinical-pbpk-comparison-1'
                or original['core']['design_version'] != 'learned-exchange-design-2'):
            raise ValueError('The learned-exchange PBPK structure must remain unchanged')
        old_receipt = state['outcomes'][oldkey]
        runtime.source_outcome(original, old_receipt, pins)
        task, audit = transformed_task(original, prior, receipt)
        value.old.collect_refs(audit, pins)
        key = retained.key_for(task['request'], task['sidecars'], task['core'])
        task.update(key=key, candidate=CANDIDATE, native_source_ref=None,
            source_prediction_receipt=old_receipt, reuse_value_receipt=old_receipt if not audit['changed'] else None)
        if audit['changed']:
            ledger = run/'inputs'/(key[:20]+'.json')
            save(ledger, {'version': VERSION, 'fu_conditioned_prior': receipt, 'evidence': task['request']['evidence']})
            task['request']['evidence_ledger'] = {**value.old.ref(ledger), 'location': 'Exact fu-conditioned CL construction input'}
            pins[str(ledger.relative_to(ROOT))] = sha256(ledger)
            value.model._check_request(task['request'])
            for rows in task['sidecars'].values(): value.model._check_sidecars(task['request'], rows, ROOT)
            build = value.model.assemble_evidence_blueprint(task['request'], ROOT)
            if build['status'] != 'assembled': raise ValueError(task['case_id']+': '+str(build.get('gaps')))
        elif source_runner.exact_input_digest(original) != source_runner.exact_input_digest(task):
            raise ValueError('Cache reuse requires full input equality')
        if task['sidecars'] != original['sidecars']:
            raise ValueError('The Vss/half-life sidecars may not change')
        if retained.key_for(task['request'], task['sidecars'], task['core']) != key:
            raise ValueError('Ledger unexpectedly changed the scientific fingerprint')
        if key in tasks and source_runner.exact_input_digest(tasks[key]) != source_runner.exact_input_digest(task):
            raise ValueError('Prediction key collision')
        tasks[key] = task; audits[key] = audit; transformations[oldkey] = key
    units, baselines = [], {}
    for unit in source['units']:
        key = transformations[unit['key']]
        units.append({**deepcopy(unit), 'key': key, 'source_key': unit['key'],
            'fu_update_eligible': audits[key]['eligible'], 'fu_update_applied': audits[key]['changed']})
        if unit['condition'] == 'all_ml': baselines[unit['case_id']] = {'fingerprint': key}
    matrix = {(c['case_id'], con) for c in source['cases'] for con in CONDITIONS}
    if len(units) != len(matrix) or {(u['case_id'], u['condition']) for u in units} != matrix:
        raise ValueError('Every source drug and condition must remain')
    keys = {(u['case_id'], u['condition']): u['key'] for u in source['units']}
    groups = {con: sorted({u['case_id'] for u in units if u['condition'] == con and u['fu_update_eligible']}) for con in CONDITIONS}
    applied = {con: sorted({u['case_id'] for u in units if u['condition'] == con and u['fu_update_applied']}) for con in CONDITIONS}
    protocol = deepcopy(source)
    protocol.update(version=VERSION, candidate=CANDIDATE, created_at=value.old.now(), tasks=tasks, units=units,
        baselines=baselines, source_pins=pins, code_pins=code, source_qualified_protocol=state['protocol'],
        source_qualified_results=state['results'], source_parameter_prior=source['parameter_prior'],
        parameter_prior=receipt, parameter_mode='fu_evidence_conditioned_relative_clearance',
        input_adjustments=audits, update_eligible_case_ids=groups, update_applied_case_ids=applied,
        stability_case_ids=[c['case_id'] for c in source['cases'] if keys[c['case_id'], 'available'] != keys[c['case_id'], 'missing_clearance']],
        fu_stability_case_ids=[c['case_id'] for c in source['cases'] if keys[c['case_id'], 'available'] != keys[c['case_id'], 'missing_fu']],
        workers=workers, native_threads=2, native_timeout_seconds=180, monitor_seconds=1200,
        directory_limit_bytes=10*1024**3, directory_budget_scope=str(BATCH.relative_to(ROOT)),
        new_unique_predictions=sum(a['changed'] for a in audits.values()),
        expected_new_native_calls=sum(a['changed'] for a in audits.values()), cache_only=not any(a['changed'] for a in audits.values()),
        new_training_calls=0, new_independent_clinical_samples=0, paper_guard={}, publication_snapshot=value.old.paper_pins(),
        no_clinical_curve_fitting=True, design_version='learned-exchange-design-2',
        adoption_rule={'coverage_all_conditions_not_decreased': True,
            'available_and_missing_clearance_common_gmfe_not_increased': True,
            'fixed_source_actual_missing_cl_drift_and_error_ratio_not_increased': True,
            'actual_fu_update_subgroup_meaningful_benefit_exceeds_harm': True,
            'zero_update_has_no_demonstrated_new_benefit': True, 'automatically_update_selected_version': False},
        limitations=['Previously viewed 31-drug development cohort; no new blinded clinical sample or data-dependent drug exclusion.',
            'Only predicted CL can change. Source-qualified fu drives a frozen bounded relative update; independently measured CL, renal evidence and the physiological model remain fixed.',
            'Source-reported and curve-derived endpoints remain separate; missing references, source qualification limitations and all construction failures remain.',
            'Unknown underlying-study overlap of explicitly plasma binding sources is retained as a limitation, not relabelled as proven independent patient evidence.',
            'Current training excludes all 40 target identities/scaffolds; inherited recipe selection historically used labels of Ethinylestradiol, Montelukast, Paroxetine and Pitavastatin.',
            'Untriggered or unit-factor requests exactly reuse old predictions and do not count as additional evidence of benefit.',
            'Prediction drift and development adoption gates are not clinical safety or calibrated risk estimates; the failed prior CQR policy is not used.'])
    value.verify(protocol, publication=True); save(run/'protocol.json', protocol)
    save(run/'state.json', {'version': VERSION, 'candidate': CANDIDATE, 'status': 'prepared',
        'protocol': value.old.ref(run/'protocol.json'), 'created_at': value.old.now(), 'outcomes': {},
        'all_predictions_locked_before_scoring': False})
    return {'status': 'prepared', 'clinical_cases': len(source['cases']), 'logical_units': len(units),
        'unique_predictions': len(tasks), 'eligible_unique_inputs': sum(a['eligible'] for a in audits.values()),
        'new_native_calls_expected': protocol['expected_new_native_calls'],
        'exact_reused_predictions': len(tasks)-protocol['new_unique_predictions'],
        'eligible_cases_by_condition': {k: len(v) for k,v in groups.items()},
        'applied_cases_by_condition': {k: len(v) for k,v in applied.items()}}


def invoke(task, folder): return clone(runtime.invoke, CANDIDATE=CANDIDATE)(task, folder)
def solve(task, run, monitor): return clone(runtime.solve, invoke=invoke)(task, run, monitor)


def update_statistics(result, previous, protocol):
    groups = {}; byid = {c['case_id']: c for c in protocol['cases']}
    for con in CONDITIONS:
        groups[con] = {}
        for label, field in (('eligible', 'update_eligible_case_ids'), ('actually_updated', 'update_applied_case_ids')):
            cases = [byid[cid] for cid in protocol[field][con]]
            after, before = infra._case_rows(result, con, cases), infra._case_rows(previous, con, cases)
            a = {k: r['curve_metrics'].get('gmfe') for k,r in after.items()}
            b = {k: r['curve_metrics'].get('gmfe') for k,r in before.items()}
            pairs = base.paired(a, b, cases)
            common = {k for k in a if value.scores.positive(a[k]) and value.scores.positive(b[k])}
            groups[con][label] = {'case_ids': [c['case_id'] for c in cases], 'requested': len(cases),
                'paired_curve_ratio': pairs,
                'source_common_gmfe': base.drug_equal({k:b[k] for k in common}, cases),
                'candidate_common_gmfe': base.drug_equal({k:a[k] for k in common}, cases),
                'meaningful_changes': statistics.benefit_statistics(pairs['per_case_ratio'], cases)}
    audits = list(protocol['input_adjustments'].values()); eligible = [a for a in audits if a['eligible']]
    applied = [a for a in audits if a['changed']]
    clipped = sum(a.get('shift_clipped', False) for a in eligible)
    return {'by_condition': groups, 'eligible_unique_requests': len(eligible), 'applied_unique_requests': len(applied),
        'clipped_unique_requests': clipped, 'clipped_fraction_of_eligible': clipped/len(eligible) if eligible else None,
        'unit_factor_exact_reuse_count': sum(a['eligible'] and not a['changed'] for a in audits),
        'groups_frozen_from_inputs_before_scoring': True, 'cached_requests_not_extra_benefit': True}


def adoption_gate(comparison, stability, updates):
    rows = comparison['candidate_over_source_qualified']['all']; checks = {}
    for con, row in rows.items(): checks[con+'_coverage'] = row['candidate_scored'] >= row['source_scored']
    for con in ('available', 'missing_clearance'):
        ratio = rows[con]['common_curve_ratio']['estimate']
        checks[con+'_common_gmfe'] = value.scores.positive(ratio) and ratio <= 1+1e-12
        changes = updates['by_condition'][con]['actually_updated']['meaningful_changes']
        checks[con+'_updated_benefit_exceeds_harm'] = changes['benefit'] > changes['harm']
    checks['nonzero_input_update'] = updates['applied_unique_requests'] > 0
    checks['missing_cl_stability'] = stability['passed']
    return {'passed': all(checks.values()), 'checks': checks, 'automatic_version_update': False,
        'interpretation': 'Development gate on actual update groups; no-op cannot establish new benefit. Clinical qualification is not inferred.'}


def report(run=DEFAULT):
    run = own(run)
    def scoring_save(path, payload):
        if Path(path) == run/'state.json' and payload.get('status') == 'complete': payload = {**payload, 'status': 'scoring'}
        save(path, payload)
    info = clone(base.report, own=own, VERSION=VERSION, save=scoring_save)(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = value.old.read(run/'state.json'); protocol = value.old.checked(state['protocol'])
        result = value.old.checked(state['results']); previous = value.old.checked(protocol['source_qualified_results'])
        comparison = statistics.comparison_statistics(result, previous, protocol)
        stability = runtime.stability_statistics(result, previous, protocol)
        # The same fixed-input grouping rule is evaluated for fu loss as well;
        # endpoint predictions and reference scoring remain the original scorer.
        def fu_rows(data):
            rows = deepcopy(data['units'])
            rows = [r for r in rows if r['condition'] in {'available', 'missing_fu'}]
            for r in rows:
                if r['condition'] == 'missing_fu': r['condition'] = 'missing_clearance'
            return {'units': rows}
        fu_stats = runtime.stability_statistics(fu_rows(result), fu_rows(previous),
            {**protocol, 'stability_case_ids': protocol['fu_stability_case_ids']})
        fu_stats.update(interpretation='Descriptive missing-fu sensitivity on the frozen source input-change group; not a clinical safety gate',
                        masked_condition='missing_fu', case_selection='Frozen source available versus missing-fu input difference')
        for label in ('source', 'candidate'):
            fu_stats[label]['missing_fu_scored'] = fu_stats[label].pop('missing_clearance_scored')
        updates = update_statistics(result, previous, protocol)
        gate = adoption_gate(comparison, stability, updates)
        comparison.update(stability_gate=stability, fu_missingness=fu_stats, update_subgroups=updates,
                          adoption_gate=gate, adoption_pass=gate['passed'], automatic_version_update=False)
        result.update(candidate=CANDIDATE, design_version='learned-exchange-design-2',
            source_qualified_results=protocol['source_qualified_results'],
            fu_conditioned_comparison=comparison, input_adjustments=protocol['input_adjustments'],
            source_missingness=previous['missingness'], source_available_over_all_ml=previous['available_over_all_ml'],
            new_training_calls=0, new_independent_clinical_samples=0, automatic_version_update=False)
        value.verify(protocol); save(run/'results.json', result)
        state.update(status='complete', completed_at=value.old.now(), results=value.old.ref(run/'results.json'))
        save(run/'state.json', state)
        lines = ['# Fu evidence-conditioned clearance on the fixed PBPK blueprint', '',
            'Thirty-one development drugs and five frozen input conditions. Qualified independent plasma fu may update predicted CL by a training-selected bounded factor. Independent CL/renal inputs, other priors and the human solver do not change. No clinical calibration or per-drug score selection occurs.', '',
            '| Condition | Source / candidate scored | Source / candidate common GMFE | Updated drugs | At least 5% benefit / harm among updated |',
            '|---|---:|---:|---:|---:|']
        fmt = lambda v: '—' if v is None else f'{v:.4f}'
        for con, row in comparison['candidate_over_source_qualified']['all'].items():
            subgroup = updates['by_condition'][con]['actually_updated']; changes = subgroup['meaningful_changes']
            lines.append(f"| {con} | {row['source_scored']} / {row['candidate_scored']} of {row['requested']} | {fmt(row['common_source_curve']['estimate'])} / {fmt(row['common_candidate_curve']['estimate'])} | {subgroup['requested']} | {changes['benefit']} / {changes['harm']} |")
        lines += ['', f"Development adoption gate passed: {gate['passed']}. Zero updates provide no new benefit. No selected model or manuscript is changed automatically.",
            f"Bounded-update clipping: {updates['clipped_unique_requests']}/{updates['eligible_unique_requests']} eligible unique requests. Eligibility and actually changed groups are both retained, so a smaller denominator cannot manufacture benefit.",
            'The original twelve-drug evidence-eligible group and six-drug missing-CL group remain. Old twenty versus added eleven, all eleven outputs, missing references, strict <2/<3/<5 thresholds, failures and 2,000-draw study-cluster intervals are in results.json/results.csv.',
            'Missing-fu and missing-CL prediction drift are sensitivity diagnostics, not clinical risk. Untriggered requests are exact cached predictions and do not count as new experiments. The previous foundation-prior CQR policy is not part of this method.', '']
        (run/'README.md').write_text('\n'.join(lines))
    return {**info, 'candidate': CANDIDATE, 'new_training_calls': 0, 'adoption_pass': gate['passed']}


class Monitor(runtime.Monitor):
    def check(self, force=False): return clone(runtime.Monitor.check, BATCH=BATCH)(self, force)


def execute(run=DEFAULT):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        return clone(retained.execute, own=own, solve=solve, Monitor=Monitor, report=report)(own(run))


def status(run=DEFAULT): return clone(base.status, own=own)(own(run))
def cleanup(run=DEFAULT): return clone(base.cleanup, own=own)(own(run))


def background(run=DEFAULT):
    run = own(run)
    with (run/'run.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    save(run/'pid.json', {'pid': child.pid, 'project': str(ROOT), 'created_at': value.old.now()})
    return {'status': 'started', 'pid': child.pid, 'log': str(run/'run.log')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'cleanup'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT); parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--workers', type=int, default=14); parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.run_dir, args.prior, args.workers)
    elif args.command == 'run': result = background(args.run_dir) if args.background else execute(args.run_dir)
    else: result = globals()[args.command](args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
