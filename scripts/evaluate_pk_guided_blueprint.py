#!/usr/bin/env python3
"""Small historical evaluation of the implemented source + PK-check algorithm.

No new parameter optimizer is introduced here. An unchanged snapshot shares
one real native prediction between arms; those aliases are reported explicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions
from pbpk_kb.blueprints.pk_constraints import check_source_pk
from pbpk_kb.blueprints.source_adaptation import load_source_model, adapt_source_model
from pbpk_kb.blueprints.source_context import apply_source_context
from pbpk_kb.blueprints.source_output_grid import ensure_output_times
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from evaluate_source_context import native_job
from source_context_cases import build_case_manifest, load_observations, _new_case, _source_pi_membership
from watch_blueprint_suite import ProcessScope

ROOT = Path(__file__).resolve().parents[1]
PK_INPUTS = ROOT / 'experiments/blueprint_design/pk_constraint_development_20260921/inputs'
CASE_IDS = (
    'midazolam_iv_source', 'midazolam_iv_microdose',
    'midazolam_source_individual_infusion', 'midazolam_oral_formulation',
    'midazolam_fed_source', 'rifampicin_iv_source',
    'rifampicin_iv_dose_infusion', 'rifampicin_repeated_protocol_individual',
    'rifampicin_eval_baneyx_repeat', 'rifampicin_eval_chattopadhyay_repeat',
)
PK_CASES = {'midazolam_iv_source': ('midazolam', 1.0),
            'midazolam_iv_microdose': ('midazolam', .001),
            'rifampicin_iv_source': ('rifampicin', 300.0),
            'rifampicin_iv_600mg_30min_pk': ('rifampicin', 600.0)}
ARMS = ('source', 'context', 'pk_guided')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def cases_for_run():
    manifest = build_case_manifest(ROOT)
    cases = [deepcopy(next(c for c in manifest['cases'] if c['case_id'] == name)) for name in CASE_IDS]
    # Add the label's 0.5 h infusion rather than confusing it with Nitti's 3 h
    # infusion. The exact official 600 mg observation mapping is checked first.
    extra = deepcopy(next(c for c in cases if c['case_id'] == 'rifampicin_iv_source'))
    extra.update(case_id='rifampicin_iv_600mg_30min_pk', request={'dose_scale': 2.0},
                 observation_refs=[], study_ids=[], required_times_min=[], source_fitting_audit={})
    snapshot = json.loads((ROOT / extra['source']['path']).read_text())
    official = _new_case((extra['case_id'], 'Rifampicin', 'diagnostic',
        'Rifampicin iv 600 mg (0.5 h)',
        ('sanofi-aventis U.S. LLC. 2013 - 600 mg - Rifampicin - IV - 600 mg - Plasma - agg. (n=12)',)),
        {'Rifampicin': snapshot})
    extra['observation_refs'] = official['observation_refs']
    for ref in extra['observation_refs']:
        observed = next(o for o in snapshot['ObservedData'] if o['Name'] == ref['observation_name'])
        ref['required_times_min'] = [v * 60 for v in observed['BaseGrid']['Values']]
        extra['source_fitting_audit'][ref['observation_name']] = _source_pi_membership(snapshot, ref['observation_name'])
    extra['required_times_min'] = sorted({t for r in extra['observation_refs'] for t in r['required_times_min']})
    extra['study_ids'] = sorted({r['study_id'] for r in extra['observation_refs']})
    expected = extra['native_expectations']['adapted']['expectations']
    for item in expected:
        if item['path'].endswith('|Dose'):
            item['value'] = 600.0
    extra['engineering_reference'] = official['engineering_reference']
    extra['native_expectations']['official_reference'] = official['native_expectations']['official_reference']
    cases.append(extra)
    # Retain only already accepted human-context edits. FeSSIF was a rejected
    # sensitivity hypothesis, not the algorithm's selected default behavior.
    for case in cases:
        case['selected_context_request'] = {
            k: v for k, v in case['context_request'].items() if k == 'human_sex'}
    return manifest, cases


def run_pk(case, snapshot, result, output):
    begin = time.monotonic()
    directory = output / case['case_id'] / 'pk_check'
    name, dose = PK_CASES[case['case_id']]
    context = json.loads((PK_INPUTS / (name + '_context.json')).read_text())
    context['dose_mg'] = dose
    evidence = json.loads((PK_INPUTS / 'measured_pk_evidence.json').read_text())
    predictor = json.loads((PK_INPUTS / (name + '_predictor_input.json')).read_text())
    before = digest(snapshot)
    manifest_path = inside(result['execution_manifest'], ROOT, exists=True)
    manifest = json.loads(manifest_path.read_text())
    bindings = [manifest_path]
    for key in ('pkml_outputs', 'curve_outputs'):
        bindings.extend(inside(item['path'], manifest_path.parent, exists=True) for item in manifest[key])
    old_hashes = {str(p): sha256(p) for p in bindings}
    report = check_source_pk(snapshot=snapshot, execution_manifest=manifest_path,
        evidence=evidence, target_context=context, predictor_input=predictor,
        target_output=case['target_output'], output_dir=directory, project_root=ROOT)
    unchanged = before == digest(snapshot) and all(sha256(Path(p)) == h for p, h in old_hashes.items())
    if not unchanged or report['parameters_modified']:
        raise ValueError('A diagnostic unexpectedly modified the native model or prediction')
    # Only matching drug, dose, infusion and PK definition enter this endpoint
    # table. All other rows remain in the raw diagnostic report, never pooled.
    references = {r['record_id']: r for r in evidence['records']}
    endpoints = []
    for row in report['comparisons']:
        ref = references.get(row['record_id'])
        if ref is None or ref['source_kind'] != 'measured' or row['status'] != 'compared':
            continue
        if any(ref['context'].get(k) != context.get(k) for k in
               ('route', 'dosing', 'dose_mg', 'infusion_minutes')):
            continue
        endpoints.append(deepcopy(row))
    return {'case_id': case['case_id'], 'status': report['status'],
            'report': str((directory / 'pk_comparison.json').relative_to(ROOT)),
            'matching_measured_endpoints': endpoints,
            'automatic_constraint_count': report['constraint_comparison_count'],
            'snapshot_and_prediction_unchanged': unchanged,
            'bound_artifacts': old_hashes, 'snapshot_sha256': before,
            'elapsed_seconds': round(time.monotonic() - begin, 3)}


def evaluate(cases, results, output):
    # Save all predictions before loading concentration values for scoring.
    predictions = {'locked_at': now(), 'jobs': {'/'.join(k): v for k, v in results.items()}}
    save(output / 'predictions_locked.json', predictions)
    observations = [dict(row, case_id=case['case_id']) for case in cases
                    for row in load_observations(case, ROOT, role=case['role'], predictions_locked=True)]
    save(output / 'observations.json', observations)
    evaluations = {}
    for arm in ARMS:
        evaluations[arm] = {}
        # Keep troughs separate from real concentration-time profiles.
        for sampling in sorted({r['sampling_type'] for r in observations}):
            rows = [r for r in observations if r['sampling_type'] == sampling]
            curves = {r['scenario_id']: results[(r['case_id'], arm)]['curve'] for r in rows}
            lock = {'method': arm, 'seed': 0, 'candidate_id': arm, 'status': 'completed',
                    'predictions_sha256': lock_predictions(curves)}
            evaluations[arm][sampling] = evaluate_locked_predictions(rows, curves, lock, device='cpu')
    save(output / 'evaluations.json', evaluations)
    scores = {arm: {r['scenario_id']: r for e in groups.values() for r in e['rows']}
              for arm, groups in evaluations.items()}
    paired = []
    for observation in observations:
        sid = observation['scenario_id']
        row = {'case_id': observation['case_id'], 'scenario_id': sid,
               'observation': observation['source_observation_name'],
               'compound_id': observation['compound_id'],
               'study_group_id': observation['study_group_id'],
               'sampling_type': observation['sampling_type'], 'arms': {}}
        for arm in ARMS:
            s = scores[arm][sid]
            row['arms'][arm] = {k: s[k] for k in ('status', 'log_rmse', 'gmfe')}
            for endpoint in ('observed_window_auc', 'sampled_cmax'):
                value = s['endpoints'][endpoint]
                row['arms'][arm][endpoint + '_fold_error'] = (
                    math.exp(value['absolute_log_error']) if value is not None else None)
        paired.append(row)
    save(output / 'paired_results.json', paired)
    return observations, evaluations, paired


def run(args):
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 bounded native workers')
    output = inside(args.output, ROOT)
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    state = {'status': 'running', 'stage': 'freeze', 'started_at': now(),
             'completed_native_jobs': 0, 'planned_native_jobs': 0, 'completed_pk_checks': 0}
    save(output / 'runner.json', {**ProcessScope().state(os.getpid()), 'started_at': now()})
    save(ROOT / 'runtime/deployment/source_adaptation_pilot.json',
         {'output': str(output.relative_to(ROOT)), 'registered_at': now()})

    def update(**values):
        state.update(values, elapsed_seconds=round(time.monotonic() - begin, 2))
        save(output / 'progress.json', state)

    update()
    try:
        manifest, cases = cases_for_run()
        code = [Path(__file__), ROOT / 'scripts/evaluate_source_context.py',
                ROOT / 'scripts/source_context_cases.py', ROOT / 'scripts/source_adaptation_cases.py']
        code += [ROOT / 'src/pbpk_kb/blueprints' / (name + '.py') for name in (
            'source_adaptation', 'source_context', 'source_output_grid', 'source_adaptation_audit',
            'design_runtime', 'source_workflow', 'pk_constraints', 'source_pk_profile',
            'pk_predictor', 'heldout_evaluation', 'design_scoring')]
        code += [ROOT / 'src/pbpk_kb/model_runtime/R' / (name + '.R') for name in (
            'blueprint_snapshot_run', 'source_adaptation_audit', 'source_pk_profile')]
        protocol = {'frozen_at': now(), 'cases': cases,
            'case_selection': 'All eight prior adaptation cases plus both additional repeat-dose studies; extra 600 mg/30 min IV endpoint case.',
            'clinical_tier': 'historical_development_not_independent_validation',
            'prior_outcomes_seen': True, 'source_training_overlap': True,
            'arms': {'source': 'Complete source + requested protocol + output grid',
                     'context': 'Source + previously accepted human sex context only; no new fitting',
                     'pk_guided': 'Context + implemented PK diagnostics; unchanged predictions explicitly aliased'},
            'prediction_reuse': 'Only identical snapshots and diagnostic-only arms share a freshly executed curve; no previous run curve is reused.',
            'max_native_jobs': 12, 'workers': args.workers, 'cpu_threads_per_job': 2,
            'pk_case_ids': list(PK_CASES), 'max_pk_checks': 4,
            'parameter_optimization': False, 'composite_score': False,
            'endpoint_policy': 'Observed-window AUC and sampled Cmax; neither applies to longitudinal trough sequences.',
            'code': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in code],
            'pk_inputs': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)}
                          for p in sorted(PK_INPUTS.iterdir()) if p.is_file()]}
        save(output / 'protocol.json', protocol)
        prepared, jobs, aliases, results = {}, [], {}, {}
        source_spec = manifest['rifampicin_pint_spec']
        spec = {'path': 'Rifampicin|Specific intestinal permeability (transcellular)',
                'unit': source_spec['unit']}
        for case in cases:
            ref = case['source']
            source = load_source_model(ROOT / ref['path'], ref['sha256'], ref['simulation_name'], ROOT)
            blueprint = adapt_source_model(source, {**case['request'], 'simulation_name': case['case_id']}, ROOT)
            if blueprint['status'] != 'adapted':
                raise ValueError('Source request failed: ' + case['case_id'])
            context = apply_source_context(blueprint, case['selected_context_request'], ROOT)
            if context['status'] != 'adapted':
                raise ValueError('Context request failed: ' + case['case_id'])
            base = ensure_output_times(blueprint['snapshot'], case['required_times_min'], ROOT, max_points=8192)
            current = ensure_output_times(context['snapshot'], case['required_times_min'], ROOT, max_points=8192)
            prepared[case['case_id']] = {'source': base, 'context': current,
                                        'context_changes': context.get('changes', [])}
            jobs.append((case, 'source', base))
            if digest(base['snapshot']) == digest(current['snapshot']):
                aliases[(case['case_id'], 'context')] = (case['case_id'], 'source')
            else:
                jobs.append((case, 'context', current))
        if len(jobs) > protocol['max_native_jobs']:
            raise ValueError('Frozen execution budget exceeded')
        save(output / 'prepared_models.json', prepared)
        update(stage='native_execution', planned_native_jobs=len(jobs))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(native_job, case, arm, grid['snapshot'], grid['manifest'],
                                  output, spec, source_spec['source_value']): (case['case_id'], arm)
                       for case, arm, grid in jobs}
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()
                update(completed_native_jobs=state['completed_native_jobs'] + 1)
                print('/'.join(key), results[key]['status'], flush=True)
        for key, target in aliases.items():
            results[key] = {**results[target], 'arm': key[1], 'shared_prediction': list(target)}
        update(stage='pk_diagnostics', planned_pk_checks=len(PK_CASES))
        checks = []
        # Run two small RF/native-analysis processes in parallel, bounded and
        # independent of the already completed native ODE execution pool.
        with ThreadPoolExecutor(max_workers=min(2, args.workers)) as pool:
            futures = {}
            for case in cases:
                if case['case_id'] not in PK_CASES:
                    continue
                native = results[(case['case_id'], 'context')]
                if native['status'] == 'passed':
                    future = pool.submit(run_pk, case, prepared[case['case_id']]['context']['snapshot'], native, output)
                    futures[future] = case['case_id']
                else:
                    checks.append({'case_id': case['case_id'], 'status': 'native_failed'})
            for future in as_completed(futures):
                try:
                    check = future.result()
                except Exception as error:
                    check = {'case_id': futures[future], 'status': 'failed', 'error': str(error)}
                checks.append(check)
                update(completed_pk_checks=len(checks))
                print(check['case_id'], 'pk_check', check['status'], flush=True)
        save(output / 'pk_checks.json', checks)
        for case in cases:
            target = (case['case_id'], 'context')
            results[(case['case_id'], 'pk_guided')] = {**results[target], 'arm': 'pk_guided',
                'shared_prediction': list(target), 'diagnostic_only_no_parameter_edit': True}
        update(stage='clinical_scoring')
        observations, evaluations, paired = evaluate(cases, results, output)
        changed = sum(r['arms']['source']['gmfe'] != r['arms']['context']['gmfe'] for r in paired)
        added_change = sum(r['arms']['context'] != r['arms']['pk_guided'] for r in paired)
        valid = lambda row: row['arms']['context']['gmfe'] is not None and row['arms']['source']['gmfe'] is not None
        passed = sum(results[(c['case_id'], a)]['status'] == 'passed' for c, a, _ in jobs)
        code_unchanged = all(sha256(ROOT / x['path']) == x['sha256'] for x in protocol['code'])
        summary = {'status': 'completed' if passed == len(jobs) and all(c['status'] == 'completed' for c in checks) else 'completed_with_failures',
            'completed_at': now(), 'case_count': len(cases), 'curve_count': len(observations),
            'endpoint_only_cases': [c['case_id'] for c in cases if not c['observation_refs']],
            'native_jobs': len(jobs), 'native_passed': passed, 'pk_checks': len(checks),
            'pk_checks_passed': sum(c['status'] == 'completed' for c in checks),
            'matched_pk_endpoints': sum(len(c.get('matching_measured_endpoints', [])) for c in checks),
            'automatic_pk_constraints': sum(c.get('automatic_constraint_count', 0) for c in checks),
            'curves_changed_by_existing_context': changed,
            'curves_improved_by_existing_context': sum(valid(r) and r['arms']['context']['gmfe'] < r['arms']['source']['gmfe'] for r in paired),
            'curves_worsened_by_existing_context': sum(valid(r) and r['arms']['context']['gmfe'] > r['arms']['source']['gmfe'] for r in paired),
            'curves_changed_by_new_pk_check': added_change,
            'parameter_optimization_executed': False, 'independent_blind_evaluation': False,
            'prediction_lock_sha256': sha256(output / 'predictions_locked.json'),
            'code_unchanged_during_execution': code_unchanged,
            'elapsed_seconds': round(time.monotonic() - begin, 2),
            'metrics': {a: {g: e['tiers']['reconstruction'] for g, e in groups.items()}
                        for a, groups in evaluations.items()}}
        if not code_unchanged:
            summary['status'] = 'invalid_code_changed'
        save(output / 'summary.json', summary)
        update(status=summary['status'], stage='completed', successful_native_jobs=passed)
        print(json.dumps({k: v for k, v in summary.items() if k != 'metrics'}, ensure_ascii=False, indent=2), flush=True)
        return summary
    except Exception as error:
        update(status='failed', stage='failed', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    raise SystemExit(0 if run(args)['status'] == 'completed' else 2)
