#!/usr/bin/env python3
"""Run a bounded two-drug design pilot, then score already locked predictions.

Every fitting run and external prediction finishes before dedicated held-out
answer files are opened. Dcal/Dsel alone are passed to the algorithm; source
JSON files can contain historical observations but are used only for audited
source fields. These are historical reconstruction studies,
not a new independent benchmark. No LLM or old fitted result is used.
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
import sys
import threading
import time
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pbpk_kb.blueprints.design_diagnostics import BudgetLedger
from pbpk_kb.blueprints.experiment_runtime import NativeExperiment, digest, file_sha, save, validate_native_parallelism
from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions
from pbpk_kb.blueprints.native_design import policy_from_template, run_native_design
from pbpk_kb.blueprints.native_design_policy import build_design_plan
from pbpk_kb.blueprints.design_objectives import validate_objective_data

MIB = 1024 ** 2
INPUTS = {drug: ROOT / 'experiments/blueprint_design/development_inputs/mechanisms_20260920' / drug / 'input_package.json'
          for drug in ('alprazolam', 'midazolam')}
GROUPS = (
    ('source_fixed', 'Rebuilt template with fixed parameters', 'source', 'fixed', 'fit_priority'),
    ('source_calibrated', 'Rebuilt template + one calibrated clearance parameter', 'source', 'estimate', 'fit_priority'),
    ('equal_search', 'Original bounded comparison: equal candidate budgets', 'compare_registered', 'auto', 'equal'),
    ('fit_priority_search', 'Improved bounded comparison: fitting-priority budgets', 'compare_registered', 'auto', 'fit_priority'),
)
PURPOSE_GROUPS = (
    ('source_fixed', 'Source formula + registered fixed parameters', 'source', 'fixed', 'fit_priority', None),
    ('coordinate_control', 'Same candidates: old coordinate search', 'compare_registered', 'auto', 'fit_priority', 'coordinate'),
    ('residual_design', 'Same candidates: new joint residual optimization', 'compare_registered', 'auto', 'fit_priority', None),
)
STUDY_GROUPS = (
    ('source_fixed', 'Source formula + registered fixed parameters', 'source', 'fixed', 'fit_priority', None),
    ('residual_control', 'Original residual-optimization control', 'compare_registered', 'auto', 'fit_priority', None),
    ('screened_control', 'Residual optimization with screened starts', 'compare_registered', 'auto', 'fit_priority', 'screened_residual'),
    ('study_consistent', 'Screened starts + per-study protection', 'compare_registered', 'auto', 'fit_priority', None),
)
STUDY_COMPARISONS = (
    ('study_consistent', 'residual_control', 'Full revision versus original algorithm'),
    ('screened_control', 'residual_control', 'Start screening only'),
    ('study_consistent', 'screened_control', 'Per-study protection only'),
)


def protocol_groups(protocol):
    if protocol == 'legacy_v2':
        return GROUPS
    if protocol == 'purpose_v3':
        return PURPOSE_GROUPS
    if protocol == 'study_v4':
        return STUDY_GROUPS
    raise ValueError('Unknown pilot protocol')


CODE = ['scripts/evaluate_native_design_pilot.py', *[str(p.relative_to(ROOT)) for p in
        sorted((ROOT / 'src/pbpk_kb/blueprints').glob('*.py'))],
        'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
        'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R']


def now():
    return datetime.now(timezone.utc).isoformat()


def inside_file(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Input file must remain inside the project')
    return path


def directory_bytes(path):
    total = 0
    for item in path.rglob('*'):
        if item.is_symlink():
            raise ValueError('Pilot output must not contain symbolic links')
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def load_bundle(path):
    # Do not call the historical load_case_bundle: no external answer file is
    # opened, even for hashing, until the common prediction barrier is durable.
    bundle = json.loads(path.read_text())
    case, external = bundle['case'], bundle['external_scenarios']
    development = {s['study_group_id'] for s in case['scenarios']}
    if case.get('tier') != 'reconstruction' or case.get('strict_main_eligible') is not False:
        raise ValueError('This pilot requires an explicit historical reconstruction tier')
    if not external or len({s['scenario_id'] for s in external}) != len(external):
        raise ValueError('External scenario identities must be unique and nonempty')
    for scenario in external:
        if ('observations' in scenario or scenario['role'] != 'external'
                or scenario['study_group_id'] in development
                or scenario['compound_id'] != case['compound_id']):
            raise ValueError('External design must be outcome-free and separate from development')
    return bundle


def template_request(case, group, *, protocol="legacy_v2"):
    scenario = next(s for s in case['scenarios'] if s['role'] == 'calibration')
    regimen = scenario['design_request']['instance']['protocol']
    parameters = {p['Name']: p for p in regimen['Parameters']}
    duration = parameters.get('Infusion time', {'Value': 0., 'Unit': 'min'})
    if regimen['ApplicationType'] not in {'Intravenous', 'IntravenousBolus'}:
        raise ValueError('The registered pilot template must be an IV scenario')
    clearance = 'shared_process_parameter' if case['compound_id'] == 'pubchem:2118' else 'single_process_kcat'
    if {p['parameter_id'] for p in case['parameter_search']['parameters']} != {clearance, 'lipophilicity'}:
        raise ValueError('Pilot parameter identities differ from the registered two-coordinate case')
    request = {'version': 'pbpk-template-request-v1', 'template_scenario_id': scenario['scenario_id'],
        'route': 'intravenous', 'dose': parameters['InputDose']['Value'],
        'dose_unit': parameters['InputDose']['Unit'],
        'infusion_minutes': duration['Value'] * (60 if duration['Unit'] == 'h' else 1),
        'max_free_parameters': 0 if group[3] == 'fixed' else 1,
        'parameter_modes': {clearance: group[3], 'lipophilicity': 'fixed'},
        'partition_mode': group[2], 'kinetics_mode': 'retain_source'}
    if protocol in {'purpose_v3', 'study_v4'}:
        algorithm = 'study_consistent' if protocol == 'study_v4' and group[0] == 'study_consistent' else 'purpose_driven'
        request.update(algorithm=algorithm, objective={'primary': 'curve'},
            max_free_parameters=0 if group[3] == 'fixed' else 2,
            parameter_modes={clearance: group[3], 'lipophilicity': group[3]})
    elif protocol != 'legacy_v2':
        raise ValueError('Unknown pilot protocol')
    return request


def verify_matched_plans(plans):
    """Reject any differing scientific contract before a matched comparison."""
    control, revised = plans['coordinate_control'], plans['residual_design']
    for key in ('algorithm', 'objective', 'candidates', 'comparison_parameter_specs',
                'comparison_request', 'source_structure_audit', 'baseline_definitions'):
        if key not in control or key not in revised or control[key] != revised[key]:
            raise ValueError('Matched optimizer plans differ: ' + key)
    if len(control['candidates']) != 8:
        raise ValueError('The registered joint-parameter pilot requires eight matched candidates')
    return {'matched': True, 'candidate_count': 8,
            'candidates_sha256': digest(control['candidates']),
            'parameter_specs_sha256': digest(control['comparison_parameter_specs']),
            'request_sha256': digest(control['comparison_request']),
            'only_intervention': 'coordinate multistart optimizer versus source-start residual optimizer',
            'shared': ['development case', 'parameter bounds and free subsets', 'source formula audit',
                       'logical budget and allocation', 'local identifiability gate', 'Dsel objective and selector'],
            'initialization_note': 'Both include registered values; coordinate additionally uses 0.1/0.9 starts.'}


def verify_study_matched_plans(plans):
    """Match complete request/policy contracts with narrowly declared exceptions.

    Per-run request provenance is checked against its frozen source before that
    one field is normalized. Drug evidence, physiology, regimen, parameter and
    objective fields remain exact comparisons. No arbitrary policy key is ignored.
    """
    names = ('residual_control', 'screened_control', 'study_consistent')
    normalized_requests, normalized_policies = {}, {}
    declarations = {group[0]: group for group in STUDY_GROUPS}
    for name in names:
        plan = plans[name]
        expected_algorithm = 'study_consistent' if name == 'study_consistent' else 'purpose_driven'
        request = deepcopy(plan['comparison_request'])
        policy = deepcopy(plan['comparison_policy'])
        if (plan['algorithm'] != expected_algorithm or request.get('algorithm') != expected_algorithm
                or policy.get('algorithm') != expected_algorithm):
            raise ValueError('Matched study plans differ from the declared algorithm: ' + name)
        if ('comparison_fit_strategy' not in plan or plan.get('comparison_fit_strategy') != declarations[name][5]
                or plan.get('comparison_budget_strategy') != 'fit_priority'):
            raise ValueError('Matched study plans differ from the declared optimizer/budget: ' + name)
        if policy.get('objective') != plan['objective']:
            raise ValueError('Matched study policy and plan objectives differ: ' + name)
        source = policy['prediction_request']['instance'].get('protocol_source')
        if source is None or source != plan.get('comparison_request_source'):
            raise ValueError('Matched study request provenance differs from its frozen source: ' + name)
        request['algorithm'] = policy['algorithm'] = 'declared_comparison_algorithm'
        policy['prediction_request']['instance']['protocol_source'] = 'verified_per_run_request_source'
        normalized_requests[name], normalized_policies[name] = request, policy
    control = plans[names[0]]
    scientific_fields = ('objective', 'candidates', 'comparison_parameter_specs',
                         'source_structure_audit', 'baseline_definitions')
    for name in names[1:]:
        for key in scientific_fields:
            if key not in control or key not in plans[name] or control[key] != plans[name][key]:
                raise ValueError('Matched study plans differ: ' + key)
        if normalized_requests[name] != normalized_requests[names[0]]:
            raise ValueError('Matched study requests differ beyond the declared algorithm')
        if normalized_policies[name] != normalized_policies[names[0]]:
            raise ValueError('Matched study policies differ beyond algorithm and verified request provenance')
    if len(control['candidates']) != 8:
        raise ValueError('The registered study-consistency pilot requires eight matched candidates')
    return {'matched': True, 'candidate_count': 8, 'groups': list(names),
        'candidates_sha256': digest(control['candidates']),
        'parameter_specs_sha256': digest(control['comparison_parameter_specs']),
        'normalized_request_sha256': digest(normalized_requests[names[0]]),
        'normalized_policy_sha256': digest(normalized_policies[names[0]]),
        'allowed_differences': ['declared algorithm', 'declared fitter override',
                                'algorithm-defined per-study selection guard', 'verified per-run request provenance'],
        'shared': ['development case', 'candidate structures and free parameter subsets',
                   'parameter bounds and sources', 'complete normalized policy and user request',
                   'PK objective and regimen', 'source formula audit', 'logical budget and allocation'],
        'comparisons': [{'new': new, 'control': old, 'label': label}
                        for new, old, label in STUDY_COMPARISONS]}


def matched_comparisons(summary, protocol):
    """Report each endpoint separately; missing/failed pairs remain explicit."""
    if protocol == 'study_v4':
        pairs = STUDY_COMPARISONS
    elif protocol == 'purpose_v3':
        pairs = (('residual_design', 'coordinate_control', 'Joint residual optimization versus coordinate search'),)
    else:
        return []
    comparisons = []
    for drug in INPUTS:
        rows = {row['group']: row for row in summary if row['drug'] == drug}
        for new_name, control_name, label in pairs:
            new, control = rows[new_name], rows[control_name]
            delta = {key: new[key] - control[key] if new[key] is not None and control[key] is not None else None
                     for key in ('heldout_log_rmse', 'auc_fold_error', 'cmax_fold_error')}
            comparisons.append({'drug': drug, 'new': new_name, 'control': control_name, 'label': label,
                'difference_new_minus_control': delta,
                'both_predictions_complete': control['heldout_complete'] and new['heldout_complete']})
    return comparisons


def protocol_defaults(protocol):
    if protocol not in {'legacy_v2', 'purpose_v3', 'study_v4'}:
        raise ValueError('Unknown pilot protocol')
    return {'max_solver_calls': 1000, 'job_workers': 4, 'scenario_workers': 3} if protocol == 'study_v4' else {
        'max_solver_calls': 500, 'job_workers': 2, 'scenario_workers': 2}


def safe_exp(value):
    if value is None:
        return None
    try:
        result = math.exp(value)
        return result if math.isfinite(result) else None
    except OverflowError:
        return None


def optimizer_activity(search):
    rows = []
    for evaluation in (search or {}).get('evaluations', []):
        result = evaluation['result']
        fit = result.get('fit', {})
        rows.append({'candidate_id': evaluation['candidate_id'], 'status': result['status'],
            'free_parameters': result.get('effective_free_parameter_ids'),
            'fit_strategy': fit.get('fit_strategy'),
            'completed_update_rounds': sum('evaluations' in step for step in fit.get('trace', [])),
            'optimizer_starts': len(fit.get('starts', [])),
            'budget_truncated': fit.get('budget_truncated'),
            'budget_stop_stage': fit.get('budget_stop_stage'),
            'local_rank': result.get('identifiability', {}).get('rank'),
            'purpose_eligible': evaluation.get('purpose_eligible'),
            'exclusion_reasons': evaluation.get('purpose_exclusion_reasons', [])})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--protocol', choices=('legacy_v2', 'purpose_v3', 'study_v4'), default='legacy_v2')
    parser.add_argument('--job-workers', type=int)
    parser.add_argument('--scenario-workers', type=int)
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    parser.add_argument('--max-solver-calls', type=int)
    args = parser.parse_args(argv)
    for name, value in protocol_defaults(args.protocol).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not 50 <= args.max_solver_calls <= 5000:
        parser.error('Use 50..5000 solver calls per independent design run')
    if not 1 <= args.job_workers <= 4 or not 600 <= args.max_wall_seconds <= 14400:
        parser.error('Use 1..4 job workers and 600..14400 seconds')
    groups = protocol_groups(args.protocol)
    import torch
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA scoring requested but unavailable in the project environment')
    validate_native_parallelism(cpu_threads=2, scenario_workers=args.scenario_workers, drug_workers=args.job_workers)
    output = (ROOT / args.output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT) or output.exists():
        parser.error('Choose a new directory inside this project')
    bundles = {drug: load_bundle(path) for drug, path in INPUTS.items()}
    output.mkdir(parents=True, exist_ok=False)
    pilot_started = time.monotonic()
    start_ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    runner = {'pid': os.getpid(), 'start_ticks': start_ticks, 'uid': os.getuid(), 'project_root': str(ROOT),
              'script': CODE[0], 'started_at': now(), 'status': 'running'}
    save(output / 'runner.json', runner)
    (output / 'runner.pid').write_text(str(os.getpid()) + '\n')
    code_sha = {path: file_sha(ROOT / path) for path in CODE}
    save(output / 'protocol.json', {'purpose': 'small_historical_native_design_pilot',
        'created_at': now(), 'protocol': args.protocol, 'drugs': list(INPUTS), 'groups': [list(g) for g in groups],
        'max_solver_calls_per_design': args.max_solver_calls, 'device': args.device,
        'max_wall_seconds': args.max_wall_seconds, 'native_call_timeout_seconds': 180,
        'deadline_policy': 'Checked between bounded native calls; an in-flight call may finish within its 180-second timeout.',
        'fit_start_strategy': {g[0]: ('screened_source_and_registered_grid' if args.protocol == 'study_v4'
            and g[0] in {'screened_control', 'study_consistent'} else 'legacy' if g[4] == 'equal' else 'source_anchor')
            for g in groups},
        'design_output_limit_bytes': 640 * MIB, 'external_output_limit_bytes': 96 * MIB,
        'group_output_limit_bytes': 768 * MIB, 'total_output_limit_bytes': 4096 * MIB,
        'job_workers': args.job_workers, 'native_cpu_threads': 2, 'scenario_workers': args.scenario_workers,
        'fitter_override': {g[0]: g[5] if args.protocol in {'purpose_v3', 'study_v4'} else None for g in groups},
        'previously_seen_holdouts': True,
        'fit_results_shared_between_groups': False, 'external_observations_available_to_designer': False,
        'external_predictions_per_group': {d: len(b['external_scenarios']) for d, b in bundles.items()},
        'input_sha256': {d: file_sha(p) for d, p in INPUTS.items()}, 'code_sha256': code_sha,
        'independent_main_benchmark': False, 'unbiased_cross_validation': False,
        'historical_parameters_may_have_been_fitted': True})
    mutex, stopped = threading.Lock(), threading.Event()
    state = {'status': 'running', 'stage': 'design_and_blind_prediction', 'started_at': now(),
             'planned_designs': len(groups) * len(bundles), 'completed_designs': 0, 'locked_submissions': 0,
             'planned_predictions': len(groups) * sum(len(b['external_scenarios']) for b in bundles.values()), 'completed_predictions': 0, 'successful_predictions': 0,
             'elapsed_seconds': 0., 'external_observations_read': False,
             'jobs': {d + '/' + g[0]: {'drug': d, 'method': g[0], 'status': 'pending', 'stage': 'pending',
                 'design_progress_path': str((output / d / g[0] / 'design/progress.json').relative_to(ROOT))}
                      for d in INPUTS for g in groups}}
    state['tasks'] = list(state['jobs'].values())
    save(output / 'progress.json', state)

    def progress(key=None, **fields):
        with mutex:
            if key is None:
                state.update(fields)
            else:
                state['jobs'].setdefault(key, {}).update(fields)
            state['updated_at'] = now()
            state['elapsed_seconds'] = time.monotonic() - pilot_started
            state['tasks'] = [deepcopy(state['jobs'][k]) for k in sorted(state['jobs'])]
            save(output / 'progress.json', state, replace=True)

    def guard(group_dir=None, *, native=True):
        if native and stopped.is_set():
            raise RuntimeError('Pilot stopped after a bounded-resource or source-integrity failure')
        if native and time.monotonic() - pilot_started >= args.max_wall_seconds:
            stopped.set()
            raise RuntimeError('Pilot reached its registered wall-clock limit; no further native execution')
        if directory_bytes(output) > 4096 * MIB or group_dir and directory_bytes(group_dir) > 768 * MIB:
            stopped.set()
            raise RuntimeError('Pilot output limit exceeded; no further native execution')
        if any(file_sha(ROOT / path) != sha for path, sha in code_sha.items()):
            stopped.set()
            raise RuntimeError('Scientific source changed during the pilot; stop before further execution')

    prepared = {}

    def run_drug(drug, selected_groups):
        bundle = bundles[drug]
        case, external = bundle['case'], bundle['external_scenarios']
        records = []
        for group in selected_groups:
            key = drug + '/' + group[0]
            directory = output / drug / group[0]
            started = time.monotonic()
            result, selection, failure, search = None, None, None, None
            prediction_errors = []
            curves = {s['scenario_id']: None for s in external}
            external_ledger = BudgetLedger(max(100, 4 * len(external)), diagnostic_fraction=0)
            progress(key, status='running', stage='design', strategy=group[4])
            try:
                guard(directory)
                policy = prepared[key]
                result = run_native_design(case=deepcopy(case), policy=policy, output=directory / 'design',
                    project_root=ROOT, execute=True, max_solver_calls=args.max_solver_calls,
                    device=args.device, cpu_threads=2, scenario_workers=args.scenario_workers,
                    max_output_bytes=640 * MIB, budget_strategy=group[4], resource_guard=lambda: guard(directory),
                    evaluation_fit_strategy=group[5] if args.protocol in {'purpose_v3', 'study_v4'} else None)
                if (directory / 'design/search_result.json').is_file():
                    search = json.loads((directory / 'design/search_result.json').read_text())
                if result.get('delivery_complete') is True:
                    selection = {'candidate': search['selected_candidate'], 'result': search['selected_result']}
                else:
                    failure = 'Design did not deliver a verified native model: ' + str(result.get('reason', result.get('status')))
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
            design_seconds = time.monotonic() - started
            with mutex:
                state['completed_designs'] += 1
            progress(key, stage='blind_external_prediction', design_status=(result or {}).get('status', 'failed'),
                     design_seconds=design_seconds, failure=failure)
            prediction_started = time.monotonic()
            experiment = None
            if selection is not None:
                try:
                    guard(directory)
                    experiment = NativeExperiment(case=deepcopy(case), output=directory / 'external_native',
                        project_root=ROOT, device=args.device, cpu_threads=2, scenario_workers=args.scenario_workers,
                        max_output_bytes=96 * MIB,
                        progress_callback=lambda **fields: progress(key, native_phase=fields.get('stage'),
                                                                  scenario_id=fields.get('scenario')))
                    context = {'ledger': external_ledger, 'model_requests': set()}
                    for scenario in external:
                        try:
                            guard(directory)
                            curves[scenario['scenario_id']] = experiment._predict(selection['candidate'], scenario,
                                [selection['result']['fitted_values']], context, 'search')[0]
                        except Exception as exc:
                            prediction_errors.append({'scenario_id': scenario['scenario_id'],
                                                      'reason': f'{type(exc).__name__}: {exc}'})
                            if stopped.is_set():
                                break
                except Exception as exc:
                    prediction_errors.append({'scenario_id': None, 'reason': f'{type(exc).__name__}: {exc}'})
                finally:
                    if experiment is not None:
                        experiment.close()
            locked = {'method': group[0], 'seed': 17,
                'candidate_id': selection['candidate']['candidate_id'] if selection else None,
                'predictions_sha256': lock_predictions(curves),
                'status': ('budget_exhausted' if result.get('status') == 'budget_exhausted' else 'completed') if selection else 'failed'}
            record = {'drug': drug, 'group': group[0], 'label': group[1], 'locked_at': now(),
                'locked_selection': locked, 'curves': curves, 'fallback_curves': None,
                'design_status': (result or {}).get('status', 'failed'),
                'design_delivery_complete': bool((result or {}).get('delivery_complete')),
                'search_complete': bool((result or {}).get('search_complete')),
                'selected_candidate_id': selection['candidate']['candidate_id'] if selection else None,
                'algorithm': (result or {}).get('algorithm'),
                'selection_objective': (result or {}).get('selection_objective'),
                'selection_consistency': (result or {}).get('selection_consistency'),
                'optimizer_activity': optimizer_activity(search),
                'fitted_values': selection['result']['fitted_values'] if selection else None,
                'calibration_log_rmse': (result or {}).get('calibration_log_rmse'),
                'selection_log_rmse': (result or {}).get('selection_log_rmse'),
                'design_budget': (result or {}).get('budget'), 'external_budget': external_ledger.snapshot(),
                'design_wall_seconds': design_seconds,
                'external_prediction_wall_seconds': time.monotonic() - prediction_started,
                'wall_seconds': time.monotonic() - started, 'failure': failure,
                'prediction_errors': prediction_errors, 'scientific_validated': False}
            save(directory / 'locked_predictions.json', record)
            records.append(record)
            with mutex:
                state['locked_submissions'] += 1
                state['completed_predictions'] += len(external)
                state['successful_predictions'] += sum(c is not None for c in curves.values())
            progress(key, status='predictions_locked', stage='awaiting_common_scoring_barrier',
                     valid_prediction_count=sum(c is not None for c in curves.values()),
                     expected_prediction_count=len(external), prediction_errors=prediction_errors)
        return records

    records, fatal = [], None
    try:
        matching = {}
        for drug, bundle in bundles.items():
            plans = {}
            for group in groups:
                directory = output / drug / group[0]
                directory.mkdir(parents=True)
                request_path = directory / 'template_request.json'
                save(request_path, template_request(bundle['case'], group, protocol=args.protocol))
                provenance = {'provenance_id': str(uuid5(NAMESPACE_URL, str(request_path))),
                    'provenance_kind': 'user_request', 'source_id': str(request_path.relative_to(ROOT)),
                    'source_checksum': file_sha(request_path), 'source_location': 'Frozen pilot request and constraints'}
                policy = policy_from_template(bundle['case'], json.loads(request_path.read_text()), request_source=provenance)
                plan = build_design_plan(bundle['case'], policy, project_root=ROOT)
                plan['comparison_parameter_specs'] = deepcopy(bundle['case']['parameter_search']['parameters'])
                plan['comparison_request'] = json.loads(request_path.read_text())
                if args.protocol == 'study_v4':
                    plan['comparison_policy'] = policy.model_dump(mode='json')
                    plan['comparison_request_source'] = deepcopy(provenance)
                    plan['comparison_fit_strategy'] = group[5]
                    plan['comparison_budget_strategy'] = group[4]
                if not plan['supported']:
                    raise ValueError('Pilot preflight rejected ' + drug + '/' + group[0] + ': ' + str(plan.get('reasons')))
                if args.protocol in {'purpose_v3', 'study_v4'}:
                    validate_objective_data(bundle['case'], policy.objective)
                save(directory / 'preflight_plan.json', plan)
                prepared[drug + '/' + group[0]] = policy
                plans[group[0]] = plan
            if args.protocol == 'purpose_v3':
                matching[drug] = verify_matched_plans(plans)
            elif args.protocol == 'study_v4':
                matching[drug] = verify_study_matched_plans(plans)
        save(output / 'comparison_contract.json', {'protocol': args.protocol, 'matched_optimizers': matching,
             'primary_objective': 'curve', 'clinical_error_threshold_invented': False})
        with ThreadPoolExecutor(max_workers=args.job_workers) as pool:
            futures = {pool.submit(run_drug, drug, [group]): (drug, group[0])
                       for group in groups for drug in INPUTS}
            for future in as_completed(futures):
                records.extend(future.result())
        if len(records) != len(groups) * len(bundles):
            raise RuntimeError('All registered design attempts and prediction submissions must be retained')
        guard(native=False)
        lock_rows = [{'drug': r['drug'], 'group': r['group'],
                     'path': str((output / r['drug'] / r['group'] / 'locked_predictions.json').relative_to(ROOT)),
                     'sha256': file_sha(output / r['drug'] / r['group'] / 'locked_predictions.json'),
                     'predictions_sha256': r['locked_selection']['predictions_sha256']} for r in records]
        save(output / 'all_predictions_locked.json', {'locked_at': now(), 'submissions': lock_rows,
            'all_design_and_prediction_attempts_finished': True, 'external_observations_read': False})
        progress(stage='external_scoring', external_observations_read=True)
        observations = {}
        for drug, bundle in bundles.items():
            observations[drug] = []
            for scenario in bundle['external_scenarios']:
                ref = scenario['holdout_reference']
                path = inside_file(ref['path'])
                if path.stat().st_size > 8 * MIB or file_sha(path) != ref['sha256']:
                    raise ValueError('External observation source changed after prediction locking')
                row = json.loads(path.read_text())
                for name in ('scenario_id', 'study_group_id', 'compound_id', 'role'):
                    if row.get(name) != scenario[name]:
                        raise ValueError('External observation identity differs from its frozen design')
                if len(row.get('observations', [])) != scenario['observation_count']:
                    raise ValueError('External observation count differs from the frozen sampling design')
                endpoint = {k: scenario[k] for k in ('auc_evaluable', 'cmax_evaluable', 'sampling_qualification')}
                if any(k in row and row[k] != value for k, value in endpoint.items()):
                    raise ValueError('External endpoint qualification differs from the frozen design')
                observations[drug].append({**row, **endpoint, 'tier': 'reconstruction'})
        summary = []
        for record in sorted(records, key=lambda r: (r['drug'], [g[0] for g in groups].index(r['group']))):
            directory = output / record['drug'] / record['group']
            registered = next(r for r in lock_rows if (r['drug'], r['group']) == (record['drug'], record['group']))
            if file_sha(directory / 'locked_predictions.json') != registered['sha256']:
                raise ValueError('Prediction submission changed after the common lock')
            report = evaluate_locked_predictions(observations[record['drug']], record['curves'],
                record['locked_selection'], fallback_curves=None, device=args.device)
            save(directory / 'external_evaluation.json', report)
            tier = report['tiers']['reconstruction']
            endpoints = tier['endpoints']
            complete = tier['analysis_complete']
            row = {k: record[k] for k in ('drug', 'group', 'label', 'design_status', 'design_delivery_complete',
                'search_complete', 'selected_candidate_id', 'algorithm', 'fitted_values', 'selection_objective',
                'selection_consistency', 'optimizer_activity',
                'design_wall_seconds', 'external_prediction_wall_seconds', 'calibration_log_rmse', 'selection_log_rmse', 'design_budget',
                'external_budget', 'wall_seconds', 'failure', 'prediction_errors')}
            row.update(heldout_log_rmse=tier['mean_curve_log_rmse'],
                available_heldout_log_rmse=tier['available_curve_log_rmse'],
                auc_fold_error=safe_exp(endpoints['observed_window_auc']['mean_absolute_log_error']) if complete else None,
                cmax_fold_error=safe_exp(endpoints['sampled_cmax']['mean_absolute_log_error']) if complete else None,
                scored_curves=tier['scored_curve_count'], expected_curves=tier['unique_curve_count'],
                heldout_complete=complete, tier='reconstruction')
            summary.append(row)
            progress(record['drug'] + '/' + record['group'], status='scored' if complete else 'incomplete', stage='finished')
        save(output / 'summary.json', {'completed_at': now(), 'rows': summary,
            'all_submissions_locked_before_external_observation_read': True, 'fallbacks_used': False,
            'protocol': args.protocol, 'independent_main_benchmark': False, 'unbiased_cross_validation': False,
            'scientific_validated': False,
            'endpoint_definitions': {'auc': 'observed-window AUC on the registered sampling grid; not AUC infinity',
                'cmax': 'maximum at registered sample times; not continuous peak',
                'fold_error': 'exp(mean study absolute log(prediction/observation)); one is ideal'},
            'limitations': ['Two known drugs and seven historical held-out studies; no independent compound generalization.',
                'Previously observed historical outcomes are a development regression set, not unbiased cross-validation.',
                'Source drug parameters may have been fitted in the original public models.',
                'Native PBPK simulation uses CPU; requested CUDA is used for numerical scoring.',
                'No scientific acceptance or winning-method claim follows from this pilot.']})
        def cell(value):
            return '—' if value is None else f'{value:.3f}'
        if args.protocol == 'study_v4':
            protocol_description = ('This is development regression on previously inspected historical studies, not unbiased cross-validation. All three search groups share complete requests, candidates, parameters, targets, dosing, and budgets. '
                'Start screening is compared first, then per-study protection separately. The fixed-source group is a baseline only.')
        elif args.protocol == 'purpose_v3':
            protocol_description = ('This is development regression on previously inspected historical studies. Optimizer controls share candidates, parameter bounds, budgets, diagnostics, and selection rules. '
                'Coordinate search also includes in-bound starting points; the residual method jointly updates from registered initial values.')
        else:
            protocol_description = 'This is the historical v1/v2 comparison, with a rebuilt-template fixed baseline.'
        lines = ['# Two-drug blueprint-design pilot evaluation', '',
            f'{len(groups)} methods x {len(bundles)} drugs: {len(groups)*len(bundles)} designs and '
            f'{len(groups)*sum(len(bundle["external_scenarios"]) for bundle in bundles.values())} registered predictions. '
            'Each group is fitted separately. Dedicated held-out answer files are read only after all predictions are locked; failures are retained without fallback results.', '',
            protocol_description, '',
            '| Drug | Method | Fit log-RMSE | Selection log-RMSE | Held-out log-RMSE | AUC fold error | Cmax fold error | Valid curves | Run status |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|']
        for r in summary:
            values = [cell(r[k]) for k in ('calibration_log_rmse', 'selection_log_rmse', 'heldout_log_rmse', 'auc_fold_error', 'cmax_fold_error')]
            lines.append('| ' + ' | '.join([r['drug'], r['label'], *values,
                f"{r['scored_curves']}/{r['expected_curves']}", r['design_status']]) + ' |')
        lines += ['', '| Drug | Method | Design solver calls (logical/physical) | Design time (seconds) | Delivery |',
                  '|---|---|---:|---:|---|']
        for row in summary:
            counts = (row['design_budget'] or {}).get('totals', {})
            lines.append('| ' + ' | '.join([row['drug'], row['label'],
                f"{counts.get('logical_calls', '—')}/{counts.get('physical_calls', '—')}",
                cell(row['design_wall_seconds']), 'Success' if row['design_delivery_complete'] else 'Failure']) + ' |')
        if args.protocol in {'purpose_v3', 'study_v4'}:
            comparisons = matched_comparisons(summary, args.protocol)
            lines += ['', 'Error differences (revised minus control; negative is better) are reported separately without a weighted total:', '',
                      '| Drug | Comparison | Curve log-RMSE difference | AUC fold-error difference | Cmax fold-error difference |',
                      '|---|---|---:|---:|---:|']
            for comparison in comparisons:
                delta = comparison['difference_new_minus_control']
                lines.append('| ' + ' | '.join([comparison['drug'], comparison['label'],
                    *[cell(value) for value in delta.values()]]) + ' |')
            save(output / 'matched_comparison.json', {'comparisons': comparisons,
                'qualification': 'Previously seen historical development studies; no new blinded validation.',
                'unbiased_cross_validation': False,
                'weighted_aggregate_score': None})
        if args.protocol == 'study_v4':
            lines += ['', 'Per-study protection checks each study in the fitting and development-selection sets so mean improvement does not hide deterioration in individual studies. '
                       'Whether it changed selection must be established from per-candidate selection_consistency reports. '
                       'This protection does not read final held-out answers and cannot guarantee improvement on unseen studies.']
        lines += ['', 'Lower error is better; ideal fold error is 1. AUC covers only the observation window and Cmax is the sampled peak.',
            '', 'This is development evaluation on two known drugs and seven historical held-out studies. Original model parameters may already have been fitted; this is not independent new-drug validation.',
            '', 'Parallel execution and native caching affect runtime; logical calls count uniformly toward the budget. Fit rounds, budget truncation, and all candidate diagnostics are in summary.json. Returning a valid model does not establish full convergence.',
            '', 'Detailed budgets, times, failures, and prediction locks are in summary.json and each group directory.']
        (output / 'README.md').write_text('\n'.join(lines) + '\n')
        guard(native=False)
        status = 'completed' if all(r['heldout_complete'] for r in summary) else 'completed_with_failures'
        progress(status=status, stage='finished', completed_at=now())
    except (Exception, KeyboardInterrupt) as exc:
        stopped.set()
        fatal = f'{type(exc).__name__}: {exc}'
        progress(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                 stage='stopped', reason=fatal, completed_at=now())
    finally:
        runner.update(status=state['status'], finished_at=now())
        save(output / 'runner.json', runner, replace=True)
    print(json.dumps({'status': state['status'], 'output': str(output), 'reason': fatal}, ensure_ascii=False))
    return 0 if state['status'] in {'completed', 'completed_with_failures'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
