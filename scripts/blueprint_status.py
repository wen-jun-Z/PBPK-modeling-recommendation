#!/usr/bin/env python3
"""Read bounded project progress only; never inspect sealed scientific scores."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import unicodedata

from watch_blueprint_suite import ProcessScope, inside, read_json

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ('Historical three-drug suite', 'reviewed_development_suite_20260920'),
    ('Two-drug mechanism suite', 'mechanism_suite_20260920'),
    ('Furosemide', 'furosemide_mechanism_suite_20260920_seed_fix_v2'),
    ('Clarithromycin original attempt', 'clarithromycin_source_suite_20260920'),
    ('Clarithromycin window repair', 'clarithromycin_window_fix_suite_20260920'),
    ('Clarithromycin reconstruction repair', 'clarithromycin_rebuild_fix_suite_20260920'),
    ('Rosuvastatin', 'rosuvastatin_source_suite_20260920'),
    ('Voriconazole', 'voriconazole_source_suite_20260920'),
    ('Voriconazole reconstruction repair', 'voriconazole_rebuild_fix_suite_20260920'),
    ('Atorvastatin', 'atorvastatin_source_suite_20260920'),
)
PIPELINES = (
    ('Clarithromycin method follow-up', 'blueprint_followup_clarithromycin_20260920_status.json'),
    ('Clarithromycin repair follow-up', 'blueprint_followup_clarithromycin_window_fix_20260920_status.json'),
    ('Clarithromycin reconstruction-repair follow-up', 'blueprint_followup_clarithromycin_rebuild_fix_20260920_status.json'),
    ('Rosuvastatin method follow-up', 'blueprint_followup_rosuvastatin_20260920_status.json'),
    ('Voriconazole method follow-up', 'blueprint_followup_voriconazole_20260920_status.json'),
    ('Voriconazole post-repair method follow-up', 'blueprint_followup_voriconazole_rebuild_fix_20260920_status.json'),
    ('Furosemide external validation', 'blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json'),
    ('Furosemide external-validation interface repair', 'blueprint_furosemide_external_contract_fix_20260920_status.json'),
    ('Voriconazole external validation', 'blueprint_voriconazole_validation_followup_status.json'),
    ('Voriconazole post-repair external validation', 'blueprint_voriconazole_rebuild_validation_followup_status.json'),
    ('Rosuvastatin external validation', 'blueprint_rosuvastatin_validation_followup_status.json'),
    ('Atorvastatin method follow-up', 'blueprint_followup_atorvastatin_20260920_status.json'),
    ('Atorvastatin external validation', 'blueprint_atorvastatin_validation_followup_status.json'),
    ('Atorvastatin proposal follow-up', 'blueprint_atorvastatin_after_qwen_status.json'),
)
HISTORICAL_ATTEMPTS = {'Clarithromycin original attempt', 'Clarithromycin window repair', 'Voriconazole'}
EXTERNAL_PIPELINES = {
    'Furosemide': 'Furosemide external-validation interface repair',
    'Rosuvastatin': 'Rosuvastatin external validation',
    'Voriconazole reconstruction repair': 'Voriconazole post-repair external validation',
    'Atorvastatin': 'Atorvastatin external validation',
}
ACTIVE_STATES = {'running', 'starting', 'waiting_for_methods', 'waiting_for_qwen', 'waiting_for_previous_suite'}
LABELS = {
    'running': 'Running', 'completed': 'Finished', 'stopped': 'Stopped',
    'completed_with_failures': 'Finished with failures retained',
    'stopped_or_stale': 'Process exited', 'not_started': 'Not started', 'not_registered': 'Not registered',
    'waiting_for_methods': 'Waiting for methods to finish', 'waiting_for_qwen': 'Waiting for Qwen',
    'waiting_for_previous_suite': 'Waiting for the preceding queue', 'predictions': 'Independent prediction running', 'scoring': 'Independent scoring running',
    'handed_off': 'Handed off', 'starting': 'Starting',
}


def load(relative):
    path = inside(relative, root=ROOT)
    return read_json(path) if path.exists() else None


def alive(record, scope):
    if not record or type(record.get('pid')) is not int or 'start_ticks' not in record:
        return None
    return scope.state(record['pid'], record['start_ticks']) is not None


def snapshot():
    scope = ProcessScope()
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'sealed_scores_read': False, 'suites': [], 'pipelines': []}
    source_pilot = load('runtime/deployment/source_adaptation_pilot.json')
    if source_pilot:
        relative = source_pilot['output']
        state = load(relative + '/progress.json') or {}
        runner = load(relative + '/runner.json') or {}
        report['status'] = 'source_adaptation_pilot'
        report['source_adaptation_pilot'] = {
            'path': relative, 'registered_runner_alive': alive(runner, scope), **state}
        auxiliary = load('runtime/deployment/blueprint_auxiliary_jobs.json') or {}
        report['auxiliary_jobs'] = []
        for job in auxiliary.get('jobs', []):
            auxiliary_state = load(job['output'] + '/progress.json') or {'status': 'not_started'}
            report['auxiliary_jobs'].append({'name': job['name'], 'path': job['output'], **auxiliary_state})
        if state.get('status') == 'running' and state.get('started_at'):
            report['source_adaptation_pilot']['elapsed_seconds'] = max(
                0, (datetime.now(timezone.utc) - datetime.fromisoformat(state['started_at'])).total_seconds())
        if state.get('status') == 'running' and report['source_adaptation_pilot']['registered_runner_alive'] is False:
            report['source_adaptation_pilot']['status'] = 'stopped_or_stale'
        return report
    hold = load('runtime/deployment/blueprint_experiment_hold.json')
    if hold and hold.get('development_status') == 'algorithm_redesign':
        report['experiment_hold'] = hold
        report['status'] = 'algorithm_redesign'
        return report
    pilot = load('runtime/deployment/native_design_pilot.json')
    if pilot:
        relative = pilot['output']
        state = load(relative + '/progress.json') or {}
        runner = load(relative + '/runner.json') or {}
        report['native_design_pilot'] = {
            'path': relative, 'registered_runner_alive': alive(runner, scope),
            **{key: state.get(key) for key in ('status', 'stage', 'completed_designs',
                    'planned_designs', 'completed_predictions', 'planned_predictions',
                    'successful_predictions', 'elapsed_seconds', 'reason')},
            'active_designs': []}
        for job in state.get('jobs', {}).values():
            drug, method = job['drug'], job['method']
            progress = load(f'{relative}/{drug}/{method}/design/progress.json') or {}
            if progress.get('status') == 'running':
                report['native_design_pilot']['active_designs'].append({
                    'drug': drug, 'method': method, 'stage': progress.get('stage'),
                    'budget': progress.get('budget', {}).get('totals', {})})
        if state.get('status') == 'running' and report['native_design_pilot']['registered_runner_alive'] is False:
            report['native_design_pilot']['status'] = 'stopped_or_stale'
    hold = load('runtime/deployment/blueprint_experiment_hold.json')
    if hold:
        report['experiment_hold'] = {key: hold.get(key) for key in
            ('status', 'requested_at', 'checked_at', 'reason', 'results_deleted',
             'development_status', 'current_algorithm', 'next_step')}
    for name, directory in SUITES:
        relative = 'experiments/blueprint_design/' + directory
        progress = load(relative + '/progress.json')
        if progress is None:
            report['suites'].append({'name': name, 'status': 'not_started'})
            continue
        row = {'name': name, 'path': relative,
               **{key: progress.get(key) for key in (
                   'status', 'stage', 'planned_method_jobs', 'method_status_counts',
                   'elapsed_seconds', 'halt_reason', 'strict_main_eligible_drugs')}}
        runner = load(relative + '/runner.json')
        row['registered_runner_alive'] = alive(runner, scope)
        row['runner_pid'] = runner.get('pid') if runner else None
        if row['status'] == 'running' and row['registered_runner_alive'] is False:
            row['recorded_status'] = row['status']
            row['status'] = 'stopped_or_stale'
        if row['status'] == 'running':
            queue = load(relative + '/queue.json') or {}
            row['active_method_progress'] = []
            for task in queue.get('tasks', []):
                if task.get('status') != 'running':
                    continue
                task_progress = load(str(Path(task['output']) / 'progress.json')) or {}
                row['active_method_progress'].append({
                    'task_id': task['task_id'],
                    **{key: task_progress.get(key) for key in
                       ('method', 'seed', 'stage', 'completed_jobs', 'planned_jobs')},
                    'budget': task_progress.get('budget', {}).get('totals', {}),
                })
        report['suites'].append(row)
    for name, filename in PIPELINES:
        record = load('runtime/deployment/' + filename)
        if record is None:
            report['pipelines'].append({'name': name, 'status': 'not_registered'})
            continue
        process = record
        if filename == 'blueprint_furosemide_validation_pipeline_status_seed_fix_v2.json':
            registration = load('runtime/deployment/blueprint_furosemide_validation_pipeline_20260920_seed_fix_v2.json')
            if (registration and registration.get('pid') == record.get('pid')
                    and registration.get('status_file') == 'runtime/deployment/' + filename):
                process = registration
        pipeline = {'name': name, 'status_path': 'runtime/deployment/' + filename,
            **{key: record.get(key) for key in ('status', 'stage', 'reason')},
            'registered_process_alive': alive(process, scope),
            **{key: process.get(key) for key in ('pid', 'start_ticks')}}
        if pipeline['status'] in ACTIVE_STATES and pipeline['registered_process_alive'] is False:
            pipeline['recorded_status'] = pipeline['status']
            pipeline['status'] = 'stopped_or_stale'
        report['pipelines'].append(pipeline)
    qwen = load('experiments/blueprint_design/qwen_atorvastatin_source_proposals_20260920/progress.json')
    if qwen:
        registration = load('runtime/deployment/blueprint_qwen_atorvastatin_20260920.json')
        report['qwen_atorvastatin'] = {key: qwen.get(key) for key in ('state', 'completed', 'total')}
        report['qwen_atorvastatin']['registered_timeout_alive'] = alive(registration, scope)
    acquisition = load('experiments/blueprint_design/independent_cohort_20260920/acquisition_summary.json')
    if acquisition:
        coverage = acquisition.get('curation_coverage', {})
        curves = acquisition.get('concentration_curve_coverage', {})
        report['data'] = {
            'candidate_drugs': coverage.get('independent_clinical_candidate_drugs'),
            'protocol_records': coverage.get('prepared_protocols'),
            'target_publications': coverage.get('target_publications'),
            'numerical_curve_drugs': curves.get('unique_compounds'),
            'numerical_curve_studies': curves.get('unique_studies'),
            'candidates_are_not_completed_independent_evaluations': True,
            'target_independent_drugs': 60,
        }
    handoff_paths = [ROOT/'experiments/blueprint_design/external_validation/expanded_cohort_20260920/handoff_manifest.json']
    round4 = ROOT/'experiments/blueprint_design/external_validation/clinical_cohort_round4_20260920'
    handoff_paths.extend(sorted(round4.glob('curated_*/handoff_manifest.json')))
    final_handoff = round4/'final_olanzapine_handoff/handoff_manifest.json'
    if final_handoff.exists():
        handoff_paths.append(final_handoff)
    verified_handoffs = []
    for handoff_file in handoff_paths:
        handoff_path = str(handoff_file.relative_to(ROOT))
        handoff = load(handoff_path)
        if not handoff or handoff.get('status') not in {
                'curation_and_static_verification_complete',
                'curation_and_static_verification_complete_retrieval_round_stopped'}:
            continue
        for key in ('curation_receipt', 'parent_baseline_summary', 'final_static_validation'):
            if key == 'final_static_validation' and key not in handoff:
                continue
            ref = handoff[key]
            path = inside(ref['path'], root=ROOT)
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != ref['sha256']:
                raise ValueError('Clinical cohort handoff resource changed: ' + ref['path'])
        drugs = handoff['drug_names']
        if len(drugs) != len(set(drugs)) or len(drugs) != handoff['candidate_drugs_total']:
            raise ValueError('Candidate drugs must be deduplicated in the verified handoff')
        verified_handoffs.append((handoff['candidate_drugs_total'], handoff_path, handoff))
    if verified_handoffs:
        _, handoff_path, handoff = max(verified_handoffs, key=lambda item: item[0])
        report.setdefault('data', {}).update(
            candidate_drugs=handoff['candidate_drugs_total'],
            protocol_records=handoff['clinical_protocol_metadata_total'],
            target_publications=handoff['unique_target_publications_total'],
            numerical_curve_drugs=handoff['full_curve_drugs_total'],
            latest_verified_handoff=handoff_path,
            candidates_are_not_completed_independent_evaluations=True,
            target_independent_drugs=handoff['target_independent_evaluation_drugs'],
            latest_round_endpoint_qualification_pending=handoff.get('round4_endpoint_qualification_pending_compounds', []),
            background_data_acquisition_running=handoff.get('background_data_acquisition_running'),
        )
    plan = load('experiments/blueprint_design/experiment_plan.json')
    if plan:
        targets = plan['cohort_targets_not_observed_counts']
        report['research_plan'] = {
            'protocol_status': plan['protocol_status'],
            'development_drugs_target': targets['development_drugs'],
            'independent_evaluation_drugs_target': targets['evaluation_drugs'],
            'planned_design_runs_target': plan['planned_mandatory_design_runs_total'],
            'all_planned_experiments_queued': False,
            'requires_later_preparation_and_launch': [
                'Native repeat-dose verification and independent scoring follow-up for the new Clarithromycin study',
                'Source qualification, native adaptation, and complete method comparisons for the other drugs',
                'Full-scale independent primary experiments, purpose/mechanism targets, robustness, and synthetic mechanism challenges',
            ],
            'counts_are_targets_not_registered_task_counts': True,
        }
    return report


def label(value):
    return LABELS.get(value, value or '-')


def duration(seconds):
    if not isinstance(seconds, (int, float)):
        return '-'
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def width(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(text))


def table(rows):
    widths = [max(width(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join(' | '.join(str(value) + ' '*(widths[i]-width(value)) for i, value in enumerate(row)) for row in rows)


def render(report, *, history=False, details=False):
    lines = ['PBPK experiment progress  ' + report['checked_at_utc'], '']
    if report.get('status') == 'source_adaptation_pilot':
        pilot = report['source_adaptation_pilot']
        counts = []
        if 'cases' in pilot and 'adaptations_accepted' in pilot:
            counts.append(f"Model adaptation {pilot['adaptations_accepted']}/{pilot['cases']}")
        counts.append(f"Native execution and readback {pilot.get('completed_native_jobs', 0)}/{pilot.get('planned_native_jobs', 0)}")
        if 'successful_native_jobs' in pilot:
            counts.append(f"Passed {pilot['successful_native_jobs']}")
        if 'planned_pk_checks' in pilot:
            counts.append(f"PK checks completed {pilot.get('completed_pk_checks', 0)}/{pilot['planned_pk_checks']}")
        evidence_mode = pilot.get('mode') == 'evidence_blueprint'
        title = 'Construction from human physiology and drug evidence' if evidence_mode else 'Complete source-model adaptation'
        lines += [f"{title}: {label(pilot.get('status'))}; Stage={pilot.get('stage')}. ",
                  '; '.join(counts) + '. ',
                  f"Elapsed {duration(pilot.get('elapsed_seconds'))}; Process alive={pilot.get('registered_runner_alive')}. ",
                  ((f"This batch covers {pilot.get('compound_count', '-')} drugs and{pilot.get('study_count', '-')} studies; fixed construction rules with extended endpoint evaluation, not independent blinded validation."
                    if pilot.get('cohort_mode') == 'multidrug_endpoints' else
                    'This is a one-drug, two-study development check of evidence-based construction, not independent predictive validation. The old full-scale experiments remain stopped.')
                   if evidence_mode else 'This is a two-drug engineering check and historical-observation regression. The old full-scale experiments remain stopped.'),
                  'Result directory: ' + str(ROOT / pilot['path'])]
        if pilot.get('reason'):
            lines.append('Reason: ' + pilot['reason'])
        for job in report.get('auxiliary_jobs', []):
            lines.append(f"Follow-up pilot: {job['name']}; {label(job.get('status'))}; Stage={job.get('stage', '-')}; "
                         f"Executed {job.get('completed_native_jobs', 0)}/{job.get('planned_native_jobs', '-')}; "
                         f"Elapsed {duration(job.get('elapsed_seconds'))}. ")
        if details:
            lines.extend(f"  {key}: {value.get('status')}" for key, value in pilot.get('jobs', {}).items())
        return '\n'.join(lines)
    if report.get('status') == 'algorithm_redesign':
        lines += ['Current phase: redesign blueprint construction around human physiology and complete drug models.',
                  'Old blueprint experiment results were removed as requested; no new experiments are scheduled.',
                  'Original models, drug evidence, clinical observations, and required construction inputs are retained.',
                  'See blueprint design in docs/recommendation_strategy.md and validation in section 9 of docs/research_plan.md.']
        return '\n'.join(lines)
    pilot = report.get('native_design_pilot')
    if pilot:
        lines.append(f"Current pilot evaluation: {label(pilot.get('status'))}; Stage={pilot.get('stage')}; "
                     f"Designs {pilot.get('completed_designs', 0)}/{pilot.get('planned_designs', 8)}; "
                     f"Prediction attempts {pilot.get('completed_predictions', 0)}/{pilot.get('planned_predictions', 28)}; "
                     f"Succeeded {pilot.get('successful_predictions', 0)}. ")
        for item in pilot['active_designs']:
            lines.append(f"  {item['drug']} / {item['method']}: {item['stage']}; Calls={item['budget'].get('logical_calls', '-')}. ")
        lines.append('The old full-scale queue remains paused; pilot evaluation is registered separately.\n')
    if report.get('experiment_hold', {}).get('status') == 'paused':
        if any(row['status'] in ACTIVE_STATES for row in report['suites'] + report['pipelines']):
            lines.append('A pause record exists, but an active queue was detected; use the live status below.\n')
        elif pilot:
            pass
        elif report['experiment_hold'].get('development_status') == 'code_verified_waiting_for_small_evaluation':
            lines.append('Old experiments are paused. Algorithm implementation and code checks are complete; a small real-data evaluation is next. Old results are retained.\n')
        else:
            lines.append('Experiments are paused pending the requirement- and evidence-guided algorithm. Results are retained; remaining tasks require a separate start.\n')
    pipelines = {row['name']: row for row in report['pipelines']}
    displayed = [row for row in report['suites'] if history or row['name'] not in HISTORICAL_ATTEMPTS]
    rows = [['Queue', 'Status', 'Finished/planned', 'Complete', 'Budget exhausted', 'Other termination', 'Elapsed', 'Independent evaluation/follow-up']]
    for row in displayed:
        counts = row.get('method_status_counts') or {}
        state = label(row['status'])
        if row['name'] == 'Atorvastatin' and row['status'] == 'not_started':
            followers = [pipelines.get(n, {}) for n in ('Atorvastatin method follow-up', 'Atorvastatin proposal follow-up')]
            live = next((p for p in followers if p.get('registered_process_alive') is True), None)
            state = label(live['status']) if live else 'Not started/action needed'
        external = pipelines.get(EXTERNAL_PIPELINES.get(row['name']))
        if external:
            evaluation = label(external.get('stage') if external['status'] == 'running' else external['status'])
            if external['status'] == 'completed':
                evaluation = 'Independent evaluation finished'
        elif row['name'] == 'Clarithromycin reconstruction repair':
            evaluation = 'Independent follow-up not connected'
        else:
            evaluation = 'See saved results'
        rows.append([row['name'], state, f"{sum(counts.values())}/{row.get('planned_method_jobs', 57)}",
                     counts.get('completed', 0), counts.get('budget_exhausted', 0),
                     sum(v for k, v in counts.items() if k not in {'completed', 'budget_exhausted'}),
                     duration(row.get('elapsed_seconds')), evaluation])
    lines.append(table(rows))
    lines.append('Finished includes budget exhaustion and other termination; it does not mean complete search or acceptable predictive quality.')
    if 'qwen_atorvastatin' in report:
        qwen = report['qwen_atorvastatin']
        lines.append(f"Atorvastatin Qwen: {label(qwen['state'])}, {qwen['completed']}/{qwen['total']} proposals.")
    if details or history:
        lines.append('\nProcesses and automatic follow-up:')
        for row in report['pipelines']:
            if not history and row['name'] in {'Clarithromycin method follow-up', 'Clarithromycin repair follow-up', 'Voriconazole method follow-up', 'Voriconazole external validation', 'Furosemide external validation'}:
                continue
            if (not history and row['name'] == 'Atorvastatin proposal follow-up' and row['status'] == 'stopped'
                    and any(s['name'] == 'Atorvastatin' and s['status'] in {'running', 'completed'} for s in displayed)):
                continue
            lines.append(f"  {row['name']}: {label(row['status'])}; PID={row.get('pid', '-')}; Alive={row.get('registered_process_alive', '-')}")
            if row.get('reason'):
                lines.append('    Reason: ' + row['reason'])
            if row.get('status_path'):
                lines.append('    Status file: ' + str(ROOT/row['status_path']))
        for row in displayed:
            if row.get('registered_runner_alive'):
                lines.append(f"  {row['name']} Experiment main process PID={row['runner_pid']}")
            for task in row.get('active_method_progress', []):
                budget = task['budget']
                lines.append(f"  {row['name']} {task['method']}/seed {task['seed']} {task['stage']}: "
                    f"Logical calls={budget.get('logical_calls', '-')}, Physical calls={budget.get('physical_calls', '-')}, "
                    f"Solver failures={budget.get('failed_calls', '-')}")
    data = report.get('data', {})
    lines.append(f"\nSources: {data.get('candidate_drugs', '-')} candidate drugs,{data.get('protocol_records', '-')} protocol records,"
                 f"{data.get('numerical_curve_drugs', '-')} drugs have complete numerical curves; candidate count is not completed evaluation count.")
    if data.get('background_data_acquisition_running') is False:
        lines.append('Source acquisition is complete for this batch; no persistent download process remains.')
    plan = report.get('research_plan')
    if plan:
        if report.get('experiment_hold', {}).get('development_status') == 'code_verified_waiting_for_small_evaluation':
            lines.append('Next: a two- to three-drug pilot comparison of the new algorithm. The historical full-scale goals below remain deferred.')
        lines.append(f"The full plan is not completely queued: {plan['development_drugs_target']} development drugs + "
            f"{plan['independent_evaluation_drugs_target']} independent evaluation drugs,{plan['planned_design_runs_target']:,} planned runs.")
        lines.extend('  Requires later preparation and launch: ' + text for text in plan['requires_later_preparation_and_launch'])
    lines.append('Read-only registered-process monitoring; sealed observations and scores are not read. Closing this display does not stop experiments.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only PBPK experiment status; no LLM calls or experiment process control.')
    parser.add_argument('--json', action='store_true', help='Machine-readable output; one JSON per line in watch mode')
    parser.add_argument('--watch', type=float, nargs='?', const=10, metavar='SECONDS', help='Refresh automatically, every 10 seconds by default; Ctrl+C exits the display only')
    parser.add_argument('--history', action='store_true', help='Include old attempts superseded by repaired versions')
    parser.add_argument('--details', action='store_true', help='Show PID, current method, call counts, and stop reasons')
    parser.add_argument('--count', type=int, metavar='COUNT', help='Exit after the specified number of refreshes')
    args = parser.parse_args(argv)
    if args.watch is not None and not 2 <= args.watch <= 3600:
        parser.error('--watch interval must be between 2 and 3600 seconds')
    if args.count is not None and (args.count < 1 or args.watch is None):
        parser.error('--count must be a positive integer used with --watch')
    try:
        iteration = 0
        while True:
            try:
                report = snapshot()
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                print('Cannot reliably read the current state: ' + str(exc), file=sys.stderr)
                if args.watch is None:
                    return 1
            else:
                if args.watch is not None and not args.json and sys.stdout.isatty():
                    print('\033[2J\033[H', end='')
                if args.json:
                    print(json.dumps(report, ensure_ascii=False, indent=None if args.watch is not None else 2), flush=True)
                else:
                    print(render(report, history=args.history, details=args.details), flush=True)
            iteration += 1
            if args.watch is None or args.count is not None and iteration >= args.count:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        if not args.json:
            print('\nProgress display closed; experiment state unchanged.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
