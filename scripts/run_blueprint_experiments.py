"""Run the actual, bounded multi-method historical development experiment.

No independent paper-test outcomes are read. All supported states are declared
up front. Missing Qwen produces an explicit M2 block, never a renamed baseline.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import threading
import time
import zipfile

from pbpk_kb.blueprints.design_diagnostics import BudgetLedger
from pbpk_kb.blueprints.experiment_cases import make_caffeine_experiment_case, prepare_caffeine_experiment_inputs
from pbpk_kb.blueprints.experiment_runtime import (
    NativeExperiment, candidate_space, digest, file_sha, save, validate_native_parallelism,
)
from pbpk_kb.blueprints.search_methods import run_method_search

ROOT = Path(__file__).resolve().parents[1]


def source_path(relative):
    path=(ROOT/relative).resolve(strict=True)
    if not path.is_relative_to(ROOT) or path==ROOT:
        raise ValueError('Experiment source leaves the project')
    return path


def load_case_bundle(path):
    """Load only calibration/selection values and sealed external study designs."""
    payload=json.loads(source_path(path).read_text())
    case=payload['case']
    external=payload.get('external_scenarios',[])
    development={s['study_group_id'] for s in case['scenarios']}
    if len({s['scenario_id'] for s in external})!=len(external):
        raise ValueError('External scenario IDs must be unique')
    if {s['scenario_id'] for s in external} & {s['scenario_id'] for s in case['scenarios']}:
        raise ValueError('External scenario must not duplicate development data')
    for scenario in external:
        if scenario['study_group_id'] in development or 'observations' in scenario:
            raise ValueError('External outcomes cannot enter calibration or selection')
        ref=scenario['holdout_reference']
        if file_sha(source_path(ref['path']))!=ref['sha256']:
            raise ValueError('Sealed external observation artifact changed')
    for source in payload.get('source_audit',case.get('source_audit',[])):
        if file_sha(source_path(source['path']))!=source['sha256']:
            raise ValueError('Locked case source changed')
    return case,external


def predict_external(experiment, case, external, chosen, output, update, *, jobs=None, fallback_candidate=None):
    """Freeze submissions, including failures, without decoding external outcomes."""
    from pbpk_kb.blueprints.heldout_evaluation import lock_predictions
    records=[]
    update(stage='external_predictions')
    fallback=None
    if fallback_candidate is not None:
        ledger=BudgetLedger(max(100,4*len(external)),diagnostic_fraction=0)
        context={'ledger':ledger,'model_requests':set()}
        initial=[p['initial_value'] for p in case['parameter_search']['parameters']]
        fallback={s['scenario_id']:experiment._predict(fallback_candidate,s,[initial],context,'search')[0]
                  for s in external}
        save(output/'external_fallback_predictions.json',{'candidate_id':fallback_candidate['candidate_id'],
             'parameter_values':initial,'curves':fallback,'budget':ledger.snapshot(),
             'policy':'Source-initialized standard human base; no external-error-based switching'})
    selections={(s['method'],s['seed']):s for s in chosen}
    submissions=jobs if jobs is not None else chosen
    for job in submissions:
        selection=selections.get((job['method'],job['seed']))
        ledger=BudgetLedger(max(100,4*len(external)),diagnostic_fraction=0)
        context={'ledger':ledger,'model_requests':set()}
        curves={s['scenario_id']:None for s in external}
        if selection:
            for scenario in external:
                curves[scenario['scenario_id']]=experiment._predict(selection['candidate'],scenario,
                            [selection['result']['fitted_values']],context,'search')[0]
        locked={'method':job['method'],'seed':job['seed'],
                'candidate_id':selection['candidate']['candidate_id'] if selection else None,
                'predictions_sha256':lock_predictions(curves,fallback),
                'status':('budget_exhausted' if job.get('status')=='budget_exhausted' else 'completed')
                         if selection else ('blocked' if job.get('status')=='blocked' else 'failed')}
        record={'method':job['method'],'seed':job['seed'],'locked_selection':locked,
                'curves':curves,'budget':ledger.snapshot(),
                'fallback_curves':fallback,'submission_status':job.get('status','completed'),
                'blocked_reason':job.get('blocked_reason'),
                'fitted_values':selection['result']['fitted_values'] if selection else None}
        directory=output/'methods'/('%s_seed_%d'%(job['method'],job['seed']))
        save(directory/'external_predictions.json',record)
        records.append(record)
        update(external_predictions_completed=len(records))
    save(output/'external_prediction_locks.json',{
        'locked_at_utc':datetime.now(UTC).isoformat(),'locks':[r['locked_selection'] for r in records],
        'all_method_predictions_finished_before_external_observation_read':True})
    return records


def score_external(case, external, records, output, update, *, device):
    """Score previously locked primary/base submissions; never select a candidate."""
    from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions
    def preserve_or_write(path,payload):
        if path.exists():
            if digest(json.loads(path.read_text()))!=digest(payload):
                raise ValueError('Resumed evaluation differs from a preserved report')
        else:
            save(path,payload,replace=True)
    update(stage='external_scoring')
    observations=[]
    for scenario in external:
        ref=scenario['holdout_reference'];path=source_path(ref['path'])
        if file_sha(path)!=ref['sha256']:
            raise ValueError('External observations changed after prediction freeze')
        row=json.loads(path.read_text())
        if (row.get('scenario_id')!=scenario['scenario_id'] or row.get('study_group_id')!=scenario['study_group_id']
                or row.get('compound_id')!=case['compound_id']):
            raise ValueError('External observation identity mismatch')
        # Eligibility is a frozen study-design property. It is not inferred from
        # an observed error, and adding it never alters the sealed measurements.
        endpoints = {k: scenario[k] for k in ('auc_evaluable','cmax_evaluable','sampling_qualification')
                     if k in scenario}
        if any(k in row and row[k] != v for k,v in endpoints.items()):
            raise ValueError('Sealed observation and frozen endpoint eligibility conflict')
        observations.append({**row,**endpoints,'tier':case.get('tier','development')})
    reports=[]
    for record in records:
        report=evaluate_locked_predictions(observations,record['curves'],record['locked_selection'],
                    fallback_curves=record.get('fallback_curves'),device=device)
        directory=output/'methods'/('%s_seed_%d'%(record['method'],record['seed']))
        preserve_or_write(directory/'external_evaluation.json',report)
        reports.append({'method':record['method'],'seed':record['seed'],'tier':case.get('tier','development'),'report':report})
    preserve_or_write(output/'external_results.json',{'compound_id':case['compound_id'],'evaluations':reports,
                'main_benchmark_completed':False,'scientific_accepted':False})
    return len(reports)


def evaluate_external(experiment, case, external, chosen, output, update):
    records=predict_external(experiment,case,external,chosen,output,update)
    return score_external(case,external,records,output,update,device=experiment.device)


def registered_qwen_proposer(case, output):
    """Use only the already user-authorized project registry, without secrets."""
    import httpx
    from urllib.parse import urlsplit
    registration = ROOT / 'runtime/deployment/qwen_service.json'
    try:
        registered = json.loads(registration.read_text())
        base = registered['base_url'].rstrip('/')
        if not isinstance(registered['served_model'], str):
            raise ValueError('Registered model must be a string')
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        save(output/'qwen_status.json', {'registered_model_available': False,
             'reason': 'project_registration_missing_or_invalid', 'credentials_used': False})
        return None
    url = urlsplit(base)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.port != 18581
            or url.path != '/v1' or url.username or url.password or url.query or url.fragment):
        save(output/'qwen_status.json', {'registered_model_available': False,
             'reason': 'registered_endpoint_changed_requires_authorization_review', 'credentials_used': False})
        return None
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            response = client.get(base + '/models')
            available = response.status_code == 200 and any(m.get('id') == registered['served_model'] for m in response.json().get('data', []))
    except (httpx.HTTPError, ValueError):
        available = False
    identity = {'registry_sha256': file_sha(registration), 'registered_model_available': available,
                'endpoint': base, 'served_model': registered['served_model'], 'credentials_used': False}
    save(output / 'qwen_status.json', identity)
    if not available:
        return None

    def propose(candidates, *, seed):
        prompt = {'task': 'Choose one registered PBPK development candidate; return JSON with candidate_id only.',
                  'restriction': 'No new equations, values or candidate IDs. No final or selection observations are available.',
                  'requirement': case.get('requirement', case['design_request']['requirement']),
                  'parameter_contracts': case['parameter_search']['parameters'],
                  'candidates': [{k: c[k] for k in ('candidate_id', 'structure_id', 'free_parameter_ids', 'structure_edit_count')} for c in candidates]}
        request = {'model': registered['served_model'], 'messages': [
                    {'role': 'system', 'content': 'You propose a legal PBPK configuration, never certify scientific validity. Output a JSON object only.'},
                    {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)}],
                   'temperature': 0.2, 'seed': seed, 'max_tokens': 512,
                   'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False}}
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(base + '/chat/completions', json=request)
            response.raise_for_status()
            returned = response.json()
        content = returned['choices'][0]['message']['content']
        selected = json.loads(content)
        save(output / ('qwen_proposal_%d.json' % seed), {'input': prompt, 'candidate_id': selected.get('candidate_id'),
                    'seed': seed, 'request_configuration': {k: request[k] for k in ('temperature', 'max_tokens', 'seed')},
                    'usage': returned.get('usage'), 'served_model': registered['served_model'], 'raw_content': content})
        return selected
    return propose


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-file',type=Path,help='Source-locked prepared drug bundle; default is the historical Caffeine case')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4,
                        help='Native cores per process (1..32), within the CPU affinity budget')
    parser.add_argument('--scenario-workers', type=int, choices=range(1,9), default=3,
                        help='Concurrent study scenarios (1..8); all methods retain the same budget')
    parser.add_argument('--methods', nargs='+', default=['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'A1', 'A2', 'A3', 'E0'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 43, 101])
    parser.add_argument('--qwen-proposals', type=Path,
                        help='Locally generated, input-bound Qwen proposals; no API or GPU model reload in this runner')
    parser.add_argument('--max-solver-calls', type=int, choices=[500, 2000, 5000], default=5000,
                        help='Prespecified total logical scenario budget, including diagnostic reserve')
    parser.add_argument('--exact-free-parameter-count', type=int,
                        help='Prespecified matched parameter dimension; cannot be chosen from final method results')
    parser.add_argument('--max-output-mib', type=int, default=1024)
    parser.add_argument('--defer-external-scoring',action='store_true',
                        help='Cohort mode: lock all submissions and a common fallback; score only after the entire cohort finishes')
    args = parser.parse_args(argv)
    parallelism=validate_native_parallelism(cpu_threads=args.cpu_threads,scenario_workers=args.scenario_workers)
    if not 128 <= args.max_output_mib <= 4096:
        raise ValueError('Use 128..4096 MiB bounded output')
    allowed = {'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'A1', 'A2', 'A3', 'E0'}
    if not set(args.methods) <= allowed or len(args.methods) != len(set(args.methods)):
        raise ValueError('Provide unique registered methods')
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or not set(args.seeds) <= {17, 43, 101}:
        raise ValueError('Choose unique registered development seeds 17/43/101')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
        raise ValueError('Choose a new project-local output directory; existing experiments are preserved')
    if args.case_file:
        case,external=load_case_bundle(args.case_file)
    else:
        prepare_caffeine_experiment_inputs(ROOT)
        case = make_caffeine_experiment_case(ROOT)
        external=[]
    candidates = candidate_space(case, root=ROOT)
    matched_count = args.exact_free_parameter_count
    if matched_count is not None and not 0 <= matched_count <= len(case['parameter_search']['parameters']):
        raise ValueError('Matched parameter count must lie in the registered parameter dimension')
    anchor_options = [c for c in candidates if c['structure_id'] == 'partition_pksim_standard'
                      and (len(c['free_parameter_ids']) == matched_count if matched_count is not None
                           else set(c['free_parameter_ids']) == set(c['parameter_ids']))]
    base = min(anchor_options, key=lambda c: c['candidate_id'])['candidate_id']
    # Historical task has no evidence-supported rule replacing the base formula.
    # Keeping M1 == M0 is an explicit legitimate outcome, not a manufactured contrast.
    rule = base
    jobs = [{'method': method, 'seed': seed} for method in args.methods
            for seed in ([17] if method in {'M0', 'M1', 'E0'} else args.seeds)]
    random.Random(20260918).shuffle(jobs)
    # Run the engineering anchor first, then the balanced reproducible order.
    jobs.sort(key=lambda job: job['method'] != 'M0')
    started = time.monotonic()
    progress_lock = threading.Lock()
    state = {'status': 'starting', 'phase': 'historical_development', 'completed_jobs': 0,
             'planned_jobs': len(jobs), 'main_benchmark_started': False, 'independent_test_performed': False}
    def update(**items):
        with progress_lock:
            state.update(items, elapsed_seconds=time.monotonic()-started)
            save(output/'progress.json', state, replace=True)
            print(json.dumps({k: state.get(k) for k in ('status', 'method', 'seed', 'stage', 'candidate', 'fit_round',
                                                       'completed_jobs', 'planned_jobs', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    experiment = NativeExperiment(case=case, output=output, project_root=ROOT, device=args.device,
                        cpu_threads=args.cpu_threads, scenario_workers=args.scenario_workers,
                        max_output_bytes=args.max_output_mib*1024**2, progress_callback=update)
    (output/'runner.pid').write_text(str(os.getpid())+'\n')
    codes = sorted({'scripts/run_blueprint_experiments.py',
                    *(str(p.relative_to(ROOT)) for p in (ROOT/'src/pbpk_kb/blueprints').glob('*.py')),
                    'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
                    'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R'})
    # This workspace has no Git history; preserve the exact small source snapshot.
    with zipfile.ZipFile(output/'runner_source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in codes:
            archive.write(ROOT/relative, arcname=relative)
    config = {'purpose': 'historical_development_experiment', 'case_sha256': file_sha(output/'case.json'),
              'compound_id':case['compound_id'],'tier':case.get('tier','development'),
              'case_bundle_path':str(source_path(args.case_file).relative_to(ROOT)) if args.case_file else None,
              'case_bundle_sha256':file_sha(source_path(args.case_file)) if args.case_file else None,
              'external_scenario_count':len(external),'external_observations_available_to_optimizer':False,
              'external_scoring_deferred':args.defer_external_scoring,
              'fallback_policy':'source_initial_standard_base_for_invalid_submissions' if args.defer_external_scoring else None,
              'jobs': jobs, 'candidates': candidates, 'base_candidate': base, 'rule_candidate': rule,
              'exact_free_parameter_count':matched_count,
              'anchor_selection':'source standard; full parameters by default, canonical candidate_id among matched dimension before fitting',
              'diagnostic_policy':'M0-M4 reporting only; M5 parameter stage and M6/A1/A2 feedback; A3 reporting only; matched dimension never reduced',
              'qwen_proposals_path':str(source_path(args.qwen_proposals).relative_to(ROOT)) if args.qwen_proposals else None,
              'qwen_manifest_sha256':file_sha(source_path(args.qwen_proposals)/'manifest.json') if args.qwen_proposals else None,
              'device': args.device, 'cpu_threads_per_native_process': args.cpu_threads, 'parallel_scenarios': args.scenario_workers,
              'parallelism':parallelism,
              'max_output_bytes': args.max_output_mib*1024**2, 'max_solver_calls_per_method': args.max_solver_calls,
              'diagnostic_reserved_fraction': .2, 'max_evaluations_per_start': 50, 'starts_per_candidate': 3,
              'selection_tolerance': .02, 'parameter_initialization_fractions': [.1, .5, .9],
              'objective': 'sqrt(mean_study(mean_curve(mean_point(log_prediction_error_squared))))',
              'objective_scope': 'development fitting/selection objective; not the preregistered mean-curve-RMSE final test endpoint',
              'native_cache_semantics': 'Only identical requested native model/parameter/scenario I/O; all logical calls charged; no cross-method score lookup.',
              'guidance_limitation': 'The current plasma case has equal requirement/evidence support across legal variants; A1/A2 may be identical and cannot establish guidance benefit.',
              'topology_changed': False, 'change_class': 'native_partition_constitutive_methods',
              'paper_protocol_frozen': False, 'independent_test_loaded': False, 'scientific_accepted': False,
              'code_sha256': {p: file_sha(ROOT/p) for p in codes}}
    save(output/'configuration.json', config)
    save(output/'external_scenario_designs.json',external)
    summary = []
    chosen = []
    try:
        if 'M2' in args.methods and args.qwen_proposals:
            from pbpk_kb.blueprints.qwen_proposals import load_proposer
            proposer = load_proposer(source_path(args.qwen_proposals), case, candidates, project_root=ROOT)
        else:
            proposer = registered_qwen_proposer(case, output) if 'M2' in args.methods else None
        update(status='running', stage='begin_methods')
        for job in jobs:
            directory = output/'methods'/('%s_seed_%d' % (job['method'], job['seed']))
            directory.mkdir(parents=True)
            ledger = BudgetLedger(args.max_solver_calls, diagnostic_fraction=.2)
            context = {'ledger': ledger, 'model_requests': set(), 'output': directory}
            update(method=job['method'], seed=job['seed'], stage='candidate_search', candidate=None, fit_round=None)
            method_started = time.monotonic()
            result = run_method_search(method=job['method'], candidates=candidates, evaluate=experiment.evaluator(context),
                    seed=job['seed'], max_candidates=24, max_solver_calls=ledger.search_limit,
                    selection_tolerance=.02, base_candidate_id=base, rule_candidate_id=rule, qwen_proposer=proposer,
                    exact_free_parameter_count=matched_count)
            save(directory/'search_result.json', result)
            if context.get('infrastructure_failure'):
                save(directory/'failure.json', context['infrastructure_failure'])
                raise RuntimeError('Native failure stopped the batch; inspect '+str(directory/'failure.json'))
            checks = None
            if result.get('selected_candidate') and result.get('selected_result'):
                update(stage='selected_model_rebuild')
                checks = experiment.rebuild_selected(result['selected_candidate'], result['selected_result'], context)
                save(directory/'selected_rebuild_checks.json', checks)
                chosen.append({'method':job['method'],'seed':job['seed'],'candidate':result['selected_candidate'],
                               'result':result['selected_result'],'directory':directory})
            row = {'method': job['method'], 'seed': job['seed'], 'status': result['status'],
                   'blocked_reason': result.get('blocked_reason'),
                   'selected_candidate_id': (result.get('selected_candidate') or {}).get('candidate_id'),
                   'calibration_log_rmse': (result.get('selected_result') or {}).get('calibration_log_rmse'),
                   'selection_log_rmse': (result.get('selected_result') or {}).get('selection_log_rmse'),
                   'effective_free_parameter_ids': (result.get('selected_result') or {}).get('effective_free_parameter_ids'),
                   'native_constitutive_variants_evaluated': len({e['structure_id'] for e in result.get('evaluations', [])}),
                   'evaluated_candidates': len(result.get('evaluations', [])), 'budget': ledger.snapshot(),
                   'selected_rebuild_verified': bool(checks) and all(c['accepted'] for c in checks),
                   'prediction_available': bool(checks) and all(c['accepted'] for c in checks),
                   'search_terminated_due_to_budget': result['status']=='budget_exhausted',
                   'wall_seconds': time.monotonic()-method_started,
                   'wall_seconds_scope':'native_search_and_selected_rebuild; offline_LLM_preparation_reported_separately',
                   'scientific_accepted': False, 'independent_test_performed': False}
            if job['method']=='M2' and args.qwen_proposals:
                proposal_path=source_path(args.qwen_proposals)/'cases'/case['case_id']/('seed_%d.json'%job['seed'])
                if proposal_path.is_file():
                    proposal=json.loads(proposal_path.read_text())
                    metrics=proposal.get('metrics',{})
                    row['offline_llm_cost']={
                        'record_path':str(proposal_path.relative_to(ROOT)),
                        'record_sha256':file_sha(proposal_path),'proposal_status':proposal.get('status'),
                        'batch_wall_seconds':metrics.get('wall_seconds'),'batch_size':metrics.get('batch_size'),
                        'batch_case_ids':metrics.get('batch_case_ids'),
                        'cuda_peak_reserved_bytes':metrics.get('cuda_peak_reserved_bytes'),
                        'cost_policy':'Batch latency is shared, not additive per case; weight verification and loading are in the proposal progress artifact'}
            save(directory/'result.json', row)
            summary.append(row)
            save(output/'results.json', {'purpose': 'historical_development', 'methods': summary,
                    'planned_jobs': len(jobs), 'completed_jobs': len(summary), 'main_benchmark_completed': False}, replace=True)
            update(completed_jobs=len(summary), stage='method_finished', output_bytes=experiment.disk_bytes())
        external_count=0
        predictions_count=0
        if external:
            if args.defer_external_scoring:
                base_candidate=next(c for c in candidates if c['candidate_id']==base)
                records=predict_external(experiment,case,external,chosen,output,update,jobs=summary,fallback_candidate=base_candidate)
                predictions_count=len(records)
            else:
                external_count=evaluate_external(experiment,case,external,chosen,output,update)
        if any(r['status'] not in {'completed', 'blocked'} for r in summary):
            status = 'completed_with_unsuccessful_methods'
        else:
            status = 'completed_with_blocked_methods' if any(r['status']=='blocked' for r in summary) else 'completed'
        update(status=status, stage='finished')
        save(output/'completion.json', {'status': status, 'methods': summary, 'elapsed_seconds': time.monotonic()-started,
                'external_study_evaluation_performed':bool(external_count),'external_evaluations':external_count,
                'external_prediction_submissions':predictions_count,'external_scoring_deferred':args.defer_external_scoring,
                'independent_test_performed': False, 'main_benchmark_completed': False, 'scientific_accepted': False},replace=True)
        return 0
    except BaseException as exc:
        update(status='failed', stage='stopped', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        experiment.close()


if __name__ == '__main__':
    raise SystemExit(main())
