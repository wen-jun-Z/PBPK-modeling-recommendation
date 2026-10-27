"""Bounded, resumable clinical execution validation; no outcome-directed tuning."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import rank_recommendation_execution as ranking

OUT, RUN = ranking.OUT, ranking.RUN
REVISION = 'engineering_repair_02'
MIGRATIONS = OUT / 'migrations_repair_03'
AMENDMENT = OUT / 'engineering_amendments/repair_03.json'
PYTHON = ROOT / 'runtime/app-env/bin/python'
SCRIPT = Path(__file__).resolve()
CODE = ['scripts/run_recommendation_execution.py', 'scripts/rank_recommendation_execution.py',
        'scripts/build_recommendation_execution_cohort.py', 'scripts/recommendation_candidate_executor.py',
        'scripts/recommendation_candidate_migration.py',
        'scripts/recommendation_execution_metrics.py', 'scripts/score_recommendation_execution.py',
        'scripts/recommendation_execution_statistics.py', 'scripts/recommendation_execution_source_audit.py',
        'scripts/recommendation_execution_tiers.py',
        'src/pbpk_kb/recommendation/model_matching/graphs.py',
        'src/pbpk_kb/recommendation/model_matching/scoring.py',
        'scripts/run_frontier_comparison.py', 'scripts/frontier_external_models.py',
        'scripts/frontier_graph_models.py', 'scripts/frontier_result_audit.py',
        'src/pbpk_kb/blueprints/source_workflow.py', 'src/pbpk_kb/blueprints/source_adaptation.py',
        'src/pbpk_kb/blueprints/source_adaptation_audit.py', 'src/pbpk_kb/blueprints/design_runtime.py',
        'src/pbpk_kb/blueprints/source_output_grid.py',
        'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
        'src/pbpk_kb/model_runtime/R/source_adaptation_audit.R',
        'src/pbpk_kb/model_runtime/R/catalog_snapshot_migration.R']


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    ranking.frontier.write_json(ranking.inside(path), value)


def ref(path):
    p = ranking.inside(path)
    return {'path': str(p.relative_to(ROOT)), 'sha256': ranking.sha(p)}


def checked(reference):
    p = ranking.inside(reference['path'])
    if ranking.sha(p) != reference['sha256']:
        raise ValueError('Frozen reference changed: ' + reference['path'])
    return p


def environment():
    return dict(PATH=f'{PYTHON.parent}:/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'scripts'),
                PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                HF_HOME=str(ROOT / 'runtime/hf_cache/frontier'),
                HF_HUB_CACHE=str(ROOT / 'runtime/hf_cache/frontier/hub'),
                XDG_CACHE_HOME=str(RUN / 'cache'), TMPDIR=str(RUN))


def pair_key(row):
    return hashlib.sha256((row['scenario_id'] + '\n' + row['candidate_id']).encode()).hexdigest()[:24] + '_r2'


def identity(pid):
    p = Path(f'/proc/{pid}')
    if not p.exists() or p.stat().st_uid != os.getuid():
        return None
    try:
        # Only the PID created and recorded for this project is inspected.
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'uid': os.getuid(), 'start_ticks': fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive():
    path = RUN / 'pid.json'
    if not path.exists():
        return False
    record = ranking.read(path)
    if record.get('script') != str(SCRIPT.relative_to(ROOT)):
        return False
    saved = record.get('identity')
    if not isinstance(saved, dict) or type(saved.get('pid')) is not int or saved['pid'] <= 0:
        return False
    if saved.get('uid') != os.getuid() or not saved.get('start_ticks'):
        return False
    if saved != identity(saved['pid']):
        return False
    process = Path(f"/proc/{saved['pid']}")
    try:
        if (process / 'cwd').resolve() != ROOT:
            return False
        args = (process / 'cmdline').read_bytes().split(b'\0')
        return str(SCRIPT).encode() in args and b'run' in args
    except (FileNotFoundError, ProcessLookupError):
        return False


def verify_protocol():
    p = ranking.read(OUT / 'protocol.json')
    for x in [*p['inputs'], *p['code']]:
        checked(x)
    from recommendation_execution_tiers import verify_declared_tiers
    verify_declared_tiers(check_original_sources=False)
    return p


def verified_native_result(job):
    directory = OUT / 'native' / job['job_id']
    result = ranking.read(directory / 'result.json')
    plan_path = directory / 'execution_plan.json'
    if ranking.read(plan_path) != job['plan'] or ranking.sha(plan_path) != result.get('execution_plan_sha256'):
        raise ValueError('Saved native result has a different frozen plan')
    if (job['plan'].get('execution_revision') != REVISION or result.get('execution_revision') != REVISION or
            result.get('implementation_sha256') != job['plan'].get('implementation_sha256')):
        raise ValueError('Native execution revision or implementation differs')
    for key in ('scenario_id', 'candidate_id'):
        if result.get(key) != job[key]:
            raise ValueError('Saved native result identity changed')
    if result.get('status') not in {'blocked', 'prediction_locked', 'native_failed', 'readback_failed', 'failed'}:
        raise ValueError('Saved native result has no terminal disposition')
    if result.get('prediction_locked'):
        prediction = checked({'path': result['prediction_file'], 'sha256': result['prediction_sha256']})
        lock = ranking.read(checked({'path': result['prediction_lock'], 'sha256': result['prediction_lock_sha256']}))
        if lock.get('execution_revision') != REVISION or lock.get('implementation_sha256') != job['plan'].get('implementation_sha256'):
            raise ValueError('Native prediction lock implementation differs')
        if lock.get('source_sha256') != job['plan'].get('source_sha256'):
            raise ValueError('Native lock source changed')
        if lock.get('scenario_sha256') != job['plan'].get('scenario_sha256'):
            raise ValueError('Native lock scenario changed')
        if lock.get('candidate_sha256') != job['plan'].get('candidate_sha256'):
            raise ValueError('Native lock candidate changed')
        for path, digest in lock.get('files', {}).items():
            checked({'path': path, 'sha256': digest})
        if not lock.get('files') or str(prediction.relative_to(ROOT)) not in lock['files']:
            raise ValueError('Prediction missing from its native lock')
    return result


def preparation_inputs():
    return [OUT / 'README.md', OUT / 'inputs/cohort.json', OUT / 'inputs/scenarios.jsonl',
            OUT / 'inputs/candidate_audit.jsonl', OUT / 'inputs/declared_tiers.jsonl',
            OUT / 'inputs/declared_tiers_manifest.json', AMENDMENT, *[ranking.RANK / 'inputs' / n for n in
                                                 ('queries.jsonl', 'candidates.jsonl', 'manifest.json')]]


def freeze_preparation():
    path = OUT / 'preparation.json'
    if path.exists():
        p = ranking.read(path)
        for item in [*p['inputs'], *p['code']]:
            checked(item)
        return p
    ranking.prepare_inputs()
    from recommendation_execution_tiers import prepare_declared_tiers
    prepare_declared_tiers()
    value = dict(schema_version='pbpk-execution-preparation-freeze-v1', frozen_at=now(),
                 execution_revision=REVISION,
                 inputs=[ref(p) for p in preparation_inputs()], code=[ref(ROOT / p) for p in CODE],
                 native_migration_only=True, migration_can_fit_parameters=False,
                 sealed_outcomes_accessed=False, migration_workers=2,
                 migration_seconds_per_stage=600, migration_seconds_per_source_total=1200,
                 migration_total_timeout_seconds=4 * 3600,
                 migration_output_budget_bytes=4 * 1024**3,
                 runtime=runtime_limits())
    ranking.frozen_write(path, value)
    return value


def prepare():
    if (OUT / 'protocol.json').exists():
        return verify_protocol()
    for folder in (OUT, RUN, RUN / 'cache', OUT / 'native'):
        folder.mkdir(parents=True, exist_ok=True)
    freeze_preparation()
    if not (MIGRATIONS / 'migration_summary.json').exists():
        raise ValueError('Native migration accounting must finish before freezing execution plans; use start')
    migration = ranking.read(MIGRATIONS / 'migration_summary.json')
    if migration.get('status') not in {'accounted', 'stopped'} or len(migration.get('results', [])) != migration.get('expected_sources'):
        raise ValueError('Native migration accounting is still incomplete; do not freeze plans yet')
    manifest = ranking.prepare_inputs()
    scenarios = ranking.rows(OUT / 'inputs/scenarios.jsonl')
    audits = ranking.rows(OUT / 'inputs/candidate_audit.jsonl')
    from recommendation_candidate_executor import prepare_candidate
    by_scenario = {s['scenario_id']: s for s in scenarios}
    jobs = []
    for row in audits:
        if row.get('identity_match') is not True:
            continue
        plan = prepare_candidate(by_scenario[row['scenario_id']], row, project_root=ROOT)
        jobs.append(dict(job_id=pair_key(row), scenario_id=row['scenario_id'],
                         candidate_id=row['candidate_id'], plan=plan))
    ranking.frozen_write(OUT / 'inputs/execution_plan.json',
        dict(schema_version='pbpk-candidate-execution-plan-v1', jobs=jobs,
             all_same_parent_candidates_included=True, selected_using_rank_or_error=False,
             non_parent_candidates='Audited in candidate_audit.jsonl; excluded from direct parent prediction, reference value remains unknown.'))
    input_paths = [OUT / 'README.md', OUT / 'preparation.json', MIGRATIONS / 'migration_summary.json', AMENDMENT,
                   OUT / 'inputs/cohort.json', OUT / 'inputs/scenarios.jsonl',
                   OUT / 'inputs/declared_tiers.jsonl', OUT / 'inputs/declared_tiers_manifest.json',
                   OUT / 'inputs/candidate_audit.jsonl', OUT / 'inputs/execution_plan.json',
                   *[ranking.RANK / 'inputs' / n for n in ('queries.jsonl', 'candidates.jsonl', 'manifest.json')]]
    p = dict(schema_version='pbpk-recommendation-execution-protocol-v1', frozen_at=now(),
             execution_revision=REVISION,
             scientific_target='Clinical PK accuracy and missed suitable candidates, with three-tier scope and explicit unknowns',
             cases=len(scenarios), compounds=len({s['compound'] for s in scenarios}), candidates=280,
             planned_candidate_executions=len(jobs), complete_scope_audit_pairs=len(audits),
             methods=list(ranking.METHODS), ranking_variants=['original'], new_clinical_requests=True,
             declared_policy='Existing deterministic engineering-use policy; original classes retained without treating blocked as scientific inapplicability',
             method_changes='No training; same pinned weights. ReasonRank official 3172-token budget from first attempt. Uniform published recovery/equivalence rules.',
             inputs=[ref(x) for x in input_paths], code=[ref(ROOT / x) for x in CODE],
             runtime=runtime_limits(),
             statistics=dict(primary='Per-endpoint selection regret and missed validated-direct candidates; no combined score.',
                             exploratory_fold_thresholds=[1.5, 2.0], bootstrap_seed=20261001,
                             bootstrap_draws=2000, cluster='compound with study-level aggregation',
                             shared_publications='Join compounds sharing a publication into one resampling/sign-flip block',
                             paired_comparison='Each method versus BM25 on shared scenarios with top1 numerical coverage exactly one',
                             minimum_independent_blocks=10, significance='two-sided paired block sign-flip',
                             multiplicity='Holm within each endpoint for methods versus BM25',
                             unknown='Never converted to failure or scientific negative',
                             tie_rule='Uniform permutation expectation within each equal-score block'),
             outcome_access=dict(sealed_values_read_by_preparation=False, all_predictions_and_rankings_locked_before_scoring=True,
                                 previous_source_curation_not_claimed_new_blind_data=True),
             completion='All ranking/execution units accounted for; scientific coverage reported separately, no clinical validation claimed for blocked cases.')
    ranking.frozen_write(OUT / 'protocol.json', p)
    save(RUN / 'status.json', dict(state='prepared', updated_at=now(), protocol=ref(OUT / 'protocol.json'),
                                planned_candidate_executions=len(jobs), scenarios=len(scenarios), methods=12))
    return p


def runtime_limits():
    return dict(native_workers=2, cpu_threads_per_native_worker=2, gpu_ranking_workers=1,
                native_timeout_seconds=300, total_timeout_seconds=48 * 3600,
                rank_worker_timeout_seconds=6 * 3600, no_progress_timeout_seconds=20 * 60,
                minimum_free_bytes=20 * 1024**3, output_limit_bytes=8 * 1024**3,
                log_limit_bytes=64 * 1024**2, retries=0)


def verify_retained_ranking(method, *, check_weights=False):
    """Validate saved identity and provenance before the rank runner can overwrite them."""
    import numpy as np
    base = ranking.RANK / 'results' / method
    status_path = base / 'status.json'
    if not status_path.exists():
        if base.exists() and any(base.iterdir()):
            raise ValueError('Ranking artifacts without provenance status must not be reused: ' + method)
        return {'method': method, 'state': 'not_started', 'complete_rows': 0}
    q, c, hashes = ranking.load_inputs()
    state = ranking.read(status_path)
    if state.get('input_sha256') != hashes:
        raise ValueError('Retained ranking inputs differ: ' + method)
    expected = {name: ranking.sha(ROOT / 'scripts' / name) for name in
                ('run_frontier_comparison.py', 'frontier_external_models.py')}
    if ranking.frontier.METHODS[method]['kind'] in {'g_reasoner', 'gfm_rag'}:
        expected['frontier_graph_models.py'] = ranking.sha(ROOT / 'scripts/frontier_graph_models.py')
    if state.get('implementation_sha256') != expected:
        raise ValueError('Retained ranking implementation differs: ' + method)
    wrapper = state.get('clinical_wrapper_sha256')
    # The wrapper stamps its own SHA after run_method completes. A partial worker
    # is also bound to the unchanged archived preparation and rank-input manifest.
    if wrapper is not None and wrapper != ranking.sha(ROOT / 'scripts/rank_recommendation_execution.py'):
        raise ValueError('Retained ranking wrapper differs')
    if state.get('state') == 'complete' and wrapper is None:
        raise ValueError('Completed method lacks wrapper provenance')
    score_path, timing_path = base / 'scores.npy', base / 'query_seconds.npy'
    if score_path.exists() != timing_path.exists():
        raise ValueError('Incomplete score/timing file pair')
    complete = 0
    if score_path.exists():
        expected_index = {'query_ids': [x['query_id'] for x in q],
                          'candidate_ids': [x['candidate_id'] for x in c], 'variants': ['original']}
        if ranking.read(base / 'index.json') != expected_index:
            raise ValueError('Retained ranking identity order differs')
        scores, timings = [np.load(p, allow_pickle=False) for p in (score_path, timing_path)]
        if scores.shape != (1, len(q), len(c)) or timings.shape != (1, len(q)):
            raise ValueError('Retained ranking array shape differs')
        complete = int((np.isfinite(scores).all(axis=2) & np.isfinite(timings) & (timings >= 0)).sum())
        if state.get('state') == 'complete' and complete != len(q):
            raise ValueError('Completed method contains incomplete rows')
    elif state.get('state') == 'complete':
        raise ValueError('Completed method has no score arrays')
    weights = list(base.glob('*weights*.json'))
    if method != 'bm25_text' and score_path.exists() and not (base / 'weights.json').exists():
        raise ValueError('Retained neural ranking lacks its weight declaration')
    for weight_path in weights:
        previous = ranking.OLD / 'results' / method / weight_path.name
        weight = ranking.read(weight_path)
        if not previous.exists() or weight != ranking.read(previous):
            raise ValueError('Retained model weight declaration changed')
        if check_weights:
            for name, sha in weight['sha256'].items():
                checked({'path': str(ranking.inside(weight['path']) / name), 'sha256': sha})
    return {'method': method, 'state': state.get('state'), 'complete_rows': complete,
            'partial_rows_reused_only_when_scores_and_timings_finite': True}


def amend_engineering():
    """Preserve revision one, then freeze a repair before any target outcomes are read."""
    with (RUN / 'start.lock').open('a') as start_lock, (RUN / 'pipeline.lock').open('a') as pipeline_lock:
        fcntl.flock(start_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record in ranking.frontier.read_json(RUN / 'children.json', {}).values():
            saved = record.get('identity')
            if isinstance(saved, dict) and type(saved.get('pid')) is int and saved.get('pid', 0) > 0:
                if saved == identity(saved['pid']):
                    raise ValueError('A recorded project child is still running; do not amend')
        return _amend_engineering_locked()


def finish_amendment(transaction_path):
    transaction = ranking.read(transaction_path)
    for record in [*transaction['code'], *transaction['unchanged_inputs']]:
        checked(record)
    for operation in transaction['moves']:
        source, destination = [ranking.inside(operation[k]) for k in ('source', 'destination')]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if source.exists() or ranking.sha(destination) != operation['sha256']:
                raise ValueError('Conflicting amendment history; never overwrite')
        else:
            if not source.exists() or ranking.sha(source) != operation['sha256']:
                raise ValueError('Amendment source is missing or changed')
            source.rename(destination)
    value = transaction['amendment']
    for record in value['archived_files']:
        checked(record)
    ranking.frozen_write(AMENDMENT, value)
    # Commit retirement before recreating an active preparation at the old
    # pathname. A crash after recreation must not replay these historical moves.
    transaction_path.rename(transaction_path.with_name(transaction_path.name.replace('pending', 'completed', 1)))
    return freeze_preparation()


def amend_migration_recovery():
    """Repair a preparation-only migration bug without replacing ranking evidence."""
    if (OUT / 'protocol.json').exists() or (OUT / 'results.json').exists() or (ranking.RANK / 'ranking_lock.json').exists():
        raise ValueError('Migration recovery amendment requires preparation-only state')
    archive_dir = OUT / 'engineering_amendments/attempt_02'
    archive_path = archive_dir / 'archive_manifest.json'
    archive = ranking.read(archive_path)
    for item in archive['files']:
        checked({'path': item['archive_path'], 'sha256': item['sha256']})
        if item['original_path'] not in CODE:
            checked({'path': item['original_path'], 'sha256': item['sha256']})
    previous = ranking.read(OUT / 'preparation.json')
    if ref(OUT / 'preparation.json') != archive['preparation']:
        raise ValueError('Preparation changed after migration pause')
    for item in previous['inputs']:
        checked(item)
    allowed = {'scripts/run_recommendation_execution.py', 'scripts/recommendation_candidate_migration.py'}
    changes = []
    for item in previous['code']:
        current = ref(ROOT / item['path'])
        if current['sha256'] != item['sha256']:
            if item['path'] not in allowed:
                raise ValueError('Non-migration code changed: ' + item['path'])
            changes.append({'path': item['path'], 'old_sha256': item['sha256'], 'new_sha256': current['sha256']})
    retained = [verify_retained_ranking(m, check_weights=True) for m in ranking.METHODS]
    source = OUT / 'preparation.json'
    destination = archive_dir / 'retired_active_files/preparation.json'
    if destination.exists():
        raise ValueError('Preparation history already exists without a pending transaction')
    original_ref = ref(source)
    value = dict(schema_version='pbpk-pre-outcome-engineering-amendment-v1', execution_revision=REVISION,
                 preparation_amendment='migration_recovery_03', created_at=now(),
                 parent_amendment=ref(OUT / 'engineering_amendments/repair_02.json'),
                 archived_files=[ref(archive_path), {'path': str(destination.relative_to(ROOT)), 'sha256': original_ref['sha256']}],
                 previous_migrations_retained_in_place=True, migration_directory=str(MIGRATIONS.relative_to(ROOT)),
                 repairs=['Re-export an original project when native schema maintenance changed its archived working copy; never alter the original source or accept unproven model equivalence'],
                 no_clinical_observations_accessed=True, ranking_inputs_and_implementations_unchanged=True,
                 retained_rankings=retained, code_changes=changes)
    transaction = dict(moves=[{'source': original_ref['path'], 'destination': str(destination.relative_to(ROOT)), 'sha256': original_ref['sha256']}],
                       amendment=value, code=[ref(ROOT / name) for name in CODE], unchanged_inputs=previous['inputs'])
    pending = OUT / 'engineering_amendments/pending_repair_03_transaction.json'
    ranking.frozen_write(pending, transaction)
    return finish_amendment(pending)


def _amend_engineering_locked():
    if alive():
        raise ValueError('Stop this project pipeline before an engineering amendment')
    pending = OUT / 'engineering_amendments/pending_repair_03_transaction.json'
    if pending.exists():
        return finish_amendment(pending)
    if AMENDMENT.exists():
        value = ranking.read(AMENDMENT)
        for record in value['archived_files']:
            checked(record)
        return freeze_preparation()
    if (OUT / 'engineering_amendments/attempt_02/archive_manifest.json').exists():
        return amend_migration_recovery()
    pending = OUT / 'engineering_amendments/pending_repair_02_transaction.json'
    if pending.exists():
        return finish_amendment(pending)
    if (OUT / 'results.json').exists() or (ranking.RANK / 'ranking_lock.json').exists():
        raise ValueError('This pre-outcome engineering amendment cannot alter a completed/scored experiment')
    old = OUT / 'engineering_amendments/attempt_01'
    archive = ranking.read(old / 'archive_manifest.json')
    for item in archive['files']:
        checked({'path': item['archive_path'], 'sha256': item['sha256']})
    by_original = {item['original_path']: item for item in archive['files']}
    previous_protocol = ranking.read(OUT / 'protocol.json')
    if ref(OUT / 'protocol.json') != archive['original_protocol'] or ref(OUT / 'preparation.json') != archive['original_preparation']:
        raise ValueError('Parent protocol/preparation changed after pause')
    for item in previous_protocol['inputs']:
        checked(item)
    allowed = {'scripts/run_recommendation_execution.py', 'scripts/recommendation_candidate_executor.py',
               'scripts/recommendation_candidate_migration.py', 'scripts/score_recommendation_execution.py',
               'src/pbpk_kb/blueprints/source_adaptation.py', 'src/pbpk_kb/blueprints/source_adaptation_audit.py',
               'src/pbpk_kb/blueprints/source_output_grid.py',
               'src/pbpk_kb/blueprints/design_runtime.py',
               'src/pbpk_kb/model_runtime/R/source_adaptation_audit.R',
               'src/pbpk_kb/model_runtime/R/catalog_snapshot_migration.R'}
    changes = []
    for item in previous_protocol['code']:
        current = ref(ROOT / item['path'])
        if current['sha256'] != item['sha256']:
            if item['path'] not in allowed:
                raise ValueError('Non-engineering code changed: ' + item['path'])
            changes.append({'path': item['path'], 'old_sha256': item['sha256'], 'new_sha256': current['sha256'],
                            'old_code_archive': by_original[item['path']]['archive_path']})
    retained = [verify_retained_ranking(m, check_weights=True) for m in ranking.METHODS]
    snapshots = []
    for method in ranking.METHODS:
        for name in ('scores.npy', 'query_seconds.npy', 'status.json', 'index.json'):
            original = ranking.RANK / 'results' / method / name
            if original.exists():
                copied = old / 'retained_rankings' / method / name
                copied.parent.mkdir(parents=True, exist_ok=True)
                if copied.exists():
                    if ranking.sha(copied) != ranking.sha(original):
                        raise ValueError('Existing retained ranking archive differs')
                else:
                    shutil.copy2(original, copied)
                snapshots.append(ref(copied))
    previous_lock = ranking.read(OUT / 'native/predictions_lock.json')
    if previous_lock['protocol'] != archive['original_protocol']:
        raise ValueError('Previous native lock did not bind the previous protocol')
    for job in previous_lock['jobs']:
        for artifact in job['artifacts']:
            checked(artifact)
    retired, operations = [], []
    for name in ('protocol.json', 'preparation.json', 'inputs/execution_plan.json', 'native/predictions_lock.json'):
        original = OUT / name
        destination = old / 'retired_active_files' / name
        if destination.exists():
            raise ValueError('Historical destination already exists without a pending transaction')
        original_ref = ref(original)
        operations.append({'source': original_ref['path'], 'destination': str(destination.relative_to(ROOT)),
                           'sha256': original_ref['sha256']})
        retired.append({'path': str(destination.relative_to(ROOT)), 'sha256': original_ref['sha256']})
    value = dict(schema_version='pbpk-pre-outcome-engineering-amendment-v1', execution_revision=REVISION,
                 created_at=now(), original_protocol=archive['original_protocol'], original_preparation=archive['original_preparation'],
                 archived_files=[ref(old / 'archive_manifest.json'), *retired, *snapshots],
                 original_native_jobs_retained_in_place=True, original_migrations_retained_in_place=True,
                 new_attempt_ids='Original pair identity plus _r2; no old native result can satisfy a revised job',
                 migration_directory=str(MIGRATIONS.relative_to(ROOT)),
                 repairs=['Field-complete native migration equivalence and bounded stage recovery',
                          'Audit actual administered parent while retaining original compound dependencies',
                          'Native output-grid decimal serialization precision',
                          'Explicit frozen repeat-dose regimen where supported, with independent readback',
                          'Native implementation identity in plan, prediction and lock'],
                 no_clinical_observations_accessed=True, ranking_inputs_and_implementations_unchanged=True,
                 retained_rankings=retained, code_changes=changes,
                 gpu_ranking_can_run_during_cpu_migration=True)
    moved = {r['source'] for r in operations}
    transaction = dict(moves=operations, amendment=value,
                       code=[ref(ROOT / name) for name in CODE],
                       unchanged_inputs=[r for r in previous_protocol['inputs'] if r['path'] not in moved])
    ranking.frozen_write(pending, transaction)
    return finish_amendment(pending)


def tree_size(directory):
    size = 0
    for parent, _, names in os.walk(directory, followlinks=False):
        for name in names:
            p = Path(parent) / name
            if p.is_symlink():
                raise ValueError('Unexpected output symlink')
            try:
                size += p.stat().st_size
            except FileNotFoundError:
                pass
    return size


class Supervisor:
    def __init__(self, protocol):
        self.p = protocol
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.children = {}
        self.started = time.monotonic()
        self.last_disk = 0.

    def guard(self):
        if self.stop_event.is_set() or (RUN / 'stop.requested').exists():
            raise InterruptedError('Project experiment stop requested')
        limits = self.p['runtime']
        if time.monotonic() - self.started > limits['total_timeout_seconds']:
            raise RuntimeError('Total runtime limit reached')
        with self.lock:
            if time.monotonic() - self.last_disk > 15:
                if shutil.disk_usage(OUT).free < limits['minimum_free_bytes']:
                    raise RuntimeError('Insufficient project disk headroom')
                if tree_size(OUT) > limits['output_limit_bytes'] or tree_size(RUN) > limits['log_limit_bytes']:
                    raise RuntimeError('Project output/log budget reached')
                self.last_disk = time.monotonic()

    def child(self, args, name, *, timeout, progress=None):
        self.guard()
        log_path = RUN / (name + '.log')
        if log_path.exists() and log_path.stat().st_size > 8 * 1024**2:
            raise RuntimeError('Per-worker log already at limit')
        with log_path.open('a') as log:
            process = subprocess.Popen([str(PYTHON), '-B', str(SCRIPT), *args], cwd=ROOT,
                env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(identity=identity(process.pid), command=args, started_at=now())
            with self.lock:
                self.children[name] = record
                save(RUN / 'children.json', self.children)
            start = last_change = time.monotonic()
            last_signature = None
            try:
                while process.poll() is None:
                    self.guard()
                    if time.monotonic() - start > timeout:
                        raise TimeoutError('Bounded worker timeout: ' + name)
                    if progress:
                        path = ranking.inside(progress)
                        signature = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
                        if signature != last_signature:
                            last_signature, last_change = signature, time.monotonic()
                        if time.monotonic() - last_change > self.p['runtime']['no_progress_timeout_seconds']:
                            raise TimeoutError('No committed progress: ' + name)
                    time.sleep(1)
                if process.returncode != 0:
                    raise RuntimeError('Worker failed: ' + name + '; see project log')
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                with self.lock:
                    self.children[name]['returncode'] = process.returncode
                    self.children[name]['finished_at'] = now()
                    save(RUN / 'children.json', self.children)

    def rankings(self, *, during_preparation=False):
        for method in ranking.METHODS:
            self.guard()
            verify_retained_ranking(method)
            status = ranking.RANK / 'results' / method / 'status.json'
            if status.exists() and ranking.read(status).get('state') == 'complete':
                continue
            action = 'rank-preparation-worker' if during_preparation else 'rank-worker'
            self.child([action, '--method', method], 'ranking_' + method,
                       timeout=self.p['runtime']['rank_worker_timeout_seconds'], progress=status)
        ranking.finalize_scores()

    def native(self):
        jobs = ranking.read(OUT / 'inputs/execution_plan.json')['jobs']
        def one(job):
            self.guard()
            directory = OUT / 'native' / job['job_id']
            result = directory / 'result.json'
            if result.exists():
                verified_native_result(job)
                return
            if directory.exists():
                # Preserve incomplete own artifacts before a new bounded attempt.
                # A known old worker must be gone; do not inspect global processes.
                previous = ranking.frontier.read_json(RUN / 'children.json', {}).get('native_' + job['job_id'], {})
                saved = previous.get('identity')
                if saved and saved == identity(saved['pid']):
                    raise RuntimeError('Previous owned candidate worker remains alive; do not duplicate it')
                saved_dir = OUT / 'native/interrupted'
                saved_dir.mkdir(exist_ok=True)
                directory.rename(saved_dir / (job['job_id'] + '_' + str(time.time_ns())))
            self.child(['native-worker', '--job', job['job_id']], 'native_' + job['job_id'], timeout=900)
        with ThreadPoolExecutor(max_workers=self.p['runtime']['native_workers']) as pool:
            futures = [pool.submit(one, job) for job in jobs]
            for future in as_completed(futures):
                future.result()
        # One record for every planned job, including blocked or failed units.
        records = []
        for job in jobs:
            directory = OUT / 'native' / job['job_id']
            verified_native_result(job)
            records.append(dict(job_id=job['job_id'], scenario_id=job['scenario_id'], candidate_id=job['candidate_id'],
                                artifacts=[ref(p) for p in sorted(directory.rglob('*')) if p.is_file()]))
        ranking.frozen_write(OUT / 'native/predictions_lock.json',
            dict(schema_version='pbpk-all-candidate-predictions-lock-v1', protocol=ref(OUT / 'protocol.json'),
                 planned_jobs=len(jobs), jobs=records, target_observations_read=False))


def native_worker(job_id):
    # SIGTERM raises through run_bounded_r's finally, which kills its own R child.
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Owned worker stopped')
    signal.signal(signal.SIGTERM, interrupted)
    verify_protocol()
    job = next(x for x in ranking.read(OUT / 'inputs/execution_plan.json')['jobs'] if x['job_id'] == job_id)
    scenario = next(x for x in ranking.rows(OUT / 'inputs/scenarios.jsonl') if x['scenario_id'] == job['scenario_id'])
    candidate = next(x for x in ranking.rows(OUT / 'inputs/candidate_audit.jsonl')
                     if x['scenario_id'] == job['scenario_id'] and x['candidate_id'] == job['candidate_id'])
    from recommendation_candidate_executor import execute_candidate, prepare_candidate
    if prepare_candidate(scenario, candidate, project_root=ROOT) != job['plan']:
        raise ValueError('Candidate execution plan changed after freeze')
    execute_candidate(scenario, candidate, OUT / 'native' / job_id, project_root=ROOT, timeout_seconds=300)


def run():
    lock = (RUN / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = freeze_preparation()
    supervisor = Supervisor(initial)
    def request_stop(*_):
        supervisor.stop_event.set()
        (RUN / 'stop.requested').touch()
    signal.signal(signal.SIGTERM, request_stop)
    needs_migration = not (OUT / 'protocol.json').exists()

    def migration_then_native():
        if needs_migration:
            from recommendation_candidate_migration import prepare_migrations
            migration_result = prepare_migrations(ranking.rows(OUT / 'inputs/scenarios.jsonl'), ranking.rows(OUT / 'inputs/candidate_audit.jsonl'),
                               MIGRATIONS, workers=initial['migration_workers'],
                               timeout_seconds=initial['migration_seconds_per_stage'],
                               max_total_bytes=initial['migration_output_budget_bytes'],
                               max_seconds=initial['migration_total_timeout_seconds'], stop_file=RUN / 'stop.requested')
            if (RUN / 'stop.requested').exists():
                raise InterruptedError('Project migration stop requested')
            if migration_result.get('stop_reason'):
                raise InterruptedError('Migration safety limit: ' + migration_result['stop_reason'])
            freeze_preparation()  # Fail closed if code/conditions changed while conversion ran.
            prepare()
        verify_protocol()
        save(RUN / 'status.json', dict(state='running_rankings_and_native_predictions', execution_revision=REVISION,
                                      started_at=now(), updated_at=now()))
        supervisor.native()

    save(RUN / 'status.json', dict(state='running_migration_and_rankings' if needs_migration else 'running_rankings_and_native_predictions',
                                  execution_revision=REVISION, started_at=now(), updated_at=now()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(supervisor.rankings, during_preparation=needs_migration), pool.submit(migration_then_native)]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                request_stop()
                raise
        supervisor.guard()
        from score_recommendation_execution import score_run
        save(RUN / 'status.json', dict(state='scoring_locked_predictions', updated_at=now()))
        report = score_run(OUT)
        save(RUN / 'status.json', dict(state='complete_with_reported_coverage', updated_at=now(),
                                     results=ref(OUT / 'results.json'), summary=report['summary']))
    except BaseException as error:
        supervisor.stop_event.set()
        save(RUN / 'status.json', dict(state='stopped' if isinstance(error, InterruptedError) else 'failed',
                                     updated_at=now(), error_type=type(error).__name__, detail=str(error)[:400]))
        raise


def status():
    result = ranking.frontier.read_json(RUN / 'status.json', {})
    result['alive'] = alive()
    jobs = ranking.frontier.read_json(OUT / 'inputs/execution_plan.json', {}).get('jobs', [])
    completed = [OUT / 'native' / j['job_id'] / 'result.json' for j in jobs
                 if (OUT / 'native' / j['job_id'] / 'result.json').exists()]
    result['native'] = dict(accounted=len(completed), planned=len(jobs),
                           states=dict(Counter(ranking.read(p).get('status', 'unspecified') for p in completed)))
    result['ranking'] = ranking.report() if (ranking.RANK / 'inputs/manifest.json').exists() else {}
    migration = MIGRATIONS / 'migration_summary.json'
    if migration.exists():
        full = ranking.read(migration)
        result['migration'] = {k: v for k, v in full.items() if k != 'results'}
        result['migration']['states'] = dict(Counter(r.get('status', 'unknown') for r in full.get('results', [])))
    return result


def watch():
    """Human progress, with accounting kept distinct from successful predictions."""
    def bar(label, done, total):
        fraction = min(done / total, 1.) if total else 0.
        blocks = int(fraction * 28)
        return f'{label:8s}  |' + '\u2588' * blocks + '\u2591' * (28 - blocks) + f'| {done:>5}/{total:<5} {fraction:6.1%}'
    try:
        while True:
            s = status()
            m, n, r = (s.get(k, {}) for k in ('migration', 'native', 'ranking'))
            lines = ['PBPK \u63a8\u8350\u540e\u9884\u6d4b\u9a8c\u8bc1', f"\u72b6\u6001: {s.get('state', 'not_started')}  |  \u8fdb\u7a0b: {'running' if s['alive'] else 'stopped'}", '',
                     bar('\u683c\u5f0f\u8fc1\u79fb', m.get('completed_sources', 0), m.get('expected_sources', 67)),
                     bar('\u63a8\u8350\u65b9\u6cd5', len(r.get('completed_methods', [])), r.get('total_methods', 12)),
                     bar('\u6267\u884c\u767b\u8bb0', n.get('accounted', 0), n.get('planned') or 177),
                     '', '\u539f\u751f\u6267\u884c: ' + json.dumps(n.get('states', {}), ensure_ascii=False),
                     '\u767b\u8bb0\u5b8c\u6210\u5305\u62ec\u6210\u529f\u3001\u5931\u8d25\u4e0e\u963b\u585e\uff1b\u4e0d\u7b49\u4e8e\u9884\u6d4b\u51c6\u786e\u3002',
                     'Ctrl+C \u53ea\u9000\u51fa\u8fdb\u5ea6\u663e\u793a\uff0c\u540e\u53f0\u5b9e\u9a8c\u7ee7\u7eed\u3002']
            print(('\033[2J\033[H' if sys.stdout.isatty() else '') + '\n'.join(lines), flush=True)
            if not sys.stdout.isatty() or not s['alive']:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['amend', 'prepare', 'start', 'run', 'status', 'watch', 'stop', 'rank-worker', 'rank-preparation-worker', 'native-worker'])
    parser.add_argument('--method', choices=ranking.METHODS)
    parser.add_argument('--job')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == 'amend':
        print(json.dumps(amend_engineering(), ensure_ascii=False))
    elif args.action == 'prepare':
        print(json.dumps(prepare(), ensure_ascii=False))
    elif args.action == 'start':
        start_lock = (RUN / 'start.lock').open('a')
        fcntl.flock(start_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_preparation()
        if (OUT / 'protocol.json').exists():
            verify_protocol()
        if alive():
            print(json.dumps(status(), ensure_ascii=False)); return
        (RUN / 'stop.requested').unlink(missing_ok=True)
        with (RUN / 'pipeline.log').open('a') as log:
            process = subprocess.Popen([str(PYTHON), '-B', str(SCRIPT), 'run'], cwd=ROOT,
                env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(RUN / 'pid.json', dict(identity=identity(process.pid), script=str(SCRIPT.relative_to(ROOT)), created_at=now()))
        print(json.dumps(dict(started=True, pid=process.pid, status_command='runtime/app-env/bin/python -B scripts/run_recommendation_execution.py status')))
    elif args.action == 'stop':
        (RUN / 'stop.requested').touch()
        if alive():
            os.kill(ranking.read(RUN / 'pid.json')['identity']['pid'], signal.SIGTERM)
        print(json.dumps(dict(stop_requested=True)))
    elif args.action == 'run':
        run()
    elif args.action == 'rank-worker':
        verify_protocol(); ranking.run_method(args.method)
    elif args.action == 'rank-preparation-worker':
        frozen = freeze_preparation()
        if frozen.get('execution_revision') != REVISION:
            raise ValueError('Preparation revision differs')
        verify_retained_ranking(args.method)
        ranking.run_method(args.method)
    elif args.action == 'native-worker':
        native_worker(args.job)
    elif args.action == 'watch':
        watch()
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
