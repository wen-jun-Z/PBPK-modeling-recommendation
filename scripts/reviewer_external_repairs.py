"""Revalidate project-owned native migrations in the new experiment directory.

Old migration receipts stay immutable. The original adapter validates their
archived provenance again and records a new receipt for the current runtime.
No clinical observations or model-fit parameters are read.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from reviewer_external_validation import OUT, CATALOG, read, ref, checked, write_once, status
import recommendation_candidate_migration as migration
from recommendation_candidate_executor import execute_candidate


def run():
    initial = read(OUT / 'native/manifest.json')
    if not initial['complete']:
        raise ValueError('First native attempts must finish before bounded repairs')
    binding = read(OUT / 'inputs/manifest.json')
    for key in ('selection', 'master_protocol', 'catalog', 'scenarios', 'queries'):
        checked(binding[key])
    scenarios = {s['scenario_id']: s for s in read(OUT / 'inputs/scenarios.json')}
    candidates = {c['candidate_id']: c for c in map(json.loads, CATALOG.read_text().splitlines())}
    failed = [r for r in initial['tasks'] if 'VERIFIED_NATIVE_MIGRATION_UNAVAILABLE' in ' '.join(r['reasons'])]
    audit = [{'identity_match': True, 'scenario_id': r['scenario_id'],
              'candidate_id': r['candidate_id'], 'provenance': candidates[r['candidate_id']]['provenance']} for r in failed]
    # Redirect only this new process. The old module file and global experiment
    # directories remain unchanged; all resolution still uses the full checks.
    migration.DEFAULT_MIGRATIONS = str((OUT / 'migrations').relative_to(ROOT))
    tasks = migration.collect_tasks(audit)
    def migrate(task):
        folder = OUT / 'migrations' / task['source_sha256']
        if (folder / 'receipt.json').exists():
            return read(folder / 'receipt.json')
        return migration.migrate_one(task, folder, timeout_seconds=300)
    migrations = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(migrate, task) for task in tasks]):
            receipt = future.result()
            migrations.append({'source_sha256': receipt['source_sha256'], 'status': receipt['status'],
                               'reasons': receipt.get('reasons', []),
                               'receipt': ref(OUT / 'migrations' / receipt['source_sha256'] / 'receipt.json')})
            status('migration_revalidation', completed=len(migrations), total=len(tasks))
    write_once(OUT / 'migrations/manifest.json', {'complete': True, 'results': sorted(migrations, key=lambda x: x['source_sha256']),
                                                'producer': ref(Path(__file__)), 'observations_opened': False})
    successful_sources = {m['source_sha256'] for m in migrations if m['status'] == 'converted'}
    rows = []
    def repair(row):
        s, c = scenarios[row['scenario_id']], candidates[row['candidate_id']]
        key = hashlib.sha256((s['scenario_id'] + '|' + c['candidate_id']).encode()).hexdigest()[:20]
        folder = OUT / 'native_repair' / key
        if (folder / 'result.json').exists():
            result = read(folder / 'result.json')
            if result.get('prediction_locked'):
                checked({'path': result['prediction_file'], 'sha256': result['prediction_sha256']})
        else:
            result = execute_candidate(s, c, folder, timeout_seconds=300)
        return {'scenario_id': s['scenario_id'], 'candidate_id': c['candidate_id'],
                'initial_attempt': row['result'], 'result': ref(folder / 'result.json'),
                'status': result['status'], 'reasons': result['reasons']}
    jobs = [r for r in failed if candidates[r['candidate_id']]['provenance']['source_sha256'] in successful_sources]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(repair, row) for row in jobs]):
            rows.append(future.result())
            status('native_repair', completed=len(rows), total=len(jobs), last_status=rows[-1]['status'])
    updates = {(r['scenario_id'], r['candidate_id']): r for r in rows}
    final = [updates.get((r['scenario_id'], r['candidate_id']), r) for r in initial['tasks']]
    write_once(OUT / 'prediction_manifest.json', {'schema': 'reviewer-new-study-predictions-v1', 'complete': True,
        'initial': ref(OUT / 'native/manifest.json'), 'migrations': ref(OUT / 'migrations/manifest.json'),
        'rankings': ref(OUT / 'rankings/manifest.json'), 'tasks': final, 'observations_opened': False,
        'parameter_fitting': False, 'producer': ref(Path(__file__))})
    status('predictions_revalidated', total=len(final), locked=sum(r['status'] == 'prediction_locked' for r in final))


if __name__ == '__main__':
    run()
