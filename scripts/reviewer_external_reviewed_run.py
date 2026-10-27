"""Execute the one precisely reviewed pregabalin snapshot representation repair."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from reviewer_external_validation import OUT, CATALOG, read, ref, checked, write_once
from reviewer_external_migration import resolve_reviewed
import recommendation_candidate_migration as migration
from recommendation_candidate_executor import execute_candidate


def run():
    candidates = [json.loads(line) for line in CATALOG.read_text().splitlines()]
    candidate = next(c for c in candidates if c['candidate_id'] == 'cb5e5b4a-0504-5714-9e93-f7101084ddc2')
    scenario = next(s for s in read(OUT / 'inputs/scenarios.json') if s['scenario_id'] == 'external:pregabalin_pmc12511804_armA80mg')
    reviewed = resolve_reviewed(candidate['provenance'], project_root=ROOT)
    original = migration.resolve_migrated_source
    expected = candidate['provenance']['source_sha256']
    def resolve(provenance, *, project_root=ROOT, **kwargs):
        if provenance['source_sha256'] == expected:
            return resolve_reviewed(provenance, project_root=project_root)
        return original(provenance, project_root=project_root, **kwargs)
    migration.resolve_migrated_source = resolve
    folder = OUT / 'native_reviewed/pregabalin_json'
    if (folder / 'result.json').exists():
        result = read(folder / 'result.json')
    else:
        result = execute_candidate(scenario, candidate, folder, timeout_seconds=300)
    if result.get('prediction_locked'):
        checked({'path': result['prediction_file'], 'sha256': result['prediction_sha256']})
    receipt = {'schema': 'reviewer-exact-representation-repair-v1', 'scenario_id': scenario['scenario_id'],
               'candidate_id': candidate['candidate_id'], 'status': result['status'], 'reasons': result['reasons'],
               'result': ref(folder / 'result.json'), 'review': ref(OUT / 'pregabalin_snapshot_representation_review.json'),
               'sources': [ref(Path(__file__)), ref(ROOT / 'scripts/reviewer_external_migration.py'),
                           ref(OUT / 'inputs/manifest.json'), ref(OUT / 'inputs/scenarios.json'), ref(CATALOG)],
               'parameters_changed': False, 'observations_opened': False}
    write_once(OUT / 'native_reviewed/manifest.json', receipt)
    print(json.dumps({'status': result['status'], 'reasons': result['reasons']}), flush=True)


if __name__ == '__main__':
    run()
