"""Independently authenticate every final prediction before opening outcomes."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from reviewer_external_validation import OUT, CATALOG, checked, read, ref, write_once
from recommendation_candidate_executor import _digest, implementation_hashes


def verify_task(row, scenario, candidate):
    path = checked(row['result'])
    result = read(path)
    plan_path = path.parent / 'execution_plan.json'
    plan = read(plan_path)
    assert result['execution_plan_sha256'] == ref(plan_path)['sha256']
    assert plan['scenario_sha256'] == _digest(scenario)
    assert plan['candidate_sha256'] == _digest(candidate)
    assert plan['implementation_sha256'] == implementation_hashes()
    assert (result['scenario_id'], result['candidate_id'], result['status']) == (row['scenario_id'], row['candidate_id'], row['status'])
    assert result['fitting_performed'] is False
    assert result['sealed_observations_accessed'] is False
    if row['status'] == 'prediction_locked':
        lock_path = checked({'path': result['prediction_lock'], 'sha256': result['prediction_lock_sha256']})
        lock = read(lock_path)
        assert lock['scenario_sha256'] == _digest(scenario)
        assert lock['candidate_sha256'] == _digest(candidate)
        assert lock['source_sha256'] == candidate['provenance']['source_sha256']
        for name, digest in lock['files'].items():
            checked({'path': name, 'sha256': digest})
        checked({'path': plan['source_path'], 'sha256': plan['source_sha256']})
        checked({'path': plan['execution_source_path'], 'sha256': plan['execution_source_sha256']})
    return {'result': row['result'], 'execution_plan': ref(plan_path), 'conditions_and_source_verified': True}


def run():
    native = read(OUT / 'prediction_manifest.json')
    reviewed = read(OUT / 'native_reviewed/manifest.json')
    rankings = read(OUT / 'rankings/manifest.json')
    for source in rankings['sources'] + [rankings['predictions']] + reviewed['sources'] + [reviewed['review']]:
        checked(source)
    for key in ('initial', 'migrations', 'rankings', 'producer'):
        checked(native[key])
    scenarios = {s['scenario_id']: s for s in read(OUT / 'inputs/scenarios.json')}
    candidates = {r['candidate_id']: r for r in map(json.loads, CATALOG.read_text().splitlines())}
    tasks = []
    for old in native['tasks']:
        if (old['scenario_id'], old['candidate_id']) == (reviewed['scenario_id'], reviewed['candidate_id']):
            row = {k: reviewed[k] for k in ('scenario_id', 'candidate_id', 'result', 'status', 'reasons')}
            row['previous_attempt'] = old['result']
            row['representation_review'] = reviewed['review']
        else:
            row = dict(old)
        s = scenarios[row['scenario_id']]
        row['independent_verification'] = verify_task(row, s, candidates[row['candidate_id']])
        if s['compound'] == 'metformin':
            row['evaluation_stratum'] = 'quarantined_ambiguous_dose_schedule'
            row['qualified_for_accuracy_scoring'] = False
        elif s['scenario_id'] == 'external:sildenafil_pmc13104799_day_minus4':
            row['evaluation_stratum'] = 'exploratory_explicit_20mg_assumption_unresolved_source_dose'
            row['qualified_for_accuracy_scoring'] = True
        else:
            row['evaluation_stratum'] = s['selection_stratum']
            row['qualified_for_accuracy_scoring'] = True
        row['family_id'] = candidates[row['candidate_id']]['family_id']
        tasks.append(row)
    expected = {(s['scenario_id'], c) for s in scenarios.values() for c in s['catalog_candidate_ids']}
    assert {(r['scenario_id'], r['candidate_id']) for r in tasks} == expected
    assert len(tasks) == len(expected) == 25
    prior = OUT / 'final_prediction_lock.json'
    if prior.exists():
        existing = read(prior)
        assert existing['tasks'] == tasks
        return existing
    lock = {'schema': 'reviewer-independent-final-prediction-lock-v1', 'state': 'complete',
        'locked_at': datetime.now(timezone.utc).isoformat(), 'tasks': tasks,
        'sources': [ref(OUT / 'selection_manifest.json'), ref(OUT / 'selection_amendment_01.json'),
                    ref(OUT / 'inputs/manifest.json'), ref(OUT / 'inputs/scenarios.json'), ref(OUT / 'inputs/queries.json'),
                    ref(OUT / 'rankings/manifest.json'), ref(OUT / 'prediction_manifest.json'),
                    ref(OUT / 'native_reviewed/manifest.json'), ref(OUT / 'scoring_protocol.json'), ref(CATALOG), ref(Path(__file__))],
        'accuracy_rule': 'Exact complete ranks, no replacement. Metformin excluded. The 20mg sildenafil arm is an explicit unresolved-dose assumption only.',
        'family_rule': 'Report both candidate-version and distinct-family counts; fewer than two measured families gives no independent-family preference/omission claim.',
        'strict_new_to_project_studies': ['doi:10.3389/fphar.2025.1692519'],
        'original_development_independence': 'not proven; repository evidence retained separately',
        'parameters_refitted': False, 'strict_new_outcomes_inspected_before_prediction_lock': False}
    write_once(prior, lock)
    print(json.dumps({'path': str(prior), 'sha256': ref(prior)['sha256'], 'tasks': len(tasks),
                      'native_locked': sum(r['status'] == 'prediction_locked' for r in tasks)}), flush=True)
    return lock


if __name__ == '__main__':
    run()
