#!/usr/bin/env python3
"""Correct endpoint eligibility using reviewed metadata; never change predictions.

The original protocol and scores remain intact. This is a post-run scientific
review, not a new prospective experiment or an opportunity to select a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from source_context_cases import build_case_manifest

ROOT = Path(__file__).resolve().parents[1]


def review(directory):
    directory = inside(directory, ROOT, exists=True)
    destinations = [directory / name for name in ('reviewed_evaluations.json', 'reviewed_summary.json')]
    if any(path.exists() for path in destinations):
        raise ValueError('Reviewed outputs already exist; do not overwrite a scientific review')
    summary = json.loads((directory / 'summary.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    observations = json.loads((directory / 'observations.json').read_text())
    if sha256(directory / 'predictions_locked.json') != summary['prediction_lock_sha256']:
        raise ValueError('Locked prediction file has changed')
    locked = json.loads((directory / 'predictions_locked.json').read_text())['jobs']
    old_cases = {c['case_id']: c for c in protocol['cases']['cases']}
    current_cases = {c['case_id']: c for c in build_case_manifest(ROOT)['cases']}
    reviewed = deepcopy(observations)
    corrections = []
    for row in reviewed:
        old, current = old_cases[row['case_id']], current_cases[row['case_id']]
        if old['source'] != current['source'] or old['role'] != current['role']:
            raise ValueError('Endpoint review must not change source or study role')
        policy = current.get('endpoint_review')
        if not policy:
            continue
        before = {key: row[key] for key in ('auc_evaluable', 'cmax_evaluable')}
        after = {key: policy[key] for key in before}
        if any(type(value) is not bool for value in after.values()):
            raise ValueError('Endpoint policy must use explicit booleans')
        row.update(after)
        corrections.append({'scenario_id': row['scenario_id'], 'before': before,
                            'after': after, 'basis': policy})
    if not corrections:
        raise ValueError('No reviewed endpoint corrections were registered')
    evaluations = {}
    for arm in ('source', 'context', 'candidate', 'selected'):
        evaluations[arm] = {}
        for role in sorted({r['dataset_role'] for r in reviewed}):
            rows = [r for r in reviewed if r['dataset_role'] == role]
            curves = {r['scenario_id']: locked[r['case_id'] + '/' + arm]['curve'] for r in rows}
            selection = {'method': arm + '_' + role, 'seed': 0, 'candidate_id': arm,
                         'status': 'completed', 'predictions_sha256': lock_predictions(curves)}
            evaluations[arm][role] = evaluate_locked_predictions(rows, curves, selection, device='cpu')
    original = json.loads((directory / 'evaluations.json').read_text())
    for arm, roles in evaluations.items():
        for role, evaluated in roles.items():
            for old, new in zip(original[arm][role]['rows'], evaluated['rows'], strict=True):
                for key in ('scenario_id', 'observation_sha256', 'status', 'log_rmse', 'gmfe'):
                    if old[key] != new[key]:
                        raise ValueError('Endpoint review changed concentration predictions or curve scoring')
    result = deepcopy(summary)
    result['metrics'] = {a: {r: v['tiers']['reconstruction'] for r, v in rs.items()}
                         for a, rs in evaluations.items()}
    result['post_run_review'] = {
        'reviewed_at': datetime.now(timezone.utc).isoformat(), 'corrections': corrections,
        'original_summary_sha256': sha256(directory / 'summary.json'),
        'original_observations_sha256': sha256(directory / 'observations.json'),
        'review_script_sha256': sha256(Path(__file__)),
        'review_metadata_sha256': sha256(ROOT / 'scripts/source_context_cases.py'),
        'predictions_changed': False, 'model_selection_changed': False,
        'observed_concentrations_changed': False, 'new_native_executions': 0,
        'curve_errors_and_all_failure_denominators_preserved': True,
        'original_scores_retained': True, 'prospectively_registered': False}
    save(destinations[0], evaluations)
    save(destinations[1], result)
    print(json.dumps(result['post_run_review'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    review(parser.parse_args().run)
