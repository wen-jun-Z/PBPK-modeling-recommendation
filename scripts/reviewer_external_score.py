"""Score locked new-study predictions; never train, repair, or refit a model."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from reviewer_external_validation import OUT, CATALOG, read, ref, checked, write_once
from reviewer_external_lock import run as verify_prediction_lock
from recommendation_endpoint_extension_predictions import clinical_nca
from recommendation_execution_statistics import numerical_miss_audit
from pbpk_kb.recommendation.pbpk_match_v2.evaluation import exact_rank_weights, hierarchical_mean


def derive(prediction, scenario):
    """The unchanged matched-sampling NCA and explicitly sampled Cmax/Tmax."""
    times = np.asarray(prediction['time_h'], float)
    values = np.asarray(prediction['concentration_ng_ml'], float)
    sample_t = np.asarray(prediction['sampling_time_h'], float)
    sample_c = np.asarray(prediction['sampled_concentration_ng_ml'], float)
    if len(times) != len(values) or len(sample_t) != len(sample_c) or not np.isfinite(values).all():
        raise ValueError('Malformed locked prediction')
    index = int(np.argmax(sample_c))
    out = {'Cmax': float(sample_c[index]), 'Tmax': float(sample_t[index])}
    design = scenario['normalized_design']
    if len(design['regimen']['dose_times_h']) != 1:
        raise ValueError('Ambiguous repeated-dose metformin must not reach scoring')
    nca, diagnostics = clinical_nca(times.tolist(), values.tolist(), sample_t.tolist(),
        dose_mg=design['dose']['value'], infusion_h=0., route='oral',
        lloq=design['sampling'].get('lloq_ng_ml'))
    unavailable = {}
    for target, source in [('AUC_inf', 'AUC_infinity'), ('terminal_half_life', 'half_life')]:
        record = nca[source]
        if record.get('value') is not None:
            out[target] = float(record['value'])
        else:
            unavailable[target] = record.get('reason', record.get('status'))
    return out, unavailable, diagnostics


def position_row(scores, candidate_ids, errors, *, rank, time_endpoint):
    weights = exact_rank_weights(scores, rank)
    known = [(c, float(w)) for c, w in zip(candidate_ids, weights) if w > 0 and c in errors]
    coverage = math.fsum(w for _, w in known)
    error = math.fsum(w * errors[c] for c, w in known) / coverage if coverage else None
    return {'rank': rank, 'coverage': coverage, 'conditional_error': error,
            'gmfe': math.exp(error) if error is not None and not time_endpoint else None,
            'tmax_mae_h': error if time_endpoint else None,
            'within_twofold': (math.fsum(w * (errors[c] <= math.log(2)) for c, w in known) / coverage
                              if coverage and not time_endpoint else None),
            'measured_pool_regret': max(0., error - min(errors.values())) if error is not None else None,
            'tied_candidate_ids': [c for c, w in zip(candidate_ids, weights) if w > 0],
            'tie_size': int((weights > 0).sum()), 'measured_candidates': len(errors)}


def csv_out(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def score():
    lock = verify_prediction_lock()
    observations = read(OUT / 'observations.json')
    if observations['prediction_lock'] != ref(OUT / 'final_prediction_lock.json'):
        raise ValueError('Observations are not bound to this prediction lock')
    for source in lock['sources']:
        checked(source)
    for source in observations['sources']:
        checked(source)
    policy = read(OUT / 'scoring_protocol.json')
    for name, digest in policy['sources'].items():
        checked({'path': name, 'sha256': digest})
    candidates = {r['candidate_id']: r for r in map(json.loads, CATALOG.read_text().splitlines())}
    scenes = {s['scenario_id']: s for s in read(OUT / 'inputs/scenarios.json')}
    meta = read(OUT / 'rankings/manifest.json')
    arrays = dict(np.load(checked(meta['predictions']), allow_pickle=False))
    labels = {(r['scenario_id'], r['endpoint']): r for r in observations['observations']}
    if len(labels) != 20:
        raise ValueError('Unexpected observation membership')
    candidate_rows, by_scene = [], defaultdict(list)
    for task in lock['tasks']:
        sid, cid = task['scenario_id'], task['candidate_id']
        s = scenes[sid]
        predictions, unavailable, diagnostics = {}, {}, {}
        if task['qualified_for_accuracy_scoring'] and task['status'] == 'prediction_locked':
            result = read(checked(task['result']))
            prediction = read(checked({'path': result['prediction_file'], 'sha256': result['prediction_sha256']}))
            predictions, unavailable, diagnostics = derive(prediction, s)
        errors = {}
        for endpoint in s['planned_endpoint_definitions']:
            observed = labels.get((sid, endpoint))
            pred = predictions.get(endpoint)
            if observed and observed['value'] is not None and pred is not None:
                expected_unit = 'ng/mL' if endpoint == 'Cmax' else 'ng*h/mL' if endpoint == 'AUC_inf' else 'h'
                if observed['unit'] != expected_unit or observed['qualification'] != 'published_scalar_unit_and_analyte_matched':
                    raise ValueError('Unexpected observation qualification/unit')
                if pred < 0 or (pred == 0 and endpoint != 'Tmax'):
                    raise ValueError('Nonpositive predicted endpoint')
                error = abs(pred - observed['value']) if endpoint == 'Tmax' else abs(math.log(pred / observed['value']))
                errors[endpoint] = {'predicted': pred, 'observed': observed['value'], 'unit': expected_unit,
                    'statistic': observed['statistic'], 'absolute_error_h' if endpoint == 'Tmax' else 'absolute_log_error': error,
                    'gmfe': None if endpoint == 'Tmax' else math.exp(error)}
        row = {**task, 'compound': s['compound'], 'study_id': s['study_id'], 'endpoints': errors,
               'unavailable_endpoints': unavailable, 'nca_diagnostics': diagnostics,
               'interpretation': 'Reference-subject prediction against a reported group statistic; original source independence not established.'}
        candidate_rows.append(row); by_scene[sid].append(row)
    positions, omissions, membership = [], [], []
    for sid, s in scenes.items():
        tasks = by_scene[sid]
        eligible = tasks[0]['qualified_for_accuracy_scoring']
        stratum = tasks[0]['evaluation_stratum']
        measured = [r for r in tasks if r['status'] == 'prediction_locked' and eligible]
        families = {r['family_id'] for r in measured}
        membership.append({'scenario_id': sid, 'compound': s['compound'], 'study_id': s['study_id'],
            'stratum': stratum, 'candidate_versions': len(tasks), 'candidate_families': len({r['family_id'] for r in tasks}),
            'executed_versions': sum(r['status'] == 'prediction_locked' for r in tasks),
            'qualified_versions': len(measured), 'qualified_families': len(families),
            'independent_family_choice_possible': len(families) >= 2, 'accuracy_eligible': eligible})
        if not eligible:
            continue
        qi = meta['query_ids'].index(sid)
        for endpoint in s['planned_endpoint_definitions']:
            observation = labels.get((sid, endpoint))
            if not observation or observation['value'] is None:
                continue
            time_endpoint = endpoint == 'Tmax'
            errors = {r['candidate_id']: r['endpoints'][endpoint]['absolute_error_h' if time_endpoint else 'absolute_log_error']
                      for r in measured if endpoint in r['endpoints']}
            endpoint_families = {candidates[c]['family_id'] for c in errors}
            for method in meta['methods']:
                scores = arrays[method][qi]
                common = {'method': method, 'scenario_id': sid, 'compound': s['compound'],
                          'publication_cluster_id': s['study_id'], 'stratum': stratum, 'endpoint': endpoint}
                for rank in (1, 2, 3):
                    positions.append({**common, **position_row(scores, meta['candidate_ids'], errors, rank=rank, time_endpoint=time_endpoint)})
                if not time_endpoint:
                    miss = numerical_miss_audit(meta['candidate_ids'], s['catalog_candidate_ids'],
                        dict(zip(meta['candidate_ids'], map(float, scores))), errors, ks=(3, 5))
                    # Versions retain their exact original ranks. A single family
                    # cannot supply independent-family selection evidence.
                    omissions.append({**common, **miss, 'measured_families': len(endpoint_families),
                        'independent_family_choice_possible': len(endpoint_families) >= 2,
                        'family_level_accuracy': None,
                        'family_interpretation': 'No independently ranked family-level estimator was registered; version-level diagnostics are not family accuracy.'})
    summary = []
    keys = sorted({(r['method'], r['stratum'], r['endpoint'], r['rank']) for r in positions})
    for method, stratum, endpoint, rank in keys:
        rows = [r for r in positions if (r['method'], r['stratum'], r['endpoint'], r['rank']) == (method, stratum, endpoint, rank)]
        error = hierarchical_mean(rows, 'conditional_error')
        summary.append({'method': method, 'stratum': stratum, 'endpoint': endpoint, 'rank': rank,
            'scenarios': len(rows), 'studies': len({r['publication_cluster_id'] for r in rows}),
            'coverage': float(np.mean([r['coverage'] for r in rows])),
            'gmfe': math.exp(error) if error is not None and endpoint != 'Tmax' else None,
            'tmax_mae_h': error if endpoint == 'Tmax' else None,
            'within_twofold': hierarchical_mean(rows, 'within_twofold'),
            'measured_pool_regret': hierarchical_mean(rows, 'measured_pool_regret')})
    out = OUT / 'report'; out.mkdir(exist_ok=True)
    payload = {'schema': 'reviewer-new-study-evaluation-v1', 'state': 'complete', 'methods': meta['methods'],
        'membership': membership, 'native_status_counts': dict(Counter(r['status'] for r in candidate_rows)),
        'candidate_results': candidate_rows, 'positions': positions, 'summary': summary, 'version_omissions': omissions,
        'sources': [ref(OUT / 'final_prediction_lock.json'), ref(OUT / 'observations.json'),
                    ref(OUT / 'observation_verification.json'), ref(OUT / 'scoring_protocol.json'),
                    ref(OUT / 'candidate_source_audit.json'), ref(Path(__file__))],
        'scientific_limits': ['Only one newly selected, outcome-unseen study remains condition-eligible (pregabalin).',
            'Its two candidate versions belong to one family; no independent-family preference result.',
            'Original model development independence remains unproven; no blind external generalization claim.',
            'Exploratory cohorts retain prior result/metadata exposure and dosing assumptions separately.',
            'No parameter updates, rank changes, new LLM labels, outcome-selected exclusions or hypothesis tests.']}
    write_once(out / 'results.json', payload)
    csv_out(out / 'summary.csv', summary)
    csv_out(out / 'positions.csv', positions)
    csv_out(out / 'membership.csv', membership)
    print(json.dumps({'state': 'complete', 'tasks': len(candidate_rows), 'positions': len(positions),
                      'strict_new_study_results': [r for r in summary if r['stratum'] == 'strict_new_study_pre_outcome_selection' and r['rank'] == 1]}), flush=True)
    return payload


if __name__ == '__main__':
    score()
