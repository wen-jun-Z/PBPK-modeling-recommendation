#!/usr/bin/env python3
"""Extend the locked curve illustration with held-out PBPK-Match rankings.

No PBPK simulation, candidate parameter fit, reranker inference, observation
selection or change to the original twelve methods occurs here. Curve metrics
are descriptive: both scenarios originate in the same publication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SOURCE = ROOT / 'results/data/model_recommendation_curve_cases.json'
SOURCE_SHA = 'e5982a64918d2badd108790deb5f474343b8090f4ea340c2f029b6bc0a28a5ab'
DEST = ROOT / 'results/data/pbpk_match_light_curves.json'
OWN = [('pbpk_match_light', 'PBPK-Match', 'scores'),
       ('pbpk_match_light_document', 'PBPK-Match (document)', 'base_scores')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve_errors(case, curve):
    """Interpolate the locked native curve at every qualified observed time."""
    t = np.asarray(curve['time_h'], dtype=float)
    y = np.asarray(curve['concentration_ng_ml'], dtype=float)
    ot = np.asarray(case['observation']['time_h'], dtype=float)
    oy = np.asarray(case['observation']['concentration_ng_ml'], dtype=float)
    if not (np.all(np.diff(t) > 0) and np.all(np.diff(ot) > 0)
            and t[0] <= ot[0] <= ot[-1] <= t[-1]
            and np.isfinite(y).all() and np.all(oy > 0)):
        raise ValueError('Curve grid, observation qualification or units changed')
    predicted = np.interp(ot, t, y)
    if not np.all(predicted > 0):
        raise ValueError('Nonpositive prediction cannot be used in a log ratio')
    errors = np.log(predicted / oy)
    gmfe = math.exp(float(np.abs(errors).mean()))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    if not (np.isclose(gmfe, curve['gmfe'], rtol=1e-10)
            and np.isclose(rmse, curve['log_rmse'], rtol=1e-10)):
        raise ValueError('Recomputed concentration-curve error differs from the locked result')
    return {'mean_abs_log_error': float(np.abs(errors).mean()),
            'mean_squared_log_error': float(np.mean(errors ** 2)),
            'within_twofold_fraction': float(np.mean(np.abs(errors) <= math.log(2))),
            'qualified_point_count': len(errors), 'curve_gmfe': gmfe, 'curve_log_rmse': rmse}


def positions(scores, candidate_ids, candidates, case):
    """Unfiltered exact ranks, with uniform expectations within exact-score ties."""
    scores = np.asarray(scores)
    if scores.shape != (len(candidate_ids),) or not np.isfinite(scores).all():
        raise ValueError('All registered candidates must have finite scores')
    labels = {c['candidate_id']: c['label'] for c in case['curves']}
    ordered = np.sort(scores)[::-1]
    result = []
    for rank in (1, 2, 3):
        score = ordered[rank - 1]
        indices = np.flatnonzero(scores == score)
        members = []
        for index in sorted(indices, key=lambda i: candidate_ids[i]):
            cid = candidate_ids[index]
            candidate = candidates[cid]
            members.append({'candidate_id': cid, 'curve_label': labels.get(cid),
                'weight': 1 / len(indices), 'name': candidate['family_name'],
                'family_id': candidate['family_id'], 'compounds': candidate['compounds'],
                'identity_match': case['compound'] in candidate['compounds']})
        result.append({'position': rank, 'score': float(score), 'tie_group_size': len(indices),
            'tie_start_rank': int(np.sum(scores > score)) + 1,
            'tie_end_rank': int(np.sum(scores >= score)), 'members': members,
            'curve_coverage': math.fsum(m['weight'] for m in members if m['curve_label'])})
    return result


def score_position(case, ranking, position, metrics):
    members = [m for m in position['members'] if m['curve_label']]
    coverage = math.fsum(m['weight'] for m in members)
    if not math.isclose(coverage, position['curve_coverage'], abs_tol=1e-12):
        raise ValueError('Curve coverage does not equal the measured tie probability')
    means = {}
    for key in ('mean_abs_log_error', 'mean_squared_log_error', 'within_twofold_fraction'):
        means[key] = (math.fsum(m['weight'] * metrics[m['candidate_id']][key] for m in members) / coverage
                      if coverage else None)
    return {'method_id': ranking['method_id'], 'method': ranking['method'],
        'scenario_id': case['scenario_id'], 'compound': case['compound'],
        'publication_cluster_id': case['publication_cluster_id'],
        'position': position['position'], 'qualified_point_count': len(case['observation']['time_h']),
        'curve_coverage': coverage, 'tie_group_size': position['tie_group_size'],
        'curve_labels': [m['curve_label'] for m in members],
        'measured_candidates': [{'candidate_id': m['candidate_id'], 'curve_label': m['curve_label'],
                                  'weight': m['weight']} for m in members],
        'curve_gmfe': math.exp(means['mean_abs_log_error']) if coverage else None,
        'curve_log_rmse': math.sqrt(means['mean_squared_log_error']) if coverage else None, **means}


def export():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('The original frozen curve export changed')
    source = json.loads(SOURCE.read_text())
    from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
    from summarize_pbpk_match_light import load_inputs
    data = load_dataset()
    protocol, _, arrays, refs, _, _ = load_inputs(data)
    candidates = {c['candidate_id']: c for c in data['candidate_records']}
    if len(candidates) != 280:
        raise ValueError('Candidate catalogue differs')
    query_ids, candidate_ids = protocol['query_ids'], protocol['candidate_ids']
    cases, rows, all_metrics = [], [], []
    for frozen_case in source['cases']:
        case = json.loads(json.dumps(frozen_case))
        qi = query_ids.index(case['scenario_id'])
        fold_ids = [i for i, f in enumerate(protocol['folds']) if qi in f['outer_test']]
        if len(fold_ids) != 1:
            raise ValueError('Curve request lacks a unique outer held-out fold')
        fold = protocol['folds'][fold_ids[0]]
        if any(protocol['groups'][i] == protocol['groups'][qi] for i in fold['outer_train']):
            raise ValueError('The held-out curve research group appears in training')
        metrics = {c['candidate_id']: curve_errors(case, c) for c in case['curves']}
        for c in case['curves']:
            all_metrics.append({'scenario_id': case['scenario_id'], 'compound': case['compound'],
                                'candidate_id': c['candidate_id'], 'curve_label': c['label'],
                                **metrics[c['candidate_id']]})
        for method_id, display, array in OWN:
            case['rankings'].append({'method_id': method_id, 'method': display,
                'outer_fold': fold_ids[0], 'positions': positions(arrays[array][0, qi], candidate_ids, candidates, case)})
        for ranking in case['rankings']:
            for position in ranking['positions']:
                rows.append(score_position(case, ranking, position, metrics))
        if case['rankings'][:-len(OWN)] != frozen_case['rankings'] or case['curves'] != frozen_case['curves']:
            raise ValueError('Original method rankings or native curves changed')
        cases.append(case)
    summary = []
    for ranking in cases[0]['rankings']:
        for rank in (1, 2, 3):
            selected = [r for r in rows if r['method_id'] == ranking['method_id'] and r['position'] == rank]
            measured = [r for r in selected if r['curve_coverage'] > 0]
            summary.append({'method_id': ranking['method_id'], 'method': ranking['method'], 'position': rank,
                'scenario_count': len(selected), 'scenarios_with_any_curve': len(measured),
                'curve_coverage': float(np.mean([r['curve_coverage'] for r in selected])),
                'curve_gmfe': math.exp(float(np.mean([r['mean_abs_log_error'] for r in measured]))) if measured else None,
                'within_twofold_fraction': float(np.mean([r['within_twofold_fraction'] for r in measured])) if measured else None,
                'aggregation': 'Equal weight per available scenario after conditional tie expectation; descriptive only'})
    # Both dose/drug branches must share one held-out research group.
    if len({protocol['groups'][query_ids.index(c['scenario_id'])] for c in cases}) != 1:
        raise ValueError('Shared-publication curve scenarios were split across groups')
    return {'schema': 'pbpk-match-light-locked-curves-v1',
        'source': {'path': str(SOURCE.relative_to(ROOT)), 'sha256': SOURCE_SHA},
        'own_methods': [{'method_id': x[0], 'method': x[1]} for x in OWN],
        'cases': cases, 'candidate_metrics': all_metrics, 'position_rows': rows, 'position_summary': summary,
        'validation': {'all_original_rankings_preserved': True, 'all_original_native_curves_preserved': True,
            'all_280_candidates_scored': True, 'outer_held_out_predictions_verified': True,
            'shared_study_group_held_out': True, 'independent_publications': 1, 'scenarios': 2,
            'candidate_curves': 6, 'unique_qualified_observations': 43,
            'rank_position_groups': len(rows), 'parameters_refitted': False, 'models_reexecuted': False},
        'metric_definitions': {'curve_gmfe': 'exp(mean_t(abs(ln(Cpred(t)/Cobs(t)))))',
            'ties': 'Uniform exact-score tie expectation at the requested unfiltered rank; unavailable curves reduce coverage; no backfill.',
            'within_twofold_fraction': 'Fraction of qualified positive observed concentration time points with predicted/observed in [0.5,2].',
            'summary': 'Equal-case geometric mean of available conditional curve GMFEs, accompanied by per-case results and coverage; not a generalization estimate.'},
        'limitations': source['limitations'] + ['No confidence interval or significance claim is supported by one publication.'],
        'provenance': refs + [{'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha(__file__)}]}


def encoded_csv(rows):
    fields = ['method_id', 'method', 'compound', 'scenario_id', 'publication_cluster_id', 'position',
              'qualified_point_count', 'curve_coverage', 'tie_group_size', 'curve_labels',
              'curve_gmfe', 'curve_log_rmse', 'within_twofold_fraction']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows([{**row, 'curve_labels': '|'.join(row['curve_labels'])} for row in rows])
    return stream.getvalue().encode()


def encoded_table(payload):
    from model_recommendation_paper import METHODS
    methods = [(mid, label) for mid, label, _ in METHODS] + [(x[0], x[1]) for x in OWN]
    indexed = {(r['method_id'], r['compound'], r['position']): r for r in payload['position_rows']}
    lines = [r'\begingroup', r'\fontsize{8.5}{10.5}\selectfont', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}', r'\toprule',
        r'Method & \multicolumn{3}{c}{Atorvastatin (22 points)} & \multicolumn{3}{c}{Rosuvastatin (21 points)} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
        r' & Rank 1 & Rank 2 & Rank 3 & Rank 1 & Rank 2 & Rank 3 \\', r'\midrule']
    for mid, label in methods:
        if mid == OWN[0][0]:
            lines.append(r'\midrule')
        values = []
        for compound in ('atorvastatin', 'rosuvastatin'):
            for rank in (1, 2, 3):
                row = indexed[mid, compound, rank]
                value = '--' if row['curve_gmfe'] is None else f"{row['curve_gmfe']:.3f}"
                if 0 < row['curve_coverage'] < 1:
                    if not math.isclose(row['curve_coverage'], .2):
                        raise ValueError('An additional partial-coverage annotation is required')
                    value += r'\textsuperscript{a}'
                elif row['tie_group_size'] > 1 and row['curve_gmfe'] is not None:
                    value += r'\textsuperscript{b}'
                values.append(value)
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup', '']
    return '\n'.join(lines).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    payload = export()
    files = {DEST: (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode(),
             DEST.with_suffix('.csv'): encoded_csv(payload['position_rows']),
             ROOT / 'results/tables/model_recommendation_curve_quality.csv': encoded_csv(payload['position_rows']),
             ROOT / 'results/tables/model_recommendation_curve_quality.tex': encoded_table(payload)}
    for path, content in files.items():
        if args.check:
            if path.read_bytes() != content:
                raise ValueError('Curve extension differs from verified inputs: ' + str(path))
        else:
            path.write_bytes(content)
    print(json.dumps({'checked': args.check, 'output': str(DEST.relative_to(ROOT)), **payload['validation']}))


if __name__ == '__main__':
    main()
