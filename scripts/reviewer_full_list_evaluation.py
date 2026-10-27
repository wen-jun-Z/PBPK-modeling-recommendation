"""Report judged-subset ranking beside full-list automatic support, without refitting.

Unknown labels occupy their original ranks. These are descriptions of the locked
automatic labels, not expert correctness or clinical accuracy estimates.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from evaluate_frontier_automatic import rank_metrics
from pbpk_kb.recommendation.pbpk_match_v2.evaluation import exact_rank_weights

OUT = ROOT / 'experiments/model_recommendation/reviewer_revision_20261009/full_list'
CORE = ROOT / 'experiments/model_recommendation/frontier_core_145_20260921'
DATA = ROOT / 'experiments/model_recommendation/pbpk_match_v2'
LIGHT = ROOT / 'experiments/model_recommendation/pbpk_match_light'
COMP = ROOT / 'experiments/model_recommendation/component_integration_20261007'
TASK_GROUP = {'plasma_concentration_time_prediction': 'direct_prediction',
    'ddi_exposure_prediction': 'direct_prediction', 'special_population_prediction': 'direct_prediction',
    'mechanistic_template_recommendation': 'mechanism_reference',
    'compound_parameterization_template': 'parameterization_template'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_list_row(scores, grades, applicability):
    scores, grades, applicability = map(np.asarray, (scores, grades, applicability))
    if scores.shape != grades.shape or scores.ndim != 1 or applicability.shape != grades.shape:
        raise ValueError('Unaligned full-catalog rows')
    if not np.isfinite(scores).all() or not np.isin(grades, [-1, 0, 1, 2, 3]).all():
        raise ValueError('Invalid scores or grades')
    if not np.isin(applicability, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid applicability labels')
    weights = exact_rank_weights(scores, 1)
    masks = {'supported': grades >= 2, 'limited': grades == 1,
             'unsupported': grades == 0, 'unknown': grades < 0}
    result = {f'top1_{name}': float(weights[mask].sum()) for name, mask in masks.items()}
    result.update(top1_applicable=float(weights[applicability == 0].sum()),
                  top1_inapplicable=float(weights[applicability == 1].sum()),
                  top1_applicability_unknown=float(weights[(applicability == 2) | (applicability < 0)].sum()),
                  top1_tie_size=int((weights > 0).sum()))
    if not np.isclose(sum(result['top1_' + name] for name in masks), 1):
        raise ValueError('Full-list label states do not partition the first rank')
    for k in (3, 10):
        expected = sum((exact_rank_weights(scores, rank) for rank in range(1, min(k, len(scores)) + 1)))
        result[f'top{k}_supported_fraction'] = float(expected[grades >= 2].sum() / min(k, len(scores)))
        result[f'top{k}_known_fraction'] = float(expected[grades >= 0].sum() / min(k, len(scores)))
    known = grades >= 0
    result['judged_only'] = rank_metrics(scores[known], grades[known])
    result['known_positive_candidates'] = int((grades >= 2).sum())
    return result


def summarize(rows):
    eligible = [r['judged_only'] for r in rows if r['judged_only'] is not None]
    result = {'requests': len(rows), 'ranking_eligible_requests': len(eligible),
              'ranking_eligible_fraction': len(eligible) / len(rows) if rows else None}
    for metric in ('ndcg@10', 'recall@10', 'mrr'):
        result[metric] = float(np.mean([r[metric] for r in eligible])) if eligible else None
    fields = [k for k in rows[0] if k.startswith('top') and k != 'top1_tie_size'] if rows else []
    result.update({k: float(np.mean([r[k] for r in rows])) for k in fields})
    result['top1_tied_requests'] = sum(r['top1_tie_size'] > 1 for r in rows)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def build():
    sources = {}
    def bind(path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Input escapes project')
        sources[str(path.relative_to(ROOT))] = digest(path)
        return path
    def read(path):
        return json.loads(bind(path).read_text())
    raw = read(DATA / 'dataset.json')
    with np.load(bind(DATA / 'arrays.npz'), allow_pickle=False) as a:
        grades, apps = a['grades'].copy(), a['applicability'].copy()
    core = raw['core_indices']; queries = [raw['query_records'][i] for i in core]
    gids = [q['query_id'] for q in queries]; ids = [c['candidate_id'] for c in raw['candidate_records']]
    legacy = read(CORE / 'assessment_repaired_v2/automatic_evaluation.json')
    cq = [json.loads(line) for line in bind(CORE / 'inputs/queries.jsonl').read_text().splitlines()]
    cc = [json.loads(line) for line in bind(CORE / 'inputs/candidates.jsonl').read_text().splitlines()]
    if ids != [c['candidate_id'] for c in cc] or set(gids) != {q['query_id'] for q in cq}:
        raise ValueError('Core identities changed')
    indices = {q['query_id']: i for i, q in enumerate(cq)}
    labels = bind(CORE / 'assessment_repaired_v2/judgments.jsonl')
    if digest(labels) != legacy['assessment']['committed_judgments_sha256']:
        raise ValueError('Frozen judgments changed')
    lookup = {(r['query_id'], r['candidate_id']): r for r in map(json.loads, labels.read_text().splitlines())}
    if len(lookup) != 145 * 280:
        raise ValueError('Incomplete frozen judgment grid')
    for i, qi in enumerate(core):
        for j, cid in enumerate(ids):
            label = lookup[gids[i], cid]
            if grades[qi, j] != (-1 if label['grade'] is None else label['grade']):
                raise ValueError('Dataset and original grade labels disagree')
    methods = {}
    paper = read(ROOT / 'results/data/model_recommendation_quality.json')
    labels_by_method = {r['method_id']: r['method'] for r in paper['rows']}
    binary = {}
    for method, item in legacy['methods'].items():
        path = bind(CORE / item['progress']['effective_score_file'])
        if digest(path) != item['progress']['effective_scores_sha256']:
            raise ValueError('Frozen baseline scores changed')
        scores = np.load(path, allow_pickle=False)
        methods[method] = scores[0, [indices[q] for q in gids]]
        cal = item['variants']['original']['calibrated_applicability']
        conf = cal['confusion']
        binary[method] = {'requests': 102, 'known_pairs': cal['test_pairs'], 'positive_pairs': cal['positive_test_pairs'],
            **conf, 'sensitivity': conf['tp'] / (conf['tp'] + conf['fn']),
            'positive_precision': conf['tp'] / (conf['tp'] + conf['fp']) if conf['tp'] + conf['fp'] else None}
    metrics = read(ROOT / 'results/data/pbpk_match_light_paper_metrics.json')
    receipt = read(LIGHT / 'predictions/seed17.json')
    score_path = bind(LIGHT / 'predictions/seed17.npz')
    if digest(score_path) != receipt['score_sha256']:
        raise ValueError('Own frozen score binding changed')
    with np.load(score_path, allow_pickle=False) as a:
        for method, key in [('pbpk_match_light_document', 'base_scores'), ('pbpk_match_light', 'scores')]:
            methods[method] = a[key][0, core]
            labels_by_method[method] = 'PBPK-Match w/o error' if key == 'base_scores' else 'PBPK-Match'
            b = metrics['quality_details'][method]['binary_applicability_102']
            binary[method] = {'requests': 102, 'known_pairs': b['known_pairs'], 'positive_pairs': b['positive_pairs'],
                **b['confusion'], 'sensitivity': b['positive_recall'], 'positive_precision': b['positive_precision']}
    manifest = read(COMP / 'predictions/manifest.json')
    path = bind(COMP / 'predictions/component_scores.npz')
    if digest(path) != manifest['predictions_sha256'] or manifest['candidate_ids'] != ids:
        raise ValueError('Component scores changed')
    with np.load(path, allow_pickle=False) as a:
        methods['pbpk_match_components'] = a['scores'][[manifest['query_ids'].index(q) for q in gids]]
    labels_by_method['pbpk_match_components'] = 'PBPK-Match + components'
    comp = read(COMP / 'report/summary.json')['reports']['pbpk_match_components']['quality']['paper_binary_102']
    binary['pbpk_match_components'] = {'requests': 102, 'known_pairs': comp['known_pairs'], 'positive_pairs': comp['positive_pairs'],
        **comp['confusion'], 'sensitivity': comp['positive_recall'], 'positive_precision': comp['positive_precision']}
    per_query, aggregates = [], []
    for method, scores in methods.items():
        if scores.shape != (145, 280):
            raise ValueError('Every method must retain 145 x 280 scores')
        rows = [{'method_id': method, 'query_id': q['query_id'], 'task_type': q['task_type'],
                 'purpose': TASK_GROUP[q['task_type']], 'compound': q['compound_id'],
                 **full_list_row(scores[i], grades[core[i]], apps[core[i]])} for i, q in enumerate(queries)]
        per_query += rows
        for scope in ['all', *sorted(set(TASK_GROUP.values())), *sorted(TASK_GROUP)]:
            chosen = rows if scope == 'all' else [r for r in rows if scope in (r['purpose'], r['task_type'])]
            aggregates.append({'method_id': method, 'method': labels_by_method[method], 'scope': scope, **summarize(chosen)})
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'pbpk-reviewer-full-list-v1', 'state': 'complete', 'expert_validation': False,
        'new_inference_or_training': False, 'methods': len(methods), 'requests': 145, 'candidates': 280,
        'metric_definitions': {'judged_only': 'Unknowns removed only for this conditional ranking metric; grade>=2 positive.',
            'full_list': 'All requests and original candidate positions; grade>=2 support, grade1 limited, grade0 unsupported, -1 unknown.',
            'ties': 'Uniform exact-score rank probabilities, no ID/order tie-breaking.',
            'topk_fractions': 'Expected fraction at exact original ranks 1..k, not full-catalog recall and not boundary-inclusive coverage.',
            'support': 'Support under unvalidated automatic labels; not expert-confirmed correctness.',
            'binary': 'Original 102-request known Y/N denominator, U/invalid excluded; original calibration protocols retained.'},
        'rows': aggregates, 'binary_applicability': binary, 'per_query': per_query,
        'sources': sources, 'producer': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': digest(__file__)}}
    (OUT / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_csv(OUT / 'summary.csv', aggregates)
    write_csv(OUT / 'applicability.csv', [{'method_id': m, 'method': labels_by_method[m], **b} for m, b in binary.items()])
    write_csv(OUT / 'per_query.csv', [{k: v for k, v in r.items() if k != 'judged_only'} for r in per_query])
    print(json.dumps({'state': 'complete', 'methods': len(methods), 'rows': len(aggregates),
                      'own': [r for r in aggregates if r['method_id'] == 'pbpk_match_light' and r['scope'] in ('all', *set(TASK_GROUP.values()))]}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    build()
