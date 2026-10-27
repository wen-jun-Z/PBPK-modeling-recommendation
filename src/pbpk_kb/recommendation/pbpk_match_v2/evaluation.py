"""Outcome-isolated, exact-position evaluation for the new supervised experiment.

This module never chooses training examples or changes a ranking using outcomes.
Clinical numbers describe the existing reference-person diagnostic estimand.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from pbpk_kb.recommendation.pbpk_match.evaluation import evaluate_ranking, compare_stability
from recommendation_execution_statistics import numerical_miss_audit, paired_method_statistics

PRIMARY = ROOT / 'experiments/model_recommendation/execution_validation_20261001/followup_20261003/primary/results.json'


def exact_rank_weights(scores, rank):
    """Probability of occupying exactly rank, not best-of-k or backfilled rank."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= rank <= len(scores):
        raise ValueError('Invalid scores or rank')
    level = np.sort(scores)[-rank]
    tied = scores == level
    return tied.astype(float) / tied.sum()


def hierarchical_mean(rows, field):
    study = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            study[(row['compound'], row['publication_cluster_id'])].append(float(row[field]))
    compounds = defaultdict(list)
    for (drug, _), values in study.items():
        compounds[drug].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in compounds.values()])) if compounds else None


def clinical_contract(path=PRIMARY):
    """Import frozen scientific matching decisions without reinterpreting them."""
    saved = json.loads(Path(path).read_text())
    cases = {}
    for selection in saved['selection']:
        if selection['stratum'] != 'reference_diagnostic':
            continue
        sid = selection['scenario_id']
        value = {'scenario_id': sid, 'compound': selection['compound'],
                 'publication_cluster_id': selection['publication_cluster_id'],
                 'endpoints': sorted(selection['required_endpoints']), 'same_parent_ids': [], 'errors': {}}
        if sid not in cases:
            cases[sid] = value
        elif cases[sid]['endpoints'] != value['endpoints']:
            raise ValueError('Methods disagree about requested endpoints')
    for row in saved['candidate_results']:
        case = cases[row['scenario_id']]
        cid = row['candidate_id']
        if row['identity_match']:
            case['same_parent_ids'].append(cid)
        errors = {}
        for endpoint, pair in row['reference_diagnostic']['endpoints'].items():
            if pair.get('predicted', 0) > 0 and pair.get('observed', 0) > 0:
                signed = math.log(pair['predicted']) - math.log(pair['observed'])
                errors[endpoint] = {'absolute': abs(signed), 'signed': signed}
        curves = [c for c in row['curves'] if c.get('state') == 'computed' and c.get('curve', {}).get('state') == 'computed']
        if curves:
            errors['curve_gmfe'] = {
                'absolute': float(np.mean([c['curve']['mean_absolute_log_error'] for c in curves])),
                'signed': float(np.mean([c['curve']['mean_signed_log_error'] for c in curves]))}
            if 'curve_gmfe' not in case['endpoints']:
                case['endpoints'].append('curve_gmfe')
        case['errors'][cid] = errors
    return cases


def clinical_evaluation(scores, query_ids, candidate_ids, *, method, contract=None):
    scores = np.asarray(scores)
    if scores.shape != (len(query_ids), len(candidate_ids)) or not np.isfinite(scores).all():
        raise ValueError('Clinical ranking is incomplete or not finite')
    contract = clinical_contract() if contract is None else contract
    rows, omissions = [], []
    for i, sid in enumerate(query_ids):
        case = contract[sid]
        ranking = dict(zip(candidate_ids, map(float, scores[i])))
        for endpoint in case['endpoints']:
            errors = {c: e[endpoint]['absolute'] for c, e in case['errors'].items() if endpoint in e}
            signed = {c: e[endpoint]['signed'] for c, e in case['errors'].items() if endpoint in e}
            for rank in (1, 2, 3):
                weights = exact_rank_weights(scores[i], rank)
                known = [(c, w) for c, w in zip(candidate_ids, weights) if w > 0 and c in errors]
                coverage = sum(w for c, w in known)
                value = sum(w * errors[c] for c, w in known) / coverage if coverage else None
                rows.append({'method': method, 'scenario_id': sid, 'compound': case['compound'],
                    'publication_cluster_id': case['publication_cluster_id'], 'endpoint': endpoint,
                    'stratum': 'reference_diagnostic', 'rank': rank, 'coverage': coverage,
                    'absolute_log_error': value,
                    'signed_log_error': sum(w * signed[c] for c, w in known) / coverage if coverage else None,
                    'within_twofold': sum(w * (errors[c] <= math.log(2)) for c, w in known) / coverage if coverage else None,
                    'within_1_5_fold': sum(w * (errors[c] <= math.log(1.5)) for c, w in known) / coverage if coverage else None,
                    'regret': max(0., value - min(errors.values())) if value is not None else None,
                    'top1_absolute_log_error': value, 'top1_evaluable_probability': coverage,
                    'tie_size': int((weights > 0).sum()), 'measured_candidates': len(errors)})
            omission = numerical_miss_audit(candidate_ids, case['same_parent_ids'], ranking, errors)
            omissions.append({'method': method, 'scenario_id': sid, 'compound': case['compound'],
                'publication_cluster_id': case['publication_cluster_id'], 'endpoint': endpoint, **omission})
    summaries = []
    for endpoint in sorted({r['endpoint'] for r in rows}):
        for rank in (1, 2, 3):
            chosen = [r for r in rows if r['endpoint'] == endpoint and r['rank'] == rank]
            mean = hierarchical_mean(chosen, 'absolute_log_error')
            summaries.append({'method': method, 'endpoint': endpoint, 'rank': rank,
                'scenarios': len(chosen), 'covered_scenarios': sum(r['coverage'] > 0 for r in chosen),
                'coverage': float(np.mean([r['coverage'] for r in chosen])),
                'gmfe': math.exp(mean) if mean is not None else None,
                'within_twofold': hierarchical_mean(chosen, 'within_twofold'),
                'regret': hierarchical_mean(chosen, 'regret')})
    return {'position_rows': rows, 'position_summary': summaries, 'omissions': omissions,
            'interpretation': 'Existing reference-person numerical diagnostics; not newly validated direct-reuse labels.'}


def quality_evaluation(data, scores, probabilities=None, indices=None):
    indices = list(data['core_indices'] if indices is None else indices)
    return evaluate_ranking(np.asarray(scores)[indices], np.asarray(data['grades'])[indices],
        np.asarray(data['applicability'])[indices], [data['query_records'][i] for i in indices],
        data['candidate_records'], None if probabilities is None else np.asarray(probabilities)[indices])


def stability_evaluation(data, scores):
    scores = np.asarray(scores)
    indices = data['stability_indices']
    changed = [data['query_records'][i].get('variants', {}).get('missing_evidence', {}).get('text_changed',
        data['query_texts'][0][i] != data['query_texts'][2][i]) for i in indices]
    base = scores[0, indices]
    return {'reordered': compare_stability(base, scores[1, indices], 'reordered'),
            'missing_evidence': compare_stability(base, scores[2, indices], 'missing_evidence', changed),
            'top1_tie_fraction': float(np.mean((base == base.max(axis=1, keepdims=True)).sum(axis=1) > 1))}


def fit_oof_temperature(probabilities, labels, groups):
    """Calibrate exclusively from pooled inner out-of-fold predictions."""
    p, y = np.asarray(probabilities, float), np.asarray(labels, int)
    _validate_probabilities(p)
    if p.shape[:-1] != y.shape or y.ndim != 2 or len(groups) != len(y) or not np.isin(y, [-1, 0, 1, 2]).all():
        raise ValueError('Calibration labels, probabilities or group lengths are invalid')
    valid = y >= 0
    support = [int((y == i).sum()) for i in range(3)]
    if not all(support):
        return {'state': 'unavailable', 'temperature': 1., 'support': support,
                'reason': 'Calibration requires observed Y, N and U'}
    weights = np.zeros(y.shape)
    groups = np.asarray(groups)
    for group in sorted(set(groups.tolist())):
        ids = np.flatnonzero(groups == group)
        ids = [i for i in ids if valid[i].any()]
        for i in ids:
            weights[i, valid[i]] = 1 / len(ids) / int(valid[i].sum())
    weights /= weights.sum()
    logp = np.log(np.clip(p, 1e-12, 1))
    grid = np.exp(np.linspace(np.log(.1), np.log(10), 161))
    losses = []
    for t in grid:
        z = logp[valid] / t
        z -= z.max(axis=-1, keepdims=True)
        nll = np.log(np.exp(z).sum(axis=-1)) - z[np.arange(valid.sum()), y[valid]]
        losses.append(float(np.sum(weights[valid] * nll)))
    return {'state': 'fitted', 'temperature': float(grid[np.argmin(losses)]),
            'support': support, 'fit_groups': sorted(set(groups.tolist())),
            'provenance': 'inner out-of-fold predictions only'}


def _validate_probabilities(p):
    if p.ndim < 2 or p.shape[-1] != 3 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=-1), 1, atol=1e-5):
        raise ValueError('Expected finite normalized Y/N/U probabilities')


def calibrate(probabilities, temperature):
    p = np.asarray(probabilities, float)
    _validate_probabilities(p)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    z = np.log(np.clip(p, 1e-12, 1)) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def paired_quality(reports, query_groups, *, baseline, draws=2000):
    """Paired group bootstrap; independent of neural training seeds."""
    rng = np.random.default_rng(20261004)
    result = []
    base = {r['query_id']: r for r in reports[baseline]['per_query']}
    for method, report in reports.items():
        if method == baseline:
            continue
        for metric in ['ndcg@10', 'recall@10', 'mrr']:
            groups = defaultdict(list)
            for r in report['per_query']:
                b = base.get(r['query_id'])
                if r['judged_only'] is not None and b is not None and b['judged_only'] is not None:
                    groups[query_groups[r['query_id']]].append(r['judged_only'][metric] - b['judged_only'][metric])
            values = np.asarray([np.mean(v) for v in groups.values()])
            ci = np.quantile(rng.choice(values, (draws, len(values))).mean(axis=1), [.025, .975]).tolist() if len(values) >= 2 else None
            null = (rng.choice([-1., 1.], (draws, len(values))) * values).mean(axis=1) if len(values) >= 2 else None
            pvalue = float((1 + np.sum(np.abs(null) >= abs(values.mean()) - 1e-12)) / (draws + 1)) if null is not None else None
            result.append({'method': method, 'baseline': baseline, 'metric': metric,
                'independent_groups': len(values), 'mean_group_delta': float(values.mean()) if len(values) else None,
                'bootstrap_95ci': ci, 'cluster_sign_flip_p': pvalue,
                'interpretation': 'Internal grouped comparison; not external validation'})
    for metric in ['ndcg@10', 'recall@10', 'mrr']:
        family = sorted([r for r in result if r['metric'] == metric and r['cluster_sign_flip_p'] is not None],
                        key=lambda r: r['cluster_sign_flip_p'])
        previous = 0.
        for index, row in enumerate(family):
            previous = max(previous, min(1., (len(family) - index) * row['cluster_sign_flip_p']))
            row['holm_p'] = previous
            row['holm_family'] = f'all reported methods versus {baseline}, {metric}'
    return result
