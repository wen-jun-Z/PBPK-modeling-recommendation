"""Post-lock decision utility with a shared, frozen PBPK candidate pool.

Selection and selective reporting are different experiments. Neither aggregate
PK reference labels nor risk-coverage outcomes enter a construction decision.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .hundred_evaluation import _checked_evaluation, _reference_values, _prediction_value
from .purpose_refinement_evaluation import tolerant_risk_order

VERSION = 'decision-value-evaluation-1'
ENDPOINTS = ('clearance', 'vss', 'terminal_half_life', 'mrt')
METHODS = ('fixed_native', 'fixed_capacity', 'global_decision', 'local_risk_only', 'purpose_decision')
COVERAGES = (.2, .4, .6, .8, 1.)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def checked(ref, root):
    path = (root / ref['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact escaped project')
    data = path.read_bytes()
    if sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('Locked artifact changed: ' + ref['path'])
    return json.loads(data)


def _barrier(protocol, locked, root):
    cases = protocol['cases']
    n = protocol['expected_compounds']
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or len(cases) != n or len({c['case_id'] for c in cases}) != n:
        raise ValueError('Complete unique cohort required')
    if len({c['compound_id'] for c in cases}) != n or len({c['identity_smiles'] for c in cases}) != n:
        raise ValueError('Chemical identities must be distinct')
    case_by_id = {c['case_id']: c for c in cases}
    if tuple(protocol['methods']) != METHODS or tuple(protocol['endpoints']) != ENDPOINTS:
        raise ValueError('Prespecified methods and endpoints changed')
    expected = {(c['case_id'], ep, m) for c in cases for ep in ENDPOINTS for m in METHODS}
    records = locked.get('records', [])
    if locked.get('all_decisions_locked_before_scoring') is not True or len(records) != len(expected):
        raise ValueError('Every decision must lock before any evaluation is read')
    if locked.get('records_sha256') != digest(records):
        raise ValueError('Decision content changed')
    indexed = {}
    outcomes = {cid: checked(ref, root) for cid, ref in protocol['candidate_outcomes'].items()}
    if set(outcomes) != {c['case_id'] for c in cases}:
        raise ValueError('Candidate cohort changed')
    for row in records:
        key = (row['case_id'], row['endpoint'], row['method'])
        if key not in expected or key in indexed:
            raise ValueError('Duplicate or unknown decision')
        if (row['compound_id'] != case_by_id[row['case_id']]['compound_id']
                or row['whole_model_source'] != protocol['candidate_outcomes'][row['case_id']]
                or outcomes[row['case_id']]['case_id'] != row['case_id']):
            raise ValueError('Selected model or decision identity mismatch')
        selected = row['selected_candidate']
        if selected is not None:
            if selected not in {'parameter_only', 'combined'}:
                raise ValueError('Unknown PBPK candidate')
            prediction = outcomes[row['case_id']]['methods'][selected]
            if digest(prediction) != row['selected_prediction_sha256']:
                raise ValueError('Whole-model selection differs from its locked prediction')
            value, reason = _prediction_value(prediction, row['endpoint'])
            if value is None or row['prediction_value'] != value:
                raise ValueError('Selected primary endpoint differs from its whole model')
        elif row['prediction_value'] is not None or row.get('selected_prediction_sha256') is not None:
            raise ValueError('Missing model cannot supply a prediction')
        for field in ('risk', 'expected_log_error', 'ranking_score'):
            val = row.get(field)
            if val is not None and (isinstance(val, bool) or not math.isfinite(val) or val < 0):
                raise ValueError('Invalid decision risk or score')
        if row.get('risk') is not None and row['risk'] > 1:
            raise ValueError('Risk outside probability domain')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('Incomplete locked decisions')
    return cases, indexed


def metrics(rows, repeats=2000, seed=20261001):
    if not rows:
        return {'n': 0, 'gmfe': None, 'gmfe_ci95': None, 'within_twofold_n': 0, 'within_twofold_rate': None}
    errors = np.array([r['absolute_ln_error'] for r in rows])
    sample = np.random.default_rng(seed).integers(len(rows), size=(repeats, len(rows)))
    ci = np.quantile(errors[sample].mean(axis=1), [.025, .975])
    good = errors <= math.log(2) + 1e-12
    return {'n': len(rows), 'gmfe': float(np.exp(errors.mean())), 'gmfe_ci95': np.exp(ci).tolist(),
            'within_twofold_n': int(good.sum()), 'within_twofold_rate': float(good.mean()),
            'large_error_rate': float((~good).mean())}


def paired(left, right, repeats, seed):
    common = sorted(left.keys() & right.keys())
    if not common:
        return {'n': 0}
    a = np.array([left[c]['absolute_ln_error'] for c in common])
    b = np.array([right[c]['absolute_ln_error'] for c in common])
    change = b - a
    sample = np.random.default_rng(seed).integers(len(common), size=(repeats, len(common)))
    changed = [c for c in common if left[c]['selected_candidate'] != right[c]['selected_candidate']]
    return {'n': len(common), 'left_gmfe': float(np.exp(a.mean())), 'right_gmfe': float(np.exp(b.mean())),
            'gmfe_ratio_right_over_left': float(np.exp(change.mean())),
            'gmfe_ratio_ci95': np.exp(np.quantile(change[sample].mean(axis=1), [.025, .975])).tolist(),
            'changed_candidate_n': len(changed),
            'changed_improved_n': sum(right[c]['absolute_ln_error'] < left[c]['absolute_ln_error'] - 1e-8 for c in changed),
            'changed_worsened_n': sum(right[c]['absolute_ln_error'] > left[c]['absolute_ln_error'] + 1e-8 for c in changed),
            'all_improved_n': int((change < -1e-8).sum()), 'all_worsened_n': int((change > 1e-8).sum()),
            'material_fold_error_reduction_n': int((change < -math.log(1.01)).sum()),
            'material_fold_error_increase_n': int((change > math.log(1.01)).sum()),
            'material_change_definition': 'more than 1 percent change in individual fold error'}


def _risk(rows, registered):
    rr = [r for r in rows if r['risk'] is not None]
    if not rr:
        return {'n': 0, 'status': 'unassessed'}
    p = np.array([r['risk'] for r in rr]); y = np.array([not r['within_twofold'] for r in rr])
    flag = p >= .5
    bins, ece = [], 0.
    for i in range(5):
        mask = (p >= i / 5) & (p < (i + 1) / 5 if i < 4 else p <= 1)
        if mask.any():
            gap = abs(float(p[mask].mean() - y[mask].mean()))
            ece += mask.mean() * gap
            bins.append({'lower': i / 5, 'upper': (i + 1) / 5, 'n': int(mask.sum()),
                         'mean_probability': float(p[mask].mean()), 'actual_error_rate': float(y[mask].mean())})
    fp, fn = int((flag & ~y).sum()), int((~flag & y).sum())
    return {'n': len(rr), 'brier': float(((p - y) ** 2).mean()), 'ece_5_fixed_bins': float(ece),
            'calibration_bins': bins, 'review_threshold': .5, 'false_alarm_n': fp, 'missed_large_error_n': fn,
            'false_alarm_rate': fp / int((~y).sum()) if (~y).any() else None,
            'missed_large_error_rate': fn / int(y.sum()) if y.any() else None,
            'unflagged_scorable_n': int((~flag).sum()), 'unflagged_fraction_of_scorable': float((~flag).mean()),
            'unflagged_scorable_fraction_of_registered': float((~flag).sum() / registered),
            'clinical_reliability_claimed': False}


def _ranking(rows):
    by_id = {r['case_id']: r for r in rows}
    order = tolerant_risk_order({cid: row['ranking_score'] for cid, row in by_id.items()})
    return order, by_id


def ranking_pair(left, right, coverage, repeats, seed):
    """Paired compound bootstrap reselects the retained subset each resample."""
    ids = sorted(left.keys() & right.keys())
    if not ids:
        return {'n': 0}
    order_a, _ = _ranking([left[c] for c in ids]); order_b, _ = _ranking([right[c] for c in ids])
    rank_a = np.array([order_a.index(c) for c in ids]); rank_b = np.array([order_b.index(c) for c in ids])
    error_a = np.array([left[c]['absolute_ln_error'] for c in ids]); error_b = np.array([right[c]['absolute_ln_error'] for c in ids])
    k = max(1, math.ceil(len(ids) * coverage))
    sample = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    selected_a = np.take_along_axis(sample, np.argsort(rank_a[sample], axis=1, kind='stable')[:, :k], axis=1)
    selected_b = np.take_along_axis(sample, np.argsort(rank_b[sample], axis=1, kind='stable')[:, :k], axis=1)
    delta = error_b[selected_b].mean(axis=1) - error_a[selected_a].mean(axis=1)
    observed = error_b[np.argsort(rank_b)[:k]].mean() - error_a[np.argsort(rank_a)[:k]].mean()
    return {'n': len(ids), 'retained_n': k, 'coverage': coverage,
            'gmfe_ratio_right_over_left': float(np.exp(observed)),
            'gmfe_ratio_ci95': np.exp(np.quantile(delta, [.025, .975])).tolist(),
            'resamples_reselect_by_frozen_rank': True}


def evaluate_decisions(protocol, locked, root, *, repeats=2000, seed=20261001):
    root = Path(root).resolve()
    cases, index = _barrier(protocol, locked, root)
    refs = {c['case_id']: _reference_values(_checked_evaluation(c['evaluation'], root), c) for c in cases}
    records = []
    for case in cases:
        for ep in ENDPOINTS:
            reference = refs[case['case_id']].get(ep)
            for method in METHODS:
                row = deepcopy(index[case['case_id'], ep, method])
                value = row['prediction_value']
                error = abs(math.log(value / reference)) if value is not None and reference is not None else None
                row.update(drug_name=case['drug_name'], reference_value=reference,
                           absolute_ln_error=error, fold_error=math.exp(error) if error is not None else None,
                           within_twofold=error <= math.log(2) + 1e-12 if error is not None else None)
                records.append(row)
    summary = {}
    for ep in ENDPOINTS:
        rows = {m: [r for r in records if r['endpoint'] == ep and r['method'] == m] for m in METHODS}
        valid = {m: {r['case_id']: r for r in rr if r['absolute_ln_error'] is not None} for m, rr in rows.items()}
        common = set.intersection(*(set(v) for v in valid.values()))
        methods = {}
        for m in METHODS:
            comparable = [valid[m][c] for c in sorted(common)]
            order, byid = _ranking(comparable)
            all_order, all_byid = _ranking([r for r in rows[m] if r['prediction_value'] is not None])
            curves, registered_curves = [], []
            for fraction in COVERAGES:
                k = max(1, math.ceil(len(order) * fraction)) if order else 0
                keep = [byid[c] for c in order[:k]]
                curves.append({'requested_coverage': fraction, 'retained_n': k,
                    'denominator': 'all_methods_common_scorable', 'denominator_n': len(common),
                    'retained_case_ids': order[:k], **metrics(keep, repeats, seed)})
                # Operational ranking uses all available drugs, never truth eligibility.
                ka = min(len(all_order), math.ceil(len(cases) * fraction))
                retained = [all_byid[c] for c in all_order[:ka]]
                measured = [r for r in retained if r['absolute_ln_error'] is not None]
                registered_curves.append({'requested_registered_coverage': fraction, 'retained_n': ka,
                    'actual_registered_coverage': ka / len(cases), 'reference_scorable_n': len(measured),
                    'reference_missing_n': ka - len(measured), 'retained_case_ids': all_order[:ka],
                    'descriptive_metrics_on_available_references': metrics(measured, repeats, seed)})
            methods[m] = {'registered_n': len(cases), 'prediction_n': len(all_order),
                'unavailable_n': len(cases) - len(all_order),
                'selection_counts': dict(Counter(r['selected_candidate'] or 'none' for r in rows[m])),
                'own_scorable': metrics(list(valid[m].values()), repeats, seed),
                'common_scorable': metrics(comparable, repeats, seed), 'risk': _risk(comparable, len(cases)),
                'matched_coverage_curve': curves, 'registered_coverage_curve': registered_curves}
        pairs = [('fixed_capacity', 'purpose_decision'), ('local_risk_only', 'purpose_decision'),
                 ('fixed_capacity', 'local_risk_only'), ('global_decision', 'purpose_decision')]
        summary[ep] = {'registered_n': len(cases), 'reference_n': sum(ep in x for x in refs.values()),
            'common_scorable_n': len(common), 'methods': methods,
            'comparisons': {b + '_vs_' + a: {
                'selection': paired(valid[a], valid[b], repeats, seed),
                'matched_coverage': [ranking_pair({c: valid[a][c] for c in common}, {c: valid[b][c] for c in common}, f, repeats, seed) for f in COVERAGES]}
                for a, b in pairs}}
    return {'version': VERSION, 'all_decisions_locked_before_scoring': True,
        'registered_compounds': len(cases), 'requests': len(cases) * len(ENDPOINTS),
        'decision_records': len(records), 'endpoint_metrics': summary, 'records': records,
        'bootstrap': {'unit': 'compound', 'repeats': repeats, 'seed': seed, 'study_clustered': False},
        'clinical_curve_validation_n': 0, 'paper_updated': False,
        'limitations': ['Previously inspected development cohort; no prospective or clinical-safety claim.',
            'Local residual distributions are empirical forecasts, not guaranteed calibrated probabilities.',
            'Capacity improvement is shared with the fixed-capacity comparator and is not decision-layer benefit.',
            'Selection and selective reporting are evaluated separately; unknown-reference drugs remain registered.',
            'No evidence-acquisition intervention or prospective risk guarantee was tested.']}
