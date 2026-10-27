"""Prespecified numerical omission diagnostics and paired, clustered comparisons.

No I/O or scientific applicability labels. Coverage-conditioned numerical findings
must never be relabeled as validated-direct recall or a whole-catalog winner.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import math
import random

import recommendation_execution_metrics as metrics

VERSION = 'pbpk-execution-statistics-v1'
SEED = 20261001
DRAWS = 2000
MIN_BLOCKS = 10


def numerical_miss_audit(expected_ids, same_parent_ids, ranking, absolute_log_errors, *, ks=(1, 3, 5)):
    """Use the complete ranking; unknown same-parent candidates remain unknown.

    Numeric thresholds are exploratory screens, not scientific tier labels.
    Scope-excluded cross-parent IDs have no direct parent-PK outcome but still
    consume their actual positions in the complete ranked list.
    """
    expected, scope = list(expected_ids), set(same_parent_ids)
    if not expected or len(expected) != len(set(expected)) or not scope.issubset(expected):
        raise ValueError('Invalid complete pool or same-parent scope')
    if not set(absolute_log_errors).issubset(scope) or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
            for v in absolute_log_errors.values()):
        raise ValueError('Numerical errors must be finite and bound to same-parent candidates')
    if any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('Top-k values must be positive integers')
    groups = metrics._tie_groups(expected, ranking)
    unknown = scope.difference(absolute_log_errors)
    best_error = min(absolute_log_errors.values(), default=None)
    best = {c for c, v in absolute_log_errors.items() if v == best_error}
    result = {'state': 'computed' if absolute_log_errors else 'unavailable',
              'same_parent_count': len(scope), 'measured_count': len(absolute_log_errors),
              'unmeasured_same_parent_count': len(unknown), 'unknown_candidate_ids': sorted(unknown),
              'complete_same_parent_numeric_pool': not unknown,
              'scientific_applicability_recall': None,
              'interpretation': 'Exploratory numerical screen on the measured parent-PK pool; not three-tier truth.',
              'best_measured_candidate_ids': sorted(best), 'best_measured_absolute_log_error': best_error,
              'choice_comparison_possible': len(absolute_log_errors) >= 2,
              'by_k': {}}
    for k in ks:
        weights = metrics._inclusion(groups, k)
        entry = {'best_measured_hit_probability': metrics._hit(groups, k, best) if len(absolute_log_errors) >= 2 else None,
                 'best_measured_miss_probability': 1 - metrics._hit(groups, k, best) if len(absolute_log_errors) >= 2 else None,
                 'thresholds': {}}
        for threshold in (1.5, 2.0):
            positives = {c for c, v in absolute_log_errors.items() if v <= math.log(threshold)}
            included = math.fsum(weights[c] for c in positives)
            recall = included / len(positives) if positives else None
            entry['thresholds'][str(threshold)] = {
                'measured_numeric_positive_count': len(positives),
                'measured_numeric_positive_ids': sorted(positives),
                'known_numeric_recall': recall,
                'exact_same_parent_numeric_recall': recall if not unknown else None,
                'recall_bounds_allowing_all_unknown_completions': metrics._recall_bounds(weights, positives, unknown),
                'known_numeric_hit_probability': metrics._hit(groups, k, positives) if positives else None,
                'expected_missed_known_numeric_positives': len(positives) - included,
                'missed_or_partly_missed_known_candidates': [
                    {'candidate_id': c, 'exclusion_probability': 1 - weights[c]}
                    for c in sorted(positives) if weights[c] < 1],
                'scientific_tier_label': False}
        result['by_k'][str(k)] = entry
    return result


def _blocks(compound_publications):
    """Join compounds sharing a publication, so the same study is never resampled independently."""
    parents = {c: c for c in compound_publications}
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    first = {}
    for compound, publications in sorted(compound_publications.items()):
        for publication in publications:
            if publication in first:
                parents[find(compound)] = find(first[publication])
            else:
                first[publication] = compound
    blocks = defaultdict(list)
    for compound in sorted(parents):
        blocks[find(compound)].append(compound)
    return list(blocks.values())


def _quantile(sorted_values, q):
    index = (len(sorted_values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    return sorted_values[lo] * (hi - index) + sorted_values[hi] * (index - lo) if hi != lo else sorted_values[lo]


def paired_method_statistics(rows, methods, *, baseline='bm25_text', draws=DRAWS, minimum_blocks=MIN_BLOCKS):
    """Per-endpoint method-minus-BM25 log-error differences on shared full coverage.

    Average cases within publication/compound, then publications within compound.
    Bootstrap connected compound/publication blocks. Sign-flip tests assume
    exchangeability of method/baseline paired differences at that block level.
    Holm includes the whole prespecified method family (unavailable tests p=1).
    """
    methods = list(methods)
    if baseline not in methods or len(methods) != len(set(methods)) or draws < 100 or minimum_blocks < 2:
        raise ValueError('Invalid registered methods or resampling policy')
    grouped = defaultdict(dict)
    for row in rows:
        if row['method'] not in methods:
            raise ValueError('Unregistered method in statistical input')
        key = (row['stratum'], row['endpoint'])
        rid = (row['method'], row['scenario_id'])
        if rid in grouped[key]:
            raise ValueError('Duplicate method/scenario/endpoint')
        grouped[key][rid] = row
    results = []
    for (stratum, endpoint), data in sorted(grouped.items()):
        sids = sorted({sid for _, sid in data})
        family = []
        for method in methods:
            if method == baseline:
                continue
            study = defaultdict(list)
            publications = defaultdict(set)
            shared = []
            excluded = []
            for sid in sids:
                a, b = data.get((method, sid)), data.get((baseline, sid))
                valid = a is not None and b is not None and all(
                    isinstance(r.get('top1_evaluable_probability'), (int, float)) and
                    math.isclose(r['top1_evaluable_probability'], 1.0, rel_tol=0., abs_tol=1e-12) and
                    isinstance(r.get('top1_absolute_log_error'), (int, float)) and
                    math.isfinite(r['top1_absolute_log_error']) and r['top1_absolute_log_error'] >= 0 and
                    r.get('compound') and r.get('publication_cluster_id') for r in (a, b))
                if not valid:
                    excluded.append(sid)
                    continue
                if a['compound'] != b['compound'] or a['publication_cluster_id'] != b['publication_cluster_id']:
                    raise ValueError('Method rows disagree on scenario identity')
                compound, publication = a['compound'], a['publication_cluster_id']
                study[(compound, publication)].append(a['top1_absolute_log_error'] - b['top1_absolute_log_error'])
                publications[compound].add(publication)
                shared.append(sid)
            by_compound = defaultdict(list)
            for (compound, _), values in study.items():
                by_compound[compound].append(math.fsum(values) / len(values))
            differences = {c: math.fsum(values) / len(values) for c, values in by_compound.items()}
            blocks = _blocks(publications)
            means = [differences[c] for c in sorted(differences)]
            estimate = math.fsum(means) / len(means) if means else None
            row = {'method': method, 'baseline': baseline, 'stratum': stratum, 'endpoint': endpoint,
                   'state': 'unavailable', 'reason': 'insufficient_shared_fully_observed_independent_blocks',
                   'shared_full_coverage_scenarios': shared, 'excluded_scenarios': excluded,
                   'eligible_scenario_fraction': len(shared) / len(sids) if sids else None,
                   'compounds': len(differences), 'independent_compound_publication_blocks': len(blocks),
                   'minimum_blocks_for_inference': minimum_blocks,
                   'mean_paired_absolute_log_error_difference': estimate,
                   'bootstrap_95ci': None, 'two_sided_p': None, 'holm_adjusted_p': None,
                   'holm_prespecified_family_size': len(methods) - 1,
                   'lower_difference_is_better': True, 'generalization': 'Shared fully observed subset only; no whole-pool winner claim.'}
            if len(blocks) >= minimum_blocks:
                seed = SEED + int(hashlib.sha256((stratum + '\n' + endpoint + '\n' + method).encode()).hexdigest()[:16], 16)
                rng = random.Random(seed)
                block_sums = [math.fsum(differences[c] for c in block) for block in blocks]
                block_n = [len(block) for block in blocks]
                bootstrap = []
                for _ in range(draws):
                    chosen = [rng.randrange(len(blocks)) for _ in blocks]
                    bootstrap.append(math.fsum(block_sums[i] for i in chosen) / sum(block_n[i] for i in chosen))
                bootstrap.sort()
                if len(blocks) <= 16:
                    permutations = itertools.product((-1, 1), repeat=len(blocks))
                    total = 2 ** len(blocks)
                    extreme = sum(abs(math.fsum(sign * value for sign, value in zip(signs, block_sums)) / len(differences)) >= abs(estimate) - 1e-14
                                  for signs in permutations)
                    p = extreme / total
                    permutation_kind = 'exact_block_sign_flip'
                else:
                    extreme = sum(abs(math.fsum(rng.choice((-1, 1)) * value for value in block_sums) / len(differences)) >= abs(estimate) - 1e-14
                                  for _ in range(draws))
                    p = (extreme + 1) / (draws + 1)
                    permutation_kind = 'monte_carlo_block_sign_flip_plus_one'
                row.update(state='computed', reason=None, bootstrap_95ci=[_quantile(bootstrap, .025), _quantile(bootstrap, .975)],
                           two_sided_p=p, bootstrap_draws=draws, seed=seed, permutation=permutation_kind)
            family.append(row)
        previous = 0.
        for rank, row in enumerate(sorted(family, key=lambda r: r['two_sided_p'] if r['two_sided_p'] is not None else 1.)):
            p = row['two_sided_p'] if row['two_sided_p'] is not None else 1.
            previous = max(previous, min(1., (len(family) - rank) * p))
            if row['two_sided_p'] is not None:
                row['holm_adjusted_p'] = previous
        results.extend(family)
    return {'version': VERSION, 'baseline': baseline, 'comparisons': results,
            'interpretation': 'Paired numerical differences within shared full coverage; reference diagnostics do not validate direct reuse.',
            'bootstrap_cluster': 'compounds; compounds sharing a publication are joined before resampling',
            'sign_flip_assumption': 'Paired method/baseline differences are exchangeable under the null at the joined-block level.',
            'multiplicity': 'Holm over all registered methods versus BM25 within each stratum/endpoint, unavailable p treated as 1.'}
