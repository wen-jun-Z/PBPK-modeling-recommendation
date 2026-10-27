"""Export accepted PBPK execution results for the paper without simulation or rescoring.

Only completed result artifacts are read. Clinical source references are retained,
but their sealed files are never opened. Main and post-scoring supplements remain
separate; candidate versions and alternative protocols are not independent trials.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from model_recommendation_paper import METHODS

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('experiments/model_recommendation/execution_validation_20261001')
FOLLOWUP = EXPERIMENT / 'followup_20261003'
CMAX = 'Cmax_reference_vs_group'
POSITION_ROLE = 'post-hoc rank-position diagnostic requested after main scoring'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    if actual is None or expected is None:
        require(actual is expected, f'{context}: null mismatch')
    else:
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                f'{context}: {actual} != {expected}')


def average(values):
    return math.fsum(values) / len(values) if values else None


def hierarchical(rows):
    """Equal cases/publication, equal publications/compound, equal compounds."""
    groups = defaultdict(list)
    for compound, publication, value in rows:
        groups[compound, publication].append(value)
    compounds = defaultdict(list)
    for (compound, _), values in groups.items():
        compounds[compound].append(average(values))
    return average([average(values) for values in compounds.values()])


def top5_miss(scores, best_indices):
    """Probability that uniform tie breaking misses every best measured candidate."""
    threshold = sorted(scores, reverse=True)[4]
    if any(scores[i] > threshold for i in best_indices):
        return 0.0
    block = [i for i, value in enumerate(scores) if value == threshold]
    draws = 5 - sum(value > threshold for value in scores)
    good = sum(i in best_indices for i in block)
    return (math.comb(len(block) - good, draws) / math.comb(len(block), draws)
            if len(block) - good >= draws else 0.0)


def exact_position(scores, candidate_ids, endpoint_values, position):
    """Score a raw rank position; unknowns never cause lower-ranked replacement.

    Every entry in the exact-score block containing the position has probability
    1 / block size. Conditional error renormalizes only those probabilities whose
    frozen endpoint is evaluable; full coverage requires every block member.
    """
    require(type(position) is int and 1 <= position <= len(candidate_ids), 'Invalid rank position')
    require(len(scores) == len(candidate_ids) == len(set(candidate_ids)), 'Invalid ranking identities')
    require(all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
                and math.isfinite(v) for v in scores), 'Nonfinite/nonnumeric ranking scores')
    value = sorted(scores, reverse=True)[position - 1]
    indices = [i for i, score in enumerate(scores) if score == value]
    members = []
    for i in indices:
        cid = candidate_ids[i]
        pair = endpoint_values.get(cid)
        valid = pair is not None and all(
            isinstance(pair.get(k), (int, float)) and not isinstance(pair[k], bool)
            and math.isfinite(pair[k]) and pair[k] > 0 for k in ('predicted', 'observed'))
        member = {'candidate_id': cid, 'weight': 1 / len(indices),
                  'scorable': valid, 'state': 'valid' if valid else 'unknown'}
        if valid:
            signed = math.log(pair['predicted']) - math.log(pair['observed'])
            member.update(predicted=pair['predicted'], observed=pair['observed'],
                          signed_log_error=signed, absolute_log_error=abs(signed))
        members.append(member)
    known = [x for x in members if x['scorable']]
    return {'position': position, 'tie_score': float(value), 'tie_group_size': len(members),
            'tie_group_first_rank': 1 + int(sum(v > value for v in scores)),
            'tie_group_last_rank': int(sum(v >= value for v in scores)), 'members': members,
            'evaluable_probability': len(known) / len(members),
            'fully_evaluable': len(known) == len(members),
            'conditional_absolute_log_error': average([x['absolute_log_error'] for x in known]),
            'conditional_signed_log_error': average([x['signed_log_error'] for x in known]),
            'conditional_within1_5fold': average([float(x['absolute_log_error'] <= math.log(1.5)) for x in known]),
            'conditional_within2fold': average([float(x['absolute_log_error'] <= math.log(2)) for x in known])}


def independent_blocks(rows):
    """Connected compounds sharing publications; catalog families are never blocks."""
    graph = defaultdict(set)
    for row in rows:
        compound = ('compound', row['compound'])
        publication = ('publication', row['publication_cluster_id'])
        require(compound[1] and publication[1], 'Missing compound/publication identity')
        graph[compound].add(publication)
        graph[publication].add(compound)
    seen, blocks = set(), []
    for node in sorted(graph):
        if node in seen:
            continue
        frontier, component = [node], set()
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(graph[current] - component)
        seen.update(component)
        blocks.append({'compounds': sorted(v for kind, v in component if kind == 'compound'),
                       'publications': sorted(v for kind, v in component if kind == 'publication')})
    return blocks


def summarize_positions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method_id'], row['stratum'], row['endpoint'], row['position']].append(row)
    result = []
    for (method, stratum, endpoint, position), group in sorted(grouped.items()):
        known = [r for r in group if r['conditional_absolute_log_error'] is not None]
        def metric(key):
            return hierarchical([(r['compound'], r['publication_cluster_id'], r[key]) for r in known])
        log, signed = metric('conditional_absolute_log_error'), metric('conditional_signed_log_error')
        result.append({'method_id': method, 'stratum': stratum, 'endpoint': endpoint, 'position': position,
                       'requested_scenarios': len(group), 'evaluable_probability': average([r['evaluable_probability'] for r in group]),
                       'scenarios_with_nonzero_coverage': len(known),
                       'fully_evaluable_scenarios': sum(r['fully_evaluable'] for r in group),
                       'covered_compounds': len({r['compound'] for r in known}),
                       'covered_publication_compound_groups': len({(r['compound'], r['publication_cluster_id']) for r in known}),
                       'independent_compound_publication_blocks': len(independent_blocks(known)),
                       'conditional_mean_absolute_log_error': log,
                       'conditional_gmfe': math.exp(log) if log is not None else None,
                       'conditional_afe': math.exp(signed) if signed is not None else None,
                       'conditional_within1_5fold': metric('conditional_within1_5fold'),
                       'conditional_within2fold': metric('conditional_within2fold')})
    return result


def common_position_summaries(rows):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_scenario = grouped[row['method_id'], row['stratum'], row['endpoint']]
        require(row['position'] not in by_scenario[row['scenario_id']], 'Duplicate position case')
        by_scenario[row['scenario_id']][row['position']] = row
    result = []
    for (method, stratum, endpoint), cases in sorted(grouped.items()):
        require(all(set(values) == {1, 2, 3} for values in cases.values()), 'Incomplete three-position case')
        shared = {sid: values for sid, values in cases.items() if all(v['fully_evaluable'] for v in values.values())}
        identities = [values[1] for values in shared.values()]
        row = {'method_id': method, 'stratum': stratum, 'endpoint': endpoint,
               'requested_scenarios': len(cases), 'common_scenarios': len(shared),
               'common_compounds': len({x['compound'] for x in identities}),
               'common_publications': len({x['publication_cluster_id'] for x in identities}),
               'common_publication_compound_groups': len({(x['compound'], x['publication_cluster_id']) for x in identities}),
               'independent_compound_publication_blocks': len(independent_blocks(identities)),
               'independent_blocks': independent_blocks(identities), 'scenario_ids': sorted(shared),
               'common_case_groups': [{'scenario_id': sid, 'compound': shared[sid][1]['compound'],
                                       'publication_cluster_id': shared[sid][1]['publication_cluster_id']}
                                      for sid in sorted(shared)],
               'excluded_scenario_ids': sorted(set(cases) - set(shared)),
               'analysis_role': POSITION_ROLE, 'inferential_test_performed': False,
               'difference_direction': 'later rank minus rank 1; positive means larger absolute ln-error'}
        for position in (1, 2, 3):
            log = hierarchical([(values[position]['compound'], values[position]['publication_cluster_id'],
                                 values[position]['conditional_absolute_log_error']) for values in shared.values()])
            row[f'rank{position}_mean_absolute_log_error'] = log
            row[f'rank{position}_gmfe'] = math.exp(log) if log is not None else None
        for position in (2, 3):
            row[f'rank{position}_minus_rank1_absolute_log_error'] = hierarchical([
                (values[1]['compound'], values[1]['publication_cluster_id'],
                 values[position]['conditional_absolute_log_error'] - values[1]['conditional_absolute_log_error'])
                for values in shared.values()])
        result.append(row)
    return result


def build_position_diagnostics(scores, query_ids, candidate_ids, candidates, selections, summaries):
    rows = []
    for method, _, _ in METHODS:
        for qi, scenario_id in enumerate(query_ids):
            for stratum in ('reference_diagnostic', 'strict'):
                selection = selections[method, scenario_id, stratum]
                for endpoint, saved_case in selection['endpoints'].items():
                    values = {cid: candidates[scenario_id, cid][stratum]['endpoints'][endpoint]
                              for cid in candidate_ids if endpoint in candidates[scenario_id, cid][stratum]['endpoints']}
                    for position in (1, 2, 3):
                        row = exact_position(scores[method][qi], candidate_ids, values, position)
                        row.update(method_id=method, scenario_id=scenario_id, endpoint=endpoint, stratum=stratum,
                                   compound=selection['compound'], publication_cluster_id=selection['publication_cluster_id'])
                        for member in row['members']:
                            original = candidates[scenario_id, member['candidate_id']]
                            member.update(family_id=original['family_id'], identity_match=original['identity_match'],
                                          native_execution_status=original[stratum]['execution_status'])
                            if not member['scorable']:
                                member['reason'] = ('endpoint_unavailable: ' + (original[stratum].get('reason') or 'not_source_qualified'))
                        if position == 1:
                            equal(row['evaluable_probability'], saved_case['top1_evaluable_probability'], 'Rank-1 case coverage')
                            equal(row['conditional_absolute_log_error'], saved_case['selected_known_conditional_absolute_log_error'],
                                  'Rank-1 case error')
                            equal(row['conditional_signed_log_error'], saved_case['top1_signed_log_error_conditioned_on_known_selection'],
                                  'Rank-1 case signed error')
                        rows.append(row)
    position_summaries = summarize_positions(rows)
    for row in position_summaries:
        if row['position'] != 1:
            continue
        old = summaries[row['method_id'], row['endpoint'], row['stratum']]
        for new, previous in [('requested_scenarios', 'scenario_count'),
                              ('evaluable_probability', 'top1_mean_evaluable_probability'),
                              ('scenarios_with_nonzero_coverage', 'top1_numerically_covered_scenarios'),
                              ('covered_compounds', 'covered_compounds'),
                              ('covered_publication_compound_groups', 'covered_publication_compound_groups'),
                              ('conditional_gmfe', 'top1_gmfe_conditioned_on_known_selection'),
                              ('conditional_afe', 'top1_afe_conditioned_on_known_selection'),
                              ('conditional_within1_5fold', 'top1_fraction_within_1_5_fold_conditioned_on_known_selection'),
                              ('conditional_within2fold', 'top1_fraction_within_2_fold_conditioned_on_known_selection')]:
            equal(row[new], old[previous], 'Rank-1 all-endpoint summary: ' + new)
    require(len(position_summaries) == 3 * len(summaries), 'Position summary scope changed')
    return rows, position_summaries, common_position_summaries(rows)


def csv_text(rows):
    stream = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                         if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue()


def table(columns, header):
    return [r'\begingroup', r'\fontsize{9}{11}\selectfont',
            r'\setlength{\tabcolsep}{3pt}',
            r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}' + columns + r'@{}}',
            r'\toprule', header, r'\midrule']


def table_end(lines):
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular*}', r'\endgroup', ''])


def fmt(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def build(root=ROOT, *, check=False):
    root = Path(root).resolve()
    sources = {}

    def read(path, expected=None):
        path = (root / path).resolve(strict=True)
        require(path.is_relative_to(root) and 'sealed' not in path.parts,
                f'Publication export cannot read outside the project or sealed sources: {path}')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, f'Changed source: {path}')
        sources[str(path.relative_to(root))] = {'sha256': digest, 'bytes': len(raw)}
        return raw

    def data(path, expected=None):
        return json.loads(read(path, expected))

    def ref_data(ref):
        return data(ref['path'], ref['sha256'])

    read(Path(__file__).resolve())
    read('scripts/model_recommendation_paper.py')
    read('tests/test_recommendation/test_model_execution_publication.py')
    accounting = data(FOLLOWUP / 'final_accounting.json')
    require(accounting['completed'] is True and accounting['main_and_supplement_metrics_not_pooled'] is True,
            'Final accounting is incomplete or pools supplementary metrics')
    bound = {v['path']: v['sha256'] for v in accounting['input_refs']}
    completion_path = str(FOLLOWUP / 'completion.json')
    completion = data(completion_path, bound[completion_path])
    require(completion['completed'] is True and completion['original_artifacts_rechecked_unchanged'] is True,
            'Scoring completion gate failed')
    audit_path = str(FOLLOWUP / 'audit.json')
    audit = data(audit_path, bound[audit_path])
    require(audit['accepted'] is True and audit['scoring_functions_reused'] is False,
            'Independent results audit is not accepted')
    require(audit['completion'] == {'path': completion_path, 'sha256': bound[completion_path]},
            'Audit does not bind this completion')
    primary_entry, = [x for x in completion['outputs'] if x['analysis'] == 'primary']
    result_ref, = [x for x in primary_entry['files'] if x['path'].endswith('/results.json')]
    report = ref_data(result_ref)
    for ref in primary_entry['files']:
        read(ref['path'], ref['sha256'])
    summary = report['summary']
    require(summary == primary_entry['summary'] == audit['analyses']['primary']['summary'],
            'Primary summary differs from completion/audit')
    require((summary['scenarios'], summary['candidates'], summary['methods']) == (41, 280, 12),
            'Unexpected clinical ranking scope')
    require(summary['native_status_counts'] == {'blocked': 41, 'prediction_locked': 136}
            and summary['numerically_evaluable_reference_pairs'] == 129
            and summary['scalar_dispositions']['valid'] == 171
            and summary['direct_validated_pairs'] == 0, 'Unexpected accepted main denominator')
    candidates = {(x['scenario_id'], x['candidate_id']): x for x in report['candidate_results']}
    require(len(candidates) == len(report['candidate_results']) == 41 * 280, 'Candidate scope is not unique/complete')
    scalar_counts = Counter(s['endpoint_key'] for x in candidates.values() for s in x['scalars'] if s['state'] == 'valid')
    require(sum(scalar_counts.values()) == 171, 'Main scalar denominator changed')

    # These are the actual 41 clinical queries, independently scored over 280 entries.
    # The earlier 145-question recommendation-quality experiment is never consumed.
    ranking_lock = data(EXPERIMENT / 'rankings/ranking_lock.json', report['locks']['ranking_lock_sha256'])
    require(ranking_lock['state'] == 'complete' and ranking_lock['scientific_outcomes_read'] is False,
            'Ranking is not a completed pre-outcome lock')
    query_ids, candidate_ids = ranking_lock['query_ids'], ranking_lock['candidate_ids']
    require(len(set(query_ids)) == 41 and len(set(candidate_ids)) == 280
            and set(query_ids) == {x[0] for x in candidates}, 'Clinical ranking index mismatch')
    require(set(ranking_lock['methods']) == {m[0] for m in METHODS}, 'Method scope changed')
    scores = {}
    for method, _, _ in METHODS:
        entry = ranking_lock['methods'][method]
        index = ref_data(entry['index'])
        require(index['query_ids'] == query_ids and index['candidate_ids'] == candidate_ids
                and index['variants'] == ['original'], 'Ranking index does not describe this clinical experiment')
        values = np.load(io.BytesIO(read(entry['path'], entry['sha256'])), allow_pickle=False)
        require(values.shape == (1, 41, 280) and np.isfinite(values).all(), 'Ranking matrix shape/nonfinite')
        scores[method] = values[0]
    selections = {(x['method'], x['scenario_id'], x['stratum']): x for x in report['selection']}
    require(len(selections) == len(report['selection']) == 12 * 41 * 2, 'Incomplete or duplicate selections')
    require(all(x['candidate_count'] == 280 for x in selections.values()), 'Selection pool was narrowed')
    require({x['tie_policy'] for x in selections.values()} == {
        'uniform permutation within exact equal-score blocks; expected fixed-K inclusion, not ID tie-break'},
        'Unexpected tie policy')

    summaries = {(x['method'], x['endpoint'], x['stratum']): x for x in report['method_endpoint_summary']}
    require(len(summaries) == len(report['method_endpoint_summary']) == 504, 'Summary scope changed')
    omissions = {(x['method'], x['endpoint']): x for x in audit['analyses']['primary']['omission_statistics']}
    main_rows, omission_cases = [], []
    for method, name, group in METHODS:
        saved = summaries[method, CMAX, 'reference_diagnostic']
        coverage, logs, folds2, misses, cases = [], [], [], [], []
        for qi, scenario_id in enumerate(query_ids):
            selection = selections[method, scenario_id, 'reference_diagnostic']
            if CMAX not in selection['endpoints']:
                continue
            endpoint = selection['endpoints'][CMAX]
            values = scores[method][qi]
            tied = np.flatnonzero(values == values.max())
            weight = 1 / len(tied)
            registered_weights = selection['by_k']['1']['inclusion_probabilities']
            for i, cid in enumerate(candidate_ids):
                equal(registered_weights[cid], weight if i in tied else 0, 'Top-1 uniform tie probability')
            valid = {}
            for i, cid in enumerate(candidate_ids):
                items = [s for s in candidates[scenario_id, cid]['scalars']
                         if s.get('endpoint_key') == CMAX and s['state'] == 'valid']
                require(len(items) <= 1, 'Duplicate candidate endpoint')
                if items:
                    scalar = items[0]
                    log = abs(math.log(scalar['predicted'] / scalar['observed']))
                    equal(math.exp(log), scalar['fold_error'], 'Candidate scalar fold')
                    equal(log, scalar['absolute_log_error'], 'Candidate scalar log')
                    valid[i] = log
            chosen = [valid[i] for i in tied if i in valid]
            prob = len(chosen) * weight
            equal(prob, endpoint['top1_evaluable_probability'], 'Case coverage')
            coverage.append(prob)
            if chosen:
                group_key = selection['compound'], selection['publication_cluster_id']
                logs.append((*group_key, average(chosen)))
                folds2.append((*group_key, average([float(math.exp(v) <= 2) for v in chosen])))
            if len(valid) >= 2:
                # Best-candidate identity uses the exact saved absolute-log errors;
                # the ratio formula above is checked, not used to introduce new ties.
                best_ids = [cid for cid in candidate_ids if any(
                    s.get('endpoint_key') == CMAX and s['state'] == 'valid'
                    and s['absolute_log_error'] == endpoint['best_observed_absolute_log_error']
                    for s in candidates[scenario_id, cid]['scalars'])]
                require(best_ids, 'Measured best-candidate set is empty')
                miss = top5_miss(values, {candidate_ids.index(cid) for cid in best_ids})
                numerical = endpoint['numerical_omission_audit']
                equal(miss, numerical['by_k']['5']['best_measured_miss_probability'], 'Top-5 tie miss')
                misses.append((selection['compound'], selection['publication_cluster_id'], miss))
                cases.append({'scenario_id': scenario_id, 'compound': selection['compound'],
                              'publication_cluster_id': selection['publication_cluster_id'],
                              'measured_candidates': numerical['measured_count'],
                              'same_parent_candidates': numerical['same_parent_count'],
                              'same_parent_pool_complete': numerical['complete_same_parent_numeric_pool'],
                              'best_measured_candidate_ids': best_ids})
        equal(average(coverage), saved['top1_mean_evaluable_probability'], 'Main coverage')
        equal(math.exp(hierarchical(logs)) if logs else None,
              saved['top1_gmfe_conditioned_on_known_selection'], 'Hierarchical GMFE')
        equal(hierarchical(folds2), saved['top1_fraction_within_2_fold_conditioned_on_known_selection'],
              'Hierarchical 2-fold proportion')
        omission = omissions[method, CMAX]
        require(len(coverage) == saved['scenario_count'] == 34 and len(cases) == omission['eligible_cases'] == 19,
                'Cmax requested/choice denominators changed')
        equal(hierarchical(misses), omission['hierarchical_miss_probability'], 'Hierarchical Top-5 miss')
        equal(math.fsum(x[2] for x in misses), omission['expected_missed_cases'], 'Expected missed cases')
        require(omission['compounds'] == omission['publication_compound_groups'] == 17, 'Top-5 hierarchy changed')
        if not omission_cases:
            omission_cases = cases
        else:
            require(cases == omission_cases, 'Measured-pool identity varies by method')
        main_rows.append({'method_id': method, 'method': name, 'group': group,
                          'cmax_requested_scenarios': 34,
                          'top1_coverage_probability': saved['top1_mean_evaluable_probability'],
                          'top1_coverage_percent': 100 * saved['top1_mean_evaluable_probability'],
                          'top1_scenarios_with_nonzero_coverage': saved['top1_numerically_covered_scenarios'],
                          'conditional_error_compounds': saved['covered_compounds'],
                          'conditional_gmfe': saved['top1_gmfe_conditioned_on_known_selection'],
                          'conditional_within2fold_percent': None if not logs else 100 * hierarchical(folds2),
                          'top5_best_measured_miss_percent': 100 * omission['hierarchical_miss_probability'],
                          'top5_expected_missed_cases': omission['expected_missed_cases'],
                          'top5_eligible_cases': 19, 'top5_compounds': 17})
    require(sum(not x['same_parent_pool_complete'] for x in omission_cases) == 2, 'Incomplete-pool count changed')

    rank_cases, position_summaries, common_positions = build_position_diagnostics(
        scores, query_ids, candidate_ids, candidates, selections, summaries)
    position_index = {(x['method_id'], x['stratum'], x['endpoint'], x['position']): x for x in position_summaries}
    for row in main_rows:
        for position in (1, 2, 3):
            new = position_index[row['method_id'], 'reference_diagnostic', CMAX, position]
            row.update({f'rank{position}_conditional_gmfe': new['conditional_gmfe'],
                        f'rank{position}_coverage_percent': 100 * new['evaluable_probability'],
                        f'rank{position}_conditional_within2fold_percent': None if new['conditional_within2fold'] is None
                        else 100 * new['conditional_within2fold']})

    paired = [x for x in report['paired_comparisons']['comparisons']
              if x['endpoint'] == CMAX and x['stratum'] == 'reference_diagnostic']
    require(len(paired) == 11 and {x['method'] for x in paired} == {m[0] for m in METHODS[1:]},
            'BM25 prespecified family mismatch')
    require(sum(x['state'] == 'computed' for x in paired) == 8, 'Paired inference availability changed')
    for row in paired:
        require(row['holm_prespecified_family_size'] == 11 and row['minimum_blocks_for_inference'] == 10
                and row['lower_difference_is_better'] is True, 'Paired estimand changed')
        if row['state'] == 'computed':
            require(row['independent_compound_publication_blocks'] == 10
                    and row['bootstrap_95ci'][0] < 0 < row['bootstrap_95ci'][1]
                    and row['holm_adjusted_p'] == 1, 'Paired result differs from accepted report')

    # All seven registered post-scoring branches, including unfavorable results.
    supplement_scalars, task_keys, supplement_pairs = [], set(), set()
    for branch in accounting['branches']:
        receipt = ref_data(branch['receipt'])
        require(not receipt.get('source_parameters_fitted', False)
                and not receipt.get('pool_into_original_main_results', False), 'Supplement is fitted/pooled')
        rows = receipt['rows']
        require(len(rows) == branch['variant_tasks'] == branch['successful_variant_tasks'], 'Branch task count changed')
        pairs = {(x['parent_scenario_id'], x['candidate_id']) for x in rows}
        require(len(pairs) == branch['original_pairs'], 'Branch original-pair count changed')
        supplement_pairs.update(pairs)
        branch_scalar_count = 0
        for row in rows:
            require(row['status'] == 'prediction_locked' and row['audit']['sampled_cmax_recomputed'] is True,
                    'Supplement lacks a locked, independently checked prediction')
            key = row['parent_scenario_id'], row['candidate_id'], row['variant']
            require(key not in task_keys, 'Duplicate supplement task')
            task_keys.add(key)
            for scalar in row['scores']:
                ratio = scalar['predicted'] / scalar['observed']
                require(math.isfinite(ratio) and ratio > 0, 'Invalid supplement scalar')
                equal(max(ratio, 1 / ratio), scalar['fold_error'], 'Supplement fold')
                within = scalar.get('within_2fold', scalar.get('within_2_fold'))
                require(within == (scalar['fold_error'] <= 2), 'Supplement threshold mismatch')
                supplement_scalars.append({
                    'branch': Path(branch['receipt']['path']).parent.name, 'receipt': branch['receipt'],
                    'compound': row['compound'], 'candidate_id': row['candidate_id'],
                    'parent_scenario_id': row['parent_scenario_id'], 'variant': row['variant'],
                    'analysis_stratum': row['analysis_stratum'], 'scalar': scalar})
                branch_scalar_count += 1
        require(branch_scalar_count == branch['scalar_comparisons'], 'Branch scalar count changed')
    require((len(accounting['branches']), len(task_keys), len(supplement_pairs), len(supplement_scalars)) == (7, 31, 27, 45),
            'Supplement denominator mismatch')
    original_pairs = accounting['all_original_pairs']
    main_pairs = {(x['scenario_id'], x['candidate_id']) for x in original_pairs if x['executed_in_main']}
    require(len(main_pairs) == 136 and not main_pairs & supplement_pairs and len(main_pairs | supplement_pairs) == 163,
            'Main/supplement overlap or union mismatch')
    counts = accounting['counts']
    require(len(original_pairs) == 177 and len({(x['scenario_id'], x['candidate_id']) for x in original_pairs}) == 177,
            'Original task identities duplicated')
    require((counts['successful_candidates_any_registered_branch'], counts['same_parent_candidates'],
             counts['completed_compounds_with_any_prediction'], counts['completed_scenarios_with_any_prediction'])
            == (124, 130, 37, 40), 'All-branch coverage changed')

    grouped = defaultdict(list)
    for row in supplement_scalars:
        scalar = row['scalar']
        window = scalar.get('relative_window_h', scalar.get('prediction_window_h'))
        key = (row['compound'], row['variant'], row['parent_scenario_id'], scalar['endpoint'], tuple(window))
        grouped[key].append(row)
    supplement_rows = []
    for (compound, variant, scenario, endpoint, window), rows in sorted(grouped.items()):
        folds = [x['scalar']['fold_error'] for x in rows]
        supplement_rows.append({'compound': compound, 'variant': variant, 'parent_scenario_id': scenario,
                                'endpoint': endpoint, 'relative_window_h': list(window), 'scalar_comparisons': len(rows),
                                'candidate_entries': len({x['candidate_id'] for x in rows}),
                                'fold_error_min': min(folds), 'fold_error_max': max(folds),
                                'within2fold_count': sum(x <= 2 for x in folds),
                                'independent_model_count': None, 'included_in_main_analysis': False})

    all_endpoint_counts = []
    for endpoint in sorted({x['endpoint'] for x in report['method_endpoint_summary']}):
        rows = [x for x in report['method_endpoint_summary'] if x['endpoint'] == endpoint and x['stratum'] == 'reference_diagnostic']
        require(len(rows) == 12 and len({x['scenario_count'] for x in rows}) == 1, 'Endpoint denominator inconsistent')
        all_endpoint_counts.append({'endpoint': endpoint, 'requested_scenarios': rows[0]['scenario_count'],
                                    'valid_scalar_comparisons': scalar_counts[endpoint]})
    publication = {
        'schema_version': 'paper-model-execution-export-v2', 'state': 'complete',
        'source_files': sources, 'scope': counts, 'main_summary': summary,
        'ranking_scope': {'clinical_scenarios': 41, 'catalogue_entries': 280, 'methods': 12,
                          'pairs_per_method': 11480, 'earlier_145_quality_queries_reused': False,
                          'ranking_lock_sha256': report['locks']['ranking_lock_sha256'],
                          'tie_policy': next(iter(selections.values()))['tie_policy']},
        'definitions': report['definitions'], 'limitations': report['limitations'],
        'weighting': 'Coverage: equal requested scenarios. Conditional errors and Top-5 miss: equal cases within publication/compound, publications within compound, then compounds.',
        'source_independence': {'absolute_independence_claimed': False,
                                'source_audit_counts': summary['source_audit_status_counts'],
                                'candidate_versions_are_independent_samples': False},
        'main_cmax_table': main_rows, 'main_endpoint_counts': all_endpoint_counts,
        'rank_positions': {
            'analysis_role': POSITION_ROLE, 'positions': [1, 2, 3],
            'position_definition': 'Exact position in the frozen full 280-entry ranking; not best-of-Top-K or a same-parent filtered list.',
            'tie_rule': 'Uniform probability 1/block_size for every candidate in the exact-score block containing that position.',
            'unknown_rule': 'Unknown/cross-drug entries retain their positions and probability mass; no backfill.',
            'coverage_weighting': 'Equal requested endpoint scenarios, including zero coverage.',
            'error_weighting': 'Within a scenario, condition on scorable tie members; equal cases/publication, publications/compound, then covered compounds.',
            'common_scope': 'Per method and endpoint, all three positions must have every tie member scorable; no independence between positions assumed.',
            'new_inferential_tests_performed': False,
            'summaries': position_summaries, 'cases': rank_cases, 'common_positions': common_positions},
        'all_main_method_endpoint_summaries': report['method_endpoint_summary'],
        'all_main_paired_comparisons': report['paired_comparisons'],
        'all_main_omission_statistics': audit['analyses']['primary']['omission_statistics'],
        'cmax_measured_choice_cases': omission_cases,
        'supplement_counts': {'branches': 7, 'original_pairs': 27, 'variant_tasks': 31, 'scalar_comparisons': 45},
        'supplement_table': supplement_rows, 'supplement_scalars': supplement_scalars,
        'checks': {'completion_and_audit_sha_verified': True, 'ranking_41_by_280_by_12_verified': True,
                   'main_cmax_independently_reaggregated': True, 'top5_ties_independently_rederived': True,
                   'all_endpoints_exported': True, 'all_seven_receipts_sha_verified': True,
                   'all_504_rank1_summaries_reproduce_frozen_metrics': True,
                   'exact_positions_never_backfilled': True,
                   'supplement_formulas_checked': True, 'sealed_files_opened_by_exporter': 0,
                   'native_or_model_inference_executed_by_exporter': False},
    }
    outputs = {}
    stem = 'results/tables/model_recommendation_execution'
    outputs['results/data/model_recommendation_execution.json'] = json.dumps(publication, ensure_ascii=False, sort_keys=True,
                                                                            separators=(',', ':'), allow_nan=False) + '\n'
    outputs[stem + '.csv'] = csv_text(main_rows)
    # Complete endpoint summaries remain accessible as both machine-readable JSON and CSV.
    outputs[stem + '_all_endpoints.csv'] = csv_text(report['method_endpoint_summary'])
    outputs[stem + '_positions.csv'] = csv_text(position_summaries)
    outputs[stem + '_rank_cases.csv'] = csv_text(rank_cases)
    outputs[stem + '_common_positions.csv'] = csv_text(common_positions)
    outputs[stem + '_supplement.csv'] = csv_text(supplement_rows)
    paired_csv = []
    by_method = {x['method']: x for x in paired}
    for method, name, group in METHODS[1:]:
        row = by_method[method]
        paired_csv.append({'method_id': method, 'method': name, 'group': group,
                           'independent_blocks': row['independent_compound_publication_blocks'],
                           'mean_paired_absolute_ln_error_difference': row['mean_paired_absolute_log_error_difference'],
                           'ci95_lower': None if row['bootstrap_95ci'] is None else row['bootstrap_95ci'][0],
                           'ci95_upper': None if row['bootstrap_95ci'] is None else row['bootstrap_95ci'][1],
                           'holm_p': row['holm_adjusted_p'], 'state': row['state'], 'reason': row['reason'],
                           'holm_prespecified_family_size': 11})
    outputs[stem + '_paired.csv'] = csv_text(paired_csv)
    lines = table('lrrrrr', r'Method & \shortstack{Rank 1 GMFE\\(Cov.\%)} & \shortstack{Rank 2 GMFE\\(Cov.\%)} & \shortstack{Rank 3 GMFE\\(Cov.\%)} & \shortstack{Rank 1\\$\leq 2$-fold (\%)} & \shortstack{Top-5\\miss (\%)} \\')
    for group, heading in [('full_catalog', 'Full-catalog scoring'), ('top20', 'BM25 top-20 reranking')]:
        if group == 'top20':
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{6}{@{}l@{}}{\textit{' + heading + r'}} \\')
        for row in (x for x in main_rows if x['group'] == group):
            rank_values = [fmt(row[f'rank{p}_conditional_gmfe']) + ' (' + fmt(row[f'rank{p}_coverage_percent'], 1) + ')'
                           for p in (1, 2, 3)]
            lines.append(' & '.join([row['method'], *rank_values, fmt(row['conditional_within2fold_percent'], 1),
                                     fmt(row['top5_best_measured_miss_percent'], 1)]) + r' \\')
    outputs[stem + '.tex'] = table_end(lines)
    common_index = {x['method_id']: x for x in common_positions
                    if x['endpoint'] == CMAX and x['stratum'] == 'reference_diagnostic'}
    lines = table('lrrrrr', r'Method & Cases & Blocks & Rank 1 GMFE & Rank 2 GMFE & Rank 3 GMFE \\')
    for method, name, _ in METHODS:
        row = common_index[method]
        lines.append(' & '.join([name, str(row['common_scenarios']),
                                 str(row['independent_compound_publication_blocks']),
                                 *[fmt(row[f'rank{p}_gmfe']) for p in (1, 2, 3)]]) + r' \\')
    outputs[stem + '_common_positions.tex'] = table_end(lines)
    lines = table('lrrrr', r'Method vs.\ BM25 & Blocks & $\Delta |\ln(P/O)|$ & 95\% CI & Holm $p$ \\')
    for row in paired_csv:
        ci = 'NA' if row['ci95_lower'] is None else '[' + fmt(row['ci95_lower']) + ', ' + fmt(row['ci95_upper']) + ']'
        lines.append(' & '.join([row['method'], str(row['independent_blocks']),
                                 fmt(row['mean_paired_absolute_ln_error_difference']), ci, fmt(row['holm_p'])]) + r' \\')
    outputs[stem + '_paired.tex'] = table_end(lines)
    labels = {
        'author_native_salt_mass': 'Salt-mass input',
        'original_CR25_active_label_reference': 'Original CR 25 mg',
        'dense_120h_6min': 'Day 1; dense grid',
        'last_morning_13_doses': '13 doses; morning',
        'last_evening_14_doses': '14 doses; evening',
        'source_bound_inactive_repag_input_removal': 'Inactive input removed',
        'author_static_combination_total_lng': 'Original combination',
        'original_static_combination_ee_reference': 'Original combination',
        'official_iv_input_route_adaptation': 'New IV input',
    }
    lines = table('llrrr', r'Drug / registered supplement & Endpoint & $n$ & Fold-error range & $\leq 2$-fold \\')
    previous = None
    for row in supplement_rows:
        label = labels[row['variant']]
        if row['compound'] == 'voriconazole':
            label += '; ' + ('4' if row['parent_scenario_id'].endswith('_4mgkg') else '6') + ' mg/kg'
        heading = row['compound'].capitalize() + ': ' + label
        if heading != previous:
            if previous is not None:
                lines.append(r'\addlinespace[2pt]')
            lines.append(r'\multicolumn{5}{@{}l@{}}{\textit{' + heading + r'}} \\')
            previous = heading
        endpoint = {'Cmax': r'$C_{\max}$', 'Cmax_ss': r'$C_{\max,\mathrm{ss}}$',
                    'AUC0_48': r'$\mathrm{AUC}_{0\text{--}48\mathrm{h}}$'}.get(row['endpoint'])
        if row['endpoint'] == 'AUC_tau_ss':
            endpoint = r'$\mathrm{AUC}_{\tau,' + str(int(row['relative_window_h'][1])) + r'\mathrm{h}}$'
        require(endpoint is not None, 'Unlabelled supplement endpoint')
        span = fmt(row['fold_error_min'])
        if row['scalar_comparisons'] > 1:
            upper = fmt(row['fold_error_max'])
            span = r'$\approx ' + span + '$' if span == upper else span + '--' + upper
        lines.append(' & '.join(['', endpoint, str(row['scalar_comparisons']), span,
                                 f"{row['within2fold_count']}/{row['scalar_comparisons']}"]) + r' \\')
    outputs[stem + '_supplement.tex'] = table_end(lines)
    # Closing recheck is limited to publication inputs, not a new scientific barrier.
    for relative, ref in list(sources.items()):
        read(relative, ref['sha256'])
    for relative, text in outputs.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root / 'results'), 'Output escapes paper results')
        if check:
            require(path.read_bytes() == text.encode('utf-8'), f'Generated paper artifact differs: {relative}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(path)
    return {'state': 'complete', 'check_only': check, 'outputs': list(outputs),
            'main_scalar_comparisons': 171, 'supplement_scalar_comparisons': 45,
            'cmax_requested_scenarios': 34, 'top5_eligible_cases': 19, 'top5_compounds': 17,
            'cmax_paired_computed': 8, 'cmax_paired_unavailable': 3, 'sources_verified': len(sources),
            'rank_position_summaries': len(position_summaries), 'rank_position_cases': len(rank_cases),
            'common_position_summaries': len(common_positions)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate inputs and byte-compare existing exports; write nothing.')
    args = parser.parse_args()
    print(json.dumps(build(check=args.check), ensure_ascii=False))
