#!/usr/bin/env python3
"""Complete, provenance-checked reporting for the user-selected light model.

Reuse completed external predictions; never infer another baseline or convert
the abandoned LoRA experiment into a completed experiment. All estimates retain
the existing unknown masks, exact ties, endpoint windows and grouping rules.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.recommendation.pbpk_match_v2 import reporting as r

OUT = ROOT / 'experiments/model_recommendation/pbpk_match_light'
OWN = 'pbpk_match_light'
DOCUMENT = 'pbpk_match_light_document'
SEED = 17
DISPLAY = {**r.DISPLAY, OWN: 'PBPK-Match', DOCUMENT: 'PBPK-Match (document)'}


def scope(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or 'sealed' in path.relative_to(ROOT.resolve()).parts:
        raise ValueError('Light reporting must remain inside the project and outside sealed data')
    return path


def read(path):
    return json.loads(scope(path).read_text())


def ref(path):
    path = scope(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': r.sha(path)}


def verify_ref(value, refs):
    if not isinstance(value, dict) or not {'path', 'sha256'} <= set(value):
        raise ValueError('Expected an artifact path and SHA256')
    path = scope(value['path'])
    if not path.is_file() or r.sha(path) != value['sha256']:
        raise ValueError('Missing or changed report input: ' + str(path))
    refs.append(ref(path))
    return path


def prediction_arrays(path, q, c, indices):
    with np.load(scope(path), allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in ('indices', 'scores', 'base_scores', 'probabilities')}
    if not np.array_equal(arrays['indices'], indices):
        raise ValueError('Prediction request indices differ from registered membership')
    for key in ('scores', 'base_scores'):
        if arrays[key].shape != (3, q, c) or not np.isfinite(arrays[key]).all():
            raise ValueError('Incomplete/nonfinite all-candidate prediction: ' + key)
    p = arrays['probabilities']
    if p.shape == (q, c, 3):
        arrays['probabilities'] = np.broadcast_to(p, (3, q, c, 3)).copy()
    elif p.shape != (3, q, c, 3):
        raise ValueError('Probability axes must cover Y/N/U for every request and candidate')
    for variant in range(3):
        r._probabilities(arrays['probabilities'][variant], (q, c))
    return arrays


def _binding_indices(data, fold_id, kind):
    fold = data['folds'][fold_id]
    supervised = set(data['core_indices']) | set(data['execution_indices'])
    if kind == 'outer':
        return sorted(supervised & set(fold['outer_train'])), [], list(fold['outer_test'])
    inner = fold['inner'][int(kind.removeprefix('inner'))]
    validation = sorted(supervised & set(inner['validation']))
    return sorted(supervised & set(inner['train'])), validation, validation


def _verify_job(data, complete_path, protocol_sha, dataset_sha, cache_sha, refs):
    """Check complete training provenance, including groups and all inner fits."""
    directory = complete_path.parent
    kind = directory.name
    fold_id = int(directory.parent.name.removeprefix('fold'))
    complete = read(complete_path)
    if complete.get('state', 'complete') != 'complete':
        raise ValueError('Registered fit is not complete: ' + str(directory))
    artifacts = complete.get('artifacts_sha256', {})
    prediction_name = 'predictions.npz' if kind == 'outer' else 'validation.npz'
    if not {'binding.json', prediction_name} <= set(artifacts):
        raise ValueError('Fit receipt lacks binding or predictions')
    for relative, digest in artifacts.items():
        path = scope(directory / relative)
        if not path.is_relative_to(directory):
            raise ValueError('Fit artifact escapes its own directory')
        verify_ref({'path': str(path), 'sha256': digest}, refs)
    binding = read(directory / 'binding.json')
    if (binding.get('protocol_sha256') != protocol_sha or binding.get('dataset_sha256') != dataset_sha
            or binding.get('cache_manifest_sha256') != cache_sha or binding.get('seed') != SEED):
        raise ValueError('Fit protocol, data, cache or seed binding differs')
    train, validation, prediction = _binding_indices(data, fold_id, kind)
    expected = {'train_indices': train, 'validation_indices': validation, 'prediction_indices': prediction}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('Fit membership differs from registered grouped folds')
    groups = np.asarray(data['groups'])
    if set(groups[train]) & set(groups[prediction]):
        raise ValueError('Training and prediction groups overlap')
    if kind == 'outer' and binding.get('selection'):
        verify_ref(binding['selection'], refs)
    # Metadata is written only after both the document head and error head have
    # completed; empty markers cannot impersonate completed training.
    if (complete.get('train_indices') != train or complete.get('validation_indices') != validation
            or complete.get('seed') != SEED or int(complete.get('epochs_completed', 0)) < 1):
        raise ValueError('Fit metadata does not establish completed registered training')
    if not set(complete.get('error_train_indices', train)) <= set(train):
        raise ValueError('Error branch was fitted outside the registered training fold')
    return complete, binding


def load_inputs(data, output=OUT):
    output = scope(output)
    refs = []
    protocol_path = output / 'protocol.json'
    protocol = read(protocol_path)
    refs.append(ref(protocol_path))
    expected = {'seeds': [SEED], 'query_ids': [q['query_id'] for q in data['query_records']],
                'candidate_ids': [c['candidate_id'] for c in data['candidate_records']], 'folds': data['folds']}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Light protocol methods, axes or grouped folds differ')
    dataset_path = verify_ref(protocol['dataset_manifest'], refs)
    cache_path = verify_ref(protocol['cache_manifest'], refs)
    dataset_sha, cache_sha = r.sha(dataset_path), r.sha(cache_path)
    cache = read(cache_path)
    if cache.get('completed') is not True:
        raise ValueError('Frozen input cache is incomplete')
    verify_ref({'path': str(cache_path.parent / 'inputs.npz'), 'sha256': cache['array_sha256']}, refs)
    verify_ref({'path': str(cache_path.parent / 'binding.json'), 'sha256': cache['binding_sha256']}, refs)
    cache_binding = read(cache_path.parent / 'binding.json')
    if (cache_binding.get('source_manifest_sha256') != dataset_sha
            or cache_binding.get('query_ids') != expected['query_ids']
            or cache_binding.get('candidate_ids') != expected['candidate_ids']
            or cache_binding.get('labels_used_as_features') is not False):
        raise ValueError('Frozen cache axes/source differ or outcome-free input binding is absent')
    for path, digest in protocol.get('implementation_sha256', {}).items():
        verify_ref({'path': path, 'sha256': digest}, refs)
    registry = protocol.get('external_methods', [])
    methods = [row['method'] for row in registry]
    if not methods or len(methods) != len(set(methods)) or r.BASELINE not in methods:
        raise ValueError('External comparison registry is missing or ambiguous')
    q, c = len(data['query_records']), len(data['candidate_records'])
    external = {}
    for row in registry:
        path = verify_ref(row['scores'], refs)
        receipt_path = verify_ref(row['receipt'], refs)
        receipt = read(receipt_path)
        if receipt.get('state') != 'complete' or receipt.get('score_sha256') != r.sha(path):
            raise ValueError('External score completion differs: ' + row['method'])
        for source in receipt.get('sources', []):
            verify_ref(source, refs)
            if source.get('exact_visible_inputs_verified') is not True:
                raise ValueError('Reused external source lacks an exact visible-input check')
        with np.load(path, allow_pickle=False) as saved:
            scores = saved['scores'].copy()
        if scores.shape != (3, q, c) or not np.isfinite(scores).all():
            raise ValueError('External scores are incomplete: ' + row['method'])
        external[row['method']] = scores
    final_path = output / 'predictions' / f'seed{SEED}.npz'
    final_receipt_path = final_path.with_suffix('.json')
    final = read(final_receipt_path)
    if (final.get('state') != 'complete' or final.get('protocol_sha256') != r.sha(protocol_path)
            or final.get('score_sha256') != r.sha(final_path)):
        raise ValueError('Final light prediction is incomplete or changed')
    arrays = prediction_arrays(final_path, q, c, np.arange(q))
    refs += [ref(final_receipt_path), ref(final_path)]
    jobs = final.get('fit_receipts', [])
    expected_jobs = {scope(output / 'runs' / f'seed{SEED}' / f'fold{i}' / kind / 'complete.json')
                     for i, fold in enumerate(data['folds'])
                     for kind in [*(f'inner{j}' for j in range(len(fold['inner']))), 'outer']}
    paths = [scope(job['path']) for job in jobs]
    if len(paths) != len(set(paths)) or set(paths) != expected_jobs:
        raise ValueError('Not every registered small-network fit is complete')
    job_records = {}
    for value in jobs:
        path = verify_ref(value, refs)
        job_records[path] = _verify_job(data, path, r.sha(protocol_path), dataset_sha, cache_sha, refs)
    if set(final.get('outer_receipts', {})) != {str(i) for i in range(len(data['folds']))}:
        raise ValueError('Final prediction does not bind every outer fold')
    covered, integrity, selections = set(), [], []
    for fold_id, fold in enumerate(data['folds']):
        path = verify_ref(final['outer_receipts'][str(fold_id)], refs)
        wanted = scope(output / 'runs' / f'seed{SEED}' / f'fold{fold_id}' / 'outer' / 'complete.json')
        if path != wanted:
            raise ValueError('Outer-fold receipt path differs')
        complete, binding = job_records[path]
        ids = list(fold['outer_test'])
        if covered & set(ids):
            raise ValueError('Outer predictions contain duplicate requests')
        covered.update(ids)
        saved = prediction_arrays(path.parent / 'predictions.npz', len(ids), c, ids)
        for name in ('scores', 'base_scores', 'probabilities'):
            if not np.array_equal(saved[name], arrays[name][:, ids]):
                raise ValueError('Final prediction differs from an outer held-out prediction')
        audit_path = path.parent / 'integrity.json'
        if 'integrity.json' not in complete['artifacts_sha256']:
            raise ValueError('Outer fold lacks repeat/order/batch inference checks')
        pending = []
        audit = r._integrity(audit_path, sorted(ids)[:2], c, refs, pending,
                             f'light/fold{fold_id}', complete['artifacts_sha256']['integrity.json'])
        if pending or audit is None or 'document_repeat' not in audit:
            raise ValueError('Outer-fold inference integrity is incomplete')
        integrity.append({'fold': fold_id, **audit})
        if binding.get('selection'):
            selections.append({'fold': fold_id, **read(scope(binding['selection']['path']))})
    if covered != set(range(q)):
        raise ValueError('Outer predictions omit registered requests')
    return protocol, external, arrays, refs, integrity, selections


def stable_paired_quality(reports, query_groups, *, baseline, draws=2000):
    """Common random draws for identical grouped differences, independent of order.

    The estimand, 2000 bootstrap draws and two-sided group sign-flip test match
    the previous evaluator. Only RNG allocation changes: a metric and its
    ordered group IDs define the seed, never the method name or call order.
    """
    base = {row['query_id']: row for row in reports[baseline]['per_query']}
    results = []
    for method in sorted(reports):
        if method == baseline:
            continue
        rows = sorted(reports[method]['per_query'], key=lambda row: row['query_id'])
        for metric in ('ndcg@10', 'recall@10', 'mrr'):
            grouped = defaultdict(list)
            for row in rows:
                other = base.get(row['query_id'])
                if row['judged_only'] is not None and other is not None and other['judged_only'] is not None:
                    grouped[query_groups[row['query_id']]].append(row['judged_only'][metric] - other['judged_only'][metric])
            keys = sorted(grouped)
            values = np.asarray([np.mean(grouped[key]) for key in keys])
            material = json.dumps(['quality', metric, keys], separators=(',', ':'))
            seed = 20261004 + int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            ci, pvalue = None, None
            if len(values) >= 2:
                bootstrap = rng.choice(values, (draws, len(values))).mean(axis=1)
                ci = np.quantile(bootstrap, [.025, .975]).tolist()
                null = (rng.choice([-1., 1.], (draws, len(values))) * values).mean(axis=1)
                pvalue = float((1 + np.sum(np.abs(null) >= abs(values.mean()) - 1e-12)) / (draws + 1))
            results.append({'method': method, 'baseline': baseline, 'metric': metric,
                'independent_groups': len(values), 'mean_group_delta': float(values.mean()) if len(values) else None,
                'bootstrap_95ci': ci, 'cluster_sign_flip_p': pvalue, 'bootstrap_draws': draws, 'seed': seed,
                'interpretation': 'Internal grouped comparison; not external validation'})
    for metric in ('ndcg@10', 'recall@10', 'mrr'):
        family = sorted((row for row in results if row['metric'] == metric),
                        key=lambda row: (row['cluster_sign_flip_p'] if row['cluster_sign_flip_p'] is not None else 1., row['method']))
        previous = 0.
        for index, row in enumerate(family):
            pvalue = row['cluster_sign_flip_p'] if row['cluster_sign_flip_p'] is not None else 1.
            previous = max(previous, min(1., (len(family) - index) * pvalue))
            if row['cluster_sign_flip_p'] is not None:
                row['holm_p'] = previous
            row['holm_family'] = f'all reported methods versus {baseline}, {metric}'
    return results


def stable_paired_execution(rows, methods, *, baseline, rank, draws=2000):
    """Reuse the established clustered estimator with a method-neutral RNG.

    Each pair is given the same rank-specific alias solely while calling the
    frozen estimator, whose seed depends on method name. Restore real names and
    recompute Holm over the complete registered family, never a two-method family.
    """
    methods = sorted(methods)
    if len(methods) != len(set(methods)) or baseline not in methods:
        raise ValueError('Invalid method family for paired execution statistics')
    if any(row['method'] not in methods for row in rows):
        raise ValueError('Unregistered method in paired execution rows')
    aliases = {baseline: 'paired_reference'}
    comparisons, template = [], None
    for method in methods:
        if method == baseline:
            continue
        aliases[method] = f'paired_comparison_rank{rank}'
        paired_rows = [{**row, 'method': aliases[row['method']]} for row in rows if row['method'] in (method, baseline)]
        value = r.ev.paired_method_statistics(paired_rows, [aliases[baseline], aliases[method]],
            baseline=aliases[baseline], draws=draws)
        template = value
        for row in value['comparisons']:
            comparisons.append({**row, 'method': method, 'baseline': baseline,
                'holm_adjusted_p': None, 'holm_prespecified_family_size': len(methods) - 1})
    families = defaultdict(list)
    for row in comparisons:
        families[(row['stratum'], row['endpoint'])].append(row)
    for family in families.values():
        previous = 0.
        ordered = sorted(family, key=lambda row: (row['two_sided_p'] if row['two_sided_p'] is not None else 1., row['method']))
        for index, row in enumerate(ordered):
            pvalue = row['two_sided_p'] if row['two_sided_p'] is not None else 1.
            previous = max(previous, min(1., (len(methods) - 1 - index) * pvalue))
            if row['two_sided_p'] is not None:
                row['holm_adjusted_p'] = previous
    if template is None:
        raise ValueError('Paired comparison requires at least one method and a baseline')
    return {**template, 'baseline': baseline,
        'comparisons': sorted(comparisons, key=lambda row: (row['stratum'], row['endpoint'], row['method'])),
        'randomization': 'Common method-neutral draws per rank/endpoint; sorted compound/publication blocks; identical paired errors receive identical Monte Carlo draws.'}


def paired(reports, data):
    methods = sorted(reports)
    groups = {row['query_id']: data['groups'][i] for i, row in enumerate(data['query_records'])}
    quality = stable_paired_quality({m: v['quality'] for m, v in reports.items()}, groups, baseline=r.BASELINE)
    execution = []
    for rank in (1, 2, 3):
        rows = [row for report in reports.values() for row in report['execution']['position_rows'] if row['rank'] == rank]
        result = stable_paired_execution(rows, methods, baseline=r.BASELINE, rank=rank)
        result.update(rank=rank, comparison_family=methods,
            multiplicity='Holm across the registered family versus F2LLM separately for each exact rank, stratum and endpoint; unavailable tests count as p=1.')
        execution.append(result)
    return {'baseline': r.BASELINE, 'comparison_family': methods, 'quality': quality, 'execution': execution,
        'randomization': 'Fixed common draws independent of method or row iteration order; this statistical reproducibility repair does not select favorable p values.'}


def common_coverage_comparisons(reports, own=OWN):
    """Descriptive same-scenario GMFE; never compare conditional different pools."""
    result = []
    own_rows = reports[own]['execution']['position_rows']
    for other, report in reports.items():
        if other == own:
            continue
        reference = {(v['endpoint'], v['rank'], v['scenario_id']): v for v in report['execution']['position_rows']}
        for endpoint in sorted({row['endpoint'] for row in own_rows}):
            for rank in (1, 2, 3):
                a, b = [], []
                for row in own_rows:
                    target = reference.get((endpoint, rank, row['scenario_id']))
                    if row['endpoint'] != endpoint or row['rank'] != rank or target is None:
                        continue
                    if all(math.isclose(item['coverage'], 1., rel_tol=0., abs_tol=1e-12) for item in (row, target)):
                        a.append(row); b.append(target)
                av, bv = r.ev.hierarchical_mean(a, 'absolute_log_error'), r.ev.hierarchical_mean(b, 'absolute_log_error')
                result.append({'method': own, 'comparator': other, 'endpoint': endpoint, 'rank': rank,
                    'shared_scenarios': len(a), 'scenario_ids': [v['scenario_id'] for v in a],
                    'own_gmfe': math.exp(av) if av is not None else None,
                    'comparator_gmfe': math.exp(bv) if bv is not None else None})
    return result


def omission_summary(rows):
    """Use the paper's equal-compound, within-compound equal-study estimand.

    Dose branches from one study first contribute their mean. Cases with fewer
    than two measured candidates do not have an identifiable selection miss and
    remain unavailable, rather than being treated as a successful retrieval.
    """
    summaries = []
    for endpoint in sorted({row['endpoint'] for row in rows}):
        source = [row for row in rows if row['endpoint'] == endpoint]
        for k in (3, 5):
            chosen = []
            for row in source:
                value = row['by_k'][str(k)]['best_measured_miss_probability']
                if value is not None:
                    chosen.append({**row, 'best_measured_miss_probability': value})
            miss = r.ev.hierarchical_mean(chosen, 'best_measured_miss_probability')
            summaries.append({'endpoint': endpoint, 'k': k, 'total_scenarios': len(source),
                'choice_comparison_scenarios': len(chosen),
                'compounds': len({row['compound'] for row in chosen}),
                'compound_study_pairs': len({(row['compound'], row['publication_cluster_id']) for row in chosen}),
                'best_measured_miss_probability': miss,
                'best_measured_hit_probability': 1 - miss if miss is not None else None,
                'aggregation': 'Mean dose/scenario branches within compound/study; mean studies within compound; mean compounds.',
                'interpretation': 'Measured-candidate numerical omission only; unmeasured candidates remain unknown.'})
    return summaries


def paper_tables(reports):
    number = lambda v: '--' if v is None else f'{v:.3f}'
    esc = lambda v: str(v).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
    start = (r'\begin{center}\small\setlength{\tabcolsep}{3pt}'
             r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X'
             r'*{4}{>{\centering\arraybackslash}X}@{}}\toprule')
    end = r'\bottomrule\end{tabularx}\end{center}'
    doc = ['\\paragraph{Grouped internal validation of the in-house method.} The frozen text encoder is reused; only the lightweight matcher and error branch are trained. '
           'Five grouped outer folds and three inner selection folds use seed 17. Controls reuse completed ranking scores, '
           'and applicability probabilities are calibrated on the same outer-training folds. NDCG, Recall, and MRR below use only requests with known relevant candidates. '
           'Unknown labels are not filled as negatives. Controls with equal supervision were not additionally trained, so differences cannot be attributed solely to network structure.',
           start, 'Method & NDCG@10 & Recall@10 & MRR & Applicability F1 \\\\ \\midrule']
    execution = ['\\paragraph{Prediction error of the in-house method.} The table reports ordinary Cmax, with coverage/GMFE in each cell. '
                 'Positions 1, 2, and 3 are evaluated separately, with equal-probability ties and no replacement by lower candidates. '
                 'Conditional GMFE at different coverage cannot establish overall superiority; paired shared-scenario analysis is in the result files.',
                 start, 'Method & Position 1 & Position 2 & Position 3 & Top-5 omission \\\\ \\midrule']
    stability = ['\\paragraph{Ranking stability of the in-house method.} Sentence reordering and evidence removal test wording stability and evidence sensitivity respectively. '
                 'The first three columns are top-set Jaccard, top-set Jaccard, and the original-request top-tie rate. '
                 'Repeat, batch-size, and candidate-order checks are additionally run on held-out requests for each outer model.',
                 start, 'Method & Reordered sentences & Removed evidence & Top-tie rate & Requests \\\\ \\midrule']
    endpoint = 'Cmax_reference_vs_group'
    for method, report in reports.items():
        name = esc(DISPLAY.get(method, method))
        q = report['quality']; metric = q['judged_only']['metrics']
        doc.append(' & '.join([name, *(number(metric[k]) for k in ('ndcg@10', 'recall@10', 'mrr')),
                              number(q['applicability']['per_class_f1']['Y'])]) + r' \\')
        ranks = {v['rank']: v for v in report['execution']['position_summary'] if v['endpoint'] == endpoint}
        cells = [f"{100*ranks[k]['coverage']:.1f}\\%/{number(ranks[k]['gmfe'])}" if k in ranks else '--' for k in (1, 2, 3)]
        omission = next((row for row in report['execution']['omission_summary']
                         if row['endpoint'] == endpoint and row['k'] == 5), None)
        execution.append(' & '.join([name, *cells,
            number(omission['best_measured_miss_probability']) if omission else '--']) + r' \\')
        st = report['stability']
        stability.append(' & '.join([name, number(st['reordered']['top1_set_jaccard']),
            number(st['missing_evidence']['top1_set_jaccard']), number(st['top1_tie_fraction']),
            str(st['reordered']['total_queries'])]) + r' \\')
    own = reports[OWN]['quality']
    doc += [end, f"Scorable ranking requests: {own['judged_only']['eligible_queries']}/{own['coverage']['queries']}; "
        f"in-house top-1 insufficient-information fraction: {number(own['coverage']['full_catalog_top1_unknown_rate'])}, "
        f"Brier={number(own['applicability']['brier'])}, ECE={number(own['applicability']['ece'])}."]
    execution += [end, 'Top-5 omission is calculated only where at least two candidates have measured endpoints. Scenarios are first aggregated within drug and study, '
                  'then averaged equally by study and drug. Unmeasured candidates remain unknown rather than being inferred unusable.']
    stability.append(end)
    return {name: '\n'.join(lines) + '\n' for name, lines in
            (('paper_document.tex', doc), ('paper_execution.tex', execution), ('paper_stability.tex', stability))}


def summarize(data=None, output=OUT):
    output = scope(output)
    if data is None:
        from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
        data = load_dataset()
    protocol, external, arrays, refs, integrity, selections = load_inputs(data, output)
    contract = r.ev.clinical_contract()
    refs.append(ref(r.ev.PRIMARY))
    reports, calibrations = {}, {}
    for method, scores in external.items():
        p, calibrations[method] = r.external_probabilities(data, scores[0])
        reports[method] = r._evaluate(data, scores, p, method, contract)
    for method, name in ((DOCUMENT, 'base_scores'), (OWN, 'scores')):
        reports[method] = r._evaluate(data, arrays[name], arrays['probabilities'][0], method, contract)
    for report in reports.values():
        report['execution']['omission_summary'] = omission_summary(report['execution']['omissions'])
    compact = r._compact(reports)
    for row in compact['quality']:
        q = reports[row['method']]['quality']
        row['applicable_Y_f1'] = q['applicability']['per_class_f1']['Y']
        row['same_parent_preference'] = q['same_drug_different_family_preference']['query_macro_accuracy']
    result = {'schema': 'pbpk-match-light-report-v1', 'state': 'complete', 'completed': True,
        'producer': {'path': 'scripts/summarize_pbpk_match_light.py', 'sha256': r.sha(__file__)},
        'seeds': [SEED], 'trained_methods': [OWN], 'separate_document_only_fit': False,
        'external_models_rerun': False, 'abandoned_lora_protocol_completed': False,
        'query_count': len(data['query_records']), 'candidate_count': len(data['candidate_records']),
        'document_request_count': len(data['core_indices']), 'execution_request_count': len(data['execution_indices']),
        'stability_request_count': len(data['stability_indices']),
        'completed_small_network_fits': sum(len(f['inner']) + 1 for f in data['folds']),
        'reports': reports, 'external_calibration': calibrations, 'outer_fold_selection': selections,
        'inference_integrity': integrity, 'paired': paired(reports, data),
        'common_fully_measurable_execution': common_coverage_comparisons(reports),
        'limitations': ['Grouped internal validation of existing data; no newly independent validation data.',
            'Only seed17 is trained; no seed variability or ensemble gain is claimed.',
            'Comparisons reuse completed frozen external scores. No equal-supervision baseline is trained in this user-selected scope.',
            'Document labels and measured PK errors remain distinct; unknown labels/outcomes are not converted to negatives.',
            'Native PBPK parameters were not refitted. Reference-person numerical diagnostics are not validated direct-reuse labels.',
            'Exact ranks1/2/3 use uniform ties and no backfill. Endpoint windows are not pooled.',
            'Conditional GMFE requires coverage; paired comparisons use the same fully measurable scenarios.',
            'Available curves support case analysis only.'], 'input_refs': refs,
        'protocol_sha256': r.sha(output / 'protocol.json')}
    for value in refs:
        if r.sha(scope(value['path'])) != value['sha256']:
            raise ValueError('Input changed while computing light report: ' + value['path'])
    destination = output / 'report'
    destination.mkdir(parents=True, exist_ok=True)
    r.write(destination / 'summary.json', result)
    files = ['summary.json']
    for name, rows in compact.items():
        path = destination / f'{name}.csv'
        path.write_text(r._csv(rows)); files.append(path.name)
    extra = {'document_per_request': [dict(method=m, **row) for m, v in reports.items() for row in v['quality']['per_query']],
        'execution_exact_positions': [row for v in reports.values() for row in v['execution']['position_rows']],
        'execution_omissions': [row for v in reports.values() for row in v['execution']['omissions']],
        'execution_omission_summary': [dict(method=m, **row) for m, v in reports.items()
                                       for row in v['execution']['omission_summary']],
        'common_execution': result['common_fully_measurable_execution']}
    for name, rows in extra.items():
        path = destination / f'{name}.csv'; path.write_text(r._csv(rows)); files.append(path.name)
    for name, value in paper_tables(reports).items():
        (destination / name).write_text(value); files.append(name)
    r.write(destination / 'completion.json', {'state': 'complete', 'completed': True,
        'scope': 'single_seed_light_own_model_and_reused_external_scores', 'outputs': [ref(destination / name) for name in files],
        'protocol_sha256': result['protocol_sha256'], 'producer': result['producer'], 'inputs_verified': len(refs)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    value = summarize(output=args.output)
    print(json.dumps({'state': value['state'], 'methods': list(value['reports']),
                      'fits': value['completed_small_network_fits']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
