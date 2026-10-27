"""Reproducible, outcome-separated reporting for the complete grouped experiment.

Single-seed variability is not a confidence interval. Ensembles average exact-tie
percentile ranks, never incompatible raw score scales. External probability
calibrators use outer-training document labels only, never PK outcomes.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import warnings

import numpy as np

from . import evaluation as ev
from .experiment import OUT, ROOT, METHODS, SEEDS, EXTERNAL, write, sha

BASELINE = 'f2llm_v2_06b'
NO_ERROR = 'no_error_branch'
DISPLAY = {'bm25_text': 'BM25', 'qwen3_embedding_06b': 'Qwen3-Emb 0.6B',
    'f2llm_v2_06b': 'F2LLM v2', 'qwen3_reranker_06b': 'Qwen3-Reranker 0.6B',
    'jina_reranker_v35': 'Jina v3.5', 'bm25_top20_rearank_7b_nf4': 'REARANK',
    'bm25_top20_reasonrank_7b_nf4': 'ReasonRank', 'prorank': 'ProRank', 'contextual': 'Contextual v2',
    'full': 'PBPK-Match v2', 'no_numeric': 'No numeric conditions', 'frozen_text': 'Frozen encoder',
    'mlp': 'MLP', 'lambdamart': 'LambdaMART', 'no_error_branch': 'No error branch'}


def percentile_ranks(scores):
    """Ascending 0..1 ranks; all exact ties receive their midpoint rank."""
    x = np.asarray(scores, float)
    if x.ndim < 1 or not x.shape[-1] or not np.isfinite(x).all():
        raise ValueError('Percentile ranks require complete finite candidate scores')
    result = np.empty_like(x)
    for a, b in zip(x.reshape(-1, x.shape[-1]), result.reshape(-1, x.shape[-1])):
        levels, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        midpoint = np.cumsum(counts) - (counts + 1) / 2
        b[:] = midpoint[inverse] / (len(a) - 1) if len(a) > 1 else .5
    return result


def _probabilities(value, shape, *, allow_unavailable=False):
    p = np.asarray(value, float)
    if p.shape != (*shape, 3):
        raise ValueError('Probability axes must be [Q,C,3]')
    available = np.isfinite(p).all(axis=(1, 2))
    if not allow_unavailable and not available.all():
        raise ValueError('Incomplete probabilities')
    if (not np.isfinite(p[available]).all() or (p[available] < 0).any() or
            (p[available] > 1).any() or not np.allclose(p[available].sum(-1), 1, atol=1e-7, rtol=0)):
        raise ValueError('Probabilities must be finite normalized Y,N,U')
    if allow_unavailable and not np.isnan(p[~available]).all():
        raise ValueError('Unavailable probability rows must be wholly NaN')
    return p, available


def external_probabilities(data, scores):
    """Five outer cross-fitted scalar multinomial calibrators; no test labels."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    x = np.asarray(scores, float)
    y = np.asarray(data['applicability'], int)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isin(y, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid external calibration inputs')
    output = np.full((*x.shape, 3), np.nan)
    groups = np.asarray(data['groups'])
    covered, records = set(), []
    for fold_id, fold in enumerate(data['folds']):
        train, test = np.asarray(fold['outer_train'], int), np.asarray(fold['outer_test'], int)
        if (set(train) & set(test) or set(train) | set(test) != set(range(len(x))) or
                covered & set(test) or set(groups[train]) & set(groups[test])):
            raise ValueError('Calibration fold leakage or duplicate outer rows')
        covered.update(test)
        labels, values = y[train], x[train]
        valid = labels >= 0
        support = [int((labels == i).sum()) for i in range(3)]
        record = {'fold': fold_id, 'fit_query_indices': train.tolist(), 'test_query_indices': test.tolist(),
            'fit_groups': sorted(set(groups[train].tolist())), 'test_groups': sorted(set(groups[test].tolist())),
            'class_order': ['Y', 'N', 'U'], 'class_support': support,
            'missing_classes': [i for i, count in enumerate(support) if not count],
            'weighting': 'equal input-connected group, then query, then valid pair',
            'uses_clinical_errors': False, 'state': 'unavailable'}
        if sum(n > 0 for n in support) < 2:
            record['reason'] = 'fewer_than_two_observed_outer_training_classes'
            records.append(record)
            continue
        weights = np.zeros(values.shape)
        for group in set(groups[train]):
            rows = [j for j in range(len(train)) if groups[train[j]] == group and valid[j].any()]
            for j in rows:
                weights[j, valid[j]] = 1 / len(rows) / valid[j].sum()
        weights /= weights.sum()
        mean = float(np.sum(weights[valid] * values[valid]))
        scale = float(np.sqrt(np.sum(weights[valid] * (values[valid] - mean) ** 2)))
        if scale == 0:
            scale = 1.
        model = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, random_state=20261004)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(((values[valid] - mean) / scale)[:, None], labels[valid],
                      sample_weight=weights[valid] * valid.sum())
        probabilities = model.predict_proba(((x[test] - mean) / scale).reshape(-1, 1)).reshape(len(test), x.shape[1], -1)
        output[test] = 0.
        for column, label in enumerate(model.classes_):
            output[test, :, int(label)] = probabilities[:, :, column]
        record.update(state='fitted', reason=None, train_mean=mean, train_std=scale,
            fitted_classes=model.classes_.tolist(), coefficients=model.coef_.tolist(),
            intercept=model.intercept_.tolist(), iterations=model.n_iter_.tolist(), regularization_C=1.)
        records.append(record)
    if covered != set(range(len(x))):
        raise ValueError('External probability calibration does not cover all outer rows')
    _probabilities(output, x.shape, allow_unavailable=True)
    return output, {'method': 'outer_crossfit_scalar_logistic_regression', 'folds': records,
                    'ranking_unchanged': True, 'class_absence_not_recoded': True}


def risk_coverage(data, scores, probabilities):
    """Fixed-threshold top-1 selective documented-support risk, unknowns retained."""
    p, available = _probabilities(probabilities, np.asarray(scores).shape, allow_unavailable=True)
    labels = np.asarray(data['applicability'])
    requests = []
    for i in data['core_indices']:
        weights = ev.exact_rank_weights(scores[i], 1)
        known = (labels[i] == 0) | (labels[i] == 1)
        known_mass = float(weights[known].sum())
        requests.append({'query_id': data['query_records'][i]['query_id'],
            'calibration_available': bool(available[i]),
            'confidence': float(weights @ p[i, :, 0]) if available[i] else None,
            'known_label_probability': known_mass,
            'unknown_label_probability': 1 - known_mass,
            'conditional_error': float(weights[labels[i] == 1].sum() / known_mass) if known_mass else None})
    points = []
    for threshold in np.linspace(0., 1., 11):
        chosen = [r for r in requests if r['confidence'] is not None and r['confidence'] >= threshold]
        risk = [r['conditional_error'] for r in chosen if r['conditional_error'] is not None]
        points.append({'threshold': float(threshold), 'accepted_requests': len(chosen),
            'total_requests': len(requests), 'recommendation_coverage': len(chosen) / len(requests) if requests else None,
            'known_label_request_count': len(risk), 'conditional_documented_N_risk': float(np.mean(risk)) if risk else None,
            'mean_unknown_label_probability': float(np.mean([r['unknown_label_probability'] for r in chosen])) if chosen else None})
    return {'interpretation': 'Documented Y/N top-1 risk; U and invalid labels remain unknown, not PK failure labels.',
            'thresholds_prespecified': True, 'points': points, 'requests': requests}


def _evaluate(data, scores, probabilities, method, contract):
    p, available = _probabilities(probabilities, scores.shape[1:], allow_unavailable=True)
    core = list(data['core_indices'])
    all_core_available = bool(available[core].all())
    quality = ev.quality_evaluation(data, scores[0], p if all_core_available else None)
    if not all_core_available:
        eligible = [i for i in core if available[i]]
        quality['calibrated_subset'] = ev.quality_evaluation(data, scores[0], p, indices=eligible) if eligible else None
    quality['calibration_coverage'] = {'eligible_queries': int(available[core].sum()), 'total_queries': len(core)}
    quality['risk_coverage'] = risk_coverage(data, scores[0], p)
    indices = data['execution_indices']
    clinical = ev.clinical_evaluation(scores[0, indices], [data['query_records'][i]['query_id'] for i in indices],
                [c['candidate_id'] for c in data['candidate_records']], method=method, contract=contract)
    common = []
    for endpoint in sorted({r['endpoint'] for r in clinical['position_rows']}):
        by_rank = {rank: {r['scenario_id']: r for r in clinical['position_rows'] if r['endpoint'] == endpoint and r['rank'] == rank}
                   for rank in (1, 2, 3)}
        shared = sorted(set.intersection(*(set(sid for sid, row in by_rank[rank].items()
                        if math.isclose(row['coverage'], 1., abs_tol=1e-12, rel_tol=0)) for rank in (1, 2, 3))))
        rows = [by_rank[1][sid] for sid in shared]
        item = {'endpoint': endpoint, 'scenario_ids': shared, 'scenarios': len(shared),
                'compounds': len({row['compound'] for row in rows}),
                'interpretation': 'Descriptive same fully measurable scenario subset at all three exact positions.'}
        for rank in (1, 2, 3):
            value = ev.hierarchical_mean([by_rank[rank][sid] for sid in shared], 'absolute_log_error')
            item[f'rank{rank}_gmfe'] = math.exp(value) if value is not None else None
        common.append(item)
    clinical['common_positions'] = common
    return {'quality': quality, 'execution': clinical, 'stability': ev.stability_evaluation(data, scores)}


def _flatten_numbers(obj, prefix=''):
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result.update(_flatten_numbers(value, prefix + ('.' if prefix else '') + key))
    elif isinstance(obj, (int, float, np.number)) and not isinstance(obj, (bool, np.bool_)) and math.isfinite(obj):
        result[prefix] = float(obj)
    return result


def seed_variability(reports):
    """Mean/sample SD of each already-evaluated seed metric; never pooled rows."""
    sections = []
    for report in reports:
        quality = {k: v for k, v in report['quality'].items() if k not in ('per_query', 'risk_coverage')}
        execution = {f"{r['endpoint']}|rank{r['rank']}": r for r in report['execution']['position_summary']}
        sections.append(_flatten_numbers({'quality': quality, 'execution': execution, 'stability': report['stability']}))
    keys = sorted(set().union(*(x.keys() for x in sections)))
    result = {}
    for key in keys:
        values = [x[key] for x in sections if key in x]
        result[key] = {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) >= 2 else None,
                       'available_seeds': len(values), 'total_seeds': len(reports)}
    return result


def _scope(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or 'sealed' in path.relative_to(ROOT).parts:
        raise ValueError('Reporting input/output must remain in project, outside sealed data')
    return path


def _ref(path):
    path = _scope(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def _integrity(path, indices, candidates, refs, pending, name, expected_sha=None):
    if not path.exists():
        pending.append('integrity/' + name)
        return None
    if expected_sha is None or sha(path) != expected_sha:
        raise ValueError('Inference integrity is not bound to completion receipt')
    value = json.loads(path.read_text())
    if value.get('state') != 'complete' or value.get('indices') != list(indices):
        raise ValueError('Inference integrity request scope differs')
    batches = value.get('batch_sizes', [])
    if len(batches) != 2 or any(type(n) is not int or n < 1 for n in batches) or batches[1] != max(1, batches[0] // 2):
        raise ValueError('Inference integrity batch configuration differs')
    comparisons = ['repeat', 'candidate_order', 'batch_size']
    documents = ['document_repeat', 'document_candidate_order', 'document_batch_size']
    if any(key in value for key in documents):
        if not all(key in value for key in documents):
            raise ValueError('Incomplete no-error-branch integrity comparisons')
        comparisons += documents
    for kind in comparisons:
        row = value.get(kind, {})
        if row.get('queries') != len(indices) or row.get('candidate_count') != candidates or type(row.get('exact_equal')) is not bool:
            raise ValueError('Inference integrity population differs')
        delta, agreement, correlation = (row.get(k) for k in ('max_absolute_score_delta', 'top1_tie_set_agreement', 'mean_spearman'))
        if (any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in (delta, agreement, correlation))
                or delta < 0 or not 0 <= agreement <= 1 or not -1 <= correlation <= 1):
            raise ValueError('Invalid inference integrity diagnostics')
        if row['exact_equal'] and (delta != 0 or agreement != 1 or not math.isclose(correlation, 1., abs_tol=1e-12)):
            raise ValueError('Contradictory exact-equality integrity claims')
    refs.append(_ref(path))
    return value


def _load_inputs(data, output):
    """Reject corrupt completed artifacts; preserve pending items explicitly."""
    q, c = len(data['query_records']), len(data['candidate_records'])
    expected_shape = (3, q, c)
    refs, pending, external, trained, integrity = [], [], {}, {}, {}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        expected = {'methods': list(METHODS), 'external_methods': list(EXTERNAL), 'seeds': list(SEEDS),
            'query_ids': [r['query_id'] for r in data['query_records']],
            'candidate_ids': [r['candidate_id'] for r in data['candidate_records']], 'folds': data['folds']}
        if any(protocol.get(k) != value for k, value in expected.items()):
            raise ValueError('Reporting protocol/data axes or registered methods differ')
        manifest = output / 'manifest.json'
        if not manifest.exists() or sha(manifest) != protocol['dataset_manifest_sha256']:
            raise ValueError('Dataset manifest binding changed')
        refs += [_ref(protocol_path), _ref(manifest)]
        for path, digest in protocol['implementation_sha256'].items():
            target = _scope(ROOT / path)
            if sha(target) != digest:
                raise ValueError('Frozen report implementation drift: ' + path)
            refs.append(_ref(target))
        for section in ('legacy_artifact_sha256', 'qwen_files_sha256'):
            for path, digest in protocol.get(section, {}).items():
                target = _scope(ROOT / path)
                if sha(target) != digest:
                    raise ValueError('Frozen model/source artifact drift: ' + path)
                refs.append(_ref(target))
        if protocol.get('external_weight_receipts_sha256'):
            from .external import WEIGHTS, MODEL_SPECS, verify_weights
            for method, digest in protocol['external_weight_receipts_sha256'].items():
                model_path = WEIGHTS / method / MODEL_SPECS[method]['revision']
                if sha(model_path / 'model_receipt.json') != digest:
                    raise ValueError('Pinned external model receipt changed')
                receipt = verify_weights(method, model_path)
                refs += [_ref(model_path / 'model_receipt.json')]
                refs += [_ref(model_path / record['name']) for record in receipt['files']]
    else:
        protocol = None
    for method in EXTERNAL:
        folder = output / 'external' / method
        receipt_path, path = folder / 'complete.json', folder / 'scores.npz'
        if not receipt_path.exists():
            pending.append('external/' + method)
            continue
        if protocol is None:
            raise ValueError('Completed score lacks frozen protocol')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('state') != 'complete' or sha(path) != receipt.get('score_sha256'):
            raise ValueError('External completion/SHA mismatch: ' + method)
        with np.load(path, allow_pickle=False) as saved:
            scores = saved['scores'].copy()
        if scores.shape != expected_shape or not np.isfinite(scores).all():
            raise ValueError('External shape/finite mismatch: ' + method)
        refs += [_ref(path), _ref(receipt_path)]
        external[method] = scores
        if method in ('prorank', 'contextual'):
            chosen = [data['core_indices'][0], data['execution_indices'][0]]
            audit = _integrity(folder / 'integrity.json', chosen, c, refs, pending,
                method, receipt.get('artifacts_sha256', {}).get('integrity.json'))
            integrity[method] = [audit] if audit is not None else []
    for method in METHODS:
        trained[method] = {}
        for seed in SEEDS:
            path = output / 'predictions' / f'{method}_seed{seed}.npz'
            seed_receipt_path = path.with_suffix('.json')
            if not path.exists() or not seed_receipt_path.exists():
                pending.append(f'{method}/seed{seed}')
                continue
            if protocol is None:
                raise ValueError('Completed prediction lacks frozen protocol')
            seed_receipt = json.loads(seed_receipt_path.read_text())
            if (seed_receipt.get('state') != 'complete' or seed_receipt.get('score_sha256') != sha(path)
                    or seed_receipt.get('protocol_sha256') != sha(protocol_path)
                    or set(seed_receipt.get('outer_receipts', {})) != {str(i) for i in range(len(data['folds']))}):
                raise ValueError('Trained prediction receipt or protocol binding differs')
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in ('scores', 'document_scores', 'probabilities')}
            if any(arrays[key].shape != expected_shape or not np.isfinite(arrays[key]).all() for key in ('scores', 'document_scores')):
                raise ValueError('Trained score shape/finite mismatch')
            _probabilities(arrays['probabilities'], (q, c))
            # Reassemble from true outer OOF rows; a final file alone is not completion.
            covered = set()
            for fold_id, fold in enumerate(data['folds']):
                directory = output / 'runs' / method / f'seed{seed}' / f'fold{fold_id}' / 'outer'
                completed, predictions = directory / 'complete.json', directory / 'predictions.npz'
                if not completed.exists() or not predictions.exists():
                    raise ValueError('Final seed artifact lacks complete outer-fold provenance')
                if sha(completed) != seed_receipt['outer_receipts'][str(fold_id)]:
                    raise ValueError('Outer completion receipt changed')
                metadata = json.loads(completed.read_text())
                artifacts = metadata.get('artifacts_sha256', {})
                if not {'binding.json', 'predictions.npz'} <= set(artifacts):
                    raise ValueError('Outer completion lacks binding or prediction hashes')
                for relative, digest in artifacts.items():
                    artifact = _scope(directory / relative)
                    if not artifact.is_relative_to(directory.resolve()) or sha(artifact) != digest:
                        raise ValueError('Completed outer artifact changed')
                    refs.append(_ref(artifact))
                binding = json.loads((directory / 'binding.json').read_text())
                if (binding.get('protocol_sha256') != sha(protocol_path) or
                    binding.get('dataset_sha256') != protocol['dataset_manifest_sha256'] or
                    binding.get('method') != method or binding.get('seed') != seed or
                    binding.get('prediction_indices') != fold['outer_test'] or binding.get('validation_indices') != [] or
                    binding.get('train_indices') != sorted((set(data['core_indices']) | set(data['execution_indices'])) & set(fold['outer_train']))):
                    raise ValueError('Outer training binding differs from registered fold')
                audit = _integrity(directory / 'integrity.json', sorted(fold['outer_test'])[:2], c, refs, pending,
                    f'{method}/seed{seed}/fold{fold_id}', artifacts.get('integrity.json'))
                if audit is not None:
                    integrity.setdefault(method, []).append({'seed': seed, 'fold': fold_id, **audit})
                    if method == 'full' and 'document_repeat' not in audit:
                        pending.append(f'integrity/{method}/seed{seed}/fold{fold_id}/document_scores')
                indices = np.asarray(fold['outer_test'], int)
                with np.load(predictions, allow_pickle=False) as saved:
                    if not np.array_equal(saved['indices'], indices) or covered & set(indices):
                        raise ValueError('Outer-fold prediction membership differs')
                    covered.update(indices)
                    for variant in range(3):
                        for key in ('scores', 'document_scores'):
                            if not np.array_equal(saved[f'{key}_{variant}'], arrays[key][variant, indices]):
                                raise ValueError('Assembled seed scores differ from outer predictions')
                    if not np.array_equal(saved['probabilities'], arrays['probabilities'][indices]):
                        raise ValueError('Assembled seed probabilities differ from outer predictions')
                refs += [_ref(completed), _ref(predictions)]
            if covered != set(range(q)):
                raise ValueError('Outer-fold predictions omit queries')
            trained[method][seed] = arrays
            refs += [_ref(path), _ref(seed_receipt_path)]
    return external, trained, pending, refs, integrity


def _paired(reports, data, *, complete=True):
    expected = set(EXTERNAL) | set(METHODS) | {NO_ERROR}
    if not complete or set(reports) != expected:
        return {'state': 'pending_registered_method_family', 'pending': sorted(expected - set(reports)), 'baseline': BASELINE}
    if BASELINE not in reports:
        return {'state': 'pending_baseline', 'baseline': BASELINE}
    methods = list(reports)
    groups = {r['query_id']: data['groups'][i] for i, r in enumerate(data['query_records'])}
    quality = ev.paired_quality({m: r['quality'] for m, r in reports.items()}, groups, baseline=BASELINE)
    clinical = []
    for rank in (1, 2, 3):
        rows = [r for report in reports.values() for r in report['execution']['position_rows'] if r['rank'] == rank]
        result = ev.paired_method_statistics(rows, methods, baseline=BASELINE)
        # Existing implementation carries a historical BM25 wording; numeric baseline is explicit above.
        result['multiplicity'] = 'Holm over all registered methods versus F2LLM within this rank/stratum/endpoint; unavailable comparisons count as p=1.'
        result['rank'] = rank
        clinical.append(result)
    return {'state': 'computed', 'baseline': BASELINE, 'quality': quality, 'execution': clinical,
            'interpretation': 'No pooling of training seeds as independent studies; each rank/endpoint has its own declared comparison family.'}


def _compact(reports):
    quality, execution, stability = [], [], []
    for method, report in reports.items():
        q = report['quality']
        quality.append({'method': method, **q['judged_only']['metrics'],
            'judged_queries': q['judged_only']['eligible_queries'], 'total_queries': q['coverage']['queries'],
            'unknown_top1': q['coverage']['full_catalog_top1_unknown_rate'],
            **{k: q['applicability'][k] for k in ('macro_f1', 'brier', 'ece')}})
        execution += [{**r, 'method': method} for r in report['execution']['position_summary']]
        for variant in ('reordered', 'missing_evidence'):
            stability.append({'method': method, **report['stability'][variant],
                              'top1_tie_fraction': report['stability']['top1_tie_fraction']})
    return {'quality': quality, 'execution': execution, 'stability': stability}


def _csv(rows):
    if not rows:
        return ''
    output = io.StringIO()
    columns = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def _tex(rows, complete):
    esc = lambda x: str(x).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
    number = lambda x: '--' if x is None else f'{x:.3f}'
    lines = ['% Descriptive internal grouped evaluation; no independent-validation claim.',
        '% ' + ('COMPLETE' if complete else 'PARTIAL: not all prespecified runs exist'),
        r'\begin{tabular}{lrrr}', r'Method & NDCG@10 & Recall@10 & MRR \\', r'\hline']
    lines += [f"{esc(r['method'])} & {number(r['ndcg@10'])} & {number(r['recall@10'])} & {number(r['mrr'])} \\\\" for r in rows]
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def _paper_tables(reports, variability):
    """Compact ordinary-Cmax table, with all other endpoint rows kept in CSV/JSON."""
    esc = lambda text: str(text).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
    number = lambda value: '--' if value is None else f'{value:.3f}'
    def seeds(method, key):
        row = variability.get(method, {}).get(key)
        if not row or row['available_seeds'] != len(SEEDS):
            return '--'
        return f"{row['mean']:.3f} $\\pm$ {row['sd']:.3f}" if row['sd'] is not None else '--'
    table_start = (r'\begin{center}\small\setlength{\tabcolsep}{3pt}'
                   r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X'
                   r'*{4}{>{\centering\arraybackslash}X}@{}}\toprule')
    table_end = r'\bottomrule\end{tabularx}\end{center}'
    document = ['\\paragraph{Grouped internal validation.} Results use existing data; documentary support and actual PK errors are evaluated separately. '
                'Trained methods report three-seed means and sample standard deviations; ensembles average tied percentile ranks equally within each query. External methods do not have fabricated seed variation. ',
                'Both NDCG columns use the first ten positions. Single-run values are seed mean $\\pm$ standard deviation; other columns are ensemble or external-method results.', table_start,
                'Method & Single-run NDCG & Ensemble NDCG & Recall@10 & MRR \\\\ \\midrule']
    execution = ['\\paragraph{Exact positions and coverage.} The table reports ordinary Cmax only; different AUC windows and curve endpoints remain separate in exports. '
                 'Each cell is coverage/conditional GMFE. Positions 2 and 3 are not backfilled or replaced by the best of three; exact ties receive uniform weights. '
                 'Conditional errors at different coverage do not establish overall superiority. Numerical comparisons between a reference individual and a study population do not provide direct-reuse labels. ',
                 'The single-run column is the seed mean $\\pm$ standard deviation of position-1 GMFE.', table_start,
                 'Method & Single-run GMFE & Position 1 & Position 2 & Position 3 \\\\ \\midrule']
    stability = ['\\paragraph{Stability.} Sentence reordering tests wording invariance; evidence removal tests sensitivity. Neither substitutes for recommendation correctness. '
                 'Jaccard uses complete top-tie sets without breaking ties by model identifier. ',
                 'The single-run column is seed mean $\\pm$ standard deviation of sentence-reordering Jaccard.', table_start,
                 'Method & Single-run reorder & Reordered & Missing evidence & Tie rate \\\\ \\midrule']
    for method, report in reports.items():
        q = report['quality']['judged_only']['metrics']; name = esc(DISPLAY.get(method, method))
        document.append(' & '.join([name, seeds(method, 'quality.judged_only.metrics.ndcg@10'),
                        number(q['ndcg@10']), number(q['recall@10']), number(q['mrr'])]) + r' \\')
        endpoint = 'Cmax_reference_vs_group'
        by_rank = {row['rank']: row for row in report['execution']['position_summary'] if row['endpoint'] == endpoint}
        cells = []
        for rank in (1, 2, 3):
            row = by_rank.get(rank)
            cells.append((f"{100*row['coverage']:.1f}\\%/{number(row['gmfe'])}") if row else '--')
        execution.append(' & '.join([name, seeds(method, f'execution.{endpoint}|rank1.gmfe'), *cells]) + r' \\')
        st = report['stability']
        stability.append(' & '.join([name, seeds(method, 'stability.reordered.top1_set_jaccard'),
            number(st['reordered']['top1_set_jaccard']), number(st['missing_evidence']['top1_set_jaccard']),
            number(st['top1_tie_fraction'])]) + r' \\')
    endings = {'paper_document.tex': [], 'paper_execution.tex': [], 'paper_stability.tex': []}
    if 'full' in reports:
        full = reports['full']; quality = full['quality']
        if quality['applicability'].get('per_class_f1', {}).get('Y') is not None:
            endings['paper_document.tex'].append(
                'PBPK-Match v2 ensemble applicable-class F1 is ' + number(quality['applicability']['per_class_f1']['Y']) +
                ', top-1 unknown-grade fraction is ' + number(quality['coverage']['full_catalog_top1_unknown_rate']) +
                '; Brier on known Y/N pairs is ' + number(quality['applicability']['brier']) +
                '. These measure agreement with document labels, not PK predictive validation.')
        common = next((row for row in full['execution'].get('common_positions', [])
                       if row['endpoint'] == 'Cmax_reference_vs_group'), None)
        if common and common['scenarios']:
            endings['paper_execution.tex'].append(
                f"On {common['scenarios']} common PBPK-Match v2 scenarios scorable at all three positions, "
                'position-1/2/3 GMFE values are ' + ', '.join(number(common[f'rank{i}_gmfe']) for i in (1,2,3)) +
                '. This is a descriptive shared-subset comparison, not an unadjusted superiority conclusion.')
        stability_full = full['stability']
        endings['paper_stability.tex'].append(
            'PBPK-Match v2 ensemble top-set Jaccard after sentence reordering is ' + number(stability_full['reordered']['top1_set_jaccard']) +
            ', and after evidence removal is ' + number(stability_full['missing_evidence']['top1_set_jaccard']) +
            '; original-request top-tie fraction is ' + number(stability_full['top1_tie_fraction']) + '.')
    return {key: '\n'.join(value + [table_end] + endings[key]) + '\n'
            for key, value in (('paper_document.tex', document), ('paper_execution.tex', execution), ('paper_stability.tex', stability))}


def summarize(data, output=OUT, require_complete=True):
    """Write a complete report only after every prespecified method/seed is present.

    With require_complete=False, completed methods are reported in a separate
    partial_report directory; pending work is never interpreted as poor scores.
    """
    output = _scope(output)
    external, trained, pending, refs, integrity = _load_inputs(data, output)
    if pending and require_complete:
        raise RuntimeError('Incomplete experiment: ' + ', '.join(pending))
    complete = not pending
    contract = ev.clinical_contract()
    refs.append(_ref(ev.PRIMARY))
    reports, per_seed, variability, calibration = {}, {}, {}, {}
    for method, scores in external.items():
        probabilities, calibration[method] = external_probabilities(data, scores[0])
        reports[method] = _evaluate(data, scores, probabilities, method, contract)
    for method, seeds in trained.items():
        if not seeds:
            continue
        per_seed[method] = {str(seed): _evaluate(data, value['scores'], value['probabilities'], method, contract)
                            for seed, value in seeds.items()}
        variability[method] = seed_variability(list(per_seed[method].values()))
        if set(seeds) != set(SEEDS):
            continue  # Partial seed ensembles are not silently promoted to three-seed ensembles.
        scores = np.mean([percentile_ranks(seeds[s]['scores']) for s in SEEDS], axis=0)
        probabilities = np.mean([seeds[s]['probabilities'] for s in SEEDS], axis=0)
        reports[method] = _evaluate(data, scores, probabilities, method, contract)
        if method == 'full':
            per_seed[NO_ERROR] = {str(seed): _evaluate(data, value['document_scores'], value['probabilities'], NO_ERROR, contract)
                                  for seed, value in seeds.items()}
            variability[NO_ERROR] = seed_variability(list(per_seed[NO_ERROR].values()))
            scores = np.mean([percentile_ranks(seeds[s]['document_scores']) for s in SEEDS], axis=0)
            reports[NO_ERROR] = _evaluate(data, scores, probabilities, NO_ERROR, contract)
    paired_seed = {}
    for seed in SEEDS:
        selected = {m: r for m, r in reports.items() if m in EXTERNAL}
        selected.update({m: seed_rows[str(seed)] for m, seed_rows in per_seed.items() if str(seed) in seed_rows})
        paired_seed[str(seed)] = _paired(selected, data, complete=complete)
    result = {'schema': 'pbpk-match-v2-report-1', 'completed': complete,
        'state': 'complete' if complete else 'partial_pending_prespecified_runs', 'pending': pending,
        'expected_external_methods': list(EXTERNAL), 'expected_trained_methods': list(METHODS), 'seeds': list(SEEDS),
        'query_count': len(data['query_records']), 'candidate_count': len(data['candidate_records']),
        'document_request_count': len(data['core_indices']), 'execution_request_count': len(data['execution_indices']),
        'ensemble': {'method': 'equal seed mean of within-query exact-tie midpoint percentile ranks',
                     'probabilities': 'arithmetic mean of outer-calibrated seed probabilities',
                     'no_error_branch': 'full trained network document_scores; no predicted-PK-error score correction'},
        'reports': reports, 'single_seed_reports': per_seed, 'seed_mean_sd': variability,
        'external_calibration': calibration, 'inference_integrity': integrity,
        'inference_integrity_scope': 'Two prespecified outer-held-out requests per checkpoint; full and document-only scores from the same real forwards are checked separately. New external methods check one core and one clinical request. Legacy baseline audits remain in their source receipts.',
        'paired_ensemble': _paired(reports, data, complete=complete), 'paired_by_seed': paired_seed,
        'limitations': ['Internal grouped reanalysis of already observed data, not independent prospective validation.',
            'Document labels describe documented support; numerical PK diagnostics remain a separate target.',
            'Coverage and conditional errors must be interpreted together; endpoint windows are not pooled.',
            'Uniform exact-score ties, exact positions 1/2/3, no backfill or best-of-three selection.',
            'Three training seeds quantify optimization variability, not independent scientific sample size.',
            'Unknown/unavailable outcomes and unfavorable measured predictions are retained.'], 'input_refs': refs}
    destination = output / ('report' if complete else 'partial_report')
    destination.mkdir(parents=True, exist_ok=True)
    # Recheck every artifact actually consumed before publishing any summary.
    for ref in refs:
        if sha(_scope(ROOT / ref['path'])) != ref['sha256']:
            raise ValueError('Report input changed during aggregation: ' + ref['path'])
    compact = _compact(reports)
    write(destination / 'summary.json', result)
    for name, rows in compact.items():
        (destination / (name + '.csv')).write_text(_csv(rows))
    (destination / 'tables.tex').write_text(_tex(compact['quality'], complete))
    if complete:
        for name, text in _paper_tables(reports, variability).items():
            (destination / name).write_text(text)
    (destination / 'README.md').write_text(
        '# PBPK-Match v2 experiment summary\n\n' + ('All preregistered methods and three training seeds are complete.' if complete else 'These are partial results; incomplete: ' + ', '.join(pending)) +
        '\n\nsummary.json retains per-request results, endpoints at Top 1/2/3, unknowns and omissions, risk–coverage, paired comparisons, and single-seed statistics. '
        'quality.csv, execution.csv, and stability.csv are compact ensemble or external-method results. '
        'Three-seed mean and sample standard deviation are in seed_mean_sd. Ensembles convert exact ties to percentile ranks within each request and then average equally.'
        '\n\nThis is grouped internal validation on existing data, not external validation on unseen data. Documentary support and actual prediction errors are reported separately; '
        'interpret error with coverage. Top 2/3 are not backfilled or replaced by the best of three; different AUC windows are not pooled. '
        'External-method applicability probabilities are calibrated using document labels from outer-training folds only. Missing classes and uncalibratable folds are recorded explicitly.\n')
    write(destination / 'completion.json', {'completed': complete, 'state': 'complete' if complete else 'partial_pending_prespecified_runs', 'pending': pending,
        'input_count': len(refs), 'outputs': [_ref(p) for p in sorted(destination.iterdir())
                                           if p.name in ('summary.json', 'quality.csv', 'execution.csv', 'stability.csv', 'tables.tex', 'README.md',
                                                         'paper_document.tex', 'paper_execution.tex', 'paper_stability.tex')]})
    return result
