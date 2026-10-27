"""Publication must refuse missing, changed or leaked held-out predictions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('light_reporting', ROOT / 'scripts/summarize_pbpk_match_light.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m.r, 'ROOT', tmp_path)
    out = tmp_path / 'out'; out.mkdir()
    q, c = 6, 4
    folds = []
    for held in ([0, 1], [2, 3], [4, 5]):
        train = sorted(set(range(q)) - set(held))
        folds.append({'outer_train': train, 'outer_test': list(held),
                      'inner': [{'train': train[:2], 'validation': train[2:]}]})
    data = {'query_records': [{'query_id': f'q{i}', 'compound_id': f'd{i}', 'compounds': [f'd{i}']} for i in range(q)],
        'candidate_records': [{'candidate_id': f'c{i}', 'compounds': ['d0'], 'family_id': str(i)} for i in range(c)],
        'query_texts': [[f'q{i}' for i in range(q)] for _ in range(3)],
        'groups': [f'g{i}' for i in range(q)], 'folds': folds,
        'core_indices': list(range(q)), 'execution_indices': list(range(q)), 'stability_indices': list(range(q)),
        'grades': np.tile([3, 2, 0, -1], (q, 1)), 'applicability': np.tile([0, 0, 1, 2], (q, 1))}
    write = m.r.write
    write(out / 'manifest.json', {'completed': True})
    write(out / 'cache/binding.json', {'source_manifest_sha256': m.r.sha(out / 'manifest.json'),
        'query_ids': [v['query_id'] for v in data['query_records']],
        'candidate_ids': [v['candidate_id'] for v in data['candidate_records']], 'labels_used_as_features': False})
    (out / 'cache/inputs.npz').write_bytes(b'cache')
    write(out / 'cache/manifest.json', {'completed': True,
        'array_sha256': m.r.sha(out / 'cache/inputs.npz'), 'binding_sha256': m.r.sha(out / 'cache/binding.json')})
    scores = np.broadcast_to(np.array([4., 3., 2., 1.]), (3, q, c)).copy()
    base = scores[..., ::-1].copy()
    probabilities = np.tile([.5, .3, .2], (3, q, c, 1))
    (out / 'external').mkdir()
    np.savez(out / 'external/scores.npz', scores=scores)
    write(out / 'external/complete.json', {'state': 'complete', 'score_sha256': m.r.sha(out / 'external/scores.npz')})
    protocol = {'seeds': [17], 'query_ids': [v['query_id'] for v in data['query_records']],
        'candidate_ids': [v['candidate_id'] for v in data['candidate_records']], 'folds': folds,
        'dataset_manifest': m.ref(out / 'manifest.json'), 'cache_manifest': m.ref(out / 'cache/manifest.json'),
        'implementation_sha256': {}, 'external_methods': [{'method': m.r.BASELINE,
            'scores': m.ref(out / 'external/scores.npz'), 'receipt': m.ref(out / 'external/complete.json')}]}
    write(out / 'protocol.json', protocol)
    jobs, outer = [], {}
    for f, fold in enumerate(folds):
        for kind in ('inner0', 'outer'):
            directory = out / f'runs/seed17/fold{f}/{kind}'; directory.mkdir(parents=True)
            train, validation, indices = m._binding_indices(data, f, kind)
            write(directory / 'binding.json', {'method': m.OWN, 'seed': 17,
                'protocol_sha256': m.r.sha(out / 'protocol.json'), 'dataset_sha256': m.r.sha(out / 'manifest.json'),
                'cache_manifest_sha256': m.r.sha(out / 'cache/manifest.json'), 'train_indices': train,
                'validation_indices': validation, 'prediction_indices': indices})
            name = 'predictions.npz' if kind == 'outer' else 'validation.npz'
            np.savez(directory / name, indices=indices, scores=scores[:, indices], base_scores=base[:, indices],
                     probabilities=probabilities[:, indices])
            files = ['binding.json', name]
            if kind == 'outer':
                metric = {'queries': 2, 'candidate_count': c, 'exact_equal': True,
                          'max_absolute_score_delta': 0., 'top1_tie_set_agreement': 1., 'mean_spearman': 1.}
                write(directory / 'integrity.json', {'state': 'complete', 'indices': sorted(indices)[:2], 'batch_sizes': [280, 140],
                    **{key: metric for key in ('repeat', 'candidate_order', 'batch_size', 'document_repeat',
                                               'document_candidate_order', 'document_batch_size')}})
                files.append('integrity.json')
            write(directory / 'complete.json', {'seed': 17, 'train_indices': train, 'validation_indices': validation,
                'epochs_completed': 1, 'artifacts_sha256': {name: m.r.sha(directory / name) for name in files}})
            jobs.append(m.ref(directory / 'complete.json'))
            if kind == 'outer':
                outer[str(f)] = jobs[-1]
    (out / 'predictions').mkdir()
    np.savez(out / 'predictions/seed17.npz', indices=np.arange(q), scores=scores, base_scores=base, probabilities=probabilities)
    write(out / 'predictions/seed17.json', {'state': 'complete', 'protocol_sha256': m.r.sha(out / 'protocol.json'),
        'score_sha256': m.r.sha(out / 'predictions/seed17.npz'), 'fit_receipts': jobs, 'outer_receipts': outer})
    contract = {f'q{i}': {'compound': f'd{i}', 'publication_cluster_id': f'p{i}', 'endpoints': ['Cmax_reference_vs_group'],
        'same_parent_ids': ['c0', 'c1', 'c2', 'c3'], 'errors': {'c0': {'Cmax_reference_vs_group': {'absolute': 0., 'signed': 0.}},
        'c1': {'Cmax_reference_vs_group': {'absolute': 1., 'signed': 1.}}}} for i in range(q)}
    primary = tmp_path / 'primary.json'; write(primary, contract)
    monkeypatch.setattr(m.r.ev, 'PRIMARY', primary)
    monkeypatch.setattr(m.r.ev, 'clinical_contract', lambda: contract)
    return data, out


def test_complete_single_seed_report_preserves_exact_positions(completed):
    data, out = completed
    value = m.summarize(data, out)
    assert value['completed'] and value['seeds'] == [17]
    assert value['completed_small_network_fits'] == 6
    assert not value['external_models_rerun']
    assert set(value['reports']) == {m.OWN, m.DOCUMENT, m.r.BASELINE}
    main = value['reports'][m.OWN]['execution']['position_summary']
    assert [row['rank'] for row in main] == [1, 2, 3]
    assert main[0]['gmfe'] == 1 and main[2]['coverage'] == 0
    document = value['reports'][m.DOCUMENT]['execution']['position_summary']
    assert document[0]['coverage'] == 0 and document[0]['gmfe'] is None
    assert value['reports'][m.OWN]['quality']['coverage']['unknown_grade_pairs'] == 6
    assert len(value['inference_integrity']) == 3
    assert 'seed_mean_sd' not in value
    tex = (out / 'report/paper_document.tex').read_text()
    assert '三个随机种子' not in tex and '种子为17' in tex
    completion = json.loads((out / 'report/completion.json').read_text())
    assert completion['completed'] and len(completion['outputs']) == 12


@pytest.mark.parametrize('change', ['missing_inner', 'duplicate_inner', 'cache_tamper', 'score_tamper', 'probabilities'])
def test_incomplete_or_changed_predictions_rejected(completed, change):
    data, out = completed
    receipt_path = out / 'predictions/seed17.json'
    receipt = m.read(receipt_path)
    if change == 'missing_inner':
        receipt['fit_receipts'].pop(0)
    elif change == 'duplicate_inner':
        receipt['fit_receipts'].append(receipt['fit_receipts'][0])
    elif change == 'cache_tamper':
        (out / 'cache/inputs.npz').write_bytes(b'changed')
    else:
        path = out / 'predictions/seed17.npz'
        with np.load(path) as z:
            arrays = {k: z[k].copy() for k in z.files}
        if change == 'score_tamper':
            arrays['scores'][0, 0, 0] += 1
        else:
            arrays['probabilities'][0, 0, 0] = [.2, .2, .2]
        np.savez(path, **arrays)
        receipt['score_sha256'] = m.r.sha(path)
    m.r.write(receipt_path, receipt)
    with pytest.raises(ValueError):
        m.load_inputs(data, out)


def test_group_leakage_rejected_even_with_complete_files(completed):
    data, out = completed
    data['groups'][0] = data['groups'][2]
    with pytest.raises(ValueError, match='groups overlap'):
        m.load_inputs(data, out)


def test_common_pool_does_not_compare_different_coverage():
    def row(sid, coverage, error):
        return {'scenario_id': sid, 'compound': sid, 'publication_cluster_id': sid, 'endpoint': 'Cmax',
                'rank': 1, 'coverage': coverage, 'absolute_log_error': error}
    reports = {m.OWN: {'execution': {'position_rows': [row('a', 1., 1.), row('b', 1., 100.)]}},
               'baseline': {'execution': {'position_rows': [row('a', 1., 2.), row('b', .5, 1.)]}}}
    value = m.common_coverage_comparisons(reports)[0]
    assert value['scenario_ids'] == ['a']
    assert value['own_gmfe'] == pytest.approx(np.e)
    assert value['comparator_gmfe'] == pytest.approx(np.e ** 2)


def test_omissions_do_not_overweight_repeated_doses_or_studies():
    def row(drug, study, miss):
        return {'endpoint': 'Cmax', 'compound': drug, 'publication_cluster_id': study,
            'by_k': {str(k): {'best_measured_miss_probability': miss} for k in (3, 5)}}
    rows = [row('a', 's1', 1.), row('a', 's1', 1.), row('a', 's2', 0.),
            row('b', 's3', 0.), row('c', 's4', None)]
    values = m.omission_summary(rows)
    for value in values:
        assert value['total_scenarios'] == 5
        assert value['choice_comparison_scenarios'] == 4
        assert value['compounds'] == 2 and value['compound_study_pairs'] == 3
        # Drug a averages studies (1+0)/2=.5; drug b has 0. The final mean
        # equals .25, not the scenario-macro estimate (1+1+0+0)/4=.5.
        assert value['best_measured_miss_probability'] == .25
        assert value['best_measured_hit_probability'] == .75


def test_quality_duplicate_results_and_iteration_order_share_random_draws():
    def report(offset):
        return {'per_query': [{'query_id': f'q{i:02d}', 'judged_only': {
            key: .5 + offset * (i / 19 - .3) for key in ('ndcg@10', 'recall@10', 'mrr')}} for i in range(20)]}
    reports = {'base': report(0.), 'own': report(.2), 'document': report(.2), 'other': report(.1)}
    groups = {f'q{i:02d}': f'g{i // 2:02d}' for i in range(20)}
    first = m.stable_paired_quality(reports, groups, baseline='base')
    reversed_reports = {name: {'per_query': list(reversed(value['per_query']))}
                        for name, value in reversed(list(reports.items()))}
    second = m.stable_paired_quality(reversed_reports, groups, baseline='base')
    assert first == second
    for metric in ('ndcg@10', 'recall@10', 'mrr'):
        own = next(row for row in first if row['method'] == 'own' and row['metric'] == metric)
        doc = next(row for row in first if row['method'] == 'document' and row['metric'] == metric)
        assert {k: v for k, v in own.items() if k != 'method'} == {k: v for k, v in doc.items() if k != 'method'}
        assert own['bootstrap_draws'] == 2000


def test_execution_duplicate_results_and_iteration_order_share_random_draws():
    methods = ['base', 'own', 'document', 'other']
    rows = []
    for method in methods:
        for i in range(18):
            delta = (i / 17 - .4) * (0 if method == 'base' else .1 if method == 'other' else .2)
            rows.append({'method': method, 'stratum': 'reference_diagnostic', 'endpoint': 'Cmax',
                'scenario_id': f'q{i:02d}', 'compound': f'c{i:02d}', 'publication_cluster_id': f'p{i:02d}',
                'top1_evaluable_probability': 1., 'top1_absolute_log_error': 1. + delta})
    first = m.stable_paired_execution(rows, methods, baseline='base', rank=1)
    second = m.stable_paired_execution(list(reversed(rows)), list(reversed(methods)), baseline='base', rank=1)
    assert first == second
    own = next(row for row in first['comparisons'] if row['method'] == 'own')
    doc = next(row for row in first['comparisons'] if row['method'] == 'document')
    assert {k: v for k, v in own.items() if k != 'method'} == {k: v for k, v in doc.items() if k != 'method'}
    assert own['bootstrap_draws'] == 2000
    assert own['holm_prespecified_family_size'] == 3
    assert own['permutation'] == 'monte_carlo_block_sign_flip_plus_one'
