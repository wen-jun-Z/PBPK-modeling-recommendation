"""Statistical reporting boundary checks without model inference or sealed data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match_v2 import reporting as r


def tiny_data():
    q, c = 6, 4
    folds = []
    for ids in ([0, 1], [2, 3], [4, 5]):
        folds.append({'outer_train': [i for i in range(q) if i not in ids], 'outer_test': list(ids), 'inner': []})
    return {'query_records': [{'query_id': f'q{i}', 'compound_id': f'd{i}', 'compounds': [f'd{i}']} for i in range(q)],
        'candidate_records': [{'candidate_id': f'c{i}', 'compounds': ['d0'], 'family_id': str(i)} for i in range(c)],
        'query_texts': [[f'q{i}' for i in range(q)] for _ in range(3)],
        'groups': [f'g{i}' for i in range(q)], 'folds': folds,
        'core_indices': list(range(q)), 'execution_indices': list(range(q)), 'stability_indices': list(range(q)),
        'grades': np.tile([3, 2, 0, -1], (q, 1)), 'applicability': np.tile([0, 0, 1, 2], (q, 1))}


def test_percentiles_ties_order_scale_and_singleton():
    a = np.array([[1000, -7, 1000, 0], [1, 1, 1, 1]])
    expected = [[5/6, 0, 5/6, 1/3], [.5]*4]
    np.testing.assert_allclose(r.percentile_ranks(a), expected)
    np.testing.assert_equal(r.percentile_ranks(a), r.percentile_ranks(10*a+700))
    np.testing.assert_equal(r.percentile_ranks([[100]]), [[.5]])
    with pytest.raises(ValueError): r.percentile_ranks([[np.nan]])


def test_crossfit_no_test_label_or_scale_leakage():
    data = tiny_data(); x = np.arange(24, dtype=float).reshape(6, 4)
    p, audit = r.external_probabilities(data, x)
    altered = deepcopy(data); altered['applicability'][:2] = [2, 1, 0, 0]
    p2, audit2 = r.external_probabilities(altered, x)
    np.testing.assert_equal(p[:2], p2[:2])
    assert audit['folds'][0] == audit2['folds'][0]
    changed = x.copy(); changed[:2] += 1e5
    _, a3 = r.external_probabilities(data, changed)
    assert audit['folds'][0]['train_mean'] == a3['folds'][0]['train_mean']
    assert audit['folds'][0]['train_std'] == a3['folds'][0]['train_std']
    assert np.isfinite(p).all(); np.testing.assert_allclose(p.sum(-1), 1)


def test_calibration_one_class_unavailable_and_two_classes_explicit():
    data = tiny_data(); data['applicability'][:] = 0
    p, audit = r.external_probabilities(data, np.arange(24.).reshape(6, 4))
    assert np.isnan(p).all() and all(f['state']=='unavailable' for f in audit['folds'])
    data['applicability'][:, 2:] = 1
    p, audit = r.external_probabilities(data, np.arange(24.).reshape(6, 4))
    assert all(f['missing_classes'] == [2] for f in audit['folds'])
    assert np.all(p[..., 2] == 0)


@pytest.mark.parametrize('kind', ['overlap', 'missing', 'group'])
def test_calibration_rejects_fold_contamination(kind):
    data = tiny_data()
    if kind == 'overlap': data['folds'][0]['outer_train'].append(0)
    if kind == 'missing': data['folds'].pop()
    if kind == 'group': data['groups'][0] = data['groups'][2]
    with pytest.raises(ValueError): r.external_probabilities(data, np.arange(24.).reshape(6, 4))


def test_risk_unknowns_not_negative_and_ties_not_backfilled():
    data = tiny_data()
    scores = np.tile([4, 3, 2, 4], (6, 1))
    p = np.tile([.8, .1, .1], (6, 4, 1))
    result = r.risk_coverage(data, scores, p)
    assert result['requests'][0]['unknown_label_probability'] == .5
    assert result['requests'][0]['conditional_error'] == 0
    assert result['points'][0]['recommendation_coverage'] == 1
    assert result['points'][-1]['recommendation_coverage'] == 0


def test_seed_sd_is_sample_sd_and_missing_not_zero():
    make = lambda v: {'quality': {'x': v}, 'execution': {'position_summary': []}, 'stability': {}}
    a = r.seed_variability([make(1), make(2), make(3)])['quality.x']
    assert a == {'mean': 2., 'sd': 1., 'available_seeds': 3, 'total_seeds': 3}
    b = r.seed_variability([make(1), make(None), make(3)])['quality.x']
    assert b['mean'] == 2 and b['available_seeds'] == 2


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'EXTERNAL', ('f2llm_v2_06b',))
    monkeypatch.setattr(r, 'METHODS', ('full',))
    monkeypatch.setattr(r, 'SEEDS', (17, 23, 41))
    data = tiny_data(); out = tmp_path / 'out'; out.mkdir()
    manifest = out / 'manifest.json'; manifest.write_text('{}')
    protocol = {'methods': list(r.METHODS), 'external_methods': list(r.EXTERNAL), 'seeds': list(r.SEEDS),
        'query_ids': [v['query_id'] for v in data['query_records']],
        'candidate_ids': [v['candidate_id'] for v in data['candidate_records']], 'folds': data['folds'],
        'dataset_manifest_sha256': r.sha(manifest), 'implementation_sha256': {}}
    r.write(out/'protocol.json', protocol)
    scores = np.broadcast_to(np.array([4, 3, 2, 1.]), (3, 6, 4)).copy()
    p = np.tile([.5, .25, .25], (6, 4, 1))
    folder = out / 'external/f2llm_v2_06b'; folder.mkdir(parents=True)
    np.savez(folder/'scores.npz', scores=scores)
    r.write(folder/'complete.json', {'state': 'complete', 'score_sha256': r.sha(folder/'scores.npz')})
    (out/'predictions').mkdir()
    for seed in r.SEEDS:
        # Variable score scales ensure raw arithmetic is not used for ensemble.
        s = scores * seed; doc = s[..., ::-1].copy()
        np.savez(out/'predictions'/f'full_seed{seed}.npz', scores=s, document_scores=doc, probabilities=p)
        for fold_id, fold in enumerate(data['folds']):
            d = out/'runs/full'/f'seed{seed}'/f'fold{fold_id}'/'outer'; d.mkdir(parents=True)
            ix = fold['outer_test']; values = {'indices': ix, 'probabilities': p[ix]}
            for variant in range(3):
                values[f'scores_{variant}'] = s[variant, ix]
                values[f'document_scores_{variant}'] = doc[variant, ix]
            np.savez(d/'predictions.npz', **values)
            r.write(d/'binding.json', {'method': 'full', 'seed': seed,
                'protocol_sha256': r.sha(out/'protocol.json'), 'dataset_sha256': r.sha(out/'manifest.json'),
                'prediction_indices': ix, 'validation_indices': [], 'train_indices': fold['outer_train']})
            metric={'queries': len(ix), 'candidate_count': 4, 'exact_equal': True,
                    'max_absolute_score_delta': 0., 'top1_tie_set_agreement': 1., 'mean_spearman': 1.}
            r.write(d/'integrity.json', {'state': 'complete', 'indices': ix, 'batch_sizes': [32,16],
                                       **{k: metric for k in ('repeat','candidate_order','batch_size',
                                           'document_repeat','document_candidate_order','document_batch_size')}})
            r.write(d/'complete.json', {'artifacts_sha256': {name: r.sha(d/name) for name in ('predictions.npz','binding.json','integrity.json')}})
        r.write(out/'predictions'/f'full_seed{seed}.json', {'state': 'complete',
            'protocol_sha256': r.sha(out/'protocol.json'), 'score_sha256': r.sha(out/'predictions'/f'full_seed{seed}.npz'),
            'outer_receipts': {str(i): r.sha(out/'runs/full'/f'seed{seed}'/f'fold{i}'/'outer/complete.json') for i in range(len(data['folds']))}})
    contract = {f'q{i}': {'compound': f'd{i}', 'publication_cluster_id': f'p{i}', 'endpoints': ['Cmax'],
        'same_parent_ids': ['c0','c1','c2','c3'], 'errors': {'c0': {'Cmax': {'absolute': 0., 'signed': 0.}},
        'c1': {'Cmax': {'absolute': 1., 'signed': 1.}}}} for i in range(6)}
    primary=tmp_path/'primary.json'; r.write(primary, contract)
    monkeypatch.setattr(r.ev, 'PRIMARY', primary)
    monkeypatch.setattr(r.ev, 'clinical_contract', lambda: contract)
    return data, out


def test_complete_report_all_sections_seeds_noerror_and_top3(files):
    data, out = files
    result = r.summarize(data, out)
    assert result['completed'] and len(result['single_seed_reports']['full']) == 3
    assert set(result['reports']) == {'f2llm_v2_06b','full','no_error_branch'}
    assert {row['rank'] for row in result['reports']['full']['execution']['position_rows']} == {1,2,3}
    assert result['reports']['full']['execution']['position_summary'][0]['coverage'] == 1
    assert result['reports']['no_error_branch']['execution']['position_summary'][0]['coverage'] == 0
    assert (out/'report/tables.tex').exists()
    completion=json.loads((out/'report/completion.json').read_text())
    assert completion['completed'] and len(completion['outputs']) == 9
    assert len(result['inference_integrity']['full']) == 9
    assert result['paired_ensemble']['baseline'] == 'f2llm_v2_06b'
    assert [x['rank'] for x in result['paired_ensemble']['execution']] == [1,2,3]
    assert all(x['holm_prespecified_family_size']==2 for x in result['paired_ensemble']['execution'][0]['comparisons'])


def test_missing_run_is_pending_not_zero_or_short_seed_ensemble(files):
    data,out=files; (out/'predictions/full_seed41.npz').unlink()
    with pytest.raises(RuntimeError): r.summarize(data,out)
    result=r.summarize(data,out,require_complete=False)
    assert not result['completed'] and 'full/seed41' in result['pending']
    assert 'full' not in result['reports'] and len(result['single_seed_reports']['full'])==2
    assert (out/'partial_report/summary.json').exists() and not (out/'report').exists()


@pytest.mark.parametrize('kind', ['sha', 'shape', 'probabilities', 'outer', 'protocol', 'code'])
def test_completed_artifact_tampering_is_rejected(files, kind):
    data,out=files
    if kind == 'sha': (out/'external/f2llm_v2_06b/scores.npz').write_bytes(b'changed')
    elif kind in ('shape','probabilities'):
        path=out/'predictions/full_seed17.npz'
        with np.load(path) as saved: values={k:saved[k] for k in saved.files}
        if kind=='shape': values['scores']=values['scores'][:1]
        else: values['probabilities'][0,0]=[1,1,1]
        np.savez(path,**values)
    elif kind=='outer': (out/'runs/full/seed17/fold0/outer/complete.json').unlink()
    elif kind=='protocol':
        p=out/'protocol.json';v=json.loads(p.read_text());v['candidate_ids'][0]='wrong';r.write(p,v)
    elif kind=='code':
        p=out/'protocol.json';v=json.loads(p.read_text());v['implementation_sha256']={'bogus.py':'0'*64};r.write(p,v)
    with pytest.raises((ValueError,FileNotFoundError)): r.summarize(data,out)


def test_final_assembled_outer_values_must_match(files):
    data,out=files; p=out/'predictions/full_seed17.npz'
    with np.load(p) as saved: values={k:saved[k] for k in saved.files}
    values['scores'][0,0,0]+=1
    np.savez(p,**values)
    receipt=p.with_suffix('.json'); value=json.loads(receipt.read_text()); value['score_sha256']=r.sha(p);r.write(receipt,value)
    with pytest.raises(ValueError,match='Assembled'): r.summarize(data,out)


def test_missing_seed_receipt_is_pending(files):
    data,out=files; (out/'predictions/full_seed41.json').unlink()
    with pytest.raises(RuntimeError,match='Incomplete'): r.summarize(data,out)


def test_no_paper_outputs_for_pending(files):
    data,out=files; (out/'predictions/full_seed41.json').unlink()
    r.summarize(data,out,require_complete=False)
    assert not list((out/'partial_report').glob('paper_*.tex'))


def test_integrity_numeric_instability_reported_not_filtered(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    path=tmp_path/'integrity.json'
    metric={'queries':2,'candidate_count':4,'exact_equal':False,'max_absolute_score_delta':.02,
            'top1_tie_set_agreement':.5,'mean_spearman':.7}
    r.write(path,{'state':'complete','indices':[0,1],'batch_sizes':[32,16],
                  **{k:metric for k in ('repeat','candidate_order','batch_size')}})
    actual=r._integrity(path,[0,1],4,[],[],'x',r.sha(path))
    assert actual['repeat']['top1_tie_set_agreement']==.5
    value=json.loads(path.read_text());value['repeat']['exact_equal']=True;r.write(path,value)
    with pytest.raises(ValueError,match='Contradictory'): r._integrity(path,[0,1],4,[],[],'x',r.sha(path))
