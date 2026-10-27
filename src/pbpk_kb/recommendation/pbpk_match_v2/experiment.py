"""Isolated protocol, checkpoints and nested-CV experiment orchestration."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'experiments/model_recommendation/pbpk_match_v2'
RUNTIME = ROOT / 'runtime/deployment/pbpk_match_v2'
OLD = ROOT / 'experiments/model_recommendation/frontier_comparison_20260914'
CLINICAL = ROOT / 'experiments/model_recommendation/execution_validation_20261001/rankings'
CORE = ROOT / 'experiments/model_recommendation/frontier_core_145_20260921'
METHODS = ('full', 'no_numeric', 'frozen_text', 'mlp', 'lambdamart')
SEEDS = (17, 23, 41)
EXTERNAL = ('bm25_text', 'qwen3_embedding_06b', 'f2llm_v2_06b', 'qwen3_reranker_06b',
            'jina_reranker_v35', 'bm25_top20_rearank_7b_nf4', 'bm25_top20_reasonrank_7b_nf4',
            'prorank', 'contextual')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False, default=clean) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def array_write(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def status(phase, **values):
    previous = read(RUNTIME / 'status.json') if (RUNTIME / 'status.json').exists() else {}
    if phase == 'validating':
        previous = {k: v for k, v in previous.items() if k not in ('error', 'error_type', 'reason', 'finished_at', 'training_progress')}
        values['attempt_started_at'] = datetime.now(timezone.utc).isoformat()
    payload = {**previous, **values, 'phase': phase, 'pid': os.getpid(),
               'updated_at': datetime.now(timezone.utc).isoformat()}
    payload.setdefault('started_at', payload['updated_at'])
    write(RUNTIME / 'status.json', payload)
    print(json.dumps({'phase': phase, **values}, ensure_ascii=False, default=clean), flush=True)
    if (RUNTIME / 'stop.requested').exists():
        raise InterruptedError('Project experiment stop requested')


def freeze_protocol(data):
    code = [ROOT / 'scripts/run_pbpk_match_v2.py', *sorted(Path(__file__).parent.glob('*.py'))]
    code += [ROOT / 'scripts/frontier_external_models.py', ROOT / 'scripts/recommendation_execution_statistics.py',
             ROOT / 'src/pbpk_kb/recommendation/pbpk_match/evaluation.py']
    value = {'schema': 'pbpk-match-v2-internal-grouped-1', 'seeds': list(SEEDS),
        'dataset_manifest_sha256': sha(OUT / 'manifest.json'),
        'methods': list(METHODS), 'external_methods': list(EXTERNAL), 'outer_folds': 5, 'inner_folds': 3,
        'maximum_epochs': 10, 'patience': 3, 'learning_rate': 1e-4, 'lora_rank': 16,
        'lora_layers': 8, 'full_query_gradient_cache': True,
        'max_length': 768, 'error_weights': [0., .1, .3, 1.],
        'ndcg_tolerance': .02, 'coverage_decline_allowed': 0.,
        'minimum_error_training_studies': 3,
        'training_scope': 'Internal grouped validation using already published outcomes; no independent validation claim',
        'selection_endpoint': 'Cmax_reference_vs_group',
        'selection_rule': 'Inner OOF: document NDCG decrease<=.02 and clinical top1 coverage nondecrease; compare every feasible alpha on the SAME intersection of full-coverage Cmax cases. No common evidence: alpha0.',
        'checkpoint_rule': 'Inner best epochs median; outer refit fixed epochs; no outer early stopping',
        'calibration_rule': 'Temperature fitted on inner OOF probabilities; outer labels never used',
        'clinical_scoring': 'Exact positions1/2/3, uniform ties, no backfill; same reference diagnostic estimand as original paper',
        'new_independent_data': False, 'new_llm_judgments': False,
        'runtime_versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'transformers', 'peft', 'numpy', 'scipy', 'scikit-learn', 'xgboost')},
        'query_ids': [q['query_id'] for q in data['query_records']],
        'candidate_ids': [c['candidate_id'] for c in data['candidate_records']],
        'groups': list(data['groups']), 'folds': data['folds'],
        'legacy_artifact_sha256': legacy_sources(),
        'implementation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in code}}
    from .model import DEFAULT_BACKBONE, BACKBONE_REVISION
    from .external import WEIGHTS, MODEL_SPECS
    value['qwen_backbone_revision'] = BACKBONE_REVISION
    value['qwen_files_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(DEFAULT_BACKBONE.iterdir()) if p.is_file()}
    value['external_weight_receipts_sha256'] = {name: sha(WEIGHTS / name / spec['revision'] / 'model_receipt.json')
                                               for name, spec in MODEL_SPECS.items()}
    path = OUT / 'protocol.json'
    if path.exists() and read(path) != value:
        raise ValueError('Frozen v2 protocol/code differs; do not mix implementations within a run')
    if not path.exists():
        write(path, value)
    return value


def legacy_sources():
    files = {OLD / 'results.json', CORE / 'assessment_repaired_v2/automatic_evaluation.json',
             CLINICAL / 'ranking_lock.json'}
    legacy = read(OLD / 'results.json')['methods']
    core = read(CORE / 'assessment_repaired_v2/automatic_evaluation.json')['methods']
    clinical = read(CLINICAL / 'ranking_lock.json')['methods']
    for method in EXTERNAL:
        if method in ('prorank', 'contextual'):
            continue
        for path in (OLD / 'results' / method / legacy[method]['effective_score_file'],
                     CORE / core[method]['progress']['effective_score_file'], ROOT / clinical[method]['path']):
            files.update([path, path.parent / 'index.json'])
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def prediction_arrays(prediction, indices):
    values = {'indices': np.asarray(indices), 'scores': np.asarray(prediction['scores'], np.float32),
              'probabilities': np.asarray(prediction['probabilities'], np.float64)}
    for key in ['base_scores', 'risk_scores', 'predicted_error', 'selected_predicted_error', 'error_supported']:
        if key in prediction:
            values[key] = np.asarray(prediction[key])
    # Risk remains outcome-free; selecting the requested endpoint is handled by the model.
    return values


def correction(pred, weight):
    if weight not in (0., .1, .3, 1.):
        raise ValueError('Error correction weight is outside the frozen grid')
    base = np.asarray(pred.get('base_scores', pred['scores']), float)
    if base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError('Invalid document score matrix')
    risk = pred.get('risk_scores', pred.get('predicted_error', pred.get('selected_predicted_error')))
    if risk is None:
        if weight:
            raise ValueError('Predictor did not expose request-conditioned risk scores')
        return base
    risk = np.asarray(risk, float)
    if risk.shape != base.shape or not np.isfinite(risk).all() or (risk < 0).any():
        raise ValueError('Invalid predicted error matrix')
    if weight and 'error_supported' not in pred:
        raise ValueError('Nonzero correction requires an explicit endpoint support mask')
    supported = np.asarray(pred.get('error_supported', np.zeros(len(base), bool)))
    if supported.dtype != bool or supported.shape not in ((len(base),), base.shape):
        raise ValueError('Invalid endpoint support mask')
    if supported.ndim == 1:
        supported = supported[:, None]
    return base - weight * risk * supported


_LAST_PROGRESS = 0.


def training_progress(event):
    global _LAST_PROGRESS
    now = time.monotonic()
    if now - _LAST_PROGRESS >= 10 or 'epoch_complete' in event.get('stage', ''):
        _LAST_PROGRESS = now
        status('training_progress', training_progress=event)


def select_error_weight(data, indices, prediction):
    from .evaluation import quality_evaluation, clinical_evaluation
    if len(set(indices)) != len(indices):
        raise ValueError('Selection indices must be unique inner OOF rows')
    core = sorted(set(indices) & set(data['core_indices']))
    clinical = sorted(set(indices) & set(data['execution_indices']))
    positions = {int(q): i for i, q in enumerate(indices)}
    fullshape = (len(data['query_records']), len(data['candidate_records']))
    base = np.zeros(fullshape)
    base[indices] = correction(prediction, 0)
    ndcg = quality_evaluation(data, base, indices=core)['judged_only']['metrics']['ndcg@10'] if core else None
    if not clinical or ndcg is None:
        return {'weight': 0., 'reason': 'No eligible inner document or clinical selection pool', 'trials': []}
    def evaluate(s):
        rows = clinical_evaluation(s[[positions[i] for i in clinical]],
            [data['query_records'][i]['query_id'] for i in clinical],
            [c['candidate_id'] for c in data['candidate_records']], method='selection')['position_rows']
        return {r['scenario_id']: r for r in rows if r['rank'] == 1 and r['endpoint'] == 'Cmax_reference_vs_group'}
    b = evaluate(correction(prediction, 0))
    if not b:
        return {'weight': 0., 'reason': 'No inner Cmax requests', 'trials': []}
    trials, rows_by_weight = [], {}
    for weight in (0., .1, .3, 1.):
        scores = correction(prediction, weight)
        allscores = base.copy(); allscores[indices] = scores
        doc = quality_evaluation(data, allscores, indices=core)['judged_only']['metrics']['ndcg@10']
        r = evaluate(scores)
        coverage = float(np.mean([x['coverage'] for x in r.values()]))
        coverage_base = float(np.mean([x['coverage'] for x in b.values()]))
        rows_by_weight[weight] = r
        feasible = doc is not None and doc >= ndcg - .02 and coverage >= coverage_base - 1e-12
        trials.append({'weight': weight, 'ndcg': doc, 'baseline_ndcg': ndcg, 'coverage': coverage,
            'baseline_coverage': coverage_base, 'feasible': bool(feasible)})
    # Do not let an alpha win by exchanging hard evaluable cases for unknown ones.
    eligible = [r for r in trials if r['feasible']]
    shared = set(b)
    for trial in eligible:
        shared &= {sid for sid, row in rows_by_weight[trial['weight']].items()
                   if row['coverage'] >= 1 - 1e-10}
    from .evaluation import hierarchical_mean
    for trial in trials:
        r = rows_by_weight[trial['weight']]
        difference = hierarchical_mean([{**r[sid], 'difference': r[sid]['absolute_log_error'] - b[sid]['absolute_log_error']}
             for sid in sorted(shared)], 'difference') if trial['feasible'] and shared else None
        trial.update(common_cases=len(shared), common_scenarios=sorted(shared),
                     common_log_error_difference=difference, feasible=trial['feasible'] and difference is not None)
    choices = [r for r in trials if r['feasible']]
    chosen = min(choices, key=lambda r: (r['common_log_error_difference'], r['weight'])) if choices else None
    return {'weight': chosen['weight'] if chosen else 0., 'reason': 'Inner OOF selection only' if chosen else 'No feasible inner evidence', 'trials': trials}


def job_binding(data, method, seed, train, validation, prediction, **configuration):
    return {'dataset_sha256': sha(OUT / 'manifest.json'), 'protocol_sha256': sha(OUT / 'protocol.json'),
            'method': method, 'seed': seed, 'train_indices': train, 'validation_indices': validation,
            'prediction_indices': prediction, 'configuration': configuration}


def verify_job(job, binding, artifact):
    job = Path(job)
    if (job / 'binding.json').exists() and read(job / 'binding.json') != binding:
        raise ValueError('Training job inputs or protocol changed: ' + str(job))
    if not (job / 'binding.json').exists():
        if (job / 'complete.json').exists():
            raise ValueError('Completed training job lacks its frozen binding')
        write(job / 'binding.json', binding)
    if not (job / 'complete.json').exists():
        return None
    meta = read(job / 'complete.json')
    if not {'binding.json', artifact} <= set(meta['artifacts_sha256']):
        raise ValueError('Completed training receipt omits required artifacts')
    for rel, digest in meta['artifacts_sha256'].items():
        p = (job / rel).resolve()
        if not p.is_relative_to(job.resolve()) or sha(p) != digest:
            raise ValueError('Training artifact changed: ' + str(p))
    with np.load(job / artifact, allow_pickle=False) as saved:
        if saved['indices'].tolist() != binding['prediction_indices']:
            raise ValueError('OOF prediction indices do not match the frozen job')
        if not all(np.isfinite(saved[key]).all() for key in saved.files):
            raise ValueError('Nonfinite OOF artifact')
    return meta


def complete_job(job, meta, artifact):
    job = Path(job)
    checkpoint = Path(meta['checkpoint']).resolve()
    if not checkpoint.is_relative_to(job.resolve()):
        raise ValueError('Checkpoint escaped its project job directory')
    files = [job / artifact, job / 'binding.json', *sorted(p for p in checkpoint.rglob('*') if p.is_file())]
    if (job / 'integrity.json').exists():
        files.append(job / 'integrity.json')
    write(job / 'complete.json', {**meta, 'artifacts_sha256': {str(p.relative_to(job)): sha(p) for p in files}})


def fit_job(train_fold, data, train, validation, job, **configuration):
    """A finished fit survives interruption during the later prediction phase."""
    receipt = Path(job) / 'trained.json'
    if receipt.exists():
        saved = read(receipt)
        for rel, digest in saved['artifacts_sha256'].items():
            path = (Path(job) / rel).resolve()
            if not path.is_relative_to(Path(job).resolve()) or sha(path) != digest:
                raise ValueError('Completed fit changed before prediction resumed')
        return saved['metadata']
    meta = train_fold(data, train, validation, job, **configuration)
    files = [Path(job) / 'binding.json', *sorted(p for p in Path(meta['checkpoint']).rglob('*') if p.is_file())]
    write(receipt, {'metadata': meta, 'artifacts_sha256': {str(p.relative_to(job)): sha(p) for p in files}})
    return meta


def train_experiment(data, *, device='cuda:0', batch_size=32):
    from .training import train_fold, predict_fold
    from .evaluation import fit_oof_temperature, calibrate
    qcount, ccount = len(data['query_records']), len(data['candidate_records'])
    supervised = set(data['core_indices']) | set(data['execution_indices'])
    total = len(METHODS) * len(SEEDS) * len(data['folds']) * 4
    done = sum(1 for p in (OUT / 'runs').glob('*/*/*/*/complete.json'))
    for method in METHODS:
        for seed in SEEDS:
            for fold_id, fold in enumerate(data['folds']):
                path = OUT / 'runs' / method / f'seed{seed}' / f'fold{fold_id}'
                pooled_indices, pooled, epochs = [], defaultdict(list), []
                for inner_id, split in enumerate(fold['inner']):
                    job = path / f'inner{inner_id}'
                    train = sorted(supervised & set(split['train']))
                    validation = sorted(supervised & set(split['validation']))
                    pairs = int(((np.asarray(data['grades'])[train] >= 0) | (np.asarray(data['applicability'])[train] >= 0)).sum())
                    status('training_inner', method=method, seed=seed, fold=fold_id, inner=inner_id,
                           training_pairs_per_epoch=pairs, maximum_epochs=10, completed_jobs=done, total_jobs=total)
                    binding = job_binding(data, method, seed, train, validation, validation,
                                          maximum_epochs=10, batch_size=batch_size, max_length=768)
                    meta = verify_job(job, binding, 'validation.npz')
                    if meta is None:
                        meta = fit_job(train_fold, data, train, validation, job, method=method, seed=seed,
                            device=device, epochs=10, batch_size=batch_size, max_length=768, learning_rate=1e-4,
                            callback=training_progress)
                        pred = predict_fold(data, validation, meta['checkpoint'], device=device, batch_size=batch_size, callback=training_progress)
                        array_write(job / 'validation.npz', **prediction_arrays(pred, validation))
                        complete_job(job, meta, 'validation.npz')
                        done += 1
                    meta = read(job / 'complete.json')
                    epochs.append(int(meta.get('best_epoch', meta.get('epochs_trained', 10))))
                    with np.load(job / 'validation.npz', allow_pickle=False) as saved:
                        pooled_indices.extend(saved['indices'].tolist())
                        for key in saved.files:
                            if key != 'indices':
                                pooled[key].append(saved[key])
                if len(set(pooled_indices)) != len(pooled_indices):
                    raise ValueError('Inner OOF rows repeated')
                inner_prediction = {k: np.concatenate(v, axis=0) for k, v in pooled.items()}
                selection = select_error_weight(data, pooled_indices, inner_prediction)
                calibration = fit_oof_temperature(inner_prediction['probabilities'], np.asarray(data['applicability'])[pooled_indices],
                    [data['groups'][i] for i in pooled_indices])
                refit_epochs = max(1, int(np.median(epochs)))
                write(path / 'selection.json', {'error_weight': selection, 'calibration': calibration,
                    'refit_epochs': refit_epochs, 'selection_indices': pooled_indices,
                    'outer_indices_not_used': fold['outer_test']})
                status('training_outer_refit', method=method, seed=seed, fold=fold_id,
                    refit_epochs=refit_epochs, maximum_epochs=refit_epochs,
                    training_pairs_per_epoch=int(((np.asarray(data['grades'])[sorted(supervised & set(fold['outer_train']))] >= 0) |
                         (np.asarray(data['applicability'])[sorted(supervised & set(fold['outer_train']))] >= 0)).sum()),
                    selected_error_weight=selection['weight'], completed_jobs=done, total_jobs=total)
                job = path / 'outer'
                train = sorted(supervised & set(fold['outer_train']))
                test = list(fold['outer_test'])
                binding = job_binding(data, method, seed, train, [], test, refit_epochs=refit_epochs,
                                      batch_size=batch_size, max_length=768,
                                      selection_sha256=sha(path / 'selection.json'))
                if verify_job(job, binding, 'predictions.npz') is not None:
                    continue
                meta = fit_job(train_fold, data, train, [], job, method=method, seed=seed, device=device,
                    epochs=refit_epochs, batch_size=batch_size, max_length=768, learning_rate=1e-4,
                    callback=training_progress, fixed_epochs=True)
                arrays = {'indices': np.asarray(test)}
                for variant_id, variant in enumerate(data['variant_names']):
                    status('predicting_outer', method=method, seed=seed, fold=fold_id, variant=variant,
                        completed_jobs=done, total_jobs=total)
                    pred = predict_fold(data, test, meta['checkpoint'], variant=variant, device=device, batch_size=batch_size, callback=training_progress)
                    arrays[f'scores_{variant_id}'] = correction(pred, selection['weight'])
                    arrays[f'document_scores_{variant_id}'] = correction(pred, 0)
                    if variant_id == 0:
                        arrays['probabilities'] = calibrate(pred['probabilities'], calibration['temperature'])
                        arrays['selected_predicted_error'] = pred.get('selected_predicted_error', np.zeros_like(pred['scores']))
                        arrays['error_supported'] = pred.get('error_supported', np.zeros(len(test), bool))
                        for endpoint, errors in pred.get('predicted_errors', {}).items():
                            arrays[f'predicted_error__{endpoint}'] = errors
                array_write(job / 'predictions.npz', **arrays)
                audit = inference_integrity(data, test, meta['checkpoint'], device=device,
                                            batch_size=batch_size, weight=selection['weight'])
                write(job / 'integrity.json', audit)
                complete_job(job, meta, 'predictions.npz')
                done += 1
            # A method/seed is reportable only when all five outer folds exist.
            arrays = {'scores': np.full((3, qcount, ccount), np.nan),
                'document_scores': np.full((3, qcount, ccount), np.nan),
                'probabilities': np.full((qcount, ccount, 3), np.nan)}
            covered = set()
            for fold_id in range(len(data['folds'])):
                p = OUT / 'runs' / method / f'seed{seed}' / f'fold{fold_id}' / 'outer/predictions.npz'
                with np.load(p, allow_pickle=False) as saved:
                    indices = saved['indices']
                    if covered & set(indices.tolist()):
                        raise ValueError('Outer groups repeated')
                    covered.update(indices.tolist())
                    for i in range(3):
                        arrays['scores'][i, indices] = saved[f'scores_{i}']
                        arrays['document_scores'][i, indices] = saved[f'document_scores_{i}']
                    arrays['probabilities'][indices] = saved['probabilities']
            if covered != set(range(qcount)) or not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Outer prediction coverage incomplete')
            array_write(OUT / 'predictions' / f'{method}_seed{seed}.npz', **arrays)
            write(OUT / 'predictions' / f'{method}_seed{seed}.json', {
                  'state': 'complete', 'protocol_sha256': sha(OUT / 'protocol.json'),
                  'score_sha256': sha(OUT / 'predictions' / f'{method}_seed{seed}.npz'),
                  'outer_receipts': {str(i): sha(OUT / 'runs' / method / f'seed{seed}' / f'fold{i}' / 'outer/complete.json')
                                      for i in range(len(data['folds']))}})
            status('seed_complete', method=method, seed=seed, completed_jobs=done, total_jobs=total)


def compare_inference(left, right):
    from scipy.stats import rankdata
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid consistency comparison')
    agreement, correlation = [], []
    for x, y in zip(a, b):
        agreement.append(np.array_equal(x == x.max(), y == y.max()))
        rx, ry = rankdata(x), rankdata(y)
        correlation.append(float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else
                           (1. if np.array_equal(rx, ry) else 0.))
    return {'queries': len(a), 'candidate_count': a.shape[1], 'exact_equal': bool(np.array_equal(a, b)),
            'max_absolute_score_delta': float(np.max(np.abs(a - b))),
            'top1_tie_set_agreement': float(np.mean(agreement)), 'mean_spearman': float(np.mean(correlation))}


def inference_integrity(data, indices, checkpoint, *, device, batch_size, weight):
    from .training import predict_fold
    # Fixed index order, never select examples using observed correctness.
    chosen = sorted(indices)[:2]
    def predict(d, batch):
        return predict_fold(d, chosen, checkpoint, device=device, batch_size=batch, use_cache=False)
    original = predict(data, batch_size)
    repeated = predict(data, batch_size)
    smaller = predict(data, max(1, batch_size // 2))
    order = np.random.default_rng(17).permutation(len(data['candidate_records']))
    permuted = dict(data)
    for key in ('candidate_records', 'document_texts', 'no_numeric_document_texts'):
        permuted[key] = [data[key][i] for i in order]
    permuted['pair_features'] = np.asarray(data['pair_features'])[:, order]
    permuted['pair_features_by_variant'] = np.asarray(data['pair_features_by_variant'])[:, :, order]
    shuffled = predict(permuted, batch_size)
    def compare(prediction, alpha, reorder=False):
        right = correction(prediction, alpha)
        if reorder:
            right = right[:, np.argsort(order)]
        return compare_inference(correction(original, alpha), right)
    return {'state': 'complete', 'indices': chosen, 'batch_sizes': [batch_size, max(1, batch_size // 2)],
            'source': 'real uncached outer-holdout inference; no outcomes used',
            'repeat': compare(repeated, weight), 'candidate_order': compare(shuffled, weight, True),
            'batch_size': compare(smaller, weight),
            'document_repeat': compare(repeated, 0), 'document_candidate_order': compare(shuffled, 0, True),
            'document_batch_size': compare(smaller, 0)}


def reuse_external(data, method):
    """Only reuse exact visible text, candidate ordering and declared effective scores."""
    result = np.full((3, len(data['query_records']), len(data['candidate_records'])), np.nan)
    qids = {q['query_id']: i for i, q in enumerate(data['query_records'])}
    cids = [c['candidate_id'] for c in data['candidate_records']]
    receipts = []
    if (OUT / 'protocol.json').exists():
        expected = read(OUT / 'protocol.json')['legacy_artifact_sha256']
        if legacy_sources() != expected:
            raise ValueError('A historical source changed after the v2 protocol was frozen')
    legacy = read(OLD / 'results.json')['methods'][method]
    if legacy['state'] != 'complete':
        raise ValueError('Legacy baseline is incomplete')
    sources = [(OLD, OLD / 'results' / method / legacy['effective_score_file'])]
    clinical_ref = read(CLINICAL / 'ranking_lock.json')['methods'][method]
    clinical_path = ROOT / clinical_ref['path']
    if sha(clinical_path) != clinical_ref['sha256']:
        raise ValueError('Clinical baseline ranking changed')
    sources.append((CLINICAL, clinical_path))
    for base, path in sources:
        qs = [json.loads(s) for s in (base / 'inputs/queries.jsonl').read_text().splitlines() if s.strip()]
        cs = [json.loads(s) for s in (base / 'inputs/candidates.jsonl').read_text().splitlines() if s.strip()]
        if [c['candidate_id'] for c in cs] != cids or [c['text'] for c in cs] != data['document_texts']:
            raise ValueError('Candidate pool/text differs; cached baseline cannot be reused')
        index = read(path.parent / 'index.json')
        scores = np.load(path, allow_pickle=False)
        if index['candidate_ids'] != cids or index['query_ids'] != [q['query_id'] for q in qs]:
            raise ValueError('Cached baseline index mismatch')
        expected_variants = list(data['variant_names']) if base == OLD else ['original']
        if index['variants'] != expected_variants or scores.shape != (len(expected_variants), len(qs), len(cids)):
            raise ValueError('Cached score dimensions/variants do not match the declared inputs')
        if base == OLD:
            core_receipt = read(CORE / 'assessment_repaired_v2/automatic_evaluation.json')['methods'][method]['progress']
            core_path = CORE / core_receipt['effective_score_file']
            if sha(core_path) != core_receipt['effective_scores_sha256']:
                raise ValueError('Frozen historical core score SHA mismatch')
            core_index = read(core_path.parent / 'index.json')
            positions = {q['query_id']: j for j, q in enumerate(qs)}
            if core_index['candidate_ids'] != cids or not np.array_equal(
                    scores[:, [positions[q] for q in core_index['query_ids']]], np.load(core_path, allow_pickle=False)):
                raise ValueError('Historical full score file differs from its certified core subset')
        for j, q in enumerate(qs):
            i = qids[q['query_id']]
            for v, variant in enumerate(data['variant_names']):
                source_text = q['text'] if v == 0 else q.get('variants', {}).get(variant, {}).get('text', q['text'])
                if source_text != data['query_texts'][v][i]:
                    raise ValueError('Query visible input differs; baseline requires new inference')
                result[v, i] = scores[v if v < scores.shape[0] else 0, j]
        receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                         'index_sha256': sha(path.parent / 'index.json'), 'exact_visible_inputs_verified': True})
    if not np.isfinite(result).all():
        raise ValueError('Cached baseline union is incomplete')
    return result, receipts


def external_experiment(data, *, device='cuda:0', batch_size=32):
    from .external import ensure_weights, create_adapter, MODEL_SPECS
    destination = OUT / 'external'
    for method in EXTERNAL:
        folder = destination / method
        payload = {'method': method, 'model': MODEL_SPECS.get(method), 'max_length': 2048,
                   'instruction': None,
                   'query_texts': data['query_texts'], 'document_texts': data['document_texts'],
                   'query_ids': [q['query_id'] for q in data['query_records']],
                   'candidate_ids': [c['candidate_id'] for c in data['candidate_records']]}
        binding = {'input_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                   'data_manifest_sha256': sha(OUT / 'manifest.json'),
                   'adapter_sha256': sha(Path(__file__).parent / 'external.py'),
                   'orchestrator_sha256': sha(Path(__file__)),
                   'shared_base_sha256': sha(ROOT / 'scripts/frontier_external_models.py')}
        if (folder / 'binding.json').exists() and read(folder / 'binding.json') != binding:
            raise ValueError('External input/model/implementation changed; refusing mixed resume')
        if not (folder / 'binding.json').exists():
            write(folder / 'binding.json', binding)
        if (folder / 'complete.json').exists():
            receipt = read(folder / 'complete.json')
            if sha(folder / 'scores.npz') != receipt['score_sha256']:
                raise ValueError('Completed external score artifact changed')
            with np.load(folder / 'scores.npz', allow_pickle=False) as saved:
                if saved['scores'].shape != (3, len(data['query_records']), len(data['candidate_records'])) or not np.isfinite(saved['scores']).all():
                    raise ValueError('Completed external scores are incomplete')
            continue
        status('external', method=method)
        if method not in ('prorank', 'contextual'):
            scores, receipt = reuse_external(data, method)
            array_write(folder / 'scores.npz', scores=scores)
            write(folder / 'complete.json', {'state': 'complete', 'training': False, 'reused': True,
                  'sources': receipt, 'score_sha256': sha(folder / 'scores.npz')})
            continue
        path = ensure_weights(method)
        adapter = create_adapter(method, path, device=device, batch_size=batch_size, max_length=2048)
        score_path = folder / 'partial.npz'
        shape = (3, len(data['query_records']), len(data['candidate_records']))
        scores = np.load(score_path, allow_pickle=False)['scores'] if score_path.exists() else np.full(shape, np.nan, np.float32)
        documents = data['document_texts']
        adapter.prepare_documents(documents)
        if not (folder / 'integrity.json').exists():
            audit_indices = [data['core_indices'][0], data['execution_indices'][0]]
            original = np.stack([adapter.score_query(data['query_texts'][0][i]) for i in audit_indices])
            repeated = np.stack([adapter.score_query(data['query_texts'][0][i]) for i in audit_indices])
            actual_batch = adapter.batch_size
            adapter.batch_size = max(1, actual_batch // 2)
            smaller = np.stack([adapter.score_query(data['query_texts'][0][i]) for i in audit_indices])
            adapter.batch_size = actual_batch
            order = np.random.default_rng(17).permutation(len(documents))
            shuffled = np.stack([adapter.score_query(data['query_texts'][0][i], [documents[j] for j in order])
                                  for i in audit_indices])[:, np.argsort(order)]
            adapter.prepare_documents(documents)
            write(folder / 'integrity.json', {'state': 'complete', 'indices': audit_indices,
                  'batch_sizes': [actual_batch, max(1, actual_batch // 2)],
                  'source': 'real frozen-weight inference at registered max_length2048',
                  'repeat': compare_inference(original, repeated), 'candidate_order': compare_inference(original, shuffled),
                  'batch_size': compare_inference(original, smaller)})
        seen = {}
        # Identical request strings are safely reused for these pointwise adapters.
        for v in range(3):
            for i, text in enumerate(data['query_texts'][v]):
                if np.isfinite(scores[v, i]).all():
                    seen[text] = scores[v, i].copy()
        work = sum(not np.isfinite(scores[v, i]).all() for v in range(3) for i in range(shape[1]))
        completed = 0
        priority = list(data['core_indices']) + list(data['execution_indices'])
        priority += [i for i in range(shape[1]) if i not in set(priority)]
        for v in range(3):
            for i in priority:
                text = data['query_texts'][v][i]
                if np.isfinite(scores[v, i]).all():
                    continue
                scores[v, i] = seen[text] if text in seen else adapter.score_query(text)
                seen[text] = scores[v, i].copy()
                # Identical documents must remain tied, not gain a batched numerical preference.
                groups = defaultdict(list)
                for j, document in enumerate(documents):
                    groups[document].append(j)
                for ids in groups.values():
                    scores[v, i, ids] = float(np.mean(scores[v, i, ids], dtype=np.float64))
                completed += 1
                if completed % 10 == 0 or completed == work:
                    array_write(score_path, scores=scores)
                    status('external_inference', method=method, variant=data['variant_names'][v],
                           completed_rows=completed, remaining_rows=work-completed, batch_size=adapter.batch_size)
        metadata = adapter.metadata
        adapter.close()
        if not np.isfinite(scores).all():
            raise ValueError('New external scores incomplete')
        array_write(folder / 'scores.npz', scores=scores)
        write(folder / 'complete.json', {'state': 'complete', 'training': False, 'reused': False,
            'metadata': metadata, 'score_sha256': sha(folder / 'scores.npz'),
            'integrity_sha256': sha(folder / 'integrity.json'),
            'artifacts_sha256': {'integrity.json': sha(folder / 'integrity.json'), 'binding.json': sha(folder / 'binding.json')}})
        score_path.unlink(missing_ok=True)
