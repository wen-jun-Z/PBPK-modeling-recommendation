#!/usr/bin/env python3
"""Run only the own small PBPK network, reusing frozen text and baseline scores."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from pbpk_kb.recommendation.pbpk_match_v2.experiment import (
    sha, read, write, array_write, EXTERNAL, verify_job, complete_job, fit_job,
    prediction_arrays, select_error_weight, correction, compare_inference)
OUT = ROOT / 'experiments/model_recommendation/pbpk_match_light'
DATA = ROOT / 'experiments/model_recommendation/pbpk_match_v2'
RUNTIME = ROOT / 'runtime/deployment/pbpk_match_light'
SCRIPT = Path(__file__).resolve()
PYTHON = ROOT / 'runtime/app-env/bin/python'
_LAST = 0.


def ref(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def prepare():
    from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
    from pbpk_match_light_inputs import load_cache
    data = load_dataset()
    cache = load_cache()
    data.update({k: cache[k] for k in ('light_query_embeddings', 'light_candidate_embeddings', 'light_embedding_manifest')})
    return data


def freeze(data):
    files = [SCRIPT, ROOT / 'scripts/pbpk_match_light_training.py', ROOT / 'scripts/pbpk_match_light_inputs.py']
    files += sorted((ROOT / 'src/pbpk_kb/recommendation/pbpk_match_v2').glob('*.py'))
    files += [ROOT / 'scripts/recommendation_execution_statistics.py',
              ROOT / 'src/pbpk_kb/recommendation/pbpk_match/evaluation.py']
    value = {'schema': 'pbpk-match-light-internal-grouped-v1', 'seeds': [17],
        'dataset_manifest': ref(DATA / 'manifest.json'), 'cache_manifest': ref(OUT / 'cache/manifest.json'),
        'implementation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'query_ids': [q['query_id'] for q in data['query_records']],
        'candidate_ids': [c['candidate_id'] for c in data['candidate_records']],
        'groups': list(data['groups']), 'folds': data['folds'],
        'outer_folds': 5, 'inner_folds': 3, 'total_fits': 20, 'methods': ['pbpk_match_light'],
        'maximum_epochs': 40, 'patience': 5, 'learning_rate': .001, 'error_epochs': 80,
        'training_batch': 'one full query, all 280 candidates', 'hidden_width': 128,
        'text_encoder_training': False, 'new_external_inference': False,
        'pretrained_encoder': data['light_embedding_manifest'],
        'error_weights': [0., .1, .3, 1.], 'minimum_error_training_studies': 3,
        'selection_rule': 'Inner OOF only: NDCG decrease <=0.02; coverage nondecrease; same intersection of measurable Cmax cases. No benefit -> 0.',
        'calibration_rule': 'Temperature from inner OOF; no outer labels',
        'refit_rule': 'Median inner best epochs; fixed outer epochs; no outer early stopping',
        'validation_scope': 'Grouped internal validation, one seed, no independent new data',
        'clinical_rule': 'Exact ranks 1/2/3, tie expectations, no backfill; exact endpoint windows',
        'external_methods': [{'method': m, 'scores': ref(DATA / 'external' / m / 'scores.npz'),
                              'receipt': ref(DATA / 'external' / m / 'complete.json')} for m in EXTERNAL]}
    path = OUT / 'protocol.json'
    if path.exists() and read(path) != value:
        raise ValueError('Frozen protocol or inputs changed; refuse to mix runs')
    if not path.exists():
        write(path, value)
    return value


def status(phase, **extra):
    if (RUNTIME / 'stop.requested').exists():
        raise InterruptedError('Own-model stop requested')
    old = read(RUNTIME / 'status.json') if (RUNTIME / 'status.json').exists() else {}
    value = {**old, **extra, 'phase': phase, 'pid': os.getpid(),
             'updated_at': datetime.now(timezone.utc).isoformat()}
    value.setdefault('started_at', value['updated_at'])
    write(RUNTIME / 'status.json', value)
    print(f'{phase}: {extra}', flush=True)


def progress(event):
    global _LAST
    now = time.monotonic()
    if now - _LAST >= 10 or event.get('stage') == 'epoch_complete':
        _LAST = now
        status('training', training_progress=event)


def binding(train, val, prediction, **config):
    return {'method': 'pbpk_match_light', 'seed': 17,
            'dataset_sha256': sha(DATA / 'manifest.json'), 'protocol_sha256': sha(OUT / 'protocol.json'),
            'cache_manifest_sha256': sha(OUT / 'cache/manifest.json'),
            'train_indices': list(train), 'validation_indices': list(val), 'prediction_indices': list(prediction),
            'configuration': config}


def integrity(data, indices, checkpoint, device, weight):
    from pbpk_match_light_training import predict_fold
    chosen = sorted(indices)[:2]
    def pred(d, batch):
        return predict_fold(d, chosen, checkpoint, device=device, batch_size=batch)
    original = pred(data, 280)
    repeated, smaller = pred(data, 280), pred(data, 140)
    order = np.random.default_rng(17).permutation(len(data['candidate_records']))
    shuffled_data = dict(data)
    shuffled_data['candidate_records'] = [data['candidate_records'][i] for i in order]
    shuffled_data['light_candidate_embeddings'] = data['light_candidate_embeddings'][order]
    shuffled_data['pair_features_by_variant'] = data['pair_features_by_variant'][:, :, order]
    shuffled = pred(shuffled_data, 280)
    def comparison(p, alpha, reorder=False):
        right = correction(p, alpha)
        if reorder:
            right = right[:, np.argsort(order)]
        return compare_inference(correction(original, alpha), right)
    return {'state': 'complete', 'indices': chosen, 'batch_sizes': [280, 140],
        'source': 'Actual fresh head inference on prespecified outer holdout requests; encoder cache fixed',
        'repeat': comparison(repeated, weight), 'candidate_order': comparison(shuffled, weight, True),
        'batch_size': comparison(smaller, weight), 'document_repeat': comparison(repeated, 0),
        'document_candidate_order': comparison(shuffled, 0, True), 'document_batch_size': comparison(smaller, 0)}


def train(data, device):
    from pbpk_match_light_training import train_fold, predict_fold
    from pbpk_kb.recommendation.pbpk_match_v2.evaluation import fit_oof_temperature, calibrate
    supervised = set(data['core_indices']) | set(data['execution_indices'])
    done = len(list((OUT / 'runs/seed17').glob('*/*/complete.json')))
    for fold_id, fold in enumerate(data['folds']):
        folder = OUT / f'runs/seed17/fold{fold_id}'
        pooled, indices, epochs = defaultdict(list), [], []
        for inner, split in enumerate(fold['inner']):
            tr, va = sorted(supervised & set(split['train'])), sorted(supervised & set(split['validation']))
            job = folder / f'inner{inner}'
            status('training', fold=fold_id, fit=f'inner{inner}', completed_jobs=done, total_jobs=20)
            bind = binding(tr, va, va, maximum_epochs=40, patience=5, learning_rate=.001)
            meta = verify_job(job, bind, 'validation.npz')
            if meta is None:
                meta = fit_job(train_fold, data, tr, va, job, seed=17, device=device,
                               epochs=40, patience=5, learning_rate=.001, pk_epochs=80, callback=progress)
                p = predict_fold(data, va, meta['checkpoint'], device=device)
                array_write(job / 'validation.npz', **prediction_arrays(p, va))
                complete_job(job, meta, 'validation.npz')
                done += 1
            epochs.append(int(meta['best_epoch']))
            with np.load(job / 'validation.npz', allow_pickle=False) as a:
                indices.extend(a['indices'].tolist())
                for key in a.files:
                    if key != 'indices':
                        pooled[key].append(a[key])
        if len(set(indices)) != len(indices):
            raise ValueError('Duplicate inner OOF requests')
        p = {k: np.concatenate(v, axis=0) for k, v in pooled.items()}
        selected = select_error_weight(data, indices, p)
        cal = fit_oof_temperature(p['probabilities'], np.asarray(data['applicability'])[indices], [data['groups'][i] for i in indices])
        refit_epochs = max(1, int(np.median(epochs)))
        selection = {'error_weight': selected, 'calibration': cal, 'refit_epochs': refit_epochs,
                     'selection_indices': indices, 'outer_indices_not_used': fold['outer_test']}
        write(folder / 'selection.json', selection)
        tr, te = sorted(supervised & set(fold['outer_train'])), list(fold['outer_test'])
        job = folder / 'outer'
        bind = binding(tr, [], te, fixed_epochs=refit_epochs, selection_sha256=sha(folder / 'selection.json'))
        bind['selection'] = ref(folder / 'selection.json')
        status('training', fold=fold_id, fit='outer', selected_error_weight=selected['weight'],
               completed_jobs=done, total_jobs=20)
        if verify_job(job, bind, 'predictions.npz') is not None:
            continue
        meta = fit_job(train_fold, data, tr, [], job, seed=17, device=device, epochs=refit_epochs,
                       fixed_epochs=True, learning_rate=.001, pk_epochs=80, callback=progress)
        arrays = {'indices': np.asarray(te)}
        scores, bases, probs, risks, supports = [], [], [], [], []
        for variant in data['variant_names']:
            status('predicting', fold=fold_id, variant=variant, completed_jobs=done, total_jobs=20)
            p = predict_fold(data, te, meta['checkpoint'], variant=variant, device=device)
            scores.append(correction(p, selected['weight'])); bases.append(correction(p, 0))
            probs.append(calibrate(p['probabilities'], cal['temperature']))
            risks.append(p['selected_predicted_error']); supports.append(p['error_supported'])
            for endpoint, values in p.get('predicted_errors', {}).items():
                arrays[f'predicted_error__{variant}__{endpoint}'] = values
        arrays.update(scores=np.asarray(scores), base_scores=np.asarray(bases), probabilities=np.asarray(probs),
                      selected_predicted_error=np.asarray(risks), error_supported=np.asarray(supports))
        array_write(job / 'predictions.npz', **arrays)
        write(job / 'integrity.json', integrity(data, te, meta['checkpoint'], device, selected['weight']))
        complete_job(job, meta, 'predictions.npz')
        done += 1
    n, c = len(data['query_records']), len(data['candidate_records'])
    arrays = {'indices': np.arange(n), 'scores': np.full((3, n, c), np.nan),
              'base_scores': np.full((3, n, c), np.nan), 'probabilities': np.full((3, n, c, 3), np.nan),
              'selected_predicted_error': np.full((3, n, c), np.nan), 'error_supported': np.zeros((3, n), bool)}
    covered = set()
    for f in range(5):
        with np.load(OUT / f'runs/seed17/fold{f}/outer/predictions.npz', allow_pickle=False) as a:
            ix = a['indices']
            if covered & set(ix.tolist()):
                raise ValueError('Outer folds overlap')
            covered.update(ix.tolist())
            for k in arrays:
                if k != 'indices':
                    arrays[k][:, ix] = a[k]
    if covered != set(range(n)) or not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Incomplete final catalogue output')
    array_write(OUT / 'predictions/seed17.npz', **arrays)
    write(OUT / 'predictions/seed17.json', {'state': 'complete', 'protocol_sha256': sha(OUT / 'protocol.json'),
        'score_sha256': sha(OUT / 'predictions/seed17.npz'),
        'outer_receipts': {str(f): ref(OUT / f'runs/seed17/fold{f}/outer/complete.json') for f in range(5)},
        'fit_receipts': [ref(OUT / f'runs/seed17/fold{f}/{k}/complete.json') for f in range(5)
                         for k in ('inner0', 'inner1', 'inner2', 'outer')]})
    status('training_complete', completed_jobs=20, total_jobs=20)


def identity(pid):
    p = Path('/proc') / str(int(pid))
    try:
        if p.stat().st_uid != os.getuid():
            return None
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (p / 'cmdline').read_bytes().split(b'\0')
        if stat[0] == 'Z' or str(SCRIPT).encode() not in argv or b'run' not in argv:
            return None
        return {'pid': int(pid), 'start_ticks': stat[19], 'uid': os.getuid()}
    except (OSError, ValueError, IndexError):
        return None


def active():
    r = read(RUNTIME / 'worker.json') if (RUNTIME / 'worker.json').exists() else None
    return r if r and identity(r['pid']) == r else None


def run(args):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(RUNTIME / 'worker.json', identity(os.getpid()))
        if args.background_log:
            from run_pbpk_match_v2 import ProjectLog
            import run_pbpk_match_v2
            run_pbpk_match_v2.RUNTIME = RUNTIME
            sys.stdout = sys.stderr = ProjectLog()
        def stop(_s, _f):
            raise InterruptedError('Project worker received stop signal')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            import torch
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            if args.device.startswith('cuda'):
                torch.cuda.set_per_process_memory_fraction(.24)
            status('validating', error=None)
            data = prepare(); freeze(data)
            train(data, args.device)
            if args.report:
                status('reporting')
                from summarize_pbpk_match_light import summarize
                summarize()
                status('complete', finished_at=datetime.now(timezone.utc).isoformat())
        except BaseException as e:
            s = read(RUNTIME / 'status.json') if (RUNTIME / 'status.json').exists() else {}
            write(RUNTIME / 'status.json', {**s, 'phase': 'stopped' if isinstance(e, InterruptedError) else 'failed',
                  'error': str(e), 'updated_at': datetime.now(timezone.utc).isoformat()})
            traceback.print_exc()
            raise


def display():
    s = read(RUNTIME / 'status.json') if (RUNTIME / 'status.json').exists() else {}
    done = len(list((OUT / 'runs/seed17').glob('*/*/complete.json')))
    fill = round(32 * min(done / 20, 1))
    p = s.get('training_progress', {})
    return '\n'.join(['PBPK-Match · 冻结权重 / 小网络训练',
        f'┃{"█" * fill}{"░" * (32-fill)}┃ {done:2d}/20  {done/20:6.1%}',
        f'阶段 {s.get("phase","未启动")} | fold {s.get("fold","—")} {s.get("fit","")}',
        f'epoch {p.get("epoch","—")} | 当前拟合已用 {p.get("elapsed_seconds",0)/60:.1f} min',
        f'进程 {active() or "未运行"}', f'更新时间(UTC) {s.get("updated_at","—")}',
        f'错误 {s["error"]}' if s.get('error') else '外部推荐器：全部复用已有结果，无重新推理',
        'Ctrl+C只退出进度显示。'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'run', 'stop', 'status', 'watch'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--background-log', action='store_true')
    parser.add_argument('--report', action='store_true', help='Run completed statistical report after training')
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == 'run':
        run(args)
    elif args.action == 'start':
        if active():
            print(display()); return
        (RUNTIME / 'stop.requested').unlink(missing_ok=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
            PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd = [str(PYTHON), '-B', str(SCRIPT), 'run', '--device', args.device, '--background-log']
        if args.report:
            cmd.append('--report')
        with (RUNTIME / 'launcher.log').open('ab') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=stream, start_new_session=True)
        print(f'已启动自建小网络任务，PID {child.pid}。')
    elif args.action == 'stop':
        worker = active()
        if worker:
            (RUNTIME / 'stop.requested').touch()
            if identity(worker['pid']) == worker:
                os.kill(worker['pid'], signal.SIGTERM)
        print(display())
    elif args.action == 'status':
        print(display())
    else:
        try:
            while True:
                print('\033[2J\033[H' + display(), end='', flush=True)
                time.sleep(2)
        except KeyboardInterrupt:
            print()


if __name__ == '__main__':
    main()
