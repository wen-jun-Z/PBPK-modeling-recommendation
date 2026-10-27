"""Full-catalog external baselines; resumable, isolated and label-free at inference.

No retired weak labels are used for training or scientific evaluation. Raw scores
are retained so independently verified judgments can be evaluated subsequently.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/model_recommendation/frontier_comparison_20260914'
RUN = ROOT / 'runtime/deployment/frontier_comparison'
CACHE = ROOT / 'runtime/hf_cache/frontier'
METHODS = {
    'bm25_text': {'kind': 'bm25', 'repo': None},
    'qwen3_embedding_06b': {'kind': 'qwen3_embedding', 'repo': 'Qwen/Qwen3-Embedding-0.6B'},
    'f2llm_v2_06b': {'kind': 'f2llm_v2', 'repo': 'codefuse-ai/F2LLM-v2-0.6B'},
    'qwen3_embedding_4b': {'kind': 'qwen3_embedding', 'repo': 'Qwen/Qwen3-Embedding-4B'},
    'qwen3_reranker_06b': {'kind': 'qwen3_reranker', 'repo': 'Qwen/Qwen3-Reranker-0.6B'},
    'jina_reranker_v35': {'kind': 'jina_reranker_v35', 'repo': 'jinaai/jina-reranker-v3.5'},
    'bm25_top20_reasonrank_7b_nf4': {'kind': 'reasonrank', 'repo': 'liuwenhan/reasonrank-7B'},
    'bm25_top20_rearank_7b_nf4': {'kind': 'rearank', 'repo': 'le723z/Rearank-7B'},
    'bm25_top20_rank_r1_3b': {'kind': 'rank_r1', 'repo': 'ielabgroup/Rank-R1-3B-v0.1',
                             'base_repo': 'Qwen/Qwen2.5-3B-Instruct'},
    'g_reasoner_34m': {'kind': 'g_reasoner', 'repo': 'rmanluo/G-reasoner-34M'},
    'gfm_rag_8m': {'kind': 'gfm_rag', 'repo': 'rmanluo/GFM-RAG-8M',
                   'embedding_repo': 'sentence-transformers/all-mpnet-base-v2'},
    'gfm_rag_8m_idf_continuous': {'kind': 'gfm_rag', 'repo':'rmanluo/GFM-RAG-8M',
                   'embedding_repo':'sentence-transformers/all-mpnet-base-v2',
                   'ranker_override':'idf_continuous'},
}
VARIANTS = ('original', 'reordered', 'missing_evidence')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean_environment():
    # Preserve the current execution environment's network routing without
    # reading user files or logging proxy values. Do not inherit model/API keys.
    network = {k: os.environ[k] for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',
               'http_proxy','https_proxy','all_proxy','no_proxy') if k in os.environ}
    return {
        **network,
        'PATH': str(ROOT / 'runtime/app-env/bin') + ':/usr/bin:/bin',
        'LANG': 'C.UTF-8', 'PYTHONPATH': str(ROOT / 'src'),
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'TMPDIR': str(RUN), 'XDG_CACHE_HOME': str(CACHE),
        'HF_HOME': str(CACHE), 'HF_HUB_CACHE': str(CACHE / 'hub'),
        'HF_MODULES_CACHE': str(CACHE / 'modules'),
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_XET': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0',
        'HF_ENDPOINT': 'https://huggingface.co',
        'HF_HUB_ETAG_TIMEOUT': '20', 'HF_HUB_DOWNLOAD_TIMEOUT': '60',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
    }


def init():
    for folder in (OUT, RUN, CACHE):
        folder.mkdir(parents=True, exist_ok=True)
    os.environ.update(clean_environment())
    # Downloader retries can contain signed transport URLs. Keep only the
    # structured, credential-free failure diagnostics written by this runner.
    for name in ('huggingface_hub','urllib3','requests'):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def load_inputs():
    paths = [OUT / 'inputs' / name for name in ('queries.jsonl', 'candidates.jsonl', 'manifest.json')]
    if not all(p.exists() for p in paths):
        raise RuntimeError('Frozen full-catalog inputs are not ready')
    q, c = [[json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in paths[:2]]
    if len(q) != 772 or len(c) != 280:
        raise ValueError('Expected the full 772-query, 280-candidate collection')
    if len({x['query_id'] for x in q}) != len(q) or len({x['candidate_id'] for x in c}) != len(c):
        raise ValueError('Duplicate input identifiers')
    return q, c, {str(p.relative_to(ROOT)): digest(p) for p in paths}


class BM25:
    """Same k1=1.2/b=.75 as existing baseline, using common masked text."""
    def __init__(self, documents):
        self.docs = [Counter(re.findall(r'[a-z0-9]+', s.casefold())) for s in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(self.docs) - n + .5) / (n + .5)) for t, n in df.items()}

    def score_query(self, query, documents):
        import numpy as np
        terms = Counter(re.findall(r'[a-z0-9]+', query.casefold()))
        scores = []
        for d, n in zip(self.docs, self.lengths):
            value = sum(count * self.idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (.25 + .75*n/max(self.avg, 1)))
                        for t, count in terms.items() if d[t])
            scores.append(value)
        return np.asarray(scores, dtype='float32')


def download(method, repo_override=None):
    """Anonymous public weights, bounded file set, pinned commit and disk guard."""
    from huggingface_hub import HfApi, snapshot_download
    repo = repo_override or METHODS[method]['repo']
    registry_path = OUT / 'external_weights.json'
    pin_lock = (RUN/'weights_registry.lock').open('a')
    fcntl.flock(pin_lock, fcntl.LOCK_EX)
    registry = read_json(registry_path, [])
    entry = next((x for x in registry if x.get('repo') == repo and x.get('sha')), None)
    if not entry:
        info = HfApi(token=False).model_info(repo, files_metadata=True, timeout=20)
        files = [x for x in info.siblings if not x.rfilename.startswith(('onnx/', 'openvino/'))
                 and (x.rfilename.endswith(('.safetensors', '.json', '.txt', '.model', '.py', '.jinja', '.md', '.pth'))
                      or x.rfilename in ('LICENSE', 'LICENSE.txt'))]
        entry = {'repo': repo, 'sha': info.sha, 'bytes': sum(x.size or 0 for x in files),
                 'files': [x.rfilename for x in files]}
        registry = [x for x in registry if x.get('repo') != repo] + [entry]
        write_json(registry_path, registry)
    fcntl.flock(pin_lock, fcntl.LOCK_UN)
    pin_lock.close()
    if entry['bytes'] > 25 * 1024**3:
        raise RuntimeError('Weight set exceeds declared 25 GiB per-model download budget')
    if shutil.disk_usage(ROOT).free < entry['bytes'] + 30 * 1024**3:
        raise RuntimeError('Insufficient project filesystem headroom')
    weight_filename = 'weights.json' if repo_override is None else 'dependency_weights_'+repo.rsplit('/',1)[-1]+'.json'
    previous = read_json(OUT / 'results' / method / weight_filename, {})
    if previous.get('revision') == entry['sha'] and Path(previous.get('path', '/nonexistent')).is_dir():
        return previous
    path = snapshot_download(repo_id=repo, revision=entry['sha'], token=False,
                             cache_dir=str(CACHE / 'hub'), allow_patterns=entry['files'], max_workers=4)
    snapshot = Path(path)
    hashes = {name: digest(snapshot / name) for name in entry['files'] if (snapshot / name).is_file()}
    result = {'repo': repo, 'revision': entry['sha'], 'path': path,
              'downloaded_at': now(), 'bytes_declared': entry['bytes'], 'sha256': hashes}
    write_json(OUT / 'results' / method / weight_filename, result)
    return result


def scored_batches(adapter, queries, documents, pending, variant, query_batch_size, guard):
    """One weight copy, several independent requests; retry OOM before committing rows."""
    import numpy as np
    position, size, oom_retries = 0, query_batch_size, 0
    while position < len(pending):
        guard()
        indices = pending[position:position + size]
        texts = [queries[i]['text'] if variant == 'original' else queries[i]['variants'][variant]['text']
                 for i in indices]
        failed_oom = False
        started = time.perf_counter()
        try:
            if hasattr(adapter, 'score_batch') and query_batch_size > 1:
                result = adapter.score_batch(texts, documents)
                values = np.asarray(result['scores'], dtype='float32')
                diagnostics = result['diagnostics']
            else:
                values = adapter.score_query(texts[0], documents)
                if isinstance(values, dict):
                    values = values['scores']
                values = np.asarray(values, dtype='float32').reshape(1, -1)
                diagnostics = [getattr(adapter, 'last_query_diagnostics', None)]
            elapsed = time.perf_counter() - started
        except Exception as exc:
            import torch
            if not isinstance(exc, torch.OutOfMemoryError) or len(indices) == 1:
                raise
            failed_oom = True
            oom_retries += 1
            size = max(1, len(indices) // 2)
        if failed_oom:
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            continue
        if values.shape != (len(indices), len(documents)) or not np.isfinite(values).all():
            raise RuntimeError('Batch returned missing, nonfinite or wrong-shape candidate scores')
        if len(diagnostics) != len(indices) or not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError('Batch diagnostics or timing coverage mismatch')
        yield indices, values, diagnostics, elapsed, oom_retries
        position += len(indices)


def reconcile_generation_journal(dest, scores, timings, queries):
    """Retain interrupted generation records separately before deterministic resumption."""
    import numpy as np
    path=dest/'generation_diagnostics.jsonl'
    lookup={row['query_id']:i for i,row in enumerate(queries)}
    committed=np.isfinite(scores).all(axis=2)&np.isfinite(timings)
    kept, interrupted, seen, invalidate=[],[],set(),[]
    lines=path.read_text().splitlines(keepends=True) if path.exists() else []
    missing_newline=bool(lines and not lines[-1].endswith('\n'))
    for i,line in enumerate(lines):
        try: row=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith('\n'):
                raise RuntimeError('Corrupt generation journal interior; refuse unsafe resumption')
            interrupted.append({'reason':'partial_tail','raw_record':line})
            continue
        if row.get('query_id') not in lookup or row.get('variant') not in VARIANTS:
            raise RuntimeError('Generation journal refers to unknown frozen input')
        key=VARIANTS.index(row['variant']),lookup[row['query_id']]
        if not committed[key]:
            interrupted.append({'reason':'no_committed_score_and_timing','record':row})
            continue
        if key in seen:
            raise RuntimeError('Duplicate committed generation journal record; explicit audit required')
        seen.add(key);kept.append(line if line.endswith('\n') else line+'\n')
    # An older worker wrote its arrays before diagnostics. Retain such an
    # unfinished row for inspection and recompute it, rather than inventing text.
    for vi,qi in np.argwhere(committed):
        if (int(vi),int(qi)) not in seen:
            interrupted.append({'reason':'committed_array_without_generation_record',
                'variant':VARIANTS[vi],'query_id':queries[qi]['query_id'],
                'scores':scores[vi,qi].tolist(),'seconds':float(timings[vi,qi])})
            invalidate.append((vi,qi))
    if interrupted:
        with (dest/'interrupted_generation_records.jsonl').open('a') as f:
            for row in interrupted:f.write(json.dumps({'recorded_at':now(),**row},allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        for key in invalidate:timings[key]=np.nan
        timings.flush()
    if interrupted or missing_newline:
        temporary=path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(''.join(kept));temporary.replace(path)


def benchmark_query_batches(adapter, queries, documents, requested, dest):
    """Same fixed requests, warmed serial versus shared-weight batching; no label tuning."""
    import numpy as np
    count = min(requested, 4)
    ids = np.linspace(0, len(queries)-1, count, dtype=int).tolist()
    texts = [queries[i]['text'] for i in ids]
    adapter.score_query(texts[0], documents)  # warmup, explicitly excluded
    serial, serial_diags, seconds = [], [], []
    for text in texts:
        started = time.perf_counter()
        serial.append(adapter.score_query(text, documents))
        seconds.append(time.perf_counter()-started)
        serial_diags.append(dict(adapter.last_query_diagnostics))
    successful = None
    attempts = []
    for size in sorted({2, count}):
        if size > count:
            continue
        batch_scores, batch_diags = [], []
        started = time.perf_counter()
        failed = False
        try:
            for offset in range(0, count, size):
                result = adapter.score_batch(texts[offset:offset+size], documents)
                batch_scores.extend(result['scores'])
                batch_diags.extend(result['diagnostics'])
        except Exception as exc:
            import torch
            if not isinstance(exc, torch.OutOfMemoryError):
                raise
            failed = True
        if failed:
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            attempts.append({'query_batch_size':size,'state':'out_of_memory'})
            continue
        elapsed = time.perf_counter()-started
        a, b = np.asarray(serial), np.asarray(batch_scores)
        if a.shape != b.shape or not np.isfinite(b).all() or len(batch_diags) != count:
            raise RuntimeError('Benchmark batch output coverage mismatch')
        attempts.append({'query_batch_size':size, 'state':'complete', 'wall_seconds':elapsed,
            'serial_over_batch_speedup':sum(seconds)/elapsed,
            'identical_score_rows':int(np.all(a==b,axis=1).sum()),
            'identical_response_rows':sum(
                (x['response']==y.get('response')) if 'response' in x else
                ([call['response'] for call in x.get('winner_calls',[])]==
                 [call['response'] for call in y.get('winner_calls',[])])
                for x,y in zip(serial_diags,batch_diags)),
            'diagnostics':batch_diags})
        successful = size
    if successful is None:
        successful = 1
    result = {'created_at':now(), 'query_ids':[queries[i]['query_id'] for i in ids],
        'selection':'fixed evenly spaced original-query indices; no relevance labels',
        'warmup_excluded':1, 'serial_wall_seconds':sum(seconds), 'serial_seconds':seconds,
        'serial_diagnostics':serial_diags,'batch_attempts':attempts,'selected_query_batch_size':successful,
        'selection_rule':'largest tested fitting batch; no selection on ranking outputs',
        'exclusive_gpu_usage_verified':False,
        'numerical_note':'BF16/NF4 batched matrix operations can change greedy tokens; execution mode recorded per row'}
    write_json(dest/'batch_benchmark.json',result)
    return successful


def tune_rank_r1_batches(adapter, queries, documents, requested, dest, input_hashes):
    """Compare the same 32 longest-input winner prompts; production still runs all stages."""
    import numpy as np
    path=dest/'batch_capacity.json'
    settings={'schema':'rank-r1-first-stage-capacity-v1','requested_max_batch':requested,
        'input_sha256':input_hashes,'max_new_tokens':adapter.max_new_tokens,
        'max_input_tokens':adapter.max_input_tokens,
        'implementation_sha256':{p.name:digest(p) for p in
            (Path(__file__),ROOT/'scripts/frontier_external_models.py')}}
    previous=read_json(path,{})
    if previous.get('state')=='complete' and previous.get('settings')==settings:
        return previous['selected_query_batch_size']
    def guard():
        if (RUN/'stop.requested').exists() or (RUN/'pause.requested').exists():
            raise InterruptedError('Batch capacity probe paused')
        if shutil.disk_usage(ROOT).free<20*1024**3:
            raise RuntimeError('Disk guard reached during batch capacity probe')
    guard()
    adapter.prepare_documents(documents)
    # The candidate text preprocessing and prompts are identical to stage one
    # of production. Select by input length only, never by model answers/labels.
    passages=[adapter._passage(doc) for doc in documents]
    prompts=[]
    for row in queries:
        order=np.argsort(-adapter.bm25.score_query(row['text'],documents),kind='stable')[:20]
        listing='\n'.join(f'[{i+1}] {passages[index]}' for i,index in enumerate(order))
        messages=[{'role':'system','content':adapter.SYSTEM_PROMPT},
                  {'role':'user','content':adapter.USER_PROMPT.format(query=row['text'],docs=listing)}]
        prompts.append(adapter.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    lengths=[len(x) for x in adapter.tokenizer(prompts,add_special_tokens=False)['input_ids']]
    count=min(32,len(queries))
    ids=sorted(range(len(queries)),key=lambda i:(-lengths[i],i))[:count]
    selected=[prompts[i] for i in ids]
    if max(lengths[i] for i in ids)>adapter.max_input_tokens:
        raise RuntimeError('Capacity probe contains an over-budget frozen input')
    result={'state':'running','created_at':now(),'settings':settings,
        'query_ids':[queries[i]['query_id'] for i in ids],
        'selection':'same longest-token first-stage prompts; frozen-order length ties; no labels',
        'input_tokens':[lengths[i] for i in ids],
        'prompt_sha256':[hashlib.sha256(text.encode()).hexdigest() for text in selected],
        'warmup_excluded':1,'attempts':[],
        'scope':'First winner stage capacity/throughput only; not full 19-stage acceleration or scientific quality',
        'exclusive_gpu_usage_verified':False}
    write_json(path,result)
    adapter._generate_prompts(selected[:1]);adapter._sync()
    for size in (4,8,16,32):
        if size>requested or size>count:
            continue
        guard();gc.collect();adapter.torch.cuda.empty_cache()
        adapter.torch.cuda.reset_peak_memory_stats()
        free_before,total=adapter.torch.cuda.mem_get_info()
        progress=read_json(dest/'status.json',{})
        progress.update(state='tuning_query_batches',capacity_probe_batch=size,updated_at=now())
        write_json(dest/'status.json',progress)
        started=time.perf_counter();outputs=[];failed=False
        try:
            for offset in range(0,count,size):
                guard()
                outputs.extend(adapter._generate_prompts(selected[offset:offset+size]))
            adapter._sync()
        except adapter.torch.OutOfMemoryError:
            failed=True
        if failed:
            outputs.clear();gc.collect();adapter.torch.cuda.empty_cache()
            result['attempts'].append({'query_batch_size':size,'state':'out_of_memory'})
            write_json(path,result)
            break
        elapsed=time.perf_counter()-started
        if len(outputs)!=count or any(row is None for row in outputs):
            raise RuntimeError('Capacity probe output coverage mismatch')
        free_after,_=adapter.torch.cuda.mem_get_info()
        result['attempts'].append({'query_batch_size':size,'state':'complete',
            'queries':count,'wall_seconds':elapsed,'queries_per_second':count/elapsed,
            'generated_tokens':sum(row['generated_token_count'] for row in outputs),
            'generated_tokens_per_second':sum(row['generated_token_count'] for row in outputs)/elapsed,
            'valid_winner_outputs':sum(adapter.parse_winner(row['response'],20) is not None for row in outputs),
            'hit_token_limit':sum(row['hit_token_limit'] for row in outputs),
            'response_sha256':[hashlib.sha256(row['response'].encode()).hexdigest() for row in outputs],
            'peak_allocated_bytes':adapter.torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':adapter.torch.cuda.max_memory_reserved(),
            'free_bytes_before':free_before,'free_bytes_after':free_after,'total_device_bytes':total})
        write_json(path,result)
    completed=[row for row in result['attempts'] if row['state']=='complete']
    safe=[row for row in completed if row['free_bytes_after']>=4*1024**3]
    if safe:
        chosen=min(safe,key=lambda row:row['wall_seconds'])['query_batch_size']
    else:
        chosen=min((row['query_batch_size'] for row in completed),default=1)
    result.update(state='complete',finished_at=now(),selected_query_batch_size=chosen,
        selection_rule='Fastest measured first-stage throughput with >=4 GiB free; otherwise smallest completed batch. No selection on ranking outputs.',
        production_validation='Pending full winner-chain batches; halve on OOM without partial commits.')
    write_json(path,result)
    gc.collect();adapter.torch.cuda.empty_cache()
    adapter.torch.cuda.reset_peak_memory_stats()
    return chosen


def run_method(method, batch_size=16, max_length=2048, query_batch_size=1, benchmark_batches=False, tune_query_batches=False):
    import numpy as np
    q, c, input_hashes = load_inputs()
    dest = OUT / 'results' / method
    dest.mkdir(parents=True, exist_ok=True)
    method_lock = (RUN/f'method_{method}.lock').open('a')
    fcntl.flock(method_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = dest / 'status.json'
    state = read_json(state_path, {})
    if state.get('input_sha256') not in (None, input_hashes):
        raise RuntimeError('Frozen inputs changed; refuse to overwrite previous results')
    if state.get('state') == 'complete':
        return
    if state.get('implementation_sha256'):
        state.setdefault('execution_attempts',[]).append({k:state[k] for k in
            ('worker_pid','worker_start_ticks','implementation_sha256','updated_at','completed_query_variants',
             'query_batch_size_requested','last_execution') if k in state})
    if state.get('state') == 'failed':
        state.setdefault('prior_attempts', []).append({k:state[k] for k in
            ('error_type','failed_phase','error_frames','exit_code','updated_at') if k in state})
        for key in ('error_type','failed_phase','error_frames','exit_code'):
            state.pop(key,None)
    state.update({'method': method, 'state': 'initializing', 'started_at': state.get('started_at', now()),
                  'worker_pid': os.getpid(), 'worker_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                  'input_sha256': input_hashes, 'queries': len(q), 'candidates': len(c),
                  'variants': list(VARIANTS), 'training': False, 'scientific_claim_eligible': False,
                  'batch_size_requested': batch_size, 'max_length': max_length,
                  'query_batch_size_requested':query_batch_size})
    implementation_files = [Path(__file__), ROOT/'scripts/frontier_external_models.py']
    if METHODS[method]['kind'] in ('g_reasoner','gfm_rag'):
        implementation_files.append(ROOT/'scripts/frontier_graph_models.py')
    state['implementation_sha256'] = {p.name:digest(p) for p in implementation_files}
    write_json(state_path, state)
    start = time.monotonic()
    documents = [row['text'] for row in c]
    if method == 'bm25_text':
        adapter = BM25(documents)
    else:
        state.update(state='downloading', updated_at=now())
        write_json(state_path, state)
        weights = download(method)
        options = {}
        if METHODS[method].get('base_repo'):
            options['base_model_path'] = download(method, METHODS[method]['base_repo'])['path']
        if METHODS[method].get('embedding_repo'):
            options['embedding_path'] = download(method, METHODS[method]['embedding_repo'])['path']
        state.update(state='waiting_for_gpu', revision=weights['revision'], updated_at=now())
        write_json(state_path, state)
        gpu_lock = (RUN/'gpu.lock').open('w')
        wait_started = time.monotonic()
        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if (RUN/'stop.requested').exists():
                    state.update(state='stopped', updated_at=now())
                    write_json(state_path,state)
                    return
                if time.monotonic()-wait_started > 50*3600:
                    raise TimeoutError('Project GPU queue wait exceeded 50 hours')
                time.sleep(2)
        state['gpu_queue_wait_seconds'] = time.monotonic()-wait_started
        start += state['gpu_queue_wait_seconds']
        state.update(state='loading_gpu',updated_at=now())
        write_json(state_path,state)
        if METHODS[method]['kind'] in ('g_reasoner','gfm_rag'):
            from frontier_graph_models import FrontierGraphScorer
            if METHODS[method].get('ranker_override'):
                options['ranker_override'] = METHODS[method]['ranker_override']
            if method == 'g_reasoner_34m':
                options['embedding_path'] = download('qwen3_embedding_06b')['path']
            adapter = FrontierGraphScorer(model_path=weights['path'],cache_dir=str(CACHE/'graphs'/method),
                device='cuda:0',batch_size=min(batch_size,4),**options)
            adapter.prepare_context(q,c)
        else:
            from frontier_external_models import load_model
            adapter = load_model(METHODS[method]['kind'], weights['path'], device='cuda:0',
                                 batch_size=min(batch_size,8) if method=='jina_reranker_v35' else batch_size,
                                 max_length=max_length, **options)
        initial_block_size = getattr(adapter,'batch_size',None)
    if not hasattr(adapter,'score_batch'):
        query_batch_size = 1
    if tune_query_batches and METHODS[method]['kind']=='rank_r1':
        state.update(state='tuning_query_batches',updated_at=now())
        write_json(state_path,state)
        counter_names=('valid_generation_count','invalid_generation_count','total_winner_calls',
                       'tokenized_input_count','truncated_input_count')
        saved_counters={name:getattr(adapter,name) for name in counter_names if hasattr(adapter,name)}
        query_batch_size=tune_rank_r1_batches(adapter,q,documents,query_batch_size,dest,input_hashes)
        for name,value in saved_counters.items():setattr(adapter,name,value)
    elif benchmark_batches and query_batch_size > 1 and not (dest/'batch_benchmark.json').exists():
        state.update(state='benchmarking_batches',updated_at=now())
        write_json(state_path,state)
        counter_names=('valid_generation_count','invalid_generation_count','total_winner_calls',
                       'tokenized_input_count','truncated_input_count')
        saved_counters={name:getattr(adapter,name) for name in counter_names if hasattr(adapter,name)}
        query_batch_size = benchmark_query_batches(adapter,q,documents,query_batch_size,dest)
        for name,value in saved_counters.items():setattr(adapter,name,value)
    elif benchmark_batches and query_batch_size > 1:
        query_batch_size = min(query_batch_size, read_json(dest/'batch_benchmark.json')['selected_query_batch_size'])
    state['query_batch_size_effective_initial'] = query_batch_size
    state.update(state='scoring', updated_at=now())
    state['initialization_seconds_this_attempt'] = time.monotonic() - start
    state['exclusive_gpu_usage_verified'] = False
    write_json(state_path, state)
    shape = (len(VARIANTS), len(q), len(c))
    score_path, timing_path = dest / 'scores.npy', dest / 'query_seconds.npy'
    if score_path.exists():
        scores = np.lib.format.open_memmap(score_path, mode='r+')
        timings = np.lib.format.open_memmap(timing_path, mode='r+')
        if scores.shape != shape or timings.shape != shape[:2]:
            raise RuntimeError('Resumption shape mismatch')
    else:
        scores = np.lib.format.open_memmap(score_path, mode='w+', dtype='float32', shape=shape)
        timings = np.lib.format.open_memmap(timing_path, mode='w+', dtype='float64', shape=shape[:2])
        scores[:] = np.nan
        timings[:] = np.nan
        scores.flush(); timings.flush()
    write_json(dest / 'index.json', {'query_ids': [r['query_id'] for r in q],
                                    'candidate_ids': [r['candidate_id'] for r in c], 'variants': list(VARIANTS)})
    if hasattr(adapter,'last_query_diagnostics'):
        reconcile_generation_journal(dest,scores,timings,q)
    attempt_id = now()
    last_flush = 0.0
    last_report = time.monotonic()
    def guard():
        if (RUN / 'stop.requested').exists() or (RUN/'pause.requested').exists():
            raise InterruptedError('Project experiment pause requested at a completed batch boundary')
        if shutil.disk_usage(ROOT).free < 20*1024**3 or time.monotonic() - start > 48*3600:
            raise RuntimeError('Disk/runtime guard reached; no automatic retry loop')
    for vi, variant in enumerate(VARIANTS):
        pending = [i for i in range(len(q)) if not
                   (np.isfinite(scores[vi,i]).all() and np.isfinite(timings[vi,i]))]
        for indices, values, diagnostics, elapsed, oom_retries in scored_batches(
                adapter,q,documents,pending,variant,query_batch_size,guard):
            if method=='jina_reranker_v35' and adapter.batch_size != initial_block_size:
                raise RuntimeError('Listwise block size changed; require a separately declared fixed-block run')
            execution = {'attempt_id':attempt_id, 'batch_id':f'{variant}:{indices[0]}',
                'query_batch_size':len(indices), 'batch_wall_seconds':elapsed,
                'query_seconds_semantics':'amortized_batch_wall_seconds' if len(indices)>1 else 'single_query_wall_seconds',
                'oom_retries_this_variant':oom_retries}
            # Diagnostics precede finite timing markers: interrupted batches can be replayed safely.
            for qi, diagnostic in zip(indices,diagnostics):
                if not diagnostic:
                    continue
                with (dest/'generation_diagnostics.jsonl').open('a') as f:
                    f.write(json.dumps({'query_id':q[qi]['query_id'],'variant':variant,**diagnostic,
                                        'execution':execution},ensure_ascii=False,allow_nan=False)+'\n')
            if query_batch_size > 1:
                with (dest/'execution_batches.jsonl').open('a') as f:
                    f.write(json.dumps({'variant':variant,'query_ids':[q[i]['query_id'] for i in indices],
                                        **execution},allow_nan=False)+'\n')
            scores[vi,indices] = values
            scores.flush()
            timings[vi,indices] = elapsed/len(indices)
            timings.flush()
            if time.monotonic() - last_flush >= 20 or indices[-1] == len(q)-1:
                scores.flush(); timings.flush()
                state.update(state='scoring', variant=variant, variant_completed_queries=int(np.isfinite(timings[vi]).sum()),
                             completed_query_variants=int(np.isfinite(timings).sum()),
                             completed_pairs=int(np.isfinite(scores).sum()), updated_at=now(),
                             last_execution=execution,
                             peak_gpu_allocated_bytes=getattr(adapter,'peak_memory_bytes',0))
                write_json(state_path, state)
                print(json.dumps({k:state[k] for k in ('method','variant','variant_completed_queries','completed_pairs','updated_at')}), flush=True)
                last_flush = time.monotonic()
            if method.startswith('bm25_top20_') and time.monotonic()-last_report >= 120:
                report()
                last_report=time.monotonic()
    metadata = getattr(adapter, 'metadata', {})
    if callable(metadata):
        metadata = metadata()
    state.update(state='complete', finished_at=now(), adapter=metadata,
                 completed_query_variants=int(np.isfinite(timings).sum()),
                 completed_pairs=int(np.isfinite(scores).sum()))
    state['document_encoding_seconds'] = getattr(adapter,'document_encoding_seconds',None)
    for key in ('error_type','failed_phase','error_frames','exit_code'):
        state.pop(key,None)
    if method != 'bm25_text':
        import torch
        state['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        state['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
        state['gpu'] = torch.cuda.get_device_name(0)
    write_json(state_path, state)
    report()


def tie_aware_summary(scores, queries):
    """Set-valued ranks expose unresolved ties rather than declaring an arbitrary winner."""
    import numpy as np
    best = scores == scores.max(axis=2,keepdims=True)
    thresholds = np.sort(scores,axis=2)[:,:,-10,None]
    inclusive_top10 = scores >= thresholds
    counts = best.sum(axis=2)
    result = {'original_unique_top1_queries':int((counts[0]==1).sum()),
              'original_maximum_top1_tie_size':int(counts[0].max()),
              'original_top10_boundary_tie_queries':int((inclusive_top10[0].sum(axis=1)>10).sum()),
              'interpretation':'Top1 is set-valued when tied; inclusive Top10 retains all boundary ties.'}
    for vi,variant in enumerate(VARIANTS[1:],start=1):
        changed = np.asarray([i for i,q in enumerate(queries) if q['text'] != q['variants'][variant]['text']])
        if not len(changed):
            result[variant]={'changed_input_queries':0};continue
        def jaccard(mask):
            left,right=mask[0,changed],mask[vi,changed]
            return float(np.mean((left&right).sum(axis=1)/(left|right).sum(axis=1)))
        unambiguous=changed[(counts[0,changed]==1)&(counts[vi,changed]==1)]
        result[variant]={'changed_input_queries':len(changed), 'top1_set_jaccard':jaccard(best),
            'inclusive_top10_set_jaccard':jaccard(inclusive_top10),
            'unique_top1_pair_queries':len(unambiguous),
            'unique_top1_consistency':float(np.mean(np.argmax(scores[0,unambiguous],axis=1)==
                np.argmax(scores[vi,unambiguous],axis=1))) if len(unambiguous) else None}
    return result


def report():
    import numpy as np
    queries, candidates, hashes = load_inputs()
    from frontier_result_audit import refresh_output_recovery
    recovery = refresh_output_recovery(OUT)
    methods = {}
    missing = {
        'ranking': 'No independently judged complete candidate pools. Rule labels excluded.',
        'same_drug_selection': 'No verified common-study independent-family comparison judgments.',
        'predictive_accuracy_and_regret': 'Historical PK outcomes lack verified exact joins to these requests.',
        'applicability_and_abstention': 'No independent applicability labels or validation-fitted thresholds.',
        'calibration_and_risk_coverage': 'Raw relevance scores are not calibrated applicability probabilities.',
        'generalization_quality': 'Evidence-derived requests lack independent held-out study annotations.',
        'citation_support': 'Scoring adapters emit candidate scores, not evidence-backed generated claims.',
    }
    for method in METHODS:
        folder = OUT / 'results' / method
        state = read_json(folder / 'status.json', {'state':'queued'})
        entry = {'state':state['state'], 'scientific_metrics':{k:{'value':None,'reason':v} for k,v in missing.items()}}
        if state['state'] == 'complete':
            scores = np.load(folder / 'scores.npy')
            original_scores=scores
            entry['original_score_file']='scores.npy'
            entry['effective_score_file']='scores.npy'
            if method in ('bm25_top20_rearank_7b_nf4','bm25_top20_reasonrank_7b_nf4'):
                corrected=folder/'official_recovered_scores.npy'
                budget=read_json(folder/'budget_repair_3172.json',{})
                if budget.get('state')=='complete':
                    corrected=folder/'budget_repaired_scores.npy'
                if corrected.exists():
                    scores=np.load(corrected)
                    if scores.shape!=original_scores.shape or not np.isfinite(scores).all():
                        raise RuntimeError('Published-output correction lacks complete valid score coverage')
                    entry['effective_score_file']=corrected.name
                    entry['original_strict_policy_tie_audit']=tie_aware_summary(original_scores,queries)
            if method in ('qwen3_embedding_06b','qwen3_embedding_4b','f2llm_v2_06b','qwen3_reranker_06b'):
                first={};scores=original_scores.copy()
                for ci,candidate in enumerate(candidates):
                    representative=first.setdefault(candidate['text'],ci)
                    scores[:,:,ci]=original_scores[:,:,representative]
                entry['effective_score_file']='text_equivalent_scores.npy'
            timings = np.load(folder / 'query_seconds.npy')
            # Stable sort uses the frozen candidate order for ties; tie prevalence disclosed.
            ranks = np.argsort(-scores, axis=-1, kind='stable')
            consistency = {}
            for vi, variant in enumerate(VARIANTS[1:], start=1):
                changed = [i for i,q in enumerate(queries) if q['text'] != q['variants'][variant]['text']]
                agreements, overlaps = [], []
                for qi in changed:
                    agreements.append(bool(ranks[0,qi,0] == ranks[vi,qi,0]))
                    overlaps.append(len(set(ranks[0,qi,:10]) & set(ranks[vi,qi,:10])) / 10)
                consistency[variant] = {'changed_input_queries':len(changed),
                    'top1_consistency':float(np.mean(agreements)) if agreements else None,
                    'top10_overlap':float(np.mean(overlaps)) if overlaps else None,
                    'meaning':'Sensitivity only; missing-evidence changes are not labeled correct/incorrect.'}
            top_ties = (scores[0] == scores[0].max(axis=1, keepdims=True)).sum(axis=1) > 1
            entry.update({'full_catalog_queries':len(queries), 'candidate_count':len(candidates),
                          'original_pairs':int(scores[0].size), 'all_variant_pairs':int(scores.size),
                          'operational_return_coverage':float(np.isfinite(scores[0]).all(axis=1).mean()),
                          'top1_score_tie_query_fraction':float(top_ties.mean()), 'robustness':consistency,
                          'latency':{'unit':'seconds_per_query_full_280_candidates',
                              'p50':float(np.median(timings[0,1:])), 'p95':float(np.quantile(timings[0,1:],.95)),
                              'first_query_seconds':float(timings[0,0]),
                              'warmup_excluded_queries':1, 'sustained_total_seconds':float(timings[0].sum()),
                              'includes_document_encoding_first_query':True},
                          'peak_gpu_allocated_gib':state.get('peak_gpu_allocated_bytes',0)/1024**3,
                          'quality_recommendation_coverage':None})
            entry['rank_tie_audit'] = tie_aware_summary(scores,queries)
            if entry['rank_tie_audit']['original_unique_top1_queries']==0:
                entry['output_interpretation']='unordered tied retrieval set; no identifiable Top1 ranking'
                for value in consistency.values():
                    value['stable_candidate_order_top1_consistency']=value['top1_consistency']
                    value['top1_consistency']=None
                    value['reason']='All original Top1 scores tied; use rank_tie_audit set comparisons.'
            if method in ('qwen3_embedding_06b','qwen3_embedding_4b','f2llm_v2_06b','qwen3_reranker_06b'):
                groups={}
                for ci,candidate in enumerate(candidates):
                    groups.setdefault(candidate['text'],[]).append(ci)
                duplicates=[v for v in groups.values() if len(v)>1]
                shared=scores.copy()
                for indices in duplicates:
                    shared[:,:,indices]=scores[:,:,indices[0],None]
                target=folder/'text_equivalent_scores.npy'
                if not target.exists():
                    temp=target.with_suffix(f'.{os.getpid()}.tmp')
                    with temp.open('wb') as f:np.save(f,shared)
                    temp.replace(target)
                entry['exact_text_equivalence']={'score_file':str(target.relative_to(OUT)),
                    'source_scores_sha256':digest(folder/'scores.npy'),
                    'policy':'Exact duplicate descriptions share the first frozen candidate representative score; original scores retained.',
                    'not_a_model_quality_improvement_claim':True,
                    'candidate_groups':[[candidates[i]['candidate_id'] for i in group] for group in duplicates],
                    'max_original_within_group_difference':max(float(np.ptp(original_scores[:,:,group],axis=2).max()) for group in duplicates),
                    'rank_tie_audit':tie_aware_summary(shared,queries)}
            batches_path=folder/'execution_batches.jsonl'
            if batches_path.exists():
                batches=[json.loads(line) for line in batches_path.read_text().splitlines() if line.strip()]
                mode=np.ones((len(VARIANTS),len(queries)),dtype=int)
                query_lookup={row['query_id']:i for i,row in enumerate(queries)}
                by_size={}
                for batch in batches:
                    size=str(batch['query_batch_size']);record=by_size.setdefault(size,{'queries':0,'batch_wall_seconds':0.0})
                    record['queries']+=len(batch['query_ids']);record['batch_wall_seconds']+=batch['batch_wall_seconds']
                    for query_id in batch['query_ids']:
                        mode[VARIANTS.index(batch['variant']),query_lookup[query_id]]=batch['query_batch_size']
                for record in by_size.values():
                    record['queries_per_second']=record['queries']/record['batch_wall_seconds']
                entry['throughput_by_query_batch_size']=by_size
                entry['latency']={'p50':None,'p95':None,
                    'reason':'Resumed serial and batched amortized costs must not be combined as single-request latency.',
                    'batch_benchmark_file':'batch_benchmark.json' if (folder/'batch_benchmark.json').exists() else None}
                entry['robustness_all_recorded_execution_modes']=consistency
                comparable={}
                for vi,variant in enumerate(VARIANTS[1:],start=1):
                    changed=[i for i,q in enumerate(queries) if q['text']!=q['variants'][variant]['text']]
                    same=[i for i in changed if mode[0,i]==mode[vi,i]]
                    comparable[variant]={'changed_input_queries':len(changed),
                        'same_execution_batch_size_pairs':len(same),
                        'cross_execution_mode_pairs_excluded':len(changed)-len(same),
                        'top1_consistency':float(np.mean([ranks[0,i,0]==ranks[vi,i,0] for i in same])) if same else None,
                        'top10_overlap':float(np.mean([len(set(ranks[0,i,:10])&set(ranks[vi,i,:10]))/10 for i in same])) if same else None,
                        'meaning':'Sensitivity among matched query-batch sizes; numerical batched decoding variation remains possible.'}
                entry['robustness']=comparable
            grouped = {}
            for dimension,key in (('by_task','task_type'),('by_compound','compound_id')):
                grouped[dimension] = {}
                for group in sorted({q[key] for q in queries}):
                    indices = [i for i,q in enumerate(queries) if q[key]==group]
                    values = {'queries':len(indices), 'scientific_quality_metrics':None}
                    for vi,variant in enumerate(VARIANTS[1:],start=1):
                        changed = [i for i in indices if queries[i]['text'] != queries[i]['variants'][variant]['text']]
                        total_changed=len(changed)
                        if batches_path.exists():
                            changed=[i for i in changed if mode[0,i]==mode[vi,i]]
                        values[variant] = {'changed_input_queries':total_changed,
                            'comparable_pairs':len(changed),'cross_execution_mode_pairs_excluded':total_changed-len(changed),
                            'top1_consistency':float(np.mean([ranks[0,i,0]==ranks[vi,i,0] for i in changed])) if changed else None,
                            'top10_overlap':float(np.mean([len(set(ranks[0,i,:10]) & set(ranks[vi,i,:10]))/10 for i in changed])) if changed else None}
                        if entry['rank_tie_audit']['original_unique_top1_queries']==0:
                            values[variant]['stable_candidate_order_top1_consistency']=values[variant]['top1_consistency']
                            values[variant]['top1_consistency']=None
                    grouped[dimension][group] = values
            entry['groups'] = grouped
            diagnostic_path = folder/'generation_diagnostics.jsonl'
            if diagnostic_path.exists():
                unique = {}
                for line in diagnostic_path.read_text().splitlines():
                    row = json.loads(line)
                    unique[row['query_id'],row['variant']] = row
                entry['generation'] = {v:{
                    'expected_queries':len(queries),
                    'records':sum(x.get('variant')==v for x in unique.values()),
                    'valid_generations':sum(x.get('variant')==v and x.get('generation_valid') is True for x in unique.values()),
                    'invalid_or_fallback_generations':sum(x.get('variant')==v and x.get('generation_valid') is not True for x in unique.values())
                    } for v in VARIANTS}
            entry['comparison_group'] = ('bm25_top20_generative_reranking' if method.startswith('bm25_top20_')
                                        else 'full_catalog_scoring')
            entry['exclusive_gpu_usage_verified'] = False
        if method == 'bm25_top20_reasonrank_7b_nf4' and recovery.get('reasonrank'):
            entry['official_output_recovery']={
                'counts':recovery['reasonrank'].get('counts',{}),
                'strict_generation_success_redefined':False,
                'score_file':'official_recovered_scores.npy'}
            correction_path=folder/'budget_repair_3172.json'
            if correction_path.exists():
                correction=read_json(correction_path)
                entry['official_budget_correction']={k:correction.get(k) for k in
                    ('state','total_count','completed_count','strict_success_count','usable_order_count','output_path')}
        if method == 'bm25_top20_rearank_7b_nf4':
            recovery_path=folder/'output_recovery.json'
            if recovery_path.exists():
                saved=read_json(recovery_path)
                entry['official_output_recovery']={
                    'file':str(recovery_path.relative_to(OUT)),
                    'counts':saved['rearank'].get('counts',{}),
                    'strict_generation_success_redefined':False,
                    'score_file':'official_recovered_scores.npy'}
                repaired_path=folder/'official_recovered_scores.npy'
                if state['state']=='complete' and repaired_path.exists():
                    repaired=np.load(repaired_path)
                    if np.isfinite(repaired).all():
                        entry['official_output_recovery']['rank_tie_audit']=tie_aware_summary(repaired,queries)
        methods[method] = entry
    result = {'created_at':now(), 'input_sha256':hashes, 'protocol_version':'frontier-zero-shot-v1.1',
              'mode':'full_catalog_zero_shot_inference_not_weak_label_training',
              'own_method_development':'paused', 'scientific_claim_eligible':False, 'methods':methods,
              'observed_outcomes_file':'observed_outcomes.json',
              'cohorts':{'same_drug_independent_families':{'scientific_result':None,'reason':'No independently adjudicated same-scenario multi-family gold'},
                         'same_family_versions':{'scientific_result':None,'reason':'Version duplicates retained; not independent families'},
                         'cross_drug_mechanism_reference':{'scientific_result':None,'reason':'Requires separate mechanism-reference judgment'},
                         'no_applicable_model':{'scientific_result':None,'reason':'Unknown candidates are not negatives'}},
              'metric_change_policy':'Keep raw scores and all originally declared metrics; version any additions/removals with rationale, never remove for unfavorable scores.'}
    write_json(OUT / 'results.json', result)
    return result


def run_queue(selected, batch_size, max_length, query_batch_size=1, benchmark_batches=False, tune_query_batches=False):
    # Kernel releases this lock if the process exits; no stale boolean lock.
    lock = (RUN / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'pid':os.getpid(), 'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]}
    write_json(RUN / 'pid.json', identity)
    state = {'state':'running', 'started_at':now(), 'methods':selected, **identity}
    for method in selected:
        if (RUN / 'stop.requested').exists() or (RUN/'pause.requested').exists():
            break
        state.update(current_method=method, updated_at=now())
        write_json(RUN / 'status.json', state)
        # Separate process frees VRAM between methods and isolates inference failures.
        command = [str(ROOT/'runtime/app-env/bin/python'), str(Path(__file__).resolve()), 'worker',
                   '--methods', method, '--batch-size',str(batch_size),'--max-length',str(max_length),
                   '--query-batch-size',str(query_batch_size)]
        if benchmark_batches:
            command.append('--benchmark-batches')
        if tune_query_batches:
            command.append('--tune-query-batches')
        child = subprocess.Popen(command, cwd=ROOT, env=clean_environment())
        state['worker_pid'] = child.pid
        write_json(RUN / 'status.json', state)
        deadline = time.monotonic() + 49*3600
        last_progress_check = 0.0
        last_download_size = -1
        last_download_progress = time.monotonic()
        while child.poll() is None:
            if (RUN / 'stop.requested').exists() or time.monotonic() > deadline:
                child.terminate()
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                break
            if time.monotonic() - last_progress_check > 60:
                ms = read_json(OUT/'results'/method/'status.json', {})
                if ms.get('state') == 'downloading':
                    size = sum(p.stat().st_size for p in (CACHE/'hub').rglob('*') if p.is_file() and not p.is_symlink())
                    if size != last_download_size:
                        last_download_size, last_download_progress = size, time.monotonic()
                    if time.monotonic() - last_download_progress > 20*60:
                        child.terminate(); child.wait(timeout=20)
                        ms.update(state='failed', error_type='DownloadStalled20Minutes', updated_at=now())
                        write_json(OUT/'results'/method/'status.json', ms)
                        break
                last_progress_check = time.monotonic()
            if (RUN/'runner.log').exists() and (RUN/'runner.log').stat().st_size > 20*1024**2:
                child.terminate(); child.wait(timeout=20)
                (RUN/'stop.requested').touch()
                break
            time.sleep(2)
        if child.returncode:
            dest = OUT/'results'/method/'status.json'
            ms = read_json(dest, {})
            ms.update(state='failed', exit_code=child.returncode, updated_at=now())
            write_json(dest, ms)
        subprocess.run([str(ROOT/'runtime/app-env/bin/python'),str(Path(__file__).resolve()),'report'],
                       cwd=ROOT,env=clean_environment(),check=True)
    statuses = [read_json(OUT/'results'/m/'status.json',{}).get('state') for m in selected]
    state.update(state='complete' if all(s=='complete' for s in statuses) else 'finished_with_incomplete_methods',
                 finished_at=now(), worker_pid=None)
    write_json(RUN/'status.json',state)


def alive():
    saved=read_json(RUN/'pid.json',{})
    if not saved: return False
    try:
        p=Path(f'/proc/{int(saved["pid"])}')
        return p.stat().st_uid == os.getuid() and p.joinpath('stat').read_text().split()[21] == saved['start_ticks'] and p.joinpath('stat').read_text().split()[2]!='Z'
    except (OSError,IndexError): return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','run','worker','retry','status','stop','pause','report'])
    parser.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--max-length',type=int,default=2048)
    parser.add_argument('--query-batch-size',type=int,default=1,
                        help='Independent generative queries sharing one weight copy (1-32)')
    parser.add_argument('--benchmark-batches',action='store_true')
    parser.add_argument('--tune-query-batches',action='store_true',
                        help='Probe Rank-R1 first-stage capacity at 4/8/16/32 then validate in production')
    args=parser.parse_args(); init()
    if not 1 <= args.query_batch_size <= 32:
        parser.error('--query-batch-size must be between 1 and 32')
    if args.action=='start':
        load_inputs()
        if alive(): raise SystemExit('Project frontier queue is already running')
        (RUN/'stop.requested').unlink(missing_ok=True)
        (RUN/'pause.requested').unlink(missing_ok=True)
        with (RUN/'runner.log').open('a') as log:
            command=[str(ROOT/'runtime/app-env/bin/python'),str(Path(__file__).resolve()),'run',
                '--methods',*args.methods,'--batch-size',str(args.batch_size),'--max-length',str(args.max_length),
                '--query-batch-size',str(args.query_batch_size)]
            if args.benchmark_batches:
                command.append('--benchmark-batches')
            if args.tune_query_batches:
                command.append('--tune-query-batches')
            p=subprocess.Popen(command,
                cwd=ROOT,env=clean_environment(),stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'started_pid':p.pid,'methods':args.methods,'log':str(RUN/'runner.log')}))
    elif args.action=='retry':
        if len(args.methods)!=1:
            raise SystemExit('Retry exactly one failed or stopped method')
        method=args.methods[0]
        load_inputs()
        with (RUN/f'retry_{method}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state_path=OUT/'results'/method/'status.json'
            previous=read_json(state_path,{})
            if previous.get('state') not in ('failed','stopped'):
                raise SystemExit('Method is not failed or stopped; duplicate retry refused')
            if (RUN/'stop.requested').exists() or (RUN/'pause.requested').exists():
                raise SystemExit('Queue stop is requested; do not launch a retry')
            with (RUN/'runner.log').open('a') as log:
                command=[str(ROOT/'runtime/app-env/bin/python'),str(Path(__file__).resolve()),
                    'worker','--methods',method,'--batch-size',str(args.batch_size),'--max-length',str(args.max_length),
                    '--query-batch-size',str(args.query_batch_size)]
                if args.benchmark_batches:
                    command.append('--benchmark-batches')
                if args.tune_query_batches:
                    command.append('--tune-query-batches')
                child=subprocess.Popen(command,
                    cwd=ROOT,env=clean_environment(),stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
            for _ in range(100):
                state=read_json(state_path,{})
                if state.get('worker_pid')==child.pid or child.poll() is not None:
                    break
                time.sleep(.05)
            if state.get('worker_pid')!=child.pid:
                raise RuntimeError('Retry worker did not register its identity')
            print(json.dumps({'retry_method':method,'worker_pid':child.pid,'state':state['state'],
                              'gpu_policy':'wait for existing project GPU holder; do not interrupt it'}))
    elif args.action=='run':run_queue(args.methods,args.batch_size,args.max_length,args.query_batch_size,args.benchmark_batches,args.tune_query_batches)
    elif args.action=='worker':
        try:run_method(args.methods[0],args.batch_size,args.max_length,args.query_batch_size,args.benchmark_batches,args.tune_query_batches)
        except InterruptedError:
            dest=OUT/'results'/args.methods[0]/'status.json';state=read_json(dest,{})
            state.update(state='stopped',updated_at=now())
            write_json(dest,state)
        except Exception as e:
            dest=OUT/'results'/args.methods[0]/'status.json';state=read_json(dest,{})
            state.update(failed_phase=state.get('state'), state='failed',error_type=type(e).__name__,updated_at=now(),
                         error_frames=[{'file':frame.filename,'line':frame.lineno,'function':frame.name}
                                       for frame in traceback.extract_tb(e.__traceback__)[-8:]])
            # Avoid recording signed download URLs, credentials or arbitrary external exception text.
            write_json(dest,state);print(json.dumps({'method':args.methods[0],'error_type':type(e).__name__}),flush=True)
            raise SystemExit(1)
    elif args.action=='stop':(RUN/'stop.requested').touch();print('Project queue stop requested')
    elif args.action=='pause':(RUN/'pause.requested').touch();print('Project queue will pause at a completed batch boundary')
    elif args.action=='status':
        print(json.dumps({'alive':alive(),'queue':read_json(RUN/'status.json',{}),
             'methods':{m:read_json(OUT/'results'/m/'status.json',{'state':'queued'}) for m in METHODS}},indent=2))
    elif args.action=='report':report();print(str(OUT/'results.json'))


if __name__=='__main__':main()
