"""Pinned, anonymous downloads and local-only published reranker adapters.

No project labels or clinical outcomes are used. Native scores are uncalibrated
ranking scores. The underlying official model and tokenizer files are validated
against a project-owned receipt before loading.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WEIGHTS = ROOT / 'artifacts/pbpk_match_v2/external_models'
CACHE = ROOT / 'runtime/cache/pbpk_match_v2_external'
OFFICIAL_COMMIT = '8e709dca09e0fdba6bc7f6d4983940fdd1b783c5'
OFFICIAL_PRORANK_SOURCE_SHA256 = '25d2904f4a3d4037b3bef2a8670cbd708288e05f4d5321788b6160595c8361c0'
MODEL_SPECS = {
    'prorank': {
        'repo': 'mixedbread-ai/mxbai-rerank-large-v2',
        'revision': 'ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6',
        'license': 'apache-2.0',
        'official_source': f'https://github.com/mixedbread-ai/mxbai-rerank/blob/{OFFICIAL_COMMIT}/mxbai_rerank/mxbai_rerank_v2.py',
        'score_semantics': 'last_logit_token_1_minus_token_0_in_model_dtype_then_float32',
    },
    'contextual': {
        'repo': 'ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b',
        'revision': '6ffef5dc552583b8db58dc4a87f79f7aee78d2d9',
        'license': 'cc-by-nc-sa-4.0',
        'official_source': 'https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b/blob/6ffef5dc552583b8db58dc4a87f79f7aee78d2d9/README.md',
        'score_semantics': 'last_logit_vocab_id_0_cast_bfloat16_then_float32_as_official_HF_example',
    },
}
_ALLOWED = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
            'tokenizer.model', 'sentencepiece.bpe.model', 'chat_template.jinja',
            'model.safetensors.index.json', 'LICENSE', 'LICENSE.txt', 'README.md'}
MAX_WEIGHT_BYTES = 25 * 1024**3
MIN_FREE_BYTES = 30 * 1024**3


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _inside(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('All model files and caches must belong to this project')
    return p


def _name(name):
    matches = [key for key, spec in MODEL_SPECS.items() if name in (key, spec['repo'])]
    if len(matches) != 1:
        raise ValueError('Unknown pinned model; use prorank or contextual')
    return matches[0]


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def verify_weights(name: str, path: str | Path) -> dict:
    key = _name(name)
    path = _inside(path)
    receipt = json.loads((path / 'model_receipt.json').read_text())
    spec = MODEL_SPECS[key]
    if any(receipt.get(k) != spec[k] for k in ('repo', 'revision', 'license')) or receipt.get('completed') is not True:
        raise ValueError('Weight receipt has the wrong model, revision, license or completion state')
    files = receipt.get('files', [])
    names = [r['name'] for r in files]
    if len(set(names)) != len(names) or not {'config.json', 'tokenizer_config.json', 'tokenizer.json'} <= set(names):
        raise ValueError('Incomplete or duplicate weight receipt files')
    if not any(x.endswith('.safetensors') for x in names):
        raise ValueError('Weight receipt lacks safetensors')
    for ref in files:
        target = _inside(path / ref['name'])
        if target.parent != path or target.stat().st_size != ref['bytes'] or _sha(target) != ref['sha256']:
            raise ValueError('Weight artifact changed: ' + ref['name'])
    return receipt


def ensure_weights(name: str) -> Path:
    """Download one immutable public revision, anonymously and with bounded retry.

    Only root-level inference files plus license/model-card provenance are fetched.
    Existing receipts are fully hash-checked; unregistered revisions are rejected.
    """
    key = _name(name)
    spec = MODEL_SPECS[key]
    directory = _inside(WEIGHTS / key / spec['revision'])
    cache = _inside(CACHE)
    directory.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / 'download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'model_receipt.json').exists():
            verify_weights(key, directory)
            return directory
        # Xet may otherwise create an implicit user cache. Disable it for these
        # anonymous direct downloads; every hub path is explicitly project-local.
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        os.environ['HF_XET_CACHE'] = str(cache / 'xet')
        from huggingface_hub import HfApi, hf_hub_download, constants
        constants.HF_HUB_DISABLE_XET = True
        constants.HF_HUB_DOWNLOAD_TIMEOUT = 30
        info = HfApi(token=False).model_info(spec['repo'], revision=spec['revision'], files_metadata=True, timeout=20)
        license_name = (info.card_data or {}).get('license')
        if info.sha != spec['revision'] or license_name != spec['license']:
            raise ValueError('Remote revision or license differs from pinned model')
        selected = [f for f in info.siblings if '/' not in f.rfilename
                    and (f.rfilename in _ALLOWED or f.rfilename.endswith('.safetensors'))]
        if not selected or any(not isinstance(f.size, int) or f.size < 0 for f in selected):
            raise ValueError('Download metadata lacks explicit file sizes')
        expected_bytes = sum(f.size for f in selected)
        if expected_bytes > MAX_WEIGHT_BYTES or shutil.disk_usage(ROOT).free < expected_bytes + MIN_FREE_BYTES:
            raise RuntimeError('Model exceeds bounded download size or filesystem reserve')
        registration = {'schema_version': 'pbpk-match-v2-external-download-v1', **spec,
                        'anonymous': True, 'expected_bytes': expected_bytes,
                        'max_attempts_per_file': 3, 'files': [f.rfilename for f in selected]}
        _atomic_json(directory / 'download_registration.json', registration)
        records = []
        for item in selected:
            for attempt in range(3):
                try:
                    filename = hf_hub_download(repo_id=spec['repo'], filename=item.rfilename,
                                               revision=spec['revision'], token=False,
                                               local_dir=str(directory), cache_dir=str(cache / 'hub'),
                                               etag_timeout=20)
                    break
                except (OSError, TimeoutError, ConnectionError) as error:
                    if attempt == 2:
                        raise RuntimeError('Bounded download failed: ' + item.rfilename) from error
                    time.sleep(min(2**attempt, 4))
            path = _inside(filename)
            digest = _sha(path)
            if path.parent != directory or path.stat().st_size != item.size:
                raise ValueError('Downloaded file path/size differs: ' + item.rfilename)
            lfs_sha = getattr(item.lfs, 'sha256', None) if item.lfs else None
            if lfs_sha and lfs_sha != digest:
                raise ValueError('Downloaded LFS SHA differs: ' + item.rfilename)
            if not lfs_sha and item.blob_id:
                blob = hashlib.sha1(b'blob ' + str(item.size).encode() + b'\0' + path.read_bytes()).hexdigest()
                if blob != item.blob_id:
                    raise ValueError('Downloaded Git blob differs: ' + item.rfilename)
            records.append({'name': item.rfilename, 'bytes': item.size, 'sha256': digest,
                            'remote_lfs_sha256': lfs_sha, 'remote_git_blob_id': item.blob_id})
            _atomic_json(directory / 'download_progress.json', {'completed_files': len(records),
                         'total_files': len(selected), 'verified_bytes': sum(x['bytes'] for x in records)})
        receipt = {'schema_version': 'pbpk-match-v2-external-weights-v1', **spec,
                   'completed': True, 'anonymous': True, 'files': records,
                   'total_bytes': sum(x['bytes'] for x in records)}
        _atomic_json(directory / 'model_receipt.json', receipt)
        verify_weights(key, directory)
        return directory


def _base_class():
    path = ROOT / 'scripts/frontier_external_models.py'
    spec = importlib.util.spec_from_file_location('_pbpk_match_v2_existing_external', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExternalModel


ExternalModel = _base_class()


class _LocalReranker(ExternalModel):
    MODEL_KEY = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_receipt = verify_weights(self.MODEL_KEY, self.model_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        options = {'local_files_only': True, 'token': False, 'trust_remote_code': False}
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), padding_side='left', **options)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype, attn_implementation='sdpa', use_safetensors=True, **options).to(self.device).eval()
        self.model_max_length = int(self.model.config.max_position_embeddings)
        self._configure_prompt()

    def score_query(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('A nonempty query is required')
        docs = self._get_documents(documents)
        self._sync()
        started = time.perf_counter()
        # The inherited bounded retry halves batch size to one; no CPU fallback,
        # truncation changes, missing candidates or partial result publication.
        scores = self._with_oom_retries(lambda: self._score_batches(query, docs))
        self._sync()
        self.last_query_seconds = time.perf_counter() - started
        return scores

    def _score_batches(self, query, docs):
        # Canonical batches avoid order-dependent BF16 roundoff; identical text
        # is scored once and expanded back to every original candidate position.
        unique_docs = sorted(set(docs))
        values = []
        for start in range(0, len(unique_docs), self.batch_size):
            inputs = self._prepare_inputs(query, unique_docs[start:start + self.batch_size]).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                scores = self._extract_scores(logits)
            values.extend(scores.cpu().float().tolist())
            del logits, scores, inputs
        by_document = dict(zip(unique_docs, values))
        return self._checked_scores([by_document[doc] for doc in docs], len(docs))

    @property
    def metadata(self):
        spec = MODEL_SPECS[self.MODEL_KEY]
        return {**super().metadata, **spec, 'candidate_pool': 'all_supplied_candidates',
                'weight_receipt_sha256': _sha(self.model_path / 'model_receipt.json'),
                'adapter_sha256': _sha(Path(__file__)),
                'shared_base_sha256': _sha(ROOT / 'scripts/frontier_external_models.py'),
                'output_score_dtype': 'float32', 'attention_implementation': 'sdpa',
                'official_template_source_sha256': (OFFICIAL_PRORANK_SOURCE_SHA256 if self.MODEL_KEY == 'prorank'
                    else next(r['sha256'] for r in self.weight_receipt['files'] if r['name'] == 'README.md')),
                'long_sequence_policy': 'official_model_template_with_declared_max_length',
                'clinical_error_used_as_input': False,
                'duplicate_document_policy': 'exact text deduplicated, lexicographic batches, restored to every original position'}


class ProRankReranker(_LocalReranker):
    MODEL_KEY = 'prorank'
    PREFIX = '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n'
    TASK = ('You are a search relevance expert who evaluates how well documents match search queries. '
            'For each query-document pair, carefully analyze the semantic relationship between them, '
            'then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:')

    def _configure_prompt(self):
        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        self.prefix_ids, self.suffix_ids, self.task_ids, self.sep_ids = map(encode, [self.PREFIX, self.SUFFIX, self.TASK, '\n'])
        yes, no = encode('1'), encode('0')
        if len(yes) != 1 or len(no) != 1 or yes == no:
            raise ValueError('ProRank requires distinct single tokens for 1 and 0')
        self.yes_id, self.no_id = yes[0], no[0]
        self.predefined_length = len(self.prefix_ids) + len(self.suffix_ids) + len(self.task_ids) + len(self.sep_ids)
        self.max_length = min(self.max_length, self.model_max_length - self.predefined_length)
        if self.max_length < 128:
            raise ValueError('Insufficient model context for the official ProRank template')

    def _prepare_inputs(self, query, docs):
        packed = []
        for doc in docs:
            q = 'query: ' + query
            if self.instruction:
                q = 'instruction: ' + self.instruction + '\n' + q
            qi = self.tokenizer(q, add_special_tokens=False, truncation=True,
                                max_length=self.max_length * 3 // 4)['input_ids']
            available = self.model_max_length - len(qi) - self.predefined_length
            raw_di = self.tokenizer('document: ' + doc, add_special_tokens=False, truncation=False)['input_ids']
            di = raw_di[:min(available, self.max_length)]
            item = self.tokenizer.prepare_for_model(qi, self.sep_ids + di, truncation='only_second',
                     max_length=self.max_length, padding=False, return_attention_mask=False,
                     return_token_type_ids=False, add_special_tokens=False)
            raw_q = self.tokenizer(q, add_special_tokens=False, truncation=False)['input_ids']
            self.tokenized_input_count += 1
            self.truncated_input_count += int(len(raw_q) != len(qi) or len(qi) + len(self.sep_ids) + len(raw_di) > self.max_length)
            ids = self.prefix_ids + item['input_ids'] + self.sep_ids + self.task_ids + self.suffix_ids
            if len(ids) > self.model_max_length:
                raise ValueError('Packed ProRank input exceeds original model context')
            packed.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
        return self.tokenizer.pad(packed, padding='longest', pad_to_multiple_of=8, return_tensors='pt')

    def _extract_scores(self, logits):
        # Official implementation subtracts before casting, relevant for BF16 ties.
        return (logits[:, self.yes_id] - logits[:, self.no_id]).float()


class ContextualReranker(_LocalReranker):
    MODEL_KEY = 'contextual'

    def _configure_prompt(self):
        self.max_length = min(self.max_length, self.model_max_length)

    def format_prompt(self, query, document):
        instruction = (' ' + self.instruction) if self.instruction else ''
        return ('Check whether a given document contains information helpful to answer the query.\n'
                f'<Document> {document}\n<Query> {query}{instruction} ??')

    def _prepare_inputs(self, query, docs):
        texts = [self.format_prompt(query, doc) for doc in docs]
        raw = self.tokenizer(texts, truncation=False, padding=False)['input_ids']
        self.tokenized_input_count += len(raw)
        self.truncated_input_count += sum(len(ids) > self.max_length for ids in raw)
        # Same left padding and tokenizer defaults as the pinned official HF path.
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                              max_length=self.max_length)

    def _extract_scores(self, logits):
        return logits[:, 0].to(self.torch.bfloat16).float()


def create_adapter(name: str, model_path: str | Path, *, device='cuda:0', batch_size=8,
                   max_length=8192, instruction=None):
    key = _name(name)
    cls = ProRankReranker if key == 'prorank' else ContextualReranker
    return cls(key, model_path, device=device, batch_size=batch_size,
               max_length=max_length, instruction=instruction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=sorted(MODEL_SPECS))
    args = parser.parse_args()
    print(ensure_weights(args.name), flush=True)
