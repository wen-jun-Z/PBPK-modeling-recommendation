"""Local, label-free inference adapters for published retrieval models.

Model cards: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B,
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B and
https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B.
Weights must already exist in a pinned project-owned snapshot. No API, training,
gold labels, or implicit model downloads are used by these adapters.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUCTION = (
    "Given a physiologically based pharmacokinetic modelling request, retrieve "
    "PBPK model descriptions relevant to the requested compounds, biological "
    "mechanisms, population, administration route and prediction task."
)


def _project_snapshot(path: str | Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Model snapshots must belong to this project")
    if not resolved.is_dir():
        raise ValueError("Model snapshot must be a directory")
    return resolved


class ExternalModel:
    """Shared timing, finite-score and complete-pool checks."""

    def __init__(self, method: str, model_path: str | Path, *, device: str,
                 batch_size: int, max_length: int, instruction: str | None):
        if batch_size < 1 or max_length < 128:
            raise ValueError("batch_size must be positive; max_length must be >= 128")
        self.method = method
        self.model_path = _project_snapshot(model_path)
        self.device = device
        self.batch_size = batch_size
        self.requested_batch_size = batch_size
        self.oom_retries = 0
        self.max_length = max_length
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.document_encoding_seconds = 0.0
        self.last_query_seconds = 0.0
        self._documents: tuple[str, ...] | None = None
        self.document_embeddings = None
        self.model = None
        self.tokenizer = None
        self.truncated_input_count = 0
        self.tokenized_input_count = 0
        # Dynamic remote-code modules must not enter a user/global cache.
        os.environ["HF_MODULES_CACHE"] = str(ROOT / "runtime/cache/frontier_hf_modules")
        import torch
        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        if device.startswith("cuda"):
            torch.cuda.init()
            torch.cuda.set_device(torch.device(device))
            torch.cuda.reset_peak_memory_stats(device)

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    @property
    def peak_memory_bytes(self) -> int:
        return (int(self.torch.cuda.max_memory_allocated(self.device))
                if self.device.startswith("cuda") else 0)

    @property
    def metadata(self) -> dict:
        return {
            "method": self.method, "model_path": str(self.model_path),
            "device": self.device, "dtype": str(self.dtype),
            "batch_size": self.batch_size, "max_length": self.max_length,
            "requested_batch_size": self.requested_batch_size, "oom_retries": self.oom_retries,
            "instruction": self.instruction,
            "local_files_only": True, "trained_on_project_labels": False,
            "probability_semantics": "No calibrated PBPK applicability probability",
            "tokenized_input_count": self.tokenized_input_count,
            "truncated_input_count": self.truncated_input_count,
            "token_count_semantics": "tokenization attempts including any retried OOM batches",
            "timing_semantics": "synchronized single-query wall time; document indexing separate",
        }

    def prepare_documents(self, documents: Sequence[str]) -> None:
        docs = tuple(documents)
        if not docs or any(not isinstance(x, str) or not x.strip() for x in docs):
            raise ValueError("A nonempty document pool with nonempty text is required")
        self._documents = docs

    def score_query(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        raise NotImplementedError

    def _get_documents(self, documents: Sequence[str] | None) -> tuple[str, ...]:
        if documents is not None and tuple(documents) != self._documents:
            self.prepare_documents(documents)
        if self._documents is None:
            raise ValueError("Call prepare_documents or supply documents")
        return self._documents

    def _checked_scores(self, values, expected: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (expected,) or not np.isfinite(result).all():
            raise RuntimeError("Inference did not return one finite score per candidate")
        return result

    def _with_oom_retries(self, operation):
        """Retry only CUDA allocation failure, halve batch and stop at one."""
        while True:
            try:
                return operation()
            except self.torch.OutOfMemoryError:
                if self.batch_size <= 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                self.oom_retries += 1
            # Outside except so traceback tensors can be released before retry.
            gc.collect()
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()

    def score_queries(self, queries: Sequence[str], documents: Sequence[str],
                      callback: Callable | None = None) -> dict:
        self.prepare_documents(documents)
        rows, durations = [], []
        for index, query in enumerate(queries):
            scores = self.score_query(query)
            rows.append(scores)
            durations.append(self.last_query_seconds)
            if callback is not None:
                callback(index, scores, self.last_query_seconds)
        return {
            "scores": np.stack(rows) if rows else np.empty((0, len(documents)), dtype=np.float32),
            "query_seconds": durations,
            "document_encoding_seconds": self.document_encoding_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "metadata": self.metadata,
        }

    def close(self):
        self.model = None
        self.document_embeddings = None
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


class LastTokenEmbedding(ExternalModel):
    """Official Qwen3/F2LLM normalization and last-attended-token pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import AutoModel, AutoTokenizer
        options = {"local_files_only": True, "token": False, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), padding_side="left", **options)
        self.model = AutoModel.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype,
            attn_implementation="sdpa", **options).to(self.device).eval()

    def _encode(self, texts: Sequence[str]):
        return self._with_oom_retries(lambda: self._encode_batches(texts))

    def _encode_batches(self, texts: Sequence[str]):
        torch = self.torch
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch_text = list(texts[start:start + self.batch_size])
            # Count truncation explicitly; never discard candidates because of length.
            lengths = [len(x) for x in self.tokenizer(batch_text, truncation=False)["input_ids"]]
            self.tokenized_input_count += len(lengths)
            self.truncated_input_count += sum(x > self.max_length for x in lengths)
            inputs = self.tokenizer(batch_text, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**inputs, use_cache=False).last_hidden_state
                # Padding is explicitly left: the last position is always attended.
                pooled = hidden[:, -1, :].float()
                embeddings.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(embeddings)

    def prepare_documents(self, documents: Sequence[str]) -> None:
        if tuple(documents) == self._documents and self.document_embeddings is not None:
            return
        super().prepare_documents(documents)
        self._sync()
        started = time.perf_counter()
        self.document_embeddings = self._encode(self._documents)
        self._sync()
        self.document_encoding_seconds = time.perf_counter() - started

    def score_query(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        docs = self._get_documents(documents)
        self._sync()
        started = time.perf_counter()
        query_embedding = self._encode([f"Instruct: {self.instruction}\nQuery: {query}"])
        values = (query_embedding @ self.document_embeddings.T)[0].cpu().numpy()
        self._sync()
        self.last_query_seconds = time.perf_counter() - started
        return self._checked_scores(values, len(docs))

    @property
    def metadata(self):
        return {**super().metadata, "score_semantics": "cosine_similarity",
                "pooling": "last_attended_token", "normalization": "L2_float32"}


class QwenReranker(ExternalModel):
    """Official Qwen yes/no prompt; logit difference avoids sigmoid saturation."""

    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
              'based on the Query and the Instruct provided. Note that the answer '
              'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        options = {"local_files_only": True, "token": False, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), padding_side="left", **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype,
            attn_implementation="sdpa", **options).to(self.device).eval()
        self.prefix_ids = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.no_id == self.yes_id or self.no_id is None or self.yes_id is None:
            raise ValueError("Model tokenizer lacks distinct yes/no tokens")

    def score_query(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        return self._with_oom_retries(lambda: self._score_query_batches(query, documents))

    def _score_query_batches(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        docs = self._get_documents(documents)
        self._sync()
        started = time.perf_counter()
        values = []
        remaining = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        for start in range(0, len(docs), self.batch_size):
            texts = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
                     for doc in docs[start:start + self.batch_size]]
            token_ids = self.tokenizer(texts, padding=False, truncation=False,
                                       add_special_tokens=False)["input_ids"]
            self.tokenized_input_count += len(token_ids)
            self.truncated_input_count += sum(len(row) > remaining for row in token_ids)
            packed = {"input_ids": [self.prefix_ids + row[:remaining] + self.suffix_ids
                                    for row in token_ids]}
            inputs = self.tokenizer.pad(packed, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                # Producing logits for the entire sequence wastes B*T*vocabulary memory.
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                difference = logits[:, self.yes_id] - logits[:, self.no_id]
            values.extend(difference.cpu().tolist())
        self._sync()
        self.last_query_seconds = time.perf_counter() - started
        return self._checked_scores(values, len(docs))

    @property
    def metadata(self):
        return {**super().metadata, "score_semantics": "yes_minus_no_logit",
                "prompt_source": "official_Qwen3_Reranker_model_card",
                "candidate_pool": "all_supplied_candidates"}


class JinaListwiseReranker(ExternalModel):
    """Jina's published block aggregation, with explicit bounded block sizing.

    Source: jinaai/jina-reranker-v3.5/modeling.py, JinaForRanking.rerank.
    All block document embeddings share the relevance-weighted query embedding;
    separately ranking blocks and concatenating their raw scores would differ.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, token=False,
            trust_remote_code=False, padding_side="left")
        # The pinned modeling.py is reviewed, contains only local model computation,
        # and inherits Transformers Qwen3. Never trust an unpinned remote checkout.
        self.model = AutoModel.from_pretrained(
            str(self.model_path), local_files_only=True, token=False,
            trust_remote_code=True, torch_dtype=self.dtype,
            attn_implementation="sdpa").to(self.device).eval()
        self.model._tokenizer = self.tokenizer
        self.block_token_budget = min(8192, self.max_length * 4)
        self.last_block_count = 0
        self.last_max_block_tokens = 0
        for attr in ("_compute_single_batch", "_calculate_cosine_scores"):
            if not hasattr(self.model, attr):
                raise ValueError(f"Pinned Jina implementation lacks required method {attr}")

    def _truncate(self, text: str, limit: int) -> tuple[str, int]:
        token_ids = self.tokenizer(text, truncation=False)["input_ids"]
        self.tokenized_input_count += 1
        self.truncated_input_count += int(len(token_ids) > limit)
        if len(token_ids) > limit:
            return self.tokenizer.decode(token_ids[:limit], skip_special_tokens=True), limit
        return text, len(token_ids)

    def score_query(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        return self._with_oom_retries(lambda: self._score_query_blocks(query, documents))

    def _score_query_blocks(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        original_docs = self._get_documents(documents)
        self._sync()
        started = time.perf_counter()
        query, query_length = self._truncate(query, min(1024, self.max_length))
        documents_with_length = [self._truncate(doc, self.max_length) for doc in original_docs]
        # Prompt overhead includes repeated query, instruction and passage tags.
        overhead = 256 + 2 * query_length + len(self.tokenizer.encode(self.instruction))
        blocks: list[list[str]] = []
        block: list[str] = []
        block_tokens = overhead
        block_lengths = []
        for doc, length in documents_with_length:
            cost = length + 24
            if block and (len(block) >= self.batch_size or block_tokens + cost > self.block_token_budget):
                blocks.append(block)
                block_lengths.append(block_tokens)
                block, block_tokens = [], overhead
            block.append(doc)
            block_tokens += cost
        if block:
            blocks.append(block)
            block_lengths.append(block_tokens)
        doc_embeddings, query_embeddings, weights = [], [], []
        with self.torch.inference_mode():
            for block in blocks:
                output = self.model._compute_single_batch(query, block, instruction=self.instruction)
                doc_embeddings.extend(output.doc_embeds[0].cpu().float().numpy())
                query_embeddings.append(output.query_embeds[0].cpu().float().numpy())
                scores = output.scores.view(-1).cpu().float().numpy()
                weights.append(float(((1.0 + scores) / 2.0).max()))
                del output
        if not np.isfinite(weights).all() or sum(weights) <= 0:
            raise RuntimeError("Jina returned invalid cross-block aggregation weights")
        pooled_query = np.average(np.asarray(query_embeddings), axis=0, weights=weights)
        all_docs = np.asarray(doc_embeddings)
        values = self.model._calculate_cosine_scores(pooled_query, all_docs).reshape(-1)
        self.last_block_count = len(blocks)
        self.last_max_block_tokens = max(block_lengths)
        self._sync()
        self.last_query_seconds = time.perf_counter() - started
        return self._checked_scores(values, len(original_docs))

    @property
    def metadata(self):
        return {**super().metadata,
                "score_semantics": "cosine_with_weighted_cross_block_query_embedding",
                "block_size_documents": self.batch_size,
                "block_token_budget": self.block_token_budget,
                "last_query_block_count": self.last_block_count,
                "last_query_max_estimated_block_tokens": self.last_max_block_tokens,
                "candidate_order": "supplied_order; listwise contextual dependence",
                "published_algorithm_adaptation": "bounded blocks and explicit PBPK instruction",
                "license": "CC-BY-NC-4.0",
                "source": "https://huggingface.co/jinaai/jina-reranker-v3.5/blob/main/modeling.py"}


class _GenerativeBatchMixin:
    """Shared, bounded left-padded decoding for independent ranking requests."""

    def _generate_prompts(self, prompts: Sequence[str], *, allow_over_budget: bool = False):
        if self.tokenizer.padding_side != "left":
            raise ValueError("Batched causal generation requires left padding")
        inputs = self.tokenizer(
            list(prompts), padding=True, truncation=False, add_special_tokens=False,
            return_attention_mask=True, return_tensors="pt")
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        accepted = [i for i, length in enumerate(lengths) if length <= self.max_input_tokens]
        if len(accepted) != len(prompts):
            if not allow_over_budget:
                raise ValueError("Generative prompt exceeds frozen input budget; refusing silent candidate truncation")
            if not accepted:
                return [None] * len(prompts)
            # Retokenize the accepted requests so an oversized rejected request
            # cannot inflate the GPU padding width of the remaining requests.
            inputs = self.tokenizer(
                [prompts[i] for i in accepted], padding=True, truncation=False,
                add_special_tokens=False, return_attention_mask=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        padded_width = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                logits_to_keep=1)
        if generated.ndim != 2 or generated.shape[0] != len(accepted):
            raise RuntimeError("Generation did not return one sequence per request")
        # Generation appends after the entire padded input, not after each
        # request's attended length. Keep the first EOS, discard post-EOS pads.
        generation_config = getattr(self.model, "generation_config", None)
        eos_ids = getattr(generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        results = [None] * len(prompts)
        for row, original_index in enumerate(accepted):
            new_ids = generated[row, padded_width:].tolist()
            for position, token_id in enumerate(new_ids):
                if token_id in eos_ids:
                    new_ids = new_ids[:position + 1]
                    break
            results[original_index] = {
                "response": self.tokenizer.decode(new_ids, skip_special_tokens=True),
                "input_tokens": int(lengths[original_index]),
                "generated_token_count": len(new_ids),
                "hit_token_limit": len(new_ids) >= self.max_new_tokens,
            }
        return results

    @staticmethod
    def _query_batch(queries: Sequence[str]) -> list[str]:
        if isinstance(queries, str):
            raise ValueError("Supply a sequence of queries, not a single string")
        queries = list(queries)
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("Every query must contain nonempty text")
        return queries

    def _batch_result(self, scores, diagnostics, started: float, candidate_count: int) -> dict:
        self._sync()
        elapsed = time.perf_counter() - started
        matrix = (np.stack(scores) if scores else
                  np.empty((0, candidate_count), dtype=np.float32))
        if matrix.shape != (len(diagnostics), candidate_count) or not np.isfinite(matrix).all():
            raise RuntimeError("Batch inference did not return a complete finite candidate pool")
        # Commit counters only after all requests succeeded. OOM is deliberately
        # left to the caller, which may retry a smaller batch without partial rows.
        self.valid_generation_count += sum(row["generation_valid"] for row in diagnostics)
        self.invalid_generation_count += sum(not row["generation_valid"] for row in diagnostics)
        self.last_query_diagnostics = diagnostics[-1] if diagnostics else {}
        self.last_query_seconds = elapsed / len(diagnostics) if diagnostics else 0.0
        self.last_batch_size = len(diagnostics)
        self.last_batch_seconds = elapsed
        return {
            "scores": matrix, "diagnostics": diagnostics, "batch_seconds": elapsed,
            "query_seconds": [self.last_query_seconds] * len(diagnostics),
            "query_timing_semantics": "synchronized batch wall time divided by batch size; not single-request latency",
            "peak_memory_bytes": self.peak_memory_bytes,
        }

    def score_query(self, query: str, documents: Sequence[str] | None = None) -> np.ndarray:
        return self.score_batch([query], documents)["scores"][0]


class GenerativeTop20Reranker(_GenerativeBatchMixin, ExternalModel):
    """Published listwise checkpoint, NF4, common BM25 shortlist of 20.

    Prompts are from ReasonRank/utils.py + listwise_prompt_r1.toml and
    Rearank/rank_gpt.py:create_permutation_instruction_rearank. We disclose the
    top-20 retrieval and quantization adaptations; this is not full-pool LLM
    scoring. Invalid permutations are flagged, never silently repaired.
    """

    REASONRANK_SYSTEM = (
        "You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query. "
        "Given a query and a passage list, you first thinks about the reasoning process in the mind and then "
        "provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed "
        "within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process "
        "here </think> <answer> answer here </answer>."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.device.startswith("cuda"):
            raise ValueError("The declared NF4 protocol requires CUDA")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, token=False,
            trust_remote_code=False, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), local_files_only=True, token=False,
            trust_remote_code=False, torch_dtype=self.dtype,
            attn_implementation="sdpa", device_map={"": self.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True)).eval()
        self.reasonrank = "reasonrank" in self.method.lower()
        self.max_new_tokens = 3072 if self.reasonrank else 2048
        self.passage_token_limit = min(self.max_length, 512)
        self.max_input_tokens = 14336
        self.valid_generation_count = 0
        self.invalid_generation_count = 0
        self.last_query_diagnostics = {}
        self.bm25 = None

    def prepare_documents(self, documents: Sequence[str]) -> None:
        if tuple(documents) == self._documents and self.bm25 is not None:
            return
        super().prepare_documents(documents)
        # Reuse the experiment's exact lexical baseline, including its tie order.
        from run_frontier_comparison import BM25
        self.bm25 = BM25(self._documents)

    def _messages(self, query: str, docs: Sequence[str]) -> list[dict]:
        num = len(docs)
        if self.reasonrank:
            query = re.sub(r"\[(\d+)\]", r"(\1)", query).strip()
            content = (f"I will provide you with {num} passages, each indicated by a numerical identifier []. "
                       f"Rank the passages based on their relevance to the search query: {query}.\n\n")
            for index, doc in enumerate(docs, start=1):
                tokens = self.tokenizer.tokenize(doc.strip())
                self.tokenized_input_count += 1
                self.truncated_input_count += int(len(tokens) > self.passage_token_limit)
                passage = self.tokenizer.convert_tokens_to_string(tokens[:self.passage_token_limit])
                passage = re.sub(r"\[(\d+)\]", r"(\1)", passage)
                content += f"[{index}] {passage}\n"
            content += (f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the "
                        "search query. All the passages should be included and listed using identifiers, in "
                        "descending order of relevance. The format of the answer should be [] > [], e.g., [2] > [1].")
            return [{"role": "system", "content": self.REASONRANK_SYSTEM},
                    {"role": "user", "content": content}]
        instruction = (
            "I will provide you with passages, each indicated by number identifier []. Rank the passages based "
            "on their relevance to the search query."
            f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query."
            "The passages should be listed in descending order using identifiers. The most relevant passages "
            "should be listed first. The output format should be <answer> [] > [] </answer>, e.g., "
            "<answer> [1] > [2] </answer>."
        )
        messages = [
            {"role": "system", "content": "You are DeepRerank, an intelligent assistant that can rank passages "
             "based on their relevancy to the search query. You first thinks about the reasoning process in "
             "the mind and then provides the user with the answer."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]
        for index, doc in enumerate(docs, start=1):
            passage = " ".join(doc.replace("Title: Content: ", "").strip().split())
            self.tokenized_input_count += 1
            self.truncated_input_count += int(len(passage) > 400)
            messages.extend([
                {"role": "user", "content": f"[{index}] {passage[:400]}"},
                {"role": "assistant", "content": f"Received passage [{index}]."},
            ])
        messages.append({"role": "user", "content": (
            f'Please rank these passages according to their relevance to the search query: "{query}"\n'
            '            Follow these steps exactly:\n'
            '            1. First, within <think> tags, analyze EACH passage individually:\n'
            '            - Evaluate how well it addresses the query\n'
            '            - Note specific relevant information or keywords\n\n'
            '            2. Then, within <answer> tags, provide ONLY the final ranking in descending order '
            'of relevance using the format: [X] > [Y] > [Z]')})
        return messages

    @staticmethod
    def parse_permutation(response: str, size: int) -> list[int] | None:
        answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer is None:
            return None
        text = answer.group(1).strip()
        if not re.fullmatch(r"\[\d+\](?:\s*>\s*\[\d+\])*", text):
            return None
        indices = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", text)]
        return indices if len(indices) == size and set(indices) == set(range(size)) else None

    def score_batch(self, queries: Sequence[str], documents: Sequence[str] | None = None) -> dict:
        queries = self._query_batch(queries)
        docs = self._get_documents(documents)
        self._sync()
        started = time.perf_counter()
        orders, prompts = [], []
        for query in queries:
            initial_order = np.argsort(-self.bm25.score_query(query, docs), kind="stable")
            orders.append(initial_order)
            messages = self._messages(query, [docs[index] for index in initial_order[:20]])
            prompts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        outputs = self._generate_prompts(prompts) if prompts else []
        scores, diagnostics = [], []
        for initial_order, output in zip(orders, outputs):
            shortlisted = initial_order[:20]
            permutation = self.parse_permutation(output["response"], len(shortlisted))
            valid = permutation is not None
            final_order = (np.concatenate([shortlisted[permutation], initial_order[len(shortlisted):]])
                           if valid else initial_order)
            # Ordinal scores preserve all candidates; no logits or probabilities
            # are inferred from a generated order.
            values = np.empty(len(docs), dtype=np.float32)
            values[final_order] = np.arange(len(docs), 0, -1, dtype=np.float32)
            scores.append(self._checked_scores(values, len(docs)))
            diagnostics.append({
                **output, "generation_valid": valid,
                "status": "valid_listwise_permutation" if valid else "invalid_generation_bm25_fallback",
                "initial_candidate_indices": shortlisted.tolist(), "parsed_permutation": permutation,
            })
        return self._batch_result(scores, diagnostics, started, len(docs))

    @property
    def metadata(self):
        return {**super().metadata,
                "score_semantics": "two_stage_ordinal_rank; not calibrated probability",
                "quantization": "bitsandbytes NF4, double quantization, bfloat16 compute",
                "shortlist": "same BM25 k1=1.2 b=0.75 top20, stable frozen-order ties",
                "residual_candidates": "preserved BM25 order below reranked top20",
                "published_algorithm_adaptation": "top20 only; NF4 weights; left-padded batched Transformers decoding",
                "generation_batching": "independent queries share one weight instance; strict parsing unchanged",
                "last_generation_batch_size": getattr(self, "last_batch_size", 0),
                "timing_semantics": "synchronized batch wall time divided by batch size; not single-request latency",
                "max_new_tokens": self.max_new_tokens, "max_input_tokens": self.max_input_tokens,
                "passage_limit": {"tokens": self.passage_token_limit} if self.reasonrank else {"characters": 400},
                "valid_generation_count": self.valid_generation_count,
                "invalid_generation_count": self.invalid_generation_count,
                "invalid_output_policy": "explicit invalid_generation_bm25_fallback; excluded from generation success",
                "instruction": "official checkpoint prompt (PBPK request is the query)",
                "source": ("https://github.com/8421BCD/ReasonRank" if self.reasonrank
                           else "https://github.com/lezhang7/Rearank")}


class RankR1Top20Reranker(_GenerativeBatchMixin, ExternalModel):
    """Official Rank-R1 winner selection over BM25's top20, repeated to order it.

    Source: https://huggingface.co/ielabgroup/Rank-R1-3B-v0.1.
    A full 20-item ordering uses 19 bounded winner calls; the last item follows
    without a redundant one-item prompt. Later smaller sets are explicitly the
    model card's zero-shot cardinality setting. The base stays in BF16.
    """

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with "
        "the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> "
        "</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    USER_PROMPT = (
        'Given the query: "{query}", which of the following documents is most relevant?\n'
        '{docs}\n'
        'After completing the reasoning process, please provide only the label of the most relevant document '
        'to the query, enclosed in square brackets, within the answer tags. For example, if the third document '
        'is the most relevant, the answer should be: <think> reasoning process here </think> <answer>[3]</answer>.'
    )

    def __init__(self, *args, base_model_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model_path = _project_snapshot(base_model_path)
        adapter_config = json.loads((self.model_path / "adapter_config.json").read_text())
        if adapter_config.get("base_model_name_or_path") != "Qwen/Qwen2.5-3B-Instruct":
            raise ValueError("Rank-R1-3B requires its declared Qwen2.5-3B-Instruct base")
        if adapter_config.get("peft_type") != "LORA":
            raise ValueError("Unexpected Rank-R1 adapter type")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path), local_files_only=True, token=False,
            trust_remote_code=False, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_path), local_files_only=True, token=False,
            trust_remote_code=False, torch_dtype=self.dtype,
            attn_implementation="sdpa").to(self.device).eval()
        model = PeftModel.from_pretrained(
            base, str(self.model_path), local_files_only=True, token=False,
            is_trainable=False)
        self.model = model.merge_and_unload(safe_merge=True).eval()
        self.bm25 = None
        self.max_new_tokens = 2048
        self.max_input_tokens = 14336
        self.passage_token_limit = min(self.max_length, 512)
        self.per_query_budget_seconds = 1800
        self.valid_generation_count = 0
        self.invalid_generation_count = 0
        self.total_winner_calls = 0
        self.last_query_diagnostics = {}

    def prepare_documents(self, documents: Sequence[str]) -> None:
        if tuple(documents) == self._documents and self.bm25 is not None:
            return
        super().prepare_documents(documents)
        from run_frontier_comparison import BM25
        self.bm25 = BM25(self._documents)

    @staticmethod
    def parse_winner(response: str, size: int) -> int | None:
        match = re.search(r"<think>.*?</think>\s*<answer>(.*?)</answer>", response, re.DOTALL)
        if match is None or re.fullmatch(r"\[\d+\]", match.group(1).strip()) is None:
            return None
        value = int(match.group(1).strip()[1:-1]) - 1
        return value if 0 <= value < size else None

    def _passage(self, doc: str) -> str:
        tokens = self.tokenizer.tokenize(doc.strip())
        self.tokenized_input_count += 1
        self.truncated_input_count += int(len(tokens) > self.passage_token_limit)
        return self.tokenizer.convert_tokens_to_string(tokens[:self.passage_token_limit])

    def score_batch(self, queries: Sequence[str], documents: Sequence[str] | None = None) -> dict:
        queries = self._query_batch(queries)
        docs = self._get_documents(documents)
        self._sync()
        started = time.perf_counter()
        states = []
        for query in queries:
            initial_order = np.argsort(-self.bm25.score_query(query, docs), kind="stable")
            shortlist = initial_order[:20].tolist()
            states.append({
                "query": query, "initial_order": initial_order, "shortlist": shortlist,
                "remaining": list(shortlist), "selected": [], "calls": [],
                "passages": {index: self._passage(docs[index]) for index in shortlist},
                "valid": True, "failure_reason": None,
            })
        # A batch's shared wall time is charged equally to its active requests.
        # At batch size one this is the previous synchronized per-query budget;
        # at larger sizes it is throughput cost, explicitly not request latency.
        initial_cost = (time.perf_counter() - started) / len(states) if states else 0.0
        for state in states:
            state["accounted_seconds"] = initial_cost
        while True:
            active = []
            for state in states:
                if not state["valid"] or len(state["remaining"]) <= 1:
                    continue
                if state["accounted_seconds"] > self.per_query_budget_seconds:
                    state["valid"] = False
                    state["failure_reason"] = "per_query_time_budget_reached"
                else:
                    active.append(state)
            if not active:
                break
            self._sync()
            step_started = time.perf_counter()
            prompts = []
            for state in active:
                listing = "\n".join(f"[{i+1}] {state['passages'][index]}"
                                    for i, index in enumerate(state["remaining"]))
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT.format(query=state["query"], docs=listing)},
                ]
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
            outputs = self._generate_prompts(prompts, allow_over_budget=True)
            for state, output in zip(active, outputs):
                if output is None:
                    state["valid"] = False
                    state["failure_reason"] = "frozen_input_budget_exceeded"
                    continue
                winner = self.parse_winner(output["response"], len(state["remaining"]))
                state["calls"].append({
                    **output, "candidate_indices": list(state["remaining"]),
                    "parsed_winner_position": winner,
                })
                if winner is None:
                    state["valid"] = False
                    state["failure_reason"] = "invalid_winner_output"
                else:
                    state["selected"].append(state["remaining"].pop(winner))
            self._sync()
            step_cost = (time.perf_counter() - step_started) / len(active)
            for state in active:
                state["accounted_seconds"] += step_cost
        scores, diagnostics = [], []
        for state in states:
            valid, initial_order = state["valid"], state["initial_order"]
            selected, calls = state["selected"], state["calls"]
            if valid:
                selected.extend(state["remaining"])
                final_order = np.asarray(selected + initial_order[len(state["shortlist"]):].tolist())
            else:
                final_order = initial_order
            values = np.empty(len(docs), dtype=np.float32)
            values[final_order] = np.arange(len(docs), 0, -1, dtype=np.float32)
            scores.append(self._checked_scores(values, len(docs)))
            diagnostics.append({
                "generation_valid": valid,
                "status": "valid_setwise_ordering" if valid else "invalid_generation_bm25_fallback",
                "failure_reason": state["failure_reason"],
                "initial_candidate_indices": state["shortlist"],
                "winner_calls": calls, "winner_call_count": len(calls),
                "generated_token_count": sum(row["generated_token_count"] for row in calls),
                "hit_token_limit": any(row["hit_token_limit"] for row in calls),
                "selected_candidate_indices": selected if valid else None,
                "budget_accounted_seconds": state["accounted_seconds"],
                "budget_timing_semantics": "shared preparation / batch size plus winner-stage wall time / active requests",
            })
        result = self._batch_result(scores, diagnostics, started, len(docs))
        self.total_winner_calls += sum(row["winner_call_count"] for row in diagnostics)
        return result

    @property
    def metadata(self):
        return {**super().metadata,
                "base_model_path": str(self.base_model_path),
                "score_semantics": "two_stage_ordinal_rank; not calibrated probability",
                "quantization": "none; base BF16 and merged published LoRA",
                "shortlist": "same BM25 k1=1.2 b=0.75 top20, stable frozen-order ties",
                "residual_candidates": "preserved BM25 order below reranked top20",
                "published_algorithm_adaptation": "iterative winner selection from20 down to2; later cardinalities are zero-shot",
                "generation_batching": "independent active requests share one weight instance at each winner stage",
                "last_generation_batch_size": getattr(self, "last_batch_size", 0),
                "timing_semantics": "synchronized batch wall time divided by batch size; not single-request latency",
                "maximum_winner_calls_per_query": 19,
                "max_new_tokens_per_call": self.max_new_tokens,
                "max_input_tokens": self.max_input_tokens,
                "passage_token_limit": self.passage_token_limit,
                "per_query_budget_seconds": self.per_query_budget_seconds,
                "budget_timing_semantics": "shared preparation / batch size plus winner-stage wall time / active requests",
                "valid_generation_count": self.valid_generation_count,
                "invalid_generation_count": self.invalid_generation_count,
                "winner_calls_this_attempt": self.total_winner_calls,
                "invalid_output_policy": "explicit invalid_generation_bm25_fallback; excluded from generation success",
                "instruction": "official Rank-R1 setwise prompt",
                "source": "https://huggingface.co/ielabgroup/Rank-R1-3B-v0.1"}


def load_model(method: str, model_path: str | Path, device: str = "cuda:0",
               batch_size: int = 8, max_length: int = 2048,
               instruction: str | None = None,
               base_model_path: str | Path | None = None) -> ExternalModel:
    """Load a published method from a pinned, already downloaded local snapshot."""
    normalized = method.lower().replace("_", "-")
    options = dict(device=device, batch_size=batch_size,
                   max_length=max_length, instruction=instruction)
    if "rank-r1" in normalized or "rankr1" in normalized:
        if base_model_path is None:
            raise ValueError("Rank-R1 requires an explicitly pinned local base_model_path")
        return RankR1Top20Reranker(method, model_path, base_model_path=base_model_path, **options)
    if "qwen3" in normalized and "rerank" in normalized:
        return QwenReranker(method, model_path, **options)
    if (("qwen3" in normalized and "embed" in normalized) or "f2llm" in normalized):
        return LastTokenEmbedding(method, model_path, **options)
    if "jina" in normalized and "rerank" in normalized:
        return JinaListwiseReranker(method, model_path, **options)
    if "reasonrank" in normalized or "rearank" in normalized:
        return GenerativeTop20Reranker(method, model_path, **options)
    raise ValueError(f"Unsupported external method: {method}")
