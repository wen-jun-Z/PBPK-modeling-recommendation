#!/usr/bin/env python3
"""Frozen input cache for the lightweight PBPK matcher; no fitted/target data.

Reuse old vectors only for exactly identical raw text. Encode each unique public
request/catalogue text once with the same local, frozen Qwen3-Embedding-0.6B. No labels, execution
outputs, candidate IDs or provenance identifiers enter an encoder or feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.recommendation.pbpk_match.data import (FIELDS, TASKS, TYPES, VARIANTS,
    node_specs, scoped, sha, write_json)

OLD = ROOT / "experiments/model_recommendation/pbpk_match"
DATA = ROOT / "experiments/model_recommendation/pbpk_match_v2"
DEFAULT_OUTPUT = ROOT / "experiments/model_recommendation/pbpk_match_light/cache"
REGISTRY = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/results/qwen3_embedding_06b/weights.json"
SCHEMA = "pbpk-match-light-inputs-v1"


def _read(path):
    return json.loads(scoped(path).read_text())


def _atomic_arrays(path, **arrays):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def visible_inputs(source=DATA):
    """Return an explicit allowlist, keeping outcome arrays entirely unopened."""
    source = scoped(source)
    manifest = _read(source / "manifest.json")
    if manifest.get("completed") is not True:
        raise ValueError("Frozen v2 input set is incomplete")
    for ref in [*manifest["outputs"], manifest["producer"]]:
        if sha(ROOT / ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen v2 input checksum differs: " + ref["path"])
    raw = _read(source / "dataset.json")
    keys = ("query_records", "candidate_records", "query_texts", "document_texts",
            "variant_names", "pair_feature_names", "numeric_feature_indices")
    data = {key: raw[key] for key in keys}
    # ID fields identify rows only. Texts, typed nodes and feature arrays are the
    # sole numerical inputs; the mixed metadata file's error_records are ignored.
    with np.load(source / "arrays.npz", allow_pickle=False) as arrays:
        data["condition_features"] = arrays["pair_features_by_variant"].astype(np.float32)
    if data["condition_features"].shape[:3] != (3, len(data["query_records"]), len(data["candidate_records"])):
        raise ValueError("Condition feature dimensions differ from public inputs")
    return data


def input_texts(data):
    """Canonical external-comparator inputs: one vector per unique raw text."""
    return sorted({text for rows in data["query_texts"] for text in rows}
                  | set(data["document_texts"]))


def old_text_vectors():
    """Reconstruct the old documented lexical ordering and verify its digest."""
    manifest = _read(OLD / "cache/embedding_manifest.json")
    snapshot = _read(OLD / "development_snapshot.json")
    core = ROOT / "experiments/model_recommendation/frontier_core_145_20260921"
    for name, expected in snapshot["input_sha256"].items():
        if sha(core / name) != expected:
            raise ValueError("Old node-cache source changed: " + name)
    dev_ids = set(snapshot["development_query_ids"])
    queries = [json.loads(line) for line in (core / "inputs/queries.jsonl").read_text().splitlines() if line]
    queries = [q for q in queries if q["query_id"] in dev_ids]
    candidates = [json.loads(line) for line in (core / "inputs/candidates.jsonl").read_text().splitlines() if line]
    texts = sorted({t for q in queries for v in VARIANTS for _, t in node_specs(q, v)}
                   | {t for c in candidates for _, t in node_specs(c, candidate=True)})
    # Match the historical json.dumps settings exactly; this is not our new
    # cache digest convention.
    token = hashlib.sha256(json.dumps({"texts": texts, "revision": manifest["revision"],
        "encoding": "last-token-l2-typed-fields-v1"}, sort_keys=True).encode()).hexdigest()
    array = OLD / "cache/frozen_embeddings.npz"
    if token != manifest["input_key"] or sha(array) != manifest["array_sha256"]:
        raise ValueError("Old typed-node mapping or embedding checksum differs")
    with np.load(array, allow_pickle=False) as saved:
        vectors = saved["embeddings"].astype(np.float16)
    if vectors.shape != (len(texts), manifest["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("Invalid old node vectors")
    return dict(zip(texts, vectors)), manifest


def _encode_missing(texts, missing, values, completed, output, binding, *, device, batch_size, callback):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from pbpk_kb.recommendation.pbpk_match.serving import _verify_encoder_registry

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Frozen new text encoding requires the requested GPU")
    registry = _verify_encoder_registry(binding["encoder"], REGISTRY, verify_weights=True)
    options = dict(local_files_only=True, token=False, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(registry["path"], padding_side="left", **options)
    lengths = {i: len(tokenizer(texts[i], truncation=False)["input_ids"]) for i in missing}
    if any(length < 1 or length > 4096 for length in lengths.values()):
        raise ValueError("Public text outside 1..4096 tokens; truncation is prohibited")
    order = sorted(missing, key=lambda i: (lengths[i], i))
    free, total = torch.cuda.mem_get_info(device)
    if free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB free for frozen embedding; no other job is changed")
    torch.cuda.set_per_process_memory_fraction(min(12 * 1024**3, total) / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModel.from_pretrained(registry["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", **options).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pos, size = 0, batch_size
    try:
        while pos < len(order):
            indices = order[pos:pos + size]
            encoded = hidden = batch = None
            try:
                encoded = tokenizer([texts[i] for i in indices], padding=True,
                    truncation=False, return_tensors="pt").to(device)
                if not bool(encoded["attention_mask"][:, -1].all()):
                    raise ValueError("Last-token pooling encountered padding")
                with torch.inference_mode():
                    hidden = model(**encoded, use_cache=False).last_hidden_state[:, -1].float()
                    batch = torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy().astype(np.float16)
                if not np.isfinite(batch).all():
                    raise ValueError("Nonfinite frozen text embedding")
                values[indices], completed[indices] = batch, True
                pos += len(indices)
                # A small text-vector cache is checkpointed after every batch;
                # interrupted jobs never re-encode an acknowledged batch.
                _atomic_arrays(output / "embeddings.partial.npz", vectors=values, completed=completed)
                callback({"phase": "frozen_embedding", "completed": int(completed.sum()),
                          "total": len(texts), "batch_size": size})
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                size = max(1, size // 2)
            finally:
                del encoded, hidden, batch
                torch.cuda.empty_cache()
        return {"maximum_new_input_tokens": max(lengths.values(), default=0),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "final_batch_size": size, "device": device, "truncated_inputs": 0}
    finally:
        del model
        torch.cuda.empty_cache()


def assemble(data, texts, vectors):
    lookup = {text: i for i, text in enumerate(texts)}
    qi = np.asarray([[lookup[text] for text in rows] for rows in data["query_texts"]], np.int32)
    ci = np.asarray([lookup[text] for text in data["document_texts"]], np.int32)
    return dict(raw_query_embeddings=vectors[qi].astype(np.float32),
        raw_candidate_embeddings=vectors[ci].astype(np.float32),
        query_text_indices=qi, candidate_text_indices=ci)


def build_cache(output=DEFAULT_OUTPUT, *, source=DATA, device="cuda:0", batch_size=32,
                callback: Callable | None = None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output, source = scoped(output), scoped(source)
    output.mkdir(parents=True, exist_ok=True)
    data = visible_inputs(source)
    texts = input_texts(data)
    old_vectors, old_manifest = old_text_vectors()
    registry = _read(REGISTRY)
    if (registry["repo"], registry["revision"]) != (old_manifest["repo"], old_manifest["revision"]):
        raise ValueError("Pinned encoder differs from the existing embedding cache")
    binding = {"schema": SCHEMA, "producer_sha256": sha(__file__),
        "source": str(source.relative_to(ROOT)), "source_manifest_sha256": sha(source / "manifest.json"),
        "texts_sha256": _hash_json(texts), "registry_sha256": sha(REGISTRY),
        "old_embedding_manifest_sha256": sha(OLD / "cache/embedding_manifest.json"),
        "old_input_source_sha256": sha(ROOT / "src/pbpk_kb/recommendation/pbpk_match/data.py"),
        "encoder": {"repo": registry["repo"], "revision": registry["revision"], "trainable": False},
        "query_ids": [q["query_id"] for q in data["query_records"]],
        "candidate_ids": [c["candidate_id"] for c in data["candidate_records"]],
        "labels_used_as_features": False, "encoding": "last-token-l2-typed-fields-v1"}
    if (output / "binding.json").exists() and _read(output / "binding.json") != binding:
        raise ValueError("Input cache binding changed; explicitly choose a new output directory")
    write_json(output / "binding.json", binding)
    if (output / "manifest.json").exists():
        return load_cache(output)
    values = np.zeros((len(texts), old_manifest["dimension"]), np.float16)
    completed = np.zeros(len(texts), bool)
    reused = 0
    for i, text in enumerate(texts):
        if text in old_vectors:
            values[i], completed[i] = old_vectors[text], True
            reused += 1
    partial = output / "embeddings.partial.npz"
    if partial.exists():
        with np.load(partial, allow_pickle=False) as saved:
            if saved["vectors"].shape != values.shape or saved["completed"].shape != completed.shape:
                raise ValueError("Partial embedding shape differs")
            if not np.isfinite(saved["vectors"][saved["completed"]]).all():
                raise ValueError("Invalid completed partial vectors")
            if not np.array_equal(saved["vectors"][completed], values[completed]):
                raise ValueError("Resumed vectors changed reused legacy embeddings")
            values, completed = saved["vectors"], saved["completed"]
    missing = np.flatnonzero(~completed).tolist()
    callback = callback or (lambda event: print(json.dumps(event), flush=True))
    callback({"phase": "input_cache", "total_unique_texts": len(texts), "reused_old_texts": reused,
              "remaining_texts": len(missing)})
    runtime = _encode_missing(texts, missing, values, completed, output, binding, device=device,
        batch_size=batch_size, callback=callback) if missing else {"reused_completed_embeddings": True}
    arrays = assemble(data, texts, values)
    _atomic_arrays(output / "inputs.npz", **arrays)
    metadata = {"schema": SCHEMA, "completed": True, "binding_sha256": sha(output / "binding.json"),
        "array_sha256": sha(output / "inputs.npz"), "unique_texts": len(texts), "reused_old_texts": reused,
        "new_texts": len(texts) - reused, "num_types": len(TYPES), "num_tasks": len(TASKS),
        "variant_names": list(VARIANTS), "embedding_dimension": values.shape[-1],
        "feature_names": data["pair_feature_names"],
        "query_records": data["query_records"], "candidate_records": data["candidate_records"],
        "numeric_feature_indices": data["numeric_feature_indices"],
        "runtime": runtime, "encoder_parameters_updated": 0,
        "representation_scope": "public request/catalogue text and typed metadata only; all 280 candidates"}
    write_json(output / "manifest.json", metadata)
    partial.unlink(missing_ok=True)
    return load_cache(output)


def load_cache(output=DEFAULT_OUTPUT):
    output = scoped(output)
    metadata = _read(output / "manifest.json")
    binding = _read(output / "binding.json")
    if (metadata.get("completed") is not True or metadata.get("schema") != SCHEMA
            or metadata["binding_sha256"] != sha(output / "binding.json")
            or metadata["array_sha256"] != sha(output / "inputs.npz")
            or binding["producer_sha256"] != sha(__file__)
            or binding["source_manifest_sha256"] != sha(ROOT / binding["source"] / "manifest.json")
            or binding["registry_sha256"] != sha(REGISTRY)):
        raise ValueError("Light input cache is incomplete or changed")
    with np.load(output / "inputs.npz", allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    data.update({key: metadata[key] for key in ("query_records", "candidate_records", "variant_names",
        "num_types", "num_tasks", "feature_names", "numeric_feature_indices")})
    data["manifest"], data["binding"] = metadata, binding
    data["light_query_embeddings"] = data["raw_query_embeddings"]
    data["light_candidate_embeddings"] = data["raw_candidate_embeddings"]
    data["light_embedding_manifest"] = {"repo": binding["encoder"]["repo"],
        "revision": binding["encoder"]["revision"], "input_sha256": binding["texts_sha256"],
        "array_sha256": metadata["array_sha256"], "binding_sha256": metadata["binding_sha256"],
        "source_manifest_sha256": binding["source_manifest_sha256"], "encoder_trainable": False,
        "maximum_input_tokens": metadata["runtime"].get("maximum_new_input_tokens"),
        "cache_manifest_sha256": sha(output / "manifest.json")}
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = load_cache(args.output) if args.check else build_cache(args.output, source=args.source,
        device=args.device, batch_size=args.batch_size)
    print(json.dumps({"queries": len(result["query_records"]), "candidates": len(result["candidate_records"]),
        "features": len(result["feature_names"]), "unique_texts": result["manifest"]["unique_texts"],
        "reused_old_texts": result["manifest"]["reused_old_texts"]}))
