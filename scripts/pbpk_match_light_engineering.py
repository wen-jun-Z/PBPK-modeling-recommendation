#!/usr/bin/env python3
"""Paper-compatible stability and measured inference cost for saved light heads.

No training, baseline inference, downloads, or evaluation-driven model selection.
The small runtime sample is fixed before observing timings. Catalogue embeddings,
structured condition features, and weights are resident/precomputed as declared.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
OUT = ROOT / "experiments/model_recommendation/pbpk_match_light"
REGISTRY = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/results/qwen3_embedding_06b/weights.json"
METHODS = {"pbpk_match_light_document": "PBPK-Match (document)", "pbpk_match_light": "PBPK-Match"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stability(data, scores, method):
    """Exactly the historical paper's fixed-size Top-10 intersection / 10."""
    indices = np.asarray(data["stability_indices"], int)
    scores = np.asarray(scores)[:, indices]
    if scores.shape != (3, 772, 280) or not np.isfinite(scores).all():
        raise ValueError("Incomplete three-condition stability predictions")
    ranks = np.argsort(-scores, axis=-1, kind="stable")
    tied = (scores[0] == scores[0].max(-1, keepdims=True)).sum(-1)
    row = {"method_id": method, "method": METHODS[method], "group": "full",
           "tie_fraction": float(np.mean(tied > 1))}
    for vi, (variant, prefix) in enumerate((("reordered", "reorder"), ("missing_evidence", "missing")), 1):
        changed = [j for j, i in enumerate(indices) if data["query_texts"][0][i] != data["query_texts"][vi][i]]
        expected = 772 if vi == 1 else 304
        if len(changed) != expected:
            raise ValueError(f"Changed-input denominator differs for {variant}")
        row[f"{prefix}_top1"] = float(np.mean(ranks[0, changed, 0] == ranks[vi, changed, 0]))
        row[f"{prefix}_top10"] = float(np.mean([len(set(ranks[0, i, :10]) & set(ranks[vi, i, :10])) / 10 for i in changed]))
        row[f"{prefix}_n"] = len(changed)
    return row


def sample_requests(data):
    """Ten lexically spaced held-out IDs per outer fold; no labels inspected."""
    eligible = set(data["stability_indices"])
    chosen = []
    for fold, split in enumerate(data["folds"]):
        pool = sorted(eligible & set(split["outer_test"]), key=lambda i: data["query_records"][i]["query_id"])
        if len(pool) < 10:
            raise ValueError("Runtime requires ten distinct holdouts per fold")
        for pos in np.linspace(0, len(pool) - 1, 10, dtype=int):
            i = pool[int(pos)]
            if i in split["outer_train"]:
                raise ValueError("Runtime fold contains training request")
            chosen.append({"fold": fold, "query_index": i, "query_id": data["query_records"][i]["query_id"]})
    if len({r["query_index"] for r in chosen}) != 50:
        raise ValueError("Runtime sample is duplicated")
    return chosen


def benchmark(data, sample, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from pbpk_kb.recommendation.pbpk_match.serving import _verify_encoder_registry
    from pbpk_kb.recommendation.pbpk_match_v2.model import ErrorHead
    from pbpk_kb.recommendation.pbpk_match_v2.training import _query_endpoint
    from pbpk_match_light_training import LightMatch, _canonical, _normalized

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This cost experiment requires the project GPU environment")
    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info(device)
    if free < 5 * 1024**3:
        raise RuntimeError("Need 5 GiB free GPU memory; no other workload is changed")
    torch.cuda.set_per_process_memory_fraction(min(8 * 1024**3, total) / total, device)
    registry = _verify_encoder_registry(data["light_embedding_manifest"], REGISTRY, verify_weights=True)
    options = {"local_files_only": True, "token": False, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(registry["path"], padding_side="left", **options)
    encoder = AutoModel.from_pretrained(registry["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", **options).to(device).eval()
    encoder.requires_grad_(False)
    candidates = data["light_candidate_embeddings"]
    measures, batch_rows, checks = [], [], []
    peak = {m: 0. for m in METHODS}
    with np.load(OUT / "predictions/seed17.npz", allow_pickle=False) as saved:
        reference = {key: saved[key] for key in ("scores", "base_scores")}

    def encode(indices):
        tokens = tokenizer([data["query_texts"][0][i] for i in indices], padding=True,
            truncation=False, return_tensors="pt").to(device)
        if tokens["input_ids"].shape[1] > 4096 or not tokens["attention_mask"][:, -1].all():
            raise ValueError("Untruncated last-token query encoding contract failed")
        with torch.inference_mode():
            last = encoder(**tokens, use_cache=False).last_hidden_state[:, -1].float()
            # Input cache uses float16 storage after L2 normalization.
            return torch.nn.functional.normalize(last, dim=-1).half().float()

    for fold in range(5):
        checkpoint = OUT / f"runs/seed17/fold{fold}/outer"
        metadata = read(checkpoint / "metadata.json")
        state = torch.load(checkpoint / "heads.pt", map_location="cpu", weights_only=True)
        selection = read(checkpoint.parent / "selection.json")
        model = LightMatch(metadata["embedding_size"], len(metadata["feature_mean"]), metadata["width"], metadata["dropout"]).to(device).eval()
        model.load_state_dict(state["heads"])
        error_metadata = state["error_metadata"]
        endpoints = error_metadata.get("enabled_endpoints", [])
        error_head = None
        if endpoints:
            error_head = ErrorHead(error_metadata["input_size"], len(endpoints)).to(device).eval()
            error_head.load_state_dict(state["error_head"])
        alpha = selection["error_weight"]["weight"]
        temperature = selection["calibration"]["temperature"]
        group = [r for r in sample if r["fold"] == fold]

        def score(indices, vectors, corrected):
            values = []
            with torch.inference_mode():
                for j, i in enumerate(indices):
                    features = _normalized(data["pair_features_by_variant"][0, i], metadata)
                    chosen, restore = _canonical(candidates, features)
                    output = model(vectors[j], torch.as_tensor(candidates[chosen], device=device),
                                   torch.as_tensor(features[chosen], device=device))
                    scores = output["scores"]
                    if corrected and alpha > 0 and error_head is not None:
                        requested = data.get("execution_requested_endpoints", {}).get(i)
                        endpoint = _query_endpoint(data["query_records"][i], requested)
                        if endpoint in endpoints:
                            e = torch.full(scores.shape, endpoints.index(endpoint), device=device, dtype=torch.long)
                            scores = scores - alpha * error_head(output["representation"], e)
                    # Materialize complete sorted output and applicability probabilities.
                    result = scores.cpu().numpy()[restore]
                    probabilities = (output["app_logits"] / temperature).softmax(-1).cpu().numpy()[restore]
                    order = np.argsort(-result, kind="stable")
                    if len(order) != 280 or not np.isfinite(probabilities).all():
                        raise ValueError("Inference did not return a complete finite catalogue")
                    values.append(result)
            return np.asarray(values)

        def timed(indices, corrected, cached):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            vectors = (torch.as_tensor(data["light_query_embeddings"][0, indices], device=device)
                       if cached else encode(indices))
            result = score(indices, vectors, corrected)
            torch.cuda.synchronize(device)
            return time.perf_counter() - started, result

        for row in group[:2]:
            for corrected in (False, True):
                timed([row["query_index"]], corrected, False)
        for local, row in enumerate(group):
            methods = list(METHODS) if local % 2 == 0 else list(reversed(METHODS))
            for method in methods:
                corrected = method == "pbpk_match_light"
                for cached in (False, True):
                    torch.cuda.reset_peak_memory_stats(device)
                    seconds, result = timed([row["query_index"]], corrected, cached)
                    allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                    if not cached:
                        peak[method] = max(peak[method], allocated)
                    key = "scores" if corrected else "base_scores"
                    actual = result[0]
                    expected = reference[key][0, row["query_index"]]
                    measures.append({**row, "method_id": method,
                        "mode": "cached_query_head" if cached else "query_encoding_and_head",
                        "seconds": seconds, "peak_gpu_gib": allocated,
                        "score_max_absolute_delta_from_saved": float(np.max(np.abs(actual - expected))),
                        "top1_agrees_with_saved": bool(np.argmax(actual) == np.argmax(expected)),
                        "top10_overlap_with_saved": len(set(np.argsort(-actual, kind="stable")[:10]) & set(np.argsort(-expected, kind="stable")[:10])) / 10})
            vectors = encode([row["query_index"]]).cpu().numpy()[0]
            frozen = data["light_query_embeddings"][0, row["query_index"]]
            checks.append({**row, "embedding_cosine_to_cache": float(np.dot(vectors, frozen) / np.linalg.norm(vectors) / np.linalg.norm(frozen))})
        for method in METHODS:
            for start in range(0, len(group), 8):
                batch = group[start:start + 8]
                seconds, _ = timed([r["query_index"] for r in batch], method == "pbpk_match_light", False)
                batch_rows.append({"method_id": method, "fold": fold, "requested_batch_size": 8,
                    "actual_batch_size": len(batch), "queries": len(batch), "seconds": seconds})
        print(f"Runtime fold {fold + 1}/5: ten held-out requests, both methods complete", flush=True)
        del model, error_head, state
        torch.cuda.empty_cache()

    runtime = []
    for method, name in METHODS.items():
        serial = [r["seconds"] for r in measures if r["method_id"] == method and r["mode"] == "query_encoding_and_head"]
        cached = [r["seconds"] for r in measures if r["method_id"] == method and r["mode"] == "cached_query_head"]
        batches = [r for r in batch_rows if r["method_id"] == method]
        runtime.append({"method_id": method, "method": name, "group": "full", "queries": len(serial),
            "p50_seconds": float(np.median(serial)), "p95_seconds": float(np.quantile(serial, .95)),
            "peak_gpu_gib": peak[method], "serial_queries_per_second": len(serial) / sum(serial),
            "cached_head_p50_seconds": float(np.median(cached)), "cached_head_p95_seconds": float(np.quantile(cached, .95)),
            "batch_size": 8, "batch_queries": sum(r["queries"] for r in batches),
            "batch_queries_per_second": sum(r["queries"] for r in batches) / sum(r["seconds"] for r in batches)})
    conditions = {"device": device, "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "encoder_dtype": "bfloat16", "encoder_attention": "sdpa", "maximum_input_tokens": 4096,
        "truncation": False, "encoder_revision": registry["revision"],
        "requests": 50, "per_outer_fold": 10, "candidate_count": 280,
        "warmup_queries_per_fold_per_method": 2, "repetitions_per_request_per_mode": 1,
        "weights_and_candidate_vectors_resident": True,
        "timed": "Tokenize raw request, GPU encode and normalize, condition standardization/canonicalization, 280-candidate head scoring, CPU copy probabilities/scores and stable sorting",
        "excluded": "Model loading, candidate embedding, retrieval/database access, feature extraction; raw condition features are already materialized",
        "cached_head_mode": "Same scoring path but request vectors reused; encoder remains resident, so its GPU allocation is retained",
        "numerical_scope": "Historical batch-encoded BF16 query vectors and freshly encoded single queries can differ numerically; this runtime audit records score deltas/top-rank agreement and does not claim encoder-level bitwise batch invariance",
        "batching": "Up to eight requests encoded jointly; all-candidate head evaluated per request; two-query tail retained per fold",
        "exclusive_gpu_usage_verified": False,
        "new_predictions_change_quality_results": False}
    return runtime, measures, batch_rows, checks, conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stability-only", action="store_true")
    args = parser.parse_args()
    from run_pbpk_match_light import prepare
    data = prepare()
    destination = OUT / "engineering"
    destination.mkdir(parents=True, exist_ok=True)
    receipt = read(OUT / "predictions/seed17.json")
    if receipt["state"] != "complete" or receipt["score_sha256"] != sha(OUT / "predictions/seed17.npz"):
        raise ValueError("Saved complete prediction receipt differs")
    sources = [OUT / "protocol.json", OUT / "predictions/seed17.npz", OUT / "predictions/seed17.json", REGISTRY,
               OUT / "cache/manifest.json", Path(__file__)]
    for fold in range(5):
        sources += [OUT / f"runs/seed17/fold{fold}/outer/heads.pt", OUT / f"runs/seed17/fold{fold}/outer/metadata.json",
                    OUT / f"runs/seed17/fold{fold}/selection.json", OUT / f"runs/seed17/fold{fold}/outer/integrity.json"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    rows = []
    with np.load(OUT / "predictions/seed17.npz", allow_pickle=False) as saved:
        for method, key in (("pbpk_match_light_document", "base_scores"), ("pbpk_match_light", "scores")):
            rows.append(stability(data, saved[key], method))
    csv_write(destination / "stability.csv", rows)
    sample = sample_requests(data)
    write(destination / "runtime_sample.json", {"selection": "Ten equally spaced query IDs within each outer test fold, original stability requests, chosen without labels or timings", "requests": sample})
    result = {"schema": "pbpk-match-light-paper-engineering-v1", "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(), "stability": rows,
        "metric_definitions": {"top1": "Equality of the stable-sort first candidate; exact ties separately reported",
            "top10": "Intersection of fixed-size stable-sort top-ten sets divided by ten, not Jaccard",
            "denominators": "772 changed sentence-order inputs; 304 changed evidence-removal inputs; identical all-candidate head execution"},
        "generation": [{"method_id": m, "method": name, "state": "not_applicable",
                        "reason": "Numeric score model, no autoregressive generation or JSON parsing", "strict_valid": None, "recovered": None, "fallback": None} for m, name in METHODS.items()],
        "inference_integrity": [read(OUT / f"runs/seed17/fold{f}/outer/integrity.json") for f in range(5)],
        "source_files": hashes}
    if not args.stability_only:
        runtime, measurements, batches, checks, conditions = benchmark(data, sample, args.device)
        result.update(runtime=runtime, runtime_conditions=conditions, embedding_checks=checks)
        csv_write(destination / "runtime.csv", runtime)
        csv_write(destination / "runtime_measurements.csv", measurements)
        csv_write(destination / "runtime_batches.csv", batches)
    else:
        result["runtime_state"] = "not_run"
    if any(sha(ROOT / p) != digest for p, digest in hashes.items()):
        raise ValueError("An immutable model/source changed during engineering measurement")
    write(destination / "summary.json", result)
    print(json.dumps({"stability": rows, "runtime": result.get("runtime"), "destination": str(destination)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
