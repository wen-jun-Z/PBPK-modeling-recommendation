"""Fold-isolated training and label-free, complete-catalogue inference.

Callers own nested partitions and correction-strength selection. Empty validation
means a fixed-epoch refit; no outer labels are consulted for stopping or fitting.
Frozen representations are input-only and may therefore be reused across folds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .model import DEFAULT_BACKBONE, ROOT, ErrorHead, MatchHeads, QwenMatchEncoder, document_loss

METHODS = {"full", "no_numeric", "frozen_text", "mlp", "lambdamart"}
CACHE_VERSION = "qwen-final-token-fp32-margin-canonical-v2"


def _index(data, indices, *, empty=False):
    values = np.asarray(indices, dtype=np.int64)
    n = len(data["query_records"])
    if values.ndim != 1 or (not empty and not len(values)) or len(set(values.tolist())) != len(values):
        raise ValueError("Indices must be unique and nonempty")
    if len(values) and (values.min() < 0 or values.max() >= n):
        raise ValueError("Query index outside data")
    return values


def _variant(data, variant):
    names = list(data.get("variant_names", ["original", "sentence_order", "missing_mechanism"]))
    return names.index(variant) if isinstance(variant, str) else int(variant)


def _texts(data, variant, method):
    key = ("no_numeric_query_texts" if "no_numeric_query_texts" in data else "query_texts_no_numeric") if method == "no_numeric" else "query_texts"
    if key not in data:
        raise ValueError("no_numeric requires explicitly redacted query_texts_no_numeric")
    values = data[key]
    if isinstance(values, dict):
        name = list(data.get("variant_names", values.keys()))[variant]
        return values[name]
    return values[variant]


def _documents(data, method):
    key = "no_numeric_document_texts" if method == "no_numeric" else "document_texts"
    if key not in data:
        raise ValueError("no_numeric requires explicitly redacted no_numeric_document_texts")
    return data[key]


def _features(data, index, variant, method):
    values = np.asarray(data.get("pair_features_by_variant", data["pair_features"]))
    result = np.array(values[variant, index] if values.ndim == 4 else values[index], dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Pair features must encode missingness explicitly, not with NaN")
    if method == "no_numeric":
        if "numeric_feature_indices" not in data:
            raise ValueError("no_numeric requires numeric_feature_indices")
        result[:, np.asarray(data["numeric_feature_indices"], dtype=np.int64)] = 0
    return result


def _normalization(data, indices, method):
    # Input statistics use training inputs only, including unlabeled execution queries.
    values = np.concatenate([_features(data, int(i), 0, method) for i in indices])
    return values.mean(0), np.maximum(values.std(0), 1e-5)


def _weights(data, indices):
    compounds = [str(data["query_records"][int(i)].get("compound_id") or
                     data["query_records"][int(i)].get("primary_compound_id") or int(i)) for i in indices]
    query_counts = Counter(compounds)
    qweights = {int(i): len(indices) / len(query_counts) / query_counts[c] for i, c in zip(indices, compounds)}
    families = [str(c.get("family_id") or c.get("family_name") or c["candidate_id"]) for c in data["candidate_records"]]
    counts = Counter(families)
    cweights = np.asarray([1 / counts[f] for f in families], dtype=np.float32)
    return qweights, cweights / cweights.mean()


def _same_drug(data, index):
    record = data["query_records"][index]
    compounds = record.get("compounds", [])
    if isinstance(compounds, str):
        compounds = [compounds]
    required = {str(c).casefold() for c in compounds}
    result = []
    for item in data["candidate_records"]:
        values = item.get("compounds", [])
        values = [values] if isinstance(values, str) else values
        result.append(bool(required.intersection(str(c).casefold() for c in values)))
    return np.asarray(result)


def _encoder(data, metadata, *, trainable=False, checkpoint=None):
    return QwenMatchEncoder(metadata["backbone_path"], device=metadata["device"], trainable=trainable,
                            gradient_checkpointing=metadata.get("gradient_checkpointing", True),
                            lora_layers=metadata.get("lora_layers", 8),
                            adapter_path=(Path(checkpoint) / "adapter") if checkpoint and (Path(checkpoint) / "adapter").exists() else None)


def _cache_key(data, index, variant, metadata):
    digest = hashlib.sha256(CACHE_VERSION.encode())
    for value in (metadata["backbone_path"], str(metadata["max_length"]),
                  _texts(data, variant, metadata["method"])[index], *_documents(data, metadata["method"])):
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little")); digest.update(encoded)
    return digest.hexdigest()


def encode_query(data, index, variant, encoder, metadata, *, cache_directory=None):
    """Exact token duplicates share a forward pass; no candidate-ID tie jitter."""
    cache = None
    if cache_directory is not None:
        cache_dir = Path(cache_directory)
        if not cache_dir.resolve().is_relative_to(ROOT):
            raise ValueError("Representation cache must belong to the project")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / (_cache_key(data, index, variant, metadata) + ".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                return saved["representation"].astype(np.float32), saved["base_score"].astype(np.float32)
    if encoder is None:
        raise FileNotFoundError(f"Missing frozen representation cache for query {index}")
    documents = _documents(data, metadata["method"])
    query = _texts(data, variant, metadata["method"])[index]
    tokens = encoder.tokenize([query] * len(documents), list(documents), metadata["max_length"])
    cpu_ids, cpu_mask = tokens["input_ids"].cpu().numpy(), tokens["attention_mask"].cpu().numpy()
    signatures, representatives = [], {}
    for i, (ids, mask) in enumerate(zip(cpu_ids, cpu_mask)):
        signature = ids[mask.astype(bool)].tobytes()
        signatures.append(signature)
        representatives.setdefault(signature, i)
    canonical = sorted(representatives)
    lookup = {signature: position for position, signature in enumerate(canonical)}
    unique = [representatives[signature] for signature in canonical]
    inverse = [lookup[signature] for signature in signatures]
    hidden, margin = [], []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(unique), metadata["batch_size"]):
            selected = unique[start:start + metadata["batch_size"]]
            h, m = encoder({k: v[selected] for k, v in tokens.items()})
            hidden.append(h.cpu().numpy()); margin.append(m.cpu().numpy())
    result = np.concatenate(hidden)[inverse], np.concatenate(margin)[inverse]
    if not all(np.isfinite(v).all() for v in result):
        raise FloatingPointError("Nonfinite encoder output")
    if cache is not None:
        temp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temp, representation=result[0].astype(np.float16), base_score=result[1])
        temp.replace(cache)
        # Use the identical stored precision on first use and cache reuse.
        result = result[0].astype(np.float16).astype(np.float32), result[1]
    return result


def _ndcg(grades, scores, k=10):
    known = np.asarray(grades) >= 0
    grades, scores = np.asarray(grades)[known], np.asarray(scores)[known]
    if not len(grades) or grades.max() <= 0:
        return None
    order = np.argsort(-scores, kind="stable")
    gains = np.power(2., grades) - 1
    ideal = np.sort(gains)[::-1][:k] @ (1 / np.log2(np.arange(min(k, len(gains))) + 2))
    dcg = 0.
    start = 0
    while start < min(k, len(order)):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        dcg += gains[order[start:end]].mean() * (1 / np.log2(np.arange(start, min(end, k)) + 2)).sum()
        start = end
    return float(dcg / ideal)


def _predict_neural(data, indices, encoder, heads, metadata, variant=0, cache_directory=None,
                    callback=None, stage="document_predict"):
    scores, logits, representations, ordinal = [], [], [], []
    device = metadata["device"]
    heads.eval()
    started = time.monotonic()
    with torch.inference_mode():
        for completed, index in enumerate(indices, start=1):
            h, margin = encode_query(data, int(index), variant, encoder, metadata, cache_directory=cache_directory)
            f = (_features(data, int(index), variant, metadata["method"]) - metadata["feature_mean"]) / metadata["feature_std"]
            # Canonical actual input order protects catalogue-permutation tests
            # from unrelated GEMM row grouping and preserves exact duplicate ties.
            signature = [h[j].tobytes() + f[j].tobytes() + np.asarray(margin[j]).tobytes() for j in range(len(h))]
            unique = {s: j for j, s in enumerate(signature)}
            canonical = sorted(unique)
            order = [unique[s] for s in canonical]
            lookup = {s: j for j, s in enumerate(canonical)}
            restore = [lookup[s] for s in signature]
            out = heads(torch.as_tensor(h[order], device=device), torch.as_tensor(f[order], device=device), torch.as_tensor(margin[order], device=device))
            scores.append(out["scores"].cpu().numpy()[restore])
            logits.append(out["app_logits"].cpu().numpy()[restore])
            ordinal.append(out["ordinal_logits"].cpu().numpy()[restore])
            representations.append(out["representation"].cpu().numpy()[restore])
            if callback:
                callback({"stage": stage, "query_index": int(index), "completed_queries": completed,
                          "total_queries": len(indices), "elapsed_seconds": time.monotonic() - started})
    c = len(data["candidate_records"])
    return {"scores": np.stack(scores) if scores else np.empty((0, c), np.float32),
            "logits": np.stack(logits) if logits else np.empty((0, c, 3), np.float32),
            "ordinal_logits": np.stack(ordinal) if ordinal else np.empty((0, c, 3), np.float32),
            "representations": np.stack(representations) if representations else np.empty((0, c, 128), np.float32)}


def _error_rows(data, allowed):
    allowed = set(int(i) for i in allowed)
    query_map = {str(r.get("query_id", r.get("scenario_id", i))): i for i, r in enumerate(data["query_records"])}
    candidate_map = {str(r["candidate_id"]): i for i, r in enumerate(data["candidate_records"])}
    result = []
    for row in data.get("error_records", []):
        qi = row.get("query_index", query_map.get(str(row.get("query_id", row.get("scenario_id", "")))))
        if qi is None or int(qi) not in allowed:
            continue  # Do not inspect labels outside the explicitly allowed training subset.
        ci = row.get("candidate_index", candidate_map.get(str(row.get("candidate_id", ""))))
        endpoint = str(row.get("endpoint_key", row.get("endpoint", "")))
        study = str(row.get("study_id", row.get("study_key", "")))
        error = row.get("absolute_log_error", row.get("abs_log_error"))
        if error is None and row.get("predicted") is not None and row.get("observed") is not None:
            if min(float(row["predicted"]), float(row["observed"])) <= 0:
                continue
            error = abs(math.log(float(row["predicted"]) / float(row["observed"])))
        if ci is None or not endpoint or not study or error is None:
            continue
        error = float(error)
        if not math.isfinite(error) or error < 0:
            raise ValueError("Execution target must be a finite nonnegative absolute log error")
        result.append({"query_index": int(qi), "candidate_index": int(ci), "endpoint": endpoint,
                       "study_id": study, "target": error})
    return result


def train_error_head(data, train_indices, representations, *, seed=17, device="cpu", epochs=80,
                     min_studies=3):
    """Fit only on eligible training studies; detached document states stay frozen.

    representations maps global query indices to [candidates, features]. Equal
    endpoint/study mass prevents dose repeats or many catalogue versions from
    creating an apparent sample-size increase.
    """
    rows = _error_rows(data, train_indices)
    studies = defaultdict(set)
    for r in rows:
        studies[r["endpoint"]].add(r["study_id"])
    endpoints = sorted(e for e, s in studies.items() if len(s) >= min_studies)
    meta = {"endpoint_studies": {e: sorted(s) for e, s in sorted(studies.items())},
            "enabled_endpoints": endpoints, "minimum_independent_studies": min_studies,
            "epochs": epochs, "training_rows": 0}
    rows = [r for r in rows if r["endpoint"] in endpoints]
    if not rows:
        return None, meta
    torch.manual_seed(seed)
    x = torch.as_tensor(np.stack([representations[r["query_index"]][r["candidate_index"]] for r in rows]), device=device)
    ids = torch.tensor([endpoints.index(r["endpoint"]) for r in rows], device=device)
    targets = torch.tensor([r["target"] for r in rows], device=device)
    counts = Counter((r["endpoint"], r["study_id"]) for r in rows)
    weights = torch.tensor([1 / counts[(r["endpoint"], r["study_id"])] / len(studies[r["endpoint"]]) for r in rows], device=device)
    weights /= weights.sum()
    model = ErrorHead(x.shape[-1], len(endpoints)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        # MAE on absolute-log-error matches the quantity aggregated into GMFE.
        loss = ((model(x, ids) - targets).abs() * weights).sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
    meta.update(training_rows=len(rows), training_studies=len({r["study_id"] for r in rows}),
                input_size=x.shape[-1], final_training_mae=float(loss.detach()))
    return model.eval(), meta


def _app_class_weights(data, indices, qweights, cweights):
    mass = np.zeros(3, np.float64)
    for index in indices:
        labels = np.asarray(data["applicability"])[index]
        for label in range(3):
            mass[label] += cweights[labels == label].sum() * qweights[int(index)]
    present = mass > 0
    result = np.zeros(3, np.float32)
    if present.any():
        frequency = mass / mass.sum()
        result[present] = np.minimum(10., 1 / np.sqrt(frequency[present]))
        result /= np.dot(frequency, result)
    return result


def full_query_backward(forward_chunk, candidates, grades, applicability, weights, same_drug,
                        *, batch_size, device, class_weights=None, query_weight=1.):
    """Exact full-query losses with bounded-memory gradient caching.

    The first train-mode pass stores scalar heads only. Its full-query loss
    computes all unequal-grade pairs. A second pass replays each chunk's CPU
    and CUDA RNG states, therefore reproducing encoder and head dropout exactly,
    and backpropagates the cached upstream gradients. No labels are sampled out.
    Call zero_grad before and optimizer.step after this function.
    """
    chunks = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    rng_states, outputs = [], []
    keys = ("scores", "app_logits", "ordinal_logits")
    use_cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for chunk in chunks:
            rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if use_cuda else None))
            out = forward_chunk(chunk)
            outputs.append({key: out[key].detach() for key in keys})
    final_cpu = torch.get_rng_state()
    final_cuda = torch.cuda.get_rng_state(device) if use_cuda else None
    leaves = {key: torch.cat([out[key] for out in outputs]).requires_grad_() for key in keys}
    loss, details = document_loss(leaves, grades, applicability, weights, same_drug, class_weights)
    loss = loss * query_weight
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite full-query objective")
    gradients = torch.autograd.grad(loss, tuple(leaves[key] for key in keys), allow_unused=True)
    cursor = 0
    for chunk, (cpu_state, cuda_state) in zip(chunks, rng_states):
        torch.set_rng_state(cpu_state)
        if use_cuda:
            torch.cuda.set_rng_state(cuda_state, device)
        out = forward_chunk(chunk)
        pairs = [(out[key], grad[cursor:cursor + len(chunk)]) for key, grad in zip(keys, gradients) if grad is not None]
        if pairs:
            torch.autograd.backward([p[0] for p in pairs], [p[1] for p in pairs])
        cursor += len(chunk)
    torch.set_rng_state(final_cpu)
    if use_cuda:
        torch.cuda.set_rng_state(final_cuda, device)
    return float(loss.detach()), details


def _validation_objective(data, indices, predictions, cweights, class_weights, device):
    ndcgs = [_ndcg(np.asarray(data["grades"])[int(i)], scores) for i, scores in zip(indices, predictions["scores"])]
    values = [v for v in ndcgs if v is not None]
    if values:
        return float(np.mean(values)), "known_positive_query_ndcg10"
    losses = []
    for local, index in enumerate(indices):
        if not ((np.asarray(data["grades"])[index] >= 0) | (np.asarray(data["applicability"])[index] >= 0)).any():
            continue
        out = {"scores": torch.tensor(predictions["scores"][local], device=device),
               "app_logits": torch.tensor(predictions["logits"][local], device=device),
               "ordinal_logits": torch.tensor(predictions["ordinal_logits"][local], device=device)}
        loss, _ = document_loss(out, torch.tensor(np.asarray(data["grades"])[index], device=device, dtype=torch.long),
                                torch.tensor(np.asarray(data["applicability"])[index], device=device, dtype=torch.long),
                                torch.tensor(cweights, device=device), class_weights=torch.tensor(class_weights, device=device))
        losses.append(float(loss))
    if losses:
        return -float(np.mean(losses)), "negative_masked_validation_loss_no_positive_query"
    return None, "no_validation_annotations_fixed_epoch_fallback"


def _checkpoint(output, heads, encoder, metadata, *, error_head=None, error_metadata=None):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if encoder is not None and metadata["method"] in {"full", "no_numeric"}:
        encoder.encoder.save_pretrained(directory / "adapter", safe_serialization=True)
    state = {"heads": heads.state_dict() if heads is not None else None,
             "error_head": error_head.state_dict() if error_head is not None else None,
             "error_metadata": error_metadata or {"enabled_endpoints": []}}
    temp = directory / "heads.pt.tmp"
    torch.save(state, temp); temp.replace(directory / "heads.pt")
    jsonable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metadata.items()}
    temp_json = directory / "metadata.json.tmp"
    temp_json.write_text(json.dumps(jsonable, ensure_ascii=False, indent=2) + "\n")
    temp_json.replace(directory / "metadata.json")
    return {**jsonable, "checkpoint": str(directory), "error_metadata": state["error_metadata"]}


def train_fold(data, train_indices, val_indices, output, *, method="full", seed=17, device="cpu",
               epochs=10, patience=3, batch_size=32, max_length=768, backbone_path=None,
               learning_rate=1e-4, pk_epochs=80, fixed_epochs=False, cache_directory=None,
               error_train_indices=None, callback=None, gradient_checkpointing=True, lora_layers=8,
               resume=True):
    """Train one explicit fold, optionally refitting for a fixed inner-selected epoch count.

    Validation is only for early stopping. Final probability calibration and
    correction alpha must be fitted by the caller on inner OOF predictions.
    """
    if method not in METHODS or epochs < 1 or batch_size < 2:
        raise ValueError("Unknown method or invalid training configuration")
    train, val = _index(data, train_indices), _index(data, val_indices, empty=True)
    if set(train) & set(val):
        raise ValueError("Train and validation indices overlap")
    if data.get("groups") is not None:
        groups = np.asarray(data["groups"])
        if set(groups[train]) & set(groups[val]):
            raise ValueError("Train and validation leakage through grouped drug/study sources")
    for indices in (train, val):
        if not np.isin(np.asarray(data["grades"])[indices], [-1, 0, 1, 2, 3]).all():
            raise ValueError("Invalid grades")
        if not np.isin(np.asarray(data["applicability"])[indices], [-1, 0, 1, 2]).all():
            raise ValueError("Invalid applicability classes")
    error_train = train if error_train_indices is None else _index(data, error_train_indices, empty=True)
    if set(error_train) & set(val):
        raise ValueError("PK training indices overlap validation")
    if data.get("groups") is not None and set(np.asarray(data["groups"])[error_train]) & set(np.asarray(data["groups"])[val]):
        raise ValueError("PK training groups overlap validation")
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(4)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    mean, std = _normalization(data, train, method)
    metadata = {"version": 2, "method": method, "seed": seed, "device": device,
                "backbone_path": str(Path(backbone_path or DEFAULT_BACKBONE).resolve()),
                "epochs_requested": epochs, "patience": patience, "batch_size": batch_size,
                "max_length": max_length, "learning_rate": learning_rate,
                "gradient_checkpointing": bool(gradient_checkpointing), "lora_layers": lora_layers,
                "ranking_loss": "full_query_gradient_cache_all_unequal_grade_pairs",
                "feature_mean": mean, "feature_std": std, "train_indices": train.tolist(),
                "validation_indices": val.tolist(), "error_train_indices": error_train.tolist(),
                "fixed_epochs": bool(fixed_epochs or not len(val)), "history": [],
                "class_order": ["Y", "N", "U"], "score_semantics": "continuous pretrained yes/no margin plus learned residual",
                "pretrained_margin_precision": "FP32 projection of final-state onto yes-minus-no vocabulary weight",
                "representation_cache_version": CACHE_VERSION,
                "token_count_scope": "document training and stopping validation, including both gradient-cache passes",
                "probability_calibration": "uncalibrated; caller must use inner OOF only"}
    if cache_directory is None:
        cache_directory = ROOT / "runtime/cache/pbpk_match_v2/frozen_representations"
    metadata["cache_directory"] = str(cache_directory)
    doc_train = np.asarray([int(i) for i in train if ((np.asarray(data["grades"])[i] >= 0) |
                                                   (np.asarray(data["applicability"])[i] >= 0)).any()])
    if not len(doc_train):
        raise ValueError("No annotated document training pairs")
    if method == "lambdamart":
        return _train_lambdamart(data, train, val, output, metadata, error_train, pk_epochs, callback)
    frozen = method in {"frozen_text", "mlp"}
    encoder = _encoder(data, metadata, trainable=not frozen)
    heads = MatchHeads(encoder.hidden_size, len(mean), residual=method != "mlp").to(device)
    metadata["hidden_size"] = encoder.hidden_size
    metadata["trainable_text_parameters"] = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    metadata["trainable_head_parameters"] = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in list(encoder.parameters()) + list(heads.parameters()) if p.requires_grad],
                                 lr=learning_rate, weight_decay=.01)
    query_weights, candidate_weights = _weights(data, doc_train)
    class_weights = _app_class_weights(data, doc_train, query_weights, candidate_weights)
    metadata["applicability_class_weights"] = class_weights.tolist()
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, stale = -math.inf, 0, None, 0
    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    progress_file = output / "training_state.pt"
    signature_data = {k: metadata[k] for k in ("method", "seed", "train_indices", "validation_indices", "max_length",
                                              "learning_rate", "batch_size", "lora_layers", "gradient_checkpointing",
                                              "representation_cache_version", "epochs_requested", "patience", "fixed_epochs")}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode())
    for indices in (train, val):
        for key in ("grades", "applicability"):
            signature.update(np.ascontiguousarray(np.asarray(data[key])[indices]).tobytes())
    signature = signature.hexdigest()
    start_epoch, epoch = 1, 0
    if resume and progress_file.exists():
        previous = torch.load(progress_file, map_location="cpu", weights_only=True)
        if previous["signature"] != signature:
            raise ValueError("Resume checkpoint belongs to different inputs or training configuration")
        heads.load_state_dict(previous["heads"])
        with torch.no_grad():
            for name, parameter in encoder.named_parameters():
                if name in previous["encoder"]:
                    parameter.copy_(previous["encoder"][name].to(device))
        optimizer.load_state_dict(previous["optimizer"])
        best, best_epoch, best_state, stale = previous["best"], previous["best_epoch"], previous["best_state"], previous["stale"]
        metadata["history"] = previous["history"]
        rng.bit_generator.state = previous["numpy_rng"]
        torch.set_rng_state(previous["torch_rng"])
        if device.startswith("cuda") and previous["cuda_rng"] is not None:
            torch.cuda.set_rng_state(previous["cuda_rng"], device)
        epoch = previous["epoch"]
        start_epoch = epochs + 1 if stale >= patience else epoch + 1
        started -= previous["elapsed_seconds"]
    for epoch in range(start_epoch, epochs + 1):
        encoder.train(not frozen); heads.train()
        losses, pair_count = [], 0
        for index in rng.permutation(doc_train):
            index = int(index)
            grades = np.asarray(data["grades"])[index]
            applicability = np.asarray(data["applicability"])[index]
            candidates = np.flatnonzero((grades >= 0) | (applicability >= 0))
            candidates = rng.permutation(candidates)
            features = (_features(data, index, 0, method) - mean) / std
            same = _same_drug(data, index)
            if frozen:
                h_all, m_all = encode_query(data, index, 0, encoder, metadata, cache_directory=cache_directory)
            def forward_chunk(chosen):
                if frozen:
                    h, margin = torch.tensor(h_all[chosen], device=device), torch.tensor(m_all[chosen], device=device)
                else:
                    query = _texts(data, 0, method)[index]
                    documents = _documents(data, method)
                    tokens = encoder.tokenize([query] * len(chosen), [documents[i] for i in chosen], max_length)
                    h, margin = encoder(tokens)
                return heads(h, torch.tensor(features[chosen], device=device), margin)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = full_query_backward(forward_chunk, candidates,
                                          torch.tensor(grades[candidates], device=device, dtype=torch.long),
                                          torch.tensor(applicability[candidates], device=device, dtype=torch.long),
                                          torch.tensor(candidate_weights[candidates], device=device),
                                          torch.tensor(same[candidates], device=device), batch_size=batch_size, device=device,
                                          class_weights=torch.tensor(class_weights, device=device), query_weight=query_weights[index])
            torch.nn.utils.clip_grad_norm_([p for p in list(heads.parameters()) + list(encoder.parameters()) if p.requires_grad], 1.)
            optimizer.step()
            losses.append(loss); pair_count += len(candidates)
            if callback:
                callback({"stage": "document_training", "epoch": epoch, "query_index": index,
                          "pairs_this_epoch": pair_count, "elapsed_seconds": time.monotonic() - started})
        value, selection_metric = None, "fixed_epochs"
        if len(val) and not fixed_epochs:
            predictions = _predict_neural(data, val, encoder, heads, metadata,
                                           cache_directory=cache_directory if frozen else None,
                                           callback=callback, stage="document_validation")
            value, selection_metric = _validation_objective(data, val, predictions, candidate_weights, class_weights, device)
        metadata["history"].append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                                    "training_pairs": pair_count, "validation_selection_value": value,
                                    "validation_ndcg10": value if selection_metric == "known_positive_query_ndcg10" else None,
                                    "selection_metric": selection_metric,
                                    "elapsed_seconds": time.monotonic() - started})
        if value is None or value > best + 1e-6:
            best, best_epoch, stale = value if value is not None else 0., epoch, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
                          {k: p.detach().cpu().clone() for k, p in encoder.named_parameters() if p.requires_grad})
        else:
            stale += 1
        progress = {"signature": signature, "epoch": epoch, "heads": heads.state_dict(),
                    "encoder": {k: p.detach().cpu().clone() for k, p in encoder.named_parameters() if p.requires_grad},
                    "optimizer": optimizer.state_dict(), "best": best, "best_epoch": best_epoch,
                    "best_state": best_state, "stale": stale, "history": metadata["history"],
                    "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
                    "elapsed_seconds": time.monotonic() - started}
        temporary = progress_file.with_suffix(".tmp.pt")
        torch.save(progress, temporary); temporary.replace(progress_file)
        if callback:
            callback({"stage": "document_epoch_complete", **metadata["history"][-1]})
        if value is not None and stale >= patience:
            break
    heads.load_state_dict(best_state[0])
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if name in best_state[1]:
                parameter.copy_(best_state[1][name].to(parameter.device))
    encoder.requires_grad_(False); encoder.trainable_text = False; heads.requires_grad_(False)
    metadata.update(best_epoch=best_epoch, epochs_completed=epoch, training_seconds=time.monotonic() - started,
                    tokenized_pairs=encoder.tokenized_pairs, truncated_pairs=encoder.truncations)
    error_indices = sorted({r["query_index"] for r in _error_rows(data, error_train)})
    error_predictions = _predict_neural(data, error_indices, encoder, heads, metadata,
                                        cache_directory=cache_directory if frozen else None, callback=callback)
    representations = dict(zip(error_indices, error_predictions["representations"]))
    error_head, error_meta = train_error_head(data, error_train, representations, seed=seed, device=device, epochs=pk_epochs)
    result = _checkpoint(output, heads, encoder, metadata, error_head=error_head, error_metadata=error_meta)
    progress_file.unlink(missing_ok=True)
    del encoder, heads, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _baseline_inputs(data, indices, encoder, metadata, variant=0, use_cache=True,
                     callback=None, stage="document_predict"):
    rows = []
    started = time.monotonic()
    for completed, index in enumerate(indices, start=1):
        h, margin = encode_query(data, int(index), variant, encoder, metadata,
                                cache_directory=metadata["cache_directory"] if use_cache else None)
        # Same frozen text state and structured inputs as the neural baseline.
        h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5)
        f = (_features(data, int(index), variant, metadata["method"]) - metadata["feature_mean"]) / metadata["feature_std"]
        rows.append(np.concatenate((h, f, margin[:, None]), -1))
        if callback:
            callback({"stage": stage, "query_index": int(index), "completed_queries": completed,
                      "total_queries": len(indices), "elapsed_seconds": time.monotonic() - started})
    return np.stack(rows)


def _train_lambdamart(data, train, val, output, metadata, error_train, pk_epochs, callback):
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    import joblib
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    encoder = _encoder(data, metadata)
    x = _baseline_inputs(data, train, encoder, metadata, callback=callback)
    rank_rows, labels, qid = [], [], []
    app_rows, app_labels, app_weights = [], [], []
    query_weights, candidate_weights = _weights(data, train)
    app_classes = _app_class_weights(data, train, query_weights, candidate_weights)
    for local, index in enumerate(train):
        g, a = np.asarray(data["grades"])[index], np.asarray(data["applicability"])[index]
        known = g >= 0
        rank_rows.extend(x[local, known]); labels.extend(g[known]); qid.extend([local] * int(known.sum()))
        known = a >= 0
        app_rows.extend(x[local, known]); app_labels.extend(a[known]); app_weights.extend(candidate_weights[known] * query_weights[int(index)] * app_classes[a[known]])
    model = xgb.XGBRanker(objective="rank:ndcg", n_estimators=300, max_depth=4, learning_rate=.05,
                          subsample=1., colsample_bytree=1., reg_lambda=1., tree_method="hist",
                          device=metadata["device"], n_jobs=4, random_state=metadata["seed"], lambdarank_pair_method="topk")
    unique_q = sorted(set(qid))
    model.fit(np.asarray(rank_rows), np.asarray(labels), qid=np.asarray(qid),
              sample_weight=np.asarray([query_weights[int(train[i])] for i in unique_q]))
    if len(set(app_labels)) >= 2:
        classifier = LogisticRegression(max_iter=1000, random_state=metadata["seed"])
        classifier.fit(np.asarray(app_rows), app_labels, sample_weight=app_weights)
        joblib.dump(classifier, directory / "applicability.joblib")
        metadata["constant_app_class"] = None
    else:
        metadata["constant_app_class"] = int(app_labels[0]) if app_labels else 2
    model.save_model(directory / "ranker.ubj")
    error_indices = sorted({r["query_index"] for r in _error_rows(data, error_train)})
    ex = _baseline_inputs(data, error_indices, encoder, metadata, callback=callback) if error_indices else []
    error_head, error_meta = train_error_head(data, error_train, dict(zip(error_indices, ex)),
                                            seed=metadata["seed"], device=metadata["device"], epochs=pk_epochs)
    metadata.update(hidden_size=encoder.hidden_size, best_epoch=1, epochs_completed=1,
                    estimator="XGBoost LambdaMART rank:ndcg", trainable_text_parameters=0,
                    score_semantics="LambdaMART continuous score")
    return _checkpoint(output, None, None, metadata, error_head=error_head, error_metadata=error_meta)


def _query_endpoint(record, requested=None):
    if requested:
        requested = list(requested)
        for preferred in ("Cmax_reference_vs_group", "Cmax", "Cmax_ss_reference_vs_group", "Cmax_ss"):
            if preferred in requested:
                return preferred
        peak = next((str(e) for e in requested if str(e).casefold().startswith("cmax")), None)
        if peak:
            return peak
        return str(requested[0])
    explicit = record.get("endpoint_key")
    if explicit:
        return str(explicit)
    endpoint = record.get("endpoint", record.get("endpoint_type"))
    if endpoint:
        return str(endpoint)
    endpoints = record.get("endpoints", record.get("features", {}).get("endpoints", []))
    if isinstance(endpoints, str):
        endpoints = [endpoints]
    if any(str(e).casefold() == "cmax" for e in endpoints):
        # Clinical execution labels retain their comparison-stratum suffix.
        # Core requests have only endpoint types: this fixed alias selects the
        # ordinary Cmax head without consulting whether this query has a result.
        return "Cmax_reference_vs_group"
    return str(endpoints[0]) if len(endpoints) == 1 else None


def predict_fold(data, indices, checkpoint, *, variant="original", device="cpu", batch_size=32,
                 return_representations=False, error_weight=0., temperature=1., callback=None, use_cache=True):
    """Inference reads public inputs only, never grades or execution outcomes.

    All endpoint risk outputs are provided separately. Corrections apply only
    when the request itself identifies one supported endpoint, never by looking
    up the available observation or execution record.
    """
    import joblib
    directory = Path(checkpoint)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata.update(device=device, batch_size=batch_size)
    for key in ("feature_mean", "feature_std"):
        metadata[key] = np.asarray(metadata[key], dtype=np.float32)
    indices = _index(data, indices, empty=True)
    variant_index = _variant(data, variant)
    if temperature <= 0 or error_weight < 0:
        raise ValueError("Temperature must be positive and correction nonnegative")
    state = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
    encoder = _encoder(data, metadata, checkpoint=directory)
    if metadata["method"] == "lambdamart":
        import xgboost as xgb
        ranker = xgb.XGBRanker(); ranker.load_model(directory / "ranker.ubj"); ranker.set_params(device=device, n_jobs=4)
        representations = _baseline_inputs(data, indices, encoder, metadata, variant_index, use_cache=use_cache,
                                            callback=callback)
        flat = representations.reshape(-1, representations.shape[-1])
        scores = ranker.predict(flat).reshape(len(indices), -1)
        probabilities = np.zeros((len(flat), 3), np.float32)
        if metadata["constant_app_class"] is not None:
            probabilities[:, metadata["constant_app_class"]] = 1
        else:
            classifier = joblib.load(directory / "applicability.joblib")
            probabilities[:, classifier.classes_.astype(int)] = classifier.predict_proba(flat)
        logits = np.log(np.maximum(probabilities.reshape(len(indices), -1, 3), 1e-8))
        result = {"scores": scores, "logits": logits, "representations": representations}
    else:
        heads = MatchHeads(metadata["hidden_size"], len(metadata["feature_mean"]), residual=metadata["method"] != "mlp").to(device)
        heads.load_state_dict(state["heads"])
        result = _predict_neural(data, indices, encoder, heads, metadata, variant_index,
                                  cache_directory=metadata["cache_directory"] if use_cache and metadata["method"] in {"frozen_text", "mlp"} else None,
                                  callback=callback)
    logits = result["logits"] / temperature
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities = exp / exp.sum(-1, keepdims=True)
    error_meta = state.get("error_metadata", {})
    endpoints = error_meta.get("enabled_endpoints", [])
    errors = {}
    if endpoints:
        error_head = ErrorHead(error_meta["input_size"], len(endpoints)).to(device)
        error_head.load_state_dict(state["error_head"]); error_head.eval()
        reps = torch.as_tensor(result["representations"], device=device)
        with torch.inference_mode():
            for e, endpoint in enumerate(endpoints):
                ids = torch.full(reps.shape[:-1], e, dtype=torch.long, device=device)
                errors[endpoint] = error_head(reps, ids).cpu().numpy()
    selected = np.zeros_like(result["scores"], dtype=np.float32)
    supported = np.zeros(len(indices), bool)
    for local, index in enumerate(indices):
        requested = data.get("execution_requested_endpoints", {}).get(int(index),
                      data.get("execution_requested_endpoints", {}).get(str(index)))
        endpoint = _query_endpoint(data["query_records"][int(index)], requested)
        if endpoint in errors:
            selected[local] = errors[endpoint][local]; supported[local] = True
    answer = {"scores": (result["scores"] - error_weight * selected).astype(np.float32),
              "base_scores": result["scores"].astype(np.float32), "probabilities": probabilities.astype(np.float32),
              "logits": result["logits"].astype(np.float32), "predicted_errors": errors,
              "selected_predicted_error": selected, "error_supported": supported,
              "metadata": {"method": metadata["method"], "variant": variant, "error_metadata": error_meta,
                           "error_weight": error_weight, "temperature": temperature}}
    if return_representations:
        answer["representations"] = result["representations"]
    if not np.isfinite(answer["scores"]).all() or not np.isfinite(answer["probabilities"]).all():
        raise FloatingPointError("Nonfinite complete-catalogue prediction")
    if callback:
        callback({"stage": "prediction_complete", "queries": len(indices), "candidates": len(data["candidate_records"])})
    return answer
