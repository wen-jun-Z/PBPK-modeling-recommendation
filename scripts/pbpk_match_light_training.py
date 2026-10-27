"""Train PBPK-Match's small heads over frozen, independently encoded text.

No language model is loaded here. The only required text inputs are
``light_query_embeddings[variant,query,dimension]`` and
``light_candidate_embeddings[candidate,dimension]``. Outcomes are read only
inside explicitly supplied fitting/early-stopping partitions.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pbpk_kb.recommendation.pbpk_match_v2.model import ErrorHead, document_loss
from pbpk_kb.recommendation.pbpk_match_v2.training import (
    _index, _variant, _weights, _same_drug, _app_class_weights, _error_rows,
    _query_endpoint, _validation_objective, train_error_head,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "pbpk-match-frozen-embedding-small-heads-v1"


class LightMatch(nn.Module):
    """Shared text projection and explicit request/candidate condition matching."""
    def __init__(self, embedding_size, feature_size, width=128, dropout=.1):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(embedding_size, width), nn.LayerNorm(width), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(4 * width + feature_size, width), nn.LayerNorm(width),
                                    nn.GELU(), nn.Dropout(dropout))
        self.rank = nn.Linear(width, 1)
        self.applicability = nn.Linear(width, 3)
        self.ordinal_location = nn.Linear(width, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.))
        self.threshold_steps = nn.Parameter(torch.zeros(2))

    def forward(self, query, candidates, features):
        q = self.projection(F.normalize(query.detach().float(), dim=-1))
        c = self.projection(F.normalize(candidates.detach().float(), dim=-1))
        if q.ndim == 1:
            q = q.expand_as(c)
        fused = self.fusion(torch.cat((q, c, (q - c).abs(), q * c, features.float()), -1))
        thresholds = torch.cat((self.threshold_start.reshape(1),
                                self.threshold_start + F.softplus(self.threshold_steps).cumsum(0)))
        return {"scores": self.rank(fused).squeeze(-1), "app_logits": self.applicability(fused),
                "ordinal_logits": self.ordinal_location(fused) - thresholds,
                "representation": torch.cat((fused, features.float()), -1)}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Checkpoint directory must remain within the project")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _inputs(data):
    q = np.asarray(data["light_query_embeddings"], dtype=np.float32)
    c = np.asarray(data["light_candidate_embeddings"], dtype=np.float32)
    features = np.asarray(data["pair_features_by_variant"], dtype=np.float32)
    expected = (len(data["variant_names"]), len(data["query_records"]), len(data["candidate_records"]))
    if q.ndim != 3 or q.shape[:2] != expected[:2] or c.ndim != 2 or c.shape != (expected[2], q.shape[2]):
        raise ValueError("Frozen embedding dimensions differ from request/catalogue identities")
    if features.ndim != 4 or features.shape[:3] != expected:
        raise ValueError("Condition features must be supplied separately for every perturbation")
    if not all(np.isfinite(x).all() for x in (q, c, features)):
        raise ValueError("Input arrays contain nonfinite values")
    return q, c, features


def _binding(data):
    """Outcome-free inference binding. IDs are checked but never model features."""
    return {"embedding_manifest": data.get("light_embedding_manifest", {}),
            "query_ids": [r["query_id"] for r in data["query_records"]],
            "candidate_ids": sorted(r["candidate_id"] for r in data["candidate_records"]),
            "feature_names": data.get("pair_feature_names", []), "variant_names": data["variant_names"]}


def _normalized(features, metadata):
    # An absent training feature keeps unit scale; it must not explode by 1e5
    # when first present in a held-out query. No test-derived moments are used.
    return (features - np.asarray(metadata["feature_mean"], np.float32)) / np.asarray(metadata["feature_std"], np.float32)


def _canonical(candidates, features):
    """Canonical unique numerical inputs, preserving genuine exact ties."""
    packed = np.concatenate((candidates, features), -1)
    lookup, positions = {}, []
    for i, row in enumerate(packed):
        key = row.tobytes()
        if key not in lookup:
            lookup[key] = i
        positions.append(key)
    keys = sorted(lookup)
    locations = {key: i for i, key in enumerate(keys)}
    chosen = np.asarray([lookup[key] for key in keys], np.int64)
    return chosen, np.asarray([locations[key] for key in positions], np.int64)


def _prediction(model, data, indices, metadata, device, variant=0, callback=None):
    q, c, features = _inputs(data)
    model.eval()
    values = defaultdict(list)
    begin = time.monotonic()
    # A fixed all-candidate pass makes caller batch-size changes immaterial.
    # Canonical dedup also removes catalogue-order numerical artifacts.
    with torch.inference_mode():
        for local, index in enumerate(indices):
            x = _normalized(features[variant, int(index)], metadata)
            chosen, restore = _canonical(c, x)
            output = model(torch.as_tensor(q[variant, int(index)], device=device),
                           torch.as_tensor(c[chosen], device=device), torch.as_tensor(x[chosen], device=device))
            for name, key in (("scores", "scores"), ("logits", "app_logits"),
                              ("ordinal_logits", "ordinal_logits"), ("representations", "representation")):
                values[name].append(output[key].cpu().numpy()[restore])
            if callback:
                callback({"stage": "document_predict", "completed_queries": local + 1,
                          "total_queries": len(indices), "query_index": int(index),
                          "elapsed_seconds": time.monotonic() - begin})
    if not len(indices):
        n, f = len(c), len(metadata["feature_mean"])
        return {"scores": np.empty((0, n), np.float32), "logits": np.empty((0, n, 3), np.float32),
                "ordinal_logits": np.empty((0, n, 3), np.float32),
                "representations": np.empty((0, n, metadata["width"] + f), np.float32)}
    return {key: np.asarray(rows, np.float32) for key, rows in values.items()}


def train_fold(data, train_indices, val_indices, output, *, method="full", seed=17, device="cpu",
               epochs=40, patience=5, batch_size=280, learning_rate=1e-3, fixed_epochs=False,
               pk_epochs=80, error_train_indices=None, callback=None, cache_directory=None,
               max_length=None, backbone_path=None):
    """Fit only small networks; validation is never included in error targets."""
    if method not in {"full", "light"}:
        raise ValueError("This run trains only the own lightweight model")
    if not 1 <= int(epochs) <= 40 or patience < 1:
        raise ValueError("Require 1..40 epochs and positive patience")
    if backbone_path is not None:
        raise ValueError("This trainer never loads a text backbone; supply frozen arrays")
    train, val = _index(data, train_indices), _index(data, val_indices, empty=True)
    error_train = train if error_train_indices is None else _index(data, error_train_indices, empty=True)
    if not set(error_train) <= set(train):
        raise ValueError("Error fitting indices must belong to the document training partition")
    if set(train) & set(val):
        raise ValueError("Training and validation indices overlap")
    groups = np.asarray(data["groups"])
    if set(groups[train]) & set(groups[val]):
        raise ValueError("Training and validation share grouped drug/study sources")
    if fixed_epochs and len(val):
        raise ValueError("Fixed-epoch refit requires empty validation")
    fixed_epochs = bool(fixed_epochs or not len(val))
    for indices in (train, val):
        if not np.isin(np.asarray(data["grades"])[indices], [-1, 0, 1, 2, 3]).all():
            raise ValueError("Invalid allowed-partition grade")
        if not np.isin(np.asarray(data["applicability"])[indices], [-1, 0, 1, 2]).all():
            raise ValueError("Invalid allowed-partition applicability")
    q, c, features = _inputs(data)
    directory = _directory(output)
    x = features[0, train]
    mean, std = x.mean((0, 1)), x.std((0, 1))
    std = np.where(std < 1e-6, 1., np.maximum(std, 1e-3)).astype(np.float32)
    metadata = {"version": VERSION, "method": "light", "seed": int(seed), "device": device,
                "embedding_size": q.shape[-1], "width": 128, "dropout": .1,
                "feature_mean": mean.tolist(), "feature_std": std.tolist(),
                "train_indices": train.tolist(), "validation_indices": val.tolist(),
                "error_train_indices": error_train.tolist(), "fixed_epochs": fixed_epochs,
                "epochs_requested": int(epochs), "patience": int(patience), "learning_rate": float(learning_rate),
                "pk_epochs": int(pk_epochs), "batch_size_requested": int(batch_size),
                "batching": "one complete query, all candidates; no negative subsampling",
                "input_binding": _binding(data), "trainable_text_encoder": False,
                "normalization": "training-only mean/std; constant features keep unit scale",
                "weight_initialization": "new small network; no earlier supervised checkpoint used"}
    allowed_labels = {"grades": np.asarray(data["grades"])[np.r_[train, val]].tolist(),
                      "applicability": np.asarray(data["applicability"])[np.r_[train, val]].tolist(),
                      "error_rows": _error_rows(data, error_train)}
    resume_signature = hashlib.sha256(_json({**metadata, "allowed_labels": allowed_labels}).encode()).hexdigest()
    if (directory / "metadata.json").exists():
        completed = json.loads((directory / "metadata.json").read_text())
        if completed.get("resume_signature") != resume_signature:
            raise ValueError("Completed checkpoint input, labels or configuration changed")
        if not (directory / "heads.pt").is_file():
            raise FileNotFoundError("Completed checkpoint has lost its learned weights")
        return {**completed, "checkpoint": str(directory), "reused_checkpoint": True}
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = LightMatch(q.shape[-1], features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.001)
    qweights, cweights = _weights(data, train)
    class_weights = _app_class_weights(data, train, qweights, cweights)
    history, best_state, best_value, best_epoch, stale, start_epoch = [], None, -float("inf"), 0, 0, 1
    resume_file = directory / "resume.pt"
    if resume_file.exists():
        resume = torch.load(resume_file, map_location="cpu", weights_only=True)
        if resume["signature"] != resume_signature:
            raise ValueError("Resume checkpoint input, labels or configuration changed")
        model.load_state_dict(resume["model"]); optimizer.load_state_dict(resume["optimizer"])
        history, best_state = resume["history"], resume["best_state"]
        best_value, best_epoch, stale, start_epoch = resume["best_value"], resume["best_epoch"], resume["stale"], resume["epoch"] + 1
        rng.bit_generator.state = resume["numpy_rng"]
        torch.set_rng_state(resume["torch_rng"])
        if str(device).startswith("cuda") and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng"], device=device)
    weights = torch.as_tensor(cweights, device=device)
    classes = torch.as_tensor(class_weights, device=device)
    candidate_tensor = torch.as_tensor(c, device=device)
    began = time.monotonic()
    for epoch in range(start_epoch, int(epochs) + 1):
        if not fixed_epochs and stale >= patience:
            break
        model.train(); losses = []
        for step, index in enumerate(rng.permutation(train)):
            index = int(index)
            grades = np.asarray(data["grades"])[index]
            apps = np.asarray(data["applicability"])[index]
            if not ((grades >= 0) | (apps >= 0)).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            result = model(torch.as_tensor(q[0, index], device=device), candidate_tensor,
                           torch.as_tensor(_normalized(features[0, index], metadata), device=device))
            loss, _ = document_loss(result, torch.as_tensor(grades, device=device, dtype=torch.long),
                                    torch.as_tensor(apps, device=device, dtype=torch.long), weights,
                                    torch.as_tensor(_same_drug(data, index), device=device), classes)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite own-model loss")
            (loss * qweights[index]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
            if callback:
                callback({"stage": "document_train", "epoch": epoch, "epoch_total": int(epochs),
                          "completed_queries": step + 1, "total_queries": len(train),
                          "elapsed_seconds": time.monotonic() - began})
        if not losses:
            raise ValueError("Training partition has no usable document annotations")
        if fixed_epochs:
            value, metric = None, "fixed_epochs_refit"
        else:
            prediction = _prediction(model, data, val, metadata, device)
            value, metric = _validation_objective(data, val, prediction, cweights, class_weights, device)
        improved = fixed_epochs or value is None or best_state is None or value > best_value + 1e-5
        if improved:
            best_value, best_epoch, stale = value if value is not None else -float("inf"), epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_value": value,
                        "selection_metric": metric, "checkpoint_selected": improved})
        _save(resume_file, {"signature": resume_signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "best_state": best_state, "best_value": best_value, "best_epoch": best_epoch, "stale": stale,
                           "history": history, "epoch": epoch, "numpy_rng": rng.bit_generator.state,
                           "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None})
        if callback:
            callback({"stage": "epoch_complete", "epoch": epoch, "epochs_requested": int(epochs),
                      "best_epoch": best_epoch, "validation_value": value, "elapsed_seconds": time.monotonic() - began})
    model.load_state_dict(best_state); model.eval()
    error_indices = sorted({r["query_index"] for r in _error_rows(data, error_train)})
    representations = _prediction(model, data, error_indices, metadata, device)["representations"]
    error_head, error_meta = train_error_head(data, error_train, dict(zip(error_indices, representations)),
                                             seed=seed, device=device, epochs=pk_epochs, min_studies=3)
    metadata.update(best_epoch=int(best_epoch), epochs_completed=len(history), history=history,
                    trainable_parameters=sum(p.numel() for p in model.parameters()),
                    class_weights_train_only=class_weights.tolist(), error_metadata=error_meta,
                    elapsed_seconds=time.monotonic() - began, resume_signature=resume_signature)
    _save(directory / "heads.pt", {"heads": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                   "error_head": {key: value.detach().cpu() for key, value in error_head.state_dict().items()} if error_head is not None else None,
                                   "error_metadata": error_meta})
    _write(directory / "metadata.json", metadata)
    resume_file.unlink(missing_ok=True)
    return {**metadata, "checkpoint": str(directory)}


def predict_fold(data, indices, checkpoint, *, variant="original", device="cpu", batch_size=280,
                 return_representations=False, error_weight=0., temperature=1., callback=None, use_cache=True):
    """Restore and predict using input arrays alone; no outcome table is accessed."""
    directory = Path(checkpoint).resolve()
    if not directory.is_relative_to(ROOT.resolve()):
        raise ValueError("Checkpoint directory must remain within the project")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["version"] != VERSION or metadata["input_binding"] != _binding(data):
        raise ValueError("Checkpoint belongs to different frozen inputs")
    if temperature <= 0 or error_weight < 0:
        raise ValueError("Require positive temperature and nonnegative error correction")
    indices = _index(data, indices, empty=True)
    state = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
    model = LightMatch(metadata["embedding_size"], len(metadata["feature_mean"]), metadata["width"], metadata["dropout"]).to(device)
    model.load_state_dict(state["heads"])
    result = _prediction(model, data, indices, metadata, device, _variant(data, variant), callback)
    logits = result["logits"] / temperature
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    error_meta, errors = state["error_metadata"], {}
    endpoints = error_meta.get("enabled_endpoints", [])
    if endpoints:
        head = ErrorHead(error_meta["input_size"], len(endpoints)).to(device)
        head.load_state_dict(state["error_head"]); head.eval()
        representations = torch.as_tensor(result["representations"], device=device)
        with torch.inference_mode():
            for e, endpoint in enumerate(endpoints):
                errors[endpoint] = head(representations, torch.full(representations.shape[:-1], e, device=device, dtype=torch.long)).cpu().numpy()
    selected, supported = np.zeros_like(result["scores"]), np.zeros(len(indices), bool)
    for local, index in enumerate(indices):
        requested = data.get("execution_requested_endpoints", {}).get(int(index), data.get("execution_requested_endpoints", {}).get(str(index)))
        endpoint = _query_endpoint(data["query_records"][int(index)], requested)
        if endpoint in errors:
            selected[local], supported[local] = errors[endpoint][local], True
    answer = {"scores": (result["scores"] - error_weight * selected).astype(np.float32),
              "base_scores": result["scores"], "probabilities": probabilities.astype(np.float32),
              "logits": result["logits"], "predicted_errors": errors, "selected_predicted_error": selected,
              "error_supported": supported,
              "metadata": {"method": "light", "variant": variant, "error_metadata": error_meta,
                           "error_weight": error_weight, "temperature": temperature,
                           "inference_batching": "canonical all-candidate pass independent of requested batch size"}}
    if return_representations:
        answer["representations"] = result["representations"]
    if not np.isfinite(answer["scores"]).all() or not np.isfinite(answer["probabilities"]).all():
        raise FloatingPointError("Nonfinite catalogue prediction")
    return answer
