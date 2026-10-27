"""Small supervised adapters for the frozen PBPK input collection.

This module never loads labels, chooses protocol partitions, or writes files.
Callers must supply explicit development indices. Unknown grades do not become
negative examples; unprocessed applicability (-1) is not the semantic U class.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def _indices(data: dict, indices: list[int]) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or not len(values) or len(set(values.tolist())) != len(values):
        raise ValueError("Query indices must be a nonempty, unique one-dimensional sequence")
    if np.any(values < 0) or np.any(values >= len(data["task_ids"])):
        raise ValueError("Query index outside the supplied input collection")
    return values


def _partitions(data: dict, train: list[int], validation: list[int], *, allow_empty_validation: bool = False) -> tuple[np.ndarray, np.ndarray]:
    train_i = _indices(data, train)
    val_i = np.asarray([], dtype=np.int64) if allow_empty_validation and not len(validation) else _indices(data, validation)
    if set(train_i.tolist()) & set(val_i.tolist()):
        raise ValueError("Training and validation queries must be disjoint")
    # Deliberately inspect labels only for the explicitly supplied development pool.
    for values in (train_i, val_i):
        if not np.isin(np.asarray(data["grades"])[values], [-1, 0, 1, 2, 3]).all():
            raise ValueError("Grades must be -1 (unknown) or 0..3")
        if not np.isin(np.asarray(data["applicability"])[values], [-1, 0, 1, 2]).all():
            raise ValueError("Applicability must be -1 (unprocessed), Y=0, N=1 or U=2")
    return train_i, val_i


def _compound(data: dict, index: int) -> str:
    record = data["query_records"][index]
    return str(record.get("compound_id") or record.get("primary_compound_id") or record["query_id"])


def _query_weights(data: dict, indices: np.ndarray) -> dict[int, float]:
    counts = Counter(_compound(data, int(i)) for i in indices)
    return {int(i): len(indices) / (len(counts) * counts[_compound(data, int(i))]) for i in indices}


def _candidate_weights(data: dict, query_index: int) -> np.ndarray:
    """Equal family mass, then equal identical visible-metadata group mass.

    ``candidate_packet_groups`` is a legacy input key, not judge-source packets.
    """
    records = data["candidate_records"]
    groups = data.get("candidate_packet_groups")
    if groups is not None and np.asarray(groups).ndim == 2:
        groups = groups[query_index]
    families = [str(c.get("family_id") or c.get("family_name") or c["candidate_id"]) for c in records]
    packets = [str(groups[i]) if groups is not None else str(c["candidate_id"]) for i, c in enumerate(records)]
    counts = Counter(zip(families, packets))
    family_packets: dict[str, set[str]] = defaultdict(set)
    for family, packet in zip(families, packets):
        family_packets[family].add(packet)
    weights = np.asarray([1. / (len(family_packets[f]) * counts[(f, p)]) for f, p in zip(families, packets)], dtype=np.float32)
    return weights / weights.mean()


def _class_weights(data: dict, indices: np.ndarray) -> np.ndarray:
    query_weights = _query_weights(data, indices)
    mass = np.zeros(3, dtype=np.float64)
    for index in indices:
        labels = np.asarray(data["applicability"])[index]
        weights = _candidate_weights(data, int(index)) * query_weights[int(index)]
        for label in range(3):
            mass[label] += weights[labels == label].sum()
    present = mass > 0
    result = np.zeros(3, dtype=np.float32)
    if present.any():
        result[present] = mass.sum() / (present.sum() * mass[present])
    return result


def _ordinal_class_weights(data: dict, indices: np.ndarray, balance: str = "none") -> np.ndarray:
    """Estimate mild grade weights from this training fold only.

    Counts use the same compound/family/visible-metadata weights as training.
    For a present grade, clip 1/sqrt(frequency) at 10, then normalize its
    expectation under the observed training-grade distribution to one. Absent
    grades receive zero in balanced mode; unknown (-1) never contributes.
    """
    if balance not in {"none", "sqrt_inverse"}:
        raise ValueError("ordinal_balance must be 'none' or 'sqrt_inverse'")
    if balance == "none":
        return np.ones(4, dtype=np.float32)
    mass = np.zeros(4, dtype=np.float64)
    query_weights = _query_weights(data, indices)
    for index in indices:
        grades = np.asarray(data["grades"])[index]
        weights = _candidate_weights(data, int(index))
        known_mass = weights[(grades >= 0) & (grades <= 3)].sum()
        if known_mass == 0:
            continue
        # The ordinal objective averages known candidates within each query.
        # Match that normalization so heavily annotated queries do not dominate.
        weights = weights / known_mass * query_weights[int(index)]
        for grade in range(4):
            mass[grade] += weights[grades == grade].sum()
    result = np.zeros(4, dtype=np.float64)
    present = mass > 0
    if present.any():
        frequency = mass / mass.sum()
        result[present] = np.minimum(10., 1. / np.sqrt(frequency[present]))
        result /= np.dot(frequency, result)
    return result.astype(np.float32)


def _variant(data: dict, variant: str) -> int:
    try:
        return list(data["variant_names"]).index(variant)
    except ValueError as exc:
        raise ValueError(f"Unknown input variant: {variant}") from exc


def _pair_inputs(data: dict, query_index: int, candidates: np.ndarray, variant: int, device: str):
    import torch

    count = len(candidates)
    def tensor(value, dtype):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    return (
        tensor(data["q_embeddings"][variant, query_index], torch.float32).unsqueeze(0).expand(count, -1, -1),
        tensor(data["q_types"][variant, query_index], torch.long).unsqueeze(0).expand(count, -1),
        tensor(data["q_mask"][variant, query_index], torch.bool).unsqueeze(0).expand(count, -1),
        tensor(data["c_embeddings"][candidates], torch.float32),
        tensor(data["c_types"][candidates], torch.long),
        tensor(data["c_mask"][candidates], torch.bool),
        tensor(data["pair_features"][variant, query_index, candidates], torch.float32),
        torch.full((count,), int(data["task_ids"][query_index]), dtype=torch.long, device=device),
    )


def _forward(model, data: dict, query_index: int, candidates: np.ndarray, variant: int, device: str, chunk_size: int):
    import torch

    scores, logits = [], []
    for start in range(0, len(candidates), chunk_size):
        output = model(*_pair_inputs(data, query_index, candidates[start:start + chunk_size], variant, device))
        if isinstance(output, dict):
            score, logit = output["rank_score"], output["app_logits"]
        else:
            score, logit = output
        scores.append(score.reshape(-1))
        logits.append(logit)
    return torch.cat(scores), torch.cat(logits)


def predict_neural(model, data: dict, query_indices: list[int], variant: str = "original", device: str = "cpu", chunk_size: int = 128):
    """Predict all candidates without labels or artificial numerical tie breaks.

    Exactly equal candidate tensors and per-query pair features share one
    forward pass. Fingerprints only shortlist possible equals: every array is
    compared before reuse. Neither family IDs nor metadata hashes establish
    equivalence, and distinct inputs' scores are never rounded or averaged.
    """
    import hashlib
    import torch

    indices = _indices(data, query_indices)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    variant_index = _variant(data, variant)
    count = len(data["candidate_records"])
    if not count:
        raise ValueError("Neural prediction requires at least one candidate")
    candidate_arrays = tuple(np.asarray(data[key]) for key in ("c_embeddings", "c_types", "c_mask"))

    def fingerprint(array):
        contiguous = np.ascontiguousarray(array)
        return contiguous.dtype.str, contiguous.shape, hashlib.sha256(contiguous.tobytes()).digest()

    # This is deliberately a local cache: caller mutations in another call must
    # be rechecked. NaNs do not compare equal, so they cannot create false reuse.
    signatures, representatives = {}, []
    candidate_groups = np.empty(count, dtype=np.int64)
    for candidate in range(count):
        signature = tuple(fingerprint(array[candidate]) for array in candidate_arrays)
        possible = signatures.setdefault(signature, [])
        group = next((g for g in possible if all(
            np.array_equal(array[candidate], array[representatives[g]]) for array in candidate_arrays
        )), None)
        if group is None:
            group = len(representatives)
            representatives.append(candidate)
            possible.append(group)
        candidate_groups[candidate] = group
    was_training = model.training
    model.to(device).eval()
    scores, probabilities = [], []
    try:
        with torch.inference_mode():
            for index in indices:
                features = np.asarray(data["pair_features"])[variant_index, int(index)]
                pair_signatures, unique = {}, []
                inverse = np.empty(count, dtype=np.int64)
                for candidate in range(count):
                    signature = (int(candidate_groups[candidate]), fingerprint(features[candidate]))
                    possible = pair_signatures.setdefault(signature, [])
                    position = next((p for p in possible if np.array_equal(
                        features[candidate], features[unique[p]]
                    )), None)
                    if position is None:
                        position = len(unique)
                        unique.append(candidate)
                        possible.append(position)
                    inverse[candidate] = position
                score, logits = _forward(model, data, int(index), np.asarray(unique, dtype=np.int64), variant_index, device, chunk_size)
                scores.append(score.float().cpu().numpy()[inverse])
                probability = logits.softmax(-1).double().cpu().numpy()
                probability = probability / probability.sum(-1, keepdims=True)
                probabilities.append(probability[inverse])
    finally:
        model.train(was_training)
    result = np.stack(scores), np.stack(probabilities)
    if not all(np.isfinite(value).all() for value in result):
        raise FloatingPointError("Nonfinite neural prediction")
    return result


def _same_compound(data: dict, query_index: int, candidates: np.ndarray) -> np.ndarray:
    query = data["query_records"][query_index]
    required = {str(v).casefold() for v in query.get("compounds", [])}
    return np.asarray([bool(required & {str(v).casefold() for v in data["candidate_records"][i].get("compounds", [])}) for i in candidates])


def _sample_candidates(data: dict, query_index: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    grades = np.asarray(data["grades"])[query_index]
    applicability = np.asarray(data["applicability"])[query_index]
    eligible = np.flatnonzero((grades >= 0) | (applicability >= 0))
    if len(eligible) <= limit:
        return eligible
    same = _same_compound(data, query_index, eligible)
    # Keep positives and known same-drug hard negatives; a requested limit is
    # soft when retaining them requires more entries. No unknown is a rank negative.
    mandatory = eligible[(grades[eligible] >= 2) | (same & (grades[eligible] >= 0))]
    others = np.setdiff1d(eligible, mandatory)
    selected = rng.choice(others, min(len(others), max(0, limit - len(mandatory))), replace=False)
    return np.sort(np.r_[mandatory, selected])


def _query_loss(scores, logits, grades, applicability, candidate_weights, class_weights, same_compound, config, ordinal_weights=None):
    import torch
    import torch.nn.functional as functional

    zero = scores.sum() * 0. + logits.sum() * 0.
    known = grades >= 0
    rank_loss, ordinal_loss, app_loss = zero, zero, zero
    if known.any():
        s, g, w = scores[known], grades[known], candidate_weights[known]
        greater = g[:, None] > g[None, :]
        if greater.any():
            pair_weights = w[:, None] * w[None, :]
            hard = same_compound[known]
            pair_weights = pair_weights * (1. + float(config.get("same_compound_pair_bonus", 1.)) * (hard[:, None] & hard[None, :]))
            rank_loss = (functional.softplus(-(s[:, None] - s[None, :]))[greater] * pair_weights[greater]).sum() / pair_weights[greater].sum()
        thresholds = scores.new_tensor([-1., 0., 1.])
        targets = (g[:, None] > torch.arange(3, device=g.device)[None, :]).float()
        ordinal = functional.binary_cross_entropy_with_logits(s[:, None] - thresholds, targets, reduction="none").mean(-1)
        if ordinal_weights is not None:
            # Preserve the fold-derived reweighting. Dividing by each query's
            # sum of grade weights would cancel it for single-grade queries.
            ordinal = ordinal * ordinal_weights[g]
        ordinal_loss = (ordinal * w).sum() / w.sum()
    app_known = applicability >= 0
    if app_known.any():
        weights = candidate_weights[app_known]
        losses = functional.cross_entropy(logits[app_known], applicability[app_known], weight=class_weights, reduction="none")
        app_loss = (losses * weights).sum() / weights.sum()
    loss = (float(config.get("rank_weight", 1.)) * rank_loss
            + float(config.get("ordinal_weight", .3)) * ordinal_loss
            + float(config.get("applicability_weight", .5)) * app_loss)
    return loss, {"rank_loss": float(rank_loss.detach()), "ordinal_loss": float(ordinal_loss.detach()), "applicability_loss": float(app_loss.detach())}


def _known_ndcg(scores: np.ndarray, grades: np.ndarray, k: int = 10) -> float | None:
    from .evaluation import tie_aware_rank_metrics

    known = grades >= 0
    result = tie_aware_rank_metrics(scores[known], grades[known])
    return None if result is None else float(result[f"ndcg@{k}"])


def _validation_ordinal_loss(data: dict, indices: np.ndarray, scores: np.ndarray) -> float | None:
    """Unweighted known-grade ordinal loss, query then compound macro.

    Only the supplied validation rows are read. Unknown grades have no target
    and contribute neither a zero value nor a denominator count.
    """
    per_compound: dict[str, list[float]] = defaultdict(list)
    for position, index in enumerate(indices):
        grades = np.asarray(data["grades"])[index]
        known = grades >= 0
        if known.any():
            logits = scores[position, known, None] - np.asarray([-1., 0., 1.])
            targets = grades[known, None] > np.arange(3)
            loss = np.logaddexp(0., logits) - logits * targets
            per_compound[_compound(data, int(index))].append(float(loss.mean()))
    if not per_compound:
        return None
    return float(np.mean([np.mean(values) for values in per_compound.values()]))


def _checkpoint_choice(value: float, best_value: float, secondary: float | None, best_secondary: float | None,
                       *, min_delta: float = 1e-5, allow_secondary: bool = False) -> tuple[bool, str]:
    """Lexicographic selection; never trade a worse NDCG for a lower loss."""
    if value > best_value + min_delta:
        return True, "primary_improvement"
    primary_tied = np.isclose(value, best_value, rtol=0., atol=1e-12)
    if (allow_secondary and primary_tied and secondary is not None and best_secondary is not None
            and np.isfinite(secondary) and secondary < best_secondary - 1e-12):
        return True, "ordinal_loss_breaks_primary_tie"
    return False, "not_improved"


def _validation_score(data: dict, indices: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    per_compound: dict[str, list[float]] = defaultdict(list)
    for position, index in enumerate(indices):
        value = _known_ndcg(scores[position], np.asarray(data["grades"])[index])
        if value is not None:
            per_compound[_compound(data, int(index))].append(value)
    if per_compound:
        return float(np.mean([np.mean(values) for values in per_compound.values()])), "known_pool_compound_macro_ndcg@10"
    # A fold without a known positive has no ranking metric. Use an explicitly
    # named proper ordinal loss, never pretend its NDCG is zero or one.
    ordinal = _validation_ordinal_loss(data, indices, scores)
    if ordinal is not None:
        return -ordinal, "negative_known_ordinal_loss_no_positive_validation"
    raise ValueError("Validation partition contains no known ordinal labels")


def train_neural(data: dict, train_query_indices: list[int], validation_query_indices: list[int], model_kind: str = "full", seed: int = 17, device: str = "cpu", config: dict | None = None):
    import torch
    import torch.nn.functional as functional
    from .model import PBPKMatch, PairMLP
    from .hybrid import PBPKMatchHybrid

    config = dict(config or {})
    config.setdefault("ordinal_balance", "none")
    if config["ordinal_balance"] not in {"none", "sqrt_inverse"}:
        raise ValueError("ordinal_balance must be 'none' or 'sqrt_inverse'")
    config.setdefault("validation_tie_break", "none")
    if config["validation_tie_break"] not in {"none", "ordinal_loss"}:
        raise ValueError("validation_tie_break must be 'none' or 'ordinal_loss'")
    fixed_epochs = config.get("fixed_epochs")
    if fixed_epochs is not None and len(validation_query_indices):
        raise ValueError("fixed_epochs refit requires an empty validation partition")
    train_i, val_i = _partitions(data, train_query_indices, validation_query_indices, allow_empty_validation=fixed_epochs is not None)
    if model_kind not in {"full", "mlp", "no_type", "no_null", "hybrid"}:
        raise ValueError(f"Unknown model kind: {model_kind}")
    epochs = int(fixed_epochs if fixed_epochs is not None else config.get("epochs", 60))
    chunk_size = int(config.get("chunk_size", 128))
    batch_queries = int(config.get("batch_queries", 4))
    if not 1 <= epochs <= 60 or chunk_size < 1 or batch_queries < 1:
        raise ValueError("Require 1..60 epochs and positive batch/chunk sizes")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kwargs = {"embedding_dim": int(data["q_embeddings"].shape[-1]), "pair_feature_dim": int(data["pair_features"].shape[-1]),
              "num_types": int(data.get("num_types", max(np.max(data["q_types"]), np.max(data["c_types"])) + 1)),
              "num_tasks": int(data.get("num_tasks", np.max(data["task_ids"]) + 1)),
              "hidden_dim": int(config.get("hidden_dim", 128)), "dropout": float(config.get("dropout", .1))}
    if model_kind == 'hybrid':
        model = PBPKMatchHybrid(**kwargs)
    elif model_kind == 'mlp':
        model = PairMLP(**kwargs)
    else:
        model = PBPKMatch(**kwargs, ablation=model_kind)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", .001)), weight_decay=float(config.get("weight_decay", .001)))
    class_weights_array = _class_weights(data, train_i)
    class_weights = torch.as_tensor(class_weights_array, device=device)
    ordinal_weights_array = _ordinal_class_weights(data, train_i, config["ordinal_balance"])
    ordinal_weights = torch.as_tensor(ordinal_weights_array, device=device)
    query_weights = _query_weights(data, train_i)
    original = _variant(data, "original")
    reordered = _variant(data, "reordered") if "reordered" in data["variant_names"] else None
    consistency_weight = float(config.get("consistency_weight", .05))
    best_state, best_value, best_epoch, stale = None, -float("inf"), 0, 0
    best_secondary = None
    history = {"model_kind": model_kind, "seed": seed, "train_query_indices": train_i.tolist(), "validation_query_indices": val_i.tolist(),
               "class_weights_train_only": class_weights_array.tolist(), "epochs": [], "config": config,
               "ordinal_balance": config["ordinal_balance"], "ordinal_weights_train_only": ordinal_weights_array.tolist(),
               "validation_tie_break": config["validation_tie_break"],
               "secondary_selector": "known_grade_compound_macro_ordinal_loss" if config["validation_tie_break"] == "ordinal_loss" else None,
               "secondary_selector_semantics": "Only when primary NDCG ties within absolute 1e-12 (rtol=0); minimize validation ordinal loss; no metric mixing; disabled when ordinal loss is already primary",
               "ordinal_weight_semantics": "sqrt inverse training-fold weighted grade frequency; clip at 10 then normalize expected weight to one; unknown excluded" if config["ordinal_balance"] == "sqrt_inverse" else "unweighted ordinal classes",
               "init_args": {**kwargs, **({"ablation": model_kind} if model_kind in {'full','no_type','no_null'} else {})},
               "fitting_mode": "fixed_epochs_refit_no_validation" if fixed_epochs is not None else "validation_early_stopping",
               "training_variants": ["original"] + (["canonical_reordered_dropout_consistency"] if reordered is not None and consistency_weight else []),
               "consistency_semantics": "dropout consistency on two passes of canonicalized reordered inputs; does not establish free-paraphrase robustness",
               "candidate_weighting": "equal family, then identical visible-metadata group, then version mass", "query_weighting": "equal compound mass"}
    for epoch in range(1, epochs + 1):
        model.train()
        ordered = rng.permutation(train_i)
        losses, components = [], defaultdict(list)
        for start in range(0, len(ordered), batch_queries):
            batch = ordered[start:start + batch_queries]
            # Global query weights have mean 1. Dividing by the local sum of
            # weights would cancel compound balancing for one-query batches.
            batch_weight = len(batch)
            optimizer.zero_grad(set_to_none=True)
            processed = False
            for index in batch:
                index = int(index)
                selected = _sample_candidates(data, index, int(config.get("max_candidates_per_query", len(data["candidate_records"]))), rng)
                if not len(selected):
                    continue
                processed = True
                scores, logits = _forward(model, data, index, selected, original, device, chunk_size)
                grades = torch.as_tensor(np.asarray(data["grades"])[index, selected], dtype=torch.long, device=device)
                app = torch.as_tensor(np.asarray(data["applicability"])[index, selected], dtype=torch.long, device=device)
                weights = torch.as_tensor(_candidate_weights(data, index)[selected], device=device)
                same = torch.as_tensor(_same_compound(data, index, selected), dtype=torch.bool, device=device)
                loss, pieces = _query_loss(scores, logits, grades, app, weights, class_weights, same, config, ordinal_weights=ordinal_weights)
                if reordered is not None and consistency_weight > 0:
                    other_scores, other_logits = _forward(model, data, index, selected, reordered, device, chunk_size)
                    consistency = functional.mse_loss(scores, other_scores) + .5 * (
                        functional.kl_div(logits.log_softmax(-1), other_logits.softmax(-1), reduction="batchmean")
                        + functional.kl_div(other_logits.log_softmax(-1), logits.softmax(-1), reduction="batchmean"))
                    loss = loss + consistency_weight * consistency
                    pieces["dropout_consistency_loss"] = float(consistency.detach())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, query {index}")
                (loss * query_weights[index] / batch_weight).backward()
                losses.append(float(loss.detach()))
                for key, value in pieces.items():
                    components[key].append(value)
            if processed:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.)))
                optimizer.step()
        if not losses:
            raise ValueError("Training partition contains no usable labels")
        if fixed_epochs is None:
            validation_scores, _ = predict_neural(model, data, val_i.tolist(), device=device, chunk_size=chunk_size)
            value, metric = _validation_score(data, val_i, validation_scores)
            allow_secondary = config["validation_tie_break"] == "ordinal_loss" and metric == "known_pool_compound_macro_ndcg@10"
            secondary = _validation_ordinal_loss(data, val_i, validation_scores) if allow_secondary else None
            improved, selection_reason = _checkpoint_choice(value, best_value, secondary, best_secondary,
                                                            min_delta=float(config.get("min_delta", 1e-5)), allow_secondary=allow_secondary)
        else:
            value, metric = None, "fixed_epochs_last_checkpoint_no_validation"
            secondary, improved, selection_reason = None, True, "fixed_epochs_last_checkpoint"
        history["epochs"].append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_value": value,
                                  "validation_secondary_ordinal_loss": secondary, "checkpoint_selected": improved,
                                  "checkpoint_selection_reason": selection_reason,
                                  **{key: float(np.mean(values)) for key, values in components.items()}})
        if improved:
            best_value, best_epoch, stale = value, epoch, 0
            best_secondary = secondary
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if fixed_epochs is None and stale >= int(config.get("patience", 8)):
            break
    model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_validation_value=best_value, selection_metric=metric,
                   best_validation_secondary_ordinal_loss=best_secondary,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   stopped_early=len(history["epochs"]) < epochs)
    return model, history


def _tree_dataset(data: dict, indices: np.ndarray, *, require_positive: bool = False):
    features, labels, qids, group_weights = [], [], [], []
    query_weights = _query_weights(data, indices)
    for index in indices:
        grade = np.asarray(data["grades"])[index]
        candidates = np.flatnonzero(grade >= 0)
        if not len(candidates) or (require_positive and not np.any(grade[candidates] >= 2)):
            continue
        # XGBRanker accepts group weights, not per-candidate weights. Remove
        # identical packet/family/grade duplicates instead of silently passing
        # row weights with the wrong semantics. Conflicting labels stay separate.
        groups = data.get("candidate_packet_groups")
        if groups is not None:
            packets = groups[index] if np.asarray(groups).ndim == 2 else groups
            kept, seen = [], set()
            for candidate in candidates:
                record = data["candidate_records"][candidate]
                key = (str(record.get("family_id") or record["candidate_id"]), str(packets[candidate]), int(grade[candidate]))
                if key not in seen:
                    kept.append(candidate)
                    seen.add(key)
            candidates = np.asarray(kept)
        features.extend(data["pair_features"][_variant(data, "original"), index, candidates])
        labels.extend(grade[candidates])
        qids.extend([len(group_weights)] * len(candidates))
        group_weights.append(query_weights[int(index)])
    if not features:
        raise ValueError("No known graded candidates in tree training/validation partition")
    return np.asarray(features, dtype=np.float32), np.asarray(labels), np.asarray(qids), np.asarray(group_weights)


def train_tree(data: dict, train_query_indices: list[int], validation_query_indices: list[int], seed: int = 17, device: str = "cpu", config: dict | None = None):
    """LambdaMART on pair features plus a train-only logistic Y/N/U head.

    Returned probabilities are uncalibrated; outer protocol calibration belongs
    to the caller. Neither classifier nor feature scaler fits validation labels.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRanker

    config = dict(config or {})
    fixed_estimators = config.get("fixed_n_estimators")
    if fixed_estimators is not None and (len(validation_query_indices) or int(fixed_estimators) < 1):
        raise ValueError("fixed_n_estimators refit requires positive estimator count and empty validation partition")
    train_i, val_i = _partitions(data, train_query_indices, validation_query_indices, allow_empty_validation=fixed_estimators is not None)
    x, y, qids, weights = _tree_dataset(data, train_i)
    fit_kwargs, early_kwargs = {}, {}
    if fixed_estimators is None:
        # XGBoost gives no-positive groups an NDCG of 1; omit such groups from
        # ranking validation when positive groups exist. If the complete fold
        # has no positive, use validation RMSE under an explicit metric name.
        has_positive = any(np.any(np.asarray(data["grades"])[i] >= 2) for i in val_i)
        vx, vy, vqids, vweights = _tree_dataset(data, val_i, require_positive=has_positive)
        fit_kwargs = {"eval_set": [(vx, vy)], "eval_qid": [vqids], "sample_weight_eval_set": [vweights]}
        early_kwargs = {"early_stopping_rounds": int(config.get("tree_patience", 20))}
    else:
        has_positive = True
    ranker = XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10", tree_method="hist", device=device,
                       n_estimators=int(fixed_estimators if fixed_estimators is not None else config.get("n_estimators", 300)), max_depth=int(config.get("max_depth", 3)),
                       learning_rate=float(config.get("tree_learning_rate", .05)), min_child_weight=3,
                       subsample=1., colsample_bytree=1., reg_lambda=5., random_state=seed,
                       n_jobs=int(config.get("n_jobs", 4)), **early_kwargs)
    if not has_positive:
        ranker.set_params(eval_metric="rmse")
        # XGB ranking sample weights are per group, whereas its RMSE metric
        # expects row weights. The explicit fallback uses unweighted heldout
        # RMSE and cannot be interpreted as a ranking-quality estimate.
        fit_kwargs.pop("sample_weight_eval_set", None)
    ranker.fit(x, y, qid=qids, sample_weight=weights, verbose=False, **fit_kwargs)
    cx, cy, cw = [], [], []
    query_weights, class_weights = _query_weights(data, train_i), _class_weights(data, train_i)
    for index in train_i:
        app = np.asarray(data["applicability"])[index]
        candidates = np.flatnonzero(app >= 0)
        cx.extend(data["pair_features"][_variant(data, "original"), index, candidates])
        cy.extend(app[candidates])
        cw.extend(_candidate_weights(data, int(index))[candidates] * query_weights[int(index)] * class_weights[app[candidates]])
    if not cy:
        raise ValueError("Training partition has no applicability labels")
    cx, cy, cw = np.asarray(cx), np.asarray(cy), np.asarray(cw)
    scaler = StandardScaler().fit(cx, sample_weight=cw)
    classes = np.unique(cy)
    classifier = None
    if len(classes) > 1:
        classifier = LogisticRegression(max_iter=1000, C=float(config.get("classifier_c", 1.)), random_state=seed)
        classifier.fit(scaler.transform(cx), cy, sample_weight=cw)
    model = {"ranker": ranker, "app_classifier": classifier, "app_scaler": scaler, "app_classes": classes,
             "method": "LambdaMART + pair-feature logistic applicability", "probability_semantics": "uncalibrated"}
    history = {"seed": seed, "train_query_indices": train_i.tolist(), "validation_query_indices": val_i.tolist(),
               "class_weights_train_only": class_weights.tolist(), "best_iteration": int(ranker.best_iteration) if fixed_estimators is None else int(fixed_estimators) - 1,
               "evaluation_history": ranker.evals_result() if fixed_estimators is None else {}, "rank_feature_count": int(x.shape[1]),
               "fitting_mode": "fixed_estimators_refit_no_validation" if fixed_estimators is not None else "validation_early_stopping",
               "selection_metric": "fixed_n_estimators" if fixed_estimators is not None else ("ndcg@10" if has_positive else "rmse_no_positive_validation"),
               "rank_train_pairs_after_packet_deduplication": int(len(y)), "app_train_pairs": int(len(cy)),
               "candidate_weighting": "rank: deduplicate identical family/visible-metadata-group/grade; app: equal family and identical visible-metadata group mass"}
    return model, history


def predict_tree(model: dict, data: dict, query_indices: list[int], variant: str = "original", **unused):
    indices = _indices(data, query_indices)
    features = data["pair_features"][_variant(data, variant), indices]
    shape = features.shape[:2]
    x = np.asarray(features).reshape(-1, features.shape[-1])
    scores = model["ranker"].predict(x).reshape(shape)
    probabilities = np.zeros((len(x), 3), dtype=np.float64)
    if model["app_classifier"] is None:
        probabilities[:, int(model["app_classes"][0])] = 1.
    else:
        probabilities[:, model["app_classes"].astype(int)] = model["app_classifier"].predict_proba(model["app_scaler"].transform(x))
    return scores, probabilities.reshape(*shape, 3)
