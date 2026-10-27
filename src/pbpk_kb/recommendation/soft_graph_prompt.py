from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus,
    RetrievedSubgraph,
    assemble_recommendation,
    canonical_json,
    parse_selection,
    sha256_file,
    validate_recommendation,
    validate_selection,
)
from pbpk_kb.recommendation.graph_prompt_dataset import (
    DATASET_VERSION,
    DEFAULT_GATE5_LLM_DTYPE,
    DEFAULT_GATE5_LLM_HIDDEN_SIZE,
    DEFAULT_GATE5_LLM_MODEL,
    DEFAULT_GATE5_LLM_REVISION,
    GAP_TOKEN_LOSS_WEIGHT,
    GATE5_ACCEPTANCE,
    MAX_SOFT_SUBGRAPHS,
    SOFT_INPUT_DIMENSION,
)
from pbpk_kb.recommendation.schemas import ModelingRequest


SOFT_PROMPT_VERSION = "pbpk-soft-graph-prompt-2.3.0"
PROJECTOR_HIDDEN_DIMENSION = 2048
SEEDS = (17, 29, 43)
PROMPT_VARIANTS = ("no_kg", "text_graph", "soft_graph", "hybrid")
TRAINING_VARIANTS = ("soft_graph", "hybrid")
SOFT_MARKER = "<PBPK_SOFT_GRAPH_TOKENS>"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_projector(
    *,
    input_dimension: int,
    hidden_dimension: int,
    output_dimension: int,
) -> Any:
    import torch

    class SoftGraphProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projector = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dimension),
                torch.nn.Linear(input_dimension, hidden_dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(hidden_dimension, output_dimension),
                torch.nn.LayerNorm(output_dimension),
            )
            self.global_attention = torch.nn.Linear(input_dimension, 1, bias=False)

        def forward(self, vectors: Any, mask: Any) -> Any:
            logits = self.global_attention(vectors).squeeze(-1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            attention = torch.softmax(logits, dim=-1)
            global_vector = (vectors * attention.unsqueeze(-1)).sum(dim=1)
            local = self.projector(vectors)
            global_token = self.projector(global_vector).unsqueeze(1)
            return torch.cat((global_token, local), dim=1), attention

    return SoftGraphProjector()


def _chat_prompt_ids(
    tokenizer: Any,
    *,
    system_instruction: str,
    user_prompt: str,
    with_soft_marker: bool,
    device: Any,
    maximum_tokens: int,
) -> Any:
    import torch

    system = system_instruction
    if with_soft_marker:
        system = system + "\n" + SOFT_MARKER
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )[0]
    if ids.numel() > maximum_tokens:
        head = maximum_tokens // 3
        tail = maximum_tokens - head
        ids = torch.cat((ids[:head], ids[-tail:]))
    return ids.to(device)


def _marker_ids(tokenizer: Any, device: Any) -> Any:
    return tokenizer(
        SOFT_MARKER,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].to(device)


def _find_subsequence(values: Any, needle: Any) -> int:
    limit = int(values.numel() - needle.numel() + 1)
    for index in range(max(0, limit)):
        if bool((values[index : index + needle.numel()] == needle).all()):
            return index
    raise RuntimeError("Soft graph marker is absent from the tokenized prompt.")


def _prompt_embeddings(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: Any,
    soft_tokens: Any | None,
) -> tuple[Any, Any]:
    import torch

    embeddings = model.get_input_embeddings()(prompt_ids)
    if soft_tokens is None:
        return embeddings.unsqueeze(0), torch.ones(
            (1, embeddings.shape[0]),
            dtype=torch.long,
            device=embeddings.device,
        )
    marker = _marker_ids(tokenizer, prompt_ids.device)
    start = _find_subsequence(prompt_ids, marker)
    before = embeddings[:start]
    after = embeddings[start + marker.numel() :]
    joined = torch.cat((before, soft_tokens, after), dim=0)
    return joined.unsqueeze(0), torch.ones(
        (1, joined.shape[0]),
        dtype=torch.long,
        device=joined.device,
    )


def _masked_soft_tokens(
    projector: Any,
    vectors: Any,
    mask: Any,
    *,
    dtype: Any,
) -> Any:
    import torch

    tokens, _ = projector(vectors.unsqueeze(0), mask.unsqueeze(0))
    local_mask = mask.unsqueeze(0).unsqueeze(-1)
    local = tokens[:, 1:, :] * local_mask
    return torch.cat((tokens[:, :1, :], local), dim=1)[0].to(dtype=dtype)


def _target_text_embeddings(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
) -> tuple[Any, Any]:
    import torch

    all_targets = []
    all_masks = []
    embedding = model.get_input_embeddings()
    for row in rows:
        values = []
        for text in row["soft_token_target_texts"][:MAX_SOFT_SUBGRAPHS]:
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).input_ids[0].to(device)
            with torch.no_grad():
                values.append(embedding(ids).float().mean(dim=0))
        padded = torch.zeros(
            (MAX_SOFT_SUBGRAPHS, model.config.hidden_size),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.zeros(
            (MAX_SOFT_SUBGRAPHS,),
            dtype=torch.bool,
            device=device,
        )
        for index, value in enumerate(values):
            padded[index] = value
            mask[index] = True
        all_targets.append(padded)
        all_masks.append(mask)
    return torch.stack(all_targets), torch.stack(all_masks)


def _alignment_loss(
    projected: Any,
    target: Any,
    mask: Any,
) -> Any:
    import torch

    local = projected[:, 1:, :]
    global_target = (
        target * mask.unsqueeze(-1)
    ).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    selected_projected = torch.cat(
        (projected[:, :1, :], local),
        dim=1,
    )
    selected_target = torch.cat(
        (global_target.unsqueeze(1), target),
        dim=1,
    )
    selected_mask = torch.cat(
        (
            torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
            mask,
        ),
        dim=1,
    )
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        selected_projected.float(),
        selected_target.float(),
        dim=-1,
    )
    mse = (
        torch.nn.functional.normalize(selected_projected.float(), dim=-1)
        - torch.nn.functional.normalize(selected_target.float(), dim=-1)
    ).pow(2).mean(dim=-1)
    return ((cosine + 0.10 * mse) * selected_mask).sum() / selected_mask.sum()


def _target_only_causal_loss(
    *,
    model: Any,
    inputs: Any,
    attention_mask: Any,
    target_ids: Any,
    target_weights: Any | None = None,
) -> tuple[Any, Any]:
    import torch

    prompt_length = int(inputs.shape[1] - target_ids.shape[1])
    positions = torch.arange(
        prompt_length - 1,
        int(inputs.shape[1]) - 1,
        dtype=torch.long,
        device=inputs.device,
    )
    output = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=positions,
    )
    if output.logits.shape[1] != target_ids.shape[1]:
        raise RuntimeError(
            "Target-only logits do not align with autoregressive target IDs."
        )
    token_losses = torch.nn.functional.cross_entropy(
        output.logits.float().reshape(-1, model.config.vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    if target_weights is None:
        loss = token_losses.mean()
    else:
        weights = target_weights.reshape(-1).float()
        if weights.shape != token_losses.shape:
            raise RuntimeError("Target loss weights do not align with target IDs.")
        loss = (token_losses * weights).sum() / weights.sum()
    return loss, output


def _target_loss_weights(
    *,
    tokenizer: Any,
    target_json: str,
    target_ids: Any,
    device: Any,
) -> Any:
    import torch

    encoded = tokenizer(
        target_json,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    gap_start = target_json.index('"gap_codes"')
    gap_end = target_json.index("]", gap_start) + 1
    weights = [
        (
            GAP_TOKEN_LOSS_WEIGHT
            if end > gap_start and start < gap_end
            else 1.0
        )
        for start, end in offsets
    ]
    weights.append(1.0)
    if len(weights) != int(target_ids.numel()):
        raise RuntimeError("Gap loss weights do not match tokenized target.")
    return torch.tensor(
        [weights],
        dtype=torch.float32,
        device=device,
    )


def _generation_loss(
    *,
    model: Any,
    tokenizer: Any,
    projector: Any,
    row: dict[str, Any],
    vectors: Any,
    mask: Any,
    variant: str,
    device: Any,
    maximum_prompt_tokens: int,
) -> Any:
    import torch

    use_soft = variant in {"soft_graph", "hybrid"}
    prompt_ids = _chat_prompt_ids(
        tokenizer,
        system_instruction=row["system_instruction"],
        user_prompt=row["prompts"][variant],
        with_soft_marker=use_soft,
        device=device,
        maximum_tokens=maximum_prompt_tokens,
    )
    soft_tokens = (
        _masked_soft_tokens(
            projector,
            vectors,
            mask,
            dtype=model.get_input_embeddings().weight.dtype,
        )
        if use_soft
        else None
    )
    prompt_embeddings, prompt_mask = _prompt_embeddings(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        soft_tokens=soft_tokens,
    )
    target_ids = tokenizer(
        row["target_json"],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    eos = torch.tensor(
        [[tokenizer.eos_token_id]],
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.cat((target_ids, eos), dim=1)
    target_weights = _target_loss_weights(
        tokenizer=tokenizer,
        target_json=row["target_json"],
        target_ids=target_ids,
        device=device,
    )
    target_embeddings = model.get_input_embeddings()(target_ids)
    inputs = torch.cat((prompt_embeddings, target_embeddings), dim=1)
    attention = torch.cat(
        (
            prompt_mask,
            torch.ones_like(target_ids),
        ),
        dim=1,
    )
    loss, _ = _target_only_causal_loss(
        model=model,
        inputs=inputs,
        attention_mask=attention,
        target_ids=target_ids,
        target_weights=target_weights,
    )
    return loss


def _f_score(predicted: set[str], expected: set[str]) -> dict[str, float | int]:
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / max(1, len(predicted))
    recall = true_positive / max(1, len(expected))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "exact_set_match": int(predicted == expected),
        "empty_target": int(not expected),
        "empty_target_correct": int(not expected and not predicted),
    }


def _decode_and_validate_handles(
    raw: str,
    row: dict[str, Any],
) -> tuple[
    dict[str, list[str]] | None,
    list[str],
    list[dict[str, str]],
]:
    try:
        selection = parse_selection(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, [f"invalid_json:{exc}"], []
    allowed_handles = {
        key: sorted((row["output_handle_map"].get(key) or {}).keys())
        for key in (
            "component_ids",
            "evidence_ids",
            "public_model_ids",
            "gap_codes",
        )
    }
    normalizations: list[dict[str, str]] = []
    for key, items in selection.items():
        handle_map = row["output_handle_map"].get(key) or {}
        reverse_map: dict[str, list[str]] = defaultdict(list)
        for handle, canonical_id in handle_map.items():
            reverse_map[str(canonical_id)].append(str(handle))
        normalized_items = []
        for item in items:
            if item in handle_map:
                normalized_items.append(item)
                continue
            candidate_handles = reverse_map.get(item) or []
            if len(candidate_handles) == 1:
                normalized = candidate_handles[0]
                normalized_items.append(normalized)
                normalizations.append(
                    {
                        "field": key,
                        "from": item,
                        "to": normalized,
                        "method": "unique_allowed_canonical_id_to_handle",
                    }
                )
            else:
                normalized_items.append(item)
        selection[key] = sorted(set(normalized_items))
    errors = validate_selection(selection, allowed_handles)
    return selection, errors, normalizations


def _translate_handles(
    selection: dict[str, list[str]],
    row: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        key: [
            row["output_handle_map"][key][handle]
            for handle in selection[key]
        ]
        for key in (
            "component_ids",
            "evidence_ids",
            "public_model_ids",
            "gap_codes",
        )
    }


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    projector: Any,
    row: dict[str, Any],
    vectors: Any,
    mask: Any,
    variant: str,
    device: Any,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    import torch

    use_soft = variant in {"soft_graph", "hybrid"}
    prompt_ids = _chat_prompt_ids(
        tokenizer,
        system_instruction=row["system_instruction"],
        user_prompt=row["prompts"][variant],
        with_soft_marker=use_soft,
        device=device,
        maximum_tokens=maximum_prompt_tokens,
    )
    started = time.time()
    with torch.inference_mode():
        if use_soft:
            soft_tokens = _masked_soft_tokens(
                projector,
                vectors,
                mask,
                dtype=model.get_input_embeddings().weight.dtype,
            )
            prompt_embeddings, attention = _prompt_embeddings(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                soft_tokens=soft_tokens,
            )
            generated = model.generate(
                inputs_embeds=prompt_embeddings,
                attention_mask=attention,
                max_new_tokens=maximum_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[0]
            output_ids = generated
            input_token_count = int(attention.sum())
        else:
            generated = model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=torch.ones(
                    (1, prompt_ids.shape[0]),
                    dtype=torch.long,
                    device=device,
                ),
                max_new_tokens=maximum_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[0]
            output_ids = generated[prompt_ids.shape[0] :]
            input_token_count = int(prompt_ids.shape[0])
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip(), {
        "input_token_count": input_token_count,
        "output_token_count": int(output_ids.shape[0]),
        "elapsed_seconds": round(time.time() - started, 4),
    }


def _validation_loss(
    *,
    model: Any,
    tokenizer: Any,
    projector: Any,
    rows: list[dict[str, Any]],
    vectors: Any,
    masks: Any,
    row_by_id: dict[str, int],
    device: Any,
    maximum_prompt_tokens: int,
) -> float:
    import torch

    projector.eval()
    values = []
    for row in rows:
        index = row_by_id[str(row["example_id"])]
        for variant in TRAINING_VARIANTS:
            with torch.no_grad():
                loss = _generation_loss(
                    model=model,
                    tokenizer=tokenizer,
                    projector=projector,
                    row=row,
                    vectors=vectors[index],
                    mask=masks[index],
                    variant=variant,
                    device=device,
                    maximum_prompt_tokens=maximum_prompt_tokens,
                )
            values.append(float(loss.detach().cpu()))
    return sum(values) / max(1, len(values))


def train_projector(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    vectors: Any,
    masks: Any,
    seed: int,
    output_path: Path,
    device: Any,
    maximum_prompt_tokens: int,
    maximum_epochs: int,
    patience: int,
) -> dict[str, Any]:
    import torch

    _seed_everything(seed)
    projector = build_projector(
        input_dimension=SOFT_INPUT_DIMENSION,
        hidden_dimension=PROJECTOR_HIDDEN_DIMENSION,
        output_dimension=int(model.config.hidden_size),
    ).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    row_by_id = {
        str(row["example_id"]): int(row["row"])
        for row in rows
    }
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    train_indices = torch.tensor(
        [row_by_id[str(row["example_id"])] for row in train_rows],
        dtype=torch.long,
        device=device,
    )
    target_embeddings, target_masks = _target_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        device=device,
    )

    history = []
    projector.train()
    for epoch in range(30):
        optimizer.zero_grad(set_to_none=True)
        projected, _ = projector(
            vectors[train_indices],
            masks[train_indices],
        )
        loss = _alignment_loss(
            projected,
            target_embeddings[train_indices],
            target_masks[train_indices],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "stage": "semantic_alignment",
                "epoch": epoch + 1,
                "train_loss": float(loss.detach().cpu()),
            }
        )

    best_validation = math.inf
    best_state = None
    epochs_without_improvement = 0
    generation_epochs = 0
    for epoch in range(maximum_epochs):
        generation_epochs = epoch + 1
        projector.train()
        order = list(train_rows)
        random.Random(seed * 1000 + epoch).shuffle(order)
        train_losses = []
        for row in order:
            index = row_by_id[str(row["example_id"])]
            for variant in TRAINING_VARIANTS:
                optimizer.zero_grad(set_to_none=True)
                loss = _generation_loss(
                    model=model,
                    tokenizer=tokenizer,
                    projector=projector,
                    row=row,
                    vectors=vectors[index],
                    mask=masks[index],
                    variant=variant,
                    device=device,
                    maximum_prompt_tokens=maximum_prompt_tokens,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
        validation_loss = _validation_loss(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            rows=validation_rows,
            vectors=vectors,
            masks=masks,
            row_by_id=row_by_id,
            device=device,
            maximum_prompt_tokens=maximum_prompt_tokens,
        )
        history.append(
            {
                "stage": "language_model_selection",
                "epoch": epoch + 1,
                "train_loss": sum(train_losses) / max(1, len(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        print(
            canonical_json(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train_loss": history[-1]["train_loss"],
                    "validation_loss": validation_loss,
                }
            ),
            flush=True,
        )
        if validation_loss < best_validation - 1e-4:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in projector.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("Projector training did not produce a validation checkpoint.")
    checkpoint = {
        "soft_prompt_version": SOFT_PROMPT_VERSION,
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "input_dimension": SOFT_INPUT_DIMENSION,
        "hidden_dimension": PROJECTOR_HIDDEN_DIMENSION,
        "output_dimension": int(model.config.hidden_size),
        "best_validation_loss": best_validation,
        "generation_epochs": generation_epochs,
        "history": history,
        "state_dict": best_state,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output_path)
    projector.load_state_dict(best_state)
    return {
        "seed": seed,
        "checkpoint_path": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "best_validation_loss": best_validation,
        "generation_epochs": generation_epochs,
        "history": history,
    }


def evaluate_variants(
    *,
    model: Any,
    tokenizer: Any,
    projector: Any,
    rows: list[dict[str, Any]],
    vectors: Any,
    masks: Any,
    device: Any,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    raw_rows = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        index = int(row["row"])
        for variant in PROMPT_VARIANTS:
            raw, generation = _generate(
                model=model,
                tokenizer=tokenizer,
                projector=projector,
                row=row,
                vectors=vectors[index],
                mask=masks[index],
                variant=variant,
                device=device,
                maximum_prompt_tokens=maximum_prompt_tokens,
                maximum_new_tokens=maximum_new_tokens,
            )
            selection, errors, handle_normalizations = (
                _decode_and_validate_handles(raw, row)
            )
            expected = {
                key: set(row["target"][key])
                for key in (
                    "component_ids",
                    "evidence_ids",
                    "public_model_ids",
                    "gap_codes",
                )
            }
            metrics = {}
            for key in expected:
                predicted = (
                    set(selection[key])
                    if selection is not None and not errors
                    else set()
                )
                metrics[key] = _f_score(predicted, expected[key])
            value = {
                "example_id": row["example_id"],
                "compound_id": row["compound_id"],
                "variant": variant,
                "raw_output": raw,
                "schema_valid": selection is not None and not errors,
                "selection": selection,
                "validation_errors": errors,
                "handle_normalizations": handle_normalizations,
                "metrics": metrics,
                "generation": generation,
            }
            raw_rows.append(value)
            by_variant[variant].append(value)
            print(
                canonical_json(
                    {
                        "evaluated": row["example_id"],
                        "variant": variant,
                        "schema_valid": value["schema_valid"],
                    }
                ),
                flush=True,
            )

    aggregate = {}
    for variant, values in by_variant.items():
        components = _aggregate_set_metrics(values, "component_ids")
        citations = _aggregate_set_metrics(values, "evidence_ids")
        public_models = _aggregate_set_metrics(values, "public_model_ids")
        gaps = _aggregate_set_metrics(values, "gap_codes")
        aggregate[variant] = {
            "example_count": len(values),
            "metric_aggregation": "micro_tp_fp_fn",
            "json_schema_valid_rate": sum(
                bool(row["schema_valid"]) for row in values
            )
            / max(1, len(values)),
            "component_precision": components["precision"],
            "component_recall": components["recall"],
            "component_f1": components["f1"],
            "component_exact_set_accuracy": components["exact_set_accuracy"],
            "component_nonempty_macro_f1": components["nonempty_macro_f1"],
            "citation_precision": citations["precision"],
            "citation_recall": citations["recall"],
            "citation_f1": citations["f1"],
            "citation_exact_set_accuracy": citations["exact_set_accuracy"],
            "citation_nonempty_macro_f1": citations["nonempty_macro_f1"],
            "public_model_precision": public_models["precision"],
            "public_model_recall": public_models["recall"],
            "public_model_f1": public_models["f1"],
            "public_model_exact_set_accuracy": public_models[
                "exact_set_accuracy"
            ],
            "public_model_nonempty_macro_f1": public_models[
                "nonempty_macro_f1"
            ],
            "evidence_gap_precision": gaps["precision"],
            "evidence_gap_recall": gaps["recall"],
            "evidence_gap_f1": gaps["f1"],
            "evidence_gap_exact_set_accuracy": gaps["exact_set_accuracy"],
            "evidence_gap_empty_target_specificity": gaps[
                "empty_target_specificity"
            ],
            "evidence_gap_nonempty_macro_f1": gaps["nonempty_macro_f1"],
            "mean_latency_seconds": sum(
                float(row["generation"]["elapsed_seconds"]) for row in values
            )
            / max(1, len(values)),
        }
    return aggregate, raw_rows


def _aggregate_set_metrics(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float | int]:
    true_positive = sum(
        int(row["metrics"][key]["true_positive"]) for row in rows
    )
    false_positive = sum(
        int(row["metrics"][key]["false_positive"]) for row in rows
    )
    false_negative = sum(
        int(row["metrics"][key]["false_negative"]) for row in rows
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    nonempty_rows = [
        row for row in rows if not int(row["metrics"][key]["empty_target"])
    ]
    empty_rows = [
        row for row in rows if int(row["metrics"][key]["empty_target"])
    ]
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_accuracy": sum(
            int(row["metrics"][key]["exact_set_match"]) for row in rows
        )
        / max(1, len(rows)),
        "nonempty_macro_f1": sum(
            float(row["metrics"][key]["f1"]) for row in nonempty_rows
        )
        / max(1, len(nonempty_rows)),
        "empty_target_count": len(empty_rows),
        "empty_target_specificity": sum(
            int(row["metrics"][key]["empty_target_correct"])
            for row in empty_rows
        )
        / max(1, len(empty_rows)),
    }


def _mean_metric(
    rows: list[dict[str, Any]],
    key: str,
    metric: str,
) -> float:
    return sum(float(row["metrics"][key][metric]) for row in rows) / max(
        1,
        len(rows),
    )


def _accepted(
    metrics: dict[str, Any],
    final_validation: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    hybrid = metrics["hybrid"]
    no_kg = metrics["no_kg"]
    checks = {
        "json_schema_valid_rate": (
            hybrid["json_schema_valid_rate"]
            >= GATE5_ACCEPTANCE["minimum_json_schema_valid_rate"]
        ),
        "component_f1_delta": (
            hybrid["component_f1"]
            >= no_kg["component_f1"]
            + GATE5_ACCEPTANCE[
                "minimum_hybrid_component_f1_delta_vs_no_kg"
            ]
        ),
        "citation_recall": (
            hybrid["citation_recall"]
            >= GATE5_ACCEPTANCE["minimum_hybrid_citation_recall"]
        ),
        "evidence_gap_f1": (
            hybrid["evidence_gap_f1"]
            >= GATE5_ACCEPTANCE["minimum_hybrid_evidence_gap_f1"]
        ),
        "citation_provenance_completeness": (
            final_validation["minimum_citation_provenance_completeness"]
            == GATE5_ACCEPTANCE["required_citation_provenance_completeness"]
        ),
        "unsupported_numeric_parameter_rate": (
            final_validation["maximum_unsupported_numeric_parameter_rate"]
            <= GATE5_ACCEPTANCE["maximum_unsupported_numeric_parameter_rate"]
        ),
        "quarantined_evidence_leakage": (
            final_validation["quarantined_evidence_leakage"]
            <= GATE5_ACCEPTANCE["maximum_quarantined_evidence_leakage"]
        ),
        "ontology_invalid_recommendation": (
            final_validation["ontology_invalid_recommendation"]
            <= GATE5_ACCEPTANCE["maximum_ontology_invalid_recommendation"]
        ),
    }
    return all(checks.values()), checks


def build_final_runs(
    *,
    corpus: GraphPromptCorpus,
    rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hybrid_by_id = {
        str(row["example_id"]): row
        for row in raw_rows
        if row["variant"] == "hybrid"
    }
    validations = []
    final_rows = []
    for row in rows:
        if row["split"] != "test":
            continue
        generation = hybrid_by_id[str(row["example_id"])]
        if not generation["schema_valid"] or generation["selection"] is None:
            validations.append(
                {
                    "example_id": row["example_id"],
                    "status": "recommendation_quarantined",
                    "errors": generation["validation_errors"],
                    "citation_provenance_completeness": 0.0,
                    "unsupported_numeric_parameter_rate": 0.0,
                    "quarantined_evidence_leakage": 0,
                }
            )
            continue
        selection = _translate_handles(generation["selection"], row)
        request = ModelingRequest.model_validate(row["modeling_request"])
        subgraphs = [
            RetrievedSubgraph(
                subgraph_id=str(trace["subgraph_id"]),
                hop=int(trace["hop"]),
                score=float(trace["final_score"]),
                score_components={
                    str(key): float(value)
                    for key, value in trace["score_components"].items()
                },
                row=corpus.subgraphs[str(trace["subgraph_id"])],
                vector=None,
            )
            for trace in row["retrieval_trace"]
        ]
        recommendation = assemble_recommendation(
            corpus=corpus,
            request=request,
            subgraphs=subgraphs,
            selection=selection,
        )
        validation = validate_recommendation(
            corpus=corpus,
            request=request,
            recommendation=recommendation,
            allowed_subgraph_ids={
                str(value) for value in row["retrieved_subgraph_ids"]
            },
        )
        validation["example_id"] = row["example_id"]
        validations.append(validation)
        run_dir = output_dir / "recommendation_runs" / str(row["example_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "recommendation_request.json", row["modeling_request"])
        _atomic_json(
            run_dir / "resolved_modeling_request.json",
            request.model_dump(mode="json"),
        )
        compound = corpus.resolve_compound(request.compound)
        _atomic_jsonl(
            run_dir / "seed_entities.jsonl",
            [
                {
                    "node_id": compound["node_id"],
                    "node_type": "Compound",
                    "preferred_name": compound["preferred_name"],
                }
            ],
        )
        _atomic_jsonl(
            run_dir / "traversed_paths.jsonl",
            [
                {
                    "path_id": path_id,
                }
                for path_id in recommendation["traversed_path_ids"]
            ],
        )
        _atomic_jsonl(
            run_dir / "coarse_retrieved_subgraphs.jsonl",
            [
                {
                    "subgraph_id": item.subgraph_id,
                    "hop": item.hop,
                    "score_components": item.score_components,
                }
                for item in subgraphs
            ],
        )
        _atomic_jsonl(
            run_dir / "reranked_subgraphs.jsonl",
            recommendation["retrieved_subgraphs"],
        )
        _atomic_json(
            run_dir / "component_coverage.json",
            {
                row["component"]: row
                for row in recommendation["component_recommendations"]
            },
        )
        _atomic_jsonl(
            run_dir / "parameter_candidates.jsonl",
            recommendation["parameter_candidates"],
        )
        _atomic_json(run_dir / "recommendation_draft.json", recommendation)
        _atomic_json(run_dir / "recommendation_validation.json", validation)
        if validation["status"] == "recommendation_validated":
            recommendation["status"] = "recommendation_validated"
            _atomic_json(run_dir / "recommendation_final.json", recommendation)
            final_rows.append(recommendation)
        else:
            _atomic_json(
                run_dir / "recommendation_final.json",
                {
                    "status": "not_emitted",
                    "validation_status": validation["status"],
                },
            )
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in run_dir.iterdir()
            if path.is_file()
        }
        _atomic_json(
            run_dir / "recommendation_run_manifest.json",
            {
                "soft_prompt_version": SOFT_PROMPT_VERSION,
                "example_id": row["example_id"],
                "prompt_variant": "hybrid",
                "artifacts": artifact_hashes,
                "generated_at": utc_now(),
            },
        )
    summary = {
        "recommendation_count": len(validations),
        "validated_count": sum(
            row["status"] == "recommendation_validated"
            for row in validations
        ),
        "repairable_count": sum(
            row["status"] == "recommendation_repairable"
            for row in validations
        ),
        "quarantined_count": sum(
            row["status"] == "recommendation_quarantined"
            for row in validations
        ),
        "minimum_citation_provenance_completeness": min(
            (
                float(row.get("citation_provenance_completeness") or 0.0)
                for row in validations
                if row["status"] == "recommendation_validated"
            ),
            default=0.0,
        ),
        "maximum_unsupported_numeric_parameter_rate": max(
            (
                float(row.get("unsupported_numeric_parameter_rate") or 0.0)
                for row in validations
            ),
            default=0.0,
        ),
        "quarantined_evidence_leakage": sum(
            int(row.get("quarantined_evidence_leakage") or 0)
            for row in validations
        ),
        "ontology_invalid_recommendation": sum(
            any(
                str(error).startswith(
                    (
                        "public_model_not_reusable",
                        "parameter_compound_mismatch",
                    )
                )
                for error in row.get("errors") or ()
            )
            for row in validations
        ),
    }
    return final_rows, summary


def _model_weight_manifest(model_path: Path) -> dict[str, Any]:
    files = sorted(
        path
        for pattern in ("*.safetensors", "*.json")
        for path in model_path.glob(pattern)
        if path.is_file()
    )
    rows = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "files": rows,
        "content_sha256": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
    }


def _verify_soft_input_compatibility(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    rows: list[dict[str, Any]],
    vectors: Any,
    masks: Any,
    maximum_prompt_tokens: int,
) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise RuntimeError("Compatibility probe requires at least one training row.")

    def probe_length(row: dict[str, Any], variant: str) -> int:
        prompt_ids = _chat_prompt_ids(
            tokenizer,
            system_instruction=row["system_instruction"],
            user_prompt=row["prompts"][variant],
            with_soft_marker=True,
            device=device,
            maximum_tokens=maximum_prompt_tokens,
        )
        target_ids = tokenizer(
            row["target_json"],
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        return int(prompt_ids.numel() + target_ids.numel() + 1)

    row, probe_variant = max(
        (
            (row, variant)
            for row in train_rows
            for variant in TRAINING_VARIANTS
        ),
        key=lambda value: probe_length(value[0], value[1]),
    )
    row_index = int(row["row"])
    projector = build_projector(
        input_dimension=SOFT_INPUT_DIMENSION,
        hidden_dimension=PROJECTOR_HIDDEN_DIMENSION,
        output_dimension=int(model.config.hidden_size),
    ).to(device)
    probe_vectors = vectors[row_index]
    probe_mask = masks[row_index]
    prompt_ids = _chat_prompt_ids(
        tokenizer,
        system_instruction=row["system_instruction"],
        user_prompt=row["prompts"][probe_variant],
        with_soft_marker=True,
        device=device,
        maximum_tokens=maximum_prompt_tokens,
    )
    soft_tokens = _masked_soft_tokens(
        projector,
        probe_vectors,
        probe_mask,
        dtype=model.get_input_embeddings().weight.dtype,
    )
    soft_tokens.retain_grad()
    prompt_embeddings, prompt_mask = _prompt_embeddings(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        soft_tokens=soft_tokens,
    )
    target_ids = tokenizer(
        row["target_json"],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    target_ids = torch.cat(
        (
            target_ids,
            torch.tensor(
                [[tokenizer.eos_token_id]],
                dtype=torch.long,
                device=device,
            ),
        ),
        dim=1,
    )
    target_embeddings = model.get_input_embeddings()(target_ids)
    inputs = torch.cat((prompt_embeddings, target_embeddings), dim=1)
    attention = torch.cat((prompt_mask, torch.ones_like(target_ids)), dim=1)

    projector.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    loss, output = _target_only_causal_loss(
        model=model,
        inputs=inputs,
        attention_mask=attention,
        target_ids=target_ids,
    )
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.detach().float().norm().cpu())
        for parameter in projector.parameters()
        if parameter.grad is not None
    )
    soft_token_gradient_norms = [
        float(value)
        for value in soft_tokens.grad.detach().float().norm(dim=-1).cpu()
    ]
    effective_soft_token_count = 1 + int(probe_mask.sum().item())
    effective_gradient_norms = soft_token_gradient_norms[
        :effective_soft_token_count
    ]
    backward_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    backward_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    loss_value = float(loss.detach().float().cpu())

    had_gradient_checkpointing = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    if had_gradient_checkpointing:
        model.gradient_checkpointing_disable()
    del output, inputs, target_embeddings, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        generated_soft_tokens = _masked_soft_tokens(
            projector,
            probe_vectors,
            probe_mask,
            dtype=model.get_input_embeddings().weight.dtype,
        )
        generated_embeddings, generated_attention = _prompt_embeddings(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            soft_tokens=generated_soft_tokens,
        )
        generated = model.generate(
            inputs_embeds=generated_embeddings,
            attention_mask=generated_attention,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generation_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    generation_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if had_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    del (
        projector,
        prompt_embeddings,
        prompt_mask,
        generated_embeddings,
        generated_attention,
        generated,
    )
    torch.cuda.empty_cache()
    return {
        "probe_example_id": str(row["example_id"]),
        "probe_variant": probe_variant,
        "probe_prompt_token_count": int(prompt_ids.numel()),
        "probe_target_token_count": int(target_ids.numel()),
        "probe_total_token_count": int(
            prompt_ids.numel() + target_ids.numel()
        ),
        "probe_loss": loss_value,
        "probe_loss_finite": math.isfinite(loss_value),
        "effective_soft_token_count": effective_soft_token_count,
        "soft_token_gradient_norms": soft_token_gradient_norms,
        "effective_soft_token_gradients_passed": bool(
            effective_gradient_norms
        )
        and all(
            math.isfinite(value) and value > 0.0
            for value in effective_gradient_norms
        ),
        "inputs_embeds_forward_passed": True,
        "inputs_embeds_generation_passed": True,
        "projector_gradient_passed": math.isfinite(gradient_norm)
        and gradient_norm > 0.0,
        "projector_gradient_norm": gradient_norm,
        "backward_peak_allocated_bytes": backward_peak_allocated,
        "backward_peak_reserved_bytes": backward_peak_reserved,
        "generation_peak_allocated_bytes": generation_peak_allocated,
        "generation_peak_reserved_bytes": generation_peak_reserved,
    }


def run_gate5(
    *,
    dataset_dir: Path,
    graph_dir: Path,
    subgraph_dir: Path,
    gnn_index_dir: Path,
    model_path: Path,
    output_dir: Path,
    device_name: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    maximum_epochs: int,
    patience: int,
    model_id: str,
    model_revision: str,
    model_source: str,
    expected_hidden_size: int,
    probe_only: bool,
    reuse_projector_dir: Path | None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    acceptance_path = dataset_dir / "preregistered_acceptance.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("accepted") is not True:
        raise RuntimeError("Gate 5 dataset is not accepted.")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("locked_before_training") is not True:
        raise RuntimeError("Gate 5 acceptance was not locked before training.")
    locked_llm = acceptance.get("llm") or {}
    expected_lock = {
        "model": model_id,
        "revision": model_revision,
        "hidden_size": expected_hidden_size,
        "dtype": DEFAULT_GATE5_LLM_DTYPE,
    }
    if locked_llm != expected_lock:
        raise RuntimeError(
            f"Gate 5 LLM lock differs from runtime: "
            f"locked={locked_llm}, runtime={expected_lock}"
        )
    training_lock = acceptance.get("training") or {}
    if training_lock.get("prompt_variants") != list(TRAINING_VARIANTS):
        raise RuntimeError("Gate 5 training prompt variants are not locked.")
    if int(training_lock.get("maximum_prompt_tokens") or 0) != int(
        maximum_prompt_tokens
    ):
        raise RuntimeError(
            "Gate 5 maximum prompt tokens differ from the training lock."
        )
    if float(training_lock.get("gap_token_loss_weight") or 0.0) != float(
        GAP_TOKEN_LOSS_WEIGHT
    ):
        raise RuntimeError("Gate 5 gap token loss weight is not locked.")
    rows = _load_jsonl(dataset_dir / "prompt_examples.jsonl")
    vectors_numpy = np.load(dataset_dir / "soft_graph_inputs.npy")
    masks_numpy = np.load(dataset_dir / "soft_graph_masks.npy")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map={"": device_name},
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    vectors = torch.from_numpy(vectors_numpy).to(device=device, dtype=torch.float32)
    masks = torch.from_numpy(masks_numpy).to(device=device, dtype=torch.bool)

    forward_parameters = inspect.signature(model.forward).parameters
    runtime_compatibility = _verify_soft_input_compatibility(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rows=rows,
        vectors=vectors,
        masks=masks,
        maximum_prompt_tokens=maximum_prompt_tokens,
    )
    weight_manifest = _model_weight_manifest(model_path)
    compatibility = {
        "model_class": model.__class__.__name__,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_source": model_source,
        "model_path": str(model_path),
        "model_dtype": str(model.get_input_embeddings().weight.dtype),
        "hidden_size": int(model.config.hidden_size),
        "inputs_embeds_supported": "inputs_embeds" in forward_parameters,
        "model_weight_manifest": weight_manifest,
        **runtime_compatibility,
    }
    if (
        compatibility["model_id"] != model_id
        or compatibility["model_revision"] != model_revision
        or compatibility["hidden_size"] != expected_hidden_size
        or compatibility["model_dtype"] != "torch.bfloat16"
        or not compatibility["inputs_embeds_supported"]
        or not compatibility["inputs_embeds_forward_passed"]
        or not compatibility["inputs_embeds_generation_passed"]
        or not compatibility["probe_loss_finite"]
        or not compatibility["effective_soft_token_gradients_passed"]
        or not compatibility["projector_gradient_passed"]
    ):
        raise RuntimeError(f"Gate 5 LLM compatibility mismatch: {compatibility}")
    compatibility_path = output_dir / "compatibility_probe.json"
    _atomic_json(compatibility_path, compatibility)
    if probe_only:
        manifest = {
            "accepted": True,
            "probe_only": True,
            "soft_prompt_version": SOFT_PROMPT_VERSION,
            "generated_at": utc_now(),
            "llm": expected_lock,
            "model_source": model_source,
            "model_path": str(model_path),
            "compatibility_probe": {
                "path": compatibility_path.name,
                "sha256": sha256_file(compatibility_path),
            },
        }
        _atomic_json(output_dir / "gate5_probe_manifest.json", manifest)
        return manifest

    training_runs = []
    if reuse_projector_dir is None:
        for seed in SEEDS:
            result = train_projector(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                vectors=vectors,
                masks=masks,
                seed=seed,
                output_path=output_dir / "projectors" / f"seed_{seed}.pt",
                device=device,
                maximum_prompt_tokens=maximum_prompt_tokens,
                maximum_epochs=maximum_epochs,
                patience=patience,
            )
            training_runs.append(result)
    else:
        for seed in SEEDS:
            checkpoint_path = reuse_projector_dir / f"seed_{seed}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Reusable Gate 5 projector is missing: {checkpoint_path}"
                )
            checkpoint_row = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if int(checkpoint_row.get("seed") or -1) != seed:
                raise RuntimeError(
                    f"Projector seed mismatch in {checkpoint_path}."
                )
            training_runs.append(
                {
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "best_validation_loss": float(
                        checkpoint_row["best_validation_loss"]
                    ),
                    "generation_epochs": int(
                        checkpoint_row["generation_epochs"]
                    ),
                    "history": checkpoint_row["history"],
                    "reused_without_training": True,
                }
            )
    selected = min(
        training_runs,
        key=lambda row: (
            float(row["best_validation_loss"]),
            int(row["seed"]),
        ),
    )
    checkpoint = torch.load(
        selected["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    projector = build_projector(
        input_dimension=SOFT_INPUT_DIMENSION,
        hidden_dimension=PROJECTOR_HIDDEN_DIMENSION,
        output_dimension=int(model.config.hidden_size),
    ).to(device)
    projector.load_state_dict(checkpoint["state_dict"])
    projector.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    metrics, raw_rows = evaluate_variants(
        model=model,
        tokenizer=tokenizer,
        projector=projector,
        rows=rows,
        vectors=vectors,
        masks=masks,
        device=device,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    raw_path = output_dir / "raw_variant_outputs.jsonl"
    _atomic_jsonl(raw_path, raw_rows)
    corpus = GraphPromptCorpus(
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        gnn_index_dir=gnn_index_dir,
    )
    final_rows, final_validation = build_final_runs(
        corpus=corpus,
        rows=rows,
        raw_rows=raw_rows,
        output_dir=output_dir,
    )
    accepted, checks = _accepted(metrics, final_validation)
    comparison = {
        "accepted": accepted,
        "soft_prompt_version": SOFT_PROMPT_VERSION,
        "compatibility": compatibility,
        "selected_seed": selected["seed"],
        "selection_basis": "lowest validation loss only",
        "training_runs": training_runs,
        "training_mode": (
            "reused_locked_three_seed_projectors"
            if reuse_projector_dir is not None
            else "trained_three_seed_projectors"
        ),
        "variant_metrics": metrics,
        "final_validation": final_validation,
        "acceptance_checks": checks,
    }
    comparison_path = output_dir / "gate5_comparison.json"
    _atomic_json(comparison_path, comparison)
    _atomic_jsonl(output_dir / "validated_recommendations.jsonl", final_rows)
    manifest = {
        "accepted": accepted,
        "soft_prompt_version": SOFT_PROMPT_VERSION,
        "generated_at": utc_now(),
        "llm": {
            "model": model_id,
            "revision": model_revision,
            "source": model_source,
            "model_path": str(model_path),
            "dtype": str(model.get_input_embeddings().weight.dtype),
            "hidden_size": int(model.config.hidden_size),
        },
        "projector": {
            "selected_seed": selected["seed"],
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
        },
        "input_hashes": {
            "dataset_manifest": sha256_file(dataset_manifest_path),
            "acceptance": sha256_file(acceptance_path),
            "graph_manifest": sha256_file(
                graph_dir / "kg_extraction_manifest.json"
            ),
            "subgraph_manifest": sha256_file(
                subgraph_dir / "gate2_manifest.json"
            ),
            "gnn_index_manifest": sha256_file(
                gnn_index_dir / "gnn_index_manifest.json"
            ),
        },
        "artifacts": {
            "comparison": {
                "path": comparison_path.name,
                "sha256": sha256_file(comparison_path),
            },
            "raw_outputs": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
            },
            "validated_recommendations": {
                "path": "validated_recommendations.jsonl",
                "sha256": sha256_file(
                    output_dir / "validated_recommendations.jsonl"
                ),
                "count": len(final_rows),
            },
            "compatibility_probe": {
                "path": compatibility_path.name,
                "sha256": sha256_file(compatibility_path),
            },
        },
    }
    _atomic_json(output_dir / "gate5_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the true Qwen3 soft graph prompt."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--gnn-index-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1408)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--model-id", default=DEFAULT_GATE5_LLM_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_GATE5_LLM_REVISION)
    parser.add_argument(
        "--model-source",
        default="modelscope:Qwen/Qwen3-8B@master",
    )
    parser.add_argument(
        "--expected-hidden-size",
        type=int,
        default=DEFAULT_GATE5_LLM_HIDDEN_SIZE,
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--reuse-projector-dir",
        type=Path,
        help=(
            "Skip training and evaluate the locked seed_17/29/43 projectors "
            "from this server-side directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_gate5(
        dataset_dir=args.dataset_dir,
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        gnn_index_dir=args.gnn_index_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device_name=args.device,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        model_id=args.model_id,
        model_revision=args.model_revision,
        model_source=args.model_source,
        expected_hidden_size=args.expected_hidden_size,
        probe_only=args.probe_only,
        reuse_projector_dir=args.reuse_projector_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
