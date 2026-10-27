#!/usr/bin/env python3
"""Apply verified component preferences to frozen, held-out PBPK-Match heads.

No training, baseline regeneration, outcome extraction or threshold selection.
Every original request and all 280 candidates remain in the paired experiment.
Unusable component outputs retain the original input and original predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import UUID, uuid5, NAMESPACE_URL

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
OLD = ROOT / "experiments/model_recommendation/pbpk_match_light"
DEFAULT = ROOT / "experiments/model_recommendation/component_integration_20261007"
RUNTIME = ROOT / "runtime/deployment/component_model_integration"
SOFT_FIELDS = ("enzymes", "transporters", "process_types", "compartments")


def scoped(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or "sealed" in path.relative_to(ROOT).parts:
        raise ValueError("Component integration files must belong to this project")
    return path


def read(path):
    return json.loads(scoped(path).read_text())


def rows(path):
    return [json.loads(line) for line in scoped(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(scoped(path).read_bytes()).hexdigest()


def ref(path):
    path = scoped(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def verify_ref(value):
    path = scoped(value["path"])
    if sha(path) != value["sha256"]:
        raise ValueError("Frozen artifact changed: " + value["path"])
    return path


def verify_baseline():
    protocol_path = OLD / "protocol.json"
    protocol, receipt = read(protocol_path), read(OLD / "predictions/seed17.json")
    if receipt.get("state") != "complete" or receipt.get("protocol_sha256") != sha(protocol_path):
        raise ValueError("Baseline completion does not bind the training protocol")
    if receipt["score_sha256"] != sha(OLD / "predictions/seed17.npz"):
        raise ValueError("Baseline prediction matrix changed")
    for field in ("dataset_manifest", "cache_manifest"):
        verify_ref(protocol[field])
    for filename, expected in protocol["implementation_sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Frozen baseline implementation changed: " + filename)
    for fold_id in range(5):
        complete_path = verify_ref(receipt["outer_receipts"][str(fold_id)])
        directory = OLD / f"runs/seed17/fold{fold_id}/outer"
        if complete_path != directory / "complete.json":
            raise ValueError("Unexpected baseline outer checkpoint path")
        complete = read(complete_path)
        for filename, expected in complete["artifacts_sha256"].items():
            if sha(directory / filename) != expected:
                raise ValueError("Frozen checkpoint artifact changed: " + filename)
        binding = read(directory / "binding.json")
        selection = verify_ref(binding["selection"])
        if selection != directory.parent / "selection.json" or sha(selection) != binding["configuration"]["selection_sha256"]:
            raise ValueError("Frozen error correction or calibration changed")
        if binding["protocol_sha256"] != sha(protocol_path):
            raise ValueError("Outer fit belongs to a different baseline protocol")
    return protocol


def write(path, value):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def array_write(path, **arrays):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def progress(phase, **extra):
    value = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    write(RUNTIME / "ranking_status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def build_augmented_inputs(run_dir):
    """Revalidate every generated recommendation before using its soft fields."""
    from pbpk_kb.recommendation.component_model_bridge import augment_model_query, ComponentModelBridgeError
    from pbpk_kb.recommendation.graph_prompt import GraphPromptCorpus
    from pbpk_kb.recommendation.schemas import ModelingRequest

    run_dir = scoped(run_dir)
    manifest_path = run_dir / "components/manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("total") != 186:
        raise ValueError("All component attempts must finish before paired ranking")
    if manifest["protocol"]["sha256"] != sha(run_dir / "protocol.json"):
        raise ValueError("Component generation protocol changed")
    protocol = read(run_dir / "protocol.json")
    for item in [*protocol["source_artifacts"], *protocol["prepared_input_artifacts"]]:
        verify_ref(item)
    generation_binding = read(run_dir / "run_binding.json")
    if generation_binding["protocol_sha256"] != sha(run_dir / "protocol.json"):
        raise ValueError("Component producer is bound to another protocol")
    for key in ("prepare_script", "generate_script"):
        verify_ref(generation_binding[key])
    entries = rows(run_dir / "inputs/requests.jsonl")
    refs = {row["query_id"]: row for row in manifest["records"]}
    if len(refs) != 186 or set(refs) != {e["query_id"] for e in entries}:
        raise ValueError("Component ledger does not cover the fixed request set")
    corpus = GraphPromptCorpus(graph_dir=scoped(manifest["graph_dir"]),
                               subgraph_dir=scoped(manifest["subgraph_dir"]))
    source_run = uuid5(NAMESPACE_URL, "pbpk-component-integration:" + sha(run_dir / "protocol.json"))
    ledger, sources = [], [ref(manifest_path), ref(run_dir / "inputs/requests.jsonl"),
                           ref(ROOT / "src/pbpk_kb/recommendation/component_model_bridge.py")]
    for entry in entries:
        record_ref = refs[entry["query_id"]]
        if sha(record_ref["path"]) != record_ref["sha256"]:
            raise ValueError("A component result changed after generation")
        sources.append(ref(record_ref["path"]))
        result = read(record_ref["path"])
        if any(result.get(key) != entry[key] for key in ("query_id", "global_index", "request_sha256")):
            raise ValueError("Component result is bound to another request")
        original = copy.deepcopy(entry["query_record"])
        item = {"query_id": entry["query_id"], "global_index": entry["global_index"],
                "cohort": entry["cohort"], "original_text": original["text"],
                "augmented_query": original, "status": result["status"],
                "component_result": record_ref,
                "audit": {"fallback": True, "reason": result.get("reason", result["status"]),
                          "original_input_retained": True}}
        if result["status"] == "validated":
            prompt_ref = result["prompt_graph"]
            if sha(prompt_ref["path"]) != prompt_ref["sha256"]:
                raise ValueError("Component prompt graph changed")
            sources.append(ref(prompt_ref["path"]))
            prompt = read(prompt_ref["path"])
            try:
                augmented = augment_model_query(original, ModelingRequest.model_validate(entry["modeling_request"]),
                    result["recommendation"], corpus, prompt, source_run)
            except ComponentModelBridgeError as exc:
                item["status"] = "bridge_rejected"
                item["audit"] = {"fallback": True, "reason": str(exc), "original_input_retained": True}
            else:
                profile_path = run_dir / "profiles" / (entry["key"] + ".json")
                write(profile_path, augmented)
                item.update(augmented_query=augmented["augmented_query"], audit=augmented["audit"],
                            status="augmented" if augmented["audit"]["status"] == "augmented" else "validated_no_soft_addition",
                            profile=ref(profile_path))
        ledger.append(item)
    path = run_dir / "inputs/augmented_queries.jsonl"
    content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in ledger)
    if path.exists() and path.read_text() != content:
        raise ValueError("Previously frozen component-augmented inputs differ")
    path.write_text(content)
    receipt = {"state": "complete", "requests": len(ledger),
               "statuses": dict(Counter(r["status"] for r in ledger)),
               "by_cohort": {c: dict(Counter(r["status"] for r in ledger if r["cohort"] == c)) for c in ("core", "execution")},
               "inputs": ref(path), "sources": sources,
               "profile_eligibility_used_for_ranking": False,
               "invalid_component_results_use_unchanged_input": True}
    write(run_dir / "inputs/bridge_manifest.json", receipt)
    progress("component_bridge_complete", **{k: receipt[k] for k in ("requests", "statuses", "by_cohort")})
    return receipt


def soft_features(original, augmented, candidates, frozen_features):
    """Change only the four allowed categorical mechanism blocks.

    Dose, species, population, food and all other original condition features
    remain byte-for-byte equal, including missing-value indicators.
    """
    from pbpk_kb.recommendation.pbpk_match_v2.data import FIELDS, fields_for, pair_feature_row

    if not augmented["text"].startswith(original["text"]):
        raise ValueError("Component text replaced part of the original request")
    for key in ("query_id", "compounds", "task_type"):
        if augmented.get(key) != original.get(key):
            raise ValueError("Component input changed request identity or task")
    old, new = fields_for(original), fields_for(augmented)
    for key in FIELDS:
        if key not in SOFT_FIELDS and old[key] != new[key]:
            raise ValueError("Component input overwrote a hard request field: " + key)
        if key in SOFT_FIELDS and not set(old[key]) <= set(new[key]):
            raise ValueError("Component input removed an original mechanism requirement")
    features = np.asarray(frozen_features, np.float32).copy()
    columns = [j for field in SOFT_FIELDS for j in range(FIELDS.index(field) * 5, FIELDS.index(field) * 5 + 5)]
    for i, candidate in enumerate(candidates):
        computed = pair_feature_row(augmented, candidate)
        features[i, columns] = computed[columns]
    protected = np.ones(features.shape[-1], bool)
    protected[columns] = False
    if not np.array_equal(features[:, protected], np.asarray(frozen_features)[:, protected]):
        raise AssertionError("Protected numerical/condition features changed")
    return features


def infer_fold(vectors, features, candidates, records, requested, directory, *, device):
    """Restore one immutable fold, retaining its alpha and calibration."""
    import torch
    from pbpk_match_light_training import LightMatch, _normalized, _canonical
    from pbpk_kb.recommendation.pbpk_match_v2.model import ErrorHead
    from pbpk_kb.recommendation.pbpk_match_v2.training import _query_endpoint
    from pbpk_kb.recommendation.pbpk_match_v2.evaluation import calibrate
    from pbpk_kb.recommendation.pbpk_match_v2.experiment import correction

    directory = scoped(directory)
    metadata = read(directory / "metadata.json")
    selection = read(directory.parent / "selection.json")
    state = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
    model = LightMatch(metadata["embedding_size"], len(metadata["feature_mean"]),
                       metadata["width"], metadata["dropout"]).to(device).eval()
    model.load_state_dict(state["heads"])
    error_meta = state["error_metadata"]
    endpoints = error_meta.get("enabled_endpoints", [])
    error_head = None
    if endpoints:
        error_head = ErrorHead(error_meta["input_size"], len(endpoints)).to(device).eval()
        error_head.load_state_dict(state["error_head"])
    base, logits, reps = [], [], []
    with torch.inference_mode():
        for i, vector in enumerate(vectors):
            normalized = _normalized(features[i], metadata)
            chosen, restore = _canonical(candidates, normalized)
            result = model(torch.as_tensor(vector, device=device),
                           torch.as_tensor(candidates[chosen], device=device),
                           torch.as_tensor(normalized[chosen], device=device))
            base.append(result["scores"].cpu().numpy()[restore])
            logits.append(result["app_logits"].cpu().numpy()[restore])
            reps.append(result["representation"].cpu().numpy()[restore])
        base, logits = np.asarray(base), np.asarray(logits)
        probability = np.exp(logits - logits.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        probability = calibrate(probability.astype(np.float32), selection["calibration"]["temperature"])
        risk = np.zeros_like(base)
        supported = np.zeros(len(records), bool)
        if error_head is not None:
            representations = torch.as_tensor(np.asarray(reps), device=device)
            predicted = {}
            for e, endpoint in enumerate(endpoints):
                eids = torch.full(representations.shape[:-1], e, dtype=torch.long, device=device)
                predicted[endpoint] = error_head(representations, eids).cpu().numpy()
            for i, record in enumerate(records):
                endpoint = _query_endpoint(record, requested[i])
                if endpoint in predicted:
                    risk[i], supported[i] = predicted[endpoint][i], True
        alpha = selection["error_weight"]["weight"]
        scores = correction({"scores": base, "base_scores": base, "selected_predicted_error": risk,
                             "error_supported": supported}, alpha)
    result = {"scores": scores, "base_scores": base, "probabilities": probability,
              "selected_predicted_error": risk, "error_supported": supported}
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite frozen-head result: " + key)
    del model, error_head
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def exact_fold_membership(indices, protocol):
    assignments = {}
    for index in indices:
        matches = [i for i, fold in enumerate(protocol["folds"]) if int(index) in fold["outer_test"]]
        if len(matches) != 1:
            raise ValueError("Request has no unique frozen outer holdout")
        fold = protocol["folds"][matches[0]]
        if int(index) in fold["outer_train"]:
            raise ValueError("Evaluated request appears in the fitted fold")
        group = protocol["groups"][int(index)]
        if group in {protocol["groups"][i] for i in fold["outer_train"]}:
            raise ValueError("Evaluated request group appears in training")
        assignments[int(index)] = matches[0]
    return assignments


def verify_bridge_inputs(run_dir):
    """The checked recommendation ledger, not a later edit, defines inputs."""
    ledger_path = scoped(run_dir) / "inputs/augmented_queries.jsonl"
    manifest = read(scoped(run_dir) / "inputs/bridge_manifest.json")
    if manifest.get("state") != "complete" or manifest.get("requests") != 186:
        raise ValueError("Component bridge is incomplete")
    if verify_ref(manifest["inputs"]) != ledger_path:
        raise ValueError("Component bridge binds a different input ledger")
    if not manifest.get("sources"):
        raise ValueError("Component bridge has no source bindings")
    for source in manifest["sources"]:
        verify_ref(source)
    return ledger_path


def run_inference(run_dir, *, device="cuda:0", batch_size=16):
    """Consume the generated/validated augmented-input ledger, without outcomes."""
    import torch
    from pbpk_match_light_inputs import visible_inputs, load_cache, _encode_missing

    started = time.monotonic()
    run_dir = scoped(run_dir)
    protocol = read(run_dir / "protocol.json")
    old_protocol = verify_baseline()
    # The bridge runner writes this immutable, per-request input ledger after
    # revalidating all source-bound recommendations.
    ledger_path = verify_bridge_inputs(run_dir)
    ledger = rows(ledger_path)
    visible, cache = visible_inputs(), load_cache()
    indices = np.asarray([row["global_index"] for row in ledger], np.int64)
    if len(indices) != 186 or len(set(indices.tolist())) != 186:
        raise ValueError("Integration must retain exactly 145+41 distinct requests")
    core_path = ROOT / "experiments/model_recommendation/frontier_core_145_20260921/inputs/queries.jsonl"
    clinical_path = ROOT / "experiments/model_recommendation/execution_validation_20261001/rankings/inputs/queries.jsonl"
    expected_ids = {r["query_id"] for r in [*rows(core_path), *rows(clinical_path)]}
    query_ids = [r["query_id"] for r in ledger]
    if set(query_ids) != expected_ids:
        raise ValueError("Integration request membership differs from the fixed comparison")
    for row, index in zip(ledger, indices):
        if row["query_id"] != visible["query_records"][index]["query_id"]:
            raise ValueError("Ledger row does not match its frozen global index")
        if row["original_text"] != visible["query_texts"][0][index]:
            raise ValueError("Original request text differs from the baseline")
    folds = exact_fold_membership(indices, old_protocol)
    candidate_ids = [r["candidate_id"] for r in visible["candidate_records"]]
    if len(candidate_ids) != 280 or candidate_ids != old_protocol["candidate_ids"]:
        raise ValueError("Full frozen candidate catalogue required")
    changed = np.asarray([r["augmented_query"]["text"] != r["original_text"] or
                          r["augmented_query"].get("features", {}) != visible["query_records"][i].get("features", {})
                          for r, i in zip(ledger, indices)], bool)
    features = visible["condition_features"][0, indices].copy()
    for j, index in enumerate(indices):
        if changed[j]:
            features[j] = soft_features(visible["query_records"][index], ledger[j]["augmented_query"],
                                       visible["candidate_records"], features[j])
    vectors = cache["light_query_embeddings"][0, indices].copy()
    texts = sorted({ledger[j]["augmented_query"]["text"] for j in np.flatnonzero(changed)})
    input_sources = [run_dir / "protocol.json", ledger_path, run_dir / "inputs/bridge_manifest.json",
                    run_dir / "components/manifest.json", OLD / "protocol.json",
                    OLD / "cache/manifest.json", OLD / "predictions/seed17.npz", Path(__file__),
                    ROOT / "src/pbpk_kb/recommendation/component_model_bridge.py"]
    bindings = [ref(path) for path in input_sources]
    for fold_id in range(5):
        directory = OLD / f"runs/seed17/fold{fold_id}/outer"
        bindings += [ref(directory / filename) for filename in ("heads.pt", "metadata.json", "complete.json")]
        bindings.append(ref(directory.parent / "selection.json"))
    binding = {"schema": "component-model-frozen-head-inference-v1", "sources": bindings,
               "query_ids": query_ids, "candidate_ids": candidate_ids,
               "texts_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()}
    existing = run_dir / "predictions/binding.json"
    if existing.exists() and read(existing) != binding:
        raise ValueError("Ranking inputs changed after freezing; use a new experiment")
    write(existing, binding)
    output = run_dir / "predictions/component_scores.npz"
    final_manifest = run_dir / "predictions/manifest.json"
    if final_manifest.exists():
        final = read(final_manifest)
        if final.get("state") == "complete" and final["predictions_sha256"] == sha(output):
            return final
        raise ValueError("Saved prediction receipt is incomplete or changed")
    encoding = {"new_texts": len(texts), "cached_candidate_texts": 280}
    if texts:
        encoding_dir = run_dir / "cache"
        encoding_dir.mkdir(parents=True, exist_ok=True)
        values = np.zeros((len(texts), vectors.shape[-1]), np.float16)
        completed = np.zeros(len(texts), bool)
        partial = encoding_dir / "embeddings.partial.npz"
        if partial.exists():
            with np.load(partial, allow_pickle=False) as saved:
                if saved["vectors"].shape != values.shape or saved["completed"].shape != completed.shape:
                    raise ValueError("Resumed component embeddings have incompatible shape")
                values, completed = saved["vectors"].copy(), saved["completed"].copy()
        missing = np.flatnonzero(~completed).tolist()
        if missing:
            encoding.update(_encode_missing(texts, missing, values, completed, encoding_dir,
                {"encoder": cache["binding"]["encoder"]}, device=device, batch_size=batch_size,
                callback=lambda e: progress("encoding_components", **{k: v for k, v in e.items() if k != "phase"})))
        if not completed.all() or not np.isfinite(values).all():
            raise ValueError("Incomplete component query embeddings")
        lookup = {text: i for i, text in enumerate(texts)}
        for j in np.flatnonzero(changed):
            vectors[j] = values[lookup[ledger[j]["augmented_query"]["text"]]].astype(np.float32)
        array_write(encoding_dir / "augmented_embeddings.npz", query_embeddings=vectors, changed=changed)
    with np.load(OLD / "predictions/seed17.npz", allow_pickle=False) as saved:
        results = {k: saved[k][0, indices].copy() for k in
                   ("scores", "base_scores", "probabilities", "selected_predicted_error", "error_supported")}
    # Only this input-derived endpoint-name map is read; the mixed dataset's
    # labels and per-candidate outcomes are never used by inference.
    requested = read(ROOT / "experiments/model_recommendation/pbpk_match_v2/dataset.json").get("execution_requested_endpoints", {})
    fold_refs, replay = [], []
    for fold_id in range(5):
        members = [j for j, i in enumerate(indices) if folds[int(i)] == fold_id]
        directory = OLD / f"runs/seed17/fold{fold_id}/outer"
        fold_refs.append({"fold": fold_id, "indices": [int(indices[j]) for j in members],
                          "checkpoint": ref(directory / "heads.pt"),
                          "selection": ref(directory.parent / "selection.json")})
        metadata = read(directory / "metadata.json")
        if metadata["input_binding"]["embedding_manifest"] != cache["light_embedding_manifest"]:
            raise ValueError("Frozen head and original text encoder cache do not match")
        # One prespecified unchanged-input replay per fold checks the inference
        # adapter against published predictions, before any component result.
        check = members[:1]
        if check:
            original = infer_fold(cache["light_query_embeddings"][0, indices[check]],
                visible["condition_features"][0, indices[check]], cache["light_candidate_embeddings"],
                [visible["query_records"][indices[j]] for j in check],
                [requested.get(str(indices[j])) for j in check], directory, device=device)
            checks = {}
            for key in ("scores", "base_scores", "probabilities"):
                left, right = results[key][check], original[key]
                if not np.allclose(left, right, atol=2e-5, rtol=2e-5):
                    raise ValueError("Frozen model replay differs from published output: " + key)
                if key != "probabilities" and not np.array_equal(np.argsort(-left, kind="stable"), np.argsort(-right, kind="stable")):
                    raise ValueError("Frozen model replay changed candidate order")
                checks[key] = float(np.max(np.abs(left - right)))
            replay.append({"fold": fold_id, "global_indices": [int(indices[j]) for j in check],
                           "maximum_absolute_differences": checks, "ranks_unchanged": True})
        selected = [j for j in members if changed[j]]
        if not selected:
            continue
        progress("ranking_components", fold=fold_id, requests=len(selected), candidates=280)
        value = infer_fold(vectors[selected], features[selected], cache["light_candidate_embeddings"],
            [visible["query_records"][indices[j]] for j in selected],
            [requested.get(str(indices[j])) for j in selected], directory, device=device)
        for key in results:
            results[key][selected] = value[key]
    for item in bindings:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("A frozen input changed during ranking")
    array_write(output, indices=indices, changed=changed, **results)
    manifest = {"state": "complete", "query_ids": query_ids, "candidate_ids": candidate_ids,
        "protocol_path": str((run_dir / "protocol.json").relative_to(ROOT)),
        "protocol_sha256": sha(run_dir / "protocol.json"),
        "baseline_protocol_sha256": sha(OLD / "protocol.json"), "predictions_sha256": sha(output),
        "source_bindings": bindings, "outer_folds": fold_refs,
        "original_request_semantics_preserved": True, "components_are_soft_requirements": True,
        "labels_used_as_features": False, "observations_used_as_features": False,
        "weights_retrained": False, "alpha_retuned": False, "temperature_refitted": False,
        "coverage": {"total_requests": len(indices), "changed_requests": int(changed.sum()),
                     "unchanged_requests": int((~changed).sum()),
                     "component_statuses": dict(Counter(r["status"] for r in ledger))},
        "embedding": encoding, "baseline_replay": replay,
        "elapsed_seconds": time.monotonic() - started}
    write(final_manifest, manifest)
    progress("ranking_complete", **manifest["coverage"], output=str(output.relative_to(ROOT)))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bridge", "rank"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "bridge":
        build_augmented_inputs(args.run_dir)
    else:
        run_inference(args.run_dir, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
