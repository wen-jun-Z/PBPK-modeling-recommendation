"""Structured new-request inference using frozen development checkpoints.

The catalog/cache and checkpoint guards are shared with the project's existing
demonstration entry point. New requests never supply labels, source judgments,
execution outcomes or a frozen benchmark query ID. Outputs remain research
recommendations, not verified PBPK applicability or simulation results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from .data import FIELDS, ROOT, TASKS, TYPES, fields_for, node_specs, scoped, sha
from .evaluation import apply_probability_calibrator
from .training import predict_neural


WEIGHT_REGISTRY = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/results/qwen3_embedding_06b/weights.json"
ENCODING = "last-token-l2-typed-fields-v1"
GPU_BUDGET_BYTES = 5 * 1024**3


def _checkpoint_helpers():
    """Load the explicit project helper, also outside a project-root cwd."""
    path = scoped(ROOT / "scripts/predict_pbpk_match.py")
    for name in ("scripts.predict_pbpk_match", "_pbpk_match_checkpoint_helpers"):
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, "__file__", "")).resolve() == path:
            return module
    name = "_pbpk_match_checkpoint_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load project checkpoint helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_request(request: dict) -> dict:
    """Strictly allow only the structured, observable request fields."""
    if not isinstance(request, dict) or set(request) - {"task_type", "compounds", "features"}:
        raise ValueError("Request accepts only task_type, compounds and features; labels/outcomes/provenance/IDs are forbidden")
    if request.get("task_type") not in TASKS:
        raise ValueError("task_type must be one of the frozen PBPK task types")
    features = request.get("features", {})
    if not isinstance(features, dict) or set(features) - set(FIELDS):
        raise ValueError("features must contain only the frozen PBPK metadata fields")
    def check_values(values, name):
        if not isinstance(values, list) or len(values) > 32 or any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in values):
            raise ValueError(f"{name} must be a list of at most 32 nonempty strings, each at most 256 characters")
    check_values(request.get("compounds"), "compounds")
    if not request["compounds"]:
        raise ValueError("At least one explicitly named compound is required")
    for name, values in features.items():
        check_values(values, f"features.{name}")
    clean = {"task_type": request["task_type"], "compounds": list(request["compounds"]), "features": dict(features)}
    normalized = fields_for(clean)
    if "compounds" in features and fields_for({"features": {"compounds": features["compounds"]}})["compounds"] != normalized["compounds"]:
        raise ValueError("Top-level compounds and features.compounds disagree")
    clean["compounds"] = normalized["compounds"]
    clean["features"] = {key: values for key, values in normalized.items() if key != "compounds" and values}
    if len(node_specs(clean)) > 128:
        raise ValueError("Request exceeds 128 typed nodes; simplify explicit metadata without silent truncation")
    return clean


def _verify_encoder_registry(manifest: dict, registry_path=WEIGHT_REGISTRY, *, verify_weights: bool = False) -> dict:
    registry_path = scoped(registry_path)
    registry = json.loads(registry_path.read_text())
    if (registry.get("repo") != "Qwen/Qwen3-Embedding-0.6B" or registry.get("repo") != manifest.get("repo")
            or registry.get("revision") != manifest.get("revision") or not re.fullmatch(r"[0-9a-f]{40}", str(registry.get("revision", "")))):
        raise ValueError("Encoder registry repo/revision differs from the frozen candidate embedding manifest")
    model_path = scoped(registry["path"])
    if not model_path.is_dir() or model_path.name != registry["revision"]:
        raise ValueError("Local encoder path is not its pinned project snapshot")
    if verify_weights:
        hashes = registry.get("sha256", {})
        required = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}
        if not required <= set(hashes):
            raise ValueError("Encoder registry lacks required local-file checksums")
        for name in sorted(required):
            if sha(scoped(model_path / name)) != hashes[name]:
                raise ValueError(f"Pinned local encoder file changed: {name}")
    return {**registry, "path": str(model_path), "registry_sha256": sha(registry_path)}


def encode_new_query(texts: list[str], manifest: dict, *, device="cuda:0", batch_size=4, registry_path=WEIGHT_REGISTRY):
    """Encode only request nodes, with a bounded project-owned GPU allocation."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not re.fullmatch(r"cuda(?::[0-9]+)?", device) or not torch.cuda.is_available():
        raise RuntimeError("New-query encoding requires the explicitly selected GPU; there is no implicit CPU fallback")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    registry = _verify_encoder_registry(manifest, registry_path, verify_weights=True)
    free, total = torch.cuda.mem_get_info(device)
    if free < GPU_BUDGET_BYTES or total < GPU_BUDGET_BYTES:
        raise RuntimeError("Less than 5 GiB free on the selected GPU; existing jobs are left untouched")
    torch.cuda.set_per_process_memory_fraction(GPU_BUDGET_BYTES / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    options = dict(local_files_only=True, token=False, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(registry["path"], padding_side="left", **options)
    if tokenizer.padding_side != "left":
        raise ValueError("Frozen embedding protocol requires left padding")
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
    if not lengths or min(lengths) < 1 or max(lengths) > 4096:
        raise ValueError("Every canonical node must have 1..4096 tokens; inputs are never truncated")
    model = None
    try:
        model = AutoModel.from_pretrained(registry["path"], torch_dtype=torch.bfloat16,
                                          attn_implementation="sdpa", **options).to(device).eval()
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        results = {}
        pos, size = 0, batch_size
        while pos < len(order):
            indices = order[pos:pos + size]
            encoded = hidden = values = None
            try:
                encoded = tokenizer([texts[i] for i in indices], padding=True, truncation=False, return_tensors="pt").to(device)
                if not bool(encoded["attention_mask"][:, -1].all()):
                    raise ValueError("Last-token pooling would read padding")
                with torch.inference_mode():
                    hidden = model(**encoded, use_cache=False).last_hidden_state[:, -1].float()
                    values = torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy()
                results.update(zip(indices, values))
                pos += len(indices)
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                size = max(1, size // 2)
            finally:
                del encoded, hidden, values
                torch.cuda.empty_cache()
        vectors = np.stack([results[i] for i in range(len(texts))]).astype(np.float16).astype(np.float32)
        return vectors, {"repo": registry["repo"], "revision": registry["revision"], "registry_sha256": registry["registry_sha256"],
                         "encoding": ENCODING, "encoder_trainable": False, "typed_node_count": len(texts),
                         "node_type_count": len(TYPES), "maximum_input_tokens": max(lengths), "truncated_inputs": 0,
                         "padding_side": "left", "embedding_storage_roundtrip": "float32 -> float16 -> float32",
                         "device": device, "gpu_memory_cap_gib": 5, "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                         "local_files_only": True, "uses_external_api": False}
    finally:
        del model
        torch.cuda.empty_cache()


def load_checkpoint(checkpoint):
    """Restore weights directly; no restriction to an existing request ID."""
    import torch
    from .model import PBPKMatch, PairMLP

    existing = _checkpoint_helpers()
    path, method_dir, output_root, protocol_path, report_path = existing._checkpoint_context(checkpoint)
    protocol = json.loads(protocol_path.read_text())
    report = json.loads(report_path.read_text())
    snapshot_path = scoped(output_root / "development_snapshot.json")
    snapshot_hash = sha(snapshot_path)
    if snapshot_hash != protocol.get("snapshot_sha256"):
        raise ValueError("Development snapshot differs from checkpoint protocol")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    kind = payload.get("model_kind") if isinstance(payload, dict) else None
    if (kind not in {"full", "mlp", "no_type", "no_null", "hybrid"} or kind not in protocol.get("methods", [])
            or payload.get("snapshot_sha256") != snapshot_hash or report.get("method") != kind
            or report.get("seed") not in protocol.get("seeds", []) or method_dir.name != f"{kind}_seed{report.get('seed')}"):
        raise ValueError("Checkpoint method, snapshot, seed or directory binding differs")
    verification = existing._verify_sources(protocol, kind)
    raw = existing._load_cached_data(output_root)
    manifest = dict(raw["embedding_manifest"])
    clean = existing.metadata_only_data(raw)
    del raw
    manifest_path = scoped(output_root / "cache/embedding_manifest.json")
    if manifest != json.loads(manifest_path.read_text()) or manifest["array_sha256"] != sha(output_root / "cache/frozen_embeddings.npz"):
        raise ValueError("Frozen embedding manifest/cache checksum differs")
    _verify_encoder_registry(manifest)
    dimensions = {"embedding_dim": clean["c_embeddings"].shape[-1], "pair_feature_dim": clean["pair_features"].shape[-1],
                  "num_types": clean["num_types"], "num_tasks": clean["num_tasks"]}
    if (any(payload.get(k) != v for k, v in dimensions.items()) or dimensions["embedding_dim"] != manifest["dimension"]
            or dimensions["pair_feature_dim"] != len(FIELDS) * 5 + 1 or dimensions["num_types"] != len(TYPES)
            or dimensions["num_tasks"] != len(TASKS)):
        raise ValueError("Checkpoint dimensions differ from the canonical frozen input schema")
    cids = [row["candidate_id"] for row in clean["candidate_records"]]
    if len(cids) != 280 or len(set(cids)) != 280:
        raise ValueError("Inference requires all 280 unique frozen candidates")
    config = payload.get("config", {})
    for name, default in (("hidden_dim", 128), ("dropout", .1)):
        if config.get(name, default) != protocol.get("config", {}).get(name, default):
            raise ValueError(f"Checkpoint {name} differs from run protocol")
    arguments = {**dimensions, "hidden_dim": config.get("hidden_dim", 128), "dropout": config.get("dropout", .1)}
    if kind == "hybrid":
        from .hybrid import PBPKMatchHybrid
        model = PBPKMatchHybrid(**arguments)
    else:
        model = PairMLP(**arguments) if kind == "mlp" else PBPKMatch(**arguments, ablation=kind)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    calibration = report.get("calibrator", {})
    if calibration.get("state") == "fitted":
        fit_ids = set(calibration.get("fit_compound_ids", []))
        excluded = set(calibration.get("excluded_compound_ids", [])) | set(calibration.get("model_fit_compound_ids", []))
        indices = report.get("partition", {}).get("calibration", [])
        if any(not isinstance(i, int) or i < 0 or i >= len(clean["query_records"]) for i in indices):
            raise ValueError("Invalid calibration query indices in fold report")
        allowed = {clean["query_records"][i]["compound_id"] for i in indices}
        if not fit_ids or fit_ids & excluded or not fit_ids <= allowed:
            raise ValueError("Fold calibration is not held out from model fitting and evaluation")
    candidates = {key: clean[key] for key in ("c_embeddings", "c_types", "c_mask", "candidate_records", "num_types", "num_tasks")}
    context = {"checkpoint": str(path), "checkpoint_sha256": sha(path), "protocol_sha256": sha(protocol_path),
               "fold_report_sha256": sha(report_path),
               "snapshot_sha256": snapshot_hash, "embedding_manifest_sha256": sha(manifest_path),
               "embedding_manifest": manifest, "model_kind": kind, "calibrator": calibration, "source_verification": verification,
               "checkpoint_embedding_binding": "Frozen cache checksum and registry revision verified; legacy checkpoint has no separate embedding-manifest hash"}
    return model, candidates, context


def build_request_data(request: dict, candidates: dict, vectors: np.ndarray) -> dict:
    """Apply precisely the frozen 61-feature definition to one new request."""
    clean = validate_request(request)
    nodes = node_specs(clean)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(nodes), candidates["c_embeddings"].shape[-1]) or not np.isfinite(vectors).all():
        raise ValueError("Encoder returned invalid request-node embeddings")
    # Candidate cache stores float16 embeddings decoded to float32. Match that
    # boundary for injected encoders as well as the default local implementation.
    vectors = vectors.astype(np.float16).astype(np.float32)
    qe = vectors[None, None, :, :]
    qt = np.asarray([kind for kind, _ in nodes], dtype=np.int64)[None, None, :]
    identifier = "structured-request:" + _digest_json(clean)
    record = {**clean, "query_id": identifier, "compound_id": identifier, "text": nodes[0][1]}
    qf = fields_for(clean)
    pair = np.zeros((1, 1, len(candidates["candidate_records"]), len(FIELDS) * 5 + 1), dtype=np.float32)
    for ci, candidate in enumerate(candidates["candidate_records"]):
        cf, values = fields_for(candidate, candidate=True), []
        for field in FIELDS:
            a, b = set(qf[field]), set(cf[field])
            shared = len(a & b)
            values.extend([bool(a), bool(b), shared / max(1, len(a | b)), shared / max(1, len(a)), bool(a and b and not shared)])
        values.append(float(vectors[0] @ candidates["c_embeddings"][ci, 0]))
        pair[0, 0, ci] = values
    # Construct an allowlist; a caller's accidental label keys cannot propagate.
    data = {**{key: candidates[key] for key in ("c_embeddings", "c_types", "c_mask", "candidate_records", "num_types", "num_tasks")},
            "q_embeddings": qe, "q_types": qt, "q_mask": np.ones(qt.shape, dtype=bool), "pair_features": pair,
            "task_ids": np.asarray([TASKS.index(clean["task_type"])]), "query_records": [record], "variant_names": ["original"]}
    return _checkpoint_helpers().metadata_only_data(data)


def _catalog_field_gaps(request: dict, candidate: dict) -> dict:
    result = _checkpoint_helpers().describe_field_gaps(request, candidate)
    requested, available = fields_for(request), fields_for(candidate, candidate=True)
    for name in FIELDS:
        if name in {"compounds", "species", "routes", "populations", "endpoints"} or not requested[name]:
            continue
        missing = sorted(set(requested[name]) - set(available[name]))
        if missing:
            result["fields"].append({"field": name, "requested_values": requested[name], "catalog_values": available[name],
                                     "not_listed_values": missing, "state": "not_recorded_in_catalog" if not available[name] else "requested_values_not_listed"})
    result["checked_fields"] = list(FIELDS)
    return result


def recommend_request(checkpoint, request: dict, top_k=10, *, device="cuda:0", batch_size=4, chunk_size=64, encoder: Callable | None = None) -> dict:
    """Rank the full catalog for a new structured request; never fit a model."""
    from .policy import catalog_identity_policy, select_policy_topk

    clean_request = validate_request(request)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be positive")
    model, candidates, context = load_checkpoint(checkpoint)
    texts = [text for _, text in node_specs(clean_request)]
    vectors, encoder_metadata = (encoder or encode_new_query)(texts, context["embedding_manifest"], device=device, batch_size=batch_size)
    if (encoder_metadata.get("repo") != context["embedding_manifest"]["repo"]
            or encoder_metadata.get("revision") != context["embedding_manifest"]["revision"]
            or encoder_metadata.get("encoding") != ENCODING):
        raise ValueError("New request encoder differs from the frozen candidate encoder protocol")
    data = build_request_data(clean_request, candidates, vectors)
    scores, probabilities = predict_neural(model, data, [0], device=device, chunk_size=chunk_size)
    calibration = context["calibrator"]
    calibration_status = "uncalibrated_development_probabilities"
    if calibration.get("state") == "fitted":
        probabilities = apply_probability_calibrator(np.log(np.clip(probabilities, 1e-12, 1)), calibration)
        calibration_status = "checkpoint_fold_independent_calibration_applied"
    if (scores.shape != (1, 280) or probabilities.shape != (1, 280, 3) or not np.isfinite(scores).all()
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(-1), 1., rtol=0., atol=1e-7)):
        raise ValueError("Incomplete or nonfinite full-catalog prediction")
    scores, probabilities = scores[0], probabilities[0]
    order = sorted(range(280), key=lambda i: (-float(scores[i]), data["candidate_records"][i]["candidate_id"]))
    threshold = float(np.sort(scores)[-min(top_k, 280)])
    ranking = []
    for i in order:
        candidate, score = data["candidate_records"][i], float(scores[i])
        higher, tied = int((scores > score).sum()), int((scores == score).sum())
        ranking.append({"candidate_id": candidate["candidate_id"], "family_id": candidate.get("family_id"),
                        "family_name": candidate.get("family_name"), "compounds": candidate.get("compounds", []),
                        "rank_min": higher + 1, "rank_max": higher + tied, "score_tie_size": tied, "score": score,
                        "probabilities": dict(zip(("Y", "N", "U"), probabilities[i].tolist())), "status": "uncertain",
                        "scientific_applicability_verified": False, "field_gaps": _catalog_field_gaps(clean_request, candidate)})
    raw_selected = [row for row in ranking if row["score"] >= threshold]
    policy = catalog_identity_policy(clean_request, data["candidate_records"])
    policy_selection = select_policy_topk(scores, policy, k=top_k)
    raw_by_id = {row["candidate_id"]: row for row in ranking}
    selected = []
    for decision in policy_selection["all_candidates"]:
        raw = raw_by_id[decision["candidate_id"]]
        raw["policy_eligible"] = decision["eligible"]
        raw["policy_eligibility_status"] = decision["status"]
        raw["policy_reason_code"] = decision["reason_code"]
        raw["policy_reason"] = decision["reason"]
        raw["raw_rank"] = decision["raw_rank"]
        raw["raw_rank_end"] = decision["raw_rank_end"]
        raw["policy_rank"] = decision["policy_rank"]
        raw["policy_rank_end"] = decision["policy_rank_end"]
    for decision in policy_selection["items"]:
        row = dict(raw_by_id[decision["candidate_id"]])
        p = row["probabilities"]
        y_leading = p["Y"] > max(p["N"], p["U"])
        row.update(rank_min=decision["policy_rank"], rank_max=decision["policy_rank_end"],
                   score_tie_size=decision["policy_rank_end"] - decision["policy_rank"] + 1,
                   status="needs_review", policy_eligible=True,
                   recommendation_status="unverified_applicability_candidate" if y_leading else "retrieval_candidate_needs_review",
                   model_applicability_y_leading=y_leading,
                   applicability_interpretation="Y has the largest predicted probability; independent applicability review is still required" if y_leading else
                   "Y is not the largest predicted probability; this is a catalog retrieval candidate for review, not a confirmed applicable model")
        selected.append(row)
    selected.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    abstain = bool(policy_selection["abstain"])
    eligible_count = int(sum(policy["eligible_mask"]))
    catalog_compounds = {name for row in data["candidate_records"] for name in fields_for(row, candidate=True)["compounds"]}
    return {"status": "needs_catalog_review" if abstain else "needs_review", "version": "identity-policy-v1",
            "mode": "new_structured_request_development_checkpoint", "production_recommendation": False,
            "abstain": abstain, "needs_catalog_review": abstain,
            "recommendation_decision": "abstain" if abstain else "retrieve_for_review",
            "abstention_reason": policy.get("abstention_reason"), "policy_eligible_candidate_count": eligible_count,
            "identity_policy": {"policy_version": policy["policy_version"], "scope": policy["scope"], "eligible_candidate_count": eligible_count,
                                "excluded_candidate_count": 280 - eligible_count, "metadata_only": True,
                                "interpretation": "Catalog identity policy is separate from the learned ranker; it does not establish scientific applicability"},
            "scientific_gold": False, "pk_simulation_executed": False, "device": device, "model_kind": context["model_kind"],
            "checkpoint": context["checkpoint"], "checkpoint_sha256": context["checkpoint_sha256"],
            "protocol_sha256": context["protocol_sha256"], "snapshot_sha256": context["snapshot_sha256"],
            "fold_report_sha256": context["fold_report_sha256"],
            "embedding_manifest_sha256": context["embedding_manifest_sha256"], "checkpoint_embedding_binding": context["checkpoint_embedding_binding"],
            "source_verification": context["source_verification"], "serving_source_sha256": sha(Path(__file__)),
            "policy_source_sha256": sha(Path(__file__).with_name("policy.py")),
            "input_sha256": _digest_json(request), "canonical_input_sha256": _digest_json(clean_request), "request": clean_request,
            "query_id": data["query_records"][0]["query_id"], "encoder": encoder_metadata, "candidate_count": 280,
            "requested_top_k": top_k, "top_k_boundary_score": selected[-1]["score"] if selected else None,
            "top_k_with_boundary_ties_count": len(selected), "raw_top_k_boundary_score": threshold,
            "raw_top_k_with_boundary_ties_count": len(raw_selected), "raw_top_k": raw_selected,
            "tie_policy": "All exact boundary-score ties retained; candidate IDs order display only",
            "probability_calibration": calibration_status, "calibration_transfer_to_new_request_verified": False,
            "calibration_metadata": {key: calibration.get(key) for key in ("state", "method", "temperature", "fit_compound_ids")},
            "probability_meaning": "Learned agreement with automatic documented Y/N/U judgments, not verified scientific suitability",
            "query_compounds_not_exactly_named_in_catalog": sorted(set(clean_request["compounds"]) - catalog_compounds),
            "catalog_gap_semantics": "Missing catalog names/fields require source review; they do not prove scientific incompatibility",
            "label_usage": "No labels, outcomes or source-judgment rows enter prediction; all 280 raw scores retained; displayed recommendations additionally apply the declared catalog identity policy",
            "top_k_with_boundary_ties": selected, "ranking": ranking}
