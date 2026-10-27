"""CPU demonstration of a frozen PBPK-Match development checkpoint.

Uses existing project embedding caches only. No training, service, download,
candidate exclusion or PBPK simulation is performed. Learned Y/N/U probabilities
are not verified scientific applicability decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from pbpk_kb.recommendation.pbpk_match.data import build_data, fields_for, scoped, sha
from pbpk_kb.recommendation.pbpk_match.evaluation import apply_probability_calibrator
from pbpk_kb.recommendation.pbpk_match.training import predict_neural

INFERENCE_SOURCE_FILES = (
    "src/pbpk_kb/recommendation/pbpk_match/model.py",
    "src/pbpk_kb/recommendation/pbpk_match/data.py",
    "src/pbpk_kb/recommendation/pbpk_match/training.py",
)
ARRAY_KEYS = (
    "q_embeddings", "q_types", "q_mask", "c_embeddings", "c_types", "c_mask",
    "pair_features", "task_ids", "variant_names", "num_types", "num_tasks",
)


def _load_cached_data(output_root):
    """Do not let the training loader create missing caches or a new snapshot."""
    for name in ("development_snapshot.json", "cache/frozen_embeddings.npz", "cache/embedding_manifest.json"):
        if not scoped(output_root / name).is_file():
            raise FileNotFoundError(f"Existing development cache required: {name}; inference will not build it")
    return build_data(output=output_root, device="cpu")


def metadata_only_data(data):
    """Allowlist the prediction contract; no grade/app/snapshot enters predictor."""
    result = {key: data[key] for key in ARRAY_KEYS}
    result["query_records"] = [
        {"query_id": row["query_id"], "compound_id": row["compound_id"],
         "compounds": list(row.get("compounds", [])), "task_type": row["task_type"],
         "features": fields_for(row), "text": row.get("text", "")}
        for row in data["query_records"]]
    result["candidate_records"] = [
        {"candidate_id": row["candidate_id"], "family_id": row.get("family_id"),
         "family_name": row.get("family_name"), "compounds": list(row.get("compounds", [])),
         "features": fields_for(row, candidate=True)}
        for row in data["candidate_records"]]
    return result


def describe_field_gaps(query, candidate):
    """Show catalog omissions, never infer incompatibility from an absent word."""
    requested, available = fields_for(query), fields_for(candidate, candidate=True)
    gaps = []
    for name in ("compounds", "species", "routes", "populations", "endpoints"):
        if not requested[name]:
            continue
        missing = sorted(set(requested[name]) - set(available[name]))
        if missing:
            gaps.append({"field": name, "requested_values": requested[name],
                         "catalog_values": available[name], "not_listed_values": missing,
                         "state": "not_recorded_in_catalog" if not available[name] else "requested_values_not_listed"})
    return {"metadata_only": True, "fields": gaps,
            "interpretation": "Catalog omissions require source review; absent values and different template-drug identities do not by themselves establish incompatibility"}


def _checkpoint_context(checkpoint):
    path = scoped(checkpoint)
    if not path.is_file() or path.suffix != ".pt":
        raise ValueError("checkpoint must be an existing project .pt file")
    # Training writes <output>/runs/<run>/<method_seed>/<fold>/model.pt.
    fold_dir, method_dir, run_dir = path.parent, path.parent.parent, path.parent.parent.parent
    if run_dir.parent.name != "runs":
        raise ValueError("Expected a frozen runs/<run>/<method>/<fold>/model.pt checkpoint")
    output_root = scoped(run_dir.parent.parent)
    protocol_path = scoped(run_dir / "protocol.json")
    report_path = scoped(fold_dir / "report.json")
    if not protocol_path.is_file() or not report_path.is_file():
        raise ValueError("Checkpoint requires its frozen run protocol and completed fold report")
    return path, method_dir, output_root, protocol_path, report_path


def _verify_sources(protocol, model_kind):
    source_hashes = protocol.get("source_sha256", {})
    required = list(INFERENCE_SOURCE_FILES[:2])
    if model_kind == "hybrid":
        required.append("src/pbpk_kb/recommendation/pbpk_match/hybrid.py")
    bindings = {}
    for name in required:
        expected = source_hashes.get(name)
        actual = sha(scoped(ROOT / name)) if scoped(ROOT / name).is_file() else None
        if not expected or actual != expected:
            raise ValueError(f"Frozen inference dependency differs from protocol: {name}; use a matching checkpoint/source version")
        bindings[name] = {"frozen_sha256": expected, "current_sha256": actual, "state": "verified"}
    # Training/early-stopping edits need not change a restored architecture.
    # Record the current predictor module explicitly; strict state loading and
    # hard architecture/preprocessing checks remain mandatory.
    name = INFERENCE_SOURCE_FILES[2]
    expected, actual = source_hashes.get(name), sha(ROOT / name)
    if not expected:
        raise ValueError("Frozen protocol lacks the original training/predictor module identity")
    changed = expected != actual
    bindings[name] = {"frozen_sha256": expected, "current_sha256": actual,
                      "state": "implementation_changed" if changed else "verified"}
    return {"files": bindings, "predictor_module_changed": changed,
            "policy": "Architecture and preprocessing hashes must match; training/predictor updates are disclosed with strict checkpoint state loading",
            "warning": "Current predictor module differs from training time; architecture and cached preprocessing remain verified, but bitwise historical prediction identity is not asserted" if changed else None}


def predict_checkpoint(checkpoint, query_id, top_k=10):
    """Restore a development checkpoint and rank every frozen candidate on CPU."""
    import torch
    from pbpk_kb.recommendation.pbpk_match.model import PBPKMatch, PairMLP

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    path, method_dir, output_root, protocol_path, report_path = _checkpoint_context(checkpoint)
    protocol = json.loads(protocol_path.read_text())
    fold_report = json.loads(report_path.read_text())
    snapshot_path = scoped(output_root / "development_snapshot.json")
    if not snapshot_path.is_file():
        raise ValueError("Frozen development snapshot is missing")
    snapshot_sha = sha(snapshot_path)
    if snapshot_sha != protocol.get("snapshot_sha256"):
        raise ValueError("Development snapshot differs from run protocol")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("snapshot_sha256") != snapshot_sha:
        raise ValueError("Checkpoint snapshot binding differs")
    kind = payload.get("model_kind")
    if kind not in {"full", "mlp", "no_type", "no_null", "hybrid"} or kind not in protocol.get("methods", []):
        raise ValueError("Unsupported or undeclared neural checkpoint method")
    source_verification = _verify_sources(protocol, kind)
    if fold_report.get("method") != kind or fold_report.get("seed") not in protocol.get("seeds", []):
        raise ValueError("Fold method/seed differs from checkpoint protocol")
    if method_dir.name != f"{kind}_seed{fold_report['seed']}":
        raise ValueError("Checkpoint directory does not match method/seed provenance")
    data = _load_cached_data(output_root)
    raw_snapshot = data.get("snapshot", {})
    allowed = set(raw_snapshot.get("development_query_ids", []))
    if query_id not in allowed or query_id in set(raw_snapshot.get("test_query_ids_excluded", [])):
        raise ValueError("Only frozen development query IDs are accepted; locked test queries remain unavailable")
    clean = metadata_only_data(data)
    del data
    qids = [row["query_id"] for row in clean["query_records"]]
    cids = [row["candidate_id"] for row in clean["candidate_records"]]
    if len(set(qids)) != len(qids) or len(set(cids)) != len(cids) or query_id not in qids:
        raise ValueError("Frozen input identities are not unique or the requested query is missing")
    if len(cids) != 280:
        raise ValueError("This development protocol requires all 280 frozen candidates")
    index = qids.index(query_id)
    actual_dimensions = {"embedding_dim": clean["q_embeddings"].shape[-1],
                         "pair_feature_dim": clean["pair_features"].shape[-1],
                         "num_types": clean["num_types"], "num_tasks": clean["num_tasks"]}
    if any(payload.get(name) != value for name, value in actual_dimensions.items()):
        raise ValueError("Checkpoint dimensions differ from frozen cached inputs")
    if clean["c_embeddings"].shape[-1] != actual_dimensions["embedding_dim"]:
        raise ValueError("Query/candidate embedding dimensions differ")
    config = payload.get("config", {})
    for name, default in (("hidden_dim", 128), ("dropout", .1)):
        if config.get(name, default) != protocol.get("config", {}).get(name, default):
            raise ValueError(f"Checkpoint {name} differs from run protocol")
    arguments = {**actual_dimensions, "hidden_dim": config.get("hidden_dim", 128), "dropout": config.get("dropout", .1)}
    if kind == "hybrid":
        from pbpk_kb.recommendation.pbpk_match.hybrid import PBPKMatchHybrid
        model = PBPKMatchHybrid(**arguments)
    else:
        model = PairMLP(**arguments) if kind == "mlp" else PBPKMatch(**arguments, ablation=kind)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    # No data label keys or source-judgment rows are available to this call.
    scores, raw_probabilities = predict_neural(model, clean, [index], device="cpu")
    scores = np.asarray(scores, dtype=float)
    probabilities = np.asarray(raw_probabilities, dtype=float)
    calibration = fold_report.get("calibrator", {})
    calibration_status = "uncalibrated_development_probabilities"
    if calibration.get("state") == "fitted":
        probabilities = apply_probability_calibrator(np.log(np.clip(probabilities, 1e-12, 1)), calibration)
        calibration_status = "inner_development_temperature_applied"
    if (scores.shape != (1, 280) or probabilities.shape != (1, 280, 3)
            or not np.isfinite(scores).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-7)):
        raise ValueError("Incomplete, invalid or unnormalized prediction")
    scores, probabilities = scores[0], probabilities[0]
    threshold = float(np.sort(scores)[-min(top_k, len(scores))])
    # Candidate ID gives a display order within ties only; identical rank ranges
    # and inclusive boundary sets preserve the unresolved scientific preference.
    order = sorted(range(len(cids)), key=lambda i: (-float(scores[i]), cids[i]))
    ranking = []
    query = clean["query_records"][index]
    for i in order:
        candidate = clean["candidate_records"][i]
        score = float(scores[i])
        higher, tied = int((scores > score).sum()), int((scores == score).sum())
        ranking.append({"candidate_id": candidate["candidate_id"], "family_id": candidate.get("family_id"),
                        "family_name": candidate.get("family_name"), "compounds": candidate.get("compounds", []),
                        "rank_min": higher + 1, "rank_max": higher + tied, "score_tie_size": tied,
                        "score": score, "probabilities": dict(zip(("Y", "N", "U"), probabilities[i].tolist())),
                        "status": "uncertain", "scientific_applicability_verified": False,
                        "field_gaps": describe_field_gaps(query, candidate)})
    selected = [row for row in ranking if row["score"] >= threshold]
    partition = fold_report.get("partition", {})
    roles = [name for name in ("outer", "calibration", "refit") if index in partition.get(name, [])]
    if len(roles) != 1:
        raise ValueError("Query role is missing or ambiguous in the frozen fold partition")
    return {
        "status": "uncertain", "mode": "development_checkpoint_demonstration",
        "production_recommendation": False, "pk_simulation_executed": False, "scientific_gold": False,
        "device": "cpu", "checkpoint": str(path), "checkpoint_sha256": sha(path),
        "protocol_sha256": sha(protocol_path), "snapshot_sha256": snapshot_sha,
        "source_verification": source_verification,
        "model_kind": kind, "query_id": query_id, "query_role_in_checkpoint": roles[0],
        "query": query, "candidate_count": len(ranking), "requested_top_k": top_k,
        "top_k_boundary_score": threshold, "top_k_with_boundary_ties_count": len(selected),
        "tie_policy": "All candidates at the TopK boundary retained; IDs order display only, not preference",
        "probability_calibration": calibration_status,
        "probability_meaning": "Learned agreement with automatic documented Y/N/U judgments, not verified scientific suitability",
        "label_usage": "No grades, applicability targets or snapshot labels passed to prediction; all candidates scored without label filtering",
        "top_k_with_boundary_ties": selected, "ranking": ranking,
    }


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Optional new project JSON file containing all 280 candidates")
    args = parser.parse_args(argv)
    destination = scoped(args.output) if args.output is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError("Output already exists; choose a new demonstration result path")
    torch.set_num_threads(2)
    result = predict_checkpoint(args.checkpoint, args.query_id, args.top_k)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    brief = {key: result[key] for key in ("status", "mode", "query_id", "query_role_in_checkpoint", "candidate_count",
                                        "requested_top_k", "top_k_with_boundary_ties_count", "probability_calibration")}
    brief["output"] = str(destination) if destination is not None else None
    brief["predictor_module_changed"] = result["source_verification"]["predictor_module_changed"]
    brief["implementation_warning"] = result["source_verification"]["warning"]
    brief["preview"] = [{key: row[key] for key in ("candidate_id", "rank_min", "rank_max", "score", "probabilities", "status")}
                        for row in result["top_k_with_boundary_ties"][:10]]
    brief["preview_is_truncated"] = len(result["top_k_with_boundary_ties"]) > 10
    print(json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
