"""Extract existing measured PK outcomes without inventing recommendation labels.

Uses only project-local, pinned audit files. Historical audit acceptance is retained
as an audit result, never promoted to an independent recommendation judgment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "experiments/model_recommendation"
BENCHMARK = f"{BASE}/runs/expanded_catalog_20260913"
DEFAULT_OUTPUT = f"{BASE}/frontier_comparison_20260914/observed_outcomes.json"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Inputs:
    def __init__(self):
        self.sources = {}

    def path(self, relative):
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)  # Never follow legacy paths outside this project.
        return path

    def read(self, relative, jsonl=False):
        path = self.path(relative)
        self.sources[str(relative)] = {"path": str(relative), "sha256": digest(path)}
        value = path.read_text(encoding="utf-8-sig")
        return ([json.loads(line) for line in value.splitlines() if line.strip()]
                if jsonl else json.loads(value))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def reference_identity(reference):
    raw = str(reference or "").strip()
    pmids = re.findall(r"(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/|PMID[:\s]+)(\d+)", raw, re.I)
    dois = re.findall(r"10\.\d{4,9}/[^\s<>\"]+", raw, re.I)
    return {"raw_reference": raw or None, "pmids": sorted(set(pmids)), "dois": sorted(set(dois)),
            "study_id": None, "independence_verified": False,
            "status": "reference_present_not_adjudicated" if raw else "reference_missing"}


def series_context(protocol, row):
    matches = []
    for simulation in protocol.get("simulations", []):
        if simulation.get("sequence") != row.get("simulation_sequence"):
            continue
        for series in [simulation.get("evaluation_series"), *simulation.get("additional_evaluation_series", [])]:
            if isinstance(series, dict) and series.get("dataset_name") == row.get("dataset_name"):
                matches.append((simulation, series))
    if len(matches) != 1:
        return {"status": "missing_or_ambiguous_protocol_series", "match_count": len(matches)}
    simulation, series = matches[0]
    return {"status": "exact_protocol_series", "role": simulation.get("role"),
            "simulation_name": simulation.get("name"), "individual": simulation.get("individual"),
            "output_path": series.get("output_path"), "observed_unit": series.get("observed_unit"),
            "molecular_weight_g_per_mol": series.get("molecular_weight_g_per_mol"),
            "time_unit": series.get("time_unit"), "time_values": series.get("time_values"),
            "observed_values": series.get("observed_values"), "scaling": series.get("scaling"),
            "reference": series.get("reference"), "source_location": series.get("source_location")}


def ratio_metrics(row):
    output = {}
    for original, name in [("auc0_last_ratio", "auc_observed_window_ratio"), ("cmax_ratio", "sampled_cmax_ratio")]:
        value = row.get(original)
        output[name] = value
        output[name.replace("_ratio", "_absolute_log_error")] = abs(math.log(value)) if finite_positive(value) else None
        output[name.replace("_ratio", "_absolute_relative_error")] = abs(value - 1) if finite_positive(value) else None
    output.update({key: row.get(key) for key in ["gmfe", "geometric_bias", "log_rmse", "two_fold_fraction"]})
    return output


def absolute_endpoints(context, row):
    """Recover observed window endpoints; predicted values derive from audited ratios."""
    times, values = context.get("time_values"), context.get("observed_values")
    unit, time_unit = context.get("observed_unit"), context.get("time_unit")
    mw = context.get("molecular_weight_g_per_mol")
    factor = None
    if unit == "µmol/l":
        factor = 1e-6
    elif finite_positive(mw) and unit in {"mg/l", "µg/ml", "ng/ml", "µg/l"}:
        factor = (1e-3 if unit in {"mg/l", "µg/ml"} else 1e-6) / mw
    if factor is None or time_unit != "h" or not times or not values or len(times) != len(values):
        return {"status": "unit_or_time_grid_unavailable"}
    ordered = sorted(zip(times, [v * factor for v in values], strict=True))
    observed_auc = sum((tb - ta) * (ca + cb) / 2 for (ta, ca), (tb, cb) in zip(ordered, ordered[1:]))
    observed_cmax = max(v for _, v in ordered)
    auc_ratio, cmax_ratio = row.get("auc0_last_ratio"), row.get("cmax_ratio")
    return {"status": "observed_values_recomputed_predictions_derived_from_audited_ratios",
            "observed_auc_in_observed_window": observed_auc if len(times) >= 2 else None,
            "predicted_auc_in_observed_window": observed_auc * auc_ratio if finite_positive(auc_ratio) else None,
            "auc_unit": "mol*h/l", "observed_sampled_cmax": observed_cmax,
            "predicted_sampled_cmax": observed_cmax * cmax_ratio if finite_positive(cmax_ratio) else None,
            "cmax_unit": "mol/l"}


def verify_curve_metrics(row):
    observed = row.get("positive_observed_mol_per_litre", [])
    predicted = row.get("positive_predicted_mol_per_litre", [])
    if len(observed) != row.get("positive_pair_count") or len(predicted) != len(observed):
        raise ValueError("Curve length differs from auditor positive-pair count")
    if not observed or not all(finite_positive(x) for x in [*observed, *predicted]):
        raise ValueError("Historical positive curve contains nonpositive or nonfinite values")
    logs = [math.log(p / o) for o, p in zip(observed, predicted, strict=True)]
    computed = {"gmfe": math.exp(sum(abs(v) for v in logs) / len(logs)),
                "geometric_bias": math.exp(sum(logs) / len(logs)),
                "log_rmse": math.sqrt(sum(v * v for v in logs) / len(logs)),
                "two_fold_fraction": sum(abs(v) <= math.log(2) for v in logs) / len(logs)}
    if not all(math.isclose(value, row[key], rel_tol=1e-12, abs_tol=1e-12) for key, value in computed.items()):
        raise ValueError("Recomputed curve metrics differ from historical audit")
    return {"recomputed_from_positive_pairs": True, "relative_tolerance": 1e-12, "absolute_tolerance": 1e-12}


def extract(output):
    inputs = Inputs()
    readiness = inputs.read(f"{BASE}/scientific_evaluation/readiness.json")
    models = inputs.read(f"{BENCHMARK}/benchmark/model_records.jsonl", True)
    queries = inputs.read(f"{BENCHMARK}/benchmark/queries.jsonl", True)
    checks = inputs.read(f"{BENCHMARK}/benchmark_verification.json")["catalog"]["source_checks"]
    checks_by_id = {row["candidate_id"]: row for row in checks}
    specifications = []
    for key, protocol_root, manifest_root in [
        ("historical_windows_audits", "artifacts/pbpk-model-assets/windows_snapshot_batches/run_20260822_v4",
         "artifacts/pbpk-model-assets/windows_execution_results/run_20260822_v4"),
        ("controlled_adaptations", f"{BASE}/runs/controlled_adaptations_windows_batch_run_20260823_v7",
         f"{BASE}/runs/controlled_adaptations_windows_results_run_20260823_v7/results"),
    ]:
        index = inputs.read(readiness[key]["source_index_path"], True)
        for summary, indexed in zip(readiness[key]["protocols"], index, strict=True):
            if summary["protocol_key"] != indexed["protocol_key"]:
                raise ValueError("Audit index order differs from pinned readiness inventory")
            specifications.append({**summary, "cohort": key,
                "protocol_path": f"{protocol_root}/{indexed['protocol_relative_path']}",
                "manifest_path": f"{manifest_root}/{summary['protocol_key']}/windows_run_manifest.json"})
    # This earlier run is the exact registered Rosuvastatin .pksim5 artifact.
    # Its observed series overlap the later batch and are not additional studies.
    specifications.append({"protocol_key": "registered_rosuvastatin_20260801", "cohort": "registered_rosuvastatin",
        "compounds": ["rosuvastatin"], "repository_or_adaptation": "Open-Systems-Pharmacology/Rosuvastatin-Model",
        "audit_path": "experiments/rosuvastatin_reuse/audit_20260801T124000Z/snapshot_reuse_audit.json",
        "metrics_path": "experiments/rosuvastatin_reuse/audit_20260801T124000Z/verification_dataset_metrics.jsonl",
        "protocol_path": "experiments/rosuvastatin_reuse/protocol_20260801T000000Z/snapshot_reuse_protocol.json",
        "manifest_path": "experiments/rosuvastatin_reuse/windows_results_v3/windows_run_manifest.json"})
    protocols, outcomes, exact_candidates = [], [], set()
    for specification in specifications:
        audit = inputs.read(specification["audit_path"])
        rows = inputs.read(specification["metrics_path"], True)
        protocol = inputs.read(specification["protocol_path"])
        manifest = inputs.read(specification["manifest_path"]) if inputs.path(specification["manifest_path"]).is_file() else None
        protocol_sha = inputs.sources[specification["protocol_path"]]["sha256"]
        if audit.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"Protocol hash differs from audit: {specification['protocol_key']}")
        if audit.get("verification_dataset_metrics_sha256") != inputs.sources[specification["metrics_path"]]["sha256"]:
            raise ValueError("Dataset metrics hash differs from audit")
        if manifest and audit.get("windows_manifest_sha256") != inputs.sources[specification["manifest_path"]]["sha256"]:
            raise ValueError("Windows manifest hash differs from audit")
        pinned_hashes = {protocol.get("source", {}).get("snapshot_sha256")}
        if manifest:
            pinned_hashes.update(manifest.get(key) for key in ["input_snapshot_sha256", "reconstructed_project_sha256", "roundtrip_snapshot_sha256"])
        pinned_hashes.discard(None)
        matches, provisional, match_basis = [], [], {}
        for model in models:
            cid = model["model_version_id"]
            check = checks_by_id[cid]
            if check["expected_sha256"] in pinned_hashes:
                source_path = inputs.path(check["resolved_source_path"])
                actual_hash = digest(source_path)
                if actual_hash != check["expected_sha256"]:
                    raise ValueError(f"Current candidate artifact hash mismatch: {cid}")
                inputs.sources[check["resolved_source_path"]] = {"path": check["resolved_source_path"], "sha256": actual_hash}
                matches.append(cid)
                exact_candidates.add(cid)
                match_basis[cid] = {"source_sha256": actual_hash, "current_source_path": check["resolved_source_path"],
                    "matches_protocol_snapshot": actual_hash == protocol.get("source", {}).get("snapshot_sha256"),
                    "matches_executed_project": bool(manifest and actual_hash == manifest.get("reconstructed_project_sha256")),
                    "matches_executed_roundtrip_snapshot": bool(manifest and actual_hash == manifest.get("roundtrip_snapshot_sha256"))}
            elif model.get("repository_commit") == protocol.get("source", {}).get("commit_sha"):
                provisional.append(cid)
        base = {"protocol_key": specification["protocol_key"], "cohort": specification["cohort"],
                "repository_or_adaptation": specification["repository_or_adaptation"],
                "compounds": specification["compounds"], "exact_candidate_ids": matches,
                "same_commit_unverified_candidate_ids": provisional,
                "exact_candidate_match_basis": match_basis,
                "audit_path": specification["audit_path"], "protocol_path": specification["protocol_path"],
                "metrics_path": specification["metrics_path"], "protocol_sha256": protocol_sha,
                "source_snapshot_sha256": protocol.get("source", {}).get("snapshot_sha256"),
                "source_snapshot_semantic_sha256": protocol.get("source_snapshot_semantic_sha256"),
                "engineering_accepted": audit.get("engineering_accepted"),
                "original_scientific_accepted": audit.get("scientific_accepted"),
                "independence_verified": False, "independent_recommendation_judgment": None,
                "dataset_errors": audit.get("dataset_errors", []), "dataset_count": len(rows),
                "state": ("unknown_no_observations" if not rows else "observed_audit_pass" if audit.get("scientific_accepted") else "observed_audit_nonpass_requires_failure_review"),
                "original_pooled_metrics": audit.get("pooled_metrics"),
                "criteria": protocol.get("criteria")}
        protocols.append(base)
        for line_number, row in enumerate(rows, 1):
            context = series_context(protocol, row)
            times = context.get("time_values") or []
            arrays = {"observed_mol_per_litre": row.get("positive_observed_mol_per_litre", []),
                      "predicted_mol_per_litre": row.get("positive_predicted_mol_per_litre", [])}
            reference = row.get("reference") or context.get("reference")
            # A screening duplicate key, not a claimed independently held-out study ID.
            series_key = fingerprint({"reference": reference, "dataset_name": row.get("dataset_name"),
                                      "times": times, "time_unit": context.get("time_unit"),
                                      "observed": arrays["observed_mol_per_litre"]})
            outcomes.append({"outcome_record_id": f"{specification['protocol_key']}:{line_number}",
                "protocol_key": specification["protocol_key"], "cohort": specification["cohort"],
                "exact_candidate_ids": matches, "query_ids": [], "independent_recommendation_judgment": None,
                "dataset_name": row.get("dataset_name"), "source_location": row.get("source_location"),
                "study_identity": reference_identity(reference), "context": context,
                "observed_window": {"start": min(times) if times else None, "end": max(times) if times else None,
                                    "time_unit": context.get("time_unit")},
                "metrics": ratio_metrics(row), "positive_pairs": arrays,
                "absolute_endpoints": absolute_endpoints(context, row), "curve_metric_verification": verify_curve_metrics(row),
                "observation_count": row.get("observation_count"), "positive_pair_count": row.get("positive_pair_count"),
                "excluded_nonpositive_observation_count": row.get("excluded_nonpositive_observation_count"),
                "excluded_nonpositive_prediction_count": row.get("excluded_nonpositive_prediction_count"),
                "dataset_guard_passed": row.get("dataset_guard_passed"),
                "endpoint_evaluable": row.get("endpoint_evaluable"), "original_endpoint_passed": row.get("endpoint_passed"),
                "source_metrics_path": specification["metrics_path"], "source_line": line_number,
                "observation_screening_fingerprint": series_key})
    caffeine_root = "experiments/caffeine_blueprint/datasets/caffeine_dataset_v3"
    caffeine_observations = inputs.read(f"{caffeine_root}/observations.jsonl", True)
    caffeine_metrics = inputs.read(f"{caffeine_root}/pk_metrics.jsonl", True)
    caffeine_manifest = inputs.read(f"{caffeine_root}/dataset_split_manifest.json")
    moxifloxacin = inputs.read(f"{BASE}/runs/model_execution_20260913/moxifloxacin_runtime_fixed/output_validation.json")
    named_queries = Counter(name for q in queries for name in q.get("active_moieties", []))
    exact_drugs = {name for m in models if m["model_version_id"] in exact_candidates for name in m.get("compounds", [])}
    unique_series = {row["observation_screening_fingerprint"]: row for row in outcomes}
    observation_candidates = defaultdict(set)
    families = {model["model_version_id"]: model["model_family"] for model in models}
    for row in outcomes:
        observation_candidates[row["observation_screening_fingerprint"]].update(row["exact_candidate_ids"])
    payload = {
        "schema_version": "pbpk-frontier-measured-outcomes-1.0.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing observed PK records normalized for reuse; no new simulation, scientific gold labels or recommender training.",
        "summary": {"queries": len(queries), "registered_candidates": len(models), "audited_protocols": len(protocols),
            "protocol_states": dict(Counter(p["state"] for p in protocols)), "dataset_records_including_repeated_audits": len(outcomes),
            "screening_unique_observed_series": len(unique_series),
            "positive_pairs_including_repeated_audits": sum(row.get("positive_pair_count") or 0 for row in outcomes),
            "screening_unique_positive_pairs": sum(row.get("positive_pair_count") or 0 for row in unique_series.values()),
            "endpoint_evaluable_records": sum(row["endpoint_evaluable"] is True for row in outcomes),
            "exact_artifact_matched_candidate_count": len(exact_candidates),
            "exact_artifact_matched_candidate_ids": sorted(exact_candidates),
            "exact_artifact_matched_candidates_with_observations": len({c for r in outcomes for c in r["exact_candidate_ids"]}),
            "exact_artifact_matched_outcome_records": sum(bool(row["exact_candidate_ids"]) for row in outcomes),
            "observed_window_start_after_zero_records": sum(row["observed_window"]["start"] > 0 for row in outcomes if row["observed_window"]["start"] is not None),
            "shared_observation_groups_with_multiple_registered_versions": sum(len(ids) > 1 for ids in observation_candidates.values()),
            "shared_observation_groups_with_multiple_registered_families": sum(len({families[c] for c in ids}) > 1 for ids in observation_candidates.values()),
            "same_compound_queries_for_exact_artifacts_not_exact_scenario_joins": sum(named_queries[name] for name in exact_drugs),
            "exact_query_candidate_outcome_joins": 0, "independently_verified_studies": None,
            "independent_same_query_candidate_comparisons": 0, "recommender_pk_regret_available": False},
        "metric_semantics": {
            "auc_observed_window_ratio": "Predicted/observed trapezoidal AUC from first to last observed time, using predictions interpolated onto the observed time grid. Legacy source field auc0_last_ratio does not imply a zero start.",
            "sampled_cmax_ratio": "Maximum prediction on the observed sampling grid / maximum observed concentration; not the maximum over the entire simulated continuous curve.",
            "log_rmse": "Root mean squared natural log(predicted/observed), using finite positive pairs retained by the historical auditor.",
            "unknown": "No observed datasets, missing paired predictions, blocked execution and missing independent provenance are not empirical failures.",
            "duplicate_handling": "Raw repeated audit outcomes retained with a screening fingerprint; duplicates must not be counted as independent studies.",
        },
        "current_benchmark_join": {"query_candidate_outcomes": [], "scientific_comparison_eligible": False,
            "missing_query_study_id_count": sum(not q.get("study_id") for q in queries),
            "reason": "The 772 existing requests contain compound/evidence/task metadata, not the observed study, dose, analyte and sampling scenario. Same-drug matches cannot establish exact measured recommendation outcomes. Candidate calibration/evidence source independence is also unverified.",
            "required_before_pk_recommendation_comparison": ["Exact held-out study and scenario per request", "Source identity and calibration/KB overlap audit", "Measured outcome of each eligible candidate on that same scenario", "At least two independent model families for within-drug preference/regret claims"]},
        "protocols": protocols, "outcomes": outcomes,
        "unpaired_caffeine_observations": {"state": "observations_present_no_registered_candidate_predictions_on_holdout",
            "study_split_manifest": caffeine_manifest, "observations": caffeine_observations, "pk_metrics": caffeine_metrics,
            "validation_result": readiness["caffeine_study_split"]["validation_result"],
            "selfbuilt_model_status": readiness["caffeine_study_split"]["scientific_status"],
            "registration_allowed": readiness["caffeine_study_split"]["registration_allowed"]},
        "moxifloxacin_engineering_only": {"state": "simulated_without_independent_observations", "validation": moxifloxacin,
                                           "measured_prediction_error": None},
        "source_files": sorted(inputs.sources.values(), key=lambda x: x["path"]),
    }
    destination = inputs.path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({"output": str(destination.relative_to(ROOT)), **payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    extract(parser.parse_args().output)
