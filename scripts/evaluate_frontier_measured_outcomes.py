"""Recompute measured PK errors and provenance without manufacturing ranking gold.

This is a model-execution assessment, not an evaluation of the 772 generic
recommendation requests. Existing audited concentration pairs are the numeric
inputs. Source role, study overlap, repeated audits and failed measurements remain
visible; a verification label alone never establishes independent validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import inspect
import json
import math
from pathlib import Path
import re
from statistics import mean, median

from extract_frontier_outcomes import (Inputs, absolute_endpoints, digest, finite_positive,
    fingerprint, ratio_metrics, reference_identity, series_context, verify_curve_metrics)
from pbpk_kb.model_assets.licensed_snapshot_reuse import _concentration_in_mol_per_litre, _time_in_hours

ROOT = Path(__file__).resolve().parents[1]
BASE = "experiments/model_recommendation/frontier_comparison_20260914"
MODELS = "experiments/model_recommendation/runs/expanded_catalog_20260913/benchmark/model_records.jsonl"
VERSION = "pbpk-measured-outcome-assessment-1.0.0"
RESULT_ROOT = "artifacts/pbpk-model-assets/windows_execution_results/run_20260822_v4"


def curve_metrics(observed, predicted):
    if not observed or len(observed) != len(predicted):
        raise ValueError("Concentration pairs must be nonempty and equally sized")
    if not all(finite_positive(v) for v in [*observed, *predicted]):
        raise ValueError("Log metrics require finite positive concentration pairs")
    logs = [math.log(p / o) for o, p in zip(observed, predicted, strict=True)]
    return {"gmfe": math.exp(mean(abs(v) for v in logs)),
            "geometric_bias": math.exp(mean(logs)),
            "log_rmse": math.sqrt(mean(v * v for v in logs)),
            "two_fold_fraction": mean(abs(v) <= math.log(2) for v in logs)}


def concentration_factor(unit, molecular_weight):
    return _concentration_in_mol_per_litre(1, unit, molecular_weight)


def time_hours(value, unit):
    # Snapshot labels use day(s), whereas the runtime parser uses day/days.
    return _time_in_hours(value, "days" if unit == "day(s)" else unit)


def recompute(row):
    """Verify alignment, recompute endpoint ratios from pairs, never from ratios."""
    context = row["context"]
    observed = row["positive_pairs"]["observed_mol_per_litre"]
    predicted = row["positive_pairs"]["predicted_mol_per_litre"]
    result = curve_metrics(observed, predicted)
    for key, value in result.items():
        if not math.isclose(value, row["metrics"][key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Audited curve metric mismatch: {row['outcome_record_id']} {key}")
    if len(observed) != row["positive_pair_count"]:
        raise ValueError("Positive-pair count mismatch")
    times = [time_hours(t, context.get("time_unit")) for t in context.get("time_values", [])]
    values = context.get("observed_values", [])
    complete = len(times) == len(values) == len(observed) == row["observation_count"]
    endpoints = {"status": "incomplete_positive_pair_alignment", "auc_ratio": None,
                 "sampled_cmax_ratio": None}
    if complete:
        if not all(math.isfinite(t) for t in times):
            raise ValueError("Unsupported or nonfinite time grid")
        factor = concentration_factor(context["observed_unit"], context.get("molecular_weight_g_per_mol"))
        if not all(math.isclose(o, v * factor, rel_tol=1e-10, abs_tol=1e-18)
                   for o, v in zip(observed, values, strict=True)):
            raise ValueError("Positive observed pairs do not match the full source time grid")
        ordered = sorted(zip(times, observed, predicted, strict=True))
        if any(a[0] == b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Duplicate observation times require an explicit aggregation policy")
        if len(ordered) < 2:
            endpoints["status"] = "single_sample_not_an_auc_or_peak_series"
        else:
            auc = lambda col: sum((b[0] - a[0]) * (b[col] + a[col]) / 2
                                  for a, b in zip(ordered, ordered[1:]))
            observed_auc, predicted_auc = auc(1), auc(2)
            observed_peak, predicted_peak = max(observed), max(predicted)
            endpoints = {"status": "recomputed_from_full_aligned_pairs", "start_h": ordered[0][0],
                         "end_h": ordered[-1][0], "auc_unit": "mol*h/l", "cmax_unit": "mol/l",
                         "observed_auc": observed_auc, "predicted_auc": predicted_auc,
                         "auc_ratio": predicted_auc / observed_auc,
                         "observed_sampled_cmax": observed_peak, "predicted_sampled_cmax": predicted_peak,
                         "sampled_cmax_ratio": predicted_peak / observed_peak}
            for new, old in [("auc_ratio", "auc_observed_window_ratio"),
                             ("sampled_cmax_ratio", "sampled_cmax_ratio")]:
                if not math.isclose(endpoints[new], row["metrics"][old], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Audited endpoint differs from aligned concentrations: {row['outcome_record_id']}")
    for key in ("auc_ratio", "sampled_cmax_ratio"):
        ratio = endpoints[key]
        endpoints[key.replace("_ratio", "_absolute_log_error")] = abs(math.log(ratio)) if ratio else None
        endpoints[key.replace("_ratio", "_absolute_relative_error")] = abs(ratio - 1) if ratio else None
    return {"curve_metrics": result, "endpoints": endpoints,
            "aligned_positive_pair_count": len(observed), "full_time_grid_alignment_verified": complete}


def study_tokens(properties, repository, reference=None):
    raw = str(reference or properties.get("Reference") or "").strip()
    tokens = set()
    for pmid in re.findall(r"(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/|PMID[:\s]+)(\d+)", raw, re.I):
        tokens.add(f"pmid:{pmid}")
    for doi in re.findall(r"10\.\d{4,9}/[^\s<>\"]+", raw, re.I):
        tokens.add(f"doi:{doi.rstrip('.,;)').lower()}")
    if raw:
        tokens.add("reference:" + fingerprint(raw.casefold()))
    declared = str(properties.get("Study Id") or "").strip()
    if declared and declared != ".":
        tokens.add("repository_declared_study:" + fingerprint([repository, declared.casefold()]))
    return sorted(tokens)


def build_role_index(protocol, snapshot):
    properties = {item["Name"]: {p["Name"]: p.get("Value") for p in item.get("ExtendedProperties", [])}
                  for item in snapshot.get("ObservedData", [])}
    roles = defaultdict(set)
    building_studies = set()
    repository = protocol.get("source", {}).get("repository_url")
    for simulation in protocol.get("simulations", []):
        for name in simulation.get("observed_dataset_names", []):
            roles[name].add(simulation.get("role", "unknown"))
            if simulation.get("role") == "model_building":
                building_studies.update(study_tokens(properties.get(name, {}), repository))
    return properties, roles, building_studies


def source_role(row, protocol, snapshot, snapshot_path):
    properties, roles, building_studies = build_role_index(protocol, snapshot)
    props = properties.get(row["dataset_name"], {})
    tokens = study_tokens(props, protocol.get("source", {}).get("repository_url"), row["context"].get("reference"))
    overlap = sorted(set(tokens) & building_studies)
    dataset_overlap = "model_building" in roles.get(row["dataset_name"], set())
    return {"source_snapshot_path": snapshot_path,
            "source_snapshot_sha256": protocol.get("source", {}).get("snapshot_sha256"),
            "source_location": row["source_location"], "declared_roles": sorted(roles.get(row["dataset_name"], set())),
            "reference": row["context"].get("reference") or props.get("Reference"),
            "declared_study_id": props.get("Study Id"), "identity_tokens": tokens,
            "same_dataset_in_model_building": dataset_overlap,
            "same_study_in_model_building": bool(overlap), "overlapping_identity_tokens": overlap,
            "independence_status": ("known_model_building_overlap" if dataset_overlap or overlap
                                    else "not_established"),
            "independence_verified": False,
            "scenario_metadata": {k: props.get(k) for k in ["Molecule", "Species", "Route", "Dose",
                                  "Times of Administration [h]", "Formulation", "Food state", "N", "Data type"]}}


def describe(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    def quantile(p):
        pos = (len(values) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (pos - lo) * (values[hi] - values[lo])
    return {"count": len(values), "mean": mean(values), "median": median(values),
            "p25": quantile(.25), "p75": quantile(.75)}


def summarize(rows, originals):
    by_id = {r["outcome_record_id"]: r for r in originals}
    observed, predicted = [], []
    for row in rows:
        original = by_id[row["representative_outcome_id"]]
        observed.extend(original["positive_pairs"]["observed_mol_per_litre"])
        predicted.extend(original["positive_pairs"]["predicted_mol_per_litre"])
    result = {"unique_prediction_series_count": len(rows), "positive_pair_count": len(observed),
              "pooled_point_weighted_curve": curve_metrics(observed, predicted) if observed else None,
              "series_macro_curve": {key: describe([r["curve_metrics"][key] for r in rows])
                                     for key in ["gmfe", "geometric_bias", "log_rmse", "two_fold_fraction"]}}
    result["endpoints"] = {}
    for key in ["auc", "sampled_cmax"]:
        ratios = [r["endpoints"][f"{key}_ratio"] for r in rows if r["endpoints"][f"{key}_ratio"] is not None]
        result["endpoints"][key] = {"ratio": describe(ratios),
            "absolute_log_error": describe([abs(math.log(v)) for v in ratios]),
            "absolute_relative_error": describe([abs(v - 1) for v in ratios]),
            "two_fold_fraction": mean(.5 <= v <= 2 for v in ratios) if ratios else None}
    result["known_building_overlap_series_count"] = sum(any(
        p["independence_status"] == "known_model_building_overlap" for p in r["source_role_audits"]) for r in rows)
    return result


def compare_snapshot_versions(assessed, originals, protocol_data):
    by_id = {r["outcome_record_id"]: r for r in originals}
    groups = defaultdict(list)
    for row in assessed:
        groups[row["observation_fingerprint"]].append(row)
    comparisons = []
    for rows in groups.values():
        for left, right in itertools.combinations(rows, 2):
            sources = []
            for row in (left, right):
                protocols = {by_id[oid]["protocol_key"] for oid in row["source_outcome_ids"]}
                identities = {(protocol_data[p][1]["source"]["repository_url"],
                               protocol_data[p][0]["source_snapshot_sha256"]) for p in protocols}
                sources.append(identities)
            # One source identity on each side; matching observed arrays alone
            # cannot identify different versions within an ambiguous execution.
            if any(len(side) != 1 for side in sources):
                continue
            source_left, source_right = next(iter(sources[0])), next(iter(sources[1]))
            if source_left[0] != source_right[0] or source_left[1] == source_right[1]:
                continue
            if source_right[1] < source_left[1]:
                left, right = right, left
                source_left, source_right = source_right, source_left
            errors = {}
            for metric, container in [("gmfe", "curve_metrics"), ("log_rmse", "curve_metrics"),
                                      ("auc_absolute_log_error", "endpoints"),
                                      ("sampled_cmax_absolute_log_error", "endpoints")]:
                a, b = left[container].get(metric), right[container].get(metric)
                errors[metric] = {"left": a, "right": b, "left_minus_right": a - b if a is not None and b is not None else None,
                    "lower_error_side": (None if a is None or b is None else "tie" if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                         else "left" if a < b else "right")}
            comparisons.append({"observation_fingerprint": left["observation_fingerprint"],
                "dataset_name": left["dataset_name"], "repository_url": source_left[0],
                "left_snapshot_sha256": source_left[1], "right_snapshot_sha256": source_right[1],
                "left_measurement_id": left["measurement_id"], "right_measurement_id": right["measurement_id"],
                "left_registered_candidate_ids": left["exact_candidate_ids"],
                "right_registered_candidate_ids": right["exact_candidate_ids"], "errors": errors,
                "independence_verified": False, "current_recommender_evaluation_eligible": False})
    return {"scope": "Same repository, distinct pinned snapshots evaluated on identical observation series; retrospective, not independent-family preference.",
            "pair_count": len(comparisons), "pairs": comparisons,
            "lower_error_counts": {metric: dict(Counter(r["errors"][metric]["lower_error_side"] for r in comparisons
                                                        if r["errors"][metric]["lower_error_side"] is not None))
                for metric in ["gmfe", "log_rmse", "auc_absolute_log_error", "sampled_cmax_absolute_log_error"]},
            "statistical_inference": "No p-values or independent-sample claims: many series belong to shared studies and calibration independence is unverified.",
            "ranking_labels_created": False}


def _reaudit_one(arguments):
    """CPU-only immutable derived audit; old audit and execution files stay intact."""
    from pbpk_kb.model_assets.licensed_snapshot_reuse import (audit_snapshot_reuse_execution,
        LicensedSnapshotReuseProtocol, SnapshotWindowsManifest, SnapshotReuseError, _dataset_metrics)
    protocol_path, results_path, output_path, identity = arguments
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Never overwrite a previous derived execution audit")
    report = audit_snapshot_reuse_execution(protocol_path=Path(protocol_path),
        windows_result_root=Path(results_path), output_directory=output)
    # A previous cleanup removed PKML/XML exports. Their absence must remain an
    # engineering failure, but hash-locked, repeated CSV predictions can still
    # support a retrospective concentration-error calculation. Never accept a
    # present-but-corrupt artifact or a failing model/roundtrip check here.
    root = Path(results_path)
    manifest = SnapshotWindowsManifest.model_validate_json((root / "windows_run_manifest.json").read_text(encoding="utf-8-sig"))
    protocol = LicensedSnapshotReuseProtocol.model_validate_json(Path(protocol_path).read_text())
    absent, unexpected = [], []
    for run in manifest.runs:
        for simulation in run.simulations:
            for artifact in simulation.artifacts:
                path = (root / artifact.relative_path.replace("\\", "/")).resolve()
                path.relative_to(root.resolve())
                if not path.is_file():
                    item = {"relative_path": artifact.relative_path.replace("\\", "/"),
                            "expected_sha256": artifact.sha256, "byte_size": artifact.byte_size}
                    absent.append(item)
                    if artifact.extension.casefold() not in {".pkml", ".xml"}:
                        unexpected.append(item)
                elif path.stat().st_size != artifact.byte_size or digest(path) != artifact.sha256:
                    unexpected.append({"relative_path": artifact.relative_path, "reason": "size_or_sha256_mismatch"})
    failed_checks = [k for k, v in report["engineering_checks"].items() if not v]
    eligible = bool(absent) and not unexpected and failed_checks == ["all_artifacts_verified"]
    rows, errors = [], []
    if eligible:
        run1 = next(run for run in manifest.runs if run.run_id == "run_1")
        executions = {s.sequence: s for s in run1.simulations}
        for simulation in protocol.simulations:
            if simulation.role != "model_verification":
                continue
            for observed in simulation.all_evaluation_series():
                try:
                    rows.append(_dataset_metrics(result_root=root, artifacts=executions[simulation.sequence].artifacts,
                        simulation=simulation, observed=observed, criteria=protocol.criteria))
                except SnapshotReuseError as exc:
                    errors.append({"simulation_sequence": simulation.sequence, "simulation_name": simulation.name,
                        "dataset_name": observed.dataset_name, "source_location": observed.source_location,
                        "reason_code": "RETAINED_OUTPUT_DATASET_EVALUATION_FAILED", "detail": str(exc)})
    metric_path = output / "retained_output_metrics.jsonl"
    metric_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
    retained = {"state": "retained_output_only" if eligible else "not_eligible", "eligible": eligible,
                "engineering_accepted": False, "independent_validation": False,
                "conditions": "All checks except missing non-CSV exports must pass; every present artifact SHA256 rechecked; no corrupted files accepted.",
                "missing_exports": absent, "unexpected_missing_or_corrupt_artifacts": unexpected,
                "failed_engineering_checks": failed_checks, "measured_series": len(rows), "dataset_errors": errors,
                "metrics_sha256": digest(metric_path)}
    retained_path = output / "retained_output_assessment.json"
    retained_path.write_text(json.dumps(retained, ensure_ascii=False, indent=2) + "\n")
    identity["new_audit_sha256"] = digest(output / "snapshot_reuse_audit.json")
    identity["retained_assessment_sha256"] = digest(retained_path)
    (output / "repair_context.json").write_text(json.dumps(identity, indent=2) + "\n")
    return {"protocol_key": identity["protocol_key"], "engineering_accepted": report["engineering_accepted"],
            "evaluated_datasets": report["counts"]["evaluated_datasets"],
            "dataset_errors": report["counts"]["dataset_errors"],
            "retained_output_measured_series": len(rows), "retained_output_errors": len(errors)}


def apply_reaudits(payload, inputs, execute=False, workers=2):
    code = {name: digest(inputs.path(name)) for name in [
        "src/pbpk_kb/model_assets/licensed_snapshot_reuse.py", "src/pbpk_kb/model_runtime/result_parser.py"]}
    code["retained_output_implementation"] = fingerprint(inspect.getsource(_reaudit_one))
    repair_root = f"{BASE}/assessment/measured_reaudits/{fingerprint(code)[:12]}"
    specifications, pending = [], []
    for protocol in payload["protocols"]:
        old_audit = inputs.read(protocol["audit_path"])
        if protocol["dataset_count"] or old_audit["engineering_accepted"]:
            continue
        key = protocol["protocol_key"]
        root = f"{RESULT_ROOT}/{key}"
        identity = {"protocol_key": key, "protocol_sha256": protocol["protocol_sha256"],
                    "old_audit_path": protocol["audit_path"],
                    "old_audit_sha256": digest(inputs.path(protocol["audit_path"])),
                    "windows_manifest_sha256": digest(inputs.path(f"{root}/windows_run_manifest.json")),
                    "auditor_code_sha256": code,
                    "reason": "Repair explicitly supported CSV metadata and equivalent roundtrip representation; unchanged scientific thresholds and source execution."}
        destination = f"{repair_root}/{key}"
        specifications.append((protocol, destination, identity))
        if not inputs.path(f"{destination}/repair_context.json").is_file():
            if execute:
                pending.append((str(inputs.path(protocol["protocol_path"])), str(inputs.path(root)),
                                str(inputs.path(destination)), identity))
            continue
        previous = inputs.read(f"{destination}/repair_context.json")
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Derived audit source or repair identity changed")
        if digest(inputs.path(f"{destination}/snapshot_reuse_audit.json")) != previous["new_audit_sha256"]:
            raise ValueError("Derived audit content changed")
        if digest(inputs.path(f"{destination}/retained_output_assessment.json")) != previous["retained_assessment_sha256"]:
            raise ValueError("Retained-output assessment changed")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_reaudit_one, arguments) for arguments in pending]
            for future in as_completed(futures):
                print(json.dumps({"reaudit_completed": future.result()}), flush=True)
    changes = []
    for protocol, destination, identity in specifications:
        if not inputs.path(f"{destination}/repair_context.json").is_file():
            changes.append({"protocol_key": protocol["protocol_key"], "state": "repair_audit_pending"})
            continue
        inputs.read(f"{destination}/repair_context.json")
        audit = inputs.read(f"{destination}/snapshot_reuse_audit.json")
        metrics_path = f"{destination}/verification_dataset_metrics.jsonl"
        if digest(inputs.path(metrics_path)) != audit["verification_dataset_metrics_sha256"]:
            raise ValueError("Derived metrics differ from the new audit hash")
        rows = inputs.read(metrics_path, True)
        retained = inputs.read(f"{destination}/retained_output_assessment.json")
        if not rows and retained["eligible"]:
            metrics_path = f"{destination}/retained_output_metrics.jsonl"
            if digest(inputs.path(metrics_path)) != retained["metrics_sha256"]:
                raise ValueError("Retained-output metric checksum mismatch")
            rows = inputs.read(metrics_path, True)
        raw = inputs.read(protocol["protocol_path"])
        manifest = inputs.read(f"{RESULT_ROOT}/{protocol['protocol_key']}/windows_run_manifest.json")
        snapshot_path = f"{RESULT_ROOT}/{protocol['protocol_key']}/" + manifest["input_snapshot_relative_path"].replace("\\", "/")
        snapshot = inputs.read(snapshot_path)
        if digest(inputs.path(snapshot_path)) != protocol["source_snapshot_sha256"]:
            raise ValueError("Reaudited snapshot mismatch")
        payload["source_files"].append({"path": snapshot_path, "sha256": protocol["source_snapshot_sha256"]})
        old = {k: protocol[k] for k in ["audit_path", "metrics_path", "dataset_count", "engineering_accepted",
                                       "original_scientific_accepted", "dataset_errors", "state"]}
        changes.append({"protocol_key": protocol["protocol_key"], "state": "reaudited",
            "previous": old, "audit_path": f"{destination}/snapshot_reuse_audit.json",
            "new_engineering_accepted": audit["engineering_accepted"], "new_evaluated_datasets": audit["counts"]["evaluated_datasets"],
            "new_dataset_errors": len(audit["dataset_errors"]),
            "remaining_engineering_failures": [k for k, v in audit["engineering_checks"].items() if v is False],
            "remaining_roundtrip_differences": audit["roundtrip"]["difference_examples"],
            "retained_output_assessment_path": f"{destination}/retained_output_assessment.json",
            "retained_output_eligible": retained["eligible"], "retained_output_evaluated_datasets": retained["measured_series"],
            "retained_output_error_count": len(retained["dataset_errors"]),
            "missing_export_count": len(retained["missing_exports"]),
            "auditor_code_sha256": identity["auditor_code_sha256"]})
        protocol.update(audit_path=f"{destination}/snapshot_reuse_audit.json", metrics_path=metrics_path,
            dataset_count=len(rows), engineering_accepted=audit["engineering_accepted"],
            original_scientific_accepted=audit["scientific_accepted"],
            dataset_errors=retained["dataset_errors"] if retained["eligible"] else audit["dataset_errors"],
            retained_output_only=retained["eligible"],
            state="retained_output_only" if retained["eligible"] else "observed_reaudit" if rows else "engineering_blocked_after_reaudit")
        for line, row in enumerate(rows, 1):
            context = series_context(raw, row)
            reference = row.get("reference") or context.get("reference")
            arrays = {"observed_mol_per_litre": row["positive_observed_mol_per_litre"],
                      "predicted_mol_per_litre": row["positive_predicted_mol_per_litre"]}
            observation_key = fingerprint({"reference": reference, "dataset_name": row.get("dataset_name"),
                "times": context.get("time_values", []), "time_unit": context.get("time_unit"),
                "observed": arrays["observed_mol_per_litre"]})
            payload["outcomes"].append({"outcome_record_id": f"{protocol['protocol_key']}:reaudit:{line}",
                "protocol_key": protocol["protocol_key"], "cohort": protocol["cohort"],
                "evaluation_basis": "retained_checksum_locked_csv" if retained["eligible"] else "complete_execution_audit",
                "exact_candidate_ids": protocol["exact_candidate_ids"], "query_ids": [],
                "independent_recommendation_judgment": None, "dataset_name": row["dataset_name"],
                "source_location": row["source_location"], "study_identity": reference_identity(reference),
                "context": context, "metrics": ratio_metrics(row), "positive_pairs": arrays,
                "absolute_endpoints": absolute_endpoints(context, row),
                "curve_metric_verification": verify_curve_metrics(row),
                "observation_count": row["observation_count"], "positive_pair_count": row["positive_pair_count"],
                "dataset_guard_passed": row["dataset_guard_passed"], "endpoint_evaluable": row["endpoint_evaluable"],
                "source_metrics_path": metrics_path, "source_line": line,
                "observation_screening_fingerprint": observation_key})
    return changes


def evaluate(input_file=BASE + "/observed_outcomes.json", output_file=BASE + "/assessment/measured_outcomes.json",
             reaudit_blocked=False, workers=2):
    inputs = Inputs()
    payload = inputs.read(input_file)
    for source in payload["source_files"]:
        if digest(inputs.path(source["path"])) != source["sha256"]:
            raise ValueError(f"Pinned source changed: {source['path']}")
    original_record_count = len(payload["outcomes"])
    repairs = apply_reaudits(payload, inputs, reaudit_blocked, workers)
    models = inputs.read(MODELS, True)
    by_id = {r["model_version_id"]: r for r in models}
    sources_by_hash = {r["sha256"]: r["path"] for r in payload["source_files"] if r["path"].endswith(".json")}
    protocol_data, summaries = {}, []
    planned_keys, error_keys, blocked_keys = set(), set(), set()
    for protocol in payload["protocols"]:
        key = protocol["protocol_key"]
        raw = inputs.read(protocol["protocol_path"])
        audit = inputs.read(protocol["audit_path"])
        snapshot_path = sources_by_hash.get(protocol["source_snapshot_sha256"])
        if not snapshot_path and protocol["cohort"] == "controlled_adaptations":
            possible = Path(protocol["protocol_path"]).parents[2] / "snapshots" / key / "adapted_snapshot.json"
            if inputs.path(possible).is_file():
                snapshot_path = str(possible)
        snapshot = inputs.read(snapshot_path) if snapshot_path else {}
        if snapshot_path and digest(inputs.path(snapshot_path)) != protocol["source_snapshot_sha256"]:
            raise ValueError("Source snapshot hash mismatch")
        protocol_data[key] = (protocol, raw, snapshot, snapshot_path)
        planned = []
        for simulation in raw.get("simulations", []):
            if simulation.get("role") != "model_verification":
                continue
            for series in [simulation.get("evaluation_series"), *simulation.get("additional_evaluation_series", [])]:
                if not isinstance(series, dict):
                    continue
                entry = [protocol["source_snapshot_sha256"], simulation["name"], series["dataset_name"], series.get("source_location")]
                planned.append(fingerprint(entry))
        planned_keys.update(planned)
        failures = [name for name, passed in audit.get("engineering_checks", {}).items() if passed is False]
        errors = protocol["dataset_errors"]
        for error in errors:
            error_keys.add(fingerprint([protocol["source_snapshot_sha256"], error.get("simulation_name"),
                                        error.get("dataset_name"), error.get("source_location")]))
        unevaluated = len(planned) - protocol["dataset_count"] - len(errors)
        if unevaluated < 0:
            raise ValueError("More measured/error outcomes than planned series")
        if failures and protocol["dataset_count"] == 0 and not errors:
            blocked_keys.update(planned)
        summaries.append({"protocol_key": key, "repository_or_adaptation": protocol["repository_or_adaptation"],
            "exact_candidate_ids": protocol["exact_candidate_ids"], "planned_quantitative_series": len(planned),
            "measured_series": protocol["dataset_count"], "dataset_evaluation_errors": len(errors),
            "unevaluated_series": unevaluated, "engineering_failures": failures,
            "state": ("retained_output_only" if protocol.get("retained_output_only") else "engineering_blocked" if failures and not protocol["dataset_count"]
                      else "measured_with_dataset_errors" if errors else "measured" if protocol["dataset_count"] else "no_planned_measurement"),
            "engineering_accepted": protocol["engineering_accepted"], "audit_path": protocol["audit_path"],
            "dataset_errors": errors})
    groups = defaultdict(list)
    for row in payload["outcomes"]:
        # Identical observations can have distinct predictions; keep each distinct
        # execution curve instead of choosing whichever error looks best.
        groups[(row["observation_screening_fingerprint"], fingerprint(row["positive_pairs"]["predicted_mol_per_litre"]))].append(row)
    assessed = []
    for group, originals in sorted(groups.items()):
        original = originals[0]
        computed = recompute(original)
        provenance = []
        for repeated in originals:
            if recompute(repeated) != computed:
                raise ValueError("Repeated predictions have conflicting endpoint semantics")
            _, raw, snapshot, snapshot_path = protocol_data[repeated["protocol_key"]]
            provenance.append(source_role(repeated, raw, snapshot, snapshot_path))
        scenarios = [{k: r["context"].get(k) for k in ["simulation_name", "individual", "output_path"]}
                     for r in originals]
        candidate_ids = sorted({c for r in originals for c in r["exact_candidate_ids"]})
        assessed.append({"measurement_id": fingerprint(group), "observation_fingerprint": group[0],
            "representative_outcome_id": original["outcome_record_id"],
            "source_outcome_ids": [r["outcome_record_id"] for r in originals],
            "evaluation_bases": sorted({r.get("evaluation_basis", "historical_complete_execution_audit") for r in originals}),
            "dataset_name": original["dataset_name"], "exact_candidate_ids": candidate_ids,
            "model_families": sorted({by_id[c]["model_family"] for c in candidate_ids}),
            "source_role_audits": provenance, "execution_scenarios": scenarios,
            "strict_benchmark_query_ids": [], "independent_recommendation_judgment": None, **computed})
    candidate_reports = []
    for candidate_id in sorted({c for r in assessed for c in r["exact_candidate_ids"]}):
        rows = [r for r in assessed if candidate_id in r["exact_candidate_ids"]]
        model = by_id[candidate_id]
        candidate_reports.append({"candidate_id": candidate_id, "family_id": model["model_family"],
            "family_name": model["model_family_name"], "compounds": model["compounds"],
            "summary": summarize(rows, payload["outcomes"]),
            "measurement_ids": [r["measurement_id"] for r in rows]})
    comparisons = []
    for row in assessed:
        for left, right in itertools.combinations(row["exact_candidate_ids"], 2):
            comparisons.append({"measurement_id": row["measurement_id"], "candidate_ids": [left, right],
                "same_family": by_id[left]["model_family"] == by_id[right]["model_family"],
                "identical_prediction_curve": True,
                "shared_executed_outcome_record": any({left, right} <= set(r["exact_candidate_ids"])
                    for r in payload["outcomes"] if r["outcome_record_id"] in row["source_outcome_ids"]),
                "preference_label": None, "recommender_regret_eligible": False,
                "reason": "Exact artifact aliases share an executed curve; this supplies no independently executed candidate preference."})
    report = {"schema_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "complete_for_available_pinned_measurements", "input_file": input_file,
        "input_sha256": digest(inputs.path(input_file)), "scope": "Retrospective model execution errors; not recommendation-method accuracy",
        "summary": {"registered_candidates": len(models), "raw_measured_records": len(payload["outcomes"]),
            "original_raw_measured_records": original_record_count,
            "additional_measured_records_after_audit_repairs": len(payload["outcomes"]) - original_record_count,
            "unique_observation_series": len({r["observation_fingerprint"] for r in assessed}),
            "unique_prediction_series": len(assessed), "repeated_measured_records_removed": len(payload["outcomes"]) - len(assessed),
            "candidates_with_measured_outcomes": len(candidate_reports), "families_with_measured_outcomes": len({r['family_id'] for r in candidate_reports}),
            "raw_planned_quantitative_series": sum(p["planned_quantitative_series"] for p in summaries),
            "unique_planned_scenario_series": len(planned_keys), "unique_dataset_evaluation_errors": len(error_keys),
            "unique_engineering_blocked_series": len(blocked_keys),
            "protocol_states": dict(Counter(p["state"] for p in summaries)),
            "known_building_dataset_overlap_series": sum(any(p["same_dataset_in_model_building"] for p in r["source_role_audits"]) for r in assessed),
            "known_building_study_overlap_series": sum(any(p["same_study_in_model_building"] for p in r["source_role_audits"]) for r in assessed),
            "series_with_resolved_reference_or_declared_study": sum(any(p["identity_tokens"] for p in r["source_role_audits"]) for r in assessed),
            "independence_verified_series": 0, "strict_current_query_candidate_joins": 0,
            "shared_curve_same_family_version_comparisons": len(comparisons),
            "independently_executed_family_preference_pairs": 0},
        "metric_semantics": {"auc": "Trapezoidal AUC from first to last observed time on the aligned observed grid; never AUC0 or AUC infinity.",
            "sampled_cmax": "Maximum of aligned observed/predicted samples, not a continuous or densely simulated peak.",
            "curve": "GMFE, geometric bias, natural-log RMSE and 2-fold coverage recomputed from strictly positive paired concentrations.",
            "aggregation": "Report pooled point-weighted and series-macro statistics separately; no overall score or cross-drug best-model claim.",
            "single_point": "Single-sample records contribute pointwise curve error only; AUC and peak-series metrics remain unavailable.",
            "deduplication": "Identical observation fingerprint plus identical predicted concentration curve counted once; distinct predictions retained.",
            "independence": "Source metadata and protocol roles audited; absence of detected building overlap does not prove independent validation."},
        "pooled_available_measurements": summarize(assessed, payload["outcomes"]),
        "candidate_measurements": candidate_reports, "protocol_coverage": summaries,
        "audit_repairs": repairs,
        "same_family_version_comparisons": comparisons, "measurements": assessed,
        "within_repository_snapshot_comparison": compare_snapshot_versions(assessed, payload["outcomes"], protocol_data),
        "recommendation_method_pk_evaluation": {"state": "not_identifiable_for_current_requests", "method_results": [],
            "reason": "Current requests have no exact study/dose/analyte/sampling identity; no measured same-scenario independent-family choice exists.",
            "regret": None, "top1_pk_error": None, "independent_external_validation": None},
        "provenance": {"pinned_source_files_verified": len(payload["source_files"]),
                       "additional_sources": sorted(inputs.sources.values(), key=lambda r: r["path"])}}
    output = inputs.path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=BASE + "/observed_outcomes.json")
    parser.add_argument("--output", default=BASE + "/assessment/measured_outcomes.json")
    parser.add_argument("--reaudit-blocked", action="store_true", help="Reaudit original 12 blocked protocols on CPU")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.output, args.reaudit_blocked, args.workers)["summary"], ensure_ascii=False, indent=2))
