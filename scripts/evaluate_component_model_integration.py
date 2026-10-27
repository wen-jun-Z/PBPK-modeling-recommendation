#!/usr/bin/env python3
"""Paired evaluation of frozen PBPK-Match before/after soft component input.

No fitting, calibration, inference, PBPK execution, or modification of the paper.
Clinical outcomes enter only this evaluator, after the input score file is frozen.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from pbpk_kb.recommendation.pbpk_match.evaluation import evaluate_ranking
from pbpk_kb.recommendation.pbpk_match_v2 import evaluation as ev
from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset
import summarize_pbpk_match_light as light
from pbpk_match_light_paper_metrics import binary_applicability, known_topk_coverage
import pbpk_match_light_curves as curves
import recommendation_endpoint_extension_metrics as extended
from recommendation_execution_statistics import numerical_miss_audit

SCHEMA = "pbpk-component-model-integration-evaluation-v1"
BEFORE = "pbpk_match_light"
AFTER = "pbpk_match_components"
METHODS = [
    {"method_id": BEFORE + "_document", "method": "PBPK-Match document: original request"},
    {"method_id": BEFORE, "method": "PBPK-Match: original request"},
    {"method_id": AFTER + "_document", "method": "PBPK-Match document: soft components"},
    {"method_id": AFTER, "method": "PBPK-Match: soft components"},
]
EXTENDED = ROOT / "results/data/model_recommendation_extended_metrics.json"
REQUIRED_FLAGS = {
    "original_request_semantics_preserved": True,
    "components_are_soft_requirements": True,
    "labels_used_as_features": False,
    "observations_used_as_features": False,
    "weights_retrained": False,
    "alpha_retuned": False,
    "temperature_refitted": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scope(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT), "Evaluation paths must remain in this project")
    return path


def sha(path):
    h = hashlib.sha256()
    with scope(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference(path):
    path = scope(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def read(path):
    return json.loads(scope(path).read_text())


def verify_ref(value, sources):
    require(isinstance(value, dict) and {"path", "sha256"} <= value.keys(), "Missing source binding")
    path = scope(value["path"])
    require(path.is_file() and sha(path) == value["sha256"], "Changed source: " + str(path))
    sources.append(reference(path))
    return path


def validate_arrays(arrays, manifest, data, *, expected_queries=186, expected_candidates=280):
    """IDs may be permuted together; missing/duplicate IDs never get imputed."""
    ids = arrays.get("indices")
    require(isinstance(ids, np.ndarray) and ids.ndim == 1 and ids.dtype.kind in "iu", "indices must be integer global indices")
    wanted = set(data["core_indices"]) | set(data["execution_indices"])
    require(len(ids) == expected_queries and len(set(ids.tolist())) == len(ids) and set(ids.tolist()) == wanted,
            "Prediction indices must cover every frozen core and execution request exactly once")
    require(all(0 <= int(i) < len(data["query_records"]) for i in ids), "Prediction index out of range")
    query_ids = [data["query_records"][int(i)]["query_id"] for i in ids]
    candidate_ids = [c["candidate_id"] for c in data["candidate_records"]]
    require(len(candidate_ids) == expected_candidates and len(set(candidate_ids)) == expected_candidates, "Candidate catalogue differs")
    require(manifest.get("query_ids") == query_ids, "Manifest query IDs and global indices differ")
    require(manifest.get("candidate_ids") == candidate_ids, "Candidate ID order differs from the frozen catalogue")
    shape = (len(ids), len(candidate_ids))
    for name in ("scores", "base_scores"):
        value = np.asarray(arrays.get(name))
        require(value.shape == shape and np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(),
                "Incomplete or nonfinite full-catalog " + name)
    p = np.asarray(arrays.get("probabilities"))
    require(p.shape == (*shape, 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            and np.allclose(p.sum(-1), 1., atol=1e-7, rtol=0), "Probabilities must be normalized Y/N/U for every pair")
    return {int(index): row for row, index in enumerate(ids)}


def validate_semantics(manifest):
    require(manifest.get("state") == "complete", "Input inference is not complete")
    for key, value in REQUIRED_FLAGS.items():
        require(manifest.get(key) is value, "Absent or incompatible input policy: " + key)


def load_baseline(data, indices, sources):
    """Verify only our existing predictions/outer fits; never load other rankers."""
    protocol_path = light.OUT / "protocol.json"
    protocol = read(protocol_path)
    require(protocol["query_ids"] == [q["query_id"] for q in data["query_records"]]
            and protocol["candidate_ids"] == [c["candidate_id"] for c in data["candidate_records"]]
            and protocol["folds"] == data["folds"], "Baseline axes or folds changed")
    protocol_sha = sha(protocol_path)
    final_path = light.OUT / "predictions/seed17.npz"
    receipt_path = final_path.with_suffix(".json")
    receipt = read(receipt_path)
    require(receipt.get("state") == "complete" and receipt.get("protocol_sha256") == protocol_sha
            and receipt.get("score_sha256") == sha(final_path), "Baseline receipt does not bind its predictions")
    sources += [reference(protocol_path), reference(final_path), reference(receipt_path)]
    dataset_path = verify_ref(protocol["dataset_manifest"], sources)
    cache_path = verify_ref(protocol["cache_manifest"], sources)
    with np.load(final_path, allow_pickle=False) as z:
        require(np.array_equal(z["indices"], np.arange(len(data["query_records"]))), "Baseline global indices changed")
        original = {name: z[name][0].copy() for name in ("scores", "base_scores", "probabilities")}
    bindings = {}
    for fold_id, fold in enumerate(data["folds"]):
        path = verify_ref(receipt["outer_receipts"][str(fold_id)], sources)
        wanted = light.OUT / "runs/seed17" / f"fold{fold_id}" / "outer/complete.json"
        require(path == wanted, "Outer completion path changed")
        complete, binding = light._verify_job(data, path, protocol_sha, sha(dataset_path), sha(cache_path), sources)
        pred = light.prediction_arrays(path.parent / "predictions.npz", len(fold["outer_test"]),
                                       len(data["candidate_records"]), fold["outer_test"])
        for name in original:
            require(np.array_equal(original[name][fold["outer_test"]], pred[name][0]), "Baseline differs from outer holdout")
        bindings[fold_id] = {"checkpoint": reference(path.parent / "heads.pt"), "selection": binding["selection"],
                            "indices": sorted(set(indices.tolist()) & set(fold["outer_test"]))}
    return {name: value[indices].copy() for name, value in original.items()}, protocol_sha, bindings


def validate_integration_manifest(manifest, predictions_path, baseline_sha, bindings, sources):
    validate_semantics(manifest)
    require(manifest.get("predictions_sha256") == sha(predictions_path), "New score file not bound to manifest")
    require(manifest.get("baseline_protocol_sha256") == baseline_sha, "Baseline training protocol changed")
    protocol_path = manifest.get("protocol_path")
    require(protocol_path is not None and manifest.get("protocol_sha256") == sha(protocol_path), "Integration protocol not bound")
    sources.append(reference(protocol_path))
    for source in manifest.get("source_bindings", []):
        verify_ref(source, sources)
    folds = manifest.get("outer_folds", [])
    require(len(folds) == len(bindings) and len({f.get("fold") for f in folds}) == len(bindings), "Missing/duplicate outer-fold bindings")
    for item in folds:
        fid = item.get("fold")
        require(fid in bindings, "Unknown outer fold")
        expected = bindings[fid]
        require(sorted(item.get("indices", [])) == expected["indices"], "Input request assigned to incorrect outer fold")
        for key in ("checkpoint", "selection"):
            path = verify_ref(item.get(key), sources)
            require(path == scope(expected[key]["path"]) and item[key]["sha256"] == expected[key]["sha256"],
                    "Weights, alpha or temperature selection changed")


def evaluate_quality(scores, probabilities, data, core_local, binary_local):
    core = data["core_indices"]
    quality = evaluate_ranking(scores[core_local], data["grades"][core], data["applicability"][core],
                               [data["query_records"][i] for i in core], data["candidate_records"], probabilities[core_local])
    quality["known_top10_coverage"] = known_topk_coverage(scores[core_local], data["grades"][core])
    binary_global = [data["core_indices"][i] for i in binary_local]
    quality["paper_binary_102"] = binary_applicability(probabilities[np.asarray(core_local)[binary_local], :, 0],
                                                      data["applicability"][binary_global])
    return quality


def evaluate_extended(scores_by_method, query_ids, candidate_ids, frozen):
    """Select from saved candidate errors; never re-extract or reinterpret NCA."""
    by_query = {qid: i for i, qid in enumerate(query_ids)}
    require(set(frozen["query_ids"]) == set(query_ids) and frozen["candidate_ids"] == candidate_ids,
            "Extended endpoint axes differ")
    definitions = {d["endpoint_key"]: d for d in frozen["endpoint_definitions"]}
    fields = ("scenario_id", "endpoint_key", "analysis_stratum", "prediction_kind")
    metadata = ("compound", "publication_cluster_id", "family", "reference_eligible")
    skeleton = {}
    for row in frozen["position_rows"]:
        key = tuple(row[k] for k in fields)
        value = {k: row[k] for k in (*fields, *metadata)}
        require(key not in skeleton or skeleton[key] == value, "Method-dependent endpoint qualification")
        skeleton[key] = value
    errors = defaultdict(dict)
    for row in frozen["candidate_errors"]:
        key = tuple(row[k] for k in fields)
        require(row["candidate_id"] not in errors[key], "Duplicate locked candidate error")
        errors[key][row["candidate_id"]] = row
    positions = []
    for key, meta in sorted(skeleton.items()):
        sid = meta["scenario_id"]
        require(sid in by_query, "Endpoint scenario missing from new rankings")
        for method, scores in scores_by_method.items():
            for rank in (1, 2, 3):
                value = extended.exact_position(scores[by_query[sid]], candidate_ids, errors[key], rank)
                require(meta["reference_eligible"] or value["coverage"] == 0, "Ineligible reference scored")
                positions.append({"method_id": method, **meta, **value})
    methods = [m for m in METHODS if m["method_id"] in scores_by_method]
    summary = extended.summarize_positions(positions, methods, definitions, len(query_ids))
    panels = {(r["endpoint_key"], r["source_stratum"]) for r in frozen["endpoint_panels"]}
    return {"position_rows": positions, "position_summary": summary,
            "publication_rows": [r for r in summary if (r["endpoint_key"], r["source_stratum"]) in panels],
            "endpoint_definitions": list(definitions.values()), "family_inventory": frozen["family_inventory"],
            "qualification": "Unchanged locked candidate errors and method-independent reference qualification; no new predictions."}


def extended_omissions(scores_by_method, query_ids, candidate_ids, frozen, contract):
    """Measured-best omission; Tmax uses hours, never fold-error thresholds."""
    index = {qid: i for i, qid in enumerate(query_ids)}
    pools = defaultdict(dict)
    identities = {}
    for row in frozen["candidate_errors"]:
        if not row["scorable"]:
            continue
        key = tuple(row[k] for k in ("scenario_id", "endpoint_key", "analysis_stratum", "prediction_kind"))
        field = "absolute_error_h" if row["endpoint"] == "Tmax" else "absolute_log_error"
        if row.get(field) is not None:
            pools[key][row["candidate_id"]] = row[field]
            identities[key] = {k: row[k] for k in ("compound", "publication_cluster_id", "endpoint")}
    rows = []
    for key, errors in sorted(pools.items()):
        sid, endpoint_key, stratum, kind = key
        identity = identities[key]
        for method, scores in scores_by_method.items():
            result = numerical_miss_audit(candidate_ids, contract[sid]["same_parent_ids"],
                                          dict(zip(candidate_ids, scores[index[sid]].tolist())), errors, ks=(3, 5))
            if identity["endpoint"] == "Tmax":
                # The tie/hit calculation is unit-agnostic. Ratio-threshold
                # diagnostics and misleading log-error names must not escape.
                for entry in result["by_k"].values():
                    entry.pop("thresholds")
                result["best_measured_absolute_error_h"] = result.pop("best_measured_absolute_log_error")
            rows.append({"method_id": method, "scenario_id": sid, "endpoint_key": endpoint_key,
                         "analysis_stratum": stratum, "prediction_kind": kind, **identity, **result})
    return rows


def paired_extended(position_rows, definitions):
    """Reuse the same clustered estimator; explicitly type the error units.

    The frozen bootstrap consumes nonnegative errors and paired differences.
    Its legacy internal ``top1_absolute_log_error`` field is only an adapter;
    Tmax outputs are renamed to absolute-hours differences before publication.
    No extra p-values from these descriptive comparisons are reported.
    """
    definitions = {r["endpoint_key"]: r for r in definitions}
    outputs, common = [], []
    for before, after in [(BEFORE, AFTER), (BEFORE + "_document", AFTER + "_document")]:
        for rank in (1, 2, 3):
            rows = []
            for row in position_rows:
                if row["method_id"] not in (before, after) or row["rank"] != rank:
                    continue
                metric = definitions[row["endpoint_key"]]
                hours = metric["endpoint"] == "Tmax"
                rows.append({"method": row["method_id"], "scenario_id": row["scenario_id"],
                    "compound": row["compound"], "publication_cluster_id": row["publication_cluster_id"],
                    "endpoint": row["endpoint_key"], "stratum": row["analysis_stratum"] + ":" + row["prediction_kind"],
                    "top1_evaluable_probability": row["coverage"],
                    "top1_absolute_log_error": row["absolute_error_h" if hours else "absolute_log_error"]})
            paired = light.stable_paired_execution(rows, [before, after], baseline=before, rank=rank, draws=2000)
            by_key = {(r["method"], r["scenario_id"], r["endpoint"], r["stratum"]): r for r in rows}
            for raw in paired["comparisons"]:
                row = descriptive(raw)
                endpoint, stratum = row["endpoint"], row["stratum"]
                hours = definitions[endpoint]["endpoint"] == "Tmax"
                row["mean_paired_error_difference"] = row.pop("mean_paired_absolute_log_error_difference")
                row.update(rank=rank, metric_kind="absolute_hours" if hours else "absolute_log_error",
                           inference_role="descriptive", p_values_reported=False)
                outputs.append(row)
                shared = row["shared_full_coverage_scenarios"]
                values = {}
                for method in (before, after):
                    selected = [by_key[method, sid, endpoint, stratum] for sid in shared]
                    estimate = ev.hierarchical_mean(selected, "top1_absolute_log_error")
                    values[method] = estimate if hours or estimate is None else math.exp(estimate)
                common.append({"before": before, "after": after, "endpoint_key": endpoint, "source_stratum": stratum,
                    "rank": rank, "metric_kind": "MAE_h" if hours else "GMFE", "shared_scenarios": len(shared),
                    "scenario_ids": shared, "before_error": values[before], "after_error": values[after]})
    return {"comparisons": outputs, "common_positions": common}


def evaluate_curves(scores_by_method, query_ids, candidate_ids, candidates, source):
    candidate_records = {c["candidate_id"]: c for c in candidates}
    by_query = {qid: i for i, qid in enumerate(query_ids)}
    rows, rankings = [], []
    for case in source["cases"]:
        metrics = {c["candidate_id"]: curves.curve_errors(case, c) for c in case["curves"]}
        for method, scores in scores_by_method.items():
            ranking = {"method_id": method, "method": next(m["method"] for m in METHODS if m["method_id"] == method)}
            positions = curves.positions(scores[by_query[case["scenario_id"]]], candidate_ids, candidate_records, case)
            rankings.append({**ranking, "scenario_id": case["scenario_id"], "positions": positions})
            rows += [curves.score_position(case, ranking, pos, metrics) for pos in positions]
    return {"position_rows": rows, "rankings": rankings, "independent_publications": 1,
            "qualified_observations": 43, "intervals": None,
            "interpretation": "Two descriptive cases from one study; no significance or population curve claim."}


def joint_primary_holm(quality, execution):
    """Exactly two prespecified final-model comparisons; unavailable counts as 1."""
    rows = [
        {"comparison": "document_ndcg@10", "metric_kind": "NDCG difference", "higher_is_better": True,
         "estimate": quality.get("mean_group_delta"), "bootstrap_95ci": quality.get("bootstrap_95ci"),
         "independent_blocks": quality.get("independent_groups", 0), "p_raw": quality.get("cluster_sign_flip_p")},
        {"comparison": "ordinary_Cmax_rank1", "metric_kind": "absolute log error difference", "higher_is_better": False,
         "estimate": execution.get("mean_paired_absolute_log_error_difference"), "bootstrap_95ci": execution.get("bootstrap_95ci"),
         "independent_blocks": execution.get("independent_compound_publication_blocks", 0), "p_raw": execution.get("two_sided_p")},
    ]
    previous = 0.
    for index, row in enumerate(sorted(rows, key=lambda r: (r["p_raw"] if r["p_raw"] is not None else 1., r["comparison"]))):
        p = row["p_raw"] if row["p_raw"] is not None else 1.
        previous = max(previous, min(1., (2 - index) * p))
        row["p_holm"] = previous if row["p_raw"] is not None else None
        row["family_size"] = 2
        row["difference"] = "soft components minus original request, final PBPK-Match score"
    return rows


def descriptive(row):
    return {k: v for k, v in row.items() if k not in
            {"holm_p", "holm_adjusted_p", "two_sided_p", "cluster_sign_flip_p", "holm_family", "holm_prespecified_family_size"}}


def paired_results(reports, data):
    groups = {r["query_id"]: data["groups"][i] for i, r in enumerate(data["query_records"])}
    result = {"document": [], "execution": [], "common_positions": []}
    main_q, main_e = None, None
    for before, after in [(BEFORE, AFTER), (BEFORE + "_document", AFTER + "_document")]:
        quality = light.stable_paired_quality({m: reports[m]["quality"] for m in (before, after)}, groups, baseline=before, draws=2000)
        result["document"] += [descriptive(row) for row in quality]
        result["common_positions"] += light.common_coverage_comparisons({m: reports[m] for m in (before, after)}, own=after)
        for rank in (1, 2, 3):
            rows = [r for m in (before, after) for r in reports[m]["execution"]["position_rows"] if r["rank"] == rank]
            paired = light.stable_paired_execution(rows, [before, after], baseline=before, rank=rank, draws=2000)
            result["execution"] += [{"rank": rank, **descriptive(r)} for r in paired["comparisons"]]
            if after == AFTER and rank == 1:
                main_e = next(r for r in paired["comparisons"] if r["endpoint"] == "Cmax_reference_vs_group")
        if after == AFTER:
            main_q = next(r for r in quality if r["metric"] == "ndcg@10")
    require(main_q is not None and main_e is not None, "Prespecified primary comparison unavailable in registry")
    result["primary_family"] = joint_primary_holm(main_q, main_e)
    result["policy"] = {"bootstrap_draws": 2000, "clinical_minimum_independent_blocks": 10,
        "primary_tests": 2, "other_comparisons": "Descriptive estimates and intervals only; no additional significance tests.",
        "clinical_common_scope": "Both methods have full evaluable tie probability at the same exact rank and endpoint.",
        "clustering": "Existing connected drug/publication groups; no candidate/time-point resampling."}
    return result


def check_baseline_reproduction(reports, sources):
    source = light.OUT / "report/summary.json"
    receipt = read(light.OUT / "report/completion.json")
    require(receipt.get("completed") is True and reference(source) in receipt.get("outputs", []), "Original report is not bound")
    saved = read(source)["reports"]
    for method in (BEFORE, BEFORE + "_document"):
        for metric in ("ndcg@10", "recall@10", "mrr"):
            a, b = reports[method]["quality"]["judged_only"]["metrics"][metric], saved[method]["quality"]["judged_only"]["metrics"][metric]
            require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), "Baseline document metric changed")
        expected = {(r["endpoint"], r["rank"]): r for r in saved[method]["execution"]["position_summary"]}
        for row in reports[method]["execution"]["position_summary"]:
            for field in ("gmfe", "coverage", "within_twofold", "regret"):
                a, b = row.get(field), expected[row["endpoint"], row["rank"]].get(field)
                require((a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)),
                        "Baseline clinical metric changed: " + field)
    sources += [reference(source), reference(light.OUT / "report/completion.json")]


def build(predictions_path, manifest_path):
    sources = [reference(predictions_path), reference(manifest_path), reference(__file__)]
    manifest = read(manifest_path)
    data = load_dataset()
    sources.append(reference(light.ROOT / data["manifest"]["outputs"][0]["path"]))
    with np.load(scope(predictions_path), allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in ("indices", "scores", "base_scores", "probabilities")}
    mapping = validate_arrays(arrays, manifest, data)
    baseline, baseline_sha, bindings = load_baseline(data, arrays["indices"], sources)
    validate_integration_manifest(manifest, predictions_path, baseline_sha, bindings, sources)
    core_local = [mapping[i] for i in data["core_indices"]]
    execution_local = [mapping[i] for i in data["execution_indices"]]
    selection_path = ROOT / "experiments/model_recommendation/frontier_core_145_20260921/selection_manifest.json"
    selection = read(selection_path)
    binary_local = [j for j, i in enumerate(data["core_indices"])
                    if selection["compound_split"][data["query_records"][i]["compound_id"]] == "test"]
    require(len(binary_local) == 102, "Paper binary scope changed")
    sources += [reference(selection_path), reference(ev.PRIMARY)]
    contract = ev.clinical_contract()
    query_ids = [data["query_records"][i]["query_id"] for i in data["execution_indices"]]
    candidate_ids = manifest["candidate_ids"]
    reports, clinical_scores = {}, {}
    for method in METHODS:
        name = method["method_id"]
        values = arrays if name.startswith(AFTER) else baseline
        scores = values["base_scores" if name.endswith("_document") else "scores"]
        clinical_scores[name] = scores[execution_local]
        quality = evaluate_quality(scores, values["probabilities"], data, core_local, binary_local)
        app = quality["paper_binary_102"]
        require(app["known_pairs"] == 16410 and app["positive_pairs"] == 74, "Paper applicability denominator changed")
        clinical = ev.clinical_evaluation(clinical_scores[name], query_ids, candidate_ids, method=name, contract=contract)
        clinical["omission_summary"] = light.omission_summary(clinical["omissions"])
        reports[name] = {"quality": quality, "execution": clinical}
    check_baseline_reproduction(reports, sources)
    frozen = read(EXTENDED)
    require(frozen["state"] == "complete", "Extended endpoint source not complete")
    for source in frozen["sources"]:
        verify_ref(source, sources)
    sources.append(reference(EXTENDED))
    require(sha(curves.SOURCE) == curves.SOURCE_SHA, "Original curve source changed")
    sources.append(reference(curves.SOURCE))
    extra = evaluate_extended(clinical_scores, query_ids, candidate_ids, frozen)
    extra["omissions"] = extended_omissions(clinical_scores, query_ids, candidate_ids, frozen, contract)
    extra["paired"] = paired_extended(extra["position_rows"], extra["endpoint_definitions"])
    extra["candidate_errors_source"] = {**reference(EXTENDED), "json_pointer": "/candidate_errors"}
    curve = evaluate_curves(clinical_scores, query_ids, candidate_ids, data["candidate_records"], read(curves.SOURCE))
    paired = paired_results(reports, data)
    # Source hashes are verified again after all computations to detect concurrent mutation.
    sources = list({r["path"]: r for r in sources}.values())
    for source in sources:
        require(sha(source["path"]) == source["sha256"], "Input changed during evaluation")
    replay = manifest.get("purpose") == "zero_effect_replay_control_not_experiment"
    return {"schema": SCHEMA, "state": "complete", "methods": METHODS,
        "purpose": manifest.get("purpose", "paired_component_input_intervention"),
        "scope": {"document_requests": 145, "execution_requests": 41, "candidates": 280,
                  "paper_binary_requests": 102, "paper_binary_known_pairs": 16410, "ranking_positive_requests": 40},
        "reports": reports, "extended_endpoints": extra, "curves": curve, "paired": paired,
        "input_policy": {**REQUIRED_FLAGS, "information_change": ("Identity replay for evaluator verification; no components were added and this is not an experiment result." if replay else
                         "Validated soft components augment the visible request; original request, labels and prediction outcomes remain unchanged."),
                         "interpretation": "Paired input intervention with frozen outer-fold weights/alpha/temperature, not a retraining or controlled architecture comparison."},
        "validation": {"all_280_candidates": True, "baseline_metrics_reproduced": True,
                       "old_results_modified": False, "pbpk_models_reexecuted": False, "new_labels": False},
        "sources": sorted(sources, key=lambda r: r["path"])}


def json_default(value):
    """Normalize numerical library scalar types without permitting NaN/Inf."""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Unsupported report value: " + type(value).__name__)


def csv_text(rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False, default=json_default)
                         if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return buffer.getvalue()


def publish(payload, directory):
    directory = scope(directory)
    if directory.exists() and any(directory.iterdir()):
        marker = directory / "summary.json"
        require(marker.is_file() and read(marker).get("schema") == SCHEMA, "Refusing to write into an existing unrelated result directory")
    directory.mkdir(parents=True, exist_ok=True)
    reports = payload["reports"]
    outputs = {"summary.json": json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n"}
    for filename, rows in {
        "document_requests.csv": [{"method_id": m, **r} for m, v in reports.items() for r in v["quality"]["per_query"]],
        "document_summary.csv": [{"method_id": m, **v["quality"]["judged_only"]["metrics"],
             "same_drug_accuracy": v["quality"]["same_drug_different_family_preference"]["query_macro_accuracy"],
             "positive_f1_102": v["quality"]["paper_binary_102"]["positive_f1"],
             "known_top10": v["quality"]["known_top10_coverage"]["mean"]} for m, v in reports.items()],
        "execution_positions.csv": [r for v in reports.values() for r in v["execution"]["position_rows"]],
        "execution_summary.csv": [r for v in reports.values() for r in v["execution"]["position_summary"]],
        "omissions.csv": [{"method_id": m, **r} for m, v in reports.items() for r in v["execution"]["omission_summary"]],
        "extended_endpoint_positions.csv": payload["extended_endpoints"]["position_rows"],
        "extended_endpoint_summary.csv": payload["extended_endpoints"]["position_summary"],
        "extended_endpoint_omissions.csv": payload["extended_endpoints"]["omissions"],
        "extended_endpoint_common_positions.csv": payload["extended_endpoints"]["paired"]["common_positions"],
        "curve_positions.csv": payload["curves"]["position_rows"],
        "paired_common_positions.csv": payload["paired"]["common_positions"],
        "primary_paired_comparisons.csv": payload["paired"]["primary_family"],
    }.items():
        outputs[filename] = csv_text(rows)
    for filename, content in outputs.items():
        path = directory / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    (directory / "completion.json").write_text(json.dumps({"schema": SCHEMA, "state": "complete",
        "sources": payload["sources"], "outputs": [reference(directory / name) for name in outputs]}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.predictions, args.manifest)
    publish(payload, args.output)
    print(json.dumps({"state": "complete", "output": str(scope(args.output)), "scope": payload["scope"],
                      "primary_family": payload["paired"]["primary_family"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
