#!/usr/bin/env python3
"""Report a locked evidence-policy pilot without fitting or plotting.

Synthetic evidence interventions measure an explicitly specified engineering
policy. Clinical scoring remains conditional on the actual study references;
scenario copies and cached computations are never new independent drugs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from report_blueprint_validation import (
    CURVE, RESAMPLES, SEED, UNITS, Store, ValidationFault,
    _case_reference, _score_prediction, _unit_row,
    clustered_geomean, paired_ratio, positive,
)

VERSION = "blueprint-evidence-policy-report-1"
PROTOCOL_VERSION = "blueprint-evidence-policy-1"
METHODS = ("evidence_policy", "fixed_template", "rf_fill", "no_conflict_guard",
           "two_compartment", "two_compartment_prior")
SCENARIOS = ("available", "missing_fu", "conflicting_fu", "wrong_species_fu",
             "missing_required_pbpk_formula", "unsupported_ddi")
SAMPLING = ("full", "five")
EXCLUDED_DRUGS = ("Nemonoxacin", "Dexmedetomidine", "Midazolam")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _all_refs(value, store):
    """Check every embedded reference, including references inside ref metadata."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            store.checksum(value["path"], value["sha256"])
        for child in value.values():
            _all_refs(child, store)
    elif isinstance(value, list):
        for child in value:
            _all_refs(child, store)


def _expected_ready(scenario, method):
    if scenario == "unsupported_ddi":
        return False
    if method.startswith("two_compartment"):
        return True
    return scenario not in {"conflicting_fu", "missing_required_pbpk_formula"}


def _validate_matrix(protocol, state):
    """Reject missing attempts before any target clinical source is decoded."""
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("Unknown evidence-policy protocol version")
    if state.get("status") not in {"predictions_locked", "complete"} or (
            state.get("all_attempts_locked_before_scoring") is not True):
        raise ValueError("All prediction attempts must be locked before target outcomes may be decoded")
    for field, allowed in (("methods", METHODS), ("scenarios", SCENARIOS), ("sampling", SAMPLING)):
        declared = protocol.get(field)
        if not isinstance(declared, list) or len(declared) != len(set(declared)) or set(declared) != set(allowed):
            raise ValueError("Incomplete or duplicated experiment conditions: " + field)
    cases = protocol.get("cases", [])
    ids = [case["case_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Cases must have distinct nonempty identifiers")
    if any(not case.get("cluster_id") for case in cases):
        raise ValueError("Each case requires a predeclared study cluster")
    pletz = [case for case in cases if case.get("drug") in {"Ertapenem", "Ceftriaxone"}]
    if len(pletz) == 2 and pletz[0]["cluster_id"] != pletz[1]["cluster_id"]:
        raise ValueError("Ertapenem and Ceftriaxone share one clinical study cluster")
    expected_decisions = {(cid, scenario, method) for cid in ids for scenario in SCENARIOS for method in METHODS}
    decisions = protocol.get("decisions", [])
    keys = [(d["case_id"], d["scenario"], d["method"]) for d in decisions]
    if len(keys) != len(set(keys)) or set(keys) != expected_decisions:
        raise ValueError("Incomplete or duplicated engineering decisions")
    if len({d["id"] for d in decisions}) != len(decisions):
        raise ValueError("Engineering decision identifiers must be unique")
    for decision in decisions:
        if decision.get("status") not in {"ready", "blocked"}:
            raise ValueError("An engineering decision has not reached a terminal status")
        expected = _expected_ready(decision["scenario"], decision["method"])
        if decision.get("expected_ready") is not expected:
            raise ValueError("Engineering oracle differs from the predefined method capability")
        nested = decision.get("decision", {})
        if "status" in nested and nested["status"] != decision["status"]:
            raise ValueError("Decision wrapper and policy result disagree")
    decision_map = dict(zip(keys, decisions))
    units = protocol.get("units", [])
    unit_keys = [(u["case_id"], u["scenario"], u["sampling"], u["method"]) for u in units]
    expected_units = {(cid, scenario, sampling, method) for cid in ids for scenario in SCENARIOS
                      for sampling in SAMPLING for method in METHODS}
    if len(unit_keys) != len(set(unit_keys)) or set(unit_keys) != expected_units:
        raise ValueError("Incomplete or duplicated prediction attempt matrix")
    if len({u["id"] for u in units}) != len(units):
        raise ValueError("Prediction attempt identifiers must be unique")
    specs, computations = protocol.get("computations", {}), state.get("computations", {})
    referenced = set()
    for unit in units:
        decision = decision_map[unit["case_id"], unit["scenario"], unit["method"]]
        if unit.get("status") == "blocked":
            if decision["status"] != "blocked" or unit.get("computation_id") is not None:
                raise ValueError("Blocked attempts cannot execute or disagree with the decision")
            continue
        if unit.get("status") != "planned" or decision["status"] != "ready":
            raise ValueError("Planned attempts must follow a ready decision")
        identifier = unit.get("computation_id")
        if identifier not in specs or identifier not in computations:
            raise ValueError("A planned attempt has no completed computation")
        specification, computed = specs[identifier], computations[identifier]
        if (specification.get("case_id") != unit["case_id"] or
                specification.get("sampling") != unit["sampling"]):
            raise ValueError("Cached computation aliases a different drug or sampling input")
        if "signature" in specification or "computation_signature_sha256" in unit:
            signature = hashlib.sha256(json.dumps(specification.get("signature"), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if unit.get("computation_signature_sha256") != signature:
                raise ValueError("Computation alias does not match its frozen numerical input signature")
        if computed.get("status") not in {"predicted", "failed"}:
            raise ValueError("A computation remains unfinished")
        if computed["status"] == "predicted" and not all(
                isinstance(computed.get(name), dict) and {"path", "sha256"} <= computed[name].keys()
                for name in ("prediction", "request")):
            raise ValueError("Predicted computations must lock their curve and request")
        referenced.add(identifier)
    if set(computations) != set(specs) or referenced != set(specs):
        raise ValueError("Unknown or unreferenced physical computations")
    if not isinstance(protocol.get("source_pins"), dict) or not protocol["source_pins"]:
        raise ValueError("All original sources and code must be checksum locked")
    bootstrap = protocol.get("bootstrap", {"seed": SEED, "resamples": RESAMPLES})
    if bootstrap.get("seed") != SEED or bootstrap.get("resamples") != RESAMPLES:
        raise ValueError("The predeclared study-cluster bootstrap settings changed")


def engineering_summary(decisions):
    """Exact counts against the engineering oracle; no sampling duplication."""
    def group(selected):
        tp = sum(d["expected_ready"] and d["status"] == "ready" for d in selected)
        tn = sum(not d["expected_ready"] and d["status"] == "blocked" for d in selected)
        fp = sum(not d["expected_ready"] and d["status"] == "ready" for d in selected)
        fn = sum(d["expected_ready"] and d["status"] == "blocked" for d in selected)
        evidence = [item for d in selected for item in d.get("decision", {}).get("selected_evidence", [])]
        field_complete = sum(all(isinstance(item.get("source", {}).get(name), str) and item["source"][name]
                                 for name in ("path", "sha256", "location")) for item in evidence)
        actions = [item for d in selected for item in d.get("decision", {}).get("actions", [])]
        return {
            "requests": len(selected), "unique_cases": len({d["case_id"] for d in selected}),
            "oracle_supportable": tp + fn, "oracle_unsupportable": tn + fp,
            "ready": tp + fp, "blocked": tn + fn,
            "correct_allow": tp, "correct_block": tn, "incorrect_allow": fp, "incorrect_block": fn,
            "correct_allow_rate": _ratio(tp, tp + fn), "incorrect_allow_rate": _ratio(fp, tn + fp),
            "incorrect_block_rate": _ratio(fn, tp + fn),
            "decision_accuracy": _ratio(tp + tn, len(selected)),
            "coverage": _ratio(tp + fp, len(selected)),
            "risk_among_allowed": _ratio(fp, tp + fp),
            "selected_evidence_records": len(evidence),
            "source_field_complete_records": field_complete,
            "source_field_completeness": _ratio(field_complete, len(evidence)),
            "action_counts": dict(Counter(item.get("action", "unspecified") for item in actions)),
            "rejection_reason_counts": dict(Counter(item.get("reason", "unspecified") for item in actions
                                                   if item.get("action") == "reject")),
        }
    return {
        "by_method": {method: group([d for d in decisions if d["method"] == method]) for method in METHODS},
        "by_scenario": {scenario: {method: group([d for d in decisions
            if d["method"] == method and d["scenario"] == scenario]) for method in METHODS} for scenario in SCENARIOS},
        "interpretation": (
            "Predeclared engineering perturbations, not independent clinical observations or learned labels. "
            "Each decision is counted once, not once per sampling condition. Missing molecular formula "
            "blocks PBPK construction but does not invalidate an otherwise identified two-compartment "
            "plasma prediction. Unsupported DDI is outside every method's validated capability. "
            "Risk is the engineering oracle's incorrect allowance rate, not a clinical harm estimate. "
            "Source-field completeness verifies recorded fields only, not correctness of the original extraction."
        ),
    }


def clinical_summary(records, cases):
    """Aggregate separately by scenario; no artificial sample size from copies."""
    ids = [case["case_id"] for case in cases]
    clusters = {case["case_id"]: case["cluster_id"] for case in cases}
    by_key = {(r["case_id"], r["scenario"], r["sampling"], r["method"]): r for r in records}
    if len(by_key) != len(records):
        raise ValueError("Duplicate clinical condition records")

    def values(scenario, sampling, method, subset):
        return {cid: by_key[cid, scenario, sampling, method].get("curve_metrics", {}).get("gmfe") for cid in subset}

    def group(subset):
        output = {}
        for scenario in SCENARIOS:
            if scenario == "unsupported_ddi":
                continue
            output[scenario] = {}
            for sampling in SAMPLING:
                output[scenario][sampling] = {}
                for method in METHODS:
                    selected = [by_key[cid, scenario, sampling, method] for cid in subset]
                    raw = values(scenario, sampling, method, subset)
                    finite = {cid: value for cid, value in raw.items() if positive(value)}
                    curve = clustered_geomean(finite, {cid: clusters[cid] for cid in finite})
                    curve.update(
                        planned_requests=len(subset), blocked=sum(r["attempt_status"] == "blocked" for r in selected),
                        failed=sum(r["attempt_status"] == "failed" for r in selected),
                        predicted=sum(r["attempt_status"] == "predicted" for r in selected),
                        scored=len(finite), coverage=_ratio(len(finite), len(subset)),
                        predicted_but_unscorable=sum(r["attempt_status"] == "predicted" and
                            not positive(r.get("curve_metrics", {}).get("gmfe")) for r in selected),
                        unavailable_case_ids=[cid for cid in subset if cid not in finite],
                        within_twofold=sum(value <= 2 for value in finite.values()),
                        within_twofold_fraction_all_requests=_ratio(sum(value <= 2 for value in finite.values()), len(subset)),
                        within_twofold_fraction_scored=_ratio(sum(value <= 2 for value in finite.values()), len(finite)),
                        ci_condition="Available scored drugs within this scenario only; see coverage and failed/blocked counts",
                    )
                    endpoints = {}
                    for endpoint in UNITS:
                        entries = [r.get("endpoints", {}).get(endpoint, {}) for r in selected]
                        errors = {r["case_id"]: item["error"] for r, item in zip(selected, entries)
                                  if item.get("error") is not None}
                        references = sum(r.get("clinical_references", {}).get(endpoint, {}).get("value") is not None
                                         for r in selected)
                        stat = {"planned_requests": len(subset), "reference_available": references,
                                "predicted": sum(item.get("value") is not None for item in entries),
                                "scored": len(errors), "coverage_of_observed": _ratio(len(errors), references),
                                "weak_scored": sum(item.get("error") is not None and item.get("weak_tail", False)
                                                   for item in entries)}
                        if endpoint == "tmax":
                            stat["mean_absolute_error_h"] = math.fsum(errors.values()) / len(errors) if errors else None
                        else:
                            stat["fold_error"] = clustered_geomean(errors, {cid: clusters[cid] for cid in errors})
                            stat["within_twofold"] = sum(value <= 2 for value in errors.values())
                            stat["within_twofold_fraction_observed"] = _ratio(stat["within_twofold"], references)
                            stat["within_twofold_fraction_scored"] = _ratio(stat["within_twofold"], len(errors))
                        stat["nonweak_scored"] = sum(item.get("error") is not None and not item.get("weak_tail", False)
                                                     for item in entries)
                        endpoints[endpoint] = stat
                    output[scenario][sampling][method] = {"curve": curve, "endpoints": endpoints}
        return output

    def paired(subset):
        return {scenario: {sampling: {method: paired_ratio(
            values(scenario, sampling, "evidence_policy", subset), values(scenario, sampling, method, subset),
            clusters, expected_ids=subset) for method in METHODS if method != "evidence_policy"}
            for sampling in SAMPLING} for scenario in SCENARIOS if scenario != "unsupported_ddi"}

    sensitivity = {}
    for excluded in [(drug,) for drug in EXCLUDED_DRUGS] + [EXCLUDED_DRUGS]:
        subset = [case["case_id"] for case in cases if case["drug"] not in excluded]
        sensitivity["exclude_" + "_".join(drug.lower() for drug in excluded)] = {
            "excluded_drugs": list(excluded), "included_case_ids": subset,
            "groups": group(subset), "paired_policy_over_comparator": paired(subset),
        }
    return {
        "groups": group(ids), "paired_policy_over_comparator": paired(ids),
        "five_over_full": {scenario: {method: paired_ratio(values(scenario, "five", method, ids),
            values(scenario, "full", method, ids), clusters, expected_ids=ids) for method in METHODS}
            for scenario in SCENARIOS if scenario != "unsupported_ddi"},
        "prespecified_source_sensitivity": sensitivity,
        "independent_drugs": len(ids), "independent_study_clusters": len(set(clusters.values())),
        "interpretation": (
            "Previously viewed development cohort. Each scenario/sampling/method has at most one value "
            "per drug; intervention copies, formula trials and cached aliases are not independent data. "
            "Paired comparisons condition on common scored cases and must be read with coverage. "
            "Healthy single-drug clinical curves do not validate unsupported DDI requests."
        ),
    }


def _resource_summary(computations, units):
    fields = ("wall_seconds", "total_native_calls", "actual_new_native_calls", "cached_native_calls", "analytic_residual_calls",
              "manual_interventions", "peak_memory_mb")
    result = {"unique_computations": len(computations),
              "planned_logical_units": sum(unit["status"] == "planned" for unit in units),
              "blocked_logical_units": sum(unit["status"] == "blocked" for unit in units),
              "status_counts": dict(Counter(item["status"] for item in computations.values()))}
    for field in fields:
        known = [item.get("resources", {}).get(field) for item in computations.values()]
        known = [value for value in known if isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and value >= 0]
        result[field] = {"recorded_computations": len(known), "missing_computations": len(computations) - len(known),
                         "sum": math.fsum(known) if known and field != "peak_memory_mb" else None,
                         "maximum": max(known) if known else None}
    result["interpretation"] = (
        "Each physical computation is counted once even when reused across scenarios/methods. "
        "Wall-time sums are not parallel makespan; memory maxima are not simultaneous process-tree peaks. "
        "No time savings over manual literature curation are inferred."
    )
    return result


def report(run, project_root=ROOT):
    store = Store(project_root)
    run = store.path(run)
    if not run.is_dir():
        raise ValueError("The prepared experiment directory is required")
    protocol = store.read(run / "protocol.json")
    state = store.read(run / "state.json")
    _validate_matrix(protocol, state)
    # These are integrity reads. Target outcomes are decoded only below, after
    # both the lock and the complete condition matrix have passed validation.
    for name, expected in protocol["source_pins"].items():
        store.checksum(name, expected)
    _all_refs(protocol, store)
    _all_refs(state, store)
    if state["status"] == "complete" and (run / "results.json").exists():
        locked_result = state.get("results")
        if not isinstance(locked_result, dict) or store.path(locked_result.get("path", "")) != run / "results.json":
            raise ValueError("A completed report requires its recorded result checksum")
        existing = store.ref(locked_result)
        if existing.get("version") != VERSION or existing.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Completed report version differs from this protocol")
        for name, expected in existing.get("source_pins", {}).items():
            store.checksum(name, expected)
        store.unchanged()
        return existing
    reference = {case["case_id"]: _case_reference(case, store) for case in protocol["cases"]}
    cases = {case["case_id"]: case for case in protocol["cases"]}
    decisions = {(d["case_id"], d["scenario"], d["method"]): d for d in protocol["decisions"]}
    scored_cache, records, rows = {}, [], []
    for unit in protocol["units"]:
        case = cases[unit["case_id"]]
        clinical, curve, endpoints = reference[unit["case_id"]]
        decision = decisions[unit["case_id"], unit["scenario"], unit["method"]]
        computed = state["computations"].get(unit.get("computation_id"), {})
        status = "blocked" if unit["status"] == "blocked" else computed["status"]
        record = {
            "unit_id": unit["id"], "case_id": case["case_id"], "drug": case["drug"],
            "study_id": case["study_id"], "cluster_id": case["cluster_id"],
            "scenario": unit["scenario"], "sampling": unit["sampling"], "method": unit["method"],
            "attempt_status": status, "scoring_status": "not_scored",
            "computation_id": unit.get("computation_id"), "expected_ready": decision["expected_ready"],
            "failure": str(computed.get("failure", "")), "resources": deepcopy(computed.get("resources", {})),
            "prediction_source": deepcopy(computed.get("prediction", {})),
            "clinical_references": {name: deepcopy(endpoints[name]) for name in UNITS},
        }
        references = {"curve": curve, "endpoints": endpoints}
        if unit["scenario"] == "unsupported_ddi":
            record["scoring_status"] = "not_applicable"
            record["failure"] = "No matched clinical DDI reference; this is an engineering capability check only"
            record["clinical_references"] = {}
            references = {}
        elif status == "blocked":
            record["failure"] = "Evidence-policy decision blocked this requested construction"
        elif status == "failed":
            record["scoring_status"] = "failed"
        else:
            identity = (unit["computation_id"], case["case_id"])
            if identity not in scored_cache:
                try:
                    score, _, _ = _score_prediction(case, clinical, curve, endpoints, computed, store)
                    score = deepcopy(score)
                    score.pop("curve", None)  # Curves already reside in the locked prediction files.
                    scored_cache[identity] = {"score": score}
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    scored_cache[identity] = {"failure": str(exc),
                        "failure_category": getattr(exc, "category", "clinical_scoring_error")}
            cached = scored_cache[identity]
            if "score" in cached:
                record.update(deepcopy(cached["score"]))
                record["scoring_status"] = "scored" if positive(record.get("curve_metrics", {}).get("gmfe")) else "partial"
            else:
                record.update(cached)
                record["scoring_status"] = "failed"
        records.append(record)
        for endpoint in (CURVE, *UNITS):
            row = _unit_row(record, endpoint, references)
            row.update(scenario=unit["scenario"], computation_id=unit.get("computation_id"),
                       expected_ready=decision["expected_ready"],
                       resource_scope="Computation-level resource copied for traceability; never sum aliases")
            rows.append(row)
    result = {
        "version": VERSION, "protocol_version": protocol["version"],
        "all_attempts_locked_before_scoring": True,
        "cases": [{key: case.get(key) for key in ("case_id", "drug", "study_id", "cluster_id", "transfer_group")}
                  for case in protocol["cases"]],
        "engineering": engineering_summary(protocol["decisions"]),
        "statistics": clinical_summary(records, protocol["cases"]),
        "resources": _resource_summary(state["computations"], protocol["units"]),
        "decisions": deepcopy(protocol["decisions"]), "units": records,
        "source_pins": {name: checksum for name, checksum in store.pins.items()
                        if store.path(name) not in {run / "state.json", run / "results.json", run / "results.csv"}},
        "limitations": [
            "This experiment starts from frozen, manually curated evidence; literature retrieval and extraction are not automated here.",
            "Synthetic interventions establish consistency with a declared engineering policy, not clinical validity of the policy.",
            "Existing ten-drug clinical studies have already informed development; no untouched independent test is claimed.",
            "Scored curves/endpoints and coverage remain separate; no weighted overall winner is selected.",
            "Reference availability differs by endpoint; missing observations and failed predictions are never imputed.",
            "Simulated DDI capability requests have no corresponding clinical outcome in this pilot and are not numerically scored.",
        ],
    }
    store.unchanged()
    # Two files only. All source checks finish before either final file replaces
    # an existing report; the runner remains responsible for the run lock/state.
    with tempfile.TemporaryDirectory(prefix=".policy-report-", dir=run) as temp:
        temp = Path(temp)
        (temp / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        with (temp / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        store.unchanged()
        for name in ("results.csv", "results.json"):
            (temp / name).replace(run / name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = report(args.run)
    print(json.dumps({"report_version": result["version"], "logical_units": len(result["units"]),
                      "engineering_decisions": len(result["decisions"]),
                      "unique_computations": result["resources"]["unique_computations"]}))


if __name__ == "__main__":
    main()
