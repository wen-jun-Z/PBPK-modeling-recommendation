"""Freeze outcome-free clinical scenarios against the existing recommendation catalog.

Only protocol metadata, source audits and model-asset hashes are read. Sealed PK
outcomes, predictions and scores are never opened. Source-ready is not run-ready;
an audit for one exact model artifact is never transferred to another candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CORE = Path("experiments/model_recommendation/frontier_core_145_20260921/inputs")
EXTERNAL = Path("experiments/blueprint_design/external_validation")
LATEST = EXTERNAL / "clinical_cohort_round4_20260920/final_olanzapine_handoff/handoff_manifest.json"
DEFAULT_OUTPUT = Path("experiments/model_recommendation/execution_validation_20261001/inputs")
ALIASES = {"acetaminophen": "paracetamol", "ethinyl estradiol": "ethinylestradiol"}
FORBIDDEN_FIELDS = {"observed_values", "observed_concentrations", "concentration_values",
                    "outcome_values", "measured_values", "predicted_values", "observations"}


def normalize_compound(value):
    name = re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()
    return ALIASES.get(name, name)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def outcome_free(value, trail="protocol"):
    """Dose values and sampling times are allowed; measured outcomes are not."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_FIELDS:
                raise ValueError(f"Outcome-bearing field refused: {trail}.{key}")
            outcome_free(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            outcome_free(item, f"{trail}[{index}]")


class Inventory:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}
        self.hash_mismatches = []

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("All resources must stay inside this project")
        return path

    def read(self, ref, *, jsonl=False):
        relative = ref["path"] if isinstance(ref, dict) else str(ref)
        path = self.path(relative)
        parts = (*Path(relative).parts, *path.relative_to(self.root).parts)
        if any(p.startswith("sealed") or p in {"curator_only", "predictions", "scores"} for p in parts):
            raise ValueError(f"Refusing outcome/curator file: {relative}")
        raw = path.read_bytes()
        actual = digest(raw)
        record = {"path": str(path.relative_to(self.root)), "sha256": actual, "bytes": len(raw)}
        self.sources[record["path"]] = record
        if isinstance(ref, dict) and ref.get("sha256") not in (None, actual):
            mismatch = {**record, "declared_sha256": ref["sha256"], "state": "hash_mismatch"}
            if mismatch not in self.hash_mismatches:
                self.hash_mismatches.append(mismatch)
        text = raw.decode("utf-8-sig")
        return ([json.loads(line) for line in text.splitlines() if line.strip()]
                if jsonl else json.loads(text))

    def reference(self, path):
        return deepcopy(self.sources[str(self.path(path).relative_to(self.root))])

    def asset(self, relative, expected):
        """Stream model bytes for integrity only; do not parse embedded observations."""
        result = {"path": relative, "declared_sha256": expected,
                  "readable": False, "actual_sha256": None, "sha256_matches": False}
        if not relative:
            result["state"] = "source_path_missing"
            return result
        try:
            path = self.path(relative)
            if not path.is_file():
                result["state"] = "source_file_missing"
                return result
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            result.update(readable=True, actual_sha256=actual, bytes=path.stat().st_size,
                          sha256_matches=bool(expected and actual == expected),
                          state="verified" if expected and actual == expected else "hash_missing_or_mismatch")
        except (OSError, ValueError) as error:
            result.update(state="source_unreadable_or_outside_project", error_type=type(error).__name__)
        return result


def refs_in(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield {k: v for k, v in value.items() if k in {"path", "sha256", "bytes", "byte_size"}}
        for item in value.values():
            yield from refs_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from refs_in(item)


def sealed_refs(value):
    found = {}
    for ref in refs_in(value):
        if any(p.startswith("sealed") for p in Path(ref["path"]).parts):
            # Raw source articles are not outcome packets, although curator-only.
            if Path(ref["path"]).suffix == ".json" and "/sources/" not in ref["path"]:
                found[ref["path"]] = ref
    return list(found.values())


def discover(inv, latest=LATEST):
    """Follow the declared handoff ancestry; use its explicit active protocol dirs."""
    manifests, directories, supplemental_refs = [], set(), []
    receipt_by_case, protocol_pins = {}, {}

    def retain_receipt(data):
        for row in data.get("records", []):
            if row.get("case_id"):
                receipt_by_case[row["case_id"]] = row
            for key in ("protocol", "protocol_reference"):
                ref = row.get(key, {})
                if isinstance(ref, dict) and ref.get("path"):
                    protocol_pins[ref["path"]] = ref
    ref, visited = latest, set()
    latest_data = None
    while ref:
        path = ref["path"] if isinstance(ref, dict) else str(ref)
        if path in visited:
            raise ValueError("Cyclic handoff chain")
        visited.add(path)
        data = inv.read(ref)
        if latest_data is None:
            latest_data = data
        manifests.append(inv.reference(path))
        if data.get("curation_receipt"):
            retain_receipt(inv.read(data["curation_receipt"]))
        if data.get("qualifying_protocols_directory"):
            directories.add(data["qualifying_protocols_directory"])
        coverage = data.get("curation_coverage", {})
        for base in coverage.get("protocol_manifests", []):
            directory = inv.path(base)
            if (directory / "active_protocol_manifest.json").is_file():
                active = inv.read(str(Path(base) / "active_protocol_manifest.json"))
                manifests.append(inv.reference(str(Path(base) / "active_protocol_manifest.json")))
                for record in active["records"]:
                    supplemental_refs.append(record["protocol_reference"])
            else:
                directories.add(str(Path(base) / "protocols"))
        # Curves are extra views of the same dose cases, not new independent cases.
        supplemental_refs.extend(data.get("concentration_curve_coverage", {}).get("protocols", []))
        ref = data.get("parent_baseline_summary")
    protocols = []
    for directory in sorted(directories):
        base = inv.path(directory).parent
        receipt = base / "curation_receipt.json"
        if receipt.is_file():
            data = inv.read(str(receipt.relative_to(inv.root)))
            retain_receipt(data)
        for path in sorted(inv.path(directory).glob("*.json")):
            relative = str(path.relative_to(inv.root))
            data = inv.read(protocol_pins.get(relative, relative))
            if "case_id" in data:
                protocols.append((relative, data))
    for ref in supplemental_refs:
        data = inv.read(ref)
        protocols.append((ref["path"], data))
    # Repeated references are deduplicated by path, never by observed performance.
    by_path = dict(protocols)
    superseded = {r["path"] for _, d in by_path.items() for r in [d.get("supersedes_protocol", {})] if r.get("path")}
    active, views, decisions = {}, [], []
    for path, data in sorted(by_path.items()):
        outcome_free(data)
        if path in superseded:
            decisions.append({"path": path, "reason": "explicitly_superseded_protocol"})
            continue
        if data.get("prior_endpoint_protocol") or data.get("parent_protocol"):
            views.append((path, data))
            continue
        case = data["case_id"]
        if case in active and canonical(active[case][1]) != canonical(data):
            raise ValueError(f"Conflicting non-superseded case definitions: {case}")
        active[case] = (path, data)
    return latest_data, manifests, active, views, decisions, receipt_by_case


def normalized_design(data):
    p = data.get("protocol", {})
    dose = deepcopy(data.get("dose"))
    if not isinstance(dose, dict):
        dose = {"value": p.get("dose"), "unit": p.get("dose_unit"), "basis": p.get("dose_basis"),
                "active_moiety_dose_mg": p.get("active_moiety_dose_mg")}
    raw_route = dose.get("route") or p.get("route") or data.get("route")
    route = {"PO": "oral", "IV": "intravenous", "intravenous_infusion": "intravenous"}.get(raw_route, raw_route)
    sampling = deepcopy(data.get("sampling", {}))
    times = (data.get("nominal_sampling_hours_from_observation_origin") or data.get("sampling_times_h")
             or sampling.get("time_h") or sampling.get("times_h") or sampling.get("nominal_times_h")
             or sampling.get("times_h_from_observation_origin"))
    origin = sampling.get("time_origin") or sampling.get("origin") or p.get("sampling_time_origin") or data.get("observation_origin")
    if times is None and p.get("sampling_times_min") is not None:
        times = [v / 60 for v in p["sampling_times_min"]]
    if times is None:
        times = data.get("prediction_contract", {}).get("nominal_sampling_times_h")
    if times is not None and (not isinstance(times, list) or any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for v in times)):
        raise ValueError("Non-numeric sampling grid in outcome-free protocol")
    target = data.get("evaluation_target", {})
    prediction = data.get("prediction_contract", {})
    endpoints = deepcopy(data.get("endpoint_contracts"))
    if endpoints is None:
        endpoints = deepcopy(prediction.get("endpoints"))
    if endpoints is None:
        units = {e["endpoint"]: e for e in target.get("unit_mappings", [])}
        endpoints = [{"endpoint": e, **units.get(e, {}), "aggregation": target.get("aggregation")}
                     for e in target.get("endpoints", [])]
    regimen_keys = ("administrations_h", "administration_times_h", "dose_times_h", "infusion_duration_h",
                    "infusion_duration_min", "administrations", "infusion_duration", "schedule",
                    "administration_time_h", "administration_count", "administration_duration_h",
                    "dosing_interval_h", "regimen", "frequency", "coadministered_drugs")
    regimen = {k: deepcopy(source[k]) for source in (p, data, dose) for k in regimen_keys if k in source}
    dose_times = data.get("native_nominal_dose_times_h")
    if dose_times is None:
        for key in ("administrations_h", "administration_times_h", "dose_times_h"):
            if regimen.get(key) is not None:
                dose_times = regimen[key]
                break
    if dose_times is None and p.get("administrations"):
        rows = p["administrations"]
        if all("time_min" in row for row in rows):
            dose_times = [row["time_min"] / 60 for row in rows]
        elif all("time_h" in row for row in rows):
            dose_times = [row["time_h"] for row in rows]
    if dose_times is None and data.get("administration_count") == 1 and "administration_time_h" in data:
        dose_times = [data["administration_time_h"]]
    observation_origin_h = sampling.get("observation_origin_h_after_first_dose", dose.get("observation_origin_h"))
    origin_basis = "explicit_numeric_protocol_field" if observation_origin_h is not None else None
    if observation_origin_h is None and origin:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*h(?:ours)?\s+after\s+(?:the\s+)?first\s+dose", origin)
        if match:
            observation_origin_h = float(match[1])
            origin_basis = "literal_hours_after_first_dose_in_source_protocol_text"
    if observation_origin_h is None and dose_times == [0]:
        observation_origin_h = 0
        origin_basis = "single_recorded_dose_at_time_zero"
    infusion_h = regimen.get("infusion_duration_h")
    if infusion_h is None and regimen.get("infusion_duration_min") is not None:
        infusion_h = regimen["infusion_duration_min"] / 60
    regimen.update(dose_times_h=deepcopy(dose_times), infusion_duration_h=infusion_h,
                   observation_origin_h=observation_origin_h, observation_origin_basis=origin_basis,
                   observation_origin_text=origin)
    return {"route": route, "source_route": raw_route, "dose": dose,
            "formulation": dose.get("formulation") or p.get("formulation") or data.get("formulation"),
            "regimen": regimen, "sampling": {"times_h": times, "time_origin": origin, "source_metadata": sampling},
            "species": "human", "species_basis": "curated_primary_human_clinical_protocol",
            "population": deepcopy(data.get("population") or data.get("population_metadata") or {k: v for k, v in p.items() if k in
                {"population", "sex", "age_range_year", "ethnicity", "country", "reported_demographics"}}),
            "food": deepcopy(data.get("food") or p.get("food_state")),
            "matrix": data.get("matrix") or target.get("matrix") or prediction.get("matrix"),
            "analyte": data.get("analyte") or target.get("analyte") or prediction.get("analyte"),
            "endpoint_contracts": endpoints, "evaluation_target": deepcopy(target),
            "prediction_contract": deepcopy(prediction),
            "reference_subject_equals_population_mean": False}


def publication_cluster(data):
    identifiers = data.get("source_identifiers") or data.get("identifiers") or data.get("publication") or {}
    for key in ("doi", "pmcid", "pmid"):
        value = identifiers.get(key)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            if key == "doi":
                value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", value, flags=re.I).casefold()
            elif key == "pmcid":
                value = value.upper()
            else:
                value = re.sub(r"^PMID\s*:\s*", "", value, flags=re.I)
            return key + ":" + value
    return data.get("publication_cluster_id") or data.get("publication_group_id") or data.get("study_id")


def source_audits(inv, data):
    audit_refs, bound, statuses = [], [], []
    for key in ("source_independence_audit", "source_audit"):
        ref = data.get(key)
        if not isinstance(ref, dict) or not ref.get("path"):
            continue
        audit = inv.read(ref)
        audit_refs.append(inv.reference(ref["path"]))
        statuses.append(audit.get("status") or (
            "source_independence_supported_at_declared_scope" if audit.get("clinical_source_independence_supported_at_declared_scope") is True
            else "audit_record_present_scope_requires_review"))
        # Only explicitly named native model artifacts, not arbitrary bibliography.
        for name in ("model_snapshots", "source_snapshot", "native_snapshot", "snapshot", "selected_model"):
            bound.extend(refs_in(audit.get(name, [])))
    native = data.get("native_binding", {})
    for name in ("snapshot", "source_snapshot"):
        bound.extend(refs_in(native.get(name, {})))
    bound = list({(r["path"], r.get("sha256")): r for r in bound}.values())
    return {"audit_references": audit_refs, "historical_audit_statuses": statuses,
            "bound_snapshot_references": bound,
            "inline_audit": deepcopy(data.get("clinical_source_audit") or data.get("source_independence") or {}),
            "status": "source_level_audit_present_candidate_recheck_required" if audit_refs or data.get("clinical_source_audit") or data.get("source_independence") else "not_established",
            "binding_basis": "Exact source references in audit or paired protocol native_binding; no inferred family or filename equivalence.",
            "all_candidate_independence_verified": False}


def bind_candidate(scenario, candidate, asset):
    drug = scenario["compound"]
    match = drug in {normalize_compound(x) for x in candidate.get("compounds", [])}
    exact_refs = [r for r in scenario["source_independence"]["bound_snapshot_references"]
                  if r.get("sha256") and r["sha256"] == asset.get("actual_sha256")
                  and asset["sha256_matches"] and r["path"] == asset["path"]]
    audited = bool(exact_refs and scenario["source_independence"]["audit_references"])
    reasons = []
    if not match:
        status = "excluded_parent_identity_mismatch"
        reasons.append("direct_clinical_parent_prediction_requires_exact_registered_compound_identity")
    elif asset["state"] != "verified":
        status = "blocked_source_integrity"
        reasons.append(asset["state"])
    else:
        status = "pending_native_execution_and_source_review"
        reasons.extend(["native_protocol_mapping_not_yet_verified", "target_study_holdout_recheck_required"])
    return {"scenario_id": scenario["scenario_id"], "case_id": scenario["case_id"],
            "compound": drug, "candidate_id": candidate["candidate_id"], "family_id": candidate.get("family_id"),
            "family_name": candidate.get("family_name"), "registered_compounds": candidate.get("compounds", []),
            "compounds": candidate.get("compounds", []),
            "identity_match": match, "identity_rule": "normalized_exact_registry_name_with_frozen_aliases",
            "source_asset": deepcopy(asset), "candidate_record_sha256": digest(canonical(candidate).encode()),
            "catalog_provenance": deepcopy(candidate.get("provenance", {})),
            "bound_snapshot_exact_match": bool(exact_refs), "exact_bound_references": exact_refs,
            "independence_status": "previously_audited_pending_recheck" if match and audited else "not_audited_for_this_candidate",
            "independence_verified": False, "execution_status": status, "reasons": reasons,
            "scientific_applicability": "unknown", "execution_failure_is_scientific_nonapplicability": False}


def frozen_write(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite frozen cohort file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def build(root=ROOT, output=DEFAULT_OUTPUT, *, latest=LATEST, include_caffeine=True):
    inv = Inventory(root)
    out = inv.path(output)
    candidates = inv.read(str(CORE / "candidates.jsonl"), jsonl=True)
    queries = inv.read(str(CORE / "queries.jsonl"), jsonl=True)
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate frozen candidate identity")
    latest_data, ancestry, active, views, dedup, receipt_records = discover(inv, latest)
    expected_drugs = {normalize_compound(d) for d in latest_data["drug_names"]}
    if set(normalize_compound(d["compound"]) for _, d in active.values()) != expected_drugs:
        raise ValueError("Discovered clinical drugs do not match the latest handoff; no silent partial cohort")
    if len(active) != latest_data["clinical_protocol_metadata_total"]:
        raise ValueError("Discovered protocol count does not match latest handoff")
    declared_count = len(active)
    if include_caffeine:
        design_path = str(EXTERNAL / "sherman2022_v1/design.json")
        design = inv.read(design_path)
        if design.get("contains_observed_pk_values") is not False:
            raise ValueError("Caffeine design lacks outcome-free assertion")
        receipt = inv.read(str(EXTERNAL / "sherman2022_v1/curation_receipt.json"))
        for arm in design["cases"]:
            value = {**deepcopy(design), **deepcopy(arm)}
            value.pop("cases")
            value["cohort_id"] = arm["case_id"]
            value["sealed_observation_reference"] = next((r for r in receipt["files"] if r["path"].endswith("/observations.json")), None)
            value["historical_outcome_access"] = {"curator_had_prior_access": receipt["curator_had_viewed_published_values_during_prior_source_audit"],
                                                   "numeric_outcome_values_shared_with_prediction_runner": receipt["numeric_outcome_values_shared_with_prediction_runner"]}
            active[arm["case_id"]] = (design_path, value)
    assets = {}
    for candidate in candidates:
        provenance = candidate.get("provenance", {})
        key = (provenance.get("source_path"), provenance.get("source_sha256"))
        if key not in assets:
            assets[key] = inv.asset(*key)
    scenarios, audit_rows = [], []
    for case, (path, data) in sorted(active.items()):
        outcome_free(data)
        drug = normalize_compound(data["compound"])
        related_views = [(vp, vd) for vp, vd in views if
                         vd.get("prior_endpoint_protocol", vd.get("parent_protocol", {})).get("path") == path]
        related_receipt = receipt_records.get(case, {})
        independent = source_audits(inv, data)
        designs = []
        for vp, vd in related_views:
            view_audit = source_audits(inv, vd)
            designs.append({"source_protocol": inv.reference(vp), "protocol_fields": deepcopy(vd),
                            "normalized_design": normalized_design(vd), "source_independence": view_audit})
        references = {r["path"]: r for r in sealed_refs(data) + sealed_refs(related_receipt)}
        for _, view in related_views:
            references.update({r["path"]: r for r in sealed_refs(view)})
        design = normalized_design(data)
        blockers = []
        if not references:
            blockers.append("sealed_observation_reference_not_resolved")
        if design["dose"].get("value") is None or not design["dose"].get("unit"):
            blockers.append("numeric_dose_or_unit_unresolved")
        if not design["sampling"]["times_h"]:
            blockers.append("nominal_sampling_grid_unresolved")
        if design["regimen"]["dose_times_h"] is None:
            blockers.append("nominal_dose_schedule_unresolved")
        if design["regimen"]["observation_origin_h"] is None:
            blockers.append("numeric_sampling_origin_unresolved")
        if data.get("protocol_execution_ambiguities"):
            blockers.append("source_protocol_execution_ambiguity")
        if any(e.get("evaluable_from_source") is False for e in design["endpoint_contracts"] if isinstance(e, dict)):
            blockers.append("one_or_more_endpoints_not_source_qualified")
        matching = [c["candidate_id"] for c in candidates if drug in {normalize_compound(x) for x in c.get("compounds", [])}]
        if not matching:
            blockers.append("no_exact_parent_in_frozen_catalog")
        scenario = {"schema_version": "pbpk-recommendation-execution-scenario-1", "scenario_id": "clinical:" + case,
                    "case_id": case, "compound": drug, "cohort_origin": "latest_clinical_handoff" if drug in expected_drugs else "separate_caffeine_external_design",
                    "study_id": data.get("study_id") or data.get("publication_cluster_id") or data.get("cohort_id"),
                    "publication_ids": deepcopy(data.get("source_identifiers") or data.get("identifiers") or data.get("publication") or {}),
                    "publication_cluster_id": publication_cluster(data),
                    "declared_publication_cluster_id": data.get("publication_cluster_id") or data.get("publication_group_id"),
                    "cohort_id": data.get("cohort_id") or data.get("study_id") or case,
                    "source_protocol": inv.reference(path), "protocol_fields": deepcopy(data),
                    "normalized_design": design, "curve_views": designs,
                    "sealed_references": list(references.values()), "sealed_files_opened": False,
                    "source_role": "previously_curated_external_clinical_candidate_pending_candidate_specific_recheck",
                    "source_independence": independent, "catalog_candidate_ids": matching,
                    "core145_query_ids": [q["query_id"] for q in queries if drug in {normalize_compound(x) for x in q.get("compounds", [])}],
                    "core145_mapping_scope": "same_parent_only_not_exact_study_dose_or_endpoint_match",
                    "readiness": "metadata_frozen_execution_pending", "blockers": blockers,
                    "execution_policy": {"allow_reference_subject_approximation": True,
                                         "purpose": "separate diagnostic stratum; never direct-reuse scientific truth",
                                         "explicit_source_prohibition_still_applies": True,
                                         "all_population_and_formulation_approximations_must_be_frozen_before_outcome_access": True},
                    "clinical_experiment_completed": False, "outcomes_used_for_case_selection": False}
        scenarios.append(scenario)
        for candidate in candidates:
            provenance = candidate.get("provenance", {})
            asset = assets[(provenance.get("source_path"), provenance.get("source_sha256"))]
            audit_rows.append(bind_candidate(scenario, candidate, asset))
    mismatch_paths = {r["path"] for r in inv.hash_mismatches}
    for scenario in scenarios:
        refs = [scenario["source_protocol"], *scenario["source_independence"]["audit_references"]]
        if any(r["path"] in mismatch_paths for r in refs):
            scenario["blockers"].append("declared_metadata_hash_changed_review_required")
    # All catalog candidates remain in the audit, including identity exclusions.
    scenario_bytes = ("\n".join(canonical(r) for r in scenarios) + "\n").encode()
    audit_bytes = ("\n".join(canonical(r) for r in audit_rows) + "\n").encode()
    drug_coverage = []
    for drug in sorted(expected_drugs | {s["compound"] for s in scenarios}):
        matched = [c for c in candidates if drug in {normalize_compound(x) for x in c.get("compounds", [])}]
        drug_coverage.append({"compound": drug, "from_latest_37_drug_handoff": drug in expected_drugs,
                             "scenario_count": sum(s["compound"] == drug for s in scenarios),
                             "catalog_candidates": len(matched), "registered_families": len({c["family_id"] for c in matched}),
                             "core145_requests": sum(drug in {normalize_compound(x) for x in q.get("compounds", [])} for q in queries)})
    report = {"schema_version": "pbpk-recommendation-execution-cohort-1", "state": "metadata_inventory_complete",
              "scope": {"latest_handoff_drugs": len(expected_drugs), "latest_handoff_scenarios": declared_count,
                        "all_scenarios": len(scenarios), "all_compounds": len(drug_coverage),
                        "publication_clusters": len({s["publication_cluster_id"] for s in scenarios if s["publication_cluster_id"]}),
                        "curve_views_attached": sum(len(s["curve_views"]) for s in scenarios),
                        "catalog_candidates": len(candidates), "core145_requests": len(queries),
                        "audit_rows": len(audit_rows), "identity_matching_pairs": sum(r["identity_match"] for r in audit_rows),
                        "previously_audited_exact_bindings": sum(r["independence_status"] == "previously_audited_pending_recheck" for r in audit_rows),
                        "clinical_execution_completed": 0, "sealed_outcome_files_read": 0},
              "latest_handoff": inv.reference(latest), "handoff_ancestry": ancestry, "deduplication": dedup,
              "builder_sha256": digest(Path(__file__).read_bytes()),
              "source_hash_mismatches": inv.hash_mismatches, "compound_coverage": drug_coverage,
              "candidate_execution_states": dict(Counter(r["execution_status"] for r in audit_rows)),
              "frozen_identity_aliases": ALIASES, "source_files": dict(sorted(inv.sources.items())),
              "outputs": {"scenarios.jsonl": {"sha256": digest(scenario_bytes), "rows": len(scenarios)},
                          "candidate_audit.jsonl": {"sha256": digest(audit_bytes), "rows": len(audit_rows)}},
              "limitations": ["Clinical source readiness does not establish runnable candidate or independent accuracy.",
                              "Original 145 requests have only drug-identity links; new exact clinical requests require fresh frozen ranking.",
                              "The same target publication/cohort and derived evidence must be excluded from fitting, retrieval, selection and calibration.",
                              "Previous audit belongs only to an exact path-and-hash-bound candidate and still requires current recheck.",
                              "Candidate execution, endpoint semantics, source ancestry and forecast locks precede outcome access.",
                              "Repeated doses, source curve views and formulation periods are not independent studies.",
                              "No blueprint prediction or outcome value is used as a candidate-execution substitute."],
              "reproduce": "runtime/app-env/bin/python scripts/build_recommendation_execution_cohort.py"}
    frozen_write(out / "scenarios.jsonl", scenario_bytes)
    frozen_write(out / "candidate_audit.jsonl", audit_bytes)
    frozen_write(out / "cohort.json", (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--without-caffeine", action="store_true")
    args = parser.parse_args()
    result = build(output=args.output, include_caffeine=not args.without_caffeine)
    print(json.dumps({"state": result["state"], "scope": result["scope"], "metadata_hash_mismatches": len(result["source_hash_mismatches"])}, ensure_ascii=False))
