from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.processing.final_evidence_attribution_stage import (
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    load_json,
    utc_now,
)


FINAL_ROLES = {"target_compound", "victim_drug", "perpetrator_drug"}
BLOCKING_CONFLICT_STATUSES = {"unresolved_conflict", "insufficient_context"}
CONTEXT_FIELDS = ("route", "population", "condition", "dose", "time_point", "study_design", "label_version")
DIRECTION_PATTERNS = {
    "no_effect": (
        r"\bno (?:clinically relevant |meaningful |significant )?(?:effect|change|interaction)\b",
        r"\bunchanged\b",
        r"\bdid not (?:affect|alter|change|influence)\b",
        r"\bnot (?:affect|alter|change|influence)\b",
    ),
    "increase": (r"\bincreas(?:e|ed|es|ing)\b", r"\brais(?:e|ed|es|ing)\b", r"\bhigher\b", r"\belevat(?:e|ed|es|ing)\b"),
    "decrease": (r"\bdecreas(?:e|ed|es|ing)\b", r"\breduc(?:e|ed|es|ing)\b", r"\blower(?:ed|ing)?\b", r"\bdiminish(?:ed|es|ing)?\b"),
}
CONTROLLED_CONDITION_PATTERNS = {
    "fed": (r"\bfed\b", r"\bwith food\b", r"\bpostprandial\b"),
    "fasted": (r"\bfast(?:ed|ing)\b", r"\bwithout food\b"),
    "renal_impairment": (r"\brenal impairment\b", r"\bkidney impairment\b", r"\bchronic kidney disease\b"),
    "hepatic_impairment": (r"\bhepatic impairment\b", r"\bliver impairment\b", r"\bcirrhosis\b"),
    "pregnancy": (r"\bpregnan(?:cy|t)\b",),
    "lactation": (r"\blactat(?:ion|ing)\b", r"\bbreastfeed(?:ing)?\b"),
    "pediatric": (r"\bpediatric\b", r"\bpaediatric\b", r"\bchildren\b", r"\binfants?\b", r"\bneonates?\b"),
    "geriatric": (r"\bgeriatric\b", r"\belderly\b", r"\bolder adults?\b"),
    "healthy": (r"\bhealthy (?:subjects?|volunteers?|adults?)\b",),
    "male": (r"\bmales?\b", r"\bmen\b"),
    "female": (r"\bfemales?\b", r"\bwomen\b"),
    "smoker": (r"\bsmokers?\b",),
    "non_smoker": (r"\bnon[- ]?smokers?\b",),
    "poor_metabolizer": (r"\bpoor metaboli[sz]ers?\b",),
    "intermediate_metabolizer": (r"\bintermediate metaboli[sz]ers?\b",),
    "extensive_metabolizer": (r"\bextensive metaboli[sz]ers?\b", r"\bnormal metaboli[sz]ers?\b"),
    "ultrarapid_metabolizer": (r"\bultra[- ]?rapid metaboli[sz]ers?\b",),
    "mild": (r"\bmild\b",),
    "moderate": (r"\bmoderate\b",),
    "severe": (r"\bsevere\b",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final-evidence conflict analysis and build evidence layers.")
    parser.add_argument("--raw-processing-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset", action="store_true", help="Intentionally replace prior Step 6/7 machine outputs.")
    args = parser.parse_args()

    processor = FinalEvidencePostProcessor(
        raw_processing_dir=args.raw_processing_dir,
        output_dir=args.output_dir,
        reset=args.reset,
    )
    print(json.dumps(processor.run(), ensure_ascii=False, indent=2, sort_keys=True))


class FinalEvidencePostProcessor:
    def __init__(self, *, raw_processing_dir: Path, output_dir: Path, reset: bool) -> None:
        self.raw_processing_dir = raw_processing_dir
        self.output_dir = output_dir
        self.reset = reset
        self.manifest_dir = output_dir / "00_run_manifest"
        self.attribution_dir = output_dir / "01_compound_attribution"
        self.domain_dir = output_dir / "02_domain_gates"
        self.dedup_dir = output_dir / "03_deduplication"
        self.conflict_dir = output_dir / "04_conflict_analysis"
        self.layers_dir = output_dir / "05_evidence_layers"

    def run(self) -> dict[str, Any]:
        state_path = self.manifest_dir / "pipeline_state.json"
        state = load_json(state_path)
        normalize_stage_summary_paths(state, self.output_dir)
        validate_stage_order(state, reset=self.reset, layers_dir=self.layers_dir)

        inputs = self._load_inputs()
        audit = audit_prior_stages(inputs, state, self.output_dir)
        atomic_write_json(self.manifest_dir / "step1_5_audit.json", audit)
        if not audit["accepted"]:
            raise RuntimeError(f"Step 1-5 audit failed: {audit['errors']}")

        conflict_report, conflict_members, conflict_summary = analyze_conflicts(inputs)
        atomic_write_jsonl(self.conflict_dir / "evidence_conflict_report.jsonl", conflict_report)
        atomic_write_jsonl(self.conflict_dir / "conflict_members.jsonl", conflict_members)
        atomic_write_json(self.conflict_dir / "conflict_summary.json", conflict_summary)
        if not conflict_summary["accepted"]:
            raise RuntimeError("Conflict analysis failed its acceptance checks.")

        layer_outputs, layer_summary = build_evidence_layers(inputs, conflict_report, conflict_members)
        for filename, rows in layer_outputs.items():
            atomic_write_jsonl(self.layers_dir / filename, rows)
        atomic_write_json(self.layers_dir / "evidence_layers_summary.json", layer_summary)
        if not layer_summary["accepted"]:
            raise RuntimeError("Evidence layers failed their acceptance checks.")

        self._update_manifests(state, audit, conflict_summary, layer_summary)
        return {
            "run_id": state.get("run_id"),
            "step1_5_audit": audit,
            "conflict_analysis": conflict_summary,
            "evidence_layers": layer_summary,
            "next_required_steps": ["rag_index", "database_ui_export"],
        }

    def _load_inputs(self) -> dict[str, list[dict[str, Any]]]:
        paths = {
            "candidates": self.raw_processing_dir / "04_evidence_candidates" / "evidence_candidates.jsonl",
            "provenance": self.manifest_dir / "provenance_validation.jsonl",
            "identities": self.attribution_dir / "compound_identity.jsonl",
            "attributions": self.attribution_dir / "compound_attribution.jsonl",
            "domain_results": self.domain_dir / "domain_gate_results.jsonl",
            "deterministic_groups": self.dedup_dir / "deterministic_duplicate_groups.jsonl",
            "deterministic_members": self.dedup_dir / "deterministic_duplicate_members.jsonl",
            "deterministic_representatives": self.dedup_dir / "deterministic_representatives.jsonl",
            "semantic_input": self.dedup_dir / "semantic_dedup_input.jsonl",
            "semantic_groups": self.dedup_dir / "semantic_duplicate_groups.jsonl",
            "semantic_members": self.dedup_dir / "semantic_duplicate_members.jsonl",
            "semantic_conflicts": self.dedup_dir / "semantic_conflict_candidates.jsonl",
            "semantic_state": self.dedup_dir / "semantic_dedup_state.jsonl",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing final-evidence prerequisite files: {missing}")
        return {name: list(iter_jsonl(path)) for name, path in paths.items()}

    def _update_manifests(
        self,
        state: dict[str, Any],
        audit: dict[str, Any],
        conflict_summary: dict[str, Any],
        layer_summary: dict[str, Any],
    ) -> None:
        state["updated_at"] = utc_now()
        state["current_stage"] = "reviewable_evidence_ready"
        state["stages"]["conflict_analysis"] = {
            "status": "completed",
            "accepted": True,
            "report_count": conflict_summary["report_count"],
            "blocking_group_count": conflict_summary["blocking_group_count"],
            "summary_path": "04_conflict_analysis/conflict_summary.json",
        }
        state["stages"]["evidence_layers"] = {
            "status": "completed",
            "accepted": True,
            "preselected_count": layer_summary["preselected_count"],
            "reviewable_count": layer_summary["reviewable_count"],
            "excluded_count": layer_summary["excluded_from_reviewable_count"],
            "summary_path": "05_evidence_layers/evidence_layers_summary.json",
        }
        state["stages"]["human_review"] = {
            "status": "optional",
            "required_for_pipeline": False,
            "requires_human_action": False,
            "accepted_count": 0,
            "modeling_count": 0,
        }
        normalize_stage_summary_paths(state, self.output_dir)
        atomic_write_json(self.manifest_dir / "pipeline_state.json", state)

        quality_path = self.manifest_dir / "evidence_quality_report.json"
        quality = load_json(quality_path)
        quality.update(
            {
                "generated_at": utc_now(),
                "step1_5_audit": audit,
                "conflict_analysis": conflict_summary,
                "evidence_layers": layer_summary,
                "later_steps_executed": True,
                "next_required_steps": ["rag_index", "database_ui_export"],
            }
        )
        atomic_write_json(quality_path, quality)


def validate_stage_order(state: dict[str, Any], *, reset: bool, layers_dir: Path) -> None:
    for stage_name in (
        "provenance",
        "compound_attribution",
        "domain_quality_gates",
        "deterministic_deduplication",
        "semantic_deduplication",
    ):
        stage = (state.get("stages") or {}).get(stage_name) or {}
        if stage.get("status") != "completed" or stage.get("accepted") is not True:
            raise RuntimeError(f"{stage_name} must be completed and accepted before conflict analysis.")
    human_review = (state.get("stages") or {}).get("human_review") or {}
    accepted_path = layers_dir / "accepted_evidence.jsonl"
    modeling_path = layers_dir / "modeling_evidence.jsonl"
    reviewed_data_exists = any(path.exists() and path.stat().st_size for path in (accepted_path, modeling_path))
    if human_review.get("status") == "completed" or reviewed_data_exists:
        raise RuntimeError("Refusing to rebuild machine layers after human-reviewed evidence exists.")
    if not reset:
        for stage_name in ("conflict_analysis", "evidence_layers"):
            if ((state.get("stages") or {}).get(stage_name) or {}).get("status") == "completed":
                raise RuntimeError(f"{stage_name} is already completed; use --reset for an intentional rebuild.")


def audit_prior_stages(inputs: dict[str, list[dict[str, Any]]], state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    for stage_name in ("provenance", "compound_attribution", "domain_quality_gates", "deterministic_deduplication", "semantic_deduplication"):
        stage = (state.get("stages") or {}).get(stage_name) or {}
        summary_ref = stage.get("summary_path")
        summary: dict[str, Any] = {}
        if stage.get("status") != "completed" or stage.get("accepted") is not True:
            errors.append(f"{stage_name} is not completed and accepted")
        if not isinstance(summary_ref, str) or not summary_ref:
            errors.append(f"{stage_name} lacks an authoritative summary path")
        else:
            summary_path = (output_dir / summary_ref).resolve()
            if not summary_path.is_relative_to(output_dir.resolve()):
                errors.append(f"{stage_name} summary is outside the final-evidence run")
            else:
                try:
                    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict) or loaded.get("accepted") is not True:
                        errors.append(f"{stage_name} summary is not accepted")
                    else:
                        summary = loaded
                except (OSError, ValueError) as exc:
                    errors.append(f"{stage_name} summary could not be read: {type(exc).__name__}")
        summaries[stage_name] = summary

    def check_count(stage_name: str, field: str, actual: int) -> None:
        expected = summaries[stage_name].get(field)
        if type(expected) is not int or expected < 0:
            errors.append(f"{stage_name} summary lacks a valid {field}")
        elif expected != actual:
            errors.append(f"{stage_name} {field} mismatch: expected {expected}, found {actual}")

    candidates = inputs["candidates"]
    candidate_by_id = unique_map(candidates, "evidence_id", "candidate", errors)
    candidate_ids = set(candidate_by_id)
    for stage_name in ("provenance", "compound_attribution", "domain_quality_gates"):
        check_count(stage_name, "candidate_count", len(candidates))

    required_fields = (
        "evidence_id",
        "run_id",
        "source_id",
        "raw_asset_id",
        "storage_path",
        "checksum",
        "source_location",
        "source_text",
        "source_text_hash",
        "extraction_method",
        "extractor_name",
        "extractor_version",
        "curation_status",
    )
    missing_fields: Counter[str] = Counter()
    hash_mismatch = 0
    missing_storage = 0
    for row in candidates:
        for field in required_fields:
            if row.get(field) in (None, ""):
                missing_fields[field] += 1
        digest = hashlib.sha256(str(row.get("source_text") or "").encode("utf-8", errors="replace")).hexdigest()
        hash_mismatch += digest != row.get("source_text_hash")
        missing_storage += not Path(str(row.get("storage_path") or "")).exists()
    if missing_fields:
        errors.append(f"missing required candidate fields: {dict(missing_fields)}")
    if hash_mismatch:
        errors.append(f"source_text_hash mismatch count: {hash_mismatch}")
    if missing_storage:
        errors.append(f"missing storage_path count: {missing_storage}")

    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ("provenance", "attributions", "domain_results"):
        maps[name] = unique_map(inputs[name], "evidence_id", name, errors)
        if set(maps[name]) != candidate_ids:
            errors.append(f"{name} evidence_id coverage differs from candidates")
    if any(row.get("provenance_status") != "passed" for row in inputs["provenance"]):
        errors.append("provenance contains non-passed rows")
    if any(row.get("attribution_status") != "resolved" for row in inputs["attributions"]):
        errors.append("compound attribution contains unresolved rows")

    identity_ids = {str(row.get("compound_id") or "") for row in inputs["identities"]}
    check_count("compound_attribution", "compound_identity_count", len(inputs["identities"]))
    if "" in identity_ids or len(identity_ids) != len(inputs["identities"]):
        errors.append("compound identity table contains missing or duplicate compound IDs")
    if any(str(row.get("compound_id") or "") not in identity_ids for row in inputs["attributions"]):
        errors.append("attribution references an unknown compound identity")

    domain_statuses = Counter(str(row.get("domain_gate_status") or "") for row in inputs["domain_results"])
    for status in ("passed", "needs_manual_review", "excluded"):
        check_count("domain_quality_gates", f"{status}_count", domain_statuses[status])
    if set(domain_statuses) - {"passed", "needs_manual_review", "excluded"}:
        errors.append("domain results contain an unsupported status")
    if any(row.get("domain_gate_status") != "passed" and not row.get("reasons") for row in inputs["domain_results"]):
        errors.append("non-passed domain result lacks a reason")

    representative_ids = {str(row.get("evidence_id") or "") for row in inputs["deterministic_representatives"]}
    nonrepresentative_ids = {
        str(row.get("evidence_id") or "")
        for row in inputs["deterministic_members"]
        if row.get("is_representative") is False
    }
    eligible_ids = {
        evidence_id
        for evidence_id, row in maps["domain_results"].items()
        if row.get("domain_gate_status") != "excluded"
    }
    if representative_ids & nonrepresentative_ids or representative_ids | nonrepresentative_ids != eligible_ids:
        errors.append("deterministic dedup does not exactly cover non-excluded domain rows")
    semantic_input_ids = {str(row.get("evidence_id") or "") for row in inputs["semantic_input"]}
    check_count("deterministic_deduplication", "representative_count", len(inputs["deterministic_representatives"]))
    check_count("deterministic_deduplication", "semantic_input_count", len(inputs["semantic_input"]))
    check_count("semantic_deduplication", "input_rows", len(inputs["semantic_input"]))
    if "" in semantic_input_ids or len(semantic_input_ids) != len(inputs["semantic_input"]):
        errors.append("semantic input contains missing or duplicate evidence IDs")
    if not semantic_input_ids <= representative_ids:
        errors.append("semantic input contains a non-representative row")
    if any(maps["domain_results"][evidence_id].get("domain_gate_status") != "passed" for evidence_id in semantic_input_ids):
        errors.append("semantic input contains a row that did not pass domain gates")

    semantic_group_by_id = unique_map(inputs["semantic_groups"], "duplicate_group_id", "semantic group", errors)
    seen_member_type: set[tuple[str, str]] = set()
    for member in inputs["semantic_members"]:
        group = semantic_group_by_id.get(str(member.get("duplicate_group_id") or ""))
        if group is None:
            errors.append("orphan semantic member")
            continue
        if member.get("evidence_id") not in semantic_input_ids:
            errors.append("semantic member is absent from semantic input")
        if member.get("group_type") != group.get("group_type"):
            errors.append("semantic member group_type mismatch")
        key = (str(member.get("evidence_id") or ""), str(member.get("group_type") or ""))
        if key in seen_member_type:
            errors.append("semantic evidence repeats within one group type")
        seen_member_type.add(key)
    automatic_members = [row for row in inputs["semantic_members"] if row.get("group_type") == "semantic_duplicate"]
    if automatic_members and min(float(row.get("similarity_score") or 0.0) for row in automatic_members) < 0.92:
        errors.append("automatic semantic duplicate member is below the 0.92 threshold")
    check_count("semantic_deduplication", "candidate_groups_total", len(inputs["semantic_state"]))
    semantic_state_keys = {str(row.get("group_key") or "") for row in inputs["semantic_state"]}
    if "" in semantic_state_keys or len(semantic_state_keys) != len(inputs["semantic_state"]):
        errors.append("semantic state contains missing or duplicate group keys")
    if any(row.get("status") not in {"processed", "skipped_small"} for row in inputs["semantic_state"]):
        errors.append("semantic state contains an incomplete or blocked group")

    semantic_stage = (state.get("stages") or {}).get("semantic_deduplication") or {}
    if semantic_stage.get("embedding_model") != "NeuML/pubmedbert-base-embeddings" or semantic_stage.get("embedding_dimension") != 768:
        errors.append("semantic stage model or dimension differs from the strategy")
    for stage_name, stage in (state.get("stages") or {}).items():
        summary_ref = str((stage or {}).get("summary_path") or "")
        if not summary_ref:
            continue
        if Path(summary_ref).is_absolute():
            warnings.append(f"{stage_name} summary path should be run-relative")
        elif not (output_dir / summary_ref).exists():
            errors.append(f"{stage_name} summary path does not resolve inside the run")

    return {
        "generated_at": utc_now(),
        "accepted": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_count": len(candidates),
        "candidate_type_counts": dict(sorted(Counter(str(row.get("evidence_type") or "") for row in candidates).items())),
        "provenance_passed_count": sum(row.get("provenance_status") == "passed" for row in inputs["provenance"]),
        "compound_identity_count": len(inputs["identities"]),
        "attribution_resolved_count": sum(row.get("attribution_status") == "resolved" for row in inputs["attributions"]),
        "domain_status_counts": dict(sorted(domain_statuses.items())),
        "deterministic_representative_count": len(representative_ids),
        "semantic_input_count": len(semantic_input_ids),
        "semantic_group_type_counts": dict(sorted(Counter(str(row.get("group_type") or "") for row in inputs["semantic_groups"]).items())),
    }


def analyze_conflicts(
    inputs: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    representative_by_id = {str(row["evidence_id"]): row for row in inputs["deterministic_representatives"]}
    semantic_duplicate_nonrepresentatives = {
        str(row["evidence_id"])
        for row in inputs["semantic_members"]
        if row.get("group_type") == "semantic_duplicate" and row.get("is_representative") is False
    }
    post_semantic = [
        row
        for row in inputs["deterministic_representatives"]
        if row.get("domain_gate_status") == "passed" and str(row["evidence_id"]) not in semantic_duplicate_nonrepresentatives
    ]

    report: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    assigned: dict[str, str] = {}
    semantic_member_ids: set[str] = set()
    for source_group in sorted(inputs["semantic_conflicts"], key=lambda row: str(row.get("duplicate_group_id") or "")):
        source_group_id = str(source_group["duplicate_group_id"])
        evidence_ids = [str(value) for value in source_group.get("evidence_ids") or []]
        rows = [representative_by_id[value] for value in evidence_ids if value in representative_by_id]
        if len(rows) != len(evidence_ids):
            raise RuntimeError(f"Semantic conflict group has unresolved members: {source_group_id}")
        analysis = analyze_conflict_rows(rows)
        conflict_group_id = stable_uuid("semantic", source_group_id)
        record = build_conflict_record(
            conflict_group_id=conflict_group_id,
            source_group_type="semantic_conflict_candidate",
            source_group_id=source_group_id,
            rows=rows,
            analysis=analysis,
        )
        report.append(record)
        for row in rows:
            evidence_id = str(row["evidence_id"])
            if evidence_id in assigned:
                raise RuntimeError(f"Evidence belongs to more than one primary conflict group: {evidence_id}")
            assigned[evidence_id] = conflict_group_id
            semantic_member_ids.add(evidence_id)
            members.append(build_conflict_member(record, row))

    strict_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in post_semantic:
        evidence_id = str(row["evidence_id"])
        if evidence_id in assigned or not has_comparable_fact(row):
            continue
        key = strict_conflict_key(row)
        if key:
            strict_buckets[key].append(row)
    for key, rows in sorted(strict_buckets.items()):
        if len(rows) < 2:
            continue
        analysis = analyze_conflict_rows(rows)
        if analysis["status"] == "compatible":
            continue
        conflict_group_id = stable_uuid("strict", key, *(str(row["evidence_id"]) for row in rows))
        record = build_conflict_record(
            conflict_group_id=conflict_group_id,
            source_group_type="strict_conflict_key",
            source_group_id=None,
            rows=rows,
            analysis=analysis,
        )
        report.append(record)
        for row in rows:
            evidence_id = str(row["evidence_id"])
            if evidence_id in assigned:
                raise RuntimeError(f"Evidence belongs to more than one primary conflict group: {evidence_id}")
            assigned[evidence_id] = conflict_group_id
            members.append(build_conflict_member(record, row))

    report.sort(key=lambda row: (row["compound_name"], row["evidence_type"], str(row.get("parameter") or ""), row["conflict_group_id"]))
    members.sort(key=lambda row: (row["conflict_group_id"], row["evidence_id"]))
    status_counts = Counter(str(row["analysis_status"]) for row in report)
    blocking_groups = [row for row in report if row["requires_human_review"]]
    blocking_evidence_ids = {value for row in blocking_groups for value in row["evidence_ids"]}
    semantic_expected = {value for row in inputs["semantic_conflicts"] for value in row.get("evidence_ids") or []}
    accepted = semantic_member_ids == semantic_expected and len(assigned) == len(members)
    summary = {
        "generated_at": utc_now(),
        "accepted": accepted,
        "post_semantic_representative_count": len(post_semantic),
        "semantic_conflict_input_group_count": len(inputs["semantic_conflicts"]),
        "semantic_conflict_input_member_count": len(semantic_expected),
        "report_count": len(report),
        "member_count": len(members),
        "status_counts": dict(sorted(status_counts.items())),
        "blocking_group_count": len(blocking_groups),
        "blocking_evidence_count": len(blocking_evidence_ids),
        "semantic_members_resolved_once": semantic_member_ids == semantic_expected,
        "report_path": "04_conflict_analysis/evidence_conflict_report.jsonl",
        "members_path": "04_conflict_analysis/conflict_members.jsonl",
    }
    return report, members, summary


def analyze_conflict_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    explicit_context_differences: list[str] = []
    partial_context: list[str] = []
    uncertain_context: list[str] = []
    context_values: dict[str, list[str]] = {}
    condition_category_values: list[str] = []
    for field in CONTEXT_FIELDS:
        values = [normalize_text(context_value(row, field)) for row in rows]
        nonempty = {value for value in values if value}
        context_values[field] = sorted(nonempty)
        if field == "condition" and len(nonempty) > 1:
            categories = [canonical_condition_signature(value) for value in values]
            condition_category_values = sorted({value for value in categories if value})
            uncertain_context.append(field)
        elif len(nonempty) > 1 and all(values):
            explicit_context_differences.append(field)
        elif nonempty and any(not value for value in values):
            partial_context.append(field)

    fact = compare_facts(rows)
    if explicit_context_differences:
        status = "explainable_variation"
        reason = "explicit_context_difference"
    elif uncertain_context:
        status = "insufficient_context"
        reason = "free_text_condition_difference_not_canonicalized"
    elif partial_context and fact["status"] in {"different", "ambiguous", "incomplete", "none"}:
        status = "insufficient_context"
        reason = "context_missing_for_comparison"
    elif fact["status"] == "different":
        status = "unresolved_conflict"
        reason = "same_or_missing_context_with_disagreement"
    elif fact["status"] in {"ambiguous", "incomplete", "none"}:
        status = "insufficient_context"
        reason = "facts_not_safely_comparable"
    else:
        status = "compatible"
        reason = "facts_compatible_within_tolerance"
    return {
        "status": status,
        "reason": reason,
        "requires_human_review": status in BLOCKING_CONFLICT_STATUSES,
        "explicit_context_differences": explicit_context_differences,
        "partial_context_fields": partial_context,
        "uncertain_context_fields": uncertain_context,
        "context_values": context_values,
        "condition_category_values": condition_category_values,
        "fact_comparison": fact,
    }


def compare_facts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [numeric_fact(row) for row in rows]
    numeric_present = [value for value in numeric if value is not None]
    direction = [direction_fact(row) for row in rows]
    direction_present = [value for value in direction if value is not None]

    numeric_status = "none"
    if numeric_present:
        if len(numeric_present) != len(rows):
            numeric_status = "incomplete"
        elif len({value[0] for value in numeric_present}) > 1:
            numeric_status = "ambiguous"
        elif all(intervals_compatible(left, right) for position, left in enumerate(numeric_present) for right in numeric_present[position + 1 :]):
            numeric_status = "same"
        else:
            numeric_status = "different"

    direction_status = "none"
    if any(value == "mixed" for value in direction_present):
        direction_status = "ambiguous"
    elif direction_present:
        if len(direction_present) != len(rows):
            direction_status = "incomplete"
        elif len(set(direction_present)) > 1:
            direction_status = "different"
        else:
            direction_status = "same"

    statuses = {numeric_status, direction_status}
    if "different" in statuses:
        status = "different"
    elif "ambiguous" in statuses:
        status = "ambiguous"
    elif "incomplete" in statuses:
        status = "incomplete"
    elif "same" in statuses:
        status = "same"
    else:
        status = "none"
    return {
        "status": status,
        "numeric_status": numeric_status,
        "direction_status": direction_status,
        "numeric_facts": [numeric_fact_record(value) for value in numeric],
        "direction_facts": direction,
    }


def numeric_fact(row: dict[str, Any]) -> tuple[str, float, float] | None:
    canonical = row.get("canonical_value") is not None or row.get("canonical_value_low") is not None
    unit = row.get("canonical_unit") if canonical else row.get("unit")
    low = row.get("canonical_value_low") if canonical else row.get("value_low")
    high = row.get("canonical_value_high") if canonical else row.get("value_high")
    value = row.get("canonical_value") if canonical else row.get("value")
    if low is None and value is not None:
        low = value
    if high is None and value is not None:
        high = value
    if low is None or high is None:
        parsed = parse_value_text(row.get("value_text"))
        if parsed is None:
            return None
        low, high = parsed
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError):
        return None
    if low_value > high_value:
        low_value, high_value = high_value, low_value
    return normalize_text(unit), low_value, high_value


def numeric_fact_record(value: tuple[str, float, float] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"unit": value[0], "low": value[1], "high": value[2]}


def parse_value_text(value: Any) -> tuple[float, float] | None:
    text = normalize_text(value)
    if not text or len(text) > 80:
        return None
    range_match = re.fullmatch(r"[~≈<>≤≥= ]*(-?\d+(?:\.\d+)?)\s*(?:to|[-–])\s*(-?\d+(?:\.\d+)?).*", text)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    number_match = re.fullmatch(r"[~≈<>≤≥= ]*(-?\d+(?:\.\d+)?).*", text)
    if number_match:
        number = float(number_match.group(1))
        return number, number
    return None


def intervals_compatible(left: tuple[str, float, float], right: tuple[str, float, float]) -> bool:
    _, left_low, left_high = left
    _, right_low, right_high = right
    if max(left_low, right_low) <= min(left_high, right_high):
        return True
    left_mid = (left_low + left_high) / 2.0
    right_mid = (right_low + right_high) / 2.0
    return math.isclose(left_mid, right_mid, rel_tol=0.01, abs_tol=1e-9)


def direction_fact(row: dict[str, Any]) -> str | None:
    text = normalize_text(row.get("source_text"))
    matches = {
        direction
        for direction, patterns in DIRECTION_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }
    if not matches:
        return None
    if len(matches) > 1:
        return "mixed"
    return next(iter(matches))


def context_value(row: dict[str, Any], field: str) -> Any:
    if row.get(field) not in (None, ""):
        return row[field]
    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
    aliases = {
        "time_point": ("time_point", "timepoint", "sampling_time", "time"),
        "study_design": ("study_design", "design"),
        "label_version": ("label_version", "effective_time", "version", "spl_version"),
    }.get(field, (field,))
    return recursive_metadata_value(metadata, aliases)


def canonical_condition_signature(value: Any) -> str:
    text = normalize_text(value)
    categories = {
        category
        for category, patterns in CONTROLLED_CONDITION_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }
    return "|".join(sorted(categories))


def recursive_metadata_value(value: Any, aliases: Iterable[str]) -> Any:
    alias_set = {normalize_key(alias) for alias in aliases}
    if isinstance(value, dict):
        for key, item in value.items():
            if normalize_key(key) in alias_set and item not in (None, "", [], {}):
                return item
        for item in value.values():
            result = recursive_metadata_value(item, aliases)
            if result not in (None, ""):
                return result
    elif isinstance(value, list):
        for item in value:
            result = recursive_metadata_value(item, aliases)
            if result not in (None, ""):
                return result
    return None


def strict_conflict_key(row: dict[str, Any]) -> str | None:
    compound = row.get("attributed_compound_id") or row.get("compound_id")
    parameter = normalize_key(row.get("parameter"))
    if not compound or not parameter:
        return None
    canonical_unit = row.get("canonical_unit") or row.get("unit")
    values = [
        compound,
        normalize_key(row.get("evidence_type")),
        parameter,
        normalize_key(canonical_unit),
        normalize_text(row.get("route")),
        normalize_text(row.get("population")),
        normalize_text(row.get("condition")),
    ]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def has_comparable_fact(row: dict[str, Any]) -> bool:
    return numeric_fact(row) is not None or direction_fact(row) is not None


def build_conflict_record(
    *,
    conflict_group_id: str,
    source_group_type: str,
    source_group_id: str | None,
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    first = rows[0]
    requires_human_review = bool(analysis["requires_human_review"])
    if source_group_type == "semantic_conflict_candidate" and analysis["status"] == "compatible":
        requires_human_review = True
    return {
        "conflict_group_id": conflict_group_id,
        "source_group_type": source_group_type,
        "source_group_id": source_group_id,
        "analysis_status": analysis["status"],
        "analysis_reason": analysis["reason"],
        "requires_human_review": requires_human_review,
        "compound_id": first.get("attributed_compound_id") or first.get("compound_id"),
        "compound_name": first.get("attributed_compound_name") or first.get("compound_name"),
        "evidence_type": first.get("evidence_type"),
        "parameter": first.get("parameter"),
        "canonical_unit": first.get("canonical_unit") or first.get("unit"),
        "evidence_count": len(rows),
        "evidence_ids": sorted(str(row["evidence_id"]) for row in rows),
        "source_ids": sorted({str(row.get("source_id") or "") for row in rows}),
        "explicit_context_differences": analysis["explicit_context_differences"],
        "partial_context_fields": analysis["partial_context_fields"],
        "uncertain_context_fields": analysis["uncertain_context_fields"],
        "context_values": analysis["context_values"],
        "condition_category_values": analysis["condition_category_values"],
        "fact_comparison": analysis["fact_comparison"],
        "resolution": conflict_resolution(analysis["status"], source_group_type),
        "analyzed_at": utc_now(),
    }


def build_conflict_member(group: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "conflict_group_id": group["conflict_group_id"],
        "evidence_id": row["evidence_id"],
        "analysis_status": group["analysis_status"],
        "requires_human_review": group["requires_human_review"],
        "source_group_type": group["source_group_type"],
        "source_group_id": group.get("source_group_id"),
        "source_id": row.get("source_id"),
        "source_location": row.get("source_location"),
    }


def conflict_resolution(status: str, source_group_type: str) -> str:
    if status == "explainable_variation":
        return "keep_as_separate_context_specific_evidence"
    if status == "compatible":
        if source_group_type == "semantic_conflict_candidate":
            return "defer_to_human_duplicate_review"
        return "keep_without_conflict_blocker"
    return "defer_to_human_review_without_averaging_or_truth_selection"


def build_evidence_layers(
    inputs: dict[str, list[dict[str, Any]]],
    conflict_report: list[dict[str, Any]],
    conflict_members: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidate_by_id = {str(row["evidence_id"]): row for row in inputs["candidates"]}
    provenance_by_id = {str(row["evidence_id"]): row for row in inputs["provenance"]}
    attribution_by_id = {str(row["evidence_id"]): row for row in inputs["attributions"]}
    domain_by_id = {str(row["evidence_id"]): row for row in inputs["domain_results"]}
    deterministic_nonrepresentatives = group_membership(
        inputs["deterministic_members"],
        predicate=lambda row: row.get("is_representative") is False,
    )
    semantic_nonrepresentatives = group_membership(
        inputs["semantic_members"],
        predicate=lambda row: row.get("group_type") == "semantic_duplicate" and row.get("is_representative") is False,
    )
    semantic_groups = group_membership(inputs["semantic_members"])
    similar_for_review = group_membership(
        inputs["semantic_members"],
        predicate=lambda row: row.get("group_type") == "similar_for_review",
    )
    conflict_membership: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in conflict_members:
        conflict_membership[str(member["evidence_id"])].append(member)

    preselected: list[dict[str, Any]] = []
    preselected_by_id: dict[str, dict[str, Any]] = {}
    for source_row in inputs["deterministic_representatives"]:
        evidence_id = str(source_row["evidence_id"])
        domain = domain_by_id[evidence_id]
        if domain.get("domain_gate_status") != "passed":
            continue
        if source_row.get("compound_role") not in FINAL_ROLES:
            continue
        if evidence_id in semantic_nonrepresentatives:
            continue
        blockers: list[str] = []
        blockers.extend(f"semantic_similarity_review:{group_id}" for group_id in similar_for_review.get(evidence_id, []))
        for member in conflict_membership.get(evidence_id, []):
            if member.get("requires_human_review"):
                blockers.append(f"conflict_{member['analysis_status']}:{member['conflict_group_id']}")
        row = dict(source_row)
        row.update(
            {
                "evidence_layer": "preselected",
                "machine_extracted": True,
                "review_blockers": sorted(set(blockers)),
                "semantic_group_ids": semantic_groups.get(evidence_id, []),
                "conflict_group_ids": [member["conflict_group_id"] for member in conflict_membership.get(evidence_id, [])],
                "conflict_statuses": sorted({str(member["analysis_status"]) for member in conflict_membership.get(evidence_id, [])}),
            }
        )
        preselected.append(row)
        preselected_by_id[evidence_id] = row

    preselected.sort(key=evidence_sort_key)
    reviewable: list[dict[str, Any]] = []
    for row in preselected:
        if row["review_blockers"]:
            continue
        reviewable_row = dict(row)
        reviewable_row["evidence_layer"] = "reviewable"
        reviewable.append(reviewable_row)
    reviewable_ids = {str(row["evidence_id"]) for row in reviewable}

    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for evidence_id, candidate in candidate_by_id.items():
        if evidence_id in reviewable_ids:
            continue
        reasons: list[str] = []
        provenance = provenance_by_id.get(evidence_id)
        attribution = attribution_by_id[evidence_id]
        domain = domain_by_id[evidence_id]
        if provenance is None or provenance.get("provenance_status") != "passed":
            reasons.append("provenance_not_passed")
        if attribution.get("attribution_status") != "resolved":
            reasons.append("compound_attribution_unresolved")
        if attribution.get("compound_role") not in FINAL_ROLES:
            reasons.append(f"compound_role_not_final:{attribution.get('compound_role')}")
        domain_status = str(domain.get("domain_gate_status") or "")
        for reason in domain.get("reasons") or []:
            reasons.append(f"domain_{domain_status}:{reason}")
        if domain_status == "needs_manual_review" and not domain.get("reasons"):
            reasons.append("domain_needs_manual_review")
        if domain_status == "excluded" and not domain.get("reasons"):
            reasons.append("domain_excluded")
        reasons.extend(f"deterministic_duplicate_member:{group_id}" for group_id in deterministic_nonrepresentatives.get(evidence_id, []))
        reasons.extend(f"semantic_duplicate_member:{group_id}" for group_id in semantic_nonrepresentatives.get(evidence_id, []))
        if evidence_id in preselected_by_id:
            reasons.extend(preselected_by_id[evidence_id]["review_blockers"])
        if not reasons:
            reasons.append("not_preselected")
        reasons = sorted(set(reasons))
        reason_counts.update(reason.split(":", 1)[0] for reason in reasons)
        row = enrich_candidate(candidate, attribution, domain)
        row.update(
            {
                "evidence_layer": "excluded",
                "exclusion_reasons": reasons,
                "preselected": evidence_id in preselected_by_id,
                "semantic_group_ids": semantic_groups.get(evidence_id, []),
                "conflict_group_ids": [member["conflict_group_id"] for member in conflict_membership.get(evidence_id, [])],
            }
        )
        excluded.append(row)
    excluded.sort(key=evidence_sort_key)

    accepted: list[dict[str, Any]] = []
    modeling: list[dict[str, Any]] = []
    accepted_checks = {
        "candidate_partition_complete": len(reviewable) + len(excluded) == len(candidate_by_id),
        "candidate_partition_disjoint": not (reviewable_ids & {str(row["evidence_id"]) for row in excluded}),
        "preselected_unique": len(preselected_by_id) == len(preselected),
        "reviewable_has_provenance": all(row.get("source_text") and row.get("source_location") for row in reviewable),
        "reviewable_needs_review": all(row.get("curation_status") == "needs_review" for row in reviewable),
        "semantic_nonrepresentatives_excluded": not (set(semantic_nonrepresentatives) & set(preselected_by_id)),
        "blocked_rows_excluded": all(not row.get("review_blockers") for row in reviewable),
        "human_layers_empty_without_events": not accepted and not modeling,
    }
    summary = {
        "generated_at": utc_now(),
        "accepted": all(accepted_checks.values()),
        "candidate_count": len(candidate_by_id),
        "preselected_count": len(preselected),
        "reviewable_count": len(reviewable),
        "excluded_from_reviewable_count": len(excluded),
        "accepted_count": 0,
        "modeling_count": 0,
        "reviewable_type_counts": dict(sorted(Counter(str(row.get("evidence_type") or "") for row in reviewable).items())),
        "reviewable_compound_counts": dict(sorted(Counter(str(row.get("compound_name") or row.get("attributed_compound_name") or "") for row in reviewable).items())),
        "exclusion_reason_category_counts": dict(sorted(reason_counts.items())),
        "acceptance_checks": accepted_checks,
        "human_review_status": "not_started",
        "table_review_queue_promoted_count": 0,
    }
    outputs = {
        "preselected_evidence.jsonl": preselected,
        "reviewable_evidence.jsonl": reviewable,
        "excluded_from_reviewable.jsonl": excluded,
        "accepted_evidence.jsonl": accepted,
        "modeling_evidence.jsonl": modeling,
    }
    return outputs, summary


def enrich_candidate(candidate: dict[str, Any], attribution: dict[str, Any], domain: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row.update(
        {
            "attributed_compound_id": attribution.get("compound_id"),
            "attributed_compound_name": attribution.get("compound_name"),
            "compound_id": attribution.get("compound_id"),
            "compound_name": attribution.get("compound_name"),
            "compound_role": attribution.get("compound_role"),
            "attribution_method": attribution.get("attribution_method"),
            "attribution_stage": attribution.get("attribution_stage"),
            "attribution_confidence": attribution.get("attribution_confidence"),
            "attribution_reason": attribution.get("attribution_reason"),
            "matched_text": attribution.get("matched_text"),
            "provenance_status": "passed",
            "domain_gate_status": domain.get("domain_gate_status"),
            "domain_classification": domain.get("domain_classification"),
            "domain_gate_reasons": domain.get("reasons") or [],
        }
    )
    return row


def group_membership(
    rows: list[dict[str, Any]],
    predicate: Any | None = None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        result[str(row["evidence_id"])].append(str(row["duplicate_group_id"]))
    return {key: sorted(set(value)) for key, value in result.items()}


def unique_map(rows: list[dict[str, Any]], field: str, label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field) or "")
        if not key:
            errors.append(f"{label} row lacks {field}")
        elif key in result:
            errors.append(f"duplicate {label} {field}: {key}")
        else:
            result[key] = row
    return result


def stable_uuid(*parts: str) -> str:
    payload = "|".join(parts)
    return str(uuid5(NAMESPACE_URL, f"pbpk:final-evidence-conflict:{payload}"))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def evidence_sort_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("compound_name") or row.get("attributed_compound_name") or ""),
        str(row.get("evidence_type") or ""),
        str(row.get("parameter") or ""),
        str(row.get("source_id") or ""),
        str(row.get("evidence_id") or ""),
    )


def normalize_stage_summary_paths(state: dict[str, Any], output_dir: Path) -> None:
    for stage in (state.get("stages") or {}).values():
        if not isinstance(stage, dict) or not stage.get("summary_path"):
            continue
        path = Path(str(stage["summary_path"]))
        if not path.is_absolute():
            continue
        try:
            stage["summary_path"] = path.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            continue


if __name__ == "__main__":
    main()
