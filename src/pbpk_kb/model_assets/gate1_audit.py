from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.model_assets.models import (
    OSPExecutionPlan,
    OSPExecutionValidationReport,
    OSPModelVersionBuildSummary,
    OSPModelVersionSnapshot,
    OSPPKMLProcessRecord,
    OSPPKMLProjectParseSummary,
    RepositoryInventorySummary,
)


AUDITOR_VERSION = "pbpk-model-assets-gate1-auditor-1.2.0"
REQUIRED_PKML_PARSER_VERSION = "1.2.0"


class Gate1AuditError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate1AuditError(f"Expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Gate1AuditError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            yield value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _process_participant_errors(
    record: OSPPKMLProcessRecord,
) -> list[str]:
    errors: list[str] = []
    if record.parser_version != REQUIRED_PKML_PARSER_VERSION:
        errors.append("parser_version")
    unresolved: set[str] = set()
    for role, identifiers, names, participants in (
        (
            "educt",
            record.educt_identifiers,
            record.educts,
            record.educt_participants,
        ),
        (
            "product",
            record.product_identifiers,
            record.products,
            record.product_participants,
        ),
        (
            "modifier",
            record.modifier_identifiers,
            record.modifiers,
            record.modifier_participants,
        ),
    ):
        participant_identifiers = {
            participant.identifier for participant in participants
        }
        if set(identifiers) != participant_identifiers:
            errors.append(f"{role}_identifier_coverage")
        resolved_names: set[str] = set()
        for participant in participants:
            if participant.resolution_status == "unresolved":
                unresolved.add(participant.identifier)
                if (
                    participant.name is not None
                    or participant.component_type is not None
                ):
                    errors.append(f"{role}_unresolved_has_identity")
                continue
            if not participant.name or not participant.component_type:
                errors.append(f"{role}_resolved_identity_incomplete")
                continue
            resolved_names.add(participant.name)
            if (
                participant.resolution_status
                == "resolved_by_component_id"
                and participant.name == participant.identifier
            ):
                errors.append(
                    f"{role}_component_id_used_as_name"
                )
            elif (
                participant.name == participant.identifier
                and participant.resolution_status
                != "resolved_by_component_name"
            ):
                errors.append(f"{role}_invalid_identity_resolution")
        if set(names) != resolved_names:
            errors.append(f"{role}_resolved_name_projection")
    if set(record.unresolved_partner_identifiers) != unresolved:
        errors.append("unresolved_identifier_projection")
    return errors


def audit_gate1_model_assets(
    *,
    plan_path: Path,
    inventory_directory: Path,
    execution_validation_path: Path,
    model_version_directory: Path,
) -> dict[str, Any]:
    plan = OSPExecutionPlan.model_validate(_read_json(plan_path))
    inventory = RepositoryInventorySummary.model_validate(
        _read_json(inventory_directory / "summary.json")
    )
    execution_validation = OSPExecutionValidationReport.model_validate(
        _read_json(execution_validation_path)
    )
    build_summary = OSPModelVersionBuildSummary.model_validate(
        _read_json(model_version_directory / "summary.json")
    )

    expected_project_count = len(plan.projects)
    expected_simulation_count = sum(
        len(project.simulations) for project in plan.projects
    )
    plan_project_by_key = {
        project.project_key: project for project in plan.projects
    }

    checksum_mismatches: list[str] = []
    for filename, expected_sha256 in sorted(
        build_summary.output_checksums.items()
    ):
        output_path = model_version_directory / filename
        if not output_path.is_file():
            checksum_mismatches.append(f"{filename}:missing")
            continue
        actual_sha256 = _sha256_file(output_path)
        if actual_sha256 != expected_sha256:
            checksum_mismatches.append(
                f"{filename}:{expected_sha256}:{actual_sha256}"
            )

    line_counts: dict[str, int] = {}
    process_participant_errors: list[str] = []
    unresolved_process_partner_count = 0
    for filename in (
        "reports.jsonl.gz",
        "simulations.jsonl",
        "parameters.jsonl.gz",
        "components.jsonl.gz",
        "processes.jsonl.gz",
        "administrations.jsonl",
        "project_summaries.jsonl",
        "model_versions.jsonl",
        "failures.jsonl",
    ):
        count = 0
        for row in _iter_jsonl(model_version_directory / filename):
            count += 1
            if filename != "processes.jsonl.gz":
                continue
            process = OSPPKMLProcessRecord.model_validate(row)
            errors = _process_participant_errors(process)
            if errors and len(process_participant_errors) < 100:
                process_participant_errors.append(
                    f"{process.project_key}:{process.simulation_id}:"
                    f"{process.process_id}:{','.join(errors)}"
                )
            unresolved_process_partner_count += len(
                process.unresolved_partner_identifiers
            )
        line_counts[filename] = count

    project_summaries = [
        OSPPKMLProjectParseSummary.model_validate(row)
        for row in _iter_jsonl(
            model_version_directory / "project_summaries.jsonl"
        )
    ]
    model_versions = [
        OSPModelVersionSnapshot.model_validate(row)
        for row in _iter_jsonl(
            model_version_directory / "model_versions.jsonl"
        )
    ]

    project_summary_errors: list[str] = []
    for summary in project_summaries:
        planned = plan_project_by_key.get(summary.project_key)
        if planned is None:
            project_summary_errors.append(
                f"{summary.project_key}:not_in_execution_plan"
            )
            continue
        expected_count = len(planned.simulations)
        if (
            summary.project_relative_path != planned.relative_path
            or summary.project_sha256 != planned.project_sha256
            or summary.simulation_count != expected_count
            or summary.parsed_simulation_count != expected_count
            or summary.failed_simulation_count != 0
            or summary.parse_status.value != "parsed"
            or summary.reason_codes
        ):
            project_summary_errors.append(
                f"{summary.project_key}:coverage_or_parse_mismatch"
            )

    model_version_errors: list[str] = []
    model_family_by_id: dict[str, str] = {}
    model_id_family_errors: list[str] = []
    model_version_ids = [
        model.model_version_id for model in model_versions
    ]
    model_version_project_keys = [
        model.project_key for model in model_versions
    ]
    for model in model_versions:
        existing_family = model_family_by_id.setdefault(
            model.model_id,
            model.model_family,
        )
        if existing_family != model.model_family:
            model_id_family_errors.append(
                f"{model.model_id}:{existing_family}:{model.model_family}"
            )
        planned = plan_project_by_key.get(model.project_key)
        if planned is None:
            model_version_errors.append(
                f"{model.model_version_id}:project_not_in_plan"
            )
            continue
        if (
            model.project_relative_path != planned.relative_path
            or model.project_sha256 != planned.project_sha256
            or model.repository_commit != plan.repository_commit
            or model.source_id != plan.source_id
            or model.execution_probe_status.value != "passed"
            or model.pkml_parse_status.value != "parsed"
            or model.simulation_count != len(planned.simulations)
        ):
            model_version_errors.append(
                f"{model.model_version_id}:identity_or_status_mismatch"
            )
        if not model.assets or any(
            not asset.source_location
            or len(asset.sha256) != 64
            or not asset.relative_path
            for asset in model.assets
        ):
            model_version_errors.append(
                f"{model.model_version_id}:asset_provenance_incomplete"
            )
        if model.reusable_candidate:
            model_version_errors.append(
                f"{model.model_version_id}:gate1_cannot_assign_scientific_reuse"
            )

    checks = {
        "inventory_static_acceptance": inventory.gate1_static_acceptance,
        "inventory_project_coverage": (
            inventory.model_project_count == expected_project_count
            and inventory.parsed_project_count == expected_project_count
            and inventory.failed_project_count == 0
        ),
        "inventory_asset_classification_coverage": (
            sum(inventory.asset_kind_counts.values())
            == inventory.asset_count
            and sum(inventory.parse_status_counts.values())
            == inventory.asset_count
        ),
        "repository_commit_consistent": (
            inventory.repository_commit
            == plan.repository_commit
            == build_summary.repository_commit
        ),
        "execution_validation_complete": (
            execution_validation.execution_probe_complete
            and execution_validation.project_count
            == expected_project_count
            and execution_validation.passed_project_count
            == expected_project_count
            and execution_validation.expected_simulation_count
            == expected_simulation_count
            and not execution_validation.reason_codes
        ),
        "model_version_build_accepted": (
            build_summary.gate1_model_version_acceptance
            and build_summary.project_count == expected_project_count
            and build_summary.model_version_count
            == expected_project_count
            and build_summary.simulation_count
            == expected_simulation_count
            and build_summary.parsed_simulation_count
            == expected_simulation_count
            and build_summary.failed_simulation_count == 0
        ),
        "output_checksums_match": not checksum_mismatches,
        "record_counts_match": (
            line_counts["parameters.jsonl.gz"]
            == build_summary.parameter_record_count
            and line_counts["components.jsonl.gz"]
            == build_summary.component_record_count
            and line_counts["processes.jsonl.gz"]
            == build_summary.process_record_count
            and line_counts["simulations.jsonl"]
            == expected_simulation_count
            and line_counts["project_summaries.jsonl"]
            == expected_project_count
            and line_counts["model_versions.jsonl"]
            == expected_project_count
            and line_counts["failures.jsonl"] == 0
        ),
        "project_summaries_valid": not project_summary_errors,
        "model_versions_valid": not model_version_errors,
        "model_id_family_consistent": not model_id_family_errors,
        "model_version_ids_unique": (
            len(model_version_ids) == len(set(model_version_ids))
        ),
        "one_model_version_per_project": (
            len(model_version_project_keys)
            == len(set(model_version_project_keys))
            == expected_project_count
        ),
        "process_participant_resolution_valid": (
            not process_participant_errors
        ),
    }
    accepted = all(checks.values())
    return {
        "audit_schema_version": "1.0.0",
        "auditor_version": AUDITOR_VERSION,
        "audited_at": datetime.now(UTC).isoformat(),
        "accepted": accepted,
        "checks": checks,
        "counts": {
            "projects": expected_project_count,
            "simulations": expected_simulation_count,
            "models": len(model_family_by_id),
            "model_versions": len(model_versions),
            "reusable_candidates": build_summary.reusable_candidate_count,
            "unresolved_process_partners": (
                unresolved_process_partner_count
            ),
            **line_counts,
        },
        "policy_blockers": build_summary.blocking_reasons,
        "errors": {
            "checksum_mismatches": checksum_mismatches,
            "project_summaries": project_summary_errors,
            "model_versions": model_version_errors,
            "model_id_families": model_id_family_errors,
            "process_participants": process_participant_errors,
        },
        "input_checksums": {
            "plan": _sha256_file(plan_path),
            "inventory_summary": _sha256_file(
                inventory_directory / "summary.json"
            ),
            "execution_validation": _sha256_file(
                execution_validation_path
            ),
            "model_version_summary": _sha256_file(
                model_version_directory / "summary.json"
            ),
        },
    }


def write_gate1_audit(path: Path, report: dict[str, Any]) -> str:
    return _write_json_atomic(path, report)
