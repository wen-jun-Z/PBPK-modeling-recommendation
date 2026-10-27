from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from pbpk_kb.model_assets.execution_plan import write_json_atomic
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    LicensedSnapshotReuseProtocol,
)
from pbpk_kb.model_assets.models import (
    ModelValidationRunLabel,
    OSPExecutionPlan,
    OSPModelVersionSnapshot,
)
from pbpk_kb.model_assets.validation_scope import build_model_validation_scope


ADAPTER_VERSION = "licensed-model-kg-input-adapter-1.0.0"


class LicensedModelKGInputError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise LicensedModelKGInputError(f"Expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise LicensedModelKGInputError(
                    f"Expected an object at {path}:{line_number}"
                )
            yield value


def _write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="\n",
            ) as text:
                for row in rows:
                    text.write(_canonical_json(row))
                    text.write("\n")
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)
    return sha256_file(path)


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _single_row(path: Path) -> dict[str, Any]:
    rows = list(_iter_jsonl(path))
    if len(rows) != 1:
        raise LicensedModelKGInputError(
            f"Expected one row in {path}; found {len(rows)}."
        )
    return rows[0]


def _stable_execution_job_id(protocol_id: str) -> str:
    digest = hashlib.sha256(
        f"licensed-model-kg\x1f{protocol_id}".encode("utf-8")
    ).hexdigest()
    return f"osp-exec-{digest[:32]}"


def prepare_licensed_model_kg_input(
    *,
    registration_directory: Path,
    protocol_path: Path,
    validation_audit_path: Path,
    runtime_plan_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    registration_directory = registration_directory.resolve()
    protocol_path = protocol_path.resolve()
    validation_audit_path = validation_audit_path.resolve()
    runtime_plan_path = runtime_plan_path.resolve()
    output_directory = output_directory.resolve()
    working = output_directory.with_name(output_directory.name + ".tmp")
    if output_directory.exists() or working.exists():
        raise LicensedModelKGInputError(
            f"Output or temporary directory already exists: {output_directory}"
        )

    registration_summary_path = (
        registration_directory / "registration_summary.json"
    )
    registration_summary = _read_json(registration_summary_path)
    protocol = LicensedSnapshotReuseProtocol.model_validate_json(
        protocol_path.read_text(encoding="utf-8")
    )
    validation_audit = _read_json(validation_audit_path)
    runtime_plan = OSPExecutionPlan.model_validate_json(
        runtime_plan_path.read_text(encoding="utf-8")
    )

    required_fact_names = (
        "model_versions",
        "simulations",
        "parameters",
        "components",
        "processes",
        "administrations",
        "validation_run_labels",
    )
    fact_paths = {
        name: registration_directory / f"{name}.jsonl.gz"
        for name in required_fact_names
    }
    checksum_errors = []
    for name, path in fact_paths.items():
        expected = (registration_summary.get("output_checksums") or {}).get(
            path.name
        )
        if not path.is_file():
            checksum_errors.append(f"{path.name}:missing")
        elif sha256_file(path) != expected:
            checksum_errors.append(f"{path.name}:checksum_mismatch")
    if checksum_errors:
        raise LicensedModelKGInputError(
            "Registration fact-layer integrity failed: "
            + ", ".join(checksum_errors)
        )

    model_payload = _single_row(fact_paths["model_versions"])
    if model_payload.get("validation_scope") is None:
        metrics_by_sequence = {
            int(row["simulation_sequence"]): row
            for row in _iter_jsonl(
                validation_audit_path.parent
                / "verification_dataset_metrics.jsonl"
            )
        }
        scope = build_model_validation_scope(
            {
                **row,
                "validation_metrics": metrics_by_sequence.get(
                    int(row["protocol_sequence"]), {}
                ),
            }
            for row in _iter_jsonl(fact_paths["simulations"])
        )
        model_payload["validation_scope"] = (
            scope.model_dump(mode="json") if scope is not None else None
        )
    model = OSPModelVersionSnapshot.model_validate(model_payload)
    label = ModelValidationRunLabel.model_validate(
        _single_row(fact_paths["validation_run_labels"])
    )
    if model.validation_run_label != label:
        raise LicensedModelKGInputError(
            "ModelVersion and validation label files disagree."
        )
    if (
        registration_summary.get("accepted") is not True
        or registration_summary.get("model_id") != model.model_id
        or registration_summary.get("model_version_id")
        != model.model_version_id
        or registration_summary.get("validation_run_id")
        != label.validation_run_id
        or registration_summary.get("run_label_grade") != 3
        or registration_summary.get("ranking_label") != "reuse_positive"
        or registration_summary.get("reusable_candidate") is not True
    ):
        raise LicensedModelKGInputError(
            "Registration summary is not a consistent grade-3 package."
        )
    if not (
        validation_audit.get("engineering_accepted") is True
        and validation_audit.get("scientific_accepted") is True
        and validation_audit.get("run_label_grade") == 3
        and validation_audit.get("ranking_label") == "reuse_positive"
        and validation_audit.get("reusable_candidate") is True
    ):
        raise LicensedModelKGInputError(
            "Validation audit is not an accepted grade-3 run."
        )

    input_checksums = registration_summary.get("input_checksums") or {}
    binding_checks = {
        "registration_accepted": registration_summary.get("accepted") is True,
        "protocol_checksum_matches": (
            sha256_file(protocol_path) == input_checksums.get("protocol")
            == label.protocol_sha256
        ),
        "audit_checksum_matches": (
            sha256_file(validation_audit_path)
            == input_checksums.get("audit_report")
            == label.audit_report_sha256
        ),
        "source_snapshot_checksum_matches": (
            protocol.source.snapshot_sha256
            == input_checksums.get("source_snapshot")
            == label.source_snapshot_sha256
        ),
        "project_checksum_matches": (
            model.project_sha256 == input_checksums.get("project")
        ),
        "runtime_version_matches": (
            runtime_plan.runtime.pksim_version
            == protocol.runtime.pksim_version
            == model.software_version
        ),
        "runtime_checksum_matches": (
            runtime_plan.runtime.pksim_portable.sha256
            == protocol.runtime.pksim_portable_sha256
        ),
        "grade_3_label_consistent": (
            label.run_label_grade == 3
            and label.validation_grade == "external_study_validated"
            and label.ranking_label == "reuse_positive"
            and label.engineering_accepted
            and label.scientific_accepted
            and label.reusable_candidate
        ),
    }
    if not all(binding_checks.values()):
        failed = sorted(
            name for name, passed in binding_checks.items() if not passed
        )
        raise LicensedModelKGInputError(
            "Licensed Model KG input binding failed: " + ", ".join(failed)
        )

    metrics_path = Path(label.validation_dataset_metrics_path)
    if not metrics_path.is_file():
        raise LicensedModelKGInputError(
            f"Validation dataset metrics are missing: {metrics_path}"
        )
    metrics = list(_iter_jsonl(metrics_path))
    metric_names = [str(row.get("dataset_name") or "") for row in metrics]
    if (
        len(metrics) != label.verification_dataset_count
        or any(not name for name in metric_names)
        or len(metric_names) != len(set(metric_names))
    ):
        raise LicensedModelKGInputError(
            "Validation dataset metrics are incomplete or non-unique."
        )

    simulations = list(_iter_jsonl(fact_paths["simulations"]))
    if len(simulations) != model.simulation_count:
        raise LicensedModelKGInputError(
            "Simulation fact count does not match ModelVersion."
        )

    working.mkdir(parents=True)
    provenance_directory = working / "provenance"
    provenance_directory.mkdir()
    for source, destination_name in (
        (registration_summary_path, "registration_summary.json"),
        (protocol_path, "snapshot_reuse_protocol.json"),
        (validation_audit_path, "snapshot_reuse_audit.json"),
        (metrics_path, "verification_dataset_metrics.jsonl"),
        (runtime_plan_path, "source_runtime_plan.json"),
    ):
        shutil.copy2(source, provenance_directory / destination_name)

    for path in fact_paths.values():
        _link_or_copy(path, working / path.name)
    reports_checksum = _write_gzip_jsonl(
        working / "reports.jsonl.gz",
        [],
    )
    validation_runs_checksum = _write_gzip_jsonl(
        working / "validation_runs.jsonl.gz",
        [
            {
                "model_id": model.model_id,
                "model_version_id": model.model_version_id,
                "label": label.model_dump(mode="json"),
                "datasets": metrics,
            }
        ],
    )
    (working / "failures.jsonl").write_bytes(b"")

    execution_plan = {
        "plan_schema_version": "1.0.0",
        "job_id": _stable_execution_job_id(protocol.protocol_id),
        "created_at": registration_summary["generated_at"],
        "source_id": model.source_id,
        "repository_url": protocol.source.repository_url,
        "repository_release_tag": protocol.source.release_tag,
        "repository_commit": protocol.source.commit_sha,
        "inventory_run_id": protocol.protocol_id,
        "inventory_summary_sha256": sha256_file(
            registration_summary_path
        ),
        "runtime": runtime_plan.runtime.model_dump(mode="json"),
        "export_flags": ["-k", "-x", "-c", "-r"],
        "command_timeout_seconds": (
            protocol.runtime.command_timeout_seconds
        ),
        "required_export_extensions": (
            protocol.runtime.required_export_extensions
        ),
        "projects": [
            {
                "project_key": model.project_key,
                "model_family": model.model_family,
                "relative_path": model.project_relative_path,
                "project_sha256": model.project_sha256,
                "project_name": model.model_family,
                "project_version": int(
                    (validation_audit.get("roundtrip") or {}).get(
                        "roundtrip_schema_version",
                        protocol.source.snapshot_schema_version,
                    )
                ),
                "simulations": [
                    {
                        "simulation_id": str(row["simulation_id"]),
                        "name": str(row["simulation_name"]),
                        "simulation_mode": "Individual",
                    }
                    for row in simulations
                ],
            }
        ],
    }
    execution_plan_path = working / "execution_plan.json"
    write_json_atomic(execution_plan_path, execution_plan)

    output_checksums = {
        path.name: sha256_file(working / path.name)
        for path in fact_paths.values()
    }
    output_checksums.update(
        {
            "reports.jsonl.gz": reports_checksum,
            "validation_runs.jsonl.gz": validation_runs_checksum,
            "failures.jsonl": sha256_file(working / "failures.jsonl"),
        }
    )
    summary = {
        "build_schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "job_id": execution_plan["job_id"],
        "source_id": model.source_id,
        "repository_commit": model.repository_commit,
        "execution_result_sha256": label.windows_manifest_sha256,
        "execution_validation_sha256": label.audit_report_sha256,
        "project_count": 1,
        "simulation_count": model.simulation_count,
        "parsed_simulation_count": model.simulation_count,
        "failed_simulation_count": 0,
        "model_version_count": 1,
        "reusable_candidate_count": 1,
        "parameter_record_count": model.parameter_record_count,
        "component_record_count": model.component_record_count,
        "process_record_count": model.process_record_count,
        "output_checksums": output_checksums,
        "gate1_model_version_acceptance": True,
        "blocking_reasons": [],
    }
    summary_path = working / "summary.json"
    write_json_atomic(summary_path, summary)
    gate_audit = {
        "audit_schema_version": "1.0.0",
        "auditor_version": ADAPTER_VERSION,
        "audited_at": datetime.now(UTC).isoformat(),
        "accepted": all(binding_checks.values()),
        "checks": binding_checks,
        "input_checksums": {
            "model_version_summary": sha256_file(summary_path),
            "registration_summary": sha256_file(
                registration_summary_path
            ),
            "protocol": sha256_file(protocol_path),
            "validation_audit": sha256_file(validation_audit_path),
            "runtime_plan": sha256_file(runtime_plan_path),
            "validation_dataset_metrics": sha256_file(metrics_path),
        },
        "counts": {
            **registration_summary["counts"],
            "validation_runs": 1,
        },
        "model_id": model.model_id,
        "model_version_id": model.model_version_id,
        "validation_run_id": label.validation_run_id,
    }
    write_json_atomic(working / "gate1_audit.json", gate_audit)
    adapter_manifest = {
        "adapter_version": ADAPTER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "accepted": gate_audit["accepted"],
        "source_registration_directory": str(registration_directory),
        "source_registration_summary_sha256": sha256_file(
            registration_summary_path
        ),
        "summary_sha256": sha256_file(summary_path),
        "gate1_audit_sha256": sha256_file(
            working / "gate1_audit.json"
        ),
        "execution_plan_sha256": sha256_file(execution_plan_path),
        "output_checksums": output_checksums,
    }
    write_json_atomic(working / "adapter_manifest.json", adapter_manifest)
    working.replace(output_directory)
    return adapter_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an immutable, audited Model KG input from a grade-3 "
            "licensed model registration."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_licensed_model_kg_input(
        registration_directory=args.registration,
        protocol_path=args.protocol,
        validation_audit_path=args.validation_audit,
        runtime_plan_path=args.runtime_plan,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
