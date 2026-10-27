from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.model_assets.execution_plan import write_json_atomic
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    LicensedSnapshotReuseProtocol,
    SnapshotWindowsManifest,
    _safe_artifact_path,
)
from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseDecision,
    LicenseStatus,
    ModelValidationRunLabel,
    OSPModelVersionAssetReference,
    OSPModelVersionSnapshot,
    ParseStatus,
)
from pbpk_kb.model_assets.osp_pkml_parser import (
    parse_pkml_simulation,
    parse_result_csv,
)
from pbpk_kb.model_assets.validation_scope import build_model_validation_scope


REGISTRAR_VERSION = "licensed-model-registrar-1.0.0"


class LicensedModelRegistrationError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_id(namespace: str, *parts: Any) -> str:
    payload = "\x1f".join(
        _canonical_json(part) if isinstance(part, (dict, list)) else str(part)
        for part in parts
    )
    return f"{namespace}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LicensedModelRegistrationError(
            f"Unable to read JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LicensedModelRegistrationError(f"Expected JSON object: {path}")
    return value


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
    os.replace(temporary, path)
    return sha256_file(path)


class _GzipJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.raw = self.temporary.open("xb")
        self.compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self.raw,
            compresslevel=6,
            mtime=0,
        )
        self.text = io.TextIOWrapper(
            self.compressed,
            encoding="utf-8",
            newline="\n",
        )
        self.count = 0

    def write(self, value: Any) -> None:
        payload = (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else value
        )
        self.text.write(_canonical_json(payload))
        self.text.write("\n")
        self.count += 1

    def close(self) -> str:
        self.text.flush()
        self.text.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        os.replace(self.temporary, self.path)
        return sha256_file(self.path)

    def abort(self) -> None:
        try:
            self.text.close()
        except Exception:
            pass
        try:
            self.raw.close()
        except Exception:
            pass
        self.temporary.unlink(missing_ok=True)


def _artifact_by_extension(
    simulation: Any,
    extension: str,
) -> Any:
    matches = [
        item
        for item in simulation.artifacts
        if item.extension.casefold() == extension.casefold()
    ]
    if len(matches) != 1:
        raise LicensedModelRegistrationError(
            f"Simulation {simulation.sequence} has {len(matches)} "
            f"{extension} artifacts; expected one."
        )
    return matches[0]


def _validation_label(
    *,
    protocol_path: Path,
    audit_path: Path,
    windows_manifest_path: Path,
    protocol: LicensedSnapshotReuseProtocol,
    audit: dict[str, Any],
) -> ModelValidationRunLabel:
    pooled = audit.get("pooled_metrics") or {}
    counts = audit.get("counts") or {}
    metrics_path = audit_path.parent / "verification_dataset_metrics.jsonl"
    if not metrics_path.is_file():
        raise LicensedModelRegistrationError(
            "Verification dataset metrics file is missing."
        )
    if sha256_file(metrics_path) != audit.get(
        "verification_dataset_metrics_sha256"
    ):
        raise LicensedModelRegistrationError(
            "Verification dataset metrics checksum mismatch."
        )
    protocol_sha256 = sha256_file(protocol_path)
    manifest_sha256 = sha256_file(windows_manifest_path)
    audit_sha256 = sha256_file(audit_path)
    validation_run_id = _stable_id(
        "snapshot-validation-run",
        protocol.protocol_id,
        protocol_sha256,
        manifest_sha256,
        audit_sha256,
    )
    return ModelValidationRunLabel(
        validation_run_id=validation_run_id,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol_sha256,
        windows_manifest_sha256=manifest_sha256,
        audit_report_sha256=audit_sha256,
        source_snapshot_sha256=protocol.source.snapshot_sha256,
        validation_grade=audit["validation_grade"],
        run_label_grade=int(audit["run_label_grade"]),
        ranking_label=audit["ranking_label"],
        engineering_accepted=bool(audit["engineering_accepted"]),
        scientific_accepted=bool(audit["scientific_accepted"]),
        reusable_candidate=bool(audit["reusable_candidate"]),
        validation_independence="repository_preregistered_holdout",
        verification_dataset_count=int(counts["evaluated_datasets"]),
        pooled_observation_count=int(
            counts["pooled_positive_observation_points"]
        ),
        pooled_gmfe=pooled.get("gmfe"),
        pooled_geometric_bias=pooled.get("geometric_bias"),
        pooled_two_fold_fraction=pooled.get("two_fold_fraction"),
        endpoint_scenario_pass_fraction=audit.get(
            "endpoint_scenario_pass_fraction"
        ),
        protocol_path=str(protocol_path.resolve()),
        windows_manifest_path=str(windows_manifest_path.resolve()),
        audit_report_path=str(audit_path.resolve()),
        validation_dataset_metrics_path=str(metrics_path.resolve()),
        validated_at=audit["audited_at"],
    )


def register_licensed_model(
    *,
    protocol_path: Path,
    audit_path: Path,
    windows_result_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    audit_path = audit_path.resolve()
    windows_result_root = windows_result_root.resolve()
    output_directory = output_directory.resolve()
    working = output_directory.with_name(output_directory.name + ".tmp")
    if output_directory.exists() or working.exists():
        raise LicensedModelRegistrationError(
            f"Output or temporary directory already exists: {output_directory}"
        )
    protocol = LicensedSnapshotReuseProtocol.model_validate_json(
        protocol_path.read_text(encoding="utf-8")
    )
    audit = _read_json(audit_path)
    manifest_path = windows_result_root / "windows_run_manifest.json"
    manifest = SnapshotWindowsManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    protocol_sha256 = sha256_file(protocol_path)
    if audit.get("protocol_id") != protocol.protocol_id or audit.get(
        "protocol_sha256"
    ) != protocol_sha256:
        raise LicensedModelRegistrationError(
            "Audit is not bound to the requested protocol."
        )
    if audit.get("windows_manifest_sha256") != sha256_file(manifest_path):
        raise LicensedModelRegistrationError(
            "Audit is not bound to the requested Windows manifest."
        )
    if not (
        audit.get("engineering_accepted") is True
        and audit.get("scientific_accepted") is True
        and audit.get("run_label_grade") == 3
        and audit.get("ranking_label") == "reuse_positive"
        and audit.get("reusable_candidate") is True
    ):
        raise LicensedModelRegistrationError(
            "Only a grade 3 reuse-positive audit can be registered."
        )
    if manifest.status != "passed" or manifest.protocol_id != protocol.protocol_id:
        raise LicensedModelRegistrationError(
            "Windows manifest is not a passed execution of this protocol."
        )
    run_1 = next((item for item in manifest.runs if item.run_id == "run_1"), None)
    if run_1 is None or run_1.status != "passed":
        raise LicensedModelRegistrationError("Passed run_1 is missing.")
    execution_by_sequence = {item.sequence: item for item in run_1.simulations}
    if set(execution_by_sequence) != {
        item.sequence for item in protocol.simulations
    }:
        raise LicensedModelRegistrationError(
            "run_1 simulation inventory does not match the protocol."
        )
    source_archive = Path(protocol.source_archive_path)
    if (
        not source_archive.is_file()
        or sha256_file(source_archive) != protocol.source.source_archive_sha256
    ):
        raise LicensedModelRegistrationError(
            "Locked source archive is absent or has the wrong checksum."
        )
    project_path = _safe_artifact_path(
        windows_result_root,
        manifest.reconstructed_project_relative_path,
    )
    if sha256_file(project_path) != manifest.reconstructed_project_sha256:
        raise LicensedModelRegistrationError(
            "Reconstructed project checksum mismatch."
        )

    working.mkdir(parents=True)
    writers = {
        name: _GzipJsonlWriter(working / f"{name}.jsonl.gz")
        for name in (
            "simulations",
            "parameters",
            "components",
            "processes",
            "administrations",
        )
    }
    project_key = protocol.protocol_id.removeprefix("snapshot-reuse-")
    compounds: set[str] = set()
    metabolites: set[str] = set()
    species: set[str] = set()
    populations: set[str] = set()
    routes: set[str] = set()
    formulations: set[str] = set()
    enzymes: set[str] = set()
    transporters: set[str] = set()
    compartments: set[str] = set()
    process_types: set[str] = set()
    output_names: set[str] = set()
    validation_scope_rows: list[dict[str, Any]] = []
    validation_metrics_by_sequence = {
        int(row["simulation_sequence"]): row
        for row in (
            json.loads(line)
            for line in (
                audit_path.parent / "verification_dataset_metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assets: list[OSPModelVersionAssetReference] = []
    try:
        for frozen in protocol.simulations:
            execution = execution_by_sequence[frozen.sequence]
            if execution.name != frozen.name or execution.status != "passed":
                raise LicensedModelRegistrationError(
                    f"Simulation identity/status mismatch: {frozen.sequence}."
                )
            csv_artifact = _artifact_by_extension(execution, ".csv")
            pkml_artifact = _artifact_by_extension(execution, ".pkml")
            csv_path = _safe_artifact_path(
                windows_result_root,
                csv_artifact.relative_path,
            )
            csv_summary = parse_result_csv(
                csv_path,
                artifact_relative_path=csv_artifact.relative_path.replace(
                    "\\", "/"
                ),
                source_relative_path=csv_artifact.simulation_relative_path.replace(
                    "\\", "/"
                ),
                expected_sha256=csv_artifact.sha256,
            )
            if csv_summary.parse_status != "parsed":
                raise LicensedModelRegistrationError(
                    f"CSV parse failed for simulation {frozen.sequence}: "
                    f"{csv_summary.error}"
                )
            pkml_path = _safe_artifact_path(
                windows_result_root,
                pkml_artifact.relative_path,
            )
            parsed = parse_pkml_simulation(
                pkml_path,
                artifact_relative_path=pkml_artifact.relative_path.replace(
                    "\\", "/"
                ),
                artifact_sha256=pkml_artifact.sha256,
                job_id=protocol.protocol_id,
                project_key=project_key,
                simulation_id=f"s{frozen.sequence:04d}",
                simulation_name=frozen.name,
                csv_results=[csv_summary],
            )
            if parsed.summary.parse_status != "parsed":
                raise LicensedModelRegistrationError(
                    f"PKML parse failed for simulation {frozen.sequence}."
                )
            simulation_row = parsed.summary.model_dump(mode="json")
            simulation_row.update(
                {
                    "protocol_sequence": frozen.sequence,
                    "protocol_role": frozen.role,
                    "observed_dataset_names": frozen.observed_dataset_names,
                    "validation_dataset_name": (
                        frozen.evaluation_series.dataset_name
                        if frozen.evaluation_series is not None
                        else None
                    ),
                    "validation_metrics": {
                        "auc0_last_ratio": validation_metrics_by_sequence.get(
                            frozen.sequence, {}
                        ).get("auc0_last_ratio"),
                        "cmax_ratio": validation_metrics_by_sequence.get(
                            frozen.sequence, {}
                        ).get("cmax_ratio"),
                    },
                }
            )
            validation_scope_rows.append(simulation_row)
            writers["simulations"].write(simulation_row)
            for name, rows in (
                ("parameters", parsed.parameters),
                ("components", parsed.components),
                ("processes", parsed.processes),
                ("administrations", parsed.administrations),
            ):
                for row in rows:
                    writers[name].write(row)
            summary = parsed.summary
            compounds.update(summary.compounds)
            metabolites.update(summary.metabolites)
            species.update(summary.species)
            populations.update(summary.populations)
            routes.update(summary.routes)
            formulations.update(summary.formulations)
            enzymes.update(summary.enzymes)
            transporters.update(summary.transporters)
            compartments.update(summary.compartments)
            process_types.update(str(item.process_type) for item in parsed.processes)
            output_names.update(
                output.name
                for result in summary.csv_results
                for output in result.outputs
            )
            for artifact in (pkml_artifact, csv_artifact):
                assets.append(
                    OSPModelVersionAssetReference(
                        asset_kind=(
                            "pkml_simulation"
                            if artifact.extension.casefold() == ".pkml"
                            else "simulation_result_csv"
                        ),
                        relative_path=artifact.relative_path.replace("\\", "/"),
                        sha256=artifact.sha256,
                        source_location=(
                            f"VALIDATION_RUN:{protocol.protocol_id}/"
                            f"{artifact.relative_path.replace(chr(92), '/')}"
                        ),
                    )
                )
            print(
                _canonical_json(
                    {
                        "simulations_parsed": frozen.sequence,
                        "simulation_total": len(protocol.simulations),
                    }
                ),
                flush=True,
            )
        output_checksums = {
            f"{name}.jsonl.gz": writer.close()
            for name, writer in writers.items()
        }
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    label = _validation_label(
        protocol_path=protocol_path,
        audit_path=audit_path,
        windows_manifest_path=manifest_path,
        protocol=protocol,
        audit=audit,
    )
    if protocol.source.model_name.casefold() not in {
        value.casefold() for value in compounds
    }:
        raise LicensedModelRegistrationError(
            "Parsed PKML assets do not contain the locked model compound."
        )
    project_relative_path = manifest.reconstructed_project_relative_path.replace(
        "\\", "/"
    )
    assets.extend(
        [
            OSPModelVersionAssetReference(
                asset_kind="pksim_project",
                relative_path=project_relative_path,
                sha256=manifest.reconstructed_project_sha256,
                source_location=(
                    f"VALIDATION_RUN:{protocol.protocol_id}/{project_relative_path}"
                ),
            ),
            OSPModelVersionAssetReference(
                asset_kind="repository_snapshot",
                relative_path=protocol.source.snapshot_relative_path,
                sha256=protocol.source.snapshot_sha256,
                source_location=(
                    f"GITHUB_BLOB:{protocol.source.commit_sha}/"
                    f"{protocol.source.snapshot_relative_path}"
                ),
            ),
            OSPModelVersionAssetReference(
                asset_kind="evaluation_plan",
                relative_path=protocol.source.evaluation_plan_relative_path,
                sha256=protocol.source.evaluation_plan_sha256,
                source_location=(
                    f"GITHUB_BLOB:{protocol.source.commit_sha}/"
                    f"{protocol.source.evaluation_plan_relative_path}"
                ),
            ),
            OSPModelVersionAssetReference(
                asset_kind="license",
                relative_path=protocol.source.license_relative_path,
                sha256=protocol.source.license_sha256,
                source_location=(
                    f"GITHUB_BLOB:{protocol.source.commit_sha}/"
                    f"{protocol.source.license_relative_path}"
                ),
            ),
            OSPModelVersionAssetReference(
                asset_kind="validation_audit",
                relative_path=audit_path.name,
                sha256=label.audit_report_sha256,
                source_location=f"VALIDATION_AUDIT:{label.validation_run_id}",
            ),
        ]
    )
    model = OSPModelVersionSnapshot(
        model_id=_stable_id(
            "licensed-pbpk-model",
            protocol.source.source_id,
            protocol.source.model_id,
        ),
        model_version_id=_stable_id(
            "licensed-pbpk-model-version",
            protocol.source.source_id,
            protocol.source.commit_sha,
            protocol.source.snapshot_sha256,
            manifest.reconstructed_project_sha256,
        ),
        model_family=protocol.source.model_name,
        project_key=project_key,
        project_relative_path=project_relative_path,
        project_sha256=manifest.reconstructed_project_sha256,
        source_id=protocol.source.source_id,
        repository_url=protocol.source.repository_url,
        repository_commit=protocol.source.commit_sha,
        software="PK-Sim",
        software_version=protocol.runtime.pksim_version,
        license=LicenseDecision(
            status=LicenseStatus.ALLOWED,
            spdx_id=protocol.source.license_spdx_id,
            source_location=(
                f"GITHUB_BLOB:{protocol.source.commit_sha}/"
                f"{protocol.source.license_relative_path}"
            ),
            source_checksum=protocol.source.license_sha256,
            reason_code="LICENSE_ALLOWED_BY_SPDX_POLICY",
        ),
        static_parse_status=ParseStatus.PARSED,
        execution_probe_status=ExecutionProbeStatus.PASSED,
        pkml_parse_status=ParseStatus.PARSED,
        reusable_candidate=True,
        compounds=sorted(compounds, key=str.casefold),
        metabolites=sorted(metabolites, key=str.casefold),
        species=sorted(species, key=str.casefold),
        populations=sorted(populations, key=str.casefold),
        routes=sorted(routes, key=str.casefold),
        formulations=sorted(formulations, key=str.casefold),
        enzymes=sorted(enzymes, key=str.casefold),
        transporters=sorted(transporters, key=str.casefold),
        compartments=sorted(compartments, key=str.casefold),
        process_types=sorted(process_types, key=str.casefold),
        output_names=sorted(output_names, key=str.casefold),
        simulation_count=writers["simulations"].count,
        parameter_record_count=writers["parameters"].count,
        component_record_count=writers["components"].count,
        process_record_count=writers["processes"].count,
        report_count=1,
        dataset_reference_count=label.verification_dataset_count,
        validation_metric_count=4,
        assets=sorted(assets, key=lambda item: (item.asset_kind, item.relative_path)),
        report_paths=[str(audit_path)],
        validation_run_label=label,
        validation_scope=build_model_validation_scope(
            validation_scope_rows
        ),
        blocking_reasons=[],
    )
    model_versions_sha256 = _write_gzip_jsonl(
        working / "model_versions.jsonl.gz",
        [model.model_dump(mode="json")],
    )
    label_sha256 = _write_gzip_jsonl(
        working / "validation_run_labels.jsonl.gz",
        [label.model_dump(mode="json")],
    )
    output_checksums.update(
        {
            "model_versions.jsonl.gz": model_versions_sha256,
            "validation_run_labels.jsonl.gz": label_sha256,
        }
    )
    summary = {
        "registration_schema_version": "1.0.0",
        "registrar_version": REGISTRAR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "accepted": True,
        "protocol_id": protocol.protocol_id,
        "model_id": model.model_id,
        "model_version_id": model.model_version_id,
        "validation_run_id": label.validation_run_id,
        "validation_grade": label.validation_grade,
        "run_label_grade": label.run_label_grade,
        "ranking_label": label.ranking_label,
        "reusable_candidate": label.reusable_candidate,
        "counts": {
            "simulations": writers["simulations"].count,
            "parameters": writers["parameters"].count,
            "components": writers["components"].count,
            "processes": writers["processes"].count,
            "administrations": writers["administrations"].count,
            "validation_datasets": label.verification_dataset_count,
        },
        "input_checksums": {
            "protocol": label.protocol_sha256,
            "windows_manifest": label.windows_manifest_sha256,
            "audit_report": label.audit_report_sha256,
            "source_snapshot": label.source_snapshot_sha256,
            "project": model.project_sha256,
        },
        "output_checksums": output_checksums,
    }
    write_json_atomic(working / "registration_summary.json", summary)
    os.replace(working, output_directory)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a grade-3 licensed PBPK model and parse all run-1 assets."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--windows-result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = register_licensed_model(
        protocol_path=args.protocol,
        audit_path=args.audit,
        windows_result_root=args.windows_result_root,
        output_directory=args.output,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
