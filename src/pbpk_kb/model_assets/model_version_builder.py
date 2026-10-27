from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.model_assets.execution_plan import (
    canonical_json_bytes,
    load_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_assets.execution_validation import validate_execution_result
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    AssetKind,
    EvaluationReportSummary,
    ExecutionProbeStatus,
    ModelAsset,
    OSPExecutionArtifact,
    OSPExecutionResult,
    OSPExecutionValidationReport,
    OSPModelVersionAssetReference,
    OSPModelVersionBuildSummary,
    OSPModelVersionSnapshot,
    OSPPKMLProjectParseSummary,
    OSPProjectParseResult,
    ParseStatus,
    RepositoryInventorySummary,
)
from pbpk_kb.model_assets.osp_pkml_parser import (
    OSPPKMLParseError,
    parse_pkml_simulation,
    parse_result_csv,
)
from pbpk_kb.model_assets.report_parser import (
    parse_markdown_report,
    parse_pdf_report,
)


class ModelVersionBuildError(RuntimeError):
    pass


class _AtomicJSONLWriter:
    def __init__(self, path: Path, *, compressed: bool) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.compressed = compressed
        self.count = 0
        self._raw = self.temporary.open("xb")
        if compressed:
            self._handle = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=self._raw,
                mtime=0,
            )
        else:
            self._handle = self._raw

    def write(self, value: Any) -> None:
        self._handle.write(canonical_json_bytes(value) + b"\n")
        self.count += 1

    def close(self) -> str:
        if self.compressed:
            self._handle.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self.temporary.replace(self.path)
        return sha256_file(self.path)

    def abort(self) -> None:
        try:
            self._handle.close()
        finally:
            if not self._raw.closed:
                self._raw.close()
            self.temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelVersionBuildError(
                    f"{path}:{line_number} contains invalid JSON."
                ) from exc


def _verify_inventory_outputs(
    inventory_directory: Path,
    summary: RepositoryInventorySummary,
) -> None:
    for name, expected_sha256 in summary.output_checksums.items():
        path = inventory_directory / name
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ModelVersionBuildError(
                f"Inventory output checksum mismatch: {path}"
            )


def _family(relative_path: str) -> str:
    return Path(relative_path).parts[0]


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\n".join(values).encode()).hexdigest()
    return f"{prefix}:{digest}"


def _execution_asset_kind(artifact: OSPExecutionArtifact) -> str:
    return {
        ".pkml": "pkml_simulation",
        ".xml": "simulation_structure_xml",
        ".csv": "simulation_result_csv",
        ".txt": "execution_log",
        ".log": "execution_log",
    }.get(artifact.extension.casefold(), "execution_artifact")


def _report_assets(
    *,
    assets: list[ModelAsset],
    repository_root: Path,
    repository_commit: str,
) -> tuple[list[EvaluationReportSummary], dict[str, list[EvaluationReportSummary]]]:
    reports: list[EvaluationReportSummary] = []
    by_family: dict[str, list[EvaluationReportSummary]] = defaultdict(list)
    for asset in assets:
        if asset.asset_kind not in {
            AssetKind.EVALUATION_REPORT_MARKDOWN,
            AssetKind.EVALUATION_REPORT_PDF,
        }:
            continue
        path = repository_root / asset.relative_path
        if not path.is_file() or sha256_file(path) != asset.sha256:
            raise ModelVersionBuildError(
                f"Evaluation report checksum mismatch: {asset.relative_path}"
            )
        if asset.asset_kind == AssetKind.EVALUATION_REPORT_MARKDOWN:
            report = parse_markdown_report(
                path,
                relative_path=asset.relative_path,
                repository_commit=repository_commit,
            )
        else:
            report = parse_pdf_report(path, relative_path=asset.relative_path)
        reports.append(report)
        by_family[_family(asset.relative_path)].append(report)
    return reports, by_family


def _artifacts_by_extension(
    artifacts: list[OSPExecutionArtifact],
    simulation_id: str,
) -> dict[str, list[OSPExecutionArtifact]]:
    selected = [
        artifact
        for artifact in artifacts
        if artifact.simulation_id == simulation_id
    ]
    output: dict[str, list[OSPExecutionArtifact]] = defaultdict(list)
    for artifact in selected:
        extension = artifact.extension.casefold()
        output[extension].append(artifact)
    if (
        set(output) != {".pkml", ".xml", ".csv"}
        or len(output[".pkml"]) != 1
        or len(output[".xml"]) != 1
        or not output[".csv"]
    ):
        raise ModelVersionBuildError(
            f"Simulation {simulation_id} does not have exactly one PKML/XML "
            "and at least one CSV artifact."
        )
    return dict(output)


def _asset_references(
    *,
    family: str,
    project_relative_path: str,
    repository_assets: list[ModelAsset],
    project_result: Any,
    job_id: str,
) -> list[OSPModelVersionAssetReference]:
    output = [
        OSPModelVersionAssetReference(
            asset_kind=asset.asset_kind.value,
            relative_path=asset.relative_path,
            sha256=asset.sha256,
            source_location=asset.source_location,
        )
        for asset in repository_assets
        if asset.model_family == family
        and (
            asset.asset_kind != AssetKind.PKSIM_PROJECT
            or asset.relative_path == project_relative_path
        )
    ]
    execution_artifacts = [
        *project_result.artifacts,
        project_result.stdout_artifact,
        project_result.stderr_artifact,
        *(
            artifact
            for simulation_result in project_result.simulation_results
            for artifact in (
                simulation_result.stdout_artifact,
                simulation_result.stderr_artifact,
            )
        ),
    ]
    output.extend(
        OSPModelVersionAssetReference(
            asset_kind=_execution_asset_kind(artifact),
            relative_path=artifact.relative_path,
            sha256=artifact.sha256,
            source_location=(
                f"OSP_EXECUTION_RESULT:{job_id}/{artifact.relative_path}"
            ),
        )
        for artifact in execution_artifacts
    )
    return sorted(
        output,
        key=lambda item: (item.asset_kind, item.relative_path.casefold()),
    )


def build_model_versions(
    *,
    plan_path: Path,
    inventory_directory: Path,
    repository_root: Path,
    execution_result_path: Path,
    execution_validation_path: Path,
    output_directory: Path,
) -> OSPModelVersionBuildSummary:
    output_directory.mkdir(parents=True, exist_ok=False)
    plan, plan_sha256 = load_execution_plan(plan_path)
    inventory_directory = inventory_directory.resolve()
    repository_root = repository_root.resolve()
    execution_result_path = execution_result_path.resolve()
    execution_result_root = execution_result_path.parent
    inventory_summary = RepositoryInventorySummary.model_validate_json(
        (inventory_directory / "summary.json").read_text(encoding="utf-8")
    )
    _verify_inventory_outputs(inventory_directory, inventory_summary)
    if inventory_summary.repository_commit != plan.repository_commit:
        raise ModelVersionBuildError("Inventory and execution plan commits differ.")

    saved_validation = OSPExecutionValidationReport.model_validate_json(
        execution_validation_path.read_text(encoding="utf-8")
    )
    live_validation = validate_execution_result(
        plan_path=plan_path,
        result_path=execution_result_path,
    )
    if (
        not saved_validation.execution_probe_complete
        or not live_validation.execution_probe_complete
        or saved_validation.job_id != live_validation.job_id
        or saved_validation.plan_sha256 != live_validation.plan_sha256
        or saved_validation.result_sha256 != live_validation.result_sha256
        or saved_validation.reason_codes
        or live_validation.reason_codes
    ):
        raise ModelVersionBuildError(
            "Execution result has not passed independent validation."
        )
    execution_result_sha256 = sha256_file(execution_result_path)
    execution_validation_sha256 = sha256_file(execution_validation_path)
    result = OSPExecutionResult.model_validate_json(
        execution_result_path.read_text(encoding="utf-8")
    )

    assets = [
        ModelAsset.model_validate(record)
        for record in _read_jsonl(inventory_directory / "assets.jsonl")
    ]
    static_projects = [
        OSPProjectParseResult.model_validate(record)
        for record in _read_jsonl(inventory_directory / "projects.jsonl")
    ]
    static_by_sha = {project.project_sha256: project for project in static_projects}
    if len(static_by_sha) != len(static_projects):
        raise ModelVersionBuildError("Static project checksums are not unique.")
    reports, reports_by_family = _report_assets(
        assets=assets,
        repository_root=repository_root,
        repository_commit=plan.repository_commit,
    )

    report_writer = _AtomicJSONLWriter(
        output_directory / "reports.jsonl.gz",
        compressed=True,
    )
    simulations_writer = _AtomicJSONLWriter(
        output_directory / "simulations.jsonl",
        compressed=False,
    )
    parameters_writer = _AtomicJSONLWriter(
        output_directory / "parameters.jsonl.gz",
        compressed=True,
    )
    components_writer = _AtomicJSONLWriter(
        output_directory / "components.jsonl.gz",
        compressed=True,
    )
    processes_writer = _AtomicJSONLWriter(
        output_directory / "processes.jsonl.gz",
        compressed=True,
    )
    administrations_writer = _AtomicJSONLWriter(
        output_directory / "administrations.jsonl",
        compressed=False,
    )
    project_summaries_writer = _AtomicJSONLWriter(
        output_directory / "project_summaries.jsonl",
        compressed=False,
    )
    model_versions_writer = _AtomicJSONLWriter(
        output_directory / "model_versions.jsonl",
        compressed=False,
    )
    failures_writer = _AtomicJSONLWriter(
        output_directory / "failures.jsonl",
        compressed=False,
    )
    writers = [
        report_writer,
        simulations_writer,
        parameters_writer,
        components_writer,
        processes_writer,
        administrations_writer,
        project_summaries_writer,
        model_versions_writer,
        failures_writer,
    ]

    parsed_simulations = 0
    failed_simulations = 0
    model_version_count = 0
    reusable_candidate_count = 0
    project_result_by_key = {project.project_key: project for project in result.projects}
    try:
        for report in sorted(reports, key=lambda item: item.relative_path.casefold()):
            report_writer.write(report)

        for project_index, project_spec in enumerate(plan.projects, start=1):
            project_result = project_result_by_key.get(project_spec.project_key)
            static_project = static_by_sha.get(project_spec.project_sha256)
            if project_result is None or static_project is None:
                raise ModelVersionBuildError(
                    f"Project inputs are incomplete: {project_spec.relative_path}"
                )
            project_compounds: set[str] = set(static_project.compounds)
            project_metabolites: set[str] = set()
            project_species: set[str] = set()
            project_populations: set[str] = set()
            project_routes: set[str] = {
                protocol.application_type
                for protocol in static_project.protocols
                if protocol.application_type
            }
            project_formulations: set[str] = set(static_project.formulations)
            project_enzymes: set[str] = set()
            project_transporters: set[str] = set()
            project_compartments: set[str] = set()
            project_process_types: set[str] = set()
            project_output_names: set[str] = set()
            project_parameter_count = 0
            project_component_count = 0
            project_process_count = 0
            project_administration_count = 0
            project_parsed_simulations = 0
            project_failed_simulations = 0
            project_reason_codes: list[str] = []

            for simulation in project_spec.simulations:
                try:
                    artifacts = _artifacts_by_extension(
                        project_result.artifacts,
                        simulation.simulation_id,
                    )
                    csv_summaries = []
                    for csv_artifact in sorted(
                        artifacts[".csv"],
                        key=lambda item: item.relative_path.casefold(),
                    ):
                        if not csv_artifact.source_relative_path:
                            raise ModelVersionBuildError(
                                "CSV artifact has no original source path."
                            )
                        csv_summaries.append(
                            parse_result_csv(
                                execution_result_root
                                / csv_artifact.relative_path,
                                artifact_relative_path=csv_artifact.relative_path,
                                source_relative_path=(
                                    csv_artifact.source_relative_path
                                ),
                                expected_sha256=csv_artifact.sha256,
                            )
                        )
                    failed_csv = [
                        summary
                        for summary in csv_summaries
                        if summary.parse_status != "parsed"
                    ]
                    timeseries_csv = [
                        summary
                        for summary in csv_summaries
                        if summary.result_kind == "timeseries"
                    ]
                    if failed_csv:
                        raise OSPPKMLParseError(
                            failed_csv[0].error
                            or "A simulation CSV could not be parsed."
                        )
                    if not timeseries_csv:
                        raise OSPPKMLParseError(
                            "Simulation has no timeseries result CSV."
                        )
                    if any(
                        summary.non_finite_value_count
                        for summary in timeseries_csv
                    ):
                        raise OSPPKMLParseError(
                            "Simulation timeseries contains non-finite values."
                        )
                    pkml_artifact = artifacts[".pkml"][0]
                    parsed = parse_pkml_simulation(
                        execution_result_root / pkml_artifact.relative_path,
                        artifact_relative_path=pkml_artifact.relative_path,
                        artifact_sha256=pkml_artifact.sha256,
                        job_id=plan.job_id,
                        project_key=project_spec.project_key,
                        simulation_id=simulation.simulation_id,
                        simulation_name=simulation.name,
                        csv_results=csv_summaries,
                    )
                    simulations_writer.write(parsed.summary)
                    for record in parsed.parameters:
                        parameters_writer.write(record)
                    for record in parsed.components:
                        components_writer.write(record)
                    for record in parsed.processes:
                        processes_writer.write(record)
                    for record in parsed.administrations:
                        administrations_writer.write(record)
                    project_parameter_count += len(parsed.parameters)
                    project_component_count += len(parsed.components)
                    project_process_count += len(parsed.processes)
                    project_administration_count += len(parsed.administrations)
                    project_compounds.update(parsed.summary.compounds)
                    project_metabolites.update(parsed.summary.metabolites)
                    project_species.update(parsed.summary.species)
                    project_populations.update(parsed.summary.populations)
                    project_routes.update(parsed.summary.routes)
                    project_formulations.update(parsed.summary.formulations)
                    project_enzymes.update(parsed.summary.enzymes)
                    project_transporters.update(parsed.summary.transporters)
                    project_compartments.update(parsed.summary.compartments)
                    project_output_names.update(parsed.summary.observers)
                    for csv_summary in timeseries_csv:
                        project_output_names.update(
                            output.name for output in csv_summary.outputs
                        )
                    project_process_types.update(
                        record.process_type for record in parsed.processes
                    )
                    project_parsed_simulations += 1
                    parsed_simulations += 1
                except Exception as exc:
                    project_failed_simulations += 1
                    failed_simulations += 1
                    project_reason_codes.append("PKML_OR_RESULT_PARSE_FAILED")
                    failures_writer.write(
                        {
                            "stage": "pkml_and_result_parse",
                            "project_key": project_spec.project_key,
                            "project_relative_path": project_spec.relative_path,
                            "simulation_id": simulation.simulation_id,
                            "simulation_name": simulation.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                print(
                    json.dumps(
                        {
                            "project": project_index,
                            "project_total": len(plan.projects),
                            "project_path": project_spec.relative_path,
                            "project_simulations_parsed": project_parsed_simulations,
                            "project_simulations_total": len(project_spec.simulations),
                            "total_simulations_parsed": parsed_simulations,
                            "total_simulations_failed": failed_simulations,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            project_parse_status = (
                ParseStatus.PARSED
                if project_failed_simulations == 0
                and project_parsed_simulations == len(project_spec.simulations)
                else ParseStatus.FAILED
            )
            project_summary = OSPPKMLProjectParseSummary(
                generated_at=datetime.now(UTC),
                job_id=plan.job_id,
                plan_sha256=plan_sha256,
                execution_result_sha256=execution_result_sha256,
                project_key=project_spec.project_key,
                project_relative_path=project_spec.relative_path,
                project_sha256=project_spec.project_sha256,
                simulation_count=len(project_spec.simulations),
                parsed_simulation_count=project_parsed_simulations,
                failed_simulation_count=project_failed_simulations,
                parameter_record_count=project_parameter_count,
                component_record_count=project_component_count,
                process_record_count=project_process_count,
                administration_record_count=project_administration_count,
                compounds=sorted(project_compounds, key=str.casefold),
                metabolites=sorted(project_metabolites, key=str.casefold),
                species=sorted(project_species, key=str.casefold),
                populations=sorted(project_populations, key=str.casefold),
                routes=sorted(project_routes, key=str.casefold),
                formulations=sorted(project_formulations, key=str.casefold),
                enzymes=sorted(project_enzymes, key=str.casefold),
                transporters=sorted(project_transporters, key=str.casefold),
                compartments=sorted(project_compartments, key=str.casefold),
                process_types=sorted(project_process_types, key=str.casefold),
                output_names=sorted(project_output_names, key=str.casefold),
                parse_status=project_parse_status,
                output_checksums={},
                reason_codes=sorted(set(project_reason_codes)),
            )
            project_summaries_writer.write(project_summary)
            family = project_spec.model_family
            family_reports = reports_by_family.get(family, [])
            model_blocking_reasons: list[str] = []
            if inventory_summary.repository_license.status.value != "allowed":
                model_blocking_reasons.append(
                    inventory_summary.repository_license.reason_code
                )
            if project_result.status != "passed":
                model_blocking_reasons.append("EXECUTION_PROBE_FAILED")
            if project_parse_status != ParseStatus.PARSED:
                model_blocking_reasons.append("PKML_PARSE_FAILED")
            # Gate 1 establishes technical runnability only. Scientific reuse is
            # assigned later by a checksum-bound Grade 3 validation package.
            reusable_candidate = False
            if reusable_candidate:
                reusable_candidate_count += 1
            model_version = OSPModelVersionSnapshot(
                model_id=_stable_id(
                    "osp-pbpk-model",
                    plan.source_id,
                    family.casefold(),
                ),
                model_version_id=_stable_id(
                    "osp-pbpk-model-version",
                    plan.source_id,
                    plan.repository_commit,
                    project_spec.relative_path,
                    project_spec.project_sha256,
                ),
                model_family=family,
                project_key=project_spec.project_key,
                project_relative_path=project_spec.relative_path,
                project_sha256=project_spec.project_sha256,
                source_id=plan.source_id,
                repository_url=plan.repository_url,
                repository_commit=plan.repository_commit,
                software="PK-Sim",
                software_version=plan.runtime.pksim_version,
                license=inventory_summary.repository_license,
                static_parse_status=static_project.parse_status,
                execution_probe_status=(
                    ExecutionProbeStatus.PASSED
                    if project_result.status == "passed"
                    else ExecutionProbeStatus.FAILED
                ),
                pkml_parse_status=project_parse_status,
                reusable_candidate=reusable_candidate,
                compounds=project_summary.compounds,
                metabolites=project_summary.metabolites,
                species=project_summary.species,
                populations=project_summary.populations,
                routes=project_summary.routes,
                formulations=project_summary.formulations,
                enzymes=project_summary.enzymes,
                transporters=project_summary.transporters,
                compartments=project_summary.compartments,
                process_types=project_summary.process_types,
                output_names=project_summary.output_names,
                simulation_count=project_summary.simulation_count,
                parameter_record_count=project_parameter_count,
                component_record_count=project_component_count,
                process_record_count=project_process_count,
                report_count=len(family_reports),
                dataset_reference_count=sum(
                    len(report.dataset_references) for report in family_reports
                ),
                validation_metric_count=sum(
                    len(report.metrics) for report in family_reports
                ),
                assets=_asset_references(
                    family=family,
                    project_relative_path=project_spec.relative_path,
                    repository_assets=assets,
                    project_result=project_result,
                    job_id=plan.job_id,
                ),
                report_paths=sorted(
                    (report.relative_path for report in family_reports),
                    key=str.casefold,
                ),
                blocking_reasons=sorted(set(model_blocking_reasons)),
            )
            model_versions_writer.write(model_version)
            model_version_count += 1

        output_checksums = {
            "reports.jsonl.gz": report_writer.close(),
            "simulations.jsonl": simulations_writer.close(),
            "parameters.jsonl.gz": parameters_writer.close(),
            "components.jsonl.gz": components_writer.close(),
            "processes.jsonl.gz": processes_writer.close(),
            "administrations.jsonl": administrations_writer.close(),
            "project_summaries.jsonl": project_summaries_writer.close(),
            "model_versions.jsonl": model_versions_writer.close(),
            "failures.jsonl": failures_writer.close(),
        }
    except Exception:
        for writer in writers:
            if not writer._raw.closed:
                writer.abort()
        raise

    technical_acceptance = (
        model_version_count == len(plan.projects)
        and parsed_simulations
        == sum(len(project.simulations) for project in plan.projects)
        and failed_simulations == 0
        and live_validation.execution_probe_complete
    )
    blocking_reasons: list[str] = []
    if not technical_acceptance:
        blocking_reasons.append("GATE1_MODEL_VERSION_TECHNICAL_ACCEPTANCE_FAILED")
    if inventory_summary.repository_license.status.value != "allowed":
        blocking_reasons.append(inventory_summary.repository_license.reason_code)
    summary = OSPModelVersionBuildSummary(
        generated_at=datetime.now(UTC),
        job_id=plan.job_id,
        source_id=plan.source_id,
        repository_commit=plan.repository_commit,
        execution_result_sha256=execution_result_sha256,
        execution_validation_sha256=execution_validation_sha256,
        project_count=len(plan.projects),
        simulation_count=sum(
            len(project.simulations) for project in plan.projects
        ),
        parsed_simulation_count=parsed_simulations,
        failed_simulation_count=failed_simulations,
        model_version_count=model_version_count,
        reusable_candidate_count=reusable_candidate_count,
        parameter_record_count=parameters_writer.count,
        component_record_count=components_writer.count,
        process_record_count=processes_writer.count,
        output_checksums=output_checksums,
        gate1_model_version_acceptance=technical_acceptance,
        blocking_reasons=blocking_reasons,
    )
    write_json_atomic(output_directory / "summary.json", summary)
    return summary
