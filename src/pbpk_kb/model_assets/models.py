from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetKind(StrEnum):
    PKSIM_PROJECT = "pksim_project"
    PKML_SIMULATION = "pkml_simulation"
    EVALUATION_REPORT_PDF = "evaluation_report_pdf"
    EVALUATION_REPORT_MARKDOWN = "evaluation_report_markdown"
    IMAGE = "image"
    DATASET = "dataset"
    LICENSE = "license"
    WORKFLOW = "workflow"
    DOCUMENTATION = "documentation"
    REPOSITORY_METADATA = "repository_metadata"
    OTHER = "other"


class LicenseStatus(StrEnum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    UNRESOLVED = "unresolved"


class ParseStatus(StrEnum):
    PARSED = "parsed"
    REGISTERED_NOT_EXECUTABLE = "registered_not_executable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class ExecutionProbeStatus(StrEnum):
    NOT_RUN = "not_run"
    QUEUED = "queued"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED_RUNTIME_UNAVAILABLE = "blocked_runtime_unavailable"


class RepositorySnapshot(StrictModel):
    source_id: str
    repository_url: str
    repository_path: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    default_branch: str
    is_clean: bool
    captured_at: datetime


class LicenseDecision(StrictModel):
    status: LicenseStatus
    spdx_id: str | None = None
    source_location: str | None = None
    source_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reason_code: str


class ModelAsset(StrictModel):
    asset_id: str
    model_family: str | None = None
    relative_path: str
    asset_kind: AssetKind
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    source_id: str
    source_location: str
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: LicenseDecision
    parse_status: ParseStatus
    parse_error: str | None = None


class OSPCompressedContentAudit(StrictModel):
    content_id: int
    filename: str
    storage_format: Literal["zip64_deflate", "zip_deflate", "utf8_xml"]
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(ge=0)
    crc32: str = Field(pattern=r"^[0-9a-f]{8}$")
    crc_valid: bool | None
    xml_root: str
    xml_version: str | None = None


class OSPBuildingBlockSummary(StrictModel):
    building_block_id: str
    name: str
    icon: str
    version: int
    structure_version: int
    content_id: int
    xml_root: str
    xml_version: str | None = None


class OSPProtocolSummary(StrictModel):
    protocol_id: str
    name: str
    application_type: str | None = None
    dosing_interval: str | None = None
    parameters: dict[str, str | float | int | None] = Field(default_factory=dict)


class OSPParameterSummary(StrictModel):
    parameter_id: str
    name: str
    value: float | str | None = None
    formula_id: str | None = None
    dimension: str | None = None
    display_unit: str | None = None
    is_default: bool | None = None


class OSPCompoundSummary(StrictModel):
    compound_id: str
    name: str
    parameters: list[OSPParameterSummary]
    process_names: list[str]


class OSPSimulationSummary(StrictModel):
    simulation_id: str
    name: str
    simulation_mode: str
    content_id: int
    has_stored_results: bool
    used_building_block_count: int = Field(ge=0)
    observed_dataset_count: int = Field(ge=0)
    organ_names: list[str] = Field(default_factory=list)
    molecule_names: list[str] = Field(default_factory=list)
    reaction_names: list[str] = Field(default_factory=list)
    transport_names: list[str] = Field(default_factory=list)
    observer_names: list[str] = Field(default_factory=list)
    parameter_count: int = Field(ge=0)
    equation_count: int = Field(ge=0)


class OSPProjectParseResult(StrictModel):
    parser_name: Literal["osp_pksim5_sqlite_parser"] = "osp_pksim5_sqlite_parser"
    parser_version: str
    project_path: str
    project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    project_id: int
    project_name: str
    project_version: int
    sqlite_integrity: Literal["ok"]
    schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parse_status: ParseStatus
    execution_probe_status: ExecutionProbeStatus
    compounds: list[str]
    compound_details: list[OSPCompoundSummary]
    expression_profiles: list[str]
    formulations: list[str]
    individuals: list[str]
    protocols: list[OSPProtocolSummary]
    simulations: list[OSPSimulationSummary]
    observed_datasets: list[str]
    parameter_identifications: list[str]
    building_blocks: list[OSPBuildingBlockSummary]
    compressed_content_audit: list[OSPCompressedContentAudit]
    null_content_ids: list[int] = Field(default_factory=list)
    table_row_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)


class EvaluationReportSummary(StrictModel):
    relative_path: str
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["markdown", "pdf"]
    title: str | None = None
    headings: list[str] = Field(default_factory=list)
    table_count: int = Field(ge=0)
    link_count: int = Field(ge=0)
    page_count: int | None = Field(default=None, ge=0)
    text_length: int = Field(ge=0)
    extraction_method: str
    warnings: list[str] = Field(default_factory=list)
    sections: list["EvaluationReportSection"] = Field(default_factory=list)
    tables: list["EvaluationReportTable"] = Field(default_factory=list)
    dataset_references: list["EvaluationDatasetReference"] = Field(
        default_factory=list
    )
    metrics: list["EvaluationMetricRecord"] = Field(default_factory=list)


class EvaluationReportSection(StrictModel):
    heading: str
    level: int = Field(ge=1, le=6)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    heading_path: list[str]


class EvaluationReportTable(StrictModel):
    table_id: str
    heading_path: list[str]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    headers: list[str]
    rows: list[list[str]]
    semantic_role: Literal[
        "model_metadata",
        "input_parameter",
        "calibration_dataset",
        "qualification_dataset",
        "validation_dataset",
        "validation_metric",
        "reference",
        "other",
    ]


class EvaluationDatasetReference(StrictModel):
    dataset_id: str
    role: Literal["calibration", "qualification", "validation", "unresolved"]
    name: str
    reference: str | None = None
    description: str | None = None
    table_id: str
    row_number: int = Field(ge=1)
    source_location: str


class EvaluationMetricRecord(StrictModel):
    metric_id: str
    metric_name: str
    value: float
    unit: str | None = None
    group: str | None = None
    table_id: str
    row_number: int = Field(ge=1)
    source_location: str


class RepositoryInventorySummary(StrictModel):
    inventory_version: str
    source_id: str
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at: datetime
    repository_snapshot: RepositorySnapshot
    repository_license: LicenseDecision
    asset_count: int = Field(ge=0)
    model_project_count: int = Field(ge=0)
    parsed_project_count: int = Field(ge=0)
    failed_project_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    asset_kind_counts: dict[str, int]
    parse_status_counts: dict[str, int]
    project_schema_fingerprints: dict[str, int]
    output_checksums: dict[str, str]
    gate1_static_acceptance: bool
    execution_probe_complete: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeDownloadLock(StrictModel):
    name: str
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OSPRuntimeLock(StrictModel):
    suite_release_name: str
    suite_version: str
    suite_installer: RuntimeDownloadLock
    pksim_version: str
    pksim_portable: RuntimeDownloadLock
    pksim_cli_relative_path: str
    ospsuite_r_version: str
    ospsuite_r_linux_binary: RuntimeDownloadLock


class OSPExecutionSimulationSpec(StrictModel):
    simulation_id: str
    name: str
    simulation_mode: str


class OSPExecutionProjectSpec(StrictModel):
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    model_family: str
    relative_path: str
    project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_name: str
    project_version: int
    simulations: list[OSPExecutionSimulationSpec]


class OSPExecutionPlan(StrictModel):
    plan_schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: str = Field(pattern=r"^osp-exec-[0-9a-f]{32}$")
    created_at: datetime
    source_id: str
    repository_url: str
    repository_release_tag: str
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    inventory_run_id: str
    inventory_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: OSPRuntimeLock
    export_flags: list[Literal["-k", "-x", "-c", "-r"]]
    command_timeout_seconds: int = Field(gt=0)
    required_export_extensions: list[Literal[".pkml", ".xml", ".csv"]]
    projects: list[OSPExecutionProjectSpec]


class OSPExecutionArtifact(StrictModel):
    relative_path: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extension: str
    source_relative_path: str | None = None
    simulation_id: str | None = None
    simulation_name: str | None = None


class OSPSimulationExecutionResult(StrictModel):
    sequence: int = Field(ge=1)
    simulation_id: str
    simulation_name: str
    status: Literal["passed", "failed"]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = Field(ge=0)
    command: list[str]
    return_code: int | None
    export_counts: dict[str, int]
    stdout_artifact: OSPExecutionArtifact
    stderr_artifact: OSPExecutionArtifact
    reason_codes: list[str] = Field(default_factory=list)


class OSPProjectExecutionResult(StrictModel):
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    relative_path: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt: int = Field(ge=1)
    status: Literal["passed", "failed"]
    execution_mode: Literal["project_batch", "per_simulation"] = "project_batch"
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = Field(ge=0)
    command: list[str]
    return_code: int | None
    expected_simulation_count: int = Field(ge=0)
    output_directory_count: int = Field(ge=0)
    export_counts: dict[str, int]
    artifacts: list[OSPExecutionArtifact]
    stdout_artifact: OSPExecutionArtifact
    stderr_artifact: OSPExecutionArtifact
    simulation_results: list[OSPSimulationExecutionResult] = Field(
        default_factory=list
    )
    reason_codes: list[str] = Field(default_factory=list)


class OSPWorkerRuntimeIdentity(StrictModel):
    platform: str
    platform_release: str
    python_version: str
    hostname: str
    runtime_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cli_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cli_relative_path: str
    cli_version_output: str
    cli_version_return_code: int
    cli_help_output: str
    cli_help_return_code: int
    cli_help_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OSPExecutionResult(StrictModel):
    result_schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: str = Field(pattern=r"^osp-exec-[0-9a-f]{32}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    completed_at: datetime
    worker: OSPWorkerRuntimeIdentity
    status: Literal["passed", "failed"]
    projects: list[OSPProjectExecutionResult]
    reason_codes: list[str] = Field(default_factory=list)


class OSPExecutionValidationReport(StrictModel):
    validation_schema_version: Literal["1.0.0"] = "1.0.0"
    validated_at: datetime
    job_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_count: int = Field(ge=0)
    expected_simulation_count: int = Field(ge=0)
    passed_project_count: int = Field(ge=0)
    verified_artifact_count: int = Field(ge=0)
    execution_probe_complete: bool
    reason_codes: list[str] = Field(default_factory=list)


class OSPArtifactSource(StrictModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_location: str


class OSPPKMLParameterRecord(StrictModel):
    parser_name: Literal["osp_pkml_stream_parser"] = "osp_pkml_stream_parser"
    parser_version: str
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    simulation_id: str
    simulation_name: str
    element_type: str
    parameter_id: str
    name: str
    value: float | str | None = None
    formula_id: str | None = None
    dimension: str | None = None
    display_unit: str | None = None
    is_default: bool | None = None
    quantity_type: str | None = None
    scope_path: str
    value_origin_source: str | None = None
    value_origin_description: str | None = None
    source: OSPArtifactSource


class OSPPKMLComponentRecord(StrictModel):
    parser_name: Literal["osp_pkml_stream_parser"] = "osp_pkml_stream_parser"
    parser_version: str
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    simulation_id: str
    simulation_name: str
    component_id: str
    name: str
    component_type: Literal[
        "compound",
        "metabolite",
        "enzyme",
        "transporter",
        "protein",
        "molecular_complex",
        "molecular_entity",
        "compartment",
        "observer",
        "individual",
        "solver",
        "other",
    ]
    element_type: str
    description: str | None = None
    quantity_type: str | None = None
    icon: str | None = None
    dimension: str | None = None
    display_unit: str | None = None
    parent_container_id: str | None = None
    parent_container_name: str | None = None
    source: OSPArtifactSource


class OSPPKMLProcessParticipant(StrictModel):
    identifier: str
    name: str | None = None
    component_type: Literal[
        "compound",
        "metabolite",
        "enzyme",
        "transporter",
        "protein",
        "molecular_complex",
        "molecular_entity",
        "compartment",
        "observer",
        "individual",
        "solver",
        "other",
    ] | None = None
    resolution_status: Literal[
        "resolved_by_component_id",
        "resolved_by_component_name",
        "unresolved",
    ]
    stoichiometric_coefficient: float | None = Field(default=None, gt=0)


class OSPPKMLProcessRecord(StrictModel):
    parser_name: Literal["osp_pkml_stream_parser"] = "osp_pkml_stream_parser"
    parser_version: str
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    simulation_id: str
    simulation_name: str
    process_id: str
    name: str
    process_type: Literal[
        "absorption",
        "distribution",
        "metabolism",
        "transport",
        "renal_excretion",
        "biliary_excretion",
        "other",
    ]
    element_type: str
    description: str | None = None
    dimension: str | None = None
    formula_id: str | None = None
    educts: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    educt_identifiers: list[str] = Field(default_factory=list)
    product_identifiers: list[str] = Field(default_factory=list)
    modifier_identifiers: list[str] = Field(default_factory=list)
    educt_participants: list[OSPPKMLProcessParticipant] = Field(
        default_factory=list
    )
    product_participants: list[OSPPKMLProcessParticipant] = Field(
        default_factory=list
    )
    modifier_participants: list[OSPPKMLProcessParticipant] = Field(
        default_factory=list
    )
    input_compartment_id: str | None = None
    output_compartment_id: str | None = None
    unresolved_partner_identifiers: list[str] = Field(default_factory=list)
    source: OSPArtifactSource


class OSPPKMLAdministrationRecord(StrictModel):
    parser_name: Literal["osp_pkml_stream_parser"] = "osp_pkml_stream_parser"
    parser_version: str
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    simulation_id: str
    simulation_name: str
    administration_id: str
    name: str
    route: str
    molecule_name: str | None = None
    formulation: str | None = None
    source: OSPArtifactSource


class OSPPKMLOutputColumn(StrictModel):
    name: str
    unit: str | None = None


class OSPPKMLResultCSVSummary(StrictModel):
    relative_path: str
    source_relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_kind: Literal["timeseries", "population", "pk_analysis", "other"]
    row_count: int = Field(ge=0)
    individual_ids: list[str]
    columns: list[OSPPKMLOutputColumn]
    metadata: dict[str, str] = Field(default_factory=dict)
    time_column: str | None = None
    time_unit: str | None = None
    time_min: float | None = None
    time_max: float | None = None
    outputs: list[OSPPKMLOutputColumn]
    numeric_value_count: int = Field(ge=0)
    text_value_count: int = Field(ge=0)
    blank_value_count: int = Field(ge=0)
    non_finite_value_count: int = Field(ge=0)
    parse_status: Literal["parsed", "failed"]
    error: str | None = None


class OSPPKMLSimulationSummary(StrictModel):
    parser_name: Literal["osp_pkml_stream_parser"] = "osp_pkml_stream_parser"
    parser_version: str
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    simulation_id: str
    simulation_name: str
    embedded_simulation_id: str | None = None
    pkml_source: OSPArtifactSource
    xml_version: str | None = None
    species: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    compounds: list[str] = Field(default_factory=list)
    metabolites: list[str] = Field(default_factory=list)
    enzymes: list[str] = Field(default_factory=list)
    transporters: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    formulations: list[str] = Field(default_factory=list)
    compartments: list[str] = Field(default_factory=list)
    observers: list[str] = Field(default_factory=list)
    solver: str | None = None
    parameter_count: int = Field(ge=0)
    component_count: int = Field(ge=0)
    process_count: int = Field(ge=0)
    administration_count: int = Field(ge=0)
    csv_results: list[OSPPKMLResultCSVSummary] = Field(default_factory=list)
    parse_status: Literal["parsed", "failed"]
    warnings: list[str] = Field(default_factory=list)


class OSPModelVersionAssetReference(StrictModel):
    asset_kind: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_location: str


class OSPPKMLProjectParseSummary(StrictModel):
    project_parse_schema_version: Literal["1.0.0"] = "1.0.0"
    generated_at: datetime
    job_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    project_relative_path: str
    project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_count: int = Field(ge=0)
    parsed_simulation_count: int = Field(ge=0)
    failed_simulation_count: int = Field(ge=0)
    parameter_record_count: int = Field(ge=0)
    component_record_count: int = Field(ge=0)
    process_record_count: int = Field(ge=0)
    administration_record_count: int = Field(ge=0)
    compounds: list[str]
    metabolites: list[str]
    species: list[str]
    populations: list[str]
    routes: list[str]
    formulations: list[str]
    enzymes: list[str]
    transporters: list[str]
    compartments: list[str]
    process_types: list[str]
    output_names: list[str]
    parse_status: ParseStatus
    output_checksums: dict[str, str]
    reason_codes: list[str] = Field(default_factory=list)


class ModelValidationRunLabel(StrictModel):
    label_schema_version: Literal["1.0.0"] = "1.0.0"
    validation_run_id: str
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    windows_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_grade: Literal[
        "execution_failed",
        "execution_verified",
        "external_study_validated",
    ]
    run_label_grade: int = Field(ge=0, le=3)
    ranking_label: Literal[
        "reuse_positive",
        "abstain_validation_failed",
        "abstain_execution_failed",
    ]
    engineering_accepted: bool
    scientific_accepted: bool
    reusable_candidate: bool
    validation_independence: Literal[
        "repository_preregistered_holdout",
        "externally_curated_independent",
    ] = "repository_preregistered_holdout"
    verification_dataset_count: int = Field(ge=0)
    pooled_observation_count: int = Field(ge=0)
    pooled_gmfe: float | None = Field(default=None, ge=1)
    pooled_geometric_bias: float | None = Field(default=None, gt=0)
    pooled_two_fold_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    endpoint_scenario_pass_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    protocol_path: str
    windows_manifest_path: str
    audit_report_path: str
    validation_dataset_metrics_path: str
    validated_at: datetime

    @model_validator(mode="after")
    def validation_grade_is_consistent(self) -> "ModelValidationRunLabel":
        expected = {
            "external_study_validated": (3, "reuse_positive", True, True, True),
            "execution_verified": (
                1,
                "abstain_validation_failed",
                True,
                False,
                False,
            ),
            "execution_failed": (
                0,
                "abstain_execution_failed",
                False,
                False,
                False,
            ),
        }[self.validation_grade]
        observed = (
            self.run_label_grade,
            self.ranking_label,
            self.engineering_accepted,
            self.scientific_accepted,
            self.reusable_candidate,
        )
        if observed != expected:
            raise ValueError(
                "Validation grade, run label, acceptance, and reuse fields "
                "are inconsistent."
            )
        if self.scientific_accepted and (
            self.verification_dataset_count == 0
            or self.pooled_observation_count == 0
            or self.pooled_gmfe is None
            or self.pooled_geometric_bias is None
            or self.pooled_two_fold_fraction is None
            or self.endpoint_scenario_pass_fraction is None
        ):
            raise ValueError(
                "Scientific acceptance requires complete validation metrics."
            )
        if (
            self.reusable_candidate
            and self.validation_independence
            not in {
                "repository_preregistered_holdout",
                "externally_curated_independent",
            }
        ):
            raise ValueError(
                "Reusable validation must declare its independence class."
            )
        return self


class ModelValidationScenario(StrictModel):
    simulation_id: str
    simulation_name: str
    dataset_name: str
    source_location: str
    species: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    formulations: list[str] = Field(default_factory=list)
    evaluated_endpoints: list[str] = Field(default_factory=list)
    dose: float | None = Field(default=None, gt=0)
    dose_unit: str | None = None
    frequency: str | None = None
    single_or_multiple: Literal["single", "multiple", "unknown"] = "unknown"

    @model_validator(mode="after")
    def scenario_is_validation_bound(self) -> "ModelValidationScenario":
        if not self.dataset_name or not self.source_location:
            raise ValueError(
                "Validation scenarios require dataset and source provenance."
            )
        if self.dose is None and self.dose_unit is not None:
            raise ValueError("A validation dose unit requires a numeric dose.")
        if self.dose is not None and self.dose_unit is None:
            raise ValueError("A validation dose requires a dose unit.")
        return self


class ModelValidationScope(StrictModel):
    scope_schema_version: Literal["1.0.0"] = "1.0.0"
    derivation_method: Literal[
        "registered_model_verification_simulations"
    ] = "registered_model_verification_simulations"
    species: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    formulations: list[str] = Field(default_factory=list)
    evaluated_endpoints: list[str] = Field(default_factory=list)
    dataset_names: list[str] = Field(default_factory=list)
    scenarios: list[ModelValidationScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def aggregate_scope_matches_scenarios(self) -> "ModelValidationScope":
        def aggregate(field: str) -> list[str]:
            return sorted(
                {
                    str(value)
                    for scenario in self.scenarios
                    for value in getattr(scenario, field)
                },
                key=str.casefold,
            )

        expected = {
            "species": aggregate("species"),
            "populations": aggregate("populations"),
            "routes": aggregate("routes"),
            "formulations": aggregate("formulations"),
            "evaluated_endpoints": aggregate("evaluated_endpoints"),
            "dataset_names": sorted(
                {scenario.dataset_name for scenario in self.scenarios},
                key=str.casefold,
            ),
        }
        observed = {
            field: getattr(self, field) for field in expected
        }
        if observed != expected:
            raise ValueError(
                "Validation-scope aggregates must be derived exactly from "
                "their registered verification scenarios."
            )
        return self


class OSPModelVersionSnapshot(StrictModel):
    model_id: str
    model_version_id: str
    model_family: str
    project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    project_relative_path: str
    project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str
    repository_url: str
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    software: Literal["PK-Sim"]
    software_version: str
    license: LicenseDecision
    static_parse_status: ParseStatus
    execution_probe_status: ExecutionProbeStatus
    pkml_parse_status: ParseStatus
    reusable_candidate: bool
    compounds: list[str]
    metabolites: list[str]
    species: list[str]
    populations: list[str]
    routes: list[str]
    formulations: list[str]
    enzymes: list[str]
    transporters: list[str]
    compartments: list[str]
    process_types: list[str]
    output_names: list[str]
    simulation_count: int = Field(ge=0)
    parameter_record_count: int = Field(ge=0)
    component_record_count: int = Field(ge=0)
    process_record_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    dataset_reference_count: int = Field(ge=0)
    validation_metric_count: int = Field(ge=0)
    assets: list[OSPModelVersionAssetReference]
    report_paths: list[str]
    validation_run_label: ModelValidationRunLabel | None = None
    validation_scope: ModelValidationScope | None = None
    blocking_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reusable_state_is_scientifically_bound(
        self,
    ) -> "OSPModelVersionSnapshot":
        label = self.validation_run_label
        label_reusable = bool(label is not None and label.reusable_candidate)
        if self.reusable_candidate != label_reusable:
            raise ValueError(
                "Model reusable_candidate must be derived from a bound "
                "Grade 3 validation label, not technical execution alone."
            )
        if not self.reusable_candidate:
            return self

        project_assets = [
            asset
            for asset in self.assets
            if asset.asset_kind == "pksim_project"
            and asset.relative_path == self.project_relative_path
            and asset.sha256 == self.project_sha256
        ]
        source_snapshots = [
            asset
            for asset in self.assets
            if asset.asset_kind == "repository_snapshot"
            and label is not None
            and asset.sha256 == label.source_snapshot_sha256
        ]
        accepted = bool(
            label is not None
            and label.validation_grade == "external_study_validated"
            and label.run_label_grade == 3
            and label.ranking_label == "reuse_positive"
            and label.engineering_accepted
            and label.scientific_accepted
            and self.license.status == LicenseStatus.ALLOWED
            and self.static_parse_status == ParseStatus.PARSED
            and self.execution_probe_status == ExecutionProbeStatus.PASSED
            and self.pkml_parse_status == ParseStatus.PARSED
            and project_assets
            and source_snapshots
            and self.validation_scope is not None
            and len(self.validation_scope.scenarios)
            == label.verification_dataset_count
            and not self.blocking_reasons
        )
        if not accepted:
            raise ValueError(
                "Reusable models require licensed, checksum-bound project and "
                "source assets, successful parsing/execution, a complete Grade "
                "3 scientific validation label, an exact validation-scenario "
                "scope, and no blockers."
            )
        return self


class OSPModelVersionBuildSummary(StrictModel):
    build_schema_version: Literal["1.0.0"] = "1.0.0"
    generated_at: datetime
    job_id: str
    source_id: str
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_count: int = Field(ge=0)
    simulation_count: int = Field(ge=0)
    parsed_simulation_count: int = Field(ge=0)
    failed_simulation_count: int = Field(ge=0)
    model_version_count: int = Field(ge=0)
    reusable_candidate_count: int = Field(ge=0)
    parameter_record_count: int = Field(ge=0)
    component_record_count: int = Field(ge=0)
    process_record_count: int = Field(ge=0)
    output_checksums: dict[str, str]
    gate1_model_version_acceptance: bool
    blocking_reasons: list[str] = Field(default_factory=list)


EvaluationReportSummary.model_rebuild()
