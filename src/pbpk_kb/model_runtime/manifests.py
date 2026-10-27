from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from pbpk_kb.model_assets.models import StrictModel


class RepeatabilityProtocol(StrictModel):
    protocol_schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_id: str = Field(pattern=r"^osp-repeat-[0-9a-f]{32}$")
    created_at: datetime
    selection_policy: Literal[
        "minimum_simulation_count_then_relative_path"
    ]
    source_plan_path: str
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_plan_path: str
    control_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_job_id: str = Field(pattern=r"^osp-exec-[0-9a-f]{32}$")
    control_project_key: str = Field(pattern=r"^[0-9a-f]{32}$")
    control_project_relative_path: str
    control_project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_ids: list[str] = Field(min_length=1)
    simulation_names: list[str] = Field(min_length=1)
    simulation_count: int = Field(ge=1)
    runtime_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(ge=0)
    output_roots_required_distinct: Literal[True] = True
    license_status: Literal["license_unresolved"] = "license_unresolved"
    control_status_only: Literal[True] = True
    reuse_label_permitted: Literal[False] = False
    ranking_label_permitted: Literal[False] = False


class RepeatabilityRunInput(StrictModel):
    run_name: Literal["run_a", "run_b"]
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_path: str
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_root: str


class WorstNumericCell(StrictModel):
    simulation_id: str
    csv_ordinal: int = Field(ge=1)
    row_number: int = Field(ge=2)
    column_number: int = Field(ge=1)
    column_name: str
    run_a_value: float
    run_b_value: float
    absolute_difference: float = Field(ge=0)
    relative_difference: float = Field(ge=0)


class RepeatabilityReport(StrictModel):
    report_schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: str
    audited_at: datetime
    accepted: bool
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_reproducibility_status: Literal[
        "runtime_reproducibility_control_passed",
        "runtime_reproducibility_control_failed",
    ]
    license_status: Literal["license_unresolved"]
    reuse_eligible: Literal[False] = False
    ranking_label_created: Literal[False] = False
    runs: list[RepeatabilityRunInput] = Field(min_length=2, max_length=2)
    tolerances: dict[str, float]
    checks: dict[str, bool]
    counts: dict[str, int]
    maximum_absolute_difference: float = Field(ge=0)
    maximum_relative_difference: float = Field(ge=0)
    worst_numeric_cell: WorstNumericCell | None = None
    mismatch_manifest_path: str
    mismatch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mismatch_examples: list[dict[str, Any]] = Field(default_factory=list)
