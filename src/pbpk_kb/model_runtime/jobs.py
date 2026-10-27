from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    load_execution_plan,
    write_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_runtime.manifests import RepeatabilityProtocol


SELECTION_POLICY = "minimum_simulation_count_then_relative_path"


class RepeatabilityPlanError(RuntimeError):
    pass


def build_repeatability_control_plan(
    *,
    source_plan_path: Path,
    output_directory: Path,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
) -> tuple[Path, Path, RepeatabilityProtocol]:
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise RepeatabilityPlanError("Numeric tolerances must be non-negative.")
    source_plan, source_plan_sha256 = load_execution_plan(source_plan_path)
    if not source_plan.projects:
        raise RepeatabilityPlanError(
            "The source execution plan has no projects."
        )
    selected = sorted(
        source_plan.projects,
        key=lambda project: (
            len(project.simulations),
            project.relative_path.casefold(),
            project.project_key,
        ),
    )[0]
    control_identity = {
        "protocol": "osp-repeatability-control-1.0.0",
        "source_plan_sha256": source_plan_sha256,
        "project_key": selected.project_key,
        "project_sha256": selected.project_sha256,
        "simulation_ids": [
            simulation.simulation_id
            for simulation in selected.simulations
        ],
        "runtime": source_plan.runtime,
    }
    control_job_id = (
        f"osp-exec-{canonical_sha256(control_identity)[:32]}"
    )
    control_plan = source_plan.model_copy(
        update={
            "job_id": control_job_id,
            "projects": [selected],
        }
    )
    plan_path, plan_sha256 = write_execution_plan(
        output_directory,
        control_plan,
    )
    runtime_lock_sha256 = canonical_sha256(source_plan.runtime)
    protocol_identity = (
        f"{plan_sha256}\n{absolute_tolerance:.17g}\n"
        f"{relative_tolerance:.17g}\n{SELECTION_POLICY}"
    )
    protocol_id = (
        "osp-repeat-"
        + hashlib.sha256(protocol_identity.encode("utf-8")).hexdigest()[:32]
    )
    protocol = RepeatabilityProtocol(
        protocol_id=protocol_id,
        created_at=datetime.now(UTC),
        selection_policy=SELECTION_POLICY,
        source_plan_path=str(source_plan_path.resolve()),
        source_plan_sha256=source_plan_sha256,
        control_plan_path=str(plan_path.resolve()),
        control_plan_sha256=plan_sha256,
        control_job_id=control_job_id,
        control_project_key=selected.project_key,
        control_project_relative_path=selected.relative_path,
        control_project_sha256=selected.project_sha256,
        simulation_ids=[
            simulation.simulation_id
            for simulation in selected.simulations
        ],
        simulation_names=[
            simulation.name for simulation in selected.simulations
        ],
        simulation_count=len(selected.simulations),
        runtime_lock_sha256=runtime_lock_sha256,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    protocol_path = output_directory / "repeatability_protocol.json"
    write_json_atomic(protocol_path, protocol, canonical=True)
    return plan_path, protocol_path, protocol
