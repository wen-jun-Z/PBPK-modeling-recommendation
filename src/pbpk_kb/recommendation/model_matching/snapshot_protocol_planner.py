from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml

from pbpk_kb.model_assets.license_policy import resolve_repository_license, sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    build_snapshot_reuse_protocol,
)


PLANNER_VERSION = "osp-snapshot-validation-protocol-planner-1.0.0"


class SnapshotProtocolPlannerError(RuntimeError):
    pass


def _run(command: list[str], *, timeout: int = 300) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_HTTP_LOW_SPEED_LIMIT": "1",
            "GIT_HTTP_LOW_SPEED_TIME": "60",
        },
    )
    if result.returncode:
        raise SnapshotProtocolPlannerError(
            result.stderr.strip() or result.stdout.strip() or "command failed"
        )
    return result.stdout.strip()


def _git(repository: Path, *args: str, timeout: int = 300) -> str:
    return _run(["git", "-C", str(repository), *args], timeout=timeout)


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_name(value: str) -> str:
    parts = PurePosixPath(value).parts
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise SnapshotProtocolPlannerError(f"Unsafe repository name: {value!r}")
    return "__".join(parts)


def _prepare_worktree(
    *, mirror: Path, checkout: Path, commit: str
) -> None:
    if checkout.exists():
        try:
            if not (checkout / ".git").exists():
                raise SnapshotProtocolPlannerError(
                    f"Existing checkout is not a Git worktree: {checkout}"
                )
            if _git(checkout, "rev-parse", "HEAD") != commit:
                raise SnapshotProtocolPlannerError(
                    f"Existing checkout commit differs: {checkout}"
                )
            if _git(checkout, "status", "--porcelain"):
                raise SnapshotProtocolPlannerError(
                    f"Existing checkout is dirty: {checkout}"
                )
            return
        except Exception:
            subprocess.run(
                ["git", "-C", str(mirror), "worktree", "remove", "--force", str(checkout)],
                check=False,
                capture_output=True,
                text=True,
            )
            shutil.rmtree(checkout, ignore_errors=True)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            _git(mirror, "worktree", "prune")
            _git(mirror, "worktree", "add", "--detach", str(checkout), commit)
            return
        except Exception as exc:
            last_error = exc
            subprocess.run(
                ["git", "-C", str(mirror), "worktree", "remove", "--force", str(checkout)],
                check=False,
                capture_output=True,
                text=True,
            )
            shutil.rmtree(checkout, ignore_errors=True)
            if attempt < 3:
                time.sleep(min(2**attempt, 20))
    assert last_error is not None
    raise last_error


def _section_index(sections: object) -> tuple[dict[int, str], dict[str, str]]:
    by_id: dict[int, str] = {}
    by_reference: dict[str, str] = {}

    def visit(items: object) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("Title") or "")
            reference = str(item.get("Reference") or "")
            label = " ".join(value for value in (reference, title) if value).strip()
            if isinstance(item.get("Id"), int):
                by_id[int(item["Id"])] = label
            if reference:
                by_reference[reference.casefold()] = label
            visit(item.get("Sections"))

    visit(sections)
    return by_id, by_reference


def _role_from_label(value: str) -> str | None:
    normalized = value.casefold().replace("_", "-")
    development_tokens = (
        "model-building",
        "model building",
        "development",
        "calibration",
        "training",
    )
    verification_tokens = (
        "model-verification",
        "model verification",
        "verification",
        "validation",
        "qualification",
    )
    development = any(token in normalized for token in development_tokens)
    verification = any(token in normalized for token in verification_tokens)
    if development and verification:
        raise SnapshotProtocolPlannerError(
            f"Section label has conflicting partition semantics: {value!r}"
        )
    return "development" if development else "verification" if verification else None


def infer_partition(
    evaluation_plan: Mapping[str, Any],
    *,
    project_ids: set[str],
) -> dict[str, Any]:
    all_plots = (evaluation_plan.get("Plots") or {}).get("AllPlots")
    if not isinstance(all_plots, list):
        raise SnapshotProtocolPlannerError("Evaluation plan has no Plots.AllPlots array.")
    section_ids, section_references = _section_index(evaluation_plan.get("Sections"))
    selected = [
        row
        for row in all_plots
        if not row.get("Project") or str(row.get("Project")) in project_ids
    ]
    if not selected:
        raise SnapshotProtocolPlannerError("Evaluation plan has no plots for the snapshot project.")
    development_ids: set[int] = set()
    verification_ids: set[int] = set()
    development_references: set[str] = set()
    verification_references: set[str] = set()
    unknown: set[str] = set()
    roles_by_simulation: dict[str, str] = {}
    for row in selected:
        simulation = str(row.get("Simulation") or "")
        if not simulation:
            raise SnapshotProtocolPlannerError("Evaluation plot has no simulation name.")
        section_id = row.get("SectionId")
        section_reference = str(row.get("SectionReference") or "")
        label = section_reference
        if isinstance(section_id, int):
            label = " ".join(value for value in (label, section_ids.get(section_id, "")) if value)
        elif section_reference:
            label = " ".join(
                value
                for value in (label, section_references.get(section_reference.casefold(), ""))
                if value
            )
        role = _role_from_label(label)
        if role is None:
            unknown.add(label or "<missing-section-selector>")
            continue
        previous = roles_by_simulation.get(simulation)
        if previous is not None and previous != role:
            raise SnapshotProtocolPlannerError(
                f"Simulation appears in both partitions: {simulation}"
            )
        roles_by_simulation[simulation] = role
        if isinstance(section_id, int):
            (development_ids if role == "development" else verification_ids).add(section_id)
        if section_reference:
            (
                development_references
                if role == "development"
                else verification_references
            ).add(section_reference)
    if unknown:
        raise SnapshotProtocolPlannerError(
            "Evaluation sections do not explicitly identify building/verification: "
            + ", ".join(sorted(unknown))
        )
    if "verification" not in set(roles_by_simulation.values()):
        raise SnapshotProtocolPlannerError(
            "Evaluation plan has no explicitly identified independent verification simulations."
        )
    return {
        "all_plots_path": "Plots.AllPlots",
        "development_section_ids": sorted(development_ids),
        "verification_section_ids": sorted(verification_ids),
        "development_section_references": sorted(development_references),
        "verification_section_references": sorted(verification_references),
        "roles_by_simulation": roles_by_simulation,
    }


def _snapshot_path(
    plan_path: Path,
    project: Mapping[str, Any],
    checkout: Path,
) -> tuple[Path, str, str]:
    raw = str(project.get("Path") or "")
    if not raw:
        raise SnapshotProtocolPlannerError("Evaluation project has no Path.")
    if raw.casefold().startswith(("http://", "https://")):
        raise SnapshotProtocolPlannerError(
            f"EXTERNAL_SNAPSHOT_REFERENCE_REQUIRES_DEPENDENCY_LOCK: {raw}"
        )
    resolved = (plan_path.parent / raw).resolve()
    resolution_mode = "declared_relative_path"
    try:
        resolved.relative_to(checkout.resolve())
    except ValueError as exc:
        raise SnapshotProtocolPlannerError("Evaluation project path escapes repository.") from exc
    if not resolved.is_file():
        candidates = sorted(
            (
                path
                for path in checkout.rglob(PurePosixPath(raw).name)
                if path.is_file() and path.suffix.casefold() == ".json"
            ),
            key=lambda path: path.relative_to(checkout).as_posix().casefold(),
        )
        if len(candidates) != 1:
            raise SnapshotProtocolPlannerError(
                "Snapshot declared path is missing and basename fallback is not "
                f"unique ({len(candidates)} candidates): {resolved}"
            )
        resolved = candidates[0].resolve()
        resolution_mode = "unique_basename_fallback"
    if resolved.suffix.casefold() != ".json":
        raise SnapshotProtocolPlannerError(f"Snapshot is not JSON: {resolved}")
    return resolved, raw, resolution_mode


def _expected_counts(snapshot: Mapping[str, Any], roles: Mapping[str, str]) -> dict[str, int]:
    fields = {
        "compounds": "Compounds",
        "expression_profiles": "ExpressionProfiles",
        "formulations": "Formulations",
        "individuals": "Individuals",
        "protocols": "Protocols",
        "simulations": "Simulations",
        "observed_data": "ObservedData",
        "simulation_classifications": "SimulationClassifications",
    }
    counts = {
        output: len(snapshot.get(source) or []) for output, source in fields.items()
    }
    simulation_names = {
        str(row.get("Name") or "") for row in snapshot.get("Simulations") or []
    }
    if "" in simulation_names or len(simulation_names) != counts["simulations"]:
        raise SnapshotProtocolPlannerError("Snapshot simulations need unique non-empty names.")
    if not set(roles) <= simulation_names:
        raise SnapshotProtocolPlannerError("Evaluation plan references simulations outside snapshot.")
    counts.update(
        {
            "model_building": sum(value == "development" for value in roles.values()),
            "model_verification": sum(value == "verification" for value in roles.values()),
            "execution_only": len(simulation_names) - len(roles),
        }
    )
    return counts


def _source_archive(checkout: Path, commit: str, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(4):
        temporary.unlink(missing_ok=True)
        try:
            _git(
                checkout,
                "archive",
                "--format=tar.gz",
                f"--output={temporary}",
                commit,
                timeout=7200,
            )
            os.replace(temporary, output_path)
            return
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(min(2**attempt, 20))
    assert last_error is not None
    raise last_error


def plan_snapshot_protocols(
    *,
    census_assets_path: Path,
    mirror_root: Path,
    workspace_root: Path,
    output_directory: Path,
    runtime_template_path: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if output_directory.exists() and not resume:
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True, exist_ok=resume)
    runtime_template = yaml.safe_load(runtime_template_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(census_assets_path)
    eligible = [row for row in rows if row.get("scientific_validation_candidate")]
    partial_protocols_path = output_directory / "partial_protocols.jsonl"
    protocol_rows: list[dict[str, Any]] = (
        _read_jsonl(partial_protocols_path)
        if resume and partial_protocols_path.is_file()
        else []
    )
    completed_versions = {
        str(row.get("version_id") or f"{row['full_name']}@{row['repository_commit']}")
        for row in protocol_rows
    }
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(eligible, 1):
        full_name = str(row["full_name"])
        commit = str(row["repository_commit"])
        version_id = str(row.get("version_id") or f"{full_name}@{commit}")
        if version_id in completed_versions:
            continue
        safe = _safe_name(full_name)
        mirror = mirror_root / safe
        checkout = workspace_root / "checkouts" / safe / commit
        row_protocol_rows: list[dict[str, Any]] = []
        row_artifacts: list[Path] = []
        try:
            _prepare_worktree(mirror=mirror, checkout=checkout, commit=commit)
            license_decision = resolve_repository_license(checkout)
            if str(license_decision.status.value) != "allowed":
                raise SnapshotProtocolPlannerError(
                    f"License is not allowed: {license_decision.reason_code}"
                )
            plan_candidates = [
                checkout / str(path)
                for path in row.get("data_assets") or []
                if PurePosixPath(str(path)).name.casefold()
                in {"evaluation_plan.json", "evaluationplan.json"}
            ]
            plan_candidates = [path for path in plan_candidates if path.is_file()]
            if len(plan_candidates) != 1:
                raise SnapshotProtocolPlannerError(
                    "Exactly one case-preserved evaluation plan is required; "
                    f"found {len(plan_candidates)}."
                )
            plan_path = plan_candidates[0]
            evaluation_plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
            projects = evaluation_plan.get("Projects") or []
            if not isinstance(projects, list) or not projects:
                raise SnapshotProtocolPlannerError("Evaluation plan has no Projects array.")
            grouped_projects: dict[Path, list[dict[str, Any]]] = defaultdict(list)
            for project in projects:
                if not isinstance(project, dict):
                    raise SnapshotProtocolPlannerError("Evaluation project is not an object.")
                snapshot_path, declared_path, resolution_mode = _snapshot_path(
                    plan_path,
                    project,
                    checkout,
                )
                grouped_projects[snapshot_path].append(
                    {
                        **project,
                        "_declared_snapshot_path": declared_path,
                        "_snapshot_resolution_mode": resolution_mode,
                    }
                )
            archive_name = f"{safe}-{commit[:12]}.tar.gz"
            archive_path = workspace_root / "archives" / archive_name
            _source_archive(checkout, commit, archive_path)
            for snapshot_path, project_rows in sorted(
                grouped_projects.items(), key=lambda item: str(item[0]).casefold()
            ):
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
                project_ids = {str(project.get("Id") or "") for project in project_rows}
                if "" in project_ids:
                    raise SnapshotProtocolPlannerError("Evaluation project ID is missing.")
                partition = infer_partition(evaluation_plan, project_ids=project_ids)
                roles = partition.pop("roles_by_simulation")
                counts = _expected_counts(snapshot, roles)
                compounds = [
                    str(item.get("Name") or "")
                    for item in snapshot.get("Compounds") or []
                    if isinstance(item, dict)
                ]
                model_name = "+".join(value for value in compounds if value) or sorted(project_ids)[0]
                relative_snapshot = snapshot_path.relative_to(checkout).as_posix()
                declared_snapshot_paths = sorted(
                    {
                        str(project["_declared_snapshot_path"])
                        for project in project_rows
                    }
                )
                resolution_modes = sorted(
                    {
                        str(project["_snapshot_resolution_mode"])
                        for project in project_rows
                    }
                )
                if len(declared_snapshot_paths) != 1 or len(resolution_modes) != 1:
                    raise SnapshotProtocolPlannerError(
                        "Projects sharing a snapshot disagree on declared path resolution."
                    )
                config = {
                    "schema_version": "1.0.0",
                    "source": {
                        "source_id": "osp_" + re_safe(model_name),
                        "model_id": re_safe(model_name),
                        "model_name": model_name,
                        "repository_url": row["clone_url"],
                        "release_tag": commit,
                        "commit_sha": commit,
                        "license_spdx_id": license_decision.spdx_id,
                        "license_relative_path": license_decision.source_location,
                        "license_sha256": license_decision.source_checksum,
                        "snapshot_relative_path": relative_snapshot,
                        "snapshot_declared_path": declared_snapshot_paths[0],
                        "snapshot_resolution_mode": resolution_modes[0],
                        "snapshot_sha256": sha256_file(snapshot_path),
                        "snapshot_schema_version": int(snapshot["Version"]),
                        "evaluation_plan_relative_path": plan_path.relative_to(checkout).as_posix(),
                        "evaluation_plan_sha256": sha256_file(plan_path),
                        "source_archive_name": archive_name,
                        "source_archive_sha256": sha256_file(archive_path),
                    },
                    "expected_counts": counts,
                    "partition": partition,
                    "runtime": runtime_template["runtime"],
                    "criteria": runtime_template["criteria"],
                }
                protocol_key = hashlib.sha256(
                    f"{full_name}\n{commit}\n{relative_snapshot}".encode("utf-8")
                ).hexdigest()[:24]
                protocol_root = output_directory / "protocols" / protocol_key
                config_path = output_directory / "configs" / f"{protocol_key}.json"
                row_artifacts.extend([protocol_root, config_path])
                _atomic_json(config_path, config)
                protocol = build_snapshot_reuse_protocol(
                    config_path=config_path,
                    repository_root=checkout,
                    source_archive=archive_path,
                    output_directory=protocol_root,
                )
                row_protocol_rows.append(
                    {
                        "protocol_key": protocol_key,
                        "protocol_id": protocol.protocol_id,
                        "full_name": full_name,
                        "version_id": row.get("version_id"),
                        "version_refs": row.get("version_refs") or ["HEAD"],
                        "repository_commit": commit,
                        "snapshot_relative_path": relative_snapshot,
                        "model_name": model_name,
                        "compounds": compounds,
                        "model_building_count": counts["model_building"],
                        "model_verification_count": counts["model_verification"],
                        "execution_only_count": counts["execution_only"],
                        "protocol_path": str(protocol_root / "snapshot_reuse_protocol.json"),
                        "protocol_sha256": sha256_file(protocol_root / "snapshot_reuse_protocol.json"),
                        "config_path": str(config_path),
                        "config_sha256": sha256_file(config_path),
                        "checkout_path": str(checkout),
                        "source_archive_path": str(archive_path),
                        "source_archive_sha256": sha256_file(archive_path),
                    }
                )
            protocol_rows.extend(row_protocol_rows)
            completed_versions.add(version_id)
        except Exception as exc:
            for artifact in row_artifacts:
                if artifact.is_dir():
                    shutil.rmtree(artifact, ignore_errors=True)
                else:
                    artifact.unlink(missing_ok=True)
            failures.append(
                {
                    "full_name": full_name,
                    "version_id": row.get("version_id"),
                    "version_refs": row.get("version_refs") or ["HEAD"],
                    "repository_commit": commit,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        protocol_rows.sort(
            key=lambda value: (
                str(value["full_name"]).casefold(),
                str(value["repository_commit"]),
                str(value["snapshot_relative_path"]).casefold(),
            )
        )
        _atomic_jsonl(partial_protocols_path, protocol_rows)
        _atomic_jsonl(output_directory / "partial_failures.jsonl", failures)
        print(
            json.dumps(
                {
                    "processed_versions": index,
                    "eligible_versions": len(eligible),
                    "completed_versions": len(completed_versions),
                    "planned_protocols": len(protocol_rows),
                    "failures": len(failures),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    protocol_rows.sort(key=lambda row: (str(row["full_name"]).casefold(), str(row["snapshot_relative_path"])))
    failures.sort(key=lambda row: str(row["full_name"]).casefold())
    outputs = {
        "protocols.jsonl": _atomic_jsonl(output_directory / "protocols.jsonl", protocol_rows),
        "failures.jsonl": _atomic_jsonl(output_directory / "failures.jsonl", failures),
    }
    summary = {
        "planner_version": PLANNER_VERSION,
        "census_assets_path": str(census_assets_path.resolve()),
        "census_assets_sha256": sha256_file(census_assets_path),
        "eligible_repository_count": len(eligible),
        "eligible_model_version_count": len(eligible),
        "completed_model_version_count": len(completed_versions),
        "planned_protocol_count": len(protocol_rows),
        "blocked_repository_count": len(failures),
        "verification_simulation_count": sum(
            int(row["model_verification_count"]) for row in protocol_rows
        ),
        "output_checksums": outputs,
        "complete": len(completed_versions) + len(failures) == len(eligible),
        "resumed": resume,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "summary.json", summary)
    return summary


def re_safe(value: str) -> str:
    normalized = "".join(character.casefold() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in normalized.split("_") if part) or "model"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build checksum-locked validation protocols from eligible OSP snapshots."
    )
    parser.add_argument("--census-assets", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = plan_snapshot_protocols(
        census_assets_path=args.census_assets,
        mirror_root=args.mirror_root,
        workspace_root=args.workspace_root,
        output_directory=args.output,
        runtime_template_path=args.runtime_template,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
