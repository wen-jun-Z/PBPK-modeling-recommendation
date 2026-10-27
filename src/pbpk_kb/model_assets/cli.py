from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pbpk_kb.model_assets.execution_plan import (
    build_execution_plan,
    write_execution_plan,
)
from pbpk_kb.model_assets.execution_validation import (
    validate_execution_result,
    write_validation_report,
)
from pbpk_kb.model_assets.gate1_audit import (
    audit_gate1_model_assets,
    write_gate1_audit,
)
from pbpk_kb.model_assets.inventory import (
    REPOSITORY_URL,
    SOURCE_ID,
    build_repository_inventory,
)
from pbpk_kb.model_assets.model_version_builder import build_model_versions
from pbpk_kb.model_assets.repository_adapter import mirror_repository
from pbpk_kb.model_assets.worker_bundle import build_windows_worker_bundle


def _run_id() -> str:
    return datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and audit public PBPK model assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mirror = subparsers.add_parser("mirror", help="Mirror and lock a model repository.")
    mirror.add_argument("--repository-url", default=REPOSITORY_URL)
    mirror.add_argument("--destination", type=Path, required=True)
    mirror.add_argument("--source-id", default=SOURCE_ID)
    mirror.add_argument("--commit")

    inventory = subparsers.add_parser(
        "inventory",
        help="Checksum and parse every asset in a locked repository checkout.",
    )
    inventory.add_argument("--repository", type=Path, required=True)
    inventory.add_argument("--output-root", type=Path, required=True)
    inventory.add_argument("--run-id", default=None)
    inventory.add_argument("--source-id", default=SOURCE_ID)
    inventory.add_argument("--repository-url", default=REPOSITORY_URL)
    inventory.add_argument("--repository-commit")
    inventory.add_argument("--default-branch", default="main")
    inventory.add_argument("--snapshot-manifest", type=Path)

    execution_plan = subparsers.add_parser(
        "build-execution-plan",
        help="Build a checksum-locked Windows execution plan from a static inventory.",
    )
    execution_plan.add_argument("--inventory", type=Path, required=True)
    execution_plan.add_argument("--repository", type=Path, required=True)
    execution_plan.add_argument("--runtime-config", type=Path, required=True)
    execution_plan.add_argument("--output", type=Path, required=True)

    validation = subparsers.add_parser(
        "validate-execution-result",
        help="Validate a Windows worker result and every exported artifact.",
    )
    validation.add_argument("--plan", type=Path, required=True)
    validation.add_argument("--result", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)

    model_versions = subparsers.add_parser(
        "build-model-versions",
        help=(
            "Parse validated PKML/results and evaluation reports into "
            "traceable ModelVersion snapshots."
        ),
    )
    model_versions.add_argument("--plan", type=Path, required=True)
    model_versions.add_argument("--inventory", type=Path, required=True)
    model_versions.add_argument("--repository", type=Path, required=True)
    model_versions.add_argument("--result", type=Path, required=True)
    model_versions.add_argument(
        "--execution-validation",
        type=Path,
        required=True,
    )
    model_versions.add_argument("--output", type=Path, required=True)

    gate1_audit = subparsers.add_parser(
        "audit-gate1",
        help=(
            "Independently verify Gate 1 checksums, record counts, schema "
            "validity, project coverage, and reuse policy."
        ),
    )
    gate1_audit.add_argument("--plan", type=Path, required=True)
    gate1_audit.add_argument("--inventory", type=Path, required=True)
    gate1_audit.add_argument(
        "--execution-validation",
        type=Path,
        required=True,
    )
    gate1_audit.add_argument(
        "--model-versions",
        type=Path,
        required=True,
    )
    gate1_audit.add_argument("--output", type=Path, required=True)

    worker_bundle = subparsers.add_parser(
        "build-windows-worker-bundle",
        help="Package a locked plan, runtime, repository, and worker code for Windows.",
    )
    worker_bundle.add_argument("--project-root", type=Path, required=True)
    worker_bundle.add_argument("--plan", type=Path, required=True)
    worker_bundle.add_argument("--repository", type=Path, required=True)
    worker_bundle.add_argument("--runtime-archive", type=Path, required=True)
    worker_bundle.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "mirror":
        snapshot = mirror_repository(
            repository_url=args.repository_url,
            destination=args.destination,
            source_id=args.source_id,
            commit=args.commit,
        )
        print(json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True))
        return

    if args.command == "build-execution-plan":
        plan = build_execution_plan(
            inventory_directory=args.inventory,
            repository_root=args.repository,
            runtime_config_path=args.runtime_config,
        )
        plan_path, plan_sha256 = write_execution_plan(args.output, plan)
        print(
            json.dumps(
                {
                    "job_id": plan.job_id,
                    "plan_path": str(plan_path),
                    "plan_sha256": plan_sha256,
                    "project_count": len(plan.projects),
                    "simulation_count": sum(
                        len(project.simulations) for project in plan.projects
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "validate-execution-result":
        report = validate_execution_result(
            plan_path=args.plan,
            result_path=args.result,
        )
        report_sha256 = write_validation_report(args.output, report)
        print(
            json.dumps(
                {
                    **report.model_dump(mode="json"),
                    "report_path": str(args.output),
                    "report_sha256": report_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not report.execution_probe_complete:
            raise SystemExit(1)
        return

    if args.command == "build-model-versions":
        summary = build_model_versions(
            plan_path=args.plan,
            inventory_directory=args.inventory,
            repository_root=args.repository,
            execution_result_path=args.result,
            execution_validation_path=args.execution_validation,
            output_directory=args.output,
        )
        print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
        if not summary.gate1_model_version_acceptance:
            raise SystemExit(1)
        return

    if args.command == "audit-gate1":
        report = audit_gate1_model_assets(
            plan_path=args.plan,
            inventory_directory=args.inventory,
            execution_validation_path=args.execution_validation,
            model_version_directory=args.model_versions,
        )
        report_sha256 = write_gate1_audit(args.output, report)
        print(
            json.dumps(
                {
                    **report,
                    "report_path": str(args.output),
                    "report_sha256": report_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not report["accepted"]:
            raise SystemExit(1)
        return

    if args.command == "build-windows-worker-bundle":
        summary = build_windows_worker_bundle(
            project_root=args.project_root,
            plan_path=args.plan,
            repository_root=args.repository,
            runtime_archive=args.runtime_archive,
            output_directory=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    output_directory = args.output_root / (args.run_id or _run_id())
    summary = build_repository_inventory(
        repository_root=args.repository,
        output_directory=output_directory,
        source_id=args.source_id,
        repository_url=args.repository_url,
        repository_commit=args.repository_commit,
        default_branch=args.default_branch,
        snapshot_manifest_path=args.snapshot_manifest,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                **summary.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
