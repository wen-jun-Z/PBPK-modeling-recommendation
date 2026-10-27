from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pbpk_kb.paths import resolve_recorded_path

LEGACY_PATTERNS = (
    "/root/autodl-tmp",
    "/Volumes/SU700",
    "/home/xuyiqian/Research_Projects/pbpk_stage1_public_kb_ui_plan_updated",
)
CONFIGURED_FILE_KEYS = ("PBPK_MODEL_MATCHING_EVALUATION_MANIFEST",)

TEXT_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml"}


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)
    elif isinstance(value, str):
        yield value


def is_allowed_legacy_reference(relative_path: Path, value: str) -> bool:
    path = relative_path.as_posix()
    if path == "scripts/audit_runtime_paths.py":
        return True
    if path == "pbpk_kb/paths.py":
        return True
    if path == "docker-compose.yml" and value.startswith("/root/autodl-tmp"):
        return True
    return path.startswith("scripts/migrate_") or path.startswith(
        "scripts/upload_su700_"
    ) or path.startswith("scripts/verify_") or path.startswith(
        "scripts/recover_"
    ) or path == (
        "scripts/consolidate_new_server_workspace.sh"
    )


def active_path_violations(project_root: Path) -> list[dict[str, Any]]:
    candidates = [
        project_root / ".env",
        project_root / ".env.example",
        project_root / "Dockerfile",
        project_root / "docker-compose.yml",
        project_root / "pyproject.toml",
    ]
    for directory in (project_root / "pbpk_kb", project_root / "scripts"):
        candidates.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    violations: list[dict[str, Any]] = []
    pattern = re.compile("|".join(re.escape(value) for value in LEGACY_PATTERNS))
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(project_root)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            for match in pattern.finditer(line):
                value = match.group(0)
                if not is_allowed_legacy_reference(relative, value):
                    violations.append(
                        {"file": str(relative), "line": line_number, "value": value}
                    )
    return violations


def configured_file_status(project_root: Path) -> list[dict[str, Any]]:
    values: dict[str, str] = {}
    supported_keys = {*CONFIGURED_FILE_KEYS, "PBPK_MODEL_MATCHING_ROOT"}
    for env_path in (project_root / ".env", project_root / "runtime/deployment/app.env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in supported_keys:
                values[key.strip()] = value.strip().strip("\"'")
    configured_root = values.get("PBPK_MODEL_MATCHING_ROOT", "")
    model_root = Path(configured_root) if configured_root else project_root / "experiments/model_recommendation/runs"
    if not model_root.is_absolute():
        model_root = project_root / model_root
    # Inspect only this project's files, even when an obsolete configuration is present.
    inside_project = model_root.is_relative_to(project_root) and model_root.resolve().is_relative_to(project_root)
    evaluations = sorted(
        model_root.glob("expanded_catalog_*/benchmark/experiment_manifest.json"),
        key=lambda path: path.parent.parent.name,
        reverse=True,
    ) if inside_project else []
    discovered = {
        "PBPK_MODEL_MATCHING_EVALUATION_MANIFEST": evaluations[0] if evaluations else None,
    }
    result = []
    for key in CONFIGURED_FILE_KEYS:
        value = values.get(key, "")
        path = Path(value) if value else discovered[key]
        if path is not None and not path.is_absolute():
            path = project_root / path
        inside_project = bool(path and path.is_relative_to(project_root) and path.resolve().is_relative_to(project_root))
        result.append({
            "key": key,
            "selection": "configured" if value else "auto_discovered",
            "configured": value or None,
            "exists": bool(inside_project and path.is_file()),
            "resolved": str(path) if path else None,
        })
    return result


def historical_manifest_summary(project_root: Path) -> dict[str, int]:
    manifest_count = 0
    legacy_value_count = 0
    unique_values: set[str] = set()
    for path in (project_root / "data_process").rglob("*.json"):
        if path.stat().st_size > 10_000_000:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        matched_values = {
            value
            for value in iter_strings(payload)
            if value.startswith(LEGACY_PATTERNS)
        }
        matched = len(matched_values)
        if matched:
            manifest_count += 1
            legacy_value_count += matched
            unique_values.update(matched_values)
    resolved_exists = sum(resolve_recorded_path(value).exists() for value in unique_values)
    return {
        "immutable_manifest_count_with_legacy_paths": manifest_count,
        "immutable_legacy_path_value_count": legacy_value_count,
        "unique_legacy_paths": len(unique_values),
        "resolved_paths_that_exist": resolved_exists,
        "historical_paths_without_current_files": len(unique_values) - resolved_exists,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit portable PBPK runtime paths.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    layout_checks = {
        "source_package": (project_root / "src/pbpk_kb").is_dir(),
        "experiment_directory": (project_root / "experiments").is_dir(),
        "project_configuration": (project_root / "pyproject.toml").is_file(),
    }
    configured = configured_file_status(project_root)
    report = {
        "project_root": str(project_root),
        "project_layout_checks": layout_checks,
        "project_layout_valid": all(layout_checks.values()),
        "active_path_violations": active_path_violations(project_root),
        "configured_files": configured,
        "configured_files_all_exist": all(row["exists"] for row in configured),
        "historical_manifests": historical_manifest_summary(project_root),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if (
        not report["project_layout_valid"]
        or report["active_path_violations"]
        or not report["configured_files_all_exist"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
