from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


ALIGNMENT_VERSION = "pbpk-perturbation-split-alignment-1.0.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _non_split_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = {key: item for key, item in row.items() if key != "split"}
        digest.update(
            (
                json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def align_perturbation_splits(
    *,
    benchmark_directory: Path,
    perturbed_tasks_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    benchmark_tasks_path = benchmark_directory / "benchmark_tasks.jsonl"
    base_tasks = {
        str(row["task_id"]): row for row in _read_jsonl(benchmark_tasks_path)
    }
    perturbed_tasks = _read_jsonl(perturbed_tasks_path)
    if len({str(row["task_id"]) for row in perturbed_tasks}) != len(
        perturbed_tasks
    ):
        raise ValueError("Perturbed task IDs are not unique.")
    unknown = sorted(
        {
            str(row.get("base_task_id") or "")
            for row in perturbed_tasks
            if str(row.get("base_task_id") or "") not in base_tasks
        }
    )
    if unknown:
        raise ValueError(f"Perturbations reference unknown base tasks: {unknown}")

    aligned: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in perturbed_tasks:
        base_task_id = str(row["base_task_id"])
        expected_split = str(base_tasks[base_task_id]["split"])
        previous_split = str(row.get("split") or "")
        aligned.append({**row, "split": expected_split})
        if previous_split != expected_split:
            changes.append(
                {
                    "task_id": row["task_id"],
                    "base_task_id": base_task_id,
                    "from_split": previous_split,
                    "to_split": expected_split,
                }
            )

    output_directory.mkdir(parents=True)
    source_non_split_sha256 = _non_split_sha256(perturbed_tasks)
    output_non_split_sha256 = _non_split_sha256(aligned)
    if source_non_split_sha256 != output_non_split_sha256:
        raise RuntimeError("Non-split perturbation fields changed during alignment.")
    tasks_sha256 = _atomic_jsonl(
        output_directory / "perturbed_tasks.jsonl", aligned
    )
    changes_sha256 = _atomic_jsonl(
        output_directory / "split_reassignments.jsonl", changes
    )
    manifest = {
        "alignment_version": ALIGNMENT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_directory": str(benchmark_directory.resolve()),
        "benchmark_tasks_path": str(benchmark_tasks_path.resolve()),
        "benchmark_tasks_sha256": _sha256(benchmark_tasks_path),
        "source_perturbed_tasks_path": str(perturbed_tasks_path.resolve()),
        "source_perturbed_tasks_sha256": _sha256(perturbed_tasks_path),
        "perturbed_task_count": len(aligned),
        "split_reassignment_count": len(changes),
        "source_non_split_sha256": source_non_split_sha256,
        "output_non_split_sha256": output_non_split_sha256,
        "output_checksums": {
            "perturbed_tasks.jsonl": tasks_sha256,
            "split_reassignments.jsonl": changes_sha256,
        },
        "non_split_fields_preserved": True,
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align perturbation splits to a frozen base-task split."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--perturbed-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = align_perturbation_splits(
        benchmark_directory=args.benchmark,
        perturbed_tasks_path=args.perturbed_tasks,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
