from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.windows_batch_sharding import (
    WindowsBatchShardingError,
    deterministic_shard,
    merge_windows_snapshot_shard_results,
    partition_protocol_rows,
)


def test_deterministic_shards_are_complete_disjoint_and_stable() -> None:
    rows = [{"protocol_key": f"protocol-{index:03d}"} for index in range(40)]

    first = partition_protocol_rows(rows, 4)
    second = partition_protocol_rows(list(reversed(rows)), 4)

    first_keys = [{row["protocol_key"] for row in shard} for shard in first]
    second_keys = [{row["protocol_key"] for row in shard} for shard in second]
    assert first_keys == second_keys
    assert set().union(*first_keys) == {row["protocol_key"] for row in rows}
    assert sum(len(keys) for keys in first_keys) == 40
    assert all(
        deterministic_shard(key, 4) == shard_index
        for shard_index, keys in enumerate(first_keys)
        for key in keys
    )


def test_duplicate_protocol_key_is_rejected() -> None:
    with pytest.raises(WindowsBatchShardingError):
        partition_protocol_rows(
            [{"protocol_key": "duplicate"}, {"protocol_key": "duplicate"}],
            1,
        )


def test_assignment_does_not_depend_on_python_hash_seed() -> None:
    expected = deterministic_shard("protocol-example", 4)
    serialized = json.dumps({"key": "protocol-example", "shard": expected})
    assert json.loads(serialized)["shard"] == expected


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_merge_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    keys: dict[int, str] = {}
    index = 0
    while len(keys) < 2:
        key = f"protocol-{index}"
        keys.setdefault(deterministic_shard(key, 2), key)
        index += 1

    full_batch = tmp_path / "full_batch"
    full_jobs = [
        {
            "protocol_key": keys[shard_index],
            "protocol_id": f"protocol-id-{shard_index}",
            "protocol_sha256": str(shard_index + 1) * 64,
        }
        for shard_index in range(2)
    ]
    _write_jsonl(full_batch / "jobs.jsonl", full_jobs)
    _write_json(
        full_batch / "batch_manifest.json",
        {
            "job_count": 2,
            "jobs_sha256": sha256_file(full_batch / "jobs.jsonl"),
        },
    )

    shards_root = tmp_path / "shards"
    results_root = tmp_path / "shard_results"
    shard_records = []
    for shard_index, job in enumerate(full_jobs):
        bundle = shards_root / f"shard_{shard_index:02d}" / "bundle"
        worker = bundle / "run_licensed_snapshot_reuse.ps1"
        worker.parent.mkdir(parents=True)
        worker.write_text("Write-Host locked\n", encoding="utf-8")
        _write_jsonl(bundle / "jobs.jsonl", [job])
        _write_json(
            bundle / "batch_manifest.json",
            {
                "job_count": 1,
                "jobs_sha256": sha256_file(bundle / "jobs.jsonl"),
            },
        )
        shard_records.append(
            {
                "shard_index": shard_index,
                "protocol_count": 1,
                "protocol_keys": [job["protocol_key"]],
                "batch_relative_path": f"shard_{shard_index:02d}/bundle",
                "batch_manifest_sha256": sha256_file(
                    bundle / "batch_manifest.json"
                ),
                "jobs_sha256": sha256_file(bundle / "jobs.jsonl"),
            }
        )
        result = results_root / f"results_{shard_index:02d}"
        result_row = {
            "protocol_key": job["protocol_key"],
            "status": "executed_passed",
        }
        _write_json(
            result / "batch_summary.json",
            {"job_count": 1, "jobs": [result_row]},
        )
        _write_json(result / "batch_progress.json", [result_row])
        _write_json(
            result / str(job["protocol_key"]) / "windows_run_manifest.json",
            {
                "protocol_id": job["protocol_id"],
                "protocol_sha256": job["protocol_sha256"],
                "worker_script_sha256": sha256_file(worker),
                "status": "passed",
            },
        )
    _write_json(
        shards_root / "shard_manifest.json",
        {
            "shard_count": 2,
            "protocol_count": 2,
            "shards": shard_records,
        },
    )
    return full_batch, shards_root, results_root


@pytest.mark.parametrize("materialization", ["hardlink", "symlink"])
def test_merge_validates_and_preserves_complete_terminal_inventory(
    tmp_path: Path, materialization: str
) -> None:
    full_batch, shards_root, results_root = _build_merge_fixture(tmp_path)
    output = tmp_path / "merged"

    manifest = merge_windows_snapshot_shard_results(
        full_batch_directory=full_batch,
        shard_directory=shards_root,
        shard_results_directory=results_root,
        output_directory=output,
        materialization=materialization,
    )

    assert manifest["inventory_mutually_exclusive"] is True
    assert manifest["inventory_complete"] is True
    assert manifest["counts"]["protocol_count"] == 2
    assert manifest["counts"]["passed_count"] == 2
    assert manifest["materialization"] == materialization
    assert manifest["counts"]["symlinked_protocol_count"] == (
        2 if materialization == "symlink" else 0
    )
    assert len(list(output.glob("*/windows_run_manifest.json"))) == 2


def test_merge_rejects_missing_shard_result(tmp_path: Path) -> None:
    full_batch, shards_root, results_root = _build_merge_fixture(tmp_path)
    missing = next(results_root.glob("results_00/*/windows_run_manifest.json"))
    missing.unlink()

    with pytest.raises(WindowsBatchShardingError):
        merge_windows_snapshot_shard_results(
            full_batch_directory=full_batch,
            shard_directory=shards_root,
            shard_results_directory=results_root,
            output_directory=tmp_path / "merged",
        )
