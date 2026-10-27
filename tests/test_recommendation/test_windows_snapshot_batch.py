from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.windows_snapshot_batch import (
    build_windows_snapshot_batch,
)


def test_windows_batch_copies_checksum_locked_protocol_and_snapshot(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts" / "windows"
    scripts.mkdir(parents=True)
    for name in (
        "run_licensed_snapshot_reuse.ps1",
        "run_snapshot_protocol_batch.ps1",
    ):
        (scripts / name).write_text("Write-Host test\n", encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    snapshot = checkout / "Drug-Model.json"
    snapshot.write_text("{}\n", encoding="utf-8")
    runtime = tmp_path / "runtime.zip"
    runtime.write_bytes(b"runtime")
    protocol = tmp_path / "snapshot_reuse_protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "protocol_id": "snapshot-reuse-" + "a" * 32,
                "runtime": {"pksim_portable_sha256": sha256_file(runtime)},
                "source": {"snapshot_sha256": sha256_file(snapshot)},
                "simulations": [
                    {
                        "evaluation_series": {"dataset_name": "study"},
                        "additional_evaluation_series": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    protocols = tmp_path / "protocols.jsonl"
    protocols.write_text(
        json.dumps(
            {
                "protocol_key": "key1",
                "protocol_id": "snapshot-reuse-" + "a" * 32,
                "protocol_path": str(protocol),
                "protocol_sha256": sha256_file(protocol),
                "checkout_path": str(checkout),
                "snapshot_relative_path": snapshot.name,
                "full_name": "OSP/Drug-Model",
                "version_id": "OSP/Drug-Model@commit",
                "repository_commit": "b" * 40,
                "model_name": "Drug",
                "compounds": ["Drug"],
                "model_verification_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_windows_snapshot_batch(
        project_root=project,
        protocols_path=protocols,
        runtime_archive=runtime,
        output_directory=tmp_path / "batch",
    )

    assert result["job_count"] == 1
    assert result["quantitative_verification_series_count"] == 1
    assert result["batch_builder_version"] == "pbpk-windows-snapshot-batch-1.2.0"
    assert Path(result["archive_path"]).is_file()


def test_windows_batch_runner_separates_negative_labels_from_runner_failures() -> None:
    script = (
        Path(__file__).parents[2]
        / "scripts"
        / "windows"
        / "run_snapshot_protocol_batch.ps1"
    ).read_text(encoding="utf-8")

    assert '"executed_negative"' in script
    assert '"runner_failed"' in script
    assert "negative_count" in script
    assert "runner_failed_count" in script
    assert "& powershell.exe @RunnerArguments" in script
    assert "$Summary.runner_failed_count" in script


def test_windows_runners_read_json_with_explicit_utf8_and_lock_worker() -> None:
    scripts = Path(__file__).parents[2] / "scripts" / "windows"
    worker = (scripts / "run_licensed_snapshot_reuse.ps1").read_text(
        encoding="utf-8"
    )
    batch = (scripts / "run_snapshot_protocol_batch.ps1").read_text(
        encoding="utf-8"
    )

    assert "Get-Content -LiteralPath $ProtocolPath -Raw -Encoding UTF8" in worker
    assert 'manifest_schema_version = "1.1.0"' in worker
    assert "worker_script_sha256 = $WorkerScriptSha256" in worker
    assert "Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8" in batch
    assert "Get-Content -LiteralPath $JobsPath -Encoding UTF8" in batch
    assert "$ExistingWorkerSha256 -ne $RunnerSha256" in batch
