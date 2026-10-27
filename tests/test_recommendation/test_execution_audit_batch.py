from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.execution_audit_batch import (
    audit_snapshot_batch,
)


def test_execution_audit_batch_preserves_scientific_negative(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    protocol = batch / "protocols" / "key1" / "protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}\n", encoding="utf-8")
    jobs = [
        {
            "protocol_key": "key1",
            "protocol_id": "snapshot-reuse-" + "a" * 32,
            "protocol_sha256": sha256_file(protocol),
            "protocol_relative_path": "protocols/key1/protocol.json",
            "full_name": "OSP/Test-Model",
            "version_id": "OSP/Test-Model@commit",
            "repository_commit": "b" * 40,
            "compounds": ["Test drug"],
        }
    ]
    jobs_path = batch / "jobs.jsonl"
    jobs_path.write_text(json.dumps(jobs[0]) + "\n", encoding="utf-8")
    (batch / "batch_manifest.json").write_text(
        json.dumps({"job_count": 1, "jobs_sha256": sha256_file(jobs_path)}),
        encoding="utf-8",
    )
    result_root = tmp_path / "results" / "key1"
    result_root.mkdir(parents=True)
    worker = batch / "run_licensed_snapshot_reuse.ps1"
    worker.write_text("Write-Host test\n", encoding="utf-8")
    (result_root / "windows_run_manifest.json").write_text(
        json.dumps({"worker_script_sha256": sha256_file(worker)}) + "\n",
        encoding="utf-8",
    )

    def fake_auditor(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output_directory"]))
        output.mkdir(parents=True)
        report = output / "snapshot_reuse_audit.json"
        report.write_text("{}\n", encoding="utf-8")
        return {
            "scientific_accepted": False,
            "engineering_accepted": True,
            "validation_grade": "execution_verified",
            "run_label_grade": 1,
            "ranking_label": "abstain_validation_failed",
            "reusable_candidate": False,
            "report_sha256": sha256_file(report),
            "windows_manifest_sha256": "c" * 64,
            "counts": {"evaluated_datasets": 2, "dataset_errors": 0},
        }

    summary = audit_snapshot_batch(
        batch_directory=batch,
        windows_results_root=tmp_path / "results",
        output_directory=tmp_path / "audits",
        auditor=fake_auditor,
    )

    assert summary["counts"]["executed_negative_count"] == 1
    assert summary["counts"]["runner_failed_count"] == 0
    assert summary["counts"]["auditor_failed_count"] == 0
    row = json.loads(
        (tmp_path / "audits" / "audit_index.jsonl").read_text().strip()
    )
    assert row["audit_status"] == "audited_negative"
    assert row["engineering_accepted"] is True
    assert row["scientific_accepted"] is False


def test_execution_audit_batch_classifies_completed_engineering_negative(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    protocol = batch / "protocols" / "key1" / "protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}\n", encoding="utf-8")
    jobs = [
        {
            "protocol_key": "key1",
            "protocol_id": "snapshot-reuse-" + "a" * 32,
            "protocol_sha256": sha256_file(protocol),
            "protocol_relative_path": "protocols/key1/protocol.json",
            "full_name": "OSP/Test-Model",
            "version_id": "OSP/Test-Model@commit",
            "repository_commit": "b" * 40,
            "compounds": ["Test drug"],
        }
    ]
    jobs_path = batch / "jobs.jsonl"
    jobs_path.write_text(json.dumps(jobs[0]) + "\n", encoding="utf-8")
    (batch / "batch_manifest.json").write_text(
        json.dumps({"job_count": 1, "jobs_sha256": sha256_file(jobs_path)}),
        encoding="utf-8",
    )
    result_root = tmp_path / "results" / "key1"
    result_root.mkdir(parents=True)
    worker = batch / "run_licensed_snapshot_reuse.ps1"
    worker.write_text("Write-Host test\n", encoding="utf-8")
    (result_root / "windows_run_manifest.json").write_text(
        json.dumps({"worker_script_sha256": sha256_file(worker)}) + "\n",
        encoding="utf-8",
    )

    def fake_auditor(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output_directory"]))
        output.mkdir(parents=True)
        report = output / "snapshot_reuse_audit.json"
        report.write_text("{}\n", encoding="utf-8")
        return {
            "scientific_accepted": False,
            "engineering_accepted": False,
            "validation_grade": "execution_failed",
            "run_label_grade": 0,
            "ranking_label": "abstain_execution_failed",
            "reusable_candidate": False,
            "report_sha256": sha256_file(report),
            "windows_manifest_sha256": "c" * 64,
            "counts": {"evaluated_datasets": 0, "dataset_errors": 0},
        }

    summary = audit_snapshot_batch(
        batch_directory=batch,
        windows_results_root=tmp_path / "results",
        output_directory=tmp_path / "audits",
        auditor=fake_auditor,
    )
    row = json.loads(
        (tmp_path / "audits" / "audit_index.jsonl").read_text().strip()
    )

    assert summary["complete"] is True
    assert summary["counts"]["executed_negative_count"] == 1
    assert summary["counts"]["runner_failed_count"] == 0
    assert row["audit_status"] == "audited_negative"
    assert row["engineering_accepted"] is False
    assert row["run_label_grade"] == 0
    assert row["ranking_label"] == "abstain_execution_failed"


def test_execution_audit_batch_keeps_runner_failures_unlabeled() -> None:
    source = Path(
        "pbpk_kb/recommendation/model_matching/execution_audit_batch.py"
    ).read_text(encoding="utf-8")

    assert '"audit_status": "runner_failed"' in source
    assert '"run_label_grade": None' in source
