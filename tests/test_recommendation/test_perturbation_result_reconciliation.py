from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.perturbation_result_reconciliation import (
    assemble_reconciled_perturbation_results,
)


def _json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256_file(path)


def _batch(root: Path, jobs: list[dict[str, object]], worker: str) -> None:
    (root / "run_licensed_snapshot_reuse.ps1").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "run_licensed_snapshot_reuse.ps1").write_text(
        worker, encoding="utf-8"
    )
    jobs_sha = _jsonl(root / "jobs.jsonl", jobs)
    _json(
        root / "batch_manifest.json",
        {"job_count": len(jobs), "jobs_sha256": jobs_sha},
    )


def test_result_reconciler_selects_previous_and_correction_results(
    tmp_path: Path,
) -> None:
    jobs = [
        {"protocol_key": "old", "protocol_id": "pid-old", "protocol_sha256": "1" * 64},
        {"protocol_key": "new", "protocol_id": "pid-new", "protocol_sha256": "2" * 64},
    ]
    full_batch = tmp_path / "full_batch"
    correction_batch = tmp_path / "correction_batch"
    _batch(full_batch, jobs, "Write-Host locked\n")
    _batch(correction_batch, [jobs[1]], "Write-Host locked\n")
    worker_sha = sha256_file(full_batch / "run_licensed_snapshot_reuse.ps1")

    decisions = [
        {
            "protocol_key": "old",
            "decision": "reuse_previous",
            "selected_protocol_id": "pid-old",
            "selected_protocol_sha256": "1" * 64,
        },
        {
            "protocol_key": "new",
            "decision": "rerun_corrected",
            "selected_protocol_id": "pid-new",
            "selected_protocol_sha256": "2" * 64,
        },
    ]
    reconciliation = tmp_path / "reconciliation"
    decisions_sha = _jsonl(reconciliation / "reconciliation.jsonl", decisions)
    _json(
        reconciliation / "manifest.json",
        {"output_checksums": {"reconciliation.jsonl": decisions_sha}},
    )
    previous_results = tmp_path / "previous_results"
    correction_results = tmp_path / "correction_results"
    for root, row in ((previous_results, jobs[0]), (correction_results, jobs[1])):
        _json(
            root / str(row["protocol_key"]) / "windows_run_manifest.json",
            {
                "protocol_id": row["protocol_id"],
                "protocol_sha256": row["protocol_sha256"],
                "worker_script_sha256": worker_sha,
                "status": "passed",
            },
        )

    output = tmp_path / "output"
    manifest = assemble_reconciled_perturbation_results(
        reconciliation_directory=reconciliation,
        full_batch_directory=full_batch,
        previous_results_root=previous_results,
        correction_batch_directory=correction_batch,
        correction_results_root=correction_results,
        output_directory=output,
    )

    assert manifest["counts"] == {
        "protocol_count": 2,
        "reused_previous_count": 1,
        "corrected_count": 1,
        "passed_count": 2,
    }
    assert (output / "old").resolve() == (previous_results / "old").resolve()
    assert (output / "new").resolve() == (correction_results / "new").resolve()
