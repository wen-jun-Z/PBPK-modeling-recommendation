"""Protect frozen evidence and bounded worker adoption during database expansion."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/deployment"))
spec = importlib.util.spec_from_file_location("expansion_controller_tests", ROOT / "scripts/deployment/expand_knowledge_base.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_process_identity_ignores_transient_running_state():
    left = {"pid": 123, "start_ticks": "1", "state": "R"}
    assert controller.same_identity(left, {**left, "state": "S"})
    assert not controller.same_identity(left, {**left, "start_ticks": "2"})


def test_status_does_not_claim_a_dead_or_reused_controller_is_running(monkeypatch):
    expected = {"pid": 123, "start_ticks": "1", "state": "R"}
    state = {"controller": expected}
    monkeypatch.setattr(controller, "process_identity", lambda *a: {**expected, "state": "S"})
    assert controller.controller_health(state)
    monkeypatch.setattr(controller, "process_identity", lambda *a: {**expected, "start_ticks": "2"})
    assert not controller.controller_health(state)
    def gone(*args):
        raise FileNotFoundError()
    monkeypatch.setattr(controller, "process_identity", gone)
    assert not controller.controller_health(state)


def test_stage_acceptance_requires_exact_predecessors_and_batch_identity():
    from pbpk_kb.pipeline_utils import PIPELINE_STAGES
    names = [stage.name for stage in PIPELINE_STAGES]
    batch = {"run_id": "run_20261009T000000Z", "compounds": ["one"]}
    manifest = {**batch, "status": "failed", "current_stage": "llm_extract", "stages": {
        name: {"status": "completed", "accepted": True} for name in names[:7]}}
    manifest["stages"]["llm_extract"] = {"status": "failed", "accepted": False}
    controller.validate_stages(manifest, batch, pilot=True)
    altered = copy.deepcopy(manifest)
    altered["stages"]["wrong_stage"] = altered["stages"].pop(names[0])
    with pytest.raises(RuntimeError, match="ordered"):
        controller.validate_stages(altered, batch, pilot=True)
    with pytest.raises(RuntimeError, match="different"):
        controller.validate_stages(manifest, {**batch, "compounds": ["other"]}, pilot=True)


def test_target_coverage_counts_aliases_once_and_requires_preserved_identity():
    baseline = {"evidence_coverage": [{"compound_id": "base"}]}
    targets = [
        {"compound_name": "new", "search_name": "new", "aliases": ["Alias"], "existing_compound_id": None},
        {"compound_name": "old", "search_name": "old", "aliases": [], "existing_compound_id": "expected"},
    ]
    result = controller.target_coverage([("n", "new", 1), ("n", "Alias", 2), ("wrong-id", "old", 4)], targets, baseline)
    assert result["covered_compounds"] == 2
    assert result["verified_new_target_count"] == 1
    assert result["missing_target_names"] == ["old"]
    assert result["all_targets_covered"] is False
    with pytest.raises(RuntimeError, match="share one"):
        controller.target_coverage([("base", "new", 1)], targets, baseline)


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    from pbpk_kb.processing import run_llm_extraction as extraction
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    run = "run_20261009T000000Z"
    directory = tmp_path / "data_process/raw_processing" / run
    layout = raw_processing_layout(directory)

    def write_rows(key, rows):
        path = layout.path(key, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    quote = "Midazolam clearance was 12 L/h in healthy adults."
    requests, responses, chunks = [], [], []
    for index in range(10):
        payload = {"target_compound": "midazolam", "input_text_or_table": quote}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        common = {"request_id": f"r{index}", "chunk_id": f"c{index}", "run_id": run,
                  "target_compound": "midazolam", "input_hash": digest, "model_name": "fixture-model",
                  "prompt_template_version": "fixture-prompt", "validator_version": extraction.VALIDATOR_VERSION}
        requests.append({**common, **payload, "input_items": [quote]})
        responses.append({**common, "status": "ok", "candidates": [], "rejected_candidates": []})
        chunks.append({"chunk_id": f"c{index}", "chunk_text": quote})
    statuses = [{"processing_status": "llm_processed"} for _ in range(10)]
    statuses.append({"processing_status": "pending_llm_request_limit"})
    write_rows("llm_extraction_requests_live.jsonl", requests)
    write_rows("llm_extraction_responses_live.jsonl", responses)
    write_rows("text_chunks.jsonl", chunks)
    write_rows("chunk_processing_status.jsonl", statuses)
    write_rows("processing_failures.jsonl", [])
    summary = {"stage": "llm_extract", "run_id": run, "output_dir": str(directory),
        "status": "in_progress", "accepted": False, "request_limit": 10, "api_request_count": 10,
        "api_status_counts": {"ok": 10}, "classify_only": False, "unresolved_api_failure_count": 0,
        "stop_reason": None, "chunks_seen": 11, "status_row_count": 11, "pending_count": 1,
        "status_path": str(layout.path("chunk_processing_status.jsonl")),
        "prompt_template_version": "fixture-prompt", "model_name": "fixture-model"}
    path = layout.path("llm_extract_summary.json", for_write=True)
    controller.write(path, summary)
    monkeypatch.setattr(extraction, "_load_upstream_index", lambda layout: object())
    monkeypatch.setattr(extraction, "_validate_candidates", lambda task, candidates, upstream: (candidates, []))
    return path, summary, requests, responses, statuses, write_rows


def test_ten_request_budget_is_the_only_automatic_resume(smoke):
    path, *_ = smoke
    assert controller.smoke_review(path)["passed"] is True


@pytest.mark.parametrize("change", ["api_error", "wrong_run", "wrong_hash", "missing_response", "duplicate_request", "other_pending", "status_outside_run", "unresolved_failure"])
def test_smoke_rejects_unexplained_or_misbound_output(smoke, change):
    path, summary, requests, responses, statuses, write_rows = smoke
    if change == "api_error":
        summary["api_status_counts"] = {"ok": 9, "failed_local_llm": 1}
    elif change == "wrong_run":
        responses[0]["run_id"] = "other"
    elif change == "wrong_hash":
        requests[0]["input_hash"] = responses[0]["input_hash"] = "same-but-not-derived-from-input"
    elif change == "missing_response":
        responses.pop()
    elif change == "duplicate_request":
        requests.append(requests[0])
    elif change == "other_pending":
        statuses[-1]["processing_status"] = "pending_failed_shared_input"
    elif change == "status_outside_run":
        summary["status_path"] = str(path.parent.parent.parent / "other/status.jsonl")
    elif change == "unresolved_failure":
        write_rows("processing_failures.jsonl", [{"stage": "llm_extract", "resolved": False}])
    write_rows("llm_extraction_requests_live.jsonl", requests)
    write_rows("llm_extraction_responses_live.jsonl", responses)
    write_rows("chunk_processing_status.jsonl", statuses)
    controller.write(path, summary)
    with pytest.raises(RuntimeError):
        controller.smoke_review(path)


def test_smoke_compares_candidate_content_not_just_counts(smoke, monkeypatch):
    from pbpk_kb.processing import run_llm_extraction as extraction
    path, _, requests, responses, _, write_rows = smoke
    responses[0]["candidates"] = [{"source_text": requests[0]["input_items"][0], "parameter": "clearance", "value": 12, "unit": "L/h"}]
    write_rows("llm_extraction_responses_live.jsonl", responses)
    monkeypatch.setattr(extraction, "_validate_candidates", lambda task, candidates, upstream:
        ([{**candidate, "value": 13} for candidate in candidates], []))
    with pytest.raises(RuntimeError, match="changed the validator"):
        controller.smoke_review(path)


def test_adoption_attaches_only_existing_project_children_and_detects_pid_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    run = "run_20261009T000000Z"
    script = tmp_path / "scripts/deployment/run_evidence_pipeline.py"
    monitor_script = tmp_path / "scripts/deployment/evidence_watchdog.py"
    raw = tmp_path / f"artifacts/pbpk_raw_runs/compound_runs/{run}"
    manifest = tmp_path / f"data_process/pipeline_runs/{run}.json"
    writer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(script),
        "--raw-run-root", str(raw), "--compounds", "one"], cwd=tmp_path, start_new_session=True)
    monitor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(monitor_script),
        "--manifest", str(manifest), "--pid", str(writer.pid)], cwd=tmp_path, start_new_session=True)
    try:
        target = controller.process_identity(writer.pid, script)
        batch = {"run_id": run, "compounds": ["one"], "pipeline_pid": writer.pid,
                 "pipeline_identity": target, "watchdog_pid": monitor.pid}
        controller.write(manifest, {"status": "completed"})
        controller.write(tmp_path / f"runtime/deployment/evidence_monitor/{run}/status.json",
            {"target": {**target, "state": "S"}, "monitor_pid": monitor.pid})
        monkeypatch.setattr(controller, "spawn", lambda *a, **k: pytest.fail("Adoption must not spawn a replacement"))
        adopted, watched, saved = controller.adopt_children(batch, manifest)
        assert adopted.pid == writer.pid and watched.pid == monitor.pid
        assert adopted.poll() is None and watched.poll() is None
        reused = controller.AdoptedProcess({**target, "start_ticks": "wrong"}, script, lambda: 0)
        with pytest.raises(RuntimeError, match="reused"):
            reused.poll()
        assert writer.poll() is None
        writer.terminate()
        writer.wait(timeout=5)
        assert adopted.wait(timeout=1) == 0
    finally:
        for process in (writer, monitor):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("exit_during_identity_read", [False, True])
def test_adopted_poll_accepts_only_same_uid_start_ticks_zombie(tmp_path, monkeypatch, exit_during_identity_read):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    script = tmp_path / "scripts/deployment/run_evidence_pipeline.py"
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(script)],
                               cwd=tmp_path, start_new_session=True)
    try:
        expected = controller.process_identity(process.pid, script)
        adopted = controller.AdoptedProcess(expected, script, lambda: 0)

        def leave_unreaped_zombie():
            process.terminate()
            # Wait for this disposable child without reaping it. Its cmdline is
            # now empty and cwd unavailable, reproducing the real exit state.
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)

        if exit_during_identity_read:
            def racing_identity(*args):
                leave_unreaped_zombie()
                raise RuntimeError("Registered process script changed")
            monkeypatch.setattr(controller, "process_identity", racing_identity)
        else:
            leave_unreaped_zombie()
            monkeypatch.setattr(controller, "process_identity",
                lambda *args: pytest.fail("An already verified zombie has no live cwd/cmdline to inspect"))
        assert adopted.poll() == 0
        # Reusing the same PID with a different saved start time must still fail,
        # even though the process is now a zombie and has no cmdline.
        reused = controller.AdoptedProcess({**expected, "start_ticks": "wrong"}, script, lambda: 0)
        with pytest.raises(RuntimeError, match="reused"):
            reused.poll()
    finally:
        process.wait(timeout=5)
