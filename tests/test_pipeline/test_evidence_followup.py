import copy
from datetime import datetime, timedelta, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/deployment"))
spec = importlib.util.spec_from_file_location("evidence_followup_tests", ROOT / "scripts/deployment/follow_evidence_expansion.py")
follow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow)


@pytest.fixture
def healthy_orphan(tmp_path, monkeypatch):
    c = follow.controller
    deployment = tmp_path / "deployment"
    directory = deployment / "followup"
    directory.mkdir(parents=True)
    monkeypatch.setattr(follow, "DIRECTORY", directory)
    monkeypatch.setattr(follow, "STATE", directory / "status.json")
    monkeypatch.setattr(follow, "PAUSE", directory / "manual_pause")
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "DEPLOY", deployment)
    monkeypatch.setattr(c, "STATE", deployment / "status.json")
    run = "run_test"
    target = {"pid": 11, "start_ticks": "1", "state": "R"}
    monitor_identity = {"pid": 12, "start_ticks": "2", "state": "S"}
    old = {"pid": 10, "start_ticks": "0", "state": "R"}
    state = {"status": "running", "controller": old,
             "current_batch": {"run_id": run, "batch_id": "batch_02", "status": "running_full",
                "compounds": ["sildenafil"], "pipeline_pid": 11, "pipeline_identity": target,
                "watchdog_pid": 12, "log": "deployment/batch.log"}}
    manifest = {"run_id": run, "compounds": ["sildenafil"], "status": "running", "current_stage": "acquisition"}
    snapshot = {"batch_bytes": 100, "files": 2, "free_bytes": 100 * c.watchdog.GIB, "pipeline_log_bytes": 0}
    watched = {"status": "monitoring", "reasons": [], "checked_at": follow.now(), "target": target,
               "monitor_pid": 12, "snapshot": snapshot}
    mon_path = tmp_path / f"runtime/deployment/evidence_monitor/{run}/status.json"
    manifest_path = tmp_path / f"data_process/pipeline_runs/{run}.json"
    c.write(c.STATE, state)
    c.write(manifest_path, manifest)
    c.write(mon_path, watched)
    writer = SimpleNamespace(pid=11, poll=lambda: None)
    monitor = SimpleNamespace(pid=12, identity=monitor_identity, script=tmp_path / "watchdog.py", poll=lambda: None)
    adoptions = []

    def adopt(batch, path):
        adoptions.append((batch, path))
        return writer, monitor, target

    monkeypatch.setattr(c, "adopt_children", adopt)
    monkeypatch.setattr(c.watchdog, "same_process", lambda expected: target)
    pauses = []
    monkeypatch.setattr(c.watchdog, "pause_target", lambda expected: pauses.append(expected) or True)
    monkeypatch.setattr(follow, "registered_status", lambda identity, script: "absent" if identity["pid"] == 10 else "healthy")
    monkeypatch.setattr(c, "controller_health", lambda state: False)
    launches = []

    def spawn(script, args, log):
        launches.append((script, args, log))
        return SimpleNamespace(pid=13, poll=lambda: None)

    monkeypatch.setattr(c, "spawn", spawn)
    monkeypatch.setattr(c, "process_identity", lambda pid, script: {"pid": pid, "start_ticks": "3", "state": "R"})
    return SimpleNamespace(state=state, manifest=manifest, manifest_path=manifest_path, watched=watched,
        monitor_path=mon_path, target=target, launches=launches, adoptions=adoptions,
        saved={"restart_counts": {}, "history": []}, pauses=pauses)


def test_healthy_registered_children_allow_only_controller_adoption(healthy_orphan):
    case = healthy_orphan
    result = follow.check_once(case.saved)
    assert result["reason"] == "replacement_controller_launched_for_existing_children"
    assert len(case.launches) == 1 and case.launches[0][1] == ["resume-controller"]
    assert len(case.adoptions) == 2
    assert case.saved["restart_counts"] == {"run_test": 1}
    assert case.saved["pending_controller"]["identity"]["pid"] == 13
    persisted = json.loads(follow.STATE.read_text())
    assert persisted["restart_counts"]["run_test"] == 1  # Written before spawn.


@pytest.mark.parametrize("status", ["paused", "stopped_for_review", "failed", "unknown"])
def test_paused_or_unknown_controller_state_cannot_restart(healthy_orphan, status):
    case = healthy_orphan
    case.state["status"] = status
    follow.controller.write(follow.controller.STATE, case.state)
    assert follow.check_once(case.saved)["status"] == "needs_attention"
    assert case.launches == []


def test_paused_writer_never_resumed(healthy_orphan):
    case = healthy_orphan
    case.target["state"] = "T"
    assert follow.check_once(case.saved)["reason"] == "writer_not_healthy_no_automatic_resume"
    assert case.launches == []


@pytest.mark.parametrize("kind", ["pause", "llm", "capacity", "stale"])
def test_watchdog_pause_failure_capacity_and_staleness_block_adoption(healthy_orphan, kind):
    case = healthy_orphan
    if kind == "pause":
        case.watched["status"] = "paused_by_monitor"
    elif kind == "llm":
        case.watched["reasons"] = ["llm_request_failure_strict_stop"]
    elif kind == "capacity":
        case.watched["snapshot"]["free_bytes"] = 0
    else:
        case.watched["checked_at"] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    follow.controller.write(case.monitor_path, case.watched)
    assert follow.check_once(case.saved)["status"] == "needs_attention"
    assert case.launches == []


def test_persistent_budget_prevents_fourth_adoption(healthy_orphan):
    case = healthy_orphan
    case.saved["restart_counts"]["run_test"] = 3
    follow.save(case.saved)
    reloaded = follow.controller.read(follow.STATE)
    assert follow.check_once(reloaded)["reason"] == "controller_restart_budget_exhausted"
    assert case.launches == []


def test_live_pending_controller_does_not_spawn_again(healthy_orphan):
    case = healthy_orphan
    follow.check_once(case.saved)
    assert follow.check_once(case.saved)["reason"] == "replacement_controller_starting"
    assert len(case.launches) == 1


def test_pending_controller_start_timeout_requires_attention(healthy_orphan):
    case = healthy_orphan
    follow.check_once(case.saved)
    case.saved["pending_controller"]["launched_at"] = (datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat()
    assert follow.check_once(case.saved)["reason"] == "replacement_controller_start_timeout"
    assert len(case.launches) == 1


def test_unconfirmed_launch_requires_review_instead_of_retry_loop(healthy_orphan):
    case = healthy_orphan
    case.saved["pending_controller"] = {"identity": None, "run_id": "run_test"}
    assert follow.check_once(case.saved)["status"] == "needs_attention"
    assert case.launches == []


def test_existing_controller_lock_prevents_competing_launch(healthy_orphan):
    case = healthy_orphan
    with (follow.controller.DEPLOY / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert follow.check_once(case.saved)["reason"] == "controller_lock_busy_no_restart"
    assert case.launches == []


def test_registration_change_between_reads_blocks_launch(healthy_orphan, monkeypatch):
    case = healthy_orphan
    original = follow.controller.read
    calls = []

    def read(path):
        value = original(path)
        if path == follow.controller.STATE:
            calls.append(path)
            if len(calls) > 1:
                value["current_batch"]["pipeline_identity"]["start_ticks"] = "changed"
        return value

    monkeypatch.setattr(follow.controller, "read", read)
    assert follow.check_once(case.saved)["reason"] == "registration_changed_before_restart"
    assert case.launches == []


def test_manual_followup_pause_blocks_recovery(healthy_orphan):
    follow.PAUSE.touch()
    assert follow.check_once(healthy_orphan.saved)["reason"] == "followup_manually_paused"
    assert healthy_orphan.launches == []


def length_response(key="request"):
    return {"request_id": key, "prompt_template_version": "pilot", "status": "failed_local_llm",
            "failure_reason": "Qwen generation did not finish: length", "failed_raw_output_truncated": False,
            "failed_raw_output": json.dumps({"choices": [{"finish_reason": "length"}]})}


def test_quality_recovery_requires_complete_length_receipt_and_latest_failure():
    row = length_response()
    assert follow.recoverable_length_failures([row], "pilot") == ["request"]
    assert follow.recoverable_length_failures([row, {**row, "status": "ok"}], "pilot") == []
    assert follow.recoverable_length_failures([row], "different_prompt") == []
    assert follow.recoverable_length_failures([length_response(str(i)) for i in range(10)], "pilot") == []


@pytest.mark.parametrize("change", [
    {"failure_reason": "HTTP 503"}, {"failed_raw_output_truncated": True},
    {"status": "invalid_schema"}, {"failed_raw_output": "partial JSON"},
    {"failed_raw_output": '{"choices": [{"finish_reason": "stop"}]}'},
])
def test_quality_recovery_never_masks_other_failures(change):
    assert follow.recoverable_length_failures([{**length_response(), **change}], "pilot") == []


def test_quality_pause_budget_and_stage_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(follow, "QUALITY_DIRECTORY", tmp_path)
    ctl = SimpleNamespace(RUNS=("run",), same_process=lambda p: False,
        paths=lambda r: {"raw": tmp_path, "state": tmp_path / "state.json"},
        read_json=lambda p: {"stages": {"llm_extract": {"status": "failed", "failures": 1}}},
        verify_originals=lambda *a, **k: {"accepted": True},
        validate_pilot=lambda p: {"prompt": "pilot", "path": "pilot.json"}, rows=lambda p: [length_response()])
    state = {"status": "failed", "current_stage": "llm_extract", "current_run": "run",
             "runs": ["run"], "apply_validated": True}
    assert follow.assess_quality(state, {}, ctl)["status"] == "eligible"
    assert follow.assess_quality(state, {"recovery_attempts": {"run": 1}}, ctl)["status"] == "needs_attention"
    for changed in ({"status": "paused"}, {"current_stage": "llm_quality_validation"}, {"apply_validated": False}):
        assert follow.assess_quality({**state, **changed}, {}, ctl)["status"] == "needs_attention"
    (tmp_path / "manual_pause").touch()
    assert follow.assess_quality(state, {}, ctl)["reason"] == "quality_followup_manually_paused"


def test_adoption_identity_failure_is_not_silently_ignored(healthy_orphan, monkeypatch):
    def invalid(*args):
        raise RuntimeError("Writer PID was reused")
    monkeypatch.setattr(follow.controller, "adopt_children", invalid)
    monkeypatch.setattr(follow.controller.watchdog, "same_process", invalid)
    with pytest.raises(RuntimeError, match="reused"):
        follow.check_once(healthy_orphan.saved)
    assert healthy_orphan.launches == []
    assert healthy_orphan.pauses == []


def test_dead_watchdog_and_missing_controller_pause_only_verified_writer(healthy_orphan, monkeypatch):
    case = healthy_orphan
    def missing(*args):
        raise RuntimeError("Adopted writer is active without its registered watchdog")
    monkeypatch.setattr(follow.controller, "adopt_children", missing)
    outcome = follow.check_once(case.saved)
    assert outcome["status"] == "needs_attention" and outcome["writer_paused"] is True
    assert case.pauses == [case.target]
    assert case.launches == []


def test_stale_watchdog_pauses_orphan_writer_without_restart(healthy_orphan):
    case = healthy_orphan
    case.watched["checked_at"] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    follow.controller.write(case.monitor_path, case.watched)
    assert follow.check_once(case.saved)["writer_paused"] is True
    assert case.pauses == [case.target] and case.launches == []


def test_status_history_is_bounded_and_unchanged_checks_do_not_append(healthy_orphan):
    saved = healthy_orphan.saved
    saved.update(status="monitoring", reason="healthy")
    follow.save(saved)
    first = copy.deepcopy(saved["history"])
    follow.save(saved)
    assert saved["history"] == first
    saved["last_audit_at"] = (datetime.now(timezone.utc) - timedelta(minutes=21)).isoformat()
    follow.save(saved)
    assert len(saved["history"]) == 2 and saved["history"][-1]["audit"] is True
    for i in range(150):
        saved["reason"] = str(i)
        follow.save(saved)
    assert len(saved["history"]) == 144


def test_duplicate_guardian_is_rejected_before_inspection(healthy_orphan):
    with (follow.DIRECTORY / "guardian.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            follow.run()
    assert healthy_orphan.launches == []
