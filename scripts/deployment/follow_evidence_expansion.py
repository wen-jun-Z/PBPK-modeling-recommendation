"""Bounded follow-up for registered project evidence jobs; never unpause a writer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import expand_knowledge_base as controller

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/deployment/follow_evidence_expansion.py"
DIRECTORY = ROOT / "runtime/deployment/evidence_expansion_20261009/followup"
STATE = DIRECTORY / "status.json"
PAUSE = DIRECTORY / "manual_pause"
CHECK_SECONDS = 60
AUDIT_SECONDS = 1200
MAX_RESTARTS = 3
MAX_MONITOR_AGE = 180
MAX_CONTROLLER_START_SECONDS = 120
QUALITY_DIRECTORY = ROOT / "runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/followup"
QUALITY_STATE = QUALITY_DIRECTORY / "status.json"


def recoverable_length_failures(responses, prompt):
    """Only complete transport receipts proving a length stop are eligible."""
    latest = {r["request_id"]: r for r in responses if r.get("prompt_template_version") == prompt}
    failed = [r for r in latest.values() if r.get("status") not in {"ok", "reused_same_target_input_hash"}]
    if not 1 <= len(failed) <= 9:
        return []
    for row in failed:
        if row.get("status") != "failed_local_llm" or row.get("failed_raw_output_truncated"):
            return []
        if row.get("failure_reason") != "Qwen generation did not finish: length":
            return []
        try:
            raw = json.loads(row["failed_raw_output"])
            if len(raw["choices"]) != 1 or raw["choices"][0]["finish_reason"] != "length":
                return []
        except (KeyError, TypeError, ValueError):
            return []
    return sorted(r["request_id"] for r in failed)


def assess_quality(state, saved, ctl):
    if (QUALITY_DIRECTORY / "manual_pause").exists():
        return result("needs_attention", "quality_followup_manually_paused")
    if state.get("status") == "completed_and_verified":
        return result("completed", "seven_repaired_batches_verified")
    if ctl.same_process(state.get("pid")):
        if registered_status(state["pid"], ctl.SCRIPT) != "healthy":
            return result("needs_attention", "quality_controller_paused_no_resume")
        if state.get("updated_at") and age(state["updated_at"]) > MAX_MONITOR_AGE:
            return result("needs_attention", "quality_controller_status_stale")
        return result("monitoring", "quality_controller_active", run_id=state.get("current_run"),
                      stage=state.get("current_stage"))
    run = state.get("current_run")
    if (state.get("status") != "failed" or state.get("current_stage") != "llm_extract"
            or run not in ctl.RUNS or tuple(state.get("runs", [])) != ctl.RUNS
            or state.get("apply_validated") is not True):
        return result("needs_attention", "pause_or_unrecognized_failure_no_restart", run_id=run)
    if saved.get("recovery_attempts", {}).get(run, 0) >= 1:
        return result("needs_attention", "one_bounded_recovery_budget_exhausted", run_id=run)
    selected = ctl.paths(run)
    stage = ctl.read_json(selected["state"]).get("stages", {}).get("llm_extract", {})
    if stage.get("status") != "failed" or int(stage.get("failures", 0)) >= 3:
        return result("needs_attention", "stage_requires_manual_review", run_id=run)
    pilot = ctl.validate_pilot(state.get("pilot", {}).get("path", ""))
    ctl.verify_originals(run, full=True)
    response = selected["raw"] / "05_llm_extraction/llm_extraction_responses_live.jsonl"
    failed = recoverable_length_failures(ctl.rows(response), pilot["prompt"])
    if not failed:
        return result("needs_attention", "failure_is_not_verified_generation_length", run_id=run)
    return result("eligible", "verified_length_failure_only", run_id=run, failed_request_ids=failed,
                  pilot_receipt=pilot["path"])


def save_quality(saved, *, force=False):
    event = {k: saved.get(k) for k in ("status", "reason", "run_id", "stage")}
    due = not saved.get("last_audit_at") or age(saved["last_audit_at"]) >= AUDIT_SECONDS
    if force or due or event != saved.get("last_event"):
        saved["history"] = [*saved.get("history", []), {"checked_at": now(), **event}][-144:]
        saved["last_event"] = event
    if due:
        saved["last_audit_at"] = now()
    saved["checked_at"] = now()
    controller.write(QUALITY_STATE, saved)


def quality_run():
    """Watch existing work; one source-validated length recovery per future batch.

    User pauses, service faults, changed identities, disk limits and failed
    recovery remain stopped. No counters are cleared and no SIGCONT is sent.
    """
    import repair_expansion_quality as ctl
    QUALITY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (QUALITY_DIRECTORY / "guardian.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = controller.read(QUALITY_STATE) if QUALITY_STATE.exists() else {"recovery_attempts": {}, "history": []}
        saved["guardian"] = controller.process_identity(os.getpid(), SCRIPT)
        child = None
        recovery_script = ROOT / "scripts/deployment/recover_bounded_evidence.py"
        try:
            # An orphan/unknown previous attempt is not retried after a crash.
            if saved.get("recovery_worker"):
                raise RuntimeError("Previous recovery worker requires ownership review")
            while True:
                state = ctl.read_json(ctl.BASE / "controller_status.json")
                assessment = assess_quality(state, saved, ctl)
                if assessment["status"] == "eligible":
                    run = assessment["run_id"]
                    output = ctl.BASE / "automatic_bounded_recovery" / run
                    saved.setdefault("recovery_attempts", {})[run] = 1
                    saved.update(result("recovering", "bounded_source_only_recovery", run_id=run))
                    save_quality(saved, force=True)  # persist budget before launch
                    output.mkdir(parents=True, exist_ok=True)
                    logfile = output / "worker.log"
                    if logfile.exists() and logfile.stat().st_size > 64 * ctl.MIB:
                        raise RuntimeError("Recovery log exceeds safety limit")
                    command = [sys.executable, "-B", str(recovery_script), "--run-id", run,
                               "--output-dir", str(output), "--resume-controller",
                               "--pilot-receipt", assessment["pilot_receipt"]]
                    with logfile.open("ab") as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=ctl.runtime_environment(),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                            preexec_fn=ctl.parent_death_signal(os.getpid()))
                    identity = None
                    for _ in range(50):
                        try:
                            identity = controller.process_identity(child.pid, recovery_script)
                            break
                        except (OSError, RuntimeError):
                            if child.poll() is not None:
                                break
                            time.sleep(.02)
                    if identity is None:
                        if child.poll() is None:
                            child.terminate()
                        raise RuntimeError("Recovery process ownership could not be verified")
                    saved["recovery_worker"] = identity
                    controller.write(output / "worker.pid.json", identity)
                    save_quality(saved)
                    started = last_progress = time.monotonic()
                    signature = None
                    while child.poll() is None:
                        if registered_status(identity, recovery_script) != "healthy":
                            raise RuntimeError("Recovery process was paused or changed")
                        status = ctl.read_json(output / "status.json")
                        current = (status.get("done"), status.get("failed"), status.get("status"))
                        if current != signature:
                            signature, last_progress = current, time.monotonic()
                        size, maxlog = ctl.owned_bytes(output)
                        if (shutil.disk_usage(output).free < 12 * ctl.GIB or size > 64 * ctl.MIB
                                or maxlog > 64 * ctl.MIB or time.monotonic()-started > 4*3600
                                or time.monotonic()-last_progress > 40*60
                                or (QUALITY_DIRECTORY / "manual_pause").exists()):
                            raise RuntimeError("Bounded recovery safety guard stopped the worker")
                        saved["recovery_progress"] = current
                        save_quality(saved)
                        time.sleep(30)
                    if child.returncode or ctl.read_json(output / "status.json").get("status") != "completed":
                        raise RuntimeError("Bounded recovery failed; preserved for review")
                    saved.pop("recovery_worker", None)
                    child = None
                    saved.update(result("monitoring", "recovered_and_resumed_without_resetting_counters", run_id=run))
                    save_quality(saved, force=True)
                else:
                    saved.update(assessment)
                    save_quality(saved)
                    if assessment["status"] in {"completed", "needs_attention"}:
                        break
                time.sleep(CHECK_SECONDS)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                # Signal only our still-owned child, never an enumerated process.
                if registered_status(saved.get("recovery_worker"), recovery_script) == "healthy":
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
            saved.update(result("needs_attention", type(exc).__name__ + ": " + str(exc)[:250]))
            save_quality(saved, force=True)
            raise


def now():
    return datetime.now(timezone.utc).isoformat()


def age(timestamp):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()


def registered_status(expected, script):
    """Inspect only a registered UID/cwd/script/start identity, including its exit."""
    if not expected or type(expected.get("pid")) is not int or not expected.get("start_ticks"):
        raise RuntimeError("Missing registered process identity")
    process = controller.AdoptedProcess(expected, script, lambda: 0)
    if process.poll() is not None:
        return "absent"
    current = controller.process_identity(expected["pid"], script)
    if not controller.same_identity(current, expected):
        raise RuntimeError("Registered PID identity changed")
    return "paused" if current["state"] in {"T", "t"} else "healthy"


def result(status, reason, **fields):
    return {"status": status, "reason": reason, **fields}


def pause_unmonitored_writer(state):
    """Fail closed only for a proven orphan; never send a continue signal."""
    if registered_status(state.get("controller"), controller.SCRIPT) != "absent":
        return False
    batch = state.get("current_batch") or {}
    target = batch.get("pipeline_identity")
    if not target or target.get("pid") != batch.get("pipeline_pid"):
        raise RuntimeError("Cannot guard an unregistered writer")
    # The watchdog verifies UID, cwd, script, start ticks and the writer's private
    # process group/session, immediately before pausing that exact group.
    current = controller.watchdog.same_process(target)
    if current is None or current.get("state") in {"T", "t", "Z"}:
        return False
    return controller.watchdog.pause_target(target)


def assess(state):
    if PAUSE.exists():
        return result("needs_attention", "followup_manually_paused")
    if state.get("status") in {"completed", "completed_with_coverage_gaps"}:
        return result(state["status"], "expansion_finished")
    if state.get("status") != "running" or state.get("error"):
        return result("needs_attention", "expansion_not_running_or_requires_review")
    batch = state.get("current_batch")
    controller_state = registered_status(state.get("controller"), controller.SCRIPT)
    if controller_state == "paused":
        return result("needs_attention", "controller_paused_no_automatic_resume")
    if not batch:
        return result("monitoring", "controller_between_batches") if controller_state == "healthy" else result(
            "needs_attention", "controller_absent_without_registered_active_batch")
    run = batch.get("run_id", "")
    manifest_path = controller.project_path(f"data_process/pipeline_runs/{run}.json")
    manifest = controller.read(manifest_path)
    if (batch.get("status") not in {"running_pilot", "running_full"}
            or manifest.get("run_id") != run or manifest.get("compounds") != batch.get("compounds")
            or manifest.get("status") != "running"):
        # Normal stage handovers can also appear here; never intervene in them.
        return result("needs_attention", "batch_transition_or_nonrunning_manifest", run_id=run)
    try:
        writer, monitor, target = controller.adopt_children(batch, manifest_path)
    except (RuntimeError, FileNotFoundError, PermissionError):
        paused = pause_unmonitored_writer(state)
        return result("needs_attention", "child_adoption_failed_requires_review", run_id=run, writer_paused=paused)
    current = controller.watchdog.same_process(target)
    if writer.poll() is not None or current is None or current.get("state") in {"T", "t", "Z"}:
        return result("needs_attention", "writer_not_healthy_no_automatic_resume", run_id=run)
    if monitor is None or registered_status(monitor.identity, monitor.script) != "healthy":
        return result("needs_attention", "registered_watchdog_not_healthy", run_id=run,
                      writer_paused=pause_unmonitored_writer(state))
    monitor_path = controller.project_path(f"runtime/deployment/evidence_monitor/{run}/status.json")
    watched = controller.read(monitor_path)
    if (watched.get("status") != "monitoring" or watched.get("reasons")
            or not controller.same_identity(watched.get("target"), target)
            or watched.get("monitor_pid") != monitor.pid):
        return result("needs_attention", "watchdog_reported_pause_or_anomaly", run_id=run,
                      writer_paused=pause_unmonitored_writer(state))
    if not -5 <= age(watched.get("checked_at", "")) <= MAX_MONITOR_AGE:
        return result("needs_attention", "watchdog_audit_stale", run_id=run,
                      writer_paused=pause_unmonitored_writer(state))
    snapshot = watched.get("snapshot") or {}
    limits = {**controller.watchdog.DEFAULT_LIMITS, **watched.get("limits", {})}
    if controller.watchdog.capacity_reasons(snapshot, snapshot["batch_bytes"], limits):
        return result("needs_attention", "capacity_check_failed", run_id=run,
                      writer_paused=pause_unmonitored_writer(state))
    if controller_state == "healthy":
        return result("monitoring", "registered_controller_writer_and_watchdog_healthy", run_id=run,
                      stage=manifest.get("current_stage"))
    return result("eligible", "controller_absent_with_healthy_registered_children", run_id=run,
                  stage=manifest.get("current_stage"))


def fingerprint(state):
    batch = state.get("current_batch") or {}
    return json.dumps({"status": state.get("status"), "controller": state.get("controller"),
        "batch": {key: batch.get(key) for key in (
            "batch_id", "run_id", "compounds", "status", "pipeline_pid", "pipeline_identity", "watchdog_pid", "log")}},
        sort_keys=True)


def save(state):
    checked = now()
    prior = state.get("last_event")
    event = {key: state.get(key) for key in ("status", "reason", "run_id", "stage")}
    due = not state.get("last_audit_at") or age(state["last_audit_at"]) >= AUDIT_SECONDS
    if due or event != prior:
        state["history"] = [*state.get("history", []), {"checked_at": checked, "audit": due, **event}][-144:]
        state["last_event"] = event
    if due:
        state["last_audit_at"] = checked
    state["checked_at"] = checked
    controller.write(STATE, state)


def check_once(saved):
    state = controller.read(controller.STATE)
    assessment = assess(state)
    if assessment["status"] != "eligible":
        if assessment["status"] == "monitoring" and controller.controller_health(state):
            saved.pop("pending_controller", None)
        return assessment
    pending = saved.get("pending_controller")
    if pending:
        if not pending.get("identity"):
            return result("needs_attention", "unconfirmed_controller_launch_requires_review", run_id=assessment["run_id"])
        pending_health = registered_status(pending["identity"], controller.SCRIPT)
        if pending_health != "absent":
            if age(pending.get("launched_at", "")) > MAX_CONTROLLER_START_SECONDS:
                return result("needs_attention", "replacement_controller_start_timeout", run_id=assessment["run_id"])
            return result("monitoring" if pending_health == "healthy" else "needs_attention",
                          "replacement_controller_starting" if pending_health == "healthy" else "replacement_controller_paused",
                          run_id=assessment["run_id"])
    run = assessment["run_id"]
    count = saved.setdefault("restart_counts", {}).get(run, 0)
    if type(count) is not int or count < 0 or count >= MAX_RESTARTS:
        return result("needs_attention", "controller_restart_budget_exhausted", run_id=run)
    # Probe the controller's own lock and reread all registrations under it.
    # The existing controller reacquires this same lock at entry; a competing
    # launcher can therefore never create two controllers or two writers.
    with (controller.DEPLOY / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return result("needs_attention", "controller_lock_busy_no_restart", run_id=run)
        latest = controller.read(controller.STATE)
        if fingerprint(latest) != fingerprint(state):
            return result("needs_attention", "registration_changed_before_restart", run_id=run)
        checked = assess(latest)
        if checked["status"] != "eligible":
            return checked
    latest = controller.read(controller.STATE)
    if PAUSE.exists() or fingerprint(latest) != fingerprint(state):
        return result("needs_attention", "registration_or_pause_changed_before_launch", run_id=run)
    # Consume the persistent budget before spawning; a guardian crash cannot
    # reset it and silently turn a single recovery into an unbounded loop.
    saved["restart_counts"][run] = count + 1
    saved["pending_controller"] = {"run_id": run, "launched_at": now(), "identity": None}
    saved.update(result("monitoring", "launching_controller_adoption", run_id=run))
    save(saved)
    log = DIRECTORY / f"controller_resume_{run}_{count + 1}.log"
    process = controller.spawn(controller.SCRIPT, ["resume-controller"], log)
    for _ in range(20):
        if process.poll() is not None:
            return result("needs_attention", "replacement_controller_exited_during_launch", run_id=run)
        try:
            identity = controller.process_identity(process.pid, controller.SCRIPT)
        except (FileNotFoundError, ProcessLookupError, RuntimeError):
            time.sleep(.1)
            continue
        saved["pending_controller"]["identity"] = identity
        return result("monitoring", "replacement_controller_launched_for_existing_children", run_id=run)
    return result("needs_attention", "replacement_controller_identity_unconfirmed", run_id=run)


def run():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "guardian.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = controller.read(STATE) if STATE.exists() else {"restart_counts": {}, "history": []}
        saved["guardian"] = controller.process_identity(os.getpid(), SCRIPT)
        while True:
            try:
                outcome = check_once(saved)
            except Exception as exc:
                outcome = result("needs_attention", "followup_check_failed:" + type(exc).__name__)
            saved.update(outcome)
            save(saved)
            if outcome["status"] in {"completed", "completed_with_coverage_gaps"}:
                break
            time.sleep(CHECK_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status", "quality-run", "quality-start", "quality-status"])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "quality-run":
        quality_run()
    elif args.action == "quality-start":
        import repair_expansion_quality as ctl
        QUALITY_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with (QUALITY_DIRECTORY / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = controller.read(QUALITY_STATE).get("guardian") if QUALITY_STATE.exists() else None
            if previous and registered_status(previous, SCRIPT) != "absent":
                raise RuntimeError("Quality follow-up already has a registered live process")
            logfile = QUALITY_DIRECTORY / "guardian.log"
            if logfile.exists() and logfile.stat().st_size > 64 * ctl.MIB:
                raise RuntimeError("Guardian log exceeds limit")
            with logfile.open("ab") as log:
                child = subprocess.Popen([sys.executable, "-B", str(SCRIPT), "quality-run"], cwd=ROOT,
                    env=ctl.runtime_environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True)
            print(json.dumps({"guardian_pid": child.pid, "log": str(logfile)}))
    elif args.action == "quality-status":
        print(json.dumps(controller.read(QUALITY_STATE), ensure_ascii=False, indent=2) if QUALITY_STATE.exists()
              else "Quality follow-up has not started.")
    elif args.action == "run":
        run()
    elif STATE.exists():
        print(json.dumps(controller.read(STATE), ensure_ascii=False, indent=2))
    else:
        print("Follow-up guardian has not started.")


if __name__ == "__main__":
    main()
