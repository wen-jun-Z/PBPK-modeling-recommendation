"""Supervise the existing evidence pipeline for the frozen 30-to-100 drug plan.

Only this project's explicitly registered child processes are inspected/signalled.
Any unexplained failure pauses the writer and stops the queue; never retry in a loop.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from service_runner import environment
import evidence_watchdog as watchdog

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_process/knowledge_expansion_20261009"
DEPLOY = ROOT / "runtime/deployment/evidence_expansion_20261009"
PYTHON = ROOT / "runtime/app-env/bin/python"
SCRIPT = ROOT / "scripts/deployment/expand_knowledge_base.py"
STATE = DEPLOY / "status.json"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("Expansion input path escaped this project")
    return path


def same_identity(left, right):
    return bool(left and right and all(left.get(key) == right.get(key) for key in ("pid", "start_ticks")))


def controller_health(state):
    expected = state.get("controller")
    if not expected:
        return False
    try:
        current = process_identity(int(expected["pid"]), SCRIPT)
    except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError):
        return False
    return same_identity(current, expected) and current.get("state") not in {"Z", "T", "t"}


def validate_stages(manifest, batch, *, pilot=False):
    from pbpk_kb.pipeline_utils import PIPELINE_STAGES
    expected = [stage.name for stage in PIPELINE_STAGES]
    stages = manifest.get("stages", {})
    if manifest.get("run_id") != batch["run_id"] or manifest.get("compounds") != batch["compounds"]:
        raise RuntimeError("Pipeline manifest belongs to a different expansion batch")
    complete = expected[:7] if pilot else expected
    if not all(stages.get(name, {}).get("accepted") is True
               and stages[name].get("status") == "completed" for name in complete):
        raise RuntimeError("Required ordered pipeline stages were not accepted")
    if pilot:
        if (manifest.get("status") != "failed" or manifest.get("current_stage") != "llm_extract"
                or set(stages) != set(expected[:8])
                or stages["llm_extract"].get("accepted") is not False
                or stages["llm_extract"].get("status") != "failed"):
            raise RuntimeError("Pilot did not stop precisely at the LLM extraction stage")
    elif manifest.get("status") != "completed" or set(stages) != set(expected):
        raise RuntimeError("The eighteen-stage acceptance is incomplete")


def process_identity(pid, script):
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid() or (proc / "cwd").resolve() != ROOT:
        raise RuntimeError("Registered child no longer belongs to this project")
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    if str(script).encode() not in (proc / "cmdline").read_bytes().split(b"\0"):
        raise RuntimeError("Registered process script changed")
    return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}


class AdoptedProcess:
    """Wait on an explicitly verified project process without spawning or killing it."""
    def __init__(self, identity, script, exit_code):
        self.identity = identity
        self.pid = int(identity["pid"])
        self.script = script
        self.exit_code = exit_code
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        proc = Path(f"/proc/{self.pid}")
        try:
            if proc.stat().st_uid != os.getuid():
                raise RuntimeError("Adopted process owner changed")
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] != self.identity["start_ticks"]:
                raise RuntimeError("Adopted process PID was reused; no signal sent")
            if fields[0] != "Z":
                try:
                    if not same_identity(process_identity(self.pid, self.script), self.identity):
                        raise RuntimeError("Adopted process identity changed")
                    return None
                except RuntimeError:
                    # It may exit between the first stat read and cwd/cmdline
                    # verification. Only the same UID/start identity in Z state
                    # is a normal exit; every live mismatch still raises.
                    if proc.stat().st_uid != os.getuid():
                        raise
                    final = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                    if final[19] != self.identity["start_ticks"] or final[0] != "Z":
                        raise
        except FileNotFoundError:
            pass
        self.returncode = self.exit_code()
        return self.returncode

    def wait(self, timeout=None):
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(str(self.script), timeout)
            time.sleep(.2)
        return self.returncode


def adopt_children(batch, manifest):
    """Validate the saved writer and monitor, including their exact batch arguments."""
    target = batch.get("pipeline_identity")
    if not target or target.get("pid") != batch.get("pipeline_pid"):
        raise RuntimeError("Cannot adopt a pipeline without its saved start identity")
    writer_script = ROOT / "scripts/deployment/run_evidence_pipeline.py"
    writer = AdoptedProcess(target, writer_script,
        lambda: 0 if read(manifest).get("status") == "completed" else 1)
    active = writer.poll() is None
    if active:
        args = (Path(f"/proc/{writer.pid}") / "cmdline").read_bytes().split(b"\0")
        raw = str(ROOT / f"artifacts/pbpk_raw_runs/compound_runs/{batch['run_id']}").encode()
        if (b"--raw-run-root" not in args or args[args.index(b"--raw-run-root") + 1] != raw
                or b"--compounds" not in args
                or args[args.index(b"--compounds") + 1] != ",".join(batch["compounds"]).encode()):
            raise RuntimeError("Adopted writer arguments belong to another batch")
    monitor_pid = batch.get("watchdog_pid")
    monitor_path = ROOT / f"runtime/deployment/evidence_monitor/{batch['run_id']}/status.json"
    state = read(monitor_path)
    if state.get("monitor_pid") != monitor_pid or not same_identity(state.get("target"), target):
        raise RuntimeError("Cannot bind the recorded watchdog to the adopted writer")
    monitor_script = ROOT / "scripts/deployment/evidence_watchdog.py"
    try:
        monitor_identity = process_identity(monitor_pid, monitor_script)
    except FileNotFoundError:
        if active:
            raise RuntimeError("Adopted writer is active without its registered watchdog") from None
        # The finished watchdog has no live PID to signal; preserve its recorded
        # terminal audit, which phase() validates before allowing any next step.
        monitor = None
    else:
        args = (Path(f"/proc/{monitor_pid}") / "cmdline").read_bytes().split(b"\0")
        if (b"--manifest" not in args or args[args.index(b"--manifest") + 1] != str(manifest).encode()
                or b"--pid" not in args or args[args.index(b"--pid") + 1] != str(writer.pid).encode()):
            raise RuntimeError("Adopted watchdog arguments belong to another batch")
        monitor = AdoptedProcess(monitor_identity, monitor_script, lambda: 0)
    return writer, monitor, target


def spawn(script, arguments, log):
    env = environment()
    env.update(PBPK_EVIDENCE_LLM_ENABLED="true", PBPK_EVIDENCE_LLM_BACKEND="vllm",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with log.open("ab", buffering=0) as stream:
        return subprocess.Popen([str(PYTHON), str(script), *arguments], cwd=ROOT,
                                env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)


def target_coverage(rows, targets, baseline):
    """Count frozen target entities once; an alias never creates another drug."""
    baseline_ids = {str(row["compound_id"]) for row in baseline["evidence_coverage"]}
    seen_ids = set(baseline_ids)
    verified, missing = [], []
    for target in targets:
        expected_id = target.get("existing_compound_id")
        names = {str(value).strip().casefold() for value in
                 [target["compound_name"], target["search_name"], *target.get("aliases", [])] if value}
        matched = [(str(cid), name, int(count)) for cid, name, count in rows
                   if count and ((expected_id and str(cid) == expected_id)
                                 or (not expected_id and str(name).strip().casefold() in names))]
        ids = {cid for cid, _, _ in matched}
        if ids & seen_ids:
            raise RuntimeError("Two target entities would share one evidence compound identity")
        if matched:
            seen_ids.update(ids)
            verified.append({"compound_name": target["compound_name"], "compound_ids": sorted(ids),
                             "evidence_count": sum(count for _, _, count in matched),
                             "mapping": "preserved_existing_id" if expected_id else "explicit_target_name_or_alias"})
        else:
            missing.append(target["compound_name"])
    return {"covered_compounds": len(baseline_ids) + len(verified),
            "verified_new_target_count": len(verified), "verified_new_targets": verified,
            "missing_target_names": missing, "all_targets_covered": not missing and len(targets) == 70}


def database_check():
    """Read only the registered project DB; verify all pre-expansion evidence."""
    from dotenv import dotenv_values
    from urllib.parse import urlsplit, unquote
    import psycopg
    baseline = read(DATA / "baseline.json")
    pidfile = ROOT / "runtime/postgres-data/postmaster.pid"
    lines = pidfile.read_text().splitlines()
    proc = Path(f"/proc/{int(lines[0])}")
    if (pidfile.stat().st_uid != os.getuid() or proc.stat().st_uid != os.getuid()
            or Path(lines[1]).resolve() != ROOT / "runtime/postgres-data"
            or (proc / "cwd").resolve() != ROOT / "runtime/postgres-data"):
        raise RuntimeError("Project database identity verification failed")
    value = dotenv_values(ROOT / "runtime/deployment/app.env", interpolate=False)["DATABASE_URL"]
    url = urlsplit(value.replace("postgresql+psycopg://", "postgresql://", 1))
    if url.hostname != "127.0.0.1" or url.port != int(lines[3]) or url.path != "/pbpk_kb":
        raise RuntimeError("Unexpected project database address")
    with psycopg.connect(host=url.hostname, port=url.port, user=unquote(url.username),
                        password=unquote(url.password or ""), dbname="pbpk_kb", connect_timeout=5) as db:
        db.read_only = True
        with db.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout='30s'")
            cursor.execute("SELECT evidence_id::text, md5(to_jsonb(e)::text) FROM auto_validated_evidence e")
            actual = set(cursor.fetchall())
            expected = {(r["evidence_id"], r["row_fingerprint"]) for r in baseline["evidence_row_fingerprints"]}
            if expected - actual:
                raise RuntimeError("Pre-expansion evidence changed or disappeared; queue stopped")
            cursor.execute("SELECT attributed_compound_id::text, lower(attributed_compound_name), count(*) "
                           "FROM auto_validated_evidence GROUP BY 1, 2")
            coverage_rows = cursor.fetchall()
    coverage = {}
    for _, name, count in coverage_rows:
        coverage[name] = coverage.get(name, 0) + int(count)
    return {"checked_at": now(), "original_evidence_unchanged": True,
            "usable_evidence_rows": sum(coverage.values()), "evidence_by_compound": coverage,
            "database_distinct_attribution_names": len(coverage),
            "database_distinct_attributed_ids": len({row[0] for row in coverage_rows}),
            **target_coverage(coverage_rows, read(DATA / "targets.json")["targets"], baseline)}


def smoke_review(summary_path):
    """Audit all accepted AND rejected candidates without changing saved outputs."""
    from pbpk_kb.processing import run_llm_extraction as extraction
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    summary_path = project_path(summary_path)
    expected_dir = summary_path.parent.parent
    expected_run = expected_dir.name
    if (expected_dir.parent != ROOT / "data_process/raw_processing"
            or summary_path.parent.name != "00_run_manifest"
            or summary_path.name != "llm_extract_summary.json"):
        raise RuntimeError("Pilot summary is outside its expected processing layout")
    summary = read(summary_path)
    if (summary.get("run_id") != expected_run
            or project_path(summary["output_dir"]) != expected_dir):
        raise RuntimeError("Pilot summary run identity mismatch")
    if (summary.get("stage") != "llm_extract" or summary.get("status") != "in_progress"
            or summary.get("accepted") is not False
            or summary.get("request_limit") != 10 or summary.get("api_request_count") != 10
            or summary.get("api_status_counts") != {"ok": 10}
            or summary.get("classify_only") is not False
            or summary.get("unresolved_api_failure_count") != 0 or summary.get("stop_reason")
            or summary.get("chunks_seen") != summary.get("status_row_count")
            or summary.get("pending_count", 0) < 1):
        raise RuntimeError("Pilot did not stop solely at its ten-request budget")
    layout = raw_processing_layout(expected_dir)
    if project_path(summary["status_path"]) != layout.path("chunk_processing_status.jsonl").resolve():
        raise RuntimeError("Pilot status ledger is outside the selected run")
    rows = lambda key: list(extraction._iter_jsonl(layout.path(key)))
    statuses = list(extraction._iter_jsonl(Path(summary["status_path"])))
    pending = [r for r in statuses if str(r.get("processing_status", "")).startswith("pending_")]
    if (len(statuses) != summary["status_row_count"]
            or any(r.get("processing_status") == "llm_failed" for r in statuses)
            or len(pending) != summary["pending_count"]
            or any(r["processing_status"] != "pending_llm_request_limit" for r in pending)):
        raise RuntimeError("Pilot has unresolved work unrelated to the request limit")
    failures = rows("processing_failures.jsonl")
    if any(r.get("stage") == "llm_extract" and r.get("resolved") is not True for r in failures):
        raise RuntimeError("Pilot retains an unresolved extraction failure")
    request_rows = rows("llm_extraction_requests_live.jsonl")
    requests = {r["request_id"]: r for r in request_rows}
    responses = rows("llm_extraction_responses_live.jsonl")
    response_ids = [r["request_id"] for r in responses]
    if (len(request_rows) != len(requests) or len(response_ids) != len(set(response_ids))
            or set(response_ids) != set(requests)):
        raise RuntimeError("Pilot requests and responses are not one-to-one")
    live = [r for r in responses if r.get("status") != "reused_same_target_input_hash"]
    if len(live) != 10 or any(r.get("status") != "ok" for r in live):
        raise RuntimeError("Pilot contains failed or incomplete real model requests")
    audit = watchdog.evidence_audit(requests, responses)
    if audit["reasons"]:
        raise RuntimeError("Pilot evidence support audit failed")
    ids = {r["chunk_id"] for r in responses}
    chunks = {r["chunk_id"]: r for r in extraction._iter_jsonl(layout.path("text_chunks.jsonl")) if r["chunk_id"] in ids}
    upstream = extraction._load_upstream_index(layout)
    valid_count = rejected_count = 0
    for response in responses:
        request = requests[response["request_id"]]
        payload = {"target_compound": request["target_compound"],
                   "input_text_or_table": request["input_text_or_table"]}
        actual_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8", errors="replace")).hexdigest()
        if (request["input_hash"] != response["input_hash"] or request["input_hash"] != actual_hash
                or request.get("run_id") != expected_run or response.get("run_id") != expected_run
                or request.get("chunk_id") != response.get("chunk_id")
                or request.get("target_compound") != response.get("target_compound")
                or request.get("prompt_template_version") != response.get("prompt_template_version")
                or response.get("prompt_template_version") != summary.get("prompt_template_version")
                or response.get("model_name") != summary.get("model_name")
                or request.get("model_name") != summary.get("model_name")
                or response.get("validator_version") != extraction.VALIDATOR_VERSION):
            raise RuntimeError("Pilot input identity mismatch")
        task = extraction.PendingTask(chunk=chunks[response["chunk_id"]],
            target_compound=request["target_compound"], status_record={},
            request_record=request, input_hash=request["input_hash"])
        candidates = copy.deepcopy([*response.get("candidates", []),
                                    *[r["candidate"] for r in response.get("rejected_candidates", [])]])
        for candidate in candidates:
            candidate.pop("validation_repairs", None)
        valid, rejected = extraction._validate_candidates(task, candidates, upstream)
        without_repairs = lambda item: {key: value for key, value in item.items() if key != "validation_repairs"}
        normalize = lambda items: sorted(json.dumps(without_repairs(item), ensure_ascii=False, sort_keys=True) for item in items)
        if (normalize(valid) != normalize(response.get("candidates", []))
                or normalize([r["candidate"] for r in rejected])
                    != normalize([r["candidate"] for r in response.get("rejected_candidates", [])])):
            raise RuntimeError("Pilot candidate replay changed the validator decision")
        valid_count += len(valid)
        rejected_count += len(rejected)
    return {"checked_at": now(), "passed": True, "real_requests": 10,
            "accepted_candidates_checked": valid_count, "rejected_candidates_checked": rejected_count,
            "audit": audit, "scope": "engineering/source-support validation; not independent expert review"}


def phase(batch, *, full, adopt=False):
    run = batch["run_id"]
    manifest = ROOT / f"data_process/pipeline_runs/{run}.json"
    log = DEPLOY / f"{batch['batch_id']}_{'full' if full else 'pilot'}.log"
    raw = ROOT / f"artifacts/pbpk_raw_runs/compound_runs/{run}"
    arguments = ["--compounds", ",".join(batch["compounds"]), "--raw-run-root", str(raw),
                 "--defer-llm-check", "--request-concurrency", "2",
                 "--max-pmc-fulltext-per-compound", "100",
                 "--max-pmc-oa-packages-per-compound", "25"]
    if full:
        arguments.append("--full-extraction")
    elif not manifest.exists():
        arguments.append("--new-run")
    if adopt:
        child, monitor, target = adopt_children(batch, manifest)
        batch["adopted_at"] = now()
        log = project_path(batch["log"])
    else:
        child = spawn(ROOT / "scripts/deployment/run_evidence_pipeline.py", arguments, log)
        batch.update(status="running_full" if full else "running_pilot", pipeline_pid=child.pid,
                     log=str(log.relative_to(ROOT)), started_at=now())
        monitor = None
        target = None
    started = time.monotonic()
    try:
        while child.poll() is None:
            if target is None:
                target = watchdog.identity(child.pid)
                batch["pipeline_identity"] = target
            current = watchdog.same_process(target)
            if current and current["state"] in {"T", "t"}:
                mon_path = ROOT / f"runtime/deployment/evidence_monitor/{run}/status.json"
                mon = read(mon_path) if mon_path.exists() else {}
                # A watchdog can observe the deliberate budget exception during
                # interpreter teardown. Only a fresh full pilot audit permits
                # that exact paused process to exit normally.
                if (not full and same_identity(mon.get("target"), target)
                        and mon.get("reasons") == ["pipeline_stage_failed"]):
                    validate_stages(read(manifest), batch, pilot=True)
                    smoke_review(ROOT / f"data_process/raw_processing/{run}/00_run_manifest/llm_extract_summary.json")
                    watchdog.same_process(target)
                    os.killpg(child.pid, signal.SIGCONT)
                    child.wait(timeout=15)
                    break
                raise RuntimeError("Evidence watchdog paused this batch; review its reasons before resuming")
            if monitor is None and manifest.exists() and (raw / "requests.jsonl").exists():
                # Audit startup traffic too; the watchdog normally starts at its attachment offset.
                history, _ = watchdog.read_new_rows(raw / "requests.jsonl", 0)
                history = watchdog.classify_verified_source_gaps(raw, history, {})
                if watchdog.request_audit(history, watchdog.DEFAULT_LIMITS)["reasons"]:
                    raise RuntimeError("Acquisition failed before watchdog attachment")
                monitor = spawn(ROOT / "scripts/deployment/evidence_watchdog.py",
                    ["--manifest", str(manifest), "--pid", str(child.pid), "--log-path", str(log)],
                    DEPLOY / f"{batch['batch_id']}_watchdog.log")
                batch["watchdog_pid"] = monitor.pid
            if monitor is None and time.monotonic() - started > 120:
                raise RuntimeError("Cannot attach evidence watchdog within startup budget")
            if monitor is not None and monitor.poll() is not None and child.poll() is None:
                raise RuntimeError("Evidence watchdog exited while its writer remains active")
            state = read(STATE)
            state.update(updated_at=now(), current_batch=batch)
            write(STATE, state)
            time.sleep(3)
        child.wait()
        if monitor is not None or adopt:
            if monitor is not None:
                try:
                    monitor.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Watchdog did not finish its final audit")
            monitor_state = read(ROOT / f"runtime/deployment/evidence_monitor/{run}/status.json")
            expected_monitor_pid = monitor.pid if monitor else batch["watchdog_pid"]
            if not same_identity(monitor_state.get("target"), target) or monitor_state.get("monitor_pid") != expected_monitor_pid:
                raise RuntimeError("Final evidence monitor status belongs to a different process")
            if monitor_state.get("status") == "monitoring":
                raise RuntimeError("Watchdog has no terminal batch audit")
            unexplained = set(monitor_state.get("reasons", [])) - ({"pipeline_stage_failed"} if not full else set())
            if unexplained:
                raise RuntimeError("Final evidence monitor audit failed")
        elif child.returncode == 0:
            raise RuntimeError("Evidence pipeline ended without an attached watchdog")
        return child.returncode
    except BaseException:
        if target and child.poll() is None and watchdog.same_process(target):
            os.killpg(child.pid, signal.SIGSTOP)
        raise


def run(*, resume=False):
    DEPLOY.mkdir(parents=True, exist_ok=True)
    lock = (DEPLOY / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists() and not resume:
        raise RuntimeError("An expansion state already exists; inspect it before starting another controller")
    if resume and not STATE.exists():
        raise RuntimeError("No existing controller state to resume")
    plan = read(DATA / "targets.json")
    targets = plan["targets"]
    names = [t["compound_name"] for t in targets]
    if len(set(names)) != 70 or set(names) & set(plan["baseline_evidence_compound_names"]):
        raise RuntimeError("Expansion targets overlap or are incomplete")
    protocol = read(DATA / "protocol.json")
    frozen = {str((DATA / name).relative_to(ROOT)): digest(DATA / name) for name in ("baseline.json", "targets.json", "protocol.json")}
    for source in [*plan["sources"], *protocol["original_experiment_inputs"]]:
        if digest(project_path(source["path"])) != source["sha256"]:
            raise RuntimeError("A frozen selection input changed")
        frozen[source["path"]] = source["sha256"]
    previous = read(STATE) if resume else None
    if previous:
        for path, sha in previous["frozen_inputs"].items():
            if digest(project_path(path)) != sha:
                raise RuntimeError("Previously frozen controller input changed")
        if previous.get("status") in {"completed", "completed_with_coverage_gaps"}:
            raise RuntimeError("A completed expansion cannot be resumed")
    run_stamp = datetime.now(timezone.utc).replace(microsecond=0)
    batches = previous["planned_batches"] if previous else [{"batch_id": f"batch_{index:02}",
                "run_id": (run_stamp + timedelta(seconds=index - 1)).strftime("run_%Y%m%dT%H%M%SZ"),
                "compounds": [t["search_name"] for t in targets if t["batch_id"] == f"batch_{index:02}"]}
               for index in range(1, 8)]
    for batch in batches:
        expected_names = [t["search_name"] for t in targets if t["batch_id"] == batch["batch_id"]]
        if (len(batch["compounds"]) != 10 or len(set(batch["compounds"])) != 10
                or batch["compounds"] != expected_names):
            raise RuntimeError("Each expansion batch must contain ten distinct targets")
        if not resume and (ROOT / f"data_process/pipeline_runs/{batch['run_id']}.json").exists():
            raise RuntimeError("A preallocated expansion run already exists")
    state = previous or {"started_at": now(), "updated_at": now(), "status": "running", "target": 100,
             "controller": process_identity(os.getpid(), SCRIPT), "frozen_inputs": frozen,
             "database": database_check(), "planned_batches": batches, "batches": [], "current_batch": None}
    if resume:
        state.update(controller=process_identity(os.getpid(), SCRIPT), resumed_at=now(),
                     updated_at=now(), status="running", frozen_inputs=frozen, database=database_check())
        state.pop("error", None)
        state.pop("error_type", None)
    write(STATE, state)
    try:
        completed_ids = {b["batch_id"] for b in state["batches"]}
        active_batch = copy.deepcopy(state.get("current_batch")) if resume else None
        for batch in batches:
            if batch["batch_id"] in completed_ids:
                continue
            batch = dict(batch)
            batch_id = batch["batch_id"]
            adopting = bool(active_batch and active_batch["batch_id"] == batch_id)
            was_full = False
            if adopting:
                if (active_batch["run_id"] != batch["run_id"]
                        or active_batch["compounds"] != batch["compounds"]
                        or active_batch.get("status") not in {"running_pilot", "running_full"}):
                    raise RuntimeError("Active batch cannot be matched to the saved plan")
                batch = active_batch
                was_full = batch["status"] == "running_full"
            state["current_batch"] = batch
            write(STATE, state)
            rc = phase(batch, full=was_full, adopt=adopting)
            manifest = read(ROOT / f"data_process/pipeline_runs/{batch['run_id']}.json")
            if manifest.get("status") != "completed":
                if was_full:
                    raise RuntimeError("Adopted full evidence pipeline failed; review before restarting")
                if rc == 0 or manifest.get("current_stage") != "llm_extract":
                    raise RuntimeError("Pilot failed outside its expected LLM request limit")
                validate_stages(manifest, batch, pilot=True)
                summary = ROOT / f"data_process/raw_processing/{batch['run_id']}/00_run_manifest/llm_extract_summary.json"
                review = smoke_review(summary)
                write(DATA / f"{batch_id}_pilot_review.json", review)
                if phase(batch, full=True):
                    raise RuntimeError("Full evidence pipeline failed; no automatic retry")
                manifest = read(ROOT / f"data_process/pipeline_runs/{batch['run_id']}.json")
            validate_stages(manifest, batch)
            for path, sha in frozen.items():
                if digest(ROOT / path) != sha:
                    raise RuntimeError("A frozen expansion/experimental input changed")
            check = database_check()
            from verify_expansion_batch import verify_batch
            check["serving_acceptance"] = verify_batch(batch["run_id"])
            if check["serving_acceptance"].get("passed") is not True:
                raise RuntimeError("Batch evidence is not verified searchable; not marking complete")
            batch.update(status="completed", completed_at=now(), database=check)
            state = read(STATE)
            state["database"] = check
            state["batches"].append(batch)
            state.update(current_batch=None, updated_at=now())
            write(STATE, state)
            write(DATA / f"{batch_id}_database_verification.json", check)
        state.update(status="completed" if state["database"]["all_targets_covered"]
                     and state["database"]["covered_compounds"] >= 100 else "completed_with_coverage_gaps",
                     updated_at=now(), current_batch=None)
        write(STATE, state)
    except BaseException as exc:
        state = read(STATE)
        state.update(status="stopped_for_review", updated_at=now(), error_type=type(exc).__name__,
                     error=str(exc) if isinstance(exc, RuntimeError) else "Inspect project-local logs; queue stopped safely")
        write(STATE, state)
        raise


def followup_status():
    """Display the registered project guardian without enumerating processes."""
    path = DEPLOY / "followup/status.json"
    if not path.exists():
        return 'Disabled'
    try:
        saved = read(path)
        expected = saved.get("guardian") or {}
        actual = process_identity(int(expected["pid"]), ROOT / "scripts/deployment/follow_evidence_expansion.py")
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(saved["checked_at"])).total_seconds()
        if not same_identity(expected, actual) or actual["state"] in {"T", "t", "Z"} or age > 180:
            return 'Follow-up process inactive or stale'
        label = {"monitoring": 'Following up', "tracking": 'Following up', "needs_attention": 'Action required',
                 "completed": 'Completed', "restarting_controller": 'Reconnect supervisor'}.get(saved.get("status"), saved.get("status", 'Unknown'))
        reason = saved.get("reason")
        return label + (" · " + str(reason) if saved.get("status") == "needs_attention" and reason else "")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return 'Follow-up process absent or identity verification failed'


def status():
    if not STATE.exists():
        print('Expansion has not started')
        return
    state = read(STATE)
    complete = len(state["batches"])
    bar = "━" * complete + "░" * (7 - complete)
    db = state["database"]
    live = controller_health(state)
    current_status = state["status"]
    if current_status == "running" and not live:
        current_status = 'controller_not_active (state file retains previous progress)'
    print(f"Knowledge-base expansion [{bar}] {complete}/7 batches | Verified coverage {db['covered_compounds']}/100 drugs | {current_status}")
    print(f"Supervisor {'Running' if live else 'Absent or unverified'} | Last updated {state.get('updated_at', 'unknown')}")
    print('Background follow-up: ' + followup_status())
    batch = state.get("current_batch")
    if batch:
        path = ROOT / f"data_process/pipeline_runs/{batch['run_id']}.json"
        manifest = read(path) if path.exists() else {}
        stages = sum(s.get("accepted") is True for s in manifest.get("stages", {}).values())
        print(f"Current {batch['batch_id']} | Stage {stages}/18 | {manifest.get('current_stage', 'starting')}")
        print('Drugs: ' + ", ".join(batch["compounds"]))
        monitor_path = ROOT / f"runtime/deployment/evidence_monitor/{batch['run_id']}/status.json"
        if monitor_path.exists():
            mon = read(monitor_path)
            snap = mon.get("snapshot", {})
            print(f"Monitor {mon['status']} | Files {snap.get('files', 0)} | Data {snap.get('batch_bytes', 0)/1024**2:.1f} MiB")
            print(f"Monitor last checked {mon.get('checked_at', 'unknown')}")
            if mon.get("reasons"):
                print('Monitor reason: ' + ", ".join(mon["reasons"]))
        if batch.get("log"):
            print('Log: ' + str(ROOT / batch["log"]))
    if state.get("error"):
        print('Stop reason: ' + state["error"])


STAGE_LABELS = {
    "raw_acquisition": 'Acquisition', "raw_finalization": 'Source completion',
    "pmc_static_transfer": 'PMC full-text transfer', "raw_reconciliation": 'Raw-source verification',
    "raw_inventory": 'Source inventory', "parse": 'File parsing',
    "deterministic_extract": 'Rule extraction', "llm_extract": 'Qwen evidence extraction',
    "table_recovery": 'Table recovery', "completeness": 'Completeness verification',
    "provenance_attribution": 'Evidence attribution', "domain_and_deterministic_dedup": 'Domain filtering and deduplication',
    "semantic_dedup": 'Semantic deduplication', "conflict_and_evidence_layers": 'Conflict checks and layering',
    "rag_index": 'Retrieval index', "deterministic_quality_control": 'Rule quality control',
    "llm_quality_validation": 'Qwen quality validation', "database_import": 'Evidence import',
    "completed": 'Waiting for retrieval verification', "starting": 'Preparing startup',
}


class DownloadMeter:
    """Tail this batch's request ledger; never rescan full logs on every refresh."""
    def __init__(self):
        self.path = None
        self.offset = self.successes = self.failures = 0
        self.drug = ""

    def update(self, path):
        if path != self.path:
            self.path = path
            self.offset = self.successes = self.failures = 0
            self.drug = ""
        if not path.exists():
            return
        if path.stat().st_size < self.offset:
            raise RuntimeError("Download ledger shrank; progress cannot be trusted")
        with path.open("rb") as stream:
            stream.seek(self.offset)
            for _ in range(20000):
                line = stream.readline(2 * 1024**2)
                if not line or not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                self.successes += row.get("status") == "fetched"
                self.failures += row.get("status") == "failed"
                self.drug = row.get("compound") or self.drug
                self.offset = stream.tell()


def progress_snapshot(meter):
    """Read small manifests plus appended download events. Does not signal jobs."""
    small = lambda path: read(path) if path.exists() else {}
    state = small(STATE)
    if not state:
        return None
    batch = state.get("current_batch") or {}
    total_batches = len(state.get("planned_batches", [])) or 7
    completed_batches = len(state.get("batches", []))
    manifest = small(ROOT / f"data_process/pipeline_runs/{batch.get('run_id', '_')}.json")
    stage = manifest.get("current_stage") or "starting"
    steps = sum(s.get("accepted") is True for s in manifest.get("stages", {}).values())
    monitor = small(ROOT / f"runtime/deployment/evidence_monitor/{batch.get('run_id', '_')}/status.json")
    detail = ('Stage active', 0, None, 'No fixed denominator yet')
    if batch:
        raw = ROOT / f"artifacts/pbpk_raw_runs/compound_runs/{batch['run_id']}"
        meter.update(raw / "requests.jsonl")
        processing = ROOT / f"data_process/raw_processing/{batch['run_id']}/00_run_manifest"
        if stage == "raw_finalization":
            targets = {s.casefold() for s in batch["compounds"]}
            done = {str(small(p).get("compound", "")).casefold()
                    for p in (raw / "finalization/pmc").glob("*.json")}
            detail = ('Completing finalization', len(done & targets), len(targets), 'drugs; completion attempts finished')
        elif stage == "parse":
            progress = small(processing / "parse_progress.json")
            detail = ('File parsing', progress.get("processed_raw_asset_count", 0),
                      progress.get("expected_raw_asset_count") or None, 'sources')
        elif stage == "llm_extract":
            progress = small(processing / "llm_extract_progress.json")
            classification = small(processing / "llm_classification_summary.json")
            total = classification.get("classify_unique_input_count") or None
            if total and batch.get("status") == "running_pilot":
                total = min(10, total)
            detail = ('Current extraction', progress.get("api_requests_completed", 0), total,
                      f"requests | Candidates {progress.get('evidence_added', 0)} | Unresolved errors {progress.get('unresolved_api_failures', 0)}")
    live = controller_health(state)
    stopped = (state.get("status") == "stopped_for_review"
               or monitor.get("status") in {"paused_by_monitor", "target_paused"})
    finished = state.get("status") in {"completed", "completed_with_coverage_gaps"}
    label = ('Completed with coverage gaps' if state.get("status") == "completed_with_coverage_gaps"
             else 'Completed' if finished else 'Paused' if stopped else 'Running' if live else 'Supervisor absent')
    elapsed = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(state["started_at"])).total_seconds())
    return {"state": state, "batch": batch, "stage": stage, "stage_name": STAGE_LABELS.get(stage, stage),
            "steps": steps if batch else (18 if finished else 0),
            "total_steps": completed_batches * 18 + (steps if batch else 0),
            "total_batches": total_batches, "completed_batches": completed_batches,
            "covered": state["database"]["covered_compounds"], "target": state.get("target", 100),
            "detail": detail, "monitor": monitor, "label": label,
            "elapsed": f"{int(elapsed)//3600:02}:{int(elapsed)//60%60:02}:{int(elapsed)%60:02}",
            "stopped": stopped, "finished": finished}


def watch_progress():
    """Training-style, read-only progress bars; Ctrl+C affects this viewer only."""
    from tqdm import tqdm
    meter = DownloadMeter()
    view = progress_snapshot(meter)
    if view is None:
        print('Expansion has not started')
        return
    terminal = sys.stdout.isatty()
    print('\n  PBPK drug-evidence expansion | 30 to 100 drugs')
    print('  Counts verified stages; stage durations differ. Ctrl+C exits this display only.\n')
    counted = "{desc}: {percentage:6.2f}%|{bar}| {n_fmt}/{total_fmt}{postfix}"
    activity = "{desc}"
    bars = [tqdm(total=total, initial=0, desc=label, position=i, file=sys.stdout,
                 dynamic_ncols=True, mininterval=0, colour=color if terminal else None,
                 bar_format=counted if total else activity, disable=not terminal, leave=False)
            for i, (label, total, color) in enumerate([
                ('Overall verification', view["total_batches"] * 18, "#38bdf8"),
                ('Current batch', 18, "#2dd4bf"),
                ('Evidence coverage', view["target"], "#a78bfa"),
                ('Current stage', None, "#fbbf24"),
                ('Acquisition activity', None, None),
                ('Run status', None, None),
            ])]
    previous = None
    frame = 0
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    try:
        while view is not None:
            instant = time.monotonic()
            rates = 'Measuring'
            run = view["batch"].get("run_id")
            if previous and previous[0] == run:
                rates = f"{max(0, meter.successes-previous[1])/max(.01, instant-previous[2]):.1f} calls/s"
            previous = (run, meter.successes, instant)
            snap = view["monitor"].get("snapshot", {})
            reasons = view["monitor"].get("reasons") or []
            detail_name, done, total, note = view["detail"]
            mark = "■" if view["stopped"] or view["finished"] else spinner[frame % len(spinner)]
            rows = [
                ('Overall verification', view["total_steps"], view["total_batches"]*18,
                 f"Completed {view['completed_batches']}/{view['total_batches']} batches"),
                ('Current batch', view["steps"], 18,
                 f"Batch {view['completed_batches']+1 if view['batch'] else view['completed_batches']}/{view['total_batches']} | "
                 + ('All batches verified' if view['finished'] else view['stage_name'])),
                ('Evidence coverage', view["covered"], view["target"], 'drugs; updated after import verification'),
                (detail_name, done, total, note if total else f"{mark} {view['stage_name']} · {note}"),
                ('Acquisition activity', 0, None,
                 f"Succeeded {meter.successes:,} calls | {rates} | {snap.get('batch_bytes', 0)/1024**2:.1f} MiB | {snap.get('files', 0):,} files"),
                ('Run status', 0, None,
                 f"{mark} {view['label']} | Elapsed {view['elapsed']} | Source gaps {view['monitor'].get('verified_source_gap_count', 0)} | "
                 + ('Anomaly ' + ", ".join(reasons) if reasons else 'Monitor healthy' if view['monitor'].get('status') == 'monitoring' else f"Monitor {view['monitor'].get('status', 'Connecting')}")
                 + (f" · {view['state']['error']}" if view['state'].get('error') else "")
                 + ' | Background: ' + followup_status()),
            ]
            for bar, (label, n, total, postfix) in zip(bars, rows):
                bar.desc = label if total else label + "  " + postfix
                bar.n, bar.total, bar.postfix = n, total, postfix if total else ""
                bar.bar_format = counted if total else activity
                if terminal:
                    bar.refresh()
                elif total:
                    print(tqdm.format_meter(n, total, 0, prefix=label, postfix=postfix,
                                           ncols=110, bar_format=counted))
                else:
                    print(bar.desc)
            if view["finished"] or not terminal:
                break
            frame += 1
            time.sleep(2)
            view = progress_snapshot(meter)
    except KeyboardInterrupt:
        pass
    finally:
        for bar in reversed(bars):
            bar.close()
        print('\nProgress display closed; background collection and processing are unaffected.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "resume-controller", "status", "watch"))
    args = parser.parse_args()
    os.chdir(ROOT)
    DEPLOY.mkdir(parents=True, exist_ok=True)
    if args.action == "start":
        if STATE.exists():
            raise RuntimeError("Existing expansion state; use status instead of duplicate start")
        process = spawn(SCRIPT, ["run"], DEPLOY / "controller.log")
        print(f"Expansion controller started, project PID {process.pid}")
    elif args.action == "run":
        run()
    elif args.action == "resume-controller":
        run(resume=True)
    elif args.action == "status":
        status()
    else:
        watch_progress()


if __name__ == "__main__":
    main()
