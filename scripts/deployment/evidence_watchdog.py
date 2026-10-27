"""Watch only the project's evidence job; pause its private session on anomalies."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import time

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "runtime/deployment"
GIB = 1024 ** 3
DEFAULT_LIMITS = {
    "stop_on_any_failure": True,
    "audit_interval_seconds": 1200,
    "capacity_interval_seconds": 30,
    "max_batch_bytes": 20 * GIB,
    "max_growth_per_audit_bytes": 5 * GIB,
    "min_free_bytes": 10 * GIB,
    "max_files": 100000,
    "max_pipeline_log_bytes": 64 * 1024 ** 2,
    "max_repeated_successes": 20,
    "max_failure_rows_per_audit": 1000,
    "max_consecutive_failed_requests": 30,
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def project_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Watchdog paths must remain inside the project.")
    return path


def identity(pid):
    # Read only the explicitly registered project PID, never enumerate processes.
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid():
        raise ValueError("Target PID is not owned by the current account.")
    if (proc / "cwd").resolve() != ROOT:
        raise ValueError("Target PID is outside the project working directory.")
    command = (proc / "cmdline").read_bytes().split(b"\0")
    script = b"scripts/deployment/run_evidence_pipeline.py"
    if script not in command and str(ROOT / script.decode()).encode() not in command:
        raise ValueError("Target PID is not the evidence pipeline launcher.")
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    if os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise ValueError("Evidence pipeline must have its own process group and session.")
    return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}


def same_process(expected):
    try:
        current = identity(expected["pid"])
    except FileNotFoundError:
        return None
    if current["start_ticks"] != expected["start_ticks"]:
        raise ValueError("Target PID has been reused; refusing to signal it.")
    return current


def disk_snapshot(roots, pipeline_log):
    sizes = files = 0
    newest = 0
    for root in roots:
        root = project_path(root)
        if not root.exists():
            continue
        for folder, directories, names in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
            for name in names:
                try:
                    item = (Path(folder) / name).lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(item.st_mode):
                    continue
                sizes += item.st_size
                files += 1
                newest = max(newest, item.st_mtime)
    usage = os.statvfs(ROOT)
    return {"batch_bytes": sizes, "files": files, "latest_data_mtime": newest,
            "free_bytes": usage.f_bavail * usage.f_frsize,
            "pipeline_log_bytes": pipeline_log.stat().st_size if pipeline_log.exists() else 0}


def capacity_reasons(snapshot, baseline, limits):
    reasons = []
    for key, limit in (("batch_bytes", "max_batch_bytes"), ("files", "max_files"),
                       ("pipeline_log_bytes", "max_pipeline_log_bytes")):
        if snapshot[key] >= limits[limit]:
            reasons.append(limit)
    if snapshot["free_bytes"] < limits["min_free_bytes"]:
        reasons.append("min_free_bytes")
    if snapshot["batch_bytes"] - baseline >= limits["max_growth_per_audit_bytes"]:
        reasons.append("max_growth_per_audit_bytes")
    return reasons


def read_new_rows(path, offset, keys=("status", "request_id", "url", "source_id", "compound", "request_key", "error")):
    """Stream complete ledger lines without retaining raw URLs or response bodies."""
    if not path.exists():
        return [], offset
    if path.stat().st_size < offset:
        raise ValueError("An acquisition ledger shrank unexpectedly.")
    rows = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            line = handle.readline(2 * 1024 ** 2)
            if not line:
                break
            if not line.endswith(b"\n"):
                if len(line) >= 2 * 1024 ** 2:
                    raise ValueError("Oversized acquisition ledger entry.")
                break  # An in-progress writer will complete this line later.
            row = json.loads(line)
            rows.append({key: row.get(key) for key in keys})
            offset = handle.tell()
            if len(rows) >= 200000:
                raise ValueError("Excessive acquisition events in one audit interval.")
    return rows, offset


def classify_verified_source_gaps(raw, rows, proofs):
    """Keep known unavailable observations out of evidence without halting other sources."""
    from pbpk_kb.acquisition.source_recovery import (
        verified_pkdb_observation_gap, verified_openfda_empty_lookup, verified_chembl_service_gap,
        verified_openfda_download_recovery,
    )
    candidates = [r for r in rows if r.get("status") == "failed"
                  and ((r.get("source_id") == "pkdb" and r.get("request_key") == "open_bulk_export_incomplete")
                       or (r.get("source_id") == "openfda_drug_label" and str(r.get("request_key") or "").startswith("label_skip_"))
                       or (r.get("source_id") == "chembl" and r.get("request_key") == "molecule_search"))]
    assets = None
    for row in candidates:
        if row["request_id"] in proofs:
            continue
        if assets is None:
            assets, _ = read_new_rows(raw / "assets.jsonl", 0, keys=(
                "asset_id", "request_id", "source_id", "compound", "request_key", "url", "storage_path",
                "checksum", "byte_count", "status_code", "processing_policy", "metadata"))
        if row.get("source_id") == "pkdb":
            proof = verified_pkdb_observation_gap(row, assets, root=ROOT)
        elif row.get("source_id") == "chembl":
            proof = verified_chembl_service_gap(raw, row, assets, root=ROOT)
        else:
            proof = verified_openfda_download_recovery(raw, row, assets, root=ROOT)
            if proof is None:
                proof = verified_openfda_empty_lookup(raw, row, assets, root=ROOT)
        if proof:
            proofs[row["request_id"]] = proof
    return [{**row, "status": proofs[row["request_id"]].get("status", "source_gap_metadata_only")}
            if row.get("status") == "failed" and row.get("request_id") in proofs
            else row for row in rows]


def request_audit(rows, limits):
    successes = Counter()
    status = Counter()
    streak = longest_streak = 0
    for row in rows:
        status[str(row.get("status"))] += 1
        if row.get("status") == "failed":
            streak += 1
            longest_streak = max(longest_streak, streak)
        elif row.get("status") == "fetched":
            streak = 0
        if row.get("status") == "fetched":
            # Same URL may be legitimate for different target compounds.
            key = (row.get("compound"), row.get("source_id"), row.get("url") or row.get("request_id"))
            successes[key] += 1
    maximum = max(successes.values(), default=0)
    failures = status["failed"]
    reasons = []
    if limits.get("stop_on_any_failure", True) and failures:
        reasons.append("download_failure_strict_stop")
    if longest_streak >= limits["max_consecutive_failed_requests"]:
        reasons.append("consecutive_download_failures")
    if maximum >= limits["max_repeated_successes"]:
        reasons.append("repeated_successful_download")
    if failures >= limits["max_failure_rows_per_audit"]:
        reasons.append("excessive_download_failures")
    if failures >= 100 and failures / max(1, status["started"]) >= .9:
        reasons.append("persistent_download_failure_loop")
    return {"events": len(rows), "status_counts": dict(status),
            "max_same_url_successes": maximum, "max_consecutive_failures": longest_streak, "reasons": reasons}


def evidence_audit(requests, responses, *, historical=False, retry_historical_oom=False):
    """Audit accepted quotes continuously; ordinary candidate rejection is expected."""
    from pbpk_kb.processing.chunk_classifier import source_assertion_completeness_error
    from pbpk_kb.processing.run_llm_extraction import (
        _joint_target_metabolite_subject_error, _parameter_owner_error, _numeric_range_source_error,
        _clearance_context_error, _numeric_parameter_owner_error,
    )
    from types import SimpleNamespace

    reasons = set()
    checked = 0
    requeued = 0
    # Only at attachment may a failed attempt be superseded by a later,
    # independently completed recovery of the exact same request and input.
    # New failures always stop the monitored process, even if retried quickly.
    identity_fields = ("request_id", "input_hash", "prompt_template_version", "validator_version")
    recovered = {}
    if historical:
        for response in responses:
            key = tuple(response.get(field) for field in identity_fields)
            if all(key) and response.get("status") in {"ok", "reused_same_target_input_hash"}:
                recovered[key] = max(recovered.get(key, 0), int(response.get("attempt_number") or 0))
    for response in responses:
        if response.get("status") not in {"ok", "reused_same_target_input_hash"}:
            if historical and retry_historical_oom and verified_resource_failure(requests.get(response.get("request_id")), response):
                requeued += 1
                continue
            key = tuple(response.get(field) for field in identity_fields)
            if recovered.get(key, 0) > int(response.get("attempt_number") or 0):
                continue
            if historical and verified_table_deferral(requests.get(response.get("request_id")), response):
                continue
            reasons.add("llm_request_failure_strict_stop")
            continue
        request = requests.get(response.get("request_id"))
        if not request:
            reasons.add("accepted_evidence_request_missing")
            continue
        for candidate in response.get("candidates") or []:
            checked += 1
            quote = candidate.get("source_text") or ""
            if quote not in (request.get("input_items") or []):
                reasons.add("accepted_evidence_quote_not_verbatim_input")
            if _joint_target_metabolite_subject_error(
                    quote, request.get("target_compound") or "", candidate.get("parameter") or ""):
                reasons.add("accepted_evidence_joint_chemical_subject")
            if _parameter_owner_error(candidate.get("evidence_type") or "", candidate.get("parameter") or "",
                                      quote, request.get("target_compound") or ""):
                reasons.add("accepted_evidence_parameter_owner_mismatch")
            interval = SimpleNamespace(**{key: candidate.get(key) for key in (
                "value", "value_low", "value_high", "unit", "parameter", "source_text")})
            if _numeric_range_source_error(interval):
                reasons.add("accepted_evidence_range_not_in_source")
            if _numeric_parameter_owner_error(candidate, request.get("target_compound") or ""):
                reasons.add("accepted_evidence_numeric_owner_mismatch")
            if _clearance_context_error(candidate.get("parameter") or "", quote,
                                        request.get("target_compound") or ""):
                reasons.add("accepted_evidence_clearance_context_mismatch")
            if source_assertion_completeness_error(
                quote, target_compound=request.get("target_compound") or "",
                allow_flattened_table=bool(request.get("parsed_table_id")),
            ):
                reasons.add("accepted_evidence_incomplete_assertion")
    return {"accepted_candidates_checked": checked, "responses_checked": len(responses),
            "historical_gpu_failures_requeued": requeued,
            "reasons": sorted(reasons)}


def verified_resource_failure(request, response):
    """Only an empty, exact-input CUDA OOM can be queued for a fresh attempt."""
    from hashlib import sha256
    if (not request or response.get("status") != "failed_local_llm" or response.get("candidates") != []
            or response.get("output_hash") is not None
            or not str(response.get("failure_reason") or "").startswith("CUDA out of memory")):
        return False
    target, fragment = request.get("target_compound"), request.get("input_text_or_table")
    if not isinstance(target, str) or not target or not isinstance(fragment, str) or not fragment:
        return False
    digest = sha256(json.dumps({"target_compound": target, "input_text_or_table": fragment},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return (request.get("request_id") == response.get("request_id")
            and digest == request.get("input_hash") == response.get("input_hash"))


def verified_table_deferral(request, response):
    """Keep a reviewed failed request explicitly deferred, never call it LLM success."""
    from hashlib import sha256
    from pbpk_kb.processing.chunk_classifier import prepare_llm_input
    from pbpk_kb.processing.run_llm_extraction import LLM_ROUTING_VERSION
    if not request or response.get("resolved") is not True or response.get("candidates"):
        return False
    if (response.get("resolution") != "no_longer_eligible_under_current_routing"
            or response.get("resolution_destination") != "table_recovery"):
        return False
    import re
    saved = re.fullmatch(r"pbpk-llm-routing-v(\d+)", str(response.get("resolution_routing_version") or ""))
    current = re.fullmatch(r"pbpk-llm-routing-v(\d+)", LLM_ROUTING_VERSION)
    if not saved or not current or int(saved[1]) > int(current[1]):
        return False
    fragment = request.get("input_text_or_table")
    target = request.get("target_compound")
    if not isinstance(fragment, str) or not fragment or not isinstance(target, str) or not target:
        return False
    digest = sha256(json.dumps({"target_compound": target, "input_text_or_table": fragment},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != response.get("input_hash") or digest != request.get("input_hash"):
        return False
    decision = prepare_llm_input({"chunk_text": fragment, "source_id": request.get("source_id"),
                                  "parsed_table_id": request.get("parsed_table_id")}, target)
    return not decision.eligible and decision.status == "deferred_unparsed_flattened_table"


def later_llm_audit(final_directory, cursors):
    """Stream later-stage decisions without retaining evidence or model replies.

    Attribution atomically rewrites its ledger when the stage completes, so a
    new inode must be audited from the start. Neither later-stage schema carries
    the full request identity needed to prove that an old failure was recovered.
    """
    ledgers = {
        "attribution": final_directory / "01_compound_attribution/llm_attribution_responses.jsonl",
        "quality": final_directory / "07_llm_quality_validation/llm_quality_results.jsonl",
    }
    terminal_quality_statuses = {
        "auto_validated", "quarantined_unrepairable", "quarantined_validation_failed",
        "quarantined", "excluded",
    }
    checked = Counter()
    reasons = set()
    for stage, path in ledgers.items():
        previous = cursors.get(stage)
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            if previous is not None:
                raise ValueError("A monitored LLM decision ledger disappeared.")
            continue
        with handle:
            file_stat = os.fstat(handle.fileno())
            identity = (file_stat.st_dev, file_stat.st_ino)
            cursor = dict(previous) if previous and previous["identity"] == identity else {
                "identity": identity, "offset": 0, "rows": 0,
            }
            if file_stat.st_size < cursor["offset"]:
                raise ValueError("An LLM decision ledger shrank without an atomic replacement.")
            handle.seek(cursor["offset"])
            while True:
                line = handle.readline(2 * 1024 ** 2)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    if len(line) >= 2 * 1024 ** 2:
                        raise ValueError("Oversized LLM decision ledger entry.")
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("An LLM decision ledger entry is not an object.")
                cursor["rows"] += 1
                if cursor["rows"] > 100000:
                    raise ValueError("Excessive LLM decisions in one batch.")
                cursor["offset"] = handle.tell()
                checked[stage] += 1
                if stage == "attribution":
                    if (row.get("raw_response") or {}).get("status") != "ok":
                        reasons.add("llm_attribution_request_failure_strict_stop")
                    elif (row.get("validation") or {}).get("status") != "valid":
                        reasons.add("llm_attribution_validation_failure_strict_stop")
                elif row.get("terminal") is not True or row.get("status") not in terminal_quality_statuses:
                    reasons.add("llm_quality_request_failure_strict_stop")
            cursors[stage] = cursor
    return {"attribution_responses_checked": checked["attribution"],
            "quality_results_checked": checked["quality"], "reasons": sorted(reasons)}


def pause_target(expected):
    if same_process(expected) is None:
        return False
    # The launcher starts a private session; its download subprocesses inherit it.
    # SIGSTOP preserves every file and freezes writers immediately, with no retry loop.
    os.killpg(expected["pid"], signal.SIGSTOP)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log-path", default=str(DEPLOY / "logs/evidence_pipeline.log"),
                        help="This batch's project-local pipeline log; legacy default retained.")
    parser.add_argument("--resume-reviewed-source-gaps", action="store_true",
                        help="Resume a paused job only when every historical failure is a verified source gap.")
    parser.add_argument("--retry-historical-oom", action="store_true",
                        help="Requeue reviewed empty CUDA OOM attempts; all new failures still stop immediately.")
    args = parser.parse_args()
    manifest_path = project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest_at_attach = manifest.get("updated_at")
    target = identity(args.pid)
    run = manifest["run_id"]
    directory = DEPLOY / "evidence_monitor" / run
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (directory / "monitor.pid").write_text(str(os.getpid()) + "\n")
    limits_path = directory / "limits.json"
    if not limits_path.exists():
        write_json(limits_path, DEFAULT_LIMITS)
    limits = {**DEFAULT_LIMITS, **json.loads(limits_path.read_text())}
    raw = project_path(manifest["raw_run_root"])
    roots = [raw, project_path(manifest["raw_processing_dir"]),
             project_path(manifest["final_evidence_dir"])]
    for source in manifest.get("shared_sources", []):
        roots.append(project_path(Path(manifest["raw_output_root"]) / "source_cache" / source / run))
    pipeline_log = project_path(args.log_path)
    state_path = directory / "status.json"
    history_path = directory / "audits.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines()][-144:] if history_path.exists() else []
    # Earlier failures are retained in audit history; a resumed process is
    # evaluated on new traffic rather than immediately tripped by old errors.
    offsets = (raw / "requests.jsonl").stat().st_size
    initial_offset = offsets
    transfer_ledger = raw / "finalization/pmc_object_transfer/transfer.jsonl"
    transfer_offset = transfer_ledger.stat().st_size if transfer_ledger.exists() else 0
    request_window = []
    llm_dir = project_path(manifest["raw_processing_dir"]) / "05_llm_extraction"
    llm_request_offset = llm_response_offset = 0
    llm_history_pending = True
    llm_requests = {}
    llm_totals = {"accepted_candidates_checked": 0, "responses_checked": 0, "historical_gpu_failures_requeued": 0}
    later_llm_cursors = {}
    later_llm_totals = {"attribution_responses_checked": 0, "quality_results_checked": 0}
    snapshot = disk_snapshot(roots, pipeline_log)
    baseline = snapshot["batch_bytes"]
    source_gap_proofs = {}
    if args.resume_reviewed_source_gaps:
        current = same_process(target)
        if current is None or current["state"] not in {"T", "t"}:
            raise RuntimeError("Source-gap resume requires the same paused project process.")
        historical, _ = read_new_rows(raw / "requests.jsonl", 0)
        classified = classify_verified_source_gaps(raw, historical, source_gap_proofs)
        if request_audit(classified, limits)["reasons"] or capacity_reasons(snapshot, baseline, limits):
            raise RuntimeError("Unresolved failure or capacity issue; project job remains paused.")
        write_json(directory / "source_gaps.json", list(source_gap_proofs.values()))
        same_process(target)
        os.killpg(target["pid"], signal.SIGCONT)
    next_audit = 0
    previous_data_mtime = snapshot["latest_data_mtime"]
    stagnant_audits = 0
    while True:
        status = "monitoring"
        reasons = []
        audit = None
        evidence_check = None
        later_llm_check = None
        try:
            current = same_process(target)
            if current is None or current["state"] == "Z":
                status = "target_exited"
            elif current["state"] in {"T", "t"}:
                status = "target_paused"
            snapshot = disk_snapshot(roots, pipeline_log)
            reasons = capacity_reasons(snapshot, baseline, limits)
            manifest = json.loads(manifest_path.read_text())
            rows, offsets = read_new_rows(raw / "requests.jsonl", offsets)
            prior_gap_count = len(source_gap_proofs)
            rows = classify_verified_source_gaps(raw, rows, source_gap_proofs)
            if len(source_gap_proofs) != prior_gap_count:
                write_json(directory / "source_gaps.json", list(source_gap_proofs.values()))
            request_window.extend(rows)
            transfer_rows, transfer_offset = read_new_rows(transfer_ledger, transfer_offset)
            request_window.extend(transfer_rows)
            new_requests, llm_request_offset = read_new_rows(
                llm_dir / "llm_extraction_requests_live.jsonl", llm_request_offset,
                keys=("request_id", "input_items", "input_text_or_table", "input_hash", "source_id", "target_compound", "parsed_table_id"),
            )
            llm_requests.update({row["request_id"]: row for row in new_requests})
            if len(llm_requests) > 100000:
                raise ValueError("Excessive evidence requests in one batch.")
            new_responses, llm_response_offset = read_new_rows(
                llm_dir / "llm_extraction_responses_live.jsonl", llm_response_offset,
                keys=("request_id", "status", "candidates", "input_hash", "prompt_template_version",
                      "validator_version", "attempt_number", "resolved", "resolution", "resolution_destination", "resolution_routing_version",
                      "failure_reason", "output_hash"),
            )
            # Re-read newly appended requests if a writer raced the first read.
            if any(row.get("request_id") not in llm_requests for row in new_responses):
                more_requests, llm_request_offset = read_new_rows(
                    llm_dir / "llm_extraction_requests_live.jsonl", llm_request_offset,
                    keys=("request_id", "input_items", "input_text_or_table", "input_hash", "source_id", "target_compound", "parsed_table_id"),
                )
                llm_requests.update({row["request_id"]: row for row in more_requests})
            evidence_check = evidence_audit(llm_requests, new_responses, historical=llm_history_pending,
                                           retry_historical_oom=args.retry_historical_oom)
            llm_history_pending = False
            for key in llm_totals:
                llm_totals[key] += evidence_check[key]
            reasons.extend(evidence_check["reasons"])
            later_llm_check = later_llm_audit(roots[2], later_llm_cursors)
            for key in later_llm_totals:
                later_llm_totals[key] += later_llm_check[key]
            reasons.extend(later_llm_check["reasons"])
            if manifest.get("status") == "failed" and manifest.get("updated_at") != manifest_at_attach:
                reasons.append("pipeline_stage_failed")
            if len(request_window) > 200000:
                raise ValueError("Excessive acquisition activity in one audit interval.")
            traffic = request_audit(request_window, limits)
            reasons.extend(traffic["reasons"])
            if time.monotonic() >= next_audit:
                audit = traffic
                request_window = []
                if snapshot["latest_data_mtime"] == previous_data_mtime and next_audit:
                    stagnant_audits += 1
                else:
                    stagnant_audits = 0
                # Report long operations as warnings; disk and repetition guards
                # decide whether to pause, so GPU computation is not mistaken for a loop.
                audit["stagnant_audits"] = stagnant_audits
                previous_data_mtime = snapshot["latest_data_mtime"]
                baseline = snapshot["batch_bytes"]
                next_audit = time.monotonic() + limits["audit_interval_seconds"]
        except Exception as exc:
            # If monitoring itself fails, do not leave the writer unguarded.
            reasons = ["monitor_check_failed:" + type(exc).__name__]
        if reasons and status == "monitoring":
            try:
                status = "paused_by_monitor" if pause_target(target) else "target_exited"
            except (ValueError, ProcessLookupError, FileNotFoundError):
                status = "target_identity_changed_no_signal_sent"
        state = {"checked_at": timestamp(), "run_id": run, "target": target,
                 "monitor_pid": os.getpid(), "status": status, "stage": manifest.get("current_stage"),
                 "pipeline_status": manifest.get("status"), "snapshot": snapshot,
                 "reasons": reasons, "audit": audit,
                 "evidence_audit": evidence_check, "evidence_totals": llm_totals,
                 "later_llm_audit": later_llm_check, "later_llm_totals": later_llm_totals,
                 "seconds_until_next_audit": max(0, round(next_audit - time.monotonic())),
                 "limits": limits}
        state["verified_source_gap_count"] = sum(p.get("status") != "resolved" for p in source_gap_proofs.values())
        state["verified_recovery_count"] = sum(p.get("status") == "resolved" for p in source_gap_proofs.values())
        state["ledger_offset"] = offsets
        state["ledger_start_offset"] = initial_offset
        state["transfer_ledger_offset"] = transfer_offset
        if audit is not None or status != "monitoring":
            history.append(state)
            history = history[-144:]  # Two days of full checks; bounded monitor output.
            temporary = history_path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history))
            temporary.replace(history_path)
        write_json(state_path, state)
        if status != "monitoring":
            return
        time.sleep(limits["capacity_interval_seconds"])


if __name__ == "__main__":
    main()
