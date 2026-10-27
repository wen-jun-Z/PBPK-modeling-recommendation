import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evidence_watchdog", ROOT / "scripts/deployment/evidence_watchdog.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_limits_guard_capacity_growth_files_and_logs():
    good = {"batch_bytes": 100, "files": 2, "free_bytes": 100 * watch.GIB, "pipeline_log_bytes": 0}
    assert watch.capacity_reasons(good, 0, watch.DEFAULT_LIMITS) == []
    bad = {**good, "batch_bytes": 21 * watch.GIB, "files": 100001,
           "free_bytes": 1, "pipeline_log_bytes": 100 * 1024 ** 2}
    assert set(watch.capacity_reasons(bad, 0, watch.DEFAULT_LIMITS)) == {
        "max_batch_bytes", "max_files", "min_free_bytes", "max_pipeline_log_bytes",
        "max_growth_per_audit_bytes",
    }


def test_repeat_and_failure_loop_guards_do_not_reject_distinct_downloads():
    def row(url):
        return {"compound": "caffeine", "source_id": "pubmed", "url": url, "status": "fetched"}
    assert watch.request_audit([row(str(i)) for i in range(100)], watch.DEFAULT_LIMITS)["reasons"] == []
    assert "repeated_successful_download" in watch.request_audit([row("same")] * 20, watch.DEFAULT_LIMITS)["reasons"]
    errors = [{"status": "started"}, {"status": "failed"}] * 100
    assert "persistent_download_failure_loop" in watch.request_audit(errors, watch.DEFAULT_LIMITS)["reasons"]


def test_single_failure_among_successes_also_stops_static_transfer(tmp_path):
    path = tmp_path / "transfer.jsonl"
    path.write_text('{' + '"status":"failed","error":"HTTP 502"' + '}\n')
    rows, _ = watch.read_new_rows(path, 0)
    result = watch.request_audit([{"status": "completed"}] * 200 + rows, watch.DEFAULT_LIMITS)
    assert "download_failure_strict_stop" in result["reasons"]


def test_only_a_verified_source_gap_can_continue_while_transport_failure_still_stops(tmp_path, monkeypatch):
    from pbpk_kb.acquisition import source_recovery
    (tmp_path / "assets.jsonl").write_text("")
    gap = {"status": "failed", "request_id": "gap", "request_key": "open_bulk_export_incomplete", "source_id": "pkdb"}
    transport = {"status": "failed", "request_id": "network", "request_key": "open_bulk_export", "source_id": "pkdb"}
    proofs = {}
    monkeypatch.setattr(source_recovery, "verified_pkdb_observation_gap", lambda *args, **kwargs: None)
    assert "download_failure_strict_stop" in watch.request_audit(watch.classify_verified_source_gaps(tmp_path, [gap], proofs), watch.DEFAULT_LIMITS)["reasons"]
    monkeypatch.setattr(source_recovery, "verified_pkdb_observation_gap", lambda *args, **kwargs: {"request_id": "gap", "accepted_for_observation_processing": False})
    classified = watch.classify_verified_source_gaps(tmp_path, [gap], proofs)
    assert watch.request_audit(classified, watch.DEFAULT_LIMITS)["reasons"] == []
    assert gap["status"] == "failed"  # Original ledger fact is preserved.
    assert "download_failure_strict_stop" in watch.request_audit(classified + [transport], watch.DEFAULT_LIMITS)["reasons"]


def test_partial_json_line_is_not_consumed_or_reported_as_corruption(tmp_path):
    path = tmp_path / "requests.jsonl"
    complete = json.dumps({"status": "fetched"}) + "\n"
    path.write_text(complete + '{"status":')
    rows, offset = watch.read_new_rows(path, 0)
    assert len(rows) == 1
    assert offset == len(complete.encode())
    with path.open("a") as handle:
        handle.write('"started"}\n')
    rows, offset = watch.read_new_rows(path, offset)
    assert rows[0]["status"] == "started"


def test_verified_chembl_recovery_is_distinct_from_a_source_gap(tmp_path, monkeypatch):
    from pbpk_kb.acquisition import source_recovery
    (tmp_path / "assets.jsonl").write_text("")
    failure = {"status": "failed", "source_id": "chembl", "request_key": "molecule_search", "request_id": "recovered"}
    monkeypatch.setattr(source_recovery, "verified_chembl_service_gap",
                        lambda *args, **kwargs: {"request_id": "recovered", "status": "resolved"})
    classified = watch.classify_verified_source_gaps(tmp_path, [failure], {})
    assert classified[0]["status"] == "resolved"
    assert failure["status"] == "failed"
    assert watch.request_audit(classified, watch.DEFAULT_LIMITS)["reasons"] == []


def test_verified_openfda_recovery_uses_complete_asset_fields_and_keeps_unknown_failure_stop(tmp_path, monkeypatch):
    from pbpk_kb.acquisition import source_recovery
    failure = {"status": "failed", "source_id": "openfda_drug_label", "compound": "sildenafil",
               "request_key": "label_skip_100", "request_id": "recovered"}
    asset = {"asset_id": "a", "request_id": "recovered", "source_id": "openfda_drug_label",
             "compound": "sildenafil", "request_key": "label_skip_100", "byte_count": 123,
             "checksum": "checksum", "processing_policy": None}
    (tmp_path / "assets.jsonl").write_text(json.dumps(asset) + "\n")
    monkeypatch.setattr(watch, "ROOT", tmp_path)

    def verify(raw, row, assets, *, root):
        assert raw == root == tmp_path
        assert assets[0]["request_id"] == "recovered" and assets[0]["byte_count"] == 123
        assert "processing_policy" in assets[0]
        return {"request_id": row["request_id"], "status": "resolved"}

    monkeypatch.setattr(source_recovery, "verified_openfda_download_recovery", verify)
    monkeypatch.setattr(source_recovery, "verified_openfda_empty_lookup", lambda *a, **k: pytest.fail("Recovery is not empty lookup"))
    proofs = {}
    unknown = {"status": "failed", "source_id": "pubmed", "request_id": "unresolved"}
    classified = watch.classify_verified_source_gaps(tmp_path, [failure, unknown], proofs)
    assert classified[0]["status"] == "resolved" and failure["status"] == "failed"
    assert watch.request_audit(classified[:1], watch.DEFAULT_LIMITS)["reasons"] == []
    assert "download_failure_strict_stop" in watch.request_audit(classified, watch.DEFAULT_LIMITS)["reasons"]
    assert proofs["recovered"]["status"] == "resolved"


def test_pid_reuse_refuses_to_signal(monkeypatch):
    monkeypatch.setattr(watch, "identity", lambda pid: {"pid": pid, "start_ticks": "new"})
    signals = []
    monkeypatch.setattr(os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(ValueError, match="reused"):
        watch.pause_target({"pid": 123, "start_ticks": "old"})
    assert signals == []


def test_pause_really_freezes_only_an_isolated_project_probe():
    # A disposable project-owned probe, not the production pipeline. The marker
    # exercises the same launcher identity check without running acquisition.
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                                "scripts/deployment/run_evidence_pipeline.py"],
                               cwd=ROOT, start_new_session=True,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                expected = watch.identity(process.pid)
                break
            except ValueError:
                time.sleep(.01)
        assert watch.pause_target(expected) is True
        for _ in range(100):
            if watch.identity(process.pid)["state"] == "T":
                break
            time.sleep(.01)
        assert watch.identity(process.pid)["state"] == "T"
    finally:
        os.killpg(process.pid, signal.SIGCONT)
        process.terminate()
        process.wait(timeout=5)


def test_evidence_monitor_checks_verbatim_complete_quotes_and_local_failures():
    quote = "Midazolam clearance was 12 L/h in healthy adults."
    requests = {"one": {"input_items": [quote], "target_compound": "midazolam"}}
    valid = {"request_id": "one", "status": "ok", "candidates": [{"source_text": quote}]}
    assert watch.evidence_audit(requests, [valid])["reasons"] == []
    changed = {**valid, "candidates": [{"source_text": quote.replace("12", "120")}]}
    assert "accepted_evidence_quote_not_verbatim_input" in watch.evidence_audit(requests, [changed])["reasons"]
    truncated = "Midazolam clearance increased from"
    request = {"one": {"input_items": [truncated], "target_compound": "midazolam"}}
    response = {**valid, "candidates": [{"source_text": truncated}]}
    assert "accepted_evidence_incomplete_assertion" in watch.evidence_audit(request, [response])["reasons"]
    assert "llm_request_failure_strict_stop" in watch.evidence_audit(
        requests, [{**valid, "status": "failed_local_llm", "candidates": []}]
    )["reasons"]


def test_rejected_candidates_are_audited_without_stopping_valid_evidence():
    requests = {"one": {"input_items": ["Midazolam clearance was measured."], "target_compound": "midazolam"}}
    response = {"request_id": "one", "status": "ok", "candidates": [],
                "rejected_candidates": [{"errors": ["unit_dimension_does_not_match_parameter"]}]}
    result = watch.evidence_audit(requests, [response])
    assert result["reasons"] == []
    assert result["accepted_candidates_checked"] == 0


def test_only_verified_historical_recovery_can_supersede_a_failed_attempt():
    request = {"one": {"input_items": [], "target_compound": "caffeine"}}
    failed = {"request_id": "one", "input_hash": "hash", "prompt_template_version": "p",
              "validator_version": "v", "attempt_number": 1, "status": "failed_local_llm"}
    recovered = {**failed, "attempt_number": 2, "status": "ok", "candidates": []}
    assert watch.evidence_audit(request, [failed, recovered], historical=True)["reasons"] == []
    assert watch.evidence_audit(request, [failed, recovered])["reasons"] == ["llm_request_failure_strict_stop"]
    for override in [{"input_hash": "different"}, {"attempt_number": 1}, {"validator_version": "other"}]:
        assert watch.evidence_audit(request, [failed, {**recovered, **override}], historical=True)["reasons"]


def test_only_exact_empty_historical_oom_can_be_requeued_explicitly():
    from hashlib import sha256
    payload = {"target_compound": "caffeine", "input_text_or_table": "Caffeine clearance was 12 L/h."}
    digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request = {**payload, "request_id": "one", "input_hash": digest}
    response = {"request_id": "one", "input_hash": digest, "status": "failed_local_llm",
                "candidates": [], "output_hash": None, "failure_reason": "CUDA out of memory while loading weights"}
    for options in ({}, {"historical": True}, {"retry_historical_oom": True}):
        assert watch.evidence_audit({"one": request}, [response], **options)["reasons"] == ["llm_request_failure_strict_stop"]
    options = {"historical": True, "retry_historical_oom": True}
    result = watch.evidence_audit({"one": request}, [response], **options)
    assert result["reasons"] == []
    assert result["historical_gpu_failures_requeued"] == 1
    assert result["accepted_candidates_checked"] == 0
    for override in ({"input_hash": "wrong"}, {"request_id": "other"}, {"output_hash": "output"},
                     {"candidates": [{"source_text": "invented"}]}, {"failure_reason": "generation unavailable"}):
        assert watch.evidence_audit({"one": request}, [{**response, **override}], **options)["reasons"]
    assert watch.evidence_audit({"one": {**request, "input_text_or_table": "changed"}}, [response], **options)["reasons"]
    successful_but_invalid = {**response, "status": "ok", "candidates": [{"source_text": "Invented evidence."}]}
    assert "accepted_evidence_quote_not_verbatim_input" in watch.evidence_audit(
        {"one": request}, [successful_but_invalid], **options)["reasons"]


def test_server_concurrency_is_separate_from_local_gpu_batches():
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("evidence_launcher", ROOT / "scripts/deployment/run_evidence_pipeline.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    args = SimpleNamespace(gpu_batch_size=1, wait_for_gpu=False, request_concurrency=None)
    assert launcher.request_concurrency(args, server=True, registered_default=8) == 8
    args.request_concurrency = 4
    assert launcher.request_concurrency(args, server=True, registered_default=8) == 4
    with pytest.raises(ValueError, match="requires"):
        launcher.request_concurrency(args, server=False)
    for key in ("gpu_batch_size", "wait_for_gpu"):
        bad = SimpleNamespace(**{**vars(args), key: 4 if key == "gpu_batch_size" else True})
        with pytest.raises(ValueError, match="not local GPU"):
            launcher.request_concurrency(bad, server=True)
    args.request_concurrency = None
    for bad in (0, 17, True, "4"):
        with pytest.raises(ValueError, match="between 1 and 16"):
            launcher.request_concurrency(args, server=True, registered_default=bad)
    args.gpu_batch_size = 4
    assert launcher.request_concurrency(args, server=False) == 8


def test_explicit_launcher_run_stays_inside_project_and_never_restarts_populated_run(tmp_path):
    spec = importlib.util.spec_from_file_location("evidence_launcher", ROOT / "scripts/deployment/run_evidence_pipeline.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    path = launcher.explicit_run_root(tmp_path, "artifacts/pbpk_raw_runs/compound_runs/run_new", new_run=True)
    assert path.is_dir()
    assert launcher.explicit_run_root(tmp_path, None) is None
    marker = path / "original.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="populated"):
        launcher.explicit_run_root(tmp_path, str(path), new_run=True)
    assert launcher.explicit_run_root(tmp_path, str(path)) == path
    assert marker.read_text() == "preserve"
    for value in ("../outside/run_new", "artifacts/pbpk_raw_runs/compound_runs", "artifacts/pbpk_raw_runs/compound_runs/not_run"):
        with pytest.raises(ValueError, match="explicit evidence run"):
            launcher.explicit_run_root(tmp_path, value)


def test_gpu_capacity_wait_overwrites_one_status_file_before_resuming(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("evidence_launcher", ROOT / "scripts/deployment/run_evidence_pipeline.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    pipeline = SimpleNamespace(manifest_path=tmp_path / "data_process/pipeline_runs/run.json", run_id="run")
    status_path = tmp_path / "runtime/deployment/evidence_monitor/run/gpu_capacity.json"
    capacities = iter(["4000\n", "24576\n"])
    sleeps = []
    def query(command, **kwargs):
        assert command == ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=next(capacities))
    def sleep(seconds):
        assert json.loads(status_path.read_text())["status"] == "waiting_for_gpu"
        sleeps.append(seconds)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(launcher.subprocess, "run", query)
    monkeypatch.setattr(launcher.time, "sleep", sleep)
    launcher.wait_for_gpu(pipeline)
    assert sleeps == [60]
    assert json.loads(status_path.read_text())["status"] == "ready"
    assert list(status_path.parent.iterdir()) == [status_path]
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(RuntimeError, match="Cannot verify GPU capacity"):
        launcher.wait_for_gpu(pipeline)


def test_monitor_stops_a_joint_chemical_assertion_accepted_by_an_old_validator():
    quote = "Urinary excretion of caffeine and 5-acetylamino-6-formylamino-3-methyluracil was unchanged."
    requests = {"one": {"input_items": [quote], "target_compound": "caffeine"}}
    responses = [{"request_id": "one", "status": "ok",
                  "candidates": [{"source_text": quote, "parameter": "urinary_excretion"}]}]
    assert "accepted_evidence_joint_chemical_subject" in watch.evidence_audit(requests, responses)["reasons"]


def test_monitor_stops_metabolite_parameter_misassigned_to_parent():
    quote = "This metabolite was excreted unchanged and was an end product of caffeine metabolism."
    requests = {"one": {"input_items": [quote], "target_compound": "caffeine"}}
    responses = [{"request_id": "one", "status": "ok", "candidates": [{
        "source_text": quote, "parameter": "excretion", "evidence_type": "adme"}]}]
    assert "accepted_evidence_parameter_owner_mismatch" in watch.evidence_audit(requests, responses)["reasons"]


LATER_ATTRIBUTION = "01_compound_attribution/llm_attribution_responses.jsonl"
LATER_QUALITY = "07_llm_quality_validation/llm_quality_results.jsonl"


def write_later_results(root, relative_path, rows):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_later_llm_monitor_accepts_terminal_scientific_exclusions_and_reads_only_new_rows(tmp_path):
    write_later_results(tmp_path, LATER_ATTRIBUTION, [{
        "evidence_id": "a", "raw_response": {"status": "ok"},
        "validation": {"status": "valid", "candidate": {"compound_role": "unrelated"}},
    }])
    path = write_later_results(tmp_path, LATER_QUALITY, [
        {"evidence_id": str(index), "terminal": True, "status": status}
        for index, status in enumerate([
            "auto_validated", "quarantined_unrepairable", "quarantined_validation_failed", "excluded",
        ])
    ])
    cursors = {}
    first = watch.later_llm_audit(tmp_path, cursors)
    assert first == {"attribution_responses_checked": 1, "quality_results_checked": 4, "reasons": []}
    assert watch.later_llm_audit(tmp_path, cursors) == {
        "attribution_responses_checked": 0, "quality_results_checked": 0, "reasons": [],
    }
    with path.open("a") as handle:
        handle.write(json.dumps({"evidence_id": "failed", "terminal": False, "status": "local_llm_error"}) + "\n")
    check = watch.later_llm_audit(tmp_path, cursors)
    assert check["quality_results_checked"] == 1
    assert check["reasons"] == ["llm_quality_request_failure_strict_stop"]
    assert all(set(cursor) == {"identity", "offset", "rows"} for cursor in cursors.values())


@pytest.mark.parametrize("row,reason", [
    ({"raw_response": {"status": "failed_local_llm"}, "validation": {"status": "invalid"}},
     "llm_attribution_request_failure_strict_stop"),
    ({"raw_response": {"status": "ok"}, "validation": {"status": "invalid", "reason": "matched_text_not_verbatim"}},
     "llm_attribution_validation_failure_strict_stop"),
])
def test_later_attribution_monitor_stops_failed_calls_or_invalid_model_decisions(tmp_path, row, reason):
    write_later_results(tmp_path, LATER_ATTRIBUTION, [{"evidence_id": "a", **row}])
    assert watch.later_llm_audit(tmp_path, {})["reasons"] == [reason]


@pytest.mark.parametrize("status,terminal", [
    ("local_llm_error", False), ("local_llm_error", True),
    ("auto_validated", False), ("unknown_result", True),
])
def test_later_quality_monitor_stops_nonterminal_or_unknown_failures(tmp_path, status, terminal):
    write_later_results(tmp_path, LATER_QUALITY, [
        {"evidence_id": "a", "status": status, "terminal": terminal},
    ])
    assert watch.later_llm_audit(tmp_path, {})["reasons"] == ["llm_quality_request_failure_strict_stop"]


def test_later_llm_monitor_does_not_infer_historical_recovery_from_evidence_id(tmp_path):
    write_later_results(tmp_path, LATER_QUALITY, [
        {"evidence_id": "a", "status": "local_llm_error", "terminal": False},
        {"evidence_id": "a", "status": "auto_validated", "terminal": True},
    ])
    assert watch.later_llm_audit(tmp_path, {})["reasons"] == ["llm_quality_request_failure_strict_stop"]


def test_later_llm_monitor_waits_for_complete_lines_and_reaudits_atomic_replacement(tmp_path):
    path = write_later_results(tmp_path, LATER_ATTRIBUTION, [{
        "evidence_id": "a", "raw_response": {"status": "ok"}, "validation": {"status": "valid"},
    }])
    cursors = {}
    assert watch.later_llm_audit(tmp_path, cursors)["reasons"] == []
    with path.open("a") as handle:
        handle.write('{"evidence_id":"b",')
    assert watch.later_llm_audit(tmp_path, cursors)["attribution_responses_checked"] == 0
    with path.open("a") as handle:
        handle.write('"raw_response":{"status":"ok"},"validation":{"status":"valid"}}\n')
    assert watch.later_llm_audit(tmp_path, cursors)["attribution_responses_checked"] == 1
    replacement = write_later_results(tmp_path, "replacement.jsonl", [{
        "evidence_id": "b", "raw_response": {"status": "failed_local_llm"},
    }])
    replacement.replace(path)
    check = watch.later_llm_audit(tmp_path, cursors)
    assert check["attribution_responses_checked"] == 1
    assert check["reasons"] == ["llm_attribution_request_failure_strict_stop"]


def test_later_llm_monitor_bounds_lines_and_rejects_unexpected_truncation(tmp_path):
    path = write_later_results(tmp_path, LATER_QUALITY, [
        {"evidence_id": "a", "status": "auto_validated", "terminal": True},
    ])
    cursors = {}
    watch.later_llm_audit(tmp_path, cursors)
    path.write_text("")
    with pytest.raises(ValueError, match="shrank"):
        watch.later_llm_audit(tmp_path, cursors)
    path.write_bytes(b"x" * (2 * 1024 ** 2))
    with pytest.raises(ValueError, match="Oversized"):
        watch.later_llm_audit(tmp_path, {})


def test_later_llm_monitor_bounds_total_decisions_and_rejects_disappearing_ledger(tmp_path):
    path = write_later_results(tmp_path, LATER_QUALITY, [
        {"evidence_id": "a", "status": "auto_validated", "terminal": True},
    ])
    stat = path.stat()
    cursors = {"quality": {"identity": (stat.st_dev, stat.st_ino), "offset": 0, "rows": 100000}}
    with pytest.raises(ValueError, match="Excessive"):
        watch.later_llm_audit(tmp_path, cursors)
    path.unlink()
    with pytest.raises(ValueError, match="disappeared"):
        watch.later_llm_audit(tmp_path, cursors)


def test_historical_table_deferral_requires_current_routing_and_exact_input():
    from hashlib import sha256
    from pbpk_kb.processing.run_llm_extraction import LLM_ROUTING_VERSION
    text = "Renal-impaired adults: 1.86 (±0.52) 3.20 (±0.45) 384 (±122). Metformin Cmax was compared in children."
    fingerprint = {"target_compound": "metformin", "input_text_or_table": text}
    digest = sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request = {**fingerprint, "request_id": "one", "input_hash": digest, "source_id": "dailymed"}
    response = {"request_id": "one", "input_hash": digest, "status": "failed_local_llm", "candidates": [],
                "resolved": True, "resolution": "no_longer_eligible_under_current_routing",
                "resolution_destination": "table_recovery", "resolution_routing_version": LLM_ROUTING_VERSION}
    assert watch.verified_table_deferral(request, response)
    # Older reviewed deferrals survive a newer classifier only after the exact
    # original fragment is independently rechecked by the current rules.
    assert watch.verified_table_deferral(request, {**response, "resolution_routing_version": "pbpk-llm-routing-v1"})
    assert not watch.verified_table_deferral(request, {**response, "resolution_routing_version": "pbpk-llm-routing-v999"})
    assert watch.evidence_audit({"one": request}, [response], historical=True)["reasons"] == []
    assert watch.evidence_audit({"one": request}, [response])["reasons"] == ["llm_request_failure_strict_stop"]
    for changed in [{"input_hash": "wrong"}, {"resolution_routing_version": "old"}, {"resolved": False}]:
        assert not watch.verified_table_deferral(request, {**response, **changed})
    assert not watch.verified_table_deferral({**request, "parsed_table_id": "already-parsed"}, response)
    assert not watch.verified_table_deferral({**request, "input_text_or_table": "Metformin clearance was 12 L/h."}, response)


def test_monitor_stops_comparator_binding_percentage_without_rejecting_target_qualitative_fact():
    quote = "Metformin is negligibly bound to plasma proteins, in contrast to sulfonylureas, which are more than 90% protein bound."
    request = {"one": {"target_compound": "metformin", "input_items": [quote]}}
    candidate = {"parameter": "protein_binding", "evidence_type": "adme", "source_text": quote,
                 "value": 90.0, "unit": "%", "value_qualifier": ">"}
    response = {"request_id": "one", "status": "ok", "candidates": [candidate]}
    assert "accepted_evidence_numeric_owner_mismatch" in watch.evidence_audit(request, [response])["reasons"]
    qualitative = {**candidate, "value": None, "unit": None, "value_qualifier": None}
    assert watch.evidence_audit(request, [{**response, "candidates": [qualitative]}])["reasons"] == []
