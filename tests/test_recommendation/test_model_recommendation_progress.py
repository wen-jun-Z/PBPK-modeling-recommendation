"""CPU checks for read-only progress accounting, independent of model runners."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/model_recommendation_progress.py"
SPEC = importlib.util.spec_from_file_location("recommendation_progress_under_test", SCRIPT)
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


def record(case="a", protocol="p", status="valid"):
    return json.dumps({"case_id": case, "pilot_protocol_sha256": protocol,
                       "final_status": status}).encode() + b"\n"


@pytest.fixture
def counter(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    path = tmp_path / "records.jsonl"
    return progress.RecordCounter(path, ("case_id",), {("a",), ("b",)}, "final_status")


def test_only_complete_unique_in_scope_protocol_bound_lines_count(counter):
    complete = record("a")
    pending = record("b", status="invalid_output")[:-1]
    counter.path.write_bytes(complete + pending)
    result = counter.update("p")
    assert result["done"] == 1
    assert result["pending_tail_bytes"] == len(pending)
    assert counter.update("p") == result
    with counter.path.open("ab") as handle:
        handle.write(b"\n" + record("a") + record("foreign") + record("a", protocol="other"))
    result = counter.update("p")
    assert result["done"] == 2
    assert result["statuses"] == {"valid": 1, "invalid_output": 1}
    assert result["duplicates"] == 1
    assert result["malformed_or_mismatched"] == 2
    assert result["pending_tail_bytes"] == 0


def test_truncate_replace_and_protocol_change_reset_old_counts(counter):
    counter.path.write_bytes(record("a") + record("b"))
    assert counter.update("p")["done"] == 2
    counter.path.write_bytes(record("b"))
    assert counter.update("p")["done"] == 1
    assert counter.seen == {("b",)}
    replacement = counter.path.with_suffix(".replacement")
    replacement.write_bytes(record("a"))
    replacement.replace(counter.path)
    assert counter.update("p")["done"] == 1
    assert counter.seen == {("a",)}
    assert counter.update("different_protocol")["done"] == 0
    counter.path.unlink()
    assert counter.update("p")["malformed_or_mismatched"] == 0


def test_unrecognized_final_status_is_not_a_completed_judgment(counter):
    counter.path.write_bytes(record(status="unfinished_or_foreign"))
    result = counter.update("p")
    assert result["done"] == 0
    assert result["malformed_or_mismatched"] == 1


def test_wrong_judge_is_excluded_but_processed_unknown_is_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    path = tmp_path / "judgments.jsonl"
    counter = progress.RecordCounter(path, ("query_id", "candidate_id"),
        {("q", "c")}, "status", expected_judge="primary")
    row = {"query_id": "q", "candidate_id": "c", "protocol_sha256": "p",
           "status": "insufficient_evidence", "judge": "secondary"}
    path.write_text(json.dumps(row) + "\n")
    result = counter.update("p")
    assert result["done"] == 0
    assert result["malformed_or_mismatched"] == 1
    row["judge"] = "primary"
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    result = counter.update("p")
    assert result["done"] == 1
    assert result["statuses"] == {"insufficient_evidence": 1}


def test_outside_project_symlink_is_rejected(counter, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    counter.path = project / "records.jsonl"
    target = tmp_path / "foreign.jsonl"
    target.write_bytes(record())
    counter.path.symlink_to(target)
    monkeypatch.setattr(progress, "ROOT", project)
    with pytest.raises(ValueError, match="outside"):
        counter.update("p")


def test_all_records_do_not_finish_without_review_and_final_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    monkeypatch.setattr(progress, "QUAL", tmp_path / "qualification")
    monkeypatch.setattr(progress, "QUAL_RUN", tmp_path / "qualification_run")
    monkeypatch.setattr(progress, "ASSESS", tmp_path / "assessment")
    monkeypatch.setattr(progress, "RUN", tmp_path / "run")
    monkeypatch.setattr(progress, "read_json", lambda path: {"state": "complete"} if path.name == "pipeline_status.json" else {})
    monkeypatch.setattr(progress, "file_sha", lambda path: None)
    monkeypatch.setattr(progress, "process_alive", lambda saved: False)
    monitor = object.__new__(progress.Monitor)
    monitor.totals = {"qwen": 2, "phi": 2, "primary": 1, "secondary": 1}
    monitor.review_ids = {"review_case"}
    keys = [(model, cohort) for model in ("qwen", "phi") for cohort in ("validation", "validation/fresh")]
    keys += ["primary", "secondary"]
    summary = {"done": 1, "statuses": {"valid": 1}, "duplicates": 0,
               "malformed_or_mismatched": 0, "pending_tail_bytes": 0, "last_record_at": None}
    monitor.counters = {key: SimpleNamespace(update=lambda digest: dict(summary)) for key in keys}
    monitor.protocol_paths = {key: tmp_path / "protocol.json" for key in keys}
    result = monitor.snapshot()
    assert result["done"] == result["total"] == 6
    assert result["complete"] is False
    assert result["review_accepted"] is False
    assert result["evaluation_complete"] is False


def test_qualification_requires_actual_frozen_file_bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    qual = tmp_path / "qualification"
    monkeypatch.setattr(progress, "QUAL", qual)
    required = [qual / "judge_protocol.json"] + [
        qual / folder / name for folder in ("validation", "validation/fresh")
        for name in ("manifest.json", "qwen_records.jsonl", "phi_records.jsonl",
                     "qwen_raw_generation_journal.jsonl", "phi_raw_generation_journal.jsonl")]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"artifact": str(path.relative_to(qual))}) + "\n")
    review = {"accepted": True,
              "protocol_sha256": progress.file_sha(qual / "judge_protocol.json"),
              "validation_manifest_sha256": progress.file_sha(qual / "validation/manifest.json"),
              "fresh_manifest_sha256": progress.file_sha(qual / "validation/fresh/manifest.json")}
    for field, folder in (("pilot_record_sha256", "validation"),
                          ("fresh_pilot_record_sha256", "validation/fresh")):
        review[field] = {model: progress.file_sha(qual / folder / f"{model}_records.jsonl")
                         for model in ("qwen", "phi")}
    review_path = qual / "source_review.json"
    review_path.write_text(json.dumps(review) + "\n")
    required.append(review_path)
    acceptance = {"accepted": True, "numeric_accepted": True, "cases_per_model": 765,
                  "profile": "v8", "source_review_sha256": progress.file_sha(review_path),
                  "files": {str(path.relative_to(tmp_path)): progress.file_sha(path) for path in required}}
    phases = {model: {"done": 765, "total": 765} for model in ("qwen", "phi")}
    assert progress.qualification_ok(review, acceptance, phases)
    phases["phi"]["done"] = 764
    assert not progress.qualification_ok(review, acceptance, phases)
    phases["phi"]["done"] = 765
    binding = str((qual / "validation/phi_records.jsonl").relative_to(tmp_path))
    digest = acceptance["files"].pop(binding)
    assert not progress.qualification_ok(review, acceptance, phases)
    acceptance["files"][binding] = digest
    (qual / "validation/phi_records.jsonl").write_text("changed after acceptance\n")
    assert not progress.qualification_ok(review, acceptance, phases)


def test_foreign_uid_pid_does_not_read_cwd_or_stat_contents(monkeypatch):
    class ForeignProcess:
        def stat(self):
            return SimpleNamespace(st_uid=456)

        def __truediv__(self, name):
            raise AssertionError("Must not read another user's cwd/stat")

    class ProcRoot:
        def __truediv__(self, name):
            assert name == "123"
            return ForeignProcess()

    monkeypatch.setattr(progress, "Path", lambda name: ProcRoot())
    monkeypatch.setattr(progress.os, "getuid", lambda: 123)
    assert progress.process_alive({"pid": 123, "start_ticks": "456"}) is False


def test_repair_layout_requires_explicit_frozen_version(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    out = tmp_path / "core"
    monkeypatch.setattr(progress, "OUT", out)
    for name in ("ASSESS", "QUAL", "RUN", "QUAL_RUN"):
        monkeypatch.setattr(progress, name, tmp_path / ("old_" + name))
    old = progress.ASSESS
    progress.select_layout()
    assert progress.ASSESS == old
    plan = out / "assessment_repaired/core_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"version": "unrecognized"}))
    progress.select_layout()
    assert progress.ASSESS == old
    plan.write_text(json.dumps({"version": progress.REPAIR_PROFILE}))
    progress.select_layout()
    assert progress.ASSESS == out / "assessment_repaired"
    assert progress.QUAL == out / "qualification_repaired"
    assert progress.RUN == tmp_path / "runtime/deployment/frontier_core_145_repair"
    assert progress.QUAL_RUN == progress.RUN / "qualification"
    newest = out / "assessment_repaired_v2/core_plan.json"
    newest.parent.mkdir(parents=True)
    newest.write_text(json.dumps({"version": "core145-repair-v2"}))
    progress.select_layout()
    assert progress.ASSESS == out / "assessment_repaired_v2"
    assert progress.QUAL == out / "qualification_repaired_v2"
    assert progress.RUN == tmp_path / "runtime/deployment/frontier_core_145_repair_v2"
    assert progress.QUAL_RUN == progress.RUN / "qualification"


def test_derived_reuse_is_distinct_from_new_gpu_generation(counter):
    rows = []
    for case, origins in (("a", ["v8_initial"]), ("b", ["v8_initial", "new_ordinary", "new_focused"])):
        row = json.loads(record(case))
        row.update(derivation_profile=progress.REPAIR_PROFILE,
                   attempts=[{"generation_origin": origin} for origin in origins])
        rows.append(json.dumps(row) + "\n")
    counter.path.write_text("".join(rows))
    result = counter.update("p")
    assert result["done"] == 2
    assert result["origins"] == {"derived": 2, "reused_without_new_generation": 1, "with_new_generation": 1}


def test_gpu_repair_journal_checks_model_kind_and_committed_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    path = tmp_path / "qwen_focused_repair_journal.jsonl"
    counter = progress.RecordCounter(path, ("case_id",), {("a",), ("b",)}, None,
        expected_fields={"model": "qwen", "repair_kind": "focused"})
    row = {"case_id": "a", "pilot_protocol_sha256": "p", "model": "qwen", "repair_kind": "focused"}
    wrong = {**row, "case_id": "b", "model": "phi"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(wrong) + "\n")
    result = counter.update("p")
    assert result["done"] == 1
    assert result["statuses"] == {"generated": 1}
    assert result["malformed_or_mismatched"] == 1


@pytest.mark.parametrize("profile,suffix", [(progress.REPAIR_PROFILE, ""), ("core145-repair-v2", "_v2")])
def test_repair_acceptance_requires_bound_actual_replays_and_journals(tmp_path, monkeypatch, profile, suffix):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    qual, assess = tmp_path / ("qualification_repaired" + suffix), tmp_path / ("assessment_repaired" + suffix)
    monkeypatch.setattr(progress, "QUAL", qual)
    monkeypatch.setattr(progress, "ASSESS", assess)
    required = [qual / "judge_protocol.json", assess / "core_plan.json"] + [
        qual / folder / name for folder in ("validation", "validation/fresh")
        for name in ("cases.jsonl", "manifest.json", "qwen_records.jsonl", "phi_records.jsonl",
                     "qwen_raw_generation_journal.jsonl", "phi_raw_generation_journal.jsonl")]
    repair_journal = qual / "validation/qwen_focused_repair_journal.jsonl"
    required.append(repair_journal)
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": profile, "artifact": path.name}) + "\n")
    review = {"accepted": True, "profile": profile,
              "protocol_sha256": progress.file_sha(qual / "judge_protocol.json"),
              "validation_manifest_sha256": progress.file_sha(qual / "validation/manifest.json"),
              "fresh_manifest_sha256": progress.file_sha(qual / "validation/fresh/manifest.json")}
    for field, folder in (("pilot_record_sha256", "validation"), ("fresh_pilot_record_sha256", "validation/fresh")):
        review[field] = {model: progress.file_sha(qual / folder / f"{model}_records.jsonl") for model in ("qwen", "phi")}
    review_path = qual / "source_review.json"
    review_path.write_text(json.dumps(review) + "\n")
    required.append(review_path)
    acceptance = {"version": profile, "accepted": True, "numeric_accepted": True,
                  "cases_per_model": 765, "scientific_gold_accuracy": None,
                  "source_review_sha256": progress.file_sha(review_path),
                  "files": {str(path.relative_to(tmp_path)): progress.file_sha(path) for path in required}}
    phases = {model: {"done": 765, "total": 765} for model in ("qwen", "phi")}
    assert progress.qualification_ok(review, acceptance, phases)
    key = str(repair_journal.relative_to(tmp_path))
    digest = acceptance["files"].pop(key)
    assert not progress.qualification_ok(review, acceptance, phases)
    acceptance["files"][key] = digest
    repair_journal.write_text("mutated actual repair\n")
    assert not progress.qualification_ok(review, acceptance, phases)


def test_repair_throughput_does_not_count_cpu_replayed_records(monkeypatch):
    clock = iter([0, 60, 120])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(clock))
    snapshot = {"pipeline": {"stage": "pilot_qwen", "state": "running"},
                "worker": {"pid": 123, "start_ticks": "456", "state": "judging"},
                "pipeline_alive": True, "worker_alive": True,
                "repair": {"profile": progress.REPAIR_PROFILE},
                "phases": {"qwen": {"done": 700, "total": 765, "new_gpu_generations": 0}}}
    throughput = progress.Throughput()
    assert throughput.update(snapshot)["rate"] is None
    snapshot["phases"]["qwen"].update(done=750, new_gpu_generations=1)
    assert throughput.update(snapshot)["rate"] is None
    snapshot["phases"]["qwen"].update(done=765, new_gpu_generations=2)
    result = throughput.update(snapshot)
    assert result["rate"] == pytest.approx(2 / 120)
    assert result["eta"] is None
    assert "reused outputs excluded from throughput" in result["note"]


def test_v2_reused_v1_call_is_not_a_new_gpu_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    path = tmp_path / "qwen_focused_repair_journal.jsonl"
    counter = progress.RecordCounter(path, ("case_id",), {("a",), ("b",)}, None,
        expected_fields={"model": "qwen", "repair_kind": "focused"})
    common = {"pilot_protocol_sha256": "v2", "model": "qwen", "repair_kind": "focused"}
    reused = {**common, "case_id": "a", "generation_origin": "reused_actual_v1_generation",
              "reused_from": {"journal_path": "parent.jsonl", "journal_sha256": "source"}}
    generated = {**common, "case_id": "b"}
    path.write_text(json.dumps(reused) + "\n" + json.dumps(generated) + "\n")
    result = counter.update("v2")
    assert result["done"] == 2
    assert result["statuses"] == {"reused_generation": 1, "generated": 1}


def test_v2_derived_case_with_only_v1_calls_is_reused(counter):
    row = json.loads(record("a"))
    row.update(derivation_profile="core145-repair-v2", attempts=[
        {"generation_origin": "v8_initial"}, {"generation_origin": "reused_actual_v1_generation"}])
    counter.path.write_text(json.dumps(row) + "\n")
    result = counter.update("p")
    assert result["done"] == 1
    assert result["origins"] == {"derived": 1, "reused_without_new_generation": 1}
