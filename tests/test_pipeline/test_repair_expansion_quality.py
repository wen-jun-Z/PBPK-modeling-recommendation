"""Isolation, resumability and bounded execution of new-batch evidence repairs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("repair_controller_tests", PROJECT / "scripts/deployment/repair_expansion_quality.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "BASE", tmp_path / "repair")
    return tmp_path


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_boundary_blocks_old_batches_and_symlink_writes(isolated):
    with pytest.raises(ValueError, match="seven"):
        controller.paths("run_20260907T134918Z")
    original = isolated / "original.jsonl"
    original.write_text("original\n")
    target = controller.BASE / "symlink.jsonl"
    target.parent.mkdir()
    target.symlink_to(original)
    with pytest.raises(ValueError, match="boundary"):
        controller.atomic_rows(target, [{"new": True}])
    assert original.read_text() == "original\n"
    with pytest.raises(ValueError, match="boundary"):
        controller.inside(isolated.parent / "other_project")


def test_seed_keeps_llm_but_rebuilds_tables_and_does_not_mutate_sources(isolated):
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    run = controller.RUNS[0]
    paths = controller.paths(run)
    layout = raw_processing_layout(paths["original_raw"])
    layout.ensure_dirs()
    state = {"run_id": run, "output_dir": str(paths["original_raw"]), "stages": {
        stage: {"status": "completed", "accepted": True}
        for stage in ("inventory", "parse", "deterministic_extract", "llm_extract", "table_recovery", "completeness")}}
    write(layout.path("pipeline_state.json"), state)
    for name in controller.READ_ONLY_INPUTS:
        write_rows(layout.path(name), [{"input": name}])
    write_rows(layout.path("evidence_candidates_deterministic.jsonl"), [{"evidence_id": "rule"}])
    write_rows(layout.path("evidence_candidates_llm.jsonl"), [
        {"evidence_id": "llm"}, {"evidence_id": "old_table", "recovered_table_id": "table1"}])
    write_rows(layout.path("chunk_processing_status.jsonl"), [{"status": "old"}])
    write_rows(paths["original_final"] / "accepted.jsonl", [{"original_final": True}])
    write(isolated / "data_process/pipeline_runs" / f"{run}.json", {"run_id": run})
    receipt = controller.seed_inputs(run)
    assert receipt["preserved_seed_count"] == 2
    assert receipt["excluded_old_table_candidates"] == 1
    new = raw_processing_layout(paths["raw"])
    assert new.path("parsed_documents.jsonl").is_symlink()
    assert not new.path("chunk_processing_status.jsonl").is_symlink()
    assert {row["evidence_id"] for row in controller.rows(new.path("evidence_candidates.jsonl"))} == {"rule", "llm"}
    fresh = controller.read_json(new.path("pipeline_state.json"))
    assert fresh["stages"]["llm_extract"] == {"status": "not_started"}
    assert fresh["stages"]["table_recovery"] == {"status": "not_started"}
    assert controller.read_json(layout.path("pipeline_state.json")) == state
    assert controller.verify_originals(run)["accepted"]
    assert controller.seed_inputs(run)["reused"]
    layout.path("parsed_documents.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        controller.verify_originals(run)


def test_pilot_requires_passed_version_and_current_code_hash(isolated):
    source = isolated / "src/pbpk_kb/processing/run_llm_extraction.py"
    source.parent.mkdir(parents=True)
    source.write_text("validated code")
    pilot = controller.BASE / "pilot_v5/receipt.json"
    receipt = {"passed": True, "results": [], "code_sha256": {str(source.relative_to(isolated)): controller.digest(source)}}
    write(pilot, receipt)
    assert controller.validate_pilot(pilot)["version"] == 5
    source.write_text("changed code")
    with pytest.raises(ValueError, match="changed"):
        controller.validate_pilot(pilot)
    receipt["passed"] = False
    write(pilot, receipt)
    with pytest.raises(ValueError, match="not passed"):
        controller.validate_pilot(pilot)


def test_internal_worker_requires_controller_parent(isolated, monkeypatch):
    monkeypatch.setattr(controller, "validate_pilot", lambda *args: {})
    monkeypatch.setattr(controller, "same_process", lambda *args: False)
    with pytest.raises(RuntimeError, match="active project controller"):
        controller.worker_main(SimpleNamespace(pilot_receipt="unused", stage="prepare_inputs", run_id=[controller.RUNS[0]]))


@pytest.mark.parametrize("total,log,free,previous,elapsed,expected", [
    (101, 0, 100, 0, 1, "isolated_output_size_limit"),
    (1, 65*controller.MIB, 100, 0, 1, "stage_log_size_limit"),
    (1, 0, 1, 0, 1, "free_disk_below_safety_reserve"),
    (21, 0, 100, 0, 1200, "excessive_output_growth_in_20_minutes"),
    (21, 0, 100, 0, 1199, None),
])
def test_disk_growth_is_bounded(total, log, free, previous, elapsed, expected):
    assert controller.capacity_problem(total, log, free, previous=previous, elapsed=elapsed,
                                        max_bytes=100, max_growth=20, min_free=10) == expected


def test_disk_budget_does_not_count_reused_source_targets(isolated):
    source = isolated / "large_source"
    source.write_bytes(b"x" * 1000)
    output = controller.BASE / "reprocessed"
    output.mkdir(parents=True)
    (output / "reused").symlink_to(source)
    (output / "own.log").write_bytes(b"abc")
    assert controller.owned_bytes(output) == (3, 3)


def test_llm_stall_requires_both_no_progress_and_no_cpu_activity():
    kwargs = {"elapsed": 2500, "since_progress": 2500, "since_activity": 2500, "timeout_seconds": 7200}
    assert controller.stalled_stage("llm_extract", **kwargs) == "llm_no_response_or_cpu_progress_for_40_minutes"
    assert controller.stalled_stage("classify", **{**kwargs, "since_activity": 10}) is None
    assert controller.stalled_stage("semantic_dedup", **kwargs) is None
    assert controller.stalled_stage("semantic_dedup", **{**kwargs, "elapsed": 7201}) == "explicit_stage_time_limit"


def test_service_failure_streak_uses_new_responses_and_not_scientific_rejections(isolated):
    path = isolated / "response.jsonl"
    write_rows(path, [{"error": "HTTP 503"}] * 3)
    initial = path.stat().st_size
    assert controller.service_failure_streak(path)
    assert not controller.service_failure_streak(path, initial_bytes=initial)
    write_rows(path, [{"status": "quarantined_validation_failed", "error": "unsupported numeric assertion"}] * 3)
    assert not controller.service_failure_streak(path)


def test_first_response_display_uses_current_worker_requests_not_backend_alone(isolated):
    run = controller.RUNS[0]
    selected = controller.paths(run)
    state = {"current_run": run, "current_stage": "llm_extract", "worker_alive": True, "pilot": {"prompt": "current"}}
    progress = {"status": "building_high_precision_routing_index", "api_requests_completed": 0}
    write(selected["state"], {"stages": {"llm_extract": {"started_at": "2026-10-14T16:00:00+00:00"}}})
    write(isolated / "runtime/deployment/evidence_expansion_20261009/backend/status.json", {"state": "generating"})
    request_path = selected["raw"] / "05_llm_extraction/llm_extraction_requests_live.jsonl"
    write_rows(request_path, [{"prompt_template_version": "current", "created_at": "2026-10-14T15:59:00+00:00"},
                              {"prompt_template_version": "old", "created_at": "2026-10-14T16:01:00+00:00"}])
    assert controller.display_llm_phase(state, progress) == progress
    write_rows(request_path, [{"prompt_template_version": "current", "created_at": "2026-10-14T16:01:00+00:00"}])
    shown = controller.display_llm_phase(state, progress)
    assert shown["status"] == "waiting_for_llm_response"
    assert shown["requests_observed_in_tail"] == 1
    assert progress["status"] == "building_high_precision_routing_index"
    assert controller.display_llm_phase({**state, "worker_alive": False}, progress) == progress


def test_status_tail_reader_is_bounded_and_ignores_partial_lines(isolated):
    path = isolated / "requests.jsonl"
    write_rows(path, [{"old": "x" * 1024}, {"new": 1}, {"new": 2}])
    assert controller.latest_json_rows(path, max_bytes=100) == [{"new": 1}, {"new": 2}]


def test_final_context_guard_preserves_partition_and_is_resumable(isolated, monkeypatch):
    from pbpk_kb.processing import source_context, final_evidence_quality_stage, auto_quality_stage
    run = controller.RUNS[0]
    quality = controller.paths(run)["final"] / "07_llm_quality_validation"
    values = [{"evidence_id": identity, "quality_status": "auto_validated", "run_id": run}
              for identity in ("good", "animal", "metabolite", "ambiguous")]
    write_rows(quality / "final_auto_validated_evidence.jsonl", values)
    write_rows(quality / "final_quarantined_evidence.jsonl", [{"evidence_id": "old_quarantine"}])
    write(quality / "quality_summary.json", {"accepted": True, "final_auto_validated_count": 4, "final_quarantined_count": 1})
    def enrich(row, **kwargs):
        return {**row, "metadata_json": {"source_context": {"match_status": "ambiguous" if row["evidence_id"] == "ambiguous" else "matched"}}}
    def gate(row):
        decision = {"animal": "needs_manual_review", "metabolite": "excluded"}.get(row["evidence_id"], "passed")
        return {"domain_gate_status": decision, "domain_classification": "fixture", "reasons": [] if decision == "passed" else [decision]}
    monkeypatch.setattr(source_context, "enrich_source_context", enrich)
    monkeypatch.setattr(final_evidence_quality_stage, "evaluate_domain_gate", gate)
    monkeypatch.setattr(auto_quality_stage, "build_kg_ready", lambda values: (values, []))
    result = controller.final_context_guard(run)
    assert result["retained_auto_validated_count"] == 1
    assert result["review_count"] == 2
    assert result["excluded_count"] == 1
    assert controller.final_context_guard(run) == result
    summary = controller.read_json(quality / "quality_summary.json")
    assert summary["final_auto_validated_count"] == 1
    assert summary["final_quarantined_count"] == 4
    assert summary["partition_complete"]
    valid_ids = {row["evidence_id"] for row in controller.rows(quality / "final_auto_validated_evidence.jsonl")}
    quarantine_ids = {row["evidence_id"] for row in controller.rows(quality / "final_quarantined_evidence.jsonl")}
    assert valid_ids == {"good"}
    assert quarantine_ids == {"old_quarantine", "animal", "metabolite", "ambiguous"}


def test_db_stage_never_applies(monkeypatch, isolated):
    import sys
    called = []
    monkeypatch.setitem(sys.modules, "revise_expansion_evidence", SimpleNamespace(
        revise=lambda run_id, apply: called.append((run_id, apply)) or {"checks": "passed"},
        input_fingerprints=lambda *args: {"fixture": "hash"}))
    monkeypatch.setattr(controller, "verify_originals", lambda *a, **k: {})
    result = controller.do_stage(controller.RUNS[0], "database_dry_run", 2)
    assert called == [(controller.RUNS[0], False)]
    assert result["database_modified"] is False


def test_application_is_opt_in_and_runs_only_after_the_complete_dry_run(monkeypatch):
    assert len(controller.stages_for()) == 17
    assert len(controller.stages_for(True)) == 19
    assert controller.stages_for(True)[-3:] == ("database_dry_run", "database_apply", "verified_retrieval")
    monkeypatch.setattr(controller, "verify_originals", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="explicit --apply-validated"):
        controller.do_stage(controller.RUNS[0], "database_apply", 2)


def test_application_rejects_an_unaccepted_prior_stage(isolated):
    with pytest.raises(RuntimeError, match="intact accepted stage: prepare_inputs"):
        controller.verify_apply_prerequisites(controller.RUNS[0])


def test_application_requires_the_same_dry_run_and_calls_apply_once(isolated, monkeypatch):
    import sys
    called = []
    dry = {"revision_plan": {"accepted_count": 12}, "input_sha256": {"fixture": "hash"}}
    monkeypatch.setattr(controller, "verify_apply_prerequisites", lambda run: dry)
    monkeypatch.setattr(controller, "recover_committed_revision", lambda *args: None)
    def revise(run, apply):
        called.append(apply)
        return {"protected_rows_unchanged": True, "revision_id": "fixture"} if apply else dry["revision_plan"]
    monkeypatch.setitem(sys.modules, "revise_expansion_evidence", SimpleNamespace(
        revise=revise, input_fingerprints=lambda *args: dry["input_sha256"]))
    assert controller.apply_validated_database(controller.RUNS[0])["accepted"]
    assert called == [False, True]
    called.clear()
    monkeypatch.setitem(sys.modules, "revise_expansion_evidence", SimpleNamespace(
        revise=lambda run, apply: called.append(apply) or {"accepted_count": 99},
        input_fingerprints=lambda *args: dry["input_sha256"]))
    with pytest.raises(RuntimeError, match="plan changed"):
        controller.apply_validated_database(controller.RUNS[0])
    assert called == [False]


def test_application_reconciles_committed_revision_without_a_second_write(isolated, monkeypatch):
    import sys
    monkeypatch.setattr(controller, "verify_apply_prerequisites", lambda run: {"fixture": True})
    recovered = {"accepted": True, "protected_rows_unchanged": True, "recovered_committed_revision": True}
    monkeypatch.setattr(controller, "recover_committed_revision", lambda *args: recovered)
    def should_not_call(*args, **kwargs):
        raise AssertionError("A committed transaction must not run again")
    monkeypatch.setitem(sys.modules, "revise_expansion_evidence", SimpleNamespace(revise=should_not_call, input_fingerprints=should_not_call))
    assert controller.apply_validated_database(controller.RUNS[0]) == recovered


@pytest.mark.parametrize("interruption", ["during_prepare", "after_matrix", "after_summary"])
def test_rag_recovery_of_interrupted_atomic_bundle(isolated, monkeypatch, interruption):
    from pbpk_kb.processing import rag_index_stage
    final = controller.paths(controller.RUNS[0])["final"]
    called = []
    class FakeRag:
        def __init__(self, **kwargs):
            self.records_path = final / "06_rag_index/retrieval_records.jsonl"
            self.embeddings_path = final / "06_rag_index/retrieval_embeddings.npy"
            self.summary_path = final / "06_rag_index/rag_index_summary.json"
            self.state_path = final / "00_run_manifest/pipeline_state.json"
            self.reset = False
            write_rows(self.records_path, [{"evidence_id": "e1"}])
            summary = {"records_sha256": controller.digest(self.records_path)}
            state = {"stages": {"rag_index": {"status": "prepared"}}}
            if interruption == "during_prepare":
                state["stages"]["rag_index"]["status"] = "not_started"
            else:
                self.embeddings_path.write_bytes(b"matrix")
            if interruption == "after_summary":
                summary["accepted"] = True
            write(self.summary_path, summary)
            write(self.state_path, state)
        def prepare(self):
            assert self.reset is True
            called.append("prepare")
            return {"accepted": False}
        def embed(self):
            assert not self.embeddings_path.exists()
            called.append("embed")
            return {"accepted": True}
        def audit(self):
            called.append("audit")
            return {"accepted": True}
        def _update_state_completed(self, state, summary):
            called.append("reconcile_completed_state")
    monkeypatch.setattr(rag_index_stage, "RagIndexStage", FakeRag)
    pipeline = SimpleNamespace(raw_processing_dir=controller.paths(controller.RUNS[0])["raw"], final_evidence_dir=final,
                               config=SimpleNamespace(embedding_model="fixture", embedding_device="cuda", allow_cpu_embedding=False, embedding_batch_size=64))
    result = controller.resumable_rag_index(pipeline)
    assert result["accepted"]
    assert called == {"during_prepare": ["prepare", "embed"], "after_matrix": ["embed"],
                      "after_summary": ["audit", "reconcile_completed_state"]}[interruption]
