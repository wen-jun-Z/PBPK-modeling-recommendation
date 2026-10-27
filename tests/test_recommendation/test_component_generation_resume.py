"""Crash/restart integration tests with a fake generator; no GPU calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import prepare_component_model_integration as prepare
import generate_component_model_integration as gen


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "sha", lambda p: hashlib.sha256(
        (Path(p) if Path(p).is_absolute() else tmp_path / p).read_bytes()).hexdigest())
    monkeypatch.setattr(gen, "corpus_for", lambda _: None)
    monkeypatch.setattr(gen, "progress", lambda *a, **k: None)
    prepare.write(tmp_path / "protocol.json", {"test_only": True})
    entry = {"query_id": "clinical:resume", "global_index": 145, "key": "resume",
        "raw_query_sha256": "raw-binding", "request_sha256": "request-binding",
        "status": "pending_retrieval", "modeling_request": {"dose": 50}}
    prepare.write_rows(tmp_path / "inputs/requests.jsonl", [entry])
    prepare.write(tmp_path / "inputs/prompts/resume.json", {"status": "ready", "prompt": "fixed 50mg"})
    calls, validations = [], []
    class Fake:
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None))
        def __init__(self, *args):
            pass
        def generate(self, tasks):
            calls.append(deepcopy(tasks))
            return ["valid" if task.get("repair") else "invalid" for task in tasks], {"batch_size": len(tasks)}
    def validate(corpus, entry, prompt, raw):
        validations.append(raw)
        valid = raw == "valid"
        return ({} if valid else None), {"status": "recommendation_validated" if valid else "invalid_selection",
            "errors": [] if valid else ["first-actual-error"]}, []
    monkeypatch.setattr(gen, "ComponentGenerator", Fake)
    monkeypatch.setattr(gen, "validate_output", validate)
    return SimpleNamespace(path=tmp_path, calls=calls, validations=validations, fake=Fake, validate=validate, entry=entry)


def result(run):
    return prepare.read(run.path / "components/resume.json")


def test_first_failed_attempt_and_exact_repair_survive_restart(run, monkeypatch):
    def stop(*a, **k):
        raise KeyboardInterrupt("after invalid attempt was queued")
    monkeypatch.setattr(gen, "progress", stop)
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    assert len(run.calls) == 1
    checkpoint = prepare.read(run.path / "components/attempts/resume.json")
    assert checkpoint["attempts"][0]["state"] == "evaluated"
    monkeypatch.setattr(gen, "progress", lambda *a, **k: None)
    gen.generate_all(run.path)
    assert len(run.calls) == 2
    assert run.calls[1][0]["repair"] == {"raw": "invalid", "errors": ["first-actual-error"]}
    assert result(run)["status"] == "validated"
    assert len(result(run)["attempts"]) == 2


def test_generated_raw_is_revalidated_without_repeating_generation(run, monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt("interrupt before validation")
    monkeypatch.setattr(gen, "validate_output", interrupt)
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    assert len(run.calls) == 1
    monkeypatch.setattr(gen, "validate_output", run.validate)
    gen.generate_all(run.path)
    assert len(run.calls) == 2  # original plus its one correction, not original again
    assert run.validations == ["invalid", "valid"]
    assert result(run)["status"] == "validated"


def test_validated_success_finalizes_after_crash_without_loading_generator(run, monkeypatch):
    original_write = gen.durable_write
    def interrupt(path, value):
        if Path(path) == run.path / "components/resume.json":
            raise KeyboardInterrupt("just before final result replace")
        original_write(path, value)
    monkeypatch.setattr(gen, "durable_write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    assert len(run.calls) == 2
    monkeypatch.setattr(gen, "durable_write", original_write)
    monkeypatch.setattr(gen, "ComponentGenerator", lambda *a: pytest.fail("Should not reload generator"))
    gen.generate_all(run.path)
    assert result(run)["status"] == "validated"
    assert run.validations == ["invalid", "valid"]


def test_second_failure_is_terminal_across_restarts(run, monkeypatch):
    monkeypatch.setattr(run.fake, "generate", lambda self, tasks: (
        run.calls.append(deepcopy(tasks)) or ["invalid"] * len(tasks), {"batch_size": len(tasks)}))
    gen.generate_all(run.path)
    assert len(run.calls) == 2 and result(run)["status"] == "failed"
    gen.generate_all(run.path)
    assert len(run.calls) == 2 and len(result(run)["attempts"]) == 2


def test_interrupted_call_without_receipt_is_not_silently_replayed(run, monkeypatch):
    def interrupt(self, tasks):
        run.calls.append(deepcopy(tasks))
        raise KeyboardInterrupt("GPU worker vanished during a reserved call")
    monkeypatch.setattr(run.fake, "generate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    monkeypatch.setattr(gen, "ComponentGenerator", lambda *a: pytest.fail("Ambiguous call must not replay"))
    gen.generate_all(run.path)
    assert len(run.calls) == 1
    assert result(run)["status"] == "failed"
    assert result(run)["attempts"][0]["timing"]["failure"] == "InterruptedUnrecordedGeneration"


def test_complete_batch_receipt_recovers_row_not_yet_checkpointed(run, monkeypatch):
    original_save = gen.save_attempts
    def interrupt(output, task):
        if task["attempts"][-1]["state"] == "generated":
            raise KeyboardInterrupt("receipt committed; row checkpoint not yet written")
        original_save(output, task)
    monkeypatch.setattr(gen, "save_attempts", interrupt)
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    assert prepare.read(run.path / "components/attempts/resume.json")["attempts"][-1]["state"] == "started"
    monkeypatch.setattr(gen, "save_attempts", original_save)
    gen.generate_all(run.path)
    assert len(run.calls) == 2 and run.validations == ["invalid", "valid"]


@pytest.mark.parametrize("field", ["query_id", "request_sha256", "raw_query_sha256", "protocol_sha256", "producer_sha256"])
def test_changed_request_or_producer_refuses_resume(run, monkeypatch, field):
    monkeypatch.setattr(gen, "progress", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    path = run.path / "components/attempts/resume.json"
    saved = prepare.read(path)
    saved["binding"][field] = "changed"
    prepare.write(path, saved)
    with pytest.raises(ValueError, match="binding changed"):
        gen.generate_all(run.path)
    assert len(run.calls) == 1


def test_over_budget_checkpoint_is_rejected(run, monkeypatch):
    monkeypatch.setattr(gen, "progress", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        gen.generate_all(run.path)
    path = run.path / "components/attempts/resume.json"
    saved = prepare.read(path)
    saved["attempts"] *= 3
    prepare.write(path, saved)
    with pytest.raises(ValueError, match="over-budget"):
        gen.generate_all(run.path)
    assert len(run.calls) == 1


def migration_fixture(run, monkeypatch):
    root = run.path
    script = root / "scripts/prepare_component_model_integration.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("fixture prepare")
    bridge = root / "src/pbpk_kb/recommendation/component_model_bridge.py"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    bridge.write_text("fixture tri-state context")
    backup = root / "revisions/002_context_and_attempt_resume/before_generate_component_model_integration.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("fixture old producer")
    monkeypatch.setattr(gen, "CONTEXT_RESUME_OLD_PRODUCER", gen.sha(backup))
    real_artifact = gen.artifact
    monkeypatch.setattr(gen, "artifact", lambda p: {
        "path": "scripts/generate_component_model_integration.py", "bytes": 3, "sha256": "new-producer"
    } if Path(p) == Path(gen.__file__) else real_artifact(p))
    prepare.write(root / "protocol.json", {"retrieval": {}, "source_artifacts": []})
    prepare.write_rows(root / "retrieval/results.jsonl", [{"old_retrieval": True}])
    prepare.write(root / "retrieval/cross_encoder.json", {"same_scores": [0.25, 0.75]})
    prepare.write(root / "retrieval/manifest.json", {"status": "completed", "statuses": {"ready": 32},
        "artifacts": [real_artifact(root / "inputs/prompts/resume.json"), real_artifact(root / "retrieval/results.jsonl")]})
    old = {"protocol_sha256": gen.sha(root / "protocol.json"), "prepare_script": real_artifact(script),
        "generate_script": {"path": "scripts/generate_component_model_integration.py", "bytes": backup.stat().st_size,
                            "sha256": gen.sha(backup)}}
    prepare.write(root / "run_binding.json", old)
    return old


def test_revision002_preserves_old_producer_prompts_and_exact_scores(run, monkeypatch):
    old = migration_fixture(run, monkeypatch)
    cross = (run.path / "retrieval/cross_encoder.json").read_bytes()
    record = gen.record_context_resume_repair(run.path)
    assert record["old_binding"] == old
    assert record["previous_retrieval_statuses"] == {"ready": 32}
    assert (run.path / "retrieval/cross_encoder.json").read_bytes() == cross
    assert not (run.path / "retrieval/manifest.json").exists()
    assert not (run.path / "inputs/prompts/resume.json").exists()
    for item in record["preserved_previous_stage"]:
        assert gen.sha(run.path / item["archive"]["path"]) == item["original"]["sha256"]
    protocol = prepare.read(run.path / "protocol.json")
    assert protocol["generation_attempt_checkpoint"]["maximum_attempts"] == 2
    assert protocol["source_artifacts"][-1]["path"].endswith("component_model_bridge.py")
    assert gen.record_context_resume_repair(run.path) == record


def test_revision002_rejects_changed_old_retrieval_artifacts(run, monkeypatch):
    migration_fixture(run, monkeypatch)
    prepare.write(run.path / "inputs/prompts/resume.json", {"changed": True})
    with pytest.raises(ValueError, match="retrieval artifact changed"):
        gen.record_context_resume_repair(run.path)


def test_revision002_cannot_migrate_after_any_component_attempt(run, monkeypatch):
    migration_fixture(run, monkeypatch)
    prepare.write(run.path / "components/attempts/existing.json", {"started": True})
    with pytest.raises(ValueError, match="after component generation"):
        gen.record_context_resume_repair(run.path)
