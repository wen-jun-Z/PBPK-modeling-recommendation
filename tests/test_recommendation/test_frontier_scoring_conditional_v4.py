"""CPU integration of the frozen v3 engine with isolated v4 dependencies."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

from frozen_frontier_sources import historical_sha

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v = importlib.import_module("run_frontier_scoring_conditional_v4")


@pytest.fixture(autouse=True)
def restore_injection():
    old_engine = {k: getattr(v.engine, k) for k in v.INJECTED}
    old_base = {k: getattr(v.base, k) for k in ("SYSTEM", "ASSESS", "RUN", "prepare", "GPUJudge", "download_primary")}
    old_assess = v.utilities.ASSESS
    yield
    for k, value in old_engine.items(): setattr(v.engine, k, value)
    for k, value in old_base.items(): setattr(v.base, k, value)
    v.utilities.ASSESS = old_assess


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    assess, old, run = tmp_path / "v4", tmp_path / "v3", tmp_path / "run"
    for path in (assess, old, run): path.mkdir()
    for key, value in (("ROOT", tmp_path), ("ASSESS", assess), ("PARENT_ASSESS", old), ("RUN", run)):
        monkeypatch.setattr(v, key, value)
    return assess, old, run


def test_explicit_injection_uses_original_grade_first_prompt_and_new_command(paths):
    v.configure()
    prompt = importlib.import_module("frontier_judge_prompt_v2")
    assert v.base.SYSTEM == prompt.SYSTEM == v.engine.SYSTEM
    assert v.base.ASSESS == v.ASSESS and v.base.RUN == v.RUN
    assert v.engine.prepare is v.prepare and v.engine.backend is v.backend
    assert v.engine.verify_source_review is v.verify_source_review
    assert v.engine.command(v.APP, "pilot", "--model", "phi")[1] == str(Path(v.__file__).resolve())
    assert '"i":1,"g":null,"a":"U","e":[]' in v.base.SYSTEM
    assert "Select the evidence BEFORE" not in v.base.SYSTEM


@pytest.mark.parametrize("model", ["phi", "qwen"])
def test_only_conditional_decoder_wraps_both_prepared_backends(monkeypatch, model):
    decoder = importlib.import_module("frontier_conditional_citation_decoding")
    wrappers = []
    class Parent:
        def __init__(self, *args, **kwargs): self.json_constraint = "official-base-constraint"
    monkeypatch.setattr(v.utilities, "PhiJudge", Parent)
    monkeypatch.setattr(v.utilities, "backend", lambda name: Parent)
    monkeypatch.setattr(decoder, "conditional_citation_constraint", lambda original: wrappers.append(original) or "conditional")
    assert v.backend(model)({}, 36).json_constraint == "conditional"
    assert wrappers == ["official-base-constraint"]


def test_only_cases_manifest_plan_reused_and_no_parent_predictions_linked(paths):
    assess, old, _ = paths
    for name in ("validation/cases.jsonl", "validation/manifest.json", "source_review_plan.json"):
        put(old / name, {"fixture": name})
    put(old / "validation/phi_records.jsonl", {"old": True})
    put(old / "judgments.jsonl", {"old": True})
    v.link_validation_inputs()
    assert not (assess / "validation").is_symlink()
    assert {p.name for p in (assess / "validation").iterdir()} == {"cases.jsonl", "manifest.json"}
    assert (assess / "source_review_plan.json").is_symlink()
    assert not (assess / "judgments.jsonl").exists()


@pytest.mark.parametrize("name", ["validation/phi_records.jsonl", "validation/qwen_raw_generation_journal.jsonl",
                                  "validation/admission.json", "judgments.jsonl", "primary_judgment_cache.jsonl"])
def test_linking_any_earlier_generated_output_is_rejected(paths, name):
    assess, old, _ = paths
    for item in ("validation/cases.jsonl", "validation/manifest.json", "source_review_plan.json"):
        put(old / item, {})
    put(old / name, {"old": True})
    (assess / name).parent.mkdir(parents=True, exist_ok=True)
    (assess / name).symlink_to(old / name)
    with pytest.raises(ValueError, match="own pilot/full outputs"): v.link_validation_inputs()


def test_whole_validation_directory_link_is_rejected(paths):
    assess, old, _ = paths
    (old / "validation").mkdir()
    (assess / "validation").symlink_to(old / "validation", target_is_directory=True)
    with pytest.raises(ValueError, match="independent directory"): v.link_validation_inputs()


@pytest.mark.parametrize("action", ["prepare", "start", "pipeline", "pilot", "run"])
def test_all_execution_entries_refuse_active_v3_or_selected_parent(paths, monkeypatch, action):
    monkeypatch.setattr(v.base, "alive", lambda value: value.get("fixture_active", False))
    put(v.ROOT / "runtime/deployment/frontier_scoring_selected_v3/pipeline_pid.json", {"fixture_active": True})
    with pytest.raises(RuntimeError, match="cannot overlap"): v.execute(action)
    assert not (v.ASSESS / "judge_protocol.json").exists()


@pytest.fixture
def pilot(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(v, "fallback_activation", lambda: {"accepted": True})
    monkeypatch.setattr(v, "prepare", lambda: None)
    put(assess / "judge_protocol.json", {"version": "fresh-v4"})
    put(assess / "validation/manifest.json", {"fixture": True})
    cases = [{"case_id": "control", "kind": "synthetic_control", "packet": {}, "query": {}},
             {"case_id": "source", "kind": "real_source_pair", "packet": {}, "query": {}}]
    v.base.append_rows(assess / "validation/cases.jsonl", cases)
    monkeypatch.setattr(v.utilities, "verify_weights", lambda path: {"repo": "fixture", "revision": "pinned"})
    monkeypatch.setattr(v.engine, "validation_error", lambda output, case: "invalid" if output == "raw" else None)
    validator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(validator, "evaluate_validation_v3", lambda rows, cases: {"complete": len(rows) == len(cases)})
    return assess, old, run


def test_actual_engine_pilot_generates_new_raw_and_one_real_repair_with_v2_system(pilot, monkeypatch):
    assess, old, run = pilot
    calls = []
    put(old / "validation/phi_records.jsonl", {"unchanged_old_prediction": True})
    before = (old / "validation/phi_records.jsonl").read_bytes()
    class Judge:
        def __init__(self, *args): assert v.base.SYSTEM == v.SYSTEM
        def select_batch_size(self, tasks, maximum): return len(tasks)
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            calls.append((max_new_tokens, repairs))
            if repairs:
                assert len(v.base.load_committed(assess / "validation/phi_raw_generation_journal.jsonl")) == 2
            return ["repaired" if repairs else "raw"] * len(tasks), {"wall_seconds": 1}
    monkeypatch.setattr(v, "backend", lambda model: Judge)
    v.execute("pilot", model="phi")
    rows = v.base.load_committed(assess / "validation/phi_records.jsonl")
    assert [len(r["attempts"]) for r in rows] == [1, 2]
    assert rows[0]["output"] == rows[1]["output"] == "raw"
    assert rows[1]["final_output"] == "repaired"
    assert calls[0][0] == 96 and calls[1][0] == 160
    assert (old / "validation/phi_records.jsonl").read_bytes() == before
    spec = v.base.read(assess / "validation/phi_protocol.json")
    assert spec["judge_protocol_sha256"] == v.base.file_sha(assess / "judge_protocol.json")
    monkeypatch.setattr(v, "backend", lambda model: pytest.fail("Complete v4 pilot resumes without a new GPU load"))
    v.execute("pilot", model="phi")
    assert len(v.base.load_committed(assess / "validation/phi_records.jsonl")) == 2


def test_old_prediction_protocol_cannot_be_resumed_as_v4(pilot, monkeypatch):
    assess, old, run = pilot
    v.base.append_rows(assess / "validation/phi_records.jsonl", [{"case_id": "control", "pilot_protocol_sha256": "old-v3"}])
    monkeypatch.setattr(v, "backend", lambda model: pytest.fail("Mixed old protocol must fail before GPU"))
    with pytest.raises(ValueError, match="another frozen protocol"): v.execute("pilot", model="phi")


@pytest.mark.parametrize("judge,model", [("primary", "phi"), ("secondary", "qwen")])
def test_full_engine_handoff_has_correct_model_prompt_and_weights(paths, monkeypatch, judge, model):
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(v, "fallback_activation", lambda: {"accepted": True})
    monkeypatch.setattr(v.engine, "check_gate", lambda: {"accepted": True})
    captured = {}
    monkeypatch.setattr(v, "backend", lambda name: captured.update(model=name) or name)
    monkeypatch.setattr(v.utilities, "verify_weights", lambda path: captured.update(primary_weights=path))
    def run(args):
        captured["args"] = args
        assert v.base.SYSTEM == v.SYSTEM
        v.base.download_primary()
    monkeypatch.setattr(v.base, "run_assessment", run)
    original_guard = v.base.validate_explicit_scope
    v.execute("run", judge=judge)
    assert captured["model"] == model and captured["primary_weights"] == v.PHI_WEIGHTS
    assert captured["args"].secondary_weights == v.QWEN_WEIGHTS
    assert captured["args"].batch_size == 16 and captured["args"].memory_gib == 36
    assert v.base.prepare is v.prepare and v.base.validate_explicit_scope is original_guard


def test_quality_guard_still_refuses_identical_v4_restart(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(v, "fallback_activation", lambda: {"accepted": True})
    put(run / "status.json", {"guard_reason": "PrimaryInvalidFractionAbove10Percent"})
    with pytest.raises(RuntimeError, match="not an identical restart"): v.execute("start")


def test_actual_scheduler_uses_v4_children_and_does_not_skip_pilots(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(v, "fallback_activation", lambda: {"accepted": True})
    monkeypatch.setattr(v, "prepare", lambda: None)
    monkeypatch.setattr(v, "publish_summary", lambda: None)
    monkeypatch.setattr(v.engine.signal, "signal", lambda *args: None)
    monkeypatch.setattr(v.base, "identity", lambda: {"pid": 100})
    monkeypatch.setattr(v.base, "stop_child", lambda child: None)
    monkeypatch.setattr(v.engine, "check_gate", lambda require_review=True: {"numeric_accepted": False, "accepted": False})
    commands = []
    class Child:
        returncode = 0
        def __init__(self, command, **kwargs):
            commands.append(command); self.pid = 200+len(commands)
            model = command[-1]
            put(run / f"pilot_{model}_status.json", {"pid": self.pid, "state": "complete", "completed": 733})
        def poll(self): return self.returncode
    monkeypatch.setattr(v.engine.subprocess, "Popen", Child)
    v.execute("pipeline")
    assert [c[-3:] for c in commands] == [["pilot", "--model", "phi"], ["pilot", "--model", "qwen"]]
    assert all(c[1] == str(Path(v.__file__).resolve()) for c in commands)
    assert [c[0] for c in commands] == [str(v.APP), str(v.MODERN)]
    status = v.base.read(run / "pipeline_status.json")
    assert status["state"] == "stopped_on_validation" and status["completed"] == 733


@pytest.fixture
def review(paths, monkeypatch):
    assess, old, run = paths
    put(assess / "judge_protocol.json", {"v4": True})
    put(assess / "validation/manifest.json", {})
    for model in ("phi", "qwen"): (assess / "validation" / f"{model}_records.jsonl").write_text('{}\n')
    cases = [{"case_id": "failure", "strata": {"validation_role": "production_failure_diagnostic"}},
             {"case_id": "sample", "strata": {"validation_role": "prospective_source_coverage"}}]
    v.base.append_rows(assess / "validation/cases.jsonl", cases)
    monkeypatch.setattr(v.engine, "verify_review_plan", lambda cases: {"case_ids": ["failure", "sample"]})
    value = {"version": "pbpk-v4-source-review-v1", "protocol_sha256": v.base.file_sha(assess / "judge_protocol.json"),
        "validation_manifest_sha256": v.base.file_sha(assess / "validation/manifest.json"),
        "pilot_record_sha256": {model: v.base.file_sha(assess / "validation" / f"{model}_records.jsonl") for model in ("phi", "qwen")},
        "checked_case_ids": ["failure", "sample"], "reviewer": "Codex evidence audit; not expert gold",
        "scientific_gold_accuracy": None, "findings": [], "known_unknown_distribution": {}, "accepted": True}
    put(assess / "source_review.json", value)
    return assess, value


def test_new_source_review_is_bound_and_not_reused_v3_review(review):
    assess, value = review
    assert v.verify_source_review()[0]
    value["version"] = "pbpk-v3-source-review-v1"; put(assess / "source_review.json", value)
    with pytest.raises(ValueError, match="these new pilot outputs"): v.verify_source_review()


@pytest.mark.parametrize("change", ["sample_missing", "old_protocol", "old_output", "gold_claim"])
def test_source_review_rejects_missing_planned_cases_and_stale_identity(review, change):
    assess, value = review
    if change == "sample_missing": value["checked_case_ids"] = ["failure"]
    elif change == "old_protocol": value["protocol_sha256"] = "v3"
    elif change == "old_output": value["pilot_record_sha256"]["phi"] = "v3"
    else: value["scientific_gold_accuracy"] = 1.
    put(assess / "source_review.json", value)
    with pytest.raises(ValueError): v.verify_source_review()


def test_frozen_amendment_and_parent_code_are_unchanged():
    amendment = v.verify_amendment()
    assert amendment["selection_timing"]["observed_qwen_v3_cases"] == 0
    assert amendment["validation_interpretation"] == v.REGRESSION_DISCLOSURE
    assert historical_sha("run_frontier_scoring_v3.py") == "f389c9093640325320f5ed7306fd0151f87f88506878bbb5dda1e45d909cc2f2"
    assert historical_sha("run_frontier_scoring_selected_v3.py") == "b05a4c7f0c13281db70ff7261e6bc049dc82736bbcf45ad065fdee8b16ec58d0"


@pytest.fixture
def activation(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v, "OUT", old.parent)
    # Use the real frozen cohort and evaluator. Only generated fixture outputs
    # are synthetic, so partial-group completeness follows production logic.
    real_cases = v.base.read_rows(
        Path(v.engine.__file__).resolve().parents[1] /
        "experiments/model_recommendation/frontier_comparison_20260914/assessment_evidence_first_v3/validation/cases.jsonl")
    folder = old / "validation"
    folder.mkdir()
    v.base.append_rows(folder / "cases.jsonl", real_cases)
    put(folder / "manifest.json", {"case_count": 733, "cases_sha256": v.base.file_sha(folder / "cases.jsonl")})
    amendment = {"parent_protocol_sha256": "parent-protocol", "validation_manifest_sha256": "manifest-hash"}
    monkeypatch.setattr(v, "verify_amendment", lambda: amendment)
    model_rows = {}
    for model in ("phi", "qwen"):
        weights = old / f"{model}_weights.json"
        put(weights, {"repo": model, "revision": "fixed"})
        monkeypatch.setattr(v, "PHI_WEIGHTS" if model == "phi" else "QWEN_WEIGHTS", weights)
        protocol = {"model_repo": model, "model_revision": "fixed", "judge_protocol_sha256": "parent-protocol",
            "validation_sha256": "manifest-hash", "maximum_batch_size": 16, "memory_gib": 36, "raw_max_tokens": 96,
            "repair_max_tokens": 160, "control_repairs": 0, "noncontrol_repairs": 1, "raw_journal_before_repairs": True}
        put(folder / f"{model}_protocol.json", protocol)
        rows = []
        for case in real_cases[:64]:
            output = '[{"i":1,"e":[],"g":null,"a":"U"}]'
            rows.append({"case_id": case["case_id"], "kind": case["kind"], "model_repo": model, "model_revision": "fixed",
                "pilot_protocol_sha256": v.base.file_sha(folder / f"{model}_protocol.json"), "validation_sha256": "manifest-hash",
                "output": output, "final_output": output, "attempts": [{"output": output}]})
        v.base.append_rows(folder / f"{model}_records.jsonl", rows)
        v.base.append_rows(folder / f"{model}_raw_generation_journal.jsonl", rows)
        model_rows[model] = rows
    return folder, real_cases, model_rows


def test_only_complete_observed_absolute_failures_activate_fallback(activation):
    folder, cases, rows = activation
    result = v.fallback_activation()
    assert result["accepted"] and not result["missing_data_is_failure"] and not result["relative_gate_used"]
    for model in ("phi", "qwen"):
        assert result["models"][model]["observed_cases"] == 64
        assert not result["models"][model]["complete_all_733"]
        assert v.CONTROL_ROLES[0] in result["models"][model]["failed_control_groups"]
        assert result["models"][model]["records_sha256"] == v.base.file_sha(folder / f"{model}_records.jsonl")


def test_partial_control_group_and_missing_review_are_not_failure(activation):
    folder, cases, rows = activation
    (folder / "qwen_records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows["qwen"][:63]))
    with pytest.raises(RuntimeError, match="potentially eligible"): v.fallback_activation()


def test_previously_accepted_selected_candidate_blocks_fallback(activation):
    put(v.OUT / "assessment_selected_v3/validation/admission.json", {"accepted": True})
    with pytest.raises(RuntimeError, match="already accepted"): v.fallback_activation()


def test_activation_allows_raw_superset_and_discloses_partial_tail_without_modifying_it(activation):
    folder, cases, rows = activation
    extra = {**rows["qwen"][0], "case_id": cases[64]["case_id"], "kind": cases[64]["kind"]}
    raw_path = folder / "qwen_raw_generation_journal.jsonl"
    v.base.append_rows(raw_path, [extra])
    with raw_path.open("ab") as handle: handle.write(b'{"uncommitted":')
    original = raw_path.read_bytes()
    result = v.fallback_activation()
    assert result["accepted"] and result["models"]["qwen"]["observed_cases"] == 64
    assert result["models"]["qwen"]["uncommitted_tail_bytes"]["raw"] == len(b'{"uncommitted":')
    assert raw_path.read_bytes() == original


@pytest.mark.parametrize("change", ["raw_identity", "duplicate_final", "different_generation", "repaired_control"])
def test_activation_cannot_use_mixed_or_modified_pilot_data(activation, change):
    folder, cases, rows = activation
    if change == "raw_identity":
        rows["qwen"][0]["model_revision"] = "different"
        (folder / "qwen_raw_generation_journal.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows["qwen"]))
    elif change == "duplicate_final": v.base.append_rows(folder / "qwen_records.jsonl", [rows["qwen"][0]])
    elif change == "different_generation":
        path = folder / "qwen_protocol.json"; spec = v.base.read(path); spec["raw_max_tokens"] = 160; put(path, spec)
    else:
        rows["qwen"][0]["attempts"].append({"output": rows["qwen"][0]["output"]})
        (folder / "qwen_records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows["qwen"]))
    with pytest.raises(ValueError): v.fallback_activation()


@pytest.mark.parametrize("action", ["start", "pipeline", "pilot", "run"])
def test_gpu_entry_cannot_bypass_fallback_activation(paths, monkeypatch, action):
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    def blocked(): raise RuntimeError("potentially eligible")
    monkeypatch.setattr(v, "fallback_activation", blocked)
    monkeypatch.setattr(v.engine, "backend", lambda *args: pytest.fail("No GPU before activation"))
    with pytest.raises(RuntimeError, match="potentially eligible"): v.execute(action)


def test_prepare_freezes_v2_prompt_new_decoder_activation_and_regression_disclosure(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    trigger = {"accepted": True, "models": {"phi": {"observed_cases": 733}, "qwen": {"observed_cases": 128}}}
    monkeypatch.setattr(v, "fallback_activation", lambda: trigger)
    amendment = {"parent_protocol_sha256": "v3-parent", "selection_timing": {"observed_qwen_v3_cases": 0},
        "validation_manifest_sha256": "old-cases-manifest", "source_review_plan_sha256": "old-plan"}
    monkeypatch.setattr(v, "verify_amendment", lambda: amendment)
    for model in ("phi", "qwen"):
        path = old / f"{model}.json"; put(path, {"repo": model, "revision": "pinned"})
        monkeypatch.setattr(v, "PHI_WEIGHTS" if model == "phi" else "QWEN_WEIGHTS", path)
    old_protocol = {"implementation_sha256": {}, "system_prompt": "old evidence-first prompt",
        "generation": {"max_new_tokens": 96, "retry_max_new_tokens": 160, "max_attempts": 2,
                       "constrained_decoding": {"old": True}}, "validation_policy": {}, "judge_development": {},
        "scope_guard": {"version": "unchanged"}, "pilot_labels_reused": False,
        "primary_weights_manifest_sha256": v.base.file_sha(v.PHI_WEIGHTS),
        "secondary_weights_manifest_sha256": v.base.file_sha(v.QWEN_WEIGHTS)}
    put(old / "judge_protocol.json", old_protocol)
    monkeypatch.setattr(v, "CODE_FILES", ())
    monkeypatch.setattr(v, "link_validation_inputs", lambda: None)
    put(assess / "validation/cases.jsonl", {"case_id": "fixture"})
    put(assess / "sources.jsonl", {"fixture": "source"})
    monkeypatch.setattr(v.utilities, "prepare_materials", lambda: ([{}], [{}], {"p": {"packet_sha256": "p", "refs": []}}, {}))
    monkeypatch.setattr(v.utilities, "execution_environments", lambda: {"phi": "app", "qwen": "modern"})
    validator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(validator, "verify_validation_v3", lambda *args: {"case_count": 733})
    monkeypatch.setattr(v.engine, "verify_review_plan", lambda cases: {})
    decoder = importlib.import_module("frontier_conditional_citation_decoding")
    monkeypatch.setattr(decoder, "decoding_protocol", lambda: {"new": "conditional-grade-first"})
    *_, protocol = v.prepare()
    assert protocol["system_prompt"] == v.SYSTEM
    assert protocol["generation"] == {"max_new_tokens": 96, "retry_max_new_tokens": 160, "max_attempts": 2,
                                      "constrained_decoding": {"new": "conditional-grade-first"}}
    assert protocol["activation_evidence"] == trigger
    assert protocol["scope_guard"] == old_protocol["scope_guard"]
    assert not protocol["full_labels_reused"] and not protocol["parent_pilot_outputs_reused"]
    assert protocol["validation_policy"]["interpretation"] == v.REGRESSION_DISCLOSURE
    assert not protocol["validation_policy"]["independent_blind_validation"]
    assert v.prepare()[3] == protocol
    assert v.base.read(old / "judge_protocol.json") == old_protocol
