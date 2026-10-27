"""Bounded queue admission and isolation checks, without loading GPU models."""
import importlib
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
scoring = importlib.import_module("run_frontier_scoring")


@pytest.fixture(autouse=True)
def restore_profile_globals():
    names = ("PROFILE", "ASSESS", "RUN", "VERSION", "QWEN_WEIGHTS")
    saved = {name: getattr(scoring, name) for name in names}
    base_saved = {name: getattr(scoring.base, name) for name in
                  ("SYSTEM", "ASSESS", "RUN", "GPUJudge", "prepare", "download_primary")}
    yield
    for name, value in saved.items():
        setattr(scoring, name, value)
    for name, value in base_saved.items():
        setattr(scoring.base, name, value)


def test_state_transition_accepts_state_keyword(monkeypatch):
    saved = []
    monkeypatch.setattr(scoring.base, "dump", lambda p, value: saved.append(value.copy()))
    snapshot = {"state": "starting"}
    scoring.status_update(Path("unused.json"), snapshot, state="loading_gpu")
    assert saved[-1]["state"] == "loading_gpu"


def test_fixed_specification_cannot_be_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring.base, "dump", lambda p, v: p.write_text(json.dumps(v)))
    path = tmp_path / "protocol.json"
    scoring.freeze(path, {"model": "first"})
    scoring.freeze(path, {"model": "first"})
    with pytest.raises(ValueError, match="Frozen specification differs"):
        scoring.freeze(path, {"model": "second"})


def test_no_manifest_means_no_weight_loading(tmp_path):
    with pytest.raises(ValueError, match="not ready"):
        scoring.verify_weights(tmp_path / "absent.json")


@pytest.mark.parametrize("mutation", [
    {"complete": False}, {"controlled_semantic_accuracy": .89},
    {"raw_format_valid_fraction": .97}, {"citation_numbers_valid_fraction": .97},
    {"controlled_citation_support_accuracy": .94},
])
def test_gate_rejects_bad_or_missing_validation(tmp_path, monkeypatch, mutation):
    validator = importlib.import_module("frontier_judge_validation")
    monkeypatch.setattr(scoring, "ASSESS", tmp_path)
    monkeypatch.setattr(scoring.base, "read_rows", lambda p: [p.stem])
    baseline = {"rows": [], "complete": True, "controlled_semantic_accuracy": .96,
                "raw_format_valid_fraction": 1., "citation_numbers_valid_fraction": 1.,
                "controlled_citation_support_accuracy": 1.}
    monkeypatch.setattr(validator, "evaluate_validation", lambda records, cases:
                        dict(baseline, **mutation) if records == ["qwen_records"] else baseline.copy())
    monkeypatch.setattr(validator, "admission_metrics", lambda result: result)
    monkeypatch.setattr(scoring.base, "dump", lambda *args: None)
    assert scoring.check_gate()["accepted"] is False


def test_gate_accepts_complete_strong_control_results_without_claiming_gold(tmp_path, monkeypatch):
    validator = importlib.import_module("frontier_judge_validation")
    monkeypatch.setattr(scoring, "ASSESS", tmp_path)
    monkeypatch.setattr(scoring.base, "read_rows", lambda p: [])
    metric = {"rows": [], "complete": True, "controlled_semantic_accuracy": 1.,
              "raw_format_valid_fraction": 1., "citation_numbers_valid_fraction": 1.,
              "controlled_citation_support_accuracy": 1.}
    monkeypatch.setattr(validator, "evaluate_validation", lambda *args: metric)
    monkeypatch.setattr(validator, "admission_metrics", lambda result: result)
    monkeypatch.setattr(scoring.base, "dump", lambda *args: None)
    result = scoring.check_gate()
    assert result["accepted"] is True
    assert result["scientific_accuracy_established"] is False


def test_waiting_user_stop_does_not_load_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "RUN", tmp_path)
    monkeypatch.setattr(scoring, "BACKEND_RUN", tmp_path / "backend")
    monkeypatch.setattr(scoring, "publish_summary", lambda: None)
    (tmp_path / "stop.requested").touch()
    seen = []
    assert scoring.wait_backend(lambda **v: seen.append(v)) is False
    assert seen[-1]["state"] == "stopped"


def test_v2_profile_isolates_records_and_changes_both_judges_system_only():
    from frontier_judge_prompt_v2 import SYSTEM
    scoring.configure_profile("v2")
    assert scoring.ASSESS.name == "assessment_qwen36_v2"
    assert scoring.RUN.name == "frontier_scoring_v2"
    assert scoring.QWEN_WEIGHTS == scoring.V1_ASSESS / "model_weights/qwen.json"
    assert scoring.base.SYSTEM == SYSTEM != scoring.V1_SYSTEM
    command = scoring.worker_command(scoring.MODERN, "run", "--judge", "primary")
    assert command[-2:] == ["--profile", "v2"]
    scoring.configure_profile()
    assert scoring.PROFILE == "v1" and scoring.base.SYSTEM == scoring.V1_SYSTEM


def test_failed_v1_start_is_refused_without_rewriting_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "V1_ASSESS", tmp_path)
    path = tmp_path / "validation/admission.json"
    path.parent.mkdir()
    original = '{"accepted":false,"controlled_joint":51}'
    path.write_text(original)
    with pytest.raises(RuntimeError, match="explicit --profile v2"):
        scoring.ensure_execution_allowed()
    assert path.read_text() == original


def test_original_runner_snapshot_requires_exact_frozen_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "V1_ASSESS", tmp_path)
    path = tmp_path / "reproducibility/run_frontier_scoring.py"
    path.parent.mkdir()
    path.write_text("original preserved source\n")
    digest = scoring.base.file_sha(path)
    (tmp_path / "judge_protocol.json").write_text(json.dumps({"implementation_sha256": {"run_frontier_scoring.py": digest}}))
    assert scoring.v1_runner_snapshot_sha() == digest
    path.write_text("modified")
    with pytest.raises(ValueError, match="does not match"):
        scoring.v1_runner_snapshot_sha()


def v2_gate_fixture():
    validator = importlib.import_module("frontier_judge_validation")
    controls = validator._synthetic_controls()
    cases = []
    for role in ("development_regression", "independent_admission"):
        for original in controls:
            case = copy.deepcopy(original)
            case["case_id"] = role + ":" + case["case_id"]
            case["strata"]["validation_role"] = role
            cases.append(case)
    for original in controls[:60]:
        case = copy.deepcopy(original)
        case.update(case_id="real:" + case["case_id"], kind="real_source_pair", expected=None)
        case["strata"]["validation_role"] = "unlabelled_real_source"
        cases.append(case)
    records = []
    for case in cases:
        expected = case["expected"] or {"grades": [None], "documented_applicability": None, "supporting_excerpt_numbers": []}
        grade, app = expected["grades"][0], expected["documented_applicability"]
        refs = expected["supporting_excerpt_numbers"][:1] if grade is not None or app is not None else []
        records.append({"case_id": case["case_id"], "output": json.dumps([{
            "i": 1, "g": grade, "a": "U" if app is None else "Y" if app else "N", "e": refs}])})
    return cases, {"phi": copy.deepcopy(records), "qwen": copy.deepcopy(records)}


def test_v2_gate_reports_two_control_groups_without_pooling():
    cases, records = v2_gate_fixture()
    result = scoring.grouped_gate(records, cases)
    assert result["accepted"] is True
    assert result["pooled_control_accuracy_used_for_admission"] is False
    assert set(result["groups"]) == {"independent_admission", "development_regression"}
    assert all(g["case_count"] == 64 for g in result["groups"].values())


@pytest.mark.parametrize("role", ["independent_admission", "development_regression"])
def test_v2_strong_other_group_cannot_hide_a_failed_group(role):
    cases, records = v2_gate_fixture()
    failing = {c["case_id"] for c in cases if c["strata"]["validation_role"] == role}
    for row in [r for r in records["qwen"] if r["case_id"] in failing][:8]:
        row["output"] = "invalid JSON"
    result = scoring.grouped_gate(records, cases)
    assert result["accepted"] is False
    assert result["groups"][role]["accepted"] is False
    other = "development_regression" if role == "independent_admission" else "independent_admission"
    assert result["groups"][other]["accepted"] is True


def test_v2_missing_real_rows_and_bad_real_format_are_not_hidden():
    cases, records = v2_gate_fixture()
    records["qwen"] = records["qwen"][:-1]
    assert scoring.grouped_gate(records, cases)["accepted"] is False
    cases, records = v2_gate_fixture()
    for row in records["qwen"][-8:]:
        row["output"] = "invalid JSON"
    result = scoring.grouped_gate(records, cases)
    assert all(group["accepted"] for group in result["groups"].values())
    assert result["all_case_engineering_passed"] is False
    assert result["accepted"] is False


def test_v2_validation_preparation_calls_readonly_verifier(tmp_path, monkeypatch):
    validator = importlib.import_module("frontier_judge_validation_v2")
    scoring.configure_profile("v2")
    monkeypatch.setattr(scoring, "ASSESS", tmp_path)
    called = []
    monkeypatch.setattr(validator, "verify_validation_v2", lambda *args: called.append(args) or {"verified": True})
    assert scoring.prepare_validation([1], [2], {"c": 3}) == {"verified": True}
    assert called == [([1], [2], {"c": 3}, tmp_path / "validation")]


def test_v2_gate_cannot_reuse_v1_pilot_records(tmp_path, monkeypatch):
    scoring.configure_profile("v2")
    monkeypatch.setattr(scoring, "ASSESS", tmp_path)
    folder = tmp_path / "validation"
    folder.mkdir()
    (folder / "manifest.json").write_text('{"fixture":true}')
    manifest_hash = scoring.base.file_sha(folder / "manifest.json")
    protocol = {"system_prompt_sha256": scoring.base.sha(scoring.base.SYSTEM.encode()),
                "validation_sha256": manifest_hash, "model_repo": "fixture", "model_revision": "commit"}
    (folder / "qwen_protocol.json").write_text(json.dumps(protocol))
    row = {"pilot_protocol_sha256": "old-v1-protocol", "validation_sha256": manifest_hash,
           "model_repo": "fixture", "model_revision": "commit"}
    with pytest.raises(ValueError, match="another profile"):
        scoring.verify_pilot_record_provenance("qwen", [row])
    row["pilot_protocol_sha256"] = scoring.base.file_sha(folder / "qwen_protocol.json")
    scoring.verify_pilot_record_provenance("qwen", [row])


@pytest.mark.parametrize("profile,child_count", [("v2", 4), ("validated_v2", 2)])
def test_pipeline_propagates_profile_to_every_gpu_child(tmp_path, monkeypatch, profile, child_count):
    scoring.configure_profile(profile)
    monkeypatch.setattr(scoring, "RUN", tmp_path)
    monkeypatch.setattr(scoring, "publish_summary", lambda *args: None)
    monkeypatch.setattr(scoring, "prepare_materials", lambda: ([], [], {}, {}))
    monkeypatch.setattr(scoring, "prepare_validation", lambda *args: {})
    monkeypatch.setattr(scoring, "prepare_protocol", lambda: None)
    monkeypatch.setattr(scoring, "wait_backend", lambda update: True)
    monkeypatch.setattr(scoring, "check_gate", lambda: {"accepted": True})
    monkeypatch.setattr(scoring, "ensure_execution_allowed", lambda: None)
    monkeypatch.setattr(scoring.signal, "signal", lambda *args: None)
    monkeypatch.setattr(scoring.base, "stop_child", lambda *args: None)
    monkeypatch.setattr(scoring.base, "judge_agreement", lambda: {})
    evaluator = importlib.import_module("evaluate_frontier_automatic")
    calls = []
    monkeypatch.setattr(evaluator, "evaluate_run", lambda *args, **kwargs: calls.append(kwargs) or {"state": "complete"})
    monkeypatch.setattr(scoring.base, "read", lambda p, default=None: json.loads(p.read_text()) if p.exists() else default)
    weights = tmp_path / "weights.json"
    weights.write_text('{"repo":"fixture"}')
    monkeypatch.setattr(scoring, "QWEN_WEIGHTS", weights)
    monkeypatch.setattr(scoring, "PHI_WEIGHTS", weights)
    monkeypatch.setattr(scoring, "ASSESS", tmp_path)
    commands = []
    class Child:
        returncode = 0
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 99900 + len(commands)
            path = tmp_path / ("pilot_" + command[command.index("--model") + 1] + "_status.json" if "--model" in command else "status.json")
            path.write_text(json.dumps({"pid": self.pid, "state": "complete"}))
        def poll(self):
            return 0
    monkeypatch.setattr(scoring.subprocess, "Popen", Child)
    scoring.pipeline()
    assert len(commands) == child_count
    assert all(cmd[-2:] == ["--profile", profile] for cmd in commands)
    if profile == "validated_v2":
        assert commands[0][0] == str(scoring.APP)
        assert commands[1][0] == str(scoring.MODERN)
        assert all("pilot" not in cmd for cmd in commands)
    assert calls[-1] == {"require_complete": True}


def test_validated_profile_has_real_phi_primary_and_independent_directories():
    scoring.configure_profile("validated_v2")
    assert scoring.ASSESS.name == "assessment_validated_v2"
    assert scoring.RUN.name == "frontier_scoring_validated_v2"
    assert scoring.selected_models() == ("phi", "qwen")
    assert scoring.weights_manifest("phi") == scoring.PHI_WEIGHTS
    assert scoring.weights_manifest("qwen") == scoring.QWEN_WEIGHTS
    with pytest.raises(RuntimeError, match="rewriting.*forbidden"):
        scoring.pilot("phi")


def test_failed_qwen_can_only_be_the_sensitivity_reference():
    cases, records = v2_gate_fixture()
    selected = [r for r in records["qwen"] if r["case_id"].startswith("independent_admission:")]
    for row in selected[:8]:
        row["output"] = "invalid JSON"
    assert scoring.grouped_gate(records, cases)["accepted"] is False
    passed = scoring.grouped_gate(records, cases, "phi", "qwen")
    assert passed["accepted"] is True
    assert passed["primary_model"] == "phi" and passed["comparison_reference"] == "qwen"


@pytest.mark.parametrize("judge,expected_backend", [("primary", "phi"), ("secondary", "qwen")])
def test_validated_full_loads_matching_model_and_manifest(monkeypatch, judge, expected_backend):
    scoring.configure_profile("validated_v2")
    monkeypatch.setattr(scoring, "check_gate", lambda: {"accepted": True})
    backends, loaded, invocations = [], [], []
    monkeypatch.setattr(scoring, "backend", lambda name: backends.append(name) or name)
    monkeypatch.setattr(scoring, "verify_weights", lambda path: loaded.append(path) or {"manifest": str(path)})
    def run(args):
        invocations.append(args)
        if args.judge == "primary":
            scoring.base.download_primary()
    monkeypatch.setattr(scoring.base, "run_assessment", run)
    scoring.run_full(judge)
    assert backends == [expected_backend]
    assert invocations[0].secondary_weights == scoring.QWEN_WEIGHTS
    assert loaded == ([scoring.PHI_WEIGHTS] if judge == "primary" else [])


def test_validated_direct_full_cannot_bypass_admission(monkeypatch):
    scoring.configure_profile("validated_v2")
    monkeypatch.setattr(scoring, "check_gate", lambda: {"accepted": False})
    monkeypatch.setattr(scoring, "backend", lambda *args: pytest.fail("GPU backend must not be loaded"))
    with pytest.raises(RuntimeError, match="must pass"):
        scoring.run_full("primary")


@pytest.fixture
def linked_validations(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "ROOT", tmp_path)
    monkeypatch.setattr(scoring, "V2_ASSESS", tmp_path / "v2")
    monkeypatch.setattr(scoring, "ASSESS", tmp_path / "validated")
    folder = scoring.V2_ASSESS / "validation"
    folder.mkdir(parents=True)
    for name in ("cases.jsonl", "manifest.json", "phi_records.jsonl", "qwen_records.jsonl",
                 "phi_protocol.json", "qwen_protocol.json", "phi_evaluation.json", "qwen_evaluation.json"):
        (folder / name).write_text(name + "\n")
    (folder / "admission.json").write_text('{"accepted":false}')
    return folder


def test_validation_references_cannot_overwrite_old_failed_admission(linked_validations):
    scoring.link_validation_records()
    folder = scoring.ASSESS / "validation"
    assert folder.is_dir() and not folder.is_symlink()
    assert (folder / "phi_records.jsonl").is_symlink()
    assert not (folder / "admission.json").exists()
    (folder / "admission.json").write_text('{"accepted":true}')
    assert (linked_validations / "admission.json").read_text() == '{"accepted":false}'


def test_whole_validation_directory_link_is_refused(linked_validations):
    scoring.ASSESS.mkdir(parents=True)
    (scoring.ASSESS / "validation").symlink_to(linked_validations, target_is_directory=True)
    with pytest.raises(ValueError, match="never a directory symlink"):
        scoring.link_validation_records()
    assert (linked_validations / "admission.json").read_text() == '{"accepted":false}'


def test_failed_v2_cannot_be_restarted(tmp_path, monkeypatch):
    scoring.configure_profile("v2")
    monkeypatch.setattr(scoring, "V2_ASSESS", tmp_path)
    path = tmp_path / "validation/admission.json"
    path.parent.mkdir()
    path.write_text('{"accepted":false}')
    with pytest.raises(RuntimeError, match="v2 Qwen admission failed"):
        scoring.ensure_execution_allowed()
