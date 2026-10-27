"""CPU regression for profile isolation, early gates and two pilot cohorts."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from frozen_frontier_sources import materialize_profile_inputs, materialize_scripts

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v = importlib.import_module("run_frontier_scoring_reviewed")
REAL_ROOT = Path(v.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_modules():
    engine = {k: getattr(v.engine, k) for k in v.INJECTED}
    base = {k: getattr(v.base, k) for k in ("SYSTEM", "ASSESS", "RUN", "prepare", "GPUJudge", "download_primary")}
    own = {k: getattr(v, k) for k in ("PROFILE", "CONFIG", "CONFIG_SHA", "ASSESS", "RUN", "SYSTEM")}
    assess = v.utilities.ASSESS
    yield
    for k, value in engine.items(): setattr(v.engine, k, value)
    for k, value in base.items(): setattr(v.base, k, value)
    for k, value in own.items(): setattr(v, k, value)
    v.utilities.ASSESS = assess


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    out = tmp_path / "experiments"
    assess, old = out / "assessment_precedence_v5", out / "assessment_evidence_first_v3"
    run = tmp_path / "runtime/deployment/frontier_scoring_reviewed_v5"
    for path in (assess, old, run): path.mkdir(parents=True)
    for key, value in (("ROOT", tmp_path), ("OUT", out), ("ASSESS", assess), ("LEGACY", old), ("RUN", run),
                       ("PROFILE", "v5"), ("SYSTEM", "frozen test prompt"), ("CONFIG_SHA", "config-hash")):
        monkeypatch.setattr(v, key, value)
    monkeypatch.setattr(v, "CONFIG", {"legacy_manifest_sha256": "legacy-hash", "fresh_manifest_sha256": "fresh-hash"})
    return assess, old, run


@pytest.fixture
def config(paths):
    prompt = v.ROOT / "scripts/frontier_judge_prompt_test.py"
    prompt.parent.mkdir(); prompt.write_text('SYSTEM = "fixture prompt"\n')
    value = {"version": "pbpk-reviewed-judge-profile-v1", "profile": "v5",
        "assessment_directory": "assessment_precedence_v5", "runtime_directory": "frontier_scoring_reviewed_v5",
        "predecessor_assessment_directory": "assessment_conditional_v4", "prompt_file": "scripts/frontier_judge_prompt_test.py",
        "prompt_file_sha256": v.base.file_sha(prompt), "system_prompt_sha256": v.base.sha(b"fixture prompt"),
        "thresholds": v.GATE, "representative_minimum": .98, "representative_window": 128,
        "representative_max_invalid_fraction": .1, "regression_disclosure": v.DISCLOSURE}
    path = v.ROOT / "runtime/deployment/judge_profiles/v5.json"
    put(path, value)
    path.with_suffix(".sha256").write_text(v.base.file_sha(path)+"\n")
    return path, value


def test_config_loads_only_frozen_project_profile_and_propagates_child_slug(config):
    path, value = config
    assert v.load_profile("v5") == value
    v.configure()
    assert v.base.SYSTEM == "fixture prompt" and v.engine.prepare is v.prepare
    assert v.engine.check_gate is v.check_gate and v.engine.backend is v.backend
    assert v.command(v.APP, "pilot", "--model", "phi")[2:] == ["--profile", "v5", "pilot", "--model", "phi"]
    assert v.CONFIG_SHA == v.base.file_sha(path)


@pytest.mark.parametrize("slug", ["../v5", "v4", "v5/other", "/v5", "v5;echo", "arbitrary"])
def test_config_slug_rejects_paths_and_unapproved_names(slug):
    with pytest.raises(ValueError, match="version slug"): v.load_profile(slug)


def test_changed_config_or_prompt_is_rejected(config):
    path, value = config
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="independently frozen"): v.load_profile("v5")
    path.with_suffix(".sha256").write_text(v.base.file_sha(path))
    (v.ROOT / value["prompt_file"]).write_text('SYSTEM="changed"')
    with pytest.raises(ValueError, match="dependency changed"): v.load_profile("v5")


def test_profile_cannot_select_old_runtime_or_external_prompt(config):
    path, value = config
    value["runtime_directory"] = "frontier_scoring_v3"
    put(path, value); path.with_suffix(".sha256").write_text(v.base.file_sha(path))
    with pytest.raises(ValueError, match="configuration mismatch"): v.load_profile("v5")
    with pytest.raises(ValueError, match="scoped project scripts"): v.script_path("../secrets.py")


def test_legacy_definitions_linked_without_any_prior_predictions(paths):
    assess, old, run = paths
    for name in ("validation/cases.jsonl", "validation/manifest.json", "source_review_plan.json"):
        put(old / name, {})
    put(old / "validation/phi_records.jsonl", {"old": True})
    v.link_inputs()
    assert (assess / "validation/cases.jsonl").is_symlink()
    assert not (assess / "validation").is_symlink()
    assert not (assess / "validation/phi_records.jsonl").exists()
    (assess / "validation/phi_records.jsonl").symlink_to(old / "validation/phi_records.jsonl")
    with pytest.raises(ValueError, match="prior predictions"): v.link_inputs()


def fixture_output(case):
    expected = case["expected"]
    if expected is None: return '[{"i":1,"g":null,"a":"U","e":[]}]'
    refs = sorted({choices[0] for choices in expected["required_support_any_of"]})
    return json.dumps([{"i": 1, "g": expected["grades"][0],
        "a": {True: "Y", False: "N", None: "U"}[expected["documented_applicability"]], "e": refs}])


@pytest.fixture
def pilot(paths, monkeypatch):
    assess, old, run = paths
    all_cases = v.base.read_rows(REAL_ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_evidence_first_v3/validation/cases.jsonl")
    cases = [copy.deepcopy(next(c for c in all_cases if c["strata"]["validation_role"] == role and c["expected"]["grades"] == [3]))
             for role in v.CONTROL_ROLES]
    real = copy.deepcopy(next(c for c in all_cases if c["kind"] == "real_source_pair"))
    cases.append(real)
    fresh = [copy.deepcopy(cases[0])]
    fresh[0]["case_id"] = "fresh-fixture"
    fresh[0]["query"]["query_id"] = "fresh-query"
    (assess / "validation/fresh").mkdir(parents=True)
    v.base.append_rows(assess / "validation/cases.jsonl", cases)
    v.base.append_rows(assess / "validation/fresh/cases.jsonl", fresh)
    put(assess / "validation/manifest.json", {"fixture": "legacy"})
    put(assess / "validation/fresh/manifest.json", {"fixture": "fresh"})
    put(assess / "judge_protocol.json", {"fixture": "new-profile"})
    for model in ("phi", "qwen"):
        path = assess / f"{model}_weights.json"; put(path, {"repo": model, "revision": "fixed"})
        monkeypatch.setattr(v, "PHI_WEIGHTS" if model == "phi" else "QWEN_WEIGHTS", path)
    monkeypatch.setattr(v, "prepare", lambda: None)
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(v.utilities, "verify_weights", lambda path: v.base.read(path))
    evaluator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(evaluator, "evaluate_validation_v3", lambda rows, cohort: {"complete": len(rows) == len(cohort)})
    return cases, fresh


def fake_backend(monkeypatch, cases, fresh, fail_ids=(), broken_source=False):
    by_query = {c["query"]["query_id"]: c for c in [*cases, *fresh]}
    calls, loads = [], []
    class Judge:
        def __init__(self, weights, *args): loads.append(weights["repo"])
        def select_batch_size(self, tasks, maximum): return len(tasks)
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            outputs = []
            for packet, queries in tasks:
                query = queries[0]
                assert "expected" not in query and "gold" not in query
                c = by_query[query["query_id"]]
                calls.append((c["case_id"], max_new_tokens, repairs is not None))
                if c["case_id"] in fail_ids: output = '[{"i":1,"g":null,"a":"U","e":[]}]'
                elif broken_source and c["kind"] != "synthetic_control" and not repairs: output = "["
                else: output = fixture_output(c)
                outputs.append(output)
            return outputs, {"wall_seconds": 1}
    monkeypatch.setattr(v, "backend", lambda model: Judge)
    return calls, loads


def test_primary_complete_group_failure_stops_before_next_group_or_fresh_and_restart(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh, fail_ids={cases[0]["case_id"]})
    v.pilot("phi")
    assert [x[0] for x in calls] == [cases[0]["case_id"]]
    state = v.base.read(v.RUN / "pilot_phi_status.json")
    assert state["state"] == "stopped_on_control_gate" and state["completed"] == 1
    assert state["control_metrics"]["complete"] and state["fresh_completed"] == 0
    assert not (v.ASSESS / "validation/fresh/phi_records.jsonl").exists()
    with pytest.raises(RuntimeError, match="identical restart"): v.execute("start")


def test_fresh_failure_stops_before_source_generation(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh, fail_ids={fresh[0]["case_id"]})
    v.pilot("phi")
    assert [c[0] for c in calls] == [c["case_id"] for c in cases[:3]] + [fresh[0]["case_id"]]
    state = v.base.read(v.RUN / "pilot_phi_status.json")
    assert state["failed_control_group"] == "fresh32" and state["legacy_completed"] == 3


def test_all_jobs_keep_gold_out_of_prompts_store_separate_journals_and_repair_real_once(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh, broken_source=True)
    v.pilot("phi")
    assert loads == ["phi"]
    assert [c[0] for c in calls[:4]] == [c["case_id"] for c in cases[:3]] + [fresh[0]["case_id"]]
    assert calls[-2:] == [(cases[-1]["case_id"], 96, False), (cases[-1]["case_id"], 160, True)]
    legacy, legacy_raw = v.read_pilot("phi", "legacy", cases)
    independent, fresh_raw = v.read_pilot("phi", "fresh", fresh)
    assert set(legacy) == {c["case_id"] for c in cases} and set(independent) == {fresh[0]["case_id"]}
    assert len(legacy[cases[-1]["case_id"]]["attempts"]) == 2
    assert all(len(r["attempts"]) == 1 for r in independent.values())
    assert v.base.read(v.RUN / "pilot_phi_status.json")["state"] == "complete"
    monkeypatch.setattr(v, "backend", lambda _: pytest.fail("Completed profile must resume without loading GPU"))
    v.pilot("phi")


def test_qwen_reference_absolute_failure_is_reported_but_does_not_stop_remaining_cases(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh, fail_ids={cases[0]["case_id"]})
    v.pilot("qwen")
    assert len(calls) == len(cases)+len(fresh)
    assert v.base.read(v.RUN / "pilot_qwen_status.json")["state"] == "complete"
    gate = v.base.read(v.ASSESS / "validation" / f"qwen_{v.CONTROL_ROLES[0]}_gate.json")
    assert gate["absolute_failed_checks"] and not gate["primary"]


def test_independent_fresh_job_does_not_claim_legacy_completion(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh)
    v.pilot("phi", "fresh")
    state = v.base.read(v.RUN / "pilot_phi_status.json")
    assert state["state"] == "job_complete" and state["legacy_completed"] == 0 and state["fresh_completed"] == 1
    assert [c[0] for c in calls] == [fresh[0]["case_id"]]
    assert not (v.ASSESS / "validation/phi_records.jsonl").exists()


def test_primary_source_only_job_requires_all_four_controls_complete(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh)
    with pytest.raises(RuntimeError, match="all four primary control"): v.pilot("phi", "legacy-sources")
    assert not calls and not loads


def test_old_cohort_protocol_or_records_cannot_resume_as_new_profile(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh)
    v.pilot("phi", "fresh")
    path = v.ASSESS / "validation/fresh/phi_records.jsonl"
    rows = v.base.load_committed(path); rows[0]["pilot_protocol_sha256"] = "old-version"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError, match="provenance mismatch"): v.pilot("phi", "fresh")


def test_missing_fresh_records_blocks_gate_before_legacy_acceptance(pilot, monkeypatch):
    cases, fresh = pilot
    fake_backend(monkeypatch, cases, fresh)
    v.pilot("phi", "legacy-controls")
    monkeypatch.setattr(v, "ENGINE_CHECK_GATE", lambda **kwargs: pytest.fail("No legacy-only acceptance"))
    with pytest.raises(ValueError, match="complete 733"): v.check_gate(False)


@pytest.mark.parametrize("fresh_primary,fresh_reference,accepted", [(1., 1., True), (.89, .9, False), (.95, 1., False)])
def test_fresh_gate_keeps_absolute_and_same_relative_thresholds(paths, monkeypatch, fresh_primary, fresh_reference, accepted):
    cases, fresh = [{"case_id": "old"}], [{"case_id": "new"}]
    monkeypatch.setattr(v, "prepare", lambda: None)
    monkeypatch.setattr(v, "cohorts", lambda: (cases, fresh))
    monkeypatch.setattr(v, "read_pilot", lambda model, cohort, cs: ({c["case_id"]: {"case_id": c["case_id"], "model": model} for c in cs}, {c["case_id"]: {} for c in cs}))
    def metric(rows, cs):
        score = fresh_primary if rows[0]["model"] == "phi" else fresh_reference
        return {"complete": True, "controlled_semantic_accuracy": score, "controlled_citation_support_accuracy": 1.,
                "citation_numbers_valid_fraction": 1., "raw_format_valid_fraction": 1.}
    monkeypatch.setattr(v, "control_metric", metric)
    monkeypatch.setattr(v, "ENGINE_CHECK_GATE", lambda require_review: {"numeric_accepted": True, "accepted": False})
    monkeypatch.setattr(v, "verify_source_review", lambda: (True, "review-sha"))
    assert v.check_gate()["accepted"] is accepted
    assert v.check_gate(False)["accepted"] is False


def test_gpu_full_entry_uses_the_combined_gate(paths, monkeypatch):
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(v, "check_gate", lambda: {"accepted": False})
    monkeypatch.setattr(v, "backend", lambda _: pytest.fail("No GPU before fresh admission"))
    with pytest.raises(RuntimeError, match="admission has not passed"): v.execute("run")


def test_other_project_profile_worker_blocks_execution(paths, monkeypatch):
    put(v.ROOT / "runtime/deployment/frontier_scoring_conditional_v4/pid.json", {"own_fixture_active": True})
    monkeypatch.setattr(v.base, "alive", lambda value: value.get("own_fixture_active", False))
    with pytest.raises(RuntimeError, match="prior project frontier"): v.execute("pilot")


def test_partial_control_metrics_do_not_fabricate_early_failure():
    assert v.absolute_failures({"complete": False, "controlled_semantic_accuracy": 0.}) == []


def test_frozen_parent_pipeline_stops_after_actual_primary_control_failure_without_qwen(pilot, monkeypatch):
    cases, fresh = pilot
    calls, loads = fake_backend(monkeypatch, cases, fresh, fail_ids={cases[0]["case_id"]})
    monkeypatch.setattr(v, "publish_summary", lambda: None)
    monkeypatch.setattr(v.engine.signal, "signal", lambda *args: None)
    monkeypatch.setattr(v.base, "stop_child", lambda child: None)
    launches = []
    class Child:
        pid, returncode = 200, 0
        def __init__(self, command, **kwargs):
            launches.append(command)
            v.pilot("phi")
            path = v.RUN / "pilot_phi_status.json"
            state = v.base.read(path); state["pid"] = self.pid; put(path, state)
        def poll(self): return self.returncode
    monkeypatch.setattr(v.engine.subprocess, "Popen", Child)
    v.execute("pipeline")
    status = v.base.read(v.RUN / "pipeline_status.json")
    assert len(launches) == 1 and launches[0][-3:] == ["pilot", "--model", "phi"]
    assert launches[0][2:4] == ["--profile", "v5"]
    assert status["state"] == "failed" and status["completed"] == 1
    assert status["guard_reason"].startswith("AbsoluteControlAdmissionFailed")
    assert status["failed_stage"] == "pilot_phi" and loads == ["phi"]


def test_actual_fresh_record_deletion_blocks_full_gate(pilot, monkeypatch):
    cases, fresh = pilot
    fake_backend(monkeypatch, cases, fresh)
    v.pilot("phi"); v.pilot("qwen")
    (v.ASSESS / "validation/fresh/qwen_records.jsonl").write_text("")
    monkeypatch.setattr(v, "ENGINE_CHECK_GATE", lambda **kwargs: pytest.fail("Cannot accept legacy-only outputs"))
    with pytest.raises(ValueError, match="complete 32"): v.check_gate(False)


@pytest.fixture
def review(paths, monkeypatch):
    assess, old, run = paths
    put(assess / "judge_protocol.json", {"profile": "v5"})
    for cohort in ("legacy", "fresh"):
        folder = v.cohort_folder(cohort); folder.mkdir(parents=True, exist_ok=True)
        for model in ("phi", "qwen"): (folder / f"{model}_records.jsonl").write_text('{}\n')
    legacy = [{"case_id": "failure", "strata": {"validation_role": "production_failure_diagnostic"}},
              {"case_id": "source", "strata": {"validation_role": "prospective_source_coverage"}}]
    monkeypatch.setattr(v, "cohorts", lambda: (legacy, []))
    monkeypatch.setattr(v.engine, "verify_review_plan", lambda cases: {"case_ids": ["failure", "source"]})
    value = {"version": "pbpk-reviewed-source-review-v1", "profile": "v5",
        "protocol_sha256": v.base.file_sha(assess / "judge_protocol.json"),
        "validation_manifest_sha256": "legacy-hash", "fresh_manifest_sha256": "fresh-hash",
        "pilot_record_sha256": {m: v.base.file_sha(assess / "validation" / f"{m}_records.jsonl") for m in ("phi", "qwen")},
        "fresh_pilot_record_sha256": {m: v.base.file_sha(assess / "validation/fresh" / f"{m}_records.jsonl") for m in ("phi", "qwen")},
        "checked_case_ids": ["failure", "source"], "reviewer": "Codex evidence audit; not expert gold",
        "scientific_gold_accuracy": None, "findings": [], "known_unknown_distribution": {}, "accepted": True}
    put(assess / "source_review.json", value)
    return value


def test_source_review_binds_new_fresh_outputs_as_well_as_39_source_plan(review):
    assert v.verify_source_review()[0]
    review["fresh_pilot_record_sha256"]["phi"] = "another-profile"
    put(v.ASSESS / "source_review.json", review)
    with pytest.raises(ValueError, match="both independent pilot cohorts"): v.verify_source_review()


def test_source_review_missing_planned_case_and_false_acceptance_are_not_ignored(review):
    review["accepted"] = False; put(v.ASSESS / "source_review.json", review)
    assert v.verify_source_review()[0] is False
    review["checked_case_ids"] = ["failure"]; put(v.ASSESS / "source_review.json", review)
    with pytest.raises(ValueError, match="planned coverage"): v.verify_source_review()


@pytest.mark.parametrize("model", ["phi", "qwen"])
def test_backend_retains_conditional_decoder_for_both_models(monkeypatch, model):
    decoder = importlib.import_module("frontier_conditional_citation_decoding")
    seen = []
    class Prepared:
        def __init__(self, *args): self.json_constraint = "official"
    monkeypatch.setattr(v.utilities, "PhiJudge", Prepared)
    monkeypatch.setattr(v.utilities, "backend", lambda model: Prepared)
    monkeypatch.setattr(decoder, "conditional_citation_constraint", lambda value: seen.append(value) or "conditional")
    assert v.backend(model)({}, 36).json_constraint == "conditional" and seen == ["official"]


def test_current_english_revision_cannot_resume_a_frozen_historical_profile(monkeypatch):
    v.load_profile("v5")
    actual_assessment = v.ASSESS
    monkeypatch.setattr(v, "assert_prior_runs_idle", lambda: None)
    with tempfile.TemporaryDirectory(prefix="reviewed-profile-fixture-", dir=REAL_ROOT / "runtime/test_tmp") as directory:
        fixture = Path(directory)
        monkeypatch.setattr(v, "ASSESS", fixture / "assessment")
        monkeypatch.setattr(v, "RUN", fixture / "runtime")
        (v.ASSESS / "validation/fresh").mkdir(parents=True)
        v.RUN.mkdir()
        for name in ("cases.jsonl", "manifest.json"):
            (v.ASSESS / "validation/fresh" / name).symlink_to(actual_assessment / "validation/fresh" / name)
        with pytest.raises(ValueError, match="Frozen engine dependency changed"):
            v.prepare()
        assert not (v.ASSESS / "validation/phi_records.jsonl").exists()
        assert not (v.ASSESS / "judgments.jsonl").exists()


def test_real_historical_profile_prepares_in_isolated_revision_and_matches_evaluator(tmp_path):
    materialize_scripts(tmp_path)
    materialize_profile_inputs(tmp_path)
    # A fresh process imports the actual old modules, not current modules with
    # substituted hashes. Only CPU interpreter metadata uses the test runtime.
    code = r'''
import importlib
from pathlib import Path
import socket
import sys
def no_network(*args, **kwargs):
    raise AssertionError("Historical preparation fixture must not use the network")
socket.socket.connect = no_network
import run_frontier_scoring_reviewed as v
v.utilities.APP = v.utilities.MODERN = Path(sys.executable)
value = v.load_profile("v5")
queries, candidates, packets, protocol = v.prepare()
assert len(queries) == 772 and len(candidates) == len(packets) == 280
assert protocol["profile_config"] == value and protocol["profile_config_sha256"] == v.CONFIG_SHA
assert protocol["system_prompt_sha256"] == value["system_prompt_sha256"]
assert protocol["generation"]["constrained_decoding"]["field_order"] == ["i", "g", "a", "e"]
assert protocol["validation_policy"]["legacy_case_count"] == 733
assert protocol["validation_policy"]["fresh_case_count"] == 32
assert protocol["implementation_sha256"][Path(v.__file__).name] == v.base.file_sha(Path(v.__file__))
assert len(v.cohorts()[0]) == 733 and len(v.cohorts()[1]) == 32
evaluator = importlib.import_module("evaluate_frontier_automatic")
result = evaluator.evaluate_run(v.OUT, v.ASSESS, protocol_only=True)
metadata = evaluator.read_judge_metadata(v.ASSESS, result["protocol"])
assert metadata["state"] == "frozen" and metadata["primary_repo"] == "microsoft/phi-4"
assert metadata["secondary_repo"] == "Qwen/Qwen3.6-27B"
assert v.prepare()[3] == protocol
assert not (v.ASSESS / "validation/phi_records.jsonl").exists()
assert not (v.ASSESS / "judgments.jsonl").exists()
print("Historical preparation and evaluator verified; no inference performed")
'''
    environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(tmp_path / "scripts") + os.pathsep + str(REAL_ROOT / "src"),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    assert "Historical preparation and evaluator verified" in completed.stdout
