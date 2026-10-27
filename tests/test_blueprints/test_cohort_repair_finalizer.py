"""Fake registered artifacts only; no process inspection, fitting or solver."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("repair_finalizer_tests", ROOT / "scripts/finalize_blueprint_cohort_repair.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    def write(path, payload):
        cli.save(path, payload)
    cohort, old, repair, out = (tmp_path / p for p in ("cohort", "cohort/cases/toy/attempt_001", "repair", "new_summary"))
    out.mkdir()
    methods = [dict(method="M0", seed=17), dict(method="M2", seed=17)]
    case = dict(case_id="toy", compound_id="toy_drug", tier="reconstruction",
                parameter_search={"parameters": [{"parameter_id": "p", "initial_value": 1}]})
    external = [dict(scenario_id="external", study_group_id="study")]
    write(tmp_path / "bundle.json", dict(case=case, external_scenarios=external))
    manifest = {"cases": [dict(case_id="toy", compound_id="toy_drug", tier="reconstruction", status="ready",
                               bundle_path="bundle.json", bundle_sha256=cli.sha("bundle.json"))]}
    write(tmp_path / "input_manifest.json", manifest)
    write(cohort / "cohort.json", manifest)
    planned = [dict(j, case_id="toy", compound_id="toy_drug", tier="reconstruction", eligibility_status="ready") for j in methods]
    write(cohort / "configuration.json", dict(input_manifest_path="input_manifest.json",
        input_manifest_sha256=cli.sha("input_manifest.json"), jobs=planned))
    queue = [dict(case_id="toy", status="failed", reason="DTO engineering failure", output=str(old.relative_to(tmp_path)))]
    write(cohort / "queue.json", dict(cases=queue))
    write(cohort / "completion.json", dict(status="queue_finished", all_eligible_queues_terminal=True))
    write(cohort / "cohort_prediction_barrier.json", dict(locked_at_utc="original-time", all_eligible_drug_queues_terminal=True))
    old_code = {cli.ALLOWED_REPAIR: b"# old DTO\n", "scripts/run_blueprint_experiments.py": b"# frozen runner\n"}
    new_code = dict(old_code, **{cli.ALLOWED_REPAIR: b"# repaired DTO\n"})
    for name, data in new_code.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    for directory, code in ((old, old_code), (repair, new_code)):
        write(directory / "case.json", case)
        write(directory / "external_scenario_designs.json", external)
        config = dict(case_bundle_path="bundle.json", case_bundle_sha256=cli.sha("bundle.json"),
            case_sha256=cli.sha(directory / "case.json"), jobs=methods, candidates=[{"candidate_id": "base"}],
            base_candidate="base", device="cpu", max_solver_calls_per_method=5000,
            external_scoring_deferred=True, external_observations_available_to_optimizer=False,
            code_sha256={name: hashlib.sha256(value).hexdigest() for name, value in code.items()})
        write(directory / "configuration.json", config)
        with zipfile.ZipFile(directory / "runner_source.zip", "x") as archive:
            for name, value in code.items(): archive.writestr(name, value)
    curve = dict(time_min=[0, 1], concentration=[1, 1], unit="umol/l", time_unit="min")
    fallback = {"external": curve}
    write(repair / "external_fallback_predictions.json", dict(candidate_id="base", parameter_values=[1], curves=fallback))
    locks, results = [], []
    for j in methods:
        blocked = j["method"] == "M2"
        curves = {"external": None if blocked else curve}
        lock = dict(j, candidate_id=None if blocked else "base", status="blocked" if blocked else "completed",
                    predictions_sha256=cli.digest(dict(predicted_curves=curves, fallback_curves=fallback)))
        record = dict(j, locked_selection=lock, curves=curves, fallback_curves=fallback, submission_status=lock["status"])
        write(repair / "methods" / ("%s_seed_%s" % (j["method"], j["seed"])) / "external_predictions.json", record)
        locks.append(lock)
        results.append(dict(j, status=lock["status"], selected_candidate_id=lock["candidate_id"],
                            evaluated_candidates=0 if blocked else 1, blocked_reason="no_service" if blocked else None))
    write(repair / "results.json", dict(methods=results))
    write(repair / "external_prediction_locks.json", dict(locks=locks, all_method_predictions_finished_before_external_observation_read=True))
    write(repair / "completion.json", dict(status="completed_with_blocked_methods"))
    return SimpleNamespace(root=tmp_path, cohort=cohort, repair=repair, out=out, old=old, queue=queue, write=write)


def contract(a):
    return cli.original_contract(a.cohort, "toy")


def test_registered_dto_only_repair_is_accepted(artifacts):
    a = artifacts
    cfg = cli.repair_contract(contract(a), a.repair)
    records, results = cli.frozen_submissions(contract(a), a.repair, cfg)
    assert len(records) == len(results["methods"]) == 2
    assert records[1]["submission_status"] == "blocked"


@pytest.mark.parametrize("change", ["budget", "candidate_space", "bundle", "extra_code"])
def test_repair_cannot_change_science_or_unregistered_code(artifacts, change):
    a = artifacts; cfg = cli.read_json(a.repair / "configuration.json")
    if change == "budget": cfg["max_solver_calls_per_method"] = 9000
    elif change == "candidate_space": cfg["candidates"] = []
    elif change == "bundle": cfg["case_bundle_sha256"] = "a" * 64
    else: cfg["code_sha256"]["scripts/run_blueprint_experiments.py"] = "a" * 64
    a.write(a.repair / "configuration.json", cfg)
    with pytest.raises(ValueError): cli.repair_contract(contract(a), a.repair)


def test_main_holdout_cannot_be_claimed_for_post_unblinding_repair(artifacts):
    a = artifacts; cfg = cli.read_json(a.cohort / "configuration.json")
    cfg["jobs"][0]["tier"] = "main_holdout"; a.write(a.cohort / "configuration.json", cfg)
    with pytest.raises(ValueError, match="strict-main"): contract(a)


@pytest.mark.parametrize("change", ["prediction", "fallback", "missing_method", "selected_model"])
def test_locked_submissions_are_checked_without_reading_final_observations(artifacts, change):
    a = artifacts; c = contract(a); cfg = cli.repair_contract(c, a.repair)
    path = a.repair / "methods/M0_seed_17/external_predictions.json"; record = cli.read_json(path)
    if change == "prediction": record["curves"]["external"]["concentration"][0] = 2
    elif change == "fallback": record["fallback_curves"]["external"]["concentration"][0] = 2
    elif change == "selected_model": record["locked_selection"]["candidate_id"] = "other"
    else:
        lock = cli.read_json(a.repair / "external_prediction_locks.json"); lock["locks"].pop()
        a.write(a.repair / "external_prediction_locks.json", lock)
    a.write(path, record)
    with pytest.raises(ValueError): cli.frozen_submissions(c, a.repair, cfg)


def test_wait_checks_both_queues_and_never_sleeps_after_ready(artifacts, monkeypatch):
    a = artifacts
    monkeypatch.setattr(cli.time, "sleep", lambda _: pytest.fail("Both terminal queues should not sleep"))
    config, queue = cli.wait_for_terminal(a.cohort, a.repair, a.out, contract(a), timeout_seconds=1)
    assert config["device"] == "cpu"
    assert queue == a.queue
    assert cli.read_json(a.out / "progress.json")["final_observations_read"] is False


def test_wait_is_bounded_and_repair_failure_does_not_trigger_scoring(artifacts, monkeypatch):
    a = artifacts; (a.repair / "completion.json").unlink()
    now = [0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    with pytest.raises(TimeoutError):
        cli.wait_for_terminal(a.cohort, a.repair, a.out, contract(a), timeout_seconds=1)
    assert now[0] == 1
    a.write(a.repair / "progress.json", dict(status="failed", error="native error"))
    with pytest.raises(RuntimeError, match="Repair queue failed"):
        cli.wait_for_terminal(a.cohort, a.repair, a.out, contract(a), timeout_seconds=1)


def test_finalize_keeps_original_failure_and_repair_as_one_scientific_submission(artifacts, monkeypatch):
    a = artifacts; c = contract(a); cfg = cli.repair_contract(c, a.repair)
    before = {p: cli.sha(p) for directory in (a.cohort, a.repair) for p in directory.rglob("*") if p.is_file()}
    def score(case, external, records, output, update, *, device):
        assert output.is_relative_to(a.out)
        assert cli.read_json(a.out / "repair_prediction_registration.json")["is_original_cohort_global_blinding_barrier"] is False
        reports = []
        for r in records:
            fallback = r["submission_status"] == "blocked"
            row = dict(scenario_id="external", curve_id="external", study_group_id="study", compound_id="toy_drug",
                tier="reconstruction", study_role="external", duplicate_of=None, status="scored_with_fallback" if fallback else "scored",
                fallback_used=fallback, log_rmse=.5, gmfe=1.5, endpoints={})
            report = dict(method=r["method"], seed=r["seed"], candidate_id=r["locked_selection"]["candidate_id"], rows=[row],
                predictions_sha256=r["locked_selection"]["predictions_sha256"], prediction_integrity_verified=True,
                candidate_selection_performed=False)
            reports.append(dict(method=r["method"], seed=r["seed"], tier="reconstruction", report=report))
        a.write(output / "external_results.json", dict(compound_id="toy_drug", evaluations=reports))
    monkeypatch.setattr(cli, "single_runner", lambda: SimpleNamespace(
        load_case_bundle=lambda _: (c["payload"]["case"], c["payload"]["external_scenarios"]), score_external=score))
    result = cli.finalize(a.cohort, a.repair, a.out, "toy", c, cfg, a.queue)
    assert result["planned_jobs"] == 2  # not original+repair=4 pseudo-submissions
    assert result["strict_independent_main_completed"] is False
    audit = cli.read_json(a.out / "repair_audit.json")
    assert len(audit["original_case_jobs"]) == len(audit["repair_case_jobs"]) == 2
    assert {j["status"] for j in audit["original_case_jobs"]} == {"failed"}
    assert audit["solver_calls"] == 0
    assert audit["replacement_adds_independent_drugs_or_studies"] is False
    assert {p: cli.sha(p) for p in before} == before
