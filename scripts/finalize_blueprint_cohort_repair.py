"""Wait for one registered DTO repair, then evaluate its locked predictions.

No process inspection, model fitting, native simulation, or LLM calls occur here.
Polling uses only the standard library. Original queues/results remain untouched.
This historical repair is explicitly not the cohort's first global blind test.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_REPAIR = "src/pbpk_kb/blueprints/human_model_snapshot.py"
POLL_SECONDS = 20
MAX_WAIT_SECONDS = 8 * 60 * 60
MAX_JSON_BYTES = 64 * 1024**2
TERMINAL_CASES = {"blocked", "failed", "evaluated", "evaluation_failed", "predictions_locked"}
COMPLETE_REPAIR = {"completed", "completed_with_blocked_methods", "completed_with_unsuccessful_methods"}


def inside(value, *, exists=True):
    path = (ROOT / value).resolve(strict=exists)
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError("Repair resources must remain inside the project")
    return path


def read_json(path):
    path = inside(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Registered JSON artifact exceeds the bounded reader limit")
    return json.loads(path.read_text())


def sha(path):
    with inside(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    path = inside(path, exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _jobs(rows):
    keys = [(r["method"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or any(not isinstance(m, str) or type(s) is not int for m, s in keys):
        raise ValueError("Method/seed registrations must be unique")
    return set(keys)


def original_contract(cohort, case_id):
    config = read_json(cohort / "configuration.json")
    manifest = read_json(cohort / "cohort.json")
    source = inside(config["input_manifest_path"])
    if sha(source) != config["input_manifest_sha256"]:
        raise ValueError("Original cohort input manifest changed")
    cases = [c for c in manifest["cases"] if c["case_id"] == case_id]
    if len(cases) != 1 or cases[0]["status"] != "ready":
        raise ValueError("Repair must refer to one originally eligible registered case")
    case = cases[0]
    if case["tier"] not in {"development", "reconstruction"}:
        raise ValueError("This post-unblinding engineering repair is historical, not main_holdout")
    if any(j["tier"] == "main_holdout" for j in config["jobs"]):
        raise ValueError("Do not mix this historical repair with a strict-main cohort")
    queue = read_json(cohort / "queue.json")["cases"]
    targets = [r for r in queue if r["case_id"] == case_id]
    if len(targets) != 1 or targets[0]["status"] != "failed":
        raise ValueError("Replacement requires a registered engineering-failed original attempt")
    old_dir = inside(targets[0]["output"])
    if not old_dir.is_relative_to(cohort):
        raise ValueError("The original attempt is not owned by this cohort")
    if (old_dir / "external_results.json").exists():
        raise ValueError("A scored original attempt cannot be replaced after seeing its performance")
    old_config = read_json(old_dir / "configuration.json")
    bundle = inside(case["bundle_path"])
    if sha(bundle) != case["bundle_sha256"] or old_config["case_bundle_sha256"] != case["bundle_sha256"]:
        raise ValueError("Original case bundle hash changed")
    payload = read_json(bundle)
    if (payload["case"]["case_id"] != case_id or payload["case"]["compound_id"] != case["compound_id"]
            or payload["case"]["tier"] != case["tier"]):
        raise ValueError("Original case identity/tier differs from its frozen bundle")
    planned = [j for j in config["jobs"] if j["case_id"] == case_id]
    if _jobs(planned) != _jobs(old_config["jobs"]):
        raise ValueError("Original method definitions differ from the cohort registration")
    if sha(old_dir / "case.json") != old_config["case_sha256"] or read_json(old_dir / "case.json") != payload["case"]:
        raise ValueError("Original case snapshot differs from its registered inputs")
    return {"configuration": config, "case": case, "payload": payload, "original_directory": old_dir,
            "original_configuration": old_config, "planned_case_jobs": planned,
            "original_queue_entry": targets[0]}


def verify_archive(directory, config):
    hashes = config["code_sha256"]
    archive_path = inside(directory / "runner_source.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(hashes):
            raise ValueError("Frozen executable archive differs from its manifest")
        for name, expected in hashes.items():
            inside(name, exists=False)
            if archive.getinfo(name).file_size > 4 * 1024**2:
                raise ValueError("Frozen executable member exceeds the bounded reader limit")
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Frozen executable archive member changed: " + name)


def repair_contract(contract, repair):
    config = read_json(repair / "configuration.json")
    original = contract["original_configuration"]
    if {k: v for k, v in config.items() if k != "code_sha256"} != {k: v for k, v in original.items() if k != "code_sha256"}:
        raise ValueError("Repair changed the inputs, methods, candidate space, budget, or execution controls")
    old_hashes, new_hashes = original["code_sha256"], config["code_sha256"]
    if old_hashes.keys() != new_hashes.keys():
        raise ValueError("Repair executable manifest changed its scope")
    changed = sorted(p for p in old_hashes if old_hashes[p] != new_hashes[p])
    if changed != [ALLOWED_REPAIR]:
        raise ValueError("Only the registered human_model_snapshot.py DTO repair is allowed")
    if not config.get("external_scoring_deferred") or config.get("external_observations_available_to_optimizer") is not False:
        raise ValueError("The repair must defer final scoring and exclude final observations from optimization")
    if read_json(repair / "case.json") != contract["payload"]["case"] or sha(repair / "case.json") != config["case_sha256"]:
        raise ValueError("Repair case inputs differ from the original case")
    if read_json(repair / "external_scenario_designs.json") != contract["payload"]["external_scenarios"]:
        raise ValueError("Repair final study designs differ from the registered study designs")
    verify_archive(contract["original_directory"], original)
    verify_archive(repair, config)
    return config


def wait_for_terminal(cohort, repair, output, contract, *, timeout_seconds=MAX_WAIT_SECONDS, poll_seconds=POLL_SECONDS):
    started = time.monotonic()
    last_message = None
    repair_config = None
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError("The bounded eight-hour wait ended before both queues became terminal")
        if repair_config is None and (repair / "configuration.json").is_file():
            # Configuration is directly written early in the child. Wait until
            # its archive and separately written designs exist before checking.
            if all((repair / n).is_file() for n in ("runner_source.zip", "case.json", "external_scenario_designs.json")):
                repair_config = repair_contract(contract, repair)
        cohort_done = read_json(cohort / "completion.json") if (cohort / "completion.json").is_file() else None
        repair_done = read_json(repair / "completion.json") if (repair / "completion.json").is_file() else None
        repair_progress = read_json(repair / "progress.json") if (repair / "progress.json").is_file() else {}
        if cohort_done and (cohort_done.get("status") == "paused" or not cohort_done.get("all_eligible_queues_terminal")):
            raise RuntimeError("The original cohort paused; no repair scoring or replacement was performed")
        if repair_progress.get("status") == "failed" and not repair_done:
            raise RuntimeError("Repair queue failed: " + str(repair_progress.get("error", "see its project log")))
        if repair_done and repair_done.get("status") not in COMPLETE_REPAIR:
            raise RuntimeError("Repair queue did not complete its registered prediction submissions")
        state = {"status": "waiting", "elapsed_seconds": elapsed, "poll_seconds": poll_seconds,
                 "maximum_wait_seconds": timeout_seconds, "cohort_terminal": bool(cohort_done),
                 "repair_terminal": bool(repair_done), "repair_configuration_verified": repair_config is not None,
                 "final_observations_read": False, "solver_calls": 0}
        save(output / "progress.json", state)
        message = (bool(cohort_done), bool(repair_done), repair_progress.get("stage"), repair_config is not None)
        if message != last_message:
            print(json.dumps(state, ensure_ascii=False), flush=True)
            last_message = message
        if cohort_done and repair_done and repair_config is not None:
            queue = read_json(cohort / "queue.json")["cases"]
            if any(r["status"] not in TERMINAL_CASES for r in queue):
                raise ValueError("Original cohort completion conflicts with nonterminal case states")
            if not (cohort / "cohort_prediction_barrier.json").is_file():
                raise ValueError("Original cohort lacks its preserved prediction barrier")
            return repair_config, queue
        time.sleep(min(poll_seconds, max(0, timeout_seconds - elapsed)))


def frozen_submissions(contract, repair, config):
    lock_document = read_json(repair / "external_prediction_locks.json")
    locks = lock_document["locks"]
    expected = _jobs(contract["planned_case_jobs"])
    if _jobs(locks) != expected or lock_document.get("all_method_predictions_finished_before_external_observation_read") is not True:
        raise ValueError("Repair prediction locks omit or add registered method/seed submissions")
    results = read_json(repair / "results.json")
    if _jobs(results["methods"]) != expected:
        raise ValueError("Repair method results omit or add registered submissions")
    methods = {(r["method"], r["seed"]): r for r in results["methods"]}
    scenario_ids = {s["scenario_id"] for s in contract["payload"]["external_scenarios"]}
    fallback = read_json(repair / "external_fallback_predictions.json")
    initial = [p["initial_value"] for p in contract["payload"]["case"]["parameter_search"]["parameters"]]
    if fallback["parameter_values"] != initial or fallback["candidate_id"] != config["base_candidate"]:
        raise ValueError("Fallback is not the registered source-initialized base")
    if set(fallback["curves"]) != scenario_ids:
        raise ValueError("The common fallback must preserve every registered final scenario")
    records = []
    for lock in locks:
        method = methods[(lock["method"], lock["seed"])]
        path = repair / "methods" / ("%s_seed_%d" % (lock["method"], lock["seed"])) / "external_predictions.json"
        record = read_json(path)
        if record["locked_selection"] != lock or (record["method"], record["seed"]) != (lock["method"], lock["seed"]):
            raise ValueError("Repair submission identity differs from the prediction lock")
        if set(record["curves"]) != scenario_ids or record.get("fallback_curves") != fallback["curves"]:
            raise ValueError("Repair submission changed its scenarios or common fallback")
        if lock["candidate_id"] != method["selected_candidate_id"] or record["submission_status"] != method["status"]:
            raise ValueError("Repair prediction is not the method's preselected submission")
        if digest({"predicted_curves": record["curves"], "fallback_curves": record.get("fallback_curves") or {}}) != lock["predictions_sha256"]:
            raise ValueError("Repair primary/fallback predictions changed after locking")
        records.append(record)
    return records, results


def _case_results(queue):
    bundles = {}
    for row in queue:
        item = {"queue_status": row["status"], "queue_error": row.get("reason")}
        if row.get("output"):
            directory = inside(row["output"])
            for key, filename in (("results", "results.json"), ("completion", "completion.json"), ("external_results", "external_results.json")):
                if (directory / filename).is_file():
                    item[key] = read_json(directory / filename)
        bundles[row["case_id"]] = item
    return bundles


def single_runner():
    spec = importlib.util.spec_from_file_location("blueprint_repair_scoring", ROOT / "scripts/run_blueprint_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finalize(cohort, repair, output, case_id, contract, repair_config, queue):
    # All heavy imports are below the terminal-queue and immutable-input checks.
    original_contract(cohort, case_id)
    repair_contract(contract, repair)
    for path, expected in repair_config["code_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Current executable differs from the frozen repair executable: " + path)
    records, method_results = frozen_submissions(contract, repair, repair_config)
    original_cases = _case_results(queue)
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (
        cohort / "configuration.json", cohort / "completion.json", cohort / "queue.json",
        cohort / "cohort_prediction_barrier.json", repair / "configuration.json",
        repair / "completion.json", repair / "results.json", repair / "external_prediction_locks.json")}
    save(output / "repair_prediction_registration.json", {
        "registered_at_utc": datetime.now(UTC).isoformat(), "source_artifact_sha256": input_hashes,
        "locks": [r["locked_selection"] for r in records], "solver_calls": 0,
        "is_original_cohort_global_blinding_barrier": False,
        "repair_predictions_frozen_before_this_scoring": True,
        "other_cohort_drugs_may_have_been_scored_before_repair": True,
        "strict_independent_main_completed": False})
    single = single_runner()
    case, external = single.load_case_bundle(contract["case"]["bundle_path"])
    evaluation_dir = output / "repair_evaluation"
    single.score_external(case, external, records, evaluation_dir, lambda **items: None, device=repair_config["device"])
    from pbpk_kb.blueprints.cohort_results import summarize_cohort_results
    original_summary = summarize_cohort_results(planned_jobs=contract["configuration"]["jobs"], case_results=original_cases)
    repaired_cases = dict(original_cases)
    repaired_cases[case_id] = {"queue_status": "evaluated", "results": method_results,
        "completion": read_json(repair / "completion.json"), "external_results": read_json(evaluation_dir / "external_results.json")}
    summary = summarize_cohort_results(planned_jobs=contract["configuration"]["jobs"], case_results=repaired_cases)
    for path, expected in input_hashes.items():
        if sha(path) != expected:
            raise ValueError("A registered source artifact changed during repair scoring")
    audit = {"schema_version": 1, "case_id": case_id, "compound_id": case["compound_id"], "tier": case["tier"],
        "repair_reason": "Registered engineering repair: route CYP3A4 inhibition DTO selection to Simulation.Interactions rather than Compound.Processes",
        "replacement_rule": "Replace this failed submission only; never choose an attempt by final error",
        "changed_executable": ALLOWED_REPAIR,
        "original_code_sha256": contract["original_configuration"]["code_sha256"][ALLOWED_REPAIR],
        "repair_code_sha256": repair_config["code_sha256"][ALLOWED_REPAIR],
        "input_method_budget_and_candidate_definitions_identical": True, "source_artifact_sha256": input_hashes,
        "original_case_jobs": [j for j in original_summary["jobs"] if j["case_id"] == case_id],
        "repair_case_jobs": [j for j in summary["jobs"] if j["case_id"] == case_id],
        "replacement_adds_independent_drugs_or_studies": False,
        "original_cohort_results_overwritten": False, "repair_predictions_overwritten": False,
        "is_original_cohort_global_blind_evaluation": False,
        "interpretation": "Historical reconstruction after a documented engineering repair; other drugs could already be unblinded. This is not main_holdout.",
        "strict_independent_main_completed": False, "scientific_acceptance_inferred": False, "solver_calls": 0}
    summary.update(repair_case_id=case_id, repair_audit="repair_audit.json", original_cohort_global_blind_evaluation=False,
                   strict_independent_main_completed=False)
    save(output / "original_summary.json", original_summary)
    save(output / "repair_audit.json", audit)
    save(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort-output", "repair-output", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args(argv)
    cohort = inside(args.cohort_output)
    repair = inside(args.repair_output, exists=False)
    output = inside(args.output, exists=False)
    for first, second in ((cohort, repair), (output, cohort), (output, repair)):
        if first == second or first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError("Original cohort, repair queue and finalizer outputs must be disjoint project directories")
    contract = original_contract(cohort, args.case_id)
    output.mkdir(parents=True)
    with (output / "finalizer_source.py").open("xb") as stream:
        stream.write(inside(Path(__file__)).read_bytes())
    save(output / "registration.json", {"started_at_utc": datetime.now(UTC).isoformat(),
        "cohort_output": str(cohort.relative_to(ROOT)), "repair_output": str(repair.relative_to(ROOT)),
        "case_id": args.case_id, "maximum_wait_seconds": MAX_WAIT_SECONDS, "poll_seconds": POLL_SECONDS,
        "allowed_changed_executable": ALLOWED_REPAIR, "original_configuration_sha256": sha(cohort / "configuration.json"),
        "finalizer_script_sha256": sha(Path(__file__)), "solver_calls": 0})
    try:
        config, queue = wait_for_terminal(cohort, repair, output, contract)
        save(output / "progress.json", {"status": "scoring_locked_repair", "solver_calls": 0})
        summary = finalize(cohort, repair, output, args.case_id, contract, config, queue)
        completed = {"status": "completed", "completed_at_utc": datetime.now(UTC).isoformat(),
                     "planned_method_jobs": summary["planned_jobs"], "status_counts": summary["status_counts"],
                     "all_jobs_terminal": summary["all_jobs_terminal"], "solver_calls": 0,
                     "strict_independent_main_completed": False}
        save(output / "completion.json", completed)
        save(output / "progress.json", completed)
        print(json.dumps(completed, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failed = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc), "solver_calls": 0}
        save(output / "completion.json", failed)
        save(output / "progress.json", failed)
        print(json.dumps(failed, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
