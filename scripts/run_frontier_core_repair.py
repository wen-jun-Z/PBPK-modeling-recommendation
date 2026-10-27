"""Isolated source-guard correction and one bounded additional LLM repair.

All v8 generations remain immutable. Replayed controls retain their initial
outputs for admission; real-source labels are never substituted by program code.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import run_frontier_core_scoring as core
import frontier_scope_repair as correction

r, base, ROOT = core.r, core.base, core.ROOT
OUT = core.OUT
PARENT = core.QUAL_ASSESS
OLD_PLAN = core.ASSESS / "core_plan.json"
QUAL = OUT / "qualification_repaired"
ASSESS = OUT / "assessment_repaired"
RUN = ROOT / "runtime/deployment/frontier_core_145_repair"
QUAL_RUN = RUN / "qualification"
SCRIPT = Path(__file__).resolve()
PROFILE = "core145-repair-v1"
FAST, APP = core.FAST, core.APP
_installed = False
_append = base.append_rows
_replay_hashes = {}


def assert_idle():
    parent = base.read(RUN / "pipeline_pid.json", {})
    allow_parent = base.alive(parent) and parent.get("pid") in {os.getpid(), os.getppid()}
    directories = list((ROOT / "runtime/deployment").glob("frontier_*")) + [QUAL_RUN]
    for directory in directories:
        for name in ("pid.json", "pipeline_pid.json"):
            saved = base.read(directory / name, {})
            if not base.alive(saved):
                continue
            if directory == RUN and name == "pipeline_pid.json" and allow_parent and saved == parent:
                continue
            if directory in {RUN, QUAL_RUN} and name == "pid.json" and saved.get("pid") == os.getpid():
                continue
            raise RuntimeError("Another project frontier worker or coordinator is active")


def append_with_explicit_attempts(path, rows):
    """The inherited commit journal lists all actual generations, including #3."""
    if Path(path).parent == ASSESS and Path(path).name in {
            "primary_generation_journal.jsonl", "secondary_generation_journal.jsonl"}:
        rows = copy.deepcopy(rows)
        for row in rows:
            attempts = []
            for attempt in row.get("attempts", []):
                extra = attempt.get("batch_metrics", {}).get("actual_repair_attempts")
                attempts.extend(extra if extra is not None else [attempt])
            if not 1 <= len(attempts) <= 3:
                raise ValueError("Production generation attempt bound exceeded")
            row["attempts"] = attempts
            row["repair_policy"] = PROFILE
    return _append(path, rows)


def configure_scope(folder=QUAL, runtime=QUAL_RUN):
    r.configure()
    base.validate_explicit_scope = correction.guard_factory(core.qualification.GUARD)
    base.OUT, base.ASSESS, base.RUN = OUT, folder, runtime
    os.environ.update(TMPDIR=str(runtime), TORCH_HOME=str(runtime / "torch_cache"),
                      TRITON_CACHE_DIR=str(runtime / "triton_cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(runtime / "torchinductor_cache"))


def initialize():
    global _installed
    if not _installed:
        core.qualification.install()
        r.load_profile("v8")
        r.configure(); r.engine.initialize(); r.configure()
        r.shared.assert_prior_runs_idle = assert_idle
        base.append_rows = append_with_explicit_attempts
        for folder in (QUAL, ASSESS, RUN, QUAL_RUN):
            folder.mkdir(parents=True, exist_ok=True)
        # Reuse only the coordinator's resource, identity and exact-timeout logic.
        core.OUT, core.ASSESS, core.RUN, core.SCRIPT = OUT, ASSESS, RUN, SCRIPT
        core.QUAL_ASSESS, core.QUAL_RUN = QUAL, QUAL_RUN
        core.initialize, core.assert_project_idle = initialize, assert_idle
        core.prepare_plan, core.prepare_core = prepare_plan, prepare_core
        core.configure_core, core.verify_acceptance = configure_core, verify_acceptance
        core.bind_qualification, core.wait_for_review = bind_qualification, wait_for_review
        _installed = True
    configure_scope()


def linked(source, target):
    if not source.exists() or not source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Missing or out-of-project source")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError("Derived source link changed")
    else:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def folder(cohort, parent=False):
    return (PARENT if parent else QUAL) / ("validation/fresh" if cohort == "fresh" else "validation")


def cases_by_cohort():
    return {name: base.read_rows(folder(name) / "cases.jsonl") for name in ("legacy", "fresh")}


def verify_files(files):
    for relative, digest in files.items():
        path = ROOT / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or
                not path.resolve().is_relative_to(ROOT.resolve()) or base.file_sha(path) != digest):
            raise ValueError("Frozen repair artifact changed: " + relative)


def prepare_plan():
    initialize(); assert_idle()
    # Use the original modules only to verify their immutable input contracts.
    r.configure(); base.OUT = r.OUT
    original_queries, candidates, packets, inherited = r.prepare()
    original_cohorts = {"legacy": r.shared.cohorts()[0], "fresh": r.shared.cohorts()[1]}
    for name, cases in original_cohorts.items():
        for model in ("qwen", "phi"):
            records, raw = r.shared.read_pilot(model, name, cases)
            if set(records) != {c["case_id"] for c in cases} or set(raw) != set(records):
                raise ValueError("Complete immutable v8 baseline cohorts are required")
    old_plan = base.read(OLD_PLAN)
    verify_files(old_plan["files"])
    queries = base.read_rows(OUT / "inputs/queries.jsonl")
    selection = base.read(OUT / "selection_manifest.json")
    core.verify_artifact_bindings(selection)
    methods = base.read(OUT / "protocol.json")["methods"]
    if len(methods) != 12 or len(set(methods)) != 12 or set(methods) != set(selection.get("methods", {})):
        raise ValueError("All 12 unchanged ranking slices are required")
    original = {q["query_id"]: q for q in original_queries}
    if (len(queries) != 145 or len({q["query_id"] for q in queries}) != 145 or
            len(candidates) != 280 or candidates != base.read_rows(OUT / "inputs/candidates.jsonl") or
            any(original.get(q["query_id"]) != q for q in queries) or
            [q["query_id"] for q in queries] != selection["selected_query_ids"] or
            {(q["compound_id"], q["task_type"]) for q in queries} !=
            {(q["compound_id"], q["task_type"]) for q in original_queries}):
        raise ValueError("Unchanged 145 requests and all 280 candidates required")
    for target in (QUAL, ASSESS):
        for name in ("sources.jsonl", "source_supplements", "measured_outcomes.json"):
            linked(PARENT / name, target / name)
    for name in ("validation/cases.jsonl", "validation/manifest.json", "validation/fresh/cases.jsonl",
                 "validation/fresh/manifest.json", "source_review_plan.json"):
        linked(PARENT / name, QUAL / name)
    files = [OLD_PLAN, SCRIPT, Path(correction.__file__), PARENT / "judge_protocol.json",
             PARENT / "validation/admission.json", PARENT / "source_review_plan.json"]
    files += [folder(c, True) / name for c in ("legacy", "fresh") for name in
              ("cases.jsonl", "manifest.json", "qwen_protocol.json", "phi_protocol.json",
               "qwen_records.jsonl", "phi_records.jsonl", "qwen_raw_generation_journal.jsonl", "phi_raw_generation_journal.jsonl")]
    plan = {"version": PROFILE, "queries": 145, "candidates": 280,
        "expected_primary_pairs": 40600, "expected_secondary_pairs": old_plan["expected_secondary_pairs"],
        "unique_nonempty_source_requests": old_plan["unique_nonempty_source_requests"],
        "files": {str(p.relative_to(ROOT)): base.file_sha(p) for p in files},
        "parent_core_plan_sha256": base.file_sha(OLD_PLAN), "correction": correction.protocol(),
        "qualification": "Replay all 765 original initial outputs for both judges. Controls retain initial outputs without repairs. Real-source cases use initial, ordinary repair if invalid, focused repair if still invalid. Historical ordinary repair is reusable only when its feedback is identical. Numerical gates and fixed 39-case review remain required.",
        "production": "Same corrected guard and at most 3 actual generations: initial, ordinary repair, focused repair. All source text, weights, reasoning budgets and score definitions unchanged.",
        "scientific_gold_accuracy": None}
    core.freeze(ASSESS / "core_plan.json", plan)
    protocol = copy.deepcopy(inherited)
    protocol.update(version=PROFILE, profile=PROFILE, repair_plan_sha256=base.file_sha(ASSESS / "core_plan.json"),
        correction=correction.protocol(), parent_pilot_outputs_reused=True, full_labels_reused=False,
        raw_generation_reuse="All original initial outputs/attempts are retained with parent hashes; new records are derived replays, not fresh inference.",
        scientific_gold_accuracy=None)
    protocol["generation"].update(max_attempts=3, ordinary_repairs=1, additional_focused_repairs=1,
        max_new_tokens_scope="96 final JSON tokens on initial output, 160 on ordinary and focused repairs; Qwen thinking remains the frozen 1024-token maximum on each actual call",
        synthetic_control_admission="Always score the unchanged initial output; no control repairs")
    protocol["implementation_sha256"].update({SCRIPT.name: base.file_sha(SCRIPT), Path(correction.__file__).name: base.file_sha(correction.__file__)})
    protocol["validation_policy"].update(post_development_replay=True, noncontrol_maximum_attempts=3,
        repair_policy="All real-source and production cases use initial generation, ordinary repair only if invalid, then one focused repair only if still invalid. Historical ordinary output is reused only with identical feedback. Controls remain initial-output-only without repair.",
        interpretation="All 192 synthetic controls are development regression; real-source coverage has no expert gold. A uniform bounded extra repair changes the operational policy, not its score rubric or admission thresholds.")
    core.freeze(QUAL / "judge_protocol.json", protocol)
    for cohort in ("legacy", "fresh"):
        for model in ("qwen", "phi"):
            spec = copy.deepcopy(base.read(folder(cohort, True) / f"{model}_protocol.json"))
            spec.update(judge_protocol_sha256=base.file_sha(QUAL / "judge_protocol.json"), noncontrol_repairs=2,
                        final_invalid_additional_repairs=1, initial_control_outputs_unchanged=True,
                        parent_pilot_protocol_sha256=base.file_sha(folder(cohort, True) / f"{model}_protocol.json"))
            core.freeze(folder(cohort) / f"{model}_protocol.json", spec)
    configure_scope()
    return queries, candidates, packets, protocol


def validation_error(output, packet, queries):
    try:
        parsed = base.parse_answers(output, list(range(1, len(queries)+1)), packet["refs"])
        for number, query in enumerate(queries, 1):
            base.validate_explicit_scope(parsed[number], query, packet["refs"], packet["body"]["model_compounds"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return str(error) if isinstance(error, ValueError) else type(error).__name__
    return None


def rows_index(path):
    rows = base.load_committed(path)
    result = {r["case_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate derived case")
    return result


def parent_binding(model, cohort, old):
    def digest(path):
        return _replay_hashes[path] if path in _replay_hashes else base.file_sha(path)
    return {"parent_records_sha256": digest(folder(cohort, True) / f"{model}_records.jsonl"),
            "parent_raw_journal_sha256": digest(folder(cohort, True) / f"{model}_raw_generation_journal.jsonl"),
            "parent_record_sha256": base.sha(base.canonical(old).encode()),
            "derivation_profile": PROFILE}


def derive_row(model, cohort, old, case, attempts):
    output = attempts[-1]["output"]
    error = validation_error(output, case["packet"], [case["query"]])
    row = copy.deepcopy(old)
    row.update(**parent_binding(model, cohort, old),
               pilot_protocol_sha256=base.file_sha(folder(cohort) / f"{model}_protocol.json"),
               derivation_at=base.now(), final_output=output, final_validation_error=error,
               final_status="valid" if error is None else "invalid_output",
               raw_validation_error=validation_error(old["output"], case["packet"], [case["query"]]),
               additional_focused_repair=len(attempts) == 3,
               parent_attempts=copy.deepcopy(old["attempts"]),
               parent_final_output=old["final_output"], attempts=copy.deepcopy(attempts),
               final_output_changed_from_parent=output != old["final_output"])
    if len(row["attempts"]) > 3:
        raise ValueError("Qualification generation attempt bound exceeded")
    return row


def repair_request(model, cohort, case, prior, error, kind):
    feedback = error[:240] if kind == "ordinary" else correction.focused_feedback(error, case["query"])
    return {"case_id": case["case_id"], "model": model, "cohort": cohort, "repair_kind": kind,
        "prior_output_sha256": base.sha(prior.encode()),
        "pilot_protocol_sha256": base.file_sha(folder(cohort) / f"{model}_protocol.json"),
        "feedback": feedback}


def adopted_attempts(model, cohort, case, old, journals):
    """Simulate the same ordered 1/2/3-attempt policy used in production."""
    initial = {**old["attempts"][0], "generation_origin": "v8_initial", "parent_attempt_index": 0}
    attempts = [initial]
    error = validation_error(initial["output"], case["packet"], [case["query"]])
    if error is None or case["kind"] == "synthetic_control":
        return attempts, None
    for kind in ("ordinary", "focused"):
        prior = attempts[-1]["output"]
        request = repair_request(model, cohort, case, prior, error, kind)
        if kind == "ordinary" and len(old["attempts"]) >= 2 and old.get("raw_validation_error") == request["feedback"]:
            attempt = {**old["attempts"][1], "generation_origin": "v8_ordinary_identical_feedback",
                       "parent_attempt_index": 1, "feedback": request["feedback"]}
        else:
            generated = journals[kind].get(case["case_id"])
            if generated is None:
                return attempts, {**request, "prior_output": prior}
            if any(generated.get(k) != v for k,v in request.items()):
                raise ValueError("Repair journal did not use the exact ordered-policy feedback")
            attempt = {"output": generated["output"], "batch_metrics": generated["metrics"],
                       "generation_origin": "new_" + kind, "repair_kind": kind, "feedback": request["feedback"]}
        attempts.append(attempt)
        error = validation_error(attempt["output"], case["packet"], [case["query"]])
        if error is None:
            break
    return attempts, None


def verify_derived(model, cohort, case, old, row, attempts):
    expected = derive_row(model, cohort, old, case, attempts)
    if any(row.get(k) != v for k,v in expected.items() if k != "derivation_at"):
        raise ValueError("Derived record diverged from immutable parents or actual ordered repair journals")


def replay(model):
    global _replay_hashes
    # Hash each immutable baseline file once per full replay, not once per row.
    _replay_hashes = {folder(c, True) / name: base.file_sha(folder(c, True) / name)
                      for c in ("legacy", "fresh") for name in
                      (f"{model}_records.jsonl", f"{model}_raw_generation_journal.jsonl")}
    pending = []
    for cohort, cases in cases_by_cohort().items():
        parent = rows_index(folder(cohort, True) / f"{model}_records.jsonl")
        raw_parent = rows_index(folder(cohort, True) / f"{model}_raw_generation_journal.jsonl")
        path = folder(cohort) / f"{model}_records.jsonl"
        done = rows_index(path)
        raw_path = folder(cohort) / f"{model}_raw_generation_journal.jsonl"
        raw_done = rows_index(raw_path)
        ids = {c["case_id"] for c in cases}
        if not set(done) <= ids or not set(raw_done) <= ids:
            raise ValueError("Foreign derived qualification case")
        journals = {k: rows_index(folder(cohort) / f"{model}_{k}_repair_journal.jsonl") for k in ("ordinary", "focused")}
        if any(not set(j) <= ids for j in journals.values()):
            raise ValueError("Foreign additional repair case")
        for case in cases:
            key = case["case_id"]; old = parent[key]
            raw = {**raw_parent[key], **parent_binding(model, cohort, old),
                   "pilot_protocol_sha256": base.file_sha(folder(cohort) / f"{model}_protocol.json"),
                   "generation_origin": "immutable_v8_initial_output_reused"}
            if key in raw_done:
                if raw_done[key] != raw:
                    raise ValueError("Original initial output was altered in derived raw journal")
            else:
                base.append_rows(raw_path, [raw])
            attempts, request = adopted_attempts(model, cohort, case, old, journals)
            if request is not None:
                if key in done:
                    raise ValueError("Completed record lacks the required repair generation")
                pending.append((cohort, case, old, request)); continue
            if key in done:
                verify_derived(model, cohort, case, old, done[key], attempts); continue
            base.append_rows(path, [derive_row(model, cohort, old, case, attempts)])
    return pending


def pilot(model):
    prepare_plan(); assert_idle()
    lock = (QUAL_RUN / "runner.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(QUAL_RUN / "pid.json", base.identity())
    state = {**base.identity(), "state": "replaying", "model": model, "expected": 765}
    def update(**values):
        r.utilities.status_update(QUAL_RUN / f"pilot_{model}_status.json", state, **values)
    update(); pending = replay(model)
    update(state="replaying", completed=765-len(pending), pending_additional_repairs=len(pending))
    judge = None
    while pending:
        cohort, case, old, request = pending[0]
        if (RUN / "stop.requested").exists() or (QUAL_RUN / "stop.requested").exists():
            update(state="stopped"); return
        failure = core.resource_failure()
        if failure:
            raise RuntimeError(failure)
        journal_path = folder(cohort) / f"{model}_{request['repair_kind']}_repair_journal.jsonl"
        if sum(p.stat().st_size for p in QUAL.rglob("*.jsonl") if not p.is_symlink()) > 4*1024**3:
            raise RuntimeError("Qualification output budget exceeded 4 GiB")
        if judge is None:
            update(state="verifying_weights")
            weights = r.utilities.verify_weights(r.QWEN_WEIGHTS if model == "qwen" else r.PHI_WEIGHTS)
            configure_scope()
            judge = r.backend(model)(weights, 36, lambda: update(state="loading_gpu"))
        feedback = request["feedback"]
        update(state="judging", current_case=case["case_id"], repair_kind=request["repair_kind"])
        outputs, metrics = judge.generate([(case["packet"], [case["query"]])], max_new_tokens=160,
            repairs=[{"output": request["prior_output"], "error": feedback}])
        if len(outputs) != 1:
            raise ValueError("Additional qualification repair cardinality mismatch")
        extra = {k:v for k,v in request.items() if k != "prior_output"}
        extra.update(output=outputs[0], metrics=metrics, at=base.now())
        base.append_rows(journal_path, [extra])
        pending = replay(model)
        update(completed=765-len(pending), pending_additional_repairs=len(pending), last_commit_at=base.now())
    del judge
    if replay(model):
        raise ValueError("Qualification replay left uncommitted cases")
    update(state="complete", completed=765, pending_additional_repairs=0)


def verify_source_review():
    review = base.read(QUAL / "source_review.json", {})
    expected = {"version": "pbpk-reviewed-source-review-v1", "profile": PROFILE,
        "protocol_sha256": base.file_sha(QUAL / "judge_protocol.json"),
        "validation_manifest_sha256": base.file_sha(QUAL / "validation/manifest.json"),
        "fresh_manifest_sha256": base.file_sha(QUAL / "validation/fresh/manifest.json"),
        "pilot_record_sha256": {m: base.file_sha(folder("legacy") / f"{m}_records.jsonl") for m in ("phi", "qwen")},
        "fresh_pilot_record_sha256": {m: base.file_sha(folder("fresh") / f"{m}_records.jsonl") for m in ("phi", "qwen")}}
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError("Source review must bind actual complete revised outputs")
    cases = cases_by_cohort()["legacy"]
    required = set(base.read(QUAL / "source_review_plan.json")["case_ids"]) | {
        c["case_id"] for c in cases if c["strata"]["validation_role"] == "production_failure_diagnostic"}
    checked = review.get("checked_case_ids", [])
    if (len(checked) != len(set(checked)) or not required <= set(checked) or
            not set(checked) <= {c["case_id"] for c in cases} or
            review.get("reviewer") != "Codex evidence audit; not expert gold" or
            review.get("scientific_gold_accuracy", "absent") is not None or
            not isinstance(review.get("findings"), list) or not isinstance(review.get("known_unknown_distribution"), dict) or
            type(review.get("accepted")) is not bool or review.get("primary") != "qwen" or review.get("secondary") != "phi"):
        raise ValueError("Source review is incomplete or lacks non-gold disclosure")
    return review["accepted"], base.file_sha(QUAL / "source_review.json")


def check_gate(require_review=True):
    prepare_plan()
    from frontier_judge_validation_v3 import evaluate_validation_v3
    cases = cases_by_cohort(); reports, fresh = {}, {}
    for model in ("qwen", "phi"):
        if replay(model):
            raise ValueError("Complete all qualification repairs before admission")
        rows = {cohort: list(rows_index(folder(cohort) / f"{model}_records.jsonl").values()) for cohort in cases}
        if any(len(rows[k]) != len(cases[k]) for k in cases):
            raise ValueError("Exact complete qualification cohorts required")
        reports[model] = evaluate_validation_v3(rows["legacy"], cases["legacy"])
        fresh[model] = r.shared.control_metric(rows["fresh"], cases["fresh"])
        base.dump(folder("legacy") / f"{model}_evaluation.json", reports[model])
        base.dump(folder("fresh") / f"{model}_evaluation.json", r.shared.controls.evaluate_validation(rows["fresh"], cases["fresh"]))
    result = r.engine.admission_from_reports({"phi": reports["qwen"], "qwen": reports["phi"]})
    result["control_groups"] = {k: {"qwen": v["phi"], "phi": v["qwen"], "accepted": v["accepted"]} for k,v in result["control_groups"].items()}
    result.update(at=base.now(), profile=PROFILE, judge_protocol_sha256=base.file_sha(QUAL / "judge_protocol.json"),
        primary="Qwen/Qwen3.6-27B", secondary="microsoft/phi-4",
        fresh_control_group={**fresh, "accepted": r.utilities.passes_gate(fresh["qwen"], fresh["phi"])})
    result["numeric_accepted"] = bool(result["accepted"] and result["fresh_control_group"]["accepted"])
    result["accepted"] = False
    if require_review:
        accepted, digest = verify_source_review()
        result.update(accepted=bool(result["numeric_accepted"] and accepted), source_review_sha256=digest)
    base.dump(QUAL / "validation/admission.json", result)
    return result


def bind_qualification():
    result = check_gate()
    if not result["accepted"]:
        raise RuntimeError("Unchanged numerical admission and actual source review must pass")
    files = [QUAL / "judge_protocol.json", QUAL / "source_review.json", ASSESS / "core_plan.json"]
    files += [p for p in (QUAL / "validation").rglob("*") if p.is_file() and p.name in {
        "cases.jsonl", "manifest.json", "qwen_records.jsonl", "phi_records.jsonl", "qwen_raw_generation_journal.jsonl",
        "phi_raw_generation_journal.jsonl", "qwen_focused_repair_journal.jsonl", "phi_focused_repair_journal.jsonl",
        "qwen_ordinary_repair_journal.jsonl", "phi_ordinary_repair_journal.jsonl"}]
    value = {"version": PROFILE, "accepted": True, "numeric_accepted": True, "cases_per_model": 765,
        "source_review_sha256": result["source_review_sha256"], "scientific_gold_accuracy": None,
        "files": {str(p.relative_to(ROOT)): base.file_sha(p) for p in files}}
    return core.freeze(ASSESS / "qualification_acceptance.json", value)


def verify_acceptance():
    result = base.read(ASSESS / "qualification_acceptance.json", {})
    if (result.get("accepted") is not True or result.get("numeric_accepted") is not True or
            result.get("cases_per_model") != 765 or result.get("scientific_gold_accuracy", "absent") is not None):
        raise RuntimeError("Qualified source-reviewed correction is required")
    verify_files(result["files"])
    return result


def configure_core():
    configure_scope(ASSESS, RUN)
    base.prepare = prepare_core


def prepare_core():
    queries, candidates, packets, inherited = prepare_plan()
    accepted = verify_acceptance()
    protocol = copy.deepcopy(inherited)
    protocol.update(version="pbpk-core145-repaired-production-v1", evaluation_cohort="145 frozen core queries against 280 candidates",
        qualification_acceptance_sha256=base.file_sha(ASSESS / "qualification_acceptance.json"),
        qualification_source_review_sha256=accepted["source_review_sha256"],
        qualification_judge_protocol_sha256=base.file_sha(QUAL / "judge_protocol.json"),
        selected_query_manifest_sha256=base.file_sha(OUT / "selection_manifest.json"),
        input_sha256={name: base.file_sha(OUT / name) for name in ("inputs/queries.jsonl", "inputs/candidates.jsonl",
            "assessment_repaired/sources.jsonl", "assessment_repaired/source_supplements/sources.jsonl")})
    # The evaluator recognizes source keys by the assessment convention.
    protocol["input_sha256"].update({"assessment/sources.jsonl": base.file_sha(ASSESS / "sources.jsonl"),
        "assessment/source_supplements/sources.jsonl": base.file_sha(ASSESS / "source_supplements/sources.jsonl")})
    core.freeze(ASSESS / "judge_protocol.json", protocol)
    core.freeze(ASSESS / "assessment_manifest.json", {"queries": 145, "candidates": 280, "expected_pairs": 40600,
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"), "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl"),
        "distinct_material_packets": len({p["packet_sha256"] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values())})
    configure_core()
    return queries, candidates, packets, protocol


def repaired_backend(parent):
    class RepairedJudge(parent):
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            outputs, metrics = super().generate(tasks, max_new_tokens=max_new_tokens, repairs=repairs)
            if repairs is None:
                return outputs, metrics
            if len(tasks) != 1 or len(outputs) != 1:
                raise ValueError("Ordinary production repair must be one prompt")
            packet, queries = tasks[0]
            if len(queries) != 1:
                raise ValueError("Frozen focused repair requires one query per prompt")
            error = validation_error(outputs[0], packet, queries)
            if error is None:
                return outputs, metrics
            feedback = correction.focused_feedback(error, queries[0])
            ordinary = {"output": outputs[0], "batch_metrics": metrics, "repair_kind": "ordinary"}
            extra, extra_metrics = super().generate(tasks, max_new_tokens=160,
                repairs=[{"output": outputs[0], "error": feedback}])
            if len(extra) != 1:
                raise ValueError("Focused production repair cardinality mismatch")
            attempts = [ordinary, {"output": extra[0], "batch_metrics": extra_metrics,
                                    "repair_kind": "additional_focused", "feedback": feedback}]
            base.append_rows(ASSESS / "focused_repair_journal.jsonl", [{"at": base.now(),
                "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
                "packet_sha256": packet["packet_sha256"], "query_ids": [q["query_id"] for q in queries],
                "initial_output": repairs[0]["output"], "attempts": attempts}])
            return extra, {**extra_metrics, "actual_repair_attempts": attempts,
                "wall_seconds": float(metrics.get("wall_seconds", 0)) + float(extra_metrics.get("wall_seconds", 0)),
                "actual_repair_generation_count": 2}
    return RepairedJudge


def run_full(judge):
    prepare_core(); assert_idle()
    base.GPUJudge = repaired_backend(r.backend("qwen" if judge == "primary" else "phi"))
    base.download_primary = lambda: r.utilities.verify_weights(r.QWEN_WEIGHTS)
    base.run_assessment(SimpleNamespace(judge=judge, memory_gib=36, batch_size=16,
        max_batches=0, secondary_weights=r.PHI_WEIGHTS))


def wait_for_review(update):
    if not check_gate(False)["numeric_accepted"]:
        update(state="stopped_on_validation", stage="admission_failed"); return False
    update(state="waiting_for_source_review", stage="source_review", worker_pid=None)
    while not (QUAL / "source_review.json").exists():
        if (RUN / "stop.requested").exists():
            update(state="stopped"); return False
        failure = core.resource_failure()
        if failure:
            update(state="stopped_on_guard", guard_reason=failure); return False
        time.sleep(20)
    if not check_gate()["accepted"]:
        update(state="stopped_on_source_review"); return False
    bind_qualification(); prepare_core(); return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "replay", "start", "pipeline", "pilot", "run", "gate", "finalize", "status", "stop"))
    parser.add_argument("--model", choices=("qwen", "phi"), default="qwen")
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    args = parser.parse_args(); initialize()
    if args.action == "stop":
        (RUN / "stop.requested").touch(); print("Repaired core scoring stop requested; all records retained"); return
    if args.action == "status":
        print(json.dumps({"alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {}),
            "qualification_qwen": base.read(QUAL_RUN / "pilot_qwen_status.json", {}),
            "qualification_phi": base.read(QUAL_RUN / "pilot_phi_status.json", {})})); return
    if args.action in {"pilot", "run"}:
        expected = FAST if (args.action == "pilot" and args.model == "qwen") or (args.action == "run" and args.judge == "primary") else APP
        if Path(sys.prefix).resolve() != expected.parent.parent.resolve():
            raise RuntimeError("Wrong interpreter for actual judge")
    try:
        if args.action == "prepare":
            prepare_plan(); print("Immutable revision prepared; no GPU generation started")
        elif args.action == "replay":
            prepare_plan(); print(json.dumps({m: len(replay(m)) for m in ("qwen", "phi")}))
        elif args.action == "start": print(json.dumps(core.start()))
        elif args.action == "pipeline": core.pipeline()
        elif args.action == "pilot": pilot(args.model)
        elif args.action == "run": run_full(args.judge)
        elif args.action == "gate": print(json.dumps(check_gate(False)))
        elif args.action == "finalize": bind_qualification(); prepare_core(); print("Source-reviewed production protocol frozen")
    except BaseException as error:
        if args.action in {"pilot", "run"}:
            path = QUAL_RUN / f"pilot_{args.model}_status.json" if args.action == "pilot" else RUN / "status.json"
            checkpoint = args.action == "run" and isinstance(error, TimeoutError) and str(error) == core.SEVEN_DAY_MESSAGE
            r.utilities.status_update(path, base.read(path, {}), state="checkpoint_time_limit" if checkpoint else "failed",
                error_type=type(error).__name__, error_message=str(error)[:300])
        raise


if __name__ == "__main__":
    main()
