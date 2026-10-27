"""Frozen profile driven scoring with separate regression and fresh controls."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import run_frontier_scoring_v3 as engine
import frontier_judge_validation as controls

base, utilities = engine.base, engine.utilities
ROOT, OUT = engine.ROOT, engine.OUT
LEGACY = OUT / "assessment_evidence_first_v3"
APP, MODERN = engine.APP, engine.MODERN
PHI_WEIGHTS, QWEN_WEIGHTS = engine.PHI_WEIGHTS, engine.QWEN_WEIGHTS
GATE, CONTROL_ROLES = engine.GATE, engine.CONTROL_ROLES
ENGINE_CHECK_GATE = engine.check_gate
PROFILE, CONFIG, CONFIG_SHA, SYSTEM = None, {}, None, None
ASSESS, RUN = None, None
INJECTED = ("ASSESS", "RUN", "SYSTEM", "prepare", "backend", "command", "check_gate", "verify_source_review", "publish_summary")
DISCLOSURE = "The prior 733 cases are regression/source coverage; only the separately frozen fresh32 are new controls; neither is scientific gold"
CONDITIONAL_DECODER_SHA = "9fb009e5f6caa27a74d6f89830a5da684f58cc8346781ecfbd6fd903dd262494"
LEGACY_PROTOCOL_SHA = "56042879f5280f9db1d50e446d34cb4c90aa0d8158263a4a7580adf0dd8882d9"


def script_path(relative):
    path = ROOT / relative
    if not re.fullmatch(r"scripts/[A-Za-z0-9_]+\.py", relative) or not path.resolve().is_relative_to((ROOT / "scripts").resolve()):
        raise ValueError("Profile Python dependencies must be scoped project scripts")
    return path


def import_frozen(relative, digest, module_name):
    path = script_path(relative)
    if base.file_sha(path) != digest: raise ValueError("Frozen profile dependency changed")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_profile(slug):
    global PROFILE, CONFIG, CONFIG_SHA, ASSESS, RUN, SYSTEM
    if not re.fullmatch(r"v[0-9]{1,3}", slug) or int(slug[1:]) < 5:
        raise ValueError("Profile must be an approved version slug v5 or later, never a path")
    directory = ROOT / "runtime/deployment/judge_profiles"
    path, checksum = directory / f"{slug}.json", directory / f"{slug}.sha256"
    if any(not p.resolve().is_relative_to(ROOT.resolve()) for p in (path, checksum)):
        raise ValueError("Profile config outside project")
    digest = checksum.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or base.file_sha(path) != digest:
        raise ValueError("Profile config is not the independently frozen file")
    value = base.read(path)
    if (value.get("version") != "pbpk-reviewed-judge-profile-v1" or value.get("profile") != slug or
            value.get("runtime_directory") != f"frontier_scoring_reviewed_{slug}" or
            not re.fullmatch(r"assessment_[a-z0-9_]+_" + slug, value.get("assessment_directory", "")) or
            not re.fullmatch(r"assessment_[a-z0-9_]+", value.get("predecessor_assessment_directory", "")) or
            value.get("thresholds") != GATE or value.get("representative_minimum") != .98 or
            value.get("representative_window") != 128 or value.get("representative_max_invalid_fraction") != .10 or
            value.get("regression_disclosure") != DISCLOSURE):
        raise ValueError("Profile identity, scope or unchanged gate configuration mismatch")
    prompt = import_frozen(value["prompt_file"], value["prompt_file_sha256"], f"reviewed_prompt_{slug}")
    if base.sha(prompt.SYSTEM.encode()) != value["system_prompt_sha256"]:
        raise ValueError("Profile prompt content differs from its frozen hash")
    PROFILE, CONFIG, CONFIG_SHA, SYSTEM = slug, value, digest, prompt.SYSTEM
    ASSESS, RUN = OUT / value["assessment_directory"], ROOT / "runtime/deployment" / value["runtime_directory"]
    if not ASSESS.resolve().is_relative_to(OUT.resolve()) or ASSESS.is_symlink() or RUN.is_symlink():
        raise ValueError("Profile output must be an independent project directory")
    return value


def configure():
    if PROFILE is None: raise ValueError("Load a frozen profile before execution")
    engine.ASSESS, engine.RUN, engine.SYSTEM = ASSESS, RUN, SYSTEM
    engine.prepare, engine.backend, engine.command = prepare, backend, command
    engine.check_gate, engine.verify_source_review, engine.publish_summary = check_gate, verify_source_review, publish_summary
    engine.configure()


def assert_prior_runs_idle():
    for directory in (ROOT / "runtime/deployment").glob("frontier_*"):
        if directory == RUN or not directory.is_dir(): continue
        for name in ("pid.json", "pipeline_pid.json"):
            if base.alive(base.read(directory / name, {})):
                raise RuntimeError("A prior project frontier worker/pipeline is active")


def ensure_not_stopped():
    engine.ensure_not_quality_stopped()
    state = base.read(RUN / "pilot_phi_status.json", {})
    if state.get("guard_reason", "").startswith("AbsoluteControlAdmissionFailed"):
        raise RuntimeError("This frozen primary control gate failed; an identical restart is forbidden")


def link_inputs():
    directory = ASSESS / "validation"
    if directory.is_symlink(): raise ValueError("Validation directory must be independent")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("validation/cases.jsonl", "validation/manifest.json", "source_review_plan.json"):
        source, target = LEGACY / name, ASSESS / name
        if not source.resolve().is_relative_to(ROOT.resolve()) or not source.is_file():
            raise ValueError("Missing frozen project input")
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError("Unexpected legacy input target")
        else: target.symlink_to(os.path.relpath(source, target.parent))
    fresh = directory / "fresh"
    if fresh.is_symlink(): raise ValueError("Fresh validation directory must be independent")
    for folder in (ASSESS, directory, fresh):
        if folder.exists():
            for path in folder.iterdir():
                if path.is_symlink() and (path.name.endswith(("records.jsonl", "generation_journal.jsonl", "evaluation.json", "protocol.json")) or
                        path.name in {"admission.json", "judgments.jsonl", "secondary_judgments.jsonl", "source_review.json"} or
                        "judgment_cache" in path.name):
                    raise ValueError("Generated profile outputs must never link to prior predictions")


def prepare():
    configure(); assert_prior_runs_idle()
    current = base.file_sha(ROOT / "runtime/deployment/judge_profiles" / f"{PROFILE}.json")
    if current != CONFIG_SHA: raise ValueError("Frozen config changed after loading")
    if (base.file_sha(script_path(CONFIG["prompt_file"])) != CONFIG["prompt_file_sha256"] or
            base.sha(SYSTEM.encode()) != CONFIG["system_prompt_sha256"] or
            base.file_sha(ROOT / "scripts/frontier_conditional_citation_decoding.py") != CONDITIONAL_DECODER_SHA):
        raise ValueError("Frozen prompt or unchanged conditional decoder changed")
    link_inputs()
    if (base.file_sha(ASSESS / "validation/manifest.json") != CONFIG["legacy_manifest_sha256"] or
            base.file_sha(ASSESS / "source_review_plan.json") != CONFIG["source_review_plan_sha256"] or
            base.file_sha(ASSESS / "validation/fresh/manifest.json") != CONFIG["fresh_manifest_sha256"] or
            base.file_sha(ASSESS / "validation/fresh/cases.jsonl") != CONFIG["fresh_cases_sha256"]):
        raise ValueError("Frozen regression/fresh validation inputs changed")
    old = base.read(LEGACY / "judge_protocol.json")
    if base.file_sha(LEGACY / "judge_protocol.json") != LEGACY_PROTOCOL_SHA:
        raise ValueError("Frozen legacy protocol changed")
    for name, digest in old["implementation_sha256"].items():
        if base.file_sha(ROOT / "scripts" / name) != digest: raise ValueError("Frozen engine dependency changed")
    from frontier_judge_validation_v3 import verify_validation_v3
    queries, candidates, packets, _ = utilities.prepare_materials()
    verify_validation_v3(queries, candidates, packets, ASSESS / "validation")
    fresh_module = import_frozen(CONFIG["fresh_validator_file"], CONFIG["fresh_validator_sha256"], f"reviewed_fresh_{PROFILE}")
    fresh_module.verify_fresh(ASSESS / "validation/fresh")
    legacy, fresh = cohorts()
    if (len(legacy) != 733 or len(fresh) != 32 or {c["case_id"] for c in legacy} & {c["case_id"] for c in fresh} or
            any(c["kind"] != "synthetic_control" for c in fresh)):
        raise ValueError("Exact disjoint 733-regression and 32-fresh cohorts required")
    engine.verify_review_plan(legacy)
    if (base.file_sha(PHI_WEIGHTS) != old["primary_weights_manifest_sha256"] or
            base.file_sha(QWEN_WEIGHTS) != old["secondary_weights_manifest_sha256"]):
        raise ValueError("Prepared model identities changed")
    predecessor = OUT / CONFIG["predecessor_assessment_directory"]
    if not predecessor.resolve().is_relative_to(OUT.resolve()): raise ValueError("Predecessor outside project results")
    if (base.file_sha(predecessor / "judge_protocol.json") != CONFIG["predecessor_protocol_sha256"] or
            base.file_sha(predecessor / "validation/phi_records.jsonl") != CONFIG["predecessor_primary_records_sha256"]):
        raise ValueError("Disclosed preceding primary run changed")
    from frontier_conditional_citation_decoding import decoding_protocol
    protocol = copy.deepcopy(old)
    implementation = {**old["implementation_sha256"], Path(__file__).name: base.file_sha(Path(__file__)),
        "frontier_conditional_citation_decoding.py": base.file_sha(ROOT / "scripts/frontier_conditional_citation_decoding.py"),
        Path(CONFIG["prompt_file"]).name: CONFIG["prompt_file_sha256"],
        Path(CONFIG["fresh_validator_file"]).name: CONFIG["fresh_validator_sha256"]}
    protocol.update(version="pbpk-reviewed-profile-scoring-v1", profile=PROFILE, profile_config=CONFIG,
        profile_config_sha256=CONFIG_SHA, system_prompt=SYSTEM, system_prompt_sha256=CONFIG["system_prompt_sha256"],
        primary_mode="Phi-4 instruction model; reviewed decision precedence; grade-first conditional citations",
        secondary_mode="Qwen3.6-27B enable_thinking=False; same reviewed prompt/decoder; fixed 5% sensitivity",
        implementation_sha256=implementation, execution_environments=utilities.execution_environments(),
        predecessor_protocol_sha256=CONFIG["predecessor_protocol_sha256"],
        parent_pilot_outputs_reused=False, full_labels_reused=False,
        fresh_validation_manifest_sha256=CONFIG["fresh_manifest_sha256"],
        pilot_execution_order=[*CONTROL_ROLES, "fresh32", "legacy_real_sources"],
        primary_early_stop="Each complete control group below any unchanged absolute gate stops before the next group or model")
    protocol["generation"]["constrained_decoding"] = decoding_protocol()
    protocol["validation_policy"].update(interpretation=DISCLOSURE, fresh_case_count=32,
        raw_control_groups_separate=True, legacy_case_count=733, real_source_case_selection_unchanged=True)
    protocol["judge_development"]["reviewed_profile"] = {"profile": PROFILE, "regression_case_count": 733,
        "new_control_count": 32, "fresh_definition": "Frozen after prompt and before this profile's first generation; synthetic contract checks, not scientific gold"}
    utilities.freeze(ASSESS / "judge_protocol.json", protocol)
    utilities.freeze(ASSESS / "assessment_manifest.json", {"queries": len(queries), "candidates": len(candidates),
        "expected_pairs": len(queries)*len(candidates), "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl"),
        "distinct_material_packets": len({p["packet_sha256"] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values())})
    return queries, candidates, packets, protocol


def backend(model):
    from frontier_conditional_citation_decoding import conditional_citation_constraint
    parent = utilities.PhiJudge if model == "phi" else utilities.backend("qwen")
    class ReviewedJudge(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.json_constraint = conditional_citation_constraint(self.json_constraint)
    return ReviewedJudge


def cohorts():
    return base.read_rows(ASSESS / "validation/cases.jsonl"), base.read_rows(ASSESS / "validation/fresh/cases.jsonl")


def cohort_folder(cohort):
    if cohort not in {"legacy", "fresh"}: raise ValueError("Unknown pilot cohort")
    return ASSESS / ("validation" if cohort == "legacy" else "validation/fresh")


def pilot_spec(model, cohort):
    weights = base.read(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS)
    return {"model_repo": weights["repo"], "model_revision": weights["revision"], "cohort": cohort,
        "judge_protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "validation_sha256": base.file_sha(cohort_folder(cohort) / "manifest.json"),
        "maximum_batch_size": 16, "memory_gib": 36, "raw_max_tokens": 96, "repair_max_tokens": 160,
        "control_repairs": 0, "noncontrol_repairs": 1, "raw_journal_before_repairs": True}


def read_pilot(model, cohort, cases, *, allow_repair=False):
    folder = cohort_folder(cohort)
    spec = pilot_spec(model, cohort)
    path = folder / f"{model}_protocol.json"
    if base.read(path) != spec: raise ValueError("Pilot belongs to a different frozen profile/cohort")
    raw = base.load_committed(folder / f"{model}_raw_generation_journal.jsonl", repair_tail=allow_repair)
    final = base.load_committed(folder / f"{model}_records.jsonl", repair_tail=allow_repair)
    expected = {c["case_id"]: c for c in cases}
    for rows in (raw, final):
        if len({r["case_id"] for r in rows}) != len(rows) or not {r["case_id"] for r in rows} <= set(expected):
            raise ValueError("Duplicate or foreign pilot case")
        if any(r.get("model_repo") != spec["model_repo"] or r.get("model_revision") != spec["model_revision"] or
               r.get("validation_sha256") != spec["validation_sha256"] or r.get("pilot_protocol_sha256") != base.file_sha(path) or
               r.get("kind") != expected[r["case_id"]]["kind"] for r in rows):
            raise ValueError("Raw/final pilot provenance mismatch")
    index = {r["case_id"]: r for r in raw}
    for row in final:
        attempts = row.get("attempts", [])
        if (row["case_id"] not in index or index[row["case_id"]]["output"] != row["output"] or
                not 1 <= len(attempts) <= (1 if row["kind"] == "synthetic_control" else 2) or
                attempts[0]["output"] != row["output"] or attempts[-1]["output"] != row["final_output"]):
            raise ValueError("Pilot attempts disagree with the original output")
    return {r["case_id"]: r for r in final}, index


def control_metric(records, cases):
    keep = {c["case_id"] for c in cases}
    return controls.admission_metrics(controls.evaluate_validation([r for r in records if r["case_id"] in keep], cases))


def absolute_failures(metric):
    if not metric["complete"]: return []
    limits = {"controlled_semantic_accuracy": GATE["controlled_semantic_minimum"],
        "controlled_citation_support_accuracy": GATE["controlled_citation_support_minimum"],
        "raw_format_valid_fraction": GATE["raw_format_valid_minimum"],
        "citation_numbers_valid_fraction": GATE["citation_numbers_valid_minimum"]}
    return [key for key, minimum in limits.items() if metric[key] < minimum]


def pilot(model, job="all"):
    configure(); ensure_not_stopped(); assert_prior_runs_idle()
    lock = (RUN / "runner.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pid.json", base.identity()); prepare()
    legacy, fresh = cohorts()
    all_cases = {"legacy": legacy, "fresh": fresh}
    units = [("legacy", role, [c for c in legacy if c["strata"]["validation_role"] == role]) for role in CONTROL_ROLES]
    units += [("fresh", "fresh32", fresh), ("legacy", "legacy_real_sources", [c for c in legacy if c["kind"] != "synthetic_control"])]
    if job != "all": units = [u for u in units if (job == "fresh" and u[0] == "fresh") or
        (job == "legacy-controls" and u[1] in CONTROL_ROLES) or (job == "legacy-sources" and u[1] == "legacy_real_sources")]
    state = {**base.identity(), "state": "verifying_weights", "model": model, "job": job, "expected": 765}
    def update(**values): utilities.status_update(RUN / f"pilot_{model}_status.json", state, **values)
    update()
    completed, generated = {}, {}
    for cohort, cases in all_cases.items():
        utilities.freeze(cohort_folder(cohort) / f"{model}_protocol.json", pilot_spec(model, cohort))
        completed[cohort], generated[cohort] = read_pilot(model, cohort, cases, allow_repair=True)
    weights, judge = None, None
    def ensure_judge():
        nonlocal weights, judge
        if judge is None:
            weights = utilities.verify_weights(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS)
            judge = backend(model)(weights, 36, lambda: update(state="loading_gpu"))
        return judge
    def check_control(cohort, role, cases):
        if role == "legacy_real_sources": return False
        metric = control_metric(list(completed[cohort].values()), cases)
        failed = absolute_failures(metric)
        base.dump(cohort_folder(cohort) / f"{model}_{role}_gate.json", {"metrics": metric,
            "absolute_failed_checks": failed, "primary": model == "phi", "relative_comparison_pending": True})
        if model == "phi" and failed:
            update(state="stopped_on_control_gate", guard_reason=f"AbsoluteControlAdmissionFailed:{cohort}:{role}",
                   failed_control_group=role, failed_checks=failed, control_metrics=metric,
                   completed=sum(map(len, completed.values())), legacy_completed=len(completed["legacy"]), fresh_completed=len(completed["fresh"]))
            return True
        return False
    control_units = [("legacy", role, [c for c in legacy if c["strata"]["validation_role"] == role]) for role in CONTROL_ROLES]
    control_units.append(("fresh", "fresh32", fresh))
    for cohort, role, cases in control_units:
        if check_control(cohort, role, cases): return
    if model == "phi" and job == "legacy-sources" and any(
            not control_metric(list(completed[cohort].values()), cases)["complete"] for cohort, _, cases in control_units):
        raise RuntimeError("Complete all four primary control groups before a source-only job")
    for cohort, role, cases in units:
        if check_control(cohort, role, cases): return
        pending = [c for c in cases if c["case_id"] not in completed[cohort]]
        folder = cohort_folder(cohort)
        position, maximum = 0, 16
        while position < len(pending):
            if (RUN / "stop.requested").exists(): update(state="stopped"); return
            batch = pending[position:position+maximum]
            known = generated[cohort]
            if batch[0]["case_id"] not in known:
                batch = batch[:next((i for i, c in enumerate(batch) if c["case_id"] in known), len(batch))]
                tasks = [(c["packet"], [c["query"]]) for c in batch]
                active = ensure_judge()
                size = active.select_batch_size(tasks, maximum)
                batch, tasks = batch[:size], tasks[:size]
                try: outputs, metrics = active.generate(tasks)
                except active.torch.cuda.OutOfMemoryError:
                    active.torch.cuda.empty_cache()
                    if size == 1: raise
                    maximum = max(1, size//2); continue
                if len(outputs) != len(batch): raise ValueError("Pilot generation cardinality mismatch")
                spec = pilot_spec(model, cohort)
                rows = [{"case_id": c["case_id"], "kind": c["kind"], "output": output, "cohort": cohort,
                    "model_repo": spec["model_repo"], "model_revision": spec["model_revision"],
                    "validation_sha256": spec["validation_sha256"], "pilot_protocol_sha256": base.file_sha(folder / f"{model}_protocol.json"),
                    "at": base.now(), "metrics": metrics} for c, output in zip(batch, outputs)]
                base.append_rows(folder / f"{model}_raw_generation_journal.jsonl", rows)
                known.update({r["case_id"]: r for r in rows})
            else:
                batch = batch[:next((i for i, c in enumerate(batch) if c["case_id"] not in known), len(batch))]
            for case in batch:
                raw = known[case["case_id"]]
                output, attempts = raw["output"], [{"output": raw["output"], "batch_metrics": raw["metrics"]}]
                error = engine.validation_error(output, case)
                if error is not None and case["kind"] != "synthetic_control":
                    outputs, metrics = ensure_judge().generate([(case["packet"], [case["query"]])], max_new_tokens=160,
                        repairs=[{"output": output, "error": error}])
                    if len(outputs) != 1: raise ValueError("Repair cardinality mismatch")
                    output = outputs[0]; attempts.append({"output": output, "batch_metrics": metrics})
                final_error = engine.validation_error(output, case)
                row = {**raw, "attempts": attempts, "final_output": output, "raw_validation_error": error,
                    "final_validation_error": final_error, "final_status": "valid" if final_error is None else "invalid_output"}
                base.append_rows(folder / f"{model}_records.jsonl", [row]); completed[cohort][case["case_id"]] = row
                update(state="judging", stage=role, completed=sum(map(len, completed.values())),
                    legacy_completed=len(completed["legacy"]), fresh_completed=len(completed["fresh"]), last_commit_at=base.now())
            position += len(batch)
        if check_control(cohort, role, cases): return
    del judge
    from frontier_judge_validation_v3 import evaluate_validation_v3
    base.dump(ASSESS / "validation" / f"{model}_evaluation.json", evaluate_validation_v3(list(completed["legacy"].values()), legacy))
    base.dump(ASSESS / "validation/fresh" / f"{model}_evaluation.json", controls.evaluate_validation(list(completed["fresh"].values()), fresh))
    complete = len(completed["legacy"]) == len(legacy) and len(completed["fresh"]) == len(fresh)
    update(state="complete" if complete else "job_complete", completed=sum(map(len, completed.values())),
           legacy_completed=len(completed["legacy"]), fresh_completed=len(completed["fresh"]))


def check_gate(require_review=True):
    configure(); ensure_not_stopped(); prepare()
    legacy, cases = cohorts()
    fresh = {}
    for model in ("phi", "qwen"):
        old_records, old_raw = read_pilot(model, "legacy", legacy)
        if set(old_records) != {c["case_id"] for c in legacy} or set(old_raw) != set(old_records):
            raise ValueError("Regression pilot must be the exact complete 733 cases for each model")
        records, raw = read_pilot(model, "fresh", cases)
        if set(records) != {c["case_id"] for c in cases} or set(raw) != set(records):
            raise ValueError("Fresh pilot must be the exact complete 32 cases for each model")
        fresh[model] = control_metric(list(records.values()), cases)
    result = ENGINE_CHECK_GATE(require_review=False)
    result["fresh_control_group"] = {**fresh, "accepted": utilities.passes_gate(fresh["phi"], fresh["qwen"])}
    result["numeric_accepted"] = bool(result["numeric_accepted"] and result["fresh_control_group"]["accepted"])
    result.update(profile=PROFILE, profile_config_sha256=CONFIG_SHA, accepted=False,
                  fresh_validation_manifest_sha256=CONFIG["fresh_manifest_sha256"])
    if require_review:
        accepted, digest = verify_source_review()
        result.update(accepted=bool(result["numeric_accepted"] and accepted), source_review_sha256=digest)
    base.dump(ASSESS / "validation/admission.json", result)
    return result


def verify_source_review():
    review = base.read(ASSESS / "source_review.json")
    expected = {"version": "pbpk-reviewed-source-review-v1", "profile": PROFILE,
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "validation_manifest_sha256": CONFIG["legacy_manifest_sha256"], "fresh_manifest_sha256": CONFIG["fresh_manifest_sha256"],
        "pilot_record_sha256": {m: base.file_sha(ASSESS / "validation" / f"{m}_records.jsonl") for m in ("phi", "qwen")},
        "fresh_pilot_record_sha256": {m: base.file_sha(ASSESS / "validation/fresh" / f"{m}_records.jsonl") for m in ("phi", "qwen")}}
    if not review or any(review.get(k) != value for k, value in expected.items()):
        raise ValueError("Source review must bind this profile and both independent pilot cohorts")
    legacy, _ = cohorts(); plan = engine.verify_review_plan(legacy)
    checked = review.get("checked_case_ids", [])
    required = set(plan["case_ids"]) | {c["case_id"] for c in legacy if c["strata"]["validation_role"] == "production_failure_diagnostic"}
    if (not checked or len(set(checked)) != len(checked) or not required <= set(checked) or
            not set(checked) <= {c["case_id"] for c in legacy} or
            review.get("reviewer") != "Codex evidence audit; not expert gold" or
            "scientific_gold_accuracy" not in review or review["scientific_gold_accuracy"] is not None or
            not isinstance(review.get("findings"), list) or not isinstance(review.get("known_unknown_distribution"), dict) or
            type(review.get("accepted")) is not bool):
        raise ValueError("Source review lacks planned coverage and non-gold disclosure")
    return review["accepted"], base.file_sha(ASSESS / "source_review.json")


def command(python, *args):
    return [str(python), str(Path(__file__).resolve()), "--profile", PROFILE, *args]


def publish_summary():
    folder = ROOT / "experiments/model_recommendation/results_summary"
    link = folder / f"09_decision_order_assessment_{PROFILE}"
    if not link.exists(): link.symlink_to(os.path.relpath(ASSESS, folder), target_is_directory=True)
    path = folder / "README.md"; text = path.read_text()
    first, last = f"<!-- reviewed-{PROFILE}:start -->", f"<!-- reviewed-{PROFILE}:end -->"
    state = base.read(RUN / "pipeline_status.json", {})
    worker = base.read(RUN / "status.json", {})
    progress = (f"{'Primary assessment' if worker.get('judge') == 'primary' else '5% repeat judgment'} — committed: {worker.get('committed_pairs', 0):,}/"
                f"{worker.get('expected_pairs', 216160):,}. " if worker.get("judge") in {"primary", "secondary"} else "Full assessment has not started.")
    block = (f"{first}\n## Decision-order assessment {PROFILE}\n\nUpdated: {base.now()}; Stage: `{state.get('stage', 'prepare')}`; "
        f"Status: `{state.get('state', 'preparing')}`. {progress}\n\n"
        "Phi and Qwen regenerate all judgments. The existing 733 cases assess regression and source coverage; 32 new controls are reported separately. "
        "Stop immediately if Phi fails any control group. Full assessment requires source coverage and the 39-case review to pass. "
        "The model selects grades and citations. Unknowns are not negative labels; valid citation syntax is not an expert gold standard.\n\n"
        f"[Status and results]({link.name}/); Command: `runtime/app-env/bin/python scripts/run_frontier_scoring_reviewed.py --profile {PROFILE} status` / `stop` / `start`. \n{last}")
    if first in text:
        if last not in text: raise ValueError("Incomplete summary markers")
        text = text[:text.index(first)] + block + text[text.index(last)+len(last):]
    else: text += "\n" + block + "\n"
    path.write_text(text)


def execute(action, model="phi", judge="primary", job="all"):
    configure()
    if action in {"start", "pipeline", "pilot", "run"}:
        ensure_not_stopped(); assert_prior_runs_idle()
    if action == "pilot": return pilot(model, job)
    if action == "run": return engine.run_full(judge)
    if action == "pipeline": return engine.pipeline()
    if action == "start": prepare(); return engine.start()
    if action == "prepare": return prepare()
    raise ValueError("Unknown profile action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("action", choices=("prepare", "start", "pipeline", "pilot", "run", "status", "stop"))
    parser.add_argument("--model", choices=("phi", "qwen"), default="phi")
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    parser.add_argument("--job", choices=("all", "legacy-controls", "fresh", "legacy-sources"), default="all")
    args = parser.parse_args(); load_profile(args.profile); configure(); engine.initialize()
    if args.action == "status":
        print(json.dumps({"profile": PROFILE, "alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {}),
            "primary_validation": base.read(RUN / "pilot_phi_status.json", {})}))
    elif args.action == "stop": (RUN / "stop.requested").touch(); print("Stop requested; outputs retained")
    else:
        try:
            result = execute(args.action, args.model, args.judge, args.job)
            if args.action == "start": print(json.dumps(result))
            elif args.action == "prepare": print("Profile protocol frozen; new pilot admission still required")
        except BaseException as error:
            if args.action in {"pilot", "run"}:
                path = RUN / (f"pilot_{args.model}_status.json" if args.action == "pilot" else "status.json")
                utilities.status_update(path, base.read(path, {}), state="failed", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    main()
