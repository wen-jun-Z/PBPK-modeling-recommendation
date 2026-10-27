"""Isolated grade-first, conditional-citation fallback; no old predictions reused."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import run_frontier_scoring_v3 as engine
from frontier_judge_prompt_v2 import SYSTEM

base, utilities = engine.base, engine.utilities
ROOT, OUT = engine.ROOT, engine.OUT
PARENT_ASSESS = OUT / "assessment_evidence_first_v3"
ASSESS = OUT / "assessment_conditional_v4"
RUN = ROOT / "runtime/deployment/frontier_scoring_conditional_v4"
APP, MODERN = engine.APP, engine.MODERN
PHI_WEIGHTS, QWEN_WEIGHTS = engine.PHI_WEIGHTS, engine.QWEN_WEIGHTS
GATE, CONTROL_ROLES = engine.GATE, engine.CONTROL_ROLES
VERSION = "pbpk-grade-first-conditional-citation-scoring-v4"
# Root freezes the prospective amendment before either pilot is generated.
AMENDMENT_SHA = "2211ef2775b71a3a5c0e0f9d10d32a79734118a09d8bd83f7d226822765eb73f"
CODE_FILES = (*engine.CODE_FILES, "frontier_conditional_citation_decoding.py", Path(__file__).name)
REGRESSION_DISCLOSURE = "All 733 cases are reused regression/source coverage checks, not new blind or scientific gold validation"
INJECTED = ("ASSESS", "RUN", "SYSTEM", "prepare", "backend", "command", "verify_source_review", "publish_summary")


def configure():
    # Explicit dependency injection in this new process only. Frozen v3 source,
    # prompt, decoder and results remain unchanged. The engine's pilot/repair,
    # gates, guard, monitor and full evaluation implementations are reused.
    engine.ASSESS, engine.RUN, engine.SYSTEM = ASSESS, RUN, SYSTEM
    engine.prepare, engine.backend, engine.command = prepare, backend, command
    engine.verify_source_review, engine.publish_summary = verify_source_review, publish_summary
    engine.configure()


def initialize():
    configure()
    engine.initialize()


def assert_prior_runs_idle():
    names = ("frontier_scoring_v3", "frontier_scoring_selected_v3", "frontier_assessment",
             *(v[1] for v in utilities.PROFILE_DIRECTORIES.values()))
    for name in names:
        for pidfile in ("pid.json", "pipeline_pid.json"):
            if base.alive(base.read(ROOT / "runtime/deployment" / name / pidfile, {})):
                raise RuntimeError("A prior project scoring worker/pipeline is active; conditional fallback cannot overlap it")


def verify_amendment():
    path = ASSESS / "amendment.json"
    if not path.exists() or base.file_sha(path) != AMENDMENT_SHA:
        raise ValueError("Frozen prospective v4 amendment missing or changed")
    value = base.read(path)
    checks = {"version": "pbpk-conditional-citation-v4-amendment-v1", "primary": "phi", "secondary": "qwen",
        "thresholds": GATE, "coverage_minimum": .98, "coverage_window": 128, "maximum_window_invalid_fraction": .10,
        "pilot_outputs_reused": False, "full_labels_reused": False, "validation_interpretation": REGRESSION_DISCLOSURE,
        "prompt_file": "scripts/frontier_judge_prompt_v2.py", "decoder_file": "scripts/frontier_conditional_citation_decoding.py"}
    if any(value.get(k) != expected for k, expected in checks.items()):
        raise ValueError("V4 amendment does not preserve the frozen qualification and no-reuse policy")
    artifacts = {PARENT_ASSESS / "judge_protocol.json": value["parent_protocol_sha256"],
        PARENT_ASSESS / "validation/manifest.json": value["validation_manifest_sha256"],
        PARENT_ASSESS / "source_review_plan.json": value["source_review_plan_sha256"],
        ROOT / checks["prompt_file"]: value["prompt_file_sha256"],
        ROOT / checks["decoder_file"]: value["decoder_file_sha256"]}
    if any(base.file_sha(path) != digest for path, digest in artifacts.items()):
        raise ValueError("V4 parent materials, original prompt or conditional decoder changed")
    if value["system_prompt_sha256"] != base.sha(SYSTEM.encode()):
        raise ValueError("V4 must use the exact original v2 prompt")
    return value


def read_finished_rows(path):
    """Read committed newline-terminated rows; disclose, never repair, a tail."""
    data = path.read_bytes()
    boundary = data.rfind(b"\n") + 1
    rows = [json.loads(line) for line in data[:boundary].splitlines() if line.strip()]
    return rows, len(data)-boundary


def fallback_activation():
    """Require observed absolute failures, never infer failure from missing data.

    A complete failed control group makes a judge ineligible regardless of its
    unfinished source cases. This bounded activation rule deliberately does not
    infer eligibility from missing reviews or combine partial-group metrics.
    """
    amendment = verify_amendment()
    for folder in (PARENT_ASSESS, OUT / "assessment_selected_v3"):
        if base.read(folder / "validation/admission.json", {}).get("accepted") is True:
            raise RuntimeError("A v3 candidate is already accepted; conditional fallback is forbidden")
    from frontier_judge_validation_v3 import evaluate_validation_v3
    cases = base.read_rows(PARENT_ASSESS / "validation/cases.jsonl")
    manifest = base.read(PARENT_ASSESS / "validation/manifest.json")
    if (manifest.get("case_count") != 733 or len(cases) != 733 or
            manifest.get("cases_sha256") != base.file_sha(PARENT_ASSESS / "validation/cases.jsonl")):
        raise ValueError("Activation must use the unchanged complete 733-case definition")
    case_by_id = {c["case_id"]: c for c in cases}
    if len(case_by_id) != len(cases): raise ValueError("Duplicate frozen activation case")
    minimums = {"controlled_semantic_accuracy": GATE["controlled_semantic_minimum"],
        "controlled_citation_support_accuracy": GATE["controlled_citation_support_minimum"],
        "citation_numbers_valid_fraction": GATE["citation_numbers_valid_minimum"],
        "raw_format_valid_fraction": GATE["raw_format_valid_minimum"]}
    result = {"policy": "Each model has at least one complete control group below an unchanged absolute gate",
        "missing_data_is_failure": False, "relative_gate_used": False,
        "parent_protocol_sha256": amendment["parent_protocol_sha256"],
        "validation_manifest_sha256": amendment["validation_manifest_sha256"], "models": {}}
    for model, weight_path in (("phi", PHI_WEIGHTS), ("qwen", QWEN_WEIGHTS)):
        folder = PARENT_ASSESS / "validation"
        protocol_path, records_path, raw_path = (folder / f"{model}_{suffix}" for suffix in
            ("protocol.json", "records.jsonl", "raw_generation_journal.jsonl"))
        if not all(p.is_file() for p in (protocol_path, records_path, raw_path)):
            result["models"][model] = {"eligible_ruled_out": False, "reason": "Observed complete control failure not available"}
            continue
        weights = base.read(weight_path)
        expected = {"model_repo": weights["repo"], "model_revision": weights["revision"],
            "judge_protocol_sha256": amendment["parent_protocol_sha256"],
            "validation_sha256": amendment["validation_manifest_sha256"],
            "maximum_batch_size": 16, "memory_gib": 36, "raw_max_tokens": 96, "repair_max_tokens": 160,
            "control_repairs": 0, "noncontrol_repairs": 1, "raw_journal_before_repairs": True}
        if base.read(protocol_path) != expected:
            raise ValueError("Activation pilot protocol differs from frozen v3 generation")
        records, record_tail = read_finished_rows(records_path)
        raw_rows, raw_tail = read_finished_rows(raw_path)
        for rows in (records, raw_rows):
            if len({r["case_id"] for r in rows}) != len(rows) or not {r["case_id"] for r in rows} <= set(case_by_id):
                raise ValueError("Duplicate or unknown activation pilot record")
            for row in rows:
                if (row.get("model_repo") != weights["repo"] or row.get("model_revision") != weights["revision"] or
                        row.get("pilot_protocol_sha256") != base.file_sha(protocol_path) or
                        row.get("validation_sha256") != amendment["validation_manifest_sha256"] or
                        row.get("kind") != case_by_id[row["case_id"]]["kind"]):
                    raise ValueError("Activation pilot record provenance mismatch")
        raw = {r["case_id"]: r for r in raw_rows}
        for row in records:
            attempts = row.get("attempts", [])
            if (row["case_id"] not in raw or row["output"] != raw[row["case_id"]]["output"] or
                    not 1 <= len(attempts) <= (1 if row["kind"] == "synthetic_control" else 2) or
                    attempts[0]["output"] != row["output"] or attempts[-1]["output"] != row["final_output"]):
                raise ValueError("Activation raw/final attempts disagree")
        report = evaluate_validation_v3(records, cases)
        failures = {}
        for role in CONTROL_ROLES:
            metric = report["admission_by_role"][role]
            if metric["complete"]:
                failed = [key for key, minimum in minimums.items() if metric[key] < minimum]
                if failed: failures[role] = {"metrics": metric, "failed_absolute_checks": failed}
        result["models"][model] = {"eligible_ruled_out": bool(failures), "observed_cases": len(records),
            "complete_all_733": report["complete"], "failed_control_groups": failures,
            "pilot_protocol_sha256": base.file_sha(protocol_path), "records_sha256": base.file_sha(records_path),
            "raw_journal_sha256": base.file_sha(raw_path), "uncommitted_tail_bytes": {"final": record_tail, "raw": raw_tail},
            "model_repo": weights["repo"], "model_revision": weights["revision"]}
    result["accepted"] = all(result["models"][m]["eligible_ruled_out"] for m in ("phi", "qwen"))
    if not result["accepted"]:
        raise RuntimeError("A v3 candidate remains potentially eligible; complete absolute control failures for BOTH models are required")
    return result


def link_validation_inputs():
    directory = ASSESS / "validation"
    if directory.is_symlink():
        raise ValueError("V4 validation must be an independent directory")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("validation/cases.jsonl", "validation/manifest.json", "source_review_plan.json"):
        source, target = PARENT_ASSESS / name, ASSESS / name
        if not source.is_file() or not source.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Missing scoped frozen validation input")
        if target.is_symlink() or target.exists():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError("Validation input is not the exact preserved parent file")
        else: target.symlink_to(os.path.relpath(source, target.parent))
    outputs = ["source_review.json", "validation/admission.json", "judgments.jsonl", "secondary_judgments.jsonl"]
    outputs += [f"validation/{model}_{suffix}" for model in ("phi", "qwen") for suffix in
                ("records.jsonl", "raw_generation_journal.jsonl", "protocol.json", "evaluation.json")]
    outputs += [f"{judge}_{suffix}.jsonl" for judge in ("primary", "secondary")
                for suffix in ("judgment_cache", "generation_journal")]
    if any((ASSESS / name).is_symlink() for name in outputs):
        raise ValueError("V4 must generate its own pilot/full outputs, never append to linked earlier results")


def prepare():
    configure()
    assert_prior_runs_idle()
    amendment = verify_amendment()
    activation = fallback_activation()
    old = base.read(PARENT_ASSESS / "judge_protocol.json")
    for name, digest in old["implementation_sha256"].items():
        if base.file_sha(ROOT / "scripts" / name) != digest:
            raise ValueError("Frozen parent pipeline dependency changed")
    if (old["primary_weights_manifest_sha256"] != base.file_sha(PHI_WEIGHTS) or
            old["secondary_weights_manifest_sha256"] != base.file_sha(QWEN_WEIGHTS)):
        raise ValueError("V4 must use the same frozen prepared Phi and Qwen weights")
    link_validation_inputs()
    queries, candidates, packets, _ = utilities.prepare_materials()
    from frontier_judge_validation_v3 import verify_validation_v3
    manifest = verify_validation_v3(queries, candidates, packets, ASSESS / "validation")
    # The existing validator verifies provenance of the reused cases, including
    # their original v3 creation prompt. The actual NEW generation prompt below
    # is the separately hashed v2 text; old predictions are never accepted.
    engine.verify_review_plan(base.read_rows(ASSESS / "validation/cases.jsonl"))
    from frontier_conditional_citation_decoding import decoding_protocol
    phi, qwen = base.read(PHI_WEIGHTS), base.read(QWEN_WEIGHTS)
    protocol = copy.deepcopy(old)
    protocol.update(version=VERSION, primary_repo=phi["repo"], primary_revision=phi["revision"],
        secondary_repo=qwen["repo"], secondary_revision=qwen["revision"],
        system_prompt=SYSTEM, system_prompt_sha256=base.sha(SYSTEM.encode()),
        primary_mode="Phi-4 instruction model; grade-first conditional-citation JSON",
        secondary_mode="Qwen3.6-27B enable_thinking=False; grade-first conditional-citation JSON; sensitivity only",
        parent_protocol_sha256=amendment["parent_protocol_sha256"], amendment_sha256=AMENDMENT_SHA,
        activation_evidence=activation,
        amendment_timing=amendment["selection_timing"], parent_pilot_outputs_reused=False, full_labels_reused=False,
        primary_weights_manifest_sha256=base.file_sha(PHI_WEIGHTS),
        secondary_weights_manifest_sha256=base.file_sha(QWEN_WEIGHTS),
        execution_environments=utilities.execution_environments(),
        validation_manifest_sha256=amendment["validation_manifest_sha256"],
        source_review_plan_sha256=amendment["source_review_plan_sha256"],
        implementation_sha256={name: base.file_sha(ROOT / "scripts" / name) for name in CODE_FILES})
    protocol["generation"]["constrained_decoding"] = decoding_protocol()
    protocol["validation_policy"].update(case_count=manifest["case_count"], interpretation=REGRESSION_DISCLOSURE,
        existing_role_names_retained_for_denominators=True, independent_blind_validation=False,
        actual_pilots="Fresh raw/final outputs from both models under v4 only; zero earlier predictions reused")
    protocol["judge_development"]["v4_validation"] = REGRESSION_DISCLOSURE
    utilities.freeze(ASSESS / "judge_protocol.json", protocol)
    utilities.freeze(ASSESS / "assessment_manifest.json", {
        "queries": len(queries), "candidates": len(candidates), "expected_pairs": len(queries)*len(candidates),
        "distinct_material_packets": len({p["packet_sha256"] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values()),
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl")})
    return queries, candidates, packets, protocol


def backend(model):
    from frontier_conditional_citation_decoding import conditional_citation_constraint
    original = utilities.PhiJudge if model == "phi" else utilities.backend("qwen")
    class ConditionalJudge(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.json_constraint = conditional_citation_constraint(self.json_constraint)
    return ConditionalJudge


def verify_source_review():
    review = base.read(ASSESS / "source_review.json")
    if not review: raise ValueError("V4 source review has not been recorded")
    expected = {"version": "pbpk-v4-source-review-v1", "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "validation_manifest_sha256": base.file_sha(ASSESS / "validation/manifest.json"),
        "pilot_record_sha256": {model: base.file_sha(ASSESS / "validation" / f"{model}_records.jsonl") for model in ("phi", "qwen")}}
    if any(review.get(k) != value for k, value in expected.items()):
        raise ValueError("V4 source review is not bound to these new pilot outputs")
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    plan = engine.verify_review_plan(cases)
    checked = review.get("checked_case_ids", [])
    required = set(plan["case_ids"]) | {c["case_id"] for c in cases if c["strata"]["validation_role"] == "production_failure_diagnostic"}
    if (not checked or len(set(checked)) != len(checked) or not required <= set(checked) or
            not set(checked) <= {c["case_id"] for c in cases} or
            review.get("reviewer") != "Codex evidence audit; not expert gold" or
            "scientific_gold_accuracy" not in review or review["scientific_gold_accuracy"] is not None or
            not isinstance(review.get("findings"), list) or not isinstance(review.get("known_unknown_distribution"), dict) or
            type(review.get("accepted")) is not bool):
        raise ValueError("V4 source review lacks planned coverage or non-gold disclosure")
    return review["accepted"], base.file_sha(ASSESS / "source_review.json")


def command(python, *args):
    return [str(python), str(Path(__file__).resolve()), *args]


def publish_summary():
    folder = ROOT / "experiments/model_recommendation/results_summary"
    link = folder / "08_conditional_citation_assessment"
    if not link.exists(): link.symlink_to(os.path.relpath(ASSESS, folder), target_is_directory=True)
    path = folder / "README.md"
    text = path.read_text()
    first, last = "<!-- conditional-v4:start -->", "<!-- conditional-v4:end -->"
    state, worker = base.read(RUN / "pipeline_status.json", {}), base.read(RUN / "status.json", {})
    judge = worker.get("judge")
    progress = (f"{'Primary assessment' if judge == 'primary' else '5% repeat judgment'} — committed: {worker.get('committed_pairs', 0):,}/"
                f"{worker.get('expected_pairs', 216160):,}. " if judge in {"primary", "secondary"} else "Full assessment has not started.")
    block = (f"{first}\n## Conditional-citation assessment v4\n\nUpdated: {base.now()}; Stage: `{state.get('stage', 'prepare')}`; "
        f"Status: `{state.get('state', 'preparing')}`. {progress}\n\n"
        "The v2 instructions and grade-before-citation order are retained. Known judgments require model-selected real citations; unknowns may have empty citations. "
        "Phi-4 and Qwen3.6-27B each rerun 733 prechecks without reusing old predictions. These are regression and source-coverage reviews, not new blinded tests. "
        "Phi primary assessment must pass the original three-group thresholds, representative-source checks, and 39-case review. Qwen is only the fixed 5% sensitivity judge. "
        "Unknowns are not negative labels and original quality-stop rules remain. Valid citation syntax does not establish semantic support or expert gold standards.\n\n"
        "[Status and results](08_conditional_citation_assessment/); Command: `runtime/app-env/bin/python scripts/run_frontier_scoring_conditional_v4.py "
        f"status` / `stop` / `start`. Continues after SSH disconnects.\n{last}")
    if first in text:
        if last not in text: raise ValueError("Incomplete README markers")
        text = text[:text.index(first)] + block + text[text.index(last)+len(last):]
    else: text += "\n" + block + "\n"
    path.write_text(text)


def execute(action, model="phi", judge="primary"):
    configure()
    if action in {"start", "pipeline", "pilot", "run"}:
        assert_prior_runs_idle()
        fallback_activation()
    if action == "start": return engine.start()
    if action == "pipeline": return engine.pipeline()
    if action == "pilot": return engine.pilot(model)
    if action == "run": return engine.run_full(judge)
    if action == "prepare": return prepare()
    raise ValueError("Unknown v4 execution action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "pipeline", "pilot", "run", "status", "stop"))
    parser.add_argument("--model", choices=("phi", "qwen"), default="phi")
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    args = parser.parse_args(); initialize()
    if args.action == "status":
        print(json.dumps({"alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {})}))
    elif args.action == "stop": (RUN / "stop.requested").touch(); print("Stop requested; outputs retained")
    else:
        try:
            result = execute(args.action, args.model, args.judge)
            if args.action == "start": print(json.dumps(result))
            elif args.action == "prepare": print("V4 protocol verified and frozen; new pilot admission is still required")
        except BaseException as error:
            if args.action in {"pilot", "run"}:
                path = RUN / (f"pilot_{args.model}_status.json" if args.action == "pilot" else "status.json")
                utilities.status_update(path, base.read(path, {}), state="failed", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    main()
