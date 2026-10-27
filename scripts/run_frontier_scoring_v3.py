"""Isolated evidence-first validation and full scoring; old runs remain frozen."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import run_frontier_assessment as base
import run_frontier_scoring as utilities
from frontier_judge_prompt_v3 import SYSTEM

ROOT, OUT = base.ROOT, base.OUT
ASSESS = OUT / "assessment_evidence_first_v3"
RUN = ROOT / "runtime/deployment/frontier_scoring_v3"
PREVIOUS = OUT / "assessment_validated_v2"
APP, MODERN = utilities.APP, utilities.MODERN
PHI_WEIGHTS, QWEN_WEIGHTS = utilities.PHI_WEIGHTS, utilities.QWEN_WEIGHTS
GATE = copy.deepcopy(utilities.GATE)
CONTROL_ROLES = ("development_v1_regression", "development_v2_regression", "independent_admission")
VERSION = "pbpk-evidence-first-full-scoring-v3"
AMENDMENT_SHA = "424abe52c5de50acb11e37ef431516a3c271e055257827124a0a63b2af9a19f5"
REVIEW_PLAN_SHA = "96c481fbddeba139ef621b78da6c4fbc1c7882c13a9c0d31c4a6c5530aef3b0f"
CODE_FILES = ("run_frontier_scoring_v3.py", "run_frontier_scoring.py", "run_frontier_assessment.py",
              "frontier_modern_judge.py", "frontier_judge_prompt_v3.py", "frontier_judge_validation_v3.py",
              "frontier_evidence_first_decoding.py", "frontier_judge_decoding.py", "evaluate_frontier_automatic.py",
              "frontier_judge_prompt_v2.py", "frontier_judge_validation.py", "frontier_judge_validation_v2.py")


def configure():
    base.SYSTEM, base.ASSESS, base.RUN = SYSTEM, ASSESS, RUN
    utilities.ASSESS = ASSESS


def initialize():
    configure()
    RUN.mkdir(parents=True, exist_ok=True)
    ASSESS.mkdir(parents=True, exist_ok=True)
    safe = base.environment()
    safe.update(TMPDIR=str(RUN), TORCH_HOME=str(RUN / "torch_cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TRITON_CACHE_DIR=str(RUN / "triton_cache"), TORCHINDUCTOR_CACHE_DIR=str(RUN / "torchinductor_cache"))
    os.environ.clear()
    os.environ.update(safe)


def ensure_not_quality_stopped():
    status = base.read(RUN / "status.json", {})
    pipeline = base.read(RUN / "pipeline_status.json", {})
    if status.get("guard_reason") == "PrimaryInvalidFractionAbove10Percent" or pipeline.get("state") in {"stopped_on_validation", "stopped_on_source_review"}:
        raise RuntimeError("This frozen v3 failed a quality gate; a new reviewed amendment and isolated protocol are required, not an identical restart")


def verify_amendment():
    path = ASSESS / "amendment.json"
    if not path.exists() or base.file_sha(path) != AMENDMENT_SHA:
        raise ValueError("Frozen v3 amendment missing or changed")
    amendment = base.read(path)
    for name, digest in amendment["predecessor_sha256"].items():
        if name not in {"judge_protocol.json", "selection_policy.json", "judgments.jsonl", "primary_judgment_cache.jsonl", "primary_generation_journal.jsonl"}:
            raise ValueError("Unexpected predecessor artifact")
        if base.file_sha(PREVIOUS / name) != digest:
            raise ValueError("Preserved failed run changed")
    if (base.sha(SYSTEM.encode()) != amendment["repair"]["system_sha256"] or
            base.file_sha(ROOT / amendment["repair"]["prompt_file"]) != amendment["repair"]["prompt_file_sha256"]):
        raise ValueError("Amended prompt changed")
    return amendment


def prepare():
    configure()
    verify_amendment()
    from frontier_judge_validation_v3 import verify_validation_v3
    from frontier_evidence_first_decoding import decoding_protocol
    queries, candidates, packets, old = utilities.prepare_materials()
    manifest = verify_validation_v3(queries, candidates, packets, ASSESS / "validation")
    review_plan = verify_review_plan(base.read_rows(ASSESS / "validation/cases.jsonl"))
    phi, qwen = base.read(PHI_WEIGHTS), base.read(QWEN_WEIGHTS)
    protocol = copy.deepcopy(old)
    protocol.update(version=VERSION, primary_repo=phi["repo"], primary_revision=phi["revision"],
        secondary_repo=qwen["repo"], secondary_revision=qwen["revision"],
        system_prompt=SYSTEM, system_prompt_sha256=base.sha(SYSTEM.encode()),
        primary_mode="Phi-4 instruction model; evidence-first constrained JSON",
        secondary_mode="Qwen3.6-27B enable_thinking=False; evidence-first constrained JSON; sensitivity only",
        predecessor_protocol_sha256=base.file_sha(PREVIOUS / "judge_protocol.json"),
        amendment_sha256=AMENDMENT_SHA, predecessor_labels_reused=False, pilot_labels_reused=False,
        source_review_plan_sha256=REVIEW_PLAN_SHA,
        primary_weights_manifest_sha256=base.file_sha(PHI_WEIGHTS),
        secondary_weights_manifest_sha256=base.file_sha(QWEN_WEIGHTS),
        execution_environments=utilities.execution_environments(),
        hardware="Sequential GPU workers; 36 GiB allocator cap; aggregate memory guard; batch maximum 16; hardware exclusivity not claimed",
        validation_manifest_sha256=base.file_sha(ASSESS / "validation/manifest.json"),
        admission_checks=GATE,
        validation_policy={"control_roles": list(CONTROL_ROLES), "raw_control_outputs_only": True,
            "representative_role": "prospective_source_coverage", "representative_final_valid_minimum": .98,
            "representative_window": 128, "representative_max_invalid_fraction": .10,
            "repair_policy": "All noncontrol cases get at most one identical production repair after parser/scope failure; controls remain unrepaired",
            "real_source_correctness_gold": False, "case_count": manifest["case_count"]},
        implementation_sha256={name: base.file_sha(ROOT / "scripts" / name) for name in CODE_FILES})
    protocol["generation"]["constrained_decoding"] = decoding_protocol()
    actual_query_ids = {q["query_id"] for q in queries}
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    protocol["judge_development"]["generated_query_ids"] = sorted(set(
        protocol["judge_development"].get("generated_query_ids", [])) |
        {c["query"]["query_id"] for c in cases if c["query"]["query_id"] in actual_query_ids})
    protocol["judge_development"]["v3_validation"] = "validation/manifest.json; development controls and source diagnostics disclosed separately from fresh controls"
    utilities.freeze(ASSESS / "judge_protocol.json", protocol)
    utilities.freeze(ASSESS / "assessment_manifest.json", {"queries": len(queries), "candidates": len(candidates),
        "expected_pairs": len(queries)*len(candidates), "distinct_material_packets": len({p['packet_sha256'] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values()),
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"), "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl")})
    return queries, candidates, packets, protocol


def backend(model):
    from frontier_evidence_first_decoding import evidence_first_constraint
    parent = utilities.PhiJudge if model == "phi" else utilities.backend("qwen")
    class EvidenceFirstJudge(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.json_constraint = evidence_first_constraint(self.json_constraint)
    return EvidenceFirstJudge


def validate_output(output, case):
    parsed = base.parse_answers(output, [1], case["packet"]["refs"])
    base.validate_explicit_scope(parsed[1], case["query"], case["packet"]["refs"], case["packet"]["body"]["model_compounds"])
    return parsed[1]


def validation_error(output, case):
    try:
        validate_output(output, case)
        return None
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return str(error)[:240] if isinstance(error, ValueError) else type(error).__name__


def pilot(model):
    ensure_not_quality_stopped()
    lock = (RUN / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pid.json", base.identity())
    prepare()
    from frontier_judge_validation_v3 import evaluate_validation_v3
    path = ASSESS / "validation" / f"{model}_records.jsonl"
    raw_path = ASSESS / "validation" / f"{model}_raw_generation_journal.jsonl"
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    state = {**base.identity(), "state": "verifying_weights", "model": model}
    status_path = RUN / f"pilot_{model}_status.json"
    def update(**values):
        utilities.status_update(status_path, state, **values)
    update()
    weights = utilities.verify_weights(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS)
    pilot_spec = {"model_repo": weights["repo"], "model_revision": weights["revision"],
        "judge_protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "validation_sha256": base.file_sha(ASSESS / "validation/manifest.json"),
        "maximum_batch_size": 16, "memory_gib": 36, "raw_max_tokens": 96, "repair_max_tokens": 160,
        "control_repairs": 0, "noncontrol_repairs": 1, "raw_journal_before_repairs": True}
    protocol_path = ASSESS / "validation" / f"{model}_protocol.json"
    utilities.freeze(protocol_path, pilot_spec)
    pilot_hash = base.file_sha(protocol_path)
    finals = base.load_committed(path, repair_tail=True)
    raw_rows = base.load_committed(raw_path, repair_tail=True)
    expected = {c["case_id"] for c in cases}
    def index(values):
        result = {r["case_id"]: r for r in values}
        if len(result) != len(values) or not set(result) <= expected:
            raise ValueError("Duplicate or unexpected pilot case")
        if any(r.get("pilot_protocol_sha256") != pilot_hash or
               r.get("model_repo") != weights["repo"] or r.get("model_revision") != weights["revision"] or
               r.get("validation_sha256") != pilot_spec["validation_sha256"] for r in values):
            raise ValueError("Pilot from another frozen protocol")
        return result
    completed, generated = index(finals), index(raw_rows)
    if any(cid not in generated or row["output"] != generated[cid]["output"] for cid, row in completed.items()):
        raise ValueError("Final pilot output lacks matching original raw journal")
    pending = [c for c in cases if c["case_id"] not in completed]
    if pending:
        judge = backend(model)(weights, 36, lambda: update(state="loading_gpu"))
        update(state="judging", expected=len(cases), completed=len(completed))
        position, maximum = 0, 16
        while position < len(pending):
            if (RUN / "stop.requested").exists():
                update(state="stopped"); return
            subset = pending[position:position+maximum]
            if subset[0]["case_id"] not in generated:
                # Stop before an already journaled case so no raw output is overwritten.
                subset = subset[:next((i for i, c in enumerate(subset) if c["case_id"] in generated), len(subset))]
                tasks = [(c["packet"], [c["query"]]) for c in subset]
                size = judge.select_batch_size(tasks, maximum)
                subset, tasks = subset[:size], tasks[:size]
                try:
                    outputs, metrics = judge.generate(tasks)
                except judge.torch.cuda.OutOfMemoryError:
                    judge.torch.cuda.empty_cache()
                    if size == 1: raise
                    maximum = max(1, size//2); continue
                if len(outputs) != len(subset):
                    raise ValueError("Pilot generation cardinality mismatch")
                records = [{"case_id": c["case_id"], "kind": c["kind"], "output": output,
                    "model_repo": weights["repo"], "model_revision": weights["revision"],
                    "validation_sha256": pilot_spec["validation_sha256"], "pilot_protocol_sha256": pilot_hash,
                    "at": base.now(), "metrics": metrics} for c, output in zip(subset, outputs)]
                base.append_rows(raw_path, records)
                generated.update({r["case_id"]: r for r in records})
            else:
                subset = subset[:next((i for i, c in enumerate(subset) if c["case_id"] not in generated), len(subset))]
            for case in subset:
                raw = generated[case["case_id"]]
                output = raw["output"]
                error = validation_error(output, case)
                attempts = [{"output": output, "batch_metrics": raw["metrics"]}]
                if error is not None and case["kind"] != "synthetic_control":
                    outputs, metrics = judge.generate([(case["packet"], [case["query"]])], max_new_tokens=160,
                        repairs=[{"output": output, "error": error}])
                    if len(outputs) != 1: raise ValueError("Repair cardinality mismatch")
                    output = outputs[0]
                    attempts.append({"output": output, "batch_metrics": metrics})
                final_error = validation_error(output, case)
                row = {**raw, "attempts": attempts, "final_output": output,
                    "raw_validation_error": error, "final_validation_error": final_error,
                    "final_status": "valid" if final_error is None else "invalid_output"}
                base.append_rows(path, [row])
                completed[case["case_id"]] = row
                update(completed=len(completed), last_commit_at=base.now())
            position += len(subset)
        del judge
    result = evaluate_validation_v3(list(completed.values()), cases)
    base.dump(ASSESS / "validation" / f"{model}_evaluation.json", result)
    update(state="complete", completed=len(completed), expected=len(cases))


def admission_from_reports(reports):
    phi, qwen = reports["phi"], reports["qwen"]
    groups = {role: {"phi": phi["admission_by_role"][role], "qwen": qwen["admission_by_role"][role],
        "accepted": utilities.passes_gate(phi["admission_by_role"][role], qwen["admission_by_role"][role])}
        for role in CONTROL_ROLES}
    rows = [row for report in phi["by_role"].values() for row in report["rows"]]
    format_rate = sum(r["schema_valid"] for r in rows)/len(rows) if rows else 0.
    citation_rate = sum(r["citation_numbers_valid"] for r in rows)/len(rows) if rows else 0.
    coverage = phi["final_validation_by_role"]["prospective_source_coverage"]
    windows = phi["coverage_sequence"]
    accepted = (phi["complete"] and qwen["complete"] and all(g["accepted"] for g in groups.values()) and
        format_rate >= GATE["raw_format_valid_minimum"] and citation_rate >= GATE["citation_numbers_valid_minimum"] and
        coverage["final_valid_fraction"] >= .98 and windows["complete"] and not windows["any_full_window_above10percent"])
    return {"accepted": bool(accepted), "thresholds": GATE, "control_groups": groups,
        "raw_format_valid_fraction": format_rate, "raw_citation_valid_fraction": citation_rate,
        "representative_final_validation": coverage, "representative_windows": windows,
        "primary": "microsoft/phi-4", "secondary": "Qwen/Qwen3.6-27B",
        "semantic_gold_accuracy": None, "real_source_entailment_proven": False}


def verify_review_plan(cases):
    path = ASSESS / "source_review_plan.json"
    if not path.exists() or base.file_sha(path) != REVIEW_PLAN_SHA:
        raise ValueError("Frozen source-review case plan missing or changed")
    plan = base.read(path)
    ids = plan.get("case_ids", [])
    if (plan.get("validation_manifest_sha256") != base.file_sha(ASSESS / "validation/manifest.json") or
            len(set(ids)) != len(ids) or len(ids) != plan.get("case_count") or
            not set(ids) <= {c["case_id"] for c in cases}):
        raise ValueError("Source-review case plan differs from validation cohort")
    return plan


def verify_source_review():
    path = ASSESS / "source_review.json"
    review = base.read(path)
    if not review:
        raise ValueError("Source review has not been recorded")
    if (review.get("version") != "pbpk-v3-source-review-v1" or
            review.get("protocol_sha256") != base.file_sha(ASSESS / "judge_protocol.json") or
            review.get("validation_manifest_sha256") != base.file_sha(ASSESS / "validation/manifest.json") or
            review.get("pilot_record_sha256") != {model: base.file_sha(ASSESS / "validation" / f"{model}_records.jsonl") for model in ("phi", "qwen")}):
        raise ValueError("Source review differs from these frozen pilot outputs")
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    plan = verify_review_plan(cases)
    checked = review.get("checked_case_ids", [])
    required = set(plan["case_ids"]) | {c["case_id"] for c in cases if c["strata"]["validation_role"] == "production_failure_diagnostic"}
    if (not checked or len(set(checked)) != len(checked) or not set(checked) <= {c["case_id"] for c in cases} or
            not required <= set(checked) or review.get("reviewer") != "Codex evidence audit; not expert gold" or
            "scientific_gold_accuracy" not in review or review["scientific_gold_accuracy"] is not None or
            not isinstance(review.get("findings"), list) or not isinstance(review.get("known_unknown_distribution"), dict) or
            type(review.get("accepted")) is not bool):
        raise ValueError("Source review lacks the required diagnostic coverage and non-gold disclosure")
    return review["accepted"], base.file_sha(path)


def wait_for_source_review(update):
    update(stage="waiting_for_source_review", worker_pid=None)
    publish_summary()
    started = time.monotonic()
    while not (ASSESS / "source_review.json").exists():
        if (RUN / "stop.requested").exists(): update(state="stopped"); return False
        if time.monotonic()-started > 1200:
            update(state="stopped_waiting_for_source_review"); return False
        time.sleep(20)
    accepted, digest = verify_source_review()
    update(source_review_sha256=digest)
    if not accepted:
        update(state="stopped_on_source_review"); return False
    return True


def check_gate(require_review=True):
    prepare()
    from frontier_judge_validation_v3 import evaluate_validation_v3
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    reports = {}
    for model in ("phi", "qwen"):
        records = base.load_committed(ASSESS / "validation" / f"{model}_records.jsonl")
        pilot_path = ASSESS / "validation" / f"{model}_protocol.json"
        protocol = base.read(pilot_path)
        if not protocol or protocol.get("judge_protocol_sha256") != base.file_sha(ASSESS / "judge_protocol.json"):
            raise ValueError("Pilot protocol does not belong to this v3 run")
        weights = base.read(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS)
        if any(row.get("pilot_protocol_sha256") != base.file_sha(pilot_path) or
               row.get("model_repo") != weights["repo"] or row.get("model_revision") != weights["revision"] or
               row.get("validation_sha256") != base.file_sha(ASSESS / "validation/manifest.json") for row in records):
            raise ValueError("Pilot record provenance mismatch")
        raw_rows = base.load_committed(ASSESS / "validation" / f"{model}_raw_generation_journal.jsonl")
        raw = {r["case_id"]: r for r in raw_rows}
        if len(raw) != len(raw_rows) or set(raw) != {c["case_id"] for c in cases}:
            raise ValueError("Raw pilot journal is not the exact complete cohort")
        kinds = {c["case_id"]: c["kind"] for c in cases}
        for row in [*raw_rows, *records]:
            if (row.get("model_repo") != weights["repo"] or row.get("model_revision") != weights["revision"] or
                    row.get("pilot_protocol_sha256") != base.file_sha(pilot_path) or
                    row.get("validation_sha256") != base.file_sha(ASSESS / "validation/manifest.json") or
                    row.get("kind") != kinds.get(row.get("case_id"))):
                raise ValueError("Raw/final pilot identity, model or validation provenance changed")
        for row in records:
            attempts = row.get("attempts", [])
            if (row["case_id"] not in raw or raw[row["case_id"]]["output"] != row["output"] or
                    not attempts or attempts[0]["output"] != row["output"] or attempts[-1]["output"] != row["final_output"] or
                    len(attempts) > (1 if row["kind"] == "synthetic_control" else 2)):
                raise ValueError("Pilot attempts are not consistent with the original raw output")
        reports[model] = evaluate_validation_v3(records, cases)
    result = {"at": base.now(), "judge_protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
              **admission_from_reports(reports)}
    result["numeric_accepted"] = result["accepted"]
    if require_review:
        accepted, review_hash = verify_source_review()
        result.update(accepted=bool(result["accepted"] and accepted), source_review_sha256=review_hash)
    else:
        result.update(accepted=False, source_review_state="not_yet_required_for_numeric_diagnostic")
    base.dump(ASSESS / "validation/admission.json", result)
    return result


def run_full(judge_name):
    ensure_not_quality_stopped()
    if not check_gate()["accepted"]:
        raise RuntimeError("Frozen v3 admission has not passed; full scoring is blocked")
    base.prepare = prepare
    base.GPUJudge = backend("phi" if judge_name == "primary" else "qwen")
    base.download_primary = lambda: utilities.verify_weights(PHI_WEIGHTS)
    base.run_assessment(SimpleNamespace(judge=judge_name, memory_gib=36, batch_size=16, max_batches=0,
                                       secondary_weights=QWEN_WEIGHTS))


def command(python, *args):
    return [str(python), str(Path(__file__).resolve()), *args]


def publish_summary():
    folder = ROOT / "experiments/model_recommendation/results_summary"
    link = folder / "06_证据优先评分"
    if not link.exists(): link.symlink_to(os.path.relpath(ASSESS, folder), target_is_directory=True)
    path = folder / "README.md"
    text = path.read_text()
    first, last = "<!-- evidence-first-v3:start -->", "<!-- evidence-first-v3:end -->"
    state = base.read(RUN / "pipeline_status.json", {})
    worker = base.read(RUN / "status.json", {})
    judge = worker.get("judge")
    progress = (f"{'主评分' if judge == 'primary' else '5% 复判'}已提交：{worker.get('committed_pairs', 0):,}/"
                f"{worker.get('expected_pairs', 216160):,}。" if judge in {"primary", "secondary"}
                else "全量评分尚未启动。")
    block = (f"{first}\n## 证据优先评分 v3\n\n"
        f"更新：{base.now()}；阶段：`{state.get('stage', 'prepare')}`；状态：`{state.get('state', 'preparing')}`。"
        f"{progress}\n\n"
        "旧评分触发质量停止的记录保留。v3 先输出模型自行选择的证据，再输出等级；无引用时保持未知。"
        "Phi-4 主评分、Qwen3.6-27B 固定 5% 敏感性复判；先通过开发回归、新控制和代表原文的一次修复后验收。"
        "语法通过不等于证据语义正确，不宣称专家金标准。\n\n"
        "[v3 状态与结果](06_证据优先评分/)；命令："
        "`runtime/app-env/bin/python scripts/run_frontier_scoring_v3.py status` / `stop` / `start`。"
        f"SSH 断开后继续；质量停止后禁止原样重启。\n{last}")
    if first in text:
        if last not in text: raise ValueError("Incomplete README markers")
        text = text[:text.index(first)] + block + text[text.index(last)+len(last):]
    else: text += "\n" + block + "\n"
    path.write_text(text)


def pipeline():
    ensure_not_quality_stopped()
    lock = (RUN / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pipeline_pid.json", base.identity())
    state = {**base.identity(), "state": "running", "started_at": base.now(), "monitor_seconds": 20,
             "scheduler_sha256": base.file_sha(Path(__file__))}
    def update(**values): utilities.status_update(RUN / "pipeline_status.json", state, **values)
    def stop_signal(signum, frame): raise KeyboardInterrupt()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop_signal)
    child = None
    try:
        update(stage="prepare"); prepare()
        stages = [("pilot_phi", APP, ["pilot", "--model", "phi"], RUN / "pilot_phi_status.json"),
                  ("pilot_qwen", MODERN, ["pilot", "--model", "qwen"], RUN / "pilot_qwen_status.json"),
                  ("primary", APP, ["run", "--judge", "primary"], RUN / "status.json"),
                  ("secondary", MODERN, ["run", "--judge", "secondary"], RUN / "status.json")]
        for stage, python, args, status_path in stages:
            if (RUN / "stop.requested").exists(): update(state="stopped"); return
            if stage == "primary":
                if not check_gate(require_review=False)["numeric_accepted"]:
                    update(state="stopped_on_validation", stage="admission_failed"); return
                if not wait_for_source_review(update): return
                if not check_gate()["accepted"]:
                    update(state="stopped_on_source_review"); return
            update(stage=stage, worker_pid=None); publish_summary()
            with (RUN / "runner.log").open("a") as log:
                child = subprocess.Popen(command(python, *args), cwd=ROOT, env=dict(os.environ),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            update(worker_pid=child.pid)
            last, changed, started = None, time.monotonic(), time.monotonic()
            while child.poll() is None:
                current = base.read(status_path, {})
                if current.get("pid") != child.pid: current = {"state": "starting"}
                signature = tuple(current.get(k) for k in ("state", "completed", "committed_pairs", "last_commit_at"))
                if signature != last:
                    last, changed = signature, time.monotonic()
                    update(worker_state=current.get("state"), completed=current.get("completed"),
                           committed_pairs=current.get("committed_pairs"), last_commit_at=current.get("last_commit_at"))
                if (RUN / "stop.requested").exists():
                    base.stop_child(child); update(state="stopped", worker_pid=None); return
                if (time.monotonic()-changed > 1200 or time.monotonic()-started > 8*86400 or
                        shutil.disk_usage(ROOT).free < 30*1024**3 or (RUN / "runner.log").stat().st_size > 20*1024**2):
                    base.stop_child(child); update(state="stopped_on_guard", worker_pid=None); return
                time.sleep(20)
            current = base.read(status_path, {})
            # Copy final worker counts even when the worker stopped between polls.
            update(completed=current.get("completed"), committed_pairs=current.get("committed_pairs"),
                   last_commit_at=current.get("last_commit_at"), guard_reason=current.get("guard_reason"))
            if child.returncode or current.get("pid") != child.pid or current.get("state") != "complete":
                update(state="failed", failed_stage=stage, worker_pid=None, worker_state=current.get("state")); return
            child = None
            if stage == "primary":
                update(stage="primary_metrics")
                from evaluate_frontier_automatic import evaluate_run
                evaluate_run(OUT, ASSESS, require_complete=True)
        update(stage="final_evaluation")
        agreement = base.judge_agreement()
        agreement.update(primary_repo=base.read(PHI_WEIGHTS)["repo"], secondary_repo=base.read(QWEN_WEIGHTS)["repo"],
                         protocol_sha256=base.file_sha(ASSESS / "judge_protocol.json"),
                         source_review_sha256=verify_source_review()[1],
                         interpretation="Fixed secondary sensitivity only; agreement does not establish source correctness or expert gold")
        base.dump(ASSESS / "judge_agreement.json", agreement)
        from evaluate_frontier_automatic import evaluate_run
        result = evaluate_run(OUT, ASSESS, require_complete=True)
        update(state="complete", completed_at=base.now(), evaluation_state=result["state"], worker_pid=None)
    except BaseException as error:
        update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__)
        raise
    finally:
        base.stop_child(child); publish_summary()


def start():
    ensure_not_quality_stopped()
    directories = [RUN, ROOT / "runtime/deployment/frontier_assessment",
        *[ROOT / "runtime/deployment" / value[1] for value in utilities.PROFILE_DIRECTORIES.values()]]
    for directory in directories:
        for name in ("pid.json", "pipeline_pid.json"):
            if base.alive(base.read(directory / name, {})):
                raise RuntimeError("A project scoring worker or pipeline is already running")
    (RUN / "stop.requested").unlink(missing_ok=True)
    with (RUN / "runner.log").open("a") as log:
        child = subprocess.Popen(command(APP, "pipeline"), cwd=ROOT, env=dict(os.environ),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return {"started_pid": child.pid, "status": str(RUN / "pipeline_status.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "pipeline", "pilot", "run", "status", "stop"))
    parser.add_argument("--model", choices=("phi", "qwen"), default="phi")
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    args = parser.parse_args(); initialize()
    if args.action == "start": print(json.dumps(start()))
    elif args.action == "pipeline": pipeline()
    elif args.action == "prepare": prepare(); print("v3 protocol verified and frozen")
    elif args.action == "stop": (RUN / "stop.requested").touch(); print("Stop requested; outputs retained")
    elif args.action == "status":
        print(json.dumps({"alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {})}))
    else:
        try:
            pilot(args.model) if args.action == "pilot" else run_full(args.judge)
        except BaseException as error:
            path = RUN / (f"pilot_{args.model}_status.json" if args.action == "pilot" else "status.json")
            current = base.read(path, {})
            utilities.status_update(path, current, state="failed", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    main()
