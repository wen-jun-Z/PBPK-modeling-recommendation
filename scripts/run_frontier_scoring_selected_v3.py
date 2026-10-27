"""Conditionally select the qualified Qwen v3 judge; preserve all prior runs."""
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
import time
from types import SimpleNamespace

import run_frontier_scoring_v3 as parent

base, utilities = parent.base, parent.utilities
ROOT, OUT = parent.ROOT, parent.OUT
PARENT_ASSESS, PARENT_RUN = parent.ASSESS, parent.RUN
ASSESS = OUT / "assessment_selected_v3"
RUN = ROOT / "runtime/deployment/frontier_scoring_selected_v3"
APP, MODERN = parent.APP, parent.MODERN
PHI_WEIGHTS, QWEN_WEIGHTS = parent.PHI_WEIGHTS, parent.QWEN_WEIGHTS
SYSTEM, GATE, CONTROL_ROLES = parent.SYSTEM, parent.GATE, parent.CONTROL_ROLES
AMENDMENT_SHA = "c263e26afa87c9bac2e01d592049951688041048c906f351f1eef88899f1c41a"
VERSION = "pbpk-qualified-evidence-first-scoring-v3"
PILOT_FILES = ("cases.jsonl", "manifest.json", *(
    f"{model}_{suffix}" for model in ("phi", "qwen") for suffix in
    ("records.jsonl", "raw_generation_journal.jsonl", "protocol.json", "evaluation.json")))


def configure():
    # These are process-local module globals; no frozen file is changed.
    parent.ASSESS, parent.RUN = ASSESS, RUN
    parent.configure()


def initialize():
    configure()
    parent.initialize()


def assert_other_runs_idle():
    directories = [PARENT_RUN, ROOT / "runtime/deployment/frontier_assessment",
        *[ROOT / "runtime/deployment" / v[1] for v in utilities.PROFILE_DIRECTORIES.values()]]
    for directory in directories:
        for name in ("pid.json", "pipeline_pid.json"):
            if base.alive(base.read(directory / name, {})):
                raise RuntimeError("A prior project scoring worker/pipeline is still running; wait for its exit")


def verify_amendment():
    path = ASSESS / "selection_amendment.json"
    if not path.exists() or base.file_sha(path) != AMENDMENT_SHA:
        raise ValueError("Frozen selection amendment missing or changed")
    value = base.read(path)
    for name, field in (("judge_protocol.json", "parent_protocol_sha256"),
                        ("validation/manifest.json", "parent_validation_manifest_sha256"),
                        ("source_review_plan.json", "parent_source_review_plan_sha256")):
        if base.file_sha(PARENT_ASSESS / name) != value[field]:
            raise ValueError("Selection parent artifact changed")
    if (value["primary_candidate"] != "qwen" or value["secondary"] != "phi" or
            value["thresholds"] != GATE or value["coverage_minimum"] != .98 or
            value["coverage_window"] != 128 or value["maximum_window_invalid_fraction"] != .10 or
            value["execution_policy"]["full_labels_reused"] is not False or
            value["execution_policy"]["ranking_scores_used_in_judge_selection"] is not False):
        raise ValueError("Selection amendment is inconsistent with this unchanged qualification rule")
    return value


def link_file(source, target):
    if not source.is_file() or not source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Only existing project files may be reused")
    if target.is_symlink() or target.exists():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError("Existing validation link has an unexpected target")
    else:
        target.symlink_to(os.path.relpath(source, target.parent))


def link_validation():
    directory = ASSESS / "validation"
    if directory.is_symlink():
        raise ValueError("Validation directory must be independent, never a directory symlink")
    directory.mkdir(parents=True, exist_ok=True)
    for name in PILOT_FILES:
        link_file(PARENT_ASSESS / "validation" / name, directory / name)
    link_file(PARENT_ASSESS / "source_review_plan.json", ASSESS / "source_review_plan.json")
    # Neither full labels nor writable admission/review reports are reused.
    for name in ("validation/admission.json", "source_review.json", "judgments.jsonl",
                 "secondary_judgments.jsonl", "primary_judgment_cache.jsonl", "secondary_judgment_cache.jsonl",
                 "primary_generation_journal.jsonl", "secondary_generation_journal.jsonl"):
        if (ASSESS / name).is_symlink():
            raise ValueError("Writable selected-run output must not link to an old run")


def verify_pilot_records(model, cases, parent_sha, validation_sha):
    folder = ASSESS / "validation"
    pilot_path = folder / f"{model}_protocol.json"
    weights = base.read(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS)
    expected_protocol = {"model_repo": weights["repo"], "model_revision": weights["revision"],
        "judge_protocol_sha256": parent_sha, "validation_sha256": validation_sha,
        "maximum_batch_size": 16, "memory_gib": 36, "raw_max_tokens": 96, "repair_max_tokens": 160,
        "control_repairs": 0, "noncontrol_repairs": 1, "raw_journal_before_repairs": True}
    if base.read(pilot_path) != expected_protocol:
        raise ValueError("Reused pilot has different generation, model, or source protocol")
    records = base.load_committed(folder / f"{model}_records.jsonl")
    raw_rows = base.load_committed(folder / f"{model}_raw_generation_journal.jsonl")
    by_id = {c["case_id"]: c for c in cases}
    for rows in (records, raw_rows):
        if len(rows) != len(by_id) or {r["case_id"] for r in rows} != set(by_id):
            raise ValueError("Reused pilot is not the exact complete duplicate-free cohort")
        for row in rows:
            if (row.get("model_repo") != weights["repo"] or row.get("model_revision") != weights["revision"] or
                    row.get("pilot_protocol_sha256") != base.file_sha(pilot_path) or
                    row.get("validation_sha256") != validation_sha or
                    row.get("kind") != by_id[row["case_id"]]["kind"]):
                raise ValueError("Reused raw/final pilot provenance changed")
    raw = {r["case_id"]: r for r in raw_rows}
    for row in records:
        attempts = row.get("attempts", [])
        maximum = 1 if row["kind"] == "synthetic_control" else 2
        if (raw[row["case_id"]]["output"] != row["output"] or not 1 <= len(attempts) <= maximum or
                attempts[0]["output"] != row["output"] or attempts[-1]["output"] != row["final_output"]):
            raise ValueError("Reused pilot raw/final attempts disagree")
    return records


def prepare():
    configure()
    assert_other_runs_idle()
    amendment = verify_amendment()
    old = base.read(PARENT_ASSESS / "judge_protocol.json")
    for name, digest in old["implementation_sha256"].items():
        if base.file_sha(ROOT / "scripts" / name) != digest:
            raise ValueError("Frozen parent scoring implementation changed")
    from frontier_evidence_first_decoding import decoding_protocol
    environments = utilities.execution_environments()
    if (old["system_prompt"] != SYSTEM or old["system_prompt_sha256"] != base.sha(SYSTEM.encode()) or
            old["generation"]["constrained_decoding"] != decoding_protocol() or
            old["execution_environments"] != environments or
            old["primary_weights_manifest_sha256"] != base.file_sha(PHI_WEIGHTS) or
            old["secondary_weights_manifest_sha256"] != base.file_sha(QWEN_WEIGHTS)):
        raise ValueError("Parent pilot and selected scoring generation are not equivalent")
    link_validation()
    queries, candidates, packets, _ = utilities.prepare_materials()
    from frontier_judge_validation_v3 import verify_validation_v3
    verify_validation_v3(queries, candidates, packets, ASSESS / "validation")
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    parent.verify_review_plan(cases)
    for model in ("phi", "qwen"):
        verify_pilot_records(model, cases, amendment["parent_protocol_sha256"],
                             amendment["parent_validation_manifest_sha256"])
    protocol = copy.deepcopy(old)
    for suffix in ("repo", "revision", "weights_manifest_sha256"):
        protocol[f"primary_{suffix}"] = old[f"secondary_{suffix}"]
        protocol[f"secondary_{suffix}"] = old[f"primary_{suffix}"]
    protocol.update(version=VERSION, primary_mode="Qwen3.6-27B enable_thinking=False; evidence-first constrained JSON",
        secondary_mode="Phi-4 instruction model; evidence-first constrained JSON; fixed 5% sensitivity only",
        parent_protocol_sha256=amendment["parent_protocol_sha256"], selection_amendment_sha256=AMENDMENT_SHA,
        selection_timing=amendment["selection_timing"], execution_environments=environments,
        parent_pilot_outputs_reused_for_validation=True, full_labels_reused=False,
        pilot_reuse_sha256={name: base.file_sha(ASSESS / "validation" / name) for name in PILOT_FILES},
        implementation_sha256={**old["implementation_sha256"], Path(__file__).name: base.file_sha(Path(__file__))})
    utilities.freeze(ASSESS / "judge_protocol.json", protocol)
    utilities.freeze(ASSESS / "assessment_manifest.json", {
        "queries": len(queries), "candidates": len(candidates), "expected_pairs": len(queries)*len(candidates),
        "distinct_material_packets": len({p["packet_sha256"] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values()),
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl")})
    return queries, candidates, packets, protocol


def admission_from_reports(reports):
    # Apply the identical frozen arithmetic with selected primary/reference roles,
    # then restore real model labels in the returned report.
    result = parent.admission_from_reports({"phi": reports["qwen"], "qwen": reports["phi"]})
    result["control_groups"] = {role: {"qwen": group["phi"], "phi": group["qwen"], "accepted": group["accepted"]}
        for role, group in result["control_groups"].items()}
    result.update(primary="Qwen/Qwen3.6-27B", secondary="microsoft/phi-4",
        secondary_interpretation="Unqualified Phi v3 sensitivity reference, not a qualified scientific gold judge",
        parent_phi_numeric_admission=parent.admission_from_reports(reports)["accepted"])
    return result


def verify_source_review():
    configure()
    accepted, digest = parent.verify_source_review()
    review = base.read(ASSESS / "source_review.json")
    if (review.get("parent_protocol_sha256") != base.file_sha(PARENT_ASSESS / "judge_protocol.json") or
            review.get("primary") != "qwen" or review.get("secondary") != "phi"):
        raise ValueError("Source review does not bind the parent pilots and selected judge roles")
    return accepted, digest


def check_gate(require_review=True):
    prepare()
    from frontier_judge_validation_v3 import evaluate_validation_v3
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    reports = {model: evaluate_validation_v3(base.load_committed(ASSESS / "validation" / f"{model}_records.jsonl"), cases)
               for model in ("phi", "qwen")}
    result = {"at": base.now(), "judge_protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "selection_amendment_sha256": AMENDMENT_SHA, **admission_from_reports(reports)}
    result["numeric_accepted"] = result["accepted"]
    if require_review:
        accepted, digest = verify_source_review()
        result.update(accepted=bool(result["accepted"] and accepted), source_review_sha256=digest)
    else:
        result.update(accepted=False, source_review_state="required_before_full_scoring")
    base.dump(ASSESS / "validation/admission.json", result)
    return result


def run_full(judge):
    configure()
    parent.ensure_not_quality_stopped()
    if not check_gate()["accepted"]:
        raise RuntimeError("Selected Qwen v3 admission has not passed; full scoring remains blocked")
    base.prepare = prepare
    base.GPUJudge = parent.backend("qwen" if judge == "primary" else "phi")
    base.download_primary = lambda: utilities.verify_weights(QWEN_WEIGHTS)
    base.run_assessment(SimpleNamespace(judge=judge, memory_gib=36, batch_size=16, max_batches=0,
                                       secondary_weights=PHI_WEIGHTS))


def command(python, *args):
    return [str(python), str(Path(__file__).resolve()), *args]


def stages():
    return [("primary", MODERN, ["run", "--judge", "primary"]),
            ("secondary", APP, ["run", "--judge", "secondary"])]


def publish_summary():
    folder = ROOT / "experiments/model_recommendation/results_summary"
    link = folder / "07_合格裁判证据评分"
    if not link.exists(): link.symlink_to(os.path.relpath(ASSESS, folder), target_is_directory=True)
    path = folder / "README.md"
    text = path.read_text()
    first, last = "<!-- selected-v3:start -->", "<!-- selected-v3:end -->"
    state, worker = base.read(RUN / "pipeline_status.json", {}), base.read(RUN / "status.json", {})
    judge = worker.get("judge")
    progress = (f"{'主评分' if judge == 'primary' else '5% 复判'}已提交：{worker.get('committed_pairs', 0):,}/"
                f"{worker.get('expected_pairs', 216160):,}。" if judge in {"primary", "secondary"} else "全量评分尚未启动。")
    block = (f"{first}\n## 合格裁判证据评分 v3\n\n更新：{base.now()}；阶段：`{state.get('stage', 'prepare')}`；"
        f"状态：`{state.get('state', 'preparing')}`。{progress}\n\n"
        "保留 Phi v3 未通过准入的全部记录；Qwen3.6-27B 必须通过相同的三组控制、代表原文检查与 39 例来源复核后才可主评分。"
        "Phi-4 仅作固定 5% 敏感性复判，未通过主裁判准入。复用经哈希核验的 733×2 条预检输出，全量标签独立重新生成；"
        "无证据和失败保持未知，原 10% 质量停止规则不变。自动来源支持评分不等于专家金标准。\n\n"
        "[状态与结果](07_合格裁判证据评分/)；命令：`runtime/app-env/bin/python scripts/run_frontier_scoring_selected_v3.py "
        f"status` / `stop` / `start`。SSH 断开后继续。\n{last}")
    if first in text:
        if last not in text: raise ValueError("Incomplete README markers")
        text = text[:text.index(first)] + block + text[text.index(last)+len(last):]
    else: text += "\n" + block + "\n"
    path.write_text(text)


def pipeline():
    configure()
    parent.ensure_not_quality_stopped()
    assert_other_runs_idle()
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
        update(stage="admission")
        if not check_gate(require_review=False)["numeric_accepted"]:
            update(state="stopped_on_validation", stage="admission_failed"); return
        update(stage="waiting_for_source_review")
        publish_summary()
        waiting = time.monotonic()
        while not (ASSESS / "source_review.json").exists():
            if (RUN / "stop.requested").exists(): update(state="stopped"); return
            if time.monotonic()-waiting > 1200:
                update(state="stopped_waiting_for_source_review"); return
            time.sleep(20)
        if not check_gate()["accepted"]:
            update(state="stopped_on_source_review"); return
        for stage, python, args in stages():
            if (RUN / "stop.requested").exists(): update(state="stopped"); return
            update(stage=stage, worker_pid=None); publish_summary()
            with (RUN / "runner.log").open("a") as log:
                child = subprocess.Popen(command(python, *args), cwd=ROOT, env=dict(os.environ),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            update(worker_pid=child.pid)
            last, changed, started = None, time.monotonic(), time.monotonic()
            while child.poll() is None:
                current = base.read(RUN / "status.json", {})
                if current.get("pid") != child.pid: current = {"state": "starting"}
                signature = tuple(current.get(k) for k in ("state", "committed_pairs", "last_commit_at"))
                if signature != last:
                    last, changed = signature, time.monotonic()
                    update(worker_state=current.get("state"), committed_pairs=current.get("committed_pairs"),
                           last_commit_at=current.get("last_commit_at"))
                if (RUN / "stop.requested").exists():
                    base.stop_child(child); update(state="stopped", worker_pid=None); return
                if (time.monotonic()-changed > 1200 or time.monotonic()-started > 8*86400 or
                        shutil.disk_usage(ROOT).free < 30*1024**3 or (RUN / "runner.log").stat().st_size > 20*1024**2):
                    base.stop_child(child); update(state="stopped_on_guard", worker_pid=None); return
                time.sleep(20)
            current = base.read(RUN / "status.json", {})
            update(committed_pairs=current.get("committed_pairs"), last_commit_at=current.get("last_commit_at"),
                   guard_reason=current.get("guard_reason"))
            if child.returncode or current.get("pid") != child.pid or current.get("state") != "complete":
                update(state="failed", failed_stage=stage, worker_pid=None, worker_state=current.get("state")); return
            child = None
            if stage == "primary":
                update(stage="primary_metrics")
                from evaluate_frontier_automatic import evaluate_run
                evaluate_run(OUT, ASSESS, require_complete=True)
        update(stage="final_evaluation")
        agreement = base.judge_agreement()
        agreement.update(primary_repo=base.read(QWEN_WEIGHTS)["repo"], secondary_repo=base.read(PHI_WEIGHTS)["repo"],
            protocol_sha256=base.file_sha(ASSESS / "judge_protocol.json"), source_review_sha256=verify_source_review()[1],
            interpretation="Fixed unqualified-secondary sensitivity only; agreement is not scientific gold or source correctness")
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
    configure()
    parent.ensure_not_quality_stopped()
    assert_other_runs_idle()
    for name in ("pid.json", "pipeline_pid.json"):
        if base.alive(base.read(RUN / name, {})):
            raise RuntimeError("Selected project scoring is already running")
    # Readiness and provenance are checked synchronously before creating a daemon.
    prepare()
    (RUN / "stop.requested").unlink(missing_ok=True)
    with (RUN / "runner.log").open("a") as log:
        child = subprocess.Popen(command(APP, "pipeline"), cwd=ROOT, env=dict(os.environ),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return {"started_pid": child.pid, "status": str(RUN / "pipeline_status.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "pipeline", "run", "status", "stop"))
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    args = parser.parse_args(); initialize()
    if args.action == "start": print(json.dumps(start()))
    elif args.action == "pipeline": pipeline()
    elif args.action == "prepare": prepare(); print("Selected v3 protocol verified and frozen; admission is separate")
    elif args.action == "stop": (RUN / "stop.requested").touch(); print("Stop requested; all outputs retained")
    elif args.action == "status":
        print(json.dumps({"alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {})}))
    else:
        try: run_full(args.judge)
        except BaseException as error:
            utilities.status_update(RUN / "status.json", base.read(RUN / "status.json", {}),
                                    state="failed", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    main()
