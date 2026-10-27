"""Reasoned Qwen primary with an explicitly named nonthinking Phi reference."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import run_frontier_scoring_reviewed as shared

base, utilities, engine = shared.base, shared.utilities, shared.engine
ROOT, OUT, LEGACY = shared.ROOT, shared.OUT, shared.LEGACY
APP, MODERN = shared.APP, shared.MODERN
PHI_WEIGHTS, QWEN_WEIGHTS = shared.PHI_WEIGHTS, shared.QWEN_WEIGHTS
GATE, CONTROL_ROLES = shared.GATE, shared.CONTROL_ROLES
SHARED_CONFIGURE = shared.configure
SHARED_PREPARE = shared.prepare
PLAIN_BACKEND = shared.backend
ORIGINAL_FREEZE = utilities.freeze
PROFILE, CONFIG, CONFIG_SHA, SYSTEM, ASSESS, RUN = None, {}, None, None, None, None


def load_profile(slug):
    global PROFILE, CONFIG, CONFIG_SHA, SYSTEM, ASSESS, RUN
    if not re.fullmatch(r"v[0-9]{1,3}", slug) or int(slug[1:]) < 6: raise ValueError("Invalid reasoned profile slug")
    value = shared.load_profile(slug)
    if (value.get("primary_model") != "qwen" or value.get("secondary_model") != "phi" or
            not re.fullmatch(r"assessment_[a-z0-9_]+", value.get("fresh_origin_assessment_directory", ""))):
        raise ValueError("Reasoned profile roles or fresh input scope mismatch")
    backend_module = shared.import_frozen(value["reasoned_backend_file"], value["reasoned_backend_sha256"], f"reasoned_backend_{slug}")
    if backend_module.generation_protocol() != value["reasoning_generation_protocol"]:
        raise ValueError("Reasoned generation protocol changed")
    PROFILE, CONFIG, CONFIG_SHA, SYSTEM = shared.PROFILE, shared.CONFIG, shared.CONFIG_SHA, shared.SYSTEM
    ASSESS, RUN = shared.ASSESS, shared.RUN
    return value


def configure():
    if PROFILE is None: raise ValueError("Load a frozen profile first")
    for name in ("PROFILE", "CONFIG", "CONFIG_SHA", "SYSTEM", "ASSESS", "RUN"):
        setattr(shared, name, globals()[name])
    shared.configure = configure
    SHARED_CONFIGURE()
    engine.prepare, engine.backend, engine.command = prepare, backend, command
    engine.check_gate, engine.verify_source_review, engine.publish_summary = check_gate, verify_source_review, publish_summary


def ensure_not_stopped():
    engine.ensure_not_quality_stopped()
    value = base.read(RUN / "pilot_qwen_status.json", {})
    if value.get("guard_reason", "").startswith("AbsoluteControlAdmissionFailed"):
        raise RuntimeError("This frozen Qwen primary failed admission; an identical restart is forbidden")


def prepare():
    configure(); shared.assert_prior_runs_idle()
    origin = OUT / CONFIG["fresh_origin_assessment_directory"] / "validation/fresh"
    target = ASSESS / "validation/fresh"
    if not origin.resolve().is_relative_to(OUT.resolve()) or target.is_symlink(): raise ValueError("Unscoped fresh inputs")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("cases.jsonl", "manifest.json"):
        source, path = origin / name, target / name
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != source.resolve(): raise ValueError("Fresh definition target changed")
        else: path.symlink_to(os.path.relpath(source, path.parent))
    module = shared.import_frozen(CONFIG["reasoned_backend_file"], CONFIG["reasoned_backend_sha256"], f"reasoned_backend_{PROFILE}")
    reasoning = module.generation_protocol()
    if reasoning != CONFIG["reasoning_generation_protocol"]: raise ValueError("Reasoned backend protocol changed")
    frozen = {}
    def freeze(path, value):
        if path == ASSESS / "judge_protocol.json":
            value = copy.deepcopy(value)
            for suffix in ("repo", "revision", "weights_manifest_sha256"):
                value[f"primary_{suffix}"], value[f"secondary_{suffix}"] = value[f"secondary_{suffix}"], value[f"primary_{suffix}"]
            value.update(version="pbpk-reasoned-primary-scoring-v1",
                primary_mode="Qwen3.6-27B native bounded thinking followed by conditional JSON",
                secondary_mode="Phi-4 instruction model; conditional JSON; no added thinking stage; sensitivity only",
                primary_early_stop="Qwen complete control group below an unchanged absolute gate stops before next group/model",
                fresh_origin="Same frozen 32 controls previously unexposed to either model; definitions reused, predictions never reused")
            value["generation"]["primary_reasoning"] = reasoning
            value["generation"]["secondary_reasoning"] = "None; Phi uses the original one-stage conditional generation"
            value["generation"]["max_new_tokens_scope"] = "96 is the final conditional JSON budget (160 on one repair); primary thinking has the separately frozen 1024-token budget"
            value["implementation_sha256"].update({Path(__file__).name: base.file_sha(Path(__file__)),
                Path(CONFIG["reasoned_backend_file"]).name: CONFIG["reasoned_backend_sha256"]})
            frozen["protocol"] = value
        return ORIGINAL_FREEZE(path, value)
    previous = utilities.freeze
    utilities.freeze = freeze
    try:
        queries, candidates, packets, _ = SHARED_PREPARE()
    finally:
        utilities.freeze = previous
    return queries, candidates, packets, frozen["protocol"]


def backend(model):
    if model == "phi": return PLAIN_BACKEND("phi")
    if model != "qwen": raise ValueError("Unknown actual model")
    return shared.import_frozen(CONFIG["reasoned_backend_file"], CONFIG["reasoned_backend_sha256"],
                                f"reasoned_backend_{PROFILE}").ReasonedGPUJudge


def pilot(model, job="all"):
    configure(); ensure_not_stopped(); shared.assert_prior_runs_idle()
    lock = (RUN / "runner.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pid.json", base.identity()); prepare()
    legacy, fresh = shared.cohorts()
    cohorts = {"legacy": legacy, "fresh": fresh}
    control_units = [("legacy", role, [c for c in legacy if c["strata"]["validation_role"] == role]) for role in CONTROL_ROLES]
    control_units += [("fresh", "fresh32", fresh)]
    units = control_units + [("legacy", "legacy_real_sources", [c for c in legacy if c["kind"] != "synthetic_control"])]
    if job != "all": units = [u for u in units if (job == "fresh" and u[0] == "fresh") or
        (job == "legacy-controls" and u[1] in CONTROL_ROLES) or (job == "legacy-sources" and u[1] == "legacy_real_sources")]
    state = {**base.identity(), "state": "verifying_weights", "model": model, "primary": model == "qwen", "job": job, "expected": 765}
    def update(**values): utilities.status_update(RUN / f"pilot_{model}_status.json", state, **values)
    update()
    completed, generated = {}, {}
    for cohort, cases in cohorts.items():
        utilities.freeze(shared.cohort_folder(cohort) / f"{model}_protocol.json", shared.pilot_spec(model, cohort))
        completed[cohort], generated[cohort] = shared.read_pilot(model, cohort, cases, allow_repair=True)
    judge = None
    def ensure_judge():
        nonlocal judge
        if judge is None:
            weights = utilities.verify_weights(QWEN_WEIGHTS if model == "qwen" else PHI_WEIGHTS)
            judge = backend(model)(weights, 36, lambda: update(state="loading_gpu"))
        return judge
    def check_control(cohort, role, cases):
        if role == "legacy_real_sources": return False
        metric = shared.control_metric(list(completed[cohort].values()), cases)
        failed = shared.absolute_failures(metric)
        base.dump(shared.cohort_folder(cohort) / f"{model}_{role}_gate.json", {"metrics": metric,
            "absolute_failed_checks": failed, "primary": model == "qwen", "relative_comparison_pending": True})
        if model == "qwen" and failed:
            update(state="stopped_on_control_gate", guard_reason=f"AbsoluteControlAdmissionFailed:{cohort}:{role}",
                failed_control_group=role, failed_checks=failed, control_metrics=metric,
                completed=sum(map(len, completed.values())), legacy_completed=len(completed["legacy"]), fresh_completed=len(completed["fresh"]))
            return True
        return False
    for cohort, role, cases in control_units:
        if check_control(cohort, role, cases): return
    if model == "qwen" and job == "legacy-sources" and any(
            not shared.control_metric(list(completed[c].values()), rows)["complete"] for c, _, rows in control_units):
        raise RuntimeError("Complete all four Qwen primary control groups before source-only generation")
    for cohort, role, cases in units:
        pending = [c for c in cases if c["case_id"] not in completed[cohort]]
        folder, position, maximum = shared.cohort_folder(cohort), 0, 16
        while position < len(pending):
            if (RUN / "stop.requested").exists(): update(state="stopped"); return
            batch, known = pending[position:position+maximum], generated[cohort]
            if batch[0]["case_id"] not in known:
                batch = batch[:next((i for i, c in enumerate(batch) if c["case_id"] in known), len(batch))]
                tasks = [(c["packet"], [c["query"]]) for c in batch]
                active = ensure_judge(); size = active.select_batch_size(tasks, maximum)
                batch, tasks = batch[:size], tasks[:size]
                try: outputs, metrics = active.generate(tasks)
                except active.torch.cuda.OutOfMemoryError:
                    active.torch.cuda.empty_cache()
                    if size == 1: raise
                    maximum = max(1, size//2); continue
                if len(outputs) != len(batch): raise ValueError("Pilot generation cardinality mismatch")
                spec = shared.pilot_spec(model, cohort)
                rows = [{"case_id": c["case_id"], "kind": c["kind"], "output": output, "cohort": cohort,
                    "model_repo": spec["model_repo"], "model_revision": spec["model_revision"],
                    "validation_sha256": spec["validation_sha256"], "pilot_protocol_sha256": base.file_sha(folder / f"{model}_protocol.json"),
                    "at": base.now(), "metrics": metrics} for c, output in zip(batch, outputs)]
                base.append_rows(folder / f"{model}_raw_generation_journal.jsonl", rows)
                known.update({r["case_id"]: r for r in rows})
            else: batch = batch[:next((i for i, c in enumerate(batch) if c["case_id"] not in known), len(batch))]
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
    base.dump(ASSESS / "validation/fresh" / f"{model}_evaluation.json", shared.controls.evaluate_validation(list(completed["fresh"].values()), fresh))
    complete = len(completed["legacy"]) == len(legacy) and len(completed["fresh"]) == len(fresh)
    update(state="complete" if complete else "job_complete", completed=sum(map(len, completed.values())),
        legacy_completed=len(completed["legacy"]), fresh_completed=len(completed["fresh"]))


def check_gate(require_review=True):
    configure(); ensure_not_stopped(); prepare()
    from frontier_judge_validation_v3 import evaluate_validation_v3
    legacy, fresh = shared.cohorts()
    reports, new = {}, {}
    for model in ("qwen", "phi"):
        rows_by_cohort = {}
        for cohort, cases in (("legacy", legacy), ("fresh", fresh)):
            rows, raw = shared.read_pilot(model, cohort, cases)
            if set(rows) != {c["case_id"] for c in cases} or set(raw) != set(rows):
                raise ValueError("Both actual models need exact complete 733 and 32 cohorts")
            rows_by_cohort[cohort] = list(rows.values())
        reports[model] = evaluate_validation_v3(rows_by_cohort["legacy"], legacy)
        new[model] = shared.control_metric(rows_by_cohort["fresh"], fresh)
    result = engine.admission_from_reports({"phi": reports["qwen"], "qwen": reports["phi"]})
    result["control_groups"] = {k: {"qwen": x["phi"], "phi": x["qwen"], "accepted": x["accepted"]}
                                for k, x in result["control_groups"].items()}
    result.update(at=base.now(), profile=PROFILE, profile_config_sha256=CONFIG_SHA,
        judge_protocol_sha256=base.file_sha(ASSESS / "judge_protocol.json"),
        primary="Qwen/Qwen3.6-27B", secondary="microsoft/phi-4",
        fresh_control_group={**new, "accepted": utilities.passes_gate(new["qwen"], new["phi"])})
    result["numeric_accepted"] = bool(result["accepted"] and result["fresh_control_group"]["accepted"])
    result["accepted"] = False
    if require_review:
        accepted, digest = verify_source_review()
        result.update(accepted=bool(result["numeric_accepted"] and accepted), source_review_sha256=digest)
    base.dump(ASSESS / "validation/admission.json", result)
    return result


def verify_source_review():
    accepted, digest = shared.verify_source_review()
    review = base.read(ASSESS / "source_review.json")
    if review.get("primary") != "qwen" or review.get("secondary") != "phi":
        raise ValueError("Source review must use the actual Qwen primary and Phi reference")
    return accepted, digest


def run_full(judge):
    configure(); ensure_not_stopped()
    if not check_gate()["accepted"]: raise RuntimeError("Reasoned primary admission has not passed")
    base.prepare, base.GPUJudge = prepare, backend("qwen" if judge == "primary" else "phi")
    base.download_primary = lambda: utilities.verify_weights(QWEN_WEIGHTS)
    base.run_assessment(SimpleNamespace(judge=judge, memory_gib=36, batch_size=16, max_batches=0, secondary_weights=PHI_WEIGHTS))


def command(python, *args):
    return [str(python), str(Path(__file__).resolve()), "--profile", PROFILE, *args]


def stages():
    return [("pilot_qwen", MODERN, ["pilot", "--model", "qwen"], RUN / "pilot_qwen_status.json"),
        ("pilot_phi", APP, ["pilot", "--model", "phi"], RUN / "pilot_phi_status.json"),
        ("primary", MODERN, ["run", "--judge", "primary"], RUN / "status.json"),
        ("secondary", APP, ["run", "--judge", "secondary"], RUN / "status.json")]


def pipeline():
    configure(); ensure_not_stopped(); shared.assert_prior_runs_idle()
    lock = (RUN / "pipeline.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pipeline_pid.json", base.identity())
    state = {**base.identity(), "state": "running", "started_at": base.now(), "monitor_seconds": 20,
             "scheduler_sha256": base.file_sha(Path(__file__))}
    def update(**values): utilities.status_update(RUN / "pipeline_status.json", state, **values)
    def stop_signal(signum, frame): raise KeyboardInterrupt()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop_signal)
    child = None
    try:
        update(stage="prepare"); prepare()
        for stage, python, args, status_path in stages():
            if (RUN / "stop.requested").exists(): update(state="stopped"); return
            if stage == "primary":
                if not check_gate(False)["numeric_accepted"]:
                    update(state="stopped_on_validation", stage="admission_failed"); return
                if not engine.wait_for_source_review(update): return
                if not check_gate()["accepted"]: update(state="stopped_on_source_review"); return
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
                if (RUN / "stop.requested").exists(): base.stop_child(child); update(state="stopped", worker_pid=None); return
                if (time.monotonic()-changed > 1200 or time.monotonic()-started > 8*86400 or
                        shutil.disk_usage(ROOT).free < 30*1024**3 or (RUN / "runner.log").stat().st_size > 20*1024**2):
                    base.stop_child(child); update(state="stopped_on_guard", worker_pid=None); return
                time.sleep(20)
            current = base.read(status_path, {})
            update(completed=current.get("completed"), committed_pairs=current.get("committed_pairs"),
                last_commit_at=current.get("last_commit_at"), guard_reason=current.get("guard_reason"))
            if child.returncode or current.get("pid") != child.pid or current.get("state") != "complete":
                update(state="failed", failed_stage=stage, worker_state=current.get("state"), worker_pid=None); return
            child = None
            if stage == "primary":
                update(stage="primary_metrics")
                from evaluate_frontier_automatic import evaluate_run
                evaluate_run(OUT, ASSESS, require_complete=True)
        update(stage="final_evaluation")
        agreement = base.judge_agreement()
        agreement.update(primary_repo=base.read(QWEN_WEIGHTS)["repo"], secondary_repo=base.read(PHI_WEIGHTS)["repo"],
            protocol_sha256=base.file_sha(ASSESS / "judge_protocol.json"), source_review_sha256=verify_source_review()[1],
            interpretation="Fixed Phi sensitivity reference; agreement is not expert gold or proof of source entailment")
        base.dump(ASSESS / "judge_agreement.json", agreement)
        from evaluate_frontier_automatic import evaluate_run
        result = evaluate_run(OUT, ASSESS, require_complete=True)
        update(state="complete", completed_at=base.now(), evaluation_state=result["state"], worker_pid=None)
    except BaseException as error:
        update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__)
        raise
    finally: base.stop_child(child); publish_summary()


def publish_summary():
    folder = ROOT / "experiments/model_recommendation/results_summary"
    link = folder / f"10_推理裁判评分_{PROFILE}"
    if not link.exists(): link.symlink_to(os.path.relpath(ASSESS, folder), target_is_directory=True)
    path = folder / "README.md"; text = path.read_text()
    first, last = f"<!-- reasoned-{PROFILE}:start -->", f"<!-- reasoned-{PROFILE}:end -->"
    state, worker = base.read(RUN / "pipeline_status.json", {}), base.read(RUN / "status.json", {})
    progress = (f"{'主评分' if worker.get('judge') == 'primary' else '5% 复判'}已提交：{worker.get('committed_pairs', 0):,}/"
        f"{worker.get('expected_pairs', 216160):,}。" if worker.get("judge") in {"primary", "secondary"} else "全量评分尚未启动。")
    block = (f"{first}\n## 推理裁判 {PROFILE}\n\n阶段：`{state.get('stage', 'prepare')}`；状态：`{state.get('state', 'preparing')}`。{progress}\n\n"
        "Qwen3.6-27B 先进行有长度上限的原生推理，再生成有引用约束的判断；Phi-4 作为不增加思考阶段的参考。"
        "提示词保持 v5，733条回归和此前未被模型调用的32条控制全部重新生成；四组门槛、来源覆盖与39例来源复核不变。"
        "主裁判任一完整控制组失败即停止，未知不记为负标签；推理和引用语法不构成专家金标准。\n\n"
        f"[状态与结果]({link.name}/)；`runtime/app-env/bin/python scripts/run_frontier_scoring_reasoned.py --profile {PROFILE} status` / `stop` / `start`。\n{last}")
    if first in text:
        if last not in text: raise ValueError("Incomplete summary markers")
        text = text[:text.index(first)]+block+text[text.index(last)+len(last):]
    else: text += "\n"+block+"\n"
    path.write_text(text)


def execute(action, model="qwen", judge="primary", job="all"):
    configure()
    if action in {"start", "pipeline", "pilot", "run"}:
        ensure_not_stopped(); shared.assert_prior_runs_idle()
    if action == "prepare": return prepare()
    if action == "pilot": return pilot(model, job)
    if action == "run": return run_full(judge)
    if action == "pipeline": return pipeline()
    if action == "start": prepare(); return engine.start()
    raise ValueError("Unknown reasoned action")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True)
    p.add_argument("action", choices=("prepare", "start", "pipeline", "pilot", "run", "status", "stop"))
    p.add_argument("--model", choices=("qwen", "phi"), default="qwen")
    p.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    p.add_argument("--job", choices=("all", "legacy-controls", "fresh", "legacy-sources"), default="all")
    args = p.parse_args(); load_profile(args.profile); configure(); engine.initialize()
    if args.action == "status":
        print(json.dumps({"profile": PROFILE, "alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {}),
            "primary_validation": base.read(RUN / "pilot_qwen_status.json", {})}))
    elif args.action == "stop": (RUN / "stop.requested").touch(); print("Stop requested; outputs retained")
    else:
        try:
            result = execute(args.action, args.model, args.judge, args.job)
            if args.action == "start": print(json.dumps(result))
            elif args.action == "prepare": print("Reasoned protocol frozen; new admission is still required")
        except BaseException as error:
            if args.action in {"pilot", "run"}:
                path = RUN / (f"pilot_{args.model}_status.json" if args.action == "pilot" else "status.json")
                utilities.status_update(path, base.read(path, {}), state="failed", error_type=type(error).__name__)
            raise


if __name__ == "__main__": main()
