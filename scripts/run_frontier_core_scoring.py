"""Resume v8 qualification, then score only the frozen 145-request cohort.

Qualification and production use different output directories. A new production
protocol is frozen only after the completed qualification and source review are
bound by their actual hashes. Synthetic controls are development regression.
"""
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

import run_frontier_scoring_citation_complete as qualification

r = qualification.r
base = r.base
ROOT = r.ROOT
OUT = ROOT / "experiments/model_recommendation/frontier_core_145_20260921"
ASSESS = OUT / "assessment"
RUN = ROOT / "runtime/deployment/frontier_core_145"
SCRIPT = Path(__file__).resolve()
FAST, APP = qualification.fast.FAST, r.APP
QUAL_ASSESS = r.OUT / "assessment_citation_complete_v8"
QUAL_RUN = ROOT / "runtime/deployment/frontier_scoring_reviewed_v8"
SEVEN_DAY_MESSAGE = "Assessment run exceeded seven days; resumable checkpoint retained"
_installed = False


def freeze(path, value):
    """Existing scientific inputs and protocols are immutable."""
    if path.exists():
        if base.read(path) != value:
            raise ValueError(f"Frozen core artifact changed: {path.name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value


def assert_project_idle():
    """Ignore only this coordinator, or the verified coordinator parent."""
    coordinator = base.read(RUN / "pipeline_pid.json", {})
    allowed = (base.alive(coordinator) and
               coordinator.get("pid") in {os.getpid(), os.getppid()})
    for directory in (ROOT / "runtime/deployment").glob("frontier_*"):
        if not directory.is_dir() or directory == QUAL_RUN:
            continue
        for name in ("pid.json", "pipeline_pid.json"):
            saved = base.read(directory / name, {})
            if not base.alive(saved):
                continue
            if directory == RUN and name == "pipeline_pid.json" and allowed and saved == coordinator:
                continue
            if directory == RUN and name == "pid.json" and saved.get("pid") == os.getpid():
                continue
            raise RuntimeError("Another project frontier worker or coordinator is active")
    # The qualification directory is exempted by the inherited pilot lock only
    # for this process; an independently launched old coordinator remains banned.
    for name in ("pid.json", "pipeline_pid.json"):
        saved = base.read(QUAL_RUN / name, {})
        if base.alive(saved) and saved.get("pid") != os.getpid():
            raise RuntimeError("An independent qualification worker/coordinator is active")


def initialize():
    global _installed, QUAL_ASSESS, QUAL_RUN
    if not _installed:
        qualification.install()
        r.load_profile("v8")
        QUAL_ASSESS, QUAL_RUN = r.ASSESS, r.RUN
        r.shared.assert_prior_runs_idle = assert_project_idle
        _installed = True
    r.configure()
    r.engine.initialize()
    r.configure()  # initialization replaces the environment; restore offline kernels.
    RUN.mkdir(parents=True, exist_ok=True)
    ASSESS.mkdir(parents=True, exist_ok=True)


def qualification_materials():
    initialize()
    r.ensure_not_stopped()
    return r.prepare()


def link_source(name):
    source, target = QUAL_ASSESS / name, ASSESS / name
    if not source.exists():
        raise ValueError(f"Missing qualified source artifact: {name}")
    if not source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Source outside project")
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError("Core source target changed")
    else:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def verify_artifact_bindings(value):
    """Verify the real ranking files, not merely the manifest naming them."""
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            relative = Path(value["path"])
            path = ROOT / relative
            if (relative.is_absolute() or ".." in relative.parts or
                    not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file() or
                    path.stat().st_size != value.get("bytes") or base.file_sha(path) != value["sha256"]):
                raise ValueError("Frozen selected ranking/source artifact changed")
        for child in value.values():
            verify_artifact_bindings(child)
    elif isinstance(value, list):
        for child in value:
            verify_artifact_bindings(child)


def prepare_plan():
    original_queries, original_candidates, packets, protocol = qualification_materials()
    queries = base.read_rows(OUT / "inputs/queries.jsonl")
    candidates = base.read_rows(OUT / "inputs/candidates.jsonl")
    selection = base.read(OUT / "selection_manifest.json")
    verify_artifact_bindings(selection)
    methods = base.read(OUT / "protocol.json")["methods"]
    if len(methods) != 12 or len(set(methods)) != 12 or set(methods) != set(selection.get("methods", {})):
        raise ValueError("All 12 original method slices are required")
    original_by_id = {q["query_id"]: q for q in original_queries}
    if (len(queries) != 145 or len({q["query_id"] for q in queries}) != 145 or
            candidates != original_candidates or len(candidates) != 280 or
            any(original_by_id.get(q["query_id"]) != q for q in queries) or
            [q["query_id"] for q in queries] != selection["selected_query_ids"]):
        raise ValueError("Exact unchanged 145 requests and original 280 candidates required")
    selected_strata = {(q["compound_id"], q["task_type"]) for q in queries}
    original_strata = {(q["compound_id"], q["task_type"]) for q in original_queries}
    if len(selected_strata) != 145 or selected_strata != original_strata:
        raise ValueError("One request per nonempty compound-task stratum required")
    for name in ("sources.jsonl", "source_supplements", "measured_outcomes.json"):
        link_source(name)
    files = [OUT / "selection_manifest.json", OUT / "protocol.json", OUT / "results.json",
             OUT / "inputs/queries.jsonl", OUT / "inputs/candidates.jsonl", OUT / "inputs/manifest.json",
             ASSESS / "sources.jsonl", ASSESS / "source_supplements/sources.jsonl",
             QUAL_ASSESS / "judge_protocol.json", SCRIPT]
    for method in methods:
        folder = OUT / "results" / method
        if not folder.resolve().is_relative_to(OUT.resolve()):
            raise ValueError("Method artifact outside the core experiment")
        for name in ("scores.npy", "query_seconds.npy", "index.json", "status.json"):
            if not (folder / name).is_file():
                raise ValueError("Missing frozen method score/timing/index/status artifact")
        # Includes retained execution-batch provenance and diagnostic references.
        files.extend(sorted(path for path in folder.iterdir() if path.is_file()))
    if any(not path.resolve().is_relative_to(ROOT.resolve()) for path in files):
        raise ValueError("Frozen artifact outside project")
    plan = {"version": "pbpk-core145-scoring-plan-v1", "queries": 145, "candidates": 280,
        "expected_primary_pairs": 40600,
        "expected_secondary_pairs": sum(base.secondary_selected(q["query_id"], c["candidate_id"])
                                        for q in queries for c in candidates),
        "unique_nonempty_source_requests": len({base.task_key(p, q) for p in packets.values()
                                                if p["refs"] for q in queries}),
        "files": {str(p.relative_to(ROOT)): base.file_sha(p) for p in files},
        "qualification_profile": "v8", "primary_repo": protocol["primary_repo"],
        "secondary_repo": protocol["secondary_repo"],
        "qualification_policy": "Resume existing v8 outputs; both 765-case cohorts, unchanged numeric gates, and actual fixed 39-source audit must pass before production. All 192 synthetic cases are development regression, not independent scientific gold.",
        "production_scope": "Only the frozen 145 requests against all original 280 candidates; reuse existing rankings, regenerate all production judge labels.",
        "resume_policy": "Automatically resume only the exact seven-day checkpoint TimeoutError; no retry on quality, no-progress, disk, provenance or other errors.",
        "scientific_gold_accuracy": None}
    freeze(ASSESS / "core_plan.json", plan)
    return queries, candidates, packets, protocol


def bind_qualification():
    initialize()
    result = r.check_gate()
    if not result.get("accepted"):
        raise RuntimeError("Complete numeric admission and actual source review are required")
    paths = [QUAL_ASSESS / "judge_protocol.json", QUAL_ASSESS / "source_review.json"]
    paths += [QUAL_ASSESS / folder / name for folder in ("validation", "validation/fresh")
              for name in ("manifest.json", "qwen_records.jsonl", "phi_records.jsonl",
                           "qwen_raw_generation_journal.jsonl", "phi_raw_generation_journal.jsonl")]
    value = {"version": "pbpk-core145-qualification-acceptance-v1", "profile": "v8",
        "accepted": True, "numeric_accepted": True, "primary": "qwen", "secondary": "phi",
        "cases_per_model": 765, "source_review_sha256": result["source_review_sha256"],
        "files": {str(p.relative_to(ROOT)): base.file_sha(p) for p in paths},
        "scientific_gold_accuracy": None,
        "interpretation": qualification.DISCLOSURE}
    return freeze(ASSESS / "qualification_acceptance.json", value)


def verify_acceptance():
    value = base.read(ASSESS / "qualification_acceptance.json")
    if (not value or value.get("accepted") is not True or value.get("cases_per_model") != 765 or
            value.get("scientific_gold_accuracy", "absent") is not None):
        raise RuntimeError("Production qualification has not been completed and frozen")
    for relative, digest in value["files"].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or base.file_sha(path) != digest:
            raise ValueError("Completed qualification artifact changed")
    return value


def configure_core():
    r.configure()
    base.OUT, base.ASSESS, base.RUN = OUT, ASSESS, RUN
    base.prepare = prepare_core
    # r.configure() also reasserts the frozen v8 explicit-request scope guard.


def prepare_core():
    queries, candidates, packets, inherited = prepare_plan()
    acceptance = verify_acceptance()
    protocol = copy.deepcopy(inherited)
    protocol.update(version="pbpk-core145-qualified-scoring-v1",
        evaluation_cohort="145 frozen compound-task core requests; 280 original candidates",
        qualification_plan_sha256=base.file_sha(ASSESS / "core_plan.json"),
        qualification_acceptance_sha256=base.file_sha(ASSESS / "qualification_acceptance.json"),
        qualification_source_review_sha256=acceptance["source_review_sha256"],
        qualification_judge_protocol_sha256=base.file_sha(QUAL_ASSESS / "judge_protocol.json"),
        selected_query_manifest_sha256=base.file_sha(OUT / "selection_manifest.json"),
        input_sha256={name: base.file_sha(OUT / name) for name in
            ("inputs/queries.jsonl", "inputs/candidates.jsonl", "assessment/sources.jsonl",
             "assessment/source_supplements/sources.jsonl")},
        parent_pilot_outputs_reused=True, full_labels_reused=False,
        pilot_reuse_scope="Resume the same frozen v8 qualification only; pilot rows never become production labels")
    protocol["implementation_sha256"][SCRIPT.name] = base.file_sha(SCRIPT)
    freeze(ASSESS / "judge_protocol.json", protocol)
    freeze(ASSESS / "assessment_manifest.json", {"queries": len(queries), "candidates": len(candidates),
        "expected_pairs": len(queries)*len(candidates),
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl"),
        "distinct_material_packets": len({p["packet_sha256"] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values())})
    configure_core()
    return queries, candidates, packets, protocol


def run_full(judge):
    initialize()
    if Path(sys.prefix).resolve() != (FAST if judge == "primary" else APP).parent.parent.resolve():
        raise RuntimeError("Wrong interpreter for the actual judge model")
    prepare_core()
    base.GPUJudge = r.backend("qwen" if judge == "primary" else "phi")
    base.download_primary = lambda: r.utilities.verify_weights(r.QWEN_WEIGHTS)
    base.run_assessment(SimpleNamespace(judge=judge, memory_gib=36, batch_size=16,
        max_batches=0, secondary_weights=r.PHI_WEIGHTS))


def command(python, *args):
    return [str(python), str(SCRIPT), *args]


def stages():
    return [("pilot_qwen", FAST, ["pilot", "--model", "qwen"], QUAL_RUN / "pilot_qwen_status.json"),
            ("pilot_phi", APP, ["pilot", "--model", "phi"], QUAL_RUN / "pilot_phi_status.json"),
            ("primary", FAST, ["run", "--judge", "primary"], RUN / "status.json"),
            ("secondary", APP, ["run", "--judge", "secondary"], RUN / "status.json")]


def retryable_timeout(status):
    return (status.get("state") == "checkpoint_time_limit" and
            status.get("error_type") == "TimeoutError" and status.get("error_message") == SEVEN_DAY_MESSAGE and
            status.get("protocol_sha256") == base.file_sha(ASSESS / "judge_protocol.json") and
            not status.get("guard_reason"))


def resource_failure():
    if shutil.disk_usage(ROOT).free < 30*1024**3:
        return "FreeDiskBelow30GiB"
    if (RUN / "runner.log").exists() and (RUN / "runner.log").stat().st_size > 20*1024**2:
        return "RunnerLogAbove20MiB"
    return None


def wait_for_review(update):
    result = r.check_gate(False)
    if not result["numeric_accepted"]:
        update(state="stopped_on_validation", stage="admission_failed")
        return False
    update(state="waiting_for_source_review", stage="source_review", worker_pid=None)
    while not (QUAL_ASSESS / "source_review.json").exists():
        if (RUN / "stop.requested").exists():
            update(state="stopped"); return False
        error = resource_failure()
        if error:
            update(state="stopped_on_guard", guard_reason=error); return False
        time.sleep(20)
    bind_qualification()
    prepare_core()
    return True


def pipeline():
    initialize(); assert_project_idle()
    lock = (RUN / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pipeline_pid.json", base.identity())
    state = {**base.identity(), "state": "running", "started_at": base.now(),
             "monitor_seconds": 20, "scheduler_sha256": base.file_sha(SCRIPT)}
    def update(**values):
        r.utilities.status_update(RUN / "pipeline_status.json", state, **values)
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    child = None
    try:
        update(stage="prepare"); prepare_plan()
        for stage, python, args, status_path in stages():
            if (RUN / "stop.requested").exists():
                update(state="stopped"); return
            if stage == "primary" and not wait_for_review(update):
                return
            retries = 0
            while True:
                update(state="running", stage=stage, worker_pid=None, checkpoint_restarts=retries)
                with (RUN / "runner.log").open("a") as log:
                    child = subprocess.Popen(command(python, *args), cwd=ROOT, env=dict(os.environ),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                update(worker_pid=child.pid)
                last, changed = None, time.monotonic()
                while child.poll() is None:
                    current = base.read(status_path, {})
                    if current.get("pid") != child.pid:
                        current = {"state": "starting"}
                    signature = tuple(current.get(k) for k in ("state", "completed", "committed_pairs", "last_commit_at"))
                    if signature != last:
                        last, changed = signature, time.monotonic()
                        update(worker_state=current.get("state"), completed=current.get("completed"),
                            committed_pairs=current.get("committed_pairs"), last_commit_at=current.get("last_commit_at"))
                    if (RUN / "stop.requested").exists():
                        base.stop_child(child); update(state="stopped", worker_pid=None); return
                    error = resource_failure()
                    if error or time.monotonic()-changed > 1200:
                        base.stop_child(child)
                        update(state="stopped_on_guard", guard_reason=error or "NoProgressFor20Minutes", worker_pid=None)
                        return
                    time.sleep(20)
                current = base.read(status_path, {})
                update(completed=current.get("completed"), committed_pairs=current.get("committed_pairs"),
                    last_commit_at=current.get("last_commit_at"), guard_reason=current.get("guard_reason"))
                matching = current.get("pid") == child.pid
                if stage in {"primary", "secondary"} and matching and retryable_timeout(current):
                    retries += 1
                    base.append_rows(RUN / "checkpoint_resumptions.jsonl", [{"at": base.now(), "stage": stage,
                        "previous_pid": child.pid, "protocol_sha256": current["protocol_sha256"],
                        "reason": SEVEN_DAY_MESSAGE, "committed_pairs": current.get("committed_pairs")}])
                    child = None
                    continue
                if child.returncode or not matching or current.get("state") != "complete":
                    update(state="failed", failed_stage=stage, worker_state=current.get("state"), worker_pid=None)
                    return
                child = None
                break
        update(stage="final_evaluation", worker_pid=None)
        configure_core()
        agreement = base.judge_agreement()
        agreement.update(primary_repo="Qwen/Qwen3.6-27B", secondary_repo="microsoft/phi-4",
            protocol_sha256=base.file_sha(ASSESS / "judge_protocol.json"),
            source_review_sha256=verify_acceptance()["source_review_sha256"],
            interpretation="Fixed Phi sensitivity reference; agreement is not scientific gold",
            limitations=["Agreement between automatic judges is not correctness against expert ground truth",
                "The Qwen primary family overlaps several evaluated rankers; report the Phi sensitivity comparison",
                "Unknowns are excluded from agreement and denominators are reported",
                "Shared documents and pretrained knowledge can induce correlated errors"])
        base.dump(ASSESS / "judge_agreement.json", agreement)
        from evaluate_frontier_automatic import evaluate_run
        result = evaluate_run(OUT, ASSESS, require_complete=True)
        update(state="complete", completed_at=base.now(), evaluation_state=result["state"])
    except BaseException as error:
        update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
               error_type=type(error).__name__, error_message=str(error)[:300])
        raise
    finally:
        base.stop_child(child)


def start():
    initialize(); assert_project_idle()
    previous = base.read(RUN / "pipeline_status.json", {})
    worker = base.read(RUN / "status.json", {})
    if (previous.get("state") in {"stopped_on_validation", "stopped_on_source_review"} or
            worker.get("guard_reason") == "PrimaryInvalidFractionAbove10Percent"):
        raise RuntimeError("A failed quality gate requires a reviewed correction, not an identical restart")
    prepare_plan()
    cleared = []
    for directory in (RUN, QUAL_RUN):
        marker = directory / "stop.requested"
        if marker.exists():
            marker.unlink(); cleared.append(str(marker.relative_to(ROOT)))
    base.append_rows(RUN / "user_starts.jsonl", [{"at": base.now(), "cleared_stop_markers": cleared,
                                                 "core_plan_sha256": base.file_sha(ASSESS / "core_plan.json")}])
    with (RUN / "runner.log").open("a") as log:
        child = subprocess.Popen(command(APP, "pipeline"), cwd=ROOT, env=dict(os.environ),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return {"started_pid": child.pid, "status": str(RUN / "pipeline_status.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "pipeline", "pilot", "run", "status", "stop", "finalize"))
    parser.add_argument("--model", choices=("qwen", "phi"), default="qwen")
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    args = parser.parse_args()
    initialize()
    if args.action == "stop":
        (RUN / "stop.requested").touch()
        print("Core scoring stop requested; outputs retained")
        return
    if args.action == "status":
        print(json.dumps({"alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
            "pipeline": base.read(RUN / "pipeline_status.json", {}), "worker": base.read(RUN / "status.json", {}),
            "qualification_qwen": base.read(QUAL_RUN / "pilot_qwen_status.json", {}),
            "qualification_phi": base.read(QUAL_RUN / "pilot_phi_status.json", {})}))
        return
    try:
        if args.action == "prepare":
            prepare_plan(); print("145-request plan frozen; qualification and real source review remain required")
        elif args.action == "start":
            print(json.dumps(start()))
        elif args.action == "pipeline":
            pipeline()
        elif args.action == "pilot":
            if Path(sys.prefix).resolve() != (FAST if args.model == "qwen" else APP).parent.parent.resolve():
                raise RuntimeError("Wrong qualification interpreter")
            r.pilot(args.model)
        elif args.action == "run":
            run_full(args.judge)
        elif args.action == "finalize":
            bind_qualification(); prepare_core(); print("Qualified 145-request production protocol frozen")
    except BaseException as error:
        if args.action in {"pilot", "run"}:
            path = (QUAL_RUN / f"pilot_{args.model}_status.json" if args.action == "pilot" else RUN / "status.json")
            checkpoint = (args.action == "run" and isinstance(error, TimeoutError) and str(error) == SEVEN_DAY_MESSAGE)
            r.utilities.status_update(path, base.read(path, {}),
                state="checkpoint_time_limit" if checkpoint else "failed",
                error_type=type(error).__name__, error_message=str(error)[:300])
        raise


if __name__ == "__main__":
    main()
