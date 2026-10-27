"""Run the expanded PBPK ranking experiment, with bounded project-owned jobs."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/model_recommendation/runs/expanded_catalog_20260913"
DEPLOY = ROOT / "runtime/deployment/full_model_recommendation"
SEEDS = (17, 29, 43, 59, 71)
VARIANTS = ("gmn",)
DEFAULT_ACTIVE_VARIANTS = VARIANTS
MODEL = ROOT / "runtime/hf_cache/embedding/models--NeuML--pubmedbert-base-embeddings/snapshots/b79526d6ef3645e0df4530322e266f24c829f5ef"
STOP_REQUESTED = False


def now():
    return datetime.now(UTC).isoformat()


def scoped(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Experiment paths must belong to this project")
    return path


def read(path):
    return json.loads(path.read_text())


def require_active_evaluation(output):
    policy = output / "evaluation_policy.json"
    if policy.is_file() and read(policy).get("allow_training") is False:
        raise RuntimeError(
            "This evaluation is retired: matching-rule labels cannot measure scientific "
            "PBPK model recommendation quality. See docs/研究计划.md."
        )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owned_process(record):
    """Only inspect the exact PID recorded by this project's own launcher."""
    try:
        proc = Path("/proc") / str(int(record["pid"]))
        if proc.stat().st_uid != os.getuid():
            return False
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(record["start_ticks"])
    except (OSError, ValueError, KeyError):
        return False


def resource_check(output):
    size = 0
    for parent, directories, files in os.walk(output, followlinks=False):
        directories[:] = [name for name in directories if not (Path(parent) / name).is_symlink()]
        for name in files:
            path = Path(parent) / name
            if path.is_symlink():
                continue
            try:
                length = path.stat().st_size
            except FileNotFoundError:
                continue
            size += length
            if path.suffix == ".log" and length > 128 * 2**20:
                raise RuntimeError("Own experiment log exceeded 128 MiB; stopped to prevent repeated writes")
    free = shutil.disk_usage(output).free
    if size > 30 * 2**30 or free < 5 * 2**30:
        raise RuntimeError("Experiment disk guard reached; only this experiment is stopped")
    return {"output_bytes": size, "free_disk_bytes": free, "checked_at": now()}


def stop_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_stage(name, arguments, output, state):
    if STOP_REQUESTED:
        raise InterruptedError("Experiment stop requested")
    log = output / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    state.update(status="running", stage=name, updated_at=now())
    write(output / "status.json", state)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    start = time.monotonic()
    stage_started_wall_time = time.time()
    with log.open("ab") as stream:
        stream.write(("\nStage started " + now() + "\n").encode())
        stream.flush()
        process = subprocess.Popen([sys.executable, *arguments], cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state["child_pid"] = process.pid
        write(output / "status.json", state)
        try:
            while process.poll() is None:
                if STOP_REQUESTED:
                    raise InterruptedError("Experiment stop requested")
                state["health"] = resource_check(output)
                state["updated_at"] = now()
                write(output / "status.json", state)
                activity_paths = [log, *(output / "jobs").rglob("*_training_status.json")]
                latest_activity = max([stage_started_wall_time, *(
                    path.stat().st_mtime for path in activity_paths if path.exists()
                )])
                if time.time() - latest_activity > 6 * 3600:
                    raise TimeoutError(f"Stage has no child progress for six hours: {name}")
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"Stage {name} exited {process.returncode}; inspect {log.relative_to(ROOT)}")
        except BaseException:
            stop_child(process)
            raise
    state.pop("child_pid", None)
    state.setdefault("completed_stages", {})[name] = {"completed_at": now(), "elapsed_seconds": time.monotonic() - start}
    write(output / "status.json", state)


def baselines(output):
    import torch
    from pbpk_kb.recommendation.model_matching.training import _load_benchmark
    from pbpk_kb.recommendation.model_matching.feature_store import load_embedding_store
    from pbpk_kb.recommendation.model_matching.metrics import evaluate_metadata_rankings
    from pbpk_kb.recommendation.model_matching.scoring import bm25_scores, score_candidate
    from pbpk_kb.recommendation.model_matching.benchmark import _atomic_jsonl

    queries, qgraphs, mgraphs, gates, labels = _load_benchmark(output / "benchmark")
    vectors, _ = load_embedding_store(output / "features")
    ids = sorted(mgraphs)
    def pooled(graph):
        rows = [vectors[node.semantic_key] for node in graph.nodes]
        value = torch.tensor(rows, dtype=torch.float32, device="cuda").mean(dim=0)
        return torch.nn.functional.normalize(value, dim=0)
    model_vectors = torch.stack([pooled(mgraphs[key]) for key in ids])
    target = output / "baselines"
    target.mkdir(exist_ok=True)
    for method in ("bm25", "uniform_rule", "pubmedbert_cosine"):
        metric_path = target / (method + "_metrics.json")
        if metric_path.is_file():
            continue
        predictions = []
        for query in queries:
            if query["split"] != "test":
                continue
            qid = query["query_id"]
            if method == "bm25":
                scores = bm25_scores(qgraphs[qid], [mgraphs[key] for key in ids])
            elif method == "pubmedbert_cosine":
                scores = (model_vectors @ pooled(qgraphs[qid])).detach().cpu().tolist()
            else:
                scores = []
                for key in ids:
                    match = score_candidate(qgraphs[qid], gates[key])
                    scores.append(statistics.mean(match.hierarchy_scores.get(level, 0.0)
                                                  for level in ("identity", "context", "mechanism", "structure")))
            for key, score in zip(ids, scores, strict=True):
                predictions.append({"query_id": qid, "candidate_id": key, "method": method,
                                    "score": float(score), "relevance_grade": labels[qid, key],
                                    "relevance_label_semantics": "metadata_compatibility_not_validated_use"})
        _atomic_jsonl(target / (method + "_predictions.jsonl"), predictions)
        write(metric_path, evaluate_metadata_rankings(predictions))
        print(json.dumps({"method": method, "evaluated_pairs": len(predictions)}), flush=True)


def summarize(output, active_variants=DEFAULT_ACTIVE_VARIANTS):
    def metadata_result(path):
        value = read(path)
        if (value.get("relevance_label_semantics") != "metadata_compatibility_not_validated_use"
                or value.get("scientific_claim_eligible") is not False):
            raise ValueError(f"Refusing to mix scientific use metrics into compatibility summary: {path.name}")
        return value
    rows = []
    for path in sorted((output / "baselines").glob("*_metrics.json")):
        rows.append({"method": path.stem.removesuffix("_metrics"), "seed": "deterministic", **metadata_result(path)["aggregate"]})
    for seed in SEEDS:
        manifest = metadata_result(output / "lambdamart" / f"seed_{seed}" / "manifest.json")
        rows.append({"method": "lambdamart", "seed": seed, **manifest["test_metrics"]})
    missing_neural_jobs = []
    for variant in VARIANTS:
        for seed in SEEDS:
            directory = output / "jobs" / variant / f"seed_{seed}"
            manifest_path = directory / "training_manifest.json"
            if not manifest_path.is_file():
                if variant in active_variants:
                    raise ValueError(f"Required neural seed is incomplete: {variant}, seed {seed}")
                missing_neural_jobs.append({"method": variant, "seed": seed, "reason": "stopped_by_user"})
                continue
            manifest = metadata_result(manifest_path)
            seeds = manifest.get("results", {}).get(variant, [])
            if len(seeds) != 1 or seeds[0].get("seed") != seed or seeds[0].get("epochs_completed") != 40:
                raise ValueError(f"Invalid completed neural seed: {variant}, seed {seed}")
            metrics = metadata_result(directory / variant / f"seed_{seed}_metrics.json")
            rows.append({"method": variant, "seed": seed, **metrics["aggregate"]})
    numeric_keys = sorted({key for row in rows for key, value in row.items()
                           if key not in {"method", "seed"} and isinstance(value, (int, float)) and not isinstance(value, bool)})
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        samples = [row for row in rows if row["method"] == method]
        entry = {"method": method, "runs": len(samples)}
        entry["planned_runs"] = len(SEEDS) if method in (*VARIANTS, "lambdamart") else 1
        entry["all_planned_runs_complete"] = len(samples) == entry["planned_runs"]
        entry["remaining_runs_stopped_by_user"] = method in VARIANTS and method not in active_variants and not entry["all_planned_runs_complete"]
        for key in numeric_keys:
            values = [float(row[key]) for row in samples if isinstance(row.get(key), (int, float))]
            if values:
                entry[key + "_mean"] = statistics.mean(values)
                entry[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(entry)
    fields = sorted({key for row in summaries for key in row if key not in {"method", "runs"}})
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "runs", *fields])
        writer.writeheader()
        writer.writerows(summaries)
    write(output / "results.json", {"completed_at": now(), "scope": "full_catalog_metadata_ranking",
        "scientific_claim_eligible": False, "limitations": [
            "Automatic evidence/model compatibility labels measure documented relevance, not simulation accuracy.",
            "Rule-generated labels evaluate weak-supervision fit and grouped generalization, not an independent gold standard of recommendation quality.",
            "UniformRule is a comparison scorer, not the labeling rule. The labeling rule's agreement with its own labels is tautological and not an independent baseline win.",
            "Connected compound/model groups remain disjoint; evidence compounds are imbalanced across groups.",
            "All five GMN seeds and completed baselines are reported; checkpoints use validation only.",
            "The current test has only two compounds and no query with two independent same-compound model families; perfect rule-label ranking does not establish model-selection quality.",
            "Physical PBPK simulation assets remain separate; compatibility labels do not establish execution validity."],
        "benchmark": read(output / "benchmark" / "experiment_manifest.json"),
        "benchmark_verification": (read(output / "benchmark_verification.json")
                                   if (output / "benchmark_verification.json").is_file() else None),
        "source_code_sha256": read(output / "status.json").get("source_code_sha256"),
        "execution_history": read(output / "status.json").get("execution_history", []),
        "active_neural_variants": list(active_variants),
        "original_neural_variants": list(VARIANTS),
        "original_neural_grid_complete": not missing_neural_jobs,
        "missing_neural_jobs": missing_neural_jobs,
        "evaluation_audit": (read(output / "lambdamart_audit.json") if (output / "lambdamart_audit.json").is_file() else None),
        "evaluation_validity": "compatibility_rule_fit_not_independent_model_quality",
        "seed_results": rows, "methods": summaries})


def run(output, workers, candidate_batch_size=128, active_variants=DEFAULT_ACTIVE_VARIANTS):
    global STOP_REQUESTED
    require_active_evaluation(output)
    from pbpk_kb.recommendation.model_matching.feature_store import DEFAULT_BIOMEDICAL_MODEL
    output.mkdir(parents=True, exist_ok=True)
    DEPLOY.mkdir(parents=True, exist_ok=True)
    lock = (DEPLOY / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    write(DEPLOY / "pid.json", {"pid": os.getpid(), "start_ticks": fields[19], "output": str(output), "created_at": now()})
    def request_stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state = read(output / "status.json") if (output / "status.json").exists() else {}
    if state.get("source_code_sha256"):
        state.setdefault("execution_history", []).append({
            "previous_pid": state.get("pid"), "last_updated_at": state.get("updated_at"),
            "configuration": state.get("configuration"),
            "source_code_sha256": state["source_code_sha256"],
        })
    if "error" in state:
        state["previous_run_error"] = {"error": state.pop("error"), "error_type": state.pop("error_type", None)}
    state.pop("completed_at", None)
    state.pop("excluded_neural_jobs", None)
    state.update(status="running", pid=os.getpid(), updated_at=now(),
                 configuration={"models_registered": 280, "seeds": SEEDS, "neural_variants": active_variants,
                    "original_neural_variants": VARIANTS,
                    "scope_note": "GMN and retrieval/rule baselines; compatibility-rule evaluation only.",
                    "epochs": 40, "all_candidates_within_audited_split": True, "early_stopping": False,
                    "device": "cuda", "max_workers": workers, "supervision_mode": "metadata_ranking",
                    "candidate_batch_size": candidate_batch_size})
    try:
        benchmark = output / "benchmark"
        if not (benchmark / "experiment_manifest.json").is_file():
            run_stage("dataset", ["-m", "pbpk_kb.recommendation.model_matching.expanded_benchmark", "--output", str(benchmark)], output, state)
        bm = read(benchmark / "experiment_manifest.json")
        if bm.get("supervision_mode") != "metadata_ranking" or bm.get("scientific_claim_eligible") is not False:
            raise ValueError("Expanded benchmark must preserve its metadata-only supervision contract")
        for name, expected in bm["artifacts"].items():
            if fingerprint(scoped(benchmark / name)) != expected:
                raise ValueError(f"Frozen benchmark artifact changed: {name}")
        if ((output / "benchmark_verification.json").is_file()
                and not read(output / "benchmark_verification.json").get("passed")):
            raise ValueError("The full dataset verification has not passed")
        state["source_code_sha256"] = {
            str(path.relative_to(ROOT)): fingerprint(path)
            for path in [Path(__file__).resolve(), *(ROOT / "src/pbpk_kb/recommendation/model_matching").glob("*.py")]
        }
        features = output / "features"
        graphs = [benchmark / "query_graphs.jsonl", benchmark / "model_graphs.jsonl"]
        if not (features / "manifest.json").is_file():
            arguments = ["-m", "pbpk_kb.recommendation.model_matching.feature_store", "--output", str(features),
                         "--model", DEFAULT_BIOMEDICAL_MODEL, "--model-path", str(MODEL), "--device", "cuda", "--batch-size", "64"]
            for graph in graphs:
                arguments.extend(["--graphs", str(graph)])
            run_stage("features", arguments, output, state)
        fm = read(features / "manifest.json")
        if fingerprint(features / "node_embeddings.jsonl") != fm["node_embeddings_sha256"]:
            raise ValueError("Feature checksum differs from its frozen manifest")
        for graph in graphs:
            if fm["graph_input_sha256"].get(str(graph.resolve())) != fingerprint(graph):
                raise ValueError("Graphs changed after feature generation")
        run_stage("baselines", [str(Path(__file__).resolve()), "baselines", "--output", str(output)], output, state)
        for seed in SEEDS:
            destination = output / "lambdamart" / f"seed_{seed}"
            if not (destination / "manifest.json").is_file():
                run_stage(f"lambdamart_{seed}", ["-m", "pbpk_kb.recommendation.model_matching.learning_to_rank",
                    "--benchmark", str(benchmark), "--output", str(destination), "--seed", str(seed), "--device", "cuda"], output, state)
        arguments = ["-m", "pbpk_kb.recommendation.model_matching.parallel_training", "--benchmark", str(benchmark),
            "--node-embeddings", str(features), "--jobs-root", str(output / "jobs"), "--merged-output", str(output / "neural"),
            "--architecture", "gmn", "--max-workers", str(workers), "--epochs", "40",
            "--candidate-batch-size", str(candidate_batch_size),
            "--supervision-mode", "metadata_ranking", "--full-candidate-training", "--disable-early-stopping"]
        for variant in active_variants:
            arguments.extend(["--variant", variant])
        for seed in SEEDS:
            arguments.extend(["--seed", str(seed)])
        run_stage("neural_training", arguments, output, state)
        summarize(output, active_variants)
        state.update(status="completed", stage="completed", completed_at=now(),
                     completion_scope="selected_neural_variants_and_all_baselines",
                     original_neural_grid_complete=read(output / "results.json")["original_neural_grid_complete"])
        write(output / "status.json", state)
    except BaseException as error:
        state.update(status="stopped" if STOP_REQUESTED else "failed", updated_at=now(),
                     error_type=type(error).__name__, error=str(error))
        state.pop("child_pid", None)
        write(output / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "status", "stop", "baselines"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--candidate-batch-size", type=int, choices=(1, 16, 32, 64, 128, 256), default=128)
    parser.add_argument("--variant", action="append", choices=VARIANTS,
                        help="Neural method to run; currently GMN.")
    args = parser.parse_args()
    active_variants = tuple(dict.fromkeys(args.variant or DEFAULT_ACTIVE_VARIANTS))
    output = scoped(args.output)
    if args.action in {"start", "run", "baselines"}:
        require_active_evaluation(output)
    record = read(DEPLOY / "pid.json") if (DEPLOY / "pid.json").exists() else {}
    if args.action == "start":
        if record and owned_process(record):
            print(json.dumps({"status": "already_running", **record}))
            return
        DEPLOY.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--output", str(output),
                   "--max-workers", str(args.max_workers), "--candidate-batch-size", str(args.candidate_batch_size)]
        for variant in active_variants:
            command.extend(("--variant", variant))
        with (DEPLOY / "runner.log").open("ab") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"status": "launched", "pid": process.pid, "output": str(output)}))
    elif args.action == "status":
        state = read(output / "status.json") if (output / "status.json").exists() else {"status": "not_started"}
        state.pop("source_code_sha256", None)
        state.pop("execution_history", None)
        if state.get("status") == "running" and "error" in state:
            state["previous_run_error"] = {"error": state.pop("error"), "error_type": state.pop("error_type", None)}
        scheduler = output / "jobs/scheduler_status.json"
        if scheduler.exists():
            progress = read(scheduler)
            state["neural_jobs"] = {key: progress.get(key) for key in (
                "total_job_count", "completed_job_count", "pending_job_count", "active", "updated_at"
            )}
            state["neural_progress"] = [read(path) for path in sorted((output / "jobs").rglob("*_training_status.json"))]
        print(json.dumps({**state, "process_alive": bool(record and owned_process(record))}, ensure_ascii=False, indent=2))
    elif args.action == "stop":
        if record and owned_process(record):
            os.kill(int(record["pid"]), signal.SIGTERM)
            print(json.dumps({"status": "stop_requested", "pid": record["pid"]}))
        else:
            print(json.dumps({"status": "not_running"}))
    elif args.action == "baselines":
        baselines(output)
    else:
        run(output, args.max_workers, args.candidate_batch_size, active_variants)


if __name__ == "__main__":
    main()
