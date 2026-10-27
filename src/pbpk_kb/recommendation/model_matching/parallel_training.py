from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json
from pbpk_kb.recommendation.model_matching.merge_parallel_seeds import (
    merge_parallel_seed_runs,
)
from pbpk_kb.recommendation.model_matching.training import select_variants, DEFAULT_ARCHITECTURES


SCHEDULER_VERSION = "pbpk-parallel-training-1.2.0"


@dataclass(frozen=True, slots=True)
class TrainingJob:
    variant: str
    seed: int
    output_directory: Path

    @property
    def job_id(self) -> str:
        return f"{self.variant}:seed_{self.seed}"


def build_jobs(
    *,
    variants: list[str],
    seeds: list[int],
    jobs_root: Path,
    legacy_main_root: Path | None,
    seed_major: bool = False,
) -> list[TrainingJob]:
    pairs = (
        [(variant, seed) for seed in seeds for variant in variants]
        if seed_major
        else [(variant, seed) for variant in variants for seed in seeds]
    )
    return [
        TrainingJob(
            variant=variant,
            seed=seed,
            output_directory=(
                legacy_main_root / f"seed_{seed}"
                if variant == "gmn" and legacy_main_root is not None
                else jobs_root / variant / f"seed_{seed}"
            ),
        )
        for variant, seed in pairs
    ]


def _completed(job: TrainingJob, expected: dict[str, Any] | None = None) -> bool:
    path = job.output_directory / "training_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # Older GMN jobs recorded every requested architecture, including unused
    # alternatives. Compare the actual selected architecture without rewriting
    # their saved manifests, metrics or checkpoint contracts.
    if expected is not None and expected.get("architectures") == ["gmn"] and (
        manifest.get("variants") and all(row.get("architecture") == "gmn" for row in manifest["variants"])
    ):
        manifest = {**manifest, "architectures": ["gmn"]}
    if expected is not None and any(manifest.get(key) != value for key, value in expected.items()):
        return False
    results = (manifest.get("results") or {}).get(job.variant) or []
    if expected is not None and expected.get("disable_early_stopping") and (
        len(results) != 1 or results[0].get("epochs_completed") != expected["epochs"]
    ):
        return False
    return (
        manifest.get("seeds") == [job.seed]
        and [row.get("name") for row in manifest.get("variants") or []]
        == [job.variant]
        and len(results) == 1
    )


def _command(
    *,
    job: TrainingJob,
    benchmark_directory: Path,
    node_embedding_directory: Path,
    epochs: int,
    candidates_per_query: int,
    learning_rate: float,
    architectures: list[str] | None = None,
    supervision_mode: str = "scientific_use",
    full_candidate_training: bool = False,
    disable_early_stopping: bool = False,
    candidate_batch_size: int = 1,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pbpk_kb.recommendation.model_matching.training",
        "--benchmark",
        str(benchmark_directory),
        "--output",
        str(job.output_directory),
        "--variant",
        job.variant,
        "--resume",
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--candidates-per-query",
        str(candidates_per_query),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(job.seed),
        "--node-embeddings",
        str(node_embedding_directory),
        "--supervision-mode",
        supervision_mode,
    ]
    for architecture in architectures or ["gmn"]:
        command.extend(("--architecture", architecture))
    if full_candidate_training:
        command.append("--full-candidate-training")
    command.extend(("--candidate-batch-size", str(candidate_batch_size)))
    if disable_early_stopping:
        command.append("--disable-early-stopping")
    return command


def run_parallel_training(
    *,
    benchmark_directory: Path,
    node_embedding_directory: Path,
    jobs_root: Path,
    merged_output_directory: Path,
    legacy_main_root: Path | None,
    seeds: list[int],
    max_workers: int,
    epochs: int,
    candidates_per_query: int,
    learning_rate: float,
    architectures: list[str] | None = None,
    variant_names: list[str] | None = None,
    supervision_mode: str = "scientific_use",
    full_candidate_training: bool = False,
    disable_early_stopping: bool = False,
    candidate_batch_size: int = 1,
) -> dict[str, Any]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    variants = [
        variant.name for variant in select_variants(architectures or ["gmn"], variant_names)
    ]
    if variant_names is not None:
        variants = list(variant_names)
    jobs = build_jobs(
        variants=variants,
        seeds=seeds,
        jobs_root=jobs_root,
        legacy_main_root=legacy_main_root,
        seed_major=variant_names is not None,
    )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Parallel training jobs are not unique")
    jobs_root.mkdir(parents=True, exist_ok=True)
    logs_root = jobs_root / "logs"
    logs_root.mkdir(exist_ok=True)
    status_path = jobs_root / "scheduler_status.json"
    expected_completion = {
        "epochs": epochs, "candidates_per_query": candidates_per_query,
        "supervision_mode": supervision_mode,
        "full_candidate_training": full_candidate_training,
        "disable_early_stopping": disable_early_stopping,
        "benchmark_directory": str(benchmark_directory),
        "architectures": architectures or ["gmn"],
        "learning_rate": learning_rate,
    }
    completion = {job.job_id: _completed(job, expected_completion) for job in jobs}
    pending = [job for job in jobs if not completion[job.job_id]]
    completed = [job for job in jobs if completion[job.job_id]]
    active: dict[subprocess.Popen[bytes], tuple[TrainingJob, Any]] = {}
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }

    def write_status() -> None:
        _atomic_json(
            status_path,
            {
                "scheduler_version": SCHEDULER_VERSION,
                "total_job_count": len(jobs),
                "architectures": architectures or ["gmn"],
                "variants": variants,
                "job_order": (
                    "seed_then_explicit_variant_order" if variant_names is not None
                    else "historical_variant_then_seed"
                ),
                "supervision_mode": supervision_mode,
                "full_candidate_training": full_candidate_training,
                "candidate_batch_size": candidate_batch_size,
                "disable_early_stopping": disable_early_stopping,
                "epochs": epochs,
                "seeds": seeds,
                "completed_job_count": len(completed),
                "pending_job_count": len(pending),
                "active": [
                    {"job_id": job.job_id, "pid": process.pid}
                    for process, (job, _) in active.items()
                ],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    try:
        while pending or active:
            if stop_requested:
                raise RuntimeError("Parallel training scheduler was interrupted")
            while pending and len(active) < max_workers:
                job = pending.pop(0)
                job.output_directory.mkdir(parents=True, exist_ok=True)
                log_path = logs_root / f"{job.variant}_seed_{job.seed}.log"
                log_handle = log_path.open("ab")
                process = subprocess.Popen(
                    _command(
                        job=job,
                        benchmark_directory=benchmark_directory,
                        node_embedding_directory=node_embedding_directory,
                        epochs=epochs,
                        candidates_per_query=candidates_per_query,
                        learning_rate=learning_rate,
                        architectures=architectures,
                        supervision_mode=supervision_mode,
                        full_candidate_training=full_candidate_training,
                        candidate_batch_size=candidate_batch_size,
                        disable_early_stopping=disable_early_stopping,
                    ),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active[process] = (job, log_handle)
            write_status()
            time.sleep(5)
            for process, (job, log_handle) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                del active[process]
                if return_code != 0 or not _completed(job, expected_completion):
                    raise RuntimeError(
                        f"Parallel training job failed: {job.job_id}, "
                        f"return_code={return_code}"
                    )
                completed.append(job)
        write_status()
    except BaseException:
        for process, (_, log_handle) in active.items():
            process.terminate()
            log_handle.close()
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    result = merge_parallel_seed_runs(
        source_directories=[job.output_directory for job in jobs],
        target_directory=merged_output_directory,
        expected_seeds=seeds,
        expected_variants=variants,
    )
    _atomic_json(
        jobs_root / "scheduler_completion.json",
        {
            "scheduler_version": SCHEDULER_VERSION,
            "job_count": len(jobs),
            "merged_output_directory": str(merged_output_directory.resolve()),
            "merge_version": result["merge_version"],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the selected PBPK architecture/variant/seed grid on GPU."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--node-embeddings", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    parser.add_argument("--legacy-main-root", type=Path)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--candidates-per-query", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--architecture", action="append", choices=DEFAULT_ARCHITECTURES)
    parser.add_argument("--variant", action="append")
    parser.add_argument("--supervision-mode", choices=("scientific_use", "metadata_ranking"), default="scientific_use")
    parser.add_argument("--full-candidate-training", action="store_true")
    parser.add_argument("--disable-early-stopping", action="store_true")
    args = parser.parse_args()
    result = run_parallel_training(
        benchmark_directory=args.benchmark,
        node_embedding_directory=args.node_embeddings,
        jobs_root=args.jobs_root,
        merged_output_directory=args.merged_output,
        legacy_main_root=args.legacy_main_root,
        seeds=args.seed or [17, 29, 43, 59, 71],
        max_workers=args.max_workers,
        epochs=args.epochs,
        candidates_per_query=args.candidates_per_query,
        learning_rate=args.learning_rate,
        architectures=args.architecture,
        variant_names=args.variant,
        supervision_mode=args.supervision_mode,
        full_candidate_training=args.full_candidate_training,
        candidate_batch_size=args.candidate_batch_size,
        disable_early_stopping=args.disable_early_stopping,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
