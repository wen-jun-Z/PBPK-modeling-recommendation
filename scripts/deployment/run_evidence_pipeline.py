"""Project-isolated configuration of the existing unified evidence pipeline.

Run through project_python.py with an explicit target list. The initial extraction
is limited to ten real requests; full extraction must be selected explicitly.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone


def request_concurrency(args, *, server: bool, registered_default: int = 1) -> int:
    if server:
        if args.gpu_batch_size != 1 or args.wait_for_gpu:
            raise ValueError("Server mode uses HTTP request concurrency, not local GPU batching or memory waiting")
        workers = args.request_concurrency if args.request_concurrency is not None else registered_default
        if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 16:
            raise ValueError("Server request concurrency must be between 1 and 16")
        return workers
    if args.request_concurrency is not None:
        raise ValueError("--request-concurrency requires the registered server backend")
    return 1 if args.gpu_batch_size == 1 else 2 * args.gpu_batch_size


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("The acquisition limit must be positive")
    return result


def explicit_run_root(root: Path, value: str | None, *, new_run: bool = False) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    allowed = (root / "artifacts/pbpk_raw_runs/compound_runs").resolve()
    if path.parent != allowed or not path.name.startswith("run_"):
        raise ValueError("An explicit evidence run must be a run_* directory directly under this project's compound_runs")
    if path.exists() and (not path.is_dir() or (new_run and any(path.iterdir()))):
        raise ValueError("--new-run cannot reuse a populated raw run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def evidence_llm_preflight() -> None:
    from pbpk_kb.processing.evidence_llm import (
        require_evidence_llm_enabled, registered_server, server_backend, verify_server_process,
    )
    require_evidence_llm_enabled()
    if server_backend():
        verify_server_process(registered_server())


def wait_for_gpu(pipeline, minimum_free_mib=24576):
    """Wait without loading weights; overwrite one small resource-status file."""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    directory = pipeline.manifest_path.parents[2] / "runtime/deployment/evidence_monitor" / pipeline.run_id
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        result = subprocess.run(["nvidia-smi", "--id=" + device, "--query-gpu=memory.free",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if result.returncode or not result.stdout.strip().isdigit():
            raise RuntimeError("Cannot verify GPU capacity; no model was loaded.")
        free_mib = int(result.stdout.strip())
        state = {"checked_at": datetime.now(timezone.utc).isoformat(), "gpu": device,
                 "free_mib": free_mib, "minimum_free_mib": minimum_free_mib,
                 "status": "ready" if free_mib >= minimum_free_mib else "waiting_for_gpu",
                 "poll_interval_seconds": 60}
        temporary = directory / "gpu_capacity.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "gpu_capacity.json")
        if state["status"] == "ready":
            return
        time.sleep(60)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.environ["PBPK_RAW_OUTPUT_ROOT"] = str(root / "artifacts/pbpk_raw_runs")
    os.environ["PBPK_PROCESSING_OUTPUT_ROOT"] = str(root / "data_process")
    os.environ["PBPK_PROJECT_BOUNDARY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sqlalchemy.engine import make_url
    from pbpk_kb.config.settings import get_settings
    from pbpk_kb.pipeline_utils import PipelineConfig, UnifiedStage1Pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", required=True,
                        help="Comma-separated target drugs; never reuse an implicit historical list.")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--raw-run-root",
                        help="Explicit project compound_runs/run_* directory, new or resumable.")
    parser.add_argument("--defer-llm-check", action="store_true",
                        help="Allow acquisition first; enforce the same backend checks at every LLM stage.")
    parser.add_argument("--max-pmc-fulltext-per-compound", type=positive_int, default=250)
    parser.add_argument("--max-pmc-oa-packages-per-compound", type=positive_int, default=100)
    parser.add_argument("--full-extraction", action="store_true",
                        help="Resume only after reviewing the ten-request smoke batch.")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--gpu-batch-size", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--request-concurrency", type=int, choices=(1, 2, 4, 8, 16),
                        help="Concurrent requests to the registered Qwen server; defaults to its project configuration.")
    parser.add_argument("--wait-for-gpu", action="store_true",
                        help="Wait for 24 GiB free GPU memory before resuming; never inspect or stop other jobs.")
    args = parser.parse_args()
    from pbpk_kb.processing.evidence_llm import backend_identity, registered_server, server_backend
    if not args.plan and not args.defer_llm_check:
        evidence_llm_preflight()
    concurrency = request_concurrency(args, server=server_backend(),
                                      registered_default=registered_server().get("concurrent_requests", 1) if server_backend() else 1)
    llm_model, llm_base_url = backend_identity()
    compounds = tuple(name.strip().lower() for name in args.compounds.split(",") if name.strip())
    os.environ["PBPK_EVIDENCE_LLM_BATCH_SIZE"] = str(args.gpu_batch_size)
    deployment = root / "runtime/deployment"
    lock = (deployment / "evidence_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    database_url = get_settings().database_url
    url = make_url(database_url)
    ports = json.loads((deployment / "ports.json").read_text())
    if url.host != "127.0.0.1" or url.port != ports["postgres"] or url.database != "pbpk_kb":
        raise RuntimeError("Evidence import requires this project's isolated PostgreSQL.")
    pipeline = UnifiedStage1Pipeline(PipelineConfig(
        compounds=compounds,
        raw_run_root=explicit_run_root(root, args.raw_run_root, new_run=args.new_run),
        resume_incomplete=not args.new_run,
        raw_output_root=root / "artifacts/pbpk_raw_runs",
        processing_output_root=root / "data_process",
        source_registry_path=root / "src/pbpk_kb/config/sources.yaml",
        shared_sources=(),
        llm_model=llm_model, llm_base_url=llm_base_url,
        llm_concurrency=concurrency,
        llm_max_retries=1,
        llm_max_requests=None if args.full_extraction else 10,
        llm_preflight=evidence_llm_preflight if args.defer_llm_check else None,
        max_pmc_fulltext_per_compound=args.max_pmc_fulltext_per_compound,
        max_pmc_oa_packages_per_compound=args.max_pmc_oa_packages_per_compound,
        embedding_device="cuda", database_url=database_url,
        replace_database=False,
        metadata={"purpose": "paired_drug_model_evidence_expansion",
                  "existing_model_collection": "model_expansion_20260909",
                  "deferred_llm_preflight": args.defer_llm_check,
                  "max_pmc_fulltext_per_compound": args.max_pmc_fulltext_per_compound,
                  "max_pmc_oa_packages_per_compound": args.max_pmc_oa_packages_per_compound},
    ))
    print(json.dumps({**pipeline.plan(), "llm_concurrency": concurrency}, ensure_ascii=False), flush=True)
    if not args.plan:
        (deployment / "pids/evidence_pipeline.pid").write_text(str(os.getpid()) + "\n")
        if args.wait_for_gpu:
            if not pipeline.manifest_path.is_file():
                raise RuntimeError("GPU waiting requires an existing resumable pipeline manifest.")
            manifest = json.loads(pipeline.manifest_path.read_text())
            manifest["status"] = "waiting_for_gpu"
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = pipeline.manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(pipeline.manifest_path)
            wait_for_gpu(pipeline)
        pipeline.run()


if __name__ == "__main__":
    main()
