"""Freeze auditable offline Qwen candidate proposals; never fit or score PBPK."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get('PBPK_BLUEPRINT_SOURCE_ROOT', str(ROOT / 'src'))).resolve(strict=True)
if SOURCE_ROOT.name != 'src' or not SOURCE_ROOT.is_relative_to(ROOT):
    raise ValueError('Qwen source override must be a project-local src directory')
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pbpk_kb.blueprints.qwen_proposals import (
    VERSION, REPO, REVISION, SYSTEM, GENERATION, LocalQwenProposer, build_request,
    canonical, digest, file_sha, messages, parse_response, project_path,
    read_frozen_record, validate_request, verify_weights,
)


def save(path, value, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)


def load_requests(args):
    requests = []
    if args.requests:
        value = json.loads(project_path(args.requests, ROOT).read_text())
        requests.extend(value["requests"] if isinstance(value, dict) else value)
    files = list(args.case_file)
    if args.cohort_manifest:
        cohort = json.loads(project_path(args.cohort_manifest, ROOT).read_text())
        files.extend(c["bundle_path"] for c in cohort["cases"] if c.get("bundle_path"))
    if files:
        from pbpk_kb.blueprints.experiment_runtime import candidate_space
        for file in files:
            payload = json.loads(project_path(file, ROOT).read_text())
            case = payload.get("case", payload)
            requests.append(build_request(case, candidate_space(case, root=ROOT)))
    if not requests or len({r["case_id"] for r in requests}) != len(requests):
        raise ValueError("Expected nonempty unique design requests")
    for request in requests:
        validate_request(request)
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", help="Sanitized proposal-input JSON; preferred interface")
    parser.add_argument("--case-file", "--case-bundle", action="append", default=[])
    parser.add_argument("--cohort-manifest")
    parser.add_argument("--seeds", default="17,43,101")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-manifest", default="experiments/model_recommendation/frontier_comparison_20260914/assessment_qwen36/model_weights/qwen.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-gib", type=int, default=36)
    parser.add_argument("--reuse-proposals", action="append", default=[],
                        help="Reuse only exactly bound frozen records; failures are preserved, never retried")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 16 or not 24 <= args.memory_gib <= 36:
        parser.error("Batch must be 1..16 and GPU allowance 24..36 GiB")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**31 for seed in seeds):
        parser.error("Seeds must be unique nonnegative 31-bit integers")
    requests = load_requests(args)
    output = project_path(args.output, ROOT, must_exist=False)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {"state": "verifying_weights", "pid": os.getpid(), "completed": 0,
              "total": len(requests) * len(seeds), "started_at": datetime.now(timezone.utc).isoformat()}
    def update(**values):
        status.update(values, elapsed_seconds=time.monotonic()-start)
        save(output / "progress.json", status, replace=True)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    update()
    save(output / "requests.json", {"requests": requests})
    # Freeze the source that generates proposals before GPU loading. A later
    # integration fix to the CPU reader cannot erase the executed version.
    source_files = [Path(__file__), SOURCE_ROOT / "pbpk_kb/blueprints/qwen_proposals.py",
                    ROOT / "scripts/frontier_modern_judge.py",
                    ROOT / "scripts/frontier_reasoned_fast_judge.py",
                    ROOT / "scripts/frontier_reasoned_judge.py",
                    ROOT / "scripts/frontier_judge_decoding.py"]
    with zipfile.ZipFile(output / "executed_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, str(path.relative_to(ROOT)))
    cache = output / "cache"
    cache.mkdir()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_HUB_KERNELS="0",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TRITON_CACHE_DIR=str(cache / "triton"),
                      TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"), TMPDIR=str(cache))
    manifest = {"version": VERSION, "model_repo": REPO, "model_revision": REVISION,
                "generation": GENERATION, "system_prompt": SYSTEM,
                "requests": {r["case_id"]: digest(r) for r in requests},
                "seeds": seeds, "record_sha256": {}, "program_sha256": file_sha(Path(__file__)),
                "module_sha256": file_sha(SOURCE_ROOT / "pbpk_kb/blueprints/qwen_proposals.py"),
                "module_source_root": str(SOURCE_ROOT.relative_to(ROOT)),
                "executed_sources_sha256": file_sha(output / "executed_sources.zip"),
                "input_scope": "intended_use_registered_candidates_source_parameter_contracts_only",
                "external_outcomes_in_prompt": False, "selection_outcomes_in_prompt": False,
                "final_test_outcomes_in_prompt": False,
                "proposal_is_scientific_validation": False,
                "model_loaded_times": 0, "python_environment": str(Path(sys.prefix).relative_to(ROOT)),
                "proposal_policy": "exactly_once_per_case_seed_no_repair_no_rules_fallback",
                "batch_rng_policy": "seed_reset_per_frozen_batch_order_not_independent_per_row",
                "record_origins": {}}
    try:
        weights, manifest["weights_manifest_sha256"] = verify_weights(args.weights_manifest, project_root=ROOT)
        manifest["all_weight_checksums_verified"] = True
        save(output / "manifest.json", manifest)
        reused = set()
        failures = 0
        for previous in args.reuse_proposals:
            previous = project_path(previous, ROOT)
            previous_manifest = json.loads((previous / "manifest.json").read_text())
            for request in requests:
                if request["case_id"] not in previous_manifest.get("requests", {}):
                    continue
                for seed in seeds:
                    key = (request["case_id"], seed)
                    if key in reused:
                        raise ValueError("Overlapping frozen proposal sources")
                    record, origin = read_frozen_record(previous, request, seed=seed, project_root=ROOT)
                    if record["weights_manifest_sha256"] != manifest["weights_manifest_sha256"]:
                        raise ValueError("Reused proposals use different weight manifest")
                    relative = f"cases/{request['case_id']}/seed_{seed}.json"
                    save(output / relative, record)
                    manifest["record_sha256"][relative] = file_sha(output / relative)
                    manifest["record_origins"][relative] = origin
                    failures += record["status"] != "completed"
                    reused.add(key)
                    update(completed=status["completed"] + 1, failures=failures, reused_records=len(reused))
        save(output / "manifest.json", manifest, replace=True)
        if len(reused) == status["total"]:
            update(state="completed" if failures == 0 else "completed_with_failed_proposals", active_cases=[])
            manifest["final_status"] = status["state"]
            save(output / "manifest.json", manifest, replace=True)
            return 0 if failures == 0 else 2
        update(state="loading_model")
        model = LocalQwenProposer(weights, memory_gib=args.memory_gib)
        manifest["model_loaded_times"] = 1
        manifest["backend_metadata"] = model.metadata
        save(output / "manifest.json", manifest, replace=True)
        for seed in seeds:
            pending = [r for r in requests if (r["case_id"], seed) not in reused]
            while pending:
                size = model.batch_size(pending, maximum=args.batch_size)
                batch, pending = pending[:size], pending[size:]
                update(state="generating", seed=seed, active_cases=[r["case_id"] for r in batch])
                try:
                    values, metrics = model.generate(batch, seed=seed)
                    error = None
                    if len(values) != len(batch):
                        raise ValueError("Model response count changed")
                except Exception as exc:
                    values, metrics, error = [None] * len(batch), {}, type(exc).__name__
                for request, raw in zip(batch, values, strict=True):
                    record = {"version": VERSION, "case_id": request["case_id"], "seed": seed,
                              "request_sha256": digest(request), "prompt_sha256": digest(messages(request)),
                              "weights_manifest_sha256": manifest["weights_manifest_sha256"],
                              "raw_response": raw, "metrics": metrics, "candidate_id": None,
                              "status": "generation_failed" if error else "invalid_response", "error_type": error}
                    if raw is not None:
                        try:
                            answer = parse_response(raw, request)
                            if metrics.get("time_limit_reached"):
                                record.update(status="timeout", error_type="GenerationTimeout")
                            else:
                                record.update(answer, status="completed", error_type=None)
                        except (ValueError, TypeError, KeyError) as exc:
                            record.update(error_type=type(exc).__name__, parse_error=str(exc)[:300])
                    failures += record["status"] != "completed"
                    relative = f"cases/{request['case_id']}/seed_{seed}.json"
                    path = output / relative
                    save(path, record)
                    manifest["record_sha256"][relative] = file_sha(path)
                    save(output / "manifest.json", manifest, replace=True)
                    update(completed=status["completed"] + 1, failures=failures)
                if error:
                    raise RuntimeError("GPU generation failed; batch records preserved, no automatic retry")
        update(state="completed" if failures == 0 else "completed_with_failed_proposals", active_cases=[])
        manifest["final_status"] = status["state"]
        save(output / "manifest.json", manifest, replace=True)
        return 0 if failures == 0 else 2
    except BaseException as exc:
        update(state="failed", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
