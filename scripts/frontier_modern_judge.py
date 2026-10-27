"""Project-isolated, pinned Qwen3.6 document judge and bounded weight download.

Only the text path is exercised; non-thinking greedy structured decoding is an
explicit experimental choice, not a reproduction of Qwen's thinking benchmarks.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_qwen36"
RUN = ROOT / "runtime/deployment/frontier_qwen36_backend"
CACHE = ROOT / "runtime/hf_cache/frontier/hub"
REPO = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
WEIGHTS = OUT / "model_weights/qwen.json"
GiB = 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    from run_frontier_comparison import clean_environment
    result = clean_environment()
    result.update(PATH=str(ROOT / "runtime/judge-env/bin") + ":/usr/bin:/bin",
                  TMPDIR=str(RUN), PIP_CONFIG_FILE="/dev/null",
                  PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1",
                  USE_HUB_KERNELS="0",
                  TRITON_CACHE_DIR=str(RUN / "triton_cache"),
                  TORCHINDUCTOR_CACHE_DIR=str(RUN / "torchinductor_cache"))
    return result


def init():
    RUN.mkdir(parents=True, exist_ok=True)
    clean = environment()
    os.environ.clear()
    os.environ.update(clean)
    for name in ("huggingface_hub", "urllib3", "requests", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def prepare_json_constraint(tokenizer, eos_token_ids):
    """Keep LMFE 0.11.3 compatible with the public Transformers 5 tokenizer base.

    The class was moved out of tokenization_utils; the alias only satisfies an
    old type import and does not alter tokenization, schema, or token selection.
    No installed package or the old experiment environment is modified.
    """
    import transformers.tokenization_utils as old_location
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(old_location, "PreTrainedTokenizerBase"):
        old_location.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from frontier_judge_decoding import JudgeJSONConstraint
    return JudgeJSONConstraint(tokenizer, eos_token_ids)


def download_worker():
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True, timeout=30)
    if info.sha != REVISION:
        raise ValueError("Pinned revision mismatch")
    selected = [f for f in info.siblings if "/" not in f.rfilename and
                (f.rfilename.endswith((".json", ".safetensors", ".txt", ".model", ".jinja", ".md"))
                 or f.rfilename == "LICENSE")]
    declared = sum(f.size or 0 for f in selected)
    if not 0 < declared < 64 * GiB or shutil.disk_usage(ROOT).free < declared + 40 * GiB:
        raise RuntimeError("Download size or free disk safety guard")
    records = [{"name": f.rfilename, "size": f.size,
                "expected_sha256": f.lfs.sha256 if f.lfs else None} for f in selected]
    dump(RUN / "download_metadata.json", {"repo": REPO, "revision": REVISION,
         "declared_bytes": declared, "files": records, "created_at": now()})
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    progress_lock = Lock()
    file_progress = {}
    def update_file(name, **values):
        with progress_lock:
            file_progress.setdefault(name, {}).update(values, updated_at=now())
            dump(RUN / "download_file_status.json", {"repo": REPO, "revision": REVISION,
                 "updated_at": now(), "files": file_progress})
    def fetch(item):
        name = item["name"]
        for attempt in range(1, 5):
            update_file(name, state="downloading", attempt=attempt, expected_bytes=item["size"])
            try:
                path = Path(hf_hub_download(REPO, name, revision=REVISION, token=False, cache_dir=str(CACHE)))
                if not path.resolve().is_relative_to(ROOT) or path.stat().st_size != item["size"]:
                    raise ValueError("Downloaded file size or project path mismatch")
                update_file(name, state="verifying_sha256")
                value = digest(path)
                if item["expected_sha256"] and value != item["expected_sha256"]:
                    raise ValueError("Official LFS SHA-256 mismatch")
                update_file(name, state="verified", sha256=value, verified_bytes=path.stat().st_size)
                return name, value, str(path.parent)
            except Exception as exc:
                # HTTP/timeout diagnostics intentionally omit messages and signed URLs.
                response = getattr(exc, "response", None)
                update_file(name, state="failed" if attempt == 4 or isinstance(exc, ValueError) else "retrying",
                            error_type=type(exc).__name__, http_status=getattr(response, "status_code", None))
                if attempt == 4 or isinstance(exc, ValueError):
                    raise
                time.sleep(min(20 * attempt, 60))
    hashes, snapshot, failures = {}, None, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, row): row["name"] for row in records}
        for future in as_completed(futures):
            try:
                name, sha, snapshot = future.result()
            except Exception as exc:
                failures.append({"name": futures[future], "error_type": type(exc).__name__})
                continue
            hashes[name] = sha
            dump(RUN / "download_verified.json", {"verified_files": len(hashes),
                 "total_files": len(records), "last_verified_file": name, "updated_at": now()})
    if failures:
        dump(RUN / "download_file_failures.json", {"files": failures, "updated_at": now()})
        raise RuntimeError("Some weight files failed bounded resumable download; see per-file status")
    result = {"repo": REPO, "revision": REVISION, "path": snapshot, "bytes": declared,
              "downloaded_at": now(), "trust_remote_code": False, "sha256": hashes,
              "official_lfs_hashes_checked": True, "architecture": "qwen3_5"}
    dump(WEIGHTS, result)


def download():
    lock = (RUN / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_path = RUN / "download.log"
    start = time.monotonic()
    blob_dir = CACHE / "models--Qwen--Qwen3.6-27B/blobs"
    last_progress, last_size = start, -1
    with log_path.open("a") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), "download-worker"],
                                 cwd=ROOT, env=environment(), stdout=log, stderr=log)
        dump(RUN / "download_pid.json", {"pid": os.getpid(), "worker_pid": child.pid,
             "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]})
        try:
            while child.poll() is None:
                blob_stats = [p.stat() for p in blob_dir.glob("*") if p.is_file()] if blob_dir.exists() else []
                size = sum(s.st_size for s in blob_stats)
                allocated = sum(s.st_blocks * 512 for s in blob_stats)
                if size != last_size:
                    last_size, last_progress = size, time.monotonic()
                if (RUN / "download.stop.requested").exists():
                    raise InterruptedError("Project download stop requested")
                if shutil.disk_usage(ROOT).free < 40 * GiB:
                    raise RuntimeError("Download free disk below 40 GiB")
                if log_path.stat().st_size > 10 * 1024**2:
                    raise RuntimeError("Download log exceeded 10 MiB")
                if time.monotonic() - last_progress > 20 * 60:
                    raise TimeoutError("Download made no byte progress for twenty minutes")
                if time.monotonic() - start > 24 * 3600:
                    raise TimeoutError("Download exceeded twenty-four hours")
                dump(RUN / "download_status.json", {"state": "downloading", "repo": REPO,
                     "revision": REVISION, "downloaded_cache_bytes": size, "updated_at": now(),
                     "allocated_cache_bytes": allocated,
                     "elapsed_seconds": time.monotonic() - start, "worker_pid": child.pid})
                time.sleep(20)
            if child.returncode:
                raise RuntimeError("Weight download worker failed; see structured worker error")
            dump(RUN / "download_status.json", {"state": "complete", "repo": REPO,
                 "revision": REVISION, "completed_at": now(), "manifest": str(WEIGHTS)})
        except BaseException as exc:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
            dump(RUN / "download_status.json", {"state": "stopped" if isinstance(exc, InterruptedError) else "failed",
                 "error_type": type(exc).__name__, "updated_at": now(), "resumable": True})
            raise


class ModernGPUJudge:
    """Drop-in document-judge interface using the pinned Qwen3.6 text path.

    The runner handles OOM by retrying a smaller batch. This class never truncates
    the source or silently drops a task, and repairs use the same fixed model.
    """

    def __init__(self, weights, memory_gib=36, loading_callback=None):
        import torch
        import transformers
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        from importlib.metadata import version
        if weights.get("repo") != REPO or weights.get("revision") != REVISION:
            raise ValueError("Modern judge requires the frozen Qwen3.6-27B revision")
        path = Path(weights["path"])
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Judge weights outside this project")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the production judge")
        if not 24 <= memory_gib <= 40:
            raise ValueError("Qwen judge GPU allowance must be between 24 and 40 GiB")
        free, total = torch.cuda.mem_get_info(0)
        if free < (memory_gib + 4) * GiB:
            raise MemoryError("Insufficient aggregate GPU headroom for the requested judge cap")
        if loading_callback:
            loading_callback()
        torch.cuda.set_per_process_memory_fraction(memory_gib * GiB / total, 0)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, device_map={"": 0},
            dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16))
        issues = {key: value for key, value in loading_info.items()
                  if key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") and value}
        if issues:
            raise ValueError("Incomplete or mismatched Qwen weights; refusing production inference")
        self.model.eval()
        self.loading_info = {key: sorted(value) if isinstance(value, set) else value
                             for key, value in loading_info.items()}
        self.weights = weights
        self.json_constraint = prepare_json_constraint(self.tokenizer, self.model.generation_config.eos_token_id)
        self.memory_cap_gib = float(memory_gib)
        self.prompt_batch_limit = 16
        self.batch_token_budget = 32000
        self.capacity_upgrade = None
        self._prompt_length_cache = {}
        self.metadata = {"backend": "native_transformers_qwen3_5", "transformers": transformers.__version__,
                         "torch": torch.__version__, "enable_thinking": False,
                         "decoding": "greedy_json_schema_constrained", "modality": "text_only",
                         "quantization": "bitsandbytes_nf4_double_quant", "compute_dtype": "bfloat16",
                         "attention": "sdpa", "input_truncation": False,
                         "gated_delta_kernels": "transformers_native_dispatch_to_fla_core",
                         "fla_core_version": version("fla-core"),
                         "triton_version": version("triton"),
                         "lmfe_transformers5_compatibility": "public_tokenizer_base_type_import_alias",
                         "max_complete_input_tokens": 12000}

    def maybe_expand_capacity(self):
        # Unlike the old shared-GPU Phi run, this run starts at the selected cap.
        return None

    @staticmethod
    def _messages(packet, queries, repair=None):
        import run_frontier_assessment as assessment
        messages = [{"role": "system", "content": assessment.SYSTEM},
                    {"role": "user", "content": assessment.prompt(packet, queries)}]
        if repair is not None:
            prior = repair["output"] if isinstance(repair, dict) else repair
            error = repair.get("error", "schema_or_reference_error") if isinstance(repair, dict) else "schema_or_reference_error"
            messages += [{"role": "assistant", "content": prior}, {"role": "user", "content":
                "The previous answer failed validation: " + error + ". Return only the required JSON array, with one answer per request and no extra fields. Every known grade (including 0) or known applicability requires 1 to 3 existing excerpt numbers in e. Reconsider the same sources: cite the model's stated scope for a scope mismatch; if no source supports a conclusion, use g=null, a=U, e=[]. A missing requested population/route/species cannot support g=2, g=3 or a=Y. Different drugs alone cannot exclude a mechanistic template request. Do not invent support to satisfy validation."}]
        return messages

    def _render(self, packet, queries, repair=None):
        return self.tokenizer.apply_chat_template(self._messages(packet, queries, repair),
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def select_batch_size(self, tasks, maximum):
        import run_frontier_assessment as assessment
        possible = tasks[:min(maximum, self.prompt_batch_limit)]
        if not possible:
            raise ValueError("Cannot plan an empty batch")
        keys = [(p["packet_sha256"], tuple(q["text"] for q in qs)) for p, qs in possible]
        missing = [(key, p, qs) for key, (p, qs) in zip(keys, possible) if key not in self._prompt_length_cache]
        if missing:
            texts = [self._render(p, qs) for _, p, qs in missing]
            rows = self.tokenizer(texts, padding=False, truncation=False)["input_ids"]
            if len(rows) != len(missing):
                raise ValueError("Tokenizer changed prompt count")
            for (key, _, _), tokens in zip(missing, rows):
                self._prompt_length_cache[key] = len(tokens)
        lengths = [self._prompt_length_cache[key] for key in keys]
        while len(self._prompt_length_cache) > 4096:
            self._prompt_length_cache.pop(next(iter(self._prompt_length_cache)))
        width, selected = 0, 0
        for count, length in enumerate(lengths, 1):
            if length > 12000:
                raise ValueError("Complete prompt exceeds frozen input limit; truncation refused")
            width = max(width, length)
            if (width + assessment.RETRY_TOKENS) * count > self.batch_token_budget:
                break
            selected = count
        return max(1, selected)

    def generate(self, tasks, max_new_tokens=96, repairs=None):
        if not tasks:
            raise ValueError("Cannot generate an empty batch")
        if repairs is not None and len(repairs) != len(tasks):
            raise ValueError("Repair count does not match task count")
        texts = [self._render(p, qs, repairs[i] if repairs is not None else None)
                 for i, (p, qs) in enumerate(tasks)]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        input_width = inputs["input_ids"].shape[1]
        if input_width > 12000:
            raise ValueError("Complete prompt exceeds frozen input limit; truncation refused")
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        self.torch.cuda.synchronize()
        started = time.monotonic()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.model.generation_config.eos_token_id,
                prefix_allowed_tokens_fn=self.json_constraint.for_tasks(tasks))
        self.torch.cuda.synchronize()
        seconds = time.monotonic() - started
        values = self.tokenizer.batch_decode(outputs[:, input_width:], skip_special_tokens=True)
        metrics = {"wall_seconds": seconds, "input_token_width": input_width,
                   "input_tokens_per_prompt": input_lengths,
                   "output_token_width": outputs.shape[1] - input_width,
                   "prompt_batch_size": len(tasks), "requests_per_prompt": [len(qs) for _, qs in tasks],
                   "gpu_memory_cap_gib": self.memory_cap_gib, "prompt_batch_limit": self.prompt_batch_limit,
                   "batch_token_budget": self.batch_token_budget, "capacity_upgrade_at": None,
                   "cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
                   "cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0), **self.metadata}
        del inputs, outputs
        return values, metrics


def smoke():
    """Real CUDA compatibility checks, excluded from scientific pilot labels.

    Run only after the project Phi pilot worker exits. Completion writes a smoke
    report; readiness is published separately after this process has exited.
    """
    import inspect
    import torch
    import run_frontier_assessment as assessment
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    started = now()
    dump(RUN / "smoke_status.json", {"state": "kernel_validation", "pid": os.getpid(), "started_at": started})
    torch.manual_seed(20260919)
    comparisons = []
    with torch.inference_mode():
        for mode, length in (("prefill", 128), ("prefill", 257), ("decode", 1)):
            q = torch.randn(2, length, 4, 128, device="cuda", dtype=torch.bfloat16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            gate = -torch.rand(2, length, 4, device="cuda", dtype=torch.float32)
            beta = torch.sigmoid(torch.randn(2, length, 4, device="cuda", dtype=torch.bfloat16))
            state = torch.randn(2, 4, 128, 128, device="cuda", dtype=torch.float32) * .05
            wrapped = modeling.torch_chunk_gated_delta_rule if mode == "prefill" else modeling.torch_recurrent_gated_delta_rule
            reference = inspect.unwrap(wrapped)
            fast = chunk_gated_delta_rule if mode == "prefill" else fused_recurrent_gated_delta_rule
            kwargs = dict(g=gate, beta=beta, initial_state=state.clone(), output_final_state=True,
                          use_qk_l2norm_in_kernel=True)
            ref, ref_state = reference(q, k, v, **kwargs)
            out, out_state = fast(q, k, v, **kwargs)
            relative = lambda a, expected: float(torch.linalg.vector_norm(a.float() - expected.float()) /
                                               torch.linalg.vector_norm(expected.float()).clamp_min(1e-8))
            error, state_error = relative(out, ref), relative(out_state, ref_state)
            passed = error < .02 and state_error < .02 and bool(torch.isfinite(out).all())
            comparisons.append({"mode": mode, "tokens": length, "relative_output_l2": error,
                                "relative_state_l2": state_error, "passed": passed})
            if not passed:
                raise ValueError("FLA kernels disagree with independent Torch reference")
        del q, k, v, gate, beta, state, out, out_state, ref, ref_state, kwargs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dump(RUN / "fla_reference_validation.json", {"created_at": now(), "seed": 20260919,
         "independent_torch_reference_compared": True, "relative_l2_tolerance": .02,
         "comparison": "inspect.unwrap of HF reference vs installed FLA", "all_passed": True,
         "results": comparisons})
    weights = json.loads(WEIGHTS.read_text())
    dump(RUN / "smoke_status.json", {"state": "loading_model", "pid": os.getpid(), "started_at": started})
    judge = ModernGPUJudge(weights, memory_gib=36)
    cases = [json.loads(row) for row in (OUT / "validation/cases.jsonl").read_text().splitlines() if row]
    lengths = [(len(judge.tokenizer(judge._render(c["packet"], [c["query"]]), truncation=False)["input_ids"]), c)
               for c in cases if c["packet"]["refs"]]
    short, longest = min(lengths, key=lambda x: x[0])[1], max(lengths, key=lambda x: x[0])[1]
    results = []
    for label, candidates in (("short_single", [short]), ("long_single", [longest]),
                              ("long_batched", [longest] * 16)):
        tasks = [(c["packet"], [c["query"]]) for c in candidates]
        count = judge.select_batch_size(tasks, 16)
        active = tasks[:count]
        values, metrics = judge.generate(active)
        for value, (packet, _) in zip(values, active):
            assessment.parse_answers(value, [1], packet["refs"])
        results.append({"case": label, "case_ids": [c["case_id"] for c in candidates[:count]],
                        "outputs": values, "schema_valid": True, "metrics": metrics})
        dump(RUN / "smoke_status.json", {"state": "generating", "pid": os.getpid(),
             "started_at": started, "completed_batches": len(results), "last_batch": metrics})
    repair_task = [(short["packet"], [short["query"]])]
    repair_values, repair_metrics = judge.generate(repair_task, max_new_tokens=160,
        repairs=[{"output": "[]", "error": "Missing or extra answers"}])
    assessment.parse_answers(repair_values[0], [1], short["packet"]["refs"])
    results.append({"case": "repair_single", "case_ids": [short["case_id"]],
                    "outputs": repair_values, "schema_valid": True, "metrics": repair_metrics})
    dump(RUN / "gpu_smoke_validation.json", {"created_at": now(), "started_at": started,
         "model_repo": REPO, "model_revision": REVISION, "weights_manifest_sha256": digest(WEIGHTS),
         "backend_sha256": digest(Path(__file__)), "scientific_pilot_labels": False,
         "pilot_cases_sha256": digest(OUT / "validation/cases.jsonl"),
         "max_complete_pilot_prompt_tokens": max(n for n, _ in lengths),
         "all_source_text_preserved": True, "all_batches_schema_valid": True,
         "weight_loading_info": judge.loading_info, "results": results})
    dump(RUN / "smoke_status.json", {"state": "complete", "pid": os.getpid(),
         "started_at": started, "completed_at": now()})


def finish_setup():
    """Bounded setup supervisor; publish readiness only after smoke GPU exit."""
    import run_frontier_assessment as assessment
    lock = (RUN / "setup.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {"pid": os.getpid(), "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]}
    dump(RUN / "setup_pid.json", identity)
    started = time.monotonic()
    phi_failure_started = None
    phi_path = ROOT / "runtime/deployment/frontier_scoring/pilot_phi_status.json"
    while True:
        if (RUN / "setup.stop.requested").exists():
            raise InterruptedError("Project setup stop requested")
        if time.monotonic() - started > 2 * 3600:
            raise TimeoutError("Setup dependencies were not ready after two hours")
        if shutil.disk_usage(ROOT).free < 40 * GiB:
            raise RuntimeError("Setup free disk below forty GiB")
        dl = assessment.read(RUN / "download_status.json", {})
        phi = assessment.read(phi_path, {})
        if dl.get("state") in ("failed", "stopped"):
            raise RuntimeError("Qwen download did not complete")
        downloaded = dl.get("state") == "complete" and WEIGHTS.is_file()
        if phi.get("state") in ("failed", "stopped", "blocked"):
            phi_failure_started = phi_failure_started or time.monotonic()
            if time.monotonic() - phi_failure_started > 20 * 60:
                dump(RUN / "setup_status.json", {**identity, "state": "blocked_phi_pilot", "updated_at": now(),
                     "phi_pilot_state": phi.get("state"), "gpu_loaded": False})
                raise RuntimeError("Phi pilot remained failed or stopped for twenty minutes; setup did not load GPU")
        else:
            phi_failure_started = None
        phi_exited = (phi.get("state") == "complete" and type(phi.get("pid")) is int
                      and isinstance(phi.get("start_ticks"), str) and not assessment.alive(phi))
        if downloaded and phi_exited:
            break
        dump(RUN / "setup_status.json", {**identity, "state": "waiting", "updated_at": now(),
             "weights_ready": downloaded, "phi_pilot_complete_and_exited": phi_exited,
             "phi_pilot_state": phi.get("state"), "gpu_loaded": False})
        time.sleep(20)
    log_path = RUN / "smoke.log"
    with log_path.open("a") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), "smoke"], cwd=ROOT,
                                 env=environment(), stdout=log, stderr=log)
        launched = time.monotonic()
        dump(RUN / "setup_status.json", {**identity, "state": "gpu_smoke", "updated_at": now(), "worker_pid": child.pid})
        try:
            while child.poll() is None:
                if (RUN / "setup.stop.requested").exists():
                    raise InterruptedError("Project setup stop requested")
                if shutil.disk_usage(ROOT).free < 40 * GiB or log_path.stat().st_size > 20 * 1024**2:
                    raise RuntimeError("Setup disk or log guard")
                if time.monotonic() - launched > 45 * 60:
                    raise TimeoutError("GPU smoke exceeded forty-five minutes")
                progress_path = RUN / "smoke_status.json"
                if progress_path.exists() and time.time() - progress_path.stat().st_mtime > 20 * 60:
                    raise TimeoutError("GPU smoke stage made no progress for twenty minutes")
                time.sleep(20)
            if child.returncode:
                raise RuntimeError("GPU smoke failed; see smoke_error.json")
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
            raise
    reports = ["backend_cpu_validation.json", "pilot_prompt_length_validation.json",
               "fla_reference_validation.json", "gpu_smoke_validation.json"]
    dump(RUN / "ready.json", {"state": "ready", "created_at": now(),
         "backend_sha256": digest(Path(__file__)), "weights_manifest_sha256": digest(WEIGHTS),
         "environment_versions": assessment.read(RUN / "environment_versions.json"),
         "validation_report_sha256": {name: digest(RUN / name) for name in reports},
         "gpu_smoke_process_exited": True, "smoke_worker_pid": child.pid,
         "enable_thinking": False, "memory_cap_gib": 36, "batch_token_budget": 32000})
    dump(RUN / "setup_status.json", {**identity, "state": "complete", "completed_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "download-worker", "smoke", "finish-setup"))
    args = parser.parse_args()
    init()
    try:
        {"download": download, "download-worker": download_worker, "smoke": smoke,
         "finish-setup": finish_setup}[args.action]()
    except Exception as exc:
        dump(RUN / ("download_worker_error.json" if args.action == "download-worker" else args.action + "_error.json"),
             {"error_type": type(exc).__name__, "at": now(),
              **({"message": str(exc)[:1200]} if args.action == "smoke" else {})})
        print(json.dumps({"state": "failed", "error_type": type(exc).__name__}), flush=True)
        sys.exit(1)
