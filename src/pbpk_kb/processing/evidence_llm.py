"""Evidence screening with local weights or the explicitly registered Qwen service."""
from __future__ import annotations

import os
import json
from pathlib import Path
from threading import Lock, Thread
from concurrent.futures import Future
from queue import Queue, Empty
import time
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "Qwen3-8B-BF16"
LOCAL_BASE_URL = "local://qwen3-8b"
MAX_EVIDENCE_OUTPUT_TOKENS = 4096
_LOCK = Lock()
_GENERATOR = None
_BATCHER = None


class EvidenceGenerationError(ValueError):
    def __init__(self, result):
        super().__init__(f"Local Qwen evidence output is not valid JSON ({result.error}); record remains unresolved.")
        self.raw_response = str(getattr(result, "raw_text", ""))


class EvidenceServerOutputError(ValueError):
    def __init__(self, message, result):
        super().__init__(message)
        self.raw_response = json.dumps(result, ensure_ascii=False)


def _normal_generation_schema(prompt, payload=None):
    from pbpk_kb.processing.llm_agent import (LLM_EXTRACTION_PROMPT, extraction_output_schema,
                                            LLM_INDEXED_EXTRACTION_PROMPT, indexed_extraction_output_schema)
    if prompt == LLM_INDEXED_EXTRACTION_PROMPT:
        return indexed_extraction_output_schema(payload or {})
    if prompt.startswith(LLM_EXTRACTION_PROMPT):
        return extraction_output_schema(payload or {})
    from pbpk_kb.processing.final_evidence import LLM_ATTRIBUTION_PROMPT, attribution_output_schema
    if prompt.startswith(LLM_ATTRIBUTION_PROMPT):
        return attribution_output_schema(payload or {})
    from pbpk_kb.processing.auto_quality_llm_stage import quality_output_schema
    return quality_output_schema(prompt, payload or {})


def _schema_group(prompt):
    from pbpk_kb.processing.llm_agent import LLM_EXTRACTION_PROMPT, LLM_INDEXED_EXTRACTION_PROMPT
    from pbpk_kb.processing.final_evidence import LLM_ATTRIBUTION_PROMPT
    if prompt == LLM_INDEXED_EXTRACTION_PROMPT:
        return "indexed_extraction"
    if prompt.startswith(LLM_EXTRACTION_PROMPT):
        return "extraction"
    if prompt.startswith(LLM_ATTRIBUTION_PROMPT):
        return "attribution"
    from pbpk_kb.processing.auto_quality_llm_stage import REPAIR_PROMPT, VALIDATION_PROMPT
    if prompt == REPAIR_PROMPT:
        return "quality_repair"
    return "quality_validation" if prompt == VALIDATION_PROMPT else "generic"


class EvidenceBatcher:
    """Bounded GPU microbatches; each caller keeps its own prompt and result."""
    def __init__(self, generator, batch_size: int, wait_seconds: float = .2):
        if not 1 <= batch_size <= 16:
            raise ValueError("Evidence GPU batch size must be between 1 and 16.")
        self.generator = generator
        self.batch_size = batch_size
        self.wait_seconds = wait_seconds
        self.queue = Queue(maxsize=32)
        Thread(target=self._run, name="project-evidence-gpu-batches", daemon=True).start()

    def submit(self, prompt, payload):
        result = Future()
        self.queue.put((prompt, payload, result))
        return result

    def _run(self):
        while True:
            batch = [self.queue.get()]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.queue.get(timeout=max(0, deadline-time.monotonic())))
                except Empty:
                    break
            try:
                groups = []
                for item in batch:
                    schema = _normal_generation_schema(item[0], item[1])
                    kind = _schema_group(item[0])
                    if not groups or groups[-1][0] != kind:
                        groups.append((kind, [], []))
                    groups[-1][1].append(item)
                    groups[-1][2].append(schema)
                for _, group, schemas in groups:
                    schema = schemas[0] if all(s == schemas[0] for s in schemas) else schemas
                    options = {"json_schema": schema} if schema is not None else {}
                    results = self.generator.generate_json_batch(
                        [(prompt, payload) for prompt, payload, _ in group],
                        max_new_tokens=MAX_EVIDENCE_OUTPUT_TOKENS, **options)
                    if len(results) != len(group):
                        raise RuntimeError("Evidence GPU batch result count mismatch.")
                    for (_, _, future), result in zip(group, results):
                        if result.parsed is None:
                            future.set_exception(EvidenceGenerationError(result))
                        else:
                            future.set_result(result.parsed)
            except Exception as exc:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(exc)


def server_backend() -> bool:
    backend = os.getenv("PBPK_EVIDENCE_LLM_BACKEND", "local")
    if backend not in {"local", "vllm"}:
        raise ValueError("Unknown evidence LLM backend")
    return backend == "vllm"


def registered_server() -> dict[str, Any]:
    config = json.loads((ROOT / "runtime/deployment/qwen_service.json").read_text())
    url = urlsplit(config["base_url"])
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
            or url.path != "/v1" or url.username or url.password or url.query or url.fragment):
        raise ValueError("Only the registered loopback Qwen endpoint is allowed")
    if not config.get("served_model") or not config.get("model_config_sha256"):
        raise ValueError("Registered Qwen identity is incomplete")
    return config


def verify_server_process(config: dict[str, Any]) -> None:
    """Check only the user-designated service PID; never discover other services."""
    expected = config["server_identity"]
    process = Path(f"/proc/{int(expected['pid'])}")
    try:
        if process.stat().st_uid != os.getuid():
            raise RuntimeError("Registered Qwen service owner changed")
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] != expected["start_ticks"] or fields[0] in {"Z", "T", "t"}:
            raise RuntimeError("Registered Qwen service identity or availability changed")
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError("Registered Qwen service is unavailable") from exc


def backend_identity() -> tuple[str, str]:
    if server_backend():
        config = registered_server()
        return config["served_model"], config["base_url"]
    return DEFAULT_MODEL, LOCAL_BASE_URL


def generation_format(local_format: str) -> str:
    if not server_backend():
        return local_format
    if registered_server().get("backend_kind") == "project_transformers_qwen36_http":
        return "project_transformers_nf4_json_schema_v1"
    return "vllm_openai_json_schema_v1"


def validate_backend(base_url: str | None = None, model: str | None = None) -> None:
    if server_backend():
        expected_model, expected_url = backend_identity()
        if (base_url and base_url.rstrip("/") != expected_url) or (model and model != expected_model):
            raise ValueError("Only the explicitly registered Qwen service and model may be used")
        return
    if base_url and base_url.rstrip("/") != LOCAL_BASE_URL:
        raise ValueError("Evidence screening requires local Qwen3-8B; external LLM APIs are disabled.")
    if model and model not in {DEFAULT_MODEL, "Qwen/Qwen3-8B", "Qwen3-8B"}:
        raise ValueError("Evidence screening requires the project Qwen3-8B model.")


def require_evidence_llm_enabled() -> None:
    if os.getenv("PBPK_EVIDENCE_LLM_ENABLED", "true").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError("Evidence LLM is disabled pending replacement model configuration; no model will be loaded or downloaded.")


def _generate_from_server(system_prompt, payload, options):
    import httpx
    from jsonschema import Draft202012Validator, ValidationError
    config = registered_server()
    verify_server_process(config)
    schema = options.get("json_schema")
    if not isinstance(schema, dict):
        raise ValueError("Server evidence generation requires an explicit JSON schema")
    body = {
        "model": config["served_model"],
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}],
        "temperature": 0, "max_tokens": MAX_EVIDENCE_OUTPUT_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "pbpk_evidence", "schema": schema, "strict": True}},
    }
    if "sampling_seed" in options:
        body.update(temperature=.7, top_p=.8, top_k=20, min_p=0.0, seed=options["sampling_seed"])
    if "repetition_penalty" in options:
        body["repetition_penalty"] = options["repetition_penalty"]
    try:
        timeout = config.get("request_timeout_seconds", 180)
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 1800:
            raise ValueError("Registered evidence timeout is outside the bounded range")
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout) as client:
            response = client.post(config["base_url"] + "/chat/completions", json=body)
    except httpx.HTTPError as exc:
        raise RuntimeError("Registered Qwen request failed: " + type(exc).__name__) from None
    if response.status_code != 200:
        raise RuntimeError(f"Registered Qwen returned HTTP {response.status_code}; no fallback or automatic retry")
    result = response.json()
    if result.get("model") != config["served_model"] or len(result.get("choices") or []) != 1:
        raise ValueError("Qwen response model or choice count does not match the request")
    choice = result["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise EvidenceServerOutputError("Qwen generation did not finish: " + str(choice.get("finish_reason")), result)
    try:
        parsed = json.loads(choice["message"]["content"])
        Draft202012Validator(schema).validate(parsed)
    except (ValueError, ValidationError) as exc:
        raise EvidenceServerOutputError("Qwen output failed JSON/schema validation: " + type(exc).__name__, result) from exc
    return parsed


def generate_evidence_json(system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    global _GENERATOR, _BATCHER
    require_evidence_llm_enabled()
    batch_size = int(os.getenv("PBPK_EVIDENCE_LLM_BATCH_SIZE", "1"))
    if not 1 <= batch_size <= 16:
        raise ValueError("Evidence GPU batch size must be between 1 and 16.")
    default_schema = _normal_generation_schema(system_prompt, payload)
    recovery_kwargs = {"json_schema": default_schema} if default_schema is not None else {}
    if payload.get("generation_recovery_mode") in {"comparison_schema_v1", "comparison_sampling_v1", "comparison_penalty_v1", "focused_assertion_v1"}:
        if batch_size != 1:
            raise ValueError("Schema recovery requires an isolated single-request call")
        from pbpk_kb.processing.llm_agent import extraction_recovery_schema
        recovery_kwargs["json_schema"] = extraction_recovery_schema(payload)
        if payload.get("generation_recovery_mode") in {"comparison_sampling_v1", "comparison_penalty_v1"}:
            recovery_kwargs["sampling_seed"] = 20260908
        if payload.get("generation_recovery_mode") == "comparison_penalty_v1":
            recovery_kwargs["repetition_penalty"] = 1.1
    if server_backend():
        return _generate_from_server(system_prompt, payload, recovery_kwargs)
    # All workers share one model; only the batch worker generates when enabled.
    with _LOCK:
        if _GENERATOR is None:
            from pbpk_kb.recommendation.local_llm import LocalStructuredGenerator

            model_path = (ROOT / "runtime/models/Qwen3-8B-BF16").resolve()
            if not model_path.is_relative_to(ROOT) or not (model_path / "config.json").is_file():
                raise FileNotFoundError("Project Qwen3-8B weights are unavailable; no external fallback is allowed.")
            _GENERATOR = LocalStructuredGenerator(
                str(model_path), device=os.getenv("PBPK_EVIDENCE_LLM_DEVICE", "cuda:0"),
                local_files_only=True, dtype="bfloat16", reject_truncation=True, enforce_json=True,
            )
        if batch_size > 1:
            if _BATCHER is None:
                _BATCHER = EvidenceBatcher(_GENERATOR, batch_size)
            future = _BATCHER.submit(system_prompt, payload)
        else:
            result = _GENERATOR.generate_json(
                system_prompt=system_prompt, payload=payload, max_new_tokens=MAX_EVIDENCE_OUTPUT_TOKENS, **recovery_kwargs)
            if result.parsed is None:
                raise EvidenceGenerationError(result)
            return result.parsed
    return future.result()
