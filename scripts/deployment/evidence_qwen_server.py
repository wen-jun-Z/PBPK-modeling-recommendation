"""Project-owned Qwen3.6 evidence endpoint; offline weights and bounded batches.

The HTTP shape is OpenAI-compatible, but inference is native Transformers NF4,
not vLLM. The frozen evaluation/component implementations remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, InvalidStateError, TimeoutError as FutureTimeout
from contextlib import nullcontext
from copy import copy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime/deployment/evidence_expansion_20261009/backend"
CONFIG = ROOT / "runtime/deployment/qwen_service.json"
MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
WEIGHTS = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_qwen36/model_weights/qwen.json"
MAX_INPUT = 12000
MAX_OUTPUT = 4096
MAX_GENERATION_SECONDS = 300
TOKEN_BUDGET = 16384
MAX_BODY = 2 * 1024**2
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    process = Path("/proc") / str(pid)
    if process.stat().st_uid != os.getuid():
        raise ValueError("Process owner does not match this project account")
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19]}


def owned_alive(record):
    try:
        process = Path("/proc") / str(record["pid"])
        return (identity(int(record["pid"])) == record
            and (process / "stat").read_text().rsplit(")", 1)[1].split()[0] not in {"Z", "T", "t"}
            and str(ROOT).encode() in (process / "cmdline").read_bytes())
    except (OSError, KeyError, ValueError):
        return False


def validate_request(body):
    from jsonschema import Draft202012Validator
    if not isinstance(body, dict) or body.get("model") != MODEL:
        raise ValueError("Only the project Qwen3.6 model is allowed")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
        raise ValueError("Invalid message count")
    if any(not isinstance(row, dict) or row.get("role") not in {"system", "user", "assistant"}
           or not isinstance(row.get("content"), str) for row in messages):
        raise ValueError("Only explicit text messages are accepted")
    if body.get("chat_template_kwargs", {}).get("enable_thinking", False) is not False:
        raise ValueError("Evidence extraction uses non-thinking JSON generation")
    output = body.get("max_tokens", MAX_OUTPUT)
    if type(output) is not int or not 1 <= output <= MAX_OUTPUT:
        raise ValueError("Invalid output token budget")
    form = body.get("response_format", {})
    schema = form.get("json_schema", {}).get("schema")
    if form.get("type") != "json_schema" or not isinstance(schema, dict):
        raise ValueError("Explicit JSON schema required")
    Draft202012Validator.check_schema(schema)
    if body.get("temperature", 0) not in (0, .7):
        raise ValueError("Only greedy generation or registered recovery sampling is allowed")
    if body.get("temperature", 0) == .7:
        if (type(body.get("seed")) is not int or not 0 <= body["seed"] < 2**32
                or body.get("top_p") != .8 or body.get("top_k") != 20 or body.get("min_p") != 0):
            raise ValueError("Recovery sampling differs from the evidence protocol")
    penalty = body.get("repetition_penalty", 1.0)
    if not isinstance(penalty, (float, int)) or not 1 <= penalty <= 1.2:
        raise ValueError("Invalid repetition penalty")
    return body


def decoding_mode(body, *, allow_prompt_only=False):
    schema = body.get("response_format", {}).get("json_schema", {}).get("schema")
    return "prompt_only" if allow_prompt_only and schema == {"type": "object"} else "lmfe"


def generation_group(body, *, allow_prompt_only=False):
    # Seeded recovery stays isolated so batching cannot change its random stream.
    if body.get("temperature", 0):
        return None
    group = (body.get("max_tokens", MAX_OUTPUT), body.get("repetition_penalty", 1.0))
    # Preserve the legacy key by default; opt-in batches must share a decoder.
    return group + (decoding_mode(body, allow_prompt_only=True),) if allow_prompt_only else group


def generation_outcome(token_ids, eos, output_limit):
    """Report each row's actual EOS, excluding other batch rows' padding."""
    for index, token in enumerate(token_ids):
        if int(token) in eos:
            return "stop", index + 1, "eos"
    return "length", len(token_ids), ("output_token_limit" if len(token_ids) >= output_limit else "generation_time_limit")


def generation_result(task, content, token_ids, eos, output_limit, *, allow_prompt_only=False):
    """Preserve invalid output for diagnosis without discarding valid batch rows."""
    from jsonschema import Draft202012Validator, ValidationError

    finish, completion_tokens, reason = generation_outcome(token_ids, eos, output_limit)
    generation = {"stop_reason": reason, "time_limit_seconds": MAX_GENERATION_SECONDS,
                  "output_token_limit": output_limit}
    prompt_only = decoding_mode(task["body"], allow_prompt_only=allow_prompt_only) == "prompt_only"
    if prompt_only:
        generation.update(decoding_mode="prompt_only", content_validation=False)
    if finish == "stop" and not prompt_only:
        try:
            parsed = json.loads(content)
            Draft202012Validator(task["body"]["response_format"]["json_schema"]["schema"]).validate(parsed)
        except json.JSONDecodeError as exc:
            finish = "invalid_json"
            generation.update(stop_reason=finish, error_type=type(exc).__name__,
                              error_position={"offset": exc.pos, "line": exc.lineno, "column": exc.colno})
        except ValidationError as exc:
            finish = "invalid_schema"
            generation.update(stop_reason=finish, error_type=type(exc).__name__,
                              error_path=list(exc.absolute_path), schema_path=list(exc.absolute_schema_path))
    return {"object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": finish}],
            "generation": generation,
            "usage": {"prompt_tokens": task["length"], "completion_tokens": completion_tokens}}


def generation_counts(outputs):
    """Count normal EOS completions; prompt-only content is not validated."""
    finishes = [result["choices"][0]["finish_reason"] for result in outputs]
    completed = finishes.count("stop")
    truncated = finishes.count("length")
    return {"responded_requests": len(outputs), "completed_requests": completed,
            "truncated_requests": truncated, "invalid_requests": len(outputs) - completed - truncated,
            "failed_requests": len(outputs) - completed}


def install_json_const_compatibility():
    """Compile non-string JSON const scalars without altering their schema.

    LMFE 0.11.3 passes Python bool/int/None to StringParsingState, which expects
    text. JSON spelling is lowercase true/false/null (not Python repr). This
    process-local adapter preserves exactly one legal literal; final validation
    still uses the untouched original schema.
    """
    from lmformatenforcer import jsonschemaparser as module
    original = module.get_parser
    if getattr(original, "pbpk_json_const_compatibility", False):
        return

    def compatible(parsing_state, value_schema):
        extras = getattr(value_schema, "extras", None)
        if extras and "const" in extras and not isinstance(extras["const"], str):
            value = extras["const"]
            if value is not None and type(value) not in {bool, int, float}:
                raise ValueError("Unsupported non-scalar const; never loosen the schema")
            literal = json.dumps(value, allow_nan=False, separators=(",", ":"))
            return module.StringParsingState(parsing_state, [literal],
                require_opening_quote=False, require_closing_quote=False)
        return original(parsing_state, value_schema)

    compatible.pbpk_json_const_compatibility = True
    module.get_parser = compatible


def evidence_tokenizer_data(tokenizer_data):
    """Isolate a JSON-string cache that excludes every raw control character.

    LMFE 0.11.3's fast string cache filters LF/CR, but admits other controls.
    Structural whitespace remains in the token tree; escaped controls remain
    legal string content. The frozen component tokenizer data is not mutated.
    """
    from lmformatenforcer.tokenizerprefixtree import JsonFreetextTokenCache

    data = copy(tokenizer_data)
    data.tokenizer_tree = copy(tokenizer_data.tokenizer_tree)
    cache = JsonFreetextTokenCache(data.use_bitmask, data.vocab_size)
    for token_id, decoded, _ in data.regular_tokens:
        if not any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in decoded):
            cache.add_token(decoded, token_id)
    cache.freeze()
    data.tokenizer_tree.json_freetext_tokens = cache
    return data


def evidence_json_callback(tokenizer_data, body):
    """Keep source Unicode representable across byte-fallback token boundaries.

    LMFE derives its alphabet from individual token prefixes. A character such
    as U+2217 (the source unit's multiplication sign) may require multiple byte
    tokens, so it is missing there despite being permitted by the string fast
    path. Without adding source characters, LMFE pops the entire JSON parser
    stack on that character and incorrectly enables EOS inside a string.
    """
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    from pbpk_kb.processing.evidence_json_constraints import (
        install_strict_object_separators, install_unicode_freetext_consumption,
    )

    install_strict_object_separators()
    install_unicode_freetext_consumption()
    schema = body["response_format"]["json_schema"]["schema"]
    request_characters = json.dumps({"messages": body["messages"], "schema": schema}, ensure_ascii=False)
    data = copy(tokenizer_data)
    data.tokenizer_alphabet = "".join(sorted({character
        for character in data.tokenizer_alphabet + request_characters
        if ord(character) >= 32 and not 0xD800 <= ord(character) <= 0xDFFF}))
    config = CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet,
        max_consecutive_whitespaces=2, max_json_array_length=512)
    properties = schema.get("properties", {})
    if (set(properties) == {"assessment", "reason", "candidates"}
            and properties["assessment"].get("enum") == ["extract", "no_evidence", "uncertain"]):
        # Decide from the source before opening the optional evidence array.
        # Otherwise an early [] can force the later assessment to abstain.
        config.force_json_field_order = True
    parser = JsonSchemaParser(schema, config=config)
    callback = build_transformers_prefix_allowed_tokens_fn(data, parser)
    # LMFE's TokenEnforcer constructor replaces the supplied config. Restore
    # the explicitly bounded settings after it has initialized its token tree.
    parser.config = config
    return callback


class EvidenceGenerator:
    def __init__(self, *, memory_gib=24, token_budget=TOKEN_BUDGET, allow_prompt_only=False):
        validate_runtime_options(memory_gib, token_budget, allow_prompt_only)
        self.token_budget, self.allow_prompt_only = token_budget, allow_prompt_only
        # Verifies every weight shard and the installed official fast kernels.
        install_json_const_compatibility()
        from generate_component_model_integration import ComponentGenerator
        self.core = ComponentGenerator(memory_gib=memory_gib)
        self.torch = self.core.torch
        self.json_tokenizer_data = evidence_tokenizer_data(self.core.constraint.tokenizer_data)

    def render(self, body):
        text = self.core.tokenizer.apply_chat_template(body["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        size = len(self.core.tokenizer(text, padding=False, truncation=False)["input_ids"])
        if size > MAX_INPUT:
            raise ValueError("Complete evidence prompt exceeds input budget; never truncated")
        return text, size

    def generate(self, tasks):
        core, torch = self.core, self.torch
        modes = {decoding_mode(task["body"], allow_prompt_only=self.allow_prompt_only) for task in tasks}
        if len(modes) != 1:
            raise ValueError("A batch must use one decoding mode")
        constraints = {}
        if modes == {"lmfe"}:
            callbacks = [evidence_json_callback(self.json_tokenizer_data, task["body"]) for task in tasks]
            constraints["prefix_allowed_tokens_fn"] = lambda i, values: callbacks[i](i, values)
        body = tasks[0]["body"]
        output_limit = body.get("max_tokens", MAX_OUTPUT)
        inputs = core.tokenizer([task["text"] for task in tasks], return_tensors="pt",
            padding=True, truncation=False)
        width = inputs["input_ids"].shape[1]
        if (width + output_limit) * len(tasks) > self.token_budget:
            raise ValueError("Padded input and output exceed bounded batch budget")
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        sample = bool(body.get("temperature", 0))
        sampling = ({"do_sample": True, "temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0.0}
                    if sample else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
        rng = torch.random.fork_rng(devices=[0]) if sample else nullcontext()
        torch.cuda.reset_peak_memory_stats(0)
        start = time.monotonic()
        with torch.inference_mode(), rng:
            if sample:
                torch.manual_seed(body["seed"])
            tokens = core.model.generate(**inputs, max_new_tokens=output_limit, **sampling,
                repetition_penalty=body.get("repetition_penalty", 1.0), use_cache=True,
                max_time=MAX_GENERATION_SECONDS, pad_token_id=core.tokenizer.pad_token_id,
                eos_token_id=core.model.generation_config.eos_token_id, **constraints)
        torch.cuda.synchronize()
        generated = tokens[:, width:]
        raw = core.tokenizer.batch_decode(generated, skip_special_tokens=True)
        eos = set(core.constraint.eos_token_ids)
        results = []
        for index, (task, content) in enumerate(zip(tasks, raw, strict=True)):
            # Length/time-limited output is never silently accepted as completed.
            results.append(generation_result(task, content, generated[index].tolist(), eos, output_limit,
                allow_prompt_only=self.allow_prompt_only))
        metrics = {"batch_size": len(tasks), "input_tokens": [task["length"] for task in tasks],
            "output_width": int(generated.shape[1]), "wall_seconds": time.monotonic() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        del inputs, tokens, generated
        return results, metrics


def validate_batch_options(max_batch_size, batch_lookahead):
    if type(max_batch_size) is not int or not 1 <= max_batch_size <= 4:
        raise ValueError("max_batch_size must be an integer from one to four")
    if type(batch_lookahead) is not int or not 1 <= batch_lookahead <= 4:
        raise ValueError("batch_lookahead must be an integer from one to four")


def validate_runtime_options(memory_gib, token_budget, allow_prompt_only):
    if type(memory_gib) is not int or memory_gib < 1:
        raise ValueError("memory_gib must be a positive integer")
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError("token_budget must be a positive integer")
    if type(allow_prompt_only) is not bool:
        raise ValueError("allow_prompt_only must be a boolean")


def collect_batch(item, pending, incoming, generator, *, max_batch_size=2, batch_lookahead=1,
                  token_budget=TOKEN_BUDGET, allow_prompt_only=False):
    """Oldest request anchors every batch; defer at most four inspected peers.

    Defaults retain the original adjacent-pair policy. Opt-in lookahead can
    bypass an incompatible peer once, but that oldest deferred peer anchors
    the next batch. The 80 ms deadline is shared across all queue reads.
    Responses still map to the selected tasks' own Futures, never queue indices.
    """
    validate_batch_options(max_batch_size, batch_lookahead)
    validate_runtime_options(24, token_budget, allow_prompt_only)
    item["text"], item["length"] = generator.render(item["body"])
    batch, deferred, failed = [item], [], 0
    group = generation_group(item["body"], allow_prompt_only=allow_prompt_only)
    if group is None:
        return batch, failed
    deadline = time.monotonic() + .08
    try:
        for _ in range(batch_lookahead):
            if len(batch) >= max_batch_size:
                break
            try:
                other = pending.popleft() if pending else incoming.get(
                    timeout=max(0., deadline - time.monotonic()))
            except queue.Empty:
                break
            if other["future"].cancelled():
                continue
            try:
                other["text"], other["length"] = generator.render(other["body"])
                width = max(other["length"], *(task["length"] for task in batch))
                if (generation_group(other["body"], allow_prompt_only=allow_prompt_only) == group
                        and (width + group[0]) * (len(batch) + 1) <= token_budget):
                    batch.append(other)
                else:
                    deferred.append(other)
            except Exception as exc:
                try:
                    other["future"].set_exception(exc)
                except InvalidStateError:
                    pass  # HTTP cancellation can race with failed rendering.
                failed += 1
    finally:
        # Older unmatched peers stay ahead of uninspected pending/queued work.
        pending.extendleft(reversed(deferred))
    return batch, failed


class Engine:
    def __init__(self, *, max_batch_size=2, batch_lookahead=1, memory_gib=24,
                 token_budget=TOKEN_BUDGET, allow_prompt_only=False):
        validate_batch_options(max_batch_size, batch_lookahead)
        validate_runtime_options(memory_gib, token_budget, allow_prompt_only)
        self.max_batch_size, self.batch_lookahead = max_batch_size, batch_lookahead
        self.memory_gib, self.token_budget, self.allow_prompt_only = memory_gib, token_budget, allow_prompt_only
        from pbpk_kb.processing.evidence_json_constraints import (
            install_strict_object_separators, object_separator_compatibility,
            install_unicode_freetext_consumption, unicode_freetext_compatibility,
        )
        install_strict_object_separators()
        install_unicode_freetext_consumption()
        self.queue = queue.Queue(maxsize=4)
        self.state = {"state": "loading", "started_at": now(), "model": MODEL,
            "revision": REVISION, "backend": "project_transformers_qwen36_http",
            "quantization": "nf4_double_quant", "memory_cap_gib": memory_gib,
            "token_budget": token_budget, "allow_prompt_only": allow_prompt_only,
            "max_batch_size": max_batch_size, "batch_lookahead": batch_lookahead,
            "completed_requests": 0, "failed_requests": 0,
            "responded_requests": 0, "truncated_requests": 0, "invalid_requests": 0,
            "schema_const_compatibility": "scalar_json_literal_v1",
            **unicode_freetext_compatibility(),
            **object_separator_compatibility(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.lock = threading.Lock()
        self.persist()
        threading.Thread(target=self.work, daemon=True, name="project-evidence-qwen").start()

    def persist(self, **kwargs):
        with self.lock:
            self.state.update(kwargs, updated_at=now())
            write_json(RUNTIME / "status.json", self.state)

    def work(self):
        try:
            generator = EvidenceGenerator(memory_gib=self.memory_gib, token_budget=self.token_budget,
                allow_prompt_only=self.allow_prompt_only)
        except BaseException as exc:
            self.persist(state="failed", error_type=type(exc).__name__)
            logging.error("Project model loading failed (%s)", type(exc).__name__)
            return
        self.persist(state="ready")
        pending = deque()
        while True:
            item = pending.popleft() if pending else self.queue.get()
            if item["future"].cancelled():
                continue
            batch = [item]
            try:
                batch, preparation_failures = collect_batch(item, pending, self.queue, generator,
                    max_batch_size=self.max_batch_size, batch_lookahead=self.batch_lookahead,
                    token_budget=self.token_budget, allow_prompt_only=self.allow_prompt_only)
                if preparation_failures:
                    self.persist(failed_requests=self.state["failed_requests"] + preparation_failures)
                self.persist(active_batch=len(batch), state="generating")
                outputs, metrics = generator.generate(batch)
                for task, result in zip(batch, outputs, strict=True):
                    if not task["future"].cancelled():
                        task["future"].set_result(result)
                counts = generation_counts(outputs)
                self.persist(state="ready", active_batch=0, last_batch=metrics,
                    **{key: self.state[key] + value for key, value in counts.items()})
            except BaseException as exc:
                for task in batch:
                    if not task["future"].done():
                        task["future"].set_exception(exc)
                self.persist(state="ready", active_batch=0, last_error_type=type(exc).__name__,
                    failed_requests=self.state["failed_requests"] + len(batch))
                logging.error("Bounded evidence generation failed (%s)", type(exc).__name__)
                generator.torch.cuda.empty_cache()


class Handler(BaseHTTPRequestHandler):
    server_version = "PBPK-Evidence/1"

    def log_message(self, *args):
        pass

    def send(self, code, value):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        state = self.server.engine.state
        if self.path == "/health":
            return self.send(200 if state["state"] in {"ready", "generating"} else 503, state)
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self.send(404, {"error": "Unknown project endpoint"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self.send(404, {"error": "Unknown project endpoint"})
        engine = self.server.engine
        if engine.state["state"] not in {"ready", "generating"}:
            return self.send(503, {"error": "Project model is not ready"})
        future = Future()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= MAX_BODY:
                raise ValueError("Evidence request body exceeds bounds")
            self.connection.settimeout(15)
            body = validate_request(json.loads(self.rfile.read(length)))
            engine.queue.put_nowait({"body": body, "future": future})
            result = future.result(timeout=850)
            self.send(200, result)
        except queue.Full:
            self.send(429, {"error": "Project evidence queue is full"})
        except FutureTimeout:
            future.cancel()
            self.send(504, {"error": "Bounded evidence request timed out"})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send(422, {"error": type(exc).__name__})
        except BaseException as exc:
            self.send(500, {"error": type(exc).__name__})


def serve(*, max_batch_size=2, batch_lookahead=1, memory_gib=24,
          token_budget=TOKEN_BUDGET, allow_prompt_only=False):
    validate_batch_options(max_batch_size, batch_lookahead)
    validate_runtime_options(memory_gib, token_budget, allow_prompt_only)
    logging.basicConfig(level=logging.ERROR, handlers=[RotatingFileHandler(
        RUNTIME / "errors.log", maxBytes=2 * 1024**2, backupCount=1)])
    for port in range(18771, 18871):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("No available project loopback port")
    weights = json.loads(WEIGHTS.read_text())
    record = {"base_url": f"http://127.0.0.1:{port}/v1", "served_model": MODEL,
        "model_revision": REVISION, "model_config_sha256": weights["sha256"]["config.json"],
        "server_identity": identity(os.getpid()), "concurrent_requests": max_batch_size,
        "batch_lookahead": batch_lookahead,
        "memory_cap_gib": memory_gib, "token_budget": token_budget, "allow_prompt_only": allow_prompt_only,
        "request_timeout_seconds": 900, "backend_kind": "project_transformers_qwen36_http",
        "project_root": str(ROOT), "weights_manifest": str(WEIGHTS.relative_to(ROOT)),
        "created_at": now()}
    write_json(RUNTIME / "pid.json", record["server_identity"])
    write_json(RUNTIME / "service.json", record)
    # Preserve the prior registration once; no previous service is stopped.
    if CONFIG.exists() and not (RUNTIME / "previous_registration.json").exists():
        write_json(RUNTIME / "previous_registration.json", json.loads(CONFIG.read_text()))
    write_json(CONFIG, record)
    server.engine = Engine(max_batch_size=max_batch_size, batch_lookahead=batch_lookahead,
        memory_gib=memory_gib, token_budget=token_budget, allow_prompt_only=allow_prompt_only)
    server.serve_forever(poll_interval=.5)


def launch(*, max_batch_size=2, batch_lookahead=1, memory_gib=24,
           token_budget=TOKEN_BUDGET, allow_prompt_only=False):
    validate_batch_options(max_batch_size, batch_lookahead)
    validate_runtime_options(memory_gib, token_budget, allow_prompt_only)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    pid_file = RUNTIME / "pid.json"
    if pid_file.exists() and owned_alive(json.loads(pid_file.read_text())):
        print(json.dumps({"state": "already_running", "status": str(RUNTIME / "status.json")}))
        return
    env = {"PATH": str(ROOT / "runtime/judge-fast-env/bin") + ":/usr/bin:/bin",
        "LANG": "C.UTF-8", "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT / "scripts"),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(RUNTIME),
        "HF_HOME": str(ROOT / "runtime/hf_cache/frontier"),
        "HF_HUB_CACHE": str(ROOT / "runtime/hf_cache/frontier/hub"),
        "XDG_CACHE_HOME": str(RUNTIME / "cache"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "USE_HUB_KERNELS": "0", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0", "TRITON_CACHE_DIR": str(RUNTIME / "triton_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(RUNTIME / "torchinductor_cache")}
    with (RUNTIME / "startup.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([str(ROOT / "runtime/judge-fast-env/bin/python"), str(Path(__file__).resolve()), "serve",
            "--max-batch-size", str(max_batch_size), "--batch-lookahead", str(batch_lookahead),
            "--memory-gib", str(memory_gib), "--token-budget", str(token_budget)]
            + (["--allow-prompt-only"] if allow_prompt_only else []),
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True, close_fds=True)
    print(json.dumps({"state": "starting", "pid": child.pid, "status": str(RUNTIME / "status.json")}))


def watch_load():
    """Observe only this registered process; never overwrite engine state."""
    started = time.monotonic()
    previous = None
    while time.monotonic() - started < 1800:
        record = json.loads((RUNTIME / "pid.json").read_text())
        state = json.loads((RUNTIME / "status.json").read_text())
        alive = owned_alive(record)
        progress = {"checked_at": now(), "state": state["state"],
            "process_identity": record, "owned_process_alive": alive,
            "note": "Own-process I/O counters are activity measures, not download or completion percentages."}
        if alive:
            proc = Path("/proc") / str(record["pid"])
            io = {k: int(v) for k, v in (line.split(": ") for line in (proc / "io").read_text().splitlines())}
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            cpu_ticks = int(fields[11]) + int(fields[12])
            rss_kib = next(int(line.split()[1]) for line in (proc / "status").read_text().splitlines() if line.startswith("VmRSS:"))
            progress.update(process_read_bytes=io["read_bytes"], process_read_characters=io["rchar"],
                process_written_bytes=io["write_bytes"], resident_memory_kib=rss_kib, cpu_ticks=cpu_ticks)
            current = (io["rchar"], io["write_bytes"], cpu_ticks, rss_kib)
            progress["activity_since_previous_check"] = previous is None or current != previous
            previous = current
        write_json(RUNTIME / "loading_progress.json", progress)
        if not alive or state["state"] != "loading":
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "serve", "status", "watch-load"])
    parser.add_argument("--max-batch-size", type=int, choices=range(1, 5), default=2,
                        help="Startup-only batch cap; default preserves pairs.")
    parser.add_argument("--batch-lookahead", type=int, choices=range(1, 5), default=1,
                        help="Startup-only peer scan bound; default is adjacent only. Use 4 with a batch cap of 4.")
    parser.add_argument("--memory-gib", type=int, default=24,
                        help="Model process memory cap in GiB; retains the existing 2 GiB free-headroom check.")
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET,
                        help="Maximum padded input-plus-output tokens across a batch.")
    parser.add_argument("--allow-prompt-only", action="store_true",
                        help="Opt in to unconstrained decoding only for the exact schema {'type':'object'}.")
    args = parser.parse_args()
    if args.action == "start":
        launch(max_batch_size=args.max_batch_size, batch_lookahead=args.batch_lookahead,
            memory_gib=args.memory_gib, token_budget=args.token_budget, allow_prompt_only=args.allow_prompt_only)
    elif args.action == "serve":
        RUNTIME.mkdir(parents=True, exist_ok=True)
        serve(max_batch_size=args.max_batch_size, batch_lookahead=args.batch_lookahead,
            memory_gib=args.memory_gib, token_budget=args.token_budget, allow_prompt_only=args.allow_prompt_only)
    elif args.action == "watch-load":
        watch_load()
    else:
        status = json.loads((RUNTIME / "status.json").read_text()) if (RUNTIME / "status.json").exists() else {"state": "not_started"}
        pid = RUNTIME / "pid.json"
        status["owned_process_alive"] = owned_alive(json.loads(pid.read_text())) if pid.exists() else False
        print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
