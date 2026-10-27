from __future__ import annotations

import json
from contextlib import nullcontext
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_LOCAL_LLM = "Qwen/Qwen3-14B-AWQ"


@dataclass(frozen=True)
class StructuredGeneration:
    parsed: dict[str, Any] | None
    raw_text: str
    error: str | None


class LocalStructuredGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_LLM,
        *,
        max_input_tokens: int = 12_288,
        device: str | None = None,
        local_files_only: bool = False,
        dtype: str = "float16",
        reject_truncation: bool = False,
        enforce_json: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to run the local structured LLM.") from exc
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.reject_truncation = reject_truncation
        self.enforce_json = enforce_json
        self._compiled_json_grammar = None
        self._compiled_schema_key = None
        self._compiled_schema_grammar = None
        self._torch = torch
        if device and device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Local evidence screening requires the configured project GPU.")
        if local_files_only:
            torch.set_num_threads(4)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype),
            device_map={"": device} if device else "auto",
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._model.generation_config.do_sample = False
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None

    def generate_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        max_new_tokens: int = 768,
        json_schema: dict[str, Any] | None = None,
        sampling_seed: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> StructuredGeneration:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=not self.reject_truncation,
            max_length=self.max_input_tokens,
        )
        if inputs["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError("Evidence context exceeds token budget; split it instead of silently truncating it.")
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        sampling = {"do_sample": False}
        rng_context = nullcontext()
        if sampling_seed is not None:
            if not isinstance(sampling_seed, int) or not 0 <= sampling_seed < 2**32:
                raise ValueError("Recovery sampling seed must be a 32-bit unsigned integer")
            sampling = {"do_sample": True, "temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0.0}
            if repetition_penalty != 1.0:
                if json_schema is None or not 1.0 < repetition_penalty <= 1.2:
                    raise ValueError("Recovery repetition penalty requires a quote-constrained schema and must be <= 1.2")
                sampling["repetition_penalty"] = repetition_penalty
            devices = [model_device.index] if model_device.type == "cuda" else []
            rng_context = self._torch.random.fork_rng(devices=devices)
        with self._torch.inference_mode(), rng_context:
            if sampling_seed is not None:
                self._torch.manual_seed(sampling_seed)
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **sampling,
                pad_token_id=self._tokenizer.eos_token_id,
                **self._json_generation_kwargs(inputs["input_ids"].shape[1], max_new_tokens, json_schema),
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        raw_text = self._tokenizer.decode(generated, skip_special_tokens=True)
        if self._output_limit_reached(generated, max_new_tokens):
            return StructuredGeneration(None, raw_text, f"output_token_limit_reached:{max_new_tokens}")
        try:
            parsed = _extract_json_object(raw_text)
        except (ValueError, json.JSONDecodeError) as exc:
            return StructuredGeneration(parsed=None, raw_text=raw_text, error=f"{type(exc).__name__}:{exc}")
        return StructuredGeneration(parsed=parsed, raw_text=raw_text, error=None)

    def generate_json_batch(
        self, items: list[tuple[str, dict[str, Any]]], *, max_new_tokens: int = 768,
        json_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> list[StructuredGeneration]:
        """Generate independent requests together; padding never becomes context."""
        if not items:
            return []
        if isinstance(json_schema, list) and (len(json_schema) != len(items) or not all(isinstance(item, dict) for item in json_schema)):
            raise ValueError("Per-request schemas must match every batch row")
        prompts = []
        for system_prompt, payload in items:
            prompts.append(self._tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
            ], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        previous_padding = self._tokenizer.padding_side
        try:
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        finally:
            self._tokenizer.padding_side = previous_padding
        width = inputs["input_ids"].shape[1]
        if width > self.max_input_tokens:
            raise ValueError("Evidence context exceeds token budget; split it instead of silently truncating it.")
        device = next(self._model.parameters()).device
        results = []
        # Bound padded input + maximum output tokens, including for unusually
        # long contexts. Tokenization stays on CPU until each group is ready.
        rows_per_group = max(1, min(16, 98304 // (width + max_new_tokens)))
        for start in range(0, len(items), rows_per_group):
            group = {key: value[start:start+rows_per_group].to(device) for key, value in inputs.items()}
            group_schema = json_schema[start:start+rows_per_group] if isinstance(json_schema, list) else json_schema
            with self._torch.inference_mode():
                outputs = self._model.generate(**group, max_new_tokens=max_new_tokens,
                                              do_sample=False, pad_token_id=self._tokenizer.pad_token_id,
                                              **self._json_generation_kwargs(width, max_new_tokens, group_schema))
            for output in outputs:
                generated = output[width:]
                raw_text = self._tokenizer.decode(generated, skip_special_tokens=True)
                if self._output_limit_reached(generated, max_new_tokens):
                    results.append(StructuredGeneration(None, raw_text, f"output_token_limit_reached:{max_new_tokens}"))
                    continue
                try:
                    parsed = _extract_json_object(raw_text)
                    results.append(StructuredGeneration(parsed, raw_text, None))
                except (ValueError, json.JSONDecodeError) as exc:
                    results.append(StructuredGeneration(None, raw_text, f"{type(exc).__name__}:{exc}"))
            del outputs, group
        return results

    def _json_generation_kwargs(self, prompt_width: int, token_budget: int, json_schema: dict[str, Any] | list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not getattr(self, "enforce_json", False):
            return {}
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        if json_schema is not None:
            key = json.dumps(json_schema, ensure_ascii=False, separators=(",", ":"))
            if getattr(self, "_compiled_schema_key", None) != key:
                info = xgr.TokenizerInfo.from_huggingface(self._tokenizer, vocab_size=self._model.config.vocab_size)
                compiler = xgr.GrammarCompiler(info, max_threads=4)
                schemas = json_schema if isinstance(json_schema, list) else [json_schema]
                if not 1 <= len(schemas) <= 16:
                    raise ValueError("One generation may compile at most 16 request schemas")
                grammars = [compiler.compile_json_schema(schema, strict_mode=False,
                            any_whitespace=True, max_whitespace_cnt=2) for schema in schemas]
                self._compiled_schema_grammar = grammars if isinstance(json_schema, list) else grammars[0]
                self._compiled_schema_key = key
            return {"logits_processor": [LogitsProcessor(self._compiled_schema_grammar)]}
        if getattr(self, "_compiled_json_grammar", None) is None:
            info = xgr.TokenizerInfo.from_huggingface(
                self._tokenizer, vocab_size=self._model.config.vocab_size)
            compiler = xgr.GrammarCompiler(info, max_threads=4)
            self._compiled_json_grammar = compiler.compile_json_schema(
                {"type": "object", "additionalProperties": True}, strict_mode=False,
                any_whitespace=True, max_whitespace_cnt=2)
        # Only immutable grammar is cached. Each generation gets independent
        # row matchers; the existing output budget and scientific checks remain.
        return {"logits_processor": [LogitsProcessor(self._compiled_json_grammar)]}

    def _output_limit_reached(self, generated, budget: int) -> bool:
        if not getattr(self, "reject_truncation", False) or len(generated) < budget:
            return False
        config = getattr(self._model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None) or getattr(self._tokenizer, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        return not any(token in eos_ids for token in generated.tolist())


def _extract_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in model output.")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be one JSON object.")
    return parsed
