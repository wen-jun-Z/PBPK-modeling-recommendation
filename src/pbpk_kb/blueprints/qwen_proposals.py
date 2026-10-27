"""Outcome-blind, auditable local Qwen proposals for registered PBPK candidates.

The CPU contract is deliberately independent of Transformers. Generation is one
proposal per case/seed: malformed answers remain failures, without repair or a
rule fallback. References identify parameter provenance; they are not excerpts
and are never presented as evidence the language model has actually read.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time

VERSION = "pbpk-local-qwen-proposals-v2"
REPO = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SYSTEM = (
    "Choose one registered human PBPK configuration for the supplied intended use. "
    "Only listed constitutive methods and free-parameter subsets are legal. "
    "Keep human physiology and all fixed source parameters unchanged. "
    "A separate common numerical fitter DOES have real calibration-study concentration observations. "
    "After your proposal it will optimize the free parameters you select, within the supplied bounds "
    "and the same numerical fitting budget used for other methods. "
    "You do not see those observed values, selection errors, or test outcomes. "
    "The absence of observed values in this prompt does NOT mean calibration data are absent. "
    "Both fixed configurations (zero free parameters) and fitted configurations remain legal. "
    "Select using the intended use and parameter contracts; do not prefer more or fewer parameters by default. "
    "Separate selection-study observations are used only to compare completed candidates, never to fit parameters. "
    "Parameter bounds are search guards, not biological confidence intervals. "
    "Source references are provenance identifiers only, not supplied scientific excerpts. "
    "Do not claim to have read them or certify scientific validity. "
    "Return exactly one JSON object with candidate_id and a short rationale."
)
GENERATION = {
    "enable_thinking": False, "do_sample": True, "temperature": 0.2,
    "top_p": 0.9, "top_k": 20, "max_new_tokens": 256,
    "input_truncation": False, "max_input_tokens": 12000,
    "quantization": "bitsandbytes_nf4_double_quant", "dtype": "bfloat16",
    "syntax_constraint": "candidate_enum_json_schema", "repair_attempts": 0,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_path(value, project_root, *, must_exist=True):
    root = Path(project_root).resolve(strict=True)
    path = Path(value)
    path = (root / path).resolve(strict=must_exist) if not path.is_absolute() else path.resolve(strict=must_exist)
    if path == root or not path.is_relative_to(root):
        raise ValueError("Qwen artifact leaves the project")
    return path


def _only(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Unexpected fields in {label}")


def build_request(case, candidates):
    """Project only declared design inputs; never traverse study observations.

    Case loaders may hold calibration/selection data, but neither scenarios,
    source_audit, parameter-search results nor external references enter this
    whitelist. Parameter source paths are replaced with opaque checksum refs.
    """
    requirement = deepcopy(case.get("requirement") or case["design_request"]["requirement"])
    specs, refs = [], []
    for parameter in case["parameter_search"]["parameters"]:
        spec = {key: deepcopy(parameter[key]) for key in (
            "parameter_id", "initial_value", "lower_bound", "upper_bound", "unit",
            "search_scale", "bound_semantics") if key in parameter}
        source = parameter.get("source", {})
        checksum = source.get("source_checksum")
        if checksum:
            ref = {"ref_id": "parameter_source_" + str(len(refs) + 1),
                   "sha256": checksum, "role": "parameter_provenance",
                   "content_supplied": False}
            refs.append(ref)
            spec["source_ref_ids"] = [ref["ref_id"]]
        else:
            spec["source_ref_ids"] = []
        specs.append(spec)
    payload = {
        "version": VERSION, "case_id": case["case_id"],
        "compound_id": case["compound_id"], "requirement": requirement,
        "parameter_contracts": specs, "source_refs": refs,
        "candidates": [{key: deepcopy(candidate[key]) for key in (
            "candidate_id", "structure_id", "free_parameter_ids", "structure_edit_count",
            "structure_semantic_hash", "parameterization_hash") if key in candidate}
            for candidate in candidates],
    }
    for source, target in zip(candidates, payload['candidates'], strict=True):
        if source.get('native_mechanism_search'):
            linear = 'mm_low_concentration_first_order' in source['structure_id']
            target['kinetics'] = {
                'law': 'low_concentration_first_order' if linear else 'source_kinetics',
                'approximation_condition': 'local_unbound_substrate_over_Km <= 0.1' if linear else None,
                'clinical_applicability_verified': False,
                'anatomy_changed': False,
            }
    validate_request(payload)
    return payload


def validate_request(request):
    _only(request, {"version", "case_id", "compound_id", "requirement", "parameter_contracts",
                    "source_refs", "candidates"}, "proposal request")
    if request.get("version") != VERSION:
        raise ValueError("Unsupported proposal request version")
    if (not isinstance(request.get("case_id"), str) or request["case_id"] in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", request["case_id"])):
        raise ValueError("Unsafe or empty case ID")
    if not isinstance(request.get("compound_id"), str) or not request["compound_id"]:
        raise ValueError("Missing compound ID")
    requirement = request["requirement"]
    _only(requirement, {"compound_semantic_id", "name", "population", "route", "targets",
                        "dose", "dose_unit", "formulation", "species"}, "requirement")
    if not {"compound_semantic_id", "name", "route", "targets", "dose", "dose_unit"} <= set(requirement):
        raise ValueError("Incomplete intended-use requirement")
    for key, value in requirement.items():
        if key not in {"targets", "dose"} and value is not None and not isinstance(value, str):
            raise ValueError("Intended-use descriptors must be text, not embedded outcome objects")
    if type(requirement["dose"]) not in (int, float) or not math.isfinite(requirement["dose"]) or requirement["dose"] <= 0:
        raise ValueError("Intended-use dose must be finite and positive")
    if not isinstance(requirement["targets"], list) or not requirement["targets"]:
        raise ValueError("At least one intended-use target is required")
    for target in requirement["targets"]:
        _only(target, {"target_id", "analyte", "matrix", "quantity", "binding", "required"}, "target")
        if any(not isinstance(value, bool if key == "required" else str) for key, value in target.items()):
            raise ValueError("Target descriptors must have scalar contract types")
    refs = request["source_refs"]
    if len({r["ref_id"] for r in refs}) != len(refs):
        raise ValueError("Duplicate source reference")
    for ref in refs:
        _only(ref, {"ref_id", "sha256", "role", "content_supplied"}, "source reference")
        if (ref["role"] != "parameter_provenance" or ref["content_supplied"] is not False
                or not re.fullmatch(r"[a-f0-9]{64}", ref["sha256"])):
            raise ValueError("Only checksum-bound parameter provenance is accepted")
    specs = request["parameter_contracts"]
    if not specs or len({p["parameter_id"] for p in specs}) != len(specs):
        raise ValueError("Missing or duplicate parameter contract")
    ids = {p["parameter_id"] for p in specs}
    for spec in specs:
        _only(spec, {"parameter_id", "initial_value", "lower_bound", "upper_bound", "unit",
                     "search_scale", "bound_semantics", "source_ref_ids"}, "parameter contract")
        values = [spec[k] for k in ("lower_bound", "initial_value", "upper_bound")]
        if not all(type(v) in (float, int) and math.isfinite(v) for v in values) or not values[0] <= values[1] <= values[2] or values[0] == values[2]:
            raise ValueError("Invalid source parameter bounds")
        if set(spec["source_ref_ids"]) - {r["ref_id"] for r in refs}:
            raise ValueError("Unknown source reference")
        if any(not isinstance(spec[key], str) for key in ("parameter_id", "unit", "search_scale", "bound_semantics") if key in spec):
            raise ValueError("Parameter descriptors must be text")
    candidates = request["candidates"]
    if not candidates or len(candidates) > 256 or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Missing, oversized or duplicate candidate set")
    for candidate in candidates:
        _only(candidate, {"candidate_id", "structure_id", "free_parameter_ids", "structure_edit_count",
                          "structure_semantic_hash", "parameterization_hash", "kinetics"}, "candidate")
        if (not isinstance(candidate["candidate_id"], str) or not candidate["candidate_id"]
                or set(candidate["free_parameter_ids"]) - ids
                or len(set(candidate["free_parameter_ids"])) != len(candidate["free_parameter_ids"])):
            raise ValueError("Invalid registered candidate")
        if 'kinetics' in candidate:
            kinetics = candidate['kinetics']
            _only(kinetics, {'law', 'approximation_condition', 'clinical_applicability_verified', 'anatomy_changed'}, 'kinetics')
            linear = 'mm_low_concentration_first_order' in candidate['structure_id']
            expected = {'law':'low_concentration_first_order' if linear else 'source_kinetics',
                        'approximation_condition':'local_unbound_substrate_over_Km <= 0.1' if linear else None,
                        'clinical_applicability_verified':False, 'anatomy_changed':False}
            if kinetics != expected:
                raise ValueError('Native kinetic descriptors must match the registered constitutive edit')
    canonical(request)  # Nested NaN/infinity is forbidden too.
    return request


def proposal_schema(request):
    validate_request(request)
    return {"type": "object", "additionalProperties": False,
            "required": ["candidate_id", "rationale"], "properties": {
                "candidate_id": {"type": "string", "enum": [c["candidate_id"] for c in request["candidates"]]},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 1200}}}


def messages(request):
    validate_request(request)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(request)}]


def parse_response(raw, request):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate answer key")
            result[key] = value
        return result
    parsed = json.loads(raw, object_pairs_hook=no_duplicates)
    if not isinstance(parsed, dict) or set(parsed) != {"candidate_id", "rationale"}:
        raise ValueError("Proposal must contain exactly candidate_id and rationale")
    if parsed["candidate_id"] not in {c["candidate_id"] for c in request["candidates"]}:
        raise ValueError("Proposal selected an unregistered candidate")
    if not isinstance(parsed["rationale"], str) or not 1 <= len(parsed["rationale"].strip()) <= 1200:
        raise ValueError("Missing or oversized proposal rationale")
    return parsed


def verify_weights(manifest_path, *, project_root):
    """Verify every recorded artifact once before GPU loading, without network."""
    manifest_path = project_path(manifest_path, project_root)
    weights = json.loads(manifest_path.read_text())
    if (weights.get("repo") != REPO or weights.get("revision") != REVISION
            or not weights.get("official_lfs_hashes_checked") or weights.get("trust_remote_code") is not False):
        raise ValueError("Unreviewed Qwen weight identity")
    snapshot = project_path(weights["path"], project_root)
    shards = [name for name in weights["sha256"] if name.endswith(".safetensors")]
    if len(shards) != 15:
        raise ValueError("Incomplete Qwen weight manifest")
    for name, expected in weights["sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid weight artifact name")
        artifact = project_path(snapshot / name, project_root)
        if file_sha(artifact) != expected:
            raise ValueError("Qwen weight artifact checksum mismatch: " + name)
    return weights, file_sha(manifest_path)


def read_frozen_record(directory, request, *, seed, project_root):
    """Read one bound record for audited cross-batch reuse, preserving failures."""
    directory = project_path(directory, project_root)
    manifest = json.loads(project_path(directory / "manifest.json", project_root).read_text())
    if (manifest.get("version") != VERSION or manifest.get("model_repo") != REPO
            or manifest.get("model_revision") != REVISION or manifest.get("generation") != GENERATION
            or manifest.get("final_status") not in {"completed", "completed_with_failed_proposals"}
            or manifest.get("requests", {}).get(request["case_id"]) != digest(request)):
        raise ValueError("Frozen Qwen reuse contract differs")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Proposal seed must be a nonnegative 31-bit integer")
    relative = f"cases/{request['case_id']}/seed_{seed}.json"
    path = project_path(directory / relative, project_root)
    if file_sha(path) != manifest.get("record_sha256", {}).get(relative):
        raise ValueError("Frozen proposal record changed")
    record = json.loads(path.read_text())
    if (record.get("case_id") != request["case_id"] or record.get("seed") != seed
            or record.get("request_sha256") != digest(request)
            or record.get("weights_manifest_sha256") != manifest["weights_manifest_sha256"]):
        raise ValueError("Frozen proposal record binding differs")
    if record.get("status") == "completed":
        parsed = parse_response(record["raw_response"], request)
        if parsed["candidate_id"] != record.get("candidate_id"):
            raise ValueError("Frozen raw answer differs from parsed candidate")
    return record, {"source_directory": str(directory.relative_to(Path(project_root).resolve())),
                    "source_manifest_sha256": file_sha(directory / "manifest.json"),
                    "source_record_sha256": file_sha(path), "source_record": relative}


def load_proposer(directory, case, candidates, *, project_root):
    """Return a CPU callback that verifies and consumes frozen local proposals."""
    directory = project_path(directory, project_root)
    manifest = json.loads(project_path(directory / "manifest.json", project_root).read_text())
    request = build_request(case, candidates)
    binding = digest(request)
    if (manifest.get("version") != VERSION or manifest.get("model_repo") != REPO
            or manifest.get("model_revision") != REVISION
            or manifest.get("final_status") not in {"completed", "completed_with_failed_proposals"}
            or manifest.get("generation") != GENERATION
            or manifest.get("requests", {}).get(case["case_id"]) != binding):
        raise ValueError("Frozen Qwen proposal inputs, model or protocol do not match")

    def propose(active_candidates, *, seed):
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("Proposal seed must be a nonnegative 31-bit integer")
        active = build_request(case, active_candidates)
        # The common search runner sorts candidate and free-parameter IDs. This
        # is an ordering normalization, not a change in the proposal contract.
        def normalized(payload):
            payload = deepcopy(payload)
            payload["candidates"].sort(key=lambda c: c["candidate_id"])
            for item in payload["candidates"]:
                item["free_parameter_ids"].sort()
            return payload
        if digest(normalized(active)) != digest(normalized(request)):
            raise ValueError("Qwen candidate contract changed")
        relative = "cases/" + case["case_id"] + "/seed_" + str(int(seed)) + ".json"
        path = project_path(directory / relative, project_root)
        if file_sha(path) != manifest.get("record_sha256", {}).get(relative):
            raise ValueError("Qwen generation record is absent or changed")
        record = json.loads(path.read_text())
        if (record.get("seed") != seed or record.get("case_id") != case["case_id"]
                or record.get("request_sha256") != binding
                or record.get("weights_manifest_sha256") != manifest["weights_manifest_sha256"]):
            raise ValueError("Qwen generation binding mismatch")
        if record.get("status") != "completed":
            raise ValueError("Qwen proposal failed: " + str(record.get("error_type", record.get("status"))))
        parsed = parse_response(record["raw_response"], request)
        if record.get("candidate_id") != parsed["candidate_id"]:
            raise ValueError("Parsed Qwen proposal differs from preserved raw response")
        return parsed

    return propose


class LocalQwenProposer:
    """One resident GPU model, using the existing verified local fast kernels."""

    def __init__(self, weights, *, memory_gib=36):
        from frontier_reasoned_fast_judge import verify_fast_environment
        from frontier_modern_judge import ModernGPUJudge
        self.fast_environment = verify_fast_environment()
        self.backend = ModernGPUJudge(weights, memory_gib=memory_gib)
        self.metadata = dict(self.backend.metadata, fast_environment=self.fast_environment,
                             decoding="sampled_candidate_enum_json_schema", task=VERSION)

    def render(self, request):
        return self.backend.tokenizer.apply_chat_template(messages(request), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)

    def batch_size(self, requests, *, maximum=8, token_budget=24000):
        texts = [self.render(r) for r in requests[:maximum]]
        lengths = [len(ids) for ids in self.backend.tokenizer(texts, truncation=False)["input_ids"]]
        width = count = 0
        for length in lengths:
            if length > GENERATION["max_input_tokens"]:
                raise ValueError("Complete Qwen input exceeds limit; truncation refused")
            width = max(width, length)
            if (width + GENERATION["max_new_tokens"]) * (count + 1) > token_budget:
                break
            count += 1
        return max(1, count)

    def generate(self, requests, *, seed):
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        backend, torch = self.backend, self.backend.torch
        callbacks = [build_transformers_prefix_allowed_tokens_fn(
            backend.json_constraint.tokenizer_data,
            JsonSchemaParser(proposal_schema(r), config=CharacterLevelParserConfig(
                max_consecutive_whitespaces=2, force_json_field_order=False))) for r in requests]
        texts = [self.render(r) for r in requests]
        inputs = backend.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        width = inputs["input_ids"].shape[1]
        if width > GENERATION["max_input_tokens"]:
            raise ValueError("Complete Qwen input exceeds limit; truncation refused")
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            generated = backend.model.generate(**inputs, **{k: GENERATION[k] for k in (
                "do_sample", "temperature", "top_p", "top_k", "max_new_tokens")},
                use_cache=True, max_time=300,
                pad_token_id=backend.tokenizer.pad_token_id,
                eos_token_id=backend.model.generation_config.eos_token_id,
                prefix_allowed_tokens_fn=lambda batch_id, tokens: callbacks[batch_id](batch_id, tokens))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        values = backend.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        metrics = {"wall_seconds": elapsed, "input_tokens": lengths,
                   "input_token_width": width, "output_token_width": generated.shape[1] - width,
                   "batch_size": len(requests), "seed": seed,
                   "batch_case_ids": [r["case_id"] for r in requests],
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                   "generation_time_limit_seconds": 300,
                   "time_limit_reached": elapsed >= 300,
                   "gpu_memory_cap_gib": backend.memory_cap_gib, **self.metadata}
        del inputs, generated
        return values, metrics
