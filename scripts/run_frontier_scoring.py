"""Validated, isolated Qwen scoring queue over the already frozen rankings.

Controlled checks certify a limited execution contract, never expert PBPK gold.
No method score participates in judge selection or source judgments.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import run_frontier_assessment as base

ROOT = base.ROOT
OUT = base.OUT
OLD = OUT / "assessment"
ASSESS = OUT / "assessment_qwen36"
RUN = ROOT / "runtime/deployment/frontier_scoring"
APP = ROOT / "runtime/app-env/bin/python"
MODERN = ROOT / "runtime/judge-env/bin/python"
QWEN_WEIGHTS = ASSESS / "model_weights/qwen.json"
PHI_WEIGHTS = OLD / "primary_weights.json"
BACKEND_RUN = ROOT / "runtime/deployment/frontier_qwen36_backend"
VERSION = "pbpk-qwen36-document-scoring-v1"
PROFILE = "v1"
V1_ASSESS = OUT / "assessment_qwen36"
V2_ASSESS = OUT / "assessment_qwen36_v2"
PROFILE_DIRECTORIES = {
    "v1": ("assessment_qwen36", "frontier_scoring"),
    "v2": ("assessment_qwen36_v2", "frontier_scoring_v2"),
    "validated_v2": ("assessment_validated_v2", "frontier_scoring_validated_v2"),
}
V1_SYSTEM = base.SYSTEM
LEGACY_GPU = base.GPUJudge
GATE = {"controlled_semantic_minimum": .90,
        "controlled_citation_support_minimum": .95,
        "citation_numbers_valid_minimum": .98,
        "maximum_semantic_drop_against_phi": .02,
        "raw_format_valid_minimum": .98,
        "meaning": "Engineering/controlled reasoning admission only, NOT scientific accuracy validation"}


def configure_profile(profile="v1"):
    """Select an isolated run before any file writes or worker construction."""
    global PROFILE, ASSESS, RUN, VERSION, QWEN_WEIGHTS
    if profile not in PROFILE_DIRECTORIES:
        raise ValueError("Unknown scoring profile")
    if profile in ("v2", "validated_v2"):
        from frontier_judge_prompt_v2 import SYSTEM
    else:
        SYSTEM = V1_SYSTEM
    PROFILE = profile
    assessment_name, runtime_name = PROFILE_DIRECTORIES[profile]
    ASSESS = OUT / assessment_name
    RUN = ROOT / "runtime/deployment" / runtime_name
    VERSION = "pbpk-qwen36-document-scoring-" + profile
    QWEN_WEIGHTS = V1_ASSESS / "model_weights/qwen.json"
    base.SYSTEM = SYSTEM
    return {"profile": profile, "assessment": ASSESS, "runtime": RUN}


def worker_command(python, *arguments):
    return [str(python), str(Path(__file__).resolve()), *arguments, "--profile", PROFILE]


def selected_models():
    return ("phi", "qwen") if PROFILE == "validated_v2" else ("qwen", "phi")


def weights_manifest(model):
    return PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS


def execution_environments():
    """Record both real interpreters; Qwen's ready file is not Phi's environment."""
    code = (
        "import json,sys,platform,importlib.metadata as m\n"
        "versions={}\n"
        "for name in ['torch','transformers','accelerate','bitsandbytes','tokenizers',"
        "'safetensors','lm-format-enforcer','fla-core','triton']:\n"
        " try: versions[name]=m.version(name)\n"
        " except m.PackageNotFoundError: versions[name]=None\n"
        "print(json.dumps({'executable':sys.executable,'python':platform.python_version(),"
        "'versions':versions}))\n"
    )
    result = {}
    for model, python in (("phi", APP), ("qwen", MODERN)):
        completed = subprocess.run([str(python), "-c", code], cwd=ROOT,
            env=dict(os.environ), check=True, text=True, capture_output=True, timeout=30)
        result[model] = json.loads(completed.stdout)
    return result


def link_validation_records():
    """Reference fixed v2 pilots; never link a writable new admission output."""
    folder = ASSESS / "validation"
    if folder.is_symlink() or folder.resolve() == (V2_ASSESS / "validation").resolve():
        raise ValueError("Validated validation directory must be independent, never a directory symlink")
    for name in ("cases.jsonl", "manifest.json", "phi_records.jsonl", "qwen_records.jsonl",
                 "phi_protocol.json", "qwen_protocol.json", "phi_evaluation.json", "qwen_evaluation.json"):
        source, target = V2_ASSESS / "validation" / name, ASSESS / "validation" / name
        if not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise ValueError("Complete project-owned v2 validation records required")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent))
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError("Validated profile must reference unchanged v2 pilot artifacts")


def validate_selection_policy():
    path = ASSESS / "selection_policy.json"
    policy = base.read(path)
    expected = {"profile": "validated_v2", "primary_model": "phi", "secondary_model": "qwen",
                "thresholds": GATE,
                "predecessor_admission_sha256": base.file_sha(V2_ASSESS / "validation/admission.json")}
    if not policy or any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen selection policy missing or inconsistent with the unchanged gate and v2 outcome")
    previous = base.read(V2_ASSESS / "judge_protocol.json")
    if (previous.get("system_prompt_sha256") != base.sha(base.SYSTEM.encode()) or
            previous.get("validation_manifest_sha256") != base.file_sha(ASSESS / "validation/manifest.json") or
            previous.get("primary_weights_manifest_sha256") != base.file_sha(QWEN_WEIGHTS) or
            previous.get("secondary_weights_manifest_sha256") != base.file_sha(PHI_WEIGHTS)):
        raise ValueError("V2 pilot source protocol, prompt or model artifacts changed")
    return base.file_sha(path)


def v1_runner_snapshot_sha():
    expected = base.read(V1_ASSESS / "judge_protocol.json")["implementation_sha256"]["run_frontier_scoring.py"]
    snapshot = V1_ASSESS / "reproducibility/run_frontier_scoring.py"
    if not snapshot.exists() or base.file_sha(snapshot) != expected:
        raise ValueError("Preserved v1 runner does not match its frozen protocol")
    return expected


def prepare_validation(qs, cs, packets):
    if PROFILE == "validated_v2":
        link_validation_records()
    if PROFILE in ("v2", "validated_v2"):
        from frontier_judge_validation_v2 import verify_validation_v2
        return verify_validation_v2(qs, cs, packets, ASSESS / "validation")
    from frontier_judge_validation import build_validation
    return build_validation(qs, cs, packets, ASSESS / "validation")


def ensure_execution_allowed():
    if PROFILE == "v1" and base.read(V1_ASSESS / "validation/admission.json", {}).get("accepted") is False:
        raise RuntimeError("Preserved v1 admission failed; new execution requires explicit --profile v2")
    if PROFILE == "v2" and base.read(V2_ASSESS / "validation/admission.json", {}).get("accepted") is False:
        raise RuntimeError("Preserved v2 Qwen admission failed; use the separately declared validated_v2 selection")


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    ASSESS.mkdir(parents=True, exist_ok=True)
    safe = base.environment()
    safe.update(TMPDIR=str(RUN), TRITON_CACHE_DIR=str(BACKEND_RUN / "triton_cache"),
                TORCHINDUCTOR_CACHE_DIR=str(BACKEND_RUN / "torchinductor_cache"),
                TORCH_HOME=str(RUN / "torch_cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.clear()
    os.environ.update(safe)


def freeze(path, value):
    if path.exists():
        if base.read(path) != value:
            raise ValueError(f"Frozen specification differs: {path.name}")
    else:
        # Writers are serialized by the pipeline lock.
        base.dump(path, value)
    return value


def link_source(name):
    source, target = OLD / name, ASSESS / name
    if not source.resolve().is_relative_to(ROOT) or not source.exists():
        raise ValueError("Missing project-owned source")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())
    if target.resolve() != source.resolve():
        raise ValueError("Assessment source redirect changed")


def prepare_materials():
    for name in ("sources.jsonl", "source_supplements", "measured_outcomes.json"):
        link_source(name)
    qs = base.read_rows(OUT / "inputs/queries.jsonl")
    cs = base.read_rows(OUT / "inputs/candidates.jsonl")
    old_protocol = base.read(OLD / "judge_protocol.json")
    for name in ("inputs/queries.jsonl", "inputs/candidates.jsonl", "assessment/sources.jsonl",
                 "assessment/source_supplements/sources.jsonl"):
        if base.file_sha(OUT / name) != old_protocol["input_sha256"][name]:
            raise ValueError("Frozen source or request changed")
    if len(qs) != 772 or len(cs) != 280:
        raise ValueError("Full frozen cohort required")
    sources = {row["candidate_id"]: row for row in base.read_rows(ASSESS / "sources.jsonl")}
    if len(sources) != 280 or set(sources) != {c["candidate_id"] for c in cs}:
        raise ValueError("Source coverage mismatch")
    for supplement in base.read_rows(ASSESS / "source_supplements/sources.jsonl"):
        cid = supplement["candidate_id"]
        original = sources[cid]
        if original.get("documents") or not supplement.get("documents"):
            raise ValueError("Unexpected source replacement")
        if any(original.get(k) != supplement.get(k) for k in
               ("family_id", "family_name", "compounds", "source_artifact")):
            raise ValueError("Supplement identity mismatch")
        sources[cid] = supplement
    packets = {cid: base.build_packet(row) for cid, row in sources.items()}
    return qs, cs, packets, old_protocol


def verify_weights(path):
    weights = base.read(path)
    if not weights:
        raise ValueError("Weights manifest not ready")
    directory = Path(weights["path"])
    if not directory.resolve().is_relative_to(ROOT) or not weights.get("sha256"):
        raise ValueError("Unscoped or unverified weights")
    for name, digest in weights["sha256"].items():
        file = directory / name
        if not file.resolve().is_relative_to(ROOT) or base.file_sha(file) != digest:
            raise ValueError("Model artifact integrity failed")
    return weights


def prepare_protocol():
    ensure_execution_allowed()
    qs, cs, packets, old = prepare_materials()
    if PROFILE in ("v2", "validated_v2"):
        prepare_validation(qs, cs, packets)
    qwen, phi = base.read(QWEN_WEIGHTS), base.read(PHI_WEIGHTS)
    if not qwen or not phi:
        raise ValueError("Pinned weights required before freezing judge protocol")
    primary_model, secondary_model = selected_models()
    weights = {"qwen": qwen, "phi": phi}
    primary, secondary = weights[primary_model], weights[secondary_model]
    modes = {"qwen": "instruct, enable_thinking=False; not extended reasoning",
             "phi": "Phi-4 instruction model; constrained JSON"}
    protocol = copy.deepcopy(old)
    protocol.update(version=VERSION, primary_repo=primary["repo"], primary_revision=primary["revision"],
                    secondary_repo=secondary["repo"], secondary_revision=secondary["revision"],
                    primary_mode=modes[primary_model], secondary_mode=modes[secondary_model],
                    hardware="Single project-owned worker at a time; aggregate free-memory guard, 36 GiB allocator cap; no claim of verified hardware exclusivity",
                    predecessor_protocol_sha256=base.file_sha(OLD / "judge_protocol.json"),
                    predecessor_labels_reused=False,
                    admission_checks=GATE,
                    validation_manifest_sha256=base.file_sha(ASSESS / "validation/manifest.json"),
                    validation_role="Fixed controlled tasks and unlabelled real source pairs; model selection data, not independent scientific gold",
                    primary_weights_manifest_sha256=base.file_sha(weights_manifest(primary_model)),
                    secondary_weights_manifest_sha256=base.file_sha(weights_manifest(secondary_model)),
                    backend_readiness_sha256=base.file_sha(BACKEND_RUN / "ready.json"),
                    backend_environment=base.read(BACKEND_RUN / "ready.json").get("environment_versions", {}),
                    implementation_sha256={name: base.file_sha(ROOT / "scripts" / name) for name in
                        ("run_frontier_scoring.py", "run_frontier_assessment.py", "frontier_modern_judge.py",
                         "frontier_judge_validation.py", "frontier_judge_decoding.py")})
    protocol["judge_development"]["additional_controls"] = "validation/manifest.json"
    protocol["judge_development"]["additional_real_source_pairs"] = "Unlabelled, selected without recommender scores; included in full scoring and flagged as judge-development examples"
    pilot_cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    protocol["judge_development"]["generated_query_ids"] = sorted(
        set(protocol["judge_development"].get("generated_query_ids", [])) |
        {c["query"]["query_id"] for c in pilot_cases if c["kind"] == "real_source_pair"})
    if PROFILE in ("v2", "validated_v2"):
        protocol.update(system_prompt=base.SYSTEM, system_prompt_sha256=base.sha(base.SYSTEM.encode()),
            previous_attempt_protocol_sha256=base.file_sha(V1_ASSESS / "judge_protocol.json"),
            validation_role="64 independent admission controls and 64 development regression controls evaluated separately; 60 unlabelled real pairs; unchanged thresholds; not expert gold",
            admission_group_policy="Both independent_admission and development_regression must independently pass every unchanged threshold; their rates are never pooled",
            predecessor_labels_reused=False)
        protocol["implementation_sha256"].update({name: base.file_sha(ROOT / "scripts" / name) for name in
            ("frontier_judge_prompt_v2.py", "frontier_judge_validation_v2.py")})
    if PROFILE == "validated_v2":
        protocol.update(selection_policy_sha256=validate_selection_policy(),
            previous_attempt_protocol_sha256=base.file_sha(V2_ASSESS / "judge_protocol.json"),
            selected_primary="phi", secondary_role="Qwen fixed 5% sensitivity only; failed independent primary admission; never gold",
            selection_basis="Phi independently passes the unchanged gate in both control roles, including the same 0.02 tolerance against Qwen; no recommender scores inspected",
            reused_pilot_record_sha256={model: base.file_sha(ASSESS / "validation" / f"{model}_records.jsonl") for model in ("phi", "qwen")},
            execution_environments=execution_environments(),
            backend_environment_scope="backend_environment above describes the Qwen secondary backend; execution_environments records the actual Phi and Qwen interpreters separately")
    freeze(ASSESS / "judge_protocol.json", protocol)
    base.dump(ASSESS / "assessment_manifest.json", {
        "queries": len(qs), "candidates": len(cs), "expected_pairs": len(qs)*len(cs),
        "distinct_material_packets": len({p["packet_sha256"] for p in packets.values()}),
        "candidates_without_usable_excerpts": sum(not p["refs"] for p in packets.values()),
        "protocol_sha256": base.file_sha(ASSESS / "judge_protocol.json"),
        "source_file_sha256": base.file_sha(ASSESS / "sources.jsonl"),
        "primary_repo": primary["repo"], "secondary_repo": secondary["repo"]})
    return qs, cs, packets, protocol


class PhiJudge(LEGACY_GPU):
    def __init__(self, weights, memory_gib=36, loading_callback=None):
        super().__init__(weights, memory_gib, loading_callback)
        self.prompt_batch_limit = 16
        self.batch_token_budget = 48000

    def maybe_expand_capacity(self):
        return None


def backend(model):
    if model == "phi":
        return PhiJudge
    from frontier_modern_judge import ModernGPUJudge
    return ModernGPUJudge


def status_update(path, snapshot, **values):
    snapshot.update(values, updated_at=base.now())
    base.dump(path, snapshot)


def pilot(model):
    ensure_execution_allowed()
    if PROFILE == "validated_v2":
        raise RuntimeError("validated_v2 references completed v2 pilots; running or rewriting them is forbidden")
    from frontier_judge_validation import evaluate_validation
    if PROFILE == "v2":
        from frontier_judge_validation_v2 import evaluate_validation_v2 as evaluate_validation
        qs, cs, packets, _ = prepare_materials()
        prepare_validation(qs, cs, packets)
    from frontier_pilot_decoding import empty_source_constraint, pilot_decoding_protocol
    path = ASSESS / "validation" / f"{model}_records.jsonl"
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    state = {**base.identity(), "state": "verifying_weights", "model": model}
    status = RUN / f"pilot_{model}_status.json"
    status_update(status, state)
    weights = verify_weights(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS)
    pilot_batch_limit = 16 if PROFILE == "v2" else 8
    pilot_spec = {"model_repo": weights["repo"], "model_revision": weights["revision"],
        "weights_manifest_sha256": base.file_sha(PHI_WEIGHTS if model == "phi" else QWEN_WEIGHTS),
        "validation_sha256": base.file_sha(ASSESS / "validation/manifest.json"),
        "system_prompt_sha256": base.sha(base.SYSTEM.encode()),
        "legacy_judge_implementation_sha256": base.file_sha(Path(base.__file__)),
        "decoder_sha256": base.file_sha(ROOT / "scripts/frontier_judge_decoding.py"),
        "pilot_function_sha256": PILOT_FUNCTION_SHA,
        "pilot_constraint": pilot_decoding_protocol(),
        "empty_source_policy": "Actually call model; citations must be empty; all grades/applicability remain selectable",
        "backend_sha256": base.file_sha(ROOT / "scripts/frontier_modern_judge.py") if model == "qwen"
                           else PHI_ADAPTER_SHA,
        "generation": {"max_new_tokens": 96, "do_sample": False, "repairs": 0,
                       "maximum_batch_size": pilot_batch_limit, "memory_cap_gib": 36,
                       "qwen_enable_thinking": False},
        "scope_guard_applied": False, "purpose": "Unrepaired raw judge comparison"}
    freeze(ASSESS / "validation" / f"{model}_protocol.json", pilot_spec)
    pilot_hash = base.file_sha(ASSESS / "validation" / f"{model}_protocol.json")
    existing = base.load_committed(path, repair_tail=True)
    expected = {c["case_id"] for c in cases}
    by_id = {r["case_id"]: r for r in existing}
    if len(by_id) != len(existing) or not set(by_id) <= expected:
        raise ValueError("Pilot duplicates or unexpected cases")
    for row in existing:
        if (row["model_revision"] != weights["revision"] or row["validation_sha256"] != base.file_sha(ASSESS / "validation/manifest.json")
                or row.get("pilot_protocol_sha256") != pilot_hash):
            raise ValueError("Pilot provenance mismatch")
    pending = [c for c in cases if c["case_id"] not in by_id]
    if pending:
        judge = backend(model)(weights, 36, lambda: status_update(status, state, state="loading_gpu"))
        judge.json_constraint = empty_source_constraint(judge.json_constraint)
        status_update(status, state, state="judging", completed=len(existing), expected=len(cases))
        position, batch_max = 0, pilot_batch_limit
        while position < len(pending):
            if (RUN / "stop.requested").exists():
                status_update(status, state, state="stopped")
                return
            subset = pending[position:position+batch_max]
            tasks = [(c["packet"], [c["query"]]) for c in subset]
            size = judge.select_batch_size(tasks, batch_max)
            subset, tasks = subset[:size], tasks[:size]
            try:
                outputs, metrics = judge.generate(tasks)
            except judge.torch.cuda.OutOfMemoryError:
                judge.torch.cuda.empty_cache()
                if size == 1:
                    raise
                batch_max = max(1, size//2)
                continue
            if len(outputs) != len(subset):
                raise ValueError("Pilot output cardinality")
            records = []
            for case, output in zip(subset, outputs):
                row = {"case_id": case["case_id"], "kind": case["kind"], "output": output,
                    "model_repo": weights["repo"], "model_revision": weights["revision"],
                    "validation_sha256": base.file_sha(ASSESS / "validation/manifest.json"),
                    "pilot_protocol_sha256": pilot_hash,
                    "at": base.now(), "metrics": metrics}
                records.append(row)
                by_id[row["case_id"]] = row
            base.append_rows(path, records)
            position += len(records)
            status_update(status, state, completed=len(by_id), last_commit_at=base.now(), last_batch=metrics)
        del judge
    result = evaluate_validation(list(by_id.values()), cases)
    base.dump(ASSESS / "validation" / f"{model}_evaluation.json", result)
    status_update(status, state, state="complete", completed=len(by_id), expected=len(cases))


def run_full(name):
    ensure_execution_allowed()
    if PROFILE in ("v2", "validated_v2") and not check_gate()["accepted"]:
        raise RuntimeError("Both frozen v2 control groups must pass before full scoring")
    primary_model, secondary_model = selected_models()
    base.ASSESS, base.RUN = ASSESS, RUN
    base.prepare = prepare_protocol
    base.GPUJudge = backend(primary_model if name == "primary" else secondary_model)
    base.download_primary = lambda: verify_weights(weights_manifest(primary_model))
    base.run_assessment(SimpleNamespace(judge=name, memory_gib=36, batch_size=16,
        max_batches=0, secondary_weights=weights_manifest(secondary_model)))


def passes_gate(q, p):
    return bool(q["complete"] and p["complete"] and
        q["controlled_semantic_accuracy"] >= GATE["controlled_semantic_minimum"] and
        q["controlled_citation_support_accuracy"] >= GATE["controlled_citation_support_minimum"] and
        q["citation_numbers_valid_fraction"] >= GATE["citation_numbers_valid_minimum"] and
        q["controlled_semantic_accuracy"] >= p["controlled_semantic_accuracy"] - GATE["maximum_semantic_drop_against_phi"] and
        q["raw_format_valid_fraction"] >= GATE["raw_format_valid_minimum"])


def grouped_gate(records_by_model, cases, primary_model="qwen", reference_model="phi"):
    """Keep fresh admission and reused development controls separate."""
    from frontier_judge_validation import evaluate_validation, admission_metrics
    expected = {"development_regression": 64, "independent_admission": 64, "unlabelled_real_source": 60}
    if len(cases) != sum(expected.values()) or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("v2 requires the complete unique 188-case validation cohort")
    for role, count in expected.items():
        selected = [c for c in cases if c.get("strata", {}).get("validation_role") == role]
        if len(selected) != count:
            raise ValueError(f"Invalid fixed v2 validation role count: {role}")
    known = {c["case_id"] for c in cases}
    complete, engineering = {}, {}
    for model, records in records_by_model.items():
        ids = [r["case_id"] for r in records]
        if len(ids) != len(set(ids)) or not set(ids) <= known:
            raise ValueError("Duplicate or extra v2 pilot records")
        complete[model] = set(ids) == known
        full_metrics = admission_metrics(evaluate_validation(records, cases))
        engineering[model] = {k: full_metrics[k] for k in ("raw_format_valid_fraction", "citation_numbers_valid_fraction")}
    groups = {}
    for role in ("independent_admission", "development_regression"):
        selected = [c for c in cases if c["strata"]["validation_role"] == role]
        ids = {c["case_id"] for c in selected}
        metrics = {model: admission_metrics(evaluate_validation(
            [r for r in records if r["case_id"] in ids], selected)) for model, records in records_by_model.items()}
        groups[role] = {"case_count": len(selected), "metrics": metrics,
                        "accepted": passes_gate(metrics[primary_model], metrics[reference_model]),
                        "independent_of_prompt_development": role == "independent_admission"}
    format_passed = (engineering[primary_model]["raw_format_valid_fraction"] >= GATE["raw_format_valid_minimum"] and
                     engineering[primary_model]["citation_numbers_valid_fraction"] >= GATE["citation_numbers_valid_minimum"])
    return {"groups": groups, "complete_records": complete,
            "primary_model": primary_model, "comparison_reference": reference_model,
            "all_case_engineering_metrics": engineering, "all_case_engineering_passed": format_passed,
            "accepted": all(complete.values()) and format_passed and all(g["accepted"] for g in groups.values()),
            "pooled_control_accuracy_used_for_admission": False}


def verify_pilot_record_provenance(model, records):
    path = ASSESS / "validation" / f"{model}_protocol.json"
    protocol = base.read(path)
    manifest_hash = base.file_sha(ASSESS / "validation/manifest.json")
    if (not protocol or protocol.get("system_prompt_sha256") != base.sha(base.SYSTEM.encode()) or
            protocol.get("validation_sha256") != manifest_hash):
        raise ValueError("Pilot protocol differs from the selected prompt or validation")
    expected = base.file_sha(path)
    for row in records:
        if (row.get("pilot_protocol_sha256") != expected or
                row.get("validation_sha256") != manifest_hash or
                row.get("model_repo") != protocol.get("model_repo") or
                row.get("model_revision") != protocol.get("model_revision")):
            raise ValueError("Pilot records from another profile, prompt or model cannot be reused")


def check_gate():
    """Parse the fixed control outputs without peeking at recommender scores."""
    from frontier_judge_validation import evaluate_validation
    if PROFILE in ("v2", "validated_v2"):
        qs, cs, packets, _ = prepare_materials()
        prepare_validation(qs, cs, packets)
    selection_sha = validate_selection_policy() if PROFILE == "validated_v2" else None
    cases = base.read_rows(ASSESS / "validation/cases.jsonl")
    results, records_by_model = {}, {}
    for model in ("phi", "qwen"):
        records = base.read_rows(ASSESS / "validation" / f"{model}_records.jsonl")
        if PROFILE in ("v2", "validated_v2"):
            verify_pilot_record_provenance(model, records)
        records_by_model[model] = records
        results[model] = evaluate_validation(records, cases)
    # Metrics are adapted to the validation module's explicit contract below.
    from frontier_judge_validation import admission_metrics
    metrics = {model: admission_metrics(result) for model, result in results.items()}
    q, p = metrics["qwen"], metrics["phi"]
    primary_model, secondary_model = selected_models()
    grouped = grouped_gate(records_by_model, cases, primary_model, secondary_model) if PROFILE != "v1" else None
    accepted = grouped["accepted"] if grouped is not None else passes_gate(q, p)
    result = {"at": base.now(), "accepted": bool(accepted), "thresholds": GATE,
              "metrics": metrics if PROFILE == "v1" else None,
              "primary_selected": ("microsoft/phi-4" if primary_model == "phi" else "Qwen/Qwen3.6-27B") if accepted else None,
              "scientific_accuracy_established": False,
              "failure_policy": "Stop for inspection; do not silently change controls, prompts or thresholds"}
    if grouped is not None:
        result["validation_groups"] = grouped
    if PROFILE == "validated_v2":
        result.update(selection_policy_sha256=selection_sha, selected_primary=primary_model,
            sensitivity_secondary=secondary_model, secondary_passed_primary_admission=False,
            selection_timing="Explicit selection revision after both pilots; not preregistered before pilot outcomes",
            source_pilots="Read-only references to v2; no pilot rerun and no full-label reuse")
    real = {model: {row["case_id"]: row for row in report["rows"] if row["kind"] == "real_source_pair"}
            for model, report in results.items()}
    comparison = []
    for cid in sorted(real["qwen"]):
        left, right = real["qwen"][cid], real["phi"][cid]
        valid = all(r["schema_valid"] and r["citation_numbers_valid"] for r in (left, right))
        comparison.append({"case_id": cid, "both_valid": valid,
            "qwen_grade": left["grade"], "phi_grade": right["grade"],
            "qwen_applicability": left["documented_applicability"],
            "phi_applicability": right["documented_applicability"],
            "same_decision": valid and (left["grade"], left["documented_applicability"]) ==
                                       (right["grade"], right["documented_applicability"])})
    result["unlabelled_real_source_comparison"] = {"pairs": len(comparison),
        "both_valid": sum(r["both_valid"] for r in comparison),
        "same_decision_including_unknown": sum(r["same_decision"] for r in comparison),
        "accuracy": None, "used_to_tune_thresholds": False, "rows": comparison,
        "interpretation": "Agreement and disagreement only, not a scientific correctness score"}
    base.dump(ASSESS / "validation/admission.json", result)
    return result


def publish_summary(evaluation=None):
    folder = ROOT / "experiments/model_recommendation/results_summary"
    link_name = {"v1": "03_automatic_document_assessment", "v2": "04_revised_document_assessment",
                 "validated_v2": "05_validated_document_assessment"}[PROFILE]
    link = folder / link_name
    if not link.exists():
        link.symlink_to(os.path.relpath(ASSESS, folder), target_is_directory=True)
    path = folder / "README.md"
    text = path.read_text()
    marker_name = "modern-scoring" if PROFILE == "v1" else "modern-scoring-" + PROFILE
    marker, end = f"<!-- {marker_name}:start -->", f"<!-- {marker_name}:end -->"
    state = base.read(RUN / "pipeline_status.json", {})
    worker = base.read(RUN / "status.json", {})
    msg = (f"{marker}\n## Follow-up assessment run\n\n"
           f"Updated: {base.now()}. Stage: `{state.get('stage', 'prepare')}`; Status: `{state.get('state', 'starting')}`. "
           f"The new assessment directory is separate; old Phi-4 labels are not mixed in.\n\n"
           "Qwen3.6-27B, with extended thinking disabled, is the candidate primary judge; Phi-4 is the fixed reference and 5% cross-family repeat judge. "
           "Controlled logic checks and real-source comparisons must pass the frozen admission criteria before all 216,160 pairs are assessed. "
           "Control questions are not expert gold standards; the real-source comparison has no human accuracy labels.\n\n"
           f"[Current status](../../../runtime/deployment/{RUN.name}/pipeline_status.json) · "
           f"[New assessment results and validation]({link_name}/)\n\n"
           f"Command from the project root: `runtime/app-env/bin/python scripts/run_frontier_scoring.py status --profile {PROFILE}`, "
           f"`... run_frontier_scoring.py stop --profile {PROFILE}`, `... run_frontier_scoring.py start --profile {PROFILE}`. "
           "Stopping retains committed checkpoints. SSH disconnects do not interrupt the run; server reboot requires manual continuation.\n"
           f"{end}")
    if PROFILE == "v2":
        disclosure = ("The v1 Qwen joint control result was 51/64, below the original admission threshold. Failures and original code remain; full scoring did not begin. "
            "v2 changes only the scoring instructions, identically for Qwen and Phi. The original 64 controls are development regression and 64 new controls are independent admission checks. "
            "Both groups must separately pass the original thresholds before full scoring. They are not pooled as 128 independent validation samples; 60 real-source cases still lack human labels.\n\n")
        msg = msg.replace("The new assessment directory is separate; old Phi-4 labels are not mixed in.\n\n",
                          "The new assessment directory is separate; old Phi-4 labels are not mixed in.\n\n" + disclosure)
    if PROFILE == "validated_v2":
        msg = msg.replace("Qwen3.6-27B, with extended thinking disabled, is the candidate primary judge; Phi-4 is the fixed reference and 5% cross-family repeat judge. "
            "Controlled logic checks and real-source comparisons must pass the frozen admission criteria before all 216,160 pairs are assessed.",
            "Phi-4 uses v2 instructions for all 216,160 primary judgments; Qwen3.6-27B is used only for the fixed 5% disagreement-sensitivity review. "
            "Phi passed the original thresholds with 59/64 independent controls and 60/64 development controls. Qwen scored 56/64 on independent controls and did not qualify as a judge or gold standard. "
            "This is an explicit judge-selection revision after viewing both prechecks, not a preregistered selection. Original v1/v2 failures remain and completed prechecks are reused read-only.")
    if marker in text:
        text = text[:text.index(marker)] + msg + text[text.index(end)+len(end):]
    else:
        text += "\n" + msg + "\n"
    path.write_text(text)


def wait_backend(update):
    update(stage="waiting_for_qwen_weights")
    publish_summary()
    waiting = time.monotonic()
    while not (BACKEND_RUN / "ready.json").exists():
        if (RUN / "stop.requested").exists():
            update(state="stopped")
            return False
        if time.monotonic()-waiting > 12*3600:
            raise TimeoutError("Weights preparation exceeded 12 hours")
        if shutil.disk_usage(ROOT).free < 40*1024**3:
            raise RuntimeError("Disk headroom below 40 GiB")
        download = base.read(BACKEND_RUN / "download_status.json", {})
        if download.get("state") in ("failed", "stopped_on_guard", "stopped"):
            raise RuntimeError("Qwen artifact preparation stopped; inspect project download status")
        update(download_state=download.get("state"),
               download_bytes=download.get("downloaded_cache_bytes"))
        time.sleep(20)
    if not QWEN_WEIGHTS.exists() or not MODERN.exists():
        raise ValueError("Backend readiness lacks required files")
    ready = base.read(BACKEND_RUN / "ready.json")
    if (ready.get("state") != "ready" or
            ready.get("backend_sha256") != base.file_sha(ROOT / "scripts/frontier_modern_judge.py") or
            ready.get("weights_manifest_sha256") != base.file_sha(QWEN_WEIGHTS)):
        raise ValueError("Backend readiness does not match pinned implementation and weights")
    return True


def pipeline():
    ensure_execution_allowed()
    lock = (RUN / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.dump(RUN / "pipeline_pid.json", base.identity())
    state = {**base.identity(), "state": "running", "started_at": base.now(),
             "profile": PROFILE, "scheduler_sha256": base.file_sha(Path(__file__)),
             "monitor_seconds": 20, "no_progress_seconds": 1200,
             "policy": "Sequential GPU stages; fixed pilot admission; full matrices; preserve unknowns and failures"}
    def update(**values):
        status_update(RUN / "pipeline_status.json", state, **values)
    child = None
    def stop_signal(signum, frame):
        raise KeyboardInterrupt()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_signal)
    try:
        update(stage="prepare")
        qs, cs, packets, _ = prepare_materials()
        prepare_validation(qs, cs, packets)
        freeze(ASSESS / "validation/admission_protocol.json", GATE)
        if PROFILE == "validated_v2":
            if not wait_backend(update):
                return
            prepare_protocol()
            stages = [("primary", APP, ["run", "--judge", "primary"], RUN / "status.json"),
                      ("secondary", MODERN, ["run", "--judge", "secondary"], RUN / "status.json")]
        else:
            stages = [("pilot_phi", APP, ["pilot", "--model", "phi"], RUN / "pilot_phi_status.json"),
                      ("pilot_qwen", MODERN, ["pilot", "--model", "qwen"], RUN / "pilot_qwen_status.json"),
                      ("primary", MODERN, ["run", "--judge", "primary"], RUN / "status.json"),
                      ("secondary", APP, ["run", "--judge", "secondary"], RUN / "status.json")]
        for stage, python, arguments, status_path in stages:
            if (RUN / "stop.requested").exists():
                update(state="stopped"); return
            if stage == "pilot_qwen":
                if not wait_backend(update):
                    return
                prepare_protocol()
            if stage == "primary":
                if not check_gate()["accepted"]:
                    update(state="stopped_on_validation", stage="admission_failed"); return
            update(stage=stage, worker_pid=None)
            publish_summary()
            with (RUN / "runner.log").open("a") as log:
                child = subprocess.Popen(worker_command(python, *arguments),
                    cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            update(worker_pid=child.pid)
            last, changed, started = None, time.monotonic(), time.monotonic()
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
                if (time.monotonic()-changed > 1200 or time.monotonic()-started > 8*86400 or
                    shutil.disk_usage(ROOT).free < 30*1024**3 or
                    (RUN / "runner.log").stat().st_size > 20*1024**2):
                    base.stop_child(child); update(state="stopped_on_guard", worker_pid=None); return
                time.sleep(20)
            current = base.read(status_path, {})
            if child.returncode or current.get("pid") != child.pid or current.get("state") != "complete":
                update(state="failed", failed_stage=stage, worker_pid=None, worker_state=current.get("state")); return
            child = None
            if stage == "primary":
                update(stage="primary_metrics")
                from evaluate_frontier_automatic import evaluate_run
                evaluate_run(OUT, ASSESS)
        update(stage="final_evaluation")
        base.ASSESS = ASSESS
        agreement = base.judge_agreement()
        # Correct legacy prose without changing the actual agreement arithmetic.
        primary_model, secondary_model = selected_models()
        agreement["primary_repo"] = base.read(weights_manifest(primary_model))["repo"]
        agreement["secondary_repo"] = base.read(weights_manifest(secondary_model))["repo"]
        agreement["limitations"] = ["Automatic agreement is not expert correctness", "Qwen family overlaps several evaluated rankers", "Unknowns excluded from known-target agreement; denominator reported", "Different model families can share source/pretraining biases"]
        if PROFILE == "validated_v2":
            agreement["secondary_passed_primary_admission"] = False
            agreement["limitations"].append("Qwen failed primary admission; secondary labels measure disagreements only and never validate or replace primary labels")
        base.dump(ASSESS / "judge_agreement.json", agreement)
        from evaluate_frontier_automatic import evaluate_run
        final = evaluate_run(OUT, ASSESS, require_complete=True)
        update(state="complete", completed_at=base.now(), evaluation_state=final["state"], worker_pid=None)
    except BaseException as exc:
        update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               error_type=type(exc).__name__, worker_pid=None)
        raise
    finally:
        base.stop_child(child)
        publish_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "pipeline", "pilot", "run", "status", "stop", "prepare"))
    parser.add_argument("--model", choices=("phi", "qwen"), default="qwen")
    parser.add_argument("--judge", choices=("primary", "secondary"), default="primary")
    parser.add_argument("--profile", choices=tuple(PROFILE_DIRECTORIES), default="v1")
    args = parser.parse_args()
    configure_profile(args.profile)
    initialize()
    if args.action == "status":
        print(json.dumps({"alive": base.alive(base.read(RUN / "pipeline_pid.json", {})),
                          "pipeline": base.read(RUN / "pipeline_status.json", {}),
                          "worker": base.read(RUN / "status.json", {})}, ensure_ascii=False))
    elif args.action == "stop":
        (RUN / "stop.requested").touch()
        print("Stop requested; committed outputs retained")
    elif args.action == "start":
        ensure_execution_allowed()
        if base.alive(base.read(RUN / "pipeline_pid.json", {})):
            raise RuntimeError("Scoring pipeline already alive")
        # Do not overlap a manually started legacy judge.
        for name in ("pid.json", "pipeline_pid.json"):
            if base.alive(base.read(ROOT / "runtime/deployment/frontier_assessment" / name, {})):
                raise RuntimeError("Legacy assessment still running")
            for other_profile, (_, directory) in PROFILE_DIRECTORIES.items():
                if other_profile == PROFILE:
                    continue
                other = ROOT / "runtime/deployment" / directory
                if base.alive(base.read(other / name, {})):
                    raise RuntimeError("Another scoring profile is already running")
        (RUN / "stop.requested").unlink(missing_ok=True)
        with (RUN / "runner.log").open("a") as log:
            child = subprocess.Popen(worker_command(APP, "pipeline"), cwd=ROOT,
                env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"started_pid": child.pid, "status": str(RUN / "pipeline_status.json")}))
    elif args.action == "pipeline":
        pipeline()
    elif args.action == "pilot":
        pilot(args.model)
    elif args.action == "run":
        try:
            run_full(args.judge)
        except BaseException as exc:
            st = base.read(RUN / "status.json", {})
            status_update(RUN / "status.json", st, state="failed", error_type=type(exc).__name__)
            raise
    else:
        qs, cs, packets, _ = prepare_materials()
        print(json.dumps(prepare_validation(qs, cs, packets), ensure_ascii=False))


PILOT_FUNCTION_SHA = base.sha(inspect.getsource(pilot).encode())
PHI_ADAPTER_SHA = base.sha(inspect.getsource(PhiJudge).encode())

if __name__ == "__main__":
    main()
