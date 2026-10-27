"""Retry exactly one reviewed failed local evidence request; never restart a pipeline."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
from uuid import UUID


def validate_focus_plan(payload, specifications):
    items = payload.get("input_items") or []
    plan = []
    for specification in specifications:
        index_text, parameter = specification.split(":", 1)
        index, parameter = int(index_text), parameter.strip()
        if not 0 <= index < len(items) or not parameter or not re.search(r"(?<!\w)" + re.escape(parameter) + r"(?!\w)", items[index], re.I):
            raise ValueError("Every recovery focus must name an explicit parameter in its original input item")
        focus = {"input_item_index": index, "parameter": parameter}
        if focus in plan:
            raise ValueError("Duplicate recovery focus")
        plan.append(focus)
    if {focus["input_item_index"] for focus in plan} != set(range(len(items))):
        raise ValueError("The reviewed focus plan must cover every original input item")
    return plan


def validate_recovery_failure(response, payload, recovery_mode, plan):
    if not recovery_mode and not plan:
        return
    if "output_token_limit_reached:" not in (response.get("failure_reason") or ""):
        raise RuntimeError("Structured recovery requires a saved output-token-limit failure")
    if plan:
        from pbpk_kb.processing.run_llm_extraction import _has_ambiguous_numeric_comparison
        if not any(_has_ambiguous_numeric_comparison(item) for item in payload.get("input_items") or []):
            raise RuntimeError("Focused recovery requires an original ambiguous comparison and a reviewed complete plan")
    else:
        from pbpk_kb.processing.llm_agent import has_repeated_complete_candidate
        if not has_repeated_complete_candidate(response.get("failed_raw_output") or ""):
            raise RuntimeError("Nonrepeating recovery requires saved repeated complete candidates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--recovery-mode", choices=("nonrepeating_v1", "comparison_schema_v1", "comparison_sampling_v1", "comparison_penalty_v1"))
    parser.add_argument("--focus", action="append", default=[], metavar="ITEM_INDEX:PARAMETER",
                        help="Reviewed independent parameter decisions; must cover every original input item.")
    args = parser.parse_args()
    if args.focus and args.recovery_mode:
        raise ValueError("Choose either focused decisions or a whole-request recovery mode")
    if not re.fullmatch(r"run_\d{8}T\d{6}Z", args.run_id):
        raise ValueError("Invalid project run ID")
    request_id = str(UUID(args.request_id))
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ["PBPK_EVIDENCE_LLM_BATCH_SIZE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from pbpk_kb.processing import run_llm_extraction as m
    from pbpk_kb.processing.llm_agent import LLMExtractionAgent
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    lock = (root / "runtime/deployment/evidence_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    layout = raw_processing_layout(root / f"data_process/raw_processing/{args.run_id}")
    requests_path = layout.path("llm_extraction_requests_live.jsonl")
    responses_path = layout.path("llm_extraction_responses_live.jsonl")
    failures_path = layout.path("processing_failures.jsonl")
    requests = [r for r in m._iter_jsonl(requests_path) if r["request_id"] == request_id]
    responses = [r for r in m._iter_jsonl(responses_path) if r["request_id"] == request_id]
    if not requests or not responses or responses[-1]["status"] in m.PROCESSED_LLM_STATUSES:
        raise RuntimeError("Only an existing failed request may receive one controlled retry")
    original = requests[-1]
    plan = validate_focus_plan(original, args.focus) if args.focus else []
    validate_recovery_failure(responses[-1], original, args.recovery_mode, plan)
    if original["input_hash"] != responses[-1]["input_hash"]:
        raise RuntimeError("Failure does not match the recorded input")
    chunk = next(r for r in m._iter_jsonl(layout.path("text_chunks.jsonl")) if r["chunk_id"] == original["chunk_id"])
    upstream = m._load_upstream_index(layout)
    if m._chunk_provenance_error(chunk, upstream):
        raise RuntimeError("Original request provenance failed validation")
    agent = LLMExtractionAgent(mode="live")
    request = agent.build_request_record({**original, "generation_recovery_mode": args.recovery_mode,
                                        "created_at": m._utc_now(), "request_mode": "live",
                                        "attempt_number": max(int(r.get("attempt_number") or 1) for r in requests) + 1})
    if plan:
        request.update(generation_recovery_mode="focused_assertions_v1", generation_recovery_plan=plan,
                       generation_format="composite_json_schema_xgrammar_0.2.5.post1_ws2",
                       effective_prompt_sha256=None, generation_schema_sha256=None)
    if request["input_hash"] != original["input_hash"]:
        raise RuntimeError("Retry changed the original evidence input")
    task = m.PendingTask(chunk, request["target_compound"], {}, request, request["input_hash"])
    m._append_jsonl(requests_path, request)
    print(json.dumps({"status": "controlled_retry_started", "request_id": request_id,
                      "attempt_number": request["attempt_number"], "generation_format": request["generation_format"],
                      "generation_recovery_mode": request.get("generation_recovery_mode"),
                      "generation_sampling": request["generation_sampling"],
                      "effective_prompt_sha256": request["effective_prompt_sha256"]}), flush=True)
    calls = []
    if plan:
        from pbpk_kb.processing.llm_agent import LLMCallResult
        candidates, error, elapsed = [], None, 0.0
        for focus in plan:
            child = agent.build_request_record({**request, "generation_recovery_mode": "focused_assertion_v1",
                                                "generation_recovery_focus": focus})
            outcome = agent.extract(child)
            elapsed += outcome.latency_seconds or 0.0
            calls.append({"focus": focus, "input_hash": child["input_hash"],
                          "effective_prompt_sha256": child["effective_prompt_sha256"],
                          "generation_schema_sha256": child["generation_schema_sha256"],
                          "status": outcome.status, "error": outcome.error,
                          "raw_response": outcome.raw_response, "latency_seconds": outcome.latency_seconds})
            print(json.dumps({"focus": focus, "status": outcome.status, "candidate_count": len(outcome.candidates)}), flush=True)
            if outcome.status != "ok":
                error = outcome.error
                break
            candidates.extend(outcome.candidates)
        complete = error is None and len(calls) == len(plan)
        result = LLMCallResult("ok" if complete else "failed_local_llm", candidates if complete else [],
                              json.dumps({"recovery_calls": calls}, ensure_ascii=False), error,
                              agent.model_name, elapsed)
    else:
        result = agent.extract(request)
    valid, rejected = m._validate_candidates(task, result.candidates, upstream) if result.status == "ok" else ([], [])
    response = m._response_record(task, status=result.status, model_name=result.model_name,
        prompt_template_version=agent.prompt_template_version, candidates=valid, rejected_candidates=rejected,
        schema_validation_status=m._schema_validation_status(valid, rejected, len(result.candidates)) if result.status == "ok" else "not_validated",
        error=result.error, latency_seconds=result.latency_seconds,
        output_hash=m._hash_optional_text(result.raw_response),
        failed_raw_output=result.raw_response if result.status != "ok" else None)
    if calls:
        response["generation_recovery_calls"] = calls
        response["generation_recovery_plan"] = plan
    m._append_jsonl(responses_path, response)
    if result.status != "ok":
        m._append_jsonl(failures_path, m._api_failure(task, result))
        raise RuntimeError("Controlled retry failed; pipeline remains stopped. Inspect the saved failure output.")
    existing_failure_ids = m._load_failure_ids(failures_path)
    for rejection in rejected:
        failure = m._candidate_failure(task, rejection)
        if failure["failure_id"] not in existing_failure_ids:
            m._append_jsonl(failures_path, failure)
            existing_failure_ids.add(failure["failure_id"])
    m._append_evidence_rows(chunk, task.target_compound, valid,
        llm_evidence_path=layout.path("evidence_candidates_llm.jsonl"),
        merged_evidence_path=layout.path("evidence_candidates.jsonl"),
        existing_evidence_ids=m._load_existing_evidence_ids(layout.path("evidence_candidates.jsonl")), counts=Counter())
    m._resolve_recovered_api_failures(failures_path, {request_id})
    spec = importlib.util.spec_from_file_location("watchdog", root / "scripts/deployment/evidence_watchdog.py")
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    audit = watchdog.evidence_audit({r["request_id"]: r for r in m._iter_jsonl(requests_path)},
                                    list(m._iter_jsonl(responses_path)), historical=True)
    print(json.dumps({"status": result.status, "request_id": request_id,
                      "accepted_candidates": len(valid), "rejected_candidates": len(rejected),
                      "latency_seconds": result.latency_seconds, "audit": audit}), flush=True)
    if audit["reasons"]:
        raise RuntimeError("Historical audit did not pass; pipeline remains stopped")


if __name__ == "__main__":
    main()
