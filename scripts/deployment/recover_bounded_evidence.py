"""Recover only failed requests by source item and named parameter family.

No incomplete JSON is salvaged. Each bounded response is source-validated, and
every planned subdecision must finish before the parent can become processed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import repair_expansion_quality as ctl
from curate_legacy_evidence import atomic
from pbpk_kb.processing.llm_agent import LLMExtractionAgent, LLM_INDEXED_EXTRACTION_PROMPT, indexed_extraction_output_schema, restore_indexed_extraction
from pbpk_kb.processing.evidence_llm import _generate_from_server
from pbpk_kb.processing.run_llm_extraction import PendingTask, UpstreamIndex, _validate_candidates, _response_record, _schema_validation_status, _hash_json, _append_jsonl

PRIOR = ctl.BASE / "bounded_recovery_20261022"
OUT = ctl.BASE / "bounded_recovery_source_only_20261022"
FAMILIES = {
    "Cmax": r"\bC\s*max\b|maximum.*concentration|peak.*concentration",
    "Cmin": r"\bC\s*min\b|trough",
    "Tmax": r"\bT\s*max\b|time to (?:peak|maximum)",
    "AUC": r"\bAUC\b|area under",
    "clearance": r"clearance|\bCL(?:/F)?\b",
    "volume of distribution": r"volume of distribution|\bV[dzsλ]|\bV/F\b",
    "half_life": r"half.life|\bt\s*1/2\b",
    "protein_binding": r"bind|bound|\bfu\b|SHBG",
    "metabolism": r"metaboli|CYP|FMO|UGT|oxidation",
    "bioavailability": r"bioavailability",
    "excretion": r"excret|urinary|urine|fecal",
    "absorption": r"absor|food|fed|fasted",
    "exposure": r"exposure|plasma concentrations|blood concentrations",
}


def rows(path):
    for line in path.open():
        if line.strip():
            yield json.loads(line)


def parameter_foci(quote):
    matched = [name for name, pattern in FAMILIES.items() if re.search(pattern, quote, re.I)]
    # Keep a residual decision so facts outside the named families are not lost.
    residual = "other explicit PBPK evidence"
    if matched:
        residual += "; exclude these separately assessed families: " + ", ".join(matched)
    return matched + [residual]


def latest_failed(responses, prompt_version):
    latest = {r["request_id"]: r for r in responses if r.get("prompt_template_version") == prompt_version}
    return {key for key, row in latest.items() if row.get("status") not in {"ok", "reused_same_target_input_hash"}}


def check_resources(out):
    if shutil.disk_usage(ctl.ROOT).free < 12 * ctl.GIB:
        raise RuntimeError("Free space fell below the project 12 GiB reserve")
    if sum(p.stat().st_size for p in out.glob("*") if p.is_file()) > 64 * ctl.MIB:
        raise RuntimeError("Bounded recovery exceeded its 64 MiB output budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=ctl.RUNS, default=ctl.RUNS[0])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-controller", action="store_true")
    parser.add_argument("--pilot-receipt", type=Path)
    args = parser.parse_args()
    out = ctl.inside(args.output_dir or (OUT if args.run_id == ctl.RUNS[0] else ctl.BASE / "bounded_recovery" / args.run_id), ctl.BASE)
    if args.resume_controller:
        if args.pilot_receipt is None:
            parser.error("--resume-controller requires --pilot-receipt")
        ctl.validate_pilot(args.pilot_receipt)
    out.mkdir(parents=True, exist_ok=True)
    check_resources(out)
    lock = (ctl.ROOT / "runtime/deployment/evidence_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if ctl.same_process(ctl.read_json(ctl.BASE / "controller.pid.json")):
        raise RuntimeError("Regular controller must be stopped for response recovery")
    base = ctl.paths(args.run_id)["raw"]
    agent = LLMExtractionAgent(mode="live")
    responses_path = base / "05_llm_extraction/llm_extraction_responses_live.jsonl"
    previous = {r["request_id"]: r for r in rows(responses_path)
                if r.get("prompt_template_version") == agent.prompt_template_version}
    existing_plan = ctl.read_json(out / "plan.json")
    failed = set(existing_plan["parent_ids"]) if existing_plan else latest_failed(previous.values(), agent.prompt_template_version)
    requests = {r["request_id"]: r for r in rows(base / "05_llm_extraction/llm_extraction_requests_live.jsonl") if r["request_id"] in failed}
    assert requests and len(requests) <= 9
    chunks = {r["chunk_id"]: r for r in rows(base / "03_chunks_and_embeddings/text_chunks.jsonl")
              if r["chunk_id"] in {q["chunk_id"] for q in requests.values()}}
    assets = {r["raw_asset_id"]: r for r in rows(base / "01_raw_inventory/raw_inventory.jsonl")
              if r["raw_asset_id"] in {q["raw_asset_id"] for q in requests.values()}}
    upstream = UpstreamIndex(assets, {q["parsed_document_id"] for q in requests.values() if q.get("parsed_document_id")},
                            {q["parsed_table_id"] for q in requests.values() if q.get("parsed_table_id")},
                            {q["target_compound"] for q in requests.values()})
    decisions = []
    for q in requests.values():
        for index, quote in enumerate(q["input_items"]):
            foci = parameter_foci(quote)
            for family in foci:
                key = hashlib.sha256(json.dumps([q["request_id"], index, family]).encode()).hexdigest()[:24]
                decisions.append({"key": key, "request_id": q["request_id"], "index": index, "family": family})
    frozen = {"run_id": args.run_id, "parent_ids": sorted(requests), "decisions": decisions, "input_sha256": _hash_json(requests),
              "implementation_sha256": ctl.digest(Path(__file__)), "method": "complete_source_items_parameter_family_decisions_v1"}
    if existing_plan and existing_plan != frozen:
        raise RuntimeError("Recovery plan changed; preserve the old receipts and use a new output directory")
    atomic(out / "plan.json", frozen)
    atomic(out / "status.json", {"status": "running", "run_id": args.run_id, "total": len(decisions), "done": 0,
                               "failed": 0, "parent_requests": len(requests), "started_at": ctl.now()})

    def run(item):
        check_resources(out)
        result_path = out / (item["key"] + ".json")
        if result_path.exists():
            old = ctl.read_json(result_path)
            if old.get("completed") and old.get("implementation_sha256") == frozen["implementation_sha256"]:
                return old
        q = requests[item["request_id"]]
        quote = q["input_items"][item["index"]]
        # A focused citation may not borrow values or dosing conditions from
        # another input item. Keep the complete selected item, without cropping.
        payload = {"target_compound": q["target_compound"], "input_items": [quote], "input_text_or_table": quote}
        schema = indexed_extraction_output_schema(payload)
        candidate = schema["properties"]["candidates"]["items"]
        candidate["properties"]["source_item_index"] = {"const": 0}
        candidate["properties"]["parameter"] = {"type": "string", "maxLength": 100}
        candidate["properties"]["condition"] = {"anyOf": [{"type": "string", "maxLength": 256}, {"type": "null"}]}
        candidate["properties"]["unit"] = {"anyOf": [{"type": "string", "maxLength": 40}, {"type": "null"}]}
        candidate["properties"]["value_qualifier"] = {"anyOf": [{"type": "string", "maxLength": 32}, {"type": "null"}]}
        schema["properties"]["reason"]["maxLength"] = 160
        schema["properties"]["candidates"]["maxItems"] = 6
        for branch in schema["oneOf"]:
            branch["properties"]["reason"] = deepcopy(schema["properties"]["reason"])
            branch["properties"]["candidates"]["items"] = deepcopy(candidate)
            if branch["properties"]["assessment"].get("const") == "extract":
                branch["properties"]["candidates"]["maxItems"] = 6
        prompt = LLM_INDEXED_EXTRACTION_PROMPT + (
            f"\nThis is a bounded recovery decision. Evaluate ONLY input_items[0] and ONLY the "
            f"parameter family {item['family']!r}. Other items and parameter families have separate decisions. "
            "Preserve endpoint windows, group/dose qualifiers and target identity. Output each distinct supported "
            "fact once, then close the JSON object. Never repeat facts to fill the array. Use brief condition labels, "
            "not reasoning or commentary. For comparisons retain the original qualitative handling. "
            "Do not infer missing table headings, column alignment, or units from a flattened table; "
            "only extract facts whose endpoint, value, and unit ownership are explicit in this item. "
            "If there are more than six distinct facts for this one family, return assessment=uncertain and explain the overflow."
        )
        result = {**item, "implementation_sha256": frozen["implementation_sha256"],
                  "schema_sha256": _hash_json(schema), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "source_payload_sha256": _hash_json(payload), "completed": False}
        started = time.monotonic()
        try:
            prior_path = PRIOR / (item["key"] + ".json")
            prior = ctl.read_json(prior_path) if prior_path.exists() else {}
            if (prior.get("raw_result") and prior.get("source_payload_sha256") == result["source_payload_sha256"]
                    and prior.get("schema_sha256") == result["schema_sha256"]
                    and prior.get("prompt_sha256") == result["prompt_sha256"]):
                parsed = prior["raw_result"]
                result["reused_identical_generation_receipt"] = {"path": str(prior_path), "sha256": ctl.digest(prior_path)}
            else:
                parsed = _generate_from_server(prompt, payload, {"json_schema": schema})
            result["raw_result"] = parsed
            if "overflow" in parsed.get("reason", "").lower() or len(parsed["candidates"]) >= 6:
                raise ValueError("Decision reached candidate bound; requires further subdivision")
            restored = restore_indexed_extraction(parsed, payload)
            task = PendingTask(chunks[q["chunk_id"]], q["target_compound"], {}, q, q["input_hash"])
            valid, rejected = _validate_candidates(task, restored, upstream)
            result.update(completed=True, valid=valid, rejected=rejected)
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc)[:300])
        result["seconds"] = time.monotonic() - started
        atomic(result_path, result)
        return result

    outputs = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, item) for item in decisions]):
            result = future.result()
            outputs.append(result)
            atomic(out / "status.json", {"status": "running", "run_id": args.run_id, "total": len(decisions), "done": len(outputs),
                   "failed": sum(not r["completed"] for r in outputs), "parent_requests": len(requests), "updated_at": ctl.now()})
            print(json.dumps({k: result.get(k) for k in ("key", "family", "completed", "error_type")}), flush=True)
    successful = []
    for qid, q in requests.items():
        parts = [x for x in outputs if x["request_id"] == qid]
        if not all(x["completed"] for x in parts):
            continue
        unique = {}
        for part in parts:
            for candidate in part["valid"]:
                unique[_hash_json({k: v for k, v in candidate.items() if k not in {"confidence", "validation_repairs"}})] = candidate
        candidates = list(unique.values())
        rejected = [x for part in parts for x in part["rejected"]]
        task = PendingTask(chunks[q["chunk_id"]], q["target_compound"], {}, q, q["input_hash"])
        response = _response_record(task, status="ok", model_name=agent.model_name,
            prompt_template_version=agent.prompt_template_version, candidates=candidates, rejected_candidates=rejected,
            schema_validation_status=_schema_validation_status(candidates, rejected, len(candidates) + len(rejected)),
            error=None, latency_seconds=sum(x["seconds"] for x in parts), output_hash=_hash_json(parts))
        response.update(generation_recovery_mode="complete_source_items_parameter_family_decisions_v1",
                        bounded_decision_receipts=[str(out / (x["key"] + ".json")) for x in parts],
                        bounded_recovery_plan_sha256=ctl.digest(out / "plan.json"))
        check_resources(out)
        _append_jsonl(responses_path, response)
        successful.append(qid)
    complete = len(successful) == len(requests)
    atomic(out / "status.json", {"status": "completed" if complete else "requires_review", "run_id": args.run_id,
           "total": len(decisions), "done": len(outputs), "failed": sum(not r["completed"] for r in outputs),
           "parent_requests": len(requests), "recovered_parents": successful, "updated_at": ctl.now(),
           "failed_or_attempt_counters_reset": False})
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    if not complete:
        raise RuntimeError("Some bounded decisions need review; the controller was not resumed")
    if args.resume_controller:
        subprocess.run([sys.executable, "-B", str(ctl.SCRIPT), "start", "--execute", "--apply-validated",
                        "--attempt-limit", "3", "--pilot-receipt", str(args.pilot_receipt)],
                       cwd=ctl.ROOT, env=ctl.runtime_environment(), check=True)


if __name__ == "__main__":
    main()
