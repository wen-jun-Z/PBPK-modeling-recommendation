"""Finalize bounded recovery, replay the narrow validator fix, then resume.

Successful generation is never replayed. Frozen receipts and old failed
responses are retained. A response cap is resolved only by an independently
parsed source table with exactly the same six distinct cells.
"""
from __future__ import annotations
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import repair_expansion_quality as ctl
import recover_bounded_evidence as bounded
from curate_legacy_evidence import atomic
from pbpk_kb.processing.llm_agent import LLMExtractionAgent, restore_indexed_extraction
from pbpk_kb.processing.run_llm_extraction import (
    PendingTask, UpstreamIndex, _validate_candidates, _response_record,
    _schema_validation_status, _hash_json, _append_jsonl, VALIDATOR_VERSION,
)


def load_bundle(base, requests):
    chunks = {r["chunk_id"]: r for r in bounded.rows(base / "03_chunks_and_embeddings/text_chunks.jsonl")
              if r["chunk_id"] in {q["chunk_id"] for q in requests.values()}}
    assets = {r["raw_asset_id"]: r for r in bounded.rows(base / "01_raw_inventory/raw_inventory.jsonl")
              if r["raw_asset_id"] in {q["raw_asset_id"] for q in requests.values()}}
    upstream = UpstreamIndex(assets, {q["parsed_document_id"] for q in requests.values() if q.get("parsed_document_id")},
                            {q["parsed_table_id"] for q in requests.values() if q.get("parsed_table_id")},
                            {q["target_compound"] for q in requests.values()})
    return chunks, upstream


def six_cell_table_proof(part, q):
    if q["request_id"] != "5faa90fd-aec3-58ae-a599-29b1ead7eb19" or part["index"] != 0:
        return None
    family = part["family"]
    indices = {"Cmax": 0, "Tmax": 1, "AUC": 2, "clearance": 3, "volume of distribution": 4}
    if family not in indices:
        return None
    quote = q["input_items"][0]
    if not quote.startswith("21 DAY DOSING PERIOD Levonorgestrel C max T max AUC CL/F Vλz/F SHBG Day"):
        return None
    total, rest = quote.split("Unbound Levonorgestrel", 1)
    unbound, _ = rest.split("Ethinyl Estradiol", 1)
    expected = []
    for kind, block in (("total", total), ("unbound", unbound)):
        matches = re.findall(r"(?<!\S)(1|6|21)\s+((?:[\d.]+\s*\([\d.]+\)\s*){6})", block)
        if [int(day) for day, _ in matches] != [1, 6, 21]:
            return None
        for day, cells in matches:
            values = [float(v) for v in re.findall(r"([\d.]+)\s*\([\d.]+\)", cells)]
            expected.append((kind, int(day), values[indices[family]]))
    parsed = part.get("raw_result") or {}
    candidates = parsed.get("candidates") or []
    actual, repaired_candidates, repaired_conditions = [], [], []
    for original in candidates:
        candidate = deepcopy(original)
        condition = candidate.get("condition") or ""
        match = re.search(r"\bday\s*(1|6|21)\b", condition, re.I)
        if candidate.get("value") is None or candidate.get("source_item_index") != 0:
            return None
        kind = "total" if "total" in condition.lower() else "unbound" if "unbound" in condition.lower() else None
        if kind is None:
            return None
        if match:
            day = int(match[1])
        else:
            matching_cells = [cell for cell in expected if cell[0] == kind and cell[2] == float(candidate["value"])]
            if len(matching_cells) != 1:
                return None
            day = matching_cells[0][1]
            candidate["condition"] = f"Day {day} of 21-day dosing period; {kind} levonorgestrel; Vλz/F"
            repaired_conditions.append({"value": candidate["value"], "old": condition, "new": candidate["condition"], "source_day": day})
        actual.append((kind, day, float(candidate["value"])))
        repaired_candidates.append(candidate)
    if sorted(actual) != sorted(expected):
        return None
    return {"method": "same_source_exact_table_cell_cardinality", "expected_cells": expected,
            "source_repaired_candidates": repaired_candidates, "source_condition_repairs": repaired_conditions,
            "source_sha256": _hash_json(quote), "note": "Exactly six cells, not an overflow; unchanged strict candidate validation follows."}


def main():
    out = bounded.OUT
    status = ctl.read_json(out / "status.json")
    if status.get("done") != status.get("total") or status.get("status") == "running":
        raise RuntimeError("Bounded recovery is still running")
    lock = (ctl.ROOT / "runtime/deployment/evidence_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = ctl.paths("run_20261009T140919Z")["raw"]
    plan = ctl.read_json(out / "plan.json")
    selected = set(plan["parent_ids"])
    requests = {q["request_id"]: q for q in bounded.rows(base / "05_llm_extraction/llm_extraction_requests_live.jsonl") if q["request_id"] in selected}
    chunks, upstream = load_bundle(base, requests)
    parts = [ctl.read_json(out / (d["key"] + ".json")) for d in plan["decisions"]]
    for part in parts:
        if part["completed"]:
            continue
        q = requests[part["request_id"]]
        proof = six_cell_table_proof(part, q)
        if proof is None:
            raise RuntimeError("Unresolved bounded decision: " + part["key"])
        candidates = restore_indexed_extraction({**part["raw_result"], "candidates": proof["source_repaired_candidates"]}, q)
        task = PendingTask(chunks[q["chunk_id"]], q["target_compound"], {}, q, q["input_hash"])
        valid, rejected = _validate_candidates(task, candidates, upstream)
        part.update(completed=True, valid=valid, rejected=rejected, source_cardinality_review=proof)
        atomic(out / (part["key"] + "_source_review.json"), part)
    agent = LLMExtractionAgent(mode="live")
    responses_path = base / "05_llm_extraction/llm_extraction_responses_live.jsonl"
    for qid, q in requests.items():
        decisions = [p for p in parts if p["request_id"] == qid]
        candidates = list({_hash_json(c): c for p in decisions for c in p["valid"]}.values())
        rejected = [c for p in decisions for c in p["rejected"]]
        task = PendingTask(chunks[q["chunk_id"]], q["target_compound"], {}, q, q["input_hash"])
        result = _response_record(task, status="ok", model_name=agent.model_name, prompt_template_version=agent.prompt_template_version,
            candidates=candidates, rejected_candidates=rejected, schema_validation_status=_schema_validation_status(candidates, rejected, len(candidates)+len(rejected)),
            error=None, latency_seconds=sum(p["seconds"] for p in decisions), output_hash=_hash_json(decisions))
        result.update(generation_recovery_mode="complete_source_items_parameter_family_decisions_v1",
                      bounded_recovery_plan_sha256=ctl.digest(out / "plan.json"), source_cardinality_review_count=sum("source_cardinality_review" in p for p in decisions))
        _append_jsonl(responses_path, result)
    # Revalidation changes only the deterministic validator, not the model,
    # request inputs, prompt or the original generation responses.
    latest = {x["request_id"]: x for x in bounded.rows(responses_path) if x.get("prompt_template_version") == agent.prompt_template_version}
    if any(x["status"] not in {"ok", "reused_same_target_input_hash"} for x in latest.values()):
        raise RuntimeError("Failed parent responses remain")
    all_requests = {q["request_id"]: q for q in bounded.rows(base / "05_llm_extraction/llm_extraction_requests_live.jsonl") if q["request_id"] in latest}
    all_chunks, all_upstream = load_bundle(base, all_requests)
    replayed = []
    for qid, response in latest.items():
        if response.get("validator_version") == VALIDATOR_VERSION:
            continue
        q = all_requests[qid]
        task = PendingTask(all_chunks[q["chunk_id"]], q["target_compound"], {}, q, q["input_hash"])
        # The ledger adds this audit field after schema validation. It is not
        # part of an LLM candidate and must not be submitted to its strict schema.
        raw_candidates = [{k: v for k, v in candidate.items() if k != "validation_repairs"}
                          for candidate in response.get("candidates") or []]
        valid, rejected = _validate_candidates(task, raw_candidates, all_upstream)
        if rejected or len(valid) != len(response.get("candidates") or []):
            raise RuntimeError("Previously accepted facts changed during validator replay: " + qid)
        replayed.append({**response, "validator_version": VALIDATOR_VERSION, "candidates": valid,
                         "validator_replay": {"prior_validator": response["validator_version"], "prior_response_sha256": _hash_json(response), "new_generation": False}})
    previous = ctl.BASE / "pilot_v8.json"
    pilot = ctl.read_json(previous)
    changed = [p for p, h in pilot["code_sha256"].items() if ctl.digest(ctl.ROOT / p) != h]
    assert changed == ["src/pbpk_kb/processing/run_llm_extraction.py"]
    spec = importlib.util.spec_from_file_location("pilot_replay", ctl.BASE / "revalidate_pilot.py")
    replay = importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)
    failed_id = "65b8d84c-f11d-50f9-8bd9-115ece008f0e"
    saved_ids = {x["request_id"] for x in pilot["results"][:6]}
    original_bundle = replay.inputs(ctl.ROOT / "data_process/raw_processing/run_20261009T140925Z", saved_ids - {failed_id}, "llm_extraction_requests_live.jsonl")
    staged_bundle = replay.inputs(base, {failed_id}, "llm_extraction_requests_live.jsonl")
    for result in pilot["results"][:6]:
        new, _ = replay.replay(result, staged_bundle if result["request_id"] == failed_id else original_bundle)
        assert {replay.fact_signature(x) for x in new["valid"]} == {replay.fact_signature(x) for x in result["valid"]}
    new_pilot = {**deepcopy(pilot), "execution_kind": "saved_live_response_validator_replay", "validator_version": VALIDATOR_VERSION,
                 "code_sha256": {p: ctl.digest(ctl.ROOT / p) for p in pilot["code_sha256"]}, "prior_pilot_sha256": ctl.digest(previous),
                 "validator_change": "Separate explicitly assigned percentages no longer trigger the shared terminal percentage guard.",
                 "cached_requests_revalidated": len(replayed), "new_LLM_calls_for_replay": 0}
    target = ctl.BASE / "pilot_v9_revalidated.json"
    if target.exists():
        raise RuntimeError("Revalidation receipt already exists; inspect before rerunning")
    atomic(target, new_pilot)
    for response in replayed:
        _append_jsonl(responses_path, response)
    selected_paths = ctl.paths("run_20261009T140919Z")
    state = ctl.read_json(selected_paths["state"])
    assert state["stages"]["llm_extract"]["failures"] == 2
    assert {s for s, entry in state["stages"].items() if entry.get("accepted")} == {"prepare_inputs", "classify"}
    atomic(out / "controller_before_resume.json", state)
    state.setdefault("pilot_history", []).append({"sha256": state["pilot_sha256"], "superseded_at": ctl.now()})
    state["pilot_sha256"] = ctl.digest(target)
    ctl.atomic_json(selected_paths["state"], state)
    atomic(out / "finalized.json", {"completed": True, "parents_recovered": len(requests), "all_cached_successes": len(latest),
           "validator_replayed": len(replayed), "failed_or_attempt_counters_reset": False, "pilot": str(target)})
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    subprocess.run([sys.executable, "-B", str(ctl.SCRIPT), "start", "--execute", "--apply-validated",
                    "--attempt-limit", "3", "--pilot-receipt", str(target)], cwd=ctl.ROOT, check=True)


if __name__ == "__main__":
    main()
