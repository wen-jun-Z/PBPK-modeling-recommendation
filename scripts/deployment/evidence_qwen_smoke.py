"""Exercise extraction, attribution and quality on stored real source records.

No old artifact or database is changed. This validates the replacement backend;
the new expansion must separately pass its own ten-request extraction gate.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import httpx

from pbpk_kb.processing.evidence_llm import registered_server, verify_server_process
from pbpk_kb.processing.llm_agent import LLMExtractionAgent
from pbpk_kb.processing.final_evidence import LLMAttributionAgent
from pbpk_kb.processing.auto_quality_llm_stage import AutoQualityLLMStage, VALIDATION_PROMPT, REPAIR_PROMPT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime/deployment/evidence_expansion_20261009/backend"
RAW = ROOT / "data_process/raw_processing/run_20260909T083120Z"
FINAL = ROOT / "data_process/final_evidence/run_20260909T083120Z"


def first(path, count):
    with path.open() as handle:
        return [json.loads(next(handle)) for _ in range(count)]


def row_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    config = registered_server()
    verify_server_process(config)
    health = config["base_url"].removesuffix("/v1") + "/health"
    for _ in range(450):
        response = httpx.get(health, trust_env=False, timeout=5)
        if response.status_code == 200:
            break
        if response.json()["state"] == "failed":
            raise RuntimeError("Replacement backend model loading failed")
        time.sleep(2)
    else:
        raise TimeoutError("Replacement backend readiness exceeded fifteen minutes")
    extract_path = RAW / "05_llm_extraction/llm_extraction_requests.jsonl"
    extraction_rows = first(extract_path, 2)
    attr_path = FINAL / "01_compound_attribution/llm_attribution_requests.jsonl"
    attribution_row = first(attr_path, 1)[0]
    quality_path = FINAL / "06_auto_quality_control/auto_validated_evidence.jsonl"
    with quality_path.open() as handle:
        quality_row = next(row for line in handle if (row := json.loads(line)).get("evidence_type") == "clinical_pk" and row.get("value") is not None)
    results = []

    def extract(row):
        fields = ["target_compound", "input_text_or_table", "input_items", "source_id", "source_url",
                  "source_location", "checksum", "raw_asset_id", "chunk_id", "content_kind", "request_mode"]
        payload = {k: row[k] for k in fields if k in row}
        result = LLMExtractionAgent(mode="live").extract(payload)
        checks = [{"target_matches": candidate.get("target_compound") == payload["target_compound"],
                   "quote_matches": candidate.get("source_text") in payload["input_items"],
                   "requires_review": candidate.get("requires_review") is True}
                  for candidate in result.candidates]
        return {"stage": "extraction", "source_path": str(extract_path.relative_to(ROOT)),
                "source_row_sha256": row_hash(row), "source_request_id": row["request_id"],
                "payload": payload, "status": result.status, "error": result.error,
                "candidates": result.candidates, "latency_seconds": result.latency_seconds,
                "checks": checks, "passed": result.status == "ok" and all(all(x.values()) for x in checks)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results.extend(pool.map(extract, extraction_rows))
    compounds = [{"preferred_name": name, "synonyms": []} if isinstance(name, str) else name
                 for name in attribution_row["allowed_compounds"]]
    attribution = LLMAttributionAgent(mode="live").attribute(row=attribution_row["evidence"], compounds=compounds)
    results.append({"stage": "attribution", "source_path": str(attr_path.relative_to(ROOT)),
        "source_row_sha256": row_hash(attribution_row), "result": attribution,
        "passed": attribution["status"] == "ok" and bool(attribution.get("source_span_binding"))})
    quote = quality_row["source_text"]
    current = {k: quality_row.get(k) for k in ["value", "value_low", "value_high", "value_qualifier", "unit", "condition"]}
    original = {"evidence_id": quality_row["evidence_id"], "target_compound": quality_row["compound_name"],
                "parameter": quality_row["parameter"], "source_text": quote, "current_fields": current,
                "source_context": {"combined_text": quote}}
    quality = AutoQualityLLMStage(final_run_dir=OUT / "smoke_quality_unused", raw_run_dir=RAW)
    requests = [("quality_repair", REPAIR_PROMPT, original),
                ("quality_validation", VALIDATION_PROMPT,
                 {"evidence_id": quality_row["evidence_id"], "target_compound": quality_row["compound_name"],
                  "original_evidence": original, "repaired_fields": {}, "source_context": {"combined_text": quote}})]

    def quality_request(request):
        stage, prompt, payload = request
        started = time.monotonic()
        try:
            result = quality._chat(prompt, payload)
            support = result.get("supporting_text", result.get("verbatim_support"))
            return {"stage": stage, "source_path": str(quality_path.relative_to(ROOT)),
                "source_row_sha256": row_hash(quality_row), "payload": payload, "result": result,
                "latency_seconds": time.monotonic() - started,
                "passed": isinstance(support, str) and bool(support) and support in quote}
        except Exception as exc:
            return {"stage": stage, "passed": False, "error_type": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results.extend(pool.map(quality_request, requests))
    report = {"purpose": "Replacement backend integration; stored real sources; not new expansion acceptance",
              "model": config["served_model"], "model_revision": config["model_revision"],
              "service_identity": config["server_identity"], "passed": all(row["passed"] for row in results),
              "results": results}
    (OUT / "smoke_real_evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "stages": [{"stage": row["stage"], "passed": row["passed"]}
        for row in results], "report": str(OUT / "smoke_real_evidence.json")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
