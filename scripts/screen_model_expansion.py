"""Screen newly collected model documentation using local Qwen and existing evidence checks."""
from pathlib import Path
import hashlib
import json
import time
import argparse

from pbpk_kb.processing.llm_agent import LLMExtractionAgent, LLMCallResult
from pbpk_kb.processing.run_llm_extraction import PendingTask, UpstreamIndex, _validate_candidates, prepare_llm_input
from pbpk_kb.processing.auto_quality_llm_stage import AutoQualityLLMStage, VALIDATION_PROMPT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_recommendation/data_expansion_20260907"
RUN = OUT / "qwen_v2"


def main():
    agent = LLMExtractionAgent(mode="live")
    files = []
    models = json.loads((OUT / "collected_models.json").read_text())
    for model in models:
        source = ROOT / model["source_directory"]
        files.extend((model, p) for p in source.glob("README.md"))
        files.extend((model, p) for p in source.glob("Evaluation/Input/Content/Section2.3*.md"))
    RUN.mkdir(exist_ok=True)
    output = RUN / "qwen_evidence_candidates.jsonl"
    if output.exists():
        raise RuntimeError("Screening output already exists; preserve it and choose a new run version.")
    results = []
    for model, path in files:
        text = path.read_text()
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        chunk_id = checksum[:24]
        chunk = {
            "run_id": "model-expansion-20260907", "source_id": "osp_pbpk_model_library",
            "compound": model["compound"], "raw_asset_id": chunk_id,
            "parsed_document_id": chunk_id, "parsed_table_id": None, "chunk_id": chunk_id,
            "storage_path": str(path), "source_url": model["source_url"], "checksum": checksum,
            "source_location": str(path.relative_to(ROOT)), "chunk_text": text,
        }
        prepared = prepare_llm_input(chunk, model["compound"])
        payload = {"target_compound": model["compound"], "input_text_or_table": prepared.input_text,
                   "input_items": list(prepared.sentences),
                   "source_id": chunk["source_id"], "source_location": chunk["source_location"]}
        task = PendingTask(chunk=chunk, target_compound=model["compound"], status_record={},
                           request_record={"request_id": chunk_id, "input_items": list(prepared.sentences)}, input_hash=checksum)
        upstream = UpstreamIndex(raw_assets={chunk_id: {"raw_asset_id": chunk_id, "checksum": checksum, "storage_path": str(path)}},
                                 parsed_document_ids={chunk_id}, parsed_table_ids=set(), target_compounds={model["compound"]})
        result = agent.extract(payload) if prepared.eligible else LLMCallResult("skipped_no_eligible_assertion", [], None, None, agent.model_name, 0)
        valid, rejected = _validate_candidates(task, result.candidates, upstream)
        row = {"compound": model["compound"], "source_path": str(path.relative_to(ROOT)),
               "source_sha256": checksum, "source_commit": model["commit"], "model": result.model_name,
               "backend": "local_gpu", "status": result.status, "seconds": result.latency_seconds,
               "prompt_version": agent.prompt_template_version, "input_items": list(prepared.sentences),
               "raw_candidates": result.candidates, "deterministically_valid_candidates": valid,
               "rejected_candidates": rejected, "error": result.error, "database_imported": False,
               "scientific_status": "candidate_evidence_only_not_independently_validated"}
        with output.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.append(row)
        print(json.dumps({"compound": model["compound"], "status": result.status, "candidates": len(result.candidates), "passed_existing_checks": len(valid), "rejected": len(rejected), "seconds": result.latency_seconds}), flush=True)

    # Real source statement plus explicit contradiction controls. These are diagnostic
    # tests, never training evidence or independent scientific validation labels.
    metformin = next(m for m in models if m["compound"] == "metformin")
    source = ROOT / metformin["source_directory"] / "README.md"
    text = source.read_text()
    statement = next(line for line in text.splitlines() if "0.001" in line)
    stage = AutoQualityLLMStage(final_run_dir=OUT / "diagnostic_only", raw_run_dir=OUT)
    controls = []
    for label, fields, expected in [
        ("source_dose_range", {"dose_range": "0.001–2550 mg", "study_count": 39}, True),
        ("wrong_dose_unit", {"dose_range": "0.001–2550 g", "study_count": 39}, False),
        ("wrong_study_count", {"dose_range": "0.001–2550 mg", "study_count": 390}, False),
    ]:
        started = time.monotonic()
        response = stage._chat(VALIDATION_PROMPT, {"evidence_id": label, "target_compound": "metformin",
            "repaired_fields": fields, "source_context": {"combined_text": statement}})
        passed = response.get("supported") is expected
        if expected:
            quote = response.get("verbatim_support")
            passed = passed and isinstance(quote, str) and bool(quote) and quote in statement
        controls.append({"case": label, "expected_supported": expected, "response": response, "passed": passed, "seconds": time.monotonic()-started})
        print(json.dumps({"validation_control": label, "passed": passed}), flush=True)
    import torch
    summary = {"model": "Qwen3-8B-BF16", "device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0),
               "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(0),
               "documents": len(results), "routed_documents": sum(r["status"] != "skipped_no_eligible_assertion" for r in results),
               "successful_calls": sum(r["status"] == "ok" for r in results),
               "raw_candidates": sum(len(r["raw_candidates"]) for r in results),
               "passed_existing_checks": sum(len(r["deterministically_valid_candidates"]) for r in results),
               "rejected_candidates": sum(len(r["rejected_candidates"]) for r in results),
               "validation_controls": controls, "independent_accuracy_estimate": False,
               "database_imported": False, "external_llm_calls": 0}
    (RUN / "qwen_screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    if summary["successful_calls"] != summary["routed_documents"] or not all(c["passed"] for c in controls):
        raise SystemExit(1)


def verify_candidates():
    stage = AutoQualityLLMStage(final_run_dir=RUN / "verification", raw_run_dir=OUT)
    destination = RUN / "verified_evidence_candidates.jsonl"
    if destination.exists():
        raise RuntimeError("Verification output exists; preserve it and choose a new version.")
    counts = {"checked": 0, "passed": 0, "rejected": 0, "failed": 0}
    for line in (RUN / "qwen_evidence_candidates.jsonl").read_text().splitlines():
        row = json.loads(line)
        source = ROOT / row["source_path"]
        assert source.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
        text = source.read_text()
        for index, candidate in enumerate(row["deterministically_valid_candidates"]):
            evidence_id = f"{row['source_sha256'][:24]}:{index}"
            error = None
            try:
                response = stage._chat(VALIDATION_PROMPT, {"evidence_id": evidence_id,
                    "target_compound": row["compound"], "repaired_fields": candidate,
                    "source_context": {"combined_text": text}})
                quote = response.get("verbatim_support")
                passed = all(response.get(key) is True for key in ["supported", "compound_role_correct", "parameter_value_unit_correct", "reference_correct", "context_sufficient"])
                passed = passed and isinstance(quote, str) and bool(quote) and quote in text and response.get("unsupported_fields") == []
            except Exception as exc:
                passed, response, error = False, None, str(exc)
            result = {"evidence_id": evidence_id, "compound": row["compound"], "candidate": candidate,
                "source_path": row["source_path"], "source_sha256": row["source_sha256"], "source_commit": row["source_commit"],
                "verification_model": "Qwen3-8B-BF16", "response": response, "screening_passed": bool(passed),
                "error": error, "independent_verifier_model": False, "database_imported": False,
                "scientific_status": "model_documentation_evidence_not_independent_validation"}
            with destination.open("a") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts["checked"] += 1
            counts["passed" if passed else "failed" if error else "rejected"] += 1
            print(json.dumps({"compound": row["compound"], "screening_passed": bool(passed), "error": error}), flush=True)
    (RUN / "verification_summary.json").write_text(json.dumps(counts, indent=2)+"\n")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify_candidates() if args.verify_only else main()
