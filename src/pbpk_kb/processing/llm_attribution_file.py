from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from pbpk_kb.processing.final_evidence import (
    COMPOUND_ROLES,
    DDI_TERMS,
    FINAL_ATTRIBUTION_DIR,
    LLMAttributionAgent,
    MULTI_DRUG_TERMS,
    ensure_final_output_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run file-level LLM attribution refinement over reviewable evidence JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compounds", default=None, help="JSONL compound identity file. Defaults to compound_identity_for_llm.jsonl beside output-dir.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum new LLM requests. Omit for all pending rows.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--reset", action="store_true", help="Delete previous refinement outputs before running.")
    parser.add_argument("--include-all-reviewable", action="store_true", help="Send every reviewable row, not only ambiguous/low-confidence rows.")
    args = parser.parse_args()

    runner = LLMAttributionFileRunner(
        input_path=args.input,
        output_dir=args.output_dir,
        compounds_path=Path(args.compounds) if args.compounds else None,
        model_name=args.model,
        base_url=args.base_url,
        limit=args.limit,
        concurrency=max(1, args.concurrency),
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        reset=args.reset,
        include_all_reviewable=args.include_all_reviewable,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


class LLMAttributionFileRunner:
    def __init__(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        compounds_path: Path | None,
        model_name: str | None,
        base_url: str | None,
        limit: int | None,
        concurrency: int,
        timeout_seconds: int,
        max_retries: int,
        reset: bool,
        include_all_reviewable: bool,
    ) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.compounds_path = compounds_path
        self.model_name = model_name
        self.base_url = base_url
        self.limit = limit
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.reset = reset
        self.include_all_reviewable = include_all_reviewable
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_final_output_dirs(self.output_dir)
        attribution_dir = self.output_dir / FINAL_ATTRIBUTION_DIR
        self.refinements_path = attribution_dir / "llm_attribution_refinements.jsonl"
        self.state_path = attribution_dir / "llm_attribution_state.jsonl"
        self.summary_path = attribution_dir / "llm_attribution_summary.json"
        self._write_lock = Lock()

    def run(self) -> dict[str, Any]:
        started = time.time()
        if self.reset:
            for path in [self.refinements_path, self.state_path, self.summary_path]:
                path.unlink(missing_ok=True)

        processed_ids = self._load_processed_ids()
        compounds = self._load_compounds()
        rows, load_counts = self._load_rows(processed_ids)
        if self.limit is not None:
            rows = rows[: max(0, self.limit)]

        agent = LLMAttributionAgent(mode="live", model_name=self.model_name, base_url=self.base_url, timeout_seconds=self.timeout_seconds)
        counts: Counter[str] = Counter(load_counts)
        counts["processed_existing"] = len(processed_ids)
        counts["requests_planned"] = len(rows)
        counts["api_key_present"] = 1 if agent.api_key else 0
        counts["compound_options"] = len(compounds)

        with self.refinements_path.open("a", encoding="utf-8") as refinements_fh, self.state_path.open("a", encoding="utf-8") as state_fh:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(self._call_agent, agent, row, compounds): row for row in rows}
                for index, future in enumerate(as_completed(futures), start=1):
                    row = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = failed_record(row, f"worker_exception:{sanitize_error(str(exc))}")
                    status = str(record.get("llm_status") or "")
                    counts["requests_done"] += 1
                    counts[f"status:{status}"] += 1
                    if record.get("accepted_refinement"):
                        counts["accepted_refinements"] += 1
                    if record.get("changed_compound_role"):
                        counts["changed_compound_role"] += 1
                    if record.get("changed_compound_name"):
                        counts["changed_compound_name"] += 1
                    with self._write_lock:
                        write_jsonl(refinements_fh, record)
                        write_jsonl(
                            state_fh,
                            {
                                "evidence_id": row.get("evidence_id"),
                                "status": status,
                                "accepted_refinement": bool(record.get("accepted_refinement")),
                                "updated_at_epoch": round(time.time(), 3),
                            },
                        )
                    if index == 1 or index % 100 == 0:
                        print(
                            f"llm_attribution progress={index}/{len(rows)} status={status} "
                            f"accepted={counts['accepted_refinements']} changed_role={counts['changed_compound_role']}",
                            flush=True,
                        )

        summary = self._summary(counts, started, agent)
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    def _load_processed_ids(self) -> set[str]:
        processed: set[str] = set()
        if not self.state_path.exists():
            return processed
        with self.state_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                evidence_id = record.get("evidence_id")
                if evidence_id:
                    processed.add(str(evidence_id))
        return processed

    def _load_rows(self, processed_ids: set[str]) -> tuple[list[dict[str, Any]], Counter[str]]:
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        with self.input_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                counts["input_rows"] += 1
                row = json.loads(line)
                evidence_id = str(row.get("evidence_id") or "")
                if not evidence_id:
                    counts["skipped_missing_evidence_id"] += 1
                    continue
                if evidence_id in processed_ids:
                    counts["skipped_processed_existing"] += 1
                    continue
                if not self.include_all_reviewable and row.get("llm_required") is False:
                    counts["skipped_llm_not_required"] += 1
                    continue
                if not self.include_all_reviewable and "llm_required" not in row and not needs_llm_attribution(row):
                    counts["skipped_not_ambiguous"] += 1
                    continue
                if not row.get("source_text"):
                    counts["skipped_missing_source_text"] += 1
                    continue
                rows.append(row)
                counts["pending_rows"] += 1
        return rows, counts

    def _load_compounds(self) -> list[dict[str, Any]]:
        compounds_path = self.compounds_path or (self.output_dir / FINAL_ATTRIBUTION_DIR / "compound_identity_for_llm.jsonl")
        if compounds_path.exists():
            compounds: list[dict[str, Any]] = []
            with compounds_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    name = str(row.get("preferred_name") or "").strip()
                    if not name:
                        continue
                    compounds.append(
                        {
                            "compound_id": row.get("compound_id"),
                            "preferred_name": name,
                            "synonyms": row.get("synonyms") or [],
                            "pubchem_cid": row.get("pubchem_cid"),
                            "chembl_id": row.get("chembl_id"),
                            "rxnorm_rxcui": row.get("rxnorm_rxcui"),
                        }
                    )
            if compounds:
                return sorted(compounds, key=lambda item: str(item["preferred_name"]).lower())

        rows: list[dict[str, Any]] = []
        with self.input_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return build_compounds(rows)

    def _call_agent(self, agent: LLMAttributionAgent, row: dict[str, Any], compounds: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.time()
        result: dict[str, Any] = {"status": "not_called"}
        for attempt in range(self.max_retries + 1):
            result = agent.attribute(row={key: string_or_empty(value) for key, value in row.items()}, compounds=compounds)
            if result.get("status") == "ok":
                break
            if result.get("status") not in {"failed_url", "failed_http", "failed_exception"}:
                break
            if attempt < self.max_retries:
                time.sleep(1.0 + attempt)
        record = {
            "evidence_id": row.get("evidence_id"),
            "previous_compound_id": row.get("attributed_compound_id") or row.get("compound_id"),
            "previous_compound_name": row.get("attributed_compound_name") or row.get("compound_name"),
            "previous_compound_role": row.get("compound_role"),
            "previous_attribution_method": row.get("attribution_method"),
            "previous_attribution_confidence": row.get("attribution_confidence"),
            "evidence_type": row.get("evidence_type"),
            "parameter": row.get("parameter"),
            "source_id": row.get("source_id"),
            "source_location": row.get("source_location"),
            "llm_status": result.get("status"),
            "llm_model": agent.model_name,
            "source_span_binding": result.get("source_span_binding"),
            "latency_seconds": round(time.time() - started, 3),
            "accepted_refinement": False,
            "rejection_reason": None,
            "candidate": result.get("candidate") if isinstance(result.get("candidate"), dict) else None,
        }
        if result.get("status") != "ok":
            record["rejection_reason"] = result.get("error") or result.get("status")
            return record
        candidate = result.get("candidate")
        accepted, reason, refined = validate_candidate(row, compounds, candidate)
        record["accepted_refinement"] = accepted
        record["rejection_reason"] = reason
        if refined:
            record.update(refined)
            record["changed_compound_role"] = refined.get("refined_compound_role") != record["previous_compound_role"]
            record["changed_compound_name"] = normalize_name(refined.get("refined_compound_name")) != normalize_name(record["previous_compound_name"])
        else:
            record["changed_compound_role"] = False
            record["changed_compound_name"] = False
        return record

    def _summary(self, counts: Counter[str], started: float, agent: LLMAttributionAgent) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "refinements_path": str(self.refinements_path),
            "state_path": str(self.state_path),
            "model": agent.model_name,
            "base_url": agent.base_url,
            "concurrency": self.concurrency,
            "max_retries": self.max_retries,
            "limit": self.limit,
            "include_all_reviewable": self.include_all_reviewable,
            "elapsed_seconds": round(time.time() - started, 3),
            **dict(counts),
        }


def needs_llm_attribution(row: dict[str, Any]) -> bool:
    source_text = str(row.get("source_text") or "")
    role = str(row.get("compound_role") or "")
    confidence = safe_float(row.get("attribution_confidence"), 0.0)
    multi_drug = contains_any(source_text, MULTI_DRUG_TERMS)
    is_ddi = str(row.get("evidence_type") or "") == "ddi" or contains_any(source_text, DDI_TERMS)
    if confidence < 0.80:
        return True
    if multi_drug:
        return True
    if is_ddi and role in {"victim_drug", "perpetrator_drug"} and confidence < 0.86:
        return True
    if len(source_text) > 900 and (multi_drug or is_ddi):
        return True
    return False


def build_compounds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        for id_key, name_key in [("attributed_compound_id", "attributed_compound_name"), ("compound_id", "compound_name")]:
            name = str(row.get(name_key) or "").strip()
            if not name:
                continue
            key = name.lower()
            by_name.setdefault(
                key,
                {
                    "compound_id": row.get(id_key),
                    "preferred_name": name,
                },
            )
    return sorted(by_name.values(), key=lambda item: str(item["preferred_name"]).lower())


def validate_candidate(row: dict[str, Any], compounds: list[dict[str, Any]], candidate: Any) -> tuple[bool, str | None, dict[str, Any] | None]:
    if not isinstance(candidate, dict):
        return False, "missing_or_invalid_candidate", None
    source_text = str(row.get("source_text") or "")
    matched_text = str(candidate.get("matched_text") or "")
    if not matched_text or matched_text not in source_text:
        return False, "matched_text_not_copied_from_source_text", None
    if len(matched_text) > 500:
        return False, "matched_text_too_long_for_attribution", None
    role = str(candidate.get("compound_role") or "")
    if role not in COMPOUND_ROLES:
        return False, "invalid_compound_role", None
    compound_name = str(candidate.get("compound_name") or "").strip()
    compound = find_compound(compounds, compound_name)
    if compound is None:
        return False, "compound_name_not_allowed", None
    # [FIX 2026-07-02 · claude-opus-4-8] (F1) substring-validate against NAME-like terms only. Numeric identifiers
    # (pubchem_cid / rxnorm_rxcui, e.g. "4091"/"6809") must not be substring-matched against free text, or a stray
    # number (dose, year, study id) would falsely confirm the attribution and defeat the anti-hallucination guard.
    if not any(term.lower() in matched_text.lower() for term in compound_name_terms(compound)):
        return False, "matched_text_does_not_contain_selected_compound", None
    confidence = max(0.0, min(1.0, safe_float(candidate.get("confidence"), 0.5)))
    return (
        True,
        None,
        {
            "refined_compound_id": compound.get("compound_id"),
            "refined_compound_name": compound.get("preferred_name"),
            "refined_compound_role": role,
            "refined_attribution_confidence": round(confidence, 3),
            "refined_matched_text": matched_text,
            "refined_matched_synonym": candidate.get("matched_synonym"),
            "refined_attribution_reason": candidate.get("role_reason"),
            "refined_attribution_method": "llm_assisted_file_attribution",
        },
    )


def find_compound(compounds: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    lowered = name.strip().lower()
    for compound in compounds:
        if lowered in {term.lower() for term in compound_validation_terms(compound)}:
            return compound
    return None


def compound_validation_terms(compound: dict[str, Any]) -> list[str]:
    # Includes identifiers; used only for EXACT-match name resolution in find_compound (safe).
    terms = [
        str(compound.get("preferred_name") or ""),
        str(compound.get("pubchem_cid") or ""),
        str(compound.get("chembl_id") or ""),
        str(compound.get("rxnorm_rxcui") or ""),
    ]
    synonyms = compound.get("synonyms") or []
    if isinstance(synonyms, str):
        terms.extend(parse_synonyms_value(synonyms))
    elif isinstance(synonyms, list):
        terms.extend(str(item) for item in synonyms)
    return [term.strip() for term in terms if term and term.strip()]


def compound_name_terms(compound: dict[str, Any]) -> list[str]:
    # [FIX 2026-07-02 · claude-opus-4-8] (F1) NAME-like terms only, for substring validation against source text:
    # preferred_name + synonyms, kept only when >=3 chars and containing a letter. Excludes numeric/short identifiers.
    terms = [str(compound.get("preferred_name") or "")]
    synonyms = compound.get("synonyms") or []
    if isinstance(synonyms, str):
        terms.extend(parse_synonyms_value(synonyms))
    elif isinstance(synonyms, list):
        terms.extend(str(item) for item in synonyms)
    out: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if len(cleaned) >= 3 and any(ch.isalpha() for ch in cleaned):
            out.append(cleaned)
    return out


def parse_synonyms_value(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1]
    return [item.strip().strip('"') for item in stripped.split(",") if item.strip()]


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def failed_record(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "evidence_id": row.get("evidence_id"),
        "llm_status": "failed_worker",
        "accepted_refinement": False,
        "rejection_reason": reason,
        "changed_compound_role": False,
        "changed_compound_name": False,
    }


def string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_error(error: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_*\\-]{6,}", "sk-***", error)


def write_jsonl(file: Any, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    file.flush()


if __name__ == "__main__":
    main()
