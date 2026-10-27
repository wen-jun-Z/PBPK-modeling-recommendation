from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value
from pbpk_kb.processing.chunk_classifier import (
    PreparedLLMInput,
    _parameter_is_explicitly_owned_by_other,
    classify_chunk,
    prepare_llm_input,
    source_assertion_completeness_error,
    source_quote_unit_boundary_error,
)
from pbpk_kb.processing.llm_agent import LLM_EXTRACTION_PROMPT, LLMCallResult, LLMExtractionAgent, extraction_prompt
from pbpk_kb.processing.raw_processing_paths import RawProcessingLayout, raw_processing_layout
from pbpk_kb.processing.raw_processor import _hash_text, _stable_id


EvidenceType = Literal[
    "physicochemical",
    "clinical_pk",
    "adme",
    "ddi",
    "special_population",
    "tissue_expression",
    "public_pbpk_model",
]

PROCESSED_LLM_STATUSES = {"ok", "reused_same_target_input_hash"}
VALIDATOR_VERSION = "pbpk-llm-validator-v25"
LLM_ROUTING_VERSION = "pbpk-llm-routing-v11"
_CANONICAL_INDEX_CACHE: dict[tuple, tuple[dict, Counter]] = {}
SOURCE_CANONICAL_PRIORITY = {
    "dailymed": 0,
    "pmc_oa": 0,
    "pkdb": 1,
    "osp_pbpk_model_library": 1,
    "pubmed": 2,
    "openfda_drug_label": 3,
    "clinicaltrials": 4,
}
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:(?<!%)[-+])?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
TIME_PARAMETER_SOURCE = r"\bt\s*max\b|time to (?:reach )?(?:peak|maximum)(?: concentration)?|half[- ]life"
TIME_UNIT_SOURCE = r"minutes?|min|hours?|hrs?|h|days?|d"
EXCRETION_DELAY_SOURCE = (
    r"\bexcret\w*\b[^.;\n]{0,80}\bdelay(?:s|ed|ing)?\b|"
    r"\bdelay(?:s|ed|ing)?\b[^.;\n]{0,80}\bexcret\w*\b"
)
TIME_RANGE_SOURCE = re.compile(
    rf"(?:{TIME_PARAMETER_SOURCE})\s*\)?\s*"
    rf"(?:(?:ranged?|ranges|was|were|is|of|from|between)\s+){{0,4}}"
    rf"(?P<low>\d+(?:\.\d+)?)\s*(?P<low_unit>{TIME_UNIT_SOURCE})\b\s*"
    rf"(?P<join>to|and|[-–])\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<high_unit>{TIME_UNIT_SOURCE})\b",
    re.IGNORECASE,
)
DIRECT_PERCENT_CHANGE_RE = re.compile(
    r"(?P<value>[-+]?(?:\d+(?:\.\d+)?))\s*%\s*(?P<direction>reduction|decrease|increase|change|lower|higher)",
    flags=re.IGNORECASE,
)
PREFIX_PERCENT_CHANGE_RE = re.compile(
    r"(?P<direction>increased|decreased|reduced|lowered|higher|lower|increase|decrease|reduction)"
    r"[^.;%]{0,80}?(?P<value>[-+]?(?:\d+(?:\.\d+)?))\s*%",
    flags=re.IGNORECASE,
)
BOUNDED_PERCENT_RE = re.compile(
    r"(?P<qualifier>less\s+than|under|below|<|≤|up\s+to|at\s+most|greater\s+than|more\s+than|over|above|>|≥|at\s+least)"
    r"\s*(?P<value>[-+]?(?:\d+(?:\.\d+)?))\s*%",
    flags=re.IGNORECASE,
)
DIRECT_FOLD_CHANGE_RE = re.compile(
    r"(?P<value>[-+]?(?:\d+(?:\.\d+)?))\s*[- ]?\s*fold\b",
    flags=re.IGNORECASE,
)
COMPARATIVE_TERM_RE = re.compile(r"\b(?:vs\.?|versus|compared\s+(?:with|to))\b", flags=re.IGNORECASE)
NUMERIC_VS_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d+)?)\s*(?:vs\.?|versus)\s*[-+]?(?:\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
RESPECTIVELY_RE = re.compile(r"\brespectively\b", flags=re.IGNORECASE)
DIALYSIS_CLEARANCE_RE = re.compile(
    r"dialy(?:sis|zable|sable|zed|sed)|ha?emodialysis|peritoneal\s+dialysis|"
    r"(?:kidney|renal)\s+replacement\s+therapy|ha?emo(?:adsorption|filtration|perfusion)|"
    r"\b(?:CRRT|CKRT|CVVHDF|CVVH|CVVHD)\b|extracorporeal\s+(?:clearance|removal|treatment)",
    flags=re.IGNORECASE,
)
RENAL_FUNCTION_MARKER_RE = re.compile(r"creatinine\s+clearance|\bcrcl\b|glomerular\s+filtration|\begfr\b", flags=re.IGNORECASE)
PBPK_MODEL_MARKER_RE = re.compile(
    r"\bpbpk\b|physiolog(?:ically|ical)[- ]based|whole[- ]body|\bpk[- ]?sim\b|\bsimcyp\b",
    flags=re.IGNORECASE,
)
SPECIAL_POPULATION_MARKER_RE = re.compile(
    r"renal\s+(?:impairment|failure|dysfunction|disease)|decreased\s+renal\s+function|\barf\b|"
    r"hepatic\s+(?:impairment|failure|dysfunction)|liver\s+disease|cirrhosis|"
    r"pediatric|paediatric|child(?:ren)?|infant|neonat|adolesc|preterm|gestational|"
    r"elderly|geriatric|older\s+(?:adult|patient|subject)|old\s+age|"
    r"pregnan|gestation|maternal|fetal|foetal|lactat|breast[- ]?feed|"
    r"obes|body\s+mass\s+index|\bbmi\b|"
    r"genotype|polymorph|poor\s+metabolizer|rapid\s+metabolizer|pharmacogen|"
    r"alcoholic|alcohol\s+use|chronic\s+alcohol|"
    r"congestive\s+heart\s+failure|critically\s+ill|intensive\s+care|malnutrition",
    flags=re.IGNORECASE,
)
PARAMETER_DOMAIN_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "physicochemical": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"molecular[_ ]?(?:weight|mass|formula)",
            r"\bpka\b|\bpkb\b|\blogp\b|\blogd\b",
            r"solubility|permeability|polar[_ ]?surface[_ ]?area|\btpsa\b|ionization|formulation",
        )
    ),
    "adme": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"absorption|distribution|metabolism|metabolic|excretion|elimination",
            r"bioavailability|fraction[_ ]?unbound|protein[_ ]?binding|blood[_ -]?to[_ -]?plasma",
            r"intrinsic[_ ]?clearance|renal[_ ]?clearance|hepatocyte|microsom|stability|permeability|efflux",
            r"substrate|inhibitor|inhibition|inducer|induction|transporter|enzyme",
            r"\bcyp[0-9a-z]+\b|\bugt[0-9a-z]+\b|p[_ -]?gp|oatp|oct[0-9]|oat[0-9]|bcrp|mrp[0-9]",
        )
    ),
    "clinical_pk": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bcmax\b|maximum[_ ]?(?:plasma[_ ]?)?concentration|peak[_ ]?concentration",
            r"\bauc\b|area[_ ]?under[_ ]?(?:the[_ ]?)?curve|\btmax\b|time[_ ]?to[_ ]?(?:peak|maximum)",
            r"half[_ -]?life|clearance|volume[_ ]?of[_ ]?distribution|\bvd\b|\bvss\b",
            r"bioavailability|accumulation|trough|steady[_ -]?state[_ ]?concentration",
            r"plasma[_ ]?concentration|blood[_ ]?concentration|serum[_ ]?concentration|exposure",
            r"mean[_ ]?residence[_ ]?time|\bmrt\b|renal[_ ]?clearance",
        )
    ),
    "ddi": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bcmax\b|\bauc\b|\btmax\b|half[_ -]?life|clearance|bioavailability|exposure|concentration",
            r"ratio|fold[_ -]?change|interaction[_ ]?type|mechanism",
            r"^excretion delay$",
            r"substrate|inhibitor|inhibition|inducer|induction|transporter|enzyme",
            r"\bcyp[0-9a-z]+\b|\bugt[0-9a-z]+\b|p[_ -]?gp|oatp|oct[0-9]|oat[0-9]|bcrp|mrp[0-9]",
            r"\bki\b|\bic50\b|fraction[_ ]?metabolized|\bfm\b",
        )
    ),
    "special_population": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bcmax\b|\bauc\b|\btmax\b|half[_ -]?life|clearance|volume[_ ]?of[_ ]?distribution",
            r"bioavailability|accumulation|trough|concentration|exposure|ratio",
            r"absorption|distribution|metabolism|excretion|elimination|fraction[_ ]?unbound|protein[_ ]?binding|fraction[_ ]?excreted|excreted[_ ]?unchanged",
        )
    ),
    "tissue_expression": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (r"expression|abundance|protein[_ ]?level|rna[_ ]?level|\btpm\b|\bntpm\b|\bncpm\b",)
    ),
    "public_pbpk_model": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"model|software|species|population|route",
            r"model[_ ]?file|parameter[_ ]?file|evaluation|validation|qualification",
        )
    ),
}


class LLMEvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)

    target_compound: str = Field(min_length=1)
    evidence_type: EvidenceType
    parameter: str = Field(min_length=1)
    value: float | None
    value_low: float | None = None
    value_high: float | None = None
    value_qualifier: str | None = None
    unit: str | None
    condition: str | None
    source_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_review: Literal[True]


@dataclass
class PendingTask:
    chunk: dict[str, Any]
    target_compound: str
    status_record: dict[str, Any]
    request_record: dict[str, Any]
    input_hash: str


@dataclass(frozen=True)
class UpstreamIndex:
    raw_assets: dict[str, dict[str, Any]]
    parsed_document_ids: set[str]
    parsed_table_ids: set[str]
    target_compounds: set[str]
    document_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    pmc_fulltext_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CanonicalSentence:
    chunk_id: str
    source_id: str
    source_location: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provenance-validated LLM extraction on existing text chunks.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--llm-max-requests", type=int, default=None)
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--classify-only", action="store_true")
    parser.add_argument("--llm-include-existing", action="store_true")
    parser.add_argument("--no-reuse-by-text-hash", action="store_false", dest="reuse_by_text_hash", default=True)
    parser.add_argument("--reset-live-ledger", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", dest="fail_fast", default=True)
    parser.add_argument("--no-fail-fast", action="store_false", dest="fail_fast")
    args = parser.parse_args()

    summary = run_llm_extraction(
        processed_dir=Path(args.processed_dir),
        llm_max_requests=args.llm_max_requests,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        concurrency=args.concurrency,
        min_chars=args.min_chars,
        fail_fast=args.fail_fast,
        classify_only=args.classify_only,
        llm_include_existing=args.llm_include_existing,
        reuse_by_text_hash=args.reuse_by_text_hash,
        reset_live_ledger=args.reset_live_ledger,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_llm_extraction(
    *,
    processed_dir: Path,
    llm_max_requests: int | None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    concurrency: int = 1,
    min_chars: int,
    fail_fast: bool,
    classify_only: bool = False,
    llm_include_existing: bool = False,
    reuse_by_text_hash: bool = True,
    reset_live_ledger: bool = False,
) -> dict[str, Any]:
    layout = raw_processing_layout(processed_dir)
    layout.ensure_dirs()
    chunks_path = layout.path("text_chunks.jsonl")
    if not chunks_path.exists():
        raise FileNotFoundError(f"text_chunks.jsonl not found: {chunks_path}")

    pipeline_state = _validate_prerequisites(layout)
    upstream = _load_upstream_index(layout)
    agent = LLMExtractionAgent(
        mode="live",
        model_name=llm_model,
        base_url=llm_base_url,
        max_retries=llm_max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    prompt_path = layout.path("llm_agent_prompt.md", for_write=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        f"# {agent.prompt_template_version}\n\n```text\n{extraction_prompt({}).rstrip()}\n```\n",
        encoding="utf-8",
    )

    requests_path = layout.path("llm_extraction_requests_live.jsonl", for_write=True)
    responses_path = layout.path("llm_extraction_responses_live.jsonl", for_write=True)
    stable_requests_path = layout.path("llm_extraction_requests.jsonl", for_write=True)
    stable_responses_path = layout.path("llm_extraction_responses.jsonl", for_write=True)
    classification_samples_path = layout.path("llm_classification_samples.jsonl", for_write=True)
    llm_evidence_path = layout.path("evidence_candidates_llm.jsonl", for_write=True)
    deterministic_evidence_path = layout.path("evidence_candidates_deterministic.jsonl")
    merged_evidence_path = layout.path("evidence_candidates.jsonl", for_write=True)
    failures_path = layout.path("processing_failures.jsonl", for_write=True)
    status_path = layout.path("chunk_processing_status.jsonl", for_write=True)
    status_temp_path = status_path.with_name(f".{status_path.name}.llm.tmp")
    progress_path = layout.path("llm_extract_progress.json", for_write=True)
    summary_path = layout.path("llm_extract_summary.json", for_write=True)

    if reset_live_ledger:
        for path in [requests_path, responses_path, stable_requests_path, stable_responses_path, llm_evidence_path]:
            path.unlink(missing_ok=True)
        _remove_stage_failures(failures_path, stage="llm_extract")
        _initialize_merged_evidence(deterministic_evidence_path, merged_evidence_path, force=True)
    else:
        _initialize_merged_evidence(deterministic_evidence_path, merged_evidence_path, force=False)

    response_by_request = _load_processed_responses(responses_path, agent.prompt_template_version)
    candidates_by_input_hash = _load_valid_candidates_by_input_hash(responses_path, agent.prompt_template_version)
    request_attempts = _load_request_attempts(requests_path)
    existing_evidence_ids = _load_existing_evidence_ids(merged_evidence_path)
    evidence_by_chunk = _load_evidence_index(merged_evidence_path)
    existing_failure_ids = _load_failure_ids(failures_path)

    started_at = _utc_now()
    counts: Counter[str] = Counter()
    evidence_added = 0
    api_failure_streak = 0
    stop_submissions = False
    stop_reason: str | None = None
    recovered_request_ids: set[str] = set()
    eligible_request_ids: set[str] = set()
    max_workers = max(1, concurrency)

    if not classify_only:
        _write_pipeline_state(
            layout,
            pipeline_state,
            status="in_progress",
            accepted=False,
            started_at=started_at,
            summary_path=summary_path,
        )
    _write_progress(
        progress_path,
        run_id=str(pipeline_state.get("run_id") or ""),
        status="classifying" if classify_only else "in_progress",
        counts=counts,
        started_at=started_at,
    )

    status_temp_path.parent.mkdir(parents=True, exist_ok=True)
    status_temp_path.unlink(missing_ok=True)
    inflight: dict[Future[LLMCallResult], PendingTask] = {}
    future_by_input_hash: dict[str, Future[LLMCallResult]] = {}
    waiting_by_input_hash: dict[str, list[PendingTask]] = {}
    classify_input_hashes: set[str] = set()
    classification_samples: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    _write_progress(
        progress_path,
        run_id=str(pipeline_state.get("run_id") or ""),
        status="building_high_precision_routing_index",
        counts=counts,
        started_at=started_at,
    )
    canonical_by_fingerprint, routing_index_counts = _cached_canonical_sentence_index(
        layout, min_chars=min_chars, upstream=upstream)

    def write_status(handle: TextIO, row: dict[str, Any]) -> None:
        _write_jsonl_line(handle, row)
        counts["status_rows"] += 1
        processing_status = str(row.get("processing_status") or "unknown")
        counts[f"processing:{processing_status}"] += 1

    def append_failure(row: dict[str, Any]) -> None:
        failure_id = str(row.get("failure_id") or "")
        if not failure_id or failure_id in existing_failure_ids:
            return
        _append_jsonl(failures_path, row)
        existing_failure_ids.add(failure_id)
        counts["failure_rows_added"] += 1

    def record_classification_sample(chunk: dict[str, Any], target_compound: str, prepared: PreparedLLMInput) -> None:
        source_id = str(chunk.get("source_id") or "unknown")
        chunk_id = str(chunk.get("chunk_id") or "")
        rank = sha256(f"{source_id}\n{chunk_id}".encode("utf-8", errors="replace")).hexdigest()
        row = {
            "run_id": chunk.get("run_id"),
            "routing_version": LLM_ROUTING_VERSION,
            "source_id": source_id,
            "target_compound": target_compound,
            "chunk_id": chunk_id,
            "raw_asset_id": chunk.get("raw_asset_id"),
            "parsed_document_id": chunk.get("parsed_document_id"),
            "source_location": chunk.get("source_location"),
            "checksum": chunk.get("checksum"),
            "input_sentence_fingerprints": list(prepared.sentence_fingerprints),
            "input_text_or_table": prepared.input_text,
            "sample_rank": rank,
        }
        bucket = classification_samples.setdefault(source_id, [])
        bucket.append((rank, row))
        bucket.sort(key=lambda item: item[0])
        del bucket[10:]

    def add_evidence(task: PendingTask, candidates: list[dict[str, Any]]) -> int:
        nonlocal evidence_added
        added = _append_evidence_rows(
            task.chunk,
            task.target_compound,
            candidates,
            llm_evidence_path=llm_evidence_path,
            merged_evidence_path=merged_evidence_path,
            existing_evidence_ids=existing_evidence_ids,
            counts=counts,
        )
        evidence_added += added
        counts["evidence_added"] += added
        return added

    def record_reuse(task: PendingTask, candidates: list[dict[str, Any]], status_handle: TextIO, reused_from: str | None) -> None:
        request = {**task.request_record, "request_mode": "reuse", "reused_from_request_id": reused_from}
        _append_jsonl(requests_path, request)
        response = _response_record(
            task,
            status="reused_same_target_input_hash",
            model_name=agent.model_name,
            prompt_template_version=agent.prompt_template_version,
            candidates=candidates,
            rejected_candidates=[],
            schema_validation_status="valid_reuse" if candidates else "valid_empty_reuse",
            error=None,
            latency_seconds=0.0,
            output_hash=_hash_json(candidates),
            reused_from_request_id=reused_from,
        )
        _append_jsonl(responses_path, response)
        added = add_evidence(task, candidates)
        counts["llm_reused_same_target_input_hash"] += 1
        write_status(
            status_handle,
            {
                **task.status_record,
                "processing_status": "llm_reused_same_target_input_hash",
                "llm_status": "reused_same_target_input_hash",
                "llm_candidate_count": len(candidates),
                "llm_evidence_added": added,
            },
        )

    def save_request_progress() -> None:
        _write_progress(progress_path, run_id=str(pipeline_state.get("run_id") or ""),
                        status="in_progress", counts=counts, started_at=started_at, stop_reason=stop_reason)

    def finish_future(future: Future[LLMCallResult], status_handle: TextIO) -> None:
        nonlocal api_failure_streak, stop_submissions, stop_reason
        task = inflight.pop(future)
        future_by_input_hash.pop(task.input_hash, None)
        waiting = waiting_by_input_hash.pop(task.input_hash, [])
        try:
            result = future.result()
        except Exception as exc:
            result = LLMCallResult("failed_exception", [], None, str(exc), agent.model_name, None)

        if result.status != "ok":
            api_failure_streak += 1
            counts[f"api_status:{result.status}"] += 1
            counts["unresolved_api_failures"] += 1
            _append_jsonl(
                responses_path,
                _response_record(
                    task,
                    status=result.status,
                    model_name=result.model_name,
                    prompt_template_version=agent.prompt_template_version,
                    candidates=[],
                    rejected_candidates=[],
                    schema_validation_status="not_validated",
                    error=result.error,
                    latency_seconds=result.latency_seconds,
                    output_hash=_hash_optional_text(result.raw_response),
                    failed_raw_output=result.raw_response,
                ),
            )
            append_failure(_api_failure(task, result))
            write_status(
                status_handle,
                {**task.status_record, "processing_status": "llm_failed", "llm_status": result.status, "llm_error": result.error},
            )
            for dependent in waiting:
                counts["llm_pending"] += 1
                write_status(
                    status_handle,
                    {
                        **dependent.status_record,
                        "processing_status": "pending_failed_shared_input",
                        "llm_status": result.status,
                        "llm_error": result.error,
                    },
                )
            if fail_fast and (_is_auth_or_quota_failure(result.error) or api_failure_streak >= 3):
                stop_submissions = True
                stop_reason = "auth_or_quota_failure" if _is_auth_or_quota_failure(result.error) else "three_consecutive_api_failures"
            save_request_progress()
            return

        api_failure_streak = 0
        counts["api_status:ok"] += 1
        valid_candidates, rejected = _validate_candidates(task, result.candidates, upstream)
        counts["candidate_rejected"] += len(rejected)
        recovered_request_ids.add(str(task.request_record["request_id"]))
        for rejection in rejected:
            append_failure(_candidate_failure(task, rejection))
        schema_status = _schema_validation_status(valid_candidates, rejected, len(result.candidates))
        response = _response_record(
            task,
            status="ok",
            model_name=result.model_name,
            prompt_template_version=agent.prompt_template_version,
            candidates=valid_candidates,
            rejected_candidates=rejected,
            schema_validation_status=schema_status,
            error=None,
            latency_seconds=result.latency_seconds,
            output_hash=_hash_optional_text(result.raw_response) or _hash_json(result.candidates),
            assessment=result.assessment,
        )
        _append_jsonl(responses_path, response)
        candidates_by_input_hash[task.input_hash] = (task.request_record["request_id"], valid_candidates)
        added = add_evidence(task, valid_candidates)
        write_status(
            status_handle,
            {
                **task.status_record,
                "processing_status": "llm_processed",
                "llm_status": "ok",
                "schema_validation_status": schema_status,
                "llm_candidate_count": len(valid_candidates),
                "llm_rejected_candidate_count": len(rejected),
                "llm_evidence_added": added,
            },
        )
        for dependent in waiting:
            record_reuse(dependent, valid_candidates, status_handle, task.request_record["request_id"])
        save_request_progress()

    def drain(status_handle: TextIO, *, all_done: bool = False) -> None:
        if not inflight:
            return
        done, _ = wait(inflight, return_when=ALL_COMPLETED if all_done else FIRST_COMPLETED)
        for future in done:
            finish_future(future, status_handle)

    try:
        with status_temp_path.open("w", encoding="utf-8") as status_handle, ThreadPoolExecutor(max_workers=max_workers) as executor, chunks_path.open(
            "r", encoding="utf-8", errors="replace"
        ) as chunks:
            for line in chunks:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                counts["chunks_seen"] += 1
                if counts["chunks_seen"] % 10000 == 0:
                    status_handle.flush()
                    _write_progress(
                        progress_path,
                        run_id=str(pipeline_state.get("run_id") or ""),
                        status="classifying" if classify_only else "in_progress",
                        counts=counts,
                        started_at=started_at,
                        stop_reason=stop_reason,
                    )
                chunk_id = str(chunk.get("chunk_id") or "")
                existing = evidence_by_chunk.get(chunk_id, {"count": 0, "methods": set(), "types": set()})
                decision = classify_chunk(chunk, min_chars=min_chars, existing_evidence_count=int(existing["count"]))
                counts[f"decision:{decision.status}"] += 1
                target_compound, target_status = _resolve_target_compound(chunk, upstream.target_compounds)
                request_id = _request_id(chunk_id, target_compound)
                previous = response_by_request.get(request_id)
                status_record = _status_record(
                    chunk,
                    decision,
                    existing,
                    request_id=request_id,
                    previous_status=str(previous.get("status")) if previous else None,
                    target_compound=target_compound,
                    target_resolution_status=target_status,
                )

                if not decision.llm_required:
                    write_status(status_handle, {**status_record, "processing_status": decision.status})
                    continue

                if not target_compound:
                    counts["llm_explicitly_skipped"] += 1
                    write_status(
                        status_handle,
                        {**status_record, "llm_required": False, "processing_status": "skipped_unresolved_target_compound"},
                    )
                    continue

                publication_duplicate_reason = _publication_duplicate_reason(chunk, upstream)
                if publication_duplicate_reason:
                    counts["llm_explicitly_skipped"] += 1
                    write_status(
                        status_handle,
                        {
                            **status_record,
                            "llm_required": False,
                            "processing_status": publication_duplicate_reason,
                            "routing_version": LLM_ROUTING_VERSION,
                        },
                    )
                    continue

                prepared = prepare_llm_input(chunk, target_compound)
                if not prepared.eligible:
                    counts["llm_explicitly_skipped"] += 1
                    write_status(
                        status_handle,
                        {
                            **status_record,
                            "llm_required": False,
                            "processing_status": prepared.status,
                            "high_precision_reason": prepared.reason,
                            "routing_version": LLM_ROUTING_VERSION,
                        },
                    )
                    continue

                prepared, duplicate_sentences = _canonical_prepared_input(chunk, prepared, canonical_by_fingerprint)
                if not prepared.eligible:
                    counts["llm_explicitly_skipped"] += 1
                    counts["llm_duplicate_sentences"] += len(duplicate_sentences)
                    write_status(
                        status_handle,
                        {
                            **status_record,
                            "llm_required": False,
                            "processing_status": prepared.status,
                            "high_precision_reason": prepared.reason,
                            "duplicate_sentences": duplicate_sentences,
                            "routing_version": LLM_ROUTING_VERSION,
                        },
                    )
                    continue

                counts["llm_required"] += 1
                counts[f"llm_source:{chunk.get('source_id') or 'unknown'}"] += 1
                counts["llm_candidate_sentences"] += len(prepared.sentences)
                counts["llm_duplicate_sentences"] += len(duplicate_sentences)
                status_record = {
                    **status_record,
                    "llm_required": True,
                    "high_precision_reason": prepared.reason,
                    "llm_input_sentence_count": len(prepared.sentences),
                    "llm_input_sentence_fingerprints": list(prepared.sentence_fingerprints),
                    "llm_duplicate_sentence_count": len(duplicate_sentences),
                    "duplicate_sentences": duplicate_sentences,
                    "routing_version": LLM_ROUTING_VERSION,
                }
                eligible_request_ids.add(request_id)

                if int(existing["count"]) > 0 and not llm_include_existing:
                    counts["llm_existing_covered"] += 1
                    write_status(status_handle, {**status_record, "processing_status": "existing_extracted_no_llm"})
                    continue

                request_record = _request_record(agent, chunk, target_compound, request_attempts, prepared)
                input_hash = str(request_record["input_hash"])

                if (previous and str(previous.get("status")) in PROCESSED_LLM_STATUSES
                        and previous.get("input_hash") == input_hash):
                    candidates = previous.get("candidates") if isinstance(previous.get("candidates"), list) else []
                    recovery_task = PendingTask(
                        chunk=chunk,
                        target_compound=target_compound,
                        status_record=status_record,
                        request_record=request_record,
                        input_hash=str(previous.get("input_hash") or ""),
                    )
                    added = add_evidence(recovery_task, [row for row in candidates if isinstance(row, dict)])
                    # A matching success remains a recovery after a restart,
                    # including valid empty results. The loader has checked the
                    # current prompt/validator, and the input hash matched above.
                    recovered_request_ids.add(request_id)
                    counts["llm_already_ok"] += 1
                    write_status(
                        status_handle,
                        {
                            **status_record,
                            "processing_status": "llm_processed_existing",
                            "llm_status": previous.get("status"),
                            "llm_evidence_recovered": added,
                        },
                    )
                    continue

                if classify_only:
                    counts["llm_pending"] += 1
                    record_classification_sample(chunk, target_compound, prepared)
                    if input_hash in classify_input_hashes:
                        counts["llm_classify_duplicate_input_hash"] += 1
                        processing_status = "pending_llm_reuse_same_target_input_hash"
                    else:
                        classify_input_hashes.add(input_hash)
                        processing_status = "pending_llm_classify_only"
                    write_status(status_handle, {**status_record, "processing_status": processing_status, "input_hash": input_hash})
                    continue

                task = PendingTask(chunk, target_compound, status_record, request_record, input_hash)

                if reuse_by_text_hash and input_hash in candidates_by_input_hash:
                    reused_from, cached_candidates = candidates_by_input_hash[input_hash]
                    record_reuse(task, cached_candidates, status_handle, reused_from)
                    continue

                if reuse_by_text_hash and input_hash in future_by_input_hash:
                    waiting_by_input_hash.setdefault(input_hash, []).append(task)
                    counts["llm_waiting_same_target_input_hash"] += 1
                    continue

                if stop_submissions:
                    counts["llm_pending"] += 1
                    write_status(status_handle, {**status_record, "processing_status": "pending_llm_stage_stopped", "stop_reason": stop_reason})
                    continue

                if llm_max_requests is not None and counts["api_requests_submitted"] >= llm_max_requests:
                    counts["llm_pending"] += 1
                    write_status(status_handle, {**status_record, "processing_status": "pending_llm_request_limit"})
                    continue

                request_record["request_mode"] = "live"
                _append_jsonl(requests_path, request_record)
                future = executor.submit(agent.extract, request_record)
                inflight[future] = task
                future_by_input_hash[input_hash] = future
                counts["api_requests_submitted"] += 1
                if len(inflight) >= max_workers:
                    drain(status_handle)

            while inflight:
                drain(status_handle, all_done=True)
            status_handle.flush()
            os.fsync(status_handle.fileno())
        os.replace(status_temp_path, status_path)
    except Exception as exc:
        if not classify_only:
            _write_pipeline_state(
                layout,
                pipeline_state,
                status="failed",
                accepted=False,
                started_at=started_at,
                summary_path=summary_path,
                error=str(exc),
            )
        _write_progress(
            progress_path,
            run_id=str(pipeline_state.get("run_id") or ""),
            status="failed",
            counts=counts,
            started_at=started_at,
            stop_reason=str(exc),
        )
        raise

    classification_sample_rows = [
        row
        for source_id in sorted(classification_samples)
        for _, row in classification_samples[source_id]
    ]
    if classify_only:
        _write_jsonl_atomic(classification_samples_path, classification_sample_rows)

    _resolve_recovered_api_failures(failures_path, recovered_request_ids)
    if not classify_only:
        _resolve_ineligible_api_failures(failures_path, eligible_request_ids)
    unresolved_stage_failures = _count_unresolved_stage_failures(failures_path, stage="llm_extract")
    accepted = bool(
        not classify_only
        and counts["status_rows"] == counts["chunks_seen"]
        and counts["llm_pending"] == 0
        and unresolved_stage_failures == 0
    )
    stage_status = "completed" if accepted else ("classified" if classify_only else "in_progress")
    completed_at = _utc_now()
    summary = {
        "stage": "llm_extract",
        "status": stage_status,
        "accepted": accepted,
        "run_id": pipeline_state.get("run_id"),
        "output_dir": str(layout.root),
        "started_at": started_at,
        "completed_at": completed_at,
        "prompt_template_version": agent.prompt_template_version,
        "routing_version": LLM_ROUTING_VERSION,
        "model_name": agent.model_name,
        "base_url": agent.base_url,
        "api_key_present": bool(agent.api_key),
        "concurrency": max_workers,
        "request_limit": llm_max_requests,
        "classify_only": classify_only,
        "chunks_seen": counts["chunks_seen"],
        "status_row_count": counts["status_rows"],
        "llm_required_chunk_count": counts["llm_required"],
        "llm_required_source_counts": _counter_subset(counts, "llm_source:"),
        "llm_candidate_sentence_count": counts["llm_candidate_sentences"],
        "llm_duplicate_sentence_count": counts["llm_duplicate_sentences"],
        "classify_unique_input_count": len(classify_input_hashes) if classify_only else None,
        "classify_duplicate_input_hash_count": counts["llm_classify_duplicate_input_hash"],
        "routing_index_counts": dict(sorted(routing_index_counts.items())),
        "classification_sample_count": len(classification_sample_rows),
        "classification_samples_path": str(classification_samples_path),
        "existing_evidence_covered_count": counts["llm_existing_covered"],
        "already_processed_count": counts["llm_already_ok"],
        "explicitly_skipped_count": counts["llm_explicitly_skipped"],
        "pending_count": counts["llm_pending"],
        "api_request_count": counts["api_requests_submitted"],
        "reused_same_target_input_hash_count": counts["llm_reused_same_target_input_hash"],
        "waiting_same_target_input_hash_count": counts["llm_waiting_same_target_input_hash"],
        "evidence_added_count": evidence_added,
        "duplicate_evidence_skipped_count": counts["duplicate_evidence"],
        "rejected_candidate_count": counts["candidate_rejected"],
        "failure_rows_added": counts["failure_rows_added"],
        "unresolved_api_failure_count": unresolved_stage_failures,
        "stop_reason": stop_reason,
        "classification_counts": _counter_subset(counts, "decision:"),
        "processing_counts": _counter_subset(counts, "processing:"),
        "api_status_counts": _counter_subset(counts, "api_status:"),
        "requests_path": str(requests_path),
        "responses_path": str(responses_path),
        "llm_evidence_path": str(llm_evidence_path),
        "merged_evidence_path": str(merged_evidence_path),
        "status_path": str(status_path),
        "processing_failures_path": str(failures_path),
    }
    _write_json_atomic(summary_path, summary)
    if not classify_only:
        _update_processing_summary(layout, summary)
    _write_progress(
        progress_path,
        run_id=str(pipeline_state.get("run_id") or ""),
        status=stage_status,
        counts=counts,
        started_at=started_at,
        stop_reason=stop_reason,
    )
    if not classify_only:
        _write_pipeline_state(
            layout,
            pipeline_state,
            status=stage_status,
            accepted=accepted,
            started_at=started_at,
            summary_path=summary_path,
            completed_at=completed_at if accepted else None,
            stop_reason=stop_reason,
        )
    if accepted:
        _copy_atomic(requests_path, stable_requests_path)
        _copy_atomic(responses_path, stable_responses_path)
    return summary


def _request_record(
    agent: LLMExtractionAgent,
    chunk: dict[str, Any],
    target_compound: str,
    request_attempts: Counter[str],
    prepared: PreparedLLMInput,
) -> dict[str, Any]:
    request_id = _request_id(str(chunk.get("chunk_id") or ""), target_compound)
    request_attempts[request_id] += 1
    payload = {
        "request_id": request_id,
        "run_id": chunk.get("run_id"),
        "source_id": chunk.get("source_id"),
        "target_compound": target_compound,
        "raw_asset_id": chunk.get("raw_asset_id"),
        "storage_path": chunk.get("storage_path"),
        "source_url": chunk.get("source_url"),
        "parsed_document_id": chunk.get("parsed_document_id"),
        "parsed_table_id": chunk.get("parsed_table_id"),
        "chunk_id": chunk.get("chunk_id"),
        "content_kind": "text_chunk",
        "source_location": chunk.get("source_location"),
        "checksum": chunk.get("checksum"),
        "input_text_or_table": prepared.input_text,
        "input_items": list(prepared.sentences),
        "input_sentence_fingerprints": list(prepared.sentence_fingerprints),
        "routing_version": LLM_ROUTING_VERSION,
        "attempt_number": request_attempts[request_id],
        "created_at": _utc_now(),
    }
    return agent.build_request_record(payload)


def _cached_canonical_sentence_index(layout, *, min_chars, upstream):
    """Reuse the classification index only while all immutable inputs match."""
    signatures = []
    for name in ('raw_inventory.jsonl', 'parsed_documents.jsonl', 'parsed_tables.jsonl', 'text_chunks.jsonl'):
        path = layout.path(name).resolve()
        stat = path.stat()
        signatures.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    key = (LLM_ROUTING_VERSION, min_chars, tuple(signatures), tuple(sorted(upstream.target_compounds)))
    if key not in _CANONICAL_INDEX_CACHE:
        value = _build_canonical_sentence_index(layout.path('text_chunks.jsonl'), min_chars=min_chars, upstream=upstream)
        _CANONICAL_INDEX_CACHE.clear()
        _CANONICAL_INDEX_CACHE[key] = value
    canonical, counts = _CANONICAL_INDEX_CACHE[key]
    return dict(canonical), Counter(counts)


def _build_canonical_sentence_index(
    chunks_path: Path,
    *,
    min_chars: int,
    upstream: UpstreamIndex,
) -> tuple[dict[str, CanonicalSentence], Counter[str]]:
    canonical: dict[str, CanonicalSentence] = {}
    counts: Counter[str] = Counter()
    for chunk in _iter_jsonl(chunks_path):
        decision = classify_chunk(chunk, min_chars=min_chars)
        if not decision.llm_required:
            continue
        target_compound, _ = _resolve_target_compound(chunk, upstream.target_compounds)
        if not target_compound:
            continue
        duplicate_reason = _publication_duplicate_reason(chunk, upstream)
        if duplicate_reason:
            counts[duplicate_reason] += 1
            continue
        prepared = prepare_llm_input(chunk, target_compound)
        if not prepared.eligible:
            counts[prepared.status] += 1
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        candidate = CanonicalSentence(
            chunk_id=chunk_id,
            source_id=str(chunk.get("source_id") or ""),
            source_location=str(chunk.get("source_location") or ""),
        )
        for fingerprint in prepared.sentence_fingerprints:
            counts["candidate_sentences"] += 1
            current = canonical.get(fingerprint)
            if current is None or _canonical_sentence_rank(candidate) < _canonical_sentence_rank(current):
                canonical[fingerprint] = candidate
    counts["canonical_sentences"] = len(canonical)
    counts["duplicate_sentences"] = counts["candidate_sentences"] - len(canonical)
    return canonical, counts


def _canonical_sentence_rank(candidate: CanonicalSentence) -> tuple[int, str, str]:
    return (
        SOURCE_CANONICAL_PRIORITY.get(candidate.source_id, 10),
        candidate.source_location,
        candidate.chunk_id,
    )


def _publication_duplicate_reason(chunk: dict[str, Any], upstream: UpstreamIndex) -> str | None:
    if chunk.get("source_id") != "pubmed":
        return None
    document_id = str(chunk.get("parsed_document_id") or "")
    metadata = upstream.document_metadata.get(document_id, {})
    pmcid = str(metadata.get("pmcid") or "").strip().upper()
    if pmcid and pmcid in upstream.pmc_fulltext_ids:
        return "skipped_pubmed_abstract_with_pmc_fulltext"
    return None


def _canonical_prepared_input(
    chunk: dict[str, Any],
    prepared: PreparedLLMInput,
    canonical_by_fingerprint: dict[str, CanonicalSentence],
) -> tuple[PreparedLLMInput, list[dict[str, str]]]:
    chunk_id = str(chunk.get("chunk_id") or "")
    sentences: list[str] = []
    fingerprints: list[str] = []
    duplicates: list[dict[str, str]] = []
    for sentence, fingerprint in zip(prepared.sentences, prepared.sentence_fingerprints, strict=True):
        canonical = canonical_by_fingerprint.get(fingerprint)
        if canonical is None or canonical.chunk_id == chunk_id:
            sentences.append(sentence)
            fingerprints.append(fingerprint)
            continue
        duplicates.append(
            {
                "sentence_fingerprint": fingerprint,
                "canonical_chunk_id": canonical.chunk_id,
                "canonical_source_id": canonical.source_id,
                "canonical_source_location": canonical.source_location,
            }
        )
    if sentences:
        return (
            PreparedLLMInput(
                True,
                "llm_required",
                prepared.reason,
                tuple(sentences),
                tuple(fingerprints),
            ),
            duplicates,
        )
    return (
        PreparedLLMInput(
            False,
            "skipped_duplicate_llm_sentence",
            "all high-precision candidate sentences are owned by canonical chunks",
        ),
        duplicates,
    )


def _request_id(chunk_id: str, target_compound: str | None) -> str:
    return _stable_id("llm_live_request_v2", chunk_id, (target_compound or "unresolved").lower())


def _response_record(
    task: PendingTask,
    *,
    status: str,
    model_name: str,
    prompt_template_version: str,
    candidates: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
    schema_validation_status: str,
    error: str | None,
    latency_seconds: float | None,
    output_hash: str | None,
    reused_from_request_id: str | None = None,
    failed_raw_output: str | None = None,
    assessment: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "request_id": task.request_record["request_id"],
        "attempt_number": task.request_record.get("attempt_number"),
        "run_id": task.chunk.get("run_id"),
        "source_id": task.chunk.get("source_id"),
        "target_compound": task.target_compound,
        "raw_asset_id": task.chunk.get("raw_asset_id"),
        "parsed_document_id": task.chunk.get("parsed_document_id"),
        "parsed_table_id": task.chunk.get("parsed_table_id"),
        "chunk_id": task.chunk.get("chunk_id"),
        "source_location": task.chunk.get("source_location"),
        "checksum": task.chunk.get("checksum"),
        "model_name": model_name,
        "prompt_template_version": prompt_template_version,
        "input_hash": task.input_hash,
        "output_hash": output_hash,
        "status": status,
        "schema_validation_status": schema_validation_status,
        "validator_version": VALIDATOR_VERSION,
        "generation_format": task.request_record.get("generation_format", "legacy_unconstrained"),
        "generation_recovery_mode": task.request_record.get("generation_recovery_mode"),
        "generation_schema_sha256": task.request_record.get("generation_schema_sha256"),
        "generation_sampling": task.request_record.get("generation_sampling", {"do_sample": False}),
        "effective_prompt_sha256": task.request_record.get("effective_prompt_sha256"),
        "failure_reason": error,
        **({"extraction_assessment": assessment} if assessment is not None else {}),
        **({"failed_raw_output": failed_raw_output[:65536],
            "failed_raw_output_truncated": len(failed_raw_output) > 65536} if failed_raw_output else {}),
        "latency_seconds": latency_seconds,
        "raw_candidate_count": len(candidates) + len(rejected_candidates),
        "candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "reused_from_request_id": reused_from_request_id,
        "completed_at": _utc_now(),
    }


def _validate_candidates(
    task: PendingTask,
    candidates: list[dict[str, Any]],
    upstream: UpstreamIndex,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    provenance_error = _chunk_provenance_error(task.chunk, upstream)
    for index, raw_candidate in enumerate(candidates):
        errors: list[str] = []
        parsed: LLMEvidenceCandidate | None = None
        candidate_for_validation, repairs = _repair_candidate_from_source_text(raw_candidate, task)
        try:
            parsed = LLMEvidenceCandidate.model_validate(candidate_for_validation)
        except ValidationError as exc:
            errors.extend(_pydantic_error_messages(exc))

        if provenance_error:
            errors.append(provenance_error)
        if parsed is not None:
            if parsed.target_compound.casefold() != task.target_compound.casefold():
                errors.append("candidate_target_compound_does_not_match_request")
            chunk_text = str(task.chunk.get("chunk_text") or "")
            if parsed.source_text not in chunk_text:
                errors.append("source_text_not_found_verbatim_in_chunk")
            unit_boundary_error = source_quote_unit_boundary_error(parsed.source_text, chunk_text)
            if unit_boundary_error:
                errors.append(unit_boundary_error)
            request_input_items = task.request_record.get("input_items")
            if isinstance(request_input_items, list) and request_input_items and parsed.source_text not in {
                str(item) for item in request_input_items
            }:
                errors.append("source_text_does_not_equal_complete_input_item")
            if not _contains_exact_name(parsed.source_text, task.target_compound):
                errors.append("source_text_does_not_name_target_compound")
            completeness_error = source_assertion_completeness_error(
                parsed.source_text,
                target_compound=task.target_compound,
                allow_flattened_table=bool(task.chunk.get("parsed_table_id")),
            )
            if completeness_error:
                errors.append("source_text_is_incomplete_assertion")
            joint_subject_error = _joint_target_metabolite_subject_error(
                parsed.source_text,
                task.target_compound,
                parsed.parameter,
            )
            if joint_subject_error:
                errors.append(joint_subject_error)
            if not _parameter_matches_evidence_domain(parsed.evidence_type, parsed.parameter):
                errors.append("parameter_outside_evidence_type_domain")
            evidence_context_error = _evidence_type_context_error(parsed.evidence_type, parsed.source_text)
            if evidence_context_error:
                errors.append(evidence_context_error)
            parameter_source_error = _parameter_source_consistency_error(parsed.parameter, parsed.source_text)
            if parameter_source_error:
                errors.append(parameter_source_error)
            owner_error = _parameter_owner_error(
                parsed.evidence_type, parsed.parameter, parsed.source_text, task.target_compound)
            if owner_error:
                errors.append(owner_error)
            numeric_owner_error = _numeric_parameter_owner_error(parsed, task.target_compound)
            if numeric_owner_error:
                errors.append(numeric_owner_error)
            clearance_context_error = _clearance_context_error(parsed.parameter, parsed.source_text, task.target_compound)
            if clearance_context_error:
                errors.append(clearance_context_error)
            has_scalar = parsed.value is not None
            has_range_value = parsed.value_low is not None or parsed.value_high is not None
            if has_scalar and has_range_value:
                errors.append("candidate_cannot_have_scalar_and_range_values")
            if has_range_value and (parsed.value_low is None or parsed.value_high is None):
                errors.append("range_candidate_requires_value_low_and_value_high")
            range_error = _numeric_range_source_error(parsed)
            if range_error:
                errors.append(range_error)
            numeric_values = [numeric for numeric in (parsed.value, parsed.value_low, parsed.value_high) if numeric is not None]
            direct_percent_values = [
                value for value in _direct_percent_change_values(parsed.source_text)
                if _numeric_assertion_status(parsed.parameter, parsed.source_text, task.target_compound, value, "%") == "supported"
            ]
            if not numeric_values and direct_percent_values:
                errors.append("direct_percent_change_missing_value")
            if numeric_values and len(direct_percent_values) > 1:
                errors.append("multiple_direct_percent_changes_are_ambiguous")
            if numeric_values and _has_ambiguous_numeric_comparison(parsed.source_text):
                errors.append("ambiguous_comparative_value_assignment")
            if not numeric_values:
                if parsed.unit not in {None, ""}:
                    errors.append("qualitative_candidate_must_not_have_unit")
                if parsed.value_qualifier not in {None, ""}:
                    errors.append("qualitative_candidate_must_not_have_value_qualifier")
            else:
                if not parsed.unit:
                    errors.append("numeric_candidate_missing_unit")
                elif normalize_unit(parsed.unit).dimension in {None, "unknown"}:
                    errors.append("unit_not_in_canonical_unit_map")
                else:
                    dimension_error = _parameter_unit_dimension_error(parsed.parameter, parsed.unit)
                    if dimension_error:
                        errors.append(dimension_error)
                    if (
                        normalize_unit(parsed.unit).dimension in {"fraction_percent", "relative_change_percent"}
                        and len(direct_percent_values) == 1
                        and not any(math.isclose(value, direct_percent_values[0], rel_tol=1e-8, abs_tol=1e-10) for value in numeric_values)
                    ):
                        errors.append("percent_change_value_does_not_match_source")
                converted_time_range = _verified_time_range_conversion(parsed)
                if converted_time_range:
                    repairs.append("verified_source_time_range_unit_conversion")
                for numeric in numeric_values:
                    if not converted_time_range and not _number_appears_in_source(numeric, parsed.source_text):
                        errors.append("numeric_value_not_found_in_source_text")
                bounds = [
                    item for item in _bounded_percent_values(parsed.source_text)
                    if _numeric_assertion_status(parsed.parameter, parsed.source_text, task.target_compound, item[0], "%") == "supported"
                ]
                if (len(bounds) == 1 and parsed.value is not None and parsed.unit
                        and normalize_unit(parsed.unit).dimension in {"fraction_percent", "relative_change_percent"}):
                    bound_value, bound_qualifier = bounds[0]
                    if math.isclose(parsed.value, bound_value) and parsed.value_qualifier != bound_qualifier:
                        errors.append("value_qualifier_does_not_match_source_bound")
                if parsed.unit and not _unit_appears_in_source(parsed.unit, parsed.source_text):
                    errors.append("unit_not_copied_from_source_text")
                if parsed.value_low is not None and parsed.value_high is not None and parsed.value_low > parsed.value_high:
                    errors.append("range_value_low_exceeds_value_high")

        if errors or parsed is None:
            rejected.append(
                {
                    "candidate_index": index,
                    "errors": sorted(set(errors)),
                    "candidate": raw_candidate,
                    "attempted_validation_repairs": repairs,
                }
            )
            continue
        validated = parsed.model_dump(mode="json")
        if repairs:
            validated["validation_repairs"] = repairs
        valid.append(validated)
    return valid, rejected


def _pydantic_error_messages(exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ())) or "candidate"
        messages.append(f"schema:{location}:{error.get('type', 'invalid')}")
    return messages


def _schema_validation_status(valid: list[dict[str, Any]], rejected: list[dict[str, Any]], raw_count: int) -> str:
    if rejected and valid:
        return "partial_rejected"
    if rejected:
        return "all_rejected"
    if raw_count == 0:
        return "valid_empty"
    return "valid"


def _repair_candidate_from_source_text(
    candidate: dict[str, Any], task: PendingTask
) -> tuple[dict[str, Any], list[str]]:
    repaired = dict(candidate)
    repairs: list[str] = []
    source_text = str(repaired.get("source_text") or "")
    input_items = task.request_record.get("input_items")
    if not isinstance(input_items, list):
        input_items = []
    normalized_input_items = [str(item) for item in input_items]
    if source_text and normalized_input_items and source_text not in normalized_input_items:
        # Restore the immutable original only on a unique soft-hyphen match.
        # Changed words, numbers and signs remain rejected.
        same_items = [item for item in normalized_input_items
                      if item.replace("\u00ad", "") == source_text.replace("\u00ad", "")]
        if len(same_items) == 1:
            repaired["source_text"] = same_items[0]
            source_text = same_items[0]
            repairs.append("restored_verbatim_source_soft_hyphen")
        containing_items = [
            str(item)
            for item in normalized_input_items
            if source_text in str(item) and _contains_exact_name(str(item), task.target_compound)
        ]
        if source_text not in normalized_input_items and len(containing_items) == 1:
            repaired["source_text"] = containing_items[0]
            source_text = containing_items[0]
            repairs.append("expanded_source_text_to_complete_input_item")

    qualifier = str(repaired.get("value_qualifier") or "").strip()
    numeric_annotation = {"to", "from", "range", "ranged from", "between", "less than", "greater than",
                          "more than", "at most", "at least", "up to", "under", "over", "above", "below",
                          "equal to", "approximately", "about", "fold"}
    plain_description = bool(re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+){0,7}", qualifier))
    descriptive_position = (repaired.get("value") is None or qualifier.casefold() in {"mean", "median", "average"})
    if (plain_description and descriptive_position and qualifier.casefold() not in numeric_annotation
            and normalize_unit(qualifier).dimension in {None, "unknown"}
            and qualifier.casefold() in source_text.replace("\u00ad", "").casefold()):
        condition = str(repaired.get("condition") or "").strip()
        if qualifier.casefold() not in condition.casefold():
            repaired["condition"] = "; ".join(item for item in (condition, qualifier) if item)
        repaired["value_qualifier"] = None
        repairs.append("source_supported_description_moved_to_condition")
    has_value = any(repaired.get(field) is not None for field in ("value", "value_low", "value_high"))
    if (repaired.get("evidence_type") == "ddi" and not has_value
            and "has the potential to interact" in source_text.casefold()):
        condition = str(repaired.get("condition") or "").strip()
        if "potential" not in condition.casefold():
            repaired["condition"] = "Potential interaction; " + condition
            repairs.append("preserved_source_potential_interaction")
    def supported(number: float, unit: str) -> bool:
        return _numeric_assertion_status(
            str(repaired.get("parameter") or ""), source_text, task.target_compound, number, unit,
        ) == "supported"

    # Never manufacture a missing measurement from another number in a quote.
    # Only repair annotations of a numeric value the model actually supplied.
    bounded_values = [item for item in _bounded_percent_values(source_text) if supported(item[0], "%")]
    fold_values = [value for value in _direct_fold_change_values(source_text) if supported(value, "fold")]
    if (has_value and repaired.get("unit")
            and normalize_unit(str(repaired["unit"])).dimension in {"fraction_percent", "relative_change_percent"}
            and len(bounded_values) == 1 and not repaired.get("value_qualifier")):
        value, qualifier = bounded_values[0]
        numeric_values = [repaired.get(field) for field in ("value", "value_low", "value_high")]
        if any(isinstance(item, (int, float)) and math.isclose(float(item), value) for item in numeric_values):
            repaired["value_qualifier"] = qualifier
            repairs.append("bounded_percent_qualifier_from_source_text")
    if has_value and len(fold_values) == 1 and not repaired.get("unit"):
        fold_value = fold_values[0]
        current_values = [repaired.get(field) for field in ("value", "value_low", "value_high")]
        if any(isinstance(item, (int, float)) and math.isclose(float(item), fold_value) for item in current_values):
            repaired["unit"] = "fold"
            repairs.append("direct_fold_unit_from_source_text")
    return repaired, repairs


def _chunk_provenance_error(chunk: dict[str, Any], upstream: UpstreamIndex) -> str | None:
    raw_asset_id = str(chunk.get("raw_asset_id") or "")
    asset = upstream.raw_assets.get(raw_asset_id)
    if asset is None:
        return "raw_asset_id_not_found_in_inventory"
    if str(asset.get("checksum") or "") != str(chunk.get("checksum") or ""):
        return "checksum_does_not_match_raw_inventory"
    if str(asset.get("storage_path") or "") != str(chunk.get("storage_path") or ""):
        return "storage_path_does_not_match_raw_inventory"
    document_id = str(chunk.get("parsed_document_id") or "")
    table_id = str(chunk.get("parsed_table_id") or "")
    if document_id and document_id not in upstream.parsed_document_ids:
        return "parsed_document_id_not_found"
    if table_id and table_id not in upstream.parsed_table_ids:
        return "parsed_table_id_not_found"
    if not document_id and not table_id:
        return "chunk_has_no_parsed_parent"
    required = ["run_id", "source_id", "raw_asset_id", "storage_path", "checksum", "source_location", "chunk_id"]
    if any(not chunk.get(field) for field in required):
        return "chunk_missing_required_provenance"
    return None


def _resolve_target_compound(chunk: dict[str, Any], target_compounds: set[str]) -> tuple[str | None, str]:
    assigned = str(chunk.get("compound") or "").strip()
    if assigned and assigned.casefold() != "shared":
        exact = [target for target in target_compounds if target.casefold() == assigned.casefold()]
        if len(exact) == 1:
            return exact[0], "chunk_compound_exact_target"
    text = str(chunk.get("chunk_text") or "")
    matches = [target for target in target_compounds if _contains_exact_name(text, target)]
    if len(matches) == 1:
        return matches[0], "unique_exact_target_in_chunk_text"
    if len(matches) > 1:
        return None, "multiple_target_compounds_in_chunk_text"
    return None, "target_compound_unresolved"


def _contains_exact_name(text: str, compound: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9-]){re.escape(compound)}(?![A-Za-z0-9-])", text, flags=re.IGNORECASE) is not None


def _parameter_matches_evidence_domain(evidence_type: str, parameter: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", parameter.casefold()).strip()
    return any(pattern.search(normalized) for pattern in PARAMETER_DOMAIN_PATTERNS.get(evidence_type, ()))



def _numeric_assertion_status(parameter: str, source_text: str, target: str, value: float, unit: str | None) -> str:
    """Check a source number in its local parameter/subject assertion.

    Contrast clauses keep their own subjects; a target mentioned earlier in the
    quote does not own every later number. Population clauses can retain the
    same drug subject. This is deliberately independent of drug/class names.
    """
    parameter_pattern = _parameter_sentence_pattern(parameter)
    percent = bool(unit and normalize_unit(unit).dimension in {"fraction_percent", "relative_change_percent"})
    population = re.compile(r"\b(?:patients?|subjects?|volunteers?|adults?|children|infants?|neonates?|"
                            r"controls?|population|healthy|renal impairment|hepatic impairment)\b", re.I)
    contrast = re.compile(r"(?:,\s*)?\b(?:in\s+contrast\s+to|unlike|whereas|while|but|"
                          r"compared\s+(?:with|to)|rather\s+than)\b", re.I)
    parameter_start = (r"(?:protein\s+binding|binding|bioavailability|absorption|excretion|"
                       r"clearance|half[- ]life|volume\s+of\s+distribution|cmax|tmax|auc)\b")
    clause_break = re.compile(
        rf",\s*(?:and\s+)?(?={re.escape(target)}\b|{parameter_start})|"
        rf"\band\s+(?={parameter_start})", re.I,
    )
    ignored_subjects = {
        "it", "which", "that", "this", "these", "they", "there", "drug", "compound",
        "binding", "protein", "bioavailability", "absorption", "excretion", "clearance",
        "half-life", "concentration", "concentrations", "distribution", "volume", "fraction",
        "amount", "value", "values", "mean", "median", "levels", "parameters", "ratio",
        "cmax", "tmax", "auc", "patients", "subjects", "volunteers", "adults", "children",
        "infants", "neonates", "controls",
    }
    statuses: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|;\s*", source_text):
        regions = contrast.split(sentence)
        for region_index, region in enumerate(regions):
            parts: list[str] = []
            part_start = 0
            for separator in clause_break.finditer(region):
                # Coordinated parameters share the following predicate/value:
                # "Cmax and AUC differed <5%" supports both parameters.
                if re.search(parameter_start + r"\s*$", region[:separator.start()], re.I):
                    continue
                parts.append(region[part_start:separator.start()])
                part_start = separator.end()
            parts.append(region[part_start:])
            for part in parts:
                part_has_target = _contains_exact_name(part, target)
                contrast_other = (
                    region_index > 0 and not part_has_target
                    and not population.search(part)
                )
                for match in NUMBER_RE.finditer(part):
                    try:
                        source_value = float(match.group().replace(",", ""))
                        if match.group().startswith("-") and re.search(r"\d\s*$", part[:match.start()]):
                            source_value = abs(source_value)
                    except ValueError:
                        continue
                    if not math.isclose(source_value, value, rel_tol=1e-8, abs_tol=1e-10):
                        continue
                    local_tail = re.sub(
                        r"^\s*(?:to|[-–])\s*\d+(?:\.\d+)?", "", part[match.end():], count=1, flags=re.I,
                    )
                    if _local_numeric_unit_conflict(unit, local_tail):
                        statuses.append("unit_conflict")
                        continue
                    if re.search(r"metaboli", parameter, re.I) and re.match(
                        rf"\s*%\s+(?:(?:is|was|remains?)\s+)?"
                        rf"(?:still\s+{re.escape(target)}\b|(?:unchanged|unmetabolized)\b)",
                        part[match.end():], re.I,
                    ):
                        statuses.append("other_parameter")
                        continue
                    if percent and not re.match(
                        r"\s*(?:%|(?:to|[-–])\s*\d+(?:\.\d+)?\s*%)", part[match.end():], re.I
                    ):
                        continue
                    owners = list(re.finditer(
                        r"(?:^|[,;]\s*)\s*(?:the\s+)?"
                        r"(?P<owner>[A-Za-z][A-Za-z0-9_-]*)\s+(?:is|are|was|were)\b",
                        part[:match.start()], re.I,
                    ))
                    owner = owners[-1]["owner"].casefold() if owners else None
                    explicit_other = bool(
                        owner and owner not in ignored_subjects
                        and owner not in target.casefold().split()
                    )
                    percent_owner = re.match(
                        r"\s*%\s+of\s+(?:the\s+)?"
                        # Dose qualifiers describe the administered amount; the
                        # following named substance still owns its percentage.
                        r"(?:(?:(?:orally|intravenously|intramuscularly|subcutaneously)\s+)?"
                        r"(?:administered\s+)?dose\s+of\s+(?:the\s+)?)?"
                        r"(?P<owner>[A-Za-z][A-Za-z0-9_-]*)\b",
                        part[match.end():], re.I,
                    )
                    if percent_owner:
                        named_owner = percent_owner["owner"].casefold()
                        explicit_other = explicit_other or (
                            named_owner not in ignored_subjects and named_owner not in target.casefold().split()
                        )
                    if contrast_other or explicit_other:
                        statuses.append("other_owner")
                    elif not parameter_pattern.search(part):
                        statuses.append("other_parameter")
                    elif part_has_target or _contains_exact_name(source_text, target):
                        statuses.append("supported")
                    else:
                        statuses.append("unresolved")
    if "supported" in statuses:
        return "supported"
    if "unit_conflict" in statuses:
        return "unit_conflict"
    if "other_owner" in statuses:
        return "other_owner"
    if "other_parameter" in statuses:
        return "other_parameter"
    return "unresolved"



def _local_numeric_unit_conflict(unit: str | None, source_tail: str) -> bool:
    """Reject explicit local percent/time unit contradictions without conversion."""
    if not unit:
        return False
    match = re.match(r"\s*(?P<unit>%|minutes?\b|min\b|hours?\b|hrs?\b|h\b|days?\b|d\b)", source_tail, re.I)
    if not match:
        return False
    expected = normalize_unit(unit).dimension
    actual = normalize_unit(match["unit"]).dimension
    if expected in {None, "unknown"} or actual in {None, "unknown"}:
        return False
    percent_dimensions = {"fraction_percent", "relative_change_percent"}
    return expected != actual and not {expected, actual} <= percent_dimensions

def _numeric_parameter_owner_error(candidate: Any, target_compound: str) -> str | None:
    """Reusable extraction/final-quality/watchdog guard; accepts a dict or object."""
    get = candidate.get if isinstance(candidate, dict) else lambda key: getattr(candidate, key, None)
    parameter = str(get("parameter") or "")
    source_text = str(get("source_text") or "")
    # A source chunk can end before CI brackets and a 'respectively' list.
    # For example AUC, Cmax and Cmin ... by 22% continues in the original
    # source with 9% and 33%. A terminal shared number is not a mapping.
    if get("unit") and normalize_unit(str(get("unit"))).dimension in {"fraction_percent", "relative_change_percent"}:
        last_clause = re.split(r"(?<=[.!?])\s+|;\s*", source_text)[-1]
        endpoints = {re.sub(r"\s+", "", item.casefold()) for item in re.findall(
            r"\b(?:AUC|C\s*max|C\s*min|T\s*max|clearance|half[- ]life)\b", last_clause, re.I)}
        if (len(endpoints) > 1 and re.search(r"\d\s*%\s*$", last_clause)
                # A terminal *shared* percentage can be a truncated list.
                # Separately assigned AUC 20%, Cmax 16%, C12h 26% is complete.
                and len(re.findall(r"(?<![\w.])\d+(?:\.\d+)?\s*%", last_clause)) == 1
                and _direct_percent_change_values(last_clause)
                and (_parameter_sentence_pattern(parameter).search(last_clause)
                     or re.sub(r"\s+", "", parameter.casefold()) in endpoints)
                and not re.search(r"\b(?:both|each|same percentage|all three)\b", last_clause, re.I)
                and any(isinstance(get(key), (int, float)) and not isinstance(get(key), bool)
                        for key in ("value", "value_low", "value_high"))):
            return "truncated_multi_endpoint_percent_assignment"
    binding = bool(re.search(r"protein[ _-]*binding|fraction[ _-]*unbound", parameter, re.I))
    for value in (get("value"), get("value_low"), get("value_high")):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        status = _numeric_assertion_status(parameter, source_text, target_compound, value, get("unit"))
        if status == "unit_conflict":
            return "numeric_unit_conflicts_with_source_assertion"
        if status == "other_owner":
            return "numeric_value_belongs_to_other_substance"
        if status == "other_parameter" or (binding and status != "supported"):
            return "numeric_value_not_supported_for_target_parameter"
    return None

def _parameter_owner_error(evidence_type: str, parameter: str, source_text: str, target: str) -> str | None:
    # A coadministered drug's PK may support a DDI relation, but must not become
    # the target's own clinical PK or ADME parameter.
    if evidence_type != "ddi" and _parameter_is_explicitly_owned_by_other(source_text, target):
        return "parameter_explicitly_owned_by_other_substance"
    if evidence_type != "ddi":
        effect = re.search(
            rf"\b{re.escape(target)}\b\s+(?:significantly\s+)?"
            r"(?:increases?|increased|decreases?|decreased|reduces?|reduced|enhances?|enhanced|inhibits?|inhibited)\s+"
            r"(?:(?:the|urinary|renal|hepatic|plasma|blood|serum)\s+)*"
            r"(?P<owner>[A-Za-z][A-Za-z0-9_-]+)\s+"
            r"(?:excretion|clearance|absorption|metabolism|concentration|elimination)\b",
            source_text, re.I,
        )
        if effect and effect['owner'].casefold() not in {target.casefold(), "own", "its", "drug", "total"}:
            return "target_effect_on_other_substance_is_not_own_parameter"
    normalized = re.sub(r"[^a-z0-9]+", " ", parameter.casefold())
    if not re.search(r"excret|absor|distribut|clearance|half life|cmax|tmax|auc|binding|bioavailability", normalized):
        return None
    parameter_pattern = _parameter_sentence_pattern(parameter).pattern
    target_pattern = rf"(?<![A-Za-z0-9-]){re.escape(target)}(?![A-Za-z0-9-])"
    for sentence in re.split(r"(?<=[.!?])\s+", source_text):
        if not re.search(parameter_pattern, sentence, re.I):
            continue
        if not re.search(r"\bmetabolites?\b|\bend products?\b", sentence, re.I):
            continue
        # Positive parent ownership within the same assertion is required.
        forward = re.compile(rf"{target_pattern}[^.;\n]{{0,45}}(?:{parameter_pattern})", re.I)
        parent_subject = any(
            not re.search(r"(?:metabolites?|products?)\s+of\s+$", sentence[max(0,m.start()-40):m.start()], re.I)
            for m in forward.finditer(sentence)
        )
        reverse = re.search(
            rf"(?:{parameter_pattern})\w*\s+(?:of|for)\s+(?:unchanged\s+)?{target_pattern}",
            sentence, re.I,
        )
        if not parent_subject and not reverse:
            return "metabolite_parameter_is_not_parent_drug_parameter"
    return None


def _parameter_source_consistency_error(parameter: str, source_text: str) -> str | None:
    normalized_parameter = re.sub(r"[^a-z0-9]+", " ", parameter.casefold()).strip()
    checks: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
        (re.compile(r"^excretion delay$"), re.compile(EXCRETION_DELAY_SOURCE, re.IGNORECASE)),
        (re.compile(r"\bcmax\b|maximum concentration|peak concentration"), re.compile(r"\bc\s*max\b|maximum (?:plasma )?concentration|peak (?:plasma )?concentration", re.IGNORECASE)),
        (re.compile(r"steady state concentration"), re.compile(r"steady[- ]state concentration", re.IGNORECASE)),
        (re.compile(r"\bauc\b|area under"), re.compile(r"\bauc(?:s|0[-–][a-z0-9]+)?\b|area under (?:the )?curve", re.IGNORECASE)),
        (re.compile(r"\btmax\b|time to (?:peak|maximum)"), re.compile(r"\bt\s*max\b|time to (?:reach )?(?:peak|maximum)", re.IGNORECASE)),
        (re.compile(r"half life"), re.compile(r"half[- ]life", re.IGNORECASE)),
        (re.compile(r"clearance"), re.compile(r"clearance", re.IGNORECASE)),
        (re.compile(r"volume of distribution|\bvd\b|\bvss\b"), re.compile(r"volume of distribution|\bvd\b|\bvss\b", re.IGNORECASE)),
        (re.compile(r"bioavailability"), re.compile(r"bioavailability", re.IGNORECASE)),
        (re.compile(r"accumulation"), re.compile(r"accumulation", re.IGNORECASE)),
        (re.compile(r"trough"), re.compile(r"trough", re.IGNORECASE)),
        (re.compile(r"renal clearance"), re.compile(r"renal clearance", re.IGNORECASE)),
        (re.compile(r"plasma concentration|blood concentration|serum concentration"), re.compile(r"(?:plasma|blood|serum) concentration", re.IGNORECASE)),
        (re.compile(r"mean residence time|\bmrt\b"), re.compile(r"mean residence time|\bmrt\b", re.IGNORECASE)),
        (re.compile(r"absorption"), re.compile(r"absorption|absorbed", re.IGNORECASE)),
        (re.compile(r"distribution"), re.compile(r"distribution|distributed", re.IGNORECASE)),
        (re.compile(r"metabolism|metabolic"), re.compile(r"metaboli[sz]", re.IGNORECASE)),
        (re.compile(r"excretion"), re.compile(r"excret", re.IGNORECASE)),
        (re.compile(r"elimination"), re.compile(r"eliminat", re.IGNORECASE)),
        (re.compile(r"fraction unbound|protein binding"), re.compile(r"fraction unbound|protein[- ]bound|protein binding|bound\s+to\s+(?:plasma\s+|serum\s+)?proteins?", re.IGNORECASE)),
    )
    for parameter_pattern, source_pattern in checks:
        if parameter_pattern.search(normalized_parameter) and not source_pattern.search(source_text):
            return "parameter_not_supported_by_source_text_terminology"
    return None


def _has_ambiguous_numeric_comparison(source_text: str) -> bool:
    if NUMERIC_VS_RE.search(source_text):
        return True
    number_count = len(NUMBER_RE.findall(source_text))
    if RESPECTIVELY_RE.search(source_text) and number_count >= 2:
        return True
    return COMPARATIVE_TERM_RE.search(source_text) is not None and number_count >= 2


def _direct_percent_change_values(source_text: str) -> list[float]:
    matches = [*DIRECT_PERCENT_CHANGE_RE.finditer(source_text), *PREFIX_PERCENT_CHANGE_RE.finditer(source_text)]
    values: list[float] = []
    for match in sorted(matches, key=lambda item: item.start()):
        value = float(match.group("value"))
        if not any(math.isclose(value, existing, rel_tol=1e-8, abs_tol=1e-10) for existing in values):
            values.append(value)
    return values


def _bounded_percent_values(source_text: str) -> list[tuple[float, str]]:
    qualifier_map = {
        "less than": "<",
        "under": "<",
        "below": "<",
        "<": "<",
        "≤": "<=",
        "up to": "<=",
        "at most": "<=",
        "greater than": ">",
        "more than": ">",
        "over": ">",
        "above": ">",
        ">": ">",
        "≥": ">=",
        "at least": ">=",
    }
    values: list[tuple[float, str]] = []
    for match in BOUNDED_PERCENT_RE.finditer(source_text):
        item = (float(match.group("value")), qualifier_map[match.group("qualifier").casefold()])
        if item not in values:
            values.append(item)
    return values


def _direct_fold_change_values(source_text: str) -> list[float]:
    values: list[float] = []
    for match in DIRECT_FOLD_CHANGE_RE.finditer(source_text):
        value = float(match.group("value"))
        if not any(math.isclose(value, existing, rel_tol=1e-8, abs_tol=1e-10) for existing in values):
            values.append(value)
    return values


def _joint_target_metabolite_subject_error(
    source_text: str,
    target_compound: str,
    parameter: str,
) -> str | None:
    target = re.escape(target_compound)
    joint_pattern = re.compile(
        rf"(?<![A-Za-z0-9-]){target}(?![A-Za-z0-9-]).{{0,50}}\b(?:and|or)\b.{{0,60}}"
        rf"(?:\bmetabolites?\b|\b(?:hydroxy|dihydroxy|desmethyl|glucuronide)[-\s]*{target}\b|\b\d+(?:-[A-Za-z0-9]+){{2,}})",
        flags=re.IGNORECASE,
    )
    parameter_pattern = _parameter_sentence_pattern(parameter)
    sentences = re.split(r"(?<=[.!?])\s+", source_text)
    relevant_sentences = [sentence for sentence in sentences if parameter_pattern.search(sentence)]
    scope = relevant_sentences or sentences
    return "joint_parent_and_metabolite_subject" if any(joint_pattern.search(sentence) for sentence in scope) else None


def _parameter_sentence_pattern(parameter: str) -> re.Pattern[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", parameter.casefold()).strip()
    checks: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^excretion delay$"), EXCRETION_DELAY_SOURCE),
        (re.compile(r"\bcmax\b|maximum concentration|peak concentration"), r"\bc\s*max\b|maximum (?:plasma )?concentration|peak (?:plasma )?concentration"),
        (re.compile(r"\bauc\b|area under"), r"\bauc(?:s|0[-–][a-z0-9]+)?\b|area under (?:the )?curve"),
        (re.compile(r"\btmax\b|time to (?:peak|maximum)"), r"\bt\s*max\b|time to (?:reach )?(?:peak|maximum)"),
        (re.compile(r"half life"), r"half[- ]life"),
        (re.compile(r"clearance"), r"clearance"),
        (re.compile(r"volume of distribution|\bvd\b|\bvss\b"), r"volume of distribution|\bvd\b|\bvss\b"),
        (re.compile(r"bioavailability"), r"bioavailability"),
        (re.compile(r"fraction unbound|protein binding"), r"fraction unbound|protein[- ]bound|protein binding|bound\s+to\s+(?:plasma\s+|serum\s+)?proteins?"),
        (re.compile(r"absorption"), r"absorption|absorbed"),
        (re.compile(r"distribution"), r"distribution|distributed"),
        (re.compile(r"metabolism|metabolic"), r"metaboli[sz]"),
        (re.compile(r"excretion"), r"excret"),
        (re.compile(r"elimination"), r"eliminat"),
    )
    for parameter_pattern, source_pattern in checks:
        if parameter_pattern.search(normalized):
            return re.compile(source_pattern, flags=re.IGNORECASE)
    terms = [term for term in normalized.split() if len(term) >= 4]
    return re.compile("|".join(re.escape(term) for term in terms) or r"(?s:.*)", flags=re.IGNORECASE)


def _clearance_context_error(parameter: str, source_text: str, target_compound: str) -> str | None:
    normalized_parameter = re.sub(r"[^a-z0-9]+", " ", parameter.casefold()).strip()
    if "clearance" not in normalized_parameter:
        return None
    if DIALYSIS_CLEARANCE_RE.search(source_text):
        return "dialysis_clearance_is_not_systemic_target_drug_clearance"
    if not RENAL_FUNCTION_MARKER_RE.search(source_text):
        return None
    target_clearance = re.search(
        rf"(?:"
        rf"{re.escape(target_compound)}\s+(?:(?:plasma|systemic|total|apparent|oral|renal)\s+)?clearance"
        rf"|(?:(?:plasma|systemic|total|apparent|oral|renal)\s+)?clearance\s+of\s+{re.escape(target_compound)}"
        rf")",
        source_text,
        flags=re.IGNORECASE,
    )
    return None if target_clearance else "renal_function_clearance_is_not_target_drug_clearance"


def _evidence_type_context_error(evidence_type: str, source_text: str) -> str | None:
    if evidence_type == "tissue_expression":
        return "tissue_expression_requires_structured_expression_source"
    if evidence_type == "public_pbpk_model" and not PBPK_MODEL_MARKER_RE.search(source_text):
        return "public_pbpk_model_lacks_explicit_pbpk_marker"
    if evidence_type == "special_population" and not SPECIAL_POPULATION_MARKER_RE.search(source_text):
        return "special_population_marker_missing_from_source_text"
    return None


def _parameter_unit_dimension_error(parameter: str, unit: str) -> str | None:
    normalized_parameter = re.sub(r"[^a-z0-9]+", " ", parameter.casefold()).strip()
    dimension = normalize_unit(unit).dimension
    relative = {"fraction_percent", "relative_change_percent", "dimensionless_ratio"}
    rules: tuple[tuple[re.Pattern[str], set[str]], ...] = (
        (re.compile(r"\bcmax\b|concentration|trough"), {"concentration", *relative}),
        (re.compile(r"\bauc\b|area under"), {"auc", *relative}),
        (re.compile(r"\btmax\b|half life|mean residence time|\bmrt\b|^excretion delay$"), {"time", *relative}),
        (re.compile(r"clearance"), {"clearance", "clearance_per_weight", "clearance_bsa_normalized", *relative}),
        (re.compile(r"volume of distribution|\bvd\b|\bvss\b"), {"volume", "volume_per_weight", *relative}),
        (re.compile(r"bioavailability|accumulation|ratio|fraction unbound|protein binding"), relative),
        (re.compile(r"elimination"), {"time", "clearance", "clearance_per_weight", *relative}),
        (re.compile(r"expression|abundance|protein level|rna level"), {"expression"}),
        (re.compile(r"molecular weight|molecular mass"), {"molar_mass"}),
    )
    for parameter_pattern, allowed_dimensions in rules:
        if parameter_pattern.search(normalized_parameter):
            return None if dimension in allowed_dimensions else "unit_dimension_does_not_match_parameter"
    return None


def _numeric_range_source_error(candidate: LLMEvidenceCandidate) -> str | None:
    """An interval must be an interval in the quote, never two group means."""
    if candidate.value_low is None or candidate.value_high is None:
        return None
    if not all(type(value) in {int, float} and math.isfinite(value)
               for value in (candidate.value_low, candidate.value_high)):
        return "range_endpoints_must_be_finite_numbers"
    if not isinstance(candidate.unit, str) or not isinstance(candidate.parameter, str):
        return "range_requires_parameter_and_unit"
    if _verified_time_range_conversion(candidate):
        return None
    number = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    unit = re.escape(candidate.unit or "").replace(r"\ ", r"\s*")
    pair = re.compile(
        rf"(?<![\w.])(?P<low>{number})\s*(?:{unit})?\s*"
        rf"(?P<join>[-–—]|to|and)\s*(?P<high>{number})(?![\w.])", re.I,
    )
    for match in pair.finditer(candidate.source_text):
        if not (math.isclose(float(match['low']), candidate.value_low)
                and math.isclose(float(match['high']), candidate.value_high)):
            continue
        prefix = candidate.source_text[max(0, match.start()-120):match.start()]
        if match['join'].casefold() == "and" and not re.search(r"\bbetween\s*$", prefix, re.I):
            continue
        if re.search(r"\b(?:increas\w*|decreas\w*|chang\w*|reduc\w*|prolong\w*)\s+from\b", prefix, re.I):
            continue
        if _local_numeric_unit_conflict(candidate.unit, candidate.source_text[match.end():]):
            return "range_unit_conflicts_with_source_interval"
        return None
    return "range_endpoints_not_supported_as_one_source_interval"


def _verified_time_range_conversion(candidate: LLMEvidenceCandidate) -> bool:
    """Accept only both endpoints of one explicit, parameter-bound time range."""
    if candidate.value is not None or candidate.value_low is None or candidate.value_high is None:
        return False
    parameter = re.sub(r"[^a-z0-9]+", " ", candidate.parameter.casefold()).strip()
    if not re.search(r"\btmax\b|half life|time to (?:peak|maximum)", parameter):
        return False
    if normalize_unit(candidate.unit).dimension != "time":
        return False
    matches = [match for match in TIME_RANGE_SOURCE.finditer(candidate.source_text)
               if _parameter_sentence_pattern(candidate.parameter).search(match.group(0))]
    if len(matches) != 1:
        return False
    match = matches[0]
    if match['join'].casefold() == "and" and not re.search(
            r"\bbetween\s*$", candidate.source_text[match.start():match.start('low')], re.I):
        return False
    low, _ = to_canonical_value(float(match['low']), match['low_unit'])
    high, _ = to_canonical_value(float(match['high']), match['high_unit'])
    actual_low, _ = to_canonical_value(candidate.value_low, candidate.unit)
    actual_high, _ = to_canonical_value(candidate.value_high, candidate.unit)
    return low <= high and all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)
                               for a, b in ((low, actual_low), (high, actual_high)))


def _number_appears_in_source(value: float, source_text: str) -> bool:
    for match in NUMBER_RE.finditer(source_text):
        try:
            source_value = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        if math.isclose(source_value, value, rel_tol=1e-8, abs_tol=1e-10):
            return True
    return False


def _unit_appears_in_source(unit: str, source_text: str) -> bool:
    compact_unit = re.sub(r"\s+", "", unit).casefold().replace("μ", "µ")
    compact_source = re.sub(r"\s+", "", source_text).casefold().replace("μ", "µ")
    return compact_unit in compact_source


def _append_evidence_rows(
    chunk: dict[str, Any],
    target_compound: str,
    candidates: list[dict[str, Any]],
    *,
    llm_evidence_path: Path,
    merged_evidence_path: Path,
    existing_evidence_ids: set[str],
    counts: Counter[str],
) -> int:
    evidence_added = 0
    for candidate in candidates:
        source_text = str(candidate["source_text"])
        value = candidate.get("value")
        value_low = candidate.get("value_low")
        value_high = candidate.get("value_high")
        unit = candidate.get("unit")
        canonical_value, canonical_unit = to_canonical_value(value, unit)
        canonical_value_low, _ = to_canonical_value(value_low, unit)
        canonical_value_high, _ = to_canonical_value(value_high, unit)
        normalized_unit = normalize_unit(unit)
        evidence_id = _stable_id(
            "llm_evidence_v7",
            str(chunk.get("raw_asset_id") or ""),
            target_compound.casefold(),
            str(candidate["evidence_type"]),
            str(candidate["parameter"]).casefold(),
            source_text,
            str(value),
            str(value_low),
            str(value_high),
            str(candidate.get("value_qualifier")),
            str(unit),
        )
        if evidence_id in existing_evidence_ids:
            counts["duplicate_evidence"] += 1
            continue
        row = {
            "run_id": chunk.get("run_id"),
            "evidence_id": evidence_id,
            "evidence_type": candidate["evidence_type"],
            "compound": target_compound,
            "compound_assignment_status": "llm_target_specific_assertion_validated",
            "source_id": chunk.get("source_id"),
            "raw_asset_id": chunk.get("raw_asset_id"),
            "parsed_document_id": chunk.get("parsed_document_id"),
            "parsed_table_id": chunk.get("parsed_table_id"),
            "chunk_id": chunk.get("chunk_id"),
            "storage_path": chunk.get("storage_path"),
            "source_url": chunk.get("source_url"),
            "checksum": chunk.get("checksum"),
            "source_location": chunk.get("source_location"),
            "source_text": source_text,
            "source_text_hash": _hash_text(source_text),
            "extraction_method": "llm_assisted",
            "extractor_name": "pbpk_llm_evidence_agent",
            "extractor_version": LLMExtractionAgent(mode="off").prompt_template_version,
            "extraction_confidence": candidate["confidence"],
            "curation_status": "needs_review",
            "parameter": candidate["parameter"],
            "value": value,
            "value_low": value_low,
            "value_high": value_high,
            "value_qualifier": candidate.get("value_qualifier"),
            "value_text": (
                f"{candidate.get('value_qualifier') or ''}{value}"
                if value is not None
                else f"{value_low}-{value_high}"
                if value_low is not None and value_high is not None
                else None
            ),
            "unit": unit,
            "canonical_value": canonical_value,
            "canonical_value_low": canonical_value_low,
            "canonical_value_high": canonical_value_high,
            "canonical_unit": canonical_unit or normalized_unit.canonical_unit,
            "unit_dimension": normalized_unit.dimension,
            "condition": candidate.get("condition"),
            "population": None,
            "route": None,
            "dose": None,
            "metadata_json": {
                "llm_candidate": candidate,
                "target_compound": target_compound,
                "schema_validation_status": "valid",
                "validator_version": VALIDATOR_VERSION,
            },
        }
        _append_jsonl(llm_evidence_path, row)
        _append_jsonl(merged_evidence_path, row)
        existing_evidence_ids.add(evidence_id)
        evidence_added += 1
    return evidence_added


def _api_failure(task: PendingTask, result: LLMCallResult) -> dict[str, Any]:
    reason = result.error or result.status
    return {
        "failure_id": _stable_id("llm_api_failure", str(task.request_record["request_id"]), str(task.request_record.get("attempt_number")), reason),
        "stage": "llm_extract",
        "failure_type": "llm_api_failure",
        "status": result.status,
        "resolved": False,
        "reason": reason,
        "request_id": task.request_record["request_id"],
        "run_id": task.chunk.get("run_id"),
        "source_id": task.chunk.get("source_id"),
        "target_compound": task.target_compound,
        "raw_asset_id": task.chunk.get("raw_asset_id"),
        "chunk_id": task.chunk.get("chunk_id"),
        "source_location": task.chunk.get("source_location"),
        "checksum": task.chunk.get("checksum"),
        "created_at": _utc_now(),
    }


def _candidate_failure(task: PendingTask, rejection: dict[str, Any]) -> dict[str, Any]:
    candidate_index = int(rejection.get("candidate_index") or 0)
    reasons = [str(reason) for reason in rejection.get("errors") or []]
    return {
        "failure_id": _stable_id("llm_candidate_rejection", str(task.request_record["request_id"]), str(candidate_index), "|".join(reasons)),
        "stage": "llm_extract",
        "failure_type": "llm_candidate_validation_rejection",
        "status": "rejected_not_written",
        "resolved": True,
        "resolution": "candidate_excluded_from_evidence",
        "reason": "; ".join(reasons),
        "request_id": task.request_record["request_id"],
        "candidate_index": candidate_index,
        "candidate": rejection.get("candidate"),
        "attempted_validation_repairs": rejection.get("attempted_validation_repairs"),
        "run_id": task.chunk.get("run_id"),
        "source_id": task.chunk.get("source_id"),
        "target_compound": task.target_compound,
        "raw_asset_id": task.chunk.get("raw_asset_id"),
        "chunk_id": task.chunk.get("chunk_id"),
        "source_location": task.chunk.get("source_location"),
        "checksum": task.chunk.get("checksum"),
        "created_at": _utc_now(),
    }


def _status_record(
    chunk: dict[str, Any],
    decision: Any,
    existing: dict[str, Any],
    *,
    request_id: str,
    previous_status: str | None,
    target_compound: str | None,
    target_resolution_status: str,
) -> dict[str, Any]:
    return {
        "run_id": chunk.get("run_id"),
        "chunk_id": chunk.get("chunk_id"),
        "raw_asset_id": chunk.get("raw_asset_id"),
        "parsed_document_id": chunk.get("parsed_document_id"),
        "parsed_table_id": chunk.get("parsed_table_id"),
        "source_id": chunk.get("source_id"),
        "compound": chunk.get("compound"),
        "target_compound": target_compound,
        "target_resolution_status": target_resolution_status,
        "storage_path": chunk.get("storage_path"),
        "source_url": chunk.get("source_url"),
        "checksum": chunk.get("checksum"),
        "source_location": chunk.get("source_location"),
        "chunk_text_hash": chunk.get("chunk_text_hash"),
        "text_length": len(str(chunk.get("chunk_text") or "")),
        "token_count": chunk.get("token_count"),
        "classification_status": decision.status,
        "classification_action": decision.action,
        "classification_reason": decision.reason,
        "llm_prefilter_candidate": decision.llm_required,
        "llm_required": False,
        "evidence_signal": decision.evidence_signal,
        "natural_language_score": round(decision.natural_language_score, 4),
        "machine_json_score": round(decision.machine_json_score, 4),
        "existing_evidence_count": existing["count"],
        "existing_evidence_methods": sorted(existing["methods"]),
        "existing_evidence_types": sorted(existing["types"]),
        "llm_request_id": request_id,
        "previous_llm_status": previous_status,
    }


def _validate_prerequisites(layout: RawProcessingLayout) -> dict[str, Any]:
    state_path = layout.path("pipeline_state.json")
    if not state_path.exists():
        raise FileNotFoundError(f"pipeline_state.json not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    for prerequisite in ("inventory", "parse", "deterministic_extract"):
        stage = stages.get(prerequisite) if isinstance(stages.get(prerequisite), dict) else {}
        if stage.get("status") != "completed" or stage.get("accepted") is not True:
            raise ValueError(f"LLM extraction prerequisite is not accepted: {prerequisite}")
    for downstream in ("completeness", "embedding"):
        status = (stages.get(downstream) or {}).get("status")
        if status not in {None, "not_started"}:
            raise ValueError(f"Downstream stage already started: {downstream}={status}")
    return state


def _load_upstream_index(layout: RawProcessingLayout) -> UpstreamIndex:
    raw_assets: dict[str, dict[str, Any]] = {}
    targets: set[str] = set()
    for row in _iter_jsonl(layout.path("raw_inventory.jsonl")):
        raw_asset_id = str(row.get("raw_asset_id") or "")
        if raw_asset_id:
            raw_assets[raw_asset_id] = row
        compound = str(row.get("compound") or "").strip()
        if compound and compound.casefold() != "shared":
            targets.add(compound)
    document_ids: set[str] = set()
    document_metadata: dict[str, dict[str, Any]] = {}
    pmc_fulltext_ids: set[str] = set()
    for row in _iter_jsonl(layout.path("parsed_documents.jsonl")):
        document_id = str(row.get("parsed_document_id") or "")
        if not document_id:
            continue
        document_ids.add(document_id)
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        document_metadata[document_id] = {
            "source_id": row.get("source_id"),
            "content_kind": row.get("content_kind"),
            "pmid": metadata.get("pmid"),
            "pmcid": metadata.get("pmcid"),
            "doi": metadata.get("doi"),
        }
        pmcid = str(metadata.get("pmcid") or "").strip().upper()
        if row.get("source_id") == "pmc_oa" and pmcid and row.get("content_kind") != "abstract":
            pmc_fulltext_ids.add(pmcid)
    table_ids = {str(row["parsed_table_id"]) for row in _iter_jsonl(layout.path("parsed_tables.jsonl")) if row.get("parsed_table_id")}
    if not raw_assets or not targets:
        raise ValueError("Raw inventory or target compound set is empty")
    return UpstreamIndex(raw_assets, document_ids, table_ids, targets, document_metadata, pmc_fulltext_ids)


def _needs_structured_reextraction(row: dict[str, Any]) -> bool:
    # This format-only regression produced schema-invalid extra fields. Preserve
    # the original response; a new real call, with its own attempt, replaces it.
    return (row.get("generation_format") == "json_object_xgrammar_0.2.5.post1_ws2"
            and row.get("status") in PROCESSED_LLM_STATUSES and not row.get("candidates")
            and any(str(error).startswith("schema:") for candidate in row.get("rejected_candidates") or []
                    for error in candidate.get("errors") or []))


def _load_processed_responses(path: Path, prompt_version: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        if row.get("prompt_template_version") != prompt_version or row.get("validator_version") != VALIDATOR_VERSION:
            continue
        request_id = str(row.get("request_id") or "")
        if request_id:
            rows[request_id] = row
    return {request_id: row for request_id, row in rows.items() if not _needs_structured_reextraction(row)}


def _load_valid_candidates_by_input_hash(path: Path, prompt_version: str) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    by_hash: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    usable_requests = {request_id for request_id, latest in _load_processed_responses(path, prompt_version).items()
                       if latest.get("status") in PROCESSED_LLM_STATUSES}
    for row in _iter_jsonl(path):
        if str(row.get("request_id") or "") not in usable_requests or _needs_structured_reextraction(row):
            continue
        if (row.get("prompt_template_version") != prompt_version or row.get("validator_version") != VALIDATOR_VERSION
                or row.get("status") not in PROCESSED_LLM_STATUSES):
            continue
        input_hash = str(row.get("input_hash") or "")
        request_id = str(row.get("request_id") or "")
        candidates = row.get("candidates")
        if input_hash and request_id and isinstance(candidates, list):
            by_hash[input_hash] = (request_id, [candidate for candidate in candidates if isinstance(candidate, dict)])
    return by_hash


def _load_request_attempts(path: Path) -> Counter[str]:
    attempts: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if request_id:
            attempts[request_id] = max(attempts[request_id], int(row.get("attempt_number") or 1))
    return attempts


def _load_existing_evidence_ids(path: Path) -> set[str]:
    return {str(row["evidence_id"]) for row in _iter_jsonl(path) if row.get("evidence_id")}


def _load_evidence_index(path: Path) -> dict[str, dict[str, Any]]:
    by_chunk: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        entry = by_chunk.setdefault(chunk_id, {"count": 0, "methods": set(), "types": set()})
        entry["count"] += 1
        if row.get("extraction_method"):
            entry["methods"].add(str(row["extraction_method"]))
        if row.get("evidence_type"):
            entry["types"].add(str(row["evidence_type"]))
    return by_chunk


def _load_failure_ids(path: Path) -> set[str]:
    return {str(row["failure_id"]) for row in _iter_jsonl(path) if row.get("failure_id")}


def _initialize_merged_evidence(deterministic_path: Path, merged_path: Path, *, force: bool) -> None:
    if not deterministic_path.exists():
        raise FileNotFoundError(f"deterministic evidence baseline not found: {deterministic_path}")
    if force or not merged_path.exists():
        _copy_atomic(deterministic_path, merged_path)


def _remove_stage_failures(path: Path, *, stage: str) -> None:
    if not path.exists():
        return
    temporary = path.with_name(f".{path.name}.filter.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in _iter_jsonl(path):
            if row.get("stage") != stage:
                _write_jsonl_line(output, row)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _resolve_recovered_api_failures(path: Path, request_ids: set[str]) -> None:
    if not path.exists() or not request_ids:
        return
    temporary = path.with_name(f".{path.name}.resolve.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in _iter_jsonl(path):
            if (
                row.get("stage") == "llm_extract"
                and row.get("failure_type") == "llm_api_failure"
                and str(row.get("request_id") or "") in request_ids
                and row.get("resolved") is not True
            ):
                row = {
                    **row,
                    "resolved": True,
                    "resolution": "later_attempt_succeeded",
                    "resolved_at": _utc_now(),
                }
            _write_jsonl_line(output, row)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _resolve_ineligible_api_failures(path: Path, eligible_request_ids: set[str]) -> None:
    if not path.exists():
        return
    temporary = path.with_name(f".{path.name}.routing-resolve.tmp")
    changed = False
    with temporary.open("w", encoding="utf-8") as output:
        for row in _iter_jsonl(path):
            if (
                row.get("stage") == "llm_extract"
                and row.get("failure_type") == "llm_api_failure"
                and row.get("resolved") is not True
                and str(row.get("request_id") or "") not in eligible_request_ids
            ):
                row = {
                    **row,
                    "resolved": True,
                    "resolution": "no_longer_eligible_under_current_routing",
                    "resolved_at": _utc_now(),
                    "resolution_routing_version": LLM_ROUTING_VERSION,
                }
                changed = True
            _write_jsonl_line(output, row)
        output.flush()
        os.fsync(output.fileno())
    if changed:
        os.replace(temporary, path)
    else:
        temporary.unlink(missing_ok=True)


def _count_unresolved_stage_failures(path: Path, *, stage: str) -> int:
    return sum(1 for row in _iter_jsonl(path) if row.get("stage") == stage and row.get("resolved") is not True)


def _write_pipeline_state(
    layout: RawProcessingLayout,
    state: dict[str, Any],
    *,
    status: str,
    accepted: bool,
    started_at: str,
    summary_path: Path,
    completed_at: str | None = None,
    error: str | None = None,
    stop_reason: str | None = None,
) -> None:
    stages = state.setdefault("stages", {})
    record: dict[str, Any] = {
        "status": status,
        "accepted": accepted,
        "started_at": started_at,
        "summary_path": str(summary_path),
    }
    if completed_at:
        record["completed_at"] = completed_at
    if error:
        record["error"] = error
    if stop_reason:
        record["stop_reason"] = stop_reason
    stages["llm_extract"] = record
    state["current_stage"] = "llm_extract"
    _write_json_atomic(layout.path("pipeline_state.json", for_write=True), state)


def _write_progress(
    path: Path,
    *,
    run_id: str,
    status: str,
    counts: Counter[str],
    started_at: str,
    stop_reason: str | None = None,
) -> None:
    _write_json_atomic(
        path,
        {
            "stage": "llm_extract",
            "status": status,
            "run_id": run_id,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "chunks_seen": counts["chunks_seen"],
            "status_rows": counts["status_rows"],
            "llm_required": counts["llm_required"],
            "api_requests_submitted": counts["api_requests_submitted"],
            "api_requests_completed": sum(value for key, value in counts.items() if key.startswith("api_status:")),
            "reused_same_target_input_hash": counts["llm_reused_same_target_input_hash"],
            "pending": counts["llm_pending"],
            "evidence_added": counts["evidence_added"],
            "unresolved_api_failures": counts["unresolved_api_failures"],
            "stop_reason": stop_reason,
        },
    )


def _update_processing_summary(layout: RawProcessingLayout, stage_summary: dict[str, Any]) -> None:
    path = layout.path("processing_summary.json", for_write=True)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(
        {
            "current_stage": "llm_extract",
            "llm_extract_status": stage_summary["status"],
            "llm_extract_accepted": stage_summary["accepted"],
            "llm_live_model": stage_summary["model_name"],
            "llm_live_base_url": stage_summary["base_url"],
            "llm_live_request_count": stage_summary["api_request_count"],
            "llm_live_evidence_added": stage_summary["evidence_added_count"],
            "chunk_processing_llm_required": stage_summary["llm_required_chunk_count"],
            "chunk_processing_llm_pending": stage_summary["pending_count"],
            "generated_at": stage_summary["completed_at"],
        }
    )
    evidence_path = layout.path("evidence_candidates.jsonl")
    type_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    total = 0
    for row in _iter_jsonl(evidence_path):
        total += 1
        type_counts[str(row.get("evidence_type") or "unknown")] += 1
        method_counts[str(row.get("extraction_method") or "unknown")] += 1
    payload["evidence_candidate_count"] = total
    payload["evidence_counts_by_type"] = dict(sorted(type_counts.items()))
    payload["evidence_counts_by_extraction_method"] = dict(sorted(method_counts.items()))
    _write_json_atomic(path, payload)


def _counter_subset(counter: Counter[str], prefix: str) -> dict[str, int]:
    return {key[len(prefix) :]: value for key, value in sorted(counter.items()) if key.startswith(prefix)}


def _is_auth_or_quota_failure(error: str | None) -> bool:
    lowered = (error or "").lower()
    return any(marker in lowered for marker in ["401", "403", "invalid_api_key", "incorrect api key", "quota", "insufficient_quota", "balance"])


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _hash_optional_text(value: str | None) -> str | None:
    return sha256(value.encode("utf-8", errors="replace")).hexdigest() if value is not None else None


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        _write_jsonl_line(file, row)
        file.flush()


def _write_jsonl_line(file: TextIO, row: dict[str, Any]) -> None:
    file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    file.write("\n")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            _write_jsonl_line(file, row)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.copy.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
