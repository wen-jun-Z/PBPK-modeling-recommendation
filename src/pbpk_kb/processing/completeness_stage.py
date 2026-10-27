from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


COMPLETENESS_VERSION = "completeness-v1"
TOPICS = (
    "identity",
    "physicochemical",
    "label",
    "literature",
    "open_full_text",
    "clinical_pk",
    "adme",
    "ddi",
    "special_population",
    "tissue_expression",
    "public_pbpk_model",
)
SOURCE_TOPICS: dict[str, tuple[str, ...]] = {
    "pubchem": ("identity", "physicochemical"),
    "rxnorm": ("identity",),
    "chembl": ("identity", "physicochemical"),
    "identity_discovery": ("identity",),
    "dailymed": ("label",),
    "openfda_drug_label": ("label",),
    "fda_label_search": ("label",),
    "pubmed": ("literature",),
    "pmc_oa": ("open_full_text",),
    "pkdb": ("clinical_pk",),
    "clinicaltrials": ("clinical_pk",),
    "osp_pbpk_model_library": ("public_pbpk_model",),
}
QUERY_TOPIC_MAP = {
    "clinical_pk": "clinical_pk",
    "adme": "adme",
    "ddi": "ddi",
    "special_population": "special_population",
    "pbpk_model": "public_pbpk_model",
    "public_pbpk_model": "public_pbpk_model",
    "formulation_food_effect": "clinical_pk",
}
DDI_RE = re.compile(r"\b(?:drug interactions?|coadministered|inhibitors?|inducers?|CYP\w+|P-gp|BCRP|OATP\w*)\b", re.I)
SPECIAL_RE = re.compile(r"\b(?:renal|hepatic|pediatric|paediatric|geriatric|elderly|pregnan(?:cy|t)|lactation)\b", re.I)
PBPK_RE = re.compile(r"\b(?:PBPK|physiologically based|whole-body|PK-Sim|Simcyp)\b", re.I)


@dataclass
class CoverageBucket:
    raw_asset_ids: set[str] = field(default_factory=set)
    parsed_document_ids: set[str] = field(default_factory=set)
    parsed_record_ids: set[str] = field(default_factory=set)
    parsed_table_ids: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)
    review_item_ids: set[str] = field(default_factory=set)
    failure_ids: set[str] = field(default_factory=set)
    review_reasons: Counter[str] = field(default_factory=Counter)
    failure_reasons: Counter[str] = field(default_factory=Counter)
    ocr_required: bool = False

    @property
    def parsed_count(self) -> int:
        return len(self.parsed_document_ids) + len(self.parsed_record_ids) + len(self.parsed_table_ids)


class CompletenessStage:
    def __init__(self, *, output_dir: Path | str) -> None:
        self.layout = raw_processing_layout(output_dir)
        self.output_dir = self.layout.root.resolve()
        self.started_at = now_iso()
        self.pipeline_state: dict[str, Any] = {}
        self.run_id = ""
        self.compounds: set[str] = set()
        self.sources: set[str] = set()
        self.inventory: dict[str, dict[str, Any]] = {}
        self.asset_topics: dict[str, set[str]] = {}
        self.buckets: dict[tuple[str, str, str], CoverageBucket] = defaultdict(CoverageBucket)
        self.known_raw_ids: set[str] = set()
        self.known_document_ids: set[str] = set()
        self.known_record_ids: set[str] = set()
        self.known_table_ids: set[str] = set()
        self.known_evidence_ids: set[str] = set()
        self.known_review_ids: set[str] = set()
        self.known_failure_ids: set[str] = set()
        self.rejected_candidate_counts: Counter[str] = Counter()

    def run(self) -> dict[str, Any]:
        self.layout.ensure_dirs()
        self._validate_prerequisites()
        self._write_progress("loading_inventory")
        self._load_inventory()
        self._load_parsed_outputs()
        self._load_evidence()
        self._load_table_review_queue()
        self._load_failures()
        self._write_progress("building_matrix")
        rows = self._build_rows()
        provenance_errors = self._validate_provenance(rows)
        summary = self._summary(rows, provenance_errors)
        if not summary["accepted"]:
            raise ValueError(f"Completeness acceptance failed: {provenance_errors[:5]}")
        self._write_csv_atomic(self.layout.path("completeness_matrix.csv", for_write=True), rows)
        self._write_provenance_atomic(
            self.layout.path("completeness_provenance.jsonl", for_write=True), rows
        )
        self._write_json_atomic(self.layout.path("completeness_summary.json", for_write=True), summary)
        self._update_pipeline_state(summary)
        self._write_progress("completed", completed=True, summary=summary)
        return summary

    def _validate_prerequisites(self) -> None:
        state_path = self.layout.path("pipeline_state.json")
        if not state_path.is_file():
            raise FileNotFoundError(f"pipeline_state.json not found: {state_path}")
        self.pipeline_state = json.loads(state_path.read_text(encoding="utf-8"))
        self.run_id = str(self.pipeline_state.get("run_id") or "")
        table_stage = (self.pipeline_state.get("stages") or {}).get("table_recovery") or {}
        if table_stage.get("status") != "completed" or table_stage.get("accepted") is not True:
            raise ValueError("completeness requires table_recovery=completed+accepted")
        embedding = (self.pipeline_state.get("stages") or {}).get("embedding") or {}
        if embedding.get("status") not in {None, "not_started"}:
            raise ValueError("Refusing completeness after embedding has started")
        required = (
            "raw_inventory.jsonl",
            "parsed_documents.jsonl",
            "parsed_records.jsonl",
            "parsed_tables.jsonl",
            "evidence_candidates.jsonl",
            "asset_parse_status.jsonl",
            "table_review_queue.jsonl",
        )
        for name in required:
            if not self.layout.path(name).is_file():
                raise FileNotFoundError(f"Required completeness input missing: {self.layout.path(name)}")

    def _load_inventory(self) -> None:
        for row in iter_jsonl(self.layout.path("raw_inventory.jsonl")):
            raw_id = clean_id(row.get("raw_asset_id"))
            source = clean_id(row.get("source_id"))
            if not raw_id or not source:
                continue
            compound = clean_id(row.get("compound")) or "shared"
            if compound != "shared":
                self.compounds.add(compound)
            self.sources.add(source)
            self.inventory[raw_id] = row
            self.known_raw_ids.add(raw_id)
            topics = topics_for_asset(row)
            self.asset_topics[raw_id] = topics
            for topic in topics:
                self.buckets[(compound, source, topic)].raw_asset_ids.add(raw_id)

    def _load_parsed_outputs(self) -> None:
        specs = (
            ("parsed_documents.jsonl", "parsed_document_id", "parsed_document_ids"),
            ("parsed_records.jsonl", "parsed_record_id", "parsed_record_ids"),
            ("parsed_tables.jsonl", "parsed_table_id", "parsed_table_ids"),
            ("recovered_tables.jsonl", "recovered_table_id", "parsed_table_ids"),
        )
        for filename, id_field, bucket_field in specs:
            path = self.layout.path(filename)
            if not path.is_file():
                continue
            for row in iter_jsonl(path):
                object_id = clean_id(row.get(id_field))
                if not object_id:
                    continue
                raw_id = clean_id(row.get("raw_asset_id"))
                source, compound = self._source_compound(row, raw_id)
                if not source:
                    continue
                topics = self.asset_topics.get(raw_id, set(SOURCE_TOPICS.get(source, ())))
                if not topics:
                    continue
                known = {
                    "parsed_document_ids": self.known_document_ids,
                    "parsed_record_ids": self.known_record_ids,
                    "parsed_table_ids": self.known_table_ids,
                }[bucket_field]
                known.add(object_id)
                for topic in topics:
                    bucket = self.buckets[(compound, source, topic)]
                    getattr(bucket, bucket_field).add(object_id)
                    if raw_id:
                        bucket.raw_asset_ids.add(raw_id)

    def _load_evidence(self) -> None:
        for row in iter_jsonl(self.layout.path("evidence_candidates.jsonl")):
            evidence_id = clean_id(row.get("evidence_id"))
            topic = clean_id(row.get("evidence_type"))
            source = clean_id(row.get("source_id"))
            compound = clean_id(row.get("compound"))
            if not evidence_id or topic not in TOPICS or not source or not compound:
                continue
            self.compounds.add(compound)
            self.sources.add(source)
            self.known_evidence_ids.add(evidence_id)
            bucket = self.buckets[(compound, source, topic)]
            bucket.evidence_ids.add(evidence_id)
            raw_id = clean_id(row.get("raw_asset_id"))
            document_id = clean_id(row.get("parsed_document_id"))
            record_id = clean_id(row.get("parsed_record_id"))
            table_id = clean_id(row.get("parsed_table_id"))
            if raw_id:
                bucket.raw_asset_ids.add(raw_id)
            if document_id:
                bucket.parsed_document_ids.add(document_id)
            if record_id:
                bucket.parsed_record_ids.add(record_id)
            if table_id:
                bucket.parsed_table_ids.add(table_id)

    def _load_table_review_queue(self) -> None:
        for row in iter_jsonl(self.layout.path("table_review_queue.jsonl")):
            review_id = clean_id(row.get("review_id"))
            source = clean_id(row.get("source_id"))
            compound = clean_id(row.get("compound") or row.get("target_compound"))
            if not review_id or not source or not compound:
                continue
            topic = topic_for_table_review(row)
            self.known_review_ids.add(review_id)
            bucket = self.buckets[(compound, source, topic)]
            bucket.review_item_ids.add(review_id)
            reason = clean_id(row.get("review_reason")) or "unspecified"
            bucket.review_reasons[reason] += 1
            raw_id = clean_id(row.get("raw_asset_id"))
            table_id = clean_id(row.get("parsed_table_id") or row.get("recovered_table_id"))
            if raw_id:
                bucket.raw_asset_ids.add(raw_id)
            if table_id:
                bucket.parsed_table_ids.add(table_id)

    def _load_failures(self) -> None:
        acquisition = self.layout.path("acquisition_failure_status.jsonl")
        if acquisition.is_file():
            for row in iter_jsonl(acquisition):
                terminal = clean_id(row.get("terminal_status"))
                if terminal == "resolved":
                    continue
                failure_id = failure_identifier(row, "acquisition")
                source, compound = self._source_compound(row, None)
                if not source:
                    continue
                self.known_failure_ids.add(failure_id)
                for topic in SOURCE_TOPICS.get(source, TOPICS):
                    bucket = self.buckets[(compound, source, topic)]
                    bucket.failure_ids.add(failure_id)
                    bucket.failure_reasons[terminal or "unresolved_acquisition"] += 1

        parse_status = self.layout.path("asset_parse_status.jsonl")
        for row in iter_jsonl(parse_status):
            terminal = clean_id(row.get("terminal_status"))
            warnings = " ".join(str(item) for item in (row.get("warnings") or []))
            if terminal not in {"unsupported", "failed", "ocr_required"} and "ocr_required" not in warnings:
                continue
            raw_id = clean_id(row.get("raw_asset_id"))
            source, compound = self._source_compound(row, raw_id)
            if not source:
                continue
            failure_id = f"parse:{raw_id or stable_hash(row)}"
            self.known_failure_ids.add(failure_id)
            for topic in self.asset_topics.get(raw_id, set(SOURCE_TOPICS.get(source, ()))):
                bucket = self.buckets[(compound, source, topic)]
                bucket.failure_ids.add(failure_id)
                bucket.failure_reasons[terminal or "ocr_required"] += 1
                bucket.ocr_required = terminal == "ocr_required" or "ocr_required" in warnings

        processing = self.layout.path("processing_failures.jsonl")
        if not processing.is_file():
            return
        for row in iter_jsonl(processing):
            status = clean_id(row.get("status"))
            source = clean_id(row.get("source_id")) or "unknown"
            if status != "failed_parse":
                self.rejected_candidate_counts[source] += 1
                continue
            failure_id = clean_id(row.get("failure_id")) or failure_identifier(row, "processing")
            raw_id = clean_id(row.get("raw_asset_id"))
            source, compound = self._source_compound(row, raw_id)
            if not source:
                continue
            self.known_failure_ids.add(failure_id)
            for topic in self.asset_topics.get(raw_id, set(SOURCE_TOPICS.get(source, ()))):
                bucket = self.buckets[(compound, source, topic)]
                bucket.failure_ids.add(failure_id)
                bucket.failure_reasons["failed_parse"] += 1

    def _source_compound(self, row: dict[str, Any], raw_id: str | None) -> tuple[str, str]:
        inventory = self.inventory.get(raw_id or "", {})
        source = clean_id(row.get("source_id") or inventory.get("source_id"))
        compound = clean_id(
            row.get("compound")
            or row.get("target_compound")
            or inventory.get("compound")
        ) or "shared"
        if source:
            self.sources.add(source)
        if compound != "shared":
            self.compounds.add(compound)
        return source, compound

    def _build_rows(self) -> list[dict[str, Any]]:
        for compound in self.compounds:
            for source in self.sources:
                for topic in TOPICS:
                    self.buckets[(compound, source, topic)]
        rows = [self._row(key, bucket) for key, bucket in sorted(self.buckets.items())]
        return rows

    def _row(self, key: tuple[str, str, str], bucket: CoverageBucket) -> dict[str, Any]:
        compound, source, topic = key
        if bucket.evidence_ids:
            status = "candidate_extracted"
        elif bucket.review_item_ids:
            status = "needs_review"
        elif bucket.ocr_required:
            status = "ocr_required"
        elif bucket.parsed_count:
            status = "parsed"
        elif bucket.failure_ids and bucket.raw_asset_ids:
            status = "failed_to_parse"
        elif bucket.raw_asset_ids:
            status = "raw_only"
        elif bucket.failure_ids:
            status = "failed_to_acquire"
        else:
            status = "missing_from_raw"
        parts = [
            f"raw={len(bucket.raw_asset_ids)}",
            f"parsed={bucket.parsed_count}",
            f"candidates={len(bucket.evidence_ids)}",
            f"reviews={len(bucket.review_item_ids)}",
            f"failures={len(bucket.failure_ids)}",
        ]
        if bucket.review_reasons:
            parts.append("review_reasons=" + json.dumps(dict(sorted(bucket.review_reasons.items())), sort_keys=True))
        if bucket.failure_reasons:
            parts.append("failure_reasons=" + json.dumps(dict(sorted(bucket.failure_reasons.items())), sort_keys=True))
        return {
            "run_id": self.run_id,
            "provenance_record_id": "completeness:" + stable_hash(
                {
                    "run_id": self.run_id,
                    "compound": compound,
                    "source_id": source,
                    "topic": topic,
                }
            ),
            "compound": compound,
            "source_id": source,
            "topic": topic,
            "status": status,
            "status_reason": "; ".join(parts),
            "raw_asset_count": len(bucket.raw_asset_ids),
            "parsed_count": bucket.parsed_count,
            "candidate_count": len(bucket.evidence_ids),
            "review_count": len(bucket.review_item_ids),
            "failure_count": len(bucket.failure_ids),
            "raw_asset_ids": sorted(bucket.raw_asset_ids),
            "parsed_document_ids": sorted(bucket.parsed_document_ids),
            "parsed_record_ids": sorted(bucket.parsed_record_ids),
            "parsed_table_ids": sorted(bucket.parsed_table_ids),
            "evidence_ids": sorted(bucket.evidence_ids),
            "review_item_ids": sorted(bucket.review_item_ids),
            "failure_ids": sorted(bucket.failure_ids),
        }

    def _validate_provenance(self, rows: list[dict[str, Any]]) -> list[str]:
        known = {
            "raw_asset_ids": self.known_raw_ids,
            "parsed_document_ids": self.known_document_ids,
            "parsed_record_ids": self.known_record_ids,
            "parsed_table_ids": self.known_table_ids,
            "evidence_ids": self.known_evidence_ids,
            "review_item_ids": self.known_review_ids,
            "failure_ids": self.known_failure_ids,
        }
        errors: list[str] = []
        for row in rows:
            row_key = f"{row['compound']}:{row['source_id']}:{row['topic']}"
            for field_name, known_ids in known.items():
                missing = set(row[field_name]) - known_ids
                if missing:
                    errors.append(f"{row_key}:{field_name}:{sorted(missing)[:3]}")
        expected = len(self.compounds) * len(self.sources) * len(TOPICS)
        actual = sum(1 for row in rows if row["compound"] != "shared")
        if actual != expected:
            errors.append(f"cross_product_rows expected={expected} actual={actual}")
        return errors

    def _summary(self, rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
        statuses = Counter(row["status"] for row in rows)
        topic_statuses = Counter(f"{row['topic']}:{row['status']}" for row in rows)
        source_statuses = Counter(f"{row['source_id']}:{row['status']}" for row in rows)
        shared_count = sum(1 for row in rows if row["compound"] == "shared")
        expected = len(self.compounds) * len(self.sources) * len(TOPICS)
        return {
            "accepted": not errors,
            "completed_at": now_iso(),
            "compound_count": len(self.compounds),
            "completeness_version": COMPLETENESS_VERSION,
            "expected_cross_product_row_count": expected,
            "matrix_row_count": len(rows),
            "missing_combination_count": statuses["missing_from_raw"],
            "output_dir": str(self.output_dir),
            "provenance_errors": errors,
            "provenance_missing_count": len(errors),
            "provenance_row_count": len(rows),
            "rejected_candidate_counts_by_source": dict(sorted(self.rejected_candidate_counts.items())),
            "run_id": self.run_id,
            "shared_row_count": shared_count,
            "source_count": len(self.sources),
            "source_status_counts": dict(sorted(source_statuses.items())),
            "stage": "completeness",
            "started_at": self.started_at,
            "status": "completed",
            "status_counts": dict(sorted(statuses.items())),
            "topic_count": len(TOPICS),
            "topic_status_counts": dict(sorted(topic_statuses.items())),
        }

    def _write_csv_atomic(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.unlink(missing_ok=True)
        id_fields = {
            "raw_asset_ids",
            "parsed_document_ids",
            "parsed_record_ids",
            "parsed_table_ids",
            "evidence_ids",
            "review_item_ids",
            "failure_ids",
        }
        fields = [name for name in rows[0] if name not in id_fields] if rows else []
        try:
            with temporary.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({name: row[name] for name in fields})
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _write_provenance_atomic(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.unlink(missing_ok=True)
        id_fields = (
            "raw_asset_ids",
            "parsed_document_ids",
            "parsed_record_ids",
            "parsed_table_ids",
            "evidence_ids",
            "review_item_ids",
            "failure_ids",
        )
        try:
            with temporary.open("w", encoding="utf-8") as file:
                for row in rows:
                    provenance = {
                        "provenance_record_id": row["provenance_record_id"],
                        "run_id": row["run_id"],
                        "compound": row["compound"],
                        "source_id": row["source_id"],
                        "topic": row["topic"],
                        **{name: row[name] for name in id_fields},
                    }
                    file.write(json.dumps(provenance, ensure_ascii=False, sort_keys=True))
                    file.write("\n")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _write_progress(self, phase: str, *, completed: bool = False, summary: dict[str, Any] | None = None) -> None:
        payload = {
            "completed": completed,
            "phase": phase,
            "run_id": self.run_id,
            "stage": "completeness",
            "updated_at": now_iso(),
        }
        if summary:
            payload["summary"] = summary
        self._write_json_atomic(self.layout.path("completeness_progress.json", for_write=True), payload)

    def _update_pipeline_state(self, summary: dict[str, Any]) -> None:
        state = dict(self.pipeline_state)
        stages = dict(state.get("stages") or {})
        stages["completeness"] = {
            "accepted": True,
            "completed_at": summary["completed_at"],
            "status": "completed",
            "summary_path": str(self.layout.path("completeness_summary.json", for_write=True)),
        }
        stages.setdefault("embedding", {"status": "not_started"})
        state["current_stage"] = "completeness"
        state["stages"] = stages
        self._write_json_atomic(self.layout.path("pipeline_state.json", for_write=True), state)


def topics_for_asset(row: dict[str, Any]) -> set[str]:
    source = clean_id(row.get("source_id"))
    topics = set(SOURCE_TOPICS.get(source, ()))
    context = " ".join(
        clean_id(row.get(name))
        for name in ("group", "request_key", "storage_path")
        if clean_id(row.get(name))
    ).lower()
    for marker, topic in QUERY_TOPIC_MAP.items():
        if re.search(rf"(?:^|[/_]){re.escape(marker)}(?:[/_.-]|$)", context):
            topics.add(topic)
    return topics


def topic_for_table_review(row: dict[str, Any]) -> str:
    reason = clean_id(row.get("review_reason")).lower()
    text = " ".join(
        [
            clean_id(row.get("caption")),
            clean_id(row.get("row_text")),
            " ".join(str(item) for item in (row.get("columns_json") or [])),
        ]
    )
    if "ddi" in reason or DDI_RE.search(text):
        return "ddi"
    if SPECIAL_RE.search(text):
        return "special_population"
    if PBPK_RE.search(text):
        return "public_pbpk_model"
    return "clinical_pk"


def failure_identifier(row: dict[str, Any], prefix: str) -> str:
    for field_name in ("failure_id", "resolution_id", "request_id"):
        value = clean_id(row.get(field_name))
        if value:
            return f"{prefix}:{value}"
    return f"{prefix}:{stable_hash(row)}"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return sha256(payload).hexdigest()[:24]


def clean_id(value: Any) -> str:
    return str(value or "").strip()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Stage 1 compound/source/topic completeness matrix")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = CompletenessStage(output_dir=args.output_dir).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
