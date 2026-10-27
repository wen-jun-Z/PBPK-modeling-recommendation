from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.processing.final_evidence_attribution_stage import atomic_write_json, load_json, utc_now
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"
EXPECTED_EMBEDDING_DIMENSION = 768
RAG_INDEX_DIR = "06_rag_index"
LAYER_PRIORITY = {"context": 0, "reviewable": 1, "accepted": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the post-final-evidence RAG retrieval index.")
    parser.add_argument("--raw-processing-dir", type=Path, required=True)
    parser.add_argument("--final-evidence-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "embed", "all", "audit"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow an explicit CPU embedding run.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reset", action="store_true", help="Replace the current run's RAG outputs intentionally.")
    args = parser.parse_args()

    stage = RagIndexStage(
        raw_processing_dir=args.raw_processing_dir,
        final_evidence_dir=args.final_evidence_dir,
        model_name=args.model,
        requested_device=args.device,
        allow_cpu=args.allow_cpu,
        batch_size=max(1, args.batch_size),
        reset=args.reset,
    )
    if args.mode == "prepare":
        result = stage.prepare()
    elif args.mode == "embed":
        result = stage.embed()
    elif args.mode == "audit":
        result = stage.audit()
    else:
        result = {"prepare": stage.prepare(), "embed": stage.embed()}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


class RagIndexStage:
    def __init__(
        self,
        *,
        raw_processing_dir: Path,
        final_evidence_dir: Path,
        model_name: str = DEFAULT_MODEL,
        requested_device: str = "auto",
        allow_cpu: bool = False,
        batch_size: int = 128,
        reset: bool = False,
    ) -> None:
        self.raw_layout = raw_processing_layout(raw_processing_dir)
        self.raw_processing_dir = self.raw_layout.root
        self.final_evidence_dir = final_evidence_dir
        self.model_name = model_name
        self.requested_device = requested_device
        self.allow_cpu = allow_cpu
        self.batch_size = batch_size
        self.reset = reset
        self.manifest_dir = final_evidence_dir / "00_run_manifest"
        self.layers_dir = final_evidence_dir / "05_evidence_layers"
        self.index_dir = final_evidence_dir / RAG_INDEX_DIR
        self.records_path = self.index_dir / "retrieval_records.jsonl"
        self.embeddings_path = self.index_dir / "retrieval_embeddings.npy"
        self.summary_path = self.index_dir / "rag_index_summary.json"
        self.state_path = self.manifest_dir / "pipeline_state.json"

    def prepare(self) -> dict[str, Any]:
        started = time.time()
        state = self._validate_prepare_state()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.reset:
            for path in (self.records_path, self.embeddings_path, self.summary_path):
                path.unlink(missing_ok=True)
        elif self.records_path.exists() or self.embeddings_path.exists():
            raise RuntimeError("RAG outputs already exist; use --reset for an intentional rebuild or --mode embed to resume.")

        reviewable = list(iter_jsonl(self.layers_dir / "reviewable_evidence.jsonl"))
        accepted = list(iter_jsonl(self.layers_dir / "accepted_evidence.jsonl"))
        records, counts = self._build_records(reviewable, accepted)
        for embedding_row, record in enumerate(records):
            record["embedding_row"] = embedding_row
            record["embedding_text"] = build_embedding_text(record)

        atomic_write_jsonl(self.records_path, records)
        records_sha256 = file_sha256(self.records_path)
        summary = {
            "run_id": state.get("run_id"),
            "status": "prepared",
            "accepted": False,
            "generated_at": utc_now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "selection_policy": "final_evidence_allowlist_v1",
            "embedding_model": self.model_name,
            "expected_embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "record_count": len(records),
            "records_sha256": records_sha256,
            "records_path": run_relative_path(self.records_path, self.final_evidence_dir),
            "embeddings_path": run_relative_path(self.embeddings_path, self.final_evidence_dir),
            **dict(counts),
        }
        atomic_write_json(self.summary_path, summary)
        self._update_state_prepared(state, summary)
        return summary

    def embed(self) -> dict[str, Any]:
        started = time.time()
        state = load_json(self.state_path)
        stage = (state.get("stages") or {}).get("rag_index") or {}
        if stage.get("status") != "prepared":
            raise RuntimeError("RAG records must be prepared before embedding.")
        if not self.records_path.exists():
            raise FileNotFoundError(self.records_path)
        if self.embeddings_path.exists() and not self.reset:
            raise RuntimeError("RAG embeddings already exist; use --reset for an intentional rerun.")

        records = list(iter_jsonl(self.records_path))
        validate_record_rows(records)
        device = self._resolve_device()
        model, np = self._load_embedding_stack(device)
        dimension = int(model.get_sentence_embedding_dimension() or 0)
        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(f"Expected {EXPECTED_EMBEDDING_DIMENSION}-dimensional embeddings, got {dimension}.")
        texts = [str(record["embedding_text"]) for record in records]
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        validate_embedding_matrix(embeddings, len(records), dimension, np)
        atomic_save_npy(self.embeddings_path, embeddings, np)

        summary = load_json(self.summary_path)
        summary.update(
            {
                "status": "completed",
                "accepted": True,
                "completed_at": utc_now(),
                "elapsed_embedding_seconds": round(time.time() - started, 3),
                "embedding_model": self.model_name,
                "embedding_device": device,
                "embedding_dimension": dimension,
                "embedding_count": int(embeddings.shape[0]),
                "embedding_dtype": str(embeddings.dtype),
                "embeddings_sha256": file_sha256(self.embeddings_path),
            }
        )
        atomic_write_json(self.summary_path, summary)
        self._update_state_completed(state, summary)
        return summary

    def audit(self) -> dict[str, Any]:
        if not self.records_path.exists() or not self.embeddings_path.exists() or not self.summary_path.exists():
            raise RuntimeError("RAG records, embeddings, and summary are all required for audit.")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to audit the RAG index.") from exc
        records = list(iter_jsonl(self.records_path))
        validate_record_rows(records)
        embeddings = np.load(self.embeddings_path, mmap_mode="r")
        validate_embedding_matrix(embeddings, len(records), EXPECTED_EMBEDDING_DIMENSION, np)
        summary = load_json(self.summary_path)
        errors: list[str] = []
        if summary.get("records_sha256") != file_sha256(self.records_path):
            errors.append("retrieval_records checksum mismatch")
        if summary.get("embeddings_sha256") != file_sha256(self.embeddings_path):
            errors.append("retrieval_embeddings checksum mismatch")
        if summary.get("record_count") != len(records):
            errors.append("record count mismatch")
        if summary.get("embedding_count") != int(embeddings.shape[0]):
            errors.append("embedding count mismatch")
        return {
            "accepted": not errors,
            "errors": errors,
            "record_count": len(records),
            "embedding_shape": list(embeddings.shape),
            "embedding_model": summary.get("embedding_model"),
            "embedding_device": summary.get("embedding_device"),
            "records_sha256": file_sha256(self.records_path),
            "embeddings_sha256": file_sha256(self.embeddings_path),
        }

    def _validate_prepare_state(self) -> dict[str, Any]:
        state = load_json(self.state_path)
        layers = (state.get("stages") or {}).get("evidence_layers") or {}
        if layers.get("status") != "completed" or layers.get("accepted") is not True:
            raise RuntimeError("Accepted evidence layers are required before RAG indexing.")
        rag = (state.get("stages") or {}).get("rag_index") or {}
        if rag.get("status") == "completed" and not self.reset:
            raise RuntimeError("RAG indexing is already completed; use --reset for an intentional rebuild.")
        for path in (self.layers_dir / "reviewable_evidence.jsonl", self.layers_dir / "accepted_evidence.jsonl"):
            if not path.exists():
                raise FileNotFoundError(path)
        return state

    def _build_records(
        self,
        reviewable: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], Counter[str]]:
        records: dict[tuple[str, str, str], dict[str, Any]] = {}
        counts: Counter[str] = Counter()
        chunk_links: dict[str, list[dict[str, Any]]] = defaultdict(list)
        parsed_table_links: dict[str, list[dict[str, Any]]] = defaultdict(list)
        recovered_table_links: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for layer, rows in (("reviewable", reviewable), ("accepted", accepted)):
            counts[f"{layer}_evidence_input_count"] = len(rows)
            for row in rows:
                link = evidence_link(row, layer)
                add_retrieval_record(records, evidence_record(row, layer), counts)
                if row.get("chunk_id"):
                    chunk_links[str(row["chunk_id"])].append(link)
                if row.get("parsed_table_id"):
                    parsed_table_links[str(row["parsed_table_id"])].append(link)
                if row.get("recovered_table_id"):
                    recovered_table_links[str(row["recovered_table_id"])].append(link)

        chunks_path = self.raw_layout.path("text_chunks.jsonl")
        if not chunks_path.exists():
            raise FileNotFoundError(chunks_path)
        anchors: dict[str, dict[str, Any]] = {}
        for chunk in iter_jsonl(chunks_path):
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id in chunk_links:
                anchors[chunk_id] = chunk
        counts["referenced_chunk_id_count"] = len(chunk_links)
        counts["resolved_referenced_chunk_id_count"] = len(anchors)
        counts["missing_referenced_chunk_id_count"] = len(set(chunk_links) - set(anchors))
        if counts["missing_referenced_chunk_id_count"]:
            raise RuntimeError("Some reviewable/accepted chunk references are missing from text_chunks.jsonl.")

        context_targets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for chunk_id, anchor in anchors.items():
            document_id = str(anchor.get("parsed_document_id") or "")
            source_prefix = chunk_location_prefix(str(anchor.get("source_location") or ""))
            chunk_index = integer_or_none(anchor.get("chunk_index"))
            if not document_id or not source_prefix or chunk_index is None:
                continue
            for link in chunk_links[chunk_id]:
                enriched = dict(link)
                enriched["anchor_chunk_id"] = chunk_id
                for offset in (-1, 0, 1):
                    context_targets[(document_id, source_prefix, chunk_index + offset)].append(enriched | {"offset": offset})

        selected_chunk_ids: set[str] = set()
        for chunk in iter_jsonl(chunks_path):
            document_id = str(chunk.get("parsed_document_id") or "")
            source_prefix = chunk_location_prefix(str(chunk.get("source_location") or ""))
            chunk_index = integer_or_none(chunk.get("chunk_index"))
            if chunk_index is None:
                continue
            links = context_targets.get((document_id, source_prefix, chunk_index)) or []
            for link in links:
                kind = "source_chunk" if int(link["offset"]) == 0 else "adjacent_context"
                add_retrieval_record(records, chunk_record(chunk, link, kind), counts)
                selected_chunk_ids.add(str(chunk.get("chunk_id") or ""))
        counts["selected_chunk_count"] = len(selected_chunk_ids)

        recovered_found: set[str] = set()
        recovered_for_reconstruction: dict[str, dict[str, Any]] = {}
        recovered_path = self.raw_layout.path("recovered_tables.jsonl")
        if recovered_table_links and not recovered_path.exists():
            raise FileNotFoundError(recovered_path)
        if recovered_path.exists():
            for table in iter_jsonl(recovered_path):
                recovered_id = str(table.get("recovered_table_id") or "")
                if recovered_id not in recovered_table_links:
                    continue
                recovered_found.add(recovered_id)
                metadata = table.get("metadata_json") if isinstance(table.get("metadata_json"), dict) else {}
                if metadata.get("structure_reliable") is not True:
                    counts["referenced_recovered_tables_unreliable"] += 1
                    continue
                parsed_id = str(table.get("parsed_table_id") or "")
                if not parsed_id:
                    recovered_for_reconstruction[recovered_id] = table
                    counts["recovered_tables_requiring_row_reconstruction"] += 1
                    continue
                for link in recovered_table_links[recovered_id]:
                    parsed_table_links[parsed_id].append(link | {"recovered_table_id": recovered_id})
        counts["referenced_recovered_table_id_count"] = len(recovered_table_links)
        counts["resolved_recovered_table_id_count"] = len(recovered_found)

        recovered_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        recovered_rows_path = self.raw_layout.path("recovered_table_rows.jsonl")
        if recovered_for_reconstruction and not recovered_rows_path.exists():
            raise FileNotFoundError(recovered_rows_path)
        if recovered_rows_path.exists():
            for row in iter_jsonl(recovered_rows_path):
                recovered_id = str(row.get("recovered_table_id") or "")
                if recovered_id in recovered_for_reconstruction:
                    recovered_rows[recovered_id].append(row)
        for recovered_id, table in recovered_for_reconstruction.items():
            rows = sorted(recovered_rows.get(recovered_id, []), key=lambda row: integer_or_none(row.get("row_index")) or 0)
            if not rows:
                counts["recovered_tables_missing_rows"] += 1
                continue
            for link in recovered_table_links[recovered_id]:
                add_retrieval_record(records, recovered_table_record(table, rows, link), counts)
            counts["reconstructed_recovered_table_count"] += 1

        tables_path = self.raw_layout.path("parsed_tables.jsonl")
        if parsed_table_links and not tables_path.exists():
            raise FileNotFoundError(tables_path)
        parsed_found: set[str] = set()
        if tables_path.exists():
            for table in iter_jsonl(tables_path):
                parsed_id = str(table.get("parsed_table_id") or "")
                if parsed_id not in parsed_table_links:
                    continue
                table_text = str(table.get("table_text") or "").strip()
                if not table_text:
                    counts["referenced_parsed_tables_without_text"] += 1
                    continue
                parsed_found.add(parsed_id)
                for link in parsed_table_links[parsed_id]:
                    add_retrieval_record(records, table_record(table, link), counts)
        counts["referenced_parsed_table_id_count"] = len(parsed_table_links)
        counts["resolved_parsed_table_id_count"] = len(parsed_found)

        result = sorted(
            records.values(),
            key=lambda row: (
                str(row.get("compound_name") or ""),
                str(row.get("source_id") or ""),
                str(row.get("source_location") or ""),
                str(row.get("normalized_text_hash") or ""),
            ),
        )
        counts["retrieval_record_count"] = len(result)
        counts["record_kind_evidence_count"] = sum("evidence" in row["record_kinds"] for row in result)
        counts["record_kind_source_chunk_count"] = sum("source_chunk" in row["record_kinds"] for row in result)
        counts["record_kind_adjacent_context_count"] = sum("adjacent_context" in row["record_kinds"] for row in result)
        counts["record_kind_table_count"] = sum(
            bool({"parsed_table", "recovered_table"} & set(row["record_kinds"])) for row in result
        )
        return result, counts

    def _resolve_device(self) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for RAG embedding generation.") from exc
        requested = self.requested_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("No supported GPU is available; RAG embedding remains incomplete.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        if requested == "cpu" and not self.allow_cpu:
            raise RuntimeError("CPU embedding requires the explicit --allow-cpu flag.")
        return requested

    def _load_embedding_stack(self, device: str) -> tuple[Any, Any]:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the embeddings extra before building the RAG index.") from exc
        return SentenceTransformer(self.model_name, device=device), np

    def _update_state_prepared(self, state: dict[str, Any], summary: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        state["current_stage"] = "rag_index_prepared"
        state.setdefault("stages", {})["rag_index"] = {
            "status": "prepared",
            "accepted": False,
            "record_count": summary["record_count"],
            "embedding_model": self.model_name,
            "expected_embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "records_sha256": summary["records_sha256"],
            "summary_path": run_relative_path(self.summary_path, self.final_evidence_dir),
        }
        atomic_write_json(self.state_path, state)

    def _update_state_completed(self, state: dict[str, Any], summary: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        state["current_stage"] = "rag_index_ready"
        state.setdefault("stages", {})["rag_index"] = {
            "status": "completed",
            "accepted": True,
            "record_count": summary["record_count"],
            "embedding_count": summary["embedding_count"],
            "embedding_model": summary["embedding_model"],
            "embedding_device": summary["embedding_device"],
            "embedding_dimension": summary["embedding_dimension"],
            "records_sha256": summary["records_sha256"],
            "embeddings_sha256": summary["embeddings_sha256"],
            "summary_path": run_relative_path(self.summary_path, self.final_evidence_dir),
        }
        atomic_write_json(self.state_path, state)


def evidence_link(row: dict[str, Any], layer: str) -> dict[str, Any]:
    return {
        "compound_id": str(row.get("compound_id") or row.get("attributed_compound_id") or ""),
        "compound_name": str(row.get("compound_name") or row.get("attributed_compound_name") or row.get("compound") or ""),
        "evidence_id": str(row.get("evidence_id") or ""),
        "evidence_layer": layer,
        "evidence_type": str(row.get("evidence_type") or ""),
        "parameter": row.get("parameter"),
        "value": row.get("canonical_value") if row.get("canonical_value") is not None else row.get("value"),
        "unit": row.get("canonical_unit") or row.get("unit"),
    }


def evidence_record(row: dict[str, Any], layer: str) -> dict[str, Any]:
    link = evidence_link(row, layer)
    return {
        "run_id": row.get("run_id"),
        "record_kinds": ["evidence"],
        "evidence_layers": [layer],
        "evidence_layer": layer,
        "compound_id": link["compound_id"],
        "compound_name": link["compound_name"],
        "evidence_ids": [link["evidence_id"]] if link["evidence_id"] else [],
        "evidence_types": [link["evidence_type"]] if link["evidence_type"] else [],
        "parameters": [str(link["parameter"])] if link["parameter"] not in (None, "") else [],
        "values": [display_value(link["value"], link["unit"])] if link["value"] not in (None, "") else [],
        "chunk_ids": string_list(row.get("chunk_id")),
        "parsed_table_ids": string_list(row.get("parsed_table_id")),
        "recovered_table_ids": string_list(row.get("recovered_table_id")),
        "raw_asset_ids": string_list(row.get("raw_asset_id")),
        "parsed_document_ids": string_list(row.get("parsed_document_id")),
        "source_id": row.get("source_id"),
        "source_location": row.get("source_location"),
        "source_url": row.get("source_url"),
        "storage_path": row.get("storage_path"),
        "checksum": row.get("checksum"),
        "source_text": str(row.get("source_text") or ""),
        "source_text_hash": row.get("source_text_hash"),
        "provenance_status": row.get("provenance_status") or "validated",
        "curation_status": row.get("curation_status"),
    }


def chunk_record(chunk: dict[str, Any], link: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "run_id": chunk.get("run_id"),
        "record_kinds": [kind],
        "evidence_layers": [link["evidence_layer"]] if kind == "source_chunk" else ["context"],
        "evidence_layer": link["evidence_layer"] if kind == "source_chunk" else "context",
        "compound_id": link["compound_id"],
        "compound_name": link["compound_name"],
        "evidence_ids": string_list(link.get("evidence_id")),
        "evidence_types": string_list(link.get("evidence_type")),
        "parameters": string_list(link.get("parameter")),
        "values": string_list(display_value(link.get("value"), link.get("unit")) if link.get("value") not in (None, "") else None),
        "chunk_ids": string_list(chunk.get("chunk_id")),
        "parsed_table_ids": string_list(chunk.get("parsed_table_id")),
        "recovered_table_ids": [],
        "raw_asset_ids": string_list(chunk.get("raw_asset_id")),
        "parsed_document_ids": string_list(chunk.get("parsed_document_id")),
        "source_id": chunk.get("source_id"),
        "source_location": chunk.get("source_location"),
        "source_url": chunk.get("source_url"),
        "storage_path": chunk.get("storage_path"),
        "checksum": chunk.get("checksum"),
        "source_text": str(chunk.get("chunk_text") or ""),
        "source_text_hash": chunk.get("chunk_text_hash"),
        "provenance_status": "validated",
        "curation_status": "needs_review" if link["evidence_layer"] == "reviewable" else "reviewed_accepted",
    }


def table_record(table: dict[str, Any], link: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": table.get("run_id"),
        "record_kinds": ["parsed_table"],
        "evidence_layers": [link["evidence_layer"]],
        "evidence_layer": link["evidence_layer"],
        "compound_id": link["compound_id"],
        "compound_name": link["compound_name"],
        "evidence_ids": string_list(link.get("evidence_id")),
        "evidence_types": string_list(link.get("evidence_type")),
        "parameters": string_list(link.get("parameter")),
        "values": string_list(display_value(link.get("value"), link.get("unit")) if link.get("value") not in (None, "") else None),
        "chunk_ids": [],
        "parsed_table_ids": string_list(table.get("parsed_table_id")),
        "recovered_table_ids": string_list(link.get("recovered_table_id")),
        "raw_asset_ids": string_list(table.get("raw_asset_id")),
        "parsed_document_ids": [],
        "source_id": table.get("source_id"),
        "source_location": table.get("source_location"),
        "source_url": table.get("source_url"),
        "storage_path": table.get("storage_path"),
        "checksum": table.get("checksum"),
        "source_text": str(table.get("table_text") or ""),
        "source_text_hash": table.get("table_hash"),
        "provenance_status": "validated",
        "curation_status": "needs_review" if link["evidence_layer"] == "reviewable" else "reviewed_accepted",
    }


def recovered_table_record(table: dict[str, Any], rows: list[dict[str, Any]], link: dict[str, Any]) -> dict[str, Any]:
    lines: list[str] = []
    caption = str(table.get("caption") or "").strip()
    if caption:
        lines.append(caption)
    columns = table.get("columns_json") if isinstance(table.get("columns_json"), list) else []
    if columns:
        lines.append(" | ".join(str(column) for column in columns))
    lines.extend(str(row.get("row_text") or "").strip() for row in rows if str(row.get("row_text") or "").strip())
    source_text = "\n".join(lines)
    return {
        "run_id": table.get("run_id"),
        "record_kinds": ["recovered_table"],
        "evidence_layers": [link["evidence_layer"]],
        "evidence_layer": link["evidence_layer"],
        "compound_id": link["compound_id"],
        "compound_name": link["compound_name"],
        "evidence_ids": string_list(link.get("evidence_id")),
        "evidence_types": string_list(link.get("evidence_type")),
        "parameters": string_list(link.get("parameter")),
        "values": string_list(display_value(link.get("value"), link.get("unit")) if link.get("value") not in (None, "") else None),
        "chunk_ids": [],
        "parsed_table_ids": [],
        "recovered_table_ids": string_list(table.get("recovered_table_id")),
        "raw_asset_ids": string_list(table.get("raw_asset_id")),
        "parsed_document_ids": [],
        "source_id": table.get("source_id"),
        "source_location": table.get("source_location"),
        "source_url": table.get("source_url"),
        "storage_path": table.get("storage_path"),
        "checksum": table.get("checksum"),
        "source_text": source_text,
        "source_text_hash": None,
        "provenance_status": "validated",
        "curation_status": "needs_review" if link["evidence_layer"] == "reviewable" else "reviewed_accepted",
    }


def add_retrieval_record(
    records: dict[tuple[str, str, str], dict[str, Any]],
    incoming: dict[str, Any],
    counts: Counter[str],
) -> None:
    text = str(incoming.get("source_text") or "").strip()
    compound_id = str(incoming.get("compound_id") or "")
    source_location = str(incoming.get("source_location") or "")
    if not text or not compound_id or not incoming.get("source_id") or not source_location:
        counts["records_skipped_missing_required_fields"] += 1
        return
    normalized_hash = normalized_text_hash(text)
    key = (compound_id, source_location, normalized_hash)
    incoming["source_text"] = text
    actual_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    upstream_text_hash = incoming.get("source_text_hash")
    incoming["upstream_text_hashes"] = string_list(upstream_text_hash) if upstream_text_hash and upstream_text_hash != actual_text_hash else []
    incoming["source_text_hash"] = actual_text_hash
    incoming["normalized_text_hash"] = normalized_hash
    if key not in records:
        incoming["retrieval_record_id"] = str(uuid5(NAMESPACE_URL, "rag|" + "|".join(key)))
        for field in merge_list_fields():
            incoming[field] = sorted(set(str(value) for value in incoming.get(field, []) if value not in (None, "")))
        records[key] = incoming
        counts["records_added"] += 1
        return
    current = records[key]
    for field in merge_list_fields():
        current[field] = sorted(set(current.get(field, [])) | set(incoming.get(field, [])))
    current_layer = str(current.get("evidence_layer") or "context")
    incoming_layer = str(incoming.get("evidence_layer") or "context")
    if LAYER_PRIORITY.get(incoming_layer, 0) > LAYER_PRIORITY.get(current_layer, 0):
        current["evidence_layer"] = incoming_layer
        current["curation_status"] = incoming.get("curation_status")
    counts["records_merged"] += 1


def merge_list_fields() -> tuple[str, ...]:
    return (
        "record_kinds",
        "evidence_layers",
        "evidence_ids",
        "evidence_types",
        "parameters",
        "values",
        "chunk_ids",
        "parsed_table_ids",
        "recovered_table_ids",
        "raw_asset_ids",
        "parsed_document_ids",
        "upstream_text_hashes",
    )


def build_embedding_text(record: dict[str, Any]) -> str:
    parts = [f"Compound: {record['compound_name'] or record['compound_id']}."]
    if record.get("evidence_types"):
        parts.append("Evidence type: " + ", ".join(record["evidence_types"]) + ".")
    if record.get("parameters"):
        parts.append("Parameter: " + ", ".join(record["parameters"]) + ".")
    if record.get("values"):
        parts.append("Value: " + ", ".join(record["values"]) + ".")
    label = "Source table" if {"parsed_table", "recovered_table"} & set(record.get("record_kinds", [])) else "Source text"
    parts.append(f"{label}: {record['source_text']}")
    return " ".join(parts)


def chunk_location_prefix(source_location: str) -> str:
    return re.sub(r"#chunk=\d+$", "", source_location.strip())


def normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def display_value(value: Any, unit: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{value} {unit}".strip() if unit not in (None, "") else str(value)


def string_list(value: Any) -> list[str]:
    return [] if value in (None, "") else [str(value)]


def integer_or_none(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def validate_record_rows(records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("The RAG retrieval corpus is empty.")
    seen: set[str] = set()
    for expected_row, record in enumerate(records):
        required = ("retrieval_record_id", "compound_id", "source_id", "source_location", "source_text", "embedding_text")
        missing = [field for field in required if not record.get(field)]
        if missing:
            raise RuntimeError(f"RAG record {expected_row} is missing required fields: {missing}")
        if record.get("embedding_row") != expected_row:
            raise RuntimeError(f"RAG embedding_row is not contiguous at row {expected_row}.")
        record_id = str(record["retrieval_record_id"])
        if record_id in seen:
            raise RuntimeError(f"Duplicate retrieval_record_id: {record_id}")
        seen.add(record_id)


def validate_embedding_matrix(matrix: Any, record_count: int, dimension: int, np: Any) -> None:
    if tuple(matrix.shape) != (record_count, dimension):
        raise RuntimeError(f"Embedding matrix shape mismatch: {tuple(matrix.shape)} != {(record_count, dimension)}")
    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError("Embedding matrix contains non-finite values.")
    if record_count:
        norms = np.linalg.norm(matrix, axis=1)
        if not bool(np.allclose(norms, np.ones(record_count), atol=1e-3)):
            raise RuntimeError("Embedding vectors are not normalized.")


def atomic_save_npy(path: Path, matrix: Any, np: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, matrix, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_relative_path(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
