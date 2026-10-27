from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5


from pbpk_kb.processing.final_evidence_quality_stage import representative_sort_key
from pbpk_kb.processing.final_evidence_attribution_stage import atomic_write_json, load_json, utc_now


FINAL_DEDUPLICATION_DIR = "03_deduplication"
DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"
DEFAULT_THRESHOLD = 0.92
DEFAULT_REVIEW_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 64
NATURAL_LANGUAGE_TYPES = {
    "physicochemical",
    "adme",
    "clinical_pk",
    "ddi",
    "special_population",
    "tissue_expression",
    "public_pbpk_model",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build file-level semantic duplicate groups for final evidence JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--review-threshold", type=float, default=DEFAULT_REVIEW_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-source-chars", type=int, default=1500)
    parser.add_argument("--min-source-chars", type=int, default=40)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--max-group-size", type=int, default=5000)
    parser.add_argument("--hf-endpoint", default=os.getenv("HF_ENDPOINT"))
    parser.add_argument("--reset", action="store_true", help="Delete prior semantic dedup outputs before running.")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = SemanticDedupRunner(
        input_path=input_path,
        output_dir=output_dir,
        model_name=args.model,
        device=args.device,
        threshold=args.threshold,
        review_threshold=args.review_threshold,
        batch_size=max(1, args.batch_size),
        max_source_chars=max(200, args.max_source_chars),
        min_source_chars=max(0, args.min_source_chars),
        min_group_size=max(2, args.min_group_size),
        max_group_size=max(100, args.max_group_size),
        reset=args.reset,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


class SemanticDedupRunner:
    def __init__(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        model_name: str,
        device: str,
        threshold: float,
        review_threshold: float,
        batch_size: int,
        max_source_chars: int,
        min_source_chars: int,
        min_group_size: int,
        max_group_size: int,
        reset: bool,
    ) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.requested_device = device
        self.threshold = threshold
        if not 0.0 < review_threshold < threshold <= 1.0:
            raise ValueError("Expected 0 < review_threshold < threshold <= 1.")
        self.review_threshold = review_threshold
        self.batch_size = batch_size
        self.max_source_chars = max_source_chars
        self.min_source_chars = min_source_chars
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.reset = reset
        deduplication_dir = output_dir / FINAL_DEDUPLICATION_DIR
        self.groups_path = deduplication_dir / "semantic_duplicate_groups.jsonl"
        self.members_path = deduplication_dir / "semantic_duplicate_members.jsonl"
        self.state_path = deduplication_dir / "semantic_dedup_state.jsonl"
        self.summary_path = deduplication_dir / "semantic_dedup_summary.json"
        self.conflicts_path = deduplication_dir / "semantic_conflict_candidates.jsonl"

    def run(self) -> dict[str, Any]:
        started = time.time()
        self.groups_path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_pipeline_state()
        if self.reset:
            for path in [self.groups_path, self.members_path, self.conflicts_path, self.state_path, self.summary_path]:
                path.unlink(missing_ok=True)

        processed_keys = self._load_processed_keys()
        groups, load_counts = self._load_candidate_groups()
        device = self._resolve_device()
        model, util, np = self._load_embedding_stack(device)
        embedding_dimension = int(model.get_sentence_embedding_dimension() or 0)

        counts: Counter[str] = Counter(load_counts)
        counts["candidate_groups_total"] = len(groups)
        counts["candidate_groups_already_processed"] = len(processed_keys)
        groups_fh = self.groups_path.open("a", encoding="utf-8")
        members_fh = self.members_path.open("a", encoding="utf-8")
        conflicts_fh = self.conflicts_path.open("a", encoding="utf-8")
        state_fh = self.state_path.open("a", encoding="utf-8")
        try:
            for index, (group_key, rows) in enumerate(sorted(groups.items()), start=1):
                if group_key in processed_keys:
                    continue
                counts["candidate_groups_seen"] += 1
                counts["candidate_rows_seen"] += len(rows)
                if len(rows) < self.min_group_size:
                    self._write_state(state_fh, group_key, "skipped_small", len(rows), 0, 0)
                    counts["candidate_groups_skipped_small"] += 1
                    continue
                if len(rows) > self.max_group_size:
                    self._write_state(state_fh, group_key, "skipped_too_large", len(rows), 0, 0)
                    counts["candidate_groups_skipped_too_large"] += 1
                    continue

                if index == 1 or index % 25 == 0 or len(rows) >= 500:
                    print(f"semantic_dedup group={index}/{len(groups)} rows={len(rows)} key={group_key[:120]} device={device}", flush=True)

                texts = [semantic_text(row, self.max_source_chars) for row in rows]
                embeddings = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
                written_groups = 0
                written_members = 0
                high_clusters = util.community_detection(
                    embeddings,
                    threshold=self.threshold,
                    min_community_size=self.min_group_size,
                    batch_size=1024,
                )
                high_nonrepresentatives: set[int] = set()
                high_conflict_members: set[int] = set()
                high_review_members: set[int] = set()
                for cluster_index, cluster in enumerate(high_clusters, start=1):
                    cluster_rows = [rows[item] for item in cluster]
                    group_type = classify_high_similarity_cluster(
                        cluster_rows,
                        embeddings,
                        cluster,
                        self.threshold,
                        np,
                    )
                    counts[f"{group_type}_groups"] += 1
                    if group_type == "semantic_conflict_candidate":
                        high_conflict_members.update(cluster)
                    elif group_type == "similar_for_review":
                        high_review_members.update(cluster)
                    representative = select_representative(cluster_rows)
                    representative_index = rows.index(representative)
                    duplicate_group_id = semantic_group_id(group_key, cluster_rows, cluster_index, group_type)
                    group_record = build_group_record(
                        duplicate_group_id=duplicate_group_id,
                        group_type=group_type,
                        group_key=group_key,
                        rows=cluster_rows,
                        representative=representative,
                        model_name=self.model_name,
                        device=device,
                        threshold=self.threshold,
                        review_threshold=self.review_threshold,
                    )
                    write_jsonl(groups_fh, group_record)
                    if group_type == "semantic_conflict_candidate":
                        write_jsonl(conflicts_fh, build_conflict_record(group_record, cluster_rows))
                    elif group_type == "semantic_duplicate":
                        high_nonrepresentatives.update(item for item in cluster if item != representative_index)
                    member_count = write_group_members(
                        members_fh=members_fh,
                        duplicate_group_id=duplicate_group_id,
                        group_type=group_type,
                        rows=rows,
                        cluster=cluster,
                        representative=representative,
                        embeddings=embeddings,
                        np=np,
                    )
                    written_groups += 1
                    written_members += member_count
                    counts[f"{group_type}_members"] += member_count

                review_exclusions = high_nonrepresentatives | high_conflict_members | high_review_members
                review_indexes = remaining_review_indexes(len(rows), review_exclusions)
                if len(review_indexes) >= self.min_group_size:
                    review_embeddings = embeddings[review_indexes]
                    review_clusters = util.community_detection(
                        review_embeddings,
                        threshold=self.review_threshold,
                        min_community_size=self.min_group_size,
                        batch_size=1024,
                    )
                else:
                    review_clusters = []
                for cluster_index, local_cluster in enumerate(review_clusters, start=1):
                    cluster = [review_indexes[item] for item in local_cluster]
                    cluster_rows = [rows[item] for item in cluster]
                    if all_pairwise_at_least(embeddings, cluster, self.threshold, np):
                        continue
                    group_type = "semantic_conflict_candidate" if has_value_or_condition_conflict(cluster_rows) else "similar_for_review"
                    representative = select_representative(cluster_rows)
                    duplicate_group_id = semantic_group_id(
                        group_key,
                        cluster_rows,
                        cluster_index + len(high_clusters),
                        group_type,
                    )
                    group_record = build_group_record(
                        duplicate_group_id=duplicate_group_id,
                        group_type=group_type,
                        group_key=group_key,
                        rows=cluster_rows,
                        representative=representative,
                        model_name=self.model_name,
                        device=device,
                        threshold=self.threshold,
                        review_threshold=self.review_threshold,
                    )
                    write_jsonl(groups_fh, group_record)
                    if group_type == "semantic_conflict_candidate":
                        write_jsonl(conflicts_fh, build_conflict_record(group_record, cluster_rows))
                        counts["semantic_conflict_candidate_groups"] += 1
                    else:
                        counts["similar_for_review_groups"] += 1
                    member_count = write_group_members(
                        members_fh=members_fh,
                        duplicate_group_id=duplicate_group_id,
                        group_type=group_type,
                        rows=rows,
                        cluster=cluster,
                        representative=representative,
                        embeddings=embeddings,
                        np=np,
                    )
                    written_groups += 1
                    written_members += member_count
                    counts[f"{group_type}_members"] += member_count
                self._write_state(state_fh, group_key, "processed", len(rows), written_groups, written_members)
        finally:
            groups_fh.close()
            members_fh.close()
            conflicts_fh.close()
            state_fh.close()

        summary = {
            "input_path": run_relative_path(self.input_path, self.output_dir),
            "output_dir": ".",
            "embedding_model": self.model_name,
            "embedding_device": device,
            "embedding_dimension": embedding_dimension,
            "threshold": self.threshold,
            "review_threshold": self.review_threshold,
            "batch_size": self.batch_size,
            "max_source_chars": self.max_source_chars,
            "min_source_chars": self.min_source_chars,
            "elapsed_seconds": round(time.time() - started, 3),
            **dict(counts),
            "groups_path": run_relative_path(self.groups_path, self.output_dir),
            "members_path": run_relative_path(self.members_path, self.output_dir),
            "state_path": run_relative_path(self.state_path, self.output_dir),
            "conflicts_path": run_relative_path(self.conflicts_path, self.output_dir),
            "accepted": counts["candidate_groups_skipped_too_large"] == 0,
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        self._update_pipeline_state(summary)
        return summary

    def _validate_pipeline_state(self) -> None:
        state_path = self.output_dir / "00_run_manifest" / "pipeline_state.json"
        state = load_json(state_path)
        stages = state.get("stages") or {}
        for stage_name in ("domain_quality_gates", "deterministic_deduplication"):
            stage = stages.get(stage_name) or {}
            if stage.get("status") != "completed" or stage.get("accepted") is not True:
                raise RuntimeError(f"{stage_name} must be completed and accepted before semantic deduplication.")
        semantic = stages.get("semantic_deduplication") or {}
        if semantic.get("status") == "completed" and not self.reset:
            raise RuntimeError("Semantic deduplication is already completed; use --reset for an intentional rerun.")
        expected_input = self.output_dir / FINAL_DEDUPLICATION_DIR / "semantic_dedup_input.jsonl"
        if self.input_path.resolve() != expected_input.resolve():
            raise RuntimeError(f"Semantic dedup input must be the deterministic stage output: {expected_input}")

    def _update_pipeline_state(self, summary: dict[str, Any]) -> None:
        state_path = self.output_dir / "00_run_manifest" / "pipeline_state.json"
        state = load_json(state_path)
        accepted = bool(summary.get("accepted"))
        state["updated_at"] = utc_now()
        state["current_stage"] = "conflict_analysis_ready" if accepted else "semantic_deduplication_blocked"
        state["stages"]["semantic_deduplication"] = {
            "status": "completed" if accepted else "blocked",
            "accepted": accepted,
            "embedding_model": summary.get("embedding_model"),
            "embedding_dimension": summary.get("embedding_dimension"),
            "embedding_device": summary.get("embedding_device"),
            "semantic_duplicate_group_count": summary.get("semantic_duplicate_groups", 0),
            "similar_for_review_group_count": summary.get("similar_for_review_groups", 0),
            "semantic_conflict_candidate_group_count": summary.get("semantic_conflict_candidate_groups", 0),
            "summary_path": run_relative_path(self.summary_path, self.output_dir),
        }
        atomic_write_json(state_path, state)
        quality_path = self.output_dir / "00_run_manifest" / "evidence_quality_report.json"
        quality = load_json(quality_path)
        quality["generated_at"] = utc_now()
        quality["semantic_deduplication"] = summary
        quality["later_steps_executed"] = False
        quality["next_required_step"] = "conflict_analysis" if accepted else "resolve_semantic_deduplication_blocker"
        atomic_write_json(quality_path, quality)

    def _load_processed_keys(self) -> set[str]:
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
                if record.get("status") in {"processed", "skipped_small", "skipped_too_large"}:
                    processed.add(str(record.get("group_key") or ""))
        return {item for item in processed if item}

    def _load_candidate_groups(self) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
        counts: Counter[str] = Counter()
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with self.input_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                counts["input_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["invalid_json_rows"] += 1
                    continue
                evidence_type = str(row.get("evidence_type") or "")
                if evidence_type not in NATURAL_LANGUAGE_TYPES:
                    counts["rows_skipped_evidence_type"] += 1
                    continue
                source_text = str(row.get("source_text") or "")
                if len(source_text.strip()) < self.min_source_chars:
                    counts["rows_skipped_short_source_text"] += 1
                    continue
                compound_id = str(row.get("attributed_compound_id") or row.get("compound_id") or "")
                if not compound_id:
                    counts["rows_skipped_missing_compound"] += 1
                    continue
                group_key = "|".join(
                    [
                        compound_id,
                        normalize_key(evidence_type),
                        normalize_key(row.get("parameter") or ""),
                    ]
                )
                groups[group_key].append(row)
                counts["candidate_rows_loaded"] += 1
        return dict(groups), counts

    def _resolve_device(self) -> str:
        try:
            import torch  # type: ignore
        except Exception:
            return "cpu"
        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
            return "cuda"
        if self.requested_device == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
            return "mps"
        if self.requested_device == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_embedding_stack(self, device: str) -> tuple[Any, Any, Any]:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        import numpy as np  # type: ignore

        model = SentenceTransformer(self.model_name, device=device)
        return model, util, np

    def _write_state(self, file: Any, group_key: str, status: str, row_count: int, group_count: int, member_count: int) -> None:
        write_jsonl(
            file,
            {
                "group_key": group_key,
                "status": status,
                "row_count": row_count,
                "semantic_group_count": group_count,
                "semantic_member_count": member_count,
                "updated_at_epoch": round(time.time(), 3),
            },
        )


def semantic_text(row: dict[str, Any], max_source_chars: int) -> str:
    parts = [
        str(row.get("attributed_compound_name") or row.get("compound_name") or row.get("compound") or ""),
        str(row.get("evidence_type") or ""),
        str(row.get("parameter") or ""),
        str(row.get("value") or row.get("value_text") or ""),
        str(row.get("unit") or ""),
        str(row.get("condition") or ""),
        str(row.get("population") or ""),
        str(row.get("route") or ""),
        str(row.get("dose") or ""),
        str(row.get("source_text") or "")[:max_source_chars],
    ]
    return "\n".join(item for item in parts if item)


def has_value_or_condition_conflict(rows: list[dict[str, Any]]) -> bool:
    for fields in [
        ("canonical_value", "canonical_unit"),
        ("canonical_value_low", "canonical_value_high", "canonical_unit"),
        ("value", "unit"),
        ("value_low", "value_high", "unit"),
        ("condition",),
        ("population",),
        ("value_qualifier",),
        ("value_text",),
        ("species",),
        ("route",),
        ("dose",),
    ]:
        values = {tuple(normalize_key(row.get(field)) for field in fields) for row in rows}
        # Unknown species or an absent qualifier/text value is not evidence of
        # equivalence to the explicit fact in another row.
        if fields not in {("species",), ("value_qualifier",), ("value_text",)}:
            values = {value for value in values if any(part for part in value)}
        if len(values) > 1:
            return True
    return False


def select_representative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=representative_sort_key)


def build_group_record(
    *,
    duplicate_group_id: str,
    group_type: str,
    group_key: str,
    rows: list[dict[str, Any]],
    representative: dict[str, Any],
    model_name: str,
    device: str,
    threshold: float,
    review_threshold: float,
) -> dict[str, Any]:
    source_ids = sorted({str(row.get("source_id") or "") for row in rows if row.get("source_id")})
    return {
        "duplicate_group_id": duplicate_group_id,
        "group_type": group_type,
        "group_key": group_key,
        "evidence_type": representative.get("evidence_type"),
        "compound_id": representative.get("attributed_compound_id") or representative.get("compound_id"),
        "compound_name": representative.get("attributed_compound_name") or representative.get("compound_name"),
        "parameter": representative.get("parameter"),
        "representative_evidence_id": representative.get("evidence_id"),
        "duplicate_count": len(rows),
        "selection_reason": semantic_selection_reason(group_type),
        "source_ids": source_ids,
        "metadata": {
            "algorithm": "pubmedbert_semantic_file_v3",
            "embedding_model": model_name,
            "embedding_device": device,
            "threshold": threshold,
            "review_threshold": review_threshold,
        },
    }


def build_member_record(
    duplicate_group_id: str,
    group_type: str,
    row: dict[str, Any],
    representative: dict[str, Any],
    rank: int,
    similarity: float,
) -> dict[str, Any]:
    return {
        "duplicate_group_id": duplicate_group_id,
        "group_type": group_type,
        "evidence_id": row.get("evidence_id"),
        "is_representative": row.get("evidence_id") == representative.get("evidence_id"),
        "duplicate_rank": rank,
        "similarity_score": round(similarity, 6),
        "duplicate_reason": semantic_selection_reason(group_type),
        "source_id": row.get("source_id"),
        "source_location": row.get("source_location"),
        "source_text_hash": row.get("source_text_hash") or sha256(str(row.get("source_text") or "").encode("utf-8", errors="replace")).hexdigest(),
    }


def semantic_group_id(group_key: str, rows: list[dict[str, Any]], cluster_index: int, group_type: str) -> str:
    evidence_part = "|".join(sorted(str(row.get("evidence_id") or "") for row in rows))
    digest = sha256(f"{group_type}|{group_key}|{cluster_index}|{evidence_part}".encode("utf-8")).hexdigest()
    return str(uuid5(NAMESPACE_URL, f"pbpk:semantic_duplicate:{digest}"))


def semantic_selection_reason(group_type: str) -> str:
    if group_type == "semantic_duplicate":
        return "biomedical embedding similarity >= 0.92 with no value or condition conflict"
    if group_type == "similar_for_review":
        return "biomedical embedding similarity requires review; automatic pairwise duplicate criteria were not satisfied"
    return "high semantic similarity with a value, unit, dose, route, population, or condition conflict"


def write_group_members(
    *,
    members_fh: Any,
    duplicate_group_id: str,
    group_type: str,
    rows: list[dict[str, Any]],
    cluster: list[int],
    representative: dict[str, Any],
    embeddings: Any,
    np: Any,
) -> int:
    representative_index = rows.index(representative)
    representative_embedding = embeddings[representative_index]
    ordered_indexes = sorted(
        cluster,
        key=lambda index: (0 if rows[index].get("evidence_id") == representative.get("evidence_id") else 1, rows[index].get("evidence_id") or ""),
    )
    for rank, index in enumerate(ordered_indexes, start=1):
        row = rows[index]
        similarity = float(np.dot(representative_embedding, embeddings[index]))
        write_jsonl(
            members_fh,
            build_member_record(duplicate_group_id, group_type, row, representative, rank, similarity),
        )
    return len(ordered_indexes)


def build_conflict_record(group: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "duplicate_group_id": group["duplicate_group_id"],
        "group_type": "semantic_conflict_candidate",
        "compound_id": group.get("compound_id"),
        "compound_name": group.get("compound_name"),
        "evidence_type": group.get("evidence_type"),
        "parameter": group.get("parameter"),
        "evidence_ids": sorted(str(row.get("evidence_id") or "") for row in rows),
        "conflicting_fields": conflicting_fields(rows),
        "resolution": "defer_to_conflict_analysis_and_human_review",
    }


def conflicting_fields(rows: list[dict[str, Any]]) -> list[str]:
    conflicts: list[str] = []
    for label, fields in [
        ("canonical_value_unit", ("canonical_value", "canonical_unit")),
        ("canonical_range_unit", ("canonical_value_low", "canonical_value_high", "canonical_unit")),
        ("value_unit", ("value", "unit")),
        ("range_unit", ("value_low", "value_high", "unit")),
        ("condition", ("condition",)),
        ("population", ("population",)),
        ("value_qualifier", ("value_qualifier",)),
        ("value_text", ("value_text",)),
        ("species", ("species",)),
        ("route", ("route",)),
        ("dose", ("dose",)),
    ]:
        values = {tuple(normalize_key(row.get(field)) for field in fields) for row in rows}
        if fields not in {("species",), ("value_qualifier",), ("value_text",)}:
            values = {value for value in values if any(part for part in value)}
        if len(values) > 1:
            conflicts.append(label)
    return conflicts


def all_pairwise_at_least(embeddings: Any, indexes: list[int], threshold: float, np: Any) -> bool:
    for position, left in enumerate(indexes):
        for right in indexes[position + 1 :]:
            if float(np.dot(embeddings[left], embeddings[right])) < threshold:
                return False
    return True


def classify_high_similarity_cluster(
    rows: list[dict[str, Any]],
    embeddings: Any,
    indexes: list[int],
    threshold: float,
    np: Any,
) -> str:
    if has_value_or_condition_conflict(rows):
        return "semantic_conflict_candidate"
    if all_pairwise_at_least(embeddings, indexes, threshold, np):
        return "semantic_duplicate"
    return "similar_for_review"


def remaining_review_indexes(row_count: int, excluded_indexes: set[int]) -> list[int]:
    return [index for index in range(row_count) if index not in excluded_indexes]


def run_relative_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def normalize_key(value: Any) -> str:
    text = str("" if value is None else value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_jsonl(file: Any, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    file.flush()


if __name__ == "__main__":
    main()
