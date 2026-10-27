from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.supervised_relation import (
    ACCEPTANCE_THRESHOLDS,
    build_supervised_pairs,
    evaluate_locked_silver,
)


RUNNER_VERSION = "pbpk-official-biorex-adapter-1.0.0"
OFFICIAL_REPOSITORY = "https://github.com/ncbi/BioREx"
OFFICIAL_COMMIT = "1f199d8c359eaceb5469538f4daa24aac44c4eae"
OFFICIAL_MODEL_ARCHIVE_SHA256 = (
    "39ac60071f0f97d325074ecdcdf4eb6b9832dc58fd70e736d871b9d6e0b45185"
)
BIOREX_LABELS = [
    "None",
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
    "None-CID",
    "CID",
    "None-PPIm",
    "PPIm",
    "None-AIMED",
    "None-DDI",
    "None-BC7",
    "None-phargkb",
    "None-GDA",
    "None-DISGENET",
    "None-EMU_BC",
    "None-EMU_PC",
    "None-HPRD50",
    "None-PHARMGKB",
    "ACTIVATOR",
    "AGONIST",
    "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR",
    "ANTAGONIST",
    "DIRECT-REGULATOR",
    "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR",
    "INHIBITOR",
    "PART-OF",
    "PRODUCT-OF",
    "SUBSTRATE",
    "SUBSTRATE_PRODUCT-OF",
    "mechanism",
    "int",
    "effect",
    "advise",
    "AIMED-Association",
    "HPRD-Association",
    "EUADR-Association",
    "None-EUADR",
    "Indirect_conversion",
    "Non_conversion",
    "None-Conversion",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and validate official NCBI BioREx DDI relation signals."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--evidence", type=Path, required=True)
    prepare.add_argument("--mentions", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--prepared-dir", type=Path, required=True)
    finalize.add_argument("--processed-tsv", type=Path, required=True)
    finalize.add_argument("--logits", type=Path, required=True)
    finalize.add_argument("--model-config", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        summary = prepare_biorex_input(
            evidence_path=args.evidence,
            mentions_path=args.mentions,
            output_dir=args.output_dir,
        )
    else:
        summary = finalize_biorex_signals(
            prepared_dir=args.prepared_dir,
            processed_tsv_path=args.processed_tsv,
            logits_path=args.logits,
            model_config_path=args.model_config,
            output_dir=args.output_dir,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def prepare_biorex_input(
    *,
    evidence_path: Path,
    mentions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = {
        str(row["evidence_id"]): row
        for row in _iter_jsonl(evidence_path)
        if row.get("evidence_type") == "ddi"
    }
    mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in _iter_jsonl(mentions_path):
        evidence_id = str(mention.get("evidence_id") or "")
        if evidence_id in evidence:
            mentions[evidence_id].append(mention)
    pairs = [
        pair
        for evidence_id, row in evidence.items()
        for pair in build_supervised_pairs(row, mentions.get(evidence_id, []))
    ]
    pairs.sort(key=lambda row: (row["evidence_id"], row["pair_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pubtator_path = output_dir / "biorex_input.pubtator"
    mapping_path = output_dir / "biorex_pair_mapping.jsonl"
    pubtator_blocks: list[str] = []
    mapping_rows: list[dict[str, Any]] = []
    title = "PBPK relation evidence"
    abstract_offset = len(title) + 1
    for index, pair in enumerate(pairs, start=1):
        document_id = f"PBPKRE{index:06d}"
        source_text = str(evidence[pair["evidence_id"]]["source_text"])
        _validate_pair_spans(pair, source_text)
        entity1_id = f"{document_id}_E1"
        entity2_id = f"{document_id}_E2"
        pubtator_blocks.append(
            "\n".join(
                [
                    f"{document_id}|t|{title}",
                    f"{document_id}|a|{source_text}",
                    _pubtator_annotation(
                        document_id,
                        abstract_offset + int(pair["entity1_start"]),
                        abstract_offset + int(pair["entity1_end"]),
                        str(pair["entity1_text"]),
                        entity1_id,
                    ),
                    _pubtator_annotation(
                        document_id,
                        abstract_offset + int(pair["entity2_start"]),
                        abstract_offset + int(pair["entity2_end"]),
                        str(pair["entity2_text"]),
                        entity2_id,
                    ),
                ]
            )
        )
        mapping_rows.append(
            {
                **pair,
                "document_id": document_id,
                "pubtator_entity1_id": entity1_id,
                "pubtator_entity2_id": entity2_id,
                "source_text": source_text,
                "runner_version": RUNNER_VERSION,
            }
        )
    _atomic_text(pubtator_path, "\n\n".join(pubtator_blocks) + ("\n\n" if pairs else ""))
    _atomic_jsonl(mapping_path, mapping_rows)
    summary = {
        "accepted": bool(pairs),
        "runner_version": RUNNER_VERSION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_model_archive_sha256": OFFICIAL_MODEL_ARCHIVE_SHA256,
        "input_evidence_path": str(evidence_path),
        "input_evidence_sha256": _sha256(evidence_path),
        "input_mentions_path": str(mentions_path),
        "input_mentions_sha256": _sha256(mentions_path),
        "ddi_evidence_count": len(evidence),
        "pair_count": len(pairs),
        "silver_label_counts": dict(
            sorted(Counter(row["silver_label"] for row in pairs).items())
        ),
        "pubtator_path": str(pubtator_path),
        "pubtator_sha256": _sha256(pubtator_path),
        "mapping_path": str(mapping_path),
        "mapping_sha256": _sha256(mapping_path),
    }
    _atomic_json(output_dir / "biorex_prepare_summary.json", summary)
    return summary


def finalize_biorex_signals(
    *,
    prepared_dir: Path,
    processed_tsv_path: Path,
    logits_path: Path,
    model_config_path: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    prepare_summary_path = prepared_dir / "biorex_prepare_summary.json"
    mapping_path = prepared_dir / "biorex_pair_mapping.jsonl"
    prepare_summary = json.loads(prepare_summary_path.read_text(encoding="utf-8"))
    if _sha256(mapping_path) != prepare_summary["mapping_sha256"]:
        raise ValueError("BioREx pair mapping checksum does not match prepare manifest.")
    mapping = {str(row["document_id"]): row for row in _iter_jsonl(mapping_path)}
    tsv_rows = _read_processed_tsv(processed_tsv_path)
    logits = _read_logits(logits_path)
    labels = _load_model_labels(model_config_path) if model_config_path else BIOREX_LABELS
    if len(tsv_rows) != len(logits):
        raise ValueError(
            f"BioREx row/logit count mismatch: {len(tsv_rows)} TSV rows vs {len(logits)} logits."
        )
    if len(tsv_rows) != len(mapping):
        raise ValueError(
            f"BioREx row/mapping count mismatch: {len(tsv_rows)} TSV rows vs {len(mapping)} pairs."
        )
    signals: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for tsv_row, row_logits in zip(tsv_rows, logits):
        document_id = tsv_row[0]
        if document_id in seen_documents:
            raise ValueError(f"Duplicate BioREx output document: {document_id}")
        seen_documents.add(document_id)
        pair = mapping.get(document_id)
        if pair is None:
            raise ValueError(f"Unknown BioREx output document: {document_id}")
        if len(row_logits) != len(labels):
            raise ValueError(
                f"{document_id} has {len(row_logits)} logits; expected {len(labels)}."
            )
        probabilities = _softmax(row_logits)
        class_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        label = labels[class_id]
        positive_probability = sum(
            probability
            for candidate_label, probability in zip(labels, probabilities)
            if not candidate_label.startswith("None")
        )
        signals.append(
            {
                **pair,
                "predicted_label": label,
                "predicted_confidence": round(probabilities[class_id], 8),
                "positive_probability": round(positive_probability, 8),
                "positive_signal": not label.startswith("None"),
                "model_repository": OFFICIAL_REPOSITORY,
                "model_commit": OFFICIAL_COMMIT,
                "model_archive_sha256": OFFICIAL_MODEL_ARCHIVE_SHA256,
                "runner_version": RUNNER_VERSION,
            }
        )
    benchmark = evaluate_locked_silver(signals)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path = output_dir / "official_biorex_relation_signals.jsonl"
    _atomic_jsonl(signals_path, signals)
    _atomic_json(output_dir / "official_biorex_relation_benchmark.json", benchmark)
    summary = {
        "accepted": benchmark["accepted"],
        "runner_version": RUNNER_VERSION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_model_archive_sha256": OFFICIAL_MODEL_ARCHIVE_SHA256,
        "prepare_summary_path": str(prepare_summary_path),
        "prepare_summary_sha256": _sha256(prepare_summary_path),
        "processed_tsv_path": str(processed_tsv_path),
        "processed_tsv_sha256": _sha256(processed_tsv_path),
        "logits_path": str(logits_path),
        "logits_sha256": _sha256(logits_path),
        "model_config_path": str(model_config_path) if model_config_path else None,
        "model_config_sha256": _sha256(model_config_path) if model_config_path else None,
        "model_label_count": len(labels),
        "pair_count": len(signals),
        "predicted_label_counts": dict(
            sorted(Counter(row["predicted_label"] for row in signals).items())
        ),
        "benchmark_thresholds": ACCEPTANCE_THRESHOLDS,
        "benchmark_metrics": benchmark["metrics"],
        "benchmark_checks": benchmark["checks"],
        "signals_path": str(signals_path),
        "signals_sha256": _sha256(signals_path),
    }
    _atomic_json(output_dir / "official_biorex_relation_summary.json", summary)
    return summary


def _validate_pair_spans(pair: dict[str, Any], source_text: str) -> None:
    for prefix in ("entity1", "entity2"):
        start = int(pair[f"{prefix}_start"])
        end = int(pair[f"{prefix}_end"])
        text = str(pair[f"{prefix}_text"])
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError(f"{pair['pair_id']} has an invalid {prefix} span.")
        if source_text[start:end] != text:
            raise ValueError(f"{pair['pair_id']} {prefix} text does not match its source span.")
    if min(int(pair["entity1_end"]), int(pair["entity2_end"])) > max(
        int(pair["entity1_start"]), int(pair["entity2_start"])
    ):
        raise ValueError(f"{pair['pair_id']} entity spans overlap.")


def _pubtator_annotation(
    document_id: str,
    start: int,
    end: int,
    text: str,
    entity_id: str,
) -> str:
    clean_text = text.replace("\t", " ").replace("\n", " ")
    return f"{document_id}\t{start}\t{end}\t{clean_text}\tChemicalEntity\t{entity_id}"


def _read_processed_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if row]
    if rows and rows[0][0] == "pmid":
        rows = rows[1:]
    for line_number, row in enumerate(rows, start=1):
        if len(row) < 10:
            raise ValueError(f"{path}:{line_number} has {len(row)} columns; expected at least 10.")
    return rows


def _read_logits(path: Path) -> list[list[float]]:
    output: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                output.append([float(value) for value in line.rstrip().split("\t")])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} contains a non-numeric logit.") from exc
    return output


def _load_model_labels(path: Path) -> list[str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError(f"{path} does not contain a non-empty id2label mapping.")
    indexed: list[tuple[int, str]] = []
    for raw_index, label in id2label.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} contains a non-integer id2label key: {raw_index}") from exc
        indexed.append((index, str(label)))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"{path} id2label indices are not contiguous from zero.")
    return [label for _, label in indexed]


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
