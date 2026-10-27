#!/usr/bin/env python3
"""Recompute manuscript KB descriptors from the hash-verified serving snapshot.

This script is offline: it neither loads configuration/credentials nor opens a
database. Counts describe the 2026-09-13 database readback, not a fresh live query.
Raw response containers and source spans are not independent articles/studies.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "experiments/model_recommendation/runs/expanded_catalog_20260913/benchmark"
FINAL = ROOT / "data_process/final_evidence"
COMPONENT = ROOT / "experiments/component_recommendation/73_gate3_multiretriever_production_index/source_records/evidence_records.jsonl"
RUNS = ("run_20260907T134918Z", "run_20260909T083120Z")
TYPE_LABELS = {
    "clinical_pk": "Clinical pharmacokinetics",
    "adme": "ADME parameters and mechanisms",
    "ddi": "Drug--drug interactions",
    "special_population": "Special populations",
    "public_pbpk_model": "Published PBPK model descriptions",
    "physicochemical": "Physicochemical properties",
}
SOURCE_LABELS = {
    "pmc_oa": "PMC Open Access", "pubmed": "PubMed",
    "dailymed": "DailyMed", "openfda_drug_label": "openFDA drug labels",
    "pubchem": "PubChem", "chembl": "ChEMBL", "pkdb": "PK-DB",
    "osp_pbpk_model_library": "Open Systems Pharmacology model library",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Repeated {key}: counts would require a different attribution policy.")
    return result


def build() -> dict:
    evidence_path = BENCHMARK / "evidence_utilization.jsonl"
    manifest_path = BENCHMARK / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(evidence_path) != manifest["artifacts"][evidence_path.name]:
        raise ValueError("Serving export differs from its frozen manifest.")
    rows = jsonl(evidence_path)
    evidence_ids = set(unique_index(rows, "evidence_id"))

    # The model-request export intentionally omits source fields for model
    # descriptions. Recover them by ID, never by text or a name-only match.
    provenance = {}
    frozen_rows = jsonl(COMPONENT)
    for evidence_id, row in unique_index(frozen_rows, "object_id").items():
        provenance[evidence_id] = row["metadata"]
    inputs = [evidence_path, manifest_path, COMPONENT]
    batch_counts = {}
    for run in RUNS:
        path = FINAL / run / "07_llm_quality_validation/final_auto_validated_evidence.jsonl"
        batch = unique_index(jsonl(path), "evidence_id")
        if not set(batch).issubset(evidence_ids):
            raise ValueError(f"{run} contains records outside the serving snapshot.")
        for evidence_id, row in batch.items():
            if row["quality_status"] != "auto_validated":
                raise ValueError(f"Unvalidated record in final file: {evidence_id}")
            provenance[evidence_id] = row
        inputs.append(path)
        batch_counts[run] = len(batch)

    import_path = FINAL / RUNS[-1] / "00_run_manifest/database_import_summary.json"
    baseline_path = FINAL / RUNS[-1] / "00_run_manifest/database_baseline.json"
    readback = json.loads(import_path.read_text())["database_readback_verification"]
    baseline = json.loads(baseline_path.read_text())
    inputs.extend((import_path, baseline_path))
    if not readback["passed"] or len(rows) != readback["total_serving_evidence_count"]:
        raise ValueError("Import readback and frozen evidence export disagree.")
    if not readback["baseline_row_fingerprints_unchanged"]:
        raise ValueError("Baseline preservation is not verified.")
    if len(rows) != baseline["counts"]["auto_validated_evidence"] + batch_counts[RUNS[-1]]:
        raise ValueError("Baseline-plus-expansion accounting does not reconcile.")

    compounds = Counter()
    types = Counter()
    sources = Counter()
    compound_types = defaultdict(set)
    type_compounds = defaultdict(set)
    spans, containers = set(), set()
    for row in rows:
        evidence_id = row["evidence_id"]
        source = provenance[evidence_id]
        source_id, location = source["source_id"], source["source_location"]
        if row.get("source_id") and row["source_id"] != source_id:
            raise ValueError(f"Source ID disagreement: {evidence_id}")
        if row.get("source_location") and row["source_location"] != location:
            raise ValueError(f"Source location disagreement: {evidence_id}")
        if not location:
            raise ValueError(f"Missing source span: {evidence_id}")
        compounds[row["compound"]] += 1
        types[row["evidence_type"]] += 1
        sources[source_id] += 1
        compound_types[row["compound"]].add(row["evidence_type"])
        type_compounds[row["evidence_type"]].add(row["compound"])
        spans.add((source_id, location))
        # API response files can contain multiple articles or source records.
        containers.add((source_id, location.split("#", 1)[0]))
    if len(compounds) != readback["evidence_covered_compound_count"]:
        raise ValueError("Compound coverage disagrees with database readback.")
    values = sorted(compounds.values())
    quantiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "schema_version": "1.0",
        "scope": "Frozen auto_validated serving-evidence snapshot; no live database query",
        "snapshot_created_at": manifest["created_at"],
        "database_readback_checked_at": readback["checked_at"],
        "count_unit": "Unique evidence_id (one attributed compound per evidence_id in this snapshot)",
        "quality_interpretation": "Automatically validated against pipeline checks, not expert approval or scientific qualification",
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in inputs],
        "source_code": [
            "src/pbpk_kb/storage/final_evidence_query.py",
            "src/pbpk_kb/recommendation/model_matching/expanded_benchmark.py",
            "src/pbpk_kb/processing/final_evidence_attribution_stage.py",
            "src/pbpk_kb/processing/final_evidence_quality_stage.py",
            "src/pbpk_kb/processing/auto_quality_stage.py",
        ],
        "evidence_count": len(rows), "compound_count": len(compounds),
        "evidence_type_count": len(types), "source_system_count": len(sources),
        "source_location_count": len(spans), "raw_response_container_count": len(containers),
        "unique_article_or_study_count": None,
        "document_count_limit": "Containers may hold multiple papers/records; spans can share a paper. Neither count is an independent study count.",
        "baseline_evidence_count": baseline["counts"]["auto_validated_evidence"],
        "final_batch_evidence_counts": batch_counts,
        "identity_join_missing_count": len(evidence_ids - provenance.keys()),
        "model_descriptive_evidence_note": "233 descriptive facts are not 233 distinct executable models; the model-version catalog is counted separately.",
        "compound_distribution": {
            "minimum": min(values), "q1": quantiles[0], "median": statistics.median(values),
            "q3": quantiles[2], "maximum": max(values), "mean": statistics.mean(values),
            "quantile_method": "linear inclusive (Python statistics.quantiles)",
            "top_five_evidence_count": sum(value for _, value in compounds.most_common(5)),
            "top_five_share": sum(value for _, value in compounds.most_common(5)) / len(rows),
            "top_five_compounds": [name for name, _ in compounds.most_common(5)],
            "evidence_type_coverage_distribution": dict(sorted(Counter(len(v) for v in compound_types.values()).items())),
        },
        "compounds": [{"name": name, "evidence_count": compounds[name],
                       "type_count": len(compound_types[name]), "evidence_types": sorted(compound_types[name])}
                      for name in sorted(compounds)],
        "evidence_types": [{"key": key, "label": label, "evidence_count": types[key],
                            "share": types[key] / len(rows), "compound_count": len(type_compounds[key])}
                           for key, label in TYPE_LABELS.items()],
        "source_systems": [{"key": key, "label": SOURCE_LABELS[key], "evidence_count": count,
                            "share": count / len(rows)} for key, count in sources.most_common()],
        "component_experiment_separation": "The earlier frozen component retrieval index is used only to recover ID-linked provenance, not to redefine the later serving snapshot or its experimental sample size.",
    }


def write_tables(stats: dict) -> None:
    tables = ROOT / "results/tables"
    header = [r"% Generated by scripts/build_kb_manuscript_stats.py; frozen snapshot 2026-09-13.",
              r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrlrr@{}}",
              r"\toprule", r"Compound & Records & Types & Compound & Records & Types \\", r"\midrule"]
    compounds = stats["compounds"]
    midpoint = (len(compounds) + 1) // 2
    for left, right in zip(compounds[:midpoint], compounds[midpoint:]):
        header.append(f"{left['name'].capitalize()} & {left['evidence_count']:,} & {left['type_count']} & "
                      f"{right['name'].capitalize()} & {right['evidence_count']:,} & {right['type_count']} " + r"\\")
    header.extend([r"\bottomrule", r"\end{tabular*}"])
    (tables / "knowledge_base_compounds.tex").write_text("\n".join(header) + "\n", encoding="utf-8")
    header = [r"% Generated by scripts/build_kb_manuscript_stats.py; frozen snapshot 2026-09-13.",
              r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}",
              r"\toprule", r"Evidence category & Records & Share (\%) & Compounds \\", r"\midrule"]
    for row in stats["evidence_types"]:
        header.append(f"{row['label']} & {row['evidence_count']:,} & {100*row['share']:.2f} & {row['compound_count']} " + r"\\")
    header.extend([r"\midrule", f"Total (unique) & {stats['evidence_count']:,} & 100.00 & {stats['compound_count']} " + r"\\",
                   r"\bottomrule", r"\end{tabular*}"])
    (tables / "knowledge_base_evidence_types.tex").write_text("\n".join(header) + "\n", encoding="utf-8")


def main() -> None:
    stats = build()
    output = ROOT / "results/data/knowledge_base_description.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tables(stats)
    print(json.dumps({key: stats[key] for key in (
        "snapshot_created_at", "evidence_count", "compound_count", "evidence_type_count",
        "source_system_count", "source_location_count", "raw_response_container_count",
        "identity_join_missing_count", "compound_distribution")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
