"""Freeze a coverage-oriented core and reuse completed ranking rows, without labels.

Selection reads request inputs only. Scores and timing rows are copied afterwards;
their values never select requests. Existing files are verified, never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import evaluate_frontier_automatic as evaluation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "experiments/model_recommendation/frontier_comparison_20260914"
DEFAULT_OUTPUT = ROOT / "experiments/model_recommendation/frontier_core_145_20260921"
SEED = "pbpk-core145-input-only-coverage-20260921-v1"
FIELDS = ("species", "routes", "populations", "endpoints")
VARIANTS = evaluation.VARIANTS
EXPECTED_COUNTS = {"source_queries": 772, "queries": 145, "candidates": 280,
                   "compounds": 30, "tasks": 5, "methods": 12}
TIMING_NOTE = ("Retained rows of the original 772-query execution. Batch-amortized costs "
               "and warm-up effects are preserved; not a new 145-query latency measurement.")


def scoped(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Outside project: {path}")
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with scoped(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def binding(path: Path) -> dict:
    return {"path": str(scoped(path).relative_to(ROOT)), "sha256": digest(path),
            "bytes": path.stat().st_size}


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def frozen_write(path: Path, data: bytes) -> None:
    path = scoped(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to replace frozen output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def read_lines(path: Path) -> tuple[list[bytes], list[dict]]:
    raw = scoped(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"Uncommitted JSONL tail: {path}")
    lines = raw.splitlines(keepends=True)
    if any(not line.strip() for line in lines):
        raise ValueError(f"Blank JSONL row: {path}")
    return lines, [json.loads(line) for line in lines]


def feature_tokens(query: dict) -> frozenset[tuple[str, str]]:
    return frozenset((field, str(value)) for field in FIELDS
                     for value in (set(query["features"].get(field, [])) or {"__unspecified__"}))


def select_queries(queries: list[dict], seed: str = SEED) -> list[dict]:
    """One per compound/task; maximize uncovered inverse-frequency input tokens.

    Exact rational arithmetic and a seeded ID hash resolve ties reproducibly.
    This intentionally targets condition coverage, not population representativeness.
    """
    ids = [query["query_id"] for query in queries]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Requests must have nonempty unique IDs")
    tokens = {q["query_id"]: feature_tokens(q) for q in queries}
    frequency = Counter(token for values in tokens.values() for token in values)
    remaining = {(q["compound_id"], q["task_type"]) for q in queries}
    covered, selected = set(), set()
    while remaining:
        choices = [q for q in queries if (q["compound_id"], q["task_type"]) in remaining]

        def key(query):
            values = tokens[query["query_id"]]
            gain = sum((Fraction(1, frequency[x]) for x in values - covered), Fraction())
            richness = sum((Fraction(1, frequency[x]) for x in values), Fraction())
            tie = hashlib.sha256((seed + "\0" + query["query_id"]).encode()).hexdigest()
            return (-gain, -richness, tie, query["query_id"])

        chosen = min(choices, key=key)
        selected.add(chosen["query_id"])
        covered.update(tokens[chosen["query_id"]])
        remaining.remove((chosen["compound_id"], chosen["task_type"]))
    return [q for q in queries if q["query_id"] in selected]


def coverage(queries: list[dict]) -> dict:
    fields = {}
    for field in FIELDS:
        counts = Counter(str(v) for q in queries for v in set(q["features"].get(field, [])))
        fields[field] = {"value_request_counts": dict(sorted(counts.items())),
                         "unspecified_requests": sum(not q["features"].get(field) for q in queries)}
    signatures = Counter(tuple(sorted(feature_tokens(q))) for q in queries)
    strata = Counter((q["compound_id"], q["task_type"]) for q in queries)
    return {"request_count": len(queries), "compound_count": len({q["compound_id"] for q in queries}),
            "task_counts": dict(sorted(Counter(q["task_type"] for q in queries).items())),
            "compound_task_strata": len(strata), "unique_joint_condition_signatures": len(signatures),
            "stratified_unique_condition_signatures": len({(q["compound_id"], q["task_type"],
                                                            tuple(sorted(feature_tokens(q)))) for q in queries}),
            "fields": fields}


def copy_batch_provenance(source: Path, destination: Path, selected_ids: set[str]) -> dict:
    result = {}
    path = source / "execution_batches.jsonl"
    if path.exists():
        _, rows = read_lines(path)
        output = []
        for row in rows:
            keep = [qid for qid in row["query_ids"] if qid in selected_ids]
            if keep:
                output.append({**row, "query_ids": keep, "source_query_ids": row["query_ids"],
                               "subset_timing_note": TIMING_NOTE})
        frozen_write(destination / path.name, b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in output))
        result["execution_batches"] = {"source": binding(path), "selected_batch_records": len(output),
                                       "query_batch_size_preserved": True}
    path = source / "generation_diagnostics.jsonl"
    if path.exists():
        references, offset, seen = [], 0, set()
        with path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    raise ValueError(f"Uncommitted diagnostic tail: {path}")
                row = json.loads(line)
                if row.get("query_id") in selected_ids:
                    key = (row["query_id"], row["variant"])
                    if key in seen or row["variant"] not in VARIANTS:
                        raise ValueError(f"Duplicate/invalid selected diagnostic: {key}")
                    seen.add(key)
                    references.append({"query_id": key[0], "variant": key[1], "source_byte_offset": offset,
                                       "source_line_bytes": len(line), "source_line_sha256": hashlib.sha256(line).hexdigest()})
                offset += len(line)
        if seen != {(qid, variant) for qid in selected_ids for variant in VARIANTS}:
            raise ValueError("Selected generation diagnostic coverage is incomplete")
        frozen_write(destination / "diagnostic_references.json", json_bytes({"source": binding(path),
                      "records": references, "policy": "Read original bytes by offset; no duplicated generation text."}))
        result["generation_diagnostics"] = {"source": binding(path), "selected_records": len(references)}
    return result


def build(source_run: Path = DEFAULT_SOURCE, output_run: Path = DEFAULT_OUTPUT, *,
          expected_counts: dict | None = EXPECTED_COUNTS) -> dict:
    source, output = scoped(source_run), scoped(output_run)
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and derived run must be separate sibling directories")
    query_lines, queries = read_lines(source / "inputs/queries.jsonl")
    _, candidates = read_lines(source / "inputs/candidates.jsonl")
    # The entire selection happens before any method output is opened.
    selected = select_queries(queries)
    selected_ids = {q["query_id"] for q in selected}
    indices = [i for i, q in enumerate(queries) if q["query_id"] in selected_ids]
    old_protocol = json.loads((source / "protocol.json").read_text())
    methods = old_protocol["methods"]
    counts = {"source_queries": len(queries), "queries": len(selected), "candidates": len(candidates),
              "compounds": len({q["compound_id"] for q in selected}),
              "tasks": len({q["task_type"] for q in selected}), "methods": len(methods)}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Unexpected frozen experiment counts: {counts}")
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    old_split = evaluation.build_protocol(queries, candidates, {})
    new_split = evaluation.build_protocol(selected, candidates, {})
    if new_split["compound_split"] != old_split["compound_split"]:
        raise ValueError("The original compound calibration split changed")
    report = json.loads((source / "results.json").read_text())
    if set(methods) != set(report["methods"]):
        raise ValueError("Declared methods and completed result report differ")
    evaluation.verify_recorded_input_hashes(source, report, "source results.json")
    # Verify all declared effective outputs before creating any derived artifact.
    for method in methods:
        _, progress = evaluation.load_method(source, method, report["methods"][method], queries, candidates)
        if not progress["included"]:
            raise ValueError(f"Source method is incomplete: {method}")
    before, after = coverage(queries), coverage(selected)
    selection = {"version": "pbpk-core145-input-selection-v1", "seed": SEED,
                 "counts": counts, "source_run": str(source.relative_to(ROOT)),
                 "strategy": "One query per compound_id/task_type; globally greedy uncovered input-condition inverse-frequency gain, total rarity, then seeded query-ID hash; output retains original row order.",
                 "selection_fields": list(FIELDS), "unspecified_condition_included": True,
                 "method_scores_or_judgments_used_for_selection": False,
                 "selected_query_ids": [q["query_id"] for q in selected], "source_query_indices": indices,
                 "selected_queries": [{"query_id": q["query_id"], "source_query_index": old,
                                       "subset_query_index": new, "compound_id": q["compound_id"],
                                       "task_type": q["task_type"], "split": new_split["query_split"][q["query_id"]]}
                                      for new, (old, q) in enumerate(zip(indices, selected))],
                 "compound_split": new_split["compound_split"],
                 "query_split_counts": dict(Counter(new_split["query_split"].values())),
                 "coverage_before": before, "coverage_after": after,
                 "lost_condition_values": {f: sorted(set(before["fields"][f]["value_request_counts"]) -
                                                     set(after["fields"][f]["value_request_counts"])) for f in FIELDS},
                 "limitations": ["Coverage-oriented subset, not an unbiased estimate of the original 772-request distribution.",
                                  "One request per compound/task cannot measure within-stratum condition variation.",
                                  "30 compounds and the original 9/21 compound calibration split remain; correlated requests are not independent compounds.",
                                  "Feature mentions do not establish clinical applicability; this selection creates no scientific gold.",
                                  TIMING_NOTE],
                 "builder": binding(Path(__file__)),
                 "calibration_split_implementation": binding(Path(evaluation.__file__)),
                 "source_files": {name: binding(source / name) for name in
                                  ("inputs/queries.jsonl", "inputs/candidates.jsonl", "inputs/manifest.json", "protocol.json", "results.json")}}
    frozen_write(output / "inputs/queries.jsonl", b"".join(query_lines[i] for i in indices))
    frozen_write(output / "inputs/candidates.jsonl", (source / "inputs/candidates.jsonl").read_bytes())
    input_manifest = {"schema_version": "pbpk-frontier-core-inputs-1.0", "query_count": len(selected),
                      "candidate_count": len(candidates), "compound_count": counts["compounds"],
                      "task_counts": after["task_counts"], "query_candidate_pairs_per_method": len(selected) * len(candidates),
                      "role": "coverage_subset_of_frozen_zero_shot_inputs_without_judgments",
                      "selection_seed": SEED, "selection_manifest": "../selection_manifest.json",
                      "source_manifest": binding(source / "inputs/manifest.json"),
                      "query_rows_preserved_byte_for_byte": True, "all_candidates_preserved_byte_for_byte": True,
                      "all_original_variants_and_graphs_preserved": True,
                      "scientific_claim_eligible": False, "independent_labels_created": False,
                      "outputs": {name: binding(output / "inputs" / name) for name in ("queries.jsonl", "candidates.jsonl")}}
    frozen_write(output / "inputs/manifest.json", json_bytes(input_manifest))
    input_hashes = {name: digest(output / "inputs" / name) for name in ("queries.jsonl", "candidates.jsonl", "manifest.json")}
    protocol = {"version": "frontier-core145-reused-inference-v1", "source_protocol": binding(source / "protocol.json"),
                "methods": methods, "variants": list(VARIANTS), "all_original_queries": len(selected),
                "same_candidate_versions_per_query": len(candidates), "input_manifest_sha256": input_hashes["manifest.json"],
                "own_method_development": "paused", "ranking_rerun": False, "new_scientific_labels_created": False,
                "input_only_coverage_subset": True, "selection_seed": SEED,
                "timing_note": TIMING_NOTE, "original_protocol_remains_authoritative_for_method_execution": True}
    frozen_write(output / "protocol.json", json_bytes(protocol))
    derived_report = {"protocol_version": protocol["version"], "input_sha256": input_hashes,
                      "mode": "frozen_query_axis_subset_of_completed_inference", "scientific_claim_eligible": False,
                      "timing_note": TIMING_NOTE, "source_results": binding(source / "results.json"), "methods": {}}
    selection["methods"] = {}
    for method in methods:
        entry, src, dest = report["methods"][method], source / "results" / method, output / "results" / method
        score_path = scoped(src / entry["effective_score_file"])
        scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
        timings = np.load(src / "query_seconds.npy", mmap_mode="r", allow_pickle=False)
        for name, values in (("scores.npy", scores[:, indices, :]), ("query_seconds.npy", timings[:, indices])):
            buffer = io.BytesIO()
            np.save(buffer, values, allow_pickle=False)
            frozen_write(dest / name, buffer.getvalue())
        frozen_write(dest / "index.json", json_bytes({"query_ids": [q["query_id"] for q in selected],
                                                     "candidate_ids": [c["candidate_id"] for c in candidates], "variants": list(VARIANTS)}))
        provenance = {"effective_scores": binding(score_path), "timings": binding(src / "query_seconds.npy"),
                      "index": binding(src / "index.json"), "status": binding(src / "status.json"),
                      **copy_batch_provenance(src, dest, selected_ids)}
        status = {"method": method, "state": "complete", "input_sha256": input_hashes,
                  "queries": len(selected), "candidates": len(candidates), "variants": list(VARIANTS),
                  "completed_query_variants": len(selected) * len(VARIANTS),
                  "completed_pairs": len(selected) * len(candidates) * len(VARIANTS),
                  "derived_from_existing_scores": True, "ranking_rerun": False, "source_artifacts": provenance,
                  "query_seconds_semantics": TIMING_NOTE, "scientific_claim_eligible": False}
        frozen_write(dest / "status.json", json_bytes(status))
        derived_report["methods"][method] = {"state": "complete", "effective_score_file": "scores.npy",
                                              "full_catalog_queries": len(selected), "candidate_count": len(candidates),
                                              "original_pairs": len(selected) * len(candidates),
                                              "ranking_rerun": False, "source_effective_score_file": str(score_path.relative_to(ROOT)),
                                              "timing_note": TIMING_NOTE}
        selection["methods"][method] = {"sources": provenance, "derived_scores": binding(dest / "scores.npy"),
                                         "derived_timings": binding(dest / "query_seconds.npy")}
    frozen_write(output / "results.json", json_bytes(derived_report))
    frozen_write(output / "selection_manifest.json", json_bytes(selection))
    for method, entry in derived_report["methods"].items():
        evaluation.load_method(output, method, entry, selected, candidates)
        expected_modes = evaluation.execution_batch_modes(source, method, queries)[:, indices]
        actual_modes = evaluation.execution_batch_modes(output, method, selected)
        if not np.array_equal(actual_modes, expected_modes):
            raise ValueError(f"Execution batch mode changed during slicing: {method}")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.source_run, args.output_run)
    print(json.dumps({"state": "frozen_and_verified", "counts": result["counts"],
                      "query_split_counts": result["query_split_counts"],
                      "lost_condition_values": result["lost_condition_values"]}, indent=2))


if __name__ == "__main__":
    main()
