"""Meaningful checks for input-only selection and faithful completed-score reuse."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_frontier_core_inputs as core


def query(qid, compound, task="task-a", **features):
    return {"query_id": qid, "compound_id": compound, "task_type": task,
            "features": features, "text": f"Original request {qid}",
            "graph": {"nodes": [{"node_id": qid}], "edges": []},
            "variants": {"reordered": {"text": f"Reordered {qid}"},
                         "missing_evidence": {"text": f"Reduced {qid}"}}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(core.json_bytes(value))


def fixture_run(tmp_path):
    source, output = tmp_path / "source", tmp_path / "core"
    queries = [query("a-common", "a", species=["human"]),
               query("a-rare", "a", species=["mouse"], routes=["inhalation"]),
               query("b-1", "b", populations=["adult"]),
               query("b-2", "b", populations=["pediatric"]),
               query("c-1", "c", endpoints=["auc"]),
               query("c-2", "c", task="task-b", endpoints=["cmax"])]
    candidates = [{"candidate_id": "m2"}, {"candidate_id": "m1"}]
    for name, rows in (("queries", queries), ("candidates", candidates)):
        path = source / "inputs" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows))
    write_json(source / "inputs/manifest.json", {"source": "fixture"})
    hashes = {n: core.digest(source / "inputs" / n) for n in ("queries.jsonl", "candidates.jsonl", "manifest.json")}
    methods = {"bm25_text": {"kind": "bm25"}, "corrected": {"kind": "fixture"}}
    write_json(source / "protocol.json", {"methods": methods})
    report = {"input_sha256": hashes, "methods": {}}
    raw = np.arange(3 * len(queries) * len(candidates), dtype=np.float32).reshape(3, len(queries), len(candidates))
    for method in methods:
        dest = source / "results" / method
        dest.mkdir(parents=True)
        np.save(dest / "scores.npy", raw)
        np.save(dest / "corrected.npy", raw + 1000)
        np.save(dest / "query_seconds.npy", raw[:, :, 0] + .5)
        write_json(dest / "index.json", {"query_ids": [q["query_id"] for q in queries],
                                          "candidate_ids": [c["candidate_id"] for c in candidates],
                                          "variants": list(core.VARIANTS)})
        write_json(dest / "status.json", {"state": "complete", "input_sha256": hashes,
                                           "completed_query_variants": 18, "completed_pairs": 36})
        report["methods"][method] = {"state": "complete", "effective_score_file": "corrected.npy" if method == "corrected" else "scores.npy"}
        batches = [{"variant": v, "query_batch_size": 8, "query_ids": [q["query_id"] for q in queries],
                    "batch_wall_seconds": 30} for v in core.VARIANTS]
        (dest / "execution_batches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in batches))
        diagnostics = [{"variant": v, "query_id": q["query_id"], "large_response": "kept in original"}
                       for v in core.VARIANTS for q in queries]
        (dest / "generation_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in diagnostics))
    write_json(source / "results.json", report)
    return source, output, queries, candidates, raw


def test_selection_is_deterministic_input_only_and_one_per_stratum():
    queries = [query("common", "a", species=["human"]), query("rare", "a", routes=["inhalation"]),
               query("forced", "b", species=["human"])]
    selected = core.select_queries(queries)
    assert {q["query_id"] for q in selected} == {"rare", "forced"}
    assert {q["query_id"] for q in core.select_queries(list(reversed(queries)))} == {"rare", "forced"}
    changed = [{**q, "method_score": 1e9 if q["query_id"] == "common" else -1e9,
                "grade": 3 if q["query_id"] == "common" else 0} for q in queries]
    assert [q["query_id"] for q in core.select_queries(changed)] == [q["query_id"] for q in selected]


def test_faithful_subset_uses_effective_scores_and_preserves_batch_semantics(tmp_path):
    source, output, queries, candidates, raw = fixture_run(tmp_path)
    before = {str(p): core.digest(p) for p in source.rglob("*") if p.is_file()}
    manifest = core.build(source, output, expected_counts=None)
    indices = manifest["source_query_indices"]
    _, selected = core.read_lines(output / "inputs/queries.jsonl")
    assert len(selected) == 4
    assert selected == [queries[i] for i in indices]
    assert (output / "inputs/candidates.jsonl").read_bytes() == (source / "inputs/candidates.jsonl").read_bytes()
    np.testing.assert_array_equal(np.load(output / "results/corrected/scores.npy"), (raw + 1000)[:, indices, :])
    np.testing.assert_array_equal(np.load(output / "results/bm25_text/query_seconds.npy"), (raw[:, :, 0] + .5)[:, indices])
    batches = core.read_lines(output / "results/corrected/execution_batches.jsonl")[1]
    assert all(len(b["query_ids"]) == 4 and b["query_batch_size"] == 8 and b["batch_wall_seconds"] == 30 for b in batches)
    assert all(len(b["source_query_ids"]) == 6 for b in batches)
    references = json.loads((output / "results/corrected/diagnostic_references.json").read_text())
    assert len(references["records"]) == 12
    original = source / "results/corrected/generation_diagnostics.jsonl"
    for reference in references["records"]:
        with original.open("rb") as handle:
            handle.seek(reference["source_byte_offset"])
            row = handle.read(reference["source_line_bytes"])
        assert core.hashlib.sha256(row).hexdigest() == reference["source_line_sha256"]
        assert json.loads(row)["query_id"] == reference["query_id"]
    assert not (output / "results/corrected/generation_diagnostics.jsonl").exists()
    assert before == {str(p): core.digest(p) for p in source.rglob("*") if p.is_file()}
    assert core.build(source, output, expected_counts=None) == manifest


def test_nonfinite_source_rejected_before_creating_subset(tmp_path):
    source, output, *_ = fixture_run(tmp_path)
    path = source / "results/corrected/corrected.npy"
    values = np.load(path)
    values[0, 0, 0] = np.nan
    np.save(path, values)
    with pytest.raises(ValueError, match="Source method is incomplete"):
        core.build(source, output, expected_counts=None)
    assert not output.exists()


def test_index_mismatch_rejected_before_creating_subset(tmp_path):
    source, output, *_ = fixture_run(tmp_path)
    path = source / "results/corrected/index.json"
    index = json.loads(path.read_text())
    index["candidate_ids"].reverse()
    write_json(path, index)
    with pytest.raises(ValueError, match="Method index differs"):
        core.build(source, output, expected_counts=None)
    assert not output.exists()


def test_existing_frozen_results_are_never_overwritten(tmp_path):
    source, output, *_ = fixture_run(tmp_path)
    core.build(source, output, expected_counts=None)
    target = output / "results/corrected/scores.npy"
    target.write_bytes(b"user change")
    with pytest.raises(ValueError, match="Refusing to replace frozen output"):
        core.build(source, output, expected_counts=None)
    assert target.read_bytes() == b"user change"


def test_source_and_destination_cannot_overlap(tmp_path):
    with pytest.raises(ValueError, match="separate sibling"):
        core.build(tmp_path, tmp_path / "nested", expected_counts=None)


def test_coverage_counts_conditions_without_inventing_labels():
    rows = [query("1", "a", species=["human", "human"]), query("2", "a", populations=["adult"])]
    result = core.coverage(rows)
    assert result["fields"]["species"]["value_request_counts"] == {"human": 1}
    assert result["fields"]["species"]["unspecified_requests"] == 1
    assert result["stratified_unique_condition_signatures"] == 2
