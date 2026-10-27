"""Audit saved PBPK model-scoring runs and export paper tables without inference."""
from __future__ import annotations

# Unicode escapes preserve historical source filenames and field names.

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


METHODS = (
    ("bm25_text", "BM25", "full_catalog"),
    ("qwen3_embedding_06b", "Qwen3-Emb-0.6B", "full_catalog"),
    ("f2llm_v2_06b", "F2LLM-v2-0.6B", "full_catalog"),
    ("qwen3_embedding_4b", "Qwen3-Emb-4B", "full_catalog"),
    ("qwen3_reranker_06b", "Qwen3-Rerank-0.6B", "full_catalog"),
    ("jina_reranker_v35", "Jina-Rerank-v3.5", "full_catalog"),
    ("g_reasoner_34m", "G-Reasoner-34M", "full_catalog"),
    ("gfm_rag_8m", "GFM-RAG-8M", "full_catalog"),
    ("gfm_rag_8m_idf_continuous", "GFM-RAG-IDF", "full_catalog"),
    ("bm25_top20_reasonrank_7b_nf4", "ReasonRank-7B-NF4", "top20"),
    ("bm25_top20_rearank_7b_nf4", "Rearank-7B-NF4", "top20"),
    ("bm25_top20_rank_r1_3b", "Rank-R1-3B", "top20"),
)
VARIANTS = ("original", "reordered", "missing_evidence")


def _equal(actual, expected, context):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"Null mismatch: {context}")
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Value mismatch: {context}: {actual} != {expected}")


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), dict)
                             else row.get(key) for key in fields})


def _fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def _table(columns):
    return [r"\begingroup", r"\fontsize{8.5}{10.5}\selectfont", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + columns + r"@{}}", r"\toprule"]


def _write_table(path, lines):
    path.write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular*}", r"\endgroup", ""]), encoding="utf-8")


def _exports(out, rows, generation):
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _csv(tables / "model_recommendation_stability.csv", rows,
         ["method_id", "method", "group", "tie_fraction", "reorder_top1", "reorder_top10", "reorder_n",
          "missing_top1", "missing_top10", "missing_n"])
    _csv(tables / "model_recommendation_runtime.csv", rows,
         ["method_id", "method", "group", "p50_seconds", "p95_seconds", "peak_gpu_gib", "throughput_by_query_batch_size"])
    _csv(tables / "model_recommendation_generation.csv", generation,
         ["method_id", "method", "total", "strict_valid", "postprocessed_or_retried_valid", "fallback"])

    stability = _table("lrrrrrrr") + [
        r"Method & Ties & \multicolumn{3}{c}{Sentence reordering} & \multicolumn{3}{c}{Evidence removal} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & (\%) & Top-1 & Top-10 & \textit{n} & Top-1 & Top-10 & \textit{n} \\", r"\midrule"]
    runtime = _table("lrrr") + [
        r"Method & \multicolumn{2}{c}{Single-query latency (s)} & Peak GPU \\",
        r"\cmidrule(lr){2-3}", r" & P50 & P95 & (GiB) \\", r"\midrule"]
    for group, heading in (("full_catalog", "Full-catalog scoring"), ("top20", "BM25 top-20 reranking")):
        if group == "top20":
            stability.append(r"\midrule"); runtime.append(r"\midrule")
        stability.append(r"\multicolumn{8}{@{}l@{}}{\textit{" + heading + r"}} \\")
        runtime.append(r"\multicolumn{4}{@{}l@{}}{\textit{" + heading + r"}} \\")
        for row in (r for r in rows if r["group"] == group):
            values = [row["method"], _fmt(100 * row["tie_fraction"], 1),
                      _fmt(row["reorder_top1"]), _fmt(row["reorder_top10"]), str(row["reorder_n"]),
                      _fmt(row["missing_top1"]), _fmt(row["missing_top10"]), str(row["missing_n"])]
            if row["method_id"] == "gfm_rag_8m":
                values[3] += r"\textsuperscript{\textdagger}"
                values[6] += r"\textsuperscript{\textdagger}"
            stability.append(" & ".join(values) + r" \\")
            runtime.append(" & ".join([row["method"], _fmt(row["p50_seconds"]),
                                       _fmt(row["p95_seconds"]), _fmt(row["peak_gpu_gib"], 2)]) + r" \\")
    runtime += [r"\midrule", r"\multicolumn{4}{@{}l@{}}{\textit{Recorded batch throughput (not single-query latency)}} \\",
                r"\cmidrule(lr){1-4}", r"Method & Batch size & Queries & Queries/s \\"]
    for row in rows:
        for batch, values in sorted(row.get("throughput_by_query_batch_size", {}).items(), key=lambda item: int(item[0])):
            runtime.append(" & ".join([row["method"], batch, str(values["queries"]),
                                       _fmt(values["queries_per_second"], 4)]) + r" \\")
    generated = _table("lrrrr") + [
        r"Method & Total & \multicolumn{3}{c}{Mutually exclusive outcomes} \\",
        r"\cmidrule(lr){3-5}", r" & & Strict valid & Recovered & Fallback \\", r"\midrule"]
    for row in generation:
        generated.append(" & ".join([row["method"], *(str(row[k]) for k in
                         ("total", "strict_valid", "postprocessed_or_retried_valid", "fallback"))]) + r" \\")
    for name, lines in (("stability", stability), ("runtime", runtime), ("generation", generated)):
        _write_table(tables / f"model_recommendation_{name}.tex", lines)


def build_model_results(root: Path, out: Path) -> dict:
    """Verify immutable saved results, export six tables, and return auditable data."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if not out.is_relative_to(root):
        raise ValueError("Paper outputs must remain inside the project.")
    base = root / "experiments/model_recommendation/results_summary"
    sources = {}

    def read(path):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Source escapes project: {path}")
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw

    def data(path):
        return json.loads(read(path))

    read(Path(__file__))
    report_path = (base / '01_\u5f53\u524d\u5bf9\u7167\u7ed3\u679c/results.json').resolve(strict=True)
    report = data(report_path)
    specification = data(base / '02_\u590d\u73b0\u8d44\u6599/protocol.json')
    manifest_path = base / '02_\u590d\u73b0\u8d44\u6599/inputs/manifest.json'
    manifest = data(manifest_path)
    inputs = {}
    for relative, expected in report["input_sha256"].items():
        raw = read(root / relative)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen input checksum changed: {relative}")
        if relative.endswith(".jsonl"):
            inputs[Path(relative).name] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if sources[str(manifest_path.resolve().relative_to(root))]["sha256"] != specification["input_manifest_sha256"]:
        raise ValueError("Protocol input-manifest checksum does not match.")
    queries, candidates = inputs["queries.jsonl"], inputs["candidates.jsonl"]
    assert len(queries) == manifest["query_count"] == 772
    assert len(candidates) == manifest["candidate_count"] == 280
    assert not manifest["independent_labels_created"] and not report["scientific_claim_eligible"]
    assert specification["variants"] == list(VARIANTS)
    changed_indices = {variant: [i for i, q in enumerate(queries) if q["text"] != q["variants"][variant]["text"]]
                       for variant in VARIANTS[1:]}
    changed = {variant: len(indices) for variant, indices in changed_indices.items()}
    assert changed == {"reordered": 772, "missing_evidence": 304}
    csv_records = list(csv.DictReader(io.StringIO(read(base / '\u5f53\u524d12\u7ec4\u5b9e\u9a8c\u6c47\u603b.csv').decode("utf-8-sig"))))
    csv_by_method = {row['\u65b9\u6cd5']: row for row in csv_records}
    assert len(csv_records) == len(csv_by_method) == 12
    assert set(csv_by_method) == set(report["methods"]) == {m[0] for m in METHODS}
    rows, generation = [], []
    for method_id, name, group in METHODS:
        saved = report["methods"][method_id]
        assert saved["state"] == "complete"
        assert all(metric["value"] is None for metric in saved["scientific_metrics"].values())
        folder = base / '01_\u5f53\u524d\u5bf9\u7167\u7ed3\u679c' / method_id
        index = data(folder / "index.json")
        assert index["variants"] == list(VARIANTS)
        assert index["query_ids"] == [q["query_id"] for q in queries]
        assert index["candidate_ids"] == [c["candidate_id"] for c in candidates]
        scores = np.load(io.BytesIO(read(folder / saved["effective_score_file"])), allow_pickle=False)
        assert scores.shape == (3, 772, 280) and np.isfinite(scores).all()
        assert saved["all_variant_pairs"] == scores.size == 648480
        ranks = np.argsort(-scores, axis=-1, kind="stable")
        timings = np.load(io.BytesIO(read(folder / "query_seconds.npy")), allow_pickle=False)
        assert timings.shape == (3, 772) and np.isfinite(timings).all() and (timings >= 0).all()
        _equal(float(((scores[0] == scores[0].max(axis=1, keepdims=True)).sum(axis=1) > 1).mean()),
               saved["top1_score_tie_query_fraction"], method_id + ":ties")
        row = {"method_id": method_id, "method": name, "group": group,
               "tie_fraction": saved["top1_score_tie_query_fraction"],
               "p50_seconds": saved["latency"]["p50"], "p95_seconds": saved["latency"]["p95"],
               "peak_gpu_gib": saved["peak_gpu_allocated_gib"], "effective_score_file": saved["effective_score_file"]}
        if row["p50_seconds"] is not None:
            assert saved["latency"]["warmup_excluded_queries"] == 1
            _equal(float(np.median(timings[0, 1:])), row["p50_seconds"], method_id + ":P50")
            _equal(float(np.quantile(timings[0, 1:], .95)), row["p95_seconds"], method_id + ":P95")
        else:
            assert row["p95_seconds"] is None
        execution_modes = np.ones((3, len(queries)), dtype=int)
        if "throughput_by_query_batch_size" in saved:
            query_lookup = {q["query_id"]: i for i, q in enumerate(queries)}
            for line in read(folder / "execution_batches.jsonl").splitlines():
                if not line.strip():
                    continue
                batch = json.loads(line)
                for query_id in batch["query_ids"]:
                    execution_modes[VARIANTS.index(batch["variant"]), query_lookup[query_id]] = batch["query_batch_size"]
        for variant, prefix in (("reordered", "reorder"), ("missing_evidence", "missing")):
            stability = saved["robustness"][variant]
            assert stability["changed_input_queries"] == changed[variant]
            n = stability.get("same_execution_batch_size_pairs", changed[variant])
            assert n + stability.get("cross_execution_mode_pairs_excluded", 0) == changed[variant]
            comparable = [i for i in changed_indices[variant]
                          if execution_modes[0, i] == execution_modes[VARIANTS.index(variant), i]]
            assert len(comparable) == n, (method_id, variant, "matched-batch denominator")
            vi = VARIANTS.index(variant)
            top1 = float(np.mean(ranks[0, comparable, 0] == ranks[vi, comparable, 0])) if comparable else None
            top10 = float(np.mean([len(set(ranks[0, i, :10]) & set(ranks[vi, i, :10])) / 10
                                   for i in comparable])) if comparable else None
            _equal(top10, stability["top10_overlap"], method_id + ":" + variant + ":Top10")
            if stability["top1_consistency"] is not None:
                _equal(top1, stability["top1_consistency"], method_id + ":" + variant + ":Top1")
            elif comparable:
                # The default GFM score is a tied retrieval set; this arbitrary
                # candidate-order statistic is checked but never promoted to Top1.
                assert saved["rank_tie_audit"]["original_unique_top1_queries"] == 0
                assert ((scores[0] == scores[0].max(axis=1, keepdims=True)).sum(axis=1) > 1).all()
                _equal(top1, stability["stable_candidate_order_top1_consistency"],
                       method_id + ":" + variant + ":tie-order diagnostic only")
            row.update({prefix + "_top1": stability["top1_consistency"],
                        prefix + "_top10": stability["top10_overlap"], prefix + "_n": n})
        if "throughput_by_query_batch_size" in saved:
            row["throughput_by_query_batch_size"] = saved["throughput_by_query_batch_size"]
            assert row["p50_seconds"] is None and row["p95_seconds"] is None
            for values in row["throughput_by_query_batch_size"].values():
                _equal(values["queries"] / values["batch_wall_seconds"], values["queries_per_second"], method_id + ":throughput")
        old = csv_by_method[method_id]
        assert old['\u72b6\u6001'] == "complete" and int(old['\u8bf7\u6c42\u6570']) == 772 and int(old['\u5019\u9009\u6570']) == 280
        assert int(old['\u5168\u90e8\u53d8\u4f53\u8bc4\u5206\u6570']) == 648480 and old['\u91c7\u7528\u5206\u6570\u6587\u4ef6'] == row["effective_score_file"]
        for field, heading in (("tie_fraction", '\u539f\u59cbTop1\u5e76\u5217\u6bd4\u4f8b'), ("reorder_top1", '\u53e5\u5b50\u91cd\u6392Top1\u4e00\u81f4\u7387'),
                               ("missing_top1", '\u7f3a\u5931\u8bc1\u636eTop1\u4e00\u81f4\u7387'), ("peak_gpu_gib", '\u5cf0\u503cGPU\u5206\u914d_GiB')):
            _equal(row[field], None if old[heading] == "" else float(old[heading]), method_id + ":CSV:" + field)
        rows.append(row)
        if group == "top20":
            total = sum(saved["generation"][v]["records"] for v in VARIANTS)
            strict = sum(saved["generation"][v]["valid_generations"] for v in VARIANTS)
            assert total == 2316
            recovered = 0
            if "official_output_recovery" in saved:
                counts = saved["official_output_recovery"]["counts"]
                assert counts["completed_records"] == total and counts["strict_complete_generations"] == strict
                recovered = counts["postprocessing_repaired_generations"]
                assert counts["unrecoverable_generations"] == total - strict - recovered
            if "official_budget_correction" in saved:
                budget = data(folder / "budget_repair_3172.json")
                assert budget["state"] == "complete" and budget["completed_count"] == 5
                assert budget["strict_success_count"] == budget["usable_order_count"] == 0
                assert hashlib.sha256(read(folder / saved["effective_score_file"])).hexdigest() == budget["output_sha256"]
            generation.append({"method_id": method_id, "method": name, "total": total, "strict_valid": strict,
                               "postprocessed_or_retried_valid": recovered, "fallback": total - strict - recovered})
    assert [(r["strict_valid"], r["postprocessed_or_retried_valid"], r["fallback"]) for r in generation] == [
        (2297, 14, 5), (311, 2002, 3), (1933, 0, 383)]
    observed = data(report_path.parent / report["observed_outcomes_file"])
    assert observed["summary"]["exact_query_candidate_outcome_joins"] == 0
    protocol = {
        "version": report["protocol_version"], "mode": report["mode"], "query_count": 772,
        "compound_count": manifest["compound_count"], "candidate_count": 280,
        "model_family_count": manifest["model_family_count"], "input_variants": list(VARIANTS),
        "changed_input_queries": changed, "independent_gold_available": False,
        "scientific_quality_metrics_available": False, "scientific_metric_status": report["methods"]["bm25_text"]["scientific_metrics"],
        "metric_definitions": {
            "tie_fraction": "Fraction of original-condition requests with multiple exactly equal maximum effective scores; not raw uncorrected scores.",
            "top1_consistency": "Fraction with identical top-ranked candidate under stable sorting; ties broken by frozen candidate order. Null when every original request has a tied Top1.",
            "top10_overlap": "Mean cardinality of intersection of two stable-sort Top10 lists divided by 10; not Jaccard or Recall@10.",
            "stability_n": "Only inputs whose text changed; when recorded batch modes differ, use matched query-batch-size pairs only.",
            "latency": "Recorded original-input P50/P95 excluding the first query. Mixed serial/batch execution remains null, never reconstructed from throughput.",
            "peak_gpu_gib": "Peak allocated GPU memory reported by the run, not total device occupancy; exclusive GPU usage was not verified.",
            "generation": "Strict valid + postprocessed/retried valid + retained BM25 fallback = all 2316 input variants. The 14 ReasonRank recoveries are postprocessing; five budget retries recovered zero.",
        },
        "score_selection": "Use each method's declared effective_score_file; preserve original scoring artifacts and report repairs/fallbacks.",
        "historical_pk_outcomes": observed["summary"], "historical_pk_join": observed["current_benchmark_join"],
        "input_limitations": manifest["limitations"],
        "limitations": [
            "Completed inference and finite scores do not establish recommendation correctness or applicable-model coverage.",
            "No independent ranking/applicability labels or exact query-scenario PK joins; no NDCG, Recall, MRR, PK regret or scientific winner is reported.",
            "Full-catalog scoring and BM25 top20 pipelines are distinct comparison groups.",
            "Input reordering is not natural-language paraphrasing; stable missing-evidence rankings are not necessarily better.",
            "Three input conditions are not random seeds or independent replications; no confidence intervals or significance claims.",
            "GFM-RAG default output is a tied set: Top1 is unavailable and its fixed-size Top10 overlap depends on candidate-order tie breaking.",
            "Batch throughput rows cover recorded batches and may exclude earlier serial execution; two-query Rearank batch-size-2 data are descriptive only.",
            "Postprocessed orders are usable format recoveries, not independently validated scientific recommendations.",
        ],
        "audit": {"csv_matched": True, "input_hashes_matched": True, "score_shapes": [3, 772, 280],
                  "all_effective_scores_finite": True, "score_indices_match_frozen_inputs": True,
                  "matched_batch_denominators_recomputed": True, "stability_metrics_recomputed_from_scores": True,
                  "non_null_latency_quantiles_recomputed_from_saved_timings": True},
    }
    _exports(out, rows, generation)
    return {"rows": rows, "generation": generation, "protocol": protocol, "source_files": sources}
