"""Audit frozen generative results and recover published listwise orderings.

This is CPU-only postprocessing of saved responses, not another generation or
scientific quality evaluation. Original scores, timings and responses remain
unchanged. Incomplete generations retain their original failure classification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments/model_recommendation/frontier_comparison_20260914"
REARANK = "bm25_top20_rearank_7b_nf4"
REASONRANK = "bm25_top20_reasonrank_7b_nf4"
VARIANTS = ("original", "reordered", "missing_evidence")
OFFICIAL_COMMIT = "43071525a99d8692cadc644bb295ad0483ccc76b"
OFFICIAL_SOURCE = (
    f"https://github.com/lezhang7/Rearank/blob/{OFFICIAL_COMMIT}/rank_gpt.py#L119-L169"
)
OFFICIAL_SOURCE_SHA256 = "4365f3bdf5cff682819f0066b07aaa31d6d2d3e26eea1fa4bdb60885c6273a9e"
REASONRANK_COMMIT = "15d87338ab2bddab30a1ea39a89315ba6d57302b"
REASONRANK_SOURCE = (
    f"https://github.com/8421BCD/ReasonRank/blob/{REASONRANK_COMMIT}/rerank/rankllm.py#L422-L486"
)
REASONRANK_SOURCE_SHA256 = "62996beb1048ad1ab324871a7e1c28fe696d8d99b6b1fb114ab11cdd7b4cdcd2"


def _path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Frontier audit paths must remain inside the project")
    return path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, out, sources):
    path = _path(path)
    data = path.read_bytes()
    sources[str(path.relative_to(out))] = {"sha256": _sha(data), "bytes": len(data)}
    return data


def _json(path, out, sources):
    return json.loads(_read(path, out, sources))


def _array(path, out, sources):
    # Hash the exact byte snapshot used, even while the worker appends results.
    return np.load(io.BytesIO(_read(path, out, sources)), allow_pickle=False)


def _rows(data, *, allow_partial_tail=False):
    rows, partial = [], False
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"Unexpected blank JSONL record on line {index + 1}")
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if allow_partial_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                partial = True
                break
            raise ValueError(f"Invalid JSONL record on line {index + 1}") from None
        if not isinstance(row, dict):
            raise ValueError(f"JSONL record {index + 1} must be an object")
        rows.append(row)
    return rows, partial


def strict_permutation(response, size):
    """The original adapter's exact success criterion, unchanged."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match is None:
        return None
    answer = match.group(1).strip()
    if not re.fullmatch(r"\[\d+\](?:\s*>\s*\[\d+\])*", answer):
        return None
    values = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", answer)]
    return values if len(values) == size and set(values) == set(range(size)) else None


def recover_rearank_order(response, size=20):
    """Official deduplicate/filter/append policy inside a closed answer only.

    The author's broad fallback also scans free-form responses. We deliberately
    decline that path: thought-process numbers are not ranking instructions.
    This conservative deviation is recorded explicitly in output metadata.
    """
    if not isinstance(response, str):
        raise ValueError("Generation response must be a string")
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match is None:
        problem = "unclosed_answer" if "<answer>" in response else "missing_answer"
        return None, {"classifications": [problem], "usable_identifiers": 0,
                      "missing_identifiers": list(range(1, size + 1)),
                      "duplicate_identifiers": [], "out_of_range_identifiers": []}
    answer = match.group(1)
    if "<think>" in answer or "</think>" in answer:
        return None, {"classifications": ["reasoning_inside_answer"], "usable_identifiers": 0,
                      "missing_identifiers": list(range(1, size + 1)),
                      "duplicate_identifiers": [], "out_of_range_identifiers": []}
    return _complete_answer_identifiers(answer, size)


def recover_reasonrank_order(response, size=20):
    """ReasonRank's own _clean_response/receive_permutation, conservatively.

    Its official parser first searches the configured think/answer pattern, then
    deduplicates valid identifiers and appends missing candidates. It also scans
    thinking text when no answer exists; that unsafe fallback is not adopted.
    """
    if not isinstance(response, str):
        raise ValueError("Generation response must be a string")
    lowered = response.lower()
    match = re.search(r"<think>.*?</think>\s*<answer>(.*?)</answer>", lowered, re.DOTALL)
    if match is None:
        match = re.search(r"<answer>(.*?)</answer>", lowered, re.DOTALL)
    if match is None:
        return None, {
            "classifications": ["unclosed_answer" if "<answer>" in lowered else "missing_answer"],
            "usable_identifiers": 0, "missing_identifiers": list(range(1, size + 1)),
            "duplicate_identifiers": [], "out_of_range_identifiers": []}
    answer = match.group(1)
    if "<think>" in answer or "</think>" in answer:
        return None, {"classifications": ["reasoning_inside_answer"], "usable_identifiers": 0,
                      "missing_identifiers": list(range(1, size + 1)),
                      "duplicate_identifiers": [], "out_of_range_identifiers": []}
    return _complete_answer_identifiers(answer, size)


def _complete_answer_identifiers(answer, size):
    # This arithmetic is present independently in both pinned official sources.
    identifiers = [int(x) for x in re.findall(r"\d+", answer)]
    counts = Counter(identifiers)
    unique = list(dict.fromkeys(x - 1 for x in identifiers if 1 <= x <= size))
    missing = [i + 1 for i in range(size) if i not in unique]
    duplicate = [x for x, count in counts.items() if count > 1]
    outside = list(dict.fromkeys(x for x in identifiers if x < 1 or x > size))
    categories = []
    if missing:
        categories.append("omitted_candidates")
    if duplicate:
        categories.append("duplicate_candidates")
    if outside:
        categories.append("out_of_range_candidates")
    if not re.fullmatch(r"\[\d+\](?:\s*>\s*\[\d+\])*", answer.strip()):
        categories.append("noncanonical_answer_format")
    if not unique:
        categories.append("no_valid_candidate_identifier")
    details = {"classifications": categories, "usable_identifiers": len(unique),
               "missing_identifiers": missing, "duplicate_identifiers": duplicate,
               "out_of_range_identifiers": outside}
    if not unique:
        return None, details
    return unique + [i for i in range(size) if i not in unique], details


def _ordinal(order):
    values = np.empty(len(order), dtype=np.float32)
    values[order] = np.arange(len(order), 0, -1, dtype=np.float32)
    return values


def _audit_method(out, method, index, input_hashes, bm25_scores, sources, *, recover):
    folder = out / "results" / method
    required = ("status.json", "index.json", "scores.npy", "query_seconds.npy", "generation_diagnostics.jsonl")
    if not all(_path(folder / name).is_file() for name in required):
        return {"state": "not_ready", "method": method}, None
    status = _json(folder / "status.json", out, sources)
    if status.get("input_sha256") != input_hashes:
        raise ValueError(f"{method}: status refers to different frozen inputs")
    if _json(folder / "index.json", out, sources) != index:
        raise ValueError(f"{method}: query/candidate/variant index mismatch")
    # Read diagnostics before arrays: every captured complete record should have
    # its score/timing visible; a just-written array row may await its diagnostic.
    diagnostics, partial_tail = _rows(
        _read(folder / "generation_diagnostics.jsonl", out, sources),
        allow_partial_tail=status.get("state") != "complete")
    scores = _array(folder / "scores.npy", out, sources)
    timings = _array(folder / "query_seconds.npy", out, sources)
    shape = bm25_scores.shape
    if scores.shape != shape or timings.shape != shape[:2]:
        raise ValueError(f"{method}: result array shape mismatch")
    if np.isinf(scores).any() or np.isinf(timings).any() or (timings[np.isfinite(timings)] < 0).any():
        raise ValueError(f"{method}: invalid infinite score or negative timing")
    completed = np.isfinite(scores).all(axis=-1) & np.isfinite(timings)
    query_indices = {value: i for i, value in enumerate(index["query_ids"])}
    variant_indices = {value: i for i, value in enumerate(index["variants"])}
    unique, duplicate_records = {}, 0
    for record in diagnostics:
        query_id, variant = record.get("query_id"), record.get("variant")
        if query_id not in query_indices or variant not in variant_indices:
            raise ValueError(f"{method}: unknown diagnostic query/variant")
        key = variant_indices[variant], query_indices[query_id]
        if key in unique:
            if record != unique[key]:
                raise ValueError(f"{method}: conflicting duplicate generation record")
            duplicate_records += 1
            continue
        unique[key] = record
    destination = np.full(shape, np.nan, dtype=np.float32) if recover else None
    count_keys = (
        "completed_records", "strict_complete_generations", "strict_invalid_generations",
        "token_limit_hits", "omitted_candidates", "duplicate_candidates", "out_of_range_candidates",
        "noncanonical_answer_format", "missing_answer", "unclosed_answer", "reasoning_inside_answer",
        "no_valid_candidate_identifier", "diagnostic_without_completed_score_and_timing",
    )
    if recover:
        count_keys += ("usable_order_generations", "postprocessing_repaired_generations",
                       "repaired_and_changed_from_bm25", "unrecoverable_generations",
                       "final_order_differs_from_bm25")
    counts = Counter({key: 0 for key in count_keys})
    by_variant = {v: Counter({key: 0 for key in count_keys}) for v in index["variants"]}
    failure_rows = []
    observed_complete = set()
    for (vi, qi), record in unique.items():
        if not completed[vi, qi]:
            counts["diagnostic_without_completed_score_and_timing"] += 1
            continue
        observed_complete.add((vi, qi))
        variant = index["variants"][vi]
        current_counts = by_variant[variant]
        initial_order = np.argsort(-bm25_scores[vi, qi], kind="stable")
        shortlist = initial_order[:20]
        supplied = record.get("initial_candidate_indices")
        if (not isinstance(supplied, list) or len(supplied) != len(shortlist)
                or any(type(x) is not int for x in supplied) or supplied != shortlist.tolist()):
            raise ValueError(f"{method}: diagnostic shortlist differs from frozen BM25 ordering")
        if "initial_candidate_ids" in record and record["initial_candidate_ids"] != [
                index["candidate_ids"][i] for i in shortlist]:
            raise ValueError(f"{method}: diagnostic candidate identifiers mismatch")
        response = record.get("response")
        if not isinstance(response, str):
            raise ValueError(f"{method}: diagnostic response must be text")
        strict = strict_permutation(response, len(shortlist))
        valid = strict is not None
        if record.get("generation_valid") is not valid or record.get("parsed_permutation") != strict:
            raise ValueError(f"{method}: original generation validity contradicts saved response")
        expected_status = "valid_listwise_permutation" if valid else "invalid_generation_bm25_fallback"
        if record.get("status") != expected_status:
            raise ValueError(f"{method}: original diagnostic status contradicts strict validity")
        strict_order = (np.concatenate([shortlist[strict], initial_order[len(shortlist):]])
                        if valid else initial_order)
        if not np.array_equal(scores[vi, qi], _ordinal(strict_order)):
            raise ValueError(f"{method}: stored strict score contradicts original generation policy")
        parser = recover_reasonrank_order if method == REASONRANK else recover_rearank_order
        order, details = parser(response, len(shortlist))
        row_counts = {"completed_records": 1, "strict_complete_generations": int(valid),
                      "strict_invalid_generations": int(not valid),
                      "token_limit_hits": int(record.get("hit_token_limit") is True)}
        for name in details["classifications"]:
            row_counts[name] = 1
        if recover:
            if order is None:
                final_order, disposition = initial_order, "unrecoverable_bm25_fallback"
                row_counts["unrecoverable_generations"] = 1
            else:
                final_order = np.concatenate([shortlist[order], initial_order[len(shortlist):]])
                disposition = "strict_complete_unchanged" if valid else "official_postprocessing_repaired"
                row_counts["usable_order_generations"] = 1
                row_counts["postprocessing_repaired_generations"] = int(not valid)
                row_counts["repaired_and_changed_from_bm25"] = int(
                    not valid and not np.array_equal(final_order, initial_order))
            row_counts["final_order_differs_from_bm25"] = int(not np.array_equal(final_order, initial_order))
            destination[vi, qi] = _ordinal(final_order)
        else:
            disposition = "audit_only_original_strict_result_preserved"
        if not valid:
            failure_rows.append({"query_id": record["query_id"], "variant": variant,
                                 "generation_valid": False, "original_status": record["status"],
                                 "disposition": disposition, "hit_token_limit": record.get("hit_token_limit"),
                                 "generated_token_count": record.get("generated_token_count"),
                                 "answer_open_tag_present": "<answer>" in response,
                                 "answer_close_tag_present": "</answer>" in response,
                                 "think_close_tag_present": "</think>" in response,
                                 **details})
        counts.update(row_counts)
        current_counts.update(row_counts)
    missing_diagnostics = int(completed.sum()) - len(observed_complete)
    if status.get("state") == "complete" and (not completed.all() or missing_diagnostics or partial_tail
            or counts["diagnostic_without_completed_score_and_timing"]):
        raise ValueError(f"{method}: completed method lacks consistent complete arrays and diagnostics")
    summary = {"state": "audited", "method": method, "worker_state": status.get("state"),
               "expected_query_variants": int(np.prod(shape[:2])),
               "complete_score_and_timing_rows": int(completed.sum()),
               "completed_rows_awaiting_diagnostic": missing_diagnostics,
               "unique_diagnostic_records": len(unique), "identical_duplicate_records": duplicate_records,
               "partial_final_jsonl_line_skipped": partial_tail,
               "counts": dict(counts), "by_variant": {k: dict(v) for k, v in by_variant.items()},
               "strict_failure_records": failure_rows,
               "postprocessing_applied": recover,
               "original_scores_preserved": True}
    if recover:
        saved_complete = np.isfinite(destination).all(axis=-1)
        if (saved_complete & ~completed).any() or int(saved_complete.sum()) != counts["completed_records"]:
            raise ValueError("Recovered score completeness disagrees with original completed rows")
        summary["recovered_score_complete_rows"] = int(saved_complete.sum())
    return summary, destination


def _atomic(path, data):
    path = _path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _refresh_output_recovery(out, include_reasonrank):
    """Apply each model's independently verified policy without another call."""
    sources = {}
    input_paths = [out / "inputs" / name for name in ("queries.jsonl", "candidates.jsonl", "manifest.json")]
    input_bytes = [_read(path, out, sources) for path in input_paths]
    queries, _ = _rows(input_bytes[0])
    candidates, _ = _rows(input_bytes[1])
    manifest = json.loads(input_bytes[2])
    query_ids, candidate_ids = [x["query_id"] for x in queries], [x["candidate_id"] for x in candidates]
    if (not queries or len(candidates) < 20 or len(set(query_ids)) != len(queries)
            or len(set(candidate_ids)) != len(candidates)):
        raise ValueError("Frozen input identifiers must be unique and contain at least 20 candidates")
    for name, raw in zip(("queries.jsonl", "candidates.jsonl"), input_bytes):
        if manifest.get("outputs", {}).get(name, {}).get("sha256") != _sha(raw):
            raise ValueError("Frozen input manifest hash mismatch")
    input_hashes = {str(path.relative_to(ROOT)): _sha(raw) for path, raw in zip(input_paths, input_bytes)}
    index = {"query_ids": query_ids, "candidate_ids": candidate_ids, "variants": list(VARIANTS)}
    baseline = out / "results" / "bm25_text"
    baseline_status = _json(baseline / "status.json", out, sources)
    if baseline_status.get("state") != "complete" or baseline_status.get("input_sha256") != input_hashes:
        raise ValueError("Recovery requires completed BM25 scores for the same frozen inputs")
    if _json(baseline / "index.json", out, sources) != index:
        raise ValueError("BM25 query/candidate/variant index mismatch")
    bm25_scores = _array(baseline / "scores.npy", out, sources)
    expected_shape = (len(VARIANTS), len(queries), len(candidates))
    if bm25_scores.shape != expected_shape or not np.isfinite(bm25_scores).all():
        raise ValueError("BM25 must contain every finite candidate score for every request variant")
    rearank, recovered = _audit_method(out, REARANK, index, input_hashes, bm25_scores, sources, recover=True)
    reasonrank, reasonrank_recovered = _audit_method(
        out, REASONRANK, index, input_hashes, bm25_scores, sources, recover=include_reasonrank)
    budget_path = out / "results" / REASONRANK / "budget_repair_3172.json"
    budget_state = _json(budget_path, out, sources) if _path(budget_path).is_file() else None
    # Keep the v1 strict-audit view for consumers that already read this key.
    repair_keys = {"usable_order_generations", "postprocessing_repaired_generations",
                   "repaired_and_changed_from_bm25", "unrecoverable_generations",
                   "final_order_differs_from_bm25"}
    reasonrank_audit = {k: v for k, v in reasonrank.items() if k != "recovered_score_complete_rows"}
    reasonrank_audit["postprocessing_applied"] = False
    reasonrank_audit["counts"] = {k: v for k, v in reasonrank.get("counts", {}).items() if k not in repair_keys}
    reasonrank_audit["by_variant"] = {
        variant: {k: v for k, v in counts.items() if k not in repair_keys}
        for variant, counts in reasonrank.get("by_variant", {}).items()}
    reasonrank_audit["strict_failure_records"] = [
        {**row, "disposition": "audit_only_original_strict_result_preserved"}
        for row in reasonrank.get("strict_failure_records", [])]
    implementation_sha = _sha(_path(__file__).read_bytes())
    fingerprint = _sha(json.dumps({"sources": sources, "implementation": implementation_sha,
                                   "include_reasonrank": include_reasonrank}, sort_keys=True).encode())
    report_path = out / "results" / REARANK / "output_recovery.json"
    if _path(report_path).is_file():
        try:
            previous = json.loads(report_path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError):
            previous = {}
        expected_methods = {method for method, value in
                            ((REARANK, recovered), (REASONRANK, reasonrank_recovered)) if value is not None}
        outputs = previous.get("recovered_scores_by_method", {})
        if (previous.get("source_fingerprint") == fingerprint and set(outputs) == expected_methods
                and all(_path(out / data["path"]).is_file()
                        and _sha(_path(out / data["path"]).read_bytes()) == data["sha256"]
                        for data in outputs.values())):
            return previous
    pending_reasonrank = [
        {"query_id": row["query_id"], "variant": row["variant"],
         "original_generated_tokens": row["generated_token_count"]}
        for row in reasonrank.get("strict_failure_records", [])
        if row.get("hit_token_limit") is True and not row.get("answer_open_tag_present")]
    report = {"schema_version": "frontier-output-recovery-v1.1",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "official_source": {"url": OFFICIAL_SOURCE, "commit": OFFICIAL_COMMIT,
                                  "rank_gpt_py_sha256": OFFICIAL_SOURCE_SHA256},
              "reasonrank_official_source": {"url": REASONRANK_SOURCE, "commit": REASONRANK_COMMIT,
                                             "rankllm_py_sha256": REASONRANK_SOURCE_SHA256,
                                             "policy_independently_verified": True},
              "policy": "Preserve first valid occurrence, discard duplicate/out-of-range identifiers, append omitted shortlist candidates in frozen BM25 order; preserve all residual candidates in BM25 order.",
              "conservative_deviation": "Only parse a closed answer body without think tags; missing/unfinished/empty usable answers explicitly retain BM25. Never mine reasoning text for candidate numbers.",
              "scientific_claim_eligible": False, "strict_generation_success_redefined": False,
              "implementation_sha256": implementation_sha,
              "source_fingerprint": fingerprint,
              "score_semantics": "Two-stage ordinal rank, not calibrated probability",
              "snapshot_semantics": "Exact hashed byte snapshots; active incomplete rows remain NaN; refresh is safe to repeat.",
              "source_snapshots": sources, "rearank": rearank,
              "reasonrank": reasonrank, "reasonrank_audit_only": reasonrank_audit,
              "reasonrank_length_followup": {
                  "state": "official_budget_correction_ready_not_started",
                  "requests": pending_reasonrank, "request_count": len(pending_reasonrank),
                  "original_max_new_tokens": 3072,
                  "official_example_generation_budget": 3172,
                  "official_budget_explanation": "run_rank_llm.sh sets reasoning_maxlen=3072; run_llm_batched adds 100 for the final answer.",
                  "official_budget_source": f"https://github.com/8421BCD/ReasonRank/blob/{REASONRANK_COMMIT}/rerank/rank_listwise_os_llm.py#L107-L114",
                  "corrected_max_new_tokens": 3172,
                  "attempts_per_request": 1,
                  "preserve_prompt_model_weights_shortlist_and_greedy_decoding": True,
                  "replace_original_full_experiment_scores": False,
                  "select_best_of_multiple_generations": False,
                  "limitation": "Saved traces have no final answer; no output-only recovery is valid. The correction performs one fresh 3172-token generation per affected request, not an exact continuation or best-of selection. Original full-experiment results remain separate."}}
    if budget_state is not None:
        report["reasonrank_length_followup"].update({
            "state": budget_state.get("state"), "completed_count": budget_state.get("completed_count", 0),
            "pid": budget_state.get("pid"), "strict_success_count": budget_state.get("strict_success_count"),
            "usable_order_count": budget_state.get("usable_order_count"),
            "checkpoint": str(budget_path.relative_to(out)),
            "corrected_scores_file": budget_state.get("output_path")})
    outputs = {}
    for method, values in ((REARANK, recovered), (REASONRANK, reasonrank_recovered)):
        if values is None:
            continue
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        raw = buffer.getvalue()
        path = _path(out / "results" / method / "official_recovered_scores.npy")
        metadata = {"path": str(path.relative_to(out)), "sha256": _sha(raw),
                    "shape": list(values.shape), "dtype": str(values.dtype)}
        outputs[method] = metadata
        if not path.is_file() or _sha(path.read_bytes()) != metadata["sha256"]:
            _atomic(path, raw)
    report["recovered_scores_by_method"] = outputs
    if REARANK in outputs:
        report["recovered_scores"] = {**outputs[REARANK], "path": "official_recovered_scores.npy"}
        _atomic(report_path, (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    return report


def refresh_output_recovery(out=None, *, include_reasonrank=True):
    """Refresh safely even when independent workers request a report together."""
    out = _path(DEFAULT_OUT if out is None else out)
    # A directory flock serializes refreshes without leaving another lock file.
    descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _refresh_output_recovery(out, include_reasonrank)
    finally:
        os.close(descriptor)


def repair_reasonrank_budget():
    """Once per truncated request, correct 3072 to the author's 3072+100.

    Explicit CLI/function only: ordinary CPU report refresh never starts a GPU.
    A persisted in-flight attempt is never repeated after a crash; it becomes
    an explicit incomplete-attempt fallback before remaining cases continue.
    """
    from run_frontier_comparison import OUT, RUN, init, download, load_inputs

    init()
    out = _path(OUT)
    folder = out / "results" / REASONRANK
    # Serialize only this repair task. GPU ownership uses the runner's lock.
    repair_lock = os.open(_path(folder), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(repair_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(repair_lock)
        raise
    state_path = folder / "budget_repair_3172.json"
    model, gpu_lock = None, None

    def save(state):
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["completed_count"] = sum(r.get("state") in ("complete", "attempt_failed", "interrupted_attempt")
                                       for r in state.get("records", []))
        _atomic(state_path, (json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())

    try:
        audit = refresh_output_recovery(out)
        queries, candidates, hashes = load_inputs()
        q_index = {row["query_id"]: i for i, row in enumerate(queries)}
        v_index = {value: i for i, value in enumerate(VARIANTS)}
        diagnostic_bytes = _path(folder / "generation_diagnostics.jsonl").read_bytes()
        diagnostics, _ = _rows(diagnostic_bytes)
        selected = [row for row in diagnostics if row.get("hit_token_limit") is True
                    and not row.get("generation_valid") and "<answer>" not in row["response"]]
        keys = [(row["query_id"], row["variant"]) for row in selected]
        if len(selected) != 5 or len(set(keys)) != 5 or any(row.get("generated_token_count") != 3072 for row in selected):
            raise ValueError("Official-budget repair requires exactly the five audited 3072-token no-answer records")
        official_path = _path(folder / "official_recovered_scores.npy")
        official_bytes = official_path.read_bytes()
        initial_values = np.load(io.BytesIO(official_bytes), allow_pickle=False)
        bm25_values = np.load(_path(out / "results/bm25_text/scores.npy"), allow_pickle=False)
        expected_shape = (len(VARIANTS), len(queries), len(candidates))
        if initial_values.shape != expected_shape or not np.isfinite(initial_values).all():
            raise ValueError("Budget repair needs the full validated official-postprocessing score array")
        parameters = {"method": REASONRANK, "max_new_tokens": 3172, "original_max_new_tokens": 3072,
                      "batch_size": 1, "do_sample": False, "max_length": 2048,
                      "quantization": "NF4 double quantization with BF16 compute",
                      "shortlist": "same frozen BM25 Top20", "attempts_per_request": 1,
                      "fresh_regeneration_not_token_exact_continuation": True,
                      "best_of_selection": False, "original_scores_overwritten": False}
        fingerprint = _sha(json.dumps({"inputs": hashes, "diagnostics": _sha(diagnostic_bytes),
                                       "official_scores": _sha(official_bytes), "parameters": parameters,
                                       "official_commit": REASONRANK_COMMIT}, sort_keys=True).encode())
        state = json.loads(state_path.read_bytes()) if state_path.exists() else {
            "schema_version": "reasonrank-official-budget-correction-v1", "records": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint": fingerprint, "parameters": parameters,
            "original_diagnostics_sha256": _sha(diagnostic_bytes),
            "original_official_scores_sha256": _sha(official_bytes),
            "input_sha256": hashes, "scientific_claim_eligible": False,
            "official_source": audit["reasonrank_official_source"],
            "official_budget_source": audit["reasonrank_length_followup"]["official_budget_source"],
            "total_count": 5}
        if state.get("source_fingerprint") != fingerprint or state.get("parameters") != parameters:
            raise ValueError("Budget-repair source/configuration changed; refusing mixed results")
        stored = {(row["query_id"], row["variant"]): row for row in state["records"]}
        if len(stored) != len(state["records"]) or any(key not in keys for key in stored):
            raise ValueError("Budget-repair checkpoint contains duplicate or unknown cases")
        for source in selected:
            key = source["query_id"], source["variant"]
            if key in stored and stored[key].get("original_diagnostic_sha256") != _sha(
                    json.dumps(source, sort_keys=True, ensure_ascii=False).encode()):
                raise ValueError("Budget-repair case source hash mismatch")
            if key in stored and stored[key].get("state") == "complete":
                row = stored[key]
                strict = strict_permutation(row.get("response", ""), 20)
                order, _ = recover_reasonrank_order(row.get("response", ""), 20)
                if (row.get("generation_valid") is not (strict is not None)
                        or row.get("parsed_permutation") != strict
                        or row.get("recovered_permutation") != order):
                    raise ValueError("Budget-repair saved response contradicts its parsed result")
        output_path = folder / "budget_repaired_scores.npy"
        if state.get("state") == "complete" and output_path.is_file():
            if len(stored) != 5 or state.get("output_sha256") != _sha(output_path.read_bytes()):
                raise ValueError("Completed budget-repair output checksum or coverage mismatch")
            return state
        state.update(pid=os.getpid(), worker_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                     implementation_sha256=_sha(_path(__file__).read_bytes()), state="waiting_for_gpu")
        for row in state["records"]:
            if row.get("state") == "running":
                row.update(state="interrupted_attempt", response=None, generation_valid=False,
                           recovered_permutation=None, disposition="interrupted_attempt_bm25_fallback",
                           error_type="InterruptedAfterAttemptWasPersisted")
        save(state)
        pending = [source for source in selected if (source["query_id"], source["variant"]) not in stored]
        if pending:
            if (RUN / "stop.requested").exists():
                state["state"] = "stopped"; save(state); return state
            weights = download(REASONRANK)
            expected_revision = "3444046f1481991fd9f2021df231e6c9cb7fcef1"
            if weights["revision"] != expected_revision:
                raise ValueError("Budget correction must use the original pinned ReasonRank checkpoint")
            state["weights_revision"] = weights["revision"]
            gpu_lock = _path(RUN / "gpu.lock").open("a")
            waiting = time.monotonic()
            while True:
                if (RUN / "stop.requested").exists():
                    state["state"] = "stopped"; save(state); return state
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - waiting > 50 * 3600:
                        raise TimeoutError("Project GPU wait exceeded the bounded repair queue budget")
                    time.sleep(2)
            state.update(state="loading_gpu", gpu_queue_wait_seconds=time.monotonic() - waiting)
            save(state)
            from frontier_external_models import load_model
            model = load_model(REASONRANK, weights["path"], device="cuda:0", batch_size=1, max_length=2048)
            model.max_new_tokens = 3172
            documents = [row["text"] for row in candidates]
            model.prepare_documents(documents)
            for source in pending:
                if (RUN / "stop.requested").exists():
                    state["state"] = "stopped"; save(state); return state
                key = source["query_id"], source["variant"]
                query = queries[q_index[key[0]]]
                text = query["text"] if key[1] == "original" else query["variants"][key[1]]["text"]
                record = {"query_id": key[0], "variant": key[1], "state": "running", "attempt": 1,
                          "original_diagnostic_sha256": _sha(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()),
                          "initial_candidate_indices": source["initial_candidate_indices"],
                          "started_at": datetime.now(timezone.utc).isoformat()}
                state["records"].append(record); stored[key] = record
                state["state"] = "scoring"; save(state)
                started = time.perf_counter()
                try:
                    model.score_query(text, documents)
                    diagnostic = dict(model.last_query_diagnostics)
                    if diagnostic.get("initial_candidate_indices") != source["initial_candidate_indices"]:
                        raise ValueError("Budget-repair generation changed the frozen shortlist")
                    response = diagnostic["response"]
                    strict = strict_permutation(response, 20)
                    if diagnostic.get("generation_valid") is not (strict is not None):
                        raise ValueError("Budget-repair diagnostic contradicts strict parser")
                    recovered_order, details = recover_reasonrank_order(response, 20)
                    record.update(state="complete", seconds=time.perf_counter() - started,
                                  response=response, generation_valid=strict is not None,
                                  parsed_permutation=strict, recovered_permutation=recovered_order,
                                  generation_diagnostics=diagnostic, recovery_details=details,
                                  disposition=("unrecoverable_bm25_fallback" if recovered_order is None else
                                               "strict_complete" if strict is not None else "official_postprocessing_repaired"))
                    save(state)
                except BaseException as error:
                    record.update(state="attempt_failed", seconds=time.perf_counter() - started,
                                  response=None, generation_valid=False, recovered_permutation=None,
                                  disposition="attempt_failed_bm25_fallback", error_type=type(error).__name__)
                    state["state"] = "failed"; save(state)
                    raise
        if len(stored) != 5 or any(row.get("state") not in ("complete", "attempt_failed", "interrupted_attempt")
                                    for row in stored.values()):
            raise ValueError("Budget correction did not account for all five bounded attempts")
        revised = initial_values.copy()
        for (query_id, variant), record in stored.items():
            vi, qi = v_index[variant], q_index[query_id]
            initial_order = np.argsort(-bm25_values[vi, qi], kind="stable")
            order = record.get("recovered_permutation")
            if order is not None and (len(order) != 20 or set(order) != set(range(20))):
                raise ValueError("Budget-repair checkpoint has a malformed completed permutation")
            final_order = (np.concatenate([initial_order[:20][order], initial_order[20:]])
                           if order is not None else initial_order)
            revised[vi, qi] = _ordinal(final_order)
        if revised.shape != expected_shape or not np.isfinite(revised).all():
            raise ValueError("Budget-corrected scores do not cover the entire original experiment")
        buffer = io.BytesIO(); np.save(buffer, revised, allow_pickle=False); raw = buffer.getvalue()
        if not output_path.is_file() or _sha(output_path.read_bytes()) != _sha(raw):
            _atomic(output_path, raw)
        state.update(state="complete", output_sha256=_sha(raw), output_shape=list(revised.shape),
                     output_path="budget_repaired_scores.npy", completed_query_variants=int(np.prod(expected_shape[:2])),
                     corrected_request_count=5,
                     strict_success_count=sum(row.get("generation_valid") is True for row in stored.values()),
                     usable_order_count=sum(row.get("recovered_permutation") is not None for row in stored.values()))
        save(state)
        return state
    finally:
        if model is not None:
            model.close()
        if gpu_lock is not None:
            gpu_lock.close()
        os.close(repair_lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reasonrank-audit-only", action="store_true",
                        help="Skip ReasonRank derived scoring; preserve its original strict-audit view")
    parser.add_argument("--repair-reasonrank-budget", action="store_true",
                        help="Run exactly one official-3172-budget GPU generation for each of the five truncated cases")
    args = parser.parse_args()
    if args.repair_reasonrank_budget:
        if _path(args.out) != DEFAULT_OUT or args.reasonrank_audit_only:
            parser.error("Budget correction requires the current frozen full experiment")
        state = repair_reasonrank_budget()
        print(json.dumps({key: state.get(key) for key in
                          ("state", "pid", "total_count", "completed_count", "strict_success_count", "usable_order_count")}))
        raise SystemExit(0)
    result = refresh_output_recovery(args.out, include_reasonrank=not args.reasonrank_audit_only)
    print(json.dumps({"schema_version": result["schema_version"],
                      "rearank": {k: v for k, v in result["rearank"].items()
                                  if k not in ("strict_failure_records", "by_variant")},
                      "reasonrank": {k: v for k, v in result["reasonrank"].items()
                                     if k not in ("strict_failure_records", "by_variant")}}, indent=2))
