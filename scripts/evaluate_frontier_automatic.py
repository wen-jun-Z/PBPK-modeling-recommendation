"""Evaluate frozen frontier scores against explicitly automatic source judgments.

This is a separate automatic-document-support benchmark, never an expert or
independent-execution benchmark. It does not relax scientific_evaluation.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "experiments/model_recommendation/frontier_comparison_20260914"
VERSION = "pbpk-automatic-source-evaluation-1.0"
KS = (1, 3, 5, 10)
VARIANTS = ("original", "reordered", "missing_evidence")
SEED = "pbpk-automatic-calibration-compound-split-20260917-v1"
STATUSES = {"judged", "insufficient_evidence", "source_conflict", "invalid_output"}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = "pbpk-compound-cluster-bootstrap-20260917-v1"
BOOTSTRAP_METRICS = ("ndcg@10", "recall@10", "mrr")


def scoped(path: Path) -> Path:
    value = path.resolve()
    if not value.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path outside project: {path}")
    return value


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> None:
    scoped(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in scoped(path).read_text().splitlines() if line.strip()]


def build_protocol(queries: list[dict], candidates: list[dict], input_hashes: dict) -> dict:
    compounds = sorted({q["compound_id"] for q in queries},
                       key=lambda key: sha((SEED + key).encode()))
    validation_count = max(1, min(len(compounds) - 1, math.ceil(len(compounds) * .3)))
    validation = set(compounds[:validation_count])
    return {
        "schema_version": VERSION, "input_sha256": input_hashes,
        "query_count": len(queries), "candidate_count": len(candidates),
        "expected_judgment_count": len(queries) * len(candidates),
        "label_source": "llm_source_grounded", "positive_grade_threshold": 2,
        "k": list(KS), "gain": "2**grade-1",
        "tie_policy": "Expected DCG, relevant counts and reciprocal rank under uniform permutations within equal-score blocks",
        "unknown_policy": "null stays unknown; full-pool metrics require all grades and at least one positive",
        "judged_only_policy": "Filter unknown candidates, rerank remaining pool; NOT full-catalog Recall/NDCG",
        "split_seed": SEED, "split_group": "primary compound_id; no pair-level random split",
        "compound_split": {key: "validation" if key in validation else "test" for key in compounds},
        "query_split": {q["query_id"]: "validation" if q["compound_id"] in validation else "test" for q in queries},
        "split_scope": "Calibration holdout only, not proof of novel-drug generalization; shared catalog and judge remain",
        "calibration": {"fit_split": "validation", "report_split": "test",
                        "feature": "one original raw method score, standardized using validation only",
                        "model": "L2 logistic regression, summed log loss plus 0.5*slope**2",
                        "threshold": .5, "ece_bins": 10,
                        "fit_requires": "all pairs processed, both target classes and at least two compounds in each split",
                        "variants": "Fit original once; apply the same parameters to every variant"},
        "variant_labels": "Original intended task retained: sentence permutation and removal of mechanism support only",
        "scientific_claim_eligible": False, "human_agreement_measured": False,
        "source_content_independence_verified": False,
    }


def freeze_protocol(run_dir: Path, assessment: Path, queries: list[dict], candidates: list[dict]) -> dict:
    hashes = {name: sha((run_dir / "inputs" / name).read_bytes()) for name in ("queries.jsonl", "candidates.jsonl")}
    expected = build_protocol(queries, candidates, hashes)
    path = assessment / "evaluation_protocol.json"
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise ValueError("Frozen evaluation protocol differs; do not silently change the split, input or metric policy")
    else:
        assessment.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents parallel reporters from replacing a protocol.
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(expected, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except FileExistsError:
            if json.loads(path.read_text()) != expected:
                raise ValueError("Concurrent incompatible evaluation protocol")
    return expected


def freeze_reporting_additions(assessment: Path, protocol: dict) -> dict:
    """Append a fixed reporting specification without rewriting either protocol."""
    expected = {
        "version": "request-decision-and-matched-perturbation-v1",
        "base_protocol_sha256": sha(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()),
        "reason": "Complete predeclared abstention and perturbation reporting; preserve original metrics and avoid no-op or unknown-label denominator errors.",
        "primary_original_metrics_unchanged": True,
        "request_decisions": {
            "split": "test only; existing validation-fitted original-score calibrator unchanged",
            "selection": "original raw-score Top-1; uniform expectation over exact score ties",
            "recommend_threshold": .5,
            "no_applicable": "all frozen candidates known N; U is never N",
            "applicable_exists": "at least one known Y, regardless of other unknowns",
            "false_recommendation_denominator": "recommended queries whose entire Top-1 tie set has known applicability",
            "wrong_rejection_denominator": "queries with at least one known Y",
            "correct_rejection_denominator": "queries with all candidates known N",
            "risk_curve": "test Top-1 confidence coverage deciles, include all threshold ties; report risk only for known Top-1 sets alongside all-query and label coverage",
            "native_abstention_or_clinical_safety_claim": False,
        },
        "perturbations": {
            "all_query_results_retained": True,
            "changed_subset": "frozen original text != frozen variant text",
            "matched_subset": "changed text and identical recorded query_batch_size; pre-journal rows were serial",
            "missing_evidence_interpretation": "support sensitivity; not proof that missing evidence should never change a ranking",
        },
        "shared_documentation": "identical supplied source packet pairs excluded from version preference, even if automatic grades disagree",
    }
    path = assessment / "evaluation_protocol_additions.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "recorded_at"} != expected:
            raise ValueError("Frozen additional reporting specification changed")
        return saved
    saved = {**expected, "recorded_at": datetime.now(timezone.utc).isoformat()}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(saved, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        return freeze_reporting_additions(assessment, protocol)
    return saved


def freeze_bounds_revision(assessment: Path, protocol: dict) -> dict:
    """Freeze a further additive analysis, without modifying earlier protocols."""
    expected = {
        "version": "full-catalog-partial-judgment-bounds-v1",
        "base_protocol_sha256": sha(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()),
        "previous_additions_file_sha256": sha((assessment / "evaluation_protocol_additions.json").read_bytes()),
        "reason": "Report full-catalog uncertainty despite missing documentation; unknown grades remain null in labels and point metrics.",
        "eligibility": "At least one known grade >= 2 guarantees metric definition for every unknown completion. Otherwise report conditional coarse [0,1] only when unknowns exist and do not include the query in eligible means.",
        "unknown_completion_domain": [0, 1, 2, 3],
        "tie_policy": "Same uniform-permutation expected ranking metric as the original evaluator",
        "ndcg": {"lower": "known-gain expected DCG / IDCG with every unknown grade 3",
                 "upper": "min(1, expected DCG with every unknown grade 3 / IDCG with every unknown grade 0)",
                 "tight": False},
        "recall": {"lower": "known-positive expected Top-k count / (known positive count + unknown count)",
                   "upper": "min(1, (known-positive expected Top-k count + unknown expected Top-k count) / known positive count)",
                   "tight": False},
        "mrr": {"lower": "unknowns hypothetically all nonrelevant", "upper": "unknowns hypothetically all relevant", "tight": True},
        "summary": "Query-macro mean of lower and upper bounds over the same guaranteed-positive queries; always report this eligibility and coverage",
        "hypothetical_completion_is_not_imputed_ground_truth": True,
        "original_point_metrics_and_judgments_unchanged": True,
    }
    path = assessment / "evaluation_protocol_bounds.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "recorded_at"} != expected:
            raise ValueError("Frozen full-catalog bounds specification changed")
        return saved
    saved = {**expected, "recorded_at": datetime.now(timezone.utc).isoformat()}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(saved, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        return freeze_bounds_revision(assessment, protocol)
    return saved


def freeze_statistics_revision(assessment: Path, protocol: dict, queries: list[dict]) -> dict:
    expected = {
        "version": "compound-cluster-bootstrap-v1",
        "base_protocol_sha256": sha(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()),
        "previous_bounds_file_sha256": sha((assessment / "evaluation_protocol_bounds.json").read_bytes()),
        "reason": "Requests sharing a compound are correlated; quantify sampling uncertainty without treating all requests as independent samples.",
        "variant": "original", "metrics": list(BOOTSTRAP_METRICS),
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "random_generator": "NumPy PCG64; integer seed from first 16 hex digits of SHA256(seed string)",
        "resampling_clusters": sorted({q["compound_id"] for q in queries}),
        "resampling": "Sample C compounds with replacement from all C frozen compounds on each replicate; reuse the same draws across metrics and methods. Whole compounds move together.",
        "aggregation": "Keep the original query-macro estimand: sum of eligible query metrics divided by eligible query count; bootstrap uses compound sums/counts, not equal-compound means.",
        "eligibility": "All candidate-request pairs processed, complete method matrices, at least two compounds with eligible metric queries. Explicit unknowns remain unknown.",
        "empty_replicates": "Omit replicates containing zero eligible queries and disclose valid/empty counts; no CI when fewer than two eligible compounds.",
        "confidence_interval": "Marginal 2.5th/97.5th percentile interval; np.quantile method='linear'. No bias correction or parameter selection.",
        "paired_reference": "bm25_text on exactly the common eligible original requests; bootstrap within-query method-minus-BM25 differences using the same compound draws",
        "bounds_endpoints": "Lower and upper full-catalog bounds each get a separate marginal sampling interval on guaranteed-positive queries; not a joint 95% confidence region or point-identification claim.",
        "limitations": "Conditional on fixed automatic judgments and available compounds; excludes judge error and shared-source bias, does not prove novel-drug generalization. No uncorrected multiple-comparison significance claim.",
        "original_point_metrics_unchanged": True,
    }
    path = assessment / "evaluation_protocol_statistics.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "recorded_at"} != expected:
            raise ValueError("Frozen compound-bootstrap specification changed")
        return saved
    saved = {**expected, "recorded_at": datetime.now(timezone.utc).isoformat()}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(saved, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        return freeze_statistics_revision(assessment, protocol, queries)
    return saved


def assessment_input_hash(judge: dict, assessment: Path, relative: str) -> str | None:
    """Accept the historic logical namespace or this explicitly named judge run.

    Moving a judge into its own directory must not force old protocols to change.
    If both names are declared, contradictory provenance is an error.
    """
    hashes = judge.get("input_sha256", {})
    keys = {"assessment/" + relative, assessment.name + "/" + relative}
    values = [hashes[key] for key in keys if key in hashes]
    if any(not isinstance(value, str) for value in values) or len(set(values)) > 1:
        raise ValueError("Conflicting or invalid assessment source hash namespaces")
    return values[0] if values else None


def read_judge_metadata(assessment: Path, protocol: dict) -> dict:
    """Copy only reproducibility fields; never model paths or arbitrary settings."""
    path = assessment / "judge_protocol.json"
    if not path.exists():
        return {"state": "judge_protocol_not_frozen"}
    raw = scoped(path).read_bytes()
    judge = json.loads(raw)
    for name, expected in protocol["input_sha256"].items():
        if judge.get("input_sha256", {}).get("inputs/" + name) != expected:
            raise ValueError("Judge protocol and evaluation use different frozen inputs")
    source_hash = assessment_input_hash(judge, assessment, "sources.jsonl")
    sources_path = assessment / "sources.jsonl"
    if not isinstance(source_hash, str) or not sources_path.exists() or sha(scoped(sources_path).read_bytes()) != source_hash:
        raise ValueError("Frozen assessment sources missing or changed")
    allowed = ("version", "primary_repo", "primary_revision", "secondary_repo", "secondary_revision", "system_prompt_sha256",
               "generation", "quantization", "packet_characters", "packet_segments", "requests_per_prompt",
               "quote_validation", "cache_policy", "secondary_selection", "source_independence_verified",
               "primary_backend", "secondary_backend")
    result = {"state": "frozen", "judge_protocol_sha256": sha(raw),
              "sources_sha256": source_hash, **{key: judge[key] for key in allowed if key in judge}}
    weights_path = assessment / "primary_weights.json"
    if weights_path.exists():
        raw_weights = scoped(weights_path).read_bytes()
        weights = json.loads(raw_weights)
        result["primary_weights"] = {"manifest_sha256": sha(raw_weights),
                                     **{key: weights[key] for key in ("repo", "revision", "sha256", "trust_remote_code") if key in weights}}
        result["weight_verification"] = "Hashes recorded by assessment runner; evaluator does not reload model weights"
    return result


def read_judgments(path: Path, queries: list[dict], candidates: list[dict], *, protocol_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    qi = {q["query_id"]: i for i, q in enumerate(queries)}
    ci = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    grades = np.full((len(qi), len(ci)), -1, dtype=np.int8)
    applicable = np.full_like(grades, -1)
    seen, statuses, sources = set(), Counter(), set()
    input_evidence = {eid for q in queries for eid in q.get("input_evidence_ids", [])}
    # Read one byte snapshot: a partially appended final line is never a commit.
    raw = scoped(path).read_bytes() if path.exists() else b""
    boundary = raw.rfind(b"\n") + 1
    committed, tail = raw[:boundary], raw[boundary:]
    for number, line in enumerate(committed.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pair = (row.get("query_id"), row.get("candidate_id"))
        if pair[0] not in qi or pair[1] not in ci or pair in seen:
            raise ValueError(f"Extra or duplicate judgment at line {number}: {pair}")
        if row.get("label_source") != "llm_source_grounded":
            raise ValueError(f"Automatic evaluator requires llm_source_grounded at line {number}")
        if protocol_sha256 is not None and (row.get("protocol_sha256") != protocol_sha256 or row.get("judge") != "primary"):
            raise ValueError(f"Judgment must come from the frozen primary judge at line {number}")
        state = row.get("status")
        if state not in STATUSES:
            raise ValueError(f"Unknown assessment status at line {number}")
        if "grade" not in row or "documented_applicability" not in row:
            raise ValueError(f"Explicit null or judgment required at line {number}")
        grade, app = row["grade"], row["documented_applicability"]
        if grade is not None and (type(grade) is not int or grade not in range(4)):
            raise ValueError(f"Invalid grade at line {number}")
        if app is not None and type(app) is not bool:
            raise ValueError(f"Invalid documented_applicability at line {number}")
        if state != "judged" and (grade is not None or app is not None):
            raise ValueError(f"Unjudged status cannot contain known labels at line {number}")
        if state == "judged" and grade is None and app is None:
            raise ValueError(f"Judged status needs at least one known target at line {number}")
        support = row.get("support")
        if not isinstance(support, list):
            raise ValueError(f"Support list required at line {number}")
        if grade is not None or app is not None:
            if not support:
                raise ValueError(f"Known automatic judgment requires source support at line {number}")
            for item in support:
                if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip() for k in ("source_id", "quote")):
                    raise ValueError(f"Source id and exact quote required at line {number}")
                sources.add(item["source_id"])
        seen.add(pair)
        statuses[state] += 1
        grades[qi[pair[0]], ci[pair[1]]] = -1 if grade is None else grade
        applicable[qi[pair[0]], ci[pair[1]]] = -1 if app is None else int(app)
    expected = grades.size
    return grades, applicable, {
        "state": "complete" if len(seen) == expected and not tail else "partial",
        "expected_pairs": expected, "processed_unique_pairs": len(seen),
        "processed_fraction": len(seen) / expected, "missing_pairs": expected - len(seen),
        "grade_known_pairs": int((grades >= 0).sum()),
        "applicability_known_pairs": int((applicable >= 0).sum()),
        "fully_graded_queries": int(np.all(grades >= 0, axis=1).sum()),
        "fully_applicability_judged_queries": int(np.all(applicable >= 0, axis=1).sum()),
        "status_counts": dict(statuses), "committed_judgments_sha256": sha(committed),
        "uncommitted_tail_bytes": len(tail), "distinct_cited_source_ids": len(sources),
        "literal_input_evidence_source_id_overlap_count": len(sources & input_evidence),
        "source_content_overlap": "not independently excluded; different ids do not establish independence",
        "source_quote_validation": "Performed by the upstream assessment pipeline; this evaluator checks schema, not entailment",
    }


def rank_metrics(scores: np.ndarray, grades: np.ndarray) -> dict | None:
    """Tie-averaged metrics; caller must select a fully known candidate pool."""
    if not len(grades) or np.any(grades < 0) or not np.any(grades >= 2):
        return None
    order = np.argsort(-scores, kind="stable")
    s, g = scores[order], grades[order].astype(int)
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], len(g)]
    gain, positive = np.zeros(len(g)), np.zeros(len(g))
    reciprocal = None
    for start, end in zip(starts, ends):
        block = g[start:end]
        gain[start:end] = np.mean(2.0**block - 1)
        positives = int((block >= 2).sum())
        positive[start:end] = positives / len(block)
        if reciprocal is None and positives:
            # Probability that the first positive in this block occurs at j.
            negative_prefix, reciprocal = 1.0, 0.0
            for j in range(len(block) - positives + 1):
                reciprocal += negative_prefix * positives / (len(block) - j) / (int(start) + j + 1)
                negative_prefix *= (len(block) - positives - j) / (len(block) - j)
    discounts = 1 / np.log2(np.arange(len(g)) + 2)
    ideal_gain = 2.0**np.sort(g)[::-1] - 1
    result = {"mrr": float(reciprocal), "top1_expected_grade": float(np.mean(g[:ends[0]])),
              "top1_tie_size": int(ends[0])}
    for k in KS:
        stop = min(k, len(g))
        result[f"ndcg@{k}"] = float(np.sum(gain[:stop] * discounts[:stop]) / np.sum(ideal_gain[:stop] * discounts[:stop]))
        result[f"recall@{k}"] = float(np.sum(positive[:stop]) / np.sum(g >= 2))
    return result


def average_metrics(values: list[dict | None]) -> dict:
    known = [value for value in values if value is not None]
    keys = ["mrr", "top1_expected_grade", *[f"{metric}@{k}" for metric in ("ndcg", "recall") for k in KS]]
    return {"eligible_queries": len(known), "metrics": {
        key: float(np.mean([value[key] for value in known])) if known else None for key in keys}}


def full_catalog_bounds(scores: np.ndarray, grades: np.ndarray) -> dict:
    """Bounds over hypothetical unknown labels, never stored as actual labels.

    DCG is linear in nonnegative gains, including tie expectations. Replacing
    unknown gains by 0 or 7 bounds its numerator and ideal denominator. Thus
    Nmin/Dmax <= N/D <= Nmax/Dmin. For recall, known positives P>0 imply
    A/(P+U) <= (A+B)/(P+V) <= (A+W)/P, with 0<=B<=W, 0<=V<=U.
    MRR is monotone under adding relevant items; all-negative/all-positive
    unknown completions attain its exact extrema, including score ties.
    """
    scores, grades = np.asarray(scores), np.asarray(grades)
    if scores.shape != grades.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Bounds require aligned finite score vectors")
    if not np.all(np.isin(grades, [-1, 0, 1, 2, 3])):
        raise ValueError("Bounds require frozen grades or explicit unknown -1")
    positive, unknown = grades >= 2, grades < 0
    p, u = int(positive.sum()), int(unknown.sum())
    names = ["mrr", *[f"{metric}@{k}" for metric in ("ndcg", "recall") for k in KS]]
    base = {"candidate_count": len(grades), "known_positive_candidates": p,
            "unknown_candidates": u, "guaranteed_defined": bool(p),
            "eligible_for_mean_bounds": bool(p),
            "ndcg_and_recall_bounds_are_conservative_not_necessarily_tight": True,
            "mrr_bounds_are_exact_when_guaranteed_defined": True}
    if not p:
        return {**base, "state": "conditional_coarse_bounds" if u else "undefined_no_relevant_candidates",
                "condition": "At least one unknown must truly be relevant; metric is undefined if none is relevant" if u else "No grade >= 2 in the complete pool",
                "metrics": {name: {"lower": 0., "upper": 1.} if u else {"lower": None, "upper": None} for name in names}}
    # These are mathematical endpoint completions only; grades is never mutated.
    low = np.where(unknown, 0, grades).astype(int)
    high = np.where(unknown, 3, grades).astype(int)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(grades)]
    def expected(values):
        ordered = np.asarray(values, dtype=float)[order].copy()
        for start, end in zip(starts, ends):
            ordered[start:end] = ordered[start:end].mean()
        return ordered
    discount = 1. / np.log2(np.arange(len(grades)) + 2)
    low_gain, high_gain = 2.**low - 1, 2.**high - 1
    low_dcg, high_dcg = np.cumsum(expected(low_gain) * discount), np.cumsum(expected(high_gain) * discount)
    low_idcg, high_idcg = np.cumsum(np.sort(low_gain)[::-1] * discount), np.cumsum(np.sort(high_gain)[::-1] * discount)
    known_top, unknown_top = np.cumsum(expected(positive)), np.cumsum(expected(unknown))
    metrics = {"mrr": {"lower": rank_metrics(scores, low)["mrr"], "upper": rank_metrics(scores, high)["mrr"]}}
    for k in KS:
        stop = min(k, len(grades)) - 1
        metrics[f"ndcg@{k}"] = {"lower": float(low_dcg[stop] / high_idcg[stop]),
                                "upper": float(min(1., high_dcg[stop] / low_idcg[stop]))}
        metrics[f"recall@{k}"] = {"lower": float(known_top[stop] / (p + u)),
                                  "upper": float(min(1., (known_top[stop] + unknown_top[stop]) / p))}
    return {**base, "state": "bounded", "condition": None, "metrics": metrics}


def summarize_catalog_bounds(values: list[dict]) -> dict:
    eligible = [value for value in values if value["eligible_for_mean_bounds"]]
    names = ["mrr", *[f"{metric}@{k}" for metric in ("ndcg", "recall") for k in KS]]
    return {"eligible_queries": len(eligible), "total_queries": len(values),
            "conditional_only_queries_excluded": sum(v["state"] == "conditional_coarse_bounds" for v in values),
            "complete_pool_without_positive_queries": sum(v["state"] == "undefined_no_relevant_candidates" for v in values),
            "metrics": {name: {side: float(np.mean([value["metrics"][name][side] for value in eligible])) if eligible else None
                                for side in ("lower", "upper")} for name in names},
            "scope": "All frozen candidates retained. Query-macro bounds over guaranteed-positive queries only; missing grades remain unknown, not zero. NDCG/Recall intervals may be loose."}


def pair_preference(scores: np.ndarray, grades: np.ndarray, candidates: list[dict], indices: np.ndarray, same_family: bool,
                    material_keys: dict[str, str] | None = None) -> dict:
    eligible, correct, indistinguishable = 0, 0.0, 0
    for offset, left in enumerate(indices):
        for right in indices[offset + 1:]:
            family_left, family_right = candidates[left].get("family_id"), candidates[right].get("family_id")
            if not family_left or not family_right or (family_left == family_right) != same_family:
                continue
            left_key = (material_keys or {}).get(candidates[left]["candidate_id"])
            right_key = (material_keys or {}).get(candidates[right]["candidate_id"])
            if left_key is not None and left_key == right_key:
                indistinguishable += 1
                continue
            if grades[left] < 0 or grades[right] < 0 or grades[left] == grades[right]:
                continue
            eligible += 1
            correct += .5 if scores[left] == scores[right] else float((scores[left] > scores[right]) == (grades[left] > grades[right]))
    return {"pairs": eligible, "correct_credit": correct, "accuracy": correct / eligible if eligible else None,
            "source_indistinguishable_pairs_excluded": indistinguishable}


def query_results(scores: np.ndarray, grades: np.ndarray, queries: list[dict], candidates: list[dict],
                  material_keys: dict[str, str] | None = None) -> list[dict]:
    result = []
    for i, query in enumerate(queries):
        s, g = scores[i], grades[i]
        known = g >= 0
        same = np.array([bool(set(query.get("compounds", [])) & set(c.get("compounds", []))) for c in candidates])
        order = np.argsort(-s, kind="stable")
        coverage = {}
        for k in KS:
            boundary = s[order[min(k, len(s)) - 1]]
            top = s >= boundary
            coverage[f"inclusive_judged@{k}"] = float(np.mean(known[top]))
            coverage[f"inclusive_count@{k}"] = int(top.sum())
        result.append({
            "query_id": query["query_id"], "compound_id": query["compound_id"], "task_type": query["task_type"],
            "known_grades": int(known.sum()), "candidate_count": len(g),
            "full_pool": rank_metrics(s, g), "judged_only": rank_metrics(s[known], g[known]),
            "full_catalog_bounds": full_catalog_bounds(s, g),
            "same_drug_full_pool": rank_metrics(s[same], g[same]),
            "different_drug_full_pool": rank_metrics(s[~same], g[~same]),
            "same_family_version_preference": pair_preference(s, g, candidates, np.flatnonzero(same), True, material_keys),
            "different_registered_family_preference": pair_preference(s, g, candidates, np.flatnonzero(same), False, material_keys),
            "topk_label_coverage": coverage,
        })
    return result


def summarize_queries(per_query: list[dict]) -> dict:
    out = {key: average_metrics([row[key] for row in per_query]) for key in (
        "full_pool", "judged_only", "same_drug_full_pool", "different_drug_full_pool")}
    out["query_count"] = len(per_query)
    out["full_catalog_bounds"] = summarize_catalog_bounds([row["full_catalog_bounds"] for row in per_query])
    out["full_pool"]["queries_with_unknown_grades"] = sum(row["known_grades"] != row["candidate_count"] for row in per_query)
    out["full_pool"]["complete_queries_without_positive"] = sum(row["known_grades"] == row["candidate_count"] and row["full_pool"] is None for row in per_query)
    out["topk_label_coverage"] = {f"inclusive_judged@{k}": float(np.mean([row["topk_label_coverage"][f"inclusive_judged@{k}"] for row in per_query])) if per_query else None for k in KS}
    for key in ("same_family_version_preference", "different_registered_family_preference"):
        known = [row[key] for row in per_query if row[key]["pairs"]]
        pairs = sum(row["pairs"] for row in known)
        out[key] = {"eligible_queries": len(known), "pairs": pairs,
                    "query_macro_accuracy": float(np.mean([row["accuracy"] for row in known])) if known else None,
                    "pair_micro_accuracy": sum(row["correct_credit"] for row in known) / pairs if pairs else None,
                    "independence_verified": False,
                    "source_indistinguishable_pairs_excluded": sum(row[key].get("source_indistinguishable_pairs_excluded", 0) for row in per_query)}
    return out


def execution_batch_modes(run_dir: Path, method: str, queries: list[dict]) -> np.ndarray:
    """Pre-journal rows were serial; committed batch records override that mode."""
    modes = np.ones((len(VARIANTS), len(queries)), dtype=int)
    path = run_dir / "results" / method / "execution_batches.jsonl"
    lookup = {q["query_id"]: i for i, q in enumerate(queries)}
    if path.exists():
        for row in rows(path):
            size = row.get("query_batch_size")
            if type(size) is not int or size < 1 or row.get("variant") not in VARIANTS:
                raise ValueError("Invalid recorded execution batch")
            for query_id in row["query_ids"]:
                if query_id not in lookup:
                    raise ValueError("Execution batch query outside frozen input")
                modes[VARIANTS.index(row["variant"]), lookup[query_id]] = size
    return modes


def perturbation_subsets(per_query: list[dict], original: list[dict], queries: list[dict],
                         variant: str, modes: np.ndarray) -> dict:
    """Do not let unchanged inputs or changed generation batches hide sensitivity."""
    if variant == "original":
        return {"state": "not_applicable", "reason": "original_reference_input"}
    if any("text" not in q or variant not in q.get("variants", {}) for q in queries):
        return {"state": "unavailable", "reason": "frozen_variant_texts_not_available"}
    vi = VARIANTS.index(variant)
    changed = [i for i, q in enumerate(queries) if q["text"] != q["variants"][variant]["text"]]
    matched = [i for i in changed if modes[0, i] == modes[vi, i]]
    return {"state": "evaluated", "changed_input_queries": len(changed),
            "unchanged_input_queries_excluded": len(queries) - len(changed),
            "changed_text_only": summarize_queries([per_query[i] for i in changed]),
            "matched_execution_batch_pairs": len(matched),
            "cross_execution_batch_pairs_excluded": len(changed) - len(matched),
            "matched_execution_batch_original": summarize_queries([original[i] for i in matched]),
            "matched_execution_batch_variant": summarize_queries([per_query[i] for i in matched]),
            "interpretation": "Only changed-text pairs with identical query batch size support paired perturbation comparisons; numerical decoding variation and shared hardware remain. Missing support is sensitivity, not a required invariance."}


def document_equivalence(assessment: Path) -> tuple[dict[str, str], dict]:
    path = assessment / "sources.jsonl"
    if not path.exists():
        return {}, {"state": "unavailable", "reason": "source_packets_not_available"}
    from run_frontier_assessment import build_packet
    source_rows = rows(path)
    sources = {source["candidate_id"]: source for source in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("Duplicate candidate in assessment sources")
    supplement_path = assessment / "source_supplements/sources.jsonl"
    protocol_path = assessment / "judge_protocol.json"
    supplement = {"state": "absent", "applied_candidate_ids": []}
    expected = None
    if protocol_path.exists():
        expected = assessment_input_hash(json.loads(scoped(protocol_path).read_bytes()), assessment,
                                         "source_supplements/sources.jsonl")
    if expected is not None and not supplement_path.exists():
        raise ValueError("Frozen assessment source supplement missing")
    if supplement_path.exists():
        raw = scoped(supplement_path).read_bytes()
        actual = sha(raw)
        # A new file cannot alter a previous judge run whose protocol did not use it.
        if protocol_path.exists() and expected is None:
            supplement = {"state": "not_declared_by_frozen_judge_protocol", "sha256": actual,
                          "applied_candidate_ids": []}
        else:
            if expected is not None and actual != expected:
                raise ValueError("Frozen assessment source supplement changed")
            seen = set()
            for source in rows(supplement_path):
                candidate_id = source.get("candidate_id")
                if candidate_id in seen or candidate_id not in sources:
                    raise ValueError("Unknown or duplicate candidate in assessment source supplement")
                seen.add(candidate_id)
                original = sources[candidate_id]
                if original.get("documents") or not source.get("documents"):
                    raise ValueError("Assessment supplement may only fill an empty source document set")
                if (source.get("family_id") != original.get("family_id")
                        or source.get("family_name") != original.get("family_name")
                        or source.get("compounds") != original.get("compounds")
                        or source.get("source_artifact") != original.get("source_artifact")
                        or source.get("source_artifact", {}).get("verified") is not True):
                    raise ValueError("Assessment supplement changes frozen candidate identity or artifact")
                sources[candidate_id] = source
            supplement = {"state": "applied", "sha256": actual,
                          "applied_candidate_ids": sorted(seen),
                          "protocol_hash_verified": expected is not None}
    keys, groups = {}, defaultdict(list)
    for source in sources.values():
        key = build_packet(source)["packet_sha256"]
        keys[source["candidate_id"]] = key
        groups[key].append(source["candidate_id"])
    shared = [values for values in groups.values() if len(values) > 1]
    return keys, {"state": "evaluated", "distinct_packets": len(groups),
                  "shared_packet_groups": len(shared), "candidates_in_shared_packets": sum(map(len, shared)),
                  "groups": shared,
                  "source_supplement": supplement,
                  "interpretation": "Identical supplied documentation cannot establish a preference between versions. These pairs are excluded even if automatic grades disagree; other version comparisons still describe document support, not predictive superiority."}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit_calibrator(scores: np.ndarray, applicability: np.ndarray, queries: list[dict], protocol: dict, judgments_complete: bool) -> dict:
    mask = np.array([protocol["query_split"][q["query_id"]] == "validation" for q in queries])
    valid = mask[:, None] & (applicability >= 0)
    target, raw = applicability[valid].astype(float), scores[valid].astype(float)
    groups = {q["compound_id"] for i, q in enumerate(queries) if mask[i] and np.any(applicability[i] >= 0)}
    base = {"state": "unavailable", "target": "documented_applicability from automatic source judgments",
            "fit_pairs": len(target), "fit_compounds": sorted(groups), "positive_fit_pairs": int(target.sum()),
            "threshold": .5, "fit_variant": "original", "fit_split": "validation"}
    if not judgments_complete:
        return {**base, "reason": "assessment_not_complete; do not repeatedly fit evolving labels"}
    if len(groups) < 2 or len(np.unique(target)) != 2:
        return {**base, "reason": "validation_requires_two_classes_and_two_compounds"}
    mean, scale = float(raw.mean()), float(raw.std())
    scale = scale if scale > 1e-12 else 1.0
    design = np.column_stack(((raw - mean) / scale, np.ones(len(raw))))
    beta = np.array([0., math.log(target.mean() / (1 - target.mean()))])
    penalty = np.diag([1., 0.])
    def loss(b):
        z = design @ b
        return np.logaddexp(0, z).sum() - target @ z + .5 * b[0] ** 2
    converged = False
    for iteration in range(100):
        p = sigmoid(design @ beta)
        grad = design.T @ (p - target) + penalty @ beta
        hess = design.T @ ((p * (1 - p))[:, None] * design) + penalty + np.eye(2) * 1e-10
        step = np.linalg.solve(hess, grad)
        rate, before = 1., loss(beta)
        while rate > 2**-20 and loss(beta - rate * step) > before:
            rate *= .5
        beta -= rate * step
        if np.max(np.abs(rate * step)) < 1e-8 or np.max(np.abs(grad)) < 1e-7:
            converged = True
            break
    return {**base, "state": "fitted" if converged else "unavailable",
            "reason": None if converged else "logistic_solver_did_not_converge",
            "mean": mean, "scale": scale, "slope": float(beta[0]), "intercept": float(beta[1]),
            "iterations": iteration + 1, "l2_slope_penalty": 1.0}


def calibrated_metrics(scores: np.ndarray, applicability: np.ndarray, queries: list[dict], protocol: dict, calibrator: dict) -> dict:
    mask = np.array([protocol["query_split"][q["query_id"]] == "test" for q in queries])
    known = mask[:, None] & (applicability >= 0)
    y = applicability[known].astype(int)
    groups = {q["compound_id"] for i, q in enumerate(queries) if mask[i] and np.any(applicability[i] >= 0)}
    base = {"state": "unavailable", "scope": "fixed test compounds; agreement with automatic documented applicability",
            "test_pairs": int(known.sum()), "test_total_pairs": int(mask.sum() * scores.shape[1]),
            "test_known_fraction": float(known.sum() / (mask.sum() * scores.shape[1])) if mask.any() else None,
            "test_compounds": sorted(groups), "positive_test_pairs": int(y.sum()),
            "macro_f1": None, "brier": None, "ece": None}
    if calibrator["state"] != "fitted":
        return {**base, "reason": calibrator["reason"]}
    if len(groups) < 2 or len(np.unique(y)) != 2:
        return {**base, "reason": "test_requires_two_classes_and_two_compounds"}
    probabilities = sigmoid((scores.astype(float) - calibrator["mean"]) / calibrator["scale"] * calibrator["slope"] + calibrator["intercept"])
    p, decision = probabilities[known], probabilities[known] >= .5
    tp, fp = int(((y == 1) & decision).sum()), int(((y == 0) & decision).sum())
    tn, fn = int(((y == 0) & ~decision).sum()), int(((y == 1) & ~decision).sum())
    f1_pos, f1_neg = 2 * tp / (2 * tp + fp + fn), 2 * tn / (2 * tn + fp + fn)
    bins, ece = [], 0.
    for bin_id in range(10):
        selected = np.minimum((p * 10).astype(int), 9) == bin_id
        if not selected.any():
            continue
        accuracy, confidence = float(y[selected].mean()), float(p[selected].mean())
        ece += selected.mean() * abs(accuracy - confidence)
        bins.append({"bin": bin_id, "count": int(selected.sum()), "observed_positive_fraction": accuracy, "mean_probability": confidence})
    return {**base, "state": "evaluated", "reason": None,
            "macro_f1": (f1_pos + f1_neg) / 2, "brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}, "calibration_bins": bins,
            "predicted_positive_fraction": float(decision.mean())}


def request_decision_metrics(scores: np.ndarray, applicability: np.ndarray, queries: list[dict],
                             protocol: dict, calibrator: dict) -> dict:
    """Post-calibrated original Top-1 decisions, with unknowns never negatives."""
    indices = [i for i, q in enumerate(queries) if protocol["query_split"][q["query_id"]] == "test"]
    app = applicability[indices]
    exists = np.any(app == 1, axis=1)
    no_applicable = np.all(app == 0, axis=1)
    catalog_known = exists | no_applicable
    top_scores = scores[indices].max(axis=1)
    top_masks = scores[indices] == top_scores[:, None]
    top_known = np.all((app >= 0) | ~top_masks, axis=1)
    expected_app = np.array([float(values[mask].mean()) if known else np.nan
                            for values, mask, known in zip(app, top_masks, top_known)])
    unavailable_names = ("recommendation_coverage", "truth_known_recommendation_coverage",
                         "false_recommendation_rate_among_known_top1", "wrong_rejection_rate_when_applicable_exists",
                         "correct_rejection_rate_when_no_applicable", "recommendation_rate_when_no_applicable")
    base = {"state": "unavailable", "scope": "original rankings, fixed test compounds, automatic documented applicability only",
            "policy": "Keep the raw-score Top-1 set; recommend when its validation-fitted applicability probability >= 0.5. Uniform expectation over exact score ties; never rerank by calibrated probability.",
            "threshold": .5, "test_query_count": len(indices),
            "truth": {"known_applicable_exists_queries": int(exists.sum()),
                      "known_no_applicable_queries": int(no_applicable.sum()),
                      "unknown_catalog_applicability_queries": int((~catalog_known).sum()),
                      "catalog_truth_known_fraction": float(catalog_known.mean()) if len(indices) else None,
                      "top1_applicability_known_queries": int(top_known.sum()),
                      "no_applicable_definition": "Every frozen candidate explicitly known N; any U without a Y makes catalog truth unknown.",
                      "top1_truth_policy": "All tied Top-1 candidates must be known; a tie containing unknown is excluded from selected-model risk."},
            **{name: None for name in unavailable_names}, "risk_coverage": [],
            "native_abstention_measured": False}
    if calibrator.get("state") != "fitted" or not indices:
        return {**base, "reason": calibrator.get("reason") or "no_test_queries"}
    probability = sigmoid((top_scores - calibrator["mean"]) / calibrator["scale"] * calibrator["slope"] + calibrator["intercept"])
    recommend = probability >= .5
    accepted_known = recommend & top_known
    unavailable = {}
    def rate(name, numerator, denominator, reason):
        if not denominator:
            unavailable[name] = reason
            return None
        return float(numerator / denominator)
    values = {
        "recommendation_coverage": float(recommend.mean()),
        "truth_known_recommendation_coverage": rate("truth_known_recommendation_coverage", int((recommend & catalog_known).sum()), int(catalog_known.sum()), "no_known_catalog_truth"),
        "false_recommendation_rate_among_known_top1": rate("false_recommendation_rate_among_known_top1", float(np.sum(1 - expected_app[accepted_known])), int(accepted_known.sum()), "no_recommended_query_with_fully_known_top1_tie_set"),
        "wrong_rejection_rate_when_applicable_exists": rate("wrong_rejection_rate_when_applicable_exists", int((~recommend & exists).sum()), int(exists.sum()), "no_query_with_a_known_applicable_candidate"),
        "correct_rejection_rate_when_no_applicable": rate("correct_rejection_rate_when_no_applicable", int((~recommend & no_applicable).sum()), int(no_applicable.sum()), "no_query_with_all_candidates_known_nonapplicable"),
        "recommendation_rate_when_no_applicable": rate("recommendation_rate_when_no_applicable", int((recommend & no_applicable).sum()), int(no_applicable.sum()), "no_query_with_all_candidates_known_nonapplicable"),
    }
    curve, seen_thresholds = [], set()
    ordered = np.sort(probability)[::-1]
    for fraction in np.linspace(.1, 1., 10):
        threshold = float(ordered[min(math.ceil(len(indices) * fraction) - 1, len(indices) - 1)])
        if threshold in seen_thresholds:
            continue
        seen_thresholds.add(threshold)
        selected = probability >= threshold
        selected_known = selected & top_known
        known_count = int(selected_known.sum())
        curve.append({"probability_threshold": threshold, "selected_queries": int(selected.sum()),
                      "coverage_all_test_queries": float(selected.mean()),
                      "selected_truth_known_queries": known_count,
                      "selected_truth_known_fraction": float(known_count / selected.sum()),
                      "risk_known_top1_only": float(np.mean(1 - expected_app[selected_known])) if known_count else None,
                      "reason": None if known_count else "selected_top1_truth_unknown",
                      "boundary_ties_included": True})
    return {**base, "state": "evaluated", "reason": None, **values,
            "recommended_queries": int(recommend.sum()), "rejected_queries": int((~recommend).sum()),
            "recommended_top1_truth_known_queries": int(accepted_known.sum()),
            "recommended_top1_truth_known_fraction": float(accepted_known.sum() / recommend.sum()) if recommend.any() else None,
            "unavailable_metrics": unavailable, "risk_coverage": curve,
            "risk_interpretation": "Conditional expected error on known Top-1 labels, alongside total and label coverage; this does not estimate risk on unknown cases or establish clinical safety."}


def compound_bootstrap_draws(compounds: list[str]) -> np.ndarray:
    """Fixed resampling indices; their unit is compound, never a query row."""
    seed = int(sha(BOOTSTRAP_SEED.encode())[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, len(compounds), size=(BOOTSTRAP_REPLICATES, len(compounds)))


def bootstrap_compound_mean(values: list[tuple[str, float]], compounds: list[str], draws: np.ndarray) -> dict:
    if not compounds or len(compounds) != len(set(compounds)):
        raise ValueError("Bootstrap requires unique frozen compound clusters")
    if draws.shape != (BOOTSTRAP_REPLICATES, len(compounds)) or np.any(draws < 0) or np.any(draws >= len(compounds)):
        raise ValueError("Invalid fixed compound resampling indices")
    lookup = {compound: i for i, compound in enumerate(compounds)}
    sums, counts = np.zeros(len(compounds)), np.zeros(len(compounds), dtype=int)
    for compound, value in values:
        if compound not in lookup or not math.isfinite(value):
            raise ValueError("Bootstrap values must be finite and belong to frozen compounds")
        sums[lookup[compound]] += value
        counts[lookup[compound]] += 1
    eligible_compounds = int((counts > 0).sum())
    base = {"point_estimate": float(sums.sum() / counts.sum()) if counts.sum() else None,
            "ci95": None, "eligible_queries": int(counts.sum()),
            "eligible_compounds": eligible_compounds, "resampling_compounds": len(compounds),
            "requested_replicates": BOOTSTRAP_REPLICATES, "valid_replicates": 0,
            "empty_replicates": None}
    if eligible_compounds < 2:
        return {**base, "state": "not_computed", "reason": "fewer_than_two_compounds_with_eligible_queries"}
    denominator = counts[draws].sum(axis=1)
    valid = denominator > 0
    bootstrapped = sums[draws][valid].sum(axis=1) / denominator[valid]
    return {**base, "state": "computed", "reason": None,
            "ci95": [float(x) for x in np.quantile(bootstrapped, [.025, .975], method="linear")],
            "valid_replicates": int(valid.sum()), "empty_replicates": int((~valid).sum())}


def original_statistical_uncertainty(methods: dict, queries: list[dict], assessment_complete: bool) -> dict:
    compounds = sorted({q["compound_id"] for q in queries})
    draws = compound_bootstrap_draws(compounds)
    baseline = methods.get("bm25_text", {})
    baseline_complete = baseline.get("progress", {}).get("included", False)
    baseline_rows = baseline.get("variants", {}).get("original", {}).get("per_query", [])
    baseline_by_id = {row["query_id"]: row for row in baseline_rows}
    results = {}
    for method, entry in methods.items():
        unavailable = ("assessment_not_complete" if not assessment_complete else
                       "method_not_complete" if not entry.get("progress", {}).get("included", False) else None)
        base = {"state": "not_computed", "variant": "original", "resampling_unit": "compound_id",
                "resampling_compounds": len(compounds), "replicates": BOOTSTRAP_REPLICATES,
                "metrics": list(BOOTSTRAP_METRICS), "paired_reference": "bm25_text",
                "judge_uncertainty_included": False, "multiple_comparison_significance_claim": False}
        if unavailable:
            results[method] = {**base, "reason": unavailable}
            continue
        per_query = entry.get("variants", {}).get("original", {}).get("per_query", [])
        if not per_query:
            results[method] = {**base, "reason": "no_known_automatic_query_metrics"}
            continue
        judged, bounds, paired = {}, {}, {}
        for metric in BOOTSTRAP_METRICS:
            values = [(row["compound_id"], float(row["judged_only"][metric]))
                      for row in per_query if row.get("judged_only") is not None]
            judged[metric] = bootstrap_compound_mean(values, compounds, draws)
            bounds[metric] = {}
            for side in ("lower", "upper"):
                values = [(row["compound_id"], float(row["full_catalog_bounds"]["metrics"][metric][side]))
                          for row in per_query if row.get("full_catalog_bounds", {}).get("eligible_for_mean_bounds")]
                bounds[metric][side] = bootstrap_compound_mean(values, compounds, draws)
            if not baseline_complete or not baseline_rows:
                paired[metric] = {"state": "not_computed", "reason": "complete_bm25_reference_unavailable"}
                continue
            differences = []
            for row in per_query:
                reference = baseline_by_id.get(row["query_id"])
                if row.get("judged_only") is None or reference is None or reference.get("judged_only") is None:
                    continue
                if row["compound_id"] != reference["compound_id"]:
                    raise ValueError("Paired comparison changed query compound identity")
                differences.append((row["compound_id"], float(row["judged_only"][metric] - reference["judged_only"][metric])))
            paired[metric] = {**bootstrap_compound_mean(differences, compounds, draws),
                              "direction": "method_minus_bm25", "eligible_set": "intersection_of_identical_query_ids_with_both_metrics_defined"}
        computed = any(value["state"] == "computed" for value in judged.values())
        results[method] = {**base, "state": "computed" if computed else "not_computed",
                           "reason": None if computed else "fewer_than_two_compounds_with_eligible_queries",
                           "judged_only": judged, "paired_judged_only_delta_vs_bm25": paired,
                           "full_catalog_bounds_endpoint_uncertainty": bounds,
                           "bounds_interpretation": "Separate marginal endpoint intervals, not a joint 95% region; conservative identification width and compound sampling uncertainty are distinct."}
    return results


def verify_recorded_input_hashes(run_dir: Path, record: dict, label: str) -> bool:
    """Reject stale declared inputs; older fixtures without provenance stay partial.

    Only frozen input files inside the selected run are read. The recorded keys
    are matched as identifiers, never followed as arbitrary filesystem paths.
    """
    hashes = record.get("input_sha256")
    if hashes is None:
        return False
    if not isinstance(hashes, dict):
        raise ValueError(f"Invalid recorded input hashes: {label}")
    names = ["queries.jsonl", "candidates.jsonl"]
    if (run_dir / "inputs/manifest.json").exists():
        names.append("manifest.json")
    for name in names:
        path = run_dir / "inputs" / name
        keys = {name, "inputs/" + name, str(path.relative_to(ROOT.resolve()))}
        values = [hashes[key] for key in keys if key in hashes]
        expected = sha(path.read_bytes())
        if not values or any(value != expected for value in values):
            raise ValueError(f"Recorded inputs differ from frozen files: {label}: {name}")
    return True


def load_method(run_dir: Path, method: str, entry: dict, queries: list[dict], candidates: list[dict]) -> tuple[np.ndarray | None, dict]:
    dest = scoped(run_dir / "results" / method)
    status_path = dest / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    progress = {"state": status.get("state", entry.get("state", "not_started")),
                "completed_query_variants": status.get("completed_query_variants", 0), "included": False}
    if progress["state"] != "complete" or entry.get("state") != "complete":
        return None, {**progress, "reason": "method_or_results_report_not_complete"}
    input_hashes_verified = verify_recorded_input_hashes(run_dir, status, method)
    index = json.loads((dest / "index.json").read_text())
    expected = {"query_ids": [q["query_id"] for q in queries], "candidate_ids": [c["candidate_id"] for c in candidates], "variants": list(VARIANTS)}
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Method index differs from frozen input: {method}")
    filename = entry.get("effective_score_file")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Completed method lacks declared effective_score_file: {method}")
    path = scoped(dest / filename)
    if not path.is_relative_to(dest):
        raise ValueError(f"Effective score file outside method directory: {method}")
    scores = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = (len(VARIANTS), len(queries), len(candidates))
    timings = np.load(dest / "query_seconds.npy", mmap_mode="r", allow_pickle=False)
    if scores.shape != shape or timings.shape != shape[:2]:
        raise ValueError(f"Unexpected score/timing shape for {method}")
    for key, expected_count in (("completed_query_variants", len(VARIANTS) * len(queries)),
                                ("completed_pairs", int(np.prod(shape)))):
        if key in status and status[key] != expected_count:
            raise ValueError(f"Complete status has inconsistent {key}: {method}")
    if not np.isfinite(scores).all() or not np.isfinite(timings).all():
        return None, {**progress, "reason": "nonfinite_or_uncommitted_rows_despite_complete_status"}
    if np.any(timings < 0):
        raise ValueError(f"Negative query timing despite complete status: {method}")
    return scores, {**progress, "included": True, "reason": None, "effective_score_file": str(path.relative_to(run_dir)),
                    "effective_scores_sha256": sha(path.read_bytes()),
                    "input_hashes_verified": input_hashes_verified, "index_ids_and_variants_verified": True,
                    "score_shape": list(shape), "finite_score_count": int(scores.size),
                    "timings_complete_finite_nonnegative": True}


def evaluate_run(run_dir: Path = DEFAULT_RUN, assessment_dir: Path | None = None, *, write_outputs: bool = True,
                 protocol_only: bool = False, require_complete: bool = False) -> dict:
    run_dir = scoped(run_dir)
    assessment = scoped(assessment_dir or run_dir / "assessment")
    queries, candidates = rows(run_dir / "inputs/queries.jsonl"), rows(run_dir / "inputs/candidates.jsonl")
    if not queries or not candidates or len({q["query_id"] for q in queries}) != len(queries) or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Nonempty unique frozen query and candidate ids required")
    protocol = freeze_protocol(run_dir, assessment, queries, candidates)
    if protocol_only:
        return {"state": "protocol_frozen", "protocol": protocol}
    reporting_additions = freeze_reporting_additions(assessment, protocol)
    bounds_revision = freeze_bounds_revision(assessment, protocol)
    statistics_revision = freeze_statistics_revision(assessment, protocol, queries)
    judge_metadata = read_judge_metadata(assessment, protocol)
    grades, applicable, audit = read_judgments(assessment / "judgments.jsonl", queries, candidates,
                                              protocol_sha256=judge_metadata.get("judge_protocol_sha256"))
    declared = json.loads((run_dir / "protocol.json").read_text())["methods"]
    if not declared or len(declared) != len(set(declared)):
        raise ValueError("Nonempty unique declared methods required")
    report = json.loads((run_dir / "results.json").read_text()) if (run_dir / "results.json").exists() else {"methods": {}}
    report_input_hashes_verified = verify_recorded_input_hashes(run_dir, report, "results.json")
    material_keys, material_summary = document_equivalence(assessment)
    methods, flat = {}, []
    for method in declared:
        scores, progress = load_method(run_dir, method, report.get("methods", {}).get(method, {}), queries, candidates)
        item = {"progress": progress}
        methods[method] = item
        if scores is None:
            continue
        if not audit["grade_known_pairs"] and not audit["applicability_known_pairs"]:
            item["metrics_state"] = "waiting_for_known_automatic_judgments"
            item["variants"] = {}
            continue
        calibrator = fit_calibrator(scores[0], applicable, queries, protocol, audit["state"] == "complete")
        item["calibrator"] = calibrator
        item["original_request_decisions"] = request_decision_metrics(scores[0], applicable, queries, protocol, calibrator)
        item["variants"] = {}
        modes = execution_batch_modes(run_dir, method, queries)
        original_per_query = None
        for vi, variant in enumerate(VARIANTS):
            per_query = query_results(scores[vi], grades, queries, candidates, material_keys)
            if variant == "original":
                original_per_query = per_query
            summary = summarize_queries(per_query)
            groups = {}
            for key in ("compound_id", "task_type"):
                grouped = defaultdict(list)
                for row in per_query:
                    grouped[row[key]].append(row)
                groups[key] = {name: summarize_queries(values) for name, values in sorted(grouped.items())}
            split_results = {split: summarize_queries([row for row in per_query if protocol["query_split"][row["query_id"]] == split]) for split in ("validation", "test")}
            calibration = calibrated_metrics(scores[vi], applicable, queries, protocol, calibrator)
            item["variants"][variant] = {"overall": summary, "by_group": groups, "by_split": split_results,
                                         "calibrated_applicability": calibration,
                                         "calibrated_applicability_unit": "candidate-query pair, not request-level rejection",
                                         "perturbation_subsets": perturbation_subsets(per_query, original_per_query, queries, variant, modes),
                                         "per_query": per_query}
            for pool in ("full_pool", "judged_only", "same_drug_full_pool", "different_drug_full_pool"):
                flat.append({"method": method, "variant": variant, "pool": pool,
                             "assessment_state": audit["state"], "eligible_queries": summary[pool]["eligible_queries"],
                             **summary[pool]["metrics"], "test_macro_f1": calibration["macro_f1"],
                             "test_brier": calibration["brier"], "test_ece": calibration["ece"]})
    uncertainty = original_statistical_uncertainty(methods, queries, audit["state"] == "complete")
    for method, value in uncertainty.items():
        methods[method]["statistical_uncertainty"] = value
    included = sum(item["progress"]["included"] for item in methods.values())
    acceptance_checks = {
        "all_declared_method_matrices_complete": included == len(declared),
        "all_expected_pairs_processed_once": audit["processed_unique_pairs"] == len(queries) * len(candidates),
        "no_uncommitted_judgment_tail": audit["uncommitted_tail_bytes"] == 0,
        "primary_judge_protocol_and_sources_verified": judge_metadata.get("state") == "frozen",
        "results_report_input_hashes_verified": report_input_hashes_verified,
        "all_method_input_hashes_verified": all(item["progress"].get("input_hashes_verified", False) for item in methods.values()),
        "unknown_and_failed_judgments_remain_null": True,
    }
    acceptance = {"state": "passed" if all(acceptance_checks.values()) else "incomplete",
                  "checks": acceptance_checks, "expected_methods": len(declared),
                  "expected_judgment_pairs": len(queries) * len(candidates),
                  "expected_score_values": len(declared) * len(VARIANTS) * len(queries) * len(candidates),
                  "processed_pair_completion_is_not_all_labels_known": True}
    result = {"schema_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
              "state": "complete" if audit["state"] == "complete" and included == len(declared) else "partial",
              "evaluation_target": "Agreement with automatic source-grounded document relevance and documented applicability",
              "scientific_claim_eligible": False, "human_agreement_measured": False,
              "independent_pk_prediction_accuracy_measured": False,
              "source_content_independence_verified": False,
              "judge_family_bias": "Actual primary and secondary identities are recorded in judge_metadata. Family overlap with evaluated methods can create correlated bias; cross-family judge agreement does not establish correct labels.",
              "protocol": protocol, "judge_metadata": judge_metadata, "assessment": audit,
              "method_counts": {"expected": len(declared), "included_complete_matrices": included, "pending": len(declared) - included},
              "family_interpretation": "Registered family ids are catalog groups, not proof of scientific independence. Pair preference is descriptive automatic agreement.",
              "judged_only_interpretation": "Unknown candidates are removed only in this explicitly named subset; its recall is not catalog recall",
              "document_equivalence": material_summary,
              "additional_reporting_policy": reporting_additions,
              "full_catalog_bounds_protocol": bounds_revision,
              "statistical_protocol": statistics_revision,
              "completion_acceptance": acceptance,
              "methods": methods}
    if require_complete and acceptance["state"] != "passed":
        failed = [key for key, passed in acceptance_checks.items() if not passed]
        raise ValueError("Full evaluation acceptance incomplete: " + ", ".join(failed))
    if write_outputs:
        dump(assessment / "automatic_evaluation.json", result)
        columns = ["method", "variant", "pool", "assessment_state", "eligible_queries", "mrr", "top1_expected_grade",
                   *[f"{metric}@{k}" for metric in ("ndcg", "recall") for k in KS], "test_macro_f1", "test_brier", "test_ece"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
        path = assessment / "automatic_metrics.csv"
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(buffer.getvalue(), encoding="utf-8")
        os.replace(temporary, path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--assessment-dir", type=Path)
    parser.add_argument("--write-protocol-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="Reject incomplete matrices, judgment coverage or provenance before publishing outputs")
    args = parser.parse_args()
    try:
        result = evaluate_run(args.run_dir, args.assessment_dir, protocol_only=args.write_protocol_only,
                              require_complete=args.require_complete)
        print(json.dumps({key: result[key] for key in ("state", "assessment", "method_counts") if key in result}, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
