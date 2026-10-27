"""CPU-only checks for non-gold, complete-pool automatic frontier evaluation."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_frontier_automatic as automatic


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


@pytest.fixture
def supplemental_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(automatic, "ROOT", tmp_path)
    assessment = tmp_path / "assessment"
    source = {"candidate_id": "c0", "family_id": "f0", "family_name": "drug",
              "compounds": ["drug"], "source_artifact": {
                  "path": "model.pksim5", "sha256": "artifact-sha", "verified": True,
                  "repository_commit": "exact-commit"}, "documents": []}
    other = {**source, "candidate_id": "c1"}
    write_rows(assessment / "sources.jsonl", [source, other])
    text = "This model describes intravenous administration in adults."
    supplement = {**source, "documents": [{"path": "same_commit/README.md", "sha256": "document-sha",
        "version_specific_validation_verified": False, "segments": [{"evidence_id": "e0",
        "evidence_kind": "model_description", "excerpt": text, "excerpt_sha256": automatic.sha(text.encode()),
        "location": {"char_start": 0, "char_end": len(text), "section": "Scope"}}]}]}
    path = assessment / "source_supplements/sources.jsonl"
    write_rows(path, [supplement])
    return assessment, source, supplement


def test_source_supplement_changes_only_the_document_empty_candidate(supplemental_sources):
    assessment, _, _ = supplemental_sources
    key = "assessment/source_supplements/sources.jsonl"
    digest = automatic.sha((assessment / "source_supplements/sources.jsonl").read_bytes())
    write_json(assessment / "judge_protocol.json", {"input_sha256": {key: digest}})
    keys, summary = automatic.document_equivalence(assessment)
    assert keys["c0"] != keys["c1"]
    assert summary["source_supplement"]["applied_candidate_ids"] == ["c0"]
    assert summary["source_supplement"]["protocol_hash_verified"] is True
    assert summary["distinct_packets"] == 2
    assert json.loads((assessment / "sources.jsonl").read_text().splitlines()[0])["documents"] == []


def test_new_supplement_does_not_change_an_old_judge_protocol(supplemental_sources):
    assessment, _, _ = supplemental_sources
    write_json(assessment / "judge_protocol.json", {"input_sha256": {}})
    keys, summary = automatic.document_equivalence(assessment)
    assert keys["c0"] == keys["c1"]
    assert summary["source_supplement"]["state"] == "not_declared_by_frozen_judge_protocol"


@pytest.mark.parametrize("mutation", ["artifact", "compound", "candidate", "duplicate", "overwrite", "hash", "missing"])
def test_source_supplement_rejects_identity_changes_and_unfrozen_bytes(supplemental_sources, mutation):
    assessment, source, supplement = supplemental_sources
    path = assessment / "source_supplements/sources.jsonl"
    if mutation == "artifact":
        supplement["source_artifact"]["sha256"] = "different-version"
    elif mutation == "compound":
        supplement["compounds"] = ["another-drug"]
    elif mutation == "candidate":
        supplement["candidate_id"] = "unknown-candidate"
    elif mutation == "overwrite":
        write_rows(assessment / "sources.jsonl", [{**source, "documents": supplement["documents"]}])
    write_rows(path, [supplement, supplement] if mutation == "duplicate" else [supplement])
    if mutation in {"hash", "missing"}:
        digest = automatic.sha(path.read_bytes())
        write_json(assessment / "judge_protocol.json", {
            "input_sha256": {"assessment/source_supplements/sources.jsonl": "wrong" if mutation == "hash" else digest}})
        if mutation == "missing":
            path.unlink()
    with pytest.raises(ValueError, match="[Ss]upplement"):
        automatic.document_equivalence(assessment)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(automatic, "ROOT", tmp_path)
    root = tmp_path / "frontier"
    queries = [{"query_id": f"q{i}", "compound_id": f"drug{i}",
                "compounds": [f"drug{i}"], "task_type": "pk" if i % 2 else "reference",
                "input_evidence_ids": ["input-source"]} for i in range(8)]
    candidates = [{"candidate_id": f"c{i}", "compounds": [f"drug{j}" for j in range(8)],
                   "family_id": "f0" if i < 2 else "f1"} for i in range(3)]
    write_rows(root / "inputs/queries.jsonl", queries)
    write_rows(root / "inputs/candidates.jsonl", candidates)
    write_json(root / "protocol.json", {"methods": {"a": {}, "pending": {}}})
    write_json(root / "results.json", {"methods": {
        "a": {"state": "complete", "effective_score_file": "effective.npy"},
        "pending": {"state": "scoring"}}})
    dest = root / "results/a"
    write_json(dest / "status.json", {"state": "complete", "completed_query_variants": 24})
    write_json(dest / "index.json", {"query_ids": [q["query_id"] for q in queries],
        "candidate_ids": [c["candidate_id"] for c in candidates], "variants": list(automatic.VARIANTS)})
    scores = np.tile([3., 2., 1.], (3, 8, 1))
    np.save(dest / "effective.npy", scores)
    np.save(dest / "scores.npy", scores[:, :, ::-1])
    np.save(dest / "query_seconds.npy", np.ones((3, 8)))
    judgments = [{"query_id": q["query_id"], "candidate_id": c["candidate_id"],
        "grade": [3, 1, 0][j], "documented_applicability": j == 0, "status": "judged",
        "support": [{"source_id": "source", "quote": "Frozen source evidence."}],
        "label_source": "llm_source_grounded", "confidence": 0.999999} for q in queries for j, c in enumerate(candidates)]
    write_rows(root / "assessment/judgments.jsonl", judgments)
    return root, queries, candidates, judgments


def test_complete_method_uses_effective_matrix_and_pending_stays_visible(run):
    root, _, _, _ = run
    result = automatic.evaluate_run(root)
    assert result["state"] == "partial"
    assert result["assessment"]["state"] == "complete"
    assert result["method_counts"] == {"expected": 2, "included_complete_matrices": 1, "pending": 1}
    method = result["methods"]["a"]
    assert method["progress"]["effective_score_file"].endswith("effective.npy")
    assert method["variants"]["original"]["overall"]["full_pool"]["metrics"]["ndcg@1"] == 1
    assert result["methods"]["pending"]["progress"]["reason"] == "method_or_results_report_not_complete"
    assert method["calibrator"]["state"] == "fitted"
    assert method["variants"]["original"]["calibrated_applicability"]["macro_f1"] == 1
    assert result["human_agreement_measured"] is False
    assert result["scientific_claim_eligible"] is False
    assert (root / "assessment/automatic_metrics.csv").exists()


def test_unknown_is_not_zero_or_catalog_recall(run):
    root, _, _, judgments = run
    judgments[0].update(grade=None, documented_applicability=None, status="insufficient_evidence", support=[])
    write_rows(root / "assessment/judgments.jsonl", judgments)
    result = automatic.evaluate_run(root, write_outputs=False)
    summary = result["methods"]["a"]["variants"]["original"]["overall"]
    assert summary["full_pool"]["eligible_queries"] == 7
    assert summary["full_pool"]["queries_with_unknown_grades"] == 1
    assert result["assessment"]["grade_known_pairs"] == 23
    first = result["methods"]["a"]["variants"]["original"]["per_query"][0]
    assert first["full_pool"] is None and first["judged_only"] is None
    assert first["topk_label_coverage"]["inclusive_judged@1"] == 0


def test_judged_only_uses_explicit_restricted_pool(run):
    root, _, _, judgments = run
    judgments[2].update(grade=None, documented_applicability=None, status="invalid_output", support=[])
    write_rows(root / "assessment/judgments.jsonl", judgments)
    result = automatic.evaluate_run(root, write_outputs=False)
    first = result["methods"]["a"]["variants"]["original"]["per_query"][0]
    assert first["full_pool"] is None
    assert first["judged_only"]["ndcg@1"] == 1


def test_partial_label_snapshot_excludes_uncommitted_tail_and_blocks_fit(run):
    root, _, _, judgments = run
    path = root / "assessment/judgments.jsonl"
    write_rows(path, judgments[:3])
    with path.open("a") as handle:
        handle.write('{"query_id": "unfinished')
    result = automatic.evaluate_run(root, write_outputs=False)
    assert result["assessment"]["processed_unique_pairs"] == 3
    assert result["assessment"]["state"] == "partial"
    assert result["assessment"]["uncommitted_tail_bytes"] > 0
    assert result["methods"]["a"]["calibrator"]["state"] == "unavailable"


@pytest.mark.parametrize("change", ["duplicate", "extra", "wrong_source", "unknown_with_grade", "bad_grade", "no_support"])
def test_invalid_judgments_are_rejected(run, change):
    root, _, _, judgments = run
    if change == "duplicate":
        judgments.append(judgments[0])
    elif change == "extra":
        judgments[0]["candidate_id"] = "missing"
    elif change == "wrong_source":
        judgments[0]["label_source"] = "blinded_expert"
    elif change == "unknown_with_grade":
        judgments[0]["status"] = "invalid_output"
    elif change == "bad_grade":
        judgments[0]["grade"] = True
    else:
        judgments[0]["support"] = []
    write_rows(root / "assessment/judgments.jsonl", judgments)
    with pytest.raises(ValueError):
        automatic.evaluate_run(root, write_outputs=False)


def test_ties_are_expected_values_not_arbitrary_candidate_order():
    grades = np.array([3, 0, 2, 1])
    tied = automatic.rank_metrics(np.ones(4), grades)
    observed = []
    for permutation in itertools.permutations(grades):
        observed.append(automatic.rank_metrics(np.arange(4, 0, -1), np.array(permutation)))
    for key in ("ndcg@1", "ndcg@3", "recall@1", "recall@3", "mrr", "top1_expected_grade"):
        assert tied[key] == pytest.approx(np.mean([r[key] for r in observed]))
    assert tied["top1_tie_size"] == 4


def test_calibrator_cannot_read_test_labels_or_llm_confidence(run):
    root, queries, _, judgments = run
    result = automatic.evaluate_run(root, write_outputs=False)
    fit = result["methods"]["a"]["calibrator"]
    splits = result["protocol"]["query_split"]
    for row in judgments:
        row["confidence"] = 0
        if splits[row["query_id"]] == "test":
            row["documented_applicability"] = not row["documented_applicability"]
    write_rows(root / "assessment/judgments.jsonl", judgments)
    changed = automatic.evaluate_run(root, write_outputs=False)
    assert changed["methods"]["a"]["calibrator"] == fit
    assert set(fit["fit_compounds"]).isdisjoint(changed["methods"]["a"]["variants"]["original"]["calibrated_applicability"]["test_compounds"])


def test_single_class_calibration_is_unavailable(run):
    root, _, _, judgments = run
    for row in judgments:
        row["documented_applicability"] = True
    write_rows(root / "assessment/judgments.jsonl", judgments)
    result = automatic.evaluate_run(root, write_outputs=False)
    assert result["methods"]["a"]["calibrator"]["state"] == "unavailable"
    metric = result["methods"]["a"]["variants"]["original"]["calibrated_applicability"]
    assert metric["macro_f1"] is None and metric["brier"] is None


def test_frozen_split_rejects_changed_inputs(run):
    root, queries, _, _ = run
    automatic.evaluate_run(root, protocol_only=True)
    queries[0]["compound_id"] = "changed"
    write_rows(root / "inputs/queries.jsonl", queries)
    with pytest.raises(ValueError, match="Frozen evaluation protocol"):
        automatic.evaluate_run(root, protocol_only=True)


def test_complete_status_does_not_hide_bad_index_or_uncommitted_rows(run):
    root, _, _, _ = run
    path = root / "results/a/query_seconds.npy"
    timings = np.load(path)
    timings[2, 7] = np.nan
    np.save(path, timings)
    result = automatic.evaluate_run(root, write_outputs=False)
    assert result["methods"]["a"]["progress"]["included"] is False
    index_path = root / "results/a/index.json"
    index = json.loads(index_path.read_text())
    index["candidate_ids"].reverse()
    write_json(index_path, index)
    with pytest.raises(ValueError, match="index differs"):
        automatic.evaluate_run(root, write_outputs=False)


def test_all_methods_and_labels_needed_for_complete_state(run):
    root, _, _, _ = run
    write_json(root / "protocol.json", {"methods": {"a": {}}})
    result = automatic.evaluate_run(root, write_outputs=False)
    assert result["state"] == "complete"
    assert result["assessment"]["processed_unique_pairs"] == result["protocol"]["expected_judgment_count"]


def test_judge_metadata_is_frozen_primary_and_excludes_private_model_path(run):
    root, _, _, judgments = run
    protocol = automatic.evaluate_run(root, protocol_only=True)["protocol"]
    sources_path = root / "assessment/sources.jsonl"
    sources_path.write_text(json.dumps({"candidate_id": "c0", "family_name": "fixture", "compounds": [],
                                       "source_artifact": {"verified": True}, "documents": []}) + "\n")
    judge = {"primary_repo": "microsoft/Phi-4-mini-instruct", "primary_revision": "frozen-revision",
             "secondary_repo": "Qwen/Qwen2.5-3B-Instruct", "system_prompt_sha256": "prompt-hash",
             "input_sha256": {"inputs/" + k: v for k, v in protocol["input_sha256"].items()}}
    judge["input_sha256"]["assessment/sources.jsonl"] = automatic.sha(sources_path.read_bytes())
    judge_path = root / "assessment/judge_protocol.json"
    write_json(judge_path, judge)
    write_json(root / "assessment/primary_weights.json", {"repo": judge["primary_repo"], "revision": "frozen-revision",
               "path": "/private/not-exported", "sha256": {"weights.safetensors": "file-hash"}})
    for row in judgments:
        row.update(judge="primary", protocol_sha256=automatic.sha(judge_path.read_bytes()))
    write_rows(root / "assessment/judgments.jsonl", judgments)
    result = automatic.evaluate_run(root, write_outputs=False)
    metadata = result["judge_metadata"]
    assert metadata["primary_repo"] == "microsoft/Phi-4-mini-instruct"
    assert "path" not in metadata["primary_weights"]
    judgments[0]["judge"] = "secondary"
    write_rows(root / "assessment/judgments.jsonl", judgments)
    with pytest.raises(ValueError, match="frozen primary"):
        automatic.evaluate_run(root, write_outputs=False)
    sources_path.write_text("changed\n")
    with pytest.raises(ValueError, match="sources missing or changed"):
        automatic.evaluate_run(root, write_outputs=False)


def decision_fixture(app, scores=None):
    app = np.array(app, dtype=int)
    query = [{"query_id": f"q{i}", "compound_id": f"d{i}"} for i in range(len(app))]
    protocol = {"query_split": {q["query_id"]: "test" for q in query}}
    calibrator = {"state": "fitted", "mean": 0., "scale": 1., "slope": 1., "intercept": 0.}
    if scores is None:
        scores = np.tile(np.arange(app.shape[1], 0, -1), (len(app), 1))
    return np.asarray(scores, dtype=float), app, query, protocol, calibrator


def test_no_applicable_requires_every_candidate_known_negative():
    args = decision_fixture([[0, 0, 0], [0, -1, 0], [0, -1, 1]])
    result = automatic.request_decision_metrics(*args)
    assert result["truth"]["known_no_applicable_queries"] == 1
    assert result["truth"]["unknown_catalog_applicability_queries"] == 1
    assert result["truth"]["known_applicable_exists_queries"] == 1
    assert result["recommendation_rate_when_no_applicable"] == 1
    assert result["false_recommendation_rate_among_known_top1"] == 1


def test_unknown_top1_never_counts_as_false_recommendation():
    args = decision_fixture([[-1, 0, 0], [1, 0, -1]])
    result = automatic.request_decision_metrics(*args)
    assert result["recommended_queries"] == 2
    assert result["recommended_top1_truth_known_queries"] == 1
    assert result["recommended_top1_truth_known_fraction"] == .5
    assert result["false_recommendation_rate_among_known_top1"] == 0
    assert result["correct_rejection_rate_when_no_applicable"] is None
    assert "correct_rejection_rate_when_no_applicable" in result["unavailable_metrics"]


def test_tied_top1_has_expected_risk_and_unknown_tie_is_excluded():
    args = decision_fixture([[1, 0], [1, -1]], scores=[[2, 2], [1, 1]])
    result = automatic.request_decision_metrics(*args)
    assert result["recommended_top1_truth_known_queries"] == 1
    assert result["false_recommendation_rate_among_known_top1"] == .5
    assert all(point["risk_known_top1_only"] == .5 for point in result["risk_coverage"])
    assert result["risk_coverage"][-1]["selected_truth_known_fraction"] == .5


def test_wrong_rejection_compares_to_known_existence_not_unknown():
    args = decision_fixture([[1, -1], [0, 0], [0, -1]], scores=[[-1, -2]] * 3)
    result = automatic.request_decision_metrics(*args)
    assert result["recommendation_coverage"] == 0
    assert result["wrong_rejection_rate_when_applicable_exists"] == 1
    assert result["correct_rejection_rate_when_no_applicable"] == 1
    assert result["false_recommendation_rate_among_known_top1"] is None
    assert result["truth"]["unknown_catalog_applicability_queries"] == 1


def test_request_decisions_use_raw_top1_not_reversed_calibrator_ranking():
    args = list(decision_fixture([[1, 0]], scores=[[1, -1]]))
    args[-1]["slope"] = -1
    result = automatic.request_decision_metrics(*args)
    assert result["recommended_queries"] == 0
    assert result["wrong_rejection_rate_when_applicable_exists"] == 1


def test_request_decisions_block_when_calibration_unavailable():
    args = list(decision_fixture([[1, 0]]))
    args[-1] = {"state": "unavailable", "reason": "assessment_not_complete"}
    result = automatic.request_decision_metrics(*args)
    assert result["state"] == "unavailable"
    assert result["recommendation_coverage"] is None
    assert result["reason"] == "assessment_not_complete"


def test_perturbation_subsets_exclude_noop_and_changed_batch(run):
    _, queries, candidates, _ = run
    queries = queries[:3]
    for i, q in enumerate(queries):
        q["text"] = f"request {i}"
        q["variants"] = {"missing_evidence": {"text": q["text"] if i == 0 else "changed"}}
    per_query = automatic.query_results(np.tile([3., 2., 1.], (3, 1)), np.tile([3, 1, 0], (3, 1)), queries, candidates)
    modes = np.ones((3, 3), dtype=int)
    modes[2, 2] = 4
    result = automatic.perturbation_subsets(per_query, per_query, queries, "missing_evidence", modes)
    assert result["changed_input_queries"] == 2
    assert result["unchanged_input_queries_excluded"] == 1
    assert result["changed_text_only"]["query_count"] == 2
    assert result["matched_execution_batch_pairs"] == 1
    assert result["cross_execution_batch_pairs_excluded"] == 1
    assert result["matched_execution_batch_variant"]["query_count"] == 1


def test_source_identical_version_pairs_excluded_even_if_grades_disagree(run):
    _, _, candidates, _ = run
    result = automatic.pair_preference(np.array([3., 2., 1.]), np.array([3, 1, 0]), candidates,
                                        np.array([0, 1]), True, {"c0": "same", "c1": "same"})
    assert result["pairs"] == 0
    assert result["accuracy"] is None
    assert result["source_indistinguishable_pairs_excluded"] == 1


def test_execution_batch_modes_retain_serial_rows_and_validate_ids(run):
    root, queries, _, _ = run
    path = root / "results/a/execution_batches.jsonl"
    write_rows(path, [{"query_batch_size": 16, "variant": "missing_evidence", "query_ids": ["q2"]}])
    modes = automatic.execution_batch_modes(root, "a", queries)
    assert modes[0, 2] == 1 and modes[2, 2] == 16
    write_rows(path, [{"query_batch_size": 16, "variant": "missing_evidence", "query_ids": ["other"]}])
    with pytest.raises(ValueError, match="outside frozen"):
        automatic.execution_batch_modes(root, "a", queries)


def brute_tie_expected_metrics(scores, grades):
    """Independent oracle: enumerate every ordering inside each score tie."""
    blocks = [tuple(i for i, value in enumerate(scores) if value == score)
              for score in sorted(set(scores), reverse=True)]
    permutations = itertools.product(*(list(itertools.permutations(block)) for block in blocks))
    gains = np.array([2.**value - 1 for value in grades])
    ideal = np.sort(gains)[::-1]
    discounts = 1 / np.log2(np.arange(len(scores)) + 2)
    results = []
    for combination in permutations:
        order = [index for block in combination for index in block]
        relevant = np.array(grades)[order] >= 2
        row = {"mrr": 1 / (int(np.flatnonzero(relevant)[0]) + 1)}
        for k in automatic.KS:
            stop = min(k, len(scores))
            row[f"ndcg@{k}"] = float(np.sum(gains[order][:stop] * discounts[:stop]) / np.sum(ideal[:stop] * discounts[:stop]))
            row[f"recall@{k}"] = float(relevant[:stop].sum() / relevant.sum())
        results.append(row)
    return {key: float(np.mean([row[key] for row in results])) for key in results[0]}


def test_full_catalog_bounds_exhaustive_three_candidate_completion_and_tie_orders():
    """All 3-score patterns, partial labels, completions and exact tie orders."""
    for scores_tuple in itertools.product(range(3), repeat=3):
        scores = np.array(scores_tuple, dtype=float)
        for partial_tuple in itertools.product([-1, 0, 1, 2, 3], repeat=3):
            grades = np.array(partial_tuple)
            if not np.any(grades >= 2):
                continue
            before = grades.copy()
            bounds = automatic.full_catalog_bounds(scores, grades)
            unknown = np.flatnonzero(grades < 0)
            possible = []
            for values in itertools.product(range(4), repeat=len(unknown)):
                completed = grades.copy()
                completed[unknown] = values
                truth = brute_tie_expected_metrics(scores_tuple, completed)
                possible.append(truth)
                for name, value in truth.items():
                    lower, upper = bounds["metrics"][name]["lower"], bounds["metrics"][name]["upper"]
                    assert lower - 1e-12 <= value <= upper + 1e-12, (scores_tuple, partial_tuple, values, name, lower, value, upper)
            # The MRR claim is exact, not only a valid conservative interval.
            assert bounds["metrics"]["mrr"]["lower"] == pytest.approx(min(x["mrr"] for x in possible))
            assert bounds["metrics"]["mrr"]["upper"] == pytest.approx(max(x["mrr"] for x in possible))
            assert np.array_equal(grades, before)
            if not len(unknown):
                for metric in bounds["metrics"].values():
                    assert metric["lower"] == pytest.approx(metric["upper"])


def test_unknown_without_known_positive_is_conditional_not_eligible():
    scores = np.array([3., 2., 1.])
    conditional = automatic.full_catalog_bounds(scores, np.array([0, -1, 1]))
    assert conditional["state"] == "conditional_coarse_bounds"
    assert not conditional["eligible_for_mean_bounds"]
    assert all(item == {"lower": 0., "upper": 1.} for item in conditional["metrics"].values())
    undefined = automatic.full_catalog_bounds(scores, np.array([0, 1, 1]))
    assert undefined["state"] == "undefined_no_relevant_candidates"
    assert all(item == {"lower": None, "upper": None} for item in undefined["metrics"].values())
    summary = automatic.summarize_catalog_bounds([conditional, undefined])
    assert summary["eligible_queries"] == 0
    assert summary["conditional_only_queries_excluded"] == 1
    assert summary["complete_pool_without_positive_queries"] == 1
    assert summary["metrics"]["mrr"] == {"lower": None, "upper": None}


def test_full_pool_stays_unknown_while_bounds_retain_all_candidates(run):
    root, _, _, judgments = run
    judgments[2].update(grade=None, documented_applicability=None, status="insufficient_evidence", support=[])
    write_rows(root / "assessment/judgments.jsonl", judgments)
    result = automatic.evaluate_run(root, write_outputs=False)
    row = result["methods"]["a"]["variants"]["original"]["per_query"][0]
    assert row["full_pool"] is None
    assert row["full_catalog_bounds"]["candidate_count"] == 3
    assert row["full_catalog_bounds"]["unknown_candidates"] == 1
    assert row["full_catalog_bounds"]["eligible_for_mean_bounds"]
    assert row["full_catalog_bounds"]["metrics"]["ndcg@1"] == {"lower": 1., "upper": 1.}
    assert row["full_catalog_bounds"]["metrics"]["recall@1"] == {"lower": .5, "upper": 1.}
    assert result["methods"]["a"]["variants"]["original"]["overall"]["full_catalog_bounds"]["eligible_queries"] == 8


def test_bounds_revision_is_additive_and_frozen(run):
    root, _, _, _ = run
    automatic.evaluate_run(root, write_outputs=False)
    base = (root / "assessment/evaluation_protocol.json").read_bytes()
    additions = (root / "assessment/evaluation_protocol_additions.json").read_bytes()
    path = root / "assessment/evaluation_protocol_bounds.json"
    revision = json.loads(path.read_text())
    assert revision["previous_additions_file_sha256"] == automatic.sha(additions)
    assert revision["recorded_at"]
    automatic.evaluate_run(root, write_outputs=False)
    assert (root / "assessment/evaluation_protocol.json").read_bytes() == base
    assert (root / "assessment/evaluation_protocol_additions.json").read_bytes() == additions
    revision["ndcg"]["tight"] = True
    write_json(path, revision)
    with pytest.raises(ValueError, match="Frozen full-catalog bounds"):
        automatic.evaluate_run(root, write_outputs=False)


def test_compound_bootstrap_does_not_treat_duplicate_queries_as_new_clusters():
    compounds = ["a", "b"]
    draws = automatic.compound_bootstrap_draws(compounds)
    values = [("a", 0.), ("b", 1.)]
    original = automatic.bootstrap_compound_mean(values, compounds, draws)
    duplicated = automatic.bootstrap_compound_mean(values * 50, compounds, draws)
    assert original["eligible_compounds"] == duplicated["eligible_compounds"] == 2
    assert original["eligible_queries"] == 2 and duplicated["eligible_queries"] == 100
    assert original["ci95"] == duplicated["ci95"] == [0., 1.]
    assert original["point_estimate"] == duplicated["point_estimate"] == .5
    assert original["valid_replicates"] == 2000
    assert np.array_equal(draws, automatic.compound_bootstrap_draws(compounds))


def test_bootstrap_keeps_query_macro_weighting_and_reports_all_frozen_compounds():
    compounds = ["a", "b", "without_eligible_queries"]
    draws = automatic.compound_bootstrap_draws(compounds)
    result = automatic.bootstrap_compound_mean([("a", 0.)] * 3 + [("b", 1.)], compounds, draws)
    assert result["point_estimate"] == .25  # Equal-compound mean would be .5.
    assert result["eligible_compounds"] == 2 and result["resampling_compounds"] == 3
    manual = []
    for indices in draws:
        a_count, b_count = int((indices == 0).sum()), int((indices == 1).sum())
        if 3 * a_count + b_count:
            manual.append(b_count / (3 * a_count + b_count))
    assert result["ci95"] == pytest.approx(np.quantile(manual, [.025, .975], method="linear"))
    assert result["valid_replicates"] == len(manual)
    assert result["empty_replicates"] == 2000 - len(manual)
    assert result["empty_replicates"] > 0


def statistics_fixture_rows():
    rows = []
    for i, value in enumerate([.2, .5, .8]):
        metrics = {key: value for key in automatic.BOOTSTRAP_METRICS}
        rows.append({"query_id": f"q{i}", "compound_id": f"d{i}", "judged_only": metrics,
                     "full_catalog_bounds": {"eligible_for_mean_bounds": True,
                         "metrics": {key: {"lower": value / 2, "upper": 1.} for key in metrics}}})
    return rows


def test_paired_bootstrap_identical_method_delta_zero_and_common_queries_only():
    rows = statistics_fixture_rows()
    baseline = {"progress": {"included": True}, "variants": {"original": {"per_query": rows}}}
    changed_rows = json.loads(json.dumps(rows))
    changed_rows[2]["judged_only"] = None
    for row in changed_rows[:2]:
        for metric in row["judged_only"]:
            row["judged_only"][metric] += .1
    methods = {"bm25_text": baseline, "same": baseline,
               "changed": {"progress": {"included": True}, "variants": {"original": {"per_query": changed_rows}}}}
    result = automatic.original_statistical_uncertainty(methods, rows, True)
    for method in ("bm25_text", "same"):
        for metric in automatic.BOOTSTRAP_METRICS:
            delta = result[method]["paired_judged_only_delta_vs_bm25"][metric]
            assert delta["point_estimate"] == 0 and delta["ci95"] == [0., 0.]
            assert delta["eligible_queries"] == 3
    for metric in automatic.BOOTSTRAP_METRICS:
        delta = result["changed"]["paired_judged_only_delta_vs_bm25"][metric]
        assert delta["point_estimate"] == pytest.approx(.1)
        assert delta["ci95"] == pytest.approx([.1, .1])
        assert delta["eligible_queries"] == delta["eligible_compounds"] == 2
        assert result["same"]["full_catalog_bounds_endpoint_uncertainty"][metric]["upper"]["ci95"] == [1., 1.]


def test_statistics_not_computed_for_partial_labels_or_methods_or_single_cluster():
    rows = statistics_fixture_rows()
    methods = {"bm25_text": {"progress": {"included": True}, "variants": {"original": {"per_query": rows}}},
               "pending": {"progress": {"included": False}}}
    partial = automatic.original_statistical_uncertainty(methods, rows, False)
    assert all(value["state"] == "not_computed" and value["reason"] == "assessment_not_complete" for value in partial.values())
    complete = automatic.original_statistical_uncertainty(methods, rows, True)
    assert complete["pending"]["reason"] == "method_not_complete"
    one_cluster = automatic.bootstrap_compound_mean([("d0", .5)] * 100, ["d0", "d1"], automatic.compound_bootstrap_draws(["d0", "d1"]))
    assert one_cluster["state"] == "not_computed" and one_cluster["ci95"] is None
    assert one_cluster["eligible_compounds"] == 1


def test_statistics_revision_is_frozen_without_overwriting_previous_protocols(run):
    root, _, _, _ = run
    automatic.evaluate_run(root, write_outputs=False)
    folder = root / "assessment"
    previous = {name: (folder / name).read_bytes() for name in ("evaluation_protocol.json", "evaluation_protocol_additions.json", "evaluation_protocol_bounds.json")}
    path = folder / "evaluation_protocol_statistics.json"
    specification = json.loads(path.read_text())
    assert specification["replicates"] == 2000
    assert specification["previous_bounds_file_sha256"] == automatic.sha(previous["evaluation_protocol_bounds.json"])
    automatic.evaluate_run(root, write_outputs=False)
    assert previous == {name: (folder / name).read_bytes() for name in previous}
    specification["replicates"] = 100
    write_json(path, specification)
    with pytest.raises(ValueError, match="Frozen compound-bootstrap"):
        automatic.evaluate_run(root, write_outputs=False)


@pytest.fixture
def new_judge_run(run):
    root, queries, candidates, judgments = run
    assessment = root / "assessment_qwen36"
    protocol = automatic.evaluate_run(root, assessment, protocol_only=True)["protocol"]
    hashes = {"inputs/" + key: value for key, value in protocol["input_sha256"].items()}
    write_json(root / "protocol.json", {"methods": {"a": {}}})
    report = json.loads((root / "results.json").read_text())
    report["input_sha256"] = hashes
    write_json(root / "results.json", report)
    status_path = root / "results/a/status.json"
    status = json.loads(status_path.read_text())
    status.update(input_sha256=hashes, completed_pairs=72)
    write_json(status_path, status)
    sources_path = assessment / "sources.jsonl"
    write_rows(sources_path, [{"candidate_id": c["candidate_id"], "family_name": "fixture",
        "compounds": [], "source_artifact": {"verified": True}, "documents": []} for c in candidates])
    judge = {"primary_repo": "Qwen/fixture-primary", "primary_revision": "primary-commit",
             "secondary_repo": "microsoft/phi-4", "secondary_revision": "secondary-commit",
             "input_sha256": {**hashes, "assessment_qwen36/sources.jsonl": automatic.sha(sources_path.read_bytes())}}
    judge_path = assessment / "judge_protocol.json"
    write_json(judge_path, judge)
    for row in judgments:
        row.update(judge="primary", protocol_sha256=automatic.sha(judge_path.read_bytes()))
    write_rows(assessment / "judgments.jsonl", judgments)
    return root, assessment, judgments


def test_new_judge_directory_preserves_old_run_and_records_actual_identities(new_judge_run):
    root, assessment, _ = new_judge_run
    old_labels = (root / "assessment/judgments.jsonl").read_bytes()
    result = automatic.evaluate_run(root, assessment, require_complete=True)
    assert result["state"] == "complete"
    assert result["completion_acceptance"]["state"] == "passed"
    assert result["completion_acceptance"]["expected_judgment_pairs"] == 24
    assert result["completion_acceptance"]["expected_score_values"] == 72
    assert result["judge_metadata"]["primary_repo"] == "Qwen/fixture-primary"
    assert result["judge_metadata"]["secondary_revision"] == "secondary-commit"
    assert "Qwen-family secondary" not in result["judge_family_bias"]
    assert (root / "assessment/judgments.jsonl").read_bytes() == old_labels
    assert not (root / "assessment/automatic_evaluation.json").exists()


@pytest.mark.parametrize("status", ["insufficient_evidence", "source_conflict", "invalid_output"])
def test_full_completion_acceptance_preserves_unknown_or_failed_label(new_judge_run, status):
    root, assessment, judgments = new_judge_run
    judgments[0].update(status=status, grade=None, documented_applicability=None, support=[])
    write_rows(assessment / "judgments.jsonl", judgments)
    result = automatic.evaluate_run(root, assessment, require_complete=True, write_outputs=False)
    assert result["completion_acceptance"]["state"] == "passed"
    assert result["assessment"]["processed_unique_pairs"] == 24
    assert result["assessment"]["grade_known_pairs"] == 23
    assert result["assessment"]["applicability_known_pairs"] == 23
    assert result["methods"]["a"]["variants"]["original"]["per_query"][0]["full_pool"] is None


@pytest.mark.parametrize("change", ["missing_pair", "tail", "missing_method"])
def test_strict_completion_refuses_partial_output_publication(new_judge_run, change):
    root, assessment, judgments = new_judge_run
    if change == "missing_pair":
        write_rows(assessment / "judgments.jsonl", judgments[:-1])
    elif change == "tail":
        with (assessment / "judgments.jsonl").open("a") as handle:
            handle.write('{"uncommitted":')
    else:
        write_json(root / "protocol.json", {"methods": {"a": {}, "pending": {}}})
    with pytest.raises(ValueError, match="acceptance incomplete"):
        automatic.evaluate_run(root, assessment, require_complete=True)
    assert not (assessment / "automatic_evaluation.json").exists()
    assert not (assessment / "automatic_metrics.csv").exists()


@pytest.mark.parametrize("change", ["stale_report_inputs", "stale_method_inputs", "wrong_completed_pairs", "negative_timing"])
def test_complete_status_cannot_hide_inconsistent_provenance_or_counts(new_judge_run, change):
    root, assessment, _ = new_judge_run
    if change in {"stale_report_inputs", "stale_method_inputs", "wrong_completed_pairs"}:
        path = root / ("results.json" if change == "stale_report_inputs" else "results/a/status.json")
        record = json.loads(path.read_text())
        if change == "wrong_completed_pairs":
            record["completed_pairs"] -= 1
        else:
            record["input_sha256"]["inputs/queries.jsonl"] = "stale"
        write_json(path, record)
    else:
        path = root / "results/a/query_seconds.npy"
        timing = np.load(path)
        timing[0, 0] = -1
        np.save(path, timing)
    with pytest.raises(ValueError, match="[Ii]nputs differ|inconsistent completed_pairs|Negative query timing"):
        automatic.evaluate_run(root, assessment, require_complete=True, write_outputs=False)


def test_assessment_hash_aliases_must_agree(new_judge_run):
    root, assessment, _ = new_judge_run
    path = assessment / "judge_protocol.json"
    judge = json.loads(path.read_text())
    judge["input_sha256"]["assessment/sources.jsonl"] = "contradiction"
    write_json(path, judge)
    with pytest.raises(ValueError, match="Conflicting.*hash namespaces"):
        automatic.evaluate_run(root, assessment, write_outputs=False)


def test_strict_completion_requires_frozen_judge_and_input_provenance(run):
    root, _, _, _ = run
    write_json(root / "protocol.json", {"methods": {"a": {}}})
    with pytest.raises(ValueError, match="primary_judge_protocol_and_sources_verified"):
        automatic.evaluate_run(root, require_complete=True)
