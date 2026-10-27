"""Prospective gold/citation consistency and read-only provenance, CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import frontier_judge_validation as original
import frontier_judge_validation_v5_fresh as fresh


def oracle(case):
    ids = next(fresh.expected_citation_sets(case))
    return {"case_id": case["case_id"], "output": fresh.expected_output(case, ids)}


def test_every_gold_combination_survives_original_parse_scope_and_conditional_grammar():
    cases = fresh.new_controls()
    checks = fresh.validate_expected_outputs(cases)
    assert checks["case_count"] == 32
    assert checks["accepted_citation_sets_checked"] == 70
    assert checks["ordered_answers_checked"] > 70
    assert set(c["expected"]["grades"][0] for c in cases) == {None, 0, 1, 2, 3}
    assert set(c["query"]["task_type"] for c in cases) == set(original.TASKS)
    assert all(r["control_passed"] for r in original.evaluate_validation([oracle(c) for c in cases], cases)["rows"])


def test_necessary_bridge_cannot_be_replaced_by_only_one_paragraph_or_irrelevant_note():
    case = next(c for c in fresh.new_controls() if c["case_id"].endswith("documented_output_matrix_conversion_bridge:base"))
    for ids, correct in (([1], False), ([2], False), ([1, 2], True), ([2, 1], True), ([1, 2, 3], False)):
        record = {"case_id": case["case_id"], "output": fresh.expected_output(case, ids)}
        assert original.evaluate_validation([record], [case])["rows"][0]["control_passed"] is correct


def test_permutation_preserves_proofs_as_quotes_not_literal_ids():
    cases = fresh.new_controls()
    for i in range(0, 32, 2):
        first, second = cases[i:i + 2]
        for c in (first, second):
            c["proof_quotes"] = {
                tuple(sorted(c["packet"]["refs"][n-1]["quote"] for n in ids))
                for ids in fresh.expected_citation_sets(c)}
        assert first["query"]["text"] == second["query"]["text"]
        assert first["expected"]["grades"] == second["expected"]["grades"]
        assert first["proof_quotes"] == second["proof_quotes"]


def test_incomplete_joint_criteria_do_not_become_partial_positive_by_nonempty_citations():
    case = next(c for c in fresh.new_controls() if c["case_id"].endswith("pairwise_ddi_studies_do_not_establish_simultaneous_triple:base"))
    record = oracle(case)
    answer = json.loads(record["output"])
    answer[0].update(g=2, a="Y", e=[1, 2])
    result = original.evaluate_validation([{**record, "output": json.dumps(answer)}], [case])["rows"][0]
    assert result["citation_numbers_valid"] and not result["control_passed"]


def test_verifier_does_not_rebuild_or_modify_saved_material(tmp_path, monkeypatch):
    out = tmp_path / "fresh"
    fresh.build_fresh(out)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    def fail():
        raise AssertionError("Read-only verifier rebuilt controls")
    monkeypatch.setattr(fresh, "new_controls", fail)
    result = fresh.verify_fresh(out)
    assert result["case_count"] == 32
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}


def test_frozen_cases_tampering_is_not_silently_repaired(tmp_path):
    out = tmp_path / "fresh"
    fresh.build_fresh(out)
    path = out / "cases.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        fresh.verify_fresh(out)
    with pytest.raises(ValueError, match="Refuse to replace"):
        fresh.build_fresh(out)


def test_cannot_author_after_new_model_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "ASSESS", tmp_path)
    (tmp_path / "phi_records.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="precede every v5 model output"):
        fresh.build_fresh(tmp_path / "fresh")
