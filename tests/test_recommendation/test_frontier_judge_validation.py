"""Meaningful CPU tests for frozen controls, blinding and citation semantics."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import frontier_judge_validation as validation


def dataset():
    queries, candidates, packets = [], [], {}
    for index in range(30):
        drug, cid = f"drug-{index:02d}", f"c{index:02d}"
        queries.append({"query_id": f"q{index:02d}", "text": f"Find documented models of {drug}",
                        "task_type": validation.TASKS[index % 5], "compounds": [drug], "features": {}})
        candidates.append({"candidate_id": cid, "compounds": [drug], "family_id": f"family-{index:02d}"})
        text = f"Exact existing source description of {drug}. " * (index + 1)
        body = {"documented_model_name": drug, "model_compounds": [drug], "scope_warning": "Original source.",
                "excerpts": [{"n": 1, "text": text, "kind": "model_description"}]}
        packets[cid] = {"body": body, "refs": [{"quote": text, "excerpt_sha256": validation._sha(text),
                                                "source_id": cid, "document_path": f"models/{cid}.md",
                                                "document_sha256": validation._sha(text)}],
                        "source_candidate_id": cid, "packet_sha256": validation._sha(validation._canonical(body)),
                        "characters": len(text)}
    return queries, candidates, packets


def expected_record(case):
    expected = case["expected"]
    if expected is None:
        grade, applicable, numbers = None, None, []
    else:
        grade = expected["grades"][0]
        applicable = expected["documented_applicability"]
        numbers = [group[0] for group in expected["required_support_any_of"]]
    return {"case_id": case["case_id"], "answer": {"grade": grade,
            "documented_applicability": applicable, "evidence_numbers": numbers}}


def test_frozen_set_covers_actual_sources_without_labelling_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    queries, candidates, packets = dataset()
    out = tmp_path / "validation"
    manifest = validation.build_validation(queries, candidates, packets, out)
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert manifest["counts"] == {"synthetic_control": 64, "real_source_pair": 60}
    assert manifest["independent_synthetic_scenarios"] == 16
    coverage = manifest["real_source_coverage"]
    assert len(coverage["compounds"]) == 30
    assert set(coverage["tasks"]) == set(validation.TASKS)
    assert coverage["compound_overlap"] == {"same_compound": 30, "cross_compound": 30}
    assert coverage["material_selection"] == {"lower_third": 30, "upper_third": 30}
    for case in cases:
        if case["kind"] == "real_source_pair":
            assert case["expected"] is None
            assert case["packet"] == packets[case["candidate_id"]]
            assert case["query"] in queries
    # Input iteration order is not a source of sample-selection variability.
    assert validation.build_validation(list(reversed(queries)), list(reversed(candidates)), packets, out) == manifest


def test_frozen_content_and_inputs_cannot_silently_change(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    queries, candidates, packets = dataset()
    out = tmp_path / "validation"
    validation.build_validation(queries, candidates, packets, out)
    queries[0]["text"] += " changed scope"
    with pytest.raises(ValueError, match="Frozen validation file differs"):
        validation.build_validation(queries, candidates, packets, out)


def test_output_boundary_and_source_tampering_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "ROOT", tmp_path / "project")
    args = dataset()
    with pytest.raises(ValueError, match="outside project"):
        validation.build_validation(*args, tmp_path / "outside")
    args[2]["c00"]["refs"][0]["quote"] += " altered"
    with pytest.raises(ValueError, match="quote provenance"):
        validation.build_validation(*args, tmp_path / "project" / "validation")


def test_controls_have_interpretable_passes_but_no_real_accuracy():
    controls = validation._synthetic_controls()
    result = validation.evaluate_validation([expected_record(c) for c in controls], controls)
    assert result["complete"]
    assert result["synthetic_controls"]["joint_pass_rate"] == 1.0
    assert result["synthetic_controls"]["independent_scenario_count"] == 16
    assert result["real_source_pairs"]["accuracy"] is None
    assert result["real_source_pairs"]["semantic_citation_support_accuracy"] is None


def test_equivalent_duplicate_and_multi_citations_do_not_need_exact_id_match():
    case = next(c for c in validation._synthetic_controls() if c["case_id"] == "control:explicit_support:redundant_support")
    record = expected_record(case)
    for numbers in ([1], [2], [1, 2], [2, 1]):
        record["answer"]["evidence_numbers"] = numbers
        result = validation.evaluate_validation([record], [case])
        assert result["rows"][0]["control_passed"]
    record["answer"]["evidence_numbers"] = [1, 3]
    result = validation.evaluate_validation([record], [case])
    assert result["rows"][0]["citation_numbers_valid"]
    assert not result["rows"][0]["control_citation_support_correct"]


def test_permuted_sources_require_support_not_original_position():
    case = next(c for c in validation._synthetic_controls() if c["case_id"] == "control:explicit_support:citation_permutation")
    record = expected_record(case)
    assert record["answer"]["evidence_numbers"] == [2]
    record["answer"]["evidence_numbers"] = [1]
    result = validation.evaluate_validation([record], [case])
    assert result["rows"][0]["control_label_correct"]
    assert not result["rows"][0]["control_passed"]


@pytest.mark.parametrize("numbers", [[], [999], [1, 1]])
def test_known_labels_need_valid_nonduplicate_actual_source_numbers(numbers):
    case = validation._synthetic_controls()[0]
    record = expected_record(case)
    record["answer"]["evidence_numbers"] = numbers
    result = validation.evaluate_validation([record], [case])
    assert result["rows"][0]["schema_valid"]
    assert not result["rows"][0]["citation_numbers_valid"]
    assert not result["rows"][0]["control_passed"]


@pytest.mark.parametrize("value", [
    '{"i":1,"g":3,"a":"Y","e":[1]}',
    '[{"i":1,"g":true,"a":"Y","e":[1]}]',
    '[{"i":1,"g":3,"a":"Y","e":[true]}]',
    '[{"i":2,"g":3,"a":"Y","e":[1]}]',
    '[{"i":1,"g":3,"a":"Y","e":[1],"reason":"extra"}]',
    '```json\n[{"i":1,"g":3,"a":"Y","e":[1]}]\n```',
    '[{"i":1,"g":3,"a":"Y","e":[1]},{"i":1,"g":3,"a":"Y","e":[1]}]',
])
def test_raw_format_is_a_separate_strict_contract(value):
    case = validation._synthetic_controls()[0]
    result = validation.evaluate_validation([{"case_id": case["case_id"], "output": value}], [case])
    assert not result["rows"][0]["schema_valid"]


def test_raw_correct_json_is_accepted():
    case = validation._synthetic_controls()[0]
    result = validation.evaluate_validation([{"case_id": case["case_id"], "output": '[{"i":1,"g":3,"a":"Y","e":[1]}]'}], [case])
    assert result["rows"][0]["control_passed"]


def test_missing_outputs_are_not_removed_from_denominator():
    cases = validation._synthetic_controls()[:2]
    result = validation.evaluate_validation([expected_record(cases[0])], cases)
    assert not result["complete"]
    assert result["synthetic_controls"]["label_accuracy"] == 0.5
    assert result["rows"][1]["error"] == "missing_record"


def test_unknown_is_not_a_negative_and_no_source_means_no_citations():
    case = next(c for c in validation._synthetic_controls() if c["case_id"] == "control:not_studied_is_unknown:original")
    record = expected_record(case)
    record["answer"].update(grade=0, documented_applicability=False, evidence_numbers=[1])
    result = validation.evaluate_validation([record], [case])
    assert not result["rows"][0]["control_label_correct"]
    absent = next(c for c in validation._synthetic_controls() if c["case_id"] == "control:no_evidence:original")
    record = expected_record(absent)
    record["answer"]["evidence_numbers"] = [1]
    assert not validation.evaluate_validation([record], [absent])["rows"][0]["citation_numbers_valid"]


def test_injection_is_untrusted_and_cannot_be_cited_as_support():
    case = next(c for c in validation._synthetic_controls() if c["case_id"] == "control:prompt_injection_is_data:original")
    record = expected_record(case)
    record["answer"]["evidence_numbers"] = [2]
    result = validation.evaluate_validation([record], [case])
    assert result["rows"][0]["control_label_correct"]
    assert not result["rows"][0]["control_citation_support_correct"]


def test_duplicate_or_unrelated_records_are_never_silently_aggregated():
    case = validation._synthetic_controls()[0]
    record = expected_record(case)
    with pytest.raises(ValueError, match="duplicate"):
        validation.evaluate_validation([record, record], [case])
    record["case_id"] = "not-in-validation"
    with pytest.raises(ValueError, match="Unknown"):
        validation.evaluate_validation([record], [case])


def test_real_source_pair_cannot_be_given_generated_gold():
    case = deepcopy(validation._synthetic_controls()[0])
    case["kind"] = "real_source_pair"
    with pytest.raises(ValueError, match="must remain unlabelled"):
        validation.evaluate_validation([expected_record(case)], [case])


def test_admission_api_is_complete_and_joint_source_grounded():
    cases = validation._synthetic_controls()[:2]
    records = [expected_record(case) for case in cases]
    records[0]["answer"]["evidence_numbers"] = [2]  # legal, irrelevant citation
    result = validation.admission_metrics(validation.evaluate_validation(records, cases))
    assert result["complete"]
    assert result["controlled_semantic_accuracy"] == 0.5
    assert result["controlled_label_accuracy"] == 1.0
    assert result["controlled_citation_support_accuracy"] == 0.5
    assert result["raw_format_valid_fraction"] == 1.0
    assert result["citation_numbers_valid_fraction"] == 1.0
    assert result["scientific_gold_accuracy"] is None
    partial = validation.admission_metrics(validation.evaluate_validation(records[:1], cases))
    assert not partial["complete"]
    assert partial["raw_format_valid_fraction"] == 0.5
