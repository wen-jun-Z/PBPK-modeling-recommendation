"""Prospective corpus immutability, independent-role accounting and citations."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import frontier_judge_validation as v1
import frontier_judge_validation_v2 as v2


def dataset():
    qs, cs, packets = [], [], {}
    for i in range(30):
        drug, cid = f"drug-{i}", f"candidate-{i}"
        qs.append({"query_id": f"q{i}", "text": f"Find {drug}", "task_type": v1.TASKS[i % 5], "compounds": [drug], "features": {}})
        cs.append({"candidate_id": cid, "family_id": cid, "compounds": [drug]})
        text = f"Existing source of {drug}. " * (i + 1)
        body = {"documented_model_name": drug, "model_compounds": [drug], "scope_warning": "Existing material",
                "excerpts": [{"n": 1, "text": text}]}
        packets[cid] = {"body": body, "refs": [{"quote": text, "excerpt_sha256": v1._sha(text), "source_id": cid}],
                        "packet_sha256": v1._sha(v1._canonical(body)), "source_candidate_id": cid, "characters": len(text)}
    return qs, cs, packets


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(v1, "ROOT", tmp_path)
    monkeypatch.setattr(v2, "ROOT", tmp_path)
    monkeypatch.setattr(v2, "OLD", tmp_path / "v1")
    args = dataset()
    v1.build_validation(*args, v2.OLD)
    out = tmp_path / "v2"
    manifest = v2.build_validation_v2(*args, out)
    cases = [json.loads(s) for s in (out / "cases.jsonl").read_text().splitlines()]
    return args, out, manifest, cases


def oracle(case):
    expected = case["expected"]
    if expected is None:
        return {"case_id": case["case_id"], "answer": {"grade": None, "documented_applicability": None, "evidence_numbers": []}}
    return {"case_id": case["case_id"], "answer": {
        "grade": expected["grades"][0], "documented_applicability": expected["documented_applicability"],
        "evidence_numbers": sorted({group[0] for group in expected["required_support_any_of"]})}}


def test_old_material_and_gold_are_preserved_exactly(frozen):
    args, out, manifest, cases = frozen
    originals = {c["case_id"]: c for c in v2._old_inputs()[1]}
    inherited = [c for c in cases if c["strata"]["validation_role"] != "independent_admission"]
    assert len(inherited) == 124 and len(cases) == 188
    for case in inherited:
        clean = deepcopy(case)
        del clean["strata"]["validation_role"]
        assert clean == originals[case["case_id"]]
    assert v2.verify_validation_v2(*args, out) == manifest
    assert v2.build_validation_v2(*args, out) == manifest


def test_verifier_only_reads_and_does_not_rebuild_cases(frozen, monkeypatch):
    args, out, manifest, _ = frozen
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    def forbid(*args, **kwargs):
        raise AssertionError("Read-only verification tried rebuilding or writing")
    monkeypatch.setattr(v2, "_new_controls", forbid)
    monkeypatch.setattr(v2, "_write_once", forbid)
    assert v2.verify_validation_v2(*args, out) == manifest
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()} == before


def test_input_drift_and_case_corruption_fail(frozen):
    args, out, _, _ = frozen
    changed = deepcopy(args)
    changed[0][0]["text"] += " changed"
    with pytest.raises(ValueError, match="provenance mismatch"):
        v2.verify_validation_v2(*changed, out)
    path = out / "cases.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash/version mismatch"):
        v2.verify_validation_v2(*args, out)


def test_prompt_drift_is_not_hidden_by_a_previously_frozen_manifest(frozen, monkeypatch):
    args, out, _, _ = frozen
    monkeypatch.setattr(v2, "PROMPT_FILE_SHA256", "wrong")
    with pytest.raises(ValueError, match="prompt changed"):
        v2.verify_validation_v2(*args, out)


def test_all_roles_evaluate_separately_and_real_pairs_have_no_gold(frozen):
    _, _, _, cases = frozen
    records = [oracle(c) for c in cases]
    result = v2.evaluate_validation_v2(records, cases)
    assert result["complete"]
    assert result["admission_by_role"]["independent_admission"]["controlled_semantic_accuracy"] == 1
    assert result["admission_by_role"]["development_regression"]["controlled_semantic_accuracy"] == 1
    assert result["by_role"]["unlabelled_real_source"]["real_source_pairs"]["accuracy"] is None
    # Making the developmental controls fail cannot be concealed in a pooled
    # score or affect the independent denominator.
    first = next(r for r in records if r["case_id"].startswith("control:explicit_support:"))
    first["answer"]["grade"] = 0
    result = v2.evaluate_validation_v2(records, cases)
    assert result["admission_by_role"]["independent_admission"]["controlled_semantic_accuracy"] == 1
    assert result["admission_by_role"]["development_regression"]["controlled_semantic_accuracy"] == 63 / 64


def test_missing_independent_records_stay_in_full_denominator(frozen):
    _, _, _, cases = frozen
    records = [oracle(c) for c in cases if c["strata"]["validation_role"] != "independent_admission"]
    result = v2.evaluate_validation_v2(records, cases)
    assert not result["complete"]
    assert result["admission_by_role"]["independent_admission"]["controlled_semantic_accuracy"] == 0


def test_joint_sources_are_required_only_when_each_supplies_different_conditions():
    case = next(c for c in v2._new_controls() if c["case_id"] == "admission-v2:joint_main_use_and_scope:base")
    record = oracle(case)
    for citations in ([1], [2]):
        record["answer"]["evidence_numbers"] = citations
        assert not v1.evaluate_validation([record], [case])["rows"][0]["control_passed"]
    record["answer"]["evidence_numbers"] = [1, 2]
    assert v1.evaluate_validation([record], [case])["rows"][0]["control_passed"]


def test_alternative_sufficient_mechanism_citations_are_not_misclassified():
    case = next(c for c in v2._new_controls() if c["case_id"] == "admission-v2:other_compound_explicit_shared_template:base")
    for citations in ([1], [2], [1, 2]):
        record = oracle(case)
        record["answer"]["evidence_numbers"] = citations
        assert v1.evaluate_validation([record], [case])["rows"][0]["control_passed"]


def test_partial_grade_does_not_require_an_extra_citation_to_prove_unknown_applicability():
    case = next(c for c in v2._new_controls() if c["case_id"] == "admission-v2:documented_scope_but_adaptation_pending:base")
    record = oracle(case)
    assert record["answer"]["evidence_numbers"] == [1]
    assert v1.evaluate_validation([record], [case])["rows"][0]["control_passed"]


def test_reordering_and_duplicates_preserve_meaning_and_allow_redundant_support():
    scenarios = {}
    for case in v2._new_controls():
        name = case["strata"]["scenario"]
        old = scenarios.setdefault(name, case)
        assert case["expected"]["grades"] == old["expected"]["grades"]
        assert case["expected"]["documented_applicability"] is old["expected"]["documented_applicability"]
        assert {ref["quote"] for ref in case["packet"]["refs"]} == {ref["quote"] for ref in old["packet"]["refs"]}
        assert v1.evaluate_validation([oracle(case)], [case])["rows"][0]["control_passed"]


def test_builder_will_not_author_new_control_set_after_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "ROOT", tmp_path)
    out = tmp_path / "v2"
    out.mkdir()
    (out / "qwen_records.jsonl").write_text("already observed")
    with pytest.raises(ValueError, match="after v2 inference"):
        v2.build_validation_v2(*dataset(), out)


def test_new_control_ids_and_texts_are_not_simple_renames_of_old_set():
    new = v2._new_controls()
    old = v1._synthetic_controls()
    assert len({c["case_id"] for c in new}) == 64
    assert {c["case_id"] for c in old}.isdisjoint(c["case_id"] for c in new)
    old_text = {v1._canonical(c["packet"]["body"]["excerpts"]) for c in old}
    assert all(v1._canonical(c["packet"]["body"]["excerpts"]) not in old_text for c in new)
    assert {c["expected"]["grades"][0] for c in new} == {None, 0, 1, 2, 3}
    assert {c["query"]["task_type"] for c in new} == set(v1.TASKS)
