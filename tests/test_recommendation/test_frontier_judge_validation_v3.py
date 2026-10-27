"""Evidence attribution, frozen-source coverage, raw/final separation and windows."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import frontier_judge_validation as v1
import frontier_judge_validation_v2 as v2
import frontier_judge_validation_v3 as v3


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _dataset():
    qs = [{"query_id": f"q{i}", "text": f"Find documented prediction for drug{i}", "task_type": v1.TASKS[i % 5],
           "compounds": [f"drug{i}"], "features": {}} for i in range(30)]
    cs, packets = [], {}
    for i in range(247):
        cid, drug = f"c{i}", f"drug{i % 30}"
        text = f"Candidate {cid} documents plasma concentration predictions for {drug}. " * (i % 7 + 1)
        body = {"documented_model_name": cid, "model_compounds": [drug], "scope_warning": "Fixture source",
                "excerpts": [{"n": 1, "text": text}]}
        packets[cid] = {"body": body, "refs": [{"quote": text, "excerpt_sha256": v1._sha(text), "source_id": cid}],
                        "packet_sha256": v1._sha(v1._canonical(body)), "source_candidate_id": cid, "characters": len(text)}
        cs.append({"candidate_id": cid, "family_id": cid, "compounds": [drug]})
    return qs, cs, packets


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    for module in (v1, v2, v3): monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(v3, "OUT", tmp_path / "results")
    monkeypatch.setattr(v3, "PARENT", v3.OUT / "assessment_qwen36_v2/validation")
    monkeypatch.setattr(v3, "PRODUCTION", v3.OUT / "assessment_validated_v2")
    monkeypatch.setattr(v2, "OLD", v3.OUT / "assessment_qwen36/validation")
    args = _dataset()
    v1.build_validation(*args, v2.OLD)
    v2.build_validation_v2(*args, v3.PARENT)
    journal, cache = [], []
    for i in range(19):
        journal.append({"generation_id": f"g{i}", "query_ids": [f"q{i}"], "packet_sha256": args[2][f"c{i}"]["packet_sha256"]})
        cache.append({"generation_id": f"g{i}", "status": "invalid_output"})
    _jsonl(v3.PRODUCTION / "primary_generation_journal.jsonl", journal)
    _jsonl(v3.PRODUCTION / "primary_judgment_cache.jsonl", cache)
    out = v3.OUT / "v3/validation"
    manifest = v3.build_validation_v3(*args, out)
    return args, out, manifest, v3._rows(out / "cases.jsonl")


def oracle(case):
    e = case["expected"]
    answer = {"i": 1, "e": [] if e is None else sorted({group[0] for group in e["required_support_any_of"]}),
              "g": None if e is None else e["grades"][0], "a": "U" if e is None else {True: "Y", False: "N", None: "U"}[e["documented_applicability"]]}
    return {"case_id": case["case_id"], "output": json.dumps([answer])}


def test_complete_733_corpus_preserves_parents_and_no_real_gold(frozen):
    args, out, manifest, cases = frozen
    assert manifest["case_count"] == len(cases) == 733
    assert v3.verify_validation_v3(*args, out) == manifest
    parent = {c["case_id"]: c for c in v3._rows(v3.PARENT / "cases.jsonl")}
    for c in cases:
        if c["strata"]["validation_role"].startswith("development_"):
            clean = deepcopy(c)
            clean["strata"]["validation_role"] = clean["strata"].pop("prior_validation_role")
            assert clean == parent[c["case_id"]]
        if c["kind"] == "real_source_pair": assert c["expected"] is None


def test_readonly_verifier_does_not_rebuild_or_write(frozen, monkeypatch):
    args, out, manifest, _ = frozen
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    def fail(*args, **kwargs): raise AssertionError("Verifier generated or mutated material")
    monkeypatch.setattr(v3, "_new_controls", fail)
    monkeypatch.setattr(v3, "_representatives", fail)
    monkeypatch.setattr(v3, "_write", fail)
    assert v3.verify_validation_v3(*args, out) == manifest
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()} == before


def test_all247_packet_classes_get_two_distinct_queries(frozen):
    args, _, manifest, cases = frozen
    representatives = [c for c in cases if c["strata"]["validation_role"] == "prospective_source_coverage"]
    by_packet = {}
    for c in representatives: by_packet.setdefault(c["packet"]["packet_sha256"], []).append(c)
    assert len(by_packet) == 247
    assert all(len(group) == 2 and len({c["query"]["query_id"] for c in group}) == 2 for group in by_packet.values())
    assert set(manifest["representative_coverage"]["task_counts"]) == set(v1.TASKS)
    assert v3._representatives(list(reversed(args[0])), args[2], set()) == v3._representatives(args[0], args[2], set())


def test_previous_exposure_does_not_change_selected_real_pairs(frozen):
    args, _, _, _ = frozen
    unseen = v3._representatives(args[0], args[2], set())
    seen = {(c["packet"]["packet_sha256"], c["query"]["text"]) for c in unseen}
    observed = v3._representatives(args[0], args[2], seen)
    assert [c["case_id"] for c in observed] == [c["case_id"] for c in unseen]
    assert all(c["strata"]["initial_input_seen_before_v3"] for c in observed)
    assert all(not c["strata"]["initial_input_seen_before_v3"] for c in unseen)


def test_known_wrong_citation_is_rejected_without_requiring_exact_reference_list():
    case = next(c for c in v3._new_controls() if c["case_id"] == "admission-v3:background_candidate_is_not_current_candidate:base")
    good = oracle(case)
    for numbers, valid in (([1], False), ([2], True), ([1, 2], True), ([2, 3], False)):
        answer = json.loads(good["output"])
        answer[0]["e"] = numbers
        result = v1.evaluate_validation([{**good, "output": json.dumps(answer)}], [case])
        assert result["rows"][0]["control_passed"] is valid


def test_new_gold_compatible_with_parser_and_existing_scope_guard():
    cases = v3._new_controls()
    records = [oracle(c) for c in cases]
    assert all(r["control_passed"] for r in v1.evaluate_validation(records, cases)["rows"])
    replay = v3._replay(records, cases)
    assert replay["final_valid_fraction"] == 1, [r for r in replay["rows"] if not r["final_valid"]]


def test_permutations_change_only_reference_order_and_indices():
    groups = {}
    for c in v3._new_controls(): groups.setdefault(c["strata"]["scenario"], []).append(c)
    assert len(groups) == 16
    for first, second in groups.values():
        assert first["query"]["text"] == second["query"]["text"]
        assert first["expected"]["grades"] == second["expected"]["grades"]
        assert first["expected"]["documented_applicability"] is second["expected"]["documented_applicability"]
        assert {r["quote"] for r in first["packet"]["refs"]} == {r["quote"] for r in second["packet"]["refs"]}


def test_raw_failures_do_not_disappear_when_repaired(frozen):
    _, _, _, cases = frozen
    records = [oracle(c) for c in cases]
    bad_id = next(c["case_id"] for c in cases if c["strata"]["validation_role"] == "prospective_source_coverage")
    row = next(r for r in records if r["case_id"] == bad_id)
    row["final_output"] = row["output"]
    row["output"] = '[{"i":1,"e":[],"g":0,"a":"N"}]'
    result = v3.evaluate_validation_v3(records, cases)
    assert result["admission_by_role"]["prospective_source_coverage"]["citation_numbers_valid_fraction"] == 493 / 494
    assert result["final_admission_by_role"]["prospective_source_coverage"]["citation_numbers_valid_fraction"] == 1
    assert result["final_validation_by_role"]["prospective_source_coverage"]["final_valid_fraction"] == 1


def test_replay_does_not_trust_reported_success():
    case = v3._new_controls()[0]
    record = {"case_id": case["case_id"], "output": '[{"i":1,"e":[],"g":0,"a":"N"}]', "final_status": "valid"}
    replay = v3._replay([record], [case])
    assert not replay["rows"][0]["final_valid"]
    assert not replay["rows"][0]["reported_status_agrees"]


def test_frozen_window_detects_cluster_even_when_global_rate_is_high(frozen):
    _, _, _, cases = frozen
    records = [oracle(c) for c in cases]
    coverage_ids = [c["case_id"] for c in cases if c["strata"]["validation_role"] == "prospective_source_coverage"]
    bad = set(coverage_ids[80:93])  # 13/128 exceeds .10; never dilute by reordering
    for row in records:
        if row["case_id"] in bad: row["output"] = '[{"i":1,"e":[],"g":0,"a":"N"}]'
    report = v3.evaluate_validation_v3(list(reversed(records)), cases)
    assert report["coverage_sequence"]["case_ids"] == coverage_ids
    assert report["coverage_sequence"]["max_invalid_in_full_window"] == 13
    assert report["coverage_sequence"]["any_full_window_above10percent"]


def test_frozen_history_cannot_be_rewritten(frozen):
    args, out, _, _ = frozen
    p = v3.PRODUCTION / "primary_judgment_cache.jsonl"
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match="Historical source/exposure"):
        v3.verify_validation_v3(*args, out)


def test_builder_refuses_after_any_v3_output(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "ROOT", tmp_path)
    out = tmp_path / "validation"; out.mkdir()
    (out / "phi_records.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="after v3 outputs"):
        v3.build_validation_v3(*_dataset(), out)
