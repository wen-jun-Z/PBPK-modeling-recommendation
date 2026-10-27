from copy import deepcopy
import json

import pytest

from pbpk_kb.blueprints.qwen_proposals import (
    GENERATION, REPO, REVISION, VERSION, build_request, digest, file_sha,
    load_proposer, parse_response, project_path, read_frozen_record, validate_request,
)


def data():
    case = {"case_id": "drug_development", "compound_id": "drug",
            "requirement": {"compound_semantic_id": "drug", "name": "Drug", "route": "oral",
                            "dose": 5, "dose_unit": "mg", "targets": [{"target_id": "plasma", "matrix": "plasma"}]},
            "parameter_search": {"parameters": [
                {"parameter_id": "CL", "initial_value": 1., "lower_bound": 0., "upper_bound": 10.,
                 "unit": "L/h", "source": {"source_checksum": "a" * 64, "source_id": "forbidden_path"}}]},
            "scenarios": [{"role": "selection", "observations": [{"concentration": 999999}]}],
            "source_audit": [{"path": "secret_test_outcome.json"}]}
    candidates = [{"candidate_id": "standard", "structure_id": "standard", "free_parameter_ids": ["CL"],
                   "structure_edit_count": 0, "selection_score": -100, "features": [999]}]
    return case, candidates


def test_whitelist_excludes_outcomes_features_and_source_paths():
    case, candidates = data()
    request = build_request(case, candidates)
    rendered = json.dumps(request)
    for value in ["observations", "999999", "secret_test", "forbidden_path", "selection_score", "features"]:
        assert value not in rendered
    assert request["parameter_contracts"][0]["initial_value"] == 1.
    assert request["source_refs"][0]["content_supplied"] is False


@pytest.mark.parametrize("location", ["root", "requirement", "candidate", "parameter", "ref", "target"])
def test_direct_payload_cannot_smuggle_outcome_fields(location):
    request = build_request(*data())
    target = {"root": request, "requirement": request["requirement"], "candidate": request["candidates"][0],
              "parameter": request["parameter_contracts"][0], "ref": request["source_refs"][0],
              "target": request["requirement"]["targets"][0]}[location]
    target["selection_log_rmse"] = 0.1
    with pytest.raises(ValueError, match="Unexpected fields"):
        validate_request(request)


@pytest.mark.parametrize("raw", [
    '{"candidate_id":"invented","rationale":"x"}',
    '{"candidate_id":"standard","rationale":"x","score":1}',
    '{"candidate_id":"standard","candidate_id":"invented","rationale":"x"}',
    '```json\n{"candidate_id":"standard","rationale":"x"}\n```',
    '{"candidate_id":"standard","rationale":""}',
])
def test_invalid_answer_never_falls_back(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_response(raw, build_request(*data()))


def fixture_records(tmp_path):
    case, candidates = data()
    request = build_request(case, candidates)
    directory = tmp_path / "proposals"
    relative = "cases/drug_development/seed_17.json"
    path = directory / relative
    path.parent.mkdir(parents=True)
    record = {"case_id": case["case_id"], "seed": 17, "status": "completed", "candidate_id": "standard",
              "weights_manifest_sha256": "b" * 64, "request_sha256": digest(request),
              "raw_response": '{"candidate_id":"standard","rationale":"source-parameter subset"}'}
    path.write_text(json.dumps(record))
    manifest = {"version": VERSION, "model_repo": REPO, "model_revision": REVISION, "generation": GENERATION,
                "final_status": "completed",
                "requests": {case["case_id"]: digest(request)}, "weights_manifest_sha256": "b" * 64,
                "record_sha256": {relative: file_sha(path)}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, path, case, candidates


def test_frozen_callback_checks_contract_and_raw_generation(tmp_path):
    directory, _, case, candidates = fixture_records(tmp_path)
    proposer = load_proposer(directory, case, candidates, project_root=tmp_path)
    assert proposer(candidates, seed=17)["candidate_id"] == "standard"
    changed = deepcopy(candidates)
    changed[0]["free_parameter_ids"] = []
    with pytest.raises(ValueError, match="contract changed"):
        proposer(changed, seed=17)


def test_record_mutation_detected(tmp_path):
    directory, path, case, candidates = fixture_records(tmp_path)
    proposer = load_proposer(directory, case, candidates, project_root=tmp_path)
    path.write_text(path.read_text().replace("standard", "invented"))
    with pytest.raises(ValueError, match="absent or changed"):
        proposer(candidates, seed=17)


def test_failed_generation_is_never_replaced_by_a_candidate(tmp_path):
    directory, path, case, candidates = fixture_records(tmp_path)
    record = json.loads(path.read_text())
    record.update(status="invalid_response", error_type="JSONDecodeError", raw_response="{")
    path.write_text(json.dumps(record))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["record_sha256"][str(path.relative_to(directory))] = file_sha(path)
    manifest_path.write_text(json.dumps(manifest))
    proposer = load_proposer(directory, case, candidates, project_root=tmp_path)
    with pytest.raises(ValueError, match="proposal failed"):
        proposer(candidates, seed=17)


def test_source_parameter_change_rejects_stale_proposal(tmp_path):
    directory, _, case, candidates = fixture_records(tmp_path)
    case["parameter_search"]["parameters"][0]["upper_bound"] = 20.
    with pytest.raises(ValueError, match="do not match"):
        load_proposer(directory, case, candidates, project_root=tmp_path)


def test_common_search_order_normalization_does_not_change_contract(tmp_path):
    directory, path, case, candidates = fixture_records(tmp_path)
    second = deepcopy(case["parameter_search"]["parameters"][0])
    second["parameter_id"] = "AA"
    case["parameter_search"]["parameters"].append(second)
    candidates[0]["free_parameter_ids"] = ["CL", "AA"]
    request = build_request(case, candidates)
    record = json.loads(path.read_text())
    record["request_sha256"] = digest(request)
    path.write_text(json.dumps(record))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["requests"][case["case_id"]] = digest(request)
    manifest["record_sha256"][str(path.relative_to(directory))] = file_sha(path)
    manifest_path.write_text(json.dumps(manifest))
    proposer = load_proposer(directory, case, candidates, project_root=tmp_path)
    normalized = deepcopy(candidates)
    normalized[0]["free_parameter_ids"].sort()
    assert proposer(normalized, seed=17)["candidate_id"] == "standard"


def test_project_scope_rejects_escape_before_read(tmp_path):
    with pytest.raises(ValueError, match="leaves the project"):
        project_path(tmp_path.parent, tmp_path)


@pytest.mark.parametrize("key,value", [
    ("formulation", {"selection_outcomes": [1, 2]}),
    ("dose", float("nan")),
    ("dose", True),
    ("targets", [{"matrix": {"observations": [1]}}]),
])
def test_scalar_contracts_cannot_embed_outcome_objects(key, value):
    request = build_request(*data())
    request["requirement"][key] = value
    with pytest.raises(ValueError):
        validate_request(request)


def test_cross_batch_reuse_preserves_raw_record_and_provenance(tmp_path):
    directory, path, case, candidates = fixture_records(tmp_path)
    record, origin = read_frozen_record(directory, build_request(case, candidates), seed=17, project_root=tmp_path)
    assert record == json.loads(path.read_text())
    assert origin["source_record_sha256"] == file_sha(path)
    changed = build_request(case, candidates)
    changed["requirement"]["dose"] = 7.
    with pytest.raises(ValueError, match="contract differs"):
        read_frozen_record(directory, changed, seed=17, project_root=tmp_path)
