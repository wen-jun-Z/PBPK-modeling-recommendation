"""Acceptance checks must not pass using old hits, missing links or outside paths."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/deployment/verify_expansion_batch.py"
spec = importlib.util.spec_from_file_location("verify_expansion_batch", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_project_boundary_rejects_traversal_and_external_symlink(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    monkeypatch.setattr(module, "ROOT", root)
    assert module._inside("file") == root / "file"
    for value in ["../file", root / "escape/outside"]:
        with pytest.raises(RuntimeError, match="escapes"):
            module._inside(value)


@pytest.mark.parametrize("available,linked", [({"old"}, {"new"}), ({"new"}, {"old"})])
def test_missing_serving_or_rag_link_fails(available, linked):
    with pytest.raises(RuntimeError, match="missing from serving/index"):
        module._check_final_ids({"new"}, available, linked)


def test_existing_extra_evidence_does_not_inflate_current_import_count():
    result = module._check_final_ids({"new"}, {"old", "new"}, {"old", "new"})
    assert result == {"expected": 1, "available": 1, "missing": 0, "missing_rag_evidence_links": 0}


def test_provenance_requires_matching_registry_and_file_checksum(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "raw.json"
    source.write_text('{"property": 1}')
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    detail = {"extracted_fields": {"evidence_id": "e1", "source_location": "$.property",
                                   "storage_path": str(source), "checksum": checksum},
              "source_document": {"raw_storage_path": str(source), "raw_checksum": checksum}}
    assert module._verify_source_detail(detail, {}) == source
    detail["source_document"]["raw_checksum"] = "wrong"
    with pytest.raises(RuntimeError, match="disagree"):
        module._verify_source_detail(detail, {})
    detail["source_document"]["raw_checksum"] = checksum
    source.write_text('{"property": 2}')
    with pytest.raises(RuntimeError, match="checksum differs"):
        module._verify_source_detail(detail, {})


def test_registered_id_survives_spelling_alias_but_not_identity_conflict():
    target = {"compound_name": "mefenamic acid", "existing_compound_id": "registered",
              "aliases": ["Mefenamic-acid"]}
    rows = [{"attributed_compound_id": "registered", "attributed_compound_name": "Mefenamic-acid"},
            {"attributed_compound_id": "wrong", "attributed_compound_name": "mefenamic acid"}]
    assert module._target_rows(rows, target) == rows[:1]
    new = {"compound_name": "aciclovir", "search_name": "acyclovir", "aliases": []}
    assert module._target_rows([{"compound_name": "Acyclovir"}], new)


def test_alias_acceptance_requires_matching_import_receipt_and_registered_id():
    from pbpk_kb.storage.expansion_identity_mapping import ExpansionIdentityMapping
    receipt = {"version": 1, "mappings": [{"original_compound_id": "source-alias",
                 "registered_compound_id": "registered"}], "targets_sha256": "frozen"}
    mapping = ExpansionIdentityMapping(receipt)
    target = {"compound_name": "canonical", "search_name": "alias", "existing_compound_id": "registered"}
    rows = [{"attributed_compound_id": "source-alias", "compound_name": "alias", "value": 1.5},
            {"attributed_compound_id": "wrong", "compound_name": "alias", "value": 3.0}]
    assert module._target_rows(rows, target) == []
    viewed = module._import_identity_view(rows, {"identity_mapping": receipt}, mapping)
    assert module._target_rows(viewed, target) == [viewed[0]]
    assert rows[0]["attributed_compound_id"] == "source-alias"
    for imported in ({}, {"identity_mapping": {**receipt, "targets_sha256": "modified"}},
                     {"identity_mapping": {"version": 1, "mappings": []}}):
        with pytest.raises(RuntimeError, match="identity mapping differs"):
            module._import_identity_view(rows, imported, mapping)


def test_semantic_results_from_another_run_do_not_pass():
    hits = [{"retrieval_record_id": "old", "score": 1.0}]
    with pytest.raises(RuntimeError, match="no current-run record"):
        module._verify_semantic_hit(hits, {"new"})
    result = module._verify_semantic_hit(hits + [{"retrieval_record_id": "new", "score": .9}], {"new"})
    assert result["current_run_matches"] == 1
    assert result["matched_retrieval_record_id"] == "new"


def test_frozen_batch_accepts_search_spelling_without_changing_identity():
    target = {"compound_name": "aciclovir", "search_name": "acyclovir", "existing_compound_id": "registered"}
    other = {"compound_name": "pregabalin", "search_name": "pregabalin"}
    plan = {"targets": [other, target], "batches": [{"compound_names": ["pregabalin", "aciclovir"]}]}
    assert module._batch_targets(["pregabalin", "acyclovir"], plan) == [other, target]
    assert module._batch_targets(["pregabalin", "aciclovir"], plan) == [other, target]
    for invalid in (["pregabalin", "other"], ["acyclovir", "pregabalin"],
                    ["pregabalin", "pregabalin"], ["acyclovir"]):
        with pytest.raises(RuntimeError, match="registered expansion targets"):
            module._batch_targets(invalid, plan)
    assert not module._target_rows([{"attributed_compound_id": "wrong", "compound_name": "acyclovir"}], target)


def _remote_embed_query(text):
    raise AssertionError("The network embedding service must not be contacted")


def test_local_encoder_injection_does_not_modify_original_search_globals():
    namespace = {"embed_query": _remote_embed_query}
    exec("def search(q, source_id=None, limit=10, db=None):\n    return embed_query(q)\n", namespace)
    original = namespace["search"]
    local = module._semantic_with_local_vectors(original, {"current": [1.0, 0.0]})
    assert local(q="current", source_id="source", limit=100, db=object()) == [1.0, 0.0]
    with pytest.raises(AssertionError, match="network"):
        original("current")


@pytest.mark.parametrize("run_id", ["../../other", "run_20261009", "run_20261009T140919Z/other"])
def test_run_id_cannot_escape_registered_paths(run_id):
    with pytest.raises(ValueError, match="identifier"):
        module.verify_batch(run_id)
