"""Source-proven identity aliases must bind consistently without changing facts."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pbpk_kb.storage.expansion_identity_mapping import (
    STATE, TARGETS, ExpansionIdentityMapping, load_expansion_identity_mapping,
)
from pbpk_kb.storage.import_processed_evidence import CopyWriter, EvidenceImporter


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in value) + "\n" if lines else json.dumps(value))


@pytest.fixture
def source_case(tmp_path):
    run = "run_20261009T140920Z"
    raw = tmp_path / "data_process/raw_processing" / run
    final = tmp_path / "data_process/final_evidence" / run
    source = tmp_path / "artifacts/synonyms.json"
    write(source, {"InformationList": {"Information": [{"CID": 123, "Synonym": ["Drug A", "drug-alias"]}]}})
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    target = {"compound_name": "drug a", "search_name": "drug-alias", "batch_id": "batch_02",
              "aliases": ["Drug A", "drug-alias"], "existing_compound_id": "registered"}
    write(tmp_path / TARGETS, {"targets": [target]})
    write(tmp_path / STATE, {"planned_batches": [{"run_id": run, "batch_id": "batch_02", "compounds": ["drug-alias"]}],
                           "frozen_inputs": {str(TARGETS): hashlib.sha256((tmp_path / TARGETS).read_bytes()).hexdigest()}})
    identity = {"compound_id": "new-alias", "preferred_name": "drug-alias", "pubchem_cid": "123",
                "pubchem_cids": ["123"], "inchikey": "EXACT-CHEMICAL", "identity_provenance": [
                    {"source_id": "pubchem", "assertion_type": "pubchem_synonym", "assertion_value": "Drug A",
                     "raw_asset_id": "asset:one", "checksum": checksum}]}
    identity_path = final / "01_compound_attribution/compound_identity.jsonl"
    write(identity_path, [identity], lines=True)
    inventory = raw / "01_raw_inventory/raw_inventory.jsonl"
    write(inventory, [{"raw_asset_id": "asset:one", "source_id": "pubchem", "checksum": checksum,
                       "storage_path": str(source)}], lines=True)
    registry = [{"compound_id": "registered", "preferred_name": "Drug A", "normalized_name": "drug a",
                 "pubchem_cid": None, "chembl_id": None, "inchi_key": None}]
    def load():
        return load_expansion_identity_mapping(final, raw, lambda ids: copy.deepcopy(registry), root=tmp_path)
    return SimpleNamespace(root=tmp_path, raw=raw, final=final, source=source, identity=identity,
                           identity_path=identity_path, inventory=inventory, registry=registry, load=load)


def test_explicit_source_proven_mapping_preserves_all_input_files(source_case):
    c = source_case
    before = {p: p.read_bytes() for p in c.root.rglob("*") if p.is_file()}
    m = c.load()
    assert m.ids == {"new-alias": "registered"}
    assert m.receipt["mappings"][0]["proof"]["pubchem_cid"] == "123"
    assert m.receipt["mappings"][0]["original_compound_id"] == "new-alias"
    assert {p: p.read_bytes() for p in before} == before


def _refreeze_test_target(c, target):
    write(c.root / TARGETS, {"targets": [target]})
    state = json.loads((c.root / STATE).read_text())
    state["planned_batches"][0]["compounds"] = [target["search_name"]]
    state["frozen_inputs"][str(TARGETS)] = hashlib.sha256((c.root / TARGETS).read_bytes()).hexdigest()
    write(c.root / STATE, state)


def _replace_test_synonyms(c, entries):
    write(c.source, {"InformationList": {"Information": entries}})
    checksum = hashlib.sha256(c.source.read_bytes()).hexdigest()
    c.identity["identity_provenance"][0]["checksum"] = checksum
    write(c.identity_path, [c.identity], lines=True)
    inventory = [json.loads(line) for line in c.inventory.read_text().splitlines()]
    inventory[0]["checksum"] = checksum
    write(c.inventory, inventory, lines=True)


@pytest.fixture
def registered_spelling_case(source_case):
    c = source_case
    target = json.loads((c.root / TARGETS).read_text())["targets"][0]
    target.update(compound_name="drug a", search_name="drug a", aliases=["Drug-a", "drug-a"])
    _refreeze_test_target(c, target)
    c.identity["preferred_name"] = "drug a"
    _replace_test_synonyms(c, [{"CID": 123, "Synonym": ["Drug A", "drug-a"]}])
    c.registry[0].update(preferred_name="Drug-a", normalized_name="drug-a",
                         pubchem_cid="123", inchi_key="EXACT-CHEMICAL")
    return c


def test_frozen_registered_alias_uses_exact_same_cid_source_without_rewriting(registered_spelling_case):
    c = registered_spelling_case
    before = {p: p.read_bytes() for p in c.root.rglob("*") if p.is_file()}
    m = c.load()
    assert m.ids == {"new-alias": "registered"}
    proof = m.receipt["mappings"][0]["proof"]
    assert proof["registered_alias"] == "drug-a"
    assert proof["search_alias"] == proof["canonical_alias"] == "drug a"
    assert proof["pubchem_cid"] == "123"
    assert proof["sha256"] == hashlib.sha256(c.source.read_bytes()).hexdigest()
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("failure", ["missing_source_alias", "alias_in_different_cid", "unlisted_registry_name",
    "multiple_registered_aliases", "conflicting_cid", "conflicting_inchikey", "already_imported_alias",
    "punctuation_not_explicit", "changed_source"])
def test_registered_spelling_branch_retains_strict_identity_guards(registered_spelling_case, failure):
    c = registered_spelling_case
    if failure == "missing_source_alias":
        _replace_test_synonyms(c, [{"CID": 123, "Synonym": ["Drug A"]}])
    elif failure == "alias_in_different_cid":
        _replace_test_synonyms(c, [{"CID": 123, "Synonym": ["Drug A"]},
                                  {"CID": 999, "Synonym": ["drug-a"]}])
    elif failure == "unlisted_registry_name":
        c.registry[0].update(preferred_name="Unlisted drug", normalized_name="unlisted drug")
    elif failure in {"multiple_registered_aliases", "punctuation_not_explicit"}:
        target = json.loads((c.root / TARGETS).read_text())["targets"][0]
        target["aliases"] = ["drug_a", "drug-a"] if failure == "multiple_registered_aliases" else ["drug_a"]
        _refreeze_test_target(c, target)
        if failure == "multiple_registered_aliases":
            c.registry[0]["normalized_name"] = "drug_a"
    elif failure == "conflicting_cid":
        c.registry[0]["pubchem_cid"] = "999"
    elif failure == "conflicting_inchikey":
        c.registry[0]["inchi_key"] = "DIFFERENT-CHEMICAL-FORM"
    elif failure == "already_imported_alias":
        c.registry.append({"compound_id": "new-alias"})
    elif failure == "changed_source":
        c.source.write_text("{}")
    with pytest.raises(RuntimeError):
        c.load()


@pytest.mark.parametrize("failure", ["unfrozen_targets", "not_explicit", "missing_registered", "already_imported_alias",
    "conflicting_stable_id", "different_CID", "missing_synonym", "changed_source", "ambiguous_source_identity"])
def test_identity_alias_binding_fails_closed(source_case, failure):
    c = source_case
    if failure == "unfrozen_targets":
        with (c.root / TARGETS).open("a") as stream: stream.write(" ")
    elif failure == "not_explicit":
        plan = json.loads((c.root / TARGETS).read_text()); plan["targets"][0]["aliases"] = ["drug-alias"]
        write(c.root / TARGETS, plan)
        state = json.loads((c.root / STATE).read_text())
        state["frozen_inputs"][str(TARGETS)] = hashlib.sha256((c.root / TARGETS).read_bytes()).hexdigest()
        write(c.root / STATE, state)
    elif failure == "missing_registered": c.registry.clear()
    elif failure == "already_imported_alias": c.registry.append({"compound_id": "new-alias"})
    elif failure == "conflicting_stable_id": c.registry[0]["inchi_key"] = "DIFFERENT-SALT"
    elif failure in {"different_CID", "missing_synonym"}:
        payload = {"InformationList": {"Information": [{"CID": 999 if failure == "different_CID" else 123,
                    "Synonym": ["drug-alias"] if failure == "missing_synonym" else ["Drug A", "drug-alias"]}]}}
        write(c.source, payload)
        checksum = hashlib.sha256(c.source.read_bytes()).hexdigest()
        c.identity["identity_provenance"][0]["checksum"] = checksum
        write(c.identity_path, [c.identity], lines=True)
        rows = [json.loads(line) for line in c.inventory.read_text().splitlines()]
        rows[0]["checksum"] = checksum; write(c.inventory, rows, lines=True)
    elif failure == "changed_source": c.source.write_text("{}")
    else: write(c.identity_path, [c.identity, c.identity], lines=True)
    with pytest.raises(RuntimeError): c.load()


def test_unregistered_run_never_opens_registry(source_case):
    c = source_case
    def forbidden(ids): raise AssertionError("Unexpected DB lookup")
    mapping = load_expansion_identity_mapping(c.final, c.raw.parent / "another-run", forbidden, root=c.root)
    assert mapping.ids == {}


def test_existing_exact_binding_needs_no_alias_or_db_lookup(source_case):
    c = source_case
    c.identity["compound_id"] = "registered"; write(c.identity_path, [c.identity], lines=True)
    def forbidden(ids): raise AssertionError("Unexpected DB lookup")
    assert not load_expansion_identity_mapping(c.final, c.raw, forbidden, root=c.root).ids


def test_all_structured_identity_fields_map_without_rewriting_quotes(source_case):
    m = source_case.load()
    row = {"compound_id": "new-alias", "attributed_compound_id": "new-alias",
           "victim_compound_id": "new-alias", "perpetrator_compound_id": "other-drug",
           "source_text": "The label literally says new-alias and drug-alias.", "value": 1.5,
           "condition": "after drug-alias 10 mg", "source_text_hash": "unchanged",
           "attribution": {"compound_id": "new-alias"},
           "source_record": json.dumps({"compound_id": "new-alias", "source_text": "new-alias"}),
           "original_compound_id": "new-alias"}
    original = copy.deepcopy(row); mapped = m.record(row)
    assert row == original
    assert mapped["compound_id"] == mapped["attributed_compound_id"] == mapped["victim_compound_id"] == "registered"
    assert mapped["attribution"]["compound_id"] == "registered"
    assert json.loads(mapped["source_record"]) == {"compound_id": "registered", "source_text": "new-alias"}
    for key in ("source_text", "condition", "value", "source_text_hash", "original_compound_id", "perpetrator_compound_id"):
        assert mapped[key] == original[key]
    assert m.record(mapped) == mapped


def test_copy_boundary_maps_rag_graph_and_typed_rows(source_case):
    importer = object.__new__(EvidenceImporter); importer.import_identity_mapping = source_case.load()
    seen = []
    writer = object.__new__(CopyWriter); writer.importer = importer; writer.rows_written = 0
    writer.columns = ["compound_id", "victim_compound_id", "metadata_json", "source_text"]
    writer.native_copy = SimpleNamespace(write_row=lambda row: seen.append(row))
    writer.write({"compound_id": "new-alias", "victim_compound_id": "new-alias",
                  "metadata_json": json.dumps({"compound_id": "new-alias"}), "source_text": "new-alias"})
    assert seen[0][:2] == ["registered", "registered"]
    assert json.loads(seen[0][2])["compound_id"] == "registered"
    assert seen[0][3] == "new-alias"


def test_normalized_serving_json_and_reconciliation_use_same_identity(source_case):
    importer = object.__new__(EvidenceImporter); importer.import_identity_mapping = source_case.load()
    importer.final_attributions = {"e": {"compound_name": "drug-alias", "compound_id": "new-alias", "compound_role": "target_compound"}}
    importer.compound_ids = {"drug-alias": "new-alias"}
    row = {"evidence_id": "e", "run_id": "run", "evidence_type": "clinical_pk", "raw_asset_id": "asset:one",
           "source_text": "drug-alias was 1.5 h", "source_location": "quote", "storage_path": str(source_case.source),
           "source_id": "pubchem", "value": 1.5, "parameter": "half_life", "unit": "h",
           "attributed_compound_id": "new-alias", "condition": "unchanged"}
    normalized = importer._normalize_evidence_row(row)
    assert normalized["compound_id"] == normalized["attributed_compound_id"] == "registered"
    assert normalized["attribution"]["compound_id"] == "registered"
    candidate = importer._candidate_row(normalized)
    assert candidate["compound_id"] == json.loads(candidate["source_record"])["compound_id"] == "registered"
    assert json.loads(candidate["source_record"])["condition"] == "unchanged"
