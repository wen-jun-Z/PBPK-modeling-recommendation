import hashlib
import json
from pathlib import Path

from pbpk_kb.processing.final_evidence_attribution_stage import (
    ProvenanceAttributionStage,
    requires_public_model_role_review,
)
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def candidate(
    evidence_id: str,
    source_text: str,
    *,
    source_id: str,
    raw_asset_id: str,
    checksum: str,
    storage_path: str,
    parsed_record_id: str | None = None,
    parsed_document_id: str | None = None,
    chunk_id: str | None = None,
    source_text_hash: str | None = None,
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "run_id": "run-test",
        "compound": "midazolam",
        "evidence_type": "clinical_pk" if source_id != "pubchem" else "physicochemical",
        "parameter": "clearance" if source_id != "pubchem" else "CID",
        "value": None,
        "value_text": source_text,
        "unit": None,
        "source_id": source_id,
        "raw_asset_id": raw_asset_id,
        "storage_path": storage_path,
        "checksum": checksum,
        "source_location": f"{source_id}:fixture#item={evidence_id}",
        "source_text": source_text,
        "source_text_hash": source_text_hash or hashlib.sha256(source_text.encode()).hexdigest(),
        "extraction_method": "source_structured_field" if source_id == "pubchem" else "rule_based",
        "extractor_name": "fixture_extractor",
        "extractor_version": "fixture-v1",
        "curation_status": "needs_review",
        "parsed_record_id": parsed_record_id,
        "parsed_document_id": parsed_document_id,
        "parsed_table_id": None,
        "chunk_id": chunk_id,
        "metadata_json": {},
        "condition": None,
        "population": None,
        "route": None,
        "dose": None,
    }


def build_fixture(run_dir: Path) -> None:
    layout = raw_processing_layout(run_dir)
    layout.ensure_dirs()
    write_json(
        layout.path("pipeline_state.json", for_write=True),
        {
            "run_id": "run-test",
            "stages": {
                name: {"status": "completed", "accepted": True}
                for name in (
                    "inventory",
                    "parse",
                    "deterministic_extract",
                    "llm_extract",
                    "table_recovery",
                    "completeness",
                )
            }
            | {"embedding": {"status": "not_started"}},
        },
    )

    assets = [
        {
            "raw_asset_id": "asset-pubchem",
            "source_id": "pubchem",
            "checksum": "checksum-pubchem",
            "storage_path": "/raw/pubchem.json",
        },
        {
            "raw_asset_id": "asset-label",
            "source_id": "dailymed",
            "checksum": "checksum-label",
            "storage_path": "/raw/label.xml",
        },
        {"raw_asset_id": "asset-warfarin", "compound": "warfarin", "source_id": "pubchem",
         "checksum": "checksum-warfarin", "storage_path": "/raw/warfarin.json"},
    ]
    write_jsonl(layout.path("raw_inventory.jsonl", for_write=True), assets)

    records = [
        {
            "parsed_record_id": "record-properties",
            "compound": "midazolam",
            "source_id": "pubchem",
            "raw_asset_id": "asset-pubchem",
            "checksum": "checksum-pubchem",
            "storage_path": "/raw/pubchem.json",
            "source_location": "pubchem:fixture#json_path=$",
            "metadata_json": {"request_key": "properties_by_name"},
            "parser_name": "json_parser",
            "record_json": {
                "PropertyTable": {
                    "Properties": [{"CID": 4192, "InChIKey": "DDLIGBOFAVUZHB-UHFFFAOYSA-N"}]
                }
            },
        },
        {
            "parsed_record_id": "record-synonyms",
            "compound": "midazolam",
            "source_id": "pubchem",
            "raw_asset_id": "asset-pubchem",
            "checksum": "checksum-pubchem",
            "storage_path": "/raw/pubchem.json",
            "source_location": "pubchem:fixture#synonyms",
            "metadata_json": {"request_key": "synonyms_by_name"},
            "parser_name": "json_parser",
            "record_json": {"InformationList": {"Information": [{"Synonym": ["Midazolam"]}]}},
        },
        {
            "parsed_record_id": "record-rxnorm",
            "compound": "midazolam",
            "source_id": "rxnorm",
            "raw_asset_id": "asset-pubchem",
            "checksum": "checksum-pubchem",
            "storage_path": "/raw/pubchem.json",
            "source_location": "rxnorm:fixture#rxcui",
            "metadata_json": {"request_key": "rxcui_by_name"},
            "parser_name": "json_parser",
            "record_json": {"idGroup": {"rxnormId": ["6960"]}},
        },
        {
            "parsed_record_id": "record-chembl",
            "compound": "midazolam",
            "source_id": "chembl",
            "raw_asset_id": "asset-pubchem",
            "checksum": "checksum-pubchem",
            "storage_path": "/raw/pubchem.json",
            "source_location": "chembl:fixture#molecule",
            "metadata_json": {"request_key": "molecule_search"},
            "parser_name": "json_parser",
            "record_json": {
                "molecules": [
                    {
                        "pref_name": "MIDAZOLAM",
                        "molecule_chembl_id": "CHEMBL655",
                        "molecule_structures": {"standard_inchi_key": "DDLIGBOFAVUZHB-UHFFFAOYSA-N"},
                    }
                ]
            },
        },
        {
            "parsed_record_id": "record-label-identity",
            "compound": "midazolam",
            "source_id": "dailymed",
            "raw_asset_id": "asset-label",
            "checksum": "checksum-label",
            "storage_path": "/raw/label.xml",
            "source_location": "dailymed:fixture#document",
            "metadata_json": {},
            "parser_name": "dailymed_spl_parser",
            "record_json": {
                "active_moieties": [{"name": "MIDAZOLAM", "unii": "R60L0SM5BC"}],
                "active_ingredients": [
                    {"name": "MIDAZOLAM HYDROCHLORIDE", "unii": "W7TTW573JJ"},
                    {"name": "ASPIRIN", "unii": "R16CO5Y76E"},
                ],
            },
        },
    ]
    write_jsonl(layout.path("parsed_records.jsonl", for_write=True), records)

    documents = [
        {
            "parsed_document_id": "document-label",
            "source_id": "dailymed",
            "raw_asset_id": "asset-label",
            "checksum": "checksum-label",
            "storage_path": "/raw/label.xml",
            "section_title": "Clinical Pharmacology",
            "content_kind": "spl_section",
            "text": "Midazolam clearance was measured. Midazolam and warfarin clearance were measured.",
        }
    ]
    chunks = [
        {
            "chunk_id": "chunk-named",
            "parsed_document_id": "document-label",
            "source_id": "dailymed",
            "raw_asset_id": "asset-label",
            "checksum": "checksum-label",
            "storage_path": "/raw/label.xml",
            "chunk_text": "Midazolam clearance was measured.",
        },
        {
            "chunk_id": "chunk-unlinked",
            "parsed_document_id": "document-label",
            "source_id": "dailymed",
            "raw_asset_id": "asset-label",
            "checksum": "checksum-label",
            "storage_path": "/raw/label.xml",
            "chunk_text": "Midazolam and warfarin clearance were measured.",
        },
    ]
    write_jsonl(layout.path("parsed_documents.jsonl", for_write=True), documents)
    write_jsonl(layout.path("text_chunks.jsonl", for_write=True), chunks)

    candidates = [
        candidate(
            "evidence-structured",
            "4192",
            source_id="pubchem",
            raw_asset_id="asset-pubchem",
            checksum="checksum-pubchem",
            storage_path="/raw/pubchem.json",
            parsed_record_id="record-properties",
        ),
        candidate(
            "evidence-named",
            "Midazolam clearance was measured.",
            source_id="dailymed",
            raw_asset_id="asset-label",
            checksum="checksum-label",
            storage_path="/raw/label.xml",
            parsed_document_id="document-label",
            chunk_id="chunk-named",
        ),
        candidate(
            "evidence-pending",
            "Midazolam and warfarin clearance were measured.",
            source_id="dailymed",
            raw_asset_id="asset-label",
            checksum="checksum-label",
            storage_path="/raw/label.xml",
            parsed_document_id="document-label",
            chunk_id="chunk-unlinked",
        ),
        candidate(
            "evidence-bad-hash",
            "Midazolam bad hash.",
            source_id="dailymed",
            raw_asset_id="asset-label",
            checksum="checksum-label",
            storage_path="/raw/label.xml",
            parsed_document_id="document-label",
            chunk_id="chunk-named",
            source_text_hash="not-a-sha256",
        ),
    ]
    write_jsonl(layout.path("evidence_candidates.jsonl", for_write=True), candidates)


def test_stage_runs_provenance_then_rule_attribution_without_embedding(tmp_path: Path) -> None:
    run_dir = tmp_path / "raw-processing"
    output_dir = tmp_path / "final-evidence"
    build_fixture(run_dir)

    result = ProvenanceAttributionStage(
        run_dir=run_dir,
        output_dir=output_dir,
        llm_mode="off",
    ).run()

    assert result["provenance"]["candidate_count"] == 4
    assert result["provenance"]["passed_count"] == 3
    assert result["provenance"]["excluded_count"] == 1
    assert result["compound_attribution"]["attribution_count"] == 3
    assert result["compound_attribution"]["llm_pending_count"] == 1
    assert result["next_required_step"] == "complete_llm_attribution"

    provenance = {
        row["evidence_id"]: row
        for row in read_jsonl(output_dir / "00_run_manifest" / "provenance_validation.jsonl")
    }
    assert provenance["evidence-bad-hash"]["provenance_status"] == "excluded"
    assert "source_text_hash_mismatch" in provenance["evidence-bad-hash"]["exclusion_reasons"]

    attribution = {
        row["evidence_id"]: row
        for row in read_jsonl(output_dir / "01_compound_attribution" / "compound_attribution.jsonl")
    }
    assert attribution["evidence-structured"]["compound_role"] == "target_compound"
    assert attribution["evidence-named"]["attribution_status"] == "resolved"
    assert attribution["evidence-pending"]["attribution_status"] == "pending_llm"
    assert "evidence-bad-hash" not in attribution

    identity = read_jsonl(output_dir / "01_compound_attribution" / "compound_identity.jsonl")
    midazolam = next(row for row in identity if row["preferred_name"] == "midazolam")
    assert midazolam["pubchem_cid"] == "4192"
    assert midazolam["chembl_id"] == "CHEMBL655"
    assert midazolam["rxnorm_rxcui"] == "6960"
    assert midazolam["unii"] == ["R60L0SM5BC"]
    assert midazolam["related_forms"][0]["name"] == "MIDAZOLAM HYDROCHLORIDE"

    state = json.loads((output_dir / "00_run_manifest" / "pipeline_state.json").read_text())
    assert state["stages"]["compound_attribution"]["status"] == "rules_completed"
    assert state["stages"]["semantic_deduplication"]["status"] == "not_started"
    assert not (output_dir / "03_deduplication").exists()


def test_stage_rejects_upstream_embedding_before_attribution(tmp_path: Path) -> None:
    run_dir = tmp_path / "raw-processing"
    output_dir = tmp_path / "final-evidence"
    build_fixture(run_dir)
    state_path = raw_processing_layout(run_dir).path("pipeline_state.json")
    state = json.loads(state_path.read_text())
    state["stages"]["embedding"] = {"status": "completed", "accepted": True}
    write_json(state_path, state)

    stage = ProvenanceAttributionStage(run_dir=run_dir, output_dir=output_dir)

    try:
        stage.run()
    except RuntimeError as exc:
        assert "embedding must remain not_started" in str(exc)
    else:
        raise AssertionError("The stage accepted an out-of-order upstream embedding run")


def test_direct_public_pbpk_model_with_nonfinal_rule_role_requires_llm_review() -> None:
    row = {
        "evidence_type": "public_pbpk_model",
        "source_text": "A PBPK model was developed for rosuvastatin and linked to PD response.",
    }

    assert requires_public_model_role_review(row, "comparator") is True
    assert requires_public_model_role_review(row, "target_compound") is False


def test_new_acquisition_drug_reaches_attribution_without_a_historical_allowlist(tmp_path):
    run_dir = tmp_path / "raw-processing"
    output = tmp_path / "final-evidence"
    build_fixture(run_dir)
    for path in run_dir.rglob("*.jsonl"):
        text = path.read_text().replace("midazolam", "atorvastatin").replace("Midazolam", "Atorvastatin").replace("MIDAZOLAM", "ATORVASTATIN")
        rows = [json.loads(line) for line in text.splitlines()]
        for row in rows:
            if row.get("source_text") and row.get("evidence_id") != "evidence-bad-hash":
                row["source_text_hash"] = hashlib.sha256(row["source_text"].encode()).hexdigest()
        write_jsonl(path, rows)
    ProvenanceAttributionStage(run_dir=run_dir, output_dir=output).run()
    identities = read_jsonl(output / "01_compound_attribution/compound_identity.jsonl")
    assert {r["preferred_name"] for r in identities} == {"atorvastatin", "warfarin"}
    target = next(r for r in identities if r["preferred_name"] == "atorvastatin")
    rows = read_jsonl(output / "01_compound_attribution/compound_attribution.jsonl")
    resolved = next(r for r in rows if r["evidence_id"] == "evidence-named")
    assert resolved["compound_id"] == target["compound_id"]
    assert resolved["attribution_status"] == "resolved"


def test_existing_model_drug_id_is_reused_without_merging_salts_or_stereoisomers():
    from pbpk_kb.processing.final_evidence_attribution_stage import (
        CompoundIdentityBuilder, bind_existing_compound_identities,
    )
    rows = CompoundIdentityBuilder(["atorvastatin", "atorvastatin calcium", "s-metoprolol"]).build()
    original = {r["preferred_name"]: r["compound_id"] for r in rows}
    bind_existing_compound_identities(rows, [
        {"compound_id": "existing-model-drug-id", "preferred_name": "Atorvastatin", "normalized_name": "atorvastatin"},
        {"compound_id": "racemic-id", "preferred_name": "Metoprolol"},
    ])
    assert rows[0]["compound_id"] == "existing-model-drug-id"
    assert rows[1]["compound_id"] == original["atorvastatin calcium"]
    assert rows[2]["compound_id"] == original[rows[2]["preferred_name"]]


def test_conflicting_or_ambiguous_identity_binding_fails_before_assigning_any_id():
    import pytest
    from pbpk_kb.processing.final_evidence_attribution_stage import (
        CompoundIdentityBuilder, bind_existing_compound_identities,
    )
    rows = CompoundIdentityBuilder(["atorvastatin", "bupropion"]).build()
    rows[1]["pubchem_cid"] = "different"
    before = json.loads(json.dumps(rows))
    with pytest.raises(ValueError, match="identifier conflict"):
        bind_existing_compound_identities(rows, [
            {"compound_id": "one", "preferred_name": "Atorvastatin"},
            {"compound_id": "two", "preferred_name": "Bupropion", "pubchem_cid": "original"},
        ])
    assert rows == before
    with pytest.raises(ValueError, match="Ambiguous"):
        bind_existing_compound_identities(rows, [
            {"compound_id": "one", "preferred_name": "Atorvastatin"},
            {"compound_id": "two", "preferred_name": "atorvastatin"},
        ])
    assert rows == before


def test_split_name_table_header_is_unlinked_and_cannot_enter_final_evidence():
    from pbpk_kb.processing.final_evidence_attribution_stage import build_rule_attributions
    from pbpk_kb.processing.final_evidence_post_stage import build_evidence_layers

    header = (
        "C oadministered Drug (Postulated Mechanism of Interaction) | "
        "C oadministered D rug Dose/Schedule | P osaconazole D ose/Schedule | "
        "C hange in Mean C max (ratio estimate*; 90% CI of the ratio estimate) | "
        "C hange in Mean A U C (ratio estimate*; 90% CI of the ratio estimate)"
    )
    source = {
        "evidence_id": "header-only", "compound": "posaconazole",
        "evidence_type": "clinical_pk", "parameter": "bioavailability",
        "source_id": "dailymed", "source_text": header,
        "source_location": "dailymed:fixture#table=1#row=1",
        "source_text_hash": hashlib.sha256(header.encode()).hexdigest(),
        "metadata_json": {"document_compound_identity": "exact_single_target_active_ingredient"},
    }
    before = json.loads(json.dumps(source))
    rows = build_rule_attributions(candidates=[source], compounds=[{
        "compound_id": "posaconazole-id", "preferred_name": "posaconazole", "synonyms": [],
    }], references={}, source_scope_identity={})
    row = rows[0]
    assert source == before
    assert row["attribution_status"] == "resolved"
    assert row["compound_role"] == "shared_or_unlinked"
    assert row["compound_id"] == "posaconazole-id"  # Retrieval audit identity only.
    assert row["attribution_method"] == "rule_no_source_name_span"
    assert row["attribution_confidence"] == 0.0
    assert row["matched_text"] is None and row["matched_synonym"] is None
    assert row["llm_required"] is False
    assert row["candidate_context"]["source_text"] == header
    # Even a permissive later domain gate cannot promote an unlinked row.
    outputs, summary = build_evidence_layers({
        "candidates": [source], "attributions": rows,
        "provenance": [{"evidence_id": source["evidence_id"], "provenance_status": "passed"}],
        "domain_results": [{"evidence_id": source["evidence_id"], "domain_gate_status": "passed"}],
        "deterministic_members": [], "semantic_members": [],
        "deterministic_representatives": [{**source, **row}],
    }, [], [])
    assert summary["accepted"] is True
    assert outputs["preselected_evidence.jsonl"] == []
    assert outputs["reviewable_evidence.jsonl"] == []
    assert outputs["accepted_evidence.jsonl"] == []
    assert "compound_role_not_final:shared_or_unlinked" in outputs["excluded_from_reviewable.jsonl"][0]["exclusion_reasons"]


def test_exact_source_span_guard_does_not_join_split_names_or_use_metadata():
    from pbpk_kb.processing.final_evidence_attribution_stage import has_allowed_source_name_span

    compounds = [{"preferred_name": "posaconazole", "synonyms": ["Noxafil", "147912", ""]}]
    assert has_allowed_source_name_span("POSACONAZOLE concentrations were measured.", compounds)
    assert has_allowed_source_name_span("Noxafil (oral suspension).", compounds)
    assert not has_allowed_source_name_span("P osaconazole Dose/Schedule", compounds)
    assert not has_allowed_source_name_span("The report has identifier 147912.", compounds)
    assert not has_allowed_source_name_span("proposaconazolelike compound", compounds)


def test_nameless_model_candidate_is_not_sent_for_impossible_span_selection():
    from pbpk_kb.processing.final_evidence_attribution_stage import build_rule_attributions

    rows = build_rule_attributions(candidates=[{
        "evidence_id": "unnamed-model", "compound": "posaconazole",
        "evidence_type": "public_pbpk_model", "source_id": "europe_pmc",
        "source_text": "A PBPK model was developed and evaluated in adults.",
    }], compounds=[{"compound_id": "one", "preferred_name": "posaconazole"}],
        references={}, source_scope_identity={})
    assert rows[0]["compound_role"] == "shared_or_unlinked"
    assert rows[0]["attribution_status"] == "resolved"
    assert rows[0]["llm_required"] is False
    assert rows[0]["matched_text"] is None


def test_real_source_synonym_still_requires_llm_for_ambiguous_multidrug_evidence():
    from pbpk_kb.processing.final_evidence_attribution_stage import build_rule_attributions

    rows = build_rule_attributions(candidates=[{
        "evidence_id": "named-multidrug", "compound": "posaconazole",
        "evidence_type": "clinical_pk", "source_id": "dailymed",
        "source_text": "Noxafil and midazolam clearance were measured.",
    }], compounds=[
        {"compound_id": "one", "preferred_name": "posaconazole", "synonyms": ["Noxafil"]},
        {"compound_id": "two", "preferred_name": "midazolam"},
    ], references={}, source_scope_identity={})
    assert rows[0]["attribution_status"] == "pending_llm"
    assert rows[0]["llm_required"] is True
    assert rows[0]["attribution_method"] != "rule_no_source_name_span"


def test_nameless_structured_compound_property_remains_resolved_by_existing_rule():
    from pbpk_kb.processing.final_evidence_attribution_stage import build_rule_attributions

    rows = build_rule_attributions(candidates=[{
        "evidence_id": "structured-property", "compound": "posaconazole",
        "evidence_type": "physicochemical", "source_id": "pubchem",
        "source_text": "147912", "parameter": "CID",
    }], compounds=[{"compound_id": "one", "preferred_name": "posaconazole"}],
        references={}, source_scope_identity={})
    assert rows[0]["attribution_status"] == "resolved"
    assert rows[0]["compound_role"] == "target_compound"
    assert rows[0]["attribution_method"] == "rule_structured_source_compound"
