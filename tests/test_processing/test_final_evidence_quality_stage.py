from pbpk_kb.processing.final_evidence_quality_stage import (
    deterministic_deduplicate,
    evaluate_domain_gate,
    semantic_input_eligible,
)


def row(evidence_id: str, evidence_type: str, parameter: str, source_text: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "parameter": parameter,
        "source_text": source_text,
        "source_text_hash": f"hash-{evidence_id}",
        "source_id": "pubmed",
        "raw_asset_id": f"asset-{evidence_id}",
        "source_location": f"pubmed:test#chunk={evidence_id}",
        "attributed_compound_id": "compound-caffeine",
        "attributed_compound_name": "caffeine",
        "compound_role": "target_compound",
        "provenance_status": "passed",
        "attribution_confidence": 0.9,
        "extraction_confidence": 0.8,
        "extraction_method": "llm_assisted",
        "domain_gate_status": "passed",
        "value": None,
        "unit": None,
        "canonical_value": None,
        "canonical_unit": None,
        "condition": None,
    }
    value.update(overrides)
    return value


def test_domain_gates_keep_supported_properties_and_reject_identity_only_fields() -> None:
    molecular_weight = row("one", "physicochemical", "MolecularWeight", "194.19")
    cid = row("two", "physicochemical", "CID", "2519")

    assert evaluate_domain_gate(molecular_weight)["domain_gate_status"] == "passed"
    assert evaluate_domain_gate(cid)["domain_gate_status"] == "excluded"


def test_ddi_without_direction_is_queued_before_embedding() -> None:
    candidate = row(
        "ddi",
        "ddi",
        "CYP3A",
        "Midazolam was included as a CYP3A probe substrate in the cocktail.",
        compound_role="victim_drug",
    )

    result = evaluate_domain_gate(candidate)

    assert result["domain_gate_status"] == "needs_manual_review"
    assert "direction" in result["reasons"]


def test_special_population_requires_explicit_population_or_species() -> None:
    renal = row(
        "renal",
        "special_population",
        "clearance",
        "Caffeine clearance was lower in subjects with renal impairment.",
    )
    generic = row(
        "generic",
        "special_population",
        "clearance",
        "Caffeine clearance was measured in the study cohort.",
    )

    assert evaluate_domain_gate(renal)["domain_classification"] == "human_special_population"
    assert evaluate_domain_gate(generic)["domain_gate_status"] == "excluded"


def test_public_model_requires_downloadable_asset_or_explicit_pbpk_work() -> None:
    downloadable = row(
        "model-file",
        "public_pbpk_model",
        "model_file",
        "Caffeine/Caffeine-Model.pksim5",
        source_id="osp_pbpk_model_library",
        metadata_json={
            "model_file_path": "Caffeine/Caffeine-Model.pksim5",
            "model_storage_path": "/raw/Caffeine-Model.pksim5",
            "model_checksum": "abc",
        },
    )
    mention = row("mention", "public_pbpk_model", "model", "A statistical model was used.")

    assert evaluate_domain_gate(downloadable)["domain_classification"] == "downloadable_public_model"
    assert evaluate_domain_gate(mention)["domain_gate_status"] == "excluded"


def test_deterministic_dedup_runs_source_then_content_and_keeps_one_final_representative() -> None:
    first = row(
        "one",
        "clinical_pk",
        "AUC",
        "Caffeine AUC was 10 mg h/L in healthy adults.",
        raw_asset_id="asset-shared",
        source_location="pubmed:test#chunk=1",
        source_text_hash="same-text",
        value=10.0,
        unit="mg h/L",
    )
    mirror = {**first, "evidence_id": "two", "extraction_confidence": 0.7}
    independent = {
        **first,
        "evidence_id": "three",
        "raw_asset_id": "asset-other",
        "source_location": "pmc:test#chunk=2",
        "source_id": "pmc_oa",
    }

    groups, members, representatives = deterministic_deduplicate([first, mirror, independent])

    assert [item["group_type"] for item in groups] == ["source", "content"]
    assert len(members) == 4
    assert [item["evidence_id"] for item in representatives] == ["one"]


def test_semantic_input_excludes_structured_tables_and_domain_review_rows() -> None:
    natural = row(
        "natural",
        "clinical_pk",
        "AUC",
        "Caffeine area under the concentration time curve increased after oral dosing in healthy adults.",
    )
    table = {**natural, "evidence_id": "table", "parsed_table_id": "table-1"}
    review = {**natural, "evidence_id": "review", "domain_gate_status": "needs_manual_review"}

    assert semantic_input_eligible(natural) is True
    assert semantic_input_eligible(table) is False
    assert semantic_input_eligible(review) is False


def test_structured_table_duplicate_ignores_condition_formatting_differences() -> None:
    first = row(
        "one",
        "clinical_pk",
        "clearance",
        "850 mg | 2.45 | 2.71 | 412",
        source_text_hash="same-table-row",
        source_id="dailymed",
        raw_asset_id="asset-one",
        source_location="dailymed:one#table=1",
        extraction_method="source_structured_table",
        value=412.0,
        unit="mL/min",
        condition="Metformin HCl 850 mg single dose",
    )
    second = {
        **first,
        "evidence_id": "two",
        "raw_asset_id": "asset-two",
        "source_location": "openfda:two#table=1",
        "source_id": "openfda_drug_label",
        "condition": "Metformin hydrochloride tablets 850 mg single dose",
    }

    groups, _, representatives = deterministic_deduplicate([first, second])

    assert [group["group_type"] for group in groups] == ["content"]
    assert len(representatives) == 1


def test_natural_language_duplicate_keeps_condition_in_content_key() -> None:
    first = row(
        "one",
        "clinical_pk",
        "AUC",
        "Caffeine AUC was measured.",
        source_text_hash="same-natural-text",
        value=10.0,
        unit="mg h/L",
        condition="fed",
    )
    second = {**first, "evidence_id": "two", "raw_asset_id": "asset-two", "source_location": "pubmed:two", "condition": "fasted"}

    groups, _, representatives = deterministic_deduplicate([first, second])

    assert groups == []
    assert len(representatives) == 2


def test_one_source_sentence_preserves_distinct_parameters_ranges_and_study_contexts() -> None:
    first = row("one", "clinical_pk", "AUC", "AUC and Cmax were measured in both groups.",
                value_low=1.0, value_high=2.0, unit="mg h/L")
    variants = [
        {**first, "evidence_id": "parameter", "parameter": "Cmax", "unit": "mg/L"},
        {**first, "evidence_id": "range", "value_high": 3.0},
        {**first, "evidence_id": "population", "population": "healthy adults"},
        {**first, "evidence_id": "dose", "dose": "100 mg"},
        {**first, "evidence_id": "compound", "attributed_compound_id": "compound-metformin"},
    ]
    duplicate = {**first, "evidence_id": "duplicate", "extraction_confidence": 0.5}
    _, _, representatives = deterministic_deduplicate([first, duplicate, *variants])
    assert {r["evidence_id"] for r in representatives} == {"one", *(r["evidence_id"] for r in variants)}


def test_animal_pk_is_explicit_and_cannot_enter_human_clinical_layer() -> None:
    candidate = row("rabbit", "clinical_pk", "half-life",
                    "The mean elimination half-life of caffeine in rabbits was 3.8 hr.",
                    value=3.8, unit="hr")
    result = evaluate_domain_gate(candidate)
    assert result["species"] == ["rabbit"]
    assert result["domain_gate_status"] == "needs_manual_review"
    assert not semantic_input_eligible({**candidate, **result})
    human = {**candidate, "source_text": "The half-life in healthy adults was 3.8 hr."}
    assert evaluate_domain_gate(human)["domain_gate_status"] == "passed"
    animal_adme = row("rat", "adme", "metabolism", "Caffeine metabolism was measured in rats.")
    assert evaluate_domain_gate(animal_adme)["species"] == ["rat"]


def test_adme_gate_separates_water_transport_effect_from_drug_absorption() -> None:
    quote = "Theophylline (5 × 10 -3 M) also increased net water absorption, an effect not previously demonstrated in renal tissue."
    for parameter in ("net water absorption", "absorption of water", "sodium excretion"):
        candidate = row("non-drug", "adme", parameter, quote, attributed_compound_name="theophylline")
        result = evaluate_domain_gate(candidate)
        assert result["domain_gate_status"] == "excluded"
        assert result["reasons"] == ["pharmacodynamic_or_non_drug_adme_context"]
    drug = row("drug", "adme", "absorption", "Theophylline absorption was measured after administration with water.")
    assert evaluate_domain_gate(drug)["domain_gate_status"] == "passed"
