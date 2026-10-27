from pydantic import ValidationError

from pbpk_kb.extraction.schemas import ClinicalPKEvidence, PhysicochemicalPropertyEvidence, TissueExpressionEvidence


def test_physicochemical_schema_requires_traceable_source_location() -> None:
    evidence = PhysicochemicalPropertyEvidence(
        compound_id=None,
        property_name="MolecularWeight",
        property_value=325.8,
        property_value_text="325.8",
        unit="g/mol",
        source_id="pubchem",
        source_location="CID:4192",
        evidence_text='{"MolecularWeight": 325.8}',
        confidence_score=0.85,
    )

    assert evidence.source_id == "pubchem"
    assert evidence.property_value == 325.8
    assert evidence.unit == "g/mol"
    assert evidence.curation_status == "machine_extracted"
    assert evidence.value_origin == "unknown"
    assert evidence.evidence_origin == "unknown"


def test_tissue_expression_schema_tracks_canonical_unit_fields() -> None:
    evidence = TissueExpressionEvidence(
        tissue="liver",
        expression_value=3367.1,
        expression_value_text="3367.1",
        unit="nTPM",
        canonical_value=3367.1,
        canonical_unit="nTPM",
        unit_dimension="expression",
        species="human",
        assay_method="HPA RNA tissue specific nTPM",
        source_id="hpa",
        source_location="HPA:GENE:CYP3A4::RNA_TISSUE_NTPM:liver",
        confidence_score=0.72,
    )

    assert evidence.tissue == "liver"
    assert evidence.canonical_unit == "nTPM"
    assert evidence.unit_dimension == "expression"


def test_evidence_schema_rejects_blank_source_location() -> None:
    try:
        ClinicalPKEvidence(
            compound_id=None,
            pk_parameter="Pharmacokinetics",
            source_id="dailymed",
            source_location=" ",
        )
    except ValidationError as exc:
        assert "source_location is required" in str(exc)
    else:
        raise AssertionError("Expected ValidationError for blank source_location")

from pbpk_kb.extraction.ddi_extractor import extract_ddi_statements


def test_extracts_cyp3a4_ddi_statement_with_auc_ratio() -> None:
    text = (
        "Drug interactions: Strong CYP3A4 inhibitors increased midazolam AUC by 500% "
        "in healthy subjects and should be avoided."
    )

    results = extract_ddi_statements(text)

    assert len(results) == 1
    assert results[0].enzyme_or_transporter == "CYP3A4"
    assert results[0].interaction_type == "CYP3A4 inhibitor interaction candidate"
    assert results[0].auc_ratio == 6.0
    assert results[0].population == "healthy subjects"


def test_ignores_non_interaction_sentence() -> None:
    assert extract_ddi_statements("The product is supplied in amber glass bottles.") == []


# [FIX 2026-07-02 · claude-opus-4-8] a compound-as-substrate sentence must not be mislabeled an inhibitor just because
# the word "inhibitors" (referring to the interacting partner) also appears.
def test_substrate_sentence_not_labeled_inhibitor() -> None:
    text = "Midazolam is a substrate of CYP3A4; avoid coadministration with strong CYP3A4 inhibitors."
    results = extract_ddi_statements(text)
    assert len(results) == 1
    assert results[0].enzyme_or_transporter == "CYP3A4"
    assert results[0].interaction_type == "CYP3A4 substrate interaction candidate"


# [FIX 2026-07-02 · claude-opus-4-8] caffeine's real enzyme (CYP1A2) is now covered.
def test_cyp1a2_target_is_covered() -> None:
    results = extract_ddi_statements("Caffeine is primarily metabolized by CYP1A2.")
    assert len(results) == 1
    assert results[0].enzyme_or_transporter == "CYP1A2"
    assert results[0].interaction_type == "CYP1A2 substrate interaction candidate"


# [FIX 2026-07-02 · claude-opus-4-8] metformin transporters (OCT2/MATE1) are now covered.
def test_transporter_targets_are_covered() -> None:
    results = extract_ddi_statements("Metformin is a substrate of the OCT2 and MATE1 transporters.")
    assert results
    assert results[0].enzyme_or_transporter in {"OCT2", "MATE1"}
    assert results[0].ambiguous_target is True
    assert set(results[0].all_targets) == {"OCT2", "MATE1"}


# [FIX 2026-07-02 · claude-opus-4-8] a sentence naming several drugs+enzymes is flagged target-ambiguous.
def test_multiple_targets_are_flagged_ambiguous() -> None:
    text = "A 50% increase in the AUC of caffeine (CYP1A2) and dextromethorphan (CYP2D6) was observed."
    results = extract_ddi_statements(text)
    assert results
    assert results[0].ambiguous_target is True
    assert "CYP1A2" in results[0].all_targets and "CYP2D6" in results[0].all_targets


def test_study_type_reflects_source() -> None:
    label = extract_ddi_statements("CYP3A4 inhibitors increase exposure.", source_id="dailymed")
    lit = extract_ddi_statements("CYP3A4 inhibitors increase exposure.", source_id="pubmed")
    assert label and label[0].study_type == "label_statement"
    assert lit and lit[0].study_type == "literature_statement"

from pbpk_kb.extraction.tissue_expression_extractor import extract_hpa_tissue_expression


def test_extracts_hpa_rna_and_protein_tissue_expression() -> None:
    payload = {
        "Gene": "CYP3A4",
        "Gene description": "Cytochrome P450 family 3 subfamily A member 4",
        "RNA tissue specific nTPM": {"liver": "3367.1"},
        "Protein tissue specific Intensity": {"liver": "110494763.1"},
    }

    results = extract_hpa_tissue_expression(payload)

    assert len(results) == 2
    assert results[0].gene_symbol == "CYP3A4"
    assert results[0].tissue == "liver"
    assert results[0].unit == "nTPM"
    assert results[0].canonical_value == 3367.1
    assert results[0].unit_dimension == "expression"
    assert results[1].unit == "intensity"
