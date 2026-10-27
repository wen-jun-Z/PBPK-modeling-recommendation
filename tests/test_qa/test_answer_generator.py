from pbpk_kb.qa.answer_generator import generate_answer
from pbpk_kb.qa.retrieval import RetrievalResult, infer_evidence_type, query_terms


def test_query_terms_removes_common_question_words() -> None:
    assert query_terms("What public clinical PK evidence exists for midazolam clearance?") == [
        "clinical",
        "midazolam",
        "clearance",
    ]


def test_infer_evidence_type_for_clinical_pk_question() -> None:
    assert infer_evidence_type("What public clinical PK evidence exists for midazolam?") == "clinical_pk"


def test_generate_answer_is_evidence_bound_and_cited() -> None:
    retrieval = RetrievalResult(
        compound={"preferred_name": "Midazolam"},
        structured_evidence=[
            {
                "evidence_type": "clinical_pk",
                "evidence_id": "e1",
                "compound_name": "Midazolam",
                "parameter_name": "total clearance",
                "numeric_value": None,
                "value_text": "9.3 to 11.0",
                "unit": "mL/min/kg",
                "context": "Elimination",
                "source_id": "dailymed",
                "source_name": "DailyMed",
                "source_location": "SPL_SET_ID:x::SECTION:Elimination",
                "evidence_text": "The mean total clearance ranged from 9.3 to 11.0 mL/min/kg.",
                "extraction_method": "rule_based",
                "curation_status": "needs_review",
            }
        ],
        model_records=[],
        graph_relationships=[
            {
                "relationship_id": "rel-1",
                "compound_name": "Midazolam",
                "relationship_type": "inhibits_or_binds",
                "target_name": "Cytochrome P450 3A4",
                "gene_symbol": "CYP3A4",
                "target_type": "enzyme",
                "source_id": "chembl",
                "source_name": "ChEMBL",
                "source_location": "CHEMBL_MOLECULE:CHEMBL655::CHEMBL_ACTIVITY:1",
                "extraction_method": "machine_extracted",
                "curation_status": "needs_review",
            }
        ],
        document_chunks=[],
        retrieval_notes=[],
    )

    result = generate_answer("What clearance evidence exists for midazolam?", retrieval)

    assert "[E1] total clearance: 9.3 to 11.0 mL/min/kg" in result["answer"]
    assert result["citations"][0]["source_id"] == "dailymed"
    assert result["citations"][0]["source_location"] == "SPL_SET_ID:x::SECTION:Elimination"
    assert "[G1] inhibits_or_binds: Cytochrome P450 3A4" in result["answer"]
    assert result["retrieval_summary"]["graph_relationship_count"] == 1
    assert "graph_relationships" in result["retrieval_summary"]["retrieval_order"]
    assert "not modeling-approved" in result["answer"]
    assert result["uncertainties_and_gaps"]


def test_generate_answer_omits_extracted_values_when_unit_is_missing() -> None:
    retrieval = RetrievalResult(
        compound={"preferred_name": "Example"},
        structured_evidence=[
            {
                "evidence_type": "clinical_pk",
                "evidence_id": "e1",
                "compound_name": "Example",
                "parameter_name": "Cmax",
                "numeric_value": 25.0,
                "value_text": None,
                "unit": None,
                "context": "PK",
                "source_id": "dailymed",
                "source_name": "DailyMed",
                "source_location": "SPL_SET_ID:x::SECTION:PK",
                "evidence_text": "Cmax was reported without a unit in the parsed candidate.",
                "extraction_method": "rule_based",
                "curation_status": "needs_review",
            }
        ],
        model_records=[],
        graph_relationships=[],
        document_chunks=[],
        retrieval_notes=[],
    )

    result = generate_answer("What is Cmax?", retrieval)

    assert "Cmax: value omitted because unit is missing" in result["answer"]
    assert "Cmax: 25" not in result["answer"]
    assert "evidence record(s) have extracted values without units" in result["answer"]


def test_generate_answer_describes_auto_validated_evidence_accurately() -> None:
    retrieval = RetrievalResult(
        compound={"preferred_name": "Midazolam"},
        structured_evidence=[
            {
                "evidence_type": "clinical_pk",
                "evidence_id": "e1",
                "compound_name": "Midazolam",
                "parameter_name": "clearance",
                "numeric_value": 29.37,
                "value_text": "29.37",
                "unit": "L/h",
                "context": "single-dose study",
                "source_id": "pmc_oa",
                "source_name": "PMC Open Access",
                "source_location": "PMC:example#page=5",
                "evidence_text": "The estimated clearance was 29.37 L/h.",
                "extraction_method": "llm_assisted",
                "curation_status": "auto_validated",
            }
        ],
        model_records=[],
        graph_relationships=[],
        document_chunks=[],
        retrieval_notes=[],
    )

    result = generate_answer("What clearance evidence exists?", retrieval)

    assert "auto-validated but not modeling-approved" in result["answer"]
    assert "still needs review" not in result["answer"]
