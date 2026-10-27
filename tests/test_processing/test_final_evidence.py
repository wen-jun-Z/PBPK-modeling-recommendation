from pbpk_kb.processing.final_evidence import (
    BIOMEDICAL_EMBEDDING_MODEL,
    build_compound_terms,
    decide_attribution,
    final_evidence_quality_issue,
)


def test_ddi_attribution_marks_retrieval_compound_as_victim() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000001",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": "4192",
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000001",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000001",
        "compound_name": "midazolam",
        "source_text": "Ketoconazole increased midazolam AUC after oral dosing.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role == "victim_drug"
    assert decision.llm_required is False
    assert decision.matched_synonym == "midazolam"


def test_plain_clinical_pk_auc_statement_is_not_reclassified_as_ddi() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000021",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": "4192",
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000021",
        "evidence_type": "clinical_pk",
        "compound_id": compounds[0]["compound_id"],
        "compound_name": "midazolam",
        "source_id": "pubmed",
        "source_text": "Midazolam AUC increased by 25% in the renal impairment group.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role == "target_compound"
    assert decision.llm_required is False


def test_explicit_probe_substrate_appositive_is_high_confidence_victim() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000022",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": "4192",
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000022",
        "evidence_type": "ddi",
        "compound_id": compounds[0]["compound_id"],
        "compound_name": "midazolam",
        "source_id": "clinicaltrials",
        "parameter": "CYP3A4",
        "source_text": "The study evaluated the effect of AZD5718 on the PK of midazolam, a known sensitive CYP3A4 substrate.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role == "victim_drug"
    assert decision.attribution_confidence >= 0.9
    assert decision.llm_required is False


def test_exact_chembl_identity_allows_structured_numeric_property() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000023",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": "4192",
            "chembl_id": "CHEMBL655",
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000023",
        "evidence_type": "physicochemical",
        "compound_id": compounds[0]["compound_id"],
        "compound_name": "midazolam",
        "source_id": "chembl",
        "parameter": "full_mwt",
        "metadata_json": '{"chembl_identity_gate":"exact"}',
        "source_text": "325.77",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role == "target_compound"
    assert decision.llm_required is False


def test_single_active_ingredient_document_allows_table_row_without_repeated_name() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000024",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": "4192",
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000024",
        "evidence_type": "clinical_pk",
        "compound_id": compounds[0]["compound_id"],
        "compound_name": "midazolam",
        "source_id": "dailymed",
        "source_location": "dailymed:label#Clinical Pharmacology#table=1#row=2",
        "metadata_json": '{"document_compound_identity":"exact_single_target_active_ingredient"}',
        "source_text": "0.5 h | 90 ng/mL | 3.1 h",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role == "target_compound"
    assert decision.llm_required is False


def test_unknown_single_compound_match_is_reviewable_target_candidate() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000002",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": "2519",
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000002",
        "evidence_type": "clinical_pk",
        "compound_id": "",
        "compound_name": "",
        "source_text": "Caffeine clearance was measured in healthy volunteers.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_name == "caffeine"
    assert decision.compound_role == "target_compound"
    assert decision.attribution_method == "rule_source_text_single_match"


def test_bare_numeric_identifiers_are_not_compound_terms() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000001111",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": "2519",
            "chembl_id": "CHEMBL113",
            "rxnorm_rxcui": "1886",
        }
    ]

    terms = build_compound_terms(compounds)

    assert any(term.synonym == "caffeine" for term in terms)
    assert any(term.synonym == "CHEMBL113" for term in terms)
    assert all(term.synonym not in {"2519", "1886"} for term in terms)


def test_label_source_retrieval_compound_without_text_match_is_not_final_evidence() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000007",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000007",
        "evidence_type": "adme",
        "compound_id": "00000000-0000-0000-0000-000000000007",
        "compound_name": "caffeine",
        "source_id": "dailymed",
        "source_text": "In plasma, about 50%-80% of the salicylic acid and its metabolites are loosely bound to plasma.",
        "source_location": "dailymed:spl_xml#section",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_role == "shared_or_unlinked"
    assert decision.attribution_method == "rule_retrieval_compound_without_text_match"
    assert decision.llm_required is False
    assert decision.llm_trigger_reason is None
    assert decision.matched_text is None


def test_combination_product_other_component_assertion_is_not_target_compound() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000008",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000008",
        "evidence_type": "adme",
        "compound_id": "00000000-0000-0000-0000-000000000008",
        "compound_name": "caffeine",
        "source_id": "dailymed",
        "source_text": (
            "The bioavailability of the aspirin component of Butalbital, Aspirin, "
            "and Caffeine Capsules is equivalent to that of a solution."
        ),
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_role == "mentioned_only"
    assert decision.llm_required is False


def test_final_evidence_uses_biomedical_embedding_model_constant() -> None:
    assert BIOMEDICAL_EMBEDDING_MODEL == "NeuML/pubmedbert-base-embeddings"


def test_clinical_trials_eligibility_list_is_mentioned_only() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000003",
            "preferred_name": "omeprazole",
            "normalized_name": "omeprazole",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000003",
        "evidence_type": "adme",
        "compound_id": "00000000-0000-0000-0000-000000000003",
        "compound_name": "omeprazole",
        "source_id": "clinicaltrials",
        "source_text": "Exclusion Criteria: use of benzodiazepine drugs, caffeine, omeprazole, dextromethorphan, lovastatin or any chemically related drug.",
        "source_title": "ClinicalTrials study",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"omeprazole": compounds[0]})

    assert decision.compound_role == "mentioned_only"
    assert decision.llm_required is False


def test_compared_with_matched_drug_is_comparator() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000004",
            "preferred_name": "warfarin",
            "normalized_name": "warfarin",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000004",
        "evidence_type": "clinical_pk",
        "compound_id": "00000000-0000-0000-0000-000000000004",
        "compound_name": "warfarin",
        "source_id": "clinicaltrials",
        "source_text": "The investigational treatment was compared with warfarin as active control.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"warfarin": compounds[0]})

    assert decision.compound_role == "comparator"
    assert decision.llm_required is False


def test_directional_ddi_misclassified_as_comparator_is_sent_to_llm() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000104",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000104",
        "evidence_type": "ddi",
        "compound_id": compounds[0]["compound_id"],
        "compound_name": "midazolam",
        "source_id": "pubmed",
        "source_text": (
            "Midazolam exposure decreased when coadministered with amenamevir "
            "compared with midazolam alone."
        ),
    }

    decision = decide_attribution(
        row,
        terms,
        {compounds[0]["compound_id"]: compounds[0]},
        {"midazolam": compounds[0]},
    )

    assert decision.compound_role == "comparator"
    assert decision.llm_required is True
    assert decision.llm_trigger_reason == "ddi_nonfinal_role_with_directional_effect"


def test_inactive_ingredient_context_is_excipient() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000005",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000005",
        "evidence_type": "physicochemical",
        "compound_id": "00000000-0000-0000-0000-000000000005",
        "compound_name": "caffeine",
        "source_id": "dailymed",
        "source_text": "Inactive ingredients include lactose, cellulose, and caffeine.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_role == "excipient"
    assert decision.llm_required is False


def test_ddi_perpetrator_rule_for_inhibitor() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000006",
            "preferred_name": "ketoconazole",
            "normalized_name": "ketoconazole",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000006",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000006",
        "compound_name": "ketoconazole",
        "source_id": "dailymed",
        "source_text": "Ketoconazole inhibits CYP3A4 and increased midazolam AUC.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"ketoconazole": compounds[0]})

    assert decision.compound_role == "perpetrator_drug"


def test_probe_cocktail_parameter_mismatch_is_not_final_evidence() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000009",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000009",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000009",
        "compound_name": "caffeine",
        "source_id": "clinicaltrials",
        "parameter": "CYP2C9",
        "source_text": "Probe cocktail: caffeine (for CYP1A2), warfarin (for CYP2C9), and midazolam (for CYP3A4).",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_role == "shared_or_unlinked"
    assert decision.attribution_method == "rule_probe_relation_parameter_mismatch"
    assert decision.llm_required is True
    assert decision.llm_trigger_reason == "probe_relation_parameter_mismatch"


def test_probe_cocktail_local_pair_allows_matching_parameter() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000010",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000010",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000010",
        "compound_name": "caffeine",
        "source_id": "clinicaltrials",
        "parameter": "CYP1A2",
        "source_text": "Probe cocktail: caffeine (for CYP1A2), warfarin (for CYP2C9), and midazolam (for CYP3A4).",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_role in {"target_compound", "victim_drug"}
    assert decision.attribution_method != "rule_probe_relation_parameter_mismatch"


def test_probe_cocktail_midazolam_cyp3a4_pair_is_allowed() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000011",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000011",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000011",
        "compound_name": "midazolam",
        "source_id": "clinicaltrials",
        "parameter": "CYP3A4",
        "source_text": "Probe cocktail: caffeine (CYP1A2), warfarin (CYP2C9), omeprazole (CYP2C19), dextromethorphan (CYP2D6), midazolam (CYP3A4).",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role in {"target_compound", "victim_drug"}
    assert decision.attribution_method != "rule_probe_relation_parameter_mismatch"


def test_transporter_probe_pair_mismatch_is_not_final_evidence() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000012",
            "preferred_name": "digoxin",
            "normalized_name": "digoxin",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000012",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000012",
        "compound_name": "digoxin",
        "source_id": "clinicaltrials",
        "parameter": "BCRP",
        "source_text": "Transporter cocktail: digoxin (P-gp substrate), rosuvastatin (BCRP and OATP1B1 substrate).",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"digoxin": compounds[0]})

    assert decision.compound_role == "shared_or_unlinked"
    assert decision.attribution_method == "rule_probe_relation_parameter_mismatch"


def test_chembl_search_property_without_text_identity_is_not_final_evidence() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000013",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000013",
        "evidence_type": "physicochemical",
        "compound_id": "00000000-0000-0000-0000-000000000013",
        "compound_name": "caffeine",
        "source_id": "chembl",
        "parameter": "full_mwt",
        "source_text": "full_mwt: 328.37",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"caffeine": compounds[0]})

    assert decision.compound_role == "shared_or_unlinked"
    assert decision.attribution_method == "rule_retrieval_compound_without_text_match"
    assert decision.llm_required is False


def test_ddi_probe_substrate_is_not_perpetrator() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000014",
            "preferred_name": "midazolam",
            "normalized_name": "midazolam",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000014",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000014",
        "compound_name": "midazolam",
        "source_id": "clinicaltrials",
        "parameter": "CYP3A4",
        "source_text": "The trial assessed CYP3A4 inhibition by aramchol using the probe substrates midazolam and atorvastatin.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"midazolam": compounds[0]})

    assert decision.compound_role == "victim_drug"
    assert decision.llm_required is False


def test_ddi_inhibited_by_phrase_is_not_perpetrator() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000015",
            "preferred_name": "metformin",
            "normalized_name": "metformin",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000015",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000015",
        "compound_name": "metformin",
        "source_id": "pubmed",
        "parameter": "OCT1",
        "source_text": "A typical OCT1 substrate metformin was inhibited by pazopanib with an IC50 value.",
    }

    decision = decide_attribution(row, terms, {compounds[0]["compound_id"]: compounds[0]}, {"metformin": compounds[0]})

    assert decision.compound_role == "victim_drug"
    assert decision.llm_required is False


def test_caffeine_combination_product_ddi_is_not_direct_perpetrator() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000016",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000016",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000016",
        "compound_name": "caffeine",
        "source_id": "dailymed",
        "parameter": "CYP3A4",
        "source_text": (
            "CYP3A4 inhibitor or inducer at frequent intervals for signs and symptoms "
            "when BUTALBITAL, ASPIRIN, CAFFEINE and CODEINE PHOSPHATE is used in conjunction with inhibitors and inducers of CYP3A4."
        ),
    }

    issue = final_evidence_quality_issue(row, compound_name="caffeine", matched_text="caffeine", compound_role="perpetrator_drug")

    assert issue is not None
    assert issue["method"] == "rule_ddi_direction_uncertain"


def test_caffeine_combination_product_cns_depression_is_not_direct_perpetrator() -> None:
    compounds = [
        {
            "compound_id": "00000000-0000-0000-0000-000000000017",
            "preferred_name": "caffeine",
            "normalized_name": "caffeine",
            "synonyms": [],
            "pubchem_cid": None,
            "chembl_id": None,
            "rxnorm_rxcui": None,
        }
    ]
    terms = build_compound_terms(compounds)
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000017",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000017",
        "compound_name": "caffeine",
        "source_id": "dailymed",
        "parameter": "CNS depression",
        "source_text": (
            "Butalbital, acetaminophen and caffeine may enhance the effects of other narcotic analgesics, "
            "tranquilizers, sedative-hypnotics, or other CNS depressants, causing increased CNS depression."
        ),
    }

    issue = final_evidence_quality_issue(row, compound_name="caffeine", matched_text="caffeine", compound_role="perpetrator_drug")

    assert issue is not None
    assert issue["method"] == "rule_ddi_direction_uncertain"


def test_caffeine_combination_product_hypoglycemia_is_not_direct_perpetrator() -> None:
    row = {
        "evidence_id": "10000000-0000-0000-0000-000000000018",
        "evidence_type": "ddi",
        "compound_id": "00000000-0000-0000-0000-000000000018",
        "compound_name": "caffeine",
        "source_id": "dailymed",
        "parameter": "hypoglycemia",
        "source_text": (
            "Oral antidiabetic agents and insulin, causing hypoglycemia by contributing an additive effect, "
            "if dosage of Butalbital, Aspirin, and Caffeine Capsules exceeds maximum recommended daily dosage."
        ),
    }

    issue = final_evidence_quality_issue(row, compound_name="caffeine", matched_text="Caffeine", compound_role="perpetrator_drug")

    assert issue is not None
    assert issue["method"] == "rule_ddi_direction_uncertain"
