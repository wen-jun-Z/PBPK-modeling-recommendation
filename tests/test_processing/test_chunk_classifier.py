from __future__ import annotations

from pbpk_kb.processing.chunk_classifier import (
    classify_chunk,
    prepare_llm_input,
    sentence_fingerprint,
    should_run_rule_extraction,
    source_assertion_completeness_error,
)


def test_compound_unit_periods_do_not_split_pk_results():
    from pbpk_kb.processing.chunk_classifier import _sentence_spans, source_quote_unit_boundary_error
    text = "The CL, volume of distribution, and T1/2 of caffeine were 0.0476 L. h -1, 1.1081 L, 16.2284 h, respectively. The study ended."
    spans = _sentence_spans(text)
    assert len(spans) == 2
    assert spans[0].text.endswith('respectively.')
    quote = text[:text.index(' h -1')]
    assert source_quote_unit_boundary_error(quote, text) == 'source_quote_truncated_inside_compound_unit'
    normal = "Caffeine volume of distribution was 0.6 L. Healthy volunteers were studied."
    assert len(_sentence_spans(normal)) == 2
    assert source_quote_unit_boundary_error(normal.split(' Healthy')[0], normal) is None


def test_compound_dose_and_auc_unit_punctuation_is_preserved():
    from pbpk_kb.processing.chunk_classifier import _sentence_spans
    for text in ["Caffeine was administered at 5 mg. kg -1 daily.",
                 "Caffeine AUC was 20 ng. h/mL.",
                 "Caffeine clearance was 9.5 (mean +/- s.d.) L/h."]:
        assert [span.text for span in _sentence_spans(text)] == [text]


def test_machine_pubchem_json_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "pubchem",
        "source_location": "pubchem:compound.json#json_path=$#chunk=1",
        "chunk_text": '{"PC_Compounds":[{"atoms":{"aid":[1,2,3],"element":[6,7,8]},"bonds":{"aid1":[1,2],"aid2":[2,3]},"coords":[1,2,3]}]}',
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "structured_or_machine_chunk"
    assert should_run_rule_extraction(chunk) is False


def test_natural_language_pk_chunk_is_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "source_location": "dailymed:label.xml#xml_path=/section#chunk=1",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": "After oral administration, the mean Cmax was 53 ng/mL and the AUC increased in patients with hepatic impairment.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is True
    assert decision.status == "llm_required"
    assert should_run_rule_extraction(chunk) is True


# [FIX 2026-07-02 · claude-opus-4-8] regression: month "Oct" must not be treated as the OCT transporter signal.
def test_month_oct_is_not_an_evidence_signal() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "source_location": "clinicaltrials:studies.json#json_path=$#chunk=1",
        "chunk_text": "The study first submit date was Oct 2015 and the last update was posted in Oct 2016 by the sponsor investigator.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.evidence_signal is False


# [FIX 2026-07-02 · claude-opus-4-8] regression: a real OCT1 transporter mention still signals evidence.
def test_oct1_transporter_is_an_evidence_signal() -> None:
    chunk = {
        "source_id": "pubmed",
        "source_location": "pubmed:abstract.txt#chunk=1",
        "chunk_text": "Metformin is a substrate of the OCT1 transporter and its renal clearance depends on MATE1 activity in patients.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.evidence_signal is True


def test_pubmed_search_result_list_is_skipped() -> None:
    chunk = {
        "source_id": "pubmed",
        "source_location": "pubmed:search.json#json_path=$#chunk=1",
        "chunk_text": '{"esearchresult":{"count":"100","retmax":"20","idlist":["12345678","23456789","34567890","45678901","56789012","67890123","78901234","89012345"]}}',
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "skipped_search_result_list"
    assert should_run_rule_extraction(chunk) is False


def test_root_json_payload_is_not_sent_to_llm_even_when_it_contains_evidence_terms() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "source_location": "clinicaltrials:studies.json#json_path=$#chunk=1",
        "chunk_text": '{"protocolSection":{"briefTitle":"Pharmacokinetics of midazolam","statusModule":{"studyFirstSubmitDate":"2026-01-01"}}}',
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "structured_or_machine_chunk"


def test_clinicaltrials_protocol_summary_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "compound": "midazolam",
        "source_location": "clinicaltrials:studies.json#json_path=$.protocolSection.descriptionModule.briefSummary#chunk=1",
        "chunk_text": "This study evaluates the pharmacokinetics of microdose midazolam in renal impairment patients after oral administration.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False


def test_clinicaltrials_posted_result_field_can_be_sent_to_llm() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "compound": "midazolam",
        "source_location": (
            "clinicaltrials:study.json#json_path=$.resultsSection.outcomeMeasuresModule."
            "outcomeMeasures[0].description#chunk=1"
        ),
        "chunk_text": "The posted result reports that midazolam reached a mean Cmax of 53 ng/mL after oral dosing.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is True


def test_clinicaltrials_free_text_without_target_compound_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "compound": "midazolam",
        "source_location": "clinicaltrials:studies.json#json_path=$.protocolSection.descriptionModule.briefSummary#chunk=1",
        "chunk_text": "This study evaluates renal impairment and CYP3A4 activity in healthy subjects using a probe cocktail.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "structured_or_machine_chunk"


def test_pubmed_abstract_without_target_compound_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "pubmed",
        "compound": "warfarin",
        "source_location": "pubmed:efetch.xml#xml_path=//abstract[1]#chunk=1",
        "chunk_text": "The pharmacokinetics of probe drugs were measured by AUC and Cmax after oral administration.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False


def test_pubmed_esummary_metadata_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "pubmed",
        "source_location": "pubmed:esummary.fcgi#xml_path=/#chunk=1",
        "chunk_text": '"uid":"12345678","title":"Midazolam pharmacokinetics","articleids":[{"idtype":"pubmed","value":"12345678"}],"recordstatus":"PubMed - indexed for MEDLINE","pubstatus":"4","fulljournalname":"Journal"',
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "structured_or_machine_chunk"


def test_hpa_navigation_filter_text_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "hpa",
        "source_location": "hpa:page.html#css=body#chunk=1",
        "chunk_text": "Search Fields Category Tissue Tau score Downloadable data Category Cell type Cluster 14:Adipose tissue - Angiogenesis Contact NEWS HELP & FAQ",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "structured_or_machine_chunk"


def test_dailymed_warning_without_pk_or_ddi_signal_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "source_location": "dailymed:label.xml#xml_path=//component[72]#chunk=1",
        "metadata_json": {"section_title": ""},
        "chunk_text": "WARNINGS Personnel and equipment for monitoring and resuscitation are required. Respiratory depression and respiratory arrest have occurred during sedation.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False


def test_dailymed_empty_mid_section_quantitative_text_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "source_location": "dailymed:label.xml#xml_path=//component[72]#chunk=11",
        "metadata_json": {"section_title": ""},
        "chunk_text": "In adults, midazolam is approximately 97% bound to plasma protein and is metabolized by CYP3A4 before urinary excretion.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False


def test_dailymed_empty_chunk_with_core_heading_can_be_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "source_location": "dailymed:label.xml#xml_path=//component[72]#chunk=1",
        "metadata_json": {"section_title": ""},
        "chunk_text": "CLINICAL PHARMACOLOGY Pharmacokinetics: The mean Cmax was 53 ng/mL after oral administration.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is True


def test_xml_markup_chunk_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "pmc_oa",
        "source_location": "pmc_oa:article.xml#xml_path=/#chunk=1",
        "chunk_text": "<article><front><journal-meta><journal-id>test</journal-id></journal-meta><article-meta><title-group><article-title>Midazolam PK</article-title></title-group></article-meta></front><body><sec><title>Results</title><p>The AUC was measured in plasma.</p></sec></body></article>",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False
    assert decision.status == "structured_or_machine_chunk"


def test_high_precision_input_keeps_quantitative_pk_assertion() -> None:
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": "After a 10 mg oral dose, midazolam reached a mean Cmax of 53 ng/mL.",
    }

    prepared = prepare_llm_input(chunk, "midazolam")

    assert prepared.eligible is True
    assert prepared.sentences == (chunk["chunk_text"],)
    assert len(prepared.sentence_fingerprints) == 1


def test_high_precision_input_keeps_qualitative_metabolism_relation() -> None:
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": "Midazolam is primarily metabolized by CYP3A4 and CYP3A5.",
    }

    prepared = prepare_llm_input(chunk, "midazolam")

    assert prepared.eligible is True


def test_high_precision_input_rejects_study_intent_and_dose_only_text() -> None:
    intent = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": "The purpose of this study was to assess the pharmacokinetics of midazolam after oral administration.",
    }
    dose_only = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Methods"},
        "chunk_text": "Midazolam was administered intravenously at 0.075 mg/kg to healthy subjects.",
    }

    assert prepare_llm_input(intent, "midazolam").eligible is False
    assert prepare_llm_input(dose_only, "midazolam").status == "skipped_non_result_section"


def test_high_precision_input_combines_adjacent_target_context_and_result() -> None:
    text = (
        "Midazolam pharmacokinetic parameters were evaluated in healthy volunteers. "
        "The mean clearance was 323 mL/min and half-life was 2.3 h."
    )
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": text,
    }

    prepared = prepare_llm_input(chunk, "midazolam")

    assert prepared.eligible is True
    assert prepared.sentences == (text,)


def test_dailymed_dosing_section_is_excluded_even_with_population_terms() -> None:
    chunk = {
        "source_id": "dailymed",
        "compound": "midazolam",
        "source_location": "dailymed:label.xml#xml_path=//section[7]#chunk=1",
        "metadata_json": {"section_title": "2.3 Dosing Recommendations"},
        "chunk_text": "Pediatric patients receive midazolam at 0.05 mg/kg and require continuous monitoring.",
    }

    decision = classify_chunk(chunk, min_chars=20)

    assert decision.llm_required is False


def test_sentence_fingerprint_normalizes_spacing_and_punctuation_only_for_comparison() -> None:
    first = sentence_fingerprint("midazolam", "Midazolam clearance was 323 mL/min.")
    second = sentence_fingerprint("MIDAZOLAM", "Midazolam   clearance was 323 mL/min!")

    assert first == second


def test_high_precision_input_rejects_hypothesized_pk_change() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "metadata_json": {"section_title": "briefsummary"},
        "chunk_text": (
            "The investigators hypothesize that nizatidine will reduce the renal clearance of metformin "
            "and increase plasma concentrations."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_high_precision_input_rejects_parameter_owned_by_another_drug() -> None:
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Discussion"},
        "chunk_text": (
            "The metabolic pathway of telmisartan did not overlap that of rosuvastatin, "
            "and the half-life of telmisartan is over 20 h."
        ),
    }

    assert prepare_llm_input(chunk, "rosuvastatin").eligible is False


def test_high_precision_input_keeps_explicit_ddi_when_parameter_belongs_to_other_drug() -> None:
    chunk = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": "Co-administration with omeprazole resulted in an increase in the escitalopram AUC.",
    }

    assert prepare_llm_input(chunk, "omeprazole").eligible is True


def test_compact_eligibilitycriteria_section_is_excluded() -> None:
    chunk = {
        "source_id": "clinicaltrials",
        "compound": "midazolam",
        "source_location": "clinicaltrials:study.json#json_path=$.eligibilityCriteria#chunk=1",
        "metadata_json": {"section_title": "eligibilitycriteria"},
        "chunk_text": "Medication that could influence the pharmacokinetics of midazolam was prohibited before enrollment.",
    }

    assert classify_chunk(chunk, min_chars=20).llm_required is False


def test_combination_label_does_not_assign_other_component_mechanism_to_target() -> None:
    chunk = {
        "source_id": "openfda_drug_label",
        "metadata_json": {"section_title": "clinicalpharmacology"},
        "chunk_text": (
            "Studies were conducted with the individual components linagliptin and metformin. "
            "Linagliptin is a weak inhibitor of CYP3A4 and is not an inducer of CYP1A2."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_pharmacodynamic_camp_result_is_not_adme_evidence() -> None:
    chunk = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": "Caffeine increased intracellular cAMP titers by about 35% in fluoride-inhibited sperm.",
    }

    assert prepare_llm_input(chunk, "caffeine").eligible is False


def test_not_studied_label_statement_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": "Digoxin pharmacokinetics has not been formally studied according to race.",
    }

    assert prepare_llm_input(chunk, "digoxin").eligible is False


def test_osp_bibliography_entry_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "osp_pbpk_model_library",
        "metadata_json": {"section_title": ""},
        "chunk_text": "Et al. Application of physiologically-based pharmacokinetic models to digoxin clearance.",
    }

    assert prepare_llm_input(chunk, "digoxin").eligible is False


def test_negative_context_is_not_merged_with_valid_following_result() -> None:
    result = (
        "Geriatric studies show that metformin clearance is decreased and the half-life is prolonged "
        "compared with healthy young subjects."
    )
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": f"No pharmacokinetic studies of metformin have been conducted in hepatic impairment. {result}",
    }

    prepared = prepare_llm_input(chunk, "metformin")

    assert prepared.eligible is True
    assert prepared.sentences == (result,)


def test_standalone_result_is_not_duplicated_as_context_plus_result() -> None:
    result = "Midazolam clearance was 323 mL/min and its half-life was 2.3 h."
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": f"The pharmacokinetics of midazolam were evaluated in volunteers. {result}",
    }

    prepared = prepare_llm_input(chunk, "midazolam")

    assert prepared.sentences == (result,)


def test_measurement_description_without_result_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": (
            "The absorption of a theophylline solution delivered to different gastrointestinal sites "
            "has been determined in three volunteers."
        ),
    }

    assert prepare_llm_input(chunk, "theophylline").eligible is False


def test_target_treatment_does_not_own_endogenous_metabolite_excretion() -> None:
    chunk = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": (
            "Treatment with metformin was associated with decreases in the excretion rates of "
            "4-hydroxyestradiol and 2-methoxyestradiol."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_sentence_ending_with_comma_is_treated_as_truncated() -> None:
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": (
            "Water soluble drugs such as digoxin show a small volume of distribution and increased "
            "plasma concentrations with reduced total body water, and thus,"
        ),
    }

    assert prepare_llm_input(chunk, "digoxin").eligible is False


def test_osp_title_fragment_without_model_result_relation_is_not_sent_to_llm() -> None:
    fragment = {
        "source_id": "osp_pbpk_model_library",
        "metadata_json": {"section_title": ""},
        "chunk_text": "of the disposition of midazolam in surgical patients by a physiologically based pharmacokinetic model.",
    }
    assertion = {
        "source_id": "osp_pbpk_model_library",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": "The PBPK model for midazolam was developed and verified with clinical pharmacokinetic data.",
    }

    assert prepare_llm_input(fragment, "midazolam").eligible is False
    assert prepare_llm_input(assertion, "midazolam").eligible is True


def test_unpunctuated_subordinate_fragment_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": (
            "Although metformin AUC increased by approximately 50% when given"
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_clinical_recommendation_is_not_a_pk_result() -> None:
    result = "Total plasma clearance of metformin is decreased in healthy elderly subjects."
    recommendation = (
        "Metformin treatment should not be initiated in patients over 80 years unless creatinine "
        "clearance is measured."
    )
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": f"{result} {recommendation}",
    }

    prepared = prepare_llm_input(chunk, "metformin")

    assert prepared.sentences == (result,)


def test_predicted_negative_ddi_is_not_sent_without_pbpk_model_context() -> None:
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": (
            "No drug interaction or significant increase in metformin AUC was predicted after "
            "coadministration."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_explicit_pbpk_model_prediction_can_be_sent_to_llm() -> None:
    chunk = {
        "source_id": "osp_pbpk_model_library",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": "The midazolam PBPK model predicted the observed plasma concentration profile.",
    }

    assert prepare_llm_input(chunk, "midazolam").eligible is True


def test_distant_cover_metadata_does_not_create_pbpk_model_assertion() -> None:
    chunk = {
        "source_id": "osp_pbpk_model_library",
        "metadata_json": {"section_title": ""},
        "chunk_text": (
            "Version 2.0 based on Model Snapshot and Evaluation Plan https://example.test/Midazolam-Model "
            "OSP Version 12.3 Qualification Framework 3.7 Building and evaluation of a PBPK model "
            "for Midazolam in healthy adults"
        ),
    }

    assert prepare_llm_input(chunk, "midazolam").eligible is False


def test_unbalanced_parenthesis_is_treated_as_truncated() -> None:
    chunk = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": "Warfarin AUC increased by 57% (90% confidence interval, 32% to 88%.",
    }

    assert prepare_llm_input(chunk, "warfarin").eligible is False


def test_subject_creatinine_clearance_is_not_target_drug_clearance() -> None:
    chunk = {
        "source_id": "pmc_oa",
        "metadata_json": {"section_title": "Results"},
        "chunk_text": (
            "Metformin-intolerant patients had lower creatinine clearance and received a lower "
            "metformin dose."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_explicit_target_drug_renal_clearance_remains_eligible() -> None:
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": "The renal clearance of metformin was 510 mL/min in healthy subjects.",
    }

    assert prepare_llm_input(chunk, "metformin").eligible is True


def test_missing_subject_suggest_that_fragment_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "openfda_drug_label",
        "metadata_json": {"section_title": "pharmacokinetics"},
        "chunk_text": (
            "suggest that total plasma clearance of metformin is decreased and the half-life is "
            "prolonged in elderly subjects."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_sentence_truncated_after_in_is_not_sent_to_llm() -> None:
    chunk = {
        "source_id": "dailymed",
        "metadata_json": {"section_title": "Pharmacokinetics"},
        "chunk_text": "The plasma half-life of midazolam was approximately two-fold higher in",
    }

    assert prepare_llm_input(chunk, "midazolam").eligible is False


def test_real_truncated_source_assertions_are_rejected_before_llm() -> None:
    truncated = (
        "The amount of midazolam excreted unchanged in the urine after",
        "CYP3A5 expressors exhibited a 1.27-fold increase in midazolam clearance compared to non-expressors, while",
        "The half-life of midazolam increased from 5 to 7 hours when midazolam",
        "The half-life of midazolam increased from 5 to 7 hours when midazolam was taken",
        "The half-life of midazolam increased from 5 to 7 hours when midazolam was taken in conjunction",
        "The mean half-life of midazolam increased 2.5-fold in the alcoholic",
        "Midazolam AUC increased by about 50% without affecting the T",
    )

    for source_text in truncated:
        assert source_assertion_completeness_error(
            source_text,
            target_compound="midazolam",
        ) is not None


def test_complete_unpunctuated_assertions_are_retained() -> None:
    complete = (
        "The amount of midazolam excreted unchanged in the urine is less than 0.5%",
        "The renal clearance of metformin is decreased",
        "Midazolam is primarily metabolized by CYP3A4 and CYP3A5",
    )

    for source_text in complete:
        assert source_assertion_completeness_error(
            source_text,
            target_compound="midazolam" if "Midazolam" in source_text or "midazolam" in source_text else "metformin",
        ) is None


def test_flattened_table_row_does_not_require_terminal_punctuation() -> None:
    table_text = (
        "Table 6 Effect of Metformin on Coadministered Drug Systemic Exposure "
        "Geometric Mean Ratio No Effect = 1 AUC Cmax Cimetidine 400 mg "
        "Metformin 850 mg cimetidine 0.95 § 1.01 ¶ Ratio of arithmetic means"
    )

    assert source_assertion_completeness_error(table_text, target_compound="metformin") is not None
    assert source_assertion_completeness_error(
        table_text,
        target_compound="metformin",
        allow_flattened_table=True,
    ) is None


def test_unparsed_flattened_table_is_deferred_before_llm() -> None:
    chunk = {
        "source_id": "openfda_drug_label",
        "metadata_json": {"section_title": "pharmacokinetics"},
        "chunk_text": (
            "Table 6 Effect of Metformin on Coadministered Drug Systemic Exposure "
            "Geometric Mean Ratio No Effect = 1 AUC Cmax Glyburide 5 mg "
            "Metformin 500 mg 0.98 ‡ 0.99 ‡ Propranolol 40 mg 850"
        ),
    }

    prepared = prepare_llm_input(chunk, "metformin")

    assert prepared.eligible is False
    assert prepared.status == "deferred_unparsed_flattened_table"


def test_dense_numeric_range_table_without_markers_is_deferred() -> None:
    chunk = {
        "source_id": "openfda_drug_label",
        "metadata_json": {"section_title": "pharmacokinetics"},
        "chunk_text": (
            "Concurrent illness Acute pulmonary edema 0.33 (0.07-2.45) 19 (3.1-82) "
            "COPD stable non-smoker 0.54 (0.44-0.64) 11 (9.4-12.6) "
            "Cystic fibrosis 1.25 (0.31-2.2) 6.0 (1.8-10.2) "
            "Liver disease cirrhosis 0.31 (0.1-0.7) 32 (10-56) "
            "Pregnancy 8.5 (3.1-13.9) 8.8 (3.8-13.8) 13.0 (8.4-17.6) "
            "Theophylline clearance is increased and half-life decreased."
        ),
    }

    prepared = prepare_llm_input(chunk, "theophylline")

    assert prepared.eligible is False
    assert prepared.status == "deferred_unparsed_flattened_table"


def test_parameter_estimation_method_is_not_observed_evidence() -> None:
    chunk = {
        "source_id": "pubmed",
        "metadata_json": {"section_title": "Abstract"},
        "chunk_text": (
            "Multi-linear regression analysis was performed to establish a model to estimate "
            "AUC(0-24 h) and Cmax of metformin by LSS method."
        ),
    }

    assert prepare_llm_input(chunk, "metformin").eligible is False


def test_explicit_abstract_truncation_notice_is_not_a_complete_source_assertion():
    from pbpk_kb.processing.chunk_classifier import source_assertion_completeness_error
    assert source_assertion_completeness_error(
        "Caffeine AUC increased with alcohol.(ABSTRACT TRUNCATED AT 250 WORDS)",
        target_compound="caffeine",
    ) == "explicit_source_truncation_notice"
    assert source_assertion_completeness_error(
        "A truncated protein inhibited caffeine metabolism.", target_compound="caffeine"
    ) is None


def test_real_mixed_metformin_sd_table_and_lettered_footnotes_are_deferred() -> None:
    # Regression for request 95db3c77-36f2-5006-a80c-18e2a8bc2e70. Its table
    # has no surviving title or symbol footnotes, and the pediatric prose begins
    # before any reliable sentence boundary separates it from the numeric rows.
    text = (
        "(±98) Renal-impaired adults: 850 mg single dose Mild (CLcr g 61-90 mL/min) "
        "(5) 1.86 (±0.52) 3.20 (±0.45) 384 (±122) Moderate (CLcr 31-60 mL/min) "
        "(4) 4.12 (±1.83) 3.75 (±0.50) 108 (±57) Severe (CLcr 10-30 mL/min) "
        "(6) 3.93 (±0.92) 4.01 (±1.10) 130 (±90) a All doses given fasting except "
        "the first 18 doses of the multiple dose studies b Peak plasma concentration "
        "c Time to peak plasma concentration d Combined results (average means) of "
        "five studies: mean age 32 years (range 23-59 years) e Kinetic study done "
        "following dose 19, given fasting f Elderly subjects, mean age 71 years "
        "(range 65-81 years) g CLcr = creatinine clearance normalized to body surface "
        "area of 1.73 m 2 Pediatrics After administration of a single oral Metformin "
        "hydrochloride tablets 500 mg with food, geometric mean metformin C max and "
        "AUC differed less than 5% between pediatric type 2 diabetic patients (12-16 "
        "years of age) and gender- and weight-matched healthy adults (20-45 years "
        "of age), all with normal renal function."
    )
    for source_text in (text, text.replace("±", "+/-")):
        chunk = {
            "source_id": "dailymed",
            "metadata_json": {"section_title": "12.3 Pharmacokinetics"},
            "chunk_text": source_text,
            "parsed_table_id": None,
        }
        prepared = prepare_llm_input(chunk, "metformin")
        assert prepared.eligible is False
        assert prepared.status == "deferred_unparsed_flattened_table"
        assert prepared.sentences == ()
        assert chunk["chunk_text"] == source_text


def test_percent_human_pk_prose_and_labeled_sd_results_remain_eligible() -> None:
    sentences = (
        "After administration of a single oral Metformin hydrochloride tablets "
        "500 mg with food, geometric mean metformin C max and AUC differed less "
        "than 5% between pediatric type 2 diabetic patients (12-16 years of age) "
        "and gender- and weight-matched healthy adults (20-45 years of age), "
        "all with normal renal function.",
        "Metformin bioavailability was 50% to 60% after a 500 mg oral dose in healthy adults.",
        "In healthy adults, metformin Cmax was 1.86 mg/L (±0.52), AUC was "
        "3.20 mg h/L (±0.45), and renal clearance was 384 mL/min (±122).",
    )
    for text in sentences:
        chunk = {
            "source_id": "dailymed",
            "metadata_json": {"section_title": "12.3 Pharmacokinetics"},
            "chunk_text": text,
        }
        prepared = prepare_llm_input(chunk, "metformin")
        assert prepared.eligible is True, (text, prepared)
        assert prepared.sentences == (text,)


def test_real_metformin_table_with_thousands_and_two_unparenthesized_sd_columns_is_deferred() -> None:
    text = (
        "Metformin HCl Tablets Steady-State Pharmacokinetic Parameters at 4 Weeks "
        "Pharmacokinetic Parameters (mean ± SD) Metformin hydrochloride extended-release "
        "tablets 2,000 mg (administered once daily after dinner) Metformin HCl tablets* "
        "2,000 mg (1,000 mg twice daily) AUC 0-24hr (ng•hr/mL) 26,811 ± 7,055 "
        "27,371 ± 5,781 T max (hr) 6 (3-10) 3 (1-8) C max (ng/mL) 2,849 ± 797 "
        "1,820 ± 370 In four single-dose studies and one multiple-dose study, the "
        "bioavailability of metformin hydrochloride extended-release tablets 2,000 mg "
        "given once daily, in the evening, under fed conditions [as measured by AUC] "
        "was similar to the same total daily dose administered as metformin HCl "
        "tablets 1,000 mg given twice daily. The geometric mean ratios (metformin "
        "hydrochloride extended-release tablets/ metformin HCl tablets) of AUC 0-24hr , "
        "AUC 0-72hr , and AUC 0-inf for these five studies ranged from 0.96 to 1.08."
    )
    chunk = {"source_id": "openfda_drug_label", "chunk_text": text,
             "metadata_json": {"section_title": "clinical_pharmacology"}}
    prepared = prepare_llm_input(chunk, "metformin")
    assert prepared.status == "deferred_unparsed_flattened_table"
    assert prepared.eligible is False and prepared.sentences == ()
    assert chunk["chunk_text"] == text


def test_real_theophylline_table_with_word_delimited_ranges_is_deferred() -> None:
    text = (
        "associated with acute viral respiratory illness (children 9 to 15 years) "
        "NR 7.0 (1.0 to 13) Liver disease cirrhosis 0.31 (0.1 to 0.7) 32 (10 to 56) "
        "acute hepatitis 0.35 (0.25 to 0.45) 19.2 (16.6 to 21.8) cholestasis "
        "0.65 (0.25 to 1.45) 14.4 (5.7 to 31.8) Pregnancy 1st trimester NR "
        "8.5 (3.1 to 13.9) 2nd trimester NR 8.8 (3.8 to 13.8) 3rd trimester NR "
        "13.0 (8.4 to 17.6) Sepsis with multi-organ failure 0.47 (0.19 to 1.9) "
        "18.8 (6.3 to 24.1) Thyroid disease hypothyroid 0.38 (0.13 to 0.57) "
        "11.6 (8.2 to 25) hyperthyroid 0.8 (0.68 to 0.97) 4.5 (3.7 to 5.6) "
        "Note: In addition to the factors listed above, theophylline clearance is "
        "increased and half-life decreased by low carbohydrate/high protein diets, "
        "parenteral nutrition, and daily consumption of charcoal-broiled beef. "
        "A high carbohydrate/low protein diet can decrease the clearance and "
        "prolong the half-life of theophylline. Absorption: Theophylline administered "
        "in the fed state is completely absorbed after oral administration."
    )
    chunk = {"source_id": "openfda_drug_label", "chunk_text": text,
             "metadata_json": {"section_title": "clinical_pharmacology"}}
    prepared = prepare_llm_input(chunk, "theophylline")
    assert prepared.status == "deferred_unparsed_flattened_table"
    assert prepared.eligible is False and prepared.sentences == ()
    assert chunk["chunk_text"] == text


def test_two_group_sd_prose_requires_matrix_structure_before_deferral() -> None:
    from pbpk_kb.processing.chunk_classifier import _looks_like_flattened_table_row

    prose = (
        "The pharmacokinetic parameters (mean ± SD) of metformin were measured in "
        "two groups. Metformin AUC increased by 2%, from 26,811 ± 7,055 ng hr/mL "
        "in healthy adults to 27,371 ± 5,781 ng hr/mL in patients."
    )
    assert _looks_like_flattened_table_row(prose) is False
    prepared = prepare_llm_input({"source_id": "pubmed", "chunk_text": prose}, "metformin")
    assert prepared.eligible is True
    assert prepared.status != "deferred_unparsed_flattened_table"
    # A pair of numbers alone is insufficient: there must also be a PK matrix
    # header. Other completeness rules can still reject a source fragment.
    no_header = "Metformin AUC 26,811 ± 7,055 27,371 ± 5,781 was compared between groups."
    assert _looks_like_flattened_table_row(no_header) is False


def test_limited_parenthesized_to_ranges_in_complete_pk_prose_remain_eligible() -> None:
    text = (
        "Theophylline clearance was 0.31 L/h/kg (0.1 to 0.7) in healthy adults, "
        "and theophylline half-life was 8.5 hours (3.1 to 13.9)."
    )
    prepared = prepare_llm_input({"source_id": "pubmed", "chunk_text": text}, "theophylline")
    assert prepared.eligible is True
    assert prepared.sentences == (text,)


def test_prepositional_pk_fragment_cannot_borrow_a_predicate_from_the_next_sentence() -> None:
    fragment = "in repaglinide median AUC and C max respectively."
    following = (
        "The decreases were from 54 ng/mL*hr to 11 ng/mL*hr for AUC and from "
        "35 ng/mL to 7.5 ng/mL for C max ."
    )
    retained = "The increase in repaglinide C max was from 40.5 ng/mL to 47.4 ng/mL."
    text = fragment + " " + following + " " + retained
    assert source_assertion_completeness_error(text, target_compound="repaglinide") == (
        "prepositional_result_fragment_without_predicate"
    )
    prepared = prepare_llm_input({"source_id": "dailymed", "chunk_text": text}, "repaglinide")
    assert prepared.sentences == (retained,)
    # The preceding immutable chunk supplies exactly this missing half-sentence.
    complete = (
        "PRANDIN administered by itself (after 7 days of once daily rifampin "
        "600 mg) resulted in an 80% and 79% decrease " + fragment
    )
    assert source_assertion_completeness_error(complete, target_compound="repaglinide") is None


def test_bare_pk_subject_fragment_is_excluded_while_other_complete_sentences_remain() -> None:
    fragment = (
        "exposure (AUC 0-12 ) and C max increased by 1.5 fold and 1.4 fold, "
        "respectively, and quinidine exposure (AUC 0-12 ) and C max increased by "
        "1.4 fold and 1.3 fold, respectively, and dextrorphan exposure (AUC 0-12 ) "
        "and C max decreased by 14% and 18%, respectively, and paroxetine exposure "
        "(AUC 0-24 ) and C max increased by 2.3 fold and 2.0 fold, respectively, "
        "when paroxetine was added to the combination dose of dextromethorphan/quinidine "
        "at steady state (Group 2)."
    )
    retained = (
        "When the combination dose of dextromethorphan/quinidine was added to "
        "paroxetine at steady state (Group 1), paroxetine exposure (AUC 0-24 ) "
        "and C max increased by 1.7 fold and 1.5 fold, respectively, while "
        "dextromethorphan and quinidine exposure did not change significantly "
        "and dextrorphan exposure (AUC 0-12 ) and C max decreased by 34% and 33%, respectively."
    )
    assert source_assertion_completeness_error(fragment, target_compound="dextromethorphan") == (
        "leading_pk_parameter_without_subject"
    )
    prepared = prepare_llm_input(
        {"source_id": "openfda_drug_label", "chunk_text": fragment + " " + retained}, "dextromethorphan",
    )
    assert prepared.sentences == (retained,)
    # This drug name is the actual final word of the preceding immutable chunk.
    assert source_assertion_completeness_error(
        "Dextromethorphan " + fragment, target_compound="dextromethorphan",
    ) is None


def test_complete_prepositional_and_explicit_parameter_ownership_sentences_are_preserved() -> None:
    complete = (
        "In healthy adults, repaglinide AUC and Cmax increased by 20% and 10%, respectively.",
        "In repaglinide studies, AUC and Cmax were 20 ng h/mL and 10 ng/mL, respectively.",
        "Exposure (AUC) of repaglinide increased by 20% in healthy adults.",
        "Clearance of repaglinide was 12 L/h in healthy adults.",
        "Exposure of midazolam increased by 20% during coadministration with repaglinide.",
    )
    for text in complete:
        assert source_assertion_completeness_error(text, target_compound="repaglinide") is None, text
    assert source_assertion_completeness_error(
        "Exposure (AUC) 20 ng h/mL", target_compound="repaglinide", allow_flattened_table=True,
    ) is None


def test_unit_bearing_pk_columns_without_mean_sd_caption_are_deferred() -> None:
    # Real DailyMed input that timed out when a flattened table was sent as prose.
    text = 'values following single and multiple doses of 400 mg moxifloxacin given by 1 hour IV infusion C max (mg/L) AUC (mg•h/L) Half-life (hr) Single Dose IV Healthy young male/female (n = 56) 3.9 ± 0.9 39.3 ± 8.6 8.2 - 15.4 Patients (n = 118) Male (n = 64) 4.4 ± 3.7 Female (n = 54) 4.5 ± 2 < 65 years (n = 58) 4.6 ± 4.2 ≥ 65 years (n = 60) 4.3 ± 1.3 Multiple Dose IV Healthy young male (n = 8) 4.2 ± 0.8 38 ± 4.7 14.8 ± 2.2 Healthy elderly (n =12; 8 male, 4 female) 6.1 ± 1.3 48.2 ± 0.9 10.1 ± 1.6 Patients (n = 107) Male (n = 58) 4.2 ± 2.6 Female (n = 49) 4.6 ± 1.5 <65 years (n = 52) 4.1 ± 1.4 ≥65 years (n = 55) 4.7 ± 2.7 Plasma concentrations increase proportionately with dose up to the highest dose tested (1200 mg single oral dose).'
    for fragment in (text, text.replace("±", "+/-"), text.replace("C max", "Cmax")):
        chunk = {"source_id": "dailymed", "chunk_text": fragment}
        prepared = prepare_llm_input(chunk, "moxifloxacin")
        assert not prepared.eligible
        assert prepared.status == "deferred_unparsed_flattened_table"
        assert prepared.sentences == ()
        assert chunk["chunk_text"] == fragment
        parsed = prepare_llm_input({**chunk, "parsed_table_id": "verified-table"}, "moxifloxacin")
        assert parsed.status != "deferred_unparsed_flattened_table"


def test_pk_column_labels_require_numeric_matrix_and_do_not_capture_result_prose() -> None:
    from pbpk_kb.processing.chunk_classifier import _looks_like_flattened_table_row
    prose = (
        "Moxifloxacin C max (mg/L) was 3.9 ± 0.9, AUC (mg h/L) was 39.3 ± 8.6, "
        "and half-life (hr) was 14.8 ± 2.2 in healthy adults."
    )
    assert not _looks_like_flattened_table_row(prose)
    assert prepare_llm_input({"source_id": "dailymed", "chunk_text": prose}, "moxifloxacin").status != "deferred_unparsed_flattened_table"
    labels_only = "Moxifloxacin C max (mg/L) AUC (mg h/L) were measured in healthy adults."
    assert not _looks_like_flattened_table_row(labels_only)
