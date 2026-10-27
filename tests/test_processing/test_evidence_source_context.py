from hashlib import sha256

import pytest
from defusedxml import ElementTree as ET

from pbpk_kb.parsers.table_structure import parse_xml_table_element
from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
from pbpk_kb.processing.source_context import (
    analyte_mismatch_reason, enrich_source_context, repair_multilevel_headers,
)
from pbpk_kb.processing.table_recovery_stage import TableRecoveryStage, detect_parameter, find_unit, parse_table_value


def extract(xml, tmp_path, compound="losartan"):
    table = repair_multilevel_headers(parse_xml_table_element(ET.fromstring(xml)))
    stage = TableRecoveryStage(output_dir=tmp_path)
    stage.run_id = "run_test"
    stage.target_compounds = {compound}
    evidence = []
    for index, cells in enumerate(table.rows):
        row = dict(evidence_id="e", table_row_id=str(index), source_id="dailymed", compound=compound,
                   columns_json=table.columns, cells_json=cells, caption=table.caption,
                   row_text=" | ".join(cells), row_hash="hash", source_location="source#table=1",
                   cell_locations=[f"source#row={index}#col={c}" for c in range(len(cells))],
                   metadata_json={"structure_reliable": True})
        evidence += stage._extract_table_evidence(row, (compound, "dailymed_unique_active_ingredient"))
    return table, evidence, stage


def test_multilevel_parent_metabolite_and_population_columns(tmp_path):
    xml = """<table><tr><td colspan="5">Table 1 Pharmacokinetic Parameters</td></tr>
    <tr><td colspan="5">Age 6 to 16 Following Multiple Dosing</td></tr>
    <tr><td></td><td colspan="2">Adults given 50 mg once daily for 7 days N=12</td>
    <td colspan="2">Age 6 to 16 given 0.7 mg/kg once daily for 7 days N=25</td></tr>
    <tr><td></td><td>Parent</td><td>Active Metabolite</td><td>Parent</td><td>Active Metabolite</td></tr>
    <tr><td>AUC0-24*(ng·h/mL)</td><td>442 ±173</td><td>1685 ±452</td><td>368 ±169</td><td>1866 ±1076</td></tr>
    </table>"""
    table, values, stage = extract(xml, tmp_path)
    assert [v["value"] for v in values] == [442, 368]
    assert stage.counts["metabolite_cells_isolated"] == 2
    assert values[1]["dose"] == "0.7 mg/kg"
    assert values[0]["population"] == "Adults" and values[1]["population"] == "Age 6 to 16"
    assert values[1]["metadata_json"]["source_conditions"]["dosing_interval"] == "once daily"
    assert "Parent" in table.columns[3] and "Active Metabolite" in table.columns[4]
    assert all("1866" not in v["condition"] and "1685" not in v["condition"] for v in values)


def test_gabapentin_food_subheaders_survive_colspan(tmp_path):
    xml = """<table><tr><td rowspan="2">Pharmacokinetic Parameter (Mean ± SD)</td>
    <td colspan="2">Gabapentin Tablets 1,800 mg (2 X 900 mg tablets)</td></tr>
    <tr><td>High fat- high calorie fed state</td><td>Low fat- low calorie fed state</td></tr>
    <tr><td>Cmax (μg/mL)</td><td>9.49 ± 1.93</td><td>6.50 ± 2.16</td></tr></table>"""
    table, values, _ = extract(xml, tmp_path, "gabapentin")
    assert len(values) == 2
    assert "Low fat- low calorie fed state" in values[1]["condition"]
    assert "High fat" not in values[1]["condition"]
    assert values[1]["metadata_json"]["source_conditions"]["food_state"] == "Low fat- low calorie fed state"
    assert values[1]["dose"] == "1,800 mg"
    assert "9.49" not in values[1]["condition"]


def test_ciprofloxacin_route_interval_are_bound_to_value_column(tmp_path):
    xml = """<table><tr><td>Parameters</td><td>500 mg</td><td>400 mg</td><td>750 mg</td><td>400 mg</td></tr>
    <tr><td></td><td>every 12 hours, orally</td><td>every 12 hours, intravenously</td>
    <td>every 12 hours, orally</td><td>every 8 hours, intravenously</td></tr>
    <tr><td>Cmax,ss (μg/mL)</td><td>2.97</td><td>4.56</td><td>3.59</td><td>4.07</td></tr></table>"""
    _, values, _ = extract(xml, tmp_path, "ciprofloxacin")
    assert len(values) == 4
    assert values[2]["dose"] == "750 mg" and values[2]["route"] == "orally"
    assert values[3]["route"] == "intravenously"
    assert values[3]["metadata_json"]["source_conditions"]["dosing_interval"] == "every 8 hours"
    assert values[0]["metadata_json"]["parameter_header"] == "Cmax,ss (μg/mL)"


def test_endpoint_columns_do_not_consume_population_data_as_headers(tmp_path):
    xml = """<table><tr><th>Population</th><th>Cmax (ng/mL)</th></tr>
    <tr><td>Adults</td><td>10</td></tr><tr><td>Children</td><td>8</td></tr></table>"""
    table, values, _ = extract(xml, tmp_path)
    assert len(table.rows) == 2 and len(values) == 2


def test_auc_window_does_not_replace_auc_unit():
    assert detect_parameter("AUC0-24h,ss (μg•h/mL)") == "AUC"
    assert find_unit("AUC0-24h,ss (μg•h/mL)") == "μg•h/mL"
    assert detect_parameter("AUCt (µg.hr/mL)") == "AUC"
    assert detect_parameter("AUCinf (µg.hr/mL)") == "AUC"


def test_bedaquiline_table5_binding_retains_threshold_and_rejects_m2_clearance(tmp_path):
    xml = """<table><tr><th>Pharmacokinetic Parameters</th><th>Bedaquiline</th></tr>
    <tr><td>Percent bound to human plasma protein</td><td>greater than 99.9%.</td></tr>
    <tr><td>Apparent clearance of bedaquiline</td><td>2.62 L/h</td></tr>
    <tr><td>Apparent clearance of M2</td><td>4.95 L/h</td></tr>
    <tr><td>%Excreted unchanged in urine</td><td>less than or equal to 0.001% of the dose in clinical studies</td></tr>
    </table>"""
    _, values, stage = extract(xml, tmp_path, "bedaquiline")
    assert [(r["parameter"], r["value"]) for r in values] == [
        ("protein_binding", 99.9), ("clearance", 2.62), ("fraction_excreted_unchanged", .001)]
    assert values[0]["metadata_json"]["value_qualifier"] == ">"
    assert values[2]["metadata_json"]["value_qualifier"] == "<="
    assert stage.counts["metabolite_cells_isolated"] == 1
    assert "greater than" in values[0]["source_text"]


@pytest.mark.parametrize("text,qualifier", [("greater than 99.9%", ">"),
    ("greater than or equal to 99%", ">="), ("less than or equal to 0.001%", "<="),
    ("<=0.001%", "<="), (">=99%", ">="),
    ("less than 5%", "<"), ("approximately 5 hours", "~")])
def test_table_spelled_out_inequality_is_not_flattened_to_equality(text, qualifier):
    assert parse_table_value(text, unit_hint=None)[3] == qualifier


def test_metabolite_code_does_not_match_body_surface_unit():
    from pbpk_kb.processing.table_recovery_stage import is_metabolite_analyte_label
    assert is_metabolite_analyte_label("Apparent clearance of M2")
    assert is_metabolite_analyte_label("M2 Cmax")
    assert not is_metabolite_analyte_label("Clearance (L/h/m2)")


def test_descriptive_pk_rows_are_not_promoted_to_column_headers(tmp_path):
    # DailyMed Table 5 starts with descriptive body rows before numeric PK rows.
    xml = """<table><tr><th>Pharmacokinetic Parameters</th><th>Bedaquiline</th></tr>
    <tr><td colspan="2">SD=Standard Deviation</td></tr>
    <tr><td>Absorption</td><td></td></tr>
    <tr><td>Food effect</td><td>High fat meal increased C max and AUC by 2-fold.</td></tr>
    <tr><td>T max</td><td>Around 5 hours after single oral dose administration</td></tr>
    <tr><td>Percent bound to human plasma protein</td><td>greater than 99.9%.</td></tr>
    <tr><td>Apparent clearance of bedaquiline</td><td>2.62 L/h</td></tr>
    <tr><td>Apparent clearance of M2</td><td>4.95 L/h</td></tr></table>"""
    table, values, stage = extract(xml, tmp_path, "bedaquiline")
    assert table.columns == ["Pharmacokinetic Parameters", "Bedaquiline"]
    assert [(r["parameter"], r["value"]) for r in values] == [
        ("Tmax", 5), ("protein_binding", 99.9), ("clearance", 2.62)]
    assert stage.counts["metabolite_cells_isolated"] == 1
    assert all("2-fold" not in (r.get("condition") or "") for r in values)


def source_row(tmp_path, *, title, abstract, mesh="", other=""):
    xml = (f"<PubmedArticleSet>{other}<PubmedArticle><ArticleTitle>{title}</ArticleTitle>"
           f"<Abstract>{abstract}</Abstract><MeshHeading>{mesh}</MeshHeading></PubmedArticle></PubmedArticleSet>").encode()
    path = tmp_path / "source.xml"
    path.write_bytes(xml)
    return dict(evidence_id="test", source_id="pubmed", storage_path=str(path), checksum=sha256(xml).hexdigest(),
                source_text="Cefazolin body clearance was 0.21 L/h/kg after administration.",
                source_text_hash="unchanged", evidence_type="clinical_pk", compound_role="target_compound",
                compound="cefazolin", parameter="clearance", value=.21, unit="L/h/kg")


@pytest.mark.parametrize("title,mesh,expected", [
    ("Cefazolin pharmacokinetics in cats under surgical conditions", "Animals", "cat"),
    ("Pharmacokinetics in cynomolgus monkeys", "Animals", "monkey"),
    ("Drug absorption study", "Animals Rabbits", "rabbit"),
    ("Pharmacokinetics in Beagle dogs", "Animals", "dog"),
])
def test_species_from_matched_title_abstract_mesh_enters_review(tmp_path, title, mesh, expected):
    quote = "Cefazolin body clearance was 0.21 L/h/kg after administration."
    row = source_row(tmp_path, title=title, abstract=quote, mesh=mesh)
    enriched = enrich_source_context(row, project_root=tmp_path)
    assert expected in enriched["population"]
    assert enriched["source_text"] == row["source_text"] and enriched["source_text_hash"] == "unchanged"
    assert "population" not in row
    result = evaluate_domain_gate(enriched)
    assert result["domain_gate_status"] == "needs_manual_review"
    assert result["domain_classification"] == "animal_pk_requires_species_review"


def test_species_does_not_leak_between_articles_in_same_download(tmp_path):
    quote = "Cefazolin body clearance was 0.21 L/h/kg after administration."
    other = "<PubmedArticle><ArticleTitle>Another study in cats</ArticleTitle><Abstract>Unrelated feline experiment.</Abstract></PubmedArticle>"
    row = source_row(tmp_path, title="Human pharmacokinetics", abstract=quote, mesh="Humans", other=other)
    enriched = enrich_source_context(row, project_root=tmp_path)
    assert not enriched["metadata_json"]["source_context"]["animal_study"]
    assert not enriched.get("population")


def test_uppercase_cat_clinical_abbreviation_is_not_an_animal(tmp_path):
    quote = "Cefazolin body clearance was 0.21 L/h/kg after administration."
    row = source_row(tmp_path, title="Human CAT imaging study", abstract=quote, mesh="Humans")
    row["condition"] = "CAT imaging"
    enriched = enrich_source_context(row, project_root=tmp_path)
    assert not enriched["metadata_json"]["source_context"]["animal_study"]
    assert evaluate_domain_gate(enriched)["domain_gate_status"] == "passed"


def test_ambiguous_article_context_is_not_used(tmp_path):
    quote = "Cefazolin body clearance was 0.21 L/h/kg after administration."
    other = f"<PubmedArticle><ArticleTitle>Another study in cats</ArticleTitle><Abstract>{quote}</Abstract></PubmedArticle>"
    row = source_row(tmp_path, title="Human pharmacokinetics", abstract=quote, mesh="Humans", other=other)
    enriched = enrich_source_context(row, project_root=tmp_path)
    assert enriched["metadata_json"]["source_context"]["match_status"] == "ambiguous"
    assert not enriched.get("population")


def test_mixed_human_animal_article_does_not_assign_animal_population(tmp_path):
    quote = "Cefazolin body clearance was 0.21 L/h/kg after administration."
    row = source_row(tmp_path, title="Comparative pharmacokinetics in rats and human volunteers", abstract=quote)
    enriched = enrich_source_context(row, project_root=tmp_path)
    assert not enriched.get("population")
    assert enriched["metadata_json"]["source_context"]["species_scope"] == "mixed_or_comparative"
    assert evaluate_domain_gate(enriched)["domain_gate_status"] == "needs_manual_review"


def test_source_context_rejects_changed_source_and_outside_project(tmp_path):
    row = source_row(tmp_path, title="Study", abstract="Cefazolin body clearance was 0.21 L/h/kg after administration.")
    with pytest.raises(ValueError, match="checksum"):
        enrich_source_context({**row, "checksum": "0" * 64}, project_root=tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        enrich_source_context(row, project_root=tmp_path / "other")


def test_ibuprofen_is_pcs_clearance_is_not_parent_drug_clearance():
    row = dict(evidence_id="ibuprofen", evidence_type="clinical_pk", compound_role="target_compound",
               compound="ibuprofen", parameter="clearance", value=20.2,
               source_text="Administration of 800 mg ibuprofen enhanced intradialytic clearance of IS and pCS up to 20.2 ml/min and 14.9 ml/min.")
    assert analyte_mismatch_reason(row) == "clearance_value_belongs_to_other_named_analyte"
    assert evaluate_domain_gate(row)["domain_gate_status"] == "excluded"


@pytest.mark.parametrize("quote", [
    "In AUC and Cmax studies, clearance of ibuprofen was 20.2 mL/min.",
    "Ibuprofen clearance in the ICU was 20.2 mL/min and AUC was measured.",
    "After ibuprofen administration, clearance of the drug was 20.2 mL/min.",
])
def test_capitalization_or_generic_drug_object_does_not_create_analyte_mismatch(quote):
    assert analyte_mismatch_reason(dict(compound="ibuprofen", parameter="clearance", value=20.2,
                                       evidence_type="clinical_pk", source_text=quote)) is None
