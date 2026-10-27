"""Clinical-source curation tests, separate from expanded runner tests.

Restores the curation coverage after the runner used the former test filename.
No source artifacts, frozen inputs, or model execution directories are written.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expanded_cohort_curation_test", ROOT / "scripts/prepare_expanded_blueprint_validation.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


@pytest.fixture(scope="module")
def cohort():
    return json.loads((ROOT / c.OUTPUT).read_text())


def observation():
    return {
        "Name": "Example parent IV mean",
        "BaseGrid": {"Values": [0., 1., 2., 3., 4., 5.], "Unit": "h"},
        "Columns": [{"Name": "Display label must not determine units [mg/L]",
                     "Values": [0., 7., 5., 3., 2., 1.], "Unit": "ng/ml",
                     "Dimension": "Concentration (mass)",
                     "QuantityInfo": {"Path": "study|ObservedData|Peripheral Venous Blood|Plasma|Example|ArithmeticMean"},
                     "RelatedColumns": [{"Values": [999.], "Name": "SD not primary observations"}]}],
        "ExtendedProperties": [{"Name": "Species", "Value": "Human"},
                               {"Name": "Organ", "Value": "Peripheral Venous Blood"},
                               {"Name": "Molecule", "Value": "Example"},
                               {"Name": "Data type", "Value": "aggregated"},
                               {"Name": "N", "Value": 8.}],
    }


def individual():
    return {"Name": "Explicit adult metadata", "OriginData": {
        "Species": "Human", "Gender": "MALE", "Population": "European",
        "Age": {"Value": 25., "Unit": "year(s)"},
        "Weight": {"Value": 80., "Unit": "kg"},
        "Height": {"Value": 180., "Unit": "cm"}}}


def protocol():
    return {"Name": "Clinical IV infusion", "ApplicationType": "Intravenous",
            "DosingInterval": "Single", "Parameters": [
                {"Name": "InputDose", "Value": 2., "Unit": "mg/kg"},
                {"Name": "Infusion time", "Value": .5, "Unit": "h"},
                {"Name": "Start time", "Value": 0., "Unit": "h"}]}


def test_frozen_cohort_reproduces_without_rewriting(cohort):
    before = c.sha(ROOT / c.OUTPUT)
    assert c.curate(ROOT) == cohort
    assert c.sha(ROOT / c.OUTPUT) == before
    for relative, digest in cohort["source_pins"].items():
        assert c.sha(c.inside(ROOT, relative)) == digest


def test_final_eleven_drug_membership_and_all_twenty_screened(cohort):
    expected = {"Amikacin", "Clarithromycin", "Ethinylestradiol", "Mexiletine",
                "Montelukast", "Paroxetine", "Pitavastatin", "Probenecid",
                "Rifampicin", "Rosuvastatin", "Vancomycin"}
    assert {r["drug"] for r in cohort["cases"]} == expected
    assert len(cohort["screening"]) == 20
    assert cohort["counts"]["clinical_metadata_eligible"] == 11
    assert cohort["counts"]["pending_or_excluded"] == 9
    assert sum(r["observation_metadata"]["positive_point_count"] for r in cohort["cases"]) == 137
    assert not expected.intersection({n.title() for n in cohort["original_drugs_excluded"]})


def test_oxycodone_is_pending_for_source_uncertainty_not_performance(cohort):
    row = next(r for r in cohort["screening"] if r["drug"] == "Oxycodone")
    assert not row["clinical_metadata_eligible"]
    reason = " ".join(row["blockers"])
    assert "placebo/rifampin" in reason and "parent-dose" in reason
    assert row["primary_reference"] == "https://pubmed.ncbi.nlm.nih.gov/19417618/"
    assert row["case_id"] not in {r["case_id"] for r in cohort["cases"]}


def test_cohort_exports_no_clinical_values_or_fitted_inputs(cohort):
    forbidden = {"observations", "concentration_umol_per_l", "clinical_reference", "predictions",
                 "reference_values", "Compounds", "Simulations", "ParameterIdentifications"}
    def visit(x):
        if isinstance(x, dict):
            assert not (set(x) & forbidden)
            for value in x.values():
                visit(value)
        elif isinstance(x, list):
            for value in x:
                visit(value)
    visit(cohort)
    for row in cohort["cases"]:
        assert row["construction_sources"] == []
        assert row["nca_policy"]["reported_original_scalar_reference_count"] == 0
        assert row["reference_locator"]["source"]["sha256"]
    assert not cohort["target_values_exported"] and not cohort["model_fitted"]


def test_curation_does_not_access_native_compound_or_prediction_blocks(monkeypatch, cohort):
    original = c.json.loads
    guarded = []
    class ClinicalOnlySnapshot(dict):
        def __getitem__(self, key):
            if key in {"Compounds", "Simulations", "ParameterIdentifications", "SimulationComparisons"}:
                raise AssertionError("Curation accessed native model input/output block")
            return super().__getitem__(key)
    def decode(text, *args, **kwargs):
        value = original(text, *args, **kwargs)
        if isinstance(value, dict) and "ObservedData" in value and "Compounds" in value:
            guarded.append(True)
            return ClinicalOnlySnapshot(value)
        return value
    monkeypatch.setattr(c.json, "loads", decode)
    assert c.curate(ROOT) == cohort
    assert len(guarded) == 20


def test_metadata_unchanged_by_positive_concentration_magnitudes():
    first = observation()
    second = deepcopy(first)
    second["Columns"][0]["Values"] = [0., 100., .001, 42., .2, 4.]
    assert c.observation_metadata(first, expected_drug="Example") == c.observation_metadata(second, expected_drug="Example")


def test_at_least_five_positive_observations_required():
    ob = observation()
    ob["Columns"][0]["Values"][1] = 0.
    with pytest.raises(ValueError, match="five positive"):
        c.observation_metadata(ob, expected_drug="Example")


def test_invalid_time_value_and_pairing_are_rejected():
    mutations = [lambda x: x["BaseGrid"]["Values"].__setitem__(2, 1.),
                 lambda x: x["BaseGrid"]["Values"].__setitem__(0, -1.),
                 lambda x: x["Columns"][0]["Values"].__setitem__(1, float("nan")),
                 lambda x: x["Columns"][0]["Values"].__setitem__(1, -1.),
                 lambda x: x["Columns"][0]["Values"].pop()]
    for mutate in mutations:
        ob = observation()
        mutate(ob)
        with pytest.raises(ValueError, match="Finite nonnegative"):
            c.observation_metadata(ob, expected_drug="Example")


def test_time_and_concentration_units_use_fields_not_display_labels():
    ob = observation()
    ob["BaseGrid"] = {"Values": [0., 60., 120., 180., 240., 300.], "Unit": "min"}
    metadata = c.observation_metadata(ob, expected_drug="Example")
    assert metadata["sampling_time_h"] == [0., 1., 2., 3., 4., 5.]
    assert metadata["concentration_unit"] == "ng/ml"
    ob["Columns"][0]["Unit"] = "arbitrary fluorescence"
    with pytest.raises(ValueError, match="Unsupported explicit observation units"):
        c.observation_metadata(ob, expected_drug="Example")


def test_nonhuman_unbound_or_whole_blood_data_stays_ineligible():
    ob = observation()
    ob["ExtendedProperties"][0]["Value"] = "Dog"
    assert c.observation_metadata(ob, expected_drug="Example")["blockers"]
    for label in ("whole blood", "arterial plasma", "unbound"):
        ob = observation()
        ob["Columns"][0]["QuantityInfo"]["Path"] = "study|" + label + "|Example|Mean"
        assert c.observation_metadata(ob, expected_drug="Example")["blockers"]


def test_parent_identity_and_ambiguous_concentration_columns_are_checked():
    ob = observation()
    assert c.observation_metadata(ob, expected_drug="AnotherParent")["blockers"]
    ob["Columns"].append(deepcopy(ob["Columns"][0]))
    with pytest.raises(ValueError, match="Ambiguous primary concentration column"):
        c.observation_metadata(ob, expected_drug="Example")


def test_missing_human_defaults_are_explicit_and_pediatric_data_rejected():
    person = individual()
    del person["OriginData"]["Weight"]
    human, audit = c.human_metadata(person)
    assert human["weight_kg"] == 70.
    assert audit["assumed_fields"] == ["weight_kg"]
    assert audit["fields"]["weight_kg"]["basis"] == "predeclared_engineering_assumption"
    assert not audit["clinical_demographics_verified"]
    person["OriginData"]["Age"]["Value"] = 17.
    with pytest.raises(ValueError, match="Adult clinical cohort"):
        c.human_metadata(person)


def test_single_iv_only_and_explicit_infusion_duration():
    for field, val in (("ApplicationType", "Oral"), ("DosingInterval", "Multiple"), ("Schemas", [{"Name": "repeat"}])):
        prot = protocol()
        prot[field] = val
        with pytest.raises(ValueError, match="single-dose IV"):
            c.dosing_metadata(prot, {"weight_kg": 80.})
    prot = protocol()
    prot["Parameters"] = [p for p in prot["Parameters"] if p["Name"] != "Infusion time"]
    with pytest.raises(ValueError, match="Infusion duration"):
        c.dosing_metadata(prot, {"weight_kg": 80.})


def test_mg_per_kg_and_hour_infusion_are_converted_with_explicit_basis():
    dosing, audit = c.dosing_metadata(protocol(), {"weight_kg": 80.})
    assert dosing == {"route": "intravenous", "dose_mg": 160., "start_h": 0., "infusion_minutes": 30.}
    assert audit["source_dose_unit"] == "mg/kg"
    assert audit["salt_to_parent_conversion_applied"] is False
    prot = protocol()
    prot["ApplicationType"] = "IntravenousBolus"
    prot["Parameters"][0]["Unit"] = "mg"
    dosing, _ = c.dosing_metadata(prot, {"weight_kg": 80.})
    assert dosing["dose_mg"] == 2. and dosing["infusion_minutes"] == 0.


def test_project_boundary_and_ambiguous_source_names_are_rejected():
    with pytest.raises(ValueError, match="inside this project"):
        c.inside(ROOT, "../outside.json")
    with pytest.raises(ValueError, match="Exactly one source"):
        c.named([{"Name": "duplicate"}, {"Name": "duplicate"}], "duplicate")
