"""Native-profile unit/definition contracts; no ODE execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints import source_pk_profile as module


@pytest.fixture
def assessment():
    parameter_values = [
        ("Organism|Weight", "kg", 70.), ("Organism|Hematocrit", "fraction", .4),
        ("Drug|Fraction unbound (plasma)", "fraction", .1),
        ("Organism|Tissue|Volume", "l", 10.), ("Organism|Tissue|Fraction vascular", "fraction", .1),
        ("Neighborhoods|Tissue|Drug|Partition coefficient (intracellular/plasma)", "fraction", 2.),
        ("Organism|VenousBlood|Volume", "l", 2.),
        ("Events|IV|Application_1|ProtocolSchemaItem|Dose", "mg", 10.),
        ("Events|IV|Application_1|ProtocolSchemaItem|Start time", "min", 0.),
        ("Events|IV|Application_1|ProtocolSchemaItem|Infusion time", "min", 5.),
    ]
    request = {"pkml_sha256": "p", "csv_sha256": "c", "target_output": "out", "request_sha256": "r",
               "parameters": [{"path": p, "unit": u} for p, u, v in parameter_values]}
    native = {**{k: request[k] for k in ("pkml_sha256", "csv_sha256", "target_output", "request_sha256")},
        "version": module.VERSION, "solver_calls": 0, "model_modified": False,
        "parameters": [{"path": p, "requested_unit": u, "status": "completed", "value": v,
                        "native_value": v, "native_unit": u, "dimension": "fixture"} for p, u, v in parameter_values],
        "nca": {"status": "completed", "parameters": [
            {"name": "CL", "values": [1.], "unit": "ml/min/kg"},
            {"name": "Vss", "values": [1000.], "unit": "ml/kg"},
            {"name": "FractionAucLastToInf", "values": [.05], "unit": ""}]}}
    context = {"compound": "Drug", "application_count": 1, "single_iv": True, "single_oral": False,
               "routes": ["Intravenous"], "interaction_count": 0, "curve_start_min": 0.,
               "application_start_times_min": [0.], "nonlinear_selected_processes": [],
               "dose_mg": None, "infusion_minutes": None, "weight_kg": None}
    plan = {"organs": [
        {"organ": "Tissue", "type": "TissueOrgansNotInGiTract", "volume_path": "Organism|Tissue|Volume",
         "fraction_vascular_path": "Organism|Tissue|Fraction vascular",
         "partition_path": "Neighborhoods|Tissue|Drug|Partition coefficient (intracellular/plasma)"},
        {"organ": "VenousBlood", "type": "VascularSystem", "volume_path": "Organism|VenousBlood|Volume"}],
        "excluded_organs": [], "interpretation": "pinned formula fixture", "calculator": "fixture"}
    return native, request, context, plan


def metrics_for(assessment):
    metrics, quality = module._assess(*assessment)
    return {m["definition"]: m for m in metrics}, quality


def test_distinct_physiochemical_and_iv_nca_values_and_unit_conversion(assessment):
    metrics, quality = metrics_for(assessment)
    assert metrics["fu_plasma"]["value"] == .1
    assert metrics["vdss_physchem"]["value"] == pytest.approx(21.8)
    assert metrics["vdss_physchem"]["eligible_for_constraint"] is False
    assert metrics["vdss_iv"]["value"] == 70.
    assert metrics["cl_systemic_iv"]["value"] == pytest.approx(4.2)
    assert metrics["cl_systemic_iv"]["eligible_for_constraint"] is True
    assert metrics["cl_systemic_iv"]["provenance"]["constraint_scope"] == "same_dose_same_context_iv_only"
    assert assessment[2]["dose_mg"] == 10.
    assert assessment[2]["infusion_minutes"] == 5.
    assert assessment[2]["weight_kg"] == 70.
    assert quality["auc_extrapolated_fraction"] == .05
    assert "does not bound" in quality["aumc_tail_error"]


def test_oral_reports_only_apparent_unqualified_endpoints(assessment):
    assessment[2].update(single_iv=False, single_oral=True, routes=["Oral"])
    metrics, _ = metrics_for(assessment)
    assert "vdss_iv" not in metrics and "cl_systemic_iv" not in metrics
    assert metrics["vdss_over_f"]["value"] == 70.
    assert not metrics["cl_over_f"]["eligible_for_constraint"]
    assert "bioavailability" in metrics["cl_over_f"]["reason"]


def test_repeated_dose_does_not_report_single_iv_pk(assessment):
    assessment[2].update(single_iv=False, single_oral=False, application_count=7)
    metrics, _ = metrics_for(assessment)
    assert metrics["cl_systemic_iv"]["value"] is None
    assert metrics["vdss_iv"]["value"] is None
    assert "Repeated" in metrics["vdss_iv"]["reason"]


@pytest.mark.parametrize("fraction", [None, -.1, .21, 1.1, float("nan")])
def test_tail_screen_does_not_silently_accept_unreliable_nca(assessment, fraction):
    assessment[0]["nca"]["parameters"][-1]["values"] = [fraction]
    metrics, _ = metrics_for(assessment)
    assert metrics["vdss_iv"]["value"] == 70.
    assert not metrics["vdss_iv"]["eligible_for_constraint"]
    assert "engineering screen" in metrics["vdss_iv"]["reason"]


def test_induction_scenario_nca_is_available_but_not_a_general_constraint(assessment):
    assessment[2]["interaction_count"] = 7
    metrics, _ = metrics_for(assessment)
    assert metrics["cl_systemic_iv"]["value"] > 0
    assert not metrics["cl_systemic_iv"]["eligible_for_constraint"]
    assert metrics["cl_systemic_iv"]["provenance"]["dose_independent_constant"] is False


def test_missing_organ_partition_is_explicit_incomplete_not_trusted_total(assessment):
    assessment[3]["organs"][0].update(partition_path=None, absent_partition_policy="zero_as_in_pinned_VSSCalculator")
    metrics, _ = metrics_for(assessment)
    result = metrics["vdss_physchem"]
    assert result["status"] == "incomplete"
    assert result["missing_organs"] == ["Tissue"]
    assert not result["eligible_for_constraint"]
    assert result["value"] == pytest.approx(1.8)


def test_broken_vss_readback_does_not_erase_valid_nca_weight(assessment):
    assessment[0]["parameters"][3]["status"] = "failed"
    metrics, _ = metrics_for(assessment)
    assert metrics["vdss_physchem"]["value"] is None
    assert metrics["vdss_iv"]["value"] == 70.


@pytest.mark.parametrize("unit", ["unknown", "1/h", "ml/min/kg_typo"])
def test_unknown_nca_units_never_guessed(assessment, unit):
    assessment[0]["nca"]["parameters"][0]["unit"] = unit
    metrics, _ = metrics_for(assessment)
    assert metrics["cl_systemic_iv"]["value"] is None
    assert not metrics["cl_systemic_iv"]["eligible_for_constraint"]


@pytest.mark.parametrize("key,value", [("solver_calls", 1), ("solver_calls", False),
    ("model_modified", True), ("pkml_sha256", "wrong"), ("csv_sha256", "wrong"), ("request_sha256", "wrong")])
def test_native_binding_and_no_solver_contract_are_enforced(assessment, key, value):
    assessment[0][key] = value
    with pytest.raises(ValueError, match="identity/read-only"):
        metrics_for(assessment)


def test_missing_administration_readback_disables_nca_constraints(assessment):
    assessment[0]["parameters"][-3]["status"] = "failed"
    metrics, _ = metrics_for(assessment)
    assert not metrics["cl_systemic_iv"]["eligible_for_constraint"]
    assert assessment[2]["dose_mg"] is None


def test_source_plan_uses_native_neighbor_ids_and_flags_zoned_liver_mapping():
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments/blueprint_design/source_context_pilot_20260921/midazolam_iv_source/source/native/outputs/snapshot-midazolam_iv_source.pkml"
    if not path.exists():
        pytest.skip("Optional existing native development artifact is not installed")
    plan = module._physchem_plan(path, "Midazolam", root)
    liver = next(row for row in plan["organs"] if row["organ"] == "Liver")
    assert liver["partition_path"] is None
    assert liver["absent_partition_policy"] == "zero_as_in_pinned_VSSCalculator"
    bone = next(row for row in plan["organs"] if row["organ"] == "Bone")
    assert bone["partition_path"].endswith("|Midazolam|Partition coefficient (intracellular/plasma)")
    assert any(p["path"].endswith("|ProtocolSchemaItem|Dose") and p["unit"] == "mg" for p in plan["parameters"])


def test_binding_rejects_changed_snapshot_before_output():
    root = Path(__file__).resolve().parents[2]
    directory = root / "experiments/blueprint_design/source_context_pilot_20260921/midazolam_iv_source/source/native"
    if not directory.exists():
        pytest.skip("Optional native development artifact not installed")
    snapshot = json.loads((directory / "snapshot.json").read_text())
    context = module._context(snapshot, {"time_min": [0., 10.], "concentration": [0., 1.]})
    assert context["species"] == "human"
    assert context["sex"] == "male"
    assert context["age_years"] == 30.
    assert context["population"] == "unknown"
    assert context["source_population"] == "European_ICRP_2002"
    assert context["nonlinear_selected_processes"]
    snapshot["Version"] = 79
    output = root / "runtime/deployment/never_created_profile_test"
    with pytest.raises(ValueError, match="Snapshot differs"):
        module.extract_source_pk_profile(snapshot, directory / "execution_manifest.json",
            "Organism|PeripheralVenousBlood|Midazolam|Plasma (Peripheral Venous Blood)", output, root)
    assert not output.exists()
