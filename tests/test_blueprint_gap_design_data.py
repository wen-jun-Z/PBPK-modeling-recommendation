"""Synthetic isolation and scorer-equivalence checks for gap acquisitions."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import blueprint_gap_design_data as data
import report_blueprint_validation as report
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS


@pytest.fixture
def development():
    req = {"compound_id": "synthetic",
           "human": {"weight_kg": 70., "age_years": 30., "sex": "MALE"},
           "dosing": {"route": "intravenous", "dose_mg": 10., "infusion_minutes": 60., "start_h": 0.},
           "evidence": [{"parameter": "molecular_weight", "value": 300., "unit": "g/mol"}]}
    full = {key: deepcopy(req[key]) for key in ("compound_id", "human", "dosing")}
    full.update(study_id="development", role="calibration_data_reconstruction", matrix="plasma",
                unit="umol/l", time_unit="min", molecular_weight_g_mol=300.,
                observations=[{"time_min": float(i), "concentration_umol_per_l": 10./(i+1)}
                              for i in range(10)])
    return full, req


@pytest.mark.parametrize("regime,selected,hidden", [
    ("early_four", [1, 2, 3, 4], [5, 6, 7, 8, 9]),
    ("late_four", [6, 7, 8, 9], [1, 2, 3, 4, 5]),
])
def test_gap_selection_uses_only_positive_times(development, regime, selected, hidden):
    full, req = development
    before = deepcopy(development)
    result = data.split_development(full, req, regime)
    assert [r["time_min"] for r in result["visible_training"]["observations"]] == selected
    assert result["design_only"] == {
        "observed_times_h": [i/60. for i in selected],
        "candidate_times_h": [i/60. for i in hidden], "candidate_ids": hidden}
    for index in hidden:
        full["observations"][index]["concentration_umol_per_l"] *= 1000.
    assert data.split_development(full, req, regime) == result
    assert req == before[1]


def test_reveal_only_selected_value_and_strip_hidden_metadata(development):
    full, req = development
    full["raw_observations"] = deepcopy(full["observations"])
    full["target"] = {"concentrations": [99., 98.]}
    full["observations"][0]["raw_curve"] = [99., 98.]
    split = data.split_development(full, req, "early_four")
    visible = split["visible_training"]
    assert not {"raw_observations", "target"} & visible.keys()
    out = data.reveal_one(full, visible, 5./60.)
    assert [r["time_min"] for r in out["observations"]] == [1., 2., 3., 4., 5.]
    for index in (6, 7, 8, 9):
        full["observations"][index]["concentration_umol_per_l"] *= 1234.
    assert data.reveal_one(full, visible, 5./60.) == out
    out["human"]["weight_kg"] = 1.
    assert visible["human"]["weight_kg"] == 70.


@pytest.mark.parametrize("regime", ["early_three", "late_three", "five", None, True])
def test_unknown_regime_rejected(development, regime):
    full, req = development
    with pytest.raises(ValueError):
        data.split_development(full, req, regime)


@pytest.mark.parametrize("time", [0., 1./60., 4./60., -1., 100., float("nan"), True])
def test_reveal_rejects_observed_or_unknown_or_zero_time(development, time):
    full, req = development
    visible = data.split_development(full, req, "early_four")["visible_training"]
    with pytest.raises(ValueError):
        data.reveal_one(full, visible, time)


@pytest.mark.parametrize("change", ["identity", "dose", "units", "weight", "concentration", "time", "too_many"])
def test_reveal_validates_visible_contract(development, change):
    full, req = development
    visible = data.split_development(full, req, "early_four")["visible_training"]
    if change == "identity": visible["study_id"] = "target"
    elif change == "dose": visible["dosing"]["dose_mg"] *= 2
    elif change == "units": visible["unit"] = "mg/l"
    elif change == "weight": visible["human"]["weight_kg"] *= 2
    elif change == "concentration": visible["observations"][0]["concentration_umol_per_l"] += .001
    elif change == "time": visible["observations"][0]["time_min"] += .001
    else: visible["observations"].append(deepcopy(full["observations"][5]))
    with pytest.raises(ValueError): data.reveal_one(full, visible, 6./60.)


def test_too_few_positive_points_and_unmatched_request_rejected(development):
    full, req = development
    short = deepcopy(full); short["observations"] = short["observations"][:5]
    with pytest.raises(ValueError, match="five positive"):
        data.split_development(short, req, "early_four")
    req["human"]["weight_kg"] += 1
    with pytest.raises(ValueError, match="request"):
        data.split_development(full, req, "early_four")


@pytest.fixture
def scoring(development):
    _, request = development
    policy = {"sampling_time_h": [0., .5, 1., 2., 3., 4., 6., 8.],
              "lloq_ng_ml": .001, "dose_ng": 10e6, "infusion_h": 1., "weight_kg": 70.,
              "integration": "linear_up_log_down"}
    case = {"case_id": "synthetic_target", "compound_id": "synthetic", "study_id": "target-study",
            "physchem": "synthetic-physchem", "nca_policy": policy}
    endpoints = {key: {"value": 2., "unit": unit} for key, unit in UNITS.items()}
    endpoints["cmax"]["unit"] = "mg/L"
    endpoints["auc_inf"]["unit"] = "mg.h/L"
    endpoints["terminal_half_life"]["unit"] = "min"
    clinical = {"case_id": case["case_id"], "study_id": case["study_id"],
                "approximations": ["representative_adult_vs_group_summary"],
                "clinical_reference": {"case_id": case["case_id"], "study_id": case["study_id"],
                                       "endpoints": endpoints, "curve": {"secret": [1234.]}}}
    time = [i/10. for i in range(81)]
    conc = [0. if t == 0. else 2.*(1.-math.exp(-t)) if t <= 1.
            else 2.*(1.-math.exp(-1.))*math.exp(-.5*(t-1.)) for t in time]
    native = {"time_min": [t*60. for t in time], "concentration": conc, "unit": "umol/l", "time_unit": "min"}
    return case, clinical, request, native


def test_target_contract_contains_metadata_never_observations(scoring):
    case, clinical, req, native = scoring
    snapshot = deepcopy(scoring)
    contract = data.build_target_contract(case, clinical, req)
    for endpoint in clinical["clinical_reference"]["endpoints"].values():
        endpoint["value"] *= 99999
        endpoint["raw_subject_values"] = [789.] * 20
    clinical["clinical_reference"]["curve"] = {"concentration": [9876543.]}
    case["nca_policy"]["observed_auc"] = 34567.
    same = data.build_target_contract(case, {"cases": [clinical]}, req)
    assert same == contract
    encoded = json.dumps(contract)
    assert "9876543" not in encoded and "observed_auc" not in encoded and "raw_subject" not in encoded
    contract["human"]["weight_kg"] = 1.
    assert req == snapshot[2]


def test_builder_never_even_accesses_endpoint_value_or_curve(scoring):
    case, clinical, req, _ = scoring
    class UnitOnly(dict):
        def get(self, key, *args):
            if key != "unit": raise AssertionError("Observed endpoint field accessed: " + key)
            return super().get(key, *args)
        def __getitem__(self, key):
            if key != "unit": raise AssertionError("Observed endpoint field accessed: " + key)
            return super().__getitem__(key)
    for key, row in clinical["clinical_reference"]["endpoints"].items():
        clinical["clinical_reference"]["endpoints"][key] = UnitOnly(row)
    clinical["clinical_reference"]["curve"] = object()
    result = data.build_target_contract(case, clinical, req)
    assert result["endpoint_units"]["cmax"] == "mg/L"


@pytest.mark.parametrize("integration", ["linear_up_log_down", "linear_trapezoidal"])
@pytest.mark.parametrize("cutoff", ["lloq", "lod", "calibration"])
def test_model_goals_exactly_match_final_scorer(scoring, integration, cutoff):
    case, clinical, req, native = scoring
    policy = case["nca_policy"]
    policy.update(integration=integration, auc_window_end_h=4., c_last_time_h=8., mrt_infusion_correction="none")
    if cutoff != "lloq":
        policy["lloq_ng_ml"] = None
        if cutoff == "lod":
            policy.update(lod_ng_ml=.001, prediction_cutoff_proxy="reported_lod")
            clinical["approximations"].append("lod_used_as_prediction_cutoff")
        else:
            policy.update(calibration_lower_bound_ng_ml=.001, prediction_cutoff_proxy="cited_assay_calibration_lower_bound")
            clinical["approximations"].append("assay_calibration_lower_bound_as_cutoff")
    contract = data.build_target_contract(case, clinical, req, molecular_weight_g_mol=300.)
    actual = data.model_goals(contract, native, req)
    class Store:
        def ref(self, name): return native if name == "prediction" else req
        def read(self, name): return {"identity": {"molecular_weight_g_mol": 300.}}
    reference_curve = {"time_h": [.5, 1., 2., 4., 8.], "concentration": [1.]*5,
                       "unit": "ng/mL", "matrix": "plasma", "statistic": "arithmetic_mean"}
    expected, _, _ = report._score_prediction(case, clinical, reference_curve,
        clinical["clinical_reference"]["endpoints"], {"prediction": "prediction", "request": "request"}, Store())
    for goal in data.GOALS:
        assert actual["values"][goal] == expected["endpoints"][goal]["value"]
        assert actual["units"][goal] == expected["endpoints"][goal]["unit"]
    assert actual["weak_endpoints"] == expected["weak_endpoints"]
    assert actual["diagnostics"]["selected_tail_times_h"] == expected["nca_diagnostics"]["selected_tail_times_h"]


def test_no_terminal_fit_returns_missing_not_infinite_time_substitute(scoring):
    case, clinical, req, native = scoring
    case["nca_policy"]["sampling_time_h"] = [0., .5, 1., 2.]
    contract = data.build_target_contract(case, clinical, req)
    goals = data.model_goals(contract, native, req)
    assert goals["values"]["cmax"] > 0
    assert goals["values"]["auc_inf"] is None
    assert goals["values"]["terminal_half_life"] is None


@pytest.mark.parametrize("change", ["compound", "dose", "weight", "infusion", "mw", "case", "study", "unit", "proxy"])
def test_target_contract_rejects_changed_scientific_conditions(scoring, change):
    case, clinical, req, native = scoring
    if change == "compound": req["compound_id"] = "other"
    elif change == "dose": req["dosing"]["dose_mg"] *= 2
    elif change == "weight": req["human"]["weight_kg"] *= 2
    elif change == "infusion": req["dosing"]["infusion_minutes"] *= 2
    elif change == "mw": req["evidence"][0]["unit"] = "kg/mol"
    elif change == "case": clinical["clinical_reference"]["case_id"] = "other"
    elif change == "study": clinical["clinical_reference"]["study_id"] = "other"
    elif change == "unit": clinical["clinical_reference"]["endpoints"]["cmax"]["unit"] = "unknown"
    else: case["nca_policy"].update(lloq_ng_ml=None, lod_ng_ml=.001, prediction_cutoff_proxy="reported_lod")
    with pytest.raises(ValueError): data.build_target_contract(case, clinical, req)


@pytest.mark.parametrize("bad", ["contract_extra", "policy_extra", "request_change", "curve_units", "mismatched_arrays", "negative", "unknown_cutoff", "unit_extra"])
def test_model_goals_rejects_unsafe_or_invalid_inputs(scoring, bad):
    case, clinical, req, native = scoring
    contract = data.build_target_contract(case, clinical, req)
    if bad == "contract_extra": contract["observed_endpoints"] = [123.]
    elif bad == "policy_extra": contract["policy"]["observations"] = [123.]
    elif bad == "request_change": req["dosing"]["dose_mg"] *= 2
    elif bad == "curve_units": native["unit"] = "mg/l"
    elif bad == "mismatched_arrays": native["concentration"].pop()
    elif bad == "negative": native["concentration"][1] = -1.
    elif bad == "unknown_cutoff": contract["cutoff_basis"] = "unknown"
    else: contract["endpoint_units"]["observed_value"] = 123.
    with pytest.raises(ValueError): data.model_goals(contract, native, req)


def test_native_duplicate_time_coalescing_is_shared_with_scorer(scoring):
    case, clinical, req, native = scoring
    contract = data.build_target_contract(case, clinical, req)
    normal = data.model_goals(contract, native, req)
    native["time_min"].insert(2, native["time_min"][1])
    native["concentration"].insert(2, native["concentration"][1])
    duplicate = data.model_goals(contract, native, req)
    assert duplicate["values"] == normal["values"]
    assert len(duplicate["diagnostics"]["coalesced_float_time_duplicates"]) == 1
    native["concentration"][2] *= 2.
    with pytest.raises(ValueError, match="Conflicting"):
        data.model_goals(contract, native, req)
