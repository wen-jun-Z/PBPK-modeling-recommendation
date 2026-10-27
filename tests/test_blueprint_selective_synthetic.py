"""Known-mechanism generation and truth isolation using synthetic physiology."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

# Reuse the existing entirely synthetic fifteen-organ/native-provenance fixture.
from test_blood_partition_calibration import physiology, req


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selective_synthetic_test_module",
    ROOT / "scripts/blueprint_selective_synthetic.py")
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)
FULL = [.05, .1, .25, .5, 1., 2., 4., 8., 12.]
FIVE = [0, 3, 5, 7, 8]


@pytest.fixture
def development_request(req):
    result = deepcopy(req)
    result["target"] = {"window_start_h": 0., "window_end_h": 12.}
    result["partition_method"] = "PKSimStandard"
    return result


def build(physiology, request, *, scenario="correct_RBC", sigma=.05, seed=1001, cid="synthetic-case"):
    return synthetic.build_case(cid, physiology, request, FULL, FIVE, scenario, sigma, seed)


@pytest.mark.parametrize("scenario", synthetic.SCENARIOS)
def test_all_mechanisms_are_serializable_and_leave_original_evidence_untouched(physiology, development_request, scenario):
    previous = deepcopy((physiology, development_request))
    case = build(physiology, development_request, scenario=scenario)
    assert (physiology, development_request) == previous
    assert json.loads(json.dumps(case, allow_nan=False)) == case
    assert case["provenance"]["clinical_target_observations_read"] is False
    assert case["provenance"]["fitted_clinical_parameters_used"] is False
    assert case["provenance"]["generator_calls"] == 2
    assert case["transfer"]["clinical_observations"] is False
    assert case["protected_truth_context"]["evidence_ledger_changed"] is False
    assert case["provenance"]["training_mass_balance"]["maximum_relative_error"] < 1e-8
    assert case["provenance"]["transfer_mass_balance"]["maximum_relative_error"] < 1e-8


def test_five_points_are_exact_subset_of_the_same_noisy_full_curve(physiology, development_request):
    case = build(physiology, development_request)
    full, five = case["calibration"]["full"], case["calibration"]["five"]
    assert five["times_h"] == [full["times_h"][i] for i in FIVE]
    assert five["observations_mg_l"] == [full["observations_mg_l"][i] for i in FIVE]
    assert five["request"] == full["request"] == development_request


def test_truth4_has_rbc_semantics_even_for_wrong_exchange(physiology, development_request):
    case = build(physiology, development_request, scenario="wrong_exchange")
    assert case["truth4"] == [0., 0., 0., 0.]
    assert case["generator_log_parameters"] == [0., 0., 0., .8]
    assert case["generator_variant"] == "exchange_shape"
    assert case["protected_truth_context"]["RBC_correction_required"] is False
    assert case["protected_truth_context"]["candidate_family_contains_generator"] is False


def test_balanced_wrong_rbc_direction_and_consistent_true_bp(physiology, development_request):
    first = build(physiology, development_request, scenario="wrong_RBC", seed=1001)
    second = build(physiology, development_request, scenario="wrong_RBC", seed=1002)
    for case, factor in ((first, .2), (second, 5.)):
        assert case["truth4"] == pytest.approx([0., 0., 0., math.log(factor)])
        truth = case["protected_truth_context"]
        assert truth["rbc_kp_plasma"] == pytest.approx(physiology["rbc_kp_plasma"] * factor)
        assert truth["blood_to_plasma_ratio"] == pytest.approx(
            1-physiology["hematocrit"]+physiology["hematocrit"]*truth["rbc_kp_plasma"])
    assert first["transfer"]["infusion_factor"] == second["transfer"]["infusion_factor"]


def test_nuisance_truth_keeps_renal_fixed_and_changes_only_nonrenal_cl(physiology, development_request):
    case = build(physiology, development_request, scenario="nuisance_error")
    assert case["truth4"] == pytest.approx([math.log(1.5), math.log(.7), math.log(2.), 0.])
    assert case["protected_truth_context"]["renal_clearance_l_h"] == pytest.approx(2.)
    assert case["protected_truth_context"]["total_clearance_l_h"] == pytest.approx(11.)
    assert case["protected_truth_context"]["RBC_correction_required"] is False


def test_noise_is_reproducible_independent_by_seed_and_absent_from_transfer(physiology, development_request):
    first = build(physiology, development_request, seed=1001)
    repeated = build(physiology, development_request, seed=1001)
    other = build(physiology, development_request, seed=1002)
    assert first == repeated
    assert first["calibration"]["full"]["observations_mg_l"] != other["calibration"]["full"]["observations_mg_l"]
    assert first["transfer"] == other["transfer"]
    assert first["transfer"]["noise_added"] is False


def test_noise_is_log_gaussian_on_clean_median_not_arithmetic_mean(physiology, development_request):
    clean = build(physiology, development_request, sigma=0.)
    noisy = build(physiology, development_request, sigma=.15)
    expected = np.random.default_rng(np.random.SeedSequence([
        1001, synthetic.NOISE_STREAM, noisy['provenance']['compound_noise_stream']
    ])).normal(0., .15, len(FULL))
    actual = np.log(np.asarray(noisy["calibration"]["full"]["observations_mg_l"])
                    / np.asarray(clean["calibration"]["full"]["observations_mg_l"]))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_compounds_have_independent_noise_streams(physiology, development_request):
    first = build(physiology, development_request, cid='one')
    second = build(physiology, development_request, cid='two')
    assert first['provenance']['compound_noise_stream'] != second['provenance']['compound_noise_stream']
    assert first['calibration']['full']['observations_mg_l'] != second['calibration']['full']['observations_mg_l']


def test_transfer_only_changes_infusion_and_uses_case_hash_not_truth(physiology, development_request):
    case = build(physiology, development_request, scenario="nuisance_error")
    req = case["transfer"]["request"]
    factor = .5 if hashlib.sha256(b"synthetic-case").digest()[0] & 1 else 2.
    expected = deepcopy(development_request)
    expected["dosing"]["infusion_minutes"] *= factor
    assert req == expected
    assert case["transfer"]["infusion_factor"] == factor
    assert case["transfer"]["times_h"][-1] == 1344.


def test_calibration_adapter_whitelists_visible_fields_and_excludes_all_hidden_truth(physiology, development_request):
    case = build(physiology, development_request, scenario="wrong_RBC")
    original = synthetic.calibration_input(case, "five")
    case["truth4"] = [999.]*4
    case["transfer"] = {"secret": "must-not-escape"}
    case["protected_truth_context"] = {"secret": "must-not-escape"}
    case["scenario"] = "must-not-escape"
    assert synthetic.calibration_input(case, "five") == original
    request, training = original
    assert request == development_request
    assert set(training) == {"compound_id", "human", "dosing", "matrix", "time_unit", "unit", "observations"}
    assert len(training["observations"]) == 5
    for row, t, c in zip(training["observations"], case["calibration"]["five"]["times_h"],
                         case["calibration"]["five"]["observations_mg_l"]):
        assert row["time_min"] == pytest.approx(t*60)
        assert row["concentration_umol_per_l"] == pytest.approx(c*1000/400)
    request["human"]["weight_kg"] = -100
    assert case["calibration"]["five"]["request"]["human"]["weight_kg"] == 70.


@pytest.mark.parametrize("indices", [[0, 1, 2, 3], [0, 1, 2, 3, 3], [0, 1, 2, 3, 9],
                                      [3, 0, 1, 2, 4], [False, 1, 2, 3, 4]])
def test_invalid_five_design_rejected_before_generation(physiology, development_request, monkeypatch, indices):
    monkeypatch.setattr(synthetic, "_simulate", lambda *args: pytest.fail("Invalid design simulated"))
    with pytest.raises(ValueError, match="Five indices"):
        synthetic.build_case("case", physiology, development_request, FULL, indices, "correct_RBC", .05, 1001)


@pytest.mark.parametrize("sigma", [-1., float("inf"), float("nan"), True])
def test_invalid_noise_rejected_before_generation(physiology, development_request, monkeypatch, sigma):
    monkeypatch.setattr(synthetic, "_simulate", lambda *args: pytest.fail("Invalid noise simulated"))
    with pytest.raises(ValueError, match="noise"):
        build(physiology, development_request, sigma=sigma)


def test_wrong_rbc_generation_does_not_overwrite_measured_partition(physiology, development_request):
    development_request["evidence"].append({"parameter": "blood_to_plasma_ratio",
        "value": physiology["blood_to_plasma_ratio"], "compound_id": "mock", "unit": "1",
        "basis": "measured", "context": {"species": "human"},
        "source": {"path": "synthetic-source.json", "sha256": "d"*64}})
    with pytest.raises(ValueError, match="protected"):
        build(physiology, development_request, scenario="wrong_RBC")


def test_unknown_scenario_rejected_before_generation(physiology, development_request, monkeypatch):
    monkeypatch.setattr(synthetic, "_simulate", lambda *args: pytest.fail("Unknown mechanism simulated"))
    with pytest.raises(ValueError, match="known synthetic scenario"):
        build(physiology, development_request, scenario="target-guided-best")
