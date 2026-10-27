"""Numerical mocked complete-source calibration; no ODE or R process."""
from copy import deepcopy
import json
import math
import threading

import pytest

from pbpk_kb.blueprints import source_calibration as module


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    model = root / "complete.pkml"
    model.write_text("<CompleteUnmodifiedSource/>")
    options = {
        "training_cases": [{"case_id": "case1", "study_group_id": "study1", "role": "calibration",
            "pkml_path": model, "target_output": "Organism|Plasma|Drug|Concentration",
            "observations": [{"time_min": 1., "concentration_umol_per_l": 2.},
                             {"time_min": 2., "concentration_umol_per_l": 2.}]}],
        "parameter_specs": [{"parameter_id": "rate", "path": "Drug-Enzyme|kcat", "unit": "1/min",
            "source_value": 1., "lower_bound": .5, "upper_bound": 3.5,
            "evidence_reference": "synthetic registered source and bounds", "scaling": "linear"}],
        "output_dir": root / "fit", "project_root": root, "device": "cpu"}
    state = {"calls": [], "prediction": lambda vector, case_index: vector[0], "fail": False,
             "invalid": None, "reverse": False}
    lock = threading.Lock()

    def mock_batch(**kwargs):
        with lock:
            state["calls"].append(deepcopy(kwargs))
        assert kwargs["cpu_threads"] == 2
        assert len(kwargs["values"]) <= 64
        if state["fail"]:
            return {"status": "failed", "error": "synthetic_native_failure", "runs": []}
        rows = []
        case_index = int(kwargs["output_dir"].name.split("_")[1])
        for index, values in enumerate(kwargs["values"]):
            value = state["prediction"](values, case_index)
            if state["invalid"] and state["invalid"](values):
                value = -1.
            rows.append({"vector_index": index, "values": values,
                "time_min": [0., 1., 2., 3.], "concentration": [value] * 4,
                "time_unit": "min", "unit": "umol/l"})
        if state["reverse"]:
            rows.reverse()
        return {"status": "completed", "runs": rows}

    monkeypatch.setattr(module, "run_parameter_batch", mock_batch)
    return options, state


def test_improves_training_and_preserves_complete_source_with_auditable_budget(calibration):
    options, state = calibration
    before = options["training_cases"][0]["pkml_path"].read_bytes()
    result = module.fit_bounded_source_parameters(**options)
    assert result["status"] == "completed"
    assert result["improved_training_objective"]
    assert result["selected_values"]["rate"] == pytest.approx(2.)
    assert result["training_score"]["log_mse"] < result["baseline_score"]["log_mse"]
    vectors = [tuple(v) for call in state["calls"] for v in call["values"]]
    assert len(vectors) == len(set(vectors)) <= 15
    assert vectors.count((1.,)) == 1
    assert result["counts"]["unique_vectors"] == len(vectors)
    assert result["counts"]["native_vector_attempts"] == len(vectors)
    assert result["counts"]["native_batches"] == 2
    assert options["training_cases"][0]["pkml_path"].read_bytes() == before
    assert result["provenance"]["structure_unchanged"] is True
    assert result["provenance"]["fitted_pkml_exported"] is False
    assert result["sensitivity_diagnostics"][0]["status"] == "computed"
    assert result["sensitivity_diagnostics"][0]["rms_log_prediction_per_normalized_coordinate"] > 0
    persisted = json.loads((options["output_dir"] / "calibration_result.json").read_text())
    assert persisted == result


def test_three_parameters_never_combine_unscored_coordinate_optima(calibration):
    options, state = calibration
    for index in [2, 3]:
        options["parameter_specs"].append({**options["parameter_specs"][0],
            "parameter_id": f"p{index}", "path": f"Drug-Enzyme{index}|kcat"})
    state["prediction"] = lambda vector, case_index: sum(vector) / 3
    result = module.fit_bounded_source_parameters(**options)
    vectors = [v for call in state["calls"] for v in call["values"]]
    assert result["selected_vector"] in vectors
    assert len(vectors) <= 43
    assert result["counts"]["vector_limit_per_case"] == 43
    assert all(all(.5 <= value <= 3.5 for value in row) for row in vectors)
    # A round's candidates vary one coordinate from its actual evaluated center.
    for row in result["trace"]:
        center = result["rounds"][row["round"] - 1]["center"]
        assert sum(a != b for a, b in zip(row["values"], center, strict=True)) <= 1


def test_studies_equal_weight_even_with_duplicate_cases_in_one_study(calibration):
    options, state = calibration
    original = options["training_cases"][0]
    options["training_cases"] = [deepcopy(original) for _ in range(3)]
    for index, case in enumerate(options["training_cases"]):
        case["case_id"] = f"case{index}"
    options["training_cases"][2]["study_group_id"] = "study2"
    state["prediction"] = lambda vector, case_index: math.exp(case_index == 2)
    for case in options["training_cases"]:
        for observation in case["observations"]:
            observation["concentration_umol_per_l"] = 1.
    result = module.fit_bounded_source_parameters(**options)
    assert result["baseline_score"]["log_mse"] == pytest.approx(.5)
    assert result["baseline_score"]["per_study"] == pytest.approx({"study1": 0., "study2": 1.})
    assert result["selected_vector"] == [1.]
    assert result["counts"]["native_vector_attempts"] == result["counts"]["unique_vectors"] * 3


def test_log_grid_and_prior_have_declared_semantics(calibration):
    options, state = calibration
    options["parameter_specs"][0].update(lower_bound=.125, upper_bound=8., scaling="log")
    state["prediction"] = lambda vector, case_index: 2.
    result = module.fit_bounded_source_parameters(**options)
    first = state["calls"][0]["values"]
    assert [v[0] for v in first] == pytest.approx([1., .125, .25, .5, 2., 4., 8.])
    assert result["selected_vector"] == [1.]
    row = next(row for row in result["trace"] if row["values"] == [8.])
    assert row["prior_penalty"] == pytest.approx(.02 * math.log(8.) ** 2)
    assert not result["improved_training_objective"]


def test_invalid_candidate_is_recorded_and_cannot_win(calibration):
    options, state = calibration
    state["invalid"] = lambda vector: vector[0] == 2.
    result = module.fit_bounded_source_parameters(**options)
    assert result["status"] == "completed"
    assert result["selected_vector"] != [2.]
    assert result["counts"]["failed_vectors"] >= 1
    assert any(not row["valid"] and row["values"] == [2.] for row in result["trace"])


@pytest.mark.parametrize("whole_batch", [False, True])
def test_bad_source_baseline_blocks_selection_even_if_other_points_work(calibration, whole_batch):
    options, state = calibration
    state["fail"] = whole_batch
    state["invalid"] = lambda vector: vector == [1.]
    result = module.fit_bounded_source_parameters(**options)
    assert result["status"] == "failed"
    assert result["error"] == "SOURCE_BASELINE_INVALID"
    assert result["selected_values"] is None
    assert len(state["calls"]) == 1


def test_vector_mapping_is_not_changed_by_result_order(calibration):
    options, state = calibration
    state["reverse"] = True
    result = module.fit_bounded_source_parameters(**options)
    assert result["selected_values"]["rate"] == pytest.approx(2.)


@pytest.mark.parametrize("mutation", [
    lambda o: o.update(max_rounds=0), lambda o: o.update(max_rounds=3),
    lambda o: o.update(max_rounds=True), lambda o: o.update(parameter_specs=[]),
    lambda o: o["parameter_specs"].extend(deepcopy(o["parameter_specs"]) * 3),
    lambda o: o["parameter_specs"][0].update(lower_bound=2.),
    lambda o: o["parameter_specs"][0].update(upper_bound=float("inf")),
    lambda o: o["parameter_specs"][0].update(lower_bound=-1., scaling="log"),
    lambda o: o["parameter_specs"][0].update(evidence_reference=""),
    lambda o: o["parameter_specs"][0].update(path="Drug|*|CL"),
    lambda o: o["parameter_specs"][0].update(source_value=True),
    lambda o: o["training_cases"][0].update(role="selection"),
    lambda o: o["training_cases"][0].update(role="test"),
    lambda o: o["training_cases"][0].update(observations=[]),
    lambda o: o["training_cases"][0]["observations"][0].update(concentration_umol_per_l=0.),
    lambda o: o["training_cases"].append(deepcopy(o["training_cases"][0])),
    lambda o: o["training_cases"][0].update(holdout_observations=[]),
    lambda o: o.update(output_dir=o["project_root"].parent / "outside"),
])
def test_invalid_or_leaking_inputs_rejected_before_output_and_native_call(calibration, mutation):
    options, state = calibration
    mutation(options)
    with pytest.raises(ValueError):
        module.fit_bounded_source_parameters(**options)
    assert not options["output_dir"].exists()
    assert not state["calls"]


def test_existing_results_are_never_overwritten(calibration):
    options, state = calibration
    options["output_dir"].mkdir()
    with pytest.raises(FileExistsError):
        module.fit_bounded_source_parameters(**options)
    assert not state["calls"]


def test_all_case_failure_prevents_selecting_subset_success(calibration):
    options, state = calibration
    options["training_cases"].append({**deepcopy(options["training_cases"][0]), "case_id": "other"})
    state["prediction"] = lambda vector, case_index: -1. if case_index else vector[0]
    result = module.fit_bounded_source_parameters(**options)
    assert result["status"] == "failed"
    assert result["counts"]["valid_vectors"] == 0


def test_one_round_stops_at_its_declared_budget(calibration):
    options, state = calibration
    result = module.fit_bounded_source_parameters(**options, max_rounds=1)
    assert len(state["calls"]) == 1
    assert result["counts"]["unique_vectors"] <= 8
    assert result["counts"]["vector_limit_per_case"] == 8


def test_failed_later_round_keeps_previously_evaluated_best(calibration, monkeypatch):
    options, state = calibration
    original = module.run_parameter_batch

    def fail_second(**kwargs):
        if kwargs["output_dir"].parent.name == "round_02":
            return {"status": "failed", "runs": [], "error": "synthetic_timeout"}
        return original(**kwargs)

    monkeypatch.setattr(module, "run_parameter_batch", fail_second)
    result = module.fit_bounded_source_parameters(**options)
    assert result["status"] == "completed"
    assert result["selected_vector"] == pytest.approx([2.])
    assert result["counts"]["failed_vectors"] > 0
    assert result["batches"][-1]["error"] == "synthetic_timeout"


def test_source_mutation_invalidates_native_result(calibration, monkeypatch):
    options, _ = calibration
    original = module.run_parameter_batch

    def mutate_source(**kwargs):
        result = original(**kwargs)
        kwargs["pkml_path"].write_text("<UnexpectedMutation/>")
        return result

    monkeypatch.setattr(module, "run_parameter_batch", mutate_source)
    result = module.fit_bounded_source_parameters(**options)
    assert result["status"] == "failed"
    assert "Source PKML changed during" in result["batches"][0]["error"]
    assert result["selected_vector"] is None
