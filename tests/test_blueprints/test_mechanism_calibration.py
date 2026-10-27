"""Scientific decision guards using controlled native outcomes, not ODE claims."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from pbpk_kb.blueprints import mechanism_calibration as mc


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    observations = [{"time_min": float(i), "concentration_umol_per_l": 1., "unit": "umol/l"}
                    for i in (1, 2, 3)]
    spec = {"parameter_id": "aadac_kcat", "path": "Drug-Enzyme|kcat", "unit": "1/min",
            "source_value": 10., "lower_bound": 5., "upper_bound": 20., "scaling": "log",
            "evidence_reference": "official source identification limits; narrower development window"}
    receipts, cases = {}, []
    for index, role in enumerate(("calibration", "selection")):
        pkml = tmp_path / f"model{index}.pkml"
        pkml.write_text(f"complete-native-model-{index}")
        case = {"case_id": f"c{index}", "study_group_id": f"study{index}", "role": role,
                "pkml_path": str(pkml), "target_output": "Organism|Drug|Plasma",
                "observations": deepcopy(observations)}
        if role == "selection":
            case["sampling_type"] = "concentration_time_profile"
        cases.append(case)
        receipt = tmp_path / f"readback{index}.json"
        receipt.write_text(json.dumps({"accepted": True, "pkml_sha256": mc._sha(pkml),
            "parameter_checks": [{"path": spec["path"], "unit": spec["unit"], "value": 10.,
                                  "passed": True, "native_value": 10., "expected_native_value": 10.}]}))
        receipts[case["case_id"]] = {"path": receipt.name, "sha256": mc._sha(receipt)}
    calls = {"fit": [], "batch": []}
    def fit(training, specs, output, root, **kwargs):
        calls["fit"].append(deepcopy(training))
        assert all(c["role"] == "calibration" for c in training)
        output.mkdir(parents=True)
        result = {"status": "completed", "selected_vector": [12.]}
        (output / "calibration_result.json").write_text(json.dumps(result))
        return result
    def batch(**kwargs):
        calls["batch"].append(kwargs)
        return {"status": "completed", "runs": [_run(0, [2., 2., 2.]), _run(1, [1., 1., 1.])]}
    monkeypatch.setattr(mc, "fit_bounded_source_parameters", fit)
    monkeypatch.setattr(mc, "run_parameter_batch", batch)
    return dict(calibration_cases=cases[:1], selection_cases=cases[1:], parameter_specs=[spec],
                output_dir=tmp_path / "result", project_root=tmp_path,
                source_readback_receipts=receipts,
                disclosed_context_assumptions=["Historical development; same study cannot cross roles"],
                device="cpu"), calls


def _run(index, values):
    return {"vector_index": index, "time_min": [1., 2., 3.], "concentration": values,
            "time_unit": "min", "unit": "umol/l"}


def test_improvement_promotes_once_and_locks_predictions(inputs):
    args, calls = inputs
    out = mc.fit_select_source_mechanism(**args)
    assert out["decision"]["candidate_promoted"] is True
    assert out["retained_values"] == [12.]
    assert len(calls["fit"]) == 1 and len(calls["batch"]) == 1
    assert calls["batch"][0]["values"] == [[10.], [12.]]
    assert out["metrics"]["candidate"]["log_mse"] == 0
    assert out["metrics"]["source"]["auc_abs_log_error"] == pytest.approx(math.log(2))
    assert mc._sha(args["output_dir"] / "selection_predictions_locked.json") == out["files"]["selection_predictions_sha256"]
    assert json.loads((args["output_dir"] / "mechanism_result.json").read_text())["retained_values"] == [12.]
    assert not out["parameters_installed"]


def test_curve_improvement_does_not_override_worse_exposure(inputs, monkeypatch):
    args, _ = inputs
    monkeypatch.setattr(mc, "run_parameter_batch", lambda **kw: {"status": "completed", "runs": [
        _run(0, [.5, 2., .5]), _run(1, [1., 1., 3.])]})
    out = mc.fit_select_source_mechanism(**args)
    assert out["metrics"]["candidate"]["log_mse"] < out["metrics"]["source"]["log_mse"]
    assert out["decision"]["candidate_promoted"] is False
    assert out["retained_values"] == [10.] and out["candidate_values"] == [12.]
    assert "selection_cmax_abs_log_error_worsened" in out["decision"]["reasons"]


def test_failed_native_selection_keeps_source_and_failure(inputs, monkeypatch):
    args, _ = inputs
    monkeypatch.setattr(mc, "run_parameter_batch", lambda **kw: {"status": "failed", "runs": [], "error": "native failure"})
    out = mc.fit_select_source_mechanism(**args)
    assert out["status"] == "failed" and out["retained_values"] == [10.]
    assert out["candidate_values"] == [12.]
    assert out["metrics"]["candidate"]["rows"][0]["error"] == "native failure"


def test_same_values_never_promoted(inputs, monkeypatch):
    args, _ = inputs
    monkeypatch.setattr(mc, "fit_bounded_source_parameters", lambda *a, **k: {"status": "completed", "selected_vector": [10.]})
    out = mc.fit_select_source_mechanism(**args)
    assert not out["decision"]["candidate_promoted"]


@pytest.mark.parametrize("kind", ["study_leakage", "trough", "missing_receipt", "bad_receipt_sha",
                                    "source_value", "wrong_role", "large_window", "missing_unit"])
def test_scientific_input_errors_fail_before_fit(inputs, kind):
    args, calls = inputs
    if kind == "study_leakage":
        args["selection_cases"][0]["study_group_id"] = "study0"
    elif kind == "trough":
        args["selection_cases"][0]["sampling_type"] = "longitudinal_predose_trough_sequence"
    elif kind == "missing_receipt":
        args["source_readback_receipts"].pop("c0")
    elif kind == "bad_receipt_sha":
        args["source_readback_receipts"]["c0"]["sha256"] = "0" * 64
    elif kind == "source_value":
        args["parameter_specs"][0].update(source_value=11., lower_bound=5.5, upper_bound=22.)
    elif kind == "wrong_role":
        args["selection_cases"][0]["role"] = "calibration"
    elif kind == "large_window":
        args["parameter_specs"][0]["upper_bound"] = 40.
    elif kind == "missing_unit":
        args["selection_cases"][0]["observations"][0].pop("unit")
    with pytest.raises(ValueError):
        mc.fit_select_source_mechanism(**args)
    assert not calls["fit"] and not args["output_dir"].exists()


def test_receipt_native_value_is_checked_even_if_passed_flag_true(inputs):
    args, _ = inputs
    path = args["project_root"] / args["source_readback_receipts"]["c0"]["path"]
    receipt = json.loads(path.read_text())
    receipt["parameter_checks"][0]["native_value"] = 8.
    path.write_text(json.dumps(receipt))
    args["source_readback_receipts"]["c0"]["sha256"] = mc._sha(path)
    with pytest.raises(ValueError, match="native value"):
        mc.fit_select_source_mechanism(**args)


def test_study_equal_aggregation_is_not_point_or_case_weighted():
    rows = [{"study_group_id": group, "valid": True, "log_mse": value,
             "auc_abs_log_error": value, "cmax_abs_log_error": value}
            for group, value in [("A", 1.), ("A", 3.), ("B", 6.)]]
    assert mc._aggregate(rows)["log_mse"] == 4.


def test_receipt_cannot_reference_another_model(inputs):
    args, _ = inputs
    args["selection_cases"][0]["pkml_path"] = args["calibration_cases"][0]["pkml_path"]
    with pytest.raises(ValueError, match="exact PKML"):
        mc.fit_select_source_mechanism(**args)
