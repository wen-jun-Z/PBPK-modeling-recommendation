"""Portable Cloud calculations on the actual exported case inputs."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from cloud.blueprints import curve_csv, predict_case, saved_result


@pytest.fixture(scope="module")
def cases():
    path = Path(__file__).resolve().parents[2] / "cloud/data/blueprints.json"
    if not path.is_file():
        pytest.skip("The audited portable blueprint snapshot has not been exported yet")
    return json.loads(path.read_text())["cases"]


@pytest.fixture(scope="module")
def caffeine(cases):
    return next(row for row in cases if str(row.get("compound") or row.get("label")).lower() == "caffeine")


def test_real_cpu_prediction_dose_response_mass_balance_and_immutability(caffeine):
    before = deepcopy(caffeine)
    first = predict_case(caffeine)
    second = predict_case(caffeine, dose_mg=2 * caffeine["request"]["dosing"]["dose_mg"])
    np.testing.assert_allclose(second["curve"]["concentration"],
                               2 * np.array(first["curve"]["concentration"]), rtol=1e-8)
    assert first["mass_balance"]["maximum_relative_error"] < 1e-6
    assert second["scenario_changed"]
    assert not first["scenario_changed"]
    assert not first["parameters_fitted"] and not first["scientific_accepted"]
    assert first["request"]["human"] == caffeine["request"]["human"]
    assert second["parameters"] == caffeine["parameters"]
    assert caffeine == before


@pytest.mark.parametrize("inputs", [
    {"dose_mg": 0}, {"dose_mg": -1}, {"dose_mg": True}, {"dose_mg": float("nan")},
    {"infusion_minutes": -1}, {"window_end_h": 169}, {"window_end_h": 0},
    {"infusion_minutes": 60, "window_end_h": 1},
])
def test_reject_invalid_input_before_solving(caffeine, inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input reached the solver")
    monkeypatch.setattr("cloud.blueprints.simulate", forbidden)
    with pytest.raises(ValueError):
        predict_case(caffeine, **inputs)


def test_calibration_human_is_distinct_from_target_and_export_preserves_truth(caffeine):
    result = saved_result(caffeine)
    assert result["request"]["human"]["age_years"] == 26
    assert result["calibration"]["human"]["age_years"] == 21.8
    assert result["calibration"]["dosing"]["dose_mg"] == 350
    assert result["calibration"]["visible_points"] == 5
    assert result["curve"] == caffeine["saved_curve"]
    assert result["request"]["human"] != result["calibration"]["human"]
    assert "during export" in result["source_verification"]
    rows = list(csv.DictReader(io.StringIO(curve_csv(result).decode("utf-8-sig"))))
    assert len(rows) == len(result["curve"]["time_min"])
    mw = next(e["value"] for e in result["request"]["evidence"] if e["parameter"] == "molecular_weight")
    assert float(rows[-1]["concentration_ng_ml"]) == result["curve"]["concentration"][-1] * mw


def test_all_exported_cases_replay_original_curves(cases):
    assert len(cases) == 30
    assert len({row["case_id"] for row in cases}) == 30
    for case in cases:
        times = np.asarray(case["saved_curve"]["time_min"]) / 60
        covered = np.flatnonzero(times <= case["request"]["target"]["window_end_h"])
        indices = covered[np.unique(np.linspace(0, len(covered) - 1, min(len(covered), 1500), dtype=int))]
        # Replay actual frozen target-window samples. The stored terminal curve
        # can extend to 1344 h; the interactive Cloud horizon remains <=168 h.
        result = predict_case(case, times_h=times[indices])
        np.testing.assert_allclose(result["curve"]["concentration"],
                                   np.asarray(case["saved_curve"]["concentration"])[indices],
                                   rtol=2e-6, atol=1e-9, err_msg=case["case_id"])
        assert result["mass_balance"]["maximum_relative_error"] < 1e-6
        assert len(result["physiology"]["organs"]) == 15
        assert result["endpoint_summary"]["endpoints"]["auc"]["value_mg_h_per_l"] >= 0
        assert saved_result(case)["curve"] == case["saved_curve"]


def test_streamlit_fixed_human_form_solves_and_does_not_relabel_draft(caffeine):
    from streamlit.testing.v1 import AppTest

    source = """
import json
from pathlib import Path
from cloud.blueprints import render_blueprints
render_blueprints(json.loads(Path('cloud/data/blueprints.json').read_text()))
"""
    app = AppTest.from_string(source, default_timeout=30).run()
    app.selectbox(key="cloud_blueprint_case").set_value(caffeine["case_id"]).run()
    assert not app.exception
    assert [field.label for field in app.number_input] == [
        "Dose (mg)", "Infusion duration (min; 0 for bolus)", "Prediction horizon (h)"]
    assert not app.text_input
    base_dose = caffeine["request"]["dosing"]["dose_mg"]
    app.number_input[0].set_value(base_dose * 2)
    next(button for button in app.button if button.label == "Generate prediction").click().run()
    assert not app.exception
    result_key = f"cloud_blueprint_result_{caffeine['case_id']}"
    result = app.session_state[result_key]
    assert result["request"]["dosing"]["dose_mg"] == base_dose * 2
    assert result["mass_balance"]["maximum_relative_error"] < 1e-6
    assert result["request"]["human"] == caffeine["request"]["human"]
    app.number_input[0].set_value(base_dose * 3).run()
    # Unsaved form edits must not relabel or overwrite the calculated output.
    assert app.session_state[result_key]["request"]["dosing"]["dose_mg"] == base_dose * 2
    next(button for button in app.button if button.label == "Restore saved prediction").click().run()
    assert not app.exception
    assert result_key not in app.session_state
