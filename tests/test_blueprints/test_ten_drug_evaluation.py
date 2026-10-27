"""Report contracts: predictions are distinct from matched clinical evaluation."""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprints.source_workflow import sha256


@pytest.fixture
def runner(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate_ten_drug_endpoints.py"
    spec = importlib.util.spec_from_file_location("ten_drug_evaluation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    # These isolated numerical cases test CSV/report contracts; the production
    # README renderer requires the complete fixed ten-drug cohort.
    monkeypatch.setattr(module, "render_readme", lambda *args: None)
    return module


def setup_case(runner, *, observed=False, curve_eligible=True, references=None):
    root = runner.ROOT
    output = root / "evaluation"
    (output / "cases" / "example").mkdir(parents=True)
    curve = {"time_min": [i * .6 for i in range(4801)],
             "concentration": [10 * math.exp(-.5 * i * .01) for i in range(4801)],
             "time_unit": "min", "unit": "umol/l"}
    request = {
        "human": {"age_years": 30., "sex": "MALE", "weight_kg": 70., "height_cm": 175.},
        "partition_method": "PKSimStandard",
        "dosing": {"route": "intravenous", "dose_mg": 100., "start_h": 0., "infusion_minutes": 0.},
        "target": {"endpoints": runner.ENDPOINTS, "primary_endpoint": "plasma_concentration_time",
                   "window_start_h": 0., "window_end_h": 8.},
        "evidence": [{"parameter": "molecular_weight", "value": 200.}],
    }
    predictions = summarize_extended_prediction(curve, request)["endpoints"]
    curve_file = output / "cases/example/prediction.json"
    curve_file.write_text(json.dumps(curve))
    times = [30., 60., 120., 240., 360., 480.]
    observation_file = root / "observations.json"
    if observed:
        observation_file.write_text(json.dumps({"observations": [
            {"time_min": time, "concentration_umol_per_l": 20 * math.exp(-.5 * time / 60)} for time in times]}))
    case = {"case_id": "example", "drug": "example", "compound_id": "example-id",
            "compound_name": "Example", "role": "separate_study_development_comparison" if observed else "prediction_only",
            "study_id": "ExampleStudy" if observed else None,
            "scenario_kind": "clinical_study" if observed else "hypothetical_grid", "condition": "reference_male",
            "request": request, "observations_path": "observations.json" if observed else None,
            "observation_times_min": times if observed else [], "curve_evaluation_eligible": curve_eligible,
            "curve_evaluation_ineligible_reason": "independent curation mismatch",
            "endpoint_references": references or {}}
    row = {key: value for key, value in case.items() if key not in {"request", "observation_times_min"}}
    row.update(status="predicted", prediction_path=str(curve_file.relative_to(root)),
               prediction_sha256=sha256(curve_file), curve_csv="evaluation/cases/example/predicted_curve.csv",
               simulation_end_h=48., endpoint_values=predictions, attempts=[])
    return output, {"cases": [case], "pins": {}}, [row]


def report_rows(output):
    with (output / "endpoint_evaluation.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {row["endpoint"]: row for row in csv.DictReader(stream)}


def reference(value, unit, *, independent=True, **extra):
    return {"value": value, "unit": unit, "independent_of_input": independent, "scenario_matched": True,
            "source": {"path": "published_source.txt", "location": "Table 2", "sha256": "a" * 64},
            "definition": "independent published endpoint", **extra}


def test_prediction_only_cases_have_eleven_outputs_and_blank_observations_and_errors(runner):
    args = setup_case(runner)
    summary = runner.make_reports(*args)
    rows = report_rows(args[0])
    assert len(rows) == 11
    assert summary["clinical_scored_cases"] == 0
    for endpoint, row in rows.items():
        assert row["prediction_status"] == "completed"
        assert row["error"] == row["observed_value"] == row["evaluation_prediction"] == ""
        assert row["evaluation_status"] == "no_observation"
        assert row["predicted_curve_csv"] if endpoint == "plasma_concentration_time" else row["predicted_value"]


def test_observed_comparisons_convert_units_and_keep_dense_and_sparse_values_distinct(runner):
    args = setup_case(runner, observed=True)
    summary = runner.make_reports(*args)
    rows = report_rows(args[0])
    assert summary["clinical_scored_cases"] == 1
    for endpoint in ("auc", "cmax", "c_last"):
        row = rows[endpoint]
        assert float(row["error"]) == pytest.approx(2.)
        assert float(row["observed_value"]) == pytest.approx(2 * float(row["evaluation_prediction"]))
        assert float(row["evaluation_start_h"]) == .5
        assert float(row["evaluation_end_h"]) == 8.
    cmax = rows["cmax"]
    assert cmax["unit"] == "mg/l"
    assert float(cmax["predicted_value"]) == 2.
    assert float(cmax["evaluation_prediction"]) == pytest.approx(2 * math.exp(-.25))
    assert float(rows["auc"]["predicted_value"]) != pytest.approx(float(rows["auc"]["evaluation_prediction"]))
    assert float(rows["tmax"]["predicted_value"]) == 0.
    assert float(rows["tmax"]["evaluation_prediction"]) == .5
    assert float(rows["tmax"]["error"]) == 0.
    assert rows["tmax"]["error_metric"] == "absolute_error_h"
    assert rows["auc_inf"]["error"] == ""
    assert rows["auc_inf"]["predicted_value"] != ""


def test_excluded_clinical_curve_is_not_scored_but_predictions_remain(runner):
    args = setup_case(runner, observed=True, curve_eligible=False)
    summary = runner.make_reports(*args)
    assert summary["clinical_curve_eligible_cases"] == summary["clinical_scored_cases"] == 0
    rows = report_rows(args[0])
    for row in rows.values():
        assert row["error"] == row["observed_value"] == row["evaluation_prediction"] == ""
        assert row["evaluation_status"] == "curve_reference_excluded"
        assert row["prediction_status"] == "completed"


def test_independent_published_terminal_reference_can_score_without_eligible_curve(runner):
    args = setup_case(runner, observed=True, curve_eligible=False,
                      references={"clearance": reference(30., "l/h")})
    summary = runner.make_reports(*args)
    row = report_rows(args[0])["clearance"]
    assert summary["clinical_scored_cases"] == 0
    assert summary["endpoint_coverage"]["clearance"]["evaluated"] == 1
    assert row["evaluation_status"] == "scored_against_published_endpoint"
    assert float(row["error"]) == pytest.approx(1.2, rel=3e-6)
    assert row["reference_source"] == "published_source.txt#Table 2"


def test_nonindependent_published_reference_does_not_leave_residual_error(runner):
    args = setup_case(runner, observed=True,
                      references={"terminal_half_life": reference(2., "h", independent=False)})
    summary = runner.make_reports(*args)
    row = report_rows(args[0])["terminal_half_life"]
    # A rejected reference must not be reported as an evaluated endpoint. If
    # the runner chooses a separate qualified curve comparison instead, its
    # status and provenance must explicitly identify that comparison.
    if row["evaluation_status"] == "reference_used_as_input":
        assert row["error"] == ""
        assert summary["endpoint_coverage"]["terminal_half_life"]["evaluated"] == 0
    else:
        assert row["evaluation_status"] == "scored_at_observation_times"
        assert float(row["error"]) == pytest.approx(1.)


def test_published_reference_from_different_scenario_is_not_scored(runner):
    args = setup_case(runner, observed=True,
                      references={"terminal_half_life": reference(2., "h", scenario_matched=False)})
    summary = runner.make_reports(*args)
    row = report_rows(args[0])["terminal_half_life"]
    assert row["error"] == ""
    assert summary["endpoint_coverage"]["terminal_half_life"]["evaluated"] == 0


def test_finite_window_published_reference_cannot_override_sparse_comparison(runner):
    args = setup_case(runner, observed=True, references={"auc": reference(123., "mg*h/l")})
    runner.make_reports(*args)
    row = report_rows(args[0])["auc"]
    assert row["evaluation_status"] != "scored_against_published_endpoint"
    assert row["error"] == ""


def test_published_reference_unit_mismatch_fails_loudly(runner):
    args = setup_case(runner, observed=True, references={"clearance": reference(30., "ml/min")})
    with pytest.raises(ValueError, match="unit mismatch"):
        runner.make_reports(*args)


def test_locked_prediction_hash_mismatch_prevents_scoring(runner):
    args = setup_case(runner, observed=True)
    prediction = runner.ROOT / args[2][0]["prediction_path"]
    prediction.write_text(prediction.read_text() + " ")
    with pytest.raises(ValueError, match="Locked prediction changed"):
        runner.make_reports(*args)


def test_scoring_error_does_not_remove_numeric_predictions(runner):
    args = setup_case(runner, observed=True)
    obs = runner.ROOT / args[1]["cases"][0]["observations_path"]
    document = json.loads(obs.read_text())
    document["observations"][0]["concentration_umol_per_l"] = 0.
    obs.write_text(json.dumps(document))
    summary = runner.make_reports(*args)
    rows = report_rows(args[0])
    assert summary["status"] == "completed_with_failures"
    assert rows["auc"]["predicted_value"] != ""
    assert rows["auc"]["error"] == ""
    assert rows["auc"]["evaluation_status"] == "scoring_failed"


def test_unavailable_dense_terminal_endpoint_does_not_retain_sparse_error(runner):
    args = setup_case(runner, observed=True,
                      references={"terminal_half_life": reference(2., "h")})
    args[2][0]["endpoint_values"]["terminal_half_life"] = {
        "status": "not_estimable", "value": None, "reason": "model terminal quality gate failed"}
    summary = runner.make_reports(*args)
    row = report_rows(args[0])["terminal_half_life"]
    assert row["predicted_value"] == row["evaluation_prediction"] == row["error"] == ""
    assert row["evaluation_status"] == "prediction_endpoint_unavailable"
    assert summary["endpoint_coverage"]["terminal_half_life"]["evaluated"] == 0


def test_independent_published_endpoint_survives_unrelated_curve_scoring_failure(runner):
    args = setup_case(runner, observed=True, references={"clearance": reference(30., "l/h")})
    obs = runner.ROOT / args[1]["cases"][0]["observations_path"]
    document = json.loads(obs.read_text())
    document["observations"][0]["concentration_umol_per_l"] = 0.
    obs.write_text(json.dumps(document))
    summary = runner.make_reports(*args)
    row = report_rows(args[0])["clearance"]
    assert summary["clinical_scored_cases"] == 0
    assert row["evaluation_status"] == "scored_against_published_endpoint"
    assert float(row["error"]) == pytest.approx(1.2, rel=3e-6)
