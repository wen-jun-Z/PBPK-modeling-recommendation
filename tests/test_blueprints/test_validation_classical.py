"""Small analytic and source-isolation checks; no PBPK/clinical experiments."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import blueprint_validation_classical as comparator


def test_analytic_infusion_matches_independent_matrix_exponential():
    result = comparator.numerical_self_check()
    assert result["status"] == "passed"
    assert result["matrix_exponential_check_points"] == 96
    assert result["maximum_relative_concentration_error"] < 3e-10
    assert result["analytic_auc_equals_dose_over_CL_maximum_relative_error"] < 3e-10


def test_prior_conversion_and_weight_do_not_depend_on_number_of_curve_points():
    request = {"compound_id": "synthetic", "evidence": [{
        "parameter": "total_clearance", "compound_id": "synthetic", "value": 2.,
        "unit": "ml/min/kg", "basis": "predicted", "source": {"path": "unused"}}]}
    prior = comparator._clearance_prior(request, 70)
    assert prior["value_L_h"] == pytest.approx(8.4)
    parameters = np.log([16.8, 10., 15., 30.])
    kwargs = dict(dose_mg=10., infusion_h=.5, start_h=0.)
    times = np.asarray([.1, .5, 1., 2., 6.])
    observed = comparator.concentration_mg_l(np.exp(parameters), times, **kwargs) / 1.5
    expected = np.log(1.5)**2 + .1
    for repeat in (1, 7):
        residual = comparator._residual_vector(parameters, np.tile(times, repeat),
                    np.tile(observed, repeat), kwargs, prior["value_L_h"])
        assert np.sum(residual**2) == pytest.approx(expected, abs=1e-14)
        assert residual[-1] == pytest.approx(np.sqrt(.1))
    split = deepcopy(request)
    split["evidence"] = [dict(request["evidence"][0], parameter="hepatic_clearance", value=1.5),
                         dict(request["evidence"][0], parameter="renal_clearance", value=.5),
                         {"parameter": "elimination_scope", "value": "hepatic_and_renal"}]
    assert comparator._clearance_prior(split, 70)["value_L_h"] == pytest.approx(8.4)
    split["evidence"][-1]["value"] = "hepatic_only"
    with pytest.raises(ValueError, match="scope"):
        comparator._clearance_prior(split, 70)
    request["evidence"][0]["unit"] = "L/h"
    with pytest.raises(ValueError, match="units"):
        comparator._clearance_prior(request, 70)


@pytest.fixture
def isolated_case(tmp_path):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use a --basetemp within this project's runtime directory")
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    physical = save("physical.json", {"identity": {"molecular_weight_g_mol": 400.}})
    evidence = save("evidence.json", {"synthetic": "independent prior"})
    human = {"age_years": 30., "weight_kg": 70., "height_cm": 175.,
             "sex": "MALE", "population": "European_ICRP_2002", "seed": 17}
    dosing = {"route": "intravenous", "dose_mg": 10., "start_h": 0., "infusion_minutes": 30.}
    times = np.asarray([.1, .5, 1., 2., 6.])
    concentrations = comparator.concentration_mg_l([8., 10., 15., 30.], times,
                                                   **comparator.dosing_kwargs(dosing))
    document = {"study_id": "synthetic_training", "compound_id": "synthetic",
        "time_unit": "min", "unit": "umol/l", "molecular_weight_g_mol": 400.,
        "human": human, "dosing": dosing, "observations": [
            {"time_min": float(t*60), "concentration_umol_per_l": float(c*1000/400)}
            for t, c in zip(times, concentrations)]}
    training = save("five_observations.json", document)
    request = {"compound_id": "synthetic", "human": human, "dosing": dict(dosing, formulation=None),
        "target": {"window_start_h": 0., "window_end_h": 8., "terminal_fit": None},
        "evidence": [{"parameter": "total_clearance", "compound_id": "synthetic", "value": 2.,
                      "unit": "ml/min/kg", "source": evidence, "basis": "predicted"}],
        "evidence_ledger": evidence, "forbidden_sources": []}
    case = {"case_id": "synthetic_case", "drug": "Synthetic", "compound_id": "synthetic",
        "study_id": "synthetic_target", "physchem": physical,
        "development": "must_not_read_full_development.json", "reference": "must_not_read_target.json",
        "target_request_metadata": {"human": dict(human, weight_kg=80.),
                                    "dosing": dict(dosing, dose_mg=20., infusion_minutes=60.), "window_end_h": 8.},
        "nca_policy": {"sampling_time_h": [0., 31/60, 31.000000000000004/60, 1., 2., 8.]}}
    return case, request, training, tmp_path


def test_fitting_uses_only_five_training_points_and_locks_converted_prediction(isolated_case, monkeypatch):
    case, request, training, folder = isolated_case
    seen = []
    original = comparator._initial_parameters
    def capture(times, observed, *args):
        seen.append((times.copy(), observed.copy()))
        return original(times, observed, *args)
    monkeypatch.setattr(comparator, "_initial_parameters", capture)
    result = comparator.fit_predict(case=case, request=request, training_source=training,
        output_dir=folder/"output", project_root=ROOT, with_cl_prior=False)
    assert len(seen) == 1 and len(seen[0][0]) == 5
    assert np.max(seen[0][0]) == 6
    assert result["status"] == "predicted" and result["converged"]
    fit = json.loads((ROOT/result["fit"]["path"]).read_text())
    prediction = json.loads((ROOT/result["prediction"]["path"]).read_text())
    assert len(fit["starts"]) == 16
    assert fit["clearance_prior"] is None
    assert fit["development_metrics"]["log_mse"] < 1e-16
    assert result["resources"]["native_execution_calls"] == 0
    assert result["resources"]["analytic_cpu_residual_calls"] > 16
    parameters = np.asarray([fit["target_parameters"][key] for key in comparator.PARAMETERS])
    hours = np.asarray(prediction["time_min"])/60
    assert (np.diff(hours) > 0).all() and hours[-1] == 1344
    assert len(hours) < 9000 and 1. in hours
    expected = comparator.concentration_mg_l(parameters, hours,
        **comparator.dosing_kwargs(case["target_request_metadata"]["dosing"]))
    np.testing.assert_allclose(prediction["concentration"], expected*1000/400, rtol=1e-14)
    assert parameters[0]/fit["development_parameters"]["CL_L_h"] == pytest.approx((80/70)**.75)
    assert parameters[1]/fit["development_parameters"]["Vc_L"] == pytest.approx(80/70)
    before = {p.name: p.read_bytes() for p in (folder/"output").iterdir()}
    replay = comparator.fit_predict(case=case, request=request, training_source=training,
        output_dir=folder/"output", project_root=ROOT, with_cl_prior=False)
    assert replay["reused"] and replay["this_invocation_residual_calls"] == 0
    assert {p.name: p.read_bytes() for p in (folder/"output").iterdir()} == before
    (ROOT/result["prediction"]["path"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Frozen source changed"):
        comparator.fit_predict(case=case, request=request, training_source=training,
            output_dir=folder/"output", project_root=ROOT, with_cl_prior=False)


def test_prior_fit_records_objective_and_rejects_changed_inputs(isolated_case):
    case, request, training, folder = isolated_case
    result = comparator.fit_predict(case=case, request=request, training_source=training,
        output_dir=folder/"prior", project_root=ROOT, with_cl_prior=True)
    fit = json.loads((ROOT/result["fit"]["path"]).read_text())
    prior = fit["clearance_prior"]["value_L_h"]
    cl = fit["development_parameters"]["CL_L_h"]
    expected = fit["development_metrics"]["log_mse"] + .1*(np.log(cl/prior)/np.log(2))**2
    assert fit["selected_objective"] == pytest.approx(expected, abs=1e-14)
    assert prior == pytest.approx(8.4)
    assert result["method"] == "two_compartment_cl_prior"
    with pytest.raises(ValueError, match="different"):
        comparator.fit_predict(case=case, request=request, training_source=training,
            output_dir=folder/"prior", project_root=ROOT, with_cl_prior=False)

