"""Direct CL invariants, data isolation, locking and paired missingness."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_direct_cl_comparison as direct
from test_blood_partition_calibration import physiology  # Reuse synthetic anatomy only.


SOURCE = {"path": "synthetic_prior.json", "sha256": "a" * 64}
PROTOCOL_REF = {"path": "synthetic_protocol.json", "sha256": "b" * 64}


@pytest.fixture
def request_data():
    context = {"species": "human", "matrix": "plasma", "route": "intravenous"}
    return {
        "compound_id": "synthetic", "partition_method": "PKSimStandard",
        "human": {"weight_kg": 50.0}, "target": {"window_end_h": 24.0},
        "dosing": {"dose_mg": 100.0, "route": "intravenous", "infusion_minutes": 30.0, "start_h": 0.0},
        "evidence_ledger": {"path": "synthetic_evidence.json", "sha256": "c" * 64},
        "evidence": [
            {"parameter": "molecular_weight", "value": 400.0, "unit": "g/mol"},
            {"parameter": "total_clearance", "value": 3.0, "unit": "ml/min/kg",
             "basis": "predicted", "prediction_method": {"name": "random_forest"}, "context": context},
            {"parameter": "renal_clearance", "value": 0.2, "unit": "ml/min/kg",
             "basis": "measured", "context": context},
        ],
    }


@pytest.fixture
def short_grid(monkeypatch):
    grid = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 4.0, 12.0, 24.0])
    monkeypatch.setattr(direct.previous.frozen.base.engine, "prediction_grid", lambda request: grid)
    return grid


def test_direct_cl_reaches_solver_without_fitting_or_observation_reads(monkeypatch, request_data, physiology, short_grid):
    request = request_data
    before_request, before_physiology = deepcopy(request), deepcopy(physiology)
    calls = []
    simulate = direct.previous.frozen.old.simulate

    def audited_simulate(method, ph, req, times, logs):
        calls.append(np.asarray(logs).tolist())
        return simulate(method, ph, req, times, logs)

    def forbidden(*args, **kwargs):
        pytest.fail("Direct prediction attempted fitting, repair or a source-file read")

    monkeypatch.setattr(direct.previous.frozen.old, "simulate", audited_simulate)
    monkeypatch.setattr(direct.previous.repair, "fit", forbidden)
    monkeypatch.setattr(direct.previous.repair, "feasibility", forbidden)
    monkeypatch.setattr(direct, "checked", forbidden)
    monkeypatch.setattr(direct, "read", forbidden)
    monkeypatch.setattr(direct.previous.frozen.gap, "checked", forbidden)
    low = direct.direct_predict(request, physiology, 2.0, "rf", SOURCE)
    high = direct.direct_predict(request, physiology, 4.0, "tabpfn", SOURCE)
    for value, result in ((2.0, low), (4.0, high)):
        assert result["effective_total_clearance_l_h"] == pytest.approx(value * 50.0 * 0.06)
        assert result["cl_input"]["renal_clearance_l_h"] == pytest.approx(0.6)
        assert result["log_parameters"] == [0.0] * 4
        assert result["mass_balance"]["maximum_relative_error"] < 1e-8
        assert not any(result[k] for k in ("calibration_performed", "repair_performed",
                                          "development_observations_used", "target_observations_used"))
    assert calls == [[0.0] * 4, [0.0] * 4]
    assert not np.allclose(low["prediction"]["concentration"], high["prediction"]["concentration"])
    assert request == before_request and physiology == before_physiology
    for old, new in zip(request["evidence"], high["request"]["evidence"]):
        if old["parameter"] != "total_clearance":
            assert old == new


def test_independent_cl_produces_identical_direct_curves_for_all_predictors(request_data, physiology, short_grid):
    request = request_data
    request["evidence"][1]["basis"] = "measured"
    outputs = [direct.direct_predict(request, physiology, value, method, SOURCE)
               for method, value in zip(direct.METHODS, [0.001, 15.0, 500.0])]
    for output in outputs:
        assert output["request"] == request
        assert output["changes"] == []
        assert output["effective_total_clearance_l_h"] == pytest.approx(9.0)
        assert output["prediction"] == outputs[0]["prediction"]


@pytest.mark.parametrize("formula", ["PKSimStandard", "RodgersRowland"])
def test_both_predeclared_formulas_keep_zero_parameters(request_data, physiology, short_grid, formula):
    request = request_data
    assert direct.FORMULAS == ("PKSimStandard", "RodgersRowland")
    request["partition_method"] = formula
    output = direct.direct_predict(request, physiology, 2.0, "rf", SOURCE)
    assert output["request"]["partition_method"] == formula
    assert output["log_parameters"] == [0.0] * 4


def test_infeasible_clearance_is_rejected_without_a_renal_floor(request_data, physiology, short_grid):
    request = request_data
    with pytest.raises(ValueError, match="positive nonrenal"):
        direct.direct_predict(request, physiology, 0.1, "tabpfn", SOURCE)


def lock_fixture(monkeypatch):
    monkeypatch.setattr(direct, "ref", lambda path: PROTOCOL_REF)
    protocol = {"jobs": {f"case_{i}": {"replaceable": i < 24} for i in range(30)},
                "priors": {m: SOURCE for m in direct.METHODS}}
    monkeypatch.setattr(direct.previous, "clearance_inputs", lambda request: {"total_clearance_l_h": 3.0})
    units = {}
    for key in direct.expected_keys(protocol):
        formula, method, cid = key.split("__", 2)
        units[key] = dict(status="predicted", locked_at=1.0, log_parameters=[0.0] * 4,
                          calibration_performed=False, repair_performed=False,
                          case_id=cid, method=method, formula=formula,
                          replaceable=protocol["jobs"][cid]["replaceable"],
                          prior=SOURCE, request={}, effective_total_clearance_l_h=3.0)
    return protocol, dict(protocol=PROTOCOL_REF, units=units, all_predictions_locked_at=2.0)


@pytest.mark.parametrize("defect", ["missing", "extra", "unlocked", "late", "calibrated", "repaired", "nan_timestamp", "identity", "cl_readback"])
def test_all_attempts_lock_before_outcomes(monkeypatch, defect):
    protocol, document = lock_fixture(monkeypatch)
    key = next(iter(document["units"]))
    if defect == "missing":
        del document["units"][key]
    elif defect == "extra":
        document["units"]["unexpected"] = document["units"][key]
    elif defect == "unlocked":
        del document["all_predictions_locked_at"]
    elif defect == "late":
        document["units"][key]["locked_at"] = 3.0
    elif defect == "calibrated":
        document["units"][key]["log_parameters"][0] = 0.1
    elif defect == "repaired":
        document["units"][key]["repair_performed"] = True
    elif defect == "nan_timestamp":
        document["all_predictions_locked_at"] = float("nan")
    elif defect == "identity":
        document["units"][key]["case_id"] = "unrelated"
    else:
        document["units"][key]["effective_total_clearance_l_h"] = 4.0
    with pytest.raises(ValueError):
        direct.require_prediction_locks(protocol, document)


def test_complete_lock_accepts_retained_failures(monkeypatch):
    protocol, document = lock_fixture(monkeypatch)
    document["units"][next(iter(document["units"]))]["status"] = "failed"
    assert len(document["units"]) == 180
    assert direct.require_prediction_locks(protocol, document)


def test_report_rejects_incomplete_predictions_before_scoring(monkeypatch):
    protocol, document = lock_fixture(monkeypatch)
    del document["units"][next(iter(document["units"]))]
    state = dict(status="predictions_locked", predictions={
        "path": str((direct.RUN / "predictions.json").relative_to(ROOT)), "sha256": "d" * 64})
    monkeypatch.setattr(direct, "read", lambda path: protocol if path.name == "protocol.json" else state)
    monkeypatch.setattr(direct, "checked", lambda reference: document)
    monkeypatch.setattr(direct.previous.frozen.data, "score_case",
                        lambda *args: pytest.fail("Opened clinical outcomes before all predictions locked"))
    with pytest.raises(ValueError):
        direct.report()


def test_modified_locked_prediction_hash_rejects_before_scoring(monkeypatch):
    protocol, _ = lock_fixture(monkeypatch)
    state = dict(status="predictions_locked", predictions={
        "path": str((direct.RUN / "predictions.json").relative_to(ROOT)), "sha256": "d" * 64})
    monkeypatch.setattr(direct, "read", lambda path: protocol if path.name == "protocol.json" else state)
    def corrupted(reference):
        assert reference == state["predictions"]
        raise ValueError("Changed prediction hash")
    monkeypatch.setattr(direct, "checked", corrupted)
    monkeypatch.setattr(direct.previous.frozen.data, "score_case",
                        lambda *args: pytest.fail("Opened outcomes with changed frozen predictions"))
    with pytest.raises(ValueError, match="Changed prediction hash"):
        direct.report()


def score(curve, *, half_life=None):
    endpoints = {ep: {"error": None} for ep in direct.ENDPOINTS if ep != "curve"}
    endpoints["terminal_half_life"]["error"] = half_life
    return {"curve_metrics": {"gmfe": curve}, "endpoints": endpoints}


def test_missing_endpoints_keep_enrollment_and_paired_intersection_explicit():
    left = [dict(case_id="a", cluster_id="study_a", status="predicted", score=score(2.0, half_life=3.0)),
            dict(case_id="b", cluster_id="study_b", status="failed", score=None)]
    right = [dict(case_id="a", cluster_id="study_a", status="predicted", score=score(4.0)),
             dict(case_id="b", cluster_id="study_b", status="predicted", score=score(1.0))]
    summary = direct.summarize(left, 2)
    assert summary["enrolled"] == 2 and summary["predicted"] == 1
    assert summary["metrics"]["curve"]["n_cases"] == 1
    assert summary["metrics"]["curve"]["unavailable"] == 1
    assert summary["metrics"]["auc_inf"]["n_cases"] == 0
    assert summary["metrics"]["auc_inf"]["estimate"] is None
    paired = direct.paired(left, right, ["a", "b"])
    assert paired["curve"]["estimate"] == pytest.approx(0.5)
    assert paired["curve"]["attempted_pairs"] == 2
    assert paired["curve"]["complete_pairs"] == 1
    assert paired["curve"]["incomplete_case_ids"] == ["b"]
    assert paired["terminal_half_life"]["complete_pairs"] == 0
    assert paired["terminal_half_life"]["estimate"] is None
