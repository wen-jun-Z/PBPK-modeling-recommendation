"""Construction-use decisions, without clinical outcomes or native processes."""
from copy import deepcopy
import json
import math

import pytest

from pbpk_kb.blueprint_policy.purpose_design import assess_design, plan_design
from test_evidence_decision import evidence, fallback, record, request


def inputs(endpoints=("auc",), predicted=("fraction_unbound", "total_clearance")):
    req = request()
    req["target"]["endpoints"] = list(endpoints)
    req["target"]["primary_endpoint"] = endpoints[0]
    req["evidence"] = [row for row in req["evidence"] if row["parameter"] not in predicted]
    backups = fallback()
    ranges = [{"parameter": row["parameter"], "lower": row["value"] / 2,
               "upper": row["value"] * 2, "unit": row["unit"], "basis": "stress_assumption",
               "source": {"path": "synthetic/range_policy.json", "sha256": "b" * 64,
                          "location": row["parameter"]}}
              for row in backups if row["parameter"] in predicted]
    limits = {name: {"metric": "absolute" if name == "tmax" else "fold",
                     "max_change": .5 if name == "tmax" else 2.} for name in endpoints}
    return req, backups, ranges, limits


def planned(endpoints=("auc",), predicted=("fraction_unbound", "total_clearance")):
    req, backups, ranges, limits = inputs(endpoints, predicted)
    return plan_design(req, backups, ranges, endpoint_limits=limits)


def predictions(plan):
    units = {"auc": "mg*h/l", "cmax": "mg/l", "tmax": "h", "c_last": "mg/l",
             "terminal_half_life": "h", "auc_inf": "mg*h/l", "clearance": "l/h",
             "vz": "l", "mrt": "h", "vss": "l"}
    return [{"scenario_id": row["scenario_id"], "status": "predicted",
             "endpoints": {name: {"value": 0. if name == "tmax" else 10., "unit": units[name]}
                           for name in plan["requested_endpoints"] if name != "plasma_concentration_time"},
             "curve": {"time_h": [0., 1., 2.], "concentration_mg_l": [0., 1., .5]}}
            for row in plan["scenarios"]]


def test_evidence_is_retained_full_factorial_deduplicated_and_inputs_unchanged():
    req, backups, ranges, limits = inputs()
    originals = deepcopy((req, backups, ranges, limits))
    plan = plan_design(req, backups, ranges, endpoint_limits=limits)
    assert (req, backups, ranges, limits) == originals
    assert plan["status"] == "ready" and len(plan["scenarios"]) == 9
    assert plan["scenarios"][0] == {"scenario_id": "baseline", "values": {"fraction_unbound": .2, "total_clearance": 6.}}
    assert len({tuple(sorted(row["values"].items())) for row in plan["scenarios"]}) == 9
    assert {row["values"]["fraction_unbound"] for row in plan["scenarios"]} == {.1, .2, .4}
    assert {row["values"]["total_clearance"] for row in plan["scenarios"]} == {3., 6., 12.}
    for row in backups:
        assert row in plan["selected_evidence"]
    assert plan["calibration"]["performed"] is False
    assert plan["calibration"]["active_parameters"] == ["clearance_scale", "partition_scale", "bidirectional_exchange_scale"]
    assert "lipophilicity" not in json.dumps(plan["calibration"])
    assert plan["clinical_validation"] is False
    assert "not confidence intervals" in plan["range_interpretation"]
    assert plan["source_verification"] == "required_by_caller_not_performed_here"
    ranges[0]["lower"] = .2
    fewer = plan_design(req, backups, ranges, endpoint_limits=limits)
    assert len(fewer["scenarios"]) == 6


@pytest.mark.parametrize("basis", ["measured", "derived"])
def test_measured_and_derived_inputs_cannot_be_ranged_or_replaced(basis):
    req, backups, ranges, limits = inputs(predicted=("total_clearance",))
    record(req, "fraction_unbound")["basis"] = basis
    plan = plan_design(req, backups, ranges, endpoint_limits=limits)
    assert record(plan["selected_request"], "fraction_unbound") == record(req, "fraction_unbound")
    assert all(row["values"]["fraction_unbound"] == .1 for row in plan["scenarios"])
    ranges.append(inputs()[2][0])
    with pytest.raises(ValueError, match="measured or derived"):
        plan_design(req, backups, ranges, endpoint_limits=limits)


@pytest.mark.parametrize("change", [
    {"lower": 0.}, {"lower": True}, {"upper": False}, {"lower": math.nan},
    {"upper": math.inf}, {"lower": .3}, {"upper": .1}, {"upper": 1.01},
    {"unit": "%"}, {"basis": "confidence_interval"}, {"parameter": "logp"},
    {"source": {"path": "source", "sha256": "no", "location": "L1"}},
])
def test_invalid_ranges_rejected(change):
    req, backups, ranges, limits = inputs()
    ranges[0].update(change)
    with pytest.raises(ValueError):
        plan_design(req, backups, ranges, endpoint_limits=limits)


def test_renal_is_fixed_and_ranges_cannot_reach_renal_floor():
    req, backups, ranges, limits = inputs()
    record(req, "elimination_scope")["value"] = "hepatic_and_renal"
    renal = evidence("renal_clearance", 3., "ml/min/kg", context={"species": "human", "matrix": "plasma", "route": "intravenous"})
    req["evidence"].append(renal)
    with pytest.raises(ValueError, match="fixed renal"):
        plan_design(req, backups, ranges, endpoint_limits=limits)
    ranges[1]["lower"] = 3.1
    plan = plan_design(req, backups, ranges, endpoint_limits=limits)
    assert renal in plan["selected_evidence"]
    assert plan["calibration"]["protect_net_renal"] is True
    assert plan["calibration"]["clearance_scale_scope"] == "nonrenal_only"


def test_conflict_and_missing_common_evidence_block_without_simulations():
    req, backups, ranges, limits = inputs()
    req["evidence"] = [row for row in req["evidence"] if row["parameter"] != "molecular_weight"]
    plan = plan_design(req, backups, ranges, endpoint_limits=limits)
    assert plan["status"] == "blocked" and not plan["scenarios"]
    assert assess_design(plan, [])["status"] == "needs_evidence"
    req, backups, ranges, limits = inputs(predicted=("total_clearance",))
    conflict = deepcopy(record(req, "fraction_unbound"))
    conflict["value"] = .3
    req["evidence"].append(conflict)
    plan = plan_design(req, backups, ranges, endpoint_limits=limits)
    assert "fraction_unbound:equal_priority_conflict" in plan["reasons"]
    assert not plan["scenarios"]


@pytest.mark.parametrize("limit", [{}, {"auc": {"metric": "fold", "max_change": .9}},
                                     {"auc": {"metric": "fold", "max_change": True}},
                                     {"auc": {"metric": "absolute", "max_change": math.inf}}])
def test_limits_cannot_be_guessed_or_invalid(limit):
    req, backups, ranges, _ = inputs()
    with pytest.raises(ValueError):
        plan_design(req, backups, ranges, endpoint_limits=limit)


def test_selected_endpoint_changes_action_and_ignored_outputs_do_not():
    plan = planned(("auc", "cmax"))
    rows = predictions(plan)
    rows[1]["endpoints"]["cmax"]["value"] = 50.
    assessment = assess_design(plan, rows)
    assert assessment["status"] == "needs_evidence"
    assert assessment["endpoint_assessments"]["auc"]["status"] == "within_declared_limits"
    assert assessment["endpoint_assessments"]["cmax"]["status"] == "sensitive"
    req, backups, ranges, _ = inputs(("auc", "cmax"))
    only_auc = plan_design(req, backups, ranges, endpoint_limits={"auc": {"metric": "fold", "max_change": 2.}}, purpose={"endpoints": ["auc"]})
    assert only_auc["selected_request"]["target"]["endpoints"] == ["auc"]
    assert assess_design(only_auc, rows)["status"] == "supported_under_assumptions"
    assert set(assess_design(only_auc, rows)["endpoint_assessments"]) == {"auc"}


def test_tmax_zero_baseline_uses_absolute_hours():
    plan = planned(("tmax",))
    rows = predictions(plan)
    rows[1]["endpoints"]["tmax"]["value"] = .4
    assert assess_design(plan, rows)["status"] == "supported_under_assumptions"
    rows[1]["endpoints"]["tmax"]["value"] = .6
    assert assess_design(plan, rows)["status"] == "needs_evidence"
    req, backups, ranges, _ = inputs(("tmax",))
    with pytest.raises(ValueError, match="absolute"):
        plan_design(req, backups, ranges, endpoint_limits={"tmax": {"metric": "fold", "max_change": 2.}})


@pytest.mark.parametrize("failure", ["failed", "missing", "endpoint_none", "bad_grid", "bad_unit", "nan_curve"])
def test_failed_or_uncomputable_scenario_prevents_stability(failure):
    plan = planned()
    rows = predictions(plan)
    if failure == "failed":
        rows[1] = {"scenario_id": rows[1]["scenario_id"], "status": "failed", "reason": "solver failure"}
    elif failure == "missing":
        rows.pop(1)
    elif failure == "endpoint_none":
        rows[1]["endpoints"]["auc"]["value"] = None
    elif failure == "bad_grid":
        rows[1]["curve"]["time_h"][-1] = 3.
    elif failure == "bad_unit":
        rows[1]["endpoints"]["auc"]["unit"] = "umol*h/l"
    else:
        rows[1]["curve"]["concentration_mg_l"][-1] = math.nan
    assessment = assess_design(plan, rows)
    assert assessment["status"] == "needs_evidence"
    endpoint = assessment["endpoint_assessments"]["auc"]
    assert endpoint["status"] == "unassessed"
    assert endpoint["scenario_range"] is None
    assert endpoint["unassessed_scenarios"]


def test_missing_ranges_and_fixed_values_do_not_imply_zero_uncertainty():
    req, backups, ranges, limits = inputs()
    partial = plan_design(req, backups, ranges[:1], endpoint_limits=limits)
    assert partial["missing_ranges"] == ["total_clearance"]
    assert assess_design(partial, predictions(partial))["status"] == "needs_evidence"
    fixed = planned(predicted=())
    result = assess_design(fixed, predictions(fixed))
    assert result["status"] == "not_assessed"
    endpoint = result["endpoint_assessments"]["auc"]
    assert endpoint["scenario_range"] is None and endpoint["max_change"] is None
    assert endpoint["status"] == "not_assessed"


def test_positive_underflow_is_not_hidden_by_floor_and_json_stays_finite():
    plan = planned(("plasma_concentration_time",))
    rows = predictions(plan)
    rows[0]["curve"]["concentration_mg_l"][-1] = 0.
    for row in rows[1:]:
        row["curve"]["concentration_mg_l"][-1] = 1e-320
    result = assess_design(plan, rows)
    endpoint = result["endpoint_assessments"]["plasma_concentration_time"]
    assert result["status"] == "needs_evidence"
    assert endpoint["unbounded_change"] is True and endpoint["max_change"] is None
    assert endpoint["observed_sensitivity"] is True
    assert result["curve_numeric_diagnostics"][rows[1]["scenario_id"]]["positive_below_floor_samples"] == 1
    json.dumps(result, allow_nan=False)
    rows[0]["curve"]["concentration_mg_l"][-1] = 1e-320
    rows[1]["curve"]["concentration_mg_l"][-1] = 1e-310
    result = assess_design(plan, rows)
    assert result["endpoint_assessments"]["plasma_concentration_time"]["max_change"] > 1e9
    assert result["status"] == "needs_evidence"


def test_unknown_duplicate_scenarios_rejected_and_unrequested_terminal_failure_ignored():
    plan = planned()
    rows = predictions(plan)
    rows[1]["endpoints"]["terminal_half_life"] = {"value": None, "unit": "h"}
    assert assess_design(plan, rows)["status"] == "supported_under_assumptions"
    with pytest.raises(ValueError, match="duplicate"):
        assess_design(plan, rows + rows[:1])
    rows[-1]["scenario_id"] = "not_in_plan"
    with pytest.raises(ValueError, match="Unknown"):
        assess_design(plan, rows)


def test_pure_policy_does_not_read_range_sources_or_mutate_predictions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The pure module must not read files")
    monkeypatch.setattr("builtins.open", forbidden)
    plan = planned()
    rows = predictions(plan)
    original = deepcopy((plan, rows))
    assessment = assess_design(plan, rows)
    assert (plan, rows) == original
    json.dumps(assessment, allow_nan=False)
