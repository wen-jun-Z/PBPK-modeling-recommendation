"""Pattern-search invariants and source isolation; no native engine is run."""
from copy import deepcopy
import json
import math

import pytest

from pbpk_kb.blueprints import curve_pattern_blueprint as pattern
from pbpk_kb.blueprints import development_curve_blueprint as legacy
from pbpk_kb.blueprints import evidence_blueprint as evidence
from pbpk_kb.blueprints.pk_anchored_blueprint import _drug_signature
from pbpk_kb.blueprints.source_workflow import sha256
from test_development_curve_blueprint import development_case, _native_success
from test_pk_anchored_blueprint import evidence_case


TIMES = [0., 1., 2., 3.]


def observed(values):
    return {"time_unit": "min", "unit": "umol/l", "observations": [
        {"time_min": t, "concentration_umol_per_l": v} for t, v in zip(TIMES, values)]}


def result(values, **extra):
    return {"status": "predicted", "clearance_l_h": None, "vss_l": None,
            "curve": {"time_min": TIMES.copy(), "concentration": list(values),
                      "time_unit": "min", "unit": "umol/l"}, **extra}


def coupled(lip, scale):
    return result([math.exp(-lip * (1 + math.log(scale) * t)) for t in TIMES])


def run(evaluate=coupled, **kwargs):
    settings = {"observations": observed(coupled(.625, math.sqrt(2.))["curve"]["concentration"]),
                "logp_prior": 0., "max_evaluations": 24, "target_clearance_l_h": 10.}
    settings.update(kwargs)
    return pattern.optimize_pattern_curve(evaluate, **settings)


def test_initial_zero_clearance_sensitivity_does_not_freeze_later_information():
    assert coupled(0., 1.)["curve"] == coupled(0., 1.1)["curve"]
    fitted = run()
    assert fitted["decision"]["clearance_scale"] != 1.
    assert fitted["decision"]["objective"] < fitted["initial_objective"] / 5
    assert fitted["sensitivity"]["frozen_parameters"] == []
    assert fitted["decision"]["clearance_l_h"] is None
    assert fitted["decision"]["derived_pk_diagnostics"]["clearance_status"] == "not_estimable"


def test_input_regularization_has_exact_fixed_formula_and_no_output_cl_dependency():
    def fixed(lip, scale):
        return result([1.] * 4, clearance_l_h=1000. * scale)
    fitted = run(fixed, observations=observed([math.e] * 4),
                 warm_start={"lipophilicity": .625, "clearance_scale": math.sqrt(2.)}, max_evaluations=2)
    warm = fitted["evaluations"][1]
    assert warm["objective_components"] == pytest.approx({
        "mean_squared_log_curve_error": 1., "clearance_input_prior": .025,
        "lipophilicity_prior": .000625})
    assert warm["objective"] == pytest.approx(1.025625)
    missing = run(lambda *_: result([1.] * 4), observations=observed([math.e] * 4), max_evaluations=1)
    assert fitted["evaluations"][0]["objective"] == missing["evaluations"][0]["objective"]


def test_failed_direction_is_recorded_and_opposite_direction_can_improve():
    calls = []
    def evaluate(lip, scale):
        calls.append((lip, scale))
        return {"status": "parameter_readback_failed", "reason": "capacity"} if lip < 0 else coupled(lip, scale)
    fitted = run(evaluate)
    assert fitted["failed_evaluation_count"] > 0
    assert fitted["improved_over_initial"]
    assert any(r["lipophilicity"] > 0 and r["status"] == "ok" for r in fitted["evaluations"])
    assert len(calls) == fitted["evaluation_count"] <= 24
    chosen = fitted["evaluations"][fitted["decision"]["evaluation"] - 1]
    assert chosen["status"] == "ok" and chosen["eligible_for_selection"]


def test_failed_initial_point_can_find_a_valid_neighbor_but_all_failures_cannot_win():
    fitted = run(lambda lip, scale: coupled(lip, scale) if lip > 0 else {"status": "native_failed"})
    assert fitted["evaluations"][0]["status"] == "evaluation_failed"
    assert fitted["eligible"] and fitted["decision"]["lipophilicity"] > 0
    failed = run(lambda *_: {"status": "parameter_readback_failed"})
    assert failed["decision"] is None and not failed["eligible"] and not failed["converged"]


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 6, 24])
def test_budget_includes_failures_and_no_duplicate_physical_calls(budget):
    calls = []
    def evaluate(lip, scale):
        calls.append((lip, scale))
        if scale < 1:
            raise RuntimeError("bounded native failure")
        return coupled(lip, scale)
    fitted = run(evaluate, max_evaluations=budget,
        warm_start={"lipophilicity": 0., "clearance_scale": 1.})
    assert len(calls) == len(set(calls)) == fitted["evaluation_count"] <= budget
    assert fitted["cache_hits"] >= 1
    assert fitted["warm_start_evaluation"] == 1
    assert all(-2.5 <= lip <= 2.5 and .25 <= scale <= 4. for lip, scale in calls)
    if fitted["status"] == "budget_exhausted":
        assert not fitted["converged"]


def test_partial_poll_does_not_shrink_or_claim_convergence():
    fitted = run(lambda *_: result([1.] * 4), max_evaluations=3)
    poll = fitted["optimizer"]["polls"][-1]
    assert not poll["complete"] and len(poll["probes"]) == 2
    assert fitted["optimizer"]["final_step"] == .25
    assert fitted["status"] == "budget_exhausted" and not fitted["converged"]


def test_complete_nonimproving_polls_shrink_and_only_numeric_convergence_is_claimed():
    fitted = run(lambda *_: result([1.] * 4), observations=observed([1.] * 4), max_evaluations=100)
    assert fitted["status"] == "poll_tolerance_reached"
    assert fitted["converged"] and fitted["optimizer"]["numerical_convergence_only"]
    steps = [p["step"] for p in fitted["optimizer"]["polls"]]
    assert all(b == a / 2 for a, b in zip(steps, steps[1:]))
    assert fitted["decision"]["evaluation"] == 1


@pytest.mark.parametrize("bad", ["native_status", "units", "coverage", "negative", "zero", "nan"])
def test_invalid_curve_or_failed_physical_status_is_never_eligible(bad):
    document = result([1.] * 4)
    if bad == "native_status": document["status"] = "parameter_readback_failed"
    elif bad == "units": document["curve"]["unit"] = "mg/l"
    elif bad == "coverage": document["curve"]["time_min"] = [0., .5, 1., 2.]
    elif bad == "negative": document["curve"]["concentration"][1] = -1.
    elif bad == "zero": document["curve"]["concentration"][1] = 0.
    else: document["curve"]["concentration"][1] = float("nan")
    fitted = run(lambda *_: deepcopy(document), max_evaluations=1)
    assert fitted["decision"] is None and fitted["failed_evaluation_count"] == 1


def test_optional_invalid_diagnostics_are_null_and_inputs_stay_immutable():
    document = result([1.] * 4, clearance_l_h=float("nan"), vss_l=True)
    observations = observed([1.] * 4)
    before = deepcopy(observations)
    fitted = run(lambda *_: document, observations=observations, max_evaluations=1)
    assert fitted["decision"]["clearance_l_h"] is None and fitted["decision"]["vss_l"] is None
    assert observations == before and math.isnan(document["clearance_l_h"])
    json.dumps(fitted, allow_nan=False)


def warm_source(case, **changes):
    root, request, _, _, source = case
    normalized = evidence.EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    logp = next(e["value"] for e in normalized["evidence"] if e["parameter"] == "logp")
    mapping = {"version": legacy.VERSION, "eligible": True, "compound_id": normalized["compound_id"],
        "drug_signature": _drug_signature(normalized), "construction_partition_method": normalized["partition_method"],
        "reference_human": normalized["human"], "reference_dosing": normalized["dosing"],
        "training_source": source, "target_vss_anchor": None, "target_vss_l": None,
        "decision": {"lipophilicity": logp + .625, "clearance_scale": 1., "objective": -1e6}}
    mapping.update(changes)
    path = root / "inputs/warm.json"
    path.write_text(json.dumps(mapping))
    return {"path": "inputs/warm.json", "sha256": sha256(path)}


def mocked_native(calls):
    def execute(snapshot, request, output, root, *, audit, require_vss):
        assert require_vss is False
        calls.append((snapshot, audit))
        result = _native_success(snapshot, request, output, root, audit=audit)
        result.update(clearance_l_h=None, vss_l=None)
        return result
    return execute


def fit(case, **kwargs):
    root, request, _, _, source = case
    return pattern.fit_pattern_mapping(reference_request=request, training_source=source,
        output_dir=root / "pattern_fit", project_root=root, **kwargs)


def test_wrapper_reevaluates_warm_point_and_mapping_is_readable_by_old_predictor(development_case, monkeypatch):
    root, request, _, _, source = development_case
    before = deepcopy(request)
    warm = warm_source(development_case)
    calls = []
    monkeypatch.setattr(pattern, "_run_development", mocked_native(calls))
    fitted = fit(development_case, max_evaluations=2, warm_start_source=warm)
    assert fitted["version"] == legacy.VERSION and fitted["algorithm_version"] == pattern.ALGORITHM_VERSION
    assert fitted["evaluation_count"] == len(calls) == 2
    assert fitted["evaluations"][1]["purpose"] == "warm_start_reassessment"
    assert fitted["decision"]["evaluation"] == 1  # old fabricated objective is ignored
    assert fitted["target_vss_anchor"] is None and fitted["target_vss_l"] is None
    assert fitted["development_curve_used"] and not fitted["comparison_curve_used"]
    assert request == before and source == fitted["training_source"]
    monkeypatch.setattr(legacy, "_run_development", mocked_native(calls))
    prediction = legacy.predict_with_development_mapping(request=request,
        mapping_path=root / "pattern_fit/mapping_locked.json", output_dir=root / "future_prediction", project_root=root)
    assert prediction["status"] == "predicted" and calls[-1][1] is True
    assert prediction["clearance_l_h"] is None


@pytest.mark.parametrize("field", ["eligible", "compound_id", "drug_signature", "construction_partition_method",
    "reference_human", "reference_dosing", "training_source", "target_vss_anchor", "target_vss_l", "decision"])
def test_mismatched_warm_start_is_rejected_before_native_or_output(development_case, monkeypatch, field):
    value = False if field == "eligible" else {"wrong": True}
    warm = warm_source(development_case, **{field: value})
    monkeypatch.setattr(pattern, "_run_development", lambda *_a, **_k: pytest.fail("native must not run"))
    with pytest.raises(ValueError, match="[Ww]arm start"):
        fit(development_case, warm_start_source=warm)
    assert not (development_case[0] / "pattern_fit").exists()


@pytest.mark.parametrize("mutation", ["hash", "forbidden_path", "forbidden_hash", "warm_bounds"])
def test_warm_source_integrity_exclusions_and_bounds(development_case, mutation):
    root, request, _, _, _ = development_case
    warm = warm_source(development_case)
    if mutation == "hash": warm["sha256"] = "0" * 64
    elif mutation == "forbidden_path": request["forbidden_sources"] = [{**warm, "sha256": "0" * 64}]
    elif mutation == "forbidden_hash": request["forbidden_sources"] = [{**warm, "path": "inputs/other.json"}]
    else: warm = warm_source(development_case, decision={"lipophilicity": 999., "clearance_scale": 1.})
    with pytest.raises(ValueError):
        fit(development_case, warm_start_source=warm)
    assert not (root / "pattern_fit").exists()


@pytest.mark.parametrize("mutation", ["hash", "forbidden_path", "forbidden_hash", "role"])
def test_development_source_leakage_guards_before_execution(development_case, mutation):
    root, request, _, _, source = development_case
    if mutation == "hash": source["sha256"] = "0" * 64
    elif mutation == "forbidden_path": request["forbidden_sources"] = [{**source, "sha256": "0" * 64}]
    elif mutation == "forbidden_hash": request["forbidden_sources"] = [{**source, "path": "inputs/comparison-copy.json"}]
    else: source["role"] = "comparison"
    with pytest.raises(ValueError):
        fit(development_case)
    assert not (root / "pattern_fit").exists()


@pytest.mark.parametrize("which", ["training", "warm"])
def test_source_mutation_during_execution_prevents_locked_mapping(development_case, monkeypatch, which):
    root, _, _, _, source = development_case
    warm = warm_source(development_case)
    def native(*args, **kwargs):
        result = mocked_native([])(*args, **kwargs)
        path = root / (source if which == "training" else warm)["path"]
        path.write_text(path.read_text() + " ")
        return result
    monkeypatch.setattr(pattern, "_run_development", native)
    with pytest.raises(ValueError):
        fit(development_case, max_evaluations=1, warm_start_source=warm)
    assert not (root / "pattern_fit/mapping_locked.json").exists()


def test_native_failures_preserved_and_existing_fit_cannot_be_overwritten(development_case, monkeypatch):
    root = development_case[0]
    monkeypatch.setattr(pattern, "_run_development", lambda *_a, **_k: {"status": "parameter_readback_failed"})
    fitted = fit(development_case, max_evaluations=2)
    assert not fitted["eligible"] and fitted["evaluation_count"] == 2
    assert json.loads((root / "pattern_fit/attempt_01/result.json").read_text())["status"] == "parameter_readback_failed"
    with pytest.raises(FileExistsError):
        fit(development_case)
