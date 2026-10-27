"""Small-pilot contracts: real static inputs, no solver or external answers."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from pbpk_kb.blueprints import experiment_runtime, native_design


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "design_pilot_protocol", ROOT / "scripts/evaluate_native_design_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


@pytest.fixture(scope="module")
def bundles():
    return {drug: pilot.load_bundle(path) for drug, path in pilot.INPUTS.items()}


@pytest.mark.parametrize("drug", ["alprazolam", "midazolam"])
def test_purpose_requests_fix_baseline_and_allow_joint_comparison(bundles, drug):
    case = bundles[drug]["case"]
    requests = {group[0]: pilot.template_request(case, group, protocol="purpose_v3")
                for group in pilot.protocol_groups("purpose_v3")}
    fixed = requests["source_fixed"]
    assert fixed["algorithm"] == "purpose_driven"
    assert fixed["partition_mode"] == "source"
    assert fixed["max_free_parameters"] == 0
    assert set(fixed["parameter_modes"].values()) == {"fixed"}
    comparison = requests["coordinate_control"]
    assert comparison == requests["residual_design"]
    assert comparison["max_free_parameters"] == 2
    assert comparison["partition_mode"] == "compare_registered"
    assert set(comparison["parameter_modes"]) == {
        spec["parameter_id"] for spec in case["parameter_search"]["parameters"]}
    assert set(comparison["parameter_modes"].values()) == {"auto"}
    assert comparison["objective"] == {"primary": "curve"}
    assert comparison["kinetics_mode"] == "retain_source"


def test_legacy_protocol_retains_one_parameter_contract(bundles):
    case = bundles["alprazolam"]["case"]
    for group in pilot.protocol_groups("legacy_v2"):
        request = pilot.template_request(case, group)
        assert request["parameter_modes"]["lipophilicity"] == "fixed"
        assert request["max_free_parameters"] == (0 if group[3] == "fixed" else 1)
        assert "algorithm" not in request
    with pytest.raises(ValueError, match="Unknown pilot protocol"):
        pilot.protocol_groups("unknown")


def compile_static_plans(bundles, protocol):
    """Compile actual source-bound plans while forbidding solver/answer access."""
    forbidden = {(ROOT / scenario["holdout_reference"]["path"]).resolve()
                 for bundle in bundles.values() for scenario in bundle["external_scenarios"]}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.resolve() not in forbidden, "Static preflight opened an external answer"
        return original_open(path, *args, **kwargs)

    def no_native(*args, **kwargs):
        pytest.fail("Static preflight must not run a native solver")

    all_plans = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        patch.setattr(native_design, "execute_snapshot", no_native)
        patch.setattr(experiment_runtime, "execute_snapshot", no_native)
        patch.setattr(experiment_runtime, "run_parameter_batch", no_native)
        for drug, bundle in bundles.items():
            case = bundle["case"]
            path = pilot.INPUTS[drug]
            provenance = {
                "provenance_id": str(uuid5(NAMESPACE_URL, str(path))),
                "provenance_kind": "user_request",
                "source_id": str(path.relative_to(ROOT)),
                "source_checksum": pilot.file_sha(path),
                "source_location": "Static test request from registered Dcal regimen; no observation fitting",
            }
            plans = {}
            for group in pilot.protocol_groups(protocol):
                request = pilot.template_request(case, group, protocol=protocol)
                policy = pilot.policy_from_template(case, request, request_source=provenance)
                plan = pilot.build_design_plan(case, policy, project_root=ROOT)
                assert plan["supported"], plan["reasons"]
                plan["comparison_parameter_specs"] = deepcopy(case["parameter_search"]["parameters"])
                plan["comparison_request"] = deepcopy(request)
                if protocol == "study_v4":
                    plan["comparison_policy"] = policy.model_dump(mode="json")
                    plan["comparison_request_source"] = deepcopy(provenance)
                    plan["comparison_fit_strategy"] = group[5]
                    plan["comparison_budget_strategy"] = group[4]
                pilot.validate_objective_data(case, policy.objective)
                plans[group[0]] = plan
            all_plans[drug] = plans
    return all_plans


@pytest.fixture(scope="module")
def static_plans(bundles):
    return compile_static_plans(bundles, "purpose_v3")


@pytest.fixture(scope="module")
def study_static_plans(bundles):
    return compile_static_plans(bundles, "study_v4")


@pytest.mark.parametrize("drug", ["alprazolam", "midazolam"])
def test_real_inputs_compile_matched_eight_candidates_without_solver(static_plans, drug):
    plans = static_plans[drug]
    contract = pilot.verify_matched_plans(plans)
    assert contract["matched"] is True
    assert contract["candidate_count"] == 8
    for name in ("coordinate_control", "residual_design"):
        plan = plans[name]
        assert plan["execution"] == {"solver_calls": 0, "llm_calls": 0}
        assert len([c for c in plan["candidates"] if len(c["free_parameter_ids"]) == 2]) == 2
        assert {c["action"] for c in plan["candidates"]} == {
            "partition_pksim_standard", "partition_rodgers_rowland"}
        assert plan["source_structure_audit"]["source_partition_action"] == "partition_rodgers_rowland"
    baseline = plans["source_fixed"]["candidates"]
    assert len(baseline) == 1
    assert baseline[0]["action"] == "partition_rodgers_rowland"
    assert baseline[0]["free_parameter_ids"] == []


def test_matching_rejects_missing_or_different_parameter_domains(static_plans):
    plans = deepcopy(static_plans["alprazolam"])
    for name in ("coordinate_control", "residual_design"):
        plans[name].pop("comparison_parameter_specs")
    with pytest.raises((ValueError, KeyError)):
        pilot.verify_matched_plans(plans)
    plans = deepcopy(static_plans["alprazolam"])
    plans["residual_design"]["comparison_parameter_specs"][0]["upper_bound"] *= 2
    with pytest.raises(ValueError, match="differ"):
        pilot.verify_matched_plans(plans)


@pytest.mark.parametrize("change", ["candidate", "objective", "count"])
def test_matching_rejects_non_optimizer_interventions(static_plans, change):
    plans = deepcopy(static_plans["alprazolam"])
    revised = plans["residual_design"]
    if change == "candidate":
        revised["candidates"][0]["free_parameter_ids"] = ["lipophilicity"]
    elif change == "objective":
        revised["objective"]["selection_tolerance"] = 0.1
    else:
        plans["coordinate_control"]["candidates"].pop()
        revised["candidates"].pop()
    with pytest.raises(ValueError):
        pilot.verify_matched_plans(plans)


@pytest.mark.parametrize("drug", ["alprazolam", "midazolam"])
def test_study_protocol_has_matched_requests_and_declared_interventions(bundles, drug):
    groups = pilot.protocol_groups("study_v4")
    assert len(groups) == 4
    requests = {group[0]: pilot.template_request(bundles[drug]["case"], group, protocol="study_v4")
                for group in groups}
    fixed = requests["source_fixed"]
    assert fixed["algorithm"] == "purpose_driven"
    assert fixed["partition_mode"] == "source"
    assert fixed["max_free_parameters"] == 0
    assert set(fixed["parameter_modes"].values()) == {"fixed"}
    control = requests["residual_control"]
    assert control == requests["screened_control"]
    revised = deepcopy(requests["study_consistent"])
    assert revised.pop("algorithm") == "study_consistent"
    old = deepcopy(control)
    assert old.pop("algorithm") == "purpose_driven"
    assert revised == old
    assert old["max_free_parameters"] == 2
    assert old["partition_mode"] == "compare_registered"
    assert set(old["parameter_modes"].values()) == {"auto"}
    assert old["objective"] == {"primary": "curve"}
    assert {group[0]: group[5] for group in groups} == {
        "source_fixed": None, "residual_control": None,
        "screened_control": "screened_residual", "study_consistent": None}


@pytest.mark.parametrize("drug", ["alprazolam", "midazolam"])
def test_study_real_preflight_matches_eight_candidates_without_native_or_answers(study_static_plans, drug):
    plans = study_static_plans[drug]
    contract = pilot.verify_study_matched_plans(plans)
    assert contract["matched"] is True
    assert contract["candidate_count"] == 8
    assert contract["groups"] == ["residual_control", "screened_control", "study_consistent"]
    assert len(contract["comparisons"]) == 3
    assert len(plans["source_fixed"]["candidates"]) == 1
    for name in contract["groups"]:
        assert plans[name]["execution"] == {"solver_calls": 0, "llm_calls": 0}
        assert len([c for c in plans[name]["candidates"] if len(c["free_parameter_ids"]) == 2]) == 2


@pytest.mark.parametrize("change", [
    "parameter_domain", "candidate", "objective", "request_dose", "request_extra", "policy_dose",
    "policy_parameter_modes", "policy_extra", "source_audit", "baseline", "policy_provenance",
    "algorithm", "fit_strategy", "budget_strategy", "missing_fit_strategy", "count",
])
def test_study_preflight_rejects_undeclared_policy_or_candidate_differences(study_static_plans, change):
    plans = deepcopy(study_static_plans["alprazolam"])
    revised = plans["study_consistent"]
    if change == "parameter_domain":
        revised["comparison_parameter_specs"][0]["upper_bound"] *= 2
    elif change == "candidate":
        revised["candidates"][0]["free_parameter_ids"] = ["lipophilicity"]
    elif change == "objective":
        revised["objective"]["selection_tolerance"] = 0.5
    elif change == "request_dose":
        revised["comparison_request"]["dose"] *= 2
    elif change == "request_extra":
        revised["comparison_request"]["unexpected"] = True
    elif change == "policy_dose":
        revised["comparison_policy"]["prediction_request"]["requirement"]["dose"] *= 2
    elif change == "policy_parameter_modes":
        revised["comparison_policy"]["parameter_modes"]["lipophilicity"] = "fixed"
    elif change == "policy_extra":
        revised["comparison_policy"]["unregistered_guard_threshold"] = 100.
    elif change == "source_audit":
        revised["source_structure_audit"]["source_partition_action"] = "partition_pksim_standard"
    elif change == "baseline":
        revised["baseline_definitions"].clear()
    elif change == "policy_provenance":
        revised["comparison_policy"]["prediction_request"]["instance"]["protocol_source"]["source_checksum"] = "0" * 64
    elif change == "algorithm":
        revised["comparison_policy"]["algorithm"] = "purpose_driven"
    elif change == "fit_strategy":
        revised["comparison_fit_strategy"] = "residual"
    elif change == "budget_strategy":
        revised["comparison_budget_strategy"] = "equal"
    elif change == "missing_fit_strategy":
        revised.pop("comparison_fit_strategy")
    else:
        for name in ("residual_control", "screened_control", "study_consistent"):
            plans[name]["candidates"].pop()
    with pytest.raises(ValueError):
        pilot.verify_study_matched_plans(plans)


def test_study_matching_normalizes_only_verified_per_run_request_provenance(study_static_plans):
    plans = deepcopy(study_static_plans["alprazolam"])
    revised = plans["screened_control"]
    source = deepcopy(revised["comparison_request_source"])
    source["provenance_id"] = str(uuid5(NAMESPACE_URL, "separately_frozen_request"))
    revised["comparison_request_source"] = source
    revised["comparison_policy"]["prediction_request"]["instance"]["protocol_source"] = deepcopy(source)
    assert pilot.verify_study_matched_plans(plans)["matched"]
    # Drug/human evidence cannot use the request-source normalization exception.
    revised["comparison_policy"]["prediction_request"]["instance"]["human_source"] = deepcopy(source)
    with pytest.raises(ValueError, match="policies differ"):
        pilot.verify_study_matched_plans(plans)


def test_study_size_resource_defaults_and_legacy_defaults_are_registered(bundles):
    groups = pilot.protocol_groups("study_v4")
    assert len(groups) * len(bundles) == 8
    assert len(groups) * sum(len(bundle["external_scenarios"]) for bundle in bundles.values()) == 28
    assert pilot.protocol_defaults("study_v4") == {"max_solver_calls": 1000, "job_workers": 4, "scenario_workers": 3}
    for protocol in ("legacy_v2", "purpose_v3"):
        assert pilot.protocol_defaults(protocol) == {"max_solver_calls": 500, "job_workers": 2, "scenario_workers": 2}


def test_study_comparisons_separate_optimizer_and_guard_effects_and_keep_missing_values():
    summary = []
    for drug in pilot.INPUTS:
        for group, value in (("source_fixed", 2.), ("residual_control", 1.),
                             ("screened_control", 0.8), ("study_consistent", 0.7)):
            summary.append({"drug": drug, "group": group, "heldout_log_rmse": value,
                "auc_fold_error": value + 1., "cmax_fold_error": None if group == "study_consistent" else value + 1.,
                "heldout_complete": group != "study_consistent"})
    comparisons = pilot.matched_comparisons(summary, "study_v4")
    assert len(comparisons) == 6
    for comparison in comparisons:
        new, old = comparison["new"], comparison["control"]
        expected = {("study_consistent", "residual_control"): -0.3,
                    ("screened_control", "residual_control"): -0.2,
                    ("study_consistent", "screened_control"): -0.1}[(new, old)]
        assert comparison["difference_new_minus_control"]["heldout_log_rmse"] == pytest.approx(expected)
        if new == "study_consistent":
            assert comparison["difference_new_minus_control"]["cmax_fold_error"] is None
            assert comparison["both_predictions_complete"] is False
        else:
            assert comparison["both_predictions_complete"] is True
