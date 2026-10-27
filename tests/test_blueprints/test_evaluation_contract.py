"""Protocol leakage and claim-boundary tests; no PK fits or clinical assertions."""
from copy import deepcopy

import pytest

from pbpk_kb.blueprint_policy.evaluation_contract import build_evaluation_contract


def _source(kind="structure", role="independent", study_id=None, **extra):
    return dict(kind=kind, role=role, study_id=study_id, compound_id="drug:A", **extra)


def _build(**overrides):
    kwargs = dict(information_regime="structure_only", compound_id="drug:A",
                  target_study_id="target", requested_endpoints=["auc", "tmax"],
                  visible_sources=[_source()])
    kwargs.update(overrides)
    return build_evaluation_contract(**kwargs)


def test_protocol_is_never_execution_or_validation_and_has_no_curve_baselines():
    result = _build()
    assert result["protocol_eligible"]
    assert result["status"] == "protocol_only_not_executed"
    assert not result["clinically_validated"]
    assert {c["name"] for c in result["comparator_contracts"]} == {
        "fixed_formula_pbpk", "rf_parameters_pbpk", "onecomp_CL_Vss"}
    oracle = result["reconstruction_contracts"][0]
    assert oracle["fit_source"] == "target_study"
    assert oracle["evaluation_role"] == "reconstruction_only"
    assert not oracle["predictive_comparison_eligible"]


@pytest.mark.parametrize("source", [
    _source("human_pk_parameter", study_id="external"),
    _source("complete_same_drug_model", study_id="external"),
    _source("target_outcome", study_id="other-id"),
    _source(role="evaluation", study_id="alias"),
    _source(study_id="target"),
    _source("clinical_curve", study_id="external"),
])
def test_structure_regime_blocks_hidden_clinical_information(source):
    assert not _build(visible_sources=[source])["protocol_eligible"]


def test_known_target_dose_is_allowed_but_target_clinical_inputs_remain_forbidden():
    protocol = _source("dosing_protocol", "input", "target",
                       route="intravenous", dose=100, dose_unit="mg")
    result = _build(visible_sources=[_source(), protocol])
    assert result["protocol_eligible"]
    assert result["source_audit"][1]["accepted"]
    for kind in ["human_pk_parameter", "clinical_curve", "target_outcome"]:
        source = dict(protocol, kind=kind)
        blocked = _build(information_regime="independent_parameters",
                         visible_sources=[source])
        assert not blocked["protocol_eligible"]
        assert "held_out_target_visible_to_model" in blocked["source_audit"][0]["reasons"]
    for role in ["independent", "target", "evaluation"]:
        assert not _build(visible_sources=[_source(), dict(protocol, role=role)])["protocol_eligible"]


def test_zero_development_points_with_human_pk_is_not_first_in_human():
    result = _build(information_regime="independent_parameters",
                    visible_sources=[_source("human_pk_parameter", study_id="independent")])
    assert result["protocol_eligible"]
    assert result["information_budget"]["first_in_human"] is False
    assert result["information_budget"]["human_pk_visible"]


@pytest.mark.parametrize("kind", ["complete_same_drug_model", "target_outcome"])
def test_forbidden_inputs_do_not_become_allowed_in_richer_regimes(kind):
    assert not _build(information_regime="independent_parameters",
                      visible_sources=[_source(kind, study_id="independent")])["protocol_eligible"]


def _development(**overrides):
    kwargs = dict(information_regime="development_curve", development_point_count=3,
                  development_study_id="development", visible_sources=[
                      _source("development_curve", "development", "development",
                              actual_point_count=3, point_basis="observed")])
    kwargs.update(overrides)
    return _build(**kwargs)


def test_development_comparison_contains_strong_compartment_controls():
    result = _development()
    assert result["protocol_eligible"]
    fitted = {c["name"] for c in result["comparator_contracts"]
              if c["fit_source"] == "development_study"}
    assert fitted == {"old_pbpk", "two_compartment", "two_compartment_CLprior"}


def test_full_development_nca_prior_cannot_bypass_three_point_information_budget():
    curve = _source("development_curve", "development", "development",
                    actual_point_count=3, point_basis="observed")
    prior = _source("human_pk_parameter", "independent", "development",
                    parameter="clearance", value=12,
                    derived_from="full_development_curve_NCA")
    result = _development(visible_sources=[curve, prior])
    assert not result["protocol_eligible"]
    assert "human_pk_prior_from_development_study_not_independent" in result["source_audit"][1]["reasons"]
    assert "full_development_curve_summary_exceeds_declared_point_budget" in result["source_audit"][1]["reasons"]
    # Relabeling provenance alone cannot hide the explicitly declared derivation.
    assert not _development(visible_sources=[curve, dict(prior, study_id="independent")])["protocol_eligible"]
    # An actually independent study is a legitimate prior under this contract.
    independent = dict(prior, study_id="independent", derived_from="independent_study_NCA")
    assert _development(visible_sources=[curve, independent])["protocol_eligible"]


@pytest.mark.parametrize("kind", ["independent_parameter", "physicochemical", "structure"])
def test_full_curve_summary_is_forbidden_even_when_aliased_as_another_source_kind(kind):
    curve = _source("development_curve", "development", "development",
                    actual_point_count=3, point_basis="observed")
    prior = _source(kind, "independent", "development", parameter="CL",
                    derived_from="full_development_curve_NCA")
    result = _development(visible_sources=[curve, prior])
    assert not result["protocol_eligible"]
    assert "full_development_curve_summary_exceeds_declared_point_budget" in result["source_audit"][1]["reasons"]


def test_same_development_independent_parameter_is_rejected_without_overblocking_physicochemical_data():
    curve = _source("development_curve", "development", "development",
                    actual_point_count=3, point_basis="observed")
    prior = _source("independent_parameter", "independent", "development", parameter="CL")
    result = _development(visible_sources=[curve, prior])
    assert not result["protocol_eligible"]
    assert "parameter_from_development_study_not_independent" in result["source_audit"][1]["reasons"]
    assay = _source("physicochemical", "independent", "development", parameter="logp")
    assert _development(visible_sources=[curve, assay])["protocol_eligible"]


def test_two_curve_records_cannot_each_claim_the_same_global_three_point_budget():
    first = _source("development_curve", "development", "development",
                    source_id="first", actual_point_count=3, point_basis="observed")
    second = dict(first, source_id="second")
    for sources in [[first, second], [first, deepcopy(first)]]:
        result = _development(visible_sources=sources)
        assert not result["protocol_eligible"]
        assert "exactly_one_authoritative_development_curve_source_required" in result["blocking_reasons"]


@pytest.mark.parametrize("overrides", [
    {"development_point_count": 2},
    {"development_study_id": "target"},
    {"development_study_id": None},
    {"visible_sources": []},
    {"visible_sources": [_source("development_curve", "development", "development",
                                  actual_point_count=3, point_basis="interpolated")]},
    {"visible_sources": [_source("development_curve", "development", "target",
                                  actual_point_count=3, point_basis="observed")]},
])
def test_development_regime_requires_actual_independent_observations(overrides):
    assert not _development(**overrides)["protocol_eligible"]


def test_curve_cannot_be_smuggled_into_independent_parameter_regime():
    assert not _build(information_regime="independent_parameters", visible_sources=[
        _source("clinical_curve", "independent", "external")])["protocol_eligible"]
    assert not _build(information_regime="independent_parameters")["protocol_eligible"]


def test_qualified_external_two_compartment_model_is_allowed_without_development_curve():
    model = dict(name="external-2cmt", compartments=2, model_version="v1",
                 information_regime="structure_only", target_compound_excluded=True,
                 target_study_excluded=True)
    result = _build(learned_compartment_baselines=[model])
    assert result["protocol_eligible"]
    learned = result["comparator_contracts"][-1]
    assert learned["compartments"] == 2
    assert learned["fit_source"] == "external_training_only"
    assert learned["predictive_comparison_eligible"]
    for field in ["target_compound_excluded", "target_study_excluded"]:
        assert not _build(learned_compartment_baselines=[dict(model, **{field: False})])["protocol_eligible"]


def test_unknown_sources_fail_closed_instead_of_becoming_structure_inputs():
    assert not _build(visible_sources=[_source("unclassified_clinical_data")])["protocol_eligible"]
    assert not _build(visible_sources=[{"kind": "structure"}])["protocol_eligible"]
    assert not _build(visible_sources=[])["protocol_eligible"]


@pytest.mark.parametrize("field,value", [
    ("kind", {}), ("role", []), ("study_id", float("nan")), ("compound_id", []),
])
def test_malformed_source_metadata_is_rejected_without_crashing(field, value):
    record = _source()
    record[field] = value
    assert not _build(visible_sources=[record])["protocol_eligible"]


def test_missing_endpoint_is_null_and_never_assumed_perfect():
    result = _build(endpoint_references={"auc": 12.0})
    assert result["protocol_eligible"]
    auc, tmax = result["endpoint_contracts"]
    assert auc["reference"] == 12.0 and auc["score"] is None
    assert tmax["reference"] is None and tmax["score"] is None
    assert tmax["status"] == "undefined_missing_reference"
    assert tmax["metric"] == "absolute_error_and_cohort_MAE_in_hours"


@pytest.mark.parametrize("reference", [0, -1, True, float("nan"), float("inf")])
def test_invalid_auc_reference_cannot_create_defined_fold_error(reference):
    result = _build(endpoint_references={"auc": reference})
    assert not result["protocol_eligible"]
    assert result["endpoint_contracts"][0]["reference"] is None


def test_tmax_at_zero_is_valid_and_does_not_require_fold_error():
    result = _build(endpoint_references={"tmax": 0})
    assert result["protocol_eligible"]
    assert result["endpoint_contracts"][1]["reference"] == 0


def test_target_reference_values_cannot_reselect_protocol_and_inputs_are_unmodified():
    sources = [_source()]
    original = deepcopy(sources)
    first = _build(visible_sources=sources, endpoint_references={"auc": 1})
    second = _build(visible_sources=sources, endpoint_references={"auc": 100})
    assert first["protocol_fingerprint"] == second["protocol_fingerprint"]
    assert first["comparator_contracts"] == second["comparator_contracts"]
    assert sources == original
    assert _build(requested_endpoints=["auc"])["protocol_fingerprint"] != first["protocol_fingerprint"]


def test_fingerprint_preserves_source_identity_version_and_nested_metadata():
    prediction = _source("predicted_parameter", model_version="v1", source_id="model-source-A",
                         target_compound_excluded=True, target_study_excluded=True,
                         provenance={"training_dataset": "cohort-A", "checksum": "abc"})
    baseline = _build(visible_sources=[_source(), prediction])
    assert baseline["protocol_eligible"]
    assert baseline["source_audit"][1]["source_metadata"] == prediction
    alternatives = [dict(prediction, model_version="v2"), dict(prediction, source_id="model-source-B"),
                    dict(prediction, provenance={"training_dataset": "cohort-B", "checksum": "abc"})]
    for changed in alternatives:
        result = _build(visible_sources=[_source(), changed])
        assert result["protocol_eligible"]
        assert result["protocol_fingerprint"] != baseline["protocol_fingerprint"]
    reordered = dict(reversed(list(prediction.items())))
    assert _build(visible_sources=[_source(), reordered])["protocol_fingerprint"] == baseline["protocol_fingerprint"]


def test_nonfinite_nested_metadata_blocks_rather_than_dropping_unhashed_fields():
    source = _source(provenance={"value": float("nan")})
    result = _build(visible_sources=[source])
    assert not result["protocol_eligible"]
    assert "source_metadata_not_finite_json" in result["source_audit"][0]["reasons"]


def test_accuracy_failures_cost_and_interval_coverage_remain_separate():
    report = _build()["reporting_contract"]
    assert "clinical_GMFE" in report["accuracy"]
    assert "signed_log_bias" in report["accuracy"]
    assert "failed" in report["execution_coverage"] and "wall_time_seconds" in report["cost"]
    assert report["uncertainty"]["empirical_coverage"] is None
    assert not report["uncertainty"]["twofold_accuracy_is_95pct_coverage"]
    assert report["composite_score"] is None
    assert not report["engineering_metrics_replace_clinical_accuracy"]
    assert not report["change_endpoints_or_metric_after_winner_known"]


def test_unsupported_endpoints_and_nondeclared_targets_are_blocked():
    assert not _build(requested_endpoints=["invented_success_metric"])["protocol_eligible"]
    with pytest.raises(ValueError):
        _build(target_study_id="")
    with pytest.raises(ValueError):
        _build(development_point_count=True)
