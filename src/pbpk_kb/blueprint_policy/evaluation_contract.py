"""Declare an information-matched PK evaluation without executing or validating it.

Source metadata is caller-attested, not independently verified here. All
``visible_sources`` are treated as model-visible inputs; held-out observations
belong only in ``endpoint_references``. That evaluation-only argument never
changes the comparator set, information regime, or protocol fingerprint.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

VERSION = "pk-evaluation-contract-2"
INFORMATION_REGIMES = frozenset({
    "structure_only", "independent_parameters", "development_curve",
})
SUPPORTED_ENDPOINTS = frozenset({
    "plasma_concentration_time", "auc", "auc_inf", "cmax", "tmax", "c_last",
    "terminal_half_life", "clearance", "vz", "mrt", "vss",
})
HUMAN_PARAMETERS = frozenset({
    "human_pk_parameter", "independent_human_pk", "clinical_parameter",
})
CURVES = frozenset({"development_curve", "clinical_curve", "concentration_time"})
STRUCTURE_INPUTS = frozenset({
    "structure", "molecular_structure", "generic_physiology", "dosing_protocol",
    "predicted_parameter", "trained_model",
})
EXPERIMENTAL_INPUTS = frozenset({
    "physicochemical", "in_vitro", "animal_pk", "independent_parameter",
})
FORBIDDEN_KINDS = frozenset({
    "complete_same_drug_model", "target_outcome", "target_clinical",
})
FORBIDDEN_ROLES = frozenset({
    "target", "test", "validation", "evaluation", "selection", "forbidden",
})
ALLOWED_ROLES = frozenset({"independent", "development", "training", "generic", "input"})


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any, *, zero_allowed: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value >= 0 if zero_allowed else value > 0))


def _list(value: Any, name: str) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _canonical_metadata(record: Mapping[str, Any]) -> dict:
    """Retain every caller-supplied metadata field in JSON-compatible form.

    Unsupported/nonfinite metadata must block the protocol, not disappear from
    an otherwise eligible protocol's fingerprint.
    """
    return json.loads(json.dumps(dict(record), sort_keys=True, allow_nan=False))


def _comparator(name: str, *, fit_source: str = "none", predictive: bool = True,
                requirements: Sequence[str] = ()) -> dict:
    return {
        "name": name,
        "status": "not_executed",
        "fit_source": fit_source,
        "evaluation_role": "predictive" if predictive else "reconstruction_only",
        "predictive_comparison_eligible": predictive,
        "requirements": list(requirements),
        "clinical_results": None,
    }


def _reference_valid(endpoint: str, value: Any) -> bool:
    if endpoint != "plasma_concentration_time":
        return _number(value, zero_allowed=endpoint == "tmax")
    if not isinstance(value, Mapping):
        return False
    times, concentrations = value.get("times"), value.get("concentrations")
    if not isinstance(times, (list, tuple)) or not isinstance(concentrations, (list, tuple)):
        return False
    return (bool(times) and len(times) == len(concentrations)
            and all(_number(t, zero_allowed=True) for t in times)
            and all(_number(c) for c in concentrations)
            and all(a < b for a, b in zip(times, times[1:])))


def build_evaluation_contract(
    *,
    information_regime: str,
    requested_endpoints: Sequence[str],
    compound_id: str,
    target_study_id: str,
    development_point_count: int = 0,
    development_study_id: str | None = None,
    visible_sources: Sequence[Mapping[str, Any]] = (),
    hypothesis: str | None = None,
    endpoint_references: Mapping[str, Any] | None = None,
    learned_compartment_baselines: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """Build a protocol; ``protocol_eligible`` never means clinically validated.

    Each source requires ``kind``, ``role``, ``study_id`` (nullable for generic
    inputs), and ``compound_id`` (nullable for generic physiology/protocol).
    Target study metadata is allowed only for ``kind='dosing_protocol'`` with
    ``role='input'``; clinical source kinds never receive this exception.
    ``development_point_count`` counts actual measured observations only, never
    interpolated/simulated points. Exactly one authoritative development-curve
    source must attest the same ``actual_point_count`` and
    ``point_basis='observed'``. This version does not union multiple curves.
    PK priors from that development study are not independent parameter inputs.

    External learned compartment models require ``name``, ``compartments``
    (1/2/3), ``model_version``, ``information_regime`` and affirmative
    ``target_compound_excluded``/``target_study_excluded`` declarations.
    Unknown or inconsistent evidence blocks the protocol instead of silently
    reclassifying its regime. No filesystem, network, fitting, or scoring occurs.
    """
    if information_regime not in INFORMATION_REGIMES:
        raise ValueError("Unknown information_regime")
    if not _identifier(compound_id) or not _identifier(target_study_id):
        raise ValueError("compound_id and target_study_id must be explicit")
    if (isinstance(development_point_count, bool)
            or not isinstance(development_point_count, int) or development_point_count < 0):
        raise ValueError("development_point_count must be a nonnegative integer")
    if hypothesis is not None and not _identifier(hypothesis):
        raise ValueError("hypothesis must be a nonempty string or None")
    endpoints = _list(requested_endpoints, "requested_endpoints")
    if (not endpoints or any(not isinstance(e, str) for e in endpoints)
            or len(set(endpoints)) != len(endpoints)):
        raise ValueError("requested_endpoints must contain distinct endpoint names")
    references = {} if endpoint_references is None else endpoint_references
    if not isinstance(references, Mapping):
        raise TypeError("endpoint_references must be a mapping")

    reasons: list[str] = []
    if any(e not in SUPPORTED_ENDPOINTS for e in endpoints):
        reasons.append("unsupported_requested_endpoint")
    if set(references) - set(endpoints):
        reasons.append("reference_for_unrequested_endpoint")
    if information_regime == "development_curve":
        if development_point_count < 3:
            reasons.append("development_curve_requires_at_least_three_actual_points")
        if not _identifier(development_study_id):
            reasons.append("missing_development_study_id")
        elif development_study_id == target_study_id:
            reasons.append("development_study_is_target_study")
    elif development_point_count or development_study_id is not None:
        reasons.append("development_curve_metadata_in_no_curve_regime")

    source_audit: list[dict] = []
    human_pk_visible = False
    actual_curve_attested = False
    curve_source_count = 0
    structure_attested = False
    independent_parameter_attested = False
    for index, source in enumerate(_list(visible_sources, "visible_sources")):
        if not isinstance(source, Mapping):
            reasons.append(f"source_{index}:invalid_source_record")
            source_audit.append({"index": index, "accepted": False,
                                 "reasons": ["invalid_source_record"]})
            continue
        record = deepcopy(dict(source))
        kind = record.get("kind") if isinstance(record.get("kind"), str) else None
        role = record.get("role") if isinstance(record.get("role"), str) else None
        study = record.get("study_id") if isinstance(record.get("study_id"), str) else None
        compound = record.get("compound_id") if isinstance(record.get("compound_id"), str) else None
        issues: list[str] = []
        try:
            source_metadata = _canonical_metadata(record)
        except (TypeError, ValueError):
            source_metadata = None
            issues.append("source_metadata_not_finite_json")
        if not all(k in record for k in ("kind", "role", "study_id", "compound_id")):
            issues.append("missing_source_metadata")
        if not isinstance(kind, str) or kind not in (
                STRUCTURE_INPUTS | EXPERIMENTAL_INPUTS | HUMAN_PARAMETERS | CURVES | FORBIDDEN_KINDS):
            issues.append("unknown_source_kind")
        if not isinstance(role, str) or role not in ALLOWED_ROLES | FORBIDDEN_ROLES:
            issues.append("unknown_source_role")
        if kind in FORBIDDEN_KINDS:
            issues.append("forbidden_source_kind")
        known_target_dose = kind == "dosing_protocol" and role == "input"
        if role in FORBIDDEN_ROLES or (study == target_study_id and not known_target_dose):
            issues.append("held_out_target_visible_to_model")
        if compound != compound_id and not (
                compound is None and kind in {"generic_physiology", "dosing_protocol", "trained_model"}):
            issues.append("source_compound_mismatch")
        if record.get("study_id") is not None and not _identifier(record["study_id"]):
            issues.append("invalid_source_study_id")
        if record.get("compound_id") is not None and not _identifier(record["compound_id"]):
            issues.append("invalid_source_compound_id")
        if kind in HUMAN_PARAMETERS | CURVES:
            human_pk_visible = True
            if not _identifier(study):
                issues.append("clinical_source_requires_study_id")
        if information_regime == "structure_only" and kind not in STRUCTURE_INPUTS:
            issues.append("source_exceeds_structure_only_budget")
        if kind in HUMAN_PARAMETERS and role != "independent":
            issues.append("human_pk_parameter_must_be_independent")
        if record.get("derived_from") == "full_development_curve_NCA":
            issues.append("full_development_curve_summary_exceeds_declared_point_budget")
        if kind in HUMAN_PARAMETERS | {"independent_parameter"} and development_study_id is not None:
            if study == development_study_id:
                issues.append("human_pk_prior_from_development_study_not_independent"
                              if kind in HUMAN_PARAMETERS else
                              "parameter_from_development_study_not_independent")
        if kind in CURVES:
            curve_source_count += 1
            if information_regime != "development_curve":
                issues.append("curve_forbidden_in_no_curve_regime")
            if role != "development" or study != development_study_id:
                issues.append("curve_not_from_declared_development_study")
            count = record.get("actual_point_count")
            if (type(count) is not int or count != development_point_count
                    or record.get("point_basis") != "observed"):
                issues.append("actual_development_points_not_attested")
            if not issues:
                actual_curve_attested = True
        if kind in {"predicted_parameter", "trained_model"}:
            if (record.get("target_compound_excluded") is not True
                    or record.get("target_study_excluded") is not True
                    or not _identifier(record.get("model_version"))):
                issues.append("prediction_training_exclusion_unverified")
        if kind in {"structure", "molecular_structure"} and not issues:
            structure_attested = True
        if kind in HUMAN_PARAMETERS | EXPERIMENTAL_INPUTS | {"predicted_parameter"} and not issues:
            independent_parameter_attested = True
        reasons.extend(f"source_{index}:{issue}" for issue in issues)
        source_audit.append({"index": index, "kind": kind, "role": role,
                             "study_id": study, "compound_id": compound,
                             "source_metadata": source_metadata,
                             "accepted": not issues, "reasons": issues})
    if information_regime == "development_curve" and not actual_curve_attested:
        reasons.append("missing_qualified_development_curve_source")
    if information_regime == "development_curve" and curve_source_count != 1:
        reasons.append("exactly_one_authoritative_development_curve_source_required")
    if information_regime == "structure_only" and not structure_attested:
        reasons.append("missing_structure_source")
    if information_regime == "independent_parameters" and not independent_parameter_attested:
        reasons.append("missing_qualified_independent_parameter_source")

    requirements = ["same_visible_input_budget", "same_target_study_and_endpoints",
                    "no_target_outcome_fitting_or_selection"]
    comparators = [
        _comparator("fixed_formula_pbpk", requirements=requirements + [
            "required_parameters_from_visible_inputs_or_qualified_predictions"]),
        _comparator("rf_parameters_pbpk", requirements=requirements + [
            "external_training_target_compound_and_study_excluded"]),
        _comparator("onecomp_CL_Vss", requirements=requirements + [
            "CL_and_Vss_available_from_declared_inputs_or_qualified_predictions",
            "no_target_curve_parameter_fit"]),
    ]
    if information_regime == "development_curve":
        comparators.extend([
            _comparator("old_pbpk", fit_source="development_study", requirements=requirements),
            _comparator("two_compartment", fit_source="development_study", requirements=requirements + [
                "parameter_identifiability_checked_separately"]),
            _comparator("two_compartment_CLprior", fit_source="development_study",
                        requirements=requirements + ["CL_prior_from_independent_source",
                                                     "parameter_identifiability_checked_separately"]),
        ])
    for index, model in enumerate(_list(learned_compartment_baselines, "learned_compartment_baselines")):
        if not isinstance(model, Mapping):
            reasons.append(f"learned_baseline_{index}:invalid_record")
            continue
        try:
            model_metadata = _canonical_metadata(model)
        except (TypeError, ValueError):
            reasons.append(f"learned_baseline_{index}:metadata_not_finite_json")
            continue
        if (not _identifier(model.get("name")) or not _identifier(model.get("model_version"))
                or type(model.get("compartments")) is not int or model["compartments"] not in {1, 2, 3}
                or model.get("information_regime") != information_regime
                or model.get("target_compound_excluded") is not True
                or model.get("target_study_excluded") is not True):
            reasons.append(f"learned_baseline_{index}:input_or_training_exclusion_unverified")
            continue
        entry = _comparator("learned_compartment:" + model["name"],
                            fit_source="external_training_only", requirements=requirements)
        entry.update({"compartments": model["compartments"], "model_version": model["model_version"],
                      "model_metadata": model_metadata,
                      "training_exclusion": "caller_attested_target_compound_and_study_excluded"})
        if any(c["name"] == entry["name"] for c in comparators):
            reasons.append("duplicate_learned_comparator_name")
        else:
            comparators.append(entry)

    # This diagnostic is never included among predictive ranking candidates.
    reconstruction = _comparator("target_fitted_compartment_model", fit_source="target_study",
                                  predictive=False,
                                  requirements=["report_only_as_reconstruction_upper_bound"])
    endpoint_contracts = []
    for endpoint in endpoints:
        reference = references.get(endpoint)
        valid_reference = reference is not None and _reference_valid(endpoint, reference)
        if reference is not None and not valid_reference:
            reasons.append(f"endpoint_{endpoint}:invalid_reference")
        metric = ("clinical_GMFE_and_log_bias" if endpoint == "plasma_concentration_time"
                  else "absolute_error_and_cohort_MAE_in_hours" if endpoint == "tmax"
                  else "fold_error")
        endpoint_contracts.append({
            "endpoint": endpoint, "metric": metric,
            "reference": deepcopy(reference) if valid_reference else None,
            "reference_role": "held_out_scoring_only",
            "score": None,
            "status": "not_executed" if valid_reference else "undefined_missing_reference",
            "missing_reference_policy": "null_never_zero_or_assumed_correct",
        })

    reporting = {
        "accuracy": ["clinical_GMFE", "signed_log_bias", "endpoint_fold_error", "Tmax_MAE_hours"],
        "aggregation": "paired_by_compound_and_study_then_macro_average_no_dense_curve_dominance",
        "denominator_policy": "report_all_attempted_cases_and_paired_evaluable_cases",
        "execution_coverage": {"attempted": None, "succeeded": None, "failed": None,
                               "failure_reasons": None, "abstained": None},
        "cost": {"wall_time_seconds": None, "cpu_seconds": None, "peak_memory_bytes": None},
        "uncertainty": {"nominal_interval_level": None, "empirical_coverage": None,
                        "interval_width": None, "status": "not_evaluated",
                        "twofold_accuracy_is_95pct_coverage": False},
        "composite_score": None,
        "engineering_metrics_replace_clinical_accuracy": False,
        "change_endpoints_or_metric_after_winner_known": False,
        "failed_cases_may_be_silently_dropped": False,
    }
    budget = {
        "regime": information_regime, "development_point_count": development_point_count,
        "development_study_id": development_study_id, "target_study_id": target_study_id,
        "human_pk_visible": human_pk_visible,
        "first_in_human": False if human_pk_visible else None,
        "task_label": ("development_curve_to_held_out_study" if information_regime == "development_curve"
                       else "independent_parameters_to_held_out_study" if human_pk_visible
                       else "prediction_without_development_curve"),
        "first_in_human_claim_requires_additional_provenance": True,
        "input_requirements": {
            "structure_only": ["molecular_structure", "declared_dose_and_physiology_at_execution",
                               "no_compound_experimental_or_human_pk_inputs",
                               "training_exclusion_for_learned_predictions"],
            "independent_parameters": ["independent_parameter_provenance",
                                       "no_clinical_curve_inputs", "no_target_outcome_inputs",
                                       "target_dosing_protocol_allowed_as_known_input"],
            "development_curve": ["at_least_three_actual_observed_points",
                                  "exactly_one_authoritative_curve_source",
                                  "development_study_distinct_from_target",
                                  "PK_priors_from_other_independent_studies_only",
                                  "identifiability_not_guaranteed_by_point_count"],
        }[information_regime],
    }
    frozen = {"schema_version": VERSION, "compound_id": compound_id,
              "information_budget": budget, "requested_endpoints": endpoints,
              "hypothesis": hypothesis, "source_audit": source_audit,
              "comparator_contracts": comparators, "reporting_contract": reporting}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {
        **frozen,
        "status": "protocol_only_not_executed", "protocol_eligible": not reasons,
        "clinically_validated": False, "clinical_superiority_established": False,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "protocol_fingerprint": fingerprint,
        "endpoint_contracts": endpoint_contracts,
        "reconstruction_contracts": [reconstruction],
        "provenance_verification": "caller_attested_not_independently_verified",
    }
