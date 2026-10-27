"""Synthetic static-contract checks; no KB, model, solver or experimental data."""
from copy import deepcopy
import hashlib
from uuid import UUID

import pytest
from pydantic import ValidationError

from pbpk_kb.blueprints.design import design_blueprint
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest, DesignBudget


def _provenance(source="fixture-rules"):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return {
        "provenance_id": str(UUID(digest[:32])),
        "provenance_kind": "ontology_rule",
        "source_id": source,
        "source_location": "synthetic-test-fixture",
        "source_checksum": digest,
    }


def _study(study_id, role, source=None, cohort=None):
    source = source or study_id
    return {
        "study_id": study_id,
        "source_id": source,
        "source_checksum": _provenance(source)["source_checksum"],
        "role": role,
        "cohort_ids": [cohort or study_id + "-cohort"],
        "observed_target_ids": ["parent_plasma"],
    }


def _payload():
    physiology_source = _provenance("fixture-human-physiology")
    compartment = {
        "compartment_id": "central", "semantic_id": "physiology:central",
        "name": "Fixture central compartment", "compartment_type": "vascular",
        "provenance": [physiology_source],
    }
    base_parameter = {
        "parameter_id": "central-volume", "semantic_id": "physiology:volume",
        "name": "Fixture physiological volume", "model_path": "central.volume",
        "dimension": "volume", "canonical_unit": "L", "origin": "observed",
        "value": 3.0, "evidence_id": physiology_source["provenance_id"],
        "context": {"species": "human", "population": "healthy_adult",
                    "compartment_id": "central"},
        "provenance": [physiology_source], "blocking": False,
    }
    parameter_spec = {
        "semantic_id": "drug:clearance", "dimension": "clearance", "unit": "L/h",
        "estimate_allowed": True, "observation_target_ids": ["parent_plasma"],
    }
    module_common = {
        "version": "fixture-1", "required_compartments": ["central"],
        "requires_evidence": False, "provenance": [_provenance()],
    }
    modules = [
        dict(module_common, module_id="distribution", mechanism="distribution",
             role="distribution", provides_targets=["parent_plasma"]),
        dict(module_common, module_id="oral_input", mechanism="oral_input",
             role="input", routes=["oral"]),
        dict(module_common, module_id="iv_input", mechanism="iv_input",
             role="input", routes=["intravenous"]),
        dict(module_common, module_id="elimination", mechanism="clearance",
             role="elimination", requires_modules=["distribution"],
             requires_evidence=True, parameter_specs=[parameter_spec]),
    ]
    return {
        "requirement": {
            "compound_semantic_id": "compound:fixture", "name": "Synthetic fixture",
            "route": "oral", "dose": 1.0, "dose_unit": "mg",
            "targets": [{"target_id": "parent_plasma", "matrix": "plasma",
                         "quantity": "concentration"}],
        },
        "base": {
            "base_id": "human-fixture", "version": "fixture-1",
            "compartments": [compartment], "parameters": [base_parameter],
            "supported_populations": ["healthy_adult"],
            "provenance": [physiology_source],
        },
        "modules": modules,
        "evidence": [
            {"evidence_id": "mechanism", "compound_semantic_id": "compound:fixture",
             "kind": "mechanism", "mechanism": "clearance",
             "provenance": _provenance("fixture-mechanism")},
            {"evidence_id": "parameter", "compound_semantic_id": "compound:fixture",
             "kind": "parameter", "parameter_semantic_id": "drug:clearance",
             "value": 1.0, "unit": "L/h", "lower_bound": 0.25, "upper_bound": 4.0,
             "provenance": _provenance("fixture-parameter")},
        ],
        "studies": [_study("calibration", "calibration")],
    }


def _design(payload):
    return design_blueprint(BlueprintDesignRequest.model_validate(payload))


def _elimination_candidates(result):
    return [c for c in result["candidates"] if "elimination" in c["module_ids"]]


def _gap_codes(candidate):
    return {gap["code"] for gap in candidate["gaps"]}


@pytest.mark.parametrize("route,correct,incorrect", [
    ("oral", "oral_input", "iv_input"),
    ("intravenous", "iv_input", "oral_input"),
])
def test_route_and_dependency_assembly_without_compound_template(route, correct, incorrect):
    payload = _payload()
    payload["requirement"]["route"] = route
    result = _design(payload)
    assembled = [c for c in result["candidates"] if _gap_codes(c) == {"BASE_MODEL_UNBOUND"}]
    assert assembled
    assert all(c["status"] == "draft_with_gaps" for c in assembled)
    assert all({correct, "distribution", "elimination"} <= set(c["module_ids"])
               for c in assembled)
    assert all(incorrect not in c["module_ids"] for c in result["candidates"])
    assert {"module_id": incorrect, "reason": "ROUTE_UNSUPPORTED"} in result["unavailable_modules"]
    assert all({"from": "distribution", "to": "elimination", "relation": "requires"}
               in c["module_edges"] for c in assembled)
    # Only generic physiology/rules and synthetic drug evidence were supplied.
    assert "template" not in payload and "model_asset" not in str(payload)


def test_fixed_individual_and_joint_parameter_variants_are_distinct_drafts():
    payload = _payload()
    module = payload["modules"][-1]
    second = deepcopy(module["parameter_specs"][0])
    second.update(semantic_id="drug:volume", dimension="volume", unit="L")
    module["parameter_specs"].append(second)
    evidence = deepcopy(payload["evidence"][1])
    evidence.update(evidence_id="volume", parameter_semantic_id="drug:volume", unit="L")
    payload["evidence"].append(evidence)
    candidates = _elimination_candidates(_design(payload))
    released = {tuple(c["estimated_parameter_ids"]) for c in candidates}
    assert {(), ("drug:clearance",), ("drug:volume",),
            ("drug:clearance", "drug:volume")} <= released
    for candidate in candidates:
        for parameter in candidate["parameters"]:
            assert parameter["value"] == (None if parameter["mode"] == "to_estimate" else 1.0)
            assert parameter["initial_value"] == 1.0
            assert parameter["calibration_study_ids"] == ["calibration"]


def test_absent_mechanism_cannot_be_inferred_from_requested_target():
    payload = _payload()
    payload["evidence"] = payload["evidence"][1:]
    result = _design(payload)
    assert not _elimination_candidates(result)
    assert {"module_id": "elimination", "reason": "MECHANISM_EVIDENCE_MISSING"} in result["unavailable_modules"]
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])


def test_missing_parameter_never_creates_a_numeric_value():
    payload = _payload()
    payload["evidence"] = payload["evidence"][:1]
    candidates = _elimination_candidates(_design(payload))
    assert candidates
    for candidate in candidates:
        parameter = candidate["parameters"][0]
        assert parameter["mode"] == "missing"
        assert parameter["value"] is None and parameter["initial_value"] is None
        assert "PARAMETER_NOT_RESOLVED" in _gap_codes(candidate)


@pytest.mark.parametrize("kind", ["value", "bounds", "unit"])
def test_conflicting_or_unmatched_parameter_evidence_stays_unresolved(kind):
    payload = _payload()
    fact = payload["evidence"][1]
    if kind == "unit":
        fact["unit"] = "mL/h"
        expected = "PARAMETER_UNIT_MISMATCH"
    else:
        other = deepcopy(fact)
        other["evidence_id"] = "conflict"
        if kind == "value":
            other["value"] = 2.0
            expected = "CONFLICTING_PARAMETER_VALUES"
        else:
            other.update(value=None, lower_bound=5.0, upper_bound=6.0)
            expected = "CONFLICTING_PARAMETER_BOUNDS"
        payload["evidence"].append(other)
    for candidate in _elimination_candidates(_design(payload)):
        parameter = candidate["parameters"][0]
        assert parameter["mode"] == "missing"
        assert parameter["value"] is None and parameter["initial_value"] is None
        assert expected in _gap_codes(candidate)


@pytest.mark.parametrize("study_role", ["selection", "test"])
def test_bounds_alone_do_not_enable_estimation_without_calibration(study_role):
    payload = _payload()
    payload["evidence"][1]["value"] = None
    payload["studies"][0]["role"] = study_role
    for candidate in _elimination_candidates(_design(payload)):
        assert candidate["parameters"][0]["mode"] == "missing"
        assert candidate["parameters"][0]["initial_value"] is None


@pytest.mark.parametrize("transfer_allowed", [False, True])
def test_cross_route_parameter_transfer_requires_explicit_spec(transfer_allowed):
    payload = _payload()
    payload["evidence"][1]["route"] = "intravenous"
    payload["modules"][-1]["parameter_specs"][0]["route_independent"] = transfer_allowed
    candidates = _elimination_candidates(_design(payload))
    modes = {p["mode"] for c in candidates for p in c["parameters"]}
    assert ("fixed_from_evidence" in modes) is transfer_allowed
    assert ("missing" in modes) is not transfer_allowed


@pytest.mark.parametrize("location", ["evidence", "base", "compartment", "base_parameter", "module"])
def test_test_source_is_rejected_at_every_provenance_entry(location):
    payload = _payload()
    payload["studies"].append(_study("held-out", "test"))
    reference = _provenance("held-out")
    if location == "evidence":
        payload["evidence"][0]["provenance"] = reference
    elif location == "base":
        payload["base"]["provenance"] = [reference]
    elif location == "compartment":
        payload["base"]["compartments"][0]["provenance"] = [reference]
    elif location == "base_parameter":
        payload["base"]["parameters"][0].update(provenance=[reference], evidence_id=reference["provenance_id"])
    else:
        payload["modules"][0]["provenance"] = [reference]
    with pytest.raises(ValidationError, match="forbidden or test source"):
        BlueprintDesignRequest.model_validate(payload)


def test_test_source_alias_checksum_cannot_bypass_isolation():
    payload = _payload()
    payload["studies"].append(_study("held-out", "test"))
    payload["evidence"][0]["provenance"] = dict(_provenance("held-out"), source_id="renamed-source")
    with pytest.raises(ValidationError, match="forbidden or test source"):
        BlueprintDesignRequest.model_validate(payload)


@pytest.mark.parametrize("identity", ["cohort", "source", "checksum"])
def test_studies_cannot_share_source_or_cohort_across_roles(identity):
    payload = _payload()
    first = payload["studies"][0]
    second = _study("held-out", "test")
    if identity == "cohort":
        second["cohort_ids"] = first["cohort_ids"]
    elif identity == "source":
        second["source_id"] = first["source_id"]
    else:
        second["source_checksum"] = first["source_checksum"]
    payload["studies"].append(second)
    with pytest.raises(ValidationError, match="cannot cross data roles"):
        BlueprintDesignRequest.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("compound_semantic_id", "compound:fixture"),
    ("species", "mouse"), ("population", "unverified_population"),
])
def test_contaminated_or_mislabelled_physiology_is_rejected(field, value):
    payload = _payload()
    payload["base"]["parameters"][0]["context"][field] = value
    with pytest.raises(ValidationError):
        BlueprintDesignRequest.model_validate(payload)


@pytest.mark.parametrize("location", ["dose", "evidence", "base", "uncertainty"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_are_rejected_including_reused_schema(location, value):
    payload = _payload()
    if location == "dose":
        payload["requirement"]["dose"] = value
    elif location == "evidence":
        payload["evidence"][1]["value"] = value
    else:
        payload["base"]["parameters"][0]["uncertainty" if location == "uncertainty" else "value"] = value
    with pytest.raises(ValidationError):
        BlueprintDesignRequest.model_validate(payload)


def test_physiology_requires_explicit_population_scope():
    payload = _payload()
    del payload["base"]["supported_populations"]
    with pytest.raises(ValidationError, match="supported_populations"):
        BlueprintDesignRequest.model_validate(payload)


def test_unsupported_population_returns_gaps_without_claiming_applicability():
    payload = _payload()
    payload["requirement"]["population"] = "unverified_population"
    result = _design(payload)
    assert result["candidates"]
    assert all("HUMAN_BASE_POPULATION_UNSUPPORTED" in _gap_codes(c)
               and c["status"] == "draft_with_gaps" for c in result["candidates"])


@pytest.mark.parametrize("collection", ["evidence", "modules", "studies"])
def test_duplicate_identifiers_are_rejected(collection):
    payload = _payload()
    payload[collection].append(deepcopy(payload[collection][0]))
    with pytest.raises(ValidationError, match="must be unique"):
        BlueprintDesignRequest.model_validate(payload)


@pytest.mark.parametrize("invalid", ["cycle", "unknown_dependency", "unknown_target", "dimension"])
def test_invalid_module_contracts_are_rejected(invalid):
    payload = _payload()
    if invalid == "cycle":
        payload["modules"][0]["requires_modules"] = ["elimination"]
    elif invalid == "unknown_dependency":
        payload["modules"][-1]["requires_modules"] = ["missing-module"]
    elif invalid == "unknown_target":
        payload["modules"][0]["provides_targets"] = ["undeclared-target"]
    else:
        spec = deepcopy(payload["modules"][-1]["parameter_specs"][0])
        spec["dimension"] = "volume"
        payload["modules"][0]["parameter_specs"] = [spec]
    with pytest.raises(ValidationError):
        BlueprintDesignRequest.model_validate(payload)


def test_compartment_parent_cycle_is_rejected():
    payload = _payload()
    payload["base"]["compartments"][0]["parent_compartment_id"] = "central"
    with pytest.raises(ValidationError, match="cycle"):
        BlueprintDesignRequest.model_validate(payload)


def test_input_permutation_keeps_result_and_candidate_identity():
    payload = _payload()
    payload["studies"].append(_study("selection", "selection"))
    expected = _design(payload)
    for collection in ("evidence", "modules", "studies"):
        payload[collection].reverse()
    assert _design(payload) == expected


def test_semantically_unordered_nested_collections_are_deterministic():
    payload = _payload()
    other = deepcopy(payload["base"]["compartments"][0])
    other.update(compartment_id="tissue", semantic_id="physiology:tissue", name="Fixture tissue")
    payload["base"]["compartments"].append(other)
    payload["modules"][0]["required_compartments"] = ["central", "tissue"]
    payload["modules"][0]["routes"] = ["oral", "intravenous"]
    expected = _design(payload)
    payload["base"]["compartments"].reverse()
    payload["modules"][0]["required_compartments"].reverse()
    payload["modules"][0]["routes"].reverse()
    assert _design(payload) == expected


def test_nearly_equal_evidence_is_resolved_deterministically():
    payload = _payload()
    other = deepcopy(payload["evidence"][1])
    other.update(evidence_id="near-equal", value=1.0000000001)
    payload["evidence"].append(other)
    expected = _design(payload)
    payload["evidence"].reverse()
    assert _design(payload) == expected


def test_budget_limits_and_unexecuted_scientific_status_are_explicit():
    payload = _payload()
    payload["budget"] = {"max_candidates": 2, "max_free_parameters": 0, "max_solver_calls": 1}
    result = _design(payload)
    assert 1 <= result["candidate_count"] <= 2
    assert result["search"]["termination"] == "candidate_budget_exhausted"
    assert result["execution"] == {
        "solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0,
        "compile_status": "not_run", "scientific_status": "not_evaluated",
    }
    for candidate in result["candidates"]:
        assert not candidate["estimated_parameter_ids"]
        assert candidate["identifiability_status"] == candidate["compile_status"] == candidate["simulation_status"] == "not_run"
        assert candidate["scientific_status"] == "not_evaluated"
    assert result["selection_basis"] == "static_contracts_only_not_predictive_performance"


@pytest.mark.parametrize("field,value", [
    ("beam_width", 5), ("max_rounds", 4), ("max_candidates", 25),
    ("max_free_parameters", 7), ("max_solver_calls", 5001),
    ("max_candidates", 0), ("beam_width", True),
])
def test_hard_budget_caps_cannot_be_bypassed(field, value):
    with pytest.raises(ValidationError):
        DesignBudget.model_validate({field: value})


def test_empty_evidence_and_catalog_are_an_explicit_gap_draft():
    payload = _payload()
    payload.update(evidence=[], modules=[])
    result = _design(payload)
    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "draft_with_gaps"
    assert candidate["parameters"] == []
    assert {"TARGET_NOT_MAPPED", "REQUIRED_ROLE_MISSING"} <= _gap_codes(candidate)


def test_mutating_validated_input_does_not_bypass_source_isolation():
    request = BlueprintDesignRequest.model_validate(_payload())
    request.forbidden_source_ids.append(request.evidence[0].provenance.source_id)
    with pytest.raises(ValidationError, match="forbidden or test source"):
        design_blueprint(request)


@pytest.mark.parametrize("constraint", ["clearance_group", "exclusion"])
def test_incompatible_elimination_modules_are_never_combined(constraint):
    payload = _payload()
    original = payload["modules"][-1]
    alternative = deepcopy(original)
    alternative["module_id"] = "alternative_elimination"
    if constraint == "clearance_group":
        original["clearance_group"] = alternative["clearance_group"] = "total-clearance"
    else:
        original["excludes_modules"] = [alternative["module_id"]]
    payload["modules"].append(alternative)
    result = _design(payload)
    assert any(_gap_codes(c) == {"BASE_MODEL_UNBOUND"} for c in result["candidates"])
    assert all(not {"elimination", "alternative_elimination"} <= set(c["module_ids"])
               for c in result["candidates"])


def test_unavoidable_estimation_over_budget_remains_a_gap_not_a_fitted_value():
    payload = _payload()
    payload["evidence"][1]["value"] = None
    payload["budget"] = {"max_free_parameters": 0}
    candidates = _elimination_candidates(_design(payload))
    assert candidates
    for candidate in candidates:
        assert "FREE_PARAMETER_BUDGET_EXCEEDED" in _gap_codes(candidate)
        assert candidate["status"] == "draft_with_gaps"
        assert candidate["parameters"][0]["value"] is None
        assert candidate["parameters"][0]["mode"] == "to_estimate"
        assert candidate["simulation_status"] == "not_run"


@pytest.mark.parametrize("aliased_source", [False, True])
def test_selection_parameter_without_study_link_is_still_blocked(aliased_source):
    payload = _payload()
    payload["studies"].append(_study("selection", "selection"))
    reference = _provenance("selection")
    if aliased_source:
        reference["source_id"] = "selection-source-alias"
    payload["evidence"][1]["provenance"] = reference
    assert "study_id" not in payload["evidence"][1]
    with pytest.raises(ValidationError, match="selection, forbidden or test source"):
        BlueprintDesignRequest.model_validate(payload)


def test_empty_physiology_parameters_cannot_produce_an_assembled_draft():
    payload = _payload()
    payload["base"]["parameters"] = []
    result = _design(payload)
    assert result["candidates"]
    assert all("PHYSIOLOGY_PARAMETERS_NOT_PROVIDED" in _gap_codes(c)
               and c["status"] == "draft_with_gaps" for c in result["candidates"])


def test_blocking_missing_physiology_parameter_stays_an_explicit_gap():
    payload = _payload()
    parameter = payload["base"]["parameters"][0]
    parameter.pop("evidence_id")
    parameter.update(origin="missing", value=None, blocking=True,
                     missing_reason_code="SYNTHETIC_MISSING_INPUT")
    result = _design(payload)
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert {"code": "PHYSIOLOGY_PARAMETER_UNRESOLVED", "module_id": None,
                "parameter_id": "central-volume", "detail": ""} in candidate["gaps"]
        assert candidate["status"] == "draft_with_gaps"


@pytest.mark.parametrize("evidence_formulation,independent,resolved", [
    ("immediate_release", False, True),
    ("extended_release", False, False),
    (None, False, False),
    ("extended_release", True, True),
])
def test_formulation_context_requires_match_or_explicit_transfer_contract(
    evidence_formulation, independent, resolved
):
    payload = _payload()
    payload["requirement"]["formulation"] = "immediate_release"
    payload["evidence"][1]["formulation"] = evidence_formulation
    payload["modules"][-1]["parameter_specs"][0]["formulation_independent"] = independent
    next(m for m in payload["modules"] if m["module_id"] == "oral_input")["formulations"] = ["immediate_release"]
    result = _design(payload)
    candidates = _elimination_candidates(result)
    assert candidates
    modes = {p["mode"] for candidate in candidates for p in candidate["parameters"]}
    assert ("fixed_from_evidence" in modes) is resolved
    assert ("missing" in modes) is not resolved
    if not resolved:
        assert all(p["value"] is None and p["initial_value"] is None
                   for c in candidates for p in c["parameters"])
    assert any(_gap_codes(c) == {"BASE_MODEL_UNBOUND"} for c in result["candidates"]) is resolved


def test_input_module_must_declare_requested_formulation():
    payload = _payload()
    payload["requirement"]["formulation"] = "immediate_release"
    payload["evidence"][1]["formulation"] = "immediate_release"
    next(m for m in payload["modules"] if m["module_id"] == "oral_input")["formulations"] = ["extended_release"]
    result = _design(payload)
    assert result["candidates"]
    assert all("FORMULATION_CONTEXT_UNVERIFIED" in _gap_codes(c)
               and c["status"] == "draft_with_gaps" for c in result["candidates"])


def test_rule_branches_recover_complete_four_module_assembly_after_incompatible_input():
    payload = _payload()
    distribution, oral_input, _, elimination = payload["modules"]
    distribution["provides_targets"] = []
    first_input = deepcopy(oral_input)
    first_input.update(module_id="a_input", excludes_modules=["distribution"])
    oral_input["module_id"] = "z_input"
    output_mapping = deepcopy(distribution)
    output_mapping.update(module_id="output_mapping", role="other",
                          mechanism="output_mapping", provides_targets=["parent_plasma"])
    payload["modules"] = [first_input, oral_input, distribution, elimination, output_mapping]
    # A complete rule seed must survive even with only one final draft allowed.
    payload["budget"] = {"beam_width": 2, "max_rounds": 1, "max_candidates": 1}
    result = _design(payload)
    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "draft_with_gaps"
    assert _gap_codes(candidate) == {"BASE_MODEL_UNBOUND"}
    assert set(candidate["module_ids"]) == {"z_input", "distribution", "elimination", "output_mapping"}
    assert candidate["compile_status"] == candidate["simulation_status"] == "not_run"
