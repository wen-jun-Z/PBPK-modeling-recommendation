"""Scientific constraints and fair, deterministic parameter grouping."""
from copy import deepcopy

import pytest

from pbpk_kb.blueprint_policy import condition_matched_selection as policy


def context(**changes):
    return {"compound_id":"pubchem:example","case_id":"example_iv","species":"human","matrix":"plasma",
        "analyte":"parent_example","population":"healthy_adult","population_basis":"reference_assumption",
        "route":"intravenous","target_study":"target","excluded_study_ids":["development"],
        "weight_kg":80.,**changes}


def candidate(identifier,parameter,value,study="source",arm="a",**changes):
    definitions = {"total_clearance":"systemic_total_plasma_clearance",
        "renal_clearance":"net_renal_plasma_clearance","fraction_unbound":"fraction_unbound_in_plasma"}
    record = {"candidate_id":identifier,"compound_id":"pubchem:example","case_id":"example_iv",
        "parameter":parameter,"value":value,"unit":"fraction" if parameter=="fraction_unbound" else "mL/min/kg",
        "basis":"measured","quality_score":3,
        "source":{"path":"local/source.pdf","sha256":"a"*64,"location":"Table 1"},
        "context":{"species":"human","matrix":"plasma","analyte":"parent_example",
            "definition":definitions[parameter],"study_id":study,"arm_id":arm,
            "population":"healthy_adult","route":"intravenous","weight_kg":50.},
        "independence":{"source_study":study,"target_study":"target","target_values_used":False}}
    for key,value in changes.items():
        if key=="context":
            record[key].update(value)
        else:
            record[key]=value
    return record


def pairs():
    return [candidate("cl_a","total_clearance",5.,arm="a"),candidate("fu_a","fraction_unbound",.1,arm="a"),
            candidate("cl_b","total_clearance",4.,arm="b"),candidate("fu_b","fraction_unbound",.2,arm="b")]


def test_joint_keeps_within_arm_groups_but_unconstrained_does_not_claim_pairing():
    joint = policy.select_bundles(pairs(),context(),method="joint")
    loose = policy.select_bundles(pairs(),context(),method="unconstrained_search")
    assert joint["valid_bundle_count"]==2 and loose["valid_bundle_count"]==4
    assert len(joint["bundle_rejections"])==2
    assert all("within_study_arm_mismatch" in r["reasons"] for r in joint["bundle_rejections"])
    assert all(b["compatibility"]["same_study_arm_pairs"]==1 for b in joint["bundles"])
    assert all(b["compatibility"]["study_arm_gate_applied"] is False for b in loose["bundles"])


def test_input_permutations_do_not_change_selection_and_inputs_are_not_mutated():
    records, request = pairs(),context()
    saved = deepcopy((records,request))
    for method in policy.METHODS:
        assert policy.select_bundles(records,request,method)==policy.select_bundles(list(reversed(records)),request,method)
    assert (records,request)==saved


def test_independent_blocks_equal_priority_conflict_instead_of_arbitrary_first_pick():
    out = policy.select_bundles(pairs(),context(),method="independent")
    assert out["status"]=="blocked_conflict" and out["bundles"]==[]
    assert out["reasons"]==["equal_priority_conflict:fraction_unbound","equal_priority_conflict:total_clearance"]


def test_common_population_gate_is_identical_for_all_methods():
    records = pairs()[:2]+[candidate("sick","total_clearance",1.,context={"population":"hepatic_impairment_moderate"})]
    results = [policy.select_bundles(records,context(),method) for method in sorted(policy.METHODS)]
    for result in results:
        assert result["common_context_rejection_count"]==1
        assert result["common_context_rejections"]==[{"candidate_id":"sick","stage":"common_context","reasons":["population_mismatch"]}]
        assert {r["candidate_id"] for r in result["eligible"]}=={"cl_a","fu_a"}


@pytest.mark.parametrize("missing_field",["population","definition"])
def test_missing_population_or_definition_never_becomes_healthy(missing_field):
    records = pairs()[:2]
    records[0]["context"].pop(missing_field)
    for method in policy.METHODS:
        result = policy.select_bundles(records,context(),method)
        assert result["status"]=="needs_evidence"
        assert result["rejected"][0]["candidate_id"]=="cl_a"


@pytest.mark.parametrize("definition",["CL/F","apparent_oral_clearance","CLint"])
def test_apparent_or_intrinsic_clearance_never_substitutes_for_systemic_cl(definition):
    records = pairs()[:2]
    records[0]["context"]["definition"]=definition
    for method in policy.METHODS:
        result = policy.select_bundles(records,context(),method)
        assert result["rejected"][0]["reasons"]==["parameter_definition_missing_or_incompatible"]


def test_absolute_clearance_uses_source_weight_not_target_weight():
    records = [candidate("cl","total_clearance",30.,unit="L/h"),pairs()[1]]
    result = policy.select_bundles(records,context())
    record = next(r for r in result["bundles"][0]["records"] if r["parameter"]=="total_clearance")
    assert record["value"]==pytest.approx(10.) and record["unit"]=="mL/min/kg"
    assert record["normalization"]["source_weight_kg"]==50.
    assert record["normalization"]["target_weight_used"] is False
    records[0]["context"].pop("weight_kg")
    assert policy.select_bundles(records,context())["rejected"][0]["reasons"]==["source_weight_required_for_absolute_clearance"]


def test_target_or_development_study_and_wrong_drug_are_rejected_for_every_method():
    bad = [candidate("target","total_clearance",5.,study="target"),
        candidate("dev","total_clearance",5.,study="development"),
        candidate("other","total_clearance",5.,compound_id="pubchem:other")]
    for method in policy.METHODS:
        result = policy.select_bundles(pairs()[:2]+bad,context(),method)
        assert {r["candidate_id"] for r in result["rejected"]}=={"target","dev","other"}


def test_renal_total_pair_requires_same_arm_but_physical_constraint_applies_to_all():
    records = pairs()[:2]+[candidate("renal","renal_clearance",2.,study="another")]
    assert policy.select_bundles(records,context(),"joint")["status"]=="blocked_incompatible"
    assert policy.select_bundles(records,context(),"unconstrained_search")["status"]=="ready"
    records[-1]["value"]=6.
    for method in policy.METHODS:
        result = policy.select_bundles(records,context(),method)
        assert result["status"]=="blocked_incompatible"
        assert "renal_clearance_exceeds_total_clearance" in result["bundle_rejections"][0]["reasons"]


def test_reported_parent_urinary_clearance_alias_is_a_valid_net_renal_definition():
    records = pairs()[:2]+[candidate("renal","renal_clearance",2.,context={"definition":"renal_parent_plasma_clearance"})]
    result = policy.select_bundles(records,context())
    assert result["status"]=="ready" and len(result["bundles"])==1
    assert result["bundles"][0]["selected"]["renal_clearance"]=="renal"


def test_cross_paper_fu_supplement_and_oral_fu_are_allowed_without_claiming_same_arm():
    records = [pairs()[0],candidate("fu_other","fraction_unbound",.1,study="other",context={"route":"oral"})]
    result = policy.select_bundles(records,context())
    assert result["status"]=="ready"
    bundle = result["bundles"][0]
    assert bundle["compatibility"]["same_study_arm_pairs"]==0
    assert "compatible_cross_study_supplement" in bundle["compatibility"]["reasons"]
    assert result["observations_used"] is False and result["accuracy_optimized"] is False


def test_concentration_dependent_fu_must_cover_the_requested_concentration_range():
    records = pairs()[:2]
    records[1]["context"]["binding"]={"concentration_dependent":True,
        "concentration_range":{"lower":0.,"upper":1.,"unit":"mg/L"}}
    request = context(concentration_range={"lower":0.,"upper":2.,"unit":"mg/L"})
    for method in policy.METHODS:
        result = policy.select_bundles(records,request,method)
        assert result["common_context_rejections"][0]["reasons"]==["request_outside_fu_concentration_range"]
    request["concentration_range"]["upper"] = 1.
    assert policy.select_bundles(records,request)["status"]=="ready"


def test_same_arm_pair_preferred_and_limit_does_not_hide_valid_candidate_count():
    records = pairs()[:2]+[candidate("fu_other","fraction_unbound",.1,study="other",quality_score=4)]
    result = policy.select_bundles(records,context(),limit=1)
    assert result["valid_bundle_count"]==2 and result["limit_truncated"] is True
    assert result["bundles"][0]["selected"]["fraction_unbound"]=="fu_a"


def test_explicit_group_preserved_without_banning_other_cross_study_supplements():
    records = pairs()[:2]
    for record in records:
        record.update(coupling_group="source:a",grouped_parameters=["total_clearance","fraction_unbound"],
            pairing_level="group_summary")
    records += [candidate("cl_pool","total_clearance",6.,study="pool1"),
                candidate("fu_pool","fraction_unbound",.1,study="pool2")]
    joint = policy.select_bundles(records,context(),"joint")
    unconstrained = policy.select_bundles(records,context(),"unconstrained_search")
    assert joint["valid_bundle_count"]==3 and unconstrained["valid_bundle_count"]==4
    assert any("preserve_explicit_cl_fu_group" in r["reasons"] for r in joint["bundle_rejections"])
    assert any(b["selected"]=={"total_clearance":"cl_pool","fraction_unbound":"fu_a"} for b in joint["bundles"])
    assert "group_summary_pairing_not_individual_correlation" in joint["bundles"][0]["compatibility"]["reasons"]


def test_group_without_an_eligible_fu_allows_cross_study_supplement():
    records = [pairs()[0],candidate("fu_other","fraction_unbound",.1,study="other")]
    records[0].update(coupling_group="source:a",grouped_parameters=["total_clearance","fraction_unbound"],
        pairing_level="group_summary")
    assert policy.select_bundles(records,context())["status"]=="ready"


def test_same_normalized_value_is_not_an_independent_conflict():
    records = pairs()[:2]+[candidate("cl_same","total_clearance",.3,unit="L/h/kg")]
    assert policy.select_bundles(records,context(),"independent")["status"]=="ready"


def test_explicit_population_assumption_is_preserved_not_upgraded_to_observation():
    result = policy.select_bundles(pairs()[:2],context())
    assert result["population_basis"]=="reference_assumption"
    with pytest.raises(ValueError,match="Unknown population"):
        policy.select_bundles(pairs()[:2],context(population="unknown"))


def test_unknown_source_organ_function_is_not_assumed_normal():
    result = policy.select_bundles(pairs()[:2],context(conditions={"hepatic_function":"normal"}))
    assert result["status"]=="needs_evidence"
    assert all("source_condition_unknown:hepatic_function" in r["reasons"] for r in result["common_context_rejections"])
