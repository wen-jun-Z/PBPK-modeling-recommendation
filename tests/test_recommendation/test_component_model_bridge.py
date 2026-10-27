from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from uuid import UUID

import pytest

from pbpk_kb.recommendation.component_model_bridge import (
    ComponentModelBridgeError, augment_model_query, validate_component_output,
    _condition_status, _context_audit,
)
from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus, RetrievedSubgraph, assemble_recommendation,
    validate_recommendation,
)
from pbpk_kb.recommendation.schemas import ModelingRequest, RequirementProfile

EVIDENCE = str(UUID(int=101))
COMPOUND = f"Compound:{UUID(int=102)}"
SOURCE_RUN = UUID(int=103)


def _case(*, context=None, components=None, public_model=False):
    """A real assembled/validated graph result, without mocks or GPU generation."""
    request = ModelingRequest(compound="Midazolam", objective="base_pk", species="human",
        route="oral", dose=5, dose_unit="mg", frequency="single dose", population="healthy adults",
        fed_or_fasted="fasted", prediction_endpoints=["AUC", "Cmax"])
    query = {"query_id": "query:test", "compounds": ["midazolam"],
        "text": "Predict oral Midazolam 5 mg in fasted healthy adults; AUC and Cmax.",
        "features": {"species": ["human"], "routes": ["oral"], "populations": ["healthy adults"],
                     "endpoints": ["auc", "cmax"], "enzymes": [], "transporters": [], "process_types": []},
        "dose": 5, "dose_unit": "mg", "food": "fasted"}
    query["variants"] = {"original": {"text": query["text"], "features": deepcopy(query["features"])},
                         "missing_evidence": {"text": "Unchanged perturbation", "features": deepcopy(query["features"])}}
    def node(nid, kind, name, **props):
        return {"node_id": nid, "node_type": kind, "preferred_name": name,
                "properties": props, "standard_identifiers": {}}
    nodes = [node(COMPOUND, "Compound", "Midazolam"),
             node(EVIDENCE, "Evidence", "metabolism:mediated", quality_status="auto_validated"),
             node("event:1", "MetabolismEvent", "Midazolam metabolism"),
             node("enzyme:1", "Enzyme", "CYP3A4"), node("transporter:1", "Transporter", "P-gp"),
             node("organ:1", "Organ", "liver"),
             node("model:private", "PBPKModel", "Private candidate model 77", reusable_candidate=True,
                  model_checksum="frozen", availability_status="repository_model", software="PK-Sim"),
             node("enzyme:via-model", "Enzyme", "CYP2D6")]
    source = "Midazolam metabolism involves CYP3A4 and P-gp in the liver."
    provenance = {"evidence_id": EVIDENCE, "quality_status": "auto_validated", "negation_status": "affirmed",
                  "source_id": "pmc_oa", "source_location": "PMC:test#p1", "supporting_text": source,
                  "source_text_hash": hashlib.sha256(source.encode()).hexdigest(), "confidence": .93}
    pairs = [("event:1", "OF_COMPOUND", COMPOUND), ("event:1", "MEDIATED_BY", "enzyme:1"),
             ("event:1", "TRANSPORTED_BY", "transporter:1"), ("event:1", "LOCATED_IN", "organ:1"),
             (COMPOUND, "DESCRIBED_BY_MODEL", "model:private"), ("model:private", "MEDIATED_BY", "enzyme:via-model")]
    lookup = {n["node_id"]: n for n in nodes}
    edges = [{"edge_id": f"edge:{i}", "subject_id": a, "object_id": b, "predicate": p,
              "subject_type": lookup[a]["node_type"], "object_type": lookup[b]["node_type"],
              "properties": {}, "provenance": deepcopy(provenance)} for i, (a, p, b) in enumerate(pairs)]
    event = {"event_id": "event:1", "event_type": "MetabolismEvent", "evidence_id": EVIDENCE,
             "argument_edge_ids": [e["edge_id"] for e in edges[:4]], "trigger_text": "metabolism",
             "context": context if context is not None else {"species": "human", "route": "oral", "dose": "5 mg",
                          "population": "healthy adults", "fed_or_fasted": "fasted"}}
    subgraph = {"subgraph_id": "sg:1", "hop_limit": 2, "nodes": nodes, "edges": edges,
                "events": [event], "citations": [deepcopy(provenance)], "path_ids": ["path:1"]}
    corpus = GraphPromptCorpus.__new__(GraphPromptCorpus)
    corpus.nodes = nodes; corpus.node_by_id = lookup
    corpus.edges = edges; corpus.edge_by_id = {e["edge_id"]: e for e in edges}
    corpus.events = [event]; corpus.event_by_id = {"event:1": event}
    corpus.compounds = {COMPOUND: lookup[COMPOUND]}; corpus.compound_by_name = {"midazolam": COMPOUND}
    corpus.edges_by_evidence = defaultdict(list, {EVIDENCE: edges}); corpus.subgraphs = {"sg:1": subgraph}
    selection = {"component_ids": ["metabolism", "transport"] if components is None else components,
                 "evidence_ids": [EVIDENCE] if components != [] else [],
                 "public_model_ids": ["model:private"] if public_model else [], "gap_codes": []}
    if public_model:
        selection["component_ids"].append("public_model_template")
    retrieved = [RetrievedSubgraph(subgraph_id="sg:1", hop=2, score=1., score_components={}, row=subgraph, vector=None)]
    recommendation = assemble_recommendation(corpus=corpus, request=request, subgraphs=retrieved, selection=selection)
    validation = validate_component_output(corpus=corpus, request=request, recommendation=recommendation,
                                         allowed_subgraph_ids={"sg:1"})
    assert validation["status"] == "recommendation_validated", validation["errors"]
    prompt = {"query_id": query["query_id"], "compound_id": COMPOUND, "modeling_request": request.model_dump(mode="json"),
              "raw_query_sha256": hashlib.sha256(json.dumps(query, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "allowed_subgraph_ids": ["sg:1"], "retrieved_subgraph_ids": ["sg:1"],
              "allowed_output_ids": {"component_ids": selection["component_ids"], "evidence_ids": selection["evidence_ids"]}}
    return dict(query_record=query, request=request, recommendation=recommendation, corpus=corpus,
                prompt_graph=prompt, source_evidence_run_id=SOURCE_RUN)


def test_verified_mechanisms_are_soft_and_hard_conditions_and_inputs_unchanged():
    case = _case(); before = deepcopy(case["query_record"]); recommendation = deepcopy(case["recommendation"])
    result = augment_model_query(**case); query = result["augmented_query"]
    assert case["query_record"] == before and case["recommendation"] == recommendation
    assert query["text"].startswith(before["text"])
    assert query["features"]["enzymes"] == ["CYP3A4"]
    assert query["features"]["transporters"] == ["P-gp"]
    assert query["features"]["compartments"] == ["liver"]
    assert query["variants"]["original"]["features"] == query["features"]
    assert query["variants"]["missing_evidence"] == before["variants"]["missing_evidence"]
    for name in ("species", "routes", "populations", "endpoints"):
        assert query["features"][name] == before["features"][name]
    assert (query["dose"], query["dose_unit"], query["food"]) == (5, "mg", "fasted")
    assert "optional" in query["text"] and "unlisted mechanisms are unknown" in query["text"]
    profile = RequirementProfile.model_validate(result["requirement_profile"])
    assert all(r.necessity == "preferred" for r in profile.requirements if r.category in {"enzyme", "transporter", "compartment"})
    assert profile.ranking_eligibility == "blocked"  # Legacy evidence gaps retained, not applied.
    assert result["audit"]["profile_eligibility_used_for_ranking"] is False
    assert result["audit"]["candidate_filtering"] is False
    assert augment_model_query(**case) == result


def test_same_name_wrong_compound_id_is_rejected():
    case = _case(); case["recommendation"]["compound"]["compound_id"] = f"Compound:{UUID(int=900)}"
    with pytest.raises(ComponentModelBridgeError, match="compound identity"):
        augment_model_query(**case)


@pytest.mark.parametrize("field,value", [("dose", 10), ("route", "intravenous"), ("fed_or_fasted", "fed"), ("objective", "food_effect")])
def test_different_original_request_cannot_be_reused(field, value):
    case = _case(); case["recommendation"]["resolved_modeling_request"][field] = value
    with pytest.raises(ComponentModelBridgeError, match="conditions differ"):
        augment_model_query(**case)


@pytest.mark.parametrize("field,value", [("supporting_text", "Invented supporting text"),
    ("source_text_hash", "0" * 64), ("evidence_id", str(UUID(int=901)))])
def test_unknown_or_tampered_citation_is_rejected(field, value):
    case = _case(); case["recommendation"]["citations"][0][field] = value
    with pytest.raises(ComponentModelBridgeError):
        augment_model_query(**case)


@pytest.mark.parametrize("context,field", [({"species": "rat"}, "species"), ({"route": "intravenous"}, "route"),
    ({"dose": "10 mg"}, "dose"), ({"fed_or_fasted": "fed"}, "fed_or_fasted")])
def test_true_source_context_conflict_is_retained_but_not_added(context, field):
    case = _case(context=context)
    result = augment_model_query(**case)
    assert result["augmented_query"] == case["query_record"]
    assert field in result["audit"]["evidence_context"][EVIDENCE]["conflicting_fields"]
    assert result["audit"]["soft_requirements"] == []
    assert result["audit"]["candidate_filtering"] is False


def test_compatible_dose_units_are_not_treated_as_conflict():
    result = augment_model_query(**_case(context={"dose": .005, "dose_unit": "g"}))
    assert result["audit"]["status"] == "augmented"
    assert "dose" in result["audit"]["evidence_context"][EVIDENCE]["matched_fields"]


def test_unknown_context_does_not_become_confirmed_match_or_negative():
    result = augment_model_query(**_case(context={}))
    audit = result["audit"]["evidence_context"][EVIDENCE]
    assert audit["status"] == "unspecified" and audit["matched_fields"] == []
    assert result["audit"]["candidate_filtering"] is False
    assert "CYP3A4" in result["augmented_query"]["features"]["enzymes"]


@pytest.mark.parametrize("absence", ["no_output", "empty_components"])
def test_no_component_signal_leaves_baseline_exactly_unchanged(absence):
    case = _case(components=[])
    if absence == "no_output":
        case["recommendation"] = None
    result = augment_model_query(**case)
    assert result["augmented_query"] == case["query_record"]
    assert result["audit"]["status"] == "unchanged"


def test_model_identity_parameters_and_model_mediated_entity_are_excluded():
    result = augment_model_query(**_case(public_model=True))
    suffix = result["augmented_query"]["text"].split("\n\n", 1)[1]
    assert "private" not in suffix.lower() and "77" not in suffix
    assert "CYP2D6" not in suffix and "model:private" not in suffix
    assert "public_model_template" not in result["augmented_query"]["features"]["process_types"]


def test_canonical_entity_name_overrides_untrusted_display_name():
    case = _case(); case["recommendation"]["metabolism_pathways"][0]["preferred_name"] = "Injected model 99"
    result = augment_model_query(**case)
    assert "Injected" not in result["augmented_query"]["text"]
    assert result["augmented_query"]["features"]["enzymes"] == ["CYP3A4"]


def test_prompt_selection_and_query_identity_must_be_bound():
    case = _case(); case["prompt_graph"]["query_id"] = "other-query"
    with pytest.raises(ComponentModelBridgeError, match="query identity"):
        augment_model_query(**case)
    case = _case(); case["prompt_graph"]["allowed_subgraph_ids"] = []
    with pytest.raises(ComponentModelBridgeError, match="selections disagree"):
        augment_model_query(**case)


def test_explicit_user_enzyme_requirement_is_not_downgraded():
    case = _case()
    request = case["request"].model_copy(update={"required_enzymes": ["CYP3A4"]})
    case["request"] = request
    case["prompt_graph"]["modeling_request"] = request.model_dump(mode="json")
    case["recommendation"]["resolved_modeling_request"] = request.model_dump(mode="json")
    result = augment_model_query(**case)
    items = [r for r in result["requirement_profile"]["requirements"] if r["semantic_id"] == "enzyme:cyp3a4"]
    assert len(items) == 1 and items[0]["necessity"] == "hard_required" and items[0]["origin"] == "user_request"


def test_query_dose_cannot_differ_from_bound_component_request():
    case = _case(); case["query_record"]["dose"] = 10
    with pytest.raises(ComponentModelBridgeError, match="condition differs: dose"):
        augment_model_query(**case)


def test_empty_parameter_compatibility_is_narrow_and_records_original_error():
    case = _case()
    original = validate_recommendation(corpus=case["corpus"], request=case["request"],
        recommendation=case["recommendation"], allowed_subgraph_ids={"sg:1"})
    assert original["errors"] == ["parameter_candidates_not_exactly_graph_derived"]
    result = augment_model_query(**case)
    assert result["audit"]["validation"]["compatibility"]["original_errors"] == original["errors"]
    assert result["audit"]["validation"]["status"] == "recommendation_validated"


@pytest.mark.parametrize("value", [None, [{"parameter": "clearance", "value": 42, "evidence_id": EVIDENCE}]])
def test_missing_or_nonempty_fabricated_parameters_cannot_use_empty_compatibility(value):
    case = _case(); case["recommendation"]["parameter_candidates"] = value
    with pytest.raises(ComponentModelBridgeError, match="validation failed"):
        augment_model_query(**case)


def test_unrecognized_validator_source_cannot_use_empty_compatibility(monkeypatch):
    case = _case()
    monkeypatch.setattr("pbpk_kb.recommendation.component_model_bridge._legacy_empty_list_bug_bound", lambda: False)
    with pytest.raises(ComponentModelBridgeError, match="validation failed"):
        augment_model_query(**case)


def test_unspecified_request_species_remains_unknown_with_human_evidence():
    case = _case()
    request = case["request"].model_copy(update={"species": "unspecified"})
    case["request"] = request
    case["prompt_graph"]["modeling_request"] = request.model_dump(mode="json")
    case["recommendation"]["resolved_modeling_request"] = request.model_dump(mode="json")
    result = augment_model_query(**case)
    audit = result["audit"]["evidence_context"][EVIDENCE]
    assert "species" in audit["unspecified_fields"] and not audit["conflicting_fields"]
    assert result["audit"]["status"] == "augmented"


def test_declared_multivalue_species_retains_set_semantics():
    case = _case()
    request = case["request"].model_copy(update={"species": "human, rat"})
    case["request"] = request
    case["prompt_graph"]["modeling_request"] = request.model_dump(mode="json")
    case["recommendation"]["resolved_modeling_request"] = request.model_dump(mode="json")
    case["query_record"]["features"]["species"] = ["human", "rat"]
    case["prompt_graph"]["raw_query_sha256"] = hashlib.sha256(json.dumps(case["query_record"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = augment_model_query(**case)
    assert result["augmented_query"]["features"]["species"] == ["human", "rat"]
    assert "species" in result["audit"]["evidence_context"][EVIDENCE]["matched_fields"]


def test_changed_clinical_dose_in_text_cannot_bypass_prompt_hash_binding():
    case = _case(); case["query_record"].pop("dose"); case["query_record"].pop("dose_unit")
    case["query_record"]["text"] += '\nAdministration: {"dose":{"value":5,"unit":"mg"}}'
    case["prompt_graph"]["raw_query_sha256"] = hashlib.sha256(json.dumps(case["query_record"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    case["query_record"]["text"] = case["query_record"]["text"].replace('"value":5', '"value":50')
    with pytest.raises(ComponentModelBridgeError, match="original query hash"):
        augment_model_query(**case)


def test_absent_upstream_drug_with_no_component_output_preserves_baseline():
    case = _case(); case["recommendation"] = None
    case["corpus"].compounds.clear(); case["corpus"].compound_by_name.clear()
    result = augment_model_query(**case)
    assert result["augmented_query"] == case["query_record"]
    assert result["requirement_profile"] is None
    assert result["audit"]["candidate_filtering"] is False


@pytest.mark.parametrize("wanted,actual,expected", [
    ("geriatric", "eight cigarette smoking male volunteers", "unspecified"),
    ("geriatric", "healthy adults", "unspecified"),
    ("geriatric", "pediatric patients", "conflicting"),
    ('{"healthy":true,"sex":"male"}', "patients with renal impairment", "conflicting"),
    ({"healthy": True, "population": "healthy_adult"}, "renal impairment patients", "conflicting"),
    ('{"healthy":true,"sex":"male","weight_mean_kg":65.96}', "healthy adults", "unspecified"),
    ('{"sex":"male"}', "healthy women", "conflicting"),
    ('{"sex":"male_and_female"}', "healthy women", "unspecified"),
    ('{"age_mean_years":35}', "healthy adults", "unspecified"),
    ('{"healthy":true}', "healthy status not reported", "unspecified"),
    ('{"healthy":true}', "patients not included", "unspecified"),
    ("geriatric", "children excluded", "unspecified"),
    ("healthy adults", "healthy adults", "matched"),
    ('{"healthy":', "patients with renal impairment", "unspecified"),
])
def test_population_compares_only_explicit_attributes(wanted, actual, expected):
    before = deepcopy((wanted, actual))
    assert _condition_status(wanted, actual, "population") == expected
    assert (wanted, actual) == before


@pytest.mark.parametrize("wanted,actual,expected", [
    ("Crestor tablet", "tablet", "unspecified"),
    ("Two0.25mg Sanofi Digoxin tablets, lotBHG0746", "tablet", "unspecified"),
    ("Losec20mg enteric-coated capsule, AstraZeneca, lot06BN020", "enteric-coated capsule", "unspecified"),
    ("reference tablet, lot SR", "immediate-release tablet", "unspecified"),
    ("Wellbutrin SR sustained-release tablet", "immediate-release tablet", "conflicting"),
    ("extended-release tablet", "immediate-release tablet", "conflicting"),
    ("enteric-coated capsule", "immediate-release capsule", "conflicting"),
    ("Crestor tablet", "oral solution", "conflicting"),
    ("tablets", "tablet", "matched"),
    ("brand alpha", "brand beta", "unspecified"),
    ("immediate-release tablet", "sustained-release status not reported", "unspecified"),
    ({"dosage_form": "tablet", "release_type": "sustained-release"}, "immediate-release tablet", "conflicting"),
])
def test_formulation_brand_batch_and_partial_description_do_not_imply_conflict(wanted, actual, expected):
    before = deepcopy((wanted, actual))
    assert _condition_status(wanted, actual, "formulation") == expected
    assert (wanted, actual) == before


def test_species_explicit_in_population_still_blocks_cross_species_evidence():
    case = _case(context={"population": "lactating goats"})
    result = augment_model_query(**case)
    assert result["augmented_query"] == case["query_record"]
    assert "species" in result["audit"]["evidence_context"][EVIDENCE]["conflicting_fields"]
    assert result["audit"]["soft_requirements"] == []


def test_unknown_age_allows_soft_evidence_without_claiming_population_match():
    request = ModelingRequest(compound="theophylline", species="human", population="geriatric")
    original = request.model_dump(mode="json")
    context = {"population": "eight cigarette smoking male volunteers"}
    audit = _context_audit(request, [context])
    assert audit["status"] == "unspecified" and audit["conflicting_fields"] == []
    assert "population" in audit["unspecified_fields"]
    assert request.model_dump(mode="json") == original
    assert context == {"population": "eight cigarette smoking male volunteers"}


def test_structured_healthy_request_retains_patient_conflict_through_full_bridge():
    case = _case(context={"population": "patients with renal impairment"})
    request = case["request"].model_copy(update={"population": '{"healthy":true,"sex":"male"}'})
    case["request"] = request
    case["prompt_graph"]["modeling_request"] = request.model_dump(mode="json")
    case["recommendation"]["resolved_modeling_request"] = request.model_dump(mode="json")
    original = deepcopy(case["query_record"]), request.model_dump(mode="json")
    result = augment_model_query(**case)
    assert result["augmented_query"] == original[0]
    assert request.model_dump(mode="json") == original[1]
    assert "population" in result["audit"]["evidence_context"][EVIDENCE]["conflicting_fields"]
    assert result["audit"]["soft_requirements"] == []
