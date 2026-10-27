"""CPU contracts for evidence-specific soft context admission, without models."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_component_model_integration as generate
import prepare_component_model_integration as prepare
from pbpk_kb.recommendation.graph_prompt import GraphPromptCorpus
from pbpk_kb.recommendation.online_retrieval import _hard_graph_compatibility
from pbpk_kb.recommendation.schemas import ModelingRequest


def _case(*contexts, population="healthy adults"):
    """Exercise the real graph-path, objective and source-context functions."""
    compound = "Compound:caffeine"
    node = {"node_id": compound, "node_type": "Compound", "preferred_name": "Caffeine"}
    request = ModelingRequest(compound="caffeine", species="human", route="oral",
                              dose=50, dose_unit="mg", population=population)
    edges, events, citations = [], [], []
    for index, context in enumerate(contexts):
        evidence_id, event_id = f"evidence:{index}", f"event:{index}"
        provenance = {"evidence_id": evidence_id, "quality_status": "auto_validated",
                      "negation_status": "affirmed"}
        edges.append({"edge_id": f"edge:{index}", "subject_id": event_id,
                      "object_id": compound, "predicate": "OF_COMPOUND", "provenance": provenance})
        events.append({"event_id": event_id, "evidence_id": evidence_id,
                       "context": deepcopy(context)})
        citations.append(deepcopy(provenance))
    corpus = GraphPromptCorpus.__new__(GraphPromptCorpus)
    corpus.compounds = {compound: node}
    corpus.compound_by_name = {"caffeine": compound}
    corpus.nodes = [node]
    corpus.node_by_id = {compound: node}
    corpus.edges = edges
    corpus.edge_by_id = {edge["edge_id"]: edge for edge in edges}
    corpus.events = events
    corpus.event_by_id = {event["event_id"]: event for event in events}
    corpus.edges_by_evidence = defaultdict(list)
    for edge in edges:
        corpus.edges_by_evidence[edge["provenance"]["evidence_id"]].append(edge)
    subgraph = {"subgraph_id": "subgraph:context-test", "center_id": compound,
                "hop_limit": 2, "nodes": [node], "edges": edges, "events": events,
                "citations": citations, "embedding_text": "Caffeine mechanism; human oral"}
    corpus.subgraphs = {subgraph["subgraph_id"]: subgraph}
    return corpus, request, subgraph


@pytest.mark.parametrize("context,population", [
    ({}, "healthy adults"),
    ({"species": "human", "route": "oral"}, '{"healthy":true,"n":18}'),
    ({"population": {"healthy": True, "n": 18}}, "healthy adults"),
    ({"population": '{"healthy":true,"n":18}'}, "healthy adults"),
])
def test_missing_or_structured_context_is_unknown_not_a_conflict(context, population):
    corpus, request, subgraph = _case(context, population=population)
    before = deepcopy((request.model_dump(), subgraph, corpus.events))
    audit = generate.component_context_audit(corpus, request, subgraph)
    assert generate.soft_graph_compatibility(corpus, request, subgraph) == (True, [])
    assert audit["eligible_evidence_ids"] == ["evidence:0"]
    assert audit["evidence_context"]["evidence:0"]["status"] == "unspecified"
    assert not audit["evidence_context"]["evidence:0"]["conflicting_fields"]
    assert (request.model_dump(), subgraph, corpus.events) == before


@pytest.mark.parametrize("context,field", [
    ({"route": "intravenous"}, "route"),
    ({"species": "rat"}, "species"),
    ({"dose": 250, "dose_unit": "mg"}, "dose"),
    ({"dose": 50, "dose_unit": "mg/kg"}, "dose"),
])
def test_explicit_conflicts_are_rejected_even_when_document_mentions_request(context, field):
    corpus, request, subgraph = _case(context)
    subgraph["embedding_text"] += " healthy adults caffeine 50 mg"
    audit = generate.component_context_audit(corpus, request, subgraph)
    assert not audit["pass"]
    assert audit["errors"] == ["all_objective_evidence_explicitly_conflicting"]
    assert field in audit["evidence_context"]["evidence:0"]["conflicting_fields"]
    assert audit["eligible_evidence_ids"] == []
    assert generate.soft_graph_compatibility(corpus, request, subgraph) == (False, audit["errors"])


def test_equivalent_units_and_route_species_aliases_remain_eligible():
    corpus, request, subgraph = _case({"species": "homo sapiens", "route": "p.o.",
                                     "dose": .05, "dose_unit": "g", "population": "healthy adults"})
    audit = generate.component_context_audit(corpus, request, subgraph)
    assert audit["pass"]
    assert audit["evidence_context"]["evidence:0"]["status"] == "matched"
    assert set(audit["evidence_context"]["evidence:0"]["matched_fields"]) == {
        "species", "route", "dose", "population"}


def test_one_eligible_objective_evidence_is_enough_but_conflict_is_retained():
    corpus, request, subgraph = _case({"route": "intravenous"}, {})
    audit = generate.component_context_audit(corpus, request, subgraph)
    assert audit["pass"] and audit["errors"] == []
    assert audit["eligible_evidence_ids"] == ["evidence:1"]
    assert audit["evidence_context"]["evidence:0"]["status"] == "conflicting"
    assert audit["evidence_context"]["evidence:1"]["status"] == "unspecified"


def test_wrong_drug_path_remains_rejected_despite_eligible_citation():
    corpus, request, subgraph = _case({})
    subgraph["center_id"] = "Compound:other"
    subgraph["edges"] = []
    audit = generate.component_context_audit(corpus, request, subgraph)
    assert not audit["pass"]
    assert "target_compound_path_absent" in audit["errors"]
    assert audit["eligible_evidence_ids"] == ["evidence:0"]


def test_non_target_center_with_actual_target_path_is_eligible():
    corpus, request, subgraph = _case({})
    subgraph["center_id"] = "event:0"
    assert generate.soft_graph_compatibility(corpus, request, subgraph) == (True, [])


def test_same_drug_without_requested_objective_evidence_is_rejected():
    corpus, request, subgraph = _case({})
    request = request.model_copy(update={"objective": "public_model_reuse"})
    audit = generate.component_context_audit(corpus, request, subgraph)
    assert not audit["pass"]
    assert audit["errors"] == ["objective_evidence_absent"]
    assert audit["evidence_context"] == {} and audit["eligible_evidence_ids"] == []


def test_unvalidated_citation_does_not_supply_objective_support():
    corpus, request, subgraph = _case({})
    subgraph["citations"][0]["quality_status"] = "unverified"
    assert generate.soft_graph_compatibility(corpus, request, subgraph) == (
        False, ["objective_evidence_absent"])


@pytest.fixture(scope="module")
def frozen_corpus_and_entries():
    return generate.corpus_for(prepare.OUT), prepare.rows(prepare.OUT / "inputs/requests.jsonl")


def test_real_caffeine_50mg_corrects_false_absence_without_changing_frozen_inputs(frozen_corpus_and_entries):
    corpus, entries = frozen_corpus_and_entries
    protocol = prepare.read(prepare.OUT / "protocol.json")
    registered = protocol["prepared_input_artifacts"]
    before = {ref["path"]: prepare.sha(ref["path"]) for ref in registered}
    assert all(before[ref["path"]] == ref["sha256"] for ref in registered)
    entry = next(row for row in entries if row["query_id"] == "clinical:caffeine_sherman2022_solution_50mg")
    request = ModelingRequest.model_validate(entry["modeling_request"])
    assert request.dose == 50 and request.population.startswith("{")
    corrected = []
    for subgraph in corpus.subgraphs.values():
        old_pass, old_errors = _hard_graph_compatibility(corpus, request, subgraph)
        audit = generate.component_context_audit(corpus, request, subgraph)
        if audit["pass"]:
            assert not old_pass
            assert "population_context_absent" in old_errors
            assert audit["eligible_evidence_ids"]
            assert all(audit["evidence_context"][eid]["status"] != "conflicting"
                       for eid in audit["eligible_evidence_ids"])
            corrected.append(subgraph["subgraph_id"])
    assert corrected, "At least one source-supported caffeine subgraph must survive JSON-population handling"
    assert {ref["path"]: prepare.sha(ref["path"]) for ref in registered} == before
    assert entry["request_sha256"] == prepare.digest(request.model_dump(mode="json"))


def test_real_bupropion_does_not_gain_missing_objective_evidence(frozen_corpus_and_entries):
    corpus, entries = frozen_corpus_and_entries
    entry = next(row for row in entries if row["query_id"] == "clinical:bupropion_teaima2020_wellbutrin_sr_150mg")
    request = ModelingRequest.model_validate(entry["modeling_request"])
    compound_id = corpus.resolve_compound(request.compound)["node_id"]
    centered = [row for row in corpus.subgraphs.values() if row.get("center_id") == compound_id]
    assert centered
    for subgraph in centered:
        audit = generate.component_context_audit(corpus, request, subgraph)
        assert not audit["pass"]
        assert "objective_evidence_absent" in audit["errors"]
        assert not audit["eligible_evidence_ids"]
    assert not any(generate.soft_graph_compatibility(corpus, request, sg)[0]
                   for sg in corpus.subgraphs.values())
