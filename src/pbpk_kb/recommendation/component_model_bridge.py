"""Validate component evidence and append soft PBPK-Match query preferences.

This adapter does not generate, train, filter candidates, or grant model reuse.
The legacy requirement profile is retained for provenance, including its gaps;
its ranking eligibility is deliberately not applied to the learned ranking.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus, _components_for_evidence, _quality_ok, _parameter_candidate_derivation,
    validate_recommendation,
)
from pbpk_kb.recommendation.requirement_profile import build_requirement_profile
from pbpk_kb.recommendation.requirement_profile_sources import COMPOUND_NAMESPACE
from pbpk_kb.recommendation.schemas import (
    Citation, ComponentRecommendation, CompoundRef, ModelingRequest,
    RecommendationResponse, RequirementContext, RequirementItem,
    RequirementProfile, requirement_profile_checksum,
)

BRIDGE_VERSION = "pbpk-component-model-soft-bridge-1"
NAMESPACE = UUID("c7e7f44b-18c9-43a0-824c-1502f4e11fc0")
SOFT_FIELDS = ("process_types", "enzymes", "transporters", "compartments")
PROCESSES = frozenset({"absorption", "distribution", "metabolism", "transport", "excretion", "ddi"})
_EXCLUDED_NODES = frozenset({"PBPKModel", "SourceDocument", "Evidence", "ParameterValue", "PKParameter", "Unit", "Regimen"})
_EXCLUDED_EDGES = frozenset({"CONTRADICTS", "DERIVED_FROM", "REPORTS", "SUPPORTED_BY", "DESCRIBED_BY_MODEL"})
_LEGACY_VALIDATOR_SHA256 = "fb8bc1ceae146128cc8f8c02129e6b40f0955b6bfa812ead8226197369a30548"


class ComponentModelBridgeError(ValueError):
    """The component output cannot be bound to the requested ranking input."""


@lru_cache(maxsize=1)
def _legacy_empty_list_bug_bound() -> bool:
    return hashlib.sha256(Path(__file__).with_name("graph_prompt.py").read_bytes()).hexdigest() == _LEGACY_VALIDATOR_SHA256


def validate_component_result(*, corpus: GraphPromptCorpus, request: ModelingRequest,
                              recommendation: dict, allowed_subgraph_ids: set[str]) -> dict:
    """Run the original validator with one source-bound empty-sequence fix.

    Its frozen implementation compares ``actual or ()`` with a derived list.
    Both empty is therefore rejected incorrectly. No other error is ignored,
    and nonempty, missing, or malformed parameter output cannot use this fix.
    """
    validation = validate_recommendation(corpus=corpus, request=request,
        recommendation=recommendation, allowed_subgraph_ids=allowed_subgraph_ids)
    if (validation.get("errors") == ["parameter_candidates_not_exactly_graph_derived"]
            and recommendation.get("parameter_candidates") == [] and _legacy_empty_list_bug_bound()):
        compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
        evidence_ids = {str(c["evidence_id"]) for c in recommendation.get("citations") or []}
        expected, _ = _parameter_candidate_derivation(corpus, evidence_ids, compound_id, request)
        if expected == []:
            validation = {**validation, "status": "recommendation_validated", "error_count": 0, "errors": [],
                "compatibility": {"code": "equal_empty_parameter_lists", "source_sha256": _LEGACY_VALIDATOR_SHA256,
                                  "original_status": validation["status"], "original_errors": validation["errors"]}}
    return validation


# Early integration clients used this spelling; both execute the same checks.
validate_component_output = validate_component_result


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def _canonical(value: Any, field: str) -> str:
    value = _norm(value)
    aliases = {
        "species": {"humans": "human", "homo sapiens": "human", "mice": "mouse", "rats": "rat",
                    "goats": "goat", "rabbits": "rabbit", "dogs": "dog", "pigs": "pig"},
        "route": {"po": "oral", "p.o.": "oral", "iv": "intravenous", "i.v.": "intravenous", "oral administration": "oral"},
        "fed_or_fasted": {"fasting": "fasted", "fast": "fasted", "with food": "fed", "without food": "fasted"},
        "formulation": {"tablets": "tablet", "capsules": "capsule"},
    }
    return aliases.get(field, {}).get(value, value)


def _condition_values(value: Any, field: str) -> set[str]:
    """Preserve declared sets; an unspecified request is not a literal species."""
    if value is None:
        return set()
    if isinstance(value, dict) or (isinstance(value, str) and value.strip().startswith("{")):
        # A structured population contract cannot be flattened to one label and
        # then claimed fully matched. Keep it unknown in this soft bridge.
        return set()
    values = value if isinstance(value, (list, tuple)) else [value]
    result = set()
    for item in values:
        for part in re.split(r"\s*[,;|]\s*", str(item)):
            normalized = _canonical(part, field)
            if normalized not in {"", "none", "unknown", "unspecified", "not specified", "not reported", "not_applicable"}:
                result.add(normalized)
    return result


def _descriptive_attributes(value: Any, field: str) -> dict[str, set[str]]:
    """Read a few explicit axes; brand names and unparsed demographics stay unknown.

    This is a conflict screen for soft evidence, not an equivalence test for
    formulations or clinical populations. No clinical defaults are inferred.
    """
    attributes: dict[str, set[str]] = {}

    def add(axis: str, label: str) -> None:
        attributes.setdefault(axis, set()).add(label)

    def read(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for part in item:
                read(part)
            return
        if isinstance(item, str) and item.strip().startswith("{"):
            try:
                item = json.loads(item)
            except (ValueError, TypeError):
                return
        if isinstance(item, dict):
            if field == "population":
                if item.get("healthy") is True:
                    add("health", "healthy")
                elif item.get("healthy") is False:
                    add("health", "not_healthy")
                keys = ("population", "health_status", "sex", "age_group", "species")
            else:
                keys = ("formulation", "dosage_form", "release_type")
            for key in keys:
                if isinstance(item.get(key), str):
                    read(item[key])
            return
        if not isinstance(item, str):
            return
        text = _norm(item).replace("_", " ").replace("–", "-").replace("‑", "-")
        # Do not turn explicitly unreported attributes or excluded cohorts into
        # positive declarations. Other independent clauses can still contribute.
        text = "; ".join(part for part in re.split(r"[;,]", text)
            if not re.search(r"\b(?:unknown|unspecified|not (?:reported|specified|known|stated|included)|excluded)\b", part))
        if field == "formulation":
            # Manufacturer and batch metadata must not supply clinical axes.
            text = re.split(r"\b(?:lot|batch)(?=\b|[a-z0-9])", text, maxsplit=1)[0]
            for label, pattern in (("tablet", r"\btablets?\b"), ("capsule", r"\bcapsules?\b"),
                                   ("solution", r"\bsolution\b"), ("suspension", r"\bsuspension\b")):
                if re.search(pattern, text):
                    add("dosage_form", label)
            for label, pattern in (("immediate", r"\b(?:immediate[ -]release|ir)\b"),
                                   ("modified", r"\b(?:(?:sustained|extended|prolonged|controlled|modified)[ -]release|sr|er|xr|cr|mr)\b"),
                                   ("delayed", r"\b(?:delayed[ -]release|enteric[ -]coated)\b")):
                if re.search(pattern, text):
                    add("release", label)
        else:
            if re.search(r"\b(?:not healthy|non[ -]?healthy)\b", text):
                add("health", "not_healthy")
            elif re.search(r"\bhealthy\b", text):
                add("health", "healthy")
            if re.search(r"\bpatients?\b", text) and not re.search(r"\b(?:not|no) patients?\b", text):
                add("health", "not_healthy")
            for label, pattern in (("male", r"\b(?:male|males|men)\b"),
                                   ("female", r"\b(?:female|females|women)\b")):
                if re.search(pattern, text):
                    add("sex", label)
            if re.search(r"\b(?:geriatric|elderly|older adults?)\b", text):
                add("age_group", "older_adult")
            elif re.search(r"\badults?\b", text):
                # An adult cohort may include older adults; not a contradiction.
                add("age_group", "adult"); add("age_group", "older_adult")
            if re.search(r"\b(?:pediatric|paediatric|children|infants?|neonates?)\b", text):
                add("age_group", "pediatric")
            for label, pattern in (("human", r"\b(?:humans?|homo sapiens)\b"),
                                   ("goat", r"\bgoats?\b"), ("rat", r"\brats?\b"),
                                   ("mouse", r"\b(?:mouse|mice)\b"), ("rabbit", r"\brabbits?\b"),
                                   ("dog", r"\bdogs?\b"), ("pig", r"\bpigs?\b")):
                if re.search(pattern, text):
                    add("species", label)

    read(value)
    return attributes


def _condition_status(wanted: Any, actual: Any, field: str) -> str:
    expected, observed = _condition_values(wanted, field), _condition_values(actual, field)
    if field in {"population", "formulation"}:
        left, right = _descriptive_attributes(wanted, field), _descriptive_attributes(actual, field)
        if any(not (left[axis] & right[axis]) for axis in left.keys() & right.keys()):
            return "conflicting"
        # Partial axis agreement cannot establish brand, release, age, or other
        # unparsed conditions. Only identical declared labels retain a match.
        if expected and observed and _canonical(wanted, field) == _canonical(actual, field):
            return "matched"
        return "unspecified"
    if not expected or not observed:
        return "unspecified"
    if observed <= expected:
        return "matched"
    return "unspecified" if observed & expected else "conflicting"


def _quantity(value: Any, unit: Any = None) -> tuple[float, str] | None:
    if isinstance(value, dict):
        return _quantity(value.get("value", value.get("dose")), value.get("unit", value.get("dose_unit", unit)))
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+]?\d+(?:\.\d+)?)\s*([a-zA-Zμµ/²0-9]+)?\s*", value)
        if not match:
            return None
        value, embedded = match.groups()
        if embedded and unit and _norm(embedded) != _norm(unit):
            return None
        unit = embedded or unit
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number) or number <= 0 or not unit:
        return None
    normalized = _norm(unit).replace("μ", "u").replace("µ", "u").replace("²", "2").replace(" ", "")
    scale = {"g": 1000., "mg": 1., "ug": .001}
    parts = normalized.split("/")
    if parts[0] not in scale or len(parts) > 2:
        return None
    basis = parts[1] if len(parts) == 2 else "absolute"
    if basis not in {"absolute", "kg", "m2"}:
        return None
    return number * scale[parts[0]], basis


def _request_binding(query: dict, request: ModelingRequest, corpus: GraphPromptCorpus,
                     prompt: dict, recommendation: dict | None) -> dict:
    resolved = corpus.resolve_compound(request.compound)
    compound_id = str(resolved["node_id"])
    if not isinstance(query.get("text"), str):
        raise ComponentModelBridgeError("query_record.text must be the original request text")
    compounds = query.get("compounds") or (query.get("features") or {}).get("compounds")
    if not isinstance(compounds, list) or not compounds:
        raise ComponentModelBridgeError("Query compound identity is missing")
    try:
        identities = {str(corpus.resolve_compound(str(name))["node_id"]) for name in compounds}
    except LookupError as exc:
        raise ComponentModelBridgeError("Query compound cannot be resolved") from exc
    if compound_id not in identities:
        raise ComponentModelBridgeError("Query compound differs from component request")
    request_json = request.model_dump(mode="json")
    if query.get("modeling_request") is not None and query["modeling_request"] != request_json:
        raise ComponentModelBridgeError("Query modeling request differs from component request")
    for field, request_field in (("species", "species"), ("routes", "route"),
                                 ("populations", "population"), ("formulations", "formulation")):
        values = (query.get("features") or {}).get(field) or []
        expected = getattr(request, request_field)
        if values and expected and _condition_status(values, expected, request_field) == "conflicting":
            raise ComponentModelBridgeError(f"Query hard condition differs: {request_field}")
    for source in (query, query.get("conditions") or {}, query.get("clinical_design") or {}):
        if not isinstance(source, dict):
            continue
        for field in ("species", "population", "route", "formulation", "fed_or_fasted"):
            wanted = getattr(request, field)
            actual = source.get(field)
            if field == "fed_or_fasted":
                actual = actual or source.get("food")
            if wanted and isinstance(actual, str) and _condition_status(wanted, actual, field) == "conflicting":
                raise ComponentModelBridgeError(f"Query hard condition differs: {field}")
        expected_dose, actual_dose = _quantity(request.dose, request.dose_unit), _quantity(source.get("dose"), source.get("dose_unit"))
        if expected_dose and actual_dose and (expected_dose[1] != actual_dose[1]
                or not math.isclose(expected_dose[0], actual_dose[0], rel_tol=1e-9, abs_tol=1e-12)):
            raise ComponentModelBridgeError("Query hard condition differs: dose")
    if recommendation is not None:
        if prompt.get("raw_query_sha256") != _digest(query):
            raise ComponentModelBridgeError("Prompt original query hash differs or is missing")
        if prompt.get("modeling_request") != request_json:
            raise ComponentModelBridgeError("Prompt modeling request differs from component request")
        if prompt.get("request_sha256", _digest(request_json)) != _digest(request_json):
            raise ComponentModelBridgeError("Prompt request hash differs")
        if prompt.get("compound_id", compound_id) != compound_id:
            raise ComponentModelBridgeError("Prompt compound identity differs")
        if prompt.get("query_id", query.get("query_id")) != query.get("query_id"):
            raise ComponentModelBridgeError("Prompt query identity differs")
        if recommendation.get("resolved_modeling_request") != request_json:
            raise ComponentModelBridgeError("Component request conditions differ")
        if (recommendation.get("compound") or {}).get("compound_id") != compound_id:
            raise ComponentModelBridgeError("Component compound identity differs")
        if recommendation.get("objective", request.objective) != request.objective:
            raise ComponentModelBridgeError("Component objective differs")
    return resolved


def _contexts(corpus: GraphPromptCorpus, evidence_id: str, compound_id: str) -> list[dict]:
    edges = [e for e in corpus.edges_by_evidence.get(evidence_id, ()) if _quality_ok(e.get("provenance") or {})]
    events = {str(e["subject_id"]) for e in edges if e.get("predicate") == "OF_COMPOUND" and str(e.get("object_id")) == compound_id}
    events |= {str(e["object_id"]) for e in edges if e.get("predicate") in {"VICTIM_IN", "PERPETRATOR_IN"} and str(e.get("subject_id")) == compound_id}
    return [dict(event.get("context") or {}) for eid in sorted(events)
            if (event := corpus.event_by_id.get(eid)) is not None
            and str(event.get("evidence_id")) == evidence_id]


def _context_audit(request: ModelingRequest, contexts: list[dict]) -> dict:
    conflicts, matched, unknown = set(), set(), set()
    for context in contexts or [{}]:
        for field in ("species", "population", "route", "formulation", "fed_or_fasted"):
            wanted = getattr(request, field)
            if not wanted:
                continue
            actual = context.get(field)
            if field == "fed_or_fasted":
                actual = actual or context.get("food") or context.get("food_state")
            status = _condition_status(wanted, actual, field)
            if field == "species":
                # Species is sometimes explicit only in the population field
                # (e.g. lactating goats). Do not lose a real cross-species conflict.
                population_species = _descriptive_attributes(context.get("population"), "population").get("species")
                if population_species and _condition_status(wanted, sorted(population_species), field) == "conflicting":
                    status = "conflicting"
            {"matched": matched, "conflicting": conflicts, "unspecified": unknown}[status].add(field)
        wanted_dose = _quantity(request.dose, request.dose_unit)
        if wanted_dose:
            actual_dose = _quantity(context.get("dose"), context.get("dose_unit"))
            if actual_dose is None:
                unknown.add("dose")
            elif actual_dose[1] != wanted_dose[1] or not math.isclose(actual_dose[0], wanted_dose[0], rel_tol=1e-9, abs_tol=1e-12):
                conflicts.add("dose")
            else:
                matched.add("dose")
    return {"status": "conflicting" if conflicts else "unspecified" if unknown else "matched",
            "conflicting_fields": sorted(conflicts), "matched_fields": sorted(matched),
            "unspecified_fields": sorted(unknown)}


def _reachable(corpus: GraphPromptCorpus, evidence_id: str, compound_id: str) -> set[str]:
    """Reach only mechanistic nodes through this citation, never through a model."""
    edges = []
    for edge in corpus.edges_by_evidence.get(evidence_id, ()):
        if not _quality_ok(edge.get("provenance") or {}) or edge.get("predicate") in _EXCLUDED_EDGES:
            continue
        ends = [corpus.node_by_id.get(str(edge.get(k)), {}) for k in ("subject_id", "object_id")]
        if any(n.get("node_type") in _EXCLUDED_NODES for n in ends):
            continue
        edges.append(edge)
    reachable = {compound_id}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            left, right = str(edge["subject_id"]), str(edge["object_id"])
            if (left in reachable) != (right in reachable):
                reachable.update((left, right)); changed = True
    return reachable


def _name(node: dict) -> str | None:
    value = str(node.get("preferred_name") or "").strip()
    # Only short typed entity names are appended, never citations/free-form prose.
    if not value or len(value) > 96 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _/().+αβγ-]*", value):
        return None
    if re.search(r"\b(?:model|candidate|gmfe|ndcg|prediction|observed|score|validated)\b", value, re.I):
        return None
    return value


def _identity(node: dict, request: ModelingRequest) -> CompoundRef:
    identifiers = {str(k): str(v) for k, v in (node.get("standard_identifiers") or {}).items() if v is not None}
    identifiers["EVIDENCE_KG_NODE"] = str(node["node_id"])
    compound_uuid = None
    for possible in (identifiers.get("PBPKKB"), str(node["node_id"]).removeprefix("Compound:")):
        try:
            compound_uuid = UUID(str(possible)); break
        except ValueError:
            pass
    compound_uuid = compound_uuid or uuid5(COMPOUND_NAMESPACE, str(node["node_id"]))
    return CompoundRef(compound_id=compound_uuid, preferred_name=str(node["preferred_name"]),
                       active_moiety=request.active_moiety or str(node["preferred_name"]), salt_form=request.salt_form,
                       external_identifiers=identifiers, identity_resolution_method="evidence_kg_canonical_node", identity_confidence=1.)


def augment_model_query(query_record: dict, request: ModelingRequest, recommendation: dict | None,
                        corpus: GraphPromptCorpus, prompt_graph: dict,
                        source_evidence_run_id: UUID) -> dict:
    """Return an independent original-input augmentation and its provenance audit.

    Explicit conflicting evidence conditions suppress that evidence's additions.
    Unspecified conditions remain unknown; their mechanism is a soft preference,
    not a matching dose parameter or a hard condition. Only the original variant
    is augmented; existing perturbation variants are not silently relabelled.
    """
    if not isinstance(request, ModelingRequest):
        request = ModelingRequest.model_validate(request)
    if not isinstance(source_evidence_run_id, UUID):
        source_evidence_run_id = UUID(str(source_evidence_run_id))
    original = deepcopy(query_record)
    if recommendation is None:
        try:
            corpus.resolve_compound(request.compound)
        except LookupError:
            # No component signal is a baseline fallback, even if this frozen
            # component graph does not contain the target drug. Do not fabricate
            # a resolved identity or convert missing upstream coverage to refusal.
            return {"augmented_query": original, "requirement_profile": None, "audit": {
                "bridge_version": BRIDGE_VERSION, "status": "unchanged", "reason": "no_component_output_and_compound_absent",
                "request_sha256": _digest(request.model_dump(mode="json")), "original_query_sha256": _digest(original),
                "augmented_query_sha256": _digest(original), "component_recommendation_sha256": None,
                "source_evidence_run_id": str(source_evidence_run_id), "compound_node_id": None,
                "validation": None, "evidence_context": {}, "soft_requirements": [],
                "added_features": {field: [] for field in SOFT_FIELDS}, "profile_eligibility_used_for_ranking": False,
                "candidate_filtering": False, "hard_conditions_modified": False, "enhanced_variants": [],
            }}
    node = _request_binding(original, request, corpus, prompt_graph, recommendation)
    compound_id = str(node["node_id"])
    identity = _identity(node, request)
    validation = None
    evidence_audits, soft_rows, adapted_components = {}, [], []
    citations, component_evidence = {}, {}
    if recommendation is not None:
        allowed = prompt_graph.get("allowed_subgraph_ids", prompt_graph.get("retrieved_subgraph_ids"))
        if not isinstance(allowed, list) or not all(isinstance(s, str) and s in corpus.subgraphs for s in allowed):
            raise ComponentModelBridgeError("Prompt subgraph selection is missing or invalid")
        if "retrieved_subgraph_ids" in prompt_graph and set(allowed) != set(prompt_graph["retrieved_subgraph_ids"]):
            raise ComponentModelBridgeError("Prompt subgraph selections disagree")
        output_ids = prompt_graph.get("allowed_output_ids")
        if output_ids is not None:
            for field, values in (("evidence_ids", [r.get("evidence_id") for r in recommendation.get("citations") or []]),
                                  ("component_ids", [r.get("component") for r in recommendation.get("component_recommendations") or []])):
                if not set(values) <= set(output_ids.get(field) or []):
                    raise ComponentModelBridgeError(f"Selection outside prompt allowance: {field}")
        try:
            validation = validate_component_result(corpus=corpus, request=request,
                                                recommendation=recommendation, allowed_subgraph_ids=set(allowed))
        except (KeyError, TypeError, ValueError, LookupError) as exc:
            raise ComponentModelBridgeError("Malformed component recommendation") from exc
        if validation["status"] != "recommendation_validated":
            raise ComponentModelBridgeError("Component validation failed: " + "; ".join(validation.get("errors") or []))
        citations = {str(c["evidence_id"]): c for c in recommendation.get("citations") or []}
        if len(citations) != len(recommendation.get("citations") or []):
            raise ComponentModelBridgeError("Duplicate evidence citation")
        for eid, citation in citations.items():
            try:
                UUID(eid)
            except ValueError as exc:
                raise ComponentModelBridgeError("Evidence citation is not a canonical UUID") from exc
            evidence_audits[eid] = {**_context_audit(request, _contexts(corpus, eid, compound_id)),
                                   "source_id": citation["source_id"], "source_location": citation["source_location"],
                                   "source_text_hash": citation["source_text_hash"]}
        for component in recommendation.get("component_recommendations") or []:
            name = component["component"]
            if name not in PROCESSES:
                continue
            eids = [str(e) for e in component["evidence_ids"]
                    if evidence_audits[str(e)]["status"] != "conflicting"
                    and name in _components_for_evidence(corpus, str(e), compound_id=compound_id)]
            if not eids:
                continue
            component_evidence[name] = set(eids)
            soft_rows.append({"field": "process_types", "value": name, "component": name, "evidence_ids": sorted(eids)})
            adapted_components.append(ComponentRecommendation(component=name, status="recommended",
                rationale="Source-validated component; ranking preference only.",
                citations=[Citation(evidence_id=e, source_id=citations[e]["source_id"],
                    source_location=citations[e]["source_location"], source_text=citations[e]["supporting_text"],
                    confidence=citations[e].get("confidence")) for e in eids],
                evidence_count=len(eids), source_count=len({citations[e]["source_id"] for e in eids}),
                confidence=min(float(citations[e].get("confidence") or 0.) for e in eids)))
        reachable = {eid: _reachable(corpus, eid, compound_id) for eids in component_evidence.values() for eid in eids}
        possible_nodes = []
        for field, source_key, node_types, components in (
            ("enzymes", "metabolism_pathways", {"Enzyme"}, {"metabolism", "ddi"}),
            ("transporters", "transport_pathways", {"Transporter"}, {"transport", "excretion", "ddi"}),
        ):
            for path in recommendation.get(source_key) or []:
                possible_nodes.append((field, corpus.node_by_id.get(str(path.get("node_id")), {}), node_types, components))
        organs = set((recommendation.get("recommended_model_structure") or {}).get("required_compartments") or [])
        for nid in set().union(*reachable.values()) if reachable else set():
            entity = corpus.node_by_id.get(nid, {})
            if entity.get("preferred_name") in organs:
                possible_nodes.append(("compartments", entity, {"Organ", "Tissue"}, PROCESSES))
        for field, entity, node_types, components in possible_nodes:
            if entity.get("node_type") not in node_types or not (name := _name(entity)):
                continue
            support = {e for c in components for e in component_evidence.get(c, ()) if str(entity["node_id"]) in reachable[e]}
            if support:
                soft_rows.append({"field": field, "value": name, "node_id": str(entity["node_id"]), "evidence_ids": sorted(support)})

    response = RecommendationResponse(compound_id=str(identity.compound_id), compound_name=identity.preferred_name,
        objective=request.objective, retrieval_policy=BRIDGE_VERSION, coverage_score=0., overall_confidence=0.,
        components=adapted_components, ranked_subgraphs=[], public_model_candidates=[],
        evidence_gaps=list((recommendation or {}).get("evidence_gaps") or []), warnings=[], generated_from_auto_validated_only=True)
    # Use canonical resolved spelling only inside the legacy builder. The user's
    # request and original ranking query remain byte-for-byte untouched.
    canonical_request = request.model_copy(update={"compound": identity.preferred_name})
    profile = build_requirement_profile(request=canonical_request, compound_identity=identity,
        source_evidence_run_id=source_evidence_run_id, ontology_version="pbpk-recommendation-ontology-1.1.0",
        component_recommendation=response if recommendation is not None else None,
        created_at=datetime(1970, 1, 1, tzinfo=timezone.utc))
    profile_value = profile.model_dump(mode="json")
    present = {(r.category, r.semantic_id) for r in profile.requirements}
    context = RequirementContext(species=request.species, population=request.population, route=request.route,
                                 formulation=request.formulation, dose=request.dose, dose_unit=request.dose_unit)
    for row in soft_rows:
        if row["field"] == "process_types":
            continue  # Broad processes already come from the validated adapter.
        category = {"enzymes": "enzyme", "transporters": "transporter", "compartments": "compartment"}[row["field"]]
        semantic = f"{category}:{_norm(row['value'])}"
        if (category, semantic) in present:
            continue  # An explicit user requirement is never replaced or weakened.
        item = RequirementItem(requirement_id=uuid5(NAMESPACE, f"{profile.request_id}:{semantic}"),
            category=category, semantic_id=semantic, preferred_name=row["value"], necessity="preferred",
            expected_value=row["value"], context=context, evidence_ids=[UUID(e) for e in row["evidence_ids"]],
            origin="validated_evidence", confidence=min(float(citations[e].get("confidence") or 0.) for e in row["evidence_ids"]))
        profile_value["requirements"].append(item.model_dump(mode="json")); present.add((category, semantic))
    profile_value["checksum"] = requirement_profile_checksum(profile_value)
    profile = RequirementProfile.model_validate(profile_value)
    additions = {field: sorted({r["value"] for r in soft_rows if r["field"] == field}, key=str.casefold) for field in SOFT_FIELDS}
    augmented = deepcopy(original)
    if any(additions.values()):
        lines = ["Soft evidence preferences (optional; original user conditions remain authoritative; unlisted mechanisms are unknown):"]
        for field in SOFT_FIELDS:
            if additions[field]:
                lines.append(f"{field}: " + ", ".join(additions[field]) + ".")
        suffix = "\n\n" + "\n".join(lines)
        augmented["text"] = original["text"] + suffix
        feature_targets = [augmented.setdefault("features", {})]
        if "original" in augmented.get("variants", {}):
            variant = augmented["variants"]["original"]
            variant["text"] = original["variants"]["original"].get("text", original["text"]) + suffix
            feature_targets.append(variant.setdefault("features", deepcopy(original.get("features", {}))))
        for features in feature_targets:
            for field, added in additions.items():
                if not added:
                    continue
                existing = features.get(field) or []
                if not isinstance(existing, list):
                    raise ComponentModelBridgeError(f"Query feature {field} must be a list")
                features[field] = list(existing) + [v for v in added if _norm(v) not in {_norm(x) for x in existing}]
    return {"augmented_query": augmented, "requirement_profile": profile.model_dump(mode="json"), "audit": {
        "bridge_version": BRIDGE_VERSION, "status": "augmented" if augmented != original else "unchanged",
        "request_sha256": _digest(request.model_dump(mode="json")), "original_query_sha256": _digest(original),
        "augmented_query_sha256": _digest(augmented), "component_recommendation_sha256": _digest(recommendation) if recommendation is not None else None,
        "source_evidence_run_id": str(source_evidence_run_id), "compound_node_id": compound_id,
        "validation": validation, "evidence_context": evidence_audits, "soft_requirements": soft_rows,
        "added_features": additions, "legacy_profile_ranking_eligibility": profile.ranking_eligibility,
        "legacy_profile_gaps": [g.model_dump(mode="json") for g in profile.critical_gaps],
        "profile_eligibility_used_for_ranking": False, "candidate_filtering": False,
        "hard_conditions_modified": False, "enhanced_variants": ["original"] if augmented != original else [],
        "original_query_bound_to_prompt_sha256": recommendation is not None,
        "profile_timestamp_policy": "Epoch sentinel for deterministic content; not an execution timestamp",
        "missing_conditions": "Unknown; never assumed matched or converted to a negative model label",
    }}
