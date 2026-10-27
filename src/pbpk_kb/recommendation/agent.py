from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from pbpk_kb.recommendation.graph_store import RecommendationGraphStore, normalize
from pbpk_kb.recommendation.schemas import (
    Citation,
    ComponentRecommendation,
    RecommendationRequest,
    RecommendationResponse,
    SubgraphResult,
)


OBJECTIVE_DOMAINS = {
    "base_pk": {"physicochemical", "adme", "clinical_pk", "public_pbpk_model"},
    "oral_absorption": {"physicochemical", "adme", "clinical_pk", "public_pbpk_model"},
    "ddi": {"adme", "clinical_pk", "ddi", "public_pbpk_model"},
    "special_population": {"clinical_pk", "special_population", "adme", "public_pbpk_model"},
    "public_model_reuse": {"public_pbpk_model", "physicochemical", "adme", "clinical_pk"},
}
COMPONENT_RULES = {
    "compound_identity": {"physicochemical"},
    "physicochemical": {"physicochemical"},
    "absorption": {"adme", "clinical_pk"},
    "distribution": {"adme", "clinical_pk"},
    "metabolism": {"adme", "ddi"},
    "excretion": {"adme", "clinical_pk"},
    "ddi": {"ddi"},
    "special_population": {"special_population"},
    "public_model_template": {"public_pbpk_model"},
}
COMPONENT_TERMS = {
    "compound_identity": ("molecularformula", "molecularweight", "inchikey", "smiles"),
    "physicochemical": ("logp", "logd", "pka", "solubility", "tpsa", "protein binding", "fraction unbound"),
    "absorption": ("absorption", "bioavailability", "permeability", "cmax", "tmax"),
    "distribution": ("distribution", "volume", "protein binding", "fraction unbound"),
    "metabolism": ("metabolism", "metabolite", "cyp", "ugt", "intrinsic clearance"),
    "excretion": ("excretion", "renal clearance", "clearance", "urinary", "fecal", "biliary"),
    "ddi": ("auc", "cmax", "clearance", "inhibit", "induce", "substrate", "transporter"),
    "special_population": ("renal", "hepatic", "pediatric", "geriatric", "pregnancy", "genotype", "obesity"),
    "public_model_template": ("pbpk", "model", "pk-sim", "simcyp", "gastroplus"),
}
REQUIRED_COMPONENTS = {
    "base_pk": {
        "compound_identity", "physicochemical", "absorption", "distribution",
        "metabolism", "excretion", "public_model_template",
    },
    "oral_absorption": {
        "compound_identity", "physicochemical", "absorption", "distribution",
        "metabolism", "excretion",
    },
    "ddi": {"metabolism", "excretion", "ddi"},
    "special_population": {"absorption", "distribution", "metabolism", "excretion", "special_population"},
    "public_model_reuse": {
        "compound_identity", "physicochemical", "absorption", "distribution",
        "metabolism", "excretion", "public_model_template",
    },
}


class PBPKRecommendationAgent:
    def __init__(self, store: RecommendationGraphStore) -> None:
        self.store = store

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        compound = self.store.resolve_compound(request.compound)
        if compound is None:
            raise LookupError(f"Compound is not present in the auto-validated KG: {request.compound}")
        compound_id = compound["compound_id"]
        rows = self.store.compound_evidence(compound_id)
        coverage = coverage_score(rows, request.objective)
        max_hops = 1 if coverage >= 0.75 else 2
        retrieval_policy = (
            "coverage_sufficient_1_hop"
            if max_hops == 1
            else "coverage_gap_expand_to_2_hop"
        )
        subgraphs = rank_subgraphs(self.store, compound_id, rows, request, max_hops=max_hops)
        components = [
            build_component(name, rows, request)
            for name in COMPONENT_RULES
        ]
        model_candidates = [item for item in components if item.component == "public_model_template"]
        gaps = [
            f"{item.component}: Insufficient automatically validated evidence"
            for item in components
            if item.component in REQUIRED_COMPONENTS[request.objective]
            and item.status == "insufficient_evidence"
        ]
        if request.objective == "ddi" and not request.interacting_drug:
            gaps.append("ddi scenario: No interacting drug specified; only general DDI mechanism suggestions are available")
        if request.objective == "special_population" and not request.population:
            gaps.append("special population scenario: No target population specified")
        confidence = weighted_confidence(components, request.objective)
        return RecommendationResponse(
            compound_id=compound_id,
            compound_name=compound["compound_name"],
            objective=request.objective,
            retrieval_policy=retrieval_policy,
            coverage_score=round(coverage, 4),
            overall_confidence=round(confidence, 4),
            components=components,
            ranked_subgraphs=subgraphs[: request.top_k_subgraphs],
            public_model_candidates=model_candidates,
            evidence_gaps=gaps,
            warnings=[
                "Recommendations use only auto_validated public evidence.",
                "Recommendations are model-design inputs, not evidence of completed calibration, validation, or applicability assessment.",
                "Values without source support are not filled automatically.",
            ],
        )


def coverage_score(rows: list[dict[str, Any]], objective: str) -> float:
    required = OBJECTIVE_DOMAINS[objective]
    present = {str(row.get("evidence_type") or "") for row in rows}
    domain_coverage = len(required & present) / len(required)
    source_coverage = min(1.0, len({str(row.get("source_id") or "") for row in rows}) / 4.0)
    return 0.75 * domain_coverage + 0.25 * source_coverage


def rank_subgraphs(
    store: RecommendationGraphStore,
    compound_id: str,
    rows: list[dict[str, Any]],
    request: RecommendationRequest,
    *,
    max_hops: int,
) -> list[SubgraphResult]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("parameter") or row.get("evidence_type") or "unknown")].append(row)
    results: list[SubgraphResult] = []
    objective_domains = OBJECTIVE_DOMAINS[request.objective]
    for parameter, group in grouped.items():
        evidence_ids = {str(row["evidence_id"]) for row in group}
        graph_node_ids = {compound_id} | evidence_ids
        graph_edges = [
            edge
            for edge in store.edges
            if str((edge.get("properties") or {}).get("evidence_id") or "") in evidence_ids
        ]
        for edge in graph_edges:
            graph_node_ids.update((str(edge["from_id"]), str(edge["to_id"])))
        domain_match = len({str(row.get("evidence_type") or "") for row in group} & objective_domains) / max(1, len(objective_domains))
        parameter_match = text_match_score(parameter, request)
        scenario_match = scenario_score(group, request)
        source_diversity = min(1.0, len({str(row.get("source_id") or "") for row in group}) / 3.0)
        quality = sum(float(row.get("extraction_confidence") or 0.5) for row in group) / len(group)
        connectivity = min(1.0, len(graph_edges) / max(1, len(graph_node_ids)))
        score = (
            0.30 * domain_match
            + 0.20 * parameter_match
            + 0.15 * scenario_match
            + 0.15 * source_diversity
            + 0.10 * quality
            + 0.10 * connectivity
        )
        results.append(
            SubgraphResult(
                center_id=f"{compound_id}:{normalize(parameter)}",
                center_label=parameter,
                hop=max_hops,
                score=round(score, 4),
                matched_domains=sorted({str(row.get("evidence_type") or "") for row in group} & objective_domains),
                evidence_ids=sorted(evidence_ids),
                node_ids=sorted(graph_node_ids),
                edge_count=len(graph_edges),
            )
        )
    return sorted(results, key=lambda item: (-item.score, item.center_label.lower()))


def build_component(
    component: str,
    rows: list[dict[str, Any]],
    request: RecommendationRequest,
) -> ComponentRecommendation:
    allowed_types = COMPONENT_RULES[component]
    terms = COMPONENT_TERMS[component]
    matches = []
    for row in rows:
        if row.get("evidence_type") not in allowed_types:
            continue
        haystack = normalize(" ".join((
            str(row.get("parameter") or ""),
            str(row.get("source_text") or ""),
            str(row.get("condition") or ""),
            str(row.get("population") or ""),
        )))
        if component in {"ddi", "special_population", "public_model_template"} or any(term in haystack for term in terms):
            if scenario_row_matches(row, request, component=component):
                matches.append(row)
    matches.sort(key=evidence_rank_key, reverse=True)
    limit = request.max_evidence_per_component
    citations = [citation_from_row(row) for row in matches[:limit]]
    sources = {str(row.get("source_id") or "") for row in matches}
    parameters = sorted({str(row.get("parameter") or "") for row in matches if row.get("parameter")})
    if len(matches) >= 3 and len(sources) >= 2:
        status = "recommended"
    elif matches:
        status = "conditional"
    else:
        status = "insufficient_evidence"
    confidence = component_confidence(matches, sources)
    rationale = component_rationale(component, status, len(matches), len(sources), request)
    return ComponentRecommendation(
        component=component,
        status=status,
        rationale=rationale,
        parameters=parameters[:20],
        citations=citations,
        evidence_count=len(matches),
        source_count=len(sources),
        confidence=round(confidence, 4),
    )


def citation_from_row(row: dict[str, Any]) -> Citation:
    value: float | str | None = row.get("canonical_value")
    if value is None:
        value = row.get("value")
    if value is None:
        value = row.get("value_text")
    return Citation(
        evidence_id=str(row["evidence_id"]),
        source_id=str(row.get("source_id") or ""),
        source_location=str(row.get("source_location") or ""),
        source_text=str(row.get("source_text") or ""),
        parameter=str(row.get("parameter")) if row.get("parameter") else None,
        value=value,
        unit=str(row.get("canonical_unit") or row.get("unit")) if row.get("canonical_unit") or row.get("unit") else None,
        confidence=float(row.get("extraction_confidence")) if row.get("extraction_confidence") is not None else None,
    )


def text_match_score(parameter: str, request: RecommendationRequest) -> float:
    query = normalize(" ".join(filter(None, (
        request.objective,
        request.route,
        request.population,
        request.interacting_drug,
        request.constraints,
    ))))
    parameter_tokens = set(normalize(parameter).split())
    query_tokens = set(query.split())
    if parameter_tokens & query_tokens:
        return 1.0
    objective_terms = {
        "base_pk": {"auc", "cmax", "clearance", "distribution", "metabolism", "absorption"},
        "oral_absorption": {"absorption", "bioavailability", "permeability", "cmax", "tmax"},
        "ddi": {"ddi", "auc", "cmax", "cyp", "inhibition", "induction"},
        "special_population": {"renal", "hepatic", "pediatric", "geriatric", "pregnancy"},
        "public_model_reuse": {"pbpk", "model", "simulation", "validation"},
    }[request.objective]
    return 0.8 if parameter_tokens & objective_terms else 0.3


def scenario_score(rows: list[dict[str, Any]], request: RecommendationRequest) -> float:
    return max((1.0 if scenario_row_matches(row, request) else 0.25 for row in rows), default=0.0)


def scenario_row_matches(
    row: dict[str, Any],
    request: RecommendationRequest,
    *,
    component: str | None = None,
) -> bool:
    haystack = normalize(" ".join((
        str(row.get("route") or ""),
        str(row.get("population") or ""),
        str(row.get("condition") or ""),
        str(row.get("source_text") or ""),
    )))
    required = []
    if request.route and component in {None, "absorption", "distribution", "excretion"}:
        required.append(request.route)
    if request.population and component in {None, "special_population"}:
        required.append(request.population)
    if request.interacting_drug and component in {None, "ddi"}:
        required.append(request.interacting_drug)
    required = [normalize(value) for value in required if value]
    return all(value in haystack for value in required)


def evidence_rank_key(row: dict[str, Any]) -> tuple[float, int, int]:
    return (
        float(row.get("extraction_confidence") or 0.5),
        1 if row.get("canonical_value") is not None or row.get("value") is not None else 0,
        len(str(row.get("source_text") or "")),
    )


def component_confidence(rows: list[dict[str, Any]], sources: set[str]) -> float:
    if not rows:
        return 0.0
    quality = sum(float(row.get("extraction_confidence") or 0.5) for row in rows) / len(rows)
    count_factor = min(1.0, math.log2(len(rows) + 1) / 3.0)
    diversity = min(1.0, len(sources) / 3.0)
    return 0.5 * quality + 0.3 * count_factor + 0.2 * diversity


def component_rationale(component: str, status: str, evidence_count: int, source_count: int, request: RecommendationRequest) -> str:
    labels = {
        "compound_identity": "Compound identity and basic structure",
        "physicochemical": "Physicochemical parameters",
        "absorption": "Absorption",
        "distribution": "Tissue distribution",
        "metabolism": "Metabolic pathways",
        "excretion": "Excretion and clearance",
        "ddi": "Drug interactions",
        "special_population": "Special populations",
        "public_model_template": "Public PBPK model template",
    }
    if status == "insufficient_evidence":
        return f"{labels[component]}: no automatically validated evidence matches this scenario; parameters must not be filled automatically."
    qualifier = "can inform model design" if status == "recommended" else "may inform candidate selection, but coverage remains incomplete"
    return f"{labels[component]}: {evidence_count} evidence records from {source_count} sources; this evidence {qualifier}."


def weighted_confidence(components: list[ComponentRecommendation], objective: str) -> float:
    relevant = [item for item in components if COMPONENT_RULES[item.component] & OBJECTIVE_DOMAINS[objective]]
    if not relevant:
        return 0.0
    return sum(item.confidence for item in relevant) / len(relevant)
