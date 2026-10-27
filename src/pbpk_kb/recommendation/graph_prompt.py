from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pbpk_kb.normalization.unit_normalizer import normalize_unit
from pbpk_kb.recommendation.embedding_benchmark import load_jsonl
from pbpk_kb.recommendation.schemas import ModelingRequest


GRAPH_PROMPT_VERSION = "pbpk-graph-prompt-1.5.0"
RECOMMENDATION_SCHEMA_VERSION = "pbpk-recommendation-2.4.0"
VALIDATOR_VERSION = "pbpk-recommendation-validator-1.4.0"
CITATION_CONTEXT_VERSION = "pbpk-citation-context-1.0.0"
PARAMETER_VALUE_GATE_VERSION = "pbpk-parameter-value-gate-1.1.0"

OBJECTIVE_RELATIONS: dict[str, frozenset[str]] = {
    "base_pk": frozenset(
        {
            "ABSORBED_VIA",
            "EXCRETED_VIA",
            "HAS_MEASUREMENT",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEASURES",
            "MEDIATED_BY",
            "OF_COMPOUND",
            "REPORTS",
            "STUDIED_IN",
            "SUBSTRATE_OF",
            "TRANSPORTED_BY",
            "UNDER_REGIMEN",
        }
    ),
    "ddi_victim": frozenset(
        {
            "AFFECTS_PARAMETER",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEDIATED_BY",
            "PERPETRATOR_IN",
            "STUDIED_IN",
            "SUBSTRATE_OF",
            "UNDER_REGIMEN",
            "VICTIM_IN",
        }
    ),
    "ddi_perpetrator": frozenset(
        {
            "AFFECTS_PARAMETER",
            "HAS_UNIT",
            "HAS_VALUE",
            "INDUCES",
            "INHIBITS",
            "MEDIATED_BY",
            "PERPETRATOR_IN",
            "STUDIED_IN",
            "UNDER_REGIMEN",
            "VICTIM_IN",
        }
    ),
    "renal_impairment": frozenset(
        {
            "EXCRETED_VIA",
            "HAS_MEASUREMENT",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEASURES",
            "OF_COMPOUND",
            "STUDIED_IN",
            "UNDER_REGIMEN",
        }
    ),
    "hepatic_impairment": frozenset(
        {
            "HAS_MEASUREMENT",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEASURES",
            "MEDIATED_BY",
            "OF_COMPOUND",
            "STUDIED_IN",
            "SUBSTRATE_OF",
            "UNDER_REGIMEN",
        }
    ),
    "pediatric": frozenset(
        {
            "HAS_MEASUREMENT",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEASURES",
            "OF_COMPOUND",
            "STUDIED_IN",
            "UNDER_REGIMEN",
        }
    ),
    "pregnancy": frozenset(
        {
            "HAS_MEASUREMENT",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEASURES",
            "OF_COMPOUND",
            "STUDIED_IN",
            "UNDER_REGIMEN",
        }
    ),
    "special_population": frozenset(
        {
            "HAS_MEASUREMENT",
            "HAS_UNIT",
            "HAS_VALUE",
            "MEASURES",
            "OF_COMPOUND",
            "STUDIED_IN",
            "UNDER_REGIMEN",
        }
    ),
    "public_model_reuse": frozenset(
        {
            "CALIBRATED_WITH",
            "DESCRIBED_BY_MODEL",
            "HAS_COMPARTMENT",
            "HAS_MODEL_VERSION",
            "HAS_SIMULATION",
            "HAS_VALIDATION_RUN",
            "HAS_VALIDATION_METRIC",
            "EVALUATES_DATASET",
            "USES_ENZYME",
            "USES_PARAMETER",
            "USES_STRUCTURE",
            "USES_TRANSPORTER",
            "SUPPORTS_ROUTE",
            "SUPPORTS_FORMULATION",
            "APPLIES_TO",
            "VALIDATED_WITH",
        }
    ),
}

OBJECTIVE_ALIASES = {
    "ddi": "ddi_victim",
    "special_population": "special_population",
    "oral_absorption": "base_pk",
}

COMPONENT_RELATIONS: dict[str, frozenset[str]] = {
    "absorption": frozenset({"ABSORBED_VIA"}),
    "distribution": frozenset(),
    "metabolism": frozenset({"MEDIATED_BY", "METABOLIZED_TO", "SUBSTRATE_OF"}),
    "transport": frozenset({"TRANSPORTED_BY"}),
    "excretion": frozenset({"EXCRETED_VIA"}),
    "ddi": frozenset(
        {
            "AFFECTS_PARAMETER",
            "INDUCES",
            "INHIBITS",
            "PERPETRATOR_IN",
            "VICTIM_IN",
        }
    ),
    "special_population": frozenset({"STUDIED_IN"}),
    "public_model_template": frozenset({"DESCRIBED_BY_MODEL"}),
}

PARAMETER_COMPONENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "physicochemical": (
        "molecular",
        "formula",
        "logp",
        "logd",
        "pka",
        "solubility",
        "tpsa",
        "hba",
        "hbd",
    ),
    "absorption": (
        "absorp",
        "bioavail",
        "fraction absorbed",
        "permeab",
        "tmax",
        "cmax",
    ),
    "distribution": (
        "distribution",
        "volume",
        "vd",
        "vss",
        "protein binding",
        "fraction unbound",
        "fu",
    ),
    "metabolism": ("metab", "intrinsic clearance", "cyp", "ugt"),
    "excretion": ("excret", "renal clearance", "urinary", "fecal", "biliary"),
}

PARAMETER_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "auc": ("auc", "area under the curve", "systemic exposure"),
    "cmax": ("cmax", "maximum concentration", "peak concentration"),
    "tmax": ("tmax", "time to peak", "time to maximum"),
    "half_life": ("half life", "half-life", "t1/2", "t 1/2"),
    "clearance": ("clearance", "cl"),
    "oral_clearance": ("oral clearance", "cl/f"),
    "renal_clearance": ("renal clearance", "kidney clearance", "clr"),
    "nonrenal_clearance": ("nonrenal clearance", "non-renal clearance"),
    "intrinsic_clearance": ("intrinsic clearance", "clint"),
    "volume_of_distribution": (
        "volume of distribution",
        "distribution volume",
        "vd",
        "vss",
        "v/f",
    ),
    "bioavailability": ("bioavailability", "absolute bioavailability"),
    "fraction_absorbed": ("fraction absorbed", "absorbed", "absorption"),
    "fraction_excreted_unchanged": (
        "excreted unchanged",
        "excreted as unchanged",
        "unchanged in the urine",
        "unchanged parent drug",
        "urinary recovery",
        "excretion",
    ),
    "fraction_metabolized": ("fraction metabolized", "metabolized", "metabolism"),
    "protein_binding": ("protein binding", "protein bound", "bound to plasma"),
    "fraction_unbound": ("fraction unbound", "unbound fraction", "free fraction"),
    "absorption_rate_constant": (
        "absorption rate constant",
        "first-order absorption rate",
        "ka",
    ),
    "molecular_weight": ("molecular weight", "molar mass", "formula weight"),
    "logp": ("logp", "alogp", "xlogp", "partition coefficient"),
    "tpsa": ("tpsa", "topological polar surface area"),
    "hba": ("hba", "hydrogen bond acceptor"),
    "hbd": ("hbd", "hydrogen bond donor"),
}

PARAMETER_EVENT_TYPES: dict[str, frozenset[str]] = {
    "fraction_absorbed": frozenset({"AbsorptionEvent", "PKMeasurement"}),
    "absorption_rate_constant": frozenset({"AbsorptionEvent", "PKMeasurement"}),
    "bioavailability": frozenset({"AbsorptionEvent", "PKMeasurement"}),
    "volume_of_distribution": frozenset({"DistributionEvent", "PKMeasurement"}),
    "protein_binding": frozenset({"DistributionEvent", "PKMeasurement"}),
    "fraction_unbound": frozenset({"DistributionEvent", "PKMeasurement"}),
    "fraction_excreted_unchanged": frozenset({"ExcretionEvent", "PKMeasurement"}),
    "renal_clearance": frozenset({"ExcretionEvent", "PKMeasurement"}),
    "nonrenal_clearance": frozenset({"ExcretionEvent", "PKMeasurement"}),
    "fraction_metabolized": frozenset({"MetabolismEvent", "PKMeasurement"}),
    "intrinsic_clearance": frozenset({"MetabolismEvent", "PKMeasurement"}),
}

REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "base_pk": frozenset(
        {
            "physicochemical",
            "absorption",
            "distribution",
            "metabolism",
            "excretion",
        }
    ),
    "ddi_victim": frozenset({"metabolism", "ddi"}),
    "ddi_perpetrator": frozenset({"metabolism", "ddi"}),
    "renal_impairment": frozenset({"excretion", "special_population"}),
    "hepatic_impairment": frozenset({"metabolism", "special_population"}),
    "pediatric": frozenset({"special_population"}),
    "pregnancy": frozenset({"special_population"}),
    "special_population": frozenset({"special_population"}),
    "public_model_reuse": frozenset({"public_model_template"}),
}

POPULATION_OBJECTIVES = (
    (re.compile(r"\brenal|kidney|dialysis\b", re.I), "renal_impairment"),
    (re.compile(r"\bhepatic|cirrhos|liver impairment\b", re.I), "hepatic_impairment"),
    (re.compile(r"\bpediatric|child|children|neonat|infant\b", re.I), "pediatric"),
    (re.compile(r"\bpregnan|maternal|fetal|foetal\b", re.I), "pregnancy"),
)


def canonical_objective(value: str) -> str:
    return OBJECTIVE_ALIASES.get(value, value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(namespace: str, *parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()
    return f"{namespace}:{digest[:24]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _quality_ok(provenance: dict[str, Any]) -> bool:
    return (
        provenance.get("quality_status") == "auto_validated"
        and provenance.get("negation_status") != "negated"
    )


@dataclass(frozen=True)
class RetrievedSubgraph:
    subgraph_id: str
    hop: int
    score: float
    score_components: dict[str, float]
    row: dict[str, Any]
    vector: np.ndarray | None


class GraphPromptCorpus:
    def __init__(
        self,
        *,
        graph_dir: Path,
        subgraph_dir: Path,
        gnn_index_dir: Path | None = None,
    ) -> None:
        self.graph_dir = graph_dir
        self.subgraph_dir = subgraph_dir
        self.gnn_index_dir = gnn_index_dir
        self.nodes = load_jsonl(graph_dir / "kg_nodes.jsonl")
        self.edges = load_jsonl(graph_dir / "kg_edges.jsonl")
        self.events = load_jsonl(graph_dir / "kg_events.jsonl")
        self.node_by_id = {str(row["node_id"]): row for row in self.nodes}
        self.edge_by_id = {str(row["edge_id"]): row for row in self.edges}
        self.event_by_id = {str(row["event_id"]): row for row in self.events}
        self.compounds = {
            str(row["node_id"]): row
            for row in self.nodes
            if row["node_type"] == "Compound"
        }
        self.compound_by_name: dict[str, str] = {}
        for node_id, node in self.compounds.items():
            names = {str(node.get("preferred_name") or "")}
            names.update(str(value) for value in (node.get("properties") or {}).get("aliases") or ())
            for name in names:
                normalized = " ".join(name.lower().split())
                if normalized:
                    self.compound_by_name[normalized] = node_id
        self.subgraphs: dict[str, dict[str, Any]] = {}
        for hop in range(3):
            for row in load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl"):
                self.subgraphs[str(row["subgraph_id"])] = row
        self.subgraph_ids_by_node: dict[str, set[str]] = defaultdict(set)
        self.subgraph_ids_by_evidence: dict[str, set[str]] = defaultdict(set)
        for subgraph_id, row in self.subgraphs.items():
            for node in row.get("nodes") or ():
                self.subgraph_ids_by_node[str(node["node_id"])].add(subgraph_id)
            for citation in row.get("citations") or ():
                self.subgraph_ids_by_evidence[str(citation["evidence_id"])].add(subgraph_id)
        self.edges_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            provenance = edge.get("provenance") or {}
            evidence_id = str(provenance.get("evidence_id") or "")
            if evidence_id:
                self.edges_by_evidence[evidence_id].append(edge)
        self.gnn_record_by_id: dict[str, dict[str, Any]] = {}
        self.gnn_vectors: np.ndarray | None = None
        if gnn_index_dir is not None:
            records_path = gnn_index_dir / "subgraph_records.jsonl"
            vectors_path = gnn_index_dir / "subgraph_vectors.npy"
            if records_path.exists() and vectors_path.exists():
                records = load_jsonl(records_path)
                self.gnn_record_by_id = {
                    str(row["subgraph_id"]): row for row in records
                }
                self.gnn_vectors = np.load(vectors_path, mmap_mode="r")

    def resolve_compound(self, value: str) -> dict[str, Any]:
        if value in self.compounds:
            return self.compounds[value]
        node_id = self.compound_by_name.get(" ".join(value.lower().split()))
        if node_id is None:
            raise LookupError(f"Compound is absent from the formal KG: {value}")
        return self.compounds[node_id]

    def vector_for(self, subgraph_id: str) -> np.ndarray | None:
        if self.gnn_vectors is None:
            return None
        record = self.gnn_record_by_id.get(subgraph_id)
        if record is None:
            return None
        return np.asarray(
            self.gnn_vectors[int(record["embedding_row"])],
            dtype=np.float32,
        )

    def retrieve(
        self,
        request: ModelingRequest,
        *,
        max_subgraphs: int | None = None,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[RetrievedSubgraph]:
        compound = self.resolve_compound(request.compound)
        compound_id = str(compound["node_id"])
        objective = canonical_objective(request.objective)
        allowed = OBJECTIVE_RELATIONS.get(
            objective,
            OBJECTIVE_RELATIONS["base_pk"],
        )
        if candidate_ids is None:
            candidates = {
                subgraph_id
                for subgraph_id, row in self.subgraphs.items()
                if str(row.get("center_id") or "") == compound_id
            }
        else:
            candidates = {
                subgraph_id
                for subgraph_id in candidate_ids
                if subgraph_id in self.subgraphs
                and (
                    str(self.subgraphs[subgraph_id].get("center_id") or "")
                    == compound_id
                    or _subgraph_has_compound_path(
                        self.subgraphs[subgraph_id],
                        compound_id,
                    )
                )
            }
        query_tokens = _tokens(
            " ".join(
                str(value or "")
                for value in (
                    request.compound,
                    objective,
                    request.population,
                    request.route,
                    request.perpetrator_drug,
                    request.victim_drug,
                    request.constraints,
                )
            )
        )
        results: list[RetrievedSubgraph] = []
        for subgraph_id in candidates:
            row = self.subgraphs[subgraph_id]
            if int(row.get("hop_limit") or 0) == 0:
                continue
            predicates = {
                str(edge["predicate"])
                for edge in row.get("edges") or ()
            }
            citations = [
                citation
                for citation in row.get("citations") or ()
                if citation.get("quality_status") == "auto_validated"
            ]
            if not citations:
                continue
            scientific_edges = [
                edge
                for edge in row.get("edges") or ()
                if _quality_ok(edge.get("provenance") or {})
            ]
            if not scientific_edges:
                continue
            relation_score = len(predicates & allowed) / max(1, len(allowed))
            text = str(row.get("embedding_text") or "")
            scenario_score = len(query_tokens & _tokens(text)) / max(
                1,
                len(query_tokens),
            )
            source_diversity = min(
                1.0,
                len({str(item.get("source_id")) for item in citations}) / 4.0,
            )
            quality = sum(float(item.get("confidence") or 0.0) for item in citations) / len(
                citations
            )
            direct = float(row.get("center_id") == compound_id)
            conflict_penalty = float(
                any(edge.get("predicate") == "CONTRADICTS" for edge in row.get("edges") or ())
            )
            score_components = {
                "objective_relation": relation_score,
                "scenario_compatibility": scenario_score,
                "source_diversity": source_diversity,
                "evidence_quality": quality,
                "compound_center": direct,
                "conflict_penalty": conflict_penalty,
            }
            score = (
                0.35 * relation_score
                + 0.15 * scenario_score
                + 0.15 * source_diversity
                + 0.15 * quality
                + 0.20 * direct
                - 0.10 * conflict_penalty
            )
            results.append(
                RetrievedSubgraph(
                    subgraph_id=subgraph_id,
                    hop=int(row["hop_limit"]),
                    score=score,
                    score_components=score_components,
                    row=row,
                    vector=self.vector_for(subgraph_id),
                )
            )
        results.sort(key=lambda item: (-item.score, item.subgraph_id))
        return results[: max_subgraphs or request.top_k_subgraphs]

    def evidence_record(self, evidence_id: str) -> dict[str, Any]:
        node = self.node_by_id[evidence_id]
        edges = self.edges_by_evidence[evidence_id]
        provenance = next(
            (
                edge.get("provenance") or {}
                for edge in edges
                if _quality_ok(edge.get("provenance") or {})
            ),
            {},
        )
        return {
            "evidence_id": evidence_id,
            "evidence_type": str(node.get("preferred_name") or "").split(":", 1)[0],
            "parameter": (
                str(node.get("preferred_name") or "").split(":", 1)[1]
                if ":" in str(node.get("preferred_name") or "")
                else None
            ),
            "source_id": provenance.get("source_id"),
            "source_location": provenance.get("source_location"),
            "supporting_text": provenance.get("supporting_text"),
            "source_text_hash": provenance.get("source_text_hash"),
            "confidence": provenance.get("confidence"),
            "quality_status": provenance.get("quality_status"),
        }


def _components_for_subgraphs(
    subgraphs: list[RetrievedSubgraph],
) -> set[str]:
    predicates = {
        str(edge["predicate"])
        for item in subgraphs
        for edge in item.row.get("edges") or ()
    }
    parameter_names = {
        str(node.get("preferred_name") or "").lower()
        for item in subgraphs
        for node in item.row.get("nodes") or ()
        if node.get("node_type") == "PKParameter"
    }
    components = {
        component
        for component, relations in COMPONENT_RELATIONS.items()
        if relations & predicates
    }
    for component, patterns in PARAMETER_COMPONENT_PATTERNS.items():
        if any(
            any(pattern in parameter for pattern in patterns)
            for parameter in parameter_names
        ):
            components.add(component)
    return components


def derive_silver_target(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    subgraphs: list[RetrievedSubgraph],
    *,
    max_evidence: int = 16,
) -> dict[str, Any]:
    objective = canonical_objective(request.objective)
    compound = corpus.resolve_compound(request.compound)
    compound_id = str(compound["node_id"])
    relevant_evidence = _objective_evidence_ids(
        corpus=corpus,
        request=request,
        subgraphs=subgraphs,
        compound_id=compound_id,
    )
    evidence_scores: dict[str, float] = defaultdict(float)
    model_ids: set[str] = set()
    path_ids: set[str] = set()
    for rank, item in enumerate(subgraphs, 1):
        for citation in item.row.get("citations") or ():
            if citation.get("quality_status") != "auto_validated":
                continue
            if str(citation["evidence_id"]) not in relevant_evidence:
                continue
            evidence_scores[str(citation["evidence_id"])] = max(
                evidence_scores[str(citation["evidence_id"])],
                item.score + 1.0 / (20 + rank),
            )
        path_ids.update(str(value) for value in item.row.get("path_ids") or ())
        for node in item.row.get("nodes") or ():
            if node.get("node_type") != "PBPKModel":
                continue
            model_id = str(node["node_id"])
            if (
                (node.get("properties") or {}).get("reusable_candidate") is True
                and any(
                    str(edge["subject_id"]) == compound_id
                    and str(edge["object_id"]) == model_id
                    and edge["predicate"] == "DESCRIBED_BY_MODEL"
                    and str((edge.get("provenance") or {}).get("evidence_id") or "")
                    in relevant_evidence
                    and _quality_ok(edge.get("provenance") or {})
                    for edge in corpus.edges
                )
            ):
                model_ids.add(model_id)
    evidence_ids = sorted(
        evidence_scores,
        key=lambda value: (-evidence_scores[value], value),
    )[:max_evidence]
    components = set()
    for evidence_id in evidence_ids:
        components.update(_components_for_evidence(corpus, evidence_id))
    if model_ids:
        components.add("public_model_template")
    required = REQUIRED_COMPONENTS.get(objective, REQUIRED_COMPONENTS["base_pk"])
    gaps = sorted(f"missing_component:{value}" for value in required - components)
    if objective == "public_model_reuse" and not model_ids:
        gaps.append("no_reusable_public_model")
    return {
        "component_ids": sorted(components),
        "evidence_ids": evidence_ids,
        "public_model_ids": sorted(model_ids),
        "gap_codes": gaps,
        "path_ids": sorted(path_ids),
    }


def _subgraph_has_compound_path(
    row: dict[str, Any],
    compound_id: str,
) -> bool:
    for edge in row.get("edges") or ():
        if not _quality_ok(edge.get("provenance") or {}):
            continue
        if edge.get("predicate") == "OF_COMPOUND" and str(
            edge.get("object_id")
        ) == compound_id:
            return True
        if edge.get("predicate") in {
            "VICTIM_IN",
            "PERPETRATOR_IN",
            "DESCRIBED_BY_MODEL",
        } and str(edge.get("subject_id")) == compound_id:
            return True
    return False


def _objective_evidence_ids(
    *,
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    subgraphs: list[RetrievedSubgraph] | None,
    compound_id: str,
) -> set[str]:
    objective = canonical_objective(request.objective)
    candidate_evidence = (
        {
            str(citation["evidence_id"])
            for item in subgraphs
            for citation in item.row.get("citations") or ()
            if citation.get("quality_status") == "auto_validated"
        }
        if subgraphs is not None
        else {
            evidence_id
            for evidence_id, edges in corpus.edges_by_evidence.items()
            if any(_quality_ok(edge.get("provenance") or {}) for edge in edges)
        }
    )
    if objective == "base_pk":
        event_ids = {
            str(edge["subject_id"])
            for edge in corpus.edges
            if edge["predicate"] == "OF_COMPOUND"
            and str(edge["object_id"]) == compound_id
            and _quality_ok(edge.get("provenance") or {})
        }
        return {
            str(event.get("evidence_id"))
            for event_id in event_ids
            if (event := corpus.event_by_id.get(event_id)) is not None
            and str(event.get("evidence_id") or "") in candidate_evidence
        }
    if objective in {"ddi_victim", "ddi_perpetrator"}:
        target_predicate = (
            "VICTIM_IN" if objective == "ddi_victim" else "PERPETRATOR_IN"
        )
        counterpart_predicate = (
            "PERPETRATOR_IN" if objective == "ddi_victim" else "VICTIM_IN"
        )
        counterpart_name = (
            request.perpetrator_drug
            if objective == "ddi_victim"
            else request.victim_drug
        )
        counterpart_id = None
        if counterpart_name:
            try:
                counterpart_id = str(
                    corpus.resolve_compound(counterpart_name)["node_id"]
                )
            except LookupError:
                return set()
        event_ids = {
            str(edge["object_id"])
            for edge in corpus.edges
            if edge["predicate"] == target_predicate
            and str(edge["subject_id"]) == compound_id
            and _quality_ok(edge.get("provenance") or {})
        }
        if counterpart_id:
            counterpart_events = {
                str(edge["object_id"])
                for edge in corpus.edges
                if edge["predicate"] == counterpart_predicate
                and str(edge["subject_id"]) == counterpart_id
                and _quality_ok(edge.get("provenance") or {})
            }
            event_ids &= counterpart_events
        return {
            str(event.get("evidence_id"))
            for event_id in event_ids
            if (event := corpus.event_by_id.get(event_id)) is not None
            and str(event.get("evidence_id") or "") in candidate_evidence
        }
    if objective == "public_model_reuse":
        return {
            str((edge.get("provenance") or {}).get("evidence_id"))
            for edge in corpus.edges
            if edge["predicate"] == "DESCRIBED_BY_MODEL"
            and str(edge["subject_id"]) == compound_id
            and _quality_ok(edge.get("provenance") or {})
            and str((edge.get("provenance") or {}).get("evidence_id") or "")
            in candidate_evidence
        }
    if objective in {
        "renal_impairment",
        "hepatic_impairment",
        "pediatric",
        "pregnancy",
        "special_population",
    }:
        event_ids = {
            str(edge["subject_id"])
            for edge in corpus.edges
            if edge["predicate"] == "OF_COMPOUND"
            and str(edge["object_id"]) == compound_id
            and _quality_ok(edge.get("provenance") or {})
        }
        population_pattern = next(
            (
                pattern
                for pattern, mapped_objective in POPULATION_OBJECTIVES
                if mapped_objective == objective
            ),
            None,
        )
        population_events = set()
        for edge in corpus.edges:
            if (
                edge["predicate"] != "STUDIED_IN"
                or str(edge["subject_id"]) not in event_ids
                or not _quality_ok(edge.get("provenance") or {})
            ):
                continue
            population = corpus.node_by_id.get(str(edge["object_id"])) or {}
            if population_pattern is None or population_pattern.search(
                str(population.get("preferred_name") or "")
            ):
                population_events.add(str(edge["subject_id"]))
        return {
            str(event.get("evidence_id"))
            for event_id in population_events
            if (event := corpus.event_by_id.get(event_id)) is not None
            and str(event.get("evidence_id") or "") in candidate_evidence
        }
    allowed = OBJECTIVE_RELATIONS.get(objective, frozenset())
    return {
        str((edge.get("provenance") or {}).get("evidence_id"))
        for edge in corpus.edges
        if edge.get("predicate") in allowed
        and _quality_ok(edge.get("provenance") or {})
        and str((edge.get("provenance") or {}).get("evidence_id") or "")
        in candidate_evidence
    }


def _source_document_key(source_location: str) -> str:
    base, _, fragment = source_location.partition("#")
    article = re.search(
        r"xml_path=//(?:PubmedArticle|article)\[\d+\]",
        fragment,
        re.I,
    )
    return f"{base}#{article.group(0)}" if article else base


def _compound_split(compound_ids: list[str]) -> dict[str, str]:
    ordered = sorted(
        compound_ids,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    count = len(ordered)
    train_end = max(1, int(round(count * 0.70)))
    validation_end = max(train_end + 1, int(round(count * 0.85)))
    validation_end = min(validation_end, max(train_end + 1, count - 1))
    return {
        compound_id: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, compound_id in enumerate(ordered)
    }


def _assign_splits_without_resource_leakage(
    provisional: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compound_ids = sorted({str(row["compound_id"]) for row in provisional})
    resources_by_row: dict[str, list[str]] = {}
    resource_compounds: dict[str, set[str]] = defaultdict(set)
    for row in provisional:
        resources = [
            *(f"source:{value}" for value in row["source_document_ids"]),
            *(f"model:{value}" for value in row["public_model_checksums"]),
        ]
        resources_by_row[str(row["example_id"])] = resources
        for resource in resources:
            resource_compounds[resource].add(str(row["compound_id"]))
    exclusive_rows: Counter[str] = Counter()
    for row in provisional:
        compound_id = str(row["compound_id"])
        resources = resources_by_row[str(row["example_id"])]
        if all(resource_compounds[value] == {compound_id} for value in resources):
            exclusive_rows[compound_id] += 1
    held_out_order = sorted(
        compound_ids,
        key=lambda value: (
            -exclusive_rows[value],
            hashlib.sha256(value.encode("utf-8")).hexdigest(),
        ),
    )
    validation_count = max(2, int(round(len(compound_ids) * 0.15)))
    test_count = max(2, int(round(len(compound_ids) * 0.15)))
    held_out = held_out_order[: validation_count + test_count]
    validation_compounds = set(held_out[::2][:validation_count])
    test_compounds = set(held_out[1::2][:test_count])
    split_by_compound = {
        compound_id: (
            "validation"
            if compound_id in validation_compounds
            else "test"
            if compound_id in test_compounds
            else "train"
        )
        for compound_id in compound_ids
    }
    resource_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in provisional:
        compound_id = str(row["compound_id"])
        split = split_by_compound[compound_id]
        resources = resources_by_row[str(row["example_id"])]
        for resource in resources:
            resource_split_counts[resource][split] += 1
    resource_owner = {
        resource: min(
            counts,
            key=lambda split: (
                -counts[split],
                ("train", "validation", "test").index(split),
            ),
        )
        for resource, counts in resource_split_counts.items()
    }
    rows = []
    for row in provisional:
        split = split_by_compound[str(row["compound_id"])]
        resources = resources_by_row[str(row["example_id"])]
        if all(resource_owner[value] == split for value in resources):
            rows.append({**row, "split": split})
    return rows


def independent_target_subgraphs(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
) -> list[RetrievedSubgraph]:
    compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
    objective_evidence = _objective_evidence_ids(
        corpus=corpus,
        request=request,
        subgraphs=None,
        compound_id=compound_id,
    )
    candidate_ids = {
        subgraph_id
        for evidence_id in objective_evidence
        for subgraph_id in corpus.subgraph_ids_by_evidence.get(evidence_id, ())
    }
    rows = []
    for subgraph_id in sorted(candidate_ids):
        row = corpus.subgraphs[subgraph_id]
        if int(row.get("hop_limit") or 0) == 0:
            continue
        if not (
            str(row.get("center_id") or "") == compound_id
            or _subgraph_has_compound_path(row, compound_id)
        ):
            continue
        rows.append(
            RetrievedSubgraph(
                subgraph_id=subgraph_id,
                hop=int(row["hop_limit"]),
                score=1.0,
                score_components={"independent_silver_support": 1.0},
                row=row,
                vector=corpus.vector_for(subgraph_id),
            )
        )
    return rows


def positive_subgraphs_for_target(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    target: dict[str, Any],
) -> list[str]:
    compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
    evidence_ids = set(map(str, target.get("evidence_ids") or ()))
    model_ids = set(map(str, target.get("public_model_ids") or ()))
    values = []
    for subgraph_id, row in corpus.subgraphs.items():
        if int(row.get("hop_limit") or 0) == 0:
            continue
        if not (
            str(row.get("center_id") or "") == compound_id
            or _subgraph_has_compound_path(row, compound_id)
        ):
            continue
        citations = {
            str(value.get("evidence_id") or "")
            for value in row.get("citations") or ()
            if value.get("quality_status") == "auto_validated"
        }
        nodes = {str(value.get("node_id") or "") for value in row.get("nodes") or ()}
        if citations & evidence_ids or nodes & model_ids:
            values.append(subgraph_id)
    return sorted(values)


def derive_independent_silver_target(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    support_subgraphs: list[RetrievedSubgraph],
    *,
    max_evidence: int = 8,
) -> dict[str, Any]:
    raw = derive_silver_target(
        corpus,
        request,
        support_subgraphs,
        max_evidence=10_000,
    )
    objective = canonical_objective(request.objective)
    required = REQUIRED_COMPONENTS.get(
        objective,
        REQUIRED_COMPONENTS["base_pk"],
    )
    metadata = {}
    for evidence_id in raw["evidence_ids"]:
        record = corpus.evidence_record(evidence_id)
        components = _components_for_evidence(corpus, evidence_id)
        edges = corpus.edges_by_evidence.get(evidence_id, ())
        value_nodes = {
            str(edge["object_id"])
            for edge in edges
            if edge["predicate"] == "HAS_VALUE"
            and str(edge["object_id"]) in corpus.node_by_id
        }
        has_numeric = any(
            any(
                (node.get("properties") or {}).get(key) is not None
                for key in ("value", "value_low", "value_high", "value_text")
            )
            for value_id in value_nodes
            if (node := corpus.node_by_id[value_id])
        )
        has_unit = any(
            edge["predicate"] == "HAS_UNIT" for edge in edges
        ) or any(
            bool(
                (corpus.node_by_id[value_id].get("properties") or {}).get(
                    "canonical_unit"
                )
            )
            for value_id in value_nodes
        )
        metadata[evidence_id] = {
            "components": components,
            "numeric_context": bool(has_numeric and has_unit),
            "source_document": _source_document_key(
                str(record.get("source_location") or "")
            ),
            "confidence": float(record.get("confidence") or 0.0),
        }
    selected: list[str] = []
    covered_components: set[str] = set()
    covered_sources: set[str] = set()
    remaining = set(metadata)
    while remaining and len(selected) < max_evidence:
        evidence_id = min(
            remaining,
            key=lambda value: (
                -len(
                    metadata[value]["components"]
                    & required
                    - covered_components
                ),
                -int(metadata[value]["numeric_context"]),
                -int(
                    metadata[value]["source_document"]
                    not in covered_sources
                ),
                -float(metadata[value]["confidence"]),
                value,
            ),
        )
        selected.append(evidence_id)
        remaining.remove(evidence_id)
        covered_components.update(metadata[evidence_id]["components"])
        covered_sources.add(metadata[evidence_id]["source_document"])
    components = set()
    for evidence_id in selected:
        components.update(metadata[evidence_id]["components"])
    if raw["public_model_ids"]:
        components.add("public_model_template")
    gaps = sorted(
        f"missing_component:{value}" for value in required - components
    )
    if objective == "public_model_reuse" and not raw["public_model_ids"]:
        gaps.append("no_reusable_public_model")
    return {
        **raw,
        "component_ids": sorted(components),
        "evidence_ids": selected,
        "gap_codes": gaps,
        "selection_policy": (
            "objective_component_numeric_unit_source_confidence_v1"
        ),
        "eligible_evidence_count": len(metadata),
    }


def build_silver_examples(
    corpus: GraphPromptCorpus,
) -> list[dict[str, Any]]:
    requests_by_compound: dict[str, list[ModelingRequest]] = defaultdict(list)
    event_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in corpus.edges:
        if (
            str(edge.get("subject_id") or "") in corpus.event_by_id
            or str(edge.get("object_id") or "") in corpus.event_by_id
        ):
            event_id = (
                str(edge["subject_id"])
                if str(edge["subject_id"]) in corpus.event_by_id
                else str(edge["object_id"])
            )
            event_edges[event_id].append(edge)
    for event_id, edges in event_edges.items():
        if not all(
            _quality_ok(edge.get("provenance") or {})
            for edge in edges
            if (edge.get("provenance") or {}).get("quality_status")
        ):
            continue
        compound_ids = {
            str(edge["object_id"])
            for edge in edges
            if edge["predicate"] == "OF_COMPOUND"
            and str(edge["object_id"]) in corpus.compounds
        }
        for compound_id in compound_ids:
            compound = corpus.compounds[compound_id]
            requests_by_compound[compound_id].append(
                ModelingRequest(
                    compound=str(compound["preferred_name"]),
                    objective="base_pk",
                    species="human",
                )
            )
            for edge in edges:
                if edge["predicate"] != "STUDIED_IN":
                    continue
                population = corpus.node_by_id.get(str(edge["object_id"])) or {}
                population_name = str(population.get("preferred_name") or "")
                for pattern, objective in POPULATION_OBJECTIVES:
                    if pattern.search(population_name):
                        requests_by_compound[compound_id].append(
                            ModelingRequest(
                                compound=str(compound["preferred_name"]),
                                objective=objective,
                                species="human",
                                population=population_name,
                            )
                        )
        victims = {
            str(edge["subject_id"])
            for edge in edges
            if edge["predicate"] == "VICTIM_IN"
            and str(edge["subject_id"]) in corpus.compounds
        }
        perpetrators = {
            str(edge["subject_id"])
            for edge in edges
            if edge["predicate"] == "PERPETRATOR_IN"
            and str(edge["subject_id"]) in corpus.compounds
        }
        for victim_id in victims:
            victim = corpus.compounds[victim_id]
            for perpetrator_id in perpetrators:
                perpetrator = corpus.compounds[perpetrator_id]
                requests_by_compound[victim_id].append(
                    ModelingRequest(
                        compound=str(victim["preferred_name"]),
                        objective="ddi_victim",
                        species="human",
                        perpetrator_drug=str(perpetrator["preferred_name"]),
                    )
                )
                requests_by_compound[perpetrator_id].append(
                    ModelingRequest(
                        compound=str(perpetrator["preferred_name"]),
                        objective="ddi_perpetrator",
                        species="human",
                        victim_drug=str(victim["preferred_name"]),
                    )
                )
    for edge in corpus.edges:
        if (
            edge["predicate"] != "DESCRIBED_BY_MODEL"
            or str(edge["subject_id"]) not in corpus.compounds
            or not _quality_ok(edge.get("provenance") or {})
        ):
            continue
        compound_id = str(edge["subject_id"])
        compound = corpus.compounds[compound_id]
        requests_by_compound[compound_id].append(
            ModelingRequest(
                compound=str(compound["preferred_name"]),
                objective="public_model_reuse",
                species="human",
            )
        )
    deduplicated: dict[str, list[ModelingRequest]] = {}
    for compound_id, requests in requests_by_compound.items():
        unique = {
            canonical_json(request.model_dump(mode="json")): request
            for request in requests
        }
        deduplicated[compound_id] = [
            unique[key] for key in sorted(unique)
        ]
    provisional: list[dict[str, Any]] = []
    for compound_id, requests in deduplicated.items():
        for request in requests:
            objective = canonical_objective(request.objective)
            support_subgraphs = independent_target_subgraphs(corpus, request)
            target = derive_independent_silver_target(
                corpus,
                request,
                support_subgraphs,
            )
            minimum_evidence = 2 if objective == "base_pk" else 1
            if len(target["evidence_ids"]) < minimum_evidence:
                continue
            request_dict = request.model_dump(mode="json")
            example_id = stable_id(
                "recommendation-silver",
                compound_id,
                objective,
                canonical_json(request_dict),
                canonical_json(target),
            )
            source_documents = sorted(
                {
                    str(corpus.evidence_record(evidence_id)["source_location"]).split(
                        "\n",
                        1,
                    )[0]
                    for evidence_id in target["evidence_ids"]
                    if corpus.evidence_record(evidence_id)["source_location"]
                }
            )
            source_documents = sorted(
                {_source_document_key(value) for value in source_documents}
            )
            model_checksums = sorted(
                {
                    str(
                        (
                            corpus.node_by_id[model_id].get("properties")
                            or {}
                        ).get("model_checksum")
                    )
                    for model_id in target["public_model_ids"]
                    if (
                        corpus.node_by_id[model_id].get("properties") or {}
                    ).get("model_checksum")
                }
            )
            provisional.append(
                {
                    "example_id": example_id,
                    "benchmark_level": "silver",
                    "compound_id": compound_id,
                    "source_document_ids": source_documents,
                    "public_model_checksums": model_checksums,
                    "modeling_request": request_dict,
                    "positive_subgraph_ids": positive_subgraphs_for_target(
                        corpus,
                        request,
                        target,
                    ),
                    "target": target,
                }
            )
    examples = _assign_splits_without_resource_leakage(provisional)
    return sorted(examples, key=lambda row: str(row["example_id"]))


def audit_silver_examples(
    corpus: GraphPromptCorpus,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    resources: dict[str, set[str]] = defaultdict(set)
    compounds_by_split: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        example_id = str(row["example_id"])
        split = str(row["split"])
        compound_id = str(row["compound_id"])
        compounds_by_split[split].add(compound_id)
        request = ModelingRequest.model_validate(row["modeling_request"])
        objective = canonical_objective(request.objective)
        target = row["target"]
        positive_ids = set(map(str, row.get("positive_subgraph_ids") or ()))
        if not positive_ids:
            errors.append(
                {"example_id": example_id, "error": "missing_positive_subgraphs"}
            )
        for resource in (
            *(f"source:{value}" for value in row["source_document_ids"]),
            *(f"model:{value}" for value in row["public_model_checksums"]),
        ):
            resources[resource].add(split)
        for evidence_id in target["evidence_ids"]:
            record = corpus.evidence_record(str(evidence_id))
            if (
                record.get("quality_status") != "auto_validated"
                or not record.get("source_id")
                or not record.get("source_location")
                or not record.get("supporting_text")
            ):
                errors.append(
                    {
                        "example_id": example_id,
                        "error": f"invalid_target_provenance:{evidence_id}",
                    }
                )
            if not (
                corpus.subgraph_ids_by_evidence.get(str(evidence_id), set())
                & positive_ids
            ):
                errors.append(
                    {
                        "example_id": example_id,
                        "error": f"target_evidence_without_positive_subgraph:{evidence_id}",
                    }
                )
        if objective in {"ddi_victim", "ddi_perpetrator"}:
            target_predicate = (
                "VICTIM_IN" if objective == "ddi_victim" else "PERPETRATOR_IN"
            )
            counterpart_predicate = (
                "PERPETRATOR_IN" if objective == "ddi_victim" else "VICTIM_IN"
            )
            counterpart_name = (
                request.perpetrator_drug
                if objective == "ddi_victim"
                else request.victim_drug
            )
            counterpart_id = (
                str(corpus.resolve_compound(counterpart_name)["node_id"])
                if counterpart_name
                else ""
            )
            for evidence_id in target["evidence_ids"]:
                event_ids = {
                    str(event["event_id"])
                    for event in corpus.events
                    if str(event.get("evidence_id") or "") == str(evidence_id)
                }
                target_events = {
                    str(edge["object_id"])
                    for edge in corpus.edges_by_evidence[str(evidence_id)]
                    if edge["predicate"] == target_predicate
                    and str(edge["subject_id"]) == compound_id
                    and _quality_ok(edge.get("provenance") or {})
                }
                counterpart_events = {
                    str(edge["object_id"])
                    for edge in corpus.edges_by_evidence[str(evidence_id)]
                    if edge["predicate"] == counterpart_predicate
                    and str(edge["subject_id"]) == counterpart_id
                    and _quality_ok(edge.get("provenance") or {})
                }
                if not event_ids & target_events & counterpart_events:
                    errors.append(
                        {
                            "example_id": example_id,
                            "error": f"ddi_role_or_counterpart_mismatch:{evidence_id}",
                        }
                    )
        for model_id in target["public_model_ids"]:
            if not any(
                edge["predicate"] == "DESCRIBED_BY_MODEL"
                and str(edge["subject_id"]) == compound_id
                and str(edge["object_id"]) == str(model_id)
                and _quality_ok(edge.get("provenance") or {})
                for edge in corpus.edges
            ):
                errors.append(
                    {
                        "example_id": example_id,
                        "error": f"indirect_public_model_binding:{model_id}",
                    }
                )

    split_names = ("train", "validation", "test")
    compound_overlap = {
        f"{left}_{right}": len(
            compounds_by_split[left] & compounds_by_split[right]
        )
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    resource_leakage = {
        resource: sorted(splits)
        for resource, splits in resources.items()
        if len(splits) > 1
    }
    return {
        "accepted": not errors
        and not any(compound_overlap.values())
        and not resource_leakage,
        "example_count": len(examples),
        "compound_overlap_counts": compound_overlap,
        "resource_cross_split_count": len(resource_leakage),
        "resource_cross_split_examples": dict(
            list(sorted(resource_leakage.items()))[:20]
        ),
        "error_count": len(errors),
        "errors": errors[:100],
    }


def serialize_text_graph(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    subgraphs: list[RetrievedSubgraph],
) -> str:
    lines = [
        "GRAPH_PROMPT_VERSION " + GRAPH_PROMPT_VERSION,
        "MODELING_REQUEST " + canonical_json(request.model_dump(mode="json")),
    ]
    components = _components_for_subgraphs(subgraphs)
    lines.append("COMPONENT_COVERAGE " + ",".join(sorted(components)))
    target_options = derive_independent_silver_target(
        corpus,
        request,
        subgraphs,
        max_evidence=30,
    )
    evidence_priority = {
        evidence_id: index
        for index, evidence_id in enumerate(
            target_options["evidence_ids"]
        )
    }
    for rank, item in enumerate(subgraphs, 1):
        row = item.row
        lines.append(
            "SUBGRAPH "
            + canonical_json(
                {
                    "rank": rank,
                    "subgraph_id": item.subgraph_id,
                    "hop": item.hop,
                    "score": round(item.score, 6),
                    "score_components": {
                        key: round(value, 6)
                        for key, value in sorted(item.score_components.items())
                    },
                    "center_id": row.get("center_id"),
                    "center_type": row.get("center_type"),
                    "center_name": row.get("center_name"),
                }
            )
        )
        edges = [
            edge
            for edge in row.get("edges") or ()
            if _quality_ok(edge.get("provenance") or {})
        ]
        edges.sort(
            key=lambda edge: (
                -float((edge.get("provenance") or {}).get("confidence") or 0.0),
                str(edge["edge_id"]),
            )
        )
        for edge in edges[:24]:
            provenance = edge.get("provenance") or {}
            lines.append(
                "EDGE "
                + canonical_json(
                    {
                        "edge_id": edge["edge_id"],
                        "subject_id": edge["subject_id"],
                        "predicate": edge["predicate"],
                        "object_id": edge["object_id"],
                        "evidence_id": provenance.get("evidence_id"),
                    }
                )
            )
        for path_id in (row.get("path_ids") or ())[:8]:
            lines.append("PATH " + str(path_id))
        citations = [
            citation
            for citation in row.get("citations") or ()
            if citation.get("quality_status") == "auto_validated"
        ]
        citations.sort(
            key=lambda citation: (
                evidence_priority.get(
                    str(citation.get("evidence_id") or ""),
                    len(evidence_priority) + 1,
                ),
                -float(citation.get("confidence") or 0.0),
                str(citation.get("evidence_id") or ""),
            )
        )
        for citation in citations[:12]:
            lines.append(
                "CITATION "
                + canonical_json(
                    {
                        "evidence_id": citation.get("evidence_id"),
                        "source_id": citation.get("source_id"),
                        "source_location": citation.get("source_location"),
                        "supporting_text": citation.get("supporting_text"),
                    }
                )
            )
    lines.append(
        "ALLOWED_OUTPUT_IDS "
        + canonical_json(
            {
                "component_ids": sorted(
                    set(COMPONENT_RELATIONS) | set(PARAMETER_COMPONENT_PATTERNS)
                ),
                "evidence_ids": target_options["evidence_ids"],
                "public_model_ids": target_options["public_model_ids"],
                "gap_codes": sorted(
                    set(target_options["gap_codes"])
                    | {
                        f"missing_component:{value}"
                        for value in set().union(*REQUIRED_COMPONENTS.values())
                    }
                    | {"no_reusable_public_model"}
                ),
            }
        )
    )
    return "\n".join(lines)


def parse_selection(value: str | dict[str, Any]) -> dict[str, list[str]]:
    if isinstance(value, str):
        text = value.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Recommendation selection must be a JSON object.")
    result: dict[str, list[str]] = {}
    for key in ("component_ids", "evidence_ids", "public_model_ids", "gap_codes"):
        items = value.get(key)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"{key} must be a list of strings.")
        result[key] = sorted(set(items))
    return result


def validate_selection(
    selection: dict[str, list[str]],
    allowed: dict[str, list[str]],
) -> list[str]:
    errors = []
    for key in ("component_ids", "evidence_ids", "public_model_ids", "gap_codes"):
        unsupported = set(selection.get(key) or ()) - set(allowed.get(key) or ())
        if unsupported:
            errors.append(f"unsupported_{key}:{','.join(sorted(unsupported))}")
    return errors


def _canonical_parameter_name(value: Any) -> str | None:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    if not text:
        return None
    if text.startswith("auc") or "area under the curve" in text:
        return "auc"
    if text.startswith("cmax") or "maximum concentration" in text:
        return "cmax"
    if text.startswith("tmax") or "time to peak" in text:
        return "tmax"
    if "half life" in text or text in {"t1 2", "t 1 2"}:
        return "half_life"
    if "absorption rate constant" in text:
        return "absorption_rate_constant"
    if "protein binding" in text or text == "protein bound":
        return "protein_binding"
    if "fraction unbound" in text or text in {"unbound fraction", "free fraction", "fu"}:
        return "fraction_unbound"
    if "bioavailability" in text:
        return "bioavailability"
    if text.startswith("fraction metabolized via"):
        pathway = re.sub(r"[^a-z0-9]+", "_", text.removeprefix("fraction metabolized via").strip())
        return f"fraction_metabolized_via_{pathway}" if pathway else "fraction_metabolized"
    if "intrinsic clearance" in text:
        return "intrinsic_clearance"
    if "nonrenal clearance" in text or "non renal clearance" in text:
        return "nonrenal_clearance"
    if "renal clearance" in text or text == "clr":
        return "renal_clearance"
    if "oral clearance" in text or text == "cl f":
        return "oral_clearance"
    if "clearance" in text or text == "cl":
        return "clearance"
    if "volume of distribution" in text or text in {"distribution", "vd", "vss", "v f"}:
        return "volume_of_distribution"
    if text in {"absorption", "fraction absorbed", "fraction of dose absorbed", "fa"}:
        return "fraction_absorbed"
    if "excretion" in text or "excreted unchanged" in text:
        return "fraction_excreted_unchanged"
    if text == "metabolism" or text == "fraction metabolized":
        return "fraction_metabolized"
    if text in {"molecularweight", "molecular weight", "full mwt", "molar mass"}:
        return "molecular_weight"
    if text in {"alogp", "xlogp", "logp", "log p"}:
        return "logp"
    if text in {"tpsa", "topological polar surface area"}:
        return "tpsa"
    if text in {"hba", "hydrogen bond acceptor", "hydrogen bond acceptors"}:
        return "hba"
    if text in {"hbd", "hydrogen bond donor", "hydrogen bond donors"}:
        return "hbd"
    return None


def _parameter_family(parameter: str | None) -> str | None:
    if parameter and parameter.startswith("fraction_metabolized_via_"):
        return "fraction_metabolized"
    if parameter in {
        "clearance",
        "oral_clearance",
        "renal_clearance",
        "nonrenal_clearance",
        "intrinsic_clearance",
    }:
        return "clearance"
    return parameter


def _compact_unit(value: Any) -> str:
    return (
        str(value or "")
        .casefold()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("å", "a")
        .replace("²", "2")
        .replace("−", "-")
        .replace("–", "-")
        .replace("·", "*")
        .replace(" ", "")
    )


def _canonical_value_text(
    value: float | None,
    value_low: float | None,
    value_high: float | None,
    fallback: Any,
) -> str | None:
    if value is not None:
        return f"{value:g}"
    if value_low is not None and value_high is not None:
        return f"{value_low:g}-{value_high:g}"
    if value_low is not None:
        return f">={value_low:g}"
    if value_high is not None:
        return f"<={value_high:g}"
    return str(fallback) if fallback not in (None, "") else None


def _canonicalize_parameter_numeric_fields(
    properties: dict[str, Any],
    supporting_text: str,
) -> tuple[dict[str, Any], list[str]]:
    reported_unit = str(
        properties.get("reported_unit") or properties.get("unit") or ""
    ).strip() or None
    stored_canonical_unit = str(
        properties.get("canonical_unit") or ""
    ).strip() or None
    normalized = normalize_unit(reported_unit)
    reasons: list[str] = []
    identical_recorded_units = bool(
        reported_unit
        and stored_canonical_unit
        and _compact_unit(reported_unit) == _compact_unit(stored_canonical_unit)
    )
    if (
        reported_unit
        and normalized.dimension == "unknown"
        and not identical_recorded_units
    ):
        reasons.append("reported_unit_not_normalizable")
    if (
        reported_unit
        and normalized.canonical_unit
        and stored_canonical_unit
        and _compact_unit(normalized.canonical_unit)
        != _compact_unit(stored_canonical_unit)
    ):
        reasons.append("reported_and_canonical_unit_mismatch")
    canonical_unit = stored_canonical_unit or normalized.canonical_unit
    factor = (
        1.0
        if identical_recorded_units
        else normalized.multiplier_to_canonical if reported_unit else 1.0
    )
    canonical_values: dict[str, float | None] = {}
    reported_values: dict[str, float | None] = {}
    representations: dict[str, str] = {}
    for field in ("value", "value_low", "value_high"):
        stored = properties.get(field)
        if stored is None:
            canonical_values[field] = None
            reported_values[f"reported_{field}"] = None
            continue
        numeric = float(stored)
        if not reported_unit:
            if not _numeric_in_text(numeric, supporting_text):
                reasons.append(f"{field}_not_grounded_in_source_text")
            canonical_values[field] = numeric
            reported_values[f"reported_{field}"] = numeric
            representations[field] = "canonical_unit_only"
            continue
        if factor is None:
            reasons.append(f"{field}_unit_conversion_unavailable")
            canonical_values[field] = numeric
            reported_values[f"reported_{field}"] = numeric
            continue
        source_has_stored = _numeric_in_text(numeric, supporting_text)
        inferred_reported = numeric / factor if factor else numeric
        source_has_inferred = (
            factor != 1.0
            and _numeric_in_text(inferred_reported, supporting_text)
        )
        if factor == 1.0 and source_has_stored:
            canonical_values[field] = numeric
            reported_values[f"reported_{field}"] = numeric
            representations[field] = "identity_conversion"
        elif source_has_stored and not source_has_inferred:
            canonical_values[field] = numeric * factor
            reported_values[f"reported_{field}"] = numeric
            representations[field] = "reported_value_converted"
        elif source_has_inferred and not source_has_stored:
            canonical_values[field] = numeric
            reported_values[f"reported_{field}"] = inferred_reported
            representations[field] = "already_canonical_value"
        elif source_has_stored and source_has_inferred:
            reasons.append(f"{field}_numeric_representation_ambiguous")
            canonical_values[field] = numeric
            reported_values[f"reported_{field}"] = numeric
        else:
            reasons.append(f"{field}_not_grounded_in_source_text")
            canonical_values[field] = numeric
            reported_values[f"reported_{field}"] = numeric
    result = {
        **canonical_values,
        **reported_values,
        "reported_value_text": properties.get("value_text"),
        "reported_unit": reported_unit,
        "canonical_unit": canonical_unit,
        "unit_conversion_modes": representations,
    }
    result["value_text"] = _canonical_value_text(
        result["value"],
        result["value_low"],
        result["value_high"],
        properties.get("value_text"),
    )
    return result, sorted(set(reasons))


def _unit_supports_parameter(
    parameter: str,
    unit: Any,
    numeric_values: list[float],
) -> bool:
    family = _parameter_family(parameter)
    compact = _compact_unit(unit)
    percent = compact in {"%", "percent", "percentage"}
    unitless = compact in {"1", "unitless", "dimensionless", "fraction", "ratio"}
    time_unit = compact in {"s", "sec", "min", "h", "hr", "hour", "d", "day"}
    inverse_time = bool(
        re.fullmatch(r"(?:1/)?(?:s|sec|min|h|hr|hour|d|day)(?:\^-?1|-1)?", compact)
    ) and (compact.startswith("1/") or compact.endswith(("-1", "^-1")))
    concentration = bool(
        re.search(r"(?:pg|ng|ug|mg|g)/(?:ul|ml|dl|l)\b", compact)
    )
    exposure = (
        bool(re.search(r"(?:pg|ng|ug|mg|g)", compact))
        and bool(re.search(r"(?:ul|ml|dl|l)", compact))
        and bool(re.search(r"(?:s|sec|min|h|hr|hour|d|day)", compact))
        and "/" in compact
    )
    clearance = bool(
        re.fullmatch(
            r"(?:ul|ml|dl|l)/(?:s|sec|min|h|hr|hour|d|day)(?:/(?:g|kg))?",
            compact,
        )
    )
    volume = bool(re.fullmatch(r"(?:ul|ml|dl|l)(?:/(?:g|kg))?", compact))
    fraction_values_ok = all(
        0.0 <= value <= (100.0 if percent else 1.0) for value in numeric_values
    )

    if family == "auc":
        return exposure and not percent and not unitless
    if family == "cmax":
        return concentration and not exposure
    if family in {"tmax", "half_life"}:
        return time_unit
    if family == "clearance":
        return clearance
    if family == "volume_of_distribution":
        return volume
    if family == "absorption_rate_constant":
        return inverse_time
    if family in {
        "bioavailability",
        "fraction_absorbed",
        "fraction_excreted_unchanged",
        "fraction_metabolized",
        "protein_binding",
        "fraction_unbound",
    }:
        return (percent or unitless) and fraction_values_ok
    if family == "molecular_weight":
        return compact in {"g/mol", "kg/mol", "da", "dalton", "daltons"}
    if family == "logp":
        return compact in {"logp", "logunits", "unitless", "dimensionless"}
    if family == "tpsa":
        return compact in {"a2", "angstrom2", "angstrom^2"}
    if family in {"hba", "hbd"}:
        return compact in {"count", "unitless", "dimensionless", "1"}
    return False


def _parameter_aliases(
    canonical_parameter: str,
    measured_node: dict[str, Any],
) -> tuple[str, ...]:
    family = _parameter_family(canonical_parameter) or canonical_parameter
    values = list(PARAMETER_TEXT_ALIASES.get(family, ()))
    values.extend(
        str(value)
        for value in (measured_node.get("properties") or {}).get("aliases") or ()
        if value
    )
    if canonical_parameter.startswith("fraction_metabolized_via_"):
        values.append(canonical_parameter.removeprefix("fraction_metabolized_via_").replace("_", " "))
    return tuple(sorted({value.casefold() for value in values if value}, key=len, reverse=True))


SPECIES_INDEPENDENT_PARAMETER_FAMILIES = frozenset(
    {"molecular_weight", "logp", "tpsa", "hba", "hbd"}
)
IMPLICIT_HUMAN_REGULATORY_SOURCES = frozenset(
    {"dailymed", "drugs_fda", "openfda_drug_label", "clinicaltrials"}
)
HUMAN_CONTEXT_PATTERN = re.compile(
    r"\b(?:human|patients?|subjects?|volunteers?|adults?|children|pediatric|"
    r"infants?|neonates?|men|women|male participants?|female participants?|"
    r"healthy participants?)\b",
    re.I,
)
NONHUMAN_CONTEXT_PATTERN = re.compile(
    r"\b(?:rats?|mice|mouse|dogs?|rabbits?|monkeys?|macaques?|hamsters?|"
    r"guinea pigs?|porcine|swine|bovine)\b",
    re.I,
)


def _species_context_assessment(
    *,
    canonical_parameter: str | None,
    request: ModelingRequest | None,
    source_id: Any,
    context: dict[str, Any],
    supporting_text: str,
) -> tuple[str, list[str]]:
    if request is None or not request.species:
        return "not_requested", []
    family = _parameter_family(canonical_parameter)
    if family in SPECIES_INDEPENDENT_PARAMETER_FAMILIES:
        return "species_independent_property", []
    target_species = str(request.species).strip().casefold()
    observed_species = str(context.get("species") or "").strip().casefold()
    if target_species == "human":
        if observed_species:
            if observed_species in {"human", "humans", "homo sapiens"}:
                return "explicit_human_event_context", []
            return "explicit_nonhuman_event_context", [
                "species_context_mismatch_for_human_request"
            ]
        if NONHUMAN_CONTEXT_PATTERN.search(supporting_text):
            return "explicit_nonhuman_source_text", [
                "species_context_mismatch_for_human_request"
            ]
        if str(source_id or "") in IMPLICIT_HUMAN_REGULATORY_SOURCES:
            return "human_regulatory_or_trial_source", []
        if HUMAN_CONTEXT_PATTERN.search(supporting_text):
            return "explicit_human_source_text", []
        return "species_context_absent", [
            "human_species_context_not_explicit"
        ]
    if not observed_species:
        return "species_context_absent", ["requested_species_context_not_explicit"]
    if observed_species != target_species:
        return "explicit_other_species_event_context", [
            "requested_species_context_mismatch"
        ]
    return "explicit_requested_species_event_context", []


def _phrase_positions(text: str, aliases: tuple[str, ...]) -> list[tuple[int, int]]:
    lowered = text.casefold()
    positions: list[tuple[int, int]] = []
    for alias in aliases:
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            re.I,
        )
        positions.extend((match.start(), match.end()) for match in pattern.finditer(lowered))
    return positions


def _single_percent_is_nearest_parameter_value(
    value: float,
    text: str,
    aliases: tuple[str, ...],
) -> bool:
    phrases = _phrase_positions(text, aliases)
    numbers = [
        (float(match.group(1)), match.start(1), match.end(1))
        for match in re.finditer(r"(?<![\d.])([-+]?\d+(?:\.\d+)?)\s*%", text)
    ]
    if not phrases or not numbers:
        return True

    selected_numbers = [
        (start, end)
        for candidate, start, end in numbers
        if abs(candidate - value) <= max(1e-9, abs(value) * 1e-6)
    ]
    if not selected_numbers:
        return False
    for left, right in phrases:
        phrase_center = (left + right) // 2
        nearest_distance = min(
            abs(((start + end) // 2) - phrase_center)
            for _, start, end in numbers
        )
        if any(
            abs(((start + end) // 2) - phrase_center) <= nearest_distance
            for start, end in selected_numbers
        ):
            return True
    return False


def _parameter_candidate_rejection_reasons(
    *,
    canonical_parameter: str | None,
    measured_node: dict[str, Any],
    evidence_node: dict[str, Any],
    event: dict[str, Any],
    numeric_values: list[float],
    canonical_unit: Any,
    source_location: str,
    supporting_text: str,
) -> list[str]:
    reasons: list[str] = []
    if canonical_parameter is None:
        return ["unsupported_parameter_semantics"]
    evidence_parameter = _canonical_parameter_name(
        str(evidence_node.get("preferred_name") or "").partition(":")[2]
        or evidence_node.get("preferred_name")
    )
    if evidence_parameter is not None and _parameter_family(evidence_parameter) != _parameter_family(canonical_parameter):
        reasons.append("evidence_parameter_semantics_mismatch")
    event_type = str(event.get("event_type") or "")
    allowed_event_types = PARAMETER_EVENT_TYPES.get(canonical_parameter)
    if allowed_event_types is None:
        allowed_event_types = PARAMETER_EVENT_TYPES.get(
            _parameter_family(canonical_parameter) or ""
        )
    if allowed_event_types is not None and event_type not in allowed_event_types:
        reasons.append("event_parameter_semantics_mismatch")
    if not numeric_values:
        reasons.append("numeric_value_missing")
    elif not canonical_unit:
        reasons.append("canonical_unit_missing")
    elif not _unit_supports_parameter(
        canonical_parameter,
        canonical_unit,
        numeric_values,
    ):
        reasons.append("parameter_unit_dimension_mismatch")
    confidence = float(
        (evidence_node.get("properties") or {}).get("extraction_confidence") or 0.0
    )
    if confidence < 0.8:
        reasons.append("extraction_confidence_below_parameter_threshold")

    aliases = _parameter_aliases(canonical_parameter, measured_node)
    structured_json = "#json_path=" in source_location
    contextual_table = "#table=" in source_location and bool(event.get("context"))
    if not structured_json and not contextual_table and not _phrase_positions(supporting_text, aliases):
        reasons.append("parameter_phrase_not_found_in_source_text")
    relative_change = re.search(
        r"(?:increas|decreas|reduc|rose|fell|chang)[^.;]{0,35}(?:\bby\b|\bfrom\b)|"
        r"\b(?:fold|relative)\s+(?:increase|decrease|change)",
        supporting_text,
        re.I,
    )
    if relative_change and _parameter_family(canonical_parameter) in {
        "auc",
        "cmax",
        "clearance",
        "volume_of_distribution",
        "half_life",
        "tmax",
        "protein_binding",
    }:
        reasons.append("relative_or_comparative_result_not_absolute_parameter")
    metabolic_pathways = sorted(
        set(re.findall(r"\bcyp\s*\d+[a-z]\d*\b", supporting_text, re.I))
    )
    if canonical_parameter == "fraction_metabolized" and len(metabolic_pathways) > 1:
        reasons.append("multiple_metabolic_pathways_not_resolved")
    if (
        len(numeric_values) == 1
        and _compact_unit(canonical_unit) == "%"
        and not structured_json
        and not _single_percent_is_nearest_parameter_value(
            numeric_values[0],
            supporting_text,
            aliases,
        )
    ):
        reasons.append("numeric_value_not_nearest_parameter_mention")
    return sorted(set(reasons))


def _parameter_candidate_derivation(
    corpus: GraphPromptCorpus,
    evidence_ids: set[str],
    compound_id: str,
    request: ModelingRequest | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for event in corpus.events:
        evidence_id = str(event.get("evidence_id") or "")
        if evidence_id not in evidence_ids:
            continue
        event_edges = [
            corpus.edge_by_id[str(edge_id)]
            for edge_id in event.get("argument_edge_ids") or ()
            if str(edge_id) in corpus.edge_by_id
        ]
        of_compound = [
            edge
            for edge in event_edges
            if edge["predicate"] == "OF_COMPOUND"
        ]
        if of_compound and not any(
            str(edge["object_id"]) == compound_id for edge in of_compound
        ):
            continue
        measured = next(
            (
                corpus.node_by_id[str(edge["object_id"])]
                for edge in event_edges
                if edge["predicate"] in {"MEASURES", "AFFECTS_PARAMETER"}
                and str(edge["object_id"]) in corpus.node_by_id
            ),
            None,
        )
        value_node = next(
            (
                corpus.node_by_id[str(edge["object_id"])]
                for edge in event_edges
                if edge["predicate"] == "HAS_VALUE"
                and str(edge["object_id"]) in corpus.node_by_id
            ),
            None,
        )
        if measured is None or value_node is None:
            continue
        properties = value_node.get("properties") or {}
        if (
            properties.get("value") is None
            and properties.get("value_low") is None
            and properties.get("value_high") is None
            and not properties.get("value_text")
        ):
            continue
        provenance = next(
            (
                edge.get("provenance") or {}
                for edge in event_edges
                if str((edge.get("provenance") or {}).get("evidence_id") or "")
                == evidence_id
                and _quality_ok(edge.get("provenance") or {})
            ),
            {},
        )
        if not provenance:
            continue
        supporting_text = str(provenance.get("supporting_text") or "")
        normalized_values, canonicalization_reasons = (
            _canonicalize_parameter_numeric_fields(
                properties,
                supporting_text,
            )
        )
        value = normalized_values["value"]
        value_low = normalized_values["value_low"]
        value_high = normalized_values["value_high"]
        value_text = normalized_values["value_text"]
        numeric_values = [
            float(candidate_value)
            for candidate_value in (value, value_low, value_high)
            if candidate_value is not None
        ]
        reported_parameter = str(measured.get("preferred_name") or "")
        canonical_parameter = _canonical_parameter_name(reported_parameter)
        if canonical_parameter == "fraction_metabolized":
            metabolic_pathways = sorted(
                {
                    re.sub(r"\s+", "", value).casefold()
                    for value in re.findall(
                        r"\bcyp\s*\d+[a-z]\d*\b",
                        supporting_text,
                        re.I,
                    )
                }
            )
            if len(metabolic_pathways) == 1:
                canonical_parameter = (
                    f"fraction_metabolized_via_{metabolic_pathways[0]}"
                )
        evidence_node = corpus.node_by_id.get(evidence_id) or {}
        canonical_unit = normalized_values["canonical_unit"]
        context = dict(event.get("context") or {})
        species_context_basis, species_reasons = _species_context_assessment(
            canonical_parameter=canonical_parameter,
            request=request,
            source_id=provenance.get("source_id"),
            context=context,
            supporting_text=supporting_text,
        )
        rejection_reasons = sorted(
            set(canonicalization_reasons)
            | set(species_reasons)
            | set(
                _parameter_candidate_rejection_reasons(
                    canonical_parameter=canonical_parameter,
                    measured_node=measured,
                    evidence_node=evidence_node,
                    event=event,
                    numeric_values=numeric_values,
                    canonical_unit=canonical_unit,
                    source_location=str(
                        provenance.get("source_location") or ""
                    ),
                    supporting_text=supporting_text,
                )
            )
        )
        if rejection_reasons:
            rejections.append(
                {
                    "event_id": event["event_id"],
                    "evidence_id": evidence_id,
                    "reported_parameter": reported_parameter,
                    "canonical_parameter": canonical_parameter,
                    "value": value,
                    "value_low": value_low,
                    "value_high": value_high,
                    "value_text": value_text,
                    "canonical_unit": canonical_unit,
                    "reported_value": normalized_values["reported_value"],
                    "reported_value_low": normalized_values[
                        "reported_value_low"
                    ],
                    "reported_value_high": normalized_values[
                        "reported_value_high"
                    ],
                    "reported_value_text": normalized_values[
                        "reported_value_text"
                    ],
                    "reported_unit": normalized_values["reported_unit"],
                    "unit_conversion_modes": normalized_values[
                        "unit_conversion_modes"
                    ],
                    "species_context_basis": species_context_basis,
                    "source_id": provenance.get("source_id"),
                    "source_location": provenance.get("source_location"),
                    "reasons": rejection_reasons,
                }
            )
            continue
        route = next(
            (
                corpus.node_by_id[str(edge["object_id"])].get("preferred_name")
                for edge in event_edges
                if edge["predicate"] == "HAS_ROUTE"
                and str(edge["object_id"]) in corpus.node_by_id
            ),
            context.get("route"),
        )
        population = next(
            (
                corpus.node_by_id[str(edge["object_id"])].get("preferred_name")
                for edge in event_edges
                if edge["predicate"] == "STUDIED_IN"
                and str(edge["object_id"]) in corpus.node_by_id
            ),
            context.get("population"),
        )
        candidates.append(
            {
                "parameter": canonical_parameter,
                "reported_parameter": reported_parameter,
                "value": value,
                "value_low": value_low,
                "value_high": value_high,
                "value_text": value_text,
                "canonical_unit": canonical_unit,
                "reported_value": normalized_values["reported_value"],
                "reported_value_low": normalized_values[
                    "reported_value_low"
                ],
                "reported_value_high": normalized_values[
                    "reported_value_high"
                ],
                "reported_value_text": normalized_values[
                    "reported_value_text"
                ],
                "reported_unit": normalized_values["reported_unit"],
                "unit_conversion_modes": normalized_values[
                    "unit_conversion_modes"
                ],
                "species_context_basis": species_context_basis,
                "compound_id": compound_id,
                "form": context.get("form"),
                "species": context.get("species"),
                "population": population,
                "route": route,
                "dose": context.get("dose"),
                "formulation": context.get("formulation"),
                "study_context": context,
                "evidence_id": evidence_id,
                "source_id": provenance.get("source_id"),
                "source_location": provenance.get("source_location"),
                "supporting_text": supporting_text,
                "selection_reason": (
                    "Deterministically retained from an auto-validated PK event "
                    "after target-compound, parameter-semantic, reported-to-canonical "
                    "unit conversion, source-text, and context checks. This remains "
                    "an evidence candidate and is not a calibrated model input."
                ),
                "event_id": event["event_id"],
                "parameter_value_gate_version": PARAMETER_VALUE_GATE_VERSION,
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = stable_id(
            "parameter-candidate",
            row["parameter"],
            row["value"],
            row["value_low"],
            row["value_high"],
            row["value_text"],
            row["canonical_unit"],
            row["evidence_id"],
        )
        unique[key] = row
    accepted = sorted(
        unique.values(),
        key=lambda row: (
            str(row["parameter"]),
            str(row["source_id"]),
            str(row["source_location"]),
        ),
    )
    rejection_unique = {
        stable_id(
            "parameter-rejection",
            row["event_id"],
            row["evidence_id"],
            row["reported_parameter"],
            row["value"],
            row["value_low"],
            row["value_high"],
            row["canonical_unit"],
        ): row
        for row in rejections
    }
    rejected = sorted(
        rejection_unique.values(),
        key=lambda row: (
            str(row["reported_parameter"]),
            str(row["source_id"]),
            str(row["source_location"]),
        ),
    )
    return accepted, rejected


def _parameter_candidates(
    corpus: GraphPromptCorpus,
    evidence_ids: set[str],
    compound_id: str,
) -> list[dict[str, Any]]:
    return _parameter_candidate_derivation(corpus, evidence_ids, compound_id)[0]


def _citation_record(
    *,
    corpus: GraphPromptCorpus,
    evidence_id: str,
    compound: dict[str, Any],
    parameter_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    compound_id = str(compound["node_id"])
    binding_edges = sorted(
        [
            edge
            for edge in corpus.edges_by_evidence.get(evidence_id, ())
            if _quality_ok(edge.get("provenance") or {})
            and (
                (
                    edge.get("predicate") == "OF_COMPOUND"
                    and str(edge.get("object_id") or "") == compound_id
                )
                or (
                    edge.get("predicate")
                    in {"VICTIM_IN", "PERPETRATOR_IN", "DESCRIBED_BY_MODEL"}
                    and str(edge.get("subject_id") or "") == compound_id
                )
            )
        ],
        key=lambda edge: str(edge["edge_id"]),
    )
    measurements = []
    for candidate in parameter_candidates:
        if str(candidate.get("evidence_id") or "") != evidence_id:
            continue
        measurements.append(
            {
                key: candidate.get(key)
                for key in (
                    "event_id",
                    "parameter",
                    "value",
                    "value_low",
                    "value_high",
                    "value_text",
                    "canonical_unit",
                    "reported_value",
                    "reported_value_low",
                    "reported_value_high",
                    "reported_value_text",
                    "reported_unit",
                    "unit_conversion_modes",
                    "species_context_basis",
                    "form",
                    "species",
                    "population",
                    "route",
                    "dose",
                    "formulation",
                    "study_context",
                )
            }
        )
    measurements.sort(
        key=lambda row: (
            str(row.get("parameter") or ""),
            str(row.get("event_id") or ""),
        )
    )
    record = corpus.evidence_record(evidence_id)
    record.update(
        {
            "target_compound": {
                "node_id": compound_id,
                "preferred_name": compound.get("preferred_name"),
                "standard_identifiers": dict(
                    sorted(
                        (compound.get("standard_identifiers") or {}).items()
                    )
                ),
            },
            "compound_attribution": {
                "status": "validated_graph_binding",
                "edge_ids": [str(edge["edge_id"]) for edge in binding_edges],
                "predicates": sorted(
                    {str(edge["predicate"]) for edge in binding_edges}
                ),
            },
            "structured_measurements": measurements,
            "context_status": (
                "structured_measurement_context_available"
                if measurements
                else "source_text_and_graph_binding_only"
            ),
        }
    )
    return record


def assemble_recommendation(
    *,
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    subgraphs: list[RetrievedSubgraph],
    selection: dict[str, list[str]],
) -> dict[str, Any]:
    compound = corpus.resolve_compound(request.compound)
    compound_id = str(compound["node_id"])
    selected_evidence = set(selection["evidence_ids"])
    parameters, parameter_rejections = _parameter_candidate_derivation(
        corpus,
        selected_evidence,
        compound_id,
        request,
    )
    citations = [
        _citation_record(
            corpus=corpus,
            evidence_id=evidence_id,
            compound=compound,
            parameter_candidates=parameters,
        )
        for evidence_id in selection["evidence_ids"]
        if evidence_id in corpus.node_by_id
    ]
    model_rows = []
    for model_id in selection["public_model_ids"]:
        model = corpus.node_by_id.get(model_id)
        if model is None or model.get("node_type") != "PBPKModel":
            continue
        properties = model.get("properties") or {}
        if properties.get("reusable_candidate") is not True:
            continue
        evidence_ids = sorted(
            {
                str((edge.get("provenance") or {}).get("evidence_id"))
                for edge in corpus.edges
                if (
                    str(edge.get("subject_id")) == compound_id
                    and str(edge.get("object_id")) == model_id
                    and edge.get("predicate") == "DESCRIBED_BY_MODEL"
                    and _quality_ok(edge.get("provenance") or {})
                    and str(
                        (edge.get("provenance") or {}).get("evidence_id")
                        or ""
                    )
                    in selected_evidence
                )
            }
            - {""}
        )
        if not evidence_ids:
            continue
        model_rows.append(
            {
                "model_id": model_id,
                "name": model.get("preferred_name"),
                "software": properties.get("software"),
                "model_checksum": properties.get("model_checksum"),
                "availability_status": properties.get("availability_status"),
                "evidence_ids": evidence_ids,
            }
        )
    component_items = []
    dropped_components: list[str] = []
    for component in selection["component_ids"]:
        component_evidence = [
            row
            for row in citations
            if component
            in _components_for_evidence(
                corpus,
                row["evidence_id"],
                compound_id=compound_id,
            )
        ]
        if component == "public_model_template" and model_rows:
            model_evidence = {
                evidence_id
                for model in model_rows
                for evidence_id in model["evidence_ids"]
            }
            component_evidence = [
                row
                for row in citations
                if str(row["evidence_id"]) in model_evidence
            ]
        if not component_evidence:
            dropped_components.append(component)
            continue
        component_items.append(
            {
                "component": component,
                "status": "evidence_supported_candidate",
                "evidence_ids": [
                    str(row["evidence_id"]) for row in component_evidence
                ],
                "source_count": len(
                    {str(row["source_id"]) for row in component_evidence}
                ),
            }
        )
    supported_components = {
        str(row["component"])
        for row in component_items
        if row["evidence_ids"]
    }
    objective = canonical_objective(request.objective)
    required_components = REQUIRED_COMPONENTS.get(
        objective,
        REQUIRED_COMPONENTS["base_pk"],
    )
    evidence_gaps = sorted(
        f"missing_component:{component}"
        for component in required_components - supported_components
    )
    if objective == "public_model_reuse" and not model_rows:
        evidence_gaps.append("no_reusable_public_model")
    llm_selected_gaps = sorted(set(selection["gap_codes"]))
    selection_normalizations = []
    if dropped_components:
        selection_normalizations.append(
            {
                "code": "unsupported_component_selection_removed",
                "values": sorted(dropped_components),
            }
        )
    if llm_selected_gaps != evidence_gaps:
        selection_normalizations.append(
            {
                "code": "evidence_gaps_recomputed_from_grounded_support",
                "llm_selected": llm_selected_gaps,
                "deterministic": evidence_gaps,
            }
        )
    supported_node_ids = _reachable_selected_node_ids(
        corpus=corpus,
        evidence_ids=selected_evidence,
        compound_id=compound_id,
    )
    paths = sorted(
        {
            str(path_id)
            for item in subgraphs
            for path_id in item.row.get("path_ids") or ()
        }
    )
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "citation_context_version": CITATION_CONTEXT_VERSION,
        "status": "recommendation_draft",
        "resolved_modeling_request": request.model_dump(mode="json"),
        "compound": {
            "compound_id": compound_id,
            "preferred_name": compound.get("preferred_name"),
            "standard_identifiers": compound.get("standard_identifiers") or {},
        },
        "selected_public_models": model_rows,
        "recommended_model_structure": {
            "whole_body_or_reduced_pbpk": (
                request.whole_body_or_reduced_pbpk or "unspecified"
            ),
            "parent_or_parent_metabolite": (
                request.parent_or_parent_metabolite or "parent"
            ),
            "required_compartments": sorted(
                {
                    str(node.get("preferred_name"))
                    for node_id in supported_node_ids
                    if (
                        (node := corpus.node_by_id.get(node_id)) is not None
                        and node.get("node_type") in {"Organ", "Tissue"}
                    )
                }
            ),
        },
        "component_recommendations": component_items,
        "absorption_strategy": _component_status(component_items, "absorption"),
        "distribution_strategy": _component_status(component_items, "distribution"),
        "metabolism_pathways": _named_nodes(
            corpus,
            supported_node_ids,
            {"Enzyme"},
        ),
        "transport_pathways": _named_nodes(
            corpus,
            supported_node_ids,
            {"Transporter"},
        ),
        "excretion_strategy": _component_status(component_items, "excretion"),
        "ddi_mechanisms": _event_rows(
            corpus=corpus,
            evidence_ids=selected_evidence,
            compound_id=compound_id,
            event_type="DDIEvent",
            request=request,
        ),
        "special_population_adjustments": (
            _named_nodes(
                corpus,
                supported_node_ids,
                {"Population"},
            )
            if objective
            in {
                "renal_impairment",
                "hepatic_impairment",
                "pediatric",
                "pregnancy",
                "special_population",
            }
            else []
        ),
        "parameter_value_gate_version": PARAMETER_VALUE_GATE_VERSION,
        "parameter_candidates": parameters,
        "parameter_candidate_rejections": parameter_rejections,
        "calibration_datasets": list(request.calibration_datasets),
        "validation_datasets": list(request.validation_datasets),
        "sensitivity_targets": sorted(
            {
                str(row["parameter"])
                for row in parameters
                if row.get("parameter")
            }
        ),
        "uncertainty_parameters": sorted(
            {
                str(row["parameter"])
                for row in parameters
                if row.get("value_low") is not None
                or row.get("value_high") is not None
            }
        ),
        "evidence_gaps": evidence_gaps,
        "llm_selected_evidence_gaps": llm_selected_gaps,
        "selection_normalizations": selection_normalizations,
        "conflicts": _conflicts(corpus, selected_evidence),
        "citations": citations,
        "retrieved_subgraphs": [
            {
                "subgraph_id": item.subgraph_id,
                "hop": item.hop,
                "score": round(item.score, 6),
                "score_components": {
                    key: round(value, 6)
                    for key, value in sorted(item.score_components.items())
                },
            }
            for item in subgraphs
        ],
        "traversed_path_ids": paths,
        "overall_confidence": _recommendation_confidence(
            objective,
            component_items,
            citations,
        ),
        "confidence_interpretation": (
            "Deterministic required-component coverage and source-diversity "
            "score; not a calibrated probability of model validity."
        ),
        "scientific_use_status": "requirements_evidence_only",
        "scientific_use_interpretation": (
            "These are evidence-supported PBPK requirements. They are not "
            "calibrated, simulation-qualified, or scientifically validated "
            "model-component selections."
        ),
        "generated_from_auto_validated_only": True,
    }


def _components_for_evidence(
    corpus: GraphPromptCorpus,
    evidence_id: str,
    *,
    compound_id: str | None = None,
) -> set[str]:
    edges = list(corpus.edges_by_evidence.get(evidence_id, ()))
    if compound_id is not None:
        target_event_ids = {
            str(edge["subject_id"])
            for edge in edges
            if edge.get("predicate") == "OF_COMPOUND"
            and str(edge.get("object_id") or "") == compound_id
            and _quality_ok(edge.get("provenance") or {})
        }
        target_event_ids.update(
            str(event["event_id"])
            for event in corpus.events
            if str(event.get("evidence_id") or "") == evidence_id
            and any(
                str(edge.get("subject_id") or "") == compound_id
                and str(edge.get("object_id") or "") == str(event["event_id"])
                and edge.get("predicate")
                in {"VICTIM_IN", "PERPETRATOR_IN"}
                and _quality_ok(edge.get("provenance") or {})
                for edge in edges
            )
        )
        edges = [
            edge
            for edge in edges
            if str(edge.get("subject_id") or "") in target_event_ids
            or str(edge.get("object_id") or "") in target_event_ids
            or (
                edge.get("predicate") == "DESCRIBED_BY_MODEL"
                and str(edge.get("subject_id") or "") == compound_id
            )
        ]
    predicates = {str(edge["predicate"]) for edge in edges}
    components = {
        component
        for component, relations in COMPONENT_RELATIONS.items()
        if relations & predicates
    }
    parameter_names = {
        str(corpus.node_by_id[str(edge["object_id"])].get("preferred_name") or "").lower()
        for edge in edges
        if edge["predicate"] in {"MEASURES", "AFFECTS_PARAMETER"}
        and str(edge["object_id"]) in corpus.node_by_id
    }
    if compound_id is None or target_event_ids:
        evidence_node = corpus.node_by_id.get(evidence_id) or {}
        preferred = str(evidence_node.get("preferred_name") or "").lower()
        parameter_names.add(preferred)
    for component, patterns in PARAMETER_COMPONENT_PATTERNS.items():
        if any(
            any(pattern in parameter for pattern in patterns)
            for parameter in parameter_names
        ):
            components.add(component)
    return components


def _component_status(
    rows: list[dict[str, Any]],
    component: str,
) -> dict[str, Any]:
    row = next(
        (value for value in rows if value["component"] == component),
        None,
    )
    if row is None:
        return {
            "status": "insufficient_evidence",
            "evidence_ids": [],
        }
    return {
        "status": row["status"],
        "evidence_ids": row["evidence_ids"],
    }


def _reachable_selected_node_ids(
    *,
    corpus: GraphPromptCorpus,
    evidence_ids: set[str],
    compound_id: str,
) -> set[str]:
    edges = {
        str(edge["edge_id"]): edge
        for evidence_id in evidence_ids
        for edge in corpus.edges_by_evidence.get(evidence_id, ())
        if _quality_ok(edge.get("provenance") or {})
        and edge.get("predicate") not in {
            "CONTRADICTS",
            "DERIVED_FROM",
            "REPORTS",
            "SUPPORTED_BY",
        }
    }
    reachable = {compound_id}
    changed = True
    while changed:
        changed = False
        for edge in edges.values():
            subject_id = str(edge["subject_id"])
            object_id = str(edge["object_id"])
            if subject_id in reachable and object_id not in reachable:
                reachable.add(object_id)
                changed = True
            elif object_id in reachable and subject_id not in reachable:
                reachable.add(subject_id)
                changed = True
    return reachable


def _named_nodes(
    corpus: GraphPromptCorpus,
    supported_node_ids: set[str],
    node_types: set[str],
) -> list[dict[str, Any]]:
    rows = {}
    for node_id in supported_node_ids:
        node = corpus.node_by_id.get(node_id)
        if node is not None and node.get("node_type") in node_types:
            rows[node_id] = {
                "node_id": node["node_id"],
                "node_type": node["node_type"],
                "preferred_name": node.get("preferred_name"),
                "standard_identifiers": node.get("standard_identifiers") or {},
            }
    return [rows[key] for key in sorted(rows)]


def _event_rows(
    *,
    corpus: GraphPromptCorpus,
    evidence_ids: set[str],
    compound_id: str,
    event_type: str,
    request: ModelingRequest,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    objective = canonical_objective(request.objective)
    counterpart_name = (
        request.perpetrator_drug
        if objective == "ddi_victim"
        else request.victim_drug
        if objective == "ddi_perpetrator"
        else None
    )
    counterpart_id = None
    if counterpart_name:
        try:
            counterpart_id = str(corpus.resolve_compound(counterpart_name)["node_id"])
        except LookupError:
            return []
    for event in corpus.events:
        evidence_id = str(event.get("evidence_id") or "")
        if evidence_id not in evidence_ids or event.get("event_type") != event_type:
            continue
        event_id = str(event["event_id"])
        argument_edges = [
            corpus.edge_by_id[str(edge_id)]
            for edge_id in event.get("argument_edge_ids") or ()
            if str(edge_id) in corpus.edge_by_id
            and _quality_ok(
                corpus.edge_by_id[str(edge_id)].get("provenance") or {}
            )
            and str(
                (
                    corpus.edge_by_id[str(edge_id)].get("provenance") or {}
                ).get("evidence_id")
                or ""
            )
            == evidence_id
        ]
        role_nodes: dict[str, set[str]] = {
            "victim": set(),
            "perpetrator": set(),
        }
        for edge in argument_edges:
            role = {
                "VICTIM_IN": "victim",
                "PERPETRATOR_IN": "perpetrator",
            }.get(str(edge.get("predicate")))
            if role is None:
                continue
            subject_id = str(edge["subject_id"])
            object_id = str(edge["object_id"])
            if object_id == event_id and subject_id in corpus.compounds:
                role_nodes[role].add(subject_id)
            elif subject_id == event_id and object_id in corpus.compounds:
                role_nodes[role].add(object_id)
        target_roles = sorted(
            role for role, node_ids in role_nodes.items() if compound_id in node_ids
        )
        if not target_roles:
            continue
        if objective == "ddi_victim" and "victim" not in target_roles:
            continue
        if objective == "ddi_perpetrator" and "perpetrator" not in target_roles:
            continue
        if counterpart_id:
            required_role = (
                "perpetrator" if objective == "ddi_victim" else "victim"
            )
            if counterpart_id not in role_nodes[required_role]:
                continue
        participants = []
        for role in ("victim", "perpetrator"):
            for node_id in sorted(role_nodes[role]):
                node = corpus.node_by_id[node_id]
                participants.append(
                    {
                        "compound_id": node_id,
                        "compound_name": node.get("preferred_name"),
                        "role": role,
                    }
                )
        rows[event_id] = {
            "event_id": event_id,
            "event_type": event["event_type"],
            "evidence_id": evidence_id,
            "context": event.get("context") or {},
            "trigger_text": event.get("trigger_text"),
            "target_compound_roles": target_roles,
            "participants": participants,
        }
    return [rows[key] for key in sorted(rows)]


def _conflicts(
    corpus: GraphPromptCorpus,
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for edge in corpus.edges:
        provenance = edge.get("provenance") or {}
        if edge.get("predicate") == "CONTRADICTS" and (
            str(edge.get("subject_id") or "") in evidence_ids
            or str(edge.get("object_id") or "") in evidence_ids
            or str(provenance.get("evidence_id") or "") in evidence_ids
        ):
            rows.append(
                {
                    "edge_id": edge["edge_id"],
                    "subject_id": edge["subject_id"],
                    "object_id": edge["object_id"],
                    "evidence_id": provenance.get("evidence_id"),
                }
            )
    return sorted(rows, key=lambda row: str(row["edge_id"]))


def _recommendation_confidence(
    objective: str,
    components: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> float:
    required = REQUIRED_COMPONENTS.get(objective, REQUIRED_COMPONENTS["base_pk"])
    supported_components = {
        str(row["component"])
        for row in components
        if row.get("evidence_ids")
    }
    supported = len(required & supported_components) / max(1, len(required))
    source_diversity = min(
        1.0,
        len({str(row["source_id"]) for row in citations}) / 4.0,
    )
    return round(0.7 * supported + 0.3 * source_diversity, 6)


def validate_recommendation(
    *,
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    recommendation: dict[str, Any],
    allowed_subgraph_ids: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    resolved_compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
    recommendation_compound_id = str(
        (recommendation.get("compound") or {}).get("compound_id") or ""
    )
    if recommendation_compound_id != resolved_compound_id:
        errors.append("recommendation_compound_mismatch")
    if recommendation.get("scientific_use_status") != "requirements_evidence_only":
        errors.append("scientific_use_status_mismatch")
    citations = recommendation.get("citations")
    if not isinstance(citations, list):
        errors.append("citations_missing")
        citations = []
    citation_ids = {
        str(row.get("evidence_id") or "")
        for row in citations
        if isinstance(row, dict)
    }
    for citation in citations:
        evidence_id = str(citation.get("evidence_id") or "")
        node = corpus.node_by_id.get(evidence_id)
        if node is None or node.get("node_type") != "Evidence":
            errors.append(f"citation_unknown_evidence:{evidence_id}")
            continue
        if (node.get("properties") or {}).get("quality_status") != "auto_validated":
            errors.append(f"citation_not_auto_validated:{evidence_id}")
        reference = corpus.evidence_record(evidence_id)
        for field in (
            "source_id",
            "source_location",
            "supporting_text",
            "source_text_hash",
        ):
            if citation.get(field) != reference.get(field):
                errors.append(f"citation_{field}_mismatch:{evidence_id}")
    selected_subgraphs = {
        str(row.get("subgraph_id") or "")
        for row in recommendation.get("retrieved_subgraphs") or ()
    }
    if not selected_subgraphs <= allowed_subgraph_ids:
        errors.append("retrieved_subgraph_outside_runtime_selection")
    selected_subgraph_rows = [
        RetrievedSubgraph(
            subgraph_id=subgraph_id,
            hop=int(corpus.subgraphs[subgraph_id].get("hop_limit") or 0),
            score=0.0,
            score_components={},
            row=corpus.subgraphs[subgraph_id],
            vector=None,
        )
        for subgraph_id in sorted(selected_subgraphs)
        if subgraph_id in corpus.subgraphs
    ]
    objective_evidence = _objective_evidence_ids(
        corpus=corpus,
        request=request,
        subgraphs=selected_subgraph_rows,
        compound_id=resolved_compound_id,
    )
    wrong_compound_citations = citation_ids - objective_evidence
    if wrong_compound_citations:
        errors.append(
            "citation_not_grounded_to_target_compound_and_objective:"
            + ",".join(sorted(wrong_compound_citations))
        )
    expected_parameters, expected_parameter_rejections = _parameter_candidate_derivation(
        corpus,
        citation_ids,
        resolved_compound_id,
        request,
    )
    resolved_compound = corpus.compounds[resolved_compound_id]
    for citation in citations:
        evidence_id = str(citation.get("evidence_id") or "")
        if evidence_id not in corpus.node_by_id:
            continue
        expected_citation = _citation_record(
            corpus=corpus,
            evidence_id=evidence_id,
            compound=resolved_compound,
            parameter_candidates=expected_parameters,
        )
        for field in (
            "target_compound",
            "compound_attribution",
            "structured_measurements",
            "context_status",
        ):
            if citation.get(field) != expected_citation.get(field):
                errors.append(f"citation_{field}_mismatch:{evidence_id}")
        if not (
            citation.get("compound_attribution") or {}
        ).get("edge_ids"):
            errors.append(f"citation_missing_compound_binding:{evidence_id}")
    evidence_in_subgraphs = {
        str(citation.get("evidence_id") or "")
        for subgraph_id in selected_subgraphs
        for citation in (
            corpus.subgraphs.get(subgraph_id, {}).get("citations") or ()
        )
    }
    unsupported_citations = citation_ids - evidence_in_subgraphs
    if unsupported_citations:
        errors.append(
            "citation_without_retrieved_path:"
            + ",".join(sorted(unsupported_citations))
        )
    component_rows = recommendation.get("component_recommendations") or ()
    component_names: set[str] = set()
    for component in component_rows:
        name = str(component.get("component") or "")
        if component.get("status") != "evidence_supported_candidate":
            errors.append(f"component_scientific_status_mismatch:{name}")
        if name in component_names:
            errors.append(f"duplicate_component_recommendation:{name}")
        component_names.add(name)
        evidence_ids = set(map(str, component.get("evidence_ids") or ()))
        if not evidence_ids or not evidence_ids <= citation_ids:
            errors.append(f"component_without_selected_citation:{name}")
            continue
        if name == "public_model_template":
            model_evidence = {
                str(evidence_id)
                for model in recommendation.get("selected_public_models") or ()
                for evidence_id in model.get("evidence_ids") or ()
            }
            if not evidence_ids <= model_evidence:
                errors.append("public_model_component_without_direct_model_evidence")
        elif any(
            name
            not in _components_for_evidence(
                corpus,
                evidence_id,
                compound_id=resolved_compound_id,
            )
            for evidence_id in evidence_ids
        ):
            errors.append(f"component_evidence_relation_mismatch:{name}")
    objective = canonical_objective(request.objective)
    required_components = REQUIRED_COMPONENTS.get(
        objective,
        REQUIRED_COMPONENTS["base_pk"],
    )
    expected_gaps = sorted(
        f"missing_component:{component}"
        for component in required_components - component_names
    )
    if objective == "public_model_reuse" and not (
        recommendation.get("selected_public_models") or ()
    ):
        expected_gaps.append("no_reusable_public_model")
    if sorted(set(recommendation.get("evidence_gaps") or ())) != expected_gaps:
        errors.append("evidence_gap_state_inconsistent_with_grounded_components")
    supported_node_ids = _reachable_selected_node_ids(
        corpus=corpus,
        evidence_ids=citation_ids,
        compound_id=resolved_compound_id,
    )
    for field, node_types in (
        ("metabolism_pathways", {"Enzyme"}),
        ("transport_pathways", {"Transporter"}),
        ("special_population_adjustments", {"Population"}),
    ):
        for row in recommendation.get(field) or ():
            node_id = str(row.get("node_id") or "")
            node = corpus.node_by_id.get(node_id)
            if (
                node_id not in supported_node_ids
                or node is None
                or node.get("node_type") not in node_types
            ):
                errors.append(f"{field}_not_grounded_to_selected_evidence:{node_id}")
    if objective not in {
        "renal_impairment",
        "hepatic_impairment",
        "pediatric",
        "pregnancy",
        "special_population",
    } and recommendation.get("special_population_adjustments"):
        errors.append("special_population_adjustment_outside_population_objective")
    structure = recommendation.get("recommended_model_structure") or {}
    for name in structure.get("required_compartments") or ():
        matched = [
            node_id
            for node_id in supported_node_ids
            if (
                (node := corpus.node_by_id.get(node_id)) is not None
                and node.get("node_type") in {"Organ", "Tissue"}
                and str(node.get("preferred_name") or "") == str(name)
            )
        ]
        if not matched:
            errors.append(f"compartment_not_grounded_to_selected_evidence:{name}")
    expected_ddi = _event_rows(
        corpus=corpus,
        evidence_ids=citation_ids,
        compound_id=resolved_compound_id,
        event_type="DDIEvent",
        request=request,
    )
    expected_ddi_ids = {str(row["event_id"]) for row in expected_ddi}
    actual_ddi_ids = {
        str(row.get("event_id") or "")
        for row in recommendation.get("ddi_mechanisms") or ()
    }
    if actual_ddi_ids != expected_ddi_ids:
        errors.append("ddi_mechanisms_not_exactly_target_grounded")
    numeric_count = 0
    unsupported_numeric_count = 0
    actual_parameter_candidates = recommendation.get("parameter_candidates") or ()
    if actual_parameter_candidates != expected_parameters:
        errors.append("parameter_candidates_not_exactly_graph_derived")
    if recommendation.get("parameter_value_gate_version") != PARAMETER_VALUE_GATE_VERSION:
        errors.append("parameter_value_gate_version_mismatch")
    if (
        recommendation.get("parameter_candidate_rejections") or []
    ) != expected_parameter_rejections:
        errors.append("parameter_candidate_rejections_not_exactly_graph_derived")
    for candidate in actual_parameter_candidates:
        evidence_id = str(candidate.get("evidence_id") or "")
        if evidence_id not in citation_ids:
            errors.append(f"parameter_without_citation:{evidence_id}")
        numeric_fields = {
            key: candidate.get(key)
            for key in ("value", "value_low", "value_high")
            if candidate.get(key) is not None
        }
        if numeric_fields:
            numeric_count += 1
            source_text = str(candidate.get("supporting_text") or "")
            reported_fields = {
                key: candidate.get(f"reported_{key}")
                for key in ("value", "value_low", "value_high")
                if candidate.get(key) is not None
            }
            unsupported_fields = [
                (key, value)
                for key, value in reported_fields.items()
                if value is None
                or not _numeric_in_text(float(value), source_text)
            ]
            unsupported_fields.extend(
                (key, value)
                for key, value in numeric_fields.items()
                if key not in reported_fields
                if not _numeric_in_text(float(value), source_text)
            )
            if unsupported_fields:
                unsupported_numeric_count += 1
            for key, value in unsupported_fields:
                errors.append(
                    f"reported_numeric_value_not_verbatim_in_source:"
                    f"{evidence_id}:{key}:{value}"
                )
        if candidate.get("parameter_value_gate_version") != PARAMETER_VALUE_GATE_VERSION:
            errors.append(f"parameter_value_gate_missing:{evidence_id}")
        if candidate.get("compound_id") != resolved_compound_id:
            errors.append(f"parameter_compound_mismatch:{evidence_id}")
    for model in recommendation.get("selected_public_models") or ():
        model_id = str(model.get("model_id") or "")
        node = corpus.node_by_id.get(model_id)
        if (
            node is None
            or node.get("node_type") != "PBPKModel"
            or (node.get("properties") or {}).get("reusable_candidate") is not True
        ):
            errors.append(f"public_model_not_reusable:{model_id}")
            continue
        model_evidence_ids = set(map(str, model.get("evidence_ids") or ()))
        exact_direct_evidence = {
            str((edge.get("provenance") or {}).get("evidence_id") or "")
            for edge in corpus.edges
            if edge.get("predicate") == "DESCRIBED_BY_MODEL"
            and str(edge.get("subject_id") or "") == resolved_compound_id
            and str(edge.get("object_id") or "") == model_id
            and _quality_ok(edge.get("provenance") or {})
        } & citation_ids
        if not model_evidence_ids or model_evidence_ids != exact_direct_evidence:
            errors.append(f"public_model_not_directly_grounded_to_target:{model_id}")
    request_dump = request.model_dump(mode="json")
    if recommendation.get("resolved_modeling_request") != request_dump:
        errors.append("resolved_request_mismatch")
    if recommendation.get("generated_from_auto_validated_only") is not True:
        errors.append("auto_validated_only_flag_missing")
    if recommendation.get("citation_context_version") != CITATION_CONTEXT_VERSION:
        errors.append("citation_context_version_mismatch")
    expected_confidence = _recommendation_confidence(
        objective,
        list(component_rows),
        citations,
    )
    if recommendation.get("overall_confidence") != expected_confidence:
        errors.append("overall_confidence_not_reproducible")
    repairable_prefixes = (
        "citation_source_id_mismatch:",
        "citation_source_location_mismatch:",
        "citation_supporting_text_mismatch:",
        "citation_source_text_hash_mismatch:",
        "citation_target_compound_mismatch:",
        "citation_compound_attribution_mismatch:",
        "citation_structured_measurements_mismatch:",
        "citation_context_status_mismatch:",
        "numeric_value_not_verbatim_in_source:",
        "parameter_without_citation:",
    )
    status = (
        "recommendation_validated"
        if not errors
        else "recommendation_repairable"
        if all(error.startswith(repairable_prefixes) for error in errors)
        else "recommendation_quarantined"
    )
    return {
        "validator_version": VALIDATOR_VERSION,
        "status": status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "citation_count": len(citation_ids),
        "numeric_parameter_count": numeric_count,
        "citation_provenance_completeness": (
            1.0
            if citations
            and not any(error.startswith("citation_") for error in errors)
            else 0.0
        ),
        "unsupported_numeric_parameter_rate": (
            unsupported_numeric_count / max(1, numeric_count)
        ),
        "quarantined_evidence_leakage": sum(
            error.startswith("citation_not_auto_validated") for error in errors
        ),
    }


def _numeric_in_text(value: float, text: str) -> bool:
    for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text):
        try:
            if abs(float(token) - value) <= max(1e-9, abs(value) * 1e-6):
                return True
        except ValueError:
            continue
    return False


def dataset_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(row["split"]) for row in examples)
    compounds_by_split: dict[str, set[str]] = defaultdict(set)
    objectives = Counter()
    for row in examples:
        compounds_by_split[str(row["split"])].add(str(row["compound_id"]))
        objectives[str(row["modeling_request"]["objective"])] += 1
    overlaps = {}
    for left in ("train", "validation", "test"):
        for right in ("train", "validation", "test"):
            if left >= right:
                continue
            overlaps[f"{left}:{right}"] = len(
                compounds_by_split[left] & compounds_by_split[right]
            )
    return {
        "example_count": len(examples),
        "split_counts": dict(sorted(by_split.items())),
        "compound_counts": {
            key: len(value)
            for key, value in sorted(compounds_by_split.items())
        },
        "compound_overlap_counts": overlaps,
        "objective_counts": dict(sorted(objectives.items())),
    }
