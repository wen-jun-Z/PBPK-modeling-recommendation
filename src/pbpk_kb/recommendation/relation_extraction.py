from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pbpk_kb.graph.final_evidence_graph import extract_graph_facts
from pbpk_kb.recommendation.entity_extraction import EntityCatalog, normalize_text
from pbpk_kb.recommendation.kg_models import (
    EntityMention,
    KGEdge,
    KGEvent,
    KGNode,
    Provenance,
    RejectedRelation,
    stable_id,
)


EXTRACTOR_VERSION = "pbpk-event-relation-extractor-1.0.0"


@dataclass
class ExtractionResult:
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    events: list[KGEvent] = field(default_factory=list)
    rejected: list[RejectedRelation] = field(default_factory=list)


@dataclass(frozen=True)
class CompoundRef:
    node_id: str
    text: str
    role: str | None


class PBPKRelationExtractor:
    def __init__(
        self,
        catalog: EntityCatalog,
        ddi_decisions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.ddi_decisions = ddi_decisions or {}

    def extract(self, row: dict[str, Any], mentions: list[EntityMention]) -> ExtractionResult:
        if row.get("quality_status") != "auto_validated":
            raise ValueError("Only auto_validated evidence may enter relation extraction.")
        result = ExtractionResult()
        result.nodes.extend(self._evidence_and_source_nodes(row))
        result.edges.extend(self._provenance_edges(row))
        evidence_type = str(row.get("evidence_type") or "")

        if evidence_type in {"clinical_pk", "physicochemical", "adme", "special_population"}:
            measurement = self._measurement_event(row, mentions)
            _extend(result, measurement)
        if evidence_type == "ddi":
            ddi = self._ddi_event(row, mentions)
            _extend(result, ddi)
        if evidence_type in {"adme", "ddi"}:
            targets = self._target_relations(row)
            _extend(result, targets)
        if evidence_type == "public_pbpk_model":
            model = self._public_model_relations(row, mentions)
            _extend(result, model)
        return result

    def _measurement_event(
        self,
        row: dict[str, Any],
        mentions: list[EntityMention],
    ) -> ExtractionResult:
        result = ExtractionResult()
        compound = _target_compound_ref(row)
        parameter = _first_linked(mentions, "PKParameter")
        if compound is None:
            result.rejected.append(_reject(row, "OF_COMPOUND", "target compound was not uniquely linked"))
            return result
        if parameter is None:
            parameter_name = str(row.get("parameter") or "").strip()
            entry, _, _ = self.catalog.link("PKParameter", parameter_name)
            if entry:
                parameter = _synthetic_mention(row, entry.node_id, "PKParameter", parameter_name)
        if parameter is None:
            result.rejected.append(_reject(row, "MEASURES", "PK parameter was not uniquely linked"))
            return result

        event_type = _measurement_event_type(row)
        event_id = stable_id("event", row["evidence_id"], event_type)
        event_node = KGNode(
            node_id=event_id,
            node_type=event_type,
            preferred_name=f"{event_type} for {compound.text}",
            properties={"evidence_type": row.get("evidence_type")},
        )
        result.nodes.append(event_node)
        provenance = _provenance(row, "structured_event_extraction", 0.99)
        edges = [
            _edge("Evidence", str(row["evidence_id"]), "REPORTS", event_type, event_id, provenance),
            _edge(event_type, event_id, "OF_COMPOUND", "Compound", compound.node_id, provenance),
            _edge(event_type, event_id, "MEASURES", "PKParameter", parameter.linked_node_id, provenance),
            _edge(event_type, event_id, "SUPPORTED_BY", "Evidence", str(row["evidence_id"]), provenance),
        ]
        if event_type == "PKMeasurement":
            edges.append(
                _edge(
                    "Evidence",
                    str(row["evidence_id"]),
                    "HAS_MEASUREMENT",
                    "PKMeasurement",
                    event_id,
                    provenance,
                )
            )
        elif event_type == "AbsorptionEvent":
            edges.append(
                _edge(
                    "Compound",
                    compound.node_id,
                    "ABSORBED_VIA",
                    "AbsorptionEvent",
                    event_id,
                    provenance,
                )
            )
        elif event_type == "ExcretionEvent":
            edges.append(
                _edge(
                    "Compound",
                    compound.node_id,
                    "EXCRETED_VIA",
                    "ExcretionEvent",
                    event_id,
                    provenance,
                )
            )
        value_node = _parameter_value_node(row, event_id)
        if value_node is not None:
            result.nodes.append(value_node)
            edges.append(_edge(event_type, event_id, "HAS_VALUE", "ParameterValue", value_node.node_id, provenance))
            unit_node_id = _linked_or_structured_node_id(
                row,
                mentions,
                self.catalog,
                entity_type="Unit",
                field_names=("canonical_unit", "unit"),
            )
            if unit_node_id is not None:
                edges.append(
                    _edge(
                        "ParameterValue",
                        value_node.node_id,
                        "HAS_UNIT",
                        "Unit",
                        unit_node_id,
                        provenance,
                    )
                )

        population_node = _context_node(row, "Population", "population")
        if population_node is not None:
            result.nodes.append(population_node)
            edges.append(_edge(event_type, event_id, "STUDIED_IN", "Population", population_node.node_id, provenance))

        regimen = _dose_regimen(row, mentions, self.catalog)
        if regimen is not None:
            regimen_node, regimen_edges, extra_nodes = regimen
            result.nodes.extend([regimen_node, *extra_nodes])
            edges.extend(regimen_edges)
            edges.append(_edge(event_type, event_id, "UNDER_REGIMEN", "DoseRegimen", regimen_node.node_id, provenance))

        result.edges.extend(edges)
        result.events.append(
            KGEvent(
                event_id=event_id,
                event_type=event_type,
                evidence_id=str(row["evidence_id"]),
                trigger_text=_event_trigger(row),
                argument_edge_ids=[edge.edge_id for edge in edges],
                context=_event_context(row),
                extraction_method="structured_event_extraction",
                extractor_version=EXTRACTOR_VERSION,
                confidence=0.99,
                validation_status="validated",
            )
        )
        return result

    def _ddi_event(self, row: dict[str, Any], mentions: list[EntityMention]) -> ExtractionResult:
        decision = self.ddi_decisions.get(str(row["evidence_id"]))
        if decision is not None:
            return self._validated_ddi_events(row, mentions, decision)

        result = ExtractionResult()
        role = str(row.get("compound_role") or "")
        target = _target_compound_ref(row)
        compounds = _linked_compounds(mentions)
        victim: CompoundRef | None = None
        perpetrator: CompoundRef | None = None

        if target is not None and role == "victim_drug":
            victim = target
            perpetrator = _as_compound_ref(
                _counterpart_from_metadata(row, mentions, "perpetrator_drug")
                or _unique_other_compound(compounds, target.node_id, "perpetrator_drug")
            )
        elif target is not None and role == "perpetrator_drug":
            perpetrator = target
            victim = _as_compound_ref(
                _counterpart_from_metadata(row, mentions, "victim_drug")
                or _unique_other_compound(compounds, target.node_id, "victim_drug")
            )
        else:
            victim = _as_compound_ref(_role_mention(compounds, "victim_drug"))
            perpetrator = _as_compound_ref(_role_mention(compounds, "perpetrator_drug"))

        if victim is None or perpetrator is None or victim.node_id == perpetrator.node_id:
            reasons = []
            if victim is None:
                reasons.append("victim compound was not uniquely linked")
            if perpetrator is None:
                reasons.append("perpetrator compound was not uniquely linked")
            if victim and perpetrator and victim.node_id == perpetrator.node_id:
                reasons.append("victim and perpetrator resolve to the same compound")
            result.rejected.append(_reject(row, "DDIEvent", *reasons))
            return result

        event_id = stable_id("event", row["evidence_id"], "DDIEvent")
        result.nodes.append(
            KGNode(
                node_id=event_id,
                node_type="DDIEvent",
                preferred_name=f"{victim.text} interaction with {perpetrator.text}",
                properties={"interaction_type": row.get("parameter") or row.get("interaction_type")},
            )
        )
        provenance = _provenance(row, "ddi_role_event_extraction", 0.95)
        edges = [
            _edge("Evidence", str(row["evidence_id"]), "REPORTS", "DDIEvent", event_id, provenance),
            _edge("Compound", perpetrator.node_id, "PERPETRATOR_IN", "DDIEvent", event_id, provenance),
            _edge("Compound", victim.node_id, "VICTIM_IN", "DDIEvent", event_id, provenance),
            _edge("DDIEvent", event_id, "SUPPORTED_BY", "Evidence", str(row["evidence_id"]), provenance),
        ]
        for mention in mentions:
            if mention.linked_node_id and mention.entity_type in {"Enzyme", "Transporter", "Protein"}:
                object_type = mention.entity_type
                if object_type == "Protein" and mention.linked_node_id.startswith("Enzyme:"):
                    object_type = "Enzyme"
                elif object_type == "Protein" and mention.linked_node_id.startswith("Transporter:"):
                    object_type = "Transporter"
                edges.append(_edge("DDIEvent", event_id, "MEDIATED_BY", object_type, mention.linked_node_id, provenance))
        parameter = _first_linked(mentions, "PKParameter")
        if parameter:
            edges.append(
                _edge("DDIEvent", event_id, "AFFECTS_PARAMETER", "PKParameter", parameter.linked_node_id, provenance)
            )
        regimen = _dose_regimen(row, mentions, self.catalog)
        if regimen is not None:
            regimen_node, regimen_edges, extra_nodes = regimen
            result.nodes.extend([regimen_node, *extra_nodes])
            edges.extend(regimen_edges)
            edges.append(_edge("DDIEvent", event_id, "UNDER_REGIMEN", "DoseRegimen", regimen_node.node_id, provenance))
        result.edges.extend(edges)
        result.events.append(
            KGEvent(
                event_id=event_id,
                event_type="DDIEvent",
                evidence_id=str(row["evidence_id"]),
                trigger_text=_event_trigger(row),
                argument_edge_ids=[edge.edge_id for edge in edges],
                context=_event_context(row),
                extraction_method="ddi_role_event_extraction",
                extractor_version=EXTRACTOR_VERSION,
                confidence=0.95,
                validation_status="validated",
            )
        )
        return result

    def _validated_ddi_events(
        self,
        row: dict[str, Any],
        mentions: list[EntityMention],
        decision: dict[str, Any],
    ) -> ExtractionResult:
        result = ExtractionResult()
        if decision.get("source_text_hash") != row.get("source_text_hash"):
            result.rejected.append(_reject(row, "DDIEvent", "DDI decision source_text_hash mismatch"))
            return result
        if decision.get("status") != "accepted":
            reasons = [
                str(item) for item in decision.get("validation_errors") or []
                if str(item).strip()
            ]
            result.rejected.append(
                _reject(row, "DDIEvent", *(reasons or ["DDI decision did not pass independent validation"]))
            )
            return result
        events = decision.get("events")
        if not isinstance(events, list) or not events:
            result.rejected.append(_reject(row, "DDIEvent", "accepted DDI decision contains no events"))
            return result

        catalog_nodes = self.catalog.entries
        for event_index, event in enumerate(events):
            victim_id = str(event.get("victim_node_id") or "")
            perpetrator_id = str(event.get("perpetrator_node_id") or "")
            victim_entry = catalog_nodes.get(victim_id)
            perpetrator_entry = catalog_nodes.get(perpetrator_id)
            if (
                victim_entry is None
                or perpetrator_entry is None
                or victim_entry.node_type != "Compound"
                or perpetrator_entry.node_type != "Compound"
                or victim_id == perpetrator_id
            ):
                result.rejected.append(
                    _reject(row, "DDIEvent", f"validated event {event_index} has invalid compound endpoints")
                )
                continue
            supporting_quote = str(event.get("supporting_quote") or "")
            try:
                confidence = float(event.get("confidence"))
            except (TypeError, ValueError):
                result.rejected.append(
                    _reject(row, "DDIEvent", f"validated event {event_index} has invalid confidence")
                )
                continue
            event_id = stable_id(
                "event",
                row["evidence_id"],
                "DDIEvent",
                victim_id,
                perpetrator_id,
                supporting_quote,
            )
            result.nodes.append(
                KGNode(
                    node_id=event_id,
                    node_type="DDIEvent",
                    preferred_name=(
                        f"{victim_entry.preferred_name} interaction with "
                        f"{perpetrator_entry.preferred_name}"
                    ),
                    properties={
                        "interaction_effect": event.get("interaction_effect"),
                        "decision_method": decision.get("decision_method"),
                        "prompt_version": decision.get("prompt_version"),
                        "validator_version": decision.get("validator_version"),
                    },
                )
            )
            provenance = _provenance(
                row,
                str(decision.get("decision_method") or "validated_ddi_relation_decision"),
                confidence,
                supporting_text=supporting_quote,
            )
            edges = [
                _edge("Evidence", str(row["evidence_id"]), "REPORTS", "DDIEvent", event_id, provenance),
                _edge("Compound", perpetrator_id, "PERPETRATOR_IN", "DDIEvent", event_id, provenance),
                _edge("Compound", victim_id, "VICTIM_IN", "DDIEvent", event_id, provenance),
                _edge("DDIEvent", event_id, "SUPPORTED_BY", "Evidence", str(row["evidence_id"]), provenance),
            ]
            for mediator_id in event.get("mediator_node_ids") or []:
                mediator_type = str(mediator_id).split(":", 1)[0]
                entry = catalog_nodes.get(str(mediator_id))
                if entry is not None and mediator_type in {"Enzyme", "Transporter", "Protein"}:
                    edges.append(
                        _edge("DDIEvent", event_id, "MEDIATED_BY", mediator_type, str(mediator_id), provenance)
                    )
            for parameter_id in event.get("parameter_node_ids") or []:
                entry = catalog_nodes.get(str(parameter_id))
                if entry is not None and entry.node_type == "PKParameter":
                    edges.append(
                        _edge(
                            "DDIEvent",
                            event_id,
                            "AFFECTS_PARAMETER",
                            "PKParameter",
                            str(parameter_id),
                            provenance,
                        )
                    )
            if len(events) == 1 or _value_is_supported_by_quote(row, supporting_quote):
                value_node = _parameter_value_node(row, event_id)
                if value_node is not None:
                    result.nodes.append(value_node)
                    edges.append(
                        _edge(
                            "DDIEvent",
                            event_id,
                            "HAS_VALUE",
                            "ParameterValue",
                            value_node.node_id,
                            provenance,
                        )
                    )
                    unit_node_id = _linked_or_structured_node_id(
                        row,
                        mentions,
                        self.catalog,
                        entity_type="Unit",
                        field_names=("canonical_unit", "unit"),
                    )
                    if unit_node_id is not None:
                        edges.append(
                            _edge(
                                "ParameterValue",
                                value_node.node_id,
                                "HAS_UNIT",
                                "Unit",
                                unit_node_id,
                                provenance,
                            )
                        )
            population_node = _context_node(row, "Population", "population")
            if population_node is not None:
                result.nodes.append(population_node)
                edges.append(
                    _edge(
                        "DDIEvent",
                        event_id,
                        "STUDIED_IN",
                        "Population",
                        population_node.node_id,
                        provenance,
                    )
                )
            regimen = _dose_regimen(row, mentions, self.catalog)
            if regimen is not None:
                regimen_node, regimen_edges, extra_nodes = regimen
                result.nodes.extend([regimen_node, *extra_nodes])
                edges.extend(regimen_edges)
                edges.append(
                    _edge("DDIEvent", event_id, "UNDER_REGIMEN", "DoseRegimen", regimen_node.node_id, provenance)
                )
            result.edges.extend(edges)
            result.events.append(
                KGEvent(
                    event_id=event_id,
                    event_type="DDIEvent",
                    evidence_id=str(row["evidence_id"]),
                    trigger_text=supporting_quote,
                    argument_edge_ids=[edge.edge_id for edge in edges],
                    context=_event_context(row),
                    extraction_method=str(
                        decision.get("decision_method") or "validated_ddi_relation_decision"
                    ),
                    extractor_version=EXTRACTOR_VERSION,
                    confidence=confidence,
                    validation_status="validated",
                )
            )
        if not result.events:
            result.rejected.append(
                _reject(row, "DDIEvent", "no validated DDI event had valid ontology endpoints")
            )
        return result

    def _target_relations(self, row: dict[str, Any]) -> ExtractionResult:
        result = ExtractionResult()
        compound_id = f"Compound:{row['compound_id']}"
        for fact in extract_graph_facts(row):
            object_type = "Enzyme" if fact.target.target_type == "enzyme" else "Transporter"
            entry, _, _ = self.catalog.link(object_type, fact.target.matched_text)
            if entry is None:
                result.rejected.append(
                    _reject(
                        row,
                        fact.relationship_type.upper(),
                        f"target entity was not linked: {fact.target.matched_text}",
                    )
                )
                continue
            predicate = fact.relationship_type.upper()
            if predicate.startswith("NOT_"):
                result.rejected.append(
                    _reject(row, predicate, "negated scientific relation is retained as rejected, not as a positive fact")
                )
                continue
            provenance = _provenance(
                row,
                "dependency_rule_relation_extraction",
                fact.confidence,
                supporting_text=fact.evidence_sentence,
            )
            result.edges.extend(
                [
                    _edge("Compound", compound_id, predicate, object_type, entry.node_id, provenance),
                    _edge("Compound", compound_id, "SUPPORTED_BY", "Evidence", str(row["evidence_id"]), provenance),
                ]
            )
        return result

    def _public_model_relations(
        self,
        row: dict[str, Any],
        mentions: list[EntityMention],
    ) -> ExtractionResult:
        result = ExtractionResult()
        compound = _target_compound_ref(row)
        if compound is None:
            result.rejected.append(_reject(row, "DESCRIBED_BY_MODEL", "target compound was not uniquely linked"))
            return result
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        model_name = str(
            _recursive_value(metadata, "model_name")
            or _recursive_value(metadata, "repository_path")
            or row.get("parameter")
            or ""
        ).strip()
        repository_path = str(_recursive_value(metadata, "repository_path") or "").strip()
        model_checksum = str(
            _recursive_value(metadata, "model_checksum")
            or _recursive_value(metadata, "commit_checksum")
            or ""
        ).strip()
        if not model_name:
            result.rejected.append(
                _reject(row, "DESCRIBED_BY_MODEL", "public model evidence lacks a model name")
            )
            return result
        reusable_candidate = bool(repository_path or model_checksum)
        availability_status = (
            "repository_model" if reusable_candidate else "literature_reported"
        )
        model_id = stable_id(
            "pbpk-model",
            repository_path,
            model_checksum,
            row.get("source_location") if not reusable_candidate else "",
            model_name,
            compound.node_id,
        )
        standard_identifiers = {}
        if repository_path:
            standard_identifiers["REPOSITORY_PATH"] = repository_path
        if model_checksum:
            standard_identifiers["MODEL_CHECKSUM"] = model_checksum
        if not standard_identifiers:
            standard_identifiers["PBPKKB.EVIDENCE"] = str(row["evidence_id"])
        result.nodes.append(
            KGNode(
                node_id=model_id,
                node_type="PBPKModel",
                preferred_name=model_name,
                standard_identifiers=standard_identifiers,
                properties={
                    "model_checksum": model_checksum or None,
                    "software": _recursive_value(metadata, "software"),
                    "availability_status": availability_status,
                    "reusable_candidate": reusable_candidate,
                    "source_document_checksum": row.get("checksum"),
                },
            )
        )
        provenance = _provenance(
            row,
            (
                "structured_model_registry_mapping"
                if reusable_candidate
                else "literature_reported_model_mapping"
            ),
            0.99,
        )
        result.edges.extend(
            [
                _edge(
                    "Compound",
                    compound.node_id,
                    "DESCRIBED_BY_MODEL",
                    "PBPKModel",
                    model_id,
                    provenance,
                ),
                _edge("PBPKModel", model_id, "SUPPORTED_BY", "Evidence", str(row["evidence_id"]), provenance),
            ]
        )
        return result

    @staticmethod
    def _evidence_and_source_nodes(row: dict[str, Any]) -> list[KGNode]:
        source_document_id = _source_document_id(row)
        return [
            KGNode(
                node_id=str(row["evidence_id"]),
                node_type="Evidence",
                preferred_name=f"{row.get('evidence_type')}:{row.get('parameter')}",
                standard_identifiers={"PBPKKB": str(row["evidence_id"])},
                properties={
                    "quality_status": row.get("quality_status"),
                    "source_id": row.get("source_id"),
                    "source_location": row.get("source_location"),
                    "source_text_hash": row.get("source_text_hash"),
                    "extraction_confidence": row.get("extraction_confidence"),
                },
            ),
            KGNode(
                node_id=source_document_id,
                node_type="SourceDocument",
                preferred_name=str(row.get("source_location") or row.get("source_id")),
                standard_identifiers={
                    "CHECKSUM": str(row.get("checksum"))
                } if row.get("checksum") else {"PBPKKB": source_document_id},
                properties={
                    "source_id": row.get("source_id"),
                    "storage_path": row.get("storage_path"),
                    "raw_asset_id": row.get("raw_asset_id"),
                },
            ),
        ]

    @staticmethod
    def _provenance_edges(row: dict[str, Any]) -> list[KGEdge]:
        provenance = _provenance(row, "provenance_mapping", 1.0)
        return [
            _edge(
                "Evidence",
                str(row["evidence_id"]),
                "DERIVED_FROM",
                "SourceDocument",
                _source_document_id(row),
                provenance,
            )
        ]


def _extend(target: ExtractionResult, source: ExtractionResult) -> None:
    target.nodes.extend(source.nodes)
    target.edges.extend(source.edges)
    target.events.extend(source.events)
    target.rejected.extend(source.rejected)


def _edge(
    subject_type: str,
    subject_id: str | None,
    predicate: str,
    object_type: str,
    object_id: str | None,
    provenance: Provenance,
    properties: dict[str, Any] | None = None,
) -> KGEdge:
    if not subject_id or not object_id:
        raise ValueError(f"Cannot build {predicate} with an empty endpoint.")
    return KGEdge(
        edge_id=stable_id("edge", subject_id, predicate, object_id, provenance.evidence_id),
        subject_id=subject_id,
        subject_type=subject_type,
        predicate=predicate,
        object_id=object_id,
        object_type=object_type,
        provenance=provenance,
        properties=properties or {},
    )


def _provenance(
    row: dict[str, Any],
    method: str,
    confidence: float,
    *,
    supporting_text: str | None = None,
) -> Provenance:
    source_text = str(row.get("source_text") or "")
    quote = supporting_text or source_text
    if quote not in source_text:
        raise ValueError("Relation supporting text is not a verbatim substring of source_text.")
    digest = hashlib.sha256(source_text.encode("utf-8", errors="replace")).hexdigest()
    expected = str(row.get("source_text_hash") or "")
    if expected and digest != expected:
        raise ValueError(f"source_text_hash mismatch for evidence {row.get('evidence_id')}")
    return Provenance(
        evidence_id=str(row["evidence_id"]),
        source_id=str(row["source_id"]),
        source_location=str(row["source_location"]),
        supporting_text=quote,
        source_text_hash=digest,
        extraction_method=method,
        extractor_version=EXTRACTOR_VERSION,
        confidence=confidence,
        quality_status="auto_validated",
        negation_status=_negation_status(quote),
        speculation_status=_speculation_status(quote),
        context_id=_context_id(row),
    )


def _measurement_event_type(row: dict[str, Any]) -> str:
    evidence_type = str(row.get("evidence_type") or "")
    parameter = normalize_text(row.get("parameter"))
    if evidence_type == "adme":
        if "absorp" in parameter:
            return "AbsorptionEvent"
        if "excret" in parameter or "renal" in parameter:
            return "ExcretionEvent"
        if "metabol" in parameter:
            return "MetabolismEvent"
    return "PKMeasurement"


def _parameter_value_node(row: dict[str, Any], event_id: str) -> KGNode | None:
    value = row.get("canonical_value")
    if value is None:
        value = row.get("value")
    value_low = row.get("value_low")
    value_high = row.get("value_high")
    qualifier = row.get("value_qualifier")
    value_text = row.get("value_text")
    if all(item in (None, "") for item in (value, value_low, value_high, qualifier, value_text)):
        return None
    node_id = stable_id("value", event_id, value, value_low, value_high, qualifier, value_text)
    return KGNode(
        node_id=node_id,
        node_type="ParameterValue",
        preferred_name=str(value if value is not None else value_text or qualifier),
        properties={
            "value": value,
            "value_low": value_low,
            "value_high": value_high,
            "qualifier": qualifier,
            "value_text": value_text,
            "canonical_unit": row.get("canonical_unit"),
            "reported_unit": row.get("unit"),
        },
    )


def _context_node(row: dict[str, Any], node_type: str, field: str) -> KGNode | None:
    value = str(row.get(field) or "").strip()
    if not value:
        return None
    normalized = normalize_text(value)
    return KGNode(
        node_id=stable_id(node_type.lower(), normalized),
        node_type=node_type,
        preferred_name=value,
        standard_identifiers={"PBPKKB.CONTEXT": normalized},
        properties={"normalization_status": "contextual_standardized"},
    )


def _dose_regimen(
    row: dict[str, Any],
    mentions: list[EntityMention],
    catalog: EntityCatalog,
) -> tuple[KGNode, list[KGEdge], list[KGNode]] | None:
    values = {
        "dose": row.get("dose"),
        "route": row.get("route"),
        "formulation": row.get("formulation"),
        "frequency": row.get("frequency") or row.get("regimen"),
    }
    if not any(value not in (None, "") for value in values.values()):
        return None
    regimen_id = stable_id("dose-regimen", json.dumps(values, sort_keys=True, default=str))
    regimen = KGNode(
        node_id=regimen_id,
        node_type="DoseRegimen",
        preferred_name="; ".join(f"{key}={value}" for key, value in values.items() if value not in (None, "")),
        properties=values,
    )
    provenance = _provenance(row, "structured_regimen_mapping", 0.99)
    edges: list[KGEdge] = []
    nodes: list[KGNode] = []
    route = _first_linked(mentions, "AdministrationRoute")
    route_node_id = route.linked_node_id if route is not None else None
    if route_node_id is None and values["route"] not in (None, ""):
        route_entry, _, _ = catalog.link("AdministrationRoute", str(values["route"]))
        route_node_id = route_entry.node_id if route_entry is not None else None
    if route_node_id is not None:
        edges.append(
            _edge(
                "DoseRegimen",
                regimen_id,
                "HAS_ROUTE",
                "AdministrationRoute",
                route_node_id,
                provenance,
            )
        )
    formulation = str(row.get("formulation") or "").strip()
    if formulation:
        formulation_node = KGNode(
            node_id=stable_id("formulation", normalize_text(formulation)),
            node_type="Formulation",
            preferred_name=formulation,
            standard_identifiers={"PBPKKB.CONTEXT": normalize_text(formulation)},
            properties={"normalization_status": "contextual_standardized"},
        )
        nodes.append(formulation_node)
        edges.append(
            _edge("DoseRegimen", regimen_id, "HAS_FORMULATION", "Formulation", formulation_node.node_id, provenance)
        )
    return regimen, edges, nodes


def _target_compound_ref(row: dict[str, Any]) -> CompoundRef | None:
    compound_id = str(row.get("compound_id") or "").strip()
    compound_name = str(
        row.get("compound_name")
        or row.get("attributed_compound_name")
        or row.get("compound")
        or ""
    ).strip()
    if not compound_id or not compound_name:
        return None
    return CompoundRef(
        node_id=f"Compound:{compound_id}",
        text=compound_name,
        role=str(row.get("compound_role") or "") or None,
    )


def _as_compound_ref(mention: EntityMention | None) -> CompoundRef | None:
    if mention is None or not mention.linked_node_id:
        return None
    return CompoundRef(node_id=mention.linked_node_id, text=mention.text, role=mention.role)


def _linked_compounds(mentions: Iterable[EntityMention]) -> list[EntityMention]:
    seen: set[tuple[str, str | None]] = set()
    output = []
    for mention in mentions:
        if not mention.linked_node_id or not mention.linked_node_id.startswith("Compound:"):
            continue
        key = (mention.linked_node_id, mention.role)
        if key in seen:
            continue
        seen.add(key)
        output.append(mention)
    return output


def _role_mention(mentions: Iterable[EntityMention], role: str) -> EntityMention | None:
    matches = [mention for mention in mentions if mention.role == role]
    unique = {mention.linked_node_id for mention in matches}
    return matches[0] if len(unique) == 1 else None


def _counterpart_from_metadata(
    row: dict[str, Any],
    mentions: list[EntityMention],
    key: str,
) -> EntityMention | None:
    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
    value = _recursive_value(metadata, key)
    if not isinstance(value, str) or not value.strip():
        return None
    wanted = normalize_text(value)
    matches = [
        mention
        for mention in mentions
        if mention.linked_node_id
        and mention.linked_node_id.startswith("Compound:")
        and normalize_text(mention.text) == wanted
    ]
    unique = {mention.linked_node_id for mention in matches}
    if len(unique) != 1:
        return None
    match = matches[0]
    return match.model_copy(update={"role": key})


def _unique_other_compound(
    mentions: Iterable[EntityMention],
    target_node_id: str,
    role: str,
) -> EntityMention | None:
    matches_by_node: dict[str, EntityMention] = {}
    for mention in mentions:
        if not mention.linked_node_id or mention.linked_node_id == target_node_id:
            continue
        matches_by_node.setdefault(mention.linked_node_id, mention)
    if len(matches_by_node) != 1:
        return None
    return next(iter(matches_by_node.values())).model_copy(update={"role": role})


def _first_linked(mentions: Iterable[EntityMention], entity_type: str) -> EntityMention | None:
    matches = [
        mention
        for mention in mentions
        if mention.linked_node_id and mention.entity_type == entity_type
    ]
    unique = {mention.linked_node_id for mention in matches}
    return matches[0] if len(unique) == 1 else None


def _linked_or_structured_node_id(
    row: dict[str, Any],
    mentions: Iterable[EntityMention],
    catalog: EntityCatalog,
    *,
    entity_type: str,
    field_names: tuple[str, ...],
) -> str | None:
    mention = _first_linked(mentions, entity_type)
    if mention is not None:
        return mention.linked_node_id
    for field_name in field_names:
        value = str(row.get(field_name) or "").strip()
        if not value:
            continue
        entry, _, _ = catalog.link(entity_type, value)
        if entry is not None:
            return entry.node_id
    return None


def _value_is_supported_by_quote(row: dict[str, Any], quote: str) -> bool:
    normalized_quote = quote.casefold()
    value_text = str(row.get("value_text") or "").strip()
    if value_text and value_text.casefold() in normalized_quote:
        return True
    for field_name in ("canonical_value", "value", "value_low", "value_high"):
        value = row.get(field_name)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        candidates = {
            str(value),
            f"{numeric:g}",
            f"{numeric:.1f}",
            f"{numeric:.2f}",
            f"{numeric:.3f}",
        }
        if any(
            re.search(rf"(?<![0-9.]){re.escape(candidate)}(?![0-9.])", quote)
            for candidate in candidates
        ):
            return True
    return False


def _synthetic_mention(
    row: dict[str, Any],
    node_id: str,
    entity_type: str,
    text: str,
) -> EntityMention:
    return EntityMention(
        mention_id=stable_id("mention", row["evidence_id"], entity_type, text),
        evidence_id=str(row["evidence_id"]),
        source_id=str(row["source_id"]),
        source_location=str(row["source_location"]),
        text=text,
        start_char=0,
        end_char=max(1, len(text)),
        entity_type=entity_type,
        normalized_text=normalize_text(text),
        linked_node_id=node_id,
        standard_identifier=node_id,
        link_method="structured_field_controlled_mapping",
        recognition_method="structured_field",
        recognition_model="deterministic",
        confidence=1.0,
    )


def _source_document_id(row: dict[str, Any]) -> str:
    return stable_id(
        "source-document",
        row.get("raw_asset_id"),
        row.get("checksum"),
        row.get("storage_path"),
    )


def _event_trigger(row: dict[str, Any]) -> str:
    text = str(row.get("source_text") or "")
    parameter = str(row.get("parameter") or "")
    match = re.search(re.escape(parameter), text, re.IGNORECASE) if parameter else None
    if match:
        return match.group(0)
    return text[: min(len(text), 240)]


def _event_context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "condition",
            "dose",
            "route",
            "population",
            "formulation",
            "matrix",
            "species",
            "assay_method",
        )
        if row.get(key) not in (None, "")
    }


def _context_id(row: dict[str, Any]) -> str | None:
    context = _event_context(row)
    return stable_id("context", json.dumps(context, sort_keys=True, default=str)) if context else None


def _negation_status(text: str) -> str:
    return "negated" if re.search(r"\b(?:no|not|neither|without|failed to)\b", text, re.I) else "affirmed"


def _speculation_status(text: str) -> str:
    return "speculative" if re.search(r"\b(?:may|might|could|suggests?|possibly|potentially)\b", text, re.I) else "asserted"


def _reject(row: dict[str, Any], predicate: str | None, *reasons: str) -> RejectedRelation:
    valid_reasons = [reason for reason in reasons if reason]
    return RejectedRelation(
        rejection_id=stable_id("rejected-relation", row.get("evidence_id"), predicate, *valid_reasons),
        evidence_id=str(row.get("evidence_id") or ""),
        predicate=predicate,
        extraction_method="ontology_constrained_relation_extraction",
        reasons=valid_reasons or ["unspecified relation validation failure"],
        supporting_text=str(row.get("source_text") or ""),
    )


def _recursive_value(value: Any, wanted_key: str) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == wanted_key.lower() and item not in (None, "", [], {}):
                return item
        for item in value.values():
            found = _recursive_value(item, wanted_key)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for item in value:
            found = _recursive_value(item, wanted_key)
            if found not in (None, "", [], {}):
                return found
    return None
