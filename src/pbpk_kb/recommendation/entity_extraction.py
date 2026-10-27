from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from pbpk_kb.graph.final_evidence_graph import STATIC_TARGETS
from pbpk_kb.recommendation.kg_models import EntityMention, KGNode, UnresolvedEntity, stable_id


NER_MODEL = "Ihor/gliner-biomed-large-v1.0"
NER_LABELS = (
    "drug or chemical compound",
    "drug metabolite",
    "enzyme",
    "drug transporter",
    "gene",
    "protein",
    "organ",
    "tissue",
    "patient population",
    "disease",
    "genetic variant or genotype",
    "drug administration route",
    "drug formulation",
    "drug dose",
    "dosing frequency",
    "time point or duration",
    "pharmacokinetic parameter",
    "measurement unit",
    "PBPK software",
    "PBPK model",
    "clinical study",
    "scientific publication identifier",
)

LABEL_TO_NODE_TYPE = {
    "drug or chemical compound": "Compound",
    "drug metabolite": "Metabolite",
    "enzyme": "Enzyme",
    "drug transporter": "Transporter",
    "gene": "Gene",
    "protein": "Protein",
    "organ": "Organ",
    "tissue": "Tissue",
    "patient population": "Population",
    "disease": "Disease",
    "genetic variant or genotype": "GeneticVariant",
    "drug administration route": "AdministrationRoute",
    "drug formulation": "Formulation",
    "drug dose": "Dose",
    "dosing frequency": "Frequency",
    "time point or duration": "TimePoint",
    "pharmacokinetic parameter": "PKParameter",
    "measurement unit": "Unit",
    "PBPK software": "Software",
    "PBPK model": "PBPKModel",
    "clinical study": "Study",
    "scientific publication identifier": "Publication",
}

PARAMETERS = {
    "auc": "AUC",
    "area under the curve": "AUC",
    "cmax": "Cmax",
    "maximum concentration": "Cmax",
    "tmax": "Tmax",
    "half life": "half_life",
    "half-life": "half_life",
    "clearance": "clearance",
    "volume of distribution": "volume_of_distribution",
    "bioavailability": "bioavailability",
    "fraction absorbed": "fraction_absorbed",
    "renal excretion": "renal_excretion",
    "absorption": "absorption",
    "distribution": "distribution",
    "metabolism": "metabolism",
    "excretion": "excretion",
}

ROUTES = {
    "oral": ("oral", "SNOMEDCT:26643006"),
    "orally": ("oral", "SNOMEDCT:26643006"),
    "intravenous": ("intravenous", "SNOMEDCT:47625008"),
    "iv": ("intravenous", "SNOMEDCT:47625008"),
    "i v": ("intravenous", "SNOMEDCT:47625008"),
    "iv infusion": ("intravenous", "SNOMEDCT:47625008"),
    "intravenous infusion": ("intravenous", "SNOMEDCT:47625008"),
    "subcutaneous": ("subcutaneous", "SNOMEDCT:34206005"),
    "intramuscular": ("intramuscular", "SNOMEDCT:78421000"),
    "inhalation": ("inhalation", "SNOMEDCT:18679011000001101"),
    "intranasal": ("intranasal", "SNOMEDCT:260617000"),
}

ANATOMY = {
    "liver": ("Organ", "liver", "UBERON:0002107"),
    "kidney": ("Organ", "kidney", "UBERON:0002113"),
    "kidneys": ("Organ", "kidney", "UBERON:0002113"),
    "intestine": ("Organ", "intestine", "UBERON:0000160"),
    "gut": ("Organ", "intestine", "UBERON:0000160"),
    "brain": ("Organ", "brain", "UBERON:0000955"),
    "heart": ("Organ", "heart", "UBERON:0000948"),
    "lung": ("Organ", "lung", "UBERON:0002048"),
    "lungs": ("Organ", "lung", "UBERON:0002048"),
    "skin": ("Organ", "skin", "UBERON:0002097"),
    "blood": ("Tissue", "blood", "UBERON:0000178"),
    "plasma": ("Tissue", "blood plasma", "UBERON:0001969"),
    "adipose tissue": ("Tissue", "adipose tissue", "UBERON:0001013"),
}

UNIT_ALIASES = {
    "mg": ("mg", "UCUM:mg"),
    "g": ("g", "UCUM:g"),
    "kg": ("kg", "UCUM:kg"),
    "ng/ml": ("ng/mL", "UCUM:ng/mL"),
    "ug/ml": ("ug/mL", "UCUM:ug/mL"),
    "µg/ml": ("ug/mL", "UCUM:ug/mL"),
    "mg/l": ("mg/L", "UCUM:mg/L"),
    "h": ("h", "UCUM:h"),
    "hr": ("h", "UCUM:h"),
    "hours": ("h", "UCUM:h"),
    "min": ("min", "UCUM:min"),
    "%": ("%", "UCUM:%"),
    "fold": ("fold", "PBPK:fold"),
    "ratio": ("ratio", "UCUM:1"),
    "g/mol": ("g/mol", "UCUM:g/mol"),
}

COMMON_ENZYMES = (
    "CYP3A",
    "CYP1A2",
    "CYP2A6",
    "CYP2B6",
    "CYP2C8",
    "CYP2C9",
    "CYP2C19",
    "CYP2D6",
    "CYP2E1",
    "CYP3A4",
    "CYP3A5",
    "UGT1A1",
    "UGT1A3",
    "UGT1A4",
    "UGT1A6",
    "UGT1A9",
    "UGT2B7",
    "UGT2B15",
    "CES1",
    "CES2",
    "FMO3",
    "NAT2",
)


def normalize_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^\w%µμ/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class NERBackend(Protocol):
    model_name: str

    def predict(self, text: str) -> list[dict[str, Any]]: ...


class GlinerNERBackend:
    def __init__(
        self,
        model_name: str = NER_MODEL,
        *,
        threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("Install the 'gliner' package to run biomedical NER.") from exc
        self.model_name = model_name
        self.threshold = threshold
        self._model = GLiNER.from_pretrained(model_name)
        self._model.to(device)

    def predict(self, text: str) -> list[dict[str, Any]]:
        return list(self._model.predict_entities(text, list(NER_LABELS), threshold=self.threshold))


class NullNERBackend:
    model_name = "none"

    def predict(self, text: str) -> list[dict[str, Any]]:
        return []


@dataclass(frozen=True)
class CatalogEntry:
    node_id: str
    node_type: str
    preferred_name: str
    aliases: tuple[str, ...]
    standard_identifiers: dict[str, str]


class EntityCatalog:
    def __init__(
        self,
        evidence_rows: Iterable[dict[str, Any]],
        external_entries: Iterable[dict[str, Any]] = (),
    ) -> None:
        rows = list(evidence_rows)
        self.entries: dict[str, CatalogEntry] = {}
        self.alias_index: dict[tuple[str, str], list[CatalogEntry]] = defaultdict(list)
        self._build_compounds(rows)
        self._build_targets()
        self._build_controlled_terms()
        self._build_evidence_terms(rows)
        self._build_external_entries(external_entries)
        self._rebuild_alias_index()

    def _add(self, entry: CatalogEntry) -> None:
        current = self.entries.get(entry.node_id)
        if current is not None:
            if (
                current.node_type != entry.node_type
                or normalize_text(current.preferred_name) != normalize_text(entry.preferred_name)
            ):
                raise ValueError(f"Conflicting entity catalog entry for {entry.node_id}.")
            entry = CatalogEntry(
                node_id=current.node_id,
                node_type=current.node_type,
                preferred_name=current.preferred_name,
                aliases=tuple(dict.fromkeys((*current.aliases, *entry.aliases))),
                standard_identifiers={
                    **current.standard_identifiers,
                    **entry.standard_identifiers,
                },
            )
        self.entries[entry.node_id] = entry
        for alias in entry.aliases:
            key = normalize_text(alias)
            if key:
                self.alias_index[(entry.node_type, key)].append(entry)

    def _build_compounds(self, rows: Iterable[dict[str, Any]]) -> None:
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            compound_id = str(row.get("compound_id") or "").strip()
            name = str(
                row.get("compound_name")
                or row.get("attributed_compound_name")
                or row.get("compound")
                or ""
            ).strip()
            if not compound_id or not name:
                continue
            item = grouped.setdefault(compound_id, {"name": name, "aliases": set(), "ids": {}})
            item["aliases"].add(name)
            metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
            for key in ("target_compound", "preferred_name", "active_ingredient"):
                value = _recursive_value(metadata, key)
                if isinstance(value, str) and 2 < len(value) < 100:
                    item["aliases"].add(value)
            for key, namespace in (
                ("pubchem_cid", "PUBCHEM.COMPOUND"),
                ("inchi_key", "INCHIKEY"),
                ("chembl_id", "CHEMBL.COMPOUND"),
                ("rxnorm_rxcui", "RXNORM"),
                ("drugcentral_id", "DRUGCENTRAL"),
                ("unii", "UNII"),
            ):
                value = row.get(key) or _recursive_value(metadata, key)
                if value:
                    item["ids"][namespace] = str(value)
        for compound_id, item in grouped.items():
            identifiers = {"PBPKKB": compound_id, **item["ids"]}
            self._add(
                CatalogEntry(
                    node_id=f"Compound:{compound_id}",
                    node_type="Compound",
                    preferred_name=item["name"],
                    aliases=tuple(sorted(item["aliases"], key=str.lower)),
                    standard_identifiers=identifiers,
                )
            )

    def _build_targets(self) -> None:
        for key, preferred, gene_symbol, target_type, aliases in STATIC_TARGETS:
            node_type = "Enzyme" if target_type == "enzyme" else "Transporter"
            identifiers = {"HGNC.SYMBOL": gene_symbol} if gene_symbol else {"PBPKKB": key}
            self._add(
                CatalogEntry(
                    node_id=f"{node_type}:{key}",
                    node_type=node_type,
                    preferred_name=preferred,
                    aliases=tuple(dict.fromkeys((*aliases, key, preferred))),
                    standard_identifiers=identifiers,
                )
            )
        for symbol in COMMON_ENZYMES:
            self._add(
                CatalogEntry(
                    node_id=f"Enzyme:{symbol}",
                    node_type="Enzyme",
                    preferred_name=symbol,
                    aliases=(symbol,),
                    standard_identifiers={"HGNC.SYMBOL": symbol},
                )
            )

    def _build_controlled_terms(self) -> None:
        for alias, preferred in PARAMETERS.items():
            node_id = f"PKParameter:{normalize_text(preferred)}"
            self._add(
                CatalogEntry(
                    node_id=node_id,
                    node_type="PKParameter",
                    preferred_name=preferred,
                    aliases=(alias, preferred),
                    standard_identifiers={"PBPKKB": normalize_text(preferred)},
                )
            )
        for alias, (preferred, identifier) in ROUTES.items():
            self._add(
                CatalogEntry(
                    node_id=f"AdministrationRoute:{identifier}",
                    node_type="AdministrationRoute",
                    preferred_name=preferred,
                    aliases=(alias, preferred),
                    standard_identifiers={identifier.split(":", 1)[0]: identifier.split(":", 1)[1]},
                )
            )
        for alias, (node_type, preferred, identifier) in ANATOMY.items():
            self._add(
                CatalogEntry(
                    node_id=f"{node_type}:{identifier}",
                    node_type=node_type,
                    preferred_name=preferred,
                    aliases=(alias, preferred),
                    standard_identifiers={"UBERON": identifier.split(":", 1)[1]},
                )
            )
        for alias, (preferred, identifier) in UNIT_ALIASES.items():
            self._add(
                CatalogEntry(
                    node_id=f"Unit:{identifier}",
                    node_type="Unit",
                    preferred_name=preferred,
                    aliases=(alias, preferred),
                    standard_identifiers={identifier.split(":", 1)[0]: identifier.split(":", 1)[1]},
                )
            )

    def _build_evidence_terms(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            parameter = str(row.get("parameter") or "").strip()
            if parameter:
                normalized = normalize_text(parameter)
                self._add(
                    CatalogEntry(
                        node_id=f"PKParameter:{normalized}",
                        node_type="PKParameter",
                        preferred_name=parameter,
                        aliases=(parameter,),
                        standard_identifiers={"PBPKKB.PARAMETER": normalized},
                    )
                )
            for unit_value in (row.get("canonical_unit"), row.get("unit")):
                unit = str(unit_value or "").strip()
                if not unit:
                    continue
                normalized_unit = normalize_text(unit)
                known = UNIT_ALIASES.get(normalized_unit)
                if known:
                    preferred, identifier = known
                    identifiers = {identifier.split(":", 1)[0]: identifier.split(":", 1)[1]}
                    node_id = f"Unit:{identifier}"
                else:
                    preferred = unit
                    identifiers = {"PBPKKB.UNIT": normalized_unit}
                    node_id = f"Unit:{stable_id('unit', normalized_unit)}"
                self._add(
                    CatalogEntry(
                        node_id=node_id,
                        node_type="Unit",
                        preferred_name=preferred,
                        aliases=(unit,),
                        standard_identifiers=identifiers,
                    )
                )

    def _build_external_entries(self, entries: Iterable[dict[str, Any]]) -> None:
        for value in entries:
            node_id = str(value.get("node_id") or "").strip()
            node_type = str(value.get("node_type") or "").strip()
            preferred_name = str(value.get("preferred_name") or "").strip()
            aliases = tuple(str(item).strip() for item in value.get("aliases") or [] if str(item).strip())
            identifiers = {
                str(key): str(item)
                for key, item in (value.get("standard_identifiers") or {}).items()
                if item not in (None, "")
            }
            if not node_id or node_type != "Compound" or not preferred_name or not identifiers:
                continue
            external_names = {normalize_text(value) for value in (preferred_name, *aliases)}
            matching_internal = [
                entry
                for entry in self.entries.values()
                if entry.node_type == "Compound"
                and not entry.node_id.startswith("Compound:PUBLIC:")
                and any(normalize_text(alias) in external_names for alias in entry.aliases)
            ]
            if len(matching_internal) == 1:
                current = matching_internal[0]
                self.entries[current.node_id] = CatalogEntry(
                    node_id=current.node_id,
                    node_type=current.node_type,
                    preferred_name=current.preferred_name,
                    aliases=tuple(dict.fromkeys((*current.aliases, preferred_name, *aliases))),
                    standard_identifiers={**current.standard_identifiers, **identifiers},
                )
                continue
            self._add(
                CatalogEntry(
                    node_id=node_id,
                    node_type=node_type,
                    preferred_name=preferred_name,
                    aliases=tuple(dict.fromkeys((preferred_name, *aliases))),
                    standard_identifiers=identifiers,
                )
            )

    def _rebuild_alias_index(self) -> None:
        self.alias_index = defaultdict(list)
        for entry in self.entries.values():
            for alias in entry.aliases:
                key = normalize_text(alias)
                if key:
                    self.alias_index[(entry.node_type, key)].append(entry)

    def link(self, entity_type: str, mention_text: str) -> tuple[CatalogEntry | None, str, float]:
        key = normalize_text(mention_text)
        if not key:
            return None, "empty_mention", 0.0
        search_types = _compatible_node_types(entity_type)
        matches: dict[str, CatalogEntry] = {}
        for node_type in search_types:
            for entry in self.alias_index.get((node_type, key), []):
                matches[entry.node_id] = entry
        if len(matches) == 1:
            return next(iter(matches.values())), "exact_controlled_alias", 1.0
        if len(matches) > 1:
            return None, "ambiguous_exact_alias", 0.5
        return None, "no_unique_identifier", 0.0

    def nodes(self) -> list[KGNode]:
        return [
            KGNode(
                node_id=entry.node_id,
                node_type=entry.node_type,
                preferred_name=entry.preferred_name,
                standard_identifiers=entry.standard_identifiers,
                properties={"aliases": list(entry.aliases)},
            )
            for entry in sorted(self.entries.values(), key=lambda value: value.node_id)
        ]


class PBPKEntityExtractor:
    def __init__(self, catalog: EntityCatalog, ner: NERBackend) -> None:
        self.catalog = catalog
        self.ner = ner

    def extract(
        self,
        row: dict[str, Any],
    ) -> tuple[list[EntityMention], list[UnresolvedEntity]]:
        text = str(row.get("source_text") or "")
        if not text:
            return [], []
        candidates = [
            *self._structured_candidates(row, text),
            *self._catalog_candidates(text),
            *self._model_candidates(text),
        ]
        mentions = self._deduplicate_and_link(row, candidates)
        unresolved = [
            UnresolvedEntity(
                mention_id=mention.mention_id,
                evidence_id=mention.evidence_id,
                entity_type=mention.entity_type,
                mention_text=mention.text,
                normalized_text=mention.normalized_text,
                rejection_reason=mention.link_method,
                recognition_confidence=mention.confidence,
            )
            for mention in mentions
            if mention.linked_node_id is None
        ]
        return mentions, unresolved

    def _structured_candidates(self, row: dict[str, Any], text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        values = (
            ("Compound", row.get("compound_name") or row.get("attributed_compound_name"), row.get("compound_role")),
            ("PKParameter", row.get("parameter"), None),
            ("Unit", row.get("canonical_unit") or row.get("unit"), None),
            ("AdministrationRoute", row.get("route"), None),
            ("Population", row.get("population"), None),
            ("Formulation", row.get("formulation"), None),
            ("Dose", row.get("dose"), None),
        )
        for entity_type, value, role in values:
            mention_text = str(value or "").strip()
            if not mention_text:
                continue
            for start, end in _find_all_spans(text, mention_text):
                candidates.append(
                    {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "confidence": 1.0,
                        "recognition_method": "structured_field_exact_span",
                        "role": str(role or "") or None,
                    }
                )
        return candidates

    def _model_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for item in self.ner.predict(text):
            label = str(item.get("label") or "")
            entity_type = LABEL_TO_NODE_TYPE.get(label)
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if not entity_type or start < 0 or end <= start or end > len(text):
                continue
            candidates.append(
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "entity_type": entity_type,
                    "confidence": float(item.get("score") or 0.0),
                    "recognition_method": "gliner_biomedical_ner",
                    "role": None,
                }
            )
        return candidates

    def _catalog_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        for entry in self.catalog.entries.values():
            for alias in sorted(entry.aliases, key=len, reverse=True):
                if len(alias.strip()) < 2:
                    continue
                for start, end in _find_all_spans(text, alias):
                    key = (start, end, entry.node_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "text": text[start:end],
                            "start": start,
                            "end": end,
                            "entity_type": entry.node_type,
                            "confidence": 0.99,
                            "recognition_method": "controlled_dictionary_exact_span",
                            "role": None,
                        }
                    )
        return candidates

    def _deduplicate_and_link(
        self,
        row: dict[str, Any],
        candidates: Iterable[dict[str, Any]],
    ) -> list[EntityMention]:
        evidence_id = str(row["evidence_id"])
        selected: dict[tuple[int, int, str], dict[str, Any]] = {}
        for candidate in candidates:
            key = (int(candidate["start"]), int(candidate["end"]), str(candidate["entity_type"]))
            current = selected.get(key)
            if current is None or _candidate_priority(candidate) > _candidate_priority(current):
                selected[key] = candidate
        output: list[EntityMention] = []
        for candidate in sorted(selected.values(), key=lambda item: (item["start"], item["end"], item["entity_type"])):
            entry, method, link_confidence = self.catalog.link(candidate["entity_type"], candidate["text"])
            mention_id = stable_id(
                "mention",
                evidence_id,
                candidate["start"],
                candidate["end"],
                candidate["entity_type"],
            )
            output.append(
                EntityMention(
                    mention_id=mention_id,
                    evidence_id=evidence_id,
                    source_id=str(row.get("source_id") or ""),
                    source_location=str(row.get("source_location") or ""),
                    text=candidate["text"],
                    start_char=candidate["start"],
                    end_char=candidate["end"],
                    entity_type=candidate["entity_type"],
                    normalized_text=normalize_text(candidate["text"]),
                    linked_node_id=entry.node_id if entry else None,
                    standard_identifier=_primary_identifier(entry),
                    link_method=method,
                    recognition_method=candidate["recognition_method"],
                    recognition_model=(
                        self.ner.model_name
                        if candidate["recognition_method"] == "gliner_biomedical_ner"
                        else "pbpk-controlled-vocabulary-1.0.0"
                    ),
                    confidence=min(float(candidate["confidence"]), link_confidence or float(candidate["confidence"])),
                    role=candidate.get("role"),
                )
            )
        return output


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, float]:
    method_rank = {
        "structured_field_exact_span": 3,
        "controlled_dictionary_exact_span": 2,
        "gliner_biomedical_ner": 1,
    }.get(str(candidate.get("recognition_method")), 0)
    return method_rank, float(candidate.get("confidence") or 0.0)


def _compatible_node_types(entity_type: str) -> tuple[str, ...]:
    if entity_type == "Compound":
        return "Compound", "ActiveMoiety", "SaltForm", "Metabolite"
    if entity_type == "Protein":
        return "Protein", "Enzyme", "Transporter"
    if entity_type in {"Dose", "Frequency", "TimePoint"}:
        return ()
    return (entity_type,)


def _find_all_spans(text: str, mention: str) -> Iterable[tuple[int, int]]:
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(mention) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    for match in pattern.finditer(text):
        yield match.start(), match.end()


def _primary_identifier(entry: CatalogEntry | None) -> str | None:
    if entry is None or not entry.standard_identifiers:
        return None
    namespace, value = next(iter(entry.standard_identifiers.items()))
    return f"{namespace}:{value}"


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
