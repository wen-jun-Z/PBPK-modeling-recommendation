from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def normalize_name(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class ResolvedCompound:
    preferred_name: str
    normalized_name: str
    synonyms: list[str] = field(default_factory=list)
    pubchem_cid: str | None = None
    chembl_id: str | None = None
    drugcentral_id: str | None = None
    rxnorm_rxcui: str | None = None
    inchi_key: str | None = None
    cas_number: str | None = None
    approval_status: str | None = None
    source_id: str | None = None
    source_location: str | None = None
    confidence_score: float = 0.5
    needs_review: bool = True


class CompoundResolver:
    """Resolve public identifiers from connector parsed records into one compound identity."""

    def resolve(self, records: list[dict[str, Any]], query: str = "midazolam") -> ResolvedCompound:
        pubchem_record = self._first_record(records, "pubchem", "compound_metadata")
        label_record = self._first_record(records, "dailymed", "label_detail")

        preferred_name = self._preferred_name(pubchem_record, label_record, query)
        synonyms = self._synonyms(pubchem_record, label_record, preferred_name)
        pubchem_payload = (pubchem_record or {}).get("record", {})

        primary_source_id = (pubchem_record or label_record or {}).get("source_id")
        confidence = 0.55
        if pubchem_record:
            confidence += 0.25
        if label_record:
            confidence += 0.15
        if pubchem_record and label_record and normalize_name(preferred_name) in {normalize_name(s) for s in synonyms}:
            confidence += 0.05

        return ResolvedCompound(
            preferred_name=preferred_name,
            normalized_name=normalize_name(preferred_name),
            synonyms=synonyms,
            pubchem_cid=self._string_or_none(pubchem_payload.get("pubchem_cid")),
            chembl_id=self._identifier_from_records(records, "chembl_id"),
            drugcentral_id=self._identifier_from_records(records, "drugcentral_id"),
            rxnorm_rxcui=self._identifier_from_records(records, "rxnorm_rxcui"),
            inchi_key=self._string_or_none(pubchem_payload.get("inchi_key")),
            cas_number=self._identifier_from_records(records, "cas_number"),
            source_id=primary_source_id,
            source_location=(pubchem_record or label_record or {}).get("source_location"),
            confidence_score=min(confidence, 0.99),
            needs_review=not (pubchem_record and label_record),
        )

    @staticmethod
    def _first_record(records: list[dict[str, Any]], source_id: str, record_type: str) -> dict[str, Any] | None:
        return next(
            (
                record
                for record in records
                if record.get("source_id") == source_id and record.get("record_type") == record_type
            ),
            None,
        )

    def _preferred_name(
        self,
        pubchem_record: dict[str, Any] | None,
        label_record: dict[str, Any] | None,
        query: str,
    ) -> str:
        query_name = " ".join(query.split())
        query_key = normalize_name(query_name)
        if pubchem_record and query_name:
            for synonym in pubchem_record.get("record", {}).get("synonyms") or []:
                synonym_value = str(synonym)
                if normalize_name(synonym_value) == query_key:
                    return synonym_value
            return query_name
        if label_record:
            label_payload = label_record.get("record", {})
            for collection_name in ("active_moieties", "active_ingredients"):
                for item in label_payload.get(collection_name) or []:
                    name = str(item.get("name") or "")
                    name_key = normalize_name(name)
                    if name_key == query_key or query_key in name_key or name_key in query_key:
                        return name
            if query_name:
                return query_name
        if pubchem_record:
            synonyms = pubchem_record.get("record", {}).get("synonyms") or []
            if synonyms:
                return str(synonyms[0])
        return query

    def _synonyms(
        self,
        pubchem_record: dict[str, Any] | None,
        label_record: dict[str, Any] | None,
        preferred_name: str,
    ) -> list[str]:
        values: list[str] = [preferred_name]
        if pubchem_record:
            values.extend(str(value) for value in pubchem_record.get("record", {}).get("synonyms") or [])
        if label_record:
            label_payload = label_record.get("record", {})
            for ingredient in label_payload.get("active_ingredients") or []:
                values.append(str(ingredient.get("name") or ""))
            for moiety in label_payload.get("active_moieties") or []:
                values.append(str(moiety.get("name") or ""))
            values.append(str(label_payload.get("title") or ""))

        seen: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            cleaned = " ".join(value.split())
            key = normalize_name(cleaned)
            if not cleaned or key in seen:
                continue
            seen.add(key)
            unique_values.append(cleaned)
        return unique_values[:50]

    @staticmethod
    def _identifier_from_records(records: list[dict[str, Any]], field_name: str) -> str | None:
        aliases = {
            "chembl_id": ["chembl_id", "molecule_chembl_id"],
            "rxnorm_rxcui": ["rxnorm_rxcui", "rxcui"],
            "drugcentral_id": ["drugcentral_id"],
            "cas_number": ["cas_number"],
        }
        # [FIX 2026-07-02 · claude-opus-4-8] (F3) scope each identifier to its authoritative source, so e.g. a
        # chembl_id is only taken from a chembl record. Previously the first matching field across ANY record was
        # returned, which could attach an identifier from an unrelated source/compound when records are mixed.
        expected_source = {
            "chembl_id": "chembl",
            "rxnorm_rxcui": "rxnorm",
            "drugcentral_id": "drugcentral",
            "cas_number": None,
        }.get(field_name)
        candidate_fields = aliases.get(field_name, [field_name])
        for record in records:
            if expected_source is not None and record.get("source_id") != expected_source:
                continue
            payload = record.get("record") or {}
            for candidate_field in candidate_fields:
                value = payload.get(candidate_field)
                if value:
                    return str(value)
        return None

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)
