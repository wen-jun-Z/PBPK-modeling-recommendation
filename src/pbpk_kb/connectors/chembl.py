from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ConnectorError, ParsedSourceRecord, RawDocumentMetadata


class ChEMBLConnector(BaseConnector):
    source_id = "chembl"
    parser_name = "json_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        base_url = str(self.source.api_url or self.source.access_url).rstrip("/")
        search_url = f"{base_url}/molecule/search.json?" + urlencode(
            {"q": query, "limit": str(retmax)}
        )
        search_payload, search_content_type, search_status_code, search_bytes = self.fetch_json(search_url)
        molecules = search_payload.get("molecules", [])
        search_raw = self.save_raw_json(
            source_url=search_url,
            payload=search_payload,
            content_type=search_content_type,
            status_code=search_status_code,
            raw_record_count=len(molecules),
            retrieval_method="chembl_molecule_search",
            extra_metadata={"query": query},
            raw_payload_bytes=search_bytes,
        )

        records: list[ParsedSourceRecord] = []
        for molecule in molecules[:retmax]:
            molecule_id = molecule.get("molecule_chembl_id")
            activities: list[dict[str, Any]] = []
            if molecule_id:
                try:
                    activities = self._fetch_activity_sample(str(molecule_id), query, retmax)
                except Exception as exc:
                    self.errors.append(
                        ConnectorError(
                            stage="chembl_activity_lookup",
                            message=str(exc),
                            retryable=True,
                            source_record_id=str(molecule_id),
                        )
                    )
            records.append(self._parse_molecule(molecule, activities, search_raw))
        if not records:
            records.append(
                self.make_record(
                    raw=search_raw,
                    source_record_id=f"no_match:{query}",
                    source_url=search_url,
                    source_location=f"CHEMBL_QUERY:{query}",
                    record_type="chembl_no_match",
                    record={"query": query, "extraction_level": "query_no_match"},
                )
            )
        return records

    def _fetch_activity_sample(self, molecule_chembl_id: str, query: str, retmax: int) -> list[dict[str, Any]]:
        base_url = str(self.source.api_url or self.source.access_url).rstrip("/")
        activity_url = f"{base_url}/activity.json?" + urlencode(
            {"molecule_chembl_id": molecule_chembl_id, "limit": str(retmax)}
        )
        activity_payload, activity_content_type, activity_status_code, activity_bytes = self.fetch_json(activity_url)
        activities = activity_payload.get("activities", [])
        self.save_raw_json(
            source_url=activity_url,
            payload=activity_payload,
            content_type=activity_content_type,
            status_code=activity_status_code,
            raw_record_count=len(activities),
            retrieval_method="chembl_activity_sample",
            extra_metadata={"query": query, "molecule_chembl_id": molecule_chembl_id},
            raw_payload_bytes=activity_bytes,
        )
        return activities

    def _parse_molecule(
        self,
        molecule: dict[str, Any],
        activities: list[dict[str, Any]],
        raw: RawDocumentMetadata,
    ) -> ParsedSourceRecord:
        molecule_id = str(molecule.get("molecule_chembl_id") or "unknown_molecule")
        molecule_properties = molecule.get("molecule_properties") if isinstance(molecule.get("molecule_properties"), dict) else {}
        structures = molecule.get("molecule_structures") if isinstance(molecule.get("molecule_structures"), dict) else {}
        record = {
            "molecule_chembl_id": molecule_id,
            "pref_name": molecule.get("pref_name"),
            "max_phase": molecule.get("max_phase"),
            "first_approval": molecule.get("first_approval"),
            "atc_classifications": molecule.get("atc_classifications", []),
            "molecule_type": molecule.get("molecule_type"),
            "molecular_formula": molecule_properties.get("full_molformula"),
            "molecular_weight": molecule_properties.get("full_mwt"),
            "alogp": molecule_properties.get("alogp"),
            "canonical_smiles": structures.get("canonical_smiles"),
            "activity_sample_count": len(activities),
            "activities": activities[:10],
            "raw_fields": molecule,
            "extraction_level": "compound_target_activity_metadata",
        }
        return self.make_record(
            raw=raw,
            source_record_id=molecule_id,
            source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{quote(molecule_id)}/",
            source_location=f"CHEMBL_MOLECULE:{molecule_id}",
            record_type="chembl_molecule",
            record=record,
        )
