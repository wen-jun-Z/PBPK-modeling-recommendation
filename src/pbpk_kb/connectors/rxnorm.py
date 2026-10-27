from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from pbpk_kb.connectors.base import BaseConnector
from pbpk_kb.connectors.models import ConnectorError, ParsedSourceRecord, RawDocumentMetadata


class RxNormConnector(BaseConnector):
    source_id = "rxnorm"
    parser_name = "json_parser.py"
    default_query = "midazolam"

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        base_url = str(self.source.api_url or self.source.access_url).rstrip("/")
        version_url = f"{base_url}/version.json"
        version_payload, version_content_type, version_status_code, version_bytes = self.fetch_json(version_url)
        self.source_version = version_payload.get("version") or version_payload.get("apiVersion")
        self.save_raw_json(
            source_url=version_url,
            payload=version_payload,
            content_type=version_content_type,
            status_code=version_status_code,
            raw_record_count=1,
            source_version=self.source_version,
            retrieval_method="rxnorm_version",
            raw_payload_bytes=version_bytes,
        )

        rxcui_url = f"{base_url}/rxcui.json?" + urlencode({"name": query, "search": "2"})
        rxcui_payload, rxcui_content_type, rxcui_status_code, rxcui_bytes = self.fetch_json(rxcui_url)
        rxcuis = [str(value) for value in rxcui_payload.get("idGroup", {}).get("rxnormId", [])]
        rxcui_raw = self.save_raw_json(
            source_url=rxcui_url,
            payload=rxcui_payload,
            content_type=rxcui_content_type,
            status_code=rxcui_status_code,
            raw_record_count=len(rxcuis),
            source_version=self.source_version,
            retrieval_method="rxnorm_rxcui_lookup",
            extra_metadata={"query": query},
            raw_payload_bytes=rxcui_bytes,
        )

        records: list[ParsedSourceRecord] = []
        for rxcui in rxcuis[:retmax]:
            properties_url = f"{base_url}/rxcui/{quote(rxcui)}/allProperties.json?" + urlencode({"prop": "all"})
            try:
                properties_payload, properties_content_type, properties_status_code, properties_bytes = self.fetch_json(properties_url)
                properties = properties_payload.get("propConceptGroup", {}).get("propConcept", [])
                properties_raw = self.save_raw_json(
                    source_url=properties_url,
                    payload=properties_payload,
                    content_type=properties_content_type,
                    status_code=properties_status_code,
                    raw_record_count=len(properties),
                    source_version=self.source_version,
                    retrieval_method="rxnorm_all_properties",
                    extra_metadata={"query": query, "rxcui": rxcui, "lookup_raw_checksum": rxcui_raw.raw_checksum},
                    raw_payload_bytes=properties_bytes,
                )
                records.append(self._parse_rxcui(rxcui, query, properties, properties_raw))
            except Exception as exc:
                self.errors.append(
                    ConnectorError(
                        stage="rxnorm_all_properties",
                        message=str(exc),
                        retryable=True,
                        source_record_id=rxcui,
                    )
                )
                records.append(self._parse_rxcui(rxcui, query, [], rxcui_raw))
        if not records:
            records.append(
                self.make_record(
                    raw=rxcui_raw,
                    source_record_id=f"no_match:{query}",
                    source_url=rxcui_url,
                    source_location=f"RXNORM_QUERY:{query}",
                    record_type="rxnorm_no_match",
                    record={"query": query, "extraction_level": "query_no_match"},
                )
            )
        return records

    def _parse_rxcui(
        self,
        rxcui: str,
        query: str,
        properties: list[dict[str, Any]],
        raw: RawDocumentMetadata,
    ) -> ParsedSourceRecord:
        names = []
        tty_values = []
        for item in properties:
            value = item.get("propValue")
            if item.get("propName") in {"RxNorm Name", "Synonym", "Display Name"} and value:
                names.append(value)
            if item.get("propName") == "TTY" and value:
                tty_values.append(value)
        record = {
            "query": query,
            "rxcui": rxcui,
            "names": sorted(set(names))[:25],
            "term_types": sorted(set(tty_values)),
            "property_count": len(properties),
            "properties": properties[:50],
            "extraction_level": "rxcui_identity_and_synonyms",
        }
        return self.make_record(
            raw=raw,
            source_record_id=rxcui,
            source_url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={quote(rxcui)}",
            source_location=f"RXCUI:{rxcui}",
            record_type="rxnorm_ingredient",
            record=record,
        )
